from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import OneSidedSelection


# =========================
# Konfiguration (Section 5.2.4)
# =========================

@dataclass
class IDPCollectiveLSTMConfig:
    """
    Konfiguration für IDP-collectiveLSTM.
    
    Werte gemäß "Hyperparameter Optimization" in 5.2.4:
      - Embedding dimension: 16
      - LSTM layer size: 128 (Final choice bold in paper text)
      - Dropout: 0.1
      - Loss Weight: 16^(1/2e + log(LIR))
    """
    embedding_dim: int = 16
    lstm_hidden_dim: int = 128
    projection_dim: int = 32  # Dimension um Branches anzugleichen (Ref Code)
    dropout: float = 0.1
    
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 20
    
    # FIX: CPU erzwingen, um cuDNN-Konflikt auf dem Server zu vermeiden
    device: str = "cpu" 
    random_state: int = 42


# =========================
# Daten-Layer
# =========================

class CollectiveLSTMDataset(Dataset):
    def __init__(
        self, 
        X_act: np.ndarray, 
        X_res: np.ndarray, 
        X_month: np.ndarray, 
        X_trace: np.ndarray, 
        y: np.ndarray
    ):
        """
        Dataset liefert die 4 Input-Komponenten und den Multi-Label Vektor y.
        """
        self.X_act = torch.from_numpy(X_act.astype(np.int64))
        self.X_res = torch.from_numpy(X_res.astype(np.int64))
        self.X_month = torch.from_numpy(X_month.astype(np.int64))
        self.X_trace = torch.from_numpy(X_trace.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)) # Float für BCE Loss

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        # Gibt (Inputs-Tupel, Label) zurück
        return (
            self.X_act[idx], 
            self.X_res[idx], 
            self.X_month[idx], 
            self.X_trace[idx]
        ), self.y[idx]


def load_lstm_data(processed_dir: Path):
    """
    Lädt Inputs (encoding_lstm.npz) und Labels (idp_labels.npz).
    """
    # 1. Inputs
    enc_path = processed_dir / "encoding_lstm.npz"
    if not enc_path.exists():
        raise FileNotFoundError(f"{enc_path} nicht gefunden.")
    enc_data = np.load(enc_path)
    inputs = {
        "X_act": enc_data["X_act_seq"],
        "X_res": enc_data["X_res_seq"],
        "X_month": enc_data["X_month_seq"],
        "X_trace": enc_data["X_trace"]
    }
    
    # 2. Labels
    lab_path = processed_dir / "idp_labels.npz"
    if not lab_path.exists():
        raise FileNotFoundError(f"{lab_path} nicht gefunden.")
    lab_data = np.load(lab_path, allow_pickle=True)
    y_all = lab_data["y"]
    case_ids = lab_data["case_ids"]
    dev_types = list(lab_data["dev_types"])
    
    # 3. Meta (für Vocab Sizes)
    meta_path = processed_dir / "encoding_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    return inputs, y_all, case_ids, dev_types, meta


# =========================
# Modell Architektur (Fig. 7)
# =========================

class IDPCollectiveLSTM(nn.Module):
    """
    Architektur gemäß Fig. 7 und BPDP Reference Code.
    
    Single model predicting ALL deviation types collectively.
    Structure:
      3x LSTM Branches (Activity, Resource, Month)
      1x Trace Branch
      Concat -> Output Layer (Size m) -> Sigmoid (implizit in Loss)
    """
    def __init__(
        self, 
        meta: Dict, 
        trace_input_dim: int, 
        num_classes: int,
        cfg: IDPCollectiveLSTMConfig
    ):
        super().__init__()
        
        # --- Branch 1: Activities ---
        self.emb_act = nn.Embedding(len(meta["activity_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_act = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_act = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 2: Resources ---
        self.emb_res = nn.Embedding(len(meta["resource_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_res = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_res = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 3: Month ---
        self.emb_month = nn.Embedding(len(meta["month_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_month = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_month = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 4: Trace Attributes ---
        self.ff_trace = nn.Linear(trace_input_dim, cfg.projection_dim)
        
        # --- Concat & Output ---
        # 4 Branches zusammengeführt
        concat_dim = cfg.projection_dim * 4 
        
        self.ln = nn.LayerNorm(concat_dim)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(cfg.dropout)
        
        # Output für m Klassen (Multi-Label)
        self.head = nn.Linear(concat_dim, num_classes)
        
    def forward(self, x_act, x_res, x_month, x_trace):
        # 1. Activities
        e_act = self.emb_act(x_act)
        _, (h_act, _) = self.lstm_act(e_act) # h_act: (1, B, Hidden)
        feat_act = self.ff_act(h_act[-1])    # (B, Proj)
        
        # 2. Resources
        e_res = self.emb_res(x_res)
        _, (h_res, _) = self.lstm_res(e_res)
        feat_res = self.ff_res(h_res[-1])
        
        # 3. Month
        e_month = self.emb_month(x_month)
        _, (h_month, _) = self.lstm_month(e_month)
        feat_month = self.ff_month(h_month[-1])
        
        # 4. Trace Attributes
        feat_trace = self.ff_trace(x_trace)
        
        # Merge
        concat = torch.cat([feat_act, feat_res, feat_month, feat_trace], dim=1)
        
        x = self.ln(concat)
        x = self.act(x)
        x = self.drop(x)
        logits = self.head(x)
        
        return logits


# =========================
# Helper: Loss Weights
# =========================

def calculate_collective_weights(y_train: np.ndarray, device: str) -> torch.Tensor:
    """
    Berechnet Gewichte Beta_d gemäß 5.2.4.
    Formula: beta_d = 16 ^ ( 1/(2*e) + log(LIR_d) )
    """
    n_samples, n_classes = y_train.shape
    weights = []
    
    for i in range(n_classes):
        dc = np.sum(y_train[:, i] == 1)
        cc = np.sum(y_train[:, i] == 0)
        
        if dc == 0:
            weights.append(1.0)
            continue
            
        lir = cc / dc
        exponent = (1.0 / (2.0 * np.e)) + np.log(lir)
        beta = 16.0 ** exponent
        weights.append(beta)
        
    return torch.tensor(weights, dtype=torch.float32).to(device)


# =========================
# Training & Preprocessing
# =========================

def train_collective_lstm(
    inputs: Dict[str, np.ndarray],
    y_all: np.ndarray,
    case_ids: np.ndarray,
    meta: Dict,
    cfg: IDPCollectiveLSTMConfig
) -> Tuple[IDPCollectiveLSTM, np.ndarray]:
    
    print("\n--- Training IDP-collective-LSTM ---")
    
    # 1. Trace Split (Manual, 2/3 Train)
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    
    train_mask = np.isin(case_ids, train_cases)
    
    # Filter Inputs
    X_act_train = inputs["X_act"][train_mask]
    X_res_train = inputs["X_res"][train_mask]
    X_month_train = inputs["X_month"][train_mask]
    X_trace_train = inputs["X_trace"][train_mask]
    y_train = y_all[train_mask]
    
    print(f"  Split: {len(train_cases)} Traces im Training (Samples: {len(y_train)}).")

    # 2. Collective Undersampling (OSS)
    print("  Collective Undersampling (OSS)...")
    
    # Hilfs-Label für OSS (Any deviation?)
    y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
    
    try:
        # Nutzung von Random State aus Config für Konsistenz
        # Parameter angepasst an Reference Code (n_neighbors=7, n_seeds=250)
        oss = OneSidedSelection(n_neighbors=7, n_seeds_S=250, random_state=cfg.random_state)
        
        # Fit auf Trace-Attributen als Proxy für Ähnlichkeit (da Sequenzen für OSS schwer sind)
        # Dies ist eine notwendige Anpassung für LSTM-Daten, analog zur Separate-LSTM Logik.
        _X_dummy, _y_dummy = oss.fit_resample(X_trace_train, y_is_deviant)
        kept_indices = oss.sample_indices_
        
        # Daten filtern
        X_act_train = X_act_train[kept_indices]
        X_res_train = X_res_train[kept_indices]
        X_month_train = X_month_train[kept_indices]
        X_trace_train = X_trace_train[kept_indices]
        y_train = y_train[kept_indices]
        
        print(f"  Resampled Train-Set: {len(y_train)} Samples.")
    except Exception as e:
        print(f"  [WARN] OSS fehlgeschlagen ({e}). Nutze volle Daten.")

    # 3. Weights calculation (on resampled data)
    pos_weights = calculate_collective_weights(y_train, cfg.device)
    
    # 4. Model Setup
    device = torch.device(cfg.device)
    model = IDPCollectiveLSTM(
        meta=meta,
        trace_input_dim=inputs["X_trace"].shape[1],
        num_classes=y_all.shape[1],
        cfg=cfg
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # DataLoader
    train_ds = CollectiveLSTMDataset(X_act_train, X_res_train, X_month_train, X_trace_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    # 5. Training Loop
    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        epoch_loss = 0.0
        for (b_act, b_res, b_month, b_trace), b_y in train_loader:
            b_act, b_res = b_act.to(device), b_res.to(device)
            b_month, b_trace = b_month.to(device), b_trace.to(device)
            b_y = b_y.to(device)
            
            optimizer.zero_grad()
            logits = model(b_act, b_res, b_month, b_trace)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    # 6. Prediction für ALLE Daten (um P_D zu erhalten)
    full_ds = CollectiveLSTMDataset(
        inputs["X_act"], inputs["X_res"], inputs["X_month"], inputs["X_trace"], y_all
    )
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=False)
    
    model.eval()
    probs_list = []
    with torch.no_grad():
        for (b_act, b_res, b_month, b_trace), _ in full_loader:
            b_act, b_res = b_act.to(device), b_res.to(device)
            b_month, b_trace = b_month.to(device), b_trace.to(device)
            
            logits = model(b_act, b_res, b_month, b_trace)
            # Sigmoid für Multi-Label Wahrscheinlichkeiten
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())
            
    P_D = np.concatenate(probs_list, axis=0)
    return model, P_D


# =========================
# Main Execution
# =========================

def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    models_dir = processed_dir / "models_idp_collective_lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Start IDP-collective-LSTM Training ---")
    cfg = IDPCollectiveLSTMConfig()
    print(f"Device set to: {cfg.device}")
    
    # 1. Daten laden
    inputs, y_all, case_ids, dev_types, meta = load_lstm_data(processed_dir)
    print(f"Deviation Types Raw: {len(dev_types)}")
    
    # -----------------------------------------------------------
    # FILTERUNG (Paper Requirement)
    # "remove deviation types, that occur in one trace only"
    # -----------------------------------------------------------
    keep_indices = []
    
    for i in range(len(dev_types)):
        dev_col = y_all[:, i]
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        if len(cases_with_dev) > 1:
            keep_indices.append(i)
    
    if not keep_indices:
        print("Keine trainierbaren Abweichungen. Abbruch.")
        return
        
    # Filtern der Labels
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    print(f"Training mit {len(dev_types_filtered)} Deviation Types.")
    print("-" * 40)
    
    # 2. Training
    model, P_dev_all = train_collective_lstm(
        inputs, y_filtered, case_ids, meta, cfg
    )
    
    # 3. Speichern
    model_path = models_dir / "collective_lstm.pt"
    torch.save(model.state_dict(), model_path)
    
    out_path = processed_dir / "idp_collective_lstm_probs.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered),
        case_ids=case_ids
    )
    print(f"Fertig. Ergebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()