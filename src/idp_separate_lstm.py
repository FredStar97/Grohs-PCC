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
# Konfiguration (Section 5.2.2)
# =========================

@dataclass
class IDPSeparateLSTMConfig:
    """
    Konfiguration für IDP-separateLSTM.
    
    Werte gemäß "Hyperparameter Optimization":
      - Embedding dimension: 16
      - LSTM layer size: 64
      - alpha_dc: 16.0
      - Dropout: 0.1
    """
    embedding_dim: int = 16
    lstm_hidden_dim: int = 64
    projection_dim: int = 64
    
    alpha_cc: float = 1.0
    alpha_dc: float = 16.0
    dropout: float = 0.1
    
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 20
    
    device: str = "cpu" 
    random_state: int = 42


# =========================
# Daten-Layer
# =========================

class LSTMDataset(Dataset):
    def __init__(
        self, 
        X_act: np.ndarray, 
        X_res: np.ndarray, 
        X_month: np.ndarray, 
        X_trace: np.ndarray, 
        y: np.ndarray
    ):
        self.X_act = torch.from_numpy(X_act.astype(np.int64))
        self.X_res = torch.from_numpy(X_res.astype(np.int64))
        self.X_month = torch.from_numpy(X_month.astype(np.int64))
        self.X_trace = torch.from_numpy(X_trace.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
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
    
    lab_path = processed_dir / "idp_labels.npz"
    if not lab_path.exists():
        raise FileNotFoundError(f"{lab_path} nicht gefunden.")
    lab_data = np.load(lab_path, allow_pickle=True)
    y_all = lab_data["y"]
    case_ids = lab_data["case_ids"]
    dev_types = list(lab_data["dev_types"])
    
    meta_path = processed_dir / "encoding_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    return inputs, y_all, case_ids, dev_types, meta


# =========================
# Modell Architektur (Fig. 5)
# =========================

class IDPLSTM(nn.Module):
    """
    Architektur gemäß Fig. 5.
    """
    def __init__(
        self, 
        meta: Dict, 
        trace_input_dim: int, 
        cfg: IDPSeparateLSTMConfig
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
        
        # --- Concat & Output Head ---
        concat_dim = cfg.projection_dim * 4 
        
        self.ln = nn.LayerNorm(concat_dim)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(concat_dim, 2)
        
    def forward(self, x_act, x_res, x_month, x_trace):
        # Branch 1
        e_act = self.emb_act(x_act)
        _, (h_act, _) = self.lstm_act(e_act)
        feat_act = self.ff_act(h_act[-1])
        
        # Branch 2
        e_res = self.emb_res(x_res)
        _, (h_res, _) = self.lstm_res(e_res)
        feat_res = self.ff_res(h_res[-1])
        
        # Branch 3
        e_month = self.emb_month(x_month)
        _, (h_month, _) = self.lstm_month(e_month)
        feat_month = self.ff_month(h_month[-1])
        
        # Branch 4
        feat_trace = self.ff_trace(x_trace)
        
        # Concat & Head
        concat = torch.cat([feat_act, feat_res, feat_month, feat_trace], dim=1)
        x = self.ln(concat)
        x = self.act(x)
        x = self.drop(x)
        logits = self.head(x)
        
        return logits


# =========================
# Training & Undersampling Logic
# =========================

def train_single_lstm_model(
    dev_idx: int,
    dev_name: str,
    inputs: Dict[str, np.ndarray],
    y_all: np.ndarray, # Hier kommen bereits die gefilterten Labels rein
    case_ids: np.ndarray,
    meta: Dict,
    cfg: IDPSeparateLSTMConfig
) -> Tuple[IDPLSTM | None, np.ndarray]:
    
    print(f"\n=== Training IDP-separateLSTM für Deviation-Typ {dev_idx} ({dev_name}) ===")
    
    # 1. Labels für diesen Typ
    y_dev = y_all[:, dev_idx].astype(np.int64)
    n_samples = y_dev.shape[0]
    
    # -------------------------------------------------------------------------
    # MANUELLER TRACE SPLIT
    # "We split the traces in the log randomly into train (2/3) and test (1/3)"
    # -------------------------------------------------------------------------
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    
    train_mask = np.isin(case_ids, train_cases)
    
    X_act_train = inputs["X_act"][train_mask]
    X_res_train = inputs["X_res"][train_mask]
    X_month_train = inputs["X_month"][train_mask]
    X_trace_train = inputs["X_trace"][train_mask]
    y_train = y_dev[train_mask]
    
    print(f"  Trace Split: {len(train_cases)} Traces Train. (Total Samples: {len(y_train)})")
    
    # Safety Check: Auch wenn wir global gefiltert haben, kann es durch den Split passieren,
    # dass im Training zufällig 0 Abweichungen landen (wenn es insgesamt sehr wenige sind).
    # Das Paper sagt: "ensure that both the training and test set contain at least one deviating trace" [cite: 417]
    # Das ist eine Bedingung für die Auswertbarkeit. Wenn das nicht erfüllt ist, können wir nicht trainieren.
    if y_train.sum() == 0:
        print(f"  [WARN] Keine Abweichungen im Train-Set für {dev_name} (obwohl global >1). Skip.")
        return None, np.zeros(n_samples, dtype=np.float32)

    # 2. Undersampling (OSS)
    print(f"  Undersampling (OSS) auf Train-Set...")
    oss = OneSidedSelection(random_state=0, n_seeds_S=250, n_neighbors=7)
    
    try:
        _X_dummy, _y_dummy = oss.fit_resample(X_trace_train, y_train)
        kept_indices = oss.sample_indices_
        
        X_act_train = X_act_train[kept_indices]
        X_res_train = X_res_train[kept_indices]
        X_month_train = X_month_train[kept_indices]
        X_trace_train = X_trace_train[kept_indices]
        y_train = y_train[kept_indices]
    except Exception as e:
        print(f"  [INFO] OSS fehlgeschlagen ({e}). Nutze volle Daten.")
        pass

    print(f"  Train-Set nach Undersampling: {len(y_train)} Samples.")

    # 3. Model Setup
    device = torch.device(cfg.device)
    model = IDPLSTM(
        meta=meta, 
        trace_input_dim=inputs["X_trace"].shape[1], 
        cfg=cfg
    ).to(device)
    
    # 4. WCEL
    weights = torch.tensor([cfg.alpha_cc, cfg.alpha_dc], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # DataLoader
    train_ds = LSTMDataset(X_act_train, X_res_train, X_month_train, X_trace_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    # 5. Training Loop
    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        for (batch_act, batch_res, batch_month, batch_trace), batch_y in train_loader:
            batch_act, batch_res = batch_act.to(device), batch_res.to(device)
            batch_month, batch_trace = batch_month.to(device), batch_trace.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_act, batch_res, batch_month, batch_trace)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
    # 6. Prediction
    full_ds = LSTMDataset(inputs["X_act"], inputs["X_res"], inputs["X_month"], inputs["X_trace"], y_dev)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=False)
    
    model.eval()
    probs_list = []
    with torch.no_grad():
        for (b_act, b_res, b_month, b_trace), _ in full_loader:
            b_act, b_res = b_act.to(device), b_res.to(device)
            b_month, b_trace = b_month.to(device), b_trace.to(device)
            
            logits = model(b_act, b_res, b_month, b_trace)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs[:, 1].cpu().numpy())
            
    return model, np.concatenate(probs_list)


# =========================
# Main Execution
# =========================

def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    models_dir = processed_dir / "models_idp_separate_lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPSeparateLSTMConfig()
    print(f"Device set to: {cfg.device}")
    
    print("Lade LSTM-Daten...")
    inputs, y_all, case_ids, dev_types, meta = load_lstm_data(processed_dir)
    
    print(f"Ursprüngliche Deviation Types: {len(dev_types)}")
    
    # -----------------------------------------------------------
    # FILTERUNG gemäß Paper: "remove deviation types, that occur in one trace only" 
    # -----------------------------------------------------------
    # Hinweis: Da y_all auf Prefix-Ebene ist und "occur in one trace" gemeint ist, 
    # müssen wir zählen, in wie vielen UNIQUE traces ein Typ vorkommt.
    # Wenn y_all[i, dev] = 1, dann ist in diesem Prefix eine Abweichung.
    # Da ein Trace mehrere Prefixe mit "1" haben kann, zählen wir unique Case-IDs pro Deviation.
    
    keep_indices = []
    for i in range(len(dev_types)):
        # Holen wir uns alle Case IDs, bei denen dieser Dev-Type >= 1 ist
        # Wir betrachten die Spalte i
        dev_col = y_all[:, i]
        # Filtern auf die Zeilen, wo dev=1 ist
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        
        count_traces = len(cases_with_dev)
        
        if count_traces > 1:
            keep_indices.append(i)
        else:
            print(f"  -> Entferne '{dev_types[i]}' (Traces: {count_traces})")
            
    if not keep_indices:
        print("Keine Deviation Types übrig nach Filterung! Abbruch.")
        return

    # Arrays filtern
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    print(f"Verbleibende Deviation Types nach Filterung: {len(dev_types_filtered)}")
    print("-" * 40)

    n_prefixes, m_dev = y_filtered.shape
    P_dev_all = np.zeros((n_prefixes, m_dev), dtype=np.float32)
    
    for i, dev_name in enumerate(dev_types_filtered):
        model, probs = train_single_lstm_model(
            dev_idx=i,
            dev_name=dev_name,
            inputs=inputs,
            y_all=y_filtered, # Gefilterte Labels
            case_ids=case_ids,
            meta=meta,
            cfg=cfg
        )
        P_dev_all[:, i] = probs
        
        if model:
            torch.save(model.state_dict(), models_dir / f"lstm_d{i}.pt")

    out_path = processed_dir / "idp_separate_lstm_probs.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), # Speichere nur die gefilterten Namen
        case_ids=case_ids
    )
    print(f"Fertig. Ergebnisse gespeichert in {out_path}")

if __name__ == "__main__":
    main()