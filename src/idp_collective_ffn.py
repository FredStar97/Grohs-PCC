from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import OneSidedSelection


# =========================
# Konfiguration (Section 5.2.3)
# =========================

@dataclass
class IDPCollectiveFFNConfig:
    """
    Konfiguration für IDP-collectiveFFN gemäß Paper.
    
    Werte aus 'Hyperparameter Optimization' in 5.2.3:
      - Hidden layer structure: 2048 x 1024
      - Dropout: 0.1
      - Loss Weight Formula: 16^(1/(2e) + log(LIR))
    """
    hidden_dim_1: int = 2048
    hidden_dim_2: int = 1024
    dropout: float = 0.1
    
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 20
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42


# =========================
# Daten-Layer
# =========================

class CollectiveDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)) # Float für BCEWithLogitsLoss

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_ffn_data(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Lädt CIBE-Features und Labels.
    """
    feat_path = processed_dir / "encoding_ffn.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} nicht gefunden.")
    X = np.load(feat_path)

    label_path = processed_dir / "idp_labels.npz"
    if not label_path.exists():
        raise FileNotFoundError(f"{label_path} nicht gefunden.")
    data = np.load(label_path, allow_pickle=True)
    
    y = data["y"]
    case_ids = data["case_ids"]
    dev_types = list(data["dev_types"])

    return X, y, case_ids, dev_types


# =========================
# Modell-Architektur (Fig. 6)
# =========================

class IDPCollectiveFFN(nn.Module):
    """
    Architektur gemäß Figure 6.
    """
    def __init__(self, input_dim: int, num_classes: int, cfg: IDPCollectiveFFNConfig):
        super().__init__()
        
        # Layer 1
        self.layer1 = nn.Linear(input_dim, cfg.hidden_dim_1)
        self.ln1 = nn.LayerNorm(cfg.hidden_dim_1)
        self.act1 = nn.LeakyReLU()
        
        # Layer 2
        self.layer2 = nn.Linear(cfg.hidden_dim_1, cfg.hidden_dim_2)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim_2)
        self.act2 = nn.LeakyReLU()
        self.drop = nn.Dropout(cfg.dropout)
        
        # Output Layer
        self.head = nn.Linear(cfg.hidden_dim_2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.ln1(self.layer1(x)))
        x = self.drop(self.act2(self.ln2(self.layer2(x))))
        logits = self.head(x)
        return logits


# =========================
# Helper: Loss Weight Calculation
# =========================

def calculate_collective_weights(y_train: np.ndarray, device: str) -> torch.Tensor:
    """
    Berechnet die Gewichte Beta_d gemäß Abschnitt 5.2.3.
    Formel: Beta = 16 ^ (1 / (2e + log(LIR)))
    """
    n_samples, n_classes = y_train.shape
    weights = []
    
    # Eulersche Zahl
    e = np.e 
    
    for i in range(n_classes):
        dc = np.sum(y_train[:, i] == 1) # Deviating Traces count
        cc = np.sum(y_train[:, i] == 0) # Conforming Traces count (für diesen Typ)
        
        # Schutz vor Division durch Null, falls Filterung versagt hat
        if dc == 0:
            weights.append(1.0)
            continue
            
        lir = cc / dc
        
        # --- KORREKTUR START ---
        # Paper: 1 geteilt durch (2e + log(LIR))
        denominator = (2.0 * e) + np.log(lir)
        exponent = 1.0 / denominator
        # --- KORREKTUR ENDE ---
        
        beta = 16.0 ** exponent
        weights.append(beta)
        
    return torch.tensor(weights, dtype=torch.float32).to(device)


# =========================
# Training & Preprocessing
# =========================

def train_collective_model(
    X: np.ndarray,
    y_all: np.ndarray,
    case_ids: np.ndarray,
    cfg: IDPCollectiveFFNConfig
) -> Tuple[IDPCollectiveFFN, np.ndarray]:
    
    print("\n--- Training IDP-collective-FFN ---")
    
    # 1. Manueller Trace Split
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    
    train_mask = np.isin(case_ids, train_cases)
    
    X_train = X[train_mask]
    y_train = y_all[train_mask]
    
    print(f"  Split: {len(train_cases)} Traces im Training.")
    print(f"  Samples Train: {X_train.shape[0]}")

    # 2. Collective Undersampling (OSS)
    print("  Collective Undersampling (OSS)...")
    
    # Hilfs-Label: Hat Trace irgendeine Abweichung?
    y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
    
    try:
        oss = OneSidedSelection(random_state=0, n_seeds_S=250, n_neighbors=7)
        
        _X_dummy, _y_dummy = oss.fit_resample(X_train, y_is_deviant)
        kept_indices = oss.sample_indices_
        
        X_train_res = X_train[kept_indices]
        y_train_res = y_train[kept_indices]
        
        print(f"  Resampled Train-Set: {len(X_train_res)} Samples.")
    except Exception as e:
        print(f"  [WARN] OSS fehlgeschlagen ({e}). Nutze volle Daten.")
        X_train_res, y_train_res = X_train, y_train

    # 3. Weights auf Resampled Data
    pos_weights = calculate_collective_weights(y_train_res, cfg.device)
    
    # 4. Model Setup
    device = torch.device(cfg.device)
    model = IDPCollectiveFFN(
        input_dim=X.shape[1], 
        num_classes=y_all.shape[1], 
        cfg=cfg
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    train_ds = CollectiveDataset(X_train_res, y_train_res)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    # 5. Training Loop
    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    # 6. Prediction
    full_ds = CollectiveDataset(X, y_all)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size * 2, shuffle=False)
    
    model.eval()
    probs_list = []
    with torch.no_grad():
        for bx, _ in full_loader:
            bx = bx.to(device)
            logits = model(bx)
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
    models_dir = processed_dir / "models_idp_collective_ffn"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Start IDP-collective-FFN Pipeline ---")
    cfg = IDPCollectiveFFNConfig()
    print(f"Device: {cfg.device}")
    
    # 1. Daten laden
    X, y_all, case_ids, dev_types = load_ffn_data(processed_dir)
    print(f"Raw Deviation Types: {len(dev_types)}")
    
    # -----------------------------------------------------------
    # FILTERUNG (Paper Requirement)
    # "remove deviation types, that occur in one trace only"
    # -----------------------------------------------------------
    keep_indices = []
    dropped_names = []
    
    for i in range(len(dev_types)):
        # Betrachte Spalte i
        dev_col = y_all[:, i]
        # Hole Unique Cases, die diese Abweichung haben (mindestens einmal im Trace)
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        count = len(cases_with_dev)
        
        if count > 1:
            keep_indices.append(i)
        else:
            dropped_names.append(dev_types[i])
            
    if dropped_names:
        print(f"Entferne Deviation Types (<= 1 Trace): {dropped_names}")
    
    if not keep_indices:
        print("Keine Deviation Types übrig. Abbruch.")
        return
        
    # Filtern
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    print(f"Training mit {len(dev_types_filtered)} Deviation Types.")
    print("-" * 40)
    
    # 2. Training
    model, P_dev_all = train_collective_model(X, y_filtered, case_ids, cfg)
    
    # 3. Speichern
    model_path = models_dir / "collective_ffn.pt"
    torch.save(model.state_dict(), model_path)
    
    out_path = processed_dir / "idp_collective_ffn_probs.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), # WICHTIG: Gefilterte Namen speichern
        case_ids=case_ids
    )
    print(f"Fertig. Ergebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()