from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# WICHTIG: train_test_split statt GroupShuffleSplit, wie im Referenz-Notebook
from sklearn.model_selection import train_test_split 
from imblearn.under_sampling import OneSidedSelection


# =========================
# Konfiguration (Paper Section 5.2.1)
# =========================

@dataclass
class IDPSeparateFFNConfig:
    """
    Konfiguration für IDP-separateFFN gemäß Paper.
    
    Hyperparameter aus 'Hyperparameter Optimization' in 5.2.1:
      - Hidden layer structure: 256 x 256
      - alpha_dc: 16 (Gewicht für Deviating Class)
      - Dropout: 0.1
    """
    hidden_dim: int = 256
    alpha_cc: float = 1.0
    alpha_dc: float = 16.0
    dropout: float = 0.1
    
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 20  # "imposing early stopping"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42 # Für Reproduzierbarkeit


# =========================
# Daten-Layer
# =========================

class PrefixDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_ffn_data(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Lädt CIBE-Features (encoding_ffn.npy) und Labels (idp_labels.npz).
    """
    # 1. Features (CIBE)
    feat_path = processed_dir / "encoding_ffn.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} nicht gefunden. Bitte erst Encoding ausführen.")
    X = np.load(feat_path)

    # 2. Labels & Meta
    label_path = processed_dir / "idp_labels.npz"
    if not label_path.exists():
        raise FileNotFoundError(f"{label_path} nicht gefunden. Bitte erst Labeling ausführen.")
    data = np.load(label_path, allow_pickle=True)
    
    y = data["y"]              # (N, m)
    case_ids = data["case_ids"]
    dev_types = list(data["dev_types"])

    assert X.shape[0] == y.shape[0], "Feature- und Label-Anzahl stimmt nicht überein."
    return X, y, case_ids, dev_types


# =========================
# Modell-Architektur (Fig. 4)
# =========================

class IDPFFN(nn.Module):
    """
    Feed-Forward Netzwerk exakt wie in Figure 4.
    
    Struktur:
      Input Layer (Size EA*)
         |
      Linear (256) -> LayerNorm -> Leaky ReLU
         |
      Linear (256) -> LayerNorm -> Leaky ReLU -> Dropout
         |
      Output Layer (Size 2) -> (Softmax implizit im Loss)
    """
    def __init__(self, input_dim: int, cfg: IDPSeparateFFNConfig):
        super().__init__()
        
        # Block 1
        self.layer1 = nn.Linear(input_dim, cfg.hidden_dim)
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.act1 = nn.LeakyReLU()
        
        # Block 2
        self.layer2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.act2 = nn.LeakyReLU()
        self.drop = nn.Dropout(cfg.dropout)
        
        # Output
        self.head = nn.Linear(cfg.hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: Feed Forward -> LayerNorm -> Leaky ReLU
        x = self.act1(self.ln1(self.layer1(x)))
        
        # Block 2: Feed Forward -> LayerNorm -> Leaky ReLU -> Dropout
        x = self.drop(self.act2(self.ln2(self.layer2(x))))
        
        # Output
        logits = self.head(x)
        return logits


# =========================
# Training & Preprocessing (Section 5.2.1)
# =========================

def train_single_classifier(
    dev_idx: int,
    dev_name: str,
    X: np.ndarray,
    y_all: np.ndarray,
    case_ids: np.ndarray,
    cfg: IDPSeparateFFNConfig
) -> Tuple[Optional[IDPFFN], np.ndarray]:
    """
    Trainiert einen spezialisierten Classifier für einen Deviation-Typ d_i.
    """
    print(f"\n--- Training Classifier für d_{dev_idx}: {dev_name} ---")
    
    # 1. Labels für diesen Typ extrahieren
    y_dev = y_all[:, dev_idx].astype(np.int64)
    n_samples, n_features = X.shape
    
    # Paper Section 5.1: "deviation types, that occur in one trace only, have to be removed"
    if np.sum(y_dev) < 2:
        print(f"  [SKIP] Zu wenige Abweichungen ({np.sum(y_dev)}) für Training.")
        return None, np.zeros(n_samples, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 2. Split (2/3 Train, 1/3 Test) - Trace-basiert (MANUELL)
    # -------------------------------------------------------------------------
    # Referenz aus Notebook: 
    # x_train_idx, x_test_idx, ... = train_test_split(range(len(log)), ..., test_size=split, ...)
    
    # A) Einzigartige Cases finden (entspricht "log" im Notebook)
    unique_cases = np.unique(case_ids)
    
    # B) Die Cases splitten (nicht die Prefixe!)
    # Paper: "We split the traces ... randomly into train (2/3) and test (1/3)"
    train_cases, test_cases = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    
    # C) Masken erstellen, um Prefixe den Cases zuzuordnen
    # (Entspricht der Logik im Notebook, wo über Indizes iteriert wird)
    train_mask = np.isin(case_ids, train_cases)
    test_mask = np.isin(case_ids, test_cases)
    
    # D) Daten aufteilen
    X_train = X[train_mask]
    y_train = y_dev[train_mask]
    
    # (Test-Daten brauchen wir hier im Training nur implizit, 
    # im Referenzcode wird X_test erstellt für Evaluation)
    
    print(f"  Trace Split: {len(train_cases)} Traces Train, {len(test_cases)} Traces Test.")
    print(f"  Prefix Split: {len(y_train)} Prefixe Train (davon {np.sum(y_train)} Dev).")

    # Check: Train-Set muss Abweichungen enthalten
    if np.sum(y_train) == 0:
        print("  [SKIP] Keine Abweichungen im Trainings-Split.")
        return None, np.zeros(n_samples, dtype=np.float32)

    # 3. Undersampling (One-Sided Selection)
    # "In this design, we can undersample each deviation type individually..."
    # Nur auf Train-Set anwenden!
    print(f"  Undersampling (OSS) auf Train-Set...")
    try:
        oss = OneSidedSelection(random_state=0, n_seeds_S=250, n_neighbors=7)
        X_train_res, y_train_res = oss.fit_resample(X_train, y_train)
        print(f"  Resampled Train-Set: {len(X_train_res)} Samples (Dev: {sum(y_train_res)})")
    except Exception as e:
        print(f"  [WARN] OSS fehlgeschlagen ({e}), nutze originale Daten.")
        X_train_res, y_train_res = X_train, y_train

    # 4. Modell Initialisierung
    device = torch.device(cfg.device)
    model = IDPFFN(input_dim=n_features, cfg=cfg).to(device)
    
    # 5. Weighted Loss Function
    # "weight alpha_DC = 16 ... alpha_CC = 1"
    weights = torch.tensor([cfg.alpha_cc, cfg.alpha_dc], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # DataLoader
    train_ds = PrefixDataset(X_train_res, y_train_res)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    # 6. Training Loop
    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # 7. Prediction (Posterior Probabilities)
    # Wir berechnen P(y=1) für ALLE Daten (Train + Test)
    full_ds = PrefixDataset(X, y_dev)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size * 2, shuffle=False)
    
    model.eval()
    probs_list = []
    with torch.no_grad():
        for bx, _ in full_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs[:, 1].cpu().numpy()) # Klasse 1 (Dev)
            
    return model, np.concatenate(probs_list)


# =========================
# Main Execution
# =========================

def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    models_dir = processed_dir / "models_idp_separate_ffn"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Start IDP-separate-FFN Training (Manual Trace Split) ---")
    cfg = IDPSeparateFFNConfig()
    print(f"Device: {cfg.device}")
    
    # 1. Daten laden
    X, y_all, case_ids, dev_types = load_ffn_data(processed_dir)
    print(f"Daten geladen: {X.shape[0]} Prefixe, {len(dev_types)} Deviation Types")
    
    n_prefixes = X.shape[0]
    m_devs = len(dev_types)
    
    # Matrix für Wahrscheinlichkeiten P_D initialisieren
    P_dev_all = np.zeros((n_prefixes, m_devs), dtype=np.float32)
    
    # 2. Loop über alle Deviation Types (Separate Classifiers)
    for i, dev_name in enumerate(dev_types):
        model, probs = train_single_classifier(
            dev_idx=i,
            dev_name=dev_name,
            X=X,
            y_all=y_all,
            case_ids=case_ids,
            cfg=cfg
        )
        
        # Ergebnisse speichern (auch wenn Modell None ist -> dann Nullen)
        P_dev_all[:, i] = probs
        
        if model is not None:
            model_path = models_dir / f"ffn_d{i}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  Model gespeichert: {model_path}")

    # 3. Wahrscheinlichkeiten speichern (für Evaluierung & DPP)
    out_path = processed_dir / "idp_separate_ffn_probs.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,
        dev_types=np.array(dev_types),
        case_ids=case_ids
    )
    print(f"\nTraining abgeschlossen. Wahrscheinlichkeiten gespeichert in: {out_path}")


if __name__ == "__main__":
    main()