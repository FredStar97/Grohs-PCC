from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# WICHTIG: train_test_split statt GroupShuffleSplit, wie im Referenz-Notebook
from sklearn.model_selection import train_test_split 
from imblearn.under_sampling import OneSidedSelection

"""
- dataclass: Konfigurationsklasse
- torch, nn: PyTorch für das Netz
- train_test_split: Train/Test-Split
- OneSidedSelection: Undersampling
"""

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
    dropout: float = 0.1 # Beim Training werden zufällig 10 % der Neuronen/Verbindungen „ausgeschaltet“, damit das Netz nicht überlernt.
    batch_size: int = 256
    lr: float = 0.0001 # auch aus BPDP-Code übernommen
    max_epochs: int = 300  # Early Stopping -> Maximum, Training kann früher stoppen
    
    # Early Stopping Parameter
    patience: int = 10
    validation_split: float = 0.2
    
    # Undersampling
    use_oss: bool = True  # One-Sided Selection für Undersampling
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42 # Für Reproduzierbarkeit


# =========================
# Daten-Layer
# =========================
"""
-  PyTorch Dataset für Prefix-Daten
- __init__: Konvertiert NumPy-Arrays zu Tensoren (float32/int64)
- __len__: Anzahl der Samples
- __getitem__: Gibt ein Sample (Features, Label) zurück
    """
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
    X: Features (N Prefixe × Features)
    y: Labels (N × m, m = Anzahl Deviation-Typen)
    case_ids: Case-IDs pro Prefix
    dev_types: Liste der Deviation-Typen
    Prüft, dass Anzahl der Prefixe übereinstimmt (X.shape[0] == y.shape[0])
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
      Output Layer (Size 2) -> Logits (ohne Softmax)

      Die Softmax wird in der Loss-Funktion angewendet (numerisch stabiler).
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
        
        # Output Layer (gibt Logits zurück, keine Wahrscheinlichkeiten)
        self.head = nn.Linear(cfg.hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: Feed Forward -> LayerNorm -> Leaky ReLU
        x = self.act1(self.ln1(self.layer1(x)))
        
        # Block 2: Feed Forward -> LayerNorm -> Leaky ReLU -> Dropout
        x = self.drop(self.act2(self.ln2(self.layer2(x))))
        
        # Output Layer -> Logits (Softmax wird in Loss-Funktion angewendet)
        logits = self.head(x)
        
        return logits

# =========================
# Training & Preprocessing (Section 5.2.1)
# =========================

def weighted_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, 
                                alpha_cc: float, alpha_dc: float) -> torch.Tensor:
    """
    Gewichteter Cross-Entropy auf Logits (Softmax wird intern angewendet).
    alpha_cc: Gewicht für Klasse 0 (Non-Deviating)
    alpha_dc: Gewicht für Klasse 1 (Deviating)
    
    Die Softmax wird numerisch stabil mit log_softmax angewendet:
    -log(softmax(logits)[correct_class]) = -log_softmax(logits)[correct_class]
    """
    device = logits.device
    
    # Gewichte: alpha_cc für Klasse 0, alpha_dc für Klasse 1
    weights = torch.where(targets == 0, 
                         torch.tensor(alpha_cc, device=device),
                         torch.tensor(alpha_dc, device=device))
    
    # Cross-entropy: -log(softmax(logits)[correct_class])
    # Numerisch stabil: log_softmax statt log(softmax(...))
    log_probs = F.log_softmax(logits, dim=1)
    
    # Negative Log-Likelihood für die korrekte Klasse
    # Gather: extrahiert log_prob für die korrekte Klasse pro Sample
    loss_per_sample = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Gewichteter Loss
    weighted_loss = weights * loss_per_sample
    
    return weighted_loss.mean()


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
    
    # Check auf Trace-Level (nicht Prefix-Level) 
    # Wir filtern die Case-IDs, wo tatsächlich eine Abweichung vorliegt (y=1)
    deviating_cases = case_ids[y_dev == 1]
    n_unique_dev_traces = len(np.unique(deviating_cases))

    # Paper Section 5.1: "deviation types, that occur in one trace only, have to be removed" 
    if n_unique_dev_traces < 2:
        print(f"  [SKIP] Zu wenige deviierende Traces ({n_unique_dev_traces}) für Training.")
        return None, np.zeros(n_samples, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 2. Split (2/3 Train, 1/3 Test) - Trace-basiert 
    # -------------------------------------------------------------------------
    # A) Einzigartige Cases finden
    unique_cases = np.unique(case_ids)
    
    # B) Die Cases splitten
    train_cases, test_cases = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    
    # C) Masken erstellen
    train_mask = np.isin(case_ids, train_cases)
    
    # D) Daten aufteilen
    X_train = X[train_mask]
    y_train = y_dev[train_mask]
    case_ids_train = case_ids[train_mask]  # Für trace-basiertes Validation Split
    
    print(f"  Trace Split: {len(train_cases)} Traces Train, {len(test_cases)} Traces Test.")
    print(f"  Prefix Split: {len(y_train)} Prefixe Train (davon {np.sum(y_train)} Dev).")

    # Check: Train-Set muss Abweichungen enthalten
    if np.sum(y_train) == 0:
        print("  [SKIP] Keine Abweichungen im Trainings-Split.")
        return None, np.zeros(n_samples, dtype=np.float32)

    # 3. One-Sided Selection zur Reduzierung der Mehrheitsklasse
    if cfg.use_oss:
        print(f"  Undersampling (OSS) auf Train-Set...")
        try:
            oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
            X_train_res, y_train_res = oss.fit_resample(X_train, y_train)
            kept_indices = oss.sample_indices_
            case_ids_train = case_ids_train[kept_indices]  # Case IDs für resampled Daten
            print(f"  Resampled Train-Set: {len(X_train_res)} Samples (Dev: {sum(y_train_res)})")
        except Exception as e:
            print(f"  [WARN] OSS fehlgeschlagen ({e}), nutze originale Daten.")
            X_train_res, y_train_res = X_train, y_train
    else:
        print(f"  Kein Undersampling (OSS deaktiviert).")
        X_train_res, y_train_res = X_train, y_train
    
    # 4. Validation Split (trace-basiert, 80% Train, 20% Validation)
    unique_train_cases = np.unique(case_ids_train)
    train_cases_final, val_cases = train_test_split(
        unique_train_cases,
        test_size=cfg.validation_split,
        random_state=cfg.random_state
    )
    
    train_final_mask = np.isin(case_ids_train, train_cases_final)
    val_mask = np.isin(case_ids_train, val_cases)
    
    # Final Training Set
    X_train_final = X_train_res[train_final_mask]
    y_train_final = y_train_res[train_final_mask]
    
    # Validation Set
    X_val = X_train_res[val_mask]
    y_val = y_train_res[val_mask]
    
    print(f"  Validation Split: {len(train_cases_final)} Traces Train, {len(val_cases)} Traces Val")
    print(f"  Samples: {len(y_train_final)} Train, {len(y_val)} Val")

    # 5. Modell Initialisierung: erstellt IDP-separate-FFN-Modell auf gewählten device
    device = torch.device(cfg.device)
    model = IDPFFN(input_dim=n_features, cfg=cfg).to(device)
    
    # 6. Weighted Loss Function 
    # Adam-Optimizer: Optimizer, der die Modell-Gewichte beim Training automatisch so anpasst, dass der Loss kleiner wird
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 7. DataLoaders
    train_ds = PrefixDataset(X_train_final, y_train_final)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    val_ds = PrefixDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # 8. Training Loop mit Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        # Training Phase
        epoch_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (gibt Logits aus)
            logits = model(bx)
            
            # Gewichteter Loss auf Logits (Softmax wird intern angewendet)
            loss = weighted_cross_entropy_loss(logits, by, cfg.alpha_cc, cfg.alpha_dc)
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation Phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                
                logits = model(bx)
                loss = weighted_cross_entropy_loss(logits, by, cfg.alpha_cc, cfg.alpha_dc)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Logging
        print(f"  Epoch {epoch}/{cfg.max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"    -> Bestes Modell gespeichert (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early Stopping nach {epoch} Epochen (Patience: {cfg.patience})")
                break
        
        model.train()
    
    # Training Zusammenfassung
    actual_epochs = epoch
    if actual_epochs < cfg.max_epochs:
        print(f"  Training beendet nach {actual_epochs} Epochen (Early Stopping)")
    else:
        print(f"  Training abgeschlossen nach {actual_epochs} Epochen (Max Epochs erreicht)")
    
    # 9. Bestes Modell wiederherstellen
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Bestes Modell wiederhergestellt (Val Loss: {best_val_loss:.4f})")
    else:
        print(f"  [WARN] Kein bestes Modell gefunden, nutze aktuelles Modell")

    # 10. Prediction (Posterior Probabilities)
    """
    Evaluation auf allen Daten
    Modell gibt Logits zurück, Softmax wird hier angewendet
    Extrahiert Wahrscheinlichkeit für Klasse 1 (Deviation)
    Gibt Modell und Wahrscheinlichkeiten zurück

    """
    full_ds = PrefixDataset(X, y_dev)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size * 2, shuffle=False)
    
    model.eval()
    probs_list = []
    with torch.no_grad():
        for bx, _ in full_loader:
            bx = bx.to(device)
            
            # Modell gibt Logits aus, Softmax wird hier angewendet
            logits = model(bx)
            probs = F.softmax(logits, dim=1)
            
            probs_list.append(probs[:, 1].cpu().numpy()) # Klasse 1 (Dev)
            
    return model, np.concatenate(probs_list)


# =========================
# Main Execution
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Trainiert separate FFN-Modelle für jeden Deviation Type"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name des Datensatzes (für Input/Output-Verzeichnis). Standard: processed/ (rückwärtskompatibel)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input-Verzeichnis für Encodings und Labels. Standard: data/processed/ oder data/processed/{dataset}/"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Verzeichnis für gespeicherte Modelle. Standard: {input-dir}/models_idp_separate_ffn/"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    
    # Verzeichnisse bestimmen (rückwärtskompatibel)
    if args.input_dir is not None:
        processed_dir = Path(args.input_dir)
    elif args.dataset is not None:
        processed_dir = project_root / "data" / "processed" / args.dataset
    else:
        # Rückwärtskompatibel: Standard-Verzeichnis
        processed_dir = project_root / "data" / "processed"
    
    if args.models_dir is not None:
        models_dir = Path(args.models_dir)
    else:
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
    suffix = "_no_oss" if not cfg.use_oss else ""
    out_path = processed_dir / f"idp_separate_ffn_probs{suffix}.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,
        dev_types=np.array(dev_types),
        case_ids=case_ids
    )
    print(f"\nTraining abgeschlossen. Wahrscheinlichkeiten gespeichert in: {out_path}")


if __name__ == "__main__":
    main()