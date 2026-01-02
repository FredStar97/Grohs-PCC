from __future__ import annotations

import argparse
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
    max_epochs: int = 300
    
    # Early Stopping Parameter
    patience: int = 10
    validation_split: float = 0.2
<<<<<<< Updated upstream
    
    # Undersampling
    use_oss: bool = True  # One-Sided Selection für Undersampling
=======
>>>>>>> Stashed changes
    
    device: str = "cpu" 
    random_state: int = 42


# =========================
# Daten-Layer
# =========================

"""
    Konvertiert NumPy-Arrays in PyTorch-Tensoren
__len__: Anzahl der Samples
__getitem__: Gibt ein Sample zurück (4 Inputs + Label)
    """

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

# =========================
# Modell Architektur (Fig. 5 - Korrigiert)
# =========================

class IDPLSTM(nn.Module):
    """
    Architektur gemäß Fig. 5 (1:1 Umsetzung).
    
    Struktur:
    - Activities: Embedding -> LSTM -> Feed Forward -> Concat
    - Resources: Embedding -> LSTM -> Feed Forward -> Concat
    - Month: Embedding -> LSTM -> Feed Forward -> Concat
    - Trace Attributes: Feed Forward -> Concat
    - Concat -> LayerNorm -> LeakyReLU -> Dropout -> Output Layer (size=2)
    """
    def __init__(
        self, 
        meta: Dict, 
        trace_input_dim: int, 
        cfg: IDPSeparateLSTMConfig
    ):
        super().__init__()
        
        # --- Branch 1: Activities ---
        # Input -> Embedding -> LSTM -> Feed Forward -> Concat
        self.emb_act = nn.Embedding(len(meta["activity_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_act = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_act = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 2: Resources ---
        # Input -> Embedding -> LSTM -> Feed Forward -> Concat
        self.emb_res = nn.Embedding(len(meta["resource_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_res = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_res = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 3: Month ---
        # Input -> Embedding -> LSTM -> Feed Forward -> Concat
        self.emb_month = nn.Embedding(len(meta["month_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_month = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_month = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 4: Trace Attributes ---
        # Input -> Feed Forward -> Concat
        self.ff_trace = nn.Linear(trace_input_dim, cfg.projection_dim)
        
        # --- Concat & Output Head ---
        # Berechnung der Dimensionen für die Zusammenführung:
        # 4x Projection Output (je 64): Activities, Resources, Month, Trace Attributes
        concat_dim = cfg.projection_dim * 4 
        
        self.ln = nn.LayerNorm(concat_dim)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(concat_dim, 2)

    """
    Forward Pass für das Modell
    x_act: Activities
    x_res: Resources
    x_month: Month
    x_trace: Trace Attributes
    """
    def forward(self, x_act, x_res, x_month, x_trace):
        # Branch 1: Activities
        e_act = self.emb_act(x_act)
        _, (h_act, _) = self.lstm_act(e_act) 
        # h_act hat Shape (1, Batch, Hidden). Wir nehmen h_act[-1].
        feat_act = self.ff_act(h_act[-1])
        
        # Branch 2: Resources
        e_res = self.emb_res(x_res)
        _, (h_res, _) = self.lstm_res(e_res)
        feat_res = self.ff_res(h_res[-1])
        
        # Branch 3: Month
        e_month = self.emb_month(x_month)
        _, (h_month, _) = self.lstm_month(e_month)
        feat_month = self.ff_month(h_month[-1])
        
        # Branch 4: Trace Attributes
        # Hier wird der Feed Forward Layer angewendet
        feat_trace = self.ff_trace(x_trace)
        
        # Concat
        # Wir verbinden die projizierten Features aller Branches
        concat = torch.cat([feat_act, feat_res, feat_month, feat_trace], dim=1)
        
        # Output Head (LayerNorm -> LeakyReLU -> Dropout -> Softmax/Logits)
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
    
    # 1. Labels für diesen Typ extrahieren
    y_dev = y_all[:, dev_idx].astype(np.int64)
    n_samples = y_dev.shape[0]
    
    # 2. Trace-basiert Split (2/3 Train, 1/3 Test)
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
    case_ids_train = case_ids[train_mask]  # Für trace-basiertes Validation Split
    
    print(f"  Trace Split: {len(train_cases)} Traces Train. (Total Samples: {len(y_train)})")
    
    # 3. Prüfen, ob Train-Set mindestens eine Abweichung enthält
    if y_train.sum() == 0:
        print(f"  [WARN] Keine Abweichungen im Train-Set für {dev_name} (obwohl global >1). Skip.")
        return None, np.zeros(n_samples, dtype=np.float32)

    # 4. Undersampling (OSS)
<<<<<<< Updated upstream
    if cfg.use_oss:
        print(f"  Undersampling (OSS) auf Train-Set...")
        oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
        
        try:
            _X_dummy, _y_dummy = oss.fit_resample(X_trace_train, y_train)
            kept_indices = oss.sample_indices_
            
            X_act_train = X_act_train[kept_indices]
            X_res_train = X_res_train[kept_indices]
            X_month_train = X_month_train[kept_indices]
            X_trace_train = X_trace_train[kept_indices]
            y_train = y_train[kept_indices]
            case_ids_train = case_ids_train[kept_indices]  # Case IDs für resampled Daten
        except Exception as e:
            print(f"  [INFO] OSS fehlgeschlagen ({e}). Nutze volle Daten.")
            pass
    else:
        print(f"  Kein Undersampling (OSS deaktiviert).")
=======
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
        case_ids_train = case_ids_train[kept_indices]  # Case IDs für resampled Daten
    except Exception as e:
        print(f"  [INFO] OSS fehlgeschlagen ({e}). Nutze volle Daten.")
        pass
>>>>>>> Stashed changes

    print(f"  Train-Set nach Undersampling: {len(y_train)} Samples.")
    
    # 5. Validation Split (trace-basiert, 80% Train, 20% Validation)
    unique_train_cases = np.unique(case_ids_train)
    train_cases_final, val_cases = train_test_split(
        unique_train_cases,
        test_size=cfg.validation_split,
        random_state=cfg.random_state
    )
    
    train_final_mask = np.isin(case_ids_train, train_cases_final)
    val_mask = np.isin(case_ids_train, val_cases)
    
    # Final Training Set
    X_act_train_final = X_act_train[train_final_mask]
    X_res_train_final = X_res_train[train_final_mask]
    X_month_train_final = X_month_train[train_final_mask]
    X_trace_train_final = X_trace_train[train_final_mask]
    y_train_final = y_train[train_final_mask]
    
    # Validation Set
    X_act_val = X_act_train[val_mask]
    X_res_val = X_res_train[val_mask]
    X_month_val = X_month_train[val_mask]
    X_trace_val = X_trace_train[val_mask]
    y_val = y_train[val_mask]
    
    print(f"  Validation Split: {len(train_cases_final)} Traces Train, {len(val_cases)} Traces Val")
    print(f"  Samples: {len(y_train_final)} Train, {len(y_val)} Val")

    # 6. Model initialisieren
    device = torch.device(cfg.device)
    model = IDPLSTM(
        meta=meta, 
        trace_input_dim=inputs["X_trace"].shape[1], 
        cfg=cfg
    ).to(device)
    
    # 7. WCEL
    weights = torch.tensor([cfg.alpha_cc, cfg.alpha_dc], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 8. DataLoaders
    train_ds = LSTMDataset(X_act_train_final, X_res_train_final, X_month_train_final, X_trace_train_final, y_train_final)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    val_ds = LSTMDataset(X_act_val, X_res_val, X_month_val, X_trace_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # 9. Training Loop mit Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        # Training Phase
        epoch_train_loss = 0.0
        for (batch_act, batch_res, batch_month, batch_trace), batch_y in train_loader:
            batch_act, batch_res = batch_act.to(device), batch_res.to(device)
            batch_month, batch_trace = batch_month.to(device), batch_trace.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_act, batch_res, batch_month, batch_trace)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation Phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for (batch_act, batch_res, batch_month, batch_trace), batch_y in val_loader:
                batch_act, batch_res = batch_act.to(device), batch_res.to(device)
                batch_month, batch_trace = batch_month.to(device), batch_trace.to(device)
                batch_y = batch_y.to(device)
                
                logits = model(batch_act, batch_res, batch_month, batch_trace)
                loss = criterion(logits, batch_y)
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
    
    # 10. Bestes Modell wiederherstellen
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Bestes Modell wiederhergestellt (Val Loss: {best_val_loss:.4f})")
    else:
        print(f"  [WARN] Kein bestes Modell gefunden, nutze aktuelles Modell")
            
    # 11. Prediction
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
    parser = argparse.ArgumentParser(
        description="Trainiert separate LSTM-Modelle für jeden Deviation Type"
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
        help="Verzeichnis für gespeicherte Modelle. Standard: {input-dir}/models_idp_separate_lstm/"
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

    suffix = "_no_oss" if not cfg.use_oss else ""
    out_path = processed_dir / f"idp_separate_lstm_probs{suffix}.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), # Speichere nur die gefilterten Namen
        case_ids=case_ids
    )
    print(f"Fertig. Ergebnisse gespeichert in {out_path}")

if __name__ == "__main__":
    main()