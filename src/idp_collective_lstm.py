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
# Konfiguration (Section 5.2.4)
# =========================

@dataclass
class IDPCollectiveLSTMConfig:
    """
    Konfiguration für IDP-collectiveLSTM gemäß Paper Section 5.2.4.
    
    Ein einzelnes LSTM-Modell wird trainiert, um alle Deviation Types gleichzeitig
    vorherzusagen (Multi-Label Klassifikation). Die Architektur verwendet 4 Branches:
    Activities, Resources, Month (jeweils LSTM) und Trace Attributes (Feed-Forward).
    
    Hyperparameter aus 'Hyperparameter Optimization' (Section 5.2.4):
      - Embedding dimension: 16
      - LSTM layer size: 128 (Final choice bold in Paper-Text)
      - Feed-forward Projektion pro Branch: 64
      - Dropout: 0.1
      - Loss Weight (WCEL): 16^(1/(2e) + log(LIR_d))
    """
    # Netzwerk-Architektur
    embedding_dim: int = 16      # Dimension der Embeddings für kategorische Features
    lstm_hidden_dim: int = 128   # Hidden Dimension der LSTM-Layer
    projection_dim: int = 64     # Dimension für Feed-Forward Projektion (gleicht Branches an)
    dropout: float = 0.1         # Dropout-Rate zur Regularisierung
    
    # Training-Parameter
    batch_size: int = 256        # Anzahl Samples pro Batch
    lr: float = 1e-3             # Learning Rate für Adam Optimizer
    max_epochs: int = 100        # Maximale Anzahl Epochen (harte Obergrenze)
    
    # Early Stopping Parameter
    patience: int = 10                    # Epochen ohne Verbesserung, bevor Training stoppt
    validation_split: float = 0.2         # Anteil des Training Sets für Validation (20%)
    
    # Undersampling
    use_oss: bool = True  # One-Sided Selection für Undersampling
    
    # System-Parameter
    device: str = "cpu"          # CPU erzwingen (cuDNN-Konflikt auf Server vermeiden)
    random_state: int = 42       # Random Seed für Reproduzierbarkeit


# =========================
# Daten-Layer
# =========================

class CollectiveLSTMDataset(Dataset):
    """
    PyTorch Dataset für Collective LSTM Modell.
    
    Das Modell benötigt 4 verschiedene Input-Komponenten:
    - Activities: Sequenz von Aktivitäten (Integer-IDs)
    - Resources: Sequenz von Ressourcen (Integer-IDs)
    - Month: Sequenz von Monaten (Integer-IDs)
    - Trace Attributes: Kontinuierliche Trace-Features (Float)
    
    Die Labels sind Multi-Label (mehrere Deviation Types können gleichzeitig auftreten).
    """
    def __init__(
        self, 
        X_act: np.ndarray,   # Activities Sequenzen (N × Sequenz-Länge)
        X_res: np.ndarray,   # Resources Sequenzen (N × Sequenz-Länge)
        X_month: np.ndarray, # Month Sequenzen (N × Sequenz-Länge)
        X_trace: np.ndarray, # Trace Attribute Features (N × Feature-Dimension)
        y: np.ndarray        # Multi-Label Matrix (N × Anzahl Deviation Types)
    ):
        # Kategorische Features als Integer (für Embeddings)
        self.X_act = torch.from_numpy(X_act.astype(np.int64))
        self.X_res = torch.from_numpy(X_res.astype(np.int64))
        self.X_month = torch.from_numpy(X_month.astype(np.int64))
        
        # Kontinuierliche Features als Float
        self.X_trace = torch.from_numpy(X_trace.astype(np.float32))
        
        # Labels als Float für Multi-Label Klassifikation (Sigmoid pro Klasse)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        """Gibt die Anzahl der Samples zurück."""
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        """
        Gibt ein Sample zurück: (4 Input-Komponenten als Tupel, Labels).
        
        Returns:
            ((X_act, X_res, X_month, X_trace), y): Input-Tupel und Labels
        """
        return (
            self.X_act[idx], 
            self.X_res[idx], 
            self.X_month[idx], 
            self.X_trace[idx]
        ), self.y[idx]


def load_lstm_data(processed_dir: Path):
    """
    Lädt die vorverarbeiteten Daten für das Collective LSTM Modell.
    
    Args:
        processed_dir: Pfad zum data/processed Verzeichnis
        
    Returns:
        inputs: Dictionary mit 4 Input-Komponenten:
            - X_act: Activities Sequenzen (N × Sequenz-Länge)
            - X_res: Resources Sequenzen (N × Sequenz-Länge)
            - X_month: Month Sequenzen (N × Sequenz-Länge)
            - X_trace: Trace Attribute Features (N × Feature-Dimension)
        y_all: Label-Matrix (N × Anzahl Deviation Types) - Multi-Label
        case_ids: Array mit Case-IDs für jeden Prefix (für trace-basiertes Splitting)
        dev_types: Liste der Deviation Type Namen
        meta: Dictionary mit Vocabularies (für Embedding-Größen)
        
    Raises:
        FileNotFoundError: Falls erforderliche Dateien fehlen
    """
    # 1. LSTM-Encodings laden (Sequenzen für Activities, Resources, Month + Trace Features)
    enc_path = processed_dir / "encoding_lstm.npz"
    if not enc_path.exists():
        raise FileNotFoundError(f"{enc_path} nicht gefunden.")
    enc_data = np.load(enc_path)
    inputs = {
        "X_act": enc_data["X_act_seq"],      # Activities Sequenzen
        "X_res": enc_data["X_res_seq"],      # Resources Sequenzen
        "X_month": enc_data["X_month_seq"],  # Month Sequenzen
        "X_trace": enc_data["X_trace"]       # Kontinuierliche Trace-Features
    }
    
    # 2. Labels und Metadaten laden
    lab_path = processed_dir / "idp_labels.npz"
    if not lab_path.exists():
        raise FileNotFoundError(f"{lab_path} nicht gefunden.")
    lab_data = np.load(lab_path, allow_pickle=True)
    y_all = lab_data["y"]              # Multi-Label Matrix
    case_ids = lab_data["case_ids"]    # Case-IDs für jeden Prefix
    dev_types = list(lab_data["dev_types"])  # Namen der Deviation Types
    
    # 3. Meta-Informationen laden (Vocabularies für Embedding-Größen)
    meta_path = processed_dir / "encoding_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    return inputs, y_all, case_ids, dev_types, meta


# =========================
# Modell Architektur (Fig. 7)
# =========================

class IDPCollectiveLSTM(nn.Module):
    """
    LSTM-Netzwerk für Collective Deviation Prediction (gemäß Paper Fig. 7).
    
    Ein einzelnes Modell, das alle Deviation Types gleichzeitig vorhersagt
    (Multi-Label Klassifikation). Die Architektur besteht aus 4 Branches:
    - 3x LSTM Branches: Activities, Resources, Month (jeweils Embedding -> LSTM -> FF)
    - 1x Feed-Forward Branch: Trace Attributes
    - Alle Branches werden zusammengeführt und durch Output Layer geleitet
    
    Die Sigmoid-Aktivierung wird bei der Vorhersage angewendet (nicht im Forward Pass).
    """
    def __init__(
        self, 
        meta: Dict,              # Vocabularies für Embedding-Größen
        trace_input_dim: int,    # Dimension der Trace Attribute Features
        num_classes: int,        # Anzahl der Deviation Types (Output-Dimension)
        cfg: IDPCollectiveLSTMConfig
    ):
        super().__init__()
        
        # --- Branch 1: Activities ---
        # Embedding: Integer-IDs -> Embedding-Vektoren
        self.emb_act = nn.Embedding(len(meta["activity_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_act = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_act = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)  # Projektion auf gemeinsame Dimension
        
        # --- Branch 2: Resources ---
        self.emb_res = nn.Embedding(len(meta["resource_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_res = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_res = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 3: Month ---
        self.emb_month = nn.Embedding(len(meta["month_vocab"]) + 1, cfg.embedding_dim, padding_idx=0)
        self.lstm_month = nn.LSTM(cfg.embedding_dim, cfg.lstm_hidden_dim, batch_first=True)
        self.ff_month = nn.Linear(cfg.lstm_hidden_dim, cfg.projection_dim)
        
        # --- Branch 4: Trace Attributes (Feed-Forward, kein LSTM) ---
        self.ff_trace = nn.Linear(trace_input_dim, cfg.projection_dim)
        
        # --- Zusammenführung und Output ---
        # Alle 4 Branches werden auf projection_dim projiziert und dann konkateniert
        concat_dim = cfg.projection_dim * 4  # 4 Branches × projection_dim
        
        self.ln = nn.LayerNorm(concat_dim)      # Normalisierung
        self.act = nn.LeakyReLU()                # Aktivierungsfunktion
        self.drop = nn.Dropout(cfg.dropout)      # Regularisierung
        
        # Output Layer: Logits für alle Deviation Types
        self.head = nn.Linear(concat_dim, num_classes)
        
    def forward(self, x_act, x_res, x_month, x_trace):
        """
        Forward Pass durch das Netzwerk.
        
        Args:
            x_act: Activities Sequenz (Batch × Sequenz-Länge)
            x_res: Resources Sequenz (Batch × Sequenz-Länge)
            x_month: Month Sequenz (Batch × Sequenz-Länge)
            x_trace: Trace Attribute Features (Batch × Feature-Dimension)
            
        Returns:
            logits: Unnormalisierte Vorhersagen (Batch × num_classes)
                   Wird später mit Sigmoid zu Wahrscheinlichkeiten konvertiert
        """
        # Branch 1: Activities
        e_act = self.emb_act(x_act)                    # Embedding
        _, (h_act, _) = self.lstm_act(e_act)           # LSTM: h_act hat Shape (1, Batch, Hidden)
        feat_act = self.ff_act(h_act[-1])              # Feed-Forward: (Batch, projection_dim)
        
        # Branch 2: Resources
        e_res = self.emb_res(x_res)
        _, (h_res, _) = self.lstm_res(e_res)
        feat_res = self.ff_res(h_res[-1])
        
        # Branch 3: Month
        e_month = self.emb_month(x_month)
        _, (h_month, _) = self.lstm_month(e_month)
        feat_month = self.ff_month(h_month[-1])
        
        # Branch 4: Trace Attributes (direkt Feed-Forward, kein LSTM)
        feat_trace = self.ff_trace(x_trace)
        
        # Zusammenführung: Alle Branches konkatenieren
        concat = torch.cat([feat_act, feat_res, feat_month, feat_trace], dim=1)
        
        # Output Head: LayerNorm -> LeakyReLU -> Dropout -> Logits
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
    Berechnet die Klassen-Gewichte Beta_d für die gewichtete Loss-Funktion.
    
    Gemäß Paper Section 5.2.4 wird für jeden Deviation Type d ein Gewicht β_d
    berechnet, um die Klassen-Ungleichgewichte auszugleichen.
    
    Formel: β_d = 16 ^ (1/(2e) + log(LIR_d))
    wobei:
      - LIR_d = CC(d) / DC(d) = Label-Imbalance-Ratio
      - CC(d) = Anzahl Conforming Traces für Typ d
      - DC(d) = Anzahl Deviating Traces für Typ d
      - e = Eulersche Zahl
    
    Args:
        y_train: Label-Matrix (N × Anzahl Deviation Types)
        device: Device für den Tensor (CPU/GPU)
        
    Returns:
        Tensor mit Gewichten für jeden Deviation Type (Länge = Anzahl Types)
    """
    n_samples, n_classes = y_train.shape
    weights = []
    
    for i in range(n_classes):
        # Zähle Samples pro Klasse für diesen Deviation Type
        dc = np.sum(y_train[:, i] == 1)  # Deviating (Klasse 1)
        cc = np.sum(y_train[:, i] == 0)  # Conforming (Klasse 0)
        
        # Schutz vor Division durch Null (sollte durch Filterung verhindert werden)
        if dc == 0:
            weights.append(1.0)
            continue
            
        # Label-Imbalance-Ratio berechnen
        lir = cc / dc  # Verhältnis Conforming zu Deviating
        
        # Exponent gemäß Paper-Formel: 1/(2e) + log(LIR_d)
        exponent = (1.0 / (2.0 * np.e)) + np.log(lir)
        
        # Gewicht berechnen: 16 hoch Exponent
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
    """
    Trainiert ein Collective LSTM Modell für alle Deviation Types gleichzeitig.
    
    Ablauf:
    1. Trace-basiertes Split (2/3 Train, 1/3 Test)
    2. Undersampling (OSS) zur Reduzierung der Mehrheitsklasse
    3. Validation Split (80% Train, 20% Validation) für Early Stopping
    4. Berechnung der Klassen-Gewichte
    5. Training mit Early Stopping
    6. Vorhersagen auf allen Daten
    
    Args:
        inputs: Dictionary mit 4 Input-Komponenten (X_act, X_res, X_month, X_trace)
        y_all: Label-Matrix (alle Prefixe × alle Deviation Types)
        case_ids: Case-IDs für jeden Prefix
        meta: Dictionary mit Vocabularies (für Embedding-Größen)
        cfg: Konfiguration
        
    Returns:
        model: Trainiertes Modell
        P_D: Vorhersage-Wahrscheinlichkeiten für alle Daten (N × Anzahl Types)
    """
    print("\n--- Training IDP-collective-LSTM ---")
    
    # 1. Trace-basiertes Split: 2/3 Train, 1/3 Test
    # Wichtig: Split erfolgt auf Trace-Ebene, nicht Prefix-Ebene, um Data Leakage zu vermeiden
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0,  # 1/3 für Test (wird nicht verwendet, bleibt für Evaluation)
        random_state=cfg.random_state
    )
    
    # Alle Prefixe zu Train-Traces gehören zum Training Set
    train_mask = np.isin(case_ids, train_cases)
    
    # Alle 4 Input-Komponenten filtern
    X_act_train = inputs["X_act"][train_mask]
    X_res_train = inputs["X_res"][train_mask]
    X_month_train = inputs["X_month"][train_mask]
    X_trace_train = inputs["X_trace"][train_mask]
    y_train = y_all[train_mask]
    case_ids_train = case_ids[train_mask]  # Wird später für Validation Split benötigt
    
    print(f"  Split: {len(train_cases)} Traces im Training (Samples: {len(y_train)}).")

    # 2. Collective Undersampling (OSS - One-Sided Selection)
    if cfg.use_oss:
        print("  Collective Undersampling (OSS)...")
        
        # Hilfs-Label erstellen: Hat ein Prefix irgendeine Abweichung? (0=conforming, 1=deviant)
        # Dies wird für OSS verwendet, das auf binärer Klassifikation basiert
        y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
        n_dev_before = int(np.sum(y_is_deviant))
        n_conf_before = int(len(y_is_deviant) - n_dev_before)
        print(f"  Vor OSS - Deviant Traces: {n_dev_before}, Conforming Traces: {n_conf_before}")
        
        try:
            # OSS: Entfernt redundante Conforming Samples, behält alle Deviating Samples
            oss = OneSidedSelection(
                n_neighbors=7,              # Anzahl Nachbarn für Tomek Links
                n_seeds_S=250,              # Anzahl Seed-Samples
                random_state=cfg.random_state
            )
            
            # OSS benötigt eine flache Feature-Repräsentation
            # Daher werden alle 4 Input-Komponenten zu einem Feature-Vektor konkateniert
            n_samples_train = X_trace_train.shape[0]
            X_oss_train = np.concatenate(
                [
                    X_act_train.reshape(n_samples_train, -1),   # Activities flach machen
                    X_res_train.reshape(n_samples_train, -1),  # Resources flach machen
                    X_month_train.reshape(n_samples_train, -1), # Month flach machen
                    X_trace_train,                              # Trace Features (bereits flach)
                ],
                axis=1,
            )
            
            # OSS anwenden (gibt Indizes der behaltenen Samples zurück)
            _X_dummy, _y_dummy = oss.fit_resample(X_oss_train, y_is_deviant)
            kept_indices = oss.sample_indices_
            
            # Alle 4 Input-Komponenten und Labels filtern
            X_act_train = X_act_train[kept_indices]
            X_res_train = X_res_train[kept_indices]
            X_month_train = X_month_train[kept_indices]
            X_trace_train = X_trace_train[kept_indices]
            y_train = y_train[kept_indices]
            case_ids_train = case_ids_train[kept_indices]  # Case IDs müssen mitgefiltert werden
            
            # Verteilung nach OSS erneut auswerten
            y_is_deviant_res = (np.sum(y_train, axis=1) > 0).astype(int)
            n_dev_after = int(np.sum(y_is_deviant_res))
            n_conf_after = int(len(y_is_deviant_res) - n_dev_after)
            print(f"  Resampled Train-Set: {len(y_train)} Samples.")
            print(f"  Nach OSS - Deviant Traces: {n_dev_after}, Conforming Traces: {n_conf_after}")
        except Exception as e:
            # Falls OSS fehlschlägt (z.B. zu wenige Samples), verwende alle Daten
            print(f"  [WARN] OSS fehlgeschlagen ({e}). Nutze volle Daten.")
    else:
        print("  Kein Undersampling (OSS deaktiviert).")
    
    # 3. Validation Split (trace-basiert, 80% Train, 20% Validation)
    # Das Training Set wird weiter aufgeteilt für Early Stopping
    # Wichtig: Wieder trace-basiert, um Data Leakage zu vermeiden
    unique_train_cases = np.unique(case_ids_train)
    train_cases_final, val_cases = train_test_split(
        unique_train_cases,
        test_size=cfg.validation_split,  # 20% für Validation
        random_state=cfg.random_state
    )
    
    # Masken für finales Training Set und Validation Set
    train_final_mask = np.isin(case_ids_train, train_cases_final)
    val_mask = np.isin(case_ids_train, val_cases)
    
    # Final Training Set (80% des ursprünglichen Training Sets)
    X_act_train_final = X_act_train[train_final_mask]
    X_res_train_final = X_res_train[train_final_mask]
    X_month_train_final = X_month_train[train_final_mask]
    X_trace_train_final = X_trace_train[train_final_mask]
    y_train_final = y_train[train_final_mask]
    
    # Validation Set (20% des ursprünglichen Training Sets)
    X_act_val = X_act_train[val_mask]
    X_res_val = X_res_train[val_mask]
    X_month_val = X_month_train[val_mask]
    X_trace_val = X_trace_train[val_mask]
    y_val = y_train[val_mask]
    
    print(f"  Validation Split: {len(train_cases_final)} Traces Train, {len(val_cases)} Traces Val")
    print(f"  Samples: {len(y_train_final)} Train, {len(y_val)} Val")

    # 4. Klassen-Gewichte berechnen
    # Die Gewichte werden auf dem finalen Training Set berechnet
    beta_d = calculate_collective_weights(y_train_final, cfg.device)
    
    # 5. Modell initialisieren
    device = torch.device(cfg.device)
    model = IDPCollectiveLSTM(
        meta=meta,                              # Vocabularies für Embedding-Größen
        trace_input_dim=inputs["X_trace"].shape[1],  # Dimension der Trace Features
        num_classes=y_all.shape[1],             # Anzahl Deviation Types
        cfg=cfg
    ).to(device)
    
    # Loss-Funktion: Multi-Label Weighted Cross-Entropy (WCEL)
    # MultiLabelSoftMarginLoss kombiniert Sigmoid + BCE Loss mit Klassen-Gewichten
    # Die Gewichte β_d gleichen Klassen-Ungleichgewichte aus
    criterion = nn.MultiLabelSoftMarginLoss(weight=beta_d)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 6. DataLoaders erstellen
    train_ds = CollectiveLSTMDataset(X_act_train_final, X_res_train_final, X_month_train_final, X_trace_train_final, y_train_final)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    val_ds = CollectiveLSTMDataset(X_act_val, X_res_val, X_month_val, X_trace_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # 7. Training Loop mit Early Stopping
    # Variablen zum Tracking des besten Modells
    best_val_loss = float('inf')      # Beste (niedrigste) Validation Loss
    patience_counter = 0               # Epochen ohne Verbesserung
    best_model_state = None            # Gewichte des besten Modells
    
    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        # Training Phase: Modell lernt auf Training Set
        epoch_train_loss = 0.0
        for (b_act, b_res, b_month, b_trace), b_y in train_loader:
            # Alle 4 Input-Komponenten auf Device verschieben
            b_act, b_res = b_act.to(device), b_res.to(device)
            b_month, b_trace = b_month.to(device), b_trace.to(device)
            b_y = b_y.to(device)
            
            optimizer.zero_grad()      # Gradienten zurücksetzen
            logits = model(b_act, b_res, b_month, b_trace)  # Forward Pass
            loss = criterion(logits, b_y)  # Loss berechnen
            loss.backward()            # Gradienten berechnen
            optimizer.step()           # Gewichte aktualisieren
            epoch_train_loss += loss.item()
        
        # Validation Phase: Modell wird auf Validation Set getestet (ohne Training)
        model.eval()  # Wichtig: Evaluation-Modus (kein Dropout, keine Gradienten)
        epoch_val_loss = 0.0
        with torch.no_grad():  # Keine Gradienten berechnen (spart Speicher)
            for (b_act, b_res, b_month, b_trace), b_y in val_loader:
                b_act, b_res = b_act.to(device), b_res.to(device)
                b_month, b_trace = b_month.to(device), b_trace.to(device)
                b_y = b_y.to(device)
                
                logits = model(b_act, b_res, b_month, b_trace)
                loss = criterion(logits, b_y)
                epoch_val_loss += loss.item()
        
        # Durchschnittliche Loss-Werte berechnen
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Logging: Fortschritt ausgeben
        print(f"  Epoch {epoch}/{cfg.max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping Check: Hat sich das Modell verbessert?
        if avg_val_loss < best_val_loss:
            # ✅ Verbesserung: Bestes Modell speichern
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # Gewichte kopieren
            print(f"    -> Bestes Modell gespeichert (Val Loss: {best_val_loss:.4f})")
        else:
            # ❌ Keine Verbesserung: Counter erhöhen
            patience_counter += 1
            if patience_counter >= cfg.patience:
                # Patience erreicht: Training stoppen
                print(f"  Early Stopping nach {epoch} Epochen (Patience: {cfg.patience})")
                break
        
        model.train()  # Zurück zu Training-Modus für nächste Epoche
    
    # Training Zusammenfassung
    actual_epochs = epoch
    if actual_epochs < cfg.max_epochs:
        print(f"  Training beendet nach {actual_epochs} Epochen (Early Stopping)")
    else:
        print(f"  Training abgeschlossen nach {actual_epochs} Epochen (Max Epochs erreicht)")
    
    # 8. Bestes Modell wiederherstellen
    # Das Modell könnte am Ende schlechter sein als in einer früheren Epoche
    # Daher laden wir die Gewichte des besten Modells (niedrigste Validation Loss)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Bestes Modell wiederhergestellt (Val Loss: {best_val_loss:.4f})")
    else:
        print(f"  [WARN] Kein bestes Modell gefunden, nutze aktuelles Modell")
            
    # 9. Vorhersagen auf allen Daten (inkl. Test Set)
    # Wichtig: Vorhersagen werden auf ALLEN Daten gemacht (Train + Validation + Test)
    # Die Evaluation erfolgt später nur auf dem Test Set
    full_ds = CollectiveLSTMDataset(
        inputs["X_act"], inputs["X_res"], inputs["X_month"], inputs["X_trace"], y_all
    )
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=False)
    
    model.eval()
    probs_list = []
    with torch.no_grad():
        for (b_act, b_res, b_month, b_trace), _ in full_loader:
            # Alle 4 Input-Komponenten auf Device verschieben
            b_act, b_res = b_act.to(device), b_res.to(device)
            b_month, b_trace = b_month.to(device), b_trace.to(device)
            
            logits = model(b_act, b_res, b_month, b_trace)
            # Sigmoid für Multi-Label: Jede Klasse unabhängig (Wahrscheinlichkeit pro Type)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())
            
    # Alle Vorhersagen zusammenführen
    P_D = np.concatenate(probs_list, axis=0)  # (N Prefixe × Anzahl Deviation Types)
    return model, P_D


# =========================
# Main Execution
# =========================

def main():
    """
    Hauptfunktion: Trainiert ein Collective LSTM Modell für alle Deviation Types.
    
    Pipeline:
    1. Daten laden (4 Input-Komponenten + Labels)
    2. Deviation Types filtern (nur Types in >1 Trace)
    3. Modell trainieren
    4. Modell und Vorhersagen speichern
    """
    parser = argparse.ArgumentParser(
        description="Trainiert ein Collective LSTM-Modell für alle Deviation Types"
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
        help="Verzeichnis für gespeicherte Modelle. Standard: {input-dir}/models_idp_collective_lstm/"
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
        models_dir = processed_dir / "models_idp_collective_lstm"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Start IDP-collective-LSTM Training ---")
    cfg = IDPCollectiveLSTMConfig()
    print(f"Device set to: {cfg.device}")
    
    # 1. Daten laden (4 Input-Komponenten: Activities, Resources, Month, Trace Features)
    inputs, y_all, case_ids, dev_types, meta = load_lstm_data(processed_dir)
    print(f"Deviation Types Raw: {len(dev_types)}")
    
    # 2. Filterung gemäß Paper Requirement
    # Paper Section 5.1: "remove deviation types, that occur in one trace only"
    # Ein Deviation Type muss in mindestens 2 verschiedenen Traces vorkommen,
    # um trainierbar zu sein
    keep_indices = []
    
    for i in range(len(dev_types)):
        # Prüfe, in wie vielen verschiedenen Traces dieser Type vorkommt
        dev_col = y_all[:, i]  # Alle Labels für diesen Deviation Type
        cases_with_dev = np.unique(case_ids[dev_col == 1])  # Unique Traces mit dieser Abweichung
        if len(cases_with_dev) > 1:
            keep_indices.append(i)  # Type behalten
    
    if not keep_indices:
        print("Keine trainierbaren Abweichungen. Abbruch.")
        return
        
    # Labels und Type-Namen filtern
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    print(f"Training mit {len(dev_types_filtered)} Deviation Types.")
    print("-" * 40)
    
    # 3. Modell trainieren
    model, P_dev_all = train_collective_lstm(
        inputs, y_filtered, case_ids, meta, cfg
    )
    
    # 4. Ergebnisse speichern
    # Modell-Gewichte speichern
    model_path = models_dir / "collective_lstm.pt"
    torch.save(model.state_dict(), model_path)
    
    # Vorhersage-Wahrscheinlichkeiten speichern (für Evaluation)
    suffix = "_no_oss" if not cfg.use_oss else ""
    out_path = processed_dir / f"idp_collective_lstm_probs{suffix}.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,                    # Vorhersage-Wahrscheinlichkeiten
        dev_types=np.array(dev_types_filtered),  # Nur gefilterte Namen
        case_ids=case_ids                    # Für Test-Split Rekonstruktion
    )
    print(f"Fertig. Ergebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()