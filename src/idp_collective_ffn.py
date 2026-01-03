from __future__ import annotations

import argparse
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
    
    einzelnes Modell wird trainiert, um alle Deviation Types gleichzeitig
    vorherzusagen (Multi-Label Klassifikation).
    
    Hyperparameter aus 'Hyperparameter Optimization' (Section 5.2.3):
      - Hidden layer structure: 2048 x 1024
      - Dropout: 0.1
      - Loss Weight Formula: 16^(1/(2e) + log(LIR))
    """
    # Netzwerk-Architektur
    hidden_dim_1: int = 2048  # Größe des ersten Hidden Layers
    hidden_dim_2: int = 1024  # Größe des zweiten Hidden Layers
    dropout: float = 0.1      # Dropout-Rate zur Regularisierung
    
    # Training-Parameter
    batch_size: int = 256     # Anzahl Samples pro Batch
    lr: float = 1e-3          # Learning Rate für Adam Optimizer
    max_epochs: int = 300      # Maximale Anzahl Epochen (harte Obergrenze)
    
    # Early Stopping Parameter
    patience: int = 10                    # Epochen ohne Verbesserung, bevor Training stoppt
    validation_split: float = 0.2         # Anteil des Training Sets für Validation (20%)
    
    # Undersampling
    use_oss: bool = True  # One-Sided Selection für Undersampling
    
    # System-Parameter
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # GPU/CPU
    random_state: int = 42    # Random Seed für Reproduzierbarkeit


# =========================
# Daten-Layer
# =========================

class CollectiveDataset(Dataset):
    """
    PyTorch Dataset für Collective FFN Modell.
    
    - Konvertiert NumPy-Arrays in PyTorch-Tensoren für das Training.
    - Labels sind als Float-Tensoren, da Multi-Label Klassifikation
      - Sigmoid-Aktivierung verwendet (jede Klasse unabhängig).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        # Float-Targets für Multi-Label Klassifikation (Sigmoid pro Klasse)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        """Gibt die Anzahl der Samples zurück."""
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        """Gibt ein Sample (Features, Labels) zurück."""
        return self.X[idx], self.y[idx]


def load_ffn_data(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Lädt die vorverarbeiteten Daten für das Collective FFN Modell.
    
    - CIBE-Features laden 
    - Labels und Metadaten laden
    - Prüfen, dass Anzahl der Prefixe übereinstimmt (X.shape[0] == y.shape[0])
    Returns:
        X: Feature-Matrix (N Prefixe × Feature-Dimension) - CIBE-Encodings
        y: Label-Matrix (N Prefixe × Anzahl Deviation Types) - Multi-Label
        case_ids: Array mit Case-IDs für jeden Prefix (für trace-basiertes Splitting)
        dev_types: Liste der Deviation Type Namen
        
    """
    # CIBE-Features laden (Continuous Interval-Based Encoding)
    feat_path = processed_dir / "encoding_ffn.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} nicht gefunden.")
    X = np.load(feat_path)

    # Labels und Metadaten laden
    label_path = processed_dir / "idp_labels.npz"
    if not label_path.exists():
        raise FileNotFoundError(f"{label_path} nicht gefunden.")
    data = np.load(label_path, allow_pickle=True)
    
    y = data["y"]              # Multi-Label Matrix
    case_ids = data["case_ids"]  # Case-IDs für jeden Prefix
    dev_types = list(data["dev_types"])  # Namen der Deviation Types

    return X, y, case_ids, dev_types


# =========================
# Modell-Architektur (Fig. 6)
# =========================

class IDPCollectiveFFN(nn.Module):
    """
    Feed-Forward Netzwerk für Collective Deviation Prediction (Fig. 6).
    
    Ein einzelnes Modell, das alle Deviation Types gleichzeitig vorhersagt
    (Multi-Label Klassifikation). Die Architektur besteht aus zwei Hidden Layers
    mit LayerNorm, LeakyReLU, Dropout und einer Sigmoid-Aktivierung im Output Layer
    (gemäß Paper Section 5.2.3 und Fig. 6).
    """
    def __init__(self, input_dim: int, num_classes: int, cfg: IDPCollectiveFFNConfig):
        """
        Args:
            input_dim: Dimension der Input-Features (CIBE-Encoding)
            num_classes: Anzahl der Deviation Types (Output-Dimension)
            cfg: Konfiguration mit Hyperparametern
        """
        super().__init__()
        
        # Erster Hidden Layer: Input -> 2048
        self.layer1 = nn.Linear(input_dim, cfg.hidden_dim_1)
        self.ln1 = nn.LayerNorm(cfg.hidden_dim_1)  # Normalisierung für Stabilität
        self.act1 = nn.LeakyReLU()                  # Aktivierungsfunktion
        
        # Zweiter Hidden Layer: 2048 -> 1024
        self.layer2 = nn.Linear(cfg.hidden_dim_1, cfg.hidden_dim_2)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim_2)
        self.act2 = nn.LeakyReLU()
        self.drop = nn.Dropout(cfg.dropout)         # Regularisierung
        
        # Output Layer: 1024 -> num_classes
        self.head = nn.Linear(cfg.hidden_dim_2, num_classes)
        # Sigmoid-Aktivierung für Multi-Label Klassifikation (gemäß Paper Fig. 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass durch das Netzwerk.
        
        Args:
            x: Input Features (Batch × Feature-Dimension)
            
        Returns:
            probs: Wahrscheinlichkeiten (Batch × num_classes) - Werte zwischen 0 und 1
        """
        # Erster Block: Linear -> LayerNorm -> LeakyReLU
        x = self.act1(self.ln1(self.layer1(x)))
        
        # Zweiter Block: Linear -> LayerNorm -> LeakyReLU -> Dropout
        x = self.drop(self.act2(self.ln2(self.layer2(x))))
        
        # Output: Logits -> Sigmoid -> Wahrscheinlichkeiten (gemäß Paper Fig. 6)
        logits = self.head(x)
        probs = self.sigmoid(logits)
        return probs


# =========================
# Helper: Loss Weight Calculation
# =========================

def calculate_collective_weights(y_train: np.ndarray, device: str) -> torch.Tensor:
    """
    Berechnet die Klassen-Gewichte Beta_d für die gewichtete Loss-Funktion.
    
    für jeden Deviation Type d ein Gewicht β_d
    berechnet, um die Klassen-Ungleichgewichte auszugleichen.
    
    Formel: β_d = 16 ^ (1/(2e) + log(LIR_d))
    wobei:
      - LIR_d = CC(d) / DC(d) = Label-Imbalance-Ratio (Verhältnis Conforming zu Deviating Traces)
      - CC(d) = Anzahl Conforming Traces für Typ d
      - DC(d) = Anzahl Deviating Traces für Typ d
      - e = Eulersche Zahl
    
    Args:
        y_train: Label-Matrix (N × Anzahl Deviation Types)
        
    Returns:
        Tensor mit Gewichten für jeden Deviation Type (Länge = Anzahl Types)
    """
    n_samples, n_classes = y_train.shape
    weights = []
    
    e = np.e  # Eulersche Zahl
    
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
        exponent = (1.0 / (2.0 * e)) + np.log(lir)

        # Gewicht berechnen: 16 hoch Exponent
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
    """
    Trainiert ein Collective FFN Modell für alle Deviation Types gleichzeitig.
    
    Ablauf:
    1. Trace-basiertes Split (2/3 Train, 1/3 Test)
    2. Undersampling (OSS) zur Reduzierung der Mehrheitsklasse
    3. Validation Split (80% Train, 20% Validation) für Early Stopping
    4. Berechnung der Klassen-Gewichte
    5. Training mit Early Stopping
    6. Vorhersagen auf allen Daten
    
    Args:
        X: Feature-Matrix (alle Prefixe)
        y_all: Label-Matrix (alle Prefixe × alle Deviation Types)
        case_ids: Case-IDs für jeden Prefix
        cfg: Konfiguration
        
    Returns:
        model: Trainiertes Modell
        P_D: Vorhersage-Wahrscheinlichkeiten für alle Daten (N × Anzahl Types)
    """
    print("\n--- Training IDP-collective-FFN ---")
    
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
    
    X_train = X[train_mask]
    y_train = y_all[train_mask]
    case_ids_train = case_ids[train_mask]  # Wird später für Validation Split benötigt
    
    print(f"  Split: {len(train_cases)} Traces im Training.")
    print(f"  Samples Train: {X_train.shape[0]}")

    # 2. Collective Undersampling (OSS - One-Sided Selection)
    # Reduziert die Anzahl der Conforming Traces, um Klassen-Ungleichgewicht zu verringern
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
                random_state=cfg.random_state,
                n_seeds_S=250,      # Anzahl Seed-Samples
                n_neighbors=7       # k im kNN-Schritt von OSS; Tomek-Links-Cleaning passiert danach separat (1-NN)
            )
            
            # OSS anwenden (gibt Indizes der behaltenen Samples zurück)
            _X_dummy, _y_dummy = oss.fit_resample(X_train, y_is_deviant)  # Tomek Links identifizieren
            kept_indices = oss.sample_indices_
            
            # Nur die behaltenen Samples verwenden
            X_train_res = X_train[kept_indices]
            y_train_res = y_train[kept_indices]
            case_ids_train = case_ids_train[kept_indices]  # Case IDs müssen mitgefiltert werden
            
            # Verteilung nach OSS erneut auswerten
            y_is_deviant_res = (np.sum(y_train_res, axis=1) > 0).astype(int)
            n_dev_after = int(np.sum(y_is_deviant_res))
            n_conf_after = int(len(y_is_deviant_res) - n_dev_after)
            print(f"  Resampled Train-Set: {len(X_train_res)} Samples.")
            print(f"  Nach OSS - Deviant Traces: {n_dev_after}, Conforming Traces: {n_conf_after}")
        except Exception as e:
            # Falls OSS fehlschlägt (z.B. zu wenige Samples), verwende alle Daten
            print(f"  [WARN] OSS fehlgeschlagen ({e}). Nutze volle Daten.")
            X_train_res, y_train_res = X_train, y_train
    else:
        print("  Kein Undersampling (OSS deaktiviert).")
        X_train_res, y_train_res = X_train, y_train
    
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
    X_train_final = X_train_res[train_final_mask]
    y_train_final = y_train_res[train_final_mask]
    
    # Validation Set (20% des ursprünglichen Training Sets)
    X_val = X_train_res[val_mask]
    y_val = y_train_res[val_mask]
    
    print(f"  Validation Split: {len(train_cases_final)} Traces Train, {len(val_cases)} Traces Val")
    print(f"  Samples: {len(y_train_final)} Train, {len(y_val)} Val")

    # 4. Klassen-Gewichte berechnen
    # Die Gewichte werden auf dem finalen Training Set berechnet
    beta_d = calculate_collective_weights(y_train_final, cfg.device)
    
    # 5. Modell initialisieren
    device = torch.device(cfg.device)
    model = IDPCollectiveFFN(
        input_dim=X.shape[1],           # Feature-Dimension
        num_classes=y_all.shape[1],     # Anzahl Deviation Types
        cfg=cfg
    ).to(device)
    
    # Loss-Funktion: Weighted Binary Cross-Entropy (WCEL)
    # Da das Modell jetzt direkt Wahrscheinlichkeiten ausgibt (Sigmoid im Modell),
    # verwenden wir BCELoss. Die Gewichte werden manuell angewendet.
    # Die Gewichte β_d gleichen Klassen-Ungleichgewichte aus
    criterion = nn.BCELoss(reduction='none')  # reduction='none' für manuelle Gewichtung
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 6. DataLoaders erstellen
    train_ds = CollectiveDataset(X_train_final, y_train_final)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    val_ds = CollectiveDataset(X_val, y_val)
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
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()      # Gradienten zurücksetzen
            probs = model(bx)          # Forward Pass (gibt Wahrscheinlichkeiten aus)
            # Gewichteter Loss: BCELoss pro Klasse, dann mit beta_d gewichtet
            loss_per_sample = criterion(probs, by)  # (batch_size, num_classes)
            weighted_loss = (loss_per_sample * beta_d.unsqueeze(0)).mean()  # Mittel über alle Klassen
            loss = weighted_loss
            loss.backward()            # Gradienten berechnen
            optimizer.step()           # Gewichte aktualisieren
            epoch_train_loss += loss.item()
        
        # Validation Phase: Modell wird auf Validation Set getestet (ohne Training)
        model.eval()  # Wichtig: Evaluation-Modus (kein Dropout, keine Gradienten)
        epoch_val_loss = 0.0
        with torch.no_grad():  # Keine Gradienten berechnen (spart Speicher)
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                probs = model(bx)       # Modell gibt Wahrscheinlichkeiten aus
                # Gewichteter Loss: BCELoss pro Klasse, dann mit beta_d gewichtet
                loss_per_sample = criterion(probs, by)  # (batch_size, num_classes)
                weighted_loss = (loss_per_sample * beta_d.unsqueeze(0)).mean()  # Mittel über alle Klassen
                loss = weighted_loss
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
            # Keine Verbesserung: Counter erhöhen
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
    full_ds = CollectiveDataset(X, y_all)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size * 2, shuffle=False)
    
    model.eval()
    probs_list = []
    with torch.no_grad():
        for bx, _ in full_loader:
            bx = bx.to(device)
            # Modell gibt direkt Wahrscheinlichkeiten aus (Sigmoid ist im Modell)
            probs = model(bx)
            probs_list.append(probs.cpu().numpy())
            
    # Alle Vorhersagen zusammenführen
    P_D = np.concatenate(probs_list, axis=0)  # (N Prefixe × Anzahl Deviation Types)
    return model, P_D


# =========================
# Main Execution
# =========================

def main():
    """
    Hauptfunktion: Trainiert ein Collective FFN Modell für alle Deviation Types.
    
    Pipeline:
    1. Daten laden
    2. Deviation Types filtern (nur Types in >1 Trace)
    3. Modell trainieren
    4. Modell und Vorhersagen speichern
    """
    parser = argparse.ArgumentParser(
        description="Trainiert ein Collective FFN-Modell für alle Deviation Types"
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
        help="Verzeichnis für gespeicherte Modelle. Standard: {input-dir}/models_idp_collective_ffn/"
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
        models_dir = processed_dir / "models_idp_collective_ffn"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Start IDP-collective-FFN Pipeline ---")
    cfg = IDPCollectiveFFNConfig()
    print(f"Device: {cfg.device}")
    
    # 1. Daten laden
    X, y_all, case_ids, dev_types = load_ffn_data(processed_dir)
    print(f"Raw Deviation Types: {len(dev_types)}")
    
    # 2. Filterung gemäß Paper Requirement
    # Paper Section 5.1: "remove deviation types, that occur in one trace only"
    # Ein Deviation Type muss in mindestens 2 verschiedenen Traces vorkommen,
    # um trainierbar zu sein
    keep_indices = []
    dropped_names = []
    
    for i in range(len(dev_types)):
        # Prüfe, in wie vielen verschiedenen Traces dieser Type vorkommt
        dev_col = y_all[:, i]  # Alle Labels für diesen Deviation Type
        cases_with_dev = np.unique(case_ids[dev_col == 1])  # Unique Traces mit dieser Abweichung
        count = len(cases_with_dev)
        
        if count > 1:
            keep_indices.append(i)  # Type behalten
        else:
            dropped_names.append(dev_types[i])  # Type entfernen
            
    if dropped_names:
        print(f"Entferne Deviation Types (<= 1 Trace): {dropped_names}")
    
    if not keep_indices:
        print("Keine Deviation Types übrig. Abbruch.")
        return
        
    # Labels und Type-Namen filtern
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    print(f"Training mit {len(dev_types_filtered)} Deviation Types.")
    print("-" * 40)
    
    # 3. Modell trainieren
    model, P_dev_all = train_collective_model(X, y_filtered, case_ids, cfg)
    
    # 4. Ergebnisse speichern
    # Modell-Gewichte speichern
    model_path = models_dir / "collective_ffn.pt"
    torch.save(model.state_dict(), model_path)
    
    # Vorhersage-Wahrscheinlichkeiten speichern (für Evaluation)
    suffix = "_no_oss" if not cfg.use_oss else ""
    out_path = processed_dir / f"idp_collective_ffn_probs{suffix}.npz"
    np.savez_compressed(
        out_path,
        P_dev=P_dev_all,                                    # Vorhersage-Wahrscheinlichkeiten
        dev_types=np.array(dev_types_filtered),            # WICHTIG: Nur gefilterte Namen
        case_ids=case_ids                                   # Für Test-Split Rekonstruktion
    )
    print(f"Fertig. Ergebnisse gespeichert: {out_path}")


if __name__ == "__main__":
    main()