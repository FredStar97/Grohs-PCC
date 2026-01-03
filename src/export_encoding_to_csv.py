from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# =========================
# Konfiguration (wie im Training)
# =========================

RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0  # 2/3 Train, 1/3 Test
VALIDATION_SPLIT = 0.2  # 80% Train, 20% Validation


# =========================
# Daten laden
# =========================

def load_all_data(processed_dir: Path):
    """
    Lädt alle Encoding-Daten und Metadaten.
    
    Returns:
        X_ffn: FFN-Encoding (N × Feature-Dimension)
        lstm_inputs: Dictionary mit LSTM-Inputs (X_act, X_res, X_month, X_trace)
        case_ids: Case-IDs für jeden Prefix
    """
    # 1. FFN-Encoding laden
    ffn_path = processed_dir / "encoding_ffn.npy"
    if not ffn_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {ffn_path}")
    X_ffn = np.load(ffn_path)
    print(f"✓ FFN-Encoding geladen: {X_ffn.shape}")
    
    # 2. LSTM-Encoding laden
    lstm_path = processed_dir / "encoding_lstm.npz"
    if not lstm_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {lstm_path}")
    lstm_data = np.load(lstm_path)
    lstm_inputs = {
        "X_act": lstm_data["X_act_seq"],
        "X_res": lstm_data["X_res_seq"],
        "X_month": lstm_data["X_month_seq"],
        "X_trace": lstm_data["X_trace"]
    }
    print(f"✓ LSTM-Encoding geladen:")
    print(f"  - Activities: {lstm_inputs['X_act'].shape}")
    print(f"  - Resources: {lstm_inputs['X_res'].shape}")
    print(f"  - Month: {lstm_inputs['X_month'].shape}")
    print(f"  - Trace: {lstm_inputs['X_trace'].shape}")
    
    # 3. Case-IDs laden
    labels_path = processed_dir / "idp_labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {labels_path}")
    labels_data = np.load(labels_path, allow_pickle=True)
    case_ids = labels_data["case_ids"]
    print(f"✓ Case-IDs geladen: {len(case_ids)} Prefixe")
    
    # Sanity Check: Alle Arrays sollten gleich viele Zeilen haben
    n_samples = X_ffn.shape[0]
    assert lstm_inputs["X_act"].shape[0] == n_samples, "Shape mismatch bei Activities"
    assert lstm_inputs["X_res"].shape[0] == n_samples, "Shape mismatch bei Resources"
    assert lstm_inputs["X_month"].shape[0] == n_samples, "Shape mismatch bei Month"
    assert lstm_inputs["X_trace"].shape[0] == n_samples, "Shape mismatch bei Trace"
    assert len(case_ids) == n_samples, "Shape mismatch bei Case-IDs"
    
    return X_ffn, lstm_inputs, case_ids


# =========================
# Split-Logik (wie im Training)
# =========================

def create_splits(case_ids: np.ndarray):
    """
    Erstellt Train/Validation/Test Splits basierend auf Case-IDs.
    Gleiche Logik wie im Training, aber OHNE OSS (da wir komplette Daten wollen).
    
    Returns:
        train_mask: Maske für Training Set (nach Validation Split)
        val_mask: Maske für Validation Set
        test_mask: Maske für Test Set
    """
    # 1. Erster Split: 2/3 Train, 1/3 Test (trace-basiert)
    unique_cases = np.unique(case_ids)
    train_cases, test_cases = train_test_split(
        unique_cases,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Masken für Train und Test
    train_mask_initial = np.isin(case_ids, train_cases)
    test_mask = np.isin(case_ids, test_cases)
    
    # Case-IDs für Train-Set (für Validation Split)
    case_ids_train = case_ids[train_mask_initial]
    unique_train_cases = np.unique(case_ids_train)
    
    # 2. Zweiter Split: 80% Train, 20% Validation (trace-basiert auf Train-Set)
    train_cases_final, val_cases = train_test_split(
        unique_train_cases,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE
    )
    
    # Masken für finales Training Set und Validation Set
    train_final_mask = np.isin(case_ids_train, train_cases_final)
    val_mask_relative = np.isin(case_ids_train, val_cases)
    
    # Absolute Masken (bezogen auf alle Daten)
    train_indices = np.where(train_mask_initial)[0]
    train_mask = np.zeros(len(case_ids), dtype=bool)
    train_mask[train_indices[train_final_mask]] = True
    
    val_mask = np.zeros(len(case_ids), dtype=bool)
    val_mask[train_indices[val_mask_relative]] = True
    
    print(f"\n📊 Split-Statistik:")
    print(f"  Train:   {np.sum(train_mask):,} Prefixe ({len(train_cases_final):,} Traces)")
    print(f"  Val:     {np.sum(val_mask):,} Prefixe ({len(val_cases):,} Traces)")
    print(f"  Test:    {np.sum(test_mask):,} Prefixe ({len(test_cases):,} Traces)")
    
    return train_mask, val_mask, test_mask


# =========================
# FFN-Encoding exportieren
# =========================

def export_ffn_encoding(X_ffn: np.ndarray, case_ids: np.ndarray, masks: dict, output_dir: Path):
    """
    Exportiert FFN-Encoding als CSV für Train/Val/Test Sets.
    
    Args:
        X_ffn: FFN-Encoding Matrix (N × Feature-Dimension)
        case_ids: Case-IDs für jeden Prefix
        masks: Dictionary mit 'train', 'val', 'test' Masken
        output_dir: Ausgabe-Verzeichnis
    """
    print("\n📝 Exportiere FFN-Encoding...")
    
    for split_name, mask in masks.items():
        # Daten filtern
        X_split = X_ffn[mask]
        case_ids_split = case_ids[mask]
        prefix_indices = np.where(mask)[0]
        
        # DataFrame erstellen
        n_features = X_split.shape[1]
        data_dict = {
            "prefix_idx": prefix_indices,
            "case_id": case_ids_split
        }
        
        # Feature-Spalten hinzufügen
        for i in range(n_features):
            data_dict[f"feature_{i}"] = X_split[:, i]
        
        df = pd.DataFrame(data_dict)
        
        # CSV speichern
        output_path = output_dir / f"encoding_ffn_{split_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ {output_path.name}: {len(df):,} Zeilen, {len(df.columns)} Spalten")


# =========================
# LSTM-Encoding exportieren
# =========================

def export_lstm_encoding(lstm_inputs: dict, case_ids: np.ndarray, masks: dict, output_dir: Path):
    """
    Exportiert LSTM-Encoding als CSV für Train/Val/Test Sets.
    Alle Komponenten (Activities, Resources, Month, Trace) werden in einer Datei zusammengeführt.
    Sequenzen werden flach gemacht (jede Position wird eine Spalte).
    
    Args:
        lstm_inputs: Dictionary mit X_act, X_res, X_month, X_trace
        case_ids: Case-IDs für jeden Prefix
        masks: Dictionary mit 'train', 'val', 'test' Masken
        output_dir: Ausgabe-Verzeichnis
    """
    print("\n📝 Exportiere LSTM-Encoding...")
    
    # Komponenten-Namen für Spalten-Präfixe
    component_info = {
        "X_act": "act",
        "X_res": "res",
        "X_month": "month",
        "X_trace": "trace"
    }
    
    # Für jeden Split (Train/Val/Test)
    for split_name, mask in masks.items():
        # Basis-Daten für diesen Split
        case_ids_split = case_ids[mask]
        prefix_indices = np.where(mask)[0]
        
        # Starte mit Basis-Spalten
        data_dict = {
            "prefix_idx": prefix_indices,
            "case_id": case_ids_split
        }
        
        # Füge alle Komponenten hinzu
        for component_key, component_short in component_info.items():
            X_component = lstm_inputs[component_key]
            X_split = X_component[mask]
            
            # Sequenzen flach machen (falls 2D)
            if len(X_split.shape) == 2:
                # Sequenz: jede Position wird eine Spalte mit Präfix
                n_positions = X_split.shape[1]
                for i in range(n_positions):
                    data_dict[f"{component_short}_pos_{i}"] = X_split[:, i]
            else:
                # Bereits flach (z.B. Trace-Features)
                n_features = X_split.shape[1]
                for i in range(n_features):
                    data_dict[f"{component_short}_feature_{i}"] = X_split[:, i]
        
        # DataFrame erstellen und speichern
        df = pd.DataFrame(data_dict)
        output_path = output_dir / f"encoding_lstm_{split_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ {output_path.name}: {len(df):,} Zeilen, {len(df.columns)} Spalten")


# =========================
# Hauptfunktion
# =========================

def main():
    """
    Hauptfunktion: Lädt Encoding-Daten und exportiert sie als CSV,
    aufgeteilt nach Train/Validation/Test Sets.
    """
    parser = argparse.ArgumentParser(
        description="Exportiert Encoding-Daten als CSV, aufgeteilt nach Train/Validation/Test Sets"
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
        help="Input-Verzeichnis für Encoding-Daten. Standard: data/processed/ oder data/processed/{dataset}/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output-Verzeichnis für CSV-Dateien. Standard: {input-dir}/encoding_csv/"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    
    # Input-Verzeichnis bestimmen (rückwärtskompatibel)
    if args.input_dir is not None:
        processed_dir = Path(args.input_dir)
    elif args.dataset is not None:
        processed_dir = project_root / "data" / "processed" / args.dataset
    else:
        # Rückwärtskompatibel: Standard-Verzeichnis
        processed_dir = project_root / "data" / "processed"
    
    # Output-Verzeichnis bestimmen
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = processed_dir / "encoding_csv"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Encoding-Daten als CSV exportieren")
    print("=" * 60)
    
    # 1. Daten laden
    print("\n📂 Lade Daten...")
    X_ffn, lstm_inputs, case_ids = load_all_data(processed_dir)
    
    # 2. Splits erstellen
    print("\n🔀 Erstelle Train/Validation/Test Splits...")
    train_mask, val_mask, test_mask = create_splits(case_ids)
    
    masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask
    }
    
    # 3. FFN-Encoding exportieren
    export_ffn_encoding(X_ffn, case_ids, masks, output_dir)
    
    # 4. LSTM-Encoding exportieren
    export_lstm_encoding(lstm_inputs, case_ids, masks, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Export abgeschlossen!")
    print(f"📁 Alle CSV-Dateien gespeichert in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

