from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import precision_score, recall_score, roc_auc_score

# =========================
# Konfiguration
# =========================

RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0  # Paper: "split ... into train (2/3) and test (1/3)"


# =========================
# Daten laden (MIT FIX FÜR GEFILTERTE TYPEN)
# =========================

def load_data(processed_dir: Path):
    """
    Lädt die Ground Truth (Labels) und die LSTM-Vorhersagen (Probabilities).
    Gleicht die Deviation Types ab, falls im Training gefiltert wurde.
    """
    # 1. Raw Ground Truth Labels laden
    labels_path = processed_dir / "idp_labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {labels_path}")
    
    labels_data = np.load(labels_path, allow_pickle=True)
    y_raw = labels_data["y"]                    # (N, m_raw)
    case_ids = labels_data["case_ids"]          # (N,)
    dev_types_raw = list(labels_data["dev_types"])

    # 2. LSTM Predictions laden
    probs_path = processed_dir / "idp_separate_lstm_probs.npz"
    if not probs_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {probs_path}. Bitte erst LSTM trainieren.")
    
    probs_data = np.load(probs_path, allow_pickle=True)
    P_dev_all = probs_data["P_dev"]             # (N, m_filtered)
    
    # Hier stehen die Typen drin, die tatsächlich trainiert wurden (nach Filterung)
    dev_types_trained = list(probs_data["dev_types"]) 

    # 3. Ground Truth anpassen (Matching)
    # Wir müssen aus y_raw nur die Spalten holen, die auch in dev_types_trained sind
    keep_indices = []
    
    # Mapping erstellen: Name -> Index in Raw Data
    raw_type_to_idx = {name: i for i, name in enumerate(dev_types_raw)}
    
    missing_types = []
    for dt in dev_types_trained:
        if dt in raw_type_to_idx:
            keep_indices.append(raw_type_to_idx[dt])
        else:
            missing_types.append(dt)
            
    if missing_types:
        raise ValueError(f"Trainierte Typen nicht in Raw Labels gefunden: {missing_types}")
        
    # Filtern der Ground Truth Matrix
    y_true_matched = y_raw[:, keep_indices]

    # Sanity Check
    if y_true_matched.shape != P_dev_all.shape:
        print(f"Shape Mismatch Debug:")
        print(f"  y_raw shape:      {y_raw.shape}")
        print(f"  y_matched shape:  {y_true_matched.shape}")
        print(f"  P_dev shape:      {P_dev_all.shape}")
        print(f"  Trained Types:    {len(dev_types_trained)}")
        print(f"  Matched Types:    {len(keep_indices)}")
        raise AssertionError("Shape Mismatch zwischen Labels und Predictions!")
    
    return y_true_matched, P_dev_all, case_ids, dev_types_trained


# =========================
# Split-Logik
# =========================

from sklearn.model_selection import train_test_split

def get_test_indices(n_samples: int, case_ids: np.ndarray) -> np.ndarray:
    """
    Rekonstruiert exakt den Test-Split basierend auf Unique Case IDs.
    Muss 1:1 der Logik im Trainings-Skript entsprechen.
    """
    unique_cases = np.unique(case_ids)
    
    # Im Training: train_cases, _ = train_test_split(...)
    # Hier brauchen wir den Test-Teil:
    _, test_cases = train_test_split(
        unique_cases, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Erstelle Maske für alle Prefixe, die zu den Test-Cases gehören
    test_mask = np.isin(case_ids, test_cases)
    
    # Rückgabe der Indizes, wo die Maske True ist
    return np.where(test_mask)[0]
# =========================
# Metrik-Berechnung
# =========================

def evaluate_single_type(
    y_true: np.ndarray, 
    y_prob: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Berechnet AUC, Precision & Recall (Dev/NoDev) für einen Typen.
    """
    # AUC benötigt beide Klassen im Test-Set
    if len(np.unique(y_true)) < 2:
        return None

    # Binary Predictions (Threshold 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Precision & Recall (Dev / Class 1) ---
    prec_dev = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_dev = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # --- Precision & Recall (No Dev / Class 0) ---
    prec_nodev = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_nodev = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    # --- AUC_ROC ---
    auc = roc_auc_score(y_true, y_prob)

    return {
        "AUC": auc,
        "Prec_Dev": prec_dev,
        "Rec_Dev": rec_dev,
        "Prec_NoDev": prec_nodev,
        "Rec_NoDev": rec_nodev
    }


# =========================
# Hauptfunktion
# =========================

def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    print(f"Lade LSTM-Ergebnisse aus: {processed_dir}")
    # Hier wird jetzt automatisch gefiltert
    y_true_all, P_dev_all, case_ids, dev_types = load_data(processed_dir)

    # 1. Test-Set isolieren
    print("Rekonstruiere Test-Split...")
    test_idx = get_test_indices(len(case_ids), case_ids)
    
    y_test = y_true_all[test_idx]
    P_test = P_dev_all[test_idx]
    
    print(f"Anzahl Test-Prefixe: {len(test_idx)} (von {len(case_ids)} gesamt)")

    # 2. Evaluierung pro Typ
    results = []
    skipped_types = []

    print("\n" + "="*100)
    print(f"{'Deviation Type (LSTM)':<40} | {'AUC':<6} | {'Rec(Dev)':<9} | {'Pre(Dev)':<9} | {'Rec(NoDev)':<9} | {'Pre(NoDev)':<9}")
    print("-" * 100)

    for i, dev_name in enumerate(dev_types):
        y_t_single = y_test[:, i]
        y_p_single = P_test[:, i]
        
        metrics = evaluate_single_type(y_t_single, y_p_single)
        
        if metrics is None:
            skipped_types.append(dev_name)
            continue
            
        results.append(metrics)
        
        print(f"{dev_name:<40} | "
              f"{metrics['AUC']:.4f} | "
              f"{metrics['Rec_Dev']:.4f}    | "
              f"{metrics['Prec_Dev']:.4f}    | "
              f"{metrics['Rec_NoDev']:.4f}    | "
              f"{metrics['Prec_NoDev']:.4f}")

    if skipped_types:
        print(f"\n[INFO] Übersprungene Typen (nur 1 Klasse im Test-Set): {skipped_types}")

    # 3. Macro Average
    if results:
        df_res = pd.DataFrame(results)
        macro_avg = df_res.mean()

        print("\n" + "="*60)
        print("FINAL RESULT: LSTM MACRO AVERAGE (Vergleichbar mit Table 5)")
        print("="*60)
        print(f"AUC_ROC:            {macro_avg['AUC']:.4f}")
        print("-" * 30)
        print("Deviating (Class 1):")
        print(f"  Recall:           {macro_avg['Rec_Dev']:.4f}")
        print(f"  Precision:        {macro_avg['Prec_Dev']:.4f}")
        print("-" * 30)
        print("No-Deviating (Class 0):")
        print(f"  Recall:           {macro_avg['Rec_NoDev']:.4f}")
        print(f"  Precision:        {macro_avg['Prec_NoDev']:.4f}")
        print("="*60)
    else:
        print("\nKeine auswertbaren Deviation Types gefunden.")

if __name__ == "__main__":
    main()