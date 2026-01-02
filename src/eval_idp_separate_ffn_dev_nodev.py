from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import precision_score, recall_score, roc_auc_score
# WICHTIG: train_test_split nutzen, exakt wie im Training!
from sklearn.model_selection import train_test_split 

# =========================
# Konfiguration
# =========================

# Exakt wie im Training definiert!
RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0  # Paper: "split ... into train (2/3) and test (1/3)"


# =========================
# Daten laden
# =========================

def load_data(processed_dir: Path):
    """
    Lädt die Ground Truth (Labels) und die Vorhersagen (Probabilities).
    """
    # 1. True Labels laden
    labels_path = processed_dir / "idp_labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {labels_path}")
    
    labels_data = np.load(labels_path, allow_pickle=True)
    y_true_all = labels_data["y"]               # (N, m)
    case_ids = labels_data["case_ids"]          # (N,)
    dev_types = list(labels_data["dev_types"])  

    # 2. Predictions laden
    probs_path = processed_dir / "idp_separate_ffn_probs.npz"
    if not probs_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {probs_path}. Bitte erst trainieren.")
    
    probs_data = np.load(probs_path, allow_pickle=True)
    P_dev_all = probs_data["P_dev"]             # (N, m) - Wahrscheinlichkeiten
    
    # case_ids aus Predictions verwenden (falls vorhanden), sonst aus Labels
    # Dies stellt sicher, dass die Reihenfolge mit den Predictions übereinstimmt
    if "case_ids" in probs_data:
        case_ids_probs = probs_data["case_ids"]
        # Sicherstellen, dass case_ids übereinstimmen
        assert np.array_equal(case_ids, case_ids_probs), "case_ids stimmen nicht überein zwischen Labels und Predictions!"
        case_ids = case_ids_probs

    # Sanity Check Shapes
    assert y_true_all.shape == P_dev_all.shape, "Shape Mismatch zwischen Labels und Predictions!"
    assert len(case_ids) == y_true_all.shape[0], "Anzahl case_ids stimmt nicht mit Anzahl Samples überein!"
    
    return y_true_all, P_dev_all, case_ids, dev_types


# =========================
# Split-Logik (KORRIGIERT)
# =========================

def get_test_mask_per_type(case_ids: np.ndarray, random_state: int = RANDOM_STATE) -> np.ndarray:
    """
    Rekonstruiert exakt den Test-Split aus dem Training.
    
    WICHTIG: Im Training wird für jeden Deviation-Typ ein separater Split durchgeführt.
    Daher müssen wir hier den gleichen Split rekonstruieren.
    
    Nutzung von train_test_split auf unique cases, exakt wie im Training.
    """
    # 1. Alle Case-IDs einmalig holen
    unique_cases = np.unique(case_ids)
    
    # 2. Split durchführen (identischer Random State wie im Training!)
    # Im Training: train_cases, test_cases = train_test_split(unique_cases, test_size=1.0/3.0, random_state=cfg.random_state)
    _, test_cases = train_test_split(
        unique_cases, 
        test_size=TEST_SIZE, 
        random_state=random_state
    )
    
    # 3. Maske erstellen: Welche Zeilen gehören zu Test-Cases?
    test_mask = np.isin(case_ids, test_cases)
    
    return test_mask


# =========================
# Metrik-Berechnung
# =========================

def evaluate_single_type(
    dev_name: str,
    y_true: np.ndarray, 
    y_prob: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Berechnet die Metriken gemäß Paper Section 5.1.
    """
    # Safety Check: Enthält das Test-Set überhaupt beide Klassen?
    # ROC AUC stürzt ab, wenn y_true nur Nullen enthält.
    if len(np.unique(y_true)) < 2:
        return None

    # Binary Predictions via Threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Metrics (Paper Eq. 1 & Section 5.1) ---
    # "calculate precision and recall for the deviating (Dev) and no-deviating class (No Dev) separately"
    
    res = {}
    
    # AUC ROC ("consider AUC_ROC... for imbalanced data")
    try:
        res["AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Fallback falls AUC nicht berechenbar (z.B. nur 1 Klasse im y_true)
        res["AUC"] = 0.5

    # Deviating Class (Pos Label = 1)
    res["Rec_Dev"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    res["Prec_Dev"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # No-Deviating Class (Pos Label = 0)
    res["Rec_NoDev"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    res["Prec_NoDev"] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)

    return res


# =========================
# Hauptfunktion
# =========================

def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    print(f"Lade Daten aus: {processed_dir}")
    try:
        y_true_all, P_dev_all, case_ids, dev_types = load_data(processed_dir)
    except FileNotFoundError as e:
        print(f"Fehler: {e}")
        return

    # 1. Test-Set isolieren (Split ist für alle Deviation-Typen identisch)
    # Im Training wird für jeden Typ ein Split gemacht, aber da die case_ids gleich sind,
    # ist der Split für alle Typen identisch (gleicher random_state, gleiche unique_cases)
    print("Rekonstruiere Test-Split (train_test_split auf Unique Cases)...")
    test_mask = get_test_mask_per_type(case_ids, random_state=RANDOM_STATE)
    
    y_test = y_true_all[test_mask]
    P_test = P_dev_all[test_mask]
    
    n_test_traces = len(np.unique(case_ids[test_mask]))
    print(f"Test-Set: {np.sum(test_mask)} Prefixe aus {n_test_traces} Traces.\n")

    # 2. Evaluierung pro Typ
    results = []
    skipped_types = []

    print("="*110)
    print(f"{'Deviation Type':<40} | {'AUC':<6} | {'Rec(Dev)':<9} | {'Pre(Dev)':<9} | {'Rec(NoDev)':<9} | {'Pre(NoDev)':<9}")
    print("-" * 110)

    for i, dev_name in enumerate(dev_types):
        metrics = evaluate_single_type(dev_name, y_test[:, i], P_test[:, i])
        
        if metrics is None:
            skipped_types.append(dev_name)
            continue
            
        metrics["dev_type"] = dev_name
        results.append(metrics)
        
        print(f"{dev_name:<40} | "
              f"{metrics['AUC']:.4f} | "
              f"{metrics['Rec_Dev']:.4f}    | "
              f"{metrics['Prec_Dev']:.4f}    | "
              f"{metrics['Rec_NoDev']:.4f}    | "
              f"{metrics['Prec_NoDev']:.4f}")

    if skipped_types:
        print(f"\n[INFO] {len(skipped_types)} Typen übersprungen (nur 1 Klasse im Test-Set).")

    # 3. Macro Average berechnen
    # Paper: "report the macro average (i.e., unweighted average over all deviation types)"
    if results:
        df_res = pd.DataFrame(results)
        # Wir mitteln über die numerischen Spalten
        macro_avg = df_res.drop(columns=["dev_type"]).mean()

        print("\n" + "="*60)
        print("FINAL RESULT: MACRO AVERAGE (Vergleichbar mit Table 5)")
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