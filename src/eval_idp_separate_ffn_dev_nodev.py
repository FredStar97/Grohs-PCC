from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# =========================
# Konfiguration
# =========================

# Exakt wie im Training definiert, um denselben Split zu erhalten
RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0  # Paper: "split ... into train (2/3) and test (1/3)" 


# =========================
# Daten laden
# =========================

def load_data(processed_dir: Path):
    """
    Lädt die Ground Truth (Labels) und die Vorhersagen (Probabilities).
    """
    # 1. True Labels laden (aus dem Labeling-Schritt)
    labels_path = processed_dir / "idp_labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {labels_path}")
    
    labels_data = np.load(labels_path, allow_pickle=True)
    y_true_all = labels_data["y"]               # Shape: (N_prefixes, m_types)
    case_ids = labels_data["case_ids"]          # Shape: (N_prefixes,)
    dev_types = list(labels_data["dev_types"])  # Liste der Namen der Deviation-Types

    # 2. Predictions laden (aus dem Training-Schritt)
    probs_path = processed_dir / "idp_separate_ffn_probs.npz"
    if not probs_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {probs_path}. Bitte erst trainieren.")
    
    probs_data = np.load(probs_path, allow_pickle=True)
    # P_dev enthält die Wahrscheinlichkeit für Klasse 1 (Deviating)
    P_dev_all = probs_data["P_dev"]             # Shape: (N_prefixes, m_types)

    # Sanity Check
    assert y_true_all.shape == P_dev_all.shape, "Shape Mismatch zwischen Labels und Predictions!"
    
    return y_true_all, P_dev_all, case_ids, dev_types


# =========================
# Split-Logik
# =========================

def get_test_indices(n_samples: int, case_ids: np.ndarray) -> np.ndarray:
    """
    Rekonstruiert exakt den Test-Split, der im Training verwendet wurde.
    Das Paper fordert Evaluation auf dem Test-Set[cite: 432].
    """
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    # Dummy-Index-Array erzeugen
    idx = np.arange(n_samples)
    
    # Split basierend auf Gruppen (Case IDs) durchführen
    # next() liefert (train_idx, test_idx)
    _, test_idx = next(gss.split(idx, groups=case_ids))
    
    return test_idx


# =========================
# Metrik-Berechnung
# =========================

def evaluate_single_type(
    y_true: np.ndarray, 
    y_prob: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Berechnet die Metriken für einen einzelnen Abweichungstypen.
    
    y_true: Binäre Labels (0/1) für diesen Typ im Test-Set
    y_prob: Wahrscheinlichkeit P(y=1) für diesen Typ im Test-Set
    """
    # Prüfen, ob im Test-Set überhaupt beide Klassen vorkommen.
    # ROC AUC ist nicht definiert, wenn nur eine Klasse vorhanden ist.
    if len(np.unique(y_true)) < 2:
        return None

    # Binary Predictions via Threshold 0.5 (Standard für Softmax/Sigmoid Output)
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Precision & Recall (Eq. 1 im Paper) ---
    # Paper: "calculate precision and recall for the deviating (Dev) and 
    # no-deviating class (No Dev) separately" 
    
    # Klasse 1: Deviating
    prec_dev = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_dev = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Klasse 0: No-Deviating (Conforming)
    prec_nodev = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_nodev = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    # --- AUC_ROC ---
    # Paper: "consider AUC_ROC which is commonly used in tasks with imbalanced target labels" 
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

    print(f"Lade Daten aus: {processed_dir}")
    y_true_all, P_dev_all, case_ids, dev_types = load_data(processed_dir)

    # 1. Test-Set isolieren
    print("Rekonstruiere Test-Split (GroupShuffleSplit)...")
    test_idx = get_test_indices(len(case_ids), case_ids)
    
    y_test = y_true_all[test_idx]
    P_test = P_dev_all[test_idx]
    
    print(f"Anzahl Test-Prefixe: {len(test_idx)} (von {len(case_ids)} gesamt)")

    # 2. Evaluierung pro Typ
    results = []
    skipped_types = []

    print("\n" + "="*100)
    print(f"{'Deviation Type':<40} | {'AUC':<6} | {'Rec(Dev)':<9} | {'Pre(Dev)':<9} | {'Rec(NoDev)':<9} | {'Pre(NoDev)':<9}")
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

    # 3. Macro Average berechnen
    # Paper: "report the macro average (i.e., unweighted average over all deviation types)" 
    if results:
        df_res = pd.DataFrame(results)
        macro_avg = df_res.mean()

        print("\n" + "="*60)
        print("FINAL RESULT: MACRO AVERAGE (Paper Table 5 Benchmark)")
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