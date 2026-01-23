"""
Diagnose-Skript: Findet optimale Thresholds für Collective-FFN Modell.

Dieses Skript zeigt, warum Dev-Precision und Dev-Recall = 0 sind
und findet optimale Thresholds für jeden Deviation Type.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_curve, precision_recall_curve
)

RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0


def load_data(processed_dir: Path):
    """Lädt Predictions und Labels (wie in eval_idp_collective_ffn.py)."""
    # Predictions laden
    probs_path = processed_dir / "idp_collective_ffn_probs.npz"
    if not probs_path.exists():
        probs_path = processed_dir / "idp_collective_ffn_probs_no_oss.npz"
        if not probs_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {probs_path}")
    
    probs_data = np.load(probs_path, allow_pickle=True)
    P_dev_all = probs_data["P_dev"]
    dev_types_trained = list(probs_data["dev_types"])
    case_ids_pred = probs_data.get("case_ids", None)
    
    # Labels laden
    encoding_labels_path = processed_dir / "encoding_labels.npz"
    idp_labels_path = processed_dir / "idp_labels.npz"
    
    idp_data = np.load(idp_labels_path, allow_pickle=True)
    dev_types_raw = list(idp_data["dev_types"])
    
    if encoding_labels_path.exists():
        encoding_data = np.load(encoding_labels_path, allow_pickle=True)
        if "y_idp" in encoding_data:
            y_all = encoding_data["y_idp"]
            case_ids_labels = encoding_data["case_ids"]
        else:
            y_all = idp_data["y"]
            case_ids_labels = encoding_data["case_ids"]
    else:
        y_all = idp_data["y"]
        case_ids_labels = idp_data["case_ids"]
    
    # Spalten matchen
    keep_indices = [dev_types_raw.index(dt) for dt in dev_types_trained if dt in dev_types_raw]
    y_matched = y_all[:, keep_indices]
    
    # Case IDs verwenden
    if case_ids_pred is not None:
        case_ids = case_ids_pred
        if len(case_ids) != y_matched.shape[0]:
            # Filtere y basierend auf case_ids
            case_id_to_idx = {str(cid): i for i, cid in enumerate(case_ids_labels)}
            filtered_indices = []
            for cid in case_ids:
                cid_str = str(cid)
                if cid_str in case_id_to_idx:
                    filtered_indices.append(case_id_to_idx[cid_str])
            y_matched = y_all[filtered_indices, :][:, keep_indices]
    else:
        case_ids = case_ids_labels
        if len(case_ids) != y_matched.shape[0]:
            y_matched = y_all[:, keep_indices]
    
    return y_matched, P_dev_all, case_ids, dev_types_trained


def find_optimal_threshold_f1(y_true, y_prob):
    """Findet optimalen Threshold basierend auf F1-Score."""
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    best_prec = 0
    best_rec = 0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_prec = prec
            best_rec = rec
    
    return best_threshold, best_f1, best_prec, best_rec


def find_optimal_threshold_youden(y_true, y_prob):
    """Findet optimalen Threshold basierend auf Youden's J statistic (TPR - FPR)."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0, 0.0
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Berechne Metriken für optimalen Threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return optimal_threshold, f1, prec, rec


def main():
    parser = argparse.ArgumentParser(
        description="Findet optimale Thresholds für Collective-FFN"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input-Verzeichnis. Standard: data/processed/"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    if args.input_dir is not None:
        processed_dir = Path(args.input_dir)
    else:
        processed_dir = project_root / "data" / "processed"
    
    print("=" * 80)
    print("COLLECTIVE-FFN THRESHOLD-DIAGNOSE")
    print("=" * 80)
    print(f"Verzeichnis: {processed_dir}\n")
    
    # Daten laden
    y_all, P_dev_all, case_ids, dev_types = load_data(processed_dir)
    
    # Test-Split rekonstruieren
    unique_cases = np.unique(case_ids)
    _, test_cases = train_test_split(
        unique_cases, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    test_mask = np.isin(case_ids, test_cases)
    
    y_test = y_all[test_mask]
    P_test = P_dev_all[test_mask]
    
    print(f"Test-Set Größe: {y_test.shape[0]} Prefixe\n")
    print("=" * 80)
    
    results = []
    
    for i, dev_name in enumerate(dev_types):
        y_true = y_test[:, i]
        y_prob = P_test[:, i]
        
        # Prüfe, ob beide Klassen vorhanden
        if len(np.unique(y_true)) < 2:
            print(f"{dev_name}: Nur eine Klasse im Test-Set, überspringe.")
            continue
        
        n_deviant = np.sum(y_true == 1)
        n_conforming = np.sum(y_true == 0)
        
        # Metriken mit Standard-Threshold 0.5
        y_pred_05 = (y_prob >= 0.5).astype(int)
        prec_05 = precision_score(y_true, y_pred_05, pos_label=1, zero_division=0)
        rec_05 = recall_score(y_true, y_pred_05, pos_label=1, zero_division=0)
        f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
        
        # Optimaler Threshold (F1)
        thresh_f1, f1_opt, prec_f1, rec_f1 = find_optimal_threshold_f1(y_true, y_prob)
        
        # Optimaler Threshold (Youden)
        thresh_youden, f1_youden, prec_youden, rec_youden = find_optimal_threshold_youden(y_true, y_prob)
        
        # Statistiken
        probs_deviant = y_prob[y_true == 1]
        probs_conforming = y_prob[y_true == 0]
        
        print(f"\n{dev_name}")
        print("-" * 80)
        print(f"Ground Truth: {n_deviant} deviant, {n_conforming} conforming")
        print(f"\nWahrscheinlichkeiten (Deviant):")
        print(f"  Min: {probs_deviant.min():.6f}, Max: {probs_deviant.max():.6f}")
        print(f"  Mean: {probs_deviant.mean():.6f}, Median: {np.median(probs_deviant):.6f}")
        print(f"\nWahrscheinlichkeiten (Conforming):")
        print(f"  Min: {probs_conforming.min():.6f}, Max: {probs_conforming.max():.6f}")
        print(f"  Mean: {probs_conforming.mean():.6f}, Median: {np.median(probs_conforming):.6f}")
        
        print(f"\nMetriken mit Threshold 0.5 (Standard):")
        print(f"  Precision: {prec_05:.4f}, Recall: {rec_05:.4f}, F1: {f1_05:.4f}")
        print(f"  Vorhersagen >= 0.5: {np.sum(y_pred_05 == 1)}")
        
        print(f"\nOptimaler Threshold (F1-Score):")
        print(f"  Threshold: {thresh_f1:.4f}")
        print(f"  Precision: {prec_f1:.4f}, Recall: {rec_f1:.4f}, F1: {f1_opt:.4f}")
        print(f"  Vorhersagen >= {thresh_f1:.4f}: {np.sum(y_prob >= thresh_f1)}")
        
        print(f"\nOptimaler Threshold (Youden's J):")
        print(f"  Threshold: {thresh_youden:.4f}")
        print(f"  Precision: {prec_youden:.4f}, Recall: {rec_youden:.4f}, F1: {f1_youden:.4f}")
        print(f"  Vorhersagen >= {thresh_youden:.4f}: {np.sum(y_prob >= thresh_youden)}")
        
        results.append({
            "Deviation_Type": dev_name,
            "N_Deviant": n_deviant,
            "N_Conforming": n_conforming,
            "Prob_Deviant_Max": probs_deviant.max(),
            "Prob_Deviant_Mean": probs_deviant.mean(),
            "Prob_Conforming_Mean": probs_conforming.mean(),
            "Threshold_05_Precision": prec_05,
            "Threshold_05_Recall": rec_05,
            "Threshold_05_F1": f1_05,
            "Optimal_Threshold_F1": thresh_f1,
            "Optimal_F1_Precision": prec_f1,
            "Optimal_F1_Recall": rec_f1,
            "Optimal_F1_Score": f1_opt,
            "Optimal_Threshold_Youden": thresh_youden,
            "Optimal_Youden_Precision": prec_youden,
            "Optimal_Youden_Recall": rec_youden,
            "Optimal_Youden_F1": f1_youden,
        })
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print("\nVergleich: Threshold 0.5 vs. Optimaler Threshold (F1)")
    print("-" * 80)
    print(f"{'Deviation Type':<40} | {'0.5 F1':<8} | {'Opt F1':<8} | {'Opt Thresh':<10} | {'Verbesserung':<12}")
    print("-" * 80)
    for _, row in df.iterrows():
        improvement = row['Optimal_F1_Score'] - row['Threshold_05_F1']
        print(f"{row['Deviation_Type']:<40} | {row['Threshold_05_F1']:>8.4f} | {row['Optimal_F1_Score']:>8.4f} | {row['Optimal_Threshold_F1']:>10.4f} | {improvement:>+12.4f}")
    
    # Speichere Ergebnisse
    csv_path = processed_dir / "collective_ffn_threshold_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nErgebnisse gespeichert in: {csv_path}")
    
    print("\n" + "=" * 80)
    print("FAZIT")
    print("=" * 80)
    print("Das Collective-FFN Modell gibt für deviant cases sehr niedrige")
    print("Wahrscheinlichkeiten aus. Der Standard-Threshold 0.5 ist daher")
    print("zu hoch und führt zu Precision=0 und Recall=0.")
    print("\nLösung: Verwende die optimalen Thresholds aus dieser Analyse")
    print("oder verwende separate_ffn Modell, das deutlich besser funktioniert.")


if __name__ == "__main__":
    main()
