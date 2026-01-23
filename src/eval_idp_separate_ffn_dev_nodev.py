from __future__ import annotations

import argparse
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
    Unterstützt gefilterte Labels aus encoding_labels.npz (nach Prefix-Filtering).
    """
    # 1. Predictions laden (prüfe beide Varianten: mit und ohne OSS)
    probs_path = processed_dir / "idp_separate_ffn_probs.npz"
    if not probs_path.exists():
        # Fallback: versuche _no_oss Variante
        probs_path = processed_dir / "idp_separate_ffn_probs_no_oss.npz"
        if not probs_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {processed_dir / 'idp_separate_ffn_probs.npz'} oder {probs_path}. Bitte erst Separate FFN trainieren.")
    
    probs_data = np.load(probs_path, allow_pickle=True)
    P_dev_all = probs_data["P_dev"]             # (N, m) - Wahrscheinlichkeiten
    case_ids_pred = probs_data.get("case_ids", None)  # Case-IDs aus Predictions (falls vorhanden)
    n_samples_pred = P_dev_all.shape[0]

    # 2. Labels laden (unterstützt gefilterte Labels)
    encoding_labels_path = processed_dir / "encoding_labels.npz"
    idp_labels_path = processed_dir / "idp_labels.npz"
    
    if not idp_labels_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {idp_labels_path}")
    
    # Lade Original-Labels (für dev_types)
    idp_data = np.load(idp_labels_path, allow_pickle=True)
    dev_types = list(idp_data["dev_types"])
    
    # Prüfe, ob gefilterte Labels existieren
    if encoding_labels_path.exists():
        # Verwende gefilterte Labels (nach Prefix-Filtering)
        encoding_data = np.load(encoding_labels_path, allow_pickle=True)
        case_ids_filtered = encoding_data["case_ids"]
        
        # Prüfe, ob y_idp in encoding_labels.npz vorhanden ist
        if "y_idp" in encoding_data:
            # Verwende gefilterte Labels direkt
            y_filtered = encoding_data["y_idp"]
            print(f"✓ Verwende gefilterte Labels (y_idp) aus encoding_labels.npz: {len(case_ids_filtered)} Prefixe")
        else:
            # y_idp nicht vorhanden: Filtere y aus idp_labels.npz basierend auf (case_id, prefix_length) Paaren
            y_all = idp_data["y"]
            case_ids_all = idp_data["case_ids"]
            prefix_lengths_all = idp_data.get("prefix_lengths", None)
            prefix_lengths_filtered = encoding_data.get("prefix_lengths", None)
            
            # Verwende (case_id, prefix_length) Paare für exakte Zuordnung
            if prefix_lengths_all is not None and prefix_lengths_filtered is not None:
                # Erstelle Mapping: (case_id, prefix_length) -> Index
                pair_to_index = {}
                for i, (cid, plen) in enumerate(zip(case_ids_all, prefix_lengths_all)):
                    pair = (str(cid), int(plen))
                    if pair not in pair_to_index:
                        pair_to_index[pair] = []
                    pair_to_index[pair].append(i)
                
                # Finde Indizes für gefilterte Paare
                filtered_indices = []
                used_indices = set()
                for cid, plen in zip(case_ids_filtered, prefix_lengths_filtered):
                    pair = (str(cid), int(plen))
                    if pair in pair_to_index:
                        for idx in pair_to_index[pair]:
                            if idx not in used_indices:
                                filtered_indices.append(idx)
                                used_indices.add(idx)
                                break
                
                if len(filtered_indices) == len(case_ids_filtered):
                    y_filtered = y_all[filtered_indices]
                    print(f"✓ Gefilterte Labels (via (case_id, prefix_length) Mapping): {len(case_ids_filtered)} Prefixe")
                else:
                    raise ValueError(
                        f"Konnte gefilterte Labels nicht korrekt mappen: "
                        f"{len(filtered_indices)} Indizes gefunden, aber {len(case_ids_filtered)} erwartet."
                    )
            else:
                # Fallback: Verwende einfaches case_id Mapping (ohne prefix_length)
                case_id_to_indices = {}
                for i, cid in enumerate(case_ids_all):
                    cid_str = str(cid)
                    if cid_str not in case_id_to_indices:
                        case_id_to_indices[cid_str] = []
                    case_id_to_indices[cid_str].append(i)
                
                # Für jedes gefilterte case_id, nimm den ersten passenden Index
                filtered_indices = []
                used_indices = set()
                for cid in case_ids_filtered:
                    cid_str = str(cid)
                    if cid_str in case_id_to_indices:
                        for idx in case_id_to_indices[cid_str]:
                            if idx not in used_indices:
                                filtered_indices.append(idx)
                                used_indices.add(idx)
                                break
                
                if len(filtered_indices) == len(case_ids_filtered):
                    y_filtered = y_all[filtered_indices]
                    print(f"✓ Gefilterte Labels (via case_id Mapping): {len(case_ids_filtered)} Prefixe")
                else:
                    raise ValueError(
                        f"Konnte gefilterte Labels nicht korrekt mappen: "
                        f"{len(filtered_indices)} Indizes gefunden, aber {len(case_ids_filtered)} erwartet. "
                        f"Bitte stelle sicher, dass prefix_lengths in beiden Dateien vorhanden sind."
                    )
        
        # Verwende case_ids aus Predictions (falls vorhanden), sonst aus encoding_labels.npz
        if case_ids_pred is not None:
            case_ids = case_ids_pred
        else:
            case_ids = case_ids_filtered
        y_true_all = y_filtered
    else:
        # Keine gefilterten Labels: Verwende Original-Labels
        # Falls case_ids in Predictions vorhanden, verwende diese für Filtering
        if case_ids_pred is not None:
            # Filtere Labels basierend auf case_ids aus Predictions
            case_ids_all = idp_data["case_ids"]
            y_all = idp_data["y"]
            
            # Erstelle Mapping: case_id -> Indizes
            case_id_to_indices = {}
            for i, cid in enumerate(case_ids_all):
                cid_str = str(cid)
                if cid_str not in case_id_to_indices:
                    case_id_to_indices[cid_str] = []
                case_id_to_indices[cid_str].append(i)
            
            # Finde Indizes für case_ids aus Predictions
            filtered_indices = []
            used_indices = set()
            for cid in case_ids_pred:
                cid_str = str(cid)
                if cid_str in case_id_to_indices:
                    for idx in case_id_to_indices[cid_str]:
                        if idx not in used_indices:
                            filtered_indices.append(idx)
                            used_indices.add(idx)
                            break
            
            if len(filtered_indices) == n_samples_pred:
                y_true_all = y_all[filtered_indices]
                case_ids = case_ids_pred
                print(f"✓ Labels gefiltert basierend auf case_ids aus Predictions: {n_samples_pred} Prefixe")
            else:
                # Fallback: Verwende alle Original-Labels
                y_true_all = y_all
                case_ids = case_ids_all
                print(f"✓ Verwende Original-Labels aus idp_labels.npz: {len(case_ids)} Prefixe")
        else:
            # Keine case_ids in Predictions: Verwende alle Original-Labels
            y_true_all = idp_data["y"]
            case_ids = idp_data["case_ids"]
            print(f"✓ Verwende Original-Labels aus idp_labels.npz: {len(case_ids)} Prefixe")

    # Sanity Check Shapes
    if y_true_all.shape[0] != P_dev_all.shape[0]:
        raise AssertionError(
            f"Shape Mismatch! Labels: {y_true_all.shape}, Preds: {P_dev_all.shape}. "
            f"Bitte stelle sicher, dass encoding_labels.npz mit y_idp existiert oder "
            f"dass die Labels korrekt gefiltert wurden."
        )
    if len(case_ids) != P_dev_all.shape[0]:
        raise AssertionError(
            f"Shape Mismatch! case_ids: {len(case_ids)}, Preds: {P_dev_all.shape[0]}. "
            f"Bitte stelle sicher, dass case_ids in idp_separate_ffn_probs.npz gespeichert wurden."
        )
    
    # Prüfe, ob Anzahl der Deviation Types übereinstimmt
    if y_true_all.shape[1] != P_dev_all.shape[1]:
        raise AssertionError(
            f"Anzahl Deviation Types stimmt nicht überein! Labels: {y_true_all.shape[1]}, Preds: {P_dev_all.shape[1]}"
        )
    
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
    parser = argparse.ArgumentParser(
        description="Evaluiert das Separate FFN-Modell (Dev vs NoDev)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name des Datensatzes (für Input-Verzeichnis). Standard: processed/ (rückwärtskompatibel)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input-Verzeichnis für Labels und Predictions. Standard: data/processed/ oder data/processed/{dataset}/"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    
    # Verzeichnis bestimmen (rückwärtskompatibel)
    if args.input_dir is not None:
        processed_dir = Path(args.input_dir)
    elif args.dataset is not None:
        processed_dir = project_root / "data" / "processed" / args.dataset
    else:
        # Rückwärtskompatibel: Standard-Verzeichnis
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