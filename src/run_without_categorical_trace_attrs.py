from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


# =========================
# Evaluierungs-Funktionen
# =========================

RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0


def get_test_mask(case_ids: np.ndarray) -> np.ndarray:
    """Rekonstruiert den Test-Split basierend auf Trace-IDs."""
    unique_cases = np.unique(case_ids)
    _, test_cases = train_test_split(
        unique_cases, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    test_mask = np.isin(case_ids, test_cases)
    return test_mask


def evaluate_single_type(
    y_true: np.ndarray, 
    y_prob: np.ndarray
) -> Optional[Dict[str, float]]:
    """Berechnet Metriken für einen einzelnen Deviation-Typ."""
    if len(np.unique(y_true)) < 2:
        return None

    y_pred = (y_prob >= 0.5).astype(int)

    prec_dev = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_dev = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec_nodev = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_nodev = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    return {
        "AUC": auc,
        "Prec_Dev": prec_dev,
        "Rec_Dev": rec_dev,
        "Prec_NoDev": prec_nodev,
        "Rec_NoDev": rec_nodev
    }


def load_data_for_evaluation(processed_dir: Path, probs_file: str):
    """
    Lädt Ground Truth und Predictions für Evaluierung.
    Unterstützt gefilterte Labels aus encoding_labels.npz.
    """
    # 1. Predictions laden
    probs_path = processed_dir / probs_file
    if not probs_path.exists():
        # Fallback: versuche _no_oss Variante
        probs_path_no_oss = processed_dir / probs_file.replace(".npz", "_no_oss.npz")
        if probs_path_no_oss.exists():
            probs_path = probs_path_no_oss
        else:
            raise FileNotFoundError(f"Datei nicht gefunden: {probs_path} oder {probs_path_no_oss}")
    
    probs_data = np.load(probs_path, allow_pickle=True)
    P_dev_all = probs_data["P_dev"]
    dev_types_trained = list(probs_data["dev_types"])
    case_ids_pred = probs_data.get("case_ids", None)
    n_samples_pred = P_dev_all.shape[0]

    # 2. Labels laden (unterstützt gefilterte Labels)
    encoding_labels_path = processed_dir / "encoding_labels.npz"
    idp_labels_path = processed_dir / "idp_labels.npz"
    
    if not idp_labels_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {idp_labels_path}")
    
    idp_data = np.load(idp_labels_path, allow_pickle=True)
    dev_types_raw = list(idp_data["dev_types"])
    
    # Prüfe, ob gefilterte Labels existieren
    if encoding_labels_path.exists():
        encoding_data = np.load(encoding_labels_path, allow_pickle=True)
        case_ids_filtered = encoding_data["case_ids"]
        
        if "y_idp" in encoding_data:
            y_filtered = encoding_data["y_idp"]
            case_ids = case_ids_filtered
        else:
            # Filtere y basierend auf (case_id, prefix_length) Paaren
            y_all = idp_data["y"]
            case_ids_all = idp_data["case_ids"]
            prefix_lengths_all = idp_data.get("prefix_lengths", None)
            prefix_lengths_filtered = encoding_data.get("prefix_lengths", None)
            
            if prefix_lengths_all is not None and prefix_lengths_filtered is not None:
                pair_to_index = {}
                for i, (cid, plen) in enumerate(zip(case_ids_all, prefix_lengths_all)):
                    pair = (str(cid), int(plen))
                    if pair not in pair_to_index:
                        pair_to_index[pair] = []
                    pair_to_index[pair].append(i)
                
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
                    case_ids = case_ids_filtered
                else:
                    raise ValueError(f"Konnte gefilterte Labels nicht korrekt mappen")
            else:
                raise ValueError("prefix_lengths nicht in beiden Dateien vorhanden")
    else:
        if case_ids_pred is not None:
            case_ids_all = idp_data["case_ids"]
            y_all = idp_data["y"]
            
            case_id_to_indices = {}
            for i, cid in enumerate(case_ids_all):
                cid_str = str(cid)
                if cid_str not in case_id_to_indices:
                    case_id_to_indices[cid_str] = []
                case_id_to_indices[cid_str].append(i)
            
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
                y_filtered = y_all[filtered_indices]
                case_ids = case_ids_pred
            else:
                y_filtered = y_all
                case_ids = case_ids_all
        else:
            y_filtered = idp_data["y"]
            case_ids = idp_data["case_ids"]

    # 3. Ground Truth filtern (Spalten-Matching)
    keep_indices = []
    raw_type_to_idx = {name: i for i, name in enumerate(dev_types_raw)}
    
    for dt in dev_types_trained:
        if dt in raw_type_to_idx:
            keep_indices.append(raw_type_to_idx[dt])
    
    y_true_matched = y_filtered[:, keep_indices]

    if y_true_matched.shape[0] != P_dev_all.shape[0]:
        raise AssertionError(f"Shape Mismatch! Labels: {y_true_matched.shape}, Preds: {P_dev_all.shape}")
    if len(case_ids) != P_dev_all.shape[0]:
        raise AssertionError(f"Shape Mismatch! case_ids: {len(case_ids)}, Preds: {P_dev_all.shape[0]}")

    return y_true_matched, P_dev_all, case_ids, dev_types_trained


def evaluate_model(processed_dir: Path, probs_file: str, model_name: str) -> Optional[Dict[str, float]]:
    """Evaluiert ein Modell und gibt Macro-Average Metriken zurück."""
    try:
        y_true_all, P_dev_all, case_ids, dev_types = load_data_for_evaluation(processed_dir, probs_file)
        
        # Test-Set isolieren
        test_mask = get_test_mask(case_ids)
        y_test = y_true_all[test_mask]
        P_test = P_dev_all[test_mask]
        
        # Evaluierung pro Typ
        results = []
        for i in range(len(dev_types)):
            y_t_col = y_test[:, i]
            y_p_col = P_test[:, i]
            
            metrics = evaluate_single_type(y_t_col, y_p_col)
            if metrics is not None:
                results.append(metrics)
        
        # Macro Average
        if results:
            df_res = pd.DataFrame(results)
            macro_avg = df_res.mean()
            return {
                "AUC": macro_avg['AUC'],
                "Prec_Dev": macro_avg['Prec_Dev'],
                "Rec_Dev": macro_avg['Rec_Dev'],
                "Prec_NoDev": macro_avg['Prec_NoDev'],
                "Rec_NoDev": macro_avg['Rec_NoDev']
            }
        else:
            return None
    except Exception as e:
        print(f"  [WARN] Evaluierung fehlgeschlagen: {e}")
        return None


def evaluate_all_models(output_dir: Path):
    """Evaluiert alle trainierten Modelle und gibt eine Zusammenfassung aus."""
    print("\n" + "="*80)
    print("EVALUIERUNG ALLER MODELLE")
    print("="*80)
    
    models = [
        ("idp_separate_ffn_probs.npz", "Separate FFN"),
        ("idp_separate_lstm_probs.npz", "Separate LSTM"),
        ("idp_collective_ffn_probs.npz", "Collective FFN"),
        ("idp_collective_lstm_probs.npz", "Collective LSTM"),
    ]
    
    results = {}
    
    for probs_file, model_name in models:
        print(f"\nEvaluiere {model_name}...")
        metrics = evaluate_model(output_dir, probs_file, model_name)
        if metrics is not None:
            results[model_name] = metrics
    
    # Zusammenfassung ausgeben
    if results:
        print("\n" + "="*80)
        print("ZUSAMMENFASSUNG: METRIKEN OHNE KATEGORISCHE TRACE-ATTRIBUTE")
        print("="*80)
        print(f"{'Modell':<25} | {'AUC':<8} | {'Precision (Dev)':<18} | {'Recall (Dev)':<15} | {'Precision (NoDev)':<20} | {'Recall (NoDev)':<17}")
        print("-" * 80)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<25} | "
                  f"{metrics['AUC']:.4f}    | "
                  f"{metrics['Prec_Dev']:.4f}              | "
                  f"{metrics['Rec_Dev']:.4f}            | "
                  f"{metrics['Prec_NoDev']:.4f}                 | "
                  f"{metrics['Rec_NoDev']:.4f}")
        
        print("="*80)
    else:
        print("\n[WARNUNG] Keine Metriken-Ergebnisse verfügbar.")


def main():
    """
    Hauptfunktion: Erstellt Encoding ohne kategorische Trace-Attribute und trainiert alle Modelle.
    
    Pipeline:
    1. Encoding ohne kategorische Trace-Attribute erstellen
    2. Separate FFN trainieren
    3. Separate LSTM trainieren
    4. Collective FFN trainieren
    5. Collective LSTM trainieren
    """
    parser = argparse.ArgumentParser(
        description="Erstellt Encoding ohne kategorische Trace-Attribute und trainiert alle Modelle"
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
        help="Input-Verzeichnis für Event-Log und Labels. Standard: data/processed/ oder data/processed/{dataset}/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output-Verzeichnis für Encodings und Modelle. Standard: data/processed_no_cat_trace/ oder data/processed_no_cat_trace/{dataset}/"
    )
    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Überspringe Encoding-Schritt (wenn bereits vorhanden)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Überspringe Training-Schritt (nur Encoding erstellen)"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    
    # Verzeichnisse bestimmen
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
    elif args.dataset is not None:
        input_dir = project_root / "data" / "processed" / args.dataset
    else:
        input_dir = project_root / "data" / "processed"
    
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.dataset is not None:
        output_dir = project_root / "data" / "processed_no_cat_trace" / args.dataset
    else:
        output_dir = project_root / "data" / "processed_no_cat_trace"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENCODING OHNE KATEGORISCHE TRACE-ATTRIBUTE")
    print("="*80)
    print(f"Input-Verzeichnis:  {input_dir}")
    print(f"Output-Verzeichnis: {output_dir}")
    print("="*80)
    
    # 1. Encoding ohne kategorische Trace-Attribute erstellen
    if not args.skip_encoding:
        print("\n>>> Schritt 1: Encoding ohne kategorische Trace-Attribute erstellen <<<")
        print("-" * 80)
        
        # Bestimme Log-Pfad
        raw_dir = project_root / "data" / "raw"
        if args.dataset == "DomesticDeclarations":
            log_path = raw_dir / "DomesticDeclarations.xes"
        else:
            log_path = raw_dir / "finale.csv"
        
        # Labels-Pfad
        labels_path = input_dir / "idp_labels.npz"
        
        # Encoding-Script aufrufen
        encoding_cmd = [
            sys.executable,
            str(project_root / "src" / "encoding.py"),
            "--log", str(log_path),
            "--labels", str(labels_path),
            "--output-dir", str(output_dir),
            "--exclude-categorical-trace-attrs",
        ]
        
        print(f"Führe aus: {' '.join(encoding_cmd)}")
        result = subprocess.run(encoding_cmd, cwd=project_root)
        
        if result.returncode != 0:
            print(f"\n[FEHLER] Encoding fehlgeschlagen mit Exit-Code {result.returncode}")
            sys.exit(1)
        
        print("\n✓ Encoding erfolgreich erstellt")
        
        # idp_labels.npz ins Output-Verzeichnis kopieren (für Training-Scripts)
        labels_src = input_dir / "idp_labels.npz"
        labels_dst = output_dir / "idp_labels.npz"
        if labels_src.exists() and not labels_dst.exists():
            print(f"\nKopiere {labels_src} nach {labels_dst}...")
            shutil.copy2(labels_src, labels_dst)
            print("✓ Labels kopiert")
    else:
        print("\n>>> Schritt 1: Encoding übersprungen (--skip-encoding) <<<")
        
        # Stelle sicher, dass idp_labels.npz vorhanden ist
        labels_src = input_dir / "idp_labels.npz"
        labels_dst = output_dir / "idp_labels.npz"
        if labels_src.exists() and not labels_dst.exists():
            print(f"\nKopiere {labels_src} nach {labels_dst}...")
            shutil.copy2(labels_src, labels_dst)
            print("✓ Labels kopiert")
    
    # 2-5. Alle Modelle trainieren
    if not args.skip_training:
        print("\n" + "="*80)
        print("TRAINING ALLER MODELLE")
        print("="*80)
        
        training_scripts = [
            ("idp_separate_ffn.py", "Separate FFN"),
            ("idp_separate_lstm.py", "Separate LSTM"),
            ("idp_collective_ffn.py", "Collective FFN"),
            ("idp_collective_lstm.py", "Collective LSTM"),
        ]
        
        for script_name, model_name in training_scripts:
            print(f"\n>>> Training: {model_name} <<<")
            print("-" * 80)
            
            train_cmd = [
                sys.executable,
                str(project_root / "src" / script_name),
                "--input-dir", str(output_dir),
            ]
            
            print(f"Führe aus: {' '.join(train_cmd)}")
            result = subprocess.run(train_cmd, cwd=project_root)
            
            if result.returncode != 0:
                print(f"\n[FEHLER] Training von {model_name} fehlgeschlagen mit Exit-Code {result.returncode}")
                sys.exit(1)
            
            print(f"\n✓ {model_name} erfolgreich trainiert")
        
        print("\n" + "="*80)
        print("ALLE MODELLE ERFOLGREICH TRAINIERT")
        print("="*80)
        print(f"Ergebnisse gespeichert in: {output_dir}")
        
        # 6. Alle Modelle evaluieren
        evaluate_all_models(output_dir)
    else:
        print("\n>>> Training übersprungen (--skip-training) <<<")
    
    print("\nFertig!")


if __name__ == "__main__":
    main()

