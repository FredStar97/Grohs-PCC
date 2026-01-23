from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import sys
import json

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import der Training-Module
from idp_separate_ffn import (
    IDPSeparateFFNConfig, load_ffn_data, train_single_classifier
)
from idp_separate_lstm import (
    IDPSeparateLSTMConfig, load_lstm_data, train_single_lstm_model
)
from idp_collective_ffn import (
    IDPCollectiveFFNConfig, load_ffn_data as load_ffn_data_collective, 
    train_collective_model
)
from idp_collective_lstm import (
    IDPCollectiveLSTMConfig, load_lstm_data as load_lstm_data_collective,
    train_collective_lstm
)
import torch

# =========================
# Konfiguration
# =========================

RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0


# =========================
# Statistik-Sammlung
# =========================

def calculate_dataset_stats(
    y_train: np.ndarray,
    case_ids_train: np.ndarray,
    prefix: str = ""
) -> Dict[str, any]:
    """
    Berechnet Statistiken für einen Trainingsdatensatz.
    
    Args:
        y_train: Labels (kann 1D für separate oder 2D für collective sein)
        case_ids_train: Case-IDs für jeden Sample
        prefix: Präfix für die Statistik-Keys (z.B. "before_oss" oder "after_oss")
    
    Returns:
        Dictionary mit Statistiken
    """
    n_samples = len(y_train)
    n_unique_traces = len(np.unique(case_ids_train))
    
    # Für separate Modelle: y_train ist 1D (binär)
    # Für collective Modelle: y_train ist 2D (multi-label)
    if y_train.ndim == 1:
        # Separate Modelle: binäre Klassifikation
        n_deviating = int(np.sum(y_train == 1))
        n_conforming = int(np.sum(y_train == 0))
    else:
        # Collective Modelle: multi-label
        # Ein Sample ist deviant, wenn es mindestens eine Abweichung hat
        y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
        n_deviating = int(np.sum(y_is_deviant))
        n_conforming = int(len(y_is_deviant) - n_deviating)
    
    # Imbalance Ratio: conforming / deviating
    if n_deviating > 0:
        imbalance_ratio = n_conforming / n_deviating
    else:
        imbalance_ratio = float('inf')
    
    stats = {
        f"{prefix}_n_samples": n_samples,
        f"{prefix}_n_deviating": n_deviating,
        f"{prefix}_n_conforming": n_conforming,
        f"{prefix}_n_unique_traces": n_unique_traces,
        f"{prefix}_imbalance_ratio": imbalance_ratio,
    }
    
    return stats


def collect_training_dataset_stats(
    model_type: str,
    use_oss: bool,
    before_stats: Dict[str, any],
    after_stats: Dict[str, any],
    deviation_type: Optional[str] = None
) -> Dict[str, any]:
    """
    Sammelt und berechnet Trainingsdatensatz-Statistiken.
    
    Args:
        model_type: Typ des Modells (separate_ffn, separate_lstm, collective_ffn, collective_lstm)
        use_oss: Ob OSS verwendet wurde
        before_stats: Statistiken vor OSS
        after_stats: Statistiken nach OSS
        deviation_type: Deviation Type Name (nur für separate Modelle)
    
    Returns:
        Dictionary mit vollständigen Statistiken inkl. Reduktion
    """
    before_n_samples = before_stats.get("before_oss_n_samples", 0)
    after_n_samples = after_stats.get("after_oss_n_samples", 0)
    
    before_n_deviating = before_stats.get("before_oss_n_deviating", 0)
    after_n_deviating = after_stats.get("after_oss_n_deviating", 0)
    
    before_n_conforming = before_stats.get("before_oss_n_conforming", 0)
    after_n_conforming = after_stats.get("after_oss_n_conforming", 0)
    
    # Reduktion berechnen
    samples_removed = before_n_samples - after_n_samples
    if before_n_samples > 0:
        reduction_percent = (samples_removed / before_n_samples) * 100.0
    else:
        reduction_percent = 0.0
    
    conforming_removed = before_n_conforming - after_n_conforming
    deviating_removed = before_n_deviating - after_n_deviating
    
    result = {
        "model_type": model_type,
        "use_oss": use_oss,
        "before_oss": {
            "n_samples": before_n_samples,
            "n_deviating": before_n_deviating,
            "n_conforming": before_n_conforming,
            "n_unique_traces": before_stats.get("before_oss_n_unique_traces", 0),
            "imbalance_ratio": before_stats.get("before_oss_imbalance_ratio", 0.0)
        },
        "after_oss": {
            "n_samples": after_n_samples,
            "n_deviating": after_n_deviating,
            "n_conforming": after_n_conforming,
            "n_unique_traces": after_stats.get("after_oss_n_unique_traces", 0),
            "imbalance_ratio": after_stats.get("after_oss_imbalance_ratio", 0.0)
        },
        "reduction": {
            "samples_removed": samples_removed,
            "reduction_percent": reduction_percent,
            "conforming_removed": conforming_removed,
            "deviating_removed": deviating_removed
        }
    }
    
    if deviation_type is not None:
        result["deviation_type"] = deviation_type
    
    return result


# =========================
# Hilfsfunktionen für Evaluation
# =========================

def get_test_mask(case_ids: np.ndarray) -> np.ndarray:
    """Rekonstruiert den Test-Split basierend auf eindeutigen Case-IDs."""
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


def evaluate_model(
    processed_dir: Path,
    probs_file: str,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluiert ein Modell und gibt Macro-Average Metriken zurück.
    Unterstützt gefilterte Labels aus encoding_labels.npz (nach Prefix-Filtering).
    
    Args:
        processed_dir: Pfad zum processed Verzeichnis
        probs_file: Name der Wahrscheinlichkeits-Datei (z.B. "idp_separate_ffn_probs.npz")
        model_name: Name des Modells (für Fehlermeldungen)
    
    Returns:
        Dictionary mit Macro-Average Metriken
    """
    # 1. Predictions laden (bestimmt die Anzahl der Samples)
    probs_path = processed_dir / probs_file
    if not probs_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {probs_path}. Bitte erst {model_name} trainieren.")
    
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
    
    # Lade Original-Labels (für dev_types, falls y_idp nicht vorhanden)
    idp_data = np.load(idp_labels_path, allow_pickle=True)
    dev_types_raw = list(idp_data["dev_types"])
    
    # Prüfe, ob gefilterte Labels existieren
    if encoding_labels_path.exists():
        # Verwende gefilterte Labels (nach Prefix-Filtering)
        encoding_data = np.load(encoding_labels_path, allow_pickle=True)
        case_ids_filtered = encoding_data["case_ids"]
        
        # Prüfe, ob y_idp in encoding_labels.npz vorhanden ist
        if "y_idp" in encoding_data:
            # Verwende gefilterte Labels direkt
            y_filtered = encoding_data["y_idp"]
            case_ids = case_ids_filtered
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
                    case_ids = case_ids_filtered
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
                    case_ids = case_ids_filtered
                else:
                    raise ValueError(
                        f"Konnte gefilterte Labels nicht korrekt mappen: "
                        f"{len(filtered_indices)} Indizes gefunden, aber {len(case_ids_filtered)} erwartet. "
                        f"Bitte stelle sicher, dass prefix_lengths in beiden Dateien vorhanden sind."
                    )
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
                y_filtered = y_all[filtered_indices]
                case_ids = case_ids_pred
            else:
                # Fallback: Verwende alle Original-Labels
                y_filtered = y_all
                case_ids = case_ids_all
        else:
            # Keine case_ids in Predictions: Verwende alle Original-Labels
            y_filtered = idp_data["y"]
            case_ids = idp_data["case_ids"]

    # 3. Ground Truth filtern (Spalten-Matching)
    # Wir brauchen aus y_filtered nur die Spalten, die das Modell auch vorhergesagt hat
    keep_indices = []
    raw_type_to_idx = {name: i for i, name in enumerate(dev_types_raw)}
    
    missing_types = []
    for dt in dev_types_trained:
        if dt in raw_type_to_idx:
            keep_indices.append(raw_type_to_idx[dt])
        else:
            missing_types.append(dt)
            
    if missing_types:
        raise ValueError(f"Trainierte Typen nicht in Raw Labels gefunden: {missing_types}")
        
    y_true_matched = y_filtered[:, keep_indices]

    # Sanity Check
    if y_true_matched.shape[0] != P_dev_all.shape[0]:
        raise AssertionError(
            f"Shape Mismatch! Labels: {y_true_matched.shape}, Preds: {P_dev_all.shape}. "
            f"Bitte stelle sicher, dass encoding_labels.npz mit y_idp existiert oder "
            f"dass die Labels korrekt gefiltert wurden."
        )
    if len(case_ids) != P_dev_all.shape[0]:
        raise AssertionError(
            f"Shape Mismatch! case_ids: {len(case_ids)}, Preds: {P_dev_all.shape[0]}. "
            f"Bitte stelle sicher, dass case_ids in {probs_file} gespeichert wurden."
        )

    # 4. Test-Set isolieren
    test_mask = get_test_mask(case_ids)
    y_test = y_true_matched[test_mask]
    P_test = P_dev_all[test_mask]

    # Evaluierung pro Typ
    results = []
    for i, dev_name in enumerate(dev_types_trained):
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
        return {
            "AUC": 0.0,
            "Prec_Dev": 0.0,
            "Rec_Dev": 0.0,
            "Prec_NoDev": 0.0,
            "Rec_NoDev": 0.0
        }


# =========================
# Training-Funktionen
# =========================

def train_separate_ffn_with_config_no_trace(processed_dir: Path, use_oss: bool) -> List[Dict[str, any]]:
    """
    Trainiert separate FFN mit gegebener OSS-Konfiguration, OHNE Trace-Attribute.
    
    Returns:
        Liste von Statistik-Dictionaries (eines pro Deviation Type)
    """
    models_dir = processed_dir / "models_idp_separate_ffn"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPSeparateFFNConfig(use_oss=use_oss)
    X, y_all, case_ids, dev_types = load_ffn_data(processed_dir)
    
    # Trace-Attribute aus X entfernen
    # Lade Metadaten um zu wissen, welche Spalten Trace-Attribute sind
    meta_path = processed_dir / "encoding_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        ffn_cols = meta.get("ffn_reference_columns", [])
        
        # Identifiziere Trace-Attribute-Spalten
        trace_numeric_cols = meta.get("trace_numeric_cols", [])
        trace_categorical_cols = meta.get("trace_categorical_cols", [])
        weekday_names = meta.get("weekday_names", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        
        # Erstelle Liste aller Trace-Attribute-Spalten
        trace_cols_to_remove = set()
        trace_cols_to_remove.update(trace_numeric_cols)
        trace_cols_to_remove.update(trace_categorical_cols)
        # Weekday-Features
        for day in weekday_names:
            trace_cols_to_remove.add(f"weekday_start_{day}")
            trace_cols_to_remove.add(f"weekday_end_{day}")
        
        # Finde Indizes der zu entfernenden Spalten
        indices_to_keep = [i for i, col in enumerate(ffn_cols) if col not in trace_cols_to_remove]
        X = X[:, indices_to_keep]
    else:
        print("[WARNUNG] encoding_meta.json nicht gefunden. Kann Trace-Attribute nicht entfernen.")
    
    n_prefixes = X.shape[0]
    m_devs = len(dev_types)
    P_dev_all = np.zeros((n_prefixes, m_devs), dtype=np.float32)
    
    all_stats = []
    
    for i, dev_name in enumerate(dev_types):
        # Statistiken sammeln: Trainingslogik nachvollziehen
        y_dev = y_all[:, i].astype(np.int64)
        unique_cases = np.unique(case_ids)
        train_cases, _ = train_test_split(
            unique_cases, 
            test_size=1.0/3.0, 
            random_state=cfg.random_state
        )
        train_mask = np.isin(case_ids, train_cases)
        y_train = y_dev[train_mask]
        case_ids_train = case_ids[train_mask]
        
        # Statistik VOR OSS
        before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
        
        # OSS anwenden (wenn aktiviert)
        if cfg.use_oss:
            from imblearn.under_sampling import OneSidedSelection
            try:
                oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
                X_train = X[train_mask]
                _, y_train_res = oss.fit_resample(X_train, y_train)
                kept_indices = oss.sample_indices_
                case_ids_train_res = case_ids_train[kept_indices]
                y_train_res = y_train_res
            except Exception:
                y_train_res = y_train
                case_ids_train_res = case_ids_train
        else:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
        
        # Statistik NACH OSS
        after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
        
        # Statistik zusammenführen
        stats = collect_training_dataset_stats(
            "separate_ffn", use_oss, before_stats, after_stats, dev_name
        )
        all_stats.append(stats)
        
        # Training durchführen
        model, probs = train_single_classifier(
            dev_idx=i, dev_name=dev_name, X=X, y_all=y_all,
            case_ids=case_ids, cfg=cfg
        )
        P_dev_all[:, i] = probs
        if model is not None:
            model_path = models_dir / f"ffn_d{i}_no_trace.pt"
            torch.save(model.state_dict(), model_path)
    
    suffix = "_no_oss_no_trace" if not use_oss else "_no_trace"
    out_path = processed_dir / f"idp_separate_ffn_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types), case_ids=case_ids
    )
    
    return all_stats


def train_separate_ffn_with_config(processed_dir: Path, use_oss: bool) -> List[Dict[str, any]]:
    """
    Trainiert separate FFN mit gegebener OSS-Konfiguration.
    
    Returns:
        Liste von Statistik-Dictionaries (eines pro Deviation Type)
    """
    models_dir = processed_dir / "models_idp_separate_ffn"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPSeparateFFNConfig(use_oss=use_oss)
    X, y_all, case_ids, dev_types = load_ffn_data(processed_dir)
    
    n_prefixes = X.shape[0]
    m_devs = len(dev_types)
    P_dev_all = np.zeros((n_prefixes, m_devs), dtype=np.float32)
    
    all_stats = []
    
    for i, dev_name in enumerate(dev_types):
        # Statistiken sammeln: Trainingslogik nachvollziehen
        y_dev = y_all[:, i].astype(np.int64)
        unique_cases = np.unique(case_ids)
        train_cases, _ = train_test_split(
            unique_cases, 
            test_size=1.0/3.0, 
            random_state=cfg.random_state
        )
        train_mask = np.isin(case_ids, train_cases)
        y_train = y_dev[train_mask]
        case_ids_train = case_ids[train_mask]
        
        # Statistik VOR OSS
        before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
        
        # OSS anwenden (wenn aktiviert)
        if cfg.use_oss:
            from imblearn.under_sampling import OneSidedSelection
            try:
                oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
                X_train = X[train_mask]
                _, y_train_res = oss.fit_resample(X_train, y_train)
                kept_indices = oss.sample_indices_
                case_ids_train_res = case_ids_train[kept_indices]
                y_train_res = y_train_res
            except Exception:
                y_train_res = y_train
                case_ids_train_res = case_ids_train
        else:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
        
        # Statistik NACH OSS
        after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
        
        # Statistik zusammenführen
        stats = collect_training_dataset_stats(
            "separate_ffn", use_oss, before_stats, after_stats, dev_name
        )
        all_stats.append(stats)
        
        # Training durchführen
        model, probs = train_single_classifier(
            dev_idx=i, dev_name=dev_name, X=X, y_all=y_all,
            case_ids=case_ids, cfg=cfg
        )
        P_dev_all[:, i] = probs
        if model is not None:
            model_path = models_dir / f"ffn_d{i}.pt"
            torch.save(model.state_dict(), model_path)
    
    suffix = "_no_oss" if not use_oss else ""
    out_path = processed_dir / f"idp_separate_ffn_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types), case_ids=case_ids
    )
    
    return all_stats


def train_separate_lstm_with_config(processed_dir: Path, use_oss: bool) -> List[Dict[str, any]]:
    """
    Trainiert separate LSTM mit gegebener OSS-Konfiguration.
    
    Returns:
        Liste von Statistik-Dictionaries (eines pro Deviation Type)
    """
    models_dir = processed_dir / "models_idp_separate_lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPSeparateLSTMConfig(use_oss=use_oss)
    inputs, y_all, case_ids, dev_types, meta = load_lstm_data(processed_dir)
    
    # Filterung gemäß Paper
    keep_indices = []
    for i in range(len(dev_types)):
        dev_col = y_all[:, i]
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        if len(cases_with_dev) > 1:
            keep_indices.append(i)
    
    if not keep_indices:
        print("Keine trainierbaren Abweichungen.")
        return []
    
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    n_prefixes, m_dev = y_filtered.shape
    P_dev_all = np.zeros((n_prefixes, m_dev), dtype=np.float32)
    
    all_stats = []
    
    for i, dev_name in enumerate(dev_types_filtered):
        # Statistiken sammeln: Trainingslogik nachvollziehen
        y_dev = y_filtered[:, i].astype(np.int64)
        unique_cases = np.unique(case_ids)
        train_cases, _ = train_test_split(
            unique_cases, 
            test_size=1.0/3.0, 
            random_state=cfg.random_state
        )
        train_mask = np.isin(case_ids, train_cases)
        y_train = y_dev[train_mask]
        case_ids_train = case_ids[train_mask]
        
        # Statistik VOR OSS
        before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
        
        # OSS anwenden (wenn aktiviert)
        if cfg.use_oss:
            from imblearn.under_sampling import OneSidedSelection
            try:
                # OSS benötigt eine flache Feature-Repräsentation
                # Daher werden alle 4 Input-Komponenten zu einem Feature-Vektor konkateniert
                # (gemäß Paper Fig. 5: Alle Komponenten gehen in "Separate Undersampling")
                X_act_train = inputs["X_act"][train_mask]
                X_res_train = inputs["X_res"][train_mask]
                X_time_train = inputs["X_time"][train_mask]
                X_trace_train = inputs["X_trace"][train_mask]
                n_samples_train = X_trace_train.shape[0]
                X_oss_train = np.concatenate(
                    [
                        X_act_train.reshape(n_samples_train, -1),   # Activities flach machen
                        X_res_train.reshape(n_samples_train, -1),   # Resources flach machen
                        X_time_train.reshape(n_samples_train, -1),  # Time flach machen
                        X_trace_train,                              # Trace Features (bereits flach)
                    ],
                    axis=1,
                )
                oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
                _, _ = oss.fit_resample(X_oss_train, y_train)
                kept_indices = oss.sample_indices_
                case_ids_train_res = case_ids_train[kept_indices]
                y_train_res = y_train[kept_indices]
            except Exception:
                y_train_res = y_train
                case_ids_train_res = case_ids_train
        else:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
        
        # Statistik NACH OSS
        after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
        
        # Statistik zusammenführen
        stats = collect_training_dataset_stats(
            "separate_lstm", use_oss, before_stats, after_stats, dev_name
        )
        all_stats.append(stats)
        
        # Training durchführen
        model, probs = train_single_lstm_model(
            dev_idx=i, dev_name=dev_name, inputs=inputs,
            y_all=y_filtered, case_ids=case_ids, meta=meta, cfg=cfg
        )
        P_dev_all[:, i] = probs
        if model:
            torch.save(model.state_dict(), models_dir / f"lstm_d{i}.pt")
    
    suffix = "_no_oss" if not use_oss else ""
    out_path = processed_dir / f"idp_separate_lstm_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), case_ids=case_ids
    )
    
    return all_stats


def train_collective_ffn_with_config_no_trace(processed_dir: Path, use_oss: bool) -> Dict[str, any]:
    """
    Trainiert collective FFN mit gegebener OSS-Konfiguration, OHNE Trace-Attribute.
    
    Returns:
        Statistik-Dictionary
    """
    models_dir = processed_dir / "models_idp_collective_ffn"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPCollectiveFFNConfig(use_oss=use_oss)
    X, y_all, case_ids, dev_types = load_ffn_data_collective(processed_dir)
    
    # Trace-Attribute aus X entfernen
    # Lade Metadaten um zu wissen, welche Spalten Trace-Attribute sind
    meta_path = processed_dir / "encoding_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        ffn_cols = meta.get("ffn_reference_columns", [])
        
        # Identifiziere Trace-Attribute-Spalten
        trace_numeric_cols = meta.get("trace_numeric_cols", [])
        trace_categorical_cols = meta.get("trace_categorical_cols", [])
        weekday_names = meta.get("weekday_names", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        
        # Erstelle Liste aller Trace-Attribute-Spalten
        trace_cols_to_remove = set()
        trace_cols_to_remove.update(trace_numeric_cols)
        trace_cols_to_remove.update(trace_categorical_cols)
        # Weekday-Features
        for day in weekday_names:
            trace_cols_to_remove.add(f"weekday_start_{day}")
            trace_cols_to_remove.add(f"weekday_end_{day}")
        
        # Finde Indizes der zu entfernenden Spalten
        indices_to_keep = [i for i, col in enumerate(ffn_cols) if col not in trace_cols_to_remove]
        X = X[:, indices_to_keep]
    else:
        print("[WARNUNG] encoding_meta.json nicht gefunden. Kann Trace-Attribute nicht entfernen.")
    
    # Filterung gemäß Paper
    keep_indices = []
    for i in range(len(dev_types)):
        dev_col = y_all[:, i]
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        if len(cases_with_dev) > 1:
            keep_indices.append(i)
    
    if not keep_indices:
        print("Keine Deviation Types übrig.")
        return {}
    
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    # Statistiken sammeln: Trainingslogik nachvollziehen
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    train_mask = np.isin(case_ids, train_cases)
    y_train = y_filtered[train_mask]
    case_ids_train = case_ids[train_mask]
    
    # Statistik VOR OSS
    before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
    
    # OSS anwenden (wenn aktiviert)
    if cfg.use_oss:
        from imblearn.under_sampling import OneSidedSelection
        try:
            y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
            X_train = X[train_mask]
            oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
            _, _ = oss.fit_resample(X_train, y_is_deviant)
            kept_indices = oss.sample_indices_
            y_train_res = y_train[kept_indices]
            case_ids_train_res = case_ids_train[kept_indices]
        except Exception:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
    else:
        y_train_res = y_train
        case_ids_train_res = case_ids_train
    
    # Statistik NACH OSS
    after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
    
    # Statistik zusammenführen
    stats = collect_training_dataset_stats(
        "collective_ffn", use_oss, before_stats, after_stats
    )
    
    # Training durchführen
    model, P_dev_all = train_collective_model(X, y_filtered, case_ids, cfg)
    
    model_path = models_dir / "collective_ffn_no_trace.pt"
    torch.save(model.state_dict(), model_path)
    
    suffix = "_no_oss_no_trace" if not use_oss else "_no_trace"
    out_path = processed_dir / f"idp_collective_ffn_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), case_ids=case_ids
    )
    
    return stats


def train_collective_ffn_with_config(processed_dir: Path, use_oss: bool) -> Dict[str, any]:
    """
    Trainiert collective FFN mit gegebener OSS-Konfiguration.
    
    Returns:
        Statistik-Dictionary
    """
    models_dir = processed_dir / "models_idp_collective_ffn"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPCollectiveFFNConfig(use_oss=use_oss)
    X, y_all, case_ids, dev_types = load_ffn_data_collective(processed_dir)
    
    # Filterung gemäß Paper
    keep_indices = []
    for i in range(len(dev_types)):
        dev_col = y_all[:, i]
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        if len(cases_with_dev) > 1:
            keep_indices.append(i)
    
    if not keep_indices:
        print("Keine Deviation Types übrig.")
        return {}
    
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    # Statistiken sammeln: Trainingslogik nachvollziehen
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    train_mask = np.isin(case_ids, train_cases)
    y_train = y_filtered[train_mask]
    case_ids_train = case_ids[train_mask]
    
    # Statistik VOR OSS
    before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
    
    # OSS anwenden (wenn aktiviert)
    if cfg.use_oss:
        from imblearn.under_sampling import OneSidedSelection
        try:
            y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
            X_train = X[train_mask]
            oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
            _, _ = oss.fit_resample(X_train, y_is_deviant)
            kept_indices = oss.sample_indices_
            y_train_res = y_train[kept_indices]
            case_ids_train_res = case_ids_train[kept_indices]
        except Exception:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
    else:
        y_train_res = y_train
        case_ids_train_res = case_ids_train
    
    # Statistik NACH OSS
    after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
    
    # Statistik zusammenführen
    stats = collect_training_dataset_stats(
        "collective_ffn", use_oss, before_stats, after_stats
    )
    
    # Training durchführen
    model, P_dev_all = train_collective_model(X, y_filtered, case_ids, cfg)
    
    model_path = models_dir / "collective_ffn.pt"
    torch.save(model.state_dict(), model_path)
    
    suffix = "_no_oss" if not use_oss else ""
    out_path = processed_dir / f"idp_collective_ffn_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), case_ids=case_ids
    )
    
    return stats


def train_collective_lstm_with_config(processed_dir: Path, use_oss: bool) -> Dict[str, any]:
    """
    Trainiert collective LSTM mit gegebener OSS-Konfiguration.
    
    Returns:
        Statistik-Dictionary
    """
    models_dir = processed_dir / "models_idp_collective_lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPCollectiveLSTMConfig(use_oss=use_oss)
    inputs, y_all, case_ids, dev_types, meta = load_lstm_data_collective(processed_dir)
    
    # Filterung gemäß Paper
    keep_indices = []
    for i in range(len(dev_types)):
        dev_col = y_all[:, i]
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        if len(cases_with_dev) > 1:
            keep_indices.append(i)
    
    if not keep_indices:
        print("Keine trainierbaren Abweichungen.")
        return {}
    
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    # Statistiken sammeln: Trainingslogik nachvollziehen
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    train_mask = np.isin(case_ids, train_cases)
    y_train = y_filtered[train_mask]
    case_ids_train = case_ids[train_mask]
    
    # Statistik VOR OSS
    before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
    
    # OSS anwenden (wenn aktiviert)
    if cfg.use_oss:
        from imblearn.under_sampling import OneSidedSelection
        from sklearn.preprocessing import OneHotEncoder
        try:
            y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
            X_trace_train = inputs["X_trace"][train_mask]
            X_act_train = inputs["X_act"][train_mask]
            X_res_train = inputs["X_res"][train_mask]
            n_samples_train = X_trace_train.shape[0]
            
            # One-Hot-Encoding für Activities und Resources (wie in idp_collective_lstm.py)
            # Flach machen für OneHotEncoder: (n_samples, seq_len) -> (n_samples * seq_len,)
            X_act_flat = X_act_train.reshape(-1, 1)
            X_res_flat = X_res_train.reshape(-1, 1)
            
            # One-Hot-Encoding für Activities und Resources
            ohe_act = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_res = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            X_act_onehot_flat = ohe_act.fit_transform(X_act_flat)
            X_res_onehot_flat = ohe_res.fit_transform(X_res_flat)
            
            # Zurück zu (n_samples, seq_len * n_categories) reshapen
            seq_len = X_act_train.shape[1]
            n_act_categories = X_act_onehot_flat.shape[1]
            n_res_categories = X_res_onehot_flat.shape[1]
            
            X_act_onehot = X_act_onehot_flat.reshape(n_samples_train, seq_len * n_act_categories)
            X_res_onehot = X_res_onehot_flat.reshape(n_samples_train, seq_len * n_res_categories)
            
            # Konkateniere One-Hot-Features mit Trace-Features für OSS
            # (X_time wird nicht verwendet, da es ebenfalls kategorial ist und weniger kritisch)
            X_oss_train = np.concatenate(
                [
                    X_act_onehot,   # Activities als One-Hot
                    X_res_onehot,   # Resources als One-Hot
                    X_trace_train,  # Trace Features (bereits numerisch)
                ],
                axis=1,
            )
            
            oss = OneSidedSelection(n_neighbors=7, n_seeds_S=250, random_state=cfg.random_state)
            _, _ = oss.fit_resample(X_oss_train, y_is_deviant)
            kept_indices = oss.sample_indices_
            y_train_res = y_train[kept_indices]
            case_ids_train_res = case_ids_train[kept_indices]
        except Exception:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
    else:
        y_train_res = y_train
        case_ids_train_res = case_ids_train
    
    # Statistik NACH OSS
    after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
    
    # Statistik zusammenführen
    stats = collect_training_dataset_stats(
        "collective_lstm", use_oss, before_stats, after_stats
    )
    
    # Training durchführen
    model, P_dev_all = train_collective_lstm(
        inputs, y_filtered, case_ids, meta, cfg
    )
    
    model_path = models_dir / "collective_lstm.pt"
    torch.save(model.state_dict(), model_path)
    
    suffix = "_no_oss" if not use_oss else ""
    out_path = processed_dir / f"idp_collective_lstm_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), case_ids=case_ids
    )
    
    return stats


def train_separate_lstm_with_config_no_trace(processed_dir: Path, use_oss: bool) -> List[Dict[str, any]]:
    """
    Trainiert separate LSTM mit gegebener OSS-Konfiguration, OHNE Trace-Attribute.
    
    Returns:
        Liste von Statistik-Dictionaries (eines pro Deviation Type)
    """
    models_dir = processed_dir / "models_idp_separate_lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPSeparateLSTMConfig(use_oss=use_oss)
    inputs, y_all, case_ids, dev_types, meta = load_lstm_data(processed_dir)
    
    # Trace-Attribute deaktivieren: X_trace auf leeres Array setzen
    n_samples = inputs["X_trace"].shape[0]
    inputs["X_trace"] = np.zeros((n_samples, 0), dtype=np.float32)
    
    # Filterung gemäß Paper
    keep_indices = []
    for i in range(len(dev_types)):
        dev_col = y_all[:, i]
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        if len(cases_with_dev) > 1:
            keep_indices.append(i)
    
    if not keep_indices:
        print("Keine trainierbaren Abweichungen.")
        return []
    
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    n_prefixes, m_dev = y_filtered.shape
    P_dev_all = np.zeros((n_prefixes, m_dev), dtype=np.float32)
    
    all_stats = []
    
    for i, dev_name in enumerate(dev_types_filtered):
        # Statistiken sammeln: Trainingslogik nachvollziehen
        y_dev = y_filtered[:, i].astype(np.int64)
        unique_cases = np.unique(case_ids)
        train_cases, _ = train_test_split(
            unique_cases, 
            test_size=1.0/3.0, 
            random_state=cfg.random_state
        )
        train_mask = np.isin(case_ids, train_cases)
        y_train = y_dev[train_mask]
        case_ids_train = case_ids[train_mask]
        
        # Statistik VOR OSS
        before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
        
        # OSS anwenden (wenn aktiviert) - OHNE Trace-Attribute
        if cfg.use_oss:
            from imblearn.under_sampling import OneSidedSelection
            try:
                X_act_train = inputs["X_act"][train_mask]
                X_res_train = inputs["X_res"][train_mask]
                X_time_train = inputs["X_time"][train_mask]
                # X_trace ist leer, wird nicht verwendet
                n_samples_train = X_act_train.shape[0]
                X_oss_train = np.concatenate(
                    [
                        X_act_train.reshape(n_samples_train, -1),   # Activities flach machen
                        X_res_train.reshape(n_samples_train, -1),   # Resources flach machen
                        X_time_train.reshape(n_samples_train, -1),  # Time flach machen
                        # KEINE Trace Features
                    ],
                    axis=1,
                )
                oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
                _, _ = oss.fit_resample(X_oss_train, y_train)
                kept_indices = oss.sample_indices_
                case_ids_train_res = case_ids_train[kept_indices]
                y_train_res = y_train[kept_indices]
            except Exception:
                y_train_res = y_train
                case_ids_train_res = case_ids_train
        else:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
        
        # Statistik NACH OSS
        after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
        
        # Statistik zusammenführen
        stats = collect_training_dataset_stats(
            "separate_lstm", use_oss, before_stats, after_stats, dev_name
        )
        all_stats.append(stats)
        
        # Training durchführen
        model, probs = train_single_lstm_model(
            dev_idx=i, dev_name=dev_name, inputs=inputs,
            y_all=y_filtered, case_ids=case_ids, meta=meta, cfg=cfg
        )
        P_dev_all[:, i] = probs
        if model:
            torch.save(model.state_dict(), models_dir / f"lstm_d{i}_no_trace.pt")
    
    suffix = "_no_oss_no_trace" if not use_oss else "_no_trace"
    out_path = processed_dir / f"idp_separate_lstm_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), case_ids=case_ids
    )
    
    return all_stats


def train_collective_lstm_with_config_no_trace(processed_dir: Path, use_oss: bool) -> Dict[str, any]:
    """
    Trainiert collective LSTM mit gegebener OSS-Konfiguration, OHNE Trace-Attribute.
    
    Returns:
        Statistik-Dictionary
    """
    models_dir = processed_dir / "models_idp_collective_lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = IDPCollectiveLSTMConfig(use_oss=use_oss)
    inputs, y_all, case_ids, dev_types, meta = load_lstm_data_collective(processed_dir)
    
    # Trace-Attribute deaktivieren: X_trace auf leeres Array setzen
    n_samples = inputs["X_trace"].shape[0]
    inputs["X_trace"] = np.zeros((n_samples, 0), dtype=np.float32)
    
    # Filterung gemäß Paper
    keep_indices = []
    for i in range(len(dev_types)):
        dev_col = y_all[:, i]
        cases_with_dev = np.unique(case_ids[dev_col == 1])
        if len(cases_with_dev) > 1:
            keep_indices.append(i)
    
    if not keep_indices:
        print("Keine trainierbaren Abweichungen.")
        return {}
    
    y_filtered = y_all[:, keep_indices]
    dev_types_filtered = [dev_types[i] for i in keep_indices]
    
    # Statistiken sammeln: Trainingslogik nachvollziehen
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(
        unique_cases, 
        test_size=1.0/3.0, 
        random_state=cfg.random_state
    )
    train_mask = np.isin(case_ids, train_cases)
    y_train = y_filtered[train_mask]
    case_ids_train = case_ids[train_mask]
    
    # Statistik VOR OSS
    before_stats = calculate_dataset_stats(y_train, case_ids_train, "before_oss")
    
    # OSS anwenden (wenn aktiviert) - OHNE Trace-Attribute
    if cfg.use_oss:
        from imblearn.under_sampling import OneSidedSelection
        from sklearn.preprocessing import OneHotEncoder
        try:
            y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
            # X_trace ist leer, wird nicht verwendet
            X_act_train = inputs["X_act"][train_mask]
            X_res_train = inputs["X_res"][train_mask]
            n_samples_train = X_act_train.shape[0]
            
            # One-Hot-Encoding für Activities und Resources
            X_act_flat = X_act_train.reshape(-1, 1)
            X_res_flat = X_res_train.reshape(-1, 1)
            
            ohe_act = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_res = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            X_act_onehot_flat = ohe_act.fit_transform(X_act_flat)
            X_res_onehot_flat = ohe_res.fit_transform(X_res_flat)
            
            seq_len = X_act_train.shape[1]
            n_act_categories = X_act_onehot_flat.shape[1]
            n_res_categories = X_res_onehot_flat.shape[1]
            
            X_act_onehot = X_act_onehot_flat.reshape(n_samples_train, seq_len * n_act_categories)
            X_res_onehot = X_res_onehot_flat.reshape(n_samples_train, seq_len * n_res_categories)
            
            # Konkateniere One-Hot-Features - OHNE Trace-Features
            X_oss_train = np.concatenate(
                [
                    X_act_onehot,   # Activities als One-Hot
                    X_res_onehot,   # Resources als One-Hot
                    # KEINE Trace Features
                ],
                axis=1,
            )
            
            oss = OneSidedSelection(n_neighbors=7, n_seeds_S=250, random_state=cfg.random_state)
            _, _ = oss.fit_resample(X_oss_train, y_is_deviant)
            kept_indices = oss.sample_indices_
            y_train_res = y_train[kept_indices]
            case_ids_train_res = case_ids_train[kept_indices]
        except Exception:
            y_train_res = y_train
            case_ids_train_res = case_ids_train
    else:
        y_train_res = y_train
        case_ids_train_res = case_ids_train
    
    # Statistik NACH OSS
    after_stats = calculate_dataset_stats(y_train_res, case_ids_train_res, "after_oss")
    
    # Statistik zusammenführen
    stats = collect_training_dataset_stats(
        "collective_lstm", use_oss, before_stats, after_stats
    )
    
    # Training durchführen
    model, P_dev_all = train_collective_lstm(
        inputs, y_filtered, case_ids, meta, cfg
    )
    
    model_path = models_dir / "collective_lstm_no_trace.pt"
    torch.save(model.state_dict(), model_path)
    
    suffix = "_no_oss_no_trace" if not use_oss else "_no_trace"
    out_path = processed_dir / f"idp_collective_lstm_probs{suffix}.npz"
    np.savez_compressed(
        out_path, P_dev=P_dev_all,
        dev_types=np.array(dev_types_filtered), case_ids=case_ids
    )
    
    return stats


def train_model_with_oss_config(model_type: str, use_oss: bool, processed_dir: Path) -> List[Dict[str, any]]:
    """
    Trainiert ein Modell mit gegebener OSS-Konfiguration.
    
    Args:
        model_type: "separate_ffn", "separate_lstm", "collective_ffn", "collective_lstm"
        use_oss: Ob OSS verwendet werden soll
        processed_dir: Pfad zum processed Verzeichnis
    
    Returns:
        Liste von Statistik-Dictionaries (für separate Modelle) oder einzelnes Dictionary (für collective)
    """
    print(f"\n{'='*80}")
    print(f"Training {model_type} mit OSS={use_oss}")
    print(f"{'='*80}\n")
    
    if model_type == "separate_ffn":
        return train_separate_ffn_with_config(processed_dir, use_oss)
    elif model_type == "separate_lstm":
        return train_separate_lstm_with_config(processed_dir, use_oss)
    elif model_type == "collective_ffn":
        stats = train_collective_ffn_with_config(processed_dir, use_oss)
        return [stats] if stats else []
    elif model_type == "collective_lstm":
        stats = train_collective_lstm_with_config(processed_dir, use_oss)
        return [stats] if stats else []
    else:
        raise ValueError(f"Unbekannter Modell-Typ: {model_type}")


def train_model_with_oss_config_no_trace(model_type: str, use_oss: bool, processed_dir: Path) -> List[Dict[str, any]]:
    """
    Trainiert ein Modell mit gegebener OSS-Konfiguration, OHNE Trace-Attribute.
    
    Args:
        model_type: "separate_ffn", "separate_lstm", "collective_ffn", "collective_lstm"
        use_oss: Ob OSS verwendet werden soll
        processed_dir: Pfad zum processed Verzeichnis
    
    Returns:
        Liste von Statistik-Dictionaries (für separate Modelle) oder einzelnes Dictionary (für collective)
    """
    print(f"\n{'='*80}")
    print(f"Training {model_type} mit OSS={use_oss}, OHNE Trace-Attribute")
    print(f"{'='*80}\n")
    
    if model_type == "separate_ffn":
        return train_separate_ffn_with_config_no_trace(processed_dir, use_oss)
    elif model_type == "separate_lstm":
        return train_separate_lstm_with_config_no_trace(processed_dir, use_oss)
    elif model_type == "collective_ffn":
        stats = train_collective_ffn_with_config_no_trace(processed_dir, use_oss)
        return [stats] if stats else []
    elif model_type == "collective_lstm":
        stats = train_collective_lstm_with_config_no_trace(processed_dir, use_oss)
        return [stats] if stats else []
    else:
        raise ValueError(f"Unbekannter Modell-Typ: {model_type}")


# =========================
# Vergleichs-Analyse
# =========================

def analyze_dataset_changes(
    stats_oss: List[Dict[str, any]],
    stats_no_oss: List[Dict[str, any]],
    model_type: str
) -> pd.DataFrame:
    """
    Analysiert die Änderungen im Trainingsdatensatz zwischen OSS und No OSS.
    
    Args:
        stats_oss: Liste von Statistik-Dictionaries mit OSS
        stats_no_oss: Liste von Statistik-Dictionaries ohne OSS
        model_type: Typ des Modells
    
    Returns:
        DataFrame mit Vergleichs-Statistiken
    """
    comparison_rows = []
    
    # Für separate Modelle: Vergleich pro Deviation Type
    # Für collective Modelle: nur ein Eintrag
    if model_type.startswith("separate"):
        # Separate Modelle: stats_oss und stats_no_oss sind Listen
        # Wir müssen sie nach deviation_type matchen
        oss_dict = {s.get("deviation_type", ""): s for s in stats_oss}
        no_oss_dict = {s.get("deviation_type", ""): s for s in stats_no_oss}
        
        all_dev_types = set(oss_dict.keys()) | set(no_oss_dict.keys())
        
        for dev_type in all_dev_types:
            oss_stat = oss_dict.get(dev_type, {})
            no_oss_stat = no_oss_dict.get(dev_type, {})
            
            if not oss_stat or not no_oss_stat:
                continue
            
            before_oss = oss_stat.get("before_oss", {})
            after_oss = oss_stat.get("after_oss", {})
            reduction_oss = oss_stat.get("reduction", {})
            
            before_no_oss = no_oss_stat.get("before_oss", {})
            after_no_oss = no_oss_stat.get("after_oss", {})
            reduction_no_oss = no_oss_stat.get("reduction", {})
            
            comparison_rows.append({
                "Model": model_type,
                "Deviation_Type": dev_type,
                "Before_OSS_Samples": before_oss.get("n_samples", 0),
                "After_OSS_Samples": after_oss.get("n_samples", 0),
                "OSS_Reduction_Percent": reduction_oss.get("reduction_percent", 0.0),
                "Before_NoOSS_Samples": before_no_oss.get("n_samples", 0),
                "After_NoOSS_Samples": after_no_oss.get("n_samples", 0),
                "NoOSS_Reduction_Percent": reduction_no_oss.get("reduction_percent", 0.0),
                "OSS_Deviating": after_oss.get("n_deviating", 0),
                "OSS_Conforming": after_oss.get("n_conforming", 0),
                "NoOSS_Deviating": after_no_oss.get("n_deviating", 0),
                "NoOSS_Conforming": after_no_oss.get("n_conforming", 0),
                "OSS_Imbalance_Ratio": after_oss.get("imbalance_ratio", 0.0),
                "NoOSS_Imbalance_Ratio": after_no_oss.get("imbalance_ratio", 0.0),
            })
    else:
        # Collective Modelle: nur ein Eintrag
        if stats_oss and stats_no_oss:
            oss_stat = stats_oss[0]
            no_oss_stat = stats_no_oss[0]
            
            before_oss = oss_stat.get("before_oss", {})
            after_oss = oss_stat.get("after_oss", {})
            reduction_oss = oss_stat.get("reduction", {})
            
            before_no_oss = no_oss_stat.get("before_oss", {})
            after_no_oss = no_oss_stat.get("after_oss", {})
            reduction_no_oss = no_oss_stat.get("reduction", {})
            
            comparison_rows.append({
                "Model": model_type,
                "Deviation_Type": "all",
                "Before_OSS_Samples": before_oss.get("n_samples", 0),
                "After_OSS_Samples": after_oss.get("n_samples", 0),
                "OSS_Reduction_Percent": reduction_oss.get("reduction_percent", 0.0),
                "Before_NoOSS_Samples": before_no_oss.get("n_samples", 0),
                "After_NoOSS_Samples": after_no_oss.get("n_samples", 0),
                "NoOSS_Reduction_Percent": reduction_no_oss.get("reduction_percent", 0.0),
                "OSS_Deviating": after_oss.get("n_deviating", 0),
                "OSS_Conforming": after_oss.get("n_conforming", 0),
                "NoOSS_Deviating": after_no_oss.get("n_deviating", 0),
                "NoOSS_Conforming": after_no_oss.get("n_conforming", 0),
                "OSS_Imbalance_Ratio": after_oss.get("imbalance_ratio", 0.0),
                "NoOSS_Imbalance_Ratio": after_no_oss.get("imbalance_ratio", 0.0),
            })
    
    return pd.DataFrame(comparison_rows)


def print_dataset_statistics(stats: List[Dict[str, any]], use_oss: bool):
    """Gibt detaillierte Trainingsdatensatz-Statistiken aus."""
    oss_label = "MIT OSS" if use_oss else "OHNE OSS"
    print(f"\n{'='*80}")
    print(f"Trainingsdatensatz-Statistiken {oss_label}")
    print(f"{'='*80}")
    
    for stat in stats:
        model_type = stat.get("model_type", "unknown")
        dev_type = stat.get("deviation_type", "all")
        
        before = stat.get("before_oss", {})
        after = stat.get("after_oss", {})
        reduction = stat.get("reduction", {})
        
        print(f"\nModell: {model_type}")
        if dev_type != "all":
            print(f"Deviation Type: {dev_type}")
        
        print(f"  Vor OSS:")
        print(f"    Samples: {before.get('n_samples', 0):,}")
        print(f"    Deviating: {before.get('n_deviating', 0):,}")
        print(f"    Conforming: {before.get('n_conforming', 0):,}")
        print(f"    Unique Traces: {before.get('n_unique_traces', 0):,}")
        print(f"    Imbalance Ratio: {before.get('imbalance_ratio', 0.0):.2f}")
        
        if use_oss:
            print(f"  Nach OSS:")
            print(f"    Samples: {after.get('n_samples', 0):,}")
            print(f"    Deviating: {after.get('n_deviating', 0):,}")
            print(f"    Conforming: {after.get('n_conforming', 0):,}")
            print(f"    Unique Traces: {after.get('n_unique_traces', 0):,}")
            print(f"    Imbalance Ratio: {after.get('imbalance_ratio', 0.0):.2f}")
            print(f"  Reduktion:")
            print(f"    Samples entfernt: {reduction.get('samples_removed', 0):,}")
            print(f"    Reduktion: {reduction.get('reduction_percent', 0.0):.2f}%")
            print(f"    Conforming entfernt: {reduction.get('conforming_removed', 0):,}")
            print(f"    Deviating entfernt: {reduction.get('deviating_removed', 0):,}")


# =========================
# Verbesserte Ausgabe-Funktionen
# =========================

def print_architecture_comparison(
    model_type: str,
    metrics_oss: Dict[str, float],
    metrics_no_oss: Dict[str, float],
    oss_stats: Dict[str, any]
) -> None:
    """
    Gibt einen übersichtlichen Vergleich für eine Architektur aus.
    
    Args:
        model_type: Name der Architektur
        metrics_oss: Metriken MIT OSS
        metrics_no_oss: Metriken OHNE OSS
        oss_stats: OSS-Reduktionsstatistiken
    """
    print("\n" + "=" * 80)
    print(f"ARCHITEKTUR: {model_type.upper()}")
    print("=" * 80)
    
    # Metriken-Tabelle
    print(f"\n{'Metrik':<12} | {'Mit OSS':>12} | {'Ohne OSS':>12} | {'Differenz':>12} | {'Besser':<10}")
    print("-" * 70)
    
    for metric in ["AUC", "Prec_Dev", "Rec_Dev", "Prec_NoDev", "Rec_NoDev"]:
        val_oss = metrics_oss.get(metric, 0.0)
        val_no_oss = metrics_no_oss.get(metric, 0.0)
        diff = val_oss - val_no_oss
        better = "Mit OSS" if diff > 0.001 else "Ohne OSS" if diff < -0.001 else "Gleich"
        
        print(f"{metric:<12} | {val_oss:>12.4f} | {val_no_oss:>12.4f} | {diff:>+12.4f} | {better:<10}")
    
    # OSS-Reduktion anzeigen
    if oss_stats:
        print("\n" + "-" * 70)
        print("OSS-REDUKTION:")
        
        before_oss = oss_stats.get("before_oss", {})
        after_oss = oss_stats.get("after_oss", {})
        reduction = oss_stats.get("reduction", {})
        
        n_before = before_oss.get("n_samples", 0)
        n_after = after_oss.get("n_samples", 0)
        reduction_pct = reduction.get("reduction_percent", 0.0)
        samples_removed = reduction.get("samples_removed", 0)
        
        print(f"  Samples vor OSS:  {n_before:>10,}")
        print(f"  Samples nach OSS: {n_after:>10,}")
        print(f"  Reduktion:        {reduction_pct:>10.2f}% ({samples_removed:,} Samples entfernt)")
        
        # Imbalance Ratio
        imb_before = before_oss.get("imbalance_ratio", 0.0)
        imb_after = after_oss.get("imbalance_ratio", 0.0)
        print(f"  Imbalance Ratio vor OSS:  {imb_before:.2f}")
        print(f"  Imbalance Ratio nach OSS: {imb_after:.2f}")
    
    print("=" * 80)


def print_final_summary(
    all_results: List[Dict[str, any]],
    all_oss_stats: Dict[str, Dict[str, any]]
) -> None:
    """
    Gibt eine finale Zusammenfassung aller Architekturen aus.
    
    Args:
        all_results: Liste aller Ergebnisse
        all_oss_stats: Dictionary mit OSS-Statistiken pro Architektur
    """
    print("\n" + "#" * 80)
    print("#" + " " * 26 + "FINALE ZUSAMMENFASSUNG" + " " * 26 + "#")
    print("#" * 80)
    
    # Gruppiere nach Architektur
    arch_results = {}
    for r in all_results:
        model = r["Model"]
        if model not in arch_results:
            arch_results[model] = {}
        arch_results[model][r["Metric"]] = {
            "with_oss": r["With_OSS"],
            "without_oss": r["Without_OSS"],
            "diff": r["Difference"],
            "better": r["Better"]
        }
    
    # Tabelle Header
    print(f"\n{'Architektur':<18} | {'Metrik':<12} | {'Mit OSS':>10} | {'Ohne OSS':>10} | {'Diff':>10} | {'Besser':<10}")
    print("-" * 85)
    
    for model_type in sorted(arch_results.keys()):
        metrics = arch_results[model_type]
        first = True
        for metric in ["AUC", "Prec_Dev", "Rec_Dev", "Prec_NoDev", "Rec_NoDev"]:
            if metric in metrics:
                m = metrics[metric]
                model_name = model_type if first else ""
                first = False
                print(f"{model_name:<18} | {metric:<12} | {m['with_oss']:>10.4f} | {m['without_oss']:>10.4f} | {m['diff']:>+10.4f} | {m['better']:<10}")
        print("-" * 85)
    
    # OSS-Reduktion Zusammenfassung
    print("\n" + "-" * 85)
    print("OSS-REDUKTION ZUSAMMENFASSUNG:")
    print("-" * 85)
    print(f"{'Architektur':<18} | {'Vor OSS':>12} | {'Nach OSS':>12} | {'Reduktion':>12} | {'Imb. vorher':>12} | {'Imb. nachher':>12}")
    print("-" * 85)
    
    for model_type, stats in all_oss_stats.items():
        if stats:
            before = stats.get("before_oss", {})
            after = stats.get("after_oss", {})
            reduction = stats.get("reduction", {})
            
            n_before = before.get("n_samples", 0)
            n_after = after.get("n_samples", 0)
            red_pct = reduction.get("reduction_percent", 0.0)
            imb_before = before.get("imbalance_ratio", 0.0)
            imb_after = after.get("imbalance_ratio", 0.0)
            
            print(f"{model_type:<18} | {n_before:>12,} | {n_after:>12,} | {red_pct:>11.2f}% | {imb_before:>12.2f} | {imb_after:>12.2f}")
    
    print("#" * 80)


# =========================
# Hauptfunktion
# =========================

def main():
    """
    Hauptfunktion: Vergleicht alle Modelltypen (FFN und LSTM) MIT vs. OHNE OSS.
    
    Das aktuelle Encoding verwendet NUR weekday_features als Trace-Attribute
    (trace_numeric_cols und trace_categorical_cols sind leer).
    
    Dieser Vergleich zeigt:
    - Precision, Recall und AUC pro Architektur
    - OSS-Reduktion (Samples vor/nach, Prozent entfernt)
    - Detaillierte Vergleichstabelle
    """
    import argparse
    import time
    
    parser = argparse.ArgumentParser(
        description="Vergleicht Modelle mit und ohne OSS (One-Sided Selection)"
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
    
    dataset_name = args.dataset if args.dataset else "default"
    
    print("#" * 80)
    print("#" + " " * 20 + "OSS vs. OHNE-OSS VERGLEICH" + " " * 20 + "#")
    print("#" + " " * 15 + "(mit weekday_features, ohne andere Trace-Attribute)" + " " * 8 + "#")
    print("#" * 80)
    print(f"\nDataset: {dataset_name}")
    print(f"Verzeichnis: {processed_dir}")
    
    # Modelle mit normalen Dateien (enthält weekday_features)
    models = [
        ("separate_ffn", "idp_separate_ffn_probs.npz", "idp_separate_ffn_probs_no_oss.npz"),
        ("separate_lstm", "idp_separate_lstm_probs.npz", "idp_separate_lstm_probs_no_oss.npz"),
        ("collective_ffn", "idp_collective_ffn_probs.npz", "idp_collective_ffn_probs_no_oss.npz"),
        ("collective_lstm", "idp_collective_lstm_probs.npz", "idp_collective_lstm_probs_no_oss.npz"),
    ]
    
    all_results: List[Dict[str, any]] = []
    all_oss_stats: Dict[str, Dict[str, any]] = {}
    all_dataset_stats: List[pd.DataFrame] = []
    
    for model_type, probs_file_oss, probs_file_no_oss in models:
        print(f"\n{'#'*80}")
        print(f"# Training: {model_type.upper()}")
        print(f"{'#'*80}")
        
        # 1. Training MIT OSS
        print("\n>>> Training MIT OSS <<<")
        stats_oss = train_model_with_oss_config(model_type, use_oss=True, processed_dir=processed_dir)
        
        time.sleep(1)
        
        # Evaluierung MIT OSS
        try:
            metrics_oss = evaluate_model(processed_dir, probs_file_oss, f"{model_type} (OSS)")
        except Exception as e:
            print(f"Fehler bei Evaluierung mit OSS: {e}")
            metrics_oss = None
        
        # 2. Training OHNE OSS
        print("\n>>> Training OHNE OSS <<<")
        stats_no_oss = train_model_with_oss_config(model_type, use_oss=False, processed_dir=processed_dir)
        
        time.sleep(1)
        
        # Evaluierung OHNE OSS
        try:
            metrics_no_oss = evaluate_model(processed_dir, probs_file_no_oss, f"{model_type} (No OSS)")
        except Exception as e:
            print(f"Fehler bei Evaluierung ohne OSS: {e}")
            metrics_no_oss = None
        
        # OSS-Statistiken sammeln (für collective Modelle)
        oss_reduction_stats = None
        if stats_oss:
            if isinstance(stats_oss, list) and len(stats_oss) > 0:
                # Für separate Modelle: Aggregiere über alle Deviation Types
                # Nehme den ersten als Referenz (alle haben gleiche Sample-Anzahl vor OSS)
                oss_reduction_stats = stats_oss[0]
            else:
                oss_reduction_stats = stats_oss
        
        all_oss_stats[model_type] = oss_reduction_stats
        
        # Ergebnisse anzeigen und sammeln
        if metrics_oss and metrics_no_oss:
            print_architecture_comparison(model_type, metrics_oss, metrics_no_oss, oss_reduction_stats)
            
            for metric_name in ["AUC", "Prec_Dev", "Rec_Dev", "Prec_NoDev", "Rec_NoDev"]:
                diff = metrics_oss[metric_name] - metrics_no_oss[metric_name]
                all_results.append({
                    "Model": model_type,
                    "Metric": metric_name,
                    "With_OSS": metrics_oss[metric_name],
                    "Without_OSS": metrics_no_oss[metric_name],
                    "Difference": diff,
                    "Better": "Mit OSS" if diff > 0.001 else "Ohne OSS" if diff < -0.001 else "Gleich"
                })
        
        # Datensatz-Vergleich
        if stats_oss and stats_no_oss:
            dataset_comparison = analyze_dataset_changes(stats_oss, stats_no_oss, model_type)
            if not dataset_comparison.empty:
                all_dataset_stats.append(dataset_comparison)
    
    # ============================================================
    # Finale Zusammenfassung
    # ============================================================
    if all_results:
        print_final_summary(all_results, all_oss_stats)
        
        # CSV speichern: Metriken
        df_results = pd.DataFrame(all_results)
        csv_path = processed_dir / "oss_comparison_detailed.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\nErgebnisse gespeichert in: {csv_path}")
    
    # CSV speichern: OSS-Reduktion
    if all_oss_stats:
        reduction_rows = []
        for model_type, stats in all_oss_stats.items():
            if stats:
                before = stats.get("before_oss", {})
                after = stats.get("after_oss", {})
                reduction = stats.get("reduction", {})
                
                reduction_rows.append({
                    "Model": model_type,
                    "Samples_Before_OSS": before.get("n_samples", 0),
                    "Samples_After_OSS": after.get("n_samples", 0),
                    "Samples_Removed": reduction.get("samples_removed", 0),
                    "Reduction_Percent": reduction.get("reduction_percent", 0.0),
                    "Imbalance_Before": before.get("imbalance_ratio", 0.0),
                    "Imbalance_After": after.get("imbalance_ratio", 0.0),
                    "Deviating_Before": before.get("n_deviating", 0),
                    "Deviating_After": after.get("n_deviating", 0),
                    "Conforming_Before": before.get("n_conforming", 0),
                    "Conforming_After": after.get("n_conforming", 0),
                })
        
        if reduction_rows:
            df_reduction = pd.DataFrame(reduction_rows)
            reduction_csv_path = processed_dir / "oss_reduction_stats.csv"
            df_reduction.to_csv(reduction_csv_path, index=False)
            print(f"OSS-Reduktionsstatistiken gespeichert in: {reduction_csv_path}")
    
    # Datensatz-Vergleich speichern
    if all_dataset_stats:
        df_dataset = pd.concat(all_dataset_stats, ignore_index=True)
        dataset_csv_path = processed_dir / "training_dataset_comparison.csv"
        df_dataset.to_csv(dataset_csv_path, index=False)
        print(f"Trainingsdatensatz-Vergleich gespeichert in: {dataset_csv_path}")
    
    if not all_results:
        print("\nKeine Ergebnisse zum Vergleichen verfügbar.")


if __name__ == "__main__":
    main()

