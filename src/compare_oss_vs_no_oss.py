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
    
    Args:
        processed_dir: Pfad zum processed Verzeichnis
        probs_file: Name der Wahrscheinlichkeits-Datei (z.B. "idp_separate_ffn_probs.npz")
        model_name: Name des Modells (für Fehlermeldungen)
    
    Returns:
        Dictionary mit Macro-Average Metriken
    """
    # Labels laden
    labels_path = processed_dir / "idp_labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {labels_path}")
    
    labels_data = np.load(labels_path, allow_pickle=True)
    y_raw = labels_data["y"]
    case_ids = labels_data["case_ids"]
    dev_types_raw = list(labels_data["dev_types"])

    # Predictions laden
    probs_path = processed_dir / probs_file
    if not probs_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {probs_path}. Bitte erst {model_name} trainieren.")
    
    probs_data = np.load(probs_path, allow_pickle=True)
    P_dev_all = probs_data["P_dev"]
    dev_types_trained = list(probs_data["dev_types"])

    # Ground Truth anpassen (Matching)
    keep_indices = []
    raw_type_to_idx = {name: i for i, name in enumerate(dev_types_raw)}
    
    for dt in dev_types_trained:
        if dt in raw_type_to_idx:
            keep_indices.append(raw_type_to_idx[dt])
    
    y_true_matched = y_raw[:, keep_indices]

    if y_true_matched.shape != P_dev_all.shape:
        raise AssertionError(f"Shape Mismatch! Labels: {y_true_matched.shape}, Preds: {P_dev_all.shape}")

    # Test-Set isolieren
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
                X_trace_train = inputs["X_trace"][train_mask]
                oss = OneSidedSelection(random_state=cfg.random_state, n_seeds_S=250, n_neighbors=7)
                _, _ = oss.fit_resample(X_trace_train, y_train)
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
        try:
            y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
            X_trace_train = inputs["X_trace"][train_mask]
            X_act_train = inputs["X_act"][train_mask]
            X_res_train = inputs["X_res"][train_mask]
            X_month_train = inputs["X_month"][train_mask]
            n_samples_train = X_trace_train.shape[0]
            X_oss_train = np.concatenate(
                [
                    X_act_train.reshape(n_samples_train, -1),
                    X_res_train.reshape(n_samples_train, -1),
                    X_month_train.reshape(n_samples_train, -1),
                    X_trace_train,
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
# Hauptfunktion
# =========================

def main():
    """
    Hauptfunktion: Trainiert alle Modelle mit und ohne OSS und vergleicht die Ergebnisse.
    """
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    
    print("="*80)
    print("OSS vs. Non-OSS Vergleich")
    print("="*80)
    
    # Modelle die verglichen werden sollen
    models = [
        ("separate_ffn", "idp_separate_ffn_probs.npz", "idp_separate_ffn_probs_no_oss.npz"),
        ("separate_lstm", "idp_separate_lstm_probs.npz", "idp_separate_lstm_probs_no_oss.npz"),
        ("collective_ffn", "idp_collective_ffn_probs.npz", "idp_collective_ffn_probs_no_oss.npz"),
        ("collective_lstm", "idp_collective_lstm_probs.npz", "idp_collective_lstm_probs_no_oss.npz"),
    ]
    
    all_results = []
    all_dataset_stats = []
    
    for model_type, probs_file_oss, probs_file_no_oss in models:
        print(f"\n{'#'*80}")
        print(f"Modell: {model_type}")
        print(f"{'#'*80}\n")
        
        # 1. Training mit OSS
        print(">>> Training MIT OSS <<<")
        stats_oss = train_model_with_oss_config(model_type, use_oss=True, processed_dir=processed_dir)
        
        # Statistiken ausgeben
        if stats_oss:
            print_dataset_statistics(stats_oss, use_oss=True)
        
        # Warte kurz, damit Dateien geschrieben werden
        import time
        time.sleep(1)
        
        # Evaluierung mit OSS
        try:
            metrics_oss = evaluate_model(processed_dir, probs_file_oss, f"{model_type} (OSS)")
            print(f"\nErgebnisse MIT OSS:")
            print(f"  AUC: {metrics_oss['AUC']:.4f}")
            print(f"  Prec_Dev: {metrics_oss['Prec_Dev']:.4f}, Rec_Dev: {metrics_oss['Rec_Dev']:.4f}")
            print(f"  Prec_NoDev: {metrics_oss['Prec_NoDev']:.4f}, Rec_NoDev: {metrics_oss['Rec_NoDev']:.4f}")
        except Exception as e:
            print(f"Fehler bei Evaluierung mit OSS: {e}")
            metrics_oss = None
        
        # 2. Training ohne OSS
        print("\n>>> Training OHNE OSS <<<")
        stats_no_oss = train_model_with_oss_config(model_type, use_oss=False, processed_dir=processed_dir)
        
        # Statistiken ausgeben
        if stats_no_oss:
            print_dataset_statistics(stats_no_oss, use_oss=False)
        
        time.sleep(1)
        
        # Evaluierung ohne OSS
        try:
            metrics_no_oss = evaluate_model(processed_dir, probs_file_no_oss, f"{model_type} (No OSS)")
            print(f"\nErgebnisse OHNE OSS:")
            print(f"  AUC: {metrics_no_oss['AUC']:.4f}")
            print(f"  Prec_Dev: {metrics_no_oss['Prec_Dev']:.4f}, Rec_Dev: {metrics_no_oss['Rec_Dev']:.4f}")
            print(f"  Prec_NoDev: {metrics_no_oss['Prec_NoDev']:.4f}, Rec_NoDev: {metrics_no_oss['Rec_NoDev']:.4f}")
        except Exception as e:
            print(f"Fehler bei Evaluierung ohne OSS: {e}")
            metrics_no_oss = None
        
        # Vergleich speichern
        if metrics_oss and metrics_no_oss:
            for metric_name in ["AUC", "Prec_Dev", "Rec_Dev", "Prec_NoDev", "Rec_NoDev"]:
                diff = metrics_oss[metric_name] - metrics_no_oss[metric_name]
                all_results.append({
                    "Model": model_type,
                    "Metric": metric_name,
                    "With_OSS": metrics_oss[metric_name],
                    "Without_OSS": metrics_no_oss[metric_name],
                    "Difference": diff,
                    "Better": "OSS" if diff > 0 else "No OSS" if diff < 0 else "Equal"
                })
        
        # Datensatz-Vergleich
        if stats_oss and stats_no_oss:
            dataset_comparison = analyze_dataset_changes(stats_oss, stats_no_oss, model_type)
            if not dataset_comparison.empty:
                all_dataset_stats.append(dataset_comparison)
    
    # Zusammenfassung
    print("\n" + "="*80)
    print("VERGLEICHS-ZUSAMMENFASSUNG")
    print("="*80)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Tabelle ausgeben
        print("\nDetaillierte Ergebnisse (Metriken):")
        print(df.to_string(index=False))
        
        # CSV speichern
        csv_path = processed_dir / "oss_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nMetriken-Ergebnisse gespeichert in: {csv_path}")
        
        # Zusammenfassung pro Metrik
        print("\n" + "-"*80)
        print("Zusammenfassung pro Metrik:")
        print("-"*80)
        for metric in ["AUC", "Prec_Dev", "Rec_Dev", "Prec_NoDev", "Rec_NoDev"]:
            metric_df = df[df["Metric"] == metric]
            avg_diff = metric_df["Difference"].mean()
            print(f"{metric}: Durchschnittliche Differenz (OSS - No OSS) = {avg_diff:+.4f}")
            if avg_diff > 0:
                print(f"  -> OSS ist im Durchschnitt besser")
            elif avg_diff < 0:
                print(f"  -> No OSS ist im Durchschnitt besser")
            else:
                print(f"  -> Kein Unterschied")
    
    # Datensatz-Vergleich
    if all_dataset_stats:
        print("\n" + "="*80)
        print("TRAININGSDATENSATZ-VERGLEICH")
        print("="*80)
        
        df_dataset = pd.concat(all_dataset_stats, ignore_index=True)
        
        print("\nDetaillierte Trainingsdatensatz-Statistiken:")
        print(df_dataset.to_string(index=False))
        
        # CSV speichern
        dataset_csv_path = processed_dir / "training_dataset_comparison.csv"
        df_dataset.to_csv(dataset_csv_path, index=False)
        print(f"\nTrainingsdatensatz-Vergleich gespeichert in: {dataset_csv_path}")
        
        # Zusammenfassung
        print("\n" + "-"*80)
        print("Zusammenfassung Trainingsdatensatz-Änderungen:")
        print("-"*80)
        if "OSS_Reduction_Percent" in df_dataset.columns:
            avg_reduction_oss = df_dataset["OSS_Reduction_Percent"].mean()
            print(f"Durchschnittliche Reduktion MIT OSS: {avg_reduction_oss:.2f}%")
        if "NoOSS_Reduction_Percent" in df_dataset.columns:
            avg_reduction_no_oss = df_dataset["NoOSS_Reduction_Percent"].mean()
            print(f"Durchschnittliche Reduktion OHNE OSS: {avg_reduction_no_oss:.2f}%")
        
        # JSON auch speichern für detaillierte Analyse
        json_path = processed_dir / "training_dataset_stats.json"
        with open(json_path, 'w') as f:
            json.dump(all_dataset_stats, f, indent=2, default=str)
        print(f"Detaillierte Statistiken (JSON) gespeichert in: {json_path}")
    
    if not all_results and not all_dataset_stats:
        print("Keine Ergebnisse zum Vergleichen verfügbar.")


if __name__ == "__main__":
    main()

