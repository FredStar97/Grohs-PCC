
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
import pm4py


# =========================
# Konfiguration
# =========================

N_ALIGNMENT_RUNS = 1  # Grohs: "We calculate the log-wide alignments 100 times"


@dataclass
class IDPLabelingResult:
    dev_types: List[str]          # Liste aller Abweichungstypen["log:Assign seriousness", "model:Resolve ticket", ...]
    y: np.ndarray                 # Binäres Label Array (Zeilen = Präfixe, Spalten = Abweichungstypen)
    case_ids: np.ndarray          # Case IDs der Präfixe
    prefix_lengths: np.ndarray    # Länge jedes Präfixes (1..n+1)


# =========================
# Hilfsfunktionen
# =========================

def is_real_activity(label) -> bool:
    """
    Prüft, ob ein Label eine echte Aktivität ist.
    ">>" und None sind keine echten Aktivitäten.
    """
    if label is None:
        return False
    if label == ">>":
        return False
    return True


# -------- Log & Modell laden --------

def load_event_log(log_path: str):
    """
    Lädt Event-Log (CSV oder XES) und wandelt es in einen PM4Py-EventLog um.
    
    CSV: Erwartete Spalten (wie im Helpdesk-Log):
      - 'Case ID'
      - 'Activity'
      - 'Complete Timestamp'
    
    XES: Standard XES-Attribute werden automatisch erkannt.
    """
    log_path_obj = Path(log_path)
    
    if log_path_obj.suffix.lower() == '.xes':
        # XES-Format: zu DataFrame, dann zu EventLog (wie bei CSV)
        log_raw = pm4py.read_xes(log_path)
        df = log_converter.apply(log_raw, variant=log_converter.Variants.TO_DATA_FRAME)
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        log = log_converter.apply(df)
        return log
    else:
        # CSV-Format (bestehende Logik)
        df = pd.read_csv(log_path)

        df = df.rename(
            columns={
                "Case ID": "case:concept:name",
                "Activity": "concept:name",
                "Complete Timestamp": "time:timestamp",
            }
        )
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        df = dataframe_utils.convert_timestamp_columns_in_df(df)

        log = log_converter.apply(df)
        return log


def load_helpdesk_log(csv_path: str):
    """
    Lädt finale.csv und wandelt sie in einen PM4Py-EventLog um.
    DEPRECATED: Verwende load_event_log() stattdessen.
    Erwartete Spalten (wie im Helpdesk-Log):
      - 'Case ID'
      - 'Activity'
      - 'Complete Timestamp'
    """
    return load_event_log(csv_path)


def load_helpdesk_model(bpmn_path: str):
    """
    Lädt das BPMN-Modell und konvertiert es in ein Petri-Netz inkl. Markings.
    """
    bpmn_graph = bpmn_importer.apply(bpmn_path)
    net, im, fm = bpmn_converter.apply(bpmn_graph)
    return net, im, fm


# -------- Deviation Types bestimmen --------

def extract_dev_types_from_alignments(alignments) -> Set[str]:
    """
    Sammelt alle unterschiedlichen Abweichungstypen aus den Alignments.
    Log Move: Aktivität nur im Log
    Model Move: Aktivität nur im Modell
    Gibt die Menge aller unterschiedlichen Abweichungstypen zurück.
    """
    dev_types: Set[str] = set()
    for aln in alignments:
        for log_mv, model_mv in aln["alignment"]:
            log_real = is_real_activity(log_mv)
            model_real = is_real_activity(model_mv)

            # Log Move: (ac, >>)
            if log_real and not model_real:
                dev_types.add(f"log:{log_mv}")

            # Model Move: (>>, ac)
            if model_real and not log_real:
                dev_types.add(f"model:{model_mv}")

    return dev_types


def compute_best_alignments(log, net, im, fm, n_runs: int = N_ALIGNMENT_RUNS):
    """
    Führt die Alignments 100-mal aus und wählt die Konfiguration
    mit der kleinsten Anzahl unterschiedlicher Deviation Types |D_{L,B}|.
    Gibt die besten Alignments und die sortierte Liste der Abweichungstypen zurück.
    """
    best_alignments = None
    best_dev_types: Set[str] | None = None

    for r in range(n_runs):
        print(f"[Alignments] Run {r + 1}/{n_runs} ...", flush=True)
        alignments = pm4py.conformance.conformance_diagnostics_alignments(
            log, net, im, fm, multi_processing=False
        )

        dev_types_run = extract_dev_types_from_alignments(alignments)
        print(f"  -> {len(dev_types_run)} deviation types in this run")

        if best_dev_types is None or len(dev_types_run) < len(best_dev_types):
            best_dev_types = dev_types_run
            best_alignments = alignments
            print("  -> New best run selected.")

    dev_types_sorted = sorted(best_dev_types) if best_dev_types is not None else []
    return best_alignments, dev_types_sorted


# -------- Positionen der Abweichungen pro Trace --------

def compute_last_positions_for_trace(alignment, n_events: int) -> Dict[str, int]:
    """
    Bestimmt die letzte Position, an der ein Abweichungstyp im Trace auftritt. (kann sein das Abweichungstyp mehrfach im Trace auftritt)
    Log Move: Position = Index des Events im Trace
    Model Move: Position = Index des nächsten Events im Trace
    Gibt ein Dictionary zurück, das für jeden Abweichungstyp die letzte Position enthält.
    """
    last_pos: Dict[str, int] = {}
    log_pos = 0  # wie viele echte Log-Events wurden schon "verbraucht"?

    for log_mv, model_mv in alignment["alignment"]:
        log_real = is_real_activity(log_mv)
        model_real = is_real_activity(model_mv)

        # Log-Teil verbraucht ein Event (sowohl bei Sync als auch bei Log Move)
        if log_real:
            log_pos += 1

        # Log Move: Aktivität nur im Log
        if log_real and not model_real:
            dev_type = f"log:{log_mv}"
            last_pos[dev_type] = log_pos  # aktueller Event-Index im Trace

        # Model Move: Aktivität nur im Modell
        if model_real and not log_real:
            # "Soll-Ereignis" nach dem aktuell letzten gesehenen Log-Event
            pos = min(log_pos + 1, n_events + 1)
            dev_type = f"model:{model_mv}"
            last_pos[dev_type] = pos

    return last_pos


# -------- Dynamisches Prefix-Labeling (IDP) --------

def build_idp_labels(log, alignments, dev_types: List[str]) -> IDPLabelingResult:
    """
    Erstellt Labels für jeden Präfix jedes Traces 
    Für jeden Trace:
        Bestimme die last_positions für jeden Abweichungstyp
        Erstellt ein Array (n+1, m) (n+1 = Länge des Präfixes, m = Anzahl der Abweichungstypen)
        Setzt label(k, d) = 1, falls pos(d) > k, sonst 0. (Abweichung liegt in der Zukunft)
        Für jeden Präfix werden Label-Vektor, Case ID und Präfix-Länge gespeichert
        Gibt ein IDPLabelingResult-Objekt zurück
    """
    m = len(dev_types)
    dev_index = {d: i for i, d in enumerate(dev_types)}

    all_labels: List[np.ndarray] = []
    case_ids: List[str] = []
    prefix_lengths: List[int] = []

    for trace, alignment in zip(log, alignments):
        case_id = trace.attributes.get("concept:name") or trace.attributes.get(
            "case:concept:name", ""
        )
        n = len(trace)
        if n == 0:
            continue

        last_positions = compute_last_positions_for_trace(alignment, n)
        # n+1 Prefixe: Längen 1..n+1 (der vollständige Trace ist Prefix n+1)
        labels_trace = np.zeros((n + 1, m), dtype=np.int8)

        # Fülle für jeden Deviation Type die 1er-Blöcke (Prefixe, in deren Zukunft d liegt)
        for dev, pos in last_positions.items():
            idx = dev_index.get(dev)
            if idx is None:
                continue
            # bis Prefix (pos-1) ist d noch in der Zukunft
            # Bei pos = n+1: alle Prefixe 1..n+1 erhalten 1
            cutoff = min(max(pos - 1, 0), n + 1)
            if cutoff > 0:
                labels_trace[:cutoff, idx] = 1

        # Jetzt Prefix-Metadaten sammeln
        for k in range(n + 1):  # k = 0..n -> Prefix-Länge = k+1 (1..n+1)
            all_labels.append(labels_trace[k])
            case_ids.append(case_id)
            prefix_lengths.append(k + 1)

    if all_labels:
        y = np.vstack(all_labels)
    else:
        y = np.zeros((0, m), dtype=np.int8)

    return IDPLabelingResult(
        dev_types=dev_types,
        y=y,
        case_ids=np.array(case_ids, dtype=object),
        prefix_lengths=np.array(prefix_lengths, dtype=np.int32),
    )


# -------- Hauptfunktion zum Ausführen als Skript --------
"""
Pfade setzen.
Log und Modell laden.
Beste Alignments berechnen.
Labels erstellen.
Ergebnisse speichern (NPZ und JSON).
"""

def main():
    parser = argparse.ArgumentParser(
        description="Erstellt IDP-Labels aus Event-Log und BPMN-Modell"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Pfad zum Event-Log (CSV oder XES). Standard: data/raw/finale.csv"
    )
    parser.add_argument(
        "--bpmn",
        type=str,
        default=None,
        help="Pfad zum BPMN-Modell. Standard: data/raw/Helpdesk.bpmn"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name des Datensatzes (für Output-Verzeichnis). Standard: processed/ (rückwärtskompatibel)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output-Verzeichnis. Standard: data/processed/ oder data/processed/{dataset}/"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    
    # Pfade bestimmen (rückwärtskompatibel)
    if args.log is None:
        # Automatische Datei-Auswahl basierend auf Dataset
        if args.dataset == "DomesticDeclarations":
            log_path = raw_dir / "DomesticDeclarations.xes"
        else:
            # Standard: Helpdesk
            log_path = raw_dir / "finale.csv"
    else:
        log_path = Path(args.log)
    
    if args.bpmn is None:
        # Automatische Datei-Auswahl basierend auf Dataset
        if args.dataset == "DomesticDeclarations":
            bpmn_path = raw_dir / "Model_ DomesticDeclarations.bpmn"
        else:
            # Standard: Helpdesk
            bpmn_path = raw_dir / "Helpdesk.bpmn"
    else:
        bpmn_path = Path(args.bpmn)
    
    # Output-Verzeichnis bestimmen
    if args.output_dir is not None:
        processed_dir = Path(args.output_dir)
    elif args.dataset is not None:
        processed_dir = project_root / "data" / "processed" / args.dataset
    else:
        # Rückwärtskompatibel: Standard-Verzeichnis
        processed_dir = project_root / "data" / "processed"
    
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade Event Log aus {log_path} ...")
    log = load_event_log(str(log_path))

    print(f"Lade BPMN-Modell aus {bpmn_path} ...")
    net, im, fm = load_helpdesk_model(str(bpmn_path))

    print("Berechne Alignments und wähle Run mit minimalem |D_{L,B}| ...")
    alignments, dev_types = compute_best_alignments(log, net, im, fm, N_ALIGNMENT_RUNS)
    print(f"Fixierte Deviation Types (m = {len(dev_types)}):")
    print(dev_types)

    print("Erzeuge dynamische Prefix-Labels (IDP) ...")
    result = build_idp_labels(log, alignments, dev_types)

    out_path = processed_dir / "idp_labels.npz"
    np.savez_compressed(
        out_path,
        y=result.y,
        case_ids=result.case_ids,
        prefix_lengths=result.prefix_lengths,
        dev_types=np.array(result.dev_types, dtype=object),
    )

    # Zusätzlich optional dev_types auch als JSON speichern
    with open(processed_dir / "idp_dev_types.json", "w", encoding="utf-8") as f:
        json.dump(result.dev_types, f, indent=2, ensure_ascii=False)

    print(
        f"Fertig. Gespeichert nach {out_path}:\n"
        f"  #Prefixe: {result.y.shape[0]}\n"
        f"  #Deviation Types (m): {result.y.shape[1]}"
    )
    


if __name__ == "__main__":
    main()
