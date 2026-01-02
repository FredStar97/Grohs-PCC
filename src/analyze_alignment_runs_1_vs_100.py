from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from labelling import (
    load_helpdesk_log,
    load_helpdesk_model,
    compute_best_alignments,
    build_idp_labels,
    compute_last_positions_for_trace,
    is_real_activity,
)


def _load_labels_100run(processed_dir: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Lädt die vorhandenen IDP-Labels aus dem Standardlauf (100 Alignments),
    wie sie von labelling.py erzeugt wurden.
    """
    npz_path = processed_dir / "idp_labels.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Erwartete Datei {npz_path} wurde nicht gefunden. "
            f"Bitte zuerst labelling.py mit N_ALIGNMENT_RUNS = 100 ausführen."
        )

    data = np.load(npz_path, allow_pickle=True)
    dev_types_100 = data["dev_types"].tolist()
    y_100 = data["y"]
    case_ids_100 = data["case_ids"]
    prefix_lengths_100 = data["prefix_lengths"]

    return dev_types_100, y_100, case_ids_100, prefix_lengths_100


def _extract_trace_info(log, case_id: str) -> List[Dict[str, Any]]:
    """
    Extrahiert den vollständigen Trace für einen gegebenen Case ID.
    Gibt eine Liste von Events zurück mit Activity und Timestamp.
    """
    for trace in log:
        trace_case_id = trace.attributes.get("concept:name") or trace.attributes.get(
            "case:concept:name", ""
        )
        if str(trace_case_id) == str(case_id):
            events = []
            for event in trace:
                activity = event.get("concept:name", "")
                timestamp = event.get("time:timestamp", "")
                events.append({
                    "activity": activity,
                    "timestamp": str(timestamp) if timestamp else "",
                })
            return events
    return []


def _compare_alignments_for_case(
    log,
    case_id: str,
    dev_type: str,
    prefix_length: int,
    alignments_1: List[Dict[str, Any]],
    alignments_100: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Vergleicht die Alignments für einen Case zwischen 1 Run und 100 Runs.
    Gibt detaillierte Informationen zurück über die Unterschiede.
    """
    # Finde Trace und Alignments
    trace = None
    alignment_1 = None
    alignment_100 = None
    
    for t, a1, a100 in zip(log, alignments_1, alignments_100):
        trace_case_id = t.attributes.get("concept:name") or t.attributes.get(
            "case:concept:name", ""
        )
        if str(trace_case_id) == str(case_id):
            trace = t
            alignment_1 = a1
            alignment_100 = a100
            break
    
    if trace is None or alignment_1 is None or alignment_100 is None:
        return {"error": "Case nicht gefunden"}
    
    n_events = len(trace)
    
    # Berechne Positionen für beide Alignments
    last_pos_1 = compute_last_positions_for_trace(alignment_1, n_events)
    last_pos_100 = compute_last_positions_for_trace(alignment_100, n_events)
    
    # Extrahiere Positionen für den betroffenen Deviation Type
    pos_1 = last_pos_1.get(dev_type)
    pos_100 = last_pos_100.get(dev_type)
    
    # Bestimme Label-Status für beide Runs
    # Label = 1 wenn pos > prefix_length, sonst 0
    label_1 = 1 if (pos_1 is not None and pos_1 > prefix_length) else 0
    label_100 = 1 if (pos_100 is not None and pos_100 > prefix_length) else 0
    
    # Erstelle Alignment-Vergleich
    alignment_comparison = []
    log_pos_1 = 0
    log_pos_100 = 0
    
    # Extrahiere Alignment-Sequenzen
    # PM4Py Alignments haben die Struktur: {"alignment": [(log_move, model_move), ...], ...}
    alignment_seq_1 = alignment_1.get("alignment", []) if isinstance(alignment_1, dict) else []
    alignment_seq_100 = alignment_100.get("alignment", []) if isinstance(alignment_100, dict) else []
    
    # Vergleiche die Alignments Schritt für Schritt
    max_len = max(len(alignment_seq_1), len(alignment_seq_100))
    for i in range(max_len):
        log_mv_1 = alignment_seq_1[i] if i < len(alignment_seq_1) else (">>", ">>")
        log_mv_100 = alignment_seq_100[i] if i < len(alignment_seq_100) else (">>", ">>")
        
        log_real_1 = is_real_activity(log_mv_1[0])
        model_real_1 = is_real_activity(log_mv_1[1])
        log_real_100 = is_real_activity(log_mv_100[0])
        model_real_100 = is_real_activity(log_mv_100[1])
        
        if log_real_1:
            log_pos_1 += 1
        if log_real_100:
            log_pos_100 += 1
        
        # Prüfe ob dieser Schritt relevant für den Deviation Type ist
        is_relevant = False
        if log_real_1 and not model_real_1 and f"log:{log_mv_1[0]}" == dev_type:
            is_relevant = True
        if model_real_1 and not log_real_1 and f"model:{log_mv_1[1]}" == dev_type:
            is_relevant = True
        if log_real_100 and not model_real_100 and f"log:{log_mv_100[0]}" == dev_type:
            is_relevant = True
        if model_real_100 and not log_real_100 and f"model:{log_mv_100[1]}" == dev_type:
            is_relevant = True
        
        alignment_comparison.append({
            "step": i + 1,
            "log_pos_1run": log_pos_1,
            "log_pos_100run": log_pos_100,
            "alignment_1": {
                "log_move": log_mv_1[0] if log_mv_1[0] != ">>" else None,
                "model_move": log_mv_1[1] if log_mv_1[1] != ">>" else None,
            },
            "alignment_100": {
                "log_move": log_mv_100[0] if log_mv_100[0] != ">>" else None,
                "model_move": log_mv_100[1] if log_mv_100[1] != ">>" else None,
            },
            "is_relevant": is_relevant,
        })
    
    # Erstelle Erklärung
    explanation = []
    if pos_1 is None and pos_100 is None:
        explanation.append("Deviation Type tritt in beiden Runs nicht auf.")
    elif pos_1 is None:
        explanation.append(f"Deviation Type tritt nur im 100-Run auf (Position {pos_100}).")
    elif pos_100 is None:
        explanation.append(f"Deviation Type tritt nur im 1-Run auf (Position {pos_1}).")
    else:
        if pos_1 != pos_100:
            explanation.append(
                f"Deviation Type tritt an unterschiedlichen Positionen auf: "
                f"1-Run: Position {pos_1}, 100-Run: Position {pos_100}."
            )
        else:
            explanation.append(
                f"Deviation Type tritt an gleicher Position auf: Position {pos_1}."
            )
    
    if label_1 != label_100:
        explanation.append(
            f"Label-Unterschied: Bei Prefix-Länge {prefix_length} ist "
            f"Label im 1-Run {label_1} (Position {pos_1} {'>' if pos_1 else 'N/A'} {prefix_length}) "
            f"und im 100-Run {label_100} (Position {pos_100} {'>' if pos_100 else 'N/A'} {prefix_length})."
        )
    
    return {
        "case_id": case_id,
        "deviation_type": dev_type,
        "prefix_length": prefix_length,
        "position_1run": pos_1,
        "position_100run": pos_100,
        "label_1run": label_1,
        "label_100run": label_100,
        "explanation": " ".join(explanation),
        "alignment_comparison": alignment_comparison,
    }


def _build_labels_1run(
    log,
    net,
    im,
    fm,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Führt die Alignments genau 1x durch und erzeugt darauf basierend IDP-Labels.
    Gibt auch die Alignments zurück für spätere Analyse.
    """
    alignments_1, dev_types_1 = compute_best_alignments(log, net, im, fm, n_runs=1)
    result_1 = build_idp_labels(log, alignments_1, dev_types_1)
    return (
        result_1.dev_types,
        result_1.y,
        result_1.case_ids,
        result_1.prefix_lengths,
        alignments_1,
    )


def _compare_labelings(
    dev_types_1: List[str],
    y_1: np.ndarray,
    case_ids_1: np.ndarray,
    prefix_lengths_1: np.ndarray,
    dev_types_100: List[str],
    y_100: np.ndarray,
    case_ids_100: np.ndarray,
    prefix_lengths_100: np.ndarray,
) -> Dict[str, Any]:
    """
    Vergleicht die Labelings von 1x vs. 100x Alignments.
    """
    set_1 = set(dev_types_1)
    set_100 = set(dev_types_100)

    common = sorted(set_1 & set_100)
    only_1 = sorted(set_1 - set_100)
    only_100 = sorted(set_100 - set_1)

    jaccard = (
        len(common) / len(set_1 | set_100) if (set_1 or set_100) else 1.0
    )

    same_prefix_order = bool(
        np.array_equal(case_ids_1, case_ids_100)
        and np.array_equal(prefix_lengths_1, prefix_lengths_100)
    )

    summary: Dict[str, Any] = {
        "n_dev_types_1run": len(dev_types_1),
        "n_dev_types_100run": len(dev_types_100),
        "n_common_dev_types": len(common),
        "dev_types_only_1run": only_1,
        "dev_types_only_100run": only_100,
        "jaccard_dev_types": jaccard,
        "n_prefixes_1run": int(len(case_ids_1)),
        "n_prefixes_100run": int(len(case_ids_100)),
        "same_prefix_order": same_prefix_order,
    }

    # --- Zusatz: Häufigkeit der einzelnen Abweichungstypen (Anzahl positiver Labels) ---
    idx_1 = {d: i for i, d in enumerate(dev_types_1)}
    idx_100 = {d: i for i, d in enumerate(dev_types_100)}

    n_prefixes_1 = int(y_1.shape[0])
    n_prefixes_100 = int(y_100.shape[0])

    per_dev_counts: Dict[str, Any] = {}
    all_devs_sorted = sorted(set_1 | set_100)
    for dev in all_devs_sorted:
        # Anzahl positiver Labels (1en) pro Abweichungstyp und Run
        pos_1 = int(np.sum(y_1[:, idx_1[dev]])) if dev in idx_1 else 0
        pos_100 = int(np.sum(y_100[:, idx_100[dev]])) if dev in idx_100 else 0

        ratio_1 = (pos_1 / n_prefixes_1) if (n_prefixes_1 > 0 and dev in idx_1) else 0.0
        ratio_100 = (pos_100 / n_prefixes_100) if (n_prefixes_100 > 0 and dev in idx_100) else 0.0

        per_dev_counts[dev] = {
            "present_in_1run": dev in idx_1,
            "present_in_100run": dev in idx_100,
            "positives_1run": pos_1,
            "positives_100run": pos_100,
            "positive_ratio_1run": ratio_1,
            "positive_ratio_100run": ratio_100,
        }

    summary["per_dev_type_label_counts"] = per_dev_counts

    if not (same_prefix_order and common):
        return summary

    # Labelvergleich nur für gemeinsame Deviation Types
    cols_1 = [idx_1[d] for d in common]
    cols_100 = [idx_100[d] for d in common]

    y1_common = y_1[:, cols_1]
    y100_common = y_100[:, cols_100]

    eq_mask = (y1_common == y100_common)
    total = int(y1_common.size)
    equal = int(eq_mask.sum())
    diff = total - equal

    summary.update(
        {
            "labels_total_common": total,
            "labels_equal_common": equal,
            "labels_diff_common": diff,
            "labels_equal_ratio_common": (equal / total) if total > 0 else 1.0,
        }
    )

    # Optional: pro Deviation Type die Übereinstimmung
    per_dev_stats: Dict[str, Any] = {}
    divergent_cases: Dict[str, List[Dict[str, Any]]] = {}
    n_prefixes = int(y1_common.shape[0])
    for j, dev in enumerate(common):
        col_equal = int(np.sum(y1_common[:, j] == y100_common[:, j]))
        per_dev_stats[dev] = {
            "labels_equal": col_equal,
            "labels_total": n_prefixes,
            "labels_equal_ratio": (col_equal / n_prefixes) if n_prefixes > 0 else 1.0,
        }
        
        # Identifiziere divergente Präfixe für diesen Deviation Type
        diff_mask = y1_common[:, j] != y100_common[:, j]
        if np.any(diff_mask):
            divergent_prefixes = []
            diff_indices = np.where(diff_mask)[0]
            for idx in diff_indices:
                divergent_prefixes.append({
                    "case_id": str(case_ids_1[idx]),
                    "prefix_length": int(prefix_lengths_1[idx]),
                    "label_1run": int(y1_common[idx, j]),
                    "label_100run": int(y100_common[idx, j]),
                })
            divergent_cases[dev] = divergent_prefixes

    summary["per_dev_type_common"] = per_dev_stats
    summary["divergent_cases"] = divergent_cases
    return summary


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "finale.csv"
    bpmn_path = raw_dir / "Helpdesk.bpmn"

    print(f"Lade Event Log aus {csv_path} ...")
    log = load_helpdesk_log(str(csv_path))

    print(f"Lade BPMN-Modell aus {bpmn_path} ...")
    net, im, fm = load_helpdesk_model(str(bpmn_path))

    print("Lade bestehende IDP-Labels (100 Alignment-Runs) ...")
    dev_types_100, y_100, case_ids_100, prefix_lengths_100 = _load_labels_100run(
        processed_dir
    )

    print("Berechne IDP-Labels mit nur 1 Alignment-Run ...")
    dev_types_1, y_1, case_ids_1, prefix_lengths_1, alignments_1 = _build_labels_1run(
        log, net, im, fm
    )
    
    print("Berechne Alignments für 100 Runs (für detaillierte Analyse) ...")
    alignments_100, dev_types_100_verify = compute_best_alignments(log, net, im, fm, n_runs=100)

    print("Vergleiche 1x- vs. 100x-Alignments ...")
    summary = _compare_labelings(
        dev_types_1,
        y_1,
        case_ids_1,
        prefix_lengths_1,
        dev_types_100,
        y_100,
        case_ids_100,
        prefix_lengths_100,
    )

    out_path = processed_dir / "idp_alignment_runs_1_vs_100_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Fertig. Zusammenfassung gespeichert nach {out_path}")
    
    # Speichere divergente Cases in JSON und CSV mit detaillierten Informationen
    divergent_cases = summary.get("divergent_cases", {})
    if divergent_cases:
        # Erweitere divergente Cases um detaillierte Informationen
        detailed_divergent_cases = {}
        
        for dev_type, prefixes in divergent_cases.items():
            detailed_prefixes = []
            for prefix_info in prefixes:
                case_id = prefix_info["case_id"]
                prefix_length = prefix_info["prefix_length"]
                
                # Extrahiere Trace-Informationen
                trace_events = _extract_trace_info(log, case_id)
                trace_prefix = trace_events[:prefix_length] if trace_events else []
                
                # Vergleiche Alignments
                alignment_comparison = _compare_alignments_for_case(
                    log,
                    case_id,
                    dev_type,
                    prefix_length,
                    alignments_1,
                    alignments_100,
                )
                
                detailed_prefix = {
                    **prefix_info,
                    "trace_events": trace_events,
                    "trace_prefix": trace_prefix,
                    "alignment_comparison": alignment_comparison,
                }
                detailed_prefixes.append(detailed_prefix)
            
            detailed_divergent_cases[dev_type] = detailed_prefixes
        
        # JSON-Export
        divergent_json_path = processed_dir / "idp_alignment_runs_1_vs_100_divergent_cases.json"
        with open(divergent_json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_divergent_cases, f, indent=2, ensure_ascii=False)
        print(f"Divergente Cases (JSON) gespeichert nach {divergent_json_path}")
        
        # CSV-Export: Flache Liste aller divergenten Präfixe
        csv_rows = []
        for dev_type, prefixes in divergent_cases.items():
            for prefix_info in prefixes:
                csv_rows.append({
                    "deviation_type": dev_type,
                    "case_id": prefix_info["case_id"],
                    "prefix_length": prefix_info["prefix_length"],
                    "label_1run": prefix_info["label_1run"],
                    "label_100run": prefix_info["label_100run"],
                })
        
        if csv_rows:
            df_divergent = pd.DataFrame(csv_rows)
            divergent_csv_path = processed_dir / "idp_alignment_runs_1_vs_100_divergent_cases.csv"
            df_divergent.to_csv(divergent_csv_path, index=False, encoding="utf-8")
            print(f"Divergente Cases (CSV) gespeichert nach {divergent_csv_path}")
    else:
        print("Keine divergenten Cases gefunden.")
    print("Wichtige Kennzahlen:")
    print(f"  #Dev-Types (1 Run):   {summary['n_dev_types_1run']}")
    print(f"  #Dev-Types (100 Run): {summary['n_dev_types_100run']}")
    print(f"  Jaccard(D_1, D_100):  {summary['jaccard_dev_types']:.4f}")
    if "labels_equal_ratio_common" in summary:
        print(
            f"  Anteil identischer Labels (nur gemeinsame Dev-Types): "
            f"{summary['labels_equal_ratio_common']:.4f}"
        )
    print(
        "  Detail pro Abweichungstyp (Anzahl/Anteil positiver Labels in 1 vs. 100 Runs): "
        "siehe Feld 'per_dev_type_label_counts' in der JSON-Datei."
    )
    print()
    print("Anzahl der Abweichungen pro Abweichungstyp:")
    print("-" * 80)
    
    per_dev_counts = summary.get("per_dev_type_label_counts", {})
    if per_dev_counts:
        # Sortiere nach Abweichungstyp für konsistente Ausgabe
        sorted_devs = sorted(per_dev_counts.keys())
        
        # Header
        print(f"{'Abweichungstyp':<50} {'1x Run':>10} {'100x Run':>10} {'Differenz':>10}")
        print("-" * 80)
        
        for dev in sorted_devs:
            counts = per_dev_counts[dev]
            pos_1 = counts.get("positives_1run", 0)
            pos_100 = counts.get("positives_100run", 0)
            diff = pos_1 - pos_100
            
            # Kürze sehr lange Abweichungstyp-Namen
            dev_display = dev if len(dev) <= 48 else dev[:45] + "..."
            
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            print(f"{dev_display:<50} {pos_1:>10} {pos_100:>10} {diff_str:>10}")
    else:
        print("  Keine Daten verfügbar.")
    
    # Zeige Details zu divergenten Cases
    divergent_cases = summary.get("divergent_cases", {})
    if divergent_cases:
        print()
        print("=" * 80)
        print("Details zu divergenten Cases:")
        print("=" * 80)
        
        total_divergent = sum(len(prefixes) for prefixes in divergent_cases.values())
        print(f"Gesamtanzahl divergenter Präfixe: {total_divergent}")
        print()
        
        for dev_type in sorted(divergent_cases.keys()):
            prefixes = divergent_cases[dev_type]
            print(f"\n{'=' * 80}")
            print(f"Abweichungstyp: {dev_type}")
            print(f"Anzahl divergenter Präfixe: {len(prefixes)}")
            print(f"{'=' * 80}")
            
            # Zeige detaillierte Informationen für die ersten 3 Beispiele
            for i, prefix_info in enumerate(prefixes[:3]):
                case_id = prefix_info["case_id"]
                prefix_length = prefix_info["prefix_length"]
                
                print(f"\n--- Beispiel {i + 1} ---")
                print(f"Case ID: {case_id}")
                print(f"Prefix-Länge: {prefix_length}")
                print(f"Label (1 Run): {prefix_info['label_1run']}, Label (100 Runs): {prefix_info['label_100run']}")
                
                # Extrahiere Trace-Informationen
                trace_events = _extract_trace_info(log, case_id)
                if trace_events:
                    print(f"\nVollständiger Trace ({len(trace_events)} Events):")
                    for j, event in enumerate(trace_events):
                        marker = " <-- Prefix Ende" if j == prefix_length - 1 else ""
                        print(f"  {j + 1}. {event['activity']}{marker}")
                    
                    print(f"\nPrefix (erste {prefix_length} Events):")
                    for j, event in enumerate(trace_events[:prefix_length]):
                        print(f"  {j + 1}. {event['activity']}")
                
                # Vergleiche Alignments
                alignment_comparison = _compare_alignments_for_case(
                    log,
                    case_id,
                    dev_type,
                    prefix_length,
                    alignments_1,
                    alignments_100,
                )
                
                if "error" not in alignment_comparison:
                    print(f"\nAlignment-Vergleich:")
                    print(f"  Position (1 Run): {alignment_comparison.get('position_1run', 'N/A')}")
                    print(f"  Position (100 Runs): {alignment_comparison.get('position_100run', 'N/A')}")
                    print(f"  Erklärung: {alignment_comparison.get('explanation', 'N/A')}")
                    
                    # Zeige relevante Alignment-Schritte
                    relevant_steps = [
                        step for step in alignment_comparison.get("alignment_comparison", [])
                        if step.get("is_relevant", False)
                    ]
                    if relevant_steps:
                        print(f"\n  Relevante Alignment-Schritte für '{dev_type}':")
                        for step in relevant_steps[:5]:  # Zeige max. 5 relevante Schritte
                            log_pos_1 = step.get('log_pos_1run', 'N/A')
                            log_pos_100 = step.get('log_pos_100run', 'N/A')
                            print(f"    Schritt {step['step']} (Log-Pos: 1-Run={log_pos_1}, 100-Run={log_pos_100}):")
                            print(f"      1-Run:  Log={step['alignment_1']['log_move']}, Model={step['alignment_1']['model_move']}")
                            print(f"      100-Run: Log={step['alignment_100']['log_move']}, Model={step['alignment_100']['model_move']}")
            
            if len(prefixes) > 3:
                print(f"\n... und {len(prefixes) - 3} weitere divergente Präfixe")
                print("  (Detaillierte Informationen für alle Cases in JSON-Datei verfügbar)")
    else:
        print()
        print("Keine divergenten Cases gefunden - alle Labels sind identisch.")


if __name__ == "__main__":
    main()


