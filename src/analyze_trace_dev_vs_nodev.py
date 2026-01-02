from __future__ import annotations

from pathlib import Path

import numpy as np


def main():
    # Projekt- und Datenpfade bestimmen
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    labels_path = processed_dir / "idp_labels.npz"

    if not labels_path.exists():
        print(f"Datei nicht gefunden: {labels_path}")
        return

    print(f"Lade Labels aus {labels_path} ...")
    data = np.load(labels_path, allow_pickle=True)

    # Erwartete Arrays aus labelling.py:
    #   - y:             (N_prefixe, m_dev_types), 0/1
    #   - case_ids:      (N_prefixe,)
    #   - dev_types:     (m_dev_types,)
    #   - prefix_lengths (N_prefixe,) – hier nicht benötigt
    y = data["y"]
    case_ids = data["case_ids"]
    dev_types = data["dev_types"]

    if y.size == 0:
        print("Warnung: y ist leer – keine Prefixe / Traces vorhanden.")
        return

    print("-" * 80)
    print("TRACE-STATISTIK: DEVIATING vs. NON-DEVIATING")
    print("-" * 80)
    print(f"Anzahl Prefixe gesamt: {y.shape[0]}")

    # 1) Dev-Flag pro Prefix:
    #    1, wenn es für diesen Prefix noch mind. eine Abweichung in der Zukunft gibt.
    row_has_dev = (y.sum(axis=1) > 0)

    # 2) Aggregation auf Trace-Ebene:
    #    - trace_dev_flags:  True, falls der Trace mindestens einen deviierenden Prefix hat
    #    - trace_dev_masks:  Bool-Vektor pro Trace, welche Deviation Types im Trace vorkommen
    trace_dev_flags = {}
    trace_dev_masks = {}
    m = y.shape[1]

    for cid, has_dev, row in zip(case_ids, row_has_dev, y):
        # Falls Case noch nicht bekannt, initialisiere
        if cid not in trace_dev_flags:
            trace_dev_flags[cid] = False
        if cid not in trace_dev_masks:
            trace_dev_masks[cid] = np.zeros(m, dtype=bool)

        # Sobald ein Prefix für diesen Case deviating ist, bleibt der Trace deviating
        if bool(has_dev):
            trace_dev_flags[cid] = True

        # Merke, welche Deviation Types in diesem Trace überhaupt vorkommen
        trace_dev_masks[cid] |= (row > 0)

    n_traces = len(trace_dev_flags)
    n_dev_traces = sum(1 for v in trace_dev_flags.values() if v)
    n_nodev_traces = n_traces - n_dev_traces

    if n_traces == 0:
        print("Keine Traces gefunden (n_traces = 0).")
        return

    perc_dev = n_dev_traces / n_traces * 100.0
    perc_nodev = n_nodev_traces / n_traces * 100.0

    print(f"Anzahl Traces gesamt:          {n_traces}")
    print(f"Traces mit Abweichung:         {n_dev_traces} ({perc_dev:.2f}%)")
    print(f"Traces ohne Abweichung:        {n_nodev_traces} ({perc_nodev:.2f}%)")
    print("-" * 80)
    if n_nodev_traces > 0:
        ratio = n_dev_traces / n_nodev_traces
        print(f"Verhältnis Dev : Non-Dev = {ratio:.4f}")
    else:
        print("Alle Traces enthalten mindestens eine Abweichung (kein Non-Dev-Trace).")

    # 3) Häufigkeit der Deviation Types auf Trace-Ebene
    print()
    print("HÄUFIGKEIT DER DEVIATION TYPES (TRACE-BASIERT)")
    print("-" * 80)

    dev_trace_counts = np.zeros(m, dtype=int)
    for mask in trace_dev_masks.values():
        dev_trace_counts += mask.astype(int)

    # Optional: prozentualer Anteil je Typ
    dev_trace_perc = dev_trace_counts / n_traces * 100.0

    # Nach Häufigkeit sortieren (absteigend)
    dev_types_list = list(dev_types)
    stats = list(zip(dev_types_list, dev_trace_counts, dev_trace_perc))
    stats.sort(key=lambda x: x[1], reverse=True)

    # Kopfzeile
    print(f"{'Deviation Type':50s} | {'Traces':>8s} | {'% der Traces':>12s}")
    print("-" * 80)
    for dev_type, count, perc in stats:
        print(f"{str(dev_type):50.50s} | {count:8d} | {perc:12.2f}")

    # 4) Prefix-Level Statistik
    print()
    print("-" * 80)
    print("PREFIX-STATISTIK: DEVIATING vs. NON-DEVIATING")
    print("-" * 80)
    
    n_prefixes = y.shape[0]
    n_dev_prefixes = np.sum(row_has_dev)
    n_nodev_prefixes = n_prefixes - n_dev_prefixes
    
    perc_dev_prefix = n_dev_prefixes / n_prefixes * 100.0
    perc_nodev_prefix = n_nodev_prefixes / n_prefixes * 100.0
    
    print(f"Anzahl Prefixe gesamt:          {n_prefixes}")
    print(f"Prefixe mit Abweichung:         {n_dev_prefixes} ({perc_dev_prefix:.2f}%)")
    print(f"Prefixe ohne Abweichung:        {n_nodev_prefixes} ({perc_nodev_prefix:.2f}%)")
    print("-" * 80)
    if n_nodev_prefixes > 0:
        ratio_prefix = n_dev_prefixes / n_nodev_prefixes
        print(f"Verhältnis Dev : Non-Dev = {ratio_prefix:.4f}")
    else:
        print("Alle Prefixe enthalten mindestens eine Abweichung (kein Non-Dev-Prefix).")

    # 5) Häufigkeit der Deviation Types auf Prefix-Ebene
    print()
    print("HÄUFIGKEIT DER DEVIATION TYPES (PREFIX-BASIERT)")
    print("-" * 80)

    # Pro Deviation Type: Anzahl Prefixe, in denen dieser Typ vorkommt
    dev_prefix_counts = np.sum(y > 0, axis=0)
    dev_prefix_perc = dev_prefix_counts / n_prefixes * 100.0

    # Nach Häufigkeit sortieren (absteigend)
    stats_prefix = list(zip(dev_types_list, dev_prefix_counts, dev_prefix_perc))
    stats_prefix.sort(key=lambda x: x[1], reverse=True)

    # Kopfzeile
    print(f"{'Deviation Type':50s} | {'Prefixe':>8s} | {'% der Prefixe':>14s}")
    print("-" * 80)
    for dev_type, count, perc in stats_prefix:
        print(f"{str(dev_type):50.50s} | {count:8d} | {perc:14.2f}")


if __name__ == "__main__":
    main()


