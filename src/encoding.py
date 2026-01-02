from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter
import pm4py


# ============================================================
# Konfiguration / Spaltennamen
# ============================================================

CASE_COL = "Case ID"
ACT_COL = "Activity"
RES_COL = "Resource"
TIME_COL = "Complete Timestamp"
MONTH_COL = "Month"  # wird aus TIME_COL abgeleitet


# ============================================================
# Metadaten-Struktur für das Encoding
# ============================================================

@dataclass
class EncodingMeta:
    max_prefix_len: int # maximale Länge eines Präfixes
    activity_vocab: Dict[str, int] # Vokabular für Activities       
    resource_vocab: Dict[str, int] # Vokabular für Resources
    month_vocab: Dict[int, int] # Vokabular für Months
    trace_numeric_cols: List[str] # Liste der numerischen Trace-Attribute
    trace_categorical_cols: List[str] # Liste der kategorialen Trace-Attribute
    trace_cat_vocabs: Dict[str, Dict[str, int]] # Vokabulare für kategoriale Trace-Attribute


# ============================================================
# Roh-Log laden und vorbereiten
# ============================================================

def load_event_log_to_df(log_path: str) -> pd.DataFrame:
    """
    Lädt Event-Log (CSV oder XES) und konvertiert es zu einem DataFrame.
    
    CSV: Direktes Laden mit erwarteten Spalten.
    XES: Konvertierung von PM4Py EventLog zu DataFrame mit Standard-Spalten.
    
    1. Log laden (CSV oder XES)
    2. Prüfen, dass Case ID, Activity, Complete Timestamp vorhanden sind
    3. Prüfen, dass Resource vorhanden ist
    4. Timestamp parsen und Month ableiten
    5. Nach Case ID und Timestamp sortieren
    """
    log_path_obj = Path(log_path)
    
    if log_path_obj.suffix.lower() == '.xes':
        # XES-Format: zu EventLog, dann zu DataFrame
        log = pm4py.read_xes(log_path)
        df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        
        # XES verwendet Standard-Attribute, umbenennen zu unseren Spaltennamen
        # XES: concept:name (Trace) = Case ID, concept:name (Event) = Activity
        # XES: time:timestamp = Complete Timestamp
        # XES: org:resource = Resource
        
        # PM4Py konvertiert bereits zu: case:concept:name, concept:name, time:timestamp
        # Wir müssen nur noch umbenennen und Resource prüfen
        df = df.rename(columns={
            "case:concept:name": CASE_COL,
            "concept:name": ACT_COL,
            "time:timestamp": TIME_COL,
        })
        
        # Resource sollte als org:resource vorhanden sein
        if "org:resource" in df.columns:
            df = df.rename(columns={"org:resource": RES_COL})
        elif RES_COL not in df.columns:
            raise ValueError(f"Required event attribute column '{RES_COL}' (org:resource) not found in XES log {log_path}")
    else:
        # CSV-Format (bestehende Logik)
        df = pd.read_csv(log_path)

    # Sicherstellen, dass die benötigten Spalten existieren
    required_cols = {CASE_COL, ACT_COL, TIME_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {log_path}: {missing}")

    # Resource ist laut Paper ein Event-Attribut; hier erzwingen wir sie ebenfalls
    if RES_COL not in df.columns:
        raise ValueError(f"Required event attribute column '{RES_COL}' not found in {log_path}")

    # Timestamp parsen und Month-Feature erzeugen
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[MONTH_COL] = df[TIME_COL].dt.month.astype("int16")

    # Nach Case und Zeit sortieren, damit die Event-Reihenfolge eindeutig ist
    df = df.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)
    return df


def load_raw_df(csv_path: str) -> pd.DataFrame:
    """
    Lädt CSV und konvertiert es zu DataFrame.
    DEPRECATED: Verwende load_event_log_to_df() stattdessen.
    1. CSV laden
    2. Prüfen, dass Case ID, Activity, Complete Timestamp vorhanden sind
    3. Prüfen, dass Resource vorhanden ist
    4. Timestamp parsen und Month ableiten
    5. Nach Case ID und Timestamp sortieren
    """
    return load_event_log_to_df(csv_path)


# ============================================================
# Trace-Attribute erkennen (alle Attribute, die pro Case konstant sind)
# ============================================================
# Spalten, die bei Trace-Attribute-Erkennung ignoriert werden (Event-Attribute keine Trace-Attribute)
IGNORE_COLS = {
    CASE_COL,
    ACT_COL,
    RES_COL,
    TIME_COL,
    MONTH_COL,
}


def detect_trace_attributes(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Schritte:
    1. Kandidaten: alle Spalten außer IGNORE_COLS
    2. Gruppieren nach Case ID
    3. Für jede Spalte prüfen, ob pro Case nur ein Wert vorkommt (max_nunique == 1)
    4. Falls ja: numerisch → trace_numeric, sonst → trace_categorical mit Vokabular
    5. Rückgabe: Listen für numerisch/kategorial und Vokabulare für kategoriale Attribute.
    """
    candidates = [c for c in df.columns if c not in IGNORE_COLS]

    group = df.groupby(CASE_COL, sort=False)

    trace_numeric: List[str] = []
    trace_categorical: List[str] = []
    trace_cat_vocabs: Dict[str, Dict[str, int]] = {}

    for col in candidates:
        # Ist der Wert in dieser Spalte pro Case konstant?
        max_nunique = group[col].nunique(dropna=False).max()
        if max_nunique != 1:
            # Dann ist es ein Event-Attribut und gehört nicht zu den Trace-Attributen
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            trace_numeric.append(col)
        else:
            trace_categorical.append(col)
            # Vokabular aller vorkommenden Ausprägungen (für One-Hot)
            values = group[col].first().dropna().astype(str).unique()
            trace_cat_vocabs[col] = {
                v: i for i, v in enumerate(sorted(values))
            }

    return trace_numeric, trace_categorical, trace_cat_vocabs


# ============================================================
# Vokabulare & Meta-Informationen
# ============================================================

def build_encoding_meta(df: pd.DataFrame) -> EncodingMeta:
    """
    Schritte:
    1. Activity-Vokabular: eindeutige Werte → IDs (1, 2, 3, ...)
    2. Resource-Vokabular: eindeutige Werte → IDs (1, 2, 3, ...)
    3. Month-Vokabular: eindeutige Werte → IDs (1, 2, 3, ...)
    4. Trace-Attribute via detect_trace_attributes() 
    5. max_prefix_len: maximale Anzahl Events pro Case (später wichtig für Padding-Länge bei LSTM-Sequenzen)
    6. Rückgabe: EncodingMeta-Objekt. (für das Encoding benötigte Metadaten)
    """
    # Activity-Vokabular (Padding-ID = 0, daher Start bei 1)
    act_values = df[ACT_COL].astype(str).unique()
    activity_vocab = {v: i + 1 for i, v in enumerate(sorted(act_values))}

    # Resource-Vokabular
    res_values = df[RES_COL].astype(str).unique()
    resource_vocab = {v: i + 1 for i, v in enumerate(sorted(res_values))}

    # Month-Vokabular (1..12) ebenfalls mit Startindex 1
    months = sorted(int(m) for m in df[MONTH_COL].unique())
    month_vocab = {m: i + 1 for i, m in enumerate(months)}

    # Trace-Attribute
    trace_numeric, trace_categorical, trace_cat_vocabs = detect_trace_attributes(df)

    # Maximale Trace-Länge (Anzahl Events pro Case, max über alle Cases)
    max_prefix_len = int(df.groupby(CASE_COL).size().max())

    return EncodingMeta(
        max_prefix_len=max_prefix_len,
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        month_vocab=month_vocab,
        trace_numeric_cols=trace_numeric,
        trace_categorical_cols=trace_categorical,
        trace_cat_vocabs=trace_cat_vocabs,
    )


# ============================================================
# Hilfsstruktur: Events pro Case
# ============================================================

def build_case_index(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Erstellt ein Dictionary: Case-ID → DataFrame der Events dieses Cases (sortiert). 
    Erleichtert später den Zugriff auf Events pro Case.
    """
    case_index: Dict[str, pd.DataFrame] = {}
    for cid, grp in df.groupby(CASE_COL, sort=False):
        case_index[str(cid)] = grp.reset_index(drop=True)
    return case_index


# ============================================================
# LSTM-Encoding (Format B)
# ============================================================

def encode_prefixes_lstm(
    df: pd.DataFrame,
    meta: EncodingMeta,
    case_ids: np.ndarray,
    prefix_lengths: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
   Erzeugt LSTM-Eingaben für alle Präfixe.
    Schritte:
    1. Arrays initialisieren: X_act, X_res, X_month (Integer), mask (Float), X_trace (Float)
    2. Case-Index erstellen
    3. Offsets für kategoriale Trace-Attribute berechnen
    4. Für jedes Präfix:
        a. Sequenzen: Activity/Resource/Month-IDs pro Position, mask = 1.0 für echte Events
        b. Trace-Attribute: numerisch direkt, kategorial als One-Hot
    6. Rückgabe: Dictionary mit X_act_seq, X_res_seq, X_month_seq, X_trace, mask.
    """
    N = len(case_ids)
    L = meta.max_prefix_len

    #leere Arrays initialisieren (automatisch Padding)
    X_act = np.zeros((N, L), dtype=np.int32)
    X_res = np.zeros((N, L), dtype=np.int32)
    X_month = np.zeros((N, L), dtype=np.int32)
    mask = np.zeros((N, L), dtype=np.float32)

    # Größe von X_trace Array berechnen (numerische + One-Hot der kategorialen Trace-Attribute)
    n_trace_features = len(meta.trace_numeric_cols) + sum(
        len(vocab) for vocab in meta.trace_cat_vocabs.values()
    )
    X_trace = np.zeros((N, n_trace_features), dtype=np.float32)
    # Case-Index für schnelle Zugriff auf Events eines Cases (wenn ein Präfix später die Events von Case C benötigt)
    case_index = build_case_index(df)

    # Offsets für One-Hot-Blöcke der kategorialen Trace-Attribute, weil One-Hot braucht mehrere Spalten pro Attribut. Damit sich die Spalten nicht überlappen -> separate Blöcke
    trace_cat_offsets: Dict[str, int] = {}
    offset = len(meta.trace_numeric_cols)
    for col in meta.trace_categorical_cols:
        trace_cat_offsets[col] = offset
        offset += len(meta.trace_cat_vocabs[col])

    for i, (cid, k) in enumerate(zip(case_ids, prefix_lengths)):
        cid = str(cid)
        k = int(k)

        trace_df = case_index[cid]
        # k kann n+1 sein (vollständiger Trace), aber trace_df hat nur n Zeilen
        actual_len = len(trace_df)
        prefix_df = trace_df.iloc[:min(k, actual_len)]

        # 1) Sequenzen: Activities / Resources / Months 
        for pos, (_, ev) in enumerate(prefix_df.iterrows()):
            if pos >= L:
                break

            act_id = meta.activity_vocab.get(str(ev[ACT_COL]), 0)
            X_act[i, pos] = act_id

            res_id = meta.resource_vocab.get(str(ev[RES_COL]), 0)
            X_res[i, pos] = res_id

            month_id = meta.month_vocab.get(int(ev[MONTH_COL]), 0)
            X_month[i, pos] = month_id

            mask[i, pos] = 1.0

        # 2) Trace-Attribute (aus einem beliebigen Event des Cases; Werte sind pro Case konstant)
        full_trace_df = trace_df
        first = full_trace_df.iloc[0]
        feat_vec = X_trace[i]

        # numerische Trace-Attribute direkt schreiben
        for j, col in enumerate(meta.trace_numeric_cols):
            val = first[col]
            feat_vec[j] = float(val) if pd.notna(val) else 0.0

        # kategoriale Trace-Attribute als One-Hot
        for col in meta.trace_categorical_cols:
            base = trace_cat_offsets[col]
            vocab = meta.trace_cat_vocabs[col]
            val = str(first[col])
            idx = vocab.get(val)
            if idx is not None:
                feat_vec[base + idx] = 1.0

    return {
        "X_act_seq": X_act,
        "X_res_seq": X_res,
        "X_month_seq": X_month,
        "X_trace": X_trace,
        "mask": mask,
    }


# ============================================================
# FFN-Encoding (Format A, CIBE)
# ============================================================


def compute_ffn_feature_size(
    meta: EncodingMeta,
) -> Tuple[int, Dict[str, object]]:
    """
    Berechnet die Gesamtgröße des CIBE-Vektors und die Offsets der Blöcke.
    Schritte:
    1. Trace-Features (numerisch)
    2. Trace-Features (kategorial, One-Hot)
    3. L × One-Hot(Activity)
    4. L × One-Hot(Resource)
    5. L × One-Hot(Month)
    6. Rückgabe: Gesamtgröße EA_star und Dictionary mit Offsets.
    """
    offsets: Dict[str, object] = {}
    offset = 0

    # Trace-numerisch
    offsets["trace_numeric"] = (offset, len(meta.trace_numeric_cols))
    offset += len(meta.trace_numeric_cols)

    # Trace-kategorial (One-Hot)
    trace_cat_offsets: Dict[str, int] = {}
    for col in meta.trace_categorical_cols:
        trace_cat_offsets[col] = offset
        offset += len(meta.trace_cat_vocabs[col])
    offsets["trace_cat"] = trace_cat_offsets

    L = meta.max_prefix_len

    # Activities
    offsets["act_start"] = offset
    n_act = len(meta.activity_vocab)
    offset += L * n_act

    # Resources
    offsets["res_start"] = offset
    n_res = len(meta.resource_vocab)
    offset += L * n_res

    # Months
    offsets["month_start"] = offset
    n_month = len(meta.month_vocab)
    offset += L * n_month

    EA_star = offset
    return EA_star, offsets


def encode_prefixes_ffn(
    df: pd.DataFrame,
    meta: EncodingMeta,
    case_ids: np.ndarray,
    prefix_lengths: np.ndarray,
) -> np.ndarray:
    """
    Erzeugt den CIBE-Feature-Vektor g_i für jedes Präfix:

      g_i = [ Trace-Features ,
              Event-Features(Activity) für Position 1..L ,
              Event-Features(Resource) für Position 1..L ,
              Event-Features(Month) für Position 1..L ]

    Für jedes Präfix:
    1. Trace-Features: numerisch direkt, kategorial als One-Hot
    2. Event-Features pro Position: Activity/Resource/Month als One-Hot
    Unterschied zu LSTM: Hier werden Event-Attribute pro Position als One-Hot kodiert (nicht als Integer-Sequenz).
    Rückgabe: Array X mit Shape (N, EA_star)
    """
    # Array X initialisieren
    N = len(case_ids)
    EA_star, offsets = compute_ffn_feature_size(meta)
    X = np.zeros((N, EA_star), dtype=np.float32)

    case_index = build_case_index(df)
    L = meta.max_prefix_len
    
    # Offsets für die einzelnen Blöcke
    trace_num_start, _ = offsets["trace_numeric"]
    trace_cat_offsets = offsets["trace_cat"]
    act_start = offsets["act_start"]
    res_start = offsets["res_start"]
    month_start = offsets["month_start"]

    # Größen der One-Hot-Blöcke
    n_act = len(meta.activity_vocab)
    n_res = len(meta.resource_vocab)
    n_month = len(meta.month_vocab)

# Für jedes Präfix i holen wir (Case-ID id, Präfixlänge k), und i gibt an, in welche Zeile die Features geschrieben werden.
    for i, (cid, k) in enumerate(zip(case_ids, prefix_lengths)):
        cid = str(cid)
        k = int(k)

        trace_df = case_index[cid]
        # k kann n+1 sein (vollständiger Trace), aber trace_df hat nur n Zeilen
        actual_len = len(trace_df)
        prefix_df = trace_df.iloc[:min(k, actual_len)]
        row = X[i]

        # ---- 1) Trace-Features ----
        first = trace_df.iloc[0]

        # numerische Trace-Attribute
        for j, col in enumerate(meta.trace_numeric_cols):
            row[trace_num_start + j] = float(first[col]) if pd.notna(first[col]) else 0.0

        # kategoriale Trace-Attribute (One-Hot)
        for col in meta.trace_categorical_cols:
            base = trace_cat_offsets[col]
            vocab = meta.trace_cat_vocabs[col]
            val = str(first[col])
            idx = vocab.get(val)
            if idx is not None:
                row[base + idx] = 1.0

        # ---- 2) Event-Features pro Position 1..L ----
        # Verwende min(k, actual_len, L) um IndexError zu vermeiden
        for pos in range(min(k, actual_len, L)):
            ev = prefix_df.iloc[pos]

            # Activity One-Hot
            a_id = meta.activity_vocab.get(str(ev[ACT_COL]), None)
            if a_id is not None and a_id > 0:
                row[act_start + pos * n_act + (a_id - 1)] = 1.0

            # Resource One-Hot
            r_id = meta.resource_vocab.get(str(ev[RES_COL]), None)
            if r_id is not None and r_id > 0:
                row[res_start + pos * n_res + (r_id - 1)] = 1.0

            # Month One-Hot
            m_id = meta.month_vocab.get(int(ev[MONTH_COL]), None)
            if m_id is not None and m_id > 0:
                row[month_start + pos * n_month + (m_id - 1)] = 1.0

        # Positionen > k bleiben 0 (Padding)

    return X


# ============================================================
# Hauptfunktion
# ============================================================
"""
    Ablauf:
    1. Pfade setzen
    2. Roh-Log laden
    3. IDP-Labels laden (case_ids, prefix_lengths)
    4. Metadaten erstellen
    5. LSTM-Encoding erzeugen und speichern
    6. FFN-Encoding erzeugen und speichern
    7. Metadaten speichern
    """
def main():
    parser = argparse.ArgumentParser(
        description="Erstellt Encodings (LSTM und FFN) aus Event-Log und Labels"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Pfad zum Event-Log (CSV oder XES). Standard: data/raw/finale.csv"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Pfad zu idp_labels.npz. Standard: automatisch aus --dataset oder processed/idp_labels.npz"
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
        help="Input-Verzeichnis für Labels. Standard: data/processed/ oder data/processed/{dataset}/"
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
        log_path = raw_dir / "finale.csv"
    else:
        log_path = Path(args.log)
    
    # Input/Output-Verzeichnisse bestimmen
    if args.output_dir is not None:
        processed_dir = Path(args.output_dir)
    elif args.dataset is not None:
        processed_dir = project_root / "data" / "processed" / args.dataset
    else:
        # Rückwärtskompatibel: Standard-Verzeichnis
        processed_dir = project_root / "data" / "processed"
    
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
    else:
        input_dir = processed_dir  # Standard: gleiches Verzeichnis
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if args.labels is not None:
        labels_path = Path(args.labels)
    else:
        labels_path = input_dir / "idp_labels.npz"

    print(f"Lade Roh-Log aus {log_path} ...")
    df = load_event_log_to_df(str(log_path))

    print(f"Lade IDP-Labels aus {labels_path} ...")
    labels = np.load(labels_path, allow_pickle=True)
    case_ids = labels["case_ids"]
    prefix_lengths = labels["prefix_lengths"]

    print("Bestimme Encoding-Metadaten (Vokabulare, Trace-Attribute, max_prefix_len) ...")
    meta = build_encoding_meta(df)

    # ---------- LSTM-Encoding ----------
    print("Erzeuge LSTM-Eingaben ...")
    lstm_inputs = encode_prefixes_lstm(df, meta, case_ids, prefix_lengths)
    np.savez_compressed(
        processed_dir / "encoding_lstm.npz",
        X_act_seq=lstm_inputs["X_act_seq"],
        X_res_seq=lstm_inputs["X_res_seq"],
        X_month_seq=lstm_inputs["X_month_seq"],
        X_trace=lstm_inputs["X_trace"],
        mask=lstm_inputs["mask"],
    )

    # ---------- FFN-Encoding (CIBE) ----------
    print("Erzeuge FFN-CIBE-Vektoren ...")
    X_ffn = encode_prefixes_ffn(df, meta, case_ids, prefix_lengths)
    np.save(processed_dir / "encoding_ffn.npy", X_ffn)

    # ---------- Meta speichern ----------
    meta_dict = {
        "max_prefix_len": meta.max_prefix_len,
        "activity_vocab": meta.activity_vocab,
        "resource_vocab": meta.resource_vocab,
        "month_vocab": meta.month_vocab,
        "trace_numeric_cols": meta.trace_numeric_cols,
        "trace_categorical_cols": meta.trace_categorical_cols,
        "trace_cat_vocabs": meta.trace_cat_vocabs,
    }
    with open(processed_dir / "encoding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    print("Fertig.")
    print(f"  LSTM-Inputs : {processed_dir / 'encoding_lstm.npz'}")
    print(f"  FFN-Inputs  : {processed_dir / 'encoding_ffn.npy'}")
    print(f"  Meta        : {processed_dir / 'encoding_meta.json'}")


if __name__ == "__main__":
    main()
