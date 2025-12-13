from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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
    max_prefix_len: int
    activity_vocab: Dict[str, int]
    resource_vocab: Dict[str, int]
    month_vocab: Dict[int, int]
    trace_numeric_cols: List[str]
    trace_categorical_cols: List[str]
    trace_cat_vocabs: Dict[str, Dict[str, int]]


# ============================================================
# Roh-Log laden und vorbereiten
# ============================================================

def load_raw_df(csv_path: str) -> pd.DataFrame:
    """
    Lädt finale.csv und bereitet die für das Encoding relevanten Spalten vor.

    Erwartete Spalten:
      - 'Case ID'
      - 'Activity'
      - 'Resource'
      - 'Complete Timestamp'

    Zusätzlich wird eine Spalte 'Month' aus dem Timestamp erzeugt.
    """
    df = pd.read_csv(csv_path)

    # Sicherstellen, dass die benötigten Spalten existieren
    required_cols = {CASE_COL, ACT_COL, TIME_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Resource ist laut Paper ein Event-Attribut; hier erzwingen wir sie ebenfalls
    if RES_COL not in df.columns:
        raise ValueError(f"Required event attribute column '{RES_COL}' not found in {csv_path}")

    # Timestamp parsen und Month-Feature erzeugen
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[MONTH_COL] = df[TIME_COL].dt.month.astype("int16")

    # Nach Case und Zeit sortieren, damit die Event-Reihenfolge eindeutig ist
    df = df.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)
    return df


# ============================================================
# Trace-Attribute erkennen (alle Attribute, die pro Case konstant sind)
# ============================================================

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
    Identifiziert Trace-Attribute gemäß Paper:
      - alle Attribute, die für den gesamten Fall gelten.

    Konkret:
      - Wir betrachten alle Spalten außer den Event-Attributen und Case/Timestamp.
      - Eine Spalte wird als Trace-Attribut gewertet, wenn ihr Wert pro Case konstant ist.
      - Numerische Trace-Attribute bleiben numerisch,
        nicht-numerische werden als kategorische Trace-Attribute behandelt
        (und später One-Hot-enkodiert).
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
    Erzeugt alle Meta-Informationen, die für das Encoding benötigt werden:

      - Vokabular für Activities, Resources, Months
      - Liste der Trace-Attribute (numerisch / kategorial)
      - Vokabulare für kategoriale Trace-Attribute
      - maximale Trace-Länge im Log
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
    Baut eine Map von Case-ID -> DataFrame der Events dieses Cases
    (bereits zeitlich sortiert).
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
    Erzeugt die LSTM-Eingaben für alle Präfixe:

      - X_act_seq   : (N, L)  Integer-Indizes der Activities
      - X_res_seq   : (N, L)  Integer-Indizes der Resources
      - X_month_seq : (N, L)  Integer-Indizes der Months
      - X_trace     : (N, n_trace_features) numerischer Vektor der Trace-Attribute
      - mask        : (N, L)  1.0 für echte Events, 0.0 für Padding

    Dabei ist:
      - N = Anzahl Präfixe
      - L = meta.max_prefix_len
    """
    N = len(case_ids)
    L = meta.max_prefix_len

    X_act = np.zeros((N, L), dtype=np.int32)
    X_res = np.zeros((N, L), dtype=np.int32)
    X_month = np.zeros((N, L), dtype=np.int32)
    mask = np.zeros((N, L), dtype=np.float32)

    # Anzahl Trace-Features = numerische + One-Hot der kategorialen
    n_trace_features = len(meta.trace_numeric_cols) + sum(
        len(vocab) for vocab in meta.trace_cat_vocabs.values()
    )
    X_trace = np.zeros((N, n_trace_features), dtype=np.float32)

    case_index = build_case_index(df)

    # Offsets für One-Hot-Blöcke der kategorialen Trace-Attribute
    trace_cat_offsets: Dict[str, int] = {}
    offset = len(meta.trace_numeric_cols)
    for col in meta.trace_categorical_cols:
        trace_cat_offsets[col] = offset
        offset += len(meta.trace_cat_vocabs[col])

    for i, (cid, k) in enumerate(zip(case_ids, prefix_lengths)):
        cid = str(cid)
        k = int(k)

        trace_df = case_index[cid]
        prefix_df = trace_df.iloc[:k]

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
    Berechnet die Gesamtgröße EA* des CIBE-Vektors und liefert
    Offsets für die einzelnen Blöcke zurück.

    Struktur:
      [Trace-Features] +
      [L * One-Hot(Activity)] +
      [L * One-Hot(Resource)] +
      [L * One-Hot(Month)]
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

    Alle Event-Attribute werden pro Position als One-Hot kodiert.
    """
    N = len(case_ids)
    EA_star, offsets = compute_ffn_feature_size(meta)
    X = np.zeros((N, EA_star), dtype=np.float32)

    case_index = build_case_index(df)
    L = meta.max_prefix_len

    trace_num_start, _ = offsets["trace_numeric"]
    trace_cat_offsets = offsets["trace_cat"]
    act_start = offsets["act_start"]
    res_start = offsets["res_start"]
    month_start = offsets["month_start"]

    n_act = len(meta.activity_vocab)
    n_res = len(meta.resource_vocab)
    n_month = len(meta.month_vocab)

    for i, (cid, k) in enumerate(zip(case_ids, prefix_lengths)):
        cid = str(cid)
        k = int(k)

        trace_df = case_index[cid]
        prefix_df = trace_df.iloc[:k]
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
        for pos in range(min(k, L)):
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

def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "finale.csv"
    labels_path = processed_dir / "idp_labels.npz"

    print(f"Lade Roh-Log aus {csv_path} ...")
    df = load_raw_df(str(csv_path))

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
