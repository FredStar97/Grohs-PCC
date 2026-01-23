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
MONTHYEAR_COL = "MonthYear"  # wird aus TIME_COL abgeleitet: "MM_YYYY"

# Weekday-Features (BPDP-Verhalten)
# Encoding-Unterschied:
# - LSTM: Ordinal-IDs (weekday_start_id, weekday_end_id) als 2 numerische Trace-Features, später per StandardScaler skaliert
# - FFN: One-Hot-Dummies (weekday_start_Monday, ..., weekday_end_Sunday) via pd.get_dummies
# weekday_end wird nur bei vollständigem Trace gesetzt (BPDP-Gating, verhindert Data Leakage)
WEEKDAY_START_COL = "weekday_start"
WEEKDAY_END_COL = "weekday_end"
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ============================================================
# Metadaten-Struktur für das Encoding
# ============================================================

@dataclass
class EncodingMeta:
    max_prefix_len: int # maximale Länge eines Präfixes
    activity_vocab: Dict[str, int] # Vokabular für Activities       
    resource_vocab: Dict[str, int] # Vokabular für Resources
    time_vocab: Dict[str, int] # Vokabular für MonthYear-Strings
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

    # Timestamp parsen und MonthYear-Feature erzeugen (BPDP-Format: ohne führende Null)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[MONTHYEAR_COL] = df[TIME_COL].dt.month.astype(str) + "_" + df[TIME_COL].dt.year.astype(str)

    # Case-ID Robustheit: Immer als String behandeln
    df[CASE_COL] = df[CASE_COL].astype(str)

    # Nach Case und Zeit sortieren, damit die Event-Reihenfolge eindeutig ist
    df = df.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)

    # Missing-Value-Behandlung: Fehlende Werte in Event-Attributen auf "No" setzen
    # (vor Vokabularbau, verhindert "nan" als Kategorie)
    for col in [ACT_COL, RES_COL, MONTHYEAR_COL]:
        df[col] = df[col].fillna("No").astype(str)
        # Ersetze explizit "nan" Strings (falls vorhanden)
        df[col] = df[col].replace("nan", "No")

    return df


def add_weekday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt weekday_start und weekday_end als case-konstante Features hinzu.
    
    weekday_start = Wochentag des ersten Events je Case
    weekday_end = Wochentag des letzten Events je Case
    
    Werte: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
    Falls Timestamp fehlt: "<UNK>"
    
    Voraussetzung: DataFrame ist bereits nach [Case ID, Complete Timestamp] sortiert.
    """
    def get_weekday_name(ts):
        """Konvertiert Timestamp zu Wochentag-String."""
        if pd.isna(ts):
            return "<UNK>"
        try:
            return WEEKDAY_NAMES[ts.weekday()]
        except (AttributeError, IndexError):
            return "<UNK>"
    
    # Gruppiere nach Case ID und bestimme erstes/letztes Event
    first_events = df.groupby(CASE_COL, sort=False).first()
    last_events = df.groupby(CASE_COL, sort=False).last()
    
    # Weekday-Mapping
    weekday_start_map = first_events[TIME_COL].apply(get_weekday_name)
    weekday_end_map = last_events[TIME_COL].apply(get_weekday_name)
    
    # Als neue Spalten hinzufügen (case-konstant)
    df[WEEKDAY_START_COL] = df[CASE_COL].map(weekday_start_map)
    df[WEEKDAY_END_COL] = df[CASE_COL].map(weekday_end_map)
    
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
# Weekday-Features werden separat behandelt (BPDP-Gating)
IGNORE_COLS = {
    CASE_COL,
    ACT_COL,
    RES_COL,
    TIME_COL,
    MONTHYEAR_COL,
    WEEKDAY_START_COL,
    WEEKDAY_END_COL,
    "Variant index",  # Explizit ignorieren: Variant index soll nicht als Trace-Attribut verwendet werden
}

# Schlüsselwörter für ID-/Leakage-verdächtige Spalten (case-insensitive)
ID_LEAKAGE_KEYWORDS = {"id", "case", "trace", "number", "nr", "key", "uid", "guid", "index", "idx"}


def _is_id_leakage_column(col: str, df: pd.DataFrame, n_cases: int) -> bool:
    """
    Prüft, ob eine Spalte ID-/Leakage-verdächtig ist.
    
    Kriterien:
    a) Spaltenname enthält (case-insensitive) ein ID-Keyword
    b) Sehr hohe Kardinalität (> 0.9 * #cases) bei string/integer-Typ
    """
    col_lower = col.lower()
    
    # Kriterium a): Spaltenname enthält ID-Keyword
    for keyword in ID_LEAKAGE_KEYWORDS:
        if keyword in col_lower:
            return True
    
    # Kriterium b): Hohe Kardinalität
    n_unique = df[col].nunique()
    if n_unique > 0.9 * n_cases:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            return True
    
    return False


def detect_trace_attributes(
    df: pd.DataFrame,
    exclude_categorical: bool = True,
    enable_id_leakage_filter: bool = False,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Schritte:
    1. Kandidaten: alle Spalten außer IGNORE_COLS
    2. Gruppieren nach Case ID
    3. Für jede Spalte prüfen, ob pro Case nur ein Wert vorkommt (max_nunique == 1)
    4. Filter: Spalten, die über den gesamten Datensatz konstant sind, werden ignoriert
    5. Falls ja: numerisch → trace_numeric, sonst → trace_categorical mit Vokabular
    6. Optional: Filtere ID-/Leakage-verdächtige Spalten heraus (nur wenn enable_id_leakage_filter=True)
    7. Vokabular für kategoriale Attribute: "<UNK>":0, echte Werte ab 1
    8. Rückgabe: Listen für numerisch/kategorial und Vokabulare für kategoriale Attribute.
    
    Args:
        df: DataFrame mit Event-Log Daten
        exclude_categorical: Wenn True, werden kategorische Trace-Attribute ignoriert (Default: True)
        enable_id_leakage_filter: Wenn True, werden ID-/Leakage-verdächtige Spalten gefiltert (Default: False)
    """
    candidates = [c for c in df.columns if c not in IGNORE_COLS]
    n_cases = df[CASE_COL].nunique()

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

        # Filter: Spalten, die über den gesamten Datensatz konstant sind, ignorieren
        global_nunique = df[col].nunique(dropna=False)
        if global_nunique <= 1:
            print(f"  -> Konstante Spalte gefiltert (nur {global_nunique} Wert(e) im Datensatz): {col}")
            continue

        # ID-/Leakage-Filter nur anwenden, wenn explizit aktiviert
        if enable_id_leakage_filter and _is_id_leakage_column(col, df, n_cases):
            print(f"  -> ID-/Leakage-Spalte gefiltert: {col}")
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            trace_numeric.append(col)
        else:
            # Kategorische Attribute nur hinzufügen, wenn nicht ausgeschlossen
            if not exclude_categorical:
                trace_categorical.append(col)
                # Vokabular: "<UNK>":0, dann echte Werte ab 1
                # Filtere "nan" Strings heraus
                values = group[col].first().dropna().astype(str).unique()
                values = sorted([v for v in values if v.lower() != "nan"])
                trace_cat_vocabs[col] = {"<UNK>": 0, **{v: i + 1 for i, v in enumerate(values)}}
            else:
                print(f"  -> Kategorische Spalte ignoriert: {col}")

    return trace_numeric, trace_categorical, trace_cat_vocabs


# ============================================================
# Vokabulare & Meta-Informationen
# ============================================================

def build_encoding_meta(
    df: pd.DataFrame,
    include_categorical_trace_attrs: bool = False,
    enable_id_leakage_filter: bool = False,
) -> EncodingMeta:
    """
    Schritte:
    1. Activity-Vokabular: "No"-Token (ID=0), dann echte Werte (ab ID=1)
    2. Resource-Vokabular: "No"-Token (ID=0), dann echte Werte (ab ID=1)
    3. Time-Vokabular: "No"-Token (ID=0), dann echte Werte (ab ID=1)
    4. Trace-Attribute via detect_trace_attributes() 
       - Nur numerische Trace-Attribute (kategorische werden standardmäßig ignoriert)
       - Spalten, die über den gesamten Datensatz konstant sind, werden entfernt
    5. max_prefix_len: maximale Anzahl Events pro Case (später wichtig für Padding-Länge bei LSTM-Sequenzen)
    6. Rückgabe: EncodingMeta-Objekt. (für das Encoding benötigte Metadaten)
    
    Args:
        df: DataFrame mit Event-Log Daten
        include_categorical_trace_attrs: Wenn True, werden kategorische Trace-Attribute einbezogen (Default: False)
        enable_id_leakage_filter: Wenn True, werden ID-/Leakage-verdächtige Spalten gefiltert (Default: False)
    """
    # Activity-Vokabular: "No"-Token für Padding (ID=0), danach echte Werte (ab ID=1)
    # Entferne "No" aus echten Werten, um Überschreiben des Padding-Tokens zu verhindern
    act_values = sorted(set(df[ACT_COL].astype(str).unique()) - {"No"})
    activity_vocab = {"No": 0, **{v: i + 1 for i, v in enumerate(act_values)}}

    # Resource-Vokabular: "No"-Token für Padding (ID=0), danach echte Werte (ab ID=1)
    # Entferne "No" aus echten Werten, um Überschreiben des Padding-Tokens zu verhindern
    res_values = sorted(set(df[RES_COL].astype(str).unique()) - {"No"})
    resource_vocab = {"No": 0, **{v: i + 1 for i, v in enumerate(res_values)}}

    # Time-Vokabular: "No"-Token für Padding (ID=0), danach echte Werte (ab ID=1)
    # Entferne "No" aus echten Werten, um Überschreiben des Padding-Tokens zu verhindern
    time_values = sorted(set(df[MONTHYEAR_COL].astype(str).unique()) - {"No"})
    time_vocab = {"No": 0, **{v: i + 1 for i, v in enumerate(time_values)}}

    # Trace-Attribute
    trace_numeric, trace_categorical, trace_cat_vocabs = detect_trace_attributes(
        df, 
        exclude_categorical=not include_categorical_trace_attrs,
        enable_id_leakage_filter=enable_id_leakage_filter,
    )

    # Maximale Trace-Länge (Anzahl Events pro Case, max über alle Cases)
    max_prefix_len = int(df.groupby(CASE_COL).size().max())

    return EncodingMeta(
        max_prefix_len=max_prefix_len,
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        time_vocab=time_vocab,
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


def filter_valid_prefixes(
    df: pd.DataFrame,
    case_ids: np.ndarray,
    prefix_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtert Prefixe, die länger als der zugehörige Trace sind (BPDP-Verhalten).
    
    Args:
        df: DataFrame mit Event-Log Daten (Case-IDs sind bereits Strings)
        case_ids: Array der Case-IDs
        prefix_lengths: Array der Präfix-Längen
        
    Returns:
        Tuple (gefilterte case_ids, gefilterte prefix_lengths, valid_mask)
    """
    # Trace-Längen berechnen (Case-IDs im df sind bereits Strings)
    trace_lengths = df.groupby(CASE_COL).size()
    
    # Filtere: nur Prefixe mit prefix_len <= trace_len
    # Case-IDs konsistent als Strings behandeln
    valid_mask = np.array([
        prefix_lengths[i] <= trace_lengths.get(str(case_ids[i]), 0)
        for i in range(len(case_ids))
    ])
    
    n_filtered = np.sum(~valid_mask)
    if n_filtered > 0:
        print(f"  -> {n_filtered} ungültige Prefixe gefiltert (prefix_len > trace_len)")
    
    return case_ids[valid_mask], prefix_lengths[valid_mask], valid_mask


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
    1. Arrays initialisieren: X_act, X_res, X_time (Integer) mit "No"-Token-ID, X_trace (Float)
    2. Case-Index erstellen
    3. Für jedes Präfix:
        a. Sequenzen: Activity/Resource/Time-IDs pro Position überschreiben, Rest bleibt "No"
        b. Trace-Attribute: numerisch direkt, kategorial als Label-Encoding
        c. Weekday-Features: 2 Ordinal-ID-Spalten (weekday_start_id, weekday_end_id)
           - weekday_start_id: immer gesetzt (Integer-ID 1-7 für Monday-Sunday, 0 für UNK)
           - weekday_end_id: nur wenn prefix_length == max_prefix_len (BPDP-Gating)
           - Diese IDs werden später gemeinsam mit anderen Trace-Features per StandardScaler skaliert
    4. Rückgabe: Dictionary mit X_act_seq, X_res_seq, X_time_seq, X_trace (keine mask).
    """
    N = len(case_ids)
    L = meta.max_prefix_len

    # "No"-Token-IDs für Padding (vorinitialisierung)
    no_id_act = meta.activity_vocab["No"]
    no_id_res = meta.resource_vocab["No"]
    no_id_time = meta.time_vocab["No"]

    # Arrays mit "No"-Token initialisieren (Padding)
    X_act = np.full((N, L), no_id_act, dtype=np.int32)
    X_res = np.full((N, L), no_id_res, dtype=np.int32)
    X_time = np.full((N, L), no_id_time, dtype=np.int32)

    # Größe von X_trace Array berechnen:
    # - numerische Trace-Attribute: 1 Feature pro Spalte
    # - kategoriale Trace-Attribute: 1 Feature pro Spalte (Label-Encoding, nicht One-Hot)
    # - Weekday-Features: 2 Ordinal-ID-Spalten (weekday_start_id, weekday_end_id)
    n_numeric = len(meta.trace_numeric_cols)
    n_cat = len(meta.trace_categorical_cols)
    n_weekday_features = 2  # weekday_start_id, weekday_end_id
    n_trace_features = n_numeric + n_cat + n_weekday_features
    X_trace = np.zeros((N, n_trace_features), dtype=np.float32)
    
    # Weekday-Mapping: String -> Ordinal-ID (1-7 für Monday-Sunday, 0 für UNK/fehlend)
    weekday_to_idx = {name: i + 1 for i, name in enumerate(WEEKDAY_NAMES)}
    # ID 0 ist explizit für unbekannte/fehlende Werte reserviert
    
    # Case-Index für schnellen Zugriff auf Events eines Cases
    case_index = build_case_index(df)

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

            act_id = meta.activity_vocab.get(str(ev[ACT_COL]), no_id_act)
            X_act[i, pos] = act_id

            res_id = meta.resource_vocab.get(str(ev[RES_COL]), no_id_res)
            X_res[i, pos] = res_id

            time_id = meta.time_vocab.get(str(ev[MONTHYEAR_COL]), no_id_time)
            X_time[i, pos] = time_id

        # 2) Trace-Attribute (aus einem beliebigen Event des Cases; Werte sind pro Case konstant)
        first = trace_df.iloc[0]
        feat_vec = X_trace[i]

        # numerische Trace-Attribute direkt schreiben
        for j, col in enumerate(meta.trace_numeric_cols):
            val = first[col]
            feat_vec[j] = float(val) if pd.notna(val) else 0.0

        # kategoriale Trace-Attribute als Label-Encoding (Integer-ID als float)
        num_offset = n_numeric
        for j, col in enumerate(meta.trace_categorical_cols):
            vocab = meta.trace_cat_vocabs[col]
            val = str(first[col]) if pd.notna(first[col]) else "<UNK>"
            if val.lower() == "nan":
                val = "<UNK>"
            idx = vocab.get(val, 0)  # 0 = <UNK> als Fallback
            feat_vec[num_offset + j] = float(idx)

        # 3) Weekday-Features (Ordinal-IDs, BPDP-Gating für weekday_end)
        weekday_offset = n_numeric + n_cat
        
        # weekday_start_id: immer setzen (Ordinal-ID 1-7 für Monday-Sunday, 0 für UNK)
        weekday_start_val = str(first[WEEKDAY_START_COL]) if WEEKDAY_START_COL in first.index else "<UNK>"
        if weekday_start_val.lower() == "nan":
            weekday_start_val = "<UNK>"
        weekday_start_id = weekday_to_idx.get(weekday_start_val, 0)  # 0 = UNK als Fallback
        feat_vec[weekday_offset] = float(weekday_start_id)
        
        # weekday_end_id: nur setzen wenn Prefix vollständig ist (per-case Gating)
        # Ein Prefix ist vollständig, wenn k >= actual_len (>= für n+1 Labeling)
        # Für unvollständige Prefixe bleibt weekday_end_id bei 0.0 (keine Info über End-Weekday)
        is_complete = (k >= actual_len)
        if is_complete:
            weekday_end_val = str(first[WEEKDAY_END_COL]) if WEEKDAY_END_COL in first.index else "<UNK>"
            if weekday_end_val.lower() == "nan":
                weekday_end_val = "<UNK>"
            weekday_end_id = weekday_to_idx.get(weekday_end_val, 0)  # 0 = UNK als Fallback
            feat_vec[weekday_offset + 1] = float(weekday_end_id)
        # else: weekday_end_id bleibt bei 0.0 (Initialisierung)

    return {
        "X_act_seq": X_act,
        "X_res_seq": X_res,
        "X_time_seq": X_time,
        "X_trace": X_trace,
    }


# ============================================================
# FFN-Encoding (Format A, CIBE) - DataFrame-basiert mit pd.get_dummies
# ============================================================


def build_ffn_reference_columns(
    df: pd.DataFrame,
    meta: EncodingMeta,
) -> List[str]:
    """
    Erstellt Referenz-Spalten für stabiles FFN-Encoding mit pd.get_dummies.
    
    Schritte:
    1. Erstelle ein Referenz-DataFrame mit allen möglichen Werten
    2. Für jeden Case: fülle Event-Positionen bis max_prefix_len
    3. Padding-Positionen erhalten NaN (keine künstlichen "_No" Dummies)
    4. Weekday-Features: weekday_start und weekday_end (BPDP-Gating)
    5. Wende pd.get_dummies an und speichere die Spaltennamen
    6. Diese Spalten werden dann für alle zukünftigen Encodings verwendet
    
    Returns:
        Liste der Spaltennamen in stabiler Reihenfolge
    """
    L = meta.max_prefix_len
    case_index = build_case_index(df)
    
    # Sammle alle Cases für Referenz-DataFrame
    ref_rows = []
    for cid in df[CASE_COL].unique():
        trace_df = case_index[str(cid)]
        first = trace_df.iloc[0]
        actual_len = len(trace_df)
        
        # Eine Zeile pro Case mit vollständigem Trace (bis max_len)
        row_data = {}
        
        # Trace-Features: numerisch als Zahl, kategorial als Original-String (One-Hot via get_dummies)
        for col in meta.trace_numeric_cols:
            row_data[col] = float(first[col]) if pd.notna(first[col]) else 0.0
        
        for col in meta.trace_categorical_cols:
            val = str(first[col]) if pd.notna(first[col]) else "<UNK>"
            if val.lower() == "nan":
                val = "<UNK>"
            row_data[col] = val
        
        # Weekday-Features (für Referenz: alle Weekdays einbeziehen)
        # weekday_start: immer setzen
        if WEEKDAY_START_COL in first.index:
            row_data[WEEKDAY_START_COL] = str(first[WEEKDAY_START_COL])
        else:
            row_data[WEEKDAY_START_COL] = "<UNK>"
        
        # weekday_end: immer setzen für Referenz-Spalten (um vollständige Dummy-Set zu erstellen)
        # Das Gating (nur vollständige Prefixe) erfolgt in encode_prefixes_ffn()
        if WEEKDAY_END_COL in first.index:
            row_data[WEEKDAY_END_COL] = str(first[WEEKDAY_END_COL])
        else:
            row_data[WEEKDAY_END_COL] = np.nan
        
        # Event-Positionen 1..L (echte Werte oder NaN für Padding)
        for pos in range(L):
            if pos < actual_len:
                ev = trace_df.iloc[pos]
                row_data[f"act_{pos+1}"] = str(ev[ACT_COL])
                row_data[f"res_{pos+1}"] = str(ev[RES_COL])
                row_data[f"time_{pos+1}"] = str(ev[MONTHYEAR_COL])
            else:
                # Padding = NaN (keine künstlichen "_No" Dummies)
                row_data[f"act_{pos+1}"] = np.nan
                row_data[f"res_{pos+1}"] = np.nan
                row_data[f"time_{pos+1}"] = np.nan
        
        ref_rows.append(row_data)
    
    ref_df = pd.DataFrame(ref_rows)
    ref_encoded = pd.get_dummies(ref_df, dummy_na=False)
    
    # Sicherstellen, dass alle Weekday-Dummies existieren (auch wenn nicht im Datensatz vorhanden)
    # weekday_start_Monday ... weekday_start_Sunday, weekday_end_Monday ... weekday_end_Sunday
    existing_cols = set(ref_encoded.columns)
    for day in WEEKDAY_NAMES:
        for prefix in [WEEKDAY_START_COL, WEEKDAY_END_COL]:
            col_name = f"{prefix}_{day}"
            if col_name not in existing_cols:
                ref_encoded[col_name] = 0
    
    return list(ref_encoded.columns)


def encode_prefixes_ffn(
    df: pd.DataFrame,
    meta: EncodingMeta,
    case_ids: np.ndarray,
    prefix_lengths: np.ndarray,
    reference_columns: List[str] = None,
) -> np.ndarray:
    """
    Erzeugt FFN-Feature-Vektoren mittels DataFrame-basiertem One-Hot-Encoding (pd.get_dummies).

    Schritte:
    1. Erstelle "wide" DataFrame mit einer Zeile pro Präfix:
       - Trace-Features: numerisch als Zahlen, kategorial als Original-Strings (One-Hot)
       - Weekday-Features: weekday_start immer, weekday_end nur bei k == max_prefix_len (BPDP-Gating)
       - Event-Positionen act_1..act_L, res_1..res_L, time_1..time_L als Strings
       - Padding mit NaN (keine künstlichen "_No" Dummies)
    2. Wende pd.get_dummies an für automatisches One-Hot-Encoding
    3. Stabilisiere Spalten mittels reference_columns
    4. Fülle verbleibende NaNs mit 0
    5. Konvertiere zu numpy array
    
    Args:
        df: DataFrame mit Event-Log Daten
        meta: EncodingMeta mit Vokabularen und Metadaten
        case_ids: Array der Case-IDs
        prefix_lengths: Array der Präfix-Längen
        reference_columns: Optional - Referenz-Spaltenliste für stabile Spaltenreihenfolge
        
    Returns:
        numpy array mit Shape (N, n_features)
    """
    N = len(case_ids)
    L = meta.max_prefix_len
    case_index = build_case_index(df)
    
    # 1) Baue "wide" DataFrame
    rows = []
    for i, (cid, k) in enumerate(zip(case_ids, prefix_lengths)):
        cid = str(cid)
        k = int(k)

        trace_df = case_index[cid]
        actual_len = len(trace_df)
        prefix_df = trace_df.iloc[:min(k, actual_len)]
        first = trace_df.iloc[0]

        row_data = {}
        
        # Trace-Features: numerisch als Zahl, kategorial als Original-String (One-Hot)
        for col in meta.trace_numeric_cols:
            row_data[col] = float(first[col]) if pd.notna(first[col]) else 0.0

        for col in meta.trace_categorical_cols:
            val = str(first[col]) if pd.notna(first[col]) else "<UNK>"
            if val.lower() == "nan":
                val = "<UNK>"
            row_data[col] = val
        
        # Weekday-Features (BPDP-Gating für weekday_end)
        # weekday_start: immer setzen
        if WEEKDAY_START_COL in first.index:
            row_data[WEEKDAY_START_COL] = str(first[WEEKDAY_START_COL])
        else:
            row_data[WEEKDAY_START_COL] = "<UNK>"
        
        # weekday_end: nur setzen wenn Prefix vollständig ist (per-case Gating)
        # Ein Prefix ist vollständig, wenn k >= actual_len (>= für n+1 Labeling)
        # Sonst als fehlend behandeln → alle weekday_end_* = 0 nach get_dummies
        is_complete = (k >= actual_len)
        if is_complete and WEEKDAY_END_COL in first.index:
            row_data[WEEKDAY_END_COL] = str(first[WEEKDAY_END_COL])
        else:
            row_data[WEEKDAY_END_COL] = np.nan  # wird zu 0 nach get_dummies
        
        # Event-Positionen 1..L (Strings für echte Events, NaN für Padding)
        for pos in range(L):
            if pos < min(k, actual_len):
                ev = prefix_df.iloc[pos]
                row_data[f"act_{pos+1}"] = str(ev[ACT_COL])
                row_data[f"res_{pos+1}"] = str(ev[RES_COL])
                row_data[f"time_{pos+1}"] = str(ev[MONTHYEAR_COL])
            else:
                # Padding = NaN (keine künstlichen "_No" Dummies)
                row_data[f"act_{pos+1}"] = np.nan
                row_data[f"res_{pos+1}"] = np.nan
                row_data[f"time_{pos+1}"] = np.nan
        
        rows.append(row_data)
    
    clean_df = pd.DataFrame(rows)
    
    # 2) One-Hot-Encoding mit pd.get_dummies
    enc_df = pd.get_dummies(clean_df, dummy_na=False)
    
    # 3) Spalten stabilisieren
    if reference_columns is None:
        # Erste Verwendung: erstelle Referenz-Spalten
        reference_columns = build_ffn_reference_columns(df, meta)
    
    # Fehlende Spalten hinzufügen (Wert 0) und Reihenfolge stabilisieren
    enc_df = enc_df.reindex(columns=reference_columns, fill_value=0)
    
    # 4) Fehlende NaNs mit 0 füllen
    enc_df = enc_df.fillna(0)
    
    # 5) Zu numpy konvertieren (dtype float32)
    X_ffn = enc_df.to_numpy(dtype=np.float32)

    return X_ffn


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
    parser.add_argument(
        "--include-categorical-trace-attrs",
        action="store_true",
        help="Bezieht kategorische Trace-Attribute in das Encoding ein (standardmäßig werden nur numerische verwendet)"
    )
    parser.add_argument(
        "--enable-id-leakage-filter",
        action="store_true",
        help="Aktiviert den ID-/Leakage-Filter für Trace-Attribute (Default: AUS für Reproduktion)"
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
    
    # Weekday-Features hinzufügen (BPDP-Verhalten)
    print("Füge Weekday-Features hinzu (weekday_start, weekday_end) ...")
    df = add_weekday_features(df)

    print(f"Lade IDP-Labels aus {labels_path} ...")
    labels = np.load(labels_path, allow_pickle=True)
    case_ids_orig = labels["case_ids"]
    prefix_lengths_orig = labels["prefix_lengths"]
    y_idp = labels["y_idp"] if "y_idp" in labels else None
    
    # Prefix-Filtering: Nur gültige Prefixe verwenden (BPDP-Verhalten)
    print(f"Filtere ungültige Prefixe (prefix_len > trace_len)...")
    case_ids, prefix_lengths, valid_mask = filter_valid_prefixes(df, case_ids_orig, prefix_lengths_orig)
    print(f"  -> {len(case_ids)} gültige Prefixe verbleiben")
    
    # y-Labels entsprechend filtern
    if y_idp is not None:
        y_idp = y_idp[valid_mask]

    print("Bestimme Encoding-Metadaten (Vokabulare, Trace-Attribute, max_prefix_len) ...")
    if args.include_categorical_trace_attrs:
        print("  -> Kategorische Trace-Attribute werden einbezogen")
    else:
        print("  -> Nur numerische Trace-Attribute (kategorische werden ignoriert)")
    if args.enable_id_leakage_filter:
        print("  -> ID-/Leakage-Filter ist aktiviert")
    meta = build_encoding_meta(
        df, 
        include_categorical_trace_attrs=args.include_categorical_trace_attrs,
        enable_id_leakage_filter=args.enable_id_leakage_filter,
    )

    # ---------- LSTM-Encoding ----------
    print("Erzeuge LSTM-Eingaben ...")
    lstm_inputs = encode_prefixes_lstm(df, meta, case_ids, prefix_lengths)
    np.savez_compressed(
        processed_dir / "encoding_lstm.npz",
        X_act_seq=lstm_inputs["X_act_seq"],
        X_res_seq=lstm_inputs["X_res_seq"],
        X_time_seq=lstm_inputs["X_time_seq"],
        X_trace=lstm_inputs["X_trace"],
    )

    # ---------- FFN-Encoding (CIBE) - DataFrame-basiert ----------
    print("Erzeuge FFN-Referenz-Spalten für stabiles One-Hot-Encoding ...")
    ffn_reference_columns = build_ffn_reference_columns(df, meta)
    print(f"  -> {len(ffn_reference_columns)} Features")
    
    print("Erzeuge FFN-CIBE-Vektoren mit pd.get_dummies ...")
    X_ffn = encode_prefixes_ffn(df, meta, case_ids, prefix_lengths, reference_columns=ffn_reference_columns)
    np.save(processed_dir / "encoding_ffn.npy", X_ffn)

    # ---------- Gefilterte Labels speichern ----------
    print("Speichere gefilterte Labels ...")
    filtered_labels = {
        "case_ids": case_ids,
        "prefix_lengths": prefix_lengths,
    }
    if y_idp is not None:
        filtered_labels["y_idp"] = y_idp
    np.savez_compressed(processed_dir / "encoding_labels.npz", **filtered_labels)

    # ---------- Meta speichern ----------
    # Berechne LSTM X_trace Feature-Informationen
    # - numerische Trace-Attribute: 1 Feature pro Spalte
    # - kategoriale Trace-Attribute: 1 Feature pro Spalte (Label-Encoding)
    # - Weekday-Features: 2 Ordinal-ID-Spalten (weekday_start_id, weekday_end_id)
    n_numeric = len(meta.trace_numeric_cols)
    n_cat = len(meta.trace_categorical_cols)
    n_weekday_features = 2  # weekday_start_id, weekday_end_id
    lstm_trace_feature_count = n_numeric + n_cat + n_weekday_features
    
    # Berechne LSTM Trace-Feature-Spalten (für Interpretierbarkeit)
    # Bei Label-Encoding: nur die Spaltennamen, nicht col_<value>
    # Weekday-Features: weekday_start_id, weekday_end_id (Ordinal-IDs, später skaliert)
    lstm_trace_columns = list(meta.trace_numeric_cols) + list(meta.trace_categorical_cols)
    lstm_trace_columns.append("weekday_start_id")
    lstm_trace_columns.append("weekday_end_id")
    
    meta_dict = {
        "max_prefix_len": meta.max_prefix_len,
        "activity_vocab": meta.activity_vocab,
        "resource_vocab": meta.resource_vocab,
        "time_vocab": meta.time_vocab,
        "trace_numeric_cols": meta.trace_numeric_cols,
        "trace_categorical_cols": meta.trace_categorical_cols,
        "trace_cat_vocabs": meta.trace_cat_vocabs,
        "lstm_trace_feature_count": lstm_trace_feature_count,
        "lstm_trace_columns": lstm_trace_columns,
        "weekday_names": WEEKDAY_NAMES,
        "ffn_reference_columns": ffn_reference_columns,
        "n_samples_original": len(case_ids_orig),
        "n_samples_filtered": len(case_ids),
    }
    with open(processed_dir / "encoding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    print("Fertig.")
    print(f"  LSTM-Inputs : {processed_dir / 'encoding_lstm.npz'}")
    print(f"  FFN-Inputs  : {processed_dir / 'encoding_ffn.npy'}")
    print(f"  Labels      : {processed_dir / 'encoding_labels.npz'}")
    print(f"  Meta        : {processed_dir / 'encoding_meta.json'}")


if __name__ == "__main__":
    main()
