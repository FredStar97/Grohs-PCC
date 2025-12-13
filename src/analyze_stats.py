from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_deviation_distribution(y_all, dev_types):
    """
    Berechnet detaillierte Statistiken zur Verteilung der Abweichungen.
    """
    # Summe der Abweichungen pro Typ (Spaltensumme)
    # y_all ist (N_prefixe, m_types). 1 = Dev, 0 = NoDev
    deviating_counts = np.sum(y_all, axis=0)
    total_samples = y_all.shape[0]
    
    conforming_counts = total_samples - deviating_counts
    
    # Berechne das Label Imbalance Ratio (LIR) gemäß Paper
    # LIR = Conforming / Deviating
    # Wir nutzen np.maximum(..., 1) um Division durch 0 zu verhindern
    lir = conforming_counts / np.maximum(deviating_counts, 1)
    
    # DataFrame erstellen
    df_dist = pd.DataFrame({
        'Deviation Type': dev_types,
        'Total Samples': total_samples,
        'Deviating (Count)': deviating_counts,
        'Deviating (%)': (deviating_counts / total_samples) * 100,
        'LIR (Imbalance Factor)': lir
    })
    
    # Sortieren: Die seltensten Fehler nach oben (höchster LIR)
    df_dist = df_dist.sort_values(by='LIR (Imbalance Factor)', ascending=False)
    return df_dist

def main():
    # Pfade definieren
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    labels_path = processed_dir / "idp_labels.npz"

    if not labels_path.exists():
        print(f"Datei nicht gefunden: {labels_path}")
        return

    print(f"Lade Daten aus {labels_path} ...")
    data = np.load(labels_path, allow_pickle=True)
    y = data["y"]
    dev_types = data["dev_types"]
    
    print("-" * 80)
    print(f"DATENSATZ STATISTIK")
    print("-" * 80)
    print(f"Anzahl Prefixe gesamt: {y.shape[0]}")
    print(f"Anzahl Deviation Types:  {y.shape[1]}")
    print("-" * 80)
    
    # Analyse ausführen
    df_stats = analyze_deviation_distribution(y, dev_types)
    
    # Schönere Ausgabe im Terminal
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df_stats)
    print("-" * 80)
    
    # Interpretationstext
    print("\nINTERPRETATION FÜR DEIN MODELL:")
    print("1. LIR (Label Imbalance Ratio) zeigt an, wie stark das Ungleichgewicht ist.")
    print("   - LIR = 100 bedeutet: Auf 1 Fehler kommen 100 normale Fälle.")
    print("   - Das Paper nutzt diesen Wert für die Loss-Gewichte: weight = 16^(...)")
    print("2. 'Deviating (%)' zeigt die absolute Seltenheit.")
    print("   - Werte unter 1% sind extrem schwer zu lernen (benötigen hohes Loss-Gewicht).")

if __name__ == "__main__":
    main()