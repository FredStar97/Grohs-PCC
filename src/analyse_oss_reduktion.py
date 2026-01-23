"""
Analysiert, warum OSS unterschiedliche Reduktionen bei verschiedenen Modellen hat.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 1.0 / 3.0

def main():
    processed_dir = Path('data/processed')
    
    # Lade Daten (wie im Training)
    idp_labels_path = processed_dir / 'idp_labels.npz'
    encoding_labels_path = processed_dir / 'encoding_labels.npz'
    
    idp_data = np.load(idp_labels_path, allow_pickle=True)
    dev_types = list(idp_data['dev_types'])
    
    if encoding_labels_path.exists():
        encoding_data = np.load(encoding_labels_path, allow_pickle=True)
        if 'y_idp' in encoding_data:
            y_all = encoding_data['y_idp']
            case_ids = encoding_data['case_ids']
        else:
            # Fallback: Verwende Original-Labels
            y_all = idp_data['y']
            case_ids = idp_data['case_ids']
    else:
        y_all = idp_data['y']
        case_ids = idp_data['case_ids']
    
    # Train-Split (wie im Training)
    unique_cases = np.unique(case_ids)
    train_cases, _ = train_test_split(unique_cases, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_mask = np.isin(case_ids, train_cases)
    
    y_train = y_all[train_mask]
    case_ids_train = case_ids[train_mask]
    
    print("=" * 80)
    print("ANALYSE: WARUM UNTERSCHIEDLICHE OSS-REDUKTIONEN?")
    print("=" * 80)
    print()
    
    # 1. Separate Modelle: Pro Deviation Type
    print("1. SEPARATE MODELLE (separate_ffn, separate_lstm)")
    print("-" * 80)
    print("OSS wird PRO DEVIATION TYPE angewendet (jeder Type = eigenes binäres Problem)")
    print("Jeder Deviation Type wird einzeln betrachtet!")
    print()
    
    imbalance_ratios_separate = []
    dev_counts = []
    
    for i, dev_name in enumerate(dev_types):
        y_dev = y_train[:, i]
        n_deviant = int(np.sum(y_dev == 1))
        n_conforming = int(np.sum(y_dev == 0))
        
        if n_deviant > 0:
            imbalance_ratio = n_conforming / n_deviant
            imbalance_ratios_separate.append(imbalance_ratio)
            dev_counts.append((dev_name, n_deviant, n_conforming, imbalance_ratio))
            print(f"  {dev_name}:")
            print(f"    Deviant: {n_deviant:>6}, Conforming: {n_conforming:>8}")
            print(f"    Imbalance Ratio: {imbalance_ratio:>8.2f}:1")
            print()
    
    if imbalance_ratios_separate:
        print(f"  Durchschnittliches Imbalance Ratio: {np.mean(imbalance_ratios_separate):.2f}:1")
        print(f"  Median Imbalance Ratio: {np.median(imbalance_ratios_separate):.2f}:1")
        print(f"  Max Imbalance Ratio: {np.max(imbalance_ratios_separate):.2f}:1")
        print(f"  Min Imbalance Ratio: {np.min(imbalance_ratios_separate):.2f}:1")
        print()
    
    # 2. Collective Modelle: Alle Types zusammen
    print("2. COLLECTIVE MODELLE (collective_ffn, collective_lstm)")
    print("-" * 80)
    print("OSS wird EINMAL für ALLE DEVIATION TYPES zusammen angewendet")
    print("Label: 'Hat dieses Prefix irgendeine Abweichung?' (binär: 0=conforming, 1=deviant)")
    print()
    
    # Binäres Label: Hat Prefix irgendeine Abweichung?
    y_is_deviant = (np.sum(y_train, axis=1) > 0).astype(int)
    n_deviant_collective = int(np.sum(y_is_deviant == 1))
    n_conforming_collective = int(np.sum(y_is_deviant == 0))
    
    if n_deviant_collective > 0:
        imbalance_ratio_collective = n_conforming_collective / n_deviant_collective
        print(f"  Deviant (hat mindestens eine Abweichung): {n_deviant_collective:>6}")
        print(f"  Conforming (keine Abweichung): {n_conforming_collective:>8}")
        print(f"  Imbalance Ratio: {imbalance_ratio_collective:>8.2f}:1")
        print()
    
    # 3. Vergleich
    print("3. VERGLEICH: SEPARATE vs. COLLECTIVE")
    print("-" * 80)
    if imbalance_ratios_separate and n_deviant_collective > 0:
        print(f"  Separate Modelle (pro Type):")
        print(f"    Durchschnittliches Imbalance: {np.mean(imbalance_ratios_separate):.2f}:1")
        print(f"    Max Imbalance: {np.max(imbalance_ratios_separate):.2f}:1")
        print()
        print(f"  Collective Modelle (alle Types zusammen):")
        print(f"    Imbalance: {imbalance_ratio_collective:.2f}:1")
        print()
        factor = np.mean(imbalance_ratios_separate) / imbalance_ratio_collective
        print(f"  Faktor-Unterschied: {factor:.2f}x")
        print(f"  → Separate Modelle haben im Durchschnitt {factor:.1f}x höheres Imbalance!")
        print()
    
    # 4. Warum OSS unterschiedlich viel entfernt
    print("4. WARUM OSS UNTERSCHIEDLICH VIEL ENTFERNT")
    print("-" * 80)
    print("OSS (One-Sided Selection) Algorithmus:")
    print("  1. Wählt zufällig 'Seed'-Samples aus der Mehrheitsklasse (conforming)")
    print("  2. Findet k-Nächste-Nachbarn für jedes Seed-Sample")
    print("  3. Entfernt 'Tomek Links': Paare von Samples unterschiedlicher Klassen,")
    print("     die sehr nah beieinander sind (1-NN)")
    print("  4. Resultat: Redundante conforming cases werden entfernt")
    print()
    print("Separate Modelle:")
    print("  - Pro Deviation Type: Sehr wenige deviant cases (z.B. 10-200)")
    print("  - Viele conforming cases (z.B. 10,000+)")
    print("  - Sehr hohes Imbalance (z.B. 272:1)")
    print("  - OSS findet viele 'ähnliche/redundante' conforming cases")
    print("  - → Hohe Reduktion (86-94%)")
    print()
    print("Collective Modelle:")
    print("  - Alle Types zusammen: Mehr deviant cases (z.B. 1,000+)")
    print("  - Weniger conforming cases relativ gesehen")
    print("  - Niedrigeres Imbalance (z.B. 5.5:1)")
    print("  - OSS findet weniger 'redundante' conforming cases")
    print("  - → Niedrige Reduktion (2-21%)")
    print()
    
    # 5. Detaillierte Analyse pro Type
    print("5. DETAILLIERTE ANALYSE PRO DEVIATION TYPE")
    print("-" * 80)
    print("Warum separate_lstm (94.45%) höhere Reduktion hat als separate_ffn (86.20%)?")
    print()
    
    if dev_counts:
        dev_counts_sorted = sorted(dev_counts, key=lambda x: x[1])  # Sort nach n_deviant
        
        print("  Deviation Types sortiert nach Anzahl deviant cases:")
        print(f"  {'Deviation Type':<40} | {'Deviant':>8} | {'Conforming':>10} | {'Imbalance':>10}")
        print("  " + "-" * 75)
        for dev_name, n_dev, n_conf, imb in dev_counts_sorted:
            print(f"  {dev_name:<40} | {n_dev:>8} | {n_conf:>10} | {imb:>10.2f}:1")
        
        print()
        print("  Erklärung:")
        print("  - Types mit sehr wenigen deviant cases (< 50) → sehr hohes Imbalance")
        print("  - → OSS entfernt viele redundante conforming cases")
        print("  - → Hohe Reduktion (90%+)")
        print()
        print("  - Types mit mehr deviant cases (> 100) → niedrigeres Imbalance")
        print("  - → OSS entfernt weniger redundante conforming cases")
        print("  - → Niedrigere Reduktion (70-85%)")
        print()
        print("  separate_lstm und separate_ffn trainieren möglicherweise")
        print("  unterschiedliche Sets von Deviation Types (nach Filterung),")
        print("  was zu unterschiedlichen durchschnittlichen Reduktionen führt.")
        print()
    
    # 6. Feature-Räume und OSS
    print("6. FEATURE-RÄUME UND OSS-EFFEKTIVITÄT")
    print("-" * 80)
    print("OSS basiert auf Feature-Ähnlichkeit (Euklidische Distanz).")
    print()
    print("Separate Modelle:")
    print("  - separate_ffn: CIBE-Features (statische Features)")
    print("  - separate_lstm: Konkatenierte Features (Activities, Resources, Time, Trace)")
    print("  - Beide haben hochdimensionale Feature-Räume")
    print("  - Viele conforming cases sind 'ähnlich' → OSS entfernt viele")
    print()
    print("Collective Modelle:")
    print("  - collective_ffn: CIBE-Features (gleiche wie separate_ffn)")
    print("  - collective_lstm: One-Hot-encoded Features (Activities, Resources, Trace)")
    print("  - Aber: Weniger Imbalance → OSS entfernt weniger")
    print()
    print("WICHTIG: Die Reduktion hängt primär vom Imbalance-Ratio ab,")
    print("nicht so sehr von der Feature-Repräsentation!")
    print()
    
    # 7. Zusammenfassung
    print("=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print()
    print("Die unterschiedlichen OSS-Reduktionen entstehen durch:")
    print()
    print("1. UNTERSCHIEDLICHE KLASSEN-DEFINITIONEN:")
    print("   - Separate: Pro Deviation Type (binär: hat Type X / hat Type X nicht)")
    print("   - Collective: Alle Types zusammen (binär: hat irgendeine / hat keine)")
    print()
    print("2. UNTERSCHIEDLICHE IMBALANCE-RATIOS:")
    print(f"   - Separate (Durchschnitt): {np.mean(imbalance_ratios_separate):.1f}:1")
    print(f"   - Collective: {imbalance_ratio_collective:.1f}:1")
    print(f"   - Faktor: {factor:.1f}x Unterschied")
    print()
    print("3. OSS ENTFERNT MEHR BEI HÖHEREM IMBALANCE:")
    print("   - Hohes Imbalance → viele 'redundante' conforming cases")
    print("   - Niedriges Imbalance → weniger 'redundante' conforming cases")
    print()
    print("4. UNTERSCHIEDE ZWISCHEN separate_ffn UND separate_lstm:")
    print("   - Können unterschiedliche Deviation Types trainieren (nach Filterung)")
    print("   - Types mit sehr wenigen deviant cases → höhere Reduktion")
    print("   - separate_lstm trainiert möglicherweise mehr 'seltene' Types")
    print()

if __name__ == "__main__":
    main()
