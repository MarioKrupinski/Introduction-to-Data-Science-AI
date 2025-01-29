# Imports
from scipy.stats import shapiro
import numpy as np
import pandas as pd

def hex_to_rgb_normalized(hex_color: str) -> tuple:
    """
    Konvertiert eine Hex-Farbe in einen RGB-Tupel mit Werten zwischen 0 und 1.

    Args:
        hex_color (str): Hexadezimale Farbwertrepräsentation (z.B. "#FF5733").

    Returns:
        tuple: RGB-Werte als normalisierte Werte im Bereich [0, 1].
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

def categorize_skewness(skew_value: float) -> str:
    """
    Kategorisiert die Schiefe (Skewness) einer Verteilung in verschiedene Kategorien.

    Args:
        skew_value (float): Der berechnete Schiefe-Wert.

    Returns:
        str: Eine textuelle Beschreibung der Schiefe (z.B. "stark rechtsschief").
    """
    if skew_value > 1:
        return "stark rechtsschief"
    elif skew_value < -1:
        return "stark linksschief"
    elif skew_value > 0.5:
        return "leicht rechtschief"
    elif skew_value < -0.5:
        return "leicht linksschief"
    else:
        return "eher symmetrisch"
    
def shapiro_wilk_test(data: np.ndarray, alpha: float) -> None:
    """
    Führt den Shapiro-Wilk-Normalitätstest durch und gibt das Ergebnis aus.

    Args:
        data (np.ndarray): Eine Liste oder ein NumPy-Array mit numerischen Werten.
        alpha (float): Das Signifikanzniveau für den Test (z.B. 0.05).

    Returns:
        None: Gibt die Testergebnisse direkt in der Konsole aus.
    """
    stat, p = shapiro(data)
    print(f'Statistics={stat:.3f}, p={p:.5f}')

    # Interpretation des p-Werts
    if p > alpha:
        print('Sample sieht normalverteilt aus (H0 wird nicht verworfen).')
    else:
        print('Sample sieht nicht normalverteilt aus (H0 wird verworfen).')

def find_similar_rows(df: pd.DataFrame, max_diff: int = 2) -> pd.DataFrame:
    """
    Findet ähnliche Zeilen in einem DataFrame, basierend auf einer maximalen Anzahl 
    an Unterschieden pro Zeile.

    Args:
        df (pd.DataFrame): Der zu untersuchende DataFrame.
        max_diff (int, optional): Maximale Anzahl an Unterschieden zwischen zwei Zeilen. 
                                  Standardwert ist 2.

    Returns:
        pd.DataFrame: Ein DataFrame mit den ähnlichen Zeilen.
    """
    # NumPy-Array für schnelleren Zugriff
    df_np = df.to_numpy()

    # Liste zur Speicherung der ähnlichen Zeilen
    similar_rows = []

    # Iteriere durch jede Zeile und vergleiche mit anderen Zeilen
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:  # Vermeidung des Vergleichs einer Zeile mit sich selbst
                # Berechne die Anzahl der Unterschiede
                diff_count = np.sum(df_np[i] != df_np[j])
                
                # Falls die Unterschiede innerhalb der max_diff-Grenze liegen, speichern
                if diff_count <= max_diff:
                    similar_rows.append(i)
                    break  # Sobald eine Übereinstimmung gefunden wird, weiter zur nächsten Zeile

    # Entferne doppelte Indizes
    similar_rows = list(set(similar_rows))

    # Gib die gefilterten Zeilen zurück
    return df.iloc[similar_rows]
