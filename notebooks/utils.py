# Imports
from scipy.stats import shapiro
import numpy as np

def hex_to_rgb_normalized(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

def categorize_skewness(skew_value):
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
    
def shapiro_wilk_test(data,alpha):
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    # interpret
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')  

def find_similar_rows(df, max_diff=2):
    # NumPy Array erstellen, um schnell auf die Daten zuzugreifen
    df_np = df.to_numpy()

    # Ergebnisse zur Speicherung der ähnlichen Zeilen
    similar_rows = []

    # Iteriere durch jede Zeile und vergleiche mit den anderen Zeilen
    for i in range(len(df)):
        # Vergleiche die i-te Zeile mit allen anderen Zeilen
        for j in range(len(df)):
            if i != j:  # Vergleiche nur unterschiedliche Zeilen
                # Berechne die Anzahl der Unterschiede zwischen der i-ten und j-ten Zeile
                diff_count = np.sum(df_np[i] != df_np[j])
                
                # Wenn die Differenz in maximal 2 Spalten liegt, füge die i-te Zeile zu den ähnlichen Zeilen hinzu
                if diff_count <= max_diff:
                    similar_rows.append(i)
                    break  # Wenn eine Übereinstimmung gefunden wurde, überspringe den Rest der Vergleiche für die Zeile

    # Entferne doppelte Zeilenindizes
    similar_rows = list(set(similar_rows))

    # Gib die gefilterten Zeilen zurück
    return df.iloc[similar_rows]
