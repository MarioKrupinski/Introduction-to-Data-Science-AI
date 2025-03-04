# Imports
from scipy.stats import shapiro

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

import itertools

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


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

def train_baseline_model(model, X_train, y_train, X_test, run_name="Baseline"):
    '''
    Trainiert ein Baseline-Modell und gibt Vorhersagen, den Modellnamen und den Durchlaufnamen zurück.
    
    Args:
        model (sklearn.base.BaseEstimator): Ein sklearn-Modell, z. B. LogisticRegression().
        X_train (np.ndarray oder pd.DataFrame): Trainings-Feature-Matrix.
        y_train (np.ndarray oder pd.Series): Trainings-Zielvariable.
        X_test (np.ndarray oder pd.DataFrame): Test-Feature-Matrix.
        run_name (str, optional): Name des Durchlaufs. Standardwert ist "Baseline".

    Returns:
        tuple: Enthält die folgenden Werte:
            - np.ndarray: Die vorhergesagten Labels (y_pred).
            - str: Der Name des verwendeten Modells.
            - str: Der Name des Durchlaufs.
            - model: Das trainierte Modell.
    '''
    
    # Falls X_train/X_test kein DataFrame ist, in DataFrame umwandeln
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Speichere die ursprünglichen Spaltennamen
    original_columns = X_train.columns.tolist()

    # Kategorische Spalten automatisch erkennen
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-Hot-Encoding nur für kategorische Features
    if categorical_features:
        X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first= True)

        # Sicherstellen, dass beide DataFrames die gleichen Spalten haben
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Modell trainieren
    model.fit(X_train, y_train)
    
    # Vorhersagen
    y_pred = model.predict(X_test)
    
    # Modellname abrufen
    model_name = model.__class__.__name__
    
    return y_pred, model_name, run_name, model, X_train, original_columns

def map_feature_name(original_columns, feature):
    for original in original_columns:
        if feature.startswith(original + "_"):  # Sicherstellen, dass es sich um ein One-Hot-Encoded-Feature handelt
            return original  # Gruppiere nach dem ursprünglichen Namen
    return feature  # Falls keine Übereinstimmung, behalte den Originalnamen

def plot_feature_importance(model, X_train, original_columns, color="royalblue", title=None, model_name=None):
    '''
    Plottet die Feature-Wichtigkeiten eines Modells.
    
    Args:
        model (sklearn.base.BaseEstimator): Das trainierte Modell (z.B. RandomForest, XGBoost).
        X_train (pd.DataFrame): Das Trainings-Feature-Set.
        color (str, optional): Die Farbe der Balken im Plot. Standard ist "royalblue".
        title (str, optional): Der Titel des Plots. Wenn nicht angegeben, wird der Modellname verwendet.
        model_name (str, optional): Der Name des Modells für den Titel. Wenn nicht angegeben, wird er aus dem Modell abgeleitet.

    Returns:
        pd.DataFrame: DataFrame mit den Feature-Wichtigkeiten.
    '''
    
    # Feature-Wichtigkeiten abrufen (falls verfügbar)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):  # Bei linearen Modellen wie LogisticRegression
        importances = np.abs(model.coef_[0])  # Absoluter Wert der Koeffizienten
    else:
        print("Warnung: Das Modell hat keine 'feature_importances_' Methode. Bitte ein Modell verwenden, das diese Eigenschaft besitzt.")
        return  # Verlasse die Funktion ohne den Plot zu erstellen
    
    feature_names = X_train.columns
    
    mapped_features = [map_feature_name(original_columns, f) for f in feature_names]
    
    # In DataFrame umwandeln und nach Wichtigkeit sortieren
    feature_importance_df = pd.DataFrame({"Feature": feature_names,"Mapped Feature": mapped_features, "Importance": importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Titel anpassen: Wenn kein Titel angegeben wird, wird der Modellname verwendet
    if title is None:
        if model_name is None:
            model_name = model.__class__.__name__
        title = f"Feature Importance nach {model_name}"

    # Visualisieren
    plt.figure(figsize=(10, 15))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color=color)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

    # Rückgabe des DataFrames für die weitere Verwendung
    return feature_importance_df

def plot_permutation_importance(model, X_test, y_test, original_columns, color="royalblue", title="Permutation Feature Importance"):
    """
    Plottet die Permutations-Feature-Wichtigkeit eines gegebenen Modells auf den Testdaten.

    Die Funktion berechnet die Permutationswichtigkeit jedes Features im Modell,
    indem sie die Feature-Werte zufällig vertauscht und den resultierenden Rückgang 
    der Modellleistung misst. Die Wichtigkeitswerte werden anschließend in einem 
    horizontalen Balkendiagramm visualisiert, wobei die wichtigsten Features oben angezeigt werden.

    Args:
        model (sklearn.base.BaseEstimator): Das trainierte Modell, dessen Feature-Wichtigkeit geplottet werden soll.
        X_test (pd.DataFrame): Die Testdaten, die verwendet werden, um die Permutationswichtigkeit zu berechnen.
        y_test (pd.Series oder pd.DataFrame): Die tatsächlichen Zielwerte der Testdaten.
        original_columns (list von str): Die ursprünglichen Feature-Namen, die für die Zuordnung der Features verwendet werden 
                                        (z.B. nach One-Hot-Encoding).
        color (str, optional): Die Farbe der Balken im Plot. Standard ist "royalblue".
        title (str, optional): Der Titel des Plots. Standard ist "Permutation Feature Importance".

    Returns:
        pd.DataFrame: Ein DataFrame, das die Features, ihre zugeordneten Namen und die entsprechende Wichtigkeit enthält.
                      Der DataFrame ist nach der Wichtigkeit in absteigender Reihenfolge sortiert.
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    importance = result.importances_mean  # Durchschnittliche Wichtigkeit über alle Wiederholungen
    feature_names = X_test.columns
    
    # Mapped Features, falls benötigt (z.B. bei One-Hot-Encoding)
    mapped_features = [map_feature_name(original_columns, f) for f in feature_names]
    
    # DataFrame für Visualisierung
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Mapped Feature": mapped_features, "Importance": importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Visualisierung
    plt.figure(figsize=(10, 15))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color=color)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

    return feature_importance_df

def forward_selection(X_train, y_train, X_test, y_test, model, feature_groups, feature_order=None):
    '''
    Führt eine Forward Selection mit Gruppen-Prüfung durch.
    
    Args:
        X_train (pd.DataFrame): Trainingsdaten (Features) **ohne One-Hot-Encoding**
        y_train (pd.Series): Trainingszielvariable
        X_test (pd.DataFrame): Testdaten (Features) **ohne One-Hot-Encoding**
        y_test (pd.Series): Testzielvariable
        model (sklearn.base.BaseEstimator): Das zu verwendende Modell
        feature_groups (dict): Dictionary von Feature-Gruppen, in denen nur ein Feature pro Gruppe ausgewählt werden soll
        feature_order (list, optional): Vordefinierte Reihenfolge der Features nach Wichtigkeit (z. B. ANOVA/Chi²).
        
    Returns:
        pd.DataFrame: DataFrame mit den Features, Accuracy und F1-Score
    '''
    
    selected_features = []  # Liste der ausgewählten Features
    used_feature_groups = set()  # Set zur Speicherung getesteter Gruppen
    results = []  # Ergebnisse der Forward Selection
    
    # Falls keine Feature-Order gegeben ist, alle Features in beliebiger Reihenfolge testen
    remaining_features = [f for f in feature_order if f in X_train.columns] if feature_order else X_train.columns.tolist()
    
    # Erkennen, welche Features kategorisch sind
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    best_accuracy = 0
    best_f1 = 0
    
    # Forward Selection mit Gruppen-Prüfung
    for feature in remaining_features:
        # Finde die Gruppe, zu der das Feature gehört (falls es eine gibt)
        feature_group = None
        for group_name, group_features in feature_groups.items():
            if feature in group_features:
                feature_group = group_name
                break  # Sobald Gruppe gefunden, abbrechen
        
        # Falls es zu einer Gruppe gehört, aber die Gruppe wurde schon getestet → weiter
        if feature_group and feature_group in used_feature_groups:
            continue

        # Falls das Feature zu einer Gruppe gehört, **teste ALLE Features aus dieser Gruppe**
        if feature_group:
            candidate_features = [f for f in feature_groups[feature_group] if f in X_train.columns]
        else:
            candidate_features = [feature]

        best_feature_in_group = None
        best_acc_in_group = 0
        best_f1_in_group = 0

        # Teste alle Features in der Gruppe
        for candidate in candidate_features:
            temp_features = selected_features + [candidate]
            
            X_train_temp = X_train[temp_features].copy()
            X_test_temp = X_test[temp_features].copy()
            
            # Falls das Feature kategorisch ist, wende One-Hot-Encoding an
            for column in X_train_temp.columns:
                if column in categorical_features:
                    # Wende One-Hot-Encoding für jede kategorische Spalte an
                    X_train_temp = pd.get_dummies(X_train_temp, columns=[column], drop_first=True)
                    X_test_temp = pd.get_dummies(X_test_temp, columns=[column], drop_first=True)

                    # Sicherstellen, dass train und test gleiche Spalten haben
                    X_train_temp, X_test_temp = X_train_temp.align(X_test_temp, join='left', axis=1, fill_value=0)

            # Modell trainieren
            model.fit(X_train_temp, y_train)
            y_pred = model.predict(X_test_temp)

            # Accuracy und F1-Score berechnen
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Bestes Feature innerhalb der Gruppe auswählen
            if acc > best_acc_in_group or f1 > best_f1_in_group:
                best_feature_in_group = candidate
                best_acc_in_group = acc
                best_f1_in_group = f1

        # Falls ein gutes Feature in der Gruppe gefunden wurde, hinzufügen
        if best_feature_in_group:
            selected_features.append(best_feature_in_group)
            best_accuracy = best_acc_in_group
            best_f1 = best_f1_in_group
            results.append((selected_features.copy(), best_accuracy, best_f1))
            
            # Markiere die Gruppe als getestet
            if feature_group:
                used_feature_groups.add(feature_group)

            remaining_features = [f for f in remaining_features if f not in selected_features]

    # DataFrame mit Ergebnissen
    result_df = pd.DataFrame(results, columns=["Selected Features", "Accuracy", "F1-Score"])
    
    return result_df

def calculate_vif(X):
    """
    Berechnet den Variance Inflation Factor (VIF) für die Features in einem DataFrame.
    
    Der VIF gibt an, wie stark ein Feature mit den anderen Features korreliert ist.
    Ein hoher VIF-Wert (>10 gilt als kritisch) deutet auf Multikollinearität hin.

    Args:
        X (pd.DataFrame): DataFrame mit numerischen Features (keine Kategorien!)

    Returns:
        pd.DataFrame: DataFrame mit zwei Spalten: "Feature" und "VIF"
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# def forward_selection_vif(X_train, y_train, X_test, y_test, model, feature_groups, feature_order=None, vif_threshold=10):
#     """
#     Führt eine Forward Selection mit Gruppen-Prüfung und VIF-Kontrolle durch.

#     Diese Funktion wählt iterativ Features aus, indem sie schrittweise das beste Feature
#     hinzufügt, das die Modellleistung verbessert. Bevor ein Feature aufgenommen wird, wird
#     überprüft, ob es zu hoher Multikollinearität führt (VIF > vif_threshold).

#     Args:
#         X_train (pd.DataFrame): Trainingsdaten (Features) **ohne One-Hot-Encoding**
#         y_train (pd.Series): Trainingszielvariable
#         X_test (pd.DataFrame): Testdaten (Features) **ohne One-Hot-Encoding**
#         y_test (pd.Series): Testzielvariable
#         model (sklearn.base.BaseEstimator): Das zu verwendende Modell (muss `.fit()` und `.predict()` unterstützen)
#         feature_groups (dict): Dictionary von Feature-Gruppen, in denen nur ein Feature pro Gruppe ausgewählt werden soll
#         feature_order (list, optional): Vordefinierte Reihenfolge der Features nach Wichtigkeit (z. B. ANOVA/Chi²).
#         vif_threshold (float, optional): Maximal erlaubter VIF-Wert (Standard = 10)

#     Returns:
#         pd.DataFrame: DataFrame mit den ausgewählten Features, Accuracy, F1-Score
#     """
#     selected_features = []  # Liste der bereits ausgewählten Features
#     used_feature_groups = set()  # Set zur Speicherung getesteter Gruppen
#     results = []  # Ergebnisse der Forward Selection

#     # Falls keine Feature-Order gegeben ist, alle Features in beliebiger Reihenfolge testen
#     remaining_features = [f for f in feature_order if f in X_train.columns] if feature_order else X_train.columns.tolist()
    
#     # Identifiziere kategoriale Features
#     categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

#     best_accuracy = 0
#     best_f1 = 0

#     # Forward Selection mit Gruppen-Prüfung
#     for feature in remaining_features:
#         # Finde die Gruppe, zu der das Feature gehört (falls es eine gibt)
#         feature_group = None
#         for group_name, group_features in feature_groups.items():
#             if feature in group_features:
#                 feature_group = group_name
#                 break  # Sobald Gruppe gefunden, abbrechen
        
#         # Falls es zu einer Gruppe gehört, aber die Gruppe wurde schon getestet → weiter
#         if feature_group and feature_group in used_feature_groups:
#             continue

#         # Falls das Feature zu einer Gruppe gehört, **teste ALLE Features aus dieser Gruppe**
#         if feature_group:
#             candidate_features = [f for f in feature_groups[feature_group] if f in X_train.columns]
#         else:
#             candidate_features = [feature]

#         best_feature_in_group = None
#         best_acc_in_group = 0
#         best_f1_in_group = 0

#         # Teste alle Features in der Gruppe
#         for candidate in candidate_features:
#             temp_features = selected_features + [candidate]
            
#             X_train_temp = X_train[temp_features].copy()
#             X_test_temp = X_test[temp_features].copy()
            
#             # Falls das Feature kategorisch ist, wende One-Hot-Encoding an
#             for column in X_train_temp.columns:
#                 if column in categorical_features:
#                     # Wende One-Hot-Encoding für jede kategorische Spalte an
#                     X_train_temp = pd.get_dummies(X_train_temp, columns=[column], drop_first=True)
#                     X_test_temp = pd.get_dummies(X_test_temp, columns=[column], drop_first=True)

#                     # Sicherstellen, dass train und test gleiche Spalten haben
#                     X_train_temp, X_test_temp = X_train_temp.align(X_test_temp, join='left', axis=1, fill_value=0)
#             # Umwandlung der One-Hot-kodierten Spalten in numerische Werte
#             X_train_temp = X_train_temp.astype('float64')
#             X_test_temp = X_test_temp.astype('float64')

#            # Berechne VIF nur, wenn mehr als ein Feature im Modell ist
#             if len(temp_features) > 1:
#                 vif_df = calculate_vif(X_train_temp)
#                 max_vif = vif_df["VIF"].max()

#                 # Falls der VIF des neuen Feature-Sets zu hoch ist, Feature nicht hinzufügen
#                 if max_vif > vif_threshold:
#                     continue  
#             # Wenn nur ein Feature im Modell ist, überspringe die VIF-Berechnung

#             # Modell trainieren und evaluieren
#             model.fit(X_train_temp, y_train)
#             y_pred = model.predict(X_test_temp)

#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)

#             # Speichere das Feature mit der besten Leistung innerhalb der Gruppe
#             if acc > best_acc_in_group or f1 > best_f1_in_group:
#                 best_feature_in_group = candidate
#                 best_acc_in_group = acc
#                 best_f1_in_group = f1

#         # Falls ein Feature die beste Leistung hat und VIF-Pass besteht, aufnehmen
#         if best_feature_in_group:
#             selected_features.append(best_feature_in_group)
#             best_accuracy = best_acc_in_group
#             best_f1 = best_f1_in_group
#             results.append((selected_features.copy(), best_accuracy, best_f1))
            
#             # Markiere die Gruppe als getestet
#             if feature_group:
#                 used_feature_groups.add(feature_group)

#             # Entferne das ausgewählte Feature aus der Liste der verbleibenden Features
#             remaining_features = [f for f in remaining_features if f not in selected_features]

#     # Ergebnisse als DataFrame zurückgeben
#     result_df = pd.DataFrame(results, columns=["Selected Features", "Accuracy", "F1-Score"])
    
#     return result_df

# def forward_selection_vif(X_train, y_train, X_test, y_test, model, feature_groups, feature_order=None, vif_threshold=10):
#     """
#     Führt eine Forward Selection mit Gruppen-Prüfung und VIF-Kontrolle durch, wobei auch
#     die Möglichkeit besteht, Features mit hohem VIF auszutauschen, um die Modell-Performance zu optimieren.
#     """
#     selected_features = []  # Liste der bereits ausgewählten Features
#     used_feature_groups = set()  # Set zur Speicherung getesteter Gruppen
#     excluded_features = set()  # Set zur Speicherung der Features mit zu hohem VIF
#     results = []  # Ergebnisse der Forward Selection

#     # Falls keine Feature-Order gegeben ist, alle Features in beliebiger Reihenfolge testen
#     remaining_features = [f for f in feature_order if f in X_train.columns] if feature_order else X_train.columns.tolist()
    
#     # Identifiziere kategoriale Features
#     categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

#     best_accuracy = 0
#     best_f1 = 0

#     iteration = 0  # Zähler für Iterationen
#     for feature in remaining_features:
#         iteration += 1
#         print(f"\nIteration {iteration}: Jetzt wird das Feature '{feature}' getestet.")

#         # Wenn das Feature bereits ausgeschlossen wurde, überspringe es
#         if feature in excluded_features:
#             print(f"Feature '{feature}' wurde aufgrund eines zu hohen VIF bereits ausgeschlossen. Überspringen.")
#             continue

#         # Finde die Gruppe, zu der das Feature gehört (falls es eine gibt)
#         feature_group = None
#         for group_name, group_features in feature_groups.items():
#             if feature in group_features:
#                 feature_group = group_name
#                 break  # Sobald Gruppe gefunden, abbrechen
        
#         # Falls es zu einer Gruppe gehört, aber die Gruppe wurde schon getestet → weiter
#         if feature_group and feature_group in used_feature_groups:
#             print(f"Gruppe '{feature_group}' wurde bereits getestet, überspringe Feature '{feature}'.")
#             continue

#         # Falls das Feature zu einer Gruppe gehört, **teste ALLE Features aus dieser Gruppe**
#         if feature_group:
#             candidate_features = [f for f in feature_groups[feature_group] if f in X_train.columns]
#         else:
#             candidate_features = [feature]

#         best_feature_in_group = None
#         best_acc_in_group = 0
#         best_f1_in_group = 0
#         best_vif_in_group = 0  # Variable für den besten VIF-Wert
#         best_vif_set = 0  # Maximaler VIF-Wert für das Set der Features

#         # Teste alle Features in der Gruppe
#         for candidate in candidate_features:
#             print(f"  Teste Feature '{candidate}'...")
#             temp_features = selected_features + [candidate]
            
#             X_train_temp = X_train[temp_features].copy()
#             X_test_temp = X_test[temp_features].copy()
            
#             # Falls das Feature kategorisch ist, wende One-Hot-Encoding an
#             for column in X_train_temp.columns:
#                 if column in categorical_features:
#                     # Wende One-Hot-Encoding für jede kategorische Spalte an
#                     X_train_temp = pd.get_dummies(X_train_temp, columns=[column], drop_first=True)
#                     X_test_temp = pd.get_dummies(X_test_temp, columns=[column], drop_first=True)

#                     # Sicherstellen, dass train und test gleiche Spalten haben
#                     X_train_temp, X_test_temp = X_train_temp.align(X_test_temp, join='left', axis=1, fill_value=0)
#             # Umwandlung der One-Hot-kodierten Spalten in numerische Werte
#             X_train_temp = X_train_temp.astype('float64')
#             X_test_temp = X_test_temp.astype('float64')

#             # Berechne VIF nur, wenn mehr als ein Feature im Modell ist
#             max_vif = 0
#             if len(temp_features) > 1:
#                 vif_df = calculate_vif(X_train_temp)
#                 max_vif = vif_df["VIF"].max()

#                 # Falls der VIF des neuen Feature-Sets zu hoch ist, Feature nicht hinzufügen
#                 if max_vif > vif_threshold:
#                     print(f"  Feature '{candidate}' hat VIF={max_vif:.2f}, was den Schwellenwert ({vif_threshold}) überschreitet.")
#                     # Versuche iterativ das Feature zu entfernen, um VIF zu reduzieren
#                     for feature_to_remove in selected_features:
#                         temp_features_removed = [f for f in temp_features if f != feature_to_remove]
#                         X_train_temp_removed = X_train[temp_features_removed].copy()
#                         X_test_temp_removed = X_test[temp_features_removed].copy()

#                         for column in X_train_temp_removed.columns:
#                             if column in categorical_features:
#                                 X_train_temp_removed = pd.get_dummies(X_train_temp_removed, columns=[column], drop_first=True)
#                                 X_test_temp_removed = pd.get_dummies(X_test_temp_removed, columns=[column], drop_first=True)
#                                 X_train_temp_removed, X_test_temp_removed = X_train_temp_removed.align(X_test_temp_removed, join='left', axis=1, fill_value=0)

#                         X_train_temp_removed = X_train_temp_removed.astype('float64')
#                         X_test_temp_removed = X_test_temp_removed.astype('float64')
#                         vif_removed = calculate_vif(X_train_temp_removed)

#                         if vif_removed["VIF"].max() <= vif_threshold:
#                             print(f"    Entferne '{feature_to_remove}', VIF sinkt unter {vif_threshold}.")
#                             temp_features = temp_features_removed
#                             break
#                     continue

#             # Modell trainieren und evaluieren
#             model.fit(X_train_temp, y_train)
#             y_pred = model.predict(X_test_temp)

#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)

#             # Speichere das Feature mit der besten Leistung innerhalb der Gruppe
#             if acc > best_acc_in_group or f1 > best_f1_in_group:
#                 best_feature_in_group = candidate
#                 best_acc_in_group = acc
#                 best_f1_in_group = f1
#                 best_vif_in_group = max_vif  # Speichere den besten VIF-Wert für diese Gruppe

#         # Falls ein Feature die beste Leistung hat und VIF-Pass besteht, aufnehmen
#         if best_feature_in_group:
#             print(f"  Feature '{best_feature_in_group}' wird ausgewählt.")
#             selected_features.append(best_feature_in_group)
#             best_accuracy = best_acc_in_group
#             best_f1 = best_f1_in_group
#             results.append((selected_features.copy(), best_accuracy, best_f1, best_vif_in_group))  # Füge max_vif hinzu

#             # Markiere die Gruppe als getestet
#             if feature_group:
#                 used_feature_groups.add(feature_group)

#             # Entferne das ausgewählte Feature aus der Liste der verbleibenden Features
#             remaining_features = [f for f in remaining_features if f not in selected_features]

#     # Ergebnisse als DataFrame zurückgeben
#     result_df = pd.DataFrame(results, columns=["Selected Features", "Accuracy", "F1-Score", "Max VIF"])
    
#     return result_df

from itertools import combinations

from itertools import combinations

from itertools import combinations

from itertools import combinations

def forward_selection_vif(X_train, y_train, X_test, y_test, model, feature_groups, feature_order=None, vif_threshold=10):
    """
    Führt eine Forward Selection mit Gruppen-Prüfung und VIF-Kontrolle durch.
    Diese Funktion wählt iterativ Features aus, indem sie schrittweise das beste Feature hinzufügt, das die Modellleistung verbessert.
    Bevor ein Feature aufgenommen wird, wird überprüft, ob es zu hoher Multikollinearität führt (VIF > vif_threshold).
    
    Wenn der VIF über den Schwellenwert hinausgeht, wird versucht, Features zu entfernen und die beste Kombination zu finden.
    
    Args:
        X_train (pd.DataFrame): Trainingsdaten (Features) **ohne One-Hot-Encoding**
        y_train (pd.Series): Trainingszielvariable
        X_test (pd.DataFrame): Testdaten (Features) **ohne One-Hot-Encoding**
        y_test (pd.Series): Testzielvariable
        model (sklearn.base.BaseEstimator): Das zu verwendende Modell (muss `.fit()` und `.predict()` unterstützen)
        feature_groups (dict): Dictionary von Feature-Gruppen, in denen nur ein Feature pro Gruppe ausgewählt werden soll
        feature_order (list, optional): Vordefinierte Reihenfolge der Features nach Wichtigkeit (z. B. ANOVA/Chi²).
        vif_threshold (float, optional): Maximal erlaubter VIF-Wert (Standard = 10)
    
    Returns:
        pd.DataFrame: DataFrame mit den ausgewählten Features, Accuracy, F1-Score und maximalem VIF-Wert
    """
    selected_features = []  # Liste der bereits ausgewählten Features
    used_feature_groups = set()  # Set zur Speicherung getesteter Gruppen
    results = []  # Ergebnisse der Forward Selection

    # Falls keine Feature-Order gegeben ist, alle Features in beliebiger Reihenfolge testen
    remaining_features = [f for f in feature_order if f in X_train.columns] if feature_order else X_train.columns.tolist()
    
    # Identifiziere kategoriale Features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    best_accuracy = 0
    best_f1 = 0
    best_vif = float('inf')

    # Forward Selection mit Gruppen-Prüfung
    for feature in remaining_features:
        # Finde die Gruppe, zu der das Feature gehört (falls es eine gibt)
        feature_group = None
        for group_name, group_features in feature_groups.items():
            if feature in group_features:
                feature_group = group_name
                break  # Sobald Gruppe gefunden, abbrechen
        
        # Falls es zu einer Gruppe gehört, aber die Gruppe wurde schon getestet → weiter
        if feature_group and feature_group in used_feature_groups:
            continue

        # Falls das Feature zu einer Gruppe gehört, **teste ALLE Features aus dieser Gruppe**
        if feature_group:
            candidate_features = [f for f in feature_groups[feature_group] if f in X_train.columns]
        else:
            candidate_features = [feature]

        best_feature_in_group = None
        best_acc_in_group = 0
        best_f1_in_group = 0

        # Teste alle Features in der Gruppe
        for candidate in candidate_features:
            temp_features = selected_features + [candidate]
            
            X_train_temp = X_train[temp_features].copy()
            X_test_temp = X_test[temp_features].copy()
            
            # Falls das Feature kategorisch ist, wende One-Hot-Encoding an
            for column in X_train_temp.columns:
                if column in categorical_features:
                    X_train_temp = pd.get_dummies(X_train_temp, columns=[column], drop_first=True)
                    X_test_temp = pd.get_dummies(X_test_temp, columns=[column], drop_first=True)
                    # Sicherstellen, dass train und test gleiche Spalten haben
                    X_train_temp, X_test_temp = X_train_temp.align(X_test_temp, join='left', axis=1, fill_value=0)
            
            # Umwandlung der One-Hot-kodierten Spalten in numerische Werte
            X_train_temp = X_train_temp.astype('float64')
            X_test_temp = X_test_temp.astype('float64')

            # Berechne VIF nur, wenn mehr als ein Feature im Modell ist
            if len(temp_features) > 1:
                vif_df = calculate_vif(X_train_temp)
                max_vif = vif_df["VIF"].max()

                # Falls der VIF des neuen Feature-Sets zu hoch ist, Feature nicht hinzufügen
                # if max_vif > vif_threshold:
                #     continue  
            else:
                max_vif = 0  # Setze max_vif auf 0, wenn nur ein Feature vorhanden ist

            # Modell trainieren und evaluieren
            model.fit(X_train_temp, y_train)
            y_pred = model.predict(X_test_temp)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Speichere das Feature mit der besten Leistung innerhalb der Gruppe
            if acc > best_acc_in_group or f1 > best_f1_in_group:
                best_feature_in_group = candidate
                best_acc_in_group = acc
                best_f1_in_group = f1

        # Falls ein Feature die beste Leistung hat und VIF-Pass besteht, aufnehmen
        if best_feature_in_group:
            selected_features.append(best_feature_in_group)
            best_accuracy = best_acc_in_group
            best_f1 = best_f1_in_group

            # Berechne VIF für das aktuelle Set von Features
            if len(selected_features) > 1:
                X_train_temp = X_train[selected_features].copy()
                X_test_temp = X_test[selected_features].copy()

                # One-Hot-Encoding für kategorische Spalten
                for column in X_train_temp.columns:
                    if column in categorical_features:
                        X_train_temp = pd.get_dummies(X_train_temp, columns=[column], drop_first=True)
                        X_test_temp = pd.get_dummies(X_test_temp, columns=[column], drop_first=True)
                        # Sicherstellen, dass train und test gleiche Spalten haben
                        X_train_temp, X_test_temp = X_train_temp.align(X_test_temp, join='left', axis=1, fill_value=0)

                X_train_temp = X_train_temp.astype('float64')
                X_test_temp = X_test_temp.astype('float64')

                vif_df = calculate_vif(X_train_temp)
                max_vif = vif_df["VIF"].max()
            else:
                max_vif = 0  # Setze max_vif auf 0, wenn nur ein Feature vorhanden ist

            results.append((selected_features.copy(), best_accuracy, best_f1, max_vif))
            
            # Markiere die Gruppe als getestet
            if feature_group:
                used_feature_groups.add(feature_group)

            # Entferne das ausgewählte Feature aus der Liste der verbleibenden Features
            remaining_features = [f for f in remaining_features if f not in selected_features]

            # Falls der VIF überschritten wird, iteriere und versuche, Features zu entfernen
            if max_vif > vif_threshold:
                # Speichere die Kombination der Features, die den VIF überschreiten
                problematic_features = selected_features.copy()

                # Iteriere über alle Kombinationen von Features und entferne jeweils eins
                best_combination = None
                best_combination_acc = -1
                best_combination_f1 = -1
                best_combination_vif = float('inf')

                for i in range(len(problematic_features)):
                    reduced_features = [f for j, f in enumerate(problematic_features) if j != i]
                    
                    X_train_reduced = X_train[reduced_features].copy()
                    X_test_reduced = X_test[reduced_features].copy()

                    # One-Hot-Encoding für kategorische Spalten
                    for column in X_train_reduced.columns:
                        if column in categorical_features:
                            X_train_reduced = pd.get_dummies(X_train_reduced, columns=[column], drop_first=True)
                            X_test_reduced = pd.get_dummies(X_test_reduced, columns=[column], drop_first=True)
                            # Sicherstellen, dass train und test gleiche Spalten haben
                            X_train_reduced, X_test_reduced = X_train_reduced.align(X_test_reduced, join='left', axis=1, fill_value=0)

                    X_train_reduced = X_train_reduced.astype('float64')
                    X_test_reduced = X_test_reduced.astype('float64')

                    # Modell trainieren und evaluieren
                    model.fit(X_train_reduced, y_train)
                    y_pred = model.predict(X_test_reduced)

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # Berechne den VIF für die reduzierte Feature-Kombination
                    vif_reduced_df = calculate_vif(X_train_reduced)
                    max_vif_reduced = vif_reduced_df["VIF"].max()

                    # Wenn der VIF unter dem Schwellenwert liegt und die Accuracy besser ist, speichere das Set
                    if max_vif_reduced <= vif_threshold and (acc > best_combination_acc or (acc == best_combination_acc and f1 > best_combination_f1)):
                        best_combination = reduced_features
                        best_combination_acc = acc
                        best_combination_f1 = f1
                        best_combination_vif = max_vif_reduced

                # Falls eine bessere Kombination gefunden wurde, ersetze die ausgewählten Features
                if best_combination:
                    selected_features = best_combination
                    best_accuracy = best_combination_acc
                    best_f1 = best_combination_f1
                    max_vif = best_combination_vif
                    results.append((selected_features.copy(), best_accuracy, best_f1, max_vif))

    # Ergebnisse als DataFrame zurückgeben
    result_df = pd.DataFrame(results, columns=["Selected Features", "Accuracy", "F1-Score", "VIF"])
    
    return result_df
