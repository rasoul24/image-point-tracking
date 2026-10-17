import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from projet_version_10 import *

def get_column_from_file(file_path, column_name, file_type='csv', **kwargs):
    """
    Charge un fichier et retourne une colonne spécifique.

    :param file_path: Chemin du fichier (CSV, Excel, etc.)
    :param column_name: Nom de la colonne à récupérer
    :param file_type: Type de fichier (par défaut 'csv', peut être 'excel' pour des fichiers .xlsx)
    :param kwargs: Autres arguments supplémentaires pour pd.read_csv ou pd.read_excel
    :return: Liste des valeurs de la colonne demandée.
    """
    # Charger le fichier en fonction de son type
    if file_type == 'csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_type == 'excel':
        df = pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError("Type de fichier non supporté. Utilisez 'csv' ou 'excel'.")

    # Vérifier si la colonne existe dans le fichier
    if column_name not in df.columns:
        raise ValueError(f"Colonne '{column_name}' non trouvée dans le fichier.")

    # Retourner la colonne demandée sous forme de liste
    return df[column_name].tolist()
    ######################################################

if __name__ == "__main__":
    # Chemin vers les fichiers
    file_images = "essai_1/image.csv"
    file_deformations="Normes_sys1_sys2.csv"
    file_TRA = "essai_1.TRA"
    ######################################################

    # Récupérer la colonne de temps
    try:
        temps_camera = get_column_from_file(file_images,"Time_0_0")
        print("Colonne 'Time_0_0' récupérée ")
    except ValueError as e:
        print(e)

    try:
        deformations_images = get_column_from_file(file_deformations, "Normes")
        print("Colonne deformations_images récupérée")
    except ValueError as e:
        print(e)
    ######################################################


    # Récupérer les colonnes de temps, déplacement machine et force
    temps_TRA, deplacement_machine, force_TRA = lire_donnees(file_TRA)
    ######################################################

    # On récupère les images seulement à partir de l'image 17
    temps_camera = np.array(temps_camera[17:])
    temps_camera -= temps_camera[0]
    maxi = max(temps_TRA)
    mini = min(temps_TRA)
    temps_camera = temps_camera[(temps_camera >= mini) & (temps_camera <= maxi)]
    deformations_images = np.array(deformations_images[17:])
    ######################################################

    #INTERPOLATION
    interpolator = interp1d(temps_TRA, force_TRA, kind='linear', fill_value="extrapolate")
    force_TRA_interpolated = interpolator(temps_camera)

    ######################################################


    #CALCULE DES DEFORMATIONS
    deformations = deformation_nominale(deformations_images,deformations_images[0])
    deformations = np.array(deformations)

    ######################################################


    # On convertit les déformations en mm
    deformations_mm = (60 * deformations) / 1222
    ######################################################


    # On calcule les contraintes
    contraintes = contrainte_nominale(force_TRA_interpolated)
    contraintes = contraintes[:283]
    ######################################################


    # Préparation pour la régression linéaire
    model = LinearRegression()
    Y = contraintes[:30]
    X = deformations_mm[:30].reshape(-1, 1)  # Reshape pour avoir une seule feature
    # Entraîner le modèle
    model.fit(X, Y)
    # Prédictions du modèle
    y_pred = model.predict(X)
    ######################################################





    plt.figure(figsize=(8, 6))
    plt.plot(deformations_mm, contraintes, label="Données réelles", color='blue', marker='o', markersize=4, linestyle='-', linewidth=2)
    # Tracé de la régression linéaire en pointillés
    plt.plot(X, y_pred, label="Régression linéaire", linestyle='--', color='red', linewidth=2)
    # Ajout des labels et du titre
    plt.xlabel("Déformations (mm)", fontsize=12)
    plt.ylabel("Contraintes (N/mm²)", fontsize=12)
    plt.title("Contraintes en fonction des déformations", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()
    ######################################################

    # Afficher les résultats
    print(f"Coefficient de régression (pente) : {model.coef_[0]} MPa")
    print(f"Intercept (ordonnée à l'origine) : {model.intercept_}")
    ######################################################