import numpy as np
import matplotlib.pyplot as plt


from Treat_Images.TreatImages import import_images2, import_images



def der_prof(profile_image, fen=1):
    """
    Calcule une dérivée numérique centrée d'un profil 1D.

    Les fen premiers et derniers éléments sont mis à zéro.

    Args:
        profile_image (array-like): Profil d'intensité.
        fen (int): Demi-largeur du calcul de la dérivée.

    Returns:
        np.ndarray: Dérivée numérique du profil.
    """
    profile = np.asarray(profile_image, dtype=np.float64)

    if profile.ndim != 1:
        raise ValueError("Le profil doit être un tableau unidimensionnel.")

    if not isinstance(fen, (int, np.integer)) or fen < 1:
        raise ValueError("fen doit être un entier strictement positif.")

    der = np.zeros_like(profile)

    if profile.size > 2 * fen:
        der[fen:-fen] = (
            profile[2 * fen:] - profile[:-2 * fen]
        ) / (2 * fen)

    return der


def der_seconde_profiles(serie_profiles, fen=1):
    """
    Calcule la dérivée seconde de chaque profil.

    Args:
        serie_profiles: Séquence de profils d'intensité.
        fen (int): Paramètre du calcul des dérivées.

    Returns:
        list[np.ndarray]: Dérivées secondes des profils.
    """
    derivees_secondes = []

    for profil in serie_profiles:
        derivee_premiere = der_prof(profil, fen)
        derivee_seconde = der_prof(derivee_premiere, fen)
        derivees_secondes.append(derivee_seconde)

    return derivees_secondes


def indice_positif(der_seconde_profile):
    """
    Renvoie le premier indice où la dérivée seconde est strictement positive.

    Renvoie -1 si aucun élément positif n'est trouvé.
    """
    derivee = np.asarray(der_seconde_profile)

    indices_positifs = np.flatnonzero(derivee > 0)

    if indices_positifs.size == 0:
        return -1

    return int(indices_positifs[0])


def indice_positif_series(der_profiles):
    """
    Détermine le premier indice positif pour chaque dérivée seconde.
    """
    return [
        indice_positif(profil)
        for profil in der_profiles
    ]


def indice_image(indices, seuil):
    """
    Détecte la première variation importante entre deux indices successifs.

    Les valeurs -1, qui indiquent l'absence de détection,
    sont ignorées en tant que points de comparaison.

    Returns:
        int | None: Indice de la première variation détectée,
        ou None si aucune variation n'est détectée.
    """
    if seuil < 0:
        raise ValueError("Le seuil doit être positif ou nul.")

    indices = np.asarray(indices)

    if indices.size < 2:
        return None

    valides = indices != -1

    changements = (
        valides[:-1]
        & valides[1:]
        & (np.abs(np.diff(indices)) > seuil)
    )

    positions = np.flatnonzero(changements)

    if positions.size == 0:
        return None

    return int(positions[0])


def profil_des_images(images, colonne=0):
    """
    Extrait le profil vertical d'intensité de chaque image en niveaux de gris.

    Args:
        images: Séquence d'images 2D.
        colonne (int): Indice de la colonne à analyser.

    Returns:
        list[np.ndarray]: Un profil vertical par image.
    """
    profils = []

    for numero, image in enumerate(images):
        image = np.asarray(image)

        if image.ndim != 2:
            raise ValueError(
                f"L'image {numero} doit être en niveaux de gris (2D)."
            )

        if not 0 <= colonne < image.shape[1]:
            raise ValueError(
                f"La colonne {colonne} est invalide pour "
                f"l'image {numero} de largeur {image.shape[1]}."
            )

        profils.append(image[:, colonne].astype(np.float64))

    return profils


def afficher_resultats(images, profils, derivees_secondes, indices,
                       indice_deplacement, fen):
    """
    Affiche l'évolution des indices détectés et les profils de la première image.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        indices,
        color="blue",
        linestyle="-",
        marker="o",
        markersize=4,
        label="Premier indice positif",
    )

    if indice_deplacement is not None:
        ax.axvline(
            indice_deplacement,
            color="red",
            linestyle="--",
            label=f"Déplacement détecté : image {indice_deplacement}",
        )

    ax.set_title("Évolution des indices détectés")
    ax.set_xlabel("Numéro de l'image")
    ax.set_ylabel("Indice du profil")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    profil = profils[0]
    derivee_seconde = derivees_secondes[0]
    positions = np.arange(profil.size)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(images[0], cmap="gray")
    ax[0].axvline(
        # Colonne analysée : définie dans le profil extrait.
        200,
        color="red",
        linestyle="--",
    )
    ax[0].set_title("Première image")
    ax[0].set_xlabel("Colonne")
    ax[0].set_ylabel("Ligne")

    ax[1].plot(profil, positions)
    ax[1].set_title("Profil d'intensité")
    ax[1].set_xlabel("Intensité")
    ax[1].set_ylabel("Position verticale")
    ax[1].invert_yaxis()
    ax[1].grid(True, alpha=0.4)

    ax[2].plot(derivee_seconde, positions)
    ax[2].axvline(0, color="black", linewidth=0.8)
    ax[2].set_title(f"Dérivée seconde (fen = {fen})")
    ax[2].set_xlabel("Dérivée seconde")
    ax[2].set_ylabel("Position verticale")
    ax[2].invert_yaxis()
    ax[2].grid(True, alpha=0.4)

    fig.tight_layout()
    plt.show()



def First_Displacement(LIEN_IMAGES):



        FEN = 50
        COLONNE = 400
        SEUIL = 1

        # Importation et filtrage des images.
        images = import_images2(LIEN_IMAGES)
        #images = filtered_images(images)
        images = np.asarray(images)

        if images.ndim != 3 or len(images) == 0:
            raise ValueError(
                "Les images doivent former un tableau de forme "
                "(nombre_images, hauteur, largeur)."
            )

        print(f"Nombre d'images : {len(images)}")
        print(f"Dimensions d'une image : {images[0].shape}")

        # Extraction des profils verticaux.
        profils = profil_des_images(images, COLONNE)

        # Calcul des dérivées secondes.
        derivees_secondes = der_seconde_profiles(profils, FEN)

        # Détection du premier indice positif pour chaque image.
        indices = indice_positif_series(derivees_secondes)
        print("Indices détectés :", indices)

        # Détection du premier changement suffisamment important.
        indice_deplacement = indice_image(indices, SEUIL)

        if indice_deplacement is None:
            print("Aucun déplacement détecté avec le seuil choisi.")
        else:
            print(
                "Indice de la première variation détectée :",
                indice_deplacement,
            )

        afficher_resultats(
            images,
            profils,
            derivees_secondes,
            indices,
            indice_deplacement,
            FEN,
        )

        return indice_deplacement



if __name__ == "__main__":

    ind = First_Displacement("Generated_Images/*png")