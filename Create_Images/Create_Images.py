import numpy as np
from tqdm import tqdm
from skimage.util import random_noise
from pathlib import Path
from PIL import Image

def creer_image_cercle(
    lon_image,
    lar_image,
    x_point1,
    y_point1,
    x_point2,
    y_point2,
    rayon_cercle,
    niveau_gris,
):
    """
    Crée une image en niveaux de gris contenant deux cercles noirs.

    Les coordonnées suivent la convention du code initial :
    x correspond à l'indice de ligne et y à l'indice de colonne.

    Args:
        lon_image (int): Nombre de lignes de l'image.
        lar_image (int): Nombre de colonnes de l'image.
        x_point1, y_point1: Coordonnées du premier centre.
        x_point2, y_point2: Coordonnées du second centre.
        rayon_cercle (float): Rayon des cercles en pixels.
        niveau_gris (int): Niveau de gris du fond, entre 0 et 255.

    Returns:
        np.ndarray: Image en niveaux de gris de type uint8.
    """
    if lon_image <= 0 or lar_image <= 0:
        raise ValueError("Les dimensions de l'image doivent être positives.")

    if rayon_cercle < 0:
        raise ValueError("Le rayon doit être positif ou nul.")

    if not 0 <= niveau_gris <= 255:
        raise ValueError("Le niveau de gris doit être compris entre 0 et 255.")

    image = np.full(
        (lon_image, lar_image),
        niveau_gris,
        dtype=np.uint8,
    )

    # Grille de coordonnées : évite de parcourir les pixels en Python.
    lignes, colonnes = np.ogrid[:lon_image, :lar_image]
    rayon2 = rayon_cercle**2

    # Masque du premier cercle.
    masque1 = (
        (colonnes - x_point1)**2
        + (lignes - y_point1)**2
        < rayon2
    )

    # Masque du second cercle.
    masque2 = (
        (colonnes - x_point2)**2
        + (lignes - y_point2)**2
        < rayon2
    )

    # Les pixels appartenant à l'un ou l'autre cercle deviennent noirs.
    image[masque1 | masque2] = 0

    return image


def creer_serie_images_deux_points(
    lon_image,
    lar_image,
    x_point1,
    y_point1,
    x_point2,
    y_point2,
    nombre_images,
    r,
    niveau_gris,
):
    """
    Génère une série d'images dans laquelle deux points se déplacent.

    Le premier centre se déplace de -2 pixels en x par image.
    Le second centre se déplace de +2 pixels en x par image.

    Returns:
        list[np.ndarray]: Liste des images générées.
    """
    if nombre_images < 0:
        raise ValueError("Le nombre d'images ne peut pas être négatif.")

    dx1 = -2
    dx2 = 2
    images = []

    for _ in range(nombre_images):
        image = creer_image_cercle(
            lon_image,
            lar_image,
            x_point1,
            y_point1,
            x_point2,
            y_point2,
            r,
            niveau_gris,
        )

        images.append(image)

        x_point1 += dx1
        x_point2 += dx2

    return images


def bruit(type_mode, images):
    """
    Ajoute un bruit aux images avec scikit-image.

    Args:
        type_mode (str): Type de bruit accepté par random_noise.
        images (list[np.ndarray]): Images en niveaux de gris.

    Returns:
        list[np.ndarray]: Nouvelles images bruitées en uint8.

    Remarque:
        Les images originales ne sont pas modifiées.
    """
    images_bruitees = []

    for image in tqdm(images, desc="Bruitage", unit="image"):
        # random_noise travaille sur des valeurs normalisées entre 0 et 1.
        image_normalisee = image.astype(np.float32) / 255.0

        bruit_image = random_noise(
            image_normalisee,
            mode=type_mode,
            clip=True,
        )

        image_uint8 = np.clip(
            bruit_image * 255.0,
            0,
            255,
        ).astype(np.uint8)

        images_bruitees.append(image_uint8)

    return images_bruitees

def CreateImages(
    lon_image = 800,
    lar_image = 800,

    x_point1 = 300,
    y_point1 = 400,

    x_point2 = 400,
    y_point2 = 400,
    nombre_images = 50,
    r = 10 ,
    niveau_gris = 255,

                ):

    images =  creer_serie_images_deux_points(
    lon_image,
    lar_image,
    x_point1,
    y_point1,
    x_point2,
    y_point2,
    nombre_images,
    r,
    niveau_gris,

        )

    return images

def enregistrer_images(images, dossier="../Generated_Images"):
    """
    Enregistre une série d'images au format PNG.

    Chaque image est nommée image_0000.png, image_0001.png, etc.
    """
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        image = np.asarray(image)

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        chemin = dossier / f"image_{i:04d}.png"
        Image.fromarray(image).save(chemin)

    print(f"{len(images)} images enregistrées dans : {dossier.resolve()}")

if __name__ == "__main__":

    images = CreateImages()

    enregistrer_images(images)






