from tqdm import tqdm
import io
import numpy as np
import glob
from skimage import io, filters
from skimage.transform import rescale

def import_images(lien_images, nombre_images=None):
    """
    Importer un nombre limité d'images à partir d'un chemin donné.

    :param lien_images: Chemin vers les images (avec un motif, ex. "*.png").
    :param nombre_images: Nombre maximum d'images à importer. Si None, importe toutes les images.
    :return: Liste des images importées.
    """
    # Récupérer les chemins des images correspondant au motif
    chemins = glob.glob(lien_images)

    # Limiter le nombre d'images si demandé
    if nombre_images is not None:
        chemins = chemins[:nombre_images]

    # Charger les images
    collection = list(tqdm([io.imread(chemin) for chemin in chemins],desc="Importation des image"))
    return collection

def import_images2(lien_images):
    """
    Importer un nombre limité d'images à partir d'un chemin donné.

    :param lien_images: Chemin vers les images (avec un motif, ex. "*.png").
    :param nombre_images: Nombre maximum d'images à importer. Si None, importe toutes les images.
    :return: Liste des images importées.
    """

    serie_images_importees = io.imread_collection(lien_images)

    return np.array(serie_images_importees)

def filtered_images (serie_images_sans_filtres):
    """
    Appliquer un filtre basé sur le seuil d'Otsu à chaque image de la série.

    :param serie_images: Liste des images (tableaux NumPy) à traiter.
    :return: Liste des images binarisées après traitement.
    """
    # Liste pour stocker les images modifiées
    images_filtrees = []

    for image in tqdm(serie_images_sans_filtres,desc="Filtrage des images"):
        # Calculer le seuil d'Otsu
        thr = filters.threshold_otsu(image)

        # Binariser l'image
        image_filtree = image > thr/2

        # Convertir en format uint8 pour les valeurs 0 et 255
        image_filtree = (image_filtree * 255).astype(np.uint8)

        # Ajouter l'image filtrée à la liste
        images_filtrees.append(image_filtree)

    return images_filtrees

def lissage(serie_images_non_lissees,s):
    """
    Lisse les images par interpolation avec un facteur s
    Retourne une liste d'images que nous avons lisser
    """
    res=[]
    for image in tqdm(serie_images_non_lissees,desc="Lissage des images"):
        res.append(rescale(image,s))
    return res