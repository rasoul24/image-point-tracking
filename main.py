import matplotlib.pyplot as plt
from Treat_Images.TreatImages import import_images2, import_images, lissage, filtered_images
from Tracking_System.System import Systeme, SystemeN
from tqdm import tqdm
from skimage.morphology import square,closing
import sys

def main(images_path = "Generated_Images/*png"):

    IS=255


    serie_images_importees = import_images(images_path, 50)

    print("nombre images  = ", len(serie_images_importees))

    # serie_images_lissees=lissage(serie_images_importees,2)

    # serie_images=filtered_images(serie_images_importees)
    # serie_images_closed=[]

    # for indice in tqdm(range(len(serie_images)),desc="closing"):
    #     serie_images_closed.append(closing(serie_images[indice],square(4)))


    s1=Systeme(serie_images_importees, IS,'sys1')
    s2=Systeme(serie_images_importees, IS,'sys2')


    s1.liste_barycentres()
    s2.liste_barycentres()
    s1.sauver_barycentres()
    s2.sauver_barycentres()

    s1.affiche_barycentres()
    s2.affiche_barycentres()

    L=[]
    L.append(s1)
    L.append(s2)

    Syst_two_points = SystemeN(L, serie_images_importees, IS)

    liste_Normes = Syst_two_points.liste_normes(s1.nom,s2.nom)
    Syst_two_points.tracer_normes(s1.nom,s2.nom)
    Syst_two_points.sauvergarder_normes(s1.nom,s2.nom)
    Syst_two_points.animation()


if __name__ == "__main__":

    main()