# Imports standard
import copy
import pandas as pd

# Imports tiers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from memory_profiler import profile
from tqdm import tqdm


class Systeme:
    """
    Définition de la classe Systeme qui regroupe les différentes méthodes
    pour notre étude
    """
    def __init__(self, images,seuil,nom):
        self.nom=nom
        self.l_bar=[]
        self.images = images
        self.zoi_x1 = 0
        self.zoi_y1 = 0
        self.zoi_x2 = 0
        self.zoi_y2 = 0

        self.fig, self.ax = plt.subplots()
        self.implot = self.ax.imshow(images[0],aspect='equal',
        origin='lower', cmap='gray')

        self.active_roi()
        plt.show()
        self.zoi_init_x1=self.zoi_x1
        self.zoi_init_y1=self.zoi_y1
        self.zoi_init_x2=self.zoi_x2
        self.zoi_init_y2=self.zoi_y2

        self.long= self.zoi_init_x2-self.zoi_init_x1
        self.lar=self.zoi_init_y2-self.zoi_init_y1
        self.IS = seuil

    def active_roi(self):
        """
        Active un sélecteur rectangulaire pour permettre à l'utilisateur
        de choisir une région d'intérêt (ROI).
        """
        # Création du RectangleSelector et association de la méthode de sélection
        self.rec = RectangleSelector(
            self.ax,
            self.select_roi,  # Appelle cette méthode lorsqu'une sélection est faite
            interactive=True
        )

    def select_roi(self, eclick, erelease):
        """
        Méthode appelée lorsque l'utilisateur sélectionne une région.
        Définit les coordonnées du ROI en fonction des clics.

        :param eclick: Événement de clic initial.
        :param erelease: Événement de relâchement du clic.
        """
        # Récupération des coordonnées de la sélection
        self.zoi_x1, self.zoi_x2 = int(eclick.xdata), int(erelease.xdata)
        self.zoi_y1, self.zoi_y2 = int(eclick.ydata), int(erelease.ydata)

        # Affichage des coordonnées de la ROI dans la console
        print(f"ROI sélectionnée: x1={self.zoi_x1}, y1={self.zoi_y1}, x2={self.zoi_x2},y2={self.zoi_y2}")



    def barycentre_p(self, num_image):
        """
        Calcule le barycentre des pixels noirs dans la ROI et déplace
        la ROI selon le déplacement du barycentre par rapport à l'image
        précédente. Affiche des diagnostics pour identifier les erreurs.
        """

        image = self.images[num_image]

        # Vérification des dimensions de l'image.
        if image.ndim != 2:
            raise ValueError(
                f"Image {num_image} : image attendue en niveaux de gris, "
                f"mais shape={image.shape}."
            )

        hauteur, largeur = image.shape

        # Coordonnées de la ROI avant calcul.
        x1 = int(round(self.zoi_x1))
        y1 = int(round(self.zoi_y1))
        x2 = int(round(self.zoi_x2))
        y2 = int(round(self.zoi_y2))

        print(f"\n{'=' * 55}")
        print(f"IMAGE {num_image}")
        print(f"Dimensions image : largeur={largeur}, hauteur={hauteur}")
        print(
            f"ROI avant calcul : "
            f"x=[{x1}, {x2}], y=[{y1}, {y2}]"
        )

        # Limiter la ROI aux dimensions de l'image.
        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(largeur, x2)
        y2_clip = min(hauteur, y2)

        if x1_clip >= x2_clip or y1_clip >= y2_clip:
            raise ValueError(
                f"ROI vide ou invalide pour l'image {num_image} : "
                f"x=[{x1_clip}, {x2_clip}], "
                f"y=[{y1_clip}, {y2_clip}]."
            )

        roi = image[y1_clip:y2_clip, x1_clip:x2_clip]

        # Diagnostic des valeurs des pixels.
        valeurs, effectifs = np.unique(roi, return_counts=True)
        print(f"Valeurs de pixels dans la ROI : {dict(zip(valeurs, effectifs))}")

        # Détection des pixels noirs.
        ys, xs = np.nonzero(roi == 0)
        nombre_noirs = len(xs)

        print(f"Nombre de pixels noirs : {nombre_noirs}")
        print(f"Dimensions de la ROI : {roi.shape}")

        if nombre_noirs == 0:
            raise ValueError(
                f"Aucun pixel noir dans la ROI, image {num_image}."
            )

        # Barycentre en coordonnées globales :
        # x = colonne, y = ligne.
        x_g = float(xs.mean() + x1_clip)
        y_g = float(ys.mean() + y1_clip)

        print(f"Barycentre détecté : x={x_g:.3f}, y={y_g:.3f}")

        # Vérifier les déplacements par rapport à l'image précédente.
        if num_image > 0:
            if not self.l_bar:
                raise ValueError(
                    "Le barycentre de l'image précédente est absent de self.l_bar."
                )

            x_precedent, y_precedent = self.l_bar[-1]

            dx = x_g - x_precedent
            dy = y_g - y_precedent

            print(
                f"Barycentre précédent : "
                f"x={x_precedent:.3f}, y={y_precedent:.3f}"
            )
            print(f"Déplacement détecté : dx={dx:.3f}, dy={dy:.3f}")

            # Diagnostic : déplacement nul ou quasi nul.
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                print(
                    "ATTENTION : déplacement nul. "
                    "Vérifie que les deux images diffèrent bien "
                    "dans la ROI et que le bon objet est suivi."
                )

            # Mettre en évidence un déplacement sur un seul axe.
            if abs(dx) < 1e-9 and abs(dy) > 1e-9:
                print(
                    "DIAGNOSTIC : déplacement uniquement vertical "
                    "(dy non nul, dx nul). Vérifie la convention "
                    "x=colonne et y=ligne dans la génération des images."
                )
            elif abs(dy) < 1e-9 and abs(dx) > 1e-9:
                print(
                    "DIAGNOSTIC : déplacement uniquement horizontal."
                )

            # Déplacement de la ROI.
            self.zoi_x1 += dx
            self.zoi_x2 += dx
            self.zoi_y1 += dy
            self.zoi_y2 += dy

            print(
                f"ROI après déplacement : "
                f"x=[{self.zoi_x1:.3f}, {self.zoi_x2:.3f}], "
                f"y=[{self.zoi_y1:.3f}, {self.zoi_y2:.3f}]"
            )

        print(f"{'=' * 55}")

        return x_g, y_g






    def liste_barycentres(self):
        """""
        Calcule une liste composée des coordonnées des barycentres des différentes images
        """
        for i in tqdm(range(0,len(self.images)),desc="Creation de la liste des barycentres"):
            self.l_bar.append(self.barycentre_p(i))
        print(self.l_bar)


    def affiche_barycentres(self):
        """""
        Affiche les barycentres du systeme
        """
        print(self.l_bar)

    def sauver_barycentres(self):
        """
        Sauvgarde la liste des barycentres dans un fichier texte
        """
        a = np.asarray(self.l_bar)

        np.savetxt(f"Fichier_barycentres_{self.nom}.txt", a)

class SystemeN:
    """"
    Classe systeme a n compossantes qui géneralise la classe Systeme pour un ensemble
    de points
    """
    def __init__(self, liste_systeme, images, seuil):
        self.liste_systeme = liste_systeme  # Liste des systèmes individuels
        self.dictionnaire = {}  # Dictionnaire des barycentres
        self.images = images  # Liste des images
        self.seuil = seuil  # Seuil utilisé pour une éventuelle analyse
        self.liste_barycentres_systemes()

    def liste_barycentres_systemes(self):
        """
        Remplit le dictionnaire avec les barycentres de chaque système.
        """
        for systeme in (self.liste_systeme):
            # Vérifiez que chaque système a un attribut 'nom' et une méthode 'liste_barycentres'
            if hasattr(systeme, 'nom') and hasattr(systeme, 'liste_barycentres'):
                # Ajouter le barycentre au dictionnaire
                self.dictionnaire[systeme.nom] = systeme.l_bar
                print(systeme.l_bar)
            else:
                raise AttributeError(
                    f"Le système {systeme} doit avoir un attribut 'nom'et une méthode'liste_barycentres'.")
            print("Le dictionnaire est bien construit")


    def norme(self,num_image,nom1,nom2):
        """"
        Calcule la norme entre les deux points à partir des coordonnÃ©es
        des barycentres
        """
        # Calcul de la norme entre deux points P1 et P2
        x1_g, y1_g = self.dictionnaire[nom1][num_image]
        x2_g, y2_g = self.dictionnaire[nom2][num_image]
        return float(((x2_g - x1_g)**2 + (y2_g- y1_g)**2)**0.5)


    def liste_normes(self, nom1, nom2):
        p1 = np.asarray(self.dictionnaire[nom1], dtype=float)
        p2 = np.asarray(self.dictionnaire[nom2], dtype=float)

        distances = np.linalg.norm(p2 - p1, axis=1)

        return np.round(distances, 2).tolist()

    def sauvergarder_normes(self,nom1,nom2):
        """
        Sauvgarde les normes dans un fichier
        """
        data = {
        "Normes": self.liste_normes(nom1,nom2)
                }
        # Créer un DataFrame
        df = pd.DataFrame(data)

        # Sauvegarder le DataFrame dans un fichier CSV
        df.to_csv(f"Normes_{nom1}_{nom2}.csv", index=False)  # index=False pour ne pas enregistrer l'index


    def affiche_normes(self,nom1,nom2):
        """
        Affiche les normes
        """
        a=self.liste_normes(nom1,nom2)
        for i in range(0,len(self.images)):
            print("La norme a l'image ",i,"entre",nom1,"et",nom2,"vaut",a[i])


    def tracer_normes(self, nom1, nom2):
        """
        Affiche les normes des déformations par image.
        """

        a = self.liste_normes(nom1, nom2)

        plt.figure(figsize=(8, 6))
        plt.plot(a, color='blue', markersize=5, linestyle='-', linewidth=2, label="Normes des déformations")

     # Ajout du titre et des labels
        plt.title("Déformation de chaque image", fontsize=14, fontweight="bold")
        plt.xlabel("Numéro d'image", fontsize=12)
        plt.ylabel("Déformation (pixels)", fontsize=12)

        # Affichage de la grille
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

        # Ajout de la légende
        plt.legend(fontsize=12)

        # Annotation du maximum pour mieux visualiser la valeur la plus élevée
        max_value = max(a)
        max_index = a.index(max_value)
        plt.annotate(f"Max: {max_value:.2f}",
                xy=(max_index, max_value),
                xytext=(max_index + 5, max_value + 5),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=12, color='red')

        # Affichage du graphe
        plt.show()


    def copy(self):
        """"
        Renvoie une copie de notre objet pour qu'on puisse
        faire des comparaisons
        """
        return copy.deepcopy(self)



    @profile
    def animation(self):
        """
        Anime le déplacement des points et des ROI.
        Optimisation : les éléments graphiques sont créés une seule fois,
        puis mis à jour à chaque frame.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.animation import FuncAnimation
        from matplotlib.lines import Line2D

        n_frames = len(self.images)
        if n_frames == 0:
            return

        fig, ax = plt.subplots()

        # Affichage de la première image
        img = ax.imshow(
            self.images[0],
            aspect="equal",
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=255,
            animated=True
        )

        ax.set_title("Suivi des points")
        ax.set_xlabel("Position X (pixels)")
        ax.set_ylabel("Position Y (pixels)")

        # Compteur de frames : un seul objet texte
        frame_text = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            color="red",
            fontsize=10,
            ha="left",
            va="top"
        )

        rectangles = []
        labels = []

        # Création des ROI et des étiquettes une seule fois
        for systeme in self.liste_systeme:
            x1, y1 = systeme.zoi_x1, systeme.zoi_y1
            x2, y2 = systeme.zoi_x2, systeme.zoi_y2

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                animated=True
            )
            ax.add_patch(rect)
            rectangles.append(rect)

            label = ax.text(
                x1, y1 - 10,
                systeme.nom,
                color="red",
                fontsize=10,
                ha="center",
                va="center",
                animated=True
            )
            labels.append(label)

        # Une seule ligne par paire de points
        lignes = []
        paires = []

        for i in range(len(self.liste_systeme)):
            for j in range(i + 1, len(self.liste_systeme)):
                s1 = self.liste_systeme[i]
                s2 = self.liste_systeme[j]

                line = Line2D(
                    [], [],
                    color="red",
                    linewidth=2,
                    animated=True
                )
                ax.add_line(line)
                lignes.append(line)
                paires.append((s1.nom, s2.nom))

        def init():
            img.set_data(self.images[0])
            frame_text.set_text(f"Image 0 / {n_frames - 1}")

            artists = [img, frame_text]
            artists.extend(rectangles)
            artists.extend(labels)
            artists.extend(lignes)
            return artists

        def update(frame):
            img.set_data(self.images[frame])
            frame_text.set_text(f"Image {frame} / {n_frames - 1}")

            # Mise à jour des ROI
            for systeme, rect, label in zip(
                self.liste_systeme, rectangles, labels
            ):
                rect.set_xy((systeme.zoi_x1, systeme.zoi_y1))
                rect.set_width(systeme.zoi_x2 - systeme.zoi_x1)
                rect.set_height(systeme.zoi_y2 - systeme.zoi_y1)

                label.set_position((
                    (systeme.zoi_x1 + systeme.zoi_x2) / 2,
                    systeme.zoi_y1 - 10
                ))

            # Mise à jour des liaisons entre barycentres
            for line, (nom1, nom2) in zip(lignes, paires):
                x1, y1 = self.dictionnaire[nom1][frame]
                x2, y2 = self.dictionnaire[nom2][frame]
                line.set_data([x1, x2], [y1, y2])

            return init_artists()

        def init_artists():
            artists = [img, frame_text]
            artists.extend(rectangles)
            artists.extend(labels)
            artists.extend(lignes)
            return artists

        ani = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            init_func=init,
            blit=True,
            interval=30,
            repeat=False,
            cache_frame_data=False
        )

        plt.show()
