\# Fiber Displacement and Young's Modulus Measurement



The goal of this small project is to measure the displacement and deformation of a fiber.



We acquire different images of a physical experiment, put some ink on the fiber (two dots) and measure the norm between them.



To test this code we artificially generate images with python `Create\_Images/Create\_Images.py` saved in `Generated\_Images` folder.



With python `main.py` we can start the measurement process by selecting two ROI around each dot, so we know where we need to search the barycenter.



The result is saved in `Normes\_sys1\_sys2.csv`.



`First\_Displacement\_Detector.py` is used to synchronise the images and the machine force measurement in the `.TRA` file (that we dont give here).



Because in real life experiment the image acquisition and machine acquisition dont start at the same time, we compute the second derivative of the intensity variation to know when the mouvement start.



Of course because `Generated\_Images` is an perfect uniform mouvement, indice is 0.



`Young\_Module.py` assemble all previous information of `Normes\_sys1\_sys2.csv`, indice and the TRA file to compute the Young modulus of the fiber.



\---



\## Prerequisites \& Installation



\### 1. Clone the Repository

```bash

git clone https://github.com/rasoul24/image-point-tracking.git

cd Image\_Point\_Tracking

