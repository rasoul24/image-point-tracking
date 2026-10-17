# Fiber Displacement and Young's Modulus Measurement

**An experimental project to measure the displacement, deformation, and Young's modulus of a fiber using image processing and mechanical data synchronization.**

The project acquires images of a physical experiment, tracks two ink dots marked on the fiber to measure their distance (norm) variation, and synchronizes this with machine force data from a `.TRA` file.

> **Project status:** Research prototype / Experimental codebase.

## Overview

The system processes visual input and mechanical logs to compute material properties.

To test the workflow without real experimental data, synthetic images can be generated using `Create_Images/Create_Images.py`. 

The primary measurement starts with `main.py` via interactive ROI selection around each dot, storing results in `Normes_sys1_sys2.csv`. Because camera and machine acquisitions typically start at different moments, `First_Diplacement_Detector.py` synchronizes them using the second derivative of intensity variation. Finally, `Young_Module.py` combines everything to compute the fiber's Young's modulus.

## Key Features

| Component | Description |
|---|---|
| **Synthetic Data Generation** | Generates artificial uniform movement test images |
| **Dot Tracking** | Interactive ROI selection and barycenter tracking |
| **Data Synchronization** | 2nd derivative intensity analysis to align image frames with `.TRA` machine logs |
| **Mechanical Analysis** | Computes the Young's modulus of the fiber |

## Architecture

```text
Synthetic / Real Images
      |
      v
main.py (ROI Selection & Barycenter Tracking)
      |
      v
Normes_sys1_sys2.csv
      |
      v
First_Diplacement_Detector.py <---> .TRA Machine File
      |                                      |
      +------------------+-------------------+
                         |
                         v
                Young_Module.py
                         |
                         v
               Young's Modulus Output


```


### 1. Clone the repository

```bash
git clone https://github.com/rasoul24/image-point-tracking.git
cd Image_Point_Tracking
```


### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in **Windows Git Bash**:

```bash
source .venv/Scripts/activate
```

On Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```





## Author

**Rasoul Islangueriev**

[GitHub](https://github.com/rasoul24)