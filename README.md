# Satellite-Derived Bathymetry (SDB) with Geospatial Artificial Intelligence (GeoAI) Model

A comprehensive Machine Learning workflow for estimating shallow water depth (bathymetry) using **Sentinel-2** and **Google Satellite Embeddings**. This repository provides an end-to-end pipeline, from radiometric correction and feature extraction to model training and spatial inference.

## Overview

Satellite-Derived Bathymetry (SDB) is a cost-effective alternative to traditional hydrographic surveys, utilizing remote sensing data to model the relationship between spectral reflectance and water depth. This project leverages the **XGBoost** (Extreme Gradient Boosting) regressor to capture the non-linear complexities of optically shallow waters.

This repository is designed for **geospatial researchers, practitioners, hydrographers, and enthusiasts** interested in:

* **Preprocessing:** Atmospheric and sunglint correction (Hedley method) for coastal remote sensing data.
* **Feature Engineering:** Utilizing spectral bands vs. deep learning embeddings.
* **ML Regression:** Training and validating bathymetric models.
* **Mapping:** Generating continuous bathymetric raster maps (GeoTIFF).

## Study Area & Dataset

**Location:** Gili Ketapang Island, Probolinggo, East Java, Indonesia.

**Field Data:**

* **Source:** Single Beam Echosounder (SBES) survey.
* **Acquisition:** Collected by the author in **2018** as part of a final academic project.
* **Context:** While the field data is from 2018, the geomorphological characteristics of the study area's seabed have remained stable. The methodology presented here is temporal-agnostic and can be applied to recent satellite imagery provided concurrent field data is available.

**Satellite Data Sources:**
This workflow utilizes data hosted on **Google Earth Engine (GEE)**, specifically `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance) and `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.

**Data Access:**
To facilitate reproduction, a Google Earth Engine script is provided to download the pre-configured Sentinel-2 dataset used in this study:

* **Download Script:** [https://code.earthengine.google.com/d77eb539aa7eced309607b0d7e156c0b](https://code.earthengine.google.com/d77eb539aa7eced309607b0d7e156c0b)

## Repository Structure

```text
├── data/
│   ├── corrected/       # Preprocessed Sentinel-2 images (deglinted & masked)
│   ├── embeddings/      # Google Satellite Embeddings rasters
│   └── sounding/        # Field survey data (Shapefile/GeoJSON)
│
├── model/
│   ├── embeddings/      # Saved XGBoost models trained on embeddings
│   └── plot/            # Regression scatterplots & feature importance graphs
│
├── output/              # Final inference results (Bathymetry GeoTIFFs)
│
├── preprocessing/       # Python modules for correction logic (Hedley deglint, NDWI)
│
├── train-test dataset/  # Extracted spectral/embedding values paired with depth
│   └── embeddings/      # Specific training sets for the embeddings workflow
│
├── 1. preprocessing.ipynb                      # Standard Sentinel-2 workflow
├── 1. preprocessing_satellite_embeddings.ipynb # Deep Learning Embeddings workflow
├── 2. model train-test.ipynb                   # Model training & evaluation
└── 3. model inference.ipynb                    # Map generation

```

## Usage Workflow

This project is structured as a sequential pipeline. Run the Jupyter Notebooks in the following order:

### 1. Data Preprocessing

Choose **one** of the following paths depending on your input data:

* **Option A: Standard Multispectral (`1. preprocessing.ipynb`)**
* Performs **Sunglint Correction** using the method by *Hedley et al. (2005)*.
* Applies **MNDWI** (Modified Normalized Difference Water Index) to mask land and clouds.
* Extracts spectral reflectance values at the coordinates of the sounding data.


* **Option B: Embeddings (`1. preprocessing_satellite_embeddings.ipynb`)**
* Utilizes 64-dimensional vectors from Google's pre-trained satellite model.
* Handles data extraction and preparation specifically for high-dimensional feature spaces.



### 2. Model Training (`2. model train-test.ipynb`)

* Loads the training/testing datasets generated in Step 1.
* Performs **Hyperparameter Tuning** on the XGBoost regressor.
* Evaluates performance using **RMSE** (Root Mean Squared Error), **MAE**, and ****.
* Visualizes Feature Importance to interpret the model.

### 3. Inference & Mapping (`3. model inference.ipynb`)

* Loads the trained model artifact (`.pkl`).
* Predicts depth values for every pixel in the target satellite scene.
* Exports the result as a **GeoTIFF**.
* Generates a side-by-side visualization (True Color vs. Predicted Depth).

## Installation

Ensure you have a Python environment (3.9+) set up. It is recommended to use `conda` or `venv`.

```bash
# Clone the repository
git clone https://github.com/mwahyur46/Satellite-Derived-Bathymetry-GeoAI.git

# Install required packages
pip install -r requirements.txt

```

## References

The methodology and code structure in this repository are adapted from and inspired by the following works:

1. **Geoscience Australia.** (n.d.). *Sun Glint Correction Algorithm*. GitHub Repository. Retrieved from [https://github.com/GeoscienceAustralia/sun-glint-correction/tree/develop](https://github.com/GeoscienceAustralia/sun-glint-correction/tree/develop)
2. **He, J., Zhang, S., Cui, X., & Feng, W.** (2024). Remote sensing for shallow bathymetry: A systematic review. *Earth-Science Reviews*, 258, 104957. [https://doi.org/10.1016/j.earscirev.2024.104957](https://doi.org/10.1016/j.earscirev.2024.104957)
3. **Hedley, J. D., Harborne, A. R., & Mumby, P. J.** (2005). Technical note: Simple and robust removal of sun glint for mapping shallow‐water benthos. *International Journal of Remote Sensing*, 26(10), 2107–2112. [https://doi.org/10.1080/01431160500034086](https://doi.org/10.1080/01431160500034086)
4. **Sagawa, T., Yamashita, Y., Okumura, T., & Yamanokuchi, T.** (2019). Satellite derived bathymetry using machine learning and Multi-Temporal satellite images. *Remote Sensing*, 11(10), 1155. [https://doi.org/10.3390/rs11101155](https://doi.org/10.3390/rs11101155)
5. **Wicaksono, P., Harahap, S. D., & Hendriana, R.** (2023). Satellite-derived bathymetry from WorldView-2 based on linear and machine learning regression in the optically complex shallow water of the coral reef ecosystem of Kemujan Island. *Remote Sensing Applications Society and Environment*, 33, 101085. [https://doi.org/10.1016/j.rsase.2023.101085](https://doi.org/10.1016/j.rsase.2023.101085)

---
*Created by [M. Wahyu R.](https://www.linkedin.com/in/mwahyur) | 2025*
