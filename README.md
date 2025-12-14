# Satellite-Derived-Bathymetry-GeoAI
Optical shallow water depth estimation using satellite imagery with GeoAI models

# Folder Structure
project_folder/
│
├── input_images/          # Your PlanetScope .tif files
├── output_images/         # Where results go
├── preprocessing/         # NEW FOLDER
│   ├── __init__.py        # Empty file, makes this a python package
│   └── corrections.py     # The script below
└── main_processing.py     # Your main batch script


# Preprocessing
Download dataset in Google Earth Engine: https://code.earthengine.google.com/d77eb539aa7eced309607b0d7e156c0b