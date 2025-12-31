"""
Satellite-Derived Bathymetry (SDB) Preprocessing Utilities.

This module provides functions for radiometric correction (sunglint removal),
water masking indices (NDWI/MNDWI), and spatial data extraction required for
processing Sentinel-2 imagery.

References:
    Hedley, J. D., Harborne, A. R., & Mumby, P. J. (2005). 
    Technical note: Simple and robust removal of sun glint for mapping shallow‐water benthos. 
    International Journal of Remote Sensing, 26(10), 2107–2112.
    https://doi.org/10.1080/01431160500034086

    Sunglint correction code adapted from github repository:
    https://github.com/GeoscienceAustralia/sun-glint-correction/tree/develop
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import linregress

def sunglint_correction(visible_bands, nir_band, output_dir=None, image_id=None, plot_graphs=False):
    """
    Applies the Hedley et al. (2005) sunglint correction method.
    
    Uses linear regression between NIR and visible bands over deep water 
    to remove the glint component.

    Args:
        visible_bands (list): List of 2D numpy arrays [Blue, Green, Red].
        nir_band (numpy.ndarray): 2D numpy array for NIR.
        output_dir (str, optional): Path to save QC plots.
        image_id (str, optional): Identifier for the plot title.
        plot_graphs (bool): Enable/Disable plotting.
        
    Returns:
        list: [Corrected_Blue, Corrected_Green, Corrected_Red]
    """
    nir = np.array(nir_band)
    
    # --- 1. Automatic Deep Water Selection ---
    valid_mask = nir > 0
    if not np.any(valid_mask):
        return np.array(visible_bands)

    # Use lowest 20% of NIR pixels as a proxy for deep water
    threshold_val = np.percentile(nir[valid_mask], 20)
    sample_mask = (nir > 0) & (nir <= threshold_val)
    
    # Fallback if insufficient deep water pixels found
    if np.sum(sample_mask) < 1000:
        if image_id:
            print(f"  [Warning] {image_id}: Insufficient deep water pixels. Simple subtraction used.")
        return [np.clip(b - nir, 0, None) for b in visible_bands]

    nir_samples = nir[sample_mask]
    min_nir = np.min(nir_samples)
    
    corrected_bands = []
    
    # Setup Plotting
    if plot_graphs and output_dir:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        band_names = ['Blue', 'Green', 'Red']

    # --- 2. Regression & Correction ---
    for i, vis_band in enumerate(visible_bands):
        vis_samples = vis_band[sample_mask]
        
        # Calculate Slope: Visible = Slope * NIR + Intercept
        slope, intercept, r_val, _, _ = linregress(nir_samples, vis_samples)
        
        # Enforce physical constraint
        if slope < 0: 
            slope = 0
        
        # Apply Hedley Formula
        glint_component = slope * (nir - min_nir)
        corrected = vis_band - glint_component
        corrected = np.clip(corrected, 0, None)  # Remove negative values
        corrected_bands.append(corrected)

        # Plot result
        if plot_graphs and output_dir:
            ax = axes[i]
            # Subsample points for performance
            subset_size = min(2000, len(nir_samples))
            subset = np.random.choice(len(nir_samples), size=subset_size, replace=False)
            
            ax.scatter(nir_samples[subset], vis_samples[subset], alpha=0.1, s=1, c='gray')
            
            x_vals = np.array([np.min(nir_samples), np.max(nir_samples)])
            ax.plot(x_vals, slope * x_vals + intercept, color='red', label=f'Slope: {slope:.2f}')
            ax.set_title(f"{band_names[i]} (R={r_val:.2f})")
            ax.legend()

    if plot_graphs and output_dir:
        plt.suptitle(f"Glint Correction: {image_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"plot_{image_id}.png"))
        plt.close()

    return corrected_bands

def simple_sunglint_correction(visible_bands, nir_band):
    """
    Applies simple glint subtraction (Visible - NIR).
    
    Args:
        visible_bands (list): List of 2D numpy arrays [Blue, Green, Red].
        nir_band (numpy.ndarray): 2D numpy array for NIR.
        
    Returns:
        list: [Corrected_Blue, Corrected_Green, Corrected_Red]
    """
    nir = np.array(nir_band)
    corrected_bands = []
    
    for band in visible_bands:
        # Simple Subtraction
        corrected = band - nir
        corrected = np.clip(corrected, 0, None)
        corrected_bands.append(corrected)
        
    return corrected_bands

def calculate_ndwi(green_band, nir_band):
    """
    Calculates the Normalized Difference Water Index (NDWI).
    Formula: (Green - NIR) / (Green + NIR)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = green_band - nir_band
        denominator = green_band + nir_band
        ndwi = numerator / denominator
        
    ndwi = np.nan_to_num(ndwi, nan=-1.0)
    return ndwi

def calculate_mndwi(green_band, swir_band):
    """
    Calculates the Modified Normalized Difference Water Index (MNDWI).
    Formula: (Green - SWIR) / (Green + SWIR)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = green_band - swir_band
        denominator = green_band + swir_band
        mndwi = numerator / denominator
    
    mndwi = np.nan_to_num(mndwi, nan=-1.0)
    return mndwi

def extract_raster_value(sample_points, raster_path, depth_column):
    """
    Extracts raster pixel values at specific point coordinates.
    
    Args:
        sample_points (GeoDataFrame): Points containing geometry and depth labels.
        raster_path (str): Path to the raster file.
        depth_column (str): Name of the column containing depth values.
        
    Returns:
        tuple: (pixel_values_array, depth_values_array)
    """
    print(f"Processing: {os.path.basename(raster_path)}")

    with rasterio.open(raster_path) as src:
        # 1. Check and match CRS
        if sample_points.crs != src.crs:
            print(f"  - Reprojecting points from {sample_points.crs} to {src.crs}...")
            sample_points = sample_points.to_crs(src.crs)
        
        # 2. Prepare coordinates for sampling
        coord_list = [(x, y) for x, y in zip(sample_points.geometry.x, sample_points.geometry.y)]

        # 3. Sample the raster
        print(f"  - Extracting pixel values for {len(coord_list)} points...")
        sampled_values = list(src.sample(coord_list))

        # Convert to Numpy Array
        pixel_values = np.array(sampled_values) 

        # 4. Get depth labels
        depth_values = sample_points[depth_column].values 

    return pixel_values, depth_values