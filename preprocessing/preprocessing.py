import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

def sunglint_correction(visible_bands, nir_band, output_dir=None, image_id=None, plot_graphs=False):
    """
    Applies the Hedley et al. (2005) sunglint correction method.
    
    Reference:
    https://github.com/GeoscienceAustralia/sun-glint-correction/tree/develop
    
    Citation:
    Hedley, J. D., Harborne, A. R., & Mumby, P. J. (2005). 
    Simple and robust removal of sun glint for mapping shallow-water benthos. 
    International Journal of Remote Sensing, 26(10), 2107-2112.
    
    Args:
        visible_bands (list): List of 2D numpy arrays [Blue, Green, Red].
        nir_band (numpy.ndarray): 2D numpy array for NIR.
        output_dir (str): Path to save QC plots.
        image_id (str): Identifier for the plot title.
        plot_graphs (bool): Enable/Disable plotting.
        
    Returns:
        list: [Corrected_Blue, Corrected_Green, Corrected_Red]
    """
    nir = np.array(nir_band)
    
    # --- 1. Automatic Deep Water Selection ---
    valid_mask = nir > 0
    if not np.any(valid_mask):
        return np.array(visible_bands)

    # Use lowest 20% of NIR as deep water proxy
    threshold_val = np.percentile(nir[valid_mask], 20)
    sample_mask = (nir > 0) & (nir <= threshold_val)
    
    if np.sum(sample_mask) < 1000:
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
        
        if slope < 0: slope = 0
        
        # Apply Hedley Formula
        glint_component = slope * (nir - min_nir)
        corrected = vis_band - glint_component
        corrected = np.clip(corrected, 0, None) # Remove negative values
        corrected_bands.append(corrected)

        # Plot result
        if plot_graphs and output_dir:
            ax = axes[i]
            subset = np.random.choice(len(nir_samples), size=min(2000, len(nir_samples)), replace=False)
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
    Applies the "Simple Glint Removal" (NIR Subtraction) method.
    This mimics the GEE logic: Corrected = Visible - NIR.
    
    Ref: Lines 63-64 of user provided GEE script.
    
    Citation (Methodological Basis):
    Hedley, J. D., Harborne, A. R., & Mumby, P. J. (2005). 
    Simple and robust removal of sun glint for mapping shallow-water benthos. 
    International Journal of Remote Sensing, 26(10), 2107-2112.
    
    Args:
        visible_bands (list): List of 2D numpy arrays [Blue, Green, Red].
        nir_band (numpy.ndarray): 2D numpy array for NIR (Band 8).
        
    Returns:
        list: [Corrected_Blue, Corrected_Green, Corrected_Red]
    """
    nir = np.array(nir_band)
    corrected_bands = []
    
    for band in visible_bands:
        # Practical Approach: Simple Subtraction
        # "Subtract the NIR signal from the Visible signal"
        corrected = band - nir
        
        # Physics Check: Light cannot be negative. 
        # Clip values < 0 to 0 (or a small epsilon like 0.0001 if needed)
        corrected = np.clip(corrected, 0, None)
        
        corrected_bands.append(corrected)
        
    return corrected_bands

def calculate_ndwi(green_band, nir_band):
    """
    Calculates the Normalized Difference Water Index (NDWI).
    Formula: (Green - NIR) / (Green + NIR)
    
    Args:
        green_band (numpy.ndarray): Green band (preferably glint-corrected).
        nir_band (numpy.ndarray): NIR band (original).
        
    Returns:
        numpy.ndarray: NDWI image (values between -1 and 1).
    """
    # Allow division by zero (results in NaN, handled later)
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = green_band - nir_band
        denominator = green_band + nir_band
        ndwi = numerator / denominator
        
    # Fill NaNs (where Green+NIR = 0) with -1 (Non-water)
    ndwi = np.nan_to_num(ndwi, nan=-1.0)
    
    return ndwi

def calculate_mndwi(green_band, swir_band):
    """
    Calculates the Modified Normalized Difference Water Index (MNDWI).
    Formula: (Green - SWIR) / (Green + SWIR)
    
    Why use this?
    SWIR is absorbed more strongly by water than NIR, making this index 
    more robust against glint and turbidity (fewer holes in the mask).
    """
    
    # Allow division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
            numerator = green_band - swir_band
            denominator = green_band + swir_band
            mndwi = numerator / denominator
    
    # Fill NaNs with -1 (Non-water)        
    mndwi = np.nan_to_num(mndwi, nan=-1.0)
    return mndwi
