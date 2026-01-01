"""
Core Decomposition Methods for terravector

Standalone implementations of signal decomposition methods for DEM analysis.
Adapted from RESIDUALS project.

Methods:
- Gaussian (Classical low-pass)
- Bilateral (Edge-preserving)
- Wavelet DWT (Multi-scale)
- Morphological Opening (Shape-based)
- Top-Hat (Small features)
- Polynomial (Trend removal)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, grey_opening, grey_closing
from scipy.ndimage import white_tophat, black_tophat
from skimage.morphology import disk
import cv2
import pywt
from typing import Tuple, Dict, Callable, Any


# Registry of all decomposition methods
DECOMPOSITION_METHODS: Dict[str, Callable] = {}

# Default parameters for each method
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {}


def _register(name: str, default_params: Dict[str, Any]):
    """Simple registration decorator."""
    def decorator(func):
        DECOMPOSITION_METHODS[name] = func
        DEFAULT_PARAMS[name] = default_params
        return func
    return decorator


# =============================================================================
# Classical Signal Processing
# =============================================================================

@_register('gaussian', {'sigma': 10})
def decompose_gaussian(dem: np.ndarray, sigma: float = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian low-pass filtering for trend extraction.
    
    Simple and fast baseline method. Treats all directions equally.
    
    Args:
        dem: Input DEM array
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Tuple of (trend, residual)
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    trend = gaussian_filter(dem_filled, sigma=sigma)
    residual = dem_filled - trend
    return trend, residual


@_register('bilateral', {'d': 9, 'sigma_color': 75, 'sigma_space': 75})
def decompose_bilateral(
    dem: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilateral filtering - edge-preserving smoothing.
    
    Smooths while preserving edges. Good for roads, walls, embankments.
    
    Args:
        dem: Input DEM array
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Tuple of (trend, residual)
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    # Normalize to 0-255 range for cv2
    dem_min, dem_max = dem_filled.min(), dem_filled.max()
    dem_range = dem_max - dem_min
    if dem_range == 0:
        dem_range = 1
    
    dem_norm = ((dem_filled - dem_min) / dem_range * 255).astype(np.float32)
    trend_norm = cv2.bilateralFilter(dem_norm, d, sigma_color, sigma_space)
    trend = trend_norm / 255 * dem_range + dem_min
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Wavelet Methods
# =============================================================================

@_register('wavelet_dwt', {'wavelet': 'db4', 'level': 3})
def decompose_wavelet_dwt(
    dem: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete Wavelet Transform decomposition.
    
    Separates into approximation (trend) and detail (residual) coefficients.
    
    Args:
        dem: Input DEM array
        wavelet: Wavelet type (e.g., 'haar', 'db4', 'sym4')
        level: Decomposition level
        
    Returns:
        Tuple of (trend, residual)
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    # Perform 2D DWT
    coeffs = pywt.wavedec2(dem_filled, wavelet, level=level)
    
    # Trend = approximation coefficients only
    trend_coeffs = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in detail) 
        for detail in coeffs[1:]
    ]
    
    # Residual = detail coefficients only
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    # Reconstruct
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    # Trim to original size
    trend = trend[:dem.shape[0], :dem.shape[1]]
    residual = residual[:dem.shape[0], :dem.shape[1]]
    
    return trend, residual


# =============================================================================
# Morphological Methods
# =============================================================================

@_register('morphological', {'operation': 'opening', 'size': 10})
def decompose_morphological(
    dem: np.ndarray,
    operation: str = 'opening',
    size: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Morphological filtering for shape-based decomposition.
    
    Opening removes bright features smaller than the element.
    Closing removes dark features smaller than the element.
    
    Args:
        dem: Input DEM array
        operation: 'opening', 'closing', or 'average'
        size: Radius of disk structuring element
        
    Returns:
        Tuple of (trend, residual)
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    selem = disk(size)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    else:  # 'average'
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    return trend, residual


@_register('tophat', {'size': 20, 'mode': 'white'})
def decompose_tophat(
    dem: np.ndarray,
    size: int = 20,
    mode: str = 'white'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-hat transform for small feature extraction.
    
    White top-hat: extracts bright features smaller than element (mounds)
    Black top-hat: extracts dark features smaller than element (pits)
    
    Args:
        dem: Input DEM array
        size: Radius of disk structuring element
        mode: 'white' or 'black'
        
    Returns:
        Tuple of (trend, residual)
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    selem = disk(size)
    
    if mode == 'white':
        residual = white_tophat(dem_filled, footprint=selem)
    else:
        residual = black_tophat(dem_filled, footprint=selem)
    
    trend = dem_filled - residual
    return trend, residual


# =============================================================================
# Polynomial/Surface Fitting
# =============================================================================

@_register('polynomial', {'degree': 2})
def decompose_polynomial(dem: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polynomial surface fitting for trend removal.
    
    Fits a polynomial surface and subtracts it to reveal local features.
    
    Args:
        dem: Input DEM array
        degree: Polynomial degree (1, 2, or 3)
        
    Returns:
        Tuple of (trend, residual)
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    rows, cols = dem_filled.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = dem_filled.flatten()
    
    # Build design matrix for 2D polynomial
    if degree == 1:
        A = np.column_stack([
            np.ones_like(X_flat),
            X_flat,
            Y_flat
        ])
    elif degree == 2:
        A = np.column_stack([
            np.ones_like(X_flat),
            X_flat, Y_flat,
            X_flat**2, X_flat*Y_flat, Y_flat**2
        ])
    else:  # degree 3
        A = np.column_stack([
            np.ones_like(X_flat),
            X_flat, Y_flat,
            X_flat**2, X_flat*Y_flat, Y_flat**2,
            X_flat**3, X_flat**2*Y_flat, X_flat*Y_flat**2, Y_flat**3
        ])
    
    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
    
    trend_flat = A @ coeffs
    trend = trend_flat.reshape(dem_filled.shape)
    residual = dem_filled - trend
    
    return trend, residual


def get_all_methods() -> Dict[str, Callable]:
    """Return all registered decomposition methods."""
    return DECOMPOSITION_METHODS.copy()


def get_default_params(method_name: str) -> Dict[str, Any]:
    """Get default parameters for a method."""
    return DEFAULT_PARAMS.get(method_name, {}).copy()

