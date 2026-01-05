"""
Extended Decomposition Methods

Additional methods for comprehensive coverage.
Ported from DIVERGE/RESIDUALS project.
"""

import numpy as np
from scipy.ndimage import (
    gaussian_filter, median_filter, uniform_filter,
    grey_opening, grey_closing, grey_erosion, grey_dilation,
    laplace
)
from skimage.morphology import disk, square, rectangle, diamond, ellipse
from skimage.filters import difference_of_gaussians
import cv2

from .registry import register_decomposition


# =============================================================================
# Additional Classical Methods
# =============================================================================

@register_decomposition(
    name='gaussian_anisotropic',
    category='classical',
    default_params={'sigma_x': 10, 'sigma_y': 10},
    param_ranges={'sigma_x': [2, 5, 10, 20, 50], 'sigma_y': [2, 5, 10, 20, 50]},
    preserves='directional features aligned with low-sigma axis',
    destroys='features perpendicular to low-sigma axis'
)
def decompose_gaussian_anisotropic(
    dem: np.ndarray,
    sigma_x: float = 10,
    sigma_y: float = 10
) -> tuple:
    """Anisotropic Gaussian filtering with different X/Y smoothing."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    trend = gaussian_filter(dem_filled, sigma=(sigma_y, sigma_x))
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='median',
    category='edge_preserving',
    default_params={'size': 5},
    param_ranges={'size': [3, 5, 7, 11, 15, 21]},
    preserves='sharp edges, step discontinuities',
    destroys='salt-and-pepper noise, thin lines'
)
def decompose_median(dem: np.ndarray, size: int = 5) -> tuple:
    """Median filter decomposition - edge-preserving, removes impulse noise."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    trend = median_filter(dem_filled, size=size)
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='uniform',
    category='classical',
    default_params={'size': 10},
    param_ranges={'size': [3, 5, 10, 20, 50, 100]},
    preserves='average local elevation',
    destroys='all local variation equally'
)
def decompose_uniform(dem: np.ndarray, size: int = 10) -> tuple:
    """Uniform (box) filter decomposition - simple averaging."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    trend = uniform_filter(dem_filled, size=size)
    residual = dem_filled - trend
    return trend, residual


# =============================================================================
# Difference of Gaussians (DoG) - Band-pass filtering
# =============================================================================

@register_decomposition(
    name='dog',
    category='multiscale',
    default_params={'sigma_low': 2, 'sigma_high': 10},
    param_ranges={'sigma_low': [1, 2, 3, 5], 'sigma_high': [5, 10, 20, 50, 100]},
    preserves='features at intermediate scales',
    destroys='very small and very large features'
)
def decompose_dog(
    dem: np.ndarray,
    sigma_low: float = 2,
    sigma_high: float = 10
) -> tuple:
    """Difference of Gaussians band-pass filtering."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    if sigma_high <= sigma_low:
        sigma_high = sigma_low * 2
    
    residual = difference_of_gaussians(dem_filled, sigma_low, sigma_high)
    trend = dem_filled - residual
    return trend, residual


@register_decomposition(
    name='log',
    category='multiscale',
    default_params={'sigma': 5},
    param_ranges={'sigma': [1, 2, 3, 5, 10, 20]},
    preserves='blob-like features at specified scale',
    destroys='linear features, edges, flat regions'
)
def decompose_log(dem: np.ndarray, sigma: float = 5) -> tuple:
    """Laplacian of Gaussian (LoG) - detects blob-like features."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    smoothed = gaussian_filter(dem_filled, sigma=sigma)
    residual = laplace(smoothed) * (sigma ** 2)
    
    trend = dem_filled - residual
    return trend, residual


# =============================================================================
# Additional Morphological Methods
# =============================================================================

@register_decomposition(
    name='morphological_square',
    category='morphological',
    default_params={'operation': 'opening', 'size': 10},
    param_ranges={'operation': ['opening', 'closing', 'average'], 'size': [3, 5, 10, 15, 20, 50]},
    preserves='rectangular features aligned with axes',
    destroys='features smaller than element, circular features'
)
def decompose_morphological_square(
    dem: np.ndarray,
    operation: str = 'opening',
    size: int = 10
) -> tuple:
    """Morphological filtering with square structuring element."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    selem = square(size)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    else:
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_diamond',
    category='morphological',
    default_params={'operation': 'opening', 'radius': 10},
    param_ranges={'operation': ['opening', 'closing', 'average'], 'radius': [3, 5, 10, 15, 20]},
    preserves='diamond/rhombus shaped features',
    destroys='features not matching diamond geometry'
)
def decompose_morphological_diamond(
    dem: np.ndarray,
    operation: str = 'opening',
    radius: int = 10
) -> tuple:
    """Morphological filtering with diamond structuring element."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    selem = diamond(radius)
    
    if operation == 'opening':
        trend = grey_opening(dem_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(dem_filled, footprint=selem)
    else:
        opened = grey_opening(dem_filled, footprint=selem)
        closed = grey_closing(dem_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = dem_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_gradient',
    category='morphological',
    default_params={'size': 5, 'shape': 'disk'},
    param_ranges={'size': [3, 5, 7, 10, 15], 'shape': ['disk', 'square', 'diamond']},
    preserves='edges, boundaries, rapid transitions',
    destroys='flat regions, gradual slopes'
)
def decompose_morphological_gradient(
    dem: np.ndarray,
    size: int = 5,
    shape: str = 'disk'
) -> tuple:
    """Morphological gradient = dilation - erosion. Highlights edges."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    if shape == 'disk':
        selem = disk(size)
    elif shape == 'square':
        selem = square(size)
    else:
        selem = diamond(size)
    
    dilated = grey_dilation(dem_filled, footprint=selem)
    eroded = grey_erosion(dem_filled, footprint=selem)
    
    residual = dilated - eroded
    trend = dem_filled - residual
    
    return trend, residual


@register_decomposition(
    name='tophat_combined',
    category='morphological',
    default_params={'size': 20},
    param_ranges={'size': [5, 10, 20, 50, 100]},
    preserves='both bright and dark small features',
    destroys='large-scale variation'
)
def decompose_tophat_combined(dem: np.ndarray, size: int = 20) -> tuple:
    """Combined white + black top-hat transform."""
    from scipy.ndimage import white_tophat, black_tophat
    
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    selem = disk(size)
    
    white_th = white_tophat(dem_filled, footprint=selem)
    black_th = black_tophat(dem_filled, footprint=selem)
    
    residual = white_th - black_th
    trend = dem_filled - residual
    
    return trend, residual


# =============================================================================
# Edge-Preserving Methods
# =============================================================================

@register_decomposition(
    name='anisotropic_diffusion',
    category='edge_preserving',
    default_params={'iterations': 10, 'kappa': 50, 'gamma': 0.1},
    param_ranges={'iterations': [5, 10, 20, 50], 'kappa': [10, 30, 50, 100], 'gamma': [0.05, 0.1, 0.15, 0.2]},
    preserves='edges above gradient threshold (kappa)',
    destroys='noise, texture below threshold'
)
def decompose_anisotropic_diffusion(
    dem: np.ndarray,
    iterations: int = 10,
    kappa: float = 50,
    gamma: float = 0.1
) -> tuple:
    """Perona-Malik anisotropic diffusion - iterative edge-preserving smoothing."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    img = dem_filled.copy().astype(np.float64)
    
    for _ in range(iterations):
        nabla_n = np.roll(img, -1, axis=0) - img
        nabla_s = np.roll(img, 1, axis=0) - img
        nabla_e = np.roll(img, -1, axis=1) - img
        nabla_w = np.roll(img, 1, axis=1) - img
        
        c_n = np.exp(-(nabla_n / kappa) ** 2)
        c_s = np.exp(-(nabla_s / kappa) ** 2)
        c_e = np.exp(-(nabla_e / kappa) ** 2)
        c_w = np.exp(-(nabla_w / kappa) ** 2)
        
        img += gamma * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)
    
    trend = img
    residual = dem_filled - trend
    
    return trend, residual


@register_decomposition(
    name='guided',
    category='edge_preserving',
    default_params={'radius': 8, 'eps': 0.01},
    param_ranges={'radius': [4, 8, 16, 32], 'eps': [0.001, 0.01, 0.1, 1.0]},
    preserves='edges defined by the guide image',
    destroys='texture not aligned with edges'
)
def decompose_guided(
    dem: np.ndarray,
    radius: int = 8,
    eps: float = 0.01
) -> tuple:
    """Guided filter decomposition - edge-aware smoothing."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    dem_min, dem_max = dem_filled.min(), dem_filled.max()
    dem_range = dem_max - dem_min
    if dem_range == 0:
        dem_range = 1
    dem_norm = (dem_filled - dem_min) / dem_range
    
    I = dem_norm
    p = dem_norm
    
    mean_I = uniform_filter(I, size=2*radius+1)
    mean_p = uniform_filter(p, size=2*radius+1)
    corr_Ip = uniform_filter(I * p, size=2*radius+1)
    corr_II = uniform_filter(I * I, size=2*radius+1)
    
    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = uniform_filter(a, size=2*radius+1)
    mean_b = uniform_filter(b, size=2*radius+1)
    
    q = mean_a * I + mean_b
    
    trend = q * dem_range + dem_min
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Rolling Ball
# =============================================================================

@register_decomposition(
    name='rolling_ball',
    category='morphological',
    default_params={'radius': 50},
    param_ranges={'radius': [10, 25, 50, 100, 200]},
    preserves='features smaller than ball radius',
    destroys='background curvature, large-scale variation'
)
def decompose_rolling_ball(dem: np.ndarray, radius: int = 50) -> tuple:
    """Rolling ball background subtraction."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    center = radius
    
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    ball = np.where(
        dist_sq <= radius ** 2,
        np.sqrt(radius ** 2 - dist_sq),
        0
    )
    ball = ball.max() - ball
    
    eroded = grey_erosion(dem_filled, footprint=ball > 0, structure=ball)
    trend = grey_dilation(eroded, footprint=ball > 0, structure=ball)
    
    residual = dem_filled - trend
    
    return trend, residual


# =============================================================================
# Additional Wavelets
# =============================================================================

@register_decomposition(
    name='wavelet_biorthogonal',
    category='wavelet',
    default_params={'wavelet': 'bior3.5', 'level': 3},
    param_ranges={'wavelet': ['bior1.3', 'bior2.4', 'bior3.5', 'bior4.4'], 'level': [1, 2, 3, 4, 5]},
    preserves='multi-scale structure with linear phase',
    destroys='high-frequency detail'
)
def decompose_wavelet_biorthogonal(
    dem: np.ndarray,
    wavelet: str = 'bior3.5',
    level: int = 3
) -> tuple:
    """Biorthogonal wavelet decomposition - symmetric filters, linear phase."""
    import pywt
    
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    coeffs = pywt.wavedec2(dem_filled, wavelet, level=level)
    
    trend_coeffs = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in detail) 
        for detail in coeffs[1:]
    ]
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    trend = trend[:dem.shape[0], :dem.shape[1]]
    residual = residual[:dem.shape[0], :dem.shape[1]]
    
    return trend, residual

