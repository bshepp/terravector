"""
Texture Feature Extraction

Image texture analysis methods applied to terrain patches.

Features:
- GLCM: Gray-Level Co-occurrence Matrix properties
- LBP: Local Binary Patterns histogram
"""

import numpy as np
from typing import Tuple, Dict, Callable, Any, List
from skimage.feature import graycomatrix, graycoprops
try:
    from skimage.feature import local_binary_pattern
except ImportError:
    local_binary_pattern = None


# Registry of texture features
TEXTURE_FEATURES: Dict[str, Callable] = {}

# Default parameters
DEFAULT_TEXTURE_PARAMS: Dict[str, Dict[str, Any]] = {}


def _register(name: str, default_params: Dict[str, Any]):
    """Registration decorator for texture features."""
    def decorator(func):
        TEXTURE_FEATURES[name] = func
        DEFAULT_TEXTURE_PARAMS[name] = default_params
        return func
    return decorator


def _normalize_to_uint8(dem: np.ndarray, levels: int = 256) -> np.ndarray:
    """Normalize DEM to uint8 range for texture analysis."""
    dem_filled = dem.copy()
    if np.any(np.isnan(dem_filled)):
        dem_filled = np.nan_to_num(dem_filled, nan=float(np.nanmean(dem)))
    
    dem_min, dem_max = dem_filled.min(), dem_filled.max()
    if dem_max - dem_min < 1e-10:
        return np.zeros(dem_filled.shape, dtype=np.uint8)
    
    normalized = (dem_filled - dem_min) / (dem_max - dem_min)
    quantized = (normalized * (levels - 1)).astype(np.uint8)
    return quantized


# =============================================================================
# GLCM Features
# =============================================================================

@_register('glcm', {'distances': [1, 2, 4], 'angles': [0, 45, 90, 135], 'levels': 64})
def compute_glcm_features(
    dem: np.ndarray,
    distances: List[int] = None,
    angles: List[float] = None,
    levels: int = 64
) -> Tuple[Dict[str, float], List[float]]:
    """
    Compute Gray-Level Co-occurrence Matrix texture features.
    
    GLCM captures spatial relationships between pixel intensities.
    
    Properties computed:
    - contrast: Local intensity variation
    - dissimilarity: Average absolute difference
    - homogeneity: Closeness to diagonal
    - energy: Sum of squared elements (uniformity)
    - correlation: Linear dependency of gray levels
    - ASM: Angular Second Moment (orderliness)
    
    Args:
        dem: Input DEM array
        distances: Pixel distances for co-occurrence
        angles: Angles in degrees (converted to radians)
        levels: Number of gray levels
        
    Returns:
        Tuple of (properties_dict, statistics_list)
    """
    if distances is None:
        distances = [1, 2, 4]
    if angles is None:
        angles = [0, 45, 90, 135]
    
    # Convert angles to radians
    angles_rad = [a * np.pi / 180 for a in angles]
    
    # Quantize DEM
    quantized = _normalize_to_uint8(dem, levels=levels)
    
    # Compute GLCM
    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles_rad,
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    # Compute properties (average over all distances and angles)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    results = {}
    stats = []
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        mean_val = float(np.mean(values))
        results[prop] = mean_val
        stats.append(mean_val)
    
    return results, stats


# =============================================================================
# LBP Features
# =============================================================================

@_register('lbp', {'radius': 3, 'n_points': 24, 'method': 'uniform'})
def compute_lbp_features(
    dem: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
    method: str = 'uniform'
) -> Tuple[np.ndarray, List[float]]:
    """
    Compute Local Binary Pattern features.
    
    LBP encodes local texture by comparing each pixel to its neighbors,
    creating a binary pattern that captures micro-textures.
    
    Args:
        dem: Input DEM array
        radius: Radius of circular neighborhood
        n_points: Number of points in circular neighborhood
        method: LBP method ('uniform', 'default', 'ror', 'nri_uniform')
        
    Returns:
        Tuple of (lbp_image, histogram_statistics)
    """
    if local_binary_pattern is None:
        # Fallback if skimage version doesn't have LBP
        return np.zeros_like(dem), [0.0] * 10
    
    # Normalize to 0-255 for LBP
    normalized = _normalize_to_uint8(dem, levels=256).astype(np.float64)
    
    # Compute LBP
    lbp = local_binary_pattern(normalized, n_points, radius, method=method)
    
    # Compute histogram
    if method == 'uniform':
        n_bins = n_points + 2  # uniform patterns + 1 for non-uniform
    else:
        n_bins = 2 ** n_points  # Can be very large
        n_bins = min(n_bins, 256)  # Cap at 256 bins
    
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Compute statistics from histogram
    stats = _lbp_histogram_stats(hist)
    
    return lbp, stats


def _lbp_histogram_stats(hist: np.ndarray) -> List[float]:
    """
    Compute statistics from LBP histogram.
    
    Returns 10 values:
    - First 6 histogram bins (or padded if fewer)
    - Mean, std, entropy, uniformity
    """
    from scipy.stats import entropy as scipy_entropy
    
    stats = []
    
    # First 6 bins (capture dominant patterns)
    for i in range(6):
        if i < len(hist):
            stats.append(float(hist[i]))
        else:
            stats.append(0.0)
    
    # Overall statistics
    stats.append(float(np.mean(hist)))  # Mean bin value
    stats.append(float(np.std(hist)))   # Spread
    
    # Entropy (texture complexity)
    hist_safe = hist + 1e-10
    stats.append(float(scipy_entropy(hist_safe)))
    
    # Uniformity (energy of histogram)
    stats.append(float(np.sum(hist ** 2)))
    
    return stats


# =============================================================================
# Combined Interface
# =============================================================================

def compute_texture_embedding(
    patch: np.ndarray,
    features: List[str] = None,
    params: Dict[str, Dict[str, Any]] = None
) -> np.ndarray:
    """
    Compute texture feature embedding for a patch.
    
    Args:
        patch: 2D elevation array
        features: List of feature names (None = all)
        params: Parameters for each feature
        
    Returns:
        1D feature vector
    """
    if features is None:
        features = list(TEXTURE_FEATURES.keys())
    
    if params is None:
        params = {}
    
    embedding_parts = []
    
    for feat_name in features:
        if feat_name not in TEXTURE_FEATURES:
            raise ValueError(f"Unknown texture feature: {feat_name}")
        
        func = TEXTURE_FEATURES[feat_name]
        feat_params = params.get(feat_name, DEFAULT_TEXTURE_PARAMS.get(feat_name, {}))
        
        try:
            _, stats = func(patch, **feat_params)
        except Exception:
            # GLCM returns 6 stats, LBP returns 10
            n_stats = 6 if feat_name == 'glcm' else 10
            stats = [0.0] * n_stats
        
        embedding_parts.extend(stats)
    
    return np.array(embedding_parts, dtype=np.float32)


def get_texture_dim(features: List[str] = None) -> int:
    """Get dimension of texture embedding."""
    if features is None:
        features = list(TEXTURE_FEATURES.keys())
    
    dim = 0
    for feat in features:
        if feat == 'glcm':
            dim += 6  # 6 GLCM properties
        elif feat == 'lbp':
            dim += 10  # 10 LBP histogram stats
    return dim


def get_texture_labels(features: List[str] = None) -> List[str]:
    """Get labels for each dimension."""
    if features is None:
        features = list(TEXTURE_FEATURES.keys())
    
    labels = []
    for feat in features:
        if feat == 'glcm':
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                labels.append(f"glcm_{prop}")
        elif feat == 'lbp':
            for i in range(6):
                labels.append(f"lbp_bin{i}")
            labels.extend(['lbp_mean', 'lbp_std', 'lbp_entropy', 'lbp_uniformity'])
    return labels

