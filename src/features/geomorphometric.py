"""
Geomorphometric Feature Extraction

Classic terrain derivatives for landform classification.
These compute local terrain properties from elevation gradients.

Features:
- Slope: Rate of elevation change
- Aspect: Direction of steepest descent  
- Curvature: Surface bending (profile, plan, mean)
- TPI: Topographic Position Index (elevation relative to neighborhood)
- TRI: Terrain Ruggedness Index
- Roughness: Local elevation variance
"""

import numpy as np
from scipy.ndimage import generic_filter, uniform_filter
from typing import Tuple, Dict, Callable, Any, List


# Registry of geomorphometric features
GEOMORPHOMETRIC_FEATURES: Dict[str, Callable] = {}

# Default parameters
DEFAULT_GEOMORPHOMETRIC_PARAMS: Dict[str, Dict[str, Any]] = {}


def _register(name: str, default_params: Dict[str, Any]):
    """Registration decorator for geomorphometric features."""
    def decorator(func):
        GEOMORPHOMETRIC_FEATURES[name] = func
        DEFAULT_GEOMORPHOMETRIC_PARAMS[name] = default_params
        return func
    return decorator


def _safe_array(dem: np.ndarray) -> np.ndarray:
    """Fill NaN values with mean for computation."""
    dem_filled = dem.copy()
    if np.any(np.isnan(dem_filled)):
        dem_filled = np.nan_to_num(dem_filled, nan=float(np.nanmean(dem)))
    return dem_filled.astype(np.float64)


# =============================================================================
# Gradient-Based Features
# =============================================================================

@_register('slope', {'cell_size': 1.0})
def compute_slope(dem: np.ndarray, cell_size: float = 1.0) -> Tuple[np.ndarray, List[float]]:
    """
    Compute slope (gradient magnitude) in degrees.
    
    Args:
        dem: Input DEM array
        cell_size: Cell size in map units (for proper scaling)
        
    Returns:
        Tuple of (slope_array, statistics)
    """
    dem_filled = _safe_array(dem)
    
    # Compute gradients using numpy (Sobel-like)
    dy, dx = np.gradient(dem_filled, cell_size)
    
    # Slope magnitude in degrees
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    stats = _compute_stats(slope_deg)
    return slope_deg, stats


@_register('aspect', {'cell_size': 1.0})
def compute_aspect(dem: np.ndarray, cell_size: float = 1.0) -> Tuple[np.ndarray, List[float]]:
    """
    Compute aspect (slope direction) in degrees from north.
    
    Args:
        dem: Input DEM array
        cell_size: Cell size in map units
        
    Returns:
        Tuple of (aspect_array, statistics)
    """
    dem_filled = _safe_array(dem)
    
    dy, dx = np.gradient(dem_filled, cell_size)
    
    # Aspect in degrees (0=North, 90=East, etc.)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = np.mod(aspect + 360, 360)  # Normalize to 0-360
    
    # For flat areas, set aspect to -1 (undefined)
    flat_mask = (np.abs(dx) < 1e-8) & (np.abs(dy) < 1e-8)
    aspect[flat_mask] = -1
    
    # Stats on valid aspects only
    valid_aspect = aspect[~flat_mask] if np.any(~flat_mask) else aspect
    stats = _compute_stats(valid_aspect)
    return aspect, stats


@_register('curvature', {'cell_size': 1.0})
def compute_curvature(dem: np.ndarray, cell_size: float = 1.0) -> Tuple[np.ndarray, List[float]]:
    """
    Compute mean curvature (average of profile and plan curvature).
    
    Positive = convex (ridges), Negative = concave (valleys)
    
    Args:
        dem: Input DEM array
        cell_size: Cell size in map units
        
    Returns:
        Tuple of (curvature_array, statistics)
    """
    dem_filled = _safe_array(dem)
    
    # Second derivatives
    dy, dx = np.gradient(dem_filled, cell_size)
    dyy, dyx = np.gradient(dy, cell_size)
    dxy, dxx = np.gradient(dx, cell_size)
    
    # Mean curvature (Laplacian)
    curvature = -(dxx + dyy)
    
    stats = _compute_stats(curvature)
    return curvature, stats


# =============================================================================
# Neighborhood-Based Features
# =============================================================================

@_register('tpi', {'radius': 10})
def compute_tpi(dem: np.ndarray, radius: int = 10) -> Tuple[np.ndarray, List[float]]:
    """
    Topographic Position Index: elevation difference from local mean.
    
    Positive = ridges/hilltops, Negative = valleys
    
    Args:
        dem: Input DEM array
        radius: Neighborhood radius in pixels
        
    Returns:
        Tuple of (tpi_array, statistics)
    """
    dem_filled = _safe_array(dem)
    
    # Local mean using uniform filter
    size = 2 * radius + 1
    local_mean = uniform_filter(dem_filled, size=size, mode='reflect')
    
    tpi = dem_filled - local_mean
    
    stats = _compute_stats(tpi)
    return tpi, stats


@_register('tri', {'window_size': 3})
def compute_tri(dem: np.ndarray, window_size: int = 3) -> Tuple[np.ndarray, List[float]]:
    """
    Terrain Ruggedness Index: mean absolute difference from neighbors.
    
    Higher values = more rugged terrain
    
    Args:
        dem: Input DEM array
        window_size: Size of neighborhood window (odd number)
        
    Returns:
        Tuple of (tri_array, statistics)
    """
    dem_filled = _safe_array(dem)
    
    def tri_func(window):
        center = window[len(window) // 2]
        return np.mean(np.abs(window - center))
    
    tri = generic_filter(dem_filled, tri_func, size=window_size, mode='reflect')
    
    stats = _compute_stats(tri)
    return tri, stats


@_register('roughness', {'window_size': 3})
def compute_roughness(dem: np.ndarray, window_size: int = 3) -> Tuple[np.ndarray, List[float]]:
    """
    Surface roughness: local standard deviation of elevation.
    
    Args:
        dem: Input DEM array
        window_size: Size of neighborhood window
        
    Returns:
        Tuple of (roughness_array, statistics)
    """
    dem_filled = _safe_array(dem)
    
    # Local mean and mean of squares
    local_mean = uniform_filter(dem_filled, size=window_size, mode='reflect')
    local_mean_sq = uniform_filter(dem_filled**2, size=window_size, mode='reflect')
    
    # Variance = E[X^2] - E[X]^2
    variance = local_mean_sq - local_mean**2
    variance = np.maximum(variance, 0)  # Numerical stability
    
    roughness = np.sqrt(variance)
    
    stats = _compute_stats(roughness)
    return roughness, stats


# =============================================================================
# Statistics
# =============================================================================

def _compute_stats(arr: np.ndarray) -> List[float]:
    """
    Compute standard statistics for a feature array.
    
    Returns: [mean, std, energy, entropy, range, median]
    """
    from scipy.stats import entropy as scipy_entropy
    
    flat = arr.flatten()
    
    if len(flat) == 0 or np.all(np.isnan(flat)):
        return [0.0] * 6
    
    flat = np.nan_to_num(flat, nan=0.0)
    
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    energy = float(np.sum(flat**2) / len(flat))
    
    # Entropy from histogram
    hist, _ = np.histogram(flat, bins=50, density=True)
    hist = hist + 1e-10
    ent = float(scipy_entropy(hist))
    
    range_val = float(np.max(flat) - np.min(flat))
    median = float(np.median(flat))
    
    return [mean, std, energy, ent, range_val, median]


def compute_geomorphometric_embedding(
    patch: np.ndarray,
    features: List[str] = None,
    params: Dict[str, Dict[str, Any]] = None
) -> np.ndarray:
    """
    Compute geomorphometric feature embedding for a patch.
    
    Args:
        patch: 2D elevation array
        features: List of feature names (None = all)
        params: Parameters for each feature
        
    Returns:
        1D feature vector
    """
    if features is None:
        features = list(GEOMORPHOMETRIC_FEATURES.keys())
    
    if params is None:
        params = {}
    
    embedding_parts = []
    
    for feat_name in features:
        if feat_name not in GEOMORPHOMETRIC_FEATURES:
            raise ValueError(f"Unknown geomorphometric feature: {feat_name}")
        
        func = GEOMORPHOMETRIC_FEATURES[feat_name]
        feat_params = params.get(feat_name, DEFAULT_GEOMORPHOMETRIC_PARAMS.get(feat_name, {}))
        
        try:
            _, stats = func(patch, **feat_params)
        except Exception:
            stats = [0.0] * 6
        
        embedding_parts.extend(stats)
    
    return np.array(embedding_parts, dtype=np.float32)


def get_geomorphometric_dim(features: List[str] = None) -> int:
    """Get dimension of geomorphometric embedding."""
    if features is None:
        features = list(GEOMORPHOMETRIC_FEATURES.keys())
    return len(features) * 6


def get_geomorphometric_labels(features: List[str] = None) -> List[str]:
    """Get labels for each dimension."""
    if features is None:
        features = list(GEOMORPHOMETRIC_FEATURES.keys())
    
    stats = ['mean', 'std', 'energy', 'entropy', 'range', 'median']
    labels = []
    for feat in features:
        for stat in stats:
            labels.append(f"{feat}_{stat}")
    return labels

