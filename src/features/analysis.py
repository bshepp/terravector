"""
Rich Feature Analysis Module

Comprehensive feature analysis for terrain patches.
Ported from DIVERGE/RESIDUALS project.

Analyzes:
- Linear features (roads, walls) via edge detection
- Spatial autocorrelation (Moran's I)
- Frequency content (low/mid/high bands)
- Feature distinctness (SNR)

Amplitude-scaled statistics are signed-log1p compressed in
``analysis_to_vector`` (see ``LOG_SCALED_KEYS``); ``analyze_features`` itself
returns raw values.
"""

import numpy as np
from typing import Dict, Any, List
from scipy import stats
from scipy.ndimage import correlate, uniform_filter

try:
    from skimage.feature import canny
    from skimage.transform import hough_line, hough_line_peaks
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def analyze_features(
    arr: np.ndarray,
    include_linear: bool = True,
    include_frequency: bool = True
) -> Dict[str, float]:
    """
    Comprehensive feature analysis of a patch or residual.
    
    Args:
        arr: 2D array to analyze
        include_linear: Include linear feature detection (slower)
        include_frequency: Include frequency analysis
        
    Returns:
        Dict of analysis metrics
    """
    analysis = {}
    
    arr_clean = np.nan_to_num(arr, nan=0)
    flat = arr_clean.flatten()
    
    # ==========================================================================
    # Basic Statistics
    # ==========================================================================
    analysis['mean'] = float(np.mean(flat))
    analysis['std'] = float(np.std(flat))
    analysis['min'] = float(np.min(flat))
    analysis['max'] = float(np.max(flat))
    analysis['range'] = analysis['max'] - analysis['min']
    analysis['median'] = float(np.median(flat))
    
    # Distribution shape
    if len(flat) > 10:
        analysis['skewness'] = float(stats.skew(flat))
        analysis['kurtosis'] = float(stats.kurtosis(flat))
    else:
        analysis['skewness'] = 0.0
        analysis['kurtosis'] = 0.0
    
    # Energy
    analysis['energy'] = float(np.sum(flat ** 2) / len(flat))
    
    # Entropy from histogram
    hist, _ = np.histogram(flat, bins=50, density=True)
    hist = hist + 1e-10
    analysis['entropy'] = float(stats.entropy(hist))
    
    # ==========================================================================
    # Spatial Autocorrelation (Moran's I)
    # ==========================================================================
    analysis['spatial_autocorr'] = compute_spatial_autocorrelation(arr_clean)
    
    # ==========================================================================
    # Linear Feature Detection
    # ==========================================================================
    if include_linear and SKIMAGE_AVAILABLE:
        try:
            linear_metrics = detect_linear_features(arr_clean)
            analysis.update(linear_metrics)
        except Exception:
            analysis['linear_feature_count'] = 0
            analysis['max_linear_strength'] = 0.0
    else:
        analysis['linear_feature_count'] = 0
        analysis['max_linear_strength'] = 0.0
    
    # ==========================================================================
    # Frequency Content
    # ==========================================================================
    if include_frequency:
        try:
            freq_metrics = analyze_frequency_content(arr_clean)
            analysis.update(freq_metrics)
        except Exception:
            analysis['low_freq_energy'] = 0.0
            analysis['mid_freq_energy'] = 0.0
            analysis['high_freq_energy'] = 0.0
    
    # ==========================================================================
    # Feature Distinctness (SNR)
    # ==========================================================================
    analysis['feature_snr'] = compute_feature_snr(arr_clean)
    
    return analysis


def compute_spatial_autocorrelation(arr: np.ndarray) -> float:
    """
    Compute Moran's I spatial autocorrelation.
    
    High values indicate spatially structured features.
    """
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]) / 4
    
    lagged = correlate(arr, kernel, mode='reflect')
    
    arr_centered = arr - arr.mean()
    lagged_centered = lagged - lagged.mean()
    
    numerator = np.sum(arr_centered * lagged_centered)
    denominator = np.sum(arr_centered ** 2)
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def detect_linear_features(
    arr: np.ndarray,
    canny_sigma: float = 2.0,
    num_peaks: int = 20
) -> Dict[str, float]:
    """
    Detect linear features using Hough transform.
    """
    if not SKIMAGE_AVAILABLE:
        return {'linear_feature_count': 0, 'max_linear_strength': 0.0}
    
    # Normalize
    arr_norm = arr - arr.min()
    arr_max = arr_norm.max()
    if arr_max > 0:
        arr_norm = arr_norm / arr_max
    
    # Edge detection
    edges = canny(arr_norm, sigma=canny_sigma)
    
    # Hough transform
    h, theta, d = hough_line(edges)
    
    # Find peaks
    accum, angles, dists = hough_line_peaks(h, theta, d, num_peaks=num_peaks)
    
    return {
        'linear_feature_count': len(accum),
        'max_linear_strength': float(np.max(accum)) if len(accum) > 0 else 0.0,
        'mean_linear_strength': float(np.mean(accum)) if len(accum) > 0 else 0.0,
    }


def analyze_frequency_content(arr: np.ndarray) -> Dict[str, float]:
    """
    Analyze frequency distribution using FFT.
    """
    fft = np.fft.fft2(arr)
    fft_mag = np.abs(np.fft.fftshift(fft))
    
    center = np.array(fft_mag.shape) // 2
    y, x = np.ogrid[:fft_mag.shape[0], :fft_mag.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    max_r = min(center)
    
    low_mask = r < max_r * 0.2
    mid_mask = (r >= max_r * 0.2) & (r < max_r * 0.6)
    high_mask = r >= max_r * 0.6
    
    low_energy = np.mean(fft_mag[low_mask]) if np.any(low_mask) else 0.0
    mid_energy = np.mean(fft_mag[mid_mask]) if np.any(mid_mask) else 0.0
    high_energy = np.mean(fft_mag[high_mask]) if np.any(high_mask) else 0.0
    
    return {
        'low_freq_energy': float(low_energy),
        'mid_freq_energy': float(mid_energy),
        'high_freq_energy': float(high_energy),
        'freq_ratio_high_low': float(high_energy / (low_energy + 1e-10)),
    }


def compute_feature_snr(arr: np.ndarray, percentile: float = 95) -> float:
    """
    Compute log-scaled signal-to-noise ratio for features.

    Returns log1p(std(signal) / std(noise)), where "signal" is the top
    ``percentile`` of |arr| and "noise" is the rest. The raw ratio is
    unbounded: on near-flat residuals the noise std approaches zero and the
    ratio reaches 1e15, which dominates z-score normalization and effectively
    zeroes the dimension for every other patch. log1p keeps it monotonic,
    bounded (~36 at 1e15), and 0 for the degenerate cases below.
    """
    threshold = np.percentile(np.abs(arr), percentile)
    
    signal_mask = np.abs(arr) >= threshold
    signal_values = arr[signal_mask]
    noise_values = arr[~signal_mask]
    
    if len(signal_values) == 0 or len(noise_values) == 0:
        return 0.0
    
    noise_std = np.std(noise_values)
    if noise_std == 0:
        return 0.0
    
    return float(np.log1p(np.std(signal_values) / noise_std))


# Keys whose raw values are amplitude-scaled residual statistics with
# intrinsically heavy tails (a residual field can span 1e-6 .. 1e3 across
# terrain). Stored raw, z-score normalization lets a handful of extreme
# patches set the per-dimension scale and squashes everyone else near zero
# (energy: 98.8% of county patches at |z| < 0.05 before this change).
# signed_log1p compresses the tails while keeping sign and monotonicity.
# Changing this set is an index-compat event; see CLAUDE.md.
LOG_SCALED_KEYS = (
    'mean', 'std', 'range', 'median', 'skewness', 'kurtosis', 'energy',
    'low_freq_energy', 'mid_freq_energy', 'high_freq_energy',
    'freq_ratio_high_low', 'min', 'max',
)


def signed_log1p(x):
    """sign(x) * log1p(|x|): odd, monotonic, identity near 0, log for |x| >> 1."""
    return np.sign(x) * np.log1p(np.abs(x))


def analysis_to_vector(analysis: Dict[str, float]) -> np.ndarray:
    """
    Convert analysis dict to fixed-length feature vector.

    Returns 20-dimensional vector with standardized order. Keys listed in
    ``LOG_SCALED_KEYS`` are passed through ``signed_log1p`` so that their
    heavy tails do not dominate z-score normalization at index build time.
    """
    keys = [
        'mean', 'std', 'range', 'median', 'skewness', 'kurtosis',
        'energy', 'entropy', 'spatial_autocorr',
        'linear_feature_count', 'max_linear_strength', 'mean_linear_strength',
        'low_freq_energy', 'mid_freq_energy', 'high_freq_energy', 'freq_ratio_high_low',
        'feature_snr', 'min', 'max'
    ]

    # Pad to 20 dimensions
    while len(keys) < 20:
        keys.append(f'reserved_{len(keys)}')

    vector = []
    for key in keys[:20]:
        value = float(analysis.get(key, 0.0))
        if key in LOG_SCALED_KEYS:
            value = float(signed_log1p(value))
        vector.append(value)

    return np.array(vector, dtype=np.float32)


def get_log_scaled_dims() -> List[int]:
    """Indices (0..19) of the analysis-vector slots that are signed-log1p scaled."""
    keys = [
        'mean', 'std', 'range', 'median', 'skewness', 'kurtosis',
        'energy', 'entropy', 'spatial_autocorr',
        'linear_feature_count', 'max_linear_strength', 'mean_linear_strength',
        'low_freq_energy', 'mid_freq_energy', 'high_freq_energy', 'freq_ratio_high_low',
        'feature_snr', 'min', 'max', 'reserved_19'
    ]
    return [i for i, k in enumerate(keys) if k in LOG_SCALED_KEYS]


def get_analysis_dim() -> int:
    """Get dimension of analysis vector."""
    return 20


def get_analysis_labels() -> List[str]:
    """Get labels for analysis vector dimensions."""
    return [
        'slog_mean', 'slog_std', 'slog_range', 'slog_median', 'slog_skewness', 'slog_kurtosis',
        'slog_energy', 'entropy', 'spatial_autocorr',
        'linear_count', 'linear_max', 'linear_mean',
        'slog_freq_low', 'slog_freq_mid', 'slog_freq_high', 'slog_freq_ratio',
        'log_snr', 'slog_min', 'slog_max', 'reserved'
    ]

