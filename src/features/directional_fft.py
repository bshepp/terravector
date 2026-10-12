"""
Directional FFT signature extraction for terrain patches.

Computes 2D FFT and extracts frequency statistics along angular slices
through the frequency domain. This captures directional texture and
linear feature orientations in terrain.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import ndimage


# Registry of available statistics for each angular slice
DIRECTIONAL_FFT_STATS = [
    'energy',           # Total energy in this direction
    'low_freq_ratio',   # Ratio of low-frequency energy
    'high_freq_ratio',  # Ratio of high-frequency energy  
    'peak_freq',        # Normalized peak frequency
    'spectral_centroid', # Center of mass of spectrum
    'spectral_spread',  # Spread around centroid
]


def compute_directional_fft_embedding(
    patch: np.ndarray,
    angles: Optional[List[float]] = None,
    stats: Optional[List[str]] = None
) -> np.ndarray:
    """
    Compute directional FFT embedding for a terrain patch.
    
    Args:
        patch: 2D array of elevation values
        angles: List of angles in degrees (default: 0, 45, 90, 135)
        stats: Which statistics to compute per angle (default: all)
    
    Returns:
        1D array of shape (n_angles * n_stats,)
    """
    if angles is None:
        angles = [0, 45, 90, 135]
    
    if stats is None:
        stats = DIRECTIONAL_FFT_STATS
    
    # Handle edge cases
    if patch is None or patch.size == 0:
        return np.zeros(len(angles) * len(stats), dtype=np.float32)
    
    patch = np.asarray(patch, dtype=np.float64)
    
    # Handle NaN values
    if np.any(np.isnan(patch)):
        patch = np.nan_to_num(patch, nan=np.nanmean(patch))
    
    # Apply window to reduce spectral leakage
    window = _create_2d_window(patch.shape)
    windowed = patch * window
    
    # Compute 2D FFT and shift zero-frequency to center
    fft2d = np.fft.fft2(windowed)
    fft2d_shifted = np.fft.fftshift(fft2d)
    magnitude = np.abs(fft2d_shifted)
    
    # Extract statistics for each angle
    embedding_parts = []
    
    for angle in angles:
        slice_stats = _extract_angular_slice_stats(
            magnitude, angle, stats
        )
        embedding_parts.extend(slice_stats)
    
    return np.array(embedding_parts, dtype=np.float32)


def _create_2d_window(shape: Tuple[int, int]) -> np.ndarray:
    """Create a 2D Hann window to reduce spectral leakage."""
    h, w = shape
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    return np.outer(win_h, win_w)


def _extract_angular_slice_stats(
    magnitude: np.ndarray,
    angle_deg: float,
    stats: List[str]
) -> List[float]:
    """
    Extract frequency statistics along a line through the FFT at given angle.
    
    The angle is measured from the horizontal axis (0° = horizontal features,
    90° = vertical features in the spatial domain).
    """
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Maximum radius to sample
    max_radius = min(center_y, center_x)
    
    # Sample points along the line at this angle
    angle_rad = np.deg2rad(angle_deg)
    radii = np.arange(1, max_radius)  # Skip DC component
    
    # Sample both directions from center (angle and angle + 180°)
    # This gives us the full slice through the FFT
    xs_pos = center_x + (radii * np.cos(angle_rad)).astype(int)
    ys_pos = center_y + (radii * np.sin(angle_rad)).astype(int)
    xs_neg = center_x - (radii * np.cos(angle_rad)).astype(int)
    ys_neg = center_y - (radii * np.sin(angle_rad)).astype(int)
    
    # Combine both halves
    xs = np.concatenate([xs_neg[::-1], xs_pos])
    ys = np.concatenate([ys_neg[::-1], ys_pos])
    
    # Clamp to valid indices
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    
    # Extract slice values
    slice_values = magnitude[ys, xs]
    
    # Compute requested statistics
    result = []
    for stat in stats:
        value = _compute_slice_stat(slice_values, radii, stat)
        result.append(value)
    
    return result


def _compute_slice_stat(
    slice_values: np.ndarray,
    radii: np.ndarray,
    stat: str
) -> float:
    """
    Compute a single statistic from an angular slice.

    ``slice_values`` is ordered [negative radii reversed, positive radii], so
    its index is NOT monotone in frequency: position 0 is the highest
    negative frequency and position n is the lowest positive one. Every
    statistic below therefore works on ``freqs``, the |frequency| of each
    sample, rather than on array position. (Before 2026-10-11 the band ratios
    sliced by position, which made low_freq_ratio measure the highest band,
    peak_freq range over 0..2, and spectral_centroid a constant.)
    """

    total_energy = np.sum(slice_values ** 2)

    if not np.isfinite(total_energy) or total_energy == 0:
        return 0.0

    n = len(radii)
    freqs = np.concatenate([radii[::-1], radii]).astype(np.float64)
    weights = slice_values ** 2

    if stat == 'energy':
        # Log-scaled total energy
        return np.log1p(total_energy)

    elif stat == 'low_freq_ratio':
        # Ratio of energy in low frequencies (large-scale features)
        low_energy = np.sum(weights[freqs < n / 3])
        return low_energy / total_energy

    elif stat == 'high_freq_ratio':
        # Ratio of energy in high frequencies (fine detail)
        high_energy = np.sum(weights[freqs >= 2 * n / 3])
        return high_energy / total_energy

    elif stat == 'peak_freq':
        # Normalized |frequency| of the peak magnitude, in [0, 1]
        peak_idx = np.argmax(slice_values)
        return freqs[peak_idx] / n

    elif stat == 'spectral_centroid':
        # Energy-weighted mean |frequency|, in [0, 1]
        return np.average(freqs, weights=weights) / n

    elif stat == 'spectral_spread':
        # Energy-weighted standard deviation of |frequency| around the centroid
        centroid = np.average(freqs, weights=weights)
        variance = np.average((freqs - centroid) ** 2, weights=weights)
        return np.sqrt(variance) / n
    
    else:
        raise ValueError(f"Unknown statistic: {stat}")


def get_directional_fft_dim(
    angles: Optional[List[float]] = None,
    stats: Optional[List[str]] = None
) -> int:
    """Get the dimensionality of the directional FFT embedding."""
    if angles is None:
        angles = [0, 45, 90, 135]
    if stats is None:
        stats = DIRECTIONAL_FFT_STATS
    return len(angles) * len(stats)


def get_directional_fft_labels(
    angles: Optional[List[float]] = None,
    stats: Optional[List[str]] = None
) -> List[str]:
    """Get labels for each dimension of the embedding."""
    if angles is None:
        angles = [0, 45, 90, 135]
    if stats is None:
        stats = DIRECTIONAL_FFT_STATS
    
    labels = []
    for angle in angles:
        for stat in stats:
            labels.append(f"fft_{int(angle)}deg_{stat}")
    return labels

