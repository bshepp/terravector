"""
Core Upsampling Methods

Implements 4 key upsampling methods:
- Bicubic (Standard baseline)
- Lanczos (Sharp edge preservation)
- B-Spline (Smooth interpolation)
- FFT Zero-padding (Frequency-preserving)

Ported from DIVERGE/RESIDUALS project.
"""

import numpy as np
from scipy.ndimage import zoom
import cv2

from .registry import register_upsampling


# =============================================================================
# Classical Interpolation
# =============================================================================

@register_upsampling(
    name='bicubic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth curves, good for continuous surfaces',
    introduces='slight ringing at sharp edges'
)
def upsample_bicubic(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Bicubic interpolation using scipy.ndimage.zoom.
    
    Standard baseline method. Order=3 for cubic interpolation.
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    return zoom(dem_filled, scale, order=3)


@register_upsampling(
    name='lanczos',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='sharp edges with smooth interpolation',
    introduces='controlled ringing (less than sinc)'
)
def upsample_lanczos(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Lanczos interpolation using OpenCV.
    
    Better edge preservation than bicubic, commonly used in image processing.
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    return cv2.resize(
        dem_filled.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )


# =============================================================================
# Spline Methods
# =============================================================================

@register_upsampling(
    name='bspline',
    category='spline',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth surface, less ringing than cubic',
    introduces='slightly more smoothing than cubic'
)
def upsample_bspline(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Quadratic (order=2) B-spline interpolation.
    
    Uses scipy.ndimage.zoom with order=2 for a genuinely different
    interpolation kernel than bicubic (order=3).
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    return zoom(dem_filled, scale, order=2)


# =============================================================================
# Frequency Domain
# =============================================================================

@register_upsampling(
    name='fft_zeropad',
    category='frequency',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='frequency content exactly (band-limited)',
    introduces='Gibbs ringing at discontinuities'
)
def upsample_fft_zeropad(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    FFT upsampling via zero-padding in frequency domain.
    
    Theoretically optimal for band-limited signals.
    Preserves all frequency content from original.
    """
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Compute FFT
    fft = np.fft.fft2(dem_filled)
    
    # Create zero-padded output array
    padded = np.zeros((new_h, new_w), dtype=complex)
    
    # Split dimensions for proper quadrant placement
    h_half = (h + 1) // 2
    w_half = (w + 1) // 2
    
    # Place frequency components in corners
    padded[:h_half, :w_half] = fft[:h_half, :w_half]
    
    if w > 1:
        padded[:h_half, -(w - w_half):] = fft[:h_half, w_half:]
    
    if h > 1:
        padded[-(h - h_half):, :w_half] = fft[h_half:, :w_half]
    
    if h > 1 and w > 1:
        padded[-(h - h_half):, -(w - w_half):] = fft[h_half:, w_half:]
    
    # Inverse FFT and scale
    result = np.real(np.fft.ifft2(padded)) * (scale ** 2)
    
    return result

