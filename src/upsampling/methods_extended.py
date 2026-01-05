"""
Extended Upsampling Methods

Additional interpolation methods for comprehensive coverage.
Ported from DIVERGE/RESIDUALS project.
"""

import numpy as np
from scipy.ndimage import zoom, convolve1d, laplace
from scipy.interpolate import RectBivariateSpline
import cv2

from .registry import register_upsampling


# =============================================================================
# Different Spline Orders
# =============================================================================

@register_upsampling(
    name='nearest',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='exact original values at sample points',
    introduces='blocky artifacts, no smoothing'
)
def upsample_nearest(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Nearest-neighbor interpolation (order=0)."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    return zoom(dem_filled, scale, order=0)


@register_upsampling(
    name='bilinear',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='linear gradients',
    introduces='slight blurring, no overshoot'
)
def upsample_bilinear(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Bilinear interpolation (order=1)."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    return zoom(dem_filled, scale, order=1)


@register_upsampling(
    name='quadratic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth curvature',
    introduces='some overshoot at discontinuities'
)
def upsample_quadratic(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Quadratic spline interpolation (order=2)."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    return zoom(dem_filled, scale, order=2)


@register_upsampling(
    name='quartic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='higher-order smoothness',
    introduces='more ringing than cubic'
)
def upsample_quartic(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Quartic spline interpolation (order=4)."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    return zoom(dem_filled, scale, order=4)


@register_upsampling(
    name='quintic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='highest available smoothness',
    introduces='most ringing of polynomial methods'
)
def upsample_quintic(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Quintic spline interpolation (order=5)."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    return zoom(dem_filled, scale, order=5)


# =============================================================================
# Windowed Sinc Methods
# =============================================================================

@register_upsampling(
    name='sinc_hamming',
    category='frequency',
    default_params={'scale': 2, 'kernel_size': 8},
    param_ranges={'scale': [2, 4, 8], 'kernel_size': [4, 8, 16]},
    preserves='band-limited signal with Hamming window',
    introduces='less ringing than pure sinc'
)
def upsample_sinc_hamming(
    dem: np.ndarray,
    scale: int = 2,
    kernel_size: int = 8
) -> np.ndarray:
    """Sinc interpolation with Hamming window."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    def hamming_sinc_kernel(size, scale):
        t = np.arange(-size, size + 1) / scale
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.where(t == 0, 1.0, np.sin(np.pi * t) / (np.pi * t))
        window = 0.54 - 0.46 * np.cos(np.pi * (np.arange(len(t)) / (len(t) - 1) * 2))
        return sinc * window
    
    kernel_1d = hamming_sinc_kernel(kernel_size, scale)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    upsampled = np.zeros((new_h, new_w))
    upsampled[::scale, ::scale] = dem_filled
    
    result = convolve1d(upsampled, kernel_1d * scale, axis=0, mode='reflect')
    result = convolve1d(result, kernel_1d * scale, axis=1, mode='reflect')
    
    return result


@register_upsampling(
    name='sinc_blackman',
    category='frequency',
    default_params={'scale': 2, 'kernel_size': 8},
    param_ranges={'scale': [2, 4, 8], 'kernel_size': [4, 8, 16]},
    preserves='band-limited signal with Blackman window',
    introduces='minimal ringing, slight blurring'
)
def upsample_sinc_blackman(
    dem: np.ndarray,
    scale: int = 2,
    kernel_size: int = 8
) -> np.ndarray:
    """Sinc interpolation with Blackman window."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    def blackman_sinc_kernel(size, scale):
        t = np.arange(-size, size + 1) / scale
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.where(t == 0, 1.0, np.sin(np.pi * t) / (np.pi * t))
        n = len(t)
        window = (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1)) + 
                  0.08 * np.cos(4 * np.pi * np.arange(n) / (n - 1)))
        return sinc * window
    
    kernel_1d = blackman_sinc_kernel(kernel_size, scale)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    upsampled = np.zeros((new_h, new_w))
    upsampled[::scale, ::scale] = dem_filled
    
    result = convolve1d(upsampled, kernel_1d * scale, axis=0, mode='reflect')
    result = convolve1d(result, kernel_1d * scale, axis=1, mode='reflect')
    
    return result


# =============================================================================
# Cubic Variations
# =============================================================================

@register_upsampling(
    name='cubic_catmull_rom',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth interpolation through control points',
    introduces='minimal overshoot'
)
def upsample_cubic_catmull_rom(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Catmull-Rom spline interpolation."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    y_out = np.linspace(0, h - 1, new_h)
    x_out = np.linspace(0, w - 1, new_w)
    
    y_in = np.arange(h)
    x_in = np.arange(w)
    
    spline = RectBivariateSpline(y_in, x_in, dem_filled, kx=3, ky=3, s=0)
    return spline(y_out, x_out)


@register_upsampling(
    name='cubic_mitchell',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='balanced sharpness and smoothness',
    introduces='minimal artifacts'
)
def upsample_cubic_mitchell(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Mitchell-Netravali filter (B=C=1/3)."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    B, C = 1/3, 1/3
    
    def mitchell_kernel(t):
        t = np.abs(t)
        t2, t3 = t * t, t * t * t
        return np.where(
            t < 1,
            ((12 - 9*B - 6*C) * t3 + (-18 + 12*B + 6*C) * t2 + (6 - 2*B)) / 6,
            np.where(
                t < 2,
                ((-B - 6*C) * t3 + (6*B + 30*C) * t2 + (-12*B - 48*C) * t + (8*B + 24*C)) / 6,
                0
            )
        )
    
    kernel_size = 4 * scale
    kernel_1d = mitchell_kernel(np.arange(-kernel_size, kernel_size + 1) / scale)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    upsampled = np.zeros((new_h, new_w))
    upsampled[::scale, ::scale] = dem_filled
    
    result = convolve1d(upsampled, kernel_1d * scale, axis=0, mode='reflect')
    result = convolve1d(result, kernel_1d * scale, axis=1, mode='reflect')
    
    return result


# =============================================================================
# Adaptive Methods
# =============================================================================

@register_upsampling(
    name='edge_directed',
    category='adaptive',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4]},
    preserves='edges with minimal jagging',
    introduces='possible artifacts in textured regions'
)
def upsample_edge_directed(dem: np.ndarray, scale: int = 2) -> np.ndarray:
    """Edge-directed interpolation blending bicubic and lanczos."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    bicubic = zoom(dem_filled, scale, order=3)
    
    gy = np.gradient(dem_filled, axis=0)
    gx = np.gradient(dem_filled, axis=1)
    
    gy_up = zoom(gy, scale, order=1)
    gx_up = zoom(gx, scale, order=1)
    
    mag = np.sqrt(gx_up**2 + gy_up**2)
    
    lanczos = cv2.resize(
        dem_filled.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )
    
    mag_norm = mag / (mag.max() + 1e-10)
    
    return bicubic * (1 - mag_norm) + lanczos * mag_norm


@register_upsampling(
    name='regularized',
    category='optimization',
    default_params={'scale': 2, 'lambda_reg': 0.01},
    param_ranges={'scale': [2, 4], 'lambda_reg': [0.001, 0.01, 0.1]},
    preserves='smooth interpolation with controlled energy',
    introduces='slight smoothing from regularization'
)
def upsample_regularized(
    dem: np.ndarray,
    scale: int = 2,
    lambda_reg: float = 0.01
) -> np.ndarray:
    """Regularized interpolation using smoothness prior."""
    dem_filled = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    
    result = zoom(dem_filled, scale, order=3)
    
    h, w = dem_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    known = np.zeros((new_h, new_w), dtype=bool)
    known[::scale, ::scale] = True
    
    original_vals = np.zeros((new_h, new_w))
    original_vals[::scale, ::scale] = dem_filled
    
    for _ in range(10):
        lap = laplace(result)
        result = result - lambda_reg * lap
        result[known] = original_vals[known]
    
    return result

