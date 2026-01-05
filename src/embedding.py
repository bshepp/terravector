"""
Embedding Extraction Module

Converts terrain patches into feature vectors using configurable signatures.

Supports five signature types (combinable via config):
- Decomposition: Signal processing residual statistics (original method)
- Geomorphometric: Classic terrain derivatives (slope, curvature, etc.)
- Texture: Image texture features (GLCM, LBP)
- Residuals: DIVERGE-style decomposition × upsampling combinations
- Directional FFT: Frequency analysis at multiple angles for oriented features
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING
from scipy.stats import entropy as scipy_entropy

from .decomposition import DECOMPOSITION_METHODS, DEFAULT_PARAMS
from .tiling import Patch

if TYPE_CHECKING:
    from .config import SignatureConfig


def compute_residual_stats(residual: np.ndarray) -> List[float]:
    """
    Compute statistical features from a residual array.
    
    Args:
        residual: 2D residual array from decomposition
        
    Returns:
        List of statistical features
    """
    flat = residual.flatten()
    
    # Handle edge cases
    if len(flat) == 0 or np.all(np.isnan(flat)):
        return [0.0] * 6
    
    # Replace NaN with 0 for statistics
    flat = np.nan_to_num(flat, nan=0.0)
    
    # Basic statistics
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    
    # Energy (sum of squares)
    energy = float(np.sum(flat ** 2))
    
    # Normalized energy (per pixel)
    energy_norm = energy / len(flat) if len(flat) > 0 else 0.0
    
    # Entropy (discretize to histogram first)
    hist, _ = np.histogram(flat, bins=50, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    ent = float(scipy_entropy(hist))
    
    # Range
    range_val = float(np.max(flat) - np.min(flat))
    
    return [mean, std, energy_norm, ent, range_val, float(np.median(flat))]


def compute_embedding(
    patch: np.ndarray,
    methods: Optional[List[str]] = None,
    method_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Compute embedding vector for a single patch.
    
    The embedding consists of statistical summaries of residuals from
    each decomposition method, concatenated into a single vector.
    
    Args:
        patch: 2D numpy array (terrain patch)
        methods: List of method names to use (None = all)
        method_params: Parameters for each method (None = defaults)
        
    Returns:
        1D numpy array (embedding vector)
    """
    if methods is None:
        methods = list(DECOMPOSITION_METHODS.keys())
    
    if method_params is None:
        method_params = {}
    
    embedding_parts = []
    
    for method_name in methods:
        if method_name not in DECOMPOSITION_METHODS:
            raise ValueError(f"Unknown method: {method_name}")
        
        method_func = DECOMPOSITION_METHODS[method_name]
        params = method_params.get(method_name, DEFAULT_PARAMS.get(method_name, {}))
        
        try:
            trend, residual = method_func(patch, **params)
            stats = compute_residual_stats(residual)
        except Exception as e:
            # If decomposition fails, use zeros
            stats = [0.0] * 6
        
        embedding_parts.extend(stats)
    
    return np.array(embedding_parts, dtype=np.float32)


def compute_embeddings_batch(
    patches: List[Patch],
    methods: Optional[List[str]] = None,
    method_params: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute embeddings for a batch of patches.
    
    Args:
        patches: List of Patch objects
        methods: List of method names to use
        method_params: Parameters for each method
        verbose: Print progress
        
    Returns:
        2D numpy array of shape (n_patches, embedding_dim)
    """
    if not patches:
        return np.array([])
    
    embeddings = []
    n_patches = len(patches)
    
    for i, patch in enumerate(patches):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Computing embeddings: {i + 1}/{n_patches}")
        
        emb = compute_embedding(patch.data, methods, method_params)
        embeddings.append(emb)
    
    if verbose:
        print(f"  Computed {n_patches} embeddings")
    
    return np.vstack(embeddings)


def get_embedding_dim(methods: Optional[List[str]] = None) -> int:
    """
    Get the dimension of embedding vectors.
    
    Args:
        methods: List of method names (None = all)
        
    Returns:
        Embedding dimension
    """
    if methods is None:
        methods = list(DECOMPOSITION_METHODS.keys())
    
    # 6 statistics per method
    return len(methods) * 6


def get_embedding_labels(methods: Optional[List[str]] = None) -> List[str]:
    """
    Get human-readable labels for each embedding dimension.
    
    Args:
        methods: List of method names (None = all)
        
    Returns:
        List of labels for each dimension
    """
    if methods is None:
        methods = list(DECOMPOSITION_METHODS.keys())
    
    stats = ['mean', 'std', 'energy', 'entropy', 'range', 'median']
    labels = []
    
    for method in methods:
        for stat in stats:
            labels.append(f"{method}_{stat}")
    
    return labels


def normalize_embeddings(
    embeddings: np.ndarray,
    method: str = 'zscore'
) -> tuple:
    """
    Normalize embeddings for better similarity computation.
    
    Args:
        embeddings: 2D array of embeddings
        method: 'zscore' (zero mean, unit variance) or 'minmax' (0-1 range)
        
    Returns:
        Tuple of (normalized_embeddings, normalization_params)
    """
    if method == 'zscore':
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (embeddings - mean) / std
        params = {'method': 'zscore', 'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(embeddings, axis=0)
        max_val = np.max(embeddings, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized = (embeddings - min_val) / range_val
        params = {'method': 'minmax', 'min': min_val, 'range': range_val}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def apply_normalization(
    embeddings: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply pre-computed normalization to new embeddings.
    
    Args:
        embeddings: Embeddings to normalize
        params: Normalization parameters from normalize_embeddings
        
    Returns:
        Normalized embeddings
    """
    method = params['method']
    
    if method == 'zscore':
        return (embeddings - params['mean']) / params['std']
    elif method == 'minmax':
        return (embeddings - params['min']) / params['range']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# RESIDUALS-Style Embedding (decomposition × upsampling)
# =============================================================================

def compute_residuals_embedding(
    patch: np.ndarray,
    decomposition_methods: List[str],
    upsampling_methods: List[str],
    use_rich_analysis: bool = True,
    decomp_params: Optional[Dict[str, Dict[str, Any]]] = None,
    upsamp_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Compute RESIDUALS-style embedding from decomposition × upsampling combinations.
    
    For each decomposition method:
      1. Decompose patch → get residual
      2. For each upsampling method:
         a. Downsample residual by 2x
         b. Upsample back using method
         c. Compute analysis on upsampled residual
    
    Args:
        patch: 2D numpy array
        decomposition_methods: List of decomposition method names
        upsampling_methods: List of upsampling method names
        use_rich_analysis: Use 20-dim analysis or 6-dim basic stats
        decomp_params: Parameters for decomposition methods
        upsamp_params: Parameters for upsampling methods
        
    Returns:
        1D feature vector
    """
    from .decomposition import run_decomposition
    from .upsampling import run_upsampling
    from .features.analysis import analyze_features, analysis_to_vector
    from scipy.ndimage import zoom
    
    if decomp_params is None:
        decomp_params = {}
    if upsamp_params is None:
        upsamp_params = {}
    
    embedding_parts = []
    
    for decomp_name in decomposition_methods:
        try:
            # Get decomposition residual
            trend, residual = run_decomposition(
                decomp_name, patch, 
                params=decomp_params.get(decomp_name)
            )
        except Exception:
            # If decomposition fails, use zeros
            n_upsamp = len(upsampling_methods)
            dim_per = 20 if use_rich_analysis else 6
            embedding_parts.append(np.zeros(n_upsamp * dim_per, dtype=np.float32))
            continue
        
        for upsamp_name in upsampling_methods:
            try:
                # Downsample residual by 2x
                residual_down = zoom(residual, 0.5, order=1)
                
                # Upsample back
                residual_up = run_upsampling(
                    upsamp_name, residual_down, scale=2,
                    params=upsamp_params.get(upsamp_name)
                )
                
                # Ensure same size as original
                min_h = min(residual.shape[0], residual_up.shape[0])
                min_w = min(residual.shape[1], residual_up.shape[1])
                residual_up = residual_up[:min_h, :min_w]
                
                # Compute analysis
                if use_rich_analysis:
                    analysis = analyze_features(residual_up)
                    stats = analysis_to_vector(analysis)
                else:
                    stats = np.array(compute_residual_stats(residual_up), dtype=np.float32)
                
                embedding_parts.append(stats)
                
            except Exception:
                dim = 20 if use_rich_analysis else 6
                embedding_parts.append(np.zeros(dim, dtype=np.float32))
    
    return np.concatenate(embedding_parts)


# =============================================================================
# Config-Based Embedding (New API)
# =============================================================================

def compute_embedding_from_config(
    patch: np.ndarray,
    config: 'SignatureConfig'
) -> np.ndarray:
    """
    Compute embedding vector using configuration-specified signature types.
    
    Concatenates features from all enabled signature types in order:
    1. Decomposition (if enabled)
    2. Geomorphometric (if enabled)
    3. Texture (if enabled)
    4. Residuals - decomp × upsamp combinations (if enabled)
    
    Args:
        patch: 2D numpy array (terrain patch)
        config: SignatureConfig specifying which features to compute
        
    Returns:
        1D numpy array (combined embedding vector)
    """
    embedding_parts = []
    
    # 1. Decomposition features
    if config.decomposition.enabled:
        decomp_emb = compute_embedding(
            patch,
            methods=config.decomposition.methods,
            method_params=config.decomposition.params
        )
        embedding_parts.append(decomp_emb)
    
    # 2. Geomorphometric features
    if config.geomorphometric.enabled:
        from .features.geomorphometric import compute_geomorphometric_embedding
        geomorph_emb = compute_geomorphometric_embedding(
            patch,
            features=config.geomorphometric.features,
            params=config.geomorphometric.params
        )
        embedding_parts.append(geomorph_emb)
    
    # 3. Texture features
    if config.texture.enabled:
        from .features.texture import compute_texture_embedding
        texture_emb = compute_texture_embedding(
            patch,
            features=config.texture.features,
            params=config.texture.params
        )
        embedding_parts.append(texture_emb)
    
    # 4. Residuals-style decomp × upsamp combinations
    if config.residuals.enabled:
        residuals_emb = compute_residuals_embedding(
            patch,
            decomposition_methods=config.residuals.decomposition_methods,
            upsampling_methods=config.residuals.upsampling_methods,
            use_rich_analysis=config.residuals.use_rich_analysis,
            decomp_params=config.residuals.decomposition_params,
            upsamp_params=config.residuals.upsampling_params
        )
        embedding_parts.append(residuals_emb)
    
    # 5. Directional FFT features
    if config.directional_fft.enabled:
        from .features.directional_fft import compute_directional_fft_embedding
        fft_emb = compute_directional_fft_embedding(
            patch,
            angles=config.directional_fft.angles,
            stats=config.directional_fft.stats
        )
        embedding_parts.append(fft_emb)
    
    if not embedding_parts:
        raise ValueError("No signature types enabled in config")
    
    return np.concatenate(embedding_parts)


def compute_embeddings_batch_from_config(
    patches: List[Patch],
    config: 'SignatureConfig',
    verbose: bool = True
) -> np.ndarray:
    """
    Compute embeddings for a batch of patches using config.
    
    Args:
        patches: List of Patch objects
        config: SignatureConfig specifying which features to compute
        verbose: Print progress
        
    Returns:
        2D numpy array of shape (n_patches, embedding_dim)
    """
    if not patches:
        return np.array([])
    
    embeddings = []
    n_patches = len(patches)
    
    # Show what's being computed
    if verbose:
        enabled = config.get_enabled_types()
        print(f"  Signature types: {', '.join(enabled)}")
        print(f"  Expected dimension: {config.get_total_dim()}")
    
    for i, patch in enumerate(patches):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Computing embeddings: {i + 1}/{n_patches}")
        
        emb = compute_embedding_from_config(patch.data, config)
        embeddings.append(emb)
    
    if verbose:
        print(f"  Computed {n_patches} embeddings")
    
    return np.vstack(embeddings)


def get_embedding_dim_from_config(config: 'SignatureConfig') -> int:
    """
    Get the dimension of embeddings for a given config.
    
    Args:
        config: SignatureConfig
        
    Returns:
        Total embedding dimension
    """
    return config.get_total_dim()


def get_embedding_labels_from_config(config: 'SignatureConfig') -> List[str]:
    """
    Get human-readable labels for each embedding dimension.
    
    Args:
        config: SignatureConfig
        
    Returns:
        List of labels for each dimension
    """
    labels = []
    
    # Decomposition labels
    if config.decomposition.enabled:
        stats = ['mean', 'std', 'energy', 'entropy', 'range', 'median']
        for method in config.decomposition.methods:
            for stat in stats:
                labels.append(f"decomp_{method}_{stat}")
    
    # Geomorphometric labels
    if config.geomorphometric.enabled:
        from .features.geomorphometric import get_geomorphometric_labels
        labels.extend(get_geomorphometric_labels(config.geomorphometric.features))
    
    # Texture labels
    if config.texture.enabled:
        from .features.texture import get_texture_labels
        labels.extend(get_texture_labels(config.texture.features))
    
    # Directional FFT labels
    if config.directional_fft.enabled:
        from .features.directional_fft import get_directional_fft_labels
        labels.extend(get_directional_fft_labels(
            config.directional_fft.angles,
            config.directional_fft.stats
        ))
    
    return labels

