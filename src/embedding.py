"""
Embedding Extraction Module

Converts terrain patches into feature vectors using decomposition signatures.
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Any
from scipy.stats import entropy as scipy_entropy

from .decomposition import DECOMPOSITION_METHODS, DEFAULT_PARAMS
from .tiling import Patch


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

