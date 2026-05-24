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
    upsamp_params: Optional[Dict[str, Dict[str, Any]]] = None,
    down_order: int = 1,
    down_factor: float = 0.5,
    up_scale: int = 2,
    bidirectional: bool = False,
    turing_intermediate: bool = False,
    turing_iterations: int = 50,
    turing_Du: float = 0.16,
    turing_Dv: float = 0.08,
    turing_F: float = 0.035,
    turing_k: float = 0.060,
    return_diagnostics: bool = False,
):
    """
    Compute RESIDUALS-style embedding from decomposition × upsampling combinations.

    For each decomposition method:
      1. Decompose patch → get residual
      2. For each upsampling method:
         a. Path A: downsample residual by down_factor → upsample by up_scale
         b. (optional Turing) run Gray-Scott n_iter steps on the round-trip output
         c. Compute analysis vector on the result (the "A" slot)
         d. (optional bidirectional) Path B: upsample first by up_scale,
            then downsample by down_factor, apply Turing if enabled,
            and compute analysis vector on (A - B) — the per-pair asymmetry
            ("MTile" scale-asymmetry feature).

    Round-trip parameters (down_order, down_factor, up_scale) used to be
    hardcoded inside the function — they are now explicit so embeddings
    are fully reproducible from their saved signature config.

    Args:
        patch: 2D numpy array.
        decomposition_methods: List of decomposition method names.
        upsampling_methods: List of upsampling method names.
        use_rich_analysis: Use 20-dim analysis or 6-dim basic stats per slot.
        decomp_params: Parameters for decomposition methods.
        upsamp_params: Parameters for upsampling methods.
        down_order: scipy.ndimage.zoom interpolation order on the down step.
        down_factor: zoom factor applied to the residual (must be 1/up_scale).
        up_scale: scale factor passed to the upsampler.
        bidirectional: also compute path B and append the (A - B) analysis
            vector as an additional per-pair slot. Doubles the residuals slice.
        turing_intermediate: run Gray-Scott on each round-trip output before
            analysis. Amplifies subtle textural differences into discrete
            attractor patterns, improving downstream HNSW separability.
        turing_iterations, turing_Du, turing_Dv, turing_F, turing_k:
            Gray-Scott parameters. Defaults yield well-developed spot/labyrinth
            patterns within ~50 iterations on the down-sampled scale.
        return_diagnostics: if True, return (embedding, diagnostics) where
            diagnostics records params_used and the list of (decomp, upsamp)
            pairs whose computation failed (and therefore embedded as zeros).
            Lets the build pipeline audit silent failures.

    Returns:
        1D float32 numpy array (default), or (array, diagnostics_dict) if
        return_diagnostics=True.
    """
    from .decomposition import run_decomposition
    from .upsampling import run_upsampling
    from .features.analysis import analyze_features, analysis_to_vector
    from .utils.turing import gray_scott_pattern
    from scipy.ndimage import zoom

    if decomp_params is None:
        decomp_params = {}
    if upsamp_params is None:
        upsamp_params = {}

    dim_per_slot = 20 if use_rich_analysis else 6
    slots_per_pair = 2 if bidirectional else 1
    failed_pairs: List[str] = []

    def _analyze(arr: np.ndarray) -> np.ndarray:
        if use_rich_analysis:
            v = analysis_to_vector(analyze_features(arr))
        else:
            v = np.array(compute_residual_stats(arr), dtype=np.float32)
        # Near-constant inputs make scipy.stats.skew/kurtosis return NaN.
        # That NaN otherwise poisons z-score normalization downstream.
        # 0 is the right neutral fill: it means "no measurable signal" for
        # that statistic, which matches the underlying condition.
        if not np.all(np.isfinite(v)):
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v.astype(np.float32, copy=False)

    def _maybe_turing(arr: np.ndarray) -> np.ndarray:
        if not turing_intermediate:
            return arr
        return gray_scott_pattern(
            arr,
            n_iter=turing_iterations,
            Du=turing_Du, Dv=turing_Dv, F=turing_F, k=turing_k,
        )

    embedding_parts = []

    for decomp_name in decomposition_methods:
        try:
            _trend, residual = run_decomposition(
                decomp_name, patch,
                params=decomp_params.get(decomp_name),
            )
        except Exception:
            # Decomposition failed: emit zeros for every (decomp, upsamp)
            # slot under this decomposition, but record each as a failure
            # so the build pipeline can audit.
            n_upsamp = len(upsampling_methods)
            embedding_parts.append(
                np.zeros(n_upsamp * slots_per_pair * dim_per_slot, dtype=np.float32)
            )
            for upsamp_name in upsampling_methods:
                failed_pairs.append(f"{decomp_name}/{upsamp_name}:decomp_error")
            continue

        for upsamp_name in upsampling_methods:
            # --- Path A: down then up. Always emits exactly one slot. ---
            a_field = None
            try:
                a_down = zoom(residual, down_factor, order=down_order)
                a_up = run_upsampling(
                    upsamp_name, a_down, scale=up_scale,
                    params=upsamp_params.get(upsamp_name),
                )
                # Crop to residual shape — required because zoom can
                # round dimensions independently. With down_factor*up_scale=1
                # the mismatch is at most 1px per axis.
                min_h = min(residual.shape[0], a_up.shape[0])
                min_w = min(residual.shape[1], a_up.shape[1])
                a_up = a_up[:min_h, :min_w]
                a_field = _maybe_turing(a_up)
                embedding_parts.append(_analyze(a_field))
            except Exception as exc:
                embedding_parts.append(np.zeros(dim_per_slot, dtype=np.float32))
                failed_pairs.append(f"{decomp_name}/{upsamp_name}:A:{type(exc).__name__}")

            # --- Path B (only if bidirectional): always emits exactly one slot. ---
            # Tracked separately from path A so a B-only failure doesn't
            # double-count zeros against the A slot already emitted above.
            if bidirectional:
                try:
                    b_up = run_upsampling(
                        upsamp_name, residual, scale=up_scale,
                        params=upsamp_params.get(upsamp_name),
                    )
                    b_down = zoom(b_up, down_factor, order=down_order)
                    min_h = min(residual.shape[0], b_down.shape[0])
                    min_w = min(residual.shape[1], b_down.shape[1])
                    b_down = b_down[:min_h, :min_w]
                    b_field = _maybe_turing(b_down)

                    if a_field is not None:
                        h = min(a_field.shape[0], b_field.shape[0])
                        w = min(a_field.shape[1], b_field.shape[1])
                        asymmetry = a_field[:h, :w] - b_field[:h, :w]
                        embedding_parts.append(_analyze(asymmetry))
                    else:
                        # A failed → can't compute A - B. Emit zeros.
                        embedding_parts.append(np.zeros(dim_per_slot, dtype=np.float32))
                        failed_pairs.append(
                            f"{decomp_name}/{upsamp_name}:B:no_A_to_compare"
                        )
                except Exception as exc:
                    embedding_parts.append(np.zeros(dim_per_slot, dtype=np.float32))
                    failed_pairs.append(
                        f"{decomp_name}/{upsamp_name}:B:{type(exc).__name__}"
                    )

    embedding = np.concatenate(embedding_parts).astype(np.float32)

    if return_diagnostics:
        diagnostics = {
            "failed_pairs": failed_pairs,
            "n_failed_pairs": len(failed_pairs),
            "params_used": {
                "down_order": down_order,
                "down_factor": down_factor,
                "up_scale": up_scale,
                "bidirectional": bidirectional,
                "turing_intermediate": turing_intermediate,
                "turing_iterations": turing_iterations if turing_intermediate else 0,
            },
        }
        return embedding, diagnostics
    return embedding


# =============================================================================
# Config-Based Embedding (New API)
# =============================================================================

def compute_embedding_from_config(
    patch: np.ndarray,
    config: 'SignatureConfig',
    return_diagnostics: bool = False,
):
    """
    Compute embedding vector using configuration-specified signature types.

    Concatenates features from all enabled signature types in order:
    1. Decomposition (if enabled)
    2. Geomorphometric (if enabled)
    3. Texture (if enabled)
    4. Residuals - decomp × upsamp combinations (if enabled)
    5. Directional FFT (if enabled)

    Args:
        patch: 2D numpy array (terrain patch)
        config: SignatureConfig specifying which features to compute
        return_diagnostics: if True, returns (embedding, diagnostics). Diagnostics
            currently surface only from the residuals slice (which is the slice with
            silent-failure modes worth auditing). The dict has the same shape as
            compute_residuals_embedding's diagnostics, plus a top-level marker
            indicating whether the residuals slice was active.

    Returns:
        1D numpy array (combined embedding vector), or (array, diagnostics_dict)
        when return_diagnostics=True.
    """
    embedding_parts = []
    residuals_diagnostics = None

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
        residuals_result = compute_residuals_embedding(
            patch,
            decomposition_methods=config.residuals.decomposition_methods,
            upsampling_methods=config.residuals.upsampling_methods,
            use_rich_analysis=config.residuals.use_rich_analysis,
            decomp_params=config.residuals.decomposition_params,
            upsamp_params=config.residuals.upsampling_params,
            down_order=config.residuals.down_order,
            down_factor=config.residuals.down_factor,
            up_scale=config.residuals.up_scale,
            bidirectional=config.residuals.bidirectional,
            turing_intermediate=config.residuals.turing_intermediate,
            turing_iterations=config.residuals.turing_iterations,
            turing_Du=config.residuals.turing_Du,
            turing_Dv=config.residuals.turing_Dv,
            turing_F=config.residuals.turing_F,
            turing_k=config.residuals.turing_k,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            residuals_emb, residuals_diagnostics = residuals_result
        else:
            residuals_emb = residuals_result
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

    embedding = np.concatenate(embedding_parts)

    if return_diagnostics:
        diagnostics = {
            "residuals_enabled": bool(config.residuals.enabled),
            "residuals": residuals_diagnostics,
        }
        return embedding, diagnostics
    return embedding


def compute_embeddings_batch_from_config(
    patches: List[Patch],
    config: 'SignatureConfig',
    verbose: bool = True,
    return_diagnostics: bool = False,
):
    """
    Compute embeddings for a batch of patches using config.

    Args:
        patches: List of Patch objects
        config: SignatureConfig specifying which features to compute
        verbose: Print progress
        return_diagnostics: if True, returns (embeddings, batch_diagnostics) where
            batch_diagnostics aggregates failures across patches:
                {
                  "n_patches": int,
                  "n_patches_with_failures": int,
                  "total_failed_pairs": int,
                  "failure_counts_by_pair": {pair_key: count, ...},
                  "patches_with_failures": [patch_index, ...],
                  "params_used": {...}   # only populated when residuals is enabled
                }

    Returns:
        2D numpy array of shape (n_patches, embedding_dim), or
        (array, batch_diagnostics_dict) if return_diagnostics=True.
    """
    if not patches:
        if return_diagnostics:
            return np.array([]), {
                "n_patches": 0,
                "n_patches_with_failures": 0,
                "total_failed_pairs": 0,
                "failure_counts_by_pair": {},
                "patches_with_failures": [],
                "params_used": None,
            }
        return np.array([])

    embeddings = []
    n_patches = len(patches)
    failure_counts: Dict[str, int] = {}
    patches_with_failures: List[int] = []
    params_used = None

    if verbose:
        enabled = config.get_enabled_types()
        print(f"  Signature types: {', '.join(enabled)}")
        print(f"  Expected dimension: {config.get_total_dim()}")

    for i, patch in enumerate(patches):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Computing embeddings: {i + 1}/{n_patches}")

        if return_diagnostics:
            emb, diag = compute_embedding_from_config(
                patch.data, config, return_diagnostics=True
            )
            res_diag = diag.get("residuals") or {}
            failed = res_diag.get("failed_pairs", [])
            if failed:
                patches_with_failures.append(i)
                for entry in failed:
                    # entry looks like "decomp/upsamp:reason" — key by pair+reason
                    failure_counts[entry] = failure_counts.get(entry, 0) + 1
            if params_used is None and res_diag.get("params_used"):
                params_used = res_diag["params_used"]
        else:
            emb = compute_embedding_from_config(patch.data, config)

        embeddings.append(emb)

    if verbose:
        print(f"  Computed {n_patches} embeddings")
        if return_diagnostics and patches_with_failures:
            total = sum(failure_counts.values())
            print(
                f"  Diagnostics: {len(patches_with_failures)}/{n_patches} patches had "
                f"{total} failed (decomp, upsamp) pairs"
            )

    arr = np.vstack(embeddings)

    if return_diagnostics:
        batch_diag = {
            "n_patches": n_patches,
            "n_patches_with_failures": len(patches_with_failures),
            "total_failed_pairs": sum(failure_counts.values()),
            "failure_counts_by_pair": failure_counts,
            "patches_with_failures": patches_with_failures,
            "params_used": params_used,
        }
        return arr, batch_diag
    return arr


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

