"""
Configuration System for terravector

Handles YAML config parsing for signature selection and feature combination.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DecompositionConfig:
    """Configuration for decomposition-based features."""
    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: [
        'gaussian', 'bilateral', 'wavelet_dwt', 
        'morphological', 'tophat', 'polynomial'
    ])
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class GeomorphometricConfig:
    """Configuration for geomorphometric features."""
    enabled: bool = False
    features: List[str] = field(default_factory=lambda: [
        'slope', 'aspect', 'curvature', 'tpi', 'tri', 'roughness'
    ])
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TextureConfig:
    """Configuration for texture features."""
    enabled: bool = False
    features: List[str] = field(default_factory=lambda: ['glcm', 'lbp'])
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ResidualsConfig:
    """Configuration for RESIDUALS-style decomposition × upsampling signatures."""
    enabled: bool = False
    decomposition_methods: List[str] = field(default_factory=lambda: [
        'gaussian', 'bilateral', 'wavelet_dwt', 'morphological'
    ])
    upsampling_methods: List[str] = field(default_factory=lambda: [
        'bicubic', 'lanczos', 'bspline', 'fft_zeropad'
    ])
    # Use rich analysis (20 dims) or basic stats (6 dims) per combination
    use_rich_analysis: bool = True
    decomposition_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    upsampling_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class DirectionalFFTConfig:
    """Configuration for directional FFT signatures."""
    enabled: bool = False
    # Angles in degrees to sample (0° = horizontal features, 90° = vertical)
    angles: List[float] = field(default_factory=lambda: [0, 45, 90, 135])
    # Statistics to compute per angle
    stats: List[str] = field(default_factory=lambda: [
        'energy', 'low_freq_ratio', 'high_freq_ratio',
        'peak_freq', 'spectral_centroid', 'spectral_spread'
    ])


@dataclass
class SignatureConfig:
    """Complete signature configuration."""
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    geomorphometric: GeomorphometricConfig = field(default_factory=GeomorphometricConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    residuals: ResidualsConfig = field(default_factory=ResidualsConfig)
    directional_fft: DirectionalFFTConfig = field(default_factory=DirectionalFFTConfig)
    
    # Index settings
    normalize: bool = True
    space: str = 'cosinesimil'
    
    def get_enabled_types(self) -> List[str]:
        """Return list of enabled signature types."""
        enabled = []
        if self.decomposition.enabled:
            enabled.append('decomposition')
        if self.geomorphometric.enabled:
            enabled.append('geomorphometric')
        if self.texture.enabled:
            enabled.append('texture')
        if self.residuals.enabled:
            enabled.append('residuals')
        if self.directional_fft.enabled:
            enabled.append('directional_fft')
        return enabled
    
    def get_total_dim(self) -> int:
        """Calculate total embedding dimension."""
        from .decomposition import DECOMPOSITION_METHODS
        from .features.geomorphometric import GEOMORPHOMETRIC_FEATURES
        from .features.texture import get_texture_dim
        from .features.analysis import get_analysis_dim
        from .features.directional_fft import get_directional_fft_dim
        
        dim = 0
        
        if self.decomposition.enabled:
            n_methods = len(self.decomposition.methods)
            dim += n_methods * 6  # 6 stats per method
        
        if self.geomorphometric.enabled:
            n_features = len(self.geomorphometric.features)
            dim += n_features * 6  # 6 stats per feature
        
        if self.texture.enabled:
            dim += get_texture_dim(self.texture.features)
        
        if self.residuals.enabled:
            n_decomp = len(self.residuals.decomposition_methods)
            n_upsamp = len(self.residuals.upsampling_methods)
            n_combos = n_decomp * n_upsamp
            if self.residuals.use_rich_analysis:
                dim += n_combos * get_analysis_dim()  # 20 dims per combo
            else:
                dim += n_combos * 6  # 6 basic stats per combo
        
        if self.directional_fft.enabled:
            dim += get_directional_fft_dim(
                self.directional_fft.angles,
                self.directional_fft.stats
            )
        
        return dim
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'signature': {
                'decomposition': {
                    'enabled': self.decomposition.enabled,
                    'methods': self.decomposition.methods,
                    'params': self.decomposition.params,
                },
                'geomorphometric': {
                    'enabled': self.geomorphometric.enabled,
                    'features': self.geomorphometric.features,
                    'params': self.geomorphometric.params,
                },
                'texture': {
                    'enabled': self.texture.enabled,
                    'features': self.texture.features,
                    'params': self.texture.params,
                },
                'residuals': {
                    'enabled': self.residuals.enabled,
                    'decomposition_methods': self.residuals.decomposition_methods,
                    'upsampling_methods': self.residuals.upsampling_methods,
                    'use_rich_analysis': self.residuals.use_rich_analysis,
                },
                'directional_fft': {
                    'enabled': self.directional_fft.enabled,
                    'angles': self.directional_fft.angles,
                    'stats': self.directional_fft.stats,
                },
            },
            'normalize': self.normalize,
            'space': self.space,
        }


def load_config(path: Union[str, Path]) -> SignatureConfig:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        SignatureConfig object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    
    return parse_config(raw)


def parse_config(raw: Dict[str, Any]) -> SignatureConfig:
    """
    Parse raw dictionary into SignatureConfig.
    
    Args:
        raw: Dictionary from YAML parsing
        
    Returns:
        SignatureConfig object
    """
    config = SignatureConfig()
    
    if raw is None:
        return config
    
    sig = raw.get('signature', {})
    
    # Decomposition config
    decomp = sig.get('decomposition', {})
    if isinstance(decomp, dict):
        config.decomposition.enabled = decomp.get('enabled', True)
        if 'methods' in decomp:
            config.decomposition.methods = decomp['methods']
        if 'params' in decomp:
            config.decomposition.params = decomp['params']
    
    # Geomorphometric config
    geomorph = sig.get('geomorphometric', {})
    if isinstance(geomorph, dict):
        config.geomorphometric.enabled = geomorph.get('enabled', False)
        if 'features' in geomorph:
            config.geomorphometric.features = geomorph['features']
        if 'params' in geomorph:
            config.geomorphometric.params = geomorph['params']
    
    # Texture config
    texture = sig.get('texture', {})
    if isinstance(texture, dict):
        config.texture.enabled = texture.get('enabled', False)
        if 'features' in texture:
            config.texture.features = texture['features']
        if 'params' in texture:
            config.texture.params = texture['params']
    
    # Residuals config (DIVERGE-style decomp × upsamp)
    residuals = sig.get('residuals', {})
    if isinstance(residuals, dict):
        config.residuals.enabled = residuals.get('enabled', False)
        if 'decomposition_methods' in residuals:
            config.residuals.decomposition_methods = residuals['decomposition_methods']
        if 'upsampling_methods' in residuals:
            config.residuals.upsampling_methods = residuals['upsampling_methods']
        if 'use_rich_analysis' in residuals:
            config.residuals.use_rich_analysis = residuals['use_rich_analysis']
    
    # Directional FFT config
    directional_fft = sig.get('directional_fft', {})
    if isinstance(directional_fft, dict):
        config.directional_fft.enabled = directional_fft.get('enabled', False)
        if 'angles' in directional_fft:
            config.directional_fft.angles = directional_fft['angles']
        if 'stats' in directional_fft:
            config.directional_fft.stats = directional_fft['stats']
    
    # Top-level settings
    config.normalize = raw.get('normalize', True)
    config.space = raw.get('space', 'cosinesimil')
    
    return config


def get_default_config() -> SignatureConfig:
    """
    Get default configuration (decomposition only, for backwards compatibility).
    
    Returns:
        SignatureConfig with only decomposition enabled
    """
    return SignatureConfig()


def validate_config(config: SignatureConfig) -> List[str]:
    """
    Validate configuration and return list of errors.
    
    Args:
        config: SignatureConfig to validate
        
    Returns:
        List of error messages (empty if valid)
    """
    from .decomposition import DECOMPOSITION_METHODS
    from .features.geomorphometric import GEOMORPHOMETRIC_FEATURES
    from .features.texture import TEXTURE_FEATURES
    
    errors = []
    
    # Check at least one signature type is enabled
    if not any([
        config.decomposition.enabled,
        config.geomorphometric.enabled,
        config.texture.enabled,
        config.residuals.enabled,
        config.directional_fft.enabled
    ]):
        errors.append("At least one signature type must be enabled")
    
    # Validate decomposition methods
    if config.decomposition.enabled:
        for method in config.decomposition.methods:
            if method not in DECOMPOSITION_METHODS:
                errors.append(f"Unknown decomposition method: {method}")
    
    # Validate geomorphometric features
    if config.geomorphometric.enabled:
        for feat in config.geomorphometric.features:
            if feat not in GEOMORPHOMETRIC_FEATURES:
                errors.append(f"Unknown geomorphometric feature: {feat}")
    
    # Validate texture features
    if config.texture.enabled:
        for feat in config.texture.features:
            if feat not in TEXTURE_FEATURES:
                errors.append(f"Unknown texture feature: {feat}")
    
    # Validate residuals config
    if config.residuals.enabled:
        from .decomposition import list_decompositions
        from .upsampling import list_upsamplings
        
        valid_decomp = list_decompositions()
        valid_upsamp = list_upsamplings()
        
        for method in config.residuals.decomposition_methods:
            if method not in valid_decomp:
                errors.append(f"Unknown residuals decomposition method: {method}")
        
        for method in config.residuals.upsampling_methods:
            if method not in valid_upsamp:
                errors.append(f"Unknown residuals upsampling method: {method}")
    
    # Validate directional FFT config
    if config.directional_fft.enabled:
        from .features.directional_fft import DIRECTIONAL_FFT_STATS
        
        for stat in config.directional_fft.stats:
            if stat not in DIRECTIONAL_FFT_STATS:
                errors.append(f"Unknown directional FFT statistic: {stat}")
        
        for angle in config.directional_fft.angles:
            if not isinstance(angle, (int, float)):
                errors.append(f"Invalid angle type: {angle} (must be numeric)")
    
    # Validate space
    valid_spaces = ['cosinesimil', 'l2', 'l1']
    if config.space not in valid_spaces:
        errors.append(f"Invalid space: {config.space}. Must be one of {valid_spaces}")
    
    return errors


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS: Dict[str, SignatureConfig] = {}


def _create_presets():
    """Create preset configurations."""
    global PRESETS
    
    # Default: decomposition only (original behavior)
    PRESETS['default'] = SignatureConfig()
    
    # Classic: geomorphometric only
    classic = SignatureConfig()
    classic.decomposition.enabled = False
    classic.geomorphometric.enabled = True
    PRESETS['classic'] = classic
    
    # Texture: texture features only
    texture_only = SignatureConfig()
    texture_only.decomposition.enabled = False
    texture_only.texture.enabled = True
    PRESETS['texture'] = texture_only
    
    # Hybrid: all feature types
    hybrid = SignatureConfig()
    hybrid.geomorphometric.enabled = True
    hybrid.texture.enabled = True
    PRESETS['hybrid'] = hybrid
    
    # Minimal: fast computation with fewer features
    minimal = SignatureConfig()
    minimal.decomposition.methods = ['gaussian', 'morphological']
    minimal.geomorphometric.enabled = True
    minimal.geomorphometric.features = ['slope', 'curvature']
    PRESETS['minimal'] = minimal
    
    # Residuals: DIVERGE-style decomposition × upsampling signatures
    residuals = SignatureConfig()
    residuals.decomposition.enabled = False
    residuals.residuals.enabled = True
    residuals.residuals.decomposition_methods = [
        'gaussian', 'bilateral', 'wavelet_dwt', 'morphological'
    ]
    residuals.residuals.upsampling_methods = [
        'bicubic', 'lanczos', 'bspline', 'fft_zeropad'
    ]
    residuals.residuals.use_rich_analysis = True
    PRESETS['residuals'] = residuals
    
    # Residuals-extended: more methods for comprehensive coverage
    residuals_ext = SignatureConfig()
    residuals_ext.decomposition.enabled = False
    residuals_ext.residuals.enabled = True
    residuals_ext.residuals.decomposition_methods = [
        'gaussian', 'bilateral', 'wavelet_dwt', 'morphological',
        'tophat', 'polynomial', 'dog', 'median'
    ]
    residuals_ext.residuals.upsampling_methods = [
        'bicubic', 'lanczos', 'bspline', 'fft_zeropad',
        'bilinear', 'cubic_mitchell'
    ]
    residuals_ext.residuals.use_rich_analysis = True
    PRESETS['residuals_extended'] = residuals_ext
    
    # Spectral: directional FFT signatures for oriented features
    spectral = SignatureConfig()
    spectral.decomposition.enabled = False
    spectral.directional_fft.enabled = True
    spectral.directional_fft.angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    PRESETS['spectral'] = spectral
    
    # Spectral-hybrid: combines decomposition with directional FFT
    spectral_hybrid = SignatureConfig()
    spectral_hybrid.directional_fft.enabled = True
    spectral_hybrid.directional_fft.angles = [0, 45, 90, 135]
    PRESETS['spectral_hybrid'] = spectral_hybrid


_create_presets()


def get_preset(name: str) -> SignatureConfig:
    """
    Get a preset configuration by name.
    
    Args:
        name: Preset name ('default', 'classic', 'texture', 'hybrid', 'minimal')
        
    Returns:
        SignatureConfig for the preset
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    
    # Return a copy to avoid mutation
    import copy
    return copy.deepcopy(PRESETS[name])

