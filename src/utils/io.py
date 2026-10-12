"""I/O utilities for terravector."""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional


def load_dem(path: Union[str, Path]) -> np.ndarray:
    """
    Load a DEM from file.
    
    Supports:
    - .npy files (numpy arrays)
    - .tif/.tiff files (GeoTIFF, requires rasterio)
    
    Args:
        path: Path to DEM file
        
    Returns:
        2D numpy array of elevation values
    """
    path = Path(path)
    
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix in ('.tif', '.tiff'):
        try:
            import rasterio
            with rasterio.open(path) as src:
                return src.read(1)
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF support: pip install rasterio")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def get_dem_info(dem: np.ndarray) -> Dict[str, Any]:
    """Get basic statistics about a DEM."""
    return {
        'shape': dem.shape,
        'min': float(np.nanmin(dem)),
        'max': float(np.nanmax(dem)),
        'mean': float(np.nanmean(dem)),
        'std': float(np.nanstd(dem)),
        'nan_count': int(np.isnan(dem).sum())
    }

