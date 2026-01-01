"""I/O utilities for terravector."""

import numpy as np
import pickle
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


def save_index(
    index: Any,
    embeddings: np.ndarray,
    metadata: Dict[int, Dict],
    path: Union[str, Path]
) -> None:
    """
    Save HNSW index and associated data.
    
    Args:
        index: hnswlib index object
        embeddings: Original embedding vectors
        metadata: Patch metadata (coordinates, etc.)
        path: Output path (without extension)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save HNSW index
    index.save_index(str(path.with_suffix('.idx')))
    
    # Save embeddings and metadata
    data = {
        'embeddings': embeddings,
        'metadata': metadata,
        'dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        'count': len(metadata)
    }
    with open(path.with_suffix('.meta'), 'wb') as f:
        pickle.dump(data, f)


def load_index(path: Union[str, Path]) -> tuple:
    """
    Load HNSW index and associated data.
    
    Args:
        path: Path to index (with or without extension)
        
    Returns:
        Tuple of (index, embeddings, metadata)
        
    Note:
        For full index loading, use TerrainIndex.load() from src.index
    """
    import nmslib
    
    path = Path(path)
    if path.suffix:
        path = path.with_suffix('')
    
    # Load metadata first to get dimensions
    with open(path.with_suffix('.meta'), 'rb') as f:
        data = pickle.load(f)
    
    # Load nmslib index
    index = nmslib.init(method='hnsw', space=data.get('space', 'cosinesimil'))
    index.loadIndex(str(path.with_suffix('.idx')), load_data=True)
    
    return index, data['embeddings'], data['metadata']


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

