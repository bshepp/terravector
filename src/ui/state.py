"""
Application State Management

Handles the persistent state for the Gradio UI including loaded DEMs,
built indices, and current query results.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io import load_dem, get_dem_info
from src.tiling import tile_dem, patches_to_metadata, get_tiling_info, Patch
from src.embedding import compute_embeddings_batch, compute_embeddings_batch_from_config
from src.index import TerrainIndex, build_index
from src.config import get_preset, load_config, validate_config, SignatureConfig
from src.visualization import hillshade


@dataclass
class AppState:
    """
    Holds application state between Gradio interactions.
    """
    # DEM data
    dem: Optional[np.ndarray] = None
    dem_path: Optional[str] = None
    dem_info: Optional[Dict[str, Any]] = None
    
    # Patches
    patches: Optional[List[Patch]] = None
    patch_size: int = 64
    
    # Index
    index: Optional[TerrainIndex] = None
    index_path: Optional[str] = None
    
    # Config
    config: Optional[SignatureConfig] = None
    
    # Query results
    last_query_id: Optional[int] = None
    last_results: Optional[List[Tuple[int, float, Dict]]] = None
    
    def load_dem_file(self, path: str) -> str:
        """Load a DEM from file path."""
        try:
            path = Path(path).resolve()
            if not path.exists():
                return f"Error: File not found: {path}"
            
            self.dem = load_dem(str(path))
            self.dem_path = str(path)
            self.dem_info = get_dem_info(self.dem)
            
            # Reset index when loading new DEM
            self.index = None
            self.patches = None
            self.last_query_id = None
            self.last_results = None
            
            return f"Loaded DEM: {path.name} ({self.dem_info['shape'][0]}×{self.dem_info['shape'][1]})"
        except Exception as e:
            return f"Error loading DEM: {str(e)}"
    
    def load_index_file(self, path: str) -> str:
        """Load an existing index from file."""
        try:
            path = Path(path).resolve()
            if not path.exists() and not path.with_suffix('.idx').exists():
                return f"Error: Index not found: {path}"
            
            self.index = TerrainIndex.load(str(path))
            self.index_path = str(path)
            
            # Try to load the source DEM
            sample_meta = next(iter(self.index.metadata.values()), {})
            dem_path = sample_meta.get('dem_path')
            
            if dem_path and Path(dem_path).exists():
                self.dem = load_dem(dem_path)
                self.dem_path = dem_path
                self.dem_info = get_dem_info(self.dem)
                self.patch_size = sample_meta.get('patch_size', 64)
            
            stats = self.index.get_stats()
            return f"Loaded index: {stats['n_patches']} patches, dim={stats['embedding_dim']}"
        except Exception as e:
            return f"Error loading index: {str(e)}"
    
    def build_new_index(
        self,
        preset_name: str = 'default',
        patch_size: int = 64,
        overlap: int = 0,
        progress_callback=None
    ) -> str:
        """Build a new index from the loaded DEM."""
        if self.dem is None:
            return "Error: No DEM loaded"
        
        try:
            self.patch_size = patch_size
            
            # Get config
            try:
                self.config = get_preset(preset_name)
            except ValueError:
                # Try loading as file path
                self.config = load_config(preset_name)
            
            errors = validate_config(self.config)
            if errors:
                return f"Config error: {'; '.join(errors)}"
            
            # Tile DEM
            if progress_callback:
                progress_callback(0.1, "Tiling DEM...")
            
            self.patches = tile_dem(
                self.dem,
                patch_size=patch_size,
                overlap=overlap,
                min_valid_fraction=0.8
            )
            
            if len(self.patches) == 0:
                return "Error: No valid patches found"
            
            # Compute embeddings
            if progress_callback:
                progress_callback(0.3, f"Computing embeddings for {len(self.patches)} patches...")
            
            embeddings = compute_embeddings_batch_from_config(
                self.patches, self.config, verbose=False
            )
            
            # Build index
            if progress_callback:
                progress_callback(0.8, "Building HNSW index...")
            
            metadata = patches_to_metadata(self.patches)
            
            # Add source info
            for pid in metadata:
                metadata[pid]['dem_path'] = self.dem_path
                metadata[pid]['patch_size'] = patch_size
            
            self.index = build_index(
                embeddings,
                metadata,
                space=self.config.space,
                normalize=self.config.normalize
            )
            
            if self.config:
                self.index.signature_config = self.config.to_dict()
            
            if progress_callback:
                progress_callback(1.0, "Done!")
            
            stats = self.index.get_stats()
            enabled = self.config.get_enabled_types()
            return f"Built index: {stats['n_patches']} patches, dim={stats['embedding_dim']}, signature={'+'.join(enabled)}"
            
        except Exception as e:
            import traceback
            return f"Error building index: {str(e)}\n{traceback.format_exc()}"
    
    def query_by_coords(self, y: int, x: int, k: int = 8) -> str:
        """Query for similar patches at given coordinates."""
        if self.index is None:
            return "Error: No index loaded"
        
        try:
            # Find patch at coordinates
            for pid, meta in self.index.metadata.items():
                ps = meta.get('patch_size', self.patch_size)
                if (meta['y_start'] <= y < meta['y_start'] + ps and
                    meta['x_start'] <= x < meta['x_start'] + ps):
                    self.last_query_id = pid
                    self.last_results = self.index.query_by_id(pid, k=k)
                    return f"Query patch #{pid}: found {len(self.last_results)} similar"
            
            return f"No patch found at ({y}, {x})"
        except Exception as e:
            return f"Query error: {str(e)}"
    
    def query_by_id(self, patch_id: int, k: int = 8) -> str:
        """Query for similar patches by patch ID."""
        if self.index is None:
            return "Error: No index loaded"
        
        try:
            if patch_id not in self.index.metadata:
                return f"Error: Patch {patch_id} not in index"
            
            self.last_query_id = patch_id
            self.last_results = self.index.query_by_id(patch_id, k=k)
            return f"Query patch #{patch_id}: found {len(self.last_results)} similar"
        except Exception as e:
            return f"Query error: {str(e)}"
    
    def save_index(self, path: str) -> str:
        """Save the current index to file."""
        if self.index is None:
            return "Error: No index to save"
        
        try:
            self.index.save(path)
            self.index_path = path
            return f"Saved index to: {path}"
        except Exception as e:
            return f"Error saving: {str(e)}"
    
    def get_dem_hillshade(self) -> Optional[np.ndarray]:
        """Get hillshade image of the loaded DEM."""
        if self.dem is None:
            return None
        return hillshade(self.dem)
    
    def get_status(self) -> str:
        """Get current status string."""
        parts = []
        
        if self.dem is not None:
            parts.append(f"DEM: {self.dem.shape[0]}×{self.dem.shape[1]}")
        else:
            parts.append("No DEM")
        
        if self.index is not None:
            stats = self.index.get_stats()
            parts.append(f"Index: {stats['n_patches']} patches")
            parts.append(f"Dim: {stats['embedding_dim']}")
        else:
            parts.append("No index")
        
        return " | ".join(parts)

