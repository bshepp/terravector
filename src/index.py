"""
HNSW Index Module

Build and query terrain similarity index using Hierarchical Navigable Small World graphs.
Uses nmslib for HNSW implementation.
"""

import numpy as np
import nmslib
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import pickle


class TerrainIndex:
    """
    HNSW-based index for terrain patch similarity search.
    
    Wraps nmslib with terrain-specific metadata handling.
    """
    
    def __init__(
        self,
        dim: int,
        space: str = 'cosinesimil',
        M: int = 16,
        ef_construction: int = 200
    ):
        """
        Initialize terrain index.
        
        Args:
            dim: Embedding dimension
            space: Distance metric ('cosinesimil', 'l2', 'l1')
            M: HNSW parameter - number of connections per layer
            ef_construction: HNSW parameter - construction time accuracy
        """
        self.dim = dim
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        
        self.index: Optional[nmslib.dist.FloatIndex] = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.normalization_params: Optional[Dict[str, Any]] = None
        self.is_built = False
    
    def build(
        self,
        embeddings: np.ndarray,
        metadata: Dict[int, Dict[str, Any]],
        normalize: bool = True,
        ef_search: int = 50
    ) -> None:
        """
        Build the index from embeddings.
        
        Args:
            embeddings: 2D array of shape (n_patches, dim)
            metadata: Dictionary mapping patch_id to metadata
            normalize: Apply z-score normalization
            ef_search: Query time accuracy parameter
        """
        n_items = len(embeddings)
        
        if n_items == 0:
            raise ValueError("Cannot build index with 0 embeddings")
        
        # Store original embeddings
        self.embeddings = embeddings.copy()
        
        # Normalize if requested
        if normalize:
            mean = np.mean(embeddings, axis=0)
            std = np.std(embeddings, axis=0)
            std[std == 0] = 1
            normalized = (embeddings - mean) / std
            self.normalization_params = {'mean': mean, 'std': std}
        else:
            normalized = embeddings
            self.normalization_params = None
        
        # Build nmslib HNSW index
        self.index = nmslib.init(method='hnsw', space=self.space)
        self.index.addDataPointBatch(normalized.astype(np.float32))
        
        self.index.createIndex({
            'M': self.M,
            'efConstruction': self.ef_construction,
            'post': 2  # Post-processing level
        }, print_progress=False)
        
        self.index.setQueryTimeParams({'efSearch': ef_search})
        
        # Store metadata
        self.metadata = metadata
        self.is_built = True
    
    def query(
        self,
        embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Find k most similar patches.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (patch_id, distance, metadata) tuples, sorted by similarity
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Normalize query if index was normalized
        if self.normalization_params is not None:
            embedding = (embedding - self.normalization_params['mean']) / self.normalization_params['std']
        
        # Query nmslib
        embedding = embedding.astype(np.float32).reshape(-1)
        ids, distances = self.index.knnQuery(embedding, k=min(k, len(self.metadata)))
        
        # Build results with metadata
        results = []
        for idx, dist in zip(ids, distances):
            meta = self.metadata.get(int(idx), {})
            results.append((int(idx), float(dist), meta))
        
        return results
    
    def query_by_id(self, patch_id: int, k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Find patches similar to an existing patch by ID.
        
        Args:
            patch_id: ID of the query patch
            k: Number of results (including the query patch itself)
            
        Returns:
            List of (patch_id, distance, metadata) tuples
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not available")
        
        if patch_id < 0 or patch_id >= len(self.embeddings):
            raise ValueError(f"Invalid patch_id: {patch_id}")
        
        return self.query(self.embeddings[patch_id], k=k)
    
    def save(self, path: str) -> None:
        """
        Save index to disk.
        
        Args:
            path: Output path (without extension)
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save nmslib index
        self.index.saveIndex(str(path.with_suffix('.idx')), save_data=True)
        
        # Save metadata and parameters
        data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'normalization_params': self.normalization_params,
            'dim': self.dim,
            'space': self.space,
            'M': self.M,
            'ef_construction': self.ef_construction
        }
        with open(path.with_suffix('.meta'), 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'TerrainIndex':
        """
        Load index from disk.
        
        Args:
            path: Path to index (with or without extension)
            
        Returns:
            Loaded TerrainIndex
        """
        path = Path(path)
        if path.suffix:
            path = path.with_suffix('')
        
        # Load metadata first
        with open(path.with_suffix('.meta'), 'rb') as f:
            data = pickle.load(f)
        
        # Create index object
        index = cls(
            dim=data['dim'],
            space=data['space'],
            M=data['M'],
            ef_construction=data['ef_construction']
        )
        
        # Load nmslib index
        index.index = nmslib.init(method='hnsw', space=data['space'])
        index.index.loadIndex(str(path.with_suffix('.idx')), load_data=True)
        
        # Restore state
        index.embeddings = data['embeddings']
        index.metadata = data['metadata']
        index.normalization_params = data['normalization_params']
        index.is_built = True
        
        return index
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'n_patches': len(self.metadata),
            'embedding_dim': self.dim,
            'space': self.space,
            'M': self.M,
            'normalized': self.normalization_params is not None,
            'is_built': self.is_built
        }


def build_index(
    embeddings: np.ndarray,
    metadata: Dict[int, Dict[str, Any]],
    space: str = 'cosinesimil',
    M: int = 16,
    ef_construction: int = 200,
    normalize: bool = True
) -> TerrainIndex:
    """
    Convenience function to build a terrain index.
    
    Args:
        embeddings: 2D array of embeddings
        metadata: Patch metadata dictionary
        space: Distance metric ('cosinesimil', 'l2', 'l1')
        M: HNSW connections parameter
        ef_construction: Construction accuracy parameter
        normalize: Apply normalization
        
    Returns:
        Built TerrainIndex
    """
    dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
    
    index = TerrainIndex(dim=dim, space=space, M=M, ef_construction=ef_construction)
    index.build(embeddings, metadata, normalize=normalize)
    
    return index
