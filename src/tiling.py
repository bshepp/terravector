"""
DEM Tiling Module

Divides DEMs into patches for embedding and similarity search.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class Patch:
    """A single terrain patch with metadata."""
    id: int
    data: np.ndarray
    row: int  # Row index in patch grid
    col: int  # Column index in patch grid
    y_start: int  # Pixel y coordinate in original DEM
    x_start: int  # Pixel x coordinate in original DEM
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center pixel coordinates in original DEM."""
        h, w = self.data.shape
        return (self.y_start + h // 2, self.x_start + w // 2)


def tile_dem(
    dem: np.ndarray,
    patch_size: int = 64,
    overlap: int = 0,
    min_valid_fraction: float = 0.8,
    pad_partial: bool = True
) -> List[Patch]:
    """
    Divide a DEM into patches.
    
    Args:
        dem: 2D numpy array of elevation values
        patch_size: Size of each patch (square)
        overlap: Number of pixels to overlap between patches
        min_valid_fraction: Minimum fraction of non-NaN pixels required
        pad_partial: If True, pad edge patches to full size; if False, skip them
        
    Returns:
        List of Patch objects
    """
    height, width = dem.shape
    stride = patch_size - overlap
    
    patches = []
    patch_id = 0
    
    # Calculate number of patches in each dimension
    n_rows = (height - overlap) // stride
    n_cols = (width - overlap) // stride
    
    # Handle partial patches at edges
    if pad_partial:
        if (height - overlap) % stride > 0:
            n_rows += 1
        if (width - overlap) % stride > 0:
            n_cols += 1
    
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * stride
            x_start = col * stride
            
            y_end = min(y_start + patch_size, height)
            x_end = min(x_start + patch_size, width)
            
            # Extract patch
            patch_data = dem[y_start:y_end, x_start:x_end]
            
            # Check validity (non-NaN fraction)
            valid_fraction = 1.0 - (np.isnan(patch_data).sum() / patch_data.size)
            if valid_fraction < min_valid_fraction:
                continue
            
            # Pad if necessary
            if pad_partial and patch_data.shape != (patch_size, patch_size):
                padded = np.full((patch_size, patch_size), np.nan)
                padded[:patch_data.shape[0], :patch_data.shape[1]] = patch_data
                # Fill NaN with mean of valid values
                patch_mean = np.nanmean(patch_data)
                padded = np.nan_to_num(padded, nan=float(patch_mean))
                patch_data = padded
            
            patches.append(Patch(
                id=patch_id,
                data=patch_data,
                row=row,
                col=col,
                y_start=y_start,
                x_start=x_start
            ))
            patch_id += 1
    
    return patches


def get_patch_at_coords(
    patches: List[Patch],
    y: int,
    x: int,
    patch_size: int
) -> Optional[Patch]:
    """
    Find the patch containing given pixel coordinates.
    
    Args:
        patches: List of patches
        y: Y pixel coordinate
        x: X pixel coordinate
        patch_size: Size of patches
        
    Returns:
        Patch containing the coordinates, or None
    """
    for patch in patches:
        if (patch.y_start <= y < patch.y_start + patch_size and
            patch.x_start <= x < patch.x_start + patch_size):
            return patch
    return None


def patches_to_metadata(patches: List[Patch]) -> Dict[int, Dict[str, Any]]:
    """
    Convert patches to metadata dictionary for storage.
    
    Args:
        patches: List of Patch objects
        
    Returns:
        Dictionary mapping patch_id to metadata
    """
    return {
        p.id: {
            'row': p.row,
            'col': p.col,
            'y_start': p.y_start,
            'x_start': p.x_start,
            'center': p.center,
            'shape': p.data.shape
        }
        for p in patches
    }


def reconstruct_from_patches(
    patches: List[Patch],
    original_shape: Tuple[int, int],
    patch_size: int,
    overlap: int = 0
) -> np.ndarray:
    """
    Reconstruct DEM from patches (for verification).
    
    Uses averaging where patches overlap.
    
    Args:
        patches: List of Patch objects
        original_shape: Shape of original DEM
        patch_size: Size of patches
        overlap: Overlap between patches
        
    Returns:
        Reconstructed DEM array
    """
    result = np.zeros(original_shape)
    count = np.zeros(original_shape)
    
    for patch in patches:
        y_end = min(patch.y_start + patch_size, original_shape[0])
        x_end = min(patch.x_start + patch_size, original_shape[1])
        
        h = y_end - patch.y_start
        w = x_end - patch.x_start
        
        result[patch.y_start:y_end, patch.x_start:x_end] += patch.data[:h, :w]
        count[patch.y_start:y_end, patch.x_start:x_end] += 1
    
    # Average overlapping regions
    count[count == 0] = 1  # Avoid division by zero
    result /= count
    
    return result


def get_tiling_info(
    dem_shape: Tuple[int, int],
    patch_size: int,
    overlap: int = 0
) -> Dict[str, Any]:
    """
    Get information about tiling configuration.
    
    Args:
        dem_shape: Shape of DEM (height, width)
        patch_size: Size of patches
        overlap: Overlap between patches
        
    Returns:
        Dictionary with tiling statistics
    """
    height, width = dem_shape
    stride = patch_size - overlap
    
    n_rows = (height - overlap) // stride
    n_cols = (width - overlap) // stride
    
    # Account for partial patches
    if (height - overlap) % stride > 0:
        n_rows += 1
    if (width - overlap) % stride > 0:
        n_cols += 1
    
    return {
        'dem_shape': dem_shape,
        'patch_size': patch_size,
        'overlap': overlap,
        'stride': stride,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'total_patches': n_rows * n_cols,
        'coverage': (n_rows * n_cols * patch_size * patch_size) / (height * width)
    }

