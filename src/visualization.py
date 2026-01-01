"""
Visualization Module

Display terrain patches and similarity results as image grids.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


def hillshade(dem: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """
    Generate hillshade from DEM.
    
    Args:
        dem: 2D elevation array
        azimuth: Light source azimuth in degrees
        altitude: Light source altitude in degrees
        
    Returns:
        Hillshaded array (0-1 range)
    """
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    return ls.hillshade(dem, vert_exag=1)


def visualize_patches(
    patches: List[np.ndarray],
    labels: Optional[List[str]] = None,
    distances: Optional[List[float]] = None,
    title: str = "Terrain Patches",
    output_path: Optional[str] = None,
    use_hillshade: bool = True,
    max_cols: int = 5,
    figsize_per_patch: float = 2.5
) -> None:
    """
    Visualize a list of patches in a grid.
    
    Args:
        patches: List of 2D arrays
        labels: Optional labels for each patch
        distances: Optional distances (shown in label)
        title: Figure title
        output_path: If provided, save to file
        use_hillshade: Apply hillshading
        max_cols: Maximum columns in grid
        figsize_per_patch: Size per patch in inches
    """
    n = len(patches)
    if n == 0:
        print("No patches to visualize")
        return
    
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_patch, n_rows * figsize_per_patch),
        squeeze=False
    )
    
    for idx, patch in enumerate(patches):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if use_hillshade:
            display = hillshade(patch)
            ax.imshow(display, cmap='gray')
        else:
            ax.imshow(patch, cmap='terrain')
        
        ax.axis('off')
        
        # Build label
        if labels is not None and idx < len(labels):
            label = labels[idx]
        else:
            label = f"Patch {idx}"
        
        if distances is not None and idx < len(distances):
            label += f"\nd={distances[idx]:.3f}"
        
        ax.set_title(label, fontsize=9)
    
    # Hide empty subplots
    for idx in range(n, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_query_results(
    query_patch: np.ndarray,
    similar_patches: List[np.ndarray],
    query_label: str = "Query",
    similar_ids: Optional[List[int]] = None,
    distances: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    use_hillshade: bool = True
) -> None:
    """
    Visualize a query patch and its similar matches.
    
    The query patch is shown larger on the left, with similar patches in a grid on the right.
    
    Args:
        query_patch: The query patch
        similar_patches: List of similar patches
        query_label: Label for query patch
        similar_ids: IDs of similar patches
        distances: Distances to similar patches
        output_path: If provided, save to file
        use_hillshade: Apply hillshading
    """
    n_similar = len(similar_patches)
    
    if n_similar == 0:
        # Just show query
        visualize_patches(
            [query_patch],
            labels=[query_label],
            title="Query Patch",
            output_path=output_path,
            use_hillshade=use_hillshade
        )
        return
    
    # Create figure with query on left, results grid on right
    fig = plt.figure(figsize=(12, 6))
    
    # Query patch (larger, left side)
    ax_query = fig.add_axes([0.02, 0.15, 0.3, 0.7])
    if use_hillshade:
        ax_query.imshow(hillshade(query_patch), cmap='gray')
    else:
        ax_query.imshow(query_patch, cmap='terrain')
    ax_query.set_title(f"{query_label}", fontsize=11, fontweight='bold')
    ax_query.axis('off')
    
    # Similar patches grid (right side)
    n_cols = min(n_similar, 4)
    n_rows = (n_similar + n_cols - 1) // n_cols
    
    # Calculate grid positions
    grid_left = 0.38
    grid_width = 0.6
    grid_height = 0.8
    cell_width = grid_width / n_cols
    cell_height = grid_height / n_rows
    
    for idx, patch in enumerate(similar_patches):
        row = idx // n_cols
        col = idx % n_cols
        
        left = grid_left + col * cell_width + 0.01
        bottom = 0.9 - (row + 1) * cell_height + 0.02
        width = cell_width - 0.02
        height = cell_height - 0.04
        
        ax = fig.add_axes([left, bottom, width, height])
        
        if use_hillshade:
            ax.imshow(hillshade(patch), cmap='gray')
        else:
            ax.imshow(patch, cmap='terrain')
        
        ax.axis('off')
        
        # Label
        label_parts = []
        if similar_ids is not None and idx < len(similar_ids):
            label_parts.append(f"#{similar_ids[idx]}")
        if distances is not None and idx < len(distances):
            label_parts.append(f"d={distances[idx]:.3f}")
        
        if label_parts:
            ax.set_title(" ".join(label_parts), fontsize=8)
    
    plt.suptitle("Similar Terrain Patches", fontsize=12, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def extract_patch_from_dem(
    dem: np.ndarray,
    y_start: int,
    x_start: int,
    patch_size: int
) -> np.ndarray:
    """
    Extract a patch from a DEM given coordinates.
    
    Args:
        dem: Full DEM array
        y_start: Y start coordinate
        x_start: X start coordinate
        patch_size: Size of patch
        
    Returns:
        Extracted patch
    """
    y_end = min(y_start + patch_size, dem.shape[0])
    x_end = min(x_start + patch_size, dem.shape[1])
    
    patch = dem[y_start:y_end, x_start:x_end]
    
    # Pad if necessary
    if patch.shape != (patch_size, patch_size):
        padded = np.full((patch_size, patch_size), np.nanmean(patch))
        padded[:patch.shape[0], :patch.shape[1]] = patch
        patch = padded
    
    return patch

