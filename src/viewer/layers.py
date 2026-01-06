"""
Layer Generation Functions

Create napari layer data for visualization overlays:
- Box outlines around similar tiles
- Heatmap showing similarity intensity
- Fade mask to dim non-matching tiles
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any


def create_box_layer_data(
    results: List[Tuple[int, float, Dict]],
    query_id: Optional[int],
    patch_size: int
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Create box overlay data for napari Shapes layer.
    
    Args:
        results: List of (patch_id, distance, metadata) tuples
        query_id: ID of the query patch (highlighted differently)
        patch_size: Size of patches
        
    Returns:
        Tuple of (box_data, colors) for napari shapes layer
    """
    boxes = []
    colors = []
    
    # Find max distance for color scaling
    if results:
        max_dist = max(r[1] for r in results) if results else 1.0
        max_dist = max(max_dist, 0.001)  # Avoid division by zero
    else:
        max_dist = 1.0
    
    for pid, dist, meta in results:
        y = meta.get('y_start', 0)
        x = meta.get('x_start', 0)
        ps = meta.get('patch_size', patch_size)
        
        # Create rectangle corners (y, x format for napari)
        # Rectangle: [top-left, top-right, bottom-right, bottom-left]
        box = np.array([
            [y, x],
            [y, x + ps],
            [y + ps, x + ps],
            [y + ps, x]
        ])
        boxes.append(box)
        
        # Color based on whether it's query or result
        if pid == query_id:
            # Query patch: bright green
            colors.append('#00ff00')
        else:
            # Result patches: orange gradient based on distance
            # Closer = more saturated orange, farther = lighter
            intensity = 1.0 - (dist / max_dist)
            r = 255
            g = int(100 + 100 * (1 - intensity))
            b = int(50 * (1 - intensity))
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    return boxes, colors


def create_heatmap_layer(
    results: List[Tuple[int, float, Dict]],
    dem_shape: Tuple[int, int],
    patch_size: int,
    threshold: float
) -> np.ndarray:
    """
    Create heatmap layer showing similarity as color intensity.
    
    Args:
        results: List of (patch_id, distance, metadata) tuples
        dem_shape: Shape of the DEM (height, width)
        patch_size: Size of patches
        threshold: Maximum distance threshold
        
    Returns:
        Labels array where each patch has a unique ID for coloring
    """
    labels = np.zeros(dem_shape, dtype=np.int32)
    
    if not results:
        return labels
    
    # Find min/max distance for normalization
    min_dist = min(r[1] for r in results)
    max_dist = max(r[1] for r in results)
    dist_range = max(max_dist - min_dist, 0.001)
    
    # Assign label IDs based on similarity (higher = more similar)
    # Use 1-100 range for label values (0 is background)
    for pid, dist, meta in results:
        if dist > threshold:
            continue
            
        y = meta.get('y_start', 0)
        x = meta.get('x_start', 0)
        ps = meta.get('patch_size', patch_size)
        
        # Normalize distance to 1-100 range (inverted: closer = higher value)
        similarity = 1.0 - ((dist - min_dist) / dist_range)
        label_value = int(1 + similarity * 99)
        
        # Fill the patch region
        y_end = min(y + ps, dem_shape[0])
        x_end = min(x + ps, dem_shape[1])
        labels[y:y_end, x:x_end] = label_value
    
    return labels


def create_fade_mask_layer(
    results: List[Tuple[int, float, Dict]],
    dem_shape: Tuple[int, int],
    patch_size: int
) -> np.ndarray:
    """
    Create fade mask to dim non-matching tiles.
    
    Similar tiles remain bright (value 0), non-matching tiles are darkened.
    
    Args:
        results: List of (patch_id, distance, metadata) tuples
        dem_shape: Shape of the DEM (height, width)
        patch_size: Size of patches
        
    Returns:
        Float array with 0 for matching tiles, negative for non-matching
    """
    # Start with dark mask everywhere
    mask = np.full(dem_shape, -0.5, dtype=np.float32)
    
    if not results:
        return np.zeros(dem_shape, dtype=np.float32)
    
    # Clear mask for matching tiles
    for pid, dist, meta in results:
        y = meta.get('y_start', 0)
        x = meta.get('x_start', 0)
        ps = meta.get('patch_size', patch_size)
        
        y_end = min(y + ps, dem_shape[0])
        x_end = min(x + ps, dem_shape[1])
        mask[y:y_end, x:x_end] = 0.0
    
    return mask


def create_highlight_for_patch(
    patch_id: int,
    metadata: Dict[int, Dict],
    patch_size: int
) -> np.ndarray:
    """
    Create a single highlight box for a specific patch.
    
    Args:
        patch_id: ID of patch to highlight
        metadata: All patch metadata
        patch_size: Default patch size
        
    Returns:
        Box coordinates as numpy array
    """
    if patch_id not in metadata:
        return np.array([])
    
    meta = metadata[patch_id]
    y = meta.get('y_start', 0)
    x = meta.get('x_start', 0)
    ps = meta.get('patch_size', patch_size)
    
    return np.array([
        [y, x],
        [y, x + ps],
        [y + ps, x + ps],
        [y + ps, x]
    ])


def get_similarity_colormap(n_colors: int = 100) -> Dict[int, np.ndarray]:
    """
    Create a colormap for similarity labels.
    
    Uses a blue (low similarity) to red (high similarity) gradient.
    
    Args:
        n_colors: Number of color levels
        
    Returns:
        Dictionary mapping label values to RGBA colors
    """
    colormap = {0: np.array([0, 0, 0, 0])}  # Background is transparent
    
    for i in range(1, n_colors + 1):
        t = (i - 1) / (n_colors - 1)  # 0 to 1
        
        # Blue to red gradient
        r = int(255 * t)
        g = int(100 * (1 - abs(2 * t - 1)))  # Peak at middle
        b = int(255 * (1 - t))
        a = 180  # Semi-transparent
        
        colormap[i] = np.array([r, g, b, a])
    
    return colormap

