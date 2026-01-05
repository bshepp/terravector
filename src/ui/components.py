"""
UI Components

Reusable visualization components for the Gradio UI.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization import hillshade


def create_hillshade_image(
    dem: np.ndarray,
    patch_grid: bool = False,
    patch_size: int = 64,
    highlight_patch: Optional[int] = None,
    metadata: Optional[Dict[int, Dict]] = None
) -> np.ndarray:
    """
    Create a hillshade visualization of a DEM.
    
    Args:
        dem: 2D elevation array
        patch_grid: Overlay patch grid lines
        patch_size: Size of patches for grid
        highlight_patch: Patch ID to highlight
        metadata: Patch metadata for highlighting
        
    Returns:
        RGB image as uint8 array
    """
    # Generate hillshade
    hs = hillshade(dem)
    
    # Convert to RGB
    rgb = np.stack([hs, hs, hs], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    
    # Draw patch grid if requested
    if patch_grid:
        rgb = draw_patch_grid(rgb, patch_size)
    
    # Highlight specific patch
    if highlight_patch is not None and metadata is not None:
        if highlight_patch in metadata:
            meta = metadata[highlight_patch]
            y, x = meta['y_start'], meta['x_start']
            ps = meta.get('patch_size', patch_size)
            rgb = draw_highlight_box(rgb, y, x, ps, color=(255, 100, 50))
    
    return rgb


def draw_patch_grid(
    image: np.ndarray,
    patch_size: int,
    color: Tuple[int, int, int] = (60, 60, 80),
    alpha: float = 0.3
) -> np.ndarray:
    """Draw grid lines on image showing patch boundaries."""
    h, w = image.shape[:2]
    result = image.copy()
    
    # Vertical lines
    for x in range(0, w, patch_size):
        result[:, x] = (
            result[:, x].astype(float) * (1 - alpha) + 
            np.array(color) * alpha
        ).astype(np.uint8)
    
    # Horizontal lines
    for y in range(0, h, patch_size):
        result[y, :] = (
            result[y, :].astype(float) * (1 - alpha) + 
            np.array(color) * alpha
        ).astype(np.uint8)
    
    return result


def draw_highlight_box(
    image: np.ndarray,
    y: int,
    x: int,
    size: int,
    color: Tuple[int, int, int] = (255, 100, 50),
    thickness: int = 2
) -> np.ndarray:
    """Draw a highlight box around a patch."""
    result = image.copy()
    h, w = image.shape[:2]
    
    y_end = min(y + size, h)
    x_end = min(x + size, w)
    
    # Top and bottom edges
    for t in range(thickness):
        if y + t < h:
            result[y + t, x:x_end] = color
        if y_end - 1 - t >= 0:
            result[y_end - 1 - t, x:x_end] = color
    
    # Left and right edges
    for t in range(thickness):
        if x + t < w:
            result[y:y_end, x + t] = color
        if x_end - 1 - t >= 0:
            result[y:y_end, x_end - 1 - t] = color
    
    return result


def create_results_gallery(
    dem: np.ndarray,
    results: List[Tuple[int, float, Dict]],
    patch_size: int,
    query_id: Optional[int] = None,
    query_meta: Optional[Dict] = None
) -> List[Tuple[np.ndarray, str]]:
    """
    Create gallery images for query results.
    
    Args:
        dem: Source DEM
        results: List of (patch_id, distance, metadata) tuples
        patch_size: Size of patches
        query_id: ID of query patch (to show first)
        query_meta: Metadata of query patch
        
    Returns:
        List of (image, caption) tuples for Gradio gallery
    """
    gallery = []
    
    # Add query patch first if provided
    if query_id is not None and query_meta is not None:
        patch_img = extract_patch_image(dem, query_meta, patch_size)
        gallery.append((patch_img, f"Query #{query_id}"))
    
    # Add result patches
    for pid, dist, meta in results:
        # Skip query patch in results
        if query_id is not None and pid == query_id:
            continue
        
        patch_img = extract_patch_image(dem, meta, patch_size)
        caption = f"#{pid} (d={dist:.3f})"
        gallery.append((patch_img, caption))
    
    return gallery


def extract_patch_image(
    dem: np.ndarray,
    meta: Dict[str, Any],
    patch_size: int
) -> np.ndarray:
    """Extract a patch from DEM and convert to hillshade image."""
    y, x = meta['y_start'], meta['x_start']
    ps = meta.get('patch_size', patch_size)
    
    y_end = min(y + ps, dem.shape[0])
    x_end = min(x + ps, dem.shape[1])
    
    patch = dem[y:y_end, x:x_end]
    
    # Pad if necessary
    if patch.shape != (ps, ps):
        padded = np.full((ps, ps), np.nanmean(patch))
        padded[:patch.shape[0], :patch.shape[1]] = patch
        patch = padded
    
    # Convert to hillshade RGB
    hs = hillshade(patch)
    rgb = np.stack([hs, hs, hs], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def format_status(
    dem_info: Optional[Dict] = None,
    index_stats: Optional[Dict] = None,
    message: str = ""
) -> str:
    """Format status string for display."""
    parts = []
    
    if dem_info:
        shape = dem_info.get('shape', (0, 0))
        parts.append(f"DEM: {shape[0]}×{shape[1]}")
    
    if index_stats:
        parts.append(f"Patches: {index_stats.get('n_patches', 0)}")
        parts.append(f"Dim: {index_stats.get('embedding_dim', 0)}")
    
    status = " │ ".join(parts) if parts else "Ready"
    
    if message:
        status = f"{message}\n{status}"
    
    return status


def create_placeholder_image(
    width: int = 512,
    height: int = 512,
    text: str = "Load a DEM to begin"
) -> np.ndarray:
    """Create a placeholder image with text."""
    # Dark background
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    
    # Try to add text using PIL
    try:
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=(100, 100, 120))
        img = np.array(pil_img)
    except Exception:
        pass
    
    return img


def coords_from_click(
    evt_data: Dict,
    dem_shape: Tuple[int, int],
    display_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Convert click coordinates from display space to DEM space.
    
    Args:
        evt_data: Gradio event data with x, y coordinates
        dem_shape: (height, width) of DEM
        display_size: (height, width) of displayed image
        
    Returns:
        (y, x) in DEM coordinates
    """
    # Get click coordinates
    click_x = evt_data.get('x', 0)
    click_y = evt_data.get('y', 0)
    
    # Scale to DEM coordinates
    scale_y = dem_shape[0] / display_size[0]
    scale_x = dem_shape[1] / display_size[1]
    
    dem_y = int(click_y * scale_y)
    dem_x = int(click_x * scale_x)
    
    # Clamp to valid range
    dem_y = max(0, min(dem_y, dem_shape[0] - 1))
    dem_x = max(0, min(dem_x, dem_shape[1] - 1))
    
    return dem_y, dem_x

