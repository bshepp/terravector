"""
Main Napari Viewer Application

TerrainViewer class that manages the napari viewer, layers, and interactions.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import napari
from napari.layers import Image, Shapes, Labels

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io import load_dem, get_dem_info
from src.index import TerrainIndex
from src.visualization import hillshade
from src.viewer.layers import (
    create_box_layer_data,
    create_heatmap_layer,
    create_fade_mask_layer
)
from src.viewer.widgets import create_control_widget


class TerrainViewer:
    """
    Interactive terrain similarity viewer using napari.
    
    Provides three visualization modes:
    - Box overlay: Colored rectangles around similar tiles
    - Heatmap: Similarity scores as color intensity
    - Fade: Dim non-matching tiles
    """
    
    def __init__(
        self,
        dem: Optional[np.ndarray] = None,
        index: Optional[TerrainIndex] = None,
        patch_size: int = 64
    ):
        """
        Initialize the terrain viewer.
        
        Args:
            dem: 2D elevation array
            index: Pre-built terrain index
            patch_size: Size of terrain patches
        """
        self.dem = dem
        self.index = index
        self.patch_size = patch_size
        
        # State
        self.query_patch_id: Optional[int] = None
        self.results: List[Tuple[int, float, Dict]] = []
        self.threshold: float = 1.0
        self.max_results: int = 50
        
        # Visualization modes
        self.show_boxes: bool = True
        self.show_heatmap: bool = False
        self.show_fade: bool = False
        
        # Napari viewer and layers
        self.viewer: Optional[napari.Viewer] = None
        self.hillshade_layer: Optional[Image] = None
        self.box_layer: Optional[Shapes] = None
        self.heatmap_layer: Optional[Labels] = None
        self.fade_layer: Optional[Image] = None
        
    def load_dem(self, path: str) -> None:
        """Load a DEM from file."""
        self.dem = load_dem(path)
        self.dem_info = get_dem_info(self.dem)
        
        # Infer patch size from index if available
        if self.index is not None:
            sample_meta = next(iter(self.index.metadata.values()), {})
            self.patch_size = sample_meta.get('patch_size', 64)
    
    def load_index(self, path: str) -> None:
        """Load an index from file."""
        self.index = TerrainIndex.load(path)
        
        # Try to get patch size from metadata
        sample_meta = next(iter(self.index.metadata.values()), {})
        self.patch_size = sample_meta.get('patch_size', 64)
        
        # Try to load source DEM if referenced
        dem_path = sample_meta.get('dem_path')
        if dem_path and Path(dem_path).exists() and self.dem is None:
            self.load_dem(dem_path)
    
    def launch(self) -> napari.Viewer:
        """Launch the napari viewer."""
        self.viewer = napari.Viewer(title="TerraVector - Terrain Similarity")
        
        # Add hillshade layer if DEM is loaded
        if self.dem is not None:
            self._add_hillshade_layer()
        
        # Add empty overlay layers
        self._init_overlay_layers()
        
        # Add control widget
        control_widget = create_control_widget(self)
        self.viewer.window.add_dock_widget(
            control_widget,
            name="Similarity Controls",
            area="right"
        )
        
        # Connect mouse callback for click-to-query
        self._connect_mouse_callback()
        
        return self.viewer
    
    def _add_hillshade_layer(self) -> None:
        """Add the hillshade base layer."""
        hs = hillshade(self.dem)
        self.hillshade_layer = self.viewer.add_image(
            hs,
            name="Terrain",
            colormap="gray",
            blending="translucent"
        )
    
    def _init_overlay_layers(self) -> None:
        """Initialize empty overlay layers."""
        # Box overlay (shapes layer)
        self.box_layer = self.viewer.add_shapes(
            data=None,
            name="Similar Tiles (Boxes)",
            edge_width=2,
            face_color='transparent',
            edge_color='orange',
            visible=self.show_boxes
        )
        
        # Heatmap overlay (labels layer for coloring tiles)
        if self.dem is not None:
            empty_labels = np.zeros(self.dem.shape, dtype=np.int32)
            self.heatmap_layer = self.viewer.add_labels(
                empty_labels,
                name="Similarity Heatmap",
                opacity=0.5,
                visible=self.show_heatmap
            )
        
        # Fade mask (image layer with dark overlay)
        if self.dem is not None:
            empty_mask = np.zeros(self.dem.shape, dtype=np.float32)
            self.fade_layer = self.viewer.add_image(
                empty_mask,
                name="Fade Mask",
                colormap="gray",
                blending="additive",
                opacity=0.6,
                visible=self.show_fade
            )
    
    def _connect_mouse_callback(self) -> None:
        """Connect mouse click handler for querying."""
        @self.hillshade_layer.mouse_drag_callbacks.append
        def on_click(layer, event):
            if event.type == 'mouse_press':
                # Get coordinates in data space
                coords = layer.world_to_data(event.position)
                if coords is not None and len(coords) >= 2:
                    y, x = int(coords[0]), int(coords[1])
                    self._query_at_coords(y, x)
    
    def _query_at_coords(self, y: int, x: int) -> None:
        """Query for similar patches at given coordinates."""
        if self.index is None:
            print("No index loaded")
            return
        
        # Find patch containing these coordinates
        for pid, meta in self.index.metadata.items():
            ps = meta.get('patch_size', self.patch_size)
            if (meta['y_start'] <= y < meta['y_start'] + ps and
                meta['x_start'] <= x < meta['x_start'] + ps):
                self.query_by_id(pid)
                return
        
        print(f"No patch found at ({y}, {x})")
    
    def query_by_id(self, patch_id: int) -> None:
        """Query for similar patches by ID."""
        if self.index is None:
            print("No index loaded")
            return
        
        if patch_id not in self.index.metadata:
            print(f"Patch {patch_id} not in index")
            return
        
        self.query_patch_id = patch_id
        self.results = self.index.query_by_id(patch_id, k=self.max_results)
        
        print(f"Query patch #{patch_id}: found {len(self.results)} similar")
        self.update_overlays()
    
    def update_overlays(self) -> None:
        """Update all overlay layers based on current results."""
        if not self.results or self.dem is None:
            return
        
        # Filter by threshold
        filtered = [(pid, dist, meta) for pid, dist, meta in self.results
                    if dist <= self.threshold]
        
        # Update box layer
        if self.box_layer is not None:
            box_data, colors = create_box_layer_data(
                filtered,
                self.query_patch_id,
                self.patch_size
            )
            self.box_layer.data = box_data
            self.box_layer.edge_color = colors
        
        # Update heatmap layer
        if self.heatmap_layer is not None:
            heatmap = create_heatmap_layer(
                filtered,
                self.dem.shape,
                self.patch_size,
                self.threshold
            )
            self.heatmap_layer.data = heatmap
        
        # Update fade layer
        if self.fade_layer is not None:
            fade = create_fade_mask_layer(
                filtered,
                self.dem.shape,
                self.patch_size
            )
            self.fade_layer.data = fade
    
    def set_threshold(self, value: float) -> None:
        """Set the distance threshold and update overlays."""
        self.threshold = value
        self.update_overlays()
    
    def set_max_results(self, value: int) -> None:
        """Set max results and re-query if needed."""
        self.max_results = value
        if self.query_patch_id is not None:
            self.query_by_id(self.query_patch_id)
    
    def toggle_boxes(self, visible: bool) -> None:
        """Toggle box overlay visibility."""
        self.show_boxes = visible
        if self.box_layer is not None:
            self.box_layer.visible = visible
    
    def toggle_heatmap(self, visible: bool) -> None:
        """Toggle heatmap overlay visibility."""
        self.show_heatmap = visible
        if self.heatmap_layer is not None:
            self.heatmap_layer.visible = visible
    
    def toggle_fade(self, visible: bool) -> None:
        """Toggle fade overlay visibility."""
        self.show_fade = visible
        if self.fade_layer is not None:
            self.fade_layer.visible = visible
    
    def run(self) -> None:
        """Run the viewer (blocking)."""
        if self.viewer is None:
            self.launch()
        napari.run()


def launch_viewer(
    dem_path: Optional[str] = None,
    index_path: Optional[str] = None,
    patch_size: int = 64
) -> TerrainViewer:
    """
    Launch the terrain viewer.
    
    Args:
        dem_path: Path to DEM file (.npy or .tif)
        index_path: Path to index file (.idx)
        patch_size: Patch size (used if not in index metadata)
        
    Returns:
        TerrainViewer instance
    """
    viewer = TerrainViewer(patch_size=patch_size)
    
    if index_path:
        viewer.load_index(index_path)
    
    if dem_path:
        viewer.load_dem(dem_path)
    
    viewer.launch()
    viewer.run()
    
    return viewer

