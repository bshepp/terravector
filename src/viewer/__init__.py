"""
Napari Terrain Similarity Viewer

Interactive desktop application for exploring terrain patch similarity
with visualization overlays (boxes, heatmaps, fade effects).
"""

from .app import TerrainViewer, launch_viewer

__all__ = ['TerrainViewer', 'launch_viewer']

