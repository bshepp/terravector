#!/usr/bin/env python3
"""
TerraVector Napari Viewer

Interactive desktop application for exploring terrain patch similarity.
Provides visualization overlays for highlighting similar tiles.

Usage:
    python viewer.py [dem_path] [index_path]
    
Examples:
    # Launch empty viewer
    python viewer.py
    
    # Load DEM only
    python viewer.py data/terrain.npy
    
    # Load DEM and index
    python viewer.py data/terrain.npy data/terrain.idx
    
    # Load just an index (DEM path stored in metadata)
    python viewer.py --index data/terrain.idx
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="TerraVector Napari Viewer - Interactive terrain similarity exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualization Modes:
  Box Outlines     Colored rectangles around similar tiles
                   (Green = query, Orange gradient = results by distance)
  
  Heatmap          Similarity scores as color intensity
                   (Blue = low similarity, Red = high similarity)
  
  Fade             Dim non-matching tiles
                   (Similar tiles remain bright)

Controls:
  - Click on terrain to query for similar patches
  - Use threshold slider to filter by distance
  - Toggle visualization modes with checkboxes
  - Pan/zoom with mouse (napari standard controls)

Keyboard Shortcuts (napari defaults):
  Space + drag     Pan
  Scroll           Zoom
  Home             Reset view
  F                Toggle fullscreen
        """
    )
    
    parser.add_argument(
        'dem_path',
        nargs='?',
        help='Path to DEM file (.npy or .tif)'
    )
    
    parser.add_argument(
        'index_path',
        nargs='?',
        help='Path to index file (.idx)'
    )
    
    parser.add_argument(
        '--index', '-i',
        dest='index_only',
        help='Load index only (DEM path from metadata)'
    )
    
    parser.add_argument(
        '--patch-size', '-p',
        type=int,
        default=64,
        help='Patch size (default: 64, overridden by index metadata)'
    )
    
    args = parser.parse_args()
    
    # Determine paths
    dem_path = args.dem_path
    index_path = args.index_path or args.index_only
    
    # Validate paths exist
    if dem_path and not Path(dem_path).exists():
        print(f"Error: DEM file not found: {dem_path}")
        sys.exit(1)
    
    if index_path:
        idx_path = Path(index_path)
        if not idx_path.exists() and not idx_path.with_suffix('.idx').exists():
            print(f"Error: Index file not found: {index_path}")
            sys.exit(1)
    
    # Check napari is available
    try:
        import napari
    except ImportError:
        print("Error: napari is not installed.")
        print("Install with: pip install napari[all]")
        sys.exit(1)
    
    # Launch viewer
    from src.viewer import launch_viewer
    
    print("=" * 50)
    print("TerraVector Napari Viewer")
    print("=" * 50)
    
    if dem_path:
        print(f"DEM: {dem_path}")
    if index_path:
        print(f"Index: {index_path}")
    if not dem_path and not index_path:
        print("No files specified - launching empty viewer")
        print("Use File menu or load programmatically")
    
    print()
    print("Controls:")
    print("  - Click terrain to query similar patches")
    print("  - Use controls panel on right")
    print("  - Pan: Space + drag | Zoom: scroll")
    print("=" * 50)
    
    launch_viewer(
        dem_path=dem_path,
        index_path=index_path,
        patch_size=args.patch_size
    )


if __name__ == '__main__':
    main()

