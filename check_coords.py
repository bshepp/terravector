#!/usr/bin/env python
"""
Check whether a WGS84 (lat, lon) point falls inside a terravector corpus,
and report which tile (and pixel) contains it.

Intended as a quick sanity check for users wiring up the REST API or
QGIS plugin: answers "is my point of interest actually indexed?" without
needing to load the full HNSW index. Uses the same CoordBridge as the API,
so its answer matches what /query/by-coords would resolve (the bridge
bounds lookups by the tile-grid extent, which overshoots the nominal
county bounds on the south and east edges).

Usage:
    python check_coords.py --lat 40.0279 --lon -82.4590
    python check_coords.py --lat 40.0279 --lon -82.4590 \\
        --corpus data/licking_county/licking_county.corpus.json

The corpus argument defaults to data/licking_county/licking_county.corpus.json
(the path produced by build_county_index.py).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.coords import CoordBridge  # noqa: E402

DEFAULT_CORPUS = Path("data/licking_county/licking_county.corpus.json")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Check whether a WGS84 point falls inside a terravector corpus."
    )
    p.add_argument("--lat", type=float, required=True, help="WGS84 latitude")
    p.add_argument("--lon", type=float, required=True, help="WGS84 longitude")
    p.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help=f"Path to corpus.json (default: {DEFAULT_CORPUS})",
    )
    args = p.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus file not found: {args.corpus}", file=sys.stderr)
        print("  Run build_county_index.py first, or pass --corpus.", file=sys.stderr)
        return 2

    with open(args.corpus) as f:
        corpus = json.load(f)

    # No index metadata: tile/pixel resolution works, patch-id resolution does not.
    bridge = CoordBridge(corpus, index_metadata={}, patch_size=corpus.get("patch_size", 64))
    x, y = bridge.wgs84_to_state_plane(args.lon, args.lat)

    print(f"Input:        lat={args.lat}, lon={args.lon} (WGS84)")
    print(f"Projected:    X={x:,.1f}, Y={y:,.1f} ({bridge.crs}, feet)")
    print(
        f"County bbox:  X=[{bridge.x_min:,.0f}, {bridge.x_max:,.0f}], "
        f"Y=[{bridge.y_min:,.0f}, {bridge.y_max:,.0f}]"
    )
    print(
        f"Tile grid:    X=[{bridge.x_min:,.0f}, {bridge.grid_x_max:,.0f}], "
        f"Y=[{bridge.grid_y_min:,.0f}, {bridge.y_max:,.0f}] "
        f"({bridge.grid_rows} rows x {bridge.grid_cols} cols of {bridge.tile_size_ft:,.0f} ft)"
    )

    hit = bridge.state_plane_to_tile_pixel(x, y)
    if hit is None:
        in_grid = (
            bridge.x_min <= x < bridge.grid_x_max
            and bridge.grid_y_min < y <= bridge.y_max
        )
        if in_grid:
            print("\nResult:       INSIDE the tile grid, but the tile there has no data")
        else:
            print("\nResult:       OUTSIDE corpus coverage")
            print(
                f"  X offset from grid min: {x - bridge.x_min:+,.0f} ft "
                f"(grid width {bridge.grid_x_max - bridge.x_min:,.0f} ft)"
            )
            print(
                f"  Y offset from grid min: {y - bridge.grid_y_min:+,.0f} ft "
                f"(grid height {bridge.y_max - bridge.grid_y_min:,.0f} ft)"
            )
        return 1

    tile_id, py, px = hit
    ti = bridge.tile_info[tile_id]
    print("\nResult:       INSIDE corpus coverage")
    print(f"  Tile:        {tile_id} (pixel row {py}, col {px})")
    print(f"  Patch cell:  row {py // bridge.patch_size}, col {px // bridge.patch_size} "
          f"(patch_size {bridge.patch_size})")
    print(f"  Patches in tile: {ti.get('n_patches', 'unknown')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
