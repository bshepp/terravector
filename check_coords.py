#!/usr/bin/env python
"""
Check whether a WGS84 (lat, lon) point falls inside a terravector corpus,
and report which tile (and patch, if available) contains it.

Intended as a quick sanity check for users wiring up the REST API or
QGIS plugin — answers "is my point of interest actually indexed?" without
needing to load the full HNSW index.

Usage:
    python check_coords.py --lat 40.0279 --lon -82.4590
    python check_coords.py --lat 40.0279 --lon -82.4590 \
        --corpus data/licking_county/licking_county.corpus.json

The corpus argument defaults to data/licking_county/licking_county.corpus.json
(the path produced by build_county_index.py).
"""

import argparse
import json
import sys
from pathlib import Path

from pyproj import Transformer

DEFAULT_CORPUS = Path("data/licking_county/licking_county.corpus.json")


def load_corpus_bounds(corpus_path: Path) -> dict:
    with open(corpus_path) as f:
        corpus = json.load(f)
    county = corpus["county_metadata"]
    return {
        "bounds": county["county_bounds"],
        "tile_size_ft": county["tile_size_ft"],
        "resolution_ft": county["resolution_ft"],
        "grid_rows": county["grid_rows"],
        "grid_cols": county["grid_cols"],
        "crs": county.get("crs", "EPSG:3735"),
        "tile_info": corpus.get("tile_info", {}),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
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

    info = load_corpus_bounds(args.corpus)
    bounds = info["bounds"]

    transformer = Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True)
    x, y = transformer.transform(args.lon, args.lat)

    print(f"Input:        lat={args.lat}, lon={args.lon} (WGS84)")
    print(f"Projected:    X={x:,.1f}, Y={y:,.1f} ({info['crs']}, feet)")
    print(
        f"Corpus bbox:  X=[{bounds['x_min']:,.0f}, {bounds['x_max']:,.0f}], "
        f"Y=[{bounds['y_min']:,.0f}, {bounds['y_max']:,.0f}]"
    )

    inside = (
        bounds["x_min"] <= x < bounds["x_max"]
        and bounds["y_min"] < y <= bounds["y_max"]
    )
    if not inside:
        print("\nResult:       OUTSIDE corpus coverage")
        print(
            f"  X offset from min: {x - bounds['x_min']:+,.0f} ft "
            f"(corpus width {bounds['x_max'] - bounds['x_min']:,.0f} ft)"
        )
        print(
            f"  Y offset from min: {y - bounds['y_min']:+,.0f} ft "
            f"(corpus height {bounds['y_max'] - bounds['y_min']:,.0f} ft)"
        )
        return 1

    col = int((x - bounds["x_min"]) / info["tile_size_ft"])
    row = int((bounds["y_max"] - y) / info["tile_size_ft"])
    col = min(col, info["grid_cols"] - 1)
    row = min(row, info["grid_rows"] - 1)
    tile_id = f"R{row:02d}_C{col:02d}"
    indexed = tile_id in info["tile_info"]

    print("\nResult:       INSIDE corpus coverage")
    print(f"  Tile:        {tile_id} ({'indexed' if indexed else 'NOT indexed — tile may be empty/no-data'})")
    if indexed:
        ti = info["tile_info"][tile_id]
        print(f"  Patches in tile: {ti.get('n_patches', 'unknown')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
