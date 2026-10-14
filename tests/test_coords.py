"""Coordinate bridge and similarity-raster placement.

The county tiles produced by RESIDUALS/tile_county.py number rows from the
SOUTH (R00 = southernmost) and store each tile's DEM with array row 0 at the
tile's SOUTHERN edge. Patch metadata (y_start) inherits that orientation.
These tests pin that convention end to end: State Plane -> tile/pixel ->
patch id -> bbox, and patch -> raster cell.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.coords import CoordBridge
from src.index import build_index

TILE_FT = 320.0  # 128 px at 2.5 ft -> 2 x 2 patches of 64 px per tile
RES = 2.5
PS = 64
ROWS, COLS = 2, 2


def _corpus():
    tile_info = {}
    for r in range(ROWS):
        for c in range(COLS):
            tile_info[f"R{r:02d}_C{c:02d}"] = {
                "tile_row": r,
                "tile_col": c,
                "tile_shape": [128, 128],
            }
    return {
        "county_metadata": {
            "grid_rows": ROWS,
            "grid_cols": COLS,
            "tile_size_ft": TILE_FT,
            "resolution_ft": RES,
            # nominal bounds stop short of the grid's north edge, as for the county
            "county_bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 640.0, "y_max": 600.0},
            "crs": "EPSG:3735",
        },
        "tile_info": tile_info,
    }


def _metadata():
    meta, pid = {}, 0
    for r in range(ROWS):
        for c in range(COLS):
            for ys in (0, 64):
                for xs in (0, 64):
                    meta[pid] = {
                        "tile_id": f"R{r:02d}_C{c:02d}",
                        "y_start": ys,
                        "x_start": xs,
                        "patch_size": PS,
                    }
                    pid += 1
    return meta


@pytest.fixture
def bridge():
    return CoordBridge(_corpus(), _metadata(), patch_size=PS)


def test_grid_extent_runs_north_from_y_min(bridge):
    assert bridge.grid_y_max == ROWS * TILE_FT
    assert bridge.grid_x_max == COLS * TILE_FT


def test_sw_corner_is_tile_r00_row0(bridge):
    tile_id, py, px = bridge.state_plane_to_tile_pixel(10.0, 10.0)
    assert tile_id == "R00_C00"
    assert (py, px) == (4, 4)


def test_rows_count_northward(bridge):
    # 630 ft north is in the second tile row, 124 px above that tile's south edge
    tile_id, py, px = bridge.state_plane_to_tile_pixel(10.0, 630.0)
    assert tile_id == "R01_C00"
    assert py == 124
    pid = bridge.state_plane_to_patch_id(10.0, 630.0)
    assert bridge.index_metadata[pid]["y_start"] == 64
    x0, y0, x1, y1 = bridge.patch_id_to_state_plane_bbox(pid)
    assert (x0, y0, x1, y1) == (0.0, 480.0, 160.0, 640.0)


def test_points_above_nominal_y_max_but_inside_grid_resolve(bridge):
    assert bridge.state_plane_to_tile_pixel(10.0, 620.0) is not None
    assert bridge.state_plane_to_tile_pixel(10.0, 640.0) is None
    assert bridge.state_plane_to_tile_pixel(10.0, -1.0) is None


def test_pixel_round_trip(bridge):
    for x, y in [(3.0, 3.0), (317.0, 319.0), (400.0, 500.0)]:
        tile_id, py, px = bridge.state_plane_to_tile_pixel(x, y)
        bx, by = bridge.tile_pixel_to_state_plane(tile_id, py, px)
        assert 0 <= x - bx < RES and 0 <= y - by < RES


def test_similarity_raster_places_north_at_row_zero(bridge):
    from src.similarity_raster import SimilarityRaster

    meta = _metadata()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(meta), 8)).astype(np.float32)
    index = build_index(emb, meta, normalize=True)
    sr = SimilarityRaster(index, bridge)

    assert sr.transform[5] == bridge.grid_y_max  # raster origin is the grid's north edge
    assert (sr.height, sr.width) == (4, 4)

    # Patch in tile R01 (north row), y_start=64 (upper half) -> raster row 0
    pid_nw = next(p for p, m in meta.items() if m["tile_id"] == "R01_C00" and m["y_start"] == 64 and m["x_start"] == 0)
    assert (sr.cell_rows[pid_nw], sr.cell_cols[pid_nw]) == (0, 0)
    # Patch in tile R00, y_start=0 (southernmost) -> last raster row
    pid_sw = next(p for p, m in meta.items() if m["tile_id"] == "R00_C00" and m["y_start"] == 0 and m["x_start"] == 0)
    assert (sr.cell_rows[pid_sw], sr.cell_cols[pid_sw]) == (3, 0)

    grid = sr.raster(pid_nw)
    assert grid[0, 0] == pytest.approx(1.0, abs=1e-5)


COUNTY_CORPUS = Path(__file__).resolve().parents[1] / "data" / "licking_county" / "licking_county.corpus.json"


@pytest.mark.skipif(not COUNTY_CORPUS.exists(), reason="county corpus not present")
def test_great_circle_center_lands_in_tile_r04_c09():
    """Verified visually against the DEM tiles on 2026-10-13: the Great Circle
    ring is centred on Eagle Mound at 40.0410151N, 82.4310873W, in tile R04_C09."""
    corpus = json.loads(COUNTY_CORPUS.read_text())
    bridge = CoordBridge(corpus, {}, patch_size=64)
    x, y = bridge.wgs84_to_state_plane(-82.4310873, 40.0410151)
    tile_id, py, px = bridge.state_plane_to_tile_pixel(x, y)
    assert tile_id == "R04_C09"
    assert 1950 <= py <= 1975 and 1590 <= px <= 1610
