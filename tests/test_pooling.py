"""Neighbourhood pooling and precomputed normalization parameters."""
import numpy as np
import pytest

from src.index import build_index
from src.pooling import grid_cells, pool_vectors


def test_pool_vectors_averages_present_neighbours_only():
    # 2x2 grid, three placed cells, 1-d vectors
    rows = np.array([0, 0, 1, -1])
    cols = np.array([0, 1, 0, -1])
    v = np.array([[1.0], [3.0], [5.0], [99.0]], dtype=np.float32)
    out = pool_vectors(v, rows, cols, (2, 2), radius=1)
    assert out[0, 0] == pytest.approx((1 + 3 + 5) / 3)
    assert out[1, 0] == pytest.approx((1 + 3 + 5) / 3)
    assert out[2, 0] == pytest.approx((1 + 3 + 5) / 3)
    assert out[3, 0] == 99.0  # unplaced patch left untouched


def test_pool_vectors_does_not_wrap_across_grid_edges():
    rows = np.array([0, 2])
    cols = np.array([0, 0])
    v = np.array([[1.0], [7.0]], dtype=np.float32)
    out = pool_vectors(v, rows, cols, (3, 1), radius=1)
    assert out[0, 0] == 1.0 and out[1, 0] == 7.0


def test_grid_cells_flip_south_up_convention():
    tile_info = {"R00_C00": {"tile_row": 0, "tile_col": 0}, "R01_C00": {"tile_row": 1, "tile_col": 0}}
    meta = {0: {"tile_id": "R00_C00", "y_start": 0, "x_start": 0},     # SW corner of the grid
            1: {"tile_id": "R01_C00", "y_start": 64, "x_start": 64}}   # NE-most cell
    rows, cols = grid_cells(meta, tile_info, grid_rows=2, patches_per_tile=2, patch_size=64)
    assert (rows[0], cols[0]) == (3, 0)
    assert (rows[1], cols[1]) == (0, 1)


def test_build_uses_given_normalization_params(rng):
    X = rng.normal(size=(50, 4)).astype(np.float32)
    meta = {i: {"row": 0, "col": i} for i in range(50)}
    params = {"mean": np.zeros(4), "std": np.full(4, 2.0)}
    idx = build_index(X, meta, normalize=True, num_threads=1, normalization_params=params)
    assert np.allclose(idx.normalization_params["mean"], 0.0)
    assert np.allclose(idx.normalization_params["std"], 2.0)
    pid, dist, _ = idx.query(X[7], k=1)[0]
    assert pid == 7 and dist < 1e-5
