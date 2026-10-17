"""Query-time neighbourhood pooling in the similarity raster.

Pooling is linear, so the cosine between pooled z-vectors can be computed
from one matrix-vector product over the UNPOOLED embeddings: box-filter the
per-cell dot products and divide by precomputed pooled norms. These tests pin
that identity against the explicit construction in ``src.pooling``.
"""
import numpy as np
import pytest

from src.coords import CoordBridge
from src.index import build_index
from src.pooling import pool_vectors
from src.similarity_raster import SimilarityRaster

DIM = 8
PS = 64
TILE_FT = 320.0  # 128 px at 2.5 ft -> 2 x 2 patches of 64 px per tile
ROWS, COLS = 3, 3  # 3 x 3 tiles -> 6 x 6 cells, so no pooling window covers the whole grid


def _corpus():
    tile_info = {f"R{r:02d}_C{c:02d}": {"tile_row": r, "tile_col": c, "tile_shape": [128, 128]}
                 for r in range(ROWS) for c in range(COLS)}
    return {
        "county_metadata": {
            "grid_rows": ROWS, "grid_cols": COLS, "tile_size_ft": TILE_FT, "resolution_ft": 2.5,
            "county_bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": COLS * TILE_FT, "y_max": ROWS * TILE_FT - 40.0},
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
                    meta[pid] = {"tile_id": f"R{r:02d}_C{c:02d}", "y_start": ys, "x_start": xs, "patch_size": PS}
                    pid += 1
    return meta


def _setup(n_drop=1, seed=0):
    """A 6x6 cell grid with the last ``n_drop`` patches missing (holes in the grid)."""
    meta = _metadata()
    for pid in range(len(meta) - n_drop, len(meta)):
        del meta[pid]
    rng = np.random.default_rng(seed)
    emb = (rng.normal(size=(len(meta), DIM)) * rng.uniform(0.5, 3, DIM) + rng.normal(size=DIM)).astype(np.float32)
    index = build_index(emb, meta, normalize=True)
    bridge = CoordBridge(_corpus(), meta, patch_size=PS)
    sr = SimilarityRaster(index, bridge)
    return index, sr, emb


def _z(index, emb):
    p = index.normalization_params
    return (emb.astype(np.float64) - p["mean"]) / p["std"]


def _cosine_rows(Z, q):
    return Z @ q / (np.linalg.norm(Z, axis=1) * np.linalg.norm(q))


def test_radius_zero_matches_direct_cosine():
    index, sr, emb = _setup()
    Z = _z(index, emb)
    for pid in (0, 5, len(emb) - 1):
        np.testing.assert_allclose(sr.scores(pid, radius=0), _cosine_rows(Z, Z[pid]), atol=1e-5)
        np.testing.assert_allclose(sr.scores(pid), sr.scores(pid, radius=0))


@pytest.mark.parametrize("radius", [1, 2])
def test_pooled_scores_equal_cosine_between_explicitly_pooled_vectors(radius):
    index, sr, emb = _setup()
    Z = _z(index, emb)
    Zp = pool_vectors(Z, sr.cell_rows.astype(np.int64), sr.cell_cols.astype(np.int64), (sr.height, sr.width), radius=radius)
    for pid in (0, 3, 7, len(emb) - 1):
        np.testing.assert_allclose(sr.scores(pid, radius=radius), _cosine_rows(Zp, Zp[pid]), atol=1e-5)


def test_pooled_norms_match_explicit_pooling_and_ignore_holes():
    index, sr, emb = _setup(n_drop=3)
    Z = _z(index, emb)
    Zp = pool_vectors(Z, sr.cell_rows.astype(np.int64), sr.cell_cols.astype(np.int64), (sr.height, sr.width), radius=1)
    np.testing.assert_allclose(sr.pooled_norms(1), np.linalg.norm(Zp, axis=1), rtol=1e-5)
    # pooling over present neighbours only: the pooled query is the mean of the
    # placed cells in the window, never diluted by holes
    pid = len(emb) - 1
    assert sr.scores(pid, radius=1)[pid] == pytest.approx(1.0, abs=1e-6)


def test_top_k_is_exact_and_sorted():
    index, sr, emb = _setup()
    s = sr.scores(4, radius=1)
    res = sr.top_k(4, k=5, radius=1)
    assert [pid for pid, _, _ in res] == list(np.argsort(-s)[:5])
    assert res[0][0] == 4 and res[0][1] == pytest.approx(0.0, abs=1e-6)
    dists = [d for _, d, _ in res]
    assert dists == sorted(dists)
    np.testing.assert_allclose(dists, 1.0 - s[[pid for pid, _, _ in res]], atol=1e-6)
    assert res[1][2]["tile_id"] == index.metadata[res[1][0]]["tile_id"]


def test_raster_with_radius_places_pooled_scores():
    index, sr, emb = _setup()
    grid = sr.raster(0, value="distance", radius=1)
    s = sr.scores(0, radius=1)
    assert grid[sr.cell_rows[0], sr.cell_cols[0]] == pytest.approx(0.0, abs=1e-6)
    assert grid[sr.cell_rows[9], sr.cell_cols[9]] == pytest.approx(1.0 - s[9], abs=1e-6)
    assert np.isnan(grid).sum() == 1  # the dropped patch's cell


def test_pooled_norms_sidecar_cache(tmp_path):
    index, sr, emb = _setup()
    prefix = tmp_path / "idx"
    sr_cached = SimilarityRaster(index, sr.bridge, cache_prefix=prefix)
    n1 = sr_cached.pooled_norms(1)
    f = tmp_path / "idx.poolnorms_r1.npy"
    assert f.exists()
    np.testing.assert_allclose(np.load(f), n1)
    # a second raster instance reads the sidecar instead of recomputing
    sr2 = SimilarityRaster(index, sr.bridge, cache_prefix=prefix)
    np.save(f, n1 * 2)  # tamper to prove the file is what gets read
    np.testing.assert_allclose(sr2.pooled_norms(1), n1 * 2)
    with pytest.raises(ValueError):
        sr.scores(0, radius=-1)
