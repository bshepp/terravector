"""Tests for DEM tiling: patch grid geometry and the nodata / constant-fill filters."""
import numpy as np

from src.tiling import patches_to_metadata, tile_dem


class TestTileGrid:
    def test_exact_grid(self, small_dem):
        patches = tile_dem(small_dem, patch_size=64)
        assert len(patches) == 4
        assert [p.id for p in patches] == [0, 1, 2, 3]
        origins = {(p.row, p.col, p.y_start, p.x_start) for p in patches}
        assert origins == {(0, 0, 0, 0), (0, 1, 0, 64), (1, 0, 64, 0), (1, 1, 64, 64)}
        assert all(p.data.shape == (64, 64) for p in patches)

    def test_patch_data_matches_source(self, small_dem):
        patches = tile_dem(small_dem, patch_size=64)
        p = next(p for p in patches if (p.row, p.col) == (1, 1))
        np.testing.assert_array_equal(p.data, small_dem[64:128, 64:128])

    def test_center_property(self, small_dem):
        p = tile_dem(small_dem, patch_size=64)[3]
        assert p.center == (64 + 32, 64 + 32)

    def test_overlap_increases_patch_count(self, small_dem):
        n_plain = len(tile_dem(small_dem, patch_size=64))
        n_overlap = len(tile_dem(small_dem, patch_size=64, overlap=32))
        assert n_overlap > n_plain

    def test_partial_edges_padded_or_skipped(self, small_dem):
        dem = small_dem[:100, :100]  # 64 + a partial 36-px remainder each way
        padded = tile_dem(dem, patch_size=64, pad_partial=True)
        skipped = tile_dem(dem, patch_size=64, pad_partial=False)
        assert len(skipped) == 1
        assert len(padded) >= len(skipped)
        assert all(p.data.shape == (64, 64) for p in padded)


class TestFilters:
    def test_constant_patch_is_dropped(self, small_dem):
        dem = small_dem.copy()
        dem[64:, 64:] = 950.0  # nodata fill the way county tiles write edges
        patches = tile_dem(dem, patch_size=64)
        assert len(patches) == 3
        assert (1, 1) not in {(p.row, p.col) for p in patches}

    def test_constant_filter_can_be_disabled(self, small_dem):
        dem = small_dem.copy()
        dem[64:, 64:] = 950.0
        assert len(tile_dem(dem, patch_size=64, min_range=0)) == 4

    def test_mostly_nan_patch_is_dropped(self, small_dem):
        dem = small_dem.copy()
        dem[0:64, 0:40] = np.nan  # 62.5% NaN in the (0, 0) patch
        patches = tile_dem(dem, patch_size=64, min_valid_fraction=0.8)
        assert len(patches) == 3
        assert (0, 0) not in {(p.row, p.col) for p in patches}

    def test_slightly_nan_patch_is_kept(self, small_dem):
        dem = small_dem.copy()
        dem[0:64, 0:6] = np.nan  # <10% NaN
        assert len(tile_dem(dem, patch_size=64, min_valid_fraction=0.8)) == 4

    def test_ids_stay_contiguous_after_drops(self, small_dem):
        dem = small_dem.copy()
        dem[0:64, 0:64] = 950.0
        patches = tile_dem(dem, patch_size=64)
        assert [p.id for p in patches] == list(range(len(patches)))


class TestMetadata:
    def test_metadata_fields(self, small_dem):
        patches = tile_dem(small_dem, patch_size=64)
        meta = patches_to_metadata(patches)
        assert set(meta) == {p.id for p in patches}
        m = meta[3]
        assert m["row"] == 1 and m["col"] == 1
        assert m["y_start"] == 64 and m["x_start"] == 64
        assert m["center"] == (96, 96)
        assert m["shape"] == (64, 64)
