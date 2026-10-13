"""Tests for the HNSW TerrainIndex wrapper: build, query, save/load, thread cap."""
import numpy as np
import pytest

from src.index import TerrainIndex, build_index


@pytest.fixture
def embeddings(rng):
    # 300 points in 32-d with 3 well-separated clusters so neighbors are meaningful.
    centers = rng.normal(scale=10.0, size=(3, 32))
    labels = np.repeat(np.arange(3), 100)
    X = centers[labels] + rng.normal(scale=0.5, size=(300, 32))
    return X.astype(np.float32), labels


@pytest.fixture
def metadata(embeddings):
    _, labels = embeddings
    return {i: {"cluster": int(c), "row": i // 10, "col": i % 10} for i, c in enumerate(labels)}


class TestBuild:
    def test_build_index_with_thread_cap(self, embeddings, metadata):
        X, _ = embeddings
        idx = build_index(X, metadata, num_threads=2)
        assert idx.is_built
        assert idx.dim == 32
        assert len(idx.metadata) == 300
        assert idx.normalization_params is not None
        assert idx.normalization_params["mean"].shape == (32,)

    def test_build_without_thread_cap_still_works(self, embeddings, metadata):
        X, _ = embeddings
        idx = build_index(X, metadata)
        assert idx.is_built

    def test_empty_build_raises(self):
        with pytest.raises(ValueError):
            TerrainIndex(dim=8).build(np.empty((0, 8), dtype=np.float32), {})

    def test_no_normalize_keeps_raw(self, embeddings, metadata):
        X, _ = embeddings
        idx = build_index(X, metadata, normalize=False, num_threads=1)
        assert idx.normalization_params is None

    def test_zero_variance_dim_does_not_divide_by_zero(self, metadata, rng):
        X = rng.normal(size=(300, 8)).astype(np.float32)
        X[:, 3] = 7.0
        idx = build_index(X, metadata, num_threads=1)
        assert idx.normalization_params["std"][3] == 1
        res = idx.query_by_id(0, k=3)
        assert all(np.isfinite(d) for _, d, _ in res)


class TestQuery:
    def test_self_is_nearest(self, embeddings, metadata):
        X, _ = embeddings
        idx = build_index(X, metadata, num_threads=2)
        for pid in (0, 150, 299):
            top_id, dist, meta = idx.query_by_id(pid, k=5)[0]
            assert top_id == pid
            assert dist == pytest.approx(0.0, abs=1e-5)
            assert meta == metadata[pid]

    def test_neighbors_share_cluster(self, embeddings, metadata):
        X, labels = embeddings
        idx = build_index(X, metadata, num_threads=2)
        res = idx.query_by_id(42, k=10)
        assert all(meta["cluster"] == labels[42] for _, _, meta in res)

    def test_query_returns_k_sorted_results(self, embeddings, metadata):
        X, _ = embeddings
        idx = build_index(X, metadata, num_threads=2)
        res = idx.query(X[7], k=8)
        assert len(res) == 8
        dists = [d for _, d, _ in res]
        assert dists == sorted(dists)

    def test_k_is_clamped_to_index_size(self, embeddings, metadata):
        X, _ = embeddings
        idx = build_index(X, metadata, num_threads=2)
        assert len(idx.query(X[0], k=10_000)) == 300

    def test_invalid_patch_id(self, embeddings, metadata):
        X, _ = embeddings
        idx = build_index(X, metadata, num_threads=1)
        with pytest.raises(ValueError):
            idx.query_by_id(300)

    def test_query_before_build_raises(self):
        with pytest.raises(RuntimeError):
            TerrainIndex(dim=4).query(np.zeros(4, dtype=np.float32))


class TestPersistence:
    def test_save_load_roundtrip(self, embeddings, metadata, tmp_path):
        X, _ = embeddings
        idx = build_index(X, metadata, num_threads=2)
        idx.signature_config = {"preset": "test", "dim": 32}
        path = tmp_path / "terrain.idx"
        idx.save(str(path))
        assert path.exists() and path.with_suffix(".meta").exists()

        loaded = TerrainIndex.load(str(path))
        assert loaded.is_built
        assert loaded.dim == 32 and loaded.M == idx.M
        assert len(loaded.metadata) == 300
        assert loaded.signature_config == {"preset": "test", "dim": 32}
        np.testing.assert_array_equal(loaded.embeddings, idx.embeddings)
        np.testing.assert_allclose(
            loaded.normalization_params["mean"], idx.normalization_params["mean"]
        )

        before = [(i, round(d, 5)) for i, d, _ in idx.query_by_id(11, k=5)]
        after = [(i, round(d, 5)) for i, d, _ in loaded.query_by_id(11, k=5)]
        assert before == after

    def test_load_accepts_path_without_extension(self, embeddings, metadata, tmp_path):
        X, _ = embeddings
        build_index(X, metadata, num_threads=1).save(str(tmp_path / "t.idx"))
        assert TerrainIndex.load(str(tmp_path / "t")).is_built

    def test_save_before_build_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            TerrainIndex(dim=4).save(str(tmp_path / "x.idx"))
