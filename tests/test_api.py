"""REST API: several indices behind one service, query-time pooling radius."""
import importlib
import json

import numpy as np
import pytest

from test_similarity_raster import _corpus, _setup, PS

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


def _save(tmp_path, name, pooling=None):
    index, sr, emb = _setup(n_drop=2)
    path = tmp_path / name
    index.save(str(path))
    corpus = dict(_corpus())
    corpus["patch_size"] = PS
    corpus["pooling"] = pooling
    (tmp_path / f"{name}.corpus.json").write_text(json.dumps(corpus))
    return path.with_suffix(".idx"), index, sr


@pytest.fixture
def client(tmp_path, monkeypatch):
    p64, index, sr = _save(tmp_path, "county_a")
    p_pooled, _, _ = _save(tmp_path, "county_pooled", pooling={"radius": 1, "kind": "mean"})
    monkeypatch.setenv("TERRAVECTOR_INDEXES", f"a={p64},pooled={p_pooled}")
    monkeypatch.delenv("TERRAVECTOR_INDEX", raising=False)
    monkeypatch.setenv("TERRAVECTOR_POOL_RADII", "1")
    import api
    api = importlib.reload(api)
    with TestClient(api.app) as c:
        yield c, index, sr


def test_info_lists_indexes_and_default(client):
    c, index, sr = client
    r = c.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["index"] == "a"
    assert [i["name"] for i in body["indexes"]] == ["a", "pooled"]
    assert body["stored_pool_radius"] == 0
    assert body["n_patches"] == len(index.embeddings)
    assert body["raster_grid"]["rows"] == sr.height
    r = c.get("/info", params={"index": "pooled"})
    assert r.json()["stored_pool_radius"] == 1
    assert c.get("/info", params={"index": "nope"}).status_code == 404
    assert c.get("/health").json()["indexes"] == ["a", "pooled"]


def test_query_by_id_without_pooling_uses_hnsw_and_echoes_index(client):
    c, index, sr = client
    r = c.post("/query/by-id", json={"patch_id": 4, "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query"]["index"] == "a" and body["query"]["pool_radius"] == 0
    assert body["results"][0]["patch_id"] == 4
    assert len(body["results"]) == 5
    assert body["results"][0]["center"] is not None


def test_query_with_pool_radius_is_the_exact_pooled_ranking(client):
    c, index, sr = client
    r = c.post("/query/by-id", json={"patch_id": 4, "k": 6, "pool_radius": 1})
    assert r.status_code == 200
    body = r.json()
    expect = sr.top_k(4, k=6, radius=1)
    assert [x["patch_id"] for x in body["results"]] == [pid for pid, _, _ in expect]
    np.testing.assert_allclose([x["distance"] for x in body["results"]], [d for _, d, _ in expect], atol=1e-6)
    assert body["query"]["pool_radius"] == 1
    # by-coords takes the same parameters
    ctr = body["query"]["coords"]
    r2 = c.post("/query/by-coords", json={"x": ctr["x"], "y": ctr["y"], "k": 6, "pool_radius": 1, "index": "a"})
    assert r2.status_code == 200
    assert [x["patch_id"] for x in r2.json()["results"]] == [pid for pid, _, _ in expect]


def test_pooling_a_stored_pooled_index_is_rejected(client):
    c, _, _ = client
    r = c.post("/query/by-id", json={"patch_id": 4, "k": 3, "pool_radius": 1, "index": "pooled"})
    assert r.status_code == 400
    assert "already" in r.json()["detail"]
    assert c.post("/query/by-id", json={"patch_id": 4, "k": 3, "index": "pooled"}).status_code == 200
    assert c.post("/query/by-id", json={"patch_id": 4, "k": 3, "pool_radius": 16}).status_code == 422


def test_raster_with_pool_radius(client):
    c, index, sr = client
    r = c.get("/similarity/raster", params={"patch_id": 4, "pool_radius": 2, "value": "distance"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/tiff"
    assert r.headers["x-terravector-pool-radius"] == "2"
    assert r.headers["x-terravector-index"] == "a"
    from rasterio.io import MemoryFile
    with MemoryFile(r.content) as mem, mem.open() as ds:
        grid = ds.read(1)
        assert ds.tags()["TERRAVECTOR_POOL_RADIUS"] == "2"
    np.testing.assert_allclose(grid, sr.raster(4, value="distance", radius=2), atol=1e-6, equal_nan=True)
    r = c.post("/similarity/raster", json={"patch_id": 4, "pool_radius": 1, "index": "pooled"})
    assert r.status_code == 400
