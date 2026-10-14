"""Retrieval probe against the county index with ground-truth sites.

For every site marked ``role: query`` in the site list, score all patches
(exact cosine via SimilarityRaster) and report, for each listed site, the
percentile of the best cell within a 3x3 patch neighbourhood in the query's
county-wide score distribution. A random location's 3x3-best sits near the
77th percentile, so values close to 100 mean "retrieved".

Usage (from the repo root; cap BLAS threads first on a shared machine):

    set OMP_NUM_THREADS=4 & set MKL_NUM_THREADS=4 & set OPENBLAS_NUM_THREADS=4
    python validation/probe_retrieval.py validation/probe_sites.json out.json [query_raster.tif]

The optional third argument writes the county-wide similarity GeoTIFF for the
first query site, for inspection in a GIS.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coords import CoordBridge  # noqa: E402
from src.index import TerrainIndex  # noqa: E402
from src.similarity_raster import SimilarityRaster  # noqa: E402

IDX = os.environ.get("TERRAVECTOR_INDEX", str(ROOT / "data" / "licking_county" / "licking_county.idx"))


def main(sites_path: str, out_path: str, raster_path: str | None = None) -> None:
    t0 = time.time()
    index = TerrainIndex.load(IDX)
    corpus = json.load(open(IDX.replace(".idx", ".corpus.json")))
    bridge = CoordBridge(corpus, index.metadata, patch_size=64)
    sr = SimilarityRaster(index, bridge)
    print(f"loaded in {time.time() - t0:.0f}s; grid {sr.height}x{sr.width}; transform {sr.transform}")

    sites = json.load(open(sites_path))
    cell = bridge.patch_size * bridge.resolution_ft

    def neigh(x, y, r=1):
        out = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                p = bridge.state_plane_to_patch_id(x + dx * cell, y + dy * cell)
                if p is not None:
                    out.append(p)
        return out

    for s in sites:
        s["pid"] = bridge.state_plane_to_patch_id(s["x"], s["y"])
        s["n3"] = neigh(s["x"], s["y"])
        m = index.metadata.get(s["pid"], {}) if s["pid"] is not None else {}
        print(f"{s['name']:32s} pid={s['pid']}  tile={m.get('tile_id')} y_start={m.get('y_start')} x_start={m.get('x_start')}")

    queries = [s for s in sites if s.get("role") == "query" and s["pid"] is not None]
    targets = [s for s in sites if s["pid"] is not None]
    rng = np.random.default_rng(0)
    rand = rng.choice(len(index.embeddings), 300, replace=False)

    print("\nPercentile of the best cell within 3x3 around each target, in the query's county-wide distribution")
    print(f"{'query':30s} " + " ".join(f"{t['name'][:12]:>12s}" for t in targets))
    results = {}
    for q in queries:
        sc = sr.scores(q["pid"])
        placed = np.sort(sc[sr._placed])
        pct = lambda v: 100.0 * np.searchsorted(placed, v, side="left") / len(placed)  # noqa: E731
        row = [pct(max(sc[p] for p in t["n3"])) for t in targets]
        results[q["name"]] = dict(zip([t["name"] for t in targets], row))
        print(f"{q['name']:30s} " + " ".join(f"{v:12.1f}" for v in row))
        rb = []
        for p in rand:
            cx, cy = bridge.patch_id_to_state_plane_center(int(p))
            rb.append(pct(max(sc[pp] for pp in neigh(cx, cy))))
        results[q["name"]]["_random_3x3_best_median_pct"] = float(np.median(rb))
        print(f"{'  random 3x3-best median pct':30s} {np.median(rb):12.1f}")

    json.dump(results, open(out_path, "w"), indent=1)
    if raster_path and queries:
        q = queries[0]
        grid = sr.raster(q["pid"])
        Path(raster_path).write_bytes(sr.to_geotiff_bytes(grid, {"query": q["name"], "patch_id": str(q["pid"])}))
        print("wrote", raster_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
