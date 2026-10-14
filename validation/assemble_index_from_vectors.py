"""Assemble a county TerrainIndex from per-tile vectors written by baseline_features.py.

Useful when the embeddings already exist (e.g. the 128 px full-pipeline run on
EC2) and only the HNSW graph and sidecars are needed. Produces the same three
files as build_county_index.py (<out>.idx, <out>.meta, <out>.corpus.json) with
metadata fields the bridge, raster and API expect.

    python validation/assemble_index_from_vectors.py --vectors <dir of R*_C*.npz> \
        --patch-size 128 --out data/licking_county/licking_county_p128.idx \
        --corpus data/licking_county/licking_county.corpus.json --threads 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True, help="existing corpus.json to copy county metadata from")
    ap.add_argument("--tiles-dir", default="F:/science-projects/RESIDUALS/results/county_tiles")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--pool", type=int, default=0, help="neighbourhood radius to mean-pool stored vectors (1 = 3x3); z-score params come from the unpooled vectors")
    a = ap.parse_args()

    from build_county_index import apply_core_cap
    apply_core_cap(a.threads)
    from src.config import load_config
    from src.index import build_index

    base = json.load(open(a.corpus))
    config = load_config(str(ROOT / "configs" / "residuals.yaml"))
    ps = a.patch_size

    all_emb, all_meta, tile_info = [], {}, {}
    gid = 0
    files = sorted(Path(a.vectors).glob("R*_C*.npz"))
    for f in files:
        tid = f.stem
        z = np.load(f)
        feats, ys, xs = z["feats"], z["y_start"], z["x_start"]
        ok = np.isfinite(feats).all(1)
        start = gid
        local = 0
        for y0, x0, v in zip(ys[ok], xs[ok], feats[ok]):
            y0, x0 = int(y0), int(x0)
            all_meta[gid] = {
                "row": y0 // ps, "col": x0 // ps, "y_start": y0, "x_start": x0,
                "center": (y0 + ps // 2, x0 + ps // 2), "shape": (ps, ps),
                "tile_id": tid, "dem_path": str(Path(a.tiles_dir) / f"{tid}_dem.npy"),
                "patch_size": ps, "local_id": local,
            }
            gid += 1
            local += 1
        all_emb.append(feats[ok].astype(np.float32))
        shape = base["tile_info"].get(tid, {}).get("tile_shape", [4224, 4224])
        tile_info[tid] = {"tile_row": int(tid[1:3]), "tile_col": int(tid[5:7]), "tile_shape": shape,
                          "n_patches": local, "global_id_start": start}
    emb = np.vstack(all_emb)
    norm_params = None
    if a.pool > 0:
        from src.pooling import grid_cells, pool_vectors
        cm = base["county_metadata"]
        ppt = int(round(cm["tile_size_ft"] / cm["resolution_ft"])) // ps
        rows, cols = grid_cells(all_meta, tile_info, cm["grid_rows"], ppt, ps)
        mean = emb.mean(axis=0, dtype=np.float64); std = emb.std(axis=0, dtype=np.float64)
        norm_params = {"mean": mean, "std": std}
        emb = pool_vectors(emb, rows, cols, (cm["grid_rows"] * ppt, cm["grid_cols"] * ppt), radius=a.pool)
        print(f"pooled vectors over a {2*a.pool+1}x{2*a.pool+1} neighbourhood; z-score params from unpooled vectors", flush=True)
    print(f"{len(files)} tiles, {emb.shape[0]:,} patches x {emb.shape[1]} dims; building HNSW with {a.threads} threads", flush=True)
    t0 = time.time()
    index = build_index(emb, all_meta, space=config.space, M=a.M, ef_construction=a.ef_construction,
                        normalize=config.normalize, num_threads=a.threads, normalization_params=norm_params)
    index.signature_config = config.to_dict()
    print(f"built in {time.time()-t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    index.save(a.out)
    corpus = {
        "source": f"vectors from {a.vectors} (baseline_features.py mode=full)",
        "county_metadata": base["county_metadata"],
        "tile_info": tile_info,
        "n_patches": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "patch_size": ps,
        "config": str(ROOT / "configs" / "residuals.yaml"),
        "signature_types": base.get("signature_types", ["residuals"]),
        "space": config.space, "M": a.M, "ef_construction": a.ef_construction,
        "normalized": config.normalize,
        "pooling": {"radius": a.pool, "kind": "mean over placed neighbours; z-score params from unpooled vectors"} if a.pool else None,
        "diagnostics": {"totals": {}, "by_tile": {}, "note": "not recorded by baseline_features.py"},
    }
    json.dump(corpus, open(Path(a.out).with_suffix(".corpus.json"), "w"), indent=2, default=str)
    print("saved", a.out)


if __name__ == "__main__":
    main()
