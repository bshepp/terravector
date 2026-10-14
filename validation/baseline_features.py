"""Baseline per-patch feature vectors for the retrieval protocol.

Computes, for every patch cell of every county tile (same 64 px stride and
tile-local (y_start, x_start) keys as the index), one of:

  simple   12-d hand terrain statistics of the raw DEM patch
           (std, range, slope mean/std, Laplacian curvature mean/std,
            TPI at 8 px and 32 px mean/std, plane-fit roughness, plane slope)
  lrm      40-d: the terravector 20-d rich analysis vector (signed-log1p scaled,
           src/features/analysis.py) on two plain local relief models
           (DEM minus Gaussian sigma = 8 px and 40 px), i.e. the same
           statistics the index uses but WITHOUT the decomposition x upsampling
           product. If this matches the full pipeline, the decomposition is not
           the contribution.
  full     the production residuals preset (640-d) at an arbitrary --patch-size,
           for the scale question (e.g. 128 px).

Output: <out_dir>/<mode>_p<patch>/<tile_id>.npz with arrays y_start, x_start,
feats (float32, NaN rows for nodata / constant patches).

    python validation/baseline_features.py --tiles <dir of .dq or _dem.npy> \
        --out <dir> --mode lrm --workers 8 [--patch-size 64] [--tiles-limit N]

Tile loading accepts the .dq format from pack_tiles.py or raw *_dem.npy.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

MIN_RANGE = 1e-6
MIN_VALID = 0.8


def load_tile(path: Path) -> np.ndarray:
    if path.suffix == ".dq":
        from pack_tiles import load_dq
        return load_dq(path)
    d = np.load(path, mmap_mode="r")
    d = np.asarray(d, dtype=np.float32)
    d[~np.isfinite(d) | (d < -1000)] = np.nan
    return d


def tile_id_of(path: Path) -> str:
    return path.name.replace("_dem.npy", "").replace(".dq", "")


# ---------------------------------------------------------------- simple ----

def simple_feats(p: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter, laplace
    gy, gx = np.gradient(p)
    slope = np.hypot(gx, gy)
    curv = laplace(p)
    tpi8 = p - uniform_filter(p, 17, mode="reflect")
    tpi32 = p - uniform_filter(p, 65, mode="reflect")
    h, w = p.shape
    yy, xx = np.mgrid[:h, :w]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(h * w)]
    coef, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
    resid = p.ravel() - A @ coef
    return np.array([
        p.std(), np.ptp(p), slope.mean(), slope.std(), curv.mean(), curv.std(),
        tpi8.mean(), tpi8.std(), tpi32.mean(), tpi32.std(), resid.std(), np.hypot(coef[0], coef[1]),
    ], dtype=np.float32)


# ------------------------------------------------------------------- lrm ----

def lrm_feats_for_tile(dem: np.ndarray, ps: int):
    from scipy.ndimage import gaussian_filter
    from src.features.analysis import analyze_features, analysis_to_vector
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    lrms = [filled - gaussian_filter(filled, s) for s in (8, 40)]

    def f(ys, xs):
        out = []
        for l in lrms:
            out.append(analysis_to_vector(analyze_features(l[ys:ys + ps, xs:xs + ps])))
        return np.concatenate(out)
    return f


# ------------------------------------------------------------------ full ----

def full_feats_for_tile(dem: np.ndarray, ps: int):
    from src.config import load_config
    from src.embedding import compute_embedding_from_config
    cfg = load_config(str(ROOT / "configs" / "residuals.yaml"))

    def f(ys, xs):
        return np.asarray(compute_embedding_from_config(dem[ys:ys + ps, xs:xs + ps].astype(np.float64), cfg), dtype=np.float32)
    return f


# ------------------------------------------------------------------- run ----

def process_tile(args):
    path, out_path, mode, ps = args
    if out_path.exists():
        return tile_id_of(path), 0, 0.0
    t0 = time.time()
    dem = load_tile(path)
    H, W = dem.shape
    valid = np.isfinite(dem)
    demf = np.where(valid, dem, np.nanmedian(dem)).astype(np.float32)  # NaN-free copy for feature extraction
    if mode == "lrm":
        f = lrm_feats_for_tile(demf, ps)
    elif mode == "full":
        f = full_feats_for_tile(demf, ps)
    else:
        f = lambda ys, xs: simple_feats(demf[ys:ys + ps, xs:xs + ps].astype(np.float64))  # noqa: E731
    ys_list, xs_list, rows = [], [], []
    dim = None
    for ys in range(0, H - ps + 1, ps):
        for xs in range(0, W - ps + 1, ps):
            ok = valid[ys:ys + ps, xs:xs + ps]
            patch = demf[ys:ys + ps, xs:xs + ps]
            if ok.mean() < MIN_VALID or float(patch.max() - patch.min()) < MIN_RANGE:
                v = None
            else:
                try:
                    v = f(ys, xs)
                except Exception:
                    v = None
            ys_list.append(ys); xs_list.append(xs); rows.append(v)
            if v is not None and dim is None:
                dim = len(v)
    dim = dim or 1
    feats = np.full((len(rows), dim), np.nan, dtype=np.float32)
    for i, v in enumerate(rows):
        if v is not None:
            feats[i] = v
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, y_start=np.array(ys_list, dtype=np.int32), x_start=np.array(xs_list, dtype=np.int32), feats=feats)
    return tile_id_of(path), len(rows), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["simple", "lrm", "full"], required=True)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiles-limit", type=int, default=None)
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    tiles_dir = Path(a.tiles)
    files = sorted(list(tiles_dir.glob("*.dq")) + list(tiles_dir.glob("R*_C*_dem.npy")))
    if a.tiles_limit:
        files = files[: a.tiles_limit]
    out_dir = Path(a.out) / f"{a.mode}_p{a.patch_size}"
    jobs = [(f, out_dir / f"{tile_id_of(f)}.npz", a.mode, a.patch_size) for f in files]
    print(f"{len(jobs)} tiles, mode={a.mode}, patch={a.patch_size}, workers={a.workers}", flush=True)
    t0 = time.time()
    done = 0
    with Pool(a.workers) as pool:
        for tid, n, dt in pool.imap_unordered(process_tile, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {tid}: {n} patches in {dt:.0f}s (elapsed {time.time()-t0:.0f}s)", flush=True)
    json.dump({"mode": a.mode, "patch_size": a.patch_size, "tiles": len(jobs), "seconds": time.time() - t0},
              open(out_dir / "_run.json", "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
