"""Retrieval protocol: do ground-truth earthwork patches retrieve each other?

Sources of per-patch vectors (all evaluated with the same protocol):

  --source index            the county index (.meta embeddings, stored z-score params);
                            --meta selects another .meta (e.g. the 128 px index)
  --source <npz_dir>        a baseline directory from baseline_features.py
                            (one <tile>.npz per tile; z-scored here over finite rows)
  --pool3                   average each cell's vector with its 3x3 neighbours
                            (in z-scored space) before scoring

Protocol, for every ground-truth site used as a query (validation/gt_sites.json):
  * score all cells by cosine similarity (z-scored vectors);
  * mask cells within --exclude-ft of the query (its own site cannot count);
  * for every OTHER site: percentile of the best cell within its 3x3 block, and
    whether that block intersects the top-k cells (k = 100, 1000, 10000);
  * average precision over the other sites, ranking each by its best cell;
  * the same percentile for N uniformly random control cells, and for the
    hand-picked controls in validation/probe_sites.json (role == control);
  * with --landform, the same again for N landform-matched control blocks:
    cells drawn from the --match-frac of the county nearest to the query in a
    log/z-scored relief-slope-tilt space built from the 12-d simple terrain
    statistics (baseline_features.py --mode simple at the same patch size).
    Enrichment against these controls asks whether the query retrieves
    earthworks beyond "terrain that looks like a river terrace".

Prints a markdown table and writes a JSON report.

    python validation/evaluate_retrieval.py --source index --out eval_index.json
    python validation/evaluate_retrieval.py --source /path/baselines/lrm_p64 --out eval_lrm.json
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
from src.coords import CoordBridge  # noqa: E402

CORPUS = ROOT / "data" / "licking_county" / "licking_county.corpus.json"
META = ROOT / "data" / "licking_county" / "licking_county.meta"


class Grid:
    """Cell grid in the raster convention (row 0 = north) with a vector per cell."""

    def __init__(self, bridge: CoordBridge, patch_size: int):
        self.b = bridge
        self.ps = patch_size
        self.cell_ft = patch_size * bridge.resolution_ft
        tile_px = int(round(bridge.tile_size_ft / bridge.resolution_ft))
        self.ppt = tile_px // patch_size
        self.rows = bridge.grid_rows * self.ppt
        self.cols = bridge.grid_cols * self.ppt
        self.n = self.rows * self.cols

    def cell_of_tile_local(self, tile_id: str, y_start: int, x_start: int) -> int:
        tr, tc = int(tile_id[1:3]), int(tile_id[5:7])
        r = (self.b.grid_rows - 1 - tr) * self.ppt + (self.ppt - 1 - y_start // self.ps)
        c = tc * self.ppt + x_start // self.ps
        return r * self.cols + c

    def cell_of_xy(self, x: float, y: float):
        c = int((x - self.b.x_min) // self.cell_ft)
        r = int((self.b.grid_y_max - y) // self.cell_ft)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r * self.cols + c
        return None

    def rc(self, cell: int):
        return divmod(cell, self.cols)

    def neighbours(self, cell: int, radius: int = 1):
        r, c = self.rc(cell)
        out = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    out.append(rr * self.cols + cc)
        return out


def load_index_vectors(grid: Grid, meta_path: Path = META):
    import pickle
    # The .meta file is produced by this project's own build (trusted input; see CLAUDE.md).
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    emb = data["embeddings"]
    meta = data["metadata"]
    params = data["normalization_params"]
    X = np.full((grid.n, emb.shape[1]), np.nan, dtype=np.float32)
    for pid, m in meta.items():
        if m.get("tile_id") is None:
            continue
        X[grid.cell_of_tile_local(m["tile_id"], m["y_start"], m["x_start"])] = emb[pid]
    mean = np.asarray(params["mean"], dtype=np.float32)
    std = np.asarray(params["std"], dtype=np.float32)
    return X, mean, std


def load_npz_vectors(grid: Grid, d: Path, extra_dims: int = 0):
    """Place every tile's patch vectors on the cell grid.

    Memory-lean: the grid array is allocated once (with ``extra_dims`` spare
    columns for a parent-scale concat) and the z-score statistics are
    accumulated per tile instead of over a copy of the placed rows.
    """
    X = None
    s = ss = None
    n = 0
    for f in sorted(d.glob("R*_C*.npz")):
        z = np.load(f)
        tid = f.stem
        feats = z["feats"]
        ys, xs = z["y_start"].astype(np.int64), z["x_start"].astype(np.int64)
        if X is None:
            dim = feats.shape[1]
            X = np.full((grid.n, dim + extra_dims), np.nan, dtype=np.float32)
            s = np.zeros(dim); ss = np.zeros(dim)
        tr, tc = int(tid[1:3]), int(tid[5:7])
        r = (grid.b.grid_rows - 1 - tr) * grid.ppt + (grid.ppt - 1 - ys // grid.ps)
        c = tc * grid.ppt + xs // grid.ps
        ok = np.isfinite(feats).all(1) & (ys // grid.ps < grid.ppt) & (xs // grid.ps < grid.ppt)
        X[(r * grid.cols + c)[ok], :dim] = feats[ok]
        fv = feats[ok].astype(np.float64)
        s += fv.sum(0); ss += (fv * fv).sum(0); n += len(fv)
    mean = (s / n).astype(np.float32)
    std = np.sqrt(np.maximum(ss / n - (s / n) ** 2, 0)).astype(np.float32)
    std[std == 0] = 1
    return X, mean, std


def zscore_inplace(X: np.ndarray, mean: np.ndarray, std: np.ndarray, dim: int, chunk: int = 200_000) -> np.ndarray:
    """Z-score the first ``dim`` columns in place; unplaced rows become zero. Returns the placed mask."""
    placed = np.empty(len(X), dtype=bool)
    for i in range(0, len(X), chunk):
        blk = X[i:i + chunk, :dim]
        p = np.isfinite(blk).all(1)
        placed[i:i + chunk] = p
        blk -= mean; blk /= std
        blk[~p] = 0
    return placed


def pool3(Z: np.ndarray, placed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean over each cell's placed 3x3 neighbours (zero-padded at the grid edges, no wrap).

    Unplaced rows of ``Z`` are zero, so no mask multiply is needed; peak memory
    is the input plus one output array.
    """
    Zg = Z.reshape(rows, cols, -1)
    Pg = placed.reshape(rows, cols).astype(np.float32)
    acc = np.zeros_like(Zg); cnt = np.zeros_like(Pg)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            tr = slice(max(0, -dr), rows - max(0, dr)); tc = slice(max(0, -dc), cols - max(0, dc))
            sr = slice(max(0, dr), rows - max(0, -dr)); sc = slice(max(0, dc), cols - max(0, -dc))
            acc[tr, tc] += Zg[sr, sc]
            cnt[tr, tc] += Pg[sr, sc]
    acc /= np.maximum(cnt, 1)[..., None]
    return acc.reshape(len(Z), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--pool3", action="store_true")
    ap.add_argument("--meta", default=str(META), help="index .meta file when --source index")
    ap.add_argument("--sites", default=str(ROOT / "validation" / "gt_sites.json"))
    ap.add_argument("--controls", default=str(ROOT / "validation" / "probe_sites.json"))
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--concat-parent", default=None, help="npz dir at a coarser patch size; its containing cell's z-scored vector is appended to each cell")
    ap.add_argument("--parent-patch-size", type=int, default=None)
    ap.add_argument("--exclude-ft", type=float, default=1500.0)
    ap.add_argument("--landform", default=None, help="npz dir of simple terrain stats at --patch-size for landform-matched controls")
    ap.add_argument("--landform-dims", default="0,2,9,11", help="indices into the 12-d simple stats: default relief std, mean slope, TPI-32 std, plane gradient")
    ap.add_argument("--match-frac", type=float, default=0.05, help="fraction of the county nearest in landform space to draw matched controls from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "4")

    corpus = json.load(open(CORPUS))
    bridge = CoordBridge(corpus, {}, patch_size=a.patch_size)
    grid = Grid(bridge, a.patch_size)
    t0 = time.time()
    pps = a.parent_patch_size
    pgrid = Grid(bridge, pps) if a.concat_parent else None
    if a.concat_parent:
        PX, pmean, pstd = load_npz_vectors(pgrid, Path(a.concat_parent))
        pplaced = zscore_inplace(PX, pmean, pstd, PX.shape[1])
        PZ = PX
    extra = PZ.shape[1] if a.concat_parent else 0
    if a.source == "index":
        X, mean, std = load_index_vectors(grid, Path(a.meta))
        if extra:
            X = np.concatenate([X, np.full((grid.n, extra), np.nan, dtype=np.float32)], axis=1)
    else:
        X, mean, std = load_npz_vectors(grid, Path(a.source), extra_dims=extra)
    dim = X.shape[1] - extra
    placed = zscore_inplace(X, mean, std, dim)
    Z = X
    if a.concat_parent:
        # child cell -> tile-local (y_start, x_start) -> parent cell, via the tile grid geometry
        rr, cc = np.divmod(np.arange(grid.n), grid.cols)
        tr = grid.b.grid_rows - 1 - rr // grid.ppt          # tile row (from south)
        tc = cc // grid.ppt
        y_start = (grid.ppt - 1 - rr % grid.ppt) * grid.ps  # tile-local, from south
        x_start = (cc % grid.ppt) * grid.ps
        prow = (grid.b.grid_rows - 1 - tr) * pgrid.ppt + (pgrid.ppt - 1 - (y_start // pps))
        pcol = tc * pgrid.ppt + x_start // pps
        ok = (y_start // pps < pgrid.ppt) & (x_start // pps < pgrid.ppt)
        pcell = np.where(ok, prow * pgrid.cols + pcol, 0)
        both = placed & ok & pplaced[pcell]
        for i in range(0, grid.n, 200_000):
            sl = slice(i, i + 200_000)
            Z[sl, dim:] = np.where(both[sl, None], PZ[pcell[sl]], 0)
        Z[~both] = 0
        placed = both
        del PZ, PX
        print(f"concatenated parent vectors ({pps} px): {Z.shape[1]} dims, {int(placed.sum())} cells with both", flush=True)
    if a.pool3:
        Z = pool3(Z, placed, grid.rows, grid.cols)
        del X
    norms = np.linalg.norm(Z, axis=1)
    norms[norms == 0] = np.inf
    print(f"vectors: {int(placed.sum())} placed cells x {Z.shape[1]} dims, loaded in {time.time()-t0:.0f}s", flush=True)

    sites = json.load(open(a.sites))
    for s in sites:
        s["cell"] = grid.cell_of_xy(s["x"], s["y"])
        s["block"] = [c for c in grid.neighbours(s["cell"]) if placed[c]]
    ctrls = [s for s in json.load(open(a.controls)) if s.get("role") == "control"]
    for s in ctrls:
        s["cell"] = grid.cell_of_xy(s["x"], s["y"])
        s["block"] = [c for c in grid.neighbours(s["cell"]) if placed[c]]
    rng = np.random.default_rng(0)
    rand_cells = rng.choice(np.flatnonzero(placed), a.n_random, replace=False)
    rand_blocks = [[c for c in grid.neighbours(int(rc)) if placed[c]] for rc in rand_cells]

    L = None
    if a.landform:
        LX, _, _ = load_npz_vectors(grid, Path(a.landform))
        dims = [int(d) for d in a.landform_dims.split(",")]
        L = np.log1p(np.abs(LX[:, dims]))
        lplaced = np.isfinite(L).all(1) & placed
        L = (L - L[lplaced].mean(0)) / L[lplaced].std(0)
        L[~lplaced] = 0
        del LX
        print(f"landform descriptor: {len(dims)} dims from {a.landform}, {int(lplaced.sum())} cells", flush=True)

    # cell centres for the exclusion radius
    rr, cc = np.divmod(np.arange(grid.n), grid.cols)
    cx = bridge.x_min + (cc + 0.5) * grid.cell_ft
    cy = bridge.grid_y_max - (rr + 0.5) * grid.cell_ft

    report = {"source": a.source, "pool3": a.pool3, "patch_size": a.patch_size, "queries": {}}
    KS = (100, 1000, 10000)
    print(f"\n{'query':24s} {'cls':6s} " + " ".join(f"{s['name'][:10]:>10s}" for s in sites) + f" | {'AP':>5s} " + " ".join(f"h@{k:<5d}" for k in KS) + " | ctl med  rand med  rand p90")
    for q in sites:
        if not q["block"]:
            print(f"{q['name']}: no placed cell"); continue
        qz = Z[q["cell"]] if placed[q["cell"]] else Z[q["block"][0]]
        sc = (Z @ qz) / (norms * np.linalg.norm(qz))
        sc[~placed] = -np.inf
        far = np.hypot(cx - q["x"], cy - q["y"]) > a.exclude_ft
        valid = placed & far
        order = np.argsort(-sc[valid])
        rank_of = np.empty(grid.n, dtype=np.int64); rank_of[:] = grid.n
        rank_of[np.flatnonzero(valid)[order]] = np.arange(int(valid.sum()))
        n_valid = int(valid.sum())

        def best_rank(block):
            rs = [rank_of[c] for c in block if valid[c]]
            return min(rs) if rs else None

        others = [s for s in sites if s is not q and s["site"] != q["site"]]
        pcts, ranks = [], []
        row = []
        for s in sites:
            if s is q or s["site"] == q["site"]:
                row.append(None); continue
            br = best_rank(s["block"])
            p = 100.0 * (1 - br / n_valid) if br is not None else None
            row.append(p); pcts.append(p); ranks.append(br)
        ranks_sorted = sorted(r for r in ranks if r is not None)
        ap_ = float(np.mean([(i + 1) / (r + 1) for i, r in enumerate(ranks_sorted)])) if ranks_sorted else 0.0
        hits = {k: int(sum(1 for r in ranks_sorted if r < k)) for k in KS}
        ctl_p = [100.0 * (1 - best_rank(s["block"]) / n_valid) for s in ctrls if best_rank(s["block"]) is not None]
        rnd_p = np.array([100.0 * (1 - best_rank(b) / n_valid) for b in rand_blocks if best_rank(b) is not None])
        matched = {}
        if L is not None:
            qcell = q["cell"] if lplaced[q["cell"]] else q["block"][0]
            ld = np.linalg.norm(L - L[qcell], axis=1)
            ld[~(valid & lplaced)] = np.inf
            n_cand = max(a.n_random, int(a.match_frac * int((valid & lplaced).sum())))
            cand = np.argpartition(ld, n_cand)[:n_cand]
            mrng = np.random.default_rng(1)
            mcells = mrng.choice(cand, min(a.n_random, len(cand)), replace=False)
            mblocks = [[c for c in grid.neighbours(int(mc)) if placed[c]] for mc in mcells]
            m_p = np.array([100.0 * (1 - best_rank(b) / n_valid) for b in mblocks if best_rank(b) is not None])
            # landform percentiles of the query cell (0-100 within the county) for the write-up
            lq = [float((L[lplaced, j] <= L[qcell, j]).mean() * 100) for j in range(L.shape[1])]
            matched = {
                "matched_random_pct_median": float(np.median(m_p)), "matched_random_pct_p90": float(np.percentile(m_p, 90)),
                "matched_random_frac_above": {str(t): float((m_p >= t).mean()) for t in (99.9, 99.5, 98.0)},
                "matched_landform_dist_median": float(np.median(ld[mcells])),
                "uniform_landform_dist_median": float(np.median(ld[rand_cells][np.isfinite(ld[rand_cells])])),
                "query_landform_pct": lq,
            }
        report["queries"][q["name"]] = {
            "cls": q["cls"], "others": {s["name"]: p for s, p in zip(sites, row) if p is not None},
            "AP": ap_, "hits": hits, "n_others": len(others),
            "controls_pct": ctl_p, "random_pct_median": float(np.median(rnd_p)), "random_pct_p90": float(np.percentile(rnd_p, 90)),
            "random_frac_above": {str(t): float((rnd_p >= t).mean()) for t in (99.9, 99.5, 98.0)},
            "others_frac_above": {str(t): float(np.mean([p >= t for p in pcts])) if pcts else 0.0 for t in (99.9, 99.5, 98.0)},
            "others_cls": {s["name"]: s["cls"] for s, p in zip(sites, row) if p is not None},
            **matched,
        }
        print(f"{q['name'][:24]:24s} {q['cls']:6s} " + " ".join(f"{(f'{p:.1f}' if p is not None else '-'):>10s}" for p in row)
              + f" | {ap_:5.3f} " + " ".join(f"{hits[k]:>7d}" for k in KS)
              + f" | {np.median(ctl_p):7.1f} {np.median(rnd_p):8.1f} {np.percentile(rnd_p, 90):8.1f}"
              + (f"  matched med {matched['matched_random_pct_median']:5.1f} p90 {matched['matched_random_pct_p90']:5.1f}" if matched else ""))
    qs = report["queries"]
    summary = {
        "mean_AP": float(np.mean([v["AP"] for v in qs.values()])),
        "mean_hits": {str(k): float(np.mean([v["hits"][k] for v in qs.values()])) for k in KS},
        "mean_other_pct": float(np.mean([p for v in qs.values() for p in v["others"].values()])),
        "mean_random_pct_median": float(np.mean([v["random_pct_median"] for v in qs.values()])),
        "enrichment": {t: (float(np.mean([v["others_frac_above"][t] for v in qs.values()]))
                           / max(float(np.mean([v["random_frac_above"][t] for v in qs.values()])), 1e-6))
                       for t in ("99.9", "99.5", "98.0")},
        "others_frac_above": {t: float(np.mean([v["others_frac_above"][t] for v in qs.values()])) for t in ("99.9", "99.5", "98.0")},
        "random_frac_above": {t: float(np.mean([v["random_frac_above"][t] for v in qs.values()])) for t in ("99.9", "99.5", "98.0")},
        **({"matched_random_frac_above": {t: float(np.mean([v["matched_random_frac_above"][t] for v in qs.values()])) for t in ("99.9", "99.5", "98.0")},
            "matched_enrichment": {t: (float(np.mean([v["others_frac_above"][t] for v in qs.values()]))
                                       / max(float(np.mean([v["matched_random_frac_above"][t] for v in qs.values()])), 1e-6))
                                   for t in ("99.9", "99.5", "98.0")},
            "mean_matched_random_pct_median": float(np.mean([v["matched_random_pct_median"] for v in qs.values()])),
            "landform": {"source": a.landform, "dims": a.landform_dims, "match_frac": a.match_frac}}
           if L is not None else {}),
        "wall_to_wall_pct": float(np.mean([p for q in sites if q["cls"] == "wall" for s, p in
                                           ((s, qs[q["name"]]["others"].get(s["name"])) for s in sites if s["cls"] == "wall")
                                           if p is not None])) if any(q["cls"] == "wall" for q in sites) else None,
    }
    report["summary"] = summary
    print("\nsummary:", json.dumps(summary, indent=1))
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
