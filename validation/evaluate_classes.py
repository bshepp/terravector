"""Multi-class query-by-example benchmark on the county grid.

For every class in validation/class_grid_p<ps>.npz, sample query cells spread
across that class's polygons (at most --per-poly per polygon), score every
placed cell by cosine similarity to the query vector, drop cells within
--exclude-ft of the query and cells of the query's own polygon, and measure

  P@k         fraction of the top-k remaining cells that carry the query's class
  base rate   fraction of all remaining cells that carry the class
  enrichment  P@k / base rate  (1x = no better than picking cells at random)
  median pct  median percentile of same-class cells (50 = random)

per query, then per class with 95% bootstrap intervals over queries, plus a
confusion table: the mean class composition of the top-1000 for each query
class. Vectors come from the same sources as evaluate_retrieval.py.

    python validation/evaluate_classes.py --source <npz dir> --patch-size 128 [--pool3] --out eval_classes_x.json
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
sys.path.insert(0, str(ROOT / "validation"))
from src.coords import CoordBridge  # noqa: E402
from evaluate_retrieval import Grid, load_index_vectors, load_npz_vectors, zscore_inplace, pool3, CORPUS, META  # noqa: E402

KS = (100, 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--pool3", action="store_true")
    ap.add_argument("--pool-radius", type=int, default=0, help="mean-pool over a (2r+1)^2 neighbourhood (overrides --pool3)")
    ap.add_argument("--meta", default=str(META))
    ap.add_argument("--class-grid", default=None)
    ap.add_argument("--classes", default=None, help="comma list to restrict")
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--per-poly", type=int, default=4)
    ap.add_argument("--min-cells", type=int, default=20, help="skip classes with fewer placed cells")
    ap.add_argument("--exclude-ft", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "4")

    corpus = json.load(open(CORPUS))
    bridge = CoordBridge(corpus, {}, patch_size=a.patch_size)
    grid = Grid(bridge, a.patch_size)
    t0 = time.time()
    if a.source == "index":
        X, mean, std = load_index_vectors(grid, Path(a.meta))
    else:
        X, mean, std = load_npz_vectors(grid, Path(a.source))
    placed = zscore_inplace(X, mean, std, X.shape[1])
    Z = X
    radius = a.pool_radius or (1 if a.pool3 else 0)
    if radius:
        Z = pool3(Z, placed, grid.rows, grid.cols, radius=radius)
        del X
    norms = np.linalg.norm(Z, axis=1)
    norms[norms == 0] = np.inf
    print(f"vectors: {int(placed.sum())} placed cells x {Z.shape[1]} dims, loaded in {time.time()-t0:.0f}s", flush=True)

    cg = np.load(a.class_grid or str(ROOT / "validation" / f"class_grid_p{a.patch_size}.npz"))
    names = [str(n) for n in cg["classes"]]
    masks = cg["masks"] & placed[None, :]
    poly_id = cg["poly_id"]
    assert masks.shape[1] == grid.n, "class grid does not match the vector grid"
    use = [n for n in names if (a.classes is None or n in a.classes.split(",")) and masks[names.index(n)].sum() >= a.min_cells]

    rr, cc = np.divmod(np.arange(grid.n), grid.cols)
    cx = bridge.x_min + (cc + 0.5) * grid.cell_ft
    cy = bridge.grid_y_max - (rr + 0.5) * grid.cell_ft
    rng = np.random.default_rng(a.seed)

    report = {"source": a.source, "pool3": a.pool3, "pool_radius": radius, "patch_size": a.patch_size, "classes": {}, "confusion": {}}
    print(f"\n{'class':26s} {'cells':>7s} {'base%':>6s} {'queries':>7s} | {'P@100':>6s} {'enr@100':>8s} | {'P@1000':>6s} {'enr@1000':>8s} | {'med pct':>7s}")
    for name in use:
        ci = names.index(name)
        m = masks[ci]
        cells = np.flatnonzero(m)
        # spread queries over polygons
        pids = poly_id[ci, cells]
        order = rng.permutation(len(cells))
        taken, count = [], {}
        for i in order:
            p = int(pids[i])
            if count.get(p, 0) < a.per_poly:
                taken.append(cells[i]); count[p] = count.get(p, 0) + 1
            if len(taken) >= a.n_queries:
                break
        per_q = []
        comp_acc = np.zeros(len(names)); n_comp = 0
        for qc in taken:
            qz = Z[qc]
            sc = (Z @ qz) / (norms * np.linalg.norm(qz))
            far = np.hypot(cx - cx[qc], cy - cy[qc]) > a.exclude_ft
            own = poly_id[ci] == poly_id[ci, qc]
            if own.sum() > 0.25 * len(cells):  # line networks merge into a few huge features; radius exclusion only
                own = np.zeros_like(own)
            valid = placed & far & ~own
            vidx = np.flatnonzero(valid)
            svals = sc[vidx]
            top = vidx[np.argpartition(-svals, max(KS))[:max(KS)]]
            top = top[np.argsort(-sc[top])]
            same = m[valid]
            base = float(same.mean())
            row = {"cell": int(qc), "base_rate": base}
            for k in KS:
                pk = float(m[top[:k]].mean())
                row[f"P@{k}"] = pk
                row[f"enr@{k}"] = pk / max(base, 1e-9)
            # median percentile of same-class cells
            ranks = np.empty(len(vidx), dtype=np.int64)
            ranks[np.argsort(-svals)] = np.arange(len(vidx))
            same_ranks = ranks[same]
            row["median_pct"] = float(100.0 * (1 - np.median(same_ranks) / len(vidx))) if len(same_ranks) else None
            per_q.append(row)
            comp_acc += masks[:, top[:1000]].mean(axis=1); n_comp += 1
        if not per_q:
            continue

        def summ(key):
            v = np.array([q[key] for q in per_q if q[key] is not None])
            boots = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(a.n_boot)]
            return [float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        s = {k: summ(k) for k in ("P@100", "enr@100", "P@1000", "enr@1000", "median_pct")}
        s["base_rate"] = float(np.mean([q["base_rate"] for q in per_q]))
        s["n_cells"] = int(m.sum()); s["n_queries"] = len(per_q); s["n_polys_sampled"] = len(count)
        report["classes"][name] = {"summary": s, "queries": per_q}
        comp = comp_acc / max(n_comp, 1)
        report["confusion"][name] = {n: float(c) for n, c in zip(names, comp)}
        print(f"{name:26s} {s['n_cells']:7d} {100*s['base_rate']:6.2f} {len(per_q):7d} | {s['P@100'][0]:6.2f} {s['enr@100'][0]:7.1f}x | "
              f"{s['P@1000'][0]:6.2f} {s['enr@1000'][0]:7.1f}x | {s['median_pct'][0]:7.1f}", flush=True)

    summ_all = {k: float(np.mean([c["summary"][k][0] for c in report["classes"].values()])) for k in ("P@100", "enr@100", "P@1000", "enr@1000", "median_pct")}
    summ_all["n_classes"] = len(report["classes"])
    summ_all["geo_mean_enr@1000"] = float(np.exp(np.mean([np.log(max(c["summary"]["enr@1000"][0], 1e-3)) for c in report["classes"].values()])))
    report["summary"] = summ_all
    print("\nmacro average:", json.dumps(summ_all, indent=1))
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
