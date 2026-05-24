#!/usr/bin/env python
"""
Turing parameter sweeps for the residuals embedding.

Two sweep modes:

    --mode iterations  (default)
        Vary turing_iterations across {0, 50, 100, 200, 500, ...}.
        For each value, score `turing` and `bidir+turing` variants.
        Vanilla and bidir are run once each as fixed reference lines.
        Answers: does pattern formation actually help, and where?

    --mode fk-grid
        Vary the Gray-Scott (F, k) parameter pair across a small grid
        covering the canonical pattern regimes (spots, mazes,
        labyrinths, chaos). Iteration count is held fixed.
        Answers: does pattern *type* matter for terrain separability?

    --mode both
        Run iterations sweep first, then fk-grid (with iter fixed to
        the best iter found in the first sweep).

Separability metric matches compare_separability.py:
    mean(across-tile cosine distance) / mean(within-tile cosine distance)

Higher = more discriminative embedding.

Usage:
    python sweep_turing.py
    python sweep_turing.py --mode fk-grid --tiles R00_C00 R03_C07 R05_C10
    python sweep_turing.py --mode both --patches-per-tile 50
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import SignatureConfig, get_preset
from src.embedding import compute_embeddings_batch_from_config
from src.tiling import tile_dem

COUNTY_TILES_DIR = Path(r"F:\science-projects\RESIDUALS\results\county_tiles")
OUTPUT_DIR = Path(r"F:\science-projects\terravector\data\separability")

DEFAULT_TILES = ["R00_C00", "R03_C07", "R05_C10"]
DEFAULT_ITERATIONS = [0, 50, 100, 200, 500]

# Canonical Gray-Scott regimes (Pearson 1993). 4x4 covers most pattern types:
#   low F, low k     → moving holes / negative solitons
#   mid F, mid k     → labyrinths / fingerprints
#   high F, high k   → chaos / solitons
DEFAULT_F_VALUES = [0.018, 0.030, 0.040, 0.062]
DEFAULT_K_VALUES = [0.050, 0.055, 0.060, 0.065]


# =============================================================================
# Shared helpers (local copies — kept independent of compare_separability.py so
# the production sweep currently running there is not coupled to changes here)
# =============================================================================

def sample_patches_from_tile(tile_id, patches_per_tile, patch_size, rng):
    dem_path = COUNTY_TILES_DIR / f"{tile_id}_dem.npy"
    if not dem_path.exists():
        raise FileNotFoundError(f"Cached DEM not found: {dem_path}")
    dem = np.load(str(dem_path))
    patches = tile_dem(dem, patch_size=patch_size, overlap=0, min_valid_fraction=0.8)
    if not patches:
        return []
    if len(patches) <= patches_per_tile:
        sampled = patches
    else:
        idx = rng.choice(len(patches), size=patches_per_tile, replace=False)
        sampled = [patches[i] for i in idx]
    for p in sampled:
        p.tile_id = tile_id  # type: ignore[attr-defined]
    return sampled


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    unit = emb / norms
    sim = unit @ unit.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def separability(dist: np.ndarray, tile_idx: np.ndarray) -> Dict[str, float]:
    iu, ju = np.triu_indices(dist.shape[0], k=1)
    same = tile_idx[iu] == tile_idx[ju]
    d_w = dist[iu[same], ju[same]]
    d_a = dist[iu[~same], ju[~same]]
    wm = float(d_w.mean()) if d_w.size else float("nan")
    am = float(d_a.mean()) if d_a.size else float("nan")
    return {
        "within_mean": wm,
        "across_mean": am,
        "separability": float(am / wm) if wm > 0 else float("nan"),
    }


def base_config(full_grid: bool) -> SignatureConfig:
    cfg = get_preset("residuals")
    if not full_grid:
        cfg.residuals.decomposition_methods = ["gaussian", "wavelet_dwt"]
        cfg.residuals.upsampling_methods = ["bicubic", "lanczos"]
    return cfg


def variant(base: SignatureConfig, **residuals_overrides) -> SignatureConfig:
    cfg = copy.deepcopy(base)
    for k, v in residuals_overrides.items():
        setattr(cfg.residuals, k, v)
    return cfg


def score(cfg: SignatureConfig, patches, tile_idx) -> Dict:
    """Embed patches, z-score normalize, return separability + timing."""
    t = time.time()
    emb, diag = compute_embeddings_batch_from_config(
        patches, cfg, verbose=False, return_diagnostics=True
    )
    elapsed = time.time() - t
    mean = emb.mean(axis=0)
    std = emb.std(axis=0)
    std[std == 0] = 1.0
    emb_norm = (emb - mean) / std
    dist = cosine_distance_matrix(emb_norm)
    sep = separability(dist, tile_idx)
    return {
        "dim": int(emb.shape[1]),
        "time_seconds": elapsed,
        "ms_per_patch": elapsed / max(len(patches), 1) * 1000,
        **sep,
        "n_patches_with_failures": diag.get("n_patches_with_failures", 0),
        "total_failed_pairs": diag.get("total_failed_pairs", 0),
    }


# =============================================================================
# Iteration sweep
# =============================================================================

def run_iterations_sweep(base, patches, tile_idx, iterations: List[int]) -> Dict:
    print("\n" + "=" * 70)
    print("  Iteration sweep")
    print("=" * 70)

    # Fixed references — these don't change with iteration count, so run once.
    print("\n  reference variants (run once):")
    ref_vanilla = score(variant(base), patches, tile_idx)
    print(f"    vanilla            dim={ref_vanilla['dim']:<4} sep={ref_vanilla['separability']:.4f}"
          f"  ({ref_vanilla['ms_per_patch']:.1f} ms/patch)")
    ref_bidir = score(variant(base, bidirectional=True), patches, tile_idx)
    print(f"    bidirectional      dim={ref_bidir['dim']:<4} sep={ref_bidir['separability']:.4f}"
          f"  ({ref_bidir['ms_per_patch']:.1f} ms/patch)")

    # Swept variants.
    sweep_results = []
    print("\n  iter  | turing            | bidir+turing")
    print("  ------+-------------------+--------------------")
    for it in iterations:
        t_solo = score(
            variant(base, turing_intermediate=True, turing_iterations=it),
            patches, tile_idx,
        )
        t_bidir = score(
            variant(base, turing_intermediate=True, turing_iterations=it,
                    bidirectional=True),
            patches, tile_idx,
        )
        print(f"  {it:>4d}  |  sep={t_solo['separability']:.4f} "
              f"({t_solo['ms_per_patch']:.0f}ms) |  sep={t_bidir['separability']:.4f} "
              f"({t_bidir['ms_per_patch']:.0f}ms)")
        sweep_results.append({
            "iterations": it,
            "turing": t_solo,
            "bidir_turing": t_bidir,
        })

    # Find best of each curve.
    best_solo = max(sweep_results, key=lambda r: r["turing"]["separability"])
    best_bidir = max(sweep_results, key=lambda r: r["bidir_turing"]["separability"])
    print()
    print(f"  Best turing alone:   iter={best_solo['iterations']}  "
          f"sep={best_solo['turing']['separability']:.4f}  "
          f"(vs vanilla {ref_vanilla['separability']:.4f})")
    print(f"  Best bidir+turing:   iter={best_bidir['iterations']}  "
          f"sep={best_bidir['bidir_turing']['separability']:.4f}  "
          f"(vs bidir   {ref_bidir['separability']:.4f})")

    return {
        "references": {"vanilla": ref_vanilla, "bidirectional": ref_bidir},
        "sweep": sweep_results,
        "best_turing_iter": best_solo["iterations"],
        "best_bidir_turing_iter": best_bidir["iterations"],
    }


# =============================================================================
# (F, k) parameter grid sweep
# =============================================================================

def run_fk_grid(
    base, patches, tile_idx, F_values: List[float], k_values: List[float],
    iter_count: int, with_bidirectional: bool,
) -> Dict:
    label = "bidir+turing" if with_bidirectional else "turing"
    print("\n" + "=" * 70)
    print(f"  (F, k) grid sweep  [{label}, iter={iter_count}]")
    print("=" * 70)

    # Reference for normalization.
    ref_cfg = variant(base, bidirectional=with_bidirectional)
    ref = score(ref_cfg, patches, tile_idx)
    ref_label = "bidir" if with_bidirectional else "vanilla"
    print(f"  reference ({ref_label}):  sep={ref['separability']:.4f}")

    matrix = []
    for F in F_values:
        row = []
        for k in k_values:
            cfg = variant(
                base,
                turing_intermediate=True,
                turing_iterations=iter_count,
                turing_F=F,
                turing_k=k,
                bidirectional=with_bidirectional,
            )
            res = score(cfg, patches, tile_idx)
            row.append({"F": F, "k": k, **res})
        matrix.append(row)

    # Render as a table.
    print()
    header = "         " + "".join(f"  k={k:.3f}" for k in k_values)
    print(header)
    best_val = -float("inf")
    best_pos = None
    for i, F in enumerate(F_values):
        cells = []
        for j, k in enumerate(k_values):
            s = matrix[i][j]["separability"]
            if s > best_val:
                best_val, best_pos = s, (i, j)
            cells.append(f"  {s:.4f}")
        print(f"  F={F:.3f}" + "".join(cells))

    if best_pos is not None:
        bi, bj = best_pos
        print()
        print(f"  Best (F, k):  ({F_values[bi]:.3f}, {k_values[bj]:.3f})"
              f"  sep={best_val:.4f}  "
              f"(reference: {ref['separability']:.4f} -> "
              f"{best_val / ref['separability']:.3f}x)")

    return {
        "iter_count": iter_count,
        "with_bidirectional": with_bidirectional,
        "F_values": F_values,
        "k_values": k_values,
        "reference": ref,
        "matrix": matrix,
        "best": {
            "F": F_values[best_pos[0]] if best_pos else None,
            "k": k_values[best_pos[1]] if best_pos else None,
            "separability": best_val if best_pos else None,
        },
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Turing parameter sweeps")
    parser.add_argument("--mode", choices=["iterations", "fk-grid", "both"],
                        default="iterations")
    parser.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    parser.add_argument("--patches-per-tile", type=int, default=30)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--full-grid", action="store_true",
                        help="Use full 4x4 decomp/upsamp grid (slow). Default 2x2.")
    parser.add_argument("--iterations", type=int, nargs="+", default=DEFAULT_ITERATIONS,
                        help="Iteration counts to sweep (iterations mode)")
    parser.add_argument("--F-values", type=float, nargs="+", default=DEFAULT_F_VALUES,
                        help="Gray-Scott F values (fk-grid mode)")
    parser.add_argument("--k-values", type=float, nargs="+", default=DEFAULT_K_VALUES,
                        help="Gray-Scott k values (fk-grid mode)")
    parser.add_argument("--fk-iter", type=int, default=500,
                        help="Iteration count for fk-grid sweep (default 500)")
    parser.add_argument("--fk-bidirectional", action="store_true",
                        help="Include bidirectional in fk-grid evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Turing sweep  [{args.mode}]")
    print("=" * 70)
    print(f"  Tiles:             {' '.join(args.tiles)}")
    print(f"  Patches per tile:  {args.patches_per_tile}")
    print(f"  Patch size:        {args.patch_size}")
    print(f"  Grid:              {'4x4 (full)' if args.full_grid else '2x2 (compact)'}")

    rng = np.random.default_rng(args.seed)

    # Sample once, reuse across every variant.
    print("\n  Sampling patches...")
    all_patches = []
    tile_indices = []
    for ti, tile_id in enumerate(args.tiles):
        sampled = sample_patches_from_tile(tile_id, args.patches_per_tile,
                                           args.patch_size, rng)
        print(f"    {tile_id}: {len(sampled)} patches")
        all_patches.extend(sampled)
        tile_indices.extend([ti] * len(sampled))
    tile_idx = np.array(tile_indices, dtype=np.int32)
    if len(all_patches) < 2:
        print("  ERROR: need at least 2 patches")
        return 1
    print(f"  Total patches: {len(all_patches)}")

    base = base_config(args.full_grid)

    payload = {
        "tiles": args.tiles,
        "patches_per_tile": args.patches_per_tile,
        "patch_size": args.patch_size,
        "full_grid": args.full_grid,
        "seed": args.seed,
        "mode": args.mode,
    }

    t_total = time.time()

    if args.mode in ("iterations", "both"):
        payload["iterations_sweep"] = run_iterations_sweep(
            base, all_patches, tile_idx, args.iterations,
        )

    if args.mode in ("fk-grid", "both"):
        # In "both" mode, use the best iteration count discovered in the
        # iterations sweep instead of the CLI default.
        fk_iter = args.fk_iter
        with_bidir = args.fk_bidirectional
        if args.mode == "both":
            best_key = "best_bidir_turing_iter" if with_bidir else "best_turing_iter"
            fk_iter = payload["iterations_sweep"][best_key]
            print(f"\n  Using best iter from iterations sweep: {fk_iter}")
        payload["fk_grid_sweep"] = run_fk_grid(
            base, all_patches, tile_idx,
            args.F_values, args.k_values, fk_iter, with_bidir,
        )

    total = time.time() - t_total
    print()
    print("=" * 70)
    print(f"  Done in {total:.1f}s ({total/60:.1f} min)")
    print("=" * 70)

    out_path = args.output
    if out_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"turing_sweep_{args.mode}_{int(time.time())}.json"
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
