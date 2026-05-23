#!/usr/bin/env python
"""
Separability sweep: does bidirectional + Turing actually make tiles more
distinguishable from each other?

Picks a handful of cached county-tile DEMs, samples patches from each, then
builds embeddings under several config variants. For each variant we compute:

    separability = mean(across-tile cosine distance) /
                   mean(within-tile cosine distance)

Higher is better: it means the embedding pulls patches from the same tile
together and pushes patches from different tiles apart. Time-per-patch and
embedding dimension are reported alongside so we can see the cost.

Default variants:
    vanilla         residuals signature as shipped (320 d)
    +bidirectional  adds path-B asymmetry per pair    (640 d)
    +turing         vanilla with Gray-Scott intermediate
    +both           bidirectional + Turing

Usage:
    python compare_separability.py
    python compare_separability.py --tiles R00_C00 R00_C05 R03_C07 R05_C10
    python compare_separability.py --patches-per-tile 50 --full-grid
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

# Default tile choices spread across the county so the across-tile signal
# isn't dominated by neighbors. User can override with --tiles.
DEFAULT_TILES = ["R00_C00", "R03_C07", "R05_C10", "R08_C03"]


def _make_variant(name: str, base: SignatureConfig, **overrides) -> SignatureConfig:
    """Return a deep copy of base with the given residuals.* attrs overridden."""
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        setattr(cfg.residuals, k, v)
    cfg._variant_name = name  # purely for downstream labelling
    return cfg


def build_variants(full_grid: bool, turing_iter: int) -> List[SignatureConfig]:
    base = get_preset("residuals")
    if not full_grid:
        # Compact grid keeps the sweep tractable. 2x2 = 4 pairs.
        base.residuals.decomposition_methods = ["gaussian", "wavelet_dwt"]
        base.residuals.upsampling_methods = ["bicubic", "lanczos"]

    return [
        _make_variant("vanilla", base),
        _make_variant("bidirectional", base, bidirectional=True),
        _make_variant(
            "turing",
            base,
            turing_intermediate=True,
            turing_iterations=turing_iter,
        ),
        _make_variant(
            "bidir+turing",
            base,
            bidirectional=True,
            turing_intermediate=True,
            turing_iterations=turing_iter,
        ),
    ]


def sample_patches_from_tile(
    tile_id: str, patches_per_tile: int, patch_size: int, rng: np.random.Generator
):
    """Load one cached DEM tile and randomly sample N patches from it."""
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
        # Tag with originating tile so separability can group correctly.
        p.tile_id = tile_id  # type: ignore[attr-defined]
    return sampled


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    """N x N cosine-distance matrix (1 - cosine similarity)."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    unit = emb / norms
    sim = unit @ unit.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def separability(dist: np.ndarray, tile_idx: np.ndarray) -> Dict[str, float]:
    """Compute within/across-tile mean distances and the separability ratio.

    Within = upper-triangle pairs with same tile_idx.
    Across = upper-triangle pairs with different tile_idx.
    """
    n = dist.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    same_tile = tile_idx[iu] == tile_idx[ju]
    d_within = dist[iu[same_tile], ju[same_tile]]
    d_across = dist[iu[~same_tile], ju[~same_tile]]
    within_mean = float(d_within.mean()) if d_within.size else float("nan")
    across_mean = float(d_across.mean()) if d_across.size else float("nan")
    return {
        "within_mean": within_mean,
        "across_mean": across_mean,
        "separability": float(across_mean / within_mean) if within_mean > 0 else float("nan"),
        "n_within_pairs": int(d_within.size),
        "n_across_pairs": int(d_across.size),
    }


def run_variant(
    cfg: SignatureConfig, all_patches, tile_idx: np.ndarray
) -> Dict[str, object]:
    name = cfg._variant_name
    print(f"\n  >> variant: {name}  (dim={cfg.get_total_dim()})")
    t = time.time()
    emb, diag = compute_embeddings_batch_from_config(
        all_patches, cfg, verbose=False, return_diagnostics=True
    )
    elapsed = time.time() - t

    # z-score normalize (matches how build_county_index → TerrainIndex normalize)
    # — so the separability we measure here is comparable to what HNSW will see.
    mean = emb.mean(axis=0)
    std = emb.std(axis=0)
    std[std == 0] = 1.0
    emb_norm = (emb - mean) / std

    dist = cosine_distance_matrix(emb_norm)
    sep = separability(dist, tile_idx)

    print(
        f"     within={sep['within_mean']:.4f}  across={sep['across_mean']:.4f}  "
        f"separability={sep['separability']:.4f}  time={elapsed:.1f}s  "
        f"({elapsed / len(all_patches) * 1000:.1f} ms/patch)"
    )
    if diag["total_failed_pairs"]:
        print(
            f"     diagnostics: {diag['n_patches_with_failures']}/{diag['n_patches']} "
            f"patches had {diag['total_failed_pairs']} failed pairs"
        )

    return {
        "variant": name,
        "dim": int(emb.shape[1]),
        "time_seconds": elapsed,
        "ms_per_patch": elapsed / len(all_patches) * 1000,
        "separability": sep,
        "params_used": diag.get("params_used"),
        "failure_counts_by_pair": diag.get("failure_counts_by_pair", {}),
        "n_patches_with_failures": diag.get("n_patches_with_failures", 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Separability sweep across residuals signature variants"
    )
    parser.add_argument(
        "--tiles", nargs="+", default=DEFAULT_TILES,
        help=f"Tile IDs to sample patches from (default: {' '.join(DEFAULT_TILES)})",
    )
    parser.add_argument(
        "--patches-per-tile", type=int, default=30,
        help="Random patches sampled from each tile (default 30)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=64,
        help="Patch edge length in pixels (default 64)",
    )
    parser.add_argument(
        "--full-grid", action="store_true",
        help="Use the full 4-decomp x 4-upsamp grid (slow). Default: 2x2.",
    )
    parser.add_argument(
        "--turing-iter", type=int, default=50,
        help="Gray-Scott iterations for turing variants (default 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for patch sampling",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: data/separability/sweep_<timestamp>.json)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Separability sweep")
    print("=" * 70)
    print(f"  Tiles:             {' '.join(args.tiles)}")
    print(f"  Patches per tile:  {args.patches_per_tile}")
    print(f"  Patch size:        {args.patch_size}")
    print(f"  Grid:              {'4x4 (full)' if args.full_grid else '2x2 (compact)'}")
    print(f"  Turing iterations: {args.turing_iter}")

    rng = np.random.default_rng(args.seed)

    # ----- Sample patches once; reuse across variants -----
    print("\n  Sampling patches...")
    all_patches = []
    tile_indices = []
    for ti, tile_id in enumerate(args.tiles):
        sampled = sample_patches_from_tile(
            tile_id, args.patches_per_tile, args.patch_size, rng
        )
        print(f"    {tile_id}: {len(sampled)} patches")
        all_patches.extend(sampled)
        tile_indices.extend([ti] * len(sampled))
    tile_idx = np.array(tile_indices, dtype=np.int32)
    n_total = len(all_patches)
    if n_total < 2:
        print("  ERROR: need at least 2 patches to compute distances")
        return 1
    print(f"  Total patches: {n_total}")

    # ----- Run each variant -----
    variants = build_variants(args.full_grid, args.turing_iter)
    results = []
    for cfg in variants:
        results.append(run_variant(cfg, all_patches, tile_idx))

    # ----- Summary table -----
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'variant':<16} {'dim':>5} {'within':>9} {'across':>9} "
          f"{'separab.':>9} {'ms/patch':>10}")
    print(f"  {'-' * 16} {'-' * 5} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 10}")
    for r in results:
        s = r["separability"]
        print(
            f"  {r['variant']:<16} {r['dim']:>5d} "
            f"{s['within_mean']:>9.4f} {s['across_mean']:>9.4f} "
            f"{s['separability']:>9.4f} {r['ms_per_patch']:>10.1f}"
        )

    best = max(results, key=lambda r: r["separability"]["separability"])
    baseline = next(r for r in results if r["variant"] == "vanilla")
    gain = best["separability"]["separability"] / baseline["separability"]["separability"]
    cost = best["ms_per_patch"] / baseline["ms_per_patch"]
    print()
    print(f"  Best variant: {best['variant']}")
    print(
        f"  Separability vs vanilla: {gain:.2f}x   "
        f"(cost vs vanilla: {cost:.2f}x compute)"
    )

    # ----- Persist -----
    out_path = args.output
    if out_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"sweep_{int(time.time())}.json"
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tiles": args.tiles,
        "patches_per_tile": args.patches_per_tile,
        "patch_size": args.patch_size,
        "full_grid": args.full_grid,
        "turing_iter": args.turing_iter,
        "seed": args.seed,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
