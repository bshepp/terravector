#!/usr/bin/env python
"""
Build HNSW index for the full Licking County DEM corpus.

Processes all 200 county tiles from RESIDUALS with the residuals config
(4 decomp × 4 upsamp × 20 dims = 320-dimensional embeddings), then
builds a single unified HNSW index across all ~871K patches.

Supports checkpointing: embeddings are saved per-tile so the build
can resume if interrupted.

Usage:
    python build_county_index.py
    python build_county_index.py --patch-size 64 --resume
    python build_county_index.py --dry-run   # show plan without processing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, get_preset, validate_config
from src.tiling import tile_dem, patches_to_metadata
from src.embedding import compute_embeddings_batch_from_config
from src.index import TerrainIndex, build_index

COUNTY_TILES_DIR = Path(r"F:\science-projects\RESIDUALS\results\county_tiles")
OUTPUT_DIR = Path(r"F:\science-projects\terravector\data\licking_county")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CONFIG_PATH = Path(r"F:\science-projects\terravector\configs\residuals.yaml")


def load_county_metadata(tiles_dir: Path) -> dict:
    meta_path = tiles_dir / "metadata.json"
    with open(meta_path) as f:
        return json.load(f)


def get_tile_dem_path(tiles_dir: Path, tile_id: str) -> Path:
    return tiles_dir / f"{tile_id}_dem.npy"


def get_checkpoint_path(tile_id: str) -> Path:
    return CHECKPOINT_DIR / f"{tile_id}.npz"


def tile_is_checkpointed(tile_id: str) -> bool:
    return get_checkpoint_path(tile_id).exists()


def save_checkpoint(
    tile_id: str,
    embeddings: np.ndarray,
    metadata: dict,
    tile_shape: tuple,
    diagnostics: dict = None,
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        get_checkpoint_path(tile_id),
        embeddings=embeddings,
        tile_shape=np.array(tile_shape),
    )
    meta_path = CHECKPOINT_DIR / f"{tile_id}_meta.json"
    serializable = {}
    for pid, m in metadata.items():
        entry = {}
        for k, v in m.items():
            if isinstance(v, tuple):
                entry[k] = list(v)
            elif isinstance(v, np.integer):
                entry[k] = int(v)
            elif isinstance(v, np.floating):
                entry[k] = float(v)
            else:
                entry[k] = v
        serializable[str(pid)] = entry
    with open(meta_path, "w") as f:
        json.dump(serializable, f)

    # Sidecar diagnostics — recorded only if the embedding pipeline produced any.
    # The build can still resume without this file (load_checkpoint ignores it).
    if diagnostics is not None:
        diag_path = CHECKPOINT_DIR / f"{tile_id}_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)


def load_checkpoint(tile_id: str):
    data = np.load(get_checkpoint_path(tile_id))
    embeddings = data["embeddings"]
    tile_shape = tuple(data["tile_shape"])

    meta_path = CHECKPOINT_DIR / f"{tile_id}_meta.json"
    with open(meta_path) as f:
        raw = json.load(f)

    metadata = {}
    for pid_str, m in raw.items():
        pid = int(pid_str)
        if "center" in m and isinstance(m["center"], list):
            m["center"] = tuple(m["center"])
        if "shape" in m and isinstance(m["shape"], list):
            m["shape"] = tuple(m["shape"])
        metadata[pid] = m
    return embeddings, metadata, tile_shape


def process_tile(tile_id: str, tiles_dir: Path, config, patch_size: int, min_valid: float):
    dem_path = get_tile_dem_path(tiles_dir, tile_id)
    dem = np.load(str(dem_path))

    nan_frac = np.isnan(dem).sum() / dem.size
    if nan_frac > 0.95:
        return None, None, dem.shape, None

    patches = tile_dem(dem, patch_size=patch_size, overlap=0, min_valid_fraction=min_valid)
    if not patches:
        return None, None, dem.shape, None

    embeddings, diagnostics = compute_embeddings_batch_from_config(
        patches, config, verbose=False, return_diagnostics=True
    )
    diagnostics["tile_id"] = tile_id
    metadata = patches_to_metadata(patches)

    for pid in metadata:
        metadata[pid]["tile_id"] = tile_id
        metadata[pid]["dem_path"] = str(dem_path)
        metadata[pid]["patch_size"] = patch_size

    return embeddings, metadata, dem.shape, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Build Licking County HNSW index")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--min-valid", type=float, default=0.8)
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "licking_county.idx"))
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without processing")
    parser.add_argument(
        "--max-tiles", type=int, default=None,
        help="Process at most N tiles (for validation runs). Default: all tiles.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Licking County HNSW Index Builder")
    print("=" * 70)

    county_meta = load_county_metadata(COUNTY_TILES_DIR)
    tile_ids = county_meta["processed_tile_ids"]
    if args.max_tiles is not None and args.max_tiles > 0:
        tile_ids = tile_ids[: args.max_tiles]
    n_tiles = len(tile_ids)

    print(f"\n  Source:     {COUNTY_TILES_DIR}")
    if args.max_tiles is not None:
        print(f"  Tiles:      {n_tiles} (validation subset; full county = "
              f"{len(county_meta['processed_tile_ids'])})")
    else:
        print(f"  Tiles:      {n_tiles} "
              f"({county_meta['grid_rows']}×{county_meta['grid_cols']} grid)")
    print(f"  Resolution: {county_meta['resolution_ft']} ft/px")
    print(f"  CRS:        {county_meta['crs']}")
    print(f"  Config:     {CONFIG_PATH}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Output:     {args.output}")

    config = load_config(str(CONFIG_PATH))
    errors = validate_config(config)
    if errors:
        print("\nConfig validation errors:")
        for err in errors:
            print(f"  - {err}")
        return 1

    enabled = config.get_enabled_types()
    dim = config.get_total_dim()
    print(f"\n  Signature:  {', '.join(enabled)}")
    print(f"  Dimensions: {dim}")

    # Estimate patches: each tile is 4224×4224, at 64×64 = 66×66 = 4356 patches
    est_patches = n_tiles * 66 * 66
    est_mem_mb = est_patches * dim * 4 / (1024 * 1024)
    print(f"\n  Est patches: ~{est_patches:,}")
    print(f"  Est memory:  ~{est_mem_mb:,.0f} MB (embeddings)")

    if args.resume:
        done = [t for t in tile_ids if tile_is_checkpointed(t)]
        remaining = [t for t in tile_ids if not tile_is_checkpointed(t)]
        print(f"\n  Checkpoints: {len(done)}/{n_tiles} tiles complete")
        print(f"  Remaining:   {len(remaining)} tiles")
    else:
        remaining = tile_ids

    if args.dry_run:
        print("\n  [DRY RUN] Would process the above. Exiting.")
        return 0

    # Phase 1: Compute embeddings per tile
    print("\n" + "=" * 70)
    print("  Phase 1: Computing embeddings")
    print("=" * 70)

    t_start = time.time()
    tiles_processed = 0
    tiles_skipped = 0
    # Per-tile failure aggregates kept in-memory for the final corpus.json roll-up.
    # Sidecar JSON per tile holds the full breakdown.
    diagnostics_by_tile: dict = {}

    for i, tile_id in enumerate(tile_ids):
        if args.resume and tile_is_checkpointed(tile_id):
            continue

        t_tile = time.time()
        print(f"\n  [{i+1}/{n_tiles}] {tile_id} ...", end="", flush=True)

        embeddings, metadata, tile_shape, diag = process_tile(
            tile_id, COUNTY_TILES_DIR, config, args.patch_size, args.min_valid
        )

        if embeddings is None:
            print(f" skipped (no valid patches)")
            tiles_skipped += 1
            continue

        save_checkpoint(tile_id, embeddings, metadata, tile_shape, diagnostics=diag)
        elapsed = time.time() - t_tile
        tiles_processed += 1

        if diag is not None:
            diagnostics_by_tile[tile_id] = {
                "n_patches": diag["n_patches"],
                "n_patches_with_failures": diag["n_patches_with_failures"],
                "total_failed_pairs": diag["total_failed_pairs"],
            }
            fail_summary = ""
            if diag["total_failed_pairs"]:
                fail_summary = (
                    f", FAILS={diag['n_patches_with_failures']}/{diag['n_patches']} "
                    f"patches ({diag['total_failed_pairs']} pairs)"
                )
            print(
                f" {len(metadata)} patches, {embeddings.shape[1]}d, "
                f"{elapsed:.1f}s{fail_summary}"
            )
        else:
            print(f" {len(metadata)} patches, {embeddings.shape[1]}d, {elapsed:.1f}s")

        if tiles_processed % 10 == 0:
            total_elapsed = time.time() - t_start
            rate = tiles_processed / total_elapsed
            tiles_left = len(remaining) - tiles_processed
            eta = tiles_left / rate if rate > 0 else 0
            print(f"         Progress: {tiles_processed} tiles, "
                  f"{rate:.2f} tiles/s, ETA {eta/60:.0f} min")

    total_phase1 = time.time() - t_start
    print(f"\n  Phase 1 complete: {tiles_processed} tiles in {total_phase1/60:.1f} min "
          f"({tiles_skipped} skipped)")

    # Phase 2: Assemble and build HNSW index
    print("\n" + "=" * 70)
    print("  Phase 2: Assembling index")
    print("=" * 70)

    all_embeddings = []
    all_metadata = {}
    tile_info = {}
    global_id = 0

    for tile_id in tile_ids:
        if not tile_is_checkpointed(tile_id):
            continue

        embeddings, metadata, tile_shape = load_checkpoint(tile_id)
        n_local = len(metadata)

        # Remap local patch IDs to global IDs
        for local_id in sorted(metadata.keys()):
            meta = metadata[local_id]
            meta["local_id"] = local_id
            all_metadata[global_id] = meta
            global_id += 1

        all_embeddings.append(embeddings)

        # Parse tile grid position from tile_id
        parts = tile_id.split("_")
        tile_row = int(parts[0][1:])
        tile_col = int(parts[1][1:])
        tile_info[tile_id] = {
            "tile_row": tile_row,
            "tile_col": tile_col,
            "tile_shape": tile_shape,
            "n_patches": n_local,
            "global_id_start": global_id - n_local,
        }

    if not all_embeddings:
        print("  ERROR: No embeddings found. Nothing to index.")
        return 1

    embeddings_matrix = np.vstack(all_embeddings)
    n_total = embeddings_matrix.shape[0]
    print(f"\n  Total patches:   {n_total:,}")
    print(f"  Embedding dim:   {embeddings_matrix.shape[1]}")
    print(f"  Tiles indexed:   {len(tile_info)}")
    print(f"  Memory:          {embeddings_matrix.nbytes / (1024**2):,.1f} MB")

    print(f"\n  Building HNSW index (M={args.M}, ef={args.ef_construction})...")
    t_build = time.time()

    index = build_index(
        embeddings_matrix,
        all_metadata,
        space=config.space,
        M=args.M,
        ef_construction=args.ef_construction,
        normalize=config.normalize,
    )
    index.signature_config = config.to_dict()

    build_time = time.time() - t_build
    print(f"  HNSW built in {build_time:.1f}s")

    # Save index
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving index to: {args.output}")
    index.save(args.output)

    # Roll up failure counts across tiles. Only present when the residuals
    # signature is active (the only slice that currently surfaces diagnostics).
    diag_totals = {
        "tiles_with_failures": sum(
            1 for d in diagnostics_by_tile.values() if d["total_failed_pairs"] > 0
        ),
        "patches_with_failures": sum(
            d["n_patches_with_failures"] for d in diagnostics_by_tile.values()
        ),
        "total_failed_pairs": sum(
            d["total_failed_pairs"] for d in diagnostics_by_tile.values()
        ),
    }

    # Save corpus metadata alongside the index
    corpus_meta = {
        "source": str(COUNTY_TILES_DIR),
        "county_metadata": county_meta,
        "tile_info": tile_info,
        "n_patches": n_total,
        "embedding_dim": int(embeddings_matrix.shape[1]),
        "patch_size": args.patch_size,
        "config": str(CONFIG_PATH),
        "signature_types": enabled,
        "space": config.space,
        "M": args.M,
        "ef_construction": args.ef_construction,
        "normalized": config.normalize,
        "diagnostics": {
            "totals": diag_totals,
            "by_tile": diagnostics_by_tile,
        },
    }
    corpus_meta_path = Path(args.output).with_suffix(".corpus.json")
    with open(corpus_meta_path, "w") as f:
        json.dump(corpus_meta, f, indent=2, default=str)
    print(f"  Saved corpus metadata: {corpus_meta_path}")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  Done! {n_total:,} patches indexed in {total_time/60:.1f} min")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
