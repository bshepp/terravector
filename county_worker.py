"""Parallel helper for build_county_index.py.

The embedding pass is single-threaded and takes ~15 min per tile (~45 h for
the 200-tile county). Run N copies of this script alongside the main build to
fill the idle cores; the 2026-10-11 county index was built with 8 workers in
~16 h wall time, and would take ~5 h with all 9 processes started together.

How it stays consistent with the main process: build_county_index.py walks
the tile list from the front and, with --resume, re-checks for a checkpoint
before every tile. Workers walk the list from the BACK, each taking a disjoint
stride (worker K of N takes reversed_tiles[K::N]), and write the same per-tile
checkpoint + diagnostics sidecar. A worker skips a tile if its checkpoint
exists or the main process's log shows it has started that tile, and re-checks
right before writing. When the main process reaches the back of the list it
finds the tiles already done, skips them, and proceeds to assembly. Phase 2
reads every tile's diagnostics sidecar, so corpus.json covers worker tiles too.

Usage (from the repo root, with the main build already running and its stdout
redirected to a log file):

    python -u build_county_index.py --resume > build.log
    python -u county_worker.py --worker-id 0 --n-workers 8 --main-log build.log
    python -u county_worker.py --worker-id 1 --n-workers 8 --main-log build.log
    ...
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_county_index as bci  # noqa: E402
from src.config import load_config  # noqa: E402


def main_started(main_log: Path, tile_id: str) -> bool:
    try:
        text = main_log.read_text(errors="replace")
    except OSError:
        return False
    return f"] {tile_id} ..." in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--main-log", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--min-valid", type=float, default=0.8)
    args = ap.parse_args()

    config = load_config(bci.CONFIG_PATH)
    meta = bci.load_county_metadata(bci.COUNTY_TILES_DIR)
    tiles = list(reversed(meta["processed_tile_ids"]))
    mine = tiles[args.worker_id :: args.n_workers]
    tag = f"[worker {args.worker_id}/{args.n_workers}]"
    print(f"{tag} {len(mine)} candidate tiles, first: {mine[:3]}", flush=True)

    done = skipped = 0
    for tile_id in mine:
        if bci.tile_is_checkpointed(tile_id):
            skipped += 1
            continue
        if main_started(args.main_log, tile_id):
            print(f"{tag} {tile_id} started by main process; skipping", flush=True)
            skipped += 1
            continue

        t0 = time.time()
        print(f"{tag} {tile_id} ...", end="", flush=True)
        emb, md, shape, diag = bci.process_tile(
            tile_id, bci.COUNTY_TILES_DIR, config, args.patch_size, args.min_valid
        )
        if emb is None:
            print(" no valid patches (not checkpointed; main will record the skip)", flush=True)
            continue

        # Re-check right before writing so we never race the main process's write.
        if bci.tile_is_checkpointed(tile_id) or main_started(args.main_log, tile_id):
            print(f" discarded (main process took {tile_id} meanwhile)", flush=True)
            continue

        bci.save_checkpoint(tile_id, emb, md, shape, diagnostics=diag)
        done += 1
        fails = ""
        if diag and diag.get("total_failed_pairs"):
            fails = f", FAILS={diag['n_patches_with_failures']}/{diag['n_patches']}"
        print(f" {len(md)} patches, {emb.shape[1]}d, {time.time() - t0:.1f}s{fails}", flush=True)

    print(f"{tag} finished: {done} tiles written, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
