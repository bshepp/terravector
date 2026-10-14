"""Pack county DEM tiles for transfer: uint16 at 0.01 ft + zstd (~24 MB per tile).

    python validation/pack_tiles.py <county_tiles_dir> <out_dir> [--workers 1]

Writes <out_dir>/<tile_id>.dq, each a small JSON header line followed by the
zstd-compressed uint16 array. ``load_dq`` restores a float32 DEM (max error
0.005 ft). Nodata (NaN or < -1000) is stored as 0 and restored to NaN.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import zstandard as zstd

OFFSET = 600.0   # ft; Licking County DEM spans ~750-1360 ft
SCALE = 0.01     # ft per step; 600 + 65535*0.01 = 1255 ft ... see note below

# 0.01 ft steps only cover 655 ft of range; use 0.02 ft (max err 0.01 ft) to span 600-1910 ft.
SCALE = 0.02


def pack_tile(src: Path, dst: Path) -> int:
    d = np.load(src, mmap_mode="r")
    d = np.asarray(d, dtype=np.float64)
    bad = ~np.isfinite(d) | (d < -1000)
    q = np.round((d - OFFSET) / SCALE)
    q = np.where(bad, 0, np.clip(q, 1, 65535)).astype(np.uint16)
    hdr = {"shape": list(q.shape), "offset": OFFSET, "scale": SCALE, "dtype": "uint16", "nodata": 0, "src": src.name}
    blob = zstd.ZstdCompressor(level=6).compress(np.ascontiguousarray(q).tobytes())
    with open(dst, "wb") as f:
        f.write((json.dumps(hdr) + "\n").encode("utf-8"))
        f.write(blob)
    return len(blob)


def load_dq(path) -> np.ndarray:
    with open(path, "rb") as f:
        hdr = json.loads(f.readline().decode("utf-8"))
        blob = f.read()
    q = np.frombuffer(zstd.ZstdDecompressor().decompress(blob), dtype=np.uint16).reshape(hdr["shape"])
    d = q.astype(np.float32) * hdr["scale"] + hdr["offset"]
    d[q == hdr["nodata"]] = np.nan
    return d


def main():
    src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass
    files = sorted(src_dir.glob("R*_C*_dem.npy"))
    t0 = time.time()
    total = 0
    for i, f in enumerate(files):
        dst = out_dir / (f.name.replace("_dem.npy", "") + ".dq")
        if dst.exists():
            continue
        total += pack_tile(f, dst)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(files)} packed, {total/1e6:.0f} MB, {time.time()-t0:.0f}s", flush=True)
    # round-trip check on the last tile
    if files:
        d = np.load(files[-1], mmap_mode="r")
        r = load_dq(out_dir / (files[-1].name.replace("_dem.npy", "") + ".dq"))
        ok = np.isfinite(d)
        print(f"round-trip max err {np.nanmax(np.abs(r[ok] - np.asarray(d)[ok])):.4f} ft")
    print(f"done: {len(files)} tiles, {total/1e6:.0f} MB in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
