"""
Similarity raster

Score every patch in a county index against one query patch and lay the
scores out on the county's patch grid as a georeferenced raster. One cell per
patch (patch_size x resolution_ft on a side, 160 ft for the Licking County
build), origin at the county's NW corner, CRS from corpus.json (EPSG:3735).

The result is a plain GeoTIFF that any GIS can style as a heat map, which
makes it the most portable way to consume the index: no plugin, no nmslib on
the client side.

Scoring is a brute-force cosine similarity over the full embedding matrix
rather than an HNSW query, because every cell needs a value. It is one
matrix-vector product over the stored (raw) embeddings; z-score
normalization is folded in algebraically so no normalized copy of the
2 GB matrix is kept:

    z(e) = (e - mu) / sigma
    z(e) . z(q) = e . (z(q) / sigma) - (mu / sigma) . z(q)

Row norms ||z(e)|| are precomputed once at construction (one chunked pass).
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .coords import CoordBridge
from .index import TerrainIndex

VALUE_KINDS = ("similarity", "distance")


class SimilarityRaster:
    def __init__(
        self,
        index: TerrainIndex,
        bridge: CoordBridge,
        chunk_rows: int = 16384,
    ):
        if index.embeddings is None:
            raise ValueError("Index has no stored embeddings; cannot score all patches.")
        if index.space != "cosinesimil":
            raise ValueError(
                f"SimilarityRaster assumes a cosine index; this index uses '{index.space}'."
            )

        self.index = index
        self.bridge = bridge

        # ---- Grid geometry -------------------------------------------------
        self.patch_size = bridge.patch_size
        self.cell_size_ft = self.patch_size * bridge.resolution_ft
        tile_px = int(round(bridge.tile_size_ft / bridge.resolution_ft))
        self.patches_per_tile = tile_px // self.patch_size
        self.height = bridge.grid_rows * self.patches_per_tile
        self.width = bridge.grid_cols * self.patches_per_tile
        self.crs = bridge.crs
        # (a, b, c, d, e, f) affine as rasterio/GDAL use it: x = c + a*col, y = f + e*row
        self.transform = (
            self.cell_size_ft, 0.0, bridge.x_min,
            0.0, -self.cell_size_ft, bridge.y_max,
        )

        # ---- Patch -> cell lookup ------------------------------------------
        emb = index.embeddings
        n = len(emb)
        self.cell_rows = np.full(n, -1, dtype=np.int32)
        self.cell_cols = np.full(n, -1, dtype=np.int32)
        tile_rc: Dict[str, Tuple[int, int]] = {
            tid: (int(info["tile_row"]), int(info["tile_col"]))
            for tid, info in bridge.tile_info.items()
        }
        for pid, meta in index.metadata.items():
            tid = meta.get("tile_id")
            if tid is None or tid not in tile_rc:
                continue
            tr, tc = tile_rc[tid]
            self.cell_rows[pid] = tr * self.patches_per_tile + meta["y_start"] // self.patch_size
            self.cell_cols[pid] = tc * self.patches_per_tile + meta["x_start"] // self.patch_size
        self._placed = self.cell_rows >= 0
        if not self._placed.any():
            raise ValueError("No patch in the index could be placed on the tile grid.")

        # ---- Normalization folded into the scoring ------------------------
        params = index.normalization_params
        if params is not None:
            mean = np.asarray(params["mean"], dtype=np.float64)
            std = np.asarray(params["std"], dtype=np.float64)
        else:
            mean = np.zeros(emb.shape[1], dtype=np.float64)
            std = np.ones(emb.shape[1], dtype=np.float64)
        self._mean = mean
        self._std = std
        self._inv_std32 = (1.0 / std).astype(np.float32)

        # Row norms of the z-scored embeddings, one chunked pass (no full copy).
        norms_sq = np.empty(n, dtype=np.float64)
        mean32 = mean.astype(np.float32)
        for start in range(0, n, chunk_rows):
            z = (emb[start : start + chunk_rows] - mean32) * self._inv_std32
            norms_sq[start : start + chunk_rows] = np.einsum(
                "ij,ij->i", z, z, dtype=np.float64
            )
        self._norms = np.sqrt(norms_sq)
        self._norms[self._norms == 0] = np.inf  # zero vectors score 0 everywhere

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def scores(self, patch_id: int) -> np.ndarray:
        """Cosine similarity of every patch to ``patch_id`` (float32, length n)."""
        emb = self.index.embeddings
        if patch_id < 0 or patch_id >= len(emb):
            raise KeyError(patch_id)

        qz = (emb[patch_id].astype(np.float64) - self._mean) / self._std
        q_norm = float(np.linalg.norm(qz))
        if q_norm == 0:
            return np.zeros(len(emb), dtype=np.float32)

        # e . (z(q)/sigma): keep the operand float32 so BLAS runs on the
        # stored matrix directly instead of upcasting a 2 GB copy.
        v32 = (qz / self._std).astype(np.float32)
        dots = emb @ v32
        dots = dots.astype(np.float64) - float(np.dot(self._mean / self._std, qz))
        sims = dots / (self._norms * q_norm)
        return np.clip(sims, -1.0, 1.0).astype(np.float32)

    def raster(self, patch_id: int, value: str = "similarity") -> np.ndarray:
        """
        County-wide grid (height x width, float32) of cosine similarity
        (or distance = 1 - similarity) to ``patch_id``. Cells with no patch
        (nodata, constant fill, missing tiles) are NaN.
        """
        if value not in VALUE_KINDS:
            raise ValueError(f"value must be one of {VALUE_KINDS}")
        s = self.scores(patch_id)
        if value == "distance":
            s = 1.0 - s
        grid = np.full((self.height, self.width), np.nan, dtype=np.float32)
        grid[self.cell_rows[self._placed], self.cell_cols[self._placed]] = s[self._placed]
        return grid

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_geotiff_bytes(self, grid: np.ndarray, tags: Optional[Dict[str, str]] = None) -> bytes:
        """Encode a grid from :meth:`raster` as an in-memory GeoTIFF."""
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import Affine

        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                height=self.height,
                width=self.width,
                count=1,
                dtype="float32",
                crs=rasterio.crs.CRS.from_string(self.crs),
                transform=Affine(*self.transform),
                nodata=np.nan,
                compress="deflate",
            ) as dst:
                dst.write(grid, 1)
                if tags:
                    dst.update_tags(**tags)
            return mem.read()

    def grid_info(self) -> Dict[str, object]:
        return {
            "rows": self.height,
            "cols": self.width,
            "cell_size_ft": self.cell_size_ft,
            "crs": self.crs,
            "transform": list(self.transform),
            "n_cells_with_patches": int(self._placed.sum()),
        }


def _main():
    """Offline use: python -m src.similarity_raster <index> --patch N -o out.tif"""
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Write a county-wide similarity GeoTIFF for one patch.")
    ap.add_argument("index", help="Path to a county index (.idx/.meta/.corpus.json alongside)")
    ap.add_argument("--patch", type=int, help="Query patch id")
    ap.add_argument("--x", type=float, help="Query State Plane x (ft), with --y")
    ap.add_argument("--y", type=float, help="Query State Plane y (ft), with --x")
    ap.add_argument("--value", choices=VALUE_KINDS, default="similarity")
    ap.add_argument("-o", "--output", required=True, help="Output GeoTIFF path")
    args = ap.parse_args()

    idx_path = Path(args.index)
    index = TerrainIndex.load(str(idx_path))
    with open(idx_path.with_suffix(".corpus.json")) as f:
        corpus = json.load(f)
    bridge = CoordBridge(corpus, index.metadata, patch_size=corpus.get("patch_size", 64))
    sr = SimilarityRaster(index, bridge)

    if args.patch is not None:
        pid = args.patch
    elif args.x is not None and args.y is not None:
        pid = bridge.state_plane_to_patch_id(args.x, args.y)
        if pid is None:
            raise SystemExit(f"No patch at State Plane ({args.x}, {args.y}).")
    else:
        raise SystemExit("Provide --patch or --x/--y.")

    grid = sr.raster(pid, value=args.value)
    Path(args.output).write_bytes(
        sr.to_geotiff_bytes(grid, tags={"TERRAVECTOR_QUERY_PATCH_ID": str(pid), "TERRAVECTOR_VALUE": args.value})
    )
    print(f"wrote {args.output}: {sr.height}x{sr.width} cells, query patch {pid}, value={args.value}")


if __name__ == "__main__":
    _main()
