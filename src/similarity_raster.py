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

Query-time neighbourhood pooling
--------------------------------
The validation (docs/VALIDATION_PLAN.md, docs/CLASS_BENCHMARK.md) found that
the best pooling radius depends on what is being searched for: none for
linear features, 3x3-7x7 for compact ones, 15x15 for broad landforms. Rather
than bake one radius into the stored vectors, ``scores(..., radius=r)``
compares the mean of the query cell's (2r+1)^2 neighbourhood with the mean of
every other cell's neighbourhood, using only the unpooled matrix. Pooling is
linear, so

    pooled_i . q = mean_{j in N(i)} z_j . q  =  box_r(z . q)_i

is a box filter over the per-cell dot products, and the only extra state is
one vector of pooled norms ||pooled_i|| per radius, computed once (chunked
over dimensions) and optionally cached beside the index as
``<prefix>.poolnorms_r<r>.npy``. The result is exactly the cosine between
pooled vectors, i.e. what an index built with ``--pool r`` would store.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .coords import CoordBridge
from .index import TerrainIndex

VALUE_KINDS = ("similarity", "distance")
MAX_POOL_RADIUS = 15


def _box_sum(grid: np.ndarray, radius: int) -> np.ndarray:
    """Sum over the (2r+1)^2 window around every cell of ``grid`` (H, W[, B]); zero-padded."""
    from scipy.ndimage import uniform_filter

    k = 2 * radius + 1
    size = (k, k) + (1,) * (grid.ndim - 2)
    out = uniform_filter(grid, size=size, mode="constant", cval=0.0, output=grid.dtype)
    out *= float(k * k)
    return out


class SimilarityRaster:
    def __init__(
        self,
        index: TerrainIndex,
        bridge: CoordBridge,
        chunk_rows: int = 16384,
        cache_prefix: Optional[Path] = None,
    ):
        """
        Args:
            index: a cosine index with stored embeddings.
            bridge: coordinate bridge for the same corpus.
            chunk_rows: rows per pass when precomputing norms.
            cache_prefix: if given, pooled norms are read from / written to
                ``<cache_prefix>.poolnorms_r<radius>.npy`` (pass the index path
                without extension to keep them beside the index).
        """
        if index.embeddings is None:
            raise ValueError("Index has no stored embeddings; cannot score all patches.")
        if index.space != "cosinesimil":
            raise ValueError(
                f"SimilarityRaster assumes a cosine index; this index uses '{index.space}'."
            )

        self.index = index
        self.bridge = bridge
        self.cache_prefix = Path(cache_prefix) if cache_prefix is not None else None

        # ---- Grid geometry -------------------------------------------------
        self.patch_size = bridge.patch_size
        self.cell_size_ft = self.patch_size * bridge.resolution_ft
        tile_px = int(round(bridge.tile_size_ft / bridge.resolution_ft))
        self.patches_per_tile = tile_px // self.patch_size
        self.height = bridge.grid_rows * self.patches_per_tile
        self.width = bridge.grid_cols * self.patches_per_tile
        self.crs = bridge.crs
        # (a, b, c, d, e, f) affine as rasterio/GDAL use it: x = c + a*col, y = f + e*row.
        # Raster row 0 is the NORTH edge of the tile grid (grid_y_max), whereas
        # tile rows and patch y_start count from the SOUTH; the lookup below flips.
        self.transform = (
            self.cell_size_ft, 0.0, bridge.x_min,
            0.0, -self.cell_size_ft, bridge.grid_y_max,
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
            row_in_tile = meta["y_start"] // self.patch_size  # counted from the south
            self.cell_rows[pid] = (
                (bridge.grid_rows - 1 - tr) * self.patches_per_tile
                + (self.patches_per_tile - 1 - row_in_tile)
            )
            self.cell_cols[pid] = tc * self.patches_per_tile + meta["x_start"] // self.patch_size
        self._placed = self.cell_rows >= 0
        if not self._placed.any():
            raise ValueError("No patch in the index could be placed on the tile grid.")
        self._pidx = np.flatnonzero(self._placed)
        self._prow = self.cell_rows[self._pidx]
        self._pcol = self.cell_cols[self._pidx]
        # inverse lookup, cell -> patch id (-1 where no patch)
        self._cell_pid = np.full((self.height, self.width), -1, dtype=np.int64)
        self._cell_pid[self._prow, self._pcol] = self._pidx

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
        self._mean32 = mean.astype(np.float32)
        self._inv_std32 = (1.0 / std).astype(np.float32)
        self._chunk_rows = chunk_rows

        # Row norms of the z-scored embeddings, one chunked pass (no full copy).
        norms_sq = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk_rows):
            z = (emb[start : start + chunk_rows] - self._mean32) * self._inv_std32
            norms_sq[start : start + chunk_rows] = np.einsum(
                "ij,ij->i", z, z, dtype=np.float64
            )
        self._norms = np.sqrt(norms_sq)
        self._norms[self._norms == 0] = np.inf  # zero vectors score 0 everywhere

        # ---- Query-time pooling state (filled lazily per radius) ----------
        self._pooled_norms: Dict[int, np.ndarray] = {}
        self._counts: Dict[int, np.ndarray] = {}
        self._pool_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Pooling support
    # ------------------------------------------------------------------

    @staticmethod
    def _check_radius(radius: int) -> int:
        radius = int(radius)
        if radius < 0 or radius > MAX_POOL_RADIUS:
            raise ValueError(f"radius must be between 0 and {MAX_POOL_RADIUS}")
        return radius

    def _neighbour_counts(self, radius: int) -> np.ndarray:
        """(H, W) float64: number of placed cells in each (2r+1)^2 window, at least 1."""
        if radius not in self._counts:
            placed = np.zeros((self.height, self.width), dtype=np.float64)
            placed[self._prow, self._pcol] = 1.0
            self._counts[radius] = np.maximum(np.rint(_box_sum(placed, radius)), 1.0)
        return self._counts[radius]

    def _sidecar(self, radius: int) -> Optional[Path]:
        if self.cache_prefix is None:
            return None
        return self.cache_prefix.parent / f"{self.cache_prefix.name}.poolnorms_r{radius}.npy"

    def pooled_norms(self, radius: int, block_dims: int = 64) -> np.ndarray:
        """||mean of z-vectors over the (2r+1)^2 placed neighbours||, per patch.

        Unplaced and zero rows get ``inf`` so they score 0. Computed once per
        radius in blocks of ``block_dims`` dimensions (peak memory about
        4 x H x W x block_dims x 4 bytes) and cached in memory and, when a
        ``cache_prefix`` was given, on disk.
        """
        radius = self._check_radius(radius)
        if radius == 0:
            return self._norms
        if radius in self._pooled_norms:
            return self._pooled_norms[radius]
        with self._pool_lock:
            if radius in self._pooled_norms:
                return self._pooled_norms[radius]
            n = len(self.index.embeddings)
            side = self._sidecar(radius)
            if side is not None and side.exists():
                norms = np.load(side)
                if norms.shape == (n,):
                    self._pooled_norms[radius] = norms
                    return norms
            norms = self._compute_pooled_norms(radius, block_dims)
            if side is not None:
                side.parent.mkdir(parents=True, exist_ok=True)
                np.save(side, norms)
            self._pooled_norms[radius] = norms
            return norms

    def _compute_pooled_norms(self, radius: int, block_dims: int) -> np.ndarray:
        emb = self.index.embeddings
        n, dim = emb.shape
        inv_cnt = (1.0 / self._neighbour_counts(radius)).astype(np.float32)
        norms_sq = np.zeros(n, dtype=np.float64)
        for d0 in range(0, dim, block_dims):
            d1 = min(d0 + block_dims, dim)
            zb = (emb[self._pidx, d0:d1] - self._mean32[d0:d1]) * self._inv_std32[d0:d1]
            grid = np.zeros((self.height, self.width, d1 - d0), dtype=np.float32)
            grid[self._prow, self._pcol] = zb
            del zb
            pooled = _box_sum(grid, radius)
            del grid
            pooled *= inv_cnt[..., None]
            rows = pooled[self._prow, self._pcol]
            del pooled
            norms_sq[self._pidx] += np.einsum("ij,ij->i", rows, rows, dtype=np.float64)
        norms = np.sqrt(norms_sq)
        norms[~self._placed] = np.inf
        norms[norms == 0] = np.inf
        return norms

    def _pooled_query(self, patch_id: int, radius: int) -> np.ndarray:
        """Mean z-vector of the placed cells within ``radius`` of ``patch_id``'s cell (float64)."""
        emb = self.index.embeddings
        if radius == 0 or not self._placed[patch_id]:
            return (emb[patch_id].astype(np.float64) - self._mean) / self._std
        r0, c0 = int(self.cell_rows[patch_id]), int(self.cell_cols[patch_id])
        window = self._cell_pid[max(r0 - radius, 0): r0 + radius + 1, max(c0 - radius, 0): c0 + radius + 1]
        pids = window[window >= 0]
        return ((emb[pids].astype(np.float64) - self._mean) / self._std).mean(axis=0)

    def _dots(self, qz: np.ndarray) -> np.ndarray:
        """z(e_i) . qz for every row i, float64, via one float32 mat-vec on the stored matrix."""
        # e . (qz/sigma): keep the operand float32 so BLAS runs on the stored
        # matrix directly instead of upcasting a 2 GB copy.
        v32 = (qz / self._std).astype(np.float32)
        dots = self.index.embeddings @ v32
        return dots.astype(np.float64) - float(np.dot(self._mean / self._std, qz))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def scores(self, patch_id: int, radius: int = 0) -> np.ndarray:
        """Cosine similarity of every patch to ``patch_id`` (float32, length n).

        With ``radius > 0`` both sides are mean-pooled over their (2r+1)^2
        placed neighbours on the county grid before the cosine (see module
        docstring); unplaced patches score 0.
        """
        emb = self.index.embeddings
        if patch_id < 0 or patch_id >= len(emb):
            raise KeyError(patch_id)
        radius = self._check_radius(radius)

        qz = self._pooled_query(patch_id, radius)
        q_norm = float(np.linalg.norm(qz))
        if q_norm == 0:
            return np.zeros(len(emb), dtype=np.float32)
        dots = self._dots(qz)

        if radius == 0:
            sims = dots / (self._norms * q_norm)
            return np.clip(sims, -1.0, 1.0).astype(np.float32)

        grid = np.zeros((self.height, self.width), dtype=np.float64)
        grid[self._prow, self._pcol] = dots[self._pidx]
        pooled = _box_sum(grid, radius) / self._neighbour_counts(radius)
        out = np.zeros(len(emb), dtype=np.float32)
        sims = pooled[self._prow, self._pcol] / (self.pooled_norms(radius)[self._pidx] * q_norm)
        out[self._pidx] = np.clip(sims, -1.0, 1.0)
        return out

    def top_k(self, patch_id: int, k: int, radius: int = 0) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Exact k nearest patches as ``(patch_id, distance, metadata)``, distance = 1 - cosine.

        Same shape as :meth:`TerrainIndex.query_by_id` (the query patch is
        included, at distance 0), but exhaustive rather than HNSW, and with
        optional query-time pooling.
        """
        s = self.scores(patch_id, radius=radius)
        k = max(1, min(int(k), len(s)))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(int(i), float(1.0 - s[i]), self.index.metadata.get(int(i), {})) for i in top]

    def raster(self, patch_id: int, value: str = "similarity", radius: int = 0) -> np.ndarray:
        """
        County-wide grid (height x width, float32) of cosine similarity
        (or distance = 1 - similarity) to ``patch_id``. Cells with no patch
        (nodata, constant fill, missing tiles) are NaN.
        """
        if value not in VALUE_KINDS:
            raise ValueError(f"value must be one of {VALUE_KINDS}")
        s = self.scores(patch_id, radius=radius)
        if value == "distance":
            s = 1.0 - s
        grid = np.full((self.height, self.width), np.nan, dtype=np.float32)
        grid[self._prow, self._pcol] = s[self._pidx]
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
    """Offline use: python -m src.similarity_raster <index> --patch N [--radius R] -o out.tif"""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Write a county-wide similarity GeoTIFF for one patch.")
    ap.add_argument("index", help="Path to a county index (.idx/.meta/.corpus.json alongside)")
    ap.add_argument("--patch", type=int, help="Query patch id")
    ap.add_argument("--x", type=float, help="Query State Plane x (ft), with --y")
    ap.add_argument("--y", type=float, help="Query State Plane y (ft), with --x")
    ap.add_argument("--value", choices=VALUE_KINDS, default="similarity")
    ap.add_argument("--radius", type=int, default=0, help="query-time pooling radius in cells (1 = 3x3)")
    ap.add_argument("-o", "--output", required=True, help="Output GeoTIFF path")
    args = ap.parse_args()

    idx_path = Path(args.index)
    index = TerrainIndex.load(str(idx_path))
    with open(idx_path.with_suffix(".corpus.json")) as f:
        corpus = json.load(f)
    bridge = CoordBridge(corpus, index.metadata, patch_size=corpus.get("patch_size", 64))
    sr = SimilarityRaster(index, bridge, cache_prefix=idx_path.with_suffix(""))

    if args.patch is not None:
        pid = args.patch
    elif args.x is not None and args.y is not None:
        pid = bridge.state_plane_to_patch_id(args.x, args.y)
        if pid is None:
            raise SystemExit(f"No patch at State Plane ({args.x}, {args.y}).")
    else:
        raise SystemExit("Provide --patch or --x/--y.")

    grid = sr.raster(pid, value=args.value, radius=args.radius)
    tags = {"TERRAVECTOR_QUERY_PATCH_ID": str(pid), "TERRAVECTOR_VALUE": args.value, "TERRAVECTOR_POOL_RADIUS": str(args.radius)}
    Path(args.output).write_bytes(sr.to_geotiff_bytes(grid, tags=tags))
    print(f"wrote {args.output}: {sr.height}x{sr.width} cells, query patch {pid}, value={args.value}, radius={args.radius}")


if __name__ == "__main__":
    _main()
