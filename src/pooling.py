"""Neighbourhood pooling of patch vectors on the county grid.

The ground-truth retrieval protocol (docs/VALIDATION_PLAN.md) found that
averaging each cell's vector with its 3x3 neighbours gives the best
wall-to-wall retrieval. Pooling is linear, so pooling raw embeddings and then
z-scoring with the UNPOOLED mean/std is identical to pooling z-scored vectors;
``TerrainIndex.build(normalization_params=...)`` keeps those statistics.
"""
from typing import Dict, Any, Tuple

import numpy as np


def grid_cells(metadata: Dict[int, Dict[str, Any]], tile_info: Dict[str, Dict[str, Any]],
               grid_rows: int, patches_per_tile: int, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Raster-convention (row 0 = north) cell row/col for every patch id 0..n-1.

    Tile rows and ``y_start`` count from the south (see src/coords.py), hence
    the two flips. Patches whose tile is unknown get row/col -1.
    """
    n = len(metadata)
    rows = np.full(n, -1, dtype=np.int64)
    cols = np.full(n, -1, dtype=np.int64)
    for pid, m in metadata.items():
        info = tile_info.get(m.get("tile_id"))
        if info is None:
            continue
        tr, tc = int(info["tile_row"]), int(info["tile_col"])
        rows[pid] = (grid_rows - 1 - tr) * patches_per_tile + (patches_per_tile - 1 - m["y_start"] // patch_size)
        cols[pid] = tc * patches_per_tile + m["x_start"] // patch_size
    return rows, cols


def pool_vectors(vectors: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                 grid_shape: Tuple[int, int], radius: int = 1) -> np.ndarray:
    """Mean of each patch's vector and its (2r+1)^2 placed neighbours.

    Cells with no patch contribute nothing (the mean is over present
    neighbours only). Returns an array aligned with ``vectors``.
    """
    R, C = grid_shape
    dim = vectors.shape[1]
    placed = (rows >= 0) & (cols >= 0)
    G = np.zeros((R, C, dim), dtype=np.float64)
    P = np.zeros((R, C), dtype=np.float64)
    G[rows[placed], cols[placed]] = vectors[placed]
    P[rows[placed], cols[placed]] = 1.0
    # Zero-pad so neighbourhoods never wrap across the grid edges.
    Gp = np.pad(G, ((radius, radius), (radius, radius), (0, 0)))
    Pp = np.pad(P, radius)
    acc = np.zeros_like(G)
    cnt = np.zeros_like(P)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            acc += Gp[radius + dr:radius + dr + R, radius + dc:radius + dc + C]
            cnt += Pp[radius + dr:radius + dr + R, radius + dc:radius + dc + C]
    pooled = acc / np.maximum(cnt, 1)[..., None]
    out = vectors.astype(np.float64).copy()
    out[placed] = pooled[rows[placed], cols[placed]]
    return out.astype(vectors.dtype)
