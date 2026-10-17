#!/usr/bin/env python
"""
terravector REST API

FastAPI service exposing terrain similarity queries over HTTP.

Usage:
    uvicorn api:app --reload --host 127.0.0.1 --port 8000

Docs at http://127.0.0.1:8000/docs

Several indices (for example the 64 px and 128 px county builds) can be served
side by side; every query names one with ``index`` (default: the first) and may
ask for query-time neighbourhood pooling with ``pool_radius`` (0 = single
cell, 1 = 3x3, 3 = 7x7, 7 = 15x15 cells). Pooled queries are exact scans over
the stored embeddings (see src/similarity_raster.py), not HNSW lookups.

Configuration:
    TERRAVECTOR_INDEXES   comma list of ``name=path`` (or bare paths, named by
                          file stem), e.g. ``p128=data/x_p128.idx,p64=data/x.idx``
    TERRAVECTOR_INDEX     single index path (legacy; used when INDEXES is unset)
    TERRAVECTOR_POOL_RADII  radii whose pooled norms are precomputed at startup
                          and cached beside each index (e.g. ``1,3``); other
                          radii are computed on first use (seconds to tens of
                          seconds for the 64 px county index) and cached too.
"""

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from src.index import TerrainIndex
from src.coords import CoordBridge
from src.similarity_raster import SimilarityRaster, MAX_POOL_RADIUS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INDEX_PATH = str(Path(__file__).parent / "data" / "licking_county" / "licking_county.idx")


def _configured_indexes() -> List[Tuple[str, str]]:
    """[(name, path)] from TERRAVECTOR_INDEXES, else TERRAVECTOR_INDEX, else the default path."""
    spec = os.environ.get("TERRAVECTOR_INDEXES", "").strip()
    out: List[Tuple[str, str]] = []
    if spec:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                name, path = item.split("=", 1)
            else:
                name, path = Path(item).with_suffix("").name, item
            out.append((name.strip(), path.strip()))
    else:
        path = os.environ.get("TERRAVECTOR_INDEX", DEFAULT_INDEX_PATH)
        out.append((Path(path).with_suffix("").name, path))
    return out


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CoordsXY(BaseModel):
    x: float
    y: float


class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class WGS84BBox(BaseModel):
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


class PatchResult(BaseModel):
    patch_id: int
    distance: float
    tile_id: Optional[str] = None
    state_plane_bbox: Optional[BBox] = None
    wgs84_bbox: Optional[WGS84BBox] = None
    center: Optional[CoordsXY] = None


class QueryResponse(BaseModel):
    query: Dict[str, Any]
    results: List[PatchResult]


POOL_RADIUS_FIELD = Field(
    default=None, ge=0, le=MAX_POOL_RADIUS,
    description="Query-time neighbourhood pooling radius in cells (0 = none, 1 = 3x3, 3 = 7x7, 7 = 15x15). "
                "Default 0. Pooled queries are exact scans, not HNSW.",
)
INDEX_FIELD = Field(default=None, description="Name of the index to query (see /info); default is the first configured.")


class QueryByIdRequest(BaseModel):
    patch_id: int
    k: int = Field(default=10, ge=1, le=100)
    index: Optional[str] = INDEX_FIELD
    pool_radius: Optional[int] = POOL_RADIUS_FIELD


class QueryByCoordsRequest(BaseModel):
    lon: Optional[float] = None
    lat: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    crs: Optional[str] = None
    k: int = Field(default=10, ge=1, le=100)
    index: Optional[str] = INDEX_FIELD
    pool_radius: Optional[int] = POOL_RADIUS_FIELD


class SimilarityRasterRequest(BaseModel):
    """Query location as a patch id, WGS84 (lon, lat), or (x, y) in ``crs``."""

    patch_id: Optional[int] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    crs: Optional[str] = None
    value: str = Field(default="similarity", pattern="^(similarity|distance)$")
    index: Optional[str] = INDEX_FIELD
    pool_radius: Optional[int] = POOL_RADIUS_FIELD


class RasterGridInfo(BaseModel):
    rows: int
    cols: int
    cell_size_ft: float
    crs: str
    transform: List[float]
    n_cells_with_patches: int


class IndexSummary(BaseModel):
    name: str
    path: str
    patch_size: int
    cell_size_ft: Optional[float] = None
    n_patches: int
    stored_pool_radius: int = Field(description="Radius baked into the stored vectors at build time (0 = unpooled)")


class IndexInfo(BaseModel):
    index: str
    indexes: List[IndexSummary]
    n_patches: int
    embedding_dim: int
    space: str
    crs: str
    patch_size: int
    stored_pool_radius: int
    pool_radius_max: int = MAX_POOL_RADIUS
    county_bounds: BBox
    county_bounds_wgs84: WGS84BBox
    signature_types: List[str]
    raster_grid: Optional[RasterGridInfo] = None


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="terravector",
    description="Terrain patch similarity search API",
    version="0.2.0",
)


@dataclass
class IndexContext:
    name: str
    path: str
    index: TerrainIndex
    corpus: Optional[Dict[str, Any]] = None
    bridge: Optional[CoordBridge] = None
    raster: Optional[SimilarityRaster] = None

    @property
    def patch_size(self) -> int:
        if self.corpus is not None and "patch_size" in self.corpus:
            return int(self.corpus["patch_size"])
        sample = next(iter(self.index.metadata.values()), {})
        return int(sample.get("patch_size", 64))

    @property
    def stored_pool_radius(self) -> int:
        pooling = (self.corpus or {}).get("pooling") or {}
        return int(pooling.get("radius", 0) or 0)

    def summary(self) -> IndexSummary:
        return IndexSummary(
            name=self.name, path=self.path, patch_size=self.patch_size,
            cell_size_ft=self.raster.cell_size_ft if self.raster is not None else None,
            n_patches=len(self.index.metadata), stored_pool_radius=self.stored_pool_radius,
        )


_contexts: Dict[str, IndexContext] = {}
_default_name: Optional[str] = None
# A raster request is ~0.6 s of CPU (full scan + GeoTIFF encode) and ~10 MB of
# transient allocation. The handlers are sync so they run in the threadpool
# instead of blocking the event loop; this cap keeps a burst of requests from
# stacking scans (BLAS is already multithreaded, so concurrency buys nothing).
# Waiters are bounded too: past RASTER_QUEUE_TIMEOUT_S they get a 503 instead
# of holding a threadpool thread, so a flood cannot starve other handlers.
# Pooled /query requests are full scans as well and share the same slots.
_raster_slots = threading.BoundedSemaphore(2)
RASTER_QUEUE_TIMEOUT_S = float(os.environ.get("TERRAVECTOR_RASTER_QUEUE_TIMEOUT", "5"))


def _load_one(name: str, path: str) -> Optional[IndexContext]:
    idx_path = Path(path)
    if not idx_path.with_suffix(".idx").exists() and not idx_path.with_suffix(".meta").exists():
        return None
    ctx = IndexContext(name=name, path=str(idx_path), index=TerrainIndex.load(str(idx_path)))
    corpus_path = idx_path.with_suffix(".corpus.json")
    if corpus_path.exists():
        with open(corpus_path) as f:
            ctx.corpus = json.load(f)
        ctx.bridge = CoordBridge(
            corpus_metadata=ctx.corpus,
            index_metadata=ctx.index.metadata,
            patch_size=ctx.corpus.get("patch_size", 64),
        )
        # One chunked pass over the embeddings (~2 s for the county index).
        # Pooled norms are cached beside the index as <stem>.poolnorms_r<r>.npy.
        ctx.raster = SimilarityRaster(ctx.index, ctx.bridge, cache_prefix=idx_path.with_suffix(""))
    return ctx


def _load():
    global _contexts, _default_name
    _contexts = {}
    _default_name = None
    for name, path in _configured_indexes():
        ctx = _load_one(name, path)
        if ctx is None:
            continue
        _contexts[name] = ctx
        if _default_name is None:
            _default_name = name
    radii = [int(r) for r in os.environ.get("TERRAVECTOR_POOL_RADII", "").split(",") if r.strip()]
    for ctx in _contexts.values():
        if ctx.raster is None or ctx.stored_pool_radius:
            continue
        for r in radii:
            ctx.raster.pooled_norms(r)


@app.on_event("startup")
async def startup():
    _load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(name: Optional[str]) -> IndexContext:
    if not _contexts:
        raise HTTPException(status_code=503, detail="Index not loaded. Set TERRAVECTOR_INDEXES (or TERRAVECTOR_INDEX) or place an index at the default path.")
    key = name or _default_name
    if key not in _contexts:
        raise HTTPException(status_code=404, detail=f"Unknown index '{name}'. Available: {sorted(_contexts)}.")
    return _contexts[key]


def _ensure_bridge(ctx: IndexContext):
    if ctx.bridge is None:
        raise HTTPException(status_code=501, detail="Coordinate bridge unavailable — no corpus metadata found alongside index.")


def _ensure_raster(ctx: IndexContext):
    if ctx.raster is None:
        raise HTTPException(status_code=501, detail="Similarity raster unavailable — no corpus metadata found alongside index.")


def _radius(ctx: IndexContext, pool_radius: Optional[int]) -> int:
    r = int(pool_radius or 0)
    if r and ctx.stored_pool_radius:
        raise HTTPException(
            status_code=400,
            detail=f"Index '{ctx.name}' already stores vectors pooled over a "
                   f"{2 * ctx.stored_pool_radius + 1}x{2 * ctx.stored_pool_radius + 1} neighbourhood; "
                   "query-time pooling is only available on unpooled indices.",
        )
    if r:
        _ensure_raster(ctx)
    return r


def _resolve_query_patch(
    ctx: IndexContext,
    patch_id: Optional[int],
    lon: Optional[float],
    lat: Optional[float],
    x: Optional[float],
    y: Optional[float],
    crs: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    """Turn one of (patch_id | lon,lat | x,y[,crs]) into a patch id + echo of the query."""
    if patch_id is not None:
        if patch_id not in ctx.index.metadata:
            raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found in index '{ctx.name}'.")
        info: Dict[str, Any] = {"patch_id": patch_id}
        if ctx.bridge is not None:
            ctr = ctx.bridge.patch_id_to_state_plane_center(patch_id)
            if ctr is not None:
                info["coords"] = {"x": ctr[0], "y": ctr[1]}
        return patch_id, info

    _ensure_bridge(ctx)
    if lon is not None and lat is not None:
        sp_x, sp_y = ctx.bridge.wgs84_to_state_plane(lon, lat)
    elif x is not None and y is not None:
        if crs and crs.upper() == "EPSG:4326":
            sp_x, sp_y = ctx.bridge.wgs84_to_state_plane(x, y)
        else:
            sp_x, sp_y = x, y
    else:
        raise HTTPException(status_code=422, detail="Provide patch_id, (lon, lat), or (x, y).")

    resolved = ctx.bridge.state_plane_to_patch_id(sp_x, sp_y)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"No patch at State Plane ({sp_x:.1f}, {sp_y:.1f}). Point may be outside coverage or in a no-data area.",
        )
    return resolved, {"patch_id": resolved, "coords": {"x": sp_x, "y": sp_y}}


def _acquire_scan_slot():
    if not _raster_slots.acquire(timeout=RASTER_QUEUE_TIMEOUT_S):
        raise HTTPException(
            status_code=503,
            detail="Similarity scan is busy; retry shortly.",
            headers={"Retry-After": "2"},
        )


def _run_query(ctx: IndexContext, patch_id: int, k: int, radius: int):
    """k nearest patches: HNSW for unpooled queries, exact pooled scan otherwise."""
    if radius == 0:
        return ctx.index.query_by_id(patch_id, k=k)
    _acquire_scan_slot()
    try:
        return ctx.raster.top_k(patch_id, k=k, radius=radius)
    finally:
        _raster_slots.release()


def _similarity_raster_response(ctx: IndexContext, patch_id: int, value: str, radius: int) -> Response:
    _acquire_scan_slot()
    try:
        grid = ctx.raster.raster(patch_id, value=value, radius=radius)
        body = ctx.raster.to_geotiff_bytes(
            grid,
            tags={
                "TERRAVECTOR_QUERY_PATCH_ID": str(patch_id),
                "TERRAVECTOR_VALUE": value,
                "TERRAVECTOR_INDEX": ctx.name,
                "TERRAVECTOR_POOL_RADIUS": str(radius),
            },
        )
        del grid
    finally:
        _raster_slots.release()
    return Response(
        content=body,
        media_type="image/tiff",
        headers={
            "Content-Disposition": f'inline; filename="terravector_{ctx.name}_{value}_{patch_id}_r{radius}.tif"',
            "X-Terravector-Query-Patch-Id": str(patch_id),
            "X-Terravector-Value": value,
            "X-Terravector-Index": ctx.name,
            "X-Terravector-Pool-Radius": str(radius),
        },
    )


def _enrich_result(ctx: IndexContext, patch_id: int, distance: float, meta: Dict[str, Any]) -> PatchResult:
    """Build a PatchResult with spatial data when the bridge is available."""
    result = PatchResult(
        patch_id=patch_id,
        distance=distance,
        tile_id=meta.get("tile_id"),
    )

    if ctx.bridge is not None:
        sp = ctx.bridge.patch_id_to_state_plane_bbox(patch_id)
        if sp is not None:
            result.state_plane_bbox = BBox(x_min=sp[0], y_min=sp[1], x_max=sp[2], y_max=sp[3])

        wgs = ctx.bridge.patch_id_to_wgs84_bbox(patch_id)
        if wgs is not None:
            result.wgs84_bbox = WGS84BBox(lon_min=wgs[0], lat_min=wgs[1], lon_max=wgs[2], lat_max=wgs[3])

        ctr = ctx.bridge.patch_id_to_state_plane_center(patch_id)
        if ctr is not None:
            result.center = CoordsXY(x=ctr[0], y=ctr[1])

    return result


def _query_response(ctx: IndexContext, patch_id: int, query_info: Dict[str, Any], k: int, radius: int) -> QueryResponse:
    raw = _run_query(ctx, patch_id, k, radius)
    query_info = {**query_info, "index": ctx.name, "pool_radius": radius}
    return QueryResponse(
        query=query_info,
        results=[_enrich_result(ctx, pid, dist, meta) for pid, dist, meta in raw],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": bool(_contexts),
        "indexes": list(_contexts),
        "default_index": _default_name,
        "bridge_available": any(c.bridge is not None for c in _contexts.values()),
    }


@app.get("/info", response_model=IndexInfo)
async def info(index: Optional[str] = Query(default=None, description="Index name; default is the first configured")):
    ctx = _ctx(index)
    _ensure_bridge(ctx)

    stats = ctx.index.get_stats()
    county = ctx.corpus["county_metadata"]
    bounds = county["county_bounds"]
    wgs = ctx.bridge.get_county_bounds_wgs84()

    return IndexInfo(
        index=ctx.name,
        indexes=[c.summary() for c in _contexts.values()],
        n_patches=stats["n_patches"],
        embedding_dim=stats["embedding_dim"],
        space=stats["space"],
        crs=county.get("crs", "EPSG:3735"),
        patch_size=ctx.patch_size,
        stored_pool_radius=ctx.stored_pool_radius,
        county_bounds=BBox(
            x_min=bounds["x_min"],
            y_min=bounds["y_min"],
            x_max=bounds["x_max"],
            y_max=bounds["y_max"],
        ),
        county_bounds_wgs84=WGS84BBox(
            lon_min=wgs[0], lat_min=wgs[1], lon_max=wgs[2], lat_max=wgs[3]
        ),
        signature_types=ctx.corpus.get("signature_types", []),
        raster_grid=RasterGridInfo(**ctx.raster.grid_info()) if ctx.raster is not None else None,
    )


@app.post("/query/by-id", response_model=QueryResponse)
def query_by_id(req: QueryByIdRequest):
    ctx = _ctx(req.index)
    radius = _radius(ctx, req.pool_radius)
    patch_id, query_info = _resolve_query_patch(ctx, req.patch_id, None, None, None, None, None)
    return _query_response(ctx, patch_id, query_info, req.k, radius)


@app.post("/query/by-coords", response_model=QueryResponse)
def query_by_coords(req: QueryByCoordsRequest):
    ctx = _ctx(req.index)
    _ensure_bridge(ctx)
    radius = _radius(ctx, req.pool_radius)
    patch_id, query_info = _resolve_query_patch(ctx, None, req.lon, req.lat, req.x, req.y, req.crs)
    return _query_response(ctx, patch_id, query_info, req.k, radius)


@app.get(
    "/similarity/raster",
    responses={200: {"content": {"image/tiff": {}}, "description": "GeoTIFF, one cell per patch"}},
)
def similarity_raster_get(
    patch_id: Optional[int] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    lat: Optional[float] = Query(default=None),
    x: Optional[float] = Query(default=None),
    y: Optional[float] = Query(default=None),
    crs: Optional[str] = Query(default=None),
    value: str = Query(default="similarity", pattern="^(similarity|distance)$"),
    index: Optional[str] = Query(default=None, description="Index name (see /info)"),
    pool_radius: Optional[int] = Query(default=None, ge=0, le=MAX_POOL_RADIUS, description="Query-time pooling radius in cells"),
):
    """
    County-wide similarity map for one query location, as a GeoTIFF.

    Every patch in the index is scored against the query (exact cosine, not
    HNSW) and written to its cell on the patch grid: one cell per patch,
    ``patch_size * resolution_ft`` on a side, in the corpus CRS (EPSG:3735).
    Cells without a patch are NaN. ``value=similarity`` (default) gives cosine
    similarity in [-1, 1] with the query cell at 1; ``value=distance`` gives
    ``1 - similarity`` to match the ``/query`` endpoints. ``pool_radius=r``
    compares (2r+1)x(2r+1)-cell neighbourhood means instead of single cells.

    Loadable directly in GDAL/QGIS as
    ``/vsicurl_streaming/http://host/similarity/raster?patch_id=N`` (the body is
    generated per request, so the range-request ``/vsicurl/`` handler won't work).
    """
    ctx = _ctx(index)
    _ensure_raster(ctx)
    radius = _radius(ctx, pool_radius)
    pid, _ = _resolve_query_patch(ctx, patch_id, lon, lat, x, y, crs)
    return _similarity_raster_response(ctx, pid, value, radius)


@app.post(
    "/similarity/raster",
    responses={200: {"content": {"image/tiff": {}}, "description": "GeoTIFF, one cell per patch"}},
)
def similarity_raster_post(req: SimilarityRasterRequest):
    """JSON-body form of ``GET /similarity/raster``."""
    ctx = _ctx(req.index)
    _ensure_raster(ctx)
    radius = _radius(ctx, req.pool_radius)
    pid, _ = _resolve_query_patch(ctx, req.patch_id, req.lon, req.lat, req.x, req.y, req.crs)
    return _similarity_raster_response(ctx, pid, req.value, radius)
