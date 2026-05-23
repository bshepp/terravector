#!/usr/bin/env python
"""
terravector REST API

FastAPI service exposing terrain similarity queries over HTTP.

Usage:
    uvicorn api:app --reload --host 127.0.0.1 --port 8000

Docs at http://127.0.0.1:8000/docs
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from src.index import TerrainIndex
from src.coords import CoordBridge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_PATH = os.environ.get(
    "TERRAVECTOR_INDEX",
    str(Path(__file__).parent / "data" / "licking_county" / "licking_county.idx"),
)

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


class QueryByIdRequest(BaseModel):
    patch_id: int
    k: int = Field(default=10, ge=1, le=100)


class QueryByCoordsRequest(BaseModel):
    lon: Optional[float] = None
    lat: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    crs: Optional[str] = None
    k: int = Field(default=10, ge=1, le=100)


class IndexInfo(BaseModel):
    n_patches: int
    embedding_dim: int
    space: str
    crs: str
    patch_size: int
    county_bounds: BBox
    county_bounds_wgs84: WGS84BBox
    signature_types: List[str]


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="terravector",
    description="Terrain patch similarity search API",
    version="0.1.0",
)

_index: Optional[TerrainIndex] = None
_bridge: Optional[CoordBridge] = None
_corpus: Optional[Dict[str, Any]] = None


def _load():
    global _index, _bridge, _corpus

    idx_path = Path(INDEX_PATH)
    if not idx_path.with_suffix(".idx").exists() and not idx_path.with_suffix(".meta").exists():
        return

    _index = TerrainIndex.load(str(idx_path))

    corpus_path = idx_path.with_suffix(".corpus.json")
    if corpus_path.exists():
        with open(corpus_path) as f:
            _corpus = json.load(f)
        _bridge = CoordBridge(
            corpus_metadata=_corpus,
            index_metadata=_index.metadata,
            patch_size=_corpus.get("patch_size", 64),
        )


@app.on_event("startup")
async def startup():
    _load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_index():
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Set TERRAVECTOR_INDEX or place index at default path.")


def _ensure_bridge():
    if _bridge is None:
        raise HTTPException(status_code=501, detail="Coordinate bridge unavailable — no corpus metadata found alongside index.")


def _enrich_result(patch_id: int, distance: float, meta: Dict[str, Any]) -> PatchResult:
    """Build a PatchResult with spatial data when the bridge is available."""
    result = PatchResult(
        patch_id=patch_id,
        distance=distance,
        tile_id=meta.get("tile_id"),
    )

    if _bridge is not None:
        sp = _bridge.patch_id_to_state_plane_bbox(patch_id)
        if sp is not None:
            result.state_plane_bbox = BBox(x_min=sp[0], y_min=sp[1], x_max=sp[2], y_max=sp[3])

        wgs = _bridge.patch_id_to_wgs84_bbox(patch_id)
        if wgs is not None:
            result.wgs84_bbox = WGS84BBox(lon_min=wgs[0], lat_min=wgs[1], lon_max=wgs[2], lat_max=wgs[3])

        ctr = _bridge.patch_id_to_state_plane_center(patch_id)
        if ctr is not None:
            result.center = CoordsXY(x=ctr[0], y=ctr[1])

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "bridge_available": _bridge is not None,
    }


@app.get("/info", response_model=IndexInfo)
async def info():
    _ensure_index()
    _ensure_bridge()

    stats = _index.get_stats()
    county = _corpus["county_metadata"]
    bounds = county["county_bounds"]
    wgs = _bridge.get_county_bounds_wgs84()

    sample_meta = next(iter(_index.metadata.values()), {})

    return IndexInfo(
        n_patches=stats["n_patches"],
        embedding_dim=stats["embedding_dim"],
        space=stats["space"],
        crs=county.get("crs", "EPSG:3735"),
        patch_size=sample_meta.get("patch_size", 64),
        county_bounds=BBox(
            x_min=bounds["x_min"],
            y_min=bounds["y_min"],
            x_max=bounds["x_max"],
            y_max=bounds["y_max"],
        ),
        county_bounds_wgs84=WGS84BBox(
            lon_min=wgs[0], lat_min=wgs[1], lon_max=wgs[2], lat_max=wgs[3]
        ),
        signature_types=_corpus.get("signature_types", []),
    )


@app.post("/query/by-id", response_model=QueryResponse)
async def query_by_id(req: QueryByIdRequest):
    _ensure_index()

    if req.patch_id not in _index.metadata:
        raise HTTPException(status_code=404, detail=f"Patch {req.patch_id} not found in index.")

    raw = _index.query_by_id(req.patch_id, k=req.k)

    query_info: Dict[str, Any] = {"patch_id": req.patch_id}
    if _bridge is not None:
        ctr = _bridge.patch_id_to_state_plane_center(req.patch_id)
        if ctr is not None:
            query_info["coords"] = {"x": ctr[0], "y": ctr[1]}

    return QueryResponse(
        query=query_info,
        results=[_enrich_result(pid, dist, meta) for pid, dist, meta in raw],
    )


@app.post("/query/by-coords", response_model=QueryResponse)
async def query_by_coords(req: QueryByCoordsRequest):
    _ensure_index()
    _ensure_bridge()

    # Resolve to State Plane
    if req.lon is not None and req.lat is not None:
        sp_x, sp_y = _bridge.wgs84_to_state_plane(req.lon, req.lat)
    elif req.x is not None and req.y is not None:
        if req.crs and req.crs.upper() == "EPSG:4326":
            sp_x, sp_y = _bridge.wgs84_to_state_plane(req.x, req.y)
        else:
            sp_x, sp_y = req.x, req.y
    else:
        raise HTTPException(status_code=422, detail="Provide (lon, lat) or (x, y).")

    patch_id = _bridge.state_plane_to_patch_id(sp_x, sp_y)
    if patch_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"No patch at State Plane ({sp_x:.1f}, {sp_y:.1f}). Point may be outside coverage or in a no-data area.",
        )

    raw = _index.query_by_id(patch_id, k=req.k)

    return QueryResponse(
        query={"patch_id": patch_id, "coords": {"x": sp_x, "y": sp_y}},
        results=[_enrich_result(pid, dist, meta) for pid, dist, meta in raw],
    )
