"""
Coordinate Bridge

Maps between coordinate systems used across terravector:
  WGS84 (lat/lon) <-> State Plane (EPSG:3735, ft) <-> tile grid <-> patch ID

Requires corpus metadata (from build_county_index.py) and index metadata
(from TerrainIndex) to resolve patch IDs.
"""

import math
from typing import Dict, Any, Optional, Tuple

from pyproj import Transformer


class CoordBridge:
    """
    Bidirectional coordinate mapping between WGS84, Ohio State Plane South,
    the county tile grid, and HNSW patch IDs.
    """

    def __init__(
        self,
        corpus_metadata: Dict[str, Any],
        index_metadata: Dict[int, Dict[str, Any]],
        patch_size: int = 64,
    ):
        county = corpus_metadata["county_metadata"]
        bounds = county["county_bounds"]

        self.x_min = bounds["x_min"]
        self.y_min = bounds["y_min"]
        self.x_max = bounds["x_max"]
        self.y_max = bounds["y_max"]
        self.tile_size_ft = county["tile_size_ft"]
        self.resolution_ft = county["resolution_ft"]
        self.grid_rows = county["grid_rows"]
        self.grid_cols = county["grid_cols"]
        self.crs = county.get("crs", "EPSG:3735")
        self.patch_size = patch_size

        self.tile_info: Dict[str, Dict[str, Any]] = corpus_metadata.get("tile_info", {})
        self.index_metadata = index_metadata

        self._to_sp = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._to_wgs = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        # Build lookup: (tile_id, y_start, x_start) -> global patch_id
        self._patch_lookup: Dict[Tuple[str, int, int], int] = {}
        for gid, meta in index_metadata.items():
            tid = meta.get("tile_id")
            if tid is not None:
                self._patch_lookup[(tid, meta["y_start"], meta["x_start"])] = gid

    # ------------------------------------------------------------------
    # CRS transforms
    # ------------------------------------------------------------------

    def wgs84_to_state_plane(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert WGS84 (lon, lat) to State Plane (x, y) in feet."""
        return self._to_sp.transform(lon, lat)

    def state_plane_to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """Convert State Plane (x, y) in feet to WGS84 (lon, lat)."""
        return self._to_wgs.transform(x, y)

    # ------------------------------------------------------------------
    # State Plane <-> tile + pixel
    # ------------------------------------------------------------------

    def state_plane_to_tile_pixel(
        self, x: float, y: float
    ) -> Optional[Tuple[str, int, int]]:
        """
        Map State Plane (x, y) to (tile_id, pixel_row, pixel_col).
        Returns None if the point falls outside the grid.
        """
        if x < self.x_min or x >= self.x_max or y <= self.y_min or y > self.y_max:
            return None

        col = int((x - self.x_min) / self.tile_size_ft)
        row = int((self.y_max - y) / self.tile_size_ft)

        col = min(col, self.grid_cols - 1)
        row = min(row, self.grid_rows - 1)

        tile_id = f"R{row:02d}_C{col:02d}"
        if tile_id not in self.tile_info:
            return None

        tile_x_origin = self.x_min + col * self.tile_size_ft
        tile_y_top = self.y_max - row * self.tile_size_ft

        px = int((x - tile_x_origin) / self.resolution_ft)
        py = int((tile_y_top - y) / self.resolution_ft)

        tile_shape = self.tile_info[tile_id].get("tile_shape")
        if tile_shape is not None:
            max_py = tile_shape[0] if isinstance(tile_shape, (list, tuple)) else int(tile_shape)
            max_px = tile_shape[1] if isinstance(tile_shape, (list, tuple)) else int(tile_shape)
            px = min(px, max_px - 1)
            py = min(py, max_py - 1)

        return tile_id, py, px

    def tile_pixel_to_state_plane(
        self, tile_id: str, py: int, px: int
    ) -> Tuple[float, float]:
        """Map (tile_id, pixel_row, pixel_col) back to State Plane (x, y)."""
        parts = tile_id.split("_")
        row = int(parts[0][1:])
        col = int(parts[1][1:])

        tile_x_origin = self.x_min + col * self.tile_size_ft
        tile_y_top = self.y_max - row * self.tile_size_ft

        x = tile_x_origin + px * self.resolution_ft
        y = tile_y_top - py * self.resolution_ft
        return x, y

    # ------------------------------------------------------------------
    # State Plane <-> patch ID
    # ------------------------------------------------------------------

    def state_plane_to_patch_id(self, x: float, y: float) -> Optional[int]:
        """
        Resolve a State Plane coordinate to its containing HNSW patch ID.
        Returns None if the point is outside coverage or no patch exists there.
        """
        result = self.state_plane_to_tile_pixel(x, y)
        if result is None:
            return None

        tile_id, py, px = result
        y_start = (py // self.patch_size) * self.patch_size
        x_start = (px // self.patch_size) * self.patch_size

        return self._patch_lookup.get((tile_id, y_start, x_start))

    def wgs84_to_patch_id(self, lon: float, lat: float) -> Optional[int]:
        """Convenience: WGS84 (lon, lat) straight to patch ID."""
        x, y = self.wgs84_to_state_plane(lon, lat)
        return self.state_plane_to_patch_id(x, y)

    # ------------------------------------------------------------------
    # Patch ID -> bounding boxes
    # ------------------------------------------------------------------

    def patch_id_to_state_plane_bbox(
        self, patch_id: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Return (x_min, y_min, x_max, y_max) in State Plane feet for a patch.
        """
        meta = self.index_metadata.get(patch_id)
        if meta is None:
            return None

        tile_id = meta.get("tile_id")
        if tile_id is None:
            return None

        y_start = meta["y_start"]
        x_start = meta["x_start"]
        ps = meta.get("patch_size", self.patch_size)

        # NW corner of the patch in State Plane
        x_nw, y_nw = self.tile_pixel_to_state_plane(tile_id, y_start, x_start)
        # SE corner
        x_se, y_se = self.tile_pixel_to_state_plane(tile_id, y_start + ps, x_start + ps)

        return (min(x_nw, x_se), min(y_nw, y_se), max(x_nw, x_se), max(y_nw, y_se))

    def patch_id_to_wgs84_bbox(
        self, patch_id: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Return (lon_min, lat_min, lon_max, lat_max) in WGS84 for a patch.
        """
        sp = self.patch_id_to_state_plane_bbox(patch_id)
        if sp is None:
            return None

        x_min, y_min, x_max, y_max = sp
        lon_min, lat_min = self.state_plane_to_wgs84(x_min, y_min)
        lon_max, lat_max = self.state_plane_to_wgs84(x_max, y_max)
        return (
            min(lon_min, lon_max),
            min(lat_min, lat_max),
            max(lon_min, lon_max),
            max(lat_min, lat_max),
        )

    def patch_id_to_state_plane_center(
        self, patch_id: int
    ) -> Optional[Tuple[float, float]]:
        """Return State Plane (x, y) of the patch center."""
        bbox = self.patch_id_to_state_plane_bbox(patch_id)
        if bbox is None:
            return None
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    # ------------------------------------------------------------------
    # Bounds helpers
    # ------------------------------------------------------------------

    def get_county_bounds_wgs84(self) -> Tuple[float, float, float, float]:
        """Return (lon_min, lat_min, lon_max, lat_max) for the full county extent."""
        lon_min, lat_min = self.state_plane_to_wgs84(self.x_min, self.y_min)
        lon_max, lat_max = self.state_plane_to_wgs84(self.x_max, self.y_max)
        return (
            min(lon_min, lon_max),
            min(lat_min, lat_max),
            max(lon_min, lon_max),
            max(lat_min, lat_max),
        )

    def get_county_bounds_state_plane(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) in State Plane feet."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)
