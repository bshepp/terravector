"""Rasterize public terrain-class polygons onto the county cell grid.

Produces validation/class_grid_p<ps>.npz with, for every grid cell (raster
convention, row 0 = north, same indexing as evaluate_retrieval.Grid):

  classes      list of class names
  masks        bool (n_classes, n_cells): cell centre falls in the class
  poly_id      int32 (n_classes, n_cells): id of the source polygon (-1 none),
               used to keep a query's own polygon out of its scoring
  cell_ft, grid_rows, grid_cols

Sources (all public, fetched by validation/fetch_class_layers.py into --layers):
  ODNR Division of Geological Survey, Quaternary Geology of Ohio 1:500k
    (glacial landform units), ArcGIS FeatureServer, EPSG:3735 export.
  ODNR Division of Geological Survey, Surficial Geology stack-unit map
    1:24,000 (GeologyMapViewer layer 13; surface unit genesis code L1G:
    T/Ti till, SG/SGi outwash sand and gravel, IC/ICi ice-contact kame and
    esker deposits, LC lacustrine fines, a alluvium, E eolian sand, m made
    land, P and SsSh bedrock at the surface).
  USDA-NRCS SSURGO soil survey (1:12,000-1:24,000) map units labelled with
    the landform of their dominant component (validation/fetch_ssurgo_landforms.py).
  ODNR Division of Mineral Resources Management, Mines of Ohio
    (historic and permitted surface mines, coal and industrial minerals).
  OpenStreetMap via Overpass (railways, motorways, water bodies, dams,
    streams, quarries, landfills, cemeteries, golf courses, residential).
  validation/gt_sites.json (the nine verified earthworks).

    python validation/build_class_grid.py --layers <dir> --patch-size 128
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data" / "licking_county" / "licking_county.corpus.json"
FT_PER_DEG = None  # OSM comes in lon/lat; reprojected with pyproj below

# class name -> list of (file, selector, kind, buffer_ft)
#   selector: callable(properties) -> bool
#   kind: "poly" fills polygons; "line" buffers lines by buffer_ft
QUAT = "quaternary_units.geojson"
SM24 = "surficial24k_polygons.geojson"
SSURGO = "ssurgo_OH089_landforms.geojson"
CLASSES = {
    "end_moraine": [(QUAT, lambda p: p.get("UNIT_NAME") == "End moraine", "poly", 0)],
    "hummocky_moraine": [(QUAT, lambda p: p.get("UNIT_NAME") == "Hummocky moraine", "poly", 0)],
    "ground_moraine": [(QUAT, lambda p: p.get("UNIT_NAME") == "Ground moraine", "poly", 0)],
    "dissected_ground_moraine": [(QUAT, lambda p: p.get("UNIT_NAME") == "Dissected ground moraine", "poly", 0)],
    "kame_esker": [(QUAT, lambda p: p.get("UNIT_NAME") in ("Kames", "Kames and kame terraces", "Eskers"), "poly", 0)],
    "outwash_terrace": [(QUAT, lambda p: p.get("UNIT_NAME") in ("High-level outwash terraces", "Intermediate-level outwash terraces",
                                                                 "Low-level valley train outwash", "Outwash"), "poly", 0)],
    "alluvium": [(QUAT, lambda p: p.get("UNIT_NAME") == "Alluvium and Alluvial terraces", "poly", 0)],
    "lacustrine": [(QUAT, lambda p: p.get("UNIT_NAME") in ("Lacustrine clay", "Lacustrine deposits"), "poly", 0)],
    "colluvium_unglaciated": [(QUAT, lambda p: p.get("UNIT_NAME") == "Cenozoic Colluvium", "poly", 0)],
    "surface_mine": [(f, lambda p: True, "poly", 0) for f in
                     ("mines_hist_topo.geojson", "mines_surface_affected.geojson", "coal_surface_past.geojson", "indmin_surface_past.geojson")]
                    + [("osm_landuse.json", lambda p: p.get("landuse") == "quarry", "poly", 0)],
    "railway_active": [("osm_transport.json", lambda p: p.get("railway") == "rail", "line", 100)],
    "railway_abandoned": [("osm_transport.json", lambda p: p.get("railway") in ("abandoned", "disused", "dismantled"), "line", 100)],
    "motorway": [("osm_transport.json", lambda p: p.get("highway") in ("motorway", "trunk"), "line", 150)],
    "runway": [("osm_transport.json", lambda p: p.get("aeroway") == "runway", "line", 150)],
    "open_water": [("osm_water_bodies.json", lambda p: p.get("natural") == "water" or p.get("landuse") == "reservoir", "poly", 0)],
    "dam_embankment": [("osm_water_bodies.json", lambda p: p.get("waterway") == "dam" or p.get("man_made") in ("dam", "dyke", "embankment"), "line", 100)],
    "stream_channel": [("osm_streams.json", lambda p: p.get("waterway") in ("river", "stream"), "line", 100)],
    "landfill": [("osm_landuse.json", lambda p: p.get("landuse") == "landfill", "poly", 0)],
    "cemetery": [("osm_landuse.json", lambda p: p.get("landuse") == "cemetery", "poly", 0)],
    "golf_course": [("osm_landuse.json", lambda p: p.get("leisure") == "golf_course", "poly", 0)],
    "residential": [("osm_landuse.json", lambda p: p.get("landuse") == "residential", "poly", 0)],
    "earthwork": [("gt_sites", lambda p: True, "point", 0)],
    # ODNR 1:24,000 surficial stack-unit map, surface unit genesis (L1G)
    "sm24_till": [(SM24, lambda p: p.get("L1G") in ("T", "Ti"), "poly", 0)],
    "sm24_outwash": [(SM24, lambda p: p.get("L1G") in ("SG", "SGi", "S", "Si"), "poly", 0)],
    "sm24_ice_contact": [(SM24, lambda p: p.get("L1G") in ("IC", "ICi"), "poly", 0)],
    "sm24_lacustrine": [(SM24, lambda p: p.get("L1G") in ("LC", "L"), "poly", 0)],
    "sm24_alluvium": [(SM24, lambda p: p.get("L1G") == "a", "poly", 0)],
    "sm24_eolian": [(SM24, lambda p: p.get("L1G") == "E", "poly", 0)],
    "sm24_bedrock_surface": [(SM24, lambda p: p.get("L1G") in ("P", "SsSh", "Sh"), "poly", 0)],
    "sm24_made_land": [(SM24, lambda p: p.get("L1G") == "m" or p.get("LITH") in ("Pit", "Quarry"), "poly", 0)],
    # SSURGO map units by dominant-component landform (ss_* classes assigned by fetch_ssurgo_landforms.py)
    **{c: [(SSURGO, (lambda c: lambda p: p.get("cls") == c)(c), "poly", 0)] for c in
       ("ss_terrace", "ss_flood_plain", "ss_end_moraine", "ss_ground_moraine", "ss_lake_plain", "ss_hills_ridges",
        "ss_depression", "ss_kame_esker", "ss_outwash_plain", "ss_knoll_rise", "ss_reclaimed", "ss_flat")},
}


def osm_to_geoms(path, to_sp):
    """Overpass 'out geom' JSON -> list of (properties, shapely geometry in EPSG:3735)."""
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for e in data["elements"]:
        tags = e.get("tags", {})
        if e["type"] == "node":
            x, y = to_sp(e["lon"], e["lat"]); out.append((tags, Point(x, y)))
        elif e["type"] == "way" and e.get("geometry"):
            pts = [to_sp(g["lon"], g["lat"]) for g in e["geometry"]]
            closed = len(pts) > 3 and pts[0] == pts[-1]
            area_tag = any(k in tags for k in ("landuse", "natural", "leisure", "aeroway", "power")) and tags.get("aeroway") != "runway"
            if closed and area_tag:
                out.append((tags, make_valid(Polygon(pts))))
            else:
                out.append((tags, LineString(pts)))
        elif e["type"] == "relation" and e.get("members"):
            rings = []
            for m in e["members"]:
                if m.get("type") == "way" and m.get("role") == "outer" and m.get("geometry"):
                    pts = [to_sp(g["lon"], g["lat"]) for g in m["geometry"]]
                    if len(pts) > 3:
                        rings.append(make_valid(Polygon(pts)))
            if rings:
                out.append((tags, unary_union(rings)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    from src.coords import CoordBridge
    sys.path.insert(0, str(ROOT / "validation"))
    from evaluate_retrieval import Grid

    corpus = json.load(open(CORPUS))
    bridge = CoordBridge(corpus, {}, patch_size=a.patch_size)
    grid = Grid(bridge, a.patch_size)
    rr, cc = np.divmod(np.arange(grid.n), grid.cols)
    cx = bridge.x_min + (cc + 0.5) * grid.cell_ft
    cy = bridge.grid_y_max - (rr + 0.5) * grid.cell_ft
    centres = gpd.GeoSeries([Point(x, y) for x, y in zip(cx, cy)], crs="EPSG:3735")
    cells = gpd.GeoDataFrame({"cell": np.arange(grid.n)}, geometry=centres)

    to_sp = Transformer.from_crs("EPSG:4326", "EPSG:3735", always_xy=True).transform
    layers = Path(a.layers)
    cache = {}

    def load(fname):
        if fname in cache:
            return cache[fname]
        p = layers / fname
        if fname == "gt_sites":
            sites = json.load(open(ROOT / "validation" / "gt_sites.json"))
            feats = [(s, Point(s["x"], s["y"])) for s in sites]
        elif not p.exists():
            print(f"  missing layer {fname}; skipped")
            feats = []
        elif fname.endswith(".geojson"):
            g = gpd.read_file(p).set_crs(3735, allow_override=True)
            feats = [(dict(row.drop("geometry")), row.geometry) for _, row in g.iterrows() if row.geometry is not None]
        else:
            feats = osm_to_geoms(p, to_sp)
        cache[fname] = feats
        return feats

    names = list(CLASSES)
    masks = np.zeros((len(names), grid.n), dtype=bool)
    poly_id = np.full((len(names), grid.n), -1, dtype=np.int32)
    for ci, name in enumerate(names):
        geoms = []
        for fname, sel, kind, buf in CLASSES[name]:
            for props, g in load(fname):
                if not sel(props) or g is None or g.is_empty:
                    continue
                if kind == "line":
                    g = g.buffer(buf)
                elif kind == "point":
                    g = g.buffer(grid.cell_ft * 1.5)
                elif g.geom_type not in ("Polygon", "MultiPolygon"):
                    continue
                geoms.append(make_valid(g))
        if not geoms:
            print(f"{name}: no geometry"); continue
        # dissolve touching pieces so a "polygon" is one physical feature
        merged = unary_union(geoms)
        parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        gdf = gpd.GeoDataFrame({"pid": np.arange(len(parts))}, geometry=parts, crs="EPSG:3735")
        hit = gpd.sjoin(cells, gdf, how="inner", predicate="within")
        masks[ci, hit["cell"].values] = True
        poly_id[ci, hit["cell"].values] = hit["pid"].values
        print(f"{name:26s} {len(parts):5d} features  {int(masks[ci].sum()):7d} cells  ({100 * masks[ci].mean():.2f}% of grid)")

    out = a.out or str(ROOT / "validation" / f"class_grid_p{a.patch_size}.npz")
    np.savez_compressed(out, classes=np.array(names), masks=masks, poly_id=poly_id,
                        cell_ft=grid.cell_ft, grid_rows=grid.rows, grid_cols=grid.cols)
    print("saved", out)


if __name__ == "__main__":
    main()
