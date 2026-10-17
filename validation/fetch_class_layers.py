"""Download the public layers used by build_class_grid.py into one directory.

All sources are open and need no account:
  * ODNR Division of Geological Survey, Quaternary Geology of Ohio 1:500,000
    and the Surficial Geology stack-unit map 1:24,000 (ArcGIS services;
    polygons exported in EPSG:3735)
  * ODNR Division of Mineral Resources Management, Mines of Ohio (surface
    mine polygons: historic from topo maps, surface affected areas, past coal
    and industrial-mineral surface mines)
  * OpenStreetMap via the Overpass API (railways, motorways, runways, water
    bodies, dams, streams, quarries, landfills, cemeteries, golf courses,
    residential), fetched in four queries with retries across mirrors.

    python validation/fetch_class_layers.py --out <dir> [--bbox W,S,E,N]

SSURGO landform polygons come from validation/fetch_ssurgo_landforms.py.
"""
import argparse
import json
import time
from pathlib import Path

import requests

ODNR = "https://gis.ohiodnr.gov/arcgis/rest/services/"
ARCGIS_LAYERS = {
    "quaternary_units.geojson": ODNR + "DGS_Services/Quaternary_Geology_500K_AGOL/FeatureServer/2",
    "surficial24k_polygons.geojson": ODNR + "DGS_Services/GeologyMapViewer/MapServer/13",
    "mines_hist_topo.geojson": ODNR + "MRM_Services/MinesOfOhio/MapServer/14",
    "mines_surface_affected.geojson": ODNR + "MRM_Services/MinesOfOhio/MapServer/8",
    "coal_surface_past.geojson": ODNR + "MRM_Services/MinesOfOhio/MapServer/13",
    "indmin_surface_past.geojson": ODNR + "MRM_Services/MinesOfOhio/MapServer/28",
}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
UA = {"User-Agent": "terravector-benchmark/0.1 (https://github.com/bshepp/terravector)"}


def osm_parts(b):
    """Overpass bodies keyed by output file; b is 'S,W,N,E'."""
    return {
        "osm_transport.json": f'(way["railway"~"^(rail|abandoned|disused|dismantled)$"]({b}); way["highway"~"^(motorway|trunk)$"]({b}); way["aeroway"="runway"]({b}););',
        "osm_water_bodies.json": f'(way["natural"="water"]({b}); relation["natural"="water"]({b}); way["landuse"="reservoir"]({b}); way["waterway"="dam"]({b}); way["man_made"~"^(dam|dyke|embankment)$"]({b}););',
        "osm_streams.json": f'(way["waterway"~"^(river|stream)$"]({b}););',
        "osm_landuse.json": f'(way["landuse"~"^(quarry|landfill|cemetery|residential|industrial)$"]({b}); relation["landuse"="quarry"]({b}); way["leisure"="golf_course"]({b}); way["power"="plant"]({b}););',
    }


def fetch_arcgis(url, out, env):
    feats, offset = [], 0
    while True:
        p = {"f": "geojson", "where": "1=1", "geometry": json.dumps(env), "geometryType": "esriGeometryEnvelope", "inSR": 4326,
             "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "returnGeometry": "true", "outSR": 3735,
             "resultOffset": offset, "resultRecordCount": 1000}
        j = requests.get(url + "/query", params=p, headers=UA, timeout=300).json()
        if "error" in j:
            print(out.name, "error", j["error"]); return
        feats += j.get("features", [])
        if len(j.get("features", [])) < 1000:
            break
        offset += 1000
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    print(out.name, len(feats), "features")


def fetch_overpass(body, out):
    ql = f"[out:json][timeout:180];{body}out geom;"
    for attempt in range(6):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": ql}, headers=UA, timeout=300)
            if r.status_code == 200 and r.content.startswith(b"{"):
                out.write_bytes(r.content); print(out.name, len(r.content), "bytes"); return
            print(out.name, url.split("/")[2], r.status_code)
        except Exception as e:  # noqa: BLE001
            print(out.name, "error", e)
        time.sleep(20)
    print("FAILED", out.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", default="-82.80,39.86,-82.16,40.28", help="W,S,E,N in degrees (default: Licking County)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    w, s, e, n = [float(v) for v in a.bbox.split(",")]
    env = {"xmin": w, "ymin": s, "xmax": e, "ymax": n, "spatialReference": {"wkid": 4326}}
    for fname, url in ARCGIS_LAYERS.items():
        fetch_arcgis(url, out / fname, env)
    for fname, body in osm_parts(f"{s},{w},{n},{e}").items():
        fetch_overpass(body, out / fname)


if __name__ == "__main__":
    main()
