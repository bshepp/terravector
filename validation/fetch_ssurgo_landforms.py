"""SSURGO map-unit polygons with a landform class per map unit, from USDA Soil Data Access.

Soil surveys are mapped at 1:12,000-1:24,000, so their component landform
descriptions (cogeomordesc: terraces, flood plains, end moraines, ground
moraines, kames, ...) are a much finer landform label than the 1:500,000
Quaternary map. Each map unit takes the landform class of its dominant
component (highest representative percentage) when that component's landform
names map to one of the classes below.

Output: <out>/ssurgo_<AREA>_landforms.geojson in EPSG:3735 with properties
mukey, muname, compname, comppct, landforms (raw names), cls.

    python validation/fetch_ssurgo_landforms.py --area OH089 --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import requests

SDA = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

# ordered: the first rule whose names intersect the component's landform set wins
LANDFORM_CLASSES = [
    ("ss_kame_esker", {"kames", "eskers", "kame terraces", "kame moraines", "crevasse fillings", "ice-contact slopes"}),
    ("ss_end_moraine", {"end moraines", "moraines", "recessional moraines", "terminal moraines"}),
    ("ss_outwash_plain", {"outwash plains", "outwash deltas", "valley trains"}),
    ("ss_terrace", {"terraces", "outwash terraces", "stream terraces", "lake terraces"}),
    ("ss_flood_plain", {"flood plains", "flood-plain steps", "flood-plain splays", "natural levees", "backswamps", "oxbows"}),
    ("ss_lake_plain", {"lake plains", "lakebeds"}),
    ("ss_ground_moraine", {"ground moraines", "till plains"}),
    ("ss_depression", {"depressions", "kettles", "swales", "bogs", "marshes"}),
    ("ss_drainageway", {"drainageways", "draws", "swales", "gullies"}),
    ("ss_hills_ridges", {"hills", "ridges", "hillslopes", "hillsides", "breaks", "loess hills", "escarpments", "mountain slopes", "shoulders", "backslopes"}),
    ("ss_knoll_rise", {"knolls", "rises", "summits"}),
    ("ss_reclaimed", {"reclaimed lands", "fills", "anthroscapes", "spoil banks", "spoil piles", "mine spoil"}),
    ("ss_flat", {"flats"}),
]


def sda(query):
    for attempt in range(6):
        try:
            r = requests.post(SDA, json={"query": query, "format": "JSON+COLUMNNAME"}, timeout=300)
            return r.json()["Table"]
        except Exception as e:  # noqa: BLE001
            print("  sda retry", attempt, e, flush=True)
            time.sleep(15)
    raise RuntimeError("Soil Data Access unavailable")


def classify(landforms):
    s = set(landforms)
    for cls, names in LANDFORM_CLASSES:
        if s & names:
            return cls
    return None


def mapunit_classes(area):
    rows = sda(
        "SELECT m.mukey, m.muname, c.cokey, c.compname, c.comppct_r, g.geomfname "
        "FROM legend l JOIN mapunit m ON m.lkey=l.lkey JOIN component c ON c.mukey=m.mukey "
        "LEFT JOIN cogeomordesc g ON g.cokey=c.cokey AND g.rvindicator='Yes' "
        "AND g.geomftname IN ('Landform','Anthropogenic Landform','Anthroscape') "
        f"WHERE l.areasymbol='{area}' ORDER BY m.mukey, c.comppct_r DESC")[1:]
    comps = {}
    for mukey, muname, cokey, compname, pct, lf in rows:
        c = comps.setdefault(cokey, {"mukey": mukey, "muname": muname, "compname": compname, "comppct": int(pct or 0), "landforms": []})
        if lf:
            c["landforms"].append(lf)
    out = {}
    for c in sorted(comps.values(), key=lambda c: -c["comppct"]):
        if c["mukey"] in out:
            continue
        cls = classify(c["landforms"])
        out[c["mukey"]] = {**c, "cls": cls}
    return out


def polygons(area, cache):
    if cache.exists():
        return json.load(open(cache))
    rows, last, n = [], 0, 500
    while True:
        t = sda(f"SELECT TOP {n} mupolygonkey, mukey, mupolygongeo.STAsText() AS wkt FROM mupolygon "
                f"WHERE areasymbol='{area}' AND mupolygonkey > {last} ORDER BY mupolygonkey")[1:]
        rows += t
        print(f"  {len(rows)} polygons", flush=True)
        if len(t) < n:
            break
        last = int(t[-1][0])
    json.dump(rows, open(cache, "w"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="OH089", help="soil survey area symbol (default Licking County, Ohio)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from shapely import wkt
    from shapely.ops import transform
    from pyproj import Transformer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mu = mapunit_classes(a.area)
    counts = {}
    for v in mu.values():
        counts[v["cls"]] = counts.get(v["cls"], 0) + 1
    print("map units per class:", counts)
    to_sp = Transformer.from_crs("EPSG:4326", "EPSG:3735", always_xy=True).transform
    feats = []
    for key, mukey, w in polygons(a.area, out / f"ssurgo_{a.area}_polygons.json"):
        info = mu.get(str(mukey))
        if info is None or info["cls"] is None:
            continue
        g = transform(to_sp, wkt.loads(w))
        feats.append({"type": "Feature", "geometry": g.__geo_interface__,
                      "properties": {"mupolygonkey": key, "mukey": mukey, "muname": info["muname"], "compname": info["compname"],
                                     "comppct": info["comppct"], "landforms": "; ".join(info["landforms"]), "cls": info["cls"]}})
    path = out / f"ssurgo_{a.area}_landforms.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    print(path.name, len(feats), "classified polygons")


if __name__ == "__main__":
    main()
