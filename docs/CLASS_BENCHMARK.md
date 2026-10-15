# Multi-class terrain retrieval benchmark (Licking County, 2026-10-14)

terravector is meant to be a general query-by-example terrain search, not an
earthwork detector. The earthwork protocol in `VALIDATION_PLAN.md` tests one
hard class with nine positives. This benchmark asks the general question: when
a cell of some terrain type is the query, how much more often does that type
appear at the top of the ranked county than its base rate predicts, and for
which types does that work?

## Classes and labels

Twenty-two classes rasterized onto the county cell grid by
`validation/build_class_grid.py` from public layers fetched on 2026-10-14 by
`validation/fetch_class_layers.py` (no account needed for any of them):

| source | classes | note |
|---|---|---|
| ODNR Geological Survey, Quaternary Geology of Ohio 1:500,000 | end moraine, hummocky moraine, ground moraine, dissected ground moraine, kames and eskers, outwash terraces, alluvium, lacustrine, unglaciated colluvium | kilometre-scale map units; see "label scale" below |
| ODNR Mineral Resources, Mines of Ohio | surface mines (historic from topo maps, surface-affected areas, past coal and industrial-mineral surface mines) plus OSM quarries | 94 features, 821 cells |
| OpenStreetMap (Overpass) | active railway, abandoned railway, motorway/trunk, runway, open water, dam/embankment, stream channel, landfill, cemetery, golf course, residential | lines buffered 100-150 ft |
| `validation/gt_sites.json` | earthwork | 7 features, 60 cells |

A cell carries a class when its centre falls inside the (dissolved) polygon.
Cell counts are 22 (landfill) to 47,865 (ground moraine) on the 215,169-cell
128 px grid. Grids for 64 px and 256 px are built the same way
(`validation/class_grid_p{64,128,256}.npz`).

## Protocol

`validation/evaluate_classes.py`: for each class, up to 40 query cells spread
over its polygons (at most 4 per polygon); each query scores every placed
cell by cosine similarity (z-scored vectors, optional neighbourhood pooling),
drops cells within 1,500 ft and cells of the query's own polygon (own-polygon
exclusion is skipped when one connected feature holds over a quarter of the
class, i.e. for line networks), and measures the fraction of the top 1,000
remaining cells that carry the class (P@1000) against the class's base rate
among the remaining cells. Enrichment = P@1000 / base rate; 1x is random.
Intervals are 95% bootstrap over queries (2,000 replicates).
`validation/summarize_classes.py` prints the cross-configuration table.

## Results: top-1,000 enrichment, point [95% CI]

Full pipeline at 128 px with pooling radius 0 (none), 1 (3x3 = 960 ft),
3 (7x7 = 2,240 ft) and 7 (15x15 = 4,800 ft); the shipped 64 px index
unpooled; the plain-LRM analysis vector and simple statistics at 128 px,
3x3 pooled.

| class | cells | base % | full 128 | full 128, 3x3 (served) | full 128, 7x7 | full 128, 15x15 | index 64 | LRM 128, 3x3 | simple 128, 3x3 |
|---|---|---|---|---|---|---|---|---|---|
| end_moraine | 33074 | 14.5 | 1.4x [1.2,1.5] | 1.5x [1.3,1.7] | 1.6x [1.3,1.8] | 1.4x [1.1,1.7] | 1.2x [1.0,1.3] | 1.4x [1.2,1.6] | 1.5x [1.3,1.7] |
| hummocky_moraine | 40870 | 18.9 | 0.9x [0.7,1.1] | 1.0x [0.8,1.2] | 1.0x [0.7,1.3] | 1.0x [0.7,1.4] | 0.8x [0.7,1.0] | 1.0x [0.8,1.2] | 0.9x [0.7,1.1] |
| ground_moraine | 47865 | 21.9 | 1.2x [1.0,1.3] | 1.3x [1.1,1.5] | 1.4x [1.1,1.8] | 1.5x [1.2,1.9] | 1.4x [1.2,1.6] | 1.3x [1.1,1.6] | 1.3x [1.1,1.5] |
| dissected_ground_moraine | 40146 | 18.5 | 2.0x [1.6,2.3] | 2.0x [1.7,2.4] | 2.1x [1.7,2.6] | 2.0x [1.5,2.6] | 1.9x [1.5,2.3] | 2.1x [1.7,2.4] | 2.3x [1.9,2.8] |
| kame_esker | 3756 | 1.6 | 0.9x [0.8,1.1] | 1.0x [0.7,1.3] | 1.1x [0.7,1.5] | 1.3x [0.5,2.4] | 0.9x [0.7,1.0] | 1.1x [0.8,1.5] | 1.1x [0.8,1.4] |
| outwash_terrace | 16768 | 7.6 | 1.5x [1.2,1.8] | 2.1x [1.5,2.8] | 2.5x [1.8,3.2] | 3.0x [2.1,4.0] | 1.4x [1.1,1.7] | 1.9x [1.5,2.4] | 2.1x [1.6,2.6] |
| alluvium | 4482 | 1.8 | 1.9x [1.3,2.5] | 2.1x [1.3,3.1] | 3.9x [2.0,6.0] | 6.5x [3.7,9.5] | 1.9x [1.2,2.7] | 2.1x [1.3,2.9] | 2.3x [1.3,3.3] |
| lacustrine | 6884 | 3.0 | 1.1x [0.9,1.3] | 1.0x [0.7,1.3] | 1.2x [0.7,1.8] | 1.8x [0.9,2.9] | 1.6x [1.1,2.5] | 1.2x [0.9,1.6] | 1.7x [1.2,2.2] |
| colluvium_unglaciated | 20226 | 9.4 | 2.9x [2.1,3.7] | 3.7x [2.4,4.9] | 4.3x [2.8,5.7] | 4.9x [2.9,6.7] | 2.2x [1.3,3.3] | 3.0x [2.1,3.9] | 3.8x [2.7,4.9] |
| surface_mine | 821 | 0.36 | 7.6x [5.1,10.4] | 13.2x [9.1,17.5] | **17.4x [12.4,22.5]** | 9.9x [6.9,13.0] | 6.3x [4.0,8.8] | 13.3x [8.8,18.0] | 7.0x [5.1,9.2] |
| railway_active | 723 | 0.33 | **32.6x [16.7,50.6]** | 9.6x [3.3,18.9] | 6.3x [2.1,11.2] | 9.4x [1.1,17.7] | 20.9x [10.0,30.2] | 10.9x [4.2,17.7] | 6.8x [3.8,10.2] |
| railway_abandoned | 503 | 0.22 | 9.2x [5.7,12.8] | 2.8x [1.9,3.8] | 3.2x [2.1,4.4] | 2.7x [1.7,3.6] | **12.1x [7.1,18.1]** | 2.4x [1.7,3.2] | 2.5x [1.9,3.1] |
| motorway | 2552 | 1.2 | 8.6x [3.6,14.7] | 5.0x [2.5,7.6] | 2.5x [0.9,4.1] | 3.5x [2.6,4.3] | **14.4x [6.2,24.2]** | 8.5x [4.9,12.6] | 4.3x [2.2,6.4] |
| runway | 52 | 0.02 | 3.1x [1.1,5.6] | 4.0x [1.8,6.6] | 3.3x [0.9,6.3] | **10.0x [5.8,14.4]** | 4.5x [1.1,8.9] | 3.3x [1.3,5.7] | 1.6x [0.5,2.6] |
| open_water | 1511 | 0.69 | 7.3x [4.6,10.3] | 7.8x [4.2,12.0] | 9.2x [5.0,14.2] | 5.7x [2.9,9.2] | **12.5x [6.1,20.4]** | 6.2x [3.4,9.5] | 2.2x [1.7,2.8] |
| stream_channel | 4095 | 1.8 | **5.5x [3.8,7.3]** | 2.4x [1.5,3.5] | 1.4x [1.0,1.8] | 1.1x [0.8,1.3] | 5.5x [3.8,7.4] | 2.1x [1.5,2.9] | 2.3x [1.6,3.1] |
| landfill | 22 | 0.01 | 5.4x [0.0,13.4] | 0.0x | 2.1x [0.0,6.2] | 2.1x [0.0,6.2] | 2.7x [0.0,8.0] | 0.0x | 0.0x |
| cemetery | 180 | 0.08 | 2.2x [1.6,3.0] | 1.4x [1.0,1.9] | 0.9x [0.4,1.5] | 3.3x [1.3,5.5] | 2.1x [1.5,2.8] | 1.5x [1.0,2.0] | 1.2x [0.8,1.8] |
| golf_course | 882 | 0.38 | 3.1x [2.1,4.3] | 4.0x [2.5,5.8] | 3.6x [2.4,4.8] | 1.9x [0.8,3.3] | 2.8x [1.9,3.9] | 3.7x [2.2,5.5] | 1.7x [1.3,2.1] |
| residential | 2775 | 1.2 | 2.9x [2.1,3.7] | 3.8x [2.7,4.9] | 3.8x [2.8,5.0] | 4.0x [2.8,5.3] | 2.7x [1.9,3.6] | 3.5x [2.5,4.6] | 2.2x [1.8,2.7] |
| earthwork | 60 | 0.02 | 7.2x [4.2,10.7] | **21.9x [12.2,32.6]** | 19.3x [10.9,28.5] | 10.9x [6.2,16.5] | 6.5x [3.0,10.2] | 15.5x [8.6,23.9] | 6.3x [2.8,10.4] |
| **macro mean** | | | 5.2x | 4.4x | 4.4x | 4.2x | 5.8x | 4.1x | 2.6x |
| **macro geometric mean** | | | 3.2x | 2.1x | 2.9x | 3.2x | 3.5x | 2.0x | 1.6x |
| median percentile of same-class cells | | | 61.4 | 61.3 | 61.9 | 61.8 | 59.1 | 61.7 | 61.8 |

256 px is near random for every class (macro 1.2x unpooled, 1.5x pooled;
`validation/results/classes_full_p256*.json`) and is not shown.

## Reading

1. **Discrete, built or excavated features retrieve well; mapped glacial
   units do not.** Active railways (33x), surface mines (17x), earthworks
   (22x), motorways (14x at 64 px), open water (12x at 64 px), abandoned
   railways (12x at 64 px) and stream channels (5.5x) are retrieved an order
   of magnitude above base rate. Every till unit (ground, end, hummocky
   moraine, kames and eskers) sits at 0.9-1.5x in every configuration,
   including the simple statistics.
2. **The glacial failure is mostly label scale, with one real gap.** At
   320 ft the till, outwash, alluvium and lacustrine units are physically
   identical: median cell relief 2.0-2.8 ft and mean slope 0.12-0.19 for all
   of them (only dissected moraine and unglaciated colluvium, 10-13 ft, and
   kames, 4.9 ft, differ). The 1:500,000 map describes kilometre-scale
   assemblages, so a 128 px cell inside "end moraine" is usually flat
   farmland indistinguishable from ground moraine. Widening the pooling
   neighbourhood recovers the water-laid units (alluvium 1.9x -> 6.5x,
   outwash 1.5x -> 3.0x, colluvium 2.9x -> 4.9x at 15x15) but not the till
   units, and not kames and eskers (1.3x at best), which are tens of feet
   high and should be retrievable at these scales. That last one is the
   real gap to chase, and it needs a larger-scale map to be sure the
   polygons contain kames at all.
3. **Pooling radius is a property of the query class, not the index.**
   Linear features want no pooling (active railway 33x unpooled -> 10x at
   3x3; streams 5.5x -> 2.4x), compact features want 3x3-7x7 (surface
   mines 7.6x -> 17x; earthworks 7x -> 22x), broad units want 15x15. A
   general tool should expose the neighbourhood as a query parameter rather
   than bake 3x3 into the index.
4. **Scale is also class-dependent, and 64 px is not simply worse.** The
   shipped 64 px index wins on motorways, abandoned railways and open water
   and has the best macro mean (5.8x). 128 px wins on compact features
   (mines, earthworks, golf courses, residential). 256 px loses everywhere.
   Serving two resolutions (64 and 128 px) with selectable pooling covers
   the classes tested.
5. **The decomposition's gain over a plain local relief model is moderate
   and broad.** Macro geometric mean 3.2x (full) vs 2.8x (LRM unpooled) vs
   2.1x (simple); the largest gains are on open water, runways, earthworks,
   surface mines and motorways. Simple slope/curvature statistics are not
   random on this benchmark (railways 12x, colluvium 3.7x) because several
   classes are defined by relief.
6. **This is top-of-list enrichment, not class separation.** The median
   percentile of same-class cells is about 61 for every configuration (50 is
   random): most cells of a class are not retrieved by a single query. The
   tool ranks candidates; it does not segment the county.

## Confusions worth knowing (served configuration, top-1,000 composition)

Surface mines pull alluvium (5x) and stream channels (5x); railways pull
motorways (6x) and each other; earthworks pull golf courses (7x) and surface
mines (6x); residential pulls outwash terraces (3x, which is where towns
were built); landfills retrieve dissected uplands. The 1:500k units confuse
freely among themselves.

## Known limits

- Labels are map polygons, not per-cell truth: a "surface mine" polygon
  includes unmined land, OSM "residential" includes open fields, lines are
  buffered 100-150 ft. Measured enrichments are therefore lower bounds.
- Tiny classes (landfill 22 cells, dam 6-25 cells, runway 52) have
  intervals that span 0.
- The earthwork class has 60 cells from 7 features and shares the limits of
  `VALIDATION_PLAN.md`.
- Queries are single cells; a user would often query with a drawn polygon
  (mean of its cells), which should only help.

## Data that would improve this benchmark

- **ODNR 1:24,000 surficial geology** (quadrangle maps of unconsolidated
  deposits, or the county surficial-geology series) to replace the 1:500k
  units for kames, eskers, outwash and terraces.
- **NRCS SSURGO / gSSURGO** map units with landform attributes (terrace,
  floodplain, kame, moraine, upland) at 1:12,000-1:24,000.
- **NHDPlus High Resolution** for channel and floodplain classes instead of
  OpenStreetMap streams.
- **ODOT landslide inventory** (Office of Geotechnical Engineering) for a
  mass-wasting class; not public by download.
- **Ohio Archaeological Inventory** (OAI) for more earthworks; access
  pending.
