# Validation plan: does residual-texture similarity find earthworks?

Status as of 2026-10-13. This document collects (1) the ground-truth site list
for Licking County, (2) the first corrected retrieval probe, (3) how comparable
work in lidar archaeology and terrain retrieval is done, and (4) public datasets
the method could be tested on beyond this county.

## 0. A georeferencing bug invalidated everything before today

`RESIDUALS/tile_county.py` numbers tile rows from the **south** and stores each
tile DEM **south-up** (array row 0 is the tile's southern edge). `CoordBridge`
and `SimilarityRaster` assumed north-down. Every coordinate lookup, every
"Great Circle" query (patch 588007), the raster endpoint's geotransform, and the
Oct 11/12 similarity rasters were vertically mirrored inside the grid. Fixed in
`src/coords.py` and `src/similarity_raster.py`; pinned by `tests/test_coords.py`.
The correct Great Circle centre (Eagle Mound) is patch **311267** in tile
R04_C09; the eastern ring wall is patch **311270**.

The bug was found by rendering the DEM around each candidate site. With the old
convention the Great Circle chip showed flat farmland; with the corrected one
the 1,200 ft ring, the Octagon, Observatory Mound, Alligator Mound, Dixon Mound
and the Flint Ridge pit field are all unmistakable.

## 1. Ground truth

`data/ground_truth/licking_county_sites.csv` holds 17 rows. Nine have
coordinates verified against the county DEM (Great Circle centre and two wall
points, Octagon/Observatory Circle junction, Observatory Mound, Wright
Earthworks wall remnant, Alligator Mound, Dixon Mound, Flint Ridge pit field),
one is approximate (Great Hopewell Road), and seven are known sites without
usable coordinates: Infirmary Mound (somewhere in a 316-acre park), Salt Run
Park mound, Etna Township Mound I, McDaniel Mound, Melick Mound (the last three
are NRHP address-restricted), Salisbury Hill Fort 1 (33Li7), and the destroyed
Cherry Valley Ellipse, which will make a good negative control once its
footprint is georeferenced from the 1862 Salisbury map.

`data/ground_truth/ross_warren_county_sites.csv` lists the six other Hopewell
Ceremonial Earthworks (UNESCO) components as a held-out set. USGS 3DEP Ohio
Phase 3 (2020-2023, QL1, 8 pts/m²) covers Ross and Warren counties; Licking is
in Phase 2 (QL1, work unit 224909).

Getting more positives requires the Ohio Archaeological Inventory. Site
locations are restricted to approved subscribers of the SHPO Online Mapping
System (agencies, scholars, preservation professionals; five-day vetting).
Contact: ArchSurvey@ohiohistory.org. The 1848 Squier & Davis plates and the
1862 Salisbury survey are public and have been georeferenced by Ohio History
Connection for Newark; the same can be done for the rest of the county.

## 2. First corrected probe (2026-10-13)

Exact cosine over all 858,642 patches (`SimilarityRaster.scores`), signed-log1p
index of 2026-10-13. For each query, the number is the percentile of the best
cell within a 3x3 patch neighbourhood of the target, in the query's county-wide
score distribution. A random location's 3x3-best sits at about the 77th
percentile by construction.

| query \ target | GC wall W | Octagon neck | Observatory Md | Alligator | Wright wall | Dixon Md | Flint Ridge | ctl Newark | ctl farmland | ctl lake | ctl hill | ctl ridge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Great Circle wall E | 100.0 | 99.7 | 99.9 | 99.6 | 78.3 | 86.9 | 93.6 | 27.4 | 89.3 | 63.1 | 92.7 | 87.9 |
| Great Circle wall W | 100.0 | 99.7 | 99.6 | 97.7 | 86.1 | 90.6 | 94.8 | 29.7 | 97.4 | 60.6 | 89.0 | 87.7 |
| Octagon neck | 79.0 | 100.0 | 100.0 | 95.4 | 98.0 | 99.2 | 98.6 | 82.9 | 51.2 | 96.6 | 88.8 | 94.9 |
| Observatory Mound | 100.0 | 100.0 | 100.0 | 99.8 | 89.0 | 97.2 | 73.8 | 43.5 | 89.8 | 72.0 | 98.1 | 97.6 |
| Alligator Mound | 91.8 | 94.8 | 93.6 | 100.0 | 95.7 | 97.6 | 80.8 | 59.9 | 95.4 | 92.7 | 96.6 | 92.9 |

Reading: wall-type queries (Great Circle, Observatory Mound) put the other
Newark walls and the Alligator effigy at the 99.6-100th percentile, while the
five hand-picked controls range 27-98. That looked encouraging, but the full
protocol below shows how little a 99.6th-percentile block means.

### Full protocol (2026-10-13/14, `validation/evaluate_retrieval.py`)

All nine ground-truth sites as queries, own site masked within 1,500 ft, 500
uniformly random 3x3 control blocks per query. "Other sites" are the eight
sites not sharing the query's own complex. Hits are counted at fractions of
the county so 64 px and 128 px grids are comparable (top 0.1% = ~860 cells at
64 px, ~215 cells at 128 px). Baselines were computed on an EC2 c7a.8xlarge
(`validation/ec2_run.sh`; 53 + 7 minutes, under $3) from the same tiles.

| vectors | patch | other sites: mean pct | random block median | wall-to-wall pct | sites in top 0.1% (of 8) | top 0.5% | mean AP |
|---|---|---|---|---|---|---|---|
| simple terrain stats (12-d) | 64 | 85.0 | 85.2 | 96.3 | 0.00 | 0.56 | 0.0001 |
| simple terrain stats, 3x3 pooled | 64 | 71.0 | 70.8 | 85.2 | 0.22 | 0.33 | 0.0001 |
| analysis vector on plain LRM (40-d) | 64 | 91.4 | 86.2 | 93.7 | 0.22 | 1.44 | 0.0004 |
| same, 3x3 pooled | 64 | 84.8 | 72.6 | 95.8 | 0.11 | 0.33 | 0.0001 |
| **production index (640-d)** | 64 | 90.0 | 85.3 | 94.2 | 0.56 | 1.11 | 0.0004 |
| production index, 3x3 pooled | 64 | 78.2 | 70.0 | 98.3 | 0.33 | 1.22 | 0.0003 |
| simple terrain stats | 128 | 82.7 | 83.5 | 94.4 | 0.00 | 0.33 | 0.0005 |
| simple terrain stats, pooled | 128 | 74.0 | 66.5 | 96.1 | 0.22 | 0.56 | 0.0017 |
| analysis vector on plain LRM | 128 | 92.1 | 86.6 | 95.1 | 0.44 | 1.33 | 0.0018 |
| same, pooled | 128 | 88.9 | 71.6 | 96.6 | 0.33 | 1.00 | 0.0030 |
| **full residuals pipeline (640-d)** | 128 | 92.5 | 86.6 | 98.9 | 1.00 | 1.67 | 0.0048 |
| **full residuals pipeline, pooled** | 128 | 89.5 | 69.5 | 99.5 | 1.11 | 1.89 | 0.0131 |

Wall-to-wall pairs scoring at or above the 99.5th percentile (16 ordered pairs
among the five wall sites): full pipeline 128 px pooled **10/16**, unpooled
7/16; production index 64 px 7-9/16; plain-LRM analysis vector 2-4/16; simple
statistics 1-3/16. Per-query tables are in `validation/results/`.

### Determination (2026-10-14)

1. **There is a real signal, and it is specific to earthwork walls.** At 128 px
   (320 ft) the full pipeline puts one of eight other sites in the top 0.1% of
   the county per query, against roughly 0.7% expected for a random 3x3 block,
   a 21x enrichment (31x with 3x3 pooling) against uniform controls, and 6x
   (8x pooled, 95% CI 3-16x) against landform-matched controls (see the two
   tables below; most of the uniform-control enrichment is terrace-like
   landform, but the remainder is specific to the earthworks). Walls retrieve walls across sites 15 miles apart
   (the Wright remnant retrieves Dixon Mound at 100.0 and the Great Circle at
   99.5; downtown Newark on the same terrace scores 60-70), so it is not just
   "Newark terrace texture". Conical mounds are retrieved only partly, the
   effigy and the pit field not at all.
2. **The decomposition x upsampling product is the contribution.** At the same
   scale and with the same 20 statistics, a plain local relief model gets 0.44
   top-0.1% hits and 3/16 wall pairs; the full pipeline gets 1.00-1.11 and
   7-10/16. Simple slope/curvature statistics are indistinguishable from random.
3. **Scale is the biggest lever and the shipped index is at the wrong one.**
   Going from 64 px to 128 px roughly doubles every retrieval measure; 3x3
   pooling adds a little more. The 64 px county index is the weakest
   configuration of the pipeline tested.
4. **It is not yet a usable detector.** Mean average precision is 0.005-0.013;
   most sites are not in each other's top 0.1%. It is a candidate ranker that
   enriches the top of the list roughly 8x beyond landform for wall-type
   features, with nine positives and therefore wide confidence intervals.

### Multi-scale (2026-10-14)

Full-pipeline vectors at 32 px and 256 px were computed on a c7a.16xlarge
(`validation/ec2_run.sh` pattern, 2 h 10 min, about $7; the 32 px pass is
3.5M patches and took 1 h 46 min of that) and evaluated with the same
protocol. The 32 px grid has 3.4M cells, so it was scored on an r7a.4xlarge
(4 minutes) after `validation/evaluate_retrieval.py` was made memory-lean.
Mean AP is **not comparable across scales** (it depends on absolute ranks and
the 32 px grid has 16x more cells than 128 px); the fraction-of-county
measures and the enrichment over random 3x3 blocks are the comparable ones.
Enrichment = (fraction of other sites whose best block cell is in the top
0.1% / 0.5% of the county) / (same fraction for 500 random blocks).

| full-pipeline vectors | cells | other sites: mean pct | random block median | wall-to-wall pct | top 0.1% enrichment | top 0.5% enrichment | wall pairs >= 99.5 | mean AP |
|---|---|---|---|---|---|---|---|---|
| 32 px (80 ft) | 3.4M | 90.5 | 76.1 | 90.1 | 18.4x | 7.1x | 6/16 | 0.0001 |
| 32 px, pooled | 3.4M | 82.9 | 62.9 | 85.8 | 20.0x | 5.8x | 2/16 | 0.0001 |
| 32 px + 128 px parent (1280-d) | 3.4M | 91.1 | 74.1 | 92.6 | 21.6x | 6.8x | 7/16 | 0.0001 |
| 32 px + 128 px parent, pooled | 3.4M | 83.7 | 61.8 | 93.7 | 18.5x | 10.6x | 6/16 | 0.0001 |
| 64 px (production index) | 859k | 90.0 | 85.3 | 94.2 | 5.9x | 4.0x | 7/16 | 0.0004 |
| 64 px, pooled | 859k | 78.2 | 70.0 | 98.3 | 7.0x | 7.3x | 9/16 | 0.0003 |
| **128 px (320 ft)** | 215k | 92.5 | 86.6 | 98.9 | 21.2x | 7.1x | 7/16 | 0.0048 |
| **128 px, pooled (served)** | 215k | 89.5 | 69.5 | **99.5** | **31.1x** | 12.4x | **10/16** | 0.0131 |
| 128 px + 256 px parent (1280-d) | 202k | 91.8 | 82.1 | 98.9 | 18.9x | 9.8x | 10/16 | 0.0107 |
| 128 px + 256 px parent, pooled | 202k | 90.8 | 69.1 | 99.0 | 19.8x | **13.2x** | 10/16 | **0.0183** |
| 256 px (640 ft) | 51k | 87.8 | 86.5 | 93.8 | 7.5x | 4.7x | 4/16 | 0.0047 |
| 256 px, pooled | 51k | 80.4 | 72.5 | 86.1 | 2.3x | 4.4x | 3/16 | 0.0022 |
| plain-LRM analysis vector, 32 px | 3.4M | 92.1 | 81.8 | 91.2 | 14.3x | 8.5x | 5/16 | 0.0001 |
| plain-LRM analysis vector, 256 px | 51k | 84.0 | 86.7 | 85.5 | 0.0x | 1.4x | 1/16 | 0.0017 |

Reading:

- **128 px is the carrier scale.** It has the best wall-to-wall percentile,
  the most wall pairs above 99.5 and the highest top-0.1% enrichment, and it
  is the only scale where 3x3 pooling helps rather than hurts. At 256 px a
  pooled block is 1,920 ft across and averages the earthwork away; at 32 px a
  single cell is a slice of wall and pooling to 240 ft blocks loses the
  cross-section without reaching the 320 ft context that 128 px has.
- **32 px alone roughly matches unpooled 128 px on top-0.1% enrichment
  (18-22x)** but is far weaker on wall-to-wall percentile (90 vs 99) and wall
  pairs (6 vs 7-10 of 16), at 16x the index size. Appending the 128 px parent
  vector to each 32 px cell does not recover the 128 px result.
- **Appending the 256 px parent to 128 px gives the best mean AP (0.018
  pooled) and top-0.5% enrichment (13.2x)** while keeping 10/16 wall pairs,
  but the top-0.1% enrichment falls from 31x to 20x and the AP gain comes from
  two queries (Eagle Mound 0.034 -> 0.071, Great Circle east wall 0.001 ->
  0.005). With nine queries that is not a reliable improvement, and it doubles
  the vector width, so no multi-scale index is being built from it.
- **The 64 px dip is unexplained.** The 32 px and 128 px rows come from
  `validation/baseline_features.py`; the 64 px rows come from the production
  `build_county_index.py` run, so the dip could be a scale effect or a
  difference between the two code paths. A 64 px `baseline_features.py` pass
  (about 25 minutes on the same instance) would settle it; it has not been run.
- **The plain-LRM vector tracks the full pipeline's scale curve** (14x at
  32 px, 7-9x at 128 px, ~0 at 256 px) at a lower level, so the scale
  behaviour is a property of the statistics, and the decomposition's gain
  (2-3x at 128 px) is on top of it.

Decision: keep serving `licking_county_p128_pool3.idx`. Multi-scale is parked
until more positives exist to tell a two-query AP gain from noise.

### Landform-matched controls and confidence intervals (2026-10-14)

The uniform random controls above sample the whole county, including ravines,
hillsides and floodplain, while every ground-truth site sits on gently sloping
terrace or upland. Part of the enrichment could therefore be "looks like a
terrace" rather than "looks like an earthwork". To separate the two,
`evaluate_retrieval.py --landform` draws 500 extra control blocks per query
from the 5% of county cells nearest the query in a four-dimensional landform
space (log relief std, log mean slope, log TPI-32 std, log plane tilt, each
z-scored; taken from the 12-d simple terrain statistics at the same patch
size). `validation/bootstrap_ci.py` resamples the nine queries with
replacement (5,000 replicates) for 95% intervals; the control blocks are not
the dominant source of uncertainty, the query count is.

| vectors (128 px) | random block median pct | matched block median pct | top 0.1% enrichment vs uniform | vs landform-matched | top 0.5% vs uniform | vs matched | wall-to-wall pct |
|---|---|---|---|---|---|---|---|
| full pipeline | 86.6 | 94.7 | 21x [4, 50] | 6x [1, 11] | 7x [4, 11] | 2x [1, 4] | 98.9 [98.7, 99.2] |
| **full pipeline, pooled (served)** | 69.5 | 90.0 | 31x [12, 52] | **8x [3, 16]** | 12x [5, 20] | 4x [2, 6] | 99.5 [99.1, 99.8] |
| full + 256 px parent, pooled | 69.1 | 87.5 | 20x [6, 42] | 11x [4, 20] | 13x [7, 21] | 5x [3, 10] | 99.0 [98.5, 99.4] |
| plain-LRM analysis vector | 86.6 | 95.2 | 7x [0, 19] | 2x [0, 6] | 5x [1, 10] | 2x [0, 4] | 95.1 [92.0, 98.2] |
| plain-LRM, pooled | 71.5 | 89.9 | 8x [0, 16] | 3x [0, 5] | 6x [4, 8] | 2x [1, 3] | 96.6 [93.1, 98.9] |
| simple terrain stats, pooled | 66.4 | 88.9 | 6x [0, 17] | 1x [0, 3] | 4x [0, 8] | 1x [0, 2] | 96.1 [92.7, 97.9] |
| production 64 px index, pooled | 70.0 | 81.9 | 7x [0, 13] | 5x [0, 7] | 7x [3, 13] | 4x [2, 8] | 98.3 [97.0, 99.4] |

Sensitivity for the served configuration: drawing matched controls from the
nearest 2% / 5% / 10% of the county gives 7.5x / 8.3x / 10.1x at the top
0.1%; matching on eight of the twelve simple statistics instead of four gives
6.2x.

Per-class fraction of other sites at or above the 99.5th percentile (served
configuration): wall->wall 0.62 (16 pairs), wall->mound 0.23 (13),
mound->wall 0.15 (13), mound->mound 0.17 (6), wall->pits 0.20 (5), nothing
retrieves the Flint Ridge pits and the pits retrieve nothing.

Reading:

- **About three quarters of the uniform-control enrichment was landform.** A
  random terrace-like block scores at the 90th-95th percentile against an
  earthwork query, against the 70th-87th for a uniformly random block. The
  simple-statistics vector, which is built from the same quantities the
  controls are matched on, drops to 1x as it should, which is the check that
  the matching works.
- **A signal beyond landform remains and its interval excludes 1x.** The
  served configuration enriches the top 0.1% 8x [3x, 16x] against
  landform-matched controls; the plain-LRM vector 3x [0x, 5x]; the 64 px
  index 5x [0x, 7x]. The decomposition's advantage over the plain LRM
  (about 3x) survives matching.
- **The 128+256 concatenation looks better under matched controls (11x
  [4x, 20x], 5x at the top 0.5%)** than under uniform ones, which is
  consistent with the 256 px parent adding context that is not landform. The
  intervals overlap the served configuration's heavily and the gain still
  rests on two queries, so it stays parked, but it is the first candidate to
  re-test when more positives exist.
- **Intervals are wide because there are nine queries.** Every "x vs y"
  comparison between configurations overlaps at 95%; only the comparisons
  against 1x (no enrichment) and against the simple statistics are clear.
  The matched controls may also contain unrecorded earthworks, which would
  deflate the enrichment, not inflate it.

### Next steps, in order of value per hour

- Done 2026-10-14: the general multi-class benchmark (22 terrain classes
  from public ODNR and OpenStreetMap layers, extended to 36 on 2026-10-17
  with the ODNR 1:24k surficial map and SSURGO landforms) is in
  `CLASS_BENCHMARK.md`;
  it supersedes "more earthwork positives" as the main validation track,
  since terravector is a general terrain search and earthworks are one hard
  class within it.

- Done 2026-10-14: `data/licking_county/licking_county_p128.idx` assembled from the
  EC2 vectors (`validation/assemble_index_from_vectors.py`, 215,169 patches,
  143 s HNSW build on 4 cores); `validation/results/eval_index_p128*.json`
  reproduces the vector-level numbers. Serve-time pooling done the same day:
  `licking_county_p128_pool3.idx` (`src/pooling.py`, z-score params from the
  unpooled vectors) reproduces the pooled row above as served
  (`validation/results/eval_index_p128_pool3_served.json`, mean AP 0.0133).
- Done 2026-10-14: multi-scale (32, 128, 256 px and the 32+128, 128+256
  concatenations; table above). 128 px pooled stays the served configuration.
  Open: a 64 px `baseline_features.py` pass to explain the 64 px dip.
- Done 2026-10-14: landform-matched controls and query-level bootstrap
  intervals (table above; `--landform`, `validation/bootstrap_ci.py`).
- More positives (OAI access, Salisbury 1862 georeferencing); every interval
  above is limited by the nine queries. Re-test the 128+256 concatenation
  when they exist.
- Ross County transfer against High Bank Works and Hopeton (open 3DEP lidar).

## 3. What a determination needs

1. **More positives and many more controls.** Georeference the Salisbury 1862
   and Squier & Davis 1848 Newark plans, locate Infirmary and Salt Run mounds,
   and apply for OAI access. Draw 500+ random controls stratified by landform
   (terrace, upland, floodplain) so that "same plain as Newark" is not the thing
   being detected.
2. **A retrieval metric.** For each positive query, rank all cells; report
   precision at k and average precision over the positive set, with the
   query's own site masked out. Report per feature class (wall, conical mound,
   effigy, pit field, road).
3. **A baseline to beat.** Run the identical protocol with (a) z-scored simple
   terrain statistics of the raw DEM patch (slope, curvature, roughness, TPI at
   two scales), and (b) the same RESIDUALS statistics computed on a plain local
   relief model (DEM minus 100 ft Gaussian) instead of the residual stack. If
   (b) matches the full pipeline, the decomposition is not the contribution.
4. **Scale.** A 64 px patch is 160 ft. Walls are 40 ft wide, the Great Circle is
   1,200 ft across. Repeat with 128 px patches and with 3x3 neighbourhood
   pooling of embeddings, and report which scale wins per feature class.
5. **Transfer.** Build a Ross County index from 3DEP Phase 3 tiles and query it
   with the Newark wall embeddings. High Bank Works is a geometric twin of the
   Newark Octagon and is the natural transfer test.

## 4. How comparable work does it

**Preprocessing: local relief is the shared idea.** The lidar-archaeology
literature almost universally works on derivatives rather than the DEM: the
Local Relief Model (DEM minus a smoothed DEM; Hesse 2010), sky-view factor,
positive/negative openness, multi-scale topographic position, and multi-direction
hillshade, usually produced with the Relief Visualization Toolbox. RESIDUALS'
decomposition is a more elaborate version of the same move. A 2026 Cahokia study
(Remote Sensing 18:2229) builds LRMs, k-means-clusters them and extracts contours
to delineate mounds and borrow pits; after manual filtering it kept 32 mounds and
47 borrow pits from 266 candidates. Code: github.com/jvilbig/EarthworkMorphologyDetection.

**Detection, not retrieval, is the norm.** Three generations:

- *Template matching / OBIA (2010-2019).* Trier et al. template-matched pits and
  mounds in Norway; Davis, Sanger & Lipo (2019, Southeastern Archaeology 38:1) used
  object-based image analysis on public Beaufort County SC lidar and found 150+
  new mound features. Guyot et al. (2018, Remote Sensing 10:225) used multi-scale
  topographic position plus random forest for Neolithic barrows in Brittany.
- *Supervised CNNs (2019-).* Verschoof-van der Vaart & Lambers (2019, J. Computer
  Applications in Archaeology 2:31) trained Faster R-CNN on Dutch AHN2 LRM tiles
  for barrows and Celtic fields; Fiorucci et al. (2022, Remote Sensing 14:1694)
  added centroid- and pixel-based evaluation measures on the same Veluwe data.
  Trier, Reksten & Løseth (2021, IJAEOG) reached 87% recall with 24% false
  positives on Norwegian grave mounds, pitfall traps and charcoal kilns, but
  only 38% recall with 89% false positives on a full landscape in Larvik, a
  warning about test sets built only from tiles that contain a target. The 2025
  technical note by Zeynali, Mandanici & Bitelli (Remote Sensing 17:2733) reviews
  45 such studies: CNNs dominate, LRM/SVF/openness are the standard inputs, and
  the stated gaps are the lack of open labelled datasets, non-comparable
  metrics, and no use yet of self-supervised pretraining.
- *Self-supervised (2026).* DINO-CV (Journal of Archaeological Science: Reports,
  2026) pretrains on lidar DEM derivatives with cross-view distillation and maps
  dry-stone walls at Budj Bim, Australia, keeping 63.8% mIoU with only 10% of
  labels. This is the first archaeology paper doing what section 2 of our earlier
  discussion described.

**Query-by-example retrieval is rare in archaeology but established in
geomorphometry.** GeoPAT (Jasiewicz, Netzel & Stepinski; University of
Cincinnati Space Informatics Lab) tessellates a raster into blocks, summarises
each block by a pattern histogram, and supports query-by-example search,
clustering and segmentation of landform types. It is the closest published
analogue to terravector's design. At planetary scale, MarScope (2026) indexes
Mars CTX tile embeddings in FAISS for image-to-image landform retrieval, and
LunarFM does the same for the Moon with L2 similarity search. Google's
AlphaEarth Foundations / Satellite Embedding dataset (2025) provides 64-d
embeddings per 10 m pixel with elevation among the inputs, and Earth Engine
supports cosine-similarity search over them; its authors note that similarity
search is the use case least systematically evaluated.

**What none of these do** is combine a residual decomposition with hand-crafted
texture statistics and query-by-example at county scale. That combination is
the claim terravector could make, if the retrieval test in section 3 supports it.

## 5. Public datasets for testing

| Dataset | Region, features | Access | Fit for a retrieval test |
|---|---|---|---|
| Ohio 3DEP lidar (Phase 2, 2020-21; Phase 3, 2020-23), QL1, 8 pts/m² | Statewide; Newark, Ross County Hopewell sites, Fort Ancient | Open (USGS rockyweb / OGRIP) | Best: same sensor family as the Licking corpus, UNESCO sites with public coordinates |
| Archaeoscape (NeurIPS 2024 D&B) | Cambodia, 888 km², 0.5 m nDTM, 31k polygons: temples 827, mounds 14,400, hydrology 16,184 | CC-BY-licensed paper; data under a custom EFEO licence, credentialed request, **no georeferencing**, no redistribution | Largest open lidar-archaeology benchmark; mounds class is directly relevant; usable as tiles without coordinates |
| Veluwe / Heritage Quest (Leiden) | Netherlands AHN2 LRM tiles; 993 barrow, 2,365 Celtic field, 332 charcoal kiln tiles | AHN DTM is open (PDOK); the labelled tile set is not clearly published | Ask the Leiden group; barrows are a conical-mound class |
| ADAF (Transport Infrastructure Ireland / ZRC SAZU, 2026) | Ireland; barrows, ringforts, enclosures from 10,718 digitised monuments | Model weights on Zenodo (10.5281/zenodo.15848662) and GitHub; training polygons not confirmed public | Pretrained detectors to compare against; Irish lidar is open |
| Davis et al. 2019 Beaufort County SC | Mounds and shell rings | Only eCognition templates on Binghamton ORB; SC lidar is open | Would need the feature list from the authors |
| Norway (Trier et al.) | Grave mounds, pitfall traps, charcoal kilns; ~3,000 instances | Not released | Low priority |
| Central Kentucky Adena-Hopewell lidar (J. Archaeol. Method Theory 2019) | 244 flagged anomalies | Paywalled; anomaly list not confirmed public | Same culture as Newark; worth an email |

## Sources

- Romain & Burks, LiDAR Assessment of the Newark Earthworks (Ohio Archaeological Council): https://ohioarchaeology.org/file_download/e7d988bd-5691-4381-b567-08edbc7907d2
- NRHP listings, Licking County: https://en.wikipedia.org/wiki/National_Register_of_Historic_Places_listings_in_Licking_County,_Ohio
- Hopewell Ceremonial Earthworks: https://en.wikipedia.org/wiki/Hopewell_Ceremonial_Earthworks
- Alligator Effigy Mound: https://en.wikipedia.org/wiki/Alligator_Effigy_Mound ; Etna Township Mounds: https://en.wikipedia.org/wiki/Etna_Township_Mounds ; Flint Ridge: https://en.wikipedia.org/wiki/Flint_Ridge_State_Memorial
- Archaeological Conservancy, Etna Mound: https://www.thearchcons.org/another-hopewell-mound-saved-ohio/
- Ohio SHPO Online Mapping System and OAI access: https://www.ohiohistory.org/preserving-ohio/survey-inventory/ohio-archaeological-inventory/
- Ohio History Connection, georeferenced Squier & Davis Newark map: https://www.ohiohistory.org/x-marks-the-spot-mapping/
- USGS Ohio Phase 2 lidar report, WU 224909 (Licking): https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/OH_Statewide_Phase2_2020_B20/OH_Statewide_Phase2_4_2020/reports/OH_Statewide_Phase_2_197536_WU224909.pdf
- Ohio Phase 3 lidar (NOAA InPort): https://www.fisheries.noaa.gov/inport/item/77756
- Zeynali, Mandanici & Bitelli 2025, Remote Sensing 17:2733: https://doi.org/10.3390/rs17152733
- Cahokia LRM study, Remote Sensing 18:2229 (2026): https://doi.org/10.3390/rs18132229
- Verschoof-van der Vaart & Lambers 2019: https://journal.caa-international.org/articles/10.5334/jcaa.32
- Fiorucci et al. 2022, Remote Sensing 14:1694: https://doi.org/10.3390/rs14071694
- Trier, Reksten & Løseth 2021: https://nr.no/publikasjon/1841017/
- Guyot et al. 2018, Remote Sensing 10:225: https://doi.org/10.3390/rs10020225
- Davis, Sanger & Lipo 2019 and data record: https://orb.binghamton.edu/anthro_data/3/
- DINO-CV 2026: https://www.sciencedirect.com/science/article/pii/S2666017226000106
- Archaeoscape: https://arxiv.org/abs/2412.05203 and https://archaeoscape.ai/data/2024/
- ADAF: https://www.tii.ie/media/rokb3z00/paper-in-journal-of-archaeological-science.pdf
- GeoPAT: https://esto.nasa.gov/forum/estf2017/abstracts/Stepinski.htm
- AlphaEarth Foundations: https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/
- Earth Embeddings similarity-search notes: https://arxiv.org/html/2608.03410v1
- MarScope: https://arxiv.org/html/2601.15949 ; LunarFM: https://arxiv.org/pdf/2607.22408
- Central Kentucky Adena-Hopewell lidar: https://link.springer.com/article/10.1007/s10816-019-09420-2
