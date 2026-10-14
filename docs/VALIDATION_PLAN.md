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
   the county per query, against roughly 0.9% expected for a random 3x3 block,
   about a 14x enrichment. Walls retrieve walls across sites 15 miles apart
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
   enriches the top of the list by an order of magnitude for wall-type
   features, with nine positives and uniform (not landform-stratified) controls.

### Next steps, in order of value per hour

- Assemble a 128 px county index from the vectors already computed
  (`s3://3body-compute-290318/terravector/baselines/full_p128/`, 200 tiles x
  up to 1,089 patches) and make it the served index; try 3x3 pooling at serve
  time. No re-embedding needed.
- Multi-scale: 256 px and a 128 px / 64 px concatenation; 32 px for the wall
  cross-section.
- More positives (OAI access, Salisbury 1862 georeferencing) and controls
  stratified by landform, then report per class with confidence intervals.
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
