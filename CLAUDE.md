# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

terravector converts Digital Elevation Models (DEMs) into searchable vector embeddings and uses an HNSW index (via `nmslib`) for O(log N) terrain-patch similarity search. It's a **pure algorithmic tool — no ML training**. Any future classifier sits as a separate layer that consumes the index; it is not a dependency inside terravector.

This is an exploratory WIP project. Expect rough edges and evolving APIs.

Sibling project: **RESIDUALS** at `F:\science-projects\RESIDUALS` produces the county-tile DEM corpus that `build_county_index.py` consumes, and is also a **runtime dependency**: the `residuals` package supplies every decomposition and upsampling method. `src/decomposition/` and `src/upsampling/` are thin facades that re-export from it. Locally it is installed editable (`pip install -e F:\science-projects\RESIDUALS`); `requirements.txt` pins a release tag for everyone else.

## Common commands

There is **no test suite** and no linter configured. Development is run-and-inspect. Install with `pip install -r requirements.txt` (plus `pip install -e ../RESIDUALS` for a local RESIDUALS checkout).

Main entry points are top-level scripts, each with `--help`:

```bash
# CLI (single DEM, most common for dev)
python cli.py build <dem.npy> --config <preset|yaml> --output terrain.idx
python cli.py query terrain.idx --patch 42 --k 10 --visualize out.png
python cli.py query terrain.idx --coords y,x --k 10
python cli.py info terrain.idx

# Full-county corpus build (long-running, checkpointed per tile)
python build_county_index.py --resume
python build_county_index.py --dry-run
python build_county_index.py --max-tiles 3   # validation subset (~25 min)
# Parallel helper: run N of these alongside the main build (see its docstring)
python -u county_worker.py --worker-id 0 --n-workers 8 --main-log build.log

# UIs
python app.py                          # Gradio web UI → http://127.0.0.1:7860
python viewer.py <dem.npy> <idx>       # napari desktop viewer

# REST API
uvicorn api:app --reload --host 127.0.0.1 --port 8000
# Index path via env var:
TERRAVECTOR_INDEX=path/to/licking_county.idx uvicorn api:app
```

`build_county_index.py` writes per-tile `.npz` checkpoints under `data/licking_county/checkpoints/` and is safe to interrupt. `--resume` skips already-checkpointed tiles. The final index is a single unified HNSW across all 858,642 patches (200 tiles × up to 66×66 patches each at patch_size=64, after dropping nodata and constant-fill patches). The embedding pass is single-threaded at ~15 min/tile; `county_worker.py` fills idle cores and writes the same checkpoints.

## Architecture

The pipeline is **DEM → patches → per-patch embedding → HNSW index → similarity query**. Entry points share this pipeline through `src/`.

### Embedding is config-driven, not method-driven

The core abstraction is `SignatureConfig` (`src/config.py`). It toggles five independent signature families, each contributing a slice of the final feature vector:

1. **Decomposition** — signal-decomposition residual stats (6 stats × N methods). Original method. Uses the six legacy core methods pinned in `src/decomposition/__init__.py`; their order defines the slice layout.
2. **Geomorphometric** — classic GIS derivatives (slope, aspect, curvature, TPI, TRI, roughness). 6 stats × N features.
3. **Texture** — GLCM + LBP.
4. **Residuals** — DIVERGE-style decomposition × upsampling Cartesian product. For each (decomp, upsamp) pair: decompose → downsample residual (path A) → upsample back → compute rich 20-dim analysis. Dimension = n_decomp × n_upsamp × slots_per_pair × 20. The shipped `configs/residuals.yaml` defaults to **640-d** (4×4 grid × 2 slots per pair: path A + bidirectional asymmetry). Two optional amplification stages live here: `bidirectional` (default ON, +2% separability at 2× cost) appends a second slot per pair carrying the (A − B) asymmetry vector; `turing_intermediate` (default OFF, experimental) passes the round-trip output through Gray-Scott reaction-diffusion before analysis (`src/utils/turing.py`). Use `compare_separability.py` for a full within/across-tile separability sweep and `sweep_turing.py` for Turing-specific iteration/Pearson-(F,k)-grid sweeps. The sweep evidence for the current defaults lives in `data/separability/*.json`.
5. **Directional FFT** — 2D FFT sampled at angles for oriented feature detection.

`compute_embedding_from_config()` in `src/embedding.py` concatenates enabled slices in the fixed order above. **Adding a signature family means: add a `@dataclass` config block, a branch in `compute_embedding_from_config`, a dim contribution in `SignatureConfig.get_total_dim`, validation in `validate_config`, and parsing in `parse_config`.** Presets are built in `_create_presets()` in `src/config.py` and mirrored by YAML files in `configs/`.

### Registry pattern for extensibility

Both `src/decomposition/` and `src/upsampling/` re-export the decorator-based registries from the `residuals` package (`register_decomposition`, `register_upsampling`, `run_*`, `list_*`). The method implementations live in RESIDUALS (`residuals/decomposition/methods*.py`, `residuals/upsampling/methods*.py`), not here. To add a method, register it in RESIDUALS — no wiring needed on this side; `validate_config` picks it up via `list_decompositions()` / `list_upsamplings()`. `src/features/analysis.py` (the 20-dim rich analysis vector) is terravector's own and has diverged from RESIDUALS' `analyze_features`; keep it local.

### Index + metadata are paired files

`TerrainIndex.save(path)` writes **two** files: `path.idx` (nmslib binary) + `path.meta` (a Python-serialized blob with embeddings, metadata dict, normalization params, signature config). `load()` needs both. `build_county_index.py` additionally writes `path.corpus.json` with county bounds, tile grid, and CRS — required by `CoordBridge` and the REST API. The `.meta` format is trusted-input only; do not load `.meta` files from untrusted sources.

Patch metadata always includes `y_start`, `x_start`, `row`, `col`, `center`, `shape`; the build pipeline adds `dem_path`, `patch_size`, and (for county builds) `tile_id` + `local_id`. Visualization and the coord bridge depend on these fields being present.

### Coordinate system (county builds only)

`src/coords.py::CoordBridge` maps **WGS84 ↔ EPSG:3735 (Ohio State Plane South, feet) ↔ (tile_id, pixel) ↔ global patch_id**. It's only meaningful when the index was built from the Licking County corpus — it needs `corpus.json` alongside the `.idx`/`.meta`. API endpoints gracefully 501 when the bridge is unavailable.

When working with spatial queries: State Plane is authoritative internally; WGS84 is a convenience at API/UI boundaries. The bridge builds a `(tile_id, y_start, x_start) → global_id` lookup at construction so point queries are O(1).

### Normalization is stored, not recomputed

Embeddings are z-score normalized at index build time; the per-dimension mean/std are saved alongside the index and re-applied to every query vector in `TerrainIndex.query()`. Do not re-normalize query vectors yourself.

## Things to watch for

- **Signature config ordering is load-bearing.** The concatenation order in `compute_embedding_from_config` defines the meaning of each dimension. Changing it silently invalidates existing indices — users have no way to detect the mismatch beyond degraded query quality. The dim change from 320 → 640 when bidirectional was enabled by default is exactly this hazard: 320-d indices remain readable via their stored `signature_config`, but a fresh build with the current `residuals.yaml` is not query-compatible with them.
- **County index is ~2 GB+ in memory.** 859K patches × 640 dims × 4 bytes ≈ 2.2 GB just for embeddings (was 1.1 GB at the old 320-d default). Be deliberate with copies.
- **Per-tile audit sidecars.** `build_county_index.py` writes `{tile_id}_diagnostics.json` next to each checkpoint, recording any (decomp, upsamp, path) pairs that silently failed and were zero-filled in the embedding. The final `corpus.json` carries a `diagnostics` block rolling these up across tiles. Historical note: the `gaussian/{bicubic,lanczos,bspline}:A:ValueError` cluster on edge tiles came from constant-valued nodata-fill patches (not NaN) — their ~1e-13 gaussian residual noise broke the histogram in `analyze_features`. `tile_dem` now drops constant patches via `min_range`, so a fresh build should show zero failures; any that remain are new.
- **Bumping the RESIDUALS pin is an index-compat event.** Any numeric change in a `residuals` method silently changes embeddings. Before adopting a new tag, recompute a sample of a checkpointed tile and compare bit-for-bit against the stored `.npz` (as was done for the v0.2.0 adoption: 117/117 rows identical). `rolling_ball` is the one method known to differ from the pre-dependency port; no shipped preset uses it.
- **The SNR analysis feature is log-scaled (2026-10-11).** `compute_feature_snr` in `src/features/analysis.py` returns `log1p(std_signal / std_noise)`. The raw ratio reached 1e15 on near-flat residuals and, under z-score normalization, zeroed 12 of the 640 dims for every other patch. Indices built before this change are not query-compatible; the county checkpoints were transformed in place (`log1p` on dims 16::20) and the index reassembled, so no re-embedding was needed. Pre-change artifacts are in `data/licking_county/_backup_2026-10-11_pre_logsnr/`.
- **Index files are gitignored** (`data/*.idx`, `data/*.npy`). Don't commit them. `data/similar_patches.png` is the one allowed artifact (README example).
- **Windows paths in `build_county_index.py`** are hardcoded absolute (`F:\science-projects\...`). If refactoring for portability, those constants are the blocker.
- **nmslib is on the roadmap to be replaced** (FAISS or Qdrant). Keep `src/index.py` as the isolation layer — callers should only touch `TerrainIndex`, not `nmslib` directly.
- **Turing branch is paused, not closed.** `sweep_turing.py` proved that iterating Gray-Scott on `(residual A) round-trip` output degrades terrain separability at every (iter, F, k) tested. The amplification idea is still open via different seeds, different reaction-diffusion systems, or different consumers — re-test with `sweep_turing.py` before re-enabling in any preset.
