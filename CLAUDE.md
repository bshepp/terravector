# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

terravector converts Digital Elevation Models (DEMs) into searchable vector embeddings and uses an HNSW index (via `nmslib`) for O(log N) terrain-patch similarity search. It's a **pure algorithmic tool — no ML training**. Any future classifier sits as a separate layer that consumes the index; it is not a dependency inside terravector.

This is an exploratory WIP project. Expect rough edges and evolving APIs.

Sibling project: **RESIDUALS** at `F:\science-projects\RESIDUALS` produces the county-tile DEM corpus that `build_county_index.py` consumes. Decomposition and upsampling methods here are ports from DIVERGE/RESIDUALS.

## Common commands

There is **no test suite** and no linter configured. Development is run-and-inspect. Install with `pip install -r requirements.txt`.

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

# UIs
python app.py                          # Gradio web UI → http://127.0.0.1:7860
python viewer.py <dem.npy> <idx>       # napari desktop viewer

# REST API
uvicorn api:app --reload --host 127.0.0.1 --port 8000
# Index path via env var:
TERRAVECTOR_INDEX=path/to/licking_county.idx uvicorn api:app
```

`build_county_index.py` writes per-tile `.npz` checkpoints under `data/licking_county/checkpoints/` and is safe to interrupt. `--resume` skips already-checkpointed tiles. The final index is a single unified HNSW across all ~871K patches (4224×4224 tiles × 66×66 patches each at patch_size=64).

## Architecture

The pipeline is **DEM → patches → per-patch embedding → HNSW index → similarity query**. Entry points share this pipeline through `src/`.

### Embedding is config-driven, not method-driven

The core abstraction is `SignatureConfig` (`src/config.py`). It toggles five independent signature families, each contributing a slice of the final feature vector:

1. **Decomposition** — signal-decomposition residual stats (6 stats × N methods). Original method. See `src/decomposition/`.
2. **Geomorphometric** — classic GIS derivatives (slope, aspect, curvature, TPI, TRI, roughness). 6 stats × N features.
3. **Texture** — GLCM + LBP.
4. **Residuals** — DIVERGE-style decomposition × upsampling Cartesian product. For each (decomp, upsamp) pair: decompose → downsample residual (path A) → upsample back → compute rich 20-dim analysis. Dimension = n_decomp × n_upsamp × slots_per_pair × 20. Default is 320-d (4×4 grid, one slot per pair) — this is what `build_county_index.py` uses. Two optional amplification stages: `bidirectional` adds a second slot per pair carrying the (A − B) asymmetry vector, doubling the slice; `turing_intermediate` passes the round-trip output through Gray-Scott reaction-diffusion before analysis (`src/utils/turing.py`). See `compare_separability.py` for the within/across-tile separability sweep that measures whether either amplification is worth its compute cost on your corpus.
5. **Directional FFT** — 2D FFT sampled at angles for oriented feature detection.

`compute_embedding_from_config()` in `src/embedding.py` concatenates enabled slices in the fixed order above. **Adding a signature family means: add a `@dataclass` config block, a branch in `compute_embedding_from_config`, a dim contribution in `SignatureConfig.get_total_dim`, validation in `validate_config`, and parsing in `parse_config`.** Presets are built in `_create_presets()` in `src/config.py` and mirrored by YAML files in `configs/`.

### Registry pattern for extensibility

Both `src/decomposition/` and `src/upsampling/` use a decorator-based registry (`register_decomposition`, `register_upsampling`). `methods.py` holds core methods, `methods_extended.py` holds DIVERGE ports. Importing the package auto-registers everything. To add a method, decorate a function returning `(trend, residual)` (decomposition) or an upsampled array (upsampling) — no other wiring needed; `validate_config` will pick it up via `list_decompositions()` / `list_upsamplings()`.

### Index + metadata are paired files

`TerrainIndex.save(path)` writes **two** files: `path.idx` (nmslib binary) + `path.meta` (a Python-serialized blob with embeddings, metadata dict, normalization params, signature config). `load()` needs both. `build_county_index.py` additionally writes `path.corpus.json` with county bounds, tile grid, and CRS — required by `CoordBridge` and the REST API. The `.meta` format is trusted-input only; do not load `.meta` files from untrusted sources.

Patch metadata always includes `y_start`, `x_start`, `row`, `col`, `center`, `shape`; the build pipeline adds `dem_path`, `patch_size`, and (for county builds) `tile_id` + `local_id`. Visualization and the coord bridge depend on these fields being present.

### Coordinate system (county builds only)

`src/coords.py::CoordBridge` maps **WGS84 ↔ EPSG:3735 (Ohio State Plane South, feet) ↔ (tile_id, pixel) ↔ global patch_id**. It's only meaningful when the index was built from the Licking County corpus — it needs `corpus.json` alongside the `.idx`/`.meta`. API endpoints gracefully 501 when the bridge is unavailable.

When working with spatial queries: State Plane is authoritative internally; WGS84 is a convenience at API/UI boundaries. The bridge builds a `(tile_id, y_start, x_start) → global_id` lookup at construction so point queries are O(1).

### Normalization is stored, not recomputed

Embeddings are z-score normalized at index build time; the per-dimension mean/std are saved alongside the index and re-applied to every query vector in `TerrainIndex.query()`. Do not re-normalize query vectors yourself.

## Things to watch for

- **Signature config ordering is load-bearing.** The concatenation order in `compute_embedding_from_config` defines the meaning of each dimension. Changing it silently invalidates existing indices — users have no way to detect the mismatch beyond degraded query quality.
- **County index is ~1 GB+ in memory.** 871K patches × 320 dims × 4 bytes ≈ 1.1 GB just for embeddings. Be deliberate with copies.
- **Index files are gitignored** (`data/*.idx`, `data/*.npy`). Don't commit them. `data/similar_patches.png` is the one allowed artifact (README example).
- **Windows paths in `build_county_index.py`** are hardcoded absolute (`F:\science-projects\...`). If refactoring for portability, those constants are the blocker.
- **nmslib is on the roadmap to be replaced** (FAISS or Qdrant). Keep `src/index.py` as the isolation layer — callers should only touch `TerrainIndex`, not `nmslib` directly.
