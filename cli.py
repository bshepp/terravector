#!/usr/bin/env python
"""
terravector CLI

Command-line interface for terrain similarity search.

Usage:
    python cli.py build <dem_path> --patch-size 64 --output terrain.idx
    python cli.py build <dem_path> --config signature.yaml --output terrain.idx
    python cli.py query <index_path> --patch 42 --k 10
    python cli.py query <index_path> --patch 42 --k 10 --dem <dem_path> --visualize results.png
    python cli.py info <index_path>
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_dem, get_dem_info
from src.tiling import tile_dem, patches_to_metadata, get_tiling_info
from src.embedding import (
    compute_embeddings_batch,
    compute_embeddings_batch_from_config,
    get_embedding_dim,
)
from src.index import TerrainIndex, build_index
from src.config import (
    load_config,
    get_default_config,
    get_preset,
    validate_config,
    SignatureConfig,
)


def cmd_build(args):
    """Build index from DEM."""
    dem_path = str(Path(args.dem_path).resolve())
    
    # Load or create configuration
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            print(f"Loading config: {args.config}")
            config = load_config(args.config)
        else:
            # Try as preset name
            try:
                print(f"Using preset: {args.config}")
                config = get_preset(args.config)
            except ValueError as e:
                print(f"Error: {e}")
                return 1
        
        # Validate config
        errors = validate_config(config)
        if errors:
            print("Config validation errors:")
            for err in errors:
                print(f"  - {err}")
            return 1
        
        # Show signature composition
        enabled = config.get_enabled_types()
        print(f"  Signature types: {', '.join(enabled)}")
        print(f"  Expected dimension: {config.get_total_dim()}")
    
    print(f"\nLoading DEM: {dem_path}")
    dem = load_dem(args.dem_path)
    
    info = get_dem_info(dem)
    print(f"  Shape: {info['shape']}")
    print(f"  Elevation range: {info['min']:.2f} - {info['max']:.2f}")
    
    print(f"\nTiling with patch_size={args.patch_size}, overlap={args.overlap}")
    patches = tile_dem(
        dem,
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_valid_fraction=args.min_valid
    )
    
    tiling_info = get_tiling_info(dem.shape, args.patch_size, args.overlap)
    print(f"  Generated {len(patches)} patches ({tiling_info['n_rows']}x{tiling_info['n_cols']} grid)")
    
    if len(patches) == 0:
        print("Error: No valid patches found")
        return 1
    
    print("\nComputing embeddings...")
    if config:
        embeddings = compute_embeddings_batch_from_config(patches, config, verbose=True)
    else:
        embeddings = compute_embeddings_batch(patches, verbose=True)
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    print("\nBuilding HNSW index...")
    metadata = patches_to_metadata(patches)
    
    # Add source DEM info to enable visualization later
    for pid in metadata:
        metadata[pid]['dem_path'] = dem_path
        metadata[pid]['patch_size'] = args.patch_size
    
    # Get space from config or args
    space = config.space if config else args.space
    normalize = config.normalize if config else not args.no_normalize
    
    index = build_index(
        embeddings,
        metadata,
        space=space,
        M=args.M,
        normalize=normalize
    )
    
    # Store config info in index for later reference
    if config:
        index.signature_config = config.to_dict()
    
    stats = index.get_stats()
    print(f"  Index built with {stats['n_patches']} patches")
    
    print(f"\nSaving index to: {args.output}")
    index.save(args.output)
    print("Done!")
    
    return 0


def cmd_query(args):
    """Query index for similar patches."""
    print(f"Loading index: {args.index_path}")
    index = TerrainIndex.load(args.index_path)
    
    stats = index.get_stats()
    print(f"  {stats['n_patches']} patches, dim={stats['embedding_dim']}")
    
    query_patch_id = None
    
    if args.patch is not None:
        # Query by patch ID
        print(f"\nQuerying for patches similar to patch {args.patch}...")
        query_patch_id = args.patch
        results = index.query_by_id(args.patch, k=args.k)
    
    elif args.coords is not None:
        # Query by coordinates - find the embedding for that location
        coords = [int(x) for x in args.coords.split(',')]
        if len(coords) != 2:
            print("Error: --coords must be y,x format")
            return 1
        
        y, x = coords
        print(f"\nFinding patch at coordinates ({y}, {x})...")
        
        # Find patch containing these coordinates
        for pid, meta in index.metadata.items():
            patch_size = meta.get('patch_size', meta.get('shape', (64, 64))[0])
            if (meta['y_start'] <= y < meta['y_start'] + patch_size and
                meta['x_start'] <= x < meta['x_start'] + patch_size):
                print(f"  Found patch {pid}")
                query_patch_id = pid
                results = index.query_by_id(pid, k=args.k)
                break
        else:
            print("Error: No patch found at those coordinates")
            return 1
    
    else:
        print("Error: Must specify --patch or --coords")
        return 1
    
    # Display results
    print(f"\nTop {len(results)} similar patches:")
    print("-" * 60)
    print(f"{'Rank':<6} {'ID':<8} {'Distance':<12} {'Location'}")
    print("-" * 60)
    
    for rank, (pid, dist, meta) in enumerate(results, 1):
        center = meta.get('center', (0, 0))
        print(f"{rank:<6} {pid:<8} {dist:<12.4f} ({center[0]}, {center[1]})")
    
    # Visualization
    if args.visualize:
        from src.visualization import visualize_query_results, extract_patch_from_dem
        
        # Get DEM path - from args or from index metadata
        dem_path = args.dem
        if dem_path is None:
            # Try to get from metadata
            sample_meta = next(iter(index.metadata.values()))
            dem_path = sample_meta.get('dem_path')
        
        if dem_path is None:
            print("\nError: Cannot visualize - DEM path not found.")
            print("  Either rebuild index with newer version, or specify --dem path")
            return 1
        
        print(f"\nLoading DEM for visualization: {dem_path}")
        dem = load_dem(dem_path)
        
        # Get patch size
        sample_meta = index.metadata[query_patch_id]
        patch_size = sample_meta.get('patch_size', sample_meta.get('shape', (64, 64))[0])
        
        # Extract query patch
        query_meta = index.metadata[query_patch_id]
        query_patch = extract_patch_from_dem(
            dem, query_meta['y_start'], query_meta['x_start'], patch_size
        )
        
        # Extract similar patches
        similar_patches = []
        similar_ids = []
        distances = []
        
        for pid, dist, meta in results:
            patch = extract_patch_from_dem(
                dem, meta['y_start'], meta['x_start'], patch_size
            )
            similar_patches.append(patch)
            similar_ids.append(pid)
            distances.append(dist)
        
        # Visualize
        visualize_query_results(
            query_patch=query_patch,
            similar_patches=similar_patches,
            query_label=f"Query (#{query_patch_id})",
            similar_ids=similar_ids,
            distances=distances,
            output_path=args.visualize,
            use_hillshade=True
        )
    
    return 0


def cmd_info(args):
    """Display index information."""
    print(f"Loading index: {args.index_path}")
    index = TerrainIndex.load(args.index_path)
    
    stats = index.get_stats()
    
    print("\nIndex Statistics:")
    print("-" * 40)
    print(f"  Patches:       {stats['n_patches']}")
    print(f"  Embedding dim: {stats['embedding_dim']}")
    print(f"  Distance:      {stats['space']}")
    print(f"  HNSW M:        {stats['M']}")
    print(f"  Normalized:    {stats['normalized']}")
    
    # Check for DEM path
    sample_meta = next(iter(index.metadata.values()), {})
    dem_path = sample_meta.get('dem_path', 'Not stored')
    patch_size = sample_meta.get('patch_size', 'Not stored')
    print(f"  Source DEM:    {dem_path}")
    print(f"  Patch size:    {patch_size}")
    
    # Signature configuration (if available)
    if hasattr(index, 'signature_config') and index.signature_config:
        sig = index.signature_config.get('signature', {})
        print("\nSignature Configuration:")
        print("-" * 40)
        
        # Decomposition
        decomp = sig.get('decomposition', {})
        if decomp.get('enabled', False):
            methods = decomp.get('methods', [])
            print(f"  Decomposition: {', '.join(methods)}")
        
        # Geomorphometric
        geomorph = sig.get('geomorphometric', {})
        if geomorph.get('enabled', False):
            features = geomorph.get('features', [])
            print(f"  Geomorphometric: {', '.join(features)}")
        
        # Texture
        texture = sig.get('texture', {})
        if texture.get('enabled', False):
            features = texture.get('features', [])
            print(f"  Texture: {', '.join(features)}")
    else:
        print("\nSignature: decomposition (default)")
    
    # Embedding statistics
    if index.embeddings is not None:
        emb = index.embeddings
        print("\nEmbedding Statistics:")
        print("-" * 40)
        print(f"  Mean:   {np.mean(emb):.4f}")
        print(f"  Std:    {np.std(emb):.4f}")
        print(f"  Min:    {np.min(emb):.4f}")
        print(f"  Max:    {np.max(emb):.4f}")
    
    # Patch grid info
    if index.metadata:
        rows = [m['row'] for m in index.metadata.values()]
        cols = [m['col'] for m in index.metadata.values()]
        print("\nPatch Grid:")
        print("-" * 40)
        print(f"  Rows: {min(rows)} - {max(rows)} ({max(rows) - min(rows) + 1} total)")
        print(f"  Cols: {min(cols)} - {max(cols)} ({max(cols) - min(cols) + 1} total)")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='terravector: Terrain patch similarity search',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build index from DEM')
    build_parser.add_argument('dem_path', help='Path to DEM file (.npy or .tif)')
    build_parser.add_argument('--output', '-o', required=True, help='Output index path')
    build_parser.add_argument('--config', '-c', help='Signature config file (YAML) or preset name (default, classic, texture, hybrid, minimal)')
    build_parser.add_argument('--patch-size', type=int, default=64, help='Patch size (default: 64)')
    build_parser.add_argument('--overlap', type=int, default=0, help='Patch overlap (default: 0)')
    build_parser.add_argument('--min-valid', type=float, default=0.8, help='Min valid pixel fraction (default: 0.8)')
    build_parser.add_argument('--space', default='cosinesimil', choices=['cosinesimil', 'l2', 'l1'], help='Distance metric')
    build_parser.add_argument('--M', type=int, default=16, help='HNSW M parameter')
    build_parser.add_argument('--no-normalize', action='store_true', help='Skip embedding normalization')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query index for similar patches')
    query_parser.add_argument('index_path', help='Path to index')
    query_parser.add_argument('--patch', type=int, help='Query by patch ID')
    query_parser.add_argument('--coords', help='Query by coordinates (y,x format)')
    query_parser.add_argument('--k', type=int, default=10, help='Number of results (default: 10)')
    query_parser.add_argument('--visualize', '-v', metavar='PATH', help='Output visualization to file')
    query_parser.add_argument('--dem', help='Path to source DEM (for visualization)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display index information')
    info_parser.add_argument('index_path', help='Path to index')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'build':
        return cmd_build(args)
    elif args.command == 'query':
        return cmd_query(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
