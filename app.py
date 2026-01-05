#!/usr/bin/env python
"""
terravector Gradio UI

Interactive web interface for terrain similarity search.
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.state import AppState
from src.ui.components import (
    create_hillshade_image,
    create_results_gallery,
    create_placeholder_image,
    format_status,
)
from src.config import PRESETS

# =============================================================================
# Global State
# =============================================================================

state = AppState()

# =============================================================================
# Event Handlers
# =============================================================================

def load_dem(path: str):
    """Load DEM from file path."""
    if not path or not path.strip():
        return (
            create_placeholder_image(text="Enter a DEM path above"),
            None,
            "Enter a path to a .npy or .tif file"
        )
    
    msg = state.load_dem_file(path.strip())
    
    if state.dem is not None:
        # Create hillshade visualization
        hs_img = create_hillshade_image(
            state.dem,
            patch_grid=True,
            patch_size=state.patch_size
        )
        return hs_img, None, format_status(state.dem_info, message=msg)
    else:
        return create_placeholder_image(text=msg), None, msg


def load_index(path: str):
    """Load existing index from file."""
    if not path or not path.strip():
        return (
            gr.update(),
            None,
            "Enter an index path"
        )
    
    msg = state.load_index_file(path.strip())
    
    if state.dem is not None:
        hs_img = create_hillshade_image(
            state.dem,
            patch_grid=True,
            patch_size=state.patch_size
        )
        stats = state.index.get_stats() if state.index else None
        return hs_img, None, format_status(state.dem_info, stats, message=msg)
    else:
        stats = state.index.get_stats() if state.index else None
        return gr.update(), None, format_status(None, stats, message=msg)


def build_index(preset: str, patch_size: int, progress=gr.Progress()):
    """Build index from loaded DEM."""
    if state.dem is None:
        return None, "Load a DEM first"
    
    def progress_callback(pct, msg):
        progress(pct, desc=msg)
    
    msg = state.build_new_index(
        preset_name=preset,
        patch_size=int(patch_size),
        progress_callback=progress_callback
    )
    
    # Update hillshade with new patch grid
    if state.dem is not None:
        hs_img = create_hillshade_image(
            state.dem,
            patch_grid=True,
            patch_size=int(patch_size)
        )
        stats = state.index.get_stats() if state.index else None
        return hs_img, format_status(state.dem_info, stats, message=msg)
    
    return gr.update(), msg


def query_click(evt: gr.SelectData, k_results: int):
    """Handle click on DEM image to query."""
    if state.index is None:
        return None, gr.update(), "Build or load an index first"
    
    if state.dem is None:
        return None, gr.update(), "No DEM loaded"
    
    # evt.index gives [y, x] in the displayed image coordinates
    # We need to scale from display space to DEM space
    click_y, click_x = evt.index[0], evt.index[1]
    
    # The image is displayed at height=500, width scales proportionally
    dem_h, dem_w = state.dem.shape[:2]
    display_h = 500
    display_w = int(500 * dem_w / dem_h)
    
    # Scale click coordinates to DEM coordinates
    y = int(click_y * dem_h / display_h)
    x = int(click_x * dem_w / display_w)
    
    # Clamp to valid range
    y = max(0, min(y, dem_h - 1))
    x = max(0, min(x, dem_w - 1))
    
    msg = state.query_by_coords(y, x, k=int(k_results))
    
    if state.last_results and state.dem is not None:
        # Create results gallery
        query_meta = state.index.metadata.get(state.last_query_id)
        gallery = create_results_gallery(
            state.dem,
            state.last_results,
            state.patch_size,
            query_id=state.last_query_id,
            query_meta=query_meta
        )
        
        # Update DEM view with highlighted query patch
        hs_img = create_hillshade_image(
            state.dem,
            patch_grid=True,
            patch_size=state.patch_size,
            highlight_patch=state.last_query_id,
            metadata=state.index.metadata
        )
        
        return gallery, hs_img, msg
    
    return None, gr.update(), msg


def query_by_id(patch_id: int, k_results: int):
    """Query by patch ID."""
    if state.index is None:
        return None, gr.update(), "Build or load an index first"
    
    msg = state.query_by_id(int(patch_id), k=int(k_results))
    
    if state.last_results and state.dem is not None:
        query_meta = state.index.metadata.get(state.last_query_id)
        gallery = create_results_gallery(
            state.dem,
            state.last_results,
            state.patch_size,
            query_id=state.last_query_id,
            query_meta=query_meta
        )
        
        hs_img = create_hillshade_image(
            state.dem,
            patch_grid=True,
            patch_size=state.patch_size,
            highlight_patch=state.last_query_id,
            metadata=state.index.metadata
        )
        
        return gallery, hs_img, msg
    
    return None, gr.update(), msg


def save_index(path: str):
    """Save current index to file."""
    if not path or not path.strip():
        return "Enter a save path"
    return state.save_index(path.strip())


def random_query(k_results: int):
    """Query a random patch."""
    if state.index is None:
        return None, gr.update(), "Build or load an index first"
    
    # Pick random patch
    patch_ids = list(state.index.metadata.keys())
    if not patch_ids:
        return None, gr.update(), "No patches in index"
    
    random_id = int(np.random.choice(patch_ids))
    return query_by_id(random_id, k_results)


# =============================================================================
# Custom CSS
# =============================================================================

custom_css = """
/* Dark theme adjustments */
.gradio-container {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
}

/* Status bar styling */
.status-text {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em;
    color: #a0a0a0;
    background: #1a1a2e;
    padding: 8px 12px;
    border-radius: 4px;
    border-left: 3px solid #4a6fa5;
}

/* Gallery styling */
.gallery-item {
    border-radius: 4px;
    overflow: hidden;
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #4a6fa5 0%, #3d5a80 100%) !important;
    border: none !important;
}

.primary-btn:hover {
    background: linear-gradient(135deg, #5a7fb5 0%, #4d6a90 100%) !important;
}

/* DEM image container */
.dem-container {
    border: 2px solid #2a2a4a;
    border-radius: 8px;
    overflow: hidden;
}

/* Header styling */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    border-left: 4px solid #4a6fa5;
}

.app-title {
    font-size: 1.8em;
    font-weight: 700;
    color: #e0e0e0;
    margin: 0;
    letter-spacing: -0.02em;
}

.app-subtitle {
    color: #808090;
    font-size: 0.9em;
    margin-top: 4px;
}
"""

# =============================================================================
# Build UI
# =============================================================================

def create_ui():
    """Create the Gradio interface."""
    
    # Get available presets
    preset_choices = list(PRESETS.keys())
    
    with gr.Blocks(
        title="terravector",
        theme=gr.themes.Base(
            primary_hue="slate",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0f0f1a",
            body_background_fill_dark="#0f0f1a",
            block_background_fill="#1a1a2e",
            block_background_fill_dark="#1a1a2e",
            block_border_color="#2a2a4a",
            block_border_color_dark="#2a2a4a",
            input_background_fill="#16213e",
            input_background_fill_dark="#16213e",
            button_primary_background_fill="#4a6fa5",
            button_primary_background_fill_dark="#4a6fa5",
        ),
        css=custom_css
    ) as app:
        
        # Header
        gr.HTML("""
            <div class="app-header">
                <div class="app-title">terravector</div>
                <div class="app-subtitle">Terrain patch similarity search</div>
            </div>
        """)
        
        with gr.Row():
            # Left column: DEM view
            with gr.Column(scale=1):
                gr.Markdown("### DEM Overview")
                gr.Markdown("*Click on the map to query that location*")
                
                dem_image = gr.Image(
                    value=create_placeholder_image(),
                    label="DEM Hillshade",
                    interactive=False,
                    show_label=False,
                    height=500,
                    elem_classes=["dem-container"]
                )
            
            # Right column: Results
            with gr.Column(scale=1):
                gr.Markdown("### Query Results")
                gr.Markdown("*Similar patches ordered by distance*")
                
                results_gallery = gr.Gallery(
                    label="Similar Patches",
                    show_label=False,
                    columns=4,
                    rows=2,
                    height=500,
                    object_fit="cover",
                    allow_preview=True
                )
        
        # Controls row
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("#### Load Data")
                    with gr.Row():
                        dem_path = gr.Textbox(
                            label="DEM Path",
                            placeholder="path/to/dem.npy",
                            scale=3
                        )
                        load_dem_btn = gr.Button("Load DEM", scale=1)
                    
                    with gr.Row():
                        index_path = gr.Textbox(
                            label="Index Path",
                            placeholder="path/to/index.idx",
                            scale=3
                        )
                        load_index_btn = gr.Button("Load Index", scale=1)
            
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("#### Build Settings")
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            choices=preset_choices,
                            value="default",
                            label="Signature Preset",
                            scale=2
                        )
                        patch_size_slider = gr.Slider(
                            minimum=32,
                            maximum=128,
                            value=64,
                            step=16,
                            label="Patch Size",
                            scale=1
                        )
                        k_slider = gr.Slider(
                            minimum=4,
                            maximum=20,
                            value=8,
                            step=1,
                            label="K Results",
                            scale=1
                        )
                    
                    with gr.Row():
                        build_btn = gr.Button("Build Index", variant="primary", scale=2)
                        random_btn = gr.Button("Random Query", scale=1)
        
        # Query by ID row
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### Query by Patch ID")
                    with gr.Row():
                        patch_id_input = gr.Number(
                            label="Patch ID",
                            value=0,
                            precision=0,
                            scale=2
                        )
                        query_id_btn = gr.Button("Query", scale=1)
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### Save Index")
                    with gr.Row():
                        save_path = gr.Textbox(
                            label="Save Path",
                            placeholder="output/terrain.idx",
                            scale=2
                        )
                        save_btn = gr.Button("Save", scale=1)
        
        # Status bar
        status_text = gr.Textbox(
            label="Status",
            value="Ready - Load a DEM or existing index to begin",
            interactive=False,
            elem_classes=["status-text"]
        )
        
        # =================================================================
        # Event bindings
        # =================================================================
        
        # Load DEM
        load_dem_btn.click(
            fn=load_dem,
            inputs=[dem_path],
            outputs=[dem_image, results_gallery, status_text]
        )
        
        dem_path.submit(
            fn=load_dem,
            inputs=[dem_path],
            outputs=[dem_image, results_gallery, status_text]
        )
        
        # Load Index
        load_index_btn.click(
            fn=load_index,
            inputs=[index_path],
            outputs=[dem_image, results_gallery, status_text]
        )
        
        index_path.submit(
            fn=load_index,
            inputs=[index_path],
            outputs=[dem_image, results_gallery, status_text]
        )
        
        # Build Index
        build_btn.click(
            fn=build_index,
            inputs=[preset_dropdown, patch_size_slider],
            outputs=[dem_image, status_text]
        )
        
        # Click to query
        dem_image.select(
            fn=query_click,
            inputs=[k_slider],
            outputs=[results_gallery, dem_image, status_text]
        )
        
        # Query by ID
        query_id_btn.click(
            fn=query_by_id,
            inputs=[patch_id_input, k_slider],
            outputs=[results_gallery, dem_image, status_text]
        )
        
        # Random query
        random_btn.click(
            fn=random_query,
            inputs=[k_slider],
            outputs=[results_gallery, dem_image, status_text]
        )
        
        # Save index
        save_btn.click(
            fn=save_index,
            inputs=[save_path],
            outputs=[status_text]
        )
    
    return app


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

