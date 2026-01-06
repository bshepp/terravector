"""
Qt Dock Widgets for Napari Viewer

Control panel widgets for threshold, visualization modes, and query controls.
Uses magicgui for automatic widget generation.
"""

from typing import TYPE_CHECKING
from magicgui import magicgui
from magicgui.widgets import Container, PushButton, Label, SpinBox, FloatSlider, CheckBox, ComboBox

if TYPE_CHECKING:
    from src.viewer.app import TerrainViewer


def create_control_widget(viewer: 'TerrainViewer') -> Container:
    """
    Create the main control widget for the similarity viewer.
    
    Args:
        viewer: TerrainViewer instance to control
        
    Returns:
        magicgui Container with all controls
    """
    # Header label
    header = Label(value="═══ Similarity Controls ═══")
    
    # Query section
    query_label = Label(value="─── Query ───")
    
    patch_id_input = SpinBox(
        value=0,
        min=0,
        max=999999,
        label="Patch ID:",
        tooltip="Enter a patch ID to query"
    )
    
    query_button = PushButton(text="Query by ID")
    
    @query_button.clicked.connect
    def on_query_click():
        viewer.query_by_id(patch_id_input.value)
    
    random_button = PushButton(text="Random Query")
    
    @random_button.clicked.connect
    def on_random_click():
        if viewer.index is not None:
            import random
            patch_ids = list(viewer.index.metadata.keys())
            if patch_ids:
                random_id = random.choice(patch_ids)
                patch_id_input.value = random_id
                viewer.query_by_id(random_id)
    
    # Threshold section
    threshold_label = Label(value="─── Filtering ───")
    
    threshold_slider = FloatSlider(
        value=1.0,
        min=0.0,
        max=2.0,
        step=0.01,
        label="Threshold:",
        tooltip="Maximum distance for similar patches"
    )
    
    @threshold_slider.changed.connect
    def on_threshold_change(value: float):
        viewer.set_threshold(value)
    
    max_results_input = SpinBox(
        value=50,
        min=1,
        max=500,
        label="Max Results:",
        tooltip="Maximum number of similar patches to show"
    )
    
    @max_results_input.changed.connect
    def on_max_results_change(value: int):
        viewer.set_max_results(value)
    
    # Visualization mode section
    viz_label = Label(value="─── Visualization ───")
    
    show_boxes = CheckBox(
        value=True,
        label="Box Outlines",
        tooltip="Show colored boxes around similar tiles"
    )
    
    @show_boxes.changed.connect
    def on_boxes_toggle(checked: bool):
        viewer.toggle_boxes(checked)
    
    show_heatmap = CheckBox(
        value=False,
        label="Heatmap Overlay",
        tooltip="Show similarity as color intensity"
    )
    
    @show_heatmap.changed.connect
    def on_heatmap_toggle(checked: bool):
        viewer.toggle_heatmap(checked)
    
    show_fade = CheckBox(
        value=False,
        label="Fade Non-matching",
        tooltip="Dim tiles that don't match"
    )
    
    @show_fade.changed.connect
    def on_fade_toggle(checked: bool):
        viewer.toggle_fade(checked)
    
    # Info section
    info_label = Label(value="─── Info ───")
    
    status_label = Label(value="Click on terrain to query")
    
    # Update status when query happens
    original_query = viewer.query_by_id
    def query_with_status(patch_id: int):
        original_query(patch_id)
        if viewer.results:
            status_label.value = f"Query #{patch_id}: {len(viewer.results)} results"
        else:
            status_label.value = f"No results for #{patch_id}"
    viewer.query_by_id = query_with_status
    
    # Help text
    help_label = Label(
        value="Tip: Click terrain to query\nGreen = query, Orange = similar"
    )
    
    # Build container
    container = Container(
        widgets=[
            header,
            query_label,
            patch_id_input,
            query_button,
            random_button,
            threshold_label,
            threshold_slider,
            max_results_input,
            viz_label,
            show_boxes,
            show_heatmap,
            show_fade,
            info_label,
            status_label,
            help_label
        ],
        labels=False
    )
    
    return container


def create_signature_selector(
    viewer: 'TerrainViewer',
    presets: list
) -> ComboBox:
    """
    Create a signature preset selector.
    
    Args:
        viewer: TerrainViewer instance
        presets: List of preset names
        
    Returns:
        ComboBox widget for preset selection
    """
    selector = ComboBox(
        choices=presets,
        value=presets[0] if presets else None,
        label="Signature:",
        tooltip="Select signature configuration"
    )
    
    # Note: Changing signatures would require re-building the index
    # This is more for display/info purposes
    
    return selector

