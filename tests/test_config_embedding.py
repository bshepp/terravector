"""Tests for signature presets and the config-driven embedding pipeline.

These run the real decomposition/upsampling methods (from the ``residuals``
package) on a single 64x64 patch per preset, so they take a second or two
each but stay far from county-scale cost.
"""
import numpy as np
import pytest

from src.config import PRESETS, get_preset, validate_config
from src.embedding import (
    compute_embedding_from_config,
    get_embedding_dim_from_config,
    get_embedding_labels_from_config,
)

PRESET_NAMES = sorted(PRESETS)


class TestPresets:
    def test_expected_presets_exist(self):
        assert {"default", "minimal", "residuals", "residuals_extended", "spectral"} <= set(PRESETS)

    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_every_preset_validates(self, name):
        config = get_preset(name)
        assert validate_config(config) == []
        assert config.get_total_dim() > 0

    def test_unknown_preset_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_preset("definitely-not-a-preset")

    def test_shipped_county_preset_is_640d(self):
        # The Licking County index is built with this preset; its dimension is an
        # index-compat contract (see CLAUDE.md).
        config = get_preset("residuals")
        assert config.get_total_dim() == 640
        assert config.normalize is True
        assert config.space == "cosinesimil"


class TestEmbeddingFromConfig:
    @pytest.mark.parametrize("name", ["minimal", "residuals"])
    def test_dim_and_finiteness(self, name, terrain_patch):
        config = get_preset(name)
        vec = compute_embedding_from_config(terrain_patch, config)
        assert vec.ndim == 1
        assert vec.shape[0] == config.get_total_dim() == get_embedding_dim_from_config(config)
        assert np.all(np.isfinite(vec))

    def test_labels_match_dim(self):
        for name in ("minimal", "residuals"):
            config = get_preset(name)
            labels = get_embedding_labels_from_config(config)
            assert len(labels) == config.get_total_dim(), name
            assert len(set(labels)) == len(labels), name

    def test_residuals_labels_mark_log_scaled_slots(self):
        labels = get_embedding_labels_from_config(get_preset("residuals"))
        assert any("slog_" in lab for lab in labels)
        assert any("log_snr" in lab for lab in labels)

    def test_deterministic(self, terrain_patch):
        config = get_preset("residuals")
        a = compute_embedding_from_config(terrain_patch, config)
        b = compute_embedding_from_config(terrain_patch, config)
        np.testing.assert_array_equal(a, b)

    def test_different_terrain_gives_different_embedding(self, terrain_patch, rng):
        config = get_preset("residuals")
        other = rng.normal(scale=5.0, size=terrain_patch.shape) + np.linspace(0, 50, 64)
        a = compute_embedding_from_config(terrain_patch, config)
        b = compute_embedding_from_config(other, config)
        assert not np.allclose(a, b)

    def test_diagnostics_report_no_failures_on_real_terrain(self, terrain_patch):
        config = get_preset("residuals")
        vec, diag = compute_embedding_from_config(terrain_patch, config, return_diagnostics=True)
        assert vec.shape[0] == 640
        assert isinstance(diag, dict)
        failed = diag.get("total_failed_pairs", diag.get("n_failed", 0))
        assert failed == 0, diag
