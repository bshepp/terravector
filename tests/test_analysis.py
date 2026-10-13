"""Tests for the rich per-residual analysis vector, including the signed-log1p
compression of heavy-tailed amplitude statistics (2026-10-12 index-compat event)."""
import numpy as np
import pytest

from src.features.analysis import (
    LOG_SCALED_KEYS,
    analysis_to_vector,
    analyze_features,
    get_analysis_dim,
    get_analysis_labels,
    get_log_scaled_dims,
    signed_log1p,
)


class TestSignedLog1p:
    def test_is_odd(self):
        x = np.array([-1000.0, -1.0, -0.1, 0.0, 0.1, 1.0, 1000.0])
        np.testing.assert_allclose(signed_log1p(-x), -signed_log1p(x))

    def test_is_monotonic(self):
        x = np.linspace(-1e4, 1e4, 2001)
        assert np.all(np.diff(signed_log1p(x)) > 0)

    def test_identity_near_zero(self):
        x = np.array([-1e-4, 1e-5, 1e-4])
        np.testing.assert_allclose(signed_log1p(x), x, rtol=1e-3)

    def test_log_for_large_values(self):
        assert signed_log1p(1e6) == pytest.approx(np.log1p(1e6))
        assert signed_log1p(1e6) < 15  # compressed from a million to ~13.8

    def test_zero_maps_to_zero(self):
        assert signed_log1p(0.0) == 0.0


class TestAnalysisVector:
    def test_shape_and_dtype(self, terrain_patch):
        vec = analysis_to_vector(analyze_features(terrain_patch))
        assert vec.shape == (get_analysis_dim(),) == (20,)
        assert vec.dtype == np.float32

    def test_finite_on_real_patch(self, terrain_patch):
        vec = analysis_to_vector(analyze_features(terrain_patch))
        assert np.all(np.isfinite(vec))

    def test_log_scaled_keys_are_compressed(self):
        # A raw analysis dict with one extreme amplitude statistic.
        analysis = {"mean": 1.0e6, "energy": -5.0e4, "entropy": 3.5}
        vec = analysis_to_vector(analysis)
        labels = get_analysis_labels()
        assert vec[labels.index("slog_mean")] == pytest.approx(np.log1p(1.0e6), rel=1e-6)
        assert vec[labels.index("slog_energy")] == pytest.approx(-np.log1p(5.0e4), rel=1e-6)

    def test_unscaled_keys_pass_through_raw(self):
        analysis = {"entropy": 3.5, "spatial_autocorr": 0.7, "feature_snr": 2.25}
        vec = analysis_to_vector(analysis)
        labels = get_analysis_labels()
        assert vec[labels.index("entropy")] == pytest.approx(3.5)
        assert vec[labels.index("spatial_autocorr")] == pytest.approx(0.7)
        assert vec[labels.index("log_snr")] == pytest.approx(2.25)

    def test_missing_keys_default_to_zero(self):
        vec = analysis_to_vector({})
        assert np.all(vec == 0)

    def test_log_scaled_dims_match_labels(self):
        labels = get_analysis_labels()
        dims = get_log_scaled_dims()
        assert dims == sorted(dims)
        assert len(dims) == len(LOG_SCALED_KEYS) == 13
        for i, label in enumerate(labels):
            assert (i in dims) == label.startswith("slog_"), (i, label)

    def test_labels_cover_every_dim(self):
        labels = get_analysis_labels()
        assert len(labels) == get_analysis_dim() == 20
        assert len(set(labels)) == len(labels)


class TestAnalyzeFeatures:
    def test_returns_all_keys_the_vector_needs(self, terrain_patch):
        analysis = analyze_features(terrain_patch)
        required = {
            "mean", "std", "range", "median", "skewness", "kurtosis", "energy", "entropy",
            "spatial_autocorr", "linear_feature_count", "max_linear_strength",
            "mean_linear_strength", "low_freq_energy", "mid_freq_energy",
            "high_freq_energy", "freq_ratio_high_low", "feature_snr", "min", "max",
        }
        assert required <= set(analysis)
        assert all(np.isfinite(v) for v in analysis.values())

    def test_basic_statistics_are_correct(self, terrain_patch):
        analysis = analyze_features(terrain_patch, include_linear=False, include_frequency=False)
        assert analysis["mean"] == pytest.approx(terrain_patch.mean())
        assert analysis["std"] == pytest.approx(terrain_patch.std())
        assert analysis["range"] == pytest.approx(terrain_patch.max() - terrain_patch.min())

    def test_nans_are_treated_as_zero(self, terrain_patch):
        patch = terrain_patch.copy()
        patch[0, :5] = np.nan
        analysis = analyze_features(patch, include_linear=False, include_frequency=False)
        assert np.isfinite(analysis["mean"])
        assert analysis["min"] == pytest.approx(0.0)
