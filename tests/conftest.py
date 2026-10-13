"""Shared fixtures for the terravector test suite.

The repo is a collection of top-level scripts plus a ``src`` package that is
not installed; put the repo root on sys.path so ``import src...`` works.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def terrain_patch(rng):
    """A 64x64 synthetic terrain patch: gentle slope + a bump + mild noise."""
    y, x = np.mgrid[:64, :64].astype(np.float64)
    patch = 100.0 + 0.05 * x + 0.02 * y
    patch += 3.0 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 6.0**2))
    patch += rng.normal(scale=0.05, size=patch.shape)
    return patch


@pytest.fixture
def small_dem(rng):
    """A 128x128 DEM whose four 64x64 quadrants are all valid, varying terrain."""
    y, x = np.mgrid[:128, :128].astype(np.float64)
    dem = 200.0 + 0.1 * x - 0.03 * y + np.sin(x / 9.0) * 2.0 + np.cos(y / 11.0) * 1.5
    dem += rng.normal(scale=0.05, size=dem.shape)
    return dem
