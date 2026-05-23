"""
Gray-Scott Reaction-Diffusion Pattern Generator

Used as an intermediate amplification stage in the residuals signature
pipeline. Reaction-diffusion is highly sensitive to initial conditions:
small differences in the seed field settle into discretely different
attractor textures (spots, stripes, labyrinths). This makes two patches
with nearly-identical statistical fingerprints diverge in the Turing
output, improving downstream HNSW distinguishability.

The solver uses periodic boundaries via np.roll for speed and
allocates only u/v plus two Laplacian buffers per iteration.
"""

from typing import Tuple

import numpy as np

# Classic Gray-Scott parameters that yield well-developed pattern within
# a few hundred iterations. Du > Dv puts the system in the Turing regime.
GRAY_SCOTT_DEFAULTS = {
    "Du": 0.16,
    "Dv": 0.08,
    "F": 0.035,
    "k": 0.060,
    "dt": 1.0,
}


def _laplacian5(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian with periodic boundaries."""
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def _seed_from_field(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize an arbitrary 2-D field into Gray-Scott initial (u, v).

    v is mapped to [0, 0.5] so the activator stays in the typical
    operating range; u = 1 - v keeps the substrate near equilibrium.
    A constant field yields a uniformly-seeded (u, v) which Gray-Scott
    will not pattern — that is the correct behavior for featureless input.
    """
    f = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fmin = float(f.min())
    fmax = float(f.max())
    span = fmax - fmin
    if span < 1e-12:
        v = np.full_like(f, 0.25, dtype=np.float32)
    else:
        v = ((f - fmin) / span) * 0.5
        v = v.astype(np.float32)
    u = (1.0 - v).astype(np.float32)
    return u, v


def gray_scott_pattern(
    seed: np.ndarray,
    n_iter: int = 50,
    Du: float = GRAY_SCOTT_DEFAULTS["Du"],
    Dv: float = GRAY_SCOTT_DEFAULTS["Dv"],
    F: float = GRAY_SCOTT_DEFAULTS["F"],
    k: float = GRAY_SCOTT_DEFAULTS["k"],
    dt: float = GRAY_SCOTT_DEFAULTS["dt"],
) -> np.ndarray:
    """Run n_iter Gray-Scott steps seeded by `seed`, return the v field.

    Args:
        seed: 2-D array used to seed the activator v. Any range is OK;
            it gets normalized internally to [0, 0.5].
        n_iter: Number of explicit-Euler integration steps.
        Du, Dv: Diffusion coefficients for substrate and activator.
        F: Feed rate.
        k: Kill rate.
        dt: Integration timestep. 1.0 is stable for the default Du/Dv on
            unit grid spacing; lower it if you push Du beyond ~0.2.

    Returns:
        2-D float32 array (same shape as seed) holding the final v field —
        the pattern. Take statistics of this as a tile-distinguishability
        feature.
    """
    if seed.ndim != 2:
        raise ValueError(f"gray_scott_pattern needs a 2-D seed, got shape {seed.shape}")
    if n_iter <= 0:
        # Caller asked for no iterations — return the normalized seed itself.
        _, v0 = _seed_from_field(seed)
        return v0

    u, v = _seed_from_field(seed)

    for _ in range(n_iter):
        Lu = _laplacian5(u)
        Lv = _laplacian5(v)
        uvv = u * v * v
        u = u + dt * (Du * Lu - uvv + F * (1.0 - u))
        v = v + dt * (Dv * Lv + uvv - (F + k) * v)

    return v
