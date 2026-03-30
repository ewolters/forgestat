"""Distribution sampling — convenience functions for Monte Carlo and simulation.

Reproducible via seeded RNG. Pure numpy.
"""

from __future__ import annotations

import numpy as np


def seeded_rng(seed: int = 42) -> np.random.Generator:
    """Create a reproducible random number generator.

    Args:
        seed: RNG seed.

    Returns:
        numpy Generator instance.
    """
    return np.random.default_rng(seed)


def sample_normal(
    mean: float = 0,
    std: float = 1,
    n: int = 100,
    seed: int | None = None,
) -> list[float]:
    """Sample from normal distribution.

    Args:
        mean: Distribution mean.
        std: Standard deviation.
        n: Number of samples.
        seed: RNG seed (None = random).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n).tolist()


def sample_exponential(
    mean: float = 1,
    n: int = 100,
    seed: int | None = None,
) -> list[float]:
    """Sample from exponential distribution.

    Args:
        mean: Mean (= 1/rate).
        n: Number of samples.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    return rng.exponential(mean, n).tolist()


def sample_weibull(
    shape: float = 2,
    scale: float = 1,
    n: int = 100,
    seed: int | None = None,
) -> list[float]:
    """Sample from Weibull distribution.

    Args:
        shape: Shape parameter (β).
        scale: Scale parameter (η).
        n: Number of samples.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    return (rng.weibull(shape, n) * scale).tolist()


def sample_uniform(
    low: float = 0,
    high: float = 1,
    n: int = 100,
    seed: int | None = None,
) -> list[float]:
    """Sample from uniform distribution.

    Args:
        low: Lower bound.
        high: Upper bound.
        n: Number of samples.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, n).tolist()


def sample_poisson(
    lam: float = 5,
    n: int = 100,
    seed: int | None = None,
) -> list[int]:
    """Sample from Poisson distribution.

    Args:
        lam: Rate parameter (λ).
        n: Number of samples.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    return rng.poisson(lam, n).tolist()
