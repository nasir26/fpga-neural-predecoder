# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""LER with Wilson confidence interval, syndrome density reduction, speedup."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def wilson_confidence_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")

    z = norm.ppf(0.5 + confidence / 2)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    adj = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    low = (center - adj) / denom
    high = (center + adj) / denom
    return float(max(0.0, low)), float(min(1.0, high))


def logical_error_rate(
    predicted_observables: np.ndarray, true_observables: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """A shot is a logical error if any observable bit mismatches.

    Returns (ler, ci_low, ci_high).
    """
    if predicted_observables.shape != true_observables.shape:
        raise ValueError("shape mismatch between predicted and true observables")
    n = predicted_observables.shape[0]
    errors = np.any(predicted_observables != true_observables, axis=1)
    k = int(np.sum(errors))
    ler = k / n
    low, high = wilson_confidence_interval(k, n, confidence=confidence)
    return ler, low, high


def syndrome_density_reduction(density_before: float, density_after: float) -> float:
    """Fraction of detection events removed: 1 - density_after / density_before."""
    if density_before <= 0:
        raise ValueError("density_before must be positive")
    return 1.0 - density_after / density_before


def speedup(baseline_us: float, treatment_us: float) -> float:
    if treatment_us <= 0:
        raise ValueError("treatment_us must be positive")
    return baseline_us / treatment_us
