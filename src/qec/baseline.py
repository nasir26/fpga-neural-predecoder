# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""PyMatching-only baseline decoder: LER and single-shot timing."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pymatching
import stim

from src.qec import metrics


@dataclass
class BaselineResult:
    distance: int
    rounds: int
    basis: str
    p: float
    seed: int
    shots: int
    ler: float
    ler_ci_low: float
    ler_ci_high: float
    timing_shots: int
    mean_latency_us_per_shot: float
    mean_latency_us_per_round: float

    def to_dict(self) -> dict:
        return asdict(self)


def decode_baseline(
    circuit: stim.Circuit,
    matcher: pymatching.Matching,
    flat_detectors: np.ndarray,
    observables: np.ndarray,
    distance: int,
    rounds: int,
    basis: str,
    p: float,
    seed: int,
    timing_shots: int,
) -> BaselineResult:
    shots = flat_detectors.shape[0]
    if timing_shots > shots:
        raise ValueError("timing_shots must not exceed shots")

    predicted = matcher.decode_batch(flat_detectors).astype(bool)
    ler, ci_low, ci_high = metrics.logical_error_rate(predicted, observables)

    # Warm up once (first call pays one-time setup cost), excluded from timing.
    matcher.decode(flat_detectors[0])

    elapsed = np.empty(timing_shots, dtype=np.float64)
    for i in range(timing_shots):
        t0 = time.perf_counter()
        matcher.decode(flat_detectors[i])
        t1 = time.perf_counter()
        elapsed[i] = t1 - t0

    mean_latency_us_per_shot = float(np.mean(elapsed) * 1e6)
    mean_latency_us_per_round = mean_latency_us_per_shot / rounds

    return BaselineResult(
        distance=distance,
        rounds=rounds,
        basis=basis,
        p=p,
        seed=seed,
        shots=shots,
        ler=ler,
        ler_ci_low=ci_low,
        ler_ci_high=ci_high,
        timing_shots=timing_shots,
        mean_latency_us_per_shot=mean_latency_us_per_shot,
        mean_latency_us_per_round=mean_latency_us_per_round,
    )
