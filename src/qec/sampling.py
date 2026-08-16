# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Detector sampling into a dense (shots, T, H, W) volume, and diff frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from src.qec.circuits import detector_coords_to_grid_index, detector_grid_shape


@dataclass
class SampledShots:
    volume: np.ndarray       # uint8, shape (shots, T, H, W)
    observables: np.ndarray  # bool, shape (shots, num_observables)
    flat_detectors: np.ndarray  # bool, shape (shots, num_detectors); pre-grid, for decoding


def sample_detection_volume(
    circuit: stim.Circuit,
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
) -> SampledShots:
    sampler = circuit.compile_detector_sampler(seed=seed)
    flat_detectors, observables = sampler.sample(shots=shots, separate_observables=True)

    grid_index = detector_coords_to_grid_index(circuit)  # (num_detectors, 3)
    T, H, W = detector_grid_shape(distance, rounds)

    volume = np.zeros((shots, T, H, W), dtype=np.uint8)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]
    volume[:, t_idx, row_idx, col_idx] = flat_detectors.astype(np.uint8)

    return SampledShots(volume=volume, observables=observables, flat_detectors=flat_detectors)


def diff_frames(volume: np.ndarray) -> np.ndarray:
    """XOR of consecutive round-layers along the T axis (axis=1).

    diff[:, 0] = volume[:, 0] (no earlier frame to diff against).
    diff[:, t] = volume[:, t] ^ volume[:, t - 1] for t >= 1.
    Same shape and dtype as the input.
    """
    out = np.empty_like(volume)
    out[:, 0] = volume[:, 0]
    out[:, 1:] = volume[:, 1:] ^ volume[:, :-1]
    return out
