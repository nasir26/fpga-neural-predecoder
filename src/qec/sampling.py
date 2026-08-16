# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Detector sampling into a dense (shots, T, H, W) volume, and diff frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import stim

from src.qec.circuits import detector_coords_to_grid_index, detector_grid_shape


@dataclass
class SampledShots:
    volume: np.ndarray       # uint8, shape (shots, T, H, W)
    observables: np.ndarray  # bool, shape (shots, num_observables)
    flat_detectors: np.ndarray  # bool, shape (shots, num_detectors); pre-grid, for decoding


@dataclass
class TrainingShots:
    diff: np.ndarray         # uint8, shape (shots, T, H, W); network input
    target: np.ndarray       # uint8, shape (shots, C, T, H, W); per-edge-channel ground truth
    volume: np.ndarray       # uint8, shape (shots, T, H, W); raw detection events
    observables: np.ndarray  # bool, shape (shots, num_observables)


def _scatter_to_grid(
    flat_bits: np.ndarray, grid_index: np.ndarray, shots: int, T: int, H: int, W: int
) -> np.ndarray:
    grid = np.zeros((shots, T, H, W), dtype=np.uint8)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]
    grid[:, t_idx, row_idx, col_idx] = flat_bits.astype(np.uint8)
    return grid


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
    volume = _scatter_to_grid(flat_detectors, grid_index, shots, T, H, W)

    return SampledShots(volume=volume, observables=observables, flat_detectors=flat_detectors)


def sample_training_shots(
    circuit: stim.Circuit,
    channel_incidence: scipy.sparse.csr_matrix,
    num_channels: int,
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
) -> TrainingShots:
    """DEM-level sampling: gives both the detection volume and which DEM
    error mechanisms fired, needed to build the ground-truth target (see
    src.qec.dem.build_error_channel_incidence). Slower than
    sample_detection_volume (no error tracking there), used only for
    training data generation.

    channel_incidence: sparse (num_errors, num_channels * num_detectors),
    from build_error_channel_incidence. num_channels must match how it was
    built (1 + len(edge_offsets)).
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    dem_sampler = dem.compile_sampler(seed=seed)
    flat_detectors, observables, error_data = dem_sampler.sample(shots=shots, return_errors=True)

    grid_index = detector_coords_to_grid_index(circuit)
    T, H, W = detector_grid_shape(distance, rounds)
    num_detectors = flat_detectors.shape[1]
    volume = _scatter_to_grid(flat_detectors, grid_index, shots, T, H, W)

    # XOR-parity of local fired-error footprints, in GF(2) via matmul + mod 2.
    flat_target = channel_incidence.T.dot(error_data.astype(np.uint8).T).T % 2  # (shots, C*num_detectors)
    flat_target = flat_target.reshape(shots, num_channels, num_detectors).astype(bool)

    target = np.zeros((shots, num_channels, T, H, W), dtype=np.uint8)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]
    target[:, :, t_idx, row_idx, col_idx] = flat_target.astype(np.uint8)

    diff = diff_frames(volume)

    return TrainingShots(diff=diff, target=target, volume=volume, observables=observables)


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
