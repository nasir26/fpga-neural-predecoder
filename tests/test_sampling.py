# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit, detector_grid_shape
from src.qec.sampling import diff_frames, sample_detection_volume


def test_sample_detection_volume_shapes_and_dtypes():
    distance, rounds, shots, seed = 3, 3, 50, 0
    noise = NoiseModel.uniform_depolarizing(0.01)
    circuit = build_rotated_surface_code_circuit(distance, rounds, "z", noise)
    sampled = sample_detection_volume(circuit, distance, rounds, shots, seed)

    T, H, W = detector_grid_shape(distance, rounds)
    assert sampled.volume.shape == (shots, T, H, W)
    assert sampled.volume.dtype == np.uint8
    assert sampled.observables.dtype == np.bool_
    assert sampled.flat_detectors.shape == (shots, circuit.num_detectors)
    # Every set bit in flat_detectors must show up exactly once in the grid.
    assert sampled.volume.sum() == sampled.flat_detectors.sum()


def test_sampling_is_deterministic_given_seed():
    distance, rounds, shots = 3, 3, 20
    noise = NoiseModel.uniform_depolarizing(0.01)
    circuit = build_rotated_surface_code_circuit(distance, rounds, "z", noise)
    a = sample_detection_volume(circuit, distance, rounds, shots, seed=42)
    b = sample_detection_volume(circuit, distance, rounds, shots, seed=42)
    np.testing.assert_array_equal(a.volume, b.volume)
    np.testing.assert_array_equal(a.observables, b.observables)


def test_diff_frames_shape_and_reconstruction():
    distance, rounds, shots = 3, 3, 20
    noise = NoiseModel.uniform_depolarizing(0.01)
    circuit = build_rotated_surface_code_circuit(distance, rounds, "z", noise)
    sampled = sample_detection_volume(circuit, distance, rounds, shots, seed=1)

    diff = diff_frames(sampled.volume)
    assert diff.shape == sampled.volume.shape
    assert diff.dtype == sampled.volume.dtype

    # Cumulative XOR of diff frames along T must reconstruct the volume.
    reconstructed = np.bitwise_xor.accumulate(diff, axis=1)
    np.testing.assert_array_equal(reconstructed, sampled.volume)
