# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from src.qec.circuits import (
    NoiseModel,
    build_rotated_surface_code_circuit,
    detector_coords_to_grid_index,
    detector_grid_shape,
)


@pytest.mark.parametrize("distance,rounds", [(3, 3), (5, 5), (7, 7)])
@pytest.mark.parametrize("basis", ["x", "z"])
def test_grid_matches_stim_detector_count(distance, rounds, basis):
    noise = NoiseModel.uniform_depolarizing(0.003)
    circuit = build_rotated_surface_code_circuit(distance, rounds, basis, noise)

    grid_index = detector_coords_to_grid_index(circuit)
    assert grid_index.shape == (circuit.num_detectors, 3)

    T, H, W = detector_grid_shape(distance, rounds)
    assert grid_index[:, 0].max() < T
    assert grid_index[:, 1].max() < H
    assert grid_index[:, 2].max() < W
    assert grid_index.min() >= 0


@pytest.mark.parametrize("distance,rounds", [(3, 3), (5, 5)])
@pytest.mark.parametrize("basis", ["x", "z"])
def test_grid_mapping_is_injective(distance, rounds, basis):
    noise = NoiseModel.uniform_depolarizing(0.003)
    circuit = build_rotated_surface_code_circuit(distance, rounds, basis, noise)
    grid_index = detector_coords_to_grid_index(circuit)

    T, H, W = detector_grid_shape(distance, rounds)
    flat = grid_index[:, 0] * H * W + grid_index[:, 1] * W + grid_index[:, 2]
    assert len(np.unique(flat)) == circuit.num_detectors


def test_full25_noise_model_not_implemented():
    noise = NoiseModel(kind="full25")
    with pytest.raises(NotImplementedError):
        build_rotated_surface_code_circuit(5, 5, "z", noise)


def test_invalid_basis_rejected():
    noise = NoiseModel.uniform_depolarizing(0.003)
    with pytest.raises(ValueError):
        build_rotated_surface_code_circuit(5, 5, "y", noise)
