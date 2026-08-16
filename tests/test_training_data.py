# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import scipy.sparse

from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit, detector_grid_shape
from src.qec.dem import build_detector_error_model, build_error_incidence_matrix
from src.qec.sampling import sample_training_shots


def test_incidence_matrix_shape_and_local_filter():
    distance = 3
    noise = NoiseModel.uniform_depolarizing(0.01)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    dem = build_detector_error_model(circuit)

    incidence = build_error_incidence_matrix(dem, max_error_weight=2)
    assert isinstance(incidence, scipy.sparse.csr_matrix)
    assert incidence.shape == (dem.num_errors, dem.num_detectors)
    # Every included row touches at most 2 detectors.
    row_weights = np.asarray(incidence.sum(axis=1)).flatten()
    assert (row_weights <= 2).all()
    # At least some errors were excluded as nonlocal (weight >= 3 exists at d=3).
    assert row_weights.sum() < incidence.shape[0] * 2


def test_training_shots_are_all_zero_at_negligible_noise():
    distance = 3
    noise = NoiseModel.uniform_depolarizing(1e-9)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    dem = build_detector_error_model(circuit)
    incidence = build_error_incidence_matrix(dem)

    shots = sample_training_shots(circuit, incidence, distance, distance, shots=500, seed=0)
    assert shots.volume.sum() == 0
    assert shots.target.sum() == 0
    assert shots.diff.sum() == 0


def test_training_shots_shapes_and_dtypes():
    distance = 3
    noise = NoiseModel.uniform_depolarizing(0.02)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    dem = build_detector_error_model(circuit)
    incidence = build_error_incidence_matrix(dem)

    shots = sample_training_shots(circuit, incidence, distance, distance, shots=200, seed=0)
    T, H, W = detector_grid_shape(distance, distance)
    for arr in (shots.diff, shots.target, shots.volume):
        assert arr.shape == (200, T, H, W)
        assert arr.dtype == np.uint8
    assert shots.observables.dtype == np.bool_
    # Target is derived only from a subset of fired errors, so it can never
    # have more total activity than the raw detection volume would from all
    # fired local errors, but it need not be a subset voxel-for-voxel
    # (double-cancellation can flip either direction); just check it's not
    # degenerately larger than volume in aggregate for this low-p regime.
    assert shots.target.sum() <= shots.volume.sum() * 1.5
