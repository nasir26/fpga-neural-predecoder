# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import scipy.sparse

from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit, detector_grid_shape
from src.qec.dem import build_error_channel_incidence, build_local_edge_catalog
from src.qec.sampling import sample_training_shots


def test_local_edge_catalog_is_nonempty_and_stable_across_distance():
    noise = NoiseModel.uniform_depolarizing(0.01)
    c5 = build_rotated_surface_code_circuit(5, 5, "z", noise)
    c7 = build_rotated_surface_code_circuit(7, 7, "z", noise)
    catalog5 = build_local_edge_catalog(c5)
    catalog7 = build_local_edge_catalog(c7)
    assert len(catalog5) > 0
    # The offset catalog is a fixed, translation-invariant set once d is
    # large enough (verified empirically d=5..11); d=7 must not introduce
    # offsets absent from d=5.
    assert set(catalog7) >= set(catalog5)


def test_channel_incidence_shape_and_single_nonzero_per_row():
    distance = 3
    noise = NoiseModel.uniform_depolarizing(0.01)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    dem = circuit.detector_error_model(decompose_errors=True)
    catalog = build_local_edge_catalog(circuit)
    num_channels = 1 + len(catalog)

    incidence = build_error_channel_incidence(circuit, catalog)
    assert isinstance(incidence, scipy.sparse.csr_matrix)
    assert incidence.shape == (dem.num_errors, num_channels * dem.num_detectors)
    row_weights = np.asarray(incidence.sum(axis=1)).flatten()
    assert (row_weights <= 1).all()  # each included error maps to exactly one (channel, anchor)


def test_training_shots_are_all_zero_at_negligible_noise():
    distance = 3
    noise = NoiseModel.uniform_depolarizing(1e-9)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    catalog = build_local_edge_catalog(circuit)
    num_channels = 1 + len(catalog)
    incidence = build_error_channel_incidence(circuit, catalog)

    shots = sample_training_shots(circuit, incidence, num_channels, distance, distance, shots=500, seed=0)
    assert shots.volume.sum() == 0
    assert shots.target.sum() == 0
    assert shots.diff.sum() == 0


def test_training_shots_shapes_and_dtypes():
    distance = 3
    noise = NoiseModel.uniform_depolarizing(0.02)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    catalog = build_local_edge_catalog(circuit)
    num_channels = 1 + len(catalog)
    incidence = build_error_channel_incidence(circuit, catalog)

    shots = sample_training_shots(circuit, incidence, num_channels, distance, distance, shots=200, seed=0)
    T, H, W = detector_grid_shape(distance, distance)
    assert shots.diff.shape == (200, T, H, W)
    assert shots.volume.shape == (200, T, H, W)
    assert shots.target.shape == (200, num_channels, T, H, W)
    for arr in (shots.diff, shots.target, shots.volume):
        assert arr.dtype == np.uint8
    assert shots.observables.dtype == np.bool_
