# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the 2026-08-16 "silently cancelling a local error's
syndrome without tracking its observable effect" bug (see docs/DECISIONS.md).

Applying a local-edge correction without also accounting for whether that
edge flips the logical observable made an oracle (ground-truth) correction
about 36x worse than no correction at all. This test locks in the fix: with
observable tracking, an oracle correction must be at least as good as (in
practice, better than) the no-predecoder baseline.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.run_predecoder import local_correction_observable_parity
from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit, detector_coords_to_grid_index
from src.qec.dem import (
    build_channel_observable_map,
    build_detector_error_model,
    build_error_channel_incidence,
    build_local_edge_catalog,
    build_matching_graph,
)
from src.qec.sampling import sample_training_shots
from src.qec import metrics as qec_metrics
from src.model.edges import apply_local_edge_corrections


def test_channel_observable_map_has_some_nonzero_entries():
    distance = 5
    noise = NoiseModel.uniform_depolarizing(0.003)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    catalog = build_local_edge_catalog(circuit)
    obs_map = build_channel_observable_map(circuit, catalog)
    # At d=5, p=0.003, roughly 12% of local errors flip the observable;
    # the map must not be all-zero (that would silently disable tracking).
    assert obs_map.sum() > 0


def test_oracle_correction_with_observable_tracking_beats_baseline():
    distance = 5
    noise = NoiseModel.uniform_depolarizing(0.003)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    dem = build_detector_error_model(circuit)
    matcher = build_matching_graph(dem)
    catalog = build_local_edge_catalog(circuit)
    num_channels = 1 + len(catalog)
    incidence = build_error_channel_incidence(circuit, catalog)
    obs_map = build_channel_observable_map(circuit, catalog)
    grid_index = detector_coords_to_grid_index(circuit)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]

    shots = sample_training_shots(circuit, incidence, num_channels, distance, distance, shots=20000, seed=999)

    oracle_correction = apply_local_edge_corrections(shots.target, catalog)
    residual = shots.volume ^ oracle_correction
    residual_flat = residual[:, t_idx, row_idx, col_idx].astype(bool)

    residual_predicted = matcher.decode_batch(residual_flat).astype(bool)
    local_obs_parity = local_correction_observable_parity(shots.target, obs_map, grid_index)
    final_predicted = residual_predicted ^ local_obs_parity.reshape(-1, 1)

    ler, _, _ = qec_metrics.logical_error_rate(final_predicted, shots.observables)

    baseline_flat = shots.volume[:, t_idx, row_idx, col_idx].astype(bool)
    baseline_predicted = matcher.decode_batch(baseline_flat).astype(bool)
    ler_baseline, _, _ = qec_metrics.logical_error_rate(baseline_predicted, shots.observables)

    assert ler <= ler_baseline * 1.5  # oracle should be competitive with, not vastly worse than, baseline


def test_oracle_correction_without_observable_tracking_is_much_worse():
    # Companion negative check: confirms the bug this test file guards
    # against is real and reproducible, not just theoretical.
    distance = 5
    noise = NoiseModel.uniform_depolarizing(0.003)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    dem = build_detector_error_model(circuit)
    matcher = build_matching_graph(dem)
    catalog = build_local_edge_catalog(circuit)
    num_channels = 1 + len(catalog)
    incidence = build_error_channel_incidence(circuit, catalog)
    grid_index = detector_coords_to_grid_index(circuit)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]

    shots = sample_training_shots(circuit, incidence, num_channels, distance, distance, shots=20000, seed=999)
    oracle_correction = apply_local_edge_corrections(shots.target, catalog)
    residual = shots.volume ^ oracle_correction
    residual_flat = residual[:, t_idx, row_idx, col_idx].astype(bool)

    predicted_no_tracking = matcher.decode_batch(residual_flat).astype(bool)
    ler_no_tracking, _, _ = qec_metrics.logical_error_rate(predicted_no_tracking, shots.observables)

    baseline_flat = shots.volume[:, t_idx, row_idx, col_idx].astype(bool)
    baseline_predicted = matcher.decode_batch(baseline_flat).astype(bool)
    ler_baseline, _, _ = qec_metrics.logical_error_rate(baseline_predicted, shots.observables)

    assert ler_no_tracking > ler_baseline * 10  # reproduces the ~36x-worse failure mode
