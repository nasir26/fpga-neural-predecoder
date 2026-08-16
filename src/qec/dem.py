# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Detector error model, PyMatching graph, and the error->detector incidence
matrix used to build ground-truth training targets from the DEM.
"""
from __future__ import annotations

import pymatching
import scipy.sparse
import stim


def build_detector_error_model(
    circuit: stim.Circuit, decompose_errors: bool = True
) -> stim.DetectorErrorModel:
    return circuit.detector_error_model(decompose_errors=decompose_errors)


def build_matching_graph(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(dem)


def build_error_incidence_matrix(
    dem: stim.DetectorErrorModel, max_error_weight: int = 2
) -> scipy.sparse.csr_matrix:
    """Sparse (num_errors, num_detectors) 0/1 incidence matrix.

    Row i lists which detectors DEM error mechanism i touches (its full
    toggle footprint, ignoring any "^" suggested-decomposition separators,
    since firing that error toggles all listed detectors regardless of how
    it's split for matching-graph construction).

    Errors touching more than `max_error_weight` detectors get an all-zero
    row: they are excluded from the ground-truth "local fault frame" target
    (see docs/DECISIONS.md), not from the DEM itself.
    """
    num_errors = dem.num_errors
    num_detectors = dem.num_detectors

    rows: list = []
    cols: list = []
    error_index = 0
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        det_ids = [t.val for t in targets if t.is_relative_detector_id()]
        if len(det_ids) <= max_error_weight:
            for det_id in det_ids:
                rows.append(error_index)
                cols.append(det_id)
        error_index += 1
    assert error_index == num_errors

    data = [1] * len(rows)
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_errors, num_detectors), dtype="uint8"
    )
