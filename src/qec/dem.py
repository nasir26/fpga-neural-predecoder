# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Detector error model and PyMatching graph construction."""
from __future__ import annotations

import pymatching
import stim


def build_detector_error_model(
    circuit: stim.Circuit, decompose_errors: bool = True
) -> stim.DetectorErrorModel:
    return circuit.detector_error_model(decompose_errors=decompose_errors)


def build_matching_graph(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(dem)
