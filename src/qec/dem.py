# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Detector error model, PyMatching graph, and the local-edge catalog used to
build ground-truth training targets from the DEM.

Local-edge design (see docs/DECISIONS.md, 2026-08-16 "per-voxel independent
thresholding breaks pair parity"): a naive per-voxel correction target lets
the network cancel one endpoint of a real two-detector fault while leaving
the other untouched, which breaks the parity structure PyMatching depends
on. Instead, weight-1 ("singleton", touches one detector) and weight-2
("paired", touches two detectors at a fixed relative offset) local DEM
errors get their own output channels, so a correction is always applied as a
complete unit (one detector for a singleton, both endpoints together for a
pair), never half-applied.

The set of distinct relative offsets among weight-2 local errors is a fixed,
finite, translation-invariant catalog for a given (basis, noise support): it
was verified empirically to be identical across d=5,7,9,11 for
rotated_memory_z (and the equivalent transposed set for rotated_memory_x).
It is still computed dynamically from the actual training circuit's DEM,
not hardcoded, and the resulting catalog is stored in the model checkpoint
so evaluation at other distances reuses exactly the same channel semantics
the model was trained with.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pymatching
import scipy.sparse
import stim

from src.qec.circuits import detector_coords_to_grid_index

Offset = Tuple[int, int, int]  # (dt, drow, dcol)


def build_detector_error_model(
    circuit: stim.Circuit, decompose_errors: bool = True
) -> stim.DetectorErrorModel:
    return circuit.detector_error_model(decompose_errors=decompose_errors)


def build_matching_graph(dem: stim.DetectorErrorModel) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(dem)


def _canonical_offset_and_anchor(
    grid_a: np.ndarray, grid_b: np.ndarray
) -> Tuple[Offset, int]:
    """Given two detectors' (t,row,col) grid coords, return (canonical
    offset, which_is_anchor) where which_is_anchor is 0 if a is the anchor
    (offset points a->b) or 1 if b is the anchor (offset points b->a).
    Canonical = lexicographically smaller of the two directed offsets, so
    the same physical pair always gets the same offset regardless of which
    detector index happened to be listed first in the DEM.
    """
    d_ab = tuple((grid_b - grid_a).tolist())
    d_ba = tuple((grid_a - grid_b).tolist())
    if d_ab <= d_ba:
        return d_ab, 0
    return d_ba, 1


def _iter_local_errors(circuit: stim.Circuit, max_error_weight: int = 2):
    """Yield (error_index, detector_ids, observable_ids) for every DEM error
    with 1 <= len(detector_ids) <= max_error_weight. error_index counts ALL
    errors (including excluded ones) so it lines up with dem row order.
    observable_ids is the (usually empty, sometimes non-empty) list of
    logical observables this error flips: see
    build_channel_observable_map for why this matters.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    error_index = 0
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        det_ids = [t.val for t in targets if t.is_relative_detector_id()]
        obs_ids = [t.val for t in targets if t.is_logical_observable_id()]
        if 1 <= len(det_ids) <= max_error_weight:
            yield error_index, det_ids, obs_ids
        error_index += 1


def _channel_and_anchor(
    det_ids: List[int], grid_index: np.ndarray, offset_to_channel: dict
):
    """(channel, anchor_detector_id) for a weight-1 or weight-2 local error,
    or None if it's a weight-2 error whose offset isn't in this catalog.
    """
    if len(det_ids) == 1:
        return 0, det_ids[0]
    a, b = det_ids
    offset, which = _canonical_offset_and_anchor(grid_index[a], grid_index[b])
    if offset not in offset_to_channel:
        return None
    return offset_to_channel[offset], (a if which == 0 else b)


def build_local_edge_catalog(circuit: stim.Circuit, max_error_weight: int = 2) -> List[Offset]:
    """Sorted list of distinct canonical offsets among weight-2 local DEM
    errors. Channel 0 (singleton, weight-1) is implicit and not included
    here; channel i+1 corresponds to catalog[i].
    """
    grid_index = detector_coords_to_grid_index(circuit)
    offsets = set()
    for _, det_ids, _ in _iter_local_errors(circuit, max_error_weight):
        if len(det_ids) == 2:
            a, b = det_ids
            offset, _ = _canonical_offset_and_anchor(grid_index[a], grid_index[b])
            offsets.add(offset)
    return sorted(offsets)


def build_error_channel_incidence(
    circuit: stim.Circuit, edge_offsets: List[Offset], max_error_weight: int = 2
) -> scipy.sparse.csr_matrix:
    """Sparse (num_errors, num_channels * num_detectors) 0/1 incidence.

    num_channels = 1 + len(edge_offsets) (channel 0 = singleton). Row i has
    exactly one nonzero entry, at column channel * num_detectors + anchor,
    for every included (weight 1 or 2, offset in catalog) error; excluded
    errors (weight >= 3, or weight 2 with an offset not in the catalog) get
    an all-zero row.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    num_errors = dem.num_errors
    num_detectors = dem.num_detectors
    num_channels = 1 + len(edge_offsets)
    offset_to_channel = {offset: i + 1 for i, offset in enumerate(edge_offsets)}
    grid_index = detector_coords_to_grid_index(circuit)

    rows: list = []
    cols: list = []
    for error_index, det_ids, _obs_ids in _iter_local_errors(circuit, max_error_weight):
        result = _channel_and_anchor(det_ids, grid_index, offset_to_channel)
        if result is None:
            continue
        channel, anchor = result
        rows.append(error_index)
        cols.append(channel * num_detectors + anchor)

    data = [1] * len(rows)
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_errors, num_channels * num_detectors), dtype="uint8"
    )


def build_channel_observable_map(
    circuit: stim.Circuit, edge_offsets: List[Offset], max_error_weight: int = 2
) -> np.ndarray:
    """(num_channels, num_detectors) uint8: obs_map[channel, anchor] = 1 if
    the local DEM error at that (channel, anchor) also flips the logical
    observable, else 0.

    Why this exists (see docs/DECISIONS.md, 2026-08-16 "silently cancelling
    a local error's syndrome without tracking its observable effect"):
    applying a local-edge correction (whether from ground truth or a model
    prediction) removes that error's contribution from the syndrome PyMatching
    sees, but does nothing on its own to account for whether that same error
    also flips the logical observable. About 12% of local errors do (measured
    at d=5, p=0.003). Silently cancelling those without tracking the implied
    logical flip made LER roughly 36x worse than no pre-decoder at all, even
    with a perfect oracle correction; this map is what lets a caller correctly
    XOR the local corrections' own observable parity into the final decoded
    result. Assumes exactly one logical observable (true for every circuit
    this project builds; asserted).
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_observables == 1, "build_channel_observable_map assumes a single logical observable"
    num_channels = 1 + len(edge_offsets)
    offset_to_channel = {offset: i + 1 for i, offset in enumerate(edge_offsets)}
    grid_index = detector_coords_to_grid_index(circuit)

    obs_map = np.zeros((num_channels, dem.num_detectors), dtype=np.uint8)
    for _error_index, det_ids, obs_ids in _iter_local_errors(circuit, max_error_weight):
        if not obs_ids:
            continue
        result = _channel_and_anchor(det_ids, grid_index, offset_to_channel)
        if result is None:
            continue
        channel, anchor = result
        # obs_ids can legitimately list the same observable more than once
        # (a single DEM error line can be a merger of multiple underlying
        # circuit-level mechanisms with the same footprint); the net effect
        # on the observable is XOR-parity, not "is the list non-empty", so
        # an even count means no net flip (a real physical cancellation).
        if len(obs_ids) % 2 == 1:
            obs_map[channel, anchor] ^= 1
    return obs_map
