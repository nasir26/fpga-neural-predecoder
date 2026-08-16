# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from src.model.edges import apply_local_edge_corrections


def test_singleton_channel_flips_only_its_own_voxel():
    activation = np.zeros((1, 1, 4, 4, 4), dtype=np.uint8)
    activation[0, 0, 1, 2, 3] = 1
    correction = apply_local_edge_corrections(activation, edge_offsets=[])
    assert correction.shape == (1, 4, 4, 4)
    assert correction.sum() == 1
    assert correction[0, 1, 2, 3] == 1


def test_paired_channel_flips_both_endpoints():
    edge_offsets = [(1, 0, 0), (0, 1, 1)]
    activation = np.zeros((1, 3, 4, 4, 4), dtype=np.uint8)
    activation[0, 1, 1, 1, 1] = 1  # channel 1 -> offset (1,0,0), anchor (1,1,1)
    correction = apply_local_edge_corrections(activation, edge_offsets)
    nonzero = set(map(tuple, np.argwhere(correction[0]).tolist()))
    assert nonzero == {(1, 1, 1), (2, 1, 1)}


def test_out_of_bounds_partner_is_dropped_not_crashed():
    edge_offsets = [(0, 1, 1)]
    activation = np.zeros((1, 2, 4, 4, 4), dtype=np.uint8)
    activation[0, 1, 0, 3, 3] = 1  # partner would be (0,4,4): out of a 4x4x4 grid
    correction = apply_local_edge_corrections(activation, edge_offsets)
    nonzero = set(map(tuple, np.argwhere(correction[0]).tolist()))
    assert nonzero == {(0, 3, 3)}  # anchor still flips; partner silently dropped


def test_overlapping_edges_xor_correctly():
    # Two pair activations sharing a partner voxel must XOR (cancel), not OR.
    edge_offsets = [(0, 0, 1)]
    activation = np.zeros((1, 2, 4, 4, 4), dtype=np.uint8)
    activation[0, 1, 0, 0, 0] = 1  # anchor (0,0,0) -> partner (0,0,1)
    activation[0, 1, 0, 0, 1] = 1  # anchor (0,0,1) -> partner (0,0,2); also flips (0,0,1) again
    correction = apply_local_edge_corrections(activation, edge_offsets)
    # (0,0,1) is touched twice (once as partner of first, once as anchor of
    # second) -> should cancel back to 0 under XOR.
    assert correction[0, 0, 0, 1] == 0
    assert correction[0, 0, 0, 0] == 1
    assert correction[0, 0, 0, 2] == 1


def test_num_channels_mismatch_raises():
    activation = np.zeros((1, 2, 4, 4, 4), dtype=np.uint8)
    try:
        apply_local_edge_corrections(activation, edge_offsets=[(1, 0, 0), (0, 1, 0)])
        assert False, "expected ValueError"
    except ValueError:
        pass
