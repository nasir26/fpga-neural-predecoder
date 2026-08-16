# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Turn a per-(voxel, edge-channel) activation grid into a single-channel
correction volume, applying every edge as a complete unit (see
src/qec/dem.py module docstring for why: never half-cancel a pair).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

Offset = Tuple[int, int, int]


def _xor_shifted(dst: np.ndarray, src: np.ndarray, offset: Offset) -> None:
    """dst[:, t+dt, y+dy, x+dx] ^= src[:, t, y, x] for every in-bounds
    position; out-of-bounds contributions are dropped (safe: just means
    that edge isn't applied right at the grid boundary).
    """
    dt, dy, dx = offset
    T, H, W = src.shape[1:]
    t0, t1 = max(0, -dt), min(T, T - dt)
    y0, y1 = max(0, -dy), min(H, H - dy)
    x0, x1 = max(0, -dx), min(W, W - dx)
    if t0 >= t1 or y0 >= y1 or x0 >= x1:
        return
    dst[:, t0 + dt : t1 + dt, y0 + dy : y1 + dy, x0 + dx : x1 + dx] ^= src[:, t0:t1, y0:y1, x0:x1]


def apply_local_edge_corrections(activation: np.ndarray, edge_offsets: List[Offset]) -> np.ndarray:
    """activation: uint8/bool, shape (shots, C, T, H, W), C = 1 + len(edge_offsets).

    Channel 0 (singleton) flips its own voxel. Channel i+1 (paired, offset
    edge_offsets[i]) flips both the anchor voxel and the anchor + offset
    voxel together, so a pair is always applied as a whole or not at all.

    Returns: uint8, shape (shots, T, H, W).
    """
    num_channels = activation.shape[1]
    if num_channels != 1 + len(edge_offsets):
        raise ValueError(
            f"activation has {num_channels} channels but edge_offsets implies "
            f"{1 + len(edge_offsets)}"
        )

    correction = activation[:, 0].astype(np.uint8).copy()
    for i, offset in enumerate(edge_offsets):
        act = activation[:, i + 1].astype(np.uint8)
        correction ^= act  # anchor endpoint
        _xor_shifted(correction, act, offset)  # partner endpoint
    return correction
