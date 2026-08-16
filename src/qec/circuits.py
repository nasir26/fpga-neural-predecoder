# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Rotated surface-code memory circuits (Stim), X and Z basis only.

Scope note: stim.Circuit.generated exposes exactly two rotated surface code
tasks, rotated_memory_x and rotated_memory_z. There is no third or fourth
built-in orientation, and the rotated surface code's boundary symmetry makes
a hand-built third/fourth orientation physically equivalent to these two
under relabeling, so only X and Z basis are implemented here. See
docs/LIMITATIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import stim

VALID_BASES = ("x", "z")

# The 15 non-identity two-qubit Pauli channel labels used by a full
# CNOT-level Pauli error model (P (x) Q for P, Q in {I, X, Y, Z}, excluding
# I (x) I). Field names for NoiseModel.cnot_pauli_probs when kind="full25".
CNOT_PAULI_LABELS: Tuple[str, ...] = (
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
)


@dataclass(frozen=True)
class NoiseModel:
    """Circuit-level noise configuration.

    kind="uniform_depolarizing" is the only implemented path: a single `p`
    maps onto stim.Circuit.generated's four built-in noise knobs (after
    Clifford depolarization, before-round data depolarization, before-measure
    flip, after-reset flip), all set to the same value.

    kind="full25" is a placeholder for a hand-built circuit with independent
    SPAM, idle, and per-Pauli-channel CNOT error rates. Not implemented; see
    docs/LIMITATIONS.md. The field names below are reserved so config files
    and downstream code can already speak the full 25-parameter shape.
    """

    kind: str
    p: Optional[float] = None
    reset_flip_prob: Optional[float] = None
    measure_flip_prob: Optional[float] = None
    idle_depolarize_prob: Optional[float] = None
    cnot_pauli_probs: Optional[Dict[str, float]] = None
    spam_extra: Optional[Dict[str, float]] = None

    @classmethod
    def uniform_depolarizing(cls, p: float) -> "NoiseModel":
        return cls(kind="uniform_depolarizing", p=p)

    def __post_init__(self) -> None:
        if self.kind == "uniform_depolarizing":
            if self.p is None:
                raise ValueError("uniform_depolarizing requires p")
        elif self.kind == "full25":
            pass
        else:
            raise ValueError(f"unknown NoiseModel kind: {self.kind!r}")


def build_rotated_surface_code_circuit(
    distance: int,
    rounds: int,
    basis: str,
    noise: NoiseModel,
) -> stim.Circuit:
    if basis not in VALID_BASES:
        raise ValueError(f"basis must be one of {VALID_BASES}, got {basis!r}")
    if noise.kind == "full25":
        raise NotImplementedError(
            "full25 noise model requires a hand-built circuit with explicit "
            "PAULI_CHANNEL_1/PAULI_CHANNEL_2 injection; not implemented. "
            "See docs/LIMITATIONS.md."
        )
    assert noise.kind == "uniform_depolarizing"
    p = noise.p
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def detector_grid_shape(distance: int, rounds: int) -> Tuple[int, int, int]:
    """(T, H, W) for the dense detector grid: T=rounds+1, H=W=distance+1."""
    return (rounds + 1, distance + 1, distance + 1)


def detector_coords_to_grid_index(circuit: stim.Circuit) -> np.ndarray:
    """Map each detector index to a dense (t, row, col) grid cell.

    Stim's rotated-surface-code detector coordinates are (x, y, t) with x, y
    on an even-integer grid (step 2) and t the round index (0..rounds
    inclusive, where t=0 and t=rounds are half-density boundary layers).
    Dividing x, y by 2 gives 0-indexed row/col in a dense (distance+1) x
    (distance+1) grid with no collisions (verified against circuit.num_detectors).

    Returns:
        int array of shape (num_detectors, 3), columns (t, row, col), ordered
        by detector index 0..num_detectors-1.
    """
    coords = circuit.get_detector_coordinates()
    num_detectors = circuit.num_detectors
    out = np.zeros((num_detectors, 3), dtype=np.int64)
    for det_index in range(num_detectors):
        x, y, t = coords[det_index]
        out[det_index] = (int(round(t)), int(round(y / 2)), int(round(x / 2)))
    return out
