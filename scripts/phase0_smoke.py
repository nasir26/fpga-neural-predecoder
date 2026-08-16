# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 sanity check: d=3, rounds=3 rotated surface code, PyMatching LER.

Throwaway script, not part of the src/qec/ API designed in Phase 1.
"""
import sys

import numpy as np
import pymatching
import stim
import torch


def assert_environment() -> None:
    if torch.cuda.is_available():
        raise SystemExit("FAIL: torch.cuda.is_available() is True; CPU-only is required")
    forbidden = ["cupy", "cuquantum", "cudaq_qec", "tensorrt", "nvidia_modelopt"]
    for name in forbidden:
        if name in sys.modules:
            raise SystemExit(f"FAIL: forbidden package already imported: {name}")
        try:
            __import__(name)
        except ImportError:
            continue
        raise SystemExit(f"FAIL: forbidden package importable: {name}")
    print("environment OK: no CUDA, no forbidden GPU packages")


def smoke_decode(distance: int = 3, rounds: int = 3, p: float = 0.003, shots: int = 1000, seed: int = 0) -> float:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)

    predictions = matcher.decode_batch(detectors)
    errors = np.any(predictions != observables, axis=1)
    ler = float(np.mean(errors))
    return ler


if __name__ == "__main__":
    assert_environment()
    ler = smoke_decode()
    print(f"d=3, rounds=3, p=0.003, shots=1000 -> LER = {ler:.4f}")
