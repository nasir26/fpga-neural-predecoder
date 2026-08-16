# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""LER threshold-curve sanity check: at p well below threshold, LER must
decrease as distance increases, with non-overlapping Wilson CIs.
"""
from src.qec.baseline import decode_baseline
from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit
from src.qec.dem import build_detector_error_model, build_matching_graph
from src.qec.sampling import sample_detection_volume

SHOTS = 20_000
P = 0.003


def _run(distance: int) -> "decode_baseline.__annotations__":
    noise = NoiseModel.uniform_depolarizing(P)
    circuit = build_rotated_surface_code_circuit(distance, distance, "z", noise)
    dem = build_detector_error_model(circuit)
    matcher = build_matching_graph(dem)
    sampled = sample_detection_volume(circuit, distance, distance, SHOTS, seed=0)
    return decode_baseline(
        circuit=circuit,
        matcher=matcher,
        flat_detectors=sampled.flat_detectors,
        observables=sampled.observables,
        distance=distance,
        rounds=distance,
        basis="z",
        p=P,
        seed=0,
        timing_shots=200,
    )


def test_ler_decreases_with_distance_below_threshold():
    results = {d: _run(d) for d in (3, 5, 7)}

    for d in (3, 5, 7):
        r = results[d]
        assert 0.0 < r.ler < 0.5
        assert r.ler_ci_low <= r.ler <= r.ler_ci_high

    # Below-threshold behavior: increasing d must reduce LER, with
    # non-overlapping confidence intervals given SHOTS=20000 at p=0.003.
    assert results[3].ler_ci_low > results[5].ler_ci_high
    assert results[5].ler_ci_low > results[7].ler_ci_high


def test_baseline_timing_is_positive_and_finite():
    r = _run(3)
    assert r.mean_latency_us_per_shot > 0
    assert r.mean_latency_us_per_round > 0
    assert r.mean_latency_us_per_round == r.mean_latency_us_per_shot / r.rounds
