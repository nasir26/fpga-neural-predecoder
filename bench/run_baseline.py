# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Run the PyMatching-only baseline and write results/metrics/baseline_*.json.

Usage:
    python bench/run_baseline.py [--config conf/default.yaml]
        [--distance D] [--rounds R] [--basis x|z] [--p P]
        [--shots N] [--timing-shots N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qec.baseline import decode_baseline
from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit
from src.qec.dem import build_detector_error_model, build_matching_graph
from src.qec.sampling import sample_detection_volume


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "conf" / "default.yaml")
    parser.add_argument("--distance", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--basis", type=str, default=None, choices=["x", "z"])
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--timing-shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    distance = args.distance if args.distance is not None else cfg["code"]["distance"]
    rounds = args.rounds if args.rounds is not None else cfg["code"]["rounds"]
    basis = args.basis if args.basis is not None else cfg["code"]["basis"]
    p = args.p if args.p is not None else cfg["noise"]["p"]
    shots = args.shots if args.shots is not None else cfg["sampling"]["shots"]
    timing_shots = args.timing_shots if args.timing_shots is not None else cfg["baseline"]["timing_shots"]
    seed = args.seed if args.seed is not None else cfg["seed"]

    noise = NoiseModel.uniform_depolarizing(p)
    circuit = build_rotated_surface_code_circuit(distance, rounds, basis, noise)
    dem = build_detector_error_model(circuit)
    matcher = build_matching_graph(dem)

    sampled = sample_detection_volume(circuit, distance, rounds, shots, seed)

    result = decode_baseline(
        circuit=circuit,
        matcher=matcher,
        flat_detectors=sampled.flat_detectors,
        observables=sampled.observables,
        distance=distance,
        rounds=rounds,
        basis=basis,
        p=p,
        seed=seed,
        timing_shots=timing_shots,
    )

    out_dir = REPO_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_d{distance}_r{rounds}_{basis}_p{p}.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"wrote {out_path}")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
