# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Phase 6: full benchmark grid across distance x noise rate x precision.

R=5 only -- R=9 training remains deferred (see docs/LIMITATIONS.md, decided
in Phase 2); this sweep does not retrain anything, only evaluates the four
already-trained R=5 checkpoints (fp32, int8, int4, ternary) at every
(distance, p) combination, generating any missing PyMatching-only baseline
results along the way. Writes one combined results/metrics/phase6_sweep.json
that scripts/make_figures.py reads to regenerate every figure with no
network access and no retraining.

Usage:
    python bench/run_phase6_sweep.py
        --distances 5 7 9 11 13 --ps 0.001 0.002 0.003 0.005
        --shots 100000 --timing-shots 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.run_predecoder import evaluate_at_distance, load_checkpoint
from bench.sweep_precision import _enrich_with_baseline, load_qat_checkpoint
from src.model.quantize import PRECISIONS
from src.qec.baseline import decode_baseline
from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit
from src.qec.dem import build_detector_error_model, build_matching_graph
from src.qec.sampling import sample_detection_volume


def ensure_baseline(
    distance: int, p: float, basis: str, shots: int, timing_shots: int, seed: int, out_dir: Path
) -> Path:
    """Returns the path to results/metrics/baseline_d{d}_r{d}_{basis}_p{p}.json,
    running and writing it first if it does not already exist. Rounds is
    always set equal to distance, matching every existing baseline result
    and bench/run_predecoder.py's own eval convention.
    """
    path = out_dir / f"baseline_d{distance}_r{distance}_{basis}_p{p}.json"
    if path.exists():
        return path

    noise = NoiseModel.uniform_depolarizing(p)
    circuit = build_rotated_surface_code_circuit(distance, distance, basis, noise)
    dem = build_detector_error_model(circuit)
    matcher = build_matching_graph(dem)
    sampled = sample_detection_volume(circuit, distance, distance, shots, seed)

    result = decode_baseline(
        circuit=circuit,
        matcher=matcher,
        flat_detectors=sampled.flat_detectors,
        observables=sampled.observables,
        distance=distance,
        rounds=distance,
        basis=basis,
        p=p,
        seed=seed,
        timing_shots=timing_shots,
    )
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  wrote baseline {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-checkpoint", type=Path, default=REPO_ROOT / "models" / "predecoder_R5.pt")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--distances", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    parser.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.002, 0.003, 0.005])
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--timing-shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = REPO_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_model, fp32_ckpt = load_checkpoint(args.fp32_checkpoint)
    R = fp32_ckpt["R"]
    basis = fp32_ckpt["basis"]

    print(f"Ensuring baselines for {len(args.distances)}x{len(args.ps)} (distance, p) combinations...")
    for p in args.ps:
        for d in args.distances:
            ensure_baseline(d, p, basis, args.shots, args.timing_shots, args.seed, out_dir)

    sweep: dict = {}

    def run_precision(name: str, model, ckpt) -> None:
        print(f"{name}: R={ckpt['R']}")
        results = [
            _enrich_with_baseline(
                evaluate_at_distance(model, ckpt, d, p, args.shots, args.timing_shots, args.seed, args.threshold),
                out_dir, basis, p,
            )
            for p in args.ps
            for d in args.distances
        ]
        sweep[name] = {"checkpoint": None, "results": results}
        for r in results:
            print(
                f"  {name:8s} d={r['distance']:2d} p={r['p']:.3f} ler={r['ler']:.5f} "
                f"sdr={r['syndrome_density_reduction']:.4f} ler_reduction_factor={r['ler_reduction_factor']}"
            )

    run_precision("fp32", fp32_model, fp32_ckpt)

    for precision in PRECISIONS:
        qat_path = args.models_dir / f"predecoder_R{R}_{precision}.pt"
        if not qat_path.exists():
            print(f"skip {precision}: {qat_path} not found")
            continue
        model, ckpt = load_qat_checkpoint(qat_path)
        run_precision(precision, model, ckpt)
        sweep[precision]["checkpoint"] = str(qat_path)

    sweep["fp32"]["checkpoint"] = str(args.fp32_checkpoint)

    out_path = out_dir / "phase6_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
