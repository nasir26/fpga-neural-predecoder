# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a trained PreDecoder3D checkpoint at one or more code distances.

Fully convolutional: a model trained at distance=rounds=R runs unchanged at
any larger distance/rounds. This script samples fresh shots at the requested
eval distance (with eval_p, not the training p), runs the model to get a
correction mask, XORs it into the raw detection volume to get a residual
syndrome, decodes the residual with PyMatching, and compares LER/SDR/speedup
against the Phase 1 baseline at the same distance.

Usage:
    python bench/run_predecoder.py --checkpoint models/predecoder_R5.pt
        --distances 5 7 9 --p 0.003 --shots 100000 --timing-shots 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qec import metrics as qec_metrics
from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit, detector_coords_to_grid_index
from src.qec.dem import build_detector_error_model, build_matching_graph
from src.qec.sampling import diff_frames, sample_detection_volume
from src.model.predecoder import PreDecoder3D


def load_checkpoint(path: Path) -> tuple:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PreDecoder3D(depth=ckpt["depth"], width=ckpt["width"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_at_distance(
    model: torch.nn.Module,
    ckpt: dict,
    distance: int,
    p: float,
    shots: int,
    timing_shots: int,
    seed: int,
    threshold: float,
) -> dict:
    rounds = distance
    basis = ckpt["basis"]

    noise = NoiseModel.uniform_depolarizing(p)
    circuit = build_rotated_surface_code_circuit(distance, rounds, basis, noise)
    dem = build_detector_error_model(circuit)
    matcher = build_matching_graph(dem)

    sampled = sample_detection_volume(circuit, distance, rounds, shots, seed)
    diff = diff_frames(sampled.volume)

    with torch.no_grad():
        x = torch.from_numpy(diff).float().unsqueeze(1)
        logits = model(x).squeeze(1)
        correction = (torch.sigmoid(logits) > threshold).numpy().astype(np.uint8)

    residual_volume = sampled.volume ^ correction

    grid_index = detector_coords_to_grid_index(circuit)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]
    residual_flat = residual_volume[:, t_idx, row_idx, col_idx].astype(bool)

    predicted = matcher.decode_batch(residual_flat).astype(bool)
    ler, ci_low, ci_high = qec_metrics.logical_error_rate(predicted, sampled.observables)

    density_before = float(sampled.flat_detectors.mean())
    density_after = float(residual_flat.mean())
    sdr = qec_metrics.syndrome_density_reduction(density_before, density_after)

    # Single-shot timing: model forward (per shot, batch=1) + matching.decode()
    matcher.decode(residual_flat[0])  # warm up
    elapsed = np.empty(timing_shots, dtype=np.float64)
    with torch.no_grad():
        for i in range(timing_shots):
            t0 = time.perf_counter()
            xi = torch.from_numpy(diff[i : i + 1]).float().unsqueeze(1)
            logit_i = model(xi).squeeze(1)
            corr_i = (torch.sigmoid(logit_i) > threshold).numpy().astype(np.uint8)
            resid_i = sampled.volume[i : i + 1] ^ corr_i
            resid_flat_i = resid_i[:, t_idx, row_idx, col_idx].astype(bool)[0]
            matcher.decode(resid_flat_i)
            t1 = time.perf_counter()
            elapsed[i] = t1 - t0
    mean_latency_us_per_shot = float(np.mean(elapsed) * 1e6)

    return {
        "distance": distance,
        "rounds": rounds,
        "basis": basis,
        "p": p,
        "R": ckpt["R"],
        "train_p": ckpt["train_p"],
        "shots": shots,
        "seed": seed,
        "threshold": threshold,
        "ler": ler,
        "ler_ci_low": ci_low,
        "ler_ci_high": ci_high,
        "syndrome_density_before": density_before,
        "syndrome_density_after": density_after,
        "syndrome_density_reduction": sdr,
        "timing_shots": timing_shots,
        "mean_latency_us_per_shot": mean_latency_us_per_shot,
        "mean_latency_us_per_round": mean_latency_us_per_shot / rounds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", required=True)
    parser.add_argument("--p", type=float, default=0.003)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--timing-shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, ckpt = load_checkpoint(args.checkpoint)

    out_dir = REPO_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in args.distances:
        result = evaluate_at_distance(
            model, ckpt, d, args.p, args.shots, args.timing_shots, args.seed, args.threshold
        )
        # LER reduction factor against the Phase 1 baseline at the same (d, basis, p), if present.
        baseline_path = out_dir / f"baseline_d{d}_r{d}_{ckpt['basis']}_p{args.p}.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            result["baseline_ler"] = baseline["ler"]
            result["ler_reduction_factor"] = baseline["ler"] / result["ler"] if result["ler"] > 0 else float("inf")
            result["speedup_vs_baseline"] = qec_metrics.speedup(
                baseline["mean_latency_us_per_round"], result["mean_latency_us_per_round"]
            )
        else:
            result["baseline_ler"] = None
            result["ler_reduction_factor"] = None
            result["speedup_vs_baseline"] = None

        out_path = out_dir / f"predecoder_d{d}_R{ckpt['R']}_p{args.p}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_path}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
