# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Contribution C1: quantization sweep.

Evaluates the FP32 checkpoint and each fine-tuned quantized checkpoint
(int8, int4, ternary) at the same distance(s)/p/shots, recording LER (with
Wilson CI), syndrome density reduction, and LER reduction factor / speedup
against the Phase 1 PyMatching-only baseline for every precision, then
writes one combined results/metrics/quantization_sweep.json. Also exports
each quantized precision's HLS header + binary weight blob.

Usage:
    python bench/sweep_precision.py --fp32-checkpoint models/predecoder_R5.pt
        --distances 5 --p 0.003 --shots 100000 --timing-shots 1000
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
from src.model.export import export_hls_header
from src.model.golden import extract_golden_layers
from src.model.qat import QATPreDecoder3D
from src.model.quantize import PRECISIONS
from src.qec import metrics as qec_metrics


def _enrich_with_baseline(result: dict, out_dir: Path, basis: str, p: float) -> dict:
    """Adds baseline_ler / ler_reduction_factor / speedup_vs_baseline from
    the matching Phase 1 results/metrics/baseline_*.json, if present.
    Mirrors bench/run_predecoder.py's main(); factored out here since
    sweep_precision.py calls evaluate_at_distance directly, not through
    run_predecoder.py's main().
    """
    d = result["distance"]
    baseline_path = out_dir / f"baseline_d{d}_r{d}_{basis}_p{p}.json"
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
    return result


def load_qat_checkpoint(path: Path) -> tuple:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = QATPreDecoder3D(
        depth=ckpt["depth"], width=ckpt["width"], out_channels=ckpt["out_channels"], precision=ckpt["precision"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-checkpoint", type=Path, default=REPO_ROOT / "models" / "predecoder_R5.pt")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--distances", type=int, nargs="+", required=True)
    parser.add_argument("--p", type=float, default=0.003)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--timing-shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hls-dir", type=Path, default=REPO_ROOT / "hls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = REPO_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep: dict = {}

    fp32_model, fp32_ckpt = load_checkpoint(args.fp32_checkpoint)
    R = fp32_ckpt["R"]
    print(f"fp32: R={R} from {args.fp32_checkpoint}")
    sweep["fp32"] = {
        "checkpoint": str(args.fp32_checkpoint),
        "sha256": None,
        "results": [
            _enrich_with_baseline(
                evaluate_at_distance(fp32_model, fp32_ckpt, d, args.p, args.shots, args.timing_shots, args.seed, args.threshold),
                out_dir, fp32_ckpt["basis"], args.p,
            )
            for d in args.distances
        ],
    }

    for precision in PRECISIONS:
        qat_path = args.models_dir / f"predecoder_R{R}_{precision}.pt"
        if not qat_path.exists():
            print(f"skip {precision}: {qat_path} not found")
            continue
        model, ckpt = load_qat_checkpoint(qat_path)
        print(f"{precision}: R={R} from {qat_path}")

        layers = extract_golden_layers(model)
        export_meta = export_hls_header(layers, precision=precision, R=R, output_dir=args.hls_dir)

        results = [
            _enrich_with_baseline(
                evaluate_at_distance(model, ckpt, d, args.p, args.shots, args.timing_shots, args.seed, args.threshold),
                out_dir, ckpt["basis"], args.p,
            )
            for d in args.distances
        ]
        sweep[precision] = {
            "checkpoint": str(qat_path),
            "sha256": export_meta["sha256"],
            "weight_bytes": export_meta["weight_bytes"],
            "header_path": export_meta["header_path"],
            "bin_path": export_meta["bin_path"],
            "results": results,
        }

    out_path = out_dir / "quantization_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"wrote {out_path}")

    for precision, entry in sweep.items():
        for r in entry["results"]:
            sha = entry.get("sha256")
            sha_str = f" sha256={sha[:12]}..." if sha else ""
            print(
                f"{precision:8s} d={r['distance']:2d} ler={r['ler']:.5f} "
                f"sdr={r['syndrome_density_reduction']:.4f} "
                f"ler_reduction_factor={r['ler_reduction_factor']}{sha_str}"
            )


if __name__ == "__main__":
    main()
