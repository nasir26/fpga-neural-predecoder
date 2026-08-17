# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Quantization-aware fine-tuning: starts from a trained FP32 PreDecoder3D
checkpoint and fine-tunes for a few epochs at a given precision
(int8/int4/ternary), reusing the same streaming data pipeline as
src/model/train.py (src.model.train.run_training_epochs) and the same
(distance=rounds=R, basis, train_p) circuit the FP32 model was trained on.

Deliberate deviation from a from-scratch quantized training run: this
project fine-tunes a few epochs from the FP32 checkpoint rather than
training each precision from random init, per the plan's "quantization-aware
fine-tuning for a few epochs at each precision." See docs/LIMITATIONS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.predecoder import PreDecoder3D
from src.model.qat import QATPreDecoder3D
from src.model.quantize import PRECISIONS
from src.model.train import build_training_circuit, run_training_epochs


def load_fp32_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def fine_tune(
    fp32_checkpoint_path: Path,
    precision: str,
    epochs: int,
    shots_per_epoch: int,
    batch_size: int,
    chunk_shots: int,
    lr: float,
    seed: int,
    models_dir: Path,
    metrics_path: Path,
) -> dict:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    ckpt = load_fp32_checkpoint(fp32_checkpoint_path)
    fp32_model = PreDecoder3D(depth=ckpt["depth"], width=ckpt["width"], out_channels=ckpt["out_channels"])
    fp32_model.load_state_dict(ckpt["model_state"])

    R, basis, train_p = ckpt["R"], ckpt["basis"], ckpt["train_p"]
    circuit, edge_offsets, num_channels, incidence = build_training_circuit(R, basis, train_p)
    assert edge_offsets == ckpt["edge_offsets"], (
        "edge-offset catalog recomputed from the FP32 checkpoint's (R, basis, train_p) "
        "does not match the catalog stored in the checkpoint; the DEM structure changed "
        "since training, which should not happen for a fixed circuit"
    )
    assert num_channels == ckpt["out_channels"]

    qat_model = QATPreDecoder3D.from_fp32(fp32_model, precision)
    qat_model.train()
    opt = torch.optim.Adam(qat_model.parameters(), lr=lr)

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"predecoder_R{R}_{precision}.pt"
    history: list = []

    def on_epoch_end(epoch, epoch_loss, shots_done, epoch_time):
        history.append({"epoch": epoch, "loss": epoch_loss, "shots": shots_done, "epoch_time_s": epoch_time})
        torch.save(
            {
                "model_state": qat_model.state_dict(),
                "depth": qat_model.depth,
                "width": qat_model.width,
                "out_channels": num_channels,
                "edge_offsets": edge_offsets,
                "R": R,
                "basis": basis,
                "train_p": train_p,
                "precision": precision,
                "fp32_checkpoint": str(fp32_checkpoint_path),
                "epoch": epoch,
            },
            checkpoint_path,
        )
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "R": R, "precision": precision, "basis": basis, "train_p": train_p,
                    "num_channels": num_channels, "edge_offsets": edge_offsets,
                    "shots_per_epoch": shots_per_epoch, "batch_size": batch_size,
                    "lr": lr, "seed": seed, "fp32_checkpoint": str(fp32_checkpoint_path),
                    "history": history,
                },
                f, indent=2,
            )

    run_training_epochs(
        qat_model, opt, circuit, incidence, num_channels, R, epochs, shots_per_epoch,
        batch_size, chunk_shots, seed, on_epoch_end=on_epoch_end,
    )

    return {"model": qat_model, "history": history, "checkpoint_path": str(checkpoint_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-checkpoint", type=Path, default=REPO_ROOT / "models" / "predecoder_R5.pt")
    parser.add_argument("--precision", type=str, required=True, choices=list(PRECISIONS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--shots-per-epoch", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-shots", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert not torch.cuda.is_available(), "CPU-only training required"

    models_dir = REPO_ROOT / "models"
    metrics_path = REPO_ROOT / "results" / "metrics" / f"train_qat_{args.precision}.json"

    print(f"QAT fine-tuning: precision={args.precision} from {args.fp32_checkpoint} "
          f"epochs={args.epochs} shots_per_epoch={args.shots_per_epoch} batch_size={args.batch_size}")

    result = fine_tune(
        fp32_checkpoint_path=args.fp32_checkpoint, precision=args.precision, epochs=args.epochs,
        shots_per_epoch=args.shots_per_epoch, batch_size=args.batch_size, chunk_shots=args.chunk_shots,
        lr=args.lr, seed=args.seed, models_dir=models_dir, metrics_path=metrics_path,
    )
    print(f"done; checkpoint at {result['checkpoint_path']}, metrics at {metrics_path}")


if __name__ == "__main__":
    main()
