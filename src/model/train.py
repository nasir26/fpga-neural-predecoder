# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""CPU training loop for PreDecoder3D.

Trains at distance = rounds = R (the model's receptive field), streaming
fresh iid shots in chunks (never reusing a shot) rather than holding an
entire epoch's shots in memory at once or reusing a fixed dataset across
epochs; QEC training data is free to resample, so there is no train/val
split and no overfitting risk from data reuse.

Deliberate deviation from the reference recipe (67M shots/epoch, 100
epochs, 8 GPUs): this is a CPU-only, single-machine, reduced-budget training
run. See docs/LIMITATIONS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit
from src.qec.dem import build_error_channel_incidence, build_local_edge_catalog
from src.qec.sampling import sample_training_shots
from src.model.predecoder import PreDecoder3D, depth_for_receptive_field


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "conf" / "default.yaml")
    parser.add_argument("--R", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--shots-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-shots", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--basis", type=str, default=None, choices=["x", "z"])
    parser.add_argument("--no-noise-upscaling", action="store_true")
    parser.add_argument("--target-p", type=float, default=None)
    parser.add_argument("--eval-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def train(
    R: int,
    width: int,
    epochs: int,
    shots_per_epoch: int,
    batch_size: int,
    chunk_shots: int,
    lr: float,
    basis: str,
    train_p: float,
    seed: int,
    models_dir: Path,
    metrics_path: Path,
) -> dict:
    depth = depth_for_receptive_field(R)
    distance = rounds = R

    noise = NoiseModel.uniform_depolarizing(train_p)
    circuit = build_rotated_surface_code_circuit(distance, rounds, basis, noise)
    edge_offsets = build_local_edge_catalog(circuit)
    num_channels = 1 + len(edge_offsets)
    incidence = build_error_channel_incidence(circuit, edge_offsets)

    model = PreDecoder3D(depth=depth, width=width, out_channels=num_channels)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"predecoder_R{R}.pt"

    history = []
    n_chunks = -(-shots_per_epoch // chunk_shots)  # ceil div
    seed_counter = seed * 1_000_000

    for epoch in range(epochs):
        epoch_t0 = time.time()
        epoch_losses = []
        shots_done = 0
        for chunk_idx in range(n_chunks):
            this_chunk = min(chunk_shots, shots_per_epoch - shots_done)
            if this_chunk <= 0:
                break
            seed_counter += 1
            shots = sample_training_shots(
                circuit, incidence, num_channels, distance, rounds, this_chunk, seed=seed_counter
            )
            x_all = torch.from_numpy(shots.diff).float().unsqueeze(1)
            y_all = torch.from_numpy(shots.target).float()

            for start in range(0, this_chunk, batch_size):
                end = min(start + batch_size, this_chunk)
                x = x_all[start:end]
                y = y_all[start:end]
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())

            shots_done += this_chunk

        epoch_loss = float(np.mean(epoch_losses))
        epoch_time = time.time() - epoch_t0
        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "shots": shots_done,
            "epoch_time_s": epoch_time,
        }
        history.append(record)
        print(f"epoch {epoch}: loss={epoch_loss:.5f} shots={shots_done} time={epoch_time:.1f}s", flush=True)

        torch.save(
            {
                "model_state": model.state_dict(),
                "depth": depth,
                "width": width,
                "out_channels": num_channels,
                "edge_offsets": edge_offsets,
                "R": R,
                "basis": basis,
                "train_p": train_p,
                "epoch": epoch,
            },
            checkpoint_path,
        )
        with open(metrics_path, "w") as f:
            json.dump({"R": R, "width": width, "basis": basis, "train_p": train_p,
                       "num_channels": num_channels, "edge_offsets": edge_offsets,
                       "shots_per_epoch": shots_per_epoch, "batch_size": batch_size,
                       "lr": lr, "seed": seed, "history": history}, f, indent=2)

    return {"model": model, "history": history, "checkpoint_path": str(checkpoint_path)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    R = args.R if args.R is not None else cfg["model"]["R"]
    width = args.width if args.width is not None else cfg["model"]["width"]
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    shots_per_epoch = args.shots_per_epoch if args.shots_per_epoch is not None else cfg["training"]["shots_per_epoch"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]
    lr = args.lr if args.lr is not None else cfg["training"]["lr"]
    basis = args.basis if args.basis is not None else cfg["training"]["basis"]
    seed = args.seed if args.seed is not None else cfg["seed"]

    noise_upscaling = cfg["training"]["noise_upscaling"]["enabled"] and not args.no_noise_upscaling
    target_p = args.target_p if args.target_p is not None else cfg["training"]["noise_upscaling"]["target_p"]
    eval_p = args.eval_p if args.eval_p is not None else cfg["training"]["eval_p"]
    train_p = target_p if noise_upscaling else eval_p

    assert not torch.cuda.is_available(), "CPU-only training required"

    models_dir = REPO_ROOT / "models"
    metrics_path = REPO_ROOT / "results" / "metrics" / f"train_R{R}.json"

    print(f"Training PreDecoder3D: R={R} depth={depth_for_receptive_field(R)} width={width} "
          f"basis={basis} train_p={train_p} (noise_upscaling={noise_upscaling}, eval_p={eval_p}) "
          f"epochs={epochs} shots_per_epoch={shots_per_epoch} batch_size={batch_size}")

    train(
        R=R, width=width, epochs=epochs, shots_per_epoch=shots_per_epoch,
        batch_size=batch_size, chunk_shots=args.chunk_shots, lr=lr, basis=basis,
        train_p=train_p, seed=seed, models_dir=models_dir, metrics_path=metrics_path,
    )
    print(f"done; checkpoint at {models_dir / f'predecoder_R{R}.pt'}, metrics at {metrics_path}")


if __name__ == "__main__":
    main()
