# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the predecoder evaluation pipeline: a tiny,
barely trained checkpoint must run through evaluate_at_distance without
error and produce internally consistent (not necessarily good) metrics.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.run_predecoder import evaluate_at_distance
from src.model.predecoder import PreDecoder3D, depth_for_receptive_field
from src.model.train import train


def test_evaluate_at_distance_runs_end_to_end(tmp_path):
    R = 5
    train_result = train(
        R=R, width=4, epochs=1, shots_per_epoch=500, batch_size=250,
        chunk_shots=500, lr=1e-3, basis="z", train_p=0.02, seed=0,
        models_dir=tmp_path / "models", metrics_path=tmp_path / "metrics" / "train.json",
    )
    model = train_result["model"]
    model.eval()
    ckpt = {"R": R, "basis": "z", "train_p": 0.02, "depth": depth_for_receptive_field(R), "width": 4}

    result = evaluate_at_distance(
        model, ckpt, distance=5, p=0.01, shots=300, timing_shots=10, seed=1, threshold=0.5
    )

    assert result["distance"] == 5
    assert 0.0 <= result["ler"] <= 1.0
    assert result["ler_ci_low"] <= result["ler"] <= result["ler_ci_high"]
    assert 0.0 <= result["syndrome_density_before"] <= 1.0
    assert 0.0 <= result["syndrome_density_after"] <= 1.0
    assert result["mean_latency_us_per_shot"] > 0


def test_evaluate_generalizes_to_larger_distance_than_training():
    # R=5 model, fully convolutional: must also run (not necessarily well)
    # at distance=7 without shape errors.
    model = PreDecoder3D(depth=depth_for_receptive_field(5), width=4)
    model.eval()
    ckpt = {"R": 5, "basis": "z", "train_p": 0.02, "depth": depth_for_receptive_field(5), "width": 4}
    result = evaluate_at_distance(
        model, ckpt, distance=7, p=0.01, shots=200, timing_shots=5, seed=2, threshold=0.5
    )
    assert result["distance"] == 7
    assert 0.0 <= result["ler"] <= 1.0
