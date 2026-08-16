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

from bench.run_predecoder import evaluate_at_distance, load_checkpoint
from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit
from src.qec.dem import build_local_edge_catalog
from src.model.predecoder import PreDecoder3D, depth_for_receptive_field
from src.model.train import train


def test_evaluate_at_distance_runs_end_to_end(tmp_path):
    R = 5
    checkpoint_path = tmp_path / "models" / f"predecoder_R{R}.pt"
    train(
        R=R, width=4, epochs=1, shots_per_epoch=500, batch_size=250,
        chunk_shots=500, lr=1e-3, basis="z", train_p=0.02, seed=0,
        models_dir=checkpoint_path.parent, metrics_path=tmp_path / "metrics" / "train.json",
    )
    model, ckpt = load_checkpoint(checkpoint_path)

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
    # at distance=7 without shape errors. Build a synthetic checkpoint with
    # a real edge-offset catalog computed from an R=5 circuit.
    R = 5
    noise = NoiseModel.uniform_depolarizing(0.02)
    circuit = build_rotated_surface_code_circuit(R, R, "z", noise)
    edge_offsets = build_local_edge_catalog(circuit)
    num_channels = 1 + len(edge_offsets)

    model = PreDecoder3D(depth=depth_for_receptive_field(R), width=4, out_channels=num_channels)
    model.eval()
    ckpt = {
        "R": R, "basis": "z", "train_p": 0.02,
        "depth": depth_for_receptive_field(R), "width": 4,
        "out_channels": num_channels, "edge_offsets": edge_offsets,
    }
    result = evaluate_at_distance(
        model, ckpt, distance=7, p=0.01, shots=200, timing_shots=5, seed=2, threshold=0.5
    )
    assert result["distance"] == 7
    assert 0.0 <= result["ler"] <= 1.0
