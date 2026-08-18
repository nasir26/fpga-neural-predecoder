# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Regression test for host/run_hw.py's honest-fallback requirement: when no
xclbin is available (the normal case right now; see docs/DECISIONS.md), it
must fall back to the golden model and label the result "simulated": True,
never silently. See docs/LIMITATIONS.md.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host.run_hw import run_golden_fallback, try_load_hardware
from src.model.qat import QATPreDecoder3D
from src.model.train import train


def test_try_load_hardware_returns_none_when_xclbin_missing(tmp_path):
    missing = tmp_path / "does_not_exist.xclbin"
    assert try_load_hardware(missing) is None


def test_golden_fallback_labels_result_as_simulated(tmp_path):
    R = 5
    train(
        R=R, width=4, epochs=1, shots_per_epoch=500, batch_size=250,
        chunk_shots=500, lr=1e-3, basis="z", train_p=0.02, seed=0,
        models_dir=tmp_path / "models", metrics_path=tmp_path / "metrics.json",
    )
    ckpt_path = tmp_path / "models" / f"predecoder_R{R}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # run_golden_fallback expects a QAT-style checkpoint (precision, edge_offsets);
    # the FP32 checkpoint from train() lacks "precision", so build a QAT wrapper
    # around it directly for this smoke test rather than a full fine-tune.
    from src.model.predecoder import PreDecoder3D

    fp32 = PreDecoder3D(depth=ckpt["depth"], width=ckpt["width"], out_channels=ckpt["out_channels"])
    fp32.load_state_dict(ckpt["model_state"])
    qat = QATPreDecoder3D.from_fp32(fp32, precision="int8")
    qat.eval()
    ckpt["precision"] = "int8"

    result = run_golden_fallback(qat, ckpt, distance=R, p=0.02, shots=200, seed=1)

    assert result["simulated"] is True
    assert "simulated_reason" in result and len(result["simulated_reason"]) > 0
    assert result["mean_latency_us_per_shot"]["host_to_device"] is None
    assert result["mean_latency_us_per_shot"]["kernel"] is None
    assert result["mean_latency_us_per_shot"]["device_to_host"] is None
    assert result["mean_latency_us_per_shot"]["predecoder_model"] > 0
    assert result["mean_latency_us_per_shot"]["pymatching_decode"] > 0
    assert 0.0 <= result["ler"] <= 1.0
