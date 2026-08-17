# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import torch

from src.model.train import train
from src.model.train_qat import fine_tune


def test_fine_tune_runs_end_to_end_and_produces_loadable_checkpoint(tmp_path):
    R = 5
    fp32_models_dir = tmp_path / "fp32_models"
    train(
        R=R, width=4, epochs=1, shots_per_epoch=500, batch_size=250,
        chunk_shots=500, lr=1e-3, basis="z", train_p=0.02, seed=0,
        models_dir=fp32_models_dir, metrics_path=tmp_path / "fp32_metrics.json",
    )
    fp32_checkpoint_path = fp32_models_dir / f"predecoder_R{R}.pt"

    qat_models_dir = tmp_path / "qat_models"
    result = fine_tune(
        fp32_checkpoint_path=fp32_checkpoint_path, precision="int8", epochs=1,
        shots_per_epoch=500, batch_size=250, chunk_shots=500, lr=1e-4, seed=1,
        models_dir=qat_models_dir, metrics_path=tmp_path / "qat_metrics.json",
    )

    assert len(result["history"]) == 1
    checkpoint_path = qat_models_dir / f"predecoder_R{R}_int8.pt"
    assert checkpoint_path.exists()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert ckpt["precision"] == "int8"
    assert ckpt["R"] == R
    assert ckpt["out_channels"] == result["model"].out_channels
    assert "act_running_max" in ckpt["model_state"]
