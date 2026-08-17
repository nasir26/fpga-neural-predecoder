# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Quantization-aware fine-tuning wrapper around PreDecoder3D.

Same architecture (depth hidden Conv3d(k=3) + ReLU layers, then a 1x1x1
output conv) as src/model/predecoder.py, but each hidden layer's weights are
fake-quantized (straight-through estimator, see src/model/quantize.py) on
every forward pass, and post-ReLU activations are fake-quantized to unsigned
8-bit using a scale calibrated by a running max over training batches.
Biases are kept at full precision: they are a small per-channel additive
term, not a weight multiplied many times, so hardware cost of keeping them
wide is negligible (see docs/DECISIONS.md).

Starts QAT fine-tuning from an already-trained FP32 PreDecoder3D's weights
(via from_fp32), matching the plan's "quantization-aware fine-tuning for a
few epochs" rather than training a quantized model from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.predecoder import PreDecoder3D
from src.model.quantize import fake_quantize_activation, fake_quantize_weight


class QATPreDecoder3D(nn.Module):
    def __init__(
        self,
        depth: int,
        width: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
        precision: str = "int8",
        act_momentum: float = 0.9,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.precision = precision
        self.act_momentum = act_momentum

        self.hidden_convs = nn.ModuleList()
        c_in = in_channels
        for _ in range(depth):
            self.hidden_convs.append(nn.Conv3d(c_in, width, kernel_size=3, padding=1))
            c_in = width
        self.out_conv = nn.Conv3d(c_in, out_channels, kernel_size=1)

        self.register_buffer("act_running_max", torch.ones(depth))

    @property
    def R(self) -> int:
        return 2 * self.depth + 1

    @classmethod
    def from_fp32(cls, fp32_model: PreDecoder3D, precision: str) -> "QATPreDecoder3D":
        qat = cls(
            depth=fp32_model.depth,
            width=fp32_model.width,
            in_channels=fp32_model.in_channels,
            out_channels=fp32_model.out_channels,
            precision=precision,
        )
        fp32_convs = [m for m in fp32_model.net if isinstance(m, nn.Conv3d)]
        assert len(fp32_convs) == fp32_model.depth + 1
        for qat_conv, fp32_conv in zip(qat.hidden_convs, fp32_convs[:-1]):
            qat_conv.weight.data.copy_(fp32_conv.weight.data)
            qat_conv.bias.data.copy_(fp32_conv.bias.data)
        qat.out_conv.weight.data.copy_(fp32_convs[-1].weight.data)
        qat.out_conv.bias.data.copy_(fp32_convs[-1].bias.data)
        return qat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.hidden_convs):
            w_q = fake_quantize_weight(conv.weight, self.precision)
            x = F.conv3d(x, w_q, conv.bias, padding=1)
            x = F.relu(x)
            if self.training:
                with torch.no_grad():
                    batch_max = x.max().clamp(min=1e-6)
                    self.act_running_max[i].mul_(self.act_momentum).add_(
                        batch_max * (1 - self.act_momentum)
                    )
            scale = (self.act_running_max[i] / 255).clamp(min=1e-9).item()
            x = fake_quantize_activation(x, scale)

        w_out_q = fake_quantize_weight(self.out_conv.weight, self.precision)
        logits = F.conv3d(x, w_out_q, self.out_conv.bias, padding=0)
        return logits
