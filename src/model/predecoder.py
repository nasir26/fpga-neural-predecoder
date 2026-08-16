# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Fully convolutional 3D CNN pre-decoder.

Kernel 3, same padding, no pooling, for the D hidden layers; the final
output head is a 1x1x1 conv (channel projection only, adds no spatial
context). Each hidden layer extends the receptive field by 1 voxel on every
side (2 total), so depth D gives receptive field R = 2*D + 1 along every one
of the (T, H, W) axes (verified empirically via input-gradient support, see
tests/test_predecoder.py). Because there is no pooling and every layer is
stride 1, a model trained at one (T, H, W) grid size runs unchanged at any
larger grid size (the distance-generalisation property the method depends
on).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def receptive_field(depth: int) -> int:
    return 2 * depth + 1


def depth_for_receptive_field(r: int) -> int:
    if r < 1 or r % 2 == 0:
        raise ValueError(f"receptive field R must be a positive odd integer, got {r}")
    return (r - 1) // 2


class PreDecoder3D(nn.Module):
    """Predicts per-voxel correction logits from a (T, H, W) diff-frame volume.

    Input:  float tensor, shape (batch, in_channels, T, H, W)
    Output: float logits, shape (batch, out_channels, T, H, W)
            (raw logits, not probabilities; use with BCEWithLogitsLoss /
            torch.sigmoid at inference)
    """

    def __init__(
        self,
        depth: int,
        width: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = []
        c_in = in_channels
        for _ in range(depth):
            layers.append(nn.Conv3d(c_in, width, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = width
        layers.append(nn.Conv3d(c_in, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    @property
    def R(self) -> int:
        return receptive_field(self.depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
