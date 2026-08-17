# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Symmetric per-output-channel weight quantization (INT8/INT4/ternary) and
straight-through-estimator fake-quantization for QAT fine-tuning.

Fixed-point datapath convention (must match src/model/golden.py exactly,
see docs/DECISIONS.md):
  - Weights: symmetric per-output-channel, INT8 in [-127,127], INT4 in
    [-7,7], or ternary in {-1,0,+1} (threshold 0.7*mean(|w|), the standard
    Ternary Weight Networks rule).
  - Activations: post-ReLU hidden activations are unsigned 8-bit
    (non-negative after ReLU), scale calibrated by a running max during QAT.
    The input layer is exact 0/1 (no quantization loss). The final 1x1x1
    layer's output is a raw accumulator value; since the eventual decision
    is sigmoid(logit) > 0.5, which is equivalent to logit > 0, the golden
    and HLS datapaths compare the integer accumulator to 0 directly and
    never compute sigmoid.
  - Rounding: round-half-up (floor(x + 0.5)), matching Vitis HLS's AP_RND.
  - Saturation: clip to the representable range, matching Vitis HLS's
    AP_SAT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

PRECISIONS = ("int8", "int4", "ternary")

_WEIGHT_QMAX = {"int8": 127, "int4": 7}
TERNARY_THRESHOLD_FACTOR = 0.7


def round_half_up(x: np.ndarray) -> np.ndarray:
    """floor(x + 0.5), matching Vitis HLS ap_fixed's AP_RND for positive and
    negative values alike (ties round toward +infinity).
    """
    return np.floor(x + 0.5)


@dataclass
class QuantizedWeight:
    values: np.ndarray  # int, shape (out_channels, ...), the quantized integers
    scales: np.ndarray  # float32, shape (out_channels,)
    precision: str


def quantize_weight_per_channel(weight: np.ndarray, precision: str) -> QuantizedWeight:
    """weight: (out_channels, ...) float array. Quantizes along axis 0."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    out_channels = weight.shape[0]
    flat = weight.reshape(out_channels, -1)

    if precision == "ternary":
        abs_flat = np.abs(flat)
        threshold = TERNARY_THRESHOLD_FACTOR * abs_flat.mean(axis=1, keepdims=True)
        mask = abs_flat > threshold
        signs = np.sign(flat)
        with np.errstate(invalid="ignore", divide="ignore"):
            count = mask.sum(axis=1, keepdims=True)
            scale = np.where(
                count > 0,
                (abs_flat * mask).sum(axis=1, keepdims=True) / np.maximum(count, 1),
                1.0,
            )
        values = np.where(mask, signs, 0).astype(np.int8)
        scales = scale.reshape(out_channels).astype(np.float32)
    else:
        qmax = _WEIGHT_QMAX[precision]
        max_abs = np.abs(flat).max(axis=1, keepdims=True)
        max_abs = np.maximum(max_abs, 1e-12)  # avoid div by zero for an all-zero channel
        scale = max_abs / qmax
        values = round_half_up(flat / scale)
        values = np.clip(values, -qmax, qmax).astype(np.int8)
        scales = scale.reshape(out_channels).astype(np.float32)

    return QuantizedWeight(
        values=values.reshape(weight.shape), scales=scales, precision=precision
    )


def dequantize_weight(qw: QuantizedWeight) -> np.ndarray:
    out_channels = qw.values.shape[0]
    flat = qw.values.reshape(out_channels, -1).astype(np.float32)
    deq = flat * qw.scales.reshape(out_channels, 1)
    return deq.reshape(qw.values.shape)


class _FakeQuantWeightSTE(torch.autograd.Function):
    """Forward: quantize-then-dequantize (simulates the quantized value).
    Backward: straight-through (gradient passes unchanged), the standard QAT
    trick since the true quantization gradient is zero almost everywhere.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, precision: str) -> torch.Tensor:
        w_np = weight.detach().cpu().numpy()
        qw = quantize_weight_per_channel(w_np, precision)
        deq = dequantize_weight(qw)
        return torch.from_numpy(deq).to(weight.device, weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize_weight(weight: torch.Tensor, precision: str) -> torch.Tensor:
    return _FakeQuantWeightSTE.apply(weight, precision)


class _FakeQuantActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        q = round_half_up(x_np / scale)
        q = np.clip(q, 0, 255)
        deq = (q * scale).astype(np.float32)
        return torch.from_numpy(deq).to(x.device, x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize_activation(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Unsigned 8-bit fake-quant for post-ReLU activations (x must be >= 0)."""
    return _FakeQuantActivationSTE.apply(x, scale)


def quantize_bias(bias: np.ndarray, weight_scale: np.ndarray, input_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """A real integer accumulator can only add an integer bias, at the
    accumulator's implicit scale (weight_scale_c * input_scale) for output
    channel c; it cannot add an arbitrary-precision float. Returns
    (bias_int, bias_dequantized), both shape (out_channels,).
    """
    accumulator_scale = weight_scale * input_scale
    bias_int = round_half_up(bias / accumulator_scale).astype(np.int64)
    bias_dequantized = (bias_int.astype(np.float64) * accumulator_scale).astype(np.float32)
    return bias_int, bias_dequantized


class _FakeQuantBiasSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias: torch.Tensor, weight_scale: np.ndarray, input_scale: float) -> torch.Tensor:
        bias_np = bias.detach().cpu().numpy()
        _, deq = quantize_bias(bias_np, weight_scale, input_scale)
        return torch.from_numpy(deq).to(bias.device, bias.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def fake_quantize_bias(bias: torch.Tensor, weight_scale: np.ndarray, input_scale: float) -> torch.Tensor:
    return _FakeQuantBiasSTE.apply(bias, weight_scale, input_scale)
