# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact NumPy replica of the intended HLS integer datapath.

This is the canonical definition of "what the FPGA kernel computes", used to
verify src/model/qat.py's quantization-aware forward pass (zero tolerance,
see tests/test_golden_vs_torch.py) now, and to verify the actual HLS C++
kernel against the same reference in Phase 4.

Per-layer datapath (matches src/model/qat.py's QATPreDecoder3D.forward
exactly; the two must be changed together):
  1. 3D convolution (kernel 3, same padding for hidden layers; kernel 1, no
     padding for the output layer), integer weights x integer activations,
     accumulated in a 64-bit NumPy integer (documented as the safe upper
     bound; a real accumulator only needs enough bits for the worst-case sum,
     sized in Phase 4 from the exported weight/activation ranges).
  2. Integer bias add: bias_int = round_half_up(bias_float / (weight_scale_c
     * input_scale)), added directly in the accumulator's integer domain (a
     real integer accumulator cannot add an arbitrary-precision float bias).
  3. ReLU in the integer domain (hidden layers only): max(acc, 0).
  4. Requantize to the next layer's unsigned 8-bit activation:
     true_value = acc * weight_scale_c * input_scale (float64, exact: acc is
     a genuine integer well within float64's exact range);
     next_int = clip(round_half_up(true_value / next_activation_scale), 0, 255).
  5. Final (output) layer: no ReLU, no requantization. Since the eventual
     decision is sigmoid(logit) > 0.5, equivalent to logit > 0, the golden
     datapath compares the final integer accumulator (per channel, per
     voxel) to 0 directly and never computes sigmoid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.model.quantize import quantize_bias, quantize_weight_per_channel, round_half_up


@dataclass
class GoldenLayer:
    weight_int: np.ndarray  # int64, (out_c, in_c, k, k, k)
    weight_scale: np.ndarray  # float64, (out_c,)
    bias_int: np.ndarray  # int64, (out_c,)
    input_scale: float  # scale of this layer's INPUT activation (1.0 for the first layer)
    kernel_size: int
    padding: int
    apply_relu: bool
    output_activation_scale: Optional[float]  # None for the final layer (no requant)


def conv3d_int(x_int: np.ndarray, weight_int: np.ndarray, kernel_size: int, padding: int) -> np.ndarray:
    """x_int: int64, (in_c, T, H, W). weight_int: int64, (out_c, in_c, k, k, k).
    Returns int64, (out_c, T, H, W). Exact integer arithmetic throughout
    (int64 has no representable-integer precision limit at these magnitudes,
    unlike float32).
    """
    in_c, T, H, W = x_int.shape
    out_c = weight_int.shape[0]
    k = kernel_size

    if padding > 0:
        x_padded = np.zeros((in_c, T + 2 * padding, H + 2 * padding, W + 2 * padding), dtype=np.int64)
        x_padded[:, padding : padding + T, padding : padding + H, padding : padding + W] = x_int
    else:
        x_padded = x_int

    acc = np.zeros((out_c, T, H, W), dtype=np.int64)
    for dz in range(k):
        for dy in range(k):
            for dx in range(k):
                window = x_padded[:, dz : dz + T, dy : dy + H, dx : dx + W]  # (in_c, T, H, W)
                w_slice = weight_int[:, :, dz, dy, dx]  # (out_c, in_c)
                acc += np.tensordot(w_slice, window, axes=([1], [0]))  # (out_c, T, H, W)
    return acc


def build_golden_layers_from_quantized(
    conv_layers: List[dict], precision: str
) -> List[GoldenLayer]:
    """conv_layers: list of dicts, one per Conv3d layer in order, each with
    keys 'weight' (np.ndarray float), 'bias' (np.ndarray float),
    'kernel_size', 'padding', 'apply_relu', 'output_activation_scale'
    (float or None), 'input_scale' (float, this layer's input scale).
    """
    layers = []
    for spec in conv_layers:
        qw = quantize_weight_per_channel(spec["weight"], precision)
        weight_int = qw.values.astype(np.int64)
        weight_scale = qw.scales.astype(np.float64)
        bias_int, _ = quantize_bias(spec["bias"], qw.scales, spec["input_scale"])
        layers.append(
            GoldenLayer(
                weight_int=weight_int,
                weight_scale=weight_scale,
                bias_int=bias_int.astype(np.int64),
                input_scale=spec["input_scale"],
                kernel_size=spec["kernel_size"],
                padding=spec["padding"],
                apply_relu=spec["apply_relu"],
                output_activation_scale=spec["output_activation_scale"],
            )
        )
    return layers


def extract_golden_layers(qat_model) -> List[GoldenLayer]:
    """Build the GoldenLayer list from a (typically eval-mode) QATPreDecoder3D
    instance, using its current weights and its calibrated
    act_running_max buffer for each hidden layer's output activation scale.
    """
    conv_layers = []
    input_scale = 1.0
    for i, conv in enumerate(qat_model.hidden_convs):
        output_scale = float(qat_model.act_running_max[i].item() / 255)
        conv_layers.append(
            {
                "weight": conv.weight.detach().cpu().numpy(),
                "bias": conv.bias.detach().cpu().numpy(),
                "kernel_size": 3,
                "padding": 1,
                "apply_relu": True,
                "output_activation_scale": output_scale,
                "input_scale": input_scale,
            }
        )
        input_scale = output_scale
    conv_layers.append(
        {
            "weight": qat_model.out_conv.weight.detach().cpu().numpy(),
            "bias": qat_model.out_conv.bias.detach().cpu().numpy(),
            "kernel_size": 1,
            "padding": 0,
            "apply_relu": False,
            "output_activation_scale": None,
            "input_scale": input_scale,
        }
    )
    return build_golden_layers_from_quantized(conv_layers, qat_model.precision)


def golden_forward(x_binary: np.ndarray, layers: List[GoldenLayer]) -> np.ndarray:
    """x_binary: uint8/int, (T, H, W), values in {0, 1} (single sample, one
    input channel). Returns the final layer's raw int64 accumulator, shape
    (out_channels, T, H, W); the correction decision is (result > 0).
    """
    x_int = x_binary.astype(np.int64)[None, :, :, :]  # (in_c=1, T, H, W)

    for layer_idx, layer in enumerate(layers):
        acc = conv3d_int(x_int, layer.weight_int, layer.kernel_size, layer.padding)
        acc = acc + layer.bias_int[:, None, None, None]

        if layer.apply_relu:
            acc = np.maximum(acc, 0)

        if layer.output_activation_scale is None:
            return acc  # final layer: raw accumulator, no requant

        true_value = acc.astype(np.float64) * (layer.weight_scale[:, None, None, None] * layer.input_scale)
        next_int = round_half_up(true_value / layer.output_activation_scale)
        next_int = np.clip(next_int, 0, 255).astype(np.int64)
        x_int = next_int

    raise ValueError("no final (output_activation_scale=None) layer found")
