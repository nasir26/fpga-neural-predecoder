# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from src.model.quantize import (
    dequantize_weight,
    fake_quantize_activation,
    fake_quantize_weight,
    quantize_weight_per_channel,
    round_half_up,
)


def test_round_half_up_ties():
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.5])
    # floor(x+0.5): -1.5->-1, -0.5->0, 0.0->0, 0.5->1, 1.5->2, 2.5->3
    expected = np.array([-1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(round_half_up(x), expected)


def test_int8_quantization_known_values():
    # Single output channel, max abs = 1.0 -> scale = 1/127.
    w = np.array([[1.0, -1.0, 0.5, 0.0]])
    qw = quantize_weight_per_channel(w, "int8")
    assert qw.values.dtype == np.int8
    assert qw.values[0, 0] == 127
    assert qw.values[0, 1] == -127
    assert qw.values[0, 3] == 0
    np.testing.assert_allclose(qw.scales, [1.0 / 127], rtol=1e-6)


def test_int4_quantization_range():
    w = np.random.RandomState(0).randn(3, 5, 3, 3, 3).astype(np.float32)
    qw = quantize_weight_per_channel(w, "int4")
    assert qw.values.min() >= -7
    assert qw.values.max() <= 7


def test_ternary_quantization_values_in_range():
    w = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    qw = quantize_weight_per_channel(w, "ternary")
    assert set(np.unique(qw.values.astype(int))) <= {-1, 0, 1}


def test_ternary_all_zero_channel_does_not_divide_by_zero():
    w = np.zeros((2, 4), dtype=np.float32)
    qw = quantize_weight_per_channel(w, "ternary")
    assert not np.any(np.isnan(qw.scales))
    np.testing.assert_array_equal(qw.values, 0)


def test_dequantize_reconstructs_within_quantization_error():
    w = np.random.RandomState(1).randn(6, 4, 3, 3, 3).astype(np.float32) * 0.1
    for precision in ("int8", "int4"):
        qw = quantize_weight_per_channel(w, precision)
        deq = dequantize_weight(qw)
        max_abs_per_channel = np.abs(w.reshape(6, -1)).max(axis=1)
        qmax = 127 if precision == "int8" else 7
        tol = (max_abs_per_channel / qmax).reshape(6, 1, 1, 1, 1)
        assert np.all(np.abs(deq - w) <= tol + 1e-6)


def test_invalid_precision_rejected():
    with pytest.raises(ValueError):
        quantize_weight_per_channel(np.zeros((2, 2)), "int16")


def test_fake_quantize_weight_forward_matches_dequantized_quantization():
    torch.manual_seed(0)
    w = torch.randn(4, 3, 3, 3, 3)
    out = fake_quantize_weight(w, "int8")
    expected = dequantize_weight(quantize_weight_per_channel(w.numpy(), "int8"))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_fake_quantize_weight_gradient_is_straight_through():
    w = torch.randn(4, 3, 3, 3, 3, requires_grad=True)
    out = fake_quantize_weight(w, "int8")
    loss = out.sum()
    loss.backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_fake_quantize_activation_nonneg_and_bounded():
    x = torch.rand(10) * 5.0  # post-ReLU-like, in [0, 5)
    scale = 5.0 / 255
    out = fake_quantize_activation(x, scale)
    assert (out >= 0).all()
    assert (out <= 5.0 + 1e-4).all()


def test_fake_quantize_activation_gradient_is_straight_through():
    x = (torch.rand(10) * 5.0).requires_grad_()
    out = fake_quantize_activation(x, 5.0 / 255)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
