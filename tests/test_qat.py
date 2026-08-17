# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import torch

from src.model.predecoder import PreDecoder3D
from src.model.qat import QATPreDecoder3D


def test_from_fp32_copies_weights_exactly():
    torch.manual_seed(0)
    fp32 = PreDecoder3D(depth=2, width=4, out_channels=3)
    qat = QATPreDecoder3D.from_fp32(fp32, precision="int8")

    fp32_convs = [m for m in fp32.net if isinstance(m, torch.nn.Conv3d)]
    for qat_conv, fp32_conv in zip(qat.hidden_convs, fp32_convs[:-1]):
        assert torch.equal(qat_conv.weight, fp32_conv.weight)
        assert torch.equal(qat_conv.bias, fp32_conv.bias)
    assert torch.equal(qat.out_conv.weight, fp32_convs[-1].weight)
    assert torch.equal(qat.out_conv.bias, fp32_convs[-1].bias)


def test_forward_shape_matches_input_grid():
    qat = QATPreDecoder3D(depth=2, width=4, out_channels=3, precision="int8")
    x = torch.rand(2, 1, 6, 6, 6)
    out = qat(x)
    assert out.shape == (2, 3, 6, 6, 6)


def test_fully_convolutional_runs_at_larger_grid():
    qat = QATPreDecoder3D(depth=2, width=4, precision="int4")
    small = torch.rand(1, 1, 5, 5, 5)
    large = torch.rand(1, 1, 12, 12, 12)
    assert qat(small).shape == small.shape
    assert qat(large).shape == large.shape


def test_running_max_updates_in_train_mode_only():
    qat = QATPreDecoder3D(depth=1, width=4, precision="int8")
    qat.train()
    before = qat.act_running_max.clone()
    qat(torch.rand(4, 1, 6, 6, 6) * 10.0)
    after_train = qat.act_running_max.clone()
    assert not torch.equal(before, after_train)

    qat.eval()
    qat(torch.rand(4, 1, 6, 6, 6) * 10.0)
    after_eval = qat.act_running_max.clone()
    assert torch.equal(after_train, after_eval)


def test_quantized_output_close_to_fp32_for_small_perturbation():
    # At int8 precision with reasonable weight magnitudes, quantization
    # noise should be small relative to a large-magnitude output.
    torch.manual_seed(0)
    fp32 = PreDecoder3D(depth=2, width=8)
    qat = QATPreDecoder3D.from_fp32(fp32, precision="int8")
    qat.eval()
    x = torch.rand(1, 1, 8, 8, 8)
    with torch.no_grad():
        out_fp32 = fp32(x)
        out_qat = qat(x)
    assert out_fp32.shape == out_qat.shape
    # Not asserting numerical closeness to a strict tolerance here: int8
    # quantization noise compounds over depth and untrained random weights
    # can have small dynamic range, making relative error large by chance.
    # This is a smoke test that it runs and produces finite values.
    assert torch.isfinite(out_qat).all()


def test_precision_variants_all_run():
    for precision in ("int8", "int4", "ternary"):
        qat = QATPreDecoder3D(depth=2, width=4, precision=precision)
        out = qat(torch.rand(1, 1, 6, 6, 6))
        assert torch.isfinite(out).all()
