# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Zero-tolerance bit-exactness test: golden.py's pure-integer NumPy
datapath must reproduce QATPreDecoder3D's (fake-quantized) forward pass.

"Zero tolerance" is enforced on the correction DECISION (sign of the final
accumulator, i.e. what actually gets applied to the syndrome): that must
match exactly, with no exceptions, since that is the only thing that affects
correctness downstream. The underlying float VALUES are compared in float64
(the QAT model is upcast via .double() for this comparison only; training
itself stays float32 for speed) to a tight (1e-6 relative) tolerance rather
than literal bit-equality: torch's F.conv3d (BLAS-based, even in float64)
and golden.py's tensordot-based integer accumulation sum the same terms in a
different order, and floating-point addition is not associative, so a
residual around 1e-8 is expected and does not indicate a bug (verified this
empirically before adopting the tolerance; see docs/DECISIONS.md). Getting
literal bit-for-bit value equality would require reimplementing QAT
training itself in pure-integer torch tensors, abandoning F.conv3d and the
batched-training speed it provides; decided with the user not to do that,
since the decision-level equality is what correctness actually depends on.
"""
import numpy as np
import torch

from src.model.golden import extract_golden_layers, golden_forward
from src.model.qat import QATPreDecoder3D

VALUE_RTOL = 1e-6
VALUE_ATOL = 1e-6


def _sign_decision(logits: np.ndarray) -> np.ndarray:
    # sigmoid(logit) > 0.5 is equivalent to logit > 0.
    return logits > 0


def _calibrated_qat(precision: str, depth: int, width: int, out_channels: int, seed: int) -> QATPreDecoder3D:
    torch.manual_seed(seed)
    qat = QATPreDecoder3D(depth=depth, width=width, out_channels=out_channels, precision=precision)
    qat.train()
    with torch.no_grad():
        qat(torch.randint(0, 2, (4, 1, 6, 6, 6)).float())
    qat.eval()
    return qat


def _compare(qat: QATPreDecoder3D, x_shape: tuple) -> None:
    qat_double = qat.double()
    x = torch.randint(0, 2, x_shape).double()
    with torch.no_grad():
        torch_logits = qat_double(x).numpy().astype(np.float64)[0]  # (out_c, T, H, W)

    layers = extract_golden_layers(qat_double)
    x_np = x.numpy()[0, 0].astype(np.uint8)  # (T, H, W)
    acc = golden_forward(x_np, layers)  # (out_c, T, H, W), int64

    final_layer = layers[-1]
    golden_logits = acc.astype(np.float64) * (
        final_layer.weight_scale[:, None, None, None] * final_layer.input_scale
    )

    np.testing.assert_array_equal(
        _sign_decision(golden_logits), _sign_decision(torch_logits), err_msg="decision mismatch"
    )
    np.testing.assert_allclose(golden_logits, torch_logits, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_golden_matches_qat_torch_forward():
    for precision in ("int8", "int4", "ternary"):
        qat = _calibrated_qat(precision, depth=2, width=8, out_channels=3, seed=0)
        _compare(qat, (1, 1, 6, 6, 6))


def test_golden_matches_at_larger_grid_than_calibration():
    # Fully convolutional: golden.py must also match at a grid larger than
    # what act_running_max was calibrated on.
    qat = _calibrated_qat("int8", depth=2, width=8, out_channels=1, seed=1)
    _compare(qat, (1, 1, 10, 10, 10))
