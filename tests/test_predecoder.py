# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from src.model.predecoder import PreDecoder3D, depth_for_receptive_field, receptive_field


@pytest.mark.parametrize("depth,expected_r", [(2, 5), (4, 9), (1, 3), (0, 1)])
def test_receptive_field_formula(depth, expected_r):
    assert receptive_field(depth) == expected_r
    assert depth_for_receptive_field(expected_r) == depth


def test_depth_for_receptive_field_rejects_even():
    with pytest.raises(ValueError):
        depth_for_receptive_field(6)


@pytest.mark.parametrize("depth,r", [(2, 5), (4, 9)])
def test_empirical_receptive_field_matches_formula(depth, r):
    model = PreDecoder3D(depth=depth, width=8)
    grid = r + 4
    x = torch.zeros(1, 1, grid, grid, grid, requires_grad=True)
    center = grid // 2
    out = model(x)
    out[0, 0, center, center, center].backward()
    grad = x.grad[0, 0].abs()
    touched = (grad > 1e-12).nonzero()
    for axis in range(3):
        span = (touched[:, axis].max() - touched[:, axis].min() + 1).item()
        assert span == r


def test_forward_shape_matches_input_grid():
    model = PreDecoder3D(depth=2, width=8, in_channels=1, out_channels=1)
    x = torch.zeros(4, 1, 6, 6, 6)
    out = model(x)
    assert out.shape == (4, 1, 6, 6, 6)


def test_fully_convolutional_runs_at_larger_grid_than_trained():
    # Same weights, no retraining, applied to a grid larger than R.
    model = PreDecoder3D(depth=2, width=8)
    small = torch.zeros(1, 1, 5, 5, 5)
    large = torch.zeros(1, 1, 12, 12, 12)
    out_small = model(small)
    out_large = model(large)
    assert out_small.shape == small.shape
    assert out_large.shape == large.shape
