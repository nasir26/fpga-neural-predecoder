# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from src.qec.metrics import (
    logical_error_rate,
    speedup,
    syndrome_density_reduction,
    wilson_confidence_interval,
)


def test_wilson_ci_zero_successes():
    low, high = wilson_confidence_interval(0, 1000)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < high < 0.01


def test_wilson_ci_all_successes():
    low, high = wilson_confidence_interval(1000, 1000)
    assert high == 1.0
    assert 0.99 < low < 1.0


def test_wilson_ci_width_shrinks_with_n():
    low_small, high_small = wilson_confidence_interval(100, 1000)
    low_large, high_large = wilson_confidence_interval(1000, 10000)
    assert (high_large - low_large) < (high_small - low_small)


def test_wilson_ci_rejects_invalid_input():
    with pytest.raises(ValueError):
        wilson_confidence_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_confidence_interval(5, 3)


def test_logical_error_rate_all_correct():
    obs = np.zeros((100, 1), dtype=bool)
    ler, low, high = logical_error_rate(obs, obs)
    assert ler == 0.0
    assert low == pytest.approx(0.0, abs=1e-12)


def test_logical_error_rate_all_wrong():
    true_obs = np.zeros((50, 1), dtype=bool)
    pred_obs = np.ones((50, 1), dtype=bool)
    ler, low, high = logical_error_rate(pred_obs, true_obs)
    assert ler == 1.0
    assert high == 1.0


def test_syndrome_density_reduction():
    assert syndrome_density_reduction(0.10, 0.05) == pytest.approx(0.5)
    assert syndrome_density_reduction(0.10, 0.10) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        syndrome_density_reduction(0.0, 0.05)


def test_speedup():
    assert speedup(100.0, 50.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        speedup(100.0, 0.0)
