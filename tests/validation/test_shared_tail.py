"""Shared distributions retain separate, observed-dependent tail checks."""

import numpy as np
import pytest

from pysceptre.test_statistic.shared_tail import shared_skew_normal
from pysceptre.test_statistic.skew_normal import fit_and_evaluate_skew_normal


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_shared_tail_matches_scalar_queries(side):
    rng = np.random.default_rng(381)
    nulls = rng.normal(size=(9, 4999))
    nulls[1] = np.exp(nulls[1] / 6)
    nulls[2] = -np.exp(nulls[2] / 6)
    nulls[3, 0] = 80
    nulls[4] *= 0
    nulls[5, 0] = np.nan
    nulls[6] = rng.standard_t(3, 4999)
    slots = np.repeat(np.arange(9), 3)
    z = np.tile([-2.5, 0.1, 2.5], 9)
    with np.errstate(all="ignore"):
        actual, params, _ = shared_skew_normal(nulls, slots, z, side)
        expected = [
            fit_and_evaluate_skew_normal(v, nulls[s], side) for s, v in zip(slots, z, strict=True)
        ]
    np.testing.assert_array_equal(actual > -0.5, [v.used for v in expected])
    np.testing.assert_allclose(
        actual, [v.p for v in expected], rtol=1e-12, atol=1e-15, equal_nan=True
    )
    np.testing.assert_allclose(
        params[slots],
        [[v.xi, v.omega, v.alpha] for v in expected],
        rtol=1e-13,
        atol=1e-15,
        equal_nan=True,
    )


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_cuda_summaries_match_the_same_scalar_tail(side):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    from pysceptre.test_statistic.shared_tail import shared_skew_normal_cuda

    rng = np.random.default_rng(681)
    nulls = rng.normal(size=(12, 4999))
    nulls[1] = np.exp(nulls[1] / 6)
    nulls[2] = -np.exp(nulls[2] / 6)
    nulls[3, 0] = 80
    nulls[4] = 0
    nulls[5, 0] = np.nan
    slots = np.repeat(np.arange(12), 3)
    observed = np.tile([-3.1, 0.13, 3.3], 12)
    with np.errstate(all="ignore"):
        actual, params, _ = shared_skew_normal_cuda(cp.asarray(nulls), slots, observed, side)
        expected = [
            fit_and_evaluate_skew_normal(z, nulls[k], side)
            for k, z in zip(slots, observed, strict=True)
        ]
    np.testing.assert_array_equal(actual > -0.5, [v.used for v in expected])
    np.testing.assert_allclose(
        actual, [v.p for v in expected], rtol=1e-10, atol=1e-14, equal_nan=True
    )
    np.testing.assert_allclose(
        params[slots],
        [[v.xi, v.omega, v.alpha] for v in expected],
        rtol=1e-9,
        atol=1e-13,
        equal_nan=True,
    )
