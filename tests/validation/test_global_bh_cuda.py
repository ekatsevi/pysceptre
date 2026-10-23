"""Whole-family correction preserves the CPU procedure's results and decisions."""

import numpy as np
import pytest

from pysceptre.pipeline.complete_family import global_bh, global_bh_cuda


@pytest.mark.parametrize("size", [0, 1, 31, 1025, 1000003])
def test_cuda_global_bh_is_exact_for_finite_results(size):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(733)
    p = np.exp(rng.uniform(-700, 0.01, size))
    p[::17] = np.nan
    p[1::19] = 0.02
    p[2::23] = 0.0
    p[3::29] = 1.004
    expected = global_bh(p)
    for actual in (global_bh_cuda(p), global_bh_cuda(cp.asarray(p), return_device=True).get()):
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = np.isfinite(expected)
        np.testing.assert_array_equal(
            actual[valid].view(np.uint64), expected[valid].view(np.uint64)
        )
        np.testing.assert_array_equal(actual < 0.1, expected < 0.1)


def test_cuda_global_bh_shape_and_invalid_input():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    p = np.array([[0.01, np.nan], [0.04, np.nan]])
    np.testing.assert_array_equal(global_bh_cuda(p), global_bh(p))
    for p in ([np.inf], [-0.01]):
        with pytest.raises(ValueError):
            global_bh_cuda(p)
    with pytest.raises(MemoryError):
        global_bh_cuda(np.ones(20), max_bytes=1)


def test_cuda_global_bh_noncontiguous_signed_zero_and_all_nan():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    p = np.array([[0.0, -0.0, 1e-310], [-0.0, 0.0, 1.1], [np.nan, 0.02, 0.02]]).T
    for values in (p, cp.asarray(p), np.full((3, 2), np.nan)):
        host = values.get() if isinstance(values, cp.ndarray) else values
        expected = global_bh(host)
        actual = global_bh_cuda(values)
        valid = np.isfinite(expected)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(
            actual[valid].view(np.uint64), expected[valid].view(np.uint64)
        )
