"""Permutation Cartesian tiles retain the shared-pool score law."""

import numpy as np
import pytest

from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.empirical_p import compute_empirical_p_value
from pysceptre.test_statistic.fold_change import estimate_log_fold_change
from pysceptre.test_statistic.score_stat import (
    PermutationPrefixSums,
    compute_observed_full_statistic,
    stack_pieces,
)


def test_prefix_direct_observed_and_empirical():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    from pysceptre.test_statistic.permutation_tiles import PermutationPool

    rng = np.random.default_rng(16)
    n = 1000
    p = 4
    ng = 4
    X = np.c_[np.ones(n), rng.normal(size=(n, p - 1))]
    Y = rng.negative_binomial(5, 0.6, size=(ng, n)).astype(float)
    beta = np.array([np.log(3), 0.02, -0.01, 0.05])
    theta = 6.0
    pieces = [compute_precomputation_pieces(y, X, beta, theta) for y in Y]
    planes = np.array([stack_pieces(q.a, q.w, q.D).T for q in pieces])
    mu = np.array([q.mu for q in pieces])
    targets = [rng.choice(n, k, replace=False) for k in [30, 50, 50, 90]]
    perms = np.array([rng.choice(n, 90, replace=False) for _ in range(259)])
    pool = PermutationPool(X, perms, targets)
    tile = pool.tile(planes, Y, mu)
    ngcode = np.repeat(np.arange(ng), 4)
    ntcode = np.tile(np.arange(4), ng)
    z, fc, se, obs_guard = tile.observed(ngcode, ntcode)
    assert not obs_guard.get().any()
    for g in range(ng):
        for t in range(4):
            j = g * 4 + t
            np.testing.assert_allclose(
                float(z[j]),
                compute_observed_full_statistic(pieces[g].a, pieces[g].w, pieces[g].D, targets[t]),
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                [float(fc[j]), float(se[j])],
                estimate_log_fold_change(Y[g], mu[g], targets[t]),
                rtol=1e-12,
                atol=1e-12,
            )
    for lo, hi in [(0, 31), (31, 130), (130, 259)]:
        shared, mapping, _, sizes, unsafe = tile.nulls([pool.target_sizes] * ng, lo, hi)
        direct, _, _, _, _ = tile.nulls([pool.target_sizes] * ng, lo, hi, strategy="direct")
        np.testing.assert_array_equal(shared.get(), direct.get())
        parallel, _, _, _, _ = tile.nulls(
            [pool.target_sizes] * ng, lo, hi, strategy="parallel_prefix"
        )
        np.testing.assert_allclose(shared.get(), parallel.get(), rtol=1e-12, atol=1e-12)
        refs = [PermutationPrefixSums(planes[g].T, perms) for g in range(ng)]
        for g in range(ng):
            for size in [30, 50, 90]:
                np.testing.assert_allclose(
                    shared[mapping[g, size]].get(),
                    refs[g].statistics(lo, hi, size),
                    rtol=1e-12,
                    atol=1e-12,
                )
        slots = mapping[ngcode, pool.target_sizes[ntcode]]
        for side in [-1, 0, 1]:
            actual, close = tile.empirical(shared, slots, z, unsafe, side)
            expected = [
                compute_empirical_p_value(shared[slots[j]].get(), float(z[j]), side)
                for j in range(len(slots))
            ]
            np.testing.assert_array_equal(actual.get(), expected)
    ragged, mapping, _, _, unsafe = tile.nulls([[30], [], [50], [30, 50]], 0, 31)
    direct, _, _, _, _ = tile.nulls([[30], [], [50], [30, 50]], 0, 31, strategy="direct")
    np.testing.assert_array_equal(ragged.get(), direct.get())
    np.testing.assert_array_equal(mapping[:, 0], [30, 0, 50, 50])
    assert not unsafe.get().any()


def test_cancellation_requests_cpu_replay_without_changing_scores():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    from pysceptre.test_statistic.permutation_tiles import PermutationPool

    X = np.ones((20, 1))
    targets = [np.arange(5)]
    permutations = np.tile(np.arange(10), (9, 1))
    pool = PermutationPool(X, permutations, targets)
    planes = np.ones((1, 3, 20))
    planes[:, 2] /= np.sqrt(5)
    tile = pool.tile(planes, np.ones((1, 20)), np.ones((1, 20)))
    z, _, _, observed_guard = tile.observed([0], [0])
    assert observed_guard.get()[0]
    nulls, mapping, _, _, unsafe = tile.nulls([[5]], 0, 9)
    assert unsafe.get()[0]
    _, close = tile.empirical(nulls, mapping[[0], [5]], z, unsafe)
    assert close.get()[0]
