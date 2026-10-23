"""Additional third-stage, compact-algebra and explicit numerical-guard gates."""

import numpy as np
import pandas as pd
from test_native_engine import cuda, engine
from test_score_batch import screen

import pysceptre as ps


def test_public_native_third_stage():
    cuda()
    y, g, X, t, _, pairs, _ = screen()
    options = dict(
        seed=2718, side="left", resampling_approximation="no_approximation", score_batch_size=4
    )
    ref = ps.run_discovery_analysis(y, g, X, t, pairs, **options)
    got = ps.run_discovery_analysis(y, g, X, t, pairs, score_backend="cuda_native", **options)
    assert 3 in got.stage.to_numpy()
    for col in ref:
        if pd.api.types.is_numeric_dtype(ref[col]):
            np.testing.assert_allclose(got[col], ref[col], rtol=1e-9, atol=1e-10, equal_nan=True)
        else:
            assert got[col].equals(ref[col])
    assert got.attrs["score_execution"]["gpu_later_stage_calls"] > 0


def test_compact_matches_original_scores():
    from types import SimpleNamespace

    from pysceptre.glm.irls import fit_poisson_glm_batch
    from pysceptre.precompute.pieces import compute_precomputation_pieces
    from pysceptre.test_statistic.score_stat import (
        compute_null_statistics_from_draws,
        compute_observed_full_statistic,
        stack_pieces,
    )

    e = engine()
    r = np.random.default_rng(51)
    y = r.negative_binomial(6, 0.6, len(e.host_X)).astype(float)
    coefs = fit_poisson_glm_batch(e.host_X, y).coefs
    e.fits["g"] = SimpleNamespace(fitted_coefs=coefs, theta=6.0)
    flat, _, _ = e._make_gene("g", y)
    bank = e.sample(np.full(1000, 0.02), 499, np.random.default_rng(71), 50)
    pieces = compute_precomputation_pieces(y, e.host_X, coefs, 6.0)
    expected = compute_null_statistics_from_draws(
        stack_pieces(pieces.a, pieces.w, pieces.D), bank.slice(0, 499)
    )
    actual = e.statistics(bank, [flat], ("t", 0, 499))[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        e.observed(flat, np.arange(50)),
        compute_observed_full_statistic(pieces.a, pieces.w, pieces.D, np.arange(50)),
        rtol=1e-11,
        atol=1e-11,
    )


def test_wedge_guard_replays_original_stream(monkeypatch):
    e = engine()
    from pysceptre.crt.native import stream
    from pysceptre.crt.sampler import crt_index_sampler

    cp = e.cp

    def ambiguous(state, n):
        words = cp.zeros(n, cp.uint64)
        words[0] = 8
        words[1] = np.uint64(2**64 - 1)
        return words

    monkeypatch.setattr(stream, "gpu_raw", ambiguous)
    p = np.full(1000, 0.01)
    a = np.random.default_rng(19)
    b = np.random.default_rng(19)
    actual = e.sample(p, 499, a, 50)
    ref = crt_index_sampler(p, 499, b, 50)
    assert e.stats["native_sampler_replays"] == 1
    for x, y in zip(actual._index_arrays(0, 499), ref, strict=True):
        np.testing.assert_array_equal(x, y)
    assert a.bit_generator.state == b.bit_generator.state
