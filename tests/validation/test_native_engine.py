"""Public-family and reference-fallback gates for the opt-in resident engine."""

import numpy as np
import pandas as pd
import pytest
from test_score_batch import screen

import pysceptre as ps
from pysceptre.glm.irls import fit_poisson_glm_batch
from pysceptre.glm.sufficient import fit_poisson_sufficient


def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA required")


def test_sufficient_fit_fractional_and_integer_responses():
    rng = np.random.default_rng(91)
    X = np.c_[np.ones(1000), rng.normal(size=(1000, 3))]
    Y = rng.poisson(2.0, size=(8, len(X))).astype(float)
    Y[0] *= 0.3
    ref = fit_poisson_glm_batch(X, Y)
    beta, mu, iterations, ok = fit_poisson_sufficient(X, Y)
    np.testing.assert_allclose(beta, ref.coefs, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(mu, ref.fitted_values, rtol=1e-10, atol=1e-11)
    np.testing.assert_array_equal(iterations, ref.n_iter)
    np.testing.assert_array_equal(ok, ref.converged)


@pytest.mark.parametrize("stage", ["calibration", "power", "discovery"])
def test_resident_complete_public_family(stage):
    cuda()
    y, genes, X, targets, nt, pairs, cal = screen()

    def run(backend):
        kw = dict(
            moi="high",
            control_group="complement",
            seed=55,
            n_jobs=1,
            score_backend=backend,
            score_batch_size=4,
            score_max_bytes=128 * 1024**2,
        )
        if stage == "calibration":
            return ps.run_calibration_check(
                y,
                genes,
                X,
                nt,
                n_calibration_pairs=len(cal),
                calibration_group_size=2,
                negative_control_pairs=cal,
                **kw,
            )
        if stage == "power":
            return ps.run_power_check(y, genes, X, targets, positive_control_pairs=pairs, **kw)
        return ps.run_discovery_analysis(y, genes, X, targets, pairs, **kw)

    ref, got = run("cpu"), run("cuda_native")
    for col in ref:
        if pd.api.types.is_numeric_dtype(ref[col]):
            np.testing.assert_allclose(
                got[col], ref[col], rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=col
            )
        else:
            assert got[col].equals(ref[col])
    info = got.attrs["score_execution"]
    assert info["effective"] == "cuda_native"
    assert info["response_gpu_genes"] > 0
    assert info["gpu_matmuls"] > 0
    assert info["gpu_observed_calls"] > 0
    assert info["cpu_later_stage_requests"] == 0
    assert len(got.attrs["resident_fits"]) > 0
    if stage == "discovery":
        assert info["gpu_later_stage_calls"] > 0


def test_reject_resident_permutations():
    y, g, X, t, _, p, _ = screen()
    with pytest.raises(ValueError, match="unweighted complement CRT"):
        ps.run_discovery_analysis(
            y, g, X, t, p, score_backend="cuda_native", resampling_mechanism="permutations"
        )


def test_unavailable_device_has_visible_reference_replay(monkeypatch):
    from pysceptre.test_statistic import native_resident as resident

    def unavailable(*args, **kwargs):
        raise ImportError("deliberately absent CUDA")

    monkeypatch.setattr(resident, "NativeEngine", unavailable)
    y, g, X, t, _, p, _ = screen()
    ref = ps.run_discovery_analysis(y, g, X, t, p, seed=88)
    with pytest.warns(RuntimeWarning, match="using reference CPU"):
        got = ps.run_discovery_analysis(y, g, X, t, p, seed=88, score_backend="cuda_native")
    pd.testing.assert_frame_equal(ref, got, check_flags=False)
    assert got.attrs["score_execution"]["effective"] == "cpu"


def engine(n=1000):
    cuda()
    from test_score_batch import telemetry

    from pysceptre.test_statistic.native_resident import NativeEngine

    X = np.c_[np.ones(n), np.random.default_rng(7).normal(size=(n, 3))]
    return NativeEngine(X, 512 * 1024**2, telemetry())


@pytest.mark.parametrize("kind", ["normal", "zero", "tiny", "search", "cached"])
def test_sampler_bank_and_stream(kind):
    from pysceptre.crt.sampler import crt_index_sampler

    e = engine()
    p = np.full(1000, 0.01)
    B = 499
    if kind == "zero":
        p[:] = 0
    if kind == "tiny":
        p[:] = 1e-15
    if kind == "search":
        p[:] = 0.4
    a = np.random.default_rng(827)
    b = np.random.default_rng(827)
    if kind == "cached":
        a.integers(0, 100, dtype=np.uint32)
        b.integers(0, 100, dtype=np.uint32)
    actual = e.sample(p, B, a, 50)
    expected = crt_index_sampler(p, B, b, 50)
    for x, y in zip(actual._index_arrays(0, B), expected, strict=True):
        np.testing.assert_array_equal(x, y)
    assert a.bit_generator.state == b.bit_generator.state
    if kind in ("search", "cached"):
        assert e.stats["native_sampler_replays"] == 1
    if kind == "normal":
        assert e.stats["native_banks"] == 1


def test_sampler_rolls_back_after_consumption(monkeypatch):
    cuda()
    from pysceptre.crt.native import bank
    from pysceptre.crt.sampler import crt_index_sampler

    e = engine()
    p = np.full(1000, 0.01)

    def failure(p, B, rng):
        rng.bit_generator.advance(10000)
        raise RuntimeError("forced numerical ambiguity")

    monkeypatch.setattr(bank, "bank", failure)
    a = np.random.default_rng(9)
    b = np.random.default_rng(9)
    got = e.sample(p, 499, a, 50)
    ref = crt_index_sampler(p, 499, b, 50)
    for x, y in zip(got._index_arrays(0, 499), ref, strict=True):
        np.testing.assert_array_equal(x, y)
    assert a.bit_generator.state == b.bit_generator.state
    assert e.stats["native_sampler_replays"] == 1


def test_geometric_integer_boundary_replays():
    e = engine()
    x = np.random.default_rng(8).standard_exponential()
    p = np.full(1000, -np.expm1(-x / 7))
    # If the constructed p falls in the search branch it is also safely replayed.
    e.sample(p, 499, np.random.default_rng(8), 50)
    assert e.stats["native_sampler_replays"] == 1


def test_near_poisson_fractional_and_zero_theta_reference_gate():
    e = engine(3000)
    cp = e.cp
    mu = np.full((4, 3000), 5.0)
    y = np.random.default_rng(92).poisson(mu).astype(float)
    y[1] *= 0.3
    y[2] = 0
    y[3] = 5
    with np.errstate(all="ignore"):
        _, safe = e.theta_batch(y, cp.asarray(y), cp.asarray(mu))
    assert not safe.any()


def test_degenerate_variance_declines():
    from pysceptre.test_statistic.resident import ResidentDeclined

    e = engine()
    cp = e.cp
    # All cells selected: the intercept projection removes all variance.
    H = e.X.T @ e.X
    flat = (cp.ones((2, 1000)), cp.linalg.inv(H))
    with pytest.raises(ResidentDeclined, match="variance"):
        e.observed(flat, np.arange(1000))


@pytest.mark.parametrize("side", [0, 1, 2])
def test_batched_tails_original_extremes(side):
    from pysceptre.test_statistic.native_schedule import batch_skew_normal
    from pysceptre.test_statistic.skew_normal import fit_and_evaluate_skew_normal

    r = np.random.default_rng(88)
    arrays = [
        r.normal(size=4999),
        r.standard_t(3, size=4999),
        np.ones(4999),
        r.exponential(size=4999),
    ]
    records = [(z, a) for a in arrays for z in [-50.0, -4.0, 0.0, 4.0, 50.0]]
    got = batch_skew_normal(records, side)
    expected = [fit_and_evaluate_skew_normal(z, a, side) for z, a in records]
    for x, y in zip(got, expected, strict=True):
        assert x.used == y.used
        np.testing.assert_allclose(
            [x.xi, x.omega, x.alpha, x.p],
            [y.xi, y.omega, y.alpha, y.p],
            rtol=1e-14,
            atol=0,
            equal_nan=True,
        )
