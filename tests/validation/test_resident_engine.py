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

    ref, got = run("cpu"), run("cuda_resident")
    for col in ref:
        if pd.api.types.is_numeric_dtype(ref[col]):
            np.testing.assert_allclose(
                got[col], ref[col], rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=col
            )
        else:
            assert got[col].equals(ref[col])
    info = got.attrs["score_execution"]
    assert info["effective"] == "cuda_resident"
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
            y, g, X, t, p, score_backend="cuda_resident", resampling_mechanism="permutations"
        )


def test_unavailable_device_has_visible_reference_replay(monkeypatch):
    from pysceptre.test_statistic import resident

    def unavailable(*args, **kwargs):
        raise ImportError("deliberately absent CUDA")

    monkeypatch.setattr(resident, "ResidentEngine", unavailable)
    y, g, X, t, _, p, _ = screen()
    ref = ps.run_discovery_analysis(y, g, X, t, p, seed=88)
    with pytest.warns(RuntimeWarning, match="using reference CPU"):
        got = ps.run_discovery_analysis(y, g, X, t, p, seed=88, score_backend="cuda_resident")
    pd.testing.assert_frame_equal(ref, got, check_flags=False)
    assert got.attrs["score_execution"]["effective"] == "cpu"
