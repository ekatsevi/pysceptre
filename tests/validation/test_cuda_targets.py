"""CUDA is opt-in; failure must safely replay the existing CPU target fit."""

import numpy as np
import pytest

from pysceptre.glm.cuda import CudaTargetDesign
from pysceptre.glm.irls import fit_binomial_glm_batch
from pysceptre.pipeline import discovery as disc


def inputs():
    rng = np.random.default_rng(43)
    X = np.c_[np.ones(700), rng.normal(size=(700, 3))]
    Y = rng.binomial(1, 0.04, size=(5, 700)).astype(float)
    return X, Y


def test_explicit_cpu_fallback_and_strict_failure(monkeypatch):
    X, Y = inputs()
    d = CudaTargetDesign(X)

    def unavailable(*args):
        raise RuntimeError("controlled unavailable device")

    monkeypatch.setattr(d, "_fit_gpu", unavailable)
    expected = fit_binomial_glm_batch(X, Y)
    with pytest.warns(RuntimeWarning, match="using CPU"):
        actual = d.fit(Y)
    for name in vars(expected):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert d.gpu_calls == 0 and d.fallback_calls == 1
    d.fallback = False
    with pytest.raises(RuntimeError, match="controlled unavailable"):
        d.fit(Y)


def test_target_engine_fallback_does_not_change_draws(monkeypatch):
    X, Y = inputs()
    targets = {f"t{i}": np.flatnonzero(row) for i, row in enumerate(Y)}
    kwargs = dict(B1=19, B2=0, B3=0, seed=19, n_jobs=1)
    expected = disc.fit_all_targets(targets, X, **kwargs)
    design = CudaTargetDesign(X)
    monkeypatch.setattr(
        design, "_fit_gpu", lambda y: (_ for _ in ()).throw(RuntimeError("unavailable"))
    )
    with pytest.warns(RuntimeWarning, match="using CPU"):
        actual = disc.fit_all_targets(targets, X, design=design, **kwargs)
    for key in expected:
        np.testing.assert_array_equal(
            actual[key].fitted_probabilities, expected[key].fitted_probabilities
        )
        # Staged draw CSR is built lazily; compare the underlying actual lists.
        a = actual[key].draws.slice(0, 19)
        b = expected[key].draws.slice(0, 19)
        for field in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_gpu_matches_existing_irls_and_difficult_cases():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device needed")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device needed")
    X, Y = inputs()
    d = CudaTargetDesign(X, fallback=False)
    expected = fit_binomial_glm_batch(X, Y)
    actual = d.fit(Y)
    np.testing.assert_allclose(actual.coefs, expected.coefs, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(actual.fitted_values, expected.fitted_values, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(actual.converged, expected.converged)
    np.testing.assert_array_equal(actual.n_iter, expected.n_iter)
    assert d.gpu_calls == 1 and not d.fallback_calls
    tiny = CudaTargetDesign(X, max_bytes=1, fallback=False)
    with pytest.raises(MemoryError, match="exceeds"):
        tiny.fit(Y)
    # A separated target must use the CPU convergence behavior when fallback is selected.
    separated = (X[:, 1] > 0).astype(float)[None, :]
    df = CudaTargetDesign(X)
    with pytest.warns(RuntimeWarning, match="CPU"):
        result = df.fit(separated)
    reference = fit_binomial_glm_batch(X, separated)
    np.testing.assert_array_equal(result.fitted_values, reference.fitted_values)


@pytest.mark.parametrize("mechanism", ["crt", "permutations"])
def test_cuda_three_workflows_match_cpu(mechanism):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device needed")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device needed")
    import pandas as pd
    from scipy import sparse

    import pysceptre as ps

    X, indicators = inputs()
    rng = np.random.default_rng(5)
    y = sparse.csr_matrix(rng.negative_binomial(6, 0.6, size=(4, len(X))).astype(float))
    genes = ["a", "b", "c", "d"]
    targets = {f"t{i}": np.flatnonzero(row) for i, row in enumerate(indicators)}
    nt = {f"n{i}": np.arange(i * 35, (i + 1) * 35, dtype=np.int64) for i in range(4)}
    pairs = pd.DataFrame({"response_id": genes, "grna_target": ["t0", "t1", "t2", "t3"]})
    cal = pd.DataFrame({"response_id": genes, "grna_target": ["n0&n1"] * 4})
    settings = dict(
        moi="high", control_group="complement", seed=55, resampling_mechanism=mechanism, n_jobs=1
    )
    for analysis in ["calibration", "power", "discovery"]:

        def run(solver, analysis=analysis):
            kw = dict(settings, glm_solver=solver)
            if analysis == "calibration":
                return ps.run_calibration_check(
                    y,
                    genes,
                    X,
                    nt,
                    n_calibration_pairs=4,
                    calibration_group_size=2,
                    negative_control_pairs=cal,
                    **kw,
                )
            if analysis == "power":
                return ps.run_power_check(y, genes, X, targets, positive_control_pairs=pairs, **kw)
            return ps.run_discovery_analysis(y, genes, X, targets, pairs, **kw)

        import warnings

        cpu = run("outer")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gpu = run("cuda_targets")
        assert not any("CUDA target fit declined" in str(w.message) for w in caught)
        assert cpu.columns.equals(gpu.columns)
        for col in cpu:
            if pd.api.types.is_numeric_dtype(cpu[col]):
                np.testing.assert_allclose(
                    cpu[col], gpu[col], rtol=1e-9, atol=1e-10, equal_nan=True, err_msg=col
                )
            else:
                assert cpu[col].equals(gpu[col]), col


def test_cuda_refuses_process_workers_before_device_initialization():
    X, _ = inputs()
    with pytest.raises(ValueError, match="n_jobs=1"):
        disc._design_for("cuda_targets", X, n_jobs=2)
