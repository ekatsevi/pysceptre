"""Controlled CPU batching and optional resident CUDA scores preserve adjusted inference."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import pysceptre as ps
from pysceptre.test_statistic.score_batch import ScoreBatch
from pysceptre.test_statistic.score_stat import compute_null_statistics_from_draws, draws_to_matrix


def telemetry():
    return dict(
        uploaded_score_bytes=0,
        uploaded_draw_bytes=0,
        downloaded_score_bytes=0,
        score_matmuls=0,
        gpu_matmuls=0,
        score_columns=0,
        peak_pool_reserved_bytes=0,
    )


@pytest.mark.parametrize("backend", ["cpu_batch", "cuda_batch"])
def test_independent_scores_empty_and_repeated_positions(backend):
    if backend == "cuda_batch":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA needed")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA needed")
    rng = np.random.default_rng(52)
    stacks = []
    for _ in range(3):
        stacks.append(
            np.c_[rng.normal(size=20), np.full(20, 3.0), rng.normal(scale=0.01, size=(20, 3))]
        )
    draws = draws_to_matrix([np.array([1, 1, 2, 3]), np.array([], dtype=int), np.array([4, 9])], 20)
    stats = telemetry()
    batch = ScoreBatch(
        np.concatenate(stacks, axis=1), 5, backend, max_bytes=1024**2, telemetry=stats
    )
    actual = batch.statistics(draws, [0, 2])
    for row, g in enumerate([0, 2]):
        expected = compute_null_statistics_from_draws(stacks[g], draws)
        np.testing.assert_allclose(actual[row], expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        # Explicit repeated cell must count twice, never collapse to membership once.
        sums = stacks[g][[1, 1, 2, 3]].sum(axis=0)
        direct = sums[0] / np.sqrt(sums[1] - np.sum(sums[2:] ** 2))
        np.testing.assert_allclose(actual[row, 0], direct, atol=1e-12)
    assert stats["score_columns"] == 2


def screen():
    rng = np.random.default_rng(43)
    n = 700
    X = np.c_[np.ones(n), rng.normal(size=(n, 3))]
    y = rng.negative_binomial(6, 0.6, size=(12, n)).astype(float)
    targets = {f"t{i}": np.sort(rng.choice(n, 50, replace=False)) for i in range(4)}
    y[:3, targets["t0"]] = rng.poisson(0.6, size=(3, len(targets["t0"])))
    nt = {f"n{i}": np.arange(i * 35, (i + 1) * 35, dtype=np.int64) for i in range(4)}
    genes = [f"g{i}" for i in range(12)]
    pairs = pd.DataFrame(
        [(g, t) for g in genes for t in targets], columns=["response_id", "grna_target"]
    )
    cal = pd.DataFrame([(g, "n0&n1") for g in genes], columns=["response_id", "grna_target"])
    return sparse.csr_matrix(y), genes, X, targets, nt, pairs, cal


@pytest.mark.parametrize("backend", ["cpu_batch", "cuda_batch"])
@pytest.mark.parametrize("mechanism", ["crt", "permutations"])
def test_complete_families_and_stages(backend, mechanism):
    if backend == "cuda_batch":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA needed")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA needed")
    y, genes, X, targets, nt, pairs, cal = screen()
    kw = dict(
        moi="high", control_group="complement", resampling_mechanism=mechanism, seed=55, n_jobs=1
    )
    for stage in ["calibration", "power", "discovery"]:

        def run(mode, stage=stage):
            opts = dict(kw, score_backend=mode, score_batch_size=3)
            if stage == "calibration":
                return ps.run_calibration_check(
                    y,
                    genes,
                    X,
                    nt,
                    n_calibration_pairs=len(cal),
                    calibration_group_size=2,
                    negative_control_pairs=cal,
                    **opts,
                )
            if stage == "power":
                return ps.run_power_check(
                    y, genes, X, targets, positive_control_pairs=pairs, **opts
                )
            return ps.run_discovery_analysis(y, genes, X, targets, pairs, **opts)

        expected = run("cpu")
        actual = run(backend)
        for col in expected:
            if pd.api.types.is_numeric_dtype(expected[col]):
                np.testing.assert_allclose(
                    actual[col], expected[col], rtol=1e-9, atol=1e-11, equal_nan=True, err_msg=col
                )
            else:
                assert actual[col].equals(expected[col]), col
        assert actual.attrs["score_execution"]["fallback_batches"] == 0
        if stage == "discovery":
            assert actual.attrs["score_execution"]["cpu_later_stage_requests"] > 0
        if backend == "cuda_batch":
            assert actual.attrs["score_execution"]["gpu_matmuls"] > 0


def test_bounds_and_unsupported_parallelism():
    stacks = np.ones((20, 6))
    with pytest.raises(MemoryError):
        ScoreBatch(stacks, 6, "cpu_batch", max_bytes=1, telemetry=telemetry())
    y, g, X, t, _, p, _ = screen()
    with pytest.raises(ValueError, match="n_jobs=1"):
        ps.run_discovery_analysis(y, g, X, t, p, score_backend="cuda_batch", n_jobs=2)
