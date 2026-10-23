"""Public Cartesian iterator against the original complete adaptive procedure."""

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.cartesian import iter_cartesian_analysis
from pysceptre.pipeline.discovery import run_discovery_ntcells_complement


@pytest.mark.parametrize("layout", ["rectangle", "compact", "hybrid"])
@pytest.mark.parametrize("parametric,B3", [(True, 299), (True, 0), (False, 299)])
@pytest.mark.parametrize("null_strategy", ["prefix", "parallel_prefix"])
@pytest.mark.parametrize(
    "fit_backend,precompute_backend,tail_backend",
    [
        ("cpu", "cpu", "cpu"),
        ("cpu", "cpu", "cuda"),
        ("cpu", "cuda", "cuda"),
        ("cuda_native", "cuda", "cuda"),
    ],
)
def test_cartesian_all_stages_and_mask(
    layout, parametric, B3, null_strategy, fit_backend, precompute_backend, tail_backend
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(121)
    n = 2000
    X = np.asfortranarray(np.c_[np.ones(n), rng.normal(size=(n, 2))])
    Y = rng.negative_binomial(5, 0.7, size=(5, n)).astype(float)
    targets = {
        f"t{j}": np.sort(rng.choice(n, k, replace=False))
        for j, k in enumerate([80, 120, 120, 250, 400])
    }
    for j in (0, 1, 2):
        Y[j, targets["t0"]] *= 4
    genes = [f"g{j}" for j in range(5)]
    mask = np.ones((5, 5), dtype=bool)
    mask[1, 3] = mask[2, 4] = mask[4, 0] = False
    pg, pt = np.nonzero(mask)
    pairs = pd.DataFrame(
        {"response_id": np.asarray(genes)[pg], "grna_target": np.asarray(list(targets))[pt]}
    )
    kw = dict(B1=99, B2=299, B3=B3, fit_parametric_curve=parametric, seed=99)
    expected = run_discovery_ntcells_complement(
        Y, genes, X, targets, pairs, resampling_mechanism="permutations", **kw
    )
    telemetry = {}
    actual = pd.concat(
        iter_cartesian_analysis(
            Y,
            genes,
            X,
            targets,
            eligible_pairs=mask,
            n_jobs=2,
            gene_tile_size=2,
            layout=layout,
            null_strategy=null_strategy,
            fit_backend=fit_backend,
            precompute_backend=precompute_backend,
            tail_backend=tail_backend,
            telemetry=telemetry,
            **kw,
        ),
        ignore_index=True,
    )
    columns = ["response_id", "grna_target"]
    actual = (
        actual.assign(
            response_id=actual.response_id.astype(str), grna_target=actual.grna_target.astype(str)
        )
        .sort_values(columns)
        .reset_index(drop=True)
    )
    expected = expected.sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[columns], expected[columns])
    np.testing.assert_array_equal(actual.stage, expected.stage)
    np.testing.assert_allclose(actual.p_value, expected.p_value, rtol=1e-9, atol=1e-12)
    for c in ["z_orig", "fold_change", "se_fold_change"]:
        np.testing.assert_allclose(actual[c], expected[c], rtol=1e-9, atol=1e-10)
    assert telemetry["emitted_pairs"] == mask.sum()
    assert sum(telemetry["stage_counts"].values()) == mask.sum()
