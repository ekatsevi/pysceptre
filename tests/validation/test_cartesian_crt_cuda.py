"""Complete CRT scheduling, reference fit cohorts, decoded draws and resume."""

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.cartesian_crt import iter_cartesian_crt_analysis
from pysceptre.pipeline.discovery import fit_all_targets, run_discovery_ntcells_complement


@pytest.mark.parametrize("precompute,tail", [("cpu", "cpu"), ("cuda", "cuda")])
@pytest.mark.parametrize("parametric,B3", [(True, 0), (True, 299), (False, 299)])
def test_complete_crt_and_reference_draws(precompute, tail, parametric, B3):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(121)
    n = 2000
    X = np.asfortranarray(np.c_[np.ones(n), rng.normal(size=(n, 2))])
    Y = rng.negative_binomial(5, 0.7, size=(5, n)).astype(float)
    targets = {
        f"t{j}": np.sort(rng.choice(n, k, replace=False)) for j, k in enumerate([3, 80, 120, 250])
    }
    for j in (0, 1, 2):
        Y[j, targets["t1"]] *= 4
    genes = [f"g{j}" for j in range(5)]
    mask = np.ones((5, 4), dtype=bool)
    mask[1, 3] = mask[2, 2] = False
    pg, pt = np.nonzero(mask)
    pairs = pd.DataFrame(
        {"response_id": np.asarray(genes)[pg], "grna_target": np.asarray(list(targets))[pt]}
    )
    kw = dict(B1=99, B2=299, B3=B3, seed=99)
    expected = run_discovery_ntcells_complement(
        Y,
        genes,
        X,
        targets,
        pairs,
        target_chunk_size=2,
        resampling_mechanism="crt",
        fit_parametric_curve=parametric,
        **kw,
    )
    reference = {}
    names = list(targets)
    for lo in range(0, len(names), 2):
        reference.update(fit_all_targets({t: targets[t] for t in names[lo : lo + 2]}, X, **kw))

    def audit(name, probabilities, bank):
        source = reference[name]
        np.testing.assert_array_equal(probabilities, source.fitted_probabilities)
        rows = source.draws._index_arrays(0, bank.n_draws)
        np.testing.assert_array_equal(bank.unpack().get(), np.concatenate(rows))
        np.testing.assert_array_equal(bank.offsets.get(), np.r_[0, np.cumsum(list(map(len, rows)))])

    options = dict(
        eligible_pairs=mask,
        n_jobs=2,
        gene_tile_size=2,
        target_chunk_size=2,
        precompute_backend=precompute,
        tail_backend=tail,
        fit_parametric_curve=parametric,
        **kw,
    )
    telemetry = {}
    frames = list(
        iter_cartesian_crt_analysis(
            Y, genes, X, targets, on_bank=audit, telemetry=telemetry, **options
        )
    )
    actual = pd.concat(frames, ignore_index=True)
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
    for c in ["p_value", "z_orig", "fold_change", "se_fold_change"]:
        np.testing.assert_allclose(actual[c], expected[c], rtol=1e-9, atol=1e-12)
    assert telemetry["emitted_pairs"] == mask.sum()
    assert telemetry["atlas_bytes"] > 0 and telemetry["bank_bytes"] > 0
    # Refit the same selected cohort, then skip only persisted output rows.
    resumed = list(
        iter_cartesian_crt_analysis(
            Y, genes, X, targets, skip_gene_indices=np.array([0, 1]), **options
        )
    )
    pd.testing.assert_frame_equal(
        pd.concat(frames[1:], ignore_index=True), pd.concat(resumed, ignore_index=True)
    )
