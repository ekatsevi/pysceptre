"""Reference replay trims unused ranks without changing the owner's arithmetic."""

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline import cartesian
from pysceptre.pipeline.discovery import run_discovery_ntcells_complement


def test_trimmed_replay_and_resume(monkeypatch):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    from pysceptre.test_statistic.permutation_tiles import PermutationTile

    rng = np.random.default_rng(912)
    n = 1500
    X = np.asfortranarray(np.c_[np.ones(n), rng.normal(size=n)])
    Y = rng.negative_binomial(5, 0.7, size=(4, n)).astype(float)
    targets = {
        f"t{j}": np.sort(rng.choice(n, 40 if j == 0 else 200, replace=False)) for j in range(64)
    }
    genes = [f"g{j}" for j in range(4)]
    mask = np.ones((4, 64), dtype=bool)
    pairs = pd.MultiIndex.from_product(
        [genes, list(targets)], names=["response_id", "grna_target"]
    ).to_frame(index=False)
    kw = dict(B1=49, B2=199, B3=0, seed=313)
    expected = run_discovery_ntcells_complement(
        Y, genes, X, targets, pairs, resampling_mechanism="permutations", **kw
    )
    original = PermutationTile.empirical

    def guarded(*args, **kwargs):
        p, near = original(*args, **kwargs)
        if len(near):
            near[0] = 1
        return p, near

    widths = []
    owner = cartesian.PermutationPrefixSums

    class RecordedPrefix(owner):
        def __init__(self, stacked, permutations):
            widths.append(permutations.shape[1])
            super().__init__(stacked, permutations)

    monkeypatch.setattr(PermutationTile, "empirical", guarded)
    monkeypatch.setattr(cartesian, "PermutationPrefixSums", RecordedPrefix)
    options = dict(eligible_pairs=mask, gene_tile_size=2, **kw)
    frames = list(cartesian.iter_cartesian_analysis(Y, genes, X, targets, **options))
    assert widths and min(widths) == 40
    actual = pd.concat(frames, ignore_index=True)
    columns = ["response_id", "grna_target"]
    expected = expected.sort_values(columns).reset_index(drop=True)
    actual = (
        actual.assign(
            response_id=actual.response_id.astype(str), grna_target=actual.grna_target.astype(str)
        )
        .sort_values(columns)
        .reset_index(drop=True)
    )
    np.testing.assert_array_equal(actual.stage, expected.stage)
    for c in ["p_value", "z_orig", "fold_change", "se_fold_change"]:
        np.testing.assert_allclose(actual[c], expected[c], rtol=1e-9, atol=1e-12)
        chosen = actual.grna_target.eq("t0") & actual.response_id.isin(["g0", "g2"])
        np.testing.assert_array_equal(actual.loc[chosen, c], expected.loc[chosen, c])
    resumed = list(
        cartesian.iter_cartesian_analysis(
            Y, genes, X, targets, skip_gene_indices=np.array([0, 1]), **options
        )
    )
    pd.testing.assert_frame_equal(
        pd.concat(frames[1:], ignore_index=True), pd.concat(resumed, ignore_index=True)
    )


@pytest.mark.parametrize(
    "options", [{"gene_indices": [2**32]}, {"skip_gene_indices": [0.5]}, {"skip_gene_indices": [9]}]
)
def test_invalid_gene_positions_raise_before_fitting(options):
    X = np.c_[np.ones(20), np.arange(20)]
    iterator = cartesian.iter_cartesian_analysis(
        np.ones((2, 20)), ["a", "b"], X, {"t": np.arange(10)}, **options
    )
    with pytest.raises(ValueError):
        next(iterator)
