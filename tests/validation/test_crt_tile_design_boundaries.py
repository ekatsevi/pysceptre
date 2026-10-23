"""Synthetic boundary coverage for the two explicit CRT query layouts."""

import numpy as np
import pytest
from scipy import sparse

from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.score_stat import compute_null_statistics_from_draws, stack_pieces


@pytest.mark.parametrize("p", [1, 16])
@pytest.mark.parametrize("packed", [True, False])
def test_query_layouts_at_design_and_draw_block_boundaries(p, packed):
    """Both layouts preserve exact per-draw operations with sparse gene requests."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    from pysceptre.test_statistic.crt_tiles import (
        CompactScoreAtlas,
        PackedDrawCollection,
        PackedDraws,
    )

    rng = np.random.default_rng(982)
    n, ng = 384, 7
    X = np.c_[np.ones(n), rng.normal(size=(n, p - 1))]
    beta = np.r_[np.log(3), np.full(p - 1, 0.01)]
    Y = rng.negative_binomial(5, 0.6, size=(ng, n)).astype(float)
    pieces = [compute_precomputation_pieces(y, X, beta, 6.0) for y in Y]
    inverses = np.array([np.linalg.inv(X.T @ (q.w[:, None] * X)) for q in pieces])
    atlas = CompactScoreAtlas(X, inverses)
    atlas.set_genes(np.arange(ng), np.array([q.a for q in pieces]), np.array([q.w for q in pieces]))
    banks, matrices = [], []
    for _ in range(2):
        rows = [rng.integers(n, size=int(rng.integers(20, 70))) for _ in range(261)]
        cells = np.concatenate(rows)
        offsets = np.r_[0, np.cumsum(list(map(len, rows)))]
        banks.append(PackedDraws(cells, offsets, n, packed=packed))
        matrices.append(sparse.csr_matrix((np.ones(len(cells)), cells, offsets), shape=(261, n)))
    collection = PackedDrawCollection(banks)
    genes, targets = np.array([0, 6, 3]), np.array([0, 1, 0])
    outputs = [
        collection.statistics_queries(atlas, genes, targets, 2, 261, strategy=s)
        for s in ("grouped", "draw")
    ]
    np.testing.assert_array_equal(
        outputs[0][0].get().view(np.uint64), outputs[1][0].get().view(np.uint64)
    )
    np.testing.assert_array_equal(outputs[0][1].get(), outputs[1][1].get())
    expected = np.array(
        [
            compute_null_statistics_from_draws(
                stack_pieces(pieces[g].a, pieces[g].w, pieces[g].D), matrices[t][2:261]
            )
            for g, t in zip(genes, targets, strict=True)
        ]
    )
    np.testing.assert_allclose(outputs[1][0].get(), expected, rtol=1e-12, atol=1e-12)
