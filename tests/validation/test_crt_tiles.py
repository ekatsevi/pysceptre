"""Packed arbitrary listings and SIMD-over-genes scores retain the source algebra."""

import numpy as np
import pytest
from scipy import sparse

from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.empirical_p import compute_empirical_p_value
from pysceptre.test_statistic.score_stat import compute_null_statistics_from_draws, stack_pieces


@pytest.mark.parametrize("cells", [2, 17, 513, 201158])
def test_packed_listings_are_lossless_including_order_and_duplicates(cells):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    from pysceptre.test_statistic.crt_tiles import PackedDraws

    rng = np.random.default_rng(383)
    values = rng.integers(cells, size=10003)
    values[1:10] = values[0]
    offsets = np.array([0, 0, 33, 318, 10003])
    for packed in (True, False):
        bank = PackedDraws(values, offsets, cells, packed=packed)
        np.testing.assert_array_equal(bank.unpack().get(), values)
        np.testing.assert_array_equal(bank.offsets.get(), offsets)
        if packed:
            assert bank.data.nbytes <= (len(values) * bank.bits + 7) // 8 + 8


@pytest.mark.parametrize("packed", [True, False])
def test_coalesced_scores_and_fused_counts_match_original_csr(packed):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA required")
    from pysceptre.test_statistic.crt_tiles import (
        CompactScoreAtlas,
        PackedDrawCollection,
        PackedDraws,
    )

    rng = np.random.default_rng(283)
    n, ng = 700, 67
    X = np.c_[np.ones(n), rng.normal(size=(n, 3))]
    Y = rng.negative_binomial(5, 0.6, size=(ng, n)).astype(float)
    beta = np.array([np.log(3), 0.02, -0.01, 0.05])
    pieces = [compute_precomputation_pieces(y, X, beta, 6.0) for y in Y]
    a = np.array([p.a for p in pieces])
    w = np.array([p.w for p in pieces])
    inverses = np.array([np.linalg.inv(X.T @ (p.w[:, None] * X)) for p in pieces])
    atlas = CompactScoreAtlas(X, inverses)
    atlas.set_genes(np.arange(0, ng, 2), a[::2], w[::2])
    atlas.set_genes(np.arange(1, ng, 2), a[1::2], w[1::2])
    rows = [
        [rng.choice(n, int(rng.integers(15, 40)), replace=True) for _ in range(39)]
        for _ in range(3)
    ]
    rows[0][1] = np.r_[rows[0][1], rows[0][1][:5]]
    banks = []
    matrices = []
    for draw_rows in rows:
        offsets = np.r_[0, np.cumsum(list(map(len, draw_rows)))]
        cells = np.concatenate(draw_rows)
        banks.append(PackedDraws(cells, offsets, n, packed=packed))
        matrices.append(sparse.csr_matrix((np.ones(len(cells)), cells, offsets), shape=(39, n)))
    collection = PackedDrawCollection(banks)
    for selected in (None, np.array([66, 0, 34, 2])):
        actual, guard = collection.statistics(atlas, 2, 37, selected=selected)
        chosen = np.arange(ng) if selected is None else selected
        expected = np.array(
            [
                [
                    compute_null_statistics_from_draws(
                        stack_pieces(pieces[g].a, pieces[g].w, pieces[g].D), m[2:37]
                    )
                    for g in chosen
                ]
                for m in matrices
            ]
        )
        np.testing.assert_allclose(actual.get(), expected, rtol=1e-12, atol=1e-12)
        assert not guard.get().any()
        query_gene = np.tile(chosen, 3)
        query_target = np.repeat(np.arange(3), len(chosen))
        order = rng.permutation(len(query_gene))[: max(1, len(query_gene) // 2)]
        query, unsafe = collection.statistics_queries(
            atlas, query_gene[order], query_target[order], 2, 37
        )
        wanted = expected.reshape(-1, 35)[order]
        np.testing.assert_allclose(query.get(), wanted, rtol=1e-12, atol=1e-12)
        assert not unsafe.get().any()
        with pytest.raises(MemoryError):
            collection.statistics_queries(atlas, query_gene, query_target, 2, 37, max_bytes=1)
        z = rng.normal(size=(3, len(chosen)))
        mask = rng.random(z.shape) > 0.1
        # Force noncontiguous views; the CUDA pointer contract requires packing.
        z = np.asfortranarray(z)
        mask = np.asfortranarray(mask)
        for side in (-1, 0, 1):
            p, guard = collection.empirical(
                atlas, z, 2, 37, selected=selected, eligible=mask, side=side
            )
            expected_p = np.array(
                [
                    [
                        compute_empirical_p_value(expected[t, j], z[t, j], side)
                        for j in range(len(chosen))
                    ]
                    for t in range(3)
                ]
            )
            expected_p[~mask] = np.nan
            np.testing.assert_array_equal(p.get(), expected_p)
