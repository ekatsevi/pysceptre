"""Complete collection is required before one global correction."""

import numpy as np
import pytest

from pysceptre.pipeline.complete_family import collect_complete_pvalues, global_bh


def test_full_collection_retains_nan_and_r_bh_values():
    mask = np.array([[True, True], [False, True]])
    p = collect_complete_pvalues(
        mask,
        [
            (np.array([1]), np.array([1]), [0.04]),
            (np.array([0, 0]), np.array([1, 0]), [np.nan, 0.01]),
        ],
    )
    np.testing.assert_array_equal(p, [[0.01, np.nan], [np.nan, 0.04]])
    # Verified against stats::p.adjust(c(.01, NA, .04), method='BH').
    np.testing.assert_array_equal(global_bh(p), [[0.02, np.nan], [np.nan, 0.04]])


@pytest.mark.parametrize(
    "chunks",
    [
        [(np.array([0]), np.array([0]), [0.1])],
        [(np.array([0, 0]), np.array([0, 0]), [0.1, 0.2])],
        [(np.array([0]), np.array([0]), [0.1]), (np.array([0]), np.array([0]), [0.2])],
        [(np.array([0, 1]), np.array([0, 0]), [0.1, 0.2])],
    ],
)
def test_incomplete_duplicate_and_outside_results_are_refused(chunks):
    mask = np.array([[True, True], [False, True]])
    with pytest.raises(ValueError):
        collect_complete_pvalues(mask, chunks)


def test_global_bh_is_not_per_batch_and_ties_are_stable():
    values = np.array([0.001, 0.02, 0.02, 0.8, 1.004])
    # stats::p.adjust output, with its default family size and upper cap.
    np.testing.assert_allclose(global_bh(values), [0.005, 1 / 30, 1 / 30, 1.0, 1.0], rtol=1e-15)
    assert global_bh(values)[0] != global_bh(values[:1])[0]
