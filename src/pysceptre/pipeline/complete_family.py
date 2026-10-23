"""Complete-family collection and the owning R procedure's global BH correction."""

import numpy as np


def collect_complete_pvalues(eligible_pairs, chunks):
    """Collect (gene codes, target codes, p-values), refusing incomplete families.

    Every eligible position must appear exactly once and no other position may
    appear. NaN results remain recorded members of the family. The consumer can
    distinguish missing results from explicitly returned NaNs.
    """
    eligible = np.asarray(eligible_pairs)
    if eligible.ndim != 2 or eligible.dtype != np.bool_:
        raise ValueError("Eligibility must be a boolean matrix")
    ng, nt = eligible.shape
    seen = np.zeros(ng * nt, dtype=bool)
    values = np.full(ng * nt, np.nan)
    for gene, target, p in chunks:
        gene, target, p = np.asarray(gene), np.asarray(target), np.asarray(p, dtype=float)
        if any(v.ndim != 1 for v in (gene, target, p)) or not len(gene) == len(target) == len(p):
            raise ValueError("Result columns must have equal one-dimensional shapes")
        if not all(np.issubdtype(v.dtype, np.integer) for v in (gene, target)):
            raise ValueError("Result positions must be integer codes")
        if np.any(gene < 0) or np.any(gene >= ng) or np.any(target < 0) or np.any(target >= nt):
            raise ValueError("Result position is outside the intended family")
        codes = gene.astype(np.int64) * nt + target
        if len(np.unique(codes)) != len(codes) or seen[codes].any():
            raise ValueError("Duplicate result positions")
        if not eligible.ravel()[codes].all():
            raise ValueError("Results include positions outside the eligible family")
        if np.isinf(p).any() or np.any(p < 0):
            raise ValueError("P-values must be nonnegative finite values or explicit NaNs")
        values[codes] = p
        seen[codes] = True
    if not np.array_equal(seen.reshape(eligible.shape), eligible):
        raise ValueError("The intended family is incomplete")
    return values.reshape(eligible.shape)


def global_bh(p_values):
    """Global stats::p.adjust(method='BH') semantics, including explicit NaNs.

    The default R procedure excludes NaNs from its multiplier, retains their
    output positions and caps adjusted values at one. Raw empirical two-sided
    values slightly above one are retained before this correction, as returned
    by the owning low-level test.
    """
    p = np.asarray(p_values, dtype=float)
    if np.isinf(p).any() or np.any(p < 0):
        raise ValueError("P-values must be nonnegative finite values or NaNs")
    adjusted = np.full(p.size, np.nan)
    positions = np.flatnonzero(np.isfinite(p.ravel()))
    n = len(positions)
    if n:
        order = np.argsort(p.ravel()[positions], kind="stable")
        ranked = positions[order]
        corrected = (n / np.arange(n, 0, -1, dtype=float)) * p.ravel()[ranked[::-1]]
        adjusted[ranked[::-1]] = np.minimum(1, np.minimum.accumulate(corrected))
    return adjusted.reshape(p.shape)
