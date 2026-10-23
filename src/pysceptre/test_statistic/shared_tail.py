"""Moment and directional checks reused by queries of the same null distribution."""

import math

import numpy as np
from scipy.stats import skewnorm

from .skew_normal import _MAX_GAMMA_1, _MIN_P


def shared_skew_normal(nulls, slots, observed, side=0):
    """Return query p-values, slot parameters and conservative replay flags.

    Each row is one (gene, treated count, stage) distribution. Moments and
    quantiles are computed once per row. Left and right checks remain distinct:
    the observed statistic chooses which check the owning test requires.
    NumPy reductions are along each original row, retaining its draw order.
    """
    nulls = np.asarray(nulls, dtype=np.float64)
    n = nulls.shape[1]
    with np.errstate(all="ignore"):
        mean = nulls.mean(axis=1)
        sd = np.sqrt((nulls**2).mean(axis=1) - mean**2)
        gamma = np.sum((nulls - mean[:, None]) ** 3, axis=1) / (n * sd**3)
    ordered = np.sort(nulls, axis=1)
    index = np.ceil(n * (np.arange(180, 199) / 200.0)).astype(int)
    return _evaluate_summary(
        mean,
        sd,
        gamma,
        ordered[:, (n - 1) // 2],
        ordered[:, 0],
        ordered[:, -1],
        ordered[:, index],
        -ordered[:, n - 1 - index],
        n,
        slots,
        observed,
        side,
    )


def shared_skew_normal_cuda(nulls, slots, observed, side=0):
    """CUDA moments/order, compact readback and the original SciPy tail evaluation.

    Floating-point reductions differ from NumPy. Conservative boundary flags
    request the original complete CPU test. No approximate CDF or different
    tail family is introduced. Only row summaries leave the device.
    """
    import cupy as cp

    n = nulls.shape[1]
    mean = nulls.mean(axis=1)
    sd = cp.sqrt((nulls**2).mean(axis=1) - mean**2)
    gamma = cp.sum((nulls - mean[:, None]) ** 3, axis=1) / (n * sd**3)
    ordered = cp.sort(nulls, axis=1)
    index = cp.asarray(np.ceil(n * (np.arange(180, 199) / 200.0)).astype(int))
    summary = cp.concatenate(
        (
            cp.stack(
                (mean, sd, gamma, ordered[:, (n - 1) // 2], ordered[:, 0], ordered[:, -1]), axis=1
            ),
            ordered[:, index],
            -ordered[:, n - 1 - index],
        ),
        axis=1,
    ).get()
    p, params, guard = _evaluate_summary(
        *summary[:, :6].T, summary[:, 6:25], summary[:, 25:44], n, slots, observed, side
    )
    guard |= np.abs(summary[np.asarray(slots), 2]) < 1e-8
    return p, params, guard


def _evaluate_summary(
    mean,
    sd,
    gamma,
    median,
    minimum,
    maximum,
    right_quantile,
    left_quantile,
    n,
    slots,
    observed,
    side,
):
    slots = np.asarray(slots, dtype=np.int32)
    observed = np.asarray(observed, dtype=np.float64)
    with np.errstate(all="ignore"):
        guarded = np.abs(gamma - _MAX_GAMMA_1) < 1e-8
        gamma = np.where(gamma > _MAX_GAMMA_1, 0.9 * _MAX_GAMMA_1, gamma)
        b = np.sqrt(np.float64(2.0) / math.pi)
        r = np.copysign(1.0, gamma) * (2 * np.abs(gamma) / (4 - math.pi)) ** (1.0 / 3.0)
        delta = r / (b * np.sqrt(1 + r * r))
        guarded |= np.abs(1 - delta * delta) < 1e-8
        alpha = delta / np.sqrt(1 - delta * delta)
        omega = sd / np.sqrt(1 - (b * delta) ** 2)
        xi = mean - omega * (b * delta)
    params = np.column_stack((xi, omega, alpha))
    finite = np.isfinite(params).all(axis=1) & np.isfinite(mean) & np.isfinite(sd)
    with np.errstate(all="ignore"):
        radius = sd * np.sqrt(2 * np.log(np.float64(n)))
        ratios = np.column_stack((maximum / (mean + radius), minimum / (mean - radius)))
    outlier_ok = (ratios <= 1.5).all(axis=1)
    guarded |= (np.abs(ratios - 1.5) < 1e-8).any(axis=1)
    q = np.arange(180, 199) / 200.0
    right = observed >= median[slots]
    accepted = []
    for sign in (1, -1):
        quantile = right_quantile if sign == 1 else left_quantile
        # Invalid parameters are rejected before calling scipy, as in the scalar test.
        probability = np.full(quantile.shape, np.nan)
        requested = np.zeros(len(mean), dtype=bool)
        requested[slots[right if sign == 1 else ~right]] = True
        valid = finite & outlier_ok & requested
        probability[valid] = skewnorm.sf(
            quantile[valid],
            a=(sign * alpha[valid])[:, None],
            loc=(sign * xi[valid])[:, None],
            scale=omega[valid, None],
        )
        with np.errstate(all="ignore"):
            ratio = (1 - q) / probability
        guarded |= (np.abs(ratio - 2) < 1e-7).any(axis=1)
        accepted.append(valid & ~((probability <= 0) | (ratio > 2)).any(axis=1))
    used = np.where(right, accepted[0][slots], accepted[1][slots])
    sf = right if side == 0 else np.full(len(slots), side == 1)
    p = np.full(len(slots), -1.0)
    for use_sf, function in ((True, skewnorm.sf), (False, skewnorm.cdf)):
        take = used & (sf == use_sf)
        which = slots[take]
        p[take] = function(observed[take], a=alpha[which], loc=xi[which], scale=omega[which])
    if side == 0:
        p[used] *= 2
    p[used] = np.maximum(p[used], _MIN_P)
    guarded_queries = guarded[slots] | (np.abs(observed - median[slots]) < 1e-8)
    return p, params, guarded_queries
