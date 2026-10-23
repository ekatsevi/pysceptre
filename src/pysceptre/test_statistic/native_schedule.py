"""Bounded requested-pair batches with unchanged empirical/tail rules."""

import numpy as np
from scipy.stats import skewnorm

from .skew_normal import _MIN_P, SkewNormalEvalResult, check_for_outliers, fit_skew_normal_funct


def batch_skew_normal(records, side):
    fits = []
    candidates = []
    tails = []
    for i, (z, y) in enumerate(records):
        fit = fit_skew_normal_funct(y)
        fits.append(SkewNormalEvalResult(fit.xi, fit.omega, fit.alpha, -1.0))
        if not all(np.isfinite(v) for v in (fit.xi, fit.omega, fit.alpha, fit.mean, fit.sd)):
            continue
        ordered = np.sort(y)
        right = z >= ordered[(len(y) - 1) // 2]
        if not check_for_outliers(ordered, fit.mean, fit.sd):
            continue
        asc = ordered if right else -ordered[::-1]
        quantiles = asc[np.ceil(len(y) * (np.arange(180, 199) / 200.0)).astype(int)]
        sign = 1 if right else -1
        candidates.append((i, z, fit, right))
        tails.append((quantiles, sign * fit.alpha, sign * fit.xi, fit.omega))
    if not tails:
        return fits
    probabilities = skewnorm.sf(
        np.array([t[0] for t in tails]),
        a=np.array([t[1] for t in tails])[:, None],
        loc=np.array([t[2] for t in tails])[:, None],
        scale=np.array([t[3] for t in tails])[:, None],
    )
    accepted = []
    for row, (i, z, fit, right) in enumerate(candidates):
        prob = probabilities[row]
        # Same <=0 / >2 rejection semantics as the original sequential checks.
        if np.any(prob <= 0) or np.any((1 - np.arange(180, 199) / 200.0) / prob > 2):
            continue
        accepted.append((i, z, fit, right))
    if not accepted:
        return fits
    z = np.array([r[1] for r in accepted])
    a = np.array([r[2].alpha for r in accepted])
    loc = np.array([r[2].xi for r in accepted])
    scale = np.array([r[2].omega for r in accepted])
    use_sf = np.array([r[3] if side == 0 else side == 1 for r in accepted])
    result = np.empty(len(accepted))
    result[use_sf] = skewnorm.sf(z[use_sf], a=a[use_sf], loc=loc[use_sf], scale=scale[use_sf])
    result[~use_sf] = skewnorm.cdf(z[~use_sf], a=a[~use_sf], loc=loc[~use_sf], scale=scale[~use_sf])
    if side == 0:
        result *= 2
    for (i, _, _, _), p in zip(accepted, result, strict=True):
        fits[i].p = max(float(p), _MIN_P)
    return fits


class NeedStage(Exception):
    def __init__(self, lo, hi):
        self.bounds = (lo, hi)


def run_jobs(jobs, st, reference_job, result_row):
    import warnings

    from .fold_change import estimate_log_fold_change
    from .resampling import PairResult, run_test_from_statistics

    engine = st["resident_engine"]
    out = {}
    ready = {}
    requests = {}
    scores = {}
    pairs = {}
    tails = {}
    for gene, targets in jobs:
        if gene in engine.force_cpu:
            st["score_stats"]["sensitive_cpu_score_jobs"] += 1
            out.update(reference_job((gene, targets)))
            continue
        y, mu = engine.host_inputs(gene, st["response_matrix"], st["gene_row_index"][gene])
        try:
            flat = engine.gene(gene, y)
            for target_id in targets:
                target = st["target_precomps"][target_id]
                if len(target.trt_idxs):
                    observed = engine.observed(flat, target.trt_idxs)
                    fc, se = estimate_log_fold_change(y, mu, target.trt_idxs)
                    pairs[gene, target_id] = (observed, fc, se)
                    requests.setdefault((target_id, 0, st["B1"]), []).append(gene)
                else:
                    out[gene, target_id] = result_row(
                        gene, target_id, PairResult(np.nan, np.nan, np.nan, np.nan, np.nan, None)
                    )
            ready[gene] = flat
        except (ValueError, RuntimeError, MemoryError) as exc:
            st["score_stats"]["fallback_batches"] += 1
            warnings.warn(
                "Native observed scores replayed on CPU: " + str(exc), RuntimeWarning, stacklevel=2
            )
            for target_id in targets:
                pairs.pop((gene, target_id), None)
            for k in requests:
                requests[k] = [g for g in requests[k] if g != gene]
            out.update(reference_job((gene, targets)))
    for _ in range(4):
        for (target_id, lo, hi), names in requests.items():
            if not names:
                continue
            try:
                values = engine.statistics(
                    st["target_precomps"][target_id].draws,
                    [ready[g] for g in names],
                    (target_id, lo, hi),
                )
                st["score_stats"]["adaptive_score_batches"] += 1
                if lo > 0:
                    st["score_stats"]["gpu_later_stage_calls"] += 1
                for j, g in enumerate(names):
                    scores[g, target_id, lo, hi] = values[j]
            except (ValueError, RuntimeError, MemoryError) as exc:
                st["score_stats"]["fallback_batches"] += 1
                warnings.warn(
                    "Native scores replayed on CPU: " + str(exc), RuntimeWarning, stacklevel=2
                )
                for g in names:
                    out.update(reference_job((g, [target_id])))
                    pairs.pop((g, target_id), None)
        # Compute all second-stage tail checks together while retaining scalar moments/order.
        tail_keys = [
            (g, t)
            for g, t in pairs
            if (g, t, st["B1"], st["B1"] + st["B2"]) in scores and (g, t) not in tails
        ]
        records = [
            (pairs[k][0], scores[k[0], k[1], st["B1"], st["B1"] + st["B2"]]) for k in tail_keys
        ]
        tails.update(zip(tail_keys, batch_skew_normal(records, st["side_code"]), strict=True))
        requests = {}
        for (g, t), (z, fc, se) in list(pairs.items()):

            def nulls(lo, hi, g=g, t=t):
                key = (g, t, lo, hi)
                if key not in scores:
                    raise NeedStage(lo, hi)
                return scores[key]

            try:
                result = run_test_from_statistics(
                    z,
                    fc,
                    se,
                    nulls,
                    B1=st["B1"],
                    B2=st["B2"],
                    B3=st["B3"],
                    fit_parametric_curve=st["fit_parametric_curve"],
                    side_code=st["side_code"],
                    skew_normal_fn=lambda *args, g=g, t=t: tails[g, t],
                )
                out[g, t] = result_row(g, t, result)
                pairs.pop((g, t))
            except NeedStage as need:
                requests.setdefault((t, *need.bounds), []).append(g)
        if not pairs:
            return out
    raise RuntimeError("Native adaptive scheduler exceeded three stages")
