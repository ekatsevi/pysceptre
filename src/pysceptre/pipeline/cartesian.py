"""Explicit experimental CUDA execution of complete permutation pair rectangles."""

import sys
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from time import perf_counter

import numpy as np
import pandas as pd

from ..crt.permutations import permutation_draws
from ..glm.irls import x_outer_flat
from ..precompute.pieces import compute_precomputation_pieces
from ..test_statistic.resampling import run_low_level_test_full
from ..test_statistic.score_stat import (
    PermutationPrefixSums,
    PermutationSliceDraws,
    prefix_scan_pays,
    stack_pieces,
)
from ..test_statistic.shared_tail import shared_skew_normal, shared_skew_normal_cuda
from .discovery import _CI_Z, _ENGINE_LOCK, _engine_blas_limit, _get_row, fit_all_genes
from .pairwise_qc import nonzero_counts


def _requested_sizes(genes, targets, sizes, count, rectangle):
    if rectangle:
        active = np.bincount(genes, minlength=count) > 0
        return [sizes if active[g] else [] for g in range(count)]
    return [sizes[targets[genes == g]] for g in range(count)]


def _frame(gene_ids, target_ids, gene_rows, pg, pt, p, z, fc, se, stage):
    return pd.DataFrame(
        {
            "response_id": pd.Categorical.from_codes(gene_rows[pg], categories=gene_ids),
            "grna_target": pd.Categorical.from_codes(pt, categories=target_ids),
            "p_value": p,
            "fold_change": fc,
            "se_fold_change": se,
            "pct_change_es": (fc - 1) * 100,
            "pct_change_es_ci_low": (fc - _CI_Z * se - 1) * 100,
            "pct_change_es_ci_high": (fc + _CI_Z * se - 1) * 100,
            "z_orig": z,
            "stage": np.where(stage == 0, np.nan, stage.astype(float)),
        }
    )


def _locked_iteration(function):
    @wraps(function)
    def iterate(*args, **kwargs):
        with _ENGINE_LOCK, _engine_blas_limit():
            yield from function(*args, **kwargs)

    return iterate


@_locked_iteration
def iter_cartesian_analysis(
    response_matrix,
    gene_ids,
    covariate_matrix,
    grna_target_cells,
    *,
    eligible_pairs=None,
    gene_indices=None,
    seed=None,
    B1=499,
    B2=4999,
    B3=24999,
    fit_parametric_curve=True,
    side="both",
    gene_tile_size=32,
    n_jobs=1,
    layout="hybrid",
    null_strategy="prefix",
    tail_backend="cpu",
    fit_backend="cpu",
    precompute_backend="cpu",
    fit_batch_size=64,
    fit_max_bytes=4 * 1024**3,
    skip_gene_indices=None,
    telemetry=None,
    on_fit=None,
):
    """Yield bounded discovery tables for an explicitly selected CUDA permutation run.

    This opt-in route supports binary union treatment, complement controls and
    the original full score statistic. By default it retains CPU width-one gene fits,
    permutation generator and empirical/skew-normal/fallback rules. No
    statistical approximation is selected implicitly. CUDA failures raise;
    numerically ambiguous queries are replayed with the owning CPU procedure.

    `eligible_pairs` is a boolean (genes, targets) mask of caller-qualified
    pairs, with columns in dictionary order. When omitted, the original
    treated/complement nonzero-count threshold of seven constructs it per tile.
    `gene_indices` selects output rows while retaining every supplied target
    when sizing the shared permutation pool. `n_jobs` counts CPU fitting and
    preparation workers; each worker uses one BLAS thread.

    `fit_backend="cuda_native"` explicitly selects the existing sufficient-
    statistic GPU Poisson fit and integer-count dispersion implementation,
    including its complete CPU model replays. Its arithmetic differs from the
    CPU route. `cpu_fork` uses the unchanged CPU fitter with process workers;
    it requires a fresh Linux process before importing a GPU runtime. The
    default `cpu` uses threads and also supports repeated calls after CUDA
    initialization. `fit_batch_size` and `fit_max_bytes` bound the native
    fitter's workspaces and cache. They do not cap whole-process memory.

    `precompute_backend="cuda"` builds the original a,w,D planes on device.
    Sensitive native fits and ill-conditioned precomputations retain original
    CPU planes. `tail_backend="cuda"` computes null moments and order summaries
    on device; the original SciPy CDF/SF still evaluates probabilities on CPU.
    The default for both is CPU. Each optional route changes FP64 reduction
    order and requires numerical qualification for the intended workload.

    `layout` is `rectangle`, `compact` or `hybrid`. Rectangle computes every
    target position for observed scores and every distinct size for each
    active gene at later stages. Hybrid retains the regular first stage and
    uses only active sizes thereafter. Compact uses eligible positions and
    sizes throughout. Failed QC positions never enter results or stage counts.
    `null_strategy` selects a shared prefix scan, a direct sequential scan per
    requested size, or `parallel_prefix`, a warp prefix scan with reordered
    additions. Every strategy reads identical draws.

    The iterator performs no correction across batches. The consumer must
    collect the complete intended family before applying global correction.
    `on_fit`, when supplied, receives the small fitted-model dictionary before
    score preparation. `skip_gene_indices` skips checkpointed output only
    after refitting the complete selected cohort, preserving native fit-batch
    membership on resume. `telemetry` receives method, budget, timing, stage
    and replay provenance; each batch also carries a CPU-replay bitmap.
    """
    from ..glm.design import validate_design_matrix

    X = np.asarray(covariate_matrix, dtype=float)
    validate_design_matrix(X, n_cells=response_matrix.shape[1])
    gene_ids = list(gene_ids)
    names = list(grna_target_cells)
    groups = [np.asarray(v) for v in grna_target_cells.values()]
    if response_matrix.shape != (len(gene_ids), len(X)) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Gene identifiers and response matrix must align uniquely")
    if not groups or any(
        g.ndim != 1
        or not np.issubdtype(g.dtype, np.integer)
        or len(g) == 0
        or np.any(g < 0)
        or np.any(g >= len(X))
        or len(np.unique(g)) != len(g)
        for g in groups
    ):
        raise ValueError("Targets must contain distinct, nonempty, in-range cell indices")
    groups = [np.asarray(g, dtype=np.int64) for g in groups]
    if not 0 < X.shape[1] <= 16 or len(X) >= 2**31:
        raise ValueError("CUDA tiles require 1 to 16 covariates and int32 cell positions")
    if side not in ("both", "left", "right") or layout not in ("rectangle", "compact", "hybrid"):
        raise ValueError("Unknown side or Cartesian layout")
    if null_strategy not in ("prefix", "direct", "parallel_prefix"):
        raise ValueError("Unknown null strategy")
    if tail_backend not in ("cpu", "cuda"):
        raise ValueError("Unknown tail-summary backend")
    if fit_backend not in ("cpu", "cpu_fork", "cuda_native") or precompute_backend not in (
        "cpu",
        "cuda",
    ):
        raise ValueError("Unknown fitting or precomputation backend")
    if fit_backend == "cpu_fork" and (
        not sys.platform.startswith("linux") or "cupy" in sys.modules or "torch" in sys.modules
    ):
        raise ValueError("cpu_fork requires a fresh Linux process before importing a GPU runtime")
    if (
        any(
            not isinstance(v, int)
            for v in (B1, B2, B3, gene_tile_size, n_jobs, fit_batch_size, fit_max_bytes)
        )
        or B1 < 1
        or B2 < 1
        or B3 < 0
        or gene_tile_size < 1
        or n_jobs < 1
        or fit_batch_size < 1
        or fit_max_bytes < 1
    ):
        raise ValueError("Positive integer budgets, tile width and worker count are required")
    if eligible_pairs is not None and (
        eligible_pairs.dtype != np.bool_ or eligible_pairs.shape != (len(gene_ids), len(names))
    ):
        raise ValueError("Eligibility must be a boolean gene by target matrix")
    if fit_parametric_curve and B2 < 100:
        raise ValueError("The original tail quantile rule requires at least 100 second-stage draws")
    if gene_indices is not None and not np.issubdtype(np.asarray(gene_indices).dtype, np.integer):
        raise ValueError("Gene selection must contain integer row indices")
    rows = (
        np.arange(len(gene_ids), dtype=np.int64)
        if gene_indices is None
        else np.asarray(gene_indices)
    )
    if (
        rows.ndim != 1
        or len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= len(gene_ids))
    ):
        raise ValueError("Gene selection must contain distinct, in-range row indices")
    skipped = (
        np.array([], dtype=np.int64) if skip_gene_indices is None else np.asarray(skip_gene_indices)
    )
    if (
        skipped.ndim != 1
        or not np.issubdtype(skipped.dtype, np.integer)
        or not set(skipped).issubset(set(rows))
    ):
        raise ValueError("Checkpoint positions must be integer members of the selected family")
    rows = rows.astype(np.int32)
    stats = {} if telemetry is None else telemetry
    stats.update(
        stage_counts={1: 0, 2: 0, 3: 0},
        replay_pairs=0,
        emitted_pairs=0,
        tiles=0,
        null_slots={1: 0, 2: 0, 3: 0},
        layout=layout,
        null_strategy=null_strategy,
        tail_backend=tail_backend,
        fit_backend=fit_backend,
        precompute_backend=precompute_backend,
        precompute_cpu_replay_genes=0,
        precompute_cpu_replay_gene_ids=[],
        fit_batch_size=fit_batch_size,
        fit_max_bytes=fit_max_bytes,
    )
    if len(rows) == 0:
        return
    side_code = {"both": 0, "left": -1, "right": 1}[side]
    started = perf_counter()
    force_cpu = set()
    if fit_backend == "cuda_native":
        from ..test_statistic.native_resident import NativeEngine

        engine = NativeEngine(X, fit_max_bytes, stats)
        engine.host_budget = 0
        fits = engine.fit_genes(
            response_matrix, [gene_ids[g] for g in rows], rows.tolist(), fit_batch_size
        )
        force_cpu = set(engine.force_cpu)
        del engine
    else:
        # The explicit fork route is restricted to a fresh process before CUDA.
        fits = fit_all_genes(
            response_matrix,
            [gene_ids[g] for g in rows],
            X,
            gene_rows=rows.tolist(),
            batch_width=1,
            n_jobs=n_jobs,
            x_outer_flat_shared=x_outer_flat(X),
            backend="fork" if fit_backend == "cpu_fork" else "thread",
        )
    stats["fit_seconds"] = perf_counter() - started
    stats["fit_cpu_replay_genes"] = len(force_cpu)
    stats["fit_cpu_replay_gene_ids"] = sorted(force_cpu)
    if on_fit is not None:
        on_fit(fits)
    if len(skipped):
        rows = np.asarray([g for g in rows if g not in set(skipped)], dtype=np.int32)
        if not len(rows):
            return
    t = perf_counter()
    permutations = permutation_draws(
        len(X), max(map(len, groups)), B1 + B2 + B3, np.random.default_rng(seed)
    )
    from ..test_statistic.permutation_tiles import PermutationPool

    pool = PermutationPool(X, permutations, groups)
    cp = pool.cp
    cp.cuda.get_current_stream().synchronize()
    stats["pool_seconds"] = perf_counter() - t
    stats["permutation_shape"] = permutations.shape
    stats["pool_device_bytes"] = pool.device_bytes
    sizes = pool.target_sizes

    def prepare(row):
        y = _get_row(response_matrix, row)
        fit = fits[gene_ids[row]]
        pieces = compute_precomputation_pieces(y, X, fit.fitted_coefs, fit.theta)
        return y, pieces

    with ThreadPoolExecutor(max_workers=n_jobs) as workers:
        for start in range(0, len(rows), gene_tile_size):
            tile_rows = rows[start : start + gene_tile_size]
            if precompute_backend == "cpu":
                prepared = list(workers.map(prepare, tile_rows))
                ys = np.array([v[0] for v in prepared])
                pieces = [v[1] for v in prepared]
                planes = np.array([stack_pieces(q.a, q.w, q.D).T for q in pieces])
                mu = np.array([q.mu for q in pieces])
                tile = pool.tile(planes, ys, mu)
            else:
                ys = np.array([_get_row(response_matrix, row) for row in tile_rows])
                coefficients = np.array([fits[gene_ids[row]].fitted_coefs for row in tile_rows])
                theta = np.array([fits[gene_ids[row]].theta for row in tile_rows])
                prepared = planes = mu = None
                pieces = [None] * len(tile_rows)
                try:
                    tile, invalid = pool.tile_from_models(ys, coefficients, theta)
                except (ValueError, RuntimeError, MemoryError):
                    invalid = np.ones(len(tile_rows), dtype=bool)
                    tile = None
                invalid |= np.array([gene_ids[row] in force_cpu for row in tile_rows])
                if tile is None:
                    prepared = list(workers.map(prepare, tile_rows))
                    pieces = [v[1] for v in prepared]
                    planes = np.array([stack_pieces(q.a, q.w, q.D).T for q in pieces])
                    mu = np.array([q.mu for q in pieces])
                    tile = pool.tile(planes, ys, mu)
                else:
                    for g in np.flatnonzero(invalid):
                        _, q = prepare(tile_rows[g])
                        pieces[g] = q
                        tile.planes[g] = cp.asarray(stack_pieces(q.a, q.w, q.D).T)
                        tile.mu[g] = cp.asarray(q.mu)
                stats["precompute_cpu_replay_genes"] += int(invalid.sum())
                stats["precompute_cpu_replay_gene_ids"].extend(
                    gene_ids[tile_rows[g]] for g in np.flatnonzero(invalid)
                )
            if eligible_pairs is None:
                treated, control = nonzero_counts(response_matrix[tile_rows], groups, len(X))
                eligible = (treated >= 7) & (control >= 7)
            else:
                eligible = np.asarray(eligible_pairs[tile_rows])
            pg, pt = np.nonzero(eligible)
            pg, pt = pg.astype(np.int32), pt.astype(np.int32)
            if layout == "compact":
                zdev, fcdev, sedev, obs_guard = tile.observed(pg, pt)
            else:
                all_g = np.repeat(np.arange(len(tile_rows), dtype=np.int32), len(names))
                all_t = np.tile(np.arange(len(names), dtype=np.int32), len(tile_rows))
                values = tile.observed(all_g, all_t)
                keep = cp.asarray(eligible.ravel())
                zdev, fcdev, sedev, obs_guard = (v[keep] for v in values)
                del values, keep
            z, fc, se = zdev.get(), fcdev.get(), sedev.get()
            stage = np.where(np.isfinite(z), 1, 0).astype(np.int8)
            replay = (~np.isfinite(z)) | obs_guard.get().astype(bool)

            def null_batch(query, lo, hi, number, pg=pg, pt=pt, tile_rows=tile_rows, tile=tile):
                requested = _requested_sizes(
                    pg[query],
                    pt[query],
                    sizes,
                    len(tile_rows),
                    layout == "rectangle" or (number == 1 and layout == "hybrid"),
                )
                nulls, mapping, _, _, unsafe = tile.nulls(requested, lo, hi, strategy=null_strategy)
                slots = mapping[pg[query], sizes[pt[query]]]
                stats["null_slots"][number] += len(nulls)
                if np.any(slots < 0):
                    raise RuntimeError("Requested eligible pair has no null distribution")
                return nulls, slots, unsafe

            first, slots, unsafe = null_batch(np.arange(len(pg)), 0, B1, 1)
            pdev, near = tile.empirical(first, slots, zdev, unsafe, side_code)
            p = pdev.get()
            replay |= near.get().astype(bool)
            del first, pdev, near, unsafe
            active = np.flatnonzero((p <= 0.02) & np.isfinite(z))
            second = None
            second_slots = None
            second_guard = None
            if len(active) and fit_parametric_curve:
                second, second_slots, second_guard = null_batch(active, B1, B1 + B2, 2)
                replay[active] |= second_guard.get()[second_slots].astype(bool)
                if tail_backend == "cuda":
                    tail_p, _, guarded = shared_skew_normal_cuda(
                        second, second_slots, z[active], side_code
                    )
                else:
                    tail_p, _, guarded = shared_skew_normal(
                        second.get(), second_slots, z[active], side_code
                    )
                replay[active] |= guarded
                p[active] = tail_p
                stage[active] = 2
                rejected = tail_p <= -0.5
                # NaN probabilities also fail SkewNormalEvalResult.used.
                rejected |= ~np.isfinite(tail_p)
                fallback = active[rejected]
            else:
                fallback = active
            if len(fallback):
                if B3:
                    second = None
                    final, final_slots, final_guard = null_batch(fallback, B1 + B2, B1 + B2 + B3, 3)
                elif fit_parametric_curve:
                    final = second
                    final_guard = second_guard
                    final_slots = second_slots[np.flatnonzero(rejected)]
                else:
                    final, final_slots, final_guard = null_batch(fallback, 0, B1, 3)
                values, near = tile.empirical(
                    final, final_slots, zdev[cp.asarray(fallback)], final_guard, side_code
                )
                p[fallback] = values.get()
                replay[fallback] |= near.get().astype(bool)
                stage[fallback] = 3
                del final, values, near, final_guard
            # Replay close comparisons and tail-decision boundaries with the CPU owner.
            replay_started = perf_counter()
            prefixes = {}
            for j in np.flatnonzero(replay):
                g, target = int(pg[j]), int(pt[j])
                q = pieces[g]
                if q is None:
                    _, q = prepare(tile_rows[g])
                    pieces[g] = q
                stacked = stack_pieces(q.a, q.w, q.D)
                if g not in prefixes:
                    demand = int(sizes[pt[pg == g]].sum())
                    prefixes[g] = (
                        PermutationPrefixSums(
                            stacked,
                            permutations[:, : int(sizes[pt[(pg == g) & replay]].max())],
                        )
                        if prefix_scan_pays(demand, pool.width)
                        else None
                    )
                prefix = prefixes[g]

                def provider(lo, hi, prefix=prefix, target=target, channels=stacked.shape[1]):
                    # Preserve the owner's full-width prefix/CSR budget decision.
                    if prefix is None or (hi - lo) * pool.width * channels * 8 > prefix._max_bytes:
                        return None
                    return prefix.statistics(lo, hi, int(sizes[target]))

                result = run_low_level_test_full(
                    ys[g],
                    q.mu,
                    q.a,
                    q.w,
                    q.D,
                    groups[target],
                    PermutationSliceDraws(permutations, int(sizes[target]), len(X)),
                    stacked=stacked,
                    B1=B1,
                    B2=B2,
                    B3=B3,
                    fit_parametric_curve=fit_parametric_curve,
                    side_code=side_code,
                    null_statistics_fn=provider,
                )
                p[j], z[j], fc[j], se[j] = (
                    result.p_value,
                    result.z_orig,
                    result.fold_change,
                    result.se_fold_change,
                )
                stage[j] = 0 if not np.isfinite(result.stage) else result.stage
            stats["replay_pairs"] += int(replay.sum())
            stats["replay_seconds"] = (
                stats.get("replay_seconds", 0.0) + perf_counter() - replay_started
            )
            for number in (1, 2, 3):
                stats["stage_counts"][number] += int((stage == number).sum())
            stats["emitted_pairs"] += len(pg)
            stats["tiles"] += 1
            stats["pool_device_bytes"] = pool.device_bytes
            result = _frame(gene_ids, names, tile_rows, pg, pt, p, z, fc, se, stage)
            result.attrs["gene_rows"] = tile_rows.tolist()
            result.attrs["cpu_replay"] = replay
            del null_batch
            # Release stage arrays before yielding; retained allocator blocks are reusable.
            del tile, prepared, pieces, planes, ys, mu, zdev, fcdev, sedev, second, prefixes
            cp.cuda.get_current_stream().synchronize()
            yield result
