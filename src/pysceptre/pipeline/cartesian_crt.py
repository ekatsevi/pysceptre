"""Explicit CRT rectangles with lossless resident draws and compact gene planes."""

import copy
import sys
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import pandas as pd

from ..glm.irls import x_outer_flat
from ..precompute.pieces import compute_precomputation_pieces
from ..test_statistic.resampling import run_low_level_test_full
from ..test_statistic.score_stat import ListDraws, stack_pieces
from ..test_statistic.shared_tail import shared_skew_normal, shared_skew_normal_cuda
from .cartesian import _frame, _locked_iteration
from .discovery import (
    _get_row,
    chunk_layout_hash,
    fit_all_genes,
    fit_all_targets,
    reference_fit_chunks,
)
from .pairwise_qc import nonzero_counts


class _DeferredSample:
    def __init__(self, probabilities, draws, state, treated):
        self.probabilities = probabilities
        self.draws = draws
        self.state = state
        self.treated = treated


class _DeferredSampler:
    """Keep CPU fitting workers free of CUDA contexts and stream ownership."""

    def sample(self, probabilities, draws, rng, treated):
        return _DeferredSample(
            probabilities, draws, copy.deepcopy(rng.bit_generator.state), treated
        )


@_locked_iteration
def iter_cartesian_crt_analysis(
    response_matrix,
    gene_ids,
    covariate_matrix,
    grna_target_cells,
    *,
    eligible_pairs=None,
    gene_indices=None,
    skip_gene_indices=None,
    seed=None,
    B1=499,
    B2=4999,
    B3=0,
    fit_parametric_curve=True,
    side="both",
    gene_tile_size=32,
    n_jobs=1,
    fit_backend="cpu",
    precompute_backend="cpu",
    tail_backend="cpu",
    target_chunk_size=200,
    chunk_memory_gb=1.0,
    device_max_bytes=150 * 1024**3,
    packed_draws=True,
    telemetry=None,
    on_fit=None,
    on_bank=None,
):
    """Yield the original adjusted CRT using resident, losslessly packed listings.

    This experimental union/complement route retains CPU width-one gene fits,
    reference binomial fit cohorts, target-name seeds, and the original exact
    or approximate sampler choice. All B1+B2+B3 rows are generated together.
    Approximate listings retain their order and multiplicities. Native sampling
    declines to the original CPU sampler under its existing numerical guards.

    Every supplied target belongs to the reference fit cohort even when only
    a subset of genes is emitted. Pass the complete intended target family in
    its original order. `target_chunk_size` and `chunk_memory_gb` must match
    the intended reference run; changing them can change binomial last bits.

    Compact a,w planes and inverse information matrices are retained for all
    selected genes. Each output tile reuses every resident bank. First-stage
    counts are fused, and later distributions are materialized only for active
    pairs, one target at a time. The caller must apply correction to the whole
    family after collecting all output. Checkpoint skipping happens after
    refitting the complete selected gene cohort.

    `device_max_bytes` bounds persistent banks and score planes, not CUDA
    runtime or transient workspaces. Insufficient capacity raises. Explicit
    CUDA precomputation/tail summaries reorder FP64 arithmetic. Ambiguous
    variances, comparisons and tail choices replay the complete original CPU
    pair, reading the identical decoded listings. `on_bank`, if supplied,
    receives each target name, fitted probabilities and packed bank for audit.
    """
    from ..glm.design import validate_design_matrix

    X = np.asarray(covariate_matrix, dtype=float)
    validate_design_matrix(X, n_cells=response_matrix.shape[1])
    gene_ids, names = list(gene_ids), list(grna_target_cells)
    groups = [np.asarray(v) for v in grna_target_cells.values()]
    if response_matrix.shape != (len(gene_ids), len(X)) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Unique gene identifiers must align with the response matrix")
    if not groups or any(
        g.ndim != 1
        or not np.issubdtype(g.dtype, np.integer)
        or not len(g)
        or np.any(g < 0)
        or np.any(g >= len(X))
        or len(np.unique(g)) != len(g)
        for g in groups
    ):
        raise ValueError("Distinct nonempty in-range treatment cell positions are required")
    if not 0 < X.shape[1] <= 16 or len(X) >= 2**31:
        raise ValueError("CRT tiles require 1 to 16 covariates and int32 cell positions")
    if side not in ("both", "left", "right"):
        raise ValueError("Unknown side")
    if fit_backend not in ("cpu", "cpu_fork") or any(
        v not in ("cpu", "cuda") for v in (precompute_backend, tail_backend)
    ):
        raise ValueError("Unknown model, precomputation or tail backend")
    if fit_backend == "cpu_fork" and (
        not sys.platform.startswith("linux") or "cupy" in sys.modules or "torch" in sys.modules
    ):
        raise ValueError("cpu_fork requires a fresh Linux process before importing a GPU runtime")
    if (
        any(
            not isinstance(v, int)
            for v in (B1, B2, B3, gene_tile_size, n_jobs, target_chunk_size, device_max_bytes)
        )
        or min(B1, B2, gene_tile_size, n_jobs, target_chunk_size, device_max_bytes) < 1
        or B3 < 0
    ):
        raise ValueError("Positive integer budgets, tile width and worker count are required")
    if fit_parametric_curve and B2 < 100:
        raise ValueError("The original tail quantile rule requires at least 100 second-stage draws")
    if not np.isfinite(chunk_memory_gb) or chunk_memory_gb <= 0:
        raise ValueError("A positive finite reference chunk budget is required")
    if eligible_pairs is not None and (
        eligible_pairs.dtype != np.bool_ or eligible_pairs.shape != (len(gene_ids), len(names))
    ):
        raise ValueError("Eligibility must be a boolean gene by target matrix")
    rows = np.arange(len(gene_ids)) if gene_indices is None else np.asarray(gene_indices)
    if (
        rows.ndim != 1
        or not np.issubdtype(rows.dtype, np.integer)
        or len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= len(gene_ids))
    ):
        raise ValueError("Distinct in-range integer gene positions are required")
    skipped = (
        np.array([], dtype=int) if skip_gene_indices is None else np.asarray(skip_gene_indices)
    )
    if (
        skipped.ndim != 1
        or not np.issubdtype(skipped.dtype, np.integer)
        or not set(skipped).issubset(set(rows))
    ):
        raise ValueError("Checkpoint positions must be integer members of the selected family")
    if not len(rows):
        return
    stats = {} if telemetry is None else telemetry
    stats.update(
        method="crt_packed_coalesced",
        B1=B1,
        B2=B2,
        B3=B3,
        packed_draws=bool(packed_draws),
        stage_counts={1: 0, 2: 0, 3: 0},
        replay_pairs=0,
        emitted_pairs=0,
        tiles=0,
    )
    start = perf_counter()
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
    stats["fit_seconds"] = perf_counter() - start
    if on_fit is not None:
        on_fit(fits)
    if len(set(skipped)) == len(rows):
        return
    chunks = reference_fit_chunks(
        grna_target_cells,
        pd.DataFrame({"grna_target": names}),
        len(X),
        B_total=B1 + B2 + B3,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
    )
    stats["target_fit_chunk_hash"] = chunk_layout_hash(chunks)
    stats["target_fit_chunk_sizes"] = list(map(len, chunks))
    import cupy as cp

    from ..test_statistic.crt_tiles import CompactScoreAtlas, PackedDrawCollection, PackedDraws
    from ..test_statistic.native_resident import DeviceDraws, NativeEngine
    from ..test_statistic.permutation_tiles import PermutationPool

    # The observed-statistic kernel needs membership/design, but no permutation rows.
    pool = PermutationPool(X, np.empty((0, max(map(len, groups))), dtype=np.int32), groups)
    atlas_bytes = ((len(rows) + 31) // 32) * 32 * 2 * len(X) * 8
    estimated_banks = int(
        (B1 + B2 + B3)
        * sum(map(len, groups))
        * (max(1, (len(X) - 1).bit_length()) if packed_draws else 32)
        / 8
    )
    if atlas_bytes + estimated_banks > device_max_bytes:
        raise MemoryError("Estimated persistent CRT atlas and draw banks exceed device_max_bytes")
    if atlas_bytes + estimated_banks + 2 * 1024**3 > cp.cuda.runtime.memGetInfo()[0]:
        raise MemoryError("Insufficient free GPU memory for the resident CRT family")
    atlas = CompactScoreAtlas(X, np.zeros((len(rows), X.shape[1], X.shape[1])))
    z = np.full((len(rows), len(names)), np.nan)
    fc, se = np.zeros_like(z), np.zeros_like(z)
    replay = np.zeros_like(z, dtype=bool)
    mask = np.empty_like(replay)

    def prepare(row):
        y = _get_row(response_matrix, row)
        f = fits[gene_ids[row]]
        return y, compute_precomputation_pieces(y, X, f.fitted_coefs, f.theta)

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=n_jobs) as workers:
        for lo in range(0, len(rows), gene_tile_size):
            rr = rows[lo : lo + gene_tile_size]
            if precompute_backend == "cpu":
                prepared = list(workers.map(prepare, rr))
                ys = np.array([v[0] for v in prepared])
                pieces = [v[1] for v in prepared]
                tile = pool.tile(
                    np.array([stack_pieces(q.a, q.w, q.D).T for q in pieces]),
                    ys,
                    np.array([q.mu for q in pieces]),
                )
                invalid = np.array(
                    [
                        q.min_eigenvalue <= q.max_eigenvalue * X.shape[1] * np.finfo(float).eps
                        for q in pieces
                    ]
                )
            else:
                ys = np.array([_get_row(response_matrix, row) for row in rr])
                tile, invalid = pool.tile_from_models(
                    ys,
                    np.array([fits[gene_ids[r]].fitted_coefs for r in rr]),
                    np.array([fits[gene_ids[r]].theta for r in rr]),
                )
            local = np.arange(lo, lo + len(rr))
            atlas.set_genes(local, tile.planes[:, 0], tile.planes[:, 1])
            wX = tile.planes[:, 1, :, None] * pool.design[None, :, :]
            H = cp.matmul(pool.design.T[None, :, :], wX)
            # Singular genes are completely replayed; their dummy inverse is never accepted.
            for j in np.flatnonzero(invalid):
                H[j] = cp.eye(X.shape[1])
            atlas.inverse[lo : lo + len(rr)] = cp.linalg.inv(H)
            if eligible_pairs is None:
                treated, control = nonzero_counts(response_matrix[rr], groups, len(X))
                mask[lo : lo + len(rr)] = (treated >= 7) & (control >= 7)
            else:
                mask[lo : lo + len(rr)] = eligible_pairs[rr]
            pg = np.repeat(np.arange(len(rr), dtype=np.int32), len(names))
            pt = np.tile(np.arange(len(names), dtype=np.int32), len(rr))
            values = tile.observed(pg, pt)
            z[lo : lo + len(rr)], fc[lo : lo + len(rr)], se[lo : lo + len(rr)] = [
                v.get().reshape(len(rr), len(names)) for v in values[:3]
            ]
            replay[lo : lo + len(rr)] = (
                values[3].get().reshape(len(rr), len(names)).astype(bool) | invalid[:, None]
            )
            del tile, ys, wX, H, values
    stats["precompute_seconds"] = perf_counter() - start
    stats["atlas_bytes"] = atlas.nbytes

    def fit_chunk(chunk):
        return fit_all_targets(
            {name: grna_target_cells[name] for name in chunk},
            X,
            B1=B1,
            B2=B2,
            B3=B3,
            seed=seed,
            n_jobs=1,
            x_outer_flat_shared=x_outer_flat(X),
            native_engine=_DeferredSampler(),
        )

    start = perf_counter()
    engine = NativeEngine(X, min(4 * 1024**3, device_max_bytes), {})
    banks = []
    bank_bytes = 0
    # Consume reference cohorts in order; workers only execute owning CPU fits.
    with ThreadPoolExecutor(max_workers=n_jobs) as workers:
        for targets in workers.map(fit_chunk, chunks):
            for name, target in targets.items():
                deferred = target.draws
                rng = np.random.default_rng()
                rng.bit_generator.state = deferred.state
                drawn = engine.sample(deferred.probabilities, deferred.draws, rng, deferred.treated)
                if isinstance(drawn, DeviceDraws):
                    cells, offsets = drawn.cells, drawn.offsets
                else:
                    listings = drawn._index_arrays(0, B1 + B2 + B3)
                    offsets = np.r_[0, np.cumsum(list(map(len, listings)))]
                    cells = (
                        np.concatenate(listings) if offsets[-1] else np.array([], dtype=np.int32)
                    )
                bank = PackedDraws(cells, offsets, len(X), packed=packed_draws)
                bank_bytes += bank.nbytes
                if bank_bytes + atlas.nbytes > device_max_bytes:
                    raise MemoryError(
                        "Actual persistent CRT bank/atlas size exceeds device_max_bytes"
                    )
                if on_bank is not None:
                    on_bank(name, deferred.probabilities, bank)
                banks.append(bank)
                del drawn, cells, offsets
    collection = PackedDrawCollection(banks)
    stats["bank_seconds"] = perf_counter() - start
    stats["bank_bytes"] = collection.nbytes
    stats["sampling"] = dict(engine.stats)
    del engine, targets
    side_code = {"both": 0, "left": -1, "right": 1}[side]
    skip = set(skipped)
    for lo in range(0, len(rows), gene_tile_size):
        selected = np.array(
            [j for j in range(lo, min(lo + gene_tile_size, len(rows))) if rows[j] not in skip]
        )
        if not len(selected):
            continue
        # Explicit tile scheduling bounds the reusable gene working set.
        zz = z[selected].copy()
        ff, ss, mm = fc[selected].copy(), se[selected].copy(), mask[selected]
        needs_replay = replay[selected].copy() | ~np.isfinite(zz)
        pdev, guard = collection.empirical(
            atlas, zz.T, 0, B1, selected=selected, eligible=mm.T, side=side_code
        )
        p = pdev.get().T
        needs_replay |= guard.get().T.astype(bool)
        stage = np.where(np.isfinite(zz), 1, 0).astype(np.int8)
        ag, at = np.nonzero(mm & np.isfinite(zz) & (p <= 0.02))
        # One bounded batch across active target/gene pairs, not one Python
        # tail invocation per target. Each pair still has its own distribution.
        batch_pairs = max(1, (512 * 1024**2) // (max(B1, B2, B3) * 8))
        for first in range(0, len(ag), batch_pairs):
            gg, tt = ag[first : first + batch_pairs], at[first : first + batch_pairs]
            second = unsafe = None
            if fit_parametric_curve:
                second, unsafe = collection.statistics_queries(atlas, selected[gg], tt, B1, B1 + B2)
                slots = np.arange(len(gg))
                fn = shared_skew_normal_cuda if tail_backend == "cuda" else shared_skew_normal
                tp, _, danger = fn(
                    second if tail_backend == "cuda" else second.get(), slots, zz[gg, tt], side_code
                )
                p[gg, tt] = tp
                stage[gg, tt] = 2
                needs_replay[gg, tt] |= unsafe.get().astype(bool) | danger
                failed = np.flatnonzero((tp <= -0.5) | ~np.isfinite(tp))
            else:
                failed = np.arange(len(gg))
            if len(failed):
                fg, ft = gg[failed], tt[failed]
                if B3:
                    final, final_guard = collection.statistics_queries(
                        atlas, selected[fg], ft, B1 + B2, B1 + B2 + B3
                    )
                    final_slots = np.arange(len(fg))
                elif fit_parametric_curve:
                    final, final_guard, final_slots = second, unsafe, failed
                else:
                    final, final_guard = collection.statistics_queries(
                        atlas, selected[fg], ft, 0, B1
                    )
                    final_slots = np.arange(len(fg))
                pp, danger = pool.empirical(
                    final, final_slots, cp.asarray(zz[fg, ft]), final_guard, side_code
                )
                p[fg, ft] = pp.get()
                needs_replay[fg, ft] |= danger.get().astype(bool)
                stage[fg, ft] = 3
        pg, pt = np.nonzero(mm)
        # Reconstruct identical listings only for targets with ambiguous pairs.
        for t in np.unique(pt[needs_replay[pg, pt]]):
            bank = banks[t]
            cells, offsets = bank.unpack().get(), bank.offsets.get()
            draws = ListDraws(
                [cells[offsets[j] : offsets[j + 1]] for j in range(bank.n_draws)], len(X)
            )
            for g in np.flatnonzero(mm[:, t] & needs_replay[:, t]):
                row = rows[selected[g]]
                y, q = prepare(row)
                result = run_low_level_test_full(
                    y,
                    q.mu,
                    q.a,
                    q.w,
                    q.D,
                    groups[t],
                    draws,
                    B1=B1,
                    B2=B2,
                    B3=B3,
                    fit_parametric_curve=fit_parametric_curve,
                    side_code=side_code,
                )
                p[g, t], zz[g, t], ff[g, t], ss[g, t] = (
                    result.p_value,
                    result.z_orig,
                    result.fold_change,
                    result.se_fold_change,
                )
                stage[g, t] = 0 if not np.isfinite(result.stage) else result.stage
        stats["replay_pairs"] += int(needs_replay[pg, pt].sum())
        stats["emitted_pairs"] += len(pg)
        stats["tiles"] += 1
        for number in (1, 2, 3):
            stats["stage_counts"][number] += int((stage[pg, pt] == number).sum())
        frame = _frame(
            gene_ids,
            names,
            rows[selected],
            pg,
            pt,
            p[pg, pt],
            zz[pg, pt],
            ff[pg, pt],
            ss[pg, pt],
            stage[pg, pt],
        )
        frame.attrs["gene_rows"] = rows[selected].tolist()
        frame.attrs["cpu_replay"] = needs_replay[pg, pt]
        cp.cuda.get_current_stream().synchronize()
        yield frame
