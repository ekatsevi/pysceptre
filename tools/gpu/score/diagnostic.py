"""Separate work reduction, resident device work and transfer-inclusive score costs."""

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from pysceptre.test_statistic.score_batch import ScoreBatch
from pysceptre.test_statistic.score_stat import compute_null_statistics_from_draws, draws_to_matrix

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--profile-only", action="store_true")
args = ap.parse_args()
threadpool_limits(1)

rng = np.random.default_rng(61)
n = 20000
B = 499
p = 4
bank = draws_to_matrix([np.sort(rng.choice(n, 400, replace=False)) for _ in range(B)], n)
rows = []
widths = [8] if args.profile_only else [1, 4, 8, 32]
for genes in widths:
    stacks = [
        np.c_[rng.normal(size=n), np.full(n, 3.0), rng.normal(scale=0.001, size=(n, p))]
        for _ in range(genes)
    ]
    flat = np.ascontiguousarray(np.concatenate(stacks, axis=1))
    results = {}
    timings = {}
    details = {}
    for mode in ["cpu_loop", "cpu_batch", "cuda_batch"]:
        telemetry = dict(
            uploaded_score_bytes=0,
            uploaded_draw_bytes=0,
            downloaded_score_bytes=0,
            score_matmuls=0,
            gpu_matmuls=0,
            score_columns=0,
            peak_pool_reserved_bytes=0,
        )
        t = time.perf_counter()
        batch = (
            None
            if mode == "cpu_loop"
            else ScoreBatch(flat, p + 2, mode, max_bytes=1024**3, telemetry=telemetry)
        )
        setup = time.perf_counter() - t

        def compute(mode=mode, batch=batch, stacks=stacks, genes=genes):
            if mode == "cpu_loop":
                return np.array([compute_null_statistics_from_draws(s, bank) for s in stacks])
            return batch.statistics(bank, list(range(genes)), cache_key="same-bank")

        t = time.perf_counter()
        got = compute()
        first = time.perf_counter() - t
        times = []
        if args.profile_only and mode == "cuda_batch":
            cp.cuda.profiler.start()
        for _ in range(5):
            t = time.perf_counter()
            got = compute()
            times.append(time.perf_counter() - t)
        if args.profile_only and mode == "cuda_batch":
            cp.cuda.profiler.stop()
        results[mode] = got
        timings[mode] = times
        details[mode] = dict(setup_seconds=setup, first_seconds=first, telemetry=telemetry)
    for mode in ["cpu_batch", "cuda_batch"]:
        np.testing.assert_allclose(
            results[mode], results["cpu_loop"], rtol=1e-10, atol=1e-11, equal_nan=True
        )
    differences = {
        mode: float(np.max(abs(value - results["cpu_loop"]))) for mode, value in results.items()
    }
    rows.append(
        dict(
            genes=genes,
            cells=n,
            B=B,
            nnz=int(bank.nnz),
            timings=timings,
            details=details,
            max_abs_difference=differences,
        )
    )
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(
    json.dumps(
        dict(
            rows=rows,
            cupy=cp.__version__,
            device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            scope="Synthetic finite adjusted-score pieces. Resident repetitions include the result download; setup and first use separate. This does not time fitting or a complete workflow.",
        ),
        indent=2,
    )
    + "\n"
)
