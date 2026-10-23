"""Fresh CPU/GPU measurements of the report's exact frozen Gasperini family."""

import argparse
import json
import os
import random
import resource
import sys
import time
import warnings
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("solver")
    ap.add_argument(
        "--score-backend", choices=["cpu", "cuda_resident", "cuda_native"], default="cpu"
    )
    ap.add_argument("--score-batch-size", type=int, default=32)
    ap.add_argument("--score-max-bytes", type=int, default=16 * 1024**3)
    ap.add_argument("--storage", choices=["ram", "backed"], default="backed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Pin all process threads before importing the numerical runtime.
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[0]})
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from threadpoolctl import threadpool_info, threadpool_limits

    import pysceptre as ps
    from pysceptre.pipeline.calibration import negative_control_pairs_from_names
    from pysceptre.pipeline.pairwise_qc import nonzero_counts

    sys.path.insert(0, str(args.root))
    from adapter import inputs

    threadpool_limits(1)
    imports = time.perf_counter() - started

    def settings(solver):
        options = dict(
            side="both",
            moi="high",
            control_group="complement",
            resampling_approximation="skew_normal",
            multiple_testing_alpha=0.1,
            seed=271828,
            target_chunk_size=200,
            chunk_memory_gb=1.0,
            n_jobs=1,
            resampling_mechanism="crt",
            glm_solver=solver,
        )
        if args.score_backend != "cpu":
            options.update(
                score_backend=args.score_backend,
                score_batch_size=args.score_batch_size,
                score_max_bytes=args.score_max_bytes,
            )
        return options

    compile_seconds = 0.0  # No benchmark warmup in clean cold-process runs.
    assert (args.root / "response-view.h5").exists(), (
        "Never create anything in the frozen baseline directory"
    )
    t = time.perf_counter()
    Y, genes, X, targets, nt, meta, base = inputs(args.root, args.storage)
    load_seconds = time.perf_counter() - t
    np.random.seed(271828)
    random.seed(271828)
    state_before = np.random.get_state()
    random_before = random.getstate()
    fit_records, draw_records = {}, {}
    clocks = []
    postprocess = []
    messages = []
    for stage in ["calibration", "power", "discovery"]:
        pairs = pd.read_csv(args.root / "inputs" / f"crt-{stage}-pairs.csv")
        fit_records.clear()
        draw_records.clear()
        t = time.perf_counter()
        prelude = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if stage == "calibration":
                result = ps.run_calibration_check(
                    Y,
                    genes,
                    X,
                    nt,
                    n_calibration_pairs=len(pairs),
                    calibration_group_size=2,
                    negative_control_pairs=pairs[["response_id", "grna_target"]],
                    **settings(args.solver),
                )
            elif stage == "power":
                result = ps.run_power_check(
                    Y,
                    genes,
                    X,
                    targets,
                    positive_control_pairs=pairs[["response_id", "grna_target"]],
                    **settings(args.solver),
                )
            else:
                result = ps.run_discovery_analysis(
                    Y,
                    genes,
                    X,
                    targets,
                    pairs[["response_id", "grna_target"]],
                    **settings(args.solver),
                )
        elapsed = time.perf_counter() - t
        clocks.append(dict(stage=stage, seconds=elapsed, calibration_prelude_seconds=prelude))
        messages.extend(
            dict(stage=stage, category=w.category.__name__, message=str(w.message)) for w in caught
        )
        assert len(result) == len(pairs)
        assert set(zip(result.response_id, result.grna_target, strict=True)) == set(
            zip(pairs.response_id, pairs.grna_target, strict=True)
        )
        post_started = time.perf_counter()
        from statsmodels.stats.multitest import multipletests

        pvalues = result["p_value"].to_numpy()
        finite = np.isfinite(pvalues)
        q = np.full(len(result), np.nan)
        if finite.any():
            q[finite] = multipletests(pvalues[finite], alpha=0.1, method="fdr_bh")[1]
        correction_seconds = time.perf_counter() - post_started
        result = result.copy()
        result["BH_q"] = q
        result["BH_discovery"] = q <= 0.1
        export_started = time.perf_counter()
        result.to_pickle(args.out / (stage + ".pkl"))
        (args.out / (stage + "-fits.json")).write_text(
            json.dumps(result.attrs.get("resident_fits", fit_records), indent=2) + "\n"
        )
        (args.out / (stage + "-draws.json")).write_text(json.dumps(draw_records, indent=2) + "\n")
        postprocess.append(
            dict(
                stage=stage,
                correction_seconds=correction_seconds,
                export_seconds=time.perf_counter() - export_started,
            )
        )
        if args.score_backend != "cpu":
            execution = result.attrs.get("score_execution")
            assert execution is not None
            assert execution["fallback_batches"] == 0, execution
            if args.score_backend in ("cuda_resident", "cuda_native"):
                assert execution["gpu_matmuls"] > 0, execution
                assert execution["response_gpu_genes"] > 0, execution
                assert execution["effective"] == args.score_backend, execution
            clocks[-1]["score_execution"] = execution
        print(stage, elapsed, flush=True)
    if base:
        base.close()
    state_after = np.random.get_state()
    unchanged = (
        state_before[0] == state_after[0]
        and np.array_equal(state_before[1], state_after[1])
        and state_before[2:] == state_after[2:]
        and random_before == random.getstate()
    )
    import platform

    receipt = dict(
        solver=args.solver,
        score_backend=args.score_backend,
        score_batch_size=args.score_batch_size,
        storage=args.storage,
        score_max_bytes=args.score_max_bytes,
        clocks=clocks,
        correction_export=postprocess,
        total_inference_seconds=sum(c["seconds"] for c in clocks),
        imports_seconds=imports,
        tiny_compile_seconds=compile_seconds,
        input_load_hash_seconds=load_seconds,
        whole_child_seconds=time.perf_counter() - started,
        host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        warnings=messages,
        legacy_global_rng_unchanged=unchanged,
        input_identity=meta,
        python=sys.version,
        versions={
            m: __import__(m).__version__ for m in ["numpy", "scipy", "pandas", "numba", "h5py"]
        },
        blas=threadpool_info(),
        affinity=sorted(os.sched_getaffinity(0)),
        host=platform.node(),
        job=os.environ.get("SLURM_JOB_ID"),
        scope="Clean public workflow: no sampler/fit hooks, audit prelude, or tiny-fixture warmup. Product input identity and numerical/runtime guards retained. Exact original 201158-cell CRT calibration/power/discovery families; import/hash/materialization/tiny compile outside inference; correction/export outside inference",
    )
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
