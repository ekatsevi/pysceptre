"""Measure unchanged CPU and CUDA binomial IRLS on actual frozen target groups."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pysceptre.glm.cuda import CudaTargetDesign
from pysceptre.glm.irls import fit_binomial_glm_batch, x_outer_flat

ap = argparse.ArgumentParser()
ap.add_argument("inputs", type=Path)
ap.add_argument("out", type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
meta = json.loads((a.inputs / "INPUT-IDENTITY.json").read_text())
X = np.fromfile(a.inputs / "design-f64le.bin", dtype="<f8").reshape(
    (meta["analysis_cells"], meta["n_covariates"]), order="F"
)
targets = json.loads((a.inputs / "target-cells.json").read_text())
pairs = __import__("pandas").read_csv(a.inputs / "crt-discovery-pairs.csv")
ids = [k for k in targets if k in set(pairs.grna_target)][:16]
Y = np.zeros((len(ids), len(X)))
for j, k in enumerate(ids):
    Y[j, np.asarray(targets[k], dtype=np.int64)] = 1
xo = x_outer_flat(X)
cuda = CudaTargetDesign(X, fallback=False)
rows = []
cpu = None
gpu = None
for rep, order in enumerate([["cpu", "gpu"], ["gpu", "cpu"], ["cpu", "gpu"]]):
    for mode in order:
        t = time.perf_counter()
        if mode == "cpu":
            cpu = fit_binomial_glm_batch(X, Y, X_outer_flat=xo)
        else:
            gpu = cuda.fit(Y)
        rows.append(
            dict(
                rep=rep,
                mode=mode,
                seconds=time.perf_counter() - t,
                detail=cuda.last_stats if mode == "gpu" else None,
            )
        )
assert cpu is not None and gpu is not None
metrics = {}
for field in ["coefs", "fitted_values", "deviance"]:
    av, bv = getattr(cpu, field), getattr(gpu, field)
    metrics[field] = dict(
        max_abs_difference=float(np.max(np.abs(av - bv))),
        max_relative_difference=float(np.max(np.abs(av - bv) / np.maximum(np.abs(av), 1e-15))),
    )
metrics["iteration_changes"] = int(np.count_nonzero(cpu.n_iter != gpu.n_iter))
metrics["convergence_changes"] = int(np.count_nonzero(cpu.converged != gpu.converged))
assert metrics["fitted_values"]["max_abs_difference"] <= 1e-7
assert metrics["convergence_changes"] == 0
receipt = dict(
    timings=rows,
    metrics=metrics,
    targets=ids,
    cells=len(X),
    columns=X.shape[1],
    gpu_calls=cuda.gpu_calls,
    fallback_calls=cuda.fallback_calls,
    scope="Component diagnostic only. No RNG or draws, p-values, correction or output export in timed fit call.",
)
(a.out / "fit-diagnostic.json").write_text(json.dumps(receipt, indent=2) + "\n")
