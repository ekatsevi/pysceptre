"""Compare actual seeded CPU/GPU CRT draw banks for frozen target groups."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pysceptre.glm.cuda import CudaTargetDesign
from pysceptre.glm.irls import x_outer_flat
from pysceptre.pipeline.discovery import fit_all_targets

ap = argparse.ArgumentParser()
ap.add_argument("inputs", type=Path)
ap.add_argument("out", type=Path)
args = ap.parse_args()
meta = json.loads((args.inputs / "INPUT-IDENTITY.json").read_text())
X = np.fromfile(args.inputs / "design-f64le.bin", dtype="<f8").reshape(
    (meta["analysis_cells"], meta["n_covariates"]), order="F"
)
targets = json.loads((args.inputs / "target-cells.json").read_text())
pairs = pd.read_csv(args.inputs / "crt-discovery-pairs.csv")
keys = [t for t in targets if t in set(pairs.grna_target)][:16]
selected = {k: np.asarray(targets[k], dtype=np.int64) for k in keys}
xo = x_outer_flat(X)
options = dict(B1=499, B2=0, B3=0, seed=271828, n_jobs=1, x_outer_flat_shared=xo)
cpu = fit_all_targets(selected, X, **options)
cuda = CudaTargetDesign(X, fallback=False)
gpu = fit_all_targets(selected, X, design=cuda, **options)
changes = 0
probability_delta = 0
for key in selected:
    probability_delta = max(
        probability_delta,
        float(np.max(np.abs(cpu[key].fitted_probabilities - gpu[key].fitted_probabilities))),
    )
    a = cpu[key].draws.slice(0, 499)
    b = gpu[key].draws.slice(0, 499)
    if any(not np.array_equal(getattr(a, f), getattr(b, f)) for f in ["data", "indices", "indptr"]):
        changes += 1
assert not changes
args.out.write_text(
    json.dumps(
        dict(
            targets=keys,
            cells=len(X),
            B=499,
            changed_banks=changes,
            maximum_probability_difference=probability_delta,
            gpu_calls=cuda.gpu_calls,
            fallback_calls=cuda.fallback_calls,
            scope="16 frozen real discovery target groups, exact upstream target-hashed CPU sampler, original seed, 499 first-stage draws per group. Does not qualify every budget/stage or a different random generator.",
        ),
        indent=2,
    )
    + "\n"
)
