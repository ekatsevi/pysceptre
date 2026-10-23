"""Actual frozen genes: test a different GPU algorithm, not another API port."""
# ruff: noqa: E402

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import cupy as cp
import numpy as np
import pandas as pd
from sufficient_fit import fit

from pysceptre.crt.sampler import crt_index_sampler_exact
from pysceptre.glm.irls import fit_binomial_glm_batch, fit_poisson_glm_batch, x_outer_flat
from pysceptre.glm.nb_theta import estimate_theta

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("out", type=Path)
a = ap.parse_args()
sys.path.insert(0, str(a.root))
from adapter import inputs

Y, genes, X, targets, _, _, base = inputs(a.root, "backed")
ids = list(dict.fromkeys(pd.read_csv(a.root / "inputs/crt-discovery-pairs.csv").response_id))[:32]
rows = [genes.index(g) for g in ids]
counts = np.array([Y[r].toarray().ravel() for r in rows], dtype=float)
if base is not None:
    base.close()
xo = x_outer_flat(X)
t = time.perf_counter()
reference = [fit_poisson_glm_batch(X, y, X_outer_flat=xo) for y in counts]
cpu_fit = time.perf_counter() - t
t = time.perf_counter()
ref_theta = [
    estimate_theta(y, f.fitted_values, len(X) - X.shape[1])
    for y, f in zip(counts, reference, strict=True)
]
cpu_theta = time.perf_counter() - t
rc = np.array([f.coefs for f in reference])
rm = np.array([f.fitted_values for f in reference])
ri = np.array([f.n_iter for f in reference])
out = []
cpu_sufficient_times = []
for _ in range(3):
    t = time.perf_counter()
    compact_cpu = fit(X, counts)
    cpu_sufficient_times.append(time.perf_counter() - t)
np.testing.assert_allclose(compact_cpu[0], rc, atol=1e-9, rtol=1e-9)
for width in [1, 8, 32]:
    t = time.perf_counter()
    dx = cp.asarray(X)
    dy = cp.asarray(counts)
    cp.cuda.runtime.deviceSynchronize()
    upload = time.perf_counter() - t
    times = []
    for _repeat in range(3):
        t = time.perf_counter()
        chunks = [fit(dx, dy[i : i + width], xp=cp) for i in range(0, len(ids), width)]
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t)
    t = time.perf_counter()
    coefs, means, iterations, convergence = [
        cp.asnumpy(cp.concatenate([c[j] for c in chunks])) for j in range(4)
    ]
    download = time.perf_counter() - t
    # Deliberately the unchanged CPU theta routine: this is a sensitivity gate,
    # not a claim that this diagnostic is a fully resident workflow.
    theta = [estimate_theta(y, m, len(X) - X.shape[1]) for y, m in zip(counts, means, strict=True)]
    ct = np.clip([v[0] for v in ref_theta], 0.01, 1000)
    gt = np.clip([v[0] for v in theta], 0.01, 1000)
    out.append(
        dict(
            width=width,
            fit_seconds=times,
            upload_seconds=upload,
            diagnostic_download_seconds=download,
            max_coefficient_abs_difference=float(np.max(abs(coefs - rc))),
            max_mean_abs_difference=float(np.max(abs(means - rm))),
            changed_iterations=int(np.count_nonzero(iterations != ri)),
            nonconverged=int(np.count_nonzero(~convergence)),
            max_clamped_theta_abs_difference=float(np.max(abs(ct - gt))),
            theta_method_changes=int(
                sum(x[1] != y[1] for x, y in zip(ref_theta, theta, strict=True))
            ),
            cpu_theta=ref_theta,
            candidate_theta=theta,
        )
    )

# Decompose the sparse sampler without changing it. NumPy geometric inversion
# consumes standard exponentials (Ziggurat, with rejection), not one uniform
# per gap. Preserve that stream on CPU and test device inversion separately.
sampling = []
for target, cells in list(targets.items())[:4]:
    treatment = np.zeros(len(X))
    treatment[cells] = 1
    prob = fit_binomial_glm_batch(X, treatment, X_outer_flat=xo).fitted_values

    class Probe:
        def __init__(self):
            self.rng = np.random.default_rng(271828)
            self.calls = []

        def geometric(self, p):
            state = self.rng.bit_generator.state
            t = time.perf_counter()
            value = self.rng.geometric(p)
            rng_seconds = time.perf_counter() - t
            other = np.random.default_rng()
            other.bit_generator.state = state
            if np.max(p) < 1 / 3:
                t = time.perf_counter()
                exponential = other.standard_exponential(p.size)
                exponential_seconds = time.perf_counter() - t
                dp, de = cp.asarray(p), cp.asarray(exponential)
                cp.cuda.runtime.deviceSynchronize()
                t = time.perf_counter()
                got = cp.minimum(
                    cp.ceil(-de / cp.log1p(-dp)), float(np.iinfo(np.int64).max)
                ).astype(cp.int64)
                cp.cuda.runtime.deviceSynchronize()
                transform_seconds = time.perf_counter() - t
                changed = int(np.count_nonzero(cp.asnumpy(got) != value))
                same_state = other.bit_generator.state == self.rng.bit_generator.state
            else:
                exponential_seconds = transform_seconds = changed = same_state = None
            self.calls.append(
                dict(
                    gaps=p.size,
                    geometric_seconds=rng_seconds,
                    exponential_seconds=exponential_seconds,
                    device_transform_seconds=transform_seconds,
                    gap_changes=changed,
                    same_rng_state=same_state,
                    maximum_probability=float(p.max()),
                )
            )
            return value

    probe = Probe()
    t = time.perf_counter()
    draws = crt_index_sampler_exact(prob, 499, probe)
    sampling.append(
        dict(
            target=target,
            inclusions=sum(len(d) for d in draws),
            instrumented_seconds=time.perf_counter() - t,
            calls=probe.calls,
        )
    )
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(
    json.dumps(
        dict(
            genes=ids,
            cells=len(X),
            covariates=X.shape[1],
            cpu_width_one_fit_seconds=cpu_fit,
            cpu_theta_seconds=cpu_theta,
            cpu_sufficient_fit_seconds=cpu_sufficient_times,
            rows=out,
            sampling=sampling,
            scope="32 actual genes; FP64 sufficient-statistic Poisson fit only. CPU unchanged theta used as a sensitivity diagnostic. No null draws, full family, inference timing, or release qualification.",
        ),
        indent=2,
        default=float,
    )
    + "\n"
)
