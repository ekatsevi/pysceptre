"""Compare complete scientific outputs, fits, RNG states and all sampled banks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=Path)
a = ap.parse_args()
report = []
for rep in [0, 1]:
    for stage in ["calibration", "power", "discovery"]:
        left, right = [a.directory / f"r{rep}-{mode}" for mode in ["cpu", "cuda_resident"]]
        x, y = [pd.read_pickle(p / f"{stage}.pkl") for p in [left, right]]
        assert x.columns.equals(y.columns)
        assert x[["response_id", "grna_target"]].equals(y[["response_id", "grna_target"]])
        delta = {}
        for col in x:
            if pd.api.types.is_numeric_dtype(x[col]) and col != "BH_discovery":
                u, v = x[col].to_numpy(dtype=float), y[col].to_numpy(dtype=float)
                assert np.array_equal(np.isfinite(u), np.isfinite(v)), col
                ok = np.isfinite(u)
                delta[col] = dict(
                    max_abs=float(np.max(abs(u[ok] - v[ok]))) if ok.any() else None,
                    changed=int(np.count_nonzero(u[ok] != v[ok])),
                )
            else:
                delta[col] = dict(
                    changed=int(
                        np.count_nonzero(~(x[col].eq(y[col]) | (x[col].isna() & y[col].isna())))
                    )
                )
        fx, fy = [json.loads((p / f"{stage}-fits.json").read_text()) for p in [left, right]]
        assert set(fx) == set(fy)
        fit_delta = dict(
            max_coefficient_abs=max(
                float(np.max(abs(np.array(fx[g]["coefs"]) - fy[g]["coefs"]))) for g in fx
            ),
            max_theta_abs=max(abs(fx[g]["theta"] - fy[g]["theta"]) for g in fx),
            theta_method_changes=sum(fx[g]["theta_method"] != fy[g]["theta_method"] for g in fx),
            clamp_changes=sum(fx[g]["theta_clamped"] != fy[g]["theta_clamped"] for g in fx),
            convergence_changes=sum(fx[g]["glm_converged"] != fy[g]["glm_converged"] for g in fx),
        )
        dx, dy = [json.loads((p / f"{stage}-draws.json").read_text()) for p in [left, right]]
        assert set(dx) == set(dy)
        rng_delta = {
            k: sum(dx[t][k] != dy[t][k] for t in dx)
            for k in ["probability_sha256", "bank_sha256", "before", "after", "B", "n_trt"]
        }
        assert not any(rng_delta.values()), rng_delta
        report.append(
            dict(
                rep=rep,
                stage=stage,
                rows=len(x),
                differences=delta,
                fit_differences=fit_delta,
                draw_differences=rng_delta,
            )
        )
stability = []
for mode in ["cpu", "cuda_resident"]:
    for stage in ["calibration", "power", "discovery"]:
        x, y = [pd.read_pickle(a.directory / f"r{rep}-{mode}" / f"{stage}.pkl") for rep in [0, 1]]
        stability.append(dict(mode=mode, stage=stage, exact=x.equals(y)))
(a.directory / "comparison.json").write_text(
    json.dumps(dict(paired=report, repeat_stability=stability), indent=2) + "\n"
)
