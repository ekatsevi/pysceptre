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
        left, right = [a.directory / f"r{rep}-{mode}" for mode in ["cpu", "cuda_native"]]
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
        fit_delta = {}
        rng_delta = {}
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
for mode in ["cpu", "cuda_native"]:
    for stage in ["calibration", "power", "discovery"]:
        x, y = [pd.read_pickle(a.directory / f"r{rep}-{mode}" / f"{stage}.pkl") for rep in [0, 1]]
        stability.append(dict(mode=mode, stage=stage, exact=x.equals(y)))
(a.directory / "comparison.json").write_text(
    json.dumps(dict(paired=report, repeat_stability=stability), indent=2) + "\n"
)
