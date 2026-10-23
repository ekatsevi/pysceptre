"""Compare controlled CPU batching and GPU batching with the stock CPU workflow."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=Path)
a = ap.parse_args()
rows = []
for rep in [0, 1]:
    for mode in ["cpu_batch", "cuda_batch"]:
        for stage in ["calibration", "power", "discovery"]:
            cpu = pd.read_pickle(a.directory / f"r{rep}-cpu" / f"{stage}.pkl")
            new = pd.read_pickle(a.directory / f"r{rep}-{mode}" / f"{stage}.pkl")
            assert cpu.columns.equals(new.columns)
            assert cpu[["response_id", "grna_target"]].equals(new[["response_id", "grna_target"]])
            delta = {}
            for col in cpu:
                if pd.api.types.is_numeric_dtype(cpu[col]) and col != "BH_discovery":
                    x, y = cpu[col].to_numpy(dtype=float), new[col].to_numpy(dtype=float)
                    assert np.array_equal(np.isfinite(x), np.isfinite(y)), col
                    ok = np.isfinite(x)
                    delta[col] = dict(
                        max_abs=float(np.max(abs(x[ok] - y[ok]))) if ok.any() else None,
                        changed=int(np.count_nonzero(x[ok] != y[ok])),
                    )
                else:
                    delta[col] = dict(
                        changed=int(
                            np.count_nonzero(
                                ~(cpu[col].eq(new[col]) | (cpu[col].isna() & new[col].isna()))
                            )
                        )
                    )
            rows.append(dict(rep=rep, mode=mode, stage=stage, rows=len(cpu), differences=delta))
stability = []
for mode in ["cpu", "cpu_batch", "cuda_batch"]:
    for stage in ["calibration", "power", "discovery"]:
        x = pd.read_pickle(a.directory / f"r0-{mode}" / f"{stage}.pkl")
        y = pd.read_pickle(a.directory / f"r1-{mode}" / f"{stage}.pkl")
        stability.append(dict(mode=mode, stage=stage, exact=x.equals(y)))
(a.directory / "comparison.json").write_text(
    json.dumps(dict(paired=rows, repeat_stability=stability), indent=2) + "\n"
)
