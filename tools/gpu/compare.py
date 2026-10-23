"""Compare full paired workflow outputs, retaining numerical and decision differences."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=Path)
args = ap.parse_args()
rows = []
for rep in [0, 1]:
    for stage in ["calibration", "power", "discovery"]:
        base = pd.read_pickle(args.directory / f"r{rep}-cpu" / f"{stage}.pkl")
        gpu = pd.read_pickle(args.directory / f"r{rep}-gpu" / f"{stage}.pkl")
        assert base.columns.equals(gpu.columns)
        assert base[["response_id", "grna_target"]].equals(gpu[["response_id", "grna_target"]])
        row = dict(rep=rep, stage=stage, rows=len(base), differences={})
        for col in base:
            if pd.api.types.is_numeric_dtype(base[col]) and col != "BH_discovery":
                a, b = base[col].to_numpy(dtype=float), gpu[col].to_numpy(dtype=float)
                assert np.array_equal(np.isfinite(a), np.isfinite(b)), col
                finite = np.isfinite(a)
                row["differences"][col] = dict(
                    max_abs=float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None,
                    changed=int(np.count_nonzero(a[finite] != b[finite])),
                )
            else:
                row["differences"][col] = dict(
                    changed=int(
                        np.count_nonzero(
                            ~(base[col].eq(gpu[col]) | (base[col].isna() & gpu[col].isna()))
                        )
                    )
                )
        rows.append(row)
# Also report run-to-run stability per mode, separately from CPU/GPU agreement.
stability = []
for mode in ["cpu", "gpu"]:
    for stage in ["calibration", "power", "discovery"]:
        a = pd.read_pickle(args.directory / f"r0-{mode}" / f"{stage}.pkl")
        b = pd.read_pickle(args.directory / f"r1-{mode}" / f"{stage}.pkl")
        stability.append(dict(mode=mode, stage=stage, exact=a.equals(b)))
(args.directory / "comparison.json").write_text(
    json.dumps(dict(paired=rows, repeat_stability=stability), indent=2) + "\n"
)
