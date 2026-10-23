"""Compare default CPU fits with an archived upstream module in one runtime."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from pysceptre.glm import irls

ap = argparse.ArgumentParser()
ap.add_argument("reference", type=Path)
ap.add_argument("receipt", type=Path)
a = ap.parse_args()
name = "pysceptre.glm._gpu_reference_irls"
spec = importlib.util.spec_from_file_location(name, a.reference)
ref = importlib.util.module_from_spec(spec)
sys.modules[name] = ref
spec.loader.exec_module(ref)
rng = np.random.default_rng(39)
count = 0
for n in [80, 700]:
    for k in [1, 8]:
        for order in ["C", "F"]:
            X = np.array(np.c_[np.ones(n), rng.normal(size=(n, 3))], order=order)
            for family in ["poisson", "binomial"]:
                Y = (
                    rng.poisson(1.4, size=(k, n))
                    if family == "poisson"
                    else rng.binomial(1, 0.04, size=(k, n))
                ).astype(float)
                for one_dimensional in [False, True]:
                    y = Y[0] if one_dimensional else Y
                    fn = "fit_" + family + "_glm_batch"
                    expected = getattr(ref, fn)(X, y)
                    actual = getattr(irls, fn)(X, y)
                    for field in vars(expected):
                        np.testing.assert_array_equal(
                            getattr(actual, field), getattr(expected, field)
                        )
                    count += 1
receipt = dict(
    passed=True,
    exact_fits=count,
    scope="Default CPU coefficients, fitted values, deviance, iterations and convergence; both families, vector/batch inputs, C/F designs in one runtime",
)
a.receipt.parent.mkdir(parents=True, exist_ok=True)
a.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
