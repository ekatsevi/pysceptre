"""Experimental Poisson IRLS through sufficient statistics; diagnostic only.

Same initialization and deviance tolerance as the existing outer solver.
FP64 reductions are reordered, so equality and downstream theta must be tested.
The implementation still uses dense counts for initialization; the formula
permits sparse initialization without a dense response on the device.
"""

import numpy as np


def fit(X, Y, *, xp=np, maxit=25, eps=1e-8):
    k, n = Y.shape
    p = X.shape[1]
    outer = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
    xy = Y @ X
    sy = Y.sum(axis=1)
    sylogy = xp.where(Y > 0, Y * xp.log(xp.maximum(Y, 1)), 0).sum(axis=1)
    mu = Y + 0.1
    # Exactly the existing first WLS initialization, before the Newton form.
    H = (mu @ outer).reshape(k, p, p)
    rhs = (mu * xp.log(mu) + Y - mu) @ X
    beta = xp.linalg.solve(H, rhs[:, :, None])[:, :, 0]
    previous = xp.full(k, xp.inf)
    converged = xp.zeros(k, dtype=bool)
    iterations = xp.zeros(k, dtype=int)
    for it in range(1, maxit + 1):
        eta = beta @ X.T
        mu = xp.maximum(xp.exp(eta), 1e-10)
        # The compact deviance assumes no clipping. Refuse that boundary.
        if bool(xp.any(eta < xp.log(1e-10))):
            raise ValueError("Clipped means need an explicit correction or fallback")
        dev = 2 * (sylogy - xp.sum(beta * xy, axis=1) - sy + mu.sum(axis=1))
        active = ~converged
        iterations[active] = it
        converged |= xp.abs(dev - previous) / (xp.abs(dev) + 0.1) < eps
        previous[active] = dev[active]
        if bool(xp.all(converged)) or it == maxit:
            break
        H = (mu @ outer).reshape(k, p, p)
        gradient = xy - mu @ X
        delta = xp.linalg.solve(H, gradient[:, :, None])[:, :, 0]
        beta = xp.where(converged[:, None], beta, beta + delta)
    return beta, mu, iterations, converged
