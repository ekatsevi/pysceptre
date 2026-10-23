"""Experimental FP64 Poisson IRLS from fixed response sufficient statistics.

This uses the existing initialization, mean floor and deviance tolerance.
The compact deviance is admitted only when clipping is absent. Arithmetic
is reordered; this is an explicit opt-in numerical backend, not the default.
"""

import numpy as np


def fit_poisson_sufficient(X, Y, *, xp=np, maxit=25, eps=1e-8):
    k, n = Y.shape
    p = X.shape[1]
    outer = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
    xy = Y @ X
    sy = Y.sum(axis=1)
    sylogy = (Y * xp.log(xp.where(Y > 0, Y, 1.0))).sum(axis=1)
    mu = Y + 0.1
    H = (mu @ outer).reshape(k, p, p)
    rhs = (mu * xp.log(mu) + Y - mu) @ X
    beta = xp.linalg.solve(H, rhs[:, :, None])[:, :, 0]
    previous = xp.full(k, xp.inf)
    converged = xp.zeros(k, dtype=bool)
    iterations = xp.zeros(k, dtype=int)
    for it in range(1, maxit + 1):
        eta = beta @ X.T
        if bool(xp.any(eta < xp.log(1e-10))):
            raise ValueError("Clipped means require the reference fit")
        mu = xp.exp(eta)
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
