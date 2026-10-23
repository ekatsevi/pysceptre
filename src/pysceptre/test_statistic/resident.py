"""Bounded resident FP64 engine. CPU reference theta is an explicit boundary.

No device state is exported in model caches/results. CUDA is loaded only by
the opt-in path. The upstream sampler, random streams, budgets and tail test
remain the source of truth. All adaptive score stages can use the device.
"""

import warnings
from collections import OrderedDict

import numpy as np

from ..glm.irls import fit_poisson_glm_batch, x_outer_flat
from ..glm.nb_theta import estimate_theta
from ..glm.sufficient import fit_poisson_sufficient


class ResidentDeclined(RuntimeError):
    pass


class ResidentEngine:
    def __init__(self, X, max_bytes, telemetry):
        import cupy as cp
        from cupyx.scipy import sparse as cs

        self.cp, self.cs = cp, cs
        self.host_X = X
        self.X = cp.asarray(X)
        self.outer = (self.X[:, :, None] * self.X[:, None, :]).reshape(len(X), -1)
        self.host_outer = x_outer_flat(X)
        free, _ = cp.cuda.runtime.memGetInfo()
        self.budget = min(max_bytes, int(0.65 * free))
        self.host_budget = max_bytes // 2
        self.cache = OrderedDict()
        self.host_cache = OrderedDict()
        self.used = self.host_used = 0
        self.fits = {}
        self.force_cpu = set()
        self.stats = telemetry
        self.stats.update(
            response_gpu_batches=0,
            response_gpu_genes=0,
            response_cpu_fallback_genes=0,
            sensitive_cpu_score_jobs=0,
            uploaded_count_bytes=0,
            downloaded_mean_bytes=0,
            gene_evictions=0,
            bank_evictions=0,
            gpu_later_stage_calls=0,
            gpu_observed_calls=0,
            peak_cached_gpu_bytes=0,
            peak_cached_host_bytes=0,
            theta_backend="unchanged_cpu_reference_on_gpu_means",
        )

    def _get(self, key):
        item = self.cache.pop(key, None)
        if item is not None:
            self.cache[key] = item
            return item[0]
        return None

    def _put(self, key, value, size):
        if size > self.budget:
            raise ResidentDeclined("A resident object exceeds the device cache budget")
        old = self.cache.pop(key, None)
        if old is not None:
            self.used -= old[1]
        while self.used + size > self.budget:
            expired, (_, amount) = self.cache.popitem(last=False)
            self.used -= amount
            self.stats["gene_evictions" if expired[0] == "gene" else "bank_evictions"] += 1
        self.cache[key] = value, size
        self.used += size
        self.stats["peak_cached_gpu_bytes"] = max(self.stats["peak_cached_gpu_bytes"], self.used)

    def _host_put(self, gene, y, mu):
        amount = y.nbytes + mu.nbytes
        if amount > self.host_budget:
            return
        while self.host_used + amount > self.host_budget:
            _, (old_y, old_mu) = self.host_cache.popitem(last=False)
            self.host_used -= old_y.nbytes + old_mu.nbytes
        self.host_cache[gene] = y.copy(), mu.copy()
        self.host_used += amount
        self.stats["peak_cached_host_bytes"] = max(
            self.stats["peak_cached_host_bytes"], self.host_used
        )

    def host_inputs(self, gene, response_matrix, row):
        item = self.host_cache.pop(gene, None)
        if item is not None:
            self.host_cache[gene] = item
            return item
        from ..pipeline.discovery import _get_row

        y = _get_row(response_matrix, row)
        return y, np.exp(self.host_X @ self.fits[gene].fitted_coefs)

    def fit_genes(self, response_matrix, ids, rows, batch_size):
        from ..pipeline.discovery import (
            GenePrecomputation,
            _get_row,
            _warn_about_degenerate_gene_fits,
        )

        cp = self.cp
        n, p = self.X.shape
        # Fit workspaces are separate from retained cache storage. Refuse a
        # single response if even its estimated transient workspace cannot fit.
        width = min(batch_size, max(1, self.budget // (n * 8 * (8 + 3 * p))))
        for start in range(0, len(ids), width):
            names = ids[start : start + width]
            y = np.array([_get_row(response_matrix, r) for r in rows[start : start + width]])
            try:
                dy = cp.asarray(y)
                self.stats["uploaded_count_bytes"] += y.nbytes
                beta, mu, _, ok = fit_poisson_sufficient(self.X, dy, xp=cp)
                beta_host, mu_host, ok_host = cp.asnumpy(beta), cp.asnumpy(mu), cp.asnumpy(ok)
                if not np.all(np.isfinite(beta_host)) or not np.all(np.isfinite(mu_host)):
                    raise ResidentDeclined("Nonfinite device response fit")
                self.stats["response_gpu_batches"] += 1
                self.stats["downloaded_mean_bytes"] += mu_host.nbytes
            except (ValueError, RuntimeError, MemoryError) as exc:
                warnings.warn(
                    "Resident response batch replayed on CPU: " + str(exc),
                    RuntimeWarning,
                    stacklevel=2,
                )
                beta_host = mu_host = ok_host = None
            for j, gene in enumerate(names):
                cpu = beta_host is None or not ok_host[j]
                if not cpu:
                    theta_est, method = estimate_theta(y[j], mu_host[j], n - p)
                    # Sensitive boundary/fallback genes replay the original fit
                    # as well as original theta. Never apply the guide tolerance.
                    cpu = method != 1 or not np.isfinite(theta_est) or not 0.01 < theta_est < 1000
                if cpu:
                    ref = fit_poisson_glm_batch(self.host_X, y[j], X_outer_flat=self.host_outer)
                    coeff, mean, converged = ref.coefs, ref.fitted_values, bool(ref.converged)
                    theta_est, method = estimate_theta(y[j], mean, n - p)
                    self.force_cpu.add(gene)
                    self.stats["response_cpu_fallback_genes"] += 1
                else:
                    coeff, mean, converged = beta_host[j], mu_host[j], bool(ok_host[j])
                    self.stats["response_gpu_genes"] += 1
                theta = max(min(theta_est, 1000.0), 0.01)
                self.fits[gene] = GenePrecomputation(
                    fitted_coefs=coeff,
                    theta=theta,
                    theta_method=method,
                    theta_clamped=not (0.01 <= theta_est <= 1000),
                    glm_converged=converged,
                    n_covariates=p,
                )
                self._host_put(gene, y[j], mean)
                if cpu:
                    from ..precompute.pieces import compute_precomputation_pieces

                    q = compute_precomputation_pieces(y[j], self.host_X, coeff, theta)
                    lo, hi = q.min_eigenvalue, q.max_eigenvalue
                else:
                    _, lo, hi = self._make_gene(gene, y[j])
                    if not np.isfinite(lo) or lo <= hi * p * np.finfo(float).eps:
                        self.force_cpu.add(gene)
                self.fits[gene].min_eigenvalue = lo
                self.fits[gene].max_eigenvalue = hi
        _warn_about_degenerate_gene_fits(self.fits)
        return self.fits

    def _make_gene(self, gene, y):
        cp = self.cp
        f = self.fits[gene]
        mu = cp.exp(self.X @ cp.asarray(f.fitted_coefs))
        denom = 1 + mu / f.theta
        w = mu / denom
        wX = w[:, None] * self.X
        H = self.X.T @ wX
        eig, V = cp.linalg.eigh(H)
        D = (V.T / cp.sqrt(eig)[:, None]) @ wX.T
        flat = cp.empty((len(y), self.X.shape[1] + 2), dtype=cp.float64, order="F")
        flat[:, 0] = (cp.asarray(y) - mu) / denom
        flat[:, 1] = w
        flat[:, 2:] = D.T
        bounds = cp.asnumpy(eig[[0, -1]])
        self._put(("gene", gene), flat, flat.nbytes)
        return flat, float(bounds[0]), float(bounds[1])

    def gene(self, gene, y):
        if gene in self.force_cpu:
            raise ResidentDeclined("Sensitive response fit uses reference CPU scores")
        found = self._get(("gene", gene))
        return self._make_gene(gene, y)[0] if found is None else found

    def bank(self, host, key):
        found = self._get(("bank", key))
        if found is not None:
            return found
        bank = self.cs.csr_matrix(host)
        amount = bank.data.nbytes + bank.indices.nbytes + bank.indptr.nbytes
        self.stats["uploaded_draw_bytes"] += amount
        self._put(("bank", key), bank, amount)
        return bank

    def statistics(self, host_bank, flats, key):
        cp = self.cp
        bank = self.bank(host_bank, key)
        flat = flats[0] if len(flats) == 1 else cp.asfortranarray(cp.concatenate(flats, axis=1))
        sums = (bank @ flat).reshape(host_bank.shape[0], len(flats), -1)
        result = sums[:, :, 0] / cp.sqrt(sums[:, :, 1] - cp.sum(sums[:, :, 2:] ** 2, axis=2))
        out = cp.asnumpy(result.T)
        self.stats["gpu_matmuls"] += 1
        self.stats["score_matmuls"] += 1
        self.stats["score_columns"] += len(flats)
        self.stats["downloaded_score_bytes"] += out.nbytes
        self.stats["peak_pool_reserved_bytes"] = max(
            self.stats["peak_pool_reserved_bytes"], cp.get_default_memory_pool().total_bytes()
        )
        return out

    def observed(self, flat, cells):
        cp = self.cp
        sums = cp.sum(flat[cp.asarray(cells)], axis=0)
        self.stats["gpu_observed_calls"] += 1
        return float(sums[0] / cp.sqrt(sums[1] - cp.sum(sums[2:] ** 2)))
