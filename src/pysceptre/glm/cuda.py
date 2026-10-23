"""Opt-in CUDA target IRLS; CPU gene fitting, RNG and inference stay upstream.

Uses the existing IRLS loop in float64 with CuPy/cuBLAS/cuSOLVER. The model,
initialization, active-response masks and stopping rule are shared with CPU.
CUDA reductions need not be bit-identical. No optional import on default paths.
"""

from __future__ import annotations

import threading
import time
import warnings

import numpy as np

from .irls import GlmFitBatchResult, _fit_batch, fit_binomial_glm_batch


class CudaTargetDesign:
    """Lazy, bounded, per-analysis GPU design with an explicit CPU fallback.

    Pipeline prefetch threads serialize on this object's lock; no CUDA state is
    sent to worker processes. ``fallback=False`` makes GPU failures fatal.
    ``max_bytes`` bounds an estimated working set, not total process memory.
    """

    def __init__(self, X, *, fallback=True, max_bytes=2 * 1024**3, device=0):
        self.X = np.asarray(X, dtype=np.float64)
        self.fallback = fallback
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.device = device
        self._lock = threading.Lock()
        self._arrays = None
        self.gpu_calls = 0
        self.fallback_calls = 0
        self.last_failure = None
        self.last_stats = None

    def fit(self, Y, *, X_outer_flat=None):
        with self._lock:
            try:
                result = self._fit_gpu(Y)
                self.gpu_calls += 1
                return result
            except (
                ImportError,
                RuntimeError,
                ValueError,
                MemoryError,
                np.linalg.LinAlgError,
            ) as exc:
                if not self.fallback:
                    raise
                self.fallback_calls += 1
                self.last_failure = f"{type(exc).__name__}: {exc}"
                if self.fallback_calls == 1:
                    warnings.warn(
                        "CUDA target fit declined; using CPU outer IRLS: " + self.last_failure,
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return fit_binomial_glm_batch(self.X, Y, X_outer_flat=X_outer_flat)

    def _fit_gpu(self, Y):
        started = time.perf_counter()
        import cupy as cp
        import cupyx

        imported = time.perf_counter()

        Y = np.asarray(Y, dtype=np.float64)
        n, p = self.X.shape
        if Y.ndim != 2 or Y.shape[1] != n or not np.isfinite(Y).all():
            raise ValueError("CUDA target responses must be finite (targets, cells)")
        # Conservative bound for resident design + outer products, loop state,
        # active-row copies, working response and solver temporaries.
        required = 8 * (n * p + n * p * p + 18 * Y.size + 4 * Y.shape[0] * p * p)
        if required > self.max_bytes:
            raise MemoryError(f"estimated target fit {required} exceeds {self.max_bytes} bytes")
        with cp.cuda.Device(self.device):
            free, _ = cp.cuda.runtime.memGetInfo()
            if required > free * 0.7:
                raise MemoryError("target fit exceeds 70% of free device memory")
            if self._arrays is None:
                X = cp.asarray(self.X)
                outer = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
                self._arrays = (X, outer)
            X, outer = self._arrays
            device_Y = cp.asarray(Y)
            cp.cuda.get_current_stream().synchronize()
            transferred = time.perf_counter()
            event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
            event_start.record()
            with cupyx.errstate(linalg="raise"):
                fit = _fit_batch(X, device_Y, "binomial", X_outer_flat=outer, _xp=cp)
            event_end.record()
            event_end.synchronize()
            computed = time.perf_counter()
            host = GlmFitBatchResult(
                **{
                    name: cp.asnumpy(getattr(fit, name))
                    for name in ("coefs", "fitted_values", "deviance", "n_iter", "converged")
                }
            )
            self.last_stats = dict(
                import_seconds=imported - started,
                setup_transfer_seconds=transferred - imported,
                compute_seconds=computed - transferred,
                device_event_seconds=cp.cuda.get_elapsed_time(event_start, event_end) / 1000,
                return_seconds=time.perf_counter() - computed,
                estimated_bytes=required,
                pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
                pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
            )
            if not np.isfinite(host.coefs).all() or not np.isfinite(host.fitted_values).all():
                raise ValueError("nonfinite CUDA fitted model")
            if not np.all(host.converged):
                raise ValueError("CUDA IRLS did not converge; replay CPU diagnostics")
            return host
