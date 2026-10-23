"""Opt-in first-stage score batching with CPU or resident CUDA arrays."""

from __future__ import annotations

import numpy as np
from scipy import sparse


class ScoreBatch:
    """Apply unchanged CSR draws to resident float64 score columns.

    ``stacked`` holds genes consecutively in columns: a, w and D's rows per
    gene. Only requested first-stage genes are reduced. Tail inference and
    all later stages stay with the caller's existing implementation.
    """

    def __init__(self, stacked, components, backend, *, max_bytes, telemetry, bank_cache=None):
        if stacked.ndim != 2 or stacked.shape[1] % components:
            raise ValueError("Score batch dimensions do not match components")
        if stacked.nbytes * 3 > max_bytes:
            raise MemoryError("Resident score batch exceeds its byte budget")
        self.max_bytes = max_bytes
        self.host = stacked
        self.components = components
        self.backend = backend
        self.telemetry = telemetry
        self.bank_cache = {} if bank_cache is None else bank_cache
        self.cp = None
        self.device = None
        if backend == "cuda_batch":
            import cupy as cp

            free, _ = cp.cuda.runtime.memGetInfo()
            if stacked.nbytes * 3 > free * 0.7:
                raise MemoryError("Resident score batch exceeds 70 percent of free VRAM")
            self.cp = cp
            self.device = cp.array(stacked, order="F")
            telemetry["uploaded_score_bytes"] += stacked.nbytes

    def statistics(self, draws, positions, *, cache_key=None):
        if not sparse.isspmatrix_csr(draws):
            raise ValueError("Unweighted CSR draws required")
        if draws.shape[1] != self.host.shape[0]:
            raise ValueError("Draw bank cell axis differs from score axis")
        width = self.components
        positions = np.asarray(positions, dtype=np.int64)
        columns = (positions[:, None] * width + np.arange(width)).ravel()
        if self.cp is None:
            sums = draws @ np.ascontiguousarray(self.host[:, columns])
        else:
            from cupyx.scipy import sparse as gpu_sparse

            cp = self.cp
            bank_bytes = draws.data.nbytes + draws.indices.nbytes + draws.indptr.nbytes
            key = id(draws) if cache_key is None else cache_key
            cached = self.bank_cache.get(key)
            cache_bytes = sum(entry[1] for entry in self.bank_cache.values())
            if cached is None:
                if (cache_bytes + bank_bytes) * 3 + self.host.nbytes * 3 > self.max_bytes:
                    self.bank_cache.clear()
                    cache_bytes = 0
                if bank_bytes * 3 + self.host.nbytes * 3 > self.max_bytes:
                    raise MemoryError("Resident score and draw arrays exceed byte budget")
                free, _ = cp.cuda.runtime.memGetInfo()
                if bank_bytes * 3 + self.host.nbytes > free * 0.7:
                    raise MemoryError("Draw bank exceeds free VRAM allowance")
                bank = gpu_sparse.csr_matrix(draws)
                self.bank_cache[key] = (bank, bank_bytes)
                self.telemetry["uploaded_draw_bytes"] += bank_bytes
            else:
                bank = cached[0]
            if len(columns) == self.device.shape[1] and np.array_equal(
                columns, np.arange(len(columns))
            ):
                dense = self.device
            elif len(columns) and np.all(np.diff(columns) == 1):
                dense = self.device[:, int(columns[0]) : int(columns[-1]) + 1]
            else:
                dense = cp.asfortranarray(self.device[:, cp.asarray(columns)])
            sums = bank @ dense
            self.telemetry["gpu_matmuls"] += 1
        G = len(positions)
        # Reductions retain multiplicity, original CSR weights and cell order.
        xp = np if self.cp is None else self.cp
        values = sums.reshape(draws.shape[0], G, width)
        variance = values[:, :, 1] - xp.sum(values[:, :, 2:] ** 2, axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = values[:, :, 0] / xp.sqrt(variance)
        if self.cp is not None:
            result = self.cp.asnumpy(result)
            self.telemetry["downloaded_score_bytes"] += result.nbytes
            self.telemetry["peak_pool_reserved_bytes"] = max(
                self.telemetry["peak_pool_reserved_bytes"],
                self.cp.get_default_memory_pool().total_bytes(),
            )
        self.telemetry["score_matmuls"] += 1
        self.telemetry["score_columns"] += G
        return result.T
