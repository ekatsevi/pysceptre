"""Experimental fully resident sampling/dispersion and compact binary CRT scores."""

import copy
import time
import weakref
from functools import wraps

import numpy as np
from scipy import sparse

from .resident import ResidentDeclined, ResidentEngine
from .score_stat import ListDraws, StagedDraws


def clocked(name):
    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                self.stats[name] = self.stats.get(name, 0.0) + time.perf_counter() - started

        return call

    return decorate


class DeviceDraws(StagedDraws):
    def __init__(self, cells, offsets, n_cells, engine):
        super().__init__(n_cells, len(offsets) - 1)
        self.cells, self.offsets = cells, offsets
        self.engine = weakref.ref(engine)
        self.host = None
        self.amount = cells.nbytes + offsets.nbytes
        engine.draw_used += self.amount
        engine.stats["peak_native_bank_bytes"] = max(
            engine.stats["peak_native_bank_bytes"], engine.draw_used
        )

    def __del__(self):
        engine = self.engine()
        if engine is not None:
            engine.draw_used -= self.amount

    def device_slice(self, lo, hi):
        return self.cells, self.offsets, max(0, int(lo)), min(int(hi), self.n_draws)

    def _host(self):
        if self.host is None:
            import cupy as cp

            cells, offsets = cp.asnumpy(self.cells), cp.asnumpy(self.offsets)
            self.host = cells, offsets
            engine = self.engine()
            if engine is not None:
                engine.stats["downloaded_bank_bytes"] += cells.nbytes + offsets.nbytes
        return self.host

    def _index_arrays(self, lo, hi):
        cells, offsets = self._host()
        return [cells[offsets[i] : offsets[i + 1]] for i in range(lo, hi)]

    def iter_host_rows(self):
        return iter(self._index_arrays(0, self.n_draws))

    def slice(self, lo, hi):
        lo, hi = max(0, int(lo)), min(int(hi), self.n_draws)
        if hi <= lo:
            return sparse.csr_matrix((0, self.n_cells))
        cells, offsets = self._host()
        start, stop = offsets[lo], offsets[hi]
        return sparse.csr_matrix(
            (np.ones(stop - start), cells[start:stop], offsets[lo : hi + 1] - start),
            shape=(hi - lo, self.n_cells),
        )


SCORE_KERNEL = r"""
extern "C" __global__ void compact(const int* cells,const long long* offsets,int lo,int n,int p,const unsigned long long* pointers,const double* inv,const double* X,double* out,unsigned int* danger){
 int draw=blockIdx.x,g=blockIdx.y,t=threadIdx.x;
 const double* aw=(const double*)pointers[g];
 extern __shared__ double sums[];
 double s=0,q=0,v[16];for(int k=0;k<p;k++)v[k]=0;
 for(long long j=offsets[draw+lo]+t;j<offsets[draw+lo+1];j+=blockDim.x){
  int cell=cells[j];double w=aw[n+cell];s+=aw[cell];q+=w;
  for(int k=0;k<p;k++)v[k]+=w*X[(long long)cell*p+k];
 }
 sums[t]=s;sums[blockDim.x+t]=q;
 for(int k=0;k<p;k++)sums[(k+2)*blockDim.x+t]=v[k];__syncthreads();
 for(int d=blockDim.x/2;d>0;d>>=1){if(t<d)for(int k=0;k<p+2;k++)sums[k*blockDim.x+t]+=sums[k*blockDim.x+t+d];__syncthreads();}
 if(t==0){double proj=0;for(int k=0;k<p;k++)for(int l=0;l<p;l++)proj+=sums[(k+2)*blockDim.x]*inv[(g*p+k)*p+l]*sums[(l+2)*blockDim.x];
  double variance=sums[blockDim.x]-proj;
  if(!isfinite(variance)||variance<=4096*0x1.0p-52*fabs(sums[blockDim.x]))atomicExch(danger,1);
  out[g*gridDim.x+draw]=sums[0]/sqrt(variance);
 }
}
"""


class NativeEngine(ResidentEngine):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        cp = self.cp
        if not 1 <= self.X.shape[1] <= 16:
            raise ResidentDeclined("Compact kernel supports 1..16 covariates")
        self.score_X = cp.ascontiguousarray(self.X)
        self.draw_used = 0
        self.kernel = cp.RawKernel(SCORE_KERNEL, "compact", options=("--std=c++17", "--fmad=false"))
        self.stats.update(
            native_banks=0,
            native_sampler_replays=0,
            downloaded_bank_bytes=0,
            peak_native_bank_bytes=0,
            theta_gpu_genes=0,
            theta_reference_replays=0,
            theta_backend="resident_integer_MLE_with_reference_replay",
            adaptive_score_batches=0,
            compact_variance_replays=0,
            compact_score_calls=0,
        )

    def _put(self, key, value, size):
        old = self.budget
        self.budget = max(0, old - self.draw_used)
        try:
            return super()._put(key, value, size)
        finally:
            self.budget = old

    @clocked("fit_genes_seconds")
    def fit_genes(self, *args, **kwargs):
        return super().fit_genes(*args, **kwargs)

    @clocked("sample_seconds")
    def sample(self, p, B, rng, n_trt):
        from ..crt.sampler import EXACT_SAMPLER_ABOVE_SHARE, crt_index_sampler

        state = copy.deepcopy(rng.bit_generator.state)
        try:
            if np.__version__ not in ("2.4.6", "2.5.4") or n_trt <= EXACT_SAMPLER_ABOVE_SHARE * len(
                p
            ):
                raise ResidentDeclined("Unsupported NumPy version or approximate sampler")
            from ..crt.native.bank import bank

            cells, offsets = bank(p, B, rng)
            amount = cells.nbytes + offsets.nbytes
            if amount + self.draw_used > self.budget // 2:
                raise ResidentDeclined("Native target bank budget")
            while self.used + self.draw_used + amount > self.budget:
                expired, (_, size) = self.cache.popitem(last=False)
                self.used -= size
                self.stats["gene_evictions" if expired[0] == "gene" else "bank_evictions"] += 1
            result = DeviceDraws(cells, offsets, len(p), self)
            self.stats["native_banks"] += 1
            return result
        except (ValueError, RuntimeError, MemoryError):
            rng.bit_generator.state = state
            self.stats["native_sampler_replays"] += 1
            return ListDraws(crt_index_sampler(p, B, rng, n_trt), len(p))

    @clocked("theta_batch_seconds")
    def theta_batch(self, y, dy, mu):
        from ..glm.native_theta import fit

        cp = self.cp
        valid = np.all(np.isfinite(y) & (y >= 0) & (y == np.floor(y)), axis=1) & (
            np.max(y, axis=1) <= 65535
        )
        estimates = np.full(len(y), np.nan)
        safe = np.zeros(len(y), bool)
        if np.any(valid):
            indices = np.flatnonzero(valid)
            th, failed, it = fit(
                dy[cp.asarray(indices)], mu[cp.asarray(indices)], int(y[valid].max()) + 1
            )
            values, failed = cp.asnumpy(th), cp.asnumpy(failed)
            # Large/near-Poisson and boundary estimates retain the complete reference fit.
            admitted = (~failed) & np.isfinite(values) & (values > 0.01001) & (values < 100)
            estimates[indices] = values
            safe[indices] = admitted
        self.stats["theta_gpu_genes"] += int(safe.sum())
        self.stats["theta_reference_replays"] += int((~safe).sum())
        return estimates, safe

    def _make_gene(self, gene, y):
        cp = self.cp
        f = self.fits[gene]
        mu = cp.exp(self.X @ cp.asarray(f.fitted_coefs))
        denom = 1 + mu / f.theta
        w = mu / denom
        H = self.X.T @ (w[:, None] * self.X)
        eig = cp.linalg.eigvalsh(H)
        aw = cp.stack(((cp.asarray(y) - mu) / denom, w))
        inverse = cp.linalg.inv(H)
        bounds = cp.asnumpy(eig[[0, -1]])
        value = (aw, inverse)
        self._put(("gene", gene), value, aw.nbytes + inverse.nbytes)
        return value, float(bounds[0]), float(bounds[1])

    @clocked("statistics_seconds")
    def statistics(self, draws, flats, key):
        cp = self.cp
        if isinstance(draws, DeviceDraws):
            cells, offsets, lo, hi = draws.device_slice(key[1], key[2])
        else:
            # Approximate banks can contain multiplicities; compact binary algebra is refused.
            raise ResidentDeclined("Non-native bank requires original algebra")
        if hi <= lo:
            return np.empty((len(flats), 0))
        p = self.X.shape[1]
        n = len(self.host_X)
        pointers = cp.asarray([flat[0].data.ptr for flat in flats], dtype=cp.uint64)
        inv = cp.stack([flat[1] for flat in flats])
        out = cp.empty((len(flats), hi - lo))
        danger = cp.zeros(1, cp.uint32)
        self.kernel(
            (hi - lo, len(flats)),
            (128,),
            (
                cells,
                offsets,
                np.int32(lo),
                np.int32(n),
                np.int32(p),
                pointers,
                inv,
                self.score_X,
                out,
                danger,
            ),
            shared_mem=(p + 2) * 128 * 8,
        )
        if int(danger.get()[0]):
            self.stats["compact_variance_replays"] += 1
            raise ResidentDeclined("Compact variance requires reference scoring")
        host = cp.asnumpy(out)
        self.stats["gpu_matmuls"] += 1
        self.stats["score_matmuls"] += 1
        self.stats["compact_score_calls"] += 1
        self.stats["score_columns"] += len(flats)
        self.stats["downloaded_score_bytes"] += host.nbytes
        self.stats["peak_pool_reserved_bytes"] = max(
            self.stats["peak_pool_reserved_bytes"], cp.get_default_memory_pool().total_bytes()
        )
        return host

    @clocked("observed_seconds")
    def observed(self, flat, cells):
        cp = self.cp
        aw, inverse = flat
        idx = cp.asarray(cells)
        w = aw[1, idx]
        s = cp.sum(aw[0, idx])
        q = cp.sum(w)
        v = cp.sum(w[:, None] * self.X[idx], axis=0)
        variance = q - v @ inverse @ v
        if not bool(cp.isfinite(variance).get()) or float(variance) <= 4096 * np.finfo(
            float
        ).eps * abs(float(q)):
            self.stats["compact_variance_replays"] += 1
            raise ResidentDeclined("Observed variance requires reference scoring")
        self.stats["gpu_observed_calls"] += 1
        return float(s / cp.sqrt(variance))
