"""Complete-family collection and the owning R procedure's global BH correction."""

import numpy as np

_PREFIX_MIN = r"""
extern "C" __global__ void prefix_min(const double* input,double* output,double* blocks,long long n){
 int t=threadIdx.x,lane=t&31,warp=t/32;long long i=(long long)blockIdx.x*256+t;
 double x=i<n?input[i]:__longlong_as_double(0x7ff0000000000000ULL);
 for(int d=1;d<32;d<<=1){double other=__shfl_up_sync(0xffffffff,x,d);if(lane>=d)x=other<x?other:x;}
 __shared__ double partial[8];if(lane==31)partial[warp]=x;__syncthreads();
 if(warp==0){double y=lane<8?partial[lane]:__longlong_as_double(0x7ff0000000000000ULL);
  for(int d=1;d<8;d<<=1){double other=__shfl_up_sync(0xffffffff,y,d);if(lane>=d)y=other<y?other:y;}
  if(lane<8)partial[lane]=y;
 }__syncthreads();
 if(warp){double other=partial[warp-1];x=other<x?other:x;}
 if(i<n)output[i]=x;if(t==255)blocks[blockIdx.x]=x;
}
extern "C" __global__ void carry_min(double* output,const double* blocks,long long n){
 long long i=(long long)blockIdx.x*256+threadIdx.x;
 if(i<n&&blockIdx.x){double other=blocks[blockIdx.x-1],x=output[i];output[i]=other<x?other:x;}
}
"""


def _cuda_prefix_min(cp, values):
    """Scan finite FP64 values with bounded recursive block summaries."""
    module = cp.RawModule(code=_PREFIX_MIN, options=("--std=c++17", "--fmad=false"))
    block_scan, carry = module.get_function("prefix_min"), module.get_function("carry_min")

    def scan(array):
        """Combine block minima while retaining the rightmost equal value."""
        n = len(array)
        blocks = (n + 255) // 256
        output, minima = cp.empty_like(array), cp.empty(blocks, cp.float64)
        block_scan((blocks,), (256,), (array, output, minima, np.int64(n)))
        if blocks > 1:
            prefix = scan(minima)
            carry((blocks,), (256,), (output, prefix, np.int64(n)))
        return output

    return scan(values)


def collect_complete_pvalues(eligible_pairs, chunks):
    """Collect (gene codes, target codes, p-values), refusing incomplete families.

    Every eligible position must appear exactly once and no other position may
    appear. NaN results remain recorded members of the family. The consumer can
    distinguish missing results from explicitly returned NaNs.
    """
    eligible = np.asarray(eligible_pairs)
    if eligible.ndim != 2 or eligible.dtype != np.bool_:
        raise ValueError("Eligibility must be a boolean matrix")
    ng, nt = eligible.shape
    seen = np.zeros(ng * nt, dtype=bool)
    values = np.full(ng * nt, np.nan)
    for gene, target, p in chunks:
        gene, target, p = np.asarray(gene), np.asarray(target), np.asarray(p, dtype=float)
        if any(v.ndim != 1 for v in (gene, target, p)) or not len(gene) == len(target) == len(p):
            raise ValueError("Result columns must have equal one-dimensional shapes")
        if not all(np.issubdtype(v.dtype, np.integer) for v in (gene, target)):
            raise ValueError("Result positions must be integer codes")
        if np.any(gene < 0) or np.any(gene >= ng) or np.any(target < 0) or np.any(target >= nt):
            raise ValueError("Result position is outside the intended family")
        codes = gene.astype(np.int64) * nt + target
        if len(np.unique(codes)) != len(codes) or seen[codes].any():
            raise ValueError("Duplicate result positions")
        if not eligible.ravel()[codes].all():
            raise ValueError("Results include positions outside the eligible family")
        if np.isinf(p).any() or np.any(p < 0):
            raise ValueError("P-values must be nonnegative finite values or explicit NaNs")
        values[codes] = p
        seen[codes] = True
    if not np.array_equal(seen.reshape(eligible.shape), eligible):
        raise ValueError("The intended family is incomplete")
    return values.reshape(eligible.shape)


def global_bh(p_values):
    """Global stats::p.adjust(method='BH') semantics, including explicit NaNs.

    The default R procedure excludes NaNs from its multiplier, retains their
    output positions and caps adjusted values at one. Raw empirical two-sided
    values slightly above one are retained before this correction, as returned
    by the owning low-level test.
    """
    p = np.asarray(p_values, dtype=float)
    if np.isinf(p).any() or np.any(p < 0):
        raise ValueError("P-values must be nonnegative finite values or NaNs")
    adjusted = np.full(p.size, np.nan)
    positions = np.flatnonzero(np.isfinite(p.ravel()))
    n = len(positions)
    if n:
        order = np.argsort(p.ravel()[positions], kind="stable")
        ranked = positions[order]
        corrected = (n / np.arange(n, 0, -1, dtype=float)) * p.ravel()[ranked[::-1]]
        adjusted[ranked[::-1]] = np.minimum(1, np.minimum.accumulate(corrected))
    return adjusted.reshape(p.shape)


def global_bh_cuda(p_values, *, return_device=False, max_bytes=8 * 1024**3):
    """Explicit CUDA global BH with the CPU procedure's NaN and upper-cap rules.

    The input must represent the complete intended family. This function does
    not validate family completeness or perform pairwise QC. `max_bytes` is
    a conservative work-array estimate, not a whole-process memory limit.
    By default the adjusted array returns to CPU; `return_device` retains it
    on CUDA. Neither CPU fallback nor correction per input batch is selected.
    """
    import cupy as cp

    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("A positive CUDA correction memory bound is required")
    original = p_values if hasattr(p_values, "size") else np.asarray(p_values)
    if original.size * 80 > max_bytes:
        raise MemoryError("Estimated whole-family BH work arrays exceed max_bytes")
    p = cp.asarray(original, dtype=cp.float64)
    if bool(cp.any(cp.isinf(p) | (p < 0))):
        raise ValueError("P-values must be nonnegative finite values or NaNs")
    adjusted = cp.full(p.size, cp.nan)
    positions = cp.flatnonzero(cp.isfinite(p.ravel()))
    n = len(positions)
    if n:
        order = cp.argsort(p.ravel()[positions], kind="stable")
        ranked = positions[order][::-1]
        corrected = (n / cp.arange(n, 0, -1, dtype=cp.float64)) * p.ravel()[ranked]
        adjusted[ranked] = cp.minimum(1, _cuda_prefix_min(cp, corrected))
    result = adjusted.reshape(p.shape)
    return result if return_device else result.get()
