"""Bounded permutation-prefix tiles and batched observed/empirical statistics."""

from __future__ import annotations

import numpy as np

_CODE = r"""
#define P __P__
#define C (P+2)
extern "C" __global__ void prefixes(const int* pool,int total,int lo,int draws,int width,const double* planes,int cells,const int* map,double* output,unsigned int* unsafe,int ng){
 int r=blockIdx.x*blockDim.x+threadIdx.x,g=blockIdx.y;if(r>=draws||g>=ng)return;
 double s[C];
 #pragma unroll
 for(int c=0;c<C;c++)s[c]=0;
 for(int k=1;k<=map[g*(width+1)];k++){
  int cell=pool[(long long)(k-1)*total+lo+r];
  #pragma unroll
  for(int c=0;c<C;c++)s[c]+=planes[((long long)g*C+c)*cells+cell];
  int slot=map[g*(width+1)+k];
  if(slot>=0){double v=0;
   #pragma unroll
   for(int c=2;c<C;c++)v+=s[c]*s[c];
   double variance=s[1]-v,z=s[0]/sqrt(variance);output[(long long)slot*draws+r]=z;
   if(!isfinite(z)||fabs(variance)<=1e-10*(fabs(s[1])+fabs(v)))atomicOr(unsafe+slot,1u);
  }
 }
}
extern "C" __global__ void direct(const int* pool,int total,int lo,int draws,int width,const double* planes,int cells,const int* slotgene,const int* slotsize,double* output,unsigned int* unsafe,int nslots){
 int r=blockIdx.x*blockDim.x+threadIdx.x,slot=blockIdx.y+blockIdx.z*65535;if(r>=draws||slot>=nslots)return;
 int g=slotgene[slot],m=slotsize[slot];double s[C];
 #pragma unroll
 for(int c=0;c<C;c++)s[c]=0;
 for(int k=0;k<m;k++){
  int cell=pool[(long long)k*total+lo+r];
  #pragma unroll
  for(int c=0;c<C;c++)s[c]+=planes[((long long)g*C+c)*cells+cell];
 }
 double v=0;
 #pragma unroll
 for(int c=2;c<C;c++)v+=s[c]*s[c];
 double variance=s[1]-v,z=s[0]/sqrt(variance);output[(long long)slot*draws+r]=z;
 if(!isfinite(z)||fabs(variance)<=1e-10*(fabs(s[1])+fabs(v)))atomicOr(unsafe+slot,1u);
}
extern "C" __global__ void parallel_prefixes(const int* pool,int total,int lo,int draws,int width,const double* planes,int cells,const int* map,double* output,unsigned int* unsafe,int ng){
 int lane=threadIdx.x&31,r=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,g=blockIdx.y;
 if(r>=draws||g>=ng)return;
 int limit=map[g*(width+1)];double carry[C];
 #pragma unroll
 for(int c=0;c<C;c++)carry[c]=0;
 for(int base=0;base<limit;base+=32){
  int k=base+lane+1;double v[C];
  int cell=k<=limit?pool[(long long)(lo+r)*width+k-1]:0;
  #pragma unroll
  for(int c=0;c<C;c++){
   double x=k<=limit?planes[((long long)g*C+c)*cells+cell]:0;
   #pragma unroll
   for(int step=1;step<32;step*=2){double y=__shfl_up_sync(0xffffffff,x,step);if(lane>=step)x+=y;}
   v[c]=x+carry[c];carry[c]=__shfl_sync(0xffffffff,v[c],31);
  }
  int slot=k<=limit?map[g*(width+1)+k]:-1;
  if(slot>=0){double q=0;
   #pragma unroll
   for(int c=2;c<C;c++)q+=v[c]*v[c];
   double variance=v[1]-q,z=v[0]/sqrt(variance);output[(long long)slot*draws+r]=z;
   if(!isfinite(z)||fabs(variance)<=1e-10*(fabs(v[1])+fabs(q)))atomicOr(unsafe+slot,1u);
  }
 }
}
extern "C" __global__ void observed(const double* planes,const double* y,const double* mu,int cells,const int* trt,const long long* ptr,const int* pairgene,const int* pairtarget,double* z,double* fc,double* se,unsigned char* unsafe,int npairs){
 int pair=blockIdx.x,t=threadIdx.x;if(pair>=npairs)return;
 int g=pairgene[pair],target=pairtarget[pair];long long a=ptr[target],b=ptr[target+1];
 extern __shared__ double sums[];double s[C+2];
 #pragma unroll
 for(int c=0;c<C+2;c++)s[c]=0;
 for(long long k=a+t;k<b;k+=blockDim.x){int cell=trt[k];
  #pragma unroll
  for(int c=0;c<C;c++)s[c]+=planes[((long long)g*C+c)*cells+cell];
  s[C]+=y[(long long)g*cells+cell];s[C+1]+=mu[(long long)g*cells+cell];
 }
 #pragma unroll
 for(int c=0;c<C+2;c++)sums[c*blockDim.x+t]=s[c];__syncthreads();
 for(int d=blockDim.x/2;d;d>>=1){if(t<d){
  #pragma unroll
  for(int c=0;c<C+2;c++)sums[c*blockDim.x+t]+=sums[c*blockDim.x+t+d];
 }__syncthreads();}
 double f=sums[C*blockDim.x]/sums[(C+1)*blockDim.x];double e=0;
 for(long long k=a+t;k<b;k+=blockDim.x){int cell=trt[k];double v=y[(long long)g*cells+cell]-f*mu[(long long)g*cells+cell];e+=v*v;}
 __shared__ double residual[128];residual[t]=e;__syncthreads();
 for(int d=blockDim.x/2;d;d>>=1){if(t<d)residual[t]+=residual[t+d];__syncthreads();}
 if(t==0){double v=0;
  #pragma unroll
  for(int c=2;c<C;c++)v+=sums[c*blockDim.x]*sums[c*blockDim.x];
  double variance=sums[blockDim.x]-v;z[pair]=sums[0]/sqrt(variance);fc[pair]=f;se[pair]=sqrt(residual[0])/sums[(C+1)*blockDim.x];
  unsafe[pair]=!isfinite(z[pair])||fabs(variance)<=1e-10*(fabs(sums[blockDim.x])+fabs(v));
 }
}
extern "C" __global__ void empirical(const double* nulls,int draws,const int* slots,const double* observed,double* p,unsigned char* close,const unsigned int* unsafe,int npairs,int side,double guard){
 int pair=blockIdx.x,t=threadIdx.x;if(pair>=npairs)return;int slot=slots[pair];double z=observed[pair];int l=0,r=0;unsigned int near=0;
 for(int j=t;j<draws;j+=blockDim.x){double x=nulls[(long long)slot*draws+j];l+=(z>=x);r+=(z<=x);near|=!isfinite(x)||(fabs(z-x)<=guard*fmax(1.,fabs(z)));}
 __shared__ int left[128],right[128];__shared__ unsigned int nearany[128];left[t]=l;right[t]=r;nearany[t]=near;__syncthreads();
 for(int d=blockDim.x/2;d;d>>=1){if(t<d){left[t]+=left[t+d];right[t]+=right[t+d];nearany[t]|=nearany[t+d];}__syncthreads();}
 if(t==0){double a=(1.+left[0])/(1.+draws),b=(1.+right[0])/(1.+draws);p[pair]=isfinite(z)?(side<0?a:side>0?b:2*fmin(a,b)):__longlong_as_double(0x7ff8000000000000ULL);close[pair]=nearany[0]||unsafe[slot];}
}
"""


class PermutationPool:
    """One unchanged CPU-generated ordering pool and observed memberships."""

    def __init__(self, X, permutations, targets):
        import cupy as cp

        self.cp = cp
        self.cells = len(X)
        self.p = X.shape[1]
        self.total, self.width = permutations.shape
        if not 0 < self.p <= 16 or self.cells >= 2**31:
            raise ValueError("Permutation tile requires 1 to 16 covariates and int32 cell indices")
        self.order = cp.asarray(np.ascontiguousarray(permutations.T, dtype=np.int32))
        self.host_order = permutations
        self.rows_order = None
        self.design = cp.asarray(np.ascontiguousarray(X, dtype=np.float64))
        self.target_sizes = np.array([len(c) for c in targets], dtype=np.int32)
        ptr = np.r_[0, np.cumsum(self.target_sizes, dtype=np.int64)]
        self.trt = cp.asarray(np.concatenate(targets).astype(np.int32))
        self.ptr = cp.asarray(ptr)
        module = cp.RawModule(
            code=_CODE.replace("__P__", str(self.p)), options=("--std=c++17", "--fmad=false")
        )
        self.kernels = {
            n: module.get_function(n)
            for n in ["prefixes", "direct", "parallel_prefixes", "observed", "empirical"]
        }

    def tile(self, planes, y, mu):
        return PermutationTile(self, planes, y, mu)

    def empirical(self, nulls, slots, z, unsafe, side=0, guard=1e-8):
        return _empirical(self, nulls, slots, z, unsafe, side, guard)

    def tile_from_models(self, y, coefficients, theta):
        """Batched original a,w,D algebra from the explicitly selected fitted models."""
        cp = self.cp
        dy = cp.asarray(np.ascontiguousarray(y, dtype=np.float64))
        mu = cp.exp(cp.asarray(coefficients) @ self.design.T)
        denominator = 1 + mu / cp.asarray(theta)[:, None]
        w = mu / denominator
        wx = w[:, :, None] * self.design[None, :, :]
        hessian = cp.matmul(self.design.T[None, :, :], wx)
        eigenvalues, eigenvectors = cp.linalg.eigh(hessian)
        invalid = (~cp.isfinite(eigenvalues).all(axis=1)) | (
            eigenvalues[:, 0] <= eigenvalues[:, -1] * self.p * np.finfo(float).eps
        )
        factor = eigenvectors.transpose(0, 2, 1) / cp.sqrt(eigenvalues)[:, :, None]
        D = cp.matmul(factor, wx.transpose(0, 2, 1))
        planes = cp.concatenate(
            ((dy - mu)[:, None, :] / denominator[:, None, :], w[:, None, :], D), axis=1
        )
        return PermutationTile(self, planes, dy, mu), invalid.get()

    @property
    def device_bytes(self):
        return int(
            self.order.nbytes
            + self.design.nbytes
            + self.trt.nbytes
            + self.ptr.nbytes
            + (0 if self.rows_order is None else self.rows_order.nbytes)
        )


class PermutationTile:
    def __init__(self, pool, planes, y, mu):
        self.pool = pool
        self.cp = pool.cp
        self.planes = self.cp.ascontiguousarray(self.cp.asarray(planes, dtype=self.cp.float64))
        self.y = self.cp.ascontiguousarray(self.cp.asarray(y, dtype=self.cp.float64))
        self.mu = self.cp.ascontiguousarray(self.cp.asarray(mu, dtype=self.cp.float64))
        self.genes = len(y)
        if self.planes.shape != (self.genes, pool.p + 2, pool.cells):
            raise ValueError("Plane shape mismatch")

    def nulls(self, requested_sizes, lo, hi, strategy="prefix"):
        cp = self.cp
        pool = self.pool
        draws = hi - lo
        if lo < 0 or hi < lo or hi > pool.total or len(requested_sizes) != self.genes:
            raise ValueError("Null request must fit the shared pool and gene tile")
        if strategy not in ("prefix", "direct", "parallel_prefix"):
            raise ValueError("Unknown permutation strategy")
        mapping = np.full((self.genes, pool.width + 1), -1, np.int32)
        slot_gene = []
        slot_size = []
        for g, sizes in enumerate(requested_sizes):
            for size in sorted(set(map(int, sizes))):
                if not 0 < size <= pool.width:
                    raise ValueError("Target size outside shared pool")
                mapping[g, size] = len(slot_size)
                slot_gene.append(g)
                slot_size.append(size)
        output = cp.empty((len(slot_size), draws), cp.float64)
        unsafe = cp.zeros(len(slot_size), cp.uint32)
        for g in range(self.genes):
            occupied = np.flatnonzero(mapping[g, 1:] >= 0)
            mapping[g, 0] = int(occupied[-1] + 1) if len(occupied) else 0
        if len(slot_size) and draws:
            if strategy == "parallel_prefix" and pool.rows_order is None:
                pool.rows_order = cp.asarray(np.ascontiguousarray(pool.host_order, dtype=np.int32))
            common = (
                pool.rows_order if strategy == "parallel_prefix" else pool.order,
                np.int32(pool.total),
                np.int32(lo),
                np.int32(draws),
                np.int32(pool.width),
                self.planes,
                np.int32(pool.cells),
            )
            if strategy == "prefix":
                pool.kernels["prefixes"](
                    ((draws + 127) // 128, self.genes),
                    (128,),
                    (*common, cp.asarray(mapping), output, unsafe, np.int32(self.genes)),
                )
            elif strategy == "parallel_prefix":
                pool.kernels["parallel_prefixes"](
                    ((draws + 3) // 4, self.genes),
                    (128,),
                    (*common, cp.asarray(mapping), output, unsafe, np.int32(self.genes)),
                )
            else:
                pool.kernels["direct"](
                    (
                        (draws + 127) // 128,
                        min(len(slot_size), 65535),
                        (len(slot_size) + 65534) // 65535,
                    ),
                    (128,),
                    (
                        *common,
                        cp.asarray(slot_gene, dtype=cp.int32),
                        cp.asarray(slot_size, dtype=cp.int32),
                        output,
                        unsafe,
                        np.int32(len(slot_size)),
                    ),
                )
        return (
            output,
            mapping,
            np.asarray(slot_gene, np.int32),
            np.asarray(slot_size, np.int32),
            unsafe,
        )

    def observed(self, pair_gene, pair_target):
        cp = self.cp
        p = self.pool
        n = len(pair_gene)
        z, fc, se = [cp.empty(n, cp.float64) for _ in range(3)]
        unsafe = cp.empty(n, cp.uint8)
        if n:
            p.kernels["observed"](
                (n,),
                (128,),
                (
                    self.planes,
                    self.y,
                    self.mu,
                    np.int32(p.cells),
                    p.trt,
                    p.ptr,
                    cp.asarray(pair_gene, dtype=cp.int32),
                    cp.asarray(pair_target, dtype=cp.int32),
                    z,
                    fc,
                    se,
                    unsafe,
                    np.int32(n),
                ),
                shared_mem=(p.p + 4) * 128 * 8,
            )
        return z, fc, se, unsafe

    def empirical(self, nulls, slots, z, unsafe, side=0, guard=1e-8):
        return _empirical(self.pool, nulls, slots, z, unsafe, side, guard)


def _empirical(pool, nulls, slots, z, unsafe, side, guard):
    cp = pool.cp
    n = len(slots)
    p = cp.empty(n, cp.float64)
    close = cp.empty(n, cp.uint8)
    if n:
        pool.kernels["empirical"](
            (n,),
            (128,),
            (
                nulls,
                np.int32(nulls.shape[1]),
                cp.asarray(slots, dtype=cp.int32),
                z,
                p,
                close,
                unsafe,
                np.int32(n),
                np.int32(side),
                np.float64(guard),
            ),
        )
    return p, close
