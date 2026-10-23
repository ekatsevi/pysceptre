"""Experimental packed draw listings and coalesced compact score tiles.

The listing order and multiplicities are retained exactly. This is a score
primitive; the owning scheduler remains responsible for models, sampling,
adaptive stages, numerical replays and complete-family collection.
"""

import numpy as np

_PACK = r"""
extern "C" __global__ void pack(const unsigned int* input,long long length,int bits,unsigned int* output,long long words){
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(j>=words)return;
 long long bit=j*32;int filled=0;unsigned long long value=0;
 while(filled<32){long long index=bit/bits;if(index>=length)break;
  int shift=bit%bits,take=min(bits-shift,32-filled);
  unsigned long long mask=(1ULL<<take)-1;
  value|=((input[index]>>shift)&mask)<<filled;filled+=take;bit+=take;
 }
 output[j]=(unsigned int)value;
}
extern "C" __global__ void unpack(const unsigned int* input,long long length,int bits,int* output){
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(j>=length)return;
 long long bit=j*bits,index=bit/32;int shift=bit%32;
 unsigned long long x=input[index]|((unsigned long long)input[index+1]<<32);
 output[j]=(x>>shift)&((1ULL<<bits)-1);
}
"""

_SCORE = r"""
#define P __P__
__device__ __forceinline__ int cell_at(const unsigned int* data,long long index,int bits){
 if(bits==32)return data[index];
 long long bit=index*bits,word=bit/32;int shift=bit%32;
 unsigned long long x=data[word]|((unsigned long long)data[word+1]<<32);
 return (x>>shift)&((1ULL<<bits)-1);
}
extern "C" __global__ void put_planes(const int* genes,int count,int cells,const double* a,const double* w,double* out){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(long long)count*cells)return;
 int g=genes[i/cells],cell=i%cells,block=g/32,lane=g%32;
 out[(((long long)block*2)*cells+cell)*32+lane]=a[i];
 out[(((long long)block*2+1)*cells+cell)*32+lane]=w[i];
}
extern "C" __global__ void scores(const unsigned long long* banks,const unsigned long long* pointers,const int* bits,const double* aw,const double* inverse,const double* X,int cells,int total_genes,const int* selected,int genes,int targets,int lo,int draws,int chunks,const double* observed,const unsigned char* eligible,double* output,unsigned int* left,unsigned int* right,unsigned int* guard,int mode){
 int lane=threadIdx.x&31,warp=threadIdx.x/32,target=blockIdx.x/chunks;
 int r=(blockIdx.x%chunks)*8+warp,g=blockIdx.y*32+lane;
 bool valid=g<genes&&r<draws&&target<targets;
 int physical=valid?selected[g]:0;
 const unsigned int* data=(const unsigned int*)banks[target];
 const long long* offsets=(const long long*)pointers[target];
 long long outindex=(long long)target*genes+g;
 if(valid&&mode&& !eligible[outindex])valid=false;
 double s=0,q=0,u[P];
 #pragma unroll
 for(int k=0;k<P;k++)u[k]=0;
 if(valid){
  int gb=physical/32,gl=physical%32;
  const double* a=aw+(long long)gb*2*cells*32;
  for(long long j=offsets[lo+r];j<offsets[lo+r+1];j++){
   int c=cell_at(data,j,bits[target]);double w=a[((long long)cells+c)*32+gl];
   s+=a[(long long)c*32+gl];q+=w;
   #pragma unroll
   for(int k=0;k<P;k++)u[k]+=w*X[(long long)c*P+k];
  }
 }
 double projection=0;
 #pragma unroll
 for(int k=0;k<P;k++){
  #pragma unroll
  for(int l=0;l<P;l++)projection+=u[k]*inverse[((long long)physical*P+k)*P+l]*u[l];
 }
 double variance=q-projection,z=s/sqrt(variance);
 unsigned int danger=valid&&(!isfinite(z)||fabs(variance)<=1e-10*(fabs(q)+fabs(projection)));
 if(!mode){if(valid){output[((long long)target*draws+r)*genes+g]=z;if(danger)atomicOr(guard+outindex,1u);}return;}
 double observed_z=valid?observed[outindex]:0;
 unsigned int a=valid&&(observed_z>=z),b=valid&&(observed_z<=z);
 danger|=valid&&(!isfinite(observed_z)||fabs(observed_z-z)<=1e-8*fmax(1.,fabs(observed_z)));
 __shared__ unsigned int L[256],R[256],D[256];int t=threadIdx.x;
 L[t]=a;R[t]=b;D[t]=danger;__syncthreads();
 for(int d=4;d;d>>=1){if(warp<d){L[t]+=L[t+d*32];R[t]+=R[t+d*32];D[t]|=D[t+d*32];}__syncthreads();}
 if(warp==0&&g<genes){atomicAdd(left+outindex,L[t]);atomicAdd(right+outindex,R[t]);if(D[t])atomicOr(guard+outindex,1u);}
}
extern "C" __global__ void queries(const unsigned long long* banks,const unsigned long long* pointers,const int* bits,const int* targets,const int* genes,const int* slots,const double* aw,const double* inverse,const double* X,int cells,int lo,int draws,int chunks,int width,int count,double* output,unsigned int* guard){
 int lane=threadIdx.x&31,warp=threadIdx.x/32,group=blockIdx.x/chunks;
 int r=(blockIdx.x%chunks)*8+warp,g=blockIdx.y*32+lane;
 if(g>=width||r>=draws)return;
 int slot=slots[group*width+g];if(slot<0)return;
 int physical=genes[group*width+g],target=targets[group];
 const unsigned int* data=(const unsigned int*)banks[target];
 const long long* offsets=(const long long*)pointers[target];
 const double* a=aw+(long long)(physical/32)*2*cells*32;int gl=physical%32;
 double s=0,q=0,u[P];
 #pragma unroll
 for(int k=0;k<P;k++)u[k]=0;
 for(long long j=offsets[lo+r];j<offsets[lo+r+1];j++){
  int c=cell_at(data,j,bits[target]);double w=a[((long long)cells+c)*32+gl];
  s+=a[(long long)c*32+gl];q+=w;
  #pragma unroll
  for(int k=0;k<P;k++)u[k]+=w*X[(long long)c*P+k];
 }
 double projection=0;
 #pragma unroll
 for(int k=0;k<P;k++){
  #pragma unroll
  for(int l=0;l<P;l++)projection+=u[k]*inverse[((long long)physical*P+k)*P+l]*u[l];
 }
 double variance=q-projection,z=s/sqrt(variance);output[(long long)r*count+slot]=z;
 if(!isfinite(z)||fabs(variance)<=1e-10*(fabs(q)+fabs(projection)))atomicOr(guard+slot,1u);
}
"""


class PackedDraws:
    """Lossless fixed-bit encoding of arbitrary ordered draw listings."""

    def __init__(self, cells, offsets, n_cells, *, packed=True):
        import cupy as cp

        self.cp = cp
        original = cp.asarray(cells)
        if original.ndim != 1 or not np.issubdtype(original.dtype, np.integer):
            raise ValueError("Draw listings must be a one-dimensional integer array")
        if n_cells < 1 or n_cells >= 2**31:
            raise ValueError("Int32 cell positions are required")
        if len(original) and bool(cp.any((original < 0) | (original >= n_cells))):
            raise ValueError("Draw cell outside the supplied cell axis")
        original_offsets = cp.asarray(offsets)
        if not np.issubdtype(original_offsets.dtype, np.integer):
            raise ValueError("Draw offsets must be integer positions")
        self.offsets = cp.asarray(original_offsets, dtype=cp.int64)
        host_offsets = self.offsets.get()
        if (
            host_offsets.ndim != 1
            or not len(host_offsets)
            or host_offsets[0] != 0
            or host_offsets[-1] != len(original)
            or np.any(np.diff(host_offsets) < 0)
        ):
            raise ValueError("Offsets must partition the exact draw listing array")
        self.n_cells = n_cells
        self.length = len(original)
        self.n_draws = len(host_offsets) - 1
        self.bits = max(1, (n_cells - 1).bit_length()) if packed else 32
        raw = cp.ascontiguousarray(original, dtype=cp.uint32)
        module = cp.RawModule(code=_PACK, options=("--std=c++17",))
        self.unpack_kernel = module.get_function("unpack")
        if packed:
            words = (self.length * self.bits + 31) // 32
            self.data = cp.zeros(words + 1, cp.uint32)
            if words:
                module.get_function("pack")(
                    ((words + 255) // 256,),
                    (256,),
                    (raw, np.int64(self.length), np.int32(self.bits), self.data, np.int64(words)),
                )
        else:
            self.data = raw

    @property
    def nbytes(self):
        return self.data.nbytes + self.offsets.nbytes

    def unpack(self):
        if self.bits == 32:
            return self.data.astype(self.cp.int32)
        output = self.cp.empty(self.length, self.cp.int32)
        if self.length:
            self.unpack_kernel(
                ((self.length + 255) // 256,),
                (256,),
                (self.data, np.int64(self.length), np.int32(self.bits), output),
            )
        return output


class CompactScoreAtlas:
    """a,w in contiguous 32-gene blocks, then component, cell and SIMD lane."""

    def __init__(self, X, inverses):
        import cupy as cp

        self.cp = cp
        self.X = cp.ascontiguousarray(cp.asarray(X, dtype=cp.float64))
        self.inverse = cp.ascontiguousarray(cp.asarray(inverses, dtype=cp.float64))
        self.cells, self.p = self.X.shape
        self.genes = len(self.inverse)
        if not 0 < self.p <= 16 or self.inverse.shape != (self.genes, self.p, self.p):
            raise ValueError("Inverse information matrices must align with the design")
        self.aw = cp.empty(((self.genes + 31) // 32, 2, self.cells, 32), cp.float64)
        self.filled = np.zeros(self.genes, dtype=bool)
        module = cp.RawModule(
            code=_SCORE.replace("__P__", str(self.p)), options=("--std=c++17", "--fmad=false")
        )
        self.put = module.get_function("put_planes")
        self.kernel = module.get_function("scores")
        self.query_kernel = module.get_function("queries")

    @property
    def nbytes(self):
        return self.aw.nbytes + self.inverse.nbytes + self.X.nbytes

    def set_genes(self, genes, a, w):
        cp = self.cp
        genes = np.asarray(genes)
        if (
            genes.ndim != 1
            or not np.issubdtype(genes.dtype, np.integer)
            or np.any(genes < 0)
            or np.any(genes >= self.genes)
            or len(np.unique(genes)) != len(genes)
        ):
            raise ValueError("Distinct in-range gene positions are required")
        a, w = cp.asarray(a, dtype=cp.float64), cp.asarray(w, dtype=cp.float64)
        if a.shape != (len(genes), self.cells) or w.shape != a.shape:
            raise ValueError("Score planes must align with selected genes and cells")
        n = len(genes) * self.cells
        if n:
            self.put(
                ((n + 255) // 256,),
                (256,),
                (
                    cp.asarray(genes, dtype=cp.int32),
                    np.int32(len(genes)),
                    np.int32(self.cells),
                    cp.ascontiguousarray(a),
                    cp.ascontiguousarray(w),
                    self.aw,
                ),
            )
        self.filled[genes] = True


class PackedDrawCollection:
    """One launch over banks, draw tiles and cache-sized contiguous gene blocks."""

    def __init__(self, banks):
        import cupy as cp

        self.cp = cp
        self.banks = list(banks)
        if not self.banks or len({b.n_cells for b in self.banks}) != 1:
            raise ValueError("Nonempty banks with one cell axis are required")
        self.data = cp.asarray([b.data.data.ptr for b in self.banks], dtype=cp.uint64)
        self.offsets = cp.asarray([b.offsets.data.ptr for b in self.banks], dtype=cp.uint64)
        self.bits = cp.asarray([b.bits for b in self.banks], dtype=cp.int32)

    @property
    def nbytes(self):
        return (
            sum(b.nbytes for b in self.banks)
            + self.data.nbytes
            + self.offsets.nbytes
            + self.bits.nbytes
        )

    def _run(self, atlas, lo, hi, selected, observed, eligible, empirical):
        cp = self.cp
        selected = np.arange(atlas.genes) if selected is None else np.asarray(selected)
        if (
            selected.ndim != 1
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0)
            or np.any(selected >= atlas.genes)
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError("Distinct selected genes are required")
        if not atlas.filled[selected].all() or any(
            b.n_cells != atlas.cells or not 0 <= lo <= hi <= b.n_draws for b in self.banks
        ):
            raise ValueError("Missing score planes or an invalid draw interval")
        t, g, draws = len(self.banks), len(selected), hi - lo
        z = (
            cp.zeros((t, g))
            if observed is None
            else cp.ascontiguousarray(cp.asarray(observed, dtype=cp.float64))
        )
        mask = (
            cp.ones((t, g), cp.uint8)
            if eligible is None
            else cp.ascontiguousarray(cp.asarray(eligible, dtype=cp.uint8))
        )
        if z.shape != (t, g) or mask.shape != z.shape:
            raise ValueError("Observed scores and eligibility must align with bank and gene axes")
        output = cp.empty((0,) if empirical else (t, draws, g), cp.float64)
        left, right, guard = [cp.zeros((t, g), cp.uint32) for _ in range(3)]
        if draws and g:
            chunks = (draws + 7) // 8
            atlas.kernel(
                (chunks * t, (g + 31) // 32),
                (256,),
                (
                    self.data,
                    self.offsets,
                    self.bits,
                    atlas.aw,
                    atlas.inverse,
                    atlas.X,
                    np.int32(atlas.cells),
                    np.int32(atlas.genes),
                    cp.asarray(selected, dtype=cp.int32),
                    np.int32(g),
                    np.int32(t),
                    np.int32(lo),
                    np.int32(draws),
                    np.int32(chunks),
                    z,
                    mask,
                    output,
                    left,
                    right,
                    guard,
                    np.int32(empirical),
                ),
            )
        return output, left, right, guard, z, mask

    def statistics(self, atlas, lo, hi, *, selected=None, max_bytes=1024**3):
        genes = atlas.genes if selected is None else len(selected)
        if not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("A positive output memory bound is required")
        if len(self.banks) * genes * max(0, hi - lo) * 8 > max_bytes:
            raise MemoryError("Requested CRT null rectangle exceeds max_bytes")
        out, _, _, guard, _, _ = self._run(atlas, lo, hi, selected, None, None, False)
        return out.transpose(0, 2, 1), guard

    def statistics_queries(self, atlas, gene, target, lo, hi, *, max_bytes=1024**3):
        """Compact requested pairs into one distribution batch, preserving query order."""
        cp = self.cp
        gene, target = np.asarray(gene), np.asarray(target)
        if (
            gene.ndim != 1
            or target.shape != gene.shape
            or any(not np.issubdtype(a.dtype, np.integer) for a in (gene, target))
            or np.any(gene < 0)
            or np.any(gene >= atlas.genes)
            or np.any(target < 0)
            or np.any(target >= len(self.banks))
        ):
            raise ValueError(
                "Aligned in-range integer gene and target query positions are required"
            )
        if not atlas.filled[gene].all() or any(
            b.n_cells != atlas.cells or not 0 <= lo <= hi <= b.n_draws for b in self.banks
        ):
            raise ValueError("Missing planes or invalid draw interval")
        if len(set(zip(gene.tolist(), target.tolist(), strict=True))) != len(gene):
            raise ValueError("Repeated query pairs are not supported")
        draws, count = hi - lo, len(gene)
        if not isinstance(max_bytes, int) or max_bytes < 1 or draws * count * 8 > max_bytes:
            raise MemoryError("Requested compact null batch exceeds max_bytes")
        named, counts = np.unique(target, return_counts=True)
        width = int(counts.max()) if count else 0
        slots = np.full((len(named), width), -1, np.int32)
        selected = np.zeros_like(slots)
        for j, t in enumerate(named):
            at = np.flatnonzero(target == t)
            slots[j, : len(at)] = at
            selected[j, : len(at)] = gene[at]
        output, guard = cp.empty((draws, count)), cp.zeros(count, cp.uint32)
        if draws and count:
            chunks = (draws + 7) // 8
            atlas.query_kernel(
                (len(named) * chunks, (width + 31) // 32),
                (256,),
                (
                    self.data,
                    self.offsets,
                    self.bits,
                    cp.asarray(named, dtype=cp.int32),
                    cp.asarray(selected),
                    cp.asarray(slots),
                    atlas.aw,
                    atlas.inverse,
                    atlas.X,
                    np.int32(atlas.cells),
                    np.int32(lo),
                    np.int32(draws),
                    np.int32(chunks),
                    np.int32(width),
                    np.int32(count),
                    output,
                    guard,
                ),
            )
        return cp.ascontiguousarray(output.T), guard

    def empirical(self, atlas, observed, lo, hi, *, selected=None, eligible=None, side=0):
        if side not in (-1, 0, 1):
            raise ValueError("Unknown tail side")
        _, left, right, guard, z, mask = self._run(
            atlas, lo, hi, selected, observed, eligible, True
        )
        a, b = (1.0 + left) / (1.0 + hi - lo), (1.0 + right) / (1.0 + hi - lo)
        p = a if side < 0 else b if side > 0 else 2 * self.cp.minimum(a, b)
        p = self.cp.where(self.cp.isfinite(z) & (mask != 0), p, self.cp.nan)
        return p, guard
