"""Experimental integer-count GPU MLE. Original MM/pilot replay remains required."""

import cupy as cp
import numpy as np

code = '\nextern "C" __global__ void moments(const double* y,const double* mu,const double* theta,const double* psi,const double* tri,const double* psi0,const double* tri0,double* os,double* oi,int n,int width,int blocks){\n int g=blockIdx.y;int j=blockIdx.x*256+threadIdx.x;int t=threadIdx.x;\n __shared__ double ss[256],ii[256];double s=0,v=0;\n if(j<n){double yt=y[(long long)g*n+j],m=mu[(long long)g*n+j],th=theta[g],mt=m+th;\n s=psi[g*width+(int)yt]-psi0[g]+log(th)+1-log(mt)-(yt+th)/mt;\n v=tri0[g]-tri[g*width+(int)yt]-1/th+2/mt-(yt+th)/(mt*mt);\n }\n ss[t]=s;ii[t]=v;__syncthreads();\n for(int d=128;d>0;d>>=1){if(t<d){ss[t]+=ss[t+d];ii[t]+=ii[t+d];}__syncthreads();}\n if(t==0){os[g*blocks+blockIdx.x]=ss[0];oi[g*blocks+blockIdx.x]=ii[0];}\n}\n'
# Use exact special-function identities, with a more stable algebraic form.
fastcode = code.replace(
    "s=psi[g*width+(int)yt]-psi0[g]+log(th)+1-log(mt)-(yt+th)/mt;",
    "s=psi[g*width+(int)yt]-log1p(m/th)+(m-yt)/mt;",
).replace(
    "v=tri0[g]-tri[g*width+(int)yt]-1/th+2/mt-(yt+th)/(mt*mt);",
    "v=tri[g*width+(int)yt]-(m*m+th*yt)/(th*mt*mt);",
)
kernel = cp.RawKernel(fastcode, "moments", options=("--fmad=false",))


def fit(dy, dm, width):
    G, n = dy.shape
    blocks = (n + 255) // 256
    theta = n / cp.sum((dy / dm - 1) ** 2, axis=1)
    delta = cp.ones(G)
    its = cp.zeros(G, cp.int32)
    active = cp.ones(G, cp.bool_)
    counts = cp.arange(width - 1)[None, :]
    zero = cp.zeros((G, 1))
    dummy = cp.zeros(G)
    ss = cp.empty((G, blocks))
    ii = cp.empty_like(ss)
    for _ in range(50):
        its += active
        active &= (its < 50) & (cp.abs(delta) > np.finfo(float).eps ** 0.25)
        if not bool(cp.any(active).get()):
            break
        theta = cp.where(active, cp.abs(theta), theta)
        reciprocal = 1 / (theta[:, None] + counts)
        h1 = cp.concatenate((zero, cp.cumsum(reciprocal, axis=1)), axis=1)
        h2 = cp.concatenate((zero, cp.cumsum(reciprocal * reciprocal, axis=1)), axis=1)
        kernel(
            (blocks, G),
            (256,),
            (
                dy,
                dm,
                theta,
                h1,
                h2,
                dummy,
                dummy,
                ss,
                ii,
                np.int32(n),
                np.int32(width),
                np.int32(blocks),
            ),
        )
        step = cp.sum(ss, axis=1) / cp.sum(ii, axis=1)
        delta = cp.where(active, step, delta)
        theta += cp.where(active, delta, 0)
    failed = (
        (theta < 0)
        | (its == 50)
        | ~cp.isfinite(theta)
        | (
            cp.abs(cp.abs(delta) - np.finfo(float).eps ** 0.25)
            <= 1e-10 * np.finfo(float).eps ** 0.25
        )
    )
    return theta, failed, its
