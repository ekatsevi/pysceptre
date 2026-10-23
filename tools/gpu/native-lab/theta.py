"""Experimental batched MLE only; failed/sensitive genes require reference replay."""
import numpy as np
import cupy as cp
from scipy.special import digamma,polygamma
code=r'''
extern "C" __global__ void moments(const double* y,const double* mu,const double* theta,const double* psi,const double* tri,const double* psi0,const double* tri0,double* os,double* oi,int n,int width,int blocks){
 int g=blockIdx.y;int j=blockIdx.x*256+threadIdx.x;int t=threadIdx.x;
 __shared__ double ss[256],ii[256];double s=0,v=0;
 if(j<n){double yt=y[(long long)g*n+j],m=mu[(long long)g*n+j],th=theta[g],mt=m+th;
 s=psi[g*width+(int)yt]-psi0[g]+log(th)+1-log(mt)-(yt+th)/mt;
 v=tri0[g]-tri[g*width+(int)yt]-1/th+2/mt-(yt+th)/(mt*mt);
 }
 ss[t]=s;ii[t]=v;__syncthreads();
 for(int d=128;d>0;d>>=1){if(t<d){ss[t]+=ss[t+d];ii[t]+=ii[t+d];}__syncthreads();}
 if(t==0){os[g*blocks+blockIdx.x]=ss[0];oi[g*blocks+blockIdx.x]=ii[0];}
}
'''
kernel=cp.RawKernel(code,'moments',options=('--fmad=false',))

def fit(y,mu):
    # Integer counts only; no claim for fractional-response inputs.
    assert np.all(y==y.astype(np.int64)) and np.min(y)>=0
    G,n=y.shape; width=int(y.max())+1; blocks=(n+255)//256
    dy=cp.asarray(y);dm=cp.asarray(mu)
    theta=cp.asnumpy(n/cp.sum((dy/dm-1)**2,axis=1));delta=np.ones(G)
    its=np.zeros(G,int);active=np.ones(G,bool)
    for _ in range(50):
        its[active]+=1
        active&=(its<50)&(np.abs(delta)>np.finfo(float).eps**.25)
        if not active.any():break
        theta[active]=np.abs(theta[active])
        # Small special-function tables retain SciPy's implementation.
        z=theta[:,None]+np.arange(width)[None,:]
        psi=cp.asarray(digamma(z));tri=cp.asarray(polygamma(1,z))
        psi0=cp.asarray(digamma(theta));tri0=cp.asarray(polygamma(1,theta));dt=cp.asarray(theta)
        ss=cp.empty((G,blocks));ii=cp.empty_like(ss)
        kernel((blocks,G),(256,),(dy,dm,dt,psi,tri,psi0,tri0,ss,ii,np.int32(n),np.int32(width),np.int32(blocks)))
        step=cp.asnumpy(cp.sum(ss,axis=1)/cp.sum(ii,axis=1))
        delta[active]=step[active];theta[active]+=delta[active]
    failed=(theta<0)|(its==50)|~np.isfinite(theta)
    return theta,failed,its
