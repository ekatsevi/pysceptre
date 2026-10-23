"""Experimental integer-count GPU MLE. Original MM/pilot replay remains required."""
import numpy as np
import cupy as cp
from theta import code
# Use exact special-function identities, with a more stable algebraic form.
fastcode=code.replace('s=psi[g*width+(int)yt]-psi0[g]+log(th)+1-log(mt)-(yt+th)/mt;','s=psi[g*width+(int)yt]-log1p(m/th)+(m-yt)/mt;').replace('v=tri0[g]-tri[g*width+(int)yt]-1/th+2/mt-(yt+th)/(mt*mt);','v=tri[g*width+(int)yt]-(m*m+th*yt)/(th*mt*mt);')
kernel=cp.RawKernel(fastcode,'moments',options=('--fmad=false',))

def fit(dy,dm,width):
    G,n=dy.shape;blocks=(n+255)//256
    theta=n/cp.sum((dy/dm-1)**2,axis=1);delta=cp.ones(G)
    its=cp.zeros(G,cp.int32);active=cp.ones(G,cp.bool_)
    counts=cp.arange(width-1)[None,:];zero=cp.zeros((G,1));dummy=cp.zeros(G)
    ss=cp.empty((G,blocks));ii=cp.empty_like(ss)
    for _ in range(50):
        its+=active
        active&=(its<50)&(cp.abs(delta)>np.finfo(float).eps**.25)
        if not bool(cp.any(active).get()):break
        theta=cp.where(active,cp.abs(theta),theta)
        reciprocal=1/(theta[:,None]+counts)
        h1=cp.concatenate((zero,cp.cumsum(reciprocal,axis=1)),axis=1)
        h2=cp.concatenate((zero,cp.cumsum(reciprocal*reciprocal,axis=1)),axis=1)
        kernel((blocks,G),(256,),(dy,dm,theta,h1,h2,dummy,dummy,ss,ii,np.int32(n),np.int32(width),np.int32(blocks)))
        step=cp.sum(ss,axis=1)/cp.sum(ii,axis=1)
        delta=cp.where(active,step,delta);theta+=cp.where(active,delta,0)
    failed=(theta<0)|(its==50)|~cp.isfinite(theta)
    return theta,failed,its
