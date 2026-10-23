import copy,json,time
from pathlib import Path
import numpy as np
import cupy as cp
source=Path(__file__).with_name('rng.cu').read_text()
module=cp.RawModule(code=source,options=('--std=c++17','--fmad=false'))
raw=module.get_function('raw'); classify=module.get_function('classify'); validity=module.get_function('validity')
MASK=(1<<64)-1

def gpu_raw(state,n):
    s=state['state']['state']; inc=state['state']['inc']
    out=cp.empty(n,dtype=cp.uint64)
    raw(((n+255)//256,),(256,),(np.uint64(s&MASK),np.uint64(s>>64),np.uint64(inc&MASK),np.uint64(inc>>64),out,np.int64(n)))
    return out

def gpu_exp(rng,n):
    state=copy.deepcopy(rng.bit_generator.state)
    assert state['bit_generator']=='PCG64' and state['has_uint32']==0
    size=int(n*1.06)+1024
    while True:
        words=gpu_raw(state,size+1)
        x=cp.empty(size,cp.float64); rare=cp.empty(size,cp.uint8); accept=cp.empty_like(rare); amb=cp.empty_like(rare)
        classify(((size+255)//256,),(256,),(words,x,rare,accept,amb,np.int64(size)))
        valid=cp.empty(size,cp.uint8)
        validity(((size+255)//256,),(256,),(rare,valid,np.int64(size)))
        emitted=cp.flatnonzero((valid!=0) & (accept!=0))
        if emitted.size>=n:break
        size*=2
    ends=int(emitted[n-1].get()); consumed=ends+1+int(rare[ends].get())
    ambiguity=bool(cp.any(amb[:ends+1] & valid[:ends+1]).get())
    if ambiguity:raise RuntimeError('Ambiguous wedge decision: replay on CPU required')
    result=x[emitted[:n]]
    rng.bit_generator.advance(consumed)
    return result,consumed

