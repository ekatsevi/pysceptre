import os,sys,time,json
from pathlib import Path
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np
import cupy as cp
from bank import bank
from pysceptre.crt.sampler import crt_index_sampler_exact
from pysceptre.glm.irls import fit_binomial_glm_batch,x_outer_flat
root=Path('/vast/projects/ekatsevi/wharton/pysceptre-profile-20261022-2a4a')
sys.path.insert(0,str(root))
from adapter import inputs
_,_,X,targets,_,_,base=inputs(root,'backed')
if base:base.close()
xo=x_outer_flat(X)
crt_index_sampler_exact(np.full(1000,.01),50,np.random.default_rng(3))
bank(np.full(1000,.01),50,np.random.default_rng(3))
records=[]
for target,cells in list(targets.items())[:4]:
    treatment=np.zeros(len(X));treatment[cells]=1
    p=fit_binomial_glm_batch(X,treatment,X_outer_flat=xo).fitted_values
    for repeat in range(2):
        a=np.random.default_rng(271828);b=np.random.default_rng(271828)
        t=time.perf_counter();gc,gp=bank(p,5498,a);cp.cuda.Stream.null.synchronize();gt=time.perf_counter()-t
        t=time.perf_counter();expected=crt_index_sampler_exact(p,5498,b);ct=time.perf_counter()-t
        c=gc.get();ptr=gp.get();ep=np.r_[0,np.cumsum([len(x) for x in expected])]
        assert np.array_equal(ptr,ep)
        assert np.array_equal(c,np.concatenate(expected))
        assert a.bit_generator.state==b.bit_generator.state
        rec=dict(target=target,n=len(p),B=5498,p_min=float(p.min()),p_max=float(p.max()),repeat=repeat,entries=len(c),gpu_seconds=gt,cpu_seconds=ct,speedup=ct/gt,bank_exact=True,state_exact=True)
        records.append(rec);print(json.dumps(rec),flush=True)
Path('real-banks.json').write_text(json.dumps(records,indent=2)+'\n')
