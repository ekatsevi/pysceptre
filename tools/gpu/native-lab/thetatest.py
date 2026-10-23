import os,sys,time,json
from pathlib import Path
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np
import pandas as pd
import cupy as cp
from theta import fit
from pysceptre.glm.irls import fit_poisson_glm_batch,x_outer_flat
from pysceptre.glm.nb_theta import estimate_theta
root=Path('/vast/projects/ekatsevi/wharton/pysceptre-profile-20261022-2a4a');sys.path.insert(0,str(root))
from adapter import inputs
Y,genes,X,_,_,_,base=inputs(root,'backed')
ids=list(dict.fromkeys(pd.read_csv(root/'inputs/crt-discovery-pairs.csv').response_id))[:32]
y=np.array([Y[genes.index(g)].toarray().ravel() for g in ids],float)
if base:base.close()
xo=x_outer_flat(X)
mu=np.array([fit_poisson_glm_batch(X,row,X_outer_flat=xo).fitted_values for row in y])
t=time.perf_counter();ref=[estimate_theta(a,b,len(X)-X.shape[1]) for a,b in zip(y,mu)];ct=time.perf_counter()-t
out=[]
for repeat in range(3):
    t=time.perf_counter();th,failed,it=fit(y,mu);cp.cuda.Stream.null.synchronize();gt=time.perf_counter()-t
    rec=dict(repeat=repeat,cpu_seconds=ct,gpu_seconds=gt,max_abs=float(np.max(np.abs(th-np.array(ref)[:,0]))),failed=int(failed.sum()),reference_non_MLE=sum(v[1]!=1 for v in ref),theta=th.tolist(),reference=ref,iterations=it.tolist())
    out.append(rec);print(json.dumps(rec),flush=True)
Path('theta.json').write_text(json.dumps(out,indent=2)+'\n')
