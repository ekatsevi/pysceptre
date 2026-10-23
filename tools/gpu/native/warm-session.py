"""Two sequential full analyses sharing validated RAM input and CUDA context."""
import importlib.util
import sys
from pathlib import Path
root,out,backend=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path.insert(0,str(root))
import adapter
original_inputs=adapter.inputs
cache={}
def inputs(*args):
    if not cache: cache['data']=original_inputs(*args)
    return cache['data']
adapter.inputs=inputs
spec=importlib.util.spec_from_file_location('workflow',Path(__file__).with_name('run-throughput.py'))
workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
for repeat in range(2):
    sys.argv=['run-throughput.py',str(root),str(out/f'r{repeat}'),'outer','--score-backend',backend,'--score-batch-size','32','--score-max-bytes','17179869184','--storage','ram']
    workflow.main()
