#!/bin/bash
set -euo pipefail
module load python/3.11.11 cuda/12.8.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$PWD/numba-cache"
export CUPY_CACHE_DIR="$PWD/cupy-cache"
PYTHON="$HOME/py-sceptre-gpu-env-20261022/bin/python"
GPU_LIBRARY_PATH="$("$PYTHON" -c 'import sysconfig,pathlib; print(":".join(str(p) for p in pathlib.Path(sysconfig.get_paths()["purelib"]).glob("nvidia/*/lib")))')"
export LD_LIBRARY_PATH="$GPU_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
BASE=/vast/projects/ekatsevi/wharton/pysceptre-profile-20261022-2a4a
export PYTHONPATH="$PWD/src"
mkdir -p tools/gpu/evidence/score-workflow
sha256sum src/pysceptre/pipeline/*.py src/pysceptre/test_statistic/score_batch.py tools/gpu/run-score-workflow.py > tools/gpu/evidence/score-workflow/source-before.sha256
"$PYTHON" -m pytest -q tests/validation/test_score_batch.py > tools/gpu/evidence/score-workflow/final-tests.log 2>&1
nvidia-smi > tools/gpu/evidence/score-workflow/nvidia-smi.txt
lscpu > tools/gpu/evidence/score-workflow/lscpu.txt
"$PYTHON" -m pip freeze > tools/gpu/evidence/score-workflow/environment.txt
# Identical RAM inputs, fresh children; reverse the three-mode order.
for spec in r0-cpu r0-cpu_batch r0-cuda_batch r1-cuda_batch r1-cpu_batch r1-cpu; do
 mode="${spec#*-}"
 if [[ "$mode" == cpu ]]; then
  export PYTHONPATH="$BASE/source/src"
 else
  export PYTHONPATH="$PWD/src"
 fi
 /usr/bin/time -v -o "tools/gpu/evidence/score-workflow/$spec-resource.txt" "$PYTHON" tools/gpu/run-score-workflow.py "$BASE" "tools/gpu/evidence/score-workflow/$spec" outer --score-backend "$mode" --score-batch-size 8 --storage ram > "tools/gpu/evidence/score-workflow/$spec.log" 2>&1
done
export PYTHONPATH="$PWD/src"
"$PYTHON" tools/gpu/score/compare-workflow.py tools/gpu/evidence/score-workflow
sha256sum -c tools/gpu/evidence/score-workflow/source-before.sha256 > tools/gpu/evidence/score-workflow/source-after.log
