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
mkdir -p tools/gpu/evidence/paired
export PYTHONPATH="$PWD/src"
"$PYTHON" -m pytest -q tests/validation/test_cuda_targets.py > tools/gpu/evidence/paired/final-cuda-tests.log 2>&1
"$PYTHON" tools/gpu/compare-default-cpu.py "$BASE/source/src/pysceptre/glm/irls.py" tools/gpu/evidence/paired/exact-default-cpu.json
"$PYTHON" tools/gpu/fit-diagnostic.py "$BASE/inputs" tools/gpu/evidence/paired
sha256sum src/pysceptre/glm/*.py src/pysceptre/pipeline/discovery.py tools/gpu/run-frozen.py > tools/gpu/evidence/paired/source-before.sha256
nvidia-smi > tools/gpu/evidence/paired/nvidia-smi.txt
"$PYTHON" -m pip freeze > tools/gpu/evidence/paired/environment.txt
# Each mode is a fresh child. Reverse order for the second pair.
for spec in r0-cpu r0-gpu r1-gpu r1-cpu; do
  if [[ "$spec" == *cpu ]]; then
    export PYTHONPATH="$BASE/source/src"
    solver=outer
  else
    export PYTHONPATH="$PWD/src"
    solver=cuda_targets
  fi
  /usr/bin/time -v -o "tools/gpu/evidence/paired/$spec-resource.txt" "$PYTHON" tools/gpu/run-frozen.py "$BASE" "tools/gpu/evidence/paired/$spec" "$solver" --storage backed > "tools/gpu/evidence/paired/$spec.log" 2>&1
done
export PYTHONPATH="$PWD/src"
"$PYTHON" tools/gpu/compare.py tools/gpu/evidence/paired
sha256sum -c tools/gpu/evidence/paired/source-before.sha256 > tools/gpu/evidence/paired/source-after.log
