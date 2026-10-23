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
mkdir -p tools/gpu/evidence/memory
"$PYTHON" tools/gpu/compare-draws.py "$BASE/inputs" tools/gpu/evidence/memory/draw-comparison.json > tools/gpu/evidence/memory/draw-comparison.log 2>&1
"$PYTHON" tools/gpu/memory-diagnostic.py "$BASE" tools/gpu/evidence/memory/observed > tools/gpu/evidence/memory/controller.log 2>&1
