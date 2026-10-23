#!/bin/bash
set -euo pipefail
module load python/3.11.11 cuda/12.8.1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/numba-cache"
PYTHON="$HOME/py-sceptre-gpu-env-20261022/bin/python"
GPU_LIBRARY_PATH="$("$PYTHON" -c 'import sysconfig,pathlib; print(":".join(str(p) for p in pathlib.Path(sysconfig.get_paths()["purelib"]).glob("nvidia/*/lib")))')"
export LD_LIBRARY_PATH="$GPU_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
mkdir -p tools/gpu/evidence
nvidia-smi > tools/gpu/evidence/nvidia-smi.txt
lscpu > tools/gpu/evidence/lscpu.txt
"$PYTHON" -m pip freeze > tools/gpu/evidence/environment.txt
"$PYTHON" -m pytest -q tests/validation/test_cuda_targets.py tests/validation/test_glm_fits.py tests/validation/test_structured_irls.py tests/validation/test_low_moi.py tests/validation/test_permutations.py > tools/gpu/evidence/preflight.log 2>&1
