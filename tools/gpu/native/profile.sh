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
export PYTHONPATH="$PWD/src"
mkdir -p tools/gpu/evidence/native-profile
"$PYTHON" -m cProfile -o tools/gpu/evidence/native-profile/profile.pstats tools/gpu/native/profile-public.py /vast/projects/ekatsevi/wharton/pysceptre-profile-20261022-2a4a tools/gpu/evidence/native-profile/wide outer --score-backend cuda_native --score-batch-size 256 --score-max-bytes 68719476736 --storage ram > tools/gpu/evidence/native-profile/run.log 2>&1
"$PYTHON" -c 'import pstats; pstats.Stats("tools/gpu/evidence/native-profile/profile.pstats").sort_stats("cumtime").print_stats(65)' > tools/gpu/evidence/native-profile/profile.txt
