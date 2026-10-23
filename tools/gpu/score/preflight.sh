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
mkdir -p tools/gpu/evidence/score
"$PYTHON" -m pytest -q tests/validation/test_score_batch.py tests/validation/test_low_moi.py tests/validation/test_permutations.py > tools/gpu/evidence/score/preflight.log 2>&1
"$PYTHON" tools/gpu/score/diagnostic.py tools/gpu/evidence/score/batch-sweep.json
if command -v nsys >/dev/null; then
 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop -o tools/gpu/evidence/score/timeline "$PYTHON" tools/gpu/score/diagnostic.py tools/gpu/evidence/score/profile-check.json --profile-only > tools/gpu/evidence/score/nsys.log 2>&1 || true
 if test -f tools/gpu/evidence/score/timeline.nsys-rep; then
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum --format csv tools/gpu/evidence/score/timeline.nsys-rep > tools/gpu/evidence/score/nsys-stats.csv 2>&1 || true
 fi
fi
if command -v ncu >/dev/null; then
 ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --launch-skip 2 --launch-count 1 --target-processes all --csv --log-file tools/gpu/evidence/score/ncu.csv "$PYTHON" tools/gpu/score/diagnostic.py tools/gpu/evidence/score/ncu-check.json --profile-only > tools/gpu/evidence/score/ncu.log 2>&1 || true
fi
