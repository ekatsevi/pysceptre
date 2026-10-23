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
OUT=tools/gpu/evidence/native-clean
BASE=/vast/projects/ekatsevi/wharton/pysceptre-profile-20261022-2a4a
mkdir -p "$OUT"
sha256sum $(find src/pysceptre -type f -not -path "*/__pycache__/*" | sort) tools/gpu/native/run-clean.py tools/gpu/native/clean-memory.py > "$OUT/source-before.sha256"
nvidia-smi > "$OUT/nvidia-smi.txt"
lscpu > "$OUT/lscpu.txt"
"$PYTHON" -m pip freeze > "$OUT/environment.txt"
for spec in r0-cpu r0-cuda_native r1-cuda_native r1-cpu; do
 mode="${spec#*-}"
 if [[ "$mode" == cpu ]]; then
  export PYTHONPATH="$BASE/source/src"
 else
  export PYTHONPATH="$PWD/src"
 fi
 /usr/bin/time -v -o "$OUT/$spec-resource.txt" "$PYTHON" tools/gpu/native/run-clean.py "$BASE" "$OUT/$spec" outer --score-backend "$mode" --score-batch-size 256 --score-max-bytes 68719476736 --storage ram > "$OUT/$spec.log" 2>&1
done
export PYTHONPATH="$PWD/src"
"$PYTHON" tools/gpu/native/compare-clean.py "$OUT"
"$PYTHON" tools/gpu/native/clean-memory.py "$BASE" "$OUT/memory"
sha256sum -c "$OUT/source-before.sha256" > "$OUT/source-after.log"
