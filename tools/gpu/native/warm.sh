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
OUT=tools/gpu/evidence/native-warmed
BASE=/vast/projects/ekatsevi/wharton/pysceptre-profile-20261022-2a4a
mkdir -p "$OUT"
sha256sum $(find src/pysceptre -type f -not -path "*/__pycache__/*" | sort) tools/gpu/native/run-throughput.py tools/gpu/native/warm-session.py > "$OUT/source-before.sha256"
for mode in cpu cuda_native; do
 if [[ "$mode" == cpu ]]; then export PYTHONPATH="$BASE/source/src"; else export PYTHONPATH="$PWD/src"; fi
 /usr/bin/time -v -o "$OUT/$mode-resource.txt" "$PYTHON" tools/gpu/native/warm-session.py "$BASE" "$OUT/$mode" "$mode" > "$OUT/$mode.log" 2>&1
done
sha256sum -c "$OUT/source-before.sha256" > "$OUT/source-after.log"
