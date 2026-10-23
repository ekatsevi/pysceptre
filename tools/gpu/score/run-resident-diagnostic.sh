#!/bin/bash
#SBATCH --job-name=pysceptre-resident-diagnostic
#SBATCH --account=ekatsevi-wharton
#SBATCH --qos=wharton-dgx-b200
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:08:00
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
mkdir -p tools/gpu/evidence/resident
sha256sum tools/gpu/score/sufficient_fit.py tools/gpu/score/resident-fit-diagnostic.py > tools/gpu/evidence/resident/source.sha256
"$PYTHON" tools/gpu/score/resident-fit-diagnostic.py /vast/projects/ekatsevi/wharton/pysceptre-profile-20261022-2a4a tools/gpu/evidence/resident/diagnostic.json > tools/gpu/evidence/resident/run.log 2>&1
sha256sum -c tools/gpu/evidence/resident/source.sha256 > tools/gpu/evidence/resident/source-readback.log
