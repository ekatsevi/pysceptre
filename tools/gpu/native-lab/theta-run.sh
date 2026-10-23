#!/bin/bash
#SBATCH --job-name=pysceptre-exact-rng
#SBATCH --account=ekatsevi-wharton
#SBATCH --qos=wharton-dgx-b200
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00
set -euo pipefail
module load python/3.11.11 cuda/12.8.1
PYTHON="$HOME/py-sceptre-gpu-env-20261022/bin/python"
GPU_LIBRARY_PATH="$("$PYTHON" -c 'import sysconfig,pathlib; print(":".join(str(p) for p in pathlib.Path(sysconfig.get_paths()["purelib"]).glob("nvidia/*/lib")))')"
export LD_LIBRARY_PATH="$GPU_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUPY_CACHE_DIR="$PWD/cupy-cache" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/vast/home/e/ekatsevi/pysceptre-integrated-resident-20261023/src
"$PYTHON" thetatest.py
