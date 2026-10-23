#!/bin/bash
#SBATCH --job-name=pysceptre-gpu-memory
#SBATCH --account=ekatsevi-wharton
#SBATCH --qos=wharton-dgx-b200
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:08:00
set -euo pipefail
bash -l tools/gpu/run-memory.sh
