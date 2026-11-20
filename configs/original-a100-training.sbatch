#!/bin/bash
#SBATCH --job-name=airfrans-a100-train
#SBATCH --account=astro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=23:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.log
set -euo pipefail

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo 'Usage: original-a100-training.sbatch TRAIN VALIDATION NORMALISER QUALIFICATION [RESUME]' >&2
    exit 2
fi

: "${EXP_OUT:?Set EXP_OUT with the registered experiment run}"
export OMP_NUM_THREADS=2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

training=(python -m airfoil_surrogate.train_original "$1" "$2" "$3"
    --qualification "$4" --out "$EXP_OUT/training")
if [ "$#" -eq 5 ]; then
    training+=(--resume "$5")
fi
"${training[@]}"
