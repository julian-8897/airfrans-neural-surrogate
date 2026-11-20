#!/bin/bash
#SBATCH --job-name=airfrans-a100-qualify
#SBATCH --account=astro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.log
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo 'Usage: original-a100-qualification.sbatch UPSTREAM TRAIN VALIDATION NORMALISER FORCE_AUDIT CASE_ID' >&2
    exit 2
fi

: "${EXP_OUT:?Set EXP_OUT with the registered experiment run}"
export OMP_NUM_THREADS=2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -m airfoil_surrogate.models.transolver_parity "$1" "$2" "$4" \
    --case "$6" --out "$EXP_OUT/transolver-parity.json"
python -m airfoil_surrogate.preflight_original "$2" "$3" "$4" "$5" \
    "$EXP_OUT/transolver-parity.json" --out "$EXP_OUT/training-preflight"
