#!/bin/bash
#SBATCH --job-name=airfrans-a100-eval
#SBATCH --account=astro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.log
set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo 'Usage: original-a100-inference.sbatch official|transfer TRAINING BASELINES FROZEN TEST_STAGE|FINAL_COHORTS' >&2
    exit 2
fi
case "$1" in
    official) module=airfoil_surrogate.eval.original; output=test ;;
    transfer) module=airfoil_surrogate.eval.transfer; output=transfer ;;
    *) echo 'Specify official or transfer' >&2; exit 2 ;;
esac
shift

: "${EXP_OUT:?Set EXP_OUT with the registered experiment run}"
export OMP_NUM_THREADS=2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -m "$module" run "$@" --out "$EXP_OUT/$output" --device cuda
