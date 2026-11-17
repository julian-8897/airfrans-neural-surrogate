#!/bin/bash
#SBATCH --job-name=airfrans-pilot
#SBATCH --account=astro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.log
set -eu
cd "$SLURM_SUBMIT_DIR"
sha256sum --check inputs.sha256
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"/parallel_scratch/$USER/gemma4/envs/train/bin/python" scripts/time_original_pilot.py data result
