#!/bin/bash
#SBATCH --job-name=airfrans-cfd-repro
#SBATCH --account=astro
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.log
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ] || { [ "$#" -eq 4 ] && [ "$4" != --reference-mesh-exception ]; }; then
    echo 'Usage: cfd-reproduction.sbatch PLAN_JSON CASE_ID GENERATOR_REPO [--reference-mesh-exception]' >&2
    exit 2
fi

: "${EXP_OUT:?Set EXP_OUT with the registered experiment run}"
if [ -n "${AIRFOIL_SCRATCH:-}" ]; then
    source configs/eureka2-scratch.sh
    load_eureka2_openfoam
fi
export OMP_NUM_THREADS=1
options=()
if [ "$#" -eq 4 ]; then
    : "${AIRFOIL_TRAIN_STAGE:?Provide the original training stage for reference mesh verification}"
    options+=("$4")
fi
case_root="$EXP_OUT/cfd"
python -m airfoil_surrogate.cfd.run_case "$1" "$2" "$3" --out "$case_root" "${options[@]}"
if [ -n "${AIRFOIL_TRAIN_STAGE:-}" ]; then
    python -m airfoil_surrogate.cfd.reproduction verify "$1" "$AIRFOIL_TRAIN_STAGE" "$2" \
        "$case_root/native/internal.vtu" "$case_root/native/aerofoil.vtp" \
        "$case_root/case/log.checkMesh" "$case_root/case/log.simpleFoam" \
        "$case_root/case/postProcessing/forceCoeffs1/0/coefficient.dat" \
        --out "$EXP_OUT/verification" "${options[@]}"
fi
