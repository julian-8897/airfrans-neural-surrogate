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

if [ "$#" -ne 3 ]; then
    echo 'Usage: cfd-reproduction.sbatch PLAN_JSON CASE_ID GENERATOR_REPO' >&2
    exit 2
fi

: "${EXP_OUT:?Set EXP_OUT with the registered experiment run}"
if [ -n "${AIRFOIL_SCRATCH:-}" ]; then
    source configs/eureka2-scratch.sh
    if ! command -v module >/dev/null; then
        source /usr/share/lmod/lmod/init/bash
    fi
    module load GCC/10.2.0 OpenMPI/4.0.5 OpenFOAM/v2112
    set +u
    source "$FOAM_BASH"
    set -eu
    test "$WM_PROJECT_VERSION" = v2112
    command -v simpleFoam checkMesh foamToVTK mpirun
fi
export OMP_NUM_THREADS=1
python -m airfoil_surrogate.cfd.run_case "$1" "$2" "$3" --out "$EXP_OUT/cfd"
