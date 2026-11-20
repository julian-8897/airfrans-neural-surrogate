#!/bin/bash
#SBATCH --job-name=airfrans-cfd-matched
#SBATCH --account=astro
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.log
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo 'Usage: cfd-matched.sbatch COHORTS_JSON CASE_ID GENERATOR_REPO QUALIFICATION_JSON' >&2
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
python -m airfoil_surrogate.cfd.run_case "$1" "$2" "$3" \
    --qualification "$4" --out "$EXP_OUT/cfd"
