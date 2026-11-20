# Source from a Slurm job after setting --chdir to the archived checkout.
: "${AIRFOIL_SCRATCH:?Set the study directory on Eureka2 parallel scratch}"
: "${AIRFOIL_RUNTIME:?Set the scratch virtual environment}"
: "${EXP_OUT:?Set the scratch output for the registered run}"

case "$(realpath -e "$AIRFOIL_SCRATCH")/" in
    "/parallel_scratch/$USER/"*) ;;
    *) echo 'The study directory must be on Eureka2 parallel scratch' >&2; return 2 ;;
esac
for path in "$PWD" "$EXP_OUT" "$AIRFOIL_RUNTIME"; do
    case "$(realpath -m "$path")/" in
        "$AIRFOIL_SCRATCH/"*) ;;
        *) printf 'Study path escapes parallel scratch: %s\n' "$path" >&2; return 2 ;;
    esac
done

export PATH="$AIRFOIL_RUNTIME/bin:$PATH" PYTHONPATH="$PWD/src"
export TMPDIR="$EXP_OUT/tmp" XDG_CACHE_HOME="$EXP_OUT/cache" XDG_CONFIG_HOME="$EXP_OUT/config"
export XDG_DATA_HOME="$EXP_OUT/xdg-data" MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export HF_HOME="$XDG_CACHE_HOME/huggingface" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
umask 077

load_eureka2_openfoam() {
    if ! command -v module >/dev/null; then
        source /usr/share/lmod/lmod/init/bash
    fi
    module load GCC/10.2.0 OpenMPI/4.0.5 OpenFOAM/v2112
    # Use empty positional arguments for the OpenFOAM initialisation.
    set +u
    source "$FOAM_BASH"
    set -eu
    test "$WM_PROJECT_VERSION" = v2112
    command -v simpleFoam checkMesh foamToVTK mpirun
}
