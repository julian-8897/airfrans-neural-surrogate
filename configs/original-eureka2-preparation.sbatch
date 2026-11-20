#!/bin/bash
#SBATCH --job-name=airfrans-prepare
#SBATCH --account=astro
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -euo pipefail

: "${EXP_OUT:?Set EXP_OUT with the registered experiment run}"
: "${AIRFOIL_SCRATCH:?Set the study directory on parallel scratch}"
: "${AIRFOIL_UV:?Set the absolute path to the cluster uv executable}"
cd "$AIRFOIL_SCRATCH"
case "$(pwd -P)/" in
    "/parallel_scratch/$USER/"*) ;;
    *) echo 'The study must run on Eureka2 parallel scratch' >&2; exit 2 ;;
esac
case "$EXP_OUT/" in
    "$AIRFOIL_SCRATCH/"*) ;;
    *) echo 'Experiment outputs must be under the study scratch directory' >&2; exit 2 ;;
esac

export TMPDIR="$EXP_OUT/tmp" XDG_CACHE_HOME="$EXP_OUT/cache"
export XDG_CONFIG_HOME="$EXP_OUT/config" XDG_DATA_HOME="$EXP_OUT/xdg-data"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv" UV_PYTHON_INSTALL_DIR="$AIRFOIL_SCRATCH/interpreters"
export HF_HOME="$XDG_CACHE_HOME/huggingface" MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
cd "$AIRFOIL_SCRATCH/checkout"

case "${1:-}" in
    dataset)
        readarray -t source < <(python3 -c 'import json; d=json.load(open("manifests/airfrans_original.json")); print(d["url"]); print(d["size"]); print(d["etag"])')
        curl --fail --location --retry 4 --connect-timeout 20 \
            --header "If-Match: ${source[2]}" --dump-header "$EXP_OUT/dataset-http.txt" \
            --output "$EXP_OUT/Dataset.zip.partial" "${source[0]}"
        test "$(stat --format=%s "$EXP_OUT/Dataset.zip.partial")" = "${source[1]}"
        mv -n "$EXP_OUT/Dataset.zip.partial" "$EXP_OUT/Dataset.zip"
        ;;
    environment)
        export UV_PROJECT_ENVIRONMENT="$EXP_OUT/venv"
        "$AIRFOIL_UV" sync --locked --extra forces --python 3.12
        "$AIRFOIL_UV" export --locked --extra forces --no-hashes --no-emit-project \
            > "$EXP_OUT/locked-constraints.txt"
        "$AIRFOIL_UV" pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
            --constraint "$EXP_OUT/locked-constraints.txt" 'timm<2' 'einops==0.8.2' 'seaborn==0.13.2'
        "$AIRFOIL_UV" pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
            > "$EXP_OUT/environment.txt"
        "$UV_PROJECT_ENVIRONMENT/bin/python" -c \
            'import airfrans, einops, pyvista, seaborn, timm, torch; print(torch.__version__, torch.version.cuda)'
        ;;
    *) echo 'Specify dataset or environment' >&2; exit 2 ;;
esac
