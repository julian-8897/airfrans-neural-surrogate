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
export UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4
export HF_HOME="$XDG_CACHE_HOME/huggingface" MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
cd "${AIRFOIL_CHECKOUT:-$AIRFOIL_SCRATCH/checkout}"
export PYTHONPATH="$PWD/src"

case "${1:-}" in
    dataset)
        readarray -t source < <(python3 -c 'import json; d=json.load(open("manifests/airfrans_original.json")); print(d["url"]); print(d["size"]); print(d["etag"])')
        curl --fail --location --retry 4 --connect-timeout 20 \
            --header "If-Match: ${source[2]}" --dump-header "$EXP_OUT/dataset-http.txt" \
            --output "$EXP_OUT/Dataset.zip.partial" "${source[0]}"
        test "$(stat --format=%s "$EXP_OUT/Dataset.zip.partial")" = "${source[1]}"
        mv -n "$EXP_OUT/Dataset.zip.partial" "$EXP_OUT/Dataset.zip"
        ;;
    dataset-ranges)
        python3 scripts/fetch_original_ranges.py "${2:?Provide existing ZIP prefix}" \
            --prefix-bytes "${3:?Provide verified prefix byte count}" --out "$EXP_OUT/ranges"
        ;;
    environment)
        # Eureka2's 535-series driver ran the pilot's CUDA 12.6 build.
        # The PyPI torch wheel in uv.lock requests CUDA 13.
        export UV_PROJECT_ENVIRONMENT="$EXP_OUT/venv"
        if [ -n "${2:-}" ]; then
            case "$(realpath -e "$2")/" in
                "$AIRFOIL_SCRATCH/"*) ;;
                *) echo 'Cache seed must be on study scratch' >&2; exit 2 ;;
            esac
            mkdir -p "$UV_CACHE_DIR"
            rsync -a "$2/" "$UV_CACHE_DIR/"
        fi
        "$AIRFOIL_UV" venv --python 3.12 "$UV_PROJECT_ENVIRONMENT"
        "$UV_PROJECT_ENVIRONMENT/bin/python" - > "$EXP_OUT/requirements.txt" <<'PY'
import tomllib
from pathlib import Path

packages = tomllib.loads(Path("uv.lock").read_text())["package"]
wanted = ("airfrans", "huggingface-hub", "matplotlib", "numpy", "pyarrow",
          "pyvista", "pyyaml", "scipy", "vtk")
for name in wanted:
    versions = {pkg["version"] for pkg in packages if pkg["name"] == name}
    (version,) = versions
    print(f"{name}=={version}")
PY
        "$AIRFOIL_UV" pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
            --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
            --requirement "$EXP_OUT/requirements.txt" \
            'torch==2.14.1+cu126' 'torchvision==0.29.1+cu126' 'timm<2' \
            'einops==0.8.2' 'seaborn==0.13.2' 'pytest>=9.2.0'
        "$AIRFOIL_UV" pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
            > "$EXP_OUT/environment.txt"
        "$UV_PROJECT_ENVIRONMENT/bin/python" -c \
            'import airfrans, einops, pyvista, seaborn, timm, torch; assert torch.version.cuda == "12.6"; print(torch.__version__, torch.version.cuda)'
        ;;
    sources)
        fetch=(curl --fail --silent --show-error --location --retry 3 --connect-timeout 20)
        "${fetch[@]}" --output "$EXP_OUT/Transolver.py" \
            https://raw.githubusercontent.com/thuml/Transolver/75e0f67643806a81cd1d3f6adc88dd8c02416fe7/Airfoil-Design-AirfRANS/models/Transolver.py
        printf '%s  %s\n' \
            1d0c1b932411ac408d8d00ca047d7f875c3176e7b001db802d30c065a7416d74 \
            "$EXP_OUT/Transolver.py" | sha256sum --check
        git clone --no-checkout https://github.com/Extrality/NACA_simulation.git "$EXP_OUT/NACA_simulation"
        git -C "$EXP_OUT/NACA_simulation" cat-file -e 2dbf00c5ecbb8d6cef720db1e72c1fe40307eeab^{commit}
        "${fetch[@]}" --output "$EXP_OUT/uiuc-index.html" \
            https://m-selig.ae.illinois.edu/ads/coord_database.html
        "${fetch[@]}" --output "$EXP_OUT/uiuc-coordinates.zip" \
            https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip
        ;;
    stages)
        interpreter=${2:?Provide the scratch Python environment}
        archive=${3:?Provide the pinned Dataset.zip on scratch}
        "$interpreter" -m airfoil_surrogate.data.stage_original "$archive" \
            --partition train --out "$EXP_OUT/train"
        "$interpreter" -m airfoil_surrogate.data.stage_original "$archive" \
            --partition validation --out "$EXP_OUT/validation"
        "$interpreter" -m airfoil_surrogate.data.fit_original_normaliser "$EXP_OUT/train" \
            --out "$EXP_OUT/normaliser.json"
        ;;
    cohorts)
        interpreter=${2:?Provide the scratch Python environment}
        "$interpreter" -m airfoil_surrogate.cfd.cohorts "$3" "$4" "$5" \
            --out "$EXP_OUT/cohorts"
        ;;
    *) echo 'Specify dataset, dataset-ranges, environment, sources, stages or cohorts' >&2; exit 2 ;;
esac
