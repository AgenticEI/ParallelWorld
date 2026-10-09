#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODE="${1:-full}"
GPU_ID="${OMNIGIBSON_RENDER_GPU_ID:-2}"
RUNTIME_ROOT="${REPO_ROOT}/.runtime/unobserved_regen_runtime"

# Activate the simulator environment before invoking this script.
if [[ -n "${CONDA_PROFILE:-}" ]]; then source "${CONDA_PROFILE}"; fi
if [[ -n "${CONDA_ENV:-}" ]]; then conda activate "${CONDA_ENV}"; fi

export PYTHONPATH="${REPO_ROOT}/src/dataset_generation/task_counting:${PYTHONPATH:-}"
export OMNIGIBSON_GPU_ID="${GPU_ID}"
export OMNIGIBSON_RENDER_GPU_ID="${GPU_ID}"
export OMNIGIBSON_PHYSICS_GPU_ID="${OMNIGIBSON_PHYSICS_GPU_ID:-0}"
export OMNIGIBSON_HEADLESS=1
export PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT="${RUNTIME_ROOT}"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/torchinductor/gpu_${GPU_ID}"
export OMNIGIBSON_APPDATA_PATH="${RUNTIME_ROOT}/omnigibson_appdata/gpu_${GPU_ID}"
export __GL_SHADER_DISK_CACHE=0
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${OMNIGIBSON_APPDATA_PATH}"

command=(
    python "${SCRIPT_DIR}/regenerate_missing_assets.py"
)

case "${MODE}" in
    smoke)
        command+=(
            --scene Merom_0_int
            --room childs_room_0
            --output-root "${REPO_ROOT}/.runtime/unobserved_smoke"
            --stage-root "${REPO_ROOT}/.runtime/unobserved_smoke_fixed/source_json"
            --fixed-root "${REPO_ROOT}/.runtime/unobserved_smoke_fixed"
        )
        ;;
    full)
        command+=(
            --output-root "${REPO_ROOT}/.runtime/unobserved_regen"
            --stage-root "${REPO_ROOT}/.runtime/unobserved_fixed/source_json"
            --fixed-root "${REPO_ROOT}/.runtime/unobserved_fixed"
        )
        ;;
    *)
        echo "Usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac

exec "${command[@]}"
