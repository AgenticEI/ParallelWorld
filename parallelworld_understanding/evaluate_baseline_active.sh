#!/usr/bin/env bash

# OPENAI_BASE_URL is optional; the SDK uses its official endpoint when unset.
# Set OPENAI_API_KEY in the environment before starting a GPT evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${PIPELINE_SUITE_CHILD:-0}" != "1" ]]; then
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --tasks)
        if [[ "$#" -lt 2 || -z "$2" ]]; then
          echo "--tasks requires a comma-separated task list." >&2
          exit 2
        fi
        export SUITE_TASKS="$2"
        shift 2
        ;;
      --all-tasks)
        export SUITE_TASKS=all
        shift
        ;;
      --list-tasks)
        export SUITE_LIST_TASKS=1
        shift
        ;;
      --gpu)
        if [[ "$#" -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
          echo "--gpu requires a non-negative physical GPU ID." >&2
          exit 2
        fi
        export GPU_ID="$2"
        shift 2
        ;;
      --gpus)
        if [[ "$#" -lt 2 || -z "$2" ]]; then
          echo "--gpus requires four comma-separated physical GPU IDs." >&2
          exit 2
        fi
        export SUITE_GPUS="$2"
        shift 2
        ;;
      --launch-delay)
        if [[ "$#" -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
          echo "--launch-delay requires a non-negative number of seconds." >&2
          exit 2
        fi
        export SUITE_LAUNCH_DELAY="$2"
        shift 2
        ;;
      --cpu-threads)
        if [[ "$#" -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
          echo "--cpu-threads requires a positive integer." >&2
          exit 2
        fi
        export CPU_THREADS="$2"
        shift 2
        ;;
      --rerun-failed)
        export RERUN_FAILED_ONLY=1
        shift
        ;;
      -h|--help)
        cat <<'EOF'
Usage: bash evaluate_baseline_active.sh [--all-tasks | --tasks NAME1,NAME2] [--gpu ID]
       bash evaluate_baseline_active.sh --all-tasks --gpus ID0,ID1,ID2,ID3

  --all-tasks          Run every task in the original-pipeline suite (default).
  --tasks NAMES        Run selected runner task names, comma-separated.
  --list-tasks         List all available task mappings and exit.
  --gpu ID             Select the physical rendering GPU (default: 0).
  --gpus IDS           Split all tasks across exactly four physical GPUs.
  --launch-delay SEC   Delay each four-GPU worker launch (default: 60).
  --cpu-threads N      Limit each Isaac worker's CPU pools (default: 4).
  --rerun-failed       Rerun only indices listed in each task's failed_indices.
EOF
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done
fi

# # Isaac Sim warns that CUDA_VISIBLE_DEVICES can cause incorrect device enumeration.
# # Let Isaac Sim pick the visible device itself, and control the target GPU through OMNIGIBSON_GPU_ID.
# unset CUDA_VISIBLE_DEVICES

# # Keep the default launch mode headless. Enable WebRTC only when explicitly requested.
# if [[ -n "${OMNIGIBSON_REMOTE_STREAMING:-}" ]]; then
#   export OMNIGIBSON_REMOTE_STREAMING
# else
#   unset OMNIGIBSON_REMOTE_STREAMING
# fi

# export OMNIGIBSON_GPU_ID="${OMNIGIBSON_GPU_ID:-0}"
# export OMNIGIBSON_HTTP_PORT="${OMNIGIBSON_HTTP_PORT:-8211}"
# export OMNIGIBSON_WEBRTC_PORT="${OMNIGIBSON_WEBRTC_PORT:-49100}"
# export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"

PROVIDER="${PROVIDER:-gpt}"
MODEL="${MODEL:-gpt-5.4}"
TASK="${TASK:-occlusion}"
METADATA="${METADATA:-dataset/json_clean/Perceptual Grounding - Partial Occlusion.json}"
JSON_ROOT="${JSON_ROOT:-dataset/json_clean}"
MAX_STEPS="${MAX_STEPS:-30}"
MIN_STEPS="${MIN_STEPS:-3}"
THRESHOLD="${THRESHOLD:-0.8}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/baseline_active/results}"
STEP_IMAGE_ROOT="${STEP_IMAGE_ROOT:-outputs/baseline_active/steps}"
PIPELINE_VARIANT="${PIPELINE_VARIANT:-baseline_active}"
PIPELINE_DISPLAY_NAME="${PIPELINE_DISPLAY_NAME:-Active baseline}"
FAILED_LOG="${FAILED_LOG:-outputs/baseline_active/failed_indices.log}"
MAX_RETRIES="${MAX_RETRIES:-3}"
BATCH_PROGRESS_FILE="${BATCH_PROGRESS_FILE:-outputs/baseline_active/pipeline_batch_progress.json}"
BATCH_START_INDEX="${BATCH_START_INDEX:-}"
GPU_ID="${GPU_ID:-0}"
PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT="${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT:-${SCRIPT_DIR}/.runtime}"
export PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT
export TMPDIR="${PARALLELWORLD_UNDERSTANDING_TMPDIR:-${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT}/tmp/gpu_${GPU_ID}}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT}/torchinductor/gpu_${GPU_ID}}"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}"

OMNIGIBSON_APPDATA_ROOT="${OMNIGIBSON_APPDATA_ROOT:-${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT}/omnigibson_appdata}"
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"
export OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-${OMNIGIBSON_APPDATA_ROOT}/gpu_${GPU_ID}}"
mkdir -p "${OMNIGIBSON_APPDATA_PATH}"

# Match the stable verifier launcher: prevent every Isaac process from creating
# host-sized task, math, and TorchInductor worker pools during simultaneous startup.
CPU_THREADS="${CPU_THREADS:-4}"
if [[ ! "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS must be a positive integer: ${CPU_THREADS}" >&2
  exit 2
fi
export CPU_THREADS
export PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS="${PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS:-${CPU_THREADS}}"
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-${CPU_THREADS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${CPU_THREADS}}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-${CPU_THREADS}}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"

if [[ "${PIPELINE_SUITE_CHILD:-0}" != "1" ]]; then
  SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-outputs/baseline_active}"
  FILTERED_METADATA_ROOT="${FILTERED_METADATA_ROOT:-${SUITE_OUTPUT_ROOT}/metadata}"
  METRICS_ROOT="${METRICS_ROOT:-${SUITE_OUTPUT_ROOT}/metrics}"
  RESULTS_SUMMARY_PATH="${RESULTS_SUMMARY_PATH:-RESULTS_SUMMARY.md}"
  ACTIVE_METRICS_ROOT="${ACTIVE_METRICS_ROOT:-outputs/baseline_active/metrics}"
  PASSIVE_METRICS_ROOT="${PASSIVE_METRICS_ROOT:-outputs/baseline_passive/metrics}"
  PARALLELWORLD_METRICS_ROOT="${PARALLELWORLD_METRICS_ROOT:-outputs/parallelworld_top1_route/metrics}"
  case "${PIPELINE_VARIANT}" in
    baseline_active) ACTIVE_METRICS_ROOT="${METRICS_ROOT}" ;;
    baseline_passive) PASSIVE_METRICS_ROOT="${METRICS_ROOT}" ;;
  esac
  mkdir -p "${FILTERED_METADATA_ROOT}" "${METRICS_ROOT}"

  if [[ -n "${SUITE_GPUS:-}" ]]; then
    if [[ "${SUITE_TASKS:-all}" != "all" ]]; then
      echo "--gpus currently runs the complete suite and cannot be combined with --tasks." >&2
      exit 2
    fi

    GPU_LIST="${SUITE_GPUS// /}"
    IFS=',' read -r -a PARALLEL_GPUS <<< "${GPU_LIST}"
    if [[ "${#PARALLEL_GPUS[@]}" -ne 4 ]]; then
      echo "--gpus requires exactly four comma-separated GPU IDs." >&2
      exit 2
    fi

    declare -A SEEN_GPUS=()
    for gpu in "${PARALLEL_GPUS[@]}"; do
      if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid physical GPU ID in --gpus: ${gpu}" >&2
        exit 2
      fi
      if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "Duplicate physical GPU ID in --gpus: ${gpu}" >&2
        exit 2
      fi
      SEEN_GPUS["${gpu}"]=1
    done

    # Keep runner aliases together so no two processes write the same progress file.
    # The groups contain approximately 883, 891, 868, and 860 questions.
    FOUR_GPU_TASK_GROUPS=(
      "triangle,size,unobserved_changes"
      "angle_confusion,cognitivemap,multiagent,slope"
      "mirror,counting,transparent,storage"
      "distance,occlusion,stacking,deformable,pour,line,touching,action"
    )
    FOUR_GPU_QUESTION_COUNTS=(883 891 868 860)
    SUITE_LAUNCH_DELAY="${SUITE_LAUNCH_DELAY:-60}"
    if [[ ! "${SUITE_LAUNCH_DELAY}" =~ ^[0-9]+$ ]]; then
      echo "SUITE_LAUNCH_DELAY must be a non-negative integer: ${SUITE_LAUNCH_DELAY}" >&2
      exit 2
    fi
    PARALLEL_LOG_ROOT="${SUITE_OUTPUT_ROOT}/logs"
    mkdir -p "${PARALLEL_LOG_ROOT}"
    PARALLEL_PIDS=()

    stop_parallel_groups() {
      echo "Stopping four-GPU suite processes..." >&2
      for pid in "${PARALLEL_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          pkill -TERM -s "${pid}" 2>/dev/null || true
        fi
      done
    }
    trap stop_parallel_groups INT TERM

    echo "Running ${PIPELINE_DISPLAY_NAME} suite across four GPUs: model=${MODEL}"
    for group_index in "${!PARALLEL_GPUS[@]}"; do
      gpu="${PARALLEL_GPUS[${group_index}]}"
      tasks="${FOUR_GPU_TASK_GROUPS[${group_index}]}"
      question_count="${FOUR_GPU_QUESTION_COUNTS[${group_index}]}"
      log_file="${PARALLEL_LOG_ROOT}/gpu_${gpu}.log"
      appdata_path="${OMNIGIBSON_APPDATA_ROOT}/gpu_${gpu}"
      mkdir -p "${appdata_path}"

      if [[ "${group_index}" -gt 0 && "${SUITE_LAUNCH_DELAY}" -gt 0 ]]; then
        echo "Waiting ${SUITE_LAUNCH_DELAY}s before launching GPU ${gpu}..."
        sleep "${SUITE_LAUNCH_DELAY}"
      fi
      echo "  gpu=${gpu} questions~${question_count} appdata=${appdata_path} tasks=${tasks}"
      printf '\n===== four-GPU run %s gpu=%s appdata=%s =====\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${gpu}" "${appdata_path}" >> "${log_file}"

      SUITE_GPUS= \
      OMNIGIBSON_HEADLESS=1 \
      OMNIGIBSON_APPDATA_PATH="${appdata_path}" \
      setsid --wait bash "$0" --tasks "${tasks}" --gpu "${gpu}" \
        > >(sed -u "s/^/[gpu ${gpu}] /" | tee -a "${log_file}") 2>&1 &
      PARALLEL_PIDS+=("$!")
    done

    parallel_status=0
    for group_index in "${!PARALLEL_PIDS[@]}"; do
      pid="${PARALLEL_PIDS[${group_index}]}"
      gpu="${PARALLEL_GPUS[${group_index}]}"
      if wait "${pid}"; then
        echo "GPU ${gpu} task group completed."
      else
        group_status=$?
        echo "GPU ${gpu} task group failed with exit code ${group_status}." >&2
        parallel_status=1
      fi
    done
    trap - INT TERM

    if [[ "${parallel_status}" -ne 0 ]]; then
      echo "One or more GPU task groups failed. Check ${PARALLEL_LOG_ROOT}/gpu_*.log." >&2
      exit 1
    fi
    echo "Finished all four original-pipeline task groups."
    exit 0
  fi

  ALL_TASK_SPECS=(
    "Action Sequencing|Action Order Inference|action"
    # "Cognitive Mapping|Connectivity|cognitivemap"
    # "Cognitive Mapping|Long-Term Navigation|cognitivemap"
    "Cognitive Mapping|Regional Boundary|cognitivemap"
    "Cognitive Mapping|Traversable Passage|cognitivemap"
    "Enumerative Perception|Category Ambiguity|counting"
    "Enumerative Perception|Counting w Occlusion|counting"
    "Enumerative Perception|Illumination Variability|counting"
    "Enumerative Perception|Merged Observation|counting"
    "Enumerative Perception|Spatial Segmentation|counting"
    "Enumerative Perception|Structural Enclosure|counting"
    "Metric Comparison|Dimensional Size|size"
    "Metric Comparison|Spatial Distance|distance"
    "Perceptual Grounding|Material Transparency|transparent"
    "Perceptual Grounding|Partial Occlusion|occlusion"
    "Perceptual Grounding|View Hallucination|angle_confusion"
    "Physical Dynamics|Inclined Plane|slope"
    "Physical Dynamics|Stacking & Stability|stacking"
    "Physical Structure|Deformable|deformable"
    "Physical Structure|Liquid Volume|pour"
    "Physical Structure|Rigid Containment|storage"
    "Spatial Relations|Geometric Configuration|triangle"
    "Spatial Relations|Linear Alignment|line"
    "Spatial Relations|Physical Contact|touching"
    "Specular Reflection|Correspondence|mirror"
    "Specular Reflection|Reflection Authoring|mirror"
    "Specular Reflection|Spatial Relations|mirror"
    "Temporal Understanding|Agent Observation|multiagent"
    "Temporal Understanding|Unobserved Change|unobserved_changes"
  )

  if [[ "${SUITE_LIST_TASKS:-0}" == "1" ]]; then
    echo "Available original-pipeline task mappings:"
    for spec in "${ALL_TASK_SPECS[@]}"; do
      IFS='|' read -r BIG_TASK SMALL_TASK RUNNER_TASK <<< "${spec}"
      printf '  %-18s %s / %s\n' "${RUNNER_TASK}" "${BIG_TASK}" "${SMALL_TASK}"
    done
    exit 0
  fi

  TASK_SELECTION="${SUITE_TASKS:-all}"
  TASK_SPECS=()
  if [[ "${TASK_SELECTION}" == "all" ]]; then
    TASK_SPECS=("${ALL_TASK_SPECS[@]}")
  else
    TASK_SELECTION="${TASK_SELECTION//,/ }"
    read -r -a REQUESTED_TASKS <<< "${TASK_SELECTION}"
    for requested_task in "${REQUESTED_TASKS[@]}"; do
      matched_task=0
      for spec in "${ALL_TASK_SPECS[@]}"; do
        IFS='|' read -r _BIG_TASK _SMALL_TASK RUNNER_TASK <<< "${spec}"
        if [[ "${RUNNER_TASK}" == "${requested_task}" ]]; then
          TASK_SPECS+=("${spec}")
          matched_task=1
        fi
      done
      if [[ "${matched_task}" == "0" ]]; then
        echo "Unknown suite task: ${requested_task}" >&2
        echo "Run bash evaluate_baseline_active.sh --list-tasks to show valid names." >&2
        exit 2
      fi
    done
  fi

  echo "Running ${PIPELINE_DISPLAY_NAME} suite: tasks=${#TASK_SPECS[@]} model=${MODEL} gpu=${GPU_ID}"
  for spec in "${TASK_SPECS[@]}"; do
    IFS='|' read -r BIG_TASK SMALL_TASK RUNNER_TASK <<< "${spec}"
    BIG_TASK_SLUG="${BIG_TASK,,}"
    BIG_TASK_SLUG="${BIG_TASK_SLUG// /_}"
    SMALL_TASK_SLUG="${SMALL_TASK,,}"
    SMALL_TASK_SLUG="${SMALL_TASK_SLUG// /_}"
    SMALL_TASK_SLUG="${SMALL_TASK_SLUG//&/and}"
    SOURCE_METADATA="dataset/json_clean/${BIG_TASK}.json"
    FILTERED_METADATA="${FILTERED_METADATA_ROOT}/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.json"
    TASK_RESULTS_ROOT="${SUITE_OUTPUT_ROOT}/results/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}"
    TASK_METRICS_PATH="${METRICS_ROOT}/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.json"

    python - "${SOURCE_METADATA}" "${SMALL_TASK}" "${FILTERED_METADATA}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
small_task = sys.argv[2]
output_path = Path(sys.argv[3])
metadata = json.loads(source_path.read_text(encoding="utf-8"))
paths = metadata.get("json_paths") if isinstance(metadata, dict) else metadata
if not isinstance(paths, list):
    raise ValueError(f"Metadata has no json_paths list: {source_path}")
filtered = [
    path for path in paths
    if len(Path(str(path)).parts) >= 2 and Path(str(path)).parts[1] == small_task
]
if not filtered:
    raise ValueError(f"No paths for small task {small_task!r} in {source_path}")
output = dict(metadata) if isinstance(metadata, dict) else {}
output["json_paths"] = filtered
output["count"] = len(filtered)
output["selected_small_task"] = small_task
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Prepared {len(filtered)} question(s): {small_task}")
PY

    echo "============================================================"
    echo "${PIPELINE_DISPLAY_NAME} task: ${BIG_TASK} / ${SMALL_TASK} (${RUNNER_TASK})"
    echo "============================================================"
    if [[ "${SUITE_PREPARE_ONLY:-0}" == "1" ]]; then
      continue
    fi
    if [[ "${RERUN_FAILED_ONLY:-0}" == "1" ]]; then
      echo "Failed-only rerun mode: ${RUNNER_TASK} will rerun only progress.failed_indices."
    fi
    if PIPELINE_SUITE_CHILD=1 \
      PROVIDER="${PROVIDER}" \
      MODEL="${MODEL}" \
      TASK="${RUNNER_TASK}" \
      METADATA="${FILTERED_METADATA}" \
      JSON_ROOT="${JSON_ROOT}" \
      RESULTS_ROOT="${TASK_RESULTS_ROOT}" \
      STEP_IMAGE_ROOT="${SUITE_OUTPUT_ROOT}/steps/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}" \
      FAILED_LOG="${SUITE_OUTPUT_ROOT}/failed/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.log" \
      BATCH_PROGRESS_FILE="${SUITE_OUTPUT_ROOT}/progress/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.json" \
      BATCH_START_INDEX= \
      RERUN_FAILED_ONLY="${RERUN_FAILED_ONLY:-0}" \
      GPU_ID="${GPU_ID}" \
      MAX_STEPS="${MAX_STEPS}" \
      MIN_STEPS="${MIN_STEPS}" \
      THRESHOLD="${THRESHOLD}" \
      bash "$0"; then
      TASK_RUN_STATUS=0
    else
      TASK_RUN_STATUS=$?
    fi

    if python src/parallelworld_understanding/summarize_task_results.py \
      --metadata "${FILTERED_METADATA}" \
      --json-root "${JSON_ROOT}" \
      --results-root "${TASK_RESULTS_ROOT}" \
      --pipeline-variant "${PIPELINE_VARIANT}" \
      --big-task "${BIG_TASK}" \
      --small-task "${SMALL_TASK}" \
      --output "${TASK_METRICS_PATH}"; then
      METRICS_STATUS=0
    else
      METRICS_STATUS=$?
    fi

    if [[ "${RERUN_FAILED_ONLY:-0}" == "1" ]]; then
      echo "Recalculated accuracy after failed-only rerun: ${TASK_METRICS_PATH}"
    fi

    if [[ "${TASK_RUN_STATUS}" -ne 0 ]]; then
      echo "Task failed with exit code ${TASK_RUN_STATUS}; partial metrics were written to ${TASK_METRICS_PATH}." >&2
      exit "${TASK_RUN_STATUS}"
    fi
    if [[ "${METRICS_STATUS}" -ne 0 ]]; then
      echo "Metrics calculation failed for ${BIG_TASK} / ${SMALL_TASK} with exit code ${METRICS_STATUS}." >&2
      exit "${METRICS_STATUS}"
    fi
    if [[ "${UPDATE_RESULTS_SUMMARY:-1}" == "1" ]]; then
      if ! python src/parallelworld_understanding/update_baseline_results_summary.py \
        --dataset-root "${JSON_ROOT}" \
        --active-metrics-root "${ACTIVE_METRICS_ROOT}" \
        --passive-metrics-root "${PASSIVE_METRICS_ROOT}" \
        --parallelworld-metrics-root "${PARALLELWORLD_METRICS_ROOT}" \
        --output "${RESULTS_SUMMARY_PATH}"; then
        echo "Warning: failed to refresh ${RESULTS_SUMMARY_PATH}." >&2
      fi
    fi
  done

  echo "Finished all ${PIPELINE_DISPLAY_NAME} suite tasks."
  exit 0
fi

TOTAL_QUESTIONS="$(python - <<'PY' "${METADATA}"
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
obj = json.loads(metadata_path.read_text(encoding='utf-8'))
if isinstance(obj, dict) and isinstance(obj.get("json_paths"), list):
    print(len(obj["json_paths"]))
elif isinstance(obj, list):
    print(len(obj))
else:
    print(1)
PY
)"

# echo "Starting ESI-Bench with WebRTC streaming"
# echo "  provider=${PROVIDER}"
# echo "  model=${MODEL}"
# echo "  task=${TASK}"
# echo "  metadata=${METADATA}"
# echo "  omnigibson_gpu_id=${OMNIGIBSON_GPU_ID}"
# echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
# echo "  remote_streaming=${OMNIGIBSON_REMOTE_STREAMING-<unset>}"
# echo "  headless=${OMNIGIBSON_HEADLESS}"
# echo "  http_port=${OMNIGIBSON_HTTP_PORT}"
# echo "  webrtc_port=${OMNIGIBSON_WEBRTC_PORT}"

echo "Running ${TOTAL_QUESTIONS} question(s) from ${METADATA}"
echo "Simulator config: gpu=${GPU_ID} headless=${OMNIGIBSON_HEADLESS} appdata=${OMNIGIBSON_APPDATA_PATH}"
echo "CPU parallelism: workers=${CPU_THREADS} carb=${PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS} pxr=${PXR_WORK_THREAD_LIMIT} torch_compile=${TORCHINDUCTOR_COMPILE_THREADS}"
mkdir -p "$(dirname "${FAILED_LOG}")"
touch "${FAILED_LOG}"
mkdir -p "$(dirname "${BATCH_PROGRESS_FILE}")"

read_progress_field() {
  local field="$1"
  python - "${BATCH_PROGRESS_FILE}" "${field}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
if not path.is_file():
    print("")
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
value = data.get(field, "")
print("" if value is None else value)
PY
}

if [[ "${RERUN_FAILED_ONLY:-0}" == "1" ]]; then
  mapfile -t FAILED_INDICES_TO_RERUN < <(
    python - "${BATCH_PROGRESS_FILE}" "${TOTAL_QUESTIONS}" <<'PY'
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1])
total_questions = int(sys.argv[2])
if not progress_path.is_file():
    raise SystemExit(0)
try:
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

failed_indices = progress.get("failed_indices", [])
if not isinstance(failed_indices, list):
    raise SystemExit(0)
for index in sorted({int(value) for value in failed_indices if str(value).lstrip("-").isdigit()}):
    if 0 <= index < total_questions:
        print(index)
PY
  )

  if [[ "${#FAILED_INDICES_TO_RERUN[@]}" -eq 0 ]]; then
    echo "No failed indices to rerun for task=${TASK}: ${BATCH_PROGRESS_FILE}"
    exit 0
  fi

  echo "Failed-only rerun mode: task=${TASK} indices=${FAILED_INDICES_TO_RERUN[*]}"
  RERUN_WORK_DIR="$(mktemp -d "${TMPDIR}/rerun_failed_${TASK}_XXXXXX")"
  RERUN_METADATA="${RERUN_WORK_DIR}/metadata.json"
  RERUN_PROGRESS="${RERUN_WORK_DIR}/progress.json"
  trap 'rm -rf "${RERUN_WORK_DIR}"' EXIT

  # A compact metadata file lets pipeline.py reuse one OmniGibson app for all
  # failed samples, while keeping the original task progress untouched.
  python - "${METADATA}" "${RERUN_METADATA}" "${FAILED_INDICES_TO_RERUN[@]}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
indices = [int(value) for value in sys.argv[3:]]
metadata = json.loads(source_path.read_text(encoding="utf-8"))
paths = metadata.get("json_paths") if isinstance(metadata, dict) else metadata
if not isinstance(paths, list):
    raise ValueError(f"Metadata has no json_paths list: {source_path}")
if any(index < 0 or index >= len(paths) for index in indices):
    raise ValueError("Failed index is outside the metadata range")

output = dict(metadata) if isinstance(metadata, dict) else {}
output["json_paths"] = [paths[index] for index in indices]
output["count"] = len(indices)
output["rerun_original_indices"] = indices
output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
PY

  read_rerun_progress_field() {
    local field="$1"
    python - "${RERUN_PROGRESS}" "${field}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
if not path.is_file():
    print("")
    raise SystemExit(0)
try:
    progress = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
value = progress.get(field, "")
print("" if value is None else value)
PY
  }

  RERUN_TOTAL="${#FAILED_INDICES_TO_RERUN[@]}"
  RERUN_CURRENT_START=0
  RERUN_BATCH_STATUS=0
  while [[ "${RERUN_CURRENT_START}" -lt "${RERUN_TOTAL}" ]]; do
    RERUN_ATTEMPT=1
    RERUN_ADVANCED=0
    while [[ "${RERUN_ATTEMPT}" -le "${MAX_RETRIES}" ]]; do
      echo "Failed-only batch range=[${RERUN_CURRENT_START},${RERUN_TOTAL}) attempt ${RERUN_ATTEMPT}/${MAX_RETRIES}"
      if CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/main.py \
        --pipeline-variant "${PIPELINE_VARIANT}" \
        --task "${TASK}" \
        --metadata "${RERUN_METADATA}" \
        --json-root "${JSON_ROOT}" \
        --provider "${PROVIDER}" \
        --model "${MODEL}" \
        --max-steps "${MAX_STEPS}" \
        --min-steps "${MIN_STEPS}" \
        --threshold "${THRESHOLD}" \
        --results-root "${RESULTS_ROOT}" \
        --step-image-root "${STEP_IMAGE_ROOT}" \
        --run-all-questions \
        --batch-range-start-index 0 \
        --start-question-index "${RERUN_CURRENT_START}" \
        --end-question-index "${RERUN_TOTAL}" \
        --batch-progress-file "${RERUN_PROGRESS}" \
        --overwrite; then
        RERUN_CURRENT_START="${RERUN_TOTAL}"
        break
      else
        RERUN_LAST_STATUS=$?
        RERUN_NEXT_INDEX="$(read_rerun_progress_field next_question_index)"
        RERUN_ACTIVE_INDEX="$(read_rerun_progress_field active_question_index)"
        if [[ "${RERUN_NEXT_INDEX}" =~ ^[0-9]+$ && "${RERUN_NEXT_INDEX}" -gt "${RERUN_CURRENT_START}" ]]; then
          echo "failed_only_batch_index=${RERUN_ACTIVE_INDEX:-${RERUN_CURRENT_START}} exit_code=${RERUN_LAST_STATUS} mode=batch_restart next_question_index=${RERUN_NEXT_INDEX}" | tee -a "${FAILED_LOG}"
          RERUN_CURRENT_START="${RERUN_NEXT_INDEX}"
          RERUN_ADVANCED=1
          sleep 1
          break
        fi

        if [[ "${RERUN_ATTEMPT}" -lt "${MAX_RETRIES}" ]]; then
          echo "Failed-only batch did not advance; retrying from temporary index=${RERUN_CURRENT_START}"
          sleep 1
        fi
        RERUN_ATTEMPT=$((RERUN_ATTEMPT + 1))
      fi
    done

    if [[ "${RERUN_CURRENT_START}" -ge "${RERUN_TOTAL}" ]]; then
      break
    fi
    if [[ "${RERUN_ADVANCED}" == "1" ]]; then
      continue
    fi
    RERUN_BATCH_STATUS="${RERUN_LAST_STATUS:-1}"
    echo "Failed-only batch stalled at temporary index=${RERUN_CURRENT_START}" | tee -a "${FAILED_LOG}"
    break
  done

  RERUN_SUCCESS_COUNT=0
  RERUN_FAIL_COUNT=0
  for QUESTION_INDEX in "${FAILED_INDICES_TO_RERUN[@]}"; do
    if python - "${METADATA}" "${JSON_ROOT}" "${RESULTS_ROOT}" "${QUESTION_INDEX}" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
json_root = Path(sys.argv[2])
results_root = Path(sys.argv[3])
question_index = int(sys.argv[4])

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
paths = metadata.get("json_paths") if isinstance(metadata, dict) else metadata
if not isinstance(paths, list) or not 0 <= question_index < len(paths):
    raise SystemExit(1)

raw_path = Path(str(paths[question_index])).expanduser()
candidates = [raw_path] if raw_path.is_absolute() else [json_root / raw_path, metadata_path.parent / raw_path, raw_path]
question_path = next((path.resolve() for path in candidates if path.exists()), candidates[0].resolve())

for result_path in results_root.rglob("*_answer.json"):
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    source = result.get("source_json") if isinstance(result, dict) else None
    if not source:
        continue
    source_path = Path(str(source)).expanduser()
    if not source_path.is_absolute():
        source_path = result_path.parent / source_path
    if source_path.resolve() == question_path:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
      RERUN_SUCCEEDED=1
      RERUN_SUCCESS_COUNT=$((RERUN_SUCCESS_COUNT + 1))
      echo "Failed-only rerun succeeded: question_index=${QUESTION_INDEX}"
    else
      RERUN_SUCCEEDED=0
      RERUN_FAIL_COUNT=$((RERUN_FAIL_COUNT + 1))
      echo "question_index=${QUESTION_INDEX} rerun_batch_exit_code=${RERUN_BATCH_STATUS} mode=failed_only_rerun reason=no_matching_result_file" | tee -a "${FAILED_LOG}"
    fi

    python - "${BATCH_PROGRESS_FILE}" "${QUESTION_INDEX}" "${RERUN_SUCCEEDED}" "${TOTAL_QUESTIONS}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

progress_path = Path(sys.argv[1])
question_index = int(sys.argv[2])
succeeded = bool(int(sys.argv[3]))
total_questions = int(sys.argv[4])
try:
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
except Exception:
    progress = {}

failed = set()
for value in progress.get("failed_indices", []):
    try:
        failed.add(int(value))
    except (TypeError, ValueError):
        pass
was_failed = question_index in failed
if succeeded:
    failed.discard(question_index)
elif 0 <= question_index < total_questions:
    failed.add(question_index)

progress["total_questions"] = int(progress.get("total_questions", total_questions) or total_questions)
progress["failed_indices"] = sorted(index for index in failed if 0 <= index < total_questions)
progress["fail_count"] = len(progress["failed_indices"])
if succeeded and was_failed:
    progress["success_count"] = min(total_questions, int(progress.get("success_count", 0) or 0) + 1)
progress["status"] = "completed" if not progress["failed_indices"] else "completed_with_failures"
progress["active_question_index"] = None
progress["next_question_index"] = total_questions
progress["last_failed_only_rerun_index"] = question_index
progress["last_failed_only_rerun_success"] = succeeded
progress["last_failed_only_rerun_at"] = datetime.now(timezone.utc).isoformat()
progress_path.parent.mkdir(parents=True, exist_ok=True)
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  done

  echo "Finished failed-only rerun: task=${TASK} succeeded=${RERUN_SUCCESS_COUNT} still_failed=${RERUN_FAIL_COUNT}"
  exit 0
fi

if [[ -n "${BATCH_START_INDEX}" ]]; then
  CURRENT_START="${BATCH_START_INDEX}"
else
  SAVED_STATUS="$(read_progress_field status)"
  SAVED_NEXT="$(read_progress_field next_question_index)"
  SAVED_END="$(read_progress_field end_question_index)"
  if [[ "${SAVED_STATUS}" == "completed" && "${SAVED_END}" == "${TOTAL_QUESTIONS}" ]]; then
    echo "Task already completed according to ${BATCH_PROGRESS_FILE}"
    exit 0
  fi
  if [[ -n "${SAVED_NEXT}" && "${SAVED_NEXT}" =~ ^[0-9]+$ && "${SAVED_NEXT}" -le "${TOTAL_QUESTIONS}" ]]; then
    CURRENT_START="${SAVED_NEXT}"
    echo "Resuming from saved progress: start_question_index=${CURRENT_START}"
  else
    CURRENT_START=0
  fi
fi

if [[ ! "${CURRENT_START}" =~ ^[0-9]+$ || "${CURRENT_START}" -gt "${TOTAL_QUESTIONS}" ]]; then
  echo "Invalid BATCH_START_INDEX: ${CURRENT_START}" >&2
  exit 2
fi

while [[ "${CURRENT_START}" -lt "${TOTAL_QUESTIONS}" ]]; do
  ATTEMPT=1
  while [[ "${ATTEMPT}" -le "${MAX_RETRIES}" ]]; do
    echo "batch start_question_index=${CURRENT_START} attempt ${ATTEMPT}/${MAX_RETRIES}"
    if CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/main.py \
      --pipeline-variant "${PIPELINE_VARIANT}" \
      --task "${TASK}" \
      --metadata "${METADATA}" \
      --json-root "${JSON_ROOT}" \
      --provider "${PROVIDER}" \
      --model "${MODEL}" \
      --max-steps "${MAX_STEPS}" \
      --min-steps "${MIN_STEPS}" \
      --threshold "${THRESHOLD}" \
      --results-root "${RESULTS_ROOT}" \
      --step-image-root "${STEP_IMAGE_ROOT}" \
      --run-all-questions \
      --batch-range-start-index 0 \
      --start-question-index "${CURRENT_START}" \
      --end-question-index "${TOTAL_QUESTIONS}" \
      --batch-progress-file "${BATCH_PROGRESS_FILE}" \
      --overwrite; then
      echo "Finished ${PIPELINE_DISPLAY_NAME} batch with one reused OmniGibson app."
      exit 0
    else
      LAST_STATUS=$?
    fi

    NEXT_INDEX="$(read_progress_field next_question_index)"
    ACTIVE_INDEX="$(read_progress_field active_question_index)"
    if [[ -n "${NEXT_INDEX}" && "${NEXT_INDEX}" =~ ^[0-9]+$ && "${NEXT_INDEX}" -gt "${CURRENT_START}" ]]; then
      echo "question_index=${ACTIVE_INDEX:-${CURRENT_START}} exit_code=${LAST_STATUS} mode=batch_restart next_question_index=${NEXT_INDEX}" | tee -a "${FAILED_LOG}"
      CURRENT_START="${NEXT_INDEX}"
      sleep 1
      break
    fi

    if [[ "${ATTEMPT}" -lt "${MAX_RETRIES}" ]]; then
      echo "Batch failed before progress advanced, retrying from question_index=${CURRENT_START}"
      sleep 1
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done

  if [[ "${ATTEMPT}" -gt "${MAX_RETRIES}" ]]; then
    echo "question_index=${CURRENT_START} exit_code=${LAST_STATUS} attempts=${MAX_RETRIES} mode=batch_stalled" | tee -a "${FAILED_LOG}"
    exit "${LAST_STATUS}"
  fi
done

# If the last question failed, the failed process advanced progress to the end
# but could not write the final completed summary before exiting.
FINAL_STATUS="$(read_progress_field status)"
if [[ "${FINAL_STATUS}" != "completed" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/main.py \
    --pipeline-variant "${PIPELINE_VARIANT}" \
    --task "${TASK}" \
    --metadata "${METADATA}" \
    --json-root "${JSON_ROOT}" \
    --provider "${PROVIDER}" \
    --model "${MODEL}" \
    --max-steps "${MAX_STEPS}" \
    --min-steps "${MIN_STEPS}" \
    --threshold "${THRESHOLD}" \
    --results-root "${RESULTS_ROOT}" \
    --step-image-root "${STEP_IMAGE_ROOT}" \
    --run-all-questions \
    --batch-range-start-index 0 \
    --start-question-index "${TOTAL_QUESTIONS}" \
    --end-question-index "${TOTAL_QUESTIONS}" \
    --batch-progress-file "${BATCH_PROGRESS_FILE}" \
    --overwrite
fi
