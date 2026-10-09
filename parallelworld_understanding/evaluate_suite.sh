#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${VERIFIER_SUITE_CHILD:-0}" != "1" ]]; then
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --gpu)
        if [[ "$#" -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
          echo "--gpu requires a non-negative physical GPU ID." >&2
          exit 2
        fi
        export OMNIGIBSON_GPU_ID="$2"
        export OMNIGIBSON_RENDER_GPU_ID="$2"
        export OMNIGIBSON_PHYSICAL_GPU_ID="$2"
        export OMNIGIBSON_PHYSICAL_RENDER_GPU_ID="$2"
        export OMNIGIBSON_PHYSICAL_PHYSICS_GPU_ID="$2"
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
      --restart-every)
        if [[ "$#" -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
          echo "--restart-every requires a non-negative integer (0 disables periodic restarts)." >&2
          exit 2
        fi
        export RESTART_EVERY="$2"
        shift 2
        ;;
      --answer-review)
        if [[ "$#" -lt 2 || ! "$2" =~ ^[01]$ ]]; then
          echo "--answer-review requires 0 or 1." >&2
          exit 2
        fi
        export ANSWER_REVIEW="$2"
        shift 2
        ;;
      --answer-evidence-mode)
        if [[ "$#" -lt 2 || ! "$2" =~ ^(all|top1-route)$ ]]; then
          echo "--answer-evidence-mode requires all or top1-route." >&2
          exit 2
        fi
        export ANSWER_EVIDENCE_MODE="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done
fi

# Backward-compatible environment configuration: treat an externally supplied
# renderer id as a physical id before applying CUDA ordinal remapping.
if [[ -z "${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID:-}" && -n "${OMNIGIBSON_RENDER_GPU_ID:-}" ]]; then
  export OMNIGIBSON_PHYSICAL_RENDER_GPU_ID="${OMNIGIBSON_RENDER_GPU_ID}"
  export OMNIGIBSON_PHYSICAL_PHYSICS_GPU_ID="${OMNIGIBSON_PHYSICAL_PHYSICS_GPU_ID:-${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID}}"
fi

RUNTIME_GPU_ID="${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID:-${OMNIGIBSON_RENDER_GPU_ID:-auto}}"
PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT="${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT:-${SCRIPT_DIR}/.runtime}"
export PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT
export TMPDIR="${PARALLELWORLD_UNDERSTANDING_TMPDIR:-${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT}/tmp/gpu_${RUNTIME_GPU_ID}}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT}/torchinductor/gpu_${RUNTIME_GPU_ID}}"
export OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-${PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT}/omnigibson_appdata/gpu_${RUNTIME_GPU_ID}}"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${OMNIGIBSON_APPDATA_PATH}"

# Prevent each Isaac process from independently using every host CPU core.
# These settings only limit worker-pool parallelism; task and simulation logic are unchanged.
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

unset QWEN_DEVICE_MAP
if [[ -n "${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID:-}" ]]; then
  # Isaac Sim GPU Foundation/Vulkan must enumerate physical devices. Do not
  # mask a non-zero card with CUDA_VISIBLE_DEVICES=N: that makes Kit report
  # "No device could be created" on many driver/Vulkan combinations.
  PHYSICAL_RENDER_GPU="${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID}"
  PHYSICAL_PHYSICS_GPU="${OMNIGIBSON_PHYSICAL_PHYSICS_GPU_ID:-0}"
  unset CUDA_VISIBLE_DEVICES
  export OMNIGIBSON_GPU_ID="${PHYSICAL_RENDER_GPU}"
  export OMNIGIBSON_RENDER_GPU_ID="${PHYSICAL_RENDER_GPU}"
  export OMNIGIBSON_PHYSICS_GPU_ID="${PHYSICAL_PHYSICS_GPU}"
  export QWEN_DEVICE="cuda:${PHYSICAL_RENDER_GPU}"
else
  export QWEN_DEVICE="${QWEN_DEVICE:-cuda:1}"
fi
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"
PIPELINE_VARIANT="parallelworld"
export ANSWER_REVIEW="${ANSWER_REVIEW:-0}"
export ANSWER_EVIDENCE_MODE="${ANSWER_EVIDENCE_MODE:-top1-route}"
ANSWER_EVIDENCE_SUFFIX=""
if [[ "${ANSWER_EVIDENCE_MODE}" == "top1-route" ]]; then
  ANSWER_EVIDENCE_SUFFIX="_top1_route"
fi

if [[ "${VERIFIER_SUITE_CHILD:-0}" != "1" ]]; then
  SUITE_JSON_ROOT="${JSON_ROOT:-dataset/json_clean}"
  SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-outputs/parallelworld${ANSWER_EVIDENCE_SUFFIX}}"
  SUITE_STATE_ROOT="${SUITE_STATE_ROOT:-${SUITE_OUTPUT_ROOT}}"
  FILTERED_METADATA_ROOT="${FILTERED_METADATA_ROOT:-${SUITE_STATE_ROOT}/metadata}"
  METRICS_ROOT="${METRICS_ROOT:-${SUITE_STATE_ROOT}/metrics}"
  RESULTS_SUMMARY_PATH="${RESULTS_SUMMARY_PATH:-${SUITE_OUTPUT_ROOT}/RESULTS_SUMMARY.md}"
  mkdir -p "${FILTERED_METADATA_ROOT}"

  ALL_TASK_SPECS=(
    "Action Sequencing|Action Order Inference|action"
    "Cognitive Mapping|Connectivity|cognitivemap"
    "Cognitive Mapping|Long-Term Navigation|cognitivemap"
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
  DEFAULT_SUITE_TASKS="cognitivemap,deformable,mirror,action"

  if [[ "${SUITE_LIST_TASKS:-0}" == "1" ]]; then
    echo "Available suite task names:"
    for spec in "${ALL_TASK_SPECS[@]}"; do
      IFS='|' read -r BIG_TASK SMALL_TASK RUNNER_TASK <<< "${spec}"
      printf '  %-18s %s / %s\n' "${RUNNER_TASK}" "${BIG_TASK}" "${SMALL_TASK}"
    done
    exit 0
  fi

  TASK_SELECTION="${SUITE_TASKS:-default}"
  if [[ "${TASK_SELECTION}" == "default" ]]; then
    TASK_SELECTION="${DEFAULT_SUITE_TASKS}"
  fi

  TASK_SPECS=()
  if [[ "${TASK_SELECTION}" == "all" ]]; then
    TASK_SPECS=("${ALL_TASK_SPECS[@]}")
  else
    TASK_SELECTION="${TASK_SELECTION//,/ }"
    read -r -a REQUESTED_TASKS <<< "${TASK_SELECTION}"
    if [[ "${#REQUESTED_TASKS[@]}" -eq 0 ]]; then
      echo "No suite tasks selected. Use SUITE_TASKS=all or a comma-separated task list." >&2
      exit 2
    fi
    for requested_task in "${REQUESTED_TASKS[@]}"; do
      matched_task=0
      for spec in "${ALL_TASK_SPECS[@]}"; do
        IFS='|' read -r _BIG_TASK _SMALL_TASK RUNNER_TASK <<< "${spec}"
        if [[ "${RUNNER_TASK}" == "${requested_task}" ]]; then
          matched_task=1
          already_selected=0
          for selected_spec in "${TASK_SPECS[@]}"; do
            if [[ "${selected_spec}" == "${spec}" ]]; then
              already_selected=1
              break
            fi
          done
          if [[ "${already_selected}" == "0" ]]; then
            TASK_SPECS+=("${spec}")
          fi
        fi
      done
      if [[ "${matched_task}" == "0" ]]; then
        echo "Unknown suite task: ${requested_task}" >&2
        echo "Run with SUITE_LIST_TASKS=1 to show valid task names." >&2
        exit 2
      fi
    done
  fi

  echo "Running verifier suite with ${#TASK_SPECS[@]} task(s): ${SUITE_TASKS:-${DEFAULT_SUITE_TASKS}}"

  sync_progress_with_missing_results() {
    local progress_path="$1"
    local metrics_path="$2"
    python - "${progress_path}" "${metrics_path}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

progress_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
if not metrics_path.is_file():
    raise SystemExit(0)

try:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
try:
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
except Exception:
    progress = {}

def valid_indices(values):
    output = set()
    if not isinstance(values, list):
        return output
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index >= 0:
            output.add(index)
    return output

metric_missing = metrics.get("missing_indices")
failed = (
    valid_indices(metric_missing)
    if isinstance(metric_missing, list)
    else valid_indices(progress.get("failed_indices"))
)
total = int(metrics.get("total_questions", progress.get("total_questions", 0)) or 0)
if total > 0:
    failed = {index for index in failed if index < total}

progress["total_questions"] = total
progress["failed_indices"] = sorted(failed)
progress["fail_count"] = len(failed)
progress["success_count"] = max(0, total - len(failed))
progress["status"] = "completed" if not failed else "completed_with_failures"
progress["active_question_index"] = None
progress["next_question_index"] = total
progress["last_metrics_reconciliation_at"] = datetime.now(timezone.utc).isoformat()
progress_path.parent.mkdir(parents=True, exist_ok=True)
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  }

  for spec in "${TASK_SPECS[@]}"; do
    IFS='|' read -r BIG_TASK SMALL_TASK RUNNER_TASK <<< "${spec}"
    BIG_TASK_SLUG="${BIG_TASK,,}"
    BIG_TASK_SLUG="${BIG_TASK_SLUG// /_}"
    SMALL_TASK_SLUG="${SMALL_TASK,,}"
    SMALL_TASK_SLUG="${SMALL_TASK_SLUG// /_}"
    SOURCE_METADATA="${SUITE_JSON_ROOT}/${BIG_TASK}.json"
    FILTERED_METADATA="${FILTERED_METADATA_ROOT}/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.json"
    TASK_RESULTS_ROOT="${SUITE_OUTPUT_ROOT}/results/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}"
    TASK_METRICS_PATH="${METRICS_ROOT}/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.json"
    TASK_PROGRESS_PATH="${SUITE_STATE_ROOT}/progress/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.json"

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
    path
    for path in paths
    if len(Path(str(path)).parts) >= 2 and Path(str(path)).parts[1] == small_task
]
if not filtered:
    raise ValueError(f"No paths for small task {small_task!r} in {source_path}")

output = dict(metadata) if isinstance(metadata, dict) else {}
output["json_paths"] = filtered
output["count"] = len(filtered)
output["total_target"] = len(filtered)
output["selected_small_task"] = small_task
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Prepared {len(filtered)} question(s): {small_task} -> {output_path}")
PY

    echo "============================================================"
    echo "Big task: ${BIG_TASK}"
    echo "Small task: ${SMALL_TASK}"
    echo "Runner task: ${RUNNER_TASK}"
    echo "============================================================"

    if [[ "${SUITE_PREPARE_ONLY:-0}" == "1" ]]; then
      continue
    fi

    if [[ "${RERUN_FAILED_ONLY:-0}" == "1" ]]; then
      echo "Refreshing missing result indices before failed-only rerun for task=${RUNNER_TASK}"
      python src/parallelworld_understanding/summarize_task_results.py \
        --metadata "${FILTERED_METADATA}" \
        --json-root "${SUITE_JSON_ROOT}" \
        --results-root "${TASK_RESULTS_ROOT}" \
        --pipeline-variant "${PIPELINE_VARIANT}" \
        --big-task "${BIG_TASK}" \
        --small-task "${SMALL_TASK}" \
        --output "${TASK_METRICS_PATH}"
      sync_progress_with_missing_results \
        "${TASK_PROGRESS_PATH}" \
        "${TASK_METRICS_PATH}"
    fi

    VERIFIER_SUITE_CHILD=1 \
    PIPELINE_VARIANT="${PIPELINE_VARIANT}" \
    TASK="${RUNNER_TASK}" \
    METADATA="${FILTERED_METADATA}" \
    JSON_ROOT="${SUITE_JSON_ROOT}" \
    RESULTS_ROOT="${TASK_RESULTS_ROOT}" \
    STEP_IMAGE_ROOT="${SUITE_OUTPUT_ROOT}/steps/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}" \
    FAILED_LOG="${SUITE_STATE_ROOT}/failed/${BIG_TASK_SLUG}/${SMALL_TASK_SLUG}.log" \
    BATCH_PROGRESS_FILE="${TASK_PROGRESS_PATH}" \
    bash "$0"

    if [[ "${RERUN_FAILED_ONLY:-0}" == "1" ]]; then
      echo "Failed-index reruns finished for task=${RUNNER_TASK}; recalculating accuracy from current results"
    fi

    python src/parallelworld_understanding/summarize_task_results.py \
      --metadata "${FILTERED_METADATA}" \
      --json-root "${SUITE_JSON_ROOT}" \
      --results-root "${TASK_RESULTS_ROOT}" \
      --pipeline-variant "${PIPELINE_VARIANT}" \
      --big-task "${BIG_TASK}" \
      --small-task "${SMALL_TASK}" \
      --output "${TASK_METRICS_PATH}"

    if ! python src/parallelworld_understanding/update_results_summary.py \
      --dataset-root "${SUITE_JSON_ROOT}" \
      --metrics-root "${METRICS_ROOT}" \
      --output "${RESULTS_SUMMARY_PATH}"; then
      echo "Warning: failed to refresh ${RESULTS_SUMMARY_PATH}." >&2
    fi

    if [[ "${RERUN_FAILED_ONLY:-0}" == "1" ]]; then
      sync_progress_with_missing_results \
        "${TASK_PROGRESS_PATH}" \
        "${TASK_METRICS_PATH}"
    fi
    if [[ "${RERUN_FAILED_ONLY:-0}" == "1" ]]; then
      echo "Recalculated accuracy for task=${RUNNER_TASK}: ${TASK_METRICS_PATH}"
    fi
  done

  echo "Finished all verifier suite tasks"
  exit 0
fi

PROVIDER="${PROVIDER:-qwen}"
MODEL="${MODEL:-qwen2.5-vl-7b}"
TASK="${TASK:-occlusion}"
if [[ "${TASK}" == "pour" ]]; then
  export PARALLELWORLD_UNDERSTANDING_POUR_PHYSICAL_ACTIONS=1
else
  unset PARALLELWORLD_UNDERSTANDING_POUR_PHYSICAL_ACTIONS
fi
METADATA="${METADATA:-dataset/json_clean/Perceptual Grounding - Partial Occlusion.json}"
JSON_ROOT="${JSON_ROOT:-dataset/json_clean}"
MAX_STEPS="${MAX_STEPS:-15}"
MIN_STEPS="${MIN_STEPS:-0}"
THRESHOLD="${THRESHOLD:-0.8}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/parallelworld/results}"
STEP_IMAGE_ROOT="${STEP_IMAGE_ROOT:-outputs/parallelworld/steps}"

VERIFIER_PROVIDER="${VERIFIER_PROVIDER:-qwen}"
VERIFIER_MODEL="${VERIFIER_MODEL:-qwen2.5-vl-7b}"
FIXED_VERIFIER_TOP_K="${FIXED_VERIFIER_TOP_K:-}"
VERIFIER_MAX_IMAGES="${VERIFIER_MAX_IMAGES:-0}"
FINAL_MAX_IMAGES="${FINAL_MAX_IMAGES:-0}"
ANSWER_REVIEW="${ANSWER_REVIEW:-0}"
ANSWER_EVIDENCE_MODE="${ANSWER_EVIDENCE_MODE:-top1-route}"

if [[ ! "${ANSWER_EVIDENCE_MODE}" =~ ^(all|top1-route)$ ]]; then
  echo "ANSWER_EVIDENCE_MODE must be all or top1-route, got ${ANSWER_EVIDENCE_MODE@Q}." >&2
  exit 2
fi

FIXED_VERIFIER_TOP_K_ARGS=()
if [[ -n "${FIXED_VERIFIER_TOP_K}" ]]; then
  if [[ ! "${FIXED_VERIFIER_TOP_K}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FIXED_VERIFIER_TOP_K must be a positive integer." >&2
    exit 2
  fi
  FIXED_VERIFIER_TOP_K_ARGS=(--fixed-verifier-top-k "${FIXED_VERIFIER_TOP_K}")
fi

PIPELINE_EXTRA_ARGS=(
  --answer-review "${ANSWER_REVIEW}"
  --answer-evidence-mode "${ANSWER_EVIDENCE_MODE}"
)

FAILED_LOG="${FAILED_LOG:-outputs/parallelworld/failed_indices_verifier.log}"
MAX_RETRIES="${MAX_RETRIES:-3}"
REUSE_SIM_SESSION="${REUSE_SIM_SESSION:-1}"
RESTART_EVERY="${RESTART_EVERY:-0}"
BATCH_PROGRESS_FILE="${BATCH_PROGRESS_FILE:-outputs/parallelworld/batch_progress_verifier.json}"
BATCH_START_INDEX="${BATCH_START_INDEX:-}"
BATCH_RANGE_START_INDEX="${BATCH_RANGE_START_INDEX:-0}"
BATCH_END_INDEX="${BATCH_END_INDEX:-}"

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
# TOTAL_QUESTIONS=1

if [[ ! "${BATCH_RANGE_START_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "BATCH_RANGE_START_INDEX must be a non-negative integer: ${BATCH_RANGE_START_INDEX}" >&2
  exit 2
fi
if [[ -n "${BATCH_END_INDEX}" && ! "${BATCH_END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "BATCH_END_INDEX must be a non-negative integer: ${BATCH_END_INDEX}" >&2
  exit 2
fi
if [[ ! "${RESTART_EVERY}" =~ ^[0-9]+$ ]]; then
  echo "RESTART_EVERY must be a non-negative integer: ${RESTART_EVERY}" >&2
  exit 2
fi
RESTART_EVERY="$((10#${RESTART_EVERY}))"
RUN_RANGE_START="${BATCH_RANGE_START_INDEX}"
RUN_END="${BATCH_END_INDEX:-${TOTAL_QUESTIONS}}"
if [[ "${RUN_RANGE_START}" -gt "${TOTAL_QUESTIONS}" ]]; then
  RUN_RANGE_START="${TOTAL_QUESTIONS}"
fi
if [[ "${RUN_END}" -gt "${TOTAL_QUESTIONS}" ]]; then
  RUN_END="${TOTAL_QUESTIONS}"
fi
if [[ "${RUN_END}" -lt "${RUN_RANGE_START}" ]]; then
  echo "Invalid batch range: [${RUN_RANGE_START},${RUN_END})" >&2
  exit 2
fi

echo "Running question range [${RUN_RANGE_START},${RUN_END}) of ${TOTAL_QUESTIONS} from ${METADATA} with verifier pipeline"
echo "Simulator devices: renderer_physical=${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID:-${OMNIGIBSON_RENDER_GPU_ID:-auto}} physics_physical=${OMNIGIBSON_PHYSICAL_PHYSICS_GPU_ID:-0} renderer_visible=${OMNIGIBSON_RENDER_GPU_ID:-auto} physics_visible=${OMNIGIBSON_PHYSICS_GPU_ID:-auto} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} appdata=${OMNIGIBSON_APPDATA_PATH:-default}"
echo "Simulator appdata: ${OMNIGIBSON_APPDATA_PATH:-default}"
echo "CPU parallelism: workers=${CPU_THREADS} carb=${PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS} pxr=${PXR_WORK_THREAD_LIMIT} torch_compile=${TORCHINDUCTOR_COMPILE_THREADS}"
mkdir -p "$(dirname "${FAILED_LOG}")"
if [[ "${RESET_FAILED_LOG:-0}" == "1" ]]; then
  : > "${FAILED_LOG}"
else
  touch "${FAILED_LOG}"
fi

SUCCESS_COUNT=0
FAIL_COUNT=0

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
valid_indices = set()
for value in failed_indices:
    try:
        index = int(value)
    except (TypeError, ValueError):
        continue
    if 0 <= index < total_questions:
        valid_indices.add(index)
for index in sorted(valid_indices):
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

  # Keep the original progress untouched while pipeline.py reuses one app for
  # the compact batch of failed samples.
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
output["total_target"] = len(indices)
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
      if python src/main.py \
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
        --verifier-provider "${VERIFIER_PROVIDER}" \
        --verifier-model "${VERIFIER_MODEL}" \
        "${FIXED_VERIFIER_TOP_K_ARGS[@]}" \
        --verifier-max-images "${VERIFIER_MAX_IMAGES}" \
        --final-max-images "${FINAL_MAX_IMAGES}" \
        "${PIPELINE_EXTRA_ARGS[@]}" \
        --run-all-questions \
        --batch-range-start-index 0 \
        --batch-range-end-index "${RERUN_TOTAL}" \
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
    if python - "${METADATA}" "${JSON_ROOT}" "${RESULTS_ROOT}" "${PIPELINE_VARIANT}" "${QUESTION_INDEX}" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
json_root = Path(sys.argv[2])
results_root = Path(sys.argv[3])
pipeline_variant = sys.argv[4]
question_index = int(sys.argv[5])

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
raw_paths = metadata.get("json_paths") if isinstance(metadata, dict) else metadata
if not isinstance(raw_paths, list) or not 0 <= question_index < len(raw_paths):
    raise SystemExit(1)

raw_question_path = Path(str(raw_paths[question_index])).expanduser()
candidates = (
    [raw_question_path]
    if raw_question_path.is_absolute()
    else [json_root / raw_question_path, metadata_path.parent / raw_question_path, raw_question_path]
)
question_path = next((path.resolve() for path in candidates if path.exists()), candidates[0].resolve())
pipeline_root = results_root / pipeline_variant
if not pipeline_root.is_dir():
    raise SystemExit(1)

for result_path in pipeline_root.rglob("*_answer.json"):
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(result, dict):
        continue
    source = result.get("source_json")
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

failed_indices = progress.get("failed_indices", [])
if not isinstance(failed_indices, list):
    failed_indices = []
failed = sorted({int(index) for index in failed_indices})
was_failed = question_index in failed
if succeeded:
    failed = [index for index in failed if index != question_index]
    if was_failed:
        progress["success_count"] = min(
            total_questions,
            int(progress.get("success_count", 0) or 0) + 1,
        )
elif question_index not in failed:
    failed.append(question_index)
    failed.sort()

progress["total_questions"] = int(progress.get("total_questions", total_questions) or total_questions)
progress["failed_indices"] = failed
progress["fail_count"] = len(failed)
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

if [[ "${REUSE_SIM_SESSION}" == "1" ]]; then
  if [[ "${RESTART_EVERY}" -gt 0 ]]; then
    echo "Reuse-sim mode enabled: restarting Python/OmniGibson every ${RESTART_EVERY} questions"
  else
    echo "Reuse-sim mode enabled: running all questions in one Python process"
  fi
  read_progress_field() {
    local field="$1"
    python - <<'PY' "${BATCH_PROGRESS_FILE}" "${field}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
if not path.exists():
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

  if [[ -n "${BATCH_START_INDEX}" ]]; then
    CURRENT_START="${BATCH_START_INDEX}"
  else
    SAVED_STATUS="$(read_progress_field status)"
    SAVED_RANGE_START="$(read_progress_field start_question_index)"
    SAVED_RANGE_END="$(read_progress_field end_question_index)"
    SAVED_NEXT="$(read_progress_field next_question_index)"
    if [[ "${SAVED_STATUS}" == "completed" \
      && "${SAVED_RANGE_START}" == "${RUN_RANGE_START}" \
      && "${SAVED_RANGE_END}" == "${RUN_END}" \
      && "${SAVED_NEXT:-0}" -ge "${RUN_END}" ]]; then
      echo "Task range already completed: task=${TASK} range=[${RUN_RANGE_START},${RUN_END})"
      exit 0
    elif [[ -n "${SAVED_NEXT}" \
      && "${SAVED_STATUS}" != "completed" \
      && "${SAVED_RANGE_START}" == "${RUN_RANGE_START}" \
      && "${SAVED_RANGE_END}" == "${RUN_END}" \
      && "${SAVED_NEXT}" -ge "${RUN_RANGE_START}" \
      && "${SAVED_NEXT}" -le "${RUN_END}" ]]; then
      CURRENT_START="${SAVED_NEXT}"
      echo "Resuming from saved progress: start_question_index=${CURRENT_START}"
    else
      CURRENT_START="${RUN_RANGE_START}"
    fi
  fi

  if [[ "${CURRENT_START}" -lt "${RUN_RANGE_START}" ]]; then
    CURRENT_START="${RUN_RANGE_START}"
  fi
  if [[ "${CURRENT_START}" -gt "${RUN_END}" ]]; then
    CURRENT_START="${RUN_END}"
  fi

  while [[ "${CURRENT_START}" -lt "${RUN_END}" ]]; do
    ATTEMPT=1
    RESOLVED_THIS_INDEX=0
    while [[ "${ATTEMPT}" -le "${MAX_RETRIES}" ]]; do
      PROCESS_END="${RUN_END}"
      if [[ "${RESTART_EVERY}" -gt 0 ]]; then
        CANDIDATE_END=$((CURRENT_START + RESTART_EVERY))
        if [[ "${CANDIDATE_END}" -lt "${PROCESS_END}" ]]; then
          PROCESS_END="${CANDIDATE_END}"
        fi
      fi
      echo "batch range=[${CURRENT_START},${PROCESS_END}) attempt ${ATTEMPT}/${MAX_RETRIES}"
      if python src/main.py \
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
        --verifier-provider "${VERIFIER_PROVIDER}" \
        --verifier-model "${VERIFIER_MODEL}" \
        "${FIXED_VERIFIER_TOP_K_ARGS[@]}" \
        --verifier-max-images "${VERIFIER_MAX_IMAGES}" \
        --final-max-images "${FINAL_MAX_IMAGES}" \
        "${PIPELINE_EXTRA_ARGS[@]}" \
        --run-all-questions \
        --batch-range-start-index "${RUN_RANGE_START}" \
        --batch-range-end-index "${RUN_END}" \
        --start-question-index "${CURRENT_START}" \
        --end-question-index "${PROCESS_END}" \
        --batch-progress-file "${BATCH_PROGRESS_FILE}" \
        --overwrite; then
        SUCCESS_COUNT="$(read_progress_field success_count)"
        FAIL_COUNT="$(read_progress_field fail_count)"
        NEXT_INDEX="$(read_progress_field next_question_index)"
        if [[ -n "${NEXT_INDEX}" && "${NEXT_INDEX}" -ge "${RUN_END}" ]]; then
          echo "Finished verifier batch run. success=${SUCCESS_COUNT:-unknown} failed=${FAIL_COUNT:-unknown}"
          exit 0
        fi
        if [[ -n "${NEXT_INDEX}" && "${NEXT_INDEX}" -gt "${CURRENT_START}" ]]; then
          echo "Scheduled simulator restart after range [${CURRENT_START},${NEXT_INDEX}); continuing from question_index=${NEXT_INDEX}."
          CURRENT_START="${NEXT_INDEX}"
          RESOLVED_THIS_INDEX=1
          sleep 1
          break
        fi
        LAST_STATUS=2
        echo "Batch exited successfully without advancing progress from question_index=${CURRENT_START}; treating it as a retryable failure." >&2
      else
        LAST_STATUS=$?
      fi
      NEXT_INDEX="$(read_progress_field next_question_index)"
      ACTIVE_INDEX="$(read_progress_field active_question_index)"
      if [[ -z "${ACTIVE_INDEX}" ]]; then
        ACTIVE_INDEX="${CURRENT_START}"
      fi
      if [[ -z "${NEXT_INDEX}" ]]; then
        NEXT_INDEX="${CURRENT_START}"
      fi

      if [[ "${NEXT_INDEX}" -gt "${CURRENT_START}" ]]; then
        echo "question_index=${ACTIVE_INDEX} exit_code=${LAST_STATUS} mode=batch_restart next_question_index=${NEXT_INDEX}" | tee -a "${FAILED_LOG}"
        echo "Batch process failed with exit_code=${LAST_STATUS}, but progress reached next_question_index=${NEXT_INDEX}; resuming there."
        CURRENT_START="${NEXT_INDEX}"
        RESOLVED_THIS_INDEX=1
        sleep 1
        break
      fi

      if [[ "${ATTEMPT}" -lt "${MAX_RETRIES}" ]]; then
        echo "Batch process failed at question_index=${ACTIVE_INDEX} with exit_code=${LAST_STATUS}, retrying..."
        sleep 1
      fi
      ATTEMPT=$((ATTEMPT + 1))
    done

    if [[ "${RESOLVED_THIS_INDEX}" -eq 1 ]]; then
      continue
    fi

    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "question_index=${CURRENT_START} exit_code=${LAST_STATUS} attempts=${MAX_RETRIES} mode=batch_resume" | tee -a "${FAILED_LOG}"
    echo "Skipping question_index=${CURRENT_START} after ${MAX_RETRIES} failed batch attempts"
    CURRENT_START=$((CURRENT_START + 1))
  done

  echo "Finished verifier batch run with resume mode. failed=${FAIL_COUNT}"
  if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    echo "Failed indices logged to ${FAILED_LOG}"
  fi
  exit 0
fi

for ((QUESTION_INDEX=RUN_RANGE_START; QUESTION_INDEX<RUN_END; QUESTION_INDEX++)); do
  echo "[${QUESTION_INDEX}/${TOTAL_QUESTIONS}] question_index=${QUESTION_INDEX}"
  ATTEMPT=1
  QUESTION_SUCCEEDED=0
  LAST_STATUS=0
  while [[ "${ATTEMPT}" -le "${MAX_RETRIES}" ]]; do
    echo "  attempt ${ATTEMPT}/${MAX_RETRIES}"
    if python src/main.py \
      --pipeline-variant "${PIPELINE_VARIANT}" \
      --task "${TASK}" \
      --metadata "${METADATA}" \
      --json-root "${JSON_ROOT}" \
      --question-index "${QUESTION_INDEX}" \
      --provider "${PROVIDER}" \
      --model "${MODEL}" \
      --max-steps "${MAX_STEPS}" \
      --min-steps "${MIN_STEPS}" \
      --threshold "${THRESHOLD}" \
      --results-root "${RESULTS_ROOT}" \
      --step-image-root "${STEP_IMAGE_ROOT}" \
      --verifier-provider "${VERIFIER_PROVIDER}" \
      --verifier-model "${VERIFIER_MODEL}" \
      "${FIXED_VERIFIER_TOP_K_ARGS[@]}" \
      --verifier-max-images "${VERIFIER_MAX_IMAGES}" \
      --final-max-images "${FINAL_MAX_IMAGES}" \
      "${PIPELINE_EXTRA_ARGS[@]}" \
      --overwrite; then
      QUESTION_SUCCEEDED=1
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      break
    else
      LAST_STATUS=$?
      if [[ "${ATTEMPT}" -lt "${MAX_RETRIES}" ]]; then
        echo "  attempt ${ATTEMPT} failed with exit_code=${LAST_STATUS}, retrying..."
        sleep 1
      fi
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done

  if [[ "${QUESTION_SUCCEEDED}" -eq 0 ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "question_index=${QUESTION_INDEX} exit_code=${LAST_STATUS} attempts=${MAX_RETRIES}" | tee -a "${FAILED_LOG}"
    echo "Continuing after ${MAX_RETRIES} failed attempts at question_index=${QUESTION_INDEX}"
  fi
done

echo "Finished verifier run. success=${SUCCESS_COUNT} failed=${FAIL_COUNT}"
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "Failed indices logged to ${FAILED_LOG}"
fi
