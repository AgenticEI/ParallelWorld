#!/usr/bin/env bash

# OPENAI_BASE_URL is optional; the SDK uses its official endpoint when unset.
# Set OPENAI_API_KEY in the environment before starting a GPT evaluation.
# OPENAI_USER_AGENT may be configured explicitly for compatible endpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash evaluate_parallelworld.sh [task selection]

Task selection:
  --tasks NAME1,NAME2  Run only the listed runner task names.
  --base-tasks         Run the base runner's default task set.
  --all-tasks          Run every task defined in the base runner catalog.
  --list-tasks         Print valid runner task names and exit.
  --rerun-failed       Rerun each failed_indices entry once for selected tasks.
  --gpu ID             Select the physical GPU used by OmniGibson.
  --cpu-threads N      Limit CPU worker pools per evaluation process (default: 4).
  --restart-every N    Restart Python/OmniGibson after every N questions;
                       use 0 to disable periodic restarts (default: 0).
  --confidence-threshold VALUE
                       Stop when confidence is greater than VALUE (default: 0.8).
  --verifier-top-k K   Retain a fixed Top-K route limit instead of verifier-selected route count.
  --answer-evidence-mode all|top1-route
                       Use the final rank-1 route by default; all includes all selected branches.
  --answer-review 0|1  Review every Linear Alignment answer checkpoint (default: 1).
  -h, --help           Show this help text.

The GPT_TASKS or SUITE_TASKS environment variable can also contain a
comma-separated task list, "default", or "all". THRESHOLD can set the
confidence threshold through the environment.
EOF
}

if [[ -n "${GPT_TASKS:-}" && -z "${SUITE_TASKS:-}" ]]; then
  export SUITE_TASKS="${GPT_TASKS}"
fi

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
    --base-tasks)
      export SUITE_TASKS="default"
      shift
      ;;
    --all-tasks)
      export SUITE_TASKS="all"
      shift
      ;;
    --list-tasks)
      export SUITE_LIST_TASKS=1
      export SUITE_PREPARE_ONLY=1
      shift
      ;;
    --rerun-failed)
      export RERUN_FAILED_ONLY=1
      shift
      ;;
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
    --confidence-threshold|--threshold)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "$1 requires a numeric value between 0 and 1." >&2
        exit 2
      fi
      if ! awk -v value="$2" 'BEGIN {
        valid_number = value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/
        exit !(valid_number && value >= 0 && value <= 1)
      }'; then
        echo "Invalid confidence threshold: $2. Expected a value between 0 and 1." >&2
        exit 2
      fi
      export THRESHOLD="$2"
      shift 2
      ;;
    --verifier-top-k)
      if [[ "$#" -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--verifier-top-k requires a positive integer." >&2
        exit 2
      fi
      export FIXED_VERIFIER_TOP_K="$2"
      shift 2
      ;;
    --answer-review)
      if [[ "$#" -lt 2 || ! "$2" =~ ^[01]$ ]]; then
        echo "$1 requires 0 or 1." >&2
        exit 2
      fi
      case "$1" in
        --answer-review) export ANSWER_REVIEW="$2" ;;
      esac
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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
# Avoid persisting NVIDIA OpenGL shader binaries between evaluation runs.
export __GL_SHADER_DISK_CACHE=0
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${OMNIGIBSON_APPDATA_PATH}"

# Isaac Sim GPU Foundation/Vulkan must enumerate physical devices. Keep CUDA
# unmasked and select the requested physical ordinal through OmniGibson vars.
if [[ -n "${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID:-}" ]]; then
  PHYSICAL_RENDER_GPU="${OMNIGIBSON_PHYSICAL_RENDER_GPU_ID}"
  PHYSICAL_PHYSICS_GPU="${OMNIGIBSON_PHYSICAL_PHYSICS_GPU_ID:-0}"
  unset CUDA_VISIBLE_DEVICES
  export OMNIGIBSON_GPU_ID="${PHYSICAL_RENDER_GPU}"
  export OMNIGIBSON_RENDER_GPU_ID="${PHYSICAL_RENDER_GPU}"
  export OMNIGIBSON_PHYSICS_GPU_ID="${PHYSICAL_PHYSICS_GPU}"
fi

# Reuse the current multi-action runner unchanged while replacing both model roles with GPT.
export PROVIDER="${PROVIDER:-gpt}"
export MODEL="${MODEL:-gpt-5.4}"
export VERIFIER_PROVIDER="${VERIFIER_PROVIDER:-gpt}"
export VERIFIER_MODEL="${VERIFIER_MODEL:-gpt-5.5}"

export ANSWER_REVIEW="${ANSWER_REVIEW:-1}"
export ANSWER_EVIDENCE_MODE="${ANSWER_EVIDENCE_MODE:-top1-route}"
ANSWER_EVIDENCE_SUFFIX=""
if [[ "${ANSWER_EVIDENCE_MODE}" == "top1-route" ]]; then
  ANSWER_EVIDENCE_SUFFIX="_top1_route"
fi

if [[ -n "${FIXED_VERIFIER_TOP_K:-}" ]]; then
  export SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-outputs/parallelworld${ANSWER_EVIDENCE_SUFFIX}}"
  export SUITE_STATE_ROOT="${SUITE_STATE_ROOT:-${SUITE_OUTPUT_ROOT}}"
else
  export SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-outputs/parallelworld${ANSWER_EVIDENCE_SUFFIX}}"
  export SUITE_STATE_ROOT="${SUITE_STATE_ROOT:-${SUITE_OUTPUT_ROOT}}"
fi
export RESULTS_ROOT="${RESULTS_ROOT:-outputs/parallelworld/results}"
export STEP_IMAGE_ROOT="${STEP_IMAGE_ROOT:-outputs/parallelworld/steps}"
export FAILED_LOG="${FAILED_LOG:-outputs/parallelworld/failed_indices_verifier.log}"
export BATCH_PROGRESS_FILE="${BATCH_PROGRESS_FILE:-outputs/parallelworld/batch_progress_verifier.json}"

if [[ "${SUITE_PREPARE_ONLY:-0}" != "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for the GPT answer model and verifier." >&2
    exit 2
  fi

  if [[ "${GPT_SKIP_AUTH_CHECK:-0}" != "1" ]]; then
    python - <<'PY'
import os
import sys

from openai import AuthenticationError, OpenAI


def report_auth_failure(exc: AuthenticationError) -> None:
    status = getattr(exc, "status_code", "unknown")
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        detail = error.get("message", error) if isinstance(error, dict) else error
    else:
        detail = body or str(exc)
    print(f"OpenAI gateway authentication failed (HTTP {status}): {detail}", file=sys.stderr)
    print(
        "Verify the key's token group and required User-Agent; this message does not necessarily mean the key is revoked.",
        file=sys.stderr,
    )

client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
if os.environ.get("OPENAI_BASE_URL"):
    client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
client = OpenAI(**client_kwargs)

try:
    available_models = sorted({model.id for model in client.models.list().data})
except AuthenticationError as exc:
    report_auth_failure(exc)
    raise SystemExit(2)
except Exception as exc:
    print(f"OpenAI API preflight failed before simulation startup: {type(exc).__name__}", file=sys.stderr)
    raise SystemExit(3)

if os.environ.get("GPT_LIST_MODELS") == "1":
    print("Available models from the configured OpenAI endpoint:")
    for model_id in available_models:
        print(model_id)
    raise SystemExit(0)

requested_models = list(dict.fromkeys([os.environ["MODEL"], os.environ["VERIFIER_MODEL"]]))
test_image = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
for model_name in requested_models:
    try:
        client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with OK if you can read this image input."},
                        {"type": "image_url", "image_url": {"url": test_image}},
                    ],
                }
            ],
            max_completion_tokens=16,
        )
    except AuthenticationError as exc:
        report_auth_failure(exc)
        raise SystemExit(2)
    except Exception as exc:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        detail = message or type(exc).__name__
        print(f"OpenAI model preflight failed for {model_name!r}: {detail}", file=sys.stderr)
        if available_models:
            likely_vision_models = [
                item
                for item in available_models
                if any(marker in item.lower() for marker in ("gpt", "vision", "4o", "4.1", "5"))
            ]
            if likely_vision_models:
                print("Candidate models reported by this endpoint: " + ", ".join(likely_vision_models[:30]), file=sys.stderr)
        raise SystemExit(4)

print("OpenAI authentication and model preflight passed for: " + ", ".join(requested_models))
PY
  fi

  if [[ "${GPT_LIST_MODELS:-0}" == "1" ]]; then
    exit 0
  fi
fi

exec bash "${SCRIPT_DIR}/evaluate_suite.sh"
