#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 baseline_active|parallelworld GENERATOR_PID" >&2
  exit 2
fi

MODE="$1"
GENERATOR_PID="$2"
if [[ "${MODE}" != "baseline_active" && "${MODE}" != "parallelworld" ]]; then
  echo "Unknown mode: ${MODE}" >&2
  exit 2
fi
if [[ ! "${GENERATOR_PID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid generator PID: ${GENERATOR_PID}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_JSON_ROOT="${REPO_ROOT}/dataset/json_clean"
GENERATED_ROOT="${REPO_ROOT}/generated/unobserved_changes_full"
EVAL_JSON_ROOT="${REPO_ROOT}/generated/unobserved_changes_eval_json"
LOCK_PATH="${REPO_ROOT}/generated/.unobserved_changes_eval_stage.lock"
READY_MARKER="${EVAL_JSON_ROOT}/.ready.json"

cd "${REPO_ROOT}"
echo "[$(date '+%F %T')] Waiting for generator PID ${GENERATOR_PID} to finish..."
while kill -0 "${GENERATOR_PID}" 2>/dev/null; do
  sleep 30
done
echo "[$(date '+%F %T')] Generator exited; staging and validating the new dataset."

mkdir -p "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
flock 9
if [[ ! -f "${READY_MARKER}" ]]; then
  python scripts/stage_unobserved_changes_generated.py \
    --source-json-root "${SOURCE_JSON_ROOT}" \
    --generated-root "${GENERATED_ROOT}" \
    --output-root "${EVAL_JSON_ROOT}"
fi
flock -u 9

if [[ ! -f "${READY_MARKER}" ]]; then
  echo "Dataset staging did not produce ${READY_MARKER}" >&2
  exit 1
fi

if [[ "${MODE}" == "baseline_active" ]]; then
  echo "[$(date '+%F %T')] Starting the active baseline on GPU 1."
  JSON_ROOT="${EVAL_JSON_ROOT}" \
  SUITE_OUTPUT_ROOT="outputs/baseline_active_unobserved_change" \
    bash evaluate_baseline_active.sh --tasks unobserved_changes --gpu 1
else
  echo "[$(date '+%F %T')] Starting parallelworld on GPU 3."
  JSON_ROOT="${EVAL_JSON_ROOT}" \
  SUITE_OUTPUT_ROOT="outputs/parallelworld_unobserved_change" \
  SUITE_STATE_ROOT="outputs/parallelworld_unobserved_change" \
    bash evaluate_parallelworld.sh \
      --tasks unobserved_changes --gpu 3
fi
