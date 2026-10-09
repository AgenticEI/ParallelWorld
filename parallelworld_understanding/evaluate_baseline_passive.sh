#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROVIDER="gpt"
export MODEL="gpt-5.4"
export PIPELINE_VARIANT="baseline_passive"
export PIPELINE_DISPLAY_NAME="Passive baseline"
export SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-outputs/baseline_passive}"
export RESULTS_ROOT="${RESULTS_ROOT:-outputs/baseline_passive/results}"
export STEP_IMAGE_ROOT="${STEP_IMAGE_ROOT:-outputs/baseline_passive/steps}"
export FAILED_LOG="${FAILED_LOG:-outputs/baseline_passive/failed_indices.log}"
export BATCH_PROGRESS_FILE="${BATCH_PROGRESS_FILE:-outputs/baseline_passive/pipeline_batch_progress.json}"
export UPDATE_RESULTS_SUMMARY="${UPDATE_RESULTS_SUMMARY:-1}"

exec bash "${SCRIPT_DIR}/evaluate_baseline_active.sh" "$@"
