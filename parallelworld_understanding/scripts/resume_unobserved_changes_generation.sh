#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/dataset/json_clean/Temporal Understanding/Unobserved Change"
OUT="${ROOT}/generated/unobserved_changes_full"
KEYS="${ROOT}/src/dataset_generation/task_confusing_relation/keys.json"
GPU_ID="${GPU_ID:-4}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
BASE_ARGS=(--seed 7 --output_root "${OUT}" --keys_json "${KEYS}" --load_full_scene --source_json_root "${SRC}" --resume_existing)
mapfile -t SCENES < <(python - "${SRC}" "${OUT}" <<'PY'
import json, sys
from pathlib import Path
src, out = map(Path, sys.argv[1:])
pending_scenes = set()
for p in sorted(src.rglob('*.json')):
    d=json.loads(p.read_text()); m=d.get('metadata_json')
    if isinstance(m,str): m=json.loads(m)
    scene=str(d.get('scene') or m.get('scene') or '').strip()
    room=str(d.get('room') or m.get('room') or '').strip()
    t=str(d.get('task_type') or m.get('task_type') or '').strip()
    q=int(d.get('question_index', m.get('question_index', 0)))
    target=out/scene/room/'question_jsons'/t/f'q_{q:03d}.json'
    if not target.is_file(): pending_scenes.add(scene)
for scene in sorted(pending_scenes): print(scene)
PY
)
if ((${#SCENES[@]} == 0)); then echo '[DONE] No pending questions.'; exit 0; fi

RUNTIME_ROOT="${ROOT}/.runtime/generation_gpu_${GPU_ID}"
mkdir -p "${RUNTIME_ROOT}/appdata" "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/torchinductor"
failures=0
echo "Resuming ${#SCENES[@]} scenes on GPU ${GPU_ID}; OmniGibson restarts after every scene."
for scene in "${SCENES[@]}"; do
  echo "[$(date '+%F %T')] [SCENE START] ${scene}"
  if env CUDA_VISIBLE_DEVICES="${GPU_ID}" OMNIGIBSON_HEADLESS=1 \
    OMNIGIBSON_APPDATA_PATH="${RUNTIME_ROOT}/appdata" \
    TMPDIR="${RUNTIME_ROOT}/tmp" TMP="${RUNTIME_ROOT}/tmp" TEMP="${RUNTIME_ROOT}/tmp" \
    TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/torchinductor" \
    PARALLELWORLD_UNDERSTANDING_CARB_TASK_THREADS=2 PXR_WORK_THREAD_LIMIT=2 OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
    "${PYTHON_BIN}" src/dataset_generation/task_unobserved_changes/batch_unobserved_changes.py \
      "${BASE_ARGS[@]}" --source_scene "${scene}"; then
    echo "[$(date '+%F %T')] [SCENE DONE] ${scene}"
  else
    status=$?
    failures=$((failures + 1))
    echo "[$(date '+%F %T')] [SCENE FAILED] ${scene} exit=${status}" >&2
  fi
done
echo "[$(date '+%F %T')] Finished scene-isolated generation; failed_scenes=${failures}."
exit "${failures}"
