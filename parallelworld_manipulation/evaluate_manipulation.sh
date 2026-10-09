#!/usr/bin/env bash
# parallelworld manipulation evaluation for Object, Goal, and LIBERO-10.
set -Eeuo pipefail

repo="${PARALLELWORLD_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PARALLELWORLD_REPO_ROOT="$repo"
python="${PW_PYTHON:-python}"
wan_python="${PW_WAN_PYTHON:-python}"
cosmos_python="${PW_COSMOS_PYTHON:-python}"
cosmos_root="${PW_COSMOS_ROOT:-$repo/third_party/Cosmos-Predict2}"
cosmos_checkpoints="${PW_COSMOS_CHECKPOINTS:-$repo/checkpoints/cosmos-predict2-14b-droid}"
wan_checkpoints="${PW_WAN_CHECKPOINTS:-$repo/checkpoints/Wan2.2-TI2V-5B}"
wan_dit="${PW_WAN_DIT:-$repo/checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors}"

model="${1:-}"
case "$model" in
  gpt_image)
    read -r -a gpus <<< "${PW_GPUS:-0 1 2 3 4 5 6 7}"
    root="${PW_OUTPUT_ROOT:-$repo/eval_runs/manipulation/gpt_image}"
    port_base="${PW_PORT_BASE:-24000}"
    wm_horizon=1
    ;;
  wan)
    read -r -a gpus <<< "${PW_GPUS:-0 1 2 3}"
    root="${PW_OUTPUT_ROOT:-$repo/eval_runs/manipulation/wan}"
    port_base="${PW_PORT_BASE:-19000}"
    wm_horizon=49
    ;;
  cosmos)
    read -r -a gpus <<< "${PW_GPUS:-4 5 6 7}"
    root="${PW_OUTPUT_ROOT:-$repo/eval_runs/manipulation/cosmos}"
    port_base="${PW_PORT_BASE:-20000}"
    wm_horizon=22
    ;;
  *)
    echo "usage: $0 {wan|cosmos|gpt_image}" >&2
    exit 64
    ;;
esac

cd "$repo"
[[ -f api.sh ]] && source api.sh
if [[ "${PARALLELWORLD_API_PROFILE:-default}" == 1 ]]; then
  # Direct PydanticAI calls read unsuffixed exports; planner/expansion select
  # the same _1 pair from api.sh via PARALLELWORLD_API_PROFILE.
  export OPENAI_API_KEY="${OPENAI_API_KEY_1:?backup API key is missing}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL_1:?backup API URL is missing}"
fi
export PYTHONPATH="$repo:$cosmos_root${PYTHONPATH:+:$PYTHONPATH}"
export PI05_CHECKPOINT_PATH="${PI05_CHECKPOINT_PATH:-$repo/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}"
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-$repo/checkpoints/sam3/sam3.pt}"
export LIBERO_TYPE=pro
export MUJOCO_GL=egl
export COSMOS_PREDICT2_ARGS="--checkpoints $cosmos_checkpoints"
max_attempts="${PW_EPISODE_ATTEMPTS:-5}"
read -r -a suites <<< "${PW_SUITES:-libero_object_task libero_object_swap libero_goal_task libero_goal_swap libero_10_task libero_10_swap}"
[[ ${#suites[@]} -gt 0 ]] || { echo "no suites configured" >&2; exit 2; }
declare -A seen_suites=()
for suite in "${suites[@]}"; do
  case "$suite" in
    libero_object_task|libero_object_swap|libero_goal_task|libero_goal_swap|libero_10_task|libero_10_swap) ;;
    *) echo "invalid suite: $suite" >&2; exit 2 ;;
  esac
  [[ -z "${seen_suites[$suite]:-}" ]] || { echo "duplicate suite: $suite" >&2; exit 2; }
  seen_suites[$suite]=1
done
total_episodes=$((${#suites[@]} * 100))
finish_marker="$root/ALL_FINISHED"
suite_slug="${suites[*]}"
[[ ${#suites[@]} == 6 ]] || finish_marker="$root/SELECTED_SUITES_FINISHED_${suite_slug// /_}"

# Planner/expansion read api.sh; direct auxiliary agents use its sourced exports.
# This preflight
# validates configuration before allocating GPUs; it sends no model request.
export PARALLELWORLD_CODEX_AUTH=api
"$python" -m parallelworld_manipulation.codex_transport

resolve_episode_models() {
  local suite=$1 task=$2 seed=$3 profile_output default_planner_model
  case "$suite" in
    libero_object_task|libero_object_swap) default_planner_model=gpt-5.5 ;;
    *) default_planner_model=gpt-5.6-sol ;;
  esac
  episode_planner_model="${PW_PLANNER_MODEL:-$default_planner_model}"
  episode_verifier_model="${PW_VERIFIER_MODEL:-codex:gpt-5.6-sol}"
  episode_enhancer_model="${PW_PROMPT_ENHANCER_MODEL:-codex:gpt-5.6-sol}"
  if [[ -n "${PW_TRIAL_MODEL_CONFIG:-}" ]]; then
    if ! profile_output=$("$python" "$repo/scripts/resolve_model_profile.py" \
      --config "$PW_TRIAL_MODEL_CONFIG" --suite "$suite" --task "$task" --seed "$seed" \
      --planner-default "$episode_planner_model" \
      --verifier-default "$episode_verifier_model" \
      --enhancer-default "$episode_enhancer_model" --image-model "${PW_IMAGE_MODEL:-gpt-image-2.5}"); then
      echo "invalid model profile: suite=$suite task=$task seed=$seed" >&2
      return 2
    fi
    mapfile -t episode_models <<< "$profile_output"
    if [[ ${#episode_models[@]} != 3 ]]; then
      echo "model profile must resolve exactly three role models" >&2
      return 2
    fi
    episode_planner_model=${episode_models[0]}
    episode_verifier_model=${episode_models[1]}
    episode_enhancer_model=${episode_models[2]}
  fi
}

episode_complete() {
  local out=$1 suite=$2 task=$3 seed=$4
  local tag=${suite#libero_}
  local transcript="$out/transcript_${tag}_t${task}_s${seed}.json"
  local result="$out/${tag}_t${task}_s${seed}.json"
  [[ -s "$transcript" && -s "$result" ]] || return 1
  "$python" - "$transcript" "$result" "$suite" "$task" "$seed" <<'PY' >/dev/null 2>&1
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(sys.argv[2], "r", encoding="utf-8") as f:
        result = json.load(f)
except (OSError, ValueError):
    raise SystemExit(1)

suite, task, seed = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
finish = transcript.get("finish") if isinstance(transcript, dict) else None
valid_transcript = (
    isinstance(finish, dict)
    and finish.get("_finish") is True
    and finish.get("status") in {"success", "failure"}
    and transcript.get("suite") == suite
    and transcript.get("task") == task
    and transcript.get("seed") == seed
)
valid_result = (
    isinstance(result, dict)
    and result.get("suite") == suite
    and result.get("task_id") == task
    and result.get("seed") == seed
    and isinstance(result.get("terminated"), bool)
)
raise SystemExit(0 if valid_transcript and valid_result else 1)
PY
}

count_complete() {
  "$python" - "$root" "${suites[@]}" <<'PY'
import json
import pathlib
import sys

count = 0
root = pathlib.Path(sys.argv[1])
for suite in sys.argv[2:]:
    tag = suite.removeprefix("libero_")
    for task in range(10):
        for seed in range(1, 11):
            out = root / suite / f"t{task}_s{seed}"
            try:
                with (out / f"transcript_{tag}_t{task}_s{seed}.json").open("r", encoding="utf-8") as f:
                    transcript = json.load(f)
                with (out / f"{tag}_t{task}_s{seed}.json").open("r", encoding="utf-8") as f:
                    result = json.load(f)
            except (OSError, ValueError):
                continue
            finish = transcript.get("finish") if isinstance(transcript, dict) else None
            if (
                isinstance(finish, dict)
                and finish.get("_finish") is True
                and finish.get("status") in {"success", "failure"}
                and transcript.get("suite") == suite
                and transcript.get("task") == task
                and transcript.get("seed") == seed
                and isinstance(result, dict)
                and result.get("suite") == suite
                and result.get("task_id") == task
                and result.get("seed") == seed
                and isinstance(result.get("terminated"), bool)
            ):
                count += 1
print(count)
PY
}

[[ ${#gpus[@]} -gt 0 ]] || { echo "no GPUs configured" >&2; exit 2; }
mkdir -p "$root/services"
if [[ -e "$finish_marker" ]]; then
  complete_before=$(count_complete)
  if (( complete_before == total_episodes )); then
    echo "already finished: $root"
    exit 0
  fi
  mv "$finish_marker" "$finish_marker.stale.$(date +%Y%m%d_%H%M%S)"
  echo "ignored stale finish marker: complete=$complete_before/$total_episodes" | tee -a "$root/launcher.log"
fi
if [[ -e "$root/API_QUOTA_EXHAUSTED" ]]; then
  mv "$root/API_QUOTA_EXHAUSTED" "$root/API_QUOTA_EXHAUSTED.stale.$(date +%Y%m%d_%H%M%S)"
fi
lock="$root/.launcher.lock"
if ! mkdir "$lock" 2>/dev/null; then
  echo "launcher already active: $lock" >&2
  exit 2
fi
echo "$$" >"$lock/pid"
launcher_pid=$$

service_pids=()
worker_pids=()
cleanup() {
  local pid
  "$python" "$repo/scripts/image_failure_guard.py" stop "$launcher_pid" || true
  for pid in "${worker_pids[@]}" "${service_pids[@]}"; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${worker_pids[@]}" "${service_pids[@]}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  rm -f "$lock/pid"
  rmdir "$lock" 2>/dev/null || true
}
abort_launcher() {
  trap - EXIT TERM INT
  cleanup
  exit 42
}
trap cleanup EXIT
trap abort_launcher TERM INT

if [[ -e "$root/IMAGE_GENERATION_FATAL.json" ]]; then
  echo "fatal image marker exists; inspect it before explicitly starting a new run" >&2
  exit 43
fi
"$python" -u "$repo/scripts/image_failure_guard.py" watch "$launcher_pid" "$root/IMAGE_GENERATION_FATAL.json" &
service_pids+=("$!")

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
wait_port() {
  local port=$1 name=$2 pid=$3 log=$4
  for _ in {1..900}; do
    port_open "$port" && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before port $port was ready" >&2
      tail -100 "$log" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "$name timed out waiting for port $port" >&2
  tail -100 "$log" >&2 || true
  return 1
}

for gpu in "${gpus[@]}"; do
  for port in $((port_base + gpu*10)) $((port_base + gpu*10 + 1)) $((port_base + 100 + gpu)); do
    port_open "$port" && { echo "port already in use: $port" >&2; exit 3; }
  done
done

for gpu in "${gpus[@]}"; do
  vla_port=$((port_base + gpu*10))
  sam_port=$((vla_port + 1))
  wm_port=$((port_base + 100 + gpu))
  vla_log="$root/services/vla_gpu${gpu}.log"
  sam_log="$root/services/sam3_gpu${gpu}.log"
  wm_log="$root/services/world_model_${model}_gpu${gpu}.log"

  CUDA_VISIBLE_DEVICES="$gpu" nohup "$python" robots/libero/vla_server.py \
    --transport http --host 127.0.0.1 --port "$vla_port" --cuda-device "$gpu" \
    >"$vla_log" 2>&1 </dev/null &
  vla_pid=$!; service_pids+=("$vla_pid")
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$python" robots/libero/sam3_server.py \
    --transport http --host 127.0.0.1 --port "$sam_port" --cuda-device "$gpu" \
    >"$sam_log" 2>&1 </dev/null &
  sam_pid=$!; service_pids+=("$sam_pid")

  if [[ "$model" == gpt_image ]]; then
    nohup "$python" robots/libero/gpt_image_world_model_server.py \
      --model "${PW_IMAGE_MODEL:-gpt-image-2.5}" --quality low --size 1024x1024 \
      --request-timeout-s 900 --max-retries 0 \
      --fatal-error-file "$root/IMAGE_GENERATION_FATAL.json" \
      --host 127.0.0.1 --port "$wm_port" \
      >"$wm_log" 2>&1 </dev/null &
  elif [[ "$model" == wan ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$wan_python" robots/libero/wan_world_model_server.py \
      --checkpoint-dir "$wan_checkpoints" --finetuned-dit "$wan_dit" \
      --sampling-steps 20 --frame-num 49 --frame-size '832*480' \
      --host 127.0.0.1 --port "$wm_port" --cuda-device "$gpu" \
      >"$wm_log" 2>&1 </dev/null &
  else
    torch_port=$((29600 + gpu))
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$cosmos_python" -m torch.distributed.run \
      --nproc_per_node=1 --master_port "$torch_port" \
      robots/libero/cosmos_predict2_droid_world_model_server.py \
      --cosmos-root "$cosmos_root" --parallelworld-root "$repo" \
      --checkpoint-root "$cosmos_checkpoints" --sampling-steps 20 \
      --guidance 7.0 --host 127.0.0.1 --port "$wm_port" \
      >"$wm_log" 2>&1 </dev/null &
  fi
  wm_pid=$!; service_pids+=("$wm_pid")

  wait_port "$vla_port" "VLA GPU $gpu" "$vla_pid" "$vla_log"
  wait_port "$sam_port" "SAM3 GPU $gpu" "$sam_pid" "$sam_log"
  wait_port "$wm_port" "$model world model GPU $gpu" "$wm_pid" "$wm_log"
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  echo "ready model=$model gpu=$gpu memory_mib=$used" | tee -a "$root/launcher.log"
done

worker_count=${#gpus[@]}
for worker_idx in "${!gpus[@]}"; do
  gpu=${gpus[$worker_idx]}
  (
    index=0
    for suite in "${suites[@]}"; do
      for task in {0..9}; do
        for seed in {1..10}; do
          if [[ -e "$root/API_QUOTA_EXHAUSTED" ]]; then
            echo "stop gpu=$gpu: API quota failure detected by another worker" | tee -a "$root/worker_gpu${gpu}.log"
            exit 0
          fi
          if (( index % worker_count != worker_idx )); then
            index=$((index + 1))
            continue
          fi
          out="$root/$suite/t${task}_s${seed}"
          mkdir -p "$out"
          tag=${suite#libero_}
          transcript="$out/transcript_${tag}_t${task}_s${seed}.json"
          if episode_complete "$out" "$suite" "$task" "$seed"; then
            index=$((index + 1))
            continue
          fi
          resolve_episode_models "$suite" "$task" "$seed"
          started=$(date --iso-8601=seconds)
          echo "start index=$index suite=$suite task=$task seed=$seed gpu=$gpu at=$started" | tee -a "$root/worker_gpu${gpu}.log"
          rc=1
          attempt=1
          attempts_used=0
          while (( attempt <= max_attempts )); do
            attempts_used=$attempt
            set +e
            "$python" -m parallelworld_manipulation.cli.main \
              --env libero --suite "$suite" --task "$task" --seed "$seed" \
              --libero-type pro --planner codex --model "$episode_planner_model" --max-turns 100 \
              --max-episode-steps 10000 --planner-timeout-s 0 --cuda-device "$gpu" \
              --vla-endpoint "http://127.0.0.1:$((port_base + gpu*10))" \
              --sam3-endpoint "http://127.0.0.1:$((port_base + gpu*10 + 1))" \
              --world-model-endpoint "http://127.0.0.1:$((port_base + 100 + gpu))" \
              --wm-timeout-s 0 --no-wm-tts --wm-rollout-steps "$wm_horizon" \
              --wm-verifier-model "$episode_verifier_model" \
              --wm-prompt-enhancer-model "$episode_enhancer_model" --output-dir "$out" \
              >"$out/stdout.log" 2>&1
            process_rc=$?
            set -e
            if grep -Eqi '402 Payment Required|余额不足|额度不足|insufficient[_ -]?(user[_ -]?)?quota|quota (is )?exceeded' "$out/stdout.log"; then
              quota_tmp="$root/.API_QUOTA_EXHAUSTED.$BASHPID"
              printf 'index=%s suite=%s task=%s seed=%s gpu=%s detected=%s\n' \
                "$index" "$suite" "$task" "$seed" "$gpu" "$(date --iso-8601=seconds)" \
                >"$quota_tmp"
              mv -f "$quota_tmp" "$root/API_QUOTA_EXHAUSTED"
              rc=42
              echo "quota stop index=$index gpu=$gpu" | tee -a "$root/worker_gpu${gpu}.log"
              break
            fi
            if episode_complete "$out" "$suite" "$task" "$seed"; then
              rc=0
              break
            fi
            rc=$process_rc
            (( rc == 0 )) && rc=1
            echo "retry index=$index attempt=$attempt rc=$rc gpu=$gpu" | tee -a "$root/worker_gpu${gpu}.log"
            attempt=$((attempt + 1))
            (( attempt <= max_attempts )) && sleep 15
          done
          if (( rc == 42 )); then
            printf 'rc=%s attempts=%s started=%s aborted=%s model=%s gpu=%s reason=api_quota\n' \
              "$rc" "$attempts_used" "$started" "$(date --iso-8601=seconds)" "$model" "$gpu" \
              >"$out/RUN_ABORTED"
            kill -TERM "$launcher_pid" 2>/dev/null || true
            exit 0
          fi
          printf 'rc=%s attempts=%s started=%s finished=%s model=%s gpu=%s\n' \
            "$rc" "$attempts_used" "$started" "$(date --iso-8601=seconds)" "$model" "$gpu" \
            >"$out/RUN_FINISHED"
          echo "finish index=$index rc=$rc gpu=$gpu" | tee -a "$root/worker_gpu${gpu}.log"
          index=$((index + 1))
        done
      done
    done
  ) &
  worker_pids+=("$!")
done

for pid in "${worker_pids[@]}"; do
  wait "$pid"
done
worker_pids=()
complete_after=$(count_complete)
if (( complete_after == total_episodes )); then
  printf 'model=%s episodes=%s suites=%s finished=%s\n' "$model" "$total_episodes" "${suites[*]}" "$(date --iso-8601=seconds)" >"$finish_marker"
  exit 0
fi
echo "incomplete model=$model complete=$complete_after/$total_episodes" | tee -a "$root/launcher.log"
[[ ! -e "$root/API_QUOTA_EXHAUSTED" ]] || exit 42
exit 1
