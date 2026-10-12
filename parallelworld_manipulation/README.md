# ParallelWorld Manipulation

This directory implements ParallelWorld for robotic manipulation on LIBERO-Pro. The agent proposes candidate manipulation actions, predicts their visual outcomes, evaluates the candidates against the task, and executes the selected action in the simulator. The implementation builds on [RPent](https://github.com/RLinf/RPent).

## Installation

Use Linux with NVIDIA GPUs and an EGL-capable MuJoCo environment. The package supports Python `>=3.10,<3.13`; the evaluation environment used Python 3.11. Install this source directory rather than a separate upstream RPent checkout so that the ParallelWorld modules and prompts are included.

The reference software versions are recorded in [configs/environment_versions.json](../configs/environment_versions.json). This file describes the evaluation environment and is not a complete dependency lockfile.

From the unified repository root:

```bash
export PARALLELWORLD_ROOT="$PWD"
cd "$PARALLELWORLD_ROOT/parallelworld_manipulation"
python3.11 -m venv "$PARALLELWORLD_ROOT/.venv-manipulation"
source "$PARALLELWORLD_ROOT/.venv-manipulation/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[full,gpt-image]'
liberopro-download-assets --skip-existing
```

The simulator asset command is provided by the LIBERO-Pro dependency and follows the [RPent installation guide](https://rpent.readthedocs.io/en/latest/rst_source/installation.html). Install a PyTorch build compatible with the host CUDA driver before starting the VLA and segmentation services.

The code package includes the static memory files in `resources/libero/memory/` and robot reference images in `resources/libero/robot_references/`. Keep these resources available when running the evaluation.

## Checkpoints and API access

The local controller requires the [RLinf Pi0.5 LIBERO checkpoint](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT). The segmentation service requires the [SAM 3 checkpoint](https://huggingface.co/facebook/sam3); follow the checkpoint access instructions in the [official SAM 3 repository](https://github.com/facebookresearch/sam3). These weights are external dependencies and are not included in this repository.

```bash
export PW_CHECKPOINT_ROOT="$PARALLELWORLD_ROOT/checkpoints"
python -m pip install --upgrade huggingface_hub
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  --exclude optimizer.pt \
  --local-dir "$PW_CHECKPOINT_ROOT/RLinf-Pi05-LIBERO-130-fullshot-SFT"

# Run after obtaining access to the official SAM 3 model.
hf auth login
hf download facebook/sam3 sam3.pt \
  --local-dir "$PW_CHECKPOINT_ROOT/sam3"

export PI05_CHECKPOINT_PATH="$PW_CHECKPOINT_ROOT/RLinf-Pi05-LIBERO-130-fullshot-SFT"
export SAM3_CHECKPOINT_PATH="$PW_CHECKPOINT_ROOT/sam3/sam3.pt"
```

Configure an inference endpoint that exposes the model IDs in the evaluation configuration below. The planner uses the Responses transport; image prediction uses the Images edit API. If the image API uses separate credentials or an endpoint, configure `GPT_IMAGE_API_KEY` and `GPT_IMAGE_BASE_URL` separately.

```bash
export PARALLELWORLD_REPO_ROOT="$PARALLELWORLD_ROOT/parallelworld_manipulation"
export PW_PYTHON="$PARALLELWORLD_ROOT/.venv-manipulation/bin/python"
export PYTHONPATH="$PARALLELWORLD_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_TYPE=pro
export MUJOCO_GL=egl
export PARALLELWORLD_CODEX_AUTH=api

# Replace this example URL with the complete API root for your provider.
export OPENAI_BASE_URL='https://your-provider.example/v1'
read -rsp 'Inference API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
export GPT_IMAGE_BASE_URL="$OPENAI_BASE_URL"
export GPT_IMAGE_API_KEY="$OPENAI_API_KEY"

# Set after downloading simulator assets and checkpoints. This keeps the
# bundled memory fixed instead of synchronizing it from a remote dataset.
export HF_HUB_OFFLINE=1

export PW_RESULTS_ROOT="$PARALLELWORLD_ROOT/eval_runs/libero_pro"
export PW_GPUS='0 1 2 3 4 5 6 7'
```

The launchers accept any nonempty list of GPU IDs. One GPU runs episodes sequentially; multiple GPUs divide the task–seed grid between workers. Each worker starts a local controller, segmentation service, and world-model service. Change `PW_PORT_BASE` when running independent batches concurrently.

## Evaluation configuration

`T` denotes instruction redirection, and `S` denotes position swap. Each setting contains 100 task–seed pairs: task IDs `0–9` and seeds `1–10`.

| Setting | Planner | Verifier | Prompt enhancer | Image model |
| --- | --- | --- | --- | --- |
| Spatial T/S | `gpt-5.5` | `gpt-5.5` | `gpt-5.5` | `gpt-image-2` |
| Object T/S | `gpt-5.5` | `gpt-5.5` | `gpt-5.5` | `gpt-image-2.5` |
| Goal T/S | `gpt-5.6-sol` | `gpt-5.6-sol` | `gpt-5.6-sol` | `gpt-image-2.5` |
| LIBERO-10 T/S | `gpt-5.6-sol` | `gpt-5.6-sol` | `gpt-5.6-sol` | `gpt-image-2.5` |

The planner backend is `codex`. Provider prefixes for verifier and enhancer models select an API adapter: `openai:` uses PydanticAI, and `codex:` uses the Codex Responses adapter. The Object evaluation loads the suite and per-trial interface profiles from `configs/manipulation_object_swap.json` in the repository root. The reference model families remain the same across these adapters.

The launchers select the planner default by suite: Spatial and Object use `gpt-5.5`, while Goal and LIBERO-10 use `gpt-5.6-sol`. Set `PW_PLANNER_MODEL` to override the suite default; a matching `PW_TRIAL_MODEL_CONFIG` profile takes precedence. The verifier, prompt enhancer, and image model should also be configured explicitly as shown in the evaluation commands below.

All GPT Image evaluations use `quality=low`, `size=1024x1024`, one terminal-state prediction per candidate, and disabled TTS. The batch launchers set 100 planner turns and 10,000 simulator steps per episode. ParallelWorld uses up to four initial candidates, at most three retained parents, four children per parent, and a maximum search depth of four.

Check the entry point and API configuration before evaluating:

```bash
cd "$PARALLELWORLD_REPO_ROOT"
"$PW_PYTHON" -m parallelworld_manipulation.cli.main --env libero --help
"$PW_PYTHON" -m parallelworld_manipulation.codex_transport
```

The transport precheck validates local configuration without sending a model inference request.

## Run the evaluation

Run the following batches from `parallelworld_manipulation/`. Each selected suite contains the complete 100-pair grid. The commands set every model role explicitly so each setting uses the reference configuration.

```bash
# Spatial: 200 episodes across T and S.
PW_PORT_BASE=23000 \
PW_OUTPUT_ROOT="$PW_RESULTS_ROOT/spatial" \
PW_PLANNER_MODEL=gpt-5.5 \
PW_VERIFIER_MODEL=openai:gpt-5.5 \
PW_PROMPT_ENHANCER_MODEL=openai:gpt-5.5 \
PW_IMAGE_MODEL=gpt-image-2 \
bash evaluate_spatial.sh gpt_image

# Object: 200 episodes across T and S.
PW_PORT_BASE=24000 \
PW_SUITES='libero_object_task libero_object_swap' \
PW_OUTPUT_ROOT="$PW_RESULTS_ROOT/object" \
PW_TRIAL_MODEL_CONFIG="$PARALLELWORLD_ROOT/configs/manipulation_object_swap.json" \
PW_PLANNER_MODEL=gpt-5.5 \
PW_VERIFIER_MODEL=openai:gpt-5.5 \
PW_PROMPT_ENHANCER_MODEL=openai:gpt-5.5 \
PW_IMAGE_MODEL=gpt-image-2.5 \
bash evaluate_manipulation.sh gpt_image

# Goal: 200 episodes across T and S.
PW_PORT_BASE=25000 \
PW_SUITES='libero_goal_task libero_goal_swap' \
PW_OUTPUT_ROOT="$PW_RESULTS_ROOT/goal" \
PW_PLANNER_MODEL=gpt-5.6-sol \
PW_VERIFIER_MODEL=codex:gpt-5.6-sol \
PW_PROMPT_ENHANCER_MODEL=codex:gpt-5.6-sol \
PW_IMAGE_MODEL=gpt-image-2.5 \
bash evaluate_manipulation.sh gpt_image

# LIBERO-10: 200 episodes across T and S.
PW_PORT_BASE=26000 \
PW_SUITES='libero_10_task libero_10_swap' \
PW_OUTPUT_ROOT="$PW_RESULTS_ROOT/libero10" \
PW_PLANNER_MODEL=gpt-5.6-sol \
PW_VERIFIER_MODEL=codex:gpt-5.6-sol \
PW_PROMPT_ENHANCER_MODEL=codex:gpt-5.6-sol \
PW_IMAGE_MODEL=gpt-image-2.5 \
bash evaluate_manipulation.sh gpt_image
```

`PW_TRIAL_MODEL_CONFIG` applies the matching suite's default role models and per-trial overrides. An image-model override must agree with the batch image model because the image service is shared by that worker.

## Calculate metrics

The task success indicator is the top-level `terminated` boolean in the episode audit. The metric calculation verifies it against the final `terminated` value in `states.json`. Primitive-level `pick_result.success`, process exit codes, and finish markers do not determine task success.

For each setting,

```text
success rate (%) = 100 × successful task–seed pairs / 100 planned pairs
```

Calculate a summary directly from the four evaluation output directories:

```bash
"$PW_PYTHON" "$PARALLELWORLD_ROOT/tools/summarize_libero_pro.py" \
  --results-root "$PW_RESULTS_ROOT/spatial" \
  --results-root "$PW_RESULTS_ROOT/object" \
  --results-root "$PW_RESULTS_ROOT/goal" \
  --results-root "$PW_RESULTS_ROOT/libero10" \
  --output "$PW_RESULTS_ROOT/summary.json"
```

The summary contains the successful episode count and success rate for each setting. Keep the episode audits, environment trajectories, and model transcripts with their evaluation outputs.

## Output structure

```text
<batch>/<suite>/t<task>_s<seed>/
├── <suite_without_libero_>_t<task>_s<seed>.json
├── states.json
├── transcript_<suite_without_libero_>_t<task>_s<seed>.json
├── parallelworld_trace.json
├── parallelworld_timing_summary.json
└── parallelworld/
```

The episode audit contains the task identity and success indicator. `states.json` records the real environment trajectory. Transcripts record the planner model and its messages; `parallelworld/` contains candidate prompts, predicted images, verifier decisions, and selected actions.

## Acknowledgments

This implementation builds on [RPent](https://github.com/RLinf/RPent), the [RLinf](https://github.com/RLinf/RLinf) runtime and Pi0.5 controller, LIBERO-Pro, and [SAM 3](https://github.com/facebookresearch/sam3). Their source code, assets, and weights remain subject to the respective upstream licenses.
