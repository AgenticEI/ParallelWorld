# parallelworld_understanding

This directory implements adaptive exploration and verification for ESI-Bench. At each exploration step, the agent proposes candidate actions, evaluates their outcomes, and selects routes to retain. It then answers the benchmark question using the accumulated visual evidence. The repository also includes the original evaluation baselines and task-specific dataset generation tools.

## Requirements

Use Linux with an NVIDIA GPU supported by your installed OmniGibson / Isaac Sim version. Follow the [official BEHAVIOR installation guide](https://behavior.stanford.edu/getting_started/installation.html) to install OmniGibson, Isaac Sim, BDDL, and the required scene and object assets. Simulator versions and their Python, CUDA, and driver requirements should be selected together.

Activate the simulator environment:

```bash
conda activate behavior
```

The evaluation code also uses PyTorch, NumPy, and SciPy from the simulator environment. Gemini and local Qwen backends are optional; they additionally require their provider SDK or model dependencies.

### Reference software environment

The reference evaluation environment uses the following versions:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| OmniGibson | 3.7.2 |
| Isaac Sim | 4.5.0.0 |
| BDDL | 3.7.0 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime for PyTorch | 12.4 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |

Install the application packages after configuring this simulator environment:

```bash
python -m pip install -r requirements-eval.txt
```

The application requirements pin OpenAI, OpenCV, PyYAML, and Pillow. The full recorded version information is in [`configs/environment_versions.json`](../configs/environment_versions.json). Follow the upstream guide for simulator installation and use versions compatible with this reference environment.

Follow ESI-Bench's [scene map configuration](https://github.com/ESI-Bench/ESI-Bench#special-omnigibson-setting) for floor-only maps. The benchmark requires the corresponding generated maps, as well as the scene assets; changing only the Python setting does not replace already generated map files.

## Benchmark data

The official sources are the [ESI-Bench repository](https://github.com/ESI-Bench/ESI-Bench) and [Hugging Face dataset](https://huggingface.co/datasets/ESI-Bench/esi-bench). The runner consumes category summary JSON files and individual question JSON files under `dataset/json_clean/`:

```text
dataset/
├── json_clean/
│   ├── Perceptual Grounding.json
│   ├── Physical Structure.json
│   ├── ...
│   └── <category>/<subcategory>/<scene>/<room>/<question>.json
└── assets/                    # Referenced images, where required
```

Download the official question files separately from this codebase. One way to obtain the repository-format data is:

```bash
git clone https://github.com/ESI-Bench/ESI-Bench.git /path/to/esi-bench-data
python ../tools/prepare_esi_data.py \
  --source /path/to/esi-bench-data/dataset/json_clean \
  --output /path/to/parallelworld-data/understanding/json_clean
export JSON_ROOT=/path/to/parallelworld-data/understanding/json_clean
```

Each category summary contains a `json_paths` list. Paths are resolved relative to `JSON_ROOT`. Individual questions may use the original task payload format or the flattened dataset schema with `metadata_json`, `options_json`, and `image_paths_json`. The Hugging Face `questions.jsonl` export is a tabular artifact and is not a direct input to the batch launcher.

The preparation tool copies question files and rebuilds category summaries with paths relative to the output question root. It does not generate simulator assets or missing reference images.

The tool also accepts the official `questions.jsonl` export as `--source`. Since that export does not contain original question filenames, it creates one JSON file per row at `<category>/<subcategory>/<scene>/<room>/q_<id>.json`, retaining the dataset fields for the runner to decode.

The question dataset and simulator assets serve different purposes. Tasks such as Unobserved Change and Deformable Objects also use reference images specified in the question payload. Prepare those files and ensure their paths resolve on your machine before running evaluation. When relocating an existing data directory, use `--path-map OLD=NEW` to replace asset path prefixes in the question payloads after copying the images to their new locations:

```bash
python ../tools/prepare_esi_data.py \
  --source /path/to/esi-bench-data/dataset/json_clean \
  --output /path/to/parallelworld-data/understanding/json_clean \
  --path-map /previous/assets=/path/to/parallelworld-data/understanding/assets
```

Multiple `--path-map` arguments are supported. `JSON_ROOT` changes the question-file root; embedded asset paths are handled separately.

## Configuration

From this directory, activate the simulator environment and configure the model API and output locations:

```bash
conda activate behavior
export OPENAI_API_KEY='your-api-key'
# Optional: export OPENAI_BASE_URL='https://your-endpoint.example/v1'
export MODEL=gpt-5.4
export VERIFIER_MODEL=gpt-5.5
export JSON_ROOT=/path/to/parallelworld-data/understanding/json_clean

export PARALLELWORLD_UNDERSTANDING_RUNTIME_ROOT="$PWD/.runtime"
export SUITE_OUTPUT_ROOT="$PWD/outputs/parallelworld"
export SUITE_STATE_ROOT="$SUITE_OUTPUT_ROOT"
export RESULTS_SUMMARY_PATH="$SUITE_OUTPUT_ROOT/RESULTS_SUMMARY.md"

export MAX_STEPS=15
export MIN_STEPS=0
export THRESHOLD=0.8
export ANSWER_REVIEW=1
export ANSWER_EVIDENCE_MODE=top1-route
```

`MODEL` supplies the exploration and final-answer model; `VERIFIER_MODEL` supplies the verifier. Use model identifiers supported by the configured endpoint. The task result records store the selected models and exploration history.

| Setting | Default | Meaning |
|---|---:|---|
| `MAX_STEPS` | 15 | Maximum exploration steps per question |
| `MIN_STEPS` | 0 | Minimum steps before confidence-based stopping |
| `THRESHOLD` | 0.8 | Confidence threshold for stopping |
| `VERIFIER_MAX_IMAGES` | 0 | Verifier image limit; 0 disables the cap |
| `FINAL_MAX_IMAGES` | 0 | Final-answer image limit; 0 disables the cap |
| `ANSWER_REVIEW` | 1 | Linear Alignment answer checkpoint review |
| `ANSWER_EVIDENCE_MODE` | `top1-route` | Final rank-1 route and its ancestors; `all` is optional |

## Evaluation

List the supported runner names:

```bash
bash evaluate_parallelworld.sh --list-tasks
```

Run a selected task with adaptive route selection:

```bash
bash evaluate_parallelworld.sh \
  --tasks occlusion --gpu 0 --answer-evidence-mode top1-route
```

The default `top1-route` mode reconstructs the final rank-1 branch and its ancestors, retaining the observations along that route. To supply observations from all verifier-retained branches instead, explicitly select the optional `all` mode:

```bash
bash evaluate_parallelworld.sh \
  --tasks touching --gpu 0 --answer-evidence-mode all
```

Evaluate the selected 28 subcategories:

```bash
bash evaluate_parallelworld.sh \
  --tasks action,cognitivemap,counting,size,distance,transparent,occlusion,angle_confusion,slope,stacking,deformable,storage,triangle,line,touching,mirror,multiagent,unobserved_changes \
  --gpu 0 --answer-evidence-mode top1-route
```

Liquid Volume (`pour`) can also be evaluated separately. Several runner modules cover multiple subcategories:

| Runner | Subcategories |
|---|---|
| `angle_confusion`, `occlusion`, `transparent` | View Hallucination, Partial Occlusion, Material Transparency |
| `storage`, `deformable` | Rigid Containment, Deformable Objects |
| `slope`, `stacking` | Inclined Plane, Stacking & Stability |
| `mirror` | Reflection Authoring, Spatial Relations, Correspondence |
| `line`, `triangle`, `touching` | Linear Alignment, Geometric Configuration, Physical Contact |
| `size`, `distance` | Dimensional Size, Spatial Distance |
| `cognitivemap` | Connectivity, Traversable Passage, Regional Boundary, Long-Term Navigation |
| `counting` | Counting with Occlusion, Spatial Segmentation, Merged Observation, Category Ambiguity, Structural Enclosure, Illumination Variability |
| `unobserved_changes`, `multiagent` | Unobserved Change, Agent Observation |
| `action` | Action Order Inference |

Use a distinct `SUITE_OUTPUT_ROOT` for each evaluation configuration. The launcher prepares task metadata, runs the questions, and writes category metrics and a Markdown summary.

## Baseline evaluation

The baseline runners use the same question files and simulator assets. `src/main.py` supports four evaluation methods through `--pipeline-variant`:

| Method | Purpose |
|---|---|
| `parallelworld` | parallelworld adaptive exploration; default entrypoint |
| `baseline_active` | Original active exploration baseline |
| `baseline_passive` | Passive observation baseline |
| `branching` | Branch exploration implementation |

Run the active and passive baselines with separate output directories:

```bash
SUITE_OUTPUT_ROOT="$PWD/outputs/baseline_active" \
  MAX_STEPS=30 MIN_STEPS=3 \
  bash evaluate_baseline_active.sh --tasks occlusion --gpu 0

SUITE_OUTPUT_ROOT="$PWD/outputs/baseline_passive" \
  MAX_STEPS=30 MIN_STEPS=3 \
  bash evaluate_baseline_passive.sh --tasks occlusion --gpu 0
```

`evaluate_baseline_active.sh` supports `--tasks`, `--all-tasks`, and `--list-tasks`. `evaluate_baseline_passive.sh` forwards the same task-selection options. These commands use the baseline launcher's 30-step budget and minimum of 3 steps; use explicit settings when comparing with adaptive exploration's 15-step default.

For individual questions, select an evaluation method explicitly. For example, run the active baseline on the first question in the Partial Occlusion metadata prepared by `evaluate_parallelworld.sh`:

```bash
python src/main.py \
  --pipeline-variant baseline_active \
  --task occlusion \
  --metadata "$SUITE_STATE_ROOT/metadata/perceptual_grounding/partial_occlusion.json" \
  --question-index 0 --json-root "$JSON_ROOT" \
  --provider gpt --model "$MODEL" \
  --max-steps 30 --min-steps 3 --threshold 0.8 \
  --results-root "$PWD/outputs/baseline_active_single/results" \
  --step-image-root "$PWD/outputs/baseline_active_single/steps"
```

The `--task` must match the selected question's runner. Use a category's prepared subcategory metadata file when its summary contains questions for several runners. The baseline model and input paths can be configured through `PROVIDER`, `MODEL`, `METADATA`, and `JSON_ROOT`.

Both baseline launchers update a comparison table with Active, Passive, and parallelworld columns through `src/parallelworld_understanding/update_baseline_results_summary.py`. The current baseline's metrics populate its corresponding column; the other metric directories can be set through `ACTIVE_METRICS_ROOT`, `PASSIVE_METRICS_ROOT`, and `PARALLELWORLD_METRICS_ROOT`. Use the same `RESULTS_SUMMARY_PATH` for both launchers to maintain a shared comparison table. Adaptive evaluation writes its own table through `src/parallelworld_understanding/update_results_summary.py`.

To generate the comparison table from existing metrics:

```bash
python src/parallelworld_understanding/update_baseline_results_summary.py \
  --dataset-root "$JSON_ROOT" \
  --active-metrics-root "$PWD/outputs/baseline_active/metrics" \
  --passive-metrics-root "$PWD/outputs/baseline_passive/metrics" \
  --parallelworld-metrics-root "$PWD/outputs/parallelworld/metrics" \
  --output "$PWD/RESULTS_SUMMARY.md"
```

Completed subcategories display `correct_count / completed_results`; incomplete entries display `—`. Set the metric roots to the output directories of the evaluations being compared.

## Dataset generation

Task construction code is retained in `src/dataset_generation/`. These scripts require the installed OmniGibson assets, scene descriptions, room-object metadata, and the task-specific object lists. Generating questions is separate from evaluating the released benchmark data.

| Directory | Representative entrypoints |
|---|---|
| `task_action_sequencing` | `batch_action.sh` |
| `task_capacity` | `batch_pour.sh`, `batch_storage.sh`, `batch_water.sh` |
| `task_cognitive_map` | `batch_cognitivemap_merge.sh`; `batch_cognitivemap_connect.py`, `batch_cognitivemap_plan.py`, `batch_cognitivemap_region.py` |
| `task_comparison` | `batch_distance.sh`, `batch_size.sh` |
| `task_confusing_relation` | `batch_line.sh`, `batch_touching.sh`, `batch_equilateral.sh` |
| `task_counting` | `batch_counting_merge.sh` |
| `task_deformable` | `batch_deformable.sh` |
| `task_hallucination` | `batch_occlusion_multiview.sh`, `batch_angle_confusion.sh`, `batch_transparent.sh` |
| `task_mirror` | `batch_mirror_merge.sh`; `batch_mirror_correspondence.py`, `batch_mirror_distance.py`, `batch_mirror_object_reality.py` |
| `task_multi_agent` | `batch_multi_agent.sh` |
| `task_physics` | `batch_slope.sh`, `batch_stack.sh` |
| `task_unobserved_changes` | `batch_unobserved_changes.sh` |

Run generation scripts from this directory after activating the simulator environment. Use absolute paths for external metadata and choose a new output directory. For example, generate Inclined Plane questions with the included slope object categories:

```bash
export SCENES_DIR=/path/to/scene-dictionaries
export ROOM_OBJECTS=/path/to/combined_room_object_list.json
export OBJECTS_JSON="$PWD/src/dataset_generation/task_physics/slope_objects.json"
export OUTPUT_ROOT="$PWD/generated/inclined_plane"
export RUNS_PER_ROOM=1

bash src/dataset_generation/task_physics/batch_slope.sh
```

`SCENES_DIR` for this runner contains `*_scene_dict.json` files, and `ROOM_OBJECTS` maps each scene and room to its object names, including its floor. Other generation runners may consume the original BEHAVIOR scene directory instead; use the input layout required by the selected script. `SCENE_FILTER` and `ROOM_FILTER` restrict the generated subset. Object inventory and asset-manifest files referenced by a generator must be supplied in the indicated paths.

The `KEYS_JSON` variable used by several generators names a task object inventory file. It is independent of model API credentials, which are supplied through environment variables. Scripts that use a model additionally require that provider's credentials and SDK. Generation utilities are also retained under `scripts/` for staging assets and relocating image paths.

## Outputs and metrics

```text
<SUITE_OUTPUT_ROOT>/
├── metadata/<category>/<subcategory>.json
├── metrics/<category>/<subcategory>.json
├── progress/<category>/<subcategory>.json
├── results/<category>/<subcategory>/.../*_answer.json
├── steps/<category>/<subcategory>/...
└── RESULTS_SUMMARY.md
```

When `SUITE_STATE_ROOT` differs from `SUITE_OUTPUT_ROOT`, the metadata, metrics, and progress files are written to the state root. Each answer record includes the prediction, correctness, source question, model configuration, and exploration history.

To aggregate an existing subcategory result directory:

```bash
python src/parallelworld_understanding/summarize_task_results.py \
  --metadata "$SUITE_STATE_ROOT/metadata/perceptual_grounding/partial_occlusion.json" \
  --json-root "$JSON_ROOT" \
  --results-root "$SUITE_OUTPUT_ROOT/results/perceptual_grounding/partial_occlusion" \
  --pipeline-variant parallelworld \
  --big-task 'Perceptual Grounding' --small-task 'Partial Occlusion' \
  --output "$SUITE_STATE_ROOT/metrics/perceptual_grounding/partial_occlusion.json"
```

The aggregator matches results to question metadata through `source_json`. It reports `accuracy_percent` over evaluated answers and records question coverage alongside the accuracy. Use these fields together when comparing evaluation configurations.
