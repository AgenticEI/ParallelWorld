# parallelworld

parallelworld augments embodied agents with candidate exploration and verification before real action execution. This repository provides implementations for embodied spatial reasoning on ESI-Bench and robot manipulation on LIBERO-Pro.

The ESI-Bench implementation uses an adaptive verifier to select candidate routes from exploration observations. The LIBERO-Pro implementation predicts candidate outcomes with an image world model, evaluates them against the task, and executes the selected robot action.

## Repository structure

```text
parallelworld/
├── parallelworld_understanding/  # Spatial reasoning on ESI-Bench
├── parallelworld_manipulation/   # Robot manipulation on LIBERO-Pro
├── configs/                      # Evaluation settings and model profiles
├── tools/                        # Validation and result aggregation
├── .env.example                  # API and resource configuration
└── README.md
```

## Installation and evaluation

Install each component in a separate Python environment and follow its guide. Each guide covers dependencies, simulator assets, benchmark data, model configuration, evaluation commands, and metric calculation.

| Component | Benchmark | Guide |
|---|---|---|
| `parallelworld_understanding` | ESI-Bench (OmniGibson / BEHAVIOR) | [Installation, data preparation, and evaluation](parallelworld_understanding/README.md) |
| `parallelworld_manipulation` | LIBERO-Pro (MuJoCo) | [Installation, checkpoints, and evaluation](parallelworld_manipulation/README.md) |

The understanding guide also documents the active and passive baselines and dataset-generation scripts. Run each component's launchers from the directory specified in its guide.

## Model services and local configuration

After completing installation and resource downloads in the selected guide, configure access to the external model services and the local data and checkpoint paths. [`.env.example`](.env.example) provides a template for credentials, model identifiers, and resource paths. From the repository root, copy the template, edit it for your environment, and load it:

```bash
export PARALLELWORLD_ROOT="$PWD"
cp .env.example .env
# Edit .env before loading it.
set -a
source .env
set +a
```

`OPENAI_API_KEY` and `OPENAI_BASE_URL` configure access to the reasoning service. LIBERO-Pro also uses an image-generation service; set `GPT_IMAGE_API_KEY` and `GPT_IMAGE_BASE_URL` if its credentials or endpoint differ. The launchers read exported environment variables. When using this template, load `.env` explicitly before running them. The file is excluded from version control.

Component-specific model roles, resource paths, and evaluation options are documented in the corresponding guide.

## Acknowledgments

This implementation builds on [ESI-Bench](https://github.com/ESI-Bench/ESI-Bench), [BEHAVIOR / OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K), [RPent](https://github.com/RLinf/RPent), and LIBERO-Pro. Please follow the upstream projects' requirements for their software, datasets, and pretrained models.
