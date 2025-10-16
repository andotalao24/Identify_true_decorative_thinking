# TrueThinking Reasoning Pipeline

Tools for extracting, scoring, and steering the reasoning traces of language models. The workflow estimates **TrueThinking Scores (TTS)** for individual reasoning steps, derives steering directions from high/low TTS examples, and applies those directions to intervene on future generations.

## Prerequisites

- Python 3.10+ with `pip`
- CUDA-capable GPU with sufficient VRAM for the chosen model size
- (Recommended) bash-compatible shell even on Windows (e.g., Git Bash) to run the helper scripts
- Install Python dependencies before running any stage:

```bash
pip install -r requirements.txt
```

## Repository Layout

- `data/` – input prompts, evaluation splits, and perturbation test sets
- `output/` – JSONL artifacts from inference, checkpoint analysis, and steering runs
- `out_pt/` – serialized steering vectors and temporary tensors
- `inference_vllm.py`, `inference_checkpoint_analysis.py` – core generation and checkpoint analysis scripts
- `tts.py`, `utils_tts.py` – utilities for computing TTS and sampling high/low steps
- `extract_hidden.py`, `intervention.py` – feature extraction and intervention logic
- `run_*.sh` – opinionated entrypoints that chain the Python modules for common experiments

## End-to-End Workflow

The typical experiment proceeds through the following stages:

1. **Baseline generation (`run_inference_vllm.sh`)** – produce model reasoning traces with vLLM.
2. **Early-exit checkpoint sweep (`run_ckpt_no_perturb.sh`)** – record internal checkpoints without perturbations.
3. **Perturbed checkpoint analysis (`run_checkpoint_analysis.sh`)** – introduce controlled perturbations per reasoning step.
4. **TrueThinking scoring (`tts.py`)** – compute per-step TTS and sample low/high TTS examples.
5. **Direction extraction (`run_diff_mean.sh`)** – derive TrueThinking steering vectors from the sampled steps.
6. **Intervention (`complete_intervene.sh`)** – apply the steering vector during generation to nudge reasoning.

Each stage consumes the artifacts from the previous step; details are below.

## Stage 1 – Baseline Inference

```bash
bash run_inference_vllm.sh
```

Key arguments (see `run_inference_vllm.sh`) control the backbone model, tensor parallelism, sampling behavior, and input file (default: `data/amc.jsonl`). The script writes a JSONL file such as `output/output_amc_nemotron1d5b.jsonl` with complete reasoning transcripts.

## Stage 2 – Early-Exit Checkpoints

```bash
bash run_ckpt_no_perturb.sh
```

This step replays the baseline runs with `inference_checkpoint_analysis.py` while disabling perturbations. The output (`output/checkpoint_analysis_*_no_perturb.jsonl`) captures the model state after each reasoning checkpoint and is required for the perturbation analyses.

Adjust the script variables (`MODEL`, `MODEL_SIZE`, `INPUT`, `USER_TAG`) to match the dataset generated in Stage 1.

## Stage 3 – Perturbed Checkpoint Analysis

```bash
bash run_checkpoint_analysis.sh
```

The script sweeps over random seeds and perturbation modes (e.g., random number replacements) to produce multiple JSONL runs under different interventions. By default outputs land in `output/ckpt/` and `output/checkpoint_analysis_*_perturb_*.jsonl`.

Tips:

- Ensure `INPUT` points to the no-perturb file from Stage 2.
- Tune `--right` to control how many checkpoints are processed.
- The script empties CUDA caches between runs; keep it if you experience OOM issues.

## Stage 4 – TrueThinking Score Extraction

Edit the top of `tts.py` so that the four `read_jsonl` calls reference the outputs from Stages 2–3:

```python
d_no_perturb = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_right_no_perturb.jsonl")
d_perturb_s = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_100_add_small_all_num_perturb_s.jsonl")
d_perturb_s_c = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_100_add_small_all_num_perturb_s_c.jsonl")
d_perturb_c = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_100_add_small_all_num_perturb_c.jsonl")
```

Then run:

```bash
python tts.py
```

The script filters for cases with low/high TTS and creates two JSONL files (defaults: `low_tts_steps.jsonl`, `high_tts_steps.jsonl`). These files drive the direction extraction step.

## Stage 5 – TrueThinking Direction

Update `run_diff_mean.sh` so that `harmful_pth` / `harmless_pth` target the low/high TTS files created in Stage 4, then run:

```bash
bash run_diff_mean.sh
```

Internally the script invokes `extract_hidden.py`, computes hidden-state averages for the two cohorts, and writes a steering vector (e.g., `out_pt/nemotron-1d5b-amc-dir-uf-ff.pt`). Temporary tensors are stored in `out_pt/tmp.pt` to allow inspection if needed.

For full sweeps over multiple thresholds, see `run_combined_pipeline.sh` which chains direction extraction and interventions across several low/high TTS cutoffs.

## Stage 6 – Steering Intervention

```bash
bash complete_intervene.sh
```

The script loads the steering vector from Stage 5 (`--intervention_vector`) and runs `intervention.py` against a perturbation test set (`data/math_nemotron_perturb_rand_small_test-rand100.json` by default). Results are written to `output/steer/`, capturing both reasoning traces and intervention metadata.

Adjust the layer range (`--layer_s`, `--layer_e`), context-masking flags, and decoding limits to experiment with where and how the steering vector is injected.

## Inspecting & Evaluating Results

- `utils_eval.py` offers helpers for accuracy, calibration, and consistency checks.
- `demo_steering.ipynb` walks through an interactive analysis of steering outcomes.
- `web/` hosts a lightweight visualization app for qualitative review.

## Troubleshooting

- **OOM during checkpoint analysis** – Reduce batch size (`--batch_size`), shorten `--right`, or run fewer seeds per pass.
- **Mismatched case counts in `tts.py`** – Confirm all four JSONL inputs stem from the same baseline set and share identical ordering.
- **Slow feature extraction** – Use the sampler flags (`--random_sample_harmful`) in `get_diff_mean.sh` to limit tokens processed during direction estimation.

## License

Released under the terms specified in `LICENSE`. Review upstream model licenses when distributing derived artifacts.
