# Identifying True and Decorative Thinking Steps in CoT

This repository contains the official implementation for the paper **"Can Aha Moments Be Fake?
Identifying True and Decorative Thinking Steps in Chain-of-Thought"**. Our research analyzes the step-wise causality to examine the faithfulness of reasoning in CoT and reveals a steering direction to mediate true thinking in LLMs.

- [Paper]()
- [Website](https://andotalao24.github.io/Identify_true_decorative_thinking/)
- [Blog]()


### Key Findings



##  Project Structure



## End-to-End Workflow

The typical experiment proceeds through the following stages:

1. **Baseline generation (`run_inference_vllm.sh`)** – produce model reasoning traces with vLLM.
2. **Early-exit checkpoint sweep (`run_ckpt_no_perturb.sh`)** – record internal checkpoints without perturbations.
3. **Perturbed checkpoint analysis (`run_checkpoint_analysis.sh`)** – introduce controlled perturbations per reasoning step.
4. **TrueThinking scoring (`tts.py`)** – compute per-step TTS and sample low/high TTS examples.
5. **Direction extraction (`run_diff_mean.sh`)** – derive TrueThinking steering vectors from the sampled steps.
6. **Intervention (`complete_intervene.sh`)** – apply the steering vector during generation to nudge reasoning.

A demo for steering with the TrueThinking direction is in `src/demo_steering.ipynb` 

## Step-wise Causality Analysis

```
sh run_ckpt_no_perturb.sh
```

This step replays the baseline runs with `inference_checkpoint_analysis.py` while disabling perturbations. The output (`output/checkpoint_analysis_*_no_perturb.jsonl`) captures the model state after each reasoning checkpoint and is required for the perturbation analyses. Adjust the script variables (`MODEL`, `MODEL_SIZE`, `INPUT`) to match the dataset generated in Stage 1.

### Perturbed Checkpoint Analysis

```
sh run_checkpoint_analysis.sh
```

The script sweeps over random seeds and perturbation modes (e.g., random number replacements) to produce multiple JSONL runs under different interventions. By default outputs land in `output/ckpt/` and `output/checkpoint_analysis_*_perturb_*.jsonl`.

Tips:
- Ensure `INPUT` points to the no-perturb file from Stage 2.
- Tune `--max_checkpoint_idx` to control how many checkpoints are processed.


### TrueThinking Score Extraction

Edit the top of `tts.py` so that the four `read_jsonl` calls reference the outputs from the previous stage:
For example 
```python
d_no_perturb = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_right_no_perturb.jsonl")
d_perturb_s = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_100_add_small_all_num_perturb_s.jsonl")
d_perturb_s_c = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_100_add_small_all_num_perturb_s_c.jsonl")
d_perturb_c = read_jsonl("output/checkpoint_analysis_math_nemotron1d5b_100_add_small_all_num_perturb_c.jsonl")
```

Then run the file. The script filters for cases with low/high TTS and creates two JSONL files (defaults: `low_tts_steps.jsonl`, `high_tts_steps.jsonl`). These files drive the direction extraction step.

## TrueThinking Direction

Update `run_diff_mean.sh` so that `harmful_pth` / `harmless_pth` target the low/high TTS files created in Stage 4, then run:

```
sh run_diff_mean.sh
```

Internally the script invokes `extract_hidden.py`, computes hidden-state averages for the two cohorts, and writes a steering vector (e.g., `out_pt/nemotron-1d5b-amc-dir-uf-ff.pt`). By default, the sh file generates the reverse direction from faithful steps to the unfaithful ones.

### Steering Intervention

```
sh complete_intervene.sh
```

The script loads the steering vector from the previous (`--intervention_vector`) and runs `intervention.py` against a steering test set (engagement test or disengagement test). 
Adjust the layer range (`--layer_s`, `--layer_e`) to experiment with where the steering vector is injected. For each layer, there will be an according output file generated.



