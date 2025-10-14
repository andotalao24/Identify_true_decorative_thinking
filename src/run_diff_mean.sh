#!/bin/sh

# Define models and keys as space-separated strings

model="nemotron"

model_size="1.5b"
harmful_pth="out_pt/unfaithful_step_amc_nemotron_rand100_replace_num_inconsistent_upper5_uf0_drop_s1_only.jsonl"
use_persuade_harmful=0 #1: use jailbroken version of harmful data
harmless_pth="out_pt/faithful_step_amc_nemotron_rand100_replace_num_inconsistent_upper5_ff0d9_drop_s1_only.jsonl"

# Create output paths with model and key information
output="out_pt/nemotron-1d5b-amc-inconsistent-no-constrain-upper5-use_abs-dir-drop-s1-only.pt"
output_harmful="out_pt/tmp.pt"
output_harmless="out_pt/tmp.pt"

sh get_diff_mean.sh "$harmful_pth" "$harmless_pth" "$use_persuade_harmful" "$output" "$output_harmful" "$output_harmless" 0 100 0 "$model" "initial_step" "$model_size"

