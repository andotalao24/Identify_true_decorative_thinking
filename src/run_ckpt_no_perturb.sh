#!/bin/bash


MODEL="nemotron"
MODEL_SIZE="1.5b"
INPUT="data/output_amc_nemotron9b_right.jsonl"
USER_TAG="\n The final answer is \\boxed{"

OUTPUT_FILE="output/checkpoint_analysis_aime_nemotron1d5b_right_no_perturb.jsonl"
python -u inference_checkpoint_analysis.py \
  --model "$MODEL" \
  --model_size "$MODEL_SIZE" \
  --input "$INPUT" \
  --output_file_name "$OUTPUT_FILE" \
  --left 0 \
  --right  999 \
  --use_template 1 \
  --no_store_logits \
  --no_store_distribution \
  --continue_after_early_exit 0 \
  --random_seed 0 \

  
