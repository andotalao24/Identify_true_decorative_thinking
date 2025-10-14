#!/bin/bash

# Checkpoint Analysis Runner - Traverse multiple random seeds

# Set CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

mkdir -p output/ckpt

MODEL="nemotron"
MODEL_SIZE="1.5b"
#INPUT="data/checkpoint_analysis_amc_right_deepseek14b_no_perturb_add_small_all_num_all.jsonl"


#INPUT="output/checkpoint_analysis_hendrycks_math_test_deepseek_llama8b_right_no_perturb.jsonl"
INPUT="output/checkpoint_analysis_math_nemotron1d5b_right_no_perturb.jsonl"
#INPUT="output/checkpoint_analysis_aime_deepseek-llama8b_right_no_perturb_correct.jsonl"
USER_TAG="\n The final answer is \\boxed{"
SEEDS="100"

for SEED in $SEEDS; do
  OUTPUT_FILE="output/checkpoint_analysis_math_nemotron1d5b_${SEED}_add_small_all_num_perturb_s.jsonl"
  echo "Running seed=${SEED} -> ${OUTPUT_FILE}"
  python -u inference_checkpoint_analysis.py \
    --model "$MODEL" \
    --model_size "$MODEL_SIZE" \
    --input "$INPUT" \
    --output_file_name "$OUTPUT_FILE" \
    --left 0 \
    --right  300 \
    --no_store_logits \
    --no_store_distribution \
    --random_number_replacement \
    --replace_number_mode "random_add_small" \
    --replace_all_numbers \
    --random_seed ${SEED} \
    --perturb_rand_context_step 0 \
    --reverse_perturb 0 \
    --reverse_perturb_all 0
  # Clean up GPU memory after each run
  python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"
  

 OUTPUT_FILE="output/checkpoint_analysis_math_nemotron1d5b_${SEED}_add_small_all_num_perturb_c.jsonl"
  echo "Running seed=${SEED} -> ${OUTPUT_FILE}"
  python -u inference_checkpoint_analysis.py \
    --model "$MODEL" \
    --model_size "$MODEL_SIZE" \
    --input "$INPUT" \
    --output_file_name "$OUTPUT_FILE" \
    --left 0 \
    --right  300 \
    --no_store_logits \
    --no_store_distribution \
    --random_number_replacement \
    --replace_number_mode "random_add_small" \
    --replace_all_numbers \
    --random_seed ${SEED} \
    --perturb_rand_context_step 0 \
    --reverse_perturb 1 \
    --reverse_perturb_all 0
  # Clean up GPU memory after each run
  python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"
  
   OUTPUT_FILE="output/checkpoint_analysis_math_nemotron1d5b_${SEED}_add_small_all_num_perturb_s_c.jsonl"
  echo "Running seed=${SEED} -> ${OUTPUT_FILE}"
  python -u inference_checkpoint_analysis.py \
    --model "$MODEL" \
    --model_size "$MODEL_SIZE" \
    --input "$INPUT" \
    --output_file_name "$OUTPUT_FILE" \
    --left 0 \
    --right  300 \
    --no_store_logits \
    --no_store_distribution \
    --random_number_replacement \
    --replace_number_mode "random_add_small" \
    --replace_all_numbers \
    --random_seed ${SEED} \
    --perturb_rand_context_step 0 \
    --reverse_perturb 0 \
    --reverse_perturb_all 1
  # Clean up GPU memory after each run
  python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"


  echo "Done seed=${SEED}"
  echo
done
    