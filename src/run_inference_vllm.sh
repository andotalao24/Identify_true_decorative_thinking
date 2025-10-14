#!/usr/bin/env bash
set -e


python inference_vllm.py \
  --model nemotron \
  --model_size 1.5b \
  --input data/amc.jsonl \
  --output_file_name output/output_amc_nemotron1d5b.jsonl \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.5 \
  --batch_size 1 \
  --use_batching 1 \
  --temperature 0.0 \
  --top_p 0.0 \
  --do_sample_decode 0 \
  --use_template 1 \
  --left 0 \
  --right 999 \
  --infer_on_perturbed_step 0 \
  --add_reasoning_step 0 \
  --omit_thinking 0 \
  --custom_prompt ""