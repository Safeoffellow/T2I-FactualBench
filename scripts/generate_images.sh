#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  src/generate_all.py \
  --model sdxl \
  --prompt_dir data/prompts \
  --output_dir results/ \
  --level SKCM \
  --knowledge_injection text
