#!/bin/bash

# Set NCCL environment variables to fix device mapping
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Sweep parameters
for lr in 2e-5 1e-4; do
  for lamb_init in 0.01 0.2 1.0; do
    for reward_type in "log_likelihood" "likelihood"; do
      CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --main_process_port 29502 \
        --mixed_precision=fp16 \
        --num_processes=2 \
        train.py \
        --num_epochs=200 \
        --num_inference_steps=10 \
        --lr=$lr \
        --batch_size=4 \
        --lora_r=8 \
        --lora_alpha=8 \
        --use_wandb=1 \
        --mixed_precision=fp16 \
        --gradient_accumulation_steps=1 \
        --prompt_unlearn="photo of a cat" \
        --prompt_close="photo of a dog" \
        --prompt_far="impressionist painting" \
        --prompt_context="photo of a cat in a grass field" \
        --eval_every=5 \
        --reward_type="$reward_type" \
        --lamb_init=$lamb_init \
        --lr_dual=0.0 \
        --b=0.00104 \
        --n_noise_samples=2 \
        --clip_num_batches=4 \
        --fid_num_batches=50 \
        --eval_batch_size=4 \
        --pre_compute_FID_stats=0 \
        --clip_every=5 \
        --fid_every=10
    done
  done
done

                
