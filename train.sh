#!/bin/bash


    
CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29502 --mixed_precision=fp16 --num_processes=1 train.py \
                --num_epochs=400 \
                --num_inference_steps=10 \
                --lr=1e-4 \
                --batch_size=1 \
                --lora_r=8 \
                --lora_alpha=8 \
                --use_wandb=0 \
                --mixed_precision=fp16 \
                --gradient_accumulation_steps=1 \
                --prompt_unlearn="photo of a cat" \
                --prompt_close="photo of a dog" \
                --prompt_far="impressionist painting" \
                --prompt_context="photo of a cat in a grass field" \
                --eval_every=10 \
                --reward_type="likelihood" \
                --lamb_init=0.0 \
                --lr_dual=10000.0 \
                --b=0.00104 \
                --gradient_accumulation_steps=1
