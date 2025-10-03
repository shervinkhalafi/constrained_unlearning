from my_utils import sample_loop, compute_likelihood, my_DDIMScheduler, evaluate, evaluate_clip_FID
from diffusers import StableDiffusionPipeline
import torch
import argparse
import sys
from accelerate import Accelerator
import wandb
from peft import LoraConfig, get_peft_model, TaskType
import copy
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from transformers import get_cosine_schedule_with_warmup
import os
from PIL import Image
import gc
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPModel, CLIPProcessor
from cleanfid import fid
import numpy as np






#arguments
def parse_args(input_args=None):

    parser = argparse.ArgumentParser(description='Training configuration for diffusion model')
    
    # General training parameters
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--use_wandb', type=int, default=1,
                       help='Use wandb')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                       help='Number of inference steps')
    parser.add_argument('--n_noise_samples', type=int, default=2,
                       help='Number of noise samples for KL computation')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lr_dual', type=float, default=0.0,
                       help='Learning rate for dual')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       help='Mixed precision')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=8,
                       help='LoRA alpha')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate every n epochs')
    parser.add_argument('--reward_type', type=str, default='likelihood',
                       help='Reward type')
    parser.add_argument('--prompt_unlearn', type=str, default='photo of a cat',
                       help='Unlearn prompt')
    parser.add_argument('--prompt_close', type=str, default='photo of a dog',
                       help='Close prompt')
    parser.add_argument('--prompt_far', type=str, default='impressionist painting',
                       help='Far prompt')
    parser.add_argument('--prompt_context', type=str, default='photo of a cat in a grass field',
                       help='Context prompt')
    parser.add_argument('--lamb_init', type=float, default=0.0,
                       help='Initial lambda')
    parser.add_argument('--b', type=float, default=0.0,
                       help='b')
    # Memory optimization arguments
    parser.add_argument('--enable_gradient_checkpointing', type=int, default=1,
                       help='Enable gradient checkpointing')
    parser.add_argument('--l_avg', type=int, default=5,
                       help='Length of running average window')
    
    parser.add_argument('--clip_num_batches', type=int, default=4,
                       help='Number of batches for CLIP evaluation')
    parser.add_argument('--fid_num_batches', type=int, default=20,
                       help='Number of batches for FID evaluation')
    
    parser.add_argument('--eval_batch_size', type=int, default=4,
                       help='Batch size for evaluation')

    
    parser.add_argument('--pre_compute_FID_stats', type=int, default=0,
                       help='Pre-compute FID stats')
    
    parser.add_argument('--clip_every', type=int, default=5,
                       help='Evaluate CLIP (and FID) every n epochs')
    parser.add_argument('--fid_every', type=int, default=10,
                       help='Evaluate FID every n times that clip is evaluated')
    parser.add_argument('--sweep_id', type=str, default=None,
                       help='Wandb sweep ID')


    if input_args is not None:
        print("Parsing provided arguments:", input_args)
        args = parser.parse_args(input_args)
    else:
        print("Parsing command line arguments:", sys.argv[1:])
        args = parser.parse_args()

    # Add debug print
    print("\nParsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()




def main(args):
    # Initialize device mapping for multi-GPU training BEFORE creating accelerator
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print(f"Process {local_rank} using GPU {local_rank}")

    # Create and setup UNet
    pipe = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-tiny", use_safetensors=True)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True, torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
    )

    unet = pipe.unet

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    
    # Enable gradient checkpointing for memory savings
    if args.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Basic fine-tuning training loop with memory optimizations
    if args.use_wandb:
        accelerator = Accelerator(
            mixed_precision="fp16", 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="wandb"
        )

        accelerator.init_trackers(project_name="unlearning_sweep_1", config=args)
    else:
        accelerator = Accelerator(
            mixed_precision="fp16", 
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

    # Move text_encoder to device once during initialization
    text_encoder = text_encoder.to(accelerator.device)
    clip_model = clip_model.to(accelerator.device)

    # Pre-compute text embeddings to avoid multi-GPU issues
    print("Pre-computing text embeddings...")
    with torch.no_grad():
        text_embeddings_list = []
        for prompt in [args.prompt_unlearn, args.prompt_close, args.prompt_far, args.prompt_context]:
            # Tokenize the prompt
            text_input = tokenizer(
                [prompt], 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            # Get text embeddings
            text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
            text_embeddings_list.append(text_embeddings)
        
    
        # Get unconditional embeddings
        uncond_input = tokenizer(
            [""], 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]
    
    print("Text embeddings pre-computed successfully")
    
    # Clean up after text encoding
    cleanup_memory()

    


    lamb = torch.tensor(args.lamb_init, requires_grad = False)
    b = torch.tensor(args.b, requires_grad = False)

    unet.train()



    unet_lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))


    scheduler = my_DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")


    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=args.lr,
        weight_decay=0.01,  # Add weight decay for regularization
    )

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10, num_training_steps=args.num_epochs)
    
    

    unet, optimizer, vae = accelerator.prepare(unet, optimizer, vae)

    r_running_avg = 0.0
    KL_running_avg = 0.0


    for epoch in range(args.num_epochs):

        epoch_loss = 0.0
        num_batches = 0
        epoch_KL = 0.0
        epoch_rewards = 0.0

        
        # Generate training data batches with gradient accumulation
        for batch_idx in range(args.gradient_accumulation_steps):
            with accelerator.accumulate(unet):
                #pre-compute FID stats if needed
                if args.pre_compute_FID_stats:
                    for i, prompt in enumerate([args.prompt_unlearn, args.prompt_close]):
                        prompt_clean = prompt.replace(" ", "_")
                        
                        custom_stats_name = prompt_clean + '_stats'
                        if fid.test_stats_exists(custom_stats_name, mode = 'clean'):
                            print(f"Custom stats {custom_stats_name} already exists")
                            fid.remove_custom_stats(custom_stats_name, mode = 'clean')
                        
                        print(f"Custom stats {custom_stats_name} does not exist, computing...")
                        save_dir = prompt_clean + '_folder'
                        with torch.no_grad():
                            unet.eval()
                            clip_score = evaluate_clip_FID(
                                accelerator,
                                scheduler,
                                unet,
                                vae,
                                tokenizer,
                                text_encoder,
                                clip_processor,
                                clip_model,
                                args.batch_size,
                                args,
                                epoch,
                                prompts=[prompt],
                                text_embeddings=text_embeddings_list[i],
                                uncond_embeddings=uncond_embeddings,
                                seed=42,
                                compute_FID=True,
                                save_dir=save_dir
                            )
                            fid.make_custom_stats(custom_stats_name, save_dir, mode = 'clean')

                
                #sample latents with gradients
                batch_log_probs, batch_KLs, batch_latents = sample_loop(
                    accelerator,
                    scheduler,
                    unet,
                    vae,
                    tokenizer,
                    text_encoder,
                    batch_size = args.batch_size,
                    with_log_prob=True,
                    with_KL=True,
                    num_inference_steps=args.num_inference_steps,
                    grad=True,
                    precomputed_text_embeddings=text_embeddings_list[0],
                    precomputed_uncond_embeddings=uncond_embeddings
                )

                with torch.no_grad():
                    unet.eval()


                    if (epoch%args.eval_every == 0 or epoch == args.num_epochs - 1) and batch_idx == 0:
                        evaluate(
                            accelerator,
                            scheduler,
                            unet,
                            vae,
                            tokenizer,
                            text_encoder,
                            args,
                            epoch,
                            torch.cat(text_embeddings_list),
                            uncond_embeddings,
                            caption_list=["unlearn concept", "close concept", "far concept", "unlearn concept in context"],
                            seed=42,
                            log_images=True
                        )

                    if (epoch%args.clip_every == 0 or epoch == args.num_epochs - 1) and batch_idx == 0:
                        if epoch % (args.fid_every*args.clip_every) == 0 or epoch == args.num_epochs - 1:
                            compute_FID = True
                            num_batches_eval = args.fid_num_batches
                        else:
                            compute_FID = False
                            num_batches_eval = args.clip_num_batches

                        clip_scores = []
                        fids = []
                        for i, prompt in enumerate([args.prompt_unlearn, args.prompt_close, args.prompt_far, args.prompt_context]):
                            clip_score, fid_score = evaluate_clip_FID(
                                accelerator,
                                scheduler,
                                unet,
                                vae,
                                tokenizer,
                                text_encoder,
                                clip_processor,
                                clip_model,
                                args.eval_batch_size, 
                                num_batches_eval,
                                args,
                                epoch,
                                prompts=[prompt],
                                text_embeddings=text_embeddings_list[i],
                                uncond_embeddings=uncond_embeddings,
                                seed=42,
                                compute_FID=compute_FID,
                                save_dir='temp_folder'
                            )
                            clip_scores.append(clip_score)
                            if compute_FID:
                                fids.append(fid_score)
                        clip_scores = torch.cat(clip_scores)
                        if compute_FID:
                            fids = np.array(fids)

                        
                        

                    # Disable adapters for reward computation
                    if hasattr(unet, 'module'):
                        # Multi-GPU case: unet is wrapped in DDP
                        unet.module.disable_adapters()
                    else:
                        # Single GPU case: unet is the model directly
                        unet.disable_adapters()
                    
                    # Compute rewards
                    rewards = compute_likelihood(
                        accelerator, 
                        scheduler, 
                        unet, 
                        vae, 
                        tokenizer, 
                        text_encoder, 
                        batch_latents, 
                        batch_size = args.batch_size,
                        precomputed_text_embeddings=text_embeddings_list[0],
                        precomputed_uncond_embeddings=uncond_embeddings,
                        num_inference_steps=args.num_inference_steps, 
                        n_noise_samples=args.n_noise_samples
                    )

                  
                    
                    
                    # print(rewards)
                    if args.reward_type == "log_likelihood":
                        pass
                    elif args.reward_type == "likelihood":
                        rewards = 0.001*torch.exp(rewards)
                    else:
                        raise ValueError(f"Invalid reward type: {args.reward_type}")

                    # lamb += (rewards.mean() - b)*lr_dual

                    # Re-enable adapters
                    if hasattr(unet, 'module'):
                        # Multi-GPU case: unet is wrapped in DDP
                        unet.module.enable_adapters()
                    else:
                        # Single GPU case: unet is the model directly
                        unet.enable_adapters()

                    # if lamb < 0.0:
                    #     lamb = 0.0
                    unet.train()
                


                r = (rewards*batch_log_probs).mean()
                KL = batch_KLs.mean()

                # FIXED: Check for NaN in loss
                if torch.isnan(r):
                    print(f"NaN loss detected in epoch {epoch+1}, batch {batch_idx+1}")

    
                
                    
                # Compute loss (MSE between predicted and actual noise)
                # Scale loss by gradient accumulation steps for proper averaging
                loss = (lamb*r + KL) / args.gradient_accumulation_steps

                # Backward pass
                #PRIMAL Gradient Step:
                accelerator.backward(loss)

                # Only update optimizer and lr_scheduler after accumulating all gradients
                if accelerator.sync_gradients:

                    accelerator.clip_grad_norm_(lora_layers, max_norm=1.0)

                    # Use accelerator's step method which handles mixed precision scaling
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    





                epoch_loss += loss.item() * args.gradient_accumulation_steps  # Undo scaling for logging
                num_batches += 1
                epoch_KL += KL.item()
                epoch_rewards += rewards.mean().item()
                
                # Clean up intermediate tensors
                del batch_log_probs, batch_KLs, batch_latents, rewards, r, loss
                cleanup_memory()
        
        avg_loss = epoch_loss / num_batches
        avg_KL = epoch_KL / num_batches
        avg_rewards = epoch_rewards / num_batches

        

        # Gather and average metrics across all processes
        avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
        avg_KL_tensor = torch.tensor(avg_KL, device=accelerator.device)
        avg_rewards_tensor = torch.tensor(avg_rewards, device=accelerator.device)
        lamb_tensor = torch.tensor(lamb.item() if hasattr(lamb, "item") else float(lamb), device=accelerator.device)
        r_running_avg_tensor = torch.tensor(r_running_avg, device=accelerator.device)
        KL_running_avg_tensor = torch.tensor(KL_running_avg, device=accelerator.device)
        clip_scores_tensor = torch.tensor(clip_scores, device=accelerator.device)

        # Gather across all processes
        avg_loss_all = accelerator.gather(avg_loss_tensor)
        avg_KL_all = accelerator.gather(avg_KL_tensor)
        avg_rewards_all = accelerator.gather(avg_rewards_tensor)
        lamb_all = accelerator.gather(lamb_tensor)
        r_running_avg_all = accelerator.gather(r_running_avg_tensor)
        KL_running_avg_all = accelerator.gather(KL_running_avg_tensor)
        len_clip_scores = len(clip_scores_tensor)
        clip_scores_all = (accelerator.gather(clip_scores_tensor)).reshape(-1, len_clip_scores)

        print('clip_scores_all shape:', clip_scores_all.shape)


        # Compute mean across all processes
        avg_loss_mean = avg_loss_all.float().mean().item()
        avg_KL_mean = avg_KL_all.float().mean().item()
        avg_rewards_mean = avg_rewards_all.float().mean().item()
        lamb_mean = lamb_all.float().mean().item()
        r_running_avg_mean = r_running_avg_all.float().mean().item()
        KL_running_avg_mean = KL_running_avg_all.float().mean().item()
        clip_scores_mean = clip_scores_all.float().mean(axis = 0)

        if epoch % args.l_avg == 0 and epoch > 0:
            r_running_avg_mean = r_running_avg_mean / args.l_avg
            KL_running_avg_mean = KL_running_avg_mean / args.l_avg

            if args.use_wandb and accelerator.is_main_process:
                accelerator.log({
                    "running_average_reward": r_running_avg_mean,
                    "running_average_KL": KL_running_avg_mean
                }, step=epoch)

            r_running_avg = 0.0
            KL_running_avg = 0.0

        r_running_avg += avg_rewards_mean
        KL_running_avg += avg_KL_mean

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss_mean:.6f}, Average KL: {avg_KL_mean:.6f}, Average Rewards: {avg_rewards_mean:.6f}")

        # Dual update start
        lamb = lamb + args.lr_dual * (avg_rewards_mean - b)
        if lamb < 0.0:
            lamb = 0.0 * lamb
        # Dual update end

        # Log to wandb (optional)
        if args.use_wandb and accelerator.is_main_process:
            accelerator.log({
                "reward (" + args.reward_type + ")": avg_rewards_mean,
                "loss": avg_loss_mean,
                "lr": lr_scheduler.get_last_lr()[0],
                "KL": avg_KL_mean,
                "lamb": lamb_mean,
            }, step=epoch)

            if epoch % args.clip_every == 0 or epoch == args.num_epochs - 1:
                for i, prompt in enumerate([args.prompt_unlearn, args.prompt_close, args.prompt_far, args.prompt_context]): 
                    accelerator.log({
                        "clip_score_" + prompt: clip_scores_mean[i].item()
                }, step=epoch)
                    if epoch % (args.fid_every*args.clip_every) == 0 or epoch == args.num_epochs - 1:
                        accelerator.log({
                            "fid_score_" + prompt: fids[i]
                        }, step=epoch)

        # Aggressive memory cleanup between epochs
        cleanup_memory()

    print("Fine-tuning completed!")

    # Finish wandb run (optional)
    if args.use_wandb:
        accelerator.end_training()
    
    return unet


if __name__ == "__main__":
    args = parse_args()
    main(args)