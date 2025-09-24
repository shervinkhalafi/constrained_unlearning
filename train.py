from my_utils import sample_loop, compute_likelihood, my_DDIMScheduler, evaluate
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






#arguments
def parse_args(input_args=None):

    parser = argparse.ArgumentParser(description='Training configuration for diffusion model')
    
    # General training parameters
    parser.add_argument('--num_epochs', type=int, default=400,
                       help='Number of epochs')
    parser.add_argument('--use_wandb', type=int, default=1,
                       help='Use wandb')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                       help='Number of inference steps')
    parser.add_argument('--n_noise_samples', type=int, default=8,
                       help='Number of noise samples for KL computation')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lr_dual', type=float, default=0.1,
                       help='Learning rate for dual')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       help='Mixed precision')
    parser.add_argument('--lora_r', type=int, default=4,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=4,
                       help='LoRA alpha')
    parser.add_argument('--eval_every', type=int, default=10,
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





def main(args):

    # Set CUDA_VISIBLE_DEVICES to GPU 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Create and setup UNet
    pipe = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-tiny")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True, torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
    )

    unet = pipe.unet

    # Basic fine-tuning training loop
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.gradient_accumulation_steps)

    # Move text_encoder to device once during initialization
    text_encoder = text_encoder.to(accelerator.device)

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

    # Initialize wandb (optional)
    if args.use_wandb:
        wandb.init(project="reverse_kl_unlearning_stablediff")
        wandb.config.update(vars(args))


 

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
    )

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10, num_training_steps=args.num_epochs)
    

    unet, optimizer, vae = accelerator.prepare(unet, optimizer, vae)

    for epoch in range(args.num_epochs):

        # if epoch == 1:
        #     assert False

        # if epoch == 1:
        #     print("LoRA parameters:")
        #     for name, param in unet.named_parameters():
        #         if param.requires_grad:
        #             print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
        #             print(param)
        #     assert False
        epoch_loss = 0.0
        num_batches = 0
        epoch_KL = 0.0
        epoch_rewards = 0.0

        
        # Generate training data batches with gradient accumulation
        for batch_idx in range(args.gradient_accumulation_steps):
            
            with accelerator.accumulate(unet):

                
                
                # Pass pre-computed embeddings instead of computing them in sample_loop
                # random_int = torch.randint(0, 100000, (1,)).item()
                
                log_probs, KL, latents = sample_loop(accelerator, scheduler, unet, vae, tokenizer, text_encoder, [args.prompt_unlearn], with_log_prob = True, with_KL = True, num_inference_steps = args.num_inference_steps, grad = True, precomputed_text_embeddings=text_embeddings_list[0], precomputed_uncond_embeddings=uncond_embeddings)
       
                # latents = latents.detach()

                with torch.no_grad():

                    unet.eval()


                    if epoch%args.eval_every == 0 and batch_idx == 0:  # Only evaluate once per epoch
                        evaluate(accelerator, scheduler, unet, vae, tokenizer, text_encoder, args, epoch, text_embeddings_list[0], uncond_embeddings, caption = "unlearn concept", seed = 42)
                        evaluate(accelerator, scheduler, unet, vae, tokenizer, text_encoder, args, epoch, text_embeddings_list[1], uncond_embeddings, caption = "close concept", seed = 42)
                        evaluate(accelerator, scheduler, unet, vae, tokenizer, text_encoder, args, epoch, text_embeddings_list[2], uncond_embeddings, caption = "far concept", seed = 42)
                        evaluate(accelerator, scheduler, unet, vae, tokenizer, text_encoder, args, epoch, text_embeddings_list[3], uncond_embeddings, caption = "unlearn concept in context", seed = 42)
        
                        
                    
                    

                    

                    unet.disable_adapters()


                    
                    # Compute rewards without LoRA modifications
                    rewards = (compute_likelihood(accelerator, scheduler, unet, vae, tokenizer, text_encoder, latents[0], [args.prompt_unlearn], num_inference_steps=args.num_inference_steps, n_noise_samples = args.n_noise_samples))
                    # print(rewards)
                    if args.reward_type == "log_likelihood":
                        pass
                    elif args.reward_type == "likelihood":
                        rewards = 0.001*torch.exp(rewards)
                    else:
                        raise ValueError(f"Invalid reward type: {args.reward_type}")

                    # lamb += (rewards.mean() - b)*lr_dual

                    unet.enable_adapters()

                    # if lamb < 0.0:
                    #     lamb = 0.0
                    unet.train()
                    

                r = rewards*log_probs.mean()

                # FIXED: Check for NaN in loss
                if torch.isnan(r):
                    print(f"NaN loss detected in epoch {epoch+1}, batch {batch_idx+1}")
                    
                # Compute loss (MSE between predicted and actual noise)
                # Scale loss by gradient accumulation steps for proper averaging
                loss = (lamb*r + KL) / args.gradient_accumulation_steps

                # Backward pass
                accelerator.backward(loss)

                # Only update optimizer and lr_scheduler after accumulating all gradients
                if accelerator.sync_gradients:
                    # Check for inf/nan gradients and handle them
                    # for param in lora_layers:
                    #     if param.grad is not None:
                    #         if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    #             print(f"WARNING: Parameter has inf/nan gradients, setting to zero")
                    #             param.grad.zero_()

                    # Skip gradient clipping for now to avoid FP16 unscale issues
                    # We'll handle gradient issues by checking for inf/nan above
                    accelerator.clip_grad_norm_(lora_layers, max_norm=1.0)

                    # Use accelerator's step method which handles mixed precision scaling
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    



                # Print gradients of trainable parameters
                # print(f"Batch {batch_idx+1}, Epoch {epoch+1} - Gradients:")
                # for name, param in unet.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         grad_norm = param.grad.norm().item()
                #         print(f"  {name}: grad_norm={grad_norm:.6f}")

                # print(f"Batch {batch_idx+1}, Epoch {epoch+1} - Gradients:")
                # for name, param in unet.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         grad_norm = param.data.norm().item()
                #         print(f"  {name}: param_norm={grad_norm:.6f}")

                epoch_loss += loss.item() * args.gradient_accumulation_steps  # Undo scaling for logging
                num_batches += 1
                epoch_KL += KL.item()
                epoch_rewards += rewards.mean().item()
                # del loss
        
        avg_loss = epoch_loss / num_batches
        avg_KL = epoch_KL / num_batches
        avg_rewards = epoch_rewards / num_batches
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.6f}, Average KL: {avg_KL:.6f}, Average Rewards: {avg_rewards:.6f}")

        lamb = lamb + args.lr_dual*(avg_rewards - b)
        if lamb < 0.0:
            lamb = 0.0*lamb



        
        # Log to wandb (optional)
        if args.use_wandb:
            wandb.log({
                "reward (" + args.reward_type + ")": avg_rewards,
                "loss": avg_loss,
                "lr": lr_scheduler.get_last_lr()[0],
                "KL": avg_KL,
                "lamb": lamb.item(),
            }, step=epoch)

        
        # Clean up memory between epochs
        # del log_probs, latents, rewards, r
        # torch.cuda.empty_cache()

    print("Fine-tuning completed!")

    # Finish wandb run (optional)
    if args.use_wandb:
        wandb.finish()
    
    return unet


if __name__ == "__main__":
    args = parse_args()
    main(args)