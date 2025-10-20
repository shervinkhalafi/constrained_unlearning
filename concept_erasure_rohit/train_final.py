import os 
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPModel, CLIPProcessor
import argparse
import wandb
from PIL import Image


sys.path.append('.')
from utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call

def load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16, device='cuda:0'):
    
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    base_unet.requires_grad_(False)
    
    esd_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    return pipe, base_unet, esd_unet

def get_esd_trainable_parameters(esd_unet, train_method='esd-x'):
    esd_params = []
    esd_param_names = []
    for name, module in esd_unet.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and 'attn2' in name:
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-u' and ('attn2' not in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-all' :
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

    return esd_param_names, esd_params


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for SDv1.4',
                    description = 'Finetuning stable-diffusion to erase the concepts')
    parser.add_argument('--concept_pair', help='(erase_concept, close_concept)', type=str, required=True)
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=50)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=3)
    
    parser.add_argument('--train_method', help='Type of method (esd-x, esd-u, esd-a, esd-x-strict)', type=str, required=True)
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=50)
    parser.add_argument('--lr', help='Learning rate', type=float, default=5e-5)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=2)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/sd/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--use_wandb', help='Use wandb', type=int, default=1)
    parser.add_argument('--ratio_scale', help='Ratio scale', type=float, default=1)
    parser.add_argument('--use_likelihood_ratio', help='Use likelihood ratio', type=int, default=0)
    parser.add_argument('--training_seed', help='Training seed', type=int, default=42)
    parser.add_argument('--beta_1', help='Beta 1', type=float, default=0.9)
    parser.add_argument('--beta_2', help='Beta 2', type=float, default=0.999)
    parser.add_argument('--num_clip_batches', help='Number of clip batches', type=int, default=2)
    parser.add_argument('--clip_batch_size', help='Clip batch size', type=int, default=4)
    
    args = parser.parse_args()

    erase_concept = args.concept_pair.split('-')[1]
    # close_concept = args.concept_pair.split('-')[1]
    erase_concept_from = args.concept_pair.split('-')[0]

    num_inference_steps = args.num_inference_steps
    
    guidance_scale = args.guidance_scale
    negative_guidance = args.negative_guidance
    train_method=args.train_method
    iterations = args.iterations
    batchsize = args.batch_size
    height=width=512
    lr = args.lr
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    torch_dtype = torch.bfloat16
    training_seed = args.training_seed

    # Set the random seed for reproducibility
    import random
    import numpy as np

    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    random.seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    
    criteria = torch.nn.MSELoss()

    base_model_id = "CompVis/stable-diffusion-v1-4"
    # base_model_id = "nota-ai/bk-sdm-tiny"

    pipe, base_unet, esd_unet = load_sd_models(basemodel_id=base_model_id, torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
    )
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_model.to(esd_unet.device)

    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(esd_params, lr=lr)

    if args.use_wandb:
        wandb.init(project="esd-sdv1.4-rohit-from", config=args)

    with torch.no_grad():
        # get prompt embeds
        erase_embeds, null_embeds = pipe.encode_prompt(prompt=erase_concept,
                                                       device=device,
                                                       num_images_per_prompt=batchsize,
                                                       do_classifier_free_guidance=True,
                                                       negative_prompt='')
        
        # close_embeds, _ = pipe.encode_prompt(prompt=close_concept,
        #                                                device=device,
        #                                                num_images_per_prompt=batchsize,
        #                                                do_classifier_free_guidance=True,
        #                                                negative_prompt='')
                                                 
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=torch_dtype)
        
        if erase_concept_from is not None:
            erase_from_embeds, _ = pipe.encode_prompt(prompt=erase_concept_from,
                                                                device=device,
                                                                num_images_per_prompt=batchsize,
                                                                do_classifier_free_guidance=False,
                                                                negative_prompt="",
                                                                )
            erase_from_embeds = erase_from_embeds.to(device)
    
    
    pbar = tqdm(range(iterations), desc='Training ESD')
    losses = []
    for iteration in pbar:
        optimizer.zero_grad()
        # get the noise predictions for erase concept
        pipe.unet = base_unet
        run_till_timestep = random.randint(0, num_inference_steps-1)
        
        run_till_timestep_scheduler = pipe.scheduler.timesteps[run_till_timestep] #runs inference only till this timestep
        seed = random.randint(0, 2**15)
        fixed_seed = 42
        with torch.no_grad():
            #evaluation:
            pipe.unet = esd_unet
            xt = pipe(erase_concept if erase_concept_from is None else erase_concept_from,
                  num_images_per_prompt=4,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  run_till_timestep = num_inference_steps,
                  generator=torch.Generator().manual_seed(fixed_seed),
                #   output_type='latent',
                  height=height,
                  width=width,
                 ).images

            def make_image_grid(images, rows=2, cols=2):
                # assumes each image in images is a PIL Image
                w, h = images[0].size
                grid = Image.new('RGB', size=(cols * w, rows * h))
                for i, img in enumerate(images):
                    r, c = divmod(i, cols)
                    grid.paste(img, (c * w, r * h))
                return grid

            image = make_image_grid(xt, rows=2, cols=2)

            # close_image = pipe(close_concept,
            #       num_images_per_prompt=4,
            #       num_inference_steps=num_inference_steps,
            #       guidance_scale=guidance_scale,
            #       run_till_timestep = num_inference_steps,
            #       generator=torch.Generator().manual_seed(fixed_seed + 1),
            #       height=height,
            #       width=width,
            #      ).images

            # close_image = make_image_grid(close_image, rows=2, cols=2)

            if args.use_wandb:
                wandb.log({f"erase_concept": wandb.Image(image)}, step=iteration)
                # wandb.log({f"close_concept": wandb.Image(close_image)}, step=iteration)

            #compute clip:
            clip_score = 0
            for i in range(args.num_clip_batches):
                images = pipe(erase_concept,
                    num_images_per_prompt=args.clip_batch_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    run_till_timestep = num_inference_steps,
                    height=height,
                    width=width,
                ).images

                inputs = clip_processor(text=erase_concept, images=images, return_tensors="pt", padding=True).to(device)
                outputs = clip_model(**inputs)
                batch_clip_score = outputs.logits_per_image.detach()
                clip_score += batch_clip_score.mean()

            clip_score /= args.num_clip_batches
            print(f"Erase concept clip score: {clip_score}")
            if args.use_wandb:
                wandb.log({f"erase_concept_clip_score": clip_score}, step=iteration)

            # close_clip_score = 0
            # for i in range(args.num_clip_batches):
            #     images = pipe(close_concept,
            #         num_images_per_prompt=args.clip_batch_size,
            #         num_inference_steps=num_inference_steps,
            #         guidance_scale=guidance_scale,
            #         run_till_timestep = num_inference_steps,
            #         height=height,
            #         width=width,
            #     ).images

            #     inputs = clip_processor(text=close_concept, images=images, return_tensors="pt", padding=True).to(device)
            #     outputs = clip_model(**inputs)
            #     batch_clip_score = outputs.logits_per_image.detach()
            #     close_clip_score += batch_clip_score.mean()
            
            # close_clip_score /= args.num_clip_batches
            # print(f"Close concept clip score: {close_clip_score}")
            # if args.use_wandb:
            #     wandb.log({f"close_concept_clip_score": close_clip_score}, step=iteration)
            #end of evaluation

            # pipe.unet = base_unet

            #computing the likelihood ratio
            #begin

            if args.use_likelihood_ratio:

                xt = pipe(erase_concept if erase_concept_from is None else erase_concept_from,
                    num_images_per_prompt=batchsize,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    run_till_timestep = num_inference_steps,
                    generator=torch.Generator().manual_seed(seed),
                    output_type='latent',
                    height=height,
                    width=width,
                    ).images


                
                Ix = torch.zeros(len(pipe.scheduler.timesteps), requires_grad=False)
                Ic = torch.zeros(len(pipe.scheduler.timesteps), requires_grad=False)

                for i, t in enumerate(pipe.scheduler.timesteps):
                    #some if statements to check if t is in run_till_timestep_scheduler?
                    noise = torch.randn_like(xt)
                    x_noisy = pipe.scheduler.add_noise(xt, noise, t)
                    
                    pipe.unet = base_unet

                    #get the noise predictions for erase concept
                    noise_pred_erase = pipe.unet(
                            xt,
                            t,
                            encoder_hidden_states=erase_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=None,
                            added_cond_kwargs=None,
                            return_dict=False,
                        )[0]
                        
                    # get the noise predictions for null embeds
                    noise_pred_null = pipe.unet(
                        xt,
                        t,
                        encoder_hidden_states=null_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=None,
                        return_dict=False,
                    )[0]

                    Ix[i] = torch.mean((noise - noise_pred_null)**2)
                    Ic[i] = torch.mean((noise - noise_pred_erase)**2)
                exponent = torch.mean(Ix[:run_till_timestep] - Ic[:run_till_timestep])
                print(exponent)
                ratio = torch.exp(args.ratio_scale*exponent)
                print(ratio)
                # ratio = args.ratio_scale*torch.exp(exponent)

                # If ratio is nan, skip this iteration
                if torch.isnan(ratio):
                    continue

            else:
                ratio = 1

            #end of computing the likelihood ratio

            pipe.unet = esd_unet

            xt = pipe(erase_concept if erase_concept_from is None else erase_concept_from,
                num_images_per_prompt=batchsize,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                run_till_timestep = run_till_timestep,
                generator=torch.Generator().manual_seed(seed),
                output_type='latent',
                height=height,
                width=width,
                ).images


            pipe.unet = base_unet

            noise_pred_erase = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=erase_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for null embeds
            noise_pred_null = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=null_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]



            
            
            # get the noise predictions for erase concept from embeds
            if erase_concept_from is not None:
                noise_pred_erase_from = pipe.unet(
                    xt,
                    run_till_timestep_scheduler,
                    encoder_hidden_states=erase_from_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase
        
        
        pipe.unet = esd_unet
        noise_pred_esd_model = pipe.unet(
            xt,
            run_till_timestep_scheduler,
            encoder_hidden_states=erase_embeds if erase_concept_from is None else erase_from_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        
        #case 1 (original method)
        # loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_null))) 
        #case 2 (likelihood ratio)
        loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*ratio*(noise_pred_erase - noise_pred_null)))


        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix(esd_loss=loss.item(),
                         timestep=run_till_timestep,)
        optimizer.step()

        if args.use_wandb and args.use_likelihood_ratio:
            wandb.log({f"esd_loss": loss.item(), "ratio": ratio, "exponent": exponent}, step=iteration)
    
    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    if erase_concept_from is None:
        erase_concept_from = erase_concept
        
    save_file(esd_param_dict, f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{erase_concept_from.replace(' ', '_')}-{train_method.replace('-','')}.safetensors")