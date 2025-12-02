import json, torch, random,sys, warnings, argparse
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from peft import LoraConfig
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler

Image.MAX_IMAGE_PIXELS = None
IMAGES_DIR = Path("images")
METADATA_FILE = Path("metadata/labels.jsonl")
OUTPUT_DIR = Path("models/sdxl-lora")
RESOLUTION = 1024
TRAIN_BATCH_SIZE = 1
LR_SCHEDULER = "cosine"
LR_WARMUP_STEPS = 100
SEED = 42


def attach_lora(pipe, rank: int):
    unet_lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                                   init_lora_weights="gaussian",)
    pipe.unet.add_adapter(unet_lora_config)
    return [p for p in pipe.unet.parameters() if p.requires_grad]


def main(args):
    succeed = True
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file {METADATA_FILE} is missing.")

    samples = []
    meta_root = METADATA_FILE.parent.resolve()
    with METADATA_FILE.open() as fh:
        for raw_line in fh:
            data = json.loads(raw_line)
            # use custom token prefix
            prompt = f"{args.token} {data.get('description', '').strip()}"
            if not prompt:
                continue

            raw_path = Path(data["image"])
            if raw_path.is_absolute():
                candidates = [raw_path]
            else:
                candidates = [
                    (meta_root / raw_path).resolve(),
                    (IMAGES_DIR / raw_path).resolve(),
                    (IMAGES_DIR / raw_path.name).resolve(),
                ]

            image_path = next((p for p in candidates if p.exists()), None)
            if image_path is None:
                continue

            samples.append({"prompt": prompt, "path": image_path})

    if not samples:
        raise ValueError("No training samples found. Where are the images?")

    if len(samples) < TRAIN_BATCH_SIZE:
        raise ValueError("Need at least as many samples as the batch size.")

    transform = transforms.Compose(
        [   transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    # load pipeline here
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    pipe.unet.to(device=device, dtype=torch.float32)
    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.enable_gradient_checkpointing()

    lora_params = attach_lora(pipe, args.rank)
    pipe.unet.train()
    
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    lr_scheduler = get_scheduler(LR_SCHEDULER, optimizer=optimizer, num_warmup_steps=LR_WARMUP_STEPS, num_training_steps=args.steps)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = noise_scheduler

    order = list(range(len(samples)))
    random.shuffle(order)

    cursor = 0
    step = 0

    progress = tqdm(total=args.steps, desc="Training LoRA", dynamic_ncols=True)

    while step < args.steps:
        batch_info = []
        while len(batch_info) < TRAIN_BATCH_SIZE:
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
                
            batch_info.append(samples[order[cursor]])
            cursor += 1

        pixel_values = []
        prompts = []
        original_sizes = []

        for entry in batch_info:
            with Image.open(entry["path"]) as img:
                img = img.convert("RGB")
                original_sizes.append((img.height, img.width))
                pixel_values.append(transform(img))
                prompts.append(entry["prompt"])

        pixel_values = torch.stack(pixel_values).to(device=device, dtype=pipe.vae.dtype)

        with torch.no_grad():
            prompt_outputs = pipe.encode_prompt(prompt=prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
            prompt_embeds = prompt_outputs[0]
            pooled_prompt_embeds = prompt_outputs[2]
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        add_time_ids = []
        for size in original_sizes:
            h, w = size
            add_time_ids.append([h, w, RESOLUTION, RESOLUTION, 0, 0])
        add_time_ids = torch.tensor(add_time_ids, device=device, dtype=prompt_embeds.dtype)

        latents = latents.to(device=device, dtype=torch.float32)
        noise = noise.to(device=device, dtype=torch.float32)
        noisy_latents = noisy_latents.to(device=device, dtype=torch.float32)
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch.float32)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch.float32)
        add_time_ids = add_time_ids.to(device=device, dtype=torch.float32)

        assert torch.isfinite(latents).all(), "latents has NaN/Inf"
        assert torch.isfinite(noise).all(), "noise has NaN/Inf"
        assert torch.isfinite(noisy_latents).all(), "noisy_latents has NaN/Inf"

        model_pred = pipe.unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs={"text_embeds": pooled_prompt_embeds, 
                                                                                           "time_ids": add_time_ids}).sample

        if not torch.isfinite(model_pred).all():
            print(f"NaN/Inf in model_pred at step {step}")
            succeed = False
            break

        loss = F.mse_loss(model_pred, noise, reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        step += 1
        progress.update(1)
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    progress.close()
    
    if succeed:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR /  OUTPUT_NAME
        pipe.unet.save_lora_adapter(OUTPUT_DIR, adapter_name="default", weight_name=OUTPUT_NAME)
        print(f"Saved LoRA to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LoRA for Stable Diffusion XL Turbo")
    parser.add_argument("--data-dir", "-d", type=str, default="images/sidig-images", help="Number of training steps")
    parser.add_argument("--model-path", "-m", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                        help="Pretrained model path")
    parser.add_argument("--output-name", "-o", type=str, default="lora", 
                          help="Output filename for LoRA weights")
    parser.add_argument("--token", "-t", type=str, default="spaceisdirty", help="Custom token prefix for training")
    parser.add_argument("--steps", "-s", type=int, default=800, help="Number of training steps")
    parser.add_argument("--lr", "-l", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--rank", "-r", type=int, default=32, help="LoRA rank")
    args = parser.parse_args()

    IMAGES_DIR = Path(args.data_dir)
    METADATA_FILE = Path(args.data_dir) / "labels.jsonl"
    OUTPUT_NAME = f"{args.output_name}.safetensors"

    main(args)
    # check the lora fine tune succeded
    from safetensors.torch import load_file
    sd = load_file(OUTPUT_DIR / OUTPUT_NAME)
    print(len(sd), list(sd.keys())[:5])