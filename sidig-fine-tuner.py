import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor2_5
from diffusers.optimization import get_scheduler


Image.MAX_IMAGE_PIXELS = None
IMAGES_DIR = Path("images")
METADATA_FILE = Path("metadata/labels.jsonl")
MODEL_PATH = Path("models/sdxl-turbo")
OUTPUT_DIR = Path("models/sdxl-turbo-lora")
RESOLUTION = 1024
TRAIN_BATCH_SIZE = 1
MAX_STEPS = 1000
LEARNING_RATE = 1e-4
RANK = 16
LR_SCHEDULER = "cosine"
LR_WARMUP_STEPS = 100
SEED = 42


def attach_lora(unet, rank: int):
    procs = {}
    for name in unet.attn_processors.keys():
        cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        procs[name] = LoRAAttnProcessor2_5(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            rank=rank,
        )

    unet.set_attn_processor(procs)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file {METADATA_FILE} is missing.")

    samples = []
    meta_root = METADATA_FILE.parent.resolve()
    with METADATA_FILE.open() as fh:
        for raw_line in fh:
            data = json.loads(raw_line)
            prompt = data.get("label", "").strip()
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
        raise ValueError("No training samples found. Check images and metadata paths.")

    if len(samples) < TRAIN_BATCH_SIZE:
        raise ValueError("Need at least as many samples as the batch size.")

    transform = transforms.Compose(
        [
            transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16 if device.type != "cpu" else torch.float32, variant="fp16", use_safetensors=True)
    pipe.to(device)
    pipe.unet.enable_gradient_checkpointing()
    pipe.vae.requires_grad_(False)

    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    attach_lora(pipe.unet, RANK)
    lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-2)
    lr_scheduler = get_scheduler(LR_SCHEDULER, optimizer=optimizer, num_warmup_steps=LR_WARMUP_STEPS, num_training_steps=MAX_STEPS)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = noise_scheduler
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    order = list(range(len(samples)))
    random.shuffle(order)

    cursor = 0
    global_step = 0

    progress = tqdm(total=MAX_STEPS, desc="Training LoRA", dynamic_ncols=True)

    while global_step < MAX_STEPS:
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
            prompt_outputs = pipe.encode_prompt(
                prompt=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            prompt_embeds = prompt_outputs[0]
            pooled_prompt_embeds = prompt_outputs[2]
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        add_time_ids = []
        for size in original_sizes:
            h, w = size
            add_time_ids.append([h, w, RESOLUTION, RESOLUTION, 0, 0])
        add_time_ids = torch.tensor(add_time_ids, device=device, dtype=prompt_embeds.dtype)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type != "cpu"):
            model_pred = pipe.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

        global_step += 1
        progress.update(1)
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    progress.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe.save_lora_weights(OUTPUT_DIR)
    print(f"Saved LoRA to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
