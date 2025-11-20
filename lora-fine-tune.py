import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_5


Image.MAX_IMAGE_PIXELS = None

IMAGES_DIR = "images"
META_FILE = "metadata/labels.jsonl"
MODEL_ID = "stabilityai/sdxl-turbo"
OUT_DIR = "models/sdxl-turbo-lora"

RES = 1024
BATCH = 1
MAX_STEPS = 1000
lr = 1e-5
RANK = 16
SCHED_NAME = "cosine"
WARMUP = 100
SEED = 42 # of course


def attach_lora(unet, rank):
    procs = {}
    blocks = unet.config.block_out_channels

    for name, process in unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            cross_dim = None
        else:
            cross_dim = unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden = blocks[-1]
        elif name.startswith("up_blocks"):
            parts = name.split(".")
            try:
                i = int(parts[1])
            except (IndexError, ValueError):
                i = 0
            hidden = list(blocks[::-1])[i]
        elif name.startswith("down_blocks"):
            parts = name.split(".")
            try:
                i = int(parts[1])
            except (IndexError, ValueError):
                i = 0
            hidden = blocks[i]
        else:
            hidden = blocks[0]

        procs[name] = LoRAAttnProcessor2_5(
            hidden_size=hidden,
            cross_attention_dim=cross_dim,
            rank=rank,
        )

    unet.set_attn_processor(procs)
    return [p for p in unet.parameters() if p.requires_grad]


def load_samples():
    meta_path = Path(META_FILE)
    if not meta_path.exists():
        raise RuntimeError("metadata file not found")

    img_root = Path(IMAGES_DIR)
    meta_root = meta_path.parent
    items = []

    with meta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = f"spaceisdirty {data['label'].strip()}"
            if not prompt:
                continue

            rel = Path(data["image"])
            cands = []
            if rel.is_absolute():
                cands.append(rel)
            else:
                cands.append(meta_root / rel)
                cands.append(img_root / rel)
                cands.append(img_root / rel.name)

            img_path = next((c for c in cands if c.exists()), None)
            if img_path is None:
                continue

            items.append({"prompt": prompt, "path": img_path})

    if not items:
        raise RuntimeError("no images found")

    if len(items) < BATCH:
        raise RuntimeError("batch size too big for dataset")

    return items


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    samples = load_samples()
    idx_order = list(range(len(samples)))
    random.shuffle(idx_order)
    cursor = 0

    tform = transforms.Compose(
        [
            transforms.Resize(RES, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(RES),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    pipe.unet.to(device=device, dtype=torch.float32)
    pipe.vae.to(device=device, dtype=torch.float32)

    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.enable_gradient_checkpointing()

    lora_params = attach_lora(pipe.unet, RANK)
    pipe.unet.train()

    opt = torch.optim.AdamW(
        lora_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    sched = get_scheduler(
        SCHED_NAME,
        optimizer=opt,
        num_warmup_steps=WARMUP,
        num_training_steps=MAX_STEPS,
    )

    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = noise_sched

    bar = tqdm(total=MAX_STEPS, dynamic_ncols=True)
    step = 0
    success = True

    while step < MAX_STEPS:
        batch = []
        while len(batch) < BATCH:
            if cursor >= len(idx_order):
                random.shuffle(idx_order)
                cursor = 0
            batch.append(samples[idx_order[cursor]])
            cursor += 1

        imgs = []
        prompts = []
        sizes = []

        for item in batch:
            with Image.open(item["path"]) as im:
                im = im.convert("RGB")
                sizes.append((im.height, im.width))
                imgs.append(tform(im))
                prompts.append(item["prompt"])

        x = torch.stack(imgs).to(device=device, dtype=pipe.vae.dtype)

        with torch.no_grad():
            enc = pipe.encode_prompt(
                prompt=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            pe = enc[0]
            pooled = enc[2]
            lat = pipe.vae.encode(x).latent_dist.sample()
            lat = lat * pipe.vae.config.scaling_factor

        noise = torch.randn_like(lat)
        ts = torch.randint(
            0,
            noise_sched.config.num_train_timesteps,
            (lat.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy = noise_sched.add_noise(lat, noise, ts)

        t_ids = []
        for h, w in sizes:
            t_ids.append([h, w, RES, RES, 0, 0])
        t_ids = torch.tensor(t_ids, device=device, dtype=pe.dtype)

        lat = lat.to(device=device, dtype=torch.float32)
        noise = noise.to(device=device, dtype=torch.float32)
        noisy = noisy.to(device=device, dtype=torch.float32)
        pe = pe.to(device=device, dtype=torch.float32)
        pooled = pooled.to(device=device, dtype=torch.float32)
        t_ids = t_ids.to(device=device, dtype=torch.float32)

        out = pipe.unet(
            noisy,
            ts,
            pe,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": t_ids},
        ).sample

        loss = F.mse_loss(out, noise, reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        sched.step()

        step += 1
        bar.update(1)
        bar.set_postfix({"loss": float(loss.detach().cpu())})

    bar.close()

    if success:
        out_dir = Path(OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "lora.safetensors"
        pipe.save_lora_weights(str(path))
        print("saved", str(path))


if __name__ == "__main__":
    main()
