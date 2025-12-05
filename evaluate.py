from diffusers import AutoPipelineForText2Image
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import numpy as np
import os

OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "evaluation-images")
NP_DIR = os.path.join(OUTPUT_DIR, "evaluation-np")

def set_device_and_optimizations(pipe):
    device = "cpu"
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        device = "cuda"
        try:
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
            print("xFormers enabled")
        except Exception:
            print("xFormers not available, using default attention")
    elif torch.backends.mps.is_available():
        pipe = pipe.to("mps")
        device = "mps"

    return device

def calculate_clip_score(clip_score_fn, images, prompts):
    images_int = (images * 255).astype("uint8")
    output_clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(output_clip_score), 4)

def load_images():
    images_base = np.load(os.path.join(NP_DIR, "images_base.npy"))
    images_lora = np.load(os.path.join(NP_DIR, "images_lora.npy"))
    return images_base, images_lora

def load_prompts():
    prompts = []
    with open('output/prompts.txt', 'r') as f:
        for line in f:
            prompts.append(line.strip())
    
    return prompts

def clip_evaluation(images_base, images_lora):
    prompts = load_prompts()

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    lora_clip_score = calculate_clip_score(clip_score_fn, images_lora, prompts)
    base_clip_score = calculate_clip_score(clip_score_fn, images_base, prompts)
    print(f"LoRA CLIP score: {lora_clip_score}")
    print(f"Base CLIP score: {base_clip_score}")

    return images_lora, images_base


def main():
    images_base, images_lora = load_images()

    clip_evaluation(images_base, images_lora)


if __name__ == "__main__":
    main()