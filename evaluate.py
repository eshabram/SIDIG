from diffusers import AutoPipelineForText2Image
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import numpy as np
import os

model_id = "stabilityai/sdxl-turbo"
lora_dir = "models/sdxl-turbo-lora"
lora_weights = "lora.safetensors"
GUIDANCE_SCALE = 1.0
DIMENSION = 512
INFER_STEPS = 4
HEAT = 1.0
OUTPUT_DIR = "output"

CLIP_TOKEN = "spaceisdirty"
INSTRUCTIONS = ("Photorealistic, high resolution image, 4k, detailed, ")

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

def create_prompts():
    prompts = [
        "astronaut driving a car on Pluto",
        "meteor",
        "a picture of the international space station above the Earth",
        "a rover on the surface of mars",
    ]

    prompts_with_lora_token = [
        f"{CLIP_TOKEN} {prompt}" for prompt in prompts
    ]

    return prompts, prompts_with_lora_token

def calculate_clip_score(clip_score_fn, images, prompts):
    images_int = (images * 255).astype("uint8")
    output_clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(output_clip_score), 4)

def save_images(images, prompts, prefix):
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img

        pil_img = Image.fromarray(img_uint8)

        safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:28])
        filename = f"{prefix}_{i}_{safe_name}.png"

        pil_img.save(os.path.join(OUTPUT_DIR, filename))

def clip_evaluation(pipe):
    prompts, prompts_with_lora_token = create_prompts()

    # generate images with the LoRA
    generator = torch.Generator("cuda").manual_seed(1234)
    images_lora = pipe(prompts_with_lora_token, height=DIMENSION, width=DIMENSION, num_inference_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE, num_images_per_prompt=1, output_type="np", generator=generator).images
    print("LoRA images generated")
    save_images(images_lora, prompts, "lora")

    # disable the LoRA
    pipe.unet.set_adapters([])
    # generate images with the base SD model
    generator = torch.Generator("cuda").manual_seed(1234)
    images_base = pipe(prompts, height=DIMENSION, width=DIMENSION, num_inference_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE, num_images_per_prompt=1, output_type="np", generator=generator).images
    print("Base Stable Diffusion images generated")
    save_images(images_base, prompts, "base")

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    lora_clip_score = calculate_clip_score(clip_score_fn, images_lora, prompts)
    base_clip_score = calculate_clip_score(clip_score_fn, images_base, prompts)
    print(f"LoRA CLIP score: {lora_clip_score}")
    print(f"Base CLIP score: {base_clip_score}")

    return images_lora


def main():
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    device = set_device_and_optimizations(pipe)
    print(device)

    pipe.unet.load_lora_adapter(lora_dir, weight_name=lora_weights, adapter_name="default", prefix=None)
    pipe.unet.set_adapters("default", weights=[HEAT])

    images_lora = clip_evaluation(pipe)


if __name__ == "__main__":
    main()