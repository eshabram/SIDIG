from diffusers import AutoPipelineForText2Image
import torch
from functools import partial
from PIL import Image
import numpy as np
import os
import itertools

model_id = "stabilityai/sdxl-turbo"
lora_dir = "models/sdxl-lora"
lora_weights = "lora.safetensors"
DIMENSION = 512
OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "evaluation-images")
NP_DIR = os.path.join(OUTPUT_DIR, "evaluation-np")

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
        "laser gun",
        "aliens chilling with some beers"
    ]


    prompts_with_lora_token = [
        f"{CLIP_TOKEN} {prompt}" for prompt in prompts
    ]
    
    prompts_with_instruction = [
        f"{CLIP_TOKEN} {INSTRUCTIONS} {prompt}" for prompt in prompts
    ]

    return prompts, prompts_with_lora_token, prompts_with_instruction

def save_images(images, prompts, prefix):
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img

        pil_img = Image.fromarray(img_uint8)

        safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:15])
        partial_filename = f"{i}_{safe_name}_{prefix}"
        filename = f"{partial_filename}.png"

        pil_img.save(os.path.join(IMAGES_DIR, filename))

def generate_images(pipe):
    prompts, prompts_with_lora_token, prompts_with_instruction = create_prompts()

    def get_num_tokens_with_pipe(pipe, text):
        return len(pipe.tokenizer(text, return_tensors=None).input_ids)

    for prompt in prompts_with_instruction:
        if (get_num_tokens_with_pipe(pipe, prompt) > 77):
            print("Prompt is too long:", prompt)
            return

    strength_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cfg_values = [1.0]
    steps_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]

    hyperparameter_combinations = itertools.product(
        strength_values,
        cfg_values,
        steps_values
    )

    for strength_value, cfg_value, steps_value in hyperparameter_combinations:
        # generate images with the LoRA
        generator = torch.Generator("cuda").manual_seed(1234)
        pipe.unet.set_adapters("default", weights=[strength_value])
        images_lora = pipe(prompts_with_instruction, height=DIMENSION, width=DIMENSION, num_inference_steps=steps_value, guidance_scale=cfg_value, num_images_per_prompt=1, output_type="np", generator=generator).images
        print("LoRA images generated")
        prefix = f"InstloraStrength{strength_value}CFG{cfg_value}Steps{steps_value}"
        save_images(images_lora, prompts, prefix)
        # images_lora_np = np.stack(images_lora)
        # np.save(os.path.join(NP_DIR, f"{prefix}.npy"), images_lora)
    

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(NP_DIR, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    device = set_device_and_optimizations(pipe)
    print(device)

    pipe.unet.load_lora_adapter(lora_dir, weight_name=lora_weights, adapter_name="default", prefix=None)

    generate_images(pipe)


if __name__ == "__main__":
    main()