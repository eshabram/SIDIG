import torch
from diffusers import AutoPipelineForText2Image

model_id = "stabilityai/sdxl-turbo"
lora_dir = "models/sdxl-turbo-lora"
lora_weights = "lora.safetensors"
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "output/"

def get_unet_pred(pipe, prompt, seed=0, t_idx=50):
    pipe = pipe.to(device)
    torch.manual_seed(seed)

    latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)

    enc = pipe.encode_prompt(
        prompt=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    prompt_embeds = enc[0].to(device=device, dtype=torch.float16)
    pooled = enc[2].to(device=device, dtype=torch.float16)

    add_time_ids = torch.tensor([[512, 512, 512, 512, 0, 0]],
                                device=device,
                                dtype=torch.float16)

    timesteps = torch.tensor([t_idx], device=device, dtype=torch.long)

    with torch.no_grad():
        out = pipe.unet(
            latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample

    return out

def main():
    prompt = "spaceisdirty spaceship flying through a colorful nebula"

    base = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    lora = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )

    lora.unet.load_lora_adapter(
        lora_dir,
        weight_name=lora_weights,
        adapter_name="default",
        prefix=None,
    )

    base_pred = get_unet_pred(base, prompt, seed=0)
    lora_pred = get_unet_pred(lora, prompt, seed=0)

    diff = (lora_pred - base_pred).abs().mean().item()
    print("mean |Δ(pred)| =", diff)

def image_compare():

    prompt = "spaceisdirty a colorful sci-fi landscape"
    generator = torch.Generator("cuda").manual_seed(1234)

    # base pipe
    pipe_base = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
    out_base = pipe_base(prompt, height=512, width=512, num_inference_steps=4, generator=generator).images[0]

    # same seed again for LoRA
    generator = torch.Generator("cuda").manual_seed(1234)

    pipe_lora = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe_lora.unet.load_lora_adapter(lora_dir, weight_name=lora_weights, adapter_name="default", prefix=None)

    out_lora = pipe_lora(prompt, height=512, width=512, num_inference_steps=4, generator=generator).images[0]

    out_base.save(OUTPUT_DIR + "base.png")
    out_lora.save(OUTPUT_DIR + "lora.png")
    print("saved base.png and lora.png")


if __name__ == "__main__":
    main()
    image_compare()