from diffusers import AutoPipelineForText2Image
from transformers import CLIPTokenizer
import safetensors.torch as st
import torch, argparse, platform, pdb, warnings
torch.backends.mps.allow_truncated_normal_ = True
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


BLUE = "\033[94m"
LIGHT_BLUE = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"
model_id = "stabilityai/sdxl-turbo"
lora_dir = "models/sdxl-turbo-lora"
lora_weights = "lora.safetensors"
GUIDANCE_SCALE = 1.0
DIMENSION = 512
INFER_STEPS = 4

CLIP_TOKEN = ""
INSTRUCTIONS = CLIP_TOKEN + " " + (
    "Photorealistic, high resolution image, 4k, detailed, "
)

def get_num_tokens(str):
    tokens = tokenizer.encode(str, add_special_tokens=True)
    count = len(tokens)
    # print(f"{count}")
    return count

def main(args):
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")

    if args.lora:
        try:
            pipe.unet.load_lora_adapter(lora_dir, weight_name=lora_weights, adapter_name="default", prefix=None)
            print(f"{BLUE}Loaded LoRA from {lora_dir}/{lora_weights}{RESET}")
            # print("active adapters:", pipe.unet.active_adapters)
        except Exception as e:
            print(f"{RED}Unable to load LoRA weights from {lora_dir}: {e}{RESET}")

    # use "cuda" on NVIDIA, "mps" on Mac, "cpu" otherwise
    system = platform.system()
    if system == "Windows":
        pipe = pipe.to("cuda")
    elif system == "Darwin":
        pipe = pipe.to("mps")
    else:
        pipe = pipe.to("cpu")

    print("unet device:", pipe.unet.device)
    print("sample param dtype:", next(pipe.unet.parameters()).dtype)

    # if there is a saftey checker, turn it off. We're not children here. 
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    print(f"{BLUE}SIDIG ready. Type 'exit' to quit.{RESET}")

    while True:
        prompt = input(f"{LIGHT_BLUE}Enter prompt:{RESET} ")
        if prompt.strip().lower() in ["exit", "quit", "q"]:
            print("Exiting.")
            break
        elif prompt == "":
            continue
        elif get_num_tokens(prompt) + get_num_tokens(INSTRUCTIONS) > 77:
            print(f"{RED}Prompt is too long. Use less tokens.{RESET}")
            continue
        
        # combine token/instructions to prompt for more control
        prompt = f"{INSTRUCTIONS} {prompt}" 
        try:

            result = pipe(prompt, height=DIMENSION, width=DIMENSION, num_inference_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE)
            image = result.images[0]

            image.show()
            # image.save("output/image.png")
            # print("Saved as output.png\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpaceIsDirty Image Generator (SIDIG) prompt.")
    parser.add_argument("--lora", "-l", action="store_true", help="Use LoRA fine tuned SDXL-Turbo model")
    args = parser.parse_args()
    
    main(args)
