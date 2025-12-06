from diffusers import AutoPipelineForText2Image
from transformers import CLIPTokenizer
import safetensors.torch as st
import torch, argparse, platform, pdb, warnings, os
from datetime import datetime
torch.backends.mps.allow_truncated_normal_ = True
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


BLUE = "\033[94m"
LIGHT_BLUE = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"
lora_dir = "models/sdxl-lora"
lora_weights = "lora.safetensors"

INSTRUCTIONS = ("Photorealistic, high resolution image, 4k, detailed, ")

def get_num_tokens(str):
    """Return number of tokens in string. We use this to limit prompt length."""
    tokens = tokenizer.encode(str, add_special_tokens=True)
    count = len(tokens)
    # print(f"{count}")
    return count

def main(args):
    pipe = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=torch.float16, variant="fp16")

    if args.lora:
        try:
            pipe.unet.load_lora_adapter(lora_dir, weight_name=args.lora_name, adapter_name="default", prefix=None)
            pipe.unet.set_adapters("default", weights=[args.heat])
            print(f"{BLUE}Loaded LoRA from {lora_dir}/{args.lora_name}{RESET}")
            # print("active adapters:", pipe.unet.active_adapters)
        except Exception as e:
            print(f"{RED}Unable to load LoRA weights from {lora_dir}: {e}{RESET}")

    # use "cuda" on NVIDIA, "mps" on Mac, "cpu" otherwise. WLS is cuda.
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    elif torch.backends.mps.is_available():
        pipe = pipe.to("mps")
    else:
        pipe = pipe.to("cpu")

    print("unet device:", pipe.unet.device)
    print("sample param dtype:", next(pipe.unet.parameters()).dtype)

    # if there is a saftey checker, turn it off. We're not children here. 
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    print(f"{BLUE}SIDIG ready. Type 'exit' to quit.{RESET}")

    # interactive prompt loop
    while True:
        prompt = input(f"{LIGHT_BLUE}Enter prompt:{RESET} ")
        if prompt.strip().lower() in ["exit", "quit", "q"]:
            print("Exiting.")
            break
        elif prompt == "":
            continue
        elif get_num_tokens(prompt) + get_num_tokens(INSTRUCTIONS) + get_num_tokens(args.token) > 77:
            print(f"{RED}Prompt is too long. Use less tokens.{RESET}")
            continue
        
        # combine token/instructions to prompt for more control
        if not args.no_token:
            prompt = f"{args.token} {INSTRUCTIONS} {prompt}"
        else:
            prompt = f"{INSTRUCTIONS} {prompt}" 

        try:

            result = pipe(prompt, height=args.dimension, width=args.dimension, num_inference_steps=args.steps, guidance_scale=args.scale)
            image = result.images[0]

            image.show()
            if args.save:
                if not os.path.exists("output"):
                    os.makedirs("output")
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                img_name = f"sidig_{stamp}.png"
                image.save(f"output/{img_name}")
                print(f"Saved as {img_name}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpaceIsDirty Image Generator (SIDIG) prompt.")
    parser.add_argument("--lora", "-l", action="store_true", help="Use LoRA fine tuned SDXL model")
    parser.add_argument("--lora-name", "-n", type=str, default="lora.safetensors", 
                        help="LoRA weight filename in models/sdxl-lora")
    parser.add_argument("--model-id", "-m", type=str, default="stabilityai/sdxl-turbo", help="Pretrained model path")
    parser.add_argument("--no-token", "-nt", action="store_true", help="Do not use CLIP token prefix in prompts")
    parser.add_argument("--dimension", "-d", type=int, default=768, help="Image dimension (height and width)")
    parser.add_argument("--steps", "-s", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--scale", "-g", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--token", "-t", type=str, default="spaceisdirty", help="Custom token prefix for training")
    parser.add_argument("--heat", "-ht", type=float, default=1.0, help="LoRA heat scaling factor")
    parser.add_argument("--save", "-sv", action="store_true", help="Save generated image to output/ directory")
    args = parser.parse_args()
    
    print(f"Using dimensions: {args.dimension}x{args.dimension}")
    main(args)
