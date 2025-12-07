from diffusers import AutoPipelineForText2Image
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import numpy as np
import os
from math import ceil

model_id = "stabilityai/sdxl-turbo"
lora_dir = "models/sdxl-lora"
lora_weights = "lora.safetensors"
GUIDANCE_SCALE = 1.0
DIMENSION = 512
INFER_STEPS = 4
STRENGTH = 0.3
BATCH_SIZE = 4
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
        "astronaut walking through a neon-lit alien marketplace at night",
        "cyberpunk city skyline during a thunderstorm",
        "spaceship landing on a frozen ocean planet",
        "robot bartender serving drinks in a futuristic nightclub",
        "massive ringworld casting a shadow over a planet",
        "android portrait with glowing circuit skin",
        "alien desert with twin suns at sunset",
        "space station orbiting a gas giant with glowing storms",
        "cybernetic soldier standing in battlefield ruins",
        "hover cars flying between skyscrapers at dusk",
        "planet with floating mountains and waterfalls",
        "deep space mining operation inside an asteroid",
        "astronaut repairing a satellite during a solar flare",
        "giant mech standing in a rainy futuristic city",
        "abandoned spaceship drifting near a black hole",
        "alien jungle with bioluminescent plants at night",
        "robot child looking at the stars from a rooftop",
        "futuristic hospital with transparent holographic displays",
        "space elevator stretching into orbit at sunrise",
        "cyborg samurai in a neon alleyway",
        "terraforming machines reshaping a red planet",
        "underwater alien city with glowing architecture",
        "android researcher studying a mysterious artifact",
        "orbital shipyard constructing a massive battleship",
        "futuristic train speeding through a megacity",
        "alien queen sitting on a crystalline throne",
        "quantum computer core inside a sci-fi laboratory",
        "robot farming crops on a terraformed moon",
        "interstellar trading hub filled with alien species",
        "astronaut exploring a crashed alien spacecraft",
        "cyberpunk street market in the rain at night",
        "planet covered entirely by futuristic city structures",
        "giant alien creature walking through foggy ruins",
        "space marine squad moving through a derelict station",
        "futuristic classroom with holographic teachers",
        "starship bridge during a hyperspace jump",
        "android face reflected in a rain-soaked window",
        "neon-lit sci-fi diner on a distant colony world",
        "alien ocean with glowing jellyfish at twilight",
        "robot police patrolling a high-tech city",
        "space probe approaching a pulsar",
        "cyberpunk hacker surrounded by holographic screens",
        "futuristic laboratory growing artificial organs",
        "alien library carved into a cliff of crystal",
        "spaceship graveyard orbiting a dead planet",
        "android artist painting with projected light",
        "orbital defense cannons firing into space",
        "cybernetic dragon flying above a megacity",
        "star cruiser traveling through a colorful nebula",
        "robot caregiver helping an elderly human",
        "alien ruins buried beneath icy glaciers",
        "futuristic sports arena with hovering athletes",
        "android detective investigating a crime scene",
        "shuttle landing on a jungle-covered exoplanet",
        "massive generation ship traveling between stars",
        "cyberpunk alley filled with steam and neon signs",
        "alien scientist inside a floating laboratory",
        "robot construction crew building a space colony",
        "planet cracked open revealing glowing core",
        "cyborg mercenary in power armor",
        "futuristic kitchen with robotic chefs",
        "asteroid base with rotating habitat",
        "alien city carved into a volcanic mountain",
        "starfighter dogfight above a ringed planet",
        "android nurse assisting in an emergency bay",
        "cyberpunk rooftop chase with holographic ads",
        "alien forest covered in glowing spores",
        "deep space research vessel near a wormhole",
        "robot explorer mapping an unknown planet",
        "planet with diamond crystal oceans",
        "futuristic prison on an asteroid",
        "android monk meditating in a data temple",
        "alien marketplace on a floating platform",
        "interstellar highway filled with warp ships",
        "cybernetic surgeon performing an operation",
        "abandoned terraforming facility on a toxic world",
        "robot archaeologist uncovering ancient tech",
        "orbital resort overlooking a blue gas giant",
        "alien waterfall made of liquid light",
        "android soldier standing guard in a starport",
        "futuristic police interrogation room",
        "space colony inside a hollowed asteroid",
        "cyberpunk subway station at rush hour",
        "alien cathedral made of living crystal",
        "robot street performer entertaining humans",
        "giant star rising behind a distant planet",
        "android librarian organizing digital knowledge",
        "battle above a city between starships",
        "alien farm using gravity-defying crops",
        "cyberpunk hospital emergency room at night",
        "robot pet following a child on a colony world",
        "futuristic laboratory with zero-gravity experiments",
        "alien coastline with purple glowing waves",
        "android diplomat meeting an alien council",
        "space junkyard orbiting a dying star",
        "cybernetic athlete training in a high-tech gym",
        "planet with constant lightning storms",
        "robot firefighter battling plasma flames",
        "alien zoo with transparent containment fields",
        "orbital sniper station watching a planet below",
        "android teacher instructing human students",
        "cyberpunk taxi flying through traffic",
        "futuristic refugee camp on a distant moon",
        "alien storm covered in glowing energy clouds",
        "robot explorer descending into an alien cave",
        "starship emerging from hyperspace above a city",
        "cybernetic bounty hunter in a dusty outpost",
        "alien desert with floating rock formations",
        "android mechanic repairing a fusion engine",
        "futuristic courtroom with holographic evidence",
        "robot gardener tending bioluminescent plants",
        "alien ice world with glowing fractal patterns",
        "orbital observatory watching a supernova",
        "cyberpunk corporate lobby at night",
        "android musician performing with light instruments",
        "futuristic marketplace inside a space station",
        "alien storm front rolling across metal plains",
        "robot medic treating injured soldiers",
        "starship racing through an asteroid belt",
        "cybernetic street samurai in rain-soaked street",
        "alien coral reef glowing in deep space ocean",
        "android pilot inside a starfighter cockpit",
        "futuristic weapons lab with plasma rifles",
        "robot courier delivering packages on a colony",
        "gas giant looming over a ringed moon city",
        "cyberpunk repair shop filled with spare limbs",
        "alien volcano erupting with blue plasma",
        "android detective analyzing holographic evidence",
        "futuristic high-speed tunnel transport",
        "robot miner drilling into glowing crystal rock",
        "alien monument taller than surrounding mountains",
        "orbital battle between massive capital ships",
        "cyberpunk rooftop garden under neon sky",
        "android emergency dispatcher in control room",
        "alien tundra with glowing snow patterns",
        "futuristic smuggler hideout on a space station",
        "robot sculptor carving metal statues",
        "starship fleet assembling above a planet",
        "cybernetic street racer with glowing motorcycle",
        "alien biotech lab growing living machines",
        "android archivist preserving ancient data",
        "futuristic energy plant inside a volcano",
        "robot pilot ejecting from damaged starfighter",
        "alien megastructure surrounding a star",
        "cyberpunk hacker apartment filled with screens",
        "android peacekeeper standing between factions",
        "futuristic underwater research base",
        "robot janitor cleaning a high-tech corridor",
        "alien sky filled with colorful plasma auroras",
        "orbital drone swarm patrolling a colony",
        "cyberpunk corporate assassination scene",
        "android gardener cultivating synthetic trees",
        "futuristic black market on a space station",
        "robot defeat scene in a ruined city",
        "alien ruins emerging from shifting sands",
        "cybernetic gladiator arena fight",
        "android scientist observing a sentient star",
        "futuristic control room during system failure",
        "robot explorer watching twin suns set"
    ]

    prompts_with_instruction = [
        f"{INSTRUCTIONS} {prompt}" for prompt in prompts
    ]

    with open('output/prompts.txt', 'w') as f:
        for item in prompts_with_instruction:
            f.write(item + '\n')

    prompts_with_lora_token = [
        f"{CLIP_TOKEN} {INSTRUCTIONS} {prompt}" for prompt in prompts
    ]

    return prompts_with_instruction, prompts_with_lora_token, prompts

def save_images(images, prompts, prefix):
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image

        pil_img = Image.fromarray(image_uint8)

        safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:28])
        partial_filename = f"{prefix}_{i}_{safe_name}"
        filename = f"{partial_filename}.png"

        pil_img.save(os.path.join(IMAGES_DIR, filename))

def generate_images(pipe):
    base_prompts, prompts_with_lora_token, prompts = create_prompts()

    # generate images with the LoRA
    generator = torch.Generator("cuda").manual_seed(1234)
    num_batches = ceil(len(prompts_with_lora_token) / BATCH_SIZE)
    images_lora = []
    for i in range(num_batches):
        batch_prompts = prompts_with_lora_token[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        with torch.inference_mode():
            images_lora_portion = pipe(batch_prompts, height=DIMENSION, width=DIMENSION, num_inference_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE, num_images_per_prompt=1, output_type="np", generator=generator).images
        images_lora.extend(images_lora_portion)

    print("LoRA images generated")
    save_images(images_lora, prompts, "lora")
    # save np arrays of images
    images_lora_np = np.stack(images_lora)
    np.save(os.path.join(NP_DIR, "images_lora.npy"), images_lora_np)

    # disable the LoRA
    pipe.unet.set_adapters([])
    # generate images with the base SD model
    generator = torch.Generator("cuda").manual_seed(1234)
    images_base = []
    for i in range(num_batches):
        batch_prompts = base_prompts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        with torch.inference_mode():
            images_base_portion = pipe(batch_prompts, height=DIMENSION, width=DIMENSION, num_inference_steps=INFER_STEPS, guidance_scale=GUIDANCE_SCALE, num_images_per_prompt=1, output_type="np", generator=generator).images
        images_base.extend(images_base_portion)

    print("Base Stable Diffusion images generated")
    save_images(images_base, prompts, "base")
    # save np arrays of images
    images_base_np = np.stack(images_base)
    np.save(os.path.join(NP_DIR, "images_base.npy"), images_base_np)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(NP_DIR, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    device = set_device_and_optimizations(pipe)
    print(device)

    pipe.unet.load_lora_adapter(lora_dir, weight_name=lora_weights, adapter_name="default", prefix=None)
    pipe.unet.set_adapters("default", weights=[STRENGTH])

    generate_images(pipe)


if __name__ == "__main__":
    main()