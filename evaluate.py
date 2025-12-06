from diffusers import AutoPipelineForText2Image
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import numpy as np
import os
import t2v_metrics
import re
from collections import defaultdict
from pprint import pprint
from math import ceil
PROGRESS_PRINT = True

OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "evaluation-images")
NP_DIR = os.path.join(OUTPUT_DIR, "evaluation-np")
BATCH_SIZE = 4

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
    num_batches = ceil(len(prompts) / BATCH_SIZE)
    weighted_sum = 0.0
    total_count = 0

    for i in range(num_batches):
        batch_prompts = prompts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_images = images_int[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        output_clip_score = clip_score_fn(torch.from_numpy(batch_images).permute(0, 3, 1, 2), batch_prompts).detach()

        batch_count = len(batch_prompts)
        weighted_sum += float(output_clip_score) * batch_count
        total_count += batch_count
        if PROGRESS_PRINT:
            print("CLIP score calculated up to image ", total_count)

    final_score = weighted_sum / total_count

    return round(float(final_score), 4)

def load_images():
    images_base = np.load(os.path.join(NP_DIR, "images_base.npy"))
    images_lora = np.load(os.path.join(NP_DIR, "images_lora.npy"))
    print("npy images loaded: ", images_lora.shape[0])
    return images_base, images_lora

def load_prompts():
    prompts = []
    with open('output/prompts.txt', 'r') as f:
        for line in f:
            prompts.append(line.strip())
    
    return prompts

def load_pngs():
    base_pattern = re.compile(r"^base_(\d+)_.*\.png$")
    lora_pattern = re.compile(r"^lora_(\d+)_.*\.png$")

    base_files = []
    lora_files = []

    for fname in os.listdir(IMAGES_DIR):
        if base_pattern.match(fname):
            base_files.append(fname)
        elif lora_pattern.match(fname):
            lora_files.append(fname)

    # Sort by numeric index i
    def extract_i(fname):
        return int(re.search(r"_(\d+)_", fname).group(1))

    base_files = sorted(base_files, key=extract_i)
    lora_files = sorted(lora_files, key=extract_i)

    # Convert to full paths
    png_base = [os.path.join(IMAGES_DIR, f) for f in base_files]
    png_lora = [os.path.join(IMAGES_DIR, f) for f in lora_files]
    print("pngs loaded: ", len(png_lora))

    return png_base, png_lora

# Used for calculating CLIP Scores of multiple configurations at once
# Used with multiple configurations generated from inference_settings_tuning.py
def load_multiconfig_pngs():
    pattern = re.compile(r"^(\d+)_.*_(.+)\.png$")
    prefix_dict = defaultdict(list)

    for fname in os.listdir(IMAGES_DIR):
        match = pattern.match(fname)
        if not match:
            continue

        index = int(match.group(1))
        prefix = match.group(2)

        full_path = os.path.join(IMAGES_DIR, fname)
        prefix_dict[prefix].append((index, full_path))

    # Sort each prefix group by image index
    for prefix in prefix_dict:
        prefix_dict[prefix].sort(key=lambda x: x[0])
        prefix_dict[prefix] = [p for _, p in prefix_dict[prefix]]

    return dict(prefix_dict)

def clip_evaluation(images_base, images_lora, prompts):
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    lora_clip_score = calculate_clip_score(clip_score_fn, images_lora, prompts)
    base_clip_score = calculate_clip_score(clip_score_fn, images_base, prompts)
    print(f"Base CLIP score: {base_clip_score}")
    print(f"LoRA CLIP score: {lora_clip_score}")

    return lora_clip_score, base_clip_score

def average_vqa_score(scores):
    if len(scores) == 0:
        return 0

    total = 0.0
    for t in scores:
        total += t.squeeze().item()

    return total / len(scores)

def vqa_evaluation(images_base, images_lora, prompts):
    clip_flant5_score = t2v_metrics.VQAScore(model="clip-flant5-xl")

    scores_base = []
    for i, (image, prompt) in enumerate(zip(images_base, prompts)):
        if PROGRESS_PRINT:
            print("Calculating VQA for image path: ", image, "prompt: ", prompt)
        score = clip_flant5_score(images=[image], texts=[prompt])
        scores_base.append(score)

    scores_lora = []
    for i, (image, prompt) in enumerate(zip(images_lora, prompts)):
        if PROGRESS_PRINT:
            print("Calculating VQA for image path: ", image, "prompt: ", prompt)
        score = clip_flant5_score(images=[image], texts=[prompt])
        scores_lora.append(score)

    print(prompts)

    average_base = average_vqa_score(scores_base)
    average_lora = average_vqa_score(scores_lora)

    print("VQA (base) average: ", average_base)
    print("VQA (lora) average: ", average_lora)

    print("VQA (base): ", scores_base)
    print("VQA (lora): ", scores_lora)

    return average_base, average_lora

def vqa_lora(images_lora, prompts):
    clip_flant5_score = t2v_metrics.VQAScore(model="clip-flant5-xl")

    prefix_scores = {}
    for prefix, image_paths in images_lora.items():
        scores = []
        for (image, prompt) in zip(image_paths, prompts):
            score = clip_flant5_score(images=[image], texts=[prompt])
            scores.append(score)
        average_score = average_vqa_score(scores)
        print(f"{prefix} score: ", average_score)
        prefix_scores[prefix] = average_score

    sorted_results = sorted(
        prefix_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_k = 0
    print("\n=== Ranked Results (Highest → Lowest) ===")
    if top_k == 0:
        for prefix, score in sorted_results:
            print(f"{prefix}: {score}")
    else:
        for prefix, score in sorted_results[:top_k]:
            print(f"{prefix}: {score}")

def main():
    images_base, images_lora = load_images()
    prompts = load_prompts()

    lora_clip_score, base_clip_score = clip_evaluation(images_base, images_lora, prompts)

    png_base, png_lora = load_pngs()
    scores_base, scores_lora = vqa_evaluation(png_base, png_lora, prompts)

    print("=================Summary=================")
    print(f"Base CLIP score: {base_clip_score}")
    print(f"LoRA CLIP score: {lora_clip_score}")
    print("VQA (base): ", scores_base)
    print("VQA (lora): ", scores_lora)

    # The following function calls can be used to compare different
    # configurations of inference settings
    # png_configs = load_multiconfig_pngs()
    # vqa_lora(png_configs, prompts)

if __name__ == "__main__":
    main()