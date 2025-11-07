import torch, transformers, huggingface
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune script for the SpaceIsDirty Image Generator (SIDIG)")
    parser.add_argument("--images-dir", type=Path, default=Path("images"), help="Directory containing PNG assets.")
    parser.add_argument("--meta-dir", type=Path, default=Path("meta"), help="Directory containing labels.")
    parser.add_argument("--model", type=Path, default=Path("models/sdxl-turbo"), help="Provide a path to the model to fune-tune.")

    print("hello world")