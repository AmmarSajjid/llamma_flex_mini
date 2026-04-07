import math
import os
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
Creating/Loading a debug subset of the HuggingFace "fineweb-edu" dataset for quick debugging during development.
'''
# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT = "train"          # change if needed
SUBSET_SIZE = 200                # number of documents for debug
SEED = 42
SUBSET_DIR = "./data/debug_fineweb_subset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def get_or_create_debug_subset():
    subset_path = Path(SUBSET_DIR)

    if subset_path.exists():
        print(f"Loading saved debug subset from: {subset_path}")
        return load_from_disk(str(subset_path))

    print("Creating new fixed debug subset...")

    # Stream instead of downloading the full dataset
    ds_stream = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    # Shuffle buffer for approximate randomization
    ds_stream = ds_stream.shuffle(seed=SEED, buffer_size=10_000)

    # Take only the small subset you need
    examples = list(ds_stream.take(SUBSET_SIZE))

    # Convert streamed examples into a regular in-memory Dataset
    ds = Dataset.from_list(examples)

    # Save only this subset
    ds.save_to_disk(str(subset_path))
    print(f"Saved debug subset to: {subset_path}")

    return ds


def main():
    print(f"Using device: {DEVICE}")
    ds = get_or_create_debug_subset()
    


if __name__ == "__main__":
    main()