
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_PATH = "./models/qwen_model"
TOKENIZER_PATH = "./models/qwen_tokenizer"


# DATASET_PATH
SUBSET_DIR = f"{PROJECT_ROOT}/data/debug_fineweb_subset"
MICRO_SUBSET_DIR = f"{PROJECT_ROOT}/data/micro_fineweb_subset"

# HYPER PARAMETERS
MAX_LENGTH = 512
BATCH_SIZE = 4