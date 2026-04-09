import math
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LENGTH = 512

def tokenize_dataset(ds, tokenizer, max_length=512):
    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)
    return tokenized


#Feature is just a batch before collating
def collate_batch(features, pad_token_id):
    max_len = max(len(f["input_ids"]) for f in features)

    input_ids = []
    attention_mask = []
    labels = []

    for f in features:
        ids = f["input_ids"]
        mask = [1] * len(ids)
        pad_len = max_len - len(ids)

        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append(mask + [0] * pad_len)
        labels.append(ids + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }