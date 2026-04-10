import math
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions.helper_functions import tokenize_dataset, collate_batch

from config import SUBSET_DIR, MAX_LENGTH, BATCH_SIZE, MODEL_PATH, TOKENIZER_PATH


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = SUBSET_DIR


def tokenize_dataset(ds, tokenizer, max_length=MAX_LENGTH):
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
# def collate_batch(features, pad_token_id):
#     max_len = max(len(f["input_ids"]) for f in features)

#     input_ids = []
#     attention_mask = []
#     labels = []

#     for f in features:
#         ids = f["input_ids"]
#         mask = [1] * len(ids)
#         pad_len = max_len - len(ids)

#         input_ids.append(ids + [pad_token_id] * pad_len)
#         attention_mask.append(mask + [0] * pad_len)
#         labels.append(ids + [-100] * pad_len)

#     return {
#         "input_ids": torch.tensor(input_ids, dtype=torch.long),
#         "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#         "labels": torch.tensor(labels, dtype=torch.long),
#     }


@torch.no_grad()
def evaluate(model, tokenized_ds, tokenizer):
    model.eval()

    #negative log likelihood nll
    total_nll = 0.0
    total_pred_tokens = 0

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    b = 0

    for start in range(0, len(tokenized_ds), BATCH_SIZE):
        batch_examples = [
            tokenized_ds[i]
            for i in range(start, min(start + BATCH_SIZE, len(tokenized_ds)))
        ]
        batch = collate_batch(batch_examples, pad_token_id)
        #sending the batch to device
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        # Count valid label positions (-100 are ignored)
        valid_tokens = (batch["labels"] != -100).sum().item()

        total_nll += loss.item() * valid_tokens
        total_pred_tokens += valid_tokens

        # print(f"Batch {b} completed")
        b += 1

    avg_loss = total_nll / total_pred_tokens
    perplexity = math.exp(avg_loss)

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "num_examples": len(tokenized_ds),
        "num_tokens": total_pred_tokens,
    }


def main():
    print(f"Using device: {DEVICE}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    ds = load_from_disk(SUBSET_DIR)
    tokenized_ds = tokenize_dataset(ds, tokenizer, MAX_LENGTH)

    metrics = evaluate(model, tokenized_ds, tokenizer)

    print("\nEvaluation results")
    print("-" * 40)
    print(f"Examples:     {metrics['num_examples']}")
    print(f"Tokens:       {metrics['num_tokens']}")
    print(f"Avg loss:     {metrics['avg_loss']:.6f}")
    print(f"Perplexity:   {metrics['perplexity']:.6f}")


if __name__ == "__main__":
    main()