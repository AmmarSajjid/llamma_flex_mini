'''
Running the Qwen Model on a small subset of the HuggingFace "fineweb-edu" dataset to get the running sum of the activations of the MLP layers
intermediate state, and using it to generate rankings.
'''
import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from helper_functions.helper_functions import tokenize_dataset, collate_batch
from config import MODEL_NAME, SUBSET_DIR, MODEL_PATH, TOKENIZER_PATH, MAX_LENGTH, BATCH_SIZE

SAVE_PATH = f"{PROJECT_ROOT}/mlp_activation_sums.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_COL = "text"


class MLPActivationSumCollector:
    """
    Accumulates absolute activation magnitudes for each MLP intermediate neuron
    in each decoder layer.

    For Qwen MLP:
        act = silu(gate_proj(x)) * up_proj(x)
    where x is the input to the MLP (i.e. output of post_attention_layernorm).
    """

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.num_layers = len(self.layers)
        self.intermediate_size = self.layers[0].mlp.gate_proj.out_features

        # shape: [num_layers, intermediate_size]
        self.activation_sums = torch.zeros(
            self.num_layers, self.intermediate_size, dtype=torch.float64
        )

        self.handles = []

    def _make_hook(self, layer_idx):
        layer = self.layers[layer_idx]

        @torch.no_grad()
        def hook(module, inputs, output):
            """
            output is the tensor after post_attention_layernorm,
            which is exactly what goes into the MLP.
            shape: [B, T, H]
            """
            x = output

            # Compute the intermediate gated MLP activation
            gate = layer.mlp.gate_proj(x)           # [B, T, D]
            up = layer.mlp.up_proj(x)               # [B, T, D]
            act = F.silu(gate) * up                 # [B, T, D]

            # Accumulate abs activation over batch and sequence dims
            # result shape: [D]
            layer_sum = act.abs().sum(dim=(0, 1)).detach().cpu().to(torch.float64)

            self.activation_sums[layer_idx] += layer_sum

        return hook

    def register(self):
        for layer_idx, layer in enumerate(self.layers):
            handle = layer.post_attention_layernorm.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def save(self, path):
        torch.save(
            {
                "activation_sums": self.activation_sums,   # [24, 4864]
                "num_layers": self.num_layers,
                "intermediate_size": self.intermediate_size,
                "model_name": MODEL_NAME,
            },
            path,
        )


@torch.no_grad()
def run_collection(model, tokenized_ds, tokenizer, collector):
    model.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    num_batches = (len(tokenized_ds) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(tokenized_ds), BATCH_SIZE), start=1):
        batch_examples = [
            tokenized_ds[i]
            for i in range(start, min(start + BATCH_SIZE, len(tokenized_ds)))
        ]
        batch = collate_batch(batch_examples, pad_token_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        _ = model(**batch)

        if batch_idx % 10 == 0 or batch_idx == num_batches:
            print(f"Processed batch {batch_idx}/{num_batches}")


def main():
    print(f"Using device: {DEVICE}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    ds = load_from_disk(SUBSET_DIR)
    text_col = 'text'
    tokenized_ds = tokenize_dataset(ds, tokenizer, MAX_LENGTH)

    print(f"Loaded {len(ds)} raw examples")
    print(f"Tokenized into {len(tokenized_ds)} usable examples")
    print("Model layers:", len(model.model.layers))
    print("MLP intermediate size:", model.model.layers[0].mlp.gate_proj.out_features)

    collector = MLPActivationSumCollector(model)
    collector.register()

    try:
        run_collection(model, tokenized_ds, tokenizer, collector)
    finally:
        collector.remove()

    collector.save(SAVE_PATH)
    print(f"Saved activation sums to: {SAVE_PATH}")
    print("activation_sums shape:", tuple(collector.activation_sums.shape))


if __name__ == "__main__":
    main()

