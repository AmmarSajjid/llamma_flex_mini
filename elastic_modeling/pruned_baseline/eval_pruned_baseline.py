import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from elastic_modeling.budget_loss import BLOCK_PARAMS_ACCOUNTING, build_parameter_count_components
from elastic_modeling.elastic_qwen import ElasticQwen2ForCausalLM
from elastic_modeling.pruned_baseline.pruned_baseline_utils import build_fixed_width_controls
from helper_functions.helper_functions import collate_batch, tokenize_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained fixed-width pruned MLP baseline checkpoint."
    )
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "models" / "qwen_model_reordered_mlp_100k"),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=str(REPO_ROOT / "models" / "qwen_tokenizer"),
    )
    parser.add_argument(
        "--dataset-path",
        default=str(REPO_ROOT / "data" / "micro_fineweb_subset"),
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--compare-base-full-budget",
        action="store_true",
        help="Also evaluate the reordered dense base model for reference.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional path to save evaluation metrics as CSV.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "resolved_d_keep" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not look like a pruned baseline checkpoint"
        )
    return checkpoint


@torch.no_grad()
def evaluate_fixed_model(model, tokenized_ds, tokenizer, batch_size, d_keep, target_budget, achieved_budget):
    model.eval()

    total_nll = 0.0
    total_pred_tokens = 0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for start in range(0, len(tokenized_ds), batch_size):
        batch_examples = [
            tokenized_ds[i]
            for i in range(start, min(start + batch_size, len(tokenized_ds)))
        ]
        batch = collate_batch(batch_examples, pad_token_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        current_batch_size = batch["input_ids"].shape[0]

        layer_keep, d_keep_tensor = build_fixed_width_controls(
            batch_size=current_batch_size,
            num_layers=model.config.num_hidden_layers,
            d_keep=d_keep,
            device=DEVICE,
        )
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            d_keep=d_keep_tensor,
            layer_keep=layer_keep,
        )

        valid_tokens = (batch["labels"] != -100).sum().item()
        total_nll += outputs.loss.item() * valid_tokens
        total_pred_tokens += valid_tokens

    avg_loss = total_nll / total_pred_tokens
    return {
        "target_budget": target_budget,
        "achieved_budget": achieved_budget,
        "d_keep": d_keep,
        "width_ratio": d_keep / float(model.config.intermediate_size),
        "avg_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "num_examples": len(tokenized_ds),
        "num_tokens": total_pred_tokens,
        "budget_accounting_mode": BLOCK_PARAMS_ACCOUNTING,
    }


@torch.no_grad()
def evaluate_base_model(model, tokenized_ds, tokenizer, batch_size):
    model.eval()

    total_nll = 0.0
    total_pred_tokens = 0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for start in range(0, len(tokenized_ds), batch_size):
        batch_examples = [
            tokenized_ds[i]
            for i in range(start, min(start + batch_size, len(tokenized_ds)))
        ]
        batch = collate_batch(batch_examples, pad_token_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        valid_tokens = (batch["labels"] != -100).sum().item()
        total_nll += outputs.loss.item() * valid_tokens
        total_pred_tokens += valid_tokens

    avg_loss = total_nll / total_pred_tokens
    return {
        "avg_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "num_examples": len(tokenized_ds),
        "num_tokens": total_pred_tokens,
    }


def write_metrics_csv(csv_path, metrics, checkpoint_path, checkpoint_step, dataset_path, base_metrics=None):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "checkpoint_path",
        "checkpoint_step",
        "dataset_path",
        "budget_accounting_mode",
        "target_budget",
        "achieved_budget",
        "d_keep",
        "width_ratio",
        "avg_loss",
        "perplexity",
        "num_examples",
        "num_tokens",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "row_type": "pruned_baseline",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": checkpoint_step,
                "dataset_path": str(dataset_path),
                "budget_accounting_mode": metrics["budget_accounting_mode"],
                "target_budget": metrics["target_budget"],
                "achieved_budget": metrics["achieved_budget"],
                "d_keep": metrics["d_keep"],
                "width_ratio": metrics["width_ratio"],
                "avg_loss": metrics["avg_loss"],
                "perplexity": metrics["perplexity"],
                "num_examples": metrics["num_examples"],
                "num_tokens": metrics["num_tokens"],
            }
        )

        if base_metrics is not None:
            writer.writerow(
                {
                    "row_type": "base",
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step,
                    "dataset_path": str(dataset_path),
                    "budget_accounting_mode": BLOCK_PARAMS_ACCOUNTING,
                    "target_budget": "",
                    "achieved_budget": "",
                    "d_keep": "",
                    "width_ratio": "",
                    "avg_loss": base_metrics["avg_loss"],
                    "perplexity": base_metrics["perplexity"],
                    "num_examples": base_metrics["num_examples"],
                    "num_tokens": base_metrics["num_tokens"],
                }
            )


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint_path)

    pruned_base_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(args.dataset_path)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    tokenized_ds = tokenize_dataset(ds, tokenizer, args.max_length)

    parameter_count_components = checkpoint.get("parameter_count_components")
    if parameter_count_components is None:
        parameter_count_components = build_parameter_count_components(
            config=pruned_base_model.config,
            layer=pruned_base_model.model.layers[0],
            accounting_mode=BLOCK_PARAMS_ACCOUNTING,
        )

    model = ElasticQwen2ForCausalLM(
        base_causallm=pruned_base_model,
        d_choices=[int(checkpoint["resolved_d_keep"])],
        router=None,
        budget_values=None,
        enable_policy_modulation=False,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["elastic_model_state_dict"])

    print(f"Using device: {DEVICE}")
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Target budget: {float(checkpoint['target_budget']):.3f}")
    print(f"Resolved d_keep: {int(checkpoint['resolved_d_keep'])}")
    print(f"Width ratio: {float(checkpoint['resolved_width_ratio']):.6f}")
    print(f"Achieved budget: {float(checkpoint['achieved_budget']):.6f}")
    print(f"Budget accounting mode: {BLOCK_PARAMS_ACCOUNTING}")
    print(f"Loaded {len(tokenized_ds)} tokenized evaluation examples")

    metrics = evaluate_fixed_model(
        model=model,
        tokenized_ds=tokenized_ds,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        d_keep=int(checkpoint["resolved_d_keep"]),
        target_budget=float(checkpoint["target_budget"]),
        achieved_budget=float(checkpoint["achieved_budget"]),
    )

    print("\nPruned baseline evaluation")
    print("-" * 88)
    print(
        f"{'target':>8} {'achieved':>9} {'d_keep':>8} {'width':>8} "
        f"{'loss':>10} {'ppl':>12} {'tokens':>12}"
    )
    print(
        f"{metrics['target_budget']:8.3f} {metrics['achieved_budget']:9.3f} "
        f"{metrics['d_keep']:8d} {metrics['width_ratio']:8.3f} "
        f"{metrics['avg_loss']:10.6f} {metrics['perplexity']:12.6f} "
        f"{metrics['num_tokens']:12d}"
    )

    base_metrics = None
    if args.compare_base_full_budget:
        reference_base_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
        base_metrics = evaluate_base_model(
            model=reference_base_model,
            tokenized_ds=tokenized_ds,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )
        print("\nBase reordered model")
        print("-" * 40)
        print(f"Avg loss:   {base_metrics['avg_loss']:.6f}")
        print(f"Perplexity: {base_metrics['perplexity']:.6f}")
        print(f"Tokens:     {base_metrics['num_tokens']}")

    if args.csv_path is not None:
        write_metrics_csv(
            csv_path=args.csv_path,
            metrics=metrics,
            checkpoint_path=args.checkpoint_path,
            checkpoint_step=checkpoint.get("step", "unknown"),
            dataset_path=args.dataset_path,
            base_metrics=base_metrics,
        )
        print(f"\nSaved CSV summary to: {args.csv_path}")


if __name__ == "__main__":
    main()
