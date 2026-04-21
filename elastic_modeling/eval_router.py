import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from elastic_modeling.budget_loss import (
    build_parameter_count_components,
    compute_budget_loss,
    resolve_budget_accounting_mode,
)
from elastic_modeling.elastic_qwen import ElasticQwen2ForCausalLM
from elastic_modeling.gumbel_utils import (
    resolve_router_controls,
    sample_router_outputs_batch_shared,
)
from elastic_modeling.router import BudgetRouter
from helper_functions.helper_functions import collate_batch, tokenize_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved phase-1 router checkpoint across fixed budgets."
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
        default=str(REPO_ROOT / "checkpoints" / "router_phase1" / "router_step_000001.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=None,
        help="Override the router logit scale k used before Gumbel-Softmax.",
    )
    parser.add_argument(
        "--budget-values",
        type=float,
        nargs="+",
        default=None,
        help="Override budgets to evaluate. Defaults to checkpoint budgets.",
    )
    parser.add_argument(
        "--compare-base-full-budget",
        action="store_true",
        help="Also evaluate the reordered base model for reference.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional path to save evaluation metrics as CSV.",
    )
    return parser.parse_args()


def load_router_from_checkpoint(checkpoint_path, model_config):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    router_state = checkpoint["router_state_dict"]
    d_choices = [int(choice) for choice in checkpoint["d_choices"]]
    budget_values = checkpoint["budget_values"]
    enable_layer_skip = checkpoint.get("enable_layer_skip", False)
    budget_accounting_mode = checkpoint.get(
        "budget_accounting_mode",
        resolve_budget_accounting_mode(enable_layer_skip),
    )
    checkpoint_logit_scale = float(checkpoint.get("logit_scale_end", 1.0))

    if "router_d.0.weight" in router_state:
        first_router_weight = router_state["router_d.0.weight"]
    else:
        first_router_weight = router_state["router_d.0.0.weight"]
    hidden_dim = first_router_weight.shape[0]

    router = BudgetRouter(
        budget_values=budget_values,
        num_layers=model_config.num_hidden_layers,
        d_choices=d_choices,
        hidden_dim=hidden_dim,
    )
    router.load_state_dict(router_state)
    return (
        checkpoint,
        router,
        d_choices,
        budget_values,
        enable_layer_skip,
        budget_accounting_mode,
        checkpoint_logit_scale,
    )


def build_always_keep(batch_size, num_layers, device):
    return torch.ones((batch_size, num_layers), device=device, dtype=torch.bool)


@torch.no_grad()
def evaluate_fixed_budget(
    model,
    router,
    tokenized_ds,
    tokenizer,
    budget_idx,
    budget_value,
    d_choices,
    batch_size,
    enable_layer_skip,
    budget_accounting_mode,
    parameter_count_components,
    logit_scale,
):
    model.eval()
    router.eval()

    total_nll = 0.0
    total_pred_tokens = 0
    total_achieved_budget = 0.0
    total_keep_ratio = 0.0
    total_width_ratio = 0.0
    num_batches = 0

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

        budget_idx_tensor = torch.full(
            (current_batch_size,),
            fill_value=budget_idx,
            device=DEVICE,
            dtype=torch.long,
        )

        router_out = router(budget_idx_tensor, device=DEVICE)
        sampled_router_out = sample_router_outputs_batch_shared(
            router_out, tau=1.0, hard=True, logit_scale=logit_scale
        )
        resolved = resolve_router_controls(sampled_router_out, d_choices)
        if not enable_layer_skip:
            resolved["layer_keep"] = build_always_keep(
                batch_size=current_batch_size,
                num_layers=model.config.num_hidden_layers,
                device=DEVICE,
            )

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            d_keep=resolved["d_keep"],
            layer_keep=resolved["layer_keep"],
        )

        valid_tokens = (batch["labels"] != -100).sum().item()
        total_nll += outputs.loss.item() * valid_tokens
        total_pred_tokens += valid_tokens

        budget_stats = compute_budget_loss(
            config=model.config,
            target_budget=torch.full((current_batch_size,), budget_value, device=DEVICE),
            d_choices=d_choices,
            layer_keep=resolved["layer_keep"],
            d_keep=resolved["d_keep"],
            accounting_mode=budget_accounting_mode,
            parameter_count_components=parameter_count_components,
        )

        total_achieved_budget += budget_stats["achieved_budget"].mean().item()
        total_keep_ratio += resolved["layer_keep"].float().mean().item()
        total_width_ratio += (
            resolved["d_keep"].float() / float(model.config.intermediate_size)
        ).mean().item()
        num_batches += 1

    avg_loss = total_nll / total_pred_tokens
    perplexity = math.exp(avg_loss)
    return {
        "budget": budget_value,
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "num_examples": len(tokenized_ds),
        "num_tokens": total_pred_tokens,
        "achieved_budget": total_achieved_budget / max(1, num_batches),
        "keep_ratio": total_keep_ratio / max(1, num_batches),
        "width_ratio": total_width_ratio / max(1, num_batches),
        "budget_accounting_mode": budget_accounting_mode,
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


def write_metrics_csv(
    csv_path,
    elastic_metrics,
    checkpoint_path,
    checkpoint_step,
    d_choices,
    dataset_path,
    base_metrics=None,
):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_type",
        "checkpoint_path",
        "checkpoint_step",
        "dataset_path",
        "d_choices",
        "budget_accounting_mode",
        "target_budget",
        "achieved_budget",
        "keep_ratio",
        "width_ratio",
        "avg_loss",
        "perplexity",
        "num_examples",
        "num_tokens",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        d_choices_str = ",".join(str(choice) for choice in d_choices)
        for metrics in elastic_metrics:
            writer.writerow(
                {
                    "row_type": "elastic",
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step,
                    "dataset_path": str(dataset_path),
                    "d_choices": d_choices_str,
                    "budget_accounting_mode": metrics["budget_accounting_mode"],
                    "target_budget": metrics["budget"],
                    "achieved_budget": metrics["achieved_budget"],
                    "keep_ratio": metrics["keep_ratio"],
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
                    "d_choices": d_choices_str,
                    "budget_accounting_mode": elastic_metrics[0]["budget_accounting_mode"] if elastic_metrics else "",
                    "target_budget": "",
                    "achieved_budget": "",
                    "keep_ratio": "",
                    "width_ratio": "",
                    "avg_loss": base_metrics["avg_loss"],
                    "perplexity": base_metrics["perplexity"],
                    "num_examples": base_metrics["num_examples"],
                    "num_tokens": base_metrics["num_tokens"],
                }
            )


def main():
    args = parse_args()
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
    (
        checkpoint,
        router,
        d_choices,
        checkpoint_budget_values,
        enable_layer_skip,
        budget_accounting_mode,
        checkpoint_logit_scale,
    ) = load_router_from_checkpoint(
        args.checkpoint_path,
        teacher_model.config,
    )
    logit_scale = args.logit_scale if args.logit_scale is not None else checkpoint_logit_scale
    parameter_count_components = build_parameter_count_components(
        config=teacher_model.config,
        layer=teacher_model.model.layers[0],
        accounting_mode=budget_accounting_mode,
    )

    budget_values = args.budget_values if args.budget_values is not None else checkpoint_budget_values

    print(f"Using device: {DEVICE}")
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"d_choices: {d_choices}")
    print(f"budgets: {budget_values}")
    print(f"budget_accounting_mode: {budget_accounting_mode}")
    print(f"logit_scale(k): {logit_scale}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(args.dataset_path)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    tokenized_ds = tokenize_dataset(ds, tokenizer, args.max_length)
    print(f"Loaded {len(tokenized_ds)} tokenized evaluation examples")

    router = router.to(DEVICE)
    elastic_model = ElasticQwen2ForCausalLM(
        base_causallm=teacher_model,
        d_choices=d_choices,
        router=router,
    ).to(DEVICE)
    if checkpoint.get("elastic_model_state_dict") is not None:
        elastic_model.load_state_dict(checkpoint["elastic_model_state_dict"])

    print("\nElastic evaluation")
    print("-" * 88)
    print(
        f"{'target':>8} {'achieved':>9} {'keep':>8} {'width':>8} "
        f"{'loss':>10} {'ppl':>12} {'tokens':>12}"
    )

    elastic_metrics = []
    for budget_idx, budget_value in enumerate(budget_values):
        metrics = evaluate_fixed_budget(
            model=elastic_model,
            router=router,
            tokenized_ds=tokenized_ds,
            tokenizer=tokenizer,
            budget_idx=budget_idx,
            budget_value=budget_value,
            d_choices=d_choices,
            batch_size=args.batch_size,
            enable_layer_skip=enable_layer_skip,
            budget_accounting_mode=budget_accounting_mode,
            parameter_count_components=parameter_count_components,
            logit_scale=logit_scale,
        )
        elastic_metrics.append(metrics)
        print(
            f"{metrics['budget']:8.3f} {metrics['achieved_budget']:9.3f} "
            f"{metrics['keep_ratio']:8.3f} {metrics['width_ratio']:8.3f} "
            f"{metrics['avg_loss']:10.6f} {metrics['perplexity']:12.6f} "
            f"{metrics['num_tokens']:12d}"
        )

    base_metrics = None
    if args.compare_base_full_budget:
        base_metrics = evaluate_base_model(
            model=teacher_model,
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
            elastic_metrics=elastic_metrics,
            checkpoint_path=args.checkpoint_path,
            checkpoint_step=checkpoint.get("step", "unknown"),
            d_choices=d_choices,
            dataset_path=args.dataset_path,
            base_metrics=base_metrics,
        )
        print(f"\nSaved CSV summary to: {args.csv_path}")


if __name__ == "__main__":
    main()
