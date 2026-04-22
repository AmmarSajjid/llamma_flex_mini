import argparse
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from elastic_modeling.budget_loss import BLOCK_PARAMS_ACCOUNTING
from elastic_modeling.elastic_qwen import ElasticQwen2ForCausalLM
from elastic_modeling.pruned_baseline.pruned_baseline_utils import (
    build_fixed_width_controls,
    checkpoint_path_for_step,
    resolve_pruned_width_for_budget,
)
from helper_functions.helper_functions import collate_batch, tokenize_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a fixed-width pruned MLP baseline at a target block-parameter budget."
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
        "--save-dir",
        default=str(REPO_ROOT / "checkpoints" / "pruned_baseline"),
    )
    parser.add_argument("--target-budget", type=float, required=True)
    parser.add_argument(
        "--width-granularity",
        type=int,
        default=1,
        help="Round the resolved d_keep to this granularity. Use 1 for exact budget matching.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--fail-on-nan", action="store_true")
    parser.add_argument("--save-failure-state", action="store_true")
    parser.add_argument("--skip-non-finite-steps", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle_tokenized_examples(tokenized_ds):
    while True:
        for idx in range(len(tokenized_ds)):
            yield tokenized_ds[idx]


def create_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def first_non_finite_name(named_tensors):
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            return name
    return None


def save_failure_state(step, args, model, optimizer, fixed_width_metadata, bad_name, context):
    if not args.save_failure_state:
        return

    failure_dir = Path(args.save_dir) / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_path = failure_dir / f"failure_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "context": context,
            "bad_tensor": bad_name,
            "elastic_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "target_budget": fixed_width_metadata["target_budget"],
            "resolved_d_keep": fixed_width_metadata["resolved_d_keep"],
            "resolved_width_ratio": fixed_width_metadata["resolved_width_ratio"],
            "achieved_budget": fixed_width_metadata["achieved_budget"],
            "width_granularity": fixed_width_metadata["width_granularity"],
            "budget_accounting_mode": BLOCK_PARAMS_ACCOUNTING,
            "parameter_count_components": dict(fixed_width_metadata["parameter_count_components"]),
        },
        failure_path,
    )


def raise_or_checkpoint_non_finite(
    *,
    step,
    args,
    model,
    optimizer,
    fixed_width_metadata,
    named_tensors,
    context,
    raise_if_requested=True,
):
    bad_name = first_non_finite_name(named_tensors)
    if bad_name is None:
        return

    save_failure_state(step, args, model, optimizer, fixed_width_metadata, bad_name, context)
    if args.fail_on_nan and raise_if_requested:
        raise RuntimeError(f"Non-finite tensor detected at step {step} during {context}: {bad_name}")


def sanitize_or_skip_non_finite_grads(step, args, model, optimizer, fixed_width_metadata):
    bad_name = None
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad_name = name
            break

    if bad_name is None:
        return False, None

    save_failure_state(
        step,
        args,
        model,
        optimizer,
        fixed_width_metadata,
        f"grad::{bad_name}",
        "backward",
    )

    if not args.skip_non_finite_steps:
        return True, bad_name

    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    return True, bad_name


def save_checkpoint(save_dir, step, model, optimizer, args, fixed_width_metadata):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = checkpoint_path_for_step(save_dir, step)
    torch.save(
        {
            "step": step,
            "elastic_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "target_budget": fixed_width_metadata["target_budget"],
            "resolved_d_keep": fixed_width_metadata["resolved_d_keep"],
            "requested_d_keep": fixed_width_metadata["requested_d_keep"],
            "resolved_width_ratio": fixed_width_metadata["resolved_width_ratio"],
            "requested_width_ratio": fixed_width_metadata["requested_width_ratio"],
            "achieved_budget": fixed_width_metadata["achieved_budget"],
            "width_granularity": fixed_width_metadata["width_granularity"],
            "budget_accounting_mode": BLOCK_PARAMS_ACCOUNTING,
            "parameter_count_components": dict(fixed_width_metadata["parameter_count_components"]),
            "gradient_checkpointing": args.gradient_checkpointing,
            "use_bf16": args.use_bf16,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "loss_objective": "fixed_pruned_model_lm_loss",
        },
        checkpoint_path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Using device: {DEVICE}")

    base_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(args.dataset_path)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    tokenized_ds = tokenize_dataset(ds, tokenizer, args.max_length)
    print(f"Loaded {len(tokenized_ds)} tokenized examples")

    fixed_width_metadata = resolve_pruned_width_for_budget(
        config=base_model.config,
        layer=base_model.model.layers[0],
        target_budget=args.target_budget,
        width_granularity=args.width_granularity,
    )
    print(f"Target budget: {fixed_width_metadata['target_budget']:.3f}")
    print(f"Resolved d_keep: {fixed_width_metadata['resolved_d_keep']}")
    print(f"Width ratio: {fixed_width_metadata['resolved_width_ratio']:.6f}")
    print(f"Achieved budget: {fixed_width_metadata['achieved_budget']:.6f}")
    print(f"Width granularity: {fixed_width_metadata['width_granularity']}")

    model = ElasticQwen2ForCausalLM(
        base_causallm=base_model,
        d_choices=[fixed_width_metadata["resolved_d_keep"]],
        router=None,
        budget_values=None,
        enable_policy_modulation=False,
    ).to(DEVICE)
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-6)
    scheduler = create_scheduler(optimizer, args.steps, args.warmup_ratio)

    example_stream = cycle_tokenized_examples(tokenized_ds)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    effective_batch_size = args.batch_size * args.grad_accum_steps

    for step in range(1, args.steps + 1):
        batch_examples = [next(example_stream) for _ in range(args.batch_size)]
        batch = collate_batch(batch_examples, tokenizer.pad_token_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        layer_keep, d_keep = build_fixed_width_controls(
            batch_size=args.batch_size,
            num_layers=model.config.num_hidden_layers,
            d_keep=fixed_width_metadata["resolved_d_keep"],
            device=DEVICE,
        )

        train_autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.use_bf16 and DEVICE == "cuda"
            else nullcontext()
        )
        with train_autocast:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                d_keep=d_keep,
                layer_keep=layer_keep,
            )
            loss = outputs.loss

        raise_or_checkpoint_non_finite(
            step=step,
            args=args,
            model=model,
            optimizer=optimizer,
            fixed_width_metadata=fixed_width_metadata,
            named_tensors=[
                ("outputs.logits", outputs.logits),
                ("loss", loss),
            ],
            context="fixed-width forward",
        )

        (loss / args.grad_accum_steps).backward()

        should_step = step % args.grad_accum_steps == 0 or step == args.steps
        if should_step:
            skipped_step, bad_grad_name = sanitize_or_skip_non_finite_grads(
                step=step,
                args=args,
                model=model,
                optimizer=optimizer,
                fixed_width_metadata=fixed_width_metadata,
            )
            if skipped_step:
                print(
                    f"warning: skipped optimizer step at step {step} due to non-finite gradient in {bad_grad_name}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())

        if step % args.log_every == 0 or step == 1:
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            print(
                f"step={step:04d} "
                f"loss={loss.item():.4f} "
                f"target={fixed_width_metadata['target_budget']:.3f} "
                f"achieved={fixed_width_metadata['achieved_budget']:.3f} "
                f"d_keep={fixed_width_metadata['resolved_d_keep']} "
                f"width={fixed_width_metadata['resolved_width_ratio']:.3f} "
                f"eff_batch={effective_batch_size} "
                f"accum={step % args.grad_accum_steps or args.grad_accum_steps}/{args.grad_accum_steps} "
                f"lr={current_lr:.2e}"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.save_dir, step, model, optimizer, args, fixed_width_metadata)

    average_loss = running_loss / max(1, args.steps)
    print(f"Training complete. Average loss: {average_loss:.4f}")


if __name__ == "__main__":
    main()
