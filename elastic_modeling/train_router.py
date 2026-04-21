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
    router_probs_from_logits,
    resolve_router_controls,
    sample_router_outputs_batch_shared,
)
from elastic_modeling.router import BudgetRouter
from helper_functions.helper_functions import collate_batch, tokenize_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a phase-1 flex-MLP elastic router.")
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
    parser.add_argument("--save-dir", default=str(REPO_ROOT / "checkpoints" / "router_phase1"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--router-lr", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.2)
    parser.add_argument("--logit-scale-start", type=float, default=1.0)
    parser.add_argument("--logit-scale-end", type=float, default=4.0)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=1.0)
    parser.add_argument("--keep-penalty-weight", type=float, default=1.0)
    parser.add_argument("--min-keep-ratio", type=float, default=0.25)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--fail-on-nan", action="store_true")
    parser.add_argument("--save-failure-state", action="store_true")
    parser.add_argument("--skip-non-finite-steps", action="store_true")
    parser.add_argument("--enable-layer-skip", action="store_true")
    parser.add_argument(
        "--training-mode",
        choices=("end_to_end", "router_only"),
        default="end_to_end",
    )
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--d-choices",
        type=int,
        nargs="+",
        default=None,
        help="MLP widths to expose to the router. Defaults to [25%, 50%, 100%].",
    )
    parser.add_argument(
        "--budget-values",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 0.75, 1.0),
        help="Target model compute ratios sampled during training.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_d_choices(intermediate_size):
    ratios = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    return sorted({max(1, int(round(intermediate_size * ratio / 64.0) * 64)) for ratio in ratios})


def configure_trainable_params(elastic_model, training_mode):
    if training_mode == "end_to_end":
        for param in elastic_model.parameters():
            param.requires_grad = True
        return

    for param in elastic_model.parameters():
        param.requires_grad = False
    for param in elastic_model.model.router.parameters():
        param.requires_grad = True


def cycle_tokenized_examples(tokenized_ds):
    while True:
        for idx in range(len(tokenized_ds)):
            yield tokenized_ds[idx]


def linear_schedule_for_step(step, total_steps, start, end):
    if total_steps <= 1:
        return end
    progress = step / float(total_steps - 1)
    return start + progress * (end - start)


def sample_budget_indices(batch_size, num_budgets, device):
    sampled = torch.randint(0, num_budgets, (1,), device=device)
    return sampled.expand(batch_size)


def resolve_learning_rates(args):
    default_backbone_lr = 1e-5
    default_router_lr = 2e-5

    if args.lr is not None:
        if args.backbone_lr is None:
            args.backbone_lr = args.lr
        if args.router_lr is None:
            args.router_lr = args.lr

    if args.backbone_lr is None:
        args.backbone_lr = default_backbone_lr
    if args.router_lr is None:
        args.router_lr = default_router_lr


def create_optimizer(elastic_model, args):
    router_params = []
    backbone_params = []
    for name, param in elastic_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("model.router."):
            router_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": args.backbone_lr,
                "weight_decay": args.weight_decay,
            }
        )
    if router_params:
        param_groups.append(
            {
                "params": router_params,
                "lr": args.router_lr,
                "weight_decay": args.weight_decay,
            }
        )

    return torch.optim.AdamW(param_groups, eps=1e-6)


def create_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_always_keep_probs(batch_size, num_layers, device):
    layer_keep_probs = torch.zeros((batch_size, num_layers, 2), device=device, dtype=torch.float32)
    layer_keep_probs[..., 1] = 1.0
    return layer_keep_probs


def first_non_finite_name(named_tensors):
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            return name
    return None


def raise_or_checkpoint_non_finite(
    *,
    step,
    args,
    elastic_model,
    optimizer,
    budget_accounting_mode,
    parameter_count_components,
    named_tensors,
    context,
    raise_if_requested=True,
):
    bad_name = first_non_finite_name(named_tensors)
    if bad_name is None:
        return

    if args.save_failure_state:
        failure_dir = Path(args.save_dir) / "failures"
        os.makedirs(failure_dir, exist_ok=True)
        failure_path = failure_dir / f"failure_step_{step:06d}.pt"
        torch.save(
            {
                "step": step,
                "context": context,
                "bad_tensor": bad_name,
                "elastic_model_state_dict": elastic_model.state_dict(),
                "router_state_dict": elastic_model.model.router.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "budget_values": list(args.budget_values),
                "d_choices": list(args.d_choices),
                "enable_layer_skip": args.enable_layer_skip,
                "budget_accounting_mode": budget_accounting_mode,
                "parameter_count_components": dict(parameter_count_components),
            },
            failure_path,
        )

    if args.fail_on_nan and raise_if_requested:
        raise RuntimeError(f"Non-finite tensor detected at step {step} during {context}: {bad_name}")


def sanitize_or_skip_non_finite_grads(
    step,
    args,
    elastic_model,
    optimizer,
    budget_accounting_mode,
    parameter_count_components,
):
    bad_name = None
    for name, param in elastic_model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad_name = name
            break

    if bad_name is None:
        return False, None

    raise_or_checkpoint_non_finite(
        step=step,
        args=args,
        elastic_model=elastic_model,
        optimizer=optimizer,
        budget_accounting_mode=budget_accounting_mode,
        parameter_count_components=parameter_count_components,
        named_tensors=[(f"grad::{bad_name}", dict(elastic_model.named_parameters())[bad_name].grad)],
        context="backward",
        raise_if_requested=False,
    )

    if not args.skip_non_finite_steps:
        return True, bad_name

    for param in elastic_model.parameters():
        if param.grad is not None:
            param.grad = None
    return True, bad_name


def save_checkpoint(
    save_dir,
    step,
    elastic_model,
    optimizer,
    args,
    budget_accounting_mode,
    parameter_count_components,
):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = Path(save_dir) / f"router_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "elastic_model_state_dict": elastic_model.state_dict(),
            "router_state_dict": elastic_model.model.router.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "budget_values": list(args.budget_values),
            "d_choices": list(args.d_choices),
            "training_mode": args.training_mode,
            "grad_accum_steps": args.grad_accum_steps,
            "use_bf16": args.use_bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "enable_layer_skip": args.enable_layer_skip,
            "budget_accounting_mode": budget_accounting_mode,
            "parameter_count_components": dict(parameter_count_components),
            "logit_scale_start": args.logit_scale_start,
            "logit_scale_end": args.logit_scale_end,
            "loss_objective": "sample_one_budget_per_batch(task_loss + budget_weight * budget_hinge)",
            "backbone_lr": args.backbone_lr,
            "router_lr": args.router_lr,
            "budget_weight": args.budget_weight,
        },
        checkpoint_path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    resolve_learning_rates(args)

    print(f"Using device: {DEVICE}")

    trainable_base_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(args.dataset_path)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    tokenized_ds = tokenize_dataset(ds, tokenizer, args.max_length)
    print(f"Loaded {len(tokenized_ds)} tokenized examples")

    intermediate_size = trainable_base_model.config.intermediate_size
    if args.d_choices is None:
        args.d_choices = get_default_d_choices(intermediate_size)
    args.d_choices = sorted({int(choice) for choice in args.d_choices})
    budget_accounting_mode = resolve_budget_accounting_mode(args.enable_layer_skip)
    parameter_count_components = build_parameter_count_components(
        config=trainable_base_model.config,
        layer=trainable_base_model.model.layers[0],
        accounting_mode=budget_accounting_mode,
    )
    print(f"Budget accounting mode: {budget_accounting_mode}")

    router = BudgetRouter(
        budget_values=args.budget_values,
        num_layers=trainable_base_model.config.num_hidden_layers,
        d_choices=args.d_choices,
        hidden_dim=args.hidden_dim,
    ).to(DEVICE)

    elastic_model = ElasticQwen2ForCausalLM(
        base_causallm=trainable_base_model,
        d_choices=args.d_choices,
        router=router,
    ).to(DEVICE)
    configure_trainable_params(elastic_model, args.training_mode)
    if args.gradient_checkpointing:
        elastic_model.model.gradient_checkpointing_enable()
    elastic_model.train()

    optimizer = create_optimizer(elastic_model, args)
    scheduler = create_scheduler(optimizer, args.steps, args.warmup_ratio)

    example_stream = cycle_tokenized_examples(tokenized_ds)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    effective_batch_size = args.batch_size * args.grad_accum_steps

    for step in range(1, args.steps + 1):
        batch_examples = [next(example_stream) for _ in range(args.batch_size)]
        batch = collate_batch(batch_examples, tokenizer.pad_token_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        batch_budget_idx = sample_budget_indices(args.batch_size, len(args.budget_values), DEVICE)
        sampled_budget_idx = int(batch_budget_idx[0].item())
        sampled_budget_value = float(args.budget_values[sampled_budget_idx])
        target_budget = torch.full(
            (args.batch_size,),
            fill_value=sampled_budget_value,
            device=DEVICE,
            dtype=torch.float32,
        )
        tau = linear_schedule_for_step(step - 1, args.steps, args.tau_start, args.tau_end)
        logit_scale = linear_schedule_for_step(
            step - 1, args.steps, args.logit_scale_start, args.logit_scale_end
        )

        router_out = router(batch_budget_idx, device=DEVICE)
        raise_or_checkpoint_non_finite(
            step=step,
            args=args,
            elastic_model=elastic_model,
            optimizer=optimizer,
            budget_accounting_mode=budget_accounting_mode,
            parameter_count_components=parameter_count_components,
            named_tensors=[
                ("router_out.d_logits", router_out["d_logits"]),
                ("router_out.layer_keep_logits", router_out["layer_keep_logits"]),
            ],
            context=f"router forward (budget={sampled_budget_value:.3f})",
        )
        sampled_router_out = sample_router_outputs_batch_shared(
            router_out, tau=tau, hard=True, logit_scale=logit_scale
        )
        router_prob_out = router_probs_from_logits(
            router_out, tau=tau, logit_scale=logit_scale
        )
        if not args.enable_layer_skip:
            sampled_router_out["layer_keep_probs"] = build_always_keep_probs(
                batch_size=args.batch_size,
                num_layers=elastic_model.config.num_hidden_layers,
                device=DEVICE,
            )
            router_prob_out["layer_keep_probs"] = build_always_keep_probs(
                batch_size=args.batch_size,
                num_layers=elastic_model.config.num_hidden_layers,
                device=DEVICE,
            )
        raise_or_checkpoint_non_finite(
            step=step,
            args=args,
            elastic_model=elastic_model,
            optimizer=optimizer,
            budget_accounting_mode=budget_accounting_mode,
            parameter_count_components=parameter_count_components,
            named_tensors=[
                ("sampled_router_out.d_probs", sampled_router_out["d_probs"]),
                ("sampled_router_out.layer_keep_probs", sampled_router_out["layer_keep_probs"]),
            ],
            context=f"router sampling (budget={sampled_budget_value:.3f})",
        )

        train_autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.use_bf16 and DEVICE == "cuda"
            else nullcontext()
        )
        with train_autocast:
            outputs = elastic_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                d_probs=sampled_router_out["d_probs"],
                layer_keep_probs=sampled_router_out["layer_keep_probs"],
            )

            router_loss = outputs.loss
            budget_stats = compute_budget_loss(
                config=elastic_model.config,
                target_budget=target_budget,
                d_choices=args.d_choices,
                layer_keep_probs=router_prob_out["layer_keep_probs"],
                d_probs=router_prob_out["d_probs"],
                accounting_mode=budget_accounting_mode,
                parameter_count_components=parameter_count_components,
            )
            budget_loss = budget_stats["loss"]
            total_loss = router_loss + args.budget_weight * budget_loss

        raise_or_checkpoint_non_finite(
            step=step,
            args=args,
            elastic_model=elastic_model,
            optimizer=optimizer,
            budget_accounting_mode=budget_accounting_mode,
            parameter_count_components=parameter_count_components,
            named_tensors=[
                ("outputs.logits", outputs.logits),
                ("router_loss", router_loss),
                ("budget_loss", budget_loss),
                ("budget_achieved", budget_stats["achieved_budget"]),
                ("budget_expected_params", budget_stats["expected_params"]),
            ],
            context=f"elastic forward (budget={sampled_budget_value:.3f})",
        )
        raise_or_checkpoint_non_finite(
            step=step,
            args=args,
            elastic_model=elastic_model,
            optimizer=optimizer,
            budget_accounting_mode=budget_accounting_mode,
            parameter_count_components=parameter_count_components,
            named_tensors=[
                ("total_loss", total_loss),
            ],
            context=f"loss aggregation (budget={sampled_budget_value:.3f})",
        )

        (total_loss / args.grad_accum_steps).backward()

        should_step = step % args.grad_accum_steps == 0 or step == args.steps
        if should_step:
            skipped_step, bad_grad_name = sanitize_or_skip_non_finite_grads(
                step=step,
                args=args,
                elastic_model=elastic_model,
                optimizer=optimizer,
                budget_accounting_mode=budget_accounting_mode,
                parameter_count_components=parameter_count_components,
            )
            if skipped_step:
                print(
                    f"warning: skipped optimizer step at step {step} due to non-finite gradient in {bad_grad_name}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in elastic_model.parameters() if p.requires_grad],
                    max_norm=args.grad_clip_norm,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(total_loss.item())

        if step % args.log_every == 0 or step == 1:
            resolved = resolve_router_controls(sampled_router_out, args.d_choices)
            keep_ratio = resolved["layer_keep"].float().mean().item()
            width_ratio = (
                resolved["d_keep"].float() / float(intermediate_size)
            ).mean().item()
            backbone_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            router_lr = optimizer.param_groups[-1]["lr"] if optimizer.param_groups else 0.0
            print(
                f"step={step:04d} "
                f"k={logit_scale:.3f} "
                f"tau={tau:.3f} "
                f"total={total_loss.item():.4f} "
                f"router={router_loss.item():.4f} "
                f"budget={budget_loss.item():.4f} "
                f"target={sampled_budget_value:.3f} "
                f"achieved={budget_stats['achieved_budget'].mean().item():.3f} "
                f"keep={keep_ratio:.3f} "
                f"width={width_ratio:.3f} "
                f"eff_batch={effective_batch_size} "
                f"accum={step % args.grad_accum_steps or args.grad_accum_steps}/{args.grad_accum_steps} "
                f"backbone_lr={backbone_lr:.2e} "
                f"router_lr={router_lr:.2e}"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                args.save_dir,
                step,
                elastic_model,
                optimizer,
                args,
                budget_accounting_mode,
                parameter_count_components,
            )

    average_loss = running_loss / max(1, args.steps)
    print(f"Training complete. Average total loss: {average_loss:.4f}")


if __name__ == "__main__":
    main()
