import argparse
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

from elastic_modeling.budget_loss import compute_budget_loss, distillation_loss
from elastic_modeling.elastic_qwen import ElasticQwen2ForCausalLM
from elastic_modeling.gumbel_utils import (
    resolve_router_controls,
    sample_router_outputs_batch_shared,
)
from elastic_modeling.router import BudgetRouter
from helper_functions.helper_functions import collate_batch, tokenize_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a phase-1 flex-MLP + layer-skip router.")
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.2)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=0)
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
        default=(0.35, 0.5, 0.75, 1.0),
        help="Target model compute ratios sampled during training.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_d_choices(intermediate_size):
    return sorted({max(1, intermediate_size // 4), max(1, intermediate_size // 2), intermediate_size})


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


def tau_for_step(step, total_steps, start, end):
    if total_steps <= 1:
        return end
    progress = step / float(total_steps - 1)
    return start + progress * (end - start)


def sample_budget_indices(batch_size, num_budgets, device):
    sampled = torch.randint(0, num_budgets, (1,), device=device)
    return sampled.expand(batch_size)


def resolve_learning_rates(args):
    default_backbone_lr = 2e-5
    default_router_lr = 1e-4

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

    return torch.optim.AdamW(param_groups)


def save_checkpoint(save_dir, step, elastic_model, optimizer, args):
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
            "backbone_lr": args.backbone_lr,
            "router_lr": args.router_lr,
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

    teacher_model = None
    if args.distill_weight > 0.0:
        teacher_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    ds = load_from_disk(args.dataset_path)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    tokenized_ds = tokenize_dataset(ds, tokenizer, args.max_length)
    print(f"Loaded {len(tokenized_ds)} tokenized examples")

    intermediate_size = trainable_base_model.config.intermediate_size
    if args.d_choices is None:
        args.d_choices = get_default_d_choices(intermediate_size)
    args.d_choices = sorted({int(choice) for choice in args.d_choices})

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

    example_stream = cycle_tokenized_examples(tokenized_ds)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    effective_batch_size = args.batch_size * args.grad_accum_steps

    for step in range(1, args.steps + 1):
        batch_examples = [next(example_stream) for _ in range(args.batch_size)]
        batch = collate_batch(batch_examples, tokenizer.pad_token_id)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        batch_budget_idx = sample_budget_indices(args.batch_size, len(args.budget_values), DEVICE)
        target_budget = torch.tensor(
            [args.budget_values[int(batch_budget_idx[0].item())]],
            device=DEVICE,
            dtype=torch.float32,
        ).expand(args.batch_size)
        tau = tau_for_step(step - 1, args.steps, args.tau_start, args.tau_end)

        router_out = router(batch_budget_idx, device=DEVICE)
        sampled_router_out = sample_router_outputs_batch_shared(
            router_out, tau=tau, hard=False
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

            lm_loss = outputs.loss
            budget_stats = compute_budget_loss(
                config=elastic_model.config,
                target_budget=target_budget,
                d_choices=args.d_choices,
                layer_keep_probs=sampled_router_out["layer_keep_probs"],
                d_probs=sampled_router_out["d_probs"],
            )

        kd_loss = lm_loss.new_zeros(())
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            kd_autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.use_bf16 and DEVICE == "cuda"
                else nullcontext()
            )
            with kd_autocast:
                kd_loss = distillation_loss(
                    student_logits=outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    labels=batch["labels"],
                    temperature=args.temperature,
                )

        total_loss = (
            args.ce_weight * lm_loss
            + args.distill_weight * kd_loss
            + args.budget_weight * budget_stats["loss"]
        )
        scaled_loss = total_loss / args.grad_accum_steps

        scaled_loss.backward()

        should_step = step % args.grad_accum_steps == 0 or step == args.steps
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(total_loss.item())

        if step % args.log_every == 0 or step == 1:
            resolved = resolve_router_controls(sampled_router_out, args.d_choices)
            mean_kept_layers = resolved["layer_keep"].float().mean().item()
            mean_width_ratio = (
                resolved["d_keep"].float() / float(intermediate_size)
            ).mean().item()
            backbone_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            router_lr = optimizer.param_groups[-1]["lr"] if optimizer.param_groups else 0.0
            print(
                f"step={step:04d} "
                f"tau={tau:.3f} "
                f"total={total_loss.item():.4f} "
                f"lm={lm_loss.item():.4f} "
                f"kd={kd_loss.item():.4f} "
                f"budget={budget_stats['loss'].item():.4f} "
                f"target={target_budget[0].item():.3f} "
                f"achieved={budget_stats['achieved_budget'].mean().item():.3f} "
                f"keep={mean_kept_layers:.3f} "
                f"width={mean_width_ratio:.3f} "
                f"eff_batch={effective_batch_size} "
                f"accum={step % args.grad_accum_steps or args.grad_accum_steps}/{args.grad_accum_steps} "
                f"backbone_lr={backbone_lr:.2e} "
                f"router_lr={router_lr:.2e}"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.save_dir, step, elastic_model, optimizer, args)

    average_loss = running_loss / max(1, args.steps)
    print(f"Training complete. Average total loss: {average_loss:.4f}")


if __name__ == "__main__":
    main()
