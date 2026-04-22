from __future__ import annotations

from pathlib import Path

import torch

from elastic_modeling.budget_loss import (
    BLOCK_PARAMS_ACCOUNTING,
    build_parameter_count_components,
    concrete_parameter_count_from_controls,
)


def budget_to_tag(budget: float) -> str:
    return str(budget).replace(".", "p")


def build_fixed_width_controls(batch_size, num_layers, d_keep, device):
    layer_keep = torch.ones((batch_size, num_layers), device=device, dtype=torch.bool)
    d_keep_tensor = torch.full(
        (batch_size, num_layers),
        fill_value=int(d_keep),
        device=device,
        dtype=torch.long,
    )
    return layer_keep, d_keep_tensor


def resolve_pruned_width_for_budget(
    *,
    config,
    layer,
    target_budget,
    width_granularity=1,
    parameter_count_components=None,
):
    if width_granularity <= 0:
        raise ValueError(f"width_granularity must be positive, got {width_granularity}")

    target_budget = float(target_budget)
    if target_budget <= 0.0 or target_budget > 1.0:
        raise ValueError(f"target_budget must be in (0, 1], got {target_budget}")

    components = (
        dict(parameter_count_components)
        if parameter_count_components is not None
        else build_parameter_count_components(
            config=config,
            layer=layer,
            accounting_mode=BLOCK_PARAMS_ACCOUNTING,
        )
    )
    if components.get("accounting_mode") != BLOCK_PARAMS_ACCOUNTING:
        raise ValueError("pruned baseline comparison must use block-parameter accounting")

    fixed_block_params = float(components["fixed_block_params"])
    mlp_full_params = float(components["mlp_full_params"])
    full_layer_params = float(components["full_layer_params"])
    min_budget = fixed_block_params / full_layer_params

    if target_budget < min_budget:
        raise ValueError(
            f"target_budget={target_budget:.6f} is below the minimum achievable width-only "
            f"budget {min_budget:.6f} under block-parameter accounting"
        )

    intermediate_size = int(config.intermediate_size)
    raw_width_ratio = (target_budget * full_layer_params - fixed_block_params) / mlp_full_params
    raw_width_ratio = min(1.0, max(0.0, raw_width_ratio))
    raw_d_keep = raw_width_ratio * intermediate_size

    if width_granularity == 1:
        resolved_d_keep = int(round(raw_d_keep))
    else:
        resolved_d_keep = int(round(raw_d_keep / width_granularity) * width_granularity)

    resolved_d_keep = max(1, min(intermediate_size, resolved_d_keep))
    if width_granularity > 1 and resolved_d_keep != intermediate_size:
        resolved_d_keep = max(width_granularity, resolved_d_keep)

    layer_keep, d_keep_tensor = build_fixed_width_controls(
        batch_size=1,
        num_layers=int(config.num_hidden_layers),
        d_keep=resolved_d_keep,
        device="cpu",
    )
    total_params, full_params, achieved_budget = concrete_parameter_count_from_controls(
        layer_keep=layer_keep,
        d_keep=d_keep_tensor,
        config=config,
        accounting_mode=BLOCK_PARAMS_ACCOUNTING,
        parameter_count_components=components,
    )

    return {
        "target_budget": target_budget,
        "min_width_only_budget": min_budget,
        "requested_width_ratio": raw_width_ratio,
        "requested_d_keep": raw_d_keep,
        "resolved_d_keep": resolved_d_keep,
        "resolved_width_ratio": resolved_d_keep / float(intermediate_size),
        "achieved_budget": float(achieved_budget.item()),
        "expected_params": float(total_params.item()),
        "full_params": float(full_params.item()),
        "width_granularity": int(width_granularity),
        "parameter_count_components": components,
    }


def checkpoint_path_for_step(save_dir, step, prefix="pruned_step"):
    return Path(save_dir) / f"{prefix}_{step:06d}.pt"
