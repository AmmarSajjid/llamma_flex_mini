import torch
import torch.nn.functional as F


MLP_ONLY_ACCOUNTING = "mlp_only"
BLOCK_PARAMS_ACCOUNTING = "block_params"
VALID_ACCOUNTING_MODES = {MLP_ONLY_ACCOUNTING, BLOCK_PARAMS_ACCOUNTING}


def _validate_accounting_mode(accounting_mode):
    if accounting_mode not in VALID_ACCOUNTING_MODES:
        raise ValueError(
            f"accounting_mode must be one of {sorted(VALID_ACCOUNTING_MODES)}, got {accounting_mode!r}"
        )
    return accounting_mode


def resolve_budget_accounting_mode(enable_layer_skip):
    return BLOCK_PARAMS_ACCOUNTING if enable_layer_skip else MLP_ONLY_ACCOUNTING


def build_parameter_count_components(config, layer=None, accounting_mode=MLP_ONLY_ACCOUNTING):
    accounting_mode = _validate_accounting_mode(accounting_mode)

    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_layers = int(config.num_hidden_layers)

    if layer is not None:
        mlp_full_params = float(sum(param.numel() for param in layer.mlp.parameters()))
        fixed_block_params = float(
            sum(param.numel() for name, param in layer.named_parameters() if not name.startswith("mlp."))
        )
    else:
        gate_proj_params = hidden_size * intermediate_size
        up_proj_params = hidden_size * intermediate_size
        down_proj_params = intermediate_size * hidden_size
        mlp_full_params = float(gate_proj_params + up_proj_params + down_proj_params)
        fixed_block_params = 0.0

    if accounting_mode == BLOCK_PARAMS_ACCOUNTING and layer is None:
        raise ValueError("layer is required to build block-parameter accounting components")

    full_layer_params = fixed_block_params + mlp_full_params
    if accounting_mode == BLOCK_PARAMS_ACCOUNTING:
        full_elastic_params = num_layers * full_layer_params
    else:
        full_elastic_params = num_layers * mlp_full_params

    return {
        "accounting_mode": accounting_mode,
        "hidden_size": float(hidden_size),
        "intermediate_size": float(intermediate_size),
        "num_layers": float(num_layers),
        "fixed_block_params": float(fixed_block_params),
        "mlp_full_params": float(mlp_full_params),
        "full_layer_params": float(full_layer_params),
        "full_elastic_params": float(full_elastic_params),
    }


def _resolve_parameter_count_components(
    config,
    parameter_count_components=None,
    accounting_mode=MLP_ONLY_ACCOUNTING,
):
    accounting_mode = _validate_accounting_mode(accounting_mode)
    if parameter_count_components is None:
        return build_parameter_count_components(config, accounting_mode=accounting_mode)

    components = dict(parameter_count_components)
    components_mode = _validate_accounting_mode(components.get("accounting_mode", accounting_mode))
    if components_mode != accounting_mode:
        raise ValueError(
            f"parameter_count_components mode {components_mode!r} does not match accounting_mode {accounting_mode!r}"
        )

    required_keys = {
        "accounting_mode",
        "fixed_block_params",
        "mlp_full_params",
        "full_layer_params",
        "full_elastic_params",
        "num_layers",
    }
    missing_keys = sorted(required_keys.difference(components))
    if missing_keys:
        raise ValueError(f"parameter_count_components missing keys: {missing_keys}")

    return components


def expected_parameter_count_from_probs(
    layer_keep_probs,
    d_probs,
    d_choices,
    config,
    accounting_mode=MLP_ONLY_ACCOUNTING,
    parameter_count_components=None,
):
    if layer_keep_probs.dim() != 3 or layer_keep_probs.shape[-1] != 2:
        raise ValueError("layer_keep_probs must have shape [batch, layers, 2]")
    if d_probs.dim() != 3:
        raise ValueError("d_probs must have shape [batch, layers, num_d_choices]")
    if layer_keep_probs.shape[:2] != d_probs.shape[:2]:
        raise ValueError("layer_keep_probs and d_probs must agree on [batch, layers]")

    components = _resolve_parameter_count_components(
        config=config,
        parameter_count_components=parameter_count_components,
        accounting_mode=accounting_mode,
    )
    d_choice_tensor = torch.as_tensor(d_choices, device=d_probs.device, dtype=d_probs.dtype)
    d_ratios = d_choice_tensor / float(config.intermediate_size)
    expected_d_ratio = (d_probs * d_ratios.view(1, 1, -1)).sum(dim=-1)

    if components["accounting_mode"] == BLOCK_PARAMS_ACCOUNTING:
        keep_prob = layer_keep_probs[..., 1]
        expected_layer_params = keep_prob * (
            components["fixed_block_params"] + components["mlp_full_params"] * expected_d_ratio
        )
    else:
        expected_layer_params = components["mlp_full_params"] * expected_d_ratio
    expected_total_params = expected_layer_params.sum(dim=-1)

    full_elastic_params = torch.full_like(expected_total_params, components["full_elastic_params"])
    achieved_budget = expected_total_params / full_elastic_params
    return expected_total_params, full_elastic_params, achieved_budget


def concrete_parameter_count_from_controls(
    layer_keep,
    d_keep,
    config,
    accounting_mode=MLP_ONLY_ACCOUNTING,
    parameter_count_components=None,
):
    if layer_keep.dim() != 2 or d_keep.dim() != 2:
        raise ValueError("layer_keep and d_keep must have shape [batch, layers]")

    components = _resolve_parameter_count_components(
        config=config,
        parameter_count_components=parameter_count_components,
        accounting_mode=accounting_mode,
    )
    d_ratio = d_keep.to(dtype=torch.float32) / float(config.intermediate_size)
    if components["accounting_mode"] == BLOCK_PARAMS_ACCOUNTING:
        keep = layer_keep.to(dtype=torch.float32)
        layer_params = keep * (components["fixed_block_params"] + components["mlp_full_params"] * d_ratio)
    else:
        layer_params = components["mlp_full_params"] * d_ratio
    total_params = layer_params.sum(dim=-1)
    full_elastic_params = torch.full_like(total_params, components["full_elastic_params"])
    achieved_budget = total_params / full_elastic_params
    return total_params, full_elastic_params, achieved_budget


def budget_hinge_loss(expected_params, target_params, reduction="mean"):
    target_params = target_params.to(device=expected_params.device, dtype=expected_params.dtype)
    excess_params = F.relu(expected_params - target_params)
    if reduction == "none":
        return excess_params
    if reduction == "sum":
        return excess_params.sum()
    return excess_params.mean()


def compute_budget_loss(
    config,
    target_budget,
    d_choices,
    layer_keep_probs=None,
    d_probs=None,
    layer_keep=None,
    d_keep=None,
    accounting_mode=MLP_ONLY_ACCOUNTING,
    parameter_count_components=None,
):
    if layer_keep_probs is not None and d_probs is not None:
        expected_params, full_params, achieved_budget = expected_parameter_count_from_probs(
            layer_keep_probs=layer_keep_probs,
            d_probs=d_probs,
            d_choices=d_choices,
            config=config,
            accounting_mode=accounting_mode,
            parameter_count_components=parameter_count_components,
        )
    elif layer_keep is not None and d_keep is not None:
        expected_params, full_params, achieved_budget = concrete_parameter_count_from_controls(
            layer_keep=layer_keep,
            d_keep=d_keep,
            config=config,
            accounting_mode=accounting_mode,
            parameter_count_components=parameter_count_components,
        )
    else:
        raise ValueError(
            "Provide either probabilistic controls (layer_keep_probs, d_probs) "
            "or concrete controls (layer_keep, d_keep)"
        )

    target_budget = torch.as_tensor(
        target_budget,
        device=achieved_budget.device,
        dtype=achieved_budget.dtype,
    )
    if target_budget.dim() == 0:
        target_budget = target_budget.expand_as(achieved_budget)

    target_params = target_budget * full_params
    loss = budget_hinge_loss(expected_params, target_params)
    return {
        "loss": loss,
        "accounting_mode": accounting_mode,
        "achieved_budget": achieved_budget,
        "target_budget": target_budget,
        "expected_params": expected_params,
        "target_params": target_params,
        "full_params": full_params,
        "excess_params": F.relu(expected_params - target_params),
    }


def keep_ratio_penalty(layer_keep_probs=None, layer_keep=None, min_keep_ratio=0.0):
    if min_keep_ratio <= 0.0:
        if layer_keep_probs is not None:
            return layer_keep_probs.new_zeros(())
        if layer_keep is not None:
            return layer_keep.float().new_zeros(())
        raise ValueError("Provide layer_keep_probs or layer_keep when min_keep_ratio > 0")

    if layer_keep_probs is not None:
        keep_ratio = layer_keep_probs[..., 1].mean(dim=-1)
    elif layer_keep is not None:
        keep_ratio = layer_keep.float().mean(dim=-1)
    else:
        raise ValueError("Provide layer_keep_probs or layer_keep")

    floor = torch.full_like(keep_ratio, float(min_keep_ratio))
    return F.relu(floor - keep_ratio).pow(2).mean()


def distillation_loss(student_logits, teacher_logits, labels, temperature=1.0):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")

    valid_positions = labels != -100
    if not valid_positions.any():
        return student_logits.new_zeros(())

    student = student_logits[valid_positions] / temperature
    teacher = teacher_logits[valid_positions] / temperature
    loss = F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    )
    return loss * (temperature ** 2)
