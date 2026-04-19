import torch
import torch.nn.functional as F


def get_layer_cost_components(config):
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))

    q_cost = hidden_size * (num_attention_heads * head_dim)
    k_cost = hidden_size * (num_key_value_heads * head_dim)
    v_cost = hidden_size * (num_key_value_heads * head_dim)
    o_cost = (num_attention_heads * head_dim) * hidden_size
    attn_cost = q_cost + k_cost + v_cost + o_cost

    gate_cost = hidden_size * intermediate_size
    up_cost = hidden_size * intermediate_size
    down_cost = intermediate_size * hidden_size
    mlp_full_cost = gate_cost + up_cost + down_cost

    return {
        "attn_cost": float(attn_cost),
        "mlp_full_cost": float(mlp_full_cost),
        "full_layer_cost": float(attn_cost + mlp_full_cost),
    }


def expected_budget_from_probs(layer_keep_probs, d_probs, d_choices, config):
    if layer_keep_probs.dim() != 3 or layer_keep_probs.shape[-1] != 2:
        raise ValueError("layer_keep_probs must have shape [batch, layers, 2]")
    if d_probs.dim() != 3:
        raise ValueError("d_probs must have shape [batch, layers, num_d_choices]")
    if layer_keep_probs.shape[:2] != d_probs.shape[:2]:
        raise ValueError("layer_keep_probs and d_probs must agree on [batch, layers]")

    d_choice_tensor = torch.as_tensor(d_choices, device=d_probs.device, dtype=d_probs.dtype)
    d_ratios = d_choice_tensor / float(config.intermediate_size)
    expected_d_ratio = (d_probs * d_ratios.view(1, 1, -1)).sum(dim=-1)

    keep_prob = layer_keep_probs[..., 1]
    costs = get_layer_cost_components(config)
    attn_ratio = costs["attn_cost"] / costs["full_layer_cost"]
    mlp_ratio = costs["mlp_full_cost"] / costs["full_layer_cost"]

    expected_layer_ratio = keep_prob * (attn_ratio + mlp_ratio * expected_d_ratio)
    return expected_layer_ratio.mean(dim=-1)


def concrete_budget_from_controls(layer_keep, d_keep, config):
    if layer_keep.dim() != 2 or d_keep.dim() != 2:
        raise ValueError("layer_keep and d_keep must have shape [batch, layers]")

    costs = get_layer_cost_components(config)
    attn_ratio = costs["attn_cost"] / costs["full_layer_cost"]
    mlp_ratio = costs["mlp_full_cost"] / costs["full_layer_cost"]

    keep = layer_keep.to(dtype=torch.float32)
    d_ratio = d_keep.to(dtype=torch.float32) / float(config.intermediate_size)
    layer_ratio = keep * (attn_ratio + mlp_ratio * d_ratio)
    return layer_ratio.mean(dim=-1)


def budget_alignment_loss(expected_budget, target_budget, reduction="mean"):
    target_budget = target_budget.to(device=expected_budget.device, dtype=expected_budget.dtype)
    return F.mse_loss(expected_budget, target_budget, reduction=reduction)


def compute_budget_loss(
    config,
    target_budget,
    d_choices,
    layer_keep_probs=None,
    d_probs=None,
    layer_keep=None,
    d_keep=None,
):
    if layer_keep_probs is not None and d_probs is not None:
        achieved_budget = expected_budget_from_probs(
            layer_keep_probs=layer_keep_probs,
            d_probs=d_probs,
            d_choices=d_choices,
            config=config,
        )
    elif layer_keep is not None and d_keep is not None:
        achieved_budget = concrete_budget_from_controls(
            layer_keep=layer_keep,
            d_keep=d_keep,
            config=config,
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

    loss = budget_alignment_loss(achieved_budget, target_budget)
    return {
        "loss": loss,
        "achieved_budget": achieved_budget,
        "target_budget": target_budget,
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
