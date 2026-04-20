import torch
import torch.nn.functional as F


def get_parameter_count_components(config):
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    vocab_size = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    embed_tokens_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else hidden_size * vocab_size
    final_norm_params = hidden_size

    q_proj_params = hidden_size * (num_attention_heads * head_dim) + (num_attention_heads * head_dim)
    k_proj_params = hidden_size * (num_key_value_heads * head_dim) + (num_key_value_heads * head_dim)
    v_proj_params = hidden_size * (num_key_value_heads * head_dim) + (num_key_value_heads * head_dim)
    o_proj_params = (num_attention_heads * head_dim) * hidden_size
    attn_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params

    norm_params = 3 * hidden_size

    gate_proj_params = hidden_size * intermediate_size
    up_proj_params = hidden_size * intermediate_size
    down_proj_params = intermediate_size * hidden_size
    mlp_full_params = gate_proj_params + up_proj_params + down_proj_params

    full_layer_params = attn_params + (2 * hidden_size) + mlp_full_params
    shared_non_layer_params = embed_tokens_params + lm_head_params + final_norm_params
    full_model_params = shared_non_layer_params + num_layers * full_layer_params

    return {
        "shared_non_layer_params": float(shared_non_layer_params),
        "fixed_layer_params": float(attn_params + (2 * hidden_size)),
        "mlp_full_params": float(mlp_full_params),
        "full_layer_params": float(full_layer_params),
        "full_model_params": float(full_model_params),
        "norm_params": float(norm_params),
    }


def expected_parameter_count_from_probs(layer_keep_probs, d_probs, d_choices, config):
    if layer_keep_probs.dim() != 3 or layer_keep_probs.shape[-1] != 2:
        raise ValueError("layer_keep_probs must have shape [batch, layers, 2]")
    if d_probs.dim() != 3:
        raise ValueError("d_probs must have shape [batch, layers, num_d_choices]")
    if layer_keep_probs.shape[:2] != d_probs.shape[:2]:
        raise ValueError("layer_keep_probs and d_probs must agree on [batch, layers]")

    components = get_parameter_count_components(config)
    d_choice_tensor = torch.as_tensor(d_choices, device=d_probs.device, dtype=d_probs.dtype)
    d_ratios = d_choice_tensor / float(config.intermediate_size)
    expected_d_ratio = (d_probs * d_ratios.view(1, 1, -1)).sum(dim=-1)

    keep_prob = layer_keep_probs[..., 1]
    expected_layer_params = keep_prob * (
        components["fixed_layer_params"] + components["mlp_full_params"] * expected_d_ratio
    )
    expected_total_params = components["shared_non_layer_params"] + expected_layer_params.sum(dim=-1)

    full_model_params = torch.full_like(expected_total_params, components["full_model_params"])
    achieved_budget = expected_total_params / full_model_params
    return expected_total_params, full_model_params, achieved_budget


def concrete_parameter_count_from_controls(layer_keep, d_keep, config):
    if layer_keep.dim() != 2 or d_keep.dim() != 2:
        raise ValueError("layer_keep and d_keep must have shape [batch, layers]")

    components = get_parameter_count_components(config)
    keep = layer_keep.to(dtype=torch.float32)
    d_ratio = d_keep.to(dtype=torch.float32) / float(config.intermediate_size)
    layer_params = keep * (
        components["fixed_layer_params"] + components["mlp_full_params"] * d_ratio
    )
    total_params = components["shared_non_layer_params"] + layer_params.sum(dim=-1)
    full_model_params = torch.full_like(total_params, components["full_model_params"])
    achieved_budget = total_params / full_model_params
    return total_params, full_model_params, achieved_budget


def budget_hinge_loss(achieved_budget, target_budget, reduction="mean"):
    target_budget = target_budget.to(device=achieved_budget.device, dtype=achieved_budget.dtype)
    excess_ratio = F.relu(achieved_budget - target_budget)
    if reduction == "none":
        return excess_ratio
    if reduction == "sum":
        return excess_ratio.sum()
    return excess_ratio.mean()


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
        expected_params, full_params, achieved_budget = expected_parameter_count_from_probs(
            layer_keep_probs=layer_keep_probs,
            d_probs=d_probs,
            d_choices=d_choices,
            config=config,
        )
    elif layer_keep is not None and d_keep is not None:
        expected_params, full_params, achieved_budget = concrete_parameter_count_from_controls(
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

    target_params = target_budget * full_params
    loss = budget_hinge_loss(achieved_budget, target_budget)
    return {
        "loss": loss,
        "achieved_budget": achieved_budget,
        "target_budget": target_budget,
        "expected_params": expected_params,
        "target_params": target_params,
        "full_params": full_params,
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
