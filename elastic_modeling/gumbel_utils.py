import torch
import torch.nn.functional as F


def _sanitize_logits(logits, clamp_value=10.0):
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    return logits.clamp(min=-clamp_value, max=clamp_value)


def logits_to_probs(logits, tau=1.0, logit_scale=1.0):
    safe_logits = _sanitize_logits(logits)
    tau = max(float(tau), 1e-6)
    scaled_logits = safe_logits * float(logit_scale)
    probs = F.softmax(scaled_logits / tau, dim=-1)
    eps = 1e-6
    probs = probs.clamp(min=eps)
    return probs / probs.sum(dim=-1, keepdim=True)


def sample_gumbel_softmax(logits, tau=1.0, hard=False, logit_scale=1.0):
    safe_logits = _sanitize_logits(logits)
    scaled_logits = safe_logits * float(logit_scale)
    samples = F.gumbel_softmax(scaled_logits, tau=tau, hard=hard, dim=-1)
    if hard:
        return samples

    # Keep router probabilities away from exact zeros/ones so the soft elastic
    # path stays numerically stable during backward.
    eps = 1e-6
    samples = samples.clamp(min=eps)
    return samples / samples.sum(dim=-1, keepdim=True)


def sample_router_outputs(router_out, tau=1.0, hard=False, logit_scale=1.0):
    d_probs = sample_gumbel_softmax(
        router_out["d_logits"], tau=tau, hard=hard, logit_scale=logit_scale
    )
    layer_keep_probs = sample_gumbel_softmax(
        router_out["layer_keep_logits"], tau=tau, hard=hard, logit_scale=logit_scale
    )

    return {
        "h": router_out["h"],
        "d_probs": d_probs,
        "layer_keep_probs": layer_keep_probs,
    }


def router_probs_from_logits(router_out, tau=1.0, logit_scale=1.0):
    return {
        "h": router_out["h"],
        "d_probs": logits_to_probs(router_out["d_logits"], tau=tau, logit_scale=logit_scale),
        "layer_keep_probs": logits_to_probs(
            router_out["layer_keep_logits"], tau=tau, logit_scale=logit_scale
        ),
    }


def sample_router_outputs_batch_shared(router_out, tau=1.0, hard=False, logit_scale=1.0):
    batch_size = router_out["d_logits"].shape[0]

    shared_d_probs = sample_gumbel_softmax(
        router_out["d_logits"][:1], tau=tau, hard=hard, logit_scale=logit_scale
    )
    shared_layer_keep_probs = sample_gumbel_softmax(
        router_out["layer_keep_logits"][:1], tau=tau, hard=hard, logit_scale=logit_scale
    )

    d_probs = shared_d_probs.expand(batch_size, -1, -1)
    layer_keep_probs = shared_layer_keep_probs.expand(batch_size, -1, -1)

    return {
        "h": router_out["h"],
        "d_probs": d_probs,
        "layer_keep_probs": layer_keep_probs,
    }


def resolve_router_controls(sampled_router_out, d_choices):
    d_choice_tensor = torch.as_tensor(
        d_choices,
        device=sampled_router_out["d_probs"].device,
        dtype=torch.long,
    )

    d_indices = sampled_router_out["d_probs"].argmax(dim=-1)
    layer_keep_indices = sampled_router_out["layer_keep_probs"].argmax(dim=-1)

    return {
        "d_keep": d_choice_tensor[d_indices],
        "layer_keep": layer_keep_indices.eq(1),
        "d_indices": d_indices,
        "layer_keep_indices": layer_keep_indices,
    }
