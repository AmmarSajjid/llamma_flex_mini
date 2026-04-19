import torch
import torch.nn.functional as F


def sample_gumbel_softmax(logits, tau=1.0, hard=False):
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


def sample_router_outputs(router_out, tau=1.0, hard=False):
    d_probs = sample_gumbel_softmax(router_out["d_logits"], tau=tau, hard=hard)
    layer_keep_probs = sample_gumbel_softmax(
        router_out["layer_keep_logits"], tau=tau, hard=hard
    )

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
