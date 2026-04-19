import torch
import torch.nn.functional as F


def sample_gumbel_softmax(logits, tau=1.0, hard=False):
    """
    logits: [B, K]
    returns: [B, K]
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


def sample_router_outputs(router_out, tau=1.0, hard=False):
    d_probs = sample_gumbel_softmax(router_out["d_logits"], tau=tau, hard=hard)

    lambda_probs = [
        sample_gumbel_softmax(lg, tau=tau, hard=hard)
        for lg in router_out["lambda_logits"]
    ]

    return {
        "h": router_out["h"],
        "d_probs": d_probs,
        "lambda_probs": lambda_probs,
    }