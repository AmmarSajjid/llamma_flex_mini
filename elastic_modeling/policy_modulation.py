import math

import torch
import torch.nn as nn


class PolicyAwareModulator(nn.Module):
    def __init__(
        self,
        hidden_size,
        d_choices,
        intermediate_size,
        embed_dim=16,
        hidden_dim=128,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.hidden_size = int(hidden_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_size = float(intermediate_size)
        self.d_choices = sorted({int(d_choice) for d_choice in d_choices})

        width_ratios = torch.tensor(
            [d_choice / self.intermediate_size for d_choice in self.d_choices],
            dtype=torch.float32,
        )
        self.register_buffer("width_ratios", width_ratios, persistent=False)

        self.fc1 = nn.Linear(self.embed_dim * 2, self.hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_size * 2)

        # Start from an identity transform so modulation can be enabled without
        # changing behavior until training learns a useful adjustment.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _sinusoidal_embedding(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.dim() != 0:
            raise ValueError(f"values must be scalar-like, got shape {tuple(values.shape)}")

        half_dim = self.embed_dim // 2
        if half_dim == 0:
            return values.new_zeros((self.embed_dim,))

        freq_exponent = torch.arange(half_dim, device=values.device, dtype=values.dtype)
        denom = max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(10000.0) * (freq_exponent / denom))
        angles = values.unsqueeze(0) * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, embedding.new_zeros((1,))], dim=0)
        return embedding

    def width_embedding(self, *, d_keep=None, d_probs=None, device=None):
        if d_probs is not None:
            if d_probs.dim() != 1 or d_probs.shape[0] != len(self.d_choices):
                raise ValueError(
                    f"d_probs must have shape ({len(self.d_choices)},), got {tuple(d_probs.shape)}"
                )
            width_embeddings = torch.stack(
                [self._sinusoidal_embedding(ratio.to(device=d_probs.device)) for ratio in self.width_ratios],
                dim=0,
            )
            return (d_probs.to(dtype=width_embeddings.dtype).unsqueeze(-1) * width_embeddings).sum(dim=0)

        if d_keep is None:
            raise ValueError("Provide d_keep or d_probs for policy-aware modulation")

        width_ratio = torch.as_tensor(d_keep, dtype=torch.float32, device=device) / self.intermediate_size
        return self._sinusoidal_embedding(width_ratio)

    def budget_embedding(self, budget_value, device=None):
        if budget_value is None:
            raise ValueError("budget_value is required when policy-aware modulation is enabled")
        budget_value = torch.as_tensor(budget_value, dtype=torch.float32, device=device)
        return self._sinusoidal_embedding(budget_value)

    def conditioning_embedding(self, *, budget_value, d_keep=None, d_probs=None, device=None):
        width_embedding = self.width_embedding(d_keep=d_keep, d_probs=d_probs, device=device)
        budget_embedding = self.budget_embedding(budget_value=budget_value, device=width_embedding.device)
        return torch.cat([width_embedding, budget_embedding], dim=0)

    def modulation_parameters(self, *, budget_value, d_keep=None, d_probs=None, device=None, dtype=None):
        conditioning = self.conditioning_embedding(
            budget_value=budget_value,
            d_keep=d_keep,
            d_probs=d_probs,
            device=device,
        )
        hidden = self.act(self.fc1(conditioning))
        modulation = self.fc2(hidden)
        delta_scale, shift = modulation.chunk(2, dim=-1)
        if device is not None:
            delta_scale = delta_scale.to(device=device)
            shift = shift.to(device=device)
        if dtype is not None:
            delta_scale = delta_scale.to(dtype=dtype)
            shift = shift.to(dtype=dtype)
        return delta_scale, shift

    def forward(self, y, *, budget_value, d_keep=None, d_probs=None):
        delta_scale, shift = self.modulation_parameters(
            budget_value=budget_value,
            d_keep=d_keep,
            d_probs=d_probs,
            device=y.device,
            dtype=y.dtype,
        )
        broadcast_shape = (1,) * (y.dim() - 1) + (self.hidden_size,)
        delta_scale = delta_scale.view(broadcast_shape)
        shift = shift.view(broadcast_shape)
        return y * (1.0 + delta_scale) + shift
