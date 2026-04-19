import torch
import torch.nn as nn


class ElasticQwen2MLP(nn.Module):
    def __init__(self, base_mlp, d_choices):
        super().__init__()
        self.base_mlp = base_mlp
        self.d_choices = sorted({int(d) for d in d_choices})
        self.intermediate_size = base_mlp.gate_proj.out_features

    def _validate_d_keep(self, d_keep):
        d_keep = int(d_keep)
        if d_keep <= 0 or d_keep > self.intermediate_size:
            raise ValueError(
                f"d_keep must be in [1, {self.intermediate_size}], got {d_keep}"
            )
        return d_keep

    def forward(self, hidden_states, d_keep=None):
        if d_keep is None:
            return self.base_mlp(hidden_states)

        d_keep = self._validate_d_keep(d_keep)

        gate_weight = self.base_mlp.gate_proj.weight[:d_keep, :]
        up_weight = self.base_mlp.up_proj.weight[:d_keep, :]
        down_weight = self.base_mlp.down_proj.weight[:, :d_keep]

        gate = nn.functional.linear(hidden_states, gate_weight)
        up = nn.functional.linear(hidden_states, up_weight)
        activated = self.base_mlp.act_fn(gate) * up
        return nn.functional.linear(activated, down_weight)

    def forward_soft(self, hidden_states, d_probs):
        if d_probs.dim() != 1 or d_probs.shape[0] != len(self.d_choices):
            raise ValueError(
                f"d_probs must have shape ({len(self.d_choices)},), got {tuple(d_probs.shape)}"
            )

        output = None
        for prob, d_choice in zip(d_probs, self.d_choices):
            mlp_out = self.forward(hidden_states, d_keep=d_choice)
            weighted_out = prob.to(dtype=mlp_out.dtype) * mlp_out
            output = weighted_out if output is None else output + weighted_out
        return output


ElasticQwenMLP = ElasticQwen2MLP
