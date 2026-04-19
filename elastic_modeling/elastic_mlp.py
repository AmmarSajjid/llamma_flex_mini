import torch
import torch.nn as nn
import torch.nn.functional as F


class ElasticQwenMLP(nn.Module):
    def __init__(self, base_mlp, d_choices):
        super().__init__()
        self.base_mlp = base_mlp
        self.d_choices = d_choices

    def forward(self, hidden_states, d_keep):
        """
        hidden_states: [B, T, H]
        d_keep: int, number of intermediate neurons to keep
        """
        gate_w = self.base_mlp.gate_proj.weight[:d_keep, :]
        up_w = self.base_mlp.up_proj.weight[:d_keep, :]
        down_w = self.base_mlp.down_proj.weight[:, :d_keep]

        gate = F.linear(hidden_states, gate_w)
        up = F.linear(hidden_states, up_w)
        act = F.silu(gate) * up
        out = F.linear(act, down_w)

        return out