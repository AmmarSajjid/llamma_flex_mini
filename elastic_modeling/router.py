import torch
import torch.nn as nn
import torch.nn.functional as F


class BudgetRouter(nn.Module):
    def __init__(self, budget_values, num_layers, d_choices, hidden_dim=128):
        super().__init__()

        self.budget_values = list(budget_values)
        self.num_budgets = len(self.budget_values)
        self.num_layers = num_layers
        self.d_choices = list(d_choices)

        # Phase 1 uses one global MLP-width router shared by every layer. This
        # keeps the elastic stack more uniform, while layer skipping remains
        # per-layer.
        self.router_d = nn.Sequential(
            nn.Linear(self.num_budgets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.d_choices)),
        )

        self.router_lambda = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_budgets, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            for _ in range(num_layers)
        ])
        self.register_buffer(
            "budget_values_tensor",
            torch.tensor(self.budget_values, dtype=torch.float32),
            persistent=False,
        )

    def budget_to_onehot(self, budget_idx, device=None):
        if device is None:
            device = budget_idx.device
        return F.one_hot(budget_idx, num_classes=self.num_budgets).float().to(device)

    def budget_to_conditioning(self, budget_idx=None, budget_value=None, device=None):
        if (budget_idx is None) == (budget_value is None):
            raise ValueError("Provide exactly one of budget_idx or budget_value")

        if budget_idx is not None:
            if not isinstance(budget_idx, torch.Tensor):
                budget_idx = torch.tensor(budget_idx, dtype=torch.long, device=device)
            else:
                budget_idx = budget_idx.to(device=device, dtype=torch.long)
            if budget_idx.dim() == 0:
                budget_idx = budget_idx.unsqueeze(0)
            return self.budget_to_onehot(budget_idx, device=device)

        if not isinstance(budget_value, torch.Tensor):
            budget_value = torch.tensor(budget_value, dtype=torch.float32, device=device)
        else:
            budget_value = budget_value.to(device=device, dtype=torch.float32)
        if budget_value.dim() == 0:
            budget_value = budget_value.unsqueeze(0)

        anchor_values = self.budget_values_tensor.to(device=budget_value.device)
        min_anchor = float(anchor_values[0].item())
        max_anchor = float(anchor_values[-1].item())
        if torch.any(budget_value < min_anchor) or torch.any(budget_value > max_anchor):
            raise ValueError(
                f"budget_value must lie within trained anchor range [{min_anchor}, {max_anchor}]"
            )

        h = torch.zeros(
            (budget_value.shape[0], self.num_budgets),
            device=budget_value.device,
            dtype=torch.float32,
        )

        for row_idx, value in enumerate(budget_value):
            right_idx = int(torch.searchsorted(anchor_values, value, right=False).item())
            if right_idx == 0:
                h[row_idx, 0] = 1.0
                continue
            if right_idx == self.num_budgets:
                h[row_idx, -1] = 1.0
                continue

            right_anchor = anchor_values[right_idx]
            if torch.isclose(value, right_anchor):
                h[row_idx, right_idx] = 1.0
                continue

            left_idx = right_idx - 1
            left_anchor = anchor_values[left_idx]
            alpha = (value - left_anchor) / (right_anchor - left_anchor)
            h[row_idx, left_idx] = float(1.0 - alpha.item())
            h[row_idx, right_idx] = float(alpha.item())
        return h

    def forward(self, budget_idx=None, budget_value=None, device=None):
        h = self.budget_to_conditioning(
            budget_idx=budget_idx,
            budget_value=budget_value,
            device=device,
        )

        global_d_logits = self.router_d(h)
        d_logits = global_d_logits.unsqueeze(1).expand(-1, self.num_layers, -1)
        layer_keep_logits = torch.stack([router(h) for router in self.router_lambda], dim=1)

        return {
            "h": h,
            "d_logits": d_logits,
            "layer_keep_logits": layer_keep_logits,
        }
