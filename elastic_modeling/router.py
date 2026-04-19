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

    def budget_to_onehot(self, budget_idx, device=None):
        if device is None:
            device = budget_idx.device
        return F.one_hot(budget_idx, num_classes=self.num_budgets).float().to(device)

    def forward(self, budget_idx, device=None):
        if budget_idx.dim() == 0:
            budget_idx = budget_idx.unsqueeze(0)

        h = self.budget_to_onehot(budget_idx, device=device)

        global_d_logits = self.router_d(h)
        d_logits = global_d_logits.unsqueeze(1).expand(-1, self.num_layers, -1)
        layer_keep_logits = torch.stack([router(h) for router in self.router_lambda], dim=1)

        return {
            "h": h,
            "d_logits": d_logits,
            "layer_keep_logits": layer_keep_logits,
        }
