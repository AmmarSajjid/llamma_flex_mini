import torch
import torch.nn as nn
import torch.nn.functional as F


class BudgetRouter(nn.Module):
    def __init__(self, budget_values, num_layers, d_choices, hidden_dim=128):
        # Budget values: list of possible budgets for whole model(e.g., [0.25, 0.5, 0.75, 1.0])
        # d_choices: list of possible MLP widths (e.g., [128, 256, 4864])
        super().__init__()

        self.budget_values = budget_values
        self.num_budgets = len(budget_values)
        self.num_layers = num_layers
        self.d_choices = d_choices

        # Router for MLP width D
        self.router_d = nn.Sequential(
            nn.Linear(self.num_budgets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(d_choices)),
        )

        # One router per layer for skip/keep
        self.router_lambda = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_budgets, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),   # [skip, keep]
            )
            for _ in range(num_layers)
        ])

    def budget_to_onehot(self, budget_idx, device):
        """
        budget_idx: tensor of shape [B] with integer budget ids
        returns: one-hot tensor [B, num_budgets]
        """
        h = F.one_hot(budget_idx, num_classes=self.num_budgets).float().to(device)
        return h

    def forward(self, budget_idx, device):
        """
        budget_idx: [B]
        """
        h = self.budget_to_onehot(budget_idx, device=device)   # [B, |B|]

        d_logits = self.router_d(h)   # [B, num_d_choices]
        lambda_logits = [router(h) for router in self.router_lambda]  # list of [B, 2]

        return {
            "h": h,
            "d_logits": d_logits,
            "lambda_logits": lambda_logits,
        }