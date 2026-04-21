import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from elastic_modeling.budget_loss import (
    BLOCK_PARAMS_ACCOUNTING,
    MLP_ONLY_ACCOUNTING,
    build_parameter_count_components,
    compute_budget_loss,
)


class WeightOnlyNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class DummySelfAttention(nn.Module):
    def __init__(self, hidden_size, kv_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class DummyMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class DummyLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, kv_size):
        super().__init__()
        self.self_attn = DummySelfAttention(hidden_size, kv_size)
        self.mlp = DummyMLP(hidden_size, intermediate_size)
        self.input_layernorm = WeightOnlyNorm(hidden_size)
        self.post_attention_layernorm = WeightOnlyNorm(hidden_size)


class BudgetLossAccountingTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
        )
        self.layer = DummyLayer(hidden_size=4, intermediate_size=8, kv_size=2)
        self.d_choices = [4, 8]
        self.mlp_components = build_parameter_count_components(
            config=self.config,
            layer=self.layer,
            accounting_mode=MLP_ONLY_ACCOUNTING,
        )
        self.block_components = build_parameter_count_components(
            config=self.config,
            layer=self.layer,
            accounting_mode=BLOCK_PARAMS_ACCOUNTING,
        )

    def test_component_counts_match_manual_values(self):
        self.assertEqual(self.mlp_components["mlp_full_params"], 96.0)
        self.assertEqual(self.mlp_components["full_elastic_params"], 192.0)
        self.assertEqual(self.block_components["fixed_block_params"], 64.0)
        self.assertEqual(self.block_components["full_layer_params"], 160.0)
        self.assertEqual(self.block_components["full_elastic_params"], 320.0)

    def test_width_only_mode_preserves_mlp_only_budgeting(self):
        stats = compute_budget_loss(
            config=self.config,
            target_budget=torch.tensor([0.5]),
            d_choices=self.d_choices,
            layer_keep=torch.tensor([[True, False]]),
            d_keep=torch.tensor([[4, 4]]),
            accounting_mode=MLP_ONLY_ACCOUNTING,
            parameter_count_components=self.mlp_components,
        )

        self.assertEqual(stats["accounting_mode"], MLP_ONLY_ACCOUNTING)
        self.assertAlmostEqual(stats["expected_params"].item(), 96.0)
        self.assertAlmostEqual(stats["full_params"].item(), 192.0)
        self.assertAlmostEqual(stats["achieved_budget"].item(), 0.5)
        self.assertAlmostEqual(stats["loss"].item(), 0.0)

    def test_block_mode_full_width_all_layers_kept_is_full_budget(self):
        stats = compute_budget_loss(
            config=self.config,
            target_budget=torch.tensor([1.0]),
            d_choices=self.d_choices,
            layer_keep=torch.tensor([[True, True]]),
            d_keep=torch.tensor([[8, 8]]),
            accounting_mode=BLOCK_PARAMS_ACCOUNTING,
            parameter_count_components=self.block_components,
        )

        self.assertAlmostEqual(stats["expected_params"].item(), 320.0)
        self.assertAlmostEqual(stats["full_params"].item(), 320.0)
        self.assertAlmostEqual(stats["achieved_budget"].item(), 1.0)
        self.assertAlmostEqual(stats["loss"].item(), 0.0)

    def test_block_mode_all_layers_skipped_is_zero_budget(self):
        stats = compute_budget_loss(
            config=self.config,
            target_budget=torch.tensor([0.0]),
            d_choices=self.d_choices,
            layer_keep=torch.tensor([[False, False]]),
            d_keep=torch.tensor([[8, 8]]),
            accounting_mode=BLOCK_PARAMS_ACCOUNTING,
            parameter_count_components=self.block_components,
        )

        self.assertAlmostEqual(stats["expected_params"].item(), 0.0)
        self.assertAlmostEqual(stats["full_params"].item(), 320.0)
        self.assertAlmostEqual(stats["achieved_budget"].item(), 0.0)
        self.assertAlmostEqual(stats["loss"].item(), 0.0)

    def test_block_mode_concrete_controls_match_manual_count(self):
        stats = compute_budget_loss(
            config=self.config,
            target_budget=torch.tensor([0.35]),
            d_choices=self.d_choices,
            layer_keep=torch.tensor([[True, False]]),
            d_keep=torch.tensor([[4, 8]]),
            accounting_mode=BLOCK_PARAMS_ACCOUNTING,
            parameter_count_components=self.block_components,
        )

        self.assertAlmostEqual(stats["expected_params"].item(), 112.0)
        self.assertAlmostEqual(stats["full_params"].item(), 320.0)
        self.assertAlmostEqual(stats["achieved_budget"].item(), 0.35)
        self.assertAlmostEqual(stats["loss"].item(), 0.0)

    def test_block_mode_soft_controls_match_manual_expectation(self):
        stats = compute_budget_loss(
            config=self.config,
            target_budget=torch.tensor([118.0 / 320.0]),
            d_choices=self.d_choices,
            layer_keep_probs=torch.tensor([[[0.25, 0.75], [0.75, 0.25]]], dtype=torch.float32),
            d_probs=torch.tensor([[[1.0, 0.0], [0.5, 0.5]]], dtype=torch.float32),
            accounting_mode=BLOCK_PARAMS_ACCOUNTING,
            parameter_count_components=self.block_components,
        )

        self.assertAlmostEqual(stats["expected_params"].item(), 118.0)
        self.assertAlmostEqual(stats["full_params"].item(), 320.0)
        self.assertAlmostEqual(stats["achieved_budget"].item(), 118.0 / 320.0)
        self.assertAlmostEqual(stats["loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
