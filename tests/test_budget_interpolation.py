import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from elastic_modeling.elastic_qwen import ElasticQwen2ForCausalLM
from elastic_modeling.router import BudgetRouter


class BudgetRouterInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.router = BudgetRouter(
            budget_values=[0.25, 0.5, 0.75, 1.0],
            num_layers=2,
            d_choices=[8, 16],
            hidden_dim=8,
        )

    def test_anchor_budget_value_maps_to_one_hot(self):
        h = self.router.budget_to_conditioning(budget_value=torch.tensor([0.75]))
        expected = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        self.assertTrue(torch.allclose(h, expected))

    def test_unseen_budget_value_interpolates_between_neighbors(self):
        h = self.router.budget_to_conditioning(budget_value=torch.tensor([0.875]))
        expected = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        self.assertTrue(torch.allclose(h, expected, atol=1e-6, rtol=1e-6))

    def test_router_forward_exposes_interpolated_conditioning(self):
        out = self.router(budget_value=torch.tensor([0.8]))
        expected = torch.tensor([[0.0, 0.0, 0.8, 0.2]])
        self.assertTrue(torch.allclose(out["h"], expected, atol=1e-6, rtol=1e-6))

    def test_out_of_range_budget_value_raises(self):
        with self.assertRaises(ValueError):
            self.router.budget_to_conditioning(budget_value=torch.tensor([1.1]))


class ElasticQwenBudgetValueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        self.base_model = Qwen2ForCausalLM(self.config).eval()
        self.model = ElasticQwen2ForCausalLM(
            base_causallm=self.base_model,
            d_choices=[8, 16, 32],
            router=None,
            budget_values=[0.25, 0.5, 0.75, 1.0],
            enable_policy_modulation=True,
            policy_modulation_embed_dim=8,
            policy_modulation_hidden_dim=16,
        ).eval()
        self.input_ids = torch.randint(0, self.config.vocab_size, (1, 6), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = self.input_ids.clone()
        self.layer_keep = torch.tensor([[True, True]])
        self.d_keep = torch.tensor([[8, 8]])

    def test_forward_accepts_scalar_budget_value(self):
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            budget_value=torch.tensor([0.875], dtype=torch.float32),
            d_keep=self.d_keep,
            layer_keep=self.layer_keep,
        )
        self.assertTrue(torch.isfinite(outputs.loss))
        self.assertEqual(outputs.logits.shape[:2], self.input_ids.shape)


if __name__ == "__main__":
    unittest.main()
