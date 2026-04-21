import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from elastic_modeling.elastic_qwen import ElasticQwen2ForCausalLM
from elastic_modeling.eval_router import load_router_from_checkpoint
from elastic_modeling.policy_modulation import PolicyAwareModulator
from elastic_modeling.router import BudgetRouter


class PolicyAwareModulatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.modulator = PolicyAwareModulator(
            hidden_size=4,
            d_choices=[4, 8],
            intermediate_size=8,
            embed_dim=8,
            hidden_dim=12,
        )

    def _set_non_identity_weights(self):
        with torch.no_grad():
            self.modulator.fc1.weight.fill_(0.1)
            self.modulator.fc1.bias.fill_(0.05)
            self.modulator.fc2.weight.fill_(0.02)
            self.modulator.fc2.bias.fill_(0.01)

    def test_zero_initialized_modulation_is_identity(self):
        y = torch.randn(2, 3, 4)
        out = self.modulator(y, budget_value=0.5, d_keep=4)
        self.assertTrue(torch.allclose(out, y))

    def test_hard_and_one_hot_soft_width_paths_match(self):
        self._set_non_identity_weights()
        y = torch.randn(2, 3, 4)
        hard_out = self.modulator(y, budget_value=0.5, d_keep=4)
        soft_out = self.modulator(y, budget_value=0.5, d_probs=torch.tensor([1.0, 0.0]))
        self.assertTrue(torch.allclose(hard_out, soft_out, atol=1e-6, rtol=1e-6))

    def test_budget_changes_conditioning_when_width_is_fixed(self):
        cond_low = self.modulator.conditioning_embedding(budget_value=0.25, d_keep=4)
        cond_high = self.modulator.conditioning_embedding(budget_value=0.75, d_keep=4)
        self.assertFalse(torch.allclose(cond_low, cond_high))

    def test_width_changes_conditioning_when_budget_is_fixed(self):
        cond_small = self.modulator.conditioning_embedding(budget_value=0.5, d_keep=4)
        cond_large = self.modulator.conditioning_embedding(budget_value=0.5, d_keep=8)
        self.assertFalse(torch.allclose(cond_small, cond_large))


class ElasticQwenPolicyModulationIntegrationTests(unittest.TestCase):
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
        self.budget_values = [0.25, 0.5, 1.0]
        self.d_choices = [8, 16, 32]
        self.router = BudgetRouter(
            budget_values=self.budget_values,
            num_layers=self.config.num_hidden_layers,
            d_choices=self.d_choices,
            hidden_dim=16,
        )
        self.input_ids = torch.randint(0, self.config.vocab_size, (1, 6), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = self.input_ids.clone()
        self.layer_keep_probs = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        self.d_probs = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        self.layer_keep = torch.tensor([[True, True]])
        self.d_keep = torch.tensor([[8, 8]])
        self.budget_idx = torch.tensor([1], dtype=torch.long)

    def _build_model(self, enable_policy_modulation):
        return ElasticQwen2ForCausalLM(
            base_causallm=self.base_model,
            d_choices=self.d_choices,
            router=self.router,
            budget_values=self.budget_values,
            enable_policy_modulation=enable_policy_modulation,
            policy_modulation_embed_dim=8,
            policy_modulation_hidden_dim=16,
        ).eval()

    def test_forward_accepts_budget_idx_with_soft_controls(self):
        model = self._build_model(enable_policy_modulation=True)
        outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            budget_idx=self.budget_idx,
            d_probs=self.d_probs,
            layer_keep_probs=self.layer_keep_probs,
        )
        self.assertTrue(torch.isfinite(outputs.loss))
        self.assertEqual(outputs.logits.shape[:2], self.input_ids.shape)

    def test_forward_accepts_budget_idx_with_hard_controls(self):
        model = self._build_model(enable_policy_modulation=True)
        outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            budget_idx=self.budget_idx,
            d_keep=self.d_keep,
            layer_keep=self.layer_keep,
        )
        self.assertTrue(torch.isfinite(outputs.loss))
        self.assertEqual(outputs.logits.shape[:2], self.input_ids.shape)

    def test_zero_initialized_modulation_matches_disabled_baseline(self):
        disabled_model = self._build_model(enable_policy_modulation=False)
        enabled_model = self._build_model(enable_policy_modulation=True)

        disabled_outputs = disabled_model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            budget_idx=self.budget_idx,
            d_keep=self.d_keep,
            layer_keep=self.layer_keep,
        )
        enabled_outputs = enabled_model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            budget_idx=self.budget_idx,
            d_keep=self.d_keep,
            layer_keep=self.layer_keep,
        )

        self.assertTrue(torch.allclose(disabled_outputs.logits, enabled_outputs.logits, atol=1e-6, rtol=1e-6))


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.budget_values = [0.25, 0.5, 1.0]
        self.d_choices = [8, 16, 32]
        self.router = BudgetRouter(
            budget_values=self.budget_values,
            num_layers=2,
            d_choices=self.d_choices,
            hidden_dim=8,
        )
        self.model_config = SimpleNamespace(num_hidden_layers=2)

    def test_old_checkpoint_defaults_policy_modulation_to_disabled(self):
        checkpoint = {
            "router_state_dict": self.router.state_dict(),
            "d_choices": self.d_choices,
            "budget_values": self.budget_values,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "old_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            loaded = load_router_from_checkpoint(str(checkpoint_path), self.model_config)

        self.assertFalse(loaded[5])
        self.assertEqual(loaded[6], 16)
        self.assertEqual(loaded[7], 128)

    def test_new_checkpoint_round_trips_policy_modulation_metadata(self):
        checkpoint = {
            "router_state_dict": self.router.state_dict(),
            "d_choices": self.d_choices,
            "budget_values": self.budget_values,
            "enable_policy_modulation": True,
            "policy_modulation_embed_dim": 12,
            "policy_modulation_hidden_dim": 48,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "new_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            loaded = load_router_from_checkpoint(str(checkpoint_path), self.model_config)

        self.assertTrue(loaded[5])
        self.assertEqual(loaded[6], 12)
        self.assertEqual(loaded[7], 48)


if __name__ == "__main__":
    unittest.main()
