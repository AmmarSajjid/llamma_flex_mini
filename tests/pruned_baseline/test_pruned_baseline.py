import unittest
from types import SimpleNamespace

from elastic_modeling.pruned_baseline.pruned_baseline_utils import resolve_pruned_width_for_budget


class PrunedBaselineBudgetResolutionTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
        )
        self.components = {
            "accounting_mode": "block_params",
            "hidden_size": 896.0,
            "intermediate_size": 4864.0,
            "num_layers": 24.0,
            "fixed_block_params": 1837952.0,
            "mlp_full_params": 13074432.0,
            "full_layer_params": 14912384.0,
            "full_elastic_params": 357897216.0,
        }

    def test_full_budget_resolves_to_full_width(self):
        resolved = resolve_pruned_width_for_budget(
            config=self.config,
            layer=None,
            target_budget=1.0,
            width_granularity=1,
            parameter_count_components=self.components,
        )

        self.assertEqual(resolved["resolved_d_keep"], 4864)
        self.assertAlmostEqual(resolved["achieved_budget"], 1.0)

    def test_target_budgets_resolve_monotonically(self):
        quarter = resolve_pruned_width_for_budget(
            config=self.config,
            layer=None,
            target_budget=0.25,
            width_granularity=1,
            parameter_count_components=self.components,
        )
        half = resolve_pruned_width_for_budget(
            config=self.config,
            layer=None,
            target_budget=0.5,
            width_granularity=1,
            parameter_count_components=self.components,
        )
        three_quarters = resolve_pruned_width_for_budget(
            config=self.config,
            layer=None,
            target_budget=0.75,
            width_granularity=1,
            parameter_count_components=self.components,
        )

        self.assertLess(quarter["resolved_d_keep"], half["resolved_d_keep"])
        self.assertLess(half["resolved_d_keep"], three_quarters["resolved_d_keep"])
        self.assertAlmostEqual(quarter["achieved_budget"], 0.25, places=4)
        self.assertAlmostEqual(half["achieved_budget"], 0.5, places=4)
        self.assertAlmostEqual(three_quarters["achieved_budget"], 0.75, places=4)

    def test_width_granularity_rounds_but_stays_close(self):
        resolved = resolve_pruned_width_for_budget(
            config=self.config,
            layer=None,
            target_budget=0.75,
            width_granularity=64,
            parameter_count_components=self.components,
        )

        self.assertEqual(resolved["resolved_d_keep"] % 64, 0)
        self.assertAlmostEqual(resolved["achieved_budget"], 0.75, places=2)

    def test_too_small_budget_raises(self):
        with self.assertRaises(ValueError):
            resolve_pruned_width_for_budget(
                config=self.config,
                layer=None,
                target_budget=0.10,
                width_granularity=1,
                parameter_count_components=self.components,
            )


if __name__ == "__main__":
    unittest.main()
