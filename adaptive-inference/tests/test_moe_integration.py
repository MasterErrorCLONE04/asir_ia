"""
tests/test_moe_integration.py — Integration test for PyTorch MoETransformer inference & R1.3 evaluator.
"""

import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from training.models.transformer import MoETransformer
    from tasks.tokenizer import CharTokenizer
    from tasks.generator import SyntheticTaskGenerator
    from analysis.model_adapter import MoEModelAdapter, create_expert_mask, load_moe_model_from_checkpoint
    from analysis.moe_profiler import profile_selector_on_split_a
    from analysis.evaluator import compute_paired_bootstrap_metrics



@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in current environment")
class TestMoEIntegration(unittest.TestCase):


    def setUp(self):
        self.device = torch.device("cpu")
        self.tokenizer = CharTokenizer()
        self.model = MoETransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=64,       # Small model for test speed
            n_layers=2,
            num_heads=2,
            d_ff=128,
            max_seq_len=128,
            moe=True,
            num_experts=8,
            top_k=2
        ).to(self.device)

        gen = SyntheticTaskGenerator(seed=42)
        raw_dataset = gen.generate_dataset(15)
        self.samples = [
            {"id": f"ex_{i}", "source_id": f"src_{i // 3}", "input": item["input"], "target": item["target"], "domain": item["domain"]}
            for i, item in enumerate(raw_dataset)
        ]

    def test_expert_mask_creation(self):
        mask = create_expert_mask((0, 2, 5), num_experts=8, device=self.device)
        self.assertEqual(mask.shape, (1, 8))
        self.assertEqual(mask[0, 0].item(), 1.0)
        self.assertEqual(mask[0, 1].item(), 0.0)
        self.assertEqual(mask[0, 2].item(), 1.0)
        self.assertEqual(mask[0, 5].item(), 1.0)

    def test_adapter_quality_evaluation(self):
        adapter = MoEModelAdapter(self.model, self.tokenizer, self.device)
        subset = (0, 1, 2, 3)

        # Test default negative_cross_entropy metric
        scores_nce = adapter.eval_subset_quality_per_example(self.samples, subset)
        self.assertEqual(len(scores_nce), 15)
        self.assertTrue(isinstance(scores_nce, np.ndarray))
        for val in scores_nce:
            self.assertTrue(val < 0.0)

        # Test optional exact_match metric
        scores_em = adapter.eval_subset_quality_per_example(self.samples, subset, metric_name="exact_match", max_gen_len=10)
        self.assertEqual(len(scores_em), 15)
        self.assertTrue(isinstance(scores_em, np.ndarray))
        for val in scores_em:
            self.assertIn(val, [0.0, 1.0])

    def test_moe_profiler_selector_selection(self):
        s_selector = profile_selector_on_split_a(self.model, self.samples, k=4, tokenizer=self.tokenizer, device=self.device)
        self.assertEqual(len(s_selector), 4)
        for exp_idx in s_selector:
            self.assertTrue(0 <= exp_idx < 8)

    def test_end_to_end_integration_paired_bootstrap(self):
        adapter = MoEModelAdapter(self.model, self.tokenizer, self.device)
        s_selector = (0, 1, 2, 3)
        s_oracle = (1, 2, 4, 5)
        s_random_list = [(0, 2, 4, 6), (1, 3, 5, 7), (2, 3, 6, 7)]

        q_selector_ex = adapter.eval_subset_quality_per_example(self.samples, s_selector, max_gen_len=5)
        q_oracle_ex = adapter.eval_subset_quality_per_example(self.samples, s_oracle, max_gen_len=5)
        
        q_random_ex_matrix = np.zeros((3, len(self.samples)))
        for j, s_rand in enumerate(s_random_list):
            q_random_ex_matrix[j] = adapter.eval_subset_quality_per_example(self.samples, s_rand, max_gen_len=5)

        res = compute_paired_bootstrap_metrics(
            q_selector_examples=q_selector_ex,
            q_oracle_examples=q_oracle_ex,
            q_random_examples_matrix=q_random_ex_matrix,
            n_replicates=100,
            confidence_level=0.95,
            seed=42
        )

        self.assertIn("ESP", res["metrics"])
        self.assertIn("RSE", res["metrics"])
        self.assertEqual(res["bootstrap_diagnostics"]["total_replicates"], 100)


if __name__ == "__main__":
    unittest.main()
