"""
tests/test_r13_real_moe_smoke.py — Nivel 3 Real PyTorch MoETransformer Smoke Test.
"""

import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.config_schema import load_and_validate_config
from analysis.partitioning import partition_dataset_abc
from analysis.oracle_search import run_oracle_search
from analysis.random_reference import sample_random_reference_subsets
from analysis.evaluator import compute_paired_bootstrap_metrics
from analysis.artifacts import save_artifact_snapshot
from tasks.generator import SyntheticTaskGenerator

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from training.models.transformer import MoETransformer
    from tasks.tokenizer import CharTokenizer
    from analysis.model_adapter import MoEModelAdapter, create_expert_mask
    from analysis.moe_profiler import profile_selector_on_split_a


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in current environment")
class TestR13RealMoESmoke(unittest.TestCase):

    def test_real_moe_forward_pass_and_routing_mask(self):
        """Validates that real PyTorch forward pass with routing mask works on MoETransformer."""
        tokenizer = CharTokenizer()
        device = torch.device("cpu")
        model = MoETransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            num_heads=2,
            d_ff=128,
            max_seq_len=128,
            moe=True,
            num_experts=8,
            top_k=2
        ).to(device)

        adapter = MoEModelAdapter(model, tokenizer, device)

        gen = SyntheticTaskGenerator(seed=42)
        raw_samples = gen.generate_dataset(10)
        samples = [
            {"id": f"ex_{i}", "source_id": f"src_{i // 2}", "input": s["input"], "target": s["target"], "domain": s["domain"]}
            for i, s in enumerate(raw_samples)
        ]

        subset_S = (0, 2, 4, 6)
        # Default metric is negative_cross_entropy
        scores_ex = adapter.eval_subset_quality_per_example(samples, subset_S)
        self.assertEqual(len(scores_ex), 10)
        for val in scores_ex:
            self.assertTrue(val < 0.0)

        # Optional metric is exact_match
        scores_em = adapter.eval_subset_quality_per_example(samples, subset_S, metric_name="exact_match", max_gen_len=5)
        self.assertEqual(len(scores_em), 10)
        for val in scores_em:
            self.assertIn(val, [0.0, 1.0])

    def test_real_moe_full_r13_pipeline_execution(self):
        """Validates full R1.3 pipeline execution using real PyTorch MoETransformer forward passes."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "r1.3_preregistration_v0.12.json"
        )
        cfg = load_and_validate_config(config_path)

        tokenizer = CharTokenizer()
        device = torch.device("cpu")
        model = MoETransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            num_heads=2,
            d_ff=128,
            max_seq_len=128,
            moe=True,
            num_experts=8,
            top_k=2
        ).to(device)

        adapter = MoEModelAdapter(model, tokenizer, device)

        gen = SyntheticTaskGenerator(seed=123)
        raw_samples = gen.generate_dataset(30)
        dataset = [
            {"id": f"ex_{i}", "source_id": f"src_{i // 3}", "input": s["input"], "target": s["target"], "domain": s["domain"]}
            for i, s in enumerate(raw_samples)
        ]

        # 1. Split A/B/C
        a_ex, b_ex, c_ex, grouping_card = partition_dataset_abc(dataset, cfg["grouping"], ratios=(0.4, 0.3, 0.3))

        # 2. Selector Profiling on A (Prompt-Only)
        s_selector = profile_selector_on_split_a(model, a_ex, k=4, tokenizer=tokenizer, device=device)
        self.assertEqual(len(s_selector), 4)

        # 3. Oracle Search on B with real PyTorch evaluation
        eval_fn_b = lambda s: adapter.eval_subset_mean_quality(b_ex, s, max_gen_len=5)
        oracle_cfg = cfg["oracle_search"].copy()
        oracle_cfg["pool_size"] = 8
        s_oracle, oracle_card = run_oracle_search(eval_fn_b, oracle_cfg)
        self.assertEqual(len(s_oracle), 4)

        # 4. Random Reference sampling & freeze
        s_random_list, random_card = sample_random_reference_subsets(pool_size=8, k=4, N_random=5)

        # 5. Evaluation on C with PyTorch model
        q_selector_ex = adapter.eval_subset_quality_per_example(c_ex, s_selector, max_gen_len=5)
        q_oracle_ex = adapter.eval_subset_quality_per_example(c_ex, s_oracle, max_gen_len=5)

        q_random_ex_matrix = np.zeros((len(s_random_list), len(c_ex)))
        for j, s_rand in enumerate(s_random_list):
            q_random_ex_matrix[j] = adapter.eval_subset_quality_per_example(c_ex, s_rand, max_gen_len=5)

        res = compute_paired_bootstrap_metrics(
            q_selector_examples=q_selector_ex,
            q_oracle_examples=q_oracle_ex,
            q_random_examples_matrix=q_random_ex_matrix,
            n_replicates=100,
            seed=42
        )

        self.assertIn("ESP", res["metrics"])
        self.assertIn("RSE", res["metrics"])

        out_dir = os.path.join("adaptive-inference", "tests", "tmp_real_moe_smoke_test")
        saved = save_artifact_snapshot(out_dir, cfg, grouping_card, oracle_card, random_card, res)

        self.assertTrue(os.path.exists(saved["config_snapshot"]))
        self.assertTrue(os.path.exists(saved["grouping"]))
        self.assertTrue(os.path.exists(saved["rse_result"]))

    def test_prompt_only_selector_profiling_isolation(self):
        """Validates that selector profiling is strictly prompt-only and target changes do not affect S_selector."""
        tokenizer = CharTokenizer()
        device = torch.device("cpu")
        model = MoETransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            num_heads=2,
            d_ff=128,
            max_seq_len=128,
            moe=True,
            num_experts=8,
            top_k=2
        ).to(device)

        # Construct two sets of dataset samples with identical inputs/prompts but different targets
        samples_a = [
            {"input": "def hello():", "target": "print('hello')"},
            {"input": "def add(x, y):", "target": "return x + y"}
        ]
        samples_b = [
            {"input": "def hello():", "target": "return 'hello world'"},
            {"input": "def add(x, y):", "target": "sum = x + y; return sum"}
        ]

        s_selector_a = profile_selector_on_split_a(model, samples_a, k=3, tokenizer=tokenizer, device=device)
        s_selector_b = profile_selector_on_split_a(model, samples_b, k=3, tokenizer=tokenizer, device=device)

        self.assertEqual(s_selector_a, s_selector_b)
        self.assertEqual(len(s_selector_a), 3)


if __name__ == "__main__":
    unittest.main()
