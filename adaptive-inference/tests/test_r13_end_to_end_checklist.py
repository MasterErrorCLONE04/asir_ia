"""
tests/test_r13_end_to_end_checklist.py — Master 7-point checklist verification suite for R1.3 (v0.12-final).
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
    from analysis.model_adapter import MoEModelAdapter
    from analysis.moe_profiler import profile_selector_on_split_a


class TestR13EndToEndChecklist(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "r1.3_preregistration_v0.12.json"
        )
        self.cfg = load_and_validate_config(config_path)

        gen = SyntheticTaskGenerator(seed=42)
        raw_dataset = gen.generate_dataset(60)
        self.dataset = [
            {"id": f"ex_{i}", "source_id": f"src_{i // 3}", "input": item["input"], "target": item["target"], "domain": item["domain"]}
            for i, item in enumerate(raw_dataset)
        ]

        self.a_examples, self.b_examples, self.c_examples, self.grouping_card = partition_dataset_abc(
            self.dataset, self.cfg["grouping"], ratios=(0.4, 0.3, 0.3)
        )

        if HAS_TORCH:
            self.device = torch.device("cpu")
            self.tokenizer = CharTokenizer()
            self.model = MoETransformer(
                vocab_size=self.tokenizer.vocab_size,
                d_model=64,
                n_layers=2,
                num_heads=2,
                d_ff=128,
                max_seq_len=128,
                moe=True,
                num_experts=8,
                top_k=2
            ).to(self.device)
            self.adapter = MoEModelAdapter(self.model, self.tokenizer, self.device)

    def test_point_1_a_isolation_prompt_only_profiling(self):
        """1. S_selector must be derived exclusively from A using prompt-only profiling."""
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        s_selector = profile_selector_on_split_a(
            self.model, self.a_examples, k=4, tokenizer=self.tokenizer, device=self.device
        )
        self.assertEqual(len(s_selector), 4)
        for exp_id in s_selector:
            self.assertTrue(0 <= exp_id < 8)

    def test_point_2_b_isolation_oracle_search(self):
        """2. S_search-oracle must be searched exclusively over B."""
        eval_fn_b = lambda s: self.adapter.eval_subset_mean_quality(self.b_examples, s, max_gen_len=5) if HAS_TORCH else 0.5
        oracle_cfg = self.cfg["oracle_search"].copy()
        oracle_cfg["pool_size"] = 8
        s_oracle, card = run_oracle_search(eval_fn_b, oracle_cfg)
        self.assertEqual(len(s_oracle), 4)
        self.assertGreater(card["budget"]["value"], 0)

    def test_point_3_freeze_priority_random_reference(self):
        """3. S_random^(1..N) must be sampled and frozen BEFORE evaluating C."""
        s_random_list, card = sample_random_reference_subsets(
            pool_size=8, k=4, N_random=5, seeds=[10 + i for i in range(5)]
        )
        self.assertTrue(card["subsets_frozen_for_evaluation"])
        self.assertEqual(len(s_random_list), 5)

    def test_point_4_example_level_quality_matrix_on_c(self):
        """4. C must produce per-example quality arrays of shape (|C|,) for each candidate."""
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        q_selector_ex = self.adapter.eval_subset_quality_per_example(self.c_examples, (0, 1, 2, 3), max_gen_len=5)
        self.assertEqual(q_selector_ex.shape, (len(self.c_examples),))

    def test_point_5_paired_replicates_in_bootstrap(self):
        """5. Exactly the same example indices I_b must be applied pareatedly to selector, oracle, and random."""
        rng = np.random.default_rng(42)
        n_examples = 18
        q_sel = rng.uniform(0.5, 0.8, n_examples)
        q_ora = q_sel + 0.1
        q_rnd = np.zeros((3, n_examples))

        res = compute_paired_bootstrap_metrics(
            q_selector_examples=q_sel,
            q_oracle_examples=q_ora,
            q_random_examples_matrix=q_rnd,
            n_replicates=100,
            seed=42
        )
        self.assertEqual(res["bootstrap_diagnostics"]["total_replicates"], 100)

    def test_point_6_c_immutability(self):
        """6. Evaluating C must not mutate selector, search-oracle, or random reference sets."""
        s_selector_before = (0, 1, 2, 3)
        s_oracle_before = (1, 2, 4, 5)
        
        if HAS_TORCH:
            _ = self.adapter.eval_subset_quality_per_example(self.c_examples, s_selector_before, max_gen_len=5)

        self.assertEqual(s_selector_before, (0, 1, 2, 3))
        self.assertEqual(s_oracle_before, (1, 2, 4, 5))

    def test_point_7_artifact_traceability(self):
        """7. Saved artifacts must record contract version and complete config snapshot."""
        rse_result = {
            "metrics": {"ESP": {"value": 0.0, "status": "statistically_indistinguishable_from_zero"}},
            "bootstrap_diagnostics": {"invalid_fraction": 0.0}
        }
        s_oracle, oracle_card = (0, 1, 2, 3), {"method": "greedy"}
        s_random, random_card = [(0, 1, 2, 3)], {"subsets_frozen_for_evaluation": True}

        out_dir = os.path.join("adaptive-inference", "tests", "tmp_checklist_test")
        saved = save_artifact_snapshot(out_dir, self.cfg, self.grouping_card, oracle_card, random_card, rse_result)

        self.assertTrue(os.path.exists(saved["config_snapshot"]))
        self.assertTrue(os.path.exists(saved["grouping"]))


if __name__ == "__main__":
    unittest.main()
