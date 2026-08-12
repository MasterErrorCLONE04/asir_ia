"""
tests/test_evaluator.py — Unit tests for Paired Example Bootstrap and RSE/ESP evaluator.
"""

import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.evaluator import compute_paired_bootstrap_metrics



class TestEvaluator(unittest.TestCase):

    def test_paired_bootstrap_evaluator_valid_case(self):
        n_examples = 100
        n_random = 5
        rng = np.random.default_rng(42)

        # Oracle is highest, selector is close, random is lowest
        q_oracle = rng.normal(loc=10.0, scale=0.5, size=n_examples)
        q_selector = q_oracle - rng.normal(loc=1.0, scale=0.2, size=n_examples) # ESP ≈ -1.0
        
        q_random_matrix = np.zeros((n_random, n_examples))
        for j in range(n_random):
            q_random_matrix[j] = rng.normal(loc=2.0, scale=0.5, size=n_examples) # Δ_SR ≈ 8.0

        res = compute_paired_bootstrap_metrics(
            q_selector_examples=q_selector,
            q_oracle_examples=q_oracle,
            q_random_examples_matrix=q_random_matrix,
            n_replicates=1000,
            confidence_level=0.95,
            seed=42
        )

        self.assertEqual(res["metrics"]["denominator"]["status"], "interpretable")
        self.assertEqual(res["metrics"]["RSE"]["ratio_status"], "estimable")
        self.assertIsNotNone(res["metrics"]["RSE"]["value"])
        self.assertTrue(0.75 <= res["metrics"]["RSE"]["value"] <= 0.95)
        self.assertEqual(res["bootstrap_diagnostics"]["invalid_denominator_replicates"], 0)
        self.assertEqual(res["bootstrap_diagnostics"]["invalid_fraction"], 0.0)

    def test_no_posthoc_exclusion_invalid_denominator_replicates(self):
        n_examples = 50
        n_random = 5
        rng = np.random.default_rng(123)

        # Oracle and random are almost equal, causing some bootstrap replicates where Δ <= 0
        q_oracle = rng.normal(loc=5.0, scale=2.0, size=n_examples)
        q_selector = q_oracle - 0.2
        
        q_random_matrix = np.zeros((n_random, n_examples))
        for j in range(n_random):
            q_random_matrix[j] = rng.normal(loc=4.9, scale=2.0, size=n_examples)

        res = compute_paired_bootstrap_metrics(
            q_selector_examples=q_selector,
            q_oracle_examples=q_oracle,
            q_random_examples_matrix=q_random_matrix,
            n_replicates=1000,
            confidence_level=0.95,
            seed=42
        )

        # Must result in ratio_status = non_estimable and RSE = null if any replicate fails or denominator <= 0
        if res["bootstrap_diagnostics"]["invalid_denominator_replicates"] > 0:
            self.assertEqual(res["metrics"]["RSE"]["ratio_status"], "non_estimable")
            self.assertIsNone(res["metrics"]["RSE"]["value"])
            self.assertIsNone(res["metrics"]["RSE"]["ci95"])
            self.assertFalse(res["bootstrap_diagnostics"]["posthoc_exclusion"])


if __name__ == "__main__":
    unittest.main()

