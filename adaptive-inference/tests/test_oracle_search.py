"""
tests/test_oracle_search.py — Unit tests for Oracle Search and deterministic tie-breaking.
"""

import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.oracle_search import run_oracle_search, tie_break_subsets



class TestOracleSearch(unittest.TestCase):

    def test_deterministic_tie_breaking(self):
        # Two tied candidates (1, 3) and (0, 4)
        candidates = [(1, 3), (0, 4)]
        chosen = tie_break_subsets(candidates, policy="deterministic", rule="lexicographic_expert_id")
        self.assertEqual(chosen, (0, 4))  # Lexicographically smaller

    def test_greedy_oracle_search(self):
        # Mock eval_fn where expert 2 and 5 are best
        def eval_fn(subset):
            val = 0.0
            if 2 in subset:
                val += 10.0
            if 5 in subset:
                val += 5.0
            return val

        oracle_cfg = {
            "method": "greedy",
            "pool_size": 8,
            "k": 2,
            "tie_breaking": {"policy": "deterministic", "rule": "lexicographic_expert_id"},
            "search_determinism": {"deterministic": True, "seeds": [42]}
        }

        best_subset, card = run_oracle_search(eval_fn, oracle_cfg)
        self.assertEqual(set(best_subset), {2, 5})
        self.assertEqual(card["method"], "greedy")
        self.assertGreater(card["budget"]["value"], 0)


if __name__ == "__main__":
    unittest.main()

