"""
tests/test_random_reference.py — Unit tests for random reference sampling and frozen card.
"""

import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.random_reference import sample_random_reference_subsets



class TestRandomReference(unittest.TestCase):

    def test_random_reference_sampling(self):
        frozen_subsets, card = sample_random_reference_subsets(
            pool_size=16,
            k=4,
            N_random=10,
            seeds=[100 + i for i in range(10)]
        )

        self.assertEqual(len(frozen_subsets), 10)
        for s in frozen_subsets:
            self.assertEqual(len(s), 4)
            self.assertEqual(len(set(s)), 4)  # Unique experts within subset

        self.assertTrue(card["subsets_frozen_for_evaluation"])
        self.assertFalse(card["resample_random_subsets_in_bootstrap"])

    def test_random_reference_min_n_error(self):
        with self.assertRaises(ValueError):
            sample_random_reference_subsets(pool_size=16, k=4, N_random=4)


if __name__ == "__main__":
    unittest.main()

