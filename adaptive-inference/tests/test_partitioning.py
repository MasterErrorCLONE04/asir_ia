"""
tests/test_partitioning.py — Unit tests for dataset example-level split and group deduplication.
"""

import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.partitioning import partition_dataset_abc, compute_example_group_id



class TestPartitioning(unittest.TestCase):

    def test_group_level_split_disjoint(self):
        # 30 examples with 10 groups of 3
        examples = []
        for i in range(30):
            examples.append({
                "id": f"ex_{i}",
                "source_id": f"src_{i // 3}",
                "content": f"Content {i}"
            })

        grouping_cfg = {
            "method": "source_id",
            "implementation_version": "v1.0",
            "threshold": None,
            "seed": 42
        }

        a, b, c, card = partition_dataset_abc(examples, grouping_cfg, ratios=(0.4, 0.3, 0.3))

        g_a = set(compute_example_group_id(ex, "source_id") for ex in a)
        g_b = set(compute_example_group_id(ex, "source_id") for ex in b)
        g_c = set(compute_example_group_id(ex, "source_id") for ex in c)

        # Disjoint groups
        self.assertEqual(len(g_a.intersection(g_b)), 0)
        self.assertEqual(len(g_a.intersection(g_c)), 0)
        self.assertEqual(len(g_b.intersection(g_c)), 0)

        # Total counts match
        self.assertEqual(len(a) + len(b) + len(c), 30)
        self.assertEqual(card["total_groups"], 10)


if __name__ == "__main__":
    unittest.main()

