"""
analysis/partitioning.py — Example-level dataset partitioning with group-level deduplication (spec v0.12-final).

Guarantees:
1. Disjoint example-level split: A ∩ B = ∅, A ∩ C = ∅, B ∩ C = ∅
2. Group-level split: g(x_i) == g(x_j) => split(x_i) == split(x_j)
3. Reproducible grouping metadata card generation.
"""

import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple


def compute_example_group_id(example: Dict[str, Any], method: str = "exact_hash") -> str:
    """
    Computes group identifier g(x) for an example.
    Supported methods:
      - 'source_id': Uses example['source_id']
      - 'exact_hash': SHA-256 hash of example text/content
      - 'none': Unique identifier per example (no deduplication)
    """
    if method == "source_id":
        return str(example.get("source_id", example.get("id")))
    elif method == "exact_hash":
        content = example.get("content", str(example))
        return hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    elif method == "none":
        return str(example.get("id", hashlib.sha256(str(example).encode("utf-8")).hexdigest()))
    else:
        # Default fallback to string hash
        return hashlib.sha256(str(example).encode("utf-8")).hexdigest()


def partition_dataset_abc(
    examples: List[Dict[str, Any]],
    grouping_cfg: Dict[str, Any],
    ratios: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Partitions examples into A (profiling/selection), B (search-oracle), C (test).
    Grouping rule: all examples sharing g(x) belong to the EXACT SAME split.
    Returns: (A_examples, B_examples, C_examples, grouping_metadata_card)
    """
    method = grouping_cfg.get("method", "exact_hash")
    seed = grouping_cfg.get("seed", 42)
    impl_ver = grouping_cfg.get("implementation_version", "v1.0")

    # 1. Group examples by g(x)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        gid = compute_example_group_id(ex, method=method)
        groups.setdefault(gid, []).append(ex)

    group_ids = sorted(list(groups.keys()))

    # 2. Shuffle group IDs deterministically with seed
    rng = np.random.default_rng(seed)
    shuffled_gids = np.array(group_ids, dtype=object)
    rng.shuffle(shuffled_gids)

    # 3. Partition groups according to ratios
    n_groups = len(shuffled_gids)
    r_a, r_b, _ = ratios
    n_a = int(round(n_groups * r_a))
    n_b = int(round(n_groups * r_b))

    gids_a = set(shuffled_gids[:n_a])
    gids_b = set(shuffled_gids[n_a : n_a + n_b])
    gids_c = set(shuffled_gids[n_a + n_b :])

    # 4. Collect examples
    a_examples = [ex for gid in gids_a for ex in groups[gid]]
    b_examples = [ex for gid in gids_b for ex in groups[gid]]
    c_examples = [ex for gid in gids_c for ex in groups[gid]]

    # 5. Build reproducible grouping card
    grouping_card = {
        "method": method,
        "implementation_version": impl_ver,
        "threshold": grouping_cfg.get("threshold"),
        "seed": seed,
        "total_examples": len(examples),
        "total_groups": n_groups,
        "split_counts": {
            "A_examples": len(a_examples),
            "B_examples": len(b_examples),
            "C_examples": len(c_examples),
            "A_groups": len(gids_a),
            "B_groups": len(gids_b),
            "C_groups": len(gids_c)
        }
    }

    return a_examples, b_examples, c_examples, grouping_card
