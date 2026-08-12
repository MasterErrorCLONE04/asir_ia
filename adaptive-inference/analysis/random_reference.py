"""
analysis/random_reference.py — Random Reference Sampler & Artifact Freezer (spec v0.12-final).

Formalization:
  S_random^{(i)} ~ iid Uniform(S_space(D,k)),  i = 1, ..., N

Guarantees:
1. Samples N_random subsets independently and uniformly from S_space(D,k)
2. Freezes random_reference BEFORE evaluation on C
3. Disables resampling of random subsets during bootstrap evaluation
"""

import itertools
import numpy as np
from typing import List, Tuple, Dict, Any


def sample_random_reference_subsets(
    pool_size: int,
    k: int,
    N_random: int = 10,
    seeds: List[int] = None
) -> Tuple[List[Tuple[int, ...]], Dict[str, Any]]:
    """
    Samples N_random random subsets i.i.d. uniformly from S_space(D,k) and freezes them.
    Returns: (list_of_frozen_subsets, random_reference_card)
    """
    if N_random < 5:
        raise ValueError("N_random must be at least 5 for valid evaluation under spec v0.12-final.")

    all_combos = list(itertools.combinations(range(pool_size), k))
    n_combos = len(all_combos)

    if seeds is None or len(seeds) < N_random:
        seeds = [42 + i for i in range(N_random)]

    frozen_subsets = []
    for i in range(N_random):
        seed = seeds[i]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_combos)
        frozen_subsets.append(all_combos[idx])

    random_reference_card = {
        "sampling_distribution": "uniform_over_feasible_subsets",
        "N_random": N_random,
        "random_seeds": seeds[:N_random],
        "subsets_frozen_for_evaluation": True,
        "resample_random_subsets_in_bootstrap": False,
        "frozen_subsets": [list(s) for s in frozen_subsets]
    }

    return frozen_subsets, random_reference_card
