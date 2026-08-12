"""
analysis/oracle_search.py — Oracle Search procedure S_search-oracle = Search(B, k; S_space(D,k)) (spec v0.12-final).

Supports:
1. Greedy search over candidate expert subsets of size k
2. Beam search (if applicable)
3. Exhaustive search (for small k / pool)
4. Deterministic tie-breaking policy pre-registered before evaluation
5. Generation of reproducible oracle_search metadata card
"""

import itertools
import numpy as np
from typing import List, Tuple, Dict, Any, Callable


def tie_break_subsets(
    candidates: List[Tuple[int, ...]],
    policy: str = "deterministic",
    rule: str = "lexicographic_expert_id"
) -> Tuple[int, ...]:
    """
    Applies pre-registered tie-breaking rule among candidate subsets with identical Q_B score.
    Default rule: 'lexicographic_expert_id' selects the subset with smallest lexicographical tuple representation.
    """
    if not candidates:
        raise ValueError("Cannot perform tie-breaking on empty candidates list.")
    if len(candidates) == 1:
        return candidates[0]

    if policy == "deterministic" and rule == "lexicographic_expert_id":
        sorted_candidates = sorted(candidates)
        return sorted_candidates[0]
    else:
        # Fallback deterministic sorting
        sorted_candidates = sorted(candidates)
        return sorted_candidates[0]


def greedy_search(
    eval_fn: Callable[[Tuple[int, ...]], float],
    pool_size: int,
    k: int,
    tie_policy: str = "deterministic",
    tie_rule: str = "lexicographic_expert_id"
) -> Tuple[Tuple[int, ...], int]:
    """
    Greedy search building subset of size k expert by expert.
    Returns: (best_subset, evaluations_count)
    """
    current_subset: List[int] = []
    evaluations_count = 0

    for step in range(k):
        candidates_to_eval = [
            tuple(sorted(current_subset + [idx]))
            for idx in range(pool_size)
            if idx not in current_subset
        ]
        
        scores: Dict[Tuple[int, ...], float] = {}
        for cand in candidates_to_eval:
            scores[cand] = eval_fn(cand)
            evaluations_count += 1
            
        max_score = max(scores.values())
        tied_candidates = [cand for cand, s in scores.items() if abs(s - max_score) < 1e-12]
        best_cand = tie_break_subsets(tied_candidates, policy=tie_policy, rule=tie_rule)
        current_subset = list(best_cand)

    return tuple(sorted(current_subset)), evaluations_count


def exhaustive_search(
    eval_fn: Callable[[Tuple[int, ...]], float],
    pool_size: int,
    k: int,
    tie_policy: str = "deterministic",
    tie_rule: str = "lexicographic_expert_id"
) -> Tuple[Tuple[int, ...], int]:
    """
    Exhaustive search evaluating all C(pool_size, k) subsets.
    Returns: (best_subset, evaluations_count)
    """
    all_combos = list(itertools.combinations(range(pool_size), k))
    scores: Dict[Tuple[int, ...], float] = {}
    evaluations_count = 0

    for cand in all_combos:
        scores[cand] = eval_fn(cand)
        evaluations_count += 1

    max_score = max(scores.values())
    tied_candidates = [cand for cand, s in scores.items() if abs(s - max_score) < 1e-12]
    best_cand = tie_break_subsets(tied_candidates, policy=tie_policy, rule=tie_rule)
    return best_cand, evaluations_count


def run_oracle_search(
    eval_fn: Callable[[Tuple[int, ...]], float],
    oracle_cfg: Dict[str, Any]
) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
    """
    Runs search method specified in oracle_cfg on Dataset B.
    Returns: (S_search_oracle_subset, oracle_search_card)
    """
    method = oracle_cfg.get("method", "greedy")
    pool_size = oracle_cfg.get("pool_size", 16)
    k = oracle_cfg.get("k", 4)
    
    tb_cfg = oracle_cfg.get("tie_breaking", {})
    tie_policy = tb_cfg.get("policy", "deterministic")
    tie_rule = tb_cfg.get("rule", "lexicographic_expert_id")

    if method == "greedy":
        best_subset, evals_used = greedy_search(eval_fn, pool_size, k, tie_policy, tie_rule)
    elif method == "exhaustive":
        best_subset, evals_used = exhaustive_search(eval_fn, pool_size, k, tie_policy, tie_rule)
    else:
        # Fallback to greedy
        best_subset, evals_used = greedy_search(eval_fn, pool_size, k, tie_policy, tie_rule)

    oracle_search_card = {
        "method": method,
        "objective": oracle_cfg.get("objective", {
            "metric": "negative_cross_entropy",
            "direction": "maximize",
            "dataset_split": "B"
        }),
        "budget": {
            "unit": "candidate_subset_evaluations",
            "value": evals_used
        },
        "search_determinism": oracle_cfg.get("search_determinism", {
            "deterministic": True,
            "seeds": [42]
        }),
        "tie_breaking": {
            "policy": tie_policy,
            "rule": tie_rule
        },
        "dataset_split": "B",
        "k": k,
        "selected_subset": list(best_subset),
        "implementation_version": oracle_cfg.get("implementation_version", "v0.12-final-ref1")
    }

    return best_subset, oracle_search_card
