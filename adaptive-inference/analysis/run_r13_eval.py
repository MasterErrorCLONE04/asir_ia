"""
analysis/run_r13_eval.py — CLI Runner for R1.3 evaluation pipeline under spec v0.12-final.

Execution Order Enforced:
1. Load & validate pre-registered config
2. Near-duplicate grouping & example-level A/B/C partition
3. Selector profiling on A
4. Oracle Search Search(B,k) on B -> produce S_search-oracle
5. Random reference sampling S_random^(1..N) -> FREEZE ARTIFACTS
6. Evaluate C -> Paired example bootstrap -> ESP@k, RSE@k, bootstrap_diagnostics
7. Save 5 reproducible artifacts under results/<domain>/k_<k>/

Usage:
    python analysis/run_r13_eval.py --config configs/r1.3_preregistration_v0.12.yaml [--dry_run]
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple

# Ensure package imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.config_schema import load_and_validate_config
from analysis.partitioning import partition_dataset_abc
from analysis.oracle_search import run_oracle_search
from analysis.random_reference import sample_random_reference_subsets
from analysis.evaluator import compute_paired_bootstrap_metrics
from analysis.artifacts import save_artifact_snapshot


def generate_synthetic_dataset(n_examples: int = 500, pool_size: int = 16) -> List[Dict[str, Any]]:
    """
    Generates synthetic example evaluation scores for contract testing/dry runs.
    """
    rng = np.random.default_rng(42)
    examples = []
    for i in range(n_examples):
        # Create synthetic expert quality profile per example
        expert_scores = rng.normal(loc=0.5, scale=0.1, size=pool_size)
        examples.append({
            "id": f"ex_{i}",
            "source_id": f"src_{i // 3}", # Shares source_id among groups of 3
            "content": f"Synthetic example prompt content {i}",
            "expert_scores": expert_scores
        })
    return examples


def mock_eval_fn_dataset(examples: List[Dict[str, Any]], subset: Tuple[int, ...]) -> float:
    """
    Computes quality Q = -L (negative loss) of a given expert subset over a list of examples.
    """
    subset_arr = list(subset)
    scores = [np.max(ex["expert_scores"][subset_arr]) for ex in examples]
    return float(np.mean(scores))


def mock_selector_profile(a_examples: List[Dict[str, Any]], pool_size: int, k: int) -> Tuple[int, ...]:
    """
    Mock selector profiling on A returning chosen subset S_selector.
    """
    scores = np.zeros(pool_size)
    for ex in a_examples:
        scores += ex["expert_scores"]
    top_indices = np.argsort(scores)[::-1][:k]
    return tuple(sorted(list(top_indices)))


def main():
    parser = argparse.ArgumentParser(description="Executes R1.3 evaluation pipeline (v0.12-final).")
    parser.add_argument("--config", type=str, required=True, help="Path to pre-registered YAML config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output results directory.")
    parser.add_argument("--dry_run", action="store_true", help="Run with synthetic dataset for testing.")
    args = parser.parse_args()

    # 1. Load and validate config
    print("=" * 80)
    print("R1.3 EVALUATION PIPELINE (v0.12-final)")
    print("=" * 80)
    print(f"Loading config: {args.config}")
    cfg = load_and_validate_config(args.config)
    print("Config validation: PASS (v0.12-final compliant)")

    domain = cfg["experiment"]["domain"]
    k = cfg["candidate_space"]["subset_size_k"]
    pool_size = cfg["candidate_space"]["expert_pool_size"]

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("results", domain, f"k_{k}")

    # 2. Data loading & Near-duplicate grouping + A/B/C split
    print("\n[Step 1/6] Partitioning dataset A/B/C with group-level split...")
    dataset = generate_synthetic_dataset(n_examples=600, pool_size=pool_size)
    a_examples, b_examples, c_examples, grouping_card = partition_dataset_abc(
        dataset, cfg["grouping"], ratios=(0.4, 0.3, 0.3)
    )
    print(f"  Split counts: A={len(a_examples)}, B={len(b_examples)}, C={len(c_examples)}")
    print(f"  Group method: {grouping_card['method']}, Total groups: {grouping_card['total_groups']}")

    # 3. Selector profiling on A
    print("\n[Step 2/6] Profiling selector on Dataset A...")
    s_selector = mock_selector_profile(a_examples, pool_size, k)
    print(f"  S_selector: {s_selector}")

    # 4. Oracle Search Search(B, k) on Dataset B
    print("\n[Step 3/6] Running Oracle Search Search(B,k) on Dataset B...")
    eval_fn_b = lambda subset: mock_eval_fn_dataset(b_examples, subset)
    oracle_cfg = cfg["oracle_search"]
    oracle_cfg["pool_size"] = pool_size
    s_oracle, oracle_search_card = run_oracle_search(eval_fn_b, oracle_cfg)
    print(f"  S_search-oracle: {s_oracle}")
    print(f"  Oracle evaluations used: {oracle_search_card['budget']['value']}")

    # 5. Random Reference sampling S_random^(1..N) and FREEZE ARTIFACTS
    print("\n[Step 4/6] Sampling S_random^(1..N) and freezing reference artifacts BEFORE C evaluation...")
    rr_cfg = cfg["random_reference"]
    s_random_list, random_reference_card = sample_random_reference_subsets(
        pool_size=pool_size,
        k=k,
        N_random=rr_cfg["N_random"],
        seeds=rr_cfg["random_seeds"]
    )
    print(f"  S_random ({len(s_random_list)} frozen subsets sampled)")

    # 6. Evaluation on C with paired example-level bootstrap
    print("\n[Step 5/6] Evaluating on Dataset C with Paired Example Bootstrap...")
    # Calculate per-example quality metrics for each candidate over dataset C
    q_selector_ex = np.array([np.max(ex["expert_scores"][list(s_selector)]) for ex in c_examples])
    q_oracle_ex = np.array([np.max(ex["expert_scores"][list(s_oracle)]) for ex in c_examples])
    
    q_random_ex_matrix = np.zeros((len(s_random_list), len(c_examples)))
    for j, s_rand in enumerate(s_random_list):
        q_random_ex_matrix[j] = [np.max(ex["expert_scores"][list(s_rand)]) for ex in c_examples]

    unc_cfg = cfg["uncertainty"]
    rse_result = compute_paired_bootstrap_metrics(
        q_selector_examples=q_selector_ex,
        q_oracle_examples=q_oracle_ex,
        q_random_examples_matrix=q_random_ex_matrix,
        n_replicates=unc_cfg["bootstrap_replicates"],
        confidence_level=unc_cfg["confidence_level"],
        seed=42
    )

    # Attach experiment metadata
    rse_result["experiment"] = cfg["experiment"]
    rse_result["candidate_space"] = cfg["candidate_space"]
    rse_result["random_reference"] = {
        "N": len(s_random_list),
        "frozen": True,
        "resampled_in_bootstrap": False
    }

    print("-" * 80)
    print("EVALUATION RESULTS SUMMARY:")
    print(f"  ESP@k:        {rse_result['metrics']['ESP']['value']:.4f}  [Status: {rse_result['metrics']['ESP']['status']}]")
    print(f"  Denominator:  {rse_result['metrics']['denominator']['delta_search_random']:.4f}  [Status: {rse_result['metrics']['denominator']['status']}]")
    print(f"  RSE@k:        {rse_result['metrics']['RSE']['value']}  [Ratio Status: {rse_result['metrics']['RSE']['ratio_status']}]")
    print(f"  Diagnostics:  invalid_fraction = {rse_result['bootstrap_diagnostics']['invalid_fraction']:.6f} "
          f"({rse_result['bootstrap_diagnostics']['invalid_ratio_replicates']}/{rse_result['bootstrap_diagnostics']['total_replicates']})")
    print("-" * 80)

    # 7. Save 5 reproducible artifacts
    print(f"\n[Step 6/6] Saving 5 reproducible artifacts to {out_dir}...")
    saved_paths = save_artifact_snapshot(
        output_dir=out_dir,
        config=cfg,
        grouping_card=grouping_card,
        oracle_search_card=oracle_search_card,
        random_reference_card=random_reference_card,
        rse_result=rse_result
    )
    for name, path in saved_paths.items():
        print(f"  Artifact [{name}]: {path}")

    print("\nPipeline execution complete. PASS.")


if __name__ == "__main__":
    main()
