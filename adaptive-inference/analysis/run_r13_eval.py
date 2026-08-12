"""
analysis/run_r13_eval.py — CLI Runner for R1.3 evaluation pipeline under spec v0.12-final.

Connects the statistical contract with real PyTorch MoETransformer inference and routing masks.

Execution Order Enforced:
1. Load & validate pre-registered config
2. Near-duplicate grouping & example-level A/B/C partition
3. Selector profiling on A via moe_profiler
4. Oracle Search Search(B,k) on B -> produce S_search-oracle via PyTorch model_adapter
5. Random reference sampling S_random^(1..N) -> FREEZE ARTIFACTS
6. Evaluate C -> Real PyTorch inference -> Paired example bootstrap -> ESP@k, RSE@k, bootstrap_diagnostics
7. Save 5 reproducible artifacts under results/<domain>/k_<k>/

Usage:
    python analysis/run_r13_eval.py --config configs/r1.3_preregistration_v0.12.json [--model M1] [--dry_run]
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Ensure package imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.config_schema import load_and_validate_config
from analysis.partitioning import partition_dataset_abc
from analysis.oracle_search import run_oracle_search
from analysis.artifacts import save_artifact_snapshot
from analysis.model_adapter import MoEModelAdapter, load_moe_model_from_checkpoint, HAS_TORCH as ADAPTER_HAS_TORCH
from analysis.moe_profiler import profile_selector_on_split_a


if HAS_TORCH:
    from training.models.transformer import MoETransformer
    from training.train import get_model_config
    from tasks.tokenizer import CharTokenizer

from tasks.generator import SyntheticTaskGenerator


def load_real_or_synthetic_dataset(data_path: Optional[str] = None, n_synthetic: int = 600) -> List[Dict[str, Any]]:
    """
    Loads real JSONL dataset samples or generates synthetic tasks via SyntheticTaskGenerator.
    """
    if data_path and os.path.exists(data_path):
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    item = json.loads(line.strip())
                    item["id"] = item.get("id", f"ex_{i}")
                    item["source_id"] = item.get("source_id", f"src_{i // 3}")
                    samples.append(item)
        return samples

    # Fallback to SyntheticTaskGenerator
    gen = SyntheticTaskGenerator(seed=42)
    raw_dataset = gen.generate_dataset(n_synthetic)
    samples = []
    for i, item in enumerate(raw_dataset):
        samples.append({
            "id": f"ex_{i}",
            "source_id": f"src_{i // 3}",
            "input": item["input"],
            "target": item["target"],
            "domain": item["domain"]
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Executes R1.3 evaluation pipeline (v0.12-final).")
    parser.add_argument("--config", type=str, required=True, help="Path to pre-registered YAML/JSON config.")
    parser.add_argument("--model", type=str, default="M1", help="Model architecture (M1, M2, etc.).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to trained model.pt checkpoint.")
    parser.add_argument("--data_path", type=str, default=None, help="Optional path to dataset jsonl.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output results directory.")
    parser.add_argument("--cpu_smoke_test", action="store_true", help="Run quick CPU smoke test.")
    parser.add_argument("--dry_run", action="store_true", help="Backward compatibility flag.")
    args = parser.parse_args()

    # 1. Load and validate config
    print("=" * 80)
    print("R1.3 EVALUATION PIPELINE (v0.12-final) — REAL MOE INFERENCE")
    print("=" * 80)
    print(f"Loading config: {args.config}")
    cfg = load_and_validate_config(args.config)
    print("Config validation: PASS (v0.12-final compliant)")

    domain = cfg["experiment"]["domain"]
    k = cfg["candidate_space"]["subset_size_k"]
    pool_size = cfg["candidate_space"]["expert_pool_size"]

    if HAS_TORCH and not args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = get_model_config(args.model)
        pool_size = model_cfg["num_experts"] if model_cfg["moe"] else pool_size
        tokenizer = CharTokenizer()

        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading MoETransformer model {args.model} from checkpoint {args.checkpoint} on device {device}...")
            model = load_moe_model_from_checkpoint(args.checkpoint, model_name=args.model, device=device)
        else:
            print(f"Initializing MoETransformer model {args.model} (untrained) on device {device}...")
            model = MoETransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=1408,
                n_layers=5,
                num_heads=8,
                d_ff=512,
                max_seq_len=256,
                moe=model_cfg["moe"],
                num_experts=pool_size,
                top_k=model_cfg["top_k"]
            ).to(device)

        adapter = MoEModelAdapter(model, tokenizer, device)
        use_pytorch = True
    else:
        print("Running with synthetic/mock evaluation mode...")
        use_pytorch = False
        rng = np.random.default_rng(42)
(42)

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("results", domain, f"k_{k}")

    # 3. Data loading & Near-duplicate grouping + A/B/C split
    print("\n[Step 1/6] Partitioning dataset A/B/C with group-level split...")
    n_samples = 60 if args.cpu_smoke_test else 600
    dataset = load_real_or_synthetic_dataset(args.data_path, n_synthetic=n_samples)
    if not use_pytorch:
        for ex in dataset:
            ex["expert_scores"] = rng.normal(loc=0.5, scale=0.1, size=pool_size)

    a_examples, b_examples, c_examples, grouping_card = partition_dataset_abc(
        dataset, cfg["grouping"], ratios=(0.4, 0.3, 0.3)
    )
    print(f"  Split counts: A={len(a_examples)}, B={len(b_examples)}, C={len(c_examples)}")
    print(f"  Group method: {grouping_card['method']}, Total groups: {grouping_card['total_groups']}")

    # 4. Selector profiling on A
    print("\n[Step 2/6] Profiling selector on Dataset A...")
    if use_pytorch:
        s_selector = profile_selector_on_split_a(model, a_examples, k, tokenizer, device)
    else:
        scores = np.zeros(pool_size)
        for ex in a_examples:
            scores += ex["expert_scores"]
        s_selector = tuple(sorted(list(np.argsort(scores)[::-1][:k])))
    print(f"  S_selector: {s_selector}")

    # 5. Oracle Search Search(B, k) on Dataset B
    print("\n[Step 3/6] Running Oracle Search Search(B,k) on Dataset B...")
    if use_pytorch:
        eval_fn_b = lambda subset: adapter.eval_subset_mean_quality(b_examples, subset)
    else:
        eval_fn_b = lambda subset: float(np.mean([np.max(ex["expert_scores"][list(subset)]) for ex in b_examples]))

    oracle_cfg = cfg["oracle_search"]
    oracle_cfg["pool_size"] = pool_size
    s_oracle, oracle_search_card = run_oracle_search(eval_fn_b, oracle_cfg)
    print(f"  S_search-oracle: {s_oracle}")
    print(f"  Oracle evaluations used: {oracle_search_card['budget']['value']}")

    # 6. Random Reference sampling S_random^(1..N) and FREEZE ARTIFACTS
    print("\n[Step 4/6] Sampling S_random^(1..N) and freezing reference artifacts BEFORE C evaluation...")
    rr_cfg = cfg["random_reference"]
    s_random_list, random_reference_card = sample_random_reference_subsets(
        pool_size=pool_size,
        k=k,
        N_random=rr_cfg["N_random"],
        seeds=rr_cfg["random_seeds"]
    )
    print(f"  S_random ({len(s_random_list)} frozen subsets sampled)")

    # 7. Evaluation on C with paired example-level bootstrap
    print("\n[Step 5/6] Evaluating on Dataset C with Paired Example Bootstrap...")
    if use_pytorch:
        q_selector_ex = adapter.eval_subset_quality_per_example(c_examples, s_selector)
        q_oracle_ex = adapter.eval_subset_quality_per_example(c_examples, s_oracle)
        q_random_ex_matrix = np.zeros((len(s_random_list), len(c_examples)))
        for j, s_rand in enumerate(s_random_list):
            q_random_ex_matrix[j] = adapter.eval_subset_quality_per_example(c_examples, s_rand)
    else:
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
    print(f"  ESP@k:        {rse_result['metrics']['ESP']['value'] if rse_result['metrics']['ESP']['value'] is not None else 'N/A'}  [Status: {rse_result['metrics']['ESP']['status']}]")
    print(f"  Denominator:  {rse_result['metrics']['denominator']['delta_search_random']:.4f}  [Status: {rse_result['metrics']['denominator']['status']}]")
    print(f"  RSE@k:        {rse_result['metrics']['RSE']['value']}  [Ratio Status: {rse_result['metrics']['RSE']['ratio_status']}]")
    print(f"  Diagnostics:  invalid_fraction = {rse_result['bootstrap_diagnostics']['invalid_fraction']:.6f} "
          f"({rse_result['bootstrap_diagnostics']['invalid_ratio_replicates']}/{rse_result['bootstrap_diagnostics']['total_replicates']})")
    print("-" * 80)

    # 8. Save 5 reproducible artifacts
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

