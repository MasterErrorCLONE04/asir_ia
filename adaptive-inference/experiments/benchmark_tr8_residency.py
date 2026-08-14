"""
experiments/benchmark_tr8_residency.py — ASIR-TR-8.1 ResidencyManager Empirical Benchmark

Evaluates memory tiering (CPU RAM <-> GPU VRAM), lease protection, and eviction policy performance.
Measures:
  1. Capacity Sweep: N_resident in {4, 8, 16, 24, 32}
  2. Eviction Policy Comparison: LRU vs LFU vs LFRU
  3. Operating Metrics: Hit-rate (%), Miss-rate (%), H2D Transfers (MB), H2D Latency (ms), Execution Latency (ms)
"""

import os
import sys
import time
import json
import torch
import numpy as np
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.moe.reference import ReferenceFFNExpert
from runtime.residency import ResidencyManager, ExpertKey


def create_synthetic_expert_weights(d_model: int = 1408, d_ff: int = 3516) -> Dict[str, torch.Tensor]:
    """Generates synthetic expert state dict matching M2 dimensions."""
    expert = ReferenceFFNExpert(d_model, d_ff)
    return expert.state_dict()


def run_capacity_sweep_benchmark(device: torch.device) -> Dict[str, Any]:
    """
    Experiment A: Capacity Sweep N_resident in {4, 8, 16, 24, 32}.
    Simulates a sequence of MoE layer requests with Zipfian/Skewed expert access.
    """
    print("\n=================== EXPERIMENT A: VRAM CAPACITY SWEEP (N_resident) ===================")

    num_layers = 5
    num_experts = 32
    top_k = 2
    num_steps = 100
    d_model, d_ff = 64, 128

    # Create single synthetic expert template state dict and reuse across keys
    sample_sd = create_synthetic_expert_weights(d_model, d_ff)
    all_keys = [ExpertKey(l, e) for l in range(num_layers) for e in range(num_experts)]

    capacities = [4, 8, 16, 24, 32]
    results = {}

    header = f"{'VRAM Cap':<10} | {'Hit Rate (%)':<13} | {'Misses':<8} | {'Evictions':<10} | {'H2D (MB)':<10} | {'H2D Time (ms)':<14}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    # Generate fixed pseudo-random routing trace (Zipfian skew towards hot experts)
    torch.manual_seed(42)
    zipf_weights = 1.0 / (torch.arange(1, num_experts + 1, dtype=torch.float) ** 0.8)
    zipf_probs = zipf_weights / zipf_weights.sum()

    trace: List[List[ExpertKey]] = []
    for step in range(num_steps):
        for l in range(num_layers):
            chosen = torch.multinomial(zipf_probs, num_samples=top_k, replacement=False)
            step_keys = [ExpertKey(l, e.item()) for e in chosen]
            trace.append(step_keys)

    for cap in capacities:
        rm = ResidencyManager(vram_capacity_experts=cap, device=device, policy="lfru")
        for key in all_keys:
            # Reuse cloned template dictionary
            rm.register_expert(key, {k: v.clone() for k, v in sample_sd.items()}, pin_memory=False)

        for step_keys in trace:
            leased_dict = rm.acquire(step_keys)
            # Simulate computation...
            rm.release(step_keys)


        stats = rm.stats()
        results[f"Cap_{cap}"] = stats

        print(
            f"{cap:<10} | {stats['hit_rate_pct']:13.2f} | {stats['misses']:8d} | "
            f"{stats['evictions']:10d} | {stats['h2d_mb_transferred']:10.1f} | "
            f"{stats['h2d_transfer_time_ms']:14.2f}"
        )

    return results


def run_policy_comparison_benchmark(device: torch.device) -> Dict[str, Any]:
    """
    Experiment B: Eviction Policy Comparison (LRU vs LFU vs LFRU).
    """
    print("\n=================== EXPERIMENT B: EVICTION POLICY COMPARISON ===================")

    num_layers = 5
    num_experts = 32
    top_k = 2
    num_steps = 100
    d_model, d_ff = 64, 128
    vram_cap = 8

    all_keys = [ExpertKey(l, e) for l in range(num_layers) for e in range(num_experts)]
    sample_sd = create_synthetic_expert_weights(d_model, d_ff)

    policies = ["lru", "lfu", "lfru"]
    results = {}

    header = f"{'Policy':<10} | {'Hit Rate (%)':<13} | {'Misses':<8} | {'Evictions':<10} | {'H2D (MB)':<10}"
    print(header)
    print("-" * len(header))

    torch.manual_seed(42)
    zipf_weights = 1.0 / (torch.arange(1, num_experts + 1, dtype=torch.float) ** 0.8)
    zipf_probs = zipf_weights / zipf_weights.sum()

    trace: List[List[ExpertKey]] = []
    for step in range(num_steps):
        for l in range(num_layers):
            chosen = torch.multinomial(zipf_probs, num_samples=top_k, replacement=False)
            step_keys = [ExpertKey(l, e.item()) for e in chosen]
            trace.append(step_keys)

    for pol in policies:
        rm = ResidencyManager(vram_capacity_experts=vram_cap, device=device, policy=pol)
        for key in all_keys:
            rm.register_expert(key, {k: v.clone() for k, v in sample_sd.items()}, pin_memory=False)

        for step_keys in trace:
            leased_dict = rm.acquire(step_keys)
            rm.release(step_keys)

        stats = rm.stats()
        results[pol] = stats

        print(
            f"{pol.upper():<10} | {stats['hit_rate_pct']:13.2f} | {stats['misses']:8d} | "
            f"{stats['evictions']:10d} | {stats['h2d_mb_transferred']:10.1f}"
        )

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ASIR-TR-8.1 Residency Benchmark on device: {device}")

    exp_a_results = run_capacity_sweep_benchmark(device)
    exp_b_results = run_policy_comparison_benchmark(device)

    summary = {
        'milestone': 'ASIR-TR-8.1',
        'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        'experiment_a_capacity_sweep': exp_a_results,
        'experiment_b_policy_comparison': exp_b_results
    }

    out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "profiling", "M2"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "asir_tr8_residency.json")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTR-8.1 Residency Benchmark results saved to: {out_path}\n", flush=True)


main()
