"""
experiments/benchmark_tr82_prefetch.py — ASIR TR-8.2 Async Prefetch & Oracle Benchmark
"""

import sys
import os
import json
import argparse
import time
import torch
import numpy as np
from typing import Tuple, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.tokenizer import CharTokenizer
from training.train import get_model_config
from training.models.transformer import MoETransformer
from runtime.expert_store import RAMExpertStore, NVMeExpertStore
from runtime.expert_cache import ExpertCache
from runtime.engine import InferenceEngine


def populate_expert_stores(
    model: MoETransformer,
    storage_dir: str,
    device: torch.device,
    nvme_bw_gbps: float = 5.0,
    max_workers: int = 4
) -> Tuple[RAMExpertStore, NVMeExpertStore]:
    """Extracts expert weights and saves them to disk simulating NVMe storage."""
    ram_store = RAMExpertStore(device=device, pin_memory=False)
    nvme_store = NVMeExpertStore(storage_dir=storage_dir, ram_store=ram_store, nvme_bw_gbps=nvme_bw_gbps, max_workers=max_workers)
    
    for l_idx, block in enumerate(model.blocks):
        if block.moe:
            moe_layer = block.ffn
            for e_idx, expert in enumerate(moe_layer.experts):
                state_dict = expert.state_dict()
                nvme_store.register_expert_disk(l_idx, e_idx, state_dict)
                
    return ram_store, nvme_store


def main():
    print("=" * 105)
    print("ASIR TR-8.2 ASYNC PREFETCH & ORACLE BENCHMARK")
    print("=" * 105)

    device = torch.device('cpu')
    tokenizer = CharTokenizer()
    
    # 1. Initialize a lightweight MoE model for fast CPU simulation
    # M1 parameters, but with tiny FFN dimension to avoid memory-bound CPU overhead
    model = MoETransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=5,
        num_heads=2,
        d_ff=128,
        moe=True,
        num_experts=8,
        top_k=2
    ).to(device)

    # 2. Setup NVMe storage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nvme_dir = os.path.join(script_dir, "..", "results", "TR9", "nvme_store_light")
    
    # Clean old light directory if exists to ensure fresh registration
    import shutil
    shutil.rmtree(nvme_dir, ignore_errors=True)
    
    ram_store, nvme_store = populate_expert_stores(model, nvme_dir, device, nvme_bw_gbps=5.0, max_workers=4)

    # 3. Create transition matrix (Markov transition probabilities for 8 experts)
    # Uniform matrix with a zipfian skew transition
    T_matrix = []
    for i in range(8):
        row = [0.05] * 8
        row[(i + 1) % 8] = 0.55
        row[(i + 2) % 8] = 0.20
        # Normalize
        row_sum = sum(row)
        T_matrix.append([val / row_sum for val in row])

    # 4. Configurations to evaluate
    # Capacity = 4 experts per layer (total 20 experts capacity)
    capacity_per_layer = 4
    total_capacity = capacity_per_layer * 5
    prompt = "Test prime factorization of big numbers"
    gen_len = 30

    test_configs = [
        ("No Prefetch", "lru", 0),
        ("Async Prefetch (Top-1)", "lru_prefetch", 1),
        ("Async Prefetch (Top-2)", "lru_prefetch", 2),
        ("Async Prefetch (Top-4)", "lru_prefetch", 4),
        ("Oracle Prefetch", "oracle", 0),
    ]

    results = {}

    print(f"Executing decode sweeps (gen_len={gen_len} steps, capacity={capacity_per_layer} experts/layer)...")
    print("-" * 105)
    print(f"  {'Policy/Config':26s} | {'Top-K':5s} | {'Decode TPS':12s} | {'Hit Rate':10s} | {'Precision':11s} | {'Recall':10s}")
    print("-" * 105)

    for name, policy, top_k_prefetch in test_configs:
        # Reset storage and stats
        ram_store.store.clear()
        nvme_store.reset_stats()
        
        cache = ExpertCache(
            capacity_experts=total_capacity,
            nvme_store=nvme_store,
            policy=policy,
            transition_matrix=T_matrix
        )
        
        engine = InferenceEngine(model, tokenizer, expert_cache=cache, device=device, top_k_prefetch=top_k_prefetch)
        
        # Warmup and execute generation
        gen_text, summary = engine.generate(prompt, max_gen_len=gen_len)
        
        tps = summary['decode_tok_per_sec']
        hit_rate = summary['cache']['hit_rate_pct']
        prec = summary['cache'].get('prefetch_precision_pct', 0.0)
        rec = summary['cache'].get('prefetch_recall_pct', 0.0)
        
        results[name] = {
            'policy': policy,
            'top_k_prefetch': top_k_prefetch,
            'summary': summary
        }
        
        top_k_str = str(top_k_prefetch) if top_k_prefetch > 0 else "N/A"
        print(f"  {name:26s} | {top_k_str:5s} | {tps:9.2f} tok/s | {hit_rate:8.2f}% | {prec:9.2f}% | {rec:8.2f}%")

    print("-" * 105)

    # 5. Clean up light store
    nvme_store.shutdown()
    shutil.rmtree(nvme_dir, ignore_errors=True)

    # 6. Save output JSON
    output_dir = os.path.join(script_dir, "..", "results", "TR9")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "prefetch_async_benchmark.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nBenchmark results saved successfully to: {out_file}")
    print("=" * 105)


if __name__ == "__main__":
    main()
