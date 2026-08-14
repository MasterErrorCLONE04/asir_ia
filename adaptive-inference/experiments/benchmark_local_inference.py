"""
experiments/benchmark_local_inference.py — ASIR Physical Local Inference Benchmark (TR-10)

Evaluates physical autoregressive token decode throughput (decode tok/s), prefill throughput,
cache hit rates, exact NVMe data transfer volume (MB/tok), and physical NVMe streaming bandwidth.

Configurations benchmarked:
  - Config A: 100% experts resident in RAM (baseline)
  - Config B: 50% experts resident in RAM / 50% NVMe
  - Config C: 25% experts resident in RAM / 75% NVMe
  - Config D: 12.5% experts resident in RAM / 87.5% NVMe

Usage:
    python experiments/benchmark_local_inference.py --model M1 --gen_len 30
"""

import sys
import os
import json
import argparse
import time
import torch
import numpy as np
from typing import Tuple, Dict, Any

# Add adaptive-inference directory to path
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
    nvme_bw_gbps: float = 5.0
) -> Tuple[RAMExpertStore, NVMeExpertStore]:
    """
    Extracts expert parameters from model and registers them into RAM & NVMe stores.
    """
    ram_store = RAMExpertStore(device=device, pin_memory=True)
    nvme_store = NVMeExpertStore(storage_dir=storage_dir, ram_store=ram_store, nvme_bw_gbps=nvme_bw_gbps)
    
    for l_idx, block in enumerate(model.blocks):
        if block.moe:
            moe_layer = block.ffn
            for e_idx, expert in enumerate(moe_layer.experts):
                state_dict = expert.state_dict()
                nvme_store.register_expert_disk(l_idx, e_idx, state_dict)
                
    return ram_store, nvme_store


def main():
    parser = argparse.ArgumentParser(description="ASIR Physical Local Inference Benchmark (TR-10)")
    parser.add_argument("--model", type=str, default="M1", help="Model config (M1, M2, M3, M4, M5, K3-Router).")
    parser.add_argument("--prompt", type=str, default="Calculate prime numbers up to 100", help="Test prompt.")
    parser.add_argument("--gen_len", type=int, default=30, help="Number of tokens to generate in decode phase.")
    parser.add_argument("--nvme_bw", type=float, default=5.0, help="PCIe Gen4 NVMe Bandwidth in GB/s.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 95)
    print("ASIR PHYSICAL LOCAL INFERENCE BENCHMARK (TR-10)")
    print("=" * 95)
    print(f"Device: {device} | Model: {args.model} | NVMe Bandwidth: {args.nvme_bw} GB/s")

    # 1. Initialize Tokenizer & Model
    tokenizer = CharTokenizer()
    config = get_model_config(args.model)
    
    model = MoETransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=1408,
        n_layers=5,
        num_heads=8,
        d_ff=config['d_ff'],
        moe=config['moe'],
        num_experts=config['num_experts'],
        top_k=config['top_k']
    ).to(device)

    # Check for trained model checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "..", "checkpoints", args.model, "model.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 2. Setup Physical Expert Storage
    nvme_dir = os.path.join(script_dir, "..", "results", "TR9", "nvme_store")
    ram_store, nvme_store = populate_expert_stores(model, nvme_dir, device, nvme_bw_gbps=args.nvme_bw)

    # Load transition matrix if available from TR-8
    locality_file = os.path.join(script_dir, "..", "results", "TR8", "m2_locality_results.json")
    T_matrix = None
    if os.path.exists(locality_file):
        with open(locality_file, 'r', encoding='utf-8') as f:
            loc_data = json.load(f)
            T_matrix = loc_data.get('transitions', {}).get('temporal_transition_matrix', None)

    # 3. Residency Configurations to Benchmark
    num_experts = config['num_experts']
    residency_configs = [
        ("Config A (100% RAM)", num_experts),
        ("Config B (50% RAM)", max(1, num_experts // 2)),
        ("Config C (25% RAM)", max(1, num_experts // 4)),
        ("Config D (12.5% RAM)", max(1, num_experts // 8))
    ]

    benchmark_results = {}

    print("\nExecuting Local Autoregressive Decode Benchmark...")
    print("-" * 95)
    print(f"  {'Configuration':22s} | {'C':3s} | {'Policy':10s} | {'Decode tok/s':13s} | {'Hit Rate':10s} | {'NVMe MB/tok':13s}")
    print("-" * 95)

    for cfg_name, cap_c in residency_configs:
        for policy in ["lru", "lru_prefetch"]:
            # Reset RAM store and nvme stats
            ram_store.store.clear()
            nvme_store.reset_stats()
            
            cache = ExpertCache(
                capacity_experts=cap_c * 5,  # capacity across 5 layers
                nvme_store=nvme_store,
                policy=policy,
                transition_matrix=T_matrix
            )
            
            engine = InferenceEngine(model, tokenizer, expert_cache=cache, device=device)
            
            # Execute Generation
            gen_text, summary = engine.generate(args.prompt, max_gen_len=args.gen_len)
            
            decode_tps = summary['decode_tok_per_sec']
            hit_rate = summary['cache']['hit_rate_pct']
            nvme_mb_per_tok = summary['nvme_mb_read'] / max(summary['decode_tokens'], 1)
            
            res_key = f"{cfg_name}_{policy}"
            benchmark_results[res_key] = {
                'config': cfg_name,
                'capacity_per_layer': cap_c,
                'policy': policy,
                'generated_text': gen_text,
                'summary': summary
            }
            
            target_str = " (TARGET HIT)" if decode_tps >= 20.0 else ""
            print(f"  {cfg_name:22s} | {cap_c:3d} | {policy:10s} | {decode_tps:10.2f} tok/s | {hit_rate:9.2f}% | {nvme_mb_per_tok:11.2f} MB{target_str}")
            
    print("-" * 95)

    # Save Output JSON
    output_dir = os.path.join(script_dir, "..", "results", "TR9")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "local_inference_benchmark.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': args.model,
            'config': config,
            'nvme_bw_gbps': args.nvme_bw,
            'benchmark_results': benchmark_results
        }, f, indent=2)

    print(f"\nBenchmark results saved successfully to: {out_file}")
    print("=" * 95)


if __name__ == "__main__":
    main()
