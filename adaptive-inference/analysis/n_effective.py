"""
analysis/n_effective.py — Descriptive trend aggregator for R1.0 scaling studies.

Aggregates all JSON results for different values of N_total and reports the
relationship between N_total, N_eff, η_cap, and quality Q.

Usage:
    python analysis/n_effective.py --configs M1,M2,M3,M4,M5
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np

def get_config_hash(model_name: str, args) -> str:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.train import get_model_config
    cfg = get_model_config(model_name)
    
    config_dict = {
        'model':       model_name,
        'moe':         cfg['moe'],
        'num_experts': cfg['num_experts'],
        'top_k':       cfg['top_k'],
        'd_ff':        cfg['d_ff'],
        'lr':          args.lr,
        'batch_size':  args.batch_size,
        'aux_coef':    args.aux_coef,
        'controlled':  args.controlled,
        'epochs':      args.epochs,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Aggregates and reports descriptive trends for MoE scaling.")
    parser.add_argument("--configs", type=str, default="M1,M2,M3,M4,M5", help="Comma-separated model names.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--controlled", action="store_true", help="Controlled MoE routing enabled.")
    parser.add_argument("--aux_coef", type=float, default=0.01, help="Aux loss coef.")
    
    args = parser.parse_args()

    model_names = [name.strip() for name in args.configs.split(",") if name.strip()]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    print("=" * 80)
    print("DESCRIPTIVE TREND AGGREGATOR (§11 v0.7)")
    print("=" * 80)
    print(f"  Aggregating configs: {model_names}")
    print("-" * 80)
    print(f"  {'Config':8s} | {'N_total':7s} | {'Seeds':5s} | {'Mean Q':8s} | {'Mean N_eff':10s} | {'Mean η_cap':10s}")
    print("-" * 80)

    for name in model_names:
        hash_val = get_config_hash(name, args)
        path = os.path.join(results_dir, "R1.0", hash_val)
        
        if not os.path.exists(path):
            print(f"  {name:8s} | (no results found under hash {hash_val})")
            continue
            
        q_vals = []
        n_eff_vals = []
        eta_cap_vals = []
        seeds = []
        
        for filename in os.listdir(path):
            if filename.startswith("seed_") and filename.endswith("_results.json"):
                parts = filename.split("_")
                try:
                    seed = int(parts[1])
                except ValueError:
                    continue
                
                filepath = os.path.join(path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    res = json.load(f)
                    
                q_vals.append(res['metrics']['quality_q'])
                n_eff_vals.append(res['metrics']['avg_n_eff'])
                eta_cap_vals.append(res['metrics']['avg_eta_cap'])
                seeds.append(seed)
                
        if seeds:
            mean_q = np.mean(q_vals)
            mean_n_eff = np.mean(n_eff_vals)
            mean_eta_cap = np.mean(eta_cap_vals)
            
            # Fetch N_total from the first result file
            n_total = res['hyperparameters']['num_experts']
            
            print(f"  {name:8s} | {n_total:<7d} | {len(seeds):<5d} | {mean_q:7.2f}% | {mean_n_eff:10.2f} | {mean_eta_cap:10.4f}")
        else:
            print(f"  {name:8s} | (no seed files found under {hash_val})")
            
    print("=" * 80)


if __name__ == "__main__":
    main()
