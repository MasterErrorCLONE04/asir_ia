"""
analysis/tr8_locality_analysis.py — TR-8: Expert Residency & Locality Analysis

Empirical telemetry collector, working-set analyzer, expert cache simulator (Static, LRU, LFU, Markov Prefetch),
NVMe traffic calculator, and 50ms latency budget evaluator for sparse MoE models (M2 / Kimi K3 derivative).

Usage:
    python analysis/tr8_locality_analysis.py --model M2 --max_steps 500
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, deque

# Add adaptive-inference directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.tokenizer import CharTokenizer
from training.train import get_model_config, SyntheticDataset, collate_fn
from training.models.transformer import MoETransformer


def compute_gini(frequencies: np.ndarray) -> float:
    """Computes Gini coefficient of inequality for expert selection frequencies."""
    if len(frequencies) == 0 or np.sum(frequencies) == 0:
        return 0.0
    sorted_freqs = np.sort(frequencies)
    n = len(sorted_freqs)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_freqs)) / (n * np.sum(sorted_freqs)))


class RouterTelemetryCollector:
    """
    Passes evaluation data through the model and captures exact token-by-token
    and layer-by-layer router expert selections.
    """
    def __init__(self, model: MoETransformer, device: torch.device):
        self.model = model
        self.device = device
        self.num_experts = model.blocks[0].ffn.num_experts if model.moe else 1
        self.top_k = model.blocks[0].ffn.top_k if model.moe else 1
        self.n_layers = len(model.blocks)

    def collect_telemetry(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, Any]:
        self.model.eval()
        
        # Format: layer_expert_selections[layer_idx] = list of (b, seq) tuples of selected top_k expert indices
        layer_selections: List[List[List[int]]] = [[] for _ in range(self.n_layers)]
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                
                # Forward pass collecting router logits
                use_amp = (self.device.type == 'cuda')
                amp_ctx = torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_amp)
                
                with amp_ctx:
                    _, all_router_logits = self.model(input_ids, mask=None)
                    
                # Process router logits per layer
                if self.model.moe and all_router_logits:
                    for layer_idx, router_logits in enumerate(all_router_logits):
                        # router_logits shape: (batch_size, seq_len, num_experts)
                        flat_logits = router_logits.view(-1, self.num_experts)
                        _, topk_indices = torch.topk(flat_logits, self.top_k, dim=-1)
                        # Convert to list of lists: each element is top_k experts for one token
                        topk_list = topk_indices.cpu().tolist()
                        layer_selections[layer_idx].extend(topk_list)
                        
        return {
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'n_layers': self.n_layers,
            'layer_selections': layer_selections, # layer -> list of [top_k expert ids] per token
            'total_tokens': len(layer_selections[0]) if layer_selections else 0
        }


def analyze_expert_frequencies(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Computes global and per-layer selection probabilities, N_eff, eta_cap, and Gini coefficient."""
    num_experts = telemetry['num_experts']
    n_layers = telemetry['n_layers']
    layer_selections = telemetry['layer_selections']
    
    per_layer_counts = np.zeros((n_layers, num_experts), dtype=np.int64)
    
    for l_idx in range(n_layers):
        for token_experts in layer_selections[l_idx]:
            for e_id in token_experts:
                per_layer_counts[l_idx, e_id] += 1
                
    global_counts = per_layer_counts.sum(axis=0)
    total_assignments = global_counts.sum()
    
    global_probs = global_counts / max(total_assignments, 1)
    n_eff = 1.0 / (np.sum(global_probs ** 2) + 1e-9)
    eta_cap = n_eff / num_experts
    gini = compute_gini(global_counts)
    
    per_layer_stats = []
    for l_idx in range(n_layers):
        l_counts = per_layer_counts[l_idx]
        l_total = l_counts.sum()
        l_probs = l_counts / max(l_total, 1)
        l_n_eff = 1.0 / (np.sum(l_probs ** 2) + 1e-9)
        l_eta = l_n_eff / num_experts
        l_gini = compute_gini(l_counts)
        per_layer_stats.append({
            'layer': l_idx,
            'counts': l_counts.tolist(),
            'probs': l_probs.tolist(),
            'n_eff': float(l_n_eff),
            'eta_cap': float(l_eta),
            'gini': float(l_gini)
        })
        
    return {
        'global_counts': global_counts.tolist(),
        'global_probs': global_probs.tolist(),
        'n_eff': float(n_eff),
        'eta_cap': float(eta_cap),
        'gini': float(gini),
        'per_layer_stats': per_layer_stats
    }


def analyze_working_sets(telemetry: Dict[str, Any], window_sizes: List[int] = [100, 500, 1000]) -> Dict[str, Any]:
    """
    Computes active working set sizes W(w) over sliding token windows w across layers.
    Returns average and 50th, 90th, 95th, 99th percentile working set sizes.
    """
    layer_selections = telemetry['layer_selections']
    n_layers = telemetry['n_layers']
    total_tokens = telemetry['total_tokens']
    
    results = {}
    
    for w in window_sizes:
        if total_tokens < w:
            continue
            
        layer_ws_stats = []
        all_unique_counts = []
        
        for l_idx in range(n_layers):
            selections = layer_selections[l_idx]
            unique_counts = []
            
            # Sliding window over tokens
            for i in range(0, total_tokens - w + 1, max(1, w // 10)):
                window_tokens = selections[i : i + w]
                unique_experts = set(e for t in window_tokens for e in t)
                unique_counts.append(len(unique_experts))
                all_unique_counts.append(len(unique_experts))
                
            layer_ws_stats.append({
                'layer': l_idx,
                'mean': float(np.mean(unique_counts)),
                'p50': float(np.percentile(unique_counts, 50)),
                'p90': float(np.percentile(unique_counts, 90)),
                'p95': float(np.percentile(unique_counts, 95)),
                'p99': float(np.percentile(unique_counts, 99)),
                'max': int(np.max(unique_counts))
            })
            
        results[f'window_{w}'] = {
            'global_mean': float(np.mean(all_unique_counts)),
            'global_p50': float(np.percentile(all_unique_counts, 50)),
            'global_p90': float(np.percentile(all_unique_counts, 90)),
            'global_p95': float(np.percentile(all_unique_counts, 95)),
            'global_p99': float(np.percentile(all_unique_counts, 99)),
            'global_max': int(np.max(all_unique_counts)),
            'per_layer': layer_ws_stats
        }
        
    return results


def analyze_transitions_and_cooccurrence(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes:
      1. Co-occurrence matrix C[i, j] (pair of experts activated together in top_k)
      2. Token temporal transition matrix T[i, j] = P(E_{t+1}=j | E_t=i)
      3. Inter-layer transition matrix L[i, j] = P(E^{(l+1)}=j | E^{(l)}=i)
    """
    num_experts = telemetry['num_experts']
    n_layers = telemetry['n_layers']
    layer_selections = telemetry['layer_selections']
    total_tokens = telemetry['total_tokens']
    
    # 1. Co-occurrence matrix (aggregated across layers)
    co_occurrence = np.zeros((num_experts, num_experts), dtype=np.int64)
    for l_idx in range(n_layers):
        for token_experts in layer_selections[l_idx]:
            if len(token_experts) >= 2:
                for i in range(len(token_experts)):
                    for j in range(i + 1, len(token_experts)):
                        e1, e2 = token_experts[i], token_experts[j]
                        co_occurrence[e1, e2] += 1
                        co_occurrence[e2, e1] += 1
                        
    co_occ_norm = co_occurrence / max(co_occurrence.sum(), 1)
    
    # 2. Token temporal transition matrix T[i, j] (aggregated across layers)
    temporal_trans = np.zeros((num_experts, num_experts), dtype=np.int64)
    for l_idx in range(n_layers):
        selections = layer_selections[l_idx]
        for t in range(len(selections) - 1):
            curr_experts = selections[t]
            next_experts = selections[t + 1]
            for e_curr in curr_experts:
                for e_next in next_experts:
                    temporal_trans[e_curr, e_next] += 1
                    
    # Row-normalize temporal transition matrix
    row_sums_t = temporal_trans.sum(axis=1, keepdims=True)
    T_matrix = np.divide(temporal_trans, row_sums_t, out=np.zeros_like(temporal_trans, dtype=float), where=row_sums_t != 0)
    
    # 3. Inter-layer transition matrix L[i, j] (between layer l and l+1 for same token)
    layer_trans = np.zeros((num_experts, num_experts), dtype=np.int64)
    for l_idx in range(n_layers - 1):
        l_curr = layer_selections[l_idx]
        l_next = layer_selections[l_idx + 1]
        for t in range(min(len(l_curr), len(l_next))):
            curr_e = l_curr[t]
            next_e = l_next[t]
            for e_c in curr_e:
                for e_n in next_e:
                    layer_trans[e_c, e_n] += 1
                    
    row_sums_l = layer_trans.sum(axis=1, keepdims=True)
    L_matrix = np.divide(layer_trans, row_sums_l, out=np.zeros_like(layer_trans, dtype=float), where=row_sums_l != 0)
    
    return {
        'co_occurrence_matrix': co_occurrence.tolist(),
        'co_occurrence_norm': co_occ_norm.tolist(),
        'temporal_transition_matrix': T_matrix.tolist(),
        'layer_transition_matrix': L_matrix.tolist()
    }


class ExpertCacheSimulator:
    """
    Simulates Expert Caches (Static Top-C, LRU, LFU, Markov Prefetch) for resident capacities C in range [2, 32].
    Calculates Hit Rates, Miss Rates, Miss Counts, and NVMe MB/token data traffic.
    """
    def __init__(self, telemetry: Dict[str, Any], global_probs: List[float], T_matrix: List[List[float]]):
        self.telemetry = telemetry
        self.num_experts = telemetry['num_experts']
        self.top_k = telemetry['top_k']
        self.n_layers = telemetry['n_layers']
        self.layer_selections = telemetry['layer_selections']
        self.total_tokens = telemetry['total_tokens']
        self.global_probs = np.array(global_probs)
        self.T_matrix = np.array(T_matrix)

    def simulate(self, capacities: List[int], expert_sizes_mb: List[float] = [50.0, 200.0, 500.0]) -> Dict[str, Any]:
        results = {}
        
        for C in capacities:
            C_res = {}
            # 1. Static Top-C Policy
            top_c_experts = set(np.argsort(self.global_probs)[::-1][:C])
            static_hits = 0
            static_total = 0
            for l_idx in range(self.n_layers):
                for token_experts in self.layer_selections[l_idx]:
                    for e_id in token_experts:
                        static_total += 1
                        if e_id in top_c_experts:
                            static_hits += 1
            static_hit_rate = (static_hits / max(static_total, 1)) * 100.0
            
            # 2. LRU Policy (Per layer cache)
            lru_hits = 0
            lru_total = 0
            for l_idx in range(self.n_layers):
                cache = deque(maxlen=C)
                cache_set = set()
                for token_experts in self.layer_selections[l_idx]:
                    for e_id in token_experts:
                        lru_total += 1
                        if e_id in cache_set:
                            lru_hits += 1
                            # Move to end (most recently used)
                            cache.remove(e_id)
                            cache.append(e_id)
                        else:
                            if len(cache) >= C:
                                evicted = cache.popleft()
                                cache_set.remove(evicted)
                            cache.append(e_id)
                            cache_set.add(e_id)
            lru_hit_rate = (lru_hits / max(lru_total, 1)) * 100.0
            
            # 3. LFU Policy with Windowed Frequency Counter (Window size = 500)
            lfu_hits = 0
            lfu_total = 0
            for l_idx in range(self.n_layers):
                cache_set: Set[int] = set()
                freq_counter: Dict[int, int] = defaultdict(int)
                for token_experts in self.layer_selections[l_idx]:
                    for e_id in token_experts:
                        lfu_total += 1
                        freq_counter[e_id] += 1
                        if e_id in cache_set:
                            lfu_hits += 1
                        else:
                            if len(cache_set) >= C:
                                # Evict expert in cache with lowest frequency count
                                evicted = min(cache_set, key=lambda x: freq_counter[x])
                                cache_set.remove(evicted)
                            cache_set.add(e_id)
            lfu_hit_rate = (lfu_hits / max(lfu_total, 1)) * 100.0
            
            # 4. Markov-Guided Prefetch Policy
            # Uses LRU for C - 1 slots, and uses transition matrix T to prefetch 1 candidate into slot C
            markov_hits = 0
            markov_total = 0
            for l_idx in range(self.n_layers):
                cache = deque(maxlen=C)
                cache_set = set()
                last_expert: Optional[int] = None
                
                for token_experts in self.layer_selections[l_idx]:
                    # Prefetch next predicted expert based on last_expert before processing current token
                    if last_expert is not None and C > 1:
                        predicted_next = int(np.argmax(self.T_matrix[last_expert]))
                        if predicted_next not in cache_set and len(cache_set) >= C:
                            evicted = cache.popleft()
                            cache_set.remove(evicted)
                            cache.append(predicted_next)
                            cache_set.add(predicted_next)
                            
                    for e_id in token_experts:
                        markov_total += 1
                        if e_id in cache_set:
                            markov_hits += 1
                            cache.remove(e_id)
                            cache.append(e_id)
                        else:
                            if len(cache) >= C:
                                evicted = cache.popleft()
                                cache_set.remove(evicted)
                            cache.append(e_id)
                            cache_set.add(e_id)
                        last_expert = e_id
                        
            markov_hit_rate = (markov_hits / max(markov_total, 1)) * 100.0
            
            # Compute NVMe Traffic (MB/token) for LRU policy
            lru_misses = lru_total - lru_hits
            total_tokens = self.total_tokens
            nvme_traffic = {}
            for size_mb in expert_sizes_mb:
                # Total misses across all layers divided by total tokens
                mb_per_token = (lru_misses * size_mb) / max(total_tokens, 1)
                nvme_traffic[f'{int(size_mb)}MB_expert'] = float(mb_per_token)
                
            C_res = {
                'capacity': C,
                'static_hit_rate': float(static_hit_rate),
                'lru_hit_rate': float(lru_hit_rate),
                'lfu_hit_rate': float(lfu_hit_rate),
                'markov_hit_rate': float(markov_hit_rate),
                'lru_miss_rate': float(100.0 - lru_hit_rate),
                'lru_miss_count': int(lru_misses),
                'nvme_traffic_mb_per_tok': nvme_traffic
            }
            results[f'capacity_{C}'] = C_res
            
        return results


def evaluate_latency_budget(
    lru_hit_rates: Dict[int, float],
    expert_size_mb: float = 50.0,
    nvme_bw_gbps: float = 5.0,
    top_k: int = 2,
    n_layers: int = 5,
    compute_time_ms: float = 15.0
) -> Dict[str, Any]:
    """
    Evaluates total per-token latency budget (Target: <= 50 ms / token for ~20 tok/s).
    Formula: T_total = T_compute + T_NVMe
    T_NVMe = (Miss Rate / 100) * top_k * n_layers * (expert_size_mb / nvme_bw_mb_per_ms)
    """
    nvme_bw_mb_per_ms = (nvme_bw_gbps * 1000.0) / 1000.0  # GB/s = MB/ms
    budget_results = {}
    
    for C, hit_rate in lru_hit_rates.items():
        miss_rate = (100.0 - hit_rate) / 100.0
        expected_misses_per_token = miss_rate * top_k * n_layers
        nvme_latency_ms = expected_misses_per_token * (expert_size_mb / nvme_bw_mb_per_ms)
        t_total_ms = compute_time_ms + nvme_latency_ms
        tokens_per_sec = 1000.0 / max(t_total_ms, 1e-3)
        
        budget_results[f'C_{C}'] = {
            'hit_rate': float(hit_rate),
            'miss_rate': float(miss_rate * 100.0),
            'expected_misses_per_token': float(expected_misses_per_token),
            'nvme_latency_ms': float(nvme_latency_ms),
            't_total_ms': float(t_total_ms),
            'est_tok_per_sec': float(tokens_per_sec),
            'meets_20tok_target': bool(tokens_per_sec >= 20.0 or t_total_ms <= 50.0)
        }
        
    return budget_results


def main():
    parser = argparse.ArgumentParser(description="TR-8: Expert Residency & Locality Analysis")
    parser.add_argument("--model", type=str, default="M2", help="Model config (M2).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--max_batches", type=int, default=100, help="Max evaluation batches to run.")
    parser.add_argument("--max_steps", type=int, default=500, help="Training steps if training is executed.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("TR-8: EXPERT RESIDENCY & LOCALITY ANALYSIS")
    print("=" * 80)
    print(f"Device: {device} | Model: {args.model}")

    # Load Tokenizer & Val Dataset
    tokenizer = CharTokenizer()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_file = os.path.join(script_dir, "..", "tasks", "eval_set", "val.jsonl")
    
    if not os.path.exists(val_file):
        print(f"Validation dataset missing at {val_file}. Please run tasks/generate_data.py.")
        sys.exit(1)
        
    val_dataset = SyntheticDataset(val_file, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))

    # Initialize M2 Model
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

    # Check for saved checkpoint or run evaluation with model
    checkpoint_path = os.path.join(script_dir, "..", "checkpoints", args.model, "model.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"No saved checkpoint at {checkpoint_path}. Using initialized weights for telemetry pass.")

    # 1. Collect Telemetry
    print("\n[1/5] Capturing token-level router telemetry...")
    collector = RouterTelemetryCollector(model, device)
    telemetry = collector.collect_telemetry(val_loader, max_batches=args.max_batches)
    print(f"  Total tokens processed: {telemetry['total_tokens']:,}")

    # 2. Analyze Frequencies (Q1)
    print("\n[2/5] Analyzing expert selection concentration (Q1)...")
    freq_results = analyze_expert_frequencies(telemetry)
    print(f"  Global N_eff : {freq_results['n_eff']:.2f} / {config['num_experts']}")
    print(f"  Global η_cap : {freq_results['eta_cap']:.4f}")
    print(f"  Gini Index   : {freq_results['gini']:.4f}")

    # 3. Analyze Working Sets (Windowed Analysis)
    print("\n[3/5] Analyzing sliding window working set sizes W(w)...")
    ws_results = analyze_working_sets(telemetry, window_sizes=[100, 500, 1000])
    for w_key, w_data in ws_results.items():
        print(f"  {w_key:12s} -> Mean: {w_data['global_mean']:.2f} experts | P90: {w_data['global_p90']:.1f} | P99: {w_data['global_p99']:.1f}")

    # 4. Analyze Transitions & Co-occurrence (Q2, Q3)
    print("\n[4/5] Analyzing temporal & inter-layer transitions (Q2, Q3)...")
    trans_results = analyze_transitions_and_cooccurrence(telemetry)

    # 5. Simulate Expert Caching & NVMe Data Traffic (Q4)
    print("\n[5/5] Simulating Expert Caches across C in [2, 32] (Q4)...")
    capacities = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    simulator = ExpertCacheSimulator(telemetry, freq_results['global_probs'], trans_results['temporal_transition_matrix'])
    cache_sim_results = simulator.simulate(capacities, expert_sizes_mb=[50.0, 200.0, 500.0])

    print("-" * 80)
    print(f"  {'C':4s} | {'Static Hit':10s} | {'LRU Hit':10s} | {'LFU Hit':10s} | {'Markov Hit':10s} | {'NVMe MB/tok (50MB)':20s}")
    print("-" * 80)
    
    lru_hits_dict = {}
    for C in capacities:
        res = cache_sim_results[f'capacity_{C}']
        lru_hits_dict[C] = res['lru_hit_rate']
        traffic_50 = res['nvme_traffic_mb_per_tok']['50MB_expert']
        print(f"  {C:4d} | {res['static_hit_rate']:9.2f}% | {res['lru_hit_rate']:9.2f}% | {res['lfu_hit_rate']:9.2f}% | {res['markov_hit_rate']:9.2f}% | {traffic_50:18.4f} MB")
    print("-" * 80)

    # 6. Evaluate 50ms Latency Budget (Q5)
    budget_results = evaluate_latency_budget(lru_hits_dict, expert_size_mb=50.0, nvme_bw_gbps=5.0, top_k=config['top_k'], n_layers=5)

    # Compile Final Output JSON
    output_data = {
        'model': args.model,
        'config': config,
        'telemetry_tokens': telemetry['total_tokens'],
        'concentration': freq_results,
        'working_sets': ws_results,
        'transitions': trans_results,
        'cache_simulation': cache_sim_results,
        'latency_budget': budget_results
    }

    results_dir = os.path.join(script_dir, "..", "results", "TR8")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "m2_locality_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nTR-8 results saved successfully to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
