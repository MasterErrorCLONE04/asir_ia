"""
training/run_experiments.py — Reproducible training & evaluation pipeline (R1.0)

Generates config hashes, trains the model, saves checkpoints, evaluates on a
fixed evaluation set, and logs comprehensive metrics (§6, §9, §10, §11).

Metrics computed on evaluation:
  - Quality Q (exact-match accuracy)
  - Autoregressive Cross-Entropy Loss
  - N_effective (aggregated over the entire eval set)
  - η_cap (effective capacity utilization efficiency)
  - Expert-to-role Pearson correlation matrix per layer
  - Many-to-one alignment f: E -> R ∪ {∅} for tau in {0.30, 0.50, 0.70}
  - Specialization Density (SD)
  - Unassigned routing mass
  - Q_role_renorm role distribution
  - Routing accuracy top-k (correctness over many-to-one aligned roles)
  - ECD: average forward time, peak memory, throughput

Usage:
    python training/run_experiments.py --model M1 --epochs 5 --seed 42
"""

import os
import sys
from typing import List, Dict, Tuple, Optional, Any
import json
import time
import argparse
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.tokenizer import CharTokenizer
from training.models.transformer import MoETransformer
from training.router.oracle import get_domain_mask_batch, get_expert_ranges
from training.losses.losses import autoregressive_cross_entropy_loss, load_balancing_loss
from training.train import SyntheticDataset, collate_fn, get_model_config, count_active_params, generate_autoregressive
from training.router.alignment import (
    ROLES,
    assign_experts_to_roles,
    specialization_density,
    unassigned_routing_mass,
    Q_role_aligned,
    compute_expert_role_correlation,
)

# ---------------------------------------------------------------------------
# Peak Memory Fallback (reads /proc/self/status under Linux/Docker)
# ---------------------------------------------------------------------------

def get_peak_memory_mb() -> float:
    """
    Returns the peak resident set size (RSS) in MB for the current process.
    Supports Linux (/proc/self/status) with a safe fallback to 0.0.
    """
    try:
        if os.path.exists('/proc/self/status'):
            with open('/proc/self/status', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('VmPeak:'):
                        # VmPeak:     284520 kB
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0
        # If CUDA is initialized, track CUDA memory
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Routing Accuracy computation over many-to-one aligned roles
# ---------------------------------------------------------------------------

def compute_routing_accuracy(
    all_router_logits: List[torch.Tensor],
    input_ids: torch.Tensor,
    domains: List[str],
    assignments: List[Dict[int, Optional[str]]],
    pad_id: int,
) -> List[float]:
    """
    Calculates the routing accuracy top-k for each MoE layer.
    
    A routing decision for a token of domain 'd' is correct if the domain 'd'
    is among the assigned roles of the top-k experts selected for that token
    (ignoring padding tokens).

    Args:
        all_router_logits: List of L tensors of shape (B, T, N_total).
        input_ids:         Input token IDs of shape (B, T) to detect padding.
        domains:           List of B domain strings (one per sample in batch).
        assignments:       List of L dicts mapping expert_id -> role | None.
        pad_id:            Padding token ID to ignore.

    Returns:
        accuracy_per_layer: List of L floats (fraction correct).
    """
    L = len(all_router_logits)
    accuracy_per_layer = []

    # Domain indices to map string -> string directly
    # domains shape is (B,) representing domain of each sentence
    B, T = input_ids.shape

    for l in range(L):
        logits = all_router_logits[l] # (B, T, N_total)
        top_k = logits.shape[-1]
        
        # Determine top-k expert indices
        # We need block properties, but we can infer top-k from logits directly
        # Let's count top_k dynamically. Actually we can do topk over the logits
        # Wait, MoELayer selects top_k. But we don't know top_k unless we inspect it.
        # Let's get the top_k index size.
        # For our models, top_k is known from the config or model.
        # Let's pass top_k or infer it from the layer's actual config.
        # Wait, we can just look at how many experts were selected.
        # But logits has shape (B, T, N_total). The routing decision indices
        # are calculated by taking torch.topk on logits.
        # Let's use the actual top_k value of this layer.
        
        # We get the assignment dict for this layer
        assignment = assignments[l]
        
        correct_tokens = 0
        total_tokens = 0
        
        # Let's find top-k indices. We can do top_k from the layer.
        # But we don't have the model here. We can just use the config's top_k.
        # To be safe, we will pass top_k to this function.
        pass

# Actually, let's write the routing accuracy logic cleanly inside the evaluation loop
# where we have access to model.blocks[l].ffn.top_k.
# So we don't need a separate helper function, we will calculate it directly in evaluate_and_log.
# That is much safer!
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training and Evaluation Orchestrator
# ---------------------------------------------------------------------------

def run_experiment(args) -> str:
    # Setup reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{args.model} | Seed {args.seed}] Using device: {device}")

    tokenizer = CharTokenizer()

    # Get model configuration and calculate d_ff (R1.0.1 round-aligned)
    cfg = get_model_config(args.model)
    
    # -----------------------------------------------------------------------
    # 1. Compute Config Hash
    # -----------------------------------------------------------------------
    # Deterministic hash of the exact model parameters and training choices
    config_dict = {
        'model':       args.model,
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
    config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]
    print(f"[{args.model}] Config Hash: {config_hash}")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "..", "tasks", "train_set", "train.jsonl")
    val_file = os.path.join(script_dir, "..", "tasks", "eval_set", "val.jsonl")
    
    checkpoint_dir = os.path.join(script_dir, "..", "checkpoints", "R1.0", config_hash, f"seed_{args.seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")

    results_dir = os.path.join(script_dir, "..", "results", "R1.0", config_hash)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"seed_{args.seed}_results.json")

    # Load datasets
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError("Dataset files not found. Please run tasks/generate_data.py first.")

    train_dataset = SyntheticDataset(train_file, tokenizer)
    val_dataset = SyntheticDataset(val_file, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))

    # Initialize model
    model = MoETransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=1408,
        n_layers=5,
        num_heads=8,
        d_ff=cfg['d_ff'],
        moe=cfg['moe'],
        num_experts=cfg['num_experts'],
        top_k=cfg['top_k']
    ).to(device)

    # Contract verification
    param_counts = count_active_params(model)
    delta_rel = abs(param_counts['A'] - 140_000_000) / 140_000_000

    # -----------------------------------------------------------------------
    # 2. Training Loop (Skip if eval_only)
    # -----------------------------------------------------------------------
    if not args.eval_only:
        print(f"[{args.model}] Starting training for {args.epochs} epochs...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                domains = batch['domains']
                
                batch_mask = None
                if args.controlled and model.moe:
                    num_experts = model.blocks[0].ffn.num_experts
                    batch_mask = get_domain_mask_batch(domains, num_experts, device)
                    
                logits, all_router_logits = model(input_ids, batch_mask)
                task_loss = autoregressive_cross_entropy_loss(logits, labels, tokenizer.pad_id)
                aux_loss = load_balancing_loss(all_router_logits, config_dict['top_k'])
                loss = task_loss + args.aux_coef * aux_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            epoch_loss /= len(train_loader)
            print(f"  Epoch {epoch}/{args.epochs} | Train Loss: {epoch_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[{args.model}] Checkpoint saved to: {checkpoint_path}")
    else:
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[{args.model}] Checkpoint loaded from: {checkpoint_path}")

    # -----------------------------------------------------------------------
    # 3. Fixed Evaluation Set Metric Collection
    # -----------------------------------------------------------------------
    model.eval()
    total_loss = 0.0
    correct_matches = 0
    total_samples = 0

    # ECD measurement initialization
    total_forward_time = 0.0
    total_forward_calls = 0
    total_eval_tokens = 0
    start_eval_time = time.perf_counter()

    # Track domain indices mapping string -> index for correlation
    # roles: ['arithmetic', 'logic', 'language']
    role_map = {role: idx for idx, role in enumerate(ROLES)}

    # Experts counts per layer
    expert_counts = []
    # Co-occurrence counts per layer: expert selection count for each domain (N_total, N_roles)
    co_occurrence_counts = []
    # Total domain tokens count per layer (1D shape N_roles)
    domain_token_counts = []

    if model.moe:
        n_layers_moe = sum(1 for block in model.blocks if block.moe)
        num_experts = model.blocks[0].ffn.num_experts
        for _ in range(n_layers_moe):
            expert_counts.append(torch.zeros(num_experts, dtype=torch.long, device=device))
            co_occurrence_counts.append(torch.zeros((num_experts, len(ROLES)), dtype=torch.long, device=device))
            domain_token_counts.append(torch.zeros(len(ROLES), dtype=torch.long, device=device))

    # Run evaluation pass
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            domains = batch['domains']
            prompts = batch['prompts']
            targets = batch['targets']

            batch_mask = None
            if args.controlled and model.moe:
                num_experts = model.blocks[0].ffn.num_experts
                batch_mask = get_domain_mask_batch(domains, num_experts, device)

            # Track forward pass duration (ECD compute metric)
            f_start = time.perf_counter()
            logits, all_router_logits = model(input_ids, batch_mask)
            f_dur = time.perf_counter() - f_start
            
            total_forward_time += f_dur
            total_forward_calls += 1

            # Ignore pad tokens when computing token throughput
            non_pad_tokens = (input_ids != tokenizer.pad_id).sum().item()
            total_eval_tokens += non_pad_tokens

            # Compute validation cross-entropy loss
            loss = autoregressive_cross_entropy_loss(logits, labels, tokenizer.pad_id)
            total_loss += loss.item() * input_ids.size(0)

            # Exact-match quality Q evaluation
            for i in range(input_ids.size(0)):
                prompt_str = prompts[i]
                target_str = targets[i]
                domain = domains[i]

                single_mask = None
                if args.controlled and model.moe:
                    single_mask = get_domain_mask_batch([domain], num_experts, device)

                prompt_encoded = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
                gen_ids = generate_autoregressive(
                    model, prompt_encoded, tokenizer.sep_id, tokenizer.eos_id, single_mask
                )
                gen_str = tokenizer.decode(gen_ids, skip_special_tokens=True)

                if gen_str.strip() == target_str.strip():
                    correct_matches += 1
                total_samples += 1

            # Accumulate routing stats per token (skipping pad tokens)
            if model.moe and all_router_logits:
                B_batch, T_batch = input_ids.shape
                # Map domain of each sample to role index
                domain_indices = torch.tensor([role_map[d] for d in domains], dtype=torch.long, device=device)

                for layer_idx, router_logits in enumerate(all_router_logits):
                    # router_logits: (B_batch, T_batch, num_experts)
                    top_k = model.blocks[0].ffn.top_k
                    _, topk_indices = torch.topk(router_logits, top_k, dim=-1) # (B_batch, T_batch, top_k)

                    # Iterate over samples and tokens to ignore pad tokens
                    for b_idx in range(B_batch):
                        r_idx = domain_indices[b_idx].item()
                        for t_idx in range(T_batch):
                            if input_ids[b_idx, t_idx].item() == tokenizer.pad_id:
                                continue
                            
                            # Increment domain token count
                            domain_token_counts[layer_idx][r_idx] += 1
                            
                            # Increment selection and co-occurrence counts for top-k experts
                            for k_idx in range(top_k):
                                exp_id = topk_indices[b_idx, t_idx, k_idx].item()
                                expert_counts[layer_idx][exp_id] += 1
                                co_occurrence_counts[layer_idx][exp_id, r_idx] += 1

    eval_duration = time.perf_counter() - start_eval_time
    avg_loss = total_loss / total_samples
    quality_q = (correct_matches / total_samples) * 100.0

    # -----------------------------------------------------------------------
    # 4. Post-process Routing Stats and Alignment (§6 many-to-one, §9)
    # -----------------------------------------------------------------------
    layers_results = {}
    avg_n_eff = 1.0
    avg_eta_cap = 1.0

    if model.moe:
        n_eff_layer_list = []
        eta_cap_layer_list = []
        
        # We also need to compute top-k routing accuracy over the aligned roles.
        # Routing accuracy is calculated as: for each non-pad token of domain r,
        # was domain r present in the assigned roles of the top-k selected experts?
        # Since this depends on f(e) which is computed at the end of the evaluation pass
        # (needs correlation over the full dataset), we must do a second quick pass
        # or calculate it directly from the token routing choices we logged, or simply
        # do it by iterating over val_loader again or simulating it.
        # Since val_loader has 1,001 samples, a second evaluation pass is fast (takes ~2 seconds).
        # We will run a second pass to compute routing accuracy and Q_role mapping.

        # Step 4a: Calculate correlation & many-to-one assignment for each layer
        assignments_by_layer = {}  # layer_idx -> {tau: assignment}
        correlation_by_layer = []

        for layer_idx in range(len(expert_counts)):
            total_tokens_layer = domain_token_counts[layer_idx].sum().item()
            corr = compute_expert_role_correlation(
                co_occurrence_counts[layer_idx],
                expert_counts[layer_idx],
                domain_token_counts[layer_idx],
                total_tokens_layer
            )
            correlation_by_layer.append(corr)

            assignments_by_layer[layer_idx] = {}
            for tau in [0.30, 0.50, 0.70]:
                assignments_by_layer[layer_idx][tau] = assign_experts_to_roles(corr, tau=tau)

        # Step 4b: Second pass to calculate routing accuracy and Q_role distribution
        routing_accuracy_sums = {l: {0.30: 0.0, 0.50: 0.0, 0.70: 0.0} for l in range(len(expert_counts))}
        total_tokens_count = {l: 0 for l in range(len(expert_counts))}

        # For Q_role we need the normalized routing probabilities over the entire evaluation set.
        # Q(e) = expert_counts[e] / total_expert_assignments
        routing_probs_by_layer = []
        for layer_idx in range(len(expert_counts)):
            tot_assign = expert_counts[layer_idx].sum().item()
            if tot_assign > 0:
                routing_probs_by_layer.append(expert_counts[layer_idx].float() / tot_assign)
            else:
                routing_probs_by_layer.append(torch.zeros(num_experts, device=device))

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                domains = batch['domains']
                
                batch_mask = None
                if args.controlled and model.moe:
                    batch_mask = get_domain_mask_batch(domains, num_experts, device)
                
                _, all_router_logits = model(input_ids, batch_mask)
                
                B_batch, T_batch = input_ids.shape
                
                for layer_idx, router_logits in enumerate(all_router_logits):
                    top_k = model.blocks[0].ffn.top_k
                    _, topk_indices = torch.topk(router_logits, top_k, dim=-1)
                    
                    for b_idx in range(B_batch):
                        correct_domain = domains[b_idx]
                        for t_idx in range(T_batch):
                            if input_ids[b_idx, t_idx].item() == tokenizer.pad_id:
                                continue
                            
                            total_tokens_count[layer_idx] += 1
                            
                            # For each tau, check if correct domain is in top-k assigned roles
                            for tau in [0.30, 0.50, 0.70]:
                                current_assignment = assignments_by_layer[layer_idx][tau]
                                selected_roles = []
                                for k_idx in range(top_k):
                                    exp_id = topk_indices[b_idx, t_idx, k_idx].item()
                                    role = current_assignment[exp_id]
                                    if role is not None:
                                        selected_roles.append(role)
                                        
                                if correct_domain in selected_roles:
                                    routing_accuracy_sums[layer_idx][tau] += 1.0

        # Step 4c: Consolidate layer results
        for layer_idx in range(len(expert_counts)):
            tot_tokens = total_tokens_count[layer_idx]
            tot_assign = expert_counts[layer_idx].sum().item()
            
            # N_eff and η_cap (aggregated counts, §9 v0.7)
            if tot_assign > 0:
                p = expert_counts[layer_idx].float() / tot_assign
                n_eff = 1.0 / (torch.sum(p ** 2).item() + 1e-9)
                eta_cap = n_eff / num_experts
            else:
                n_eff = 1.0
                eta_cap = 1.0 / num_experts

            n_eff_layer_list.append(n_eff)
            eta_cap_layer_list.append(eta_cap)

            layer_info = {
                'n_eff':   n_eff,
                'eta_cap': eta_cap,
                'sensitivity': {}
            }

            for tau in [0.30, 0.50, 0.70]:
                asn = assignments_by_layer[layer_idx][tau]
                sd = specialization_density(asn, num_experts, tau=tau)
                
                # Unassigned routing mass
                unassigned_mass = unassigned_routing_mass(asn, routing_probs_by_layer[layer_idx])
                
                # Q_role_aligned renormalization
                qr_aligned = Q_role_aligned(asn, routing_probs_by_layer[layer_idx])
                
                # Routing accuracy top-k
                acc = routing_accuracy_sums[layer_idx][tau] / tot_tokens if tot_tokens > 0 else 0.0

                # Convert expert mapping to serializable dict
                serialized_asn = {str(k): v for k, v in asn.items()}

                layer_info['sensitivity'][f'tau_{tau:.2f}'] = {
                    'specialization_density':  sd,
                    'unassigned_routing_mass': unassigned_mass,
                    'routing_accuracy':        acc,
                    'f_expert_role_alignment': serialized_asn,
                    'Q_role_raw':              qr_aligned['Q_role_raw'],
                    'Q_role_renorm':           qr_aligned['Q_role_renorm'],
                }

            layers_results[f'layer_{layer_idx}'] = layer_info

        avg_n_eff = sum(n_eff_layer_list) / len(n_eff_layer_list)
        avg_eta_cap = sum(eta_cap_layer_list) / len(eta_cap_layer_list)

    else:
        # Dense configs
        layers_results = {}

    # -----------------------------------------------------------------------
    # 5. ECD Metrics (§10)
    # -----------------------------------------------------------------------
    avg_forward_time_ms = (total_forward_time / total_forward_calls) * 1000.0 if total_forward_calls > 0 else 0.0
    peak_mem_mb = get_peak_memory_mb()
    throughput_tps = total_eval_tokens / eval_duration if eval_duration > 0 else 0.0

    ecd_metrics = {
        'average_forward_time_ms': avg_forward_time_ms,
        'peak_memory_mb':          peak_mem_mb,
        'token_throughput_tps':   throughput_tps
    }

    # -----------------------------------------------------------------------
    # 6. Save results JSON
    # -----------------------------------------------------------------------
    results = {
        'phase': "R1.0",
        'config_hash': config_hash,
        'model_name': args.model,
        'seed': args.seed,
        'hyperparameters': config_dict,
        'parameter_contract': {
            'B': param_counts['B'],
            'K': param_counts['K'],
            'E': param_counts['E'],
            'A': param_counts['A'],
            'target_A': param_counts['total'] if not cfg['moe'] else 140_000_000,
            'delta_rel': delta_rel
        },
        'metrics': {
            'val_loss': avg_loss,
            'quality_q': quality_q,
            'avg_n_eff': avg_n_eff,
            'avg_eta_cap': avg_eta_cap,
            'ecd': ecd_metrics,
            'layers': layers_results
        }
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("-" * 80)
    print(f"EXPERIMENT RESULTS FOR {args.model} (SEED {args.seed}):")
    print(f"  Quality Q (EM): {quality_q:.2f}% | Loss: {avg_loss:.4f}")
    if cfg['moe']:
        print(f"  Avg N_eff:      {avg_n_eff:.2f} | Avg η_cap: {avg_eta_cap:.4f}")
        # Print layer 0 specialization info for tau=0.50 as a quick sanity check
        l0_sens = layers_results['layer_0']['sensitivity']['tau_0.50']
        print(f"  Layer 0 SD:     {l0_sens['specialization_density']:.3f} | "
              f"Unassigned mass: {l0_sens['unassigned_routing_mass']:.3f} | "
              f"Routing Acc: {l0_sens['routing_accuracy']*100:.2f}%")
    print(f"  ECD Forward:    {avg_forward_time_ms:.2f} ms | Peak Memory: {peak_mem_mb:.2f} MB | Throughput: {throughput_tps:.1f} tps")
    print(f"  Results saved to: {results_path}")
    print("-" * 80)

    return results_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reproducible pipeline run for R1.0 training & eval.")
    parser.add_argument("--model", type=str, default="M1", help="Model name (M0-M5, C1-C4).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--controlled", action="store_true", help="Enable Controlled MoE routing.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for training reproducibility.")
    parser.add_argument("--aux_coef", type=float, default=0.01, help="Coefficient for load-balancing loss.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation only.")
    
    args = parser.parse_args()
    
    try:
        run_experiment(args)
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
