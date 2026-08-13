import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional

# Add the adaptive-inference folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.tokenizer import CharTokenizer
from training.models.transformer import MoETransformer
from training.router.oracle import get_domain_mask_batch
from training.losses.losses import autoregressive_cross_entropy_loss, load_balancing_loss

class SyntheticDataset(Dataset):
    """
    Loads synthetic task datasets from JSONL, tokenizes inputs and targets,
    and prepares sequence masks for autoregressive training.
    """
    def __init__(self, filepath: str, tokenizer: CharTokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        prompt = sample['input']
        target = sample['target']
        domain = sample['domain']
        
        # Tokenize components
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        sep_id = [self.tokenizer.sep_id]
        target_ids = self.tokenizer.encode(target, add_bos=False, add_eos=True)
        
        # Combined sequence
        input_ids = prompt_ids + sep_id + target_ids
        
        # Sequence lengths
        prompt_len = len(prompt_ids) + len(sep_id)
        target_len = len(target_ids)
        total_len = len(input_ids)
        
        # Create labels: mask out the prompt tokens so the model only learns to predict target tokens
        labels = [self.tokenizer.pad_id] * prompt_len + target_ids
        
        # Check sequence length limits
        if total_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'domain': domain,
            'prompt': prompt,
            'target': target
        }


def collate_fn(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    """
    Collates samples into padded batches.
    """
    input_ids_list = [s['input_ids'] for s in batch]
    labels_list = [s['labels'] for s in batch]
    domains = [s['domain'] for s in batch]
    prompts = [s['prompt'] for s in batch]
    targets = [s['target'] for s in batch]
    
    # Pad sequences to max length in the batch
    padded_input_ids = nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    padded_labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=pad_id)
    
    return {
        'input_ids': padded_input_ids,
        'labels': padded_labels,
        'domains': domains,
        'prompts': prompts,
        'targets': targets
    }


def _compute_backbone_params(
    d_model: int,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
    num_experts: int,
    is_moe: bool,
) -> int:
    """
    Analytically computes B: all always-active (non-sparse) parameters.

    For MoE models the router is counted in B, because it executes for every
    token — it is not sparse (§2 v0.7 decision: router ∈ B).  The router size
    scales with N_total (d_model × num_experts per layer), so B differs across
    M1–M5 and C1–C4, and d_ff is adjusted accordingly to keep A = 140M.

    Components:
      Embedding block : token_embed + pos_embed + ln_embed
      Per layer       : ln1 + attention(q,k,v,out projections) + ln2
                        [+ router if MoE]
      Output block    : ln_out + lm_head (weight + bias)
    """
    # Embedding block
    B = (
        vocab_size * d_model           # token_embed  (Embedding)
        + max_seq_len * d_model        # pos_embed    (Embedding)
        + 2 * d_model                  # ln_embed     (weight + bias)
    )

    # Per-layer shared params
    per_layer = (
        2 * d_model                             # ln1 (weight + bias)
        + 4 * (d_model * d_model + d_model)     # q,k,v,out Linear (weight + bias each)
        + 2 * d_model                           # ln2 (weight + bias)
    )
    if is_moe:
        per_layer += d_model * num_experts      # router Linear (bias=False)
    B += n_layers * per_layer

    # Output block
    B += 2 * d_model                            # ln_out (weight + bias)
    B += vocab_size * d_model + vocab_size      # lm_head (weight + bias)

    return B


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Returns architecture hyperparameters for model_name, with A = B + K·E = 140M.

    §4 v0.7 definitions (frozen):
      B  = backbone active params — all shared, always-active components:
           embeddings + attention + layer norms + router (router ∈ B, §2 v0.7).
           B varies with N_total because router size = d_model × N_total per layer.
      E  = params of ONE expert across ALL n_layers MoE layers:
           E = n_layers × params(FFNExpert_single_layer)
      K  = top_k (number of experts activated per token)
      A  = B + K·E  (target: exactly 140,000,000)

    For Dense models (M0, Dense-A):
      No router; K = 1; E = all FFN params across n_layers.

    Note on exactness:
      Integer d_ff may produce A slightly below 140M (typically < 0.1% off).
      §2 v0.7: "exactamente construible" means the formula is correctly applied,
      not that integer d_ff can always hit 140M to the last parameter.
      M1–M5 / C1–C4 tolerance: < 0.1%.  Dense-A / M0 tolerance: ≤ 1%.
    """
    CONFIGS: Dict[str, Dict[str, Any]] = {
        # EXP-B / M-family  (§4 table)
        'M0':      {'moe': False, 'num_experts': 1,   'top_k': 1},
        'Dense-A': {'moe': False, 'num_experts': 1,   'top_k': 1},
        'M1':      {'moe': True,  'num_experts': 8,   'top_k': 2},
        'M2':      {'moe': True,  'num_experts': 32,  'top_k': 2},
        'M3':      {'moe': True,  'num_experts': 128, 'top_k': 4},
        'M4':      {'moe': True,  'num_experts': 512, 'top_k': 8},
        'M5':      {'moe': True,  'num_experts': 896, 'top_k': 8},
        # EXP-C (K=2 fixed, N_total varies — §4 EXP-C table)
        'C1':      {'moe': True,  'num_experts': 32,  'top_k': 2},
        'C2':      {'moe': True,  'num_experts': 128, 'top_k': 2},
        'C3':      {'moe': True,  'num_experts': 512, 'top_k': 2},
        'C4':      {'moe': True,  'num_experts': 896, 'top_k': 2},
    }
    if model_name not in CONFIGS:
        raise ValueError(
            f"Unknown model: '{model_name}'. Valid names: {sorted(CONFIGS)}"
        )

    cfg = CONFIGS[model_name].copy()

    # Fixed architectural constants shared across all configs
    d_model     = 1408
    n_layers    = 5
    max_seq_len = 512
    vocab_size  = 100       # CharTokenizer: 5 special + 95 printable ASCII
    TARGET_A    = 140_000_000

    is_moe      = cfg['moe']
    num_experts = cfg['num_experts']
    K           = cfg['top_k']

    # --- Step 1: compute B analytically (router included for MoE) ---
    B = _compute_backbone_params(
        d_model, n_layers, vocab_size, max_seq_len, num_experts, is_moe
    )

    # --- Step 2: solve for d_ff so that A = B + K*E = TARGET_A ---
    # FFNExpert per layer: in_proj + out_proj (both with bias)
    #   params_per_layer = 2*d_model*d_ff + d_ff + d_model
    #                    = (2*d_model + 1)*d_ff + d_model
    # E = n_layers * params_per_layer
    # A = B + K * E
    numerator   = TARGET_A - B - K * n_layers * d_model
    denominator = K * n_layers * (2 * d_model + 1)
    d_ff        = round(numerator / denominator)

    # --- Step 3: compute actual A with integer d_ff ---
    E_per_layer = 2 * d_model * d_ff + d_ff + d_model   # FFNExpert (in+out with bias)
    E           = n_layers * E_per_layer                 # one expert, ALL layers
    A           = B + K * E

    cfg.update({
        'd_ff':       d_ff,
        # §4 contract fields — must be reported in every result table:
        'B':          B,
        'K':          K,
        'E':          E,
        'A':          A,
        # Metadata for verification / reporting
        'd_model':    d_model,
        'n_layers':   n_layers,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'TARGET_A':   TARGET_A,
    })
    return cfg


def count_active_params(model: nn.Module) -> Dict[str, int]:
    """
    Computes B, E, K, A from the instantiated model, verifying A = B + K·E.

    §4 v0.7 definitions:
      B = backbone params (everything except expert FFN weights)
          Router is counted in B because it is always active (§2 v0.7).
      E = params of ONE expert across ALL MoE layers
          = n_moe_layers × params(single FFNExpert)
      K = top_k (number of experts activated per token)
      A = B + K·E  (active params per token)

    For Dense models (M0, Dense-A):
      K = 1; E = all FFN weights across n_layers; A = B + E = total.

    Returns:
      dict with keys: 'total', 'B', 'K', 'E', 'A'
    """
    is_moe = model.moe
    total  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_moe:
        # Experts are sparse; router stays in B (always active).
        single_expert_size: Optional[int] = None
        all_expert_params  = 0
        n_moe_layers       = 0
        K: Optional[int]   = None

        for block in model.blocks:
            if block.moe:
                moe = block.ffn
                n_moe_layers += 1
                K    = moe.top_k
                e_sz = sum(
                    p.numel() for p in moe.experts[0].parameters() if p.requires_grad
                )
                if single_expert_size is None:
                    single_expert_size = e_sz
                else:
                    assert e_sz == single_expert_size, (
                        f"Expert size mismatch across layers: {e_sz} != {single_expert_size}"
                    )
                all_expert_params += moe.num_experts * e_sz

        # B = total minus all expert weights (router already included in remainder)
        B = total - all_expert_params
        # E = one expert, all MoE layers
        E = n_moe_layers * single_expert_size
        A = B + K * E

    else:
        # Dense: one FFN per layer, always active.
        ffn_total = sum(
            p.numel()
            for block in model.blocks
            for p in block.ffn.parameters()
            if p.requires_grad
        )
        B = total - ffn_total
        E = ffn_total   # all layers
        K = 1
        A = B + K * E   # == total

        assert A == total, (
            f"Dense model: A={A:,} != total={total:,}. Parameter accounting error."
        )

    return {'total': total, 'B': B, 'K': K, 'E': E, 'A': A}


@torch.no_grad()
def generate_autoregressive(model: MoETransformer, prompt_ids: List[int], sep_id: int, eos_id: int,
                            mask: Optional[torch.Tensor], max_gen_len: int = 30) -> List[int]:
    """
    Generates tokens autoregressively using greedy decoding.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Input starts with prompt and separator
    curr_ids = prompt_ids + [sep_id]
    input_tensor = torch.tensor([curr_ids], dtype=torch.long, device=device)
    
    generated_ids = []
    
    for _ in range(max_gen_len):
        # We only pass input_tensor to model. Max sequence length is handled inside model.
        if input_tensor.size(1) > model.max_seq_len:
            break
            
        # Model forward
        logits, _ = model(input_tensor, mask)
        
        # Predict next token (greedy)
        next_token = torch.argmax(logits[0, -1, :]).item()
        
        if next_token == eos_id:
            break
            
        generated_ids.append(next_token)
        # Append to input tensor
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
        
    return generated_ids


def evaluate(model: MoETransformer, dataloader: DataLoader, tokenizer: CharTokenizer,
             controlled: bool, device: torch.device) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    Evaluates the model on cross-entropy loss, quality Q (exact-match),
    and collects expert routing statistics (N_eff, η_cap).
    """
    model.eval()
    total_loss = 0.0
    correct_matches = 0
    total_samples = 0
    
    # Routing statistics
    # Initialize expert selection counts for each MoE layer
    expert_counts = []
    if model.moe:
        for block in model.blocks:
            if block.moe:
                expert_counts.append(torch.zeros(block.ffn.num_experts, device=device))
                
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            domains = batch['domains']
            prompts = batch['prompts']
            targets = batch['targets']
            
            # Domain mask for Controlled MoE
            batch_mask = None
            if controlled and model.moe:
                # Retrieve mask matching this batch's domains
                num_experts = model.blocks[0].ffn.num_experts
                batch_mask = get_domain_mask_batch(domains, num_experts, device)
                
            # Forward pass
            logits, all_router_logits = model(input_ids, batch_mask)
            
            # Loss calculation
            loss = autoregressive_cross_entropy_loss(logits, labels, tokenizer.pad_id)
            total_loss += loss.item() * input_ids.size(0)
            
            # Autoregressive generation for Quality Q evaluation
            for i in range(input_ids.size(0)):
                prompt_str = prompts[i]
                target_str = targets[i]
                domain = domains[i]
                
                # Setup specific single mask if controlled
                single_mask = None
                if controlled and model.moe:
                    single_mask = get_domain_mask_batch([domain], num_experts, device)
                    
                prompt_encoded = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
                gen_ids = generate_autoregressive(model, prompt_encoded, tokenizer.sep_id, tokenizer.eos_id, single_mask)
                gen_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                # Exact-match comparison
                if gen_str.strip() == target_str.strip():
                    correct_matches += 1
                total_samples += 1
                
            # Collect expert routing counts
            if model.moe and all_router_logits:
                # Loop over MoE layers
                for layer_idx, router_logits in enumerate(all_router_logits):
                    # router_logits shape: (batch_size, seq_len, num_experts)
                    flat_logits = router_logits.view(-1, router_logits.size(-1))
                    
                    # For each token, find which expert was selected in top-k
                    top_k = model.blocks[0].ffn.top_k
                    _, topk_indices = torch.topk(flat_logits, top_k, dim=-1) # (T, top_k)
                    
                    # Add counts
                    expert_counts[layer_idx].scatter_add_(
                        0, topk_indices.view(-1), torch.ones(flat_logits.size(0) * top_k, device=device)
                    )
                    
    # Metrics
    avg_loss = total_loss / total_samples
    quality_q = (correct_matches / total_samples) * 100.0 # percentage
    
    # Calculate N_eff, η_cap per layer
    routing_stats = {}
    if model.moe:
        n_eff_list = []
        eta_cap_list = []
        
        for layer_idx, counts in enumerate(expert_counts):
            total_assignments = counts.sum().item()
            num_experts = counts.size(0)
            if total_assignments > 0:
                p = counts / total_assignments
                n_eff = 1.0 / (torch.sum(p ** 2).item() + 1e-9)
                eta_cap = n_eff / num_experts
            else:
                n_eff = 1.0
                eta_cap = 1.0 / num_experts
                
            n_eff_list.append(n_eff)
            eta_cap_list.append(eta_cap)
            
        routing_stats['n_eff_per_layer'] = n_eff_list
        routing_stats['eta_cap_per_layer'] = eta_cap_list
        # Average across all layers
        avg_n_eff = sum(n_eff_list) / len(n_eff_list)
        avg_eta_cap = sum(eta_cap_list) / len(eta_cap_list)
    else:
        avg_n_eff = 1.0
        avg_eta_cap = 1.0
        routing_stats['n_eff_per_layer'] = [1.0]
        routing_stats['eta_cap_per_layer'] = [1.0]
        
    return avg_loss, quality_q, avg_n_eff, avg_eta_cap, routing_stats


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate adaptive sparse models.")
    parser.add_argument("--model", type=str, default="M1", help="Model name (M0-M5, C1-C4).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--controlled", action="store_true", help="Enable Controlled MoE (Oracle masked routing).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for training reproducibility.")
    parser.add_argument("--aux_coef", type=float, default=0.01, help="Coefficient for load-balancing loss.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps.")
    
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Tokenizer
    tokenizer = CharTokenizer()

    # Load datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "..", "tasks", "train_set", "train.jsonl")
    val_file = os.path.join(script_dir, "..", "tasks", "eval_set", "val.jsonl")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("Dataset files not found. Please run tasks/generate_data.py first.")
        sys.exit(1)

    print("Loading datasets...")
    train_dataset = SyntheticDataset(train_file, tokenizer)
    val_dataset = SyntheticDataset(val_file, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))

    # Initialize model matching exact spec parameters (L=5, D=1408)
    config = get_model_config(args.model)
    print(f"Initializing model {args.model} (moe={config['moe']}, experts={config['num_experts']}, top_k={config['top_k']})...")
    
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

    # Verify Parameter Counts — §4 contract: A = B + K·E
    counts = count_active_params(model)
    delta_rel = abs(counts['A'] - 140_000_000) / 140_000_000
    print(f"Total Parameters:  {counts['total']/1e6:.4f}M")
    print(f"Active params  A:  {counts['A']/1e6:.4f}M  "
          f"(B={counts['B']/1e6:.4f}M  "
          f"K={counts['K']}  "
          f"E={counts['E']/1e6:.4f}M)  "
          f"ΔA_rel={delta_rel*100:.4f}%")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print("Starting training...")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_aux_loss = 0.0
        
        should_stop = False
        steps_in_epoch = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            domains = batch['domains']
            
            # Domain mask for Controlled MoE
            batch_mask = None
            if args.controlled and model.moe:
                num_experts = model.blocks[0].ffn.num_experts
                batch_mask = get_domain_mask_batch(domains, num_experts, device)
                
            # Forward pass
            logits, all_router_logits = model(input_ids, batch_mask)
            
            # Task loss
            task_loss = autoregressive_cross_entropy_loss(logits, labels, tokenizer.pad_id)
            
            # Load-balancing loss (for uncontrolled training in later phases)
            aux_loss = load_balancing_loss(all_router_logits, config['top_k'])
            
            # Combined loss
            loss = task_loss + args.aux_coef * aux_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_task_loss += task_loss.item()
            epoch_aux_loss += aux_loss.item()
            steps_in_epoch += 1
            
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"\nReached max steps limit: {global_step} steps. Stopping training.")
                should_stop = True
                break
            
        epoch_loss /= max(steps_in_epoch, 1)
        epoch_task_loss /= max(steps_in_epoch, 1)
        epoch_aux_loss /= max(steps_in_epoch, 1)
        
        # Validation evaluation
        val_loss, val_q, val_n_eff, val_eta_cap, _ = evaluate(model, val_loader, tokenizer, args.controlled, device)
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss: {epoch_loss:.4f} (Task: {epoch_task_loss:.4f}, Aux: {epoch_aux_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} | Val Quality Q (EM): {val_q:.2f}% | "
              f"N_eff: {val_n_eff:.2f} | η_cap: {val_eta_cap:.4f}")

        if should_stop:
            break

    # Save checkpoint at the end of training
    checkpoint_dir = os.path.join(script_dir, "..", "checkpoints", args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved successfully to: {checkpoint_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()
