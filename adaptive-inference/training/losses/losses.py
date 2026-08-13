import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def autoregressive_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Computes standard autoregressive cross-entropy loss, shifting logits and targets.
    Ignore PAD tokens and tokens before the target separator.
    """
    # Shift logits and targets so that token at index i predicts token at index i+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    
    # Flatten to (batch_size * (seq_len - 1), vocab_size) and (batch_size * (seq_len - 1))
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_targets = shift_targets.view(-1)
    
    # Use PyTorch CrossEntropyLoss ignoring pad_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    return loss_fn(flat_logits, flat_targets)


def load_balancing_loss(all_router_logits: List[torch.Tensor], top_k: int) -> torch.Tensor:
    """
    Computes the auxiliary load-balancing loss (Switch Transformer style) across all MoE layers.
    
    all_router_logits: List of Tensors of shape (batch_size, seq_len, num_experts)
    """
    if not all_router_logits:
        return torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    total_loss = 0.0
    
    for router_logits in all_router_logits:
        # Flatten across batch and sequence: (T, num_experts)
        flat_logits = router_logits.view(-1, router_logits.size(-1))
        T, num_experts = flat_logits.shape
        
        # Softmax gates: probability of routing each token to each expert
        gates = F.softmax(flat_logits, dim=-1) # (T, num_experts)
        
        # Average probability allocated to each expert: P_i = 1/T * sum_t gates_{t, i}
        P = gates.mean(dim=0) # (num_experts,)
        
        # Hard assignment: count tokens assigned to each expert in their top-k
        _, topk_indices = torch.topk(flat_logits, top_k, dim=-1) # (T, top_k)
        
        # Compute fraction of tokens routed to each expert: f_i = count_i / (T * top_k)
        # Note: we normalize by (T * top_k) so that the sum of f_i is 1
        expert_counts = torch.zeros(num_experts, device=gates.device)
        expert_counts.scatter_add_(0, topk_indices.view(-1), torch.ones(T * top_k, device=gates.device))
        f = expert_counts / (T * top_k + 1e-9) # (num_experts,)
        
        # Auxiliary loss for this layer: L_aux = N * sum_i (f_i * P_i)
        layer_loss = num_experts * torch.sum(f * P)
        total_loss += layer_loss
        
    return total_loss / len(all_router_logits)
