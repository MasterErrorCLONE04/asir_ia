import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from training.moe.reference import ReferenceFFNExpert
from training.moe.combine import combine_expert_outputs
from training.profiling.dispatch_timing import get_global_timing_profiler

class SparseMoEDispatcher(nn.Module):
    r"""
    Optimized Sparse MoE Dispatcher.
    Only evaluates experts that actually receive tokens, avoiding redundant $O(K \cdot E)$
    loops over inactive expert masks. Preserves exact mathematical equivalence with ReferenceMoELayer.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            ReferenceFFNExpert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        profiler = get_global_timing_profiler()

        # --- Stage 1: Router ---
        if profiler: profiler.start("router")
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        router_logits = self.router(flat_x)

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            flat_mask = expanded_mask.reshape(-1, self.num_experts)
            router_logits = router_logits + (1.0 - flat_mask) * -1e9
        if profiler: profiler.stop("router")

        # --- Stage 2: Grouped Dispatch ---
        if profiler: profiler.start("dispatch")
        routing_gates = F.softmax(router_logits, dim=-1)
        topk_gates, topk_indices = torch.topk(routing_gates, self.top_k, dim=-1)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)

        flat_out = torch.zeros_like(flat_x)
        if profiler: profiler.stop("dispatch")

        # --- Stage 3: Vectorized / Active Experts Only ---
        if profiler: profiler.start("experts")
        computed_outputs = []
        for k in range(self.top_k):
            gates_k = topk_gates[:, k]
            indices_k = topk_indices[:, k]
            
            # Find unique experts active in this top-k assignment
            active_expert_indices = torch.unique(indices_k)
            for exp_idx_tensor in active_expert_indices:
                exp_idx = exp_idx_tensor.item()
                token_mask = (indices_k == exp_idx)
                exp_tokens = flat_x[token_mask]
                exp_out = self.experts[exp_idx](exp_tokens)
                computed_outputs.append((token_mask, gates_k[token_mask], exp_out))
        if profiler: profiler.stop("experts")

        # --- Stage 4: Combine ---
        if profiler: profiler.start("combine")
        for token_mask, gates_sub, exp_out in computed_outputs:
            combine_expert_outputs(flat_out, token_mask, gates_sub, exp_out)
        if profiler: profiler.stop("combine")

        output = flat_out.view(batch_size, seq_len, d_model)
        return output, router_logits.view(batch_size, seq_len, -1)
