import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from training.profiling.dispatch_timing import get_global_timing_profiler

class ReferenceFFNExpert(nn.Module):
    """
    Reference single FFN expert implementation.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.out_proj(self.act(self.in_proj(x))))


class ReferenceMoELayer(nn.Module):
    """
    Reference MoE Layer implementation (Oracle baseline).
    Loops over top-K assignments and expert indices in Python.
    Instrumented with sub-stage timing for router, dispatch, experts, and combine.
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

        # --- Sub-stage 1: Router ---
        if profiler: profiler.start("router")
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        router_logits = self.router(flat_x)
        
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            flat_mask = expanded_mask.reshape(-1, self.num_experts)
            router_logits = router_logits + (1.0 - flat_mask) * -1e9
        if profiler: profiler.stop("router")

        # --- Sub-stage 2: Dispatch ---
        if profiler: profiler.start("dispatch")
        routing_gates = F.softmax(router_logits, dim=-1)
        topk_gates, topk_indices = torch.topk(routing_gates, self.top_k, dim=-1)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)
        flat_out = torch.zeros_like(flat_x)
        if profiler: profiler.stop("dispatch")

        # --- Sub-stage 3: Experts ---
        if profiler: profiler.start("experts")
        expert_outputs_cache = {}
        for k in range(self.top_k):
            gates = topk_gates[:, k]
            indices = topk_indices[:, k]
            
            for exp_idx in range(self.num_experts):
                token_mask = (indices == exp_idx)
                if not token_mask.any():
                    continue
                exp_tokens = flat_x[token_mask]
                exp_out = self.experts[exp_idx](exp_tokens)
                expert_outputs_cache[(k, exp_idx, token_mask)] = (gates[token_mask], exp_out)
        if profiler: profiler.stop("experts")

        # --- Sub-stage 4: Combine ---
        if profiler: profiler.start("combine")
        for (k, exp_idx, token_mask), (exp_gates, exp_out) in expert_outputs_cache.items():
            flat_out[token_mask] += exp_gates.unsqueeze(-1) * exp_out
        if profiler: profiler.stop("combine")

        output = flat_out.view(batch_size, seq_len, d_model)
        return output, router_logits.view(batch_size, seq_len, -1)
