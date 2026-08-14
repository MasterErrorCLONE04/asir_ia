import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from training.moe.reference import ReferenceFFNExpert
from training.profiling.dispatch_timing import get_global_timing_profiler


class BatchedMoEDispatcher(nn.Module):
    r"""
    Batched MoE Dispatcher — Token Packing + Grouped Expert Execution.

    Instead of iterating over experts with boolean masks (SparseMoEDispatcher),
    this implementation:
      1. Sorts tokens by expert assignment using argsort (one GPU op per top-k slot).
      2. Computes expert boundary offsets from cumulative counts.
      3. Slices contiguous token segments per expert from the sorted buffer.
      4. Scatters gated outputs back to original token positions.

    This reduces Python-level iteration overhead and produces contiguous memory
    access patterns for expert GEMMs, enabling better GPU SM utilization.

    Preserves exact mathematical equivalence with ReferenceMoELayer and
    SparseMoEDispatcher: same router, same top-k, same gate normalization,
    same expert weights, same output shape, same backward behavior.
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
        num_tokens = flat_x.shape[0]
        router_logits = self.router(flat_x)

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            flat_mask = expanded_mask.reshape(-1, self.num_experts)
            router_logits = router_logits + (1.0 - flat_mask) * -1e9
        if profiler: profiler.stop("router")

        # --- Stage 2: Dispatch (Top-K + Gate Normalization) ---
        if profiler: profiler.start("dispatch")
        routing_gates = F.softmax(router_logits, dim=-1)
        topk_gates, topk_indices = torch.topk(routing_gates, self.top_k, dim=-1)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)

        flat_out = torch.zeros_like(flat_x)
        if profiler: profiler.stop("dispatch")

        # --- Stage 3: Sort/Pack + Expert Execution ---
        if profiler: profiler.start("experts")
        for k in range(self.top_k):
            indices_k = topk_indices[:, k]  # (num_tokens,)
            gates_k = topk_gates[:, k]      # (num_tokens,)

            # Sort tokens by expert assignment: single argsort on GPU
            sorted_order = torch.argsort(indices_k, stable=True)
            sorted_expert_ids = indices_k[sorted_order]
            sorted_tokens = flat_x[sorted_order]     # contiguous sorted buffer
            sorted_gates = gates_k[sorted_order]

            # Compute expert boundary offsets via bincount
            expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)

            # Accumulate offsets
            offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x.device)
            torch.cumsum(expert_counts, dim=0, out=offsets[1:])

            # Execute experts on contiguous slices
            sorted_out = torch.zeros_like(sorted_tokens)
            for exp_idx in range(self.num_experts):
                start = offsets[exp_idx].item()
                end = offsets[exp_idx + 1].item()
                if start == end:
                    continue  # no tokens for this expert
                exp_tokens = sorted_tokens[start:end]
                sorted_out[start:end] = self.experts[exp_idx](exp_tokens)

            # Gate and scatter back to original positions
            gated_out = sorted_gates.unsqueeze(-1) * sorted_out
            flat_out.scatter_add_(0, sorted_order.unsqueeze(-1).expand_as(gated_out), gated_out)

        if profiler: profiler.stop("experts")

        output = flat_out.view(batch_size, seq_len, d_model)
        return output, router_logits.view(batch_size, seq_len, -1)
