import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from training.moe.reference import ReferenceFFNExpert
from training.profiling.dispatch_timing import get_global_timing_profiler


class BatchedMoEDispatcher(nn.Module):
    r"""
    Batched MoE Dispatcher — Token Packing + Grouped Expert Execution (Batched GEMM).

    This implementation:
      1. Sorts tokens by expert assignment using argsort (one GPU op per top-k slot).
      2. Packs tokens into a batched 3D tensor `(num_experts, max_tokens, d_model)`.
      3. Executes ALL experts in a single batched GEMM call via `torch.bmm`.
      4. Scatters gated outputs back to original token positions.

    Reduces Python-level iteration overhead and CUDA kernel launches from O(N_experts * K)
    to O(K) batched operations per layer step.

    Preserves exact mathematical equivalence with ReferenceMoELayer and
    SparseMoEDispatcher: same router, same top-k, same gate normalization,
    same expert weights, same output shape, same backward behavior.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
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

        # --- Stage 3: Stack Expert Weights & Execute Grouped GEMM ---
        if profiler: profiler.start("experts")

        # Stack weights and biases across all experts
        w1_weight = torch.stack([exp.in_proj.weight.t() for exp in self.experts])  # (E, d_model, d_ff)
        w1_bias = torch.stack([exp.in_proj.bias for exp in self.experts]).unsqueeze(1)  # (E, 1, d_ff)

        w2_weight = torch.stack([exp.out_proj.weight.t() for exp in self.experts])  # (E, d_ff, d_model)
        w2_bias = torch.stack([exp.out_proj.bias for exp in self.experts]).unsqueeze(1)  # (E, 1, d_model)

        for k in range(self.top_k):
            indices_k = topk_indices[:, k]  # (num_tokens,)
            gates_k = topk_gates[:, k]      # (num_tokens,)

            # Sort tokens by expert assignment
            sorted_order = torch.argsort(indices_k, stable=True)
            sorted_expert_ids = indices_k[sorted_order]
            sorted_tokens = flat_x[sorted_order]
            sorted_gates = gates_k[sorted_order]

            # Compute expert counts and max tokens per expert
            expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
            max_tokens_per_exp = expert_counts.max().item()

            if max_tokens_per_exp == 0:
                continue

            # Offsets per expert
            offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x.device)
            torch.cumsum(expert_counts, dim=0, out=offsets[1:])

            # Local index within each expert slot
            local_pos = torch.arange(num_tokens, device=x.device) - offsets[sorted_expert_ids]

            # Construct 3D packed input buffer: (num_experts, max_tokens, d_model)
            packed_input = torch.zeros(
                (self.num_experts, max_tokens_per_exp, d_model),
                dtype=flat_x.dtype,
                device=x.device
            )
            packed_input[sorted_expert_ids, local_pos] = sorted_tokens

            # Grouped GEMM 1: (E, M, d_model) x (E, d_model, d_ff) -> (E, M, d_ff)
            h1 = torch.bmm(packed_input, w1_weight) + w1_bias
            h1_act = F.gelu(h1)

            # Grouped GEMM 2: (E, M, d_ff) x (E, d_ff, d_model) -> (E, M, d_model)
            h2 = torch.bmm(h1_act, w2_weight) + w2_bias

            # Gather outputs back for assigned sorted tokens
            sorted_out = h2[sorted_expert_ids, local_pos]

            # Gate and scatter add to final tensor
            gated_out = sorted_gates.unsqueeze(-1) * sorted_out
            flat_out.scatter_add_(0, sorted_order.unsqueeze(-1).expand_as(gated_out), gated_out)

        if profiler: profiler.stop("experts")

        output = flat_out.view(batch_size, seq_len, d_model)
        return output, router_logits.view(batch_size, seq_len, -1)

