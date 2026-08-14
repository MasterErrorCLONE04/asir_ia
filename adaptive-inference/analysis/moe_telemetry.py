import torch
from typing import Dict, List, Any, Optional

class MoETelemetryTracker:
    """
    ASIR Expert Telemetry Tracker.
    Measures expert selection frequency, transition matrix between steps/tokens,
    and co-occurrence matrix for routing analysis.
    """
    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        self.frequency = torch.zeros(self.num_experts, dtype=torch.long)
        self.co_occurrence = torch.zeros((self.num_experts, self.num_experts), dtype=torch.long)
        self.transition = torch.zeros((self.num_experts, self.num_experts), dtype=torch.long)
        self._prev_indices: Optional[torch.Tensor] = None

    def update(self, topk_indices: torch.Tensor) -> None:
        """
        topk_indices shape: (num_tokens, top_k)
        """
        num_tokens, k = topk_indices.shape
        flat_indices = topk_indices.view(-1).cpu()

        # Update frequency
        for idx in flat_indices:
            self.frequency[idx.item()] += 1

        # Update co-occurrence matrix for tokens with top_k > 1
        if k > 1:
            for t in range(num_tokens):
                token_experts = topk_indices[t].cpu().tolist()
                for i in range(len(token_experts)):
                    for j in range(i + 1, len(token_experts)):
                        e1, e2 = token_experts[i], token_experts[j]
                        self.co_occurrence[e1, e2] += 1
                        self.co_occurrence[e2, e1] += 1

        # Update transition matrix across consecutive token sequences if applicable
        if self._prev_indices is not None and self._prev_indices.shape == topk_indices.shape:
            for t in range(num_tokens):
                prev_e = self._prev_indices[t, 0].item()
                curr_e = topk_indices[t, 0].item()
                self.transition[prev_e, curr_e] += 1

        self._prev_indices = topk_indices.clone().detach()

    def get_metrics(self) -> Dict[str, Any]:
        total_assignments = self.frequency.sum().item()
        p = self.frequency.float() / max(total_assignments, 1)
        n_eff = 1.0 / (torch.sum(p ** 2).item() + 1e-9) if total_assignments > 0 else 1.0

        return {
            'total_assignments': total_assignments,
            'n_eff': n_eff,
            'capacity_utilization': n_eff / self.num_experts,
            'expert_frequencies': self.frequency.tolist(),
            'co_occurrence_matrix': self.co_occurrence.tolist(),
            'transition_matrix': self.transition.tolist()
        }
