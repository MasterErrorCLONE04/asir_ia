"""
analysis/moe_profiler.py — Router profiling module on Split A to derive S_selector.

Passes Dataset A examples through unmasked MoETransformer, accumulates router activation
counts/logits across all MoE layers, and selects the top-k experts as S_selector.
"""

import sys
import os
from typing import List, Dict, Any, Tuple, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if HAS_TORCH:
    from training.models.transformer import MoETransformer
    from tasks.tokenizer import CharTokenizer


def profile_selector_on_split_a(
    model: Any,
    a_samples: List[Dict[str, Any]],
    k: int,
    tokenizer: Any,
    device: Optional[Any] = None
) -> Tuple[int, ...]:
    """
    Profiles the router on Dataset A with unmasked routing.
    Returns the top-k expert IDs with the highest accumulated selection frequency.

    Args:
        model: MoETransformer instance.
        a_samples: List of sample dicts from Split A.
        k: Number of experts to select.
        tokenizer: CharTokenizer instance.
        device: PyTorch device.

    Returns:
        Tuple of k expert indices sorted lexicographically.
    """
    model.eval()
    dev = device if device is not None else next(model.parameters()).device

    if not model.moe:
        # Non-MoE baseline model fallback
        return tuple(range(min(k, 1)))

    num_experts = model.blocks[0].ffn.num_experts
    accumulated_counts = torch.zeros(num_experts, device=dev)

    with torch.no_grad():
        for sample in a_samples:
            prompt_str = sample["input"]
            target_str = sample["target"]

            prompt_ids = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
            sep_ids = [tokenizer.sep_id]
            target_ids = tokenizer.encode(target_str, add_bos=False, add_eos=True)

            input_ids = torch.tensor([prompt_ids + sep_ids + target_ids], dtype=torch.long, device=dev)

            # Forward pass without mask
            _, all_router_logits = model(input_ids, mask=None)

            for layer_logits in all_router_logits:
                # layer_logits shape: (1, seq_len, num_experts)
                flat_logits = layer_logits.view(-1, num_experts)
                top_k_layer = model.blocks[0].ffn.top_k
                _, topk_indices = torch.topk(flat_logits, top_k_layer, dim=-1)

                accumulated_counts.scatter_add_(
                    0,
                    topk_indices.view(-1),
                    torch.ones(topk_indices.numel(), device=dev)
                )

    # Pick top-k experts by accumulated counts
    topk_vals, topk_experts = torch.topk(accumulated_counts, k)
    selected_experts = tuple(sorted(topk_experts.cpu().tolist()))
    return selected_experts
