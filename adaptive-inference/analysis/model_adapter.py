"""
analysis/model_adapter.py — Real PyTorch inference adapter for MoETransformer evaluation.

Converts an expert subset S = {e_1, ..., e_k} into a routing mask tensor `mask_S`
and evaluates per-example quality Q_i(S) over dataset samples using autoregressive decoding.
"""

import sys
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if HAS_TORCH:
    from training.models.transformer import MoETransformer
    from tasks.tokenizer import CharTokenizer
    from training.train import generate_autoregressive


def create_expert_mask(subset_S: Tuple[int, ...], num_experts: int, device: Any) -> Any:
    """
    Creates a 1D/2D boolean mask tensor where mask[e] = 1.0 if e in subset_S else 0.0.
    Shape: (1, num_experts)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for create_expert_mask")
    mask = torch.zeros((1, num_experts), dtype=torch.float32, device=device)
    for exp_idx in subset_S:
        if 0 <= exp_idx < num_experts:
            mask[0, exp_idx] = 1.0
    return mask


class MoEModelAdapter:
    """
    Adapter to run real PyTorch inference on MoETransformer with explicit expert subset masks.
    """
    def __init__(self, model: Any, tokenizer: Any, device: Optional[Any] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MoEModelAdapter")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.num_experts = model.blocks[0].ffn.num_experts if model.moe else 1


    def eval_subset_quality_per_example(
        self,
        samples: List[Dict[str, Any]],
        subset_S: Tuple[int, ...],
        max_gen_len: int = 30
    ) -> np.ndarray:
        """
        Evaluates Exact-Match Quality Q_i(S) for each individual sample in `samples`
        under the forced routing mask for expert subset `subset_S`.

        Returns:
            np.ndarray of shape (len(samples),) with 1.0 for exact match, 0.0 otherwise.
        """
        self.model.eval()
        scores = []
        mask_S = create_expert_mask(subset_S, self.num_experts, self.device) if self.model.moe else None

        with torch.no_grad():
            for sample in samples:
                prompt_str = sample["input"]
                target_str = sample["target"]

                prompt_encoded = self.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
                gen_ids = generate_autoregressive(
                    self.model,
                    prompt_encoded,
                    self.tokenizer.sep_id,
                    self.tokenizer.eos_id,
                    mask_S,
                    max_gen_len=max_gen_len
                )
                gen_str = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                is_exact_match = 1.0 if gen_str.strip() == target_str.strip() else 0.0
                scores.append(is_exact_match)

        return np.array(scores, dtype=np.float64)

    def eval_subset_mean_quality(
        self,
        samples: List[Dict[str, Any]],
        subset_S: Tuple[int, ...],
        max_gen_len: int = 30
    ) -> float:
        """
        Returns the mean quality Q(S) over all samples in `samples`.
        """
        scores = self.eval_subset_quality_per_example(samples, subset_S, max_gen_len=max_gen_len)
        return float(np.mean(scores))
