import torch
from typing import List, Dict, Tuple

def get_expert_partition_counts(num_experts: int) -> Tuple[int, int, int]:
    """
    Computes the exact number of experts allocated to each domain:
    - Arithmetic: 40%
    - Logic: 35%
    - Language: 25%
    """
    n_arithmetic = int(round(num_experts * 0.40))
    n_logic = int(round(num_experts * 0.35))
    n_language = num_experts - n_arithmetic - n_logic
    
    # Adjust in case rounding leads to empty pools when num_experts is small
    if n_arithmetic == 0 and num_experts >= 1:
        n_arithmetic = 1
        n_language = max(0, num_experts - n_arithmetic - n_logic)
    if n_logic == 0 and num_experts - n_arithmetic >= 1:
        n_logic = 1
        n_language = max(0, num_experts - n_arithmetic - n_logic)
    if n_language == 0 and num_experts - n_arithmetic - n_logic >= 1:
        n_language = 1
        
    # Re-normalize to make sure sum matches num_experts exactly
    total_alloc = n_arithmetic + n_logic + n_language
    if total_alloc != num_experts:
        diff = num_experts - total_alloc
        n_arithmetic += diff # Add difference to arithmetic
        
    return n_arithmetic, n_logic, n_language


def get_expert_ranges(num_experts: int) -> Dict[str, Tuple[int, int]]:
    """
    Returns the start and end expert indices (exclusive) for each domain.
    """
    n_arith, n_logic, n_lang = get_expert_partition_counts(num_experts)
    
    ranges = {
        'arithmetic': (0, n_arith),
        'logic': (n_arith, n_arith + n_logic),
        'language': (n_arith + n_logic, num_experts)
    }
    return ranges


def get_expert_domain(expert_id: int, num_experts: int) -> str:
    """
    Returns the domain name for a given expert ID.
    """
    ranges = get_expert_ranges(num_experts)
    for domain, (start, end) in ranges.items():
        if start <= expert_id < end:
            return domain
    raise ValueError(f"Expert ID {expert_id} out of range for {num_experts} experts.")


def get_domain_mask(domain: str, num_experts: int, device: torch.device = None) -> torch.Tensor:
    """
    Returns a binary mask of shape (num_experts,) where 1 indicates that
    the expert belongs to the given domain.
    """
    mask = torch.zeros(num_experts, device=device)
    ranges = get_expert_ranges(num_experts)
    
    if domain in ranges:
        start, end = ranges[domain]
        mask[start:end] = 1.0
    return mask


def get_domain_mask_batch(domains: List[str], num_experts: int, device: torch.device = None) -> torch.Tensor:
    """
    Returns a binary mask of shape (batch_size, num_experts) for a batch of domains.
    """
    batch_size = len(domains)
    mask = torch.zeros(batch_size, num_experts, device=device)
    ranges = get_expert_ranges(num_experts)
    
    for i, domain in enumerate(domains):
        if domain in ranges:
            start, end = ranges[domain]
            mask[i, start:end] = 1.0
    return mask
