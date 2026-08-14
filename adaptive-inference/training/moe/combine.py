import torch

def combine_expert_outputs(
    flat_out: torch.Tensor,
    token_mask: torch.Tensor,
    gates: torch.Tensor,
    exp_outputs: torch.Tensor
) -> torch.Tensor:
    """
    Accumulates gated expert output tensors into flat output container.
    """
    flat_out[token_mask] += gates.unsqueeze(-1) * exp_outputs
    return flat_out
