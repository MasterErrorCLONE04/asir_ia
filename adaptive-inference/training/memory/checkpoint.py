import torch
import torch.nn as nn
from typing import Optional

def apply_gradient_checkpointing(model: nn.Module, enabled: bool = True) -> nn.Module:
    """
    Enables or disables activation gradient checkpointing on transformer blocks.
    When enabled, intermediate activations are recomputed during backward pass
    to significantly reduce peak VRAM consumption.
    """
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = enabled
    elif hasattr(model, "blocks"):
        for block in model.blocks:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = enabled
    return model
