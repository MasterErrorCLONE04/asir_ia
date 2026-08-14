import torch
import torch.nn as nn
from typing import Iterator, Any
from training.optim.adamw import get_adamw_optimizer
from training.optim.adam8bit import get_adam8bit_optimizer

def create_optimizer(model: nn.Module, opt_name: str = "adamw", lr: float = 5e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """
    Factory function for instantiating model optimizers.
    
    Supported options:
      - 'adamw': Standard PyTorch FP32 AdamW
      - 'adam8bit': 8-bit AdamW via bitsandbytes
    """
    opt_name = opt_name.lower()
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adamw":
        return get_adamw_optimizer(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam8bit":
        return get_adam8bit_optimizer(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: '{opt_name}'. Valid options: ['adamw', 'adam8bit']")
