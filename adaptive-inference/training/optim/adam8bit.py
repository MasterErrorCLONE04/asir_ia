import torch
import torch.nn as nn
from typing import Iterator, Any

def get_adam8bit_optimizer(params: Iterator[nn.Parameter], lr: float = 5e-4, weight_decay: float = 0.01) -> Any:
    """
    8-bit AdamW optimizer via bitsandbytes library.
    Falls back gracefully or raises an explicit error if bitsandbytes is not installed.
    """
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for --optimizer adam8bit. "
            "Please install it using: pip install bitsandbytes"
        )
