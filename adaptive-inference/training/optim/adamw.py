import torch
import torch.nn as nn
from typing import Iterator

def get_adamw_optimizer(params: Iterator[nn.Parameter], lr: float = 5e-4, weight_decay: float = 0.01) -> torch.optim.AdamW:
    """
    Standard PyTorch AdamW optimizer (FP32 state).
    """
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
