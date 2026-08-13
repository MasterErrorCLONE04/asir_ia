import torch
import torch.nn as nn
from typing import Dict, Any

class MemoryManager:
    """
    Utility for inspecting VRAM limits and precision conversion safety.
    """
    @staticmethod
    def convert_precision_selective(model: nn.Module, storage_precision: str) -> nn.Module:
        """
        Converts model parameters to specified storage precision (e.g. 'bf16-storage').
        Maintains LayerNorm, Embedding, and Router parameters in FP32 for numerical stability.
        """
        if storage_precision == "bf16-storage":
            target_dtype = torch.bfloat16
            for name, param in model.named_parameters():
                # Keep LayerNorm, Embedding, and Router weights in FP32
                if any(k in name.lower() for k in ["ln_", "ln1", "ln2", "norm", "embed", "router"]):
                    param.data = param.data.to(torch.float32)
                else:
                    param.data = param.data.to(target_dtype)
        elif storage_precision == "fp32":
            model = model.to(torch.float32)
            
        return model
