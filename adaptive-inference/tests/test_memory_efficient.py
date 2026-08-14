import unittest
import torch
import torch.nn as nn

from training.optim.factory import create_optimizer
from training.memory.manager import MemoryManager
from training.memory.checkpoint import apply_gradient_checkpointing
from training.models.transformer import MoETransformer

class TestMemoryEfficientSubsystem(unittest.TestCase):
    """
    CPU-safe unit tests for optimizer factory, precision conversion,
    and gradient checkpointing in ASIR-TR-1.
    """

    def setUp(self):
        self.model = MoETransformer(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            num_heads=2,
            d_ff=128,
            moe=True,
            num_experts=4,
            top_k=2
        )

    def test_optimizer_factory_adamw(self):
        optimizer = create_optimizer(self.model, opt_name="adamw", lr=1e-3)
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_optimizer_factory_invalid(self):
        with self.assertRaises(ValueError):
            create_optimizer(self.model, opt_name="unknown_opt")

    def test_selective_bf16_storage_conversion(self):
        model_converted = MemoryManager.convert_precision_selective(self.model, "bf16-storage")
        
        # Verify Linear layers are bfloat16
        linear_dtypes = [p.dtype for name, p in model_converted.named_parameters() if "ffn" in name and "in_proj" in name]
        self.assertTrue(all(dt == torch.bfloat16 for dt in linear_dtypes))

        # Verify LayerNorm weights remain float32 for numerical stability
        ln_dtypes = [p.dtype for name, p in model_converted.named_parameters() if "ln_" in name or "ln1" in name or "ln2" in name]
        self.assertTrue(all(dt == torch.float32 for dt in ln_dtypes))

    def test_gradient_checkpointing_flag(self):
        self.assertFalse(self.model.gradient_checkpointing)
        apply_gradient_checkpointing(self.model, enabled=True)
        self.assertTrue(self.model.gradient_checkpointing)

if __name__ == "__main__":
    unittest.main()
