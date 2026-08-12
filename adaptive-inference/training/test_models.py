import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.models.transformer import MoETransformer
from training.router.oracle import get_domain_mask_batch
from training.train import get_model_config, count_parameters

class TestModelArchitecture(unittest.TestCase):
    def test_parameter_counts(self):
        """
        Verify that active parameter counts match the spec target (~140M) within <1% relative tolerance.
        """
        vocab_size = 100
        d_model = 1408
        n_layers = 5
        num_heads = 8
        
        # Test configurations M0 to M5
        models_to_test = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
        
        for name in models_to_test:
            config = get_model_config(name)
            model = MoETransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                num_heads=num_heads,
                d_ff=config['d_ff'],
                moe=config['moe'],
                num_experts=config['num_experts'],
                top_k=config['top_k']
            )
            
            total_params, active_params = count_parameters(model)
            
            # Target active parameters: 140M
            target_active = 140e6
            rel_diff = abs(active_params - target_active) / target_active
            
            print(f"Model {name:8s} | Total Params: {total_params/1e6:6.2f}M | Active Params: {active_params/1e6:6.2f}M | Rel Diff: {rel_diff*100:6.4f}%")
            
            # Tolerance is 1%
            self.assertLessEqual(rel_diff, 0.01, f"Active parameters of {name} deviate from 140M by more than 1% ({rel_diff*100:.3f}%).")

    def test_controlled_routing_mask(self):
        """
        Verify that the router mask prevents selection of out-of-domain experts.
        """
        vocab_size = 100
        d_model = 1408
        n_layers = 5
        num_heads = 8
        
        # Use M1 (8 experts, top_k = 2)
        config = get_model_config('M1')
        model = MoETransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            d_ff=config['d_ff'],
            moe=config['moe'],
            num_experts=config['num_experts'],
            top_k=config['top_k']
        )
        
        # Batch size 2, seq_len 4
        input_ids = torch.randint(0, vocab_size, (2, 4))
        
        # Define domains: first is arithmetic, second is language
        domains = ['arithmetic', 'language']
        # Mask shape: (batch_size, num_experts) = (2, 8)
        mask = get_domain_mask_batch(domains, num_experts=8)
        
        # Expert partitions for num_experts=8:
        # Arithmetic: experts 0, 1, 2 (count=3)
        # Logic: experts 3, 4, 5 (count=3)
        # Language: experts 6, 7 (count=2)
        
        # Let's inspect MoELayer forward pass directly
        moe_layer = model.blocks[0].ffn
        
        x = torch.randn(2, 4, d_model)
        output, router_logits = moe_layer(x, mask)
        
        # Let's check which experts were routed to
        flat_logits = router_logits.view(-1, 8)
        # Softmax gates
        gates = torch.softmax(flat_logits, dim=-1)
        
        # First 4 tokens (batch 0) correspond to 'arithmetic' -> experts 0, 1, 2 only
        for token_idx in range(4):
            # Gates for experts 3 to 7 should be exactly 0
            for exp_idx in range(3, 8):
                self.assertAlmostEqual(gates[token_idx, exp_idx].item(), 0.0, places=5)
                
        # Last 4 tokens (batch 1) correspond to 'language' -> experts 6, 7 only
        for token_idx in range(4, 8):
            # Gates for experts 0 to 5 should be exactly 0
            for exp_idx in range(0, 6):
                self.assertAlmostEqual(gates[token_idx, exp_idx].item(), 0.0, places=5)

    def test_backprop_gradients(self):
        """
        Verify that forward and backward passes run without errors, and gradients are computed.
        """
        vocab_size = 100
        d_model = 256  # Smaller for speed in testing
        n_layers = 2
        num_heads = 4
        
        model = MoETransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            d_ff=512,
            moe=True,
            num_experts=4,
            top_k=2
        )
        
        input_ids = torch.randint(0, vocab_size, (2, 10))
        domains = ['arithmetic', 'logic']
        mask = get_domain_mask_batch(domains, num_experts=4)
        
        logits, all_router_logits = model(input_ids, mask)
        
        # Check output shape
        self.assertEqual(logits.shape, (2, 10, vocab_size))
        self.assertEqual(len(all_router_logits), n_layers)
        self.assertEqual(all_router_logits[0].shape, (2, 10, 4))
        
        # Compute dummy loss and backward
        loss = logits.sum()
        loss.backward()
        
        # Verify gradients exist for embeddings and active router
        self.assertIsNotNone(model.token_embed.weight.grad)
        self.assertIsNotNone(model.blocks[0].ffn.router.weight.grad)
        
if __name__ == "__main__":
    unittest.main()
