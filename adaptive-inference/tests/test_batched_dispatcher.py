import unittest
import torch
from training.moe.reference import ReferenceMoELayer
from training.moe.batched_dispatcher import BatchedMoEDispatcher


class TestBatchedMoEDispatcherCorrectness(unittest.TestCase):
    """
    CPU-first correctness tests comparing ReferenceMoELayer (Oracle)
    against BatchedMoEDispatcher using torch.testing.assert_close.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.d_ff = 128
        self.num_experts = 8
        self.top_k = 2

        self.ref_moe = ReferenceMoELayer(self.d_model, self.d_ff, self.num_experts, self.top_k)
        self.bat_moe = BatchedMoEDispatcher(self.d_model, self.d_ff, self.num_experts, self.top_k)

        # Copy weights from ref_moe to bat_moe to ensure identical parameters
        self.bat_moe.load_state_dict(self.ref_moe.state_dict())

    def test_forward_equivalence(self):
        """Batched dispatcher must produce identical outputs to Reference."""
        x = torch.randn(4, 16, self.d_model)
        
        ref_out, ref_logits = self.ref_moe(x)
        bat_out, bat_logits = self.bat_moe(x)

        torch.testing.assert_close(bat_out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(bat_logits, ref_logits, rtol=1e-5, atol=1e-5)

    def test_backward_gradients_equivalence(self):
        """Gradient flow through batched dispatcher must match Reference."""
        x_ref = torch.randn(4, 16, self.d_model, requires_grad=True)
        x_bat = x_ref.clone().detach().requires_grad_(True)

        ref_out, _ = self.ref_moe(x_ref)
        bat_out, _ = self.bat_moe(x_bat)

        loss_ref = ref_out.pow(2).sum()
        loss_bat = bat_out.pow(2).sum()

        torch.testing.assert_close(loss_bat, loss_ref, rtol=1e-5, atol=1e-5)

        loss_ref.backward()
        loss_bat.backward()

        # Input gradients must match
        torch.testing.assert_close(x_bat.grad, x_ref.grad, rtol=1e-4, atol=1e-4)

        # Parameter gradients must match
        for (n_ref, p_ref), (n_bat, p_bat) in zip(
            self.ref_moe.named_parameters(), self.bat_moe.named_parameters()
        ):
            self.assertEqual(n_ref, n_bat, f"Parameter name mismatch: {n_ref} vs {n_bat}")
            if p_ref.grad is not None:
                torch.testing.assert_close(
                    p_bat.grad, p_ref.grad, rtol=1e-4, atol=1e-4,
                    msg=f"Gradient mismatch on parameter {n_ref}"
                )

    def test_state_dict_compatibility(self):
        """BatchedMoEDispatcher must have identical state_dict keys as ReferenceMoELayer."""
        ref_keys = set(self.ref_moe.state_dict().keys())
        bat_keys = set(self.bat_moe.state_dict().keys())
        self.assertEqual(ref_keys, bat_keys, f"state_dict key mismatch: {ref_keys ^ bat_keys}")

    def test_various_shapes(self):
        """Forward equivalence across different batch/seq combinations."""
        shapes = [(1, 4, self.d_model), (2, 32, self.d_model), (8, 1, self.d_model)]
        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(*shape)
                ref_out, ref_logits = self.ref_moe(x)
                bat_out, bat_logits = self.bat_moe(x)
                torch.testing.assert_close(bat_out, ref_out, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(bat_logits, ref_logits, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
