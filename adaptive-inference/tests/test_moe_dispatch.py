import unittest
import torch
import torch.nn as nn
from training.moe.reference import ReferenceMoELayer
from training.moe.dispatcher import SparseMoEDispatcher
from analysis.moe_telemetry import MoETelemetryTracker

class TestMoEDispatchCorrectness(unittest.TestCase):
    """
    CPU-first correctness tests comparing ReferenceMoELayer (Oracle)
    against SparseMoEDispatcher using torch.testing.assert_close.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.d_ff = 128
        self.num_experts = 8
        self.top_k = 2

        self.ref_moe = ReferenceMoELayer(self.d_model, self.d_ff, self.num_experts, self.top_k)
        self.opt_moe = SparseMoEDispatcher(self.d_model, self.d_ff, self.num_experts, self.top_k)

        # Copy weights from ref_moe to opt_moe to ensure identical parameters
        self.opt_moe.load_state_dict(self.ref_moe.state_dict())

    def test_forward_equivalence(self):
        x = torch.randn(4, 16, self.d_model)
        
        ref_out, ref_logits = self.ref_moe(x)
        opt_out, opt_logits = self.opt_moe(x)

        # Mathematical equivalence assertions
        torch.testing.assert_close(opt_out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(opt_logits, ref_logits, rtol=1e-5, atol=1e-5)

    def test_backward_gradients_equivalence(self):
        x_ref = torch.randn(4, 16, self.d_model, requires_grad=True)
        x_opt = x_ref.clone().detach().requires_grad_(True)

        ref_out, _ = self.ref_moe(x_ref)
        opt_out, _ = self.opt_moe(x_opt)

        loss_ref = ref_out.pow(2).sum()
        loss_opt = opt_out.pow(2).sum()

        torch.testing.assert_close(loss_opt, loss_ref, rtol=1e-5, atol=1e-5)

        loss_ref.backward()
        loss_opt.backward()

        torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=1e-5, atol=1e-5)

        for (n1, p1), (n2, p2) in zip(self.ref_moe.named_parameters(), self.opt_moe.named_parameters()):
            self.assertEqual(n1, n2)
            torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-5, atol=1e-5)

    def test_moe_telemetry_tracker(self):
        tracker = MoETelemetryTracker(num_experts=self.num_experts, top_k=self.top_k)
        dummy_indices = torch.tensor([[0, 1], [2, 3], [0, 4]])
        tracker.update(dummy_indices)

        metrics = tracker.get_metrics()
        self.assertEqual(metrics['total_assignments'], 6)
        self.assertGreater(metrics['n_eff'], 1.0)
        self.assertEqual(metrics['expert_frequencies'][0], 2)

if __name__ == "__main__":
    unittest.main()
