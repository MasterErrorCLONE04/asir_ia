import unittest
import torch
import torch.nn as nn
from unittest.mock import patch

from training.profiling.memory import MemoryProfiler
from training.profiling.dispatch_timing import DispatchTimingProfiler
from training.train import get_model_config

class TestGPUConfigAndProfiling(unittest.TestCase):
    """
    CPU-safe tests for GPU configuration flags and profiling modules.
    Verifies that memory and timing profilers function correctly on CPU
    and that explicit CUDA requests fail cleanly when CUDA is absent.
    """

    def setUp(self):
        # Create a small dummy model for CPU-safe profiling tests
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def test_memory_profiler_cpu_safe(self):
        profiler = MemoryProfiler(self.model, self.optimizer, device=torch.device('cpu'))
        profiler.reset_cuda_peak_stats()
        
        profiler.profile_checkpoint("STEP_START")
        
        # Dummy forward & backward
        x = torch.randn(4, 64)
        out = self.model(x)
        loss = out.sum()
        profiler.profile_checkpoint("POST_FORWARD")
        
        loss.backward()
        profiler.profile_checkpoint("POST_BACKWARD")
        
        self.optimizer.step()
        profiler.profile_checkpoint("AFTER_OPTIMIZER_STEP")
        
        breakdown = profiler.get_breakdown(active_params_count=1000)
        self.assertIn('total_params', breakdown)
        self.assertIn('param_memory_gb', breakdown)
        self.assertIn('gradient_memory_gb', breakdown)
        self.assertIn('optimizer_state_memory_gb', breakdown)
        self.assertGreater(breakdown['total_params'], 0)
        self.assertGreater(breakdown['optimizer_state_memory_gb'], 0.0)

        report = profiler.format_report("TestModel", "fp32", active_params_count=1000)
        self.assertIn("ASIR MEMORY PROFILE", report)
        self.assertIn("FP32", report)

    def test_dispatch_timing_profiler_cpu_safe(self):
        profiler = DispatchTimingProfiler(device=torch.device('cpu'))
        profiler.start("router")
        torch.zeros((10, 10))
        profiler.stop("router")
        
        profiler.record_stage("dispatch", 1.5)
        profiler.record_stage("experts", 5.0)
        profiler.record_stage("combine", 0.5)

        summary = profiler.get_summary()
        self.assertIn("router", summary['averages_ms'])
        self.assertIn("dispatch", summary['averages_ms'])
        self.assertIn("experts", summary['averages_ms'])
        self.assertIn("combine", summary['averages_ms'])

        report = profiler.format_report("TestModel")
        self.assertIn("MoE Timing Baseline", report)
        self.assertIn("Router:", report)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_strict_error(self, mock_cuda):
        """
        Verify that requesting --device cuda when CUDA is not available raises a RuntimeError.
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        args = parser.parse_args(["--device", "cuda"])

        with self.assertRaises(RuntimeError) as ctx:
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("ERROR: CUDA requested but torch.cuda.is_available() == False")
        self.assertIn("CUDA requested but torch.cuda.is_available() == False", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
