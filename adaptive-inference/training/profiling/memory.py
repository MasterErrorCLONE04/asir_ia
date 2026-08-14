import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class MemoryProfiler:
    """
    Memory Profiler for tracking VRAM allocation, parameter sizes,
    gradients, and optimizer state overhead during training steps.
    """
    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, device: Optional[torch.device] = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device('cpu'))
        self.checkpoints: Dict[str, Dict[str, float]] = {}

    def profile_checkpoint(self, stage_name: str) -> Dict[str, float]:
        """
        Record current memory state for a specific stage checkpoint.
        """
        stats: Dict[str, float] = {}
        if self.device.type == 'cuda' and torch.cuda.is_available():
            stats = {
                'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024 ** 3),
                'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024 ** 3),
                'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
                'max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
            }
        else:
            stats = {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'max_allocated_gb': 0.0,
                'max_reserved_gb': 0.0,
            }
        self.checkpoints[stage_name] = stats
        return stats

    def reset_cuda_peak_stats(self) -> None:
        """
        Resets peak memory stats for accurate per-step measurement.
        """
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_breakdown(self, active_params_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute precise breakdown of parameters, gradients, and optimizer states in GB.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        param_dtypes = sorted(list(set(str(p.dtype) for p in self.model.parameters())))
        
        grad_bytes = sum(
            p.grad.numel() * p.grad.element_size()
            for p in self.model.parameters()
            if p.grad is not None
        )
        
        opt_bytes = 0
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        opt_bytes += v.numel() * v.element_size()

        param_gb = param_bytes / (1024 ** 3)
        grad_gb = grad_bytes / (1024 ** 3)
        opt_gb = opt_bytes / (1024 ** 3)

        result = {
            'total_params': total_params,
            'active_params': active_params_count if active_params_count is not None else total_params,
            'param_dtypes': param_dtypes,
            'param_memory_gb': param_gb,
            'gradient_memory_gb': grad_gb,
            'optimizer_state_memory_gb': opt_gb,
            'total_static_memory_gb': param_gb + grad_gb + opt_gb,
            'stage_checkpoints': self.checkpoints
        }
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            result['peak_allocated_gb'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            result['peak_reserved_gb'] = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
        else:
            result['peak_allocated_gb'] = 0.0
            result['peak_reserved_gb'] = 0.0
            
        return result

    def format_report(self, model_name: str, precision: str, active_params_count: Optional[int] = None) -> str:
        """
        Generates a formatted report string summarizing memory usage.
        """
        breakdown = self.get_breakdown(active_params_count=active_params_count)
        device_name = torch.cuda.get_device_name(self.device) if (self.device.type == 'cuda' and torch.cuda.is_available()) else "CPU"

        lines = [
            "════════ ASIR MEMORY PROFILE ════════",
            f"Device:                   {device_name}",
            f"Precision:                {precision.upper()}",
            f"Model:                    {model_name}",
            f"Parameters (Total):       {breakdown['total_params'] / 1e6:.4f}M ({breakdown['param_memory_gb']:.2f} GB)",
            f"Active Parameters:        {breakdown['active_params'] / 1e6:.4f}M",
            f"Gradients:                {breakdown['gradient_memory_gb']:.2f} GB",
            f"Optimizer states:         {breakdown['optimizer_state_memory_gb']:.2f} GB",
            f"Static Total (P+G+O):     {breakdown['total_static_memory_gb']:.2f} GB",
            f"Peak Allocated VRAM:      {breakdown['peak_allocated_gb']:.2f} GB",
            f"Peak Reserved VRAM:       {breakdown['peak_reserved_gb']:.2f} GB",
            "────────────────────────────────────",
            "Stage Checkpoints:"
        ]
        
        for stage, stats in self.checkpoints.items():
            lines.append(f"  [{stage:<22}] Allocated: {stats['allocated_gb']:.2f} GB | Reserved: {stats['reserved_gb']:.2f} GB | Peak Alloc: {stats['max_allocated_gb']:.2f} GB")
            
        lines.append("════════════════════════════════════")
        return "\n".join(lines)
