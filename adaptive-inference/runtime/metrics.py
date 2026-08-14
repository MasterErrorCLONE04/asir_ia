"""
runtime/metrics.py — Inference Performance & Memory Telemetry Tracker for ASIR

Tracks physical execution latency, separate prefill vs autoregressive decode token throughput (decode tok/s),
NVMe data transfer volume/bandwidth, and physical RAM/VRAM memory footprint.
"""

import time
import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from typing import Dict, Any, Optional


class RuntimeMetrics:
    """
    Performance and Resource telemetry collector for ASIR Inference Engine.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.prefill_tokens = 0
        self.prefill_time_ms = 0.0
        self.decode_tokens = 0
        self.decode_time_ms = 0.0
        
        self.nvme_bytes_read = 0
        self.nvme_io_time_ms = 0.0
        
        self._prefill_start: Optional[float] = None
        self._decode_start: Optional[float] = None

    def start_prefill(self) -> None:
        self._prefill_start = time.perf_counter()

    def end_prefill(self, num_tokens: int) -> None:
        if self._prefill_start is not None:
            self.prefill_time_ms += (time.perf_counter() - self._prefill_start) * 1000.0
            self.prefill_tokens += num_tokens
            self._prefill_start = None

    def start_decode(self) -> None:
        self._decode_start = time.perf_counter()

    def step_decode(self, tokens_generated: int = 1) -> None:
        self.decode_tokens += tokens_generated

    def end_decode(self) -> None:
        if self._decode_start is not None:
            self.decode_time_ms += (time.perf_counter() - self._decode_start) * 1000.0
            self._decode_start = None

    def record_nvme_read(self, bytes_read: int, duration_ms: float) -> None:
        self.nvme_bytes_read += bytes_read
        self.nvme_io_time_ms += duration_ms

    def get_summary(self) -> Dict[str, Any]:
        prefill_tps = (self.prefill_tokens / max(self.prefill_time_ms / 1000.0, 1e-6))
        decode_tps = (self.decode_tokens / max(self.decode_time_ms / 1000.0, 1e-6))
        
        nvme_mb = self.nvme_bytes_read / (1024 * 1024)
        nvme_throughput_mbps = (nvme_mb / max(self.nvme_io_time_ms / 1000.0, 1e-6))
        
        ram_mb = (psutil.Process().memory_info().rss / (1024 * 1024)) if HAS_PSUTIL else 0.0
        vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        
        return {
            'prefill_tokens': self.prefill_tokens,
            'prefill_time_ms': float(self.prefill_time_ms),
            'prefill_tok_per_sec': float(prefill_tps),
            'decode_tokens': self.decode_tokens,
            'decode_time_ms': float(self.decode_time_ms),
            'decode_tok_per_sec': float(decode_tps),
            'nvme_bytes_read': self.nvme_bytes_read,
            'nvme_mb_read': float(nvme_mb),
            'nvme_io_time_ms': float(self.nvme_io_time_ms),
            'nvme_throughput_mbps': float(nvme_throughput_mbps),
            'peak_ram_mb': float(ram_mb),
            'peak_vram_mb': float(vram_mb)
        }
