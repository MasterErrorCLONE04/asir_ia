import time
import torch
from typing import Dict, Optional, Any

_ACTIVE_TIMING_PROFILER: Optional['DispatchTimingProfiler'] = None

def set_global_timing_profiler(profiler: Optional['DispatchTimingProfiler']) -> None:
    global _ACTIVE_TIMING_PROFILER
    _ACTIVE_TIMING_PROFILER = profiler

def get_global_timing_profiler() -> Optional['DispatchTimingProfiler']:
    return _ACTIVE_TIMING_PROFILER

class DispatchTimingProfiler:
    """
    Timing Profiler for measuring execution latency of MoE routing,
    dispatching, expert execution, and combining steps with CUDA synchronization.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.is_cuda = (self.device.type == 'cuda' and torch.cuda.is_available())
        self.reset()

    def reset(self) -> None:
        """
        Reset timing records.
        """
        self.durations: Dict[str, float] = {
            'router': 0.0,
            'dispatch': 0.0,
            'experts': 0.0,
            'combine': 0.0,
            'total_step': 0.0
        }
        self.counts: Dict[str, int] = {k: 0 for k in self.durations}
        self._start_times: Dict[str, float] = {}
        self._cuda_events: Dict[str, torch.cuda.Event] = {}

    def start(self, stage_name: str) -> None:
        """
        Start timing a stage.
        """
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self._cuda_events[stage_name] = start_event
        else:
            self._start_times[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        """
        Stop timing a stage and accumulate duration in milliseconds.
        """
        elapsed_ms = 0.0
        if self.is_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize(self.device)
            start_event = self._cuda_events.pop(stage_name, None)
            if start_event:
                elapsed_ms = start_event.elapsed_time(end_event)
        else:
            start_time = self._start_times.pop(stage_name, None)
            if start_time is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if stage_name not in self.durations:
            self.durations[stage_name] = 0.0
            self.counts[stage_name] = 0
            
        self.durations[stage_name] += elapsed_ms
        self.counts[stage_name] += 1
        return elapsed_ms

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """
        Manually record a stage duration in milliseconds.
        """
        if stage_name not in self.durations:
            self.durations[stage_name] = 0.0
            self.counts[stage_name] = 0
        self.durations[stage_name] += duration_ms
        self.counts[stage_name] += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns average duration per stage in milliseconds.
        """
        averages = {}
        for stage, total_ms in self.durations.items():
            count = self.counts.get(stage, 0)
            averages[stage] = (total_ms / count) if count > 0 else 0.0
        return {
            'total_durations_ms': self.durations,
            'averages_ms': averages,
            'counts': self.counts
        }

    def format_report(self, model_name: str) -> str:
        """
        Generates formatted table of timing results.
        """
        summary = self.get_summary()
        avg = summary['averages_ms']
        
        lines = [
            f"──────── {model_name} — MoE Timing Baseline ────────",
            f"Router:          {avg.get('router', 0.0):.2f} ms",
            f"Dispatch:        {avg.get('dispatch', 0.0):.2f} ms",
            f"Experts:         {avg.get('experts', 0.0):.2f} ms",
            f"Combine:         {avg.get('combine', 0.0):.2f} ms",
            "──────────────────────────────────────────────",
            f"Total Step:      {avg.get('total_step', 0.0):.2f} ms",
            "──────────────────────────────────────────────"
        ]
        return "\n".join(lines)
