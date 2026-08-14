"""
runtime/residency.py — ASIR ResidencyManager & Memory Tiering Abstraction (TR-8.1)

Manages the placement, ownership, leases, and transfers of MoE expert parameters
between CPU RAM (pinned memory) and GPU VRAM.

Key concepts:
  - ExpertKey: NamedTuple(layer_id, expert_id)
  - ResidencyState: FREE, RAM_RESIDENT, VRAM_LOADING, VRAM_RESIDENT, VRAM_LEASED, EVICTING
  - ExpertRecord: Metadata container per expert tracking location, heat, leases, size
  - ResidencyManager: Active manager for acquiring leases, triggering H2D migrations,
    enforcing capacity limits via LRU/LFU/LFRU eviction, and logging memory telemetry.
"""

import time
from enum import Enum
from typing import NamedTuple, Dict, List, Set, Optional, Tuple, Any
import torch


class ExpertKey(NamedTuple):
    layer_id: int
    expert_id: int

    def __str__(self) -> str:
        return f"L{self.layer_id}_E{self.expert_id}"


class ResidencyState(Enum):
    FREE = "FREE"
    RAM_RESIDENT = "RAM_RESIDENT"
    VRAM_LOADING = "VRAM_LOADING"
    VRAM_RESIDENT = "VRAM_RESIDENT"
    VRAM_LEASED = "VRAM_LEASED"
    EVICTING = "EVICTING"


class ExpertRecord:
    """Tracking record for an expert in memory."""

    def __init__(self, key: ExpertKey, size_bytes: int = 0):
        self.key = key
        self.state = ResidencyState.FREE
        self.device = torch.device('cpu')
        self.size_bytes = size_bytes
        self.access_count = 0
        self.last_access_step = 0
        self.lease_count = 0
        self.ram_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.vram_state_dict: Optional[Dict[str, torch.Tensor]] = None

    @property
    def heat_score(self) -> float:
        """LFRU heat score combining frequency and recency."""
        return float(self.access_count) / max(1.0, float(self.last_access_step))

    def is_leased(self) -> bool:
        return self.lease_count > 0


class ResidencyManager:
    """
    ASIR Residency Manager.
    Coordinates memory placement between CPU RAM and GPU VRAM.
    Supports Lease protection and LRU/LFU/LFRU eviction.
    """

    def __init__(
        self,
        vram_capacity_experts: int,
        device: torch.device,
        policy: str = "lfru"
    ):
        self.vram_capacity = vram_capacity_experts
        self.device = device if (device and device.type == 'cuda') else torch.device('cpu')
        self.policy = policy.lower()

        # Registry of all known experts
        self.records: Dict[ExpertKey, ExpertRecord] = {}

        # Tracking active step counter for recency
        self.current_step = 0

        # Telemetry
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.leases_granted = 0
        self.h2d_bytes_transferred = 0
        self.h2d_transfer_time_ms = 0.0

    def register_expert(
        self,
        key: ExpertKey,
        state_dict: Dict[str, torch.Tensor],
        pin_memory: bool = True
    ) -> ExpertRecord:
        """Registers expert weights into CPU RAM tier with optional pinned memory."""
        size_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
        record = ExpertRecord(key, size_bytes=size_bytes)

        # Pin memory on CPU for faster non-blocking CUDA transfers
        processed_ram = {}
        for k, v in state_dict.items():
            tensor_cpu = v.detach().cpu()
            if pin_memory and self.device.type == 'cuda':
                tensor_cpu = tensor_cpu.pin_memory()
            processed_ram[k] = tensor_cpu

        record.ram_state_dict = processed_ram
        record.state = ResidencyState.RAM_RESIDENT
        record.device = torch.device('cpu')
        self.records[key] = record
        return record

    def lookup(self, key: ExpertKey) -> Optional[ExpertRecord]:
        return self.records.get(key)

    def advance_step(self):
        """Advances global step counter for recency tracking."""
        self.current_step += 1

    def acquire(self, keys: List[ExpertKey]) -> Dict[ExpertKey, Dict[str, torch.Tensor]]:
        """
        Acquires Leases for requested experts.
        Ensures all requested experts are loaded into VRAM, triggering H2D migration if needed.
        Leased experts cannot be evicted until release() is called.
        """
        self.advance_step()
        result: Dict[ExpertKey, Dict[str, torch.Tensor]] = {}

        for key in keys:
            if key not in self.records:
                raise KeyError(f"Expert {key} is not registered in ResidencyManager.")

            record = self.records[key]
            record.access_count += 1
            record.last_access_step = self.current_step

            # Check if already resident in VRAM
            if record.state in (ResidencyState.VRAM_RESIDENT, ResidencyState.VRAM_LEASED):
                self.hits += 1
                record.lease_count += 1
                record.state = ResidencyState.VRAM_LEASED
                self.leases_granted += 1
                result[key] = record.vram_state_dict  # type: ignore
                continue

            # VRAM Cache Miss
            self.misses += 1

            # Ensure room in VRAM before loading
            vram_count = sum(
                1 for r in self.records.values()
                if r.state in (ResidencyState.VRAM_RESIDENT, ResidencyState.VRAM_LEASED)
            )

            while vram_count >= self.vram_capacity and self.vram_capacity > 0:
                evicted_key = self._find_eviction_candidate()
                if evicted_key is None:
                    # All VRAM resident experts are currently LEASED! Cannot evict.
                    break
                self.evict(evicted_key)
                vram_count -= 1

            # Migrate from CPU RAM to GPU VRAM
            record.state = ResidencyState.VRAM_LOADING
            t0 = time.perf_counter()

            vram_dict = {}
            for param_name, param_tensor in record.ram_state_dict.items():  # type: ignore
                if self.device.type == 'cuda':
                    vram_dict[param_name] = param_tensor.to(self.device, non_blocking=True)
                else:
                    vram_dict[param_name] = param_tensor.clone()

            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.h2d_transfer_time_ms += elapsed_ms
            self.h2d_bytes_transferred += record.size_bytes

            record.vram_state_dict = vram_dict
            record.lease_count += 1
            record.state = ResidencyState.VRAM_LEASED
            record.device = self.device
            self.leases_granted += 1
            result[key] = record.vram_state_dict

        return result

    def release(self, keys: List[ExpertKey]):
        """
        Releases Leases for requested experts after execution.
        Decrements lease_count and transitions state from VRAM_LEASED to VRAM_RESIDENT.
        """
        for key in keys:
            if key not in self.records:
                continue
            record = self.records[key]
            if record.lease_count > 0:
                record.lease_count -= 1
            if record.lease_count == 0 and record.state == ResidencyState.VRAM_LEASED:
                record.state = ResidencyState.VRAM_RESIDENT

    def _find_eviction_candidate(self) -> Optional[ExpertKey]:
        """
        Finds an unleased VRAM resident candidate for eviction based on policy.
        Candidates must be in VRAM_RESIDENT state (lease_count == 0).
        """
        candidates = [
            r for r in self.records.values()
            if r.state == ResidencyState.VRAM_RESIDENT and r.lease_count == 0
        ]

        if not candidates:
            return None

        if self.policy == "lru":
            # Least recently used: smallest last_access_step
            candidates.sort(key=lambda r: r.last_access_step)
        elif self.policy == "lfu":
            # Least frequently used: smallest access_count
            candidates.sort(key=lambda r: r.access_count)
        else:  # "lfru" (default)
            # Least frequently/recently used: smallest heat_score
            candidates.sort(key=lambda r: (r.heat_score, r.last_access_step))

        return candidates[0].key

    def evict(self, key: ExpertKey):
        """Evicts expert from VRAM back to CPU RAM state."""
        if key not in self.records:
            return
        record = self.records[key]
        if record.lease_count > 0:
            raise RuntimeError(f"Cannot evict expert {key}: Lease is active (lease_count={record.lease_count}).")

        record.state = ResidencyState.EVICTING
        record.vram_state_dict = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        record.state = ResidencyState.RAM_RESIDENT
        record.device = torch.device('cpu')
        self.evictions += 1

    def stats(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry dictionary."""
        total_requests = max(self.hits + self.misses, 1)
        hit_rate = (self.hits / total_requests) * 100.0
        vram_resident_count = sum(
            1 for r in self.records.values()
            if r.state in (ResidencyState.VRAM_RESIDENT, ResidencyState.VRAM_LEASED)
        )
        leased_count = sum(1 for r in self.records.values() if r.is_leased())

        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'leases_granted': self.leases_granted,
            'total_requests': total_requests,
            'hit_rate_pct': float(hit_rate),
            'miss_rate_pct': float(100.0 - hit_rate),
            'vram_resident_count': vram_resident_count,
            'vram_capacity': self.vram_capacity,
            'currently_leased_count': leased_count,
            'h2d_bytes_transferred': self.h2d_bytes_transferred,
            'h2d_mb_transferred': self.h2d_bytes_transferred / (1024 * 1024),
            'h2d_transfer_time_ms': float(self.h2d_transfer_time_ms),
            'policy': self.policy
        }

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.leases_granted = 0
        self.h2d_bytes_transferred = 0
        self.h2d_transfer_time_ms = 0.0
