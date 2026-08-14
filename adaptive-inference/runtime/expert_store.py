"""
runtime/expert_store.py — Multi-tier Expert Storage Engine for ASIR (TR-10)

Defines storage backends for expert parameter weights indexed by (layer_id, expert_id) tuples.
Supports RAM resident tensors (with PyTorch pinned memory) and NVMe disk-backed persistence
with CUDA async I/O streams and non-blocking Host-to-Device transfers.
"""

import os
import time
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Set, Optional, Any


class BaseExpertStore(ABC):
    """
    Abstract interface for Expert parameter stores.
    The primary key is (layer_id, expert_id).
    """

    @abstractmethod
    def get(self, layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
        """Retrieves expert weights for (layer_id, expert_id). Performs load if not resident."""
        pass

    @abstractmethod
    def prefetch(self, layer_id: int, expert_ids: List[int]) -> None:
        """Asynchronously or synchronously prefetches expert weights into high-speed tier."""
        pass

    @abstractmethod
    def evict(self, layer_id: int, expert_id: int) -> None:
        """Evicts (layer_id, expert_id) weights from the store."""
        pass

    @abstractmethod
    def is_resident(self, layer_id: int, expert_id: int) -> bool:
        """Returns True if (layer_id, expert_id) is currently resident in RAM/VRAM."""
        pass


class RAMExpertStore(BaseExpertStore):
    """
    In-memory Expert Storage tier (RAM / VRAM).
    Keeps expert parameters resident as PyTorch state dicts (pinned CPU memory or CUDA tensors).
    """

    def __init__(self, device: torch.device, pin_memory: bool = True):
        self.device = device
        self.pin_memory = pin_memory and (device.type == 'cuda')
        self.store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

    def put(self, layer_id: int, expert_id: int, state_dict: Dict[str, torch.Tensor]) -> None:
        """Registers expert weights into RAM store, using pinned memory for fast CUDA transfers."""
        key = (layer_id, expert_id)
        processed = {}
        for k, v in state_dict.items():
            if self.pin_memory and v.device.type == 'cpu':
                processed[k] = v.pin_memory()
            else:
                processed[k] = v.to(self.device)
        self.store[key] = processed

    def get(self, layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
        key = (layer_id, expert_id)
        if key not in self.store:
            raise KeyError(f"Expert (layer={layer_id}, expert={expert_id}) not found in RAM store.")
            
        weights = self.store[key]
        # Fast non-blocking transfer to GPU if pinned CPU memory
        if self.device.type == 'cuda':
            return {k: v.to(self.device, non_blocking=True) for k, v in weights.items()}
        return weights

    def prefetch(self, layer_id: int, expert_ids: List[int]) -> None:
        pass

    def evict(self, layer_id: int, expert_id: int) -> None:
        key = (layer_id, expert_id)
        if key in self.store:
            del self.store[key]

    def is_resident(self, layer_id: int, expert_id: int) -> bool:
        return (layer_id, expert_id) in self.store

    def total_memory_mb(self) -> float:
        """Calculates total RAM memory occupied by resident experts in MB."""
        total_bytes = 0
        for state_dict in self.store.values():
            for t in state_dict.values():
                total_bytes += t.numel() * t.element_size()
        return total_bytes / (1024 * 1024)


class NVMeExpertStore(BaseExpertStore):
    """
    Disk-backed Expert Storage tier (NVMe / SSD persistence & streaming).
    Stores expert weights on physical disk and measures physical I/O transfer latency
    and exact byte volume when loading weights into RAM upon cache miss.
    """

    def __init__(self, storage_dir: str, ram_store: RAMExpertStore, nvme_bw_gbps: float = 5.0):
        self.storage_dir = storage_dir
        self.ram_store = ram_store
        self.nvme_bw_mb_per_ms = (nvme_bw_gbps * 1000.0) / 1000.0  # 5 GB/s = 5 MB/ms
        os.makedirs(storage_dir, exist_ok=True)
        
        # Dedicated CUDA stream for async I/O transfers
        self.io_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Disk registry: (layer_id, expert_id) -> (filepath, size_bytes)
        self.disk_registry: Dict[Tuple[int, int], Tuple[str, int]] = {}
        
        # Physical Telemetry Counters
        self.total_bytes_read = 0
        self.total_read_time_ms = 0.0
        self.total_miss_count = 0

    def register_expert_disk(self, layer_id: int, expert_id: int, state_dict: Dict[str, torch.Tensor]) -> None:
        """Saves expert state_dict to disk simulating NVMe binary weight persistence."""
        key = (layer_id, expert_id)
        filepath = os.path.join(self.storage_dir, f"layer_{layer_id}_expert_{expert_id}.pt")
        
        if not os.path.exists(filepath):
            torch.save(state_dict, filepath)
            
        size_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
        self.disk_registry[key] = (filepath, size_bytes)

    def get(self, layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
        key = (layer_id, expert_id)
        
        # Check if already resident in RAM
        if self.ram_store.is_resident(layer_id, expert_id):
            return self.ram_store.get(layer_id, expert_id)
            
        # Cache Miss: Fetch from NVMe into RAM
        if key not in self.disk_registry:
            raise KeyError(f"Expert (layer={layer_id}, expert={expert_id}) not found on NVMe store.")
            
        filepath, size_bytes = self.disk_registry[key]
        size_mb = size_bytes / (1024 * 1024)
        
        start_time = time.perf_counter()
        
        # Optional CUDA async stream context for weight loading
        if self.io_stream is not None:
            with torch.cuda.stream(self.io_stream):
                state_dict = torch.load(filepath, map_location='cpu')
        else:
            state_dict = torch.load(filepath, map_location='cpu')
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Physical/simulated NVMe bandwidth minimum bounds
        min_expected_ms = size_mb / self.nvme_bw_mb_per_ms
        simulated_io_ms = max(elapsed_ms, min_expected_ms)
        
        # Accumulate exact telemetry
        self.total_bytes_read += size_bytes
        self.total_read_time_ms += simulated_io_ms
        self.total_miss_count += 1
        
        # Put into RAM store
        self.ram_store.put(layer_id, expert_id, state_dict)
        return self.ram_store.get(layer_id, expert_id)

    def prefetch(self, layer_id: int, expert_ids: List[int]) -> None:
        for e_id in expert_ids:
            if not self.ram_store.is_resident(layer_id, e_id):
                self.get(layer_id, e_id)

    def evict(self, layer_id: int, expert_id: int) -> None:
        self.ram_store.evict(layer_id, expert_id)

    def is_resident(self, layer_id: int, expert_id: int) -> bool:
        return self.ram_store.is_resident(layer_id, expert_id)

    def reset_stats(self) -> None:
        self.total_bytes_read = 0
        self.total_read_time_ms = 0.0
        self.total_miss_count = 0
