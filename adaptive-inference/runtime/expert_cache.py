"""
runtime/expert_cache.py — Dynamic Expert Cache Manager for ASIR (TR-10)

Manages RAM resident capacity C of (layer_id, expert_id) tensor weights.
Implements Static, LRU, and LRUWithPrefetch eviction policies, seamlessly triggering
weight transfers between NVMe/Disk storage and RAM.
"""

from typing import Dict, Tuple, List, Set, Optional, Any
from collections import deque, defaultdict
import torch

from runtime.expert_store import NVMeExpertStore, RAMExpertStore


class ExpertCache:
    """
    Dynamic Online Expert Cache manager.
    Key unit: (layer_id, expert_id)
    """

    def __init__(
        self,
        capacity_experts: int,
        nvme_store: NVMeExpertStore,
        policy: str = "lru",
        transition_matrix: Optional[List[List[float]]] = None
    ):
        self.capacity = capacity_experts
        self.nvme_store = nvme_store
        self.ram_store = nvme_store.ram_store
        self.policy = policy.lower()
        self.transition_matrix = transition_matrix
        
        # Cache queue for LRU: deque of (layer_id, expert_id)
        self.lru_queue: deque = deque()
        self.cache_keys: Set[Tuple[int, int]] = set()
        
        # Telemetry
        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self.evictions = 0

    def access(self, layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
        """
        Accesses expert weights for (layer_id, expert_id).
        Hits if resident in RAM cache, misses if NVMe load is required.
        """
        key = (layer_id, expert_id)
        
        if self.ram_store.is_resident(layer_id, expert_id):
            self.hits += 1
            if self.policy in ["lru", "lru_prefetch"]:
                # Update LRU recency position
                if key in self.cache_keys:
                    self.lru_queue.remove(key)
                self.lru_queue.append(key)
                self.cache_keys.add(key)
            return self.ram_store.get(layer_id, expert_id)
            
        # Cache Miss
        self.misses += 1
        
        # Enforce capacity ceiling before loading
        if len(self.cache_keys) >= self.capacity and self.capacity > 0:
            self._evict_one()
            
        # Fetch from NVMe into RAM
        weights = self.nvme_store.get(layer_id, expert_id)
        self.lru_queue.append(key)
        self.cache_keys.add(key)
        return weights

    def _evict_one(self) -> None:
        """Evicts least recently used (layer_id, expert_id) from RAM cache."""
        if not self.lru_queue:
            return
            
        evicted_key = self.lru_queue.popleft()
        self.cache_keys.remove(evicted_key)
        self.ram_store.evict(evicted_key[0], evicted_key[1])
        self.evictions += 1

    def prefetch_next(self, current_layer: int, current_expert: int, n_layers: int) -> None:
        """
        Prefetches predicted (layer_id, expert_id) for next step if policy includes prefetching.
        """
        if self.policy != "lru_prefetch" or self.transition_matrix is None:
            return
            
        if current_expert >= len(self.transition_matrix):
            return

        # Target next layer or token transition
        next_layer = (current_layer + 1) % n_layers
        predicted_expert = int(torch.tensor(self.transition_matrix[current_expert]).argmax().item())
        prefetch_key = (next_layer, predicted_expert)
        
        if prefetch_key not in self.cache_keys:
            if len(self.cache_keys) >= self.capacity and self.capacity > 0:
                self._evict_one()
            self.nvme_store.get(next_layer, predicted_expert)
            self.lru_queue.append(prefetch_key)
            self.cache_keys.add(prefetch_key)
            self.prefetches += 1

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self.evictions = 0
        self.nvme_store.reset_stats()

    def get_stats(self) -> Dict[str, Any]:
        total = max(self.hits + self.misses, 1)
        hit_rate = (self.hits / total) * 100.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'prefetches': self.prefetches,
            'evictions': self.evictions,
            'total_requests': total,
            'hit_rate_pct': float(hit_rate),
            'miss_rate_pct': float(100.0 - hit_rate),
            'resident_experts': len(self.cache_keys),
            'capacity': self.capacity,
            'policy': self.policy,
            'nvme_bytes_read': self.nvme_store.total_bytes_read,
            'nvme_io_time_ms': float(self.nvme_store.total_read_time_ms)
        }
