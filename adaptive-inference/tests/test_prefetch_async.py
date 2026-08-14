"""
tests/test_prefetch_async.py — Unit Tests for Async Prefetch, futures and prefetch isolation
"""

import unittest
import time
import torch
from concurrent.futures import Future

from runtime.expert_store import RAMExpertStore, NVMeExpertStore
from runtime.expert_cache import ExpertCache


class TestAsyncPrefetch(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        self.ram_store = RAMExpertStore(device=self.device, pin_memory=False)
        self.nvme_store = NVMeExpertStore(storage_dir="scratch/test_nvme", ram_store=self.ram_store)
        
        # Register a few synthetic experts
        self.dummy_sd = {"weight": torch.randn(10, 10)}
        for l in range(2):
            for e in range(4):
                self.nvme_store.register_expert_disk(l, e, self.dummy_sd)
                
    def tearDown(self):
        # Shutdown executor
        self.nvme_store.shutdown()
        # Clean up files
        import shutil
        shutil.rmtree("scratch/test_nvme", ignore_errors=True)

    def test_async_prefetch_non_blocking(self):
        # Create cache with capacity = 2 per layer (total 4)
        cache = ExpertCache(capacity_experts=4, nvme_store=self.nvme_store, policy="lru_prefetch")
        
        # Mock transition matrix (Markov prediction: state 0 goes to state 1)
        # 4 experts transition matrix
        cache.transition_matrix = [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0]
        ]
        
        # Access (0, 0)
        cache.access(0, 0)
        self.assertTrue(self.ram_store.is_resident(0, 0))
        
        # Trigger prefetch for next layer. Since current layer is 0, next is 1. Current expert is 0, argmax predicts 1.
        # So it prefetches (1, 1).
        cache.prefetch_next(current_layer=0, current_expert=0, n_layers=2, top_k_prefetch=1)
        
        # Check that (1, 1) is in prefetch_cache as a Future
        self.assertIn((1, 1), cache.prefetch_cache)
        self.assertIsInstance(cache.prefetch_cache[(1, 1)], Future)
        
        # Active cache should NOT contain (1, 1) yet
        self.assertNotIn((1, 1), cache.cache_keys)
        
        # Now access (1, 1), it must hit the prefetch_cache, wait for future, and move to active cache
        weights = cache.access(1, 1)
        self.assertIn("weight", weights)
        self.assertNotIn((1, 1), cache.prefetch_cache)
        self.assertIn((1, 1), cache.cache_keys)
        
        # Telemetry should count hits/prefetches
        stats = cache.get_stats()
        self.assertEqual(stats['prefetches'], 1)
        self.assertEqual(stats['hits'], 1)  # the access to (1, 1) was a hit in prefetch_cache
        self.assertEqual(stats['prefetch_precision_pct'], 100.0)

    def test_prefetch_cache_isolation_prevents_thrashing(self):
        # Create cache with capacity = 2
        cache = ExpertCache(capacity_experts=2, nvme_store=self.nvme_store, policy="lru_prefetch")
        cache.transition_matrix = [
            [0.0, 1.0, 0.0, 0.0]
        ]
        
        # Access two keys to fill capacity
        cache.access(0, 0)
        cache.access(0, 1)
        self.assertEqual(len(cache.cache_keys), 2)
        
        # Now trigger speculative prefetch. It should NOT evict any active cache keys because preprefetch_cache is isolated!
        cache.prefetch_next(current_layer=0, current_expert=0, n_layers=2)
        
        # active cache keys must still be (0, 0) and (0, 1)
        self.assertEqual(len(cache.cache_keys), 2)
        self.assertIn((0, 0), cache.cache_keys)
        self.assertIn((0, 1), cache.cache_keys)
        
        # The prefetched key (1, 1) should reside in prefetch_cache
        self.assertIn((1, 1), cache.prefetch_cache)

    def test_oracle_prefetch(self):
        cache = ExpertCache(capacity_experts=2, nvme_store=self.nvme_store, policy="oracle")
        
        # Pre-program oracle prefetch for next layer
        cache.prefetch_oracle(layer_id=1, expert_ids=[2, 3])
        
        self.assertIn((1, 2), cache.prefetch_cache)
        self.assertIn((1, 3), cache.prefetch_cache)
        
        # Accessing (1, 2) should hit the prefetch cache
        cache.access(1, 2)
        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 0)


if __name__ == '__main__':
    unittest.main()
