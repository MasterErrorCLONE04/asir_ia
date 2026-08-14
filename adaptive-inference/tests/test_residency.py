"""
tests/test_residency.py — Unit Tests for ResidencyManager and Lease Contracts
"""

import unittest
import torch
from runtime.residency import ResidencyManager, ExpertKey, ResidencyState


class TestResidencyManager(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.capacity = 2
        self.rm = ResidencyManager(vram_capacity_experts=self.capacity, device=self.device, policy="lfru")

        # Register 4 dummy experts (Layer 0, Experts 0..3)
        self.keys = [ExpertKey(0, i) for i in range(4)]
        self.dummy_state_dicts = {}
        for key in self.keys:
            sd = {
                'w1': torch.randn(16, 32, dtype=torch.bfloat16),
                'w2': torch.randn(32, 16, dtype=torch.bfloat16)
            }
            self.dummy_state_dicts[key] = sd
            self.rm.register_expert(key, sd)

    def test_registration_state(self):
        """All registered experts should initially be in RAM_RESIDENT state."""
        for key in self.keys:
            rec = self.rm.lookup(key)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.state, ResidencyState.RAM_RESIDENT)  # type: ignore

    def test_acquire_and_release_lease(self):
        """Acquiring lease must load into VRAM and increment lease_count."""
        k0 = self.keys[0]
        res = self.rm.acquire([k0])

        self.assertIn(k0, res)
        rec = self.rm.lookup(k0)
        self.assertEqual(rec.state, ResidencyState.VRAM_LEASED)  # type: ignore
        self.assertEqual(rec.lease_count, 1)  # type: ignore

        # Release lease
        self.rm.release([k0])
        self.assertEqual(rec.state, ResidencyState.VRAM_RESIDENT)  # type: ignore
        self.assertEqual(rec.lease_count, 0)  # type: ignore

    def test_lease_prevents_eviction(self):
        """Leased experts must NOT be evicted when capacity is reached."""
        k0, k1, k2 = self.keys[0], self.keys[1], self.keys[2]

        # Acquire k0 (capacity=2)
        _ = self.rm.acquire([k0])
        # Acquire k1 and release it
        _ = self.rm.acquire([k1])
        self.rm.release([k1])

        # k0 is LEASED, k1 is VRAM_RESIDENT (unleased)
        # Now acquire k2: should evict k1 (unleased), NOT k0 (leased)
        _ = self.rm.acquire([k2])

        rec_k0 = self.rm.lookup(k0)
        rec_k1 = self.rm.lookup(k1)
        rec_k2 = self.rm.lookup(k2)

        self.assertEqual(rec_k0.state, ResidencyState.VRAM_LEASED)  # type: ignore
        self.assertEqual(rec_k1.state, ResidencyState.RAM_RESIDENT)  # type: ignore (evicted)
        self.assertEqual(rec_k2.state, ResidencyState.VRAM_LEASED)  # type: ignore

    def test_eviction_policies(self):
        """Test LRU vs LFU eviction behavior."""
        # LRU manager with capacity 2
        rm_lru = ResidencyManager(vram_capacity_experts=2, device=self.device, policy="lru")
        for key in self.keys:
            rm_lru.register_expert(key, self.dummy_state_dicts[key])

        # Access k0, release
        _ = rm_lru.acquire([self.keys[0]])
        rm_lru.release([self.keys[0]])

        # Access k1, release
        _ = rm_lru.acquire([self.keys[1]])
        rm_lru.release([self.keys[1]])

        # Access k0 again (makes k0 more recent than k1)
        _ = rm_lru.acquire([self.keys[0]])
        rm_lru.release([self.keys[0]])

        # Now access k2: LRU should evict k1
        _ = rm_lru.acquire([self.keys[2]])
        rm_lru.release([self.keys[2]])

        self.assertEqual(rm_lru.lookup(self.keys[1]).state, ResidencyState.RAM_RESIDENT)  # type: ignore
        self.assertEqual(rm_lru.lookup(self.keys[0]).state, ResidencyState.VRAM_RESIDENT)  # type: ignore

    def test_telemetry_counters(self):
        """Verify hits, misses, evictions, and byte transfer telemetry."""
        k0, k1 = self.keys[0], self.keys[1]

        # First acquire: Miss
        _ = self.rm.acquire([k0])
        # Second acquire while resident: Hit
        _ = self.rm.acquire([k0])

        stats = self.rm.stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['h2d_bytes_transferred'], 0)


if __name__ == '__main__':
    unittest.main()
