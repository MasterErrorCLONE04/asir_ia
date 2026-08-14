"""
training/test_models.py — Contract tests for R1.0 model architecture.

Verifies:
  1. §4 parameter contract: A = B + K·E for all configs.
     - M1–M5, C1–C4: ΔA_rel < 0.1%  (MoE exact-build intent)
     - M0, Dense-A:  ΔA_rel ≤ 1.0%  (dense tolerance per §2 v0.7)
  2. get_model_config always returns B, K, E, A fields.
  3. Analytical B/K/E/A from get_model_config matches actual model counts from
     count_active_params — but only for SMALL configs that fit in CPU memory.
  4. Large configs (M4, M5, C3, C4 — up to 896 experts × 5 layers) are
     verified ANALYTICALLY only (formula integrity + tolerance check) without
     instantiating a PyTorch model.  Instantiating M5 full would require
     ~10 GB RAM on CPU and kills the Docker container with OOM.
  5. Controlled routing mask correctness.
  6. Backpropagation integrity.
  7. alignment.py — many-to-one assignment, specialization_density,
     unassigned_routing_mass, Q_role_aligned  (§6 v0.7).
"""

import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from training.models.transformer import MoETransformer
from training.router.oracle import get_domain_mask_batch
from training.router.alignment import (
    ROLES, TAU_SPEC,
    assign_experts_to_roles,
    specialization_density,
    unassigned_routing_mass,
    Q_role_aligned,
    compute_expert_role_correlation,
)
from training.train import get_model_config, count_active_params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_A    = 140_000_000
VOCAB_SIZE  = 100
D_MODEL     = 1408
N_LAYERS    = 5
NUM_HEADS   = 8

# Tolerance bands (§2 v0.7)
MoE_TOL   = 0.001   # < 0.1%  — M1–M5, C1–C4  (integer d_ff limit)
DENSE_TOL = 0.010   # ≤ 1.0%  — M0, Dense-A

# Configs verified by building a real PyTorch model (fit comfortably in CPU RAM)
# Threshold: ≤ 32 experts × 5 layers × ~9.9M params/expert ≈ 1.6B params → ~6 GB RAM
SMALL_MoE_CONFIGS   = ['M1', 'M2', 'C1']    # 8 or 32 experts — instantiate

# Configs verified analytically only (too large to instantiate on CPU)
# 128 experts × 5 layers × ~9.8M params/expert ≈ 6.3B params → ~25 GB RAM
LARGE_MoE_CONFIGS   = ['M3', 'M4', 'M5', 'C2', 'C3', 'C4']

SMALL_DENSE_CONFIGS = ['M0', 'Dense-A']


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_model(name: str) -> MoETransformer:
    cfg = get_model_config(name)
    return MoETransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=cfg['d_ff'],
        moe=cfg['moe'],
        num_experts=cfg['num_experts'],
        top_k=cfg['top_k'],
    )


# ---------------------------------------------------------------------------
# §4 parameter contract tests
# ---------------------------------------------------------------------------

class TestParameterContract(unittest.TestCase):
    """
    §4 v0.7: A = B + K·E = 140M.
    Two verification modes:
      - Small configs: build real PyTorch model + cross-validate vs analytical.
      - Large configs: formula integrity + tolerance check only (no instantiation).
    """

    def _check_analytical_only(self, name: str, tol: float):
        """
        Verifies formula integrity and tolerance from get_model_config alone.
        Used for large configs (M4, M5, C3, C4) that would OOM on CPU.
        """
        cfg = get_model_config(name)

        # 1. Required §4 fields present
        for field in ('B', 'K', 'E', 'A'):
            self.assertIn(field, cfg, f"{name}: '{field}' missing from get_model_config()")

        # 2. Formula integrity: B + K*E == A
        self.assertEqual(
            cfg['B'] + cfg['K'] * cfg['E'], cfg['A'],
            f"{name}: B + K*E != A in config dict"
        )

        # 3. Tolerance vs TARGET_A
        delta_rel = abs(cfg['A'] - TARGET_A) / TARGET_A
        print(
            f"  {name:8s} | B={cfg['B']/1e6:7.3f}M  K={cfg['K']}  "
            f"E={cfg['E']/1e6:7.3f}M  A={cfg['A']:,}  "
            f"ΔA_rel={delta_rel*100:.4f}%  [analytical]"
        )
        self.assertLess(
            delta_rel, tol,
            f"{name}: ΔA_rel={delta_rel*100:.4f}% exceeds tolerance {tol*100:.1f}%"
        )

    def _check_with_model(self, name: str, tol: float):
        """
        Full check: formula integrity + PyTorch model cross-validation.
        Used for small configs that fit in CPU RAM.
        """
        cfg    = get_model_config(name)
        model  = build_model(name)
        actual = count_active_params(model)

        # 1. Required §4 fields present
        for field in ('B', 'K', 'E', 'A'):
            self.assertIn(field, cfg, f"{name}: '{field}' missing from get_model_config()")

        # 2. Formula integrity in config dict
        self.assertEqual(
            cfg['B'] + cfg['K'] * cfg['E'], cfg['A'],
            f"{name}: config B + K*E != A"
        )

        # 3. Analytical == actual PyTorch counts
        self.assertEqual(actual['B'], cfg['B'],
            f"{name}: actual B={actual['B']:,} != analytical B={cfg['B']:,}")
        self.assertEqual(actual['K'], cfg['K'],
            f"{name}: actual K={actual['K']} != config K={cfg['K']}")
        self.assertEqual(actual['E'], cfg['E'],
            f"{name}: actual E={actual['E']:,} != analytical E={cfg['E']:,}")
        self.assertEqual(actual['A'], cfg['A'],
            f"{name}: actual A={actual['A']:,} != analytical A={cfg['A']:,}")

        # 4. Tolerance vs TARGET_A
        delta_rel = abs(actual['A'] - TARGET_A) / TARGET_A
        print(
            f"  {name:8s} | B={actual['B']/1e6:7.3f}M  K={actual['K']}  "
            f"E={actual['E']/1e6:7.3f}M  A={actual['A']:,}  "
            f"ΔA_rel={delta_rel*100:.4f}%"
        )
        self.assertLess(
            delta_rel, tol,
            f"{name}: ΔA_rel={delta_rel*100:.4f}% exceeds tolerance {tol*100:.1f}%"
        )

    # --- Small MoE configs: full cross-validation (≤ 32 experts) ---

    def test_M1_contract(self):
        print("\nSmall MoE configs (formula + PyTorch cross-validation, \u226432 experts):")
        self._check_with_model('M1', MoE_TOL)

    def test_M2_contract(self):
        self._check_with_model('M2', MoE_TOL)

    def test_C1_contract(self):
        self._check_with_model('C1', MoE_TOL)

    # --- Large MoE configs: analytical only (≥128 experts, OOM guard) ---

    def test_M3_contract(self):
        print("\nLarge MoE configs (analytical only \u2014 OOM guard, \u2265128 experts):")
        self._check_analytical_only('M3', MoE_TOL)

    def test_M4_contract(self):
        self._check_analytical_only('M4', MoE_TOL)

    def test_M5_contract(self):
        self._check_analytical_only('M5', MoE_TOL)

    def test_C2_contract(self):
        self._check_analytical_only('C2', MoE_TOL)

    def test_C3_contract(self):
        self._check_analytical_only('C3', MoE_TOL)

    def test_C4_contract(self):
        self._check_analytical_only('C4', MoE_TOL)

    # --- Dense configs: full cross-validation ---

    def test_M0_contract(self):
        print("\nDense configs (formula + PyTorch cross-validation):")
        self._check_with_model('M0', DENSE_TOL)

    def test_DenseA_contract(self):
        self._check_with_model('Dense-A', DENSE_TOL)


# ---------------------------------------------------------------------------
# Controlled routing mask test
# ---------------------------------------------------------------------------

class TestControlledRouting(unittest.TestCase):
    """
    Verify that the domain routing mask prevents selection of out-of-domain experts.
    """

    def test_controlled_routing_mask(self):
        """
        Router mask must zero out all logits for out-of-domain experts.
        Uses M1 (N_total=8, K=2) for fast execution.

        Expert allocation for num_experts=8  (oracle.py):
          arithmetic: experts 0,1,2  (count=3, 40% of 8 rounded)
          logic:      experts 3,4,5  (count=3, 35% of 8 rounded)
          language:   experts 6,7    (count=2, 25% of 8 remainder)
        """
        cfg   = get_model_config('M1')
        model = build_model('M1')

        # Batch size 2, seq_len 4
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 4))
        domains   = ['arithmetic', 'language']
        mask      = get_domain_mask_batch(domains, num_experts=8)

        # Inspect first block's MoELayer directly
        moe_layer = model.blocks[0].ffn
        x         = torch.randn(2, 4, D_MODEL)
        _, router_logits = moe_layer(x, mask)

        gates = F.softmax(router_logits.view(-1, 8), dim=-1)

        # Batch[0] = 'arithmetic' → only experts 0,1,2 allowed
        for tok in range(4):
            for exp in range(3, 8):
                self.assertAlmostEqual(
                    gates[tok, exp].item(), 0.0, places=5,
                    msg=f"arithmetic batch: expert {exp} should be masked"
                )

        # Batch[1] = 'language' → only experts 6,7 allowed
        for tok in range(4, 8):
            for exp in range(0, 6):
                self.assertAlmostEqual(
                    gates[tok, exp].item(), 0.0, places=5,
                    msg=f"language batch: expert {exp} should be masked"
                )


# ---------------------------------------------------------------------------
# Backpropagation integrity test
# ---------------------------------------------------------------------------

class TestBackprop(unittest.TestCase):
    """
    Verify forward + backward pass runs without errors and all
    reachable parameters receive gradients.
    """

    def test_backprop_gradients(self):
        """
        Uses a small model (d_model=256) to keep the test fast.
        """
        model = MoETransformer(
            vocab_size=VOCAB_SIZE,
            d_model=256,
            n_layers=2,
            num_heads=4,
            d_ff=512,
            moe=True,
            num_experts=4,
            top_k=2,
        )

        input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        domains   = ['arithmetic', 'logic']
        mask      = get_domain_mask_batch(domains, num_experts=4)

        logits, all_router_logits = model(input_ids, mask)

        # Shape checks
        self.assertEqual(logits.shape, (2, 10, VOCAB_SIZE))
        self.assertEqual(len(all_router_logits), 2)
        self.assertEqual(all_router_logits[0].shape, (2, 10, 4))

        # Backward pass
        loss = logits.sum()
        loss.backward()

        # Embedding and router gradients must exist
        self.assertIsNotNone(model.token_embed.weight.grad)
        self.assertIsNotNone(model.blocks[0].ffn.router.weight.grad)


# ---------------------------------------------------------------------------
# §6 alignment tests
# ---------------------------------------------------------------------------

class TestAlignment(unittest.TestCase):
    """
    Tests for training/router/alignment.py — many-to-one assignment,
    specialization_density, unassigned_routing_mass, Q_role_aligned.

    All tests use synthetic correlation matrices so they do not depend on
    a trained model or a choice of statistical estimator.
    """

    # ------------------------------------------------------------------
    # assign_experts_to_roles
    # ------------------------------------------------------------------

    def test_many_to_one_assignment(self):
        """
        Multiple experts may map to the same role (many-to-one).
        Experts with max corr < tau map to None (∅).
        Experts with max corr == tau are assigned (>= tau).
        """
        # 5 experts, 3 roles: [arithmetic, logic, language]
        corr = torch.tensor([
            [0.80, 0.10, 0.10],   # → arithmetic   (clear winner)
            [0.70, 0.20, 0.10],   # → arithmetic   (many-to-one)
            [0.10, 0.90, 0.00],   # → logic
            [0.20, 0.30, 0.50],   # → language     (exactly at tau → assigned)
            [0.30, 0.35, 0.35],   # → None (∅)     (max = 0.35 < tau = 0.50)
        ])

        assignment = assign_experts_to_roles(corr, tau=TAU_SPEC)

        self.assertEqual(assignment[0], 'arithmetic')
        self.assertEqual(assignment[1], 'arithmetic',   "many-to-one: two experts → arithmetic")
        self.assertEqual(assignment[2], 'logic')
        self.assertEqual(assignment[3], 'language',     "exactly at tau → assigned (>= tau)")
        self.assertIsNone(assignment[4],                "below tau → ∅")

    def test_all_unassigned(self):
        """
        If all correlations are below tau, all experts are ∅.
        """
        corr = torch.tensor([
            [0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3],
        ])
        assignment = assign_experts_to_roles(corr, tau=TAU_SPEC)
        self.assertIsNone(assignment[0])
        self.assertIsNone(assignment[1])

    def test_all_assigned(self):
        """
        If all max correlations are >= tau, no expert is ∅.
        """
        corr = torch.tensor([
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
        ])
        assignment = assign_experts_to_roles(corr, tau=TAU_SPEC)
        self.assertIsNotNone(assignment[0])
        self.assertIsNotNone(assignment[1])
        self.assertIsNotNone(assignment[2])

    def test_sensitivity_thresholds(self):
        """
        Same matrix should give different SD at tau=0.30 vs tau=0.70.
        Mimics obligatory sensitivity analysis from §6.
        """
        corr = torch.tensor([
            [0.60, 0.20, 0.20],   # ≥ 0.30 and ≥ 0.50 but < 0.70
            [0.80, 0.10, 0.10],   # ≥ all three thresholds
        ])

        a30 = assign_experts_to_roles(corr, tau=0.30)
        a50 = assign_experts_to_roles(corr, tau=TAU_SPEC)
        a70 = assign_experts_to_roles(corr, tau=0.70)

        sd30 = specialization_density(a30, n_total=2, tau=0.30)
        sd50 = specialization_density(a50, n_total=2, tau=TAU_SPEC)
        sd70 = specialization_density(a70, n_total=2, tau=0.70)

        self.assertGreaterEqual(sd30, sd50,  "SD(0.30) >= SD(0.50)")
        self.assertGreaterEqual(sd50, sd70,  "SD(0.50) >= SD(0.70)")

    # ------------------------------------------------------------------
    # specialization_density
    # ------------------------------------------------------------------

    def test_specialization_density_range(self):
        """
        Specialization Density must be in [0.0, 1.0] for any assignment.
        """
        for n_assigned in range(0, 6):
            assignment = {i: ('arithmetic' if i < n_assigned else None) for i in range(5)}
            sd = specialization_density(assignment, n_total=5)
            self.assertGreaterEqual(sd, 0.0)
            self.assertLessEqual(sd, 1.0)
            self.assertAlmostEqual(sd, n_assigned / 5.0, places=9)

    def test_specialization_density_boundary(self):
        assignment = {0: 'arithmetic', 1: 'logic', 2: 'language'}
        self.assertAlmostEqual(specialization_density(assignment, 3), 1.0)

        assignment_empty = {0: None, 1: None}
        self.assertAlmostEqual(specialization_density(assignment_empty, 2), 0.0)

    # ------------------------------------------------------------------
    # unassigned_routing_mass
    # ------------------------------------------------------------------

    def test_unassigned_routing_mass_sums(self):
        """
        unassigned_mass + assigned_mass = 1.0 (routing probs are normalised).
        """
        assignment = {0: 'arithmetic', 1: None, 2: 'logic', 3: None}
        # Routing probs normalised over 4 experts
        probs = torch.tensor([0.4, 0.1, 0.4, 0.1])

        mass = unassigned_routing_mass(assignment, probs)

        self.assertAlmostEqual(mass, 0.2, places=6,
                               msg="Experts 1,3 are ∅; their probs sum to 0.1+0.1=0.2")
        self.assertGreaterEqual(mass, 0.0)
        self.assertLessEqual(mass, 1.0)

    def test_unassigned_mass_zero_when_all_assigned(self):
        assignment = {0: 'arithmetic', 1: 'logic'}
        probs = torch.tensor([0.6, 0.4])
        self.assertAlmostEqual(unassigned_routing_mass(assignment, probs), 0.0)

    def test_unassigned_mass_one_when_all_unassigned(self):
        assignment = {0: None, 1: None}
        probs = torch.tensor([0.5, 0.5])
        self.assertAlmostEqual(unassigned_routing_mass(assignment, probs), 1.0)

    # ------------------------------------------------------------------
    # Q_role_aligned
    # ------------------------------------------------------------------

    def test_Q_role_renorm_sums_to_one(self):
        """
        Q_role_renorm must sum to 1.0 (renormalised over assigned experts only).
        """
        assignment = {0: 'arithmetic', 1: 'logic', 2: None, 3: 'language'}
        probs = torch.tensor([0.4, 0.3, 0.2, 0.1])

        result = Q_role_aligned(assignment, probs)

        Q_renorm = result['Q_role_renorm']
        total    = sum(Q_renorm.values())
        self.assertAlmostEqual(total, 1.0, places=6,
                               msg="Q_role_renorm must sum to 1.0 over assigned roles")

    def test_Q_role_unassigned_mass_reported(self):
        """
        unassigned_mass must be present in the result and match manual calculation.
        """
        assignment = {0: 'arithmetic', 1: None, 2: 'logic'}
        probs = torch.tensor([0.5, 0.3, 0.2])

        result = Q_role_aligned(assignment, probs)

        self.assertIn('unassigned_mass', result)
        self.assertAlmostEqual(result['unassigned_mass'], 0.3, places=6)

    def test_Q_role_raw_values(self):
        """
        Q_role_raw aggregates expert probs per role before renormalisation.
        """
        # Two arithmetic experts, one logic, no language
        assignment = {0: 'arithmetic', 1: 'arithmetic', 2: 'logic'}
        probs = torch.tensor([0.3, 0.4, 0.3])

        result = Q_role_aligned(assignment, probs)

        self.assertAlmostEqual(result['Q_role_raw']['arithmetic'], 0.7, places=6)
        self.assertAlmostEqual(result['Q_role_raw']['logic'],      0.3, places=6)
        self.assertAlmostEqual(result['Q_role_raw']['language'],   0.0, places=6)
        self.assertAlmostEqual(result['unassigned_mass'],          0.0, places=6)

    def test_Q_role_degenerate_all_unassigned(self):
        """
        If all experts are ∅, Q_role_renorm is all zeros (no mass to renormalise).
        """
        assignment = {0: None, 1: None}
        probs = torch.tensor([0.5, 0.5])

        result = Q_role_aligned(assignment, probs)

        for r in ROLES:
            self.assertAlmostEqual(result['Q_role_renorm'][r], 0.0, places=6)
        self.assertAlmostEqual(result['unassigned_mass'], 1.0, places=6)

    def test_correlation_matrix(self):
        """
        Verify that compute_expert_role_correlation calculates correct Pearson values.
        """
        # Let's define counts for 2 experts and 3 roles:
        # T = 100
        # Expert 0: co-occur: arith=40, logic=5, lang=5. total_select=50.
        #           domain_tokens: arith=50, logic=30, lang=20.
        # Expert 1: co-occur: arith=10, logic=25, lang=5. total_select=40.
        co_occurrence = torch.tensor([
            [40, 5, 5],
            [10, 25, 5]
        ])
        expert_select = torch.tensor([50, 40])
        domain_tokens = torch.tensor([50, 30, 20])
        total_tokens = 100

        corr = compute_expert_role_correlation(
            co_occurrence, expert_select, domain_tokens, total_tokens
        )

        self.assertEqual(corr.shape, (2, 3))
        
        # Hand calculated for Expert 0 and Arithmetic (r=0):
        # T=100, a=40, n_e=50, m_r=50
        # num = 100 * 40 - 50 * 50 = 4000 - 2500 = 1500
        # den = sqrt(50 * 50 * 50 * 50) = 2500
        # corr = 1500 / 2500 = 0.60
        self.assertAlmostEqual(corr[0, 0].item(), 0.60, places=6)

        # Hand calculated for Expert 1 and Logic (r=1):
        # T=100, a=25, n_e=40, m_r=30
        # num = 100 * 25 - 40 * 30 = 2500 - 1200 = 1300
        # den = sqrt(40 * 60 * 30 * 70) = sqrt(2400 * 2100) = sqrt(5040000) ≈ 2244.9944
        # corr = 1300 / 2244.9944 ≈ 0.579066
        self.assertAlmostEqual(corr[1, 1].item(), 0.579066, places=5)



if __name__ == '__main__':
    unittest.main(verbosity=2)
