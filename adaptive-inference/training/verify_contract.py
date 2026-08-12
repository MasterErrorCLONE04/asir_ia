"""
training/verify_contract.py — R1.0 Model Contract Verification

Produces the acceptance-criteria table required before advancing to R1.1:

  R1.0 MODEL CONTRACT
  ===================
  M1  B=...  K=2  E=...  A=...  ΔA_rel=...%  PASS
  ...
  Dense-A: A=...  ΔA_rel=...%  PASS

  Alignment:
      many-to-one            PASS
      tau_spec = 0.50        PASS
      specialization density [0,1] PASS
      unassigned mass        [0,1] PASS
      Q_role renormalization PASS

Usage (from /workspace):
    python training/verify_contract.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.models.transformer import MoETransformer
from training.train import get_model_config, count_active_params
from training.router.alignment import (
    ROLES, TAU_SPEC,
    assign_experts_to_roles,
    specialization_density,
    unassigned_routing_mass,
    Q_role_aligned,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_A    = 140_000_000
VOCAB_SIZE  = 100
D_MODEL     = 1408
N_LAYERS    = 5
NUM_HEADS   = 8

MoE_TOL   = 0.001   # < 0.1%  per §2 v0.7
DENSE_TOL = 0.010   # ≤ 1.0%  per §2 v0.7

# Build real PyTorch models only for configs that fit comfortably in CPU RAM.
# M4 (512 experts×5 layers) and M5/C3/C4 (896 experts×5) require >4 GB RAM
# for the full parameter tensor allocation and will OOM the container.
MoE_SMALL   = ['M1', 'M2', 'M3']        # < 128 experts — instantiate
MoE_LARGE   = ['M4', 'M5']             # 512/896 experts — analytical only
EXP_C_SMALL = ['C1', 'C2']             # 32/128 experts — instantiate
EXP_C_LARGE = ['C3', 'C4']             # 512/896 experts — analytical only
DENSE_CONFIGS = ['M0', 'Dense-A']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(name: str) -> MoETransformer:
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


def _check_model(name: str, tol: float, analytical_only: bool = False) -> tuple:
    """Returns (counts_dict, delta_rel, passed).

    If analytical_only=True, skips PyTorch model instantiation and verifies
    only the formula integrity from get_model_config.  Use for large configs
    (M4, M5, C3, C4) that would OOM a CPU-only Docker container.
    """
    cfg = get_model_config(name)

    if analytical_only:
        # Formula-only verification — no model instantiation
        assert cfg['B'] + cfg['K'] * cfg['E'] == cfg['A'], \
            f"{name}: B + K*E != A in config dict"
        counts = {'B': cfg['B'], 'K': cfg['K'], 'E': cfg['E'], 'A': cfg['A'], 'total': None}
    else:
        model  = _build(name)
        counts = count_active_params(model)
        # Cross-check analytical vs actual
        assert counts['B'] == cfg['B'], \
            f"{name}: B mismatch analytical={cfg['B']:,} actual={counts['B']:,}"
        assert counts['K'] == cfg['K'], \
            f"{name}: K mismatch analytical={cfg['K']} actual={counts['K']}"
        assert counts['E'] == cfg['E'], \
            f"{name}: E mismatch analytical={cfg['E']:,} actual={counts['E']:,}"
        assert counts['A'] == cfg['A'], \
            f"{name}: A mismatch analytical={cfg['A']:,} actual={counts['A']:,}"

    delta_rel = abs(counts['A'] - TARGET_A) / TARGET_A
    passed = delta_rel < tol
    return counts, delta_rel, passed


def _print_model_row(name: str, counts: dict, delta_rel: float, passed: bool, note: str = ''):
    status = "PASS" if passed else "FAIL"
    print(
        f"  {name:<8s}  "
        f"B={counts['B']:>13,}  "
        f"K={counts['K']}  "
        f"E={counts['E']:>13,}  "
        f"A={counts['A']:>13,}  "
        f"ΔA_rel={delta_rel*100:6.4f}%  "
        f"{status}"
        + (f"  [{note}]" if note else "")
    )


# ---------------------------------------------------------------------------
# Alignment contract check (synthetic matrices — no trained model needed)
# ---------------------------------------------------------------------------

def _check_alignment() -> bool:
    """
    Verifies the §6 alignment contract using synthetic correlation matrices.
    Returns True if all checks pass.
    """
    all_pass = True
    results  = {}

    # --- many-to-one ---
    corr_m2o = torch.tensor([
        [0.80, 0.10, 0.10],   # → arithmetic
        [0.70, 0.20, 0.10],   # → arithmetic  (many-to-one)
        [0.10, 0.90, 0.00],   # → logic
        [0.20, 0.30, 0.50],   # → language    (exactly at tau)
        [0.30, 0.35, 0.35],   # → None (∅)
    ])
    a = assign_experts_to_roles(corr_m2o, tau=TAU_SPEC)
    m2o_ok = (
        a[0] == 'arithmetic' and
        a[1] == 'arithmetic' and   # many-to-one
        a[2] == 'logic'      and
        a[3] == 'language'   and   # >= tau → assigned
        a[4] is None               # < tau → ∅
    )
    results['many-to-one'] = m2o_ok
    all_pass = all_pass and m2o_ok

    # --- tau_spec = 0.50 ---
    tau_ok = abs(TAU_SPEC - 0.50) < 1e-9
    results['tau_spec = 0.50'] = tau_ok
    all_pass = all_pass and tau_ok

    # --- specialization density ∈ [0, 1] ---
    sd_values = []
    for n_assigned in range(0, 6):
        asn = {i: ('arithmetic' if i < n_assigned else None) for i in range(5)}
        sd  = specialization_density(asn, n_total=5)
        sd_values.append(sd)
    sd_ok = all(0.0 <= v <= 1.0 for v in sd_values)
    results['specialization density [0,1]'] = sd_ok
    all_pass = all_pass and sd_ok

    # --- unassigned_routing_mass ∈ [0, 1] ---
    asn_mixed = {0: 'arithmetic', 1: None, 2: 'logic', 3: None}
    probs     = torch.tensor([0.4, 0.1, 0.4, 0.1])
    mass      = unassigned_routing_mass(asn_mixed, probs)
    mass_ok   = (0.0 <= mass <= 1.0) and abs(mass - 0.2) < 1e-6
    results['unassigned mass [0,1]'] = mass_ok
    all_pass = all_pass and mass_ok

    # --- Q_role renormalization sums to 1 ---
    asn_renorm = {0: 'arithmetic', 1: 'logic', 2: None, 3: 'language'}
    probs_r    = torch.tensor([0.4, 0.3, 0.2, 0.1])
    qr         = Q_role_aligned(asn_renorm, probs_r)
    renorm_sum = sum(qr['Q_role_renorm'].values())
    renorm_ok  = abs(renorm_sum - 1.0) < 1e-6
    results['Q_role renormalization'] = renorm_ok
    all_pass = all_pass and renorm_ok

    print("\nAlignment:")
    for check, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"    {check:<35s}  {status}")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("R1.0 MODEL CONTRACT")
    print("=" * 80)

    all_pass = True

    # --- EXP-B: M1–M3 (full cross-validation) ---
    print(f"\nEXP-B (small configs, formula + PyTorch cross-validation):")
    for name in MoE_SMALL:
        counts, delta_rel, ok = _check_model(name, MoE_TOL, analytical_only=False)
        _print_model_row(name, counts, delta_rel, ok)
        all_pass = all_pass and ok

    # --- EXP-B: M4–M5 (analytical only — OOM guard) ---
    print(f"\nEXP-B (large configs, analytical only — OOM guard):")
    for name in MoE_LARGE:
        counts, delta_rel, ok = _check_model(name, MoE_TOL, analytical_only=True)
        _print_model_row(name, counts, delta_rel, ok, note='analytical')
        all_pass = all_pass and ok

    # --- EXP-C: C1–C2 (full cross-validation) ---
    print(f"\nEXP-C (small configs, formula + PyTorch cross-validation):")
    for name in EXP_C_SMALL:
        counts, delta_rel, ok = _check_model(name, MoE_TOL, analytical_only=False)
        _print_model_row(name, counts, delta_rel, ok)
        all_pass = all_pass and ok

    # --- EXP-C: C3–C4 (analytical only — OOM guard) ---
    print(f"\nEXP-C (large configs, analytical only — OOM guard):")
    for name in EXP_C_LARGE:
        counts, delta_rel, ok = _check_model(name, MoE_TOL, analytical_only=True)
        _print_model_row(name, counts, delta_rel, ok, note='analytical')
        all_pass = all_pass and ok

    # --- Dense baselines (full cross-validation) ---
    print(f"\nDense baselines (tolerance ≤ 1%):")
    for name in DENSE_CONFIGS:
        counts, delta_rel, ok = _check_model(name, DENSE_TOL, analytical_only=False)
        _print_model_row(name, counts, delta_rel, ok)
        all_pass = all_pass and ok

    # --- Alignment ---
    align_ok = _check_alignment()
    all_pass = all_pass and align_ok

    # --- Summary ---
    print()
    print("-" * 80)
    if all_pass:
        print("CONTRACT STATUS: ALL PASS — R1.0 model contract satisfied.")
    else:
        print("CONTRACT STATUS: FAIL — see FAIL rows above before proceeding.")
    print()
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
