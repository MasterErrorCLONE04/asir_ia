"""
training/router/alignment.py — Expert-to-role alignment (§6 spec v0.7)

§6 defines the assignment function:

    f: E → R ∪ {∅}   (many-to-one, not bijective)

    f(e) = argmax_r corr(e, r)   if max_r corr(e, r) >= tau_spec
           ∅                      otherwise

    tau_spec = 0.50  (frozen primary threshold, §6 v0.7)

DESIGN NOTE — Why this module does NOT define compute_expert_role_correlation():

    §6 freezes the assignment rule (argmax + threshold) but does NOT specify
    which statistical estimator produces corr(e, r) — Pearson, Spearman,
    mutual information, etc. are all possible and the choice is a methodological
    decision that must be documented explicitly when made.

    Introducing an implicit estimator here would create an undocumented degree
    of freedom. Instead, this module receives a pre-computed correlation matrix
    from the caller, who is responsible for declaring which estimator was used.

    The correlation computation will be added to analysis/n_effective.py (or
    similar) when the estimator choice is formally decided, not before.
"""

from typing import Dict, List, Optional
import torch

# Frozen role order — must match oracle.py domain ordering
ROLES: List[str] = ['arithmetic', 'logic', 'language']

# Frozen primary specialisation threshold (§6 v0.7)
TAU_SPEC: float = 0.50


# ---------------------------------------------------------------------------
# Core assignment
# ---------------------------------------------------------------------------

def assign_experts_to_roles(
    correlation: torch.Tensor,
    tau: float = TAU_SPEC,
) -> Dict[int, Optional[str]]:
    """
    Many-to-one assignment f: E → R ∪ {∅} per §6 v0.7.

    Multiple experts may map to the same role (many-to-one).
    Hungarian / bijective matching is NOT used here — it is only valid when
    N_total == N_roles exactly, which no current configuration satisfies (§6).

    Args:
        correlation: Tensor of shape (N_total, N_roles).
                     correlation[e, r] = pre-computed correlation between
                     expert e and role r.  The caller must declare which
                     statistical estimator produced this matrix.
        tau:         Specialisation threshold.  Default = TAU_SPEC = 0.50.
                     Sensitivity analysis with tau ∈ {0.30, 0.50, 0.70} is
                     obligatory per §6; this default is the primary metric.

    Returns:
        assignment: dict  expert_id (int) → role (str) | None
                    None  ≡  unassigned (∅).
    """
    if correlation.ndim != 2:
        raise ValueError(
            f"correlation must be 2-D (N_total, N_roles), got shape {tuple(correlation.shape)}"
        )
    n_total, n_roles = correlation.shape
    if n_roles != len(ROLES):
        raise ValueError(
            f"correlation has {n_roles} role columns, expected {len(ROLES)} ({ROLES})"
        )

    assignment: Dict[int, Optional[str]] = {}
    for e in range(n_total):
        max_corr_val, max_role_idx = correlation[e].max(dim=0)
        if max_corr_val.item() >= tau:
            assignment[e] = ROLES[int(max_role_idx.item())]
        else:
            assignment[e] = None  # ∅

    return assignment


# ---------------------------------------------------------------------------
# Specialisation Density
# ---------------------------------------------------------------------------

def specialization_density(
    assignment: Dict[int, Optional[str]],
    n_total: int,
    tau: float = TAU_SPEC,
) -> float:
    """
    Specialization Density per §6 v0.7:

        SD(tau) = |{ e : f(e) != ∅ }| / N_total

    The primary metric uses tau = TAU_SPEC = 0.50.
    Obligatory sensitivity analysis: call with tau ∈ {0.30, 0.50, 0.70}.

    Args:
        assignment: output of assign_experts_to_roles().
        n_total:    total number of experts (denominator).
        tau:        threshold used (logged for documentation; not re-applied here).

    Returns:
        float in [0.0, 1.0]
    """
    if len(assignment) != n_total:
        raise ValueError(
            f"assignment has {len(assignment)} entries but n_total={n_total}"
        )
    assigned_count = sum(1 for role in assignment.values() if role is not None)
    return assigned_count / n_total


# ---------------------------------------------------------------------------
# Unassigned routing mass
# ---------------------------------------------------------------------------

def unassigned_routing_mass(
    assignment: Dict[int, Optional[str]],
    routing_probs: torch.Tensor,
) -> float:
    """
    Unassigned routing mass per §6 v0.7:

        unassigned_mass = Σ_{e : f(e)=∅} Q(e)

    This value is ALWAYS reported alongside Q_role_renorm — never silently
    discarded.  A model with high unassigned mass can have a low KL (because
    the renorm excludes ∅) while being poorly specialised; both numbers are
    needed to tell the full story (§6).

    Args:
        assignment:    output of assign_experts_to_roles().
        routing_probs: (N_total,) tensor of routing probabilities aggregated
                       over the ENTIRE eval set (normalised, sums to 1).
                       Must be computed once over the full eval set, never
                       averaged per-batch (§9 v0.7: Jensen's inequality).

    Returns:
        float in [0.0, 1.0]
    """
    n_total = len(assignment)
    if routing_probs.shape != (n_total,):
        raise ValueError(
            f"routing_probs shape {tuple(routing_probs.shape)} != ({n_total},)"
        )

    unassigned_indices = [e for e, role in assignment.items() if role is None]
    if not unassigned_indices:
        return 0.0
    mass = routing_probs[unassigned_indices].sum().item()
    return float(mass)


# ---------------------------------------------------------------------------
# Role-aligned routing distribution  (§6 steps 1–3)
# ---------------------------------------------------------------------------

def Q_role_aligned(
    assignment: Dict[int, Optional[str]],
    routing_probs: torch.Tensor,
    roles: List[str] = ROLES,
) -> Dict[str, object]:
    """
    Converts per-expert routing probabilities → per-role distribution,
    renormalised excluding unassigned experts (∅), per §6 steps 1–3.

    Step 1 — Aggregate per role:
        Q_role(r) = Σ_{e : f(e)=r} Q(e)
        Q_role(∅) = Σ_{e : f(e)=∅} Q(e)

    Step 2 — The Oracle has no ∅ category (P_oracle sums to 1 over real roles),
    so the unassigned mass cannot be compared directly to P_oracle.
    For the primary KL divergence, renormalise excluding ∅:
        Q_role'(r) = Q_role(r) / Σ_{e: f(e)≠∅} Q(e)

    Step 3 — Report unassigned_mass separately; never discard it silently.

    Args:
        assignment:    output of assign_experts_to_roles().
        routing_probs: (N_total,) aggregated routing probs over eval set.
        roles:         list of role names; default = ROLES.

    Returns:
        dict with keys:
          'Q_role_raw'    : {role: float}  — mass before renorm
          'Q_role_renorm' : {role: float}  — use for KL vs P_oracle
          'unassigned_mass': float          — always reported alongside
    """
    n_total = len(assignment)
    if routing_probs.shape != (n_total,):
        raise ValueError(
            f"routing_probs shape {tuple(routing_probs.shape)} != ({n_total},)"
        )

    Q_raw: Dict[str, float] = {r: 0.0 for r in roles}
    unassigned_mass = 0.0

    for e, role in assignment.items():
        p = routing_probs[e].item()
        if role is not None:
            if role in Q_raw:
                Q_raw[role] += p
            # If role is not in expected roles list — shouldn't happen, but safe
        else:
            unassigned_mass += p

    assigned_mass = sum(Q_raw.values())

    if assigned_mass > 1e-9:
        Q_renorm: Dict[str, float] = {r: Q_raw[r] / assigned_mass for r in roles}
    else:
        # Degenerate: all routing mass is unassigned
        Q_renorm = {r: 0.0 for r in roles}

    return {
        'Q_role_raw':     Q_raw,
        'Q_role_renorm':  Q_renorm,
        'unassigned_mass': unassigned_mass,
    }


# ---------------------------------------------------------------------------
# Correlation computation (Interface implementation)
# ---------------------------------------------------------------------------

def compute_expert_role_correlation(
    co_occurrence_counts: torch.Tensor,
    expert_select_counts: torch.Tensor,
    domain_token_counts: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """
    Computes the correlation matrix between experts and roles/domains.

    METHODOLOGICAL NOTE:
        This implementation uses the Pearson correlation coefficient (Phi coefficient)
        for binary variables representing:
          X_e: Whether expert e was selected for a token (in top-k).
          Y_r: Whether the token belongs to domain r.

        Formula for Pearson correlation between binary variables:
          corr(e, r) = (T * a - n_e * m_r) / sqrt(n_e * (T - n_e) * m_r * (T - m_r))
        where:
          T = total_tokens
          a = co_occurrence_counts[e, r] (count where X_e=1 and Y_r=1)
          n_e = expert_select_counts[e] (count where X_e=1)
          m_r = domain_token_counts[r] (count where Y_r=1)

    Args:
        co_occurrence_counts: Tensor of shape (N_total, N_roles).
                              co_occurrence_counts[e, r] is the count of tokens of domain r
                              for which expert e was selected.
        expert_select_counts: Tensor of shape (N_total,).
                              expert_select_counts[e] is the total number of tokens for
                              which expert e was selected.
        domain_token_counts: Tensor of shape (N_roles,).
                             domain_token_counts[r] is the total number of tokens of domain r.
        total_tokens:        Total number of tokens evaluated.

    Returns:
        correlation: Tensor of shape (N_total, N_roles) with values in [-1.0, 1.0].
                     If denominator is 0 for an expert/role, correlation is set to 0.0.
    """
    n_total, n_roles = co_occurrence_counts.shape
    device = co_occurrence_counts.device
    
    correlation = torch.zeros((n_total, n_roles), dtype=torch.float32, device=device)
    
    T = float(total_tokens)
    if T <= 0:
        return correlation

    for e in range(n_total):
        n_e = float(expert_select_counts[e].item())
        for r in range(n_roles):
            m_r = float(domain_token_counts[r].item())
            a = float(co_occurrence_counts[e, r].item())
            
            num = T * a - n_e * m_r
            den = (n_e * (T - n_e) * m_r * (T - m_r)) ** 0.5
            
            if den > 1e-9:
                correlation[e, r] = num / den
            else:
                correlation[e, r] = 0.0
                
    return correlation
