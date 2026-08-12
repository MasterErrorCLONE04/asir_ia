"""
analysis/config_schema.py — Rigorous configuration schema and validation for R1.3 (spec v0.12-final).

Enforces that:
1. `bootstrap_replicates` is explicitly declared (no unstated experimental defaults).
2. `oracle_search.method` is explicitly specified (e.g. 'greedy', 'beam', 'exhaustive').
3. `posthoc_exclusion` is strictly False.
4. `subsets_frozen_for_evaluation` is True and random subsets resampling is False.
5. All required metadata for auditable results is present.
"""

import json
from typing import Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

VALID_ORACLE_METHODS = {"greedy", "beam", "exhaustive", "random_search"}
REQUIRED_METRIC_FIELDS = {"name", "source_metric", "transform", "direction", "unit"}
REQUIRED_DIAGNOSTIC_FIELDS = {
    "total_replicates",
    "invalid_denominator_replicates",
    "invalid_ratio_replicates",
    "invalid_fraction"
}


class ConfigValidationError(ValueError):
    """Raised when configuration violates the v0.12-final specification contract."""
    pass


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a loaded configuration dictionary against spec v0.12-final rules.
    Raises ConfigValidationError if any mandatory rule is violated.
    """
    # 1. Experiment metadata
    exp = cfg.get("experiment", {})
    if exp.get("contract_version") != "v0.12-final":
        raise ConfigValidationError(
            f"Invalid contract_version '{exp.get('contract_version')}'. Expected 'v0.12-final'."
        )
    if "phase" not in exp or "domain" not in exp:
        raise ConfigValidationError("Experiment section must contain 'phase' and 'domain'.")

    # 2. Quality metric
    qm = cfg.get("quality_metric", {})
    missing_qm = REQUIRED_METRIC_FIELDS - set(qm.keys())
    if missing_qm:
        raise ConfigValidationError(f"quality_metric is missing required fields: {missing_qm}")
    if qm.get("direction") != "maximize":
        raise ConfigValidationError("quality_metric.direction must be 'maximize'.")

    # 3. Candidate space
    cs = cfg.get("candidate_space", {})
    if "expert_pool_size" not in cs or "subset_size_k" not in cs:
        raise ConfigValidationError("candidate_space requires 'expert_pool_size' and 'subset_size_k'.")

    # 4. Oracle search
    os_cfg = cfg.get("oracle_search", {})
    method = os_cfg.get("method")
    if method not in VALID_ORACLE_METHODS:
        raise ConfigValidationError(
            f"oracle_search.method '{method}' is invalid. Must be one of {VALID_ORACLE_METHODS}."
        )
    tb = os_cfg.get("tie_breaking", {})
    if not tb.get("policy") or not tb.get("rule"):
        raise ConfigValidationError("oracle_search.tie_breaking requires 'policy' and 'rule'.")
    sd = os_cfg.get("search_determinism", {})
    if "deterministic" not in sd:
        raise ConfigValidationError("oracle_search.search_determinism requires 'deterministic'.")

    # 5. Random reference
    rr = cfg.get("random_reference", {})
    if not rr.get("subsets_frozen_for_evaluation", False):
        raise ConfigValidationError("random_reference.subsets_frozen_for_evaluation must be True.")
    if rr.get("resample_random_subsets_in_bootstrap", True):
        raise ConfigValidationError("random_reference.resample_random_subsets_in_bootstrap must be False.")
    if rr.get("N_random", 0) < 5:
        raise ConfigValidationError("random_reference.N_random must be at least 5.")

    # 6. Grouping (near-duplicates)
    grp = cfg.get("grouping", {})
    if "method" not in grp or "implementation_version" not in grp:
        raise ConfigValidationError("grouping requires 'method' and 'implementation_version'.")

    # 7. Uncertainty & Bootstrap
    unc = cfg.get("uncertainty", {})
    if "bootstrap_replicates" not in unc:
        raise ConfigValidationError(
            "uncertainty.bootstrap_replicates MUST be explicitly declared in config (no default allowed)."
        )
    b_reps = unc["bootstrap_replicates"]
    if not isinstance(b_reps, int) or b_reps <= 0:
        raise ConfigValidationError("uncertainty.bootstrap_replicates must be a positive integer.")

    # 8. Ratio uncertainty
    r_unc = cfg.get("ratio_uncertainty", {})
    if r_unc.get("posthoc_exclusion", True) is not False:
        raise ConfigValidationError("ratio_uncertainty.posthoc_exclusion MUST be strictly False.")
    
    diag_fields = set(r_unc.get("diagnostics_reporting", {}).get("required_fields", []))
    if not REQUIRED_DIAGNOSTIC_FIELDS.issubset(diag_fields):
        raise ConfigValidationError(
            f"ratio_uncertainty.diagnostics_reporting must include all required fields: {REQUIRED_DIAGNOSTIC_FIELDS}"
        )

    return cfg


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """
    Loads JSON or YAML config from path and validates it against spec v0.12-final rules.
    """
    if config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif HAS_YAML:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        # If pyyaml is missing, check for sibling .json file
        json_path = config_path.rsplit(".", 1)[0] + ".json"
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    return validate_config(cfg)

