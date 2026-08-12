"""
analysis/evaluator.py — Statistical evaluation engine for ESP@k, RSE@k, and Paired Bootstrap (spec v0.12-final).

Strictly enforces:
1. Resampling unit is EXPLICITLY the example of C.
2. Paired evaluation: Single set of boot_indices I_b per replicate b applied to selector, search-oracle, and random matrix.
3. Denominator stability check: LCI95%(Δ_search-random) > 0 => denominator_status = interpretable.
4. NO POST-HOC EXCLUSION: If any replicate b has Δ_b <= 0 or non-finite, ratio_status = non_estimable, RSE@k = null.
5. Complete auditability via bootstrap_diagnostics.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def compute_paired_bootstrap_metrics(
    q_selector_examples: np.ndarray,      # Shape: (N_examples_C,)
    q_oracle_examples: np.ndarray,        # Shape: (N_examples_C,)
    q_random_examples_matrix: np.ndarray, # Shape: (N_random, N_examples_C)
    n_replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes ESP@k, RSE@k, confidence intervals, and bootstrap diagnostics.
    """
    n_examples = len(q_selector_examples)
    n_random = q_random_examples_matrix.shape[0]

    # Point estimates on test dataset C
    q_selector_pt = float(np.mean(q_selector_examples))
    q_oracle_pt = float(np.mean(q_oracle_examples))
    q_random_mean_per_ex = np.mean(q_random_examples_matrix, axis=0) # (N_examples_C,)
    q_random_pt = float(np.mean(q_random_mean_per_ex))

    esp_pt = q_selector_pt - q_oracle_pt
    delta_sr_pt = q_oracle_pt - q_random_pt
    rse_pt = (q_selector_pt - q_random_pt) / delta_sr_pt if abs(delta_sr_pt) > 1e-12 else None

    # --- PAIRED BOOTSTRAP ---
    # 1. Single resample of example indices per replicate b
    rng = np.random.default_rng(seed)
    boot_indices = rng.choice(n_examples, size=(n_replicates, n_examples), replace=True)

    # 2. Per-replicate mean scores over resampled example indices I_b
    q_selector_b = np.mean(q_selector_examples[boot_indices], axis=1) # (n_replicates,)
    q_oracle_b = np.mean(q_oracle_examples[boot_indices], axis=1)     # (n_replicates,)

    # Random matrix: (N_random, n_replicates, n_examples) -> mean per example -> mean per replicate
    q_random_b_matrix = np.mean(q_random_examples_matrix[:, boot_indices], axis=2) # (N_random, n_replicates)
    q_random_mean_b = np.mean(q_random_b_matrix, axis=0)                            # (n_replicates,)

    # 3. Bootstrap distributions
    esp_b = q_selector_b - q_oracle_b
    delta_sr_b = q_oracle_b - q_random_mean_b
    num_b = q_selector_b - q_random_mean_b

    # Confidence interval percentiles
    alpha = 1.0 - confidence_level
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100

    esp_ci95 = [float(np.percentile(esp_b, lower_p)), float(np.percentile(esp_b, upper_p))]
    delta_sr_ci95 = [float(np.percentile(delta_sr_b, lower_p)), float(np.percentile(delta_sr_b, upper_p))]

    # 4. Status determination for ESP
    if esp_ci95[0] > 0:
        esp_status = "selector_search_inversion_on_C"
    elif esp_ci95[1] < 0:
        esp_status = "selector_below_search_oracle"
    else:
        esp_status = "statistically_indistinguishable_from_zero"

    # 5. Status determination for Denominator (LCI95% > 0 rule)
    if delta_sr_ci95[0] > 0:
        denominator_status = "interpretable"
    elif delta_sr_ci95[1] < 0:
        denominator_status = "oracle_below_random"
    else:
        denominator_status = "unstable_denominator"

    # 6. Exact ratio replicate diagnostics (definition exactness)
    invalid_denom_mask = (delta_sr_b <= 0.0) | (~np.isfinite(delta_sr_b))
    invalid_denom_count = int(np.sum(invalid_denom_mask))

    # Ratio is finite real only when denominator > 0 and numerator is finite
    invalid_ratio_mask = invalid_denom_mask | (~np.isfinite(num_b))
    invalid_ratio_count = int(np.sum(invalid_ratio_mask))
    invalid_fraction = float(invalid_ratio_count / n_replicates)

    # 7. Ratio status determination under strictly FALSE post-hoc exclusion
    if invalid_ratio_count > 0 or denominator_status != "interpretable":
        ratio_status = "non_estimable"
        rse_final = None
        rse_ci95 = None
    else:
        ratio_status = "estimable"
        rse_b = num_b / delta_sr_b
        rse_final = float(rse_pt)
        rse_ci95 = [float(np.percentile(rse_b, lower_p)), float(np.percentile(rse_b, upper_p))]

    # 8. Assemble reproducible rse_result structure
    rse_result = {
        "metrics": {
            "ESP": {
                "value": float(esp_pt),
                "ci95": esp_ci95,
                "status": esp_status
            },
            "denominator": {
                "delta_search_random": float(delta_sr_pt),
                "ci95": delta_sr_ci95,
                "status": denominator_status
            },
            "RSE": {
                "value": rse_final,
                "ci95": rse_ci95,
                "denominator_status": denominator_status,
                "ratio_status": ratio_status
            }
        },
        "bootstrap_diagnostics": {
            "total_replicates": n_replicates,
            "invalid_denominator_replicates": invalid_denom_count,
            "invalid_ratio_replicates": invalid_ratio_count,
            "invalid_fraction": invalid_fraction,
            "posthoc_exclusion": False
        }
    }

    return rse_result
