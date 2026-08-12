"""
tests/test_rse_contract.py — Master contract test suite for R1.3 (spec v0.12-final).
"""

import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.config_schema import validate_config, ConfigValidationError, load_and_validate_config

from analysis.partitioning import partition_dataset_abc
from analysis.oracle_search import run_oracle_search
from analysis.random_reference import sample_random_reference_subsets
from analysis.artifacts import save_artifact_snapshot


class TestRSEContract(unittest.TestCase):

    def test_config_missing_bootstrap_replicates(self):
        invalid_cfg = {
            "experiment": {"contract_version": "v0.12-final", "phase": "R1.3", "domain": "Coding"},
            "quality_metric": {"name": "loss", "source_metric": "ce", "transform": "negate", "direction": "maximize", "unit": "nats"},
            "candidate_space": {"expert_pool_size": 16, "subset_size_k": 4},
            "oracle_search": {"method": "greedy", "tie_breaking": {"policy": "deterministic", "rule": "lexicographic_expert_id"}, "search_determinism": {"deterministic": True}},
            "random_reference": {"subsets_frozen_for_evaluation": True, "resample_random_subsets_in_bootstrap": False, "N_random": 10},
            "grouping": {"method": "exact_hash", "implementation_version": "v1.0"},
            "uncertainty": {}, # MISSING bootstrap_replicates
            "ratio_uncertainty": {"posthoc_exclusion": False, "diagnostics_reporting": {"required_fields": ["total_replicates", "invalid_denominator_replicates", "invalid_ratio_replicates", "invalid_fraction"]}}
        }
        with self.assertRaises(ConfigValidationError):
            validate_config(invalid_cfg)

    def test_config_posthoc_exclusion_must_be_false(self):
        invalid_cfg = {
            "experiment": {"contract_version": "v0.12-final", "phase": "R1.3", "domain": "Coding"},
            "quality_metric": {"name": "loss", "source_metric": "ce", "transform": "negate", "direction": "maximize", "unit": "nats"},
            "candidate_space": {"expert_pool_size": 16, "subset_size_k": 4},
            "oracle_search": {"method": "greedy", "tie_breaking": {"policy": "deterministic", "rule": "lexicographic_expert_id"}, "search_determinism": {"deterministic": True}},
            "random_reference": {"subsets_frozen_for_evaluation": True, "resample_random_subsets_in_bootstrap": False, "N_random": 10},
            "grouping": {"method": "exact_hash", "implementation_version": "v1.0"},
            "uncertainty": {"bootstrap_replicates": 1000},
            "ratio_uncertainty": {"posthoc_exclusion": True, "diagnostics_reporting": {"required_fields": ["total_replicates", "invalid_denominator_replicates", "invalid_ratio_replicates", "invalid_fraction"]}} # POSTHOC EXCLUSION = TRUE (INVALID)
        }
        with self.assertRaises(ConfigValidationError):
            validate_config(invalid_cfg)

    def test_preregistration_config_validates_cleanly(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "r1.3_preregistration_v0.12.yaml"
        )
        cfg = load_and_validate_config(config_path)
        self.assertEqual(cfg["experiment"]["contract_version"], "v0.12-final")
        self.assertEqual(cfg["uncertainty"]["bootstrap_replicates"], 10000)

    def test_full_pipeline_artifacts_saving(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "r1.3_preregistration_v0.12.yaml"
        )
        cfg = load_and_validate_config(config_path)

        # Synthetic data
        examples = [{"id": f"ex_{i}", "content": f"data {i}"} for i in range(30)]
        a, b, c, grouping_card = partition_dataset_abc(examples, cfg["grouping"])

        s_oracle, oracle_card = run_oracle_search(lambda s: 1.0, cfg["oracle_search"])
        s_random, random_card = sample_random_reference_subsets(16, 4, 5)

        rse_result = {
            "metrics": {"ESP": {"value": -0.1, "status": "ok"}},
            "bootstrap_diagnostics": {"invalid_fraction": 0.0}
        }

        out_dir = os.path.join("adaptive-inference", "tests", "tmp_results_test")
        saved = save_artifact_snapshot(out_dir, cfg, grouping_card, oracle_card, random_card, rse_result)

        self.assertTrue(os.path.exists(saved["config_snapshot"]))
        self.assertTrue(os.path.exists(saved["grouping"]))
        self.assertTrue(os.path.exists(saved["oracle_search"]))
        self.assertTrue(os.path.exists(saved["random_reference"]))
        self.assertTrue(os.path.exists(saved["rse_result"]))


if __name__ == "__main__":
    unittest.main()

