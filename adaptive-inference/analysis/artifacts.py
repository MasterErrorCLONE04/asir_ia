"""
analysis/artifacts.py — Serializer and artifact manager for reproducible R1.3 evaluation results.

Saves structured JSON/YAML artifacts under results/<domain>/k_<k>/ to guarantee complete auditability:
- config_snapshot.json / config_snapshot.yaml
- grouping.json
- oracle_search.json
- random_reference.json
- rse_result.json
"""

import os
import json
from typing import Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def ensure_dir(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)


def save_artifact_snapshot(
    output_dir: str,
    config: Dict[str, Any],
    grouping_card: Dict[str, Any],
    oracle_search_card: Dict[str, Any],
    random_reference_card: Dict[str, Any],
    rse_result: Dict[str, Any]
):
    """
    Saves all 5 required reproducible artifacts into output_dir.
    """
    ensure_dir(output_dir)

    # 1. config_snapshot.json / yaml
    if HAS_YAML:
        config_path = os.path.join(output_dir, "config_snapshot.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)
    else:
        config_path = os.path.join(output_dir, "config_snapshot.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    # 2. grouping.json
    grouping_path = os.path.join(output_dir, "grouping.json")
    with open(grouping_path, "w", encoding="utf-8") as f:
        json.dump(grouping_card, f, indent=2)

    # 3. oracle_search.json
    oracle_path = os.path.join(output_dir, "oracle_search.json")
    with open(oracle_path, "w", encoding="utf-8") as f:
        json.dump(oracle_search_card, f, indent=2)

    # 4. random_reference.json
    random_path = os.path.join(output_dir, "random_reference.json")
    with open(random_path, "w", encoding="utf-8") as f:
        json.dump(random_reference_card, f, indent=2)

    # 5. rse_result.json
    result_path = os.path.join(output_dir, "rse_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(rse_result, f, indent=2)

    return {
        "config_snapshot": config_path,
        "grouping": grouping_path,
        "oracle_search": oracle_path,
        "random_reference": random_path,
        "rse_result": result_path
    }

