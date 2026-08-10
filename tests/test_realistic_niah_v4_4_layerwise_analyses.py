from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _design() -> dict:
    return {
        "rank": 3,
        "confirmation_seeds": [1, 2, 3],
        "prompt_removal": {
            "counts": [2, 3],
            "layers": {"Tiny": [0, 1]},
            "realized_norm_relative_tolerance": 0.025,
        },
        "answer_transport": {
            "pairs": [[1, 2], [2, 1]],
            "boundaries": {"Tiny": [[0, 1], [1, 2]]},
            "primary_endpoint": "target_donor_fraction",
        },
        "multiplicity": {
            "prompt_removal": "Holm across registered layers",
            "answer_transport": "Holm across registered boundaries",
        },
    }


def test_prompt_removal_analysis_accepts_a_complete_registered_grid(
    tmp_path: Path, monkeypatch
) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    rows = []
    for seed in (1, 2, 3):
        for count in (2, 3):
            for layer in (0, 1):
                for condition in (
                    "actual_rank3_remove",
                    "actual_normmatched_orthogonal",
                ):
                    candidate = condition == "actual_rank3_remove"
                    rows.append(
                        {
                            "model_label": "Tiny",
                            "seed": seed,
                            "gold_count": count,
                            "layer": layer,
                            "normalized_depth": float(layer),
                            "condition": condition,
                            "prediction": count + int(candidate),
                            "correct": int(not candidate),
                            "absolute_error": int(candidate),
                            "signed_error": int(candidate),
                            "clean_prediction": count,
                            "clean_correct": 1,
                            "clean_absolute_error": 0,
                            "removed_fro_norm": 2.0,
                            "target_removed_fro_norm": 2.0,
                            "norm_ratio": 1.0,
                            "completion": str(count),
                            "runtime_seconds": 0.1,
                        }
                    )
    input_path = tmp_path / "layerwise_prompt_removal_detail.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output = tmp_path / "prompt-analysis"
    module = _load_script("analyze_v446_layerwise_prompt_removal")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            str(input_path),
            "--design-config",
            str(design_path),
            "--output",
            str(output),
            "--bootstraps",
            "100",
        ],
    )
    module.main()

    statistics = pd.read_csv(output / "layerwise_prompt_removal_statistics.csv")
    assert len(statistics) == 8
    assert set(statistics["mean_effect"]) == {1.0}
    audit = json.loads((output / "analysis_audit.json").read_text())
    assert audit["status"] == "PASS"


def test_transport_analysis_accepts_a_complete_registered_grid(
    tmp_path: Path, monkeypatch
) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    rows = []
    condition_values = {
        "aligned_dose_1": 0.4,
        "aligned_dose_2": 0.8,
        "matched_orthogonal": 0.0,
    }
    for seed in (1, 2, 3):
        for receiver, donor in ((1, 2), (2, 1)):
            for source, target in ((0, 1), (1, 2)):
                for condition, effect in condition_values.items():
                    dose = 2.0 if condition == "aligned_dose_2" else 1.0
                    rows.append(
                        {
                            "model_label": "Tiny",
                            "seed": seed,
                            "receiver_count": receiver,
                            "donor_count": donor,
                            "support": "answer_query_relay",
                            "condition": condition,
                            "source_layer": source,
                            "target_layer": target,
                            "normalized_depth": target / 2,
                            "replacement_delta_norm": dose,
                            "aligned_dose_1_norm": 1.0,
                            "clean_donor_log_odds": -1.0,
                            "condition_donor_log_odds": -1.0 + effect,
                            "donor_log_odds_gain": effect,
                            "target_donor_fraction": effect,
                            "argmax_token_changed": 0,
                            "geometry_discovery_centroid_r2": 0.9,
                            "runtime_seconds": 0.1,
                        }
                    )
    input_path = tmp_path / "layerwise_transport_patch.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output = tmp_path / "transport-analysis"
    module = _load_script("analyze_v446_layerwise_transport_patch")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            str(input_path),
            "--design-config",
            str(design_path),
            "--output",
            str(output),
            "--bootstraps",
            "100",
        ],
    )
    module.main()

    statistics = pd.read_csv(output / "layerwise_transport_statistics.csv")
    assert len(statistics) == 12
    primary = statistics[
        statistics["contrast"] == "aligned_dose_1_minus_orthogonal"
    ]
    assert set(primary["mean_contrast"].round(8)) == {0.4}
    audit = json.loads((output / "analysis_audit.json").read_text())
    assert audit["status"] == "PASS"
