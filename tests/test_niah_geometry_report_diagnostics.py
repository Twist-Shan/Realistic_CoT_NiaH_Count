from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_native_geometry_bands import (
    center_within_trajectory,
    two_band_fit,
)
from scripts.augment_niah_geometry_comparison_report import (
    END,
    BEGIN,
    TRACE_CATEGORIES,
    fisher_exact_two_sided,
    legacy_compatible_marker_summary,
    qwen_band_marker_analysis,
    remove_block,
    trace_category_summary,
    unresolved_trace_examples,
)
from scripts.build_niah_geometry_comparison_report_v7 import _band_verdict


def test_center_within_trajectory_removes_each_trajectory_mean() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["discovery"] * 4,
            "seed": [1, 1, 2, 2],
            "gold_count": [10] * 4,
        }
    )
    states = np.asarray([[1.0, 2.0], [3.0, 6.0], [10.0, 4.0], [14.0, 8.0]])
    centered = center_within_trajectory(states, metadata)
    np.testing.assert_allclose(centered[:2].mean(axis=0), 0.0)
    np.testing.assert_allclose(centered[2:].mean(axis=0), 0.0)


def test_two_band_fit_names_the_higher_display_cluster_upper() -> None:
    coordinates = np.asarray(
        [
            [-0.1, -10.0, 0.0],
            [0.1, -9.0, 0.0],
            [-0.1, 9.0, 0.0],
            [0.1, 10.0, 0.0],
        ]
    )
    result = two_band_fit(coordinates, random_state=0)
    assert result["cluster_sizes"] == {"upper": 2, "lower": 2}
    assert result["display_vertical_means"]["upper"] > result[
        "display_vertical_means"
    ]["lower"]


def test_trace_category_summary_uses_trajectory_denominators() -> None:
    rows = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for index in range(300):
            rows.append(
                {
                    "model_label": model,
                    "split": "discovery" if index < 200 else "confirmation",
                    "trace_category": TRACE_CATEGORIES[index % len(TRACE_CATEGORIES)],
                }
            )
    summary = trace_category_summary(rows)
    assert summary["Qwen3-8B"]["all"]["total"] == 300
    assert summary["Gemma4-E4B"]["discovery"]["total"] == 200
    assert summary["Gemma4-E4B"]["confirmation"]["total"] == 100
    assert sum(summary["Qwen3-8B"]["all"]["counts"].values()) == 300


def test_remove_block_also_removes_adjacent_newlines() -> None:
    text = f"before\n\n{BEGIN}payload{END}\n\nafter"
    assert remove_block(text, BEGIN, END) == "beforeafter"


def test_fisher_exact_two_sided_matches_qwen_trajectory_table() -> None:
    assert np.isclose(
        fisher_exact_two_sided(((4, 0), (0, 6))), 0.004761904761904762
    )


def test_qwen_band_marker_analysis_distinguishes_marker_taxonomies() -> None:
    points = [
        {
            "request_id": "recap",
            "seed": "1",
            "marker_kind": "completion_recap",
            "band": "lower",
        },
        {
            "request_id": "recap",
            "seed": "1",
            "marker_kind": "completion_recap",
            "band": "lower",
        },
        {
            "request_id": "indexed",
            "seed": "2",
            "marker_kind": "indexed",
            "band": "upper",
        },
        {
            "request_id": "indexed",
            "seed": "2",
            "marker_kind": "indexed",
            "band": "upper",
        },
    ]
    parser_rows = [
        {
            "request_id": "recap",
            "model_label": "Qwen3-8B",
            "marker_kind": "inline_count",
        },
        {
            "request_id": "indexed",
            "model_label": "Qwen3-8B",
            "marker_kind": "inline_count",
        },
    ]
    result = qwen_band_marker_analysis(points, parser_rows)
    assert result["legacy_family_counts"] == {
        "completion_recap": {"lower": 2, "upper": 0},
        "other_marker": {"lower": 0, "upper": 2},
    }
    assert result["hybrid_counts"] == {
        "inline_count": {"lower": 2, "upper": 2}
    }
    assert result["hybrid_nmi"] == 0.0


def test_legacy_marker_summary_keeps_unresolved_in_full_denominator() -> None:
    rows = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for index in range(300):
            rows.append(
                {
                    "model_label": model,
                    "split": "discovery" if index < 200 else "confirmation",
                    "old_marker_kind": None if index == 0 else "indexed",
                }
            )
    summary = legacy_compatible_marker_summary(rows)
    assert summary["Qwen3-8B"]["all"]["total"] == 300
    assert summary["Qwen3-8B"]["all"]["counts"]["unresolved"] == 1
    assert summary["Gemma4-E4B"]["discovery"]["counts"]["indexed"] == 199


def test_unresolved_trace_examples_attaches_raw_reasoning(tmp_path: Path) -> None:
    trace_dir = tmp_path / "Qwen3-8B"
    trace_dir.mkdir(parents=True)
    record = {
        "request_id": "qwen/miss",
        "raw_output_text": "<think>Paris is one. Lima is second.</think>\nTotal: 2",
        "trace_parse": {
            "parser": {
                "status": "no_terminated_gold_city_list",
                "candidates_considered": 0,
            }
        },
    }
    (trace_dir / "generations.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    rows = [
        {
            "request_id": "qwen/miss",
            "model_label": "Qwen3-8B",
            "seed": 1,
            "split": "confirmation",
            "gold_count": 2,
            "final_parsed_count": 2,
            "old_marker_kind": None,
            "marker_kind": "inline_count",
            "item_count": 2,
            "trace_category": "one_to_one",
        }
    ]
    examples, inputs = unresolved_trace_examples(rows, tmp_path)
    assert inputs == [trace_dir / "generations.jsonl"]
    assert examples[0]["reasoning_text"] == "Paris is one. Lima is second."
    assert examples[0]["old_candidates"] == 0


def test_band_verdict_prefers_trajectory_nuisance_over_running_index() -> None:
    audit = {
        "trajectory_band_summary": {
            "mean_within_trajectory_band_purity": 0.95
        },
        "categorical_associations": [
            {"column": "marker_kind", "nmi": 0.70},
            {"column": "seed", "nmi": 0.55},
            {"column": "occurrence", "nmi": 0.10},
        ],
    }
    assert "nuisance direction" in _band_verdict(audit)
