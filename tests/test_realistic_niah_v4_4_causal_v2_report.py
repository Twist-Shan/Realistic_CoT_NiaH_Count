from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realistic_niah_v4_4_causal_v2_report.py"
REPORT = ROOT / "reports" / "realistic_niah_v4_4_causal_v2_report.html"
DATA = ROOT / "reports" / "v4_non-thinking_causal" / "v4_4_causal_v2"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_report_is_standalone_structured_and_explicit_about_limits() -> None:
    text = REPORT.read_text(encoding="utf-8")

    assert SCRIPT.is_file()
    assert "<!doctype html>" in text.lower()
    assert 'lang="zh-CN"' in text
    assert text.count("<figure") == 7
    assert text.count("<figcaption") == 7
    assert text.count("本节结论") >= 18
    assert "matched-control-adjusted strict normalized transport" in text
    assert "min p=0.0625" in text
    assert "Holm-significant 仍是 0" in text
    assert "空白 layer 表示未通过 screen" in text
    assert "不要求唯一计数回路" in text
    assert "非单调性本身不是否定 head usefulness 的理由" in text
    assert "broad-aggregation top-5" in text
    assert "至少 10 个全新 seed clusters" in text
    assert "Patching 已提供充分的功能干预证据" in text
    assert "@media(max-width:860px)" in text
    assert "@media print" in text
    assert "@@" not in text
    assert "TODO" not in text
    element_ids = re.findall(r'\bid="([^"]+)"', text)
    assert len(element_ids) == len(set(element_ids))
    assert "{figure_id}" not in text


def test_headline_counts_and_audits_match_frozen_results() -> None:
    summary = json.loads((DATA / "report_summary.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == "realistic_niah_v4_4_causal_v2_report_v2"
    assert summary["implementation_commit"] == "dd409f2dff82ccd6400dfc3d7704025cb6939940"
    assert summary["alignment_policy"] == "monotonic_endpoint_preserving_nearest_neighbor_v1"
    assert {
        key: summary["alignment"]["Qwen3-8B"][key]
        for key in ("exact", "remapped", "unsupported")
    } == {"exact": 540, "remapped": 0, "unsupported": 0}
    assert {
        key: summary["alignment"]["Gemma4-E4B"][key]
        for key in ("exact", "remapped", "unsupported")
    } == {"exact": 178, "remapped": 362, "unsupported": 0}
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        assert summary["audits"][model]["status"] == "PASS"
        assert summary["audits"][model]["checks"] == 302
        assert summary["audits"][model]["errors"] == 0
        assert summary["exports"][model]["archive_sha256_verified"] is True
        assert summary["exports"][model]["source_copy_manifests_identical"] is True

    expected = {
        "Qwen3-8B::prompt_patching": 126,
        "Gemma4-E4B::prompt_patching": 102,
        "Qwen3-8B::answer_patching": 149,
        "Gemma4-E4B::answer_patching": 177,
        "Qwen3-8B::steering": 45,
        "Gemma4-E4B::steering": 54,
    }
    for key, conditions in expected.items():
        row = summary["primary_confirmation_family_summary"][key]
        assert row["conditions"] == conditions
        assert row["ci95_excludes_zero"] == conditions
        assert row["holm_p_le_0_05"] == 0

    assert summary["claim_sufficiency"]["answer_query_hidden_state_usable_information"]["verdict"].startswith(
        "对受限主张充分"
    )
    assert summary["claim_sufficiency"]["ranked_head_bank_functional_contribution"]["verdict"].startswith(
        "现有结果为支持性"
    )


def test_family_summary_is_recomputed_from_checked_condition_table() -> None:
    rows = _read_csv("primary_confirmation_conditions.csv")
    summaries = {
        (row["model_label"], row["family"]): row
        for row in _read_csv("primary_confirmation_family_summary.csv")
    }
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_label"], row["family"])].append(row)

    for key, selected in groups.items():
        output = summaries[key]
        effects = [float(row["mean_control_adjusted_transport"]) for row in selected]
        assert int(output["conditions"]) == len(selected)
        assert math.isclose(float(output["mean_effect"]), statistics.fmean(effects))
        assert math.isclose(float(output["median_effect"]), statistics.median(effects))
        assert math.isclose(float(output["min_effect"]), min(effects))
        assert math.isclose(float(output["max_effect"]), max(effects))
        assert int(output["ci95_excludes_zero"]) == sum(float(row["ci95_low"]) > 0 for row in selected)
        assert int(output["holm_p_le_0_05"]) == sum(float(row["holm_p"]) <= 0.05 for row in selected)


def test_source_ledger_and_machine_tables_are_complete() -> None:
    expected_files = {
        "ablation_top_k_sweep.csv",
        "ablation_support_summary.csv",
        "audit_category_summary.csv",
        "audit_summary.csv",
        "baseline_by_count.csv",
        "baseline_by_split.csv",
        "export_verification.csv",
        "evidence_sufficiency.csv",
        "primary_confirmation_conditions.csv",
        "primary_confirmation_family_summary.csv",
        "primary_confirmation_group_summary.csv",
        "prompt_alignment_summary.csv",
        "report_summary.json",
        "selection_summary.csv",
        "source_ledger.csv",
        "stage_inventory.csv",
    }
    assert expected_files <= {path.name for path in DATA.iterdir()}

    ledger = _read_csv("source_ledger.csv")
    assert len(ledger) >= 40
    assert len({row["source_label"] for row in ledger}) == len(ledger)
    for row in ledger:
        assert int(row["size_bytes"]) > 0
        assert len(row["sha256"]) == 64
        int(row["sha256"], 16)


def test_ablation_support_summary_is_recomputed_and_identifies_frozen_candidate() -> None:
    sweep = _read_csv("ablation_top_k_sweep.csv")
    output = {
        (row["model_label"], row["head_bank"]): row
        for row in _read_csv("ablation_support_summary.csv")
    }
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in sweep:
        groups[(row["model_label"], row["head_bank"])].append(row)

    for key, selected in groups.items():
        either = sum(
            float(row["accuracy_effect"]) < 0 or float(row["absolute_error_effect"]) > 0
            for row in selected
        )
        both = sum(
            float(row["accuracy_effect"]) < 0 and float(row["absolute_error_effect"]) > 0
            for row in selected
        )
        assert int(output[key]["top_n_tested"]) == len(selected) == 32
        assert int(output[key]["either_metric_harmful_count"]) == either
        assert int(output[key]["both_metrics_harmful_count"]) == both
        assert output[key]["held_out_confirmation"] == "False"
        assert output[key]["supports_monotone_dose_response"] == "False"

    assert output[("Qwen3-8B", "broad_aggregation")]["both_metrics_harmful_count"] == "5"
    assert output[("Gemma4-E4B", "broad_aggregation")]["both_metrics_harmful_count"] == "13"
    assert output[("Qwen3-8B", "broad_aggregation")]["cross_model_shared_both_metrics_top_n"] == "5"
    assert output[("Gemma4-E4B", "broad_aggregation")]["cross_model_shared_both_metrics_top_n"] == "5"


def test_stage_inventory_has_no_missing_or_skipped_rows() -> None:
    rows = _read_csv("stage_inventory.csv")
    assert len(rows) == 12
    for row in rows:
        assert row["status"] == "complete"
        assert int(row["successful_rows"]) == int(row["logical_rows"])
        assert int(row["skipped_rows"]) == 0
        assert int(row["capture_index_rows"]) == 90
