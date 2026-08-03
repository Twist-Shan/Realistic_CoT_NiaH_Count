from __future__ import annotations

import csv
import json
import math
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
    assert "最小可能 p 值是 2/2⁵ = 0.0625" in text
    assert "Holm-significant 仍是 0" in text
    assert "空白 layer 表示未通过 screen" in text
    assert "不证明唯一机制" in text
    assert "@media(max-width:860px)" in text
    assert "@media print" in text
    assert "@@" not in text
    assert "TODO" not in text


def test_headline_counts_and_audits_match_frozen_results() -> None:
    summary = json.loads((DATA / "report_summary.json").read_text(encoding="utf-8"))

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
        "audit_category_summary.csv",
        "audit_summary.csv",
        "baseline_by_count.csv",
        "baseline_by_split.csv",
        "export_verification.csv",
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


def test_stage_inventory_has_no_missing_or_skipped_rows() -> None:
    rows = _read_csv("stage_inventory.csv")
    assert len(rows) == 12
    for row in rows:
        assert row["status"] == "complete"
        assert int(row["successful_rows"]) == int(row["logical_rows"])
        assert int(row["skipped_rows"]) == 0
        assert int(row["capture_index_rows"]) == 90
