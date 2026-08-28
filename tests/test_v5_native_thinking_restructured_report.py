from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_EVIDENCE_SENTINEL = (
    REPO_ROOT
    / "reports"
    / "v5_native_final_localizers"
    / "analysis"
    / "qwen_final_merged_dose_grid.json"
)
SECTION_IDS = (
    "summary",
    "baseline",
    "representation",
    "formation",
    "retrieval",
    "write",
    "answer",
    "integrated-chain",
    "ledger",
    "extension-audit",
    "limitations",
    "appendix",
)


def _assert_current_report_contract(text: str, payload: dict) -> None:
    offsets = [text.index(f'<section id="{section_id}">') for section_id in SECTION_IDS]
    assert offsets == sorted(offsets)
    assert "本文主张（仅限 Qwen3-8B 的自然 no-index trace）" in text
    assert "Gemma 尚无对应的自然 no-index 因果结果" in text
    assert "simulatively confirmed†" in text
    assert "可诱发的机制能力，不是自然使用" in text
    assert "20 discovery / 10 confirmation" in text
    assert "J.1 显式 index positive control" in text
    assert "Appendix K · Gemma prompt-conditioned no-visible-index" in text
    assert 'id="native-running-canvas"' in text
    assert 'id="native-final-canvas"' in text
    assert '<img ' not in text
    assert len(re.findall(r"<svg(?:\s|>)", text)) == text.count("</svg>")
    assert len(text.encode("utf-8")) < 4_500_000

    assert payload["status"] == "PASS"
    assert payload["schema_version"] == (
        "realistic_niah_v5_native_thinking_restructured_v11"
    )
    contract = payload["scientific_contract"]
    assert contract["discovery_seed_count"] == 20
    assert contract["confirmation_seed_count"] == 10
    assert contract["outcome_blind"] is True
    assert contract["indexed_positive_control_active_confirmation_layer"] == {
        "Qwen3-8B": 16,
        "Gemma4-E4B": 16,
    }

    scope = payload["claim_scope"]
    assert scope["distributed_content_bound_event_progress_state_supported"] is True
    assert scope["memoryless_arithmetic_plus_one_recurrence_supported"] is False
    assert scope["qwen_no_index_scope_result_extrapolated_to_gemma"] is False
    assert scope["gemma_natural_no_index_causal_result_available"] is False
    assert scope["gemma_next_item_routing_status"] == (
        "simulatively confirmed under auxiliary settings"
    )
    assert scope["gemma_prompt_conditioned_supports_natural_no_index_claim"] is False


@pytest.mark.skipif(
    not EXTERNAL_EVIDENCE_SENTINEL.exists(),
    reason="external Native-thinking evidence bundle is not installed",
)
def test_restructured_report_is_audited_and_nonthinking_ordered(
    tmp_path: Path,
) -> None:
    output = tmp_path / "native.html"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_v5_native_thinking_report_restructured.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    text = output.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    _assert_current_report_contract(text, payload)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_v5_native_thinking_report.py"),
            str(output),
            str(manifest),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_shipped_restructured_report_is_self_contained() -> None:
    report = REPO_ROOT / "reports" / "NiaH_Native-Thinking_report.html"
    manifest = (
        REPO_ROOT
        / "reports"
        / "v5_native_final_localizers"
        / "report_manifest_restructured.json"
    )
    text = report.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    _assert_current_report_contract(text, payload)
