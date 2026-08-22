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


@pytest.mark.skipif(
    not EXTERNAL_EVIDENCE_SENTINEL.exists(),
    reason="external Native-thinking evidence bundle is not installed",
)
def test_restructured_report_is_audited_defined_and_nonthinking_ordered(tmp_path: Path) -> None:
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

    section_ids = (
        "definitions",
        "summary",
        "design",
        "task",
        "representation",
        "retrieval",
        "write",
        "answer",
        "walkthrough",
        "comparison",
        "appendix",
        "audit",
    )
    offsets = [text.index(f'<section id="{section_id}">') for section_id in section_ids]
    assert offsets == sorted(offsets)
    assert "Qwen3-8B · frozen Top-128" in text
    assert "Gemma4-E4B · frozen Top-6" in text
    assert "20 discovery / 10 confirmation" in text
    assert "descriptive null" in text
    assert "does not walk 1→10" in text
    assert "natural end-to-end sufficiency 仍未证明" in text
    assert "BalancedAccuracy" in text
    assert "PC1, PC2, PC3" in text
    assert 'id="native-running-canvas"' in text
    assert 'id="native-final-canvas"' in text
    assert 'id="native-geometry-model"' in text
    assert 'id="native-running-layer"' in text
    assert 'id="native-final-layer"' in text
    assert "class NativePointCloud3D" in text
    assert "const NATIVE_GEOMETRY=" in text
    assert "frozen default" in text
    assert '"default_layer":18,"layers":{"0":' in text
    assert '"default_layer":34,"layers":{"0":' in text
    assert "实验目的" in text
    assert "简单例子" in text
    assert "confirmed†" in text
    assert "local matched-control specificity" in text
    assert "Raw attention mass 与 bank-summed mass" in text
    assert "图 2c · 可切换的 Native targeted-retrieval attention maps" in text
    assert "图 2d · Qwen Top-128 layer×head atlas（全宽）" in text
    assert "图 2e · Gemma Top-6 layer×head atlas（L0–20 / L21–41 分栏放大）" in text
    assert 'class="attention-atlas-stack"' in text
    assert "frozen Top-6" in text
    assert "data-attention-selector" in text
    assert "Appendix E · 其他 grammar 的 attention-map 对应版本" in text
    assert "图 E8 · Gemma structural-invariant bullet · ordinal×head" in text
    assert text.count('<details class="appendix-block">') == 5
    assert '<details class="appendix-block" open>' not in text
    assert "Gemma4-E4B · frozen Top-8" not in text
    assert "完整串行链获得confirmation" not in text
    assert len(re.findall(r"<svg(?:\s|>)", text)) == text.count("</svg>")
    assert len(text.encode("utf-8")) < 4_500_000

    assert payload["status"] == "PASS"
    assert payload["scientific_contract"] == {
        "discovery_seed_count": 20,
        "confirmation_seed_count": 10,
        "outcome_blind": True,
        "selection_rank_used": False,
        "qwen_targeted_bank": 128,
        "gemma_targeted_bank": 6,
    }
    assert payload["claim_scope"]["recurrent_pathway_supported"] is True
    assert payload["claim_scope"]["exclusive_circuit_claimed"] is False
    assert payload["claim_scope"]["natural_end_to_end_single_state_sufficiency"] is False
    assert payload["claim_scope"]["single_seed_walkthrough_inferential"] is False
    assert payload["claim_scope"]["gemma_commit_to_query_direct_effect_confirmed"] is True
    assert payload["claim_scope"]["gemma_commit_to_query_local_specificity_qualified"] is True
    assert payload["claim_scope"]["gemma_narrow_pre_o_query_mediation_confirmed"] is False
    assert payload["claim_scope"]["qwen_free_running_terminal_restoration_confirmed"] is False
    assert payload["claim_scope"]["all_layer_pca3_is_descriptive"] is True
    assert payload["schema_version"] == "realistic_niah_v5_native_thinking_restructured_v3"
    assert "geometry_3d" in payload["derived_display_data_sha256"]


def test_shipped_restructured_report_is_self_contained() -> None:
    report = REPO_ROOT / "reports" / "NiaH_Native-Thinking_report.html"
    text = report.read_text(encoding="utf-8")
    assert '<img ' not in text
    assert len(re.findall(r"<svg(?:\s|>)", text)) == text.count("</svg>")
    assert 'id="native-running-canvas"' in text
    assert 'id="native-final-canvas"' in text
    assert "图 E8 · Gemma structural-invariant bullet · ordinal×head" in text
    assert text.count('<details class="appendix-block">') == 5
    assert '<details class="appendix-block" open>' not in text
    assert len(text.encode("utf-8")) < 4_500_000
