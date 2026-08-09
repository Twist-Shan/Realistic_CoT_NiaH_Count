from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realistic_niah_v4_4_report.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")
INTEGRATED_SOURCE = (
    ROOT / "scripts" / "build_realistic_niah_v4_4_integrated_report.py"
).read_text(encoding="utf-8")


def _literal_assignment(name: str):
    tree = ast.parse(SOURCE)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")


def test_v4_4_report_has_the_requested_evidence_order_and_controls() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    assert _literal_assignment("FOCUS_VARIANT") == "v4.4"
    assert template.index('<section id="prompt">') < template.index(
        '<section id="answer">'
    )
    assert template.index('<section id="answer">') < template.index(
        '<section id="attention">'
    )
    assert template.index('<section id="attention">') < template.index(
        '<section id="causal">'
    )
    assert "Head ablation" in template
    assert "Needle-end patching" in template
    assert "Answer-query patching" in template
    assert "Steering v1" in template
    assert "Steering v2" in template

    assert "discovery-ranked span-end top-4 / top-8" in SOURCE
    assert "layer-matched" in SOURCE
    assert "没有 matched-random residual control" in SOURCE
    assert "h_receiver(q,l) ← h_donor(q,l)" in SOURCE
    assert "等范数且正交" in SOURCE
    assert "invalid-as-failure" in SOURCE


def test_v4_4_report_exposes_all_layer_interactions_and_both_attention_poolings() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    assert 'id="{prefix}-layer"' in SOURCE
    assert '_projector_controls("prompt", answer=False)' in SOURCE
    assert '_projector_controls("answer", answer=True)' in SOURCE
    assert 'id="joint-layer"' in SOURCE
    assert 'id="{prefix}-canvas"' in SOURCE
    assert 'id="joint-canvas"' in SOURCE
    assert 'data-atlas="span_end"' in SOURCE
    assert 'data-atlas="span_sum"' in SOURCE
    assert "完整 needle span 的 literal sum" in template
    assert "correct-only basis" in template


def test_v4_4_report_replaces_endpoint_locator_table_with_complete_span_result() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    assert "@@FIRST_SPAN_SECTION@@" in template
    assert "@@PHENOTYPE_TABLE@@" not in template
    assert "Complete-first-span attention phenotype" in SOURCE
    assert "layer+M₁₀-nearest" in SOURCE
    assert "不支持“first-span locator 是独特必要 circuit”" in SOURCE
    assert "V4.4 endpoint phenotype counts and representatives" not in SOURCE
    assert '<h3>8.3 Complete-first-span attention phenotype</h3>' in INTEGRATED_SOURCE
    assert "ensure_complete_first_span_section" in INTEGRATED_SOURCE
    assert "base = ensure_complete_first_span_section(base)" in INTEGRATED_SOURCE
    assert "remove_legacy_endpoint_phenotype_table" in INTEGRATED_SOURCE
    assert "base = remove_legacy_endpoint_phenotype_table(base)" in INTEGRATED_SOURCE


def test_v4_4_causal_subsections_state_results_and_inference_limits() -> None:
    template = _literal_assignment("REPORT_TEMPLATE")

    for placeholder in (
        "@@ABLATION_RESULT@@",
        "@@ENDPOINT_PATCH_RESULT@@",
        "@@ANSWER_PATCH_RESULT@@",
        "@@STEERING_V1_RESULT@@",
        "@@STEERING_V2_RESULT@@",
    ):
        assert placeholder in template
    assert "bank-level、mixed-phenotype necessity screen" in SOURCE
    assert "不等同于变成 donor gold" in SOURCE
    assert "不否定 full-span、多 token" in SOURCE
    assert "multi-layer 没有稳定超过 single-layer" in SOURCE


def test_variant_filter_rejects_cross_panel_leakage() -> None:
    assert 'frame["design_variant"].astype(str) == FOCUS_VARIANT' in SOURCE
    assert 'raise RuntimeError(f"{model}/{stage}: no {FOCUS_VARIANT} rows")' in SOURCE
    assert "_nested_v44_rows" in SOURCE
    assert 'answer_patch["design_variant"].astype(str) == FOCUS_VARIANT' in SOURCE
