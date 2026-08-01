from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realistic_niah_v4_representation_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("v4_report_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_uses_the_registered_aurora_palette() -> None:
    report = _load_report_module()
    assert report.AURORA == {
        "midnight_indigo": "#23165C",
        "polar_violet": "#6750E8",
        "ice_cyan": "#00C2FF",
        "aurora_yellow": "#F6E36A",
        "aurora_teal": "#00D4B4",
        "aurora_green": "#39E58C",
        "polar_magenta": "#C04DFF",
        "sunset_pink": "#FF5FA2",
        "night_black": "#161923",
        "snow_white": "#F8FBFF",
        "frost_gray": "#8190A5",
        "warm_brown": "#765347",
    }
    old_viridis = {
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    }
    assert not any(color in report.REPORT_TEMPLATE.lower() for color in old_viridis)


def test_report_template_covers_the_full_mechanistic_argument() -> None:
    report = _load_report_module()
    template = report.REPORT_TEMPLATE
    section_ids = (
        "overview",
        "design",
        "definitions",
        "behavior",
        "metrics",
        "counter",
        "attention-heads",
        "span-end-attention",
        "causal",
        "synthesis",
        "limits",
        "reproducibility",
    )
    for section_id in section_ids:
        assert f'<section id="{section_id}">' in template
    assert template.count('class="section-conclusion"') >= len(section_ids)
    assert "broad evidence aggregation" in template
    assert "late executable query state" in template
    assert "@@BEHAVIOR_ACCURACY_SVG@@" in template
    assert "@@REPRESENTATION_R2_SVG@@" in template
    assert "@@ATTENTION_BREADTH_SVG@@" in template
    assert "@@CAUSAL_ABLATION_SVG@@" in template
    assert "@@ANSWER_QUERY_ADOPTION_SVG@@" in template
    assert "@@CAUSAL_STEERING_SVG@@" in template


def test_core_figures_define_axes_and_accessibility_text() -> None:
    report = _load_report_module()
    behavior_rows = [
        {
            "model": model,
            "variant": variant,
            "count": count,
            "accuracy": count / 10,
        }
        for model in report.MODELS
        for variant in report.VARIANTS
        for count in range(1, 11)
    ]
    behavior_svg = report._behavior_accuracy_svg(behavior_rows)
    assert "true needle count N" in behavior_svg
    assert "greedy exact-match accuracy" in behavior_svg
    assert "<title" in behavior_svg and "<desc" in behavior_svg

    metric_rows = [
        {
            "model_label": model,
            "design_variant": variant,
            "pooling": pooling,
            "confirmation_r2": 0.5,
        }
        for model in report.MODELS
        for variant in report.VARIANTS
        for pooling in report.POOLINGS
    ]
    representation_svg = report._representation_r2_svg(metric_rows)
    assert "confirmation R²" in representation_svg
    assert "R²=0" in representation_svg
    assert "<title" in representation_svg and "<desc" in representation_svg
