from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    ROOT
    / "reports/v4_non-thinking_causal/realistic_niah_v4_4_non_thinking_integrated_mechanism_report.html"
)


def _report() -> str:
    # Read at test execution time so a preceding deterministic rebuild in the
    # same test session is always observed.
    return REPORT_PATH.read_text(encoding="utf-8")


def _section(section_id: str) -> str:
    report = _report()
    match = re.search(
        rf'<section id="{re.escape(section_id)}".*?</section>', report, re.S
    )
    assert match is not None, section_id
    return match.group(0)


def test_publication_report_follows_mechanism_representation_causal_order() -> None:
    report = _report()
    ordered = (
        "mechanism-overview",
        "scope",
        "methods",
        "prompt",
        "representation-extension",
        "formation-tests",
        "answer",
        "attention",
        "causal",
        "natural-ov",
        "synthesis",
        "limits",
        "appendix-first-locator",
    )
    positions = [report.index(f'id="{section_id}"') for section_id in ordered]
    assert positions == sorted(positions)
    assert 'id="question-audit"' not in report


def test_exploratory_views_are_removed_from_the_paper_main_text() -> None:
    report = _report()
    assert 'id="prompt-canvas"' not in report
    assert 'id="answer-canvas"' not in report
    assert "Full-span all-head atlas" not in report
    assert "Prompt 十分类器逐层比较" not in report
    assert "Answer classification：全样本" not in report
    assert "Complete-first-span first-locator" in _section(
        "appendix-first-locator"
    )


def test_subspace_interventions_are_in_the_causal_chapter() -> None:
    formation = _section("formation-tests")
    causal = _section("causal")
    assert "Subspace 因果实验" not in formation
    assert causal.index("8.3 Subspace 因果实验") < causal.index(
        "8.4 Full-span Top-K ablation"
    )
    assert causal.index("8.4 Full-span Top-K ablation") < causal.index(
        "8.5 Answer-query patching"
    )
    for title in (
        "8.3C 结论",
        "8.3D 结论",
        "跨层结论",
    ):
        assert title in causal


def test_new_concepts_have_operational_definitions_and_claim_boundaries() -> None:
    report = _report()
    for formula in (
        "stable rank 定义为",
        "ε<sub>c,s</sub>=H[c,s]−μ<sub>c</sub>",
        "S<sub>h</sub>=M<sub>h</sub>C<sub>h</sub>",
        "A<sub>ℓ</sub>=R<sub>ℓ</sub>S<sub>ℓ</sub>",
        "T<sub>ℓ</sub>=U<sub>ℓ</sub>A<sub>ℓ</sub>U<sub>ℓ+1</sub><sup>T</sup>",
        "d<sub>i</sub><sup>abs</sup>",
        "T=(y<sub>P</sub>−y<sub>0</sub>)/(D−R)",
    ):
        assert formula in report
    assert "不把一个 token 或一个 PCA 轴写成离散整数寄存器" in report
    assert "任一单项 positive 都不推出唯一机制" in report


def test_every_figure_has_one_explanatory_caption() -> None:
    report = _report()
    figures = len(re.findall(r"<figure\b", report))
    captions = re.findall(r"<figcaption\b[^>]*>(.*?)</figcaption>", report, re.S)
    assert figures == len(captions) == 28
    for caption in captions:
        plain = re.sub(r"<[^>]+>", " ", caption)
        assert any(
            marker in plain
            for marker in (
                "横轴",
                "纵轴",
                "三个轴",
                "坐标轴",
                "没有数值坐标轴",
                "定义与图",
                "与左图完全相同",
                "与图 4C 相同",
                "坐标、冻结规则",
            )
        ), plain


def test_experimental_subsections_end_with_explicit_conclusions() -> None:
    report = _report()
    headings = re.finditer(r"<h3[^>]*>(.*?)</h3>", report, re.S)
    matches = list(headings)
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        assert 'class="conclusion"' in report[start:end], re.sub(
            r"<[^>]+>", "", match.group(1)
        )
