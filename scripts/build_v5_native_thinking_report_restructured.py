#!/usr/bin/env python3
"""Build the concise Native-thinking mechanism report.

The report follows the section grammar of the Non-thinking report while keeping
Native-specific causal edges and unresolved boundaries explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_v5_native_thinking_report_final import (
    load_representation,
    load_token_ablation_evidence,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
SHORT = {"Qwen3-8B": "Qwen", "Gemma4-E4B": "Gemma"}
COLORS = {"Qwen3-8B": "#0f766e", "Gemma4-E4B": "#7c3aed"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def effect(rows: Iterable[Mapping[str, Any]], estimand: str) -> Mapping[str, Any]:
    matches = [row for row in rows if row.get("estimand") == estimand]
    require(len(matches) == 1, f"Expected one estimand {estimand}, got {len(matches)}")
    return matches[0]


def ci(row: Mapping[str, Any], digits: int = 3) -> str:
    return (
        f"{float(row['mean_effect']):+.{digits}f} "
        f"[{float(row['ci_low']):+.{digits}f}, {float(row['ci_high']):+.{digits}f}]"
    )


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def extract_css(path: Path) -> str:
    match = re.search(r"<style>(.*?)</style>", path.read_text(encoding="utf-8"), re.S)
    require(match is not None, f"No CSS block in {path}")
    return match.group(1)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def selected_row(analysis: Mapping[str, Any], k: int) -> Mapping[str, Any]:
    rows = [
        row
        for row in analysis["overall"]
        if row["scope"] == "all_registered_grammars" and int(row["bank_size"]) == k
    ]
    require(len(rows) == 1, f"Missing all-grammar K={k} row")
    return rows[0]


def line_chart(
    panels: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> str:
    width, panel_w, height = 980, 410, 310
    parts = [
        f'<svg class="line-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Selected and random targeted retrieval ablation dose response">'
    ]
    for panel_index, (label, rows) in enumerate(panels):
        x0 = 72 + panel_index * 490
        y0, plot_w, plot_h = 52, panel_w, 190
        ks = [int(row["bank_size"]) for row in rows]
        k_min, k_max = min(ks), max(ks)

        def sx(k: int) -> float:
            return x0 + (k - k_min) / max(k_max - k_min, 1) * plot_w

        def sy(value: float) -> float:
            return y0 + (1.0 - value) * plot_h

        parts.append(f'<text x="{x0}" y="25" class="heat-title">{esc(label)}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = sy(tick)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>')
        for k in ks:
            x = sx(k)
            parts.append(f'<text x="{x:.1f}" y="{y0+plot_h+20}" text-anchor="middle" class="tick">{k}</text>')
        selected = " ".join(
            f"{sx(int(row['bank_size'])):.1f},{sy(float(row['selected_failure_rate'])):.1f}"
            for row in rows
        )
        random = " ".join(
            f"{sx(int(row['bank_size'])):.1f},{sy(float(row['random_failure_rate'])):.1f}"
            for row in rows
        )
        parts.append(f'<polyline points="{selected}" class="series-line" stroke="#0f766e"/>')
        parts.append(f'<polyline points="{random}" class="series-line" stroke="#98a2b3"/>')
        for row in rows:
            x = sx(int(row["bank_size"]))
            for key, color in (("selected_failure_rate", "#0f766e"), ("random_failure_rate", "#98a2b3")):
                parts.append(f'<circle cx="{x:.1f}" cy="{sy(float(row[key])):.1f}" r="4" fill="{color}" class="series-dot"/>')
        parts.append(f'<text x="{x0+plot_w/2}" y="{height-24}" text-anchor="middle" class="axis-label">Frozen bank size K</text>')
    parts.extend(
        [
            '<line x1="375" y1="280" x2="401" y2="280" stroke="#0f766e" stroke-width="3"/>',
            '<text x="407" y="284" class="legend-label">selected</text>',
            '<line x1="485" y1="280" x2="511" y2="280" stroke="#98a2b3" stroke-width="3"/>',
            '<text x="517" y="284" class="legend-label">layer-matched random</text>',
            '<text transform="translate(18 150) rotate(-90)" text-anchor="middle" class="axis-label">Retrieval failure rate</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def grouped_bars_svg(
    title: str,
    groups: Sequence[tuple[str, Mapping[str, float]]],
    *,
    maximum: float | None = None,
    suffix: str = "",
) -> str:
    values = [value for _, group in groups for value in group.values()]
    max_value = maximum if maximum is not None else max(max(values), 1e-9) * 1.12
    width, height = 900, 92 + len(groups) * 80
    x0, plot_w = 250, 560
    parts = [
        f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<text x="{x0}" y="25" class="heat-title">{esc(title)}</text>',
    ]
    for group_index, (label, model_values) in enumerate(groups):
        y = 62 + group_index * 80
        parts.append(f'<text x="{x0-14}" y="{y+18}" text-anchor="end" class="bar-label">{esc(label)}</text>')
        for model_index, model in enumerate(MODELS):
            value = float(model_values[model])
            yy = y + model_index * 25
            bar_w = max(0.0, min(value / max_value, 1.0)) * plot_w
            parts.append(f'<rect x="{x0}" y="{yy}" width="{bar_w:.1f}" height="15" rx="2" fill="{COLORS[model]}" opacity=".88"/>')
            parts.append(f'<text x="{x0+bar_w+7:.1f}" y="{yy+12}" class="bar-value">{value:.3f}{esc(suffix)}</text>')
        parts.append(f'<text x="{x0+plot_w+18}" y="{y+12}" class="mini-model" fill="{COLORS[MODELS[0]]}">Q</text>')
        parts.append(f'<text x="{x0+plot_w+18}" y="{y+37}" class="mini-model" fill="{COLORS[MODELS[1]]}">G</text>')
    parts.append("</svg>")
    return "".join(parts)


def walkthrough_svg(walkthrough: Mapping[str, Mapping[str, Any]]) -> str:
    width, height = 980, 380
    parts = [
        f'<svg class="line-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Single-seed expected-count paths under full context scrubbing and state restoration">'
    ]
    condition_style = {
        "full_item_restore": ("full item", "#0f766e"),
        "counter_carrier_restore": ("marker / tail carrier", "#7c3aed"),
        "counter_carrier_matched_control": ("matched ordinary state", "#98a2b3"),
    }
    for panel_index, model in enumerate(MODELS):
        data = walkthrough[model]
        x0 = 70 + panel_index * 490
        y0, plot_w, plot_h = 54, 400, 230

        def sx(k: int) -> float:
            return x0 + (k - 1) / 9 * plot_w

        def sy(value: float) -> float:
            return y0 + (10 - value) / 9 * plot_h

        parts.append(f'<text x="{x0}" y="25" class="heat-title">{esc(SHORT[model])} · seed {data["seed"]}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>')
        ideal = " ".join(f"{sx(k):.1f},{sy(k):.1f}" for k in range(1, 11))
        parts.append(f'<polyline points="{ideal}" fill="none" stroke="#d97706" stroke-width="1.5" stroke-dasharray="5 5"/>')
        for tick in (1, 4, 7, 10):
            y = sy(tick)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}</text>')
        for k in range(1, 11):
            parts.append(f'<text x="{sx(k):.1f}" y="{y0+plot_h+20}" text-anchor="middle" class="tick">{k}</text>')
        for condition, (_, color) in condition_style.items():
            path = data["conditions"][condition]["expected_count_path"]
            points = " ".join(f"{sx(k):.1f},{sy(float(value)):.1f}" for k, value in enumerate(path, 1))
            parts.append(f'<polyline points="{points}" class="series-line" stroke="{color}"/>')
            for k, value in enumerate(path, 1):
                parts.append(f'<circle cx="{sx(k):.1f}" cy="{sy(float(value)):.1f}" r="3" fill="{color}"/>')
    legend_positions = (180, 370, 570)
    for x, (condition, (label, color)) in zip(legend_positions, condition_style.items()):
        parts.append(f'<line x1="{x}" y1="340" x2="{x+24}" y2="340" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+30}" y="344" class="legend-label">{esc(label)}</text>')
    parts.append('<line x1="800" y1="340" x2="824" y2="340" stroke="#d97706" stroke-dasharray="5 5"/>')
    parts.append('<text x="830" y="344" class="legend-label">ideal y=k</text>')
    parts.append('<text transform="translate(18 170) rotate(-90)" text-anchor="middle" class="axis-label">Expected count</text>')
    parts.append('<text x="490" y="374" text-anchor="middle" class="axis-label">Restored item occurrence k</text>')
    parts.append("</svg>")
    return "".join(parts)


def chain_svg() -> str:
    stages = (
        ("Targeted retrieval", "bank ablation", "strong", "strong"),
        ("Grammar carrier", "marker / tail state", "strong", "strong"),
        ("Commit state", "progress submission", "strong", "strong"),
        ("Next query", "targeted-head routing", "strong", "directional"),
        ("Answer", "terminal local bridge", "conditional", "conditional"),
    )
    width, height, box_w, gap = 1080, 260, 170, 28
    x0 = 80
    parts = [f'<svg class="chain-figure" viewBox="0 0 {width} {height}" role="img" aria-label="Native-thinking causal chain by model">']
    for idx, (title, subtitle, _, _) in enumerate(stages):
        x = x0 + idx * (box_w + gap)
        parts.append(f'<rect x="{x}" y="35" width="{box_w}" height="68" rx="4" fill="#fff" stroke="#cfd6e2"/>')
        parts.append(f'<text x="{x+box_w/2}" y="61" text-anchor="middle" class="chain-title">{esc(title)}</text>')
        parts.append(f'<text x="{x+box_w/2}" y="82" text-anchor="middle" class="chain-sub">{esc(subtitle)}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<path d="M{x+box_w} 69 H{x+box_w+gap-8}" stroke="#98a2b3" stroke-width="2"/>')
            parts.append(f'<path d="M{x+box_w+gap-14} 63 L{x+box_w+gap-8} 69 L{x+box_w+gap-14} 75" fill="none" stroke="#98a2b3" stroke-width="2"/>')
    for row_index, model in enumerate(MODELS):
        y = 142 + row_index * 48
        parts.append(f'<text x="{x0-8}" y="{y+16}" text-anchor="end" class="chain-model">{esc(SHORT[model])}</text>')
        for idx, (_, _, q_status, g_status) in enumerate(stages):
            status = q_status if model == MODELS[0] else g_status
            color = {"strong": "#0f766e", "directional": "#d97706", "conditional": "#7c3aed"}[status]
            label = {"strong": "confirmed", "directional": "directional", "conditional": "controlled only"}[status]
            x = x0 + idx * (box_w + gap)
            parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="29" rx="3" fill="{color}" opacity=".12" stroke="{color}"/>')
            parts.append(f'<text x="{x+box_w/2}" y="{y+19}" text-anchor="middle" class="chain-status" fill="{color}">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    q_analysis = read_json(args.qwen_targeted_analysis)
    g_analysis = read_json(args.gemma_targeted_analysis)
    q_primary = selected_row(q_analysis, 128)
    g_primary = selected_row(g_analysis, 6)
    q_dose = [row for row in q_analysis["overall"] if row["scope"] == "all_registered_grammars"]
    g_dose = [row for row in g_analysis["overall"] if row["scope"] == "all_registered_grammars"]
    representation = load_representation(args.representation_root, args.dual_endpoint_root)
    token_evidence = {
        model: load_token_ablation_evidence(args.token_ablation_root / model, model)
        for model in MODELS
    }
    write = {
        model: read_json(args.snapshot_root / "targeted_counter_write_20260822" / model / "complete.json")
        for model in MODELS
    }
    commit_query = {
        model: read_json(args.snapshot_root / "commit_state_query_20260822" / model / "commit_to_query_complete.json")
        for model in MODELS
    }
    grammar_span = {
        "Qwen3-8B": read_json(args.snapshot_root / "qwen_grammar_span_decomposition_complete.json"),
        "Gemma4-E4B": read_json(args.snapshot_root / "gemma_grammar_span_decomposition_complete.json"),
    }
    q_free = read_json(args.snapshot_root / "targeted_counter_20260822" / "Qwen3-8B" / "targeted_counter_complete.json")
    walkthrough = {
        model: read_json(args.snapshot_root / "single_seed_walkthrough_20260822_v2" / model / "analysis" / "walkthrough_complete.json")
        for model in MODELS
    }

    require(q_analysis.get("status") == "PASS" and g_analysis.get("status") == "PASS", "Targeted analyses must PASS")
    require(float(g_primary["selected_minus_random_failure_rate"]) > float(selected_row(g_analysis, 8)["selected_minus_random_failure_rate"]), "Gemma K6 must remain the frozen primary over K8")
    for model in MODELS:
        for evidence_name, evidence in (("write", write[model]), ("commit-query", commit_query[model])):
            require(evidence.get("status") == "PASS", f"{model} {evidence_name} not PASS")
            require(evidence.get("discovery_seed_count") == 20, f"{model} {evidence_name} discovery seed drift")
            require(evidence.get("confirmation_seed_count") == 10, f"{model} {evidence_name} confirmation seed drift")
            require(evidence.get("outcome_blind") is True, f"{model} {evidence_name} not outcome blind")
            require(evidence.get("selection_rank_used") is False, f"{model} {evidence_name} uses selection_rank")
        require(grammar_span[model].get("status") == "PASS", f"{model} grammar span not PASS")
        require(grammar_span[model].get("discovery_seed_count") == 20, f"{model} grammar span discovery drift")
        require(grammar_span[model].get("confirmation_seed_count") == 10, f"{model} grammar span confirmation drift")
        require(walkthrough[model].get("status") == "PASS", f"{model} walkthrough not PASS")
        require(walkthrough[model].get("case_study_not_inferential") is True, f"{model} walkthrough scope drift")
        require(walkthrough[model].get("case_selected_by_outcome") is False, f"{model} walkthrough selected by outcome")
        require(walkthrough[model].get("answer_query_patched") is False, f"{model} walkthrough patched answer query")
    require(q_free.get("complete_strong_gate_pass") is False, "Qwen free-running negative boundary changed")

    write_effects: dict[str, dict[str, Mapping[str, Any]]] = {}
    query_effects: dict[str, dict[str, Mapping[str, Any]]] = {}
    terminal_effects: dict[str, dict[str, Mapping[str, Any]]] = {}
    for model in MODELS:
        write_rows = write[model]["confirmation"]["all_estimands"]
        write_effects[model] = {
            key: effect(write_rows, key)
            for key in (
                "selected_carrier_deformation",
                "clean_carrier_restoration",
                "restoration_position_specificity",
            )
        }
        query_rows = commit_query[model]["confirmation"]["estimands"]
        query_effects[model] = {
            "self": effect(query_rows, "full_commit_targeted_attention_vs_self_distance_1"),
            "orthogonal": effect(query_rows, "full_commit_targeted_attention_vs_orthogonal_distance_1"),
        }
        terminal_rows = grammar_span[model]["confirmation"]["primary_estimands"]
        terminal_effects[model] = {
            row["contrast"]: row
            for row in terminal_rows
            if row["geometry"] == "marker_core"
        }
        require(set(terminal_effects[model]) == {"restoration", "matched_random_specificity"}, f"{model} missing marker terminal effects")

    rep_groups = [
        (
            "running commit · confirmation balanced accuracy",
            {model: float(representation[model]["running"]["confirmation_logistic_balanced_accuracy"]) for model in MODELS},
        ),
        (
            "answer query · confirmation balanced accuracy",
            {model: float(representation[model]["final"]["confirmation_logistic_balanced_accuracy"]) for model in MODELS},
        ),
    ]
    write_groups = [
        ("targeted bank → carrier deformation", {model: float(write_effects[model]["selected_carrier_deformation"]["mean_effect"]) for model in MODELS}),
        ("carrier restore → commit recovery", {model: float(write_effects[model]["clean_carrier_restoration"]["mean_effect"]) for model in MODELS}),
        ("same-position specificity", {model: float(write_effects[model]["restoration_position_specificity"]["mean_effect"]) for model in MODELS}),
    ]
    query_groups = [
        ("full commit vs self patch", {model: float(query_effects[model]["self"]["mean_effect"]) for model in MODELS}),
        ("full commit vs orthogonal", {model: float(query_effects[model]["orthogonal"]["mean_effect"]) for model in MODELS}),
        ("terminal marker restoration", {model: float(terminal_effects[model]["restoration"]["mean_effect"]) for model in MODELS}),
    ]
    source_groups = [
        (
            "clean exact count",
            {model: float(token_evidence[model]["answer"]["trace_items"]["clean"]["exact_rate"]) for model in MODELS},
        ),
        (
            "trace blank exact count",
            {model: float(token_evidence[model]["answer"]["trace_items"]["trace_all_blank"]["exact_rate"]) for model in MODELS},
        ),
    ]

    custom_css = """
.report-note{max-width:920px;color:#475467}.status-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}.status-card{padding:18px;border:1px solid var(--line);background:#fbfcfe}.status-card h3{margin:0 0 8px}.status-card p{margin:6px 0;font-size:14px}.status-good{color:#075e58;font-weight:750}.status-open{color:#9a4b00;font-weight:750}.chain-figure{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fbfcfe}.chain-title{fill:#172033;font-size:13px;font-weight:750}.chain-sub{fill:#667085;font-size:11px}.chain-model{fill:#344054;font-size:12px;font-weight:750}.chain-status{font-size:11px;font-weight:750}.mini-model{font-size:11px;font-weight:800}.metric-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}.metric{padding:15px;border-top:3px solid var(--teal);background:#f8fafc}.metric strong,.metric span{display:block}.metric strong{font-size:22px}.metric span{color:#667085;font-size:12px}.negative-result{padding:17px 19px;border-left:4px solid var(--amber);background:#fff8eb}.audit-list{font-size:12px;color:#667085;overflow-wrap:anywhere}.compact-table td,.compact-table th{padding:7px 8px}.walkthrough-callout{display:grid;grid-template-columns:1fr 1fr;gap:14px}.walkthrough-callout>div{padding:15px;border:1px solid var(--line);background:#fbfcfe}@media(max-width:760px){.status-grid,.walkthrough-callout,.metric-strip{grid-template-columns:1fr}.chain-figure{min-width:850px}.chain-scroll{overflow-x:auto}}
"""
    css = extract_css(args.reference_report) + custom_css
    generated = datetime.now(timezone.utc).isoformat()

    targeted_table = table(
        ("Model", "Frozen bank", "Selected failure", "Random failure", "Selected − random"),
        (
            (
                SHORT[model],
                "Top-128" if model == "Qwen3-8B" else "Top-6",
                pct(float(row["selected_failure_rate"])),
                pct(float(row["random_failure_rate"])),
                f"{100*float(row['selected_minus_random_failure_rate']):+.1f} pp",
            )
            for model, row in (("Qwen3-8B", q_primary), ("Gemma4-E4B", g_primary))
        ),
    )
    write_table = table(
        ("Edge / control", "Qwen confirmation", "Gemma confirmation", "Interpretation"),
        (
            (
                "targeted bank → grammar carrier",
                ci(write_effects["Qwen3-8B"]["selected_carrier_deformation"]),
                ci(write_effects["Gemma4-E4B"]["selected_carrier_deformation"]),
                "both positive",
            ),
            (
                "clean carrier → commit restoration",
                ci(write_effects["Qwen3-8B"]["clean_carrier_restoration"]),
                ci(write_effects["Gemma4-E4B"]["clean_carrier_restoration"]),
                "both positive",
            ),
            (
                "restoration − equal-token nearby control",
                ci(write_effects["Qwen3-8B"]["restoration_position_specificity"]),
                ci(write_effects["Gemma4-E4B"]["restoration_position_specificity"]),
                "position-specific",
            ),
        ),
    )
    query_table = table(
        ("Endpoint", "Qwen confirmation", "Gemma confirmation", "Scope"),
        (
            (
                "full commit → next-query targeted attention (vs self)",
                ci(query_effects["Qwen3-8B"]["self"]),
                ci(query_effects["Gemma4-E4B"]["self"]),
                "direct routing endpoint",
            ),
            (
                "full commit → next-query targeted attention (vs orthogonal)",
                ci(query_effects["Qwen3-8B"]["orthogonal"]),
                ci(query_effects["Gemma4-E4B"]["orthogonal"]),
                "Qwen strong; Gemma directional/non-specific",
            ),
            (
                "marker-core → answer count margin",
                ci(terminal_effects["Qwen3-8B"]["restoration"]),
                ci(terminal_effects["Gemma4-E4B"]["restoration"]),
                "fixed-suffix controlled bridge",
            ),
        ),
    )
    walkthrough_table = table(
        ("Model", "Case", "Uninformative baseline", "Full-item path", "Carrier path", "Conclusion"),
        (
            (
                SHORT[model],
                f"seed {walkthrough[model]['seed']}, count 10",
                f"candidate={walkthrough[model]['baselines']['uninformative']['candidate_prediction']}; P(10)={walkthrough[model]['baselines']['uninformative']['gold_count_probability']:.3f}",
                f"exact {walkthrough[model]['conditions']['full_item_restore']['candidate_exact_path_count']}/10; r={walkthrough[model]['conditions']['full_item_restore']['expected_count_correlation_with_occurrence']:+.2f}",
                f"exact {walkthrough[model]['conditions']['counter_carrier_restore']['candidate_exact_path_count']}/10; r={walkthrough[model]['conditions']['counter_carrier_restore']['expected_count_correlation_with_occurrence']:+.2f}",
                "does not walk 1→10",
            )
            for model in MODELS
        ),
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native-thinking counting mechanism：targeted retrieval、recurrent state 与 terminal readout</title><style>{css}</style></head>
<body><article class="page"><header><p class="eyebrow">Realistic CoT NiaH · Native-thinking mechanism</p>
<h1>Native-thinking counting mechanism</h1>
<p class="dek">按 Non-thinking 报告的证据顺序重排：先给结论，再依次检查表征、targeted retrieval、状态写入与传播、终端读取。主文只保留能改变机制判断的结果。</p>
<div class="meta"><span>Qwen3-8B · frozen Top-128</span><span>Gemma4-E4B · frozen Top-6</span><span>formal: 20 discovery / 10 confirmation</span><span>generated {esc(generated)}</span></div></header>
<nav><a href="#summary">结论</a><a href="#design">设计</a><a href="#representation">表征</a><a href="#retrieval">检索</a><a href="#write">写入与循环</a><a href="#answer">终端读取</a><a href="#walkthrough">单 seed</a><a href="#comparison">模型比较</a><a href="#audit">边界与复现</a></nav>
<main>

<section id="summary"><p class="eyebrow">Conclusion first</p><h2>一条 recurrent counting pathway 已经接上；natural end-to-end sufficiency 仍未证明</h2>
<p class="lead">两模型都支持同一类局部因果链：targeted heads 检索下一条 city，改变 grammar-specific marker/tail carrier；carrier 写入 commit state；commit state 再改变下一次 targeted query。终端 marker state 在固定 suffix 的受控实验中能恢复 answer count margin，但把全部上下文抹掉后，仅恢复任一单 item 并不能让答案随 k 从 1 走到 10。</p>
<div class="chain-scroll">{chain_svg()}</div>
<div class="status-grid"><div class="status-card"><h3>Qwen3-8B</h3><p class="status-good">recurrent loop：强 confirmation</p><p>Top-128 retrieval、carrier→commit、commit→next query 都有大效应。terminal marker 的局部受控 restoration 为正。</p><p class="status-open">仍开放：free-running answer count-margin 与全上下文擦除后的单点 sufficiency。</p></div>
<div class="status-card"><h3>Gemma4-E4B</h3><p class="status-good">write 与 terminal local bridge：confirmed</p><p>Top-6 retrieval、carrier→commit 都成立；commit→next query 对 self control 为正，但对 orthogonal control 较弱。</p><p class="status-open">仍开放：query edge 的 selection specificity 与全上下文擦除后的单点 sufficiency。</p></div></div>
<div class="claim"><strong>允许的主张。</strong> Native-thinking trace 中存在一条可重复干预的 recurrent counting pathway；它不是已证明唯一或排他的 counting circuit。</div></section>

<section id="design"><p class="eyebrow">01 · Experimental contract</p><h2>设计与判据</h2>
<p class="lead">正式因果实验固定使用 discovery seeds 1234–1253 与 confirmation seeds 1254–1263；所有正式 pair plan 都是 outcome-blind，且不使用 selection_rank。单 seed walkthrough 只作 case study。</p>
<div class="reading-protocol"><div class="protocol-step"><span class="protocol-no">Discovery</span><h3>定位与冻结</h3><p>20 seeds。选择层、head bank、geometry 与 primary estimand。</p></div><div class="protocol-step"><span class="protocol-no">Confirmation</span><h3>独立复现</h3><p>10 seeds。设计不因 partial outcomes 改动。</p></div><div class="protocol-step"><span class="protocol-no">Claim scope</span><h3>效应优先</h3><p>正文同时给 mean effect 与 95% CI，但机制判断首先看效应方向、大小和控制组。</p></div></div>
<p class="report-note">Head bank：Qwen Top-128；Gemma Top-6。Gemma K=6 的 selected−random retrieval failure 比 K=8 更大，因此最新主线固定 K=6；旧 K=8 仅属于历史实验。</p></section>

<section id="representation"><p class="eyebrow">02 · Running-state representation</p><h2>Trace stream 与 answer query 都表征 count</h2>
<p class="lead">在 discovery 拟合 PCA16 与分类器后，confirmation 上的 count-balanced accuracy 均明显高于十分类 chance=0.10。它证明 count 可读出，但单独不构成因果链。</p>
<figure><h3 class="figure-title">图 1 · Confirmation count decoding</h3>{grouped_bars_svg('Frozen-decoder balanced accuracy', rep_groups, maximum=1.0)}<figcaption>Qwen running commit L18: {float(representation['Qwen3-8B']['running']['confirmation_logistic_balanced_accuracy']):.3f}; answer query L26: {float(representation['Qwen3-8B']['final']['confirmation_logistic_balanced_accuracy']):.3f}. Gemma running commit L16: {float(representation['Gemma4-E4B']['running']['confirmation_logistic_balanced_accuracy']):.3f}; answer query L34: {float(representation['Gemma4-E4B']['final']['confirmation_logistic_balanced_accuracy']):.3f}.</figcaption></figure>
<div class="claim"><strong>结论。</strong> count 信息同时存在于循环中的 commit state 与末端 answer state；后续实验判断这些状态是否被 retrieval、write 和 readout 实际使用。</div></section>

<section id="retrieval"><p class="eyebrow">03 · Targeted retrieval</p><h2>下一条 city 由 model-specific targeted banks 检索</h2>
<p class="lead">在 next-city query 持续 mask 冻结 selected heads，并与 layer-matched random heads 比较。Qwen 需要宽 bank；Gemma 使用窄 bank。</p>
<figure><h3 class="figure-title">图 2 · Targeted-bank dose response</h3>{line_chart((('Qwen3-8B', q_dose), ('Gemma4-E4B', g_dose)))}<figcaption>纵轴是 retrieval failure rate。Selected 曲线随 K 增大显著上升，而 layer-matched random 保持低位；Gemma 的注册 grid 在 K=6 达到最大 selected−random gap。</figcaption></figure>
{targeted_table}
<div class="claim"><strong>结论。</strong> targeted retrieval 是强必要边：Qwen Top-128 的 selected−random failure 为 {100*float(q_primary['selected_minus_random_failure_rate']):+.1f} pp；Gemma Top-6 为 {100*float(g_primary['selected_minus_random_failure_rate']):+.1f} pp。</div></section>

<section id="write"><p class="eyebrow">04 · Write and recurrent propagation</p><h2>Targeted retrieval 写入 grammar carrier，并通过 commit 驱动下一次 query</h2>
<h3>4.1 targeted bank → carrier → commit</h3>
<p>保持 trace tokens 不变，mask selected retrieval bank；随后在相同 token 位置恢复 clean carrier hidden state，并与等 token 数、相近深度的普通状态控制比较。</p>
<figure><h3 class="figure-title">图 3 · Write-edge effect sizes</h3>{grouped_bars_svg('Confirmation mean effects', write_groups)}<figcaption>三组量纲不同，因此条长只在同一行内用于比较模型；精确 effect 与 CI 见下表。Qwen 效应整体更大，Gemma 效应较小但方向一致。</figcaption></figure>
{write_table}
<h3>4.2 commit → next targeted query</h3>
<p>把 donor 的完整 commit hidden state patch 到 receiver 的 commit 位置，然后测量下一次 query 对 donor-successor city 的 targeted-bank attention。Answer query 不参与该干预。</p>
<figure><h3 class="figure-title">图 4 · Recurrent propagation 与 terminal local bridge</h3>{grouped_bars_svg('Confirmation mean effects', query_groups)}<figcaption>Qwen 的 commit→query 对 self 与 orthogonal controls 都很强。Gemma 对 self patch 为 +0.491，但对 orthogonal control 为 +0.126、CI 跨 0，因此只把 selection-specific query edge称为 directional。</figcaption></figure>
{query_table}
<div class="claim"><strong>结论。</strong> Qwen 的 recurrent loop 得到强 confirmation；Gemma 的 write edge 强，但 commit→query specificity 较弱。两者都不要求 count 只存在于一个低维 subspace：完整 hidden state 的因果效应更符合分布式 state。</div></section>

<section id="answer"><p class="eyebrow">05 · Terminal readout</p><h2>Trace 是最终答案的重要信息源；局部 marker bridge 成立，但不是无条件 sufficiency</h2>
<p class="lead">Broad answer-time heads 可以读取 trace 与 prompt；这里先用 source blank 确认 trace 的自然必要性，再用 grammar-aware terminal restoration 检查 state→answer 的局部因果边。</p>
<figure><h3 class="figure-title">图 5 · 移除 trace source 后 exact count accuracy</h3>{grouped_bars_svg('Exact-count accuracy', source_groups, maximum=1.0)}<figcaption>Qwen: 0.97→0.01；Gemma: 0.70→0.12。仅 blank prompt records 不产生同等损伤，说明 answer 主要依赖 trace source，而非回到 prompt 全量重数。</figcaption></figure>
<p>在固定 suffix 的 grammar-span patch 中，marker-core clean-state restoration 对 correct-count margin 为 Qwen {ci(terminal_effects['Qwen3-8B']['restoration'])}、Gemma {ci(terminal_effects['Gemma4-E4B']['restoration'])}；matched-random specificity 也为正。这支持 terminal marker/tail state 被 answer readout 使用。</p>
<div class="negative-result"><strong>Qwen 的边界。</strong> 更自由的 targeted-counter / count-margin 实验没有通过 strong gate；distribution、expected count 与 exact count 也没有形成稳定 recovery。因此正文不再把 Qwen final-answer edge 写成普遍的 end-to-end restoration，只保留“固定 suffix 的局部 terminal bridge”。</div>
<div class="claim"><strong>结论。</strong> final answer 依赖 trace，且 terminal marker state 在受控上下文中有因果作用；尚不能说最后答案只由该 state 或单一 broad bank 决定。</div></section>

<section id="walkthrough"><p class="eyebrow">06 · Non-thinking-style case study</p><h2>一个 seed 从第 1 个 item 走到第 10 个 item：Native-thinking 不复现旧式单点 sufficiency</h2>
<p class="lead">固定一个 count=10 confirmation case；将 prompt records 与完整 trace context 替换为等长普通文本，再分别恢复第 k 个完整 item、grammar-aware marker/tail carrier，或等 token 数普通状态。Patch 从 source layer 延伸到最后 decoder block，但不 patch answer query。</p>
<figure><h3 class="figure-title">图 6 · Full-context scrub 后的 expected-count path</h3>{walkthrough_svg(walkthrough)}<figcaption>橙色虚线是理想路径 y=k。Qwen 三条恢复路径大多停在高 count 区；Gemma 大多停在低 count 区。两者都没有形成 1→10 的对角路径。</figcaption></figure>
{walkthrough_table}
<div class="walkthrough-callout"><div><strong>控制成功。</strong><p>Clean case 均输出 10；uninformative baseline 分别退化为 Qwen candidate 9、Gemma candidate 1，且 P(10) 分别仅 {walkthrough['Qwen3-8B']['baselines']['uninformative']['gold_count_probability']:.4f} / {walkthrough['Gemma4-E4B']['baselines']['uninformative']['gold_count_probability']:.4f}。</p></div><div><strong>Restoration 失败。</strong><p>单 item hidden state 不能在被擦除的下游 trace 动力学中独立决定答案。V1 只擦 parsed items，遗漏 trace tail 的答案泄露，作为 failed-control audit 保留，不进入结果。</p></div></div>
<div class="claim boundary"><strong>结论。</strong> 该 case study 是 descriptive null，不做群体推断。它区分了 Native 与 Non-thinking：Native count 更像循环状态，需要后续 trace 配合传播，而不是一枚可被直接搬到答案端的独立 token code。</div></section>

<section id="comparison"><p class="eyebrow">07 · Mechanism comparison</p><h2>最终机制图景</h2>
<div class="mechanism"><div class="stage"><span class="stage-no">01</span><h3>Retrieve</h3><p>Targeted heads 读取下一条 prompt city。Qwen 使用宽 Top-128；Gemma 使用窄 Top-6。</p><span class="evidence">causal necessity</span></div><div class="stage"><span class="stage-no">02</span><h3>Write & commit</h3><p>检索改变 grammar-specific carrier，carrier 将进度提交到 residual stream。</p><span class="evidence">deform + restore</span></div><div class="stage"><span class="stage-no">03</span><h3>Loop & read</h3><p>commit 改变下一次 targeted query；终端 state 再被 answer-time readout 使用。</p><span class="evidence">recurrent + conditional terminal</span></div></div>
{table(('Model','Strongest connected path','Remaining confound'),((
    'Qwen','Top-128 retrieval → carrier → commit → next query','terminal readout 只在固定 suffix 局部成立；free-running recovery 与 single-seed walk 不成立'),(
    'Gemma','Top-6 retrieval → carrier → commit；terminal local bridge','commit→query 对 orthogonal specificity 较弱；single-seed walk 不成立')))}
<div class="claim"><strong>与 Non-thinking 的核心差异。</strong> Non-thinking 更接近 answer-time broad aggregation + late write；Native-thinking 的主要证据落在 trace 内部的 targeted retrieval 与 recurrent state update。两种模式都可在 answer query 表征 count，但形成 count 的路径不同。</div></section>

<section id="audit"><p class="eyebrow">08 · Boundaries and reproducibility</p><h2>边界、复现与底层文件</h2>
<ul><li>本报告证明一条 pathway，不证明唯一性、排他性或所有 grammar 共用完全相同的 heads。</li><li>CI 与 p-value 保留用于审计；正文的“强/弱”判断同时考虑 effect size、控制组和跨 phase 复现。</li><li>单 seed walkthrough 不进入 discovery/confirmation gate；V2 是在 V1 暴露 trace-tail 泄露后修正的 exploratory control。</li><li>Qwen 与 Gemma 的状态几何、bank 宽度和最后一条边不同，不强行合并成完全同构 circuit。</li></ul>
<details class="paper-appendix"><summary>底层报告与证据文件</summary><div class="source-list"><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">P0 targeted-retrieval atlas</a><br><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a><br><a href="NiaH_Geometry_Comparison.html">Representation geometry</a><br><a href="../work_remote_snapshots/targeted_counter_write_20260822/Qwen3-8B/complete.json">Qwen targeted counter write</a><br><a href="../work_remote_snapshots/targeted_counter_write_20260822/Gemma4-E4B/complete.json">Gemma targeted counter write</a><br><a href="../work_remote_snapshots/commit_state_query_20260822/Qwen3-8B/commit_to_query_complete.json">Qwen commit→query</a><br><a href="../work_remote_snapshots/commit_state_query_20260822/Gemma4-E4B/commit_to_query_complete.json">Gemma commit→query</a><br><a href="../work_remote_snapshots/single_seed_walkthrough_20260822_v2/Qwen3-8B/analysis/walkthrough_complete.json">Qwen single-seed walkthrough</a><br><a href="../work_remote_snapshots/single_seed_walkthrough_20260822_v2/Gemma4-E4B/analysis/walkthrough_complete.json">Gemma single-seed walkthrough</a></div></details>
<p class="audit">Generated UTC: {esc(generated)}<br>Schema: realistic_niah_v5_native_thinking_restructured_v1</p></section>

</main></article></body></html>"""

    input_paths = [
        args.reference_report,
        args.qwen_targeted_analysis,
        args.gemma_targeted_analysis,
        args.representation_root / "site_selected.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_selected.csv",
        *(args.snapshot_root / "targeted_counter_write_20260822" / model / "complete.json" for model in MODELS),
        *(args.snapshot_root / "commit_state_query_20260822" / model / "commit_to_query_complete.json" for model in MODELS),
        args.snapshot_root / "qwen_grammar_span_decomposition_complete.json",
        args.snapshot_root / "gemma_grammar_span_decomposition_complete.json",
        args.snapshot_root / "targeted_counter_20260822" / "Qwen3-8B" / "targeted_counter_complete.json",
        *(args.snapshot_root / "single_seed_walkthrough_20260822_v2" / model / "analysis" / "walkthrough_complete.json" for model in MODELS),
        *(Path(path) for evidence in token_evidence.values() for path in evidence["input_files"]),
    ]
    manifest = {
        "schema_version": "realistic_niah_v5_native_thinking_restructured_v1",
        "status": "PASS",
        "generated_at": generated,
        "output": str(args.output),
        "scientific_contract": {
            "discovery_seed_count": 20,
            "confirmation_seed_count": 10,
            "outcome_blind": True,
            "selection_rank_used": False,
            "qwen_targeted_bank": 128,
            "gemma_targeted_bank": 6,
        },
        "claim_scope": {
            "recurrent_pathway_supported": True,
            "exclusive_circuit_claimed": False,
            "natural_end_to_end_single_state_sufficiency": False,
            "single_seed_walkthrough_inferential": False,
        },
        "inputs_sha256": {str(path): sha256(path) for path in input_paths},
    }
    return html_text, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, default=Path("reports/NiaH_Non-thinking_report.html"))
    parser.add_argument("--qwen-targeted-analysis", type=Path, default=Path("reports/v5_native_final_localizers/analysis/qwen_final_merged_dose_grid.json"))
    parser.add_argument("--gemma-targeted-analysis", type=Path, default=Path("reports/v5_native_hybrid_supplement/Gemma4-E4B/analysis_hybrid_supplement_registered_v1/hybrid_dose_grid_complete.json"))
    parser.add_argument("--representation-root", type=Path, default=Path("reports/v5_native_causal_aligned_geometry"))
    parser.add_argument("--dual-endpoint-root", type=Path, default=Path("reports/v5_dual_endpoint_geometry_full300"))
    parser.add_argument("--token-ablation-root", type=Path, default=Path("reports/v5_native_token_level_ablation"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("work_remote_snapshots"))
    parser.add_argument("--output", type=Path, default=Path("reports/NiaH_Native-Thinking_report.html"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/v5_native_final_localizers/report_manifest_restructured.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, manifest = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    manifest["output_sha256"] = sha256(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": manifest["output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
