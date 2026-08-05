from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | int | None, digits: int = 4, *, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    value = float(value)
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def holm_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """Return monotone Holm family-wise adjusted p-values in input order."""
    order = sorted(range(len(pvalues)), key=lambda index: float(pvalues[index]))
    adjusted = [1.0] * len(pvalues)
    running_max = 0.0
    family_size = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * float(pvalues[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def ci_text(
    row: dict[str, Any],
    *,
    mean: str = "mean",
    low: str = "ci95_low",
    high: str = "ci95_high",
    digits: int = 4,
) -> str:
    return (
        f"{fmt(row[mean], digits)} [{fmt(row[low], digits)}, {fmt(row[high], digits)}]"
    )


def table(
    headers: list[str], rows: Iterable[Iterable[str]], *, classes: str = "paper-table"
) -> str:
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="table-scroll"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details_table(
    title: str,
    headers: list[str],
    rows: Iterable[Iterable[str]],
    *,
    opened: bool = False,
) -> str:
    opened_attr = " open" if opened else ""
    rendered = list(rows)
    return (
        f'<details class="data-table"{opened_attr}>'
        f"<summary>{html.escape(title)} · {len(rendered)} rows</summary>"
        f"{table(headers, rendered)}"
        "</details>"
    )


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if low == high:
        return [low]
    span = high - low
    raw = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(abs(raw)))
    normalized = raw / power
    if normalized <= 1:
        step = 1 * power
    elif normalized <= 2:
        step = 2 * power
    elif normalized <= 5:
        step = 5 * power
    else:
        step = 10 * power
    start = math.floor(low / step) * step
    stop = math.ceil(high / step) * step
    ticks: list[float] = []
    value = start
    while value <= stop + step * 0.25 and len(ticks) < 20:
        ticks.append(value)
        value += step
    return ticks


def forest_svg(
    rows: list[dict[str, Any]],
    *,
    title: str,
    description: str,
    x_label: str,
    width: int = 1180,
    left: int = 310,
    right: int = 300,
    zero: float = 0.0,
    colors: tuple[str, ...] = ("#6750E8", "#00A88F", "#D94B86", "#D6B52C"),
) -> str:
    top, bottom = 44, 74
    row_h = 54
    height = top + bottom + row_h * len(rows)
    svg_slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "effect"
    title_id = f"forest-{svg_slug}-title"
    desc_id = f"forest-{svg_slug}-desc"
    values = [zero]
    for row in rows:
        values.extend([float(row["low"]), float(row["high"]), float(row["mean"])])
    low, high = min(values), max(values)
    pad = max((high - low) * 0.12, 0.01)
    low, high = low - pad, high + pad
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + (value - low) / (high - low) * plot_w

    parts = [
        f'<svg class="stat-svg integrated-forest" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {desc_id}">',
        f'<title id="{title_id}">{html.escape(title)}</title>',
        f'<desc id="{desc_id}">{html.escape(description)}</desc>',
    ]
    ticks = _nice_ticks(low, high, 6)
    y_axis = top + row_h * len(rows)
    for tick in ticks:
        if tick < low - 1e-12 or tick > high + 1e-12:
            continue
        tx = x(tick)
        parts.append(
            f'<line class="grid" x1="{tx:.1f}" y1="{top - 12}" x2="{tx:.1f}" y2="{y_axis}"/>'
        )
        parts.append(
            f'<text class="tick" x="{tx:.1f}" y="{y_axis + 22}" text-anchor="middle">{fmt(tick, 3)}</text>'
        )
    zx = x(zero)
    parts.append(
        f'<line class="zero" x1="{zx:.1f}" y1="{top - 14}" x2="{zx:.1f}" y2="{y_axis}"/>'
    )
    for idx, row in enumerate(rows):
        cy = top + idx * row_h + row_h / 2
        lo, hi, mean = float(row["low"]), float(row["high"]), float(row["mean"])
        color = str(row.get("color") or colors[idx % len(colors)])
        parts.append(
            f'<text class="row-label" x="{left - 18}" y="{cy + 4:.1f}" text-anchor="end">{html.escape(str(row["label"]))}</text>'
        )
        parts.append(
            f'<line class="ci" x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" style="stroke:{color}"/>'
        )
        parts.append(
            f'<line class="cap" x1="{x(lo):.1f}" y1="{cy - 6:.1f}" x2="{x(lo):.1f}" y2="{cy + 6:.1f}" style="stroke:{color}"/>'
        )
        parts.append(
            f'<line class="cap" x1="{x(hi):.1f}" y1="{cy - 6:.1f}" x2="{x(hi):.1f}" y2="{cy + 6:.1f}" style="stroke:{color}"/>'
        )
        parts.append(
            f'<circle class="dot" cx="{x(mean):.1f}" cy="{cy:.1f}" r="6" style="fill:{color}"/>'
        )
        parts.append(
            f'<text class="value-label" x="{x(hi) + 10:.1f}" y="{cy + 4:.1f}">{html.escape(str(row.get("value", fmt(mean, 4))))}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 14}" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def write_trace_svg(
    rows: list[dict[str, Any]],
    *,
    id_prefix: str = "write",
    title: str = "L28 natural OV write propagates through L35",
    description: str = "Layer is on the horizontal axis. Natural-minus-orthogonal count-axis coefficient is on the vertical axis. Points are seed means and bars are 95 percent bootstrap confidence intervals.",
) -> str:
    width, height = 1040, 500
    left, right, top, bottom = 90, 38, 36, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    layers = [int(row["layer"]) for row in rows]
    raw_low = min(0.0, min(float(row["ci95_low"]) for row in rows))
    raw_high = max(0.0, max(float(row["ci95_high"]) for row in rows))
    span = max(raw_high - raw_low, 1e-6)
    ymin = raw_low - (0.10 * span if raw_low < 0 else 0.0)
    ymax = raw_high + (0.18 * span if raw_high > 0 else 0.10 * span)

    def x(layer: int) -> float:
        return left + (layer - min(layers)) / max(max(layers) - min(layers), 1) * plot_w

    def y(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    parts = [
        f'<svg class="stat-svg write-trace" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{id_prefix}-title {id_prefix}-desc">',
        f'<title id="{id_prefix}-title">{html.escape(title)}</title>',
        f'<desc id="{id_prefix}-desc">{html.escape(description)}</desc>',
    ]
    for tick in _nice_ticks(ymin, ymax, 6):
        if tick < ymin or tick > ymax:
            continue
        yy = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end">{fmt(tick, 3)}</text>'
        )
    if ymin < 0 < ymax:
        zero_y = y(0.0)
        parts.append(
            f'<line class="zero" x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}"/>'
        )
    path = " ".join(
        ("M" if idx == 0 else "L")
        + f" {x(int(row['layer'])):.1f} {y(float(row['mean'])):.1f}"
        for idx, row in enumerate(rows)
    )
    parts.append(f'<path class="trace-line" d="{path}"/>')
    for row in rows:
        xx = x(int(row["layer"]))
        yy = y(float(row["mean"]))
        lo_y, hi_y = y(float(row["ci95_low"])), y(float(row["ci95_high"]))
        parts.append(
            f'<line class="trace-ci" x1="{xx:.1f}" y1="{hi_y:.1f}" x2="{xx:.1f}" y2="{lo_y:.1f}"/>'
        )
        parts.append(f'<circle class="trace-dot" cx="{xx:.1f}" cy="{yy:.1f}" r="6"/>')
        parts.append(
            f'<text class="value-label" x="{xx:.1f}" y="{yy - 12:.1f}" text-anchor="middle">{fmt(float(row["mean"]), 3)}</text>'
        )
        parts.append(
            f'<text class="tick" x="{xx:.1f}" y="{height - bottom + 24}" text-anchor="middle">L{int(row["layer"])}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle">decoder layer</text>'
    )
    parts.append(
        f'<text class="axis-label" transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">natural − orthogonal count-axis coefficient</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def evidence_gate_svg(
    families: list[dict[str, Any]], *, id_prefix: str = "gate"
) -> str:
    width, height = 1040, 470
    positions = [(28, 38), (530, 38), (28, 246), (530, 246)]
    parts = [
        f'<svg class="stat-svg gate-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{id_prefix}-title {id_prefix}-desc">',
        f'<title id="{id_prefix}-title">Four preregistered natural-OV evidence gates</title>',
        f'<desc id="{id_prefix}-desc">Four boxes summarize natural signal, true pre-O sufficiency, centered z-space necessity, and path mediation. A check or cross marks the family-level decision; the global intersection-union p value is the largest family p value.</desc>',
    ]
    for idx, family in enumerate(families):
        x, y = positions[idx]
        passed = bool(family.get("passed", True))
        status_class = "gate-pass" if passed else "gate-fail"
        parts.append(
            f'<rect class="gate-box {status_class}" x="{x}" y="{y}" width="482" height="174" rx="8"/>'
        )
        parts.append(
            f'<circle class="gate-check {status_class}" cx="{x + 34}" cy="{y + 34}" r="15"/>'
        )
        parts.append(
            f'<text class="gate-check-text" x="{x + 34}" y="{y + 39}" text-anchor="middle">{"✓" if passed else "×"}</text>'
        )
        parts.append(
            f'<text class="gate-heading" x="{x + 60}" y="{y + 40}">{html.escape(family["title"])}</text>'
        )
        parts.append(
            f'<text class="gate-main" x="{x + 24}" y="{y + 83}">{html.escape(family["main"])}</text>'
        )
        parts.append(
            f'<text class="gate-sub" x="{x + 24}" y="{y + 112}">{html.escape(family["sub"])}</text>'
        )
        parts.append(
            f'<text class="gate-p" x="{x + 24}" y="{y + 145}">{html.escape(family["p"])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def relay_gate_svg(metrics: list[dict[str, Any]]) -> str:
    width, height = 1040, 300
    box_w, gap, y = 185, 22, 70
    parts = [
        f'<svg class="stat-svg relay-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="relay-title relay-desc">',
        '<title id="relay-title">Registered tail-64 relay hypothesis fails downstream causal gates</title>',
        '<desc id="relay-desc">The selected late position set carries count and permits a mechanical first stage, but behavioral transport, OV mediation and natural removal fail their registered directional tests.</desc>',
    ]
    for idx, metric in enumerate(metrics):
        x = 20 + idx * (box_w + gap)
        passed = bool(metric["passed"])
        klass = "relay-pass" if passed else "relay-fail"
        parts.append(
            f'<rect class="relay-box {klass}" x="{x}" y="{y}" width="{box_w}" height="138" rx="7"/>'
        )
        parts.append(
            f'<text class="relay-mark" x="{x + box_w / 2}" y="{y + 32}" text-anchor="middle">{"✓" if passed else "×"}</text>'
        )
        parts.append(
            f'<text class="relay-heading" x="{x + box_w / 2}" y="{y + 60}" text-anchor="middle">{html.escape(str(metric["label"]))}</text>'
        )
        parts.append(
            f'<text class="relay-value" x="{x + box_w / 2}" y="{y + 91}" text-anchor="middle">{html.escape(str(metric["value"]))}</text>'
        )
        parts.append(
            f'<text class="relay-p" x="{x + box_w / 2}" y="{y + 118}" text-anchor="middle">{html.escape(str(metric["p"]))}</text>'
        )
        if idx < len(metrics) - 1:
            x1 = x + box_w + 4
            x2 = x + box_w + gap - 4
            parts.append(
                f'<line class="relay-arrow" x1="{x1}" y1="{y + 69}" x2="{x2}" y2="{y + 69}"/>'
            )
    parts.append(
        '<text class="relay-summary" x="520" y="258" text-anchor="middle">global IUT p = 0.9981 · registered serial path NOT SUPPORTED</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def mechanism_svg() -> str:
    width, height = 1180, 520
    nodes = [
        (
            20,
            82,
            200,
            122,
            "Prompt running index",
            ["needle-end states", "ordered across occurrences"],
        ),
        (
            252,
            82,
            200,
            122,
            "Broad retrieval bank",
            ["L23H28, L23H29", "L26H20, L27H18"],
        ),
        (
            484,
            82,
            220,
            122,
            "L28 mixed read",
            ["α-routing + V-content", "H16–H19 mediator set"],
        ),
        (
            736,
            82,
            196,
            122,
            "Natural OV write",
            ["pre-O z → W_O span", "H19 nonredundant"],
        ),
        (
            964,
            82,
            196,
            122,
            "Late answer state",
            ["L29–L35 propagation", "LM-head count distribution"],
        ),
    ]
    parts = [
        f'<svg class="stat-svg mechanism-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="mech-title mech-desc">',
        '<title id="mech-title">Supported non-thinking counting mechanism</title>',
        '<desc id="mech-desc">Prompt-side running-index representations are read by an early broad retrieval bank. Independent serial mediation supports transport through the L28 H16 to H19 set, which reads both attention routing and value content and writes a natural count component that propagates to the late answer state. A tested tail-64 relay is rejected.</desc>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#6750E8"/></marker><marker id="arrow-dashed" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#718096"/></marker></defs>',
    ]
    for idx, (x, y, w, h, title, lines) in enumerate(nodes):
        parts.append(
            f'<rect class="mech-node mech-{idx}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>'
        )
        parts.append(
            f'<text class="mech-heading" x="{x + w / 2}" y="{y + 34}" text-anchor="middle">{html.escape(title)}</text>'
        )
        for line_idx, line in enumerate(lines):
            parts.append(
                f'<text class="mech-sub" x="{x + w / 2}" y="{y + 68 + line_idx * 23}" text-anchor="middle">{html.escape(line)}</text>'
            )
        if idx < len(nodes) - 1:
            next_x = nodes[idx + 1][0]
            parts.append(
                f'<line class="mech-arrow" x1="{x + w + 5}" y1="{y + h / 2}" x2="{next_x - 8}" y2="{y + h / 2}" marker-end="url(#arrow)"/>'
            )
    parts.append(
        '<text class="mech-evidence" x="600" y="38" text-anchor="middle">solid arrows: causal transport/mediation supported · boxes: localization granularity of current evidence</text>'
    )
    parts.append(
        '<rect class="mech-negative" x="454" y="318" width="272" height="110" rx="10"/>'
    )
    parts.append(
        '<text class="mech-heading" x="590" y="351" text-anchor="middle">Rejected relay candidate</text>'
    )
    parts.append(
        '<text class="mech-sub" x="590" y="381" text-anchor="middle">pre-query non-slot tail-64</text>'
    )
    parts.append(
        '<text class="mech-sub" x="590" y="405" text-anchor="middle">carrier present; natural mediation absent</text>'
    )
    parts.append(
        '<line class="mech-dashed" x1="590" y1="318" x2="590" y2="214" marker-end="url(#arrow-dashed)"/>'
    )
    parts.append(
        '<text class="mech-boundary" x="590" y="479" text-anchor="middle">Not established: clean-run necessity of each early head, a unique tokenwise +1 operator, or cross-model identity of this microcircuit.</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def ablation_topk_svg_legacy() -> str:
    """Plot the observed magnitude of ranked-minus-random count-shift contrasts."""
    width, height = 1040, 520
    left, right, top, bottom = 108, 72, 58, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    series = {
        "Qwen3-8B": {"color": "#6750E8", "values": [(4, 0.425), (8, 0.025)]},
        "Gemma4-E4B": {"color": "#00A88F", "values": [(4, 2.025), (8, 2.625)]},
    }
    ymax = 3.0

    def x(k: int) -> float:
        return left + (k - 4) / 4 * plot_w

    def y(value: float) -> float:
        return top + (ymax - value) / ymax * plot_h

    parts = [
        f'<svg class="stat-svg ablation-topk" viewBox="0 0 {width} {height}" role="img" aria-labelledby="ablation-topk-title ablation-topk-desc">',
        '<title id="ablation-topk-title">Top-k ranked-bank ablation effect magnitude</title>',
        '<desc id="ablation-topk-desc">The horizontal axis is k, either four or eight heads. The vertical axis is the absolute ranked-minus-layer-matched-random generated-count shift. Qwen decreases from 0.425 to 0.025; Gemma increases from 2.025 to 2.625.</desc>',
    ]
    for tick in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
        yy = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 14}" y="{yy + 4:.1f}" text-anchor="end">{tick:.1f}</text>'
        )
    for k in (4, 8):
        xx = x(k)
        parts.append(
            f'<line class="x-guide" x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height - bottom}"/>'
        )
        parts.append(
            f'<text class="tick" x="{xx:.1f}" y="{height - bottom + 28}" text-anchor="middle">{k}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>'
    )
    for label, payload in series.items():
        color = payload["color"]
        values = payload["values"]
        path = " ".join(
            ("M" if idx == 0 else "L") + f" {x(k):.1f} {y(value):.1f}"
            for idx, (k, value) in enumerate(values)
        )
        parts.append(f'<path class="series-line" d="{path}" style="stroke:{color}"/>')
        for k, value in values:
            xx, yy = x(k), y(value)
            label_y = yy - 15 if value > 0.2 else yy - 17
            anchor = "start" if k == 4 else "end"
            label_x = xx + 12 if k == 4 else xx - 12
            parts.append(
                f'<circle class="series-dot" cx="{xx:.1f}" cy="{yy:.1f}" r="8" style="fill:{color}"/>'
            )
            parts.append(
                f'<text class="point-label" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" style="fill:{color}">{html.escape(label)} · {value:.3f}</text>'
            )
    parts.append(
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle">top-k head-set size</text>'
    )
    parts.append(
        f'<text class="axis-label" transform="translate(25 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">|ranked − random count shift|</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def ablation_topk_svg(seed_confirmation: dict[str, Any]) -> str:
    """Fresh-seed frozen top-k ablation effects with seed-cluster CIs."""
    width, height = 1120, 520
    panels = [
        {
            "x0": 80,
            "width": 455,
            "title": "All examples: |generated-count shift|",
            "metric": "all_absolute_shift",
            "ymax": 0.30,
            "ylabel": "ranked - random |count shift|",
        },
        {
            "x0": 635,
            "width": 405,
            "title": "Clean-correct: correct-to-wrong excess",
            "metric": "clean_correct_to_wrong",
            "ymax": 0.20,
            "ylabel": "ranked - random failure rate",
        },
    ]
    colors = {"Qwen3-8B": "#6750E8", "Gemma4-E4B": "#00A88F"}
    top, bottom = 82, 96
    plot_h = height - top - bottom
    parts = [
        f'<svg class="stat-svg ablation-topk" viewBox="0 0 {width} {height}" role="img" aria-labelledby="ablation-topk-title ablation-topk-desc">',
        '<title id="ablation-topk-title">Fresh-seed frozen top-k head-bank ablation</title>',
        '<desc id="ablation-topk-desc">Two panels show ranked-minus-layer-matched-random ablation effects at frozen top-k values. Points are effects and vertical bars are seed-cluster bootstrap 95 percent confidence intervals.</desc>',
    ]
    for panel_index, panel in enumerate(panels):
        x0 = float(panel["x0"])
        panel_w = float(panel["width"])
        ymax = float(panel["ymax"])
        plot_left, plot_right = x0 + 58, x0 + panel_w - 18

        def x(k: int) -> float:
            return plot_left + (int(k) - 1) / 3 * (plot_right - plot_left)

        def y(value: float) -> float:
            return top + (ymax - float(value)) / ymax * plot_h

        parts.append(
            f'<text class="panel-title" x="{x0 + panel_w / 2:.1f}" y="34" text-anchor="middle">{html.escape(str(panel["title"]))}</text>'
        )
        for tick_index in range(5):
            tick = ymax * tick_index / 4
            yy = y(tick)
            parts.append(
                f'<line class="grid" x1="{plot_left:.1f}" y1="{yy:.1f}" x2="{plot_right:.1f}" y2="{yy:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{plot_left - 9:.1f}" y="{yy + 4:.1f}" text-anchor="end">{tick:.2f}</text>'
            )
        parts.append(
            f'<line class="axis" x1="{plot_left:.1f}" y1="{height - bottom}" x2="{plot_right:.1f}" y2="{height - bottom}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{plot_left:.1f}" y1="{top}" x2="{plot_left:.1f}" y2="{height - bottom}"/>'
        )
        for k in (1, 2, 3, 4):
            xx = x(k)
            parts.append(
                f'<text class="tick" x="{xx:.1f}" y="{height - bottom + 25}" text-anchor="middle">{k}</text>'
            )
        parts.append(
            f'<text class="axis-label" x="{(plot_left + plot_right) / 2:.1f}" y="{height - 34}" text-anchor="middle">frozen top-k</text>'
        )
        parts.append(
            f'<text class="axis-label" transform="translate({x0 + 12:.1f} {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(str(panel["ylabel"]))}</text>'
        )
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            model_rows = seed_confirmation["models"][model]
            points = []
            for k_text, metrics in sorted(
                model_rows.items(), key=lambda item: int(item[0])
            ):
                item = metrics[str(panel["metric"])]
                points.append((int(k_text), item))
            if len(points) > 1:
                path = " ".join(
                    ("M" if index == 0 else "L")
                    + f" {x(k):.1f} {y(item['effect']):.1f}"
                    for index, (k, item) in enumerate(points)
                )
                parts.append(
                    f'<path class="series-line" d="{path}" style="stroke:{colors[model]}"/>'
                )
            for k, item in points:
                xx, yy = x(k), y(item["effect"])
                low_y, high_y = y(item["ci95_low"]), y(item["ci95_high"])
                parts.append(
                    f'<line class="ci" x1="{xx:.1f}" y1="{low_y:.1f}" x2="{xx:.1f}" y2="{high_y:.1f}" style="stroke:{colors[model]}"/>'
                )
                parts.append(
                    f'<line class="cap" x1="{xx - 5:.1f}" y1="{low_y:.1f}" x2="{xx + 5:.1f}" y2="{low_y:.1f}" style="stroke:{colors[model]}"/>'
                )
                parts.append(
                    f'<line class="cap" x1="{xx - 5:.1f}" y1="{high_y:.1f}" x2="{xx + 5:.1f}" y2="{high_y:.1f}" style="stroke:{colors[model]}"/>'
                )
                parts.append(
                    f'<circle class="series-dot" cx="{xx:.1f}" cy="{yy:.1f}" r="7" style="fill:{colors[model]}"/>'
                )
                label_y = max(top + 12, high_y - 9)
                parts.append(
                    f'<text class="point-label" x="{xx:.1f}" y="{label_y:.1f}" text-anchor="middle" style="fill:{colors[model]}">{item["effect"]:.3f}</text>'
                )
        if panel_index == 0:
            parts.append(
                '<text class="legend-label" x="150" y="61" style="fill:#6750E8">● Qwen3-8B</text>'
            )
            parts.append(
                '<text class="legend-label" x="285" y="61" style="fill:#00A88F">● Gemma4-E4B</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def build_mechanism_overview(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    gemma_story: dict[str, Any],
) -> str:
    global_ov_p = float(ov["primary_decision"]["global_intersection_union_p"])
    upstream_p = float(upstream["primary_decision"]["intersection_union_p"])
    read_metric_names = {
        "read_routing_behavior_transport",
        "read_value_behavior_transport",
    }
    rw = {
        row["metric"]: row
        for row in read_write["summary"]
        if row.get("stratum") == "all" and row.get("metric") in read_metric_names
    }
    routing_p = float(rw["read_routing_behavior_transport"]["exact_sign_flip_p"])
    value_p = float(rw["read_value_behavior_transport"]["exact_sign_flip_p"])
    gemma_status = html.escape(str(gemma_story["summary"]))
    gemma_p = (
        fmt_p(gemma_story.get("global_p"))
        if gemma_story.get("global_p") is not None
        else "未形成全局通过值"
    )
    return f"""
<section id="mechanism-overview" class="mechanism-main">
<div class="main-figure-kicker">PAPER MAIN FIGURE · SHARED COMPUTATION + MODEL-SPECIFIC CAUSAL RESOLUTION</div>
<h2>Non-thinking counting：从读 prompt 到写出 <code>Total:N</code></h2>
<p class="figure-intro">点击“下一步”或“播放一次”查看五阶段计算。关键新增点是：<strong>OV 不是可省略的命名，而是把 attention head-space 中读出的 state 写入 residual 坐标系的线性变换</strong>。框内 layer/head 是 Qwen 已闭合路径的具体实例；Gemma 采用由强到弱的冻结证据阶梯，不强迫复制 Qwen 的 head identity。当前 Gemma 判定为：{gemma_status}</p>
<figure class="mechanism-walkthrough" aria-labelledby="walkthrough-caption">
<div class="mechanism-canvas-wrap">
<svg viewBox="0 0 1180 430" role="img" aria-labelledby="walk-main-title walk-main-desc">
<title id="walk-main-title">Stepwise non-thinking counting mechanism</title>
<desc id="walk-main-desc">Five stages show repeated records forming a running-index state, a broad retrieval bank, a routing and value read, an output projection that changes representation coordinates while writing to the residual stream, and a late answer state that produces Total colon N.</desc>
<defs><marker id="walk-arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="currentColor"/></marker></defs>
<g class="walk-input" data-walk-step="0">
  <rect x="18" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="113" y="112" text-anchor="middle">Repeated records</text>
  <text class="walk-token" x="43" y="151">① city · score</text><text class="walk-token" x="43" y="184">② city · score</text>
  <text class="walk-token" x="43" y="217">…</text><text class="walk-token" x="43" y="250">⑩ city · score</text>
  <text class="walk-sub" x="113" y="282" text-anchor="middle">10k-token haystack</text>
</g>
<path class="walk-edge" data-walk-edge="1" d="M214 191 L272 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node" data-walk-step="1">
  <rect x="280" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="375" y="112" text-anchor="middle">Running index</text>
  <path class="mini-manifold" d="M310 258 C325 241 334 229 346 212 S369 178 386 168 S419 142 442 128"/>
  <circle cx="310" cy="258" r="7"/><circle cx="346" cy="212" r="7"/><circle cx="386" cy="168" r="7"/><circle cx="442" cy="128" r="7"/>
  <text class="walk-sub" x="375" y="282" text-anchor="middle">needle-end residuals</text>
</g>
<path class="walk-edge" data-walk-edge="2" d="M476 191 L534 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node" data-walk-step="2">
  <rect x="542" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="637" y="112" text-anchor="middle">Broad retrieval</text>
  <text class="walk-head" x="637" y="157" text-anchor="middle">L23H28 · L23H29</text>
  <text class="walk-head" x="637" y="190" text-anchor="middle">L26H20 · L27H18</text>
  <path class="fan-line" d="M578 235 L620 209 M610 248 L632 209 M660 209 L692 246"/>
  <text class="walk-sub" x="637" y="282" text-anchor="middle">distributed slot-state read</text>
</g>
<path class="walk-edge" data-walk-edge="3" d="M738 191 L796 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node" data-walk-step="3">
  <rect x="804" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="899" y="112" text-anchor="middle">Read → OV coordinate write</text>
  <text class="walk-head" x="899" y="151" text-anchor="middle">Qwen L28 · H16/H19</text>
  <text class="walk-formula" x="899" y="190" text-anchor="middle">z<tspan baseline-shift="sub">S</tspan> = {{Σ α<tspan baseline-shift="sub">h</tspan>V<tspan baseline-shift="sub">h</tspan>}}</text>
  <text class="walk-formula" x="899" y="225" text-anchor="middle">w<tspan baseline-shift="sub">S</tspan> = Σ W<tspan baseline-shift="sub">O</tspan><tspan baseline-shift="super">h</tspan>z<tspan baseline-shift="sub">h</tspan></text>
  <text class="walk-sub" x="899" y="258" text-anchor="middle">count preserved; coordinates may rotate</text>
  <text class="walk-sub" x="899" y="282" text-anchor="middle">u<tspan baseline-shift="sub">P</tspan> need not be parallel to u<tspan baseline-shift="sub">A</tspan></text>
</g>
<path class="walk-edge" data-walk-edge="4" d="M1000 191 L1050 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node walk-output" data-walk-step="4">
  <rect x="1058" y="80" width="104" height="222" rx="12"/>
  <text class="walk-title" x="1110" y="112" text-anchor="middle">Answer</text>
  <text class="walk-answer" x="1110" y="188" text-anchor="middle">Total:</text>
  <text class="walk-answer-number" x="1110" y="234" text-anchor="middle">N</text>
  <text class="walk-sub" x="1110" y="282" text-anchor="middle">L29–L35</text>
</g>
<text class="walk-boundary walk-model-status" x="590" y="342" text-anchor="middle">Qwen: localized natural OV confirmed, global IUT p={fmt_p(global_ov_p)}</text>
<text class="walk-boundary walk-model-status" x="590" y="366" text-anchor="middle">Gemma: {html.escape(str(gemma_story["label"]))} effective residual write, p={gemma_p}; localized OV set unresolved</text>
<text class="walk-boundary" x="590" y="402" text-anchor="middle">solid path = causal transport/mediation support; node width does not encode effect size</text>
</svg>
</div>
<div class="mechanism-controls" aria-label="Mechanism animation controls">
  <button type="button" id="mechanism-prev">← 上一步</button>
  <button type="button" id="mechanism-play">▶ 播放一次</button>
  <button type="button" id="mechanism-next">下一步 →</button>
  <div class="step-dots" role="group" aria-label="直接选择机制阶段">
    <button type="button" data-mechanism-step="0" aria-label="步骤 1">1</button><button type="button" data-mechanism-step="1" aria-label="步骤 2">2</button>
    <button type="button" data-mechanism-step="2" aria-label="步骤 3">3</button><button type="button" data-mechanism-step="3" aria-label="步骤 4">4</button>
    <button type="button" data-mechanism-step="4" aria-label="步骤 5">5</button>
  </div>
</div>
<div id="mechanism-live" class="mechanism-live" aria-live="polite"></div>
<figcaption id="walkthrough-caption"><strong>Main Figure · Stepwise non-thinking mechanism.</strong> 图中没有数值坐标轴；高亮按时间顺序展示抽象证据链。Qwen early→L28 fresh-seed IUT p={fmt_p(upstream_p)}，routing/value p={fmt_p(routing_p)} / {fmt_p(value_p)}，natural-OV global IUT p={fmt_p(global_ov_p)}；Gemma 的最强完整证据层级为 <code>{html.escape(str(gemma_story["kind"]))}</code>，联合 p={gemma_p}。OV 框表示 head output 经 <em>W</em><sub>O</sub> 写回 residual，而不是假定 prompt 与 answer 的 count axis 是同一向量。所有 effect、CI 与单门定义见第 8–10 节。</figcaption>
</figure>
<div class="plain-protocol ov-coordinate-note">
<h4>为什么 prompt counter 与 answer counter 可以不在同一方向？</h4>
<ol>
  <li>在 prompt 位置，用单位向量 <code>u<sub>P</sub></code> 表示 occurrence index 的 residual-space count direction。</li>
  <li>answer query 的 head set 先形成 <code>z<sub>S</sub>(q,c)={{Σ<sub>j</sub>α<sub>h</sub>(q,j)W<sub>V</sub><sup>g(h)</sup>x<sub>j</sub>}}<sub>h∈S</sub></code>；这是 head-space state，不要求与 <code>u<sub>P</sub></code> 共线。</li>
  <li>OV 写回为 <code>w<sub>S</sub>(c)=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q,c)</code>；后续 attention/MLP 的局部 Jacobian 继续传播，因此 <code>u<sub>A</sub> ∝ J<sub>ℓ→A</sub>w<sub>S</sub></code>。</li>
</ol>
<div class="equation">保留的是 count ordering / decodability / causal transport；不要求 u<sub>P</sub> ∥ u<sub>A</sub>。</div>
</div>
{table(
    ["模型", "当前写入证据", "允许的机制表述"],
    [
        [
            "Qwen3-8B",
            f"L28 H16/H19 natural signal + true pre-O injection + centered removal + mediation；global IUT p={fmt_p(global_ov_p)}",
            "已定位 set-level OV 坐标变换/写回，再沿 L29–L35 传播",
        ],
        [
            "Gemma4-E4B",
            f"localized OV 候选未闭合；{html.escape(str(gemma_story['label']))} distributed residual path p={gemma_p}",
            "已确认有效的分布式 residual 写入；尚不能指定唯一 W_O head set",
        ],
    ],
)}
<div class="conclusion"><strong>这张图的主张</strong>non-thinking counting 是“prompt running state 被读取，经 OV/后续 block 改换坐标并写成 answer state”。OV 在计算图上必然存在，但“存在 OV 运算”与“某个冻结 head set 已被因果定位”是两个命题：Qwen 两者都成立；Gemma 目前只闭合到 distributed effective write。第 3–11 节给出表征、干预、p 值与边界。</div>
</section>
"""


def build_running_index_block() -> str:
    return """
<div class="figure-block running-index-block">
<h3>3.1 Running-index 3D · 逐个 occurrence 播放</h3>
<p class="figure-intro">这张图先把 prompt counter 解释成“读取进度”：播放 n=1→10 时，彩色大点沿冻结 PCA 空间前进；当前 n 的半透明小点是 30 个 V4.4 seeds 的真实 needle-end states。它先展示 centroid trajectory，下一张 3D 再提供 layer、split、outcome 与 PC 轴的完整交互。</p>
<figure>
<div class="running-controls">
  <label>model <select id="running-model"><option value="Qwen3-8B">Qwen3-8B</option><option value="Gemma4-E4B">Gemma4-E4B</option></select></label>
  <button type="button" id="running-prev">← n−1</button>
  <button type="button" id="running-play">▶ 播放一次</button>
  <button type="button" id="running-next">n+1 →</button>
  <label class="running-slider">running index <input id="running-step" type="range" min="1" max="10" value="1" step="1" aria-label="Running index from one to ten"></label>
</div>
<div class="plot-shell running-shell"><canvas id="running-index-canvas" aria-label="Interactive three-dimensional running-index centroid trajectory"></canvas></div>
<div id="running-status" class="running-status" aria-live="polite"></div>
<figcaption><strong>Figure · Running-index trajectory in frozen PCA coordinates.</strong> 横轴、纵轴、深度轴分别是 PC1、PC2、PC3；Qwen 默认使用 prompt manifold-display L8，Gemma 使用其数据中标记的 manifold-display layer。PCA basis 在 disjoint V4.1 discovery rows 上冻结，随后投影全部 30 个 V4.4 seeds（1234–1263）；所以播放不会重新拟合 basis。拖拽旋转、滚轮缩放。三维距离只作可视化，显著性与效应大小仍由 full-space 统计和因果实验判断。</figcaption>
</figure>
<div class="conclusion"><strong>图的判读</strong>轨迹有序说明 occurrence index 在 residual 中可解码；它本身不说明模型是否读取这条轴，也不等于每个 n 都被一个单独 head 保存。</div>
</div>
"""


def extract_embedded_json(document: str, variable_name: str) -> dict[str, Any]:
    marker = f"const {variable_name}="
    start = document.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not locate embedded {variable_name}")
    start += len(marker)
    end = document.find(";\nconst ", start)
    if end < 0:
        end = document.find(";</script>", start)
    if end < 0:
        raise RuntimeError(f"Could not locate end of embedded {variable_name}")
    return json.loads(document[start:end])


def _centroid_distance_correlation(
    left_rows: list[list[Any]], right_rows: list[list[Any]]
) -> float:
    def centroids(rows: list[list[Any]]) -> list[list[float]]:
        grouped: dict[int, list[list[float]]] = {}
        for row in rows:
            grouped.setdefault(int(row[5]), []).append(
                [float(row[6]), float(row[7]), float(row[8])]
            )
        result: list[list[float]] = []
        for count in range(1, 11):
            points = grouped[count]
            result.append(
                [
                    sum(point[axis] for point in points) / len(points)
                    for axis in range(3)
                ]
            )
        return result

    def distances(points: list[list[float]]) -> list[float]:
        return [
            math.sqrt(
                sum((points[i][axis] - points[j][axis]) ** 2 for axis in range(3))
            )
            for i in range(len(points))
            for j in range(i + 1, len(points))
        ]

    left = distances(centroids(left_rows))
    right = distances(centroids(right_rows))
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    return numerator / math.sqrt(left_ss * right_ss)


def build_answer_fit_sensitivity(answer_data: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    correlations: list[float] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        selected = sorted(
            (
                item
                for item in answer_data.values()
                if item["model"] == model
                and item["fit_cohort"] == "all"
                and (item.get("manifold_display") or item.get("probe_optimal"))
            ),
            key=lambda item: int(item["layer"]),
        )
        for all_fit in selected:
            layer = int(all_fit["layer"])
            correct_fit = answer_data[f"{model}|{layer}|correct_only"]
            support = [
                int(value) for value in correct_fit["fit_count_support"].values()
            ]
            missing = [
                str(count)
                for count, value in correct_fit["fit_count_support"].items()
                if int(value) == 0
            ]
            distance_corr = _centroid_distance_correlation(
                all_fit["rows"], correct_fit["rows"]
            )
            correlations.append(distance_corr)
            role = (
                "M · manifold display"
                if all_fit.get("manifold_display")
                else "P · probe optimal"
            )
            rows.append(
                [
                    model,
                    f"L{layer} · {role}",
                    f"{int(all_fit['fit_rows'])} → {int(correct_fit['fit_rows'])}",
                    f"{min(support)}–{max(support)}"
                    + (f"; missing N={','.join(missing)}" if missing else ""),
                    f"{fmt(all_fit['common_v41_variance_capture'][2], 3)} → {fmt(correct_fit['common_v41_variance_capture'][2], 3)}",
                    f"{fmt(all_fit['pca3_discovery_cv_r2'], 3)} → {fmt(correct_fit['pca3_discovery_cv_r2'], 3)}",
                    f"{fmt(all_fit['count_signal_capture_pc1_3'], 3)} → {fmt(correct_fit['count_signal_capture_pc1_3'], 3)}",
                    fmt(distance_corr, 3),
                ]
            )
    return f"""
<div class="fit-sensitivity-block">
<h3>5.1 all-fit 与 correct-only-fit：错误样本是否制造了 geometry？</h3>
<p>两种 basis 都在 V4.1 discovery 上拟合，再投影完全相同的 V4.4 answer-query states。<code>common capture</code> 使用共同的 V4.1 全样本方差分母；最后一列比较相同 V4.4 states 所形成的十个 count centroids 的 45 个两两距离，因此不受 PCA 旋转和轴正负号影响。</p>
{table(["model", "layer/use", "fit n all→correct", "correct per-count support", "common capture PC1–3", "PCA3 CV R²", "count-axis capture", "V4.4 centroid distance corr"], rows)}
<p>Gemma 两层的 centroid-distance correlation 为 0.994–1.000，PCA3 CV R² 只变化约 0.003–0.007；Qwen 为 0.956–0.980，correct-only basis 的 R² 与 count-axis capture 有更明显下降。关键原因是 Qwen correct-only fit 在 N=7、9、10 没有样本，而 Gemma 每个 count 至少仍有 1 个正确样本；所以 correct-only 不是平衡的“更干净主分析”，而是有类别截断的敏感性分析。</p>
<div class="conclusion"><strong>本段结论</strong>四个主层的 V4.4 centroid-distance correlation 均≥{min(correlations):.3f}，因此有序 answer geometry 不是由错误样本凭空制造；但 Qwen 的 correct-only basis 因高 count 缺类而较不稳定。主文继续使用 all-fit，correct-only 只用于确认结论方向。</div>
</div>
"""


def build_causal_design(
    ov: dict[str, Any],
    upstream: dict[str, Any],
    seed_confirmation: dict[str, Any],
    gemma_l37_ov: dict[str, Any],
    gemma_singles: dict[str, dict[str, Any]],
    gemma_read_writes: dict[str, dict[str, Any]],
    gemma_cross_layer: dict[str, Any] | None,
    gemma_residuals: dict[str, dict[str, Any]],
) -> str:
    gemma_rows: list[list[str]] = []
    for label, document in [
        ("L37 H1/H2", gemma_l37_ov),
        *((ov_candidate_label(doc), doc) for doc in gemma_singles.values()),
    ]:
        cfg = document["config"]
        gemma_rows.append(
            [
                "Gemma natural OV",
                f"{label}: natural carrier、true pre-O injection、centered removal、mediation",
                f"{seed_span(cfg['confirmation_seeds'])} confirmation",
                f"direction {seed_span(cfg['direction_discovery_seeds'])}; center {seed_span(cfg['center_seeds'])}",
                f"candidate+matched IUT；global p={fmt_p(document['primary_decision']['global_intersection_union_p'])}",
            ]
        )
    if gemma_cross_layer is not None:
        cfg = gemma_cross_layer["config"]
        gemma_rows.append(
            [
                "Gemma cross-layer K2",
                "joint natural OV + L29 donor patch + exact L35 block",
                f"{seed_span(cfg['confirmation_seeds'])} confirmation",
                f"{len(cfg['mediation_pairs'])} directed pairs × candidate/3 controls",
                f"OV p={fmt_p(gemma_cross_layer['primary_decision']['global_intersection_union_p'])}; relay p={fmt_p(gemma_cross_layer['relay_decision']['intersection_union_p'])}",
            ]
        )
    for residual_name, gemma_residual in gemma_residuals.items():
        cfg = gemma_residual["config"]
        endpoint_count = len(gemma_residual["primary_decision"]["families"])
        gemma_rows.append(
            [
                f"Gemma {residual_name.upper()} residual relay",
                (
                    "clean bank ablation + "
                    if cfg.get("require_clean_necessity", False)
                    else ""
                )
                + "source patch + exact/count-axis blocks + L41 adoption",
                f"{seed_span(cfg['confirmation_seeds'])} confirmation",
                f"{len(cfg['donor_pairs'])} pairs × 5 conditions × candidate/3 controls",
                f"{2 * endpoint_count}-component IUT；global p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}",
            ]
        )
    for label, document in gemma_read_writes.items():
        cfg = document["config"]
        gemma_rows.append(
            [
                "Gemma read/write derivative",
                f"{label}: sliding-window-aware crossed α/V + downstream trace",
                seed_span(cfg["evaluation_seeds"]),
                f"{len(cfg['donor_pairs'])} directed pairs",
                "复用 parent seeds；机制分解，不是独立 replication",
            ]
        )
    seed_rows = [
        [
            "Macro V4.4",
            "ranked-bank ablation",
            "10 confirmation seeds 1254–1263",
            "N=7–10；每模型 40 prompts；ranked vs layer-matched random",
            "先在每 seed 内求差，再跨 seed 推断",
        ],
        [
            "Macro V4.4",
            "needle-end / answer-query patch；steering",
            "10 confirmation seeds 1254–1263",
            "paired nested prompts；层/剂量为重复条件",
            "seed-cluster mean；bootstrap CI；family correction",
        ],
        [
            "causal-v2",
            "baseline + clean-correct patch/ablation supplement",
            "20 discovery + 10 confirmation",
            "N=0–10：220/110 examples 每模型",
            "计划在 discovery 冻结；confirmation 独立评估",
        ],
        [
            "correct-only frozen confirmation",
            "broad-retrieval top-k ablation vs 3 layer-matched random sets",
            "20 fresh seeds 1296–1315",
            "N=1–5；100 examples/model；Qwen K=2/4、Gemma K=1/2 事先冻结",
            f"seed-cluster bootstrap 95% CI；audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS",
        ],
        [
            "Natural OV",
            "carrier、pre-O injection、centered removal、mediation",
            "20 direction + 10 center/control + 20 confirmation",
            "1234–1253；1264–1273；1274–1293",
            f"四证据族 IUT；global p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}",
        ],
        *gemma_rows,
        [
            "Read/write",
            "crossed α/V + downstream write",
            "20 evaluation seeds 1274–1293",
            "六个 directed donor pairs",
            "复用 parent seeds；机制扩展，不算独立 replication",
        ],
        [
            "Relay screen",
            "carrier→edge patch→behavior→OV→removal",
            "20 confirmation seeds 1274–1293",
            "冻结 tail-64 position set",
            "五门 conjunction；任一失败即不支持",
        ],
        [
            "Upstream confirmation",
            "early donor patch + L28 exact block + LOO",
            "20 fresh seeds 1294–1313",
            "六个 directed donor pairs；120 primary rows",
            f"early 与 mediation conjunction；IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}",
        ],
    ]
    method_rows = [
        [
            "Ablation",
            "把 ranked head set 的输出置零，并与同层、同数量随机 head set 比较",
            "这组 heads 对维持生成 count 是否有特异贡献",
        ],
        [
            "State patching",
            "把 donor 的 hidden state / z state 拷到 receiver",
            "某个位置或通道是否足以运输 donor count 信息",
        ],
        [
            "Directional injection",
            "在真实 pre-O z slice 加入 ±β natural count step",
            "该 OV channel 是否具备有符号充分性与 dose response",
        ],
        [
            "Centered removal",
            "只删除自然 count component；与 same-span equal-norm orthogonal removal 比较",
            "模型自然运行是否特异依赖该 component",
        ],
        [
            "Serial mediation",
            "先做 upstream donor patch，再在 L28 阻断自然通道；与正交阻断比较",
            "upstream effect 是否确实经过冻结的 L28 channel",
        ],
        [
            "Crossed α/V",
            "分别替换 routing α 与 value content V，构造 RR/RD/DR/DD",
            "读取来自“看哪里”、还是“读到什么”，或两者都有",
        ],
    ]
    return f"""
<h3>7.1 因果实验总设计：如何把不同方法放在同一证据链中</h3>
<p><strong>先说聚合原则：</strong>这些实验没有被平均成一个“总机制分数”。每种实验检验不同 estimand、量纲也不同；我们先在同一个 seed 内对干预与匹配对照求 paired difference，再把 <strong>seed 作为独立统计单位</strong>。结论通过多种方法的方向一致与 conjunction/IUT 收敛，而不是把 patch accuracy、log-odds、count shift 和几何距离直接相加。</p>
{table(["方法", "具体做什么", "回答的问题"], method_rows)}
{table(["campaign", "主要方法", "独立 seeds", "行/条件规模", "如何折算与判定"], seed_rows)}
<div class="plain-protocol">
<h4>统一统计流程</h4>
<ol>
  <li><strong>行级计算：</strong>对每个 receiver 或 donor→receiver pair，先计算 intervention−control；有方向的 donor test 统一转成“正值=向 donor count 移动”。</li>
  <li><strong>seed 内折算：</strong>同一 seed 的 counts、donor pairs、layers 或 doses 先取平均，避免把同一 haystack 的多行当成独立样本。</li>
  <li><strong>跨 seed 推断：</strong>以 seed means 做 cluster bootstrap 95% CI；exact two-sided sign-flip 检验 seed-level paired effect 是否以 0 为中心。</li>
  <li><strong>多重比较：</strong>同一 family 的 layers、K 或 LOO heads 使用 Holm；natural OV 与 serial path 使用 IUT，global p 取各必要门中最大的 p，只有所有门都过线才通过。</li>
  <li><strong>显著性口径：</strong>本报告统一以校正后或预注册 exact <em>p</em>&lt;0.05 且方向符合假说为“显著”。p 值不是 effect size；效应大小和 95% CI 必须同时报告。</li>
</ol>
</div>
<div class="callout warning"><strong>独立性边界。</strong>Qwen natural-OV confirmation（1274–1293）与 upstream confirmation（1294–1313）不重叠；read/write 与 relay 复用 parent seeds，因此是机制分解而非第二次独立复制。Gemma 每个 evidence-ladder 分支都把 discovery/center 与 confirmation seeds 分开，但后备分支是在前一分支失败后才启动，整个搜索树没有全局 family-wise 校正；Gemma read/write 同样复用各自 parent seeds。correct-only frozen ablation 使用 1296–1315，与 Qwen upstream 的 1296–1313 有重叠，所以不能把这些 p 当作完全独立研究再相乘或 meta-combine。</div>
<div class="conclusion"><strong>本段结论</strong>后续因果证据按“功能定位 → 自然 OV 充分/必要 → α/V 读取分解 → 上游串行 mediation”逐级加严；不同实验互相约束，但不被压成一个不可解释的 pooled number。</div>
"""


def extract_js_json(document: str, name: str) -> dict[str, Any]:
    match = re.search(rf"const {re.escape(name)}=(.*?);\n", document)
    if not match:
        raise RuntimeError(f"Could not find embedded JavaScript object: {name}")
    return json.loads(match.group(1))


def quantile(values: Iterable[float], probability: float) -> float:
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return 0.0
    position = (len(finite) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] + weight * (finite[upper] - finite[lower])


def _mix_rgb(
    left: tuple[int, int, int], right: tuple[int, int, int], weight: float
) -> str:
    weight = min(max(float(weight), 0.0), 1.0)
    channels = [
        round(start + (end - start) * weight)
        for start, end in zip(left, right, strict=True)
    ]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


def cue_attention_color(value: float | None, cap: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "#d8d3ca"
    if cap <= 0:
        return "#e5e0d7"
    scaled = min(max(float(value) / cap, 0.0), 1.0)
    if scaled < 0.55:
        return _mix_rgb((247, 243, 234), (88, 139, 210), scaled / 0.55)
    return _mix_rgb((88, 139, 210), (35, 22, 92), (scaled - 0.55) / 0.45)


def cue_attention_svg(atlas: dict[str, Any], model: str) -> str:
    """Render the V4.4.2 non-thinking cue comparison as a literal head-by-layer table."""
    mode = atlas["models"][model]["modes"]["nonthinking"]
    layers = [int(layer) for layer in mode["layers"]]
    heads = int(mode["heads"])
    present = mode["conditions"]["cue_present"]["layer_head_score"]
    absent = mode["conditions"]["cue_absent"]["layer_head_score"]
    all_values = [
        float(value)
        for matrix in (present, absent)
        for row in matrix
        for value in row
        if value is not None and math.isfinite(float(value))
    ]
    positive_values = [value for value in all_values if value > 0]
    cap = quantile(positive_values, 0.995) if positive_values else 0.0

    cell_width = 12.0
    cell_height = 12.0 if heads >= 16 else 24.0
    plot_width = len(layers) * cell_width
    plot_height = heads * cell_height
    left_margin = 48.0
    right_margin = 12.0
    top_margin = 58.0
    bottom_margin = 88.0
    panel_gap = 64.0
    panel_width = left_margin + plot_width + right_margin
    total_width = 2 * panel_width + panel_gap
    total_height = top_margin + plot_height + bottom_margin
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", model).strip("-")

    parts = [
        f'<svg class="cue-attention-svg" role="img" aria-labelledby="{safe_id}-cue-title {safe_id}-cue-desc" '
        f'viewBox="0 0 {total_width:.1f} {total_height:.1f}" style="display:block;width:100%;height:auto">',
        f'<title id="{safe_id}-cue-title">{html.escape(model)} non-thinking broad-retrieval attention with and without the opening cue</title>',
        f'<desc id="{safe_id}-cue-desc">Two head-by-layer heat maps share one raw broad-retrieval score scale capped at the pooled 99.5th percentile. Layer is horizontal and head is vertical.</desc>',
        "<defs>",
        f'<linearGradient id="{safe_id}-cue-gradient" x1="0" x2="1" y1="0" y2="0">',
        '<stop offset="0%" stop-color="#f7f3ea"/><stop offset="55%" stop-color="#588bd2"/><stop offset="100%" stop-color="#23165c"/>',
        "</linearGradient></defs>",
    ]
    for panel_index, (condition, label, matrix) in enumerate(
        (
            ("cue_present", "有开头提示 · cue-present", present),
            ("cue_absent", "无开头提示 · cue-absent", absent),
        )
    ):
        origin_x = panel_index * (panel_width + panel_gap)
        plot_x = origin_x + left_margin
        parts.append(
            f'<text x="{plot_x + plot_width / 2:.1f}" y="25" text-anchor="middle" '
            f'font-size="16" font-weight="700" fill="#172033">{html.escape(label)}</text>'
        )
        for head in range(heads):
            for layer_index, layer in enumerate(layers):
                raw_value = matrix[layer_index][head]
                value = (
                    float(raw_value)
                    if raw_value is not None and math.isfinite(float(raw_value))
                    else None
                )
                x = plot_x + layer_index * cell_width
                y = top_margin + head * cell_height
                title_value = "N/A" if value is None else f"{value:.6g}"
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width + 0.2:.1f}" '
                    f'height="{cell_height + 0.2:.1f}" fill="{cue_attention_color(value, cap)}">'
                    f"<title>{html.escape(condition)} · L{layer} H{head} · S_broad={title_value}</title></rect>"
                )
        parts.append(
            f'<rect x="{plot_x:.1f}" y="{top_margin:.1f}" width="{plot_width:.1f}" '
            f'height="{plot_height:.1f}" fill="none" stroke="#8e887f" stroke-width="1"/>'
        )
        head_step = 4 if heads > 12 else 1
        for head in range(0, heads, head_step):
            y = top_margin + (head + 0.67) * cell_height
            parts.append(
                f'<text x="{plot_x - 6:.1f}" y="{y:.1f}" text-anchor="end" '
                f'font-size="9" font-family="Consolas,monospace" fill="#626b78">H{head}</text>'
            )
        layer_step = 5
        for layer_index, layer in enumerate(layers):
            if layer_index % layer_step != 0 and layer_index != len(layers) - 1:
                continue
            x = plot_x + (layer_index + 0.5) * cell_width
            y = top_margin + plot_height + 12
            parts.append(
                f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate(55 {x:.1f} {y:.1f})" '
                f'font-size="9" font-family="Consolas,monospace" fill="#626b78">L{layer}</text>'
            )
        parts.append(
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{top_margin + plot_height + 52:.1f}" '
            f'text-anchor="middle" font-size="11" fill="#626b78">layer</text>'
        )
        parts.append(
            f'<text x="{origin_x + 10:.1f}" y="{top_margin + plot_height / 2:.1f}" '
            f'transform="rotate(-90 {origin_x + 10:.1f} {top_margin + plot_height / 2:.1f})" '
            f'text-anchor="middle" font-size="11" fill="#626b78">head</text>'
        )
        if cap <= 0:
            parts.append(
                f'<text x="{plot_x + plot_width / 2:.1f}" y="{top_margin + plot_height / 2:.1f}" '
                f'text-anchor="middle" font-size="15" font-weight="700" fill="#6d665d">'
                "capture mask 内 direct raw-needle score 全为 0</text>"
            )

    legend_width = min(330.0, plot_width)
    legend_x = (total_width - legend_width) / 2
    legend_y = total_height - 22
    parts.extend(
        [
            f'<rect x="{legend_x:.1f}" y="{legend_y:.1f}" width="{legend_width:.1f}" height="9" fill="url(#{safe_id}-cue-gradient)"/>',
            f'<text x="{legend_x:.1f}" y="{legend_y - 4:.1f}" font-size="9" fill="#626b78">0</text>',
            f'<text x="{legend_x + legend_width:.1f}" y="{legend_y - 4:.1f}" text-anchor="end" font-size="9" fill="#626b78">p99.5 cap {cap:.6g}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def cue_attention_summary_rows(atlas: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        mode = atlas["models"][model]["modes"]["nonthinking"]
        present = [
            float(value)
            for layer in mode["conditions"]["cue_present"]["layer_head_score"]
            for value in layer
        ]
        absent = [
            float(value)
            for layer in mode["conditions"]["cue_absent"]["layer_head_score"]
            for value in layer
        ]
        present_norm = math.sqrt(sum(value * value for value in present))
        absent_norm = math.sqrt(sum(value * value for value in absent))
        if present_norm == 0 or absent_norm == 0:
            rows.append(
                [
                    model,
                    "not defined (both maps are zero)",
                    "not defined",
                    "not defined",
                    f"{sum(present):.4f} → {sum(absent):.4f}",
                ]
            )
            continue
        cosine = sum(
            left * right for left, right in zip(present, absent, strict=True)
        ) / (present_norm * absent_norm)
        denominator = 0.5 * (
            sum(abs(value) for value in present) + sum(abs(value) for value in absent)
        )
        relative_l1 = (
            sum(abs(left - right) for left, right in zip(present, absent, strict=True))
            / denominator
        )
        top_k = min(10, len(present))
        present_top = {
            index
            for index, _ in sorted(
                enumerate(present), key=lambda item: item[1], reverse=True
            )[:top_k]
        }
        absent_top = {
            index
            for index, _ in sorted(
                enumerate(absent), key=lambda item: item[1], reverse=True
            )[:top_k]
        }
        rows.append(
            [
                model,
                fmt(cosine, 3),
                fmt(relative_l1, 3),
                f"{len(present_top & absent_top)}/{top_k}",
                f"{sum(present):.4f} → {sum(absent):.4f}",
            ]
        )
    return rows


def find_summary(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    stratum: str = "all",
    layer: int | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("metric") == metric
        and row.get("stratum") == stratum
        and (layer is None or int(row.get("layer")) == int(layer))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one summary row for {metric}/{stratum}/{layer}, got {len(matches)}"
        )
    return matches[0]


def sig_badge(
    p_value: float | None,
    *,
    label: str | None = None,
    alpha: float = 0.05,
) -> str:
    significant = (
        p_value is not None
        and math.isfinite(float(p_value))
        and float(p_value) <= float(alpha)
    )
    klass = "sig-yes" if significant else "sig-no"
    text = label if label is not None else ("显著" if significant else "不显著")
    return f'<span class="{klass}">{html.escape(text)}</span>'


def evidence_badge(
    supported: bool,
    positive: str = "确认",
    negative: str = "未确认",
) -> str:
    klass = "confirmed" if supported else "rejected"
    label = positive if supported else negative
    return f'<span class="evidence {klass}">{html.escape(label)}</span>'


def seed_span(values: Iterable[int]) -> str:
    seeds = [int(value) for value in values]
    if not seeds:
        return "none"
    if len(seeds) == 1:
        return str(seeds[0])
    return f"{min(seeds)}–{max(seeds)}（{len(seeds)} seeds）"


def append_to_section(section_html: str, appendix_html: str) -> str:
    marker = "</section>"
    index = section_html.rfind(marker)
    if index < 0:
        raise RuntimeError("Could not append to generated section")
    return (
        section_html[:index]
        + "\n"
        + appendix_html.strip()
        + "\n"
        + section_html[index:]
    )


def build_scope(
    causal_v2: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    relay: dict[str, Any],
    upstream: dict[str, Any],
    gemma_l37_ov: dict[str, Any],
    gemma_story: dict[str, Any],
) -> str:
    q_baseline = causal_v2["baseline"]["Qwen3-8B"]["confirmation"]
    g_baseline = causal_v2["baseline"]["Gemma4-E4B"]["confirmation"]
    g_supported = bool(gemma_story["support"])
    l37_supported = bool(
        gemma_l37_ov["primary_decision"]["full_natural_ov_transporter_support"]
    )
    claim_rows = [
        [
            "Prompt-side running index",
            "PCA / frozen-basis generalization; cue-present/absent shared-basis audit",
            "两模型均保留有序 occurrence geometry；提示改变 full-space state，但不创造序结构",
            '<span class="evidence descriptive">表征证据</span>',
        ],
        [
            "Distributed broad retrieval",
            "all-layer attention atlas + correct-only frozen top-k ablation",
            "Gemma K=1/K=2 的 clean-correct failure 与 ΔMAE 均过四比较 Holm；Qwen K=4 的 ΔMAE 过 Holm，但 clean-correct failure 仅 pointwise/CI 支持",
            '<span class="evidence functional">bank-level 功能支持</span>',
        ],
        [
            "Late answer count state",
            "answer-query donor patch + norm-matched steering",
            "完整 donor state 高概率运输 donor prediction；count direction 可定向操纵输出",
            '<span class="evidence functional">功能因果</span>',
        ],
        [
            "Qwen L28 natural OV transporter",
            "natural signal + true pre-O injection + centered removal + mediation IUT",
            f"H16/H19 四个证据族全部通过；global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}",
            '<span class="evidence confirmed">确认</span>',
        ],
        [
            "Qwen L28 mixed read/write",
            "crossed α/V decomposition + L28→L35 propagation",
            "routing 与 value/content 都贡献；写入沿冻结 count axes 存活到 L35",
            '<span class="evidence supported">机制扩展</span>',
        ],
        [
            "Qwen early slot-state → L28 → answer",
            "fresh-seed donor patch / exact block / orthogonal-control serial mediation",
            f"独立确认；IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}；H19 为 set 内非冗余成员",
            '<span class="evidence confirmed">独立确认</span>',
        ],
        [
            "Gemma L37 terminal natural OV",
            "与 Qwen 同构的四族 pre-O IUT",
            f"global IUT p={fmt_p(gemma_l37_ov['primary_decision']['global_intersection_union_p'])}",
            evidence_badge(l37_supported, "确认", "否定该候选"),
        ],
        [
            f"Gemma strongest completed path: {html.escape(str(gemma_story['label']))}",
            "冻结 evidence ladder；matched controls；fresh confirmation seeds",
            html.escape(str(gemma_story["summary"])),
            evidence_badge(g_supported, "机制支持", "未闭合"),
        ],
        [
            "Qwen tail-64 terminal relay",
            "registered carrier / edge patch / mediation / removal conjunction",
            f"不支持；global IUT p={fmt_p(relay['primary_decision']['global_intersection_union_p'])}",
            '<span class="evidence rejected">否定该候选</span>',
        ],
    ]
    return f"""
<section id="scope">
<h2>1 · 结论先行：当前最小可辩护机制</h2>
<p class="abstract"><strong>核心结论。</strong>在 non-thinking V4.4 中，模型并不是依赖一个严格单头、单位置的显式整数寄存器。两模型都在 prompt needle-end residual 中形成随 occurrence index 有序变化的分布式 state，并以 broad-retrieval head bank 汇集与计数相关的 slot states；late answer-query state 则携带可执行的 count prediction。prompt counter 与 answer counter 不需要在 residual space 中共线：attention 先在 head-space 形成 <em>z</em>，<em>W</em><sub>O</sub> 再把它写入新的 residual direction，后续 blocks 还可继续旋转/整合。Qwen3-8B 已将这一步闭合为“early broad set → L28 H16/H19 mixed α/V read → natural OV write → L29–L35 answer state”的受限因果链。Gemma 的最强可辩护结果是：{html.escape(str(gemma_story["summary"]))}</p>
<p>证据强度被严格分层：PCA 与 attention map 只定位可解码结构和候选路径；patching、steering 与 frozen top-k ablation 建立功能关系；真实 pre-O injection、centered z-space removal、same-span equal-norm control 和 fresh-seed serial mediation才用于自然机制主张。跨模型比较共享 estimand 与判定规则，不强迫两模型共享层号、head identity 或注意力可见窗口。</p>
{table(["机制命题", "直接检验", "当前结果", "证据等级"], claim_rows)}
<div class="baseline-strip">
  <div><span>Qwen confirmation</span><strong>{100 * q_baseline["accuracy"]:.1f}%</strong><small>accuracy · MAE {q_baseline["mean_absolute_error"]:.3f} · signed error {q_baseline["mean_signed_error"]:.3f}</small></div>
  <div><span>Gemma confirmation</span><strong>{100 * g_baseline["accuracy"]:.1f}%</strong><small>accuracy · MAE {g_baseline["mean_absolute_error"]:.3f} · signed error {g_baseline["mean_signed_error"]:.3f}</small></div>
  <div><span>Fine-grained scope</span><strong>Qwen L28 · {html.escape(str(gemma_story["label"]))}</strong><small>各模型只按实际通过的最强 evidence layer 表述</small></div>
</div>
<div class="conclusion"><strong>本节结论</strong>论文级主张应写成“分布式 prompt representation → broad retrieval → tested write/relay → late answer state”，而不是“某个 head 自己从原始 needle 数数”。Qwen 路径已经闭合；Gemma 的结论停在 <code>{html.escape(str(gemma_story["kind"]))}</code> 层级，不能自动升级为 Qwen 的逐头复制。</div>
</section>
"""


def build_methods(
    ov: dict[str, Any],
    upstream: dict[str, Any],
    gemma_l37_ov: dict[str, Any],
    gemma_singles: dict[str, dict[str, Any]],
    gemma_read_writes: dict[str, dict[str, Any]],
    gemma_cross_layer: dict[str, Any] | None,
    gemma_residuals: dict[str, dict[str, Any]],
) -> str:
    prompt_text = """You will need to count all city-score audit records in the passage below.\nA city-score audit record names one city and gives that city's numeric score.\n\n<passage>\n... approximately 10,000 tokens ...\n</passage>\n\nHow many city-score audit records are in the passage?\nDo not explain, reason aloud, quote, or list any records.\nWrite the count using ordinary decimal digits, with no space after the colon.\nYour entire response must be exactly one line:\nTotal:<integer>"""
    gemma_seed_rows: list[list[str]] = []
    for label, document in [
        ("L37 H1/H2 retained negative", gemma_l37_ov),
        *((ov_candidate_label(doc), doc) for doc in gemma_singles.values()),
    ]:
        candidate_cfg = document["config"]
        gemma_seed_rows.append(
            [
                "Gemma natural OV",
                f"Gemma4-E4B {label}",
                f"{seed_span(candidate_cfg['direction_discovery_seeds'])} direction; {seed_span(candidate_cfg['center_seeds'])} center/control; {seed_span(candidate_cfg['confirmation_seeds'])} confirmation",
                "N=1…10; causal counts 2/5/8",
                f"四证据族 IUT；matched sets；branch α={fmt(float(document['primary_decision']['alpha']), 3)}",
            ]
        )
    if gemma_cross_layer is not None:
        cross_cfg = gemma_cross_layer["config"]
        gemma_seed_rows.append(
            [
                "Gemma cross-layer fallback",
                ov_candidate_label(gemma_cross_layer),
                f"{seed_span(cross_cfg['direction_discovery_seeds'])} direction; {seed_span(cross_cfg['center_seeds'])} center; {seed_span(cross_cfg['confirmation_seeds'])} confirmation",
                "N=1…10; 3 directed pairs",
                "joint four-family natural OV + frozen L29→L35 relay; α=.025",
            ]
        )
    for residual_name, gemma_residual in gemma_residuals.items():
        residual_cfg = gemma_residual["config"]
        clean_text = (
            "; clean zero-z necessity"
            if residual_cfg.get("require_clean_necessity", False)
            else ""
        )
        gemma_seed_rows.append(
            [
                f"Gemma {residual_name.upper()} residual fallback",
                f"{residual_variant_label(gemma_residual)} → L{int(gemma_residual['selected_mediator_layer'])} → L41",
                f"{seed_span(residual_cfg['discovery_seeds'])} layer discovery; {seed_span(residual_cfg['confirmation_seeds'])} confirmation",
                f"N=1…10; 3 directed pairs; 5 path conditions{clean_text}",
                f"layer frozen before confirmation; {len(gemma_residual['primary_decision']['families'])} endpoint families × candidate/matched specificity; α=.025",
            ]
        )
    for label, document in gemma_read_writes.items():
        rw_cfg = document["config"]
        gemma_seed_rows.append(
            [
                "Gemma read/write extension",
                f"{label}: L{int(rw_cfg['mediator_layer'])} "
                + "/".join(f"H{int(head)}" for head in rw_cfg["heads"]),
                seed_span(rw_cfg["evaluation_seeds"]),
                f"{len(rw_cfg['donor_pairs'])} directed donor pairs",
                "复用 parent candidate seeds；derivative decomposition，不算独立确认",
            ]
        )
    seed_rows = [
        [
            "V4.4 representation",
            "Qwen3-8B; Gemma4-E4B",
            "1234–1253 discovery; 1254–1263 confirmation",
            "N=1…10",
            "V4.1 discovery 冻结 PCA/layer，再投影 V4.4",
        ],
        [
            "V4.4.2 cue robustness",
            "Qwen3-8B; Gemma4-E4B",
            "1234–1243",
            "N=1…10; cue present/absent",
            "两提示共享 PCA；seed 为 paired cluster",
        ],
        [
            "V4.4.4 natural OV",
            "Qwen3-8B L28 H16/H19",
            "1234–1253 direction; 1264–1273 center/control; 1274–1293 confirmation",
            "N=1…10; causal counts 2/5/8",
            "四证据族 IUT；matched head sets",
        ],
        *gemma_seed_rows,
        [
            "Read/write extension",
            "Qwen3-8B L28 H16/H19",
            "1264–1273 discovery; 1274–1293 evaluation",
            "six directed donor pairs",
            "复用 parent evaluation seeds；非独立复制",
        ],
        [
            "Upstream confirmation",
            "Qwen3-8B early top-4; L28 H16–H19",
            "1294–1313",
            "six directed donor pairs",
            "route/head set/endpoint/control 全部冻结",
        ],
        [
            "Correct-only low-count routes",
            "Qwen3-8B + Gemma4-E4B",
            "20 fresh seeds/model",
            "counts 1–3; six directed donor pairs",
            "仅纳入 donor/receiver clean 均正确；冻结 source/writer sets",
        ],
    ]
    return f"""
<section id="methods">
<h2>2 · 实验设定、符号与统计口径</h2>
<h3>2.1 V4.4 任务与 prompt</h3>
<p>每个 stimulus 是约 10,000-token 的 realistic haystack，内含十个可控 slot。对同一 seed，N 与 N+1 只在一个 slot 的 active/inactive 内容上变化；V4.4 同时跨 seed 随机化 slot 位置、city-score 内容及其顺序，随机 slot 最小间隔为 256 tokens。non-thinking 条件关闭模型原生 thinking flag，并在 assistant 侧预填 <code>Total:</code>，模型只生成十进制续写。主报告使用带开头定义提示的 frozen V4.4 prompt；V4.4.2 另作 cue-absent 表征敏感性分析。</p>
<pre class="prompt-block"><code>{html.escape(prompt_text)}</code></pre>
<div class="conclusion"><strong>本段结论</strong>V4.4 的 running-index geometry 若能跨 seed 保留，就不能只由固定绝对位置、固定 city identity 或固定内容顺序解释；但它仍可能依赖任务格式或分布式上下文。</div>

<h3>2.2 数据分割与推断单位</h3>
{table(["campaign", "模型/候选", "seeds", "counts / pairs", "冻结规则"], seed_rows)}
<p>所有主要置信区间都以 seed 为独立 cluster 做 bootstrap；符号检验使用 seed-level exact sign flip。自然 OV 的四个必要证据族采用 intersection–union test（IUT）：family p 是该族中最弱组成检验的 p，global p 是四个 family p 的最大值。LOO 的四个 head decrement 使用 Holm 校正。正确/错误分层只作 sensitivity analysis，任何 PCA/count axis 均先冻结，不在分层后重新选择。</p>
<div class="conclusion"><strong>本段结论</strong>确认性结论的独立单位是 seed 而非 token、head 或 donor pair；多重比较与 discovery/confirmation 分割必须在解释效应时一起保留。</div>

<h3>2.3 Representation 定义</h3>
<p>令 <code>h<sup>P</sup><sub>s,n,l</sub></code> 表示 seed <em>s</em> 中第 <em>n</em> 个 active needle 最后 token 经第 <em>l</em> 个 block 后的 residual；这就是 prompt running-index state。令 <code>h<sup>A</sup><sub>s,N,l</sub></code> 表示同一 prompt 最终 <code>Total:</code> query 的 residual；这是 answer count state。主 PCA 在 disjoint V4.1 discovery rows 上拟合后投影 V4.4；因此三维图只负责显示，full-space ridge、η²、CKA 与 causal tests 承担统计推断。</p>
<div class="equation">count-signal capture = ||P<sub>PC1:m</sub> b||² / ||b||², &nbsp; where b is the full-space OLS count direction</div>
<div class="equation">count η² = SS<sub>between count centroids</sub>/SS<sub>total</sub>; &nbsp;&nbsp; linear CKA(X,Y)=||X<sup>T</sup>Y||<sub>F</sub>²/(||X<sup>T</sup>X||<sub>F</sub>·||Y<sup>T</sup>Y||<sub>F</sub>).</div>
<p>这里 <code>X</code> 与 <code>Y</code> 是各自减去 grand centroid 后的 count-centroid matrices。η² 衡量完整 hidden space 中 count bucket 解释的变异比例；CKA 比较两条 centroid trajectory 的 Gram geometry，对共同旋转与各 PCA 轴正负号不敏感。二者都不由屏幕上的 PC1–3 距离直接计算。</p>
<p><strong>all-fit 与 correct-only-fit。</strong>all-fit 是主分析，因为它估计模型在真实运行分布中的 representation；correct-only-fit 只检查错误样本是否扭曲可视化。后者在高 count 可能没有任何正确样本，因此不能作为完整 count manifold 的无偏主 basis。报告中的 fit 切换只改变 basis，不改变被投影的 V4.4 states。</p>
<div class="conclusion"><strong>本段结论</strong>PCA 中出现有序轨迹只证明 count/index 可解码，不能单独证明该坐标被模型读取，更不能证明某个单点 state 是充分因果载体。</div>

<h3>2.4 Attention read、OV write 与 mediation 定义</h3>
<p><strong>Broad-retrieval atlas 的分数。</strong>令最终 <code>Total:</code> query 对第 <em>i</em> 个 needle 的 pooled attention 为 <code>m<sub>i</sub></code>：endpoint 视图只取该 needle 最后一个 token，full-span 视图则对该 needle 的全部 tokens 做 literal sum。定义总 needle mass <code>M=Σ<sub>i</sub>m<sub>i</sub></code>、occurrence profile <code>p<sub>i</sub>=m<sub>i</sub>/M</code>、entropy effective number <code>N<sub>eff,H</sub>=exp(−Σp<sub>i</sub>log p<sub>i</sub>)</code> 与 coverage <code>C<sub>H</sub>=N<sub>eff,H</sub>/N</code>；atlas 的 discovery primary score 为：</p>
<div class="equation">S<sub>broad</sub> = M × C<sub>H</sub>; &nbsp;&nbsp; atlas color = log<sub>10</sub>(S<sub>broad</sub>) within each model/pooling.</div>
<p>因此亮色同时要求“读到较多 needle mass”与“不要只压在一个 occurrence 上”，但仍不是 causal importance。phenotype breadth 另用 participation effective number <code>N<sub>eff,2</sub>=1/Σp<sub>i</sub><sup>2</sup></code>；global-broad 的冻结形状门为 mean <code>N<sub>eff,2</sub>≥6</code> 且任一 occurrence 的 mean normalized share≤0.25，并先要求 needle 对 matched hard negatives 的 enrichment&gt;1。只有 key window 覆盖全部 needles 的 heads 才能进入 global atlas；Gemma 的灰色 local-attention layers 表示该全局 estimand 不可定义，不表示 attention=0。</p>
<p>对 query head <em>h</em>，attention 的 pre-O state 与写回 residual 的输出分别为：</p>
<div class="equation">z<sub>h</sub>(q)=Σ<sub>j</sub> α<sub>h</sub>(q,j)V<sub>g(h)</sub>x<sub>j</sub>, &nbsp;&nbsp; o<sub>h</sub>(q)=W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q).</div>
<p>QK/α 决定读哪里，V 决定读出什么内容，W<sub>O</sub> 决定向 residual 写入什么方向。若 prompt residual 的单位 count direction 是 <code>u<sub>P</sub></code>，则 head set <em>S</em> 的写回为 <code>w<sub>S</sub>=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>z<sub>h</sub></code>；到 answer layer 的局部传播可写成 <code>u<sub>A</sub>∝J<sub>ℓ→A</sub>w<sub>S</sub></code>。因此 count ordering 可以保留而 <code>u<sub>P</sub></code> 与 <code>u<sub>A</sub></code> 不共线；跨位置比较应检验可解码性、transport 与轴特异阻断，不应要求两个 PCA 方向视觉平行。</p>
<div class="equation">w<sub>S</sub>(c)=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q,c), &nbsp;&nbsp; u<sub>A</sub>∝J<sub>ℓ→A</sub>w<sub>S</sub>, &nbsp;&nbsp; generally u<sub>P</sub>∦u<sub>A</sub>.</div>
<p>对 head set <em>S</em>，自然一单位 count step 记为 <code>d<sub>S</sub></code>，其 set-output direction 为 <code>m<sub>S</sub>=W<sub>O</sub><sup>S</sup>d<sub>S</sub></code>。centered removal 从 <code>z<sub>S</sub>−z<sub>0,S</sub></code> 中移除沿 <code>m<sub>S</sub></code> 的自然成分；matched control 位于同一 <code>W<sub>O</sub><sup>S</sup></code> span、具有相同 post-O norm，并与 <code>m<sub>S</sub></code> 正交。</p>
<div class="equation">injection: z<sub>S</sub>←z<sub>S</sub>+βd<sub>S</sub>; &nbsp;&nbsp; u<sub>m</sub>=m<sub>S</sub>/||m<sub>S</sub>||; &nbsp;&nbsp; c<sub>S</sub>=⟨W<sub>O</sub><sup>S</sup>(z<sub>S</sub>−z<sub>0,S</sub>),u<sub>m</sub>⟩; &nbsp;&nbsp; removal: Δz<sub>S</sub>=−c<sub>S</sub>d<sub>S</sub>/||m<sub>S</sub>||.</div>
<p>由此 <code>W<sub>O</sub><sup>S</sup>Δz<sub>S</sub>=−c<sub>S</sub>u<sub>m</sub></code>，所以 removal 真正在 pre-O z-space 中完成，却精确删除 selected-head output span 内的自然 count component；没有把 answer axis 直接注入 residual。<code>z<sub>0,S</sub></code> 只用独立 center/control seeds 估计，避免把静态 offset 当成 count signal。</p>
<div class="equation">G = ([ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>intervention</sub> − [ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>clean</sub>), &nbsp;&nbsp; M = G<sub>orth</sub> − G<sub>natural-block</sub>.</div>
<p><code>G</code> 是 donor-vs-receiver candidate-sequence log-odds gain；<code>M</code> 是自然轴阻断相对 same-span orthogonal control 额外消除的 donor effect。成员分析定义 <code>D<sub>h</sub>=M<sub>full</sub>−M<sub>−h</sub></code>；正值表示移除 head <em>h</em> 后 set mediation 下降。</p>
<div class="equation">Δz<sub>value</sub>=½[(z<sub>RD</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>DR</sub>)], &nbsp; Δz<sub>route</sub>=½[(z<sub>DR</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>RD</sub>)].</div>
<p>其中第一个字母指定 receiver/donor attention routing，第二个字母指定 receiver/donor V content；<code>Δz<sub>full</sub>=Δz<sub>value</sub>+Δz<sub>route</sub></code>。这一分解不要求 QK heads 与 OV heads 是同一集合。</p>
<div class="conclusion"><strong>本节结论</strong>OV 作为 attention block 的写回变换在架构上必然存在；但只有“自然 carrier + true pre-O sufficiency + centered necessity + path mediation”同时成立，才支持模型自然使用某个指定 OV head set。Qwen 达到后者；Gemma 当前只定位到分布式 residual write。</div>
</section>
"""


def build_cue_section(cue_doc: str) -> str:
    nt = extract_js_json(cue_doc, "NT_GEOM")
    prompt = extract_js_json(cue_doc, "PROMPT_GEOM")
    atlas = extract_js_json(cue_doc, "ATLAS")
    rows: list[list[str]] = []
    for payload, site, label in (
        (prompt, "prompt_counter", "Prompt running index"),
        (nt, "answer_query", "Answer query"),
    ):
        for model, landmarks in payload["landmarks"].items():
            for role in ("display", "probe"):
                layer = int(landmarks[role])
                stat = payload["statistics"][f"{model}|{site}|{layer}"]
                rows.append(
                    [
                        model,
                        label,
                        f"L{layer} ({role})",
                        fmt(stat["centroid_cka"], 3),
                        f"{fmt(stat['count_eta_present'], 3)} → {fmt(stat['count_eta_absent'], 3)}",
                        fmt_p(stat["interaction_q"]),
                        fmt_p(stat["count_eta_q"]),
                    ]
                )
    attention_figures = "\n".join(
        f"""
<figure class="cue-attention-figure">
{cue_attention_svg(atlas, model)}
<figcaption><strong>Figure · {html.escape(model)} non-thinking broad-retrieval attention under cue removal.</strong> 横轴为 transformer layer，纵轴为 attention head；每个格子的颜色是最后一个 <code>Total:</code> query 对所有完整 active needle spans 的 <code>S<sub>broad</sub>=M×exp(H(p))/N</code>。左右图共享该模型 pooled raw score 的 p99.5 上限，超过上限的值只在显示时截断；因此同一模型内可以逐格比较有无开头提示，但颜色不能跨模型比较。鼠标悬停可读出 layer、head 与未截断分数。</figcaption>
</figure>
"""
        for model in ("Qwen3-8B", "Gemma4-E4B")
    )
    attention_rows = cue_attention_summary_rows(atlas)
    return f"""
<section id="cue-robustness">
<h2>4 · 开头提示的表征敏感性：拓扑保留不等于逐点不变</h2>
<h3>4.1 Hidden-state geometry</h3>
<p>V4.4.2 在相同 non-thinking flag 下，只删除开头两句 city-score 定义提示；每个 model × site × layer 使用 cue-present 与 cue-absent 的 pooled shared PCA。<code>centroid CKA</code> 比较 count 1–10 的两条 centroid geometry；<code>count×cue q</code> 在原始 full hidden space 检验 cue 是否以 count-dependent 方式改变状态；<code>Δ strength q</code> 检验 full-space count η² 是否改变。后二者均按 layer 做 BH-FDR。</p>
{table(["model", "counter", "layer", "centroid CKA", "count η² present → absent", "count×cue q", "Δ strength q"], rows)}
<p>表中 CKA 均为 0.981–0.997，说明 ordinal path 的整体 pairwise geometry 在删除提示后高度保留；与此同时，所有列出的 count×cue interaction 都显著，说明各 count 的向量并非只做同一个刚体平移。对 prompt counter，Qwen L29 和 Gemma L37/L39 的 count strength 也有显著但量级不同的改变；对 answer query，strength 差异未过 FDR。</p>
<div class="callout warning"><strong>如何解释“图形几乎不变”。</strong>高 CKA 回答的是“count 之间的相对几何是否相似”；interaction 回答的是“cue-induced delta 是否随 count 改变”。两者可以同时成立：提示可改变 gain、局部方向或 role offset，却不破坏 running-index 的排序拓扑。</div>

<h3>4.2 Attention map：同一 broad-retrieval score 的左右对照</h3>
<p>这里不再混用多种横轴或颜色定义。两模型都只显示 non-thinking 的 frozen broad-retrieval score；每个模型恰好两张大表，左边有提示、右边无提示。下表的 cosine、relative L1 与 top-10 overlap 只是 layer×head map 的描述性相似度，不是跨 seed 显著性检验；它们用于量化“亮区是否仍在同一批 heads”，不能替代后面的 frozen-set ablation。</p>
{attention_figures}
{table(["model", "map cosine", "relative L1 change", "top-10 overlap", "total S_broad present → absent"], attention_rows)}
<p>Qwen 的 map cosine 为 0.896，但 relative L1 change 为 0.422，top-10 仅重叠 5/10；也就是说，删除提示没有清空 broad-retrieval bank，却明显重新分配了 bank 内的读权重。Gemma 两图全零不是“所有 attention 都为零”，而是其 local/sliding attention architecture 使最后 answer query 在该 capture mask 下不能直接看见原始远端 needles；因此这个 direct raw-needle score 对 Gemma 是结构性不可用，不能据此否定经中间 residual/relay state 的读取。</p>
<div class="conclusion"><strong>本节结论</strong>开头提示不是 running-index geometry 的生成源；模型会从重复的 record 格式与累积上下文本身形成序结构。但提示仍调制 full-space representation，并在 Qwen 中重分配 broad-retrieval head map，因此不能声称 cue 完全没有机制影响。Gemma 的 direct raw-needle map 则受 attention window 限制；后续因果链路必须直接检验中间 state/relay，而不能把结构性零图误读为没有读取。V4.4.4 因果链路使用 cue-present 主设置，尚未完成逐环节的 cue-absent causal replication。</div>
</section>
"""


def build_causal_v2_intro(
    causal_v2: dict[str, Any], seed_confirmation: dict[str, Any]
) -> str:
    frozen_sets = {
        ("Qwen3-8B", "2"): "L27H18, L28H19",
        ("Qwen3-8B", "4"): "L27H18, L28H19, L23H29, L23H13",
        ("Gemma4-E4B", "1"): "L29H4",
        ("Gemma4-E4B", "2"): "L29H4, L35H2",
    }
    baseline_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for split in ("discovery", "confirmation"):
            row = causal_v2["baseline"][model][split]
            baseline_rows.append(
                [
                    model,
                    split,
                    str(row["examples"]),
                    f"{100 * row['accuracy']:.1f}%",
                    fmt(row["mean_absolute_error"], 3),
                    fmt(row["mean_signed_error"], 3),
                ]
            )
    patch_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for family in ("prompt_patching", "answer_patching"):
            item = causal_v2["correct_interventions"]["patch_pooled"][
                f"{model}::{family}"
            ]
            patch_rows.append(
                [
                    model,
                    "prompt full/multi-token"
                    if family == "prompt_patching"
                    else "answer query",
                    str(item["groups"]),
                    str(item["pair_instances"]),
                    f"{100 * item['pooled_average_patching_acc']:.1f}%",
                    f"{100 * item['group_min_average_patching_acc']:.1f}%–{100 * item['group_max_average_patching_acc']:.1f}%",
                ]
            )
    ablation_rows: list[list[str]] = []
    for key, item in causal_v2["correct_interventions"]["ablation_candidates"].items():
        ablation_rows.append(
            [
                item["model_label"],
                item["analysis_population"],
                f"top-{item['candidate_top_n']}",
                fmt(item["primary_effect"], 4),
                f"[{fmt(item['ci95_low'], 4)}, {fmt(item['ci95_high'], 4)}]",
                "unfrozen n=1…5 discovery",
            ]
        )
    comparison_order = [
        (model, k_text)
        for model in ("Qwen3-8B", "Gemma4-E4B")
        for k_text in sorted(
            seed_confirmation["models"][model], key=lambda value: int(value)
        )
    ]
    companion_ps = [
        float(seed_confirmation["models"][model][k_text]["absolute_error"]["exact_p"])
        for model, k_text in comparison_order
    ]
    companion_holm = dict(
        zip(comparison_order, holm_adjusted_pvalues(companion_ps), strict=True)
    )
    confirmation_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for k_text, metrics in sorted(
            seed_confirmation["models"][model].items(),
            key=lambda item: int(item[0]),
        ):
            all_shift = metrics["all_absolute_shift"]
            correct = metrics["clean_correct_to_wrong"]
            error = metrics["absolute_error"]
            all_pass = float(all_shift["ci95_low"]) > 0
            correct_pass = float(correct["ci95_low"]) > 0
            correct_family_pass = (
                float(correct["holm_p_across_four_frozen_sets"]) <= 0.05
            )
            status = (
                "clean-correct familywise 支持"
                if all_pass and correct_pass and correct_family_pass
                else (
                    "clean-correct pointwise；Holm 未过"
                    if all_pass and correct_pass
                    else "仅 all-example 主 CI；correct-only 未确认"
                )
            )
            confirmation_rows.append(
                [
                    model,
                    f"K={k_text}",
                    frozen_sets[(model, str(k_text))],
                    f"{fmt(all_shift['effect'], 4)} [{fmt(all_shift['ci95_low'], 4)}, {fmt(all_shift['ci95_high'], 4)}]",
                    f'{fmt(correct["effect"], 4)} [{fmt(correct["ci95_low"], 4)}, {fmt(correct["ci95_high"], 4)}]<br><span class="small">p/Holm={fmt_p(correct["two_sided_exact_seed_sign_flip_p"])}/{fmt_p(correct["holm_p_across_four_frozen_sets"])}</span>',
                    f"{fmt(error['effect'], 4)} [{fmt(error['ci95_low'], 4)}, {fmt(error['ci95_high'], 4)}]",
                    f"{fmt_p(error['exact_p'])} / {fmt_p(companion_holm[(model, str(k_text))])}",
                    status,
                ]
            )
    return f"""
<div class="callout evidence-note"><strong>更新后的 audit-grade V4.4 causal-v2。</strong>下方旧 V4.4 图保留原始 panel-restricted onset 与 steering 结果；本段补充重新跑完并通过 302/302 checks per model 的 causal-v2，以及 clean-correct supplement。两批实验的 estimand 不完全相同，数值不应直接合并成一个 meta-effect。</div>
<h3>7.2 Baseline 与 correct-only transport</h3>
{table(["model", "split", "examples", "accuracy", "MAE", "signed error"], baseline_rows)}
<p>causal-v2 在原 N=1…10 nested family 上增加 N=0，共 20 discovery seeds 与 10 confirmation seeds。两模型 confirmation 的 signed error 均为负，说明主要失败模式是 high-count undercount，而不是格式失败（valid rate=1）。</p>
{table(["model", "patch site", "groups", "eligible pair instances", "pooled donor-target adoption", "group range"], patch_rows)}
<div class="equation">donor-target adoption = mean I[patched receiver prediction = donor gold] &nbsp; | &nbsp; donor clean-correct ∧ receiver clean-correct.</div>
<p>clean-correct donor/receiver 条件下，answer-query patch 的 pooled donor-target adoption 在 Qwen/Gemma 分别为 96.6%/96.0%；prompt-side full/multi-token patch 为 81.5%/91.9%。这与“单个 needle endpoint patch 接近零”并不矛盾：前者协调搬运已筛选的 full-span/multi-token state，后者只搬一个 endpoint。</p>
{details_table("Ablation candidate effects（探索性 n 扫描）", ["model", "population", "candidate", "effect", "95% CI", "status"], ablation_rows)}
<p>ablation supplement 在 fresh seeds 上找到正向功能信号，但 top-n 是 n=1…5 同 seed 扫描后选出的候选；因此它支持“ranked attention bank 有可重复功能贡献”，不支持把某个 bank/top-n 写成冻结的独立确认。对大量单独 selected conditions 做 Holm 后没有条件 p≤.05，论文正文应报告 pooled/family-level 结论与这一 multiplicity 边界。</p>
<h3>7.3 冻结 top-k 的独立 seed 外推</h3>
{table(["model", "frozen set size", "frozen heads", "all-example |count shift| [95% CI]", "clean-correct c→w [95% CI]; exact/Holm", "companion ΔMAE [95% CI]", "exact p / Holm p (ΔMAE)", "primary interpretation"], confirmation_rows)}
<figure>{ablation_topk_svg(seed_confirmation)}<figcaption><strong>Figure · Frozen top-k ablation on fresh seeds.</strong> 左图横轴为事先冻结的 head-set size K，纵轴为 ranked−random 的绝对 generated-count shift；右图横轴同样为 K，纵轴为 clean-correct correct-to-wrong rate excess。圆点是 20 个 seed-cluster 的 pooled effect，竖线是 10,000 次 seed-cluster bootstrap 95% CI；CI 跨过 0 表示该主 estimand 未确认。两模型的 K 网格不同，连线只帮助看同模型剂量变化，不假设连续函数，也不用于跨模型比较绝对效应。</figcaption></figure>
<div class="equation">D<sub>abs</sub> = |ŷ<sub>ranked</sub>−ŷ<sub>clean</sub>| − mean<sub>r=1..3</sub>|ŷ<sub>random,r</sub>−ŷ<sub>clean</sub>|; &nbsp;&nbsp; D<sub>cw</sub> = I[ranked wrong] − mean<sub>r</sub>I[random<sub>r</sub> wrong] &nbsp; | &nbsp; clean correct.</div>
<p>这一轮在查看结果前冻结 Qwen K=2/4、Gemma K=1/2；具体前缀为 Qwen K2={frozen_sets[("Qwen3-8B", "2")]}、K4={frozen_sets[("Qwen3-8B", "4")]}，Gemma K1={frozen_sets[("Gemma4-E4B", "1")]}、K2={frozen_sets[("Gemma4-E4B", "2")]}。实验使用 20 个全新 seeds（1296–1315）、count 1–5、每模型 100 个 examples，并对每个 ranked set 配置 3 个 layer-matched random replicates。主 all-example estimand 是 ranked−random 的绝对 generated-count shift；主 clean-correct estimand 是原本答对样本中 ranked−random 的 correct-to-wrong rate。两套注册主分析都用 seed-cluster bootstrap 95% CI 判定；为让“显著性”完全可审计，报告另外对 clean-correct 的 20 个 seed-cluster contributions 做双侧 exact sign flip，并将四个 model×K 作为一个 Holm family。它是 multiplicity sensitivity analysis，不悄悄替换注册的 pointwise bootstrap 判据。</p>
<p>原 causal-v2 helper 在 n&gt;16 时实际切换为 100,000 次 Monte-Carlo sign flip，却保留了 exact 字段名；本报告从保存的 20 个 seed effects 重新枚举全部 2<sup>20</sup>=1,048,576 个符号组合，重算 clean-correct 与 companion ΔMAE 的真正双侧 exact p。Qwen K=2 的 clean-correct CI 跨 0（exact/Holm p=0.5/0.5），不能写成稳定 necessity；Qwen K=4 的 clean-correct excess 为 0.0650 [0.0238, 0.1124]，pointwise exact p=0.03125，但跨四比较 Holm p=0.0625，因此是“点估计与注册 CI 支持、family-wise sensitivity 未过”。它的 all-example ΔMAE=0.0500 [0.0200, 0.0833] 则 exact/Holm p=0.015625/0.03125。Gemma K=1/K=2 的 clean-correct failure excess 为 0.1231 [0.0595, 0.1857] 与 0.1282 [0.0690, 0.1882]，clean-correct exact/Holm p=0.0078125/0.0234375 与 0.0019531/0.0078125；相应 ΔMAE exact/Holm p=0.001595/0.004784 与 0.000944/0.003777。四个冻结比较应作为一个 family 阅读，而不是再从中选择最有利的 K。</p>
<div class="callout warning"><strong>不要混合两个 Qwen early set。</strong>本节 correct-only Qwen K4 是 L27H18/L28H19/L23H29/L23H13，用来确认 clean-run bank-level ablation；第 10.2 节 fresh-seed serial source 则冻结自更早 V4.4.2 路径筛选，为 L23H28/L23H29/L26H20/L27H18。二者共享部分成员但不是同一 set，不能把 clean necessity 与 serial mediation 逐头拼接。Gemma 第 10.4 节则有意直接复用本节 K2=L29H4/L35H2，以检验这一冻结 bank 是否接到 L37。</div>
<div class="conclusion"><strong>本段结论</strong>完整 count state 可由 prompt full/multi-token representation 与 late answer query 搬运。冻结 broad-retrieval ablation 提供 bank-level 功能必要性证据：Qwen K=4 的 all-example harm 通过四比较 Holm，clean-correct 仅达到注册 CI/pointwise exact、未过该附加 Holm sensitivity；Gemma K=1/K=2 的 clean-correct 与 ΔMAE 均通过 Holm。它仍定位到 ranked bank 而非唯一 head；更细的自然读写通路由后续 pre-O removal 与 serial mediation 决定。</div>
"""


def build_natural_ov_section(ov: dict[str, Any]) -> str:
    families = ov["primary_decision"]["families"]

    def component(family: str, endpoint: str) -> dict[str, Any]:
        hits = [
            item
            for item in families[family]["components"]
            if item["endpoint"] == endpoint
        ]
        if len(hits) != 1:
            raise RuntimeError(f"Missing OV component {family}/{endpoint}")
        return hits[0]

    natural = component("natural_signal", "natural_carrier_count_slope")
    injection = component("pre_o_injection", "injection_dose_slope")
    removal_error = component("centered_removal", "removal_error_axis_minus_control")
    removal_margin = component("centered_removal", "removal_margin_axis_minus_control")
    donor = component("path_mediation", "donor_patch_transport")
    mediation = component("path_mediation", "mediation_control_minus_axis_block")
    metric_rows = [
        [
            "1 · 自然信号",
            "clean forward 中测 H16/H19 centered z-output 对 count 的斜率",
            "斜率≤0",
            ci_text(natural),
            fmt_p(natural["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "2 · pre-O 充分性",
            "只在真实 pre-O z slice 加 β·d；让 heads 自身 W_O 写出",
            "dose slope≤0",
            ci_text(injection),
            fmt_p(injection["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "3a · centered 必要性",
            "删自然轴后增加的 |error|，减去 same-span 等范数正交删除",
            "额外误差≤0",
            ci_text(removal_error),
            fmt_p(removal_error["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "3b · centered 必要性",
            "同一 removal 对 correct-count margin 的额外影响",
            "margin 下降≥0",
            ci_text(removal_margin),
            fmt_p(removal_margin["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "4a · 路径前提",
            "把 donor z state patch 到 receiver，测 donor-count transport",
            "transport≤0",
            ci_text(donor),
            fmt_p(donor["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "4b · 路径 mediation",
            "正交 block 保留的 donor effect − 自然轴 block 保留的 effect",
            "specificity≤0",
            ci_text(mediation),
            fmt_p(mediation["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
    ]
    gate = evidence_gate_svg(
        [
            {
                "title": "Natural signal",
                "main": f"carrier slope {ci_text(natural)}",
                "sub": "candidate also exceeds matched-set mean",
                "p": f"family IUT p = {fmt_p(families['natural_signal']['intersection_union_p'])}",
            },
            {
                "title": "True pre-O sufficiency",
                "main": f"dose slope {ci_text(injection)}",
                "sub": "V-path z injection; no answer-axis injection",
                "p": f"family IUT p = {fmt_p(families['pre_o_injection']['intersection_union_p'])}",
            },
            {
                "title": "Centered necessity",
                "main": f"extra |error| {ci_text(removal_error)}",
                "sub": f"correct margin {ci_text(removal_margin)}",
                "p": f"family IUT p = {fmt_p(families['centered_removal']['intersection_union_p'])}",
            },
            {
                "title": "Path mediation",
                "main": f"donor transport {ci_text(donor)}",
                "sub": f"specific block {ci_text(mediation)}",
                "p": f"family IUT p = {fmt_p(families['path_mediation']['intersection_union_p'])}",
            },
        ]
    )
    nested_rows = [
        [
            str(row["k"]),
            ",".join(str(h) for h in row["heads"]),
            fmt_p(row["families"]["natural_signal"]["holm_p_across_k"]),
            fmt_p(row["families"]["pre_o_injection"]["holm_p_across_k"]),
            fmt_p(row["families"]["centered_removal"]["holm_p_across_k"]),
        ]
        for row in ov["nested_k"]
    ]
    return f"""
<section id="natural-ov">
<h2>8 · Qwen L28 natural OV transporter：从“可以推动”到“模型自然使用”</h2>
<h3>8.1 要验证的具体假说</h3>
<div class="test-card"><h4>假说：L28 H16/H19 是自然使用的 count-to-answer OV transporter</h4><dl>
<dt>候选从哪里来</dt><dd>在 discovery 数据上冻结 Qwen L28 H16/H19；confirmation 使用 seeds 1274–1293。</dd>
<dt>这里的 OV 是什么</dt><dd>head 先得到 pre-O state <code>z</code>，再由该 head 自己的 <code>W<sub>O</sub></code> 写回 residual。干预发生在 z-space，不把答案方向直接加到 residual。</dd>
<dt>不要求什么</dt><dd>不要求 H16/H19 自己从原始 needle 做 QK 定位；earlier heads 可以先构造或汇集 count-bearing source state。</dd>
<dt>通过标准</dt><dd>自然信号、pre-O 充分性、centered 必要性、path mediation 四门都必须显著；global IUT p 是四个 family 中最大的 p。</dd>
</dl></div>
<p>四个 K=2 matched control sets（H28/H31、H20/H23、H0/H3、H8/H11）在不查看 causal outcome 时，按 natural-step norm、answer-axis cosine、W<sub>O</sub>-span reachability 与 baseline output norm 匹配。这样，“删掉任意同范数方向都会伤模型”不能冒充自然 count channel 的特异作用。</p>
<div class="conclusion"><strong>本段结论</strong>本节验证的是“模型是否自然使用这个下游写入通道”，不是“同一组 heads 是否完成从原始 needle 定位到答案的全部工作”。</div>

<h3>8.2 四步实验分别做了什么</h3>
<figure>{gate}<figcaption><strong>Figure · Natural-OV evidence gates.</strong> 这是证据流程图而非坐标图，因此没有数值坐标轴。每个框给出一个预先规定的必要证据族、seed-cluster mean 与 95% bootstrap CI；family IUT p 取该族最弱组成检验。四框同时通过才判定 natural transporter；global IUT p={fmt_p(ov["primary_decision"]["global_intersection_union_p"])}。</figcaption></figure>
{table(["步骤", "具体操作", "零假设/失败边界", "effect [95% CI]", "exact p", "p<0.05?"], metric_rows)}
<div class="step-result"><strong>如何把六行折成一个确认结论。</strong>每个 family 内还要求 candidate 优于 matched-set mean；family IUT p 分别为 natural={fmt_p(families["natural_signal"]["intersection_union_p"])}、injection={fmt_p(families["pre_o_injection"]["intersection_union_p"])}、removal={fmt_p(families["centered_removal"]["intersection_union_p"])}、mediation={fmt_p(families["path_mediation"]["intersection_union_p"])}。global IUT 取最大值 {fmt_p(ov["primary_decision"]["global_intersection_union_p"])}，小于 0.05，故四门联合结论显著。</div>
<p><strong>逐步解释：</strong>步骤 1 排除“这个 span 完全没有自然 count signal”；步骤 2 排除“它只能在 post-O 被人工 steering”；步骤 3 排除“它只是一个可达但自然运行不需要的方向”；步骤 4 排除“它虽重要，却不介导 donor state transport”。mediation specificity 0.0136 相当于约 18.2% 的 donor-z transport，因此作用真实但只是部分路径。</p>
<div class="conclusion"><strong>本段结论</strong>H16/H19 不只是一个可操纵子空间：自然信号、真实 pre-O 充分性、matched-control 必要性和路径 mediation 同时成立（global IUT p={fmt_p(ov["primary_decision"]["global_intersection_union_p"])}）。这支持“自然使用的部分 OV transporter”，不支持“完整 count circuit 已全部找到”。</div>

<h3>8.3 Set size 与成员边界</h3>
{details_table("Nested-K secondary analysis", ["K", "heads", "natural Holm p", "injection Holm p", "removal Holm p"], nested_rows)}
<p>K=2/3/4/6/8 的 natural signal 与 injection 均通过 Holm；centered removal 只有 K=2 和 K=4 通过。扩大 K 同时扩大可干预 span，并没有“越多头越显著”的单调模式。这里显著性的统一阈值仍是 Holm p&lt;0.05。H16/H19 的 injection 近似可加，旧 factorial analysis 未确认超加性 synergy。</p>
<div class="conclusion"><strong>本节结论</strong>最稳健的 natural-OV 主张来自冻结的 K=2 H16/H19 matched-set test；更大 set 只作为二级稳健性结果，不能据此宣称更大的 head bank 更真实。</div>
</section>
"""


def ov_candidate_label(ov: dict[str, Any]) -> str:
    cfg = ov["config"]
    if cfg.get("candidate_sites"):
        return " + ".join(
            f"L{int(layer)}H{int(head)}" for layer, head in cfg["candidate_sites"]
        )
    return f"L{int(cfg['layer'])} " + "/".join(
        f"H{int(head)}" for head in cfg["candidate_heads"]
    )


def residual_variant_label(residual: dict[str, Any]) -> str:
    cfg = residual["config"]
    variant = str(cfg.get("mechanism_variant", "k2")).upper()
    return f"{variant} {{{ov_candidate_label(residual)}}}"


def ov_matched_set_labels(ov: dict[str, Any]) -> list[str]:
    cfg = ov["config"]
    if cfg.get("matched_control_sets"):
        return [
            " + ".join(f"L{int(layer)}H{int(head)}" for layer, head in site_set)
            for site_set in cfg["matched_control_sets"]
        ]
    labels = []
    for set_id, role in sorted(
        {(str(row["set_id"]), str(row["set_role"])) for row in ov["summary"]}
    ):
        if role != "matched_control":
            continue
        match = re.search(r"matched_control_L(\d+)_H(\d+(?:_H\d+)*)$", set_id)
        if match:
            layer = int(match.group(1))
            heads = "/".join(f"H{int(value)}" for value in match.group(2).split("_H"))
            labels.append(f"L{layer} {heads}")
        else:
            labels.append(set_id)
    return labels


def resolve_gemma_story(
    *,
    l37: dict[str, Any],
    singles: dict[str, dict[str, Any]],
    read_writes: dict[str, dict[str, Any]],
    cross_layer: dict[str, Any] | None,
    residuals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Choose the strongest *completed* Gemma mechanism without hiding failures.

    The order is part of the registered evidence ladder: a localized natural-OV
    result is stronger than a cross-layer set, which is stronger than a
    distributed residual-mediation result.  Read/write extensions are attached
    only to their exact parent candidate and never upgrade a failed parent.
    """
    supported_single = next(
        (
            (name, doc)
            for name, doc in singles.items()
            if bool(
                doc.get("primary_decision", {}).get(
                    "full_natural_ov_transporter_support", False
                )
            )
        ),
        None,
    )
    if supported_single is not None:
        name, doc = supported_single
        rw = read_writes.get(name)
        return {
            "kind": "single",
            "label": ov_candidate_label(doc),
            "natural": doc,
            "read_write": rw,
            "support": True,
            "global_p": float(doc["primary_decision"]["global_intersection_union_p"]),
            "alpha": float(doc["primary_decision"]["alpha"]),
            "summary": (
                f"{ov_candidate_label(doc)} 在独立 confirmation seeds 上通过四族 "
                "natural-OV IUT；其 α/V 分解只在对应 extension 完成后解释。"
            ),
        }
    if cross_layer is not None and bool(
        cross_layer.get("full_cross_layer_mechanism_support", False)
    ):
        return {
            "kind": "cross_layer",
            "label": ov_candidate_label(cross_layer),
            "natural": cross_layer,
            "read_write": None,
            "support": True,
            "global_p": max(
                float(cross_layer["primary_decision"]["global_intersection_union_p"]),
                float(cross_layer["relay_decision"]["intersection_union_p"]),
            ),
            "alpha": float(cross_layer["primary_decision"]["alpha"]),
            "summary": (
                "冻结的 L29H4+L35H2 跨层 set 同时通过 joint natural-OV "
                "与 L29→L35 relay 两道门。"
            ),
        }
    supported_residual = next(
        (
            (name, document)
            for name, document in residuals.items()
            if bool(
                document.get("primary_decision", {}).get(
                    "full_residual_count_path_support", False
                )
            )
        ),
        None,
    )
    if supported_residual is not None:
        residual_name, residual = supported_residual
        layer = int(residual["selected_mediator_layer"])
        clean_clause = (
            "、clean-run bank necessity"
            if residual["config"].get("require_clean_necessity", False)
            else ""
        )
        return {
            "kind": "residual",
            "residual_variant": residual_name,
            "label": (f"{residual_variant_label(residual)} → L{layer} residual → L41"),
            "natural": None,
            "read_write": None,
            "support": True,
            "global_p": float(
                residual["primary_decision"]["global_intersection_union_p"]
            ),
            "alpha": float(residual["primary_decision"]["alpha"]),
            "summary": (
                f"冻结 {residual_variant_label(residual)} 对 L{layer} count-aligned "
                f"residual 的写入{clean_clause}、精确阻断、count-axis 阻断与 "
                "L41 adoption 同时通过；该结论不定位唯一 downstream head。"
            ),
        }
    completed_local = [
        "L37 H1/H2",
        *(ov_candidate_label(doc) for doc in singles.values()),
    ]
    completed_fallbacks = []
    if cross_layer is not None:
        completed_fallbacks.append("cross-layer set")
    completed_fallbacks.extend(f"{name} residual path" for name in residuals)
    pending_fallbacks = [
        label
        for label, document in (
            ("cross-layer set", cross_layer),
            ("K2 residual path", residuals.get("k2")),
            ("K6 residual contingency", residuals.get("k6")),
        )
        if document is None
    ]
    completed_text = "、".join(completed_local + completed_fallbacks)
    pending_text = (
        "；尚未完成或未触发的后备分支为 " + "、".join(pending_fallbacks)
        if pending_fallbacks
        else ""
    )
    return {
        "kind": "partial",
        "label": "Gemma distributed counting evidence",
        "natural": None,
        "read_write": None,
        "support": False,
        "global_p": None,
        "alpha": 0.025,
        "summary": (
            f"已完成的冻结分支（{completed_text}）没有闭合完整机制 gate{pending_text}。"
            "当前只保留 independently confirmed ablation、prompt/answer patching 与各单项"
            "正效应，不把尚未完成的分支写成负结果，也不写成完整 head-level circuit。"
        ),
    }


def build_gemma_natural_ov_appendix(
    ov: dict[str, Any],
    *,
    heading: str = "8.4",
    context_label: str = "冻结 L37 假说",
) -> str:
    families = ov["primary_decision"]["families"]

    def component(family: str, endpoint: str) -> dict[str, Any]:
        hits = [
            item
            for item in families[family]["components"]
            if item["endpoint"] == endpoint
        ]
        if len(hits) != 1:
            raise RuntimeError(f"Missing Gemma OV component {family}/{endpoint}")
        return hits[0]

    natural = component("natural_signal", "natural_carrier_count_slope")
    injection = component("pre_o_injection", "injection_dose_slope")
    removal_error = component("centered_removal", "removal_error_axis_minus_control")
    removal_margin = component("centered_removal", "removal_margin_axis_minus_control")
    donor = component("path_mediation", "donor_patch_transport")
    mediation = component("path_mediation", "mediation_control_minus_axis_block")
    supported = bool(ov["primary_decision"]["full_natural_ov_transporter_support"])
    global_p = float(ov["primary_decision"]["global_intersection_union_p"])
    cfg = ov["config"]
    candidate_label = ov_candidate_label(ov)
    matched_labels = ov_matched_set_labels(ov)
    alpha = float(ov["primary_decision"]["alpha"])
    family_order = [
        "natural_signal",
        "pre_o_injection",
        "centered_removal",
        "path_mediation",
    ]
    failed = [name for name in family_order if not bool(families[name]["passes_alpha"])]
    gate_id = "gemma-gate-" + re.sub(r"[^a-z0-9]+", "-", candidate_label.lower()).strip(
        "-"
    )
    gate = evidence_gate_svg(
        [
            {
                "title": "Natural signal",
                "main": f"carrier slope {ci_text(natural)}",
                "sub": f"candidate must also exceed {len(matched_labels)} matched sets",
                "p": f"family IUT p = {fmt_p(families['natural_signal']['intersection_union_p'])}",
                "passed": families["natural_signal"]["passes_alpha"],
            },
            {
                "title": "True pre-O sufficiency",
                "main": f"dose slope {ci_text(injection)}",
                "sub": "real Gemma V projection + value normalization",
                "p": f"family IUT p = {fmt_p(families['pre_o_injection']['intersection_union_p'])}",
                "passed": families["pre_o_injection"]["passes_alpha"],
            },
            {
                "title": "Centered necessity",
                "main": f"extra |error| {ci_text(removal_error)}",
                "sub": f"correct margin {ci_text(removal_margin)}",
                "p": f"family IUT p = {fmt_p(families['centered_removal']['intersection_union_p'])}",
                "passed": families["centered_removal"]["passes_alpha"],
            },
            {
                "title": "Path mediation",
                "main": f"donor transport {ci_text(donor)}",
                "sub": f"specific block {ci_text(mediation)}",
                "p": f"family IUT p = {fmt_p(families['path_mediation']['intersection_union_p'])}",
                "passed": families["path_mediation"]["passes_alpha"],
            },
        ],
        id_prefix=gate_id,
    )
    metric_rows = [
        [
            "natural carrier",
            ci_text(natural),
            fmt_p(natural["p"]),
            sig_badge(natural["p"], alpha=alpha),
        ],
        [
            "pre-O dose response",
            ci_text(injection),
            fmt_p(injection["p"]),
            sig_badge(injection["p"], alpha=alpha),
        ],
        [
            "removal: extra |error|",
            ci_text(removal_error),
            fmt_p(removal_error["p"]),
            sig_badge(removal_error["p"], alpha=alpha),
        ],
        [
            "removal: correct-margin effect",
            ci_text(removal_margin),
            fmt_p(removal_margin["p"]),
            sig_badge(removal_margin["p"], alpha=alpha),
        ],
        [
            "donor-z transport",
            ci_text(donor),
            fmt_p(donor["p"]),
            sig_badge(donor["p"], alpha=alpha),
        ],
        [
            "mediation specificity",
            ci_text(mediation),
            fmt_p(mediation["p"]),
            sig_badge(mediation["p"], alpha=alpha),
        ],
    ]
    nested_rows = [
        [
            str(row["k"]),
            ",".join(str(head) for head in row["heads"]),
            fmt_p(row["families"]["natural_signal"]["holm_p_across_k"]),
            fmt_p(row["families"]["pre_o_injection"]["holm_p_across_k"]),
            fmt_p(row["families"]["centered_removal"]["holm_p_across_k"]),
        ]
        for row in ov.get("nested_k", [])
    ]
    conclusion = (
        f"Gemma {candidate_label} 通过四族联合标准，可称为本实验确认的自然 OV transporter；这使 terminal natural-write 结论获得跨模型支持。"
        if supported
        else f"Gemma {candidate_label} 没有通过四族联合标准；因此不能把局部正效应升级为完整 natural transporter。失败的必要证据族为："
        + "、".join(failed)
        + "。"
    )
    return f"""
<h3>{heading} Gemma natural-OV 检验：{candidate_label}（{html.escape(context_label)}）</h3>
<p>Gemma 使用与 Qwen 同一四族判定逻辑，但不复用 Qwen 的线性 value 近似：候选 set 为事先冻结的 {candidate_label}；direction seeds 为 {seed_span(cfg["direction_discovery_seeds"])}，center/control seeds 为 {seed_span(cfg["center_seeds"])}，confirmation seeds 为 {seed_span(cfg["confirmation_seeds"])}。{len(matched_labels)} 个未看本轮 causal outcome 即冻结的 matched sets 为 {html.escape("、".join(matched_labels))}；主候选必须分别通过方向检验并优于 matched-set mean，因此单个 endpoint 的 p&lt;{fmt(alpha, 3)} 仍不足以让整个 family 通过。</p>
<div class="conclusion"><strong>设计结论</strong>selection status 为 <code>{html.escape(str(cfg.get("selection_status", "frozen_before_confirmation")))}</code>。本轮在独立 confirmation seeds 上检验冻结 set；Qwen 与 Gemma 共享因果定义和判定门，不强迫两者共享层号或 head identity。</div>
<figure>{gate}<figcaption><strong>Figure · Gemma natural-OV evidence gates.</strong> 四个框依次对应自然载荷、真实 pre-O 充分性、centered z-space 必要性与 donor-path mediation；绿色勾表示整个 family（含 matched-set superiority）通过，粉色叉表示失败。框内 effect 是 seed mean [95% bootstrap CI]，family IUT p 是该门最弱必要检验；全局判定取四门中最大的 p={fmt_p(global_p)}。</figcaption></figure>
{table(["endpoint", "effect [95% CI]", "directional p", f"endpoint p<{fmt(alpha, 3)}?"], metric_rows)}
<p>联合判定不对六个 endpoint 做简单多数票，也不把不同单位的 effect 相加。每个 family 先对 candidate direction 与 matched controls 做 intersection–union test；再令 global IUT p 等于四个 family p 的最大值。最终 <code>full_natural_ov_transporter_support={str(supported).lower()}</code>，global IUT p={fmt_p(global_p)}。</p>
<div class="conclusion"><strong>结果结论</strong>{conclusion}</div>
{details_table("Gemma nested-K secondary analysis", ["K", "heads", "natural Holm p", "injection Holm p", "removal Holm p"], nested_rows) if nested_rows else ""}
<p class="small">Nested-K 只用于 set-size 稳健性分析；Holm 校正后结果不能反过来替换预先冻结的 K=2 主检验，也不能据此声称单个 head 是完整 counter。</p>
<div class="conclusion"><strong>本小节边界</strong>即使四族全部通过，允许的表述仍是“{candidate_label} 构成自然使用的部分 transporter”；它不证明该 set 直接从原始 needle 读取，也不排除其他并行写入通道。</div>
"""


def build_gemma_evidence_ladder(
    *,
    l37: dict[str, Any],
    singles: dict[str, dict[str, Any]],
    cross_layer: dict[str, Any] | None,
    residuals: dict[str, dict[str, Any]],
    story: dict[str, Any],
) -> str:
    rows: list[list[str]] = []

    def natural_row(label: str, doc: dict[str, Any], role: str) -> None:
        decision = doc["primary_decision"]
        supported = bool(decision["full_natural_ov_transporter_support"])
        alpha = float(decision["alpha"])
        rows.append(
            [
                label,
                role,
                f"global IUT p={fmt_p(decision['global_intersection_union_p'])}; α={fmt(alpha, 3)}",
                evidence_badge(supported, "通过", "保留负结果"),
            ]
        )

    natural_row("L37 H1/H2", l37, "最初冻结的 terminal-OV 假说")
    for name, label in (("l29h4", "L29H4"), ("l35h2", "L35H2")):
        if name in singles:
            natural_row(label, singles[name], "independent ablation-ranked single")
        else:
            rows.append(
                [
                    label,
                    "gated single-head fallback",
                    "前一单头已通过，按序贯规则未运行",
                    '<span class="evidence qualified">预设跳过</span>',
                ]
            )
    if cross_layer is not None:
        cross_p = max(
            float(cross_layer["primary_decision"]["global_intersection_union_p"]),
            float(cross_layer["relay_decision"]["intersection_union_p"]),
        )
        rows.append(
            [
                "L29H4 + L35H2",
                "cross-layer joint OV + L29→L35 relay",
                f"joint max-IUT p={fmt_p(cross_p)}; α={fmt(float(cross_layer['primary_decision']['alpha']), 3)}",
                evidence_badge(
                    bool(cross_layer["full_cross_layer_mechanism_support"]),
                    "通过",
                    "保留负结果",
                ),
            ]
        )
    else:
        rows.append(
            [
                "L29H4 + L35H2",
                "cross-layer fallback",
                "单头已通过，按序贯规则未运行",
                '<span class="evidence qualified">预设跳过</span>',
            ]
        )
    for residual_name, residual in residuals.items():
        decision = residual["primary_decision"]
        layer = int(residual["selected_mediator_layer"])
        rows.append(
            [
                f"{residual_variant_label(residual)} → L{layer} residual → L41",
                f"{residual_name.upper()} distributed residual mediation",
                f"global IUT p={fmt_p(decision['global_intersection_union_p'])}; α={fmt(float(decision['alpha']), 3)}",
                evidence_badge(
                    bool(decision["full_residual_count_path_support"]),
                    "通过",
                    "保留负结果",
                ),
            ]
        )
    if "k2" not in residuals:
        rows.append(
            [
                "K2 bank → residual → L41",
                "registered residual fallback",
                "较强分支已通过，按序贯规则未运行",
                '<span class="evidence qualified">预设跳过</span>',
            ]
        )
    if "k6" not in residuals:
        k2_passed = bool(
            residuals.get("k2", {})
            .get("primary_decision", {})
            .get("full_residual_count_path_support", False)
        )
        rows.append(
            [
                "K6 bank → residual → L41",
                "last registered exploratory contingency",
                (
                    "K2 residual conjunction 已通过，按冻结序贯规则停止；K6 未运行"
                    if k2_passed
                    else "只有 K2 residual 完整失败后才触发；当前未运行"
                ),
                '<span class="evidence qualified">预设跳过</span>',
            ]
        )
    return f"""
<h3>8.4 Gemma 证据阶梯：先冻结、后揭示、失败不删除</h3>
<p>Gemma 没有被要求复刻 Qwen 的 layer/head identity。顺序固定为：最初 L37 terminal set → independent-ablation 排名得到的 L29H4 → 条件式 L35H2 → 跨层 K2 → K2 residual mediation → K6 residual contingency。后一个分支只有在前一个完整 conjunction 失败时才启动；因此它是透明的 mechanism search，而不是在同一批 confirmation outcomes 上反复换定义。</p>
{table(["候选", "检验层级", "联合统计", "判定"], rows)}
<div class="callout warning"><strong>多重性边界。</strong>每个分支内部都要求 candidate core 与 matched-control superiority 同时成立，并在独立 confirmation seeds 上用 IUT；fallback 分支把阈值收紧到 α=0.025。整个跨分支搜索树没有再做一个全局 family-wise 校正，所以 Gemma 的最终机制应标为“顺序探索后独立 seed 确认”，不能写成一次性预注册的唯一候选验证。</div>
<div class="conclusion"><strong>本段结论</strong>{html.escape(str(story["summary"]))} 最强允许层级是 <code>{html.escape(str(story["kind"]))}</code>；所有更强但失败的定位仍在下文逐项展示。</div>
"""


def build_gemma_cross_layer_appendix(cross: dict[str, Any]) -> str:
    relay = cross["relay_decision"]
    components = relay["components"]
    forest_rows = [
        {
            "label": (
                "L29 donor gain"
                if item["endpoint"].startswith("l29_donor_gain")
                else "L35 exact-block mediation"
            )
            + (
                " · candidate−control"
                if item["role"] == "candidate_specificity"
                else " · candidate"
            ),
            "mean": item["mean"],
            "low": item["ci95_low"],
            "high": item["ci95_high"],
            "value": f"{ci_text(item)} · p={fmt_p(item['p'])}",
        }
        for item in components
    ]
    figure = forest_svg(
        forest_rows,
        title="Gemma L29 to L35 frozen cross-layer relay",
        description=(
            "Candidate and candidate-minus-matched-control effects are shown for "
            "the early L29 donor gain and the L35 exact-block mediation effect."
        ),
        x_label="normalized donor-count transport (positive supports the registered relay)",
    )
    alpha = float(cross["primary_decision"]["alpha"])
    supported = bool(cross["full_cross_layer_mechanism_support"])
    return f"""
<h3>10.3 Gemma 跨层 relay：L29 output 是否经 L35 传到 answer</h3>
<p><strong>操作定义。</strong>先把 donor 的 L29H4 answer-query pre-O <code>z</code> patch 到 receiver，测量 donor-count transport；随后在 L35H2 精确删除由该 L29 patch 诱发的自然 <code>z</code> 增量，并与同一 <code>W<sub>O</sub></code> span、相同 post-O norm、但与该增量正交的 block 比较。前者检验 L29 是否能推动 state，后者检验这部分推动是否确实经过 L35，而非仅仅在别处平行传播。</p>
<div class="equation">relay mediation = transport(L29 patch + orthogonal L35 block) − transport(L29 patch + exact induced-Δz<sub>L35</sub> block).</div>
<figure>{figure}<figcaption><strong>Figure · Gemma cross-layer relay.</strong> 横轴是归一化 donor-count transport；0 表示无 donor shift 或 exact block 不比正交 block 多消除 transport。点是 {len(cross["config"]["confirmation_seeds"])} 个 confirmation seeds 的均值，横线是 seed-cluster bootstrap 95% CI；每个 endpoint 分 candidate core 与 candidate−3 matched-set mean 两行。</figcaption></figure>
<p>relay IUT p={fmt_p(relay["intersection_union_p"])}，阈值 α={fmt(alpha, 3)}；joint natural-OV global IUT p={fmt_p(cross["primary_decision"]["global_intersection_union_p"])}。完整跨层机制要求两者都通过，因此 <code>full_cross_layer_mechanism_support={str(supported).lower()}</code>。</p>
<div class="conclusion"><strong>本段结论</strong>{"冻结 K2 同时满足 joint OV 与 L29→L35 relay，支持一条局部跨层 transporter。" if supported else "至少一个必要门失败；不能把 L29H4 与 L35H2 串成自然跨层 transporter，即使某个单项 effect 为正。"}</div>
"""


def build_gemma_residual_appendix(residual: dict[str, Any]) -> str:
    summary = residual["summary"]
    decision = residual["primary_decision"]
    cfg = residual["config"]
    selected = int(residual["selected_mediator_layer"])
    alpha = float(decision["alpha"])
    bank_label = residual_variant_label(residual)
    labels = {
        "clean_correct_failure_rate": "clean-correct failure-rate increase",
        "clean_delta_absolute_error": "clean expected-count absolute-error increase",
        "source_donor_transport": "source-bank donor transport",
        "exact_residual_mediation": "exact induced-Δ residual mediation",
        "count_axis_mediation": "frozen count-axis mediation",
        "terminal_count_adoption": "L41 terminal count adoption",
    }
    # The table retains the full estimand names; the figure uses compact labels
    # so its left margin remains readable at ordinary laptop viewport widths.
    plot_labels = {
        "source_donor_transport": "source transport",
        "exact_residual_mediation": "induced-Δ mediation",
        "count_axis_mediation": "count-axis mediation",
        "terminal_count_adoption": "L41 adoption",
    }
    clean_endpoints = (
        (
            (
                "clean_correct_failure_rate",
                "failure-rate increase under zero-z ablation",
            ),
            ("clean_delta_absolute_error", "Δ expected-count absolute error"),
        )
        if cfg.get("require_clean_necessity", False)
        else ()
    )
    path_endpoints = (
        "source_donor_transport",
        "exact_residual_mediation",
        "count_axis_mediation",
        "terminal_count_adoption",
    )
    forest_rows = []
    table_rows = []
    endpoint_rows: dict[str, list[dict[str, Any]]] = {}
    for endpoint in (*[item[0] for item in clean_endpoints], *path_endpoints):
        endpoint_rows[endpoint] = []
        for role in ("candidate_core", "candidate_specificity"):
            hits = [
                row
                for row in summary
                if row["endpoint"] == endpoint and row["set_role"] == role
            ]
            if len(hits) != 1:
                raise RuntimeError(f"Missing residual summary {endpoint}/{role}")
            row = hits[0]
            suffix = (
                "candidate" if role == "candidate_core" else "candidate−control mean"
            )
            label = f"{plot_labels.get(endpoint, labels[endpoint])} · {suffix}"
            p_value = float(row["one_sided_exact_sign_flip_p"])
            plotted = {
                "label": label,
                "mean": row["mean"],
                "low": row["ci95_low"],
                "high": row["ci95_high"],
                "value": f"{ci_text(row)} · p={fmt_p(p_value)}",
            }
            endpoint_rows[endpoint].append(plotted)
            if endpoint in path_endpoints:
                forest_rows.append(plotted)
            table_rows.append(
                [
                    labels[endpoint],
                    suffix,
                    ci_text(row),
                    fmt_p(p_value),
                    sig_badge(p_value, alpha=alpha),
                ]
            )
    figure = forest_svg(
        forest_rows,
        title=f"Gemma {bank_label} distributed residual mediation",
        description=(
            "Four registered endpoints are shown for the candidate source bank "
            "and its difference from the mean of three matched source banks."
        ),
        x_label="normalized donor-count effect (positive supports the registered path)",
        width=1260,
        left=390,
        right=320,
    )
    clean_figures = "".join(
        "<figure>"
        + forest_svg(
            endpoint_rows[endpoint],
            title=f"Gemma {bank_label} clean necessity: {labels[endpoint]}",
            description=(
                "Candidate source-bank damage and candidate-minus-matched-control "
                "damage on held-out clean runs."
            ),
            x_label=x_label,
            width=1260,
        )
        + (
            f"<figcaption><strong>Figure · {labels[endpoint]}.</strong> 横轴单位为"
            f" {html.escape(x_label)}；点是 confirmation-seed mean，横线是 95% "
            "seed-cluster bootstrap CI。candidate 与 candidate−3 matched-set mean "
            "两行都必须为正。</figcaption></figure>"
        )
        for endpoint, x_label in clean_endpoints
    )
    clean_block = (
        '<div class="test-card"><h4>K6 clean-run natural necessity</h4>'
        "<p>在每个正确 baseline prompt 的 answer query，把冻结 K6 bank 的 pre-O "
        "z slices 置零；与三组同层组成、同 set size 的 K6 controls 比较。第一个 "
        "estimand 只在 baseline clean-correct cases 中计算转错率，第二个在全部 counts "
        "中计算 expected-count absolute error 增量。该门防止把 donor-patch sufficiency "
        "误写成模型自然使用。</p></div>" + clean_figures
        if clean_endpoints
        else ""
    )
    discovery_rows = sorted(
        residual["discovery_layer_scores"], key=lambda row: int(row["layer"])
    )
    discovery_table = [
        [
            f"L{int(row['layer'])}",
            fmt(row["mean_aligned_induced_norm"], 4),
            fmt(row["positive_fraction"], 3),
            str(int(row["samples"])),
            '<span class="evidence confirmed">selected</span>'
            if int(row["layer"]) == selected
            else "",
        ]
        for row in discovery_rows
    ]
    supported = bool(decision["full_residual_count_path_support"])
    return f"""
<h3>10.4 Gemma 分布式 residual relay：不强迫唯一 downstream head</h3>
<p>当前定义把 independently frozen 的 source bank <code>{html.escape(bank_label)}</code> 当作一个整体。在 discovery seeds 上，只用 source patch 引发的 residual change，从 L36–L40 选择 mean count-aligned induced change 最大的一个边界；选定 L{selected} 后锁定该 layer、count axis、matched banks 与所有 endpoints，再对不重叠的 {seed_span(cfg["confirmation_seeds"])} confirmation seeds 评估。</p>
{table(["discovery layer", "mean aligned induced Δ", "positive fraction", "samples", "selection"], discovery_table)}
{clean_block}
<div class="test-card"><h4>五个 forward conditions 如何折成四个 endpoints</h4><dl>
<dt>source patch</dt><dd>把 donor 的 {html.escape(bank_label)} pre-O states 写到 receiver，测 donor transport。</dd>
<dt>exact block / exact orthogonal</dt><dd>在 L{selected} 删除这次 source patch 实际诱发的 residual Δ；对照删除等范数正交方向。二者差为 exact mediation。</dd>
<dt>count-axis block / count-axis orthogonal</dt><dd>删除 discovery-frozen natural count-axis 分量；对照删除等范数 axis-orthogonal 分量。二者差为 count-axis mediation。</dd>
<dt>L41 adoption</dt><dd>source patch 后 L41 residual 在 frozen count step 上是否朝 donor count 移动。</dd>
</dl></div>
<figure>{figure}<figcaption><strong>Figure · Gemma distributed residual-path gates.</strong> 横轴统一为正向 donor-count effect；0 是相应零假设。每个 endpoint 的第一行是冻结 source bank 本身，第二行是它减去 3 个 layer-matched banks 的 seed-wise mean；点为 {len(cfg["confirmation_seeds"])} 个 confirmation seed means，横线为 95% bootstrap CI。四个 path endpoints 的八行必须同时 CI&gt;0 且 exact sign-flip p≤{fmt(alpha, 3)}；若上方存在 clean-necessity 图，它们也属于同一全局 conjunction。</figcaption></figure>
{table(["endpoint", "contrast", "effect [95% CI]", "one-sided exact p", f"p≤{fmt(alpha, 3)}?"], table_rows)}
<p>global IUT p={fmt_p(decision["global_intersection_union_p"])}，取 {2 * len(decision["families"])} 个必要组成检验中最大的 p；<code>full_residual_count_path_support={str(supported).lower()}</code>。</p>
<div class="callout warning"><strong>解释边界。</strong>该实验可证明 frozen source bank 的 causal effect 经过一个 count-aligned residual channel 到达 L41；它不定位唯一 downstream attention head，也不排除 MLP 或其他 heads 在 L{selected} 前后共同实现 relay。它比 localized natural-OV transporter 是更弱、但仍可反驳的机制主张。</div>
<div class="conclusion"><strong>本段结论</strong>{"冻结 " + bank_label + "→L" + str(selected) + " residual→L41 的完整 conjunction 通过，支持一条分布式 counting relay。" if supported else "至少一个必要 endpoint 或 matched-specificity 门失败；该 residual 边界不能被写成完整 causal relay。"}</div>
"""


def build_read_write_section(read_write: dict[str, Any]) -> str:
    summary = read_write["summary"]
    read_metrics = [
        find_summary(summary, "read_value_behavior_transport"),
        find_summary(summary, "read_routing_behavior_transport"),
        find_summary(summary, "read_full_behavior_transport"),
        find_summary(summary, "read_value_minus_routing_transport"),
    ]
    read_labels = [
        "V/content component",
        "α/routing component",
        "full donor-z patch",
        "value − routing",
    ]
    read_rows = []
    for label, row in zip(read_labels, read_metrics):
        read_rows.append(
            {
                "label": label,
                "mean": row["mean"],
                "low": row["ci95_low"],
                "high": row["ci95_high"],
                "value": f"{ci_text(row)} · p={fmt_p(row['exact_sign_flip_p'])}",
            }
        )
    read_forest = forest_svg(
        read_rows,
        title="Factorized read contributions at Qwen L28 H16/H19",
        description="Horizontal axis is normalized donor behavioral transport. Value and routing components are both positive and indistinguishable in magnitude; the full patch is larger.",
        x_label="normalized donor behavioral transport (positive = donor count gains probability)",
    )
    write_rows = sorted(
        [
            row
            for row in summary
            if row.get("metric") == "write_residual_specificity"
            and row.get("stratum") == "all"
        ],
        key=lambda row: int(row["layer"]),
    )
    write_svg = write_trace_svg(write_rows)
    mediation_rows = []
    for metric, label in (
        ("read_value_ov_mediation_specificity", "V/content component"),
        ("read_routing_ov_mediation_specificity", "α/routing component"),
    ):
        row = find_summary(summary, metric)
        mediation_rows.append(
            [
                label,
                ci_text(row),
                fmt_p(row["exact_sign_flip_p"]),
                f"{100 * row['positive_seed_fraction']:.0f}%",
            ]
        )
    value_transport, routing_transport, full_transport, value_minus_routing = (
        read_metrics
    )
    routing_mediation = find_summary(summary, "read_routing_ov_mediation_specificity")
    value_mediation = find_summary(summary, "read_value_ov_mediation_specificity")
    read_test_rows = [
        [
            "1 · routing-only",
            "用 donor α、receiver V（DR−RR，并在 donor-V background 复算）",
            ci_text(routing_transport),
            fmt_p(routing_transport["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "2 · value-only",
            "用 receiver α、donor V（RD−RR，并在 donor-α background 复算）",
            ci_text(value_transport),
            fmt_p(value_transport["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "3 · full donor",
            "同时换成 donor α 与 donor V（DD−RR）",
            ci_text(full_transport),
            fmt_p(full_transport["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "4 · value−routing",
            "直接比较两部分的 transport 大小",
            ci_text(value_minus_routing),
            fmt_p(value_minus_routing["exact_sign_flip_p"]),
            '<span class="sig-no">不显著</span>',
        ],
        [
            "5a · routing 经 OV",
            "自然轴 block 比正交 block 额外削弱 routing-only effect",
            ci_text(routing_mediation),
            fmt_p(routing_mediation["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "5b · value 经 OV",
            "自然轴 block 比正交 block 额外削弱 value-only effect",
            ci_text(value_mediation),
            fmt_p(value_mediation["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
    ]
    write_behavior = find_summary(summary, "write_behavior_specificity")
    write_table_rows = [
        [
            f"L{int(row['layer'])}",
            ci_text(row),
            fmt_p(row["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ]
        for row in write_rows
    ]
    write_table_rows.append(
        [
            "answer distribution",
            ci_text(write_behavior),
            fmt_p(write_behavior["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ]
    )
    return f"""
<section id="read-write">
<h2>9 · State 如何被读出并写回：mixed α/V read 与下游传播</h2>
<h3>9.1 第一步：把“看哪里”和“读到什么”拆开</h3>
<div class="test-card"><h4>Crossed α/V intervention 的四个 endpoint</h4><dl>
<dt>RR</dt><dd>receiver 的 attention routing α + receiver 的 value content V；这是 reference。</dd>
<dt>RD</dt><dd>receiver α + donor V；只改变“读到什么”。</dd>
<dt>DR</dt><dd>donor α + receiver V；只改变“看哪里/以什么权重读”。</dd>
<dt>DD</dt><dd>donor α + donor V；完整 donor pre-O endpoint。</dd>
</dl></div>
<p>如果 routing-only 能推动 donor count，说明选择 source 的 α 参与读取；如果 value-only 能推动，说明 source residual 中已经有可用 content。只有 component 的行为效应还会被第 8 节冻结的 natural OV axis 特异阻断，才把它视作当前自然通路的一部分。</p>
<figure>{read_forest}<figcaption><strong>Figure · L28 read decomposition.</strong> 横轴是 normalized donor behavioral transport；0 表示 component 没有把 answer distribution 推向 donor count，正值表示 donor count 获得概率。点为 20 个 evaluation seed 的 paired mean，横线为 seed bootstrap 95% CI。前三行是 component/full transport；最后一行是 value−routing 差，因此其区间跨 0 表示两者量级无法区分。</figcaption></figure>
{table(["检验", "具体替换/阻断", "effect [95% CI]", "exact p", "p<0.05?"], read_test_rows)}
<div class="step-result"><strong>显著性判读。</strong>routing=0.0517、value=0.0524、full=0.1140 的 exact p 都是 9.54×10<sup>−7</sup>，均显著；value−routing=0.0008 [−0.0105, 0.0129], p=0.451，不显著。因此证据支持“两部分都有”，但不能说哪一部分更大。routing 与 value 的 natural-OV mediation p 分别为 9.54×10<sup>−7</sup> 和 5.63×10<sup>−5</sup>，也都显著。</div>
<div class="conclusion"><strong>本段结论</strong>L28 H16/H19 采用 mixed read：既依赖 α 决定从哪些 source states 取信息，也依赖那些 source states 的 V content。当前精度下两者大小无显著差异（p=0.451）。</div>

<h3>9.2 第二步：检查 L28 写入是否一路存活到输出</h3>
<p><strong>具体操作：</strong>在 L28 真实 pre-O 边界施加 +β 与 −β natural z-step，经过 H16/H19 自身 W<sub>O</sub> 后继续正常 forward；matched control 位于相同 W<sub>O</sub> span、post-O norm 相同但与自然方向正交。每一层的 coefficient 是中心差分 residual change 在 discovery-frozen answer-count step <code>s<sub>l</sub></code> 上的归一化投影。</p>
<div class="equation">coefficient<sub>l</sub> = ⟨[h<sub>l</sub>(+β)−h<sub>l</sub>(−β)]/(2β), s<sub>l</sub>⟩ / ||s<sub>l</sub>||².</div>
<figure>{write_svg}<figcaption><strong>Figure · Downstream survival of the L28 OV write.</strong> 横轴为 decoder layer L28–L35；纵轴是 natural intervention coefficient 减 same-span orthogonal-control coefficient，单位为该层自然 answer-count step。点为 seed mean，竖线为 95% CI，0 表示 natural 与 orthogonal directions 的传播相同。所有 layer 的 Holm p≤2.29×10<sup>−5</sup>。</figcaption></figure>
{table(["readout site", "natural−orth specificity [95% CI]", "exact p", "p<0.05?"], write_table_rows)}
<p><strong>零假设：</strong>natural 与同 span 的正交方向传播相同，即 specificity=0。L28–L35 每层的区间都在 0 以上；layer family 校正后的最大 p 出现在 L35，为 2.29×10<sup>−5</sup>，仍小于 0.05。answer distribution specificity=0.0685 [0.0478, 0.0912], p=9.54×10<sup>−7</sup>，也显著。</p>
<div class="callout warning"><strong>证据边界。</strong>read/write extension 复用了 parent V4.4.4 的 evaluation seeds；它是冻结候选后的机制扩展，但不是全新 seed 的独立复制。axes 在 outcome 分层前冻结，因此 correct/wrong sensitivity 不会通过重新拟合改变 geometry。</div>
<div class="conclusion"><strong>本节结论</strong>当前可定位的 terminal chain 是：输入 L28 的 state → H16/H19 mixed α/V read → natural OV write → L29–L35 count-aligned residual → count distribution。这里已经验证“如何读、如何写”；上游是谁把可读 state 送到 L28，由第 10 节单独检验。</div>
</section>
"""


def build_gemma_read_write_appendix(
    read_write: dict[str, Any],
    natural_ov: dict[str, Any],
    *,
    heading: str = "9.3",
    natural_heading: str = "8.4",
) -> str:
    summary = read_write["summary"]
    cfg = read_write["config"]
    mediator_layer = int(cfg["mediator_layer"])
    heads = tuple(int(head) for head in cfg["heads"])
    candidate_label = f"L{mediator_layer} " + "/".join(f"H{head}" for head in heads)
    if candidate_label != ov_candidate_label(natural_ov):
        raise RuntimeError(
            f"Gemma read/write parent mismatch: {candidate_label} vs "
            f"{ov_candidate_label(natural_ov)}"
        )
    alpha = float(cfg.get("primary_alpha", 0.05))
    routing = find_summary(summary, "read_routing_behavior_transport")
    value = find_summary(summary, "read_value_behavior_transport")
    full = find_summary(summary, "read_full_behavior_transport")
    difference = find_summary(summary, "read_value_minus_routing_transport")
    routing_mediation = find_summary(summary, "read_routing_ov_mediation_specificity")
    value_mediation = find_summary(summary, "read_value_ov_mediation_specificity")
    write_behavior = find_summary(summary, "write_behavior_specificity")
    write_rows = sorted(
        [
            row
            for row in summary
            if row.get("metric") == "write_residual_specificity"
            and row.get("stratum") == "all"
        ],
        key=lambda row: int(row["layer"]),
    )
    read_forest = forest_svg(
        [
            {
                "label": "α/routing component",
                "mean": routing["mean"],
                "low": routing["ci95_low"],
                "high": routing["ci95_high"],
                "value": f"{ci_text(routing)} · p={fmt_p(routing['exact_sign_flip_p'])}",
            },
            {
                "label": "V/content component",
                "mean": value["mean"],
                "low": value["ci95_low"],
                "high": value["ci95_high"],
                "value": f"{ci_text(value)} · p={fmt_p(value['exact_sign_flip_p'])}",
            },
            {
                "label": "full donor-z patch",
                "mean": full["mean"],
                "low": full["ci95_low"],
                "high": full["ci95_high"],
                "value": f"{ci_text(full)} · p={fmt_p(full['exact_sign_flip_p'])}",
            },
            {
                "label": "value − routing",
                "mean": difference["mean"],
                "low": difference["ci95_low"],
                "high": difference["ci95_high"],
                "value": f"{ci_text(difference)} · p={fmt_p(difference['exact_sign_flip_p'])}",
            },
        ],
        title=f"Factorized read contributions at Gemma {candidate_label}",
        description="Routing, value, full-patch and value-minus-routing effects on the frozen Gemma evaluation seeds.",
        x_label="normalized donor behavioral transport (positive = donor count gains probability)",
    )
    write_svg = write_trace_svg(
        write_rows,
        id_prefix="gemma-write-"
        + re.sub(r"[^a-z0-9]+", "-", candidate_label.lower()).strip("-"),
        title=f"Gemma {candidate_label} natural OV write propagation",
        description="Layer is on the horizontal axis. Natural-minus-orthogonal count-axis coefficient is on the vertical axis. Points are seed means and bars are 95 percent bootstrap confidence intervals.",
    )

    def positive_badge(row: dict[str, Any], p_key: str) -> str:
        supported = float(row["mean"]) > 0 and float(row[p_key]) < alpha
        return sig_badge(
            0.0 if supported else 1.0, label="支持" if supported else "不支持"
        )

    read_rows = [
        [
            "routing-only",
            "donor α + receiver V",
            ci_text(routing),
            fmt_p(routing["exact_sign_flip_p"]),
            positive_badge(routing, "exact_sign_flip_p"),
        ],
        [
            "value-only",
            "receiver α + donor V",
            ci_text(value),
            fmt_p(value["exact_sign_flip_p"]),
            positive_badge(value, "exact_sign_flip_p"),
        ],
        [
            "full donor",
            "donor α + donor V",
            ci_text(full),
            fmt_p(full["exact_sign_flip_p"]),
            positive_badge(full, "exact_sign_flip_p"),
        ],
        [
            "value−routing",
            "两种 component 量级之差",
            ci_text(difference),
            fmt_p(difference["exact_sign_flip_p"]),
            sig_badge(difference["exact_sign_flip_p"], alpha=alpha),
        ],
        [
            "routing 经 natural OV",
            "orthogonal block − natural-axis block",
            ci_text(routing_mediation),
            fmt_p(routing_mediation["exact_sign_flip_p"]),
            positive_badge(routing_mediation, "exact_sign_flip_p"),
        ],
        [
            "value 经 natural OV",
            "orthogonal block − natural-axis block",
            ci_text(value_mediation),
            fmt_p(value_mediation["exact_sign_flip_p"]),
            positive_badge(value_mediation, "exact_sign_flip_p"),
        ],
    ]
    write_table_rows = [
        [
            f"L{int(row['layer'])}",
            ci_text(row),
            fmt_p(row["exact_sign_flip_p"]),
            fmt_p(row.get("holm_p_within_family_metric")),
            positive_badge(row, "holm_p_within_family_metric"),
        ]
        for row in write_rows
    ]
    write_table_rows.append(
        [
            "answer distribution",
            ci_text(write_behavior),
            fmt_p(write_behavior["exact_sign_flip_p"]),
            "N/A",
            positive_badge(write_behavior, "exact_sign_flip_p"),
        ]
    )
    decision = read_write["primary_decision"]
    read_mode = decision["read_mode"]
    write_decision = decision["write_propagation"]
    full_supported = bool(decision["serial_read_write_supported"])
    natural_supported = bool(
        natural_ov["primary_decision"]["full_natural_ov_transporter_support"]
    )
    read_description = {
        "mixed": "routing 与 value/content 两部分都通过其 transport+OV-mediation family，属于 mixed read",
        "routing_only": "只确认 routing component",
        "value_only": "只确认 value/content component",
        "none": "未确认 routing 或 value component",
    }.get(str(read_mode["classification"]), str(read_mode["classification"]))
    if full_supported and natural_supported:
        result_conclusion = "Gemma 复现了 terminal mixed-read → natural-OV-write → downstream count-aligned state 这一段路径。"
    elif full_supported:
        result_conclusion = (
            f"Gemma 的候选 {candidate_label} channel 通过了 factorized read 与 intervention-induced downstream propagation；"
            "但其 parent natural-OV 四族检验未通过，所以这里只能称为候选轴的机械 read/write coherence，"
            "不能升级为模型 clean forward 中自然使用的 read/write replication。"
        )
    else:
        result_conclusion = f"Gemma 没有通过完整 read/write 联合判定：read classification={read_mode['classification']}，write supported={str(bool(write_decision['supported'])).lower()}。"
    return f"""
<h3>{heading} Gemma {candidate_label}：可访问 state 的 α/V 读取与写入传播</h3>
<p>Gemma 的 α/V 分解复用 natural-OV confirmation seeds {seed_span(cfg["evaluation_seeds"])}，因此是冻结候选后的机制分解，不是第二次独立复制。计算构造 RR、RD、DR、DD 四个 pre-O endpoint；value content 必须经过 Gemma 自己的 V projection 与 value normalization，不能套用 Qwen 的线性 <em>W</em><sub>V</sub> 近似。</p>
<div class="callout warning"><strong>可见窗口定义。</strong>{candidate_label} 的 <code>all_positions</code> 定义为该层实际 capture 到的全部 keys；slot、early non-slot、tail non-slot 和 query-self 都先与该窗口取交集，完全落在窗外的组记为 0，而不是当作缺失。因而本实验检验的是“该 set 如何读取当前可访问 state”，<strong>不检验也不声称它直接注意原始 needles</strong>。</div>
<figure>{read_forest}<figcaption><strong>Figure · Gemma {candidate_label} factorized read.</strong> 横轴是 normalized donor behavioral transport；正值表示答案分布向 donor count 移动。点为 {len(cfg["evaluation_seeds"])} 个 seed means，横线为 seed-cluster bootstrap 95% CI。最后一行是 value−routing 差值，而不是第三种 transport component。</figcaption></figure>
{table(["检验", "替换", "effect [95% CI]", "exact p", f"p<{fmt(alpha, 3)}?"], read_rows)}
<p>联合 read classification 为 <code>{html.escape(str(read_mode["classification"]))}</code>：{read_description}。component 是否属于自然通路，不只看 behavioral transport，还要求相应 effect 被第 {html.escape(natural_heading)} 节冻结的 natural OV axis 特异阻断。</p>
<div class="conclusion"><strong>读取结论</strong>{read_description}；value−routing 的差异是否显著必须按其 own p={fmt_p(difference["exact_sign_flip_p"])} 判断，不能仅凭两个点估计的高低排序。</div>
<figure>{write_svg}<figcaption><strong>Figure · Gemma L{mediator_layer}→L{int(write_rows[-1]["layer"])} write propagation.</strong> 横轴是 decoder layer；纵轴是在 frozen layer-specific answer-count step 上，natural pre-O intervention coefficient 减 same-span equal-post-O-norm orthogonal control coefficient。点为 seed mean，竖线为 95% CI，0 表示 natural 与正交方向传播相同。</figcaption></figure>
{table(["readout site", "natural−orth specificity [95% CI]", "exact p", "Holm p", f"校正后 p<{fmt(alpha, 3)}?"], write_table_rows)}
<p>write family 的最终层为 L{int(write_decision["final_layer"])}：specificity={fmt(write_decision["final_residual_specificity_mean"], 4)}，Holm p={fmt_p(write_decision["final_residual_specificity_holm_p"])}；answer-distribution specificity={fmt(write_decision["behavior_specificity_mean"], 4)}，p={fmt_p(write_decision["behavior_specificity_p"])}。完整 read/write 判定为 <code>serial_read_write_supported={str(full_supported).lower()}</code>。</p>
<div class="conclusion"><strong>本小节结论</strong>{result_conclusion}无论结果正负，它都只描述 {candidate_label} 可访问的 state 与后续传播；其上游来源必须由独立 serial/relay intervention 判定。</div>
"""


def build_upstream_section(relay: dict[str, Any], upstream: dict[str, Any]) -> str:
    metric_by_name = {row["metric"]: row for row in relay["metric_summary"]}
    relay_svg = relay_gate_svg(
        [
            {
                "label": "carrier",
                "value": fmt(metric_by_name["natural_relay_slope"]["mean"], 4),
                "p": f"p={fmt_p(metric_by_name['natural_relay_slope']['exact_sign_flip_p'])}",
                "passed": True,
            },
            {
                "label": "V-only first stage",
                "value": fmt(
                    metric_by_name["edge_patch_first_stage_transport"]["mean"], 4
                ),
                "p": f"p={fmt_p(metric_by_name['edge_patch_first_stage_transport']['exact_sign_flip_p'])}",
                "passed": True,
            },
            {
                "label": "behavior",
                "value": fmt(
                    metric_by_name["edge_patch_behavior_transport"]["mean"], 4
                ),
                "p": f"p={fmt_p(metric_by_name['edge_patch_behavior_transport']['exact_sign_flip_p'])}",
                "passed": False,
            },
            {
                "label": "OV mediation",
                "value": fmt(metric_by_name["ov_mediation_specificity"]["mean"], 4),
                "p": f"p={fmt_p(metric_by_name['ov_mediation_specificity']['exact_sign_flip_p'])}",
                "passed": False,
            },
            {
                "label": "natural removal",
                "value": "wrong direction",
                "p": "family p=0.9981",
                "passed": False,
            },
        ]
    )
    relay_table_rows = [
        [
            "1 · carrier",
            "clean forward 中 tail-64 natural contribution 对 count 的斜率",
            ci_text(metric_by_name["natural_relay_slope"]),
            fmt_p(metric_by_name["natural_relay_slope"]["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "2 · V-only first stage",
            "固定 receiver Q/K/α，只 patch tail-64 value content",
            ci_text(metric_by_name["edge_patch_first_stage_transport"]),
            fmt_p(
                metric_by_name["edge_patch_first_stage_transport"]["exact_sign_flip_p"]
            ),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "3 · answer behavior",
            "检查 V-only patch 是否把最终分布推向 donor count",
            ci_text(metric_by_name["edge_patch_behavior_transport"]),
            fmt_p(metric_by_name["edge_patch_behavior_transport"]["exact_sign_flip_p"]),
            '<span class="sig-no">不显著</span>',
        ],
        [
            "4 · L28 OV mediation",
            "自然 L28 block 是否比正交 block 多消除 patch effect",
            ci_text(metric_by_name["ov_mediation_specificity"]),
            fmt_p(metric_by_name["ov_mediation_specificity"]["exact_sign_flip_p"]),
            '<span class="sig-no">不显著</span>',
        ],
        [
            "5a · removal error",
            "删除 tail-64 natural axis 是否比正交删除增加更多误差",
            ci_text(metric_by_name["relay_removal_error_specificity"]),
            fmt_p(
                metric_by_name["relay_removal_error_specificity"]["exact_sign_flip_p"]
            ),
            '<span class="sig-no">反方向</span>',
        ],
        [
            "5b · removal margin",
            "删除 tail-64 natural axis 是否比正交删除降低更多正确 margin",
            ci_text(metric_by_name["relay_removal_margin_specificity"]),
            fmt_p(
                metric_by_name["relay_removal_margin_specificity"]["exact_sign_flip_p"]
            ),
            '<span class="sig-no">反方向</span>',
        ],
    ]
    primary = upstream["primary_decision"]
    early = primary["early_effect"]
    mediation = primary["mediation"]
    path_forest = forest_svg(
        [
            {
                "label": "early top-4 slot-state patch",
                "mean": early["mean"],
                "low": early["ci_low"],
                "high": early["ci_high"],
                "value": f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}] · p={fmt_p(early['exact_two_sided_p'])}",
            },
            {
                "label": "L28 mediation specificity",
                "mean": mediation["mean"],
                "low": mediation["ci_low"],
                "high": mediation["ci_high"],
                "value": f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}] · p={fmt_p(mediation['exact_two_sided_p'])}",
            },
        ],
        title="Independent serial-path confirmation",
        description="Early donor log-odds gain and the orthogonal-control minus exact-L28-block mediation specificity are both positive on fresh seeds.",
        x_label="donor-vs-receiver candidate-sequence log-odds units",
    )
    loo_rows = []
    loo_table_rows = []
    for item in upstream["leave_one_out"]:
        dec = item["decrement"]
        supported = bool(item["incremental_contribution_supported"])
        loo_rows.append(
            {
                "label": f"remove {item['removed_head']}",
                "mean": dec["mean"],
                "low": dec["ci_low"],
                "high": dec["ci_high"],
                "color": "#D94B86" if supported else "#718096",
                "value": f"{fmt(dec['mean'], 4)} · Holm p={fmt_p(item['decrement_holm_p'])}",
            }
        )
        loo_table_rows.append(
            [
                item["removed_head"],
                fmt(item["loo_mediation"]["mean"], 4),
                f"{fmt(dec['mean'], 4)} [{fmt(dec['ci_low'], 4)}, {fmt(dec['ci_high'], 4)}]",
                fmt_p(dec["exact_two_sided_p"]),
                fmt_p(item["decrement_holm_p"]),
                "necessary within tested set"
                if supported
                else "no unique decrement resolved",
            ]
        )
    loo_forest = forest_svg(
        loo_rows,
        title="Leave-one-out membership in the L28 H16-H19 mediator set",
        description="Positive full-minus-LOO decrement means removing the named head weakens mediation. Only H19 survives Holm correction.",
        x_label="full-set mediation − leave-one-out mediation (log-odds units)",
    )
    return f"""
<section id="upstream">
<h2>10 · 上游 relay 与独立 serial-path confirmation</h2>
<h3>10.1 候选 relay：为什么“有信息”仍然不够</h3>
<div class="test-card"><h4>被检验的链路</h4><dl>
<dt>候选位置</dt><dd><code>pre_query_non_slot_tail_64</code>：answer query 前最后 64 个非 slot tokens；在 discovery 上冻结。</dd>
<dt>假说</dt><dd>这些 late positions 保存 count content，receiver attention 自然读取它，再经 L28 H16/H19 写到答案。</dd>
<dt>判定规则</dt><dd>carrier、V-only first stage、answer behavior、L28 mediation、natural removal 必须全部沿预定方向显著；任一门失败就不能称为自然 relay。</dd>
</dl></div>
<figure>{
        relay_svg
    }<figcaption><strong>Figure · tail-64 relay gate.</strong> 这是串行证据门图，没有共同数值坐标轴；每个框显示该阶段的 seed mean 与 exact p。绿色框通过预定方向，粉色框失败。carrier 与机械 first stage 成立，但 answer-level transport 区间跨 0，OV mediation 为 0，removal 方向相反，因此 global IUT p=0.9981。</figcaption></figure>
{
        table(
            [
                "步骤",
                "具体操作/问题",
                "effect [95% CI]",
                "exact directional p",
                "p<0.05 且方向正确?",
            ],
            relay_table_rows,
        )
    }
<p><strong>逐门判读：</strong>carrier p=9.54×10<sup>−7</sup>、V-only first-stage p=0.000405，均显著，说明 tail-64 中确有可读 content；但 answer behavior p=0.0693、OV mediation p=0.508，均不显著。两个 removal endpoint 还朝预注册方向的反面变化，directional p 分别为 0.9948 和 0.9981。global IUT p=0.9981，不显著。</p>
<div class="conclusion"><strong>本段结论</strong>tail-64 position set “可解码且可机械访问”，但没有证据表明模型自然依赖它把 count 送到 answer。否定仅针对这个冻结 position set，不否定其他 token set 或 MLP relay。</div>

<h3>10.2 真正得到支持的上游路径：fresh-seed serial mediation</h3>
<div class="plain-protocol"><h4>三次 forward 比较</h4><ol>
<li><strong>Source patch：</strong>把 donor 在 slot-query positions 上的 early top-4 set-output state patch 给 receiver。top-4 冻结为 L23H28、L23H29、L26H20、L27H18。</li>
<li><strong>Exact L28 block：</strong>在 L28 H16–H19 的 pre-O z 中，精确删除 source patch 诱发的自然 change。</li>
<li><strong>Matched control：</strong>删除同一 W<sub>O</sub> span、相同 post-O norm、但与自然 change 正交的方向。若 natural block 比 control 多消除 donor effect，才叫 mediation specificity。</li>
</ol></div>
<p>确认实验使用全新 seeds 1294–1313、六个 directed donor pairs；route、head set、endpoint 与 control construction 均在看这些 seeds 前冻结。零假设有两个：early source patch 不产生 donor gain；或 natural L28 block 不比正交 control 多消除 gain。</p>
<figure>{
        path_forest
    }<figcaption><strong>Figure · Independent serial mediation.</strong> 横轴统一为 donor-vs-receiver candidate-sequence log-odds units。第一行是 early slot-state patch 相对 clean 的 donor log-odds gain；第二行是 orthogonal control 保留的 gain 减 exact natural block 保留的 gain，即 L28 mediation specificity。点为 20 个 fresh-seed paired mean，线为 95% bootstrap CI，0 是无效应边界。</figcaption></figure>
{
        table(
            ["必要门", "effect [95% CI]", "exact two-sided p", "p<0.05?", "验证内容"],
            [
                [
                    "early source effect",
                    f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}]",
                    fmt_p(early["exact_two_sided_p"]),
                    '<span class="sig-yes">显著</span>',
                    "early top-4 slot-state patch 能推动 donor count",
                ],
                [
                    "L28 mediation specificity",
                    f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}]",
                    fmt_p(mediation["exact_two_sided_p"]),
                    '<span class="sig-yes">显著</span>',
                    "该 donor effect 特异经过 L28 H16–H19 natural change",
                ],
            ],
        )
    }
<p>两门 conjunction 的 IUT p 取较大值 0.005884，小于 0.05，故串行路径显著。120 primary rows、480 LOO rows以及 block closure、orthogonality、deterministic prefill audits 全部通过。</p>
<div class="conclusion"><strong>本段结论</strong>“early broad top-4 slot-state → L28 H16–H19 → answer”在独立 seeds 上复现。由于这是 donor-induced path mediation，它证明该通路能够并确实介导受控 source perturbation；它尚未单独证明 early top-4 在未干预 clean forward 中逐头必要。</div>

<h3>10.3 哪个 L28 head 对 set mediation 不可替代</h3>
<p><strong>具体操作：</strong>从完整 H16–H19 mediator set 中每次去掉一个 head，重新计算 mediation；定义 decrement=完整 set mediation−leave-one-out mediation。正 decrement 表示被去掉的 head 贡献不能由剩余 heads 替代。四次比较使用 Holm 校正，因此显著阈值看 Holm p&lt;0.05，而不是 raw p。</p>
<figure>{
        loo_forest
    }<figcaption><strong>Figure · Leave-one-out head-set membership.</strong> 横轴是 full H16–H19 mediation 减去移除指定 head 后的 mediation；正值表示该 head 对 set mediation 有不可由剩余成员替代的增量贡献。点与区间均为 20 个 fresh seeds 的 paired estimates；颜色仅区分 Holm-corrected support，统计意义不依赖颜色。</figcaption></figure>
{
        table(
            [
                "removed",
                "LOO mediation",
                "full−LOO [95% CI]",
                "exact p",
                "Holm p",
                "interpretation",
            ],
            loo_table_rows,
        )
    }
<p>移除 H19 的 decrement=0.1538 [0.0783, 0.2291]，raw p=0.00101、Holm p=0.00404，显著；剩余 H16–H18 的 mediation=0.0171，p=0.518，不显著。H16 decrement 的 raw p=0.0391，但 Holm p=0.117，不显著；H17/H18 Holm p=1，也不显著。</p>
<div class="conclusion"><strong>本节结论</strong>H19 是当前 H16–H19 mediator set 内的非冗余锚点；H16/H17/H18 更像冗余或支持性 companion subspace。这个结果不证明 H19 单头充分，也不把 counting 简化成 H19 的单头算法。</div>
</section>
"""


def build_gemma_serial_appendix(upstream: dict[str, Any]) -> str:
    cfg = upstream["config"]
    primary = upstream["primary_decision"]
    early = primary["early_effect"]
    mediation = primary["mediation"]
    confirmed = bool(primary["serial_chain_confirmed"])
    mediator_layer = int(cfg["mediator_layer"])
    early_heads = [f"L{int(row[0])}H{int(row[1])}" for row in cfg["early_candidates"]]
    primary_late = str(cfg["primary_late_set"])
    late_sets = {
        str(name): [int(head) for head in heads]
        for name, heads in cfg["late_head_sets"]
    }
    late_heads = late_sets[primary_late]
    late_label = "/".join(f"H{head}" for head in late_heads)
    path_forest = forest_svg(
        [
            {
                "label": "frozen broad-set slot-state patch",
                "mean": early["mean"],
                "low": early["ci_low"],
                "high": early["ci_high"],
                "value": f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}] · p={fmt_p(early['exact_two_sided_p'])}",
            },
            {
                "label": f"L{mediator_layer} natural-block specificity",
                "mean": mediation["mean"],
                "low": mediation["ci_low"],
                "high": mediation["ci_high"],
                "value": f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}] · p={fmt_p(mediation['exact_two_sided_p'])}",
            },
        ],
        title="Independent Gemma early-to-L37 serial mediation",
        description="The first row is the donor log-odds gain caused by the frozen broad-set slot-state patch. The second is orthogonal-control minus exact-natural-block retained gain at L37.",
        x_label="donor-vs-receiver candidate-sequence log-odds units",
    )
    loo_rows = []
    for item in upstream.get("leave_one_out", []):
        decrement = item["decrement"]
        loo_rows.append(
            [
                str(item["removed_head"]),
                fmt(item["loo_mediation"]["mean"], 4),
                f"{fmt(decrement['mean'], 4)} [{fmt(decrement['ci_low'], 4)}, {fmt(decrement['ci_high'], 4)}]",
                fmt_p(decrement["exact_two_sided_p"]),
                fmt_p(item["decrement_holm_p"]),
                "set 内非冗余"
                if bool(item["incremental_contribution_supported"])
                else "未解析出独立增量",
            ]
        )
    path_conclusion = (
        f"冻结 broad set（{', '.join(early_heads)}）产生 donor gain，且该 gain 被 L{mediator_layer} {late_label} 的自然 pre-O change 特异介导；Gemma 的受限 early→terminal 串联路径在 fresh seeds 上确认。"
        if confirmed
        else f"这组 frozen broad set → L{mediator_layer} {late_label} 没有通过两门 IUT；不能把它写成 Gemma 已确认的串联来源，即使某一单门或某个 LOO 对比为正。"
    )
    return f"""
<h3>10.4 Gemma fresh-seed 串联检验：frozen broad set → L{mediator_layer} {
        late_label
    }</h3>
<p>上游 set 没有根据本轮 outcome 重新排序：它冻结自 correct-only causal-v2 的 broad-aggregation K=2（{
        ", ".join(early_heads)
    }），晚端 mediator 冻结为 L{mediator_layer} {late_label}。确认数据使用 {
        seed_span(cfg["evaluation_seeds"])
    }、counts {min(cfg["counts"])}–{max(cfg["counts"])} 与 {
        len(cfg["donor_pairs"])
    } 个 directed donor pairs；每个 seed 而非 pair/token 是独立推断单位。</p>
<div class="plain-protocol"><h4>Gemma 串联对比的四个 forward</h4><ol>
<li>clean receiver；</li>
<li>只在 registered active-slot query positions patch frozen broad-set output；</li>
<li>同一 early patch，再精确恢复其诱发的 L{mediator_layer} {
        late_label
    } pre-O z change（natural block）；</li>
<li>同一 early patch，再删除相同 <em>W</em><sub>O</sub> span、相同 post-O norm、但与自然 change 正交的 control。</li>
</ol></div>
<p>第一门要求 early patch 的 donor-vs-receiver candidate-sequence log-odds gain&gt;0；第二门要求 <code>M=gain<sub>orthogonal</sub>−gain<sub>natural block</sub>&gt;0</code>。全局 IUT p 是两门 exact seed-level sign-flip p 的最大值；closure≤10<sup>−5</sup>、orthogonality≤10<sup>−4</sup> 与 deterministic-prefill≤10<sup>−5</sup> 还必须全部通过。</p>
<div class="conclusion"><strong>设计结论</strong>这个检验验证的是受控 source perturbation 是否沿 frozen late channel 传播；它足以建立一条受支持的逻辑通路，但不要求证明这是模型唯一 relay，也不把每个 early head 都写成 clean forward 中逐头必要。</div>
<figure>{
        path_forest
    }<figcaption><strong>Figure · Gemma independent serial mediation.</strong> 横轴统一为 donor-vs-receiver candidate-sequence log-odds。第一行是 early slot-state patch 相对 clean 的 gain；第二行是正交 control 保留的 gain 减 exact natural block 保留的 gain。点是 {
        len(cfg["evaluation_seeds"])
    } 个 fresh-seed means，横线是 seed-cluster bootstrap 95% CI，0 为无效应。</figcaption></figure>
{
        table(
            ["必要门", "effect [95% CI]", "exact two-sided p", "p<0.05?", "含义"],
            [
                [
                    "early source effect",
                    f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}]",
                    fmt_p(early["exact_two_sided_p"]),
                    sig_badge(
                        early["exact_two_sided_p"] if early["mean"] > 0 else 1.0,
                        label="支持"
                        if early["mean"] > 0 and early["exact_two_sided_p"] < 0.05
                        else "不支持",
                    ),
                    "frozen broad set 能否推动 donor count",
                ],
                [
                    f"L{mediator_layer} mediation specificity",
                    f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}]",
                    fmt_p(mediation["exact_two_sided_p"]),
                    sig_badge(
                        mediation["exact_two_sided_p"]
                        if mediation["mean"] > 0
                        else 1.0,
                        label="支持"
                        if mediation["mean"] > 0
                        and mediation["exact_two_sided_p"] < 0.05
                        else "不支持",
                    ),
                    "donor effect 是否特异经过 frozen late set",
                ],
            ],
        )
    }
<p>串联判定为 <code>serial_chain_confirmed={str(confirmed).lower()}</code>，IUT p={
        fmt_p(primary["intersection_union_p"])
    }。这里的 p 不是把 donor pairs 当独立样本得到的；exact sign-flip 只作用于 seed means，95% CI 也按 seed cluster bootstrap。</p>
<div class="conclusion"><strong>串联结果</strong>{path_conclusion}</div>
{
        details_table(
            "Gemma late-set leave-one-out",
            [
                "removed",
                "LOO mediation",
                "full−LOO [95% CI]",
                "exact p",
                "Holm p",
                "interpretation",
            ],
            loo_rows,
        )
        if loo_rows
        else ""
    }
<p class="small">LOO 只回答“某个晚端 head 在当前 frozen set 内是否有不可由另一成员替代的增量贡献”；它不检验单头充分性，也不允许把 counting 简化为一个 head 的算法。</p>
<div class="conclusion"><strong>本小节边界</strong>正结果建立一条可复现的 Gemma early→late 逻辑通路；负结果只否定这组冻结 broad set 作为已确认上游来源，不否定其他 early sets、MLP-mediated relay 或并行通道。</div>
"""


def build_synthesis_section() -> str:
    claim_rows = [
        [
            "模型在 prompt 读取阶段形成 running index",
            "跨 position/order/content 的 frozen-basis geometry；cue removal 高 CKA",
            "强表征证据；不是因果运算证明",
        ],
        [
            "检索是分布式而非严格单头",
            "attention atlas；ranked bank perturbation；early top-4 route",
            "支持；特定 early heads 的 clean necessity 未完成",
        ],
        [
            "late answer state 是可执行 count carrier",
            "answer-state patch、steering、late geometry",
            "跨 Qwen/Gemma 功能因果支持",
        ],
        [
            "Qwen L28 H16/H19 是自然 OV transporter",
            "四族 IUT：signal/injection/removal/mediation",
            "确认；部分 mediation",
        ],
        [
            "Qwen L28 read 同时依赖 α 与 V",
            "crossed α–V decomposition + OV block",
            "支持；parent seeds 机制扩展",
        ],
        [
            "early slot state 经 L28 H16–H19 到 answer",
            "fresh-seed exact-block serial mediation",
            "独立确认的受限链路",
        ],
        ["tail-64 是自然 relay", "registered four-family relay test", "不支持"],
        ["存在唯一逐 token +1 head", "当前没有直接测试支持", "不可声称"],
    ]
    step_rows = [
        [
            "1 · Prompt representation",
            "读取第 n 个 active record 后，在 needle-end residual 观察 ordered running-index state",
            "跨 seed 的 frozen-basis geometry；cue-present/absent shared-basis audit",
            "N/A：表征图不是单一因果检验",
            "存在可解码的累计进度；不等于模型已使用",
        ],
        [
            "2 · Early source read",
            "patch early top-4 在 slot-query positions 的 set-output state",
            "donor log-odds gain=0.1057 [0.0412, 0.1683]",
            "p=0.005884，显著",
            "early broad bank 可把 slot-state signal 送向答案",
        ],
        [
            "3 · L28 mixed read",
            "在 L28 H16–H19 分别替换 α routing 与 V content",
            "routing=0.0517；value=0.0524；value−routing=0.0008",
            "两 component p=9.54×10⁻⁷，显著；差值 p=0.451，不显著",
            "两种读取方式都参与，大小无法区分",
        ],
        [
            "4 · Natural OV write",
            "pre-O injection、centered removal、donor-path block 与 matched controls",
            "四个必要 evidence families 全部通过",
            "global IUT p=0.004541，显著",
            "H16/H19 是自然使用的部分 transporter",
        ],
        [
            "5 · Downstream survival",
            "比较 natural 与 same-span orthogonal step 在 L28–L35 的 count-axis projection",
            "L35 specificity=0.0156 [0.0108, 0.0201]",
            "layer-family 最大校正 p=2.29×10⁻⁵，显著",
            "L28 写入没有在后层立即消失",
        ],
        [
            "6 · Answer readout",
            "patch 完整 Total: query state，观察是否采用 donor prediction",
            "Qwen clean-correct pooled adoption=96.6%",
            "描述性 pooled rate；此 supplement 不提供单一确认 p",
            "late answer state 可执行地携带已算出的 prediction",
        ],
        [
            "Rejected branch",
            "对 tail-64 relay 做 carrier、edge、behavior、mediation、removal conjunction",
            "behavior/mediation 不通过，removal 反方向",
            "global IUT p=0.9981，不显著",
            "不能把 tail-64 写成自然 relay",
        ],
    ]
    return f"""
<section id="synthesis">
<h2>11 · Mechanism synthesis：把前面的实验翻译成一条可读的计数流程</h2>
<figure>{mechanism_svg()}<figcaption><strong>Figure · Supported non-thinking counting mechanism.</strong> 该图是因果结构图，没有数值坐标轴。实线箭头表示已有 transport/mediation 支持；框的粒度就是当前定位精度。灰色虚线分支标出被否定的 tail-64 relay。图中 early top-4→L28 链路已在 fresh seeds 复现；mixed read/write 分解仍是 parent-seed 机制扩展。</figcaption></figure>
<h3>11.1 六步机制与每一步的统计判定</h3>
<p>下面每一行只回答一个问题。先形成可解码 representation，再验证 source patch 能否到达答案；随后拆分 L28 的读取、验证其自然 OV 写入、检查写入能否存活，最后定位可执行的 answer state。不同 effect 量纲不相同，因此不跨行求平均。</p>
{table(["步骤", "具体做了什么", "主要观察", "显著性", "这一步允许的结论"], step_rows)}
<div class="step-result"><strong>最关键的 conjunction。</strong>early source effect 与 L28 mediation 的 fresh-seed IUT p=0.005884；natural OV 四证据族 global IUT p=0.004541；两者都小于 0.05。故可以把“early slot-state → L28 natural OV → late answer”写成受支持的串行机制。tail-64 的 p=0.9981，必须作为被否定的具体 relay 分支保留。</div>
<div class="conclusion"><strong>本段结论</strong>最小机制是：prompt 中形成分布式 running-index state；early bank 读取/汇集 slot-state signal；Qwen L28 H16/H19 以 mixed α/V 方式读取并通过自然 OV 方向写回；该写入存活至 late answer state。这里的 <em>W</em><sub>O</sub> 与后续 Jacobian 正是 prompt counter 和 answer counter 可以方向不同、但 count ordering 与因果信息保持的机制。我们没有证明唯一单头 +1 运算，也没有穷尽所有 relay。</div>

<h3>11.2 Claim matrix</h3>
{table(["可写入论文的命题", "证据", "允许的强度"], claim_rows)}
<div class="paper-wording"><strong>建议正文表述。</strong>“Across realistic 10k-token counting prompts, non-thinking models expressed an ordered prompt-side running-index geometry and a late answer-query count state. In Qwen3-8B, preregistered pre-output interventions identified a natural L28 OV transport channel: the H16/H19 set carried a count-correlated component, supported signed pre-O injection, was selectively necessary under centered z-space removal, and mediated donor-state transport. A factorized α–V intervention indicated mixed routing and value-content readout, while the induced OV write remained count-aligned through the final layer. Finally, an independent fresh-seed experiment confirmed serial mediation from a frozen early broad-retrieval slot-state set through L28 H16–H19 to the answer distribution; leave-one-out analysis identified H19 as nonredundant within this tested set.”</div>
<div class="conclusion"><strong>本节结论</strong>这套证据足以支持一个 set-level、分布式的 non-thinking read–write mechanism；不足以支持唯一单头、显式整数寄存器或完整无遗漏 circuit 的表述。论文正文应同时报告 p、effect、CI 与 seed 独立性边界。</div>
</section>
"""


def build_gemma_synthesis_appendix(
    gemma_ov: dict[str, Any],
    gemma_read_write: dict[str, Any],
    gemma_upstream: dict[str, Any],
) -> str:
    natural = bool(gemma_ov["primary_decision"]["full_natural_ov_transporter_support"])
    read_write = bool(
        gemma_read_write["primary_decision"]["serial_read_write_supported"]
    )
    serial = bool(gemma_upstream["primary_decision"]["serial_chain_confirmed"])
    full = natural and read_write and serial
    rows = [
        [
            "Prompt running-index representation",
            "frozen-basis PCA / full-space statistics / cue-paired audit",
            "支持；结构在 cue removal 后保留",
            '<span class="evidence descriptive">表征</span>',
        ],
        [
            "Frozen broad-retrieval function",
            "correct-only K=1/K=2 ablation vs 3 layer-matched random sets",
            "K1/K2 clean-correct failure CI 均排除 0",
            '<span class="evidence functional">独立功能支持</span>',
        ],
        [
            "L37 H1/H2 natural OV",
            "carrier + true pre-O injection + centered removal + mediation",
            f"global IUT p={fmt_p(gemma_ov['primary_decision']['global_intersection_union_p'])}",
            sig_badge(0.0 if natural else 1.0, label="通过" if natural else "未通过"),
        ],
        [
            "L37 terminal read/write",
            "sliding-window-aware crossed α/V + L37→L41 propagation",
            f"classification={html.escape(str(gemma_read_write['primary_decision']['read_mode']['classification']))}；supported={read_write}",
            sig_badge(
                0.0 if read_write else 1.0, label="通过" if read_write else "未通过"
            ),
        ],
        [
            "Frozen broad set → L37 → answer",
            "fresh-seed exact natural block vs same-span orthogonal control",
            f"IUT p={fmt_p(gemma_upstream['primary_decision']['intersection_union_p'])}",
            sig_badge(0.0 if serial else 1.0, label="通过" if serial else "未通过"),
        ],
        [
            "Late answer state",
            "clean-correct full-state patch",
            "pooled donor-target adoption 96.0%",
            '<span class="evidence functional">功能因果</span>',
        ],
    ]
    if full:
        claim = "Gemma 在冻结候选与 fresh seeds 上复现了与 Qwen 同构的分布式 read/write 路径；可做跨模型 mechanism replication，但 head/layer identity 不相同。"
        paper_sentence = (
            "In Gemma4-E4B, a separately frozen L37 H1/H2 set satisfied the same natural-OV "
            "signal, pre-output sufficiency, centered-necessity, and path-mediation criteria. "
            "A sliding-window-aware α–V decomposition supported terminal read/write, and a "
            "fresh-seed exact-block experiment confirmed serial mediation from the frozen "
            "L29H4/L35H2 broad set through L37. We therefore interpret Gemma as a tested-path "
            "replication of the distributed read/write mechanism, not a replication of Qwen head identity."
        )
    elif natural and read_write:
        claim = "Gemma 的 terminal natural transporter 与内部 read/write 已成立，但 frozen L29H4/L35H2 尚未被确认是该 transporter 的上游来源。"
        paper_sentence = (
            "Gemma4-E4B replicated the terminal natural-OV and sliding-window-aware read/write "
            "signatures at L37 H1/H2; however, the fresh-seed serial test did not confirm the "
            "frozen L29H4/L35H2 set as its upstream source."
        )
    elif natural:
        claim = "Gemma L37 的自然使用证据成立，但当前 α/V 分解或 downstream propagation 没有闭合，不能写成完整 terminal read/write chain。"
        paper_sentence = (
            "Gemma4-E4B showed natural causal use of the frozen L37 H1/H2 OV channel, but the "
            "factorized read/write or downstream-propagation criteria did not close; we therefore "
            "do not claim a full Gemma read/write-chain replication."
        )
    else:
        claim = "Gemma 尚未复制完整 natural-transporter 路径；positive representation、ablation 或 derivative effects 只能作为各自层级的局部证据。"
        paper_sentence = (
            "Gemma4-E4B retained prompt-side geometry, broad-bank ablation effects, and a late "
            "answer state, but the preregistered natural-transporter conjunction was not satisfied; "
            "we therefore restrict Gemma claims to the individually supported representational or functional links."
        )
    return f"""
<h3>11.3 Gemma 跨模型 synthesis：哪些 link 真正复制</h3>
{table(["link", "直接检验", "结果", "证据等级"], rows)}
<p>跨模型复制使用“同一因果问题、同一推断单位、同一 matched-control 逻辑”，而不是要求同层同头。Gemma 的 L37 滑窗还意味着 terminal read 的 source 是进入该层前已形成的可访问 state；因此即便整条路径通过，也应写成 broad-set/relay → terminal read/write，而不是 L37 直接从 10k-token prompt 原始 needles 做全局 QK。</p>
<div class="conclusion"><strong>跨模型结论</strong>{claim}</div>
<p>联合判断遵循预先写定的 interpretation matrix：natural OV、α/V read/write、fresh-seed early→L37 mediation 三项全部通过，才叫 full tested-path replication；任一环节失败，就只保留已单独通过的前缀或局部 link。没有把三个 p 值相乘，也没有在看到结果后更换 Gemma heads。</p>
<div class="paper-wording"><strong>Suggested cross-model wording.</strong> {html.escape(paper_sentence)}</div>
<div class="conclusion"><strong>论文写作边界</strong>这里建立的是一条冻结、受控、可复现的逻辑通路；它不要求否定所有其他 relay，也不支持“唯一 circuit”或“唯一单头 counter”。</div>
"""


def build_gemma_synthesis_ladder(
    *,
    l37: dict[str, Any],
    singles: dict[str, dict[str, Any]],
    read_writes: dict[str, dict[str, Any]],
    cross_layer: dict[str, Any] | None,
    residuals: dict[str, dict[str, Any]],
    story: dict[str, Any],
) -> str:
    rows: list[list[str]] = [
        [
            "Prompt running-index representation",
            "frozen-basis PCA / full-space cue-paired statistics",
            "有序 geometry 在 cue removal 后保留",
            '<span class="evidence descriptive">表征</span>',
        ],
        [
            "Frozen broad-retrieval function",
            "correct-only K=1/K=2 ablation vs 3 layer-matched controls",
            "fresh-seed clean-correct failure 与 ΔMAE 显著",
            '<span class="evidence functional">独立功能支持</span>',
        ],
        [
            "L37 H1/H2 localized natural OV",
            "四族 true-pre-O conjunction",
            f"global IUT p={fmt_p(l37['primary_decision']['global_intersection_union_p'])}",
            evidence_badge(
                bool(l37["primary_decision"]["full_natural_ov_transporter_support"]),
                "通过",
                "否定该候选",
            ),
        ],
    ]
    for name, document in singles.items():
        decision = document["primary_decision"]
        rows.append(
            [
                f"{ov_candidate_label(document)} localized natural OV",
                "independent-ablation-ranked single; four-family IUT",
                f"global IUT p={fmt_p(decision['global_intersection_union_p'])}",
                evidence_badge(
                    bool(decision["full_natural_ov_transporter_support"]),
                    "通过",
                    "否定该候选",
                ),
            ]
        )
        if name in read_writes:
            rw = read_writes[name]
            rows.append(
                [
                    f"{ov_candidate_label(document)} α/V read/write",
                    "crossed α/V + downstream trace",
                    f"classification={html.escape(str(rw['primary_decision']['read_mode']['classification']))}; serial={rw['primary_decision']['serial_read_write_supported']}",
                    evidence_badge(
                        bool(rw["primary_decision"]["serial_read_write_supported"]),
                        "机制分解支持",
                        "未完整支持",
                    ),
                ]
            )
    if cross_layer is not None:
        rows.append(
            [
                "L29H4+L35H2 cross-layer set",
                "joint natural OV + exact L29→L35 relay",
                f"OV p={fmt_p(cross_layer['primary_decision']['global_intersection_union_p'])}; relay p={fmt_p(cross_layer['relay_decision']['intersection_union_p'])}",
                evidence_badge(
                    bool(cross_layer["full_cross_layer_mechanism_support"]),
                    "通过",
                    "未闭合",
                ),
            ]
        )
    for residual_name, residual in residuals.items():
        rows.append(
            [
                f"{residual_variant_label(residual)}→L{int(residual['selected_mediator_layer'])} residual→L41",
                (
                    "clean necessity + "
                    if residual["config"].get("require_clean_necessity", False)
                    else ""
                )
                + "source patch + exact/count-axis mediation + terminal adoption",
                f"global IUT p={fmt_p(residual['primary_decision']['global_intersection_union_p'])}",
                evidence_badge(
                    bool(
                        residual["primary_decision"]["full_residual_count_path_support"]
                    ),
                    "通过",
                    "未闭合",
                ),
            ]
        )
    if story["kind"] == "single":
        rw = story.get("read_write")
        rw_clause = (
            "；factorized α/V 与 downstream propagation 也满足其 extension 判据"
            if rw is not None
            and bool(rw["primary_decision"]["serial_read_write_supported"])
            else "；α/V extension 不作为独立复制"
        )
        claim = (
            f"Gemma 的最强定位是 {story['label']} natural OV transporter"
            f"（global IUT p={fmt_p(story['global_p'])}）{rw_clause}。"
        )
        paper = (
            f"In Gemma4-E4B, the independently ranked {story['label']} candidate "
            "satisfied the frozen natural-signal, true pre-output sufficiency, "
            "centered-necessity, and donor-path mediation conjunction on held-out seeds."
        )
    elif story["kind"] == "cross_layer":
        claim = (
            f"Gemma 的最强定位是 {story['label']} 跨层 set：joint natural OV 与 "
            f"L29→L35 exact-block relay 同时通过（joint p={fmt_p(story['global_p'])}）。"
        )
        paper = (
            "In Gemma4-E4B, a frozen cross-layer L29H4/L35H2 set satisfied both "
            "the joint natural-OV conjunction and an exact-block L29-to-L35 relay test "
            "on held-out seeds."
        )
    elif story["kind"] == "residual":
        claim = (
            f"Gemma 未定位到通过全部门的单头/局部 OV set，但 {story['label']} 的 "
            f"分布式 residual relay 通过（global IUT p={fmt_p(story['global_p'])}）。"
        )
        paper = (
            "In Gemma4-E4B, localized natural-OV hypotheses did not satisfy their full "
            "conjunctions. A separately confirmed fallback nevertheless showed that the "
            f"frozen {story['label']} path causally wrote a count-aligned distributed "
            "residual state whose registered removal reduced donor-count transport to "
            "the terminal layer."
        )
    else:
        claim = (
            "Gemma 的 representation、answer-state patching 与 broad-bank necessity 成立，"
            "但当前冻结的 localized/cross-layer/residual conjunction 均未闭合。"
        )
        paper = (
            "Gemma4-E4B retained an ordered running-index representation, a causally "
            "effective answer state, and broad-bank ablation effects, but none of the "
            "frozen localized or distributed transport conjunctions closed; we therefore "
            "do not claim a complete Gemma head-level circuit."
        )
    return f"""
<h3>11.3 Gemma synthesis：由实际通过的最强门决定表述</h3>
{table(["link", "直接检验", "结果", "证据等级"], rows)}
<div class="conclusion"><strong>跨模型结论</strong>{html.escape(claim)}</div>
<p>跨模型复制指相同的计算问题与 matched-control 逻辑，不要求相同 layer/head identity。若 Gemma 最强层级是 residual，就只能主张“同构的分布式 state transport”，不能称为 Qwen L28 局部 OV circuit 的逐头复制。</p>
<div class="paper-wording"><strong>Suggested cross-model wording.</strong> {html.escape(paper)}</div>
<div class="conclusion"><strong>论文写作边界</strong>这里寻找并验证一条可复现的逻辑通路，不需要否定所有并行 relay；但任何失败的更强定位都必须与较弱正结果并列保留。</div>
"""


def build_correct_state_boundary(
    analysis: dict[str, Any], geometry_rows: list[dict[str, str]]
) -> str:
    if not analysis.get("audits", {}).get("all_checks_pass", False):
        raise RuntimeError("Correct-only state-route audit did not pass")
    geometry_table_rows: list[list[str]] = []
    location_labels = {
        "prompt_running_counter_source_bank_z": "prompt endpoint · frozen source-bank z",
        "answer_query_read_aggregate_source_bank_z": "answer query · source-bank aggregate z",
    }
    for row in sorted(
        geometry_rows,
        key=lambda item: (str(item["model_label"]), str(item["location"])),
    ):
        supported = str(row["geometry_supported_beyond_position"]).lower() == "true"
        geometry_table_rows.append(
            [
                html.escape(str(row["model_label"])),
                html.escape(location_labels.get(str(row["location"]), str(row["location"]))),
                fmt(float(row["oof_rounded_accuracy"]), 3),
                fmt_p(float(row["position_adjusted_iut_p"])),
                fmt_p(float(row["position_adjusted_iut_holm_p"])),
                evidence_badge(supported, "超过 position control", "未超过 position control"),
            ]
        )

    route_labels = {
        "answer_query_aggregate": "answer-query aggregate patch",
        "slot_endpoint_state": "single prompt endpoint patch",
    }
    route_table_rows: list[list[str]] = []
    for row in sorted(
        analysis["route_results"],
        key=lambda item: (str(item["model_label"]), str(item["route"])),
    ):
        route_table_rows.append(
            [
                html.escape(str(row["model_label"])),
                html.escape(route_labels.get(str(row["route"]), str(row["route"]))),
                f"{fmt(float(row['source_donor_log_odds_gain_mean']), 4, signed=True)} "
                f"[{fmt(float(row['source_donor_log_odds_gain_ci95_low']), 4)}, {fmt(float(row['source_donor_log_odds_gain_ci95_high']), 4)}]"
                f"; p={fmt_p(float(row['source_donor_log_odds_gain_p']))}",
                f"{fmt(float(row['writer_log_odds_mediation_specificity_mean']), 4, signed=True)} "
                f"[{fmt(float(row['writer_log_odds_mediation_specificity_ci95_low']), 4)}, {fmt(float(row['writer_log_odds_mediation_specificity_ci95_high']), 4)}]"
                f"; p={fmt_p(float(row['writer_log_odds_mediation_specificity_p']))}",
                evidence_badge(bool(row["route_supported"]), "完整 route 通过", "完整 route 未通过"),
            ]
        )

    return f"""
<h3>11.4 Correct-only low-count boundary：读到 state 不等于旧 writer set 完成写入</h3>
<p>这项补充实验只保留<strong>两个冻结模型在 clean forward 都正确回答 count 1–3</strong>的样本。20 个全新 seeds 内，每个 seed 平均六个有向 donor→receiver count pairs；推断单位仍是 seed。source patch 与 writer-specific block 在同一 forward family 内评估，所有 480 条 effect rows 的复现、norm、正交与 closure audits 均通过。</p>
{table(
    ["模型", "state location", "OOF rounded acc.", "position-adjusted IUT p", "Holm p", "判定"],
    geometry_table_rows,
)}
<p>answer-query source-bank aggregate 在两模型中都显著超过 cubic normalized-position control（Holm p=3.81×10<sup>−6</sup>）；Gemma 的 prompt source-bank z 也通过（Holm p=0.001427），而 Qwen 这一个<strong>特定 frozen bank 的 pre-O z</strong>未通过（Holm p=0.3768）。后者不否定前文在 full residual 上得到的 Qwen running-index geometry，因为 location、subspace 与 estimand 不同。</p>
{table(
    ["模型", "route", "source donor log-odds gain [95% CI]", "old writer-set mediation [95% CI]", "联合判定"],
    route_table_rows,
)}
<p>在 answer query 直接搬运 aggregate state 时，Qwen source gain 为 +0.6776 [0.4564, 0.8978]（p=1.43×10<sup>−5</sup>），Gemma 为 +12.1734 [11.3239, 12.9635]（p=9.54×10<sup>−7</sup>）；说明正确低-count 运行中确有可执行 readout state。但旧冻结 writer set 的 axis-specific mediation 不成立：Qwen −0.0176（p=0.7092），Gemma −0.0594（p=0.9899，且方向相反）。single prompt endpoint patch 对 Gemma 为严格 0，对 Qwen source gain 为负，因此也没有闭合 prompt-endpoint→writer route。</p>
<div class="conclusion"><strong>本小节结论</strong>这轮 correct-only 实验强化“两个模型都形成可执行 answer-query count state”，同时否定“旧低-count writer set 普遍介导该 state”的更强说法。它不推翻 Qwen 主实验中 H16/H19 的 all-count natural-OV conjunction，因为两者的冻结 set、被干预位置与 count regime 不同；更合适的解释是 Qwen 已有一个确认的 OV 通路，但其支配性可能随 regime/route 改变。对 Gemma，它进一步支持只写成 distributed effective residual write，而不指定 localized OV heads。</div>
"""


def build_limits_dynamic(
    *,
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    relay: dict[str, Any],
    upstream: dict[str, Any],
    gemma_l37: dict[str, Any],
    gemma_singles: dict[str, dict[str, Any]],
    gemma_read_writes: dict[str, dict[str, Any]],
    gemma_cross_layer: dict[str, Any] | None,
    gemma_residuals: dict[str, dict[str, Any]],
    gemma_story: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    provenance_rows = [
        [
            "Representation + macro mechanism",
            "reports/v4_non-thinking_causal/v4_4_3/realistic_niah_v4_4_mechanism_report.html",
            "self-contained V4.4 interactive report",
        ],
        [
            "causal-v2",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json",
            f"schema {causal_v2['schema_version']}",
        ],
        [
            "correct-only seed extrapolation",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/seed_extrapolation_summary.json",
            f"audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS",
        ],
        [
            "Qwen natural OV",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json",
            f"schema {ov['schema_version']}",
        ],
        [
            "Qwen read/write",
            "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
            f"schema {read_write['schema_version']}",
        ],
        [
            "Qwen relay",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_relay_analysis.json",
            f"schema {relay['schema_version']}",
        ],
        [
            "Qwen upstream",
            "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
            f"schema {upstream['schema_version']}",
        ],
        [
            "Gemma L37 retained negative",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/natural_ov/realistic_niah_v4_4_4_analysis.json",
            f"schema {gemma_l37['schema_version']}",
        ],
        [
            "correct-only low-count state routes",
            "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json",
            f"schema {correct_state['schema_version']} · 480/480 effect rows audited",
        ],
    ]
    for name, document in gemma_singles.items():
        provenance_rows.append(
            [
                f"Gemma {name} natural OV",
                f"reports/v4_non-thinking_causal/v4_4_4/gemma/search/{name}/realistic_niah_v4_4_4_analysis.json",
                f"schema {document['schema_version']}",
            ]
        )
    for name, document in gemma_read_writes.items():
        provenance_rows.append(
            [
                f"Gemma {name} read/write",
                f"reports/v4_non-thinking_causal/v4_4_4/gemma/search/{name}/realistic_niah_v4_4_4_read_write_analysis.json",
                f"schema {document['schema_version']}",
            ]
        )
    if gemma_cross_layer is not None:
        provenance_rows.append(
            [
                "Gemma cross-layer K2",
                "reports/v4_non-thinking_causal/v4_4_4/gemma/cross_layer/realistic_niah_v4_4_4_cross_layer_analysis.json",
                f"schema {gemma_cross_layer['schema_version']}",
            ]
        )
    for residual_name, gemma_residual in gemma_residuals.items():
        provenance_rows.append(
            [
                f"Gemma {residual_name.upper()} residual relay",
                f"reports/v4_non-thinking_causal/v4_4_4/gemma/residual/{residual_name}/realistic_niah_v4_4_4_residual_analysis.json",
                f"schema {gemma_residual['schema_version']}",
            ]
        )
    limit_rows = [
        [
            "跨模型身份",
            f"Gemma strongest layer={gemma_story['kind']}；Qwen/Gemma 使用不同 layer/head set",
            "只主张相同计算问题上的路径同构，不主张 head identity 或架构普适性",
        ],
        [
            "搜索树多重性",
            "各 Gemma 分支有独立 confirmation seeds 与内部 IUT；跨分支无单一全局 FWER",
            "写作中标明 sequential exploratory search + held-out confirmation，保留所有负分支",
        ],
        [
            "read/write 独立性",
            "factorized α/V extension 复用 parent candidate seeds",
            "只解释已冻结通道如何工作，不当作第二个独立 replication",
        ],
        [
            "并行机制",
            "验证一条 causal path 不穷尽网络中的其他 heads、MLPs 或 token relays",
            "不写唯一 circuit；负结果只否定对应冻结候选",
        ],
        [
            "PCA 推断",
            "三维仅显示冻结 basis 的前三方向；显著性来自 full-space / causal endpoints",
            "不从视觉间距直接推断 effect size 或 p 值",
        ],
    ]
    return f"""
<section id="limits">
<h2>12 · 证据边界与可复现性</h2>
{table(["边界", "当前事实", "写作约束"], limit_rows)}
<p>Gemma 证据阶梯的目的不是“试到显著为止”，而是把越来越宽松的机制定位逐层分开：localized head、cross-layer set、distributed residual。更弱分支通过不会抹去更强分支的失败，也不会获得更强分支的术语。</p>
<div class="conclusion"><strong>本段结论</strong>Qwen 可写成一条闭合的受限 OV mechanism；Gemma 只能写到 <code>{
        html.escape(str(gemma_story["kind"]))
    }</code>，其具体表述为：{html.escape(str(gemma_story["summary"]))}</div>
{
        details_table(
            "Source ledger",
            ["component", "relative path", "audit/schema note"],
            provenance_rows,
        )
    }
{
        details_table(
            "FileStream raw/derived roots",
            ["campaign", "FileStream path", "role"],
            [
                [
                    "V4.4 representation source",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3",
                    "raw generations/captures and frozen stimuli",
                ],
                [
                    "Qwen V4.4.4",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_natural_ov/run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1",
                    "natural OV / read-write / upstream derivatives",
                ],
                [
                    "correct-only frozen top-k",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260804_v4_4_ablation_seed_extrapolation_qwen_n2_n4_gemma_n1_n2",
                    "Qwen/Gemma ranked-vs-random necessity",
                ],
                [
                    "Gemma retained L37",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_gemma_replication/v4_4_4_natural_ov/run_20260805_gemma_l37_h1_h2_frozen_v1",
                    "retained negative localized hypothesis",
                ],
                [
                    "Gemma evidence ladder",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_gemma_mechanism_search",
                    "single-head, cross-layer and residual gated branches",
                ],
                [
                    "correct-only low-count routes",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_correct_state_routes/run_20260805_dual_model_lowcount_correct",
                    "answer-query aggregate / prompt-endpoint patches and writer mediation",
                ],
            ],
        )
    }
<p class="small">报告只嵌入聚合统计与可视化，不复制 raw hidden states、full V tensors 或 raw attention rows；原始数据保留在 FileStream。builder 对每个实际存在的 analysis JSON 强制 audit PASS；causal-v2 与 correct-only audits 也必须通过。</p>
<div class="conclusion"><strong>最终结论</strong>non-thinking counting 最符合“分布式 running-index representation → broad retrieval → causal write/relay → 可执行 answer count state”。Qwen 已解析到 localized OV set；Gemma 的定位粒度严格由冻结证据阶梯的实际通过层级决定。</div>
</section>
"""


def build_limits_section(
    repo_root: Path,
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    relay: dict[str, Any],
    upstream: dict[str, Any],
    gemma_ov: dict[str, Any],
    gemma_read_write: dict[str, Any],
    gemma_upstream: dict[str, Any],
) -> str:
    gemma_full = (
        bool(gemma_ov["primary_decision"]["full_natural_ov_transporter_support"])
        and bool(gemma_read_write["primary_decision"]["serial_read_write_supported"])
        and bool(gemma_upstream["primary_decision"]["serial_chain_confirmed"])
    )
    provenance_rows = [
        [
            "Representation + macro mechanism",
            "reports/v4_non-thinking_causal/v4_4_3/realistic_niah_v4_4_mechanism_report.html",
            "self-contained V4.4 interactive report",
        ],
        [
            "causal-v2",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json",
            f"schema {causal_v2['schema_version']}",
        ],
        [
            "correct-only seed extrapolation",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/seed_extrapolation_summary.json",
            f"audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS",
        ],
        [
            "20-seed exact sign-flip reanalysis",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/exact_sign_flip_reanalysis.json",
            "full 2^20 enumeration; Holm across four frozen sets",
        ],
        [
            "natural OV",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json",
            f"schema {ov['schema_version']}",
        ],
        [
            "read/write",
            "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
            f"schema {read_write['schema_version']}",
        ],
        [
            "relay",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_relay_analysis.json",
            f"schema {relay['schema_version']}",
        ],
        [
            "upstream confirmation",
            "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
            f"schema {upstream['schema_version']}",
        ],
        [
            "Gemma natural OV",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/natural_ov/realistic_niah_v4_4_4_analysis.json",
            f"schema {gemma_ov['schema_version']}",
        ],
        [
            "Gemma read/write",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
            f"schema {gemma_read_write['schema_version']}",
        ],
        [
            "Gemma upstream confirmation",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
            f"schema {gemma_upstream['schema_version']}",
        ],
    ]
    limit_rows = [
        [
            "跨模型身份",
            f"Gemma full tested-path replication={gemma_full}；无论联合结果如何，Qwen 与 Gemma 使用不同 layer/head set",
            "只主张同构计算路径，不主张相同 head identity 或架构普适性",
        ],
        [
            "Gemma L37 可见窗口",
            "L37 的 semantic source groups 与实际 capture key window 取交集；窗外组显式为 0",
            "把 L37 写成 terminal accessible-state reader，不写成直接回看原始 10k-token needles",
        ],
        [
            "early-set 因果范围",
            "frozen top-k clean-correct ablation确认 bank-level function；serial mediation确认 donor-induced path",
            "不把每个 early head 写成逐头 clean-run 必要，也不要求排除所有其他 relay",
        ],
    ]
    return f"""
<section id="limits">
<h2>12 · 证据边界与可复现性</h2>
{table(["边界", "当前事实", "如何处理"], limit_rows)}
<p>Gemma 等价实验与 correct-only frozen top-k ablation 已经并入，不再作为“待补实验”。tail-64 仍作为一个被严格否定的具体候选保留，但“穷尽并否定所有其他 relay”不属于当前论文主张的必要条件；我们的正面结论是一条冻结通路获得多层证据收敛，而不是全网络唯一性证明。</p>
<div class="conclusion"><strong>本段结论</strong>现有数据已经足以对 Qwen 给出闭合的受限机制链，并对 Gemma 给出由其三项联合判定所允许的最强跨模型结论。尚存边界主要是外部模型/架构普适性与未测试并行路径，而不是报告中仍缺一块已承诺的数据。</div>
{
        details_table(
            "Source ledger",
            ["component", "relative path", "audit/schema note"],
            provenance_rows,
        )
    }
{
        details_table(
            "FileStream raw/derived roots",
            ["campaign", "FileStream path", "role"],
            [
                [
                    "V4.4 representation source",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3",
                    "raw generations/captures and frozen stimuli",
                ],
                [
                    "Qwen V4.4.4 natural/read-write",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_natural_ov/run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1",
                    "Qwen natural-OV plus derivative analyses",
                ],
                [
                    "correct-only frozen top-k",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260804_v4_4_ablation_seed_extrapolation_qwen_n2_n4_gemma_n1_n2",
                    "Qwen/Gemma clean baseline, ranked/random ablation and audit",
                ],
                [
                    "Gemma natural/read-write",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_gemma_replication/v4_4_4_natural_ov/run_20260805_gemma_l37_h1_h2_frozen_v1",
                    "Gemma natural-OV and derivative α/V read-write",
                ],
                [
                    "Gemma fresh serial path",
                    "/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_gemma_replication/v4_4_4_serial_path/run_20260805_gemma_l29h4_l35h2_to_l37h1h2_fresh_v1",
                    "independent early→L37 mediation",
                ],
            ],
        )
    }
<p class="small">本报告只嵌入聚合统计与可视化，不复制 raw hidden states、full V tensors 或 raw attention rows；原始数据保留在 FileStream。Qwen/Gemma natural-OV、read/write 与 upstream audits 均须 PASS，builder 才会生成报告；Qwen relay audit 亦为 PASS。causal-v2 每模型 302/302 checks 通过，correct-only seed extrapolation audit 为 {
        seed_confirmation["audit"]["passed"]
    }/{seed_confirmation["audit"]["checks"]} PASS。</p>
<div class="conclusion"><strong>最终结论</strong>当前 non-thinking counting 最符合“分布式 running-index representation，经 broad retrieval 汇集，由 late set-level OV channel 写入可执行 answer count state”的机制。Qwen 的 early top-4→L28 H16–H19→answer 已独立复现；Gemma 的最强结论严格由 natural-OV、read/write 与 fresh-seed serial 三门联合结果决定。任何跨模型正结论都是 set-level 路径复制，而不是单头 counter 的证明。</div>
</section>
"""


EXTRA_CSS = r"""
.abstract{font-size:18px;line-height:1.72;max-width:96ch}.paper-table{font-size:13px}.paper-table td:first-child{font-weight:650;color:var(--indigo)}
.baseline-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:24px 0}.baseline-strip>div{background:var(--surface);border:1px solid var(--line);padding:14px 16px}.baseline-strip span,.baseline-strip small{display:block;color:var(--muted);font-size:12px}.baseline-strip strong{display:block;font:700 22px/1.3 Consolas,monospace;color:var(--indigo);margin:4px 0}
.evidence{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700;white-space:nowrap}.descriptive{background:#E7E2F7;color:#34257B}.functional{background:#E3F4F7;color:#075C6E}.confirmed{background:#DDF3E8;color:#155C41}.supported{background:#F3EDCF;color:#685613}.rejected{background:#F6DCE8;color:#7D204D}
.prompt-block{white-space:pre-wrap;overflow:auto;background:#15112B;color:#F8F4EA;padding:18px 20px;border-radius:5px;border:1px solid #312B4A;max-width:94ch}.prompt-block code{background:transparent;color:inherit;padding:0}.evidence-note{border-left-color:var(--violet)}
.paper-wording{background:#E9E3D8;border:1px solid var(--line);padding:18px 20px;margin:22px 0;line-height:1.7}.paper-wording strong{color:var(--indigo)}
.integrated-forest text,.write-trace text,.gate-svg text,.relay-svg text,.mechanism-svg text{font-family:"Segoe UI",Arial,sans-serif}.integrated-forest .grid,.write-trace .grid{stroke:#D7D0C5;stroke-width:1}.integrated-forest .zero{stroke:#20242D;stroke-width:1.5}.integrated-forest .ci{stroke-width:3}.integrated-forest .cap{stroke-width:2}.integrated-forest .dot{stroke:#FFFDF8;stroke-width:2}.integrated-forest .tick,.write-trace .tick{fill:#69717B;font-size:12px}.integrated-forest .row-label{fill:#20242D;font-size:14px;font-weight:650}.integrated-forest .value-label,.write-trace .value-label{fill:#4D5560;font:12px Consolas,monospace}.integrated-forest .axis-label,.write-trace .axis-label{fill:#303744;font-size:13px;font-weight:650}
.write-trace .trace-line{fill:none;stroke:#6750E8;stroke-width:3}.write-trace .trace-ci{stroke:#6750E8;stroke-width:2}.write-trace .trace-dot{fill:#6750E8;stroke:#FFFDF8;stroke-width:2}
.gate-svg .gate-box{fill:#F8F5EE;stroke:#BDB4A7;stroke-width:1.5}.gate-svg .gate-box.gate-fail{fill:#FFF4F8;stroke:#C96B91}.gate-svg .gate-check{fill:#00A88F}.gate-svg .gate-check.gate-fail{fill:#B73B70}.gate-svg .gate-check-text{fill:white;font-weight:700;font-size:18px}.gate-svg .gate-heading{fill:#23165C;font-weight:700;font-size:18px}.gate-svg .gate-main{fill:#20242D;font:15px Consolas,monospace}.gate-svg .gate-sub{fill:#5E6672;font-size:13px}.gate-svg .gate-p{fill:#08705E;font:13px Consolas,monospace;font-weight:700}
.relay-svg .relay-box{stroke-width:2}.relay-svg .relay-pass{fill:#E4F4EC;stroke:#00A88F}.relay-svg .relay-fail{fill:#F8E6EE;stroke:#D94B86}.relay-svg .relay-mark{font-size:24px;font-weight:700;fill:#20242D}.relay-svg .relay-heading{font-size:13px;font-weight:700;fill:#23165C}.relay-svg .relay-value,.relay-svg .relay-p{font:12px Consolas,monospace;fill:#303744}.relay-svg .relay-arrow{stroke:#718096;stroke-width:2}.relay-svg .relay-summary{font:14px Consolas,monospace;fill:#7D204D;font-weight:700}
.mechanism-svg .mech-node{fill:#F8F5EE;stroke:#6750E8;stroke-width:2}.mechanism-svg .mech-2,.mechanism-svg .mech-3{fill:#E9E4FA}.mechanism-svg .mech-4{fill:#E4F4EC;stroke:#00A88F}.mechanism-svg .mech-heading{font-size:15px;font-weight:700;fill:#23165C}.mechanism-svg .mech-sub{font-size:12px;fill:#4F5863}.mechanism-svg .mech-arrow{stroke:#6750E8;stroke-width:3}.mechanism-svg .mech-evidence,.mechanism-svg .mech-boundary{font-size:12px;fill:#5E6672}.mechanism-svg .mech-negative{fill:#EFECE6;stroke:#718096;stroke-width:1.5}.mechanism-svg .mech-dashed{stroke:#718096;stroke-width:2;stroke-dasharray:7 6}
.mechanism-main{background:linear-gradient(145deg,#F4F0E8 0%,#ECE8F8 58%,#E5F3EE 100%);border:1px solid #CFC6BA;padding:28px;margin-top:24px}.main-figure-kicker{font:700 11px/1.4 Consolas,monospace;letter-spacing:.13em;color:#6750E8;margin-bottom:8px}.mechanism-walkthrough{margin-top:18px}.mechanism-canvas-wrap{background:#15112B;border:1px solid #302A49;overflow:auto}.mechanism-canvas-wrap svg{display:block;min-width:980px;width:100%;height:auto;color:#6750E8}.mechanism-canvas-wrap text{font-family:"Segoe UI",Arial,sans-serif}.walk-input rect,.walk-node rect{fill:#211B3D;stroke:#5D557B;stroke-width:2;transition:fill .28s ease,stroke .28s ease,filter .28s ease}.walk-input circle,.walk-node circle{fill:#6750E8;stroke:#F8F5EE;stroke-width:1.5}.walk-title{fill:#F6F2E8;font-size:15px;font-weight:700}.walk-token,.walk-head,.walk-formula{fill:#CFC8E7;font-size:13px}.walk-sub,.walk-boundary{fill:#9DA2B4;font-size:11px}.mini-manifold,.fan-line{fill:none;stroke:#70688E;stroke-width:3}.walk-edge{fill:none;stroke:#58516F;stroke-width:4;color:#58516F;transition:stroke .28s ease,color .28s ease}.walk-input.is-complete rect,.walk-node.is-complete rect{fill:#29224D;stroke:#8D7FF1}.walk-input.is-active rect,.walk-node.is-active rect{fill:#3B2E74;stroke:#D6B52C;filter:drop-shadow(0 0 12px rgba(214,181,44,.42))}.walk-node.is-active .mini-manifold,.walk-node.is-active .fan-line{stroke:#D6B52C}.walk-node.walk-output.is-active rect{fill:#124A42;stroke:#2DBE77}.walk-edge.is-complete{stroke:#6750E8;color:#6750E8}.walk-edge.is-active{stroke:#D6B52C;color:#D6B52C;stroke-dasharray:9 7;animation:walk-dash .8s linear infinite}.walk-answer{fill:#F8F5EE;font:700 18px Consolas,monospace}.walk-answer-number{fill:#D6B52C;font:800 36px Consolas,monospace}.mechanism-controls,.running-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:14px 0}.mechanism-controls button,.running-controls button,.step-dots button{border:1px solid #9188A6;background:#FFFDF8;color:#23165C;padding:8px 12px;border-radius:4px;font-weight:650;cursor:pointer}.mechanism-controls button:hover,.running-controls button:hover,.step-dots button:hover{background:#EEE9FA}.step-dots{display:flex;gap:5px;margin-left:auto}.step-dots button{width:34px;height:34px;padding:0;border-radius:50%}.step-dots button[aria-current="step"]{background:#6750E8;color:white;border-color:#6750E8}.mechanism-live{min-height:76px;background:#FFFDF8;border-left:4px solid #6750E8;padding:12px 16px;line-height:1.55}.mechanism-live strong{display:block;color:#23165C;margin-bottom:3px}.mechanism-live .live-evidence{font:12px Consolas,monospace;color:#08705E}
.running-index-block{margin-top:30px}.running-controls{background:#F4F0E8;border:1px solid #D7D0C5;padding:12px 14px}.running-controls label{display:flex;align-items:center;gap:8px;font-size:13px;color:#4F5863}.running-controls select{padding:7px 9px}.running-slider{flex:1;min-width:240px}.running-slider input{width:100%}.running-shell{height:560px}.running-shell canvas{width:100%;height:100%;display:block;touch-action:none}.running-status{font:13px Consolas,monospace;color:#4F5863;background:#F4F0E8;border:1px solid #D7D0C5;border-top:0;padding:9px 12px}
.plain-protocol{background:#F8F5EE;border:1px solid #CEC5B8;padding:16px 20px;margin:20px 0}.plain-protocol h4{margin:0 0 10px;color:#23165C}.plain-protocol li{margin:.55em 0;line-height:1.55}.sig-yes,.sig-no{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:750;white-space:nowrap}.sig-yes{background:#DDF3E8;color:#155C41}.sig-no{background:#F6DCE8;color:#7D204D}.test-card{border:1px solid #CFC7BA;background:#FFFDF8;padding:16px 18px;margin:16px 0}.test-card h4{margin:0 0 10px;color:#23165C}.test-card dl{display:grid;grid-template-columns:130px 1fr;gap:8px 14px;margin:0}.test-card dt{font-weight:700;color:#4E5661}.test-card dd{margin:0;line-height:1.55}.step-result{border-left:4px solid #6750E8;background:#F5F1FB;padding:12px 15px;margin:14px 0}.step-result strong{color:#23165C}.ablation-topk text{font-family:"Segoe UI",Arial,sans-serif}.ablation-topk .grid{stroke:#D7D0C5;stroke-width:1}.ablation-topk .x-guide{stroke:#D7D0C5;stroke-width:1;stroke-dasharray:4 5}.ablation-topk .axis{stroke:#2F3540;stroke-width:1.5}.ablation-topk .series-line{fill:none;stroke-width:3}.ablation-topk .series-dot{stroke:#FFFDF8;stroke-width:2}.ablation-topk .ci{stroke-width:2.5}.ablation-topk .cap{stroke-width:2}.ablation-topk .tick{fill:#68717C;font-size:12px}.ablation-topk .axis-label{fill:#303744;font-size:13px;font-weight:700}.ablation-topk .point-label{font-size:12px;font-weight:750}.ablation-topk .panel-title{fill:#23165C;font-size:15px;font-weight:750}.ablation-topk .legend-label{font-size:12px;font-weight:700}
@keyframes walk-dash{to{stroke-dashoffset:-32}}
@media(prefers-reduced-motion:reduce){.walk-input rect,.walk-node rect,.walk-edge{transition:none}.walk-edge.is-active{animation:none}}
@media(max-width:760px){.baseline-strip{grid-template-columns:1fr}.paper-table{min-width:760px}.integrated-forest,.write-trace,.gate-svg,.relay-svg,.mechanism-svg,.ablation-topk{min-width:760px}.figure-block,figure{overflow:auto}.mechanism-main{padding:18px}.step-dots{margin-left:0}.running-shell{height:460px}.test-card dl{grid-template-columns:1fr}.test-card dd{margin-bottom:8px}}
"""


EXTRA_JS = r"""
function makeMechanismWalkthrough(){
 const root=document.getElementById('mechanism-overview');if(!root)return;
 const nodes=[...root.querySelectorAll('[data-walk-step]')],edges=[...root.querySelectorAll('[data-walk-edge]')];
 const live=document.getElementById('mechanism-live'),prev=document.getElementById('mechanism-prev'),next=document.getElementById('mechanism-next'),play=document.getElementById('mechanism-play');
 const dots=[...root.querySelectorAll('[data-mechanism-step]')];
 const stages=[
  {title:'步骤 1/5 · 顺序读取重复 record',body:'模型以 non-thinking 模式读完整个 10k-token prompt。V4.4 随机化 needle 的位置、内容和顺序，因此后续稳定结构不能只由固定位置解释。',evidence:'设计事实；此步不单独进行显著性检验。'},
  {title:'步骤 2/5 · 形成 prompt running-index state',body:'每个 active needle 末端 residual 随 occurrence index n=1…10 沿有序 manifold 移动。删除开头定义提示后，相对拓扑仍高度保留，但 full-space state 会被调制。',evidence:'表征证据：PCA 只显示结构；显著性由 full-space tests 承担。'},
  {title:'步骤 3/5 · early broad bank 汇集 slot states',body:'冻结的 L23H28、L23H29、L26H20、L27H18 set-output donor patch 把 answer distribution 推向 donor count；随后阻断 L28 通道会特异削弱该效应。',evidence:'fresh seeds 1294–1313；serial-path conjunction IUT p=0.005884（显著）。'},
  {title:'步骤 4/5 · mixed read 经 OV 改换坐标并写回',body:'Qwen L28 H16/H19 的 pre-O z 同时依赖 routing α 与 value content V；W_O 把 head-space count state 写入 residual，所以 prompt count axis 与 answer count axis 不需要平行。H19 是测试 set 内非冗余锚点。',evidence:'Qwen routing/value p=9.54×10⁻⁷；natural-OV global IUT p=0.004541。Gemma localized OV 未闭合，只确认 distributed residual write。'},
  {title:'步骤 5/5 · late answer state 形成并驱动数字输出',body:'Qwen L28 的自然写入沿 L29–L35 的冻结 answer-count axes 保留；Gemma 的 K2 source bank 则写入 L37 count-aligned residual 并传到 L41。最终 Total: query residual 可搬运 donor prediction。',evidence:'Qwen L35 最大校正 p=2.29×10⁻⁵；Gemma residual-path IUT p=9.54×10⁻⁷；两模型 correct-only answer aggregate patch 均显著。'}
 ];
 let step=0,timer=null;
 function stop(){if(timer){clearInterval(timer);timer=null}play.textContent='▶ 播放一次';play.setAttribute('aria-pressed','false')}
 function render(){
  nodes.forEach(node=>{const i=+node.dataset.walkStep;node.classList.toggle('is-active',i===step);node.classList.toggle('is-complete',i<step)});
  edges.forEach(edge=>{const i=+edge.dataset.walkEdge;edge.classList.toggle('is-active',i===step);edge.classList.toggle('is-complete',i<step)});
  dots.forEach(dot=>{const active=+dot.dataset.mechanismStep===step;dot.setAttribute('aria-current',active?'step':'false')});
  prev.disabled=step===0;next.disabled=step===stages.length-1;
  live.innerHTML=`<strong>${stages[step].title}</strong><span>${stages[step].body}</span><span class="live-evidence">${stages[step].evidence}</span>`;
 }
 prev.addEventListener('click',()=>{stop();step=Math.max(0,step-1);render()});
 next.addEventListener('click',()=>{stop();step=Math.min(stages.length-1,step+1);render()});
 dots.forEach(dot=>dot.addEventListener('click',()=>{stop();step=+dot.dataset.mechanismStep;render()}));
 play.addEventListener('click',()=>{
  if(timer){stop();return}if(step===stages.length-1)step=0;render();play.textContent='❚❚ 暂停';play.setAttribute('aria-pressed','true');
  timer=setInterval(()=>{if(step>=stages.length-1){stop();return}step+=1;render()},1200);
 });
 render();
}

function makeRunningIndex(){
 const canvas=document.getElementById('running-index-canvas');if(!canvas)return;
 const ctx=canvas.getContext('2d'),model=document.getElementById('running-model'),slider=document.getElementById('running-step');
 const prev=document.getElementById('running-prev'),next=document.getElementById('running-next'),play=document.getElementById('running-play'),status=document.getElementById('running-status');
 let step=1,yaw=-.72,pitch=.43,zoom=1,drag=false,lastX=0,lastY=0,timer=null;
 function active(){const options=Object.values(PROMPT_DATA).filter(item=>item.model===model.value);return options.find(item=>item.manifold_display)||options[Math.floor(options.length/2)]}
 function centroids(item){const out=[];for(let n=1;n<=10;n++){const rows=item.rows.filter(row=>row[5]===n);const p=[0,1,2].map(pc=>rows.reduce((sum,row)=>sum+row[6+pc],0)/rows.length);out.push({n,p,rows})}return out}
 function geometry(item,w,h){
  const values=[0,1,2].map(pc=>item.rows.map(row=>row[6+pc])),mins=values.map(v=>Math.min(...v)),maxs=values.map(v=>Math.max(...v));
  const center=mins.map((v,i)=>(v+maxs[i])/2),common=Math.max(...mins.map((v,i)=>maxs[i]-v),1e-8),radius=Math.min(w,h)*.34*zoom;
  const project=p=>{let x=(p[0]-center[0])*2/common,y=(p[1]-center[1])*2/common,z=(p[2]-center[2])*2/common;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:w/2+x1*radius,y:h/2-y1*radius,z:z2}};
  return{project,center,common};
 }
 function line(a,b,stroke,width=1,dash=[]){ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([])}
 function draw(){
  const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,item=active();ctx.clearRect(0,0,w,h);ctx.fillStyle='#15112B';ctx.fillRect(0,0,w,h);if(!item)return;
  const cs=centroids(item),g=geometry(item,w,h),project=g.project,half=g.common*.47;
  const axisColors=['#8D7FF1','#52C4B1','#D6B52C'];
  for(let axis=0;axis<3;axis++){const a=[...g.center],b=[...g.center];a[axis]-=half;b[axis]+=half;const qa=project(a),qb=project(b);line(qa,qb,axisColors[axis],1.2,[4,6]);ctx.fillStyle=axisColors[axis];ctx.font='12px Consolas';ctx.fillText(`PC${axis+1}`,qb.x+5,qb.y-4)}
  const qcs=cs.map(c=>({...c,q:project(c.p)}));
  for(let i=1;i<qcs.length;i++)line(qcs[i-1].q,qcs[i].q,'rgba(244,240,232,.18)',2,[5,7]);
  for(let i=1;i<step;i++)line(qcs[i-1].q,qcs[i].q,'rgba(255,253,248,.88)',4);
  const current=cs[step-1],cloud=current.rows.map(row=>({row,q:project([row[6],row[7],row[8]])})).sort((a,b)=>a.q.z-b.q.z);
  for(const point of cloud){ctx.globalAlpha=.28;ctx.fillStyle=COUNT_COLORS[step-1];ctx.beginPath();ctx.arc(point.q.x,point.q.y,3.2,0,Math.PI*2);ctx.fill()}
  ctx.globalAlpha=1;
  for(const c of qcs){const reached=c.n<=step;ctx.globalAlpha=reached?1:.23;ctx.fillStyle=COUNT_COLORS[c.n-1];ctx.strokeStyle=reached?'#FFFDF8':'#746D87';ctx.lineWidth=c.n===step?3:1.4;ctx.beginPath();ctx.arc(c.q.x,c.q.y,c.n===step?10:6.5,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle=reached?'#FFFDF8':'#9CA0AE';ctx.font=c.n===step?'700 13px Segoe UI':'11px Segoe UI';ctx.fillText(String(c.n),c.q.x+10,c.q.y-9)}
  ctx.globalAlpha=1;ctx.fillStyle='#F8F5EE';ctx.font='700 16px Segoe UI';ctx.fillText(`running index n=${step}`,18,28);ctx.fillStyle='#A8ACB8';ctx.font='12px Consolas';ctx.fillText(`${model.value} · display L${item.layer} · actual V4.4 states`,18,48);
  const seeds=new Set(current.rows.map(row=>row[0])).size;status.textContent=`n=${step}/10 · ${model.value} · prompt manifold-display L${item.layer} · ${seeds} seeds · basis=${item.fit_variant} ${item.fit_split}`;
 }
 function resize(){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
 function stop(){if(timer){clearInterval(timer);timer=null}play.textContent='▶ 播放一次';play.setAttribute('aria-pressed','false')}
 function setStep(value){step=Math.max(1,Math.min(10,value));slider.value=String(step);prev.disabled=step===1;next.disabled=step===10;draw()}
 prev.addEventListener('click',()=>{stop();setStep(step-1)});next.addEventListener('click',()=>{stop();setStep(step+1)});slider.addEventListener('input',()=>{stop();setStep(+slider.value)});
 model.addEventListener('change',()=>{stop();setStep(1)});
 play.addEventListener('click',()=>{if(timer){stop();return}if(step===10)setStep(1);play.textContent='❚❚ 暂停';play.setAttribute('aria-pressed','true');timer=setInterval(()=>{if(step===10){stop();return}setStep(step+1)},750)});
 canvas.addEventListener('pointerdown',event=>{drag=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId)});
 canvas.addEventListener('pointermove',event=>{if(!drag)return;yaw+=(event.clientX-lastX)*.008;pitch=Math.max(-1.35,Math.min(1.35,pitch+(event.clientY-lastY)*.008));lastX=event.clientX;lastY=event.clientY;draw()});
 canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointercancel',()=>drag=false);
 canvas.addEventListener('wheel',event=>{event.preventDefault();zoom=Math.max(.55,Math.min(2.1,zoom*(event.deltaY>0?.92:1.08)));draw()},{passive:false});
 new ResizeObserver(resize).observe(canvas);setStep(1);
}
"""


CLEAR_CSS = r"""
.mechanism-clear{background:#F5F4EF;border-color:#C9C7BE;color:#252923}.mechanism-clear .main-figure-kicker{color:#27685F}.mechanism-clear h2{max-width:900px}.mechanism-clear-intro{max-width:980px;font-size:16px;line-height:1.7}.model-mechanism{margin:30px 0 38px;border-top:3px solid #27685F;padding-top:16px}.model-mechanism.gemma{border-top-color:#A66A45}.model-mechanism-header{display:grid;grid-template-columns:minmax(220px,.7fr) minmax(420px,1.8fr);gap:28px;align-items:start;margin-bottom:18px}.model-mechanism-header h3{margin:0;color:#202520;font-size:24px}.model-mechanism-header p{margin:0;line-height:1.65}.mechanism-step-list{list-style:none;margin:0;padding:0;border-top:1px solid #CBC9C0}.mechanism-step-list li{display:grid;grid-template-columns:52px minmax(180px,.65fr) minmax(340px,1.55fr) minmax(210px,.8fr);gap:18px;align-items:start;padding:18px 0;border-bottom:1px solid #D7D5CD}.mechanism-step-number{font:700 22px/1 Consolas,monospace;color:#27685F}.gemma .mechanism-step-number{color:#A66A45}.mechanism-step-title{font-weight:750;color:#202520}.mechanism-step-action,.mechanism-step-evidence{line-height:1.55}.mechanism-step-evidence{font:12px/1.55 Consolas,monospace;color:#3C625C}.mechanism-principle{border-top:1px solid #BDBBB2;padding-top:18px;margin-top:8px}.mechanism-principle .equation{margin:10px 0}.causal-roadmap{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border-top:1px solid #C8C6BD;border-bottom:1px solid #C8C6BD;margin:22px 0}.causal-roadmap>div{padding:16px 18px;border-right:1px solid #D5D3CB}.causal-roadmap>div:last-child{border-right:0}.causal-roadmap strong{display:block;margin-bottom:6px;color:#24302D}.causal-roadmap span{font-size:13px;line-height:1.5;color:#555E5A}.answer-patch-svg text{font-family:"Segoe UI",Arial,sans-serif}.answer-patch-svg .grid{stroke:#D7D5CD;stroke-width:1}.answer-patch-svg .axis{stroke:#343A37;stroke-width:1.5}.answer-patch-svg .tick{fill:#66706B;font-size:12px}.answer-patch-svg .panel-title{fill:#202520;font-size:15px;font-weight:700}.answer-patch-svg .axis-label{fill:#343A37;font-size:13px;font-weight:700}.answer-patch-svg .bar-label{fill:#202520;font:700 13px Consolas,monospace}.positive-mechanism-model{margin:26px 0 42px;border-top:3px solid #27685F;padding-top:16px}.positive-mechanism-model.gemma{border-top-color:#A66A45}.positive-mechanism-model h3{margin-top:0}.result-sentence{font-size:17px;line-height:1.7;max-width:980px}.compact-result-table td,.compact-result-table th{vertical-align:top}.scope-lines{border-top:1px solid #C8C6BD}.scope-line{display:grid;grid-template-columns:220px 1fr 190px;gap:20px;padding:15px 0;border-bottom:1px solid #D8D6CF}.scope-line strong{color:#202520}.scope-line .status{font:12px Consolas,monospace;color:#27685F}.provenance-note{font-size:11px;color:#727873;margin-top:24px}
@media(max-width:820px){.model-mechanism-header{grid-template-columns:1fr}.mechanism-step-list li{grid-template-columns:42px 1fr}.mechanism-step-action,.mechanism-step-evidence{grid-column:2}.causal-roadmap{grid-template-columns:1fr 1fr}.causal-roadmap>div:nth-child(2){border-right:0}.scope-line{grid-template-columns:1fr}.paper-table{min-width:720px}}
@media(max-width:520px){.causal-roadmap{grid-template-columns:1fr}.causal-roadmap>div{border-right:0;border-bottom:1px solid #D5D3CB}}
.mechanism-paper-figure{margin:26px 0 38px}.mechanism-paper-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 12px}.mechanism-paper-controls button{border:1px solid #969A93;background:#FBFAF5;color:#252923;padding:7px 11px;border-radius:4px;cursor:pointer;font-weight:650}.mechanism-paper-controls button:disabled{opacity:.42;cursor:default}.mechanism-paper-dots{display:flex;gap:5px;margin-left:auto}.mechanism-paper-dots button{width:32px;height:32px;padding:0;border-radius:50%}.mechanism-paper-dots button[aria-current="step"]{background:#27685F;color:#FFF;border-color:#27685F}.mechanism-paper-svg{display:block;width:100%;height:auto;min-width:960px;background:#FBFAF5;border:1px solid #C9C7BE}.mechanism-paper-svg text{font-family:"Segoe UI",Arial,sans-serif}.mechanism-paper-svg .lane-label{font-size:20px;font-weight:750;fill:#202520}.mechanism-paper-svg .lane-sub{font-size:12px;fill:#66706B}.mechanism-paper-svg .paper-node rect{fill:#F0F1EC;stroke:#A9ADA5;stroke-width:1.5;transition:fill .24s ease,stroke .24s ease,filter .24s ease}.mechanism-paper-svg .paper-node .node-title{font-size:14px;font-weight:700;fill:#252923}.mechanism-paper-svg .paper-node .node-sub{font-size:11px;fill:#606862}.mechanism-paper-svg .paper-edge{stroke:#AEB2AB;stroke-width:2.5;fill:none;transition:stroke .24s ease,stroke-width .24s ease}.mechanism-paper-svg .paper-node.is-complete rect{fill:#E5F0EC;stroke:#6A958A}.mechanism-paper-svg .paper-edge.is-complete{stroke:#6A958A}.mechanism-paper-svg .gemma-node.is-complete rect{fill:#F3EAE3;stroke:#B78767}.mechanism-paper-svg .gemma-edge.is-complete{stroke:#B78767}.mechanism-paper-svg .paper-node.is-active rect{fill:#DCECE7;stroke:#27685F;stroke-width:3;filter:drop-shadow(0 3px 5px rgba(39,104,95,.2))}.mechanism-paper-svg .gemma-node.is-active rect{fill:#F3E2D7;stroke:#A66A45;filter:drop-shadow(0 3px 5px rgba(166,106,69,.2))}.mechanism-paper-svg .paper-edge.is-active{stroke:#27685F;stroke-width:4}.mechanism-paper-svg .gemma-edge.is-active{stroke:#A66A45}.mechanism-paper-svg .window-strip{font:11px Consolas,monospace;fill:#59615C}.mechanism-paper-svg .window-s{fill:#E7E7E1;stroke:#BFC1BA}.mechanism-paper-svg .window-f{fill:#D7E9E3;stroke:#579183}.mechanism-paper-svg .window-label{font:700 10px Consolas,monospace;fill:#303632}.mechanism-live-grid{display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid #C9C7BE;border-top:0}.mechanism-live-grid>div{padding:13px 16px;min-height:72px}.mechanism-live-grid>div+div{border-left:1px solid #D3D1C9}.mechanism-live-grid strong{display:block;margin-bottom:4px}.mechanism-live-grid span{font-size:13px;line-height:1.55;color:#4F5752}.mechanism-definitions{margin:30px 0}.mechanism-definitions h3{margin-bottom:8px}.mechanism-definition-grid{display:grid;grid-template-columns:190px 1fr;border-top:1px solid #C8C6BD}.mechanism-definition-grid>div{display:contents}.mechanism-definition-grid strong,.mechanism-definition-grid span{padding:13px 0;border-bottom:1px solid #D8D6CF;line-height:1.62}.mechanism-definition-grid strong{padding-right:22px;color:#24302D}.mechanism-definition-grid code{white-space:normal}.window-explainer{margin:20px 0 26px;padding:18px 0;border-top:2px solid #A66A45;border-bottom:1px solid #D1CEC5}.window-explainer h4{margin:0 0 10px;font-size:18px}.window-explainer-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px}.window-explainer-grid strong{display:block;margin-bottom:5px}.window-explainer-grid p{margin:0;line-height:1.62}.mechanism-step-action .formula-line{display:block;margin-top:6px;font:12px/1.55 Consolas,monospace;color:#38433E}.mechanism-index-note{font-size:12px;color:#68706B;margin-top:6px}
@media(max-width:820px){.mechanism-paper-figure{overflow:auto}.mechanism-live-grid{grid-template-columns:1fr}.mechanism-live-grid>div+div{border-left:0;border-top:1px solid #D3D1C9}.mechanism-definition-grid{grid-template-columns:1fr}.mechanism-definition-grid>div{display:block;border-bottom:1px solid #D8D6CF}.mechanism-definition-grid strong,.mechanism-definition-grid span{display:block;border-bottom:0;padding:8px 0}.window-explainer-grid{grid-template-columns:1fr}.mechanism-paper-dots{margin-left:0}}
"""


def _route_row(
    analysis: dict[str, Any], model: str, route: str
) -> dict[str, Any]:
    hits = [
        row
        for row in analysis["route_results"]
        if row["model_label"] == model and row["route"] == route
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Missing route row for {model}/{route}")
    return hits[0]


def answer_patch_comparison_svg(causal_v2: dict[str, Any]) -> str:
    models = ["Qwen3-8B", "Gemma4-E4B"]
    colors = {"Qwen3-8B": "#27685F", "Gemma4-E4B": "#A66A45"}
    all_values = {
        model: float(
            causal_v2["primary_confirmation_family_summary"][
                f"{model}::answer_patching"
            ]["mean_effect"]
        )
        for model in models
    }
    correct_values = {
        model: float(
            causal_v2["correct_interventions"]["patch_pooled"][
                f"{model}::answer_patching"
            ]["pooled_average_patching_acc"]
        )
        for model in models
    }
    width, height = 1040, 430
    top, bottom = 70, 80
    panels = [
        (70, 430, "All samples", "control-adjusted donor transport", all_values),
        (610, 360, "Correct-only", "donor-target adoption rate", correct_values),
    ]
    parts = [
        f'<svg class="stat-svg answer-patch-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="answer-patch-title answer-patch-desc">',
        '<title id="answer-patch-title">Answer-query patching under all-sample and correct-only estimands</title>',
        '<desc id="answer-patch-desc">The left panel shows mean control-adjusted donor transport for all samples. The right panel shows donor-target adoption probability when donor and receiver are both clean-correct.</desc>',
    ]
    for x0, panel_w, title, ylabel, values in panels:
        left, right = x0 + 62, x0 + panel_w - 18
        plot_h = height - top - bottom

        def y(value: float) -> float:
            return top + (1.0 - value) * plot_h

        parts.append(
            f'<text class="panel-title" x="{x0 + panel_w / 2:.1f}" y="30" text-anchor="middle">{html.escape(title)}</text>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y(tick)
            parts.append(
                f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end">{tick:.2f}</text>'
            )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{right}" y2="{height-bottom}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>'
        )
        centers = [left + (right-left)*0.32, left + (right-left)*0.72]
        bar_w = min(88.0, (right-left)*0.22)
        for model, xx in zip(models, centers, strict=True):
            value = values[model]
            yy = y(value)
            parts.append(
                f'<rect x="{xx-bar_w/2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{height-bottom-yy:.1f}" rx="4" style="fill:{colors[model]}"/>'
            )
            parts.append(
                f'<text class="bar-label" x="{xx:.1f}" y="{yy-10:.1f}" text-anchor="middle">{value:.3f}</text>'
            )
            parts.append(
                f'<text class="tick" x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle">{html.escape(model)}</text>'
            )
        parts.append(
            f'<text class="axis-label" x="{(left+right)/2:.1f}" y="{height-20}" text-anchor="middle">{html.escape(ylabel)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_mechanism_overview_clear(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
    causal_v2: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    q_patch = _route_row(correct_state, "Qwen3-8B", "answer_query_aggregate")
    g_patch = _route_row(correct_state, "Gemma4-E4B", "answer_query_aggregate")
    q_all = causal_v2["primary_confirmation_family_summary"]["Qwen3-8B::answer_patching"]
    g_all = causal_v2["primary_confirmation_family_summary"]["Gemma4-E4B::answer_patching"]
    q_correct = causal_v2["correct_interventions"]["patch_pooled"]["Qwen3-8B::answer_patching"]
    g_correct = causal_v2["correct_interventions"]["patch_pooled"]["Gemma4-E4B::answer_patching"]
    return f"""
<section id="mechanism-overview" class="mechanism-main mechanism-clear">
<div class="main-figure-kicker">NON-THINKING COUNTING · MODEL-SPECIFIC MECHANISMS</div>
<h2>模型怎样把 prompt 中的累计状态变成最终数字？</h2>
<p class="mechanism-clear-intro">两模型共享同一个计算问题：先在 prompt 中形成随已读取 occurrence 数量变化的 state，再把这个 state 汇集到 answer query。区别在于目前的因果定位粒度：Qwen 已定位到具体 L28 OV head set；Gemma 已定位到 broad head bank 写入 L37 residual 的分布式路径。下面每一行依次写明“模型在做什么”以及“哪项实验支持这一步”。</p>

<article class="model-mechanism qwen">
<div class="model-mechanism-header"><h3>Qwen3-8B</h3><p><strong>一句话机制：</strong>prompt running counter → early broad retrieval → L28 H16/H19 读取 → W<sub>O</sub> 改换坐标并写回 → L29–L35 answer state → 输出 <code>Total:N</code>。</p></div>
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">形成 running counter</span><span class="mechanism-step-action">每读完一个 active needle，needle-end residual 就更新一次；第 n 个 endpoint 的 state 编码“已经读到第 n 个 occurrence”。</span><span class="mechanism-step-evidence">frozen-basis prompt geometry · n=1…10 ordered trajectory</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">汇集多个 slot state</span><span class="mechanism-step-action">L23H28、L23H29、L26H20、L27H18 组成 early broad bank，把分散在 prompt positions 的累计信息送向后面的 answer computation。</span><span class="mechanism-step-evidence">fresh-seed serial IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">L28 读取 state</span><span class="mechanism-step-action">H16/H19 在 answer query 处形成 pre-O head state <code>z</code>。attention routing α 决定从哪些可访问位置取信息，V content 携带被取出的 count-related content。</span><span class="mechanism-step-evidence">routing p={fmt_p(read_write['primary_decision']['read_mode']['routing_family_p'])} · value p={fmt_p(read_write['primary_decision']['read_mode']['value_family_p'])}</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">OV 写成 answer direction</span><span class="mechanism-step-action"><code>w=ΣW<sub>O</sub><sup>h</sup>z<sub>h</sub></code> 把 head-space state 写回 residual。这里发生坐标变换，所以 prompt counter direction 与 answer counter direction 不必平行。</span><span class="mechanism-step-evidence">natural OV global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">形成并读取 answer state</span><span class="mechanism-step-action">写入后的 count component 沿 L29–L35 保留；最终 <code>Total:</code> query residual 驱动数字 token。把 donor answer state patch 给 receiver，会把输出推向 donor count。</span><span class="mechanism-step-evidence">all-sample transport={q_all['mean_effect']:.3f} · correct-only adoption={100*q_correct['pooled_average_patching_acc']:.1f}% · fresh low-count p={fmt_p(q_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<article class="model-mechanism gemma">
<div class="model-mechanism-header"><h3>Gemma4-E4B</h3><p><strong>一句话机制：</strong>prompt running counter → L29H4/L35H2 broad bank → 写入 L37 count-aligned residual → residual 传播至 L41 → answer query 输出 <code>Total:N</code>。</p></div>
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">形成 running counter</span><span class="mechanism-step-action">与 Qwen 相同，Gemma 在每个 active needle endpoint 更新一个有序 prompt-side state；它表示已读取 occurrence 的累计进度。</span><span class="mechanism-step-evidence">frozen-basis prompt geometry · n=1…10 ordered trajectory</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">broad bank 汇集可访问 state</span><span class="mechanism-step-action">L29H4 与 L35H2 组成冻结 K2 bank。由于 sliding/local attention，后层不必直接看到原始远端 needles；bank 读取的是进入当前窗口前已经形成的可访问 state。</span><span class="mechanism-step-evidence">fresh top-k ablation · K1/K2 clean-correct Holm-significant</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">写入 L37 residual</span><span class="mechanism-step-action">patch K2 source bank 会在 L37 产生 count-aligned residual change；精确删除这部分 change 会特异削弱 donor-count transport。</span><span class="mechanism-step-evidence">source + exact/count-axis mediation · global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">传播到 terminal layer</span><span class="mechanism-step-action">L37 的分布式 count state 沿 residual stream 传播至 L41，并提高 donor count 在 terminal answer distribution 中的采用程度。</span><span class="mechanism-step-evidence">terminal adoption p={fmt_p(gemma_residual['primary_decision']['families']['terminal_count_adoption']['intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">answer query 读出数字</span><span class="mechanism-step-action">最终 answer-query state 已包含可执行的 count prediction；把 donor aggregate state 搬到 receiver，会显著推动 receiver 采用 donor count。</span><span class="mechanism-step-evidence">all-sample transport={g_all['mean_effect']:.3f} · correct-only adoption={100*g_correct['pooled_average_patching_acc']:.1f}% · fresh low-count p={fmt_p(g_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<div class="mechanism-principle"><strong>两模型共享的表示原则。</strong><div class="equation">prompt count direction u<sub>P</sub> → head state z → residual write w=ΣW<sub>O</sub>z → downstream answer direction u<sub>A</sub>∝Jw</div><p>语义上保持的是 count ordering 与 causal transport，不是同一欧氏方向。因此 3D PCA 中 prompt counter 与 answer counter 可以旋转、缩放或压缩，而仍然实现同一个计数变量。</p></div>
</section>
    """


def build_mechanism_overview_detailed(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
    causal_v2: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    q_patch = _route_row(correct_state, "Qwen3-8B", "answer_query_aggregate")
    g_patch = _route_row(correct_state, "Gemma4-E4B", "answer_query_aggregate")
    q_all = causal_v2["primary_confirmation_family_summary"]["Qwen3-8B::answer_patching"]
    g_all = causal_v2["primary_confirmation_family_summary"]["Gemma4-E4B::answer_patching"]
    q_correct = causal_v2["correct_interventions"]["patch_pooled"]["Qwen3-8B::answer_patching"]
    g_correct = causal_v2["correct_interventions"]["patch_pooled"]["Gemma4-E4B::answer_patching"]
    return f"""
<section id="mechanism-overview" class="mechanism-main mechanism-clear">
<div class="main-figure-kicker">NON-THINKING COUNTING · MODEL-SPECIFIC MECHANISMS</div>
<h2>模型如何从分散的 needle 形成、读取并写出 count state？</h2>
<p class="mechanism-clear-intro">这里把“mechanism”拆成四件可测量的事：状态在什么 token/layer 提取、count direction 怎样拟合、attention head 怎样读取并经 <em>W</em><sub>O</sub> 写回、以及替换或阻断该状态是否改变数字输出。Qwen 与 Gemma 的数学对象相同，但 Gemma 的 512-token sliding window 与周期性 full-attention layers 使信息流具有明显的“全局刷新—局部传递”节奏。</p>

<figure class="mechanism-paper-figure" id="mechanism-paper-figure" aria-labelledby="mechanism-paper-caption">
<div class="mechanism-paper-controls" aria-label="Mechanism step controls">
<button type="button" id="mechanism-prev">← 上一步</button>
<button type="button" id="mechanism-play" aria-pressed="false">▶ 播放一次</button>
<button type="button" id="mechanism-next">下一步 →</button>
<div class="mechanism-paper-dots" role="group" aria-label="直接选择机制步骤">
<button type="button" data-mechanism-step="0" aria-label="步骤 1">1</button><button type="button" data-mechanism-step="1" aria-label="步骤 2">2</button><button type="button" data-mechanism-step="2" aria-label="步骤 3">3</button><button type="button" data-mechanism-step="3" aria-label="步骤 4">4</button><button type="button" data-mechanism-step="4" aria-label="步骤 5">5</button>
</div></div>
<svg class="mechanism-paper-svg" viewBox="0 0 1220 610" role="img" aria-labelledby="mechanism-svg-title mechanism-svg-desc">
<title id="mechanism-svg-title">Qwen and Gemma non-thinking counting mechanisms</title>
<desc id="mechanism-svg-desc">Two aligned lanes show prompt running-index extraction, source-head retrieval, answer-query pre-output states, residual writing and propagation, and numerical output. Gemma additionally shows periodic full-attention layers separated by 512-token sliding-attention layers.</desc>
<defs><marker id="mechanism-arrow-q" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="#6A958A"/></marker><marker id="mechanism-arrow-g" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="#B78767"/></marker></defs>
<text class="lane-label" x="22" y="48">Qwen3-8B</text><text class="lane-sub" x="22" y="68">all layers can address the full causal prefix</text>
<g class="paper-node qwen-node" data-mechanism-stage="0"><rect x="22" y="92" width="190" height="112" rx="8"/><text class="node-title" x="117" y="122" text-anchor="middle">Prompt running index</text><text class="node-sub" x="117" y="148" text-anchor="middle">needle-end hᴾ(s,n,ℓ)</text><text class="node-sub" x="117" y="169" text-anchor="middle">n = 1 … 10</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="1" d="M216 148 L260 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="1"><rect x="268" y="92" width="190" height="112" rx="8"/><text class="node-title" x="363" y="119" text-anchor="middle">Early slot-state bank</text><text class="node-sub" x="363" y="145" text-anchor="middle">L23H28 · L23H29</text><text class="node-sub" x="363" y="166" text-anchor="middle">L26H20 · L27H18</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="2" d="M462 148 L506 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="2"><rect x="514" y="92" width="190" height="112" rx="8"/><text class="node-title" x="609" y="119" text-anchor="middle">L28 H16/H19 read</text><text class="node-sub" x="609" y="145" text-anchor="middle">z = Σⱼ α(q,j)V(j)</text><text class="node-sub" x="609" y="166" text-anchor="middle">routing + value content</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="3" d="M708 148 L752 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="3"><rect x="760" y="92" width="190" height="112" rx="8"/><text class="node-title" x="855" y="119" text-anchor="middle">OV residual write</text><text class="node-sub" x="855" y="145" text-anchor="middle">w = Σₕ Wᴼʰ zₕ</text><text class="node-sub" x="855" y="166" text-anchor="middle">propagate through L29–L35</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="4" d="M954 148 L998 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="4"><rect x="1006" y="92" width="190" height="112" rx="8"/><text class="node-title" x="1101" y="122" text-anchor="middle">Executable answer state</text><text class="node-sub" x="1101" y="148" text-anchor="middle">Total: query → LM head</text><text class="node-sub" x="1101" y="169" text-anchor="middle">greedy digit N</text></g>

<text class="lane-label" x="22" y="272">Gemma4-E4B</text><text class="lane-sub" x="22" y="292">five sliding layers, then one full-attention layer; window W = 512</text>
<g class="paper-node gemma-node" data-mechanism-stage="0"><rect x="22" y="316" width="190" height="112" rx="8"/><text class="node-title" x="117" y="346" text-anchor="middle">Prompt running index</text><text class="node-sub" x="117" y="372" text-anchor="middle">needle-end hᴾ(s,n,ℓ)</text><text class="node-sub" x="117" y="393" text-anchor="middle">local state + periodic refresh</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="1" d="M216 372 L260 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="1"><rect x="268" y="316" width="190" height="112" rx="8"/><text class="node-title" x="363" y="343" text-anchor="middle">Global K2 source bank</text><text class="node-sub" x="363" y="369" text-anchor="middle">L29H4 · L35H2</text><text class="node-sub" x="363" y="390" text-anchor="middle">both full-attention layers</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="2" d="M462 372 L506 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="2"><rect x="514" y="316" width="190" height="112" rx="8"/><text class="node-title" x="609" y="343" text-anchor="middle">Donor pre-O z patch</text><text class="node-sub" x="609" y="369" text-anchor="middle">replace z₍₂₉,₄₎ and z₍₃₅,₂₎</text><text class="node-sub" x="609" y="390" text-anchor="middle">model Wᴼ performs the write</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="3" d="M708 372 L752 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="3"><rect x="760" y="316" width="190" height="112" rx="8"/><text class="node-title" x="855" y="343" text-anchor="middle">Distributed residual path</text><text class="node-sub" x="855" y="369" text-anchor="middle">L37 count-aligned mediator</text><text class="node-sub" x="855" y="390" text-anchor="middle">residual carry → L41</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="4" d="M954 372 L998 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="4"><rect x="1006" y="316" width="190" height="112" rx="8"/><text class="node-title" x="1101" y="346" text-anchor="middle">Executable answer state</text><text class="node-sub" x="1101" y="372" text-anchor="middle">L41 query → LM head</text><text class="node-sub" x="1101" y="393" text-anchor="middle">greedy digit N</text></g>

<text class="window-strip" x="22" y="476">Gemma layer schedule near the causal path:</text>
<g transform="translate(268 452)"><rect class="window-s" x="0" y="0" width="50" height="34"/><text class="window-label" x="25" y="22" text-anchor="middle">S28</text><rect class="window-f" x="54" y="0" width="50" height="34"/><text class="window-label" x="79" y="22" text-anchor="middle">F29</text><rect class="window-s" x="108" y="0" width="50" height="34"/><text class="window-label" x="133" y="22" text-anchor="middle">S30</text><rect class="window-s" x="162" y="0" width="50" height="34"/><text class="window-label" x="187" y="22" text-anchor="middle">S31</text><text class="window-label" x="236" y="22" text-anchor="middle">…</text><rect class="window-f" x="264" y="0" width="50" height="34"/><text class="window-label" x="289" y="22" text-anchor="middle">F35</text><rect class="window-s" x="318" y="0" width="50" height="34"/><text class="window-label" x="343" y="22" text-anchor="middle">S36</text><rect class="window-s" x="372" y="0" width="50" height="34"/><text class="window-label" x="397" y="22" text-anchor="middle">S37</text><text class="window-label" x="446" y="22" text-anchor="middle">…</text><rect class="window-f" x="474" y="0" width="50" height="34"/><text class="window-label" x="499" y="22" text-anchor="middle">F41</text></g>
<text class="lane-sub" x="268" y="516">S: query sees only its previous 512-token causal window · F: query can address the full causal prefix</text>
<text class="lane-sub" x="22" y="562">Boxes name the localization granularity supported by the experiments; arrow length and box size do not encode effect size.</text>
</svg>
<div class="mechanism-live-grid" aria-live="polite"><div><strong id="mechanism-live-q-title"></strong><span id="mechanism-live-q-body"></span></div><div><strong id="mechanism-live-g-title"></strong><span id="mechanism-live-g-body"></span></div></div>
<figcaption id="mechanism-paper-caption"><strong>Figure 1 · Stepwise non-thinking counting mechanism.</strong> 上下两条 lane 使用相同的五步语义，但不是同一组 heads。Qwen 的读取/写入被定位到 L28 H16/H19；Gemma 的 K2 source heads 位于周期性 full-attention layers L29/L35，随后由 L37 residual mediator 将其影响传到 L41。图没有数值坐标轴；高亮仅表示当前讲解步骤。</figcaption>
</figure>

<div class="mechanism-definitions">
<h3>0.1 先定义“提取 state”和“计算 count direction”</h3>
<div class="mechanism-definition-grid">
<div><strong>Prompt state</strong><span>对 seed <em>s</em> 的同一个 N=10 prompt，定位第 <em>n</em> 个 active needle 的最后一个 token <code>t<sup>end</sup><sub>s,n</sub></code>，保存第 ℓ 个 decoder block 后的完整 residual：<code>h<sup>P</sup><sub>s,n,ℓ</sub>=h<sub>ℓ</sub>(t<sup>end</sup><sub>s,n</sub>)</code>。所以 n=1…10 是同一条 prompt 内的读取进度，不是十条不同 prompt。</span></div>
<div><strong>Answer state</strong><span>对 gold count 为 N 的 prompt，在生成第一个答案 token 之前，保存 prompt-final <code>Total:</code> query 的 post-block residual：<code>h<sup>A</sup><sub>s,N,ℓ</sub></code>。这一位置之后直接连接 LM head，因此 answer patching 在此处检验“state 是否可执行”。</span></div>
<div><strong>Count step</strong><span>在独立 discovery seeds 上对完整 residual 做逐维 OLS：<code>b<sub>ℓ</sub>=Σ<sub>i</sub>(c<sub>i</sub>−c̄)(h<sub>i,ℓ</sub>−h̄<sub>ℓ</sub>)/Σ<sub>i</sub>(c<sub>i</sub>−c̄)²</code>。它表示 count 增加 1 时 residual 的平均向量变化；单位轴为 <code>u<sub>ℓ</sub>=b<sub>ℓ</sub>/||b<sub>ℓ</sub>||</code>。PCA 只把 frozen discovery basis 投影成 3D，不参与因果干预。</span></div>
<div><strong>Head read/write</strong><span>在 query q，<code>α<sub>h</sub>(q,j)=softmax(q<sub>h</sub>k<sub>j</sub>/√d)</code>，pre-O state 为 <code>z<sub>h</sub>(q)=Σ<sub>j</sub>α<sub>h</sub>(q,j)V<sub>g(h)</sub>h(j)</code>，写回为 <code>o<sub>h</sub>(q)=W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q)</code>。α 回答“从哪里取”，V 回答“取到什么”，W<sub>O</sub> 回答“以什么 residual direction 写回”。</span></div>
<div><strong>Causal transport</strong><span>head/path intervention 用 gold-count 间距归一化：<code>T<sub>E</sub>=[E(C)<sub>I</sub>−E(C)<sub>R</sub>]/(D−R)</code>。answer-query full-state patch 则以两条自然预测的间距归一化：<code>T<sub>pred</sub>=(y<sub>patch</sub>−y<sub>R</sub>)/(y<sub>D</sub>−y<sub>R</sub>)</code>，只在两端预测均为有效且不同的数字时定义。correct-only 分析要求 donor 与 receiver 的原始输出都正确，再计算 patch 后采用 donor gold count 的比例。</span></div>
</div><p class="mechanism-index-note">除非另行说明，本报告中的 layer/head index 均为 zero-based；所有 axes、head sets 与 mediator layer 都在 confirmation outcome 之前冻结。</p>
</div>

<article class="model-mechanism qwen">
<div class="model-mechanism-header"><h3>Qwen3-8B：从全上下文读取到局部 OV 写入</h3><p><strong>完整路径：</strong>needle-end running state → early slot-state bank → L28 H16/H19 的 α/V mixed read → H16/H19 自身 W<sub>O</sub> 写回 → L29–L35 answer-query count state → <code>Total:N</code>。</p></div>
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">在 needle endpoint 提取 running state</span><span class="mechanism-step-action">按上面的 <code>h<sup>P</sup><sub>s,n,ℓ</sub></code> 定义，在同一 N=10 prompt 中依次读取十个 endpoint。对 discovery seeds 拟合 <code>b<sup>P</sup><sub>ℓ</sub></code>；V4.4 states 只投影进冻结的 V4.1 basis。它检验的是“读到第 n 个 occurrence 后 residual 是否有序变化”，不假设神经元里存在字面整数 n。<span class="formula-line">hᴾ(s,1,ℓ) → hᴾ(s,2,ℓ) → … → hᴾ(s,10,ℓ)</span></span><span class="mechanism-step-evidence">prompt full-space CV R²：L1 0.989；display L8 0.982</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">early bank 改写 slot states</span><span class="mechanism-step-action">在 discovery 中用 <code>mean(total needle-end attention mass × relative coverage)</code> 排序 broad-retrieval heads，并冻结 L23H28、L23H29、L26H20、L27H18。因果确认只在注册的 slot-query positions 把这些 heads 的 donor pre-O <code>z</code> 写入 receiver，再测量后续 L28 与答案分布的变化。随后把 L28 induced <code>Δz</code> 精确恢复到 receiver baseline；若 donor shift 被特异消除，则建立 early bank → L28 的串联关系。</span><span class="mechanism-step-evidence">fresh-seed serial-path IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">L28 H16/H19 同时用 routing 与 value 读取</span><span class="mechanism-step-action">在 answer query q 保存 receiver/donor 的全部 α 与 V，并构造四个 pre-O endpoint：RR、RD、DR、DD；第一个字母表示 α 来源，第二个表示 V 来源。Shapley 分解把同一个 donor movement 精确分账：<span class="formula-line">Δz<sub>value</sub>=½[(z<sub>RD</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>DR</sub>)]</span><span class="formula-line">Δz<sub>route</sub>=½[(z<sub>DR</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>RD</sub>)]</span>两个分量都必须既推动 donor count，又通过冻结 natural-OV axis 才算自然读取。</span><span class="mechanism-step-evidence">routing family p={fmt_p(read_write['primary_decision']['read_mode']['routing_family_p'])} · value family p={fmt_p(read_write['primary_decision']['read_mode']['value_family_p'])}</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">真实 pre-O OV 写入并改变坐标</span><span class="mechanism-step-action">先把 prompt 单位 count step 投影到每个 GQA value path：<code>d<sub>z,h</sub>=W<sub>V</sub><sup>g(h)</sup>b<sup>P</sup></code>；再定义 set 的自然写入方向 <code>m<sub>S</sub>=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>d<sub>z,h</sub></code>。真实 injection 在 W<sub>O</sub> 之前做 <code>z<sub>h</sub>←z<sub>h</sub>+βd<sub>z,h</sub></code>；centered removal 从 <code>z−z₀</code> 中删除沿 <code>m<sub>S</sub></code> 的自然分量，并与同一 W<sub>O</sub> span、等 post-O 范数的正交控制比较。所有 residual 变化都必须经过 heads 自己的 W<sub>O</sub>。</span><span class="mechanism-step-evidence">natural OV global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">沿 frozen answer axes 传播并驱动数字</span><span class="mechanism-step-action">对每个 downstream layer 用 discovery answer-query states 拟合自然 count step <code>b<sup>A</sup><sub>ℓ</sub></code>，再计算注入差分在该轴上的系数：<span class="formula-line">a<sub>ℓ</sub>=&lt;[h<sub>ℓ</sub>(+β)−h<sub>ℓ</sub>(−β)]/(2β), b<sup>A</sup><sub>ℓ</sub>&gt;/||b<sup>A</sup><sub>ℓ</sub>||²</span>最后，用单层 full-residual donor patch 替换 receiver 的 <code>Total:</code> query state，并从原 receiver context 完整 greedy 生成；这检验的是可执行信息，不是 PCA 相似度。</span><span class="mechanism-step-evidence">all-sample transport={q_all['mean_effect']:.3f} · correct-only adoption={100*q_correct['pooled_average_patching_acc']:.1f}% · fresh p={fmt_p(q_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<article class="model-mechanism gemma">
<div class="model-mechanism-header"><h3>Gemma4-E4B：周期性全局读取与窗口内 residual 传递</h3><p><strong>完整路径：</strong>needle-end running state → full-attention L29H4/L35H2 K2 bank → donor pre-O z 经各自 W<sub>O</sub> 写回 → L37 distributed residual mediator → L41 terminal state → <code>Total:N</code>。</p></div>
<div class="window-explainer"><h4>512-token window 实际改变了什么？</h4><div class="window-explainer-grid"><div><strong>Sliding layer S</strong><p>query q 只能访问 <code>j∈[max(0,q−511),q]</code>。在约 10k-token prompt 的末端，S layer 无法直接重新读取大部分远端 needles。</p></div><div><strong>Full layer F</strong><p>每六层出现一次 full attention；zero-based 为 L5、11、17、23、29、35、41。确认的 L29H4/L35H2 正好都在 F layers，因此能够在 answer query 直接汇集全 prompt。</p></div><div><strong>机制后果</strong><p>L35 先把全局信息写进 answer-query residual；之后 L36–L40 的 S layers 即使看不到远端 needles，residual skip 仍携带这个 state，并可结合近端 query context 局部变换；L41 再进行一次全局层更新。</p></div></div><p class="mechanism-index-note">这不是说 window 内的单个 token 必须保存完整整数；它说明已确认的因果路径具有“F layer 全局刷新 → S layer 在同一 query residual 上保持/变换 → terminal readout”的架构节奏。Gemma4 配置中的 <a href="https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json">layer_types 与 sliding_window</a>固定为本实验所用 revision。</p></div>
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">用同一定义提取 prompt running state</span><span class="mechanism-step-action">仍然保存每个 active needle 最后 token 的 post-block residual <code>h<sup>P</sup><sub>s,n,ℓ</sub></code>，并在独立 discovery seeds 上拟合完整空间 count step。因为 S layer 的感受野有限，某层 endpoint state 可能是局部累计、前面 full layer 的全局刷新以及 residual/MLP 变换的合成；ordered geometry 本身不把三者强行拆开。</span><span class="mechanism-step-evidence">prompt full-space CV R²：display L9 0.913；probe L22 0.932</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">冻结 full-attention K2 source bank</span><span class="mechanism-step-action">从预先排序的 correct-only broad-retrieval bank 冻结 L29H4 与 L35H2，并准备三个 layer-matched K2 controls。对 donor/receiver count pair，只在 receiver 的 answer-query pre-O slice 替换这两个 heads 的 <code>z<sub>h</sub></code>；其余 heads、tokens 与 receiver context 不变。因为 replacement 位于 W<sub>O</sub> 输入端，任何 downstream effect 都由 Gemma 自己的 output projections 写入。</span><span class="mechanism-step-evidence">fresh top-k ablation：K1/K2 correct-only Holm-significant</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">从候选 layers 中冻结 L37 residual mediator</span><span class="mechanism-step-action">在 discovery seeds 上先对自然 answer-query residual 拟合 <code>h<sub>ℓ</sub>(c)=a<sub>ℓ</sub>+c·b<sub>ℓ</sub></code>；再对 L36–L40 计算 K2 donor patch 引起的 <code>Δh<sub>ℓ</sub></code> 沿单位 count step 的平均投影。按 <code>mean&lt;Δh<sub>ℓ</sub>,u<sub>ℓ</sub>&gt;</code> 最大且 layer index 打破并列的预注册规则选择 L37，之后不再用 confirmation outcome 重选。</span><span class="mechanism-step-evidence">discovery seeds 1456–1465 · fit counts 1/3/5/7/9 · selected L37</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">用 exact block 与 count-axis block 验证传播</span><span class="mechanism-step-action">在 confirmation trial 中先测 source patch 在 L37 诱发的精确变化 <code>δ=h<sub>37</sub><sup>patch</sup>−h<sub>37</sub><sup>receiver</sup></code>。随后分别加入 <code>−δ</code>（exact block）或 <code>−proj<sub>b37</sub>(δ)</code>（count-axis block）；对照向量与被删分量等范数且正交。若 block 比对照更强地消除 donor log-odds gain，并同时降低 L41 沿 frozen count step 的 adoption，才认为 L37 中介 source-bank effect。</span><span class="mechanism-step-evidence">source + exact/count-axis mediation · global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">L41 terminal state 与完整答案输出</span><span class="mechanism-step-action">L41 的 state change 用 <code>&lt;Δh<sub>41</sub>,b<sub>41</sub>&gt;/||b<sub>41</sub>||²/(D−R)</code> 量化 donor count adoption；独立 answer-query full-state patch 再检验整个聚合 state 是否足以改变 greedy 数字。前者追踪 K2→L37→L41 的特定路径，后者验证最终 state 的可执行性。</span><span class="mechanism-step-evidence">terminal adoption p={fmt_p(gemma_residual['primary_decision']['families']['terminal_count_adoption']['intersection_union_p'])} · all-sample transport={g_all['mean_effect']:.3f} · correct-only adoption={100*g_correct['pooled_average_patching_acc']:.1f}% · fresh p={fmt_p(g_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<div class="mechanism-principle"><strong>为什么 prompt counter 与 answer counter 可以方向不同？</strong><div class="equation">u<sub>P</sub> → d<sub>z</sub>=W<sub>V</sub>u<sub>P</sub> → m<sub>S</sub>=ΣW<sub>O</sub>d<sub>z</sub> → u<sub>A</sub>∝J<sub>write→answer</sub>m<sub>S</sub></div><p>模型需要保持的是 count ordering、可解码性和 donor-directed causal transport，而不是让两个 token role 在欧氏空间共用一条直线。V projection、W<sub>O</sub> 与后续 attention/MLP Jacobian 都会旋转、缩放或压缩表示；所以判断“同一个 counter”应看 frozen-axis transport 与干预特异性，而不是要求两张 PCA 图视觉平行。</p></div>
</section>
"""


def build_scope_clear(
    causal_v2: dict[str, Any], ov: dict[str, Any], upstream: dict[str, Any], gemma_residual: dict[str, Any]
) -> str:
    return f"""
<section id="scope">
<h2>1 · 结论先行</h2>
<div class="scope-lines">
<div class="scope-line"><strong>Prompt representation</strong><span>Qwen 与 Gemma 都形成随 occurrence index 有序变化的 running-counter geometry。</span><span class="status">REPRESENTATION</span></div>
<div class="scope-line"><strong>Broad retrieval</strong><span>冻结 top-k bank 的 ablation 相对 layer-matched random heads 造成更大的计数行为变化。</span><span class="status">FUNCTIONAL</span></div>
<div class="scope-line"><strong>Qwen causal path</strong><span>early broad bank → L28 H16/H19 mixed α/V read → natural OV write → L35 answer state。</span><span class="status">IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}</span></div>
<div class="scope-line"><strong>Gemma causal path</strong><span>L29H4/L35H2 bank → L37 count-aligned residual → L41 answer state。</span><span class="status">IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}</span></div>
<div class="scope-line"><strong>Answer readout</strong><span>all-sample 与 correct-only answer-query patching 都显示 donor state 能推动 receiver 采用 donor count。</span><span class="status">CAUSAL PATCH</span></div>
</div>
<div class="conclusion"><strong>当前机制</strong>两模型都执行“prompt running state → distributed retrieval → coordinate-changing write → executable answer state”。Qwen 的写入已定位到具体 OV set；Gemma 的写入定位在 L37 residual-level。</div>
</section>
"""


def build_methods_clear(
    ov: dict[str, Any], upstream: dict[str, Any], gemma_residual: dict[str, Any]
) -> str:
    return f"""
<section id="methods">
<h2>2 · 实验设定与统计口径</h2>
<p>每个 stimulus 是约 10,000-token realistic haystack，包含十个可控 slots。non-thinking 条件关闭原生 thinking，并在 assistant 侧预填 <code>Total:</code>。V4.4 随 seed 随机化 needle 位置、内容与顺序；hidden state、attention 与 causal effects 都按模型分别分析。</p>
<div class="causal-roadmap">
<div><strong>Representation</strong><span>在冻结 PCA / full-space axis 上检验 prompt running state 与 answer state 是否携带 count。</span></div>
<div><strong>Ablation</strong><span>置零 ranked top-k head outputs，并减去三个 layer-matched random sets 的平均影响。</span></div>
<div><strong>Patching</strong><span>把 donor state 搬到 receiver，观察 receiver 是否向 donor count 移动。</span></div>
<div><strong>OV / mediation</strong><span>在真实 pre-O z-space 做 injection/removal，或精确阻断写入 residual 的自然 count component。</span></div>
</div>
{table(
    ["实验", "独立单位", "样本/seed 设计", "主判定"],
    [
        ["V4.4 representation", "seed", "30 V4.4 seeds；count 1–10", "冻结 basis 投影；full-space statistics"],
        ["Frozen top-k ablation", "seed", "20 fresh seeds；count 1–5；100 examples/model", "seed-cluster CI；exact sign flip；4-way Holm"],
        ["Answer-query patching", "seed / directed pair", "all-sample held-out confirmation + clean-correct supplement", "control-adjusted donor transport / donor-target adoption"],
        ["Qwen natural OV", "seed", "20 confirmation seeds；matched W<sub>O</sub>-span controls", f"four-family IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}"],
        ["Qwen serial path", "seed", "20 fresh seeds；route与head sets冻结", f"source+mediation IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}"],
        ["Gemma residual path", "seed", "20 confirmation seeds；candidate vs 3 matched banks", f"four-endpoint IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}"],
    ],
)}
<p><strong>统一统计单位：</strong>同一 seed 内的 counts、layers 与 donor pairs 先聚合，再跨 seed bootstrap 或 exact sign flip。图中的 point 是 seed-level effect mean，error bar 是 95% seed-cluster CI。多个必要条件组成一条机制时使用 intersection–union test：global p 取必要条件中最大的 p。</p>
<div class="conclusion"><strong>本节结论</strong>不同实验量纲不相加；机制由 representation、functional perturbation 与 causal mediation 在同一方向上收敛而建立。</div>
</section>
"""


def build_attention_estimand_note() -> str:
    return r"""
<div class="plain-protocol attention-estimand-note">
<h3>5.1 两种 key mass 回答的是不同问题</h3>
<p>两种图的 query 都固定为最终 <code>Total:</code> token；变化的只是 key pool。设第 <em>i</em> 条 active needle 的 token span 为 <code>S<sub>i</sub></code>、最后一个 token 为 <code>e<sub>i</sub></code>，head <em>h</em> 的 attention row 为 <code>α<sub>h</sub>(q,j)</code>：</p>
<div class="equation">endpoint-key mass: m<sub>i,h</sub><sup>end</sup>=α<sub>h</sub>(q,e<sub>i</sub>); &nbsp;&nbsp; full-span literal mass: m<sub>i,h</sub><sup>span</sup>=Σ<sub>j∈S<sub>i</sub></sub>α<sub>h</sub>(q,j).</div>
<p>对任一 pooling，令 <code>M<sub>h</sub>=Σ<sub>i</sub>m<sub>i,h</sub></code>、<code>p<sub>i,h</sub>=m<sub>i,h</sub>/M<sub>h</sub></code>、<code>C<sub>h</sub>=exp(−Σ<sub>i</sub>p<sub>i,h</sub>log p<sub>i,h</sub>)/N</code>；图中 broad score 是 <code>S<sub>h</sub>=M<sub>h</sub>C<sub>h</sub></code>。因此高分要求既有较大总 mass，又覆盖多个 occurrences，而不是只盯一个 needle。</p>
<div class="causal-roadmap">
<div><strong>Endpoint-key</strong><span>检验 answer query 是否直接读取我们预先定义的 needle-end running-state carrier。它是<strong>位置特异的 carrier-localization 指标</strong>，不是无意义，但会漏掉落在 needle 内其他 tokens 的读取。</span></div>
<div><strong>Full-span literal</strong><span>不预设 carrier 在 span 中的哪个 token；检验 head 是否从整条 needle 文本获得 attention evidence。它更适合作为<strong>通用 retrieval-head discovery</strong> 的主排序。</span></div>
</div>
<p><strong>长度敏感性。</strong>literal sum 会随 span token 数增加而机械变大；当前 needle 使用同一固定模板，因此它比 span mean 更贴近“分给整条 needle 的总注意力概率”，但正式重排仍应同时报告 token-length-adjusted sensitivity（例如按 span length 分层，或把 mass 对 token length 回归后使用残差）。这样可以确认 top heads 来自检索强度，而不是某些 city/score 文本恰好分词更长。</p>
<div class="conclusion"><strong>本报告的判定规则</strong>后面的 full-span 图默认首先显示，并作为未来 generic retrieval bank 的主筛选；endpoint 图保留为“是否集中读取 endpoint carrier”的二级定位诊断。现有 causal-v2 bank 是按历史 <code>span_end</code> 定义冻结并确认的，因此它仍然证明 endpoint-ranked bank 的功能贡献，但不能在不重排、不做 fresh-seed ablation 的情况下改称 full-span-ranked bank。</div>
</div>
"""


def build_causal_section_clear(
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    frozen_sets = {
        ("Qwen3-8B", "2"): "L27H18, L28H19",
        ("Qwen3-8B", "4"): "L27H18, L28H19, L23H29, L23H13",
        ("Gemma4-E4B", "1"): "L29H4",
        ("Gemma4-E4B", "2"): "L29H4, L35H2",
    }
    topk_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for k_text, metrics in sorted(
            seed_confirmation["models"][model].items(), key=lambda item: int(item[0])
        ):
            all_shift = metrics["all_absolute_shift"]
            correct = metrics["clean_correct_to_wrong"]
            topk_rows.append(
                [
                    model,
                    f"K={k_text}",
                    frozen_sets[(model, str(k_text))],
                    f"{fmt(all_shift['effect'], 4)} [{fmt(all_shift['ci95_low'], 4)}, {fmt(all_shift['ci95_high'], 4)}]",
                    f"{fmt(correct['effect'], 4)} [{fmt(correct['ci95_low'], 4)}, {fmt(correct['ci95_high'], 4)}]",
                    f"{fmt_p(correct.get('two_sided_exact_seed_sign_flip_p'))} / {fmt_p(correct.get('holm_p_across_four_frozen_sets'))}",
                ]
            )

    patch_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        all_patch = causal_v2["primary_confirmation_family_summary"][
            f"{model}::answer_patching"
        ]
        correct_patch = causal_v2["correct_interventions"]["patch_pooled"][
            f"{model}::answer_patching"
        ]
        fresh = _route_row(correct_state, model, "answer_query_aggregate")
        patch_rows.append(
            [
                model,
                fmt(all_patch["mean_effect"], 4),
                f"{int(all_patch['ci95_excludes_zero'])}/{int(all_patch['conditions'])}",
                f"{100 * correct_patch['pooled_average_patching_acc']:.1f}%",
                f"{fmt(fresh['source_donor_log_odds_gain_mean'], 4, signed=True)} "
                f"[{fmt(fresh['source_donor_log_odds_gain_ci95_low'], 4)}, {fmt(fresh['source_donor_log_odds_gain_ci95_high'], 4)}]",
                fmt_p(fresh["source_donor_log_odds_gain_p"]),
            ]
        )

    return f"""
<section id="causal">
<h2>6 · 因果实验：先定位功能 bank，再验证可执行 answer state</h2>
<h3>6.1 这部分只回答两个问题</h3>
<div class="causal-roadmap">
<div><strong>问题 A</strong><span>排序得到的 top-k heads 是否比同层随机 heads 更影响 counting？</span></div>
<div><strong>实验 A</strong><span>Ablate ranked set，并减去三个 layer-matched random sets 的平均影响。</span></div>
<div><strong>问题 B</strong><span>answer-query hidden state 是否已经包含能驱动数字输出的 count state？</span></div>
<div><strong>实验 B</strong><span>把 donor answer state patch 到 receiver，测 receiver 是否采用 donor count。</span></div>
</div>

<h3>6.2 Effect 的逐样本定义与跨 seed 聚合</h3>
<div class="test-card"><h4>Ablation：ranked bank 必须超过同层随机删除</h4><dl>
<dt>记号</dt><dd>对样本 <em>i</em>，<code>y<sub>0i</sub></code> 是 clean greedy count，<code>y<sub>Ki</sub></code> 是删除 ranked top-K 后的 count，<code>y<sub>Kir</sub><sup>rand</sup></code> 是第 <em>r</em> 个 layer-matched random set 的输出；本实验每个样本有 <code>R=3</code> 个随机 set。</dd>
<dt>All-sample absolute-shift effect</dt><dd><span class="formula-line">d<sub>i</sub><sup>abs</sup>=|y<sub>Ki</sub>−y<sub>0i</sub>|−R<sup>−1</sup>Σ<sub>r</sub>|y<sub>Kir</sub><sup>rand</sup>−y<sub>0i</sub>|</span>正值表示 ranked bank 的删除比删除同层随机 heads 更能改变模型实际生成的 count；它只度量变化大小，不把 over-count 与 under-count 抵消。</dd>
<dt>Correct-only failure effect</dt><dd>先限制到 clean 输出正确且格式有效的样本 <code>y<sub>0i</sub>=g<sub>i</sub></code>，再计算：<span class="formula-line">d<sub>i</sub><sup>fail</sup>=1[y<sub>Ki</sub>≠g<sub>i</sub>]−R<sup>−1</sup>Σ<sub>r</sub>1[y<sub>Kir</sub><sup>rand</sup>≠g<sub>i</sub>]</span>正值是 ranked bank 额外造成的 correct→wrong 概率。</dd>
<dt>Companion ΔMAE</dt><dd><span class="formula-line">d<sub>i</sub><sup>MAE</sup>=(|y<sub>Ki</sub>−g<sub>i</sub>|−|y<sub>0i</sub>−g<sub>i</sub>|)−R<sup>−1</sup>Σ<sub>r</sub>(|y<sub>Kir</sub><sup>rand</sup>−g<sub>i</sub>|−|y<sub>0i</sub>−g<sub>i</sub>|)</span>正值表示 ranked ablation 相对随机 ablation 额外增加绝对计数误差。</dd>
</dl></div>
<div class="test-card"><h4>Patching：把 receiver state 朝 donor count 搬运多少</h4><dl>
<dt>All-sample normalized transport</dt><dd>receiver/donor 的真实 counts 分别为 <code>R</code> 与 <code>D</code>，clean receiver 与 patch 后输出为 <code>y<sub>0</sub></code>、<code>y<sub>P</sub></code>：<span class="formula-line">T=(y<sub>P</sub>−y<sub>0</sub>)/(D−R)</span><code>T=1</code> 表示生成变化恰好覆盖完整 donor–receiver count gap，<code>T=0</code> 表示未沿 donor 方向移动，负值表示反向移动；无效生成在 strict estimand 中记为 0。主 effect 再减去同一样本 self-patch / same-count controls 的平均 transport：<span class="formula-line">T<sub>adj</sub>=T<sub>donor</sub>−mean(T<sub>control</sub>)</span></dd>
<dt>Correct-only donor adoption</dt><dd>只保留 donor 与 receiver 的 clean 输出都等于各自 gold count 的 pair；<span class="formula-line">A=1[y<sub>P</sub>=D]</span>表中百分比是所有 eligible pairs 的 pooled mean <code>mean(A)</code>，即 patch 后精确生成 donor gold count 的比例。</dd>
<dt>Fresh low-count source gain</dt><dd>对候选数字序列计算 donor-vs-receiver log-odds，令 <code>ℓ<sub>D</sub>−ℓ<sub>R</sub></code> 为 donor count 相对 receiver count 的优势：<span class="formula-line">G=([ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>patch</sub>−[ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>clean</sub>)</span>正值表示 patch 提高 donor count 的相对概率；这是 fresh correct-only path 实验的连续 endpoint。</dd>
</dl></div>
<p>所有逐样本值先在 seed 内平均，再把 seed 当作独立 cluster 求总体均值；95% CI 用 10,000 次 seed-cluster bootstrap。注册的 p 值来自 seed-level exact sign-flip；四个冻结 model×K 比较使用 Holm 校正。图中的点不是把 token、head 或 donor pair 当独立样本得到的。</p>

<h3>6.3 Top-k ablation：随着 K 增加，行为影响怎样变化？</h3>
<figure>{ablation_topk_svg(seed_confirmation)}<figcaption><strong>Figure · Frozen top-k ablation effect.</strong> 两图横轴都是冻结 head-set size K。左图纵轴是 ranked set 相对三个 layer-matched random sets 多造成的绝对 generated-count shift；右图纵轴是 clean-correct 样本中多造成的 correct-to-wrong rate。点为 20 个 seed means 的总体 effect，竖线为 seed-cluster bootstrap 95% CI。两模型使用各自冻结的 K 网格；连线只显示同模型随 K 的变化。</figcaption></figure>
{table(["model", "K", "frozen heads", "all-sample effect [95% CI]", "correct-only effect [95% CI]", "correct-only exact/Holm p"], topk_rows, classes="paper-table compact-result-table")}
<p>Qwen 从 K=2 增加到 K=4 后，all-sample effect 从 0.0367 增至 0.0567，clean-correct effect 从 0.0203 增至 0.0650。Gemma 的 K=1 与 K=2 均产生较大影响：all-sample effect 为 0.2433 与 0.2200，clean-correct effect 为 0.1231 与 0.1282。Gemma 两个 clean-correct 比较均通过四比较 Holm；Qwen K=4 的 companion ΔMAE Holm p=0.03125。</p>
<div class="conclusion"><strong>本小节结论</strong>两模型的 counting 都依赖分布式 ranked bank；K 的增加不是简单单调剂量，说明新增 heads 可能带来冗余或功能混合。</div>
<div class="callout warning"><strong>是否需要 ablate 更多 heads？</strong>有必要做一个<strong>有限、预冻结的 nested-K 补充实验</strong>，目的不是追求更大的 effect，而是区分“稀疏核心”“冗余 bank”与“加入无关 heads 后稀释”。建议用 full-span score 重新冻结排序，Qwen 取 K={{1,2,4,8,16}}，Gemma 取 K={{1,2,4,8}}，并在全新 seeds 上同时报告上面的 <code>d<sup>abs</sup></code>、<code>d<sup>fail</sup></code>、<code>d<sup>MAE</sup></code> 与 layer-matched random controls。若 effect 在某个 K 后平台或下降，应解释为范围边界，而不是继续扩大 K 直到显著。当前图只包含 Qwen K=2/4 与 Gemma K=1/2，不能据此外推完整 dose-response。</div>

<h3>6.4 Answer-query patching：all samples 与 correct-only</h3>
<figure>{answer_patch_comparison_svg(causal_v2)}<figcaption><strong>Figure · Answer-query state transport.</strong> 左图纵轴是 all-sample held-out confirmation 中，answer-query patch 相对 control 的平均 donor transport；右图纵轴是 donor 与 receiver 都 clean-correct 时，patch 后 receiver 生成 donor gold count 的比例。两个 panel 的统计量不同，因此只在各自 panel 内比较模型。</figcaption></figure>
{table(["model", "all-sample mean transport", "conditions with CI>0", "correct-only donor adoption", "fresh low-count source gain [95% CI]", "fresh p"], patch_rows, classes="paper-table compact-result-table")}
<p>all-sample estimand 中，Qwen 的平均 control-adjusted donor transport 为 0.7580，149/149 个冻结条件的 held-out CI 均大于 0；Gemma 为 0.7010，177/177 个条件均大于 0。clean-correct donor adoption 分别为 96.6% 与 96.0%。在另一组 20 fresh seeds、count 1–3 且 donor/receiver 都正确的实验中，Qwen 与 Gemma 的 answer aggregate source gain 也分别显著（p=1.43×10<sup>−5</sup> 与 9.54×10<sup>−7</sup>）。</p>
<div class="conclusion"><strong>本小节结论</strong>answer-query hidden state 不是只与 count 相关；它包含足以改变后续数字输出的可执行 count information，而且这一结果同时出现在全样本与 correct-only 分析中。</div>
</section>
"""


def _ov_component(
    ov: dict[str, Any], family: str, endpoint: str
) -> dict[str, Any]:
    hits = [
        item
        for item in ov["primary_decision"]["families"][family]["components"]
        if item["endpoint"] == endpoint
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Missing OV result {family}/{endpoint}")
    return hits[0]


def build_positive_mechanism_section(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
) -> str:
    q_rows = []
    for label, family, endpoint, meaning in (
        ("Natural carrier", "natural_signal", "natural_carrier_count_slope", "clean forward 中 H16/H19 output 随 count 有序变化"),
        ("True pre-O injection", "pre_o_injection", "injection_dose_slope", "在真实 z slice 加 natural step，输出按剂量向更高 count 移动"),
        ("Centered removal: error", "centered_removal", "removal_error_axis_minus_control", "删除自然 component 比等范数正交删除造成更大误差"),
        ("Centered removal: margin", "centered_removal", "removal_margin_axis_minus_control", "删除自然 component 特异降低正确 count margin"),
        ("Donor transport", "path_mediation", "donor_patch_transport", "donor z patch 把 receiver 推向 donor count"),
        ("Path mediation", "path_mediation", "mediation_control_minus_axis_block", "自然轴 block 比正交 block 消除更多 donor effect"),
    ):
        row = _ov_component(ov, family, endpoint)
        q_rows.append(
            [label, meaning, ci_text(row), fmt_p(row["p"])]
        )

    g_rows: list[list[str]] = []
    labels = {
        "source_donor_transport": ("Source transport", "K2 source-bank patch 把 answer computation 推向 donor count"),
        "exact_residual_mediation": ("Exact L37 mediation", "精确删除 patch-induced L37 residual change，特异削弱 transport"),
        "count_axis_mediation": ("Count-axis mediation", "只删除 L37 residual 中的 count-aligned component，同样削弱 transport"),
        "terminal_count_adoption": ("L41 adoption", "L37 写入提高 terminal layer 对 donor count 的采用"),
    }
    for family, document in gemma_residual["primary_decision"]["families"].items():
        core = next(
            item for item in document["components"] if item["role"] == "candidate_core"
        )
        label, meaning = labels[family]
        g_rows.append(
            [
                label,
                meaning,
                f"{fmt(core['mean'], 4)} [{fmt(core['ci95_low'], 4)}, {fmt(core['ci95_high'], 4)}]",
                fmt_p(core["p"]),
            ]
        )

    rw = read_write["primary_decision"]
    up = upstream["primary_decision"]
    return f"""
<section id="natural-ov">
<h2>7 · 已确认的写入与传播机制</h2>

<h3>7.1 共同的行为 readout 与“写入”判据</h3>
<p>所有连续行为 effect 都从同一组候选数字序列概率计算。若 <code>p(c)</code> 是候选 count <code>c</code> 的 sequence probability，则：</p>
<div class="equation">E[C]=Σ<sub>c</sub>c·p(c)/Σ<sub>c</sub>p(c); &nbsp;&nbsp; T=(E[C]<sub>I</sub>−E[C]<sub>R</sub>)/(D−R).</div>
<p><code>E[C]</code> 把整个数字候选分布压成可比较的 expected count；<code>T</code> 是 intervention 相对 donor–receiver count gap 的归一化运输。这里“head 写入 residual”不只是架构上存在 <code>W<sub>O</sub></code>：实验必须在真实 pre-O <code>z</code> 边界干预，让变化通过被选 heads 自己的 <code>W<sub>O</sub></code>，并证明自然方向比同一输出子空间、等 post-O 范数的正交方向更重要。传播则要求该写入造成的 residual change 在后续层仍可沿冻结 count axis 追踪，且阻断中间 residual component 会削弱行为运输。</p>
<div class="conclusion"><strong>判定标准</strong>pre-O sufficiency 说明“这个 channel 能写”；centered removal 说明“自然运行依赖它”；donor patch + 中介阻断说明“上游 effect 正通过它传递”；downstream frozen-axis trace 说明“写入没有在下一层立即消失”。四者承担不同命题，不能由一个显著 injection 互相替代。</div>

<article class="positive-mechanism-model qwen">
<h3>7.2 Qwen：L28 H16/H19 的 localized natural-OV write</h3>
<p class="result-sentence">H16/H19 在 answer query 的 pre-O head state 中读取 count-related content，并通过自身 <em>W</em><sub>O</sub> 把该 state 写入 residual。四类证据——自然载荷、pre-O 充分性、centered 必要性、donor-path mediation——全部显著，global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}。</p>
{table(["证据", "具体验证什么", "effect [95% CI]", "p"], q_rows, classes="paper-table compact-result-table")}
<div class="test-card"><h4>Qwen 的 effect 是怎样算出来的</h4><dl>
<dt>① 自然 carrier</dt><dd>用独立 center seeds 估计 count-neutral pre-O center <code>z<sub>0,S</sub></code>，并令 natural one-count output step 为 <code>m<sub>S</sub>=W<sub>O</sub><sup>S</sup>d<sub>S</sub></code>。自然系数为 <span class="formula-line">a<sub>S</sub>(z)=&lt;W<sub>O</sub><sup>S</sup>(z<sub>S</sub>−z<sub>0,S</sub>),m̂<sub>S</sub>&gt;/||m<sub>S</sub>||</span>表中的 carrier effect 是每个 seed 内 <code>a<sub>S</sub></code> 对 gold count 的 OLS slope；正值说明 clean forward 的 set output 自然携带递增 count。</dd>
<dt>② True pre-O injection</dt><dd>在 answer query 的真实 H16/H19 head slices 做 <code>z<sub>S</sub>←z<sub>S</sub>+βd<sub>S</sub></code>，不直接向 residual 加 answer axis。对每个 count 拟合 <code>ΔE[C]</code> 对剂量 <code>β</code> 的 slope，再在 seed 内平均；正 slope 表示变化经过自身 <code>W<sub>O</sub></code> 后按剂量提高 expected count。</dd>
<dt>③ Centered removal</dt><dd>从 <code>z<sub>S</sub>−z<sub>0,S</sub></code> 删除 natural component，并与同一 <code>W<sub>O</sub><sup>S</sup></code> span、等 post-O norm 的正交删除比较。error effect=<code>Δ|E[C]−gold|<sub>axis</sub>−Δ|E[C]−gold|<sub>orth</sub></code>，预期为正；margin effect=<code>Δmargin<sub>axis</sub>−Δmargin<sub>orth</sub></code>，预期为负。</dd>
<dt>④ Donor-path mediation</dt><dd>先把 donor 的 L28 pre-O <code>z<sub>S</sub></code> patch 给 receiver，得到 <code>T<sub>patch</sub></code>；随后分别阻断 natural axis 或加入等范数正交 control。中介 effect 为 <span class="formula-line">M=T<sub>orth</sub>−T<sub>natural-block</sub></span>正值表示阻断同一自然写入轴比正交扰动多消除 donor transport。</dd>
<dt>⑤ Downstream propagation</dt><dd>对每个下游 layer ℓ，用 discovery clean states 冻结 answer count step <code>b<sub>ℓ</sub><sup>A</sup></code>，并把双侧 injection 差分投影到它上面：<span class="formula-line">a<sub>ℓ</sub>=&lt;[h<sub>ℓ</sub>(+β)−h<sub>ℓ</sub>(−β)]/(2β),b<sub>ℓ</sub><sup>A</sup>&gt;/||b<sub>ℓ</sub><sup>A</sup>||²</span>再减去 matched orthogonal propagation；L35 specificity={rw['write_propagation']['final_residual_specificity_mean']:.4f}，Holm p={fmt_p(rw['write_propagation']['final_residual_specificity_holm_p'])}。</dd>
</dl></div>
<p><strong>读取机制。</strong>在 H16/H19 上固定 receiver/donor 的 attention routing <code>α</code> 与 value content <code>V</code>，构造 RR、RD、DR、DD 四个 pre-O states；Shapley identity 将完整 donor movement 分为 routing 与 value 两部分。两者都显著：routing family p={fmt_p(rw['read_mode']['routing_family_p'])}，value family p={fmt_p(rw['read_mode']['value_family_p'])}。因此这里不是“只靠 QK 指向哪里”或“只靠 V 中已有内容”，而是路由与被路由内容共同决定写入。</p>
<p><strong>上游接入。</strong>fresh-seed 串联实验先 patch early broad bank，再在 L28 恢复/阻断该自然 channel：early source gain={up['early_effect']['mean']:.4f} [{up['early_effect']['ci_low']:.4f}, {up['early_effect']['ci_high']:.4f}]，L28 mediation={up['mediation']['mean']:.4f} [{up['mediation']['ci_low']:.4f}, {up['mediation']['ci_high']:.4f}]，联合 IUT p={fmt_p(up['intersection_union_p'])}。这验证的是 early retrieval effect 会经过 L28 writer，而不是两个独立显著模块的并列。</p>
<div class="conclusion"><strong>Qwen 机制结论</strong>prompt running state 经 early broad bank 到达 L28；H16/H19 用 α 与 V 共同读取，再由 W<sub>O</sub> 写入新的 answer-residual direction，并传播到最终输出。</div>
</article>

<article class="positive-mechanism-model gemma">
<h3>7.3 Gemma：K2 bank 写入 L37 distributed residual</h3>
<p class="result-sentence">冻结的 L29H4/L35H2 source bank 把 count information 写入 L37 的分布式 residual state；精确 residual block 与 count-axis block 都能特异削弱 donor transport，而 L41 terminal state 会采用这部分写入。四个注册 endpoints 全部通过，global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}。</p>
{table(["证据", "具体验证什么", "effect [95% CI]", "p"], g_rows, classes="paper-table compact-result-table")}
<div class="test-card"><h4>Gemma 的 source write 与 residual propagation 如何验证</h4><dl>
<dt>① True pre-O source write</dt><dd>在 answer query 只把 receiver 的 <code>z<sub>29,4</sub></code> 与 <code>z<sub>35,2</sub></code> 替换为 donor 值，其他 heads 与 residual 保持 receiver。两个 head outputs 必须分别经过自身 <code>W<sub>O</sub></code>；source effect 是该 patch 的 normalized donor transport <code>T<sub>source</sub></code>。</dd>
<dt>② 定义 L37 mediator</dt><dd>在 discovery seeds 上计算同一 source patch 引起的 <span class="formula-line">δ<sub>37</sub>=h<sub>37</sub><sup>source-patch</sup>−h<sub>37</sub><sup>clean receiver</sup></span>并冻结 L37、该 induced change 的构造、count axis 与三个 layer-matched K2 controls；confirmation 不重新挑 layer。</dd>
<dt>③ Exact residual mediation</dt><dd>source patch 后在 L37 删除完整 <code>δ<sub>37</sub></code>；matched condition 删除等范数、与 <code>δ<sub>37</sub></code> 正交的 residual direction。表中正向 effect 为 <span class="formula-line">M<sub>exact</sub>=T<sub>orth-exact</sub>−T<sub>exact-block</sub></span>它检验 source effect 是否真的通过 source patch 在 L37 造成的那段 residual change。</dd>
<dt>④ Count-axis mediation</dt><dd>只删除 <code>δ<sub>37</sub></code> 在冻结 L37 count direction <code>b<sub>37</sub></code> 上的投影，并与等范数正交删除比较：<span class="formula-line">M<sub>count</sub>=T<sub>orth-count</sub>−T<sub>count-block</sub></span>正值说明 exact mediator 中至少一部分是 count-aligned，而不只是任意 source-specific residual disturbance。</dd>
<dt>⑤ Terminal adoption</dt><dd>追踪 source patch 在 L41 引起的 state change <code>Δh<sub>41</sub></code>，投影到 discovery 时冻结的 terminal count step：<span class="formula-line">A<sub>41</sub>=&lt;Δh<sub>41</sub>,b<sub>41</sub>&gt;/||b<sub>41</sub>||²/(D−R)</span>正值表示 L41 state 采用 donor count 的比例。</dd>
</dl></div>
<p><strong>Gemma window 带来的机制差异。</strong>L29H4 与 L35H2 位于周期性 full-attention layers，因而 source write 时可从完整 causal prefix 汇集远端 needles；它们之间及之后的 sliding-attention layers 只能直接访问有限窗口。我们的结果不要求每个 sliding layer 重新检索所有 needles：full-attention K2 先把全局 count information 写进 answer-query residual，随后 residual connection 与局部 block transformation 把这一 state 送到 L37，再传播到 L41。exact block、count-axis block 与 terminal adoption 分别验证“有特定 induced residual”“其中含 count component”“该 component 到达 terminal state”三步。</p>
<p>表中的四个 candidate-core effects 均显著；完整判定还要求每个 endpoint 同时优于三个冻结 layer-matched K2 controls。family p 取候选效应与 candidate-minus-control specificity 中较弱者，global IUT p 再取四个 family p 的最大值；因此 global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])} 表示四步 conjunction 全部通过，而不是从多个 endpoint 中挑最小 p。</p>
<div class="conclusion"><strong>Gemma 机制结论</strong>L29H4/L35H2 在 full-attention answer queries 聚合全 prompt，并经各自 W<sub>O</sub> 写回；写入形成 L37 的分布式、部分 count-aligned residual mediator。后续 sliding layers 不必重新访问远端 needles，而是让该 residual state 继续传播到 L41 answer computation。</div>
</article>
</section>
"""


def build_synthesis_clear(ov: dict[str, Any], gemma_residual: dict[str, Any]) -> str:
    return f"""
<section id="synthesis">
<h2>8 · 最终机制对照</h2>
{table(
    ["阶段", "Qwen3-8B", "Gemma4-E4B"],
    [
        ["Prompt state", "needle-end running counter", "needle-end running counter"],
        ["Retrieval", "L23H28/L23H29/L26H20/L27H18 early broad bank", "L29H4/L35H2 K2 broad bank"],
        ["Read / write", "L28 H16/H19 mixed α/V read + localized W<sub>O</sub> write", "K2 source-bank output writes L37 distributed residual"],
        ["Propagation", "L29–L35 answer-count axes", "L37 → L41 residual path"],
        ["Answer", "Total query residual drives N", "Total query residual drives N"],
        ["Causal conjunction", f"natural OV IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}", f"residual path IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}"],
    ],
    classes="paper-table compact-result-table",
)}
<p class="paper-wording"><strong>论文式表述。</strong>Both non-thinking models formed an ordered prompt-side running state, aggregated this state through distributed attention-head banks, and transformed it into an executable answer-query count representation. In Qwen3-8B, a localized L28 H16/H19 OV channel performed a mixed routing/value read and wrote the count signal into a downstream answer-residual direction. In Gemma4-E4B, a frozen L29H4/L35H2 bank causally wrote a count-aligned distributed residual state at L37 that propagated to the terminal answer computation.</p>
<div class="conclusion"><strong>最终结论</strong>两模型的共同算法是“累计状态 → 分布式读取 → 坐标变换/写入 → answer readout”；差异主要在 state 被因果定位的空间粒度，而不是是否存在 prompt running counter。</div>
</section>
"""


def build_limits_clear(
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    rows = [
        ["Macro representation / patching", "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json", causal_v2["schema_version"]],
        ["Frozen top-k extrapolation", "reports/v4_non-thinking_causal/v4_4_causal_v2/seed_extrapolation_summary.json", f"audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS"],
        ["Qwen natural OV", "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json", ov["schema_version"]],
        ["Qwen α/V read-write", "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json", read_write["schema_version"]],
        ["Qwen fresh serial path", "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json", upstream["schema_version"]],
        ["Gemma K2 residual path", "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k2/realistic_niah_v4_4_4_residual_analysis.json", gemma_residual["schema_version"]],
        ["Fresh correct-only answer routes", "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json", correct_state["schema_version"]],
    ]
    return f"""
<section id="limits">
<h2>9 · 复现信息</h2>
<p>报告中的显著性均来自保存的 seed-level 聚合与审计文件；HTML 不复制 raw hidden states、full value tensors 或 raw attention rows。原始捕获继续保留在 FileStream。</p>
{details_table("Source ledger", ["component", "relative path", "schema/audit"], rows, opened=True)}
<p class="provenance-note">Interface restructuring followed the imported design-taste-frontend workflow (MIT; upstream source commit 9bad53f2426e310c33ef5bacf9f845855197be6a). Scientific definitions and numerical results remain sourced from the V4.4 analysis files above.</p>
<div class="conclusion"><strong>复现结论</strong>所有进入正文的因果结果都对应 audit PASS 的机器可读 analysis；图形只负责展示 effect 与结构，不替代统计文件。</div>
</section>
"""


def validate_inputs(repo_root: Path) -> dict[str, Path]:
    paths = {
        "base": repo_root
        / "reports/v4_non-thinking_causal/v4_4_3/realistic_niah_v4_4_mechanism_report.html",
        "causal_v2": repo_root
        / "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json",
        "seed_confirmation": repo_root
        / "reports/v4_non-thinking_causal/v4_4_causal_v2/seed_extrapolation_summary.json",
        "exact_reanalysis": repo_root
        / "reports/v4_non-thinking_causal/v4_4_causal_v2/exact_sign_flip_reanalysis.json",
        "cue": repo_root
        / "reports/v4_non-thinking_causal/v4_4_2/realistic_niah_v4_4_2_mode_geometry_attention_report.html",
        "ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json",
        "read_write": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
        "relay": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_relay_analysis.json",
        "upstream": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
        "gemma_l37_ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/natural_ov/realistic_niah_v4_4_4_analysis.json",
        "gemma_l29_ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l29h4/realistic_niah_v4_4_4_analysis.json",
        "gemma_l35_ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l35h2/realistic_niah_v4_4_4_analysis.json",
        "gemma_l29_read_write": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l29h4/realistic_niah_v4_4_4_read_write_analysis.json",
        "gemma_l35_read_write": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l35h2/realistic_niah_v4_4_4_read_write_analysis.json",
        "gemma_cross_layer": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/cross_layer/realistic_niah_v4_4_4_cross_layer_analysis.json",
        "gemma_residual_k2": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k2/realistic_niah_v4_4_4_residual_analysis.json",
        "gemma_residual_k6": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k6/realistic_niah_v4_4_4_residual_analysis.json",
        "correct_state": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json",
        "correct_state_geometry": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/geometry_summary.csv",
    }
    required = {
        "base",
        "causal_v2",
        "seed_confirmation",
        "exact_reanalysis",
        "cue",
        "ov",
        "read_write",
        "relay",
        "upstream",
        "gemma_l37_ov",
        "correct_state",
        "correct_state_geometry",
    }
    missing = [
        str(paths[name]) for name in sorted(required) if not paths[name].is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing integrated-report inputs: {missing}")
    return paths


def replace_section(document: str, section_id: str, replacement: str) -> str:
    pattern = re.compile(rf'<section id="{re.escape(section_id)}">.*?</section>', re.S)
    updated, count = pattern.subn(replacement.strip(), document, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one section #{section_id}; replaced {count}")
    return updated


def build_report(repo_root: Path, output: Path) -> None:
    paths = validate_inputs(repo_root)
    base = paths["base"].read_text(encoding="utf-8")
    answer_data = extract_embedded_json(base, "ANSWER_DATA")
    causal_v2 = read_json(paths["causal_v2"])
    seed_confirmation = read_json(paths["seed_confirmation"])
    exact_reanalysis = read_json(paths["exact_reanalysis"])
    cue_doc = paths["cue"].read_text(encoding="utf-8")
    ov = read_json(paths["ov"])
    read_write = read_json(paths["read_write"])
    relay = read_json(paths["relay"])
    upstream = read_json(paths["upstream"])
    gemma_l37_ov = read_json(paths["gemma_l37_ov"])
    gemma_singles = {
        name: read_json(paths[path_key])
        for name, path_key in (
            ("l29h4", "gemma_l29_ov"),
            ("l35h2", "gemma_l35_ov"),
        )
        if paths[path_key].is_file()
    }
    gemma_read_writes = {
        name: read_json(paths[path_key])
        for name, path_key in (
            ("l29h4", "gemma_l29_read_write"),
            ("l35h2", "gemma_l35_read_write"),
        )
        if paths[path_key].is_file()
    }
    gemma_cross_layer = (
        read_json(paths["gemma_cross_layer"])
        if paths["gemma_cross_layer"].is_file()
        else None
    )
    gemma_residuals = {
        name: read_json(paths[path_key])
        for name, path_key in (
            ("k2", "gemma_residual_k2"),
            ("k6", "gemma_residual_k6"),
        )
        if paths[path_key].is_file()
    }
    correct_state = read_json(paths["correct_state"])
    correct_state_geometry = read_csv_rows(paths["correct_state_geometry"])
    gemma_story = resolve_gemma_story(
        l37=gemma_l37_ov,
        singles=gemma_singles,
        read_writes=gemma_read_writes,
        cross_layer=gemma_cross_layer,
        residuals=gemma_residuals,
    )

    if int(exact_reanalysis["method"]["assignments_enumerated"]) != 2**20:
        raise RuntimeError(
            "Correct-only exact reanalysis did not enumerate all 2^20 assignments"
        )
    exact_rows = {
        (str(row["model"]), str(row["top_k"])): row
        for row in exact_reanalysis["results"]
    }
    for model, model_rows in seed_confirmation["models"].items():
        for k_text, metrics in model_rows.items():
            audit_row = exact_rows[(str(model), str(k_text))]
            # The original summary used deterministic Monte Carlo for n=20 but
            # retained an ``exact`` field name.  The separately audited full
            # 2^20 enumeration is authoritative and is injected into the
            # in-memory report payload without mutating the archived source.
            metrics["clean_correct_to_wrong"]["two_sided_exact_seed_sign_flip_p"] = (
                float(audit_row["clean_correct_failure"]["exact_p"])
            )
            metrics["clean_correct_to_wrong"]["holm_p_across_four_frozen_sets"] = float(
                audit_row["clean_correct_failure"]["holm_p"]
            )
            metrics["absolute_error"]["exact_p"] = float(
                audit_row["absolute_error"]["exact_p"]
            )
            metrics["absolute_error"]["holm_p_across_four_frozen_sets"] = float(
                audit_row["absolute_error"]["holm_p"]
            )
    confirmation_order = [
        ("Qwen3-8B", "2"),
        ("Qwen3-8B", "4"),
        ("Gemma4-E4B", "1"),
        ("Gemma4-E4B", "2"),
    ]
    absolute_error_holm = holm_adjusted_pvalues(
        [
            float(
                seed_confirmation["models"][model][k_text]["absolute_error"]["exact_p"]
            )
            for model, k_text in confirmation_order
        ]
    )
    for (model, k_text), adjusted_p in zip(
        confirmation_order, absolute_error_holm, strict=True
    ):
        if not math.isclose(
            adjusted_p,
            float(exact_rows[(model, k_text)]["absolute_error"]["holm_p"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"Absolute-error Holm-p mismatch for {model} K={k_text}")

    if not ov["audit"]["all_checks_pass"]:
        raise RuntimeError("Natural-OV audit did not pass")
    if not read_write["audit"]["all_checks_pass"]:
        raise RuntimeError("Read/write audit did not pass")
    if not relay["audit"]["all_checks_pass"]:
        raise RuntimeError("Relay audit did not pass")
    if not upstream["audit"]["all_checks_pass"]:
        raise RuntimeError("Upstream-confirmation audit did not pass")
    gemma_audits: list[tuple[str, dict[str, Any]]] = [
        ("L37 natural-OV", gemma_l37_ov),
        *((f"{name} natural-OV", doc) for name, doc in gemma_singles.items()),
        *((f"{name} read/write", doc) for name, doc in gemma_read_writes.items()),
    ]
    if gemma_cross_layer is not None:
        gemma_audits.append(("cross-layer", gemma_cross_layer))
    gemma_audits.extend(
        (f"{name} residual", document) for name, document in gemma_residuals.items()
    )
    for label, document in gemma_audits:
        if not document.get("audit", {}).get("all_checks_pass", False):
            raise RuntimeError(f"Gemma {label} audit did not pass")
    if not correct_state.get("audits", {}).get("all_checks_pass", False):
        raise RuntimeError("Correct-only state-route audit did not pass")

    base = re.sub(
        r"<title>.*?</title>",
        "<title>Realistic NIAH V4.4 · non-thinking integrated mechanism</title>",
        base,
        count=1,
    )
    if "</style>" not in base:
        raise RuntimeError("Base report has no style terminator")
    base = base.replace("</style>", EXTRA_CSS + "\n</style>", 1)
    nav = """<nav><a href="#mechanism-overview">Main figure</a><a href="#scope">结论</a><a href="#methods">设定/定义</a><a href="#prompt">Prompt geometry</a><a href="#cue-robustness">Cue robustness</a><a href="#answer">Answer geometry</a><a href="#attention">Attention</a><a href="#causal">Causal design</a><a href="#natural-ov">Natural OV</a><a href="#read-write">Read/write</a><a href="#upstream">Serial path</a><a href="#synthesis">Synthesis</a><a href="#limits">边界</a></nav>"""
    base, nav_count = re.subn(r"<nav>.*?</nav>", nav, base, count=1, flags=re.S)
    if nav_count != 1:
        raise RuntimeError("Could not replace report navigation")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"""<header>
<div class="eyebrow">Realistic NIAH · V4.4 · non-thinking · integrated evidence</div>
<h1>从 running-index representation 到自然 read–write causal circuit</h1>
<p class="lead">一份统一的 representation → retrieval → state transport → causal write/relay 报告。Qwen3-8B 与 Gemma4-E4B 使用相同 estimands、各自冻结的候选与独立 seed 外推；Gemma 的定位粒度由顺序证据阶梯中实际通过的最强 conjunction 决定，而不是强迫复制 Qwen 的 layer/head identity。</p>
<p class="meta">generated {generated} · source campaigns V4.4 / V4.4.2 / causal-v2 / V4.4.4 · self-contained HTML</p>
</header>"""
    base, header_count = re.subn(
        r"<header>.*?</header>", header, base, count=1, flags=re.S
    )
    if header_count != 1:
        raise RuntimeError("Could not replace report header")

    base = replace_section(
        base,
        "scope",
        build_scope(
            causal_v2,
            ov,
            read_write,
            relay,
            upstream,
            gemma_l37_ov,
            gemma_story,
        ),
    )
    overview = build_mechanism_overview(ov, read_write, upstream, gemma_story)
    base = base.replace(
        '<section id="scope">', overview + '\n\n<section id="scope">', 1
    )
    methods = build_methods(
        ov,
        upstream,
        gemma_l37_ov,
        gemma_singles,
        gemma_read_writes,
        gemma_cross_layer,
        gemma_residuals,
    )
    base = base.replace(
        '<section id="prompt">', methods + '\n\n<section id="prompt">', 1
    )
    base = base.replace(
        "<h2>1 · Prompt-reading counter representation</h2>",
        "<h2>3 · Prompt-reading counter representation</h2>",
        1,
    )
    running_block = build_running_index_block()
    prompt_marker = (
        '<div class="figure-block"><h3>1.1 Interactive V4.4 prompt counter</h3>'
    )
    if prompt_marker not in base:
        raise RuntimeError("Could not locate prompt 3D figure block")
    base = base.replace(
        prompt_marker,
        running_block
        + '\n\n<div class="figure-block"><h3>3.2 Seed-level prompt counter · 完整交互</h3>',
        1,
    )
    cue_section = build_cue_section(cue_doc)
    base = base.replace(
        '<section id="answer">', cue_section + '\n\n<section id="answer">', 1
    )
    base = base.replace(
        "<h2>2 · Answer-query counter representation</h2>",
        "<h2>5 · Answer-query counter representation</h2>",
        1,
    )
    base = base.replace(
        "<h3>2.1 Interactive V4.4 answer-query counter</h3>",
        "<h3>5.2 Interactive V4.4 answer-query counter</h3>",
        1,
    )
    base = base.replace(
        "<h3>2.2 Prompt 与 answer counter 的共同坐标</h3>",
        "<h3>5.2 Prompt 与 answer counter 的共同坐标</h3>",
        1,
    )
    base = base.replace("<h3>5.2 Prompt ", "<h3>5.3 Prompt ", 1)
    answer_marker = (
        '<div class="figure-block"><h3>5.2 Interactive V4.4 answer-query counter</h3>'
    )
    if answer_marker not in base:
        raise RuntimeError("Could not locate answer-query 3D figure block")
    base = base.replace(
        answer_marker,
        build_answer_fit_sensitivity(answer_data) + "\n\n" + answer_marker,
        1,
    )
    base = base.replace(
        "<h2>3 · V4.4 attention-head representation</h2>",
        "<h2>6 · V4.4 attention-head representation</h2>",
        1,
    )
    base = base.replace(
        "<h3>3.1 All-head V4.4 atlas</h3>", "<h3>6.1 All-head V4.4 atlas</h3>", 1
    )
    causal_header = (
        '<section id="causal">\n<h2>7 · 因果证据：设计、聚合与宏观定位</h2>\n'
        + build_causal_design(
            ov,
            upstream,
            seed_confirmation,
            gemma_l37_ov,
            gemma_singles,
            gemma_read_writes,
            gemma_cross_layer,
            gemma_residuals,
        )
        + build_causal_v2_intro(causal_v2, seed_confirmation)
    )
    base, causal_count = re.subn(
        r'<section id="causal">\s*<h2>.*?</h2>',
        causal_header,
        base,
        count=1,
        flags=re.S,
    )
    if causal_count != 1:
        raise RuntimeError("Could not update causal section")
    # Renumber only the legacy causal subsections.  The cue-robustness section
    # that is inserted above also has 4.1/4.2 headings; a document-wide replace
    # would rename those first and then let the ablation regex backtrack across
    # the answer-query figure, deleting its canvas.
    causal_section_pattern = re.compile(
        r'(<section id="causal">)(.*?)(</section>)', re.S
    )

    def renumber_causal_subsections(match: re.Match[str]) -> str:
        body = match.group(2)
        for old, new in (
            ("4.1", "7.4"),
            ("4.2", "7.5"),
            ("4.3", "7.6"),
            ("4.4", "7.7"),
            ("4.5", "7.8"),
        ):
            body = body.replace(f"<h3>{old} ", f"<h3>{new} ", 1)
        return match.group(1) + body + match.group(3)

    base, causal_renumber_count = causal_section_pattern.subn(
        renumber_causal_subsections, base, count=1
    )
    if causal_renumber_count != 1:
        raise RuntimeError("Could not isolate causal section for subsection renumbering")
    ablation_pattern = re.compile(
        r'(<h3>7\.4 [^<]*</h3>)\s*<p class="figure-intro">.*?</p>\s*<figure>.*?</figure>',
        re.S,
    )
    ablation_replacement = r"""\1
<div class="callout warning"><strong>旧高-count screen 的定位。</strong>下表保留原 V4.4 count 7–10、K=4/8 screen 作为历史敏感性分析；它没有 correct-only eligibility，也没有本轮冻结 K 的独立 seed 外推，因此不再承担主 head-bank necessity 结论。主图与主统计见上方 7.3。</div>"""
    base, ablation_count = ablation_pattern.subn(ablation_replacement, base, count=1)
    if ablation_count != 1:
        raise RuntimeError("Could not replace top-k ablation figure")

    natural_appendices = [
        build_gemma_evidence_ladder(
            l37=gemma_l37_ov,
            singles=gemma_singles,
            cross_layer=gemma_cross_layer,
            residuals=gemma_residuals,
            story=gemma_story,
        ),
        build_gemma_natural_ov_appendix(
            gemma_l37_ov,
            heading="8.5",
            context_label="最初冻结且完整保留的负结果",
        ),
    ]
    for index, (name, document) in enumerate(gemma_singles.items(), start=6):
        natural_appendices.append(
            build_gemma_natural_ov_appendix(
                document,
                heading=f"8.{index}",
                context_label=f"independent-ablation candidate {name}",
            )
        )
    if gemma_cross_layer is not None:
        natural_appendices.append(
            build_gemma_natural_ov_appendix(
                gemma_cross_layer,
                heading=f"8.{6 + len(gemma_singles)}",
                context_label="条件式跨层 K2 fallback",
            )
        )
    natural_section = append_to_section(
        build_natural_ov_section(ov), "\n".join(natural_appendices)
    )

    read_write_appendices: list[str] = []
    for index, (name, document) in enumerate(gemma_read_writes.items(), start=3):
        parent = gemma_singles.get(name)
        if parent is None:
            raise RuntimeError(
                f"Gemma read/write has no parent natural-OV result: {name}"
            )
        read_write_appendices.append(
            build_gemma_read_write_appendix(
                document,
                parent,
                heading=f"9.{index}",
                natural_heading=(f"8.{6 + list(gemma_singles).index(name)}"),
            )
        )
    read_write_section = build_read_write_section(read_write)
    if read_write_appendices:
        read_write_section = append_to_section(
            read_write_section, "\n".join(read_write_appendices)
        )

    upstream_appendices: list[str] = []
    if gemma_cross_layer is not None:
        upstream_appendices.append(build_gemma_cross_layer_appendix(gemma_cross_layer))
    upstream_appendices.extend(
        build_gemma_residual_appendix(document) for document in gemma_residuals.values()
    )
    upstream_section = build_upstream_section(relay, upstream)
    if upstream_appendices:
        upstream_section = append_to_section(
            upstream_section, "\n".join(upstream_appendices)
        )
    synthesis_section = append_to_section(
        build_synthesis_section(),
        "\n".join(
            [
                build_gemma_synthesis_ladder(
                    l37=gemma_l37_ov,
                    singles=gemma_singles,
                    read_writes=gemma_read_writes,
                    cross_layer=gemma_cross_layer,
                    residuals=gemma_residuals,
                    story=gemma_story,
                ),
                build_correct_state_boundary(correct_state, correct_state_geometry),
            ]
        ),
    )
    additions = "\n\n".join(
        [natural_section, read_write_section, upstream_section, synthesis_section]
    )
    base = base.replace(
        '<section id="limits">', additions + '\n\n<section id="limits">', 1
    )
    base = replace_section(
        base,
        "limits",
        build_limits_dynamic(
            causal_v2=causal_v2,
            seed_confirmation=seed_confirmation,
            ov=ov,
            read_write=read_write,
            relay=relay,
            upstream=upstream,
            gemma_l37=gemma_l37_ov,
            gemma_singles=gemma_singles,
            gemma_read_writes=gemma_read_writes,
            gemma_cross_layer=gemma_cross_layer,
            gemma_residuals=gemma_residuals,
            gemma_story=gemma_story,
            correct_state=correct_state,
        ),
    )

    if "function makeProjector" not in base:
        raise RuntimeError("Could not locate embedded visualization script")
    base = base.replace(
        "function makeProjector", EXTRA_JS + "\nfunction makeProjector", 1
    )
    old_boot = "makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    new_boot = "makeMechanismWalkthrough();makeRunningIndex();makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    if old_boot not in base:
        raise RuntimeError("Could not locate visualization bootstrap")
    base = base.replace(old_boot, new_boot, 1)

    required_sections = [
        "mechanism-overview",
        "scope",
        "methods",
        "prompt",
        "cue-robustness",
        "answer",
        "attention",
        "causal",
        "natural-ov",
        "read-write",
        "upstream",
        "synthesis",
        "limits",
    ]
    for section_id in required_sections:
        if base.count(f'id="{section_id}"') != 1:
            raise RuntimeError(f"Section id count is not one: {section_id}")
    for canvas_id in (
        "running-index-canvas",
        "prompt-canvas",
        "answer-canvas",
        "joint-canvas",
    ):
        if base.count(f'id="{canvas_id}"') != 1:
            raise RuntimeError(f"Interactive canvas id count is not one: {canvas_id}")
    for heading in (
        "<h3>4.1 Hidden-state geometry</h3>",
        "<h3>4.2 Attention map：同一 broad-retrieval score 的左右对照</h3>",
        "<h3>7.4 Head ablation · mixed ranked bank 是否比 layer-matched random 更重要？</h3>",
        "<h3>7.5 Needle-end patching · 单个 toggled endpoint state 是否足以运输 count increment？</h3>",
    ):
        if base.count(heading) != 1:
            raise RuntimeError(f"Expected one renumbered report heading: {heading}")
    if len(re.findall(r"<figcaption\b", base)) != base.count("</figcaption>"):
        raise RuntimeError("Unbalanced figure captions")
    if base.count("<section") != base.count("</section>"):
        raise RuntimeError("Unbalanced sections")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(base, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sections": required_sections,
                "figures": len(re.findall(r"<figure\b", base)),
                "figcaptions": len(re.findall(r"<figcaption\b", base)),
                "conclusion_boxes": base.count('<div class="conclusion">'),
                "natural_ov_global_iut_p": ov["primary_decision"][
                    "global_intersection_union_p"
                ],
                "upstream_global_iut_p": upstream["primary_decision"][
                    "intersection_union_p"
                ],
                "gemma_strongest_kind": gemma_story["kind"],
                "gemma_strongest_global_iut_p": gemma_story.get("global_p"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


MECHANISM_DETAILED_JS = r"""
function makeMechanismWalkthrough(){
 const root=document.getElementById('mechanism-paper-figure');if(!root)return;
 const nodes=[...root.querySelectorAll('[data-mechanism-stage]')];
 const edges=[...root.querySelectorAll('[data-mechanism-edge]')];
 const dots=[...root.querySelectorAll('[data-mechanism-step]')];
 const prev=document.getElementById('mechanism-prev');
 const next=document.getElementById('mechanism-next');
 const play=document.getElementById('mechanism-play');
 const qTitle=document.getElementById('mechanism-live-q-title');
 const qBody=document.getElementById('mechanism-live-q-body');
 const gTitle=document.getElementById('mechanism-live-g-title');
 const gBody=document.getElementById('mechanism-live-g-body');
 if(!prev||!next||!play||!qTitle||!qBody||!gTitle||!gBody)return;
 const stages=[
  {qTitle:'1/5 · Qwen 提取 prompt running state',qBody:'在同一 N=10 prompt 内，于第 n 个 active needle 的最后 token 保存 post-block residual；n=1…10 是读取进度。',gTitle:'1/5 · Gemma 提取 prompt running state',gBody:'使用相同 endpoint 定义；sliding layers 的 state 可以同时包含局部更新、前序 full-layer 刷新与 residual/MLP 变换。'},
  {qTitle:'2/5 · Qwen early slot-state bank',qBody:'冻结 early top-4，并只在注册 slot-query positions 替换 donor pre-O z；后续 L28 exact restoration 检验串联中介。',gTitle:'2/5 · Gemma full-attention K2 bank',gBody:'L29H4 与 L35H2 位于 full-attention layers，可在 answer query 汇集整个 causal prefix；三个 layer-matched K2 sets 作为对照。'},
  {qTitle:'3/5 · Qwen α/V mixed read',qBody:'在 L28 H16/H19 构造 RR、RD、DR、DD 四个 pre-O endpoints，用 Shapley identity 把 donor-z movement 分成 routing 与 value 两部分。',gTitle:'3/5 · Gemma true pre-O source patch',gBody:'只替换 answer-query 的 z(29,4) 与 z(35,2)；写入必须经过这两个 heads 自己的 W_O，receiver 其他状态保持不变。'},
  {qTitle:'4/5 · Qwen natural OV write',qBody:'在 W_O 前注入或删除 natural V-path count step，并与同 W_O span、等 post-O 范数的正交方向比较；影响沿 L29–L35 frozen count axes 追踪。',gTitle:'4/5 · Gemma L37 residual mediator',gBody:'测量 K2 patch 在 L37 诱发的 δ；exact block 删除整个 δ，count-axis block 只删除其 count-aligned component，再追踪至 L41。'},
  {qTitle:'5/5 · Qwen answer state 执行输出',qBody:'把 donor 的 Total: query full residual 单层替换给 receiver，再从 receiver context 完整 greedy 生成；输出显著向 donor count 移动。',gTitle:'5/5 · Gemma terminal state 执行输出',gBody:'L41 frozen count-axis adoption 与独立 full-state answer patch 共同显示：窗口化传播后的 terminal query state 可以改变最终数字。'}
 ];
 let step=0,timer=null;
 function stop(){if(timer){clearInterval(timer);timer=null}play.textContent='▶ 播放一次';play.setAttribute('aria-pressed','false')}
 function render(){
  nodes.forEach(node=>{const i=Number(node.dataset.mechanismStage);node.classList.toggle('is-active',i===step);node.classList.toggle('is-complete',i<step)});
  edges.forEach(edge=>{const i=Number(edge.dataset.mechanismEdge);edge.classList.toggle('is-active',i===step);edge.classList.toggle('is-complete',i<step)});
  dots.forEach(dot=>dot.setAttribute('aria-current',Number(dot.dataset.mechanismStep)===step?'step':'false'));
  prev.disabled=step===0;next.disabled=step===stages.length-1;
  qTitle.textContent=stages[step].qTitle;qBody.textContent=stages[step].qBody;
  gTitle.textContent=stages[step].gTitle;gBody.textContent=stages[step].gBody;
 }
 prev.addEventListener('click',()=>{stop();step=Math.max(0,step-1);render()});
 next.addEventListener('click',()=>{stop();step=Math.min(stages.length-1,step+1);render()});
 dots.forEach(dot=>dot.addEventListener('click',()=>{stop();step=Number(dot.dataset.mechanismStep);render()}));
 play.addEventListener('click',()=>{if(timer){stop();return}if(step===stages.length-1)step=0;render();play.textContent='Ⅱ 暂停';play.setAttribute('aria-pressed','true');timer=setInterval(()=>{if(step>=stages.length-1){stop();return}step+=1;render()},1600)});
 render();
}
"""


def build_report_clear(repo_root: Path, output: Path) -> None:
    paths = validate_inputs(repo_root)
    base = paths["base"].read_text(encoding="utf-8")
    answer_data = extract_embedded_json(base, "ANSWER_DATA")
    causal_v2 = read_json(paths["causal_v2"])
    seed_confirmation = read_json(paths["seed_confirmation"])
    exact_reanalysis = read_json(paths["exact_reanalysis"])
    ov = read_json(paths["ov"])
    read_write = read_json(paths["read_write"])
    upstream = read_json(paths["upstream"])
    gemma_residual = read_json(paths["gemma_residual_k2"])
    correct_state = read_json(paths["correct_state"])

    exact_rows = {
        (str(row["model"]), str(row["top_k"])): row
        for row in exact_reanalysis["results"]
    }
    for model, model_rows in seed_confirmation["models"].items():
        for k_text, metrics in model_rows.items():
            audited = exact_rows[(str(model), str(k_text))]
            metrics["clean_correct_to_wrong"][
                "two_sided_exact_seed_sign_flip_p"
            ] = float(audited["clean_correct_failure"]["exact_p"])
            metrics["clean_correct_to_wrong"][
                "holm_p_across_four_frozen_sets"
            ] = float(audited["clean_correct_failure"]["holm_p"])

    if any(
        causal_v2["audits"][model]["status"] != "PASS"
        for model in ("Qwen3-8B", "Gemma4-E4B")
    ):
        raise RuntimeError("causal-v2 audit did not pass")
    if seed_confirmation["audit"]["status"] != "PASS":
        raise RuntimeError("Frozen top-k audit did not pass")
    for label, document in (
        ("Qwen natural OV", ov),
        ("Qwen read/write", read_write),
        ("Qwen upstream", upstream),
        ("Gemma residual", gemma_residual),
    ):
        if not document.get("audit", {}).get("all_checks_pass", False):
            raise RuntimeError(f"{label} audit did not pass")
    if not correct_state.get("audits", {}).get("all_checks_pass", False):
        raise RuntimeError("Correct-only route audit did not pass")

    base = re.sub(
        r"<title>.*?</title>",
        "<title>Realistic NIAH V4.4 · non-thinking mechanism</title>",
        base,
        count=1,
    )
    if "</style>" not in base:
        raise RuntimeError("Base report has no style terminator")
    base = base.replace("</style>", EXTRA_CSS + CLEAR_CSS + "\n</style>", 1)
    nav = """<nav><a href="#mechanism-overview">Mechanism</a><a href="#scope">结论</a><a href="#methods">设定</a><a href="#prompt">Prompt geometry</a><a href="#answer">Answer geometry</a><a href="#attention">Attention</a><a href="#causal">Ablation / patching</a><a href="#natural-ov">Write / propagation</a><a href="#synthesis">对照</a><a href="#limits">复现</a></nav>"""
    base, nav_count = re.subn(r"<nav>.*?</nav>", nav, base, count=1, flags=re.S)
    if nav_count != 1:
        raise RuntimeError("Could not replace report navigation")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"""<header>
<div class="eyebrow">Realistic NIAH · V4.4 · non-thinking</div>
<h1>从 prompt running counter 到 answer count：representation 与 causal mechanism</h1>
<p class="lead">Qwen3-8B 与 Gemma4-E4B 分开描述：先看各自如何形成和读取 count state，再看 top-k ablation、answer-query patching，以及得到显著支持的写入/传播通路。</p>
<p class="meta">generated {generated} · self-contained HTML · raw tensors remain in FileStream</p>
</header>"""
    base, header_count = re.subn(
        r"<header>.*?</header>", header, base, count=1, flags=re.S
    )
    if header_count != 1:
        raise RuntimeError("Could not replace report header")

    scope = build_scope_clear(causal_v2, ov, upstream, gemma_residual)
    base = replace_section(base, "scope", scope)
    overview = build_mechanism_overview_detailed(
        ov, read_write, upstream, gemma_residual, causal_v2, correct_state
    )
    base = base.replace('<section id="scope">', overview + '\n\n<section id="scope">', 1)
    methods = build_methods_clear(ov, upstream, gemma_residual)
    base = base.replace('<section id="prompt">', methods + '\n\n<section id="prompt">', 1)

    base = base.replace(
        "<h2>1 · Prompt-reading counter representation</h2>",
        "<h2>3 · Prompt running-counter representation</h2>",
        1,
    )
    prompt_marker = '<div class="figure-block"><h3>1.1 Interactive V4.4 prompt counter</h3>'
    if prompt_marker not in base:
        raise RuntimeError("Could not locate prompt counter figure")
    base = base.replace(
        prompt_marker,
        build_running_index_block()
        + '\n\n<div class="figure-block"><h3>3.2 Seed-level prompt counter · 完整交互</h3>',
        1,
    )

    base = base.replace(
        "<h2>2 · Answer-query counter representation</h2>",
        "<h2>4 · Answer-query counter representation</h2>",
        1,
    )
    base = base.replace(
        "<h3>2.1 Interactive V4.4 answer-query counter</h3>",
        "<h3>4.2 Interactive V4.4 answer-query counter</h3>",
        1,
    )
    base = base.replace(
        "<h3>2.2 Prompt 与 answer counter 的共同坐标</h3>",
        "<h3>4.3 Prompt 与 answer counter 的共同坐标</h3>",
        1,
    )
    answer_marker = '<div class="figure-block"><h3>4.2 Interactive V4.4 answer-query counter</h3>'
    if answer_marker not in base:
        raise RuntimeError("Could not locate answer counter figure")
    fit_block = build_answer_fit_sensitivity(answer_data).replace(
        "<h3>5.1 ", "<h3>4.1 ", 1
    )
    base = base.replace(answer_marker, fit_block + "\n\n" + answer_marker, 1)

    base = base.replace(
        "<h2>3 · V4.4 attention-head representation</h2>",
        "<h2>5 · Attention-head retrieval representation</h2>",
        1,
    )
    attention_heading = "<h2>5 · Attention-head retrieval representation</h2>"
    if attention_heading not in base:
        raise RuntimeError("Could not locate attention section heading")
    base = base.replace(
        attention_heading,
        attention_heading + "\n" + build_attention_estimand_note(),
        1,
    )
    base = base.replace(
        "<h3>3.1 All-head V4.4 atlas</h3>",
        "<h3>5.2 All-head V4.4 atlas</h3>",
        1,
    )
    atlas_default_swaps = (
        (
            '<button type="button" data-atlas="span_end" aria-pressed="true">endpoint-key mass</button>',
            '<button type="button" data-atlas="span_end" aria-pressed="false">endpoint-key mass</button>',
        ),
        (
            '<button type="button" data-atlas="span_sum" aria-pressed="false">full-span literal mass</button>',
            '<button type="button" data-atlas="span_sum" aria-pressed="true">full-span literal mass</button>',
        ),
        (
            '<div class="atlas-panel" data-atlas-panel="span_end">',
            '<div class="atlas-panel" data-atlas-panel="span_end" hidden>',
        ),
        (
            '<div class="atlas-panel" data-atlas-panel="span_sum" hidden>',
            '<div class="atlas-panel" data-atlas-panel="span_sum">',
        ),
    )
    for old, new in atlas_default_swaps:
        if old not in base:
            raise RuntimeError(f"Could not set full-span atlas default: {old}")
        base = base.replace(old, new, 1)

    base = replace_section(
        base,
        "causal",
        build_causal_section_clear(causal_v2, seed_confirmation, correct_state),
    )
    positive_section = build_positive_mechanism_section(
        ov, read_write, upstream, gemma_residual
    )
    synthesis_section = build_synthesis_clear(ov, gemma_residual)
    base = base.replace(
        '<section id="limits">',
        positive_section + "\n\n" + synthesis_section + '\n\n<section id="limits">',
        1,
    )
    base = replace_section(
        base,
        "limits",
        build_limits_clear(
            causal_v2,
            seed_confirmation,
            ov,
            read_write,
            upstream,
            gemma_residual,
            correct_state,
        ),
    )

    if "function makeProjector" not in base:
        raise RuntimeError("Could not locate embedded visualization script")
    running_only_js = (
        MECHANISM_DETAILED_JS
        + EXTRA_JS[EXTRA_JS.index("function makeRunningIndex") :]
    )
    base = base.replace(
        "function makeProjector", running_only_js + "\nfunction makeProjector", 1
    )
    old_boot = "makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    new_boot = "makeMechanismWalkthrough();makeRunningIndex();makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    if old_boot not in base:
        raise RuntimeError("Could not locate visualization bootstrap")
    base = base.replace(old_boot, new_boot, 1)

    required_sections = [
        "mechanism-overview",
        "scope",
        "methods",
        "prompt",
        "answer",
        "attention",
        "causal",
        "natural-ov",
        "synthesis",
        "limits",
    ]
    for section_id in required_sections:
        if base.count(f'id="{section_id}"') != 1:
            raise RuntimeError(f"Section id count is not one: {section_id}")
    for removed_section in ("cue-robustness", "read-write", "upstream"):
        if f'id="{removed_section}"' in base:
            raise RuntimeError(f"Removed section unexpectedly present: {removed_section}")
    for canvas_id in (
        "running-index-canvas",
        "prompt-canvas",
        "answer-canvas",
        "joint-canvas",
    ):
        if base.count(f'id="{canvas_id}"') != 1:
            raise RuntimeError(f"Interactive canvas id count is not one: {canvas_id}")
    if len(re.findall(r"<figcaption\b", base)) != base.count("</figcaption>"):
        raise RuntimeError("Unbalanced figure captions")
    if base.count("<section") != base.count("</section>"):
        raise RuntimeError("Unbalanced sections")
    if "cue-present/absent" in base or "Cue robustness" in base:
        raise RuntimeError("Cue-sensitivity content was not fully removed")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(base, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sections": required_sections,
                "figures": len(re.findall(r"<figure\b", base)),
                "figcaptions": len(re.findall(r"<figcaption\b", base)),
                "qwen_ov_global_iut_p": ov["primary_decision"]["global_intersection_union_p"],
                "gemma_residual_global_iut_p": gemma_residual["primary_decision"]["global_intersection_union_p"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the paper-grade integrated V4.4 non-thinking mechanism report"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report_clear(args.repo_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
