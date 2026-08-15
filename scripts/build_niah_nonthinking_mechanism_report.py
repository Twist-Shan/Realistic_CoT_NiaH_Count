#!/usr/bin/env python3
"""Build the concise Non-thinking V4.4 mechanism report.

The rendered report is deliberately organized by the proposed computation:

1. mechanism at a glance, followed by the behavioral target and measurements;
2. distributed prompt-side evidence formation;
3. broad, partial answer-query retrieval;
4. late consolidation plus architecture-specific OV/residual write;
5. evidence synthesis, extension audit, and explicitly prioritized open work.

All displayed numbers are loaded from frozen report artifacts.  The only
embedded observations are four pre-computed, frozen-PCA projections used by
the two interactive 3-D figures.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "v4_non-thinking_causal"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pick(rows: Iterable[dict[str, str]], **criteria: object) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    rendered = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
    raise KeyError(f"No row matched {rendered}")


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


def f(value: object, digits: int = 3) -> str:
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def p_text(value: object) -> str:
    number = float(value)
    return f"{number:.2e}" if number < 0.001 else f"{number:.3f}"


def ci(row: dict[str, str], low: str, high: str, digits: int = 3) -> str:
    return f"{f(row[low], digits)}–{f(row[high], digits)}"


def svg_line_chart(
    title: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str, str]],
    *,
    x_label: str = "Transformer layer ℓ",
    x_value_prefix: str = "L",
    x_ticks: list[float] | None = None,
    y_domain: tuple[float, float] | None = None,
    reference: tuple[float, str] | None = None,
    vertical_references: list[tuple[float, str]] | None = None,
    intervals: dict[str, list[tuple[float, float, float, bool]]] | None = None,
    width: int = 620,
    height: int = 320,
) -> str:
    """Return a compact responsive SVG line chart.

    Each series is (label, points, color, dash).  The data domain is derived
    from the observations unless an interpretable fixed domain is supplied.
    """

    all_points = [point for _, points, _, _ in series for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    if intervals:
        for interval_rows in intervals.values():
            for _, low, high, _ in interval_rows:
                y_values.extend((low, high))
    if reference:
        y_values.append(reference[0])
    x_min, x_max = min(x_values), max(x_values)
    if y_domain is None:
        y_min, y_max = min(y_values), max(y_values)
        pad = max((y_max - y_min) * 0.12, 0.02)
        y_min, y_max = y_min - pad, y_max + pad
    else:
        y_min, y_max = y_domain
    left, right, top, bottom = 68, 24, 34, 58
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h

    tick_values = x_ticks or [x_min + i * (x_max - x_min) / 5 for i in range(6)]
    y_ticks = [y_min + i * (y_max - y_min) / 4 for i in range(5)]
    parts = [
        f'<svg class="line-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}。横轴为{html.escape(x_label)}，纵轴为{html.escape(y_label)}。">',
        f"<title>{html.escape(title)}</title>",
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{tick:.2f}</text>')
    for tick in tick_values:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">{tick:.0f}</text>')
    if reference:
        y = sy(reference[0])
        parts.append(f'<line class="reference" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="reference-label" x="{width-right-4}" y="{y-7:.2f}" text-anchor="end">{html.escape(reference[1])}</text>')
    for value, label in vertical_references or []:
        if value < x_min or value > x_max:
            continue
        x = sx(value)
        anchor = "end" if value > x_min + 0.72 * (x_max - x_min) else "start"
        label_x = x - 5 if anchor == "end" else x + 5
        parts.append(
            f'<line class="reference" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{height-bottom}"/>'
        )
        parts.append(
            f'<text class="reference-label" x="{label_x:.2f}" y="{top+15}" '
            f'text-anchor="{anchor}">{html.escape(label)}</text>'
        )
    for label, points, color, dash in series:
        interval_lookup = {
            x: (low, high, significant)
            for x, low, high, significant in (intervals or {}).get(label, [])
        }
        for x, y in points:
            if x not in interval_lookup:
                continue
            low, high, _ = interval_lookup[x]
            x_pos = sx(x)
            low_pos, high_pos = sy(low), sy(high)
            parts.append(
                f'<line class="ci-whisker" x1="{x_pos:.2f}" y1="{high_pos:.2f}" '
                f'x2="{x_pos:.2f}" y2="{low_pos:.2f}" stroke="{color}"/>'
            )
            parts.append(
                f'<line class="ci-whisker" x1="{x_pos-4:.2f}" y1="{high_pos:.2f}" '
                f'x2="{x_pos+4:.2f}" y2="{high_pos:.2f}" stroke="{color}"/>'
            )
            parts.append(
                f'<line class="ci-whisker" x1="{x_pos-4:.2f}" y1="{low_pos:.2f}" '
                f'x2="{x_pos+4:.2f}" y2="{low_pos:.2f}" stroke="{color}"/>'
            )
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline class="series-line" points="{coords}" stroke="{color}"{dash_attr}/>')
        for x, y in points:
            interval = interval_lookup.get(x)
            if interval is None:
                marker_class = "series-dot"
                marker_fill = color
                marker_stroke = "#fff"
                marker_radius = 2.6
                title_suffix = ""
            else:
                low, high, significant = interval
                marker_class = "ci-dot"
                marker_fill = color if significant else "#fff"
                marker_stroke = "#fff" if significant else color
                marker_radius = 3.2
                title_suffix = (
                    f" · 95% CI [{low:.3f}, {high:.3f}]"
                    f" · exact sign-flip p {'< 0.05' if significant else '≥ 0.05'}"
                )
            parts.append(
                f'<circle class="{marker_class}" cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{marker_radius:.1f}" '
                f'fill="{marker_fill}" stroke="{marker_stroke}"><title>{html.escape(label)} · '
                f'{html.escape(x_value_prefix)}{int(x)} · {y:.3f}{html.escape(title_suffix)}</title></circle>'
            )
    legend_x = left
    for label, _, color, dash in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line class="legend-line" x1="{legend_x}" y1="18" x2="{legend_x+24}" y2="18" stroke="{color}"{dash_attr}/>')
        parts.append(f'<text class="legend-label" x="{legend_x+30}" y="22">{html.escape(label)}</text>')
        legend_x += 28 + max(84, len(label) * 7)
    parts.extend(
        [
            f'<text class="axis-label" x="{left + plot_w/2:.2f}" y="{height-10}" text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="axis-label" transform="translate(17 {top + plot_h/2:.2f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def svg_bar_chart(
    title: str,
    x_label: str,
    rows: list[tuple[str, float, str]],
    *,
    domain: tuple[float, float] | None = None,
    references: list[tuple[float, str]] | None = None,
    width: int = 780,
    row_height: int = 40,
) -> str:
    height = 74 + row_height * len(rows)
    values = [value for _, value, _ in rows]
    lo, hi = domain or (min(0.0, min(values)), max(values) * 1.12)
    left, right, top, bottom = 210, 48, 30, 44
    plot_w = width - left - right

    def sx(value: float) -> float:
        return left + (value - lo) / max(hi - lo, 1e-9) * plot_w

    zero = sx(0.0)
    parts = [
        f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{height-top-bottom}"/>',
        f'<line class="zero" x1="{zero:.2f}" y1="{top}" x2="{zero:.2f}" y2="{height-bottom}"/>',
    ]
    ticks = [lo + i * (hi - lo) / 4 for i in range(5)]
    for tick in ticks:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle">{tick:.2f}</text>')
    for value, label in references or []:
        x = sx(value)
        parts.append(f'<line class="reference" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="reference-label" x="{x+5:.2f}" y="{top+13}">{html.escape(label)}</text>')
    for index, (label, value, color) in enumerate(rows):
        y = top + index * row_height + 8
        x = min(zero, sx(value))
        bar_w = abs(sx(value) - zero)
        parts.append(f'<text class="bar-label" x="{left-12}" y="{y+17}" text-anchor="end">{html.escape(label)}</text>')
        parts.append(f'<rect class="bar" x="{x:.2f}" y="{y}" width="{bar_w:.2f}" height="22" fill="{color}"><title>{value:.4f}</title></rect>')
        anchor = "start" if value >= 0 else "end"
        dx = 7 if value >= 0 else -7
        parts.append(f'<text class="bar-value" x="{sx(value)+dx:.2f}" y="{y+17}" text-anchor="{anchor}">{value:.3f}</text>')
    parts.append(f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-7}" text-anchor="middle">{html.escape(x_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_interval_chart(
    title: str,
    x_label: str,
    rows: list[tuple[str, float, float, float, str]],
    *,
    domain: tuple[float, float] | None = None,
    reference: tuple[float, str] = (0.0, "no matched-control advantage"),
    width: int = 820,
    row_height: int = 44,
) -> str:
    """Return a horizontal point-and-interval chart for paired contrasts."""

    height = 78 + row_height * len(rows)
    observed = [value for _, mean, low, high, _ in rows for value in (mean, low, high)]
    observed.append(reference[0])
    if domain is None:
        lo, hi = min(observed), max(observed)
        pad = max((hi - lo) * 0.12, 0.002)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = domain
    left, right, top, bottom = 260, 62, 34, 46
    plot_w = width - left - right

    def sx(value: float) -> float:
        return left + (value - lo) / max(hi - lo, 1e-9) * plot_w

    parts = [
        f'<svg class="interval-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{height-top-bottom}"/>',
    ]
    for tick in [lo + i * (hi - lo) / 4 for i in range(5)]:
        x = sx(tick)
        parts.append(
            f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{height-bottom}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" '
            f'text-anchor="middle">{tick:.3f}</text>'
        )
    reference_x = sx(reference[0])
    parts.append(
        f'<line class="reference" x1="{reference_x:.2f}" y1="{top}" '
        f'x2="{reference_x:.2f}" y2="{height-bottom}"/>'
    )
    parts.append(
        f'<text class="reference-label" x="{reference_x+5:.2f}" y="{top+14}">'
        f'{html.escape(reference[1])}</text>'
    )
    for index, (label, mean, low, high, color) in enumerate(rows):
        y = top + index * row_height + 24
        parts.append(
            f'<text class="bar-label" x="{left-12}" y="{y+5:.2f}" '
            f'text-anchor="end">{html.escape(label)}</text>'
        )
        parts.append(
            f'<line class="ci-whisker" x1="{sx(low):.2f}" y1="{y:.2f}" '
            f'x2="{sx(high):.2f}" y2="{y:.2f}" stroke="{color}"/>'
        )
        for bound in (low, high):
            x = sx(bound)
            parts.append(
                f'<line class="ci-whisker" x1="{x:.2f}" y1="{y-6:.2f}" '
                f'x2="{x:.2f}" y2="{y+6:.2f}" stroke="{color}"/>'
            )
        parts.append(
            f'<circle cx="{sx(mean):.2f}" cy="{y:.2f}" r="5.5" fill="{color}">'
            f'<title>mean={mean:.6f}; 95% CI [{low:.6f}, {high:.6f}]</title></circle>'
        )
        anchor = "start" if sx(mean) < width - right - 66 else "end"
        dx = 9 if anchor == "start" else -9
        parts.append(
            f'<text class="bar-value" x="{sx(mean)+dx:.2f}" y="{y+5:.2f}" '
            f'text-anchor="{anchor}">{mean:+.3f}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-7}" '
        f'text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_map_error_chart(
    model: str,
    rows: list[dict[str, str]],
    *,
    width: int = 660,
    height: int = 390,
) -> str:
    """Plot predictive and refit errors for adjacent-layer rank-3 maps."""

    rows = sorted(rows, key=lambda row: int(row["target_layer"]))
    x_values = [int(row["target_layer"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.003, 10.0
    left, right, top, bottom = 78, 24, 44, 60
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * plot_w

    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def sy(value: float) -> float:
        value = min(max(float(value), y_min), y_max)
        return top + (log_max - math.log10(value)) / (log_max - log_min) * plot_h

    stable = {
        int(row["target_layer"]): (
            float(row["cv_centroid_r2"]) >= 0.9
            and float(row["bootstrap_map_relative_frobenius_median"]) <= 0.1
        )
        for row in rows
    }
    series: list[tuple[str, str, str, list[tuple[int, float, int]]]] = [
        (
            "CV centroid NRMSE",
            "#0f766e",
            "",
            [
                (
                    int(row["target_layer"]),
                    float(row["cv_centroid_normalized_rmse"]),
                    int(row["source_layer"]),
                )
                for row in rows
            ],
        ),
        (
            "bootstrap map error",
            "#7c3aed",
            "6 4",
            [
                (
                    int(row["target_layer"]),
                    float(row["bootstrap_map_relative_frobenius_median"]),
                    int(row["source_layer"]),
                )
                for row in rows
            ],
        ),
    ]
    parts = [
        f'<svg class="line-chart map-error-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} adjacent-layer rank-3 map errors across depth">',
        f'<title>{html.escape(model)} adjacent-layer rank-3 map errors</title>',
        '<desc>Two log-scale errors across answer-query layer boundaries. '
        'Lower green values indicate better held-out centroid prediction; lower purple values indicate '
        'better reproducibility after seed resampling and PCA-gauge alignment.</desc>',
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>',
    ]
    half_step = plot_w / max(x_max - x_min, 1) / 2
    for target_layer, is_stable in stable.items():
        if not is_stable:
            continue
        x = max(left, sx(target_layer) - half_step)
        x2 = min(width - right, sx(target_layer) + half_step)
        parts.append(
            f'<rect class="stable-band" x="{x:.2f}" y="{top}" width="{x2-x:.2f}" height="{plot_h}">'
            f'<title>L{target_layer-1}→L{target_layer}: both local-error cutoffs pass</title></rect>'
        )
    y_ticks = (0.003, 0.01, 0.03, 0.1, math.sqrt(0.1), 1.0, 3.0, 10.0)
    for tick in y_ticks:
        y = sy(tick)
        label = f"{tick:.3f}" if tick < 0.01 else (f"{tick:.2f}" if tick < 1 else f"{tick:g}")
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{label}</text>')
    x_ticks = sorted(
        {
            x_min,
            x_max,
            *[round(x_min + i * (x_max - x_min) / 5) for i in range(1, 5)],
        }
    )
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">{tick}</text>')
    for value, label in ((math.sqrt(0.1), "CV R²=.90"), (0.1, "bootstrap=.10")):
        y = sy(value)
        parts.append(f'<line class="reference" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="reference-label" x="{width-right-4}" y="{y-6:.2f}" text-anchor="end">{label}</text>')
    for label, color, dash, points in series:
        coords = " ".join(f"{sx(target):.2f},{sy(value):.2f}" for target, value, _ in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline class="series-line" points="{coords}" stroke="{color}"{dash_attr}/>')
        for target, value, source in points:
            local = "pass" if stable[target] else "fail"
            parts.append(
                f'<circle class="series-dot" cx="{sx(target):.2f}" cy="{sy(value):.2f}" r="2.6" '
                f'fill="{color}" stroke="#fff"><title>{html.escape(model)} L{source}→L{target}; '
                f'{html.escape(label)}={value:.4f}; local rule={local}</title></circle>'
            )
    legend_x = left
    for label, color, dash, _ in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line class="legend-line" x1="{legend_x}" y1="18" x2="{legend_x+24}" y2="18" stroke="{color}"{dash_attr}/>')
        parts.append(f'<text class="legend-label" x="{legend_x+30}" y="22">{html.escape(label)}</text>')
        legend_x += 32 + len(label) * 7
    parts.extend(
        [
            f'<rect class="stable-band legend-band" x="{legend_x}" y="11" width="14" height="12"/>',
            f'<text class="legend-label" x="{legend_x+20}" y="22">both cutoffs pass</text>',
            f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-10}" text-anchor="middle">Adjacent answer-query boundary (target layer shown)</text>',
            f'<text class="axis-label" transform="translate(17 {top+plot_h/2:.2f}) rotate(-90)" text-anchor="middle">Relative error (log; lower is better)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def svg_map_cosine_chart(
    model: str,
    rows: list[dict[str, str]],
    *,
    width: int = 660,
    height: int = 390,
) -> str:
    """Plot gauge-invariant orientation continuity of consecutive map operators."""

    rows = sorted(rows, key=lambda row: int(row["target_layer"]))
    points = [
        (
            int(row["target_layer"]),
            float(row["full_operator_cosine_to_next"]),
            int(row["source_layer"]),
        )
        for row in rows
        if row["full_operator_cosine_to_next"]
    ]
    x_values = [int(row["target_layer"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = -0.10, 1.05
    left, right, top, bottom = 78, 24, 44, 60
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg class="line-chart map-cosine-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} consecutive full-operator cosine across depth">',
        f'<title>{html.escape(model)} consecutive full-operator cosine</title>',
        '<desc>Frobenius cosine between each gauge-invariant ambient rank-3 map operator and the '
        'operator at the next answer-query boundary. Higher values mean more similar operator orientation; '
        'one means the same direction up to a positive scalar.</desc>',
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>',
    ]
    for tick in (-0.10, 0.0, 0.25, 0.50, 0.75, 1.0):
        y = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{tick:.2f}</text>')
    x_ticks = sorted(
        {
            x_min,
            x_max,
            *[round(x_min + i * (x_max - x_min) / 5) for i in range(1, 5)],
        }
    )
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">{tick}</text>')
    for value, label in ((1.0, "same orientation up to scale"), (0.0, "Frobenius-orthogonal")):
        y = sy(value)
        parts.append(f'<line class="reference" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="reference-label" x="{width-right-4}" y="{y-6:.2f}" text-anchor="end">{label}</text>')
    coords = " ".join(f"{sx(target):.2f},{sy(value):.2f}" for target, value, _ in points)
    parts.append(f'<polyline class="series-line" points="{coords}" stroke="#d97706"/>')
    for target, value, source in points:
        parts.append(
            f'<circle class="series-dot" cx="{sx(target):.2f}" cy="{sy(value):.2f}" r="2.8" '
            f'fill="#d97706" stroke="#fff"><title>{html.escape(model)}: T(L{source}→L{target}) '
            f'vs T(L{target}→L{target+1}); full-operator cosine={value:.4f}</title></circle>'
        )
    parts.extend(
        [
            '<line class="legend-line" x1="78" y1="18" x2="102" y2="18" stroke="#d97706"/>',
            '<text class="legend-label" x="108" y="22">full-operator cosine to next boundary</text>',
            f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-10}" text-anchor="middle">Adjacent answer-query boundary (target layer shown)</text>',
            f'<text class="axis-label" transform="translate(17 {top+plot_h/2:.2f}) rotate(-90)" text-anchor="middle">Cosine to next boundary (higher is closer)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _blend_rgb(start: tuple[int, int, int], end: tuple[int, int, int], value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    channels = [round(left + (right - left) * value) for left, right in zip(start, end)]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


def svg_accuracy_heatmap(model: str, rows: list[dict[str, str]]) -> str:
    """Render per-count exact accuracy and mean absolute error as two heat rows."""

    rows = sorted(rows, key=lambda row: int(row["gold_count"]))
    width, height = 820, 210
    left, right, top = 132, 22, 42
    cell_w = (width - left - right) / len(rows)
    cell_h = 46
    maximum_error = max(float(row["mean_absolute_error"]) for row in rows) or 1.0
    parts = [
        f'<svg class="baseline-heatmap" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} baseline exact accuracy and mean absolute error by gold needle count">',
        f"<title>{html.escape(model)} baseline by gold needle count</title>",
        f'<text class="heat-title" x="{left}" y="22">{html.escape(model)}</text>',
    ]
    for index, row in enumerate(rows):
        x = left + index * cell_w
        count = int(row["gold_count"])
        accuracy = float(row["accuracy"])
        error = float(row["mean_absolute_error"])
        prediction = float(row["mean_prediction"])
        acc_fill = _blend_rgb((241, 245, 249), (15, 118, 110), accuracy)
        err_fill = _blend_rgb((255, 247, 237), (180, 35, 24), error / maximum_error)
        parts.extend(
            [
                f'<text class="heat-x" x="{x + cell_w / 2:.2f}" y="37" text-anchor="middle">{count}</text>',
                f'<rect class="heat-cell" x="{x:.2f}" y="{top}" width="{cell_w:.2f}" height="{cell_h}" fill="{acc_fill}">'
                f'<title>N={count}; exact accuracy={accuracy:.3f}; examples={row["examples"]}</title></rect>',
                f'<text class="heat-value {"inverse" if accuracy > 0.53 else ""}" x="{x + cell_w / 2:.2f}" y="{top + 29}" text-anchor="middle">{100 * accuracy:.0f}%</text>',
                f'<rect class="heat-cell" x="{x:.2f}" y="{top + cell_h + 8}" width="{cell_w:.2f}" height="{cell_h}" fill="{err_fill}">'
                f'<title>N={count}; mean absolute error={error:.3f} counts; mean prediction={prediction:.3f}</title></rect>',
                f'<text class="heat-value {"inverse" if error / maximum_error > 0.55 else ""}" x="{x + cell_w / 2:.2f}" y="{top + cell_h + 37}" text-anchor="middle">{error:.2f}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="heat-row" x="{left - 10}" y="{top + 29}" text-anchor="end">Exact accuracy</text>',
            f'<text class="heat-row" x="{left - 10}" y="{top + cell_h + 37}" text-anchor="end">Mean |ŷ−N|</text>',
            f'<text class="axis-label" x="{left + (width-left-right)/2:.2f}" y="{height-8}" text-anchor="middle">Gold needle count N</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def svg_head_score_map(
    model: str,
    atlas_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
) -> str:
    """Render a layer-by-head map of answer-query broad-retrieval score."""

    values = {
        (int(row["layer"]), int(row["head"])): float(row["pool_primary"])
        for row in atlas_rows
    }
    selected = {
        (int(row["layer"]), int(row["head"])): int(row["rank"])
        for row in selected_rows
    }
    layers = list(range(max(layer for layer, _ in values) + 1))
    heads = list(range(max(head for _, head in values) + 1))
    maximum = max(values.values()) or 1.0
    width = 1010
    left, right, top, bottom = 66, 28, 40, 66
    cell_w = (width - left - right) / len(heads)
    cell_h = 15 if len(layers) >= 36 else 18
    height = top + bottom + cell_h * len(layers)
    parts = [
        f'<svg class="attention-map" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} layer by head broad retrieval score map at the answer query">',
        f"<title>{html.escape(model)} answer-query broad retrieval score map</title>",
    ]
    for layer in layers:
        y = top + layer * cell_h
        for head in heads:
            x = left + head * cell_w
            score = values.get((layer, head), 0.0)
            fill = _blend_rgb((247, 250, 252), (15, 118, 110), score / maximum)
            rank = selected.get((layer, head))
            class_name = "attention-cell selected" if rank else "attention-cell"
            parts.append(
                f'<rect class="{class_name}" x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}">'
                f'<title>L{layer}H{head}; broad score={score:.4f}'
                f'{f"; frozen rank={rank}" if rank else ""}</title></rect>'
            )
            if rank and cell_w >= 20:
                parts.append(
                    f'<text class="heat-rank" x="{x + cell_w/2:.2f}" y="{y + cell_h - 3:.2f}" text-anchor="middle">{rank}</text>'
                )
    head_step = 2 if len(heads) > 16 else 1
    for head in heads[::head_step]:
        x = left + (head + 0.5) * cell_w
        parts.append(f'<text class="heat-x" x="{x:.2f}" y="{height-bottom+19}" text-anchor="middle">{head}</text>')
    layer_step = 4 if len(layers) > 38 else 3
    for layer in layers[::layer_step]:
        y = top + (layer + 0.5) * cell_h + 4
        parts.append(f'<text class="heat-y" x="{left-9}" y="{y:.2f}" text-anchor="end">{layer}</text>')
    legend_x = left
    legend_y = height - 23
    for index in range(6):
        value = maximum * index / 5
        fill = _blend_rgb((247, 250, 252), (15, 118, 110), index / 5)
        parts.append(f'<rect x="{legend_x + index*44}" y="{legend_y-11}" width="44" height="10" fill="{fill}"/>')
    parts.extend(
        [
            f'<text class="heat-legend" x="{legend_x}" y="{legend_y+13}">0</text>',
            f'<text class="heat-legend" x="{legend_x+264}" y="{legend_y+13}" text-anchor="end">{maximum:.2f}</text>',
            f'<text class="heat-legend" x="{legend_x+282}" y="{legend_y+5}">broad score B</text>',
            f'<text class="axis-label" x="{left + (width-left-right)/2:.2f}" y="{height-bottom+42}" text-anchor="middle">Attention head H (zero-based)</text>',
            f'<text class="axis-label" transform="translate(17 {top + cell_h*len(layers)/2:.2f}) rotate(-90)" text-anchor="middle">Transformer layer L (zero-based)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def classifier_data() -> dict[str, list[dict[str, str]]]:
    root = REPORTS / "v4_4_extension" / "classification"
    paths = {
        "Qwen3-8B": root / "classification_all_qwen" / "answer_classifier_metrics.csv",
        "Gemma4-E4B": root / "classification_all_gemma" / "answer_classifier_metrics.csv",
    }
    reported_algorithms = {"logistic_l2", "nearest_centroid"}
    return {
        key: [
            row for row in read_csv(path)
            if row["algorithm"] in reported_algorithms
        ]
        for key, path in paths.items()
    }


def build(output: Path) -> None:
    extension = REPORTS / "v4_4_extension"
    base_html = (
        REPORTS
        / "v4_4_3"
        / "realistic_niah_v4_4_mechanism_report.html"
    ).read_text(encoding="utf-8")
    prompt_all = extract_embedded_json(base_html, "PROMPT_DATA")
    answer_all = extract_embedded_json(base_html, "ANSWER_DATA")
    # Keep all layers selectable while stripping unused PC4--PC6 columns from
    # the embedded rows.  The first nine columns contain metadata plus PC1--3.
    display_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "prompt": {"Qwen3-8B": {}, "Gemma4-E4B": {}},
        "answer": {"Qwen3-8B": {}, "Gemma4-E4B": {}},
    }
    for key, payload in prompt_all.items():
        model, layer = key.split("|")
        display_data["prompt"][model][layer] = {
            "layer": int(layer),
            "rows": [row[:9] for row in payload["rows"]],
        }
    for key, payload in answer_all.items():
        model, layer, cohort = key.split("|")
        if cohort != "all":
            continue
        display_data["answer"][model][layer] = {
            "layer": int(layer),
            "rows": [row[:9] for row in payload["rows"]],
        }

    classifiers = classifier_data()
    regression = read_csv(extension / "geometry" / "count_regression_summary.csv")
    ranks = read_csv(extension / "geometry" / "rank_and_compression_by_layer.csv")
    clusters = read_csv(extension / "geometry" / "clustering_summary.csv")
    counter_rows = read_csv(
        extension / "counter_properties" / "counter_property_metrics_by_layer.csv"
    )
    counter_selected = read_csv(
        extension / "counter_properties" / "selected_layer_counter_properties.csv"
    )
    gated_formula_rows = read_csv(
        extension / "all_token" / "gated_curve_formula_tests.csv"
    )
    followup_root = REPORTS / "v4_4_5_followup"
    followup = json.loads(
        (followup_root / "campaign_summary.json").read_text(encoding="utf-8")
    )
    model_order = ("Qwen3-8B", "Gemma4-E4B")
    exp19: dict[str, dict[str, Any]] = {}
    exp22: dict[str, dict[str, Any]] = {}
    exp22_registration: dict[str, dict[str, Any]] = {}
    exp22_synthetic: dict[str, dict[str, Any]] = {}
    exp23: dict[str, dict[str, Any]] = {}
    exp23_registration: dict[str, dict[str, Any]] = {}
    for model in model_order:
        exp19_payload = json.loads(
            (followup_root / "exp19" / model / "serial_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if exp19_payload.get("status") != "PASS" or set(exp19_payload.get("models", {})) != {model}:
            raise RuntimeError(f"Experiment 19 summary failed audit for {model}")
        exp19[model] = exp19_payload["models"][model]

        exp22[model] = json.loads(
            (followup_root / "exp22_v3" / model / "analysis_summary.json").read_text(
                encoding="utf-8"
            )
        )
        exp22_registration[model] = json.loads(
            (followup_root / "exp22_v3" / model / "canonical_registration.json").read_text(
                encoding="utf-8"
            )
        )
        exp22_synthetic[model] = json.loads(
            (followup_root / "exp22_v3" / model / "synthetic_audit.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            exp22[model].get("status") != "PASS"
            or exp22[model].get("model") != model
            or exp22[model].get("scientific_decision") != "not_supported"
            or not exp22[model].get("synthetic_relation_gate")
            or exp22[model].get("canonical_matched_block_gate")
        ):
            raise RuntimeError(f"Unexpected experiment 22 verdict for {model}")

        exp23[model] = json.loads(
            (followup_root / "exp23_v2" / model / "analysis_summary.json").read_text(
                encoding="utf-8"
            )
        )
        exp23_registration[model] = json.loads(
            (followup_root / "exp23_v2" / model / "outside_context_registration.json").read_text(
                encoding="utf-8"
            )
        )
        exp23_audit = json.loads(
            (followup_root / "exp23_v2" / model / "analysis_audit.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            exp23[model].get("status") != "PASS"
            or exp23[model].get("model") != model
            or exp23[model]["outside_context"].get("candidate_exceeds_both_controls")
            or exp23_audit != {
                "status": "PASS",
                "factorial_rows": 240,
                "outside_context_rows": 400,
            }
        ):
            raise RuntimeError(f"Unexpected experiment 23 audit or verdict for {model}")
    span_layerwise = json.loads(
        (
            REPORTS
            / "v4_4_5_followup"
            / "span_restoration"
            / "layerwise_seed_statistics.json"
        ).read_text(encoding="utf-8")
    )
    if span_layerwise.get("status") != "PASS":
        raise RuntimeError("Dense span layerwise seed statistics did not pass audit")
    span_attention_response = read_csv(
        REPORTS
        / "v4_4_5_followup"
        / "span_restoration"
        / "attention_response_canonical.csv"
    )
    attention_response_counts = {
        model: sum(row["model_label"] == model for row in span_attention_response)
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    if attention_response_counts != {"Qwen3-8B": 36, "Gemma4-E4B": 42}:
        raise RuntimeError(
            "Unexpected canonical attention-response coverage: "
            f"{attention_response_counts}"
        )
    for model, expected_layers, expected_heads in (
        ("Qwen3-8B", set(range(36)), 32),
        ("Gemma4-E4B", set(range(42)), 8),
    ):
        rows = [row for row in span_attention_response if row["model_label"] == model]
        if {int(row["patch_layer"]) for row in rows} != expected_layers:
            raise RuntimeError(f"Incomplete canonical attention layers for {model}")
        if any(
            int(row["frozen_heads"]) != expected_heads
            or int(row["seeds"]) != 30
            or row["counts"] != "1-10"
            for row in rows
        ):
            raise RuntimeError(f"Unexpected canonical attention provenance for {model}")

    # Question 21 is answered by the earlier V4.4.2 cue-removal grid.  The
    # intervention deletes only the two opening definition sentences; it keeps
    # the passage, counting question, numeric-output instruction, and assistant
    # formatting fixed.  Parse the frozen embedded analysis so the appendix
    # cannot drift from the original paired result.
    cue_report_path = (
        REPORTS
        / "v4_4_2"
        / "realistic_niah_v4_4_2_mode_geometry_attention_report.html"
    )
    cue_report = cue_report_path.read_text(encoding="utf-8")
    cue_prompt_geometry = extract_embedded_json(cue_report, "PROMPT_GEOM")
    if cue_prompt_geometry.get("schema_version") != "realistic_niah_v4_4_2_prompt_counter_geometry_v1":
        raise RuntimeError("Unexpected V4.4.2 prompt-cue geometry schema")
    cue_expected_coverage = {
        "Qwen3-8B": {"paired_prompts": 10, "paired_endpoint_states": 100, "layers": 36, "seeds": 10},
        "Gemma4-E4B": {"paired_prompts": 10, "paired_endpoint_states": 100, "layers": 42, "seeds": 10},
    }
    for model, expected in cue_expected_coverage.items():
        observed = cue_prompt_geometry["coverage"][model]
        if any(int(observed[key]) != value for key, value in expected.items()):
            raise RuntimeError(f"Unexpected V4.4.2 cue-removal coverage for {model}: {observed}")
        if int(observed["prompt_gold_count"]) != 10:
            raise RuntimeError(f"Cue-removal prompt counter must use final N=10 for {model}")
    cue_statistics = list(cue_prompt_geometry["statistics"].values())
    cue_display_layers = {"Qwen3-8B": 8, "Gemma4-E4B": 9}
    cue_display_rows = {
        model: cue_prompt_geometry["statistics"][f"{model}|prompt_counter|{layer}"]
        for model, layer in cue_display_layers.items()
    }
    cue_cka_chart = svg_line_chart(
        "Opening-cue removal: running-index centroid topology across depth",
        "Linear CKA between cue-present and cue-absent centroids",
        [
            (
                model,
                [
                    (float(row["layer"]), float(row["centroid_cka"]))
                    for row in sorted(
                        [item for item in cue_statistics if item["model"] == model],
                        key=lambda item: int(item["layer"]),
                    )
                ],
                color,
                dash,
            )
            for model, color, dash in (
                ("Qwen3-8B", "#0f766e", ""),
                ("Gemma4-E4B", "#7c3aed", "6 5"),
            )
        ],
        x_ticks=[0, 8, 16, 24, 32, 40],
        y_domain=(0.94, 1.005),
        reference=(1.0, "identical centroid relations"),
        width=760,
        height=350,
    )
    cue_ridge_chart = svg_bar_chart(
        "Opening-cue removal: shallow running-index readout",
        "Seed-held-out ridge R²",
        [
            (
                f"Qwen L8 · cue present",
                float(cue_display_rows["Qwen3-8B"]["r2_present"]),
                "#0f766e",
            ),
            (
                f"Qwen L8 · cue absent",
                float(cue_display_rows["Qwen3-8B"]["r2_absent"]),
                "#d97706",
            ),
            (
                f"Gemma L9 · cue present",
                float(cue_display_rows["Gemma4-E4B"]["r2_present"]),
                "#7c3aed",
            ),
            (
                f"Gemma L9 · cue absent",
                float(cue_display_rows["Gemma4-E4B"]["r2_absent"]),
                "#2563eb",
            ),
        ],
        domain=(-0.05, 0.90),
        references=[(0.0, "R² = 0")],
        width=760,
    )
    cue_appendix_rows = "".join(
        f"""<tr><td>{model}</td><td>L{cue_display_layers[model]}</td>
        <td>{f(row['centroid_cka'], 4)}</td>
        <td>{f(row['r2_present'])} / {f(row['r2_absent'])}</td>
        <td>{f(row['count_eta_present'])} / {f(row['count_eta_absent'])}</td>
        <td>{f(row['interaction_eta_sq'])}</td></tr>"""
        for model, row in cue_display_rows.items()
    )

    display_layers = {
        ("Qwen3-8B", "prompt_running"): 8,
        ("Gemma4-E4B", "prompt_running"): 9,
        ("Qwen3-8B", "answer_query"): 29,
        ("Gemma4-E4B", "answer_query"): 37,
    }
    geometry_rows: list[dict[str, object]] = []
    for (model, role), layer in display_layers.items():
        reg = pick(regression, model_label=model, role=role, layer=layer, algorithm="ridge")
        rank_population = "discovery" if role == "prompt_running" else "all_available"
        rank = pick(
            ranks,
            model_label=model,
            role=role,
            layer=layer,
            fit_population=rank_population,
        )
        cluster = pick(clusters, model_label=model, role=role, layer=layer)
        logistic = (
            pick(classifiers[model], layer=layer, algorithm="logistic_l2")
            if role == "answer_query"
            else None
        )
        nearest_centroid = (
            pick(classifiers[model], layer=layer, algorithm="nearest_centroid")
            if role == "answer_query"
            else None
        )
        geometry_rows.append(
            {
                "model": model,
                "role": role,
                "layer": layer,
                "accuracy": (
                    float(logistic["accuracy"])
                    if logistic is not None
                    else None
                ),
                "class_mad": (
                    float(logistic["count_mae"])
                    if logistic is not None
                    else None
                ),
                "centroid_accuracy": (
                    float(nearest_centroid["accuracy"])
                    if nearest_centroid is not None
                    else None
                ),
                "centroid_mad": (
                    float(nearest_centroid["count_mae"])
                    if nearest_centroid is not None
                    else None
                ),
                "r2": float(reg["r2_mean"]),
                "reg_mad": float(reg["mae_mean"]),
                "stable_rank": float(rank["stable_rank"]),
                "rank3_all": float(rank["total_variance_capture_k3"]),
                "rank3_centroid": float(rank["centroid_curve_capture_k3"]),
                "eta2": float(rank["count_eta_squared"]),
                "silhouette": float(cluster["silhouette_cosine_mean"]),
            }
        )

    behavior: dict[str, dict[str, float]] = {}
    for model, model_layers in display_data["answer"].items():
        payload = model_layers[str(display_layers[(model, "answer_query")])]
        rows = payload["rows"]
        gold = [int(row[5]) for row in rows]
        prediction = [int(row[3]) for row in rows]
        errors = [pred - target for pred, target in zip(prediction, gold)]
        wrong = [abs(error) for error in errors if error != 0]
        behavior[model] = {
            "accuracy": sum(error == 0 for error in errors) / len(errors),
            "mad": sum(abs(error) for error in errors) / len(errors),
            "signed": sum(errors) / len(errors),
            "wrong_mad": sum(wrong) / len(wrong),
            "rows": float(len(rows)),
        }

    # Layerwise representation plots deliberately separate supervised
    # decodability from unsupervised rank-3 geometry.  They answer different
    # questions and should not be used as substitutes for one another.
    representation_charts: list[str] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        prompt_rows = sorted(
            [
                row for row in regression
                if row["model_label"] == model
                and row["role"] == "prompt_running"
                and row["algorithm"] == "ridge"
            ],
            key=lambda row: int(row["layer"]),
        )
        answer_logistic_rows = sorted(
            [row for row in classifiers[model] if row["algorithm"] == "logistic_l2"],
            key=lambda row: int(row["layer"]),
        )
        answer_centroid_rows = sorted(
            [row for row in classifiers[model] if row["algorithm"] == "nearest_centroid"],
            key=lambda row: int(row["layer"]),
        )
        prompt_rank_rows = sorted(
            [
                row for row in ranks
                if row["model_label"] == model
                and row["role"] == "prompt_running"
                and row["fit_population"] == "discovery"
            ],
            key=lambda row: int(row["layer"]),
        )
        answer_rank_rows = sorted(
            [
                row for row in ranks
                if row["model_label"] == model
                and row["role"] == "answer_query"
                and row["fit_population"] == "all_available"
            ],
            key=lambda row: int(row["layer"]),
        )
        representation_charts.append(
            '<div class="chart-pair">'
            + svg_line_chart(
                f"{model}: prompt running-index ridge",
                "Seed-held-out ridge R²",
                [("needle-end ridge", [(float(row["layer"]), float(row["r2_mean"])) for row in prompt_rows], "#d97706", "")],
                y_domain=(-0.1, 1.0),
                reference=(0.0, "R² = 0"),
            )
            + svg_line_chart(
                f"{model}: answer-query exact-count classification",
                "Seed-held-out accuracy",
                [
                    ("L2 logistic", [(float(row["layer"]), float(row["accuracy"])) for row in answer_logistic_rows], "#2563eb", ""),
                    ("nearest centroid", [(float(row["layer"]), float(row["accuracy"])) for row in answer_centroid_rows], "#0f766e", "5 4"),
                ],
                y_domain=(0.0, 0.75),
                reference=(0.1, "10-class chance = 0.10"),
            )
            + "</div>"
            + '<div class="chart-pair geometry-pair">'
            + svg_line_chart(
                f"{model}: prompt rank-3 geometry",
                "PCA rank-3 variance capture",
                [
                    ("all endpoint states", [(float(row["layer"]), float(row["total_variance_capture_k3"])) for row in prompt_rank_rows], "#d97706", ""),
                    ("count centroids", [(float(row["layer"]), float(row["centroid_curve_capture_k3"])) for row in prompt_rank_rows], "#7c3aed", "5 4"),
                ],
                y_domain=(0.0, 1.03),
            )
            + svg_line_chart(
                f"{model}: answer-query rank-3 geometry",
                "PCA rank-3 variance capture",
                [
                    ("all answer states", [(float(row["layer"]), float(row["total_variance_capture_k3"])) for row in answer_rank_rows], "#2563eb", ""),
                    ("count centroids", [(float(row["layer"]), float(row["centroid_curve_capture_k3"])) for row in answer_rank_rows], "#0f766e", "5 4"),
                ],
                y_domain=(0.0, 1.03),
            )
            + "</div>"
        )

    baseline_by_count = read_csv(REPORTS / "v4_4_causal_v2" / "baseline_by_count.csv")
    baseline_heatmaps = {
        model: svg_accuracy_heatmap(
            model,
            [row for row in baseline_by_count if row["model_label"] == model],
        )
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    baseline_summary: dict[str, dict[str, float]] = {}
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        rows = [row for row in baseline_by_count if row["model_label"] == model]
        examples = sum(int(row["examples"]) for row in rows)
        baseline_summary[model] = {
            "examples": float(examples),
            "accuracy": sum(float(row["accuracy"]) * int(row["examples"]) for row in rows) / examples,
            "mad": sum(float(row["mean_absolute_error"]) * int(row["examples"]) for row in rows) / examples,
            "signed": sum(float(row["mean_signed_error"]) * int(row["examples"]) for row in rows) / examples,
        }

    token_stats = read_csv(extension / "token_corruption" / "token_corruption_statistics.csv")
    prompt_remove = read_csv(extension / "prompt_subspace_ablation" / "subspace_ablation_statistics.csv")
    formation_rows: list[tuple[str, float, str]] = []
    for model, color in (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed")):
        token = pick(
            token_stats,
            population="all",
            model_label=model,
            endpoint="absolute_error_increase",
            estimand="specificity",
        )
        remove = pick(
            prompt_remove,
            population="all",
            model_label=model,
            condition="actual_rank3_remove",
            endpoint="absolute_error_increase",
        )
        formation_rows.extend(
            [
                (f"{model} · active-needle corruption", float(token["mean"]), color),
                (f"{model} · endpoint rank-3 removal", float(remove["mean"]), color),
            ]
        )
    formation_chart = svg_bar_chart(
        "Prompt evidence versus decoded endpoint subspace",
        "Control-adjusted increase in absolute count error",
        formation_rows,
        domain=(-0.5, 10.0),
    )

    earlier_qwen = sorted(
        read_csv(extension / "endpoint_attention_mask" / "earlier_span_head_confirmation.csv"),
        key=lambda row: float(row["confirmation_preference_mean"]),
        reverse=True,
    )[0]
    earlier_gemma = sorted(
        read_csv(extension / "endpoint_attention_mask" / "gemma_earlier_span_head_confirmation.csv"),
        key=lambda row: float(row["confirmation_preference_mean"]),
        reverse=True,
    )[0]

    topk_root = REPORTS / "v4_4_causal_v2" / "full_span_topk"
    topk = read_csv(topk_root / "full_span_topk_primary_statistics.csv")
    topk_membership = read_csv(topk_root / "full_span_topk_membership.csv")
    selected_heads = {
        "Qwen3-8B": [
            row for row in topk_membership
            if row["model_label"] == "Qwen3-8B" and row["top_n"] == "32"
        ],
        "Gemma4-E4B": [
            row for row in topk_membership
            if row["model_label"] == "Gemma4-E4B" and row["top_n"] == "8"
        ],
    }
    head_atlas = read_csv(REPORTS / "v4_4" / "realistic_niah_v4_head_atlas.csv")
    attention_maps: dict[str, str] = {}
    selected_head_tables: dict[str, str] = {}
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        atlas_rows = [
            row for row in head_atlas
            if row["model"] == model
            and row["variant"] == "v4.4"
            and row["pooling"] == "span_sum"
        ]
        attention_maps[model] = svg_head_score_map(model, atlas_rows, selected_heads[model])
        atlas_lookup = {
            (int(row["layer"]), int(row["head"])): row for row in atlas_rows
        }
        selected_head_tables[model] = "".join(
            f"<tr><td>{row['rank']}</td><td>{row['head_label']}</td>"
            f"<td>{f(atlas_lookup[(int(row['layer']), int(row['head']))]['pool_sum'])}</td>"
            f"<td>{f(atlas_lookup[(int(row['layer']), int(row['head']))]['pool_coverage'])}</td>"
            f"<td>{f(atlas_lookup[(int(row['layer']), int(row['head']))]['pool_primary'])}</td></tr>"
            for row in selected_heads[model]
        )
    retrieval_series = []
    retrieval_intervals: dict[str, list[tuple[float, float, float, bool]]] = {}
    for model, color, dash in (
        ("Qwen3-8B", "#0f766e", ""),
        ("Gemma4-E4B", "#7c3aed", "6 5"),
    ):
        rows = [
            row
            for row in topk
            if row["model_label"] == model and row["analysis_population"] == "all_examples_signed"
        ]
        rows.sort(key=lambda row: int(row["top_n"]))
        retrieval_series.append(
            (
                model,
                [(float(row["top_n"]), float(row["primary_effect"])) for row in rows],
                color,
                dash,
            )
        )
        retrieval_intervals[model] = [
            (
                float(row["top_n"]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
                float(row["exact_sign_flip_p"]) < 0.05,
            )
            for row in rows
        ]
    retrieval_chart = svg_line_chart(
        "Broad-head ranked ablation dose",
        "Δ absolute count shift vs matched random heads",
        retrieval_series,
        x_label="Number of ablated heads K",
        x_value_prefix="K",
        x_ticks=[1, 2, 4, 8, 16, 32],
        y_domain=(-0.15, 2.25),
        reference=(0.0, "no excess shift"),
        intervals=retrieval_intervals,
        width=760,
        height=340,
    )

    retrieval_damage_series = []
    retrieval_damage_intervals: dict[str, list[tuple[float, float, float, bool]]] = {}
    for model, color, dash in (
        ("Qwen3-8B", "#0f766e", ""),
        ("Gemma4-E4B", "#7c3aed", "6 5"),
    ):
        rows = [
            row
            for row in topk
            if row["model_label"] == model
            and row["analysis_population"] == "clean_correct_only"
        ]
        rows.sort(key=lambda row: int(row["top_n"]))
        retrieval_damage_series.append(
            (
                model,
                [(float(row["top_n"]), float(row["primary_effect"])) for row in rows],
                color,
                dash,
            )
        )
        retrieval_damage_intervals[model] = [
            (
                float(row["top_n"]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
                float(row["exact_sign_flip_p"]) < 0.05,
            )
            for row in rows
        ]
    retrieval_damage_chart = svg_line_chart(
        "Broad-head ablation on clean-correct examples",
        "Δ correct→wrong rate vs matched random",
        retrieval_damage_series,
        x_label="Number of ablated heads K",
        x_value_prefix="K",
        x_ticks=[1, 2, 4, 8, 16, 32],
        y_domain=(-0.12, 0.75),
        reference=(0.0, "no excess accuracy damage"),
        intervals=retrieval_damage_intervals,
        width=760,
        height=340,
    )

    topk_result_rows: list[str] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for top_n in (1, 2, 4, 8, 16, 32):
            shift = pick(
                topk,
                model_label=model,
                top_n=top_n,
                analysis_population="all_examples_signed",
            )
            damage = pick(
                topk,
                model_label=model,
                top_n=top_n,
                analysis_population="clean_correct_only",
            )
            positive_seeds = round(
                float(shift["positive_seed_fraction"]) * int(shift["seed_clusters"])
            )
            shift_sig = float(shift["exact_sign_flip_p"]) < 0.05
            damage_sig = float(damage["exact_sign_flip_p"]) < 0.05
            topk_result_rows.append(
                "<tr>"
                f"<td>{model}</td><td>{top_n}</td>"
                f"<td>{float(shift['primary_effect']):+.3f} "
                f"[{f(shift['ci95_low'])}, {f(shift['ci95_high'])}]</td>"
                f"<td>{positive_seeds}/{shift['seed_clusters']}</td>"
                f"<td>{'是' if shift_sig else '否'} (p={p_text(shift['exact_sign_flip_p'])})</td>"
                f"<td>{float(damage['primary_effect']):+.3f} "
                f"[{f(damage['ci95_low'])}, {f(damage['ci95_high'])}]</td>"
                f"<td>{'是' if damage_sig else '否'} (p={p_text(damage['exact_sign_flip_p'])})</td>"
                "</tr>"
            )
    if len(topk_result_rows) != 12:
        raise RuntimeError("Top-K result table must contain 12 model-by-K rows")
    topk_result_table = "".join(topk_result_rows)

    answer_remove = read_csv(
        extension / "layerwise_subspace" / "answer_query_removal" / "layerwise_answer_query_removal_statistics.csv"
    )
    removal_series = []
    for model, color, dash in (
        ("Qwen3-8B", "#0f766e", ""),
        ("Gemma4-E4B", "#7c3aed", "6 5"),
    ):
        rows = [
            row
            for row in answer_remove
            if row["model_label"] == model
            and row["population"] == "all"
            and row["endpoint"] == "absolute_error_specificity"
        ]
        rows.sort(key=lambda row: int(row["layer"]))
        removal_series.append(
            (
                model,
                [(float(row["layer"]), float(row["mean_effect"])) for row in rows],
                color,
                dash,
            )
        )
    removal_chart = svg_line_chart(
        "Answer-query rank-3 removal across layers",
        "Δ absolute error vs orthogonal removal",
        removal_series,
        y_domain=(-0.25, 1.45),
        reference=(0.0, "no specificity"),
        width=760,
        height=350,
    )

    patch_rows = read_csv(REPORTS / "v4_4" / "v4_4_answer_query_patching.csv")
    patch_series = []
    for model, color, dash in (
        ("Qwen3-8B", "#0f766e", ""),
        ("Gemma4-E4B", "#7c3aed", "6 5"),
    ):
        rows = sorted(
            [row for row in patch_rows if row["model"] == model],
            key=lambda row: int(row["layer"]),
        )
        patch_series.append(
            (
                model,
                [(float(row["layer"]), float(row["eligible_donor_adoption_rate"])) for row in rows],
                color,
                dash,
            )
        )
    patch_chart = svg_line_chart(
        "Answer-state patching onset",
        "Donor-prediction adoption rate",
        patch_series,
        y_domain=(0.0, 1.05),
        width=760,
        height=350,
    )

    layer_maps = read_csv(
        extension / "layerwise_subspace" / "layer_maps" / "layerwise_linear_map_summary.csv"
    )
    answer_rank3_maps = {
        model: [
            row
            for row in layer_maps
            if row["model_label"] == model
            and row["role"] == "answer_query"
            and row["rank"] == "3"
        ]
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    map_error_charts = {
        model: svg_map_error_chart(model, rows)
        for model, rows in answer_rank3_maps.items()
    }
    map_cosine_charts = {
        model: svg_map_cosine_chart(model, rows)
        for model, rows in answer_rank3_maps.items()
    }
    map_stable_counts = {
        model: sum(
            float(row["cv_centroid_r2"]) >= 0.9
            and float(row["bootstrap_map_relative_frobenius_median"]) <= 0.1
            for row in rows
        )
        for model, rows in answer_rank3_maps.items()
    }
    selected_map_rows = {
        "Qwen3-8B": pick(
            answer_rank3_maps["Qwen3-8B"], source_layer=28, target_layer=29
        ),
        "Gemma4-E4B": pick(
            answer_rank3_maps["Gemma4-E4B"], source_layer=36, target_layer=37
        ),
    }

    transport = read_csv(
        extension / "layerwise_subspace" / "transport" / "layerwise_transport_condition_summary.csv"
    )
    transport_statistics = read_csv(
        extension / "layerwise_subspace" / "transport" / "layerwise_transport_statistics.csv"
    )
    transport_contrasts = {
        (model, contrast): pick(
            transport_statistics,
            model_label=model,
            source_layer=source,
            target_layer=target,
            contrast=contrast,
            metric="target_donor_fraction",
        )
        for model, source, target in (
            ("Qwen3-8B", 28, 29),
            ("Gemma4-E4B", 36, 37),
        )
        for contrast in ("aligned_dose_1_minus_orthogonal",)
    }
    transport_rows: list[tuple[str, float, str]] = []
    for model, source, target, color in (
        ("Qwen3-8B", 28, 29, "#0f766e"),
        ("Gemma4-E4B", 36, 37, "#7c3aed"),
    ):
        short_model = "Qwen" if model == "Qwen3-8B" else "Gemma"
        for condition, label in (
            ("matched_orthogonal", "orthogonal 1×"),
            ("aligned_dose_1", "aligned 1×"),
        ):
            row = pick(
                transport,
                model_label=model,
                source_layer=source,
                target_layer=target,
                condition=condition,
            )
            transport_rows.append((f"{short_model} L{source}→{target} · {label}", float(row["mean_target_donor_fraction"]), color))
    transport_chart = svg_bar_chart(
        "Adjacent-layer aligned transport",
        "Propagated target-chord coefficient F (1 = one R→D chord unit)",
        transport_rows,
        domain=(-0.08, 1.12),
        references=[(1.0, "one target chord")],
        width=860,
    )

    pooled = read_csv(REPORTS / "v4_4_causal_v2" / "correct_patching_pooled.csv")
    answer_patching = {
        model: pick(pooled, model_label=model, family="answer_patching")
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    prompt_patching = {
        model: pick(pooled, model_label=model, family="prompt_patching")
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }

    qwen_natural = json.loads(
        (REPORTS / "v4_4_4" / "realistic_niah_v4_4_4_analysis.json").read_text(encoding="utf-8")
    )
    qwen_rw = json.loads(
        (REPORTS / "v4_4_4" / "read_write" / "realistic_niah_v4_4_4_read_write_analysis.json").read_text(encoding="utf-8")
    )
    qwen_upstream = json.loads(
        (
            REPORTS
            / "v4_4_4"
            / "upstream_confirmation"
            / "realistic_niah_v4_4_4_upstream_confirmation_analysis.json"
        ).read_text(encoding="utf-8")
    )
    gemma_residual = json.loads(
        (
            REPORTS
            / "v4_4_4"
            / "gemma"
            / "residual"
            / "k2"
            / "realistic_niah_v4_4_4_residual_analysis.json"
        ).read_text(encoding="utf-8")
    )
    gemma_l29h4_analysis = json.loads(
        (
            REPORTS
            / "v4_4_4"
            / "gemma"
            / "search"
            / "l29h4"
            / "realistic_niah_v4_4_4_analysis.json"
        ).read_text(encoding="utf-8")
    )

    gemma_candidate = {
        row["endpoint"]: row
        for row in gemma_residual["summary"]
        if row["set_role"] == "candidate_core"
    }
    gemma_l29h4 = {
        endpoint: pick(
            gemma_l29h4_analysis["summary"],
            endpoint=endpoint,
            set_role="candidate_core",
        )
        for endpoint in (
            "natural_carrier_count_slope",
            "injection_dose_slope",
            "removal_error_axis_minus_control",
            "removal_margin_axis_minus_control",
            "donor_patch_transport",
        )
    }
    gemma_l29h4_specificity = {
        endpoint: pick(
            gemma_l29h4_analysis["summary"],
            endpoint=endpoint,
            set_role="candidate_specificity",
        )
        for endpoint in (
            "natural_carrier_count_slope__candidate_minus_control_mean",
            "injection_dose_slope__candidate_minus_control_mean",
            "removal_error_axis_minus_control__candidate_minus_control_mean",
            "removal_margin_axis_minus_control__candidate_minus_control_mean",
            "donor_patch_transport__candidate_minus_control_mean",
            "mediation_control_minus_axis_block__candidate_minus_control_mean",
        )
    }

    # These values are frozen primary estimands in the corresponding analysis
    # artifacts.  Pulling them through the JSON schema where possible keeps the
    # prose synchronized with the experiment outputs.
    qwen_natural_summary = qwen_natural["summary"]
    qwen_rw_summary = qwen_rw["summary"]
    qwen_upstream_primary = qwen_upstream["primary_decision"]
    qwen_nat = {
        endpoint: pick(
            qwen_natural_summary,
            endpoint=endpoint,
            set_role="candidate_core",
        )
        for endpoint in (
            "natural_carrier_count_slope",
            "injection_dose_slope",
            "removal_error_axis_minus_control",
            "removal_margin_axis_minus_control",
            "donor_patch_transport",
            "mediation_control_minus_axis_block",
        )
    }
    qwen_read_write = {
        metric: pick(qwen_rw_summary, metric=metric, stratum="all")
        for metric in (
            "read_full_behavior_transport",
            "read_routing_behavior_transport",
            "read_value_behavior_transport",
            "write_behavior_specificity",
        )
    }
    qwen_h19_loo = next(
        row for row in qwen_upstream["leave_one_out"] if row["removed_head"] == "H19"
    )
    qwen_axis_mediated_fraction = (
        float(qwen_nat["mediation_control_minus_axis_block"]["mean"])
        / float(qwen_nat["donor_patch_transport"]["mean"])
    )

    model_order = {"Qwen3-8B": 0, "Gemma4-E4B": 1}
    prompt_geometry_rows = sorted(
        [row for row in geometry_rows if row["role"] == "prompt_running"],
        key=lambda row: model_order[str(row["model"])],
    )
    answer_geometry_rows = sorted(
        [row for row in geometry_rows if row["role"] == "answer_query"],
        key=lambda row: model_order[str(row["model"])],
    )
    expected_models = ["Qwen3-8B", "Gemma4-E4B"]
    if [str(row["model"]) for row in prompt_geometry_rows] != expected_models:
        raise RuntimeError("Prompt geometry table must contain Qwen and Gemma exactly once")
    if [str(row["model"]) for row in answer_geometry_rows] != expected_models:
        raise RuntimeError("Answer geometry table must contain Qwen and Gemma exactly once")

    prompt_geometry_table_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row['model']))}</td>
          <td>needle-end · L{row['layer']}</td>
          <td>{f(row['r2'])}</td><td>{f(row['reg_mad'])}</td>
          <td>{f(row['rank3_all'])}</td><td>{f(row['rank3_centroid'])}</td>
          <td>{f(row['stable_rank'])}</td><td>{f(row['eta2'])}</td><td>{f(row['silhouette'])}</td>
        </tr>"""
        for row in prompt_geometry_rows
    )
    answer_geometry_table_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row['model']))}</td>
          <td>answer query · L{row['layer']}</td>
          <td>{pct(row['accuracy'])}</td><td>{f(row['class_mad'])}</td>
          <td>{pct(row['centroid_accuracy'])}</td><td>{f(row['centroid_mad'])}</td>
          <td>{f(row['r2'])}</td><td>{f(row['reg_mad'])}</td>
          <td>{f(row['rank3_all'])}</td><td>{f(row['rank3_centroid'])}</td>
          <td>{f(row['stable_rank'])}</td><td>{f(row['eta2'])}</td><td>{f(row['silhouette'])}</td>
        </tr>"""
        for row in answer_geometry_rows
    )

    behavior_rows = "".join(
        f"""<tr><td>{model}</td><td>{int(values['rows'])}</td><td>{pct(values['accuracy'])}</td>
        <td>{f(values['mad'])}</td><td>{f(values['signed'])}</td><td>{f(values['wrong_mad'])}</td></tr>"""
        for model, values in behavior.items()
    )

    best_rows = []
    algorithm_labels = {
        "logistic_l2": "L2 logistic",
        "nearest_centroid": "nearest centroid",
    }
    for model, rows in classifiers.items():
        for algorithm, label in algorithm_labels.items():
            algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
            # Accuracy is the primary metric; use lower count MAD only to make
            # tied maxima deterministic (Qwen logistic ties at L24 and L29).
            best = max(
                algorithm_rows,
                key=lambda row: (float(row["accuracy"]), -float(row["count_mae"])),
            )
            best_rows.append(
                f"{model} {label}: "
                f"L{best['layer']} / {pct(best['accuracy'])} / MAD {f(best['count_mae'])}"
            )

    counter_plot_path = extension / "counter_properties" / "counter_properties_by_layer.png"
    counter_plot_uri = (
        "data:image/png;base64,"
        + base64.b64encode(counter_plot_path.read_bytes()).decode("ascii")
    )
    counter_by_model = {row["model_label"]: row for row in counter_selected}
    counter_table_rows = "".join(
        f"""<tr><td>{model}</td><td>L{row['layer']}</td>
        <td>{f(row['trajectory_line_r2'])}</td>
        <td>{f(row['centroid_distance_vs_count_gap_spearman'])}</td>
        <td>{f(row['adjacent_step_pairwise_cosine_mean'])}</td>
        <td>{f(row['adjacent_step_length_cv'])}</td>
        <td>{f(row['discovery_confirmation_same_step_cosine_mean'])}</td>
        <td>{f(row['frozen_pc3_ridge_r2'])} / {f(row['frozen_pc3_ridge_mad'])}</td>
        <td>{f(row['confirmation_seed_projection_spearman_mean'])} / {pct(row['confirmation_adjacent_increment_positive_fraction'])}</td>
        <td>{f(row['position_count_spearman'])}</td>
        <td>{f(row['position_residual_pc3_grouped_ridge_r2'])} / {f(row['position_residual_pc3_grouped_ridge_mad'])}</td></tr>"""
        for model, row in counter_by_model.items()
    )
    gated_table_rows = []
    for model, layer in (("Qwen3-8B", 8), ("Gemma4-E4B", 9)):
        values = {
            (category, formula): float(
                pick(
                    gated_formula_rows,
                    model_label=model,
                    layer=layer,
                    category=category,
                    model=formula,
                )["incremental_r2_vs_category_baseline"]
            )
            for category, formula in (
                ("needle_endpoint", "endpoint_gated_curve"),
                ("needle_interior", "needle_span_gated_curve"),
                ("ordinary_passage", "ungated_prefix_curve"),
                ("hard_negative", "ungated_prefix_curve"),
            )
        }
        gated_table_rows.append(
            f"<tr><td>{model}</td><td>L{layer}</td>"
            f"<td>{f(values[('needle_endpoint', 'endpoint_gated_curve')])}</td>"
            f"<td>{f(values[('needle_interior', 'needle_span_gated_curve')])}</td>"
            f"<td>{f(values[('ordinary_passage', 'ungated_prefix_curve')])}</td>"
            f"<td>{f(values[('hard_negative', 'ungated_prefix_curve')])}</td></tr>"
        )
    gated_table_rows_html = "".join(gated_table_rows)

    span = followup["dense_span_restoration"]
    span_layer_rows_by_model = {
        model: sorted(
            [
                row
                for row in span_layerwise["layer_rows"]
                if row["model_label"] == model
            ],
            key=lambda row: int(row["layer"]),
        )
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    span_summary_by_model = {
        row["model_label"]: row for row in span_layerwise["model_summaries"]
    }

    def render_layer_segments(segments: list[list[int]]) -> str:
        rendered: list[str] = []
        for segment in segments:
            if len(segment) == 1:
                rendered.append(f"L{segment[0]}")
            else:
                rendered.append(f"L{segment[0]}–L{segment[-1]}")
        return ", ".join(rendered) if rendered else "none"

    span_layerwise_charts: dict[str, str] = {}
    span_transition_rows: list[str] = []
    for model, color, ticks in (
        ("Qwen3-8B", "#0f766e", [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]),
        ("Gemma4-E4B", "#7c3aed", [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 41]),
    ):
        rows = span_layer_rows_by_model[model]
        summary = span_summary_by_model[model]
        drop = summary["largest_adjacent_drop"]
        primary_label = "full span − ordinary"
        span_layerwise_charts[model] = svg_line_chart(
            f"{model} dense span restoration on confirmation seeds",
            "Needle-specific expected-error repair (counts)",
            [
                (
                    primary_label,
                    [(float(row["layer"]), float(row["mean"])) for row in rows],
                    color,
                    "",
                ),
                (
                    "endpoint − ordinary",
                    [
                        (
                            float(row["layer"]),
                            float(row["endpoint_minus_ordinary_mean"]),
                        )
                        for row in rows
                    ],
                    "#d97706",
                    "6 4",
                ),
            ],
            x_label="One-time post-block restoration layer ℓ (zero-based)",
            x_ticks=[float(value) for value in ticks],
            y_domain=(-0.5, 3.4),
            reference=(0.0, "no needle-specific repair"),
            vertical_references=[
                (
                    float(drop["from_layer"]) + 0.5,
                    f"max drop L{drop['from_layer']}→L{drop['to_layer']}: "
                    f"{float(drop['mean']):.3f}",
                )
            ],
            intervals={
                primary_label: [
                    (
                        float(row["layer"]),
                        float(row["ci95_low"]),
                        float(row["ci95_high"]),
                        bool(row["nominal_p_lt_0_05"]),
                    )
                    for row in rows
                ]
            },
            width=860,
            height=350,
        )
        span_transition_rows.append(
            f"<tr><td>{model}</td>"
            f"<td>{render_layer_segments(summary['positive_nominal_segments'])} "
            f"({int(summary['positive_nominal_layer_count'])} layers)</td>"
            f"<td>L{drop['from_layer']}→L{drop['to_layer']}: "
            f"{f(drop['mean'])} [{f(drop['ci95_low'])}, {f(drop['ci95_high'])}], "
            f"p={p_text(drop['exact_signflip_p'])}</td>"
            f"<td>{render_layer_segments(summary['negative_nominal_segments'])}</td>"
            f"<td>{f(summary['endpoint_minus_ordinary_min'])}…"
            f"{f(summary['endpoint_minus_ordinary_max'])}</td></tr>"
        )
    span_transition_rows_html = "".join(span_transition_rows)
    span_attention_charts: dict[str, str] = {}
    for model, color, ticks, usable_through in (
        ("Qwen3-8B", "#0f766e", [0, 4, 8, 12, 16, 20, 24, 28, 32, 35], 20),
        ("Gemma4-E4B", "#7c3aed", [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 41], 16),
    ):
        rows = sorted(
            [row for row in span_attention_response if row["model_label"] == model],
            key=lambda row: int(row["patch_layer"]),
        )
        span_attention_charts[model] = svg_line_chart(
            f"{model} answer-query attention response to full-span restoration",
            "Needle restoration response minus ordinary restoration response",
            [
                (
                    "needle-mass specificity",
                    [
                        (
                            float(row["patch_layer"]),
                            float(row["mass_specificity"]),
                        )
                        for row in rows
                    ],
                    "#2563eb",
                    "",
                ),
                (
                    "broad-score specificity",
                    [
                        (
                            float(row["patch_layer"]),
                            float(row["broad_specificity"]),
                        )
                        for row in rows
                    ],
                    color,
                    "6 4",
                ),
            ],
            x_label="One-time post-block full-span restoration layer ℓ",
            x_ticks=[float(value) for value in ticks],
            y_domain=(-0.02, 0.44),
            reference=(0.0, "needle response = ordinary response"),
            vertical_references=[
                (float(usable_through) + 0.5, f"canonical reuse through ≈L{usable_through}")
            ],
            width=860,
            height=350,
        )
    span_layer_lookup = {
        (row["model_label"], int(row["layer"])): row
        for row in span_layerwise["layer_rows"]
    }
    qwen_l20 = span_layer_lookup[("Qwen3-8B", 20)]
    qwen_l21 = span_layer_lookup[("Qwen3-8B", 21)]
    gemma_l16 = span_layer_lookup[("Gemma4-E4B", 16)]
    gemma_l17 = span_layer_lookup[("Gemma4-E4B", 17)]
    qwen_span_drop = span_summary_by_model["Qwen3-8B"]["largest_adjacent_drop"]
    gemma_span_drop = span_summary_by_model["Gemma4-E4B"]["largest_adjacent_drop"]
    span_landmarks = {row["model_label"]: row for row in span["formal_landmarks"]}
    span_landmark_chart = svg_bar_chart(
        "Dense full-span restoration landmarks",
        "full-needle minus ordinary expected-error repair (counts)",
        [
            ("Qwen discovery early plateau", span_landmarks["Qwen3-8B"]["discovery_early_plateau"], "#0f766e"),
            ("Qwen confirmation L19", span_landmarks["Qwen3-8B"]["half_boundary_confirmation_specificity"], "#0f766e"),
            ("Qwen confirmation L23", span_landmarks["Qwen3-8B"]["near_zero_confirmation_specificity"], "#0f766e"),
            ("Gemma discovery early plateau", span_landmarks["Gemma4-E4B"]["discovery_early_plateau"], "#7c3aed"),
            ("Gemma confirmation L17", span_landmarks["Gemma4-E4B"]["half_boundary_confirmation_specificity"], "#7c3aed"),
            ("Gemma confirmation L18", span_landmarks["Gemma4-E4B"]["near_zero_confirmation_specificity"], "#7c3aed"),
        ],
        domain=(-0.3, 3.2),
        references=[(0.0, "no direction-specific repair")],
        width=860,
    )
    span_landmark_rows = "".join(
        f"""<tr><td>{model}</td><td>{f(row['discovery_early_plateau'])}</td>
        <td>L{row['half_boundary_layer']}: {f(row['half_boundary_confirmation_specificity'])}</td>
        <td>L{row['near_zero_boundary_layer']}: {f(row['near_zero_confirmation_specificity'])}</td></tr>"""
        for model, row in span_landmarks.items()
    )

    answer_reuse_rows = "".join(
        f"""<tr><td>{row['model_label']}</td><td>L{row['layer']}</td>
        <td>{pct(row['nearest_centroid_accuracy'])}</td><td>{f(row['integer_mad'])}</td>
        <td>{f(row['discovery_rank3_all_state_capture'])}</td></tr>"""
        for row in followup["answer_geometry_reuse"]["rows"]
    )
    retrieval_geometry_rows = "".join(
        f"""<tr><td>{row['model_label']}</td><td>L{row['layer']}</td>
        <td>{pct(row['exact_classifier_accuracy'])}</td>
        <td>{pct(row['nearest_centroid_accuracy'])}</td>
        <td>{f(row['classifier_mad'])}</td></tr>"""
        for row in followup["retrieval_geometry"]["rows"]
    )
    retrieval_geometry_charts: list[str] = []
    for model, color in (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed")):
        rows = [
            row
            for row in followup["retrieval_geometry"]["rows"]
            if row["model_label"] == model
        ]
        layers = [float(row["layer"]) for row in rows]
        retrieval_geometry_charts.append(
            svg_line_chart(
                f"{model}: count readout from the summed broad-bank write",
                "held-out accuracy (fraction)",
                [
                    (
                        "exact classifier",
                        [
                            (float(row["layer"]), float(row["exact_classifier_accuracy"]))
                            for row in rows
                        ],
                        color,
                        "",
                    ),
                    (
                        "nearest centroid",
                        [
                            (float(row["layer"]), float(row["nearest_centroid_accuracy"]))
                            for row in rows
                        ],
                        "#d97706",
                        "6 4",
                    ),
                ],
                x_label="Frozen broad-bank layer (zero-based)",
                x_ticks=layers,
                y_domain=(0.0, 0.60),
                reference=(0.10, "10-class chance = 0.10"),
                width=720,
                height=340,
            )
        )

    retrieval_subspace = followup["retrieval_subspace"]
    retrieval_subspace_rows = "".join(
        f"""<tr><td>{row['model_label']}</td><td>L{row['layer']}</td>
        <td>{f(row['natural_specificity_mean'])}</td>
        <td>{f(row['restoration_mediation_mean'])}</td>
        <td>{f(row['mediated_fraction_mean'])}</td></tr>"""
        for row in retrieval_subspace["rows"]
    )
    retrieval_subspace_charts: list[str] = []
    for model, color in (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed")):
        rows = [row for row in retrieval_subspace["rows"] if row["model_label"] == model]
        retrieval_subspace_charts.append(
            svg_line_chart(
                f"{model}: frozen retrieval-subspace removal",
                "direction-specific damage (counts)",
                [
                    (
                        "natural specificity",
                        [(float(row["layer"]), float(row["natural_specificity_mean"])) for row in rows],
                        color,
                        "",
                    ),
                    (
                        "restoration mediation",
                        [(float(row["layer"]), float(row["restoration_mediation_mean"])) for row in rows],
                        "#d97706",
                        "6 4",
                    ),
                ],
                x_label="Frozen intervention layer",
                y_domain=(-0.1, 0.6),
                reference=(0.0, "aligned = orthogonal"),
                x_ticks=[float(row["layer"]) for row in rows],
            )
        )

    exp19_chart = svg_bar_chart(
        "Same-forward ordered partial serial mediation",
        "direction-specific expected-count effect (counts)",
        [
            (
                f"{model.replace('3-8B', '').replace('4-E4B', '')} · {label}",
                float(exp19[model][metric]["mean"]),
                color,
            )
            for model, color in (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed"))
            for label, metric in (
                ("source repair", "source_repair"),
                ("retrieval mediation", "retrieval_mediation"),
                ("late mediation", "late_mediation"),
            )
        ],
        domain=(0.0, 3.2),
        references=[(0.0, "matched-control difference = 0")],
        width=840,
    )
    exp19_rows = "".join(
        f"""<tr><td>{model}</td>
        <td>{f(values['source_repair']['mean'])} [{f(values['source_repair']['ci95_low'])}, {f(values['source_repair']['ci95_high'])}]</td>
        <td>{f(values['retrieval_mediation']['mean'])} [{f(values['retrieval_mediation']['ci95_low'])}, {f(values['retrieval_mediation']['ci95_high'])}]</td>
        <td>{f(values['late_mediation']['mean'])} [{f(values['late_mediation']['ci95_low'])}, {f(values['late_mediation']['ci95_high'])}]</td>
        <td>{f(values['joint_interaction']['mean'])} [{f(values['joint_interaction']['ci95_low'])}, {f(values['joint_interaction']['ci95_high'])}]</td>
        <td>{f(values['remaining_repair']['mean'])} [{f(values['remaining_repair']['ci95_low'])}, {f(values['remaining_repair']['ci95_high'])}]</td></tr>"""
        for model, values in exp19.items()
    )

    exp22_selected: dict[str, dict[str, Any]] = {}
    for model in model_order:
        registration = exp22_registration[model]
        exp22_selected[model] = next(
            row
            for row in exp22_synthetic[model]["head_summaries"]
            if int(row["layer"]) == int(registration["source_layer"])
            and int(row["head"]) == int(registration["source_head"])
        )
    exp22_chart = svg_interval_chart(
        "Canonical test of the frozen induction-like edge registry",
        "candidate minus attention/distance-matched control: expected absolute error (counts)",
        [
            (
                model,
                float(exp22[model]["metrics"]["expected_absolute_error_candidate_minus_control"]["mean"]),
                float(exp22[model]["metrics"]["expected_absolute_error_candidate_minus_control"]["ci95_low"]),
                float(exp22[model]["metrics"]["expected_absolute_error_candidate_minus_control"]["ci95_high"]),
                color,
            )
            for model, color in (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed"))
        ],
        domain=(-0.04, 0.012),
        reference=(0.0, "registered specificity requires > 0"),
        width=840,
    )
    exp22_rows = "".join(
        f"""<tr><td>{model}</td><td>L{int(exp22_registration[model]['source_layer'])}H{int(exp22_registration[model]['source_head'])}</td>
        <td>{f(exp22_selected[model]['repeated_relation_mean'], 5)}</td>
        <td>{f(exp22_selected[model]['reassignment_follow_mean'], 5)}</td>
        <td>{f(exp22_selected[model]['unique_anchor_abs_mean'], 5)} / {f(exp22_selected[model]['ordinary_repeat_abs_mean'], 5)}</td>
        <td>{f(exp22[model]['metrics']['expected_absolute_error_candidate_minus_control']['mean'], 5)} [{f(exp22[model]['metrics']['expected_absolute_error_candidate_minus_control']['ci95_low'], 5)}, {f(exp22[model]['metrics']['expected_absolute_error_candidate_minus_control']['ci95_high'], 5)}]</td>
        <td>not supported</td></tr>"""
        for model in model_order
    )

    exp23_factor_chart = svg_bar_chart(
        "Controlled identity/context/position deformation in the frozen rank-3 basis",
        "confirmation held-out incremental ΔR²",
        [
            (
                f"{model.replace('3-8B', '').replace('4-E4B', '')} · {factor}",
                float(exp23[model]["factorial"]["confirmation_incremental_delta_r2"][factor]),
                color,
            )
            for model, color in (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed"))
            for factor in ("identity", "context", "position")
        ],
        domain=(-0.005, 0.020),
        references=[(0.0, "no incremental held-out prediction")],
        width=840,
    )
    exp23_specificity_chart = svg_interval_chart(
        "Outside-context halo-edge specificity",
        "candidate removal minus matched control: expected absolute error (counts)",
        [
            (
                f"{model.replace('3-8B', '').replace('4-E4B', '')} · {label}",
                float(exp23[model]["outside_context"]["metrics"][metric]["mean"]),
                float(exp23[model]["outside_context"]["metrics"][metric]["ci95_low"]),
                float(exp23[model]["outside_context"]["metrics"][metric]["ci95_high"]),
                color,
            )
            for model, color in (("Qwen3-8B", "#0f766e"), ("Gemma4-E4B", "#7c3aed"))
            for label, metric in (
                ("vs distance-random", "expected_error_candidate_minus_distance_random"),
                ("vs attention-mass", "expected_error_candidate_minus_attention_mass"),
            )
        ],
        domain=(-0.014, 0.030),
        reference=(0.0, "registered specificity requires CI > 0"),
        width=840,
    )
    exp23_rows = "".join(
        f"""<tr><td>{model}</td>
        <td>{f(exp23[model]['factorial']['confirmation_full_model_r2'], 4)}</td>
        <td>{f(exp23[model]['factorial']['confirmation_incremental_delta_r2']['identity'], 4)} / {f(exp23[model]['factorial']['confirmation_incremental_delta_r2']['context'], 4)} / {f(exp23[model]['factorial']['confirmation_incremental_delta_r2']['position'], 4)}</td>
        <td>L{int(exp23_registration[model]['source_layer'])}H{int(exp23_registration[model]['source_head'])}</td>
        <td>{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_distance_random']['mean'], 4)} [{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_distance_random']['ci95_low'], 4)}, {f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_distance_random']['ci95_high'], 4)}]</td>
        <td>{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_attention_mass']['mean'], 4)} [{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_attention_mass']['ci95_low'], 4)}, {f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_attention_mass']['ci95_high'], 4)}]</td>
        <td>not supported</td></tr>"""
        for model in model_order
    )

    status_labels = {
        "verified": "已验证",
        "falsified": "已证伪",
        "partial": "部分回答",
        "open": "未完成",
        "closed": "已关闭",
    }
    extension_claims = [
        (1, "Prompt centroid trajectory 是否低维但 noisy", "verified", "30 seeds×10 running indices；discovery-fitted PCA/rank metrics", "centroid rank-3 94.7%–97.9%，all-state 59.8%–72.9%，silhouette 接近 0", "平均 count trajectory 低维；单样本不是十个紧密、彼此分离的 clusters。"),
        (2, "Needle-end running index 是否可预测", "verified", "seed-held-out PCA-32 ridge；另审计 frozen PC1–PC3 ordinal geometry", "Qwen L8 full probe R²=0.945/MAD=0.561；Gemma L9 0.719/1.249；3-PC held-out R²={:.3f}/{:.3f}".format(float(counter_by_model['Qwen3-8B']['frozen_pc3_ridge_r2']), float(counter_by_model['Gemma4-E4B']['frozen_pc3_ridge_r2'])), "running index 可线性读取；这里不需要、也不汇报 prompt exact-count classifier。"),
        (3, "Active needle evidence 是否因果必要", "verified", "全部 active spans 替换；同长度、同 token-budget ordinary replacement control", "absolute-error specificity Qwen +8.930、Gemma +8.780 counts", "模型确实依赖 active needle evidence；这不是把 candidate 与无干预 baseline 混比。"),
        (4, "Prompt 的可复用信息是 endpoint 还是 whole span", "verified", "canonical seeds 1234–1263×counts 1–10；endpoint/full-span/ordinary restoration 在 Qwen L0–35、Gemma L0–41 逐层扫描", "endpoint−ordinary 全层接近 0；whole-span 正向 nominal window 为 Qwen L0–20、Gemma L0–16", "可复用 source 分散在完整 needle span，不是 endpoint 单点寄存器；Qwen 后段渐降，Gemma L16→17 cliff。"),
        (5, "Broad score、attention map 与 frozen head sets 是否定义清楚", "verified", "discovery 上按 B=M×C 排名；Qwen top-32、Gemma top-8 冻结后进入干预", "报告给出 layer×head map、完整 membership、mass M、coverage C 与 score B", "broad 同时要求较大 needle mass 与多-span coverage；attention map 只描述 routing。"),
        (6, "Broad-ranked heads 是否有 matched-control 因果效应", "verified", "top-K answer-query ablation vs layer-distribution-matched random heads", "Qwen K32 +1.623 counts；Gemma K8 +0.767；correct-only damage 同方向", "这些 head banks 被自然行为使用，但非单调 K 曲线不支持“独立标量计数头”的解释。"),
        (7, "Broad-bank output 是否含低秩 count geometry", "verified", "3,000 broad-bank answer-query states；discovery fit、confirmation readout", "rank-3 centroid capture 0.968–0.995；exact accuracy 38%–54%；silhouette −0.098–0.011", "存在低秩可读 geometry，但单样本仍 noisy，且 fitted basis 不是跨层固定 counter。"),
        (8, "Broad-bank rank-3 是否自然参与 aggregation", "verified", "7 个 frozen layers；每层 100 paired units；aligned removal vs equal-norm orthogonal；natural/restored 两种状态", "Qwen L21/L23 restoration mediation +0.166/+0.265；Gemma L29 +0.527；其余 frozen late layers约 0", "count-aligned retrieval subspace 在局部 aggregation window 被自然计算使用。"),
        (9, "Answer query 是否形成可比较的 exact-count manifold", "verified", "seed-held-out PCA-32 nearest-centroid 为主要紧致度/可分性读数；L2 logistic 为辅助 robustness；3D 层只服务展示", "代表层 accuracy 约 53%–56%，nearest-centroid MAD 0.615–0.640；post-hoc 3D display 的 61%/63% 不作正式比较", "建立 non-thinking counter-manifold baseline，供后续按同一协议与 native thinking 比较；不把 classifier 当作机制证据。"),
        (10, "Late answer state 是否可执行并影响输出", "verified", "full donor answer-state patch + answer rank-3 vs equal-norm orthogonal removal", "correct-only donor hit Qwen 96.6%、Gemma 96.0%；peak removal +0.878/+1.222 counts", "晚层完整 answer state 具有充分性；其 count-aligned component 具有方向特异必要性。"),
        (11, "Late write/transport 是否能定位", "verified", "Qwen natural OV injection/removal/mediation；Gemma exact residual mediation；adjacent 1× matched control", "Qwen OV slope +0.0640、removal +0.0732；Gemma L37 residual mediation +0.0864；1× transport约 0.95/0.98", "Qwen 有较局部的 OV writer，Gemma 有较分布式的 residual path；功能阶段相同，实现粒度不同。"),
        (12, "同一 γ(n) 是否适用于 needle span 内所有 token", "falsified", "all-token frozen-PCA formula test：endpoint-gated、span-gated、ungated prefix curves vs category baseline", "endpoint ΔR² Qwen/Gemma +0.551/+0.326；interior span-gated −0.057/−0.036", "支持 endpoint-restricted descriptive curve；否定整个 span 每个 token 共用同一 γ(n) 的强版本。"),
        (13, "Decoded endpoint rank-3 是否是强局部必要 counter", "falsified", "所有 endpoints 同层 rank-3 removal vs realized-norm-matched orthogonal residual basis", "Qwen 层范围 −0.022…+0.056；Gemma −0.011…+0.022", "当前线性 endpoint rank-3 不能解释主要行为效应，不能称为强局部寄存器。"),
        (14, "同一 broad rank-3 counter 是否持续到更晚层", "falsified", "frozen retrieval-subspace 在 aggregation 与 late layers 逐层干预", "Qwen L23 +0.265 后 L24/L26/L27=+0.002/+0.006/+0.008；Gemma L29 +0.527、L35 −0.048", "因果作用集中在局部 retrieval window，不是跨后层不变的 persistent counter。"),
        (15, "Residual 是否逐层保持严格 identity map", "falsified", "adjacent rank-3 maps + ambient-operator cosine + one-block intervention", "selected map cosine 0.767/0.796，而非 1；aligned transport高、orthogonal近 0", "后续 block 选择性接收 count-aligned change，但不会原样复制完整三维 operator。"),
        (16, "Gemma 是否与 Qwen 一样有局部 OV head set", "falsified", "L35H2 与 {L29H4,L35H2} natural carrier/injection/removal package", "完整 localized-OV 判据未通过；L37 exact residual mediation +0.0864", "Gemma 的支持结论是分布式 residual mediator，而不是与 Qwen 同构的局部 OV writer。"),
        (17, "Prompt noise 的主要来源是什么", "partial", "count/seed-context/interaction decomposition + grouped cubic absolute-position control；与原 23 的 attention controls 联合解释", "代表层 count/seed/interaction 方差占比：Qwen 0.599/0.161/0.241，Gemma 0.385/0.228/0.386；position-count ρ≈0.965；去 position 后 3-PC R² Qwen {:.3f}、Gemma {:.3f}".format(float(counter_by_model['Qwen3-8B']['position_residual_pc3_grouped_ridge_r2']), float(counter_by_model['Gemma4-E4B']['position_residual_pc3_grouped_ridge_r2'])), "绝对位置解释 frozen 前三维的大部分 ordering；seed/context 与交互仍贡献噪声，但 token identity、context、attention 的独立因果份额尚未完全识别。当前论文无需继续细分。"),
        (18, "为什么 prompt manifold 浅层出现、answer manifold 深层出现", "partial", "跨层 prompt probe、answer classifier、dense restoration、restoration→attention response、answer patch/removal timing", "prompt 浅层已可读；source reuse 在 Qwen 约到 L20、Gemma约到 L16；answer 可执行性在中后层上升", "局部 occurrence 可早期记录；全局 answer query 需等待 retrieval 与 consolidation。时序与因果边界成立，但“架构为何必然如此”不是单一干预可证明的命题。"),
        (19, "是否建立完整 distributed prompt evidence→retrieval→late answer→output 因果链", "verified", "canonical confirmation seeds 1254–1263×counts 1–10；同一 forward 的 11-arm source restoration、retrieval/late aligned-vs-orthogonal removal 与 2×2 joint block；每模型 1,100 rows、100 paired units、10,000 bootstrap draws", "source repair Qwen/Gemma +2.674/+2.670 counts；retrieval mediation +0.327/+0.521；late mediation +1.118/+1.215；三项 ordered criteria 均 PASS，且更晚 block 对已计算的 retrieval readout 变化严格为 0", "同一试次内支持 ordered partial serial mediation：分布式 span evidence 会重配 retrieval，局部 count-aligned retrieval 会影响后续 late state，late state 再影响输出。负 interaction 与剩余 repair 表明路径有重叠和 bypass；不支持唯一通道或一枚固定 basis 原样跨层传递。详见 Appendix B。"),
        (20, "是否需要对所有 non-needle token 做 frozen-PCA census", "partial", "all-token capture 已含 endpoint、interior、hard-negative 与确定性 ordinary-passage samples", "ordinary/hard-negative 的 ungated prefix curve ΔR² 为负，未显示与 endpoint 相同 trajectory", "已有足够多类负对照支持当前限定结论；逐 token 无遗漏 census 成本高且不会改变 span-level mechanism，故不再扩展。"),
        (22, "经典 induction-head micro-circuit 是否是 canonical running-index update 的特异机制", "falsified", "独立 30×4 synthetic relation-following assay 冻结一个 head/model；随后在 seeds 1254–1263×counts 1–10 对 previous-successor natural edges 做 pre-O αV subtraction，并与 layer/head/distance/edge-count/attention-mass matched ordinary edges 比较；counts 2–10 为主分析", "synthetic gate 保留 Qwen L5H13 与 Gemma L5H0；但 canonical candidate-minus-control expected-error 为 −0.02193 [−0.03311,−0.01076] 与 −0.01207 [−0.02499,0.00127]，两模型决策均 not_supported", "存在 induction-like relation-following head，但预注册的 canonical edge-specific necessity 不成立；因此不能把 earlier-span routing 定名为已验证的 classical induction-head mechanism。该否定不排除分布式 span evidence、其他 registry 或 fully renormalized QK counterfactual。详见 Appendix C。"),
        (23, "预注册的 identity/context/position nuisance model 与 selected outside-halo edge specificity 是否成立", "falsified", "冻结 Qwen L8/Gemma L9 rank-3 basis 做 30 seeds×8 cells factorial（160 discovery、80 confirmation、2,400 endpoint states/model）；另在 100 confirmation units 上阻断 natural-attention-ranked ordinary halo edges，并分别匹配 exact-distance random 与 attention-mass controls", "factorial held-out full R² 为 −0.0221/−0.0893；最大 factor ΔR² 仅 Qwen position +0.0175、Gemma identity +0.0031。candidate removal 对两个 controls 的 expected-error CI 在两模型均跨 0，candidate_exceeds_both_controls=false", "强解释包被否定：三类受控操作未形成稳定的 held-out nuisance model，选定 halo edges 也没有超出两个 matched controls 的特异必要性。该结果不把自然 prompt noise 唯一分解，也不否定广泛 outside context 与 needle span 的分布式协同。详见 Appendix D。"),
        (25, "固定 prompt endpoint rank-3 是否直接变成 answer rank-3", "falsified", "prompt endpoint aligned removal、full-span restoration、retrieval-subspace mediation 与 late answer interventions 联合判定", "endpoint rank-3 removal 近 0；full-span restoration 强正；retrieval basis 只在局部窗口有效且跨层 operator 非 identity", "否定“一枚固定三维 prompt counter 直接搬到 answer”的强版本。它不是原 19 所缺的必要箭头；原 19 需要的是允许各阶段重参数化的串联中介。"),
        (21, "Opening counting-definition cue 是否为 running geometry 的必要条件", "falsified", "V4.4.2 paired opening-definition removal；完整删除边界、coverage、计算定义与数值见 Appendix A", "两个预选浅层均保留近乎相同的 centroid topology 与相近的 seed-held-out readout；同时 full-state cue displacement 仍随 running index 系统变化", "“opening definition cue 是必要条件”的强命题被证伪：没有它仍形成有序、可读的 running geometry；但 cue 会调制完整 state，因此非必要不等于无作用。该结果不能外推为删除全部 task/query instructions。"),
        (24, "是否继续做跨 final-count N 的 prefix invariance", "closed", "不再运行；当前全部 running-index capture 的 final N=10，报告始终限定为 position-confounded counter-like record", "无新增实验；现有证据不用于声称跨 final-N 的抽象 counter", "该实验只有在升级为 abstract-counter 主张时才必要；当前 mechanism 不作此主张，因此关闭并保留为范围边界。"),
    ]
    status_counts = {key: 0 for key in status_labels}
    for _, _, status, _, _, _ in extension_claims:
        status_counts[status] += 1
    if len(extension_claims) != 25 or status_counts != {
        "verified": 12,
        "falsified": 9,
        "partial": 3,
        "open": 0,
        "closed": 1,
    }:
        raise RuntimeError(f"Unexpected extension audit partition: {status_counts}")
    extension_audit_rows = "".join(
        f"""<tr><td>{source_id}</td><td>{question}</td>
        <td><span class="status status-{status}">{status_labels[status]}</span></td>
        <td>{setting}</td><td>{result}</td><td>{conclusion}</td></tr>"""
        for source_id, question, status, setting, result, conclusion in extension_claims
    )

    payload_json = json.dumps(display_data, separators=(",", ":"), ensure_ascii=False)

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Non-thinking V4.4/V4.4.5：从双位置表征到 broad retrieval 与 late write</title>
<style>
:root {{ --ink:#172033; --muted:#5f6b7a; --line:#d8dee9; --paper:#ffffff; --wash:#f4f6f8;
  --teal:#0f766e; --violet:#7c3aed; --amber:#d97706; --blue:#2563eb; --red:#b42318; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; color:var(--ink); background:#eef1f5; font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.68; }}
a {{ color:#155eef; text-decoration:none; }}
.page {{ width:min(1180px,calc(100% - 32px)); margin:24px auto 72px; background:var(--paper); box-shadow:0 18px 50px rgba(23,32,51,.10); }}
header {{ padding:54px 64px 36px; border-bottom:1px solid var(--line); background:linear-gradient(135deg,#f8fafc 0%,#eef8f7 52%,#f6f1ff 100%); }}
.eyebrow {{ margin:0 0 8px; color:var(--teal); letter-spacing:.12em; text-transform:uppercase; font-size:12px; font-weight:700; }}
h1 {{ max-width:900px; margin:0; font-family:Georgia,"Noto Serif SC",serif; font-size:clamp(32px,5vw,55px); line-height:1.12; letter-spacing:-.02em; font-weight:600; }}
.dek {{ max-width:880px; margin:20px 0 0; color:#3c4858; font-size:18px; }}
.meta {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:24px; color:var(--muted); font-size:13px; }}
nav {{ position:sticky; top:0; z-index:20; display:flex; gap:18px; padding:11px 64px; overflow-x:auto; white-space:nowrap; background:rgba(255,255,255,.95); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); font-size:13px; }}
main {{ padding:0 64px 64px; }}
section {{ padding:54px 0 20px; border-bottom:1px solid var(--line); }}
section:last-child {{ border-bottom:0; }}
h2 {{ margin:0 0 16px; font-family:Georgia,"Noto Serif SC",serif; font-size:32px; line-height:1.25; font-weight:600; }}
h3 {{ margin:34px 0 10px; font-size:20px; line-height:1.35; }}
h4 {{ margin:24px 0 8px; font-size:16px; }}
p {{ margin:10px 0; }}
.lead {{ max-width:920px; color:#344054; font-size:17px; }}
.claim {{ padding:18px 20px; margin:22px 0; border-left:4px solid var(--teal); background:#f0fdfa; }}
.claim strong {{ color:#075e58; }}
.boundary {{ border-left-color:var(--amber); background:#fff8eb; }}
.boundary strong {{ color:#9a4b00; }}
.mechanism {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:28px 0; }}
.stage {{ position:relative; min-height:210px; padding:20px; border:1px solid var(--line); background:#fff; }}
.stage::after {{ content:"→"; position:absolute; right:-20px; top:43%; z-index:2; color:#98a2b3; font-size:28px; }}
.stage:last-child::after {{ display:none; }}
.stage-no {{ color:var(--muted); font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:.12em; }}
.stage h3 {{ margin:8px 0; }}
.stage p {{ color:#475467; font-size:14px; }}
.evidence {{ display:inline-block; margin-top:9px; padding:5px 8px; background:#edf7f6; color:#08675f; font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace; }}
.table-wrap {{ overflow-x:auto; margin:18px 0; }}
.collapsible-list {{ margin:12px 0 22px; border-top:1px solid #e5e9f0; }}
.collapsible-list > summary {{ padding:10px 2px; cursor:pointer; color:#475467; font-size:13px; font-weight:650; }}
.collapsible-list[open] > summary {{ margin-bottom:4px; }}
.collapsible-list > .table-wrap {{ margin:0 0 12px; }}
.collapsible-list > ul,.collapsible-list > ol {{ margin-top:4px; margin-bottom:14px; }}
.head-table td,.head-table th {{ padding:5px 7px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; line-height:1.45; }}
th,td {{ padding:10px 9px; border-bottom:1px solid #e5e9f0; text-align:left; vertical-align:top; }}
th {{ background:#f8fafc; color:#475467; font-weight:650; white-space:nowrap; }}
td:not(:first-child) {{ font-variant-numeric:tabular-nums; }}
code,.math {{ font-family:"Iowan Old Style",Cambria,Georgia,serif; color:#203251; }}
.formula {{ margin:18px 0; padding:16px 18px; overflow-x:auto; background:#f8fafc; border:1px solid var(--line); font-family:"Iowan Old Style",Cambria,Georgia,serif; font-size:16px; }}
.example {{ display:block; margin-top:8px; color:#475467; font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; font-size:13px; }}
figure {{ margin:30px 0 38px; }}
figcaption {{ max-width:940px; margin:10px auto 0; color:#586579; font-size:13px; line-height:1.55; }}
.figure-title {{ margin:0 0 12px; font-size:17px; }}
.figure-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.figure-stack {{ display:grid; grid-template-columns:1fr; gap:18px; }}
.report-image {{ display:block; width:100%; height:auto; border:1px solid var(--line); background:#fff; }}
.three-d {{ border:1px solid var(--line); background:linear-gradient(#fbfcfe,#f5f8fb); }}
.three-d-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; border-bottom:1px solid var(--line); }}
.three-d-head label {{ color:#475467; font-size:13px; }}
.three-d-head select {{ margin-left:8px; padding:6px 8px; border:1px solid #b8c1cf; background:#fff; color:var(--ink); }}
.three-d canvas {{ display:block; width:100%; height:460px; cursor:grab; touch-action:none; }}
.three-d canvas:active {{ cursor:grabbing; }}
.chart-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:18px; }}
.line-chart,.bar-chart,.baseline-heatmap,.attention-map {{ display:block; width:100%; height:auto; border:1px solid var(--line); background:#fff; }}
.heat-cell {{ stroke:#fff; stroke-width:1.2; }}
.heat-title {{ fill:#172033; font-size:15px; font-weight:650; }}
.heat-row,.heat-x,.heat-y,.heat-legend {{ fill:#536074; font-size:11px; }}
.heat-value {{ fill:#172033; font-size:11px; font-weight:650; }}
.heat-value.inverse,.heat-rank {{ fill:#fff; }}
.attention-cell {{ stroke:#fff; stroke-width:.55; }}
.attention-cell.selected {{ stroke:#172033; stroke-width:1.5; }}
.heat-rank {{ font-size:7px; font-weight:650; pointer-events:none; }}
.plot-bg {{ fill:#fbfcfe; stroke:#cfd6e2; }}
.grid {{ stroke:#e7ebf0; stroke-width:1; }}
.grid.vertical {{ stroke:#eef1f4; }}
.zero {{ stroke:#768196; stroke-width:1.2; }}
.reference {{ stroke:#98a2b3; stroke-width:1.2; stroke-dasharray:4 4; }}
.reference-label,.tick,.legend-label,.bar-label,.bar-value {{ fill:#536074; font-size:12px; }}
.stable-band {{ fill:#dff4ee; opacity:.72; }}
.legend-band {{ stroke:#a8d9cc; stroke-width:1; opacity:1; }}
.axis-label {{ fill:#2f3b4d; font-size:13px; font-weight:600; }}
.series-line {{ fill:none; stroke-width:2.2; stroke-linejoin:round; stroke-linecap:round; }}
.series-dot {{ stroke:#fff; stroke-width:1; }}
.ci-whisker {{ stroke-width:1.25; opacity:.82; }}
.ci-dot {{ stroke-width:1.5; }}
.legend-line {{ stroke-width:2.4; }}
.bar {{ opacity:.88; }}
.experiment {{ display:grid; grid-template-columns:150px 1fr; gap:18px; margin:18px 0; padding:18px 0; border-top:1px solid #e7ebf0; }}
.experiment-label {{ color:var(--muted); font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace; text-transform:uppercase; letter-spacing:.08em; }}
.experiment h4 {{ margin:0 0 6px; }}
.result-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin:18px 0; }}
.result {{ padding:15px 16px; border:1px solid var(--line); }}
.result .value {{ display:block; color:var(--teal); font:650 23px ui-monospace,SFMono-Regular,Consolas,monospace; }}
.result .label {{ color:#596579; font-size:12px; }}
.path {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); align-items:stretch; gap:12px; margin:22px 0; }}
.node {{ position:relative; padding:14px 12px; border:1px solid var(--line); background:#f9fafb; font-size:13px; }}
.node::after {{ content:"→"; position:absolute; right:-18px; top:35%; color:#98a2b3; font-size:20px; }}
.node:last-child::after {{ display:none; }}
.node strong {{ display:block; margin-bottom:5px; }}
.node small {{ color:#667085; }}
.contrast-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:18px 0; }}
.contrast-lane {{ padding:16px; border:1px solid var(--line); border-top:4px solid var(--teal); background:#fbfcfe; }}
.contrast-lane.restored {{ border-top-color:var(--amber); }}
.contrast-source {{ margin-bottom:12px; font-weight:700; }}
.contrast-source small {{ display:block; margin-top:3px; color:#667085; font-weight:400; }}
.contrast-branches {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.contrast-arm {{ min-height:108px; padding:12px; border:1px solid #cfd6e2; background:#fff; font-size:13px; }}
.contrast-arm strong {{ display:block; margin-bottom:5px; }}
.contrast-arm small {{ color:#667085; }}
.contrast-result {{ margin-top:11px; padding-top:10px; border-top:1px solid var(--line); color:#344054; font-size:13px; }}
.contrast-result strong {{ color:#172033; }}
.evidence-map td:nth-child(1) {{ white-space:nowrap; font-weight:650; }}
.source-list {{ columns:2; column-gap:32px; font-size:12px; color:#596579; }}
.source-list li {{ break-inside:avoid; margin:0 0 8px; }}
.pill {{ display:inline-block; padding:2px 7px; border:1px solid #b8d9d5; color:#08675f; background:#f0fdfa; font-size:11px; white-space:nowrap; }}
.audit-summary {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:22px 0; }}
.audit-card {{ padding:15px 16px; border:1px solid var(--line); background:#fff; }}
.audit-card strong {{ display:block; font:700 26px ui-monospace,SFMono-Regular,Consolas,monospace; }}
.audit-card span {{ color:#667085; font-size:12px; }}
.status {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:700; white-space:nowrap; }}
.status-verified {{ color:#075e58; background:#dff7f2; border:1px solid #a8d9cc; }}
.status-falsified {{ color:#9f1f18; background:#fff0ee; border:1px solid #f5b7b1; }}
.status-partial {{ color:#8a4b08; background:#fff6df; border:1px solid #f0cf83; }}
.status-open {{ color:#475467; background:#f2f4f7; border:1px solid #d0d5dd; }}
.status-closed {{ color:#344054; background:#eaecf0; border:1px solid #b9c0cc; }}
.extension-audit td:nth-child(1) {{ width:38px; color:#667085; }}
.extension-audit td:nth-child(2) {{ min-width:190px; font-weight:650; }}
.extension-audit td:nth-child(4) {{ min-width:260px; }}
.extension-audit td:nth-child(5),.extension-audit td:nth-child(6) {{ min-width:230px; }}
@media(max-width:900px) {{ header,main {{ padding-left:28px; padding-right:28px; }} nav {{ padding-left:28px; }} .mechanism,.figure-grid,.chart-pair,.result-grid,.audit-summary,.contrast-grid {{ grid-template-columns:1fr; }} .stage::after {{ display:none; }} .path {{ grid-template-columns:1fr; }} .node::after {{ display:none; }} .source-list {{ columns:1; }} }}
@media(max-width:560px) {{ .page {{ width:100%; margin:0; }} header,main {{ padding-left:18px; padding-right:18px; }} nav {{ padding-left:18px; }} h1 {{ font-size:34px; }} h2 {{ font-size:27px; }} .experiment {{ grid-template-columns:1fr; gap:5px; }} .three-d canvas {{ height:380px; }} .contrast-branches {{ grid-template-columns:1fr; }} .contrast-arm {{ min-height:0; }} }}
@media print {{ body {{ background:#fff; }} .page {{ width:100%; margin:0; box-shadow:none; }} nav {{ display:none; }} section,figure {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<article class="page">
<header>
  <p class="eyebrow">Realistic NIAH · Non-thinking V4.4/V4.4.5 · Mechanistic analysis</p>
  <h1>Non-thinking 模型如何计数：分布式证据、广域检索与晚层写入</h1>
  <p class="dek">本报告按计算机制而非实验产生顺序组织证据：先陈述可证伪的三阶段机制，再依次检验 prompt-side evidence formation、answer-query broad retrieval、late consolidation 与 architecture-specific write。全文严格区分 <strong>representation decodability、causal use、sufficiency 与 mediation</strong>，避免把可读方向直接解释为模型实际使用的计数器。</p>
  <div class="meta"><span>模型：Qwen3-8B / Gemma4-E4B</span><span>计数范围：1–10</span><span>canonical seeds：1234–1263</span><span>位置：needle end / <code>Total:</code> 后首数字前</span><span>更新：2026-08-15</span></div>
</header>
<nav aria-label="report sections">
  <a href="#summary">机制总览</a><a href="#baseline">任务与行为</a><a href="#representation">测量与 geometry</a><a href="#formation">Stage I · form</a>
  <a href="#retrieval">Stage II · retrieve</a><a href="#write">Stage III · consolidate</a><a href="#ov-write">Stage IIIb · write</a><a href="#ledger">证据表</a><a href="#extension-audit">Extension 审计</a><a href="#limitations">边界与下一步</a><a href="#appendix">Appendix</a>
</nav>
<main>

<section id="baseline">
  <h2>1. 任务与行为基线：模型需要解释什么？</h2>
  <p class="lead">在进入 hidden-state geometry 前，先固定模型要解释的外部行为。每个模型、每个 gold needle count <span class="math">N∈{{0,…,10}}</span> 有 30 个样本；下图将 Qwen 与 Gemma 分开显示。</p>
  <div class="formula"><strong>单例误差与按 count 准确率。</strong>生成计数为 <span class="math">ŷ</span> 时，signed error 为 <span class="math">e_{{signed}}=ŷ−N</span>，absolute error 为 <span class="math">e_{{abs}}=|ŷ−N|</span>；<span class="math">Acc(N)=S_N^{{−1}}Σ_s 𝟙[ŷ_{{s,N}}=N]</span>。行为 MAD 是所有单例 absolute error 的平均：<span class="math">MAD_{{behavior}}=M^{{−1}}Σ_i e_{{abs,i}}</span>；wrong-only MAD 只在 <span class="math">e_{{abs}}>0</span> 的样本上取平均。单位均为 counts，而不是 median absolute deviation。<span class="example">例：gold <em>N</em>=8、输出 ŷ=6，则 signed error=−2、absolute error=2；若 30 个 N=8 样本中 7 个答对，则 Acc(8)=7/30=23.3%；若三例 absolute errors 为 [0,1,2]，behavior MAD=1，而 wrong-only MAD=(1+2)/2=1.5 counts。</span></div>
  <div class="figure-stack">
    <figure><h4 class="figure-title">图 1a · Qwen3-8B baseline</h4>{baseline_heatmaps['Qwen3-8B']}<figcaption>每一列是 gold needle count N；上排色块与数字是 exact accuracy（深绿=更高），下排是 mean absolute error（深红=更大，单位 counts）。每格聚合 30 个样本。全计数加权汇总：accuracy={pct(baseline_summary['Qwen3-8B']['accuracy'])}，MAD={f(baseline_summary['Qwen3-8B']['mad'])}，mean signed error={f(baseline_summary['Qwen3-8B']['signed'])}。</figcaption></figure>
    <figure><h4 class="figure-title">图 1b · Gemma4-E4B baseline</h4>{baseline_heatmaps['Gemma4-E4B']}<figcaption>坐标、颜色和每格样本数与左图相同。全计数加权汇总：accuracy={pct(baseline_summary['Gemma4-E4B']['accuracy'])}，MAD={f(baseline_summary['Gemma4-E4B']['mad'])}，mean signed error={f(baseline_summary['Gemma4-E4B']['signed'])}。两模型在较大 N 上主要低估，因此后文同时报告 accuracy、absolute error 与 count shift。</figcaption></figure>
  </div>
</section>

<section id="summary">
  <h2>Mechanism at a glance</h2>
  <div class="claim"><strong>中心主张。</strong>Non-thinking 模型没有把同一枚低维整数寄存器从 prompt endpoint 原样传到输出。更符合全部 matched-control 证据的解释是：<strong>浅层在 active needle spans 中形成分布式、可复用的证据；中层 answer query 通过 broad-attention bank 聚合这些证据；晚层将聚合结果转化为可执行的 count-aligned answer state</strong>。Qwen 的最终写入较局部化，Gemma 的写入更分布式。</div>
  <div class="mechanism" role="img" aria-label="三阶段非思考计数机制">
    <div class="stage"><span class="stage-no">STAGE I · FORM</span><h3>Distributed prompt evidence</h3><p>Needle-end states 很早出现 running-index ordering，Qwen 强于 Gemma；但前三维受到 absolute position 混淆。真正强的因果载体是完整 needle span：endpoint rank-3 removal 近零，而 early full-span restoration 可恢复约 2.8–2.9 counts。</p><span class="evidence">readable endpoint · causal span</span></div>
    <div class="stage"><span class="stage-no">STAGE II · RETRIEVE</span><h3>Broad, partial aggregation path</h3><p>Answer query 的 broad heads 对多个 active needle spans 分配 attention。Frozen top-bank ablation 超过 layer-matched random control；source patch 与 retrieval-subspace block 进一步证明其中一条 count-aligned 路径被自然 computation 使用，但不要求它是唯一通道。</p><span class="evidence">Qwen L23 +0.265 · Gemma L29 +0.527</span></div>
    <div class="stage"><span class="stage-no">STAGE III · CONSOLIDATE / WRITE</span><h3>Executable answer state</h3><p>中后层 answer state 变得可离散读取、可 donor-transfer、并对方向特异 removal 敏感。Qwen 可定位 L28 H16/H19 的局部 OV writer；Gemma 的 L29 heads 参与写入，但更强的闭环位于 L37 distributed residual mediator。</p><span class="evidence">answer patch Qwen 96.6% · Gemma 96.0%</span></div>
  </div>
  <div class="result-grid">
    <div class="result"><strong>Stage I 的决定性对照</strong><p>Active-token corruption 相对 ordinary-token corruption 增加 Qwen/Gemma 8.930/8.780 counts error；early full-span restoration 的 discovery plateau 为 2.832/2.880 counts，而 endpoint rank-3 specificity 仅在约 ±0.06 内。</p></div>
    <div class="result"><strong>Stage II 的决定性对照</strong><p>Top-bank ablation 的主要 matched-random effects 为 Qwen K32 +1.623、Gemma K8 +0.767 counts；restoration mediation 在 Qwen L23 为 +0.265，在 Gemma L29 为 +0.527，之后该 frozen basis 的效应消失。</p></div>
    <div class="result"><strong>Stage III 的决定性对照</strong><p>Full answer-state patch 的 donor-gold hit 为 96.6%/96.0%；late answer rank-3 removal 峰值为 +0.878/+1.222 counts；aligned 1× across-block transport 为 0.949/0.978，而等范数 orthogonal control 约为 0。</p></div>
  </div>
  <figure><h4 class="figure-title">图 1c · 同一 forward 把三阶段串起来</h4>{exp19_chart}<figcaption>每一横条是 canonical confirmation panel（10 seeds×10 counts）上、以 seed–count 为 paired unit 的平均 expected-count effect，单位均为 counts；横轴越向右表示候选操作相对其 matched control 的方向特异效应越大。Source repair 比较完整 needle-span restoration 与等 token ordinary-span restoration；retrieval/late mediation 分别比较 count-aligned removal 与同层、同 realized-norm orthogonal removal。三组效应在 Qwen 与 Gemma 均为正，95% seed-unit bootstrap CI 见 Appendix B。该图把此前分段证据升级为同一试次的有序 partial serial mediation，但条长不能相加成“通路解释比例”。</figcaption></figure>
  <div class="claim"><strong>链条已在同一试次闭环。</strong>Source repair 为 Qwen/Gemma +2.674/+2.670 counts；retrieval mediation 为 +0.327/+0.521；late mediation 为 +1.118/+1.215。Source restoration 同时改变后续 broad score 与 retrieval coordinate；retrieval-aligned removal 削弱随后 late coordinate；late-aligned removal 改变输出而不反向改变已经计算完的 retrieval readout。<strong>目前结论：</strong>支持“distributed span evidence → local retrieval mediator → late answer mediator → output”的有序部分中介。</div>
  <div class="claim boundary"><strong>论文级结论边界。</strong>我们建立的是一条<strong>中尺度、部分中介的自然计算路径</strong>，而不是完整且唯一的 circuit。Joint retrieval×late interaction 在两模型均为负，且 fully aligned block 后仍保留正 source repair，说明两个冻结 mediator 有重叠、饱和与 bypass，而不是穷尽全部计算。Prompt endpoint geometry 是 descriptive readout；span-level evidence 是被因果支持的 source unit；broad retrieval 是被自然使用的一条 aggregation pathway；late answer state 是充分且方向特异必要的执行状态。现有证据不要求解析 span 内哪个 token 是“地址”或“内容”，也不要求 Qwen 与 Gemma 共享同一组 writer heads。</div>
</section>

<section id="representation">
  <h2>2. 测量框架：两个位置、三种证据强度</h2>
  <p class="lead"><strong>Prompt site</strong> 定义为每个 active needle 的最后一个 token，状态记为 <span class="math">h<sup>P</sup><sub>s,n,ℓ</sub></span>；其中 <span class="math">n∈{{1,…,10}}</span> 是当前 running index。<strong>Answer site</strong> 定义为生成首个数字前 <code>Total:</code> 的 query token，状态记为 <span class="math">h<sup>A</sup><sub>s,ℓ</sub></span>，标签是 prompt 的最终 gold count。本节只建立 representation facts 与统一 readouts；真正的机制判断由后续 matched interventions 给出。</p>
  <div class="formula"><strong>证据层级。</strong><em>Decodability</em> 问某个 held-out readout 能否从状态预测 count；<em>sufficiency</em> 问把 donor state 写入 receiver 后能否驱动 donor-directed output；<em>direction-specific necessity</em> 问删除候选方向是否比删除同层、同位置、同 realized norm 的正交方向更伤行为；<em>mediation</em> 问阻断候选中介是否会特异地削弱一个已产生的上游效应。<span class="example">例：分类准确率高只说明 count 可读；只有 donor patch 能转移答案、或候选方向 removal 比等范数 orthogonal removal 更伤，才说明该状态具备可执行性或方向特异必要性。</span></div>

  <h3>2.1 Frozen-PCA 三维流形</h3>
  <div class="figure-grid">
    <figure>
      <h4 class="figure-title">图 2a · Prompt needle-end running-index manifold</h4>
      <div class="three-d"><div class="three-d-head"><strong>Needle-end state</strong><div><label>模型<select id="prompt-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label><label>层<select id="prompt-layer" aria-label="Prompt manifold layer"></select></label></div></div><canvas id="prompt-canvas" aria-label="Prompt needle-end hidden states projected onto frozen PC1 PC2 PC3. Drag to rotate.">Your browser needs Canvas support.</canvas></div>
      <figcaption>每个半透明点是一条 seed 在第 <em>n</em> 个 needle end 的 hidden state；颜色与数字表示 running index <em>n</em>。较大的编号圆点是 30 seeds 的 count centroid，连线只连接相邻 <em>n</em>。PC1/PC2/PC3 在 discovery rows 上按所选层独立冻结，数值无物理单位；因此可比较同层 geometry，不能把跨层坐标值当作同一基底。默认 Qwen L8、Gemma L9 是冻结的早期代表层，用于展示 running geometry 已经形成，而不是按最高 R² 重新选出的层；可切换任意层，拖动旋转，双击复位。</figcaption>
    </figure>
    <figure>
      <h4 class="figure-title">图 2b · Answer-query consolidated-count manifold</h4>
      <div class="three-d"><div class="three-d-head"><strong>Answer-query state</strong><div><label>模型<select id="answer-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label><label>层<select id="answer-layer" aria-label="Answer manifold layer"></select></label></div></div><canvas id="answer-canvas" aria-label="Answer query hidden states projected onto frozen PC1 PC2 PC3. Drag to rotate.">Your browser needs Canvas support.</canvas></div>
      <figcaption>每个点是一条 seed 的 answer-query state；颜色与编号表示最终 gold count。PC1/PC2/PC3 同样按所选层冻结；图展示 3D 投影 geometry，而 exact-count classifier 使用 PCA-32。默认 Qwen L28、Gemma L37 是在 V4.4.5 reused-state analysis 中按 confirmation 三维 nearest-centroid accuracy 选出的 display-only 层；这种 post-hoc 选择只服务可视化，不进入因果或跨层推断。层选择器可检查 formation、aggregation 与 consolidation 的完整跨层变化。</figcaption>
    </figure>
  </div>

  <div class="claim"><strong>为什么默认展示浅层 prompt 与深层 answer？</strong><p>两张图对应不同计算阶段，而不是 layer-matched representation comparison。Prompt needle end 在阅读中刚完成局部 phrase，prefix-local running index 因而可在浅层形成；answer query 位于全文之后，必须经过远距离 retrieval、aggregation 与 residual write，行为上可执行的 final-count state 才在中后层出现。</p><p>数据支持这种时序，但不支持“prompt representation 只在浅层、answer representation 只在深层”的排他说法。Prompt ridge 的实际最高点是 Qwen L13（R²=0.960，MAD=0.428）和 Gemma L11（R²=0.857，MAD=0.859）；answer classifier 也会逐层上升。交互图的 answer 默认层改为 reused-state 3D 最易分辨的 Qwen L28 / Gemma L37，而 causal timing 仍由独立的 dense full-state patching、rank-3 removal 与 retrieval-subspace experiments 判断。</p><p><strong>解释边界：</strong>直接比较 L8/L9 与 L28/L37 同时改变 token site 与 layer depth，只能示意“formation → retrieval → consolidation”，不能单独证明差异来自位置而非深度；层选择器与下方 layerwise curves 用于审计这一点。<strong>目前结论</strong>是 shallow/deep timing 与机制顺序一致，但其架构原因仍是解释而非专门干预的结果。</p></div>

  <h3>2.2 Counter-manifold 的紧致程度：classifier 只是统一的描述性标尺</h3>
  <p class="lead"><strong>本小节只有一个目的：</strong>用完全相同的 seed-held-out protocol，量化不同 count 对应的 hidden-state clouds 是否紧致、是否容易彼此区分，从而给后续 non-thinking 与 native thinking 提供可直接比较的 representation baseline。Classifier 不参与 mechanism discovery，不用于证明 causal use，也不意味着模型内部存在一个离散整数寄存器。</p>
  <div class="formula"><strong>Prompt continuous probe。</strong>每个 seed fold 只用训练 rows 拟合 PCA-32 与 ridge regression，再在 held-out seeds 上预测 running index <span class="math">n</span>。<span class="math">R²=1−Σ_i(n_i−n̂_i)²/Σ_i(n_i−n̄)²</span>；ridge MAD=<span class="math">M⁻¹Σ_i|n̂_i−n_i|</span> counts。Prompt needle ends 不做 exact-count classifier：每个 prompt 内是连续 occurrence positions，而不是十个独立的最终计数样本；这里要检验的是 running-index curve 的连续可解码性。<span class="example">例：held-out predictions 解释了 gold running index 方差的 90%，则 R²=0.90；若三例 |预测−gold| 为 [0.2,0.5,1.1]，ridge MAD=0.6 count。</span></div>
  <div class="formula"><strong>Answer exact-count classifier：如何读。</strong>按 seed 做五折 GroupKFold；每折只在训练 seeds 上拟合 StandardScaler 与 PCA-32，再在 held-out seeds 上预测 count。<strong>Nearest centroid</strong> 直接问测试 state 是否靠近正确 count 的训练 centroid，因此是本文最接近“manifold 紧致度”的 operational readout；<strong>L2 logistic</strong> 只作为 learned linear-boundary robustness check。Accuracy 越高表示 count clouds 越容易区分；classifier MAD=<span class="math">M⁻¹Σ_i|n̂_i−n_i|</span> 越低表示即使分错也更接近正确 count。严格说 classifier 同时受 within-count dispersion 与 between-count separation 影响，所以这里称为<strong>紧致度/可分性诊断</strong>，而不是纯粹的几何半径。<span class="example">例：四个 gold counts [2,4,7,9] 被预测为 [2,5,7,6]，accuracy=2/4=50%，MAD=(0+1+0+3)/4=1 count；若 native thinking 在完全相同 split 下达到 80%/0.3，就说明它的 count-conditioned manifold 更紧致且更可分。</span></div>
  <details class="collapsible-list"><summary>展开：辅助 manifold quantities 的定义</summary>
    <div class="formula"><strong>Manifold quantities。</strong>令中心化 state matrix 为 <span class="math">X</span>。stable rank=<span class="math">‖X‖²_F/‖X‖²_2</span>；all-state rank-3 capture=<span class="math">Σ_{{k=1}}^3σ_k²/Σ_kσ_k²</span>，centroid rank-3 capture 对十个 count centroids 做同一计算；<span class="math">η²_{{count}}=Σ_nN_n‖μ_n−μ‖²/Σ_i‖h_i−μ‖²</span>；silhouette=<span class="math">(b-a)/max(a,b)</span>，其中 <em>a</em> 是同 count 平均 cosine distance，<em>b</em> 是最近异 count cluster distance。<span class="example">例：奇异值 [2,1] 时 stable rank=(4+1)/4=1.25；rank-3 capture=0.80 表示前三轴承载 80% 方差；η²=0.60 表示 60% 总方差来自 count centroids 之间；若 a=1、b=2，则 silhouette=0.5。</span></div>
    <div class="formula"><strong>为什么还保留 PCA quantities。</strong>Classifier 回答“count clouds 是否容易区分”；centroid rank-3 capture 回答“十个平均 state 是否沿低维轨迹排列”；all-state rank-3 capture 与 silhouette 则补充单样本 cloud 是否真的紧致。高 centroid rank-3 不保证高 classifier accuracy，也不保证形成干净 clusters。</div>
  </details>

  <h4>Prompt running-index geometry（manifold quantities；不含 exact classifier）</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Site / layer</th><th>Ridge R²</th><th>Ridge MAD</th><th>PCA rank-3: all states</th><th>PCA rank-3: centroids</th><th>Stable rank</th><th>η² count</th><th>Cosine silhouette</th></tr></thead><tbody>{prompt_geometry_table_rows}</tbody></table></div>
  <h4>Answer-query final-count geometry（exact classifiers + manifold quantities）</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Site / layer</th><th>L2-logistic acc.</th><th>L2-logistic MAD</th><th>Nearest-centroid acc.</th><th>Nearest-centroid MAD</th><th>Ridge R²</th><th>Ridge MAD</th><th>PCA rank-3: all states</th><th>PCA rank-3: centroids</th><th>Stable rank</th><th>η² count</th><th>Cosine silhouette</th></tr></thead><tbody>{answer_geometry_table_rows}</tbody></table></div>
  <h4>V4.4.5 reused-state display-only 三维层</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Display layer</th><th>Confirmation 3-PC nearest-centroid accuracy</th><th>Integer MAD</th><th>Discovery rank-3 all-state capture</th></tr></thead><tbody>{answer_reuse_rows}</tbody></table></div>
  <p>这张补充表复用 canonical dense run 的每模型 300 个 clean natural forwards（共 23,400 个 layer-state rows）；basis 只在 discovery seeds 1234–1253 拟合，数值在 confirmation seeds 1254–1263 计算。层是按 confirmation 三维 nearest-centroid accuracy 最大化后选出的，所以 61%/63% <strong>只用于选择最好看的三维展示层</strong>。后续与 native thinking 做正式比较时，应预先冻结相同的 layer-selection 与 held-out-seed protocol，而不使用这两个 post-hoc display 数值。</p>
  <div class="claim"><strong>本小节的唯一结论。</strong>Non-thinking answer-query manifold 的 exact-count accuracy 约为 53%–56%（chance 10%），nearest-centroid MAD 为 0.615–0.640 count：final count 明显可读，但 count-conditioned clouds 还不是非常紧致。Centroid rank-3 capture 很高而 silhouette 接近 0，进一步说明“平均 count 轨迹低维”不等于“每个 count 形成干净、分离的 cluster”。这些数值只建立 non-thinking 的 representation baseline，留给后续用同一 classifier protocol 与 native thinking 比较；它们不参与 retrieval、patching 或 OV-write 的机制结论。</div>

  <figure>
    <h4 class="figure-title">图 3 · Counter manifold 的可读性与低维结构如何跨层变化（descriptive baseline）</h4>
    {''.join(representation_charts)}
    <figcaption>每个模型有两排 panel，横轴均为 zero-based transformer layer。第一排是统一的 held-out compactness/separability diagnostic：左图为 prompt running-index ridge R²；右图为 answer-query L2-logistic accuracy（蓝色实线）与 nearest-centroid accuracy（绿色虚线），水平虚线为十分类 chance=0.10。第二排是辅助的无监督 geometry：实线为全部单样本 cloud 的 PCA rank-3 capture，虚线为十个 count centroids 的 rank-3 capture。未来与 native thinking 比较时应复用完全相同的 split、PCA、readout 与 layer-selection rule；这些曲线不用于确定 causal mechanism。当前 non-thinking 的代表层：{'；'.join(best_rows)}。</figcaption>
  </figure>

  <h3>2.3 Running-index manifold 是否具有 counter 性质？</h3>
  <p class="lead">仅有高 ridge <span class="math">R²</span> 只能说明 running index 可读，不能说明十个状态构成稳定的逐步累加器。这里在原报告已经冻结的 PC1–PC3 中，追加检验 centroid 的有序距离、相邻更新方向、跨 discovery/confirmation 的稳定性，并单独做绝对位置对照。该分析不重新拟合 PCA，也不是因果干预。</p>
  <div class="formula"><strong>Ordered 与 additive counter quantities。</strong>令 discovery seeds 上第 <span class="math">n</span> 个 endpoint 的三维 centroid 为 <span class="math">μ_n</span>。<strong>Line R²</strong> 将 <span class="math">μ_n</span> 拟合为 <span class="math">a+n,v</span> 后计算 explained variance；<strong>distance-gap ρ</strong> 是 45 对 centroids 的欧氏距离 <span class="math">‖μ_i−μ_j‖</span> 与 count gap <span class="math">|i−j|</span> 的 Spearman correlation。令相邻 step <span class="math">Δ_n=μ_{{n+1}}−μ_n</span>，报告所有 step-pair cosine 的平均、step length 的 coefficient of variation <span class="math">SD(‖Δ_n‖)/mean(‖Δ_n‖)</span>，以及 discovery step 与同一 confirmation step 的 cosine。理想等步长直线 counter 会同时具有 line R²=1、distance-gap ρ=1、step cosine=1、length CV=0。<span class="example">例：centroids [0,1,2,3] 在一维中等距排列，则相邻 steps 都为 +1，四个指标分别为 1、1、1、0；若 steps 变为 [+3,−1,+3]，count 仍可能被某个 regression 读取，但 step cosine 与 distance ordering 会明显下降。</span></div>
  <div class="formula"><strong>Position control。</strong>all-token confirmation capture 中，每个 endpoint 有绝对 token position <span class="math">p</span>。我们以 seed 为 group 做 5-fold GroupKFold：每折只在训练 seeds 上拟合 cubic position baseline <span class="math">n∼1+p+p²+p³</span>；同时把 PC1–PC3 对同一 cubic design 的训练折拟合值减掉，再用 residual PC1–PC3 ridge-predict held-out running index。表中的 residual <span class="math">R²</span> 因而问“冻结前三维中还有多少不能由平滑绝对位置解释的 count signal”。<span class="example">例：若原始三维 probe R²=0.80，而每个 PC 都几乎是 position 的平滑函数，residual probe 可降到 0；这不会证明完整 hidden state 没有 counter，只说明当前前三维没有 position-independent evidence。</span></div>
  <figure><h4 class="figure-title">图 3b · Frozen PC1–PC3 的 counter-property 跨层审计</h4><img class="report-image" src="{counter_plot_uri}" alt="Prompt needle-end counter properties across all layers for Qwen and Gemma"><figcaption>四个 panel 的横轴均为 zero-based transformer layer；绿色为 Qwen、紫色为 Gemma，大圆点标出报告代表层 Qwen L8 / Gemma L9。左上纵轴是十个 discovery centroids 的直线拟合 R²；右上是 centroid distance 与 count gap 的 Spearman ρ；左下是九个相邻 centroid steps 两两 cosine 的平均；右下是同一 step 在 discovery 与 confirmation centroids 之间的 cosine 平均。四者越接近 1，越接近有序、同向、跨 seed split 稳定的增量轨迹；它们不衡量行为因果效应。</figcaption></figure>
  <h4>代表层 counter-property 与 position robustness</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>Line R²</th><th>Distance-gap ρ</th><th>Mean step cosine</th><th>Step-length CV</th><th>Split step cosine</th><th>Frozen 3-PC ridge R² / MAD</th><th>Per-seed projection ρ / positive steps</th><th>Position-count ρ</th><th>Position-residual 3-PC R² / MAD</th></tr></thead><tbody>{counter_table_rows}</tbody></table></div>
  <p>Qwen L8 的 line R²=0.633、distance-gap ρ=0.648、mean step cosine=0.761、cross-split step cosine=0.817，说明冻结三维中确有中等偏强的有序、同向 trajectory；Gemma L9 对应为 0.403、0.488、0.084、0.412，明显更弯曲且 step directions 不稳定。两模型 step-length CV 分别为 1.438 与 1.729，也远离理想等步长 counter。更重要的是 position-count ρ 均约 0.965，cubic position baseline 的 grouped R² 约 0.91；去除 cubic position 后三维 R² 只剩 Qwen 0.043、Gemma 0.004。<strong>目前结论：</strong>Qwen 的 endpoint 3D 轨迹具有 counter-like ordering，Gemma 较弱；但两者都不能与 ordinal/absolute-position code 区分，因此不能称为已识别的抽象 counter。</p>
  <div class="formula"><strong>Endpoint-gated formula 的 incremental R²。</strong>对每类 token 先以 category mean 为 baseline，再把在 endpoint 学到的 frozen curve <span class="math">γ(n)</span> 按不同 gate 应用于 endpoint、needle interior 或所有 prefix-count tokens。定义 <span class="math">ΔR²_{{inc}}=(SSE_{{baseline}}−SSE_{{curve}})/SSE_{{baseline}}</span>；正值表示 curve 比 category mean 多解释方差，负值表示强行套用后反而更差。<span class="example">例：baseline SSE=100、curve SSE=40，则 ΔR²=0.60；若 curve SSE=110，则 ΔR²=−0.10。</span></div>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>Endpoint-gated ΔR² on endpoints</th><th>Span-gated ΔR² on interiors</th><th>Ungated ΔR² on ordinary tokens</th><th>Ungated ΔR² on hard negatives</th></tr></thead><tbody>{gated_table_rows_html}</tbody></table></div>
  <div class="claim boundary"><strong>Counter-property 结论。</strong>在 representational level，endpoint centroids 确实按 running index 排列，尤其 Qwen 在浅层具有同向、跨 split 可复现的三维 step；但它不是等步长，Gemma 的 step alignment 很弱，而且 position residual 几乎不再可解码。all-token 公式进一步支持“curve 只在 needle endpoint gate 上出现”，证伪“needle span 内每个 token 共享同一 γ(n)”的强版本。full-hidden-space 的位置去混淆尚未完成；跨 final-N prefix invariance 未运行且已按当前论文的主张范围关闭。因此报告只写 <em>position-confounded counter-like record</em>，不升级为独立、跨任务不变的 abstract counter。</div>

  <h3>2.4 原始行为是 representation 的外部标尺</h3>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Rows</th><th>Exact accuracy</th><th>Behavior MAD</th><th>Signed error</th><th>Wrong-only MAD</th></tr></thead><tbody>{behavior_rows}</tbody></table></div>
  <p>本表对应 Frozen-PCA payload 使用的 20 seeds × 10 counts（N=1,…,10）geometry cohort；开头热图对应 30 seeds × 11 counts（N=0,…,10）的完整 baseline grid。两者用途与分母不同，不混合汇总。geometry cohort 上 Qwen 与 Gemma 都主要低估：signed error 分别为 −0.815 与 −1.320。因而后续干预既看 exact accuracy，也看 absolute count shift / error；只看命中率会丢掉“向正确 count 移动了多少”的信息。<strong>目前结论：</strong>representation 图回答“信息是否存在以及以何种 geometry 存在”，不能替代后续因果实验；Qwen 较强、Gemma 较弱的 prompt ordering 与两者行为差异一致，但尚未被证明是 performance gap 的因果来源。</p>
</section>

<section id="formation">
  <h2>3. Stage I — Prompt-side formation：可读 endpoint，因果 span</h2>
  <p class="lead">这一阶段区分两个经常被混淆的问题：needle evidence 是否进入网络，以及 endpoint probe 解出的三维曲线是否就是模型自然使用的存储格式。前者有强因果证据；后者仅是描述性 readout。Qwen 的 running-index ordering 明显强于 Gemma，这与 Qwen 更好的计数行为一致，但不是单独的因果解释。</p>
  <div class="claim"><strong>Stage-I hypothesis。</strong>浅层首先形成与 needle occurrences 相关的分布式 span-level evidence；needle endpoint 可以作为观察该过程的 readout site，但不是已经确认的独立 counter register。本文把“完整 active needle span”作为上游 causal source unit，不进一步要求区分 endpoint 是检索地址、内容载体还是与 interior 协同。</div>

  <h3>3.1 Interventions and paired controls</h3>
  <div class="table-wrap"><table><thead><tr><th>Experiment</th><th>Candidate intervention / measurement</th><th>Control 的具体构造</th><th>保持匹配的量</th><th>唯一希望改变的量</th></tr></thead><tbody>
    <tr><td>1A · earlier-span attention</td><td>在 clean forward 中，从第 <em>n</em> 个 needle endpoint query 计算指向前 <em>n</em>−1 个完整 active-needle spans 的 attention mass。</td><td>对每个 needle span 构造一个同 token 长度的 ordinary non-needle segment：从该 span 之前 8–511 tokens 中选择最近可用区间；排除全部 slot/hard-negative spans，且各 control segments 互不重叠。只比较前 <em>n</em>−1 对 spans。Heads 由 discovery seeds 的 earlier-needle mass 冻结排序；control 与 confirmation 估计不重新选 head。</td><td>同一模型、seed、prompt、query、head、layer、span 数量、每个 span 的 token 长度与近似相对深度。</td><td>被 query 指向的是 active needle evidence，还是附近的 ordinary passage tokens。它不是 random-head control。</td></tr>
    <tr><td>1B · token corruption</td><td>把每条完整 active-needle span 从第一个到最后一个 token，逐 span 替换成同一 prompt 中抽取的等长 ordinary token sequence；随后从头 greedy generation。</td><td>第三个 paired forward 在 ordinary passage 中分配与各 needle span 一一等长的 target segments，再用另一组、不重叠的 ordinary sequences 替换。所有 source/target segments 均避开 slot 与 hard-negative spans，并限制在 slot 区域前后 64 tokens 的 passage window 内。另有完全不改输入的 clean forward。</td><td>同一 seed/count prompt、总替换 token budget、每段长度、总序列长度、后续绝对位置、answer query 位置与 generation/parser。匹配的是预算，不强求替换前后碰巧相同的 token IDs 数也完全相等。</td><td>机械上同规模的文本替换发生在 active-needle evidence，还是 ordinary passage。</td></tr>
    <tr><td>1C · prompt rank-3 removal</td><td>在每个测试层，同时从同一 prompt 的全部 active needle-end states 删除 discovery-fitted count-centroid rank-3 basis 上的 within-prompt centered projection。</td><td>先从 discovery rows 减去各自 count centroid，再删除 count-basis 分量；对剩余 within-count residuals 做 PCA，取前三轴并再次正交化到 count basis。测试时从这个数据驱动的 orthogonal rank-3 basis 删除分量，并按每个 prompt 缩放到与 candidate 实际删除量相同的 Frobenius norm。另有 clean forward 作为共同基线。</td><td>同一 prompt、layer、全部 endpoint positions、rank=3、hook 时点、实际删除 Frobenius norm、后续 forward 与 greedy generation。</td><td>删除方向是否与 count-centroid geometry 对齐。Control 不是随机方向，而是高方差的 within-count nuisance subspace。</td></tr>
  </tbody></table></div>
  <p class="lead">样本范围：1A 使用 confirmation seeds 1254–1263、N=10 prompts、occurrences 2/4/6/8/10；1B 使用同十个 seeds、counts 1–10（每模型 100 个 paired prompts）；1C 使用同十个 seeds、counts 2–10，并按报告列出的 Qwen 10 层与 Gemma 13 层逐层配对。1A 在同一 seed/occurrence/head 内做 needle-minus-ordinary 后再平均 occurrences；1B/1C 在同一 seed/count prompt 内做 candidate-minus-control 后再跨 counts 与 seeds 汇总，避免把不同 prompts 的自然难度当作干预效应。</p>

  <div class="formula"><strong>Earlier-span preference。</strong>在当前 needle endpoint query 上，将全部较早 active-needle spans 的 attention mass 记为 <span class="math">A_{{needle}}</span>，将长度和相对位置匹配的 ordinary spans mass 记为 <span class="math">A_{{control}}</span>；preference=<span class="math">A_{{needle}}−A_{{control}}</span>，再先在 seed 内平均 occurrence rows、后跨 seeds 平均。<span class="example">例：某 head 对 earlier needles 的总 mass=0.80，对 matched ordinary spans 的 mass=0.10，则 preference=0.70。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 1A</div><div><h4>Earlier-span attention：endpoint 在形成时会回看此前 evidence</h4><p>在当前 needle endpoint 作为 query，计算对所有 earlier active-needle spans 的 attention mass，再减去长度与位置匹配的 ordinary spans。confirmation seeds 中最强 Qwen head 为 L{earlier_qwen['layer']}H{earlier_qwen['head']}，preference={f(earlier_qwen['confirmation_preference_mean'])}（区间 {ci(earlier_qwen,'ci95_low','ci95_high')}）；Gemma 为 L{earlier_gemma['layer']}H{earlier_gemma['head']}，preference={f(earlier_gemma['confirmation_preference_mean'])}（{ci(earlier_gemma,'ci95_low','ci95_high')}）。这支持“读到此前 needles 后更新当前 state”，但不证明该单头保存完整整数。</p></div></div>
  <div class="formula"><strong>Absolute-error increase 与 specificity。</strong>对 clean、needle-corrupt、matched-control 输出分别记为 <span class="math">ŷ_0,ŷ_N,ŷ_C</span>，条件 <span class="math">c</span> 的 error increase 为 <span class="math">Δe_{{abs}}(c)=|ŷ_c−N|−|ŷ_0−N|</span>；token specificity=<span class="math">Δe_{{abs}}(needle)−Δe_{{abs}}(control)=|ŷ_N−N|−|ŷ_C−N|</span>。Accuracy damage 是 <span class="math">d_{{acc}}(c)=𝟙[ŷ_0=N]−𝟙[ŷ_c=N]</span>，accuracy-damage specificity 同样为 needle damage 减 control damage。<span class="example">例：gold N=8，clean 输出 8、needle corruption 输出 1、control 输出 7；两种 absolute-error damage 分别为 7 与 1，所以 specificity=6 counts。此例 needle/control 都从 correct 变 wrong，accuracy damages 都为 1，故 accuracy specificity=0；若 control 仍输出 8，则为 1−0=1。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 1B</div><div><h4>Token corruption：active needle 本身是强因果输入</h4><p>Candidate 与 ordinary control 都使用同一 prompt 内的 ordinary token sequences 做等长度替换；candidate targets 是全部 active-needle spans，control targets 是 slot 区域附近、与每条 needle 一一等长且互不重叠的 ordinary passage spans。两者保持总 token budget、序列长度与 query position 不变。needle-minus-control 的 absolute-error specificity：Qwen +8.930 counts（8.700–9.180），Gemma +8.780（8.590–8.950）；accuracy damage specificity 分别 +0.450 与 +0.360。ordinary control 自身的 absolute-error increase 接近零（Qwen −0.010；Gemma +0.040），所以大效应不是任意等规模文本替换的普遍后果。</p></div></div>
  <div class="formula"><strong>Prompt rank-3 removal。</strong>对同一 prompt 的所有 active needle-end states <span class="math">H</span>，删除 discovery-fitted count basis <span class="math">U_3</span> 上的 centered component：<span class="math">H′=H−(H−H̄)U_3U_3^⊤</span>。Orthogonal control basis 不是任取随机方向：它由 discovery rows 的 within-count residuals 拟合，先减各 count centroid、再移除 <span class="math">U_3</span>、取 residual PCA 前三轴并正交化到 <span class="math">U_3</span>。在每个测试 prompt 上，将 control projection 缩放到与 candidate 实际移除量相同的 Frobenius norm。报告的 absolute-error specificity 为 <span class="math">[|ŷ_{{rank3}}−N|−|ŷ_0−N|]−[|ŷ_{{orth}}−N|−|ŷ_0−N|]=|ŷ_{{rank3}}−N|−|ŷ_{{orth}}−N|</span>；正值才表示 count-aligned removal 比同位置、同 rank、等删除量的 nuisance-direction removal 更伤。<span class="example">例：gold N=8，clean 输出 8，rank-3 removal 输出 6（error increase 2），orthogonal removal 输出 7（increase 1），则 specificity=(2−0)−(1−0)=1 count。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 1C</div><div><h4>Prompt endpoint rank-3 removal：可解码曲线没有显示局部必要性</h4><p>除主展示层 Qwen L8 / Gemma L9 外，我们也逐层测试：Qwen L0/4/8/12/16/20/24/28/32/35，Gemma L0/4/8/9/12/16/20/24/28/32/36/40/41；各层均在所有 active needle ends 删除该层冻结的 rank-3 count component，并与 actual-norm-matched orthogonal removal 配对。逐层 absolute-error specificity 均很小（Qwen 范围 −0.022 至 +0.056；Gemma −0.011 至 +0.022）。独立的 endpoint 实验为 Qwen +0.056（−0.011–0.133）与 Gemma −0.022（−0.089–0.033）。当前干预下看不到足以解释行为的 localized register effect。</p></div></div>

  <figure><h4 class="figure-title">图 4 · “输入 evidence 必要”与“decoded endpoint subspace 必要”不是同一命题</h4>{formation_chart}<figcaption>横轴均为 candidate 相对其 paired control 多增加的 absolute count error，单位是 counts。长条的 candidate 是全 active-needle token replacement，control 是同一 prompt 内、相同 span-length vector 与总 token budget 的 ordinary-passage replacement；短条的 candidate 是全部 needle endpoints 上的 count rank-3 removal，control 是同层、同位置、同 rank、每个 prompt 实际删除 Frobenius norm 相同的 orthogonal within-count-residual removal。两者相差约两个数量级。该图不能推出 prompt states 完全无因果作用；它只排除了“当前线性 rank-3 endpoint subspace 是一个强、局部、必要 counter”这一较窄主张。</figcaption></figure>
  <div class="claim boundary"><strong>证据边界。</strong>我们将早层表述为 <em>noisy counter-like record</em>，而不是“无 causal effect 的 counter”。更准确的结论是：其 geometry 可解码，active evidence 强因果，但当前 endpoint rank-3 ablation 未检测到相称的局部必要效应。可能原因包括信息分散在整个 span/多个 token、非线性编码、跨位置冗余，或 broad heads 后续重新从原始 evidence 聚合。</div>

  <h3>3.2 Dense span restoration：逐层追踪“prompt evidence 还来得及被使用吗？”</h3>
  <p class="lead">可以把这个实验理解为<strong>“先把 needles 擦掉，再在网络内部把 clean 记忆还回去”</strong>。我们先运行 clean prompt，保存每一层 needle positions 的 hidden states；再把输入中的全部 active needles 替换成等长 ordinary text，使模型失去计数证据。随后只在一个指定层 <span class="math">ℓ</span>，把 clean hidden states 写回 corrupt run，并让后面的层照常运行。如果答案因此变好，就说明在 L<span class="math">ℓ</span> 时，把这些上游 states 还给模型仍然来得及影响最终计数。</p>
  <div class="path" aria-label="Dense span restoration workflow"><div class="node"><strong>Clean run</strong><small>保存每层 needle states</small></div><div class="node"><strong>Corrupt run</strong><small>用等长 ordinary text 擦掉 needles</small></div><div class="node"><strong>只在 Lℓ 恢复一次</strong><small>endpoint、whole span 或 ordinary control</small></div><div class="node"><strong>继续正常 forward</strong><small>后续不再 clamp 或重复 patch</small></div><div class="node"><strong>比较答案误差</strong><small>看恢复是否真的救回计数</small></div></div>
  <div class="formula"><strong>一条 layerwise 曲线在问什么。</strong>横轴是“在哪一层把 clean states 还回去”，纵轴是相对 ordinary control 多修复了多少 expected-count error。早层为正，表示 retrieval 尚未结束，恢复后的 prompt evidence 还能被后续 answer query 读取；晚层趋近 0，表示此时再修复 prompt positions 已经太晚，相关信息要么已经被取走并写到 answer state，要么后续层已不再读取这些位置。<strong>因此下降位置定位的是 reusable-source window 的结束，而不是声称该层把信息物理删除了。</strong></div>
  <details class="collapsible-list"><summary>展开：四组 forward 与 matched control 的完整设置</summary><div class="table-wrap"><table><thead><tr><th>Condition</th><th>具体设置</th><th>Matched control / 保持不变</th><th>目的</th></tr></thead><tbody>
    <tr><td>clean / needle-corrupt</td><td>clean prompt；或将所有 active needle spans 用同 prompt 的等长 ordinary source tokens 替换。</td><td>序列长度、每段 token 数、answer-query 绝对位置、generation/parser 固定。</td><td>建立未破坏与破坏后的行为端点。</td></tr>
    <tr><td>endpoint restoration</td><td>在 corrupt run 的单个 post-block layer，只把每个 active needle 的最后一个 token state 换成 clean run 同层、同位置 state。</td><td>与 full-span restoration 使用同一 corrupt prompt、layer 与后续 forward；只改变被恢复的位置集合。</td><td>检验 endpoint 是否局部充分。</td></tr>
    <tr><td>full-span restoration</td><td>同一层把每个 active needle span 的全部 token states 从 clean run 写回 corrupt run，然后继续后续 layers 与严格生成。</td><td>single-layer intervention，不在之后各层重复 clamp。</td><td>检验分布式 span state 是否可复用。</td></tr>
    <tr><td>ordinary-corrupt / ordinary restoration</td><td>在 ordinary passage 中分配两组互不重叠、与 needle span-length vector 完全相同的 target/source banks；破坏后恢复 clean ordinary target states。</td><td>同一 seed/count、总 token budget、span 数、每段长度、layer 与 hook 时点。</td><td>排除“修复任意同样大的 hidden region”本身造成的收益。</td></tr>
  </tbody></table></div></details>
  <p>每个 seed-count 单元包含三个 baseline conditions，以及每层的 endpoint/full-span/ordinary 三种 restoration：Qwen 每单元 3+36×3=111 行、共 33,300 行；Gemma 3+42×3=129 行、共 38,700 行。总计 72,000 unique intervention rows；70,200 行含 patch，51,617 行使用优化后的两次 hook application，18,583 行为兼容保留的三次 application，审计 mismatch=0。<strong>目前结论：</strong>数据覆盖完整逐层扫描，不是从少数“好看层”外推的 coarse sweep。</p>
  <div class="formula"><strong>主分数只做两次减法。</strong>第一步，计算恢复 clean needle states 后少了多少误差：<span class="math">A^N_{{repair}}(ℓ)=|E_{{N-corrupt}}−N|−|E_{{N-restored,ℓ}}−N|</span>。第二步，减去恢复同样多 ordinary-position states 带来的机械收益：<span class="math">S_{{restore}}(ℓ)=A^N_{{repair}}(ℓ)−A^O_{{repair}}(ℓ)</span>。这里 <span class="math">E[c]=Σ_{{c=1}}^{{10}}c·p(c)</span> 是十个数字候选的 softmax expected count。<span class="example">例：gold N=8。擦掉 needles 后 E[c]=3，误差为 5；在 L8 恢复 whole-span states 后 E[c]=6，误差降为 2，因此 needle repair=3 counts。若同层 ordinary control 只修复 0.2 count，则图上的 L8 点为 3−0.2=2.8 counts。正值表示恢复真正的 needle evidence 有额外收益，0 表示它并不比恢复一块同样大的 ordinary hidden region 更有用。</span><details class="collapsible-list"><summary>展开：辅助 normalized recovery</summary><p><span class="math">R_{{restore}}=(E_{{restored}}−E_{{corrupt}})/(E_{{clean}}−E_{{corrupt}})</span>。0.6 表示恢复 clean–corrupt displacement 的 60%；它可因 overshoot 大于 1，也可因方向错误小于 0，所以正文以有 count 单位且更稳定的 <span class="math">S_{{restore}}</span> 为主。</p></details></div>
  <figure><h4 class="figure-title">图 4b · Canonical dense span restoration 的完整逐层曲线</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{span_layerwise_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{span_layerwise_charts['Gemma4-E4B']}</div></div><figcaption>每个 panel 的横轴是只执行一次 clean-state restoration 的 zero-based post-block layer；纵轴是 <span class="math">S_{{restore}}(ℓ)</span>，即 full-needle restoration 相对等 token-budget ordinary restoration 多减少的 expected-count absolute error，单位 counts。实线是 whole-span 主分数；橙色虚线由 <span class="math">(full−ordinary)−(full−endpoint)</span> 得到，等价于 endpoint−ordinary。每层先在同一 confirmation seed 内平均 counts 1–10，再对十个 seeds 等权平均；whisker 是 50,000 次 seed-cluster bootstrap 95% 区间。实心主曲线点表示 two-sided exact <span class="math">2^{{10}}</span> seed sign-flip nominal <span class="math">p&lt;0.05</span>，空心点表示未达到；这里按要求不做 Holm 或其他跨层校正，所以“显著”只指逐层 nominal evidence。竖虚线标出 paired seed-level 最大相邻层下降，不是事后拟合的连续 change-point model。</figcaption></figure>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>正向 nominally detectable layers</th><th>最大相邻层下降 [95% CI], exact p</th><th>微小负向 nominal layers</th><th>Endpoint−ordinary 全层范围</th></tr></thead><tbody>{span_transition_rows_html}</tbody></table></div>
  <p><strong>Qwen 是多层斜坡，而不是单层开关。</strong>Whole-span specificity 从 L0–L14 大致保持在 2.5–2.7 counts，随后连续下降；L20 仍为 {f(qwen_l20['mean'])} [{f(qwen_l20['ci95_low'])}, {f(qwen_l20['ci95_high'])}]，L21 只剩 {f(qwen_l21['mean'])} [{f(qwen_l21['ci95_low'])}, {f(qwen_l21['ci95_high'])}] 且不再 nominally detectable。最大一步是 L{qwen_span_drop['from_layer']}→L{qwen_span_drop['to_layer']} 的 {f(qwen_span_drop['mean'])} counts [{f(qwen_span_drop['ci95_low'])}, {f(qwen_span_drop['ci95_high'])}]，exact p={p_text(qwen_span_drop['exact_signflip_p'])}。因此 Qwen 有效窗口是 L0–L20 共 21 层，转换主要铺在约 L15–L22。</p>
  <p><strong>Gemma 更接近真正的 cliff。</strong>L16 仍有 {f(gemma_l16['mean'])} [{f(gemma_l16['ci95_low'])}, {f(gemma_l16['ci95_high'])}] counts；到 L17 立刻变成 {f(gemma_l17['mean'])} [{f(gemma_l17['ci95_low'])}, {f(gemma_l17['ci95_high'])}]。L{gemma_span_drop['from_layer']}→L{gemma_span_drop['to_layer']} 的 paired drop 为 {f(gemma_span_drop['mean'])} [{f(gemma_span_drop['ci95_low'])}, {f(gemma_span_drop['ci95_high'])}]，exact p={p_text(gemma_span_drop['exact_signflip_p'])}，十个 seed 的差值全部为负。Gemma 的正向有效窗口因此是 L0–L16 共 17 层；discovery 选出的 L17 虽未复现“恰好减半”的数值，却准确落在 confirmation cliff 上。</p>
  <h4>同一次 restoration 是否会改变后续 answer-query attention？</h4>
  <div class="formula"><strong>Attention-response specificity。</strong>对同一 frozen head <span class="math">h</span>，先算 true-needle restoration 相对 needle-corrupt baseline 的变化 <span class="math">δM_h^N(ℓ)=M_h^{{N-restored(ℓ)}}−M_h^{{N-corrupt}}</span>；再算等 token-budget ordinary restoration 相对 ordinary-corrupt baseline 的机械变化 <span class="math">δM_h^O(ℓ)=M_h^{{O-restored(ℓ)}}−M_h^{{O-corrupt}}</span>。图中蓝线为 <span class="math">ΔM(ℓ)=|ℋ|^{{-1}}Σ_{{h∈ℋ}}𝔼_{{s,N}}[δM_h^N(ℓ)−δM_h^O(ℓ)]</span>；对 broad score <span class="math">B=M×C</span> 做同样两次减法得到 <span class="math">ΔB(ℓ)</span>。正值表示修复 true needle evidence 比修复同样大的 ordinary hidden region 更能改变后续 answer-query routing。<span class="example">例：needle baseline/restored 的 mass 为 0.20/0.48，ordinary baseline/restored 为 0.21/0.22，则 ΔM=(0.48−0.20)−(0.22−0.21)=0.27。若 broad score 两组为 0.15/0.39 与 0.16/0.165，则 ΔB=0.235。它们是 attention-derived units，不是修复了 0.27 或 0.235 个 count。</span></div>
  <div class="experiment"><div class="experiment-label">Canonical routing diagnostic</div><div><h4>Restore span states once, rebuild the downstream cache, and re-measure the final frozen broad bank</h4><p>本图直接复用正式 dense sweep 的全部 canonical seeds 1234–1263 × counts 1–10，并在每个 layer/head 条目上平均 300 prompts；Qwen/Gemma 使用最终预冻结的 top-32/top-8 causal head registry，不根据这条 response curve 重新选 head 或 layer。Qwen 扫 L0–L35，Gemma 扫 L0–L41。每个 patched forward 只在指定 post-block layer 恢复一次；随后按冻结实现重建 downstream attention cache，记录同一 forward 的 answer-query attention。它不是“又独立算了一张 attention map”，而是同一干预的 downstream routing readout。曲线是 descriptive matched-control estimate，不在这里追加逐层显著性筛选。</p></div></div>
  <figure><h4 class="figure-title">图 4c · Canonical full-span restoration 对后续 broad retrieval attention 的逐层影响</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{span_attention_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{span_attention_charts['Gemma4-E4B']}</div></div><figcaption>横轴是只恢复一次 clean full-needle-span states 的 zero-based post-block layer；纵轴是 true-needle response 再减 ordinary-restoration response 的 matched specificity。蓝线为 active needle 总 attention mass specificity <span class="math">ΔM</span>；模型色虚线为同时考虑 mass 与多-span coverage 的 broad-score specificity <span class="math">ΔB</span>。Qwen 的 ΔB 在 L0/L16/L20/L21/L24/L26 为 0.310/0.302/0.196/0.061/0.023/0.010，L27 后为 0；Gemma 在 L0/L16/L17/L20/L22/L23 为 0.208/0.327/0.134/0.102/0.102/0。ordinary-control response 全程接近 0，因此主要变化来自 true needle evidence。竖虚线标出独立 behavior restoration curve 的主要 reusable-source 边界；attention-only 的衰减尾部不等于仍能显著修复最终答案。</figcaption></figure>
  <div class="claim"><strong>Restoration→retrieval 结论。</strong>在较早层恢复完整 needle-span evidence，会重新配置后续 answer-query 的 broad retrieval；这一影响在 Qwen 约持续到 L20、Gemma 约持续到 L16。这里的“持续”专指同时伴随 behavior repair 的主要可用窗口；canonical attention readout 还显示较弱尾部，分别延至 Qwen L26 与 Gemma L22，但此时已不足以恢复最终 count。它定位的是“prompt evidence 在深度上何时仍可被后续 retrieval 使用”，而不是 retrieval head 直接读取了哪个历史层。Transformer 的某个 retrieval head 只读取其自身输入深度上的 token states；layerwise restoration 改变的是这些 token states 经后续 blocks 演化后，是否还能影响该 head 的 routing。</div>
  <div class="claim boundary"><strong>如何读晚层的负值。</strong>Qwen L24–L27/L29–L30 与 Gemma L20–L22 有约 −0.04 至 −0.12 count 的 nominal negative specificity：此时恢复 needle positions 反而比 ordinary control 略差。它们比早层 +2–3 counts 小一个数量级，只说明 late patch 有轻微 overshoot/control imbalance；不能解释成模型在这些层使用“反向 counter”。Endpoint−ordinary 在全层仅 Qwen {f(span_summary_by_model['Qwen3-8B']['endpoint_minus_ordinary_min'])}…+{f(span_summary_by_model['Qwen3-8B']['endpoint_minus_ordinary_max'])}、Gemma {f(span_summary_by_model['Gemma4-E4B']['endpoint_minus_ordinary_min'])}…+{f(span_summary_by_model['Gemma4-E4B']['endpoint_minus_ordinary_max'])} counts，继续支持 endpoint 单点不充分。</div>
  <details class="collapsible-list"><summary>展开：discovery pilot 与预冻结 transition landmarks</summary><div class="experiment"><div class="experiment-label">Discovery-only contrast</div><div><h4>Endpoint 几乎不能修复，whole span 在早层能修复 2–3 counts</h4><p>用于冻结设计的 fresh pilot 只含 seeds 2000–2003 与 counts 3/6/9，不与 canonical confirmation 合并。Qwen endpoint normalized recovery 在所有层仅 −0.047…+0.032；whole-span recovery 在 L0/4/8/12 为 1.018/0.997/0.976/0.982，L16=0.849、L20=0.198、L24 后约 0。其 full-minus-ordinary expected-error repair 在 L0–L12 为 2.870–3.037 counts，L16=2.611、L20=0.745。Gemma endpoint repair 为 −0.042…+0.062；whole-span specificity 在 L0/4/8/12/16/20/24 为 3.327/3.497/3.350/2.618/2.297/−0.032/0。Broad-score change 也在相同窗口衰减：Qwen L0/L16/L20/L24/L28 为 0.337/0.326/0.239/0.022/0；Gemma L0/L16/L20/L24 为 0.231/0.408/0.127/0。</p></div></div>
  <figure><h4 class="figure-title">图 4d · Discovery 预冻结 landmarks 在 confirmation 上的 readout</h4>{span_landmark_chart}<figcaption>横轴是 full-needle restoration 相对等 token-budget ordinary restoration 多减少的 expected-count absolute error，单位 counts；竖向每条 bar 是一个预定义 landmark。Early plateau 是 discovery seeds 1234–1253 前四分之一层的 median specificity；half-boundary 与 near-zero boundary 的层号只由 discovery curve 冻结，图中 bar value 则在 confirmation seeds 1254–1263 计算。Qwen 在 L19 仍有 +1.294，L23 已为 −0.074；Gemma L17 的 confirmation 值为 −0.088。完整曲线表明，Gemma 的 literal half-height 没有复现，但 L17 确实是 +2.018→−0.088 的 cliff 位置。不同 bar 来自不同阶段/层，不把 discovery 与 confirmation 混成一个总体均值。</figcaption></figure>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Discovery early plateau</th><th>Frozen half-boundary: confirmation specificity</th><th>Frozen near-zero boundary: confirmation specificity</th></tr></thead><tbody>{span_landmark_rows}</tbody></table></div></details>
  <div class="claim"><strong>Stage-I conclusion。</strong>旧 endpoint rank-3 null 不是“prompt states 没有 causal effect”：完整 needle span 在早层含有强、可复用的因果信息，而单个 endpoint 既不充分、其三维 count component 也不必要。逐层曲线把两个模型区分得更清楚：Qwen 的 reusable-source effect 在 L15–L22 逐步衰减，Gemma 在 L16→L17 突降。两者都表明在该边界之后才恢复 prompt positions 已经太晚，符合后续 retrieval 已开始/完成并把信息转入 answer-side state 的机制；该实验本身不定位是哪一个 span token 或哪一个 head 完成转换。我们因此将上游状态概括为 <em>distributed span-level evidence</em>。</div>
</section>

<section id="retrieval">
  <h2>4. Stage II — Broad retrieval：一条被自然使用的部分聚合路径</h2>
  <p class="lead">在 answer query <span class="math">q</span> 上，head 是否“broad”不能只由总 attention mass 决定：只盯住一个 needle 的 head 不是 aggregation head。我们同时要求 mass 高且覆盖多个 needles。该阶段的目标不是证明唯一 counting channel，而是证明一组预先冻结的 broad heads 及其 count-aligned output subspace 在自然 computation 中具有 matched-control causal effect。</p>
  <div class="claim"><strong>Stage-II hypothesis。</strong>Answer query 在中层通过多个 broad heads 并行读取 active needle spans，并把其中一部分聚合结果写入 count-aligned answer residual。Attention map 只用于定义和可视化 routing；最终机制主张由 head ablation、donor source patch 和 subspace mediation 共同支持。</div>
  <div class="formula"><strong>Broad retrieval score。</strong>对第 <span class="math">i</span> 个完整 active-needle span <span class="math">S_i</span>，<span class="math">m_{{i,h}}=Σ_{{j∈S_i}}α_h(q,j)</span>；总 needle mass <span class="math">M_h=Σ_i m_{{i,h}}</span>；<span class="math">p_{{i,h}}=m_{{i,h}}/M_h</span>；coverage <span class="math">C_h=exp(−Σ_ip_{{i,h}} log p_{{i,h}})/N</span>；最终 <span class="math">B_h=M_hC_h</span>。<span class="math">C=1</span> 表示均匀覆盖全部 N 个 needles，接近 <span class="math">1/N</span> 表示只覆盖一个。排名使用 discovery 数据上 <span class="math">B_h</span> 的平均值。<span class="example">例：N=4，四个 span masses 都为 0.10，则 M=0.40、C=1、B=0.40；若全部 0.40 只落在一个 span，则 M 仍为 0.40，但 C=1/4，B=0.10。</span></div>
  <div class="path" aria-label="Broad aggregation computation">
    <div class="node"><strong>Needle spans S₁…Sₙ</strong><small>分散在约 10k-token prompt</small></div>
    <div class="node"><strong>Answer query q</strong><small>每个 head 产生 α(q,j)</small></div>
    <div class="node"><strong>Broad head bank</strong><small>Qwen L23/L27；Gemma full L29/L35</small></div>
    <div class="node"><strong>pre-O z<sub>h</sub></strong><small>Σ αVh，先聚合再经各自 W<sub>O</sub></small></div>
    <div class="node"><strong>Answer residual</strong><small>写入 donor/count-related state</small></div>
  </div>
  <figure><h4 class="figure-title">图 5 · Answer-query broad-score attention maps</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{attention_maps['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{attention_maps['Gemma4-E4B']}</div></div><figcaption>每个 panel 的横轴是 attention head H，纵轴是 transformer layer L，均为 zero-based；每个 cell 是该 head 从最后一个 answer query（<code>Total:</code> 后首数字前）指向所有完整 active needle spans 的 discovery-mean broad score B。颜色越深 B 越大；黑框标出后续因果消融所用的 frozen Qwen top-32 / Gemma top-8，cell 内数字是 frozen rank。这里是 layer×head 的聚合 attention-score map，不是单个样本的 token×token raw attention matrix。</figcaption></figure>

  <h4>Frozen broad-head membership</h4>
  <div class="figure-grid">
    <div><h4>Qwen3-8B top-32</h4><div class="table-wrap"><table class="head-table"><thead><tr><th>Rank</th><th>Head</th><th>M</th><th>C</th><th>B</th></tr></thead><tbody>{selected_head_tables['Qwen3-8B']}</tbody></table></div></div>
    <div><h4>Gemma4-E4B top-8</h4><div class="table-wrap"><table class="head-table"><thead><tr><th>Rank</th><th>Head</th><th>M</th><th>C</th><th>B</th></tr></thead><tbody>{selected_head_tables['Gemma4-E4B']}</tbody></table></div><p class="lead">M 是完整 needle spans 的 attention mass，C 是 occurrence coverage，B=M×C。表内顺序严格来自 causal run 使用的 frozen membership；由于源 registry 与汇总 atlas 的数值聚合版本略有差异，B 列用于量级解释，rank/membership 以 frozen registry 为准。</p></div>
  </div>

  <div class="formula"><strong>Absolute count shift。</strong>对同一个样本，clean 生成数为 <span class="math">ŷ_0</span>，消融后为 <span class="math">ŷ_a</span>，定义 <span class="math">shift_{{abs}}=|ŷ_a−ŷ_0|</span>；它衡量输出被移动多少，不以 gold N 为参照，也不是 absolute error。Top-K 主效应在每个 seed 内计算 <span class="math">mean(shift_{{ranked}})−mean(shift_{{layer-matched random}})</span>，再对 20 seeds 等权平均。Clean-correct correct→wrong=<span class="math">𝟙[ŷ_0=N∧ŷ_a≠N]</span>。<span class="example">例：clean 输出 8；ranked-head ablation 输出 5，shift=3；matched random ablation 输出 7，shift=1；ranked-minus-random absolute count shift=3−1=2 counts。若 gold=8，该 ranked trial 的 correct→wrong=1；若 gold 不是 8，则它不进入 clean-correct 指标。</span></div>
  <div class="formula"><strong>逐 K 的区间与“统计可检出”判据。</strong>令第 <span class="math">s</span> 个 seed 的 ranked-minus-random 效应为 <span class="math">δ_s(K)</span>；报告值是 20 seeds 的等权均值 <span class="math">δ̄(K)=20^{{−1}}Σ_sδ_s(K)</span>。95% 区间由 10,000 次 seed-cluster bootstrap 得到。表中的 <span class="math">p&lt;0.05</span> 专指 two-sided exact seed sign-flip：固定 20 个效应绝对值，枚举全部 <span class="math">2^{{20}}</span> 个正负号并比较 <span class="math">|δ̄|</span>；这里不展示多重比较校正。这个 p 值回答“方向是否跨 seeds 稳定”，而 effect 的 count 单位回答“移动有多大”，两者不能互换。由于很多 seed effect 可以恰为零，bootstrap 区间与 sign-flip 判据偶尔会不同，因此同时列出正效应 seed 数。<span class="example">例：若 20 个 seed effects 都只有 +0.05 count，均值仍只有 +0.05，是很小的效应；但只有全正和全负两种符号分配达到同样大的 |mean|，所以 p=2/2²⁰=1.91×10⁻⁶。反过来，若大多数 seed effects 为 0，即使非零 seeds 都为正，p 仍可能不小。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 2A</div><div><h4>Ranked top-K ablation with layer-matched random-head controls</h4><p>在 answer-query forward pass 中消融 frozen broad top-K heads；control 抽取相同 layer distribution、相同数量的 random heads。主叙述的 Qwen K32（+1.623 counts，1.117–2.137，p=1.91e−06）与 Gemma K8（+0.767，0.607–0.950，p=1.91e−06）同时具有非小的 effect size 与稳定的 seed-level direction。逐 K 看，absolute-shift endpoint 在 Qwen 的 K4/K16/K32 达到 nominal p&lt;0.05，K1/K2/K8 未达到；Gemma 六个 K 均达到，但这不表示六个效应都大：例如 Gemma K4 只有 +0.087 count。Qwen K2 是需要特别说明的边界：均值 +0.040、bootstrap 区间 0.010–0.080，但只有 4/20 seeds 为正、其余为零，exact sign-flip p=0.125，因此不称为稳定可检出。Clean-correct correct→wrong endpoint 则在 Qwen K4/K16/K32 与 Gemma K1/K8/K16 达到 nominal p&lt;0.05。</p></div></div>
  <figure><h4 class="figure-title">图 6a · Broad-head ablation 的 absolute-shift 剂量曲线</h4>{retrieval_chart}<figcaption>横轴是同时消融的 frozen top-K heads（K=1/2/4/8/16/32）；纵轴是 ranked ablation 相对 layer-matched random ablation 增加的 absolute count shift，单位 counts。竖向 whisker 是 20-seed equal-weight mean 的 95% seed-bootstrap 区间；实心点表示 two-sided exact seed sign-flip nominal p&lt;0.05，空心点表示未达到该判据。主叙述使用 Qwen K32 和 Gemma K8；完整曲线非单调，说明这些 heads 不能解释为可简单相加的独立计数器，增加 K 同时改变冗余、补偿与扰动子空间。</figcaption></figure>
  <figure><h4 class="figure-title">图 6b · Clean-correct correct→wrong damage 的剂量曲线</h4>{retrieval_damage_chart}<figcaption>只保留 clean baseline 答对且格式有效、并在 ranked/random 条件间具有相同 stimulus ID 的样本。纵轴是 <span class="math">P(wrong|ranked, clean correct)−P(wrong|layer-matched random, clean correct)</span>；0.20 表示额外增加 20 percentage points 的 correct→wrong rate。Whisker、实心/空心点与图 6a 使用同一 20-seed bootstrap 和 exact sign-flip 判据。该 endpoint 直接衡量原本正确的行为是否被破坏：Qwen 的主要 damage 集中在 K16/K32，Gemma 集中在 K8/K16。</figcaption></figure>
  <h4>表 2 · 每个 K 的完整 matched-control ablation 结果</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>K</th><th>Δ absolute shift [95% CI]</th><th>Positive seeds</th><th>Shift nominal p&lt;.05?</th><th>Δ correct→wrong [95% CI]</th><th>Damage nominal p&lt;.05?</th></tr></thead><tbody>{topk_result_table}</tbody></table></div>
  <p class="lead">两列效应都是 ranked top-K 减去 layer-matched random-head control；正值表示 broad-ranked heads 被消融后，输出移动得更多或更容易由 clean-correct 变为错误。每个 K、每个模型均使用 20 seeds；每个 seed 有 5 个 ranked count examples，random control 为同 5 examples 的 3 个 layer-matched replicates。</p>
  <div class="claim boundary"><strong>如何解读大小。</strong>“p 很小”只表示 matched-control effect 的方向在 seeds 间很稳定，不表示 effect 很大。Absolute count shift 的量纲就是 counts：+0.087 仍然只是平均多移动 0.087 count；只有在所有额外变化都恰为一步时，它才可类比为约 8.7% 的样本额外移动一步。因而小 K 的结果最多支持“该 frozen head set 有可检出的行为特异性”，不能单独支持“它解释了主要计数行为”。更有机制意义的是效应量较大的 Qwen K32（+1.623）与 Gemma K8（+0.767），并且还需要后续 source patch、mediation 与 late-state interventions 共同闭环。</div>

  <div class="formula"><strong>Source transport、terminal adoption 与 mediation。</strong>令 <span class="math">E_R[c]</span> 与 <span class="math">E_{{patch}}[c]</span> 为 counts 1–10 的 candidate-sequence softmax expected count；continuous normalized transport=<span class="math">(E_{{patch}}[c]−E_R[c])/(N_D−N_R)</span>。若改用实际生成数，则 strict generated transport=<span class="math">(ŷ_{{patch}}−ŷ_R)/(N_D−N_R)</span>，invalid generation 记 0；两者都不裁剪，1 表示完成一个 receiver→donor displacement。Terminal adoption 把 L41 residual change 投到该层 frozen one-count step <span class="math">s_T</span> 后再除以 count gap：<span class="math">A_T=⟨h'_T−h_T,s_T⟩/[‖s_T‖²(N_D−N_R)]</span>。Qwen sequence readout 定义 <span class="math">g(x)=log p(a_D|x)−log p(a_R|x)</span>，source gain=<span class="math">g(x_{{patch}})−g(x_R)</span>。Exact-component mediation specificity 是 source intervention 后“正交 control block 保留的 gain/transport − exact induced-component block 保留的 gain/transport”。<span class="example">例：receiver N=3、donor N=8，E<sub>R</sub>[c]=3.2、E<sub>patch</sub>[c]=6.2，则 continuous transport=(6.2−3.2)/5=0.6；若 L41 投影得到 2.5 count-axis units，则 terminal adoption=2.5/5=0.5。若 source gain=0.40，orthogonal block 后为 0.35、exact block 后为 0.10，则 mediation specificity=0.25。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 2B</div><div><h4>Donor source-state patch and downstream mediation</h4><p>把 donor count 的 frozen broad-bank source/pre-O states 写到 receiver 的注册 query slice。独立 seeds 上，Qwen early top-4 产生 +{f(qwen_upstream_primary['early_effect']['mean'],4)} donor log-odds gain；在 L28 H16–H19 精确删除由该 patch 诱发的 natural component，相对等范数正交 block 的 mediation specificity 为 +{f(qwen_upstream_primary['mediation']['mean'],4)}。Gemma K2 source donor transport 为 +{f(gemma_candidate['source_donor_transport']['mean'],4)}，并在 L37 被 exact residual block 中介 +{f(gemma_candidate['exact_residual_mediation']['mean'],4)}。于是 head ablation 的集合效应被更具体的 source→mediator chain 支持。</p></div></div>
  <div class="claim"><strong>Attention 图的解释边界。</strong>Broad heads 更像“对多个已出现 records/evidence 进行并行读取并写入 answer query”，而不是从 prompt 某一个位置读取一枚稳定整数。图 5 的亮区只说明某层某头对完整 active spans 具有较高 broad score；它不区分 span 内哪个 token 是地址或内容。当前机制不依赖这一细分：行为因果性由 matched head ablation 与 source→mediator experiments 提供。</div>

  <div class="claim"><strong>先把 4.1 与 4.2 分开读。</strong><strong>4.1 只问“这些 broad heads 合起来写出的向量里，能不能看出 count？”</strong>这是 representation/decodability 问题。<strong>4.2 再问“模型自然运行时是否真的依赖其中的 count-aligned 方向？”</strong>这是 matched intervention 的因果问题。前者高于 chance 不能替代后者。</div>

  <h3>4.1 Broad-bank representation：这些 heads 合起来写出了什么？</h3>
  <p class="lead">把每个 broad head 想成一个并行“取证员”：attention 决定它从哪些 needle spans 取信息，value/output projection 把取回的内容写到 answer-query residual。我们在每个冻结层把这一组 heads 的<strong>实际 post-O writes 相加</strong>，得到一个 broad-bank state；随后只问能否从这个合计向量预测最终 count。这里没有删除、patch 或改变模型输出，所以本节仍是描述性 representation analysis。</p>
  <figure><h4 class="figure-title">图 6c · 4.1 实际测量的对象：从多头读取到一个合计写入向量</h4>
    <div class="path" aria-label="How the broad-bank representation is constructed">
      <div class="node"><strong>完整 needle spans</strong><small>计数证据分散在多个远端位置</small></div>
      <div class="node"><strong>Frozen broad heads</strong><small>每个 head 用 attention 读取多个 spans</small></div>
      <div class="node"><strong>每头 post-O write</strong><small>每个 head 真正加进 answer residual 的向量</small></div>
      <div class="node"><strong>合计 state w<sub>ℓ</sub>(q)</strong><small>把 frozen bank 的 writes 在同层相加</small></div>
      <div class="node"><strong>Held-out count readout</strong><small>discovery 拟合，confirmation 评估</small></div>
    </div>
    <figcaption>这是计算流程示意图，不是额外实验结果。被分类的对象不是 raw attention map，也不是完整 answer hidden state，而是指定层、指定 frozen broad-head bank 在 answer query 上实际写入 residual 的向量之和。若该向量能预测 count，只说明聚合输出携带 count information。</figcaption>
  </figure>
  <figure><h4 class="figure-title">图 6d · Broad-bank 合计输出的 held-out exact-count readout</h4>
    <div class="figure-stack"><div><h4>Qwen3-8B</h4>{retrieval_geometry_charts[0]}</div><div><h4>Gemma4-E4B</h4>{retrieval_geometry_charts[1]}</div></div>
    <figcaption>横轴是冻结的 broad-bank layer；纵轴是 confirmation seeds 1254–1263 上的十类 count accuracy，0.50 即 50%。实线为 regularized exact-count classifier，橙色虚线为 nearest-centroid；水平虚线 0.10 是十类 chance。两种简单 readout 得到接近结果，说明可读性不是由某一个复杂 classifier 单独制造的，但 38%–54% 也远未形成接近完美的离散 register。折线只连接实际测试层以方便阅读，尤其 Gemma 只测试 L29/L35，不能据连线推断中间层趋势。</figcaption>
  </figure>
  <p><strong>怎么读图 6d。</strong>Qwen 在 L21/23/24/26/27 的 exact accuracy 为 49%/54%/39%/44%/39%；Gemma 在 L29/L35 均为 38%，都高于 10% chance。对应 classifier MAD 为 0.69–1.45 count，表示预测即使不完全正确，通常仍离真实 count 约一 count。十个 count centroids 的前三维方差占比很高（0.968–0.995），但单样本 cosine silhouette 接近 0（−0.098…+0.011），所以更准确的图像是“十个平均点沿低维轨迹移动，但每个 count 周围的样本云彼此重叠”，而不是十个干净分开的 clusters。</p>
  <details class="collapsible-list"><summary>展开：4.1 的完整公式、稳定性指标与逐层数值</summary>
    <div class="formula"><strong>Broad-bank state。</strong>对 layer <span class="math">ℓ</span> 的 frozen head bank <span class="math">𝒮_ℓ</span>，定义 <span class="math">w_ℓ(q)=Σ_{{h∈𝒮_ℓ}}W_O^hΣ_jα_h(q,j)W_V^hh_j</span>。Canonical dense run 保存了 3,000 个 clean broad-bank states；basis 与 classifier 只用 discovery seeds 1234–1253 拟合，所有报告 prediction 都在 confirmation seeds 1254–1263 上计算。</div>
    <div class="formula"><strong>Retrieval-geometry readouts。</strong>Exact classifier 是训练折内 StandardScaler/PCA 后的十类线性分类器；nearest centroid 将 held-out state 分配给欧氏距离最近的 discovery count centroid；classifier MAD=<span class="math">mean|ĉ−c|</span>。Rank-3 centroid capture 衡量十个 count centroids 的前三奇异方向方差占比；cosine silhouette 衡量单样本同 count 是否比异 count 更近；bootstrap maximum principal angle 比较 seed-resampled rank-3 basis 与 full-discovery basis，角度越大表示估计的 subspace 越不稳定。<span class="example">例：counts [4,8] 被预测为 [5,6] 时，accuracy=0、MAD=(1+2)/2=1.5；centroid rank-3 capture=0.99 但 silhouette=0，表示 mean curve 近三维，却没有干净的 sample clusters。</span></div>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th>Frozen layer</th><th>Exact classifier acc.</th><th>Nearest-centroid acc.</th><th>Classifier MAD</th></tr></thead><tbody>{retrieval_geometry_rows}</tbody></table></div>
    <p>Seed-bootstrap 95th-percentile maximum principal angle 约 60°–87°，进一步说明 fitted rank-3 basis 会随 discovery seeds 明显变化；它不是一枚跨样本完全固定的三维寄存器。</p>
  </details>
  <div class="claim"><strong>4.1 目前结论。</strong>Broad-head bank 的合计写入中确实存在 noisy、低秩、可解码的 final-count geometry；但 classifier 只衡量“含不含、紧不紧”，不能证明模型自然生成答案时使用了这三个方向。这个因果问题交给 4.2。</div>

  <h3>4.2 Retrieval-subspace intervention：模型真的使用这部分 count 方向吗？</h3>
  <p class="lead">4.2 不再训练一个 classifier，而是直接改模型内部状态。对同一个 seed、count 和 layer，我们从 broad-bank write 中删除 fitted count-aligned rank-3 component；matched control 则在<strong>同一 output span、同一实际删除 norm</strong>下删除一个与 count basis 正交的 component。如果前一种删除让答案误差增加得更多，差异就不能简单归因于“删掉了一段同样大的向量”，而支持模型自然依赖 count-aligned content。</p>
  <figure><h4 class="figure-title">图 6e · 四条件配对设计：每个上游状态都比较 aligned removal 与等范数 orthogonal removal</h4>
    <div class="contrast-grid" aria-label="Four-condition retrieval-subspace intervention design">
      <div class="contrast-lane">
        <div class="contrast-source">Natural clean state<small>模型未经上游修复的正常 forward</small></div>
        <div class="contrast-branches">
          <div class="contrast-arm"><strong>A · 删 count-aligned rank-3</strong><small>得到 error e<sub>N,A</sub></small></div>
          <div class="contrast-arm"><strong>B · 删 equal-norm orthogonal</strong><small>得到 matched-control error e<sub>N,O</sub></small></div>
        </div>
        <div class="contrast-result"><strong>Natural specificity</strong> = e<sub>N,A</sub> − e<sub>N,O</sub><br>正值：自然计算更依赖 aligned component。</div>
      </div>
      <div class="contrast-lane restored">
        <div class="contrast-source">Full-span-restored state<small>先把 clean needle-span evidence 恢复到 corrupt run</small></div>
        <div class="contrast-branches">
          <div class="contrast-arm"><strong>C · 删 count-aligned rank-3</strong><small>得到 error e<sub>R,A</sub></small></div>
          <div class="contrast-arm"><strong>D · 删 equal-norm orthogonal</strong><small>得到 matched-control error e<sub>R,O</sub></small></div>
        </div>
        <div class="contrast-result"><strong>Restoration mediation</strong> = e<sub>R,A</sub> − e<sub>R,O</sub><br>正值：恢复出来的上游收益有一部分经过该 subspace。</div>
      </div>
    </div>
    <figcaption>四个条件在同一 seed-count-layer 内配对。左半边回答“正常 forward 是否使用该方向”；右半边回答“3.2 恢复回来的 span evidence 是否会经过该方向”。两条 contrast 的单位都是 expected-count absolute error 的 counts；它们不是 classifier accuracy，也不是 attention mass。</figcaption>
  </figure>
  <p><strong>图 6f 应该怎样读。</strong>模型色实线对应图 6e 左侧的 natural specificity；橙色虚线对应右侧的 restoration mediation。纵轴大于 0 表示 aligned removal 比等范数 orthogonal removal 更伤，0 表示目前没有方向特异证据。晚层回到 0 不表示 count information 消失，而表示<strong>这一个在 earlier broad-bank output 上拟合的 rank-3 basis</strong>已不再是晚层正在使用的参数化。</p>
  <figure><h4 class="figure-title">图 6f · Frozen retrieval rank-3 的方向特异损伤随层变化</h4>
    <div class="figure-stack"><div><h4>Qwen3-8B</h4>{retrieval_subspace_charts[0]}</div><div><h4>Gemma4-E4B</h4>{retrieval_subspace_charts[1]}</div></div>
    <figcaption>横轴是预先冻结的 zero-based intervention layer，纵轴是 aligned removal 相对 equal-realized-norm orthogonal removal 多造成的 expected-count absolute error，单位 counts。Qwen 在 L21–L23 为正、从 L24 起约为 0；Gemma 在 L29 明显为正、L35 回到约 0。曲线定位的是 retrieval-basis 的自然使用窗口，不是完整 count information 的寿命。只在图示 frozen layers 做了 intervention；Gemma L29 与 L35 之间的连线仅连接两个观测点，不代表测过 L30–L34 或证明线性下降。</figcaption>
  </figure>
  <p><strong>具体结果。</strong>Qwen L21 的 natural/restored effects 为 +0.198/+0.166 counts，L23 增至 +0.333/+0.265；L24/L26/L27 两条 contrast 都约为 0，因此 Qwen 的该条 aggregation path 集中在 L21–L23。Gemma L29 为 +0.525/+0.527，L35 为 −0.010/−0.048，因此 Gemma 的对应窗口集中在 L29。这个时序与 3.2 一致：prompt-span restoration 在更早边界后开始来不及，而 broad-bank aligned component 随后在窄窗口内表现出方向特异因果作用。</p>
  <details class="collapsible-list"><summary>展开：4.2 的层、样本量、完整公式、逐层表与 clean-correct robustness</summary>
    <div class="table-wrap"><table><thead><tr><th>Design item</th><th>Frozen setting</th></tr></thead><tbody>
      <tr><td>Layers</td><td>Qwen L21/L23/L24/L26/L27；Gemma L29/L35。层在新 confirmation outcome 之前冻结。</td></tr>
      <tr><td>Population</td><td>confirmation seeds 1254–1263 × counts 1–10 = 100 paired seed-count units / layer。</td></tr>
      <tr><td>Four conditions</td><td>natural clean 与 full-span-restored 两种 upstream state；每种分别删除 fitted count-aligned rank-3 或 same output span 内的 equal-realized-norm orthogonal component。</td></tr>
      <tr><td>Coverage audit</td><td>400 unique rows / layer，7 layers 共 2,800；每层 paired-key、removed norm 与 orthogonal-overlap audits 均 PASS。</td></tr>
    </tbody></table></div>
    <div class="formula"><strong>Natural specificity 与 restoration mediation。</strong>Gold count 为 <span class="math">N</span> 时，<span class="math">S_{{natural}}=|E_{{clean+aligned}}−N|−|E_{{clean+orth}}−N|</span>；在 full-span restoration 后定义 <span class="math">M_{{restore}}=|E_{{restored+aligned}}−N|−|E_{{restored+orth}}−N|</span>。<span class="example">例：gold N=8；aligned removal 后 E[c]=6.5、orthogonal removal 后 7.5，则 specificity=|6.5−8|−|7.5−8|=1 count。</span></div>
    <div class="formula"><strong>Mediated fraction。</strong>对同一 seed-count 单元，以未 block 的 full-span expected-error repair <span class="math">A_{{repair}}</span> 为分母，定义 <span class="math">F_{{med}}=M_{{restore}}/A_{{repair}}</span>。它不裁剪、不是概率；分母很小或为负时可超出 [0,1]，所以 mean 与 median 只作量级描述。<span class="example">例：full-span restoration 原本修复 2 counts，aligned-specific block 额外损失 0.5 count，则 fraction=0.5/2=0.25；不能把它读成“25% 的 heads”。</span></div>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>Natural specificity mean</th><th>Restoration mediation mean</th><th>Mediated fraction mean</th></tr></thead><tbody>{retrieval_subspace_rows}</tbody></table></div>
    <p>Qwen L23 的 natural/restored medians 为 0.171/0.100，fraction median=0.031；Gemma L29 对应为 0.499/0.523、0.170。Clean-correct robustness 子集有 Qwen 44、Gemma 37 units：Qwen L23 restoration mediation +0.267、mean fraction 0.417；Gemma L29 +0.210、0.491。Clean-correct 是条件化 robustness，不替代 100-unit primary population。</p>
  </details>
  <div class="claim"><strong>4.2 与 Stage-II 目前结论：模型确实使用了这条聚合路径，但只在取回信息的阶段使用。</strong>
    <p><strong>第一，正常作答时，模型会依赖 broad-bank 中的 count-aligned component。</strong>在未经上游修复的自然 forward 中，删除该 component 比删除同样大小的正交 component 更伤答案：Qwen L23 多增加 <strong>0.333 count</strong> 的 expected-count error，Gemma L29 多增加 <strong>0.525 count</strong>。因此 4.1 中可解码的 count geometry 不只是 classifier 找到的相关结构；模型自身的计算也在使用它。</p>
    <p><strong>第二，这条 component 承接了 prompt span 中的一部分因果信息。</strong>先在 corrupt prompt 中恢复 clean needle-span states，再删除该 component，恢复收益会被特异地削弱：Qwen L23 损失 <strong>0.265 count</strong>，Gemma L29 损失 <strong>0.527 count</strong>，而等范数正交删除没有造成同样损失。换句话说，Stage I 恢复回来的 span evidence 中，至少有一部分随后经过这些 broad heads 的 count-aligned output 到达 answer query。</p>
    <p><strong>第三，这个实验没有说明全部计数都经过这里。</strong>阻断该 component 只消除了 full-span restoration 收益的一部分，模型仍保留剩余作用，因此还可能存在其他 heads、其他维度或后续 residual paths。我们据此称它为<strong>一条被自然使用的部分聚合路径</strong>，而不是唯一 counting channel。</p>
    <p><strong>第四，晚层效应回到约 0，不等于 count information 消失。</strong>它只说明在较早 broad-bank output 上拟合的这组 rank-3 directions，到 Qwen L24 以后或 Gemma L35 时已不再是模型当前使用的表示坐标。后续 Stage III 的 answer-state patching 与 removal 仍显示 count state 存在并可控制输出。因此更完整的机制是：<strong>中层 broad heads 取回并聚合一部分 span evidence，随后该信息被重新写成晚层 answer-side state</strong>；不是同一枚固定三维 counter 从 retrieval 原样保存到输出。</p>
  </div>
</section>

<section id="write">
  <h2>5. Stage III — Answer-side consolidation：从可解码到可执行</h2>
  <p class="lead">Section 2 已经表明 final count 在晚层 answer query 可由 exact-count classifiers 读取，但 decodability 本身不证明模型依赖该状态。本阶段因此组合四类互补证据：完整 donor-state patch 检验充分性，rank-3 removal 检验方向特异的必要性，相邻层三维 map 描述 centroid geometry 的可靠性，aligned 1× intervention 检验单个 block 对 count-aligned change 的选择性传播。</p>
  <div class="claim"><strong>Stage-III hypothesis。</strong>Broad retrieval 之后，模型在 answer-query residual 中逐渐形成一个可直接控制输出的 consolidated count state。若该假说成立，完整 donor state 应在中后层诱发 donor-answer adoption；删除冻结的 count-aligned component 应比等范数正交删除更伤答案；一个沿 count chord 的局部扰动应被下一 block 选择性接收。</div>

  <div class="formula"><strong>Full-state donor patch。</strong>对 receiver prompt R 与 donor prompt D，在层 <span class="math">ℓ</span> 的 answer-query 位置 <span class="math">q</span> 执行 <span class="math">h^R_ℓ(q)←h^D_ℓ(q)</span>，其余 receiver states 和 tokens 不变，再从 receiver 的下一步计算继续生成。Dense layerwise 指标只在 clean donor prediction 与 clean receiver prediction 不同的 eligible pairs 上定义 adoption=<span class="math">𝟙[ŷ_{{patch}}=ŷ_D]</span>。Correct-only pooled patching accuracy 则用更严格的 <span class="math">𝟙[ŷ_{{patch}}=N_D]</span>，并要求 receiver/donor clean 均答对。<span class="example">例：receiver clean 输出 5，donor clean/gold 都为 8；patched 输出 8，则 donor-prediction adoption=1 且 strict donor-gold hit=1；若 patched 输出 7，两者均为 0。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3A</div><div><h4>我们 patch 了什么、从哪里到哪里</h4><div class="table-wrap"><table><thead><tr><th>Patch object</th><th>Donor → receiver</th><th>Position / layer protocol</th><th>Readout</th></tr></thead><tbody>
    <tr><td>Prompt endpoint / full span residual</td><td>Gold count <em>N</em><sub>D</sub> prompt → different-count <em>N</em><sub>R</sub> prompt</td><td>改变的 k∈{{1,3,5}} 个 nested slots；needle-end 或整个 span；single layer 或从 start layer 累积 clamp 到最后层</td><td>strict patched count=donor gold；pooled Qwen {prompt_patching['Qwen3-8B']['patching_acc_successes']}/{prompt_patching['Qwen3-8B']['patching_acc_denominator']}={pct(prompt_patching['Qwen3-8B']['pooled_average_patching_acc'])}，Gemma {prompt_patching['Gemma4-E4B']['patching_acc_successes']}/{prompt_patching['Gemma4-E4B']['patching_acc_denominator']}={pct(prompt_patching['Gemma4-E4B']['pooled_average_patching_acc'])}。这是上游信息充分性，不与 rank-3 removal 的局部必要性矛盾。</td></tr>
    <tr><td>Full answer-query residual</td><td>Donor answer state → receiver 同层 <code>Total:</code> query</td><td>single-layer dense sweep；另有 frozen single/cumulative protocols 与 self-patch、same-count-seed controls</td><td>dense donor-prediction adoption；correct-only strict donor-gold hit。</td></tr>
    <tr><td>Broad-bank source / pre-O state</td><td>Donor slot-query state or donor pre-O z → receiver registered slice</td><td>Qwen early top-4 → L28 H16–H19；Gemma L29H4/L35H2 → L37 residual</td><td>donor candidate log-odds / normalized transport；再用 exact induced-component block 做 mediation。</td></tr>
    <tr><td>Count-aligned component</td><td>Receiver state + one frozen receiver→donor displacement</td><td>Qwen L28→29；Gemma L36→37；只报告 aligned 1×，并配 actual-norm-matched orthogonal 1× control</td><td>下一层相对 clean state 的 target-chord propagation coefficient F。</td></tr>
  </tbody></table></div><p>Correct-only full answer patch：Qwen {answer_patching['Qwen3-8B']['patching_acc_successes']}/{answer_patching['Qwen3-8B']['patching_acc_denominator']}={pct(answer_patching['Qwen3-8B']['pooled_average_patching_acc'])}；Gemma {answer_patching['Gemma4-E4B']['patching_acc_successes']}/{answer_patching['Gemma4-E4B']['patching_acc_denominator']}={pct(answer_patching['Gemma4-E4B']['pooled_average_patching_acc'])}。Dense sweep 中 Qwen L0/L9/L18 adoption=0，L26=53.3%，L29=98.3%，L35=100%；Gemma L0/L10/L20=0，L31=87.5%，L35=98.8%，L38–41=100%。<strong>目前结论：</strong>完整 answer state 的因果充分性不是浅层已有属性，而是在中后层快速形成，并在终层前接近饱和。</p></div></div>
  <figure><h4 class="figure-title">图 7 · Answer state 的因果可执行性在中后层出现</h4>{patch_chart}<figcaption>横轴是被替换的 single layer，纵轴是 eligible donor-prediction adoption rate。每点将 donor 的完整 post-block answer-query residual 写入 receiver 同层 query，随后继续 receiver forward pass。早层近 0，中后层陡升并接近 1；这证明完整 state 的充分性，不等同于某一个线性 count direction 的必要性。</figcaption></figure>

  <div class="formula"><strong>Answer-query absolute-error specificity。</strong>在每层只对 answer query state 删除相对全局 center 的 frozen count rank-3 projection；control 在正交 rank-3 basis 上删除相同实际 norm。定义 <span class="math">S<sub>abs</sub>(ℓ)=|ŷ<sub>count-remove,ℓ</sub>−N|−|ŷ<sub>orth-remove,ℓ</sub>−N|</span>，等价于两者相对 clean 的 absolute-error increase 之差。<span class="example">例：gold N=8，count removal 输出 5（error 3），orthogonal removal 输出 7（error 1），则 S<sub>abs</sub>=3−1=2 counts。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3B</div><div><h4>Layerwise answer-query rank-3 removal</h4><p>峰值 absolute-error specificity：Qwen L28 +0.878 counts（0.556–1.244；L24 +0.733，L32 +0.789）；Gemma L32 +1.222（1.044–1.411；L36 +1.189，L41 +1.133）。与 prompt endpoint 的近零效应相比，晚层 answer subspace 显示明确位置与方向特异性。<strong>目前结论：</strong>晚层 count-aligned geometry 不只是一个可读 correlational trace；在 matched orthogonal control 下，它对正确生成具有局部方向特异的必要性。</p></div></div>
  <figure><h4 class="figure-title">图 8 · Answer-query count subspace 的必要性随深度增加</h4>{removal_chart}<figcaption>横轴是层，纵轴是 count rank-3 removal 相对 actual-norm-matched orthogonal removal 增加的 absolute error，单位 counts；0 表示无方向特异性。共同模式是中后层强于浅层，也强于 prompt endpoint 的逐层 rank-3 removal。</figcaption></figure>

  <div class="formula"><strong>相邻层的三维 centroid-coordinate map。</strong>在每个 answer-query boundary <span class="math">ℓ→ℓ+1</span>，只用 20 个 discovery seeds 分别计算 counts 1–10 的 centroids <span class="math">C_ℓ,C_{{ℓ+1}}</span>，并分别拟合 rank-3 centroid-PCA bases <span class="math">U_ℓ,U_{{ℓ+1}}</span>。中心化三维坐标为 <span class="math">Z_ℓ=(C_ℓ−C̄_ℓ)U_ℓ</span>，再拟合 ridge map <span class="math">Â_ℓ=argmin_A ‖Z_{{ℓ+1}}−Z_ℓA‖²_F+λ‖A‖²_F</span>。这只是对相邻层 centroid geometry 的局部三维坐标映射，不假设整个 manifold 全局线性，也不要求所有层共享同一个 <span class="math">A</span>。</div>
  <div class="formula"><strong>图 9 左列：两个 error 的定义与意义。</strong>第一，5-fold seed-held-out centroid normalized RMSE 为 <span class="math">E^{{CV}}_ℓ=[Σ_f‖Z^{{test}}_{{ℓ+1,f}}−Z^{{test}}_{{ℓ,f}}Â_{{ℓ,f}}‖²_F/Σ_f‖Z^{{test}}_{{ℓ+1,f}}−Z̄^{{test}}_{{ℓ+1,f}}‖²_F]^{{1/2}}</span>，并满足 <span class="math">R²_{{CV}}=1−(E^{{CV}}_ℓ)²</span>。它问的是：只在训练 seeds 上拟合的局部三维 map，能否预测未见 seeds 的下一层 count-centroid coordinates；<span class="math">E^{{CV}}=0</span> 为完美预测，<span class="math">E^{{CV}}=1</span> 表示 residual energy 与“只预测该 test fold 的 target mean”相同。第二，对 20 个 discovery seeds 有放回重采样 500 次，每次重拟合两端 PCA bases 与 map；用 orthogonal Procrustes 把两端 PCA gauges 对齐到 full-discovery gauges 后，定义 <span class="math">E^{{boot}}_ℓ=median_b ‖Ã^{{(b)}}_ℓ−Â_ℓ‖_F/‖Â_ℓ‖_F</span>。它问的是：换一批 discovery seeds 后，估计出的 map 参数是否可复现；0 表示对齐后的重拟合 map 完全相同。两者都是无量纲的 geometry-estimation error，<strong>都不是</strong>生成数字的 absolute count error、分类错误率或 intervention 的行为效应。浅绿色 boundary 同时满足 <span class="math">R²_{{CV}}≥0.90</span>（等价于 <span class="math">E^{{CV}}≤√0.1≈0.316</span>）与 <span class="math">E^{{boot}}≤0.10</span>。<span class="example">例：若 held-out target centered energy=100、map residual energy=4，则 E<sup>CV</sup>=√(4/100)=0.20、R²=0.96；若 ‖Â‖<sub>F</sub>=10，而一次 gauge-aligned bootstrap map 与它的 Frobenius distance 为 0.5，则该次 relative map error=0.5/10=0.05。</span></div>
  <div class="formula"><strong>图 9 右列：跨 boundary 的 full-operator cosine。</strong>raw PCA axes 在不同层可任意翻转或旋转，因此先把三维坐标 map 重建为 ambient low-rank operator <span class="math">T_ℓ=U_ℓÂ_ℓU^⊤_{{ℓ+1}}</span>，再定义相邻两个 maps 的 Frobenius cosine：<span class="math">C^{{next}}_ℓ=⟨T_ℓ,T_{{ℓ+1}}⟩_F/(‖T_ℓ‖_F‖T_{{ℓ+1}}‖_F)</span>。它衡量连续两个 boundary 的映射<strong>方向与结构</strong>是否一致：1 表示同向且只允许整体正比例缩放，0 表示 Frobenius-orthogonal，−1 表示完全反向。这里用 cosine 而不用 drift，是因为问题关心三维映射方向能否跨层延续，不希望把整体增益变化混进来；它同样不衡量最终计数行为。<span class="example">例：若 <span class="math">T_{{ℓ+1}}=2T_ℓ</span>，两者 scale 不同但方向完全相同，因此 <span class="math">C^{{next}}_ℓ=1</span>；若 <span class="math">T_{{ℓ+1}}=−T_ℓ</span>，则 cosine=−1。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3C</div><div><h4>三维 map 的 held-out predictability、seed reproducibility 与方向连续性</h4><p>对 Qwen 的全部 35 个、Gemma 的全部 41 个相邻 answer-query boundaries 执行上述 discovery-only fit。Qwen 有 {map_stable_counts['Qwen3-8B']}/35 个 boundaries 同时通过两个 local-error cutoffs，Gemma 有 {map_stable_counts['Gemma4-E4B']}/41；两模型最后一个未通过的 boundary 都是 L18→19，此后全部通过。用于 1× 因果实验的 Qwen L28→29：<span class="math">E^{{CV}}={f(selected_map_rows['Qwen3-8B']['cv_centroid_normalized_rmse'],4)}</span>、<span class="math">E^{{boot}}={f(selected_map_rows['Qwen3-8B']['bootstrap_map_relative_frobenius_median'],4)}</span>，且它与下一张 L29→30 map 的 <span class="math">C^{{next}}={f(selected_map_rows['Qwen3-8B']['full_operator_cosine_to_next'],4)}</span>；Gemma L36→37 对应为 {f(selected_map_rows['Gemma4-E4B']['cv_centroid_normalized_rmse'],4)}、{f(selected_map_rows['Gemma4-E4B']['bootstrap_map_relative_frobenius_median'],4)}、{f(selected_map_rows['Gemma4-E4B']['full_operator_cosine_to_next'],4)}。因此选中的 late boundaries 具有可靠的局部三维 centroid relation；cosine 较高但低于 1，说明映射方向跨层连续、却不是逐层复制同一个 operator。</p></div></div>
  <figure><h4 class="figure-title">图 9 · Answer-query 三维相邻层映射：error 与跨层 cosine</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4><div class="chart-pair"><div><h4>Local-map errors（越低越好）</h4>{map_error_charts['Qwen3-8B']}</div><div><h4>Full-operator cosine（越高越连续）</h4>{map_cosine_charts['Qwen3-8B']}</div></div></div><div><h4>Gemma4-E4B</h4><div class="chart-pair"><div><h4>Local-map errors（越低越好）</h4>{map_error_charts['Gemma4-E4B']}</div><div><h4>Full-operator cosine（越高越连续）</h4>{map_cosine_charts['Gemma4-E4B']}</div></div></div></div><figcaption>四幅图的横轴均显示 current map 的 target layer，因此点 <span class="math">29</span> 表示 current map 是 L28→29；右图该点比较 L28→29 与下一张 L29→30 map。最后一张 map 没有可比较的 next boundary，所以每个右图都比对应左图少一个点。左列使用 log 纵轴：绿色是未见 seed 上的 centroid-coordinate prediction error，紫色是 500 次 seed-bootstrap 重拟合后的 gauge-aligned relative map error；浅绿色只标记两个 error cutoffs 同时通过。右列使用 linear 纵轴，橙线是 gauge-invariant ambient operators 的 Frobenius cosine，虚线 1 表示方向相同但允许整体 scale 不同。晚层 error 降低而 cosine 总体升高，支持局部 map 变得可预测、可复现且方向更连续；cosine 仍低于 1，因此不支持所有晚层共享完全相同的固定 operator。</figcaption></figure>

  <div class="claim boundary"><strong>Map 与 causal basis 的关系。</strong>图 9 的 <span class="math">A_ℓ</span> 是三维 centroid coordinates 之间的描述性 map。下面的 1× intervention 使用相关但不相同的 source transport basis <span class="math">B_ℓ</span>：把 source ambient centroids ridge-regress 到 target 三维 coordinates，再对 regression weights 做 QR。两者共享 discovery centroids 与 target rank-3 geometry，但不能把“<span class="math">A_ℓ</span> 稳定”直接当作“模型自然使用 <span class="math">B_ℓ</span>”的因果证据。</div>

  <div class="formula"><strong>1× count-aligned intervention 与 control。</strong>令 source-layer centroid chord 为 <span class="math">c_ℓ(R→D)=μ^ℓ_D−μ^ℓ_R</span>，frozen transport basis 为 <span class="math">B_ℓ∈ℝ^{{d×3}}</span>，则唯一报告的 aligned displacement 是 <span class="math">δ^{{align}}_ℓ=B_ℓB^⊤_ℓc_ℓ(R→D)</span>，并在 answer query 执行 <span class="math">h^{{int}}_ℓ(q)=h^{{clean}}_ℓ(q)+δ^{{align}}_ℓ</span>。Control axis 是先从 discovery within-count residuals 删除 <span class="math">B_ℓ</span> 分量后得到的 top residual PC，因此与 <span class="math">B_ℓ</span> 正交；将它缩放到与 BF16 实际写入的 aligned displacement 完全相同的 norm，再加到同一 clean receiver state。两组只改变一个 source layer 的同一个 answer-query position，随后只继续运行一个 block。</div>
  <div class="formula"><strong>Target-chord propagation coefficient。</strong>在 target layer 令 <span class="math">d_{{ℓ+1}}=μ^{{ℓ+1}}_D−μ^{{ℓ+1}}_R</span>，定义 <span class="math">F_{{ℓ→ℓ+1}}=⟨h^{{int}}_{{ℓ+1}}(q)−h^{{clean}}_{{ℓ+1}}(q),d_{{ℓ+1}}⟩/‖d_{{ℓ+1}}‖²</span>。<span class="math">F=0</span> 表示干预未在 target chord 方向造成变化；<span class="math">F=1</span> 表示变化含一个完整的 <span class="math">R→D</span> centroid-chord 单位。它不表示最终 state 等于 donor centroid，也不等于最终生成数移动了一 count。<span class="example">例：target chord d=[10,0]，同一样本的 clean state=[3,4]、干预后=[12.5,6]；则 Δh=[9.5,2]，F=⟨[9.5,2],[10,0]⟩/100=0.95。第二维变化被投影忽略，而且基线是该样本的 clean state，不是 receiver centroid，所以不能把 0.95 读成“95% donor state”。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3D</div><div><h4>单剂量 aligned 1× 与等范数 orthogonal 1×</h4><p>固定测试四个相邻-count 方向 <span class="math">1→2,2→1,5→6,6→5</span> 和 10 个 confirmation seeds，因此每个 condition、每个 boundary 有 40 个 observations；统计单位是每个 seed 对四个 directions 的均值。Qwen L28→29 的 raw means 为 orthogonal 0.0069、aligned 1× 0.9486，matched contrast={f(transport_contrasts[('Qwen3-8B', 'aligned_dose_1_minus_orthogonal')]['mean_contrast'],4)} [{f(transport_contrasts[('Qwen3-8B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_low'],4)}, {f(transport_contrasts[('Qwen3-8B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_high'],4)}]。Gemma L36→37 的 raw means 为 0.0020、0.9779，contrast={f(transport_contrasts[('Gemma4-E4B', 'aligned_dose_1_minus_orthogonal')]['mean_contrast'],4)} [{f(transport_contrasts[('Gemma4-E4B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_low'],4)}, {f(transport_contrasts[('Gemma4-E4B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_high'],4)}]。</p></div></div>
  <div class="claim"><strong>支持的结论。</strong>等范数 orthogonal control 接近 0，而 aligned 1× 接近一个 target chord，说明下一 block 对 frozen count-aligned direction 具有选择性，而不是只对干预 norm 敏感。由于这里只测试一种非零 aligned intervention magnitude，这个实验不估计线性、次线性或任何 scaling law；它只检验一个预先定义方向的单步局部因果传播。自然 forward pass 是否使用该方向，仍需结合图 8 removal、broad-head ablation、source mediation 与 full-state answer patching。</div>
  <figure><h4 class="figure-title">图 10 · Aligned 1× 跨一个 block 的方向选择性传播</h4>{transport_chart}<figcaption>每条横条是一个 condition 的 mean <span class="math">F</span>；横轴单位是 target-layer <span class="math">R→D</span> centroid chord，虚线 1 表示一个 chord unit，而不是“到达 donor centroid”。每个模型只比较 actual-norm-matched orthogonal 1× 与 aligned 1×。该图读取的是相对于同一样本 clean target state 的下一层 hidden-state change，并只保留其沿 target chord 的分量。</figcaption></figure>
  <div class="claim"><strong>Stage-III conclusion。</strong>四类证据共同把晚层 answer representation 从“可解码”提升为“可执行”：full-state patch 证明充分性，rank-3 removal 证明自然 computation 对 count-aligned component 的方向特异依赖，aligned 1× 证明相邻 block 具有选择性接收能力，而跨层 map 表明 centroid relation 在晚层可预测且可复现。Map 是描述性证据，不承担因果证明；cosine 低于 1 也说明模型并非逐层复制一枚固定三维 counter，而是在连续重参数化一个可执行的 answer-side state。</div>
</section>

<section id="ov-write">
  <h2>6. Stage IIIb — Architecture-specific write：Qwen 的局部 OV 与 Gemma 的分布式 residual</h2>
  <div class="claim"><strong>先把 OV 讲清楚。</strong>一个 attention head 做两件事：attention weights 决定“去哪些 token 取信息”，这是 <em>where to read</em>；V projection 把那些 token 变成可传递的内容，再由 <span class="math">W_O</span> 加进 answer-query residual，这是 <em>what to write</em>。所谓 OV write，就是后半步 <span class="math">W_OV</span>：它不再问头看了哪里，而是问“这个头最终给答案位置增加了什么向量”。只有 attention map 还不能证明该向量包含计数，也不能证明模型使用了它。</div>
  <div class="path"><div class="node"><strong>Attention routing</strong><small>选中 prompt 中的证据</small></div><div class="node"><strong>Value content</strong><small>把证据变成 head 内部向量 z</small></div><div class="node"><strong>Output projection W<sub>O</sub></strong><small>把 z 写入共享 residual</small></div><div class="node"><strong>Later blocks</strong><small>继续整合、维持或修改</small></div><div class="node"><strong>Count logits</strong><small>最终数字分布</small></div></div>
  <p class="lead">这一节进一步问：Stage III 的可执行 state 由哪些组件写入？我们按强度递增检查四件事：自然 head output 是否随真实 count 有序变化；沿自然方向增减是否按符号移动 expected count；删除自然方向是否比等范数无关方向更伤答案；上游 patch 的效应是否经该组件传到后层。模型间不必共享同一个微观电路：Qwen 满足局部 OV-writer 证据链，Gemma 则更符合“若干 heads 参与、后续 residual 分布式承接”的实现。</p>
  <div class="claim"><strong>Stage-IIIb hypothesis。</strong>Broad retrieval 不会直接等同于最终 count logits；retrieved content 还要经 attention value/output path 或后续 residual blocks 写成 answer-side state。我们检验的是一条自然使用的写入路径，而非穷尽所有并行或冗余通道。</div>

  <h3>6.1 Qwen3-8B：可以定位到 L28 的局部 OV writer</h3>
  <div class="path"><div class="node"><strong>Prompt records</strong><small>有序但 noisy</small></div><div class="node"><strong>L23/L27 broad heads</strong><small>从多个位置取回证据</small></div><div class="node"><strong>L28 H16/H19 core</strong><small>把证据写进 count-relevant residual</small></div><div class="node"><strong>L29–L35 residual</strong><small>这部分信号继续存在</small></div><div class="node"><strong>Answer</strong><small>晚层状态决定数字</small></div></div>
  <div class="experiment"><div class="experiment-label">OV-1</div><div><h4>自然状态中确实带着 count 信息</h4><p>在未经干预的 forward pass 中，把 L28 core set {{H16,H19}} 的实际输出投影到预先冻结的“多写一个 count”方向。Gold count 每增加 1，投影坐标平均增加 <strong>{f(qwen_nat['natural_carrier_count_slope']['mean'],4)}</strong>。这说明自然 head output 的排列与 count 一致；这个数是 hidden-state 坐标的斜率，不是“输出数字增加 0.2174”。</p></div></div>
  <div class="experiment"><div class="experiment-label">OV-2</div><div><h4>沿自然写入方向增减，会按符号推动答案</h4><p>在 heads 的 pre-O state 上沿冻结方向施加 <span class="math">+β</span> 或 <span class="math">−β</span>，再经过它们自己的 <span class="math">W_O</span>。每增加一个 β，softmax expected count 平均改变 <strong>{f(qwen_nat['injection_dose_slope']['mean'],4)}</strong>。因此这条通道具有 signed steering capacity；β 是干预剂量，不等于一个输出 count。</p></div></div>
  <div class="experiment"><div class="experiment-label">OV-3</div><div><h4>模型自然运行时也在使用这条方向</h4><p>从自然 head output 中删掉 count-aligned component，并与“在同一个 <span class="math">W_O</span> output span 内删掉相同 norm、但方向正交”的 control 比较。自然轴 removal 使 expected-count absolute error 比 control 多增加 <strong>{f(qwen_nat['removal_error_axis_minus_control']['mean'],4)}</strong>，并使正确答案相对最佳错误答案的 log-prob margin 多下降 <strong>{abs(float(qwen_nat['removal_margin_axis_minus_control']['mean'])):.4f}</strong>。效应不大，但方向和 matched control 都对，支持“自然 computation 使用了这部分”，而不只是“我们可以人工 steer”。</p></div></div>
  <div class="experiment"><div class="experiment-label">OV-4</div><div><h4>上游取回的证据有一部分经这里传到后层</h4><p>Early broad-head donor patch 使 donor-vs-receiver candidate log odds 增加 <strong>{f(qwen_upstream_primary['early_effect']['mean'],4)}</strong>。随后精确阻断 L28 H16–H19 中由该 patch 诱发的 component，相对等范数正交阻断消去 <strong>{f(qwen_upstream_primary['mediation']['mean'],4)}</strong> 的 gain。另一条 donor-z 实验中，冻结的自然 count axis 解释 donor transport 的 <strong>{pct(qwen_axis_mediated_fraction)}</strong>。所以这个 axis 是路径的一部分，但不是完整路径。</p></div></div>
  <div class="claim"><strong>Qwen 的简化结论。</strong>Broad heads 先从多个 prompt positions 收集证据；L28 core set {{H16,H19}} 再通过自己的 V→W<sub>O</sub> 通道，把其中一部分变成 answer residual 中可影响 count 的有符号变化；更宽的 H16–H19 set 用于上游 mediation 检验。Full/routing/value normalized transports 分别为 {f(qwen_read_write['read_full_behavior_transport']['mean'],4)} / {f(qwen_read_write['read_routing_behavior_transport']['mean'],4)} / {f(qwen_read_write['read_value_behavior_transport']['mean'],4)}，说明“看哪里”和“取到什么内容”都重要。H19 leave-one-out decrement={f(qwen_h19_loo['decrement']['mean'],4)}，说明 H19 在该集合中不可完全替代，但不能据此称它为单头计数器。</div>

  <h3>6.2 Gemma4-E4B：候选 heads 参与写入，但可确认的主要对象是分布式 residual path</h3>
  <div class="path"><div class="node"><strong>L29H4 / L35H2</strong><small>full-attention layers 可直接读取远端</small></div><div class="node"><strong>L37 answer residual</strong><small>出现可中介的分布式变化</small></div><div class="node"><strong>L38–L40</strong><small>同一 query 位置继续传递</small></div><div class="node"><strong>L41</strong><small>进入终端 count representation</small></div></div>
  <div class="experiment"><div class="experiment-label">Gemma head-level evidence</div><div><h4>L29H4 的自然信号、steering 与 removal 均为正，但并非全部具有 matched-head specificity</h4><div class="table-wrap"><table><thead><tr><th>Test</th><th>L29H4 raw candidate effect [95% CI]</th><th>Candidate-minus-control interpretation</th></tr></thead><tbody>
    <tr><td>Natural carrier slope</td><td>{f(gemma_l29h4['natural_carrier_count_slope']['mean'],4)} [{f(gemma_l29h4['natural_carrier_count_slope']['ci95_low'],4)}, {f(gemma_l29h4['natural_carrier_count_slope']['ci95_high'],4)}]</td><td>+{f(gemma_l29h4_specificity['natural_carrier_count_slope__candidate_minus_control_mean']['mean'],4)}；自然 count ordering 强于 matched heads。</td></tr>
    <tr><td>Signed pre-O injection slope</td><td>{f(gemma_l29h4['injection_dose_slope']['mean'],4)} [{f(gemma_l29h4['injection_dose_slope']['ci95_low'],4)}, {f(gemma_l29h4['injection_dose_slope']['ci95_high'],4)}]</td><td>{f(gemma_l29h4_specificity['injection_dose_slope__candidate_minus_control_mean']['mean'],4)}；能 steer，但不优于 matched heads。</td></tr>
    <tr><td>Removal error specificity</td><td>{f(gemma_l29h4['removal_error_axis_minus_control']['mean'],4)} [{f(gemma_l29h4['removal_error_axis_minus_control']['ci95_low'],4)}, {f(gemma_l29h4['removal_error_axis_minus_control']['ci95_high'],4)}]</td><td>+{f(gemma_l29h4_specificity['removal_error_axis_minus_control__candidate_minus_control_mean']['mean'],4)}；自然方向删除更伤 expected-count error。</td></tr>
    <tr><td>Removal margin specificity</td><td>{f(gemma_l29h4['removal_margin_axis_minus_control']['mean'],4)} [{f(gemma_l29h4['removal_margin_axis_minus_control']['ci95_low'],4)}, {f(gemma_l29h4['removal_margin_axis_minus_control']['ci95_high'],4)}]</td><td>{f(gemma_l29h4_specificity['removal_margin_axis_minus_control__candidate_minus_control_mean']['mean'],4)}；负值表示正确答案 margin 下降更多。</td></tr>
    <tr><td>Donor-z transport</td><td>{f(gemma_l29h4['donor_patch_transport']['mean'],4)} [{f(gemma_l29h4['donor_patch_transport']['ci95_low'],4)}, {f(gemma_l29h4['donor_patch_transport']['ci95_high'],4)}]</td><td>+{f(gemma_l29h4_specificity['donor_patch_transport__candidate_minus_control_mean']['mean'],4)}；候选头确实可传 donor-directed content。</td></tr>
  </tbody></table></div><p>这组结果不能被概括为“Gemma 没有 write effect”：L29H4 自然携带 count、可被 signed steering、方向删除会伤输出，并能传递 donor-directed content。未通过完整 natural-OV 判据的原因更具体：injection 的 candidate-minus-control 为 {f(gemma_l29h4_specificity['injection_dose_slope__candidate_minus_control_mean']['mean'],4)}，而 path-mediation specificity 仅 {f(gemma_l29h4_specificity['mediation_control_minus_axis_block__candidate_minus_control_mean']['mean'],4)}；因此无法声称 L29H4 是相对相似 heads 唯一或显著更强的局部 writer。<strong>目前结论：</strong>Gemma 有 head-level participation，但没有 Qwen 那样清楚的 localized-head exclusivity。</p></div></div>

  <div class="experiment"><div class="experiment-label">Gemma residual path</div><div><h4>L37 以后承接 source effect，并把一部分 count-aligned change 传到终端</h4><p>把 {{L29H4,L35H2}} 的 donor source 写给 receiver 后，normalized donor-count transport 为 <strong>{f(gemma_candidate['source_donor_transport']['mean'],4)}</strong>。在 L37 精确阻断这次 source patch 诱发的完整 residual change，可消去 <strong>{f(gemma_candidate['exact_residual_mediation']['mean'],4)}</strong>；只阻断预先冻结的线性 count axis，则消去 <strong>{f(gemma_candidate['count_axis_mediation']['mean'],4)}</strong>。到 L41，terminal count-adoption coefficient 为 <strong>{f(gemma_candidate['terminal_count_adoption']['mean'],4)}</strong>；它是 residual projection 相对 donor–receiver count gap 的归一化系数，不是 exact accuracy。结合 Gemma answer rank-3 removal 在 L32 达 +1.222 counts、correct-only full answer patch 达 96.0%，以及 L36→37 aligned/orthogonal 1× 的 0.9779/0.0020，Gemma 的晚层 write/consolidation 是正结果，只是定位粒度不同于 Qwen。<strong>目前结论：</strong>L37 的分布式 residual state 承接了主要被测 source effect，其中只有一部分落入预定义线性 count axis；后续层继续把它转换为终端 count state。</p></div></div>
  <div class="claim boundary"><strong>Gemma 的证据边界。</strong>不能写成“已经定位到与 Qwen 同等强度的单一 OV head set”，也不能写成“Gemma 没有 OV/write mechanism”。当前数据支持的是：周期性 full-attention heads 参与远端 retrieval 与写入，随后由 L37 及更晚层 answer-query residual 分布式承接。该模型差异与 Gemma 较弱的 prompt ordering 和较差的行为表现相容，但现有实验没有证明前者导致后者。</div>

  <details class="collapsible-list"><summary>展开：OV 各分数的精确定义与例子</summary>
    <div class="formula"><strong>1. Head output 与 natural carrier slope。</strong>Head <span class="math">h</span> 在 answer query 聚合出的 pre-O content 为 <span class="math">z_h(q)=Σ_jα_h(q,j)W_V^hh_j</span>，实际写入 residual 的是 <span class="math">W_O^hz_h(q)</span>。若候选 head set 的冻结一-count写入方向为 <span class="math">m_S</span>，则 carrier coefficient <span class="math">κ</span> 是自然 set output 在单位方向 <span class="math">m̂_S</span> 上的投影；natural carrier slope 是每个 seed 内 <span class="math">κ</span> 对 gold count 的 OLS slope。<span class="example">例：counts [1,2,3] 的 κ=[0.2,0.4,0.6]，则 slope=0.2 hidden-coordinate units/count；它不是 0.2 output counts。</span></div>
    <div class="formula"><strong>2. Injection slope 与 write behavior specificity。</strong>对 counts 1–10 的 candidate scores 做 softmax，定义 <span class="math">E[c]=Σ_{{c=1}}^{{10}}cp(c)</span>。在自然写入方向施加 <span class="math">±β</span>，中心差分 <span class="math">g=(ΔE[c]_{{+β}}−ΔE[c]_{{−β}})/(2β)</span>；write specificity 是自然方向的 <span class="math">g</span> 减去等 post-O norm 正交方向的 <span class="math">g</span>。<span class="example">例：β=1 时自然方向的 ΔE 为 +0.08/−0.04，正交方向为 +0.01/−0.01；两条 slope 为 0.06 与 0.01，specificity=0.05 expected counts/β。</span></div>
    <div class="formula"><strong>3. Removal error、margin 与 mediation。</strong>Expected-count error 为 <span class="math">e_E=|E[c]−N|</span>；removal error specificity=<span class="math">e_E(axis-remove)−e_E(orth-remove)</span>。Correct margin=<span class="math">log p(a_N)−max_{{c≠N}}log p(a_c)</span>；margin specificity 也取 axis-remove 减 orth-remove，负值表示自然轴 removal 更伤正确答案。Path mediation=<span class="math">T_{{orth-block}}−T_{{exact/axis-block}}</span>，mediated fraction 再除以未阻断的 donor-patch transport。<span class="example">例：gold N=8，axis/orth removal 的 E[c] 为 6.5/7.5，则 error specificity=|6.5−8|−|7.5−8|=1；若 donor transport=0.20、orth-block 后=0.18、axis-block 后=0.14，则 mediation=0.04、fraction=20%。</span></div>
    <div class="formula"><strong>4. Routing/value decomposition。</strong>RR/DD 分别是 receiver/donor 的真实 pre-O endpoints；RD 保留 receiver attention weights 但换成 donor values，DR 则保留 donor attention weights但使用 receiver values。定义 <span class="math">Δz_{{value}}=½[(z_{{RD}}−z_{{RR}})+(z_{{DD}}−z_{{DR}})]</span>、<span class="math">Δz_{{route}}=½[(z_{{DR}}−z_{{RR}})+(z_{{DD}}−z_{{RD}})]</span>，两者相加等于 full donor movement。Behavior transport 是各 component 引起的 expected-count change 除以 donor–receiver count gap。<span class="example">例：receiver count=3、donor count=8，某 component 使 E[c] 从 3.2 变为 4.2，则 normalized transport=(4.2−3.2)/(8−3)=0.20。</span></div>
  </details>
  <div class="claim"><strong>Stage-IIIb conclusion。</strong>两模型共享功能阶段，但不共享已解析到同一粒度的电路。Qwen 给出“broad retrieval → L28 H16/H19 OV write → late residual”的局部路径；Gemma 给出“full-attention-head participation → L37 distributed residual mediator → terminal state”的路径。报告因此把 OV/residual write 作为一条经验证的部分通道，而不声称它是唯一实现。</div>
</section>

<section id="ledger">
  <h2>7. Evidence synthesis：按机制排序的实验—证据—结论表</h2>
  <div class="table-wrap"><table class="evidence-map"><thead><tr><th>Stage</th><th>Experiment / operation</th><th>Readout</th><th>Observed value</th><th>Supported conclusion</th></tr></thead><tbody>
    <tr><td>Representation</td><td>Seed-held-out ridge at needle ends</td><td>R² / MAD</td><td>Qwen L8 0.945 / 0.561；Gemma L9 0.719 / 1.249</td><td>Running index follows a continuous, noisy, linearly decodable geometry.</td></tr>
    <tr><td>Representation</td><td>Frozen 3-PC counter-property + grouped cubic position control</td><td>line R² / step cosine / position-residual R²</td><td>Qwen L8 0.633 / 0.761 / 0.043；Gemma L9 0.403 / 0.084 / 0.004</td><td>Qwen has a stronger counter-like endpoint trajectory, but neither model's frozen 3D curve is separable from absolute-position structure.</td></tr>
    <tr><td>Representation baseline</td><td>Seed-held-out exact classifiers at answer query</td><td>L2 logistic and nearest-centroid accuracy / MAD</td><td>Qwen L29: logistic 56.0% / 0.635, centroid 54.5% / 0.640；Gemma L37: logistic 53.0% / 0.720, centroid 55.0% / 0.615</td><td>Descriptive compactness/separability baseline for later native-thinking comparison; not causal evidence or a discrete-counter claim.</td></tr>
    <tr><td>Representation</td><td>Centroid vs all-state rank-3 capture</td><td>explained variance</td><td>centroid 94.7%–98.8%；all state 59.8%–79.9%</td><td>Low-dimensional mean trajectory sits inside a higher-dimensional contextual cloud.</td></tr>
    <tr><td>Representation</td><td>V4.4.5 reused-state display-only 3-PC nearest centroid</td><td>confirmation accuracy / integer MAD</td><td>Qwen L28 61% / 0.72；Gemma L37 63% / 0.43</td><td>These layers provide clearer 3D answer visualization; post-hoc display selection is excluded from causal inference.</td></tr>
    <tr><td>Formation</td><td>Active-needle token corruption vs ordinary-token control</td><td>Δ absolute error</td><td>Qwen +8.930；Gemma +8.780</td><td>Needle evidence is causally required.</td></tr>
    <tr><td>Formation</td><td>All-endpoint prompt rank-3 removal vs orthogonal</td><td>Δ absolute error</td><td>Qwen +0.056；Gemma −0.022</td><td>No strong localized necessity for the decoded endpoint curve.</td></tr>
    <tr><td>Formation</td><td>72,000-row dense endpoint/full-span/ordinary restoration</td><td>full-minus-ordinary expected-error repair；seed-cluster CI / exact sign flip</td><td>positive nominal window Qwen L0–20（21 layers）、Gemma L0–16（17）；largest drop Qwen L20→21 −0.796、Gemma L16→17 −2.106 counts；endpoint−ordinary near 0</td><td>Prompt evidence is causally reusable as a distributed whole-span state; Qwen loses reuse gradually over L15–22, whereas Gemma has a sharp L16→17 boundary.</td></tr>
    <tr><td>Formation → retrieval</td><td>Canonical 一次性 full-span restoration 后重建 cache 并读取 frozen answer-query broad bank</td><td>(needle restored−needle corrupt)−(ordinary restored−ordinary corrupt) 的 mass ΔM / broad score ΔB</td><td>Qwen ΔB：L0 0.310、L16 0.302、L20 0.196、L21 0.061、L26 0.010；Gemma：L0 0.208、L16 0.327、L17 0.134、L22 0.102、L23 0</td><td>较早层修复 prompt evidence 会特异地重新配置后续 broad retrieval；behavior-coupled 主窗口约止于 Qwen L20 / Gemma L16，另有不再修复答案的 attention-only tail。该曲线不表示 retrieval head 直接读取某个历史层。</td></tr>
    <tr><td>Retrieval</td><td>Broad top-K head ablation vs layer-matched random</td><td>absolute count shift</td><td>Qwen K32 +1.623 [1.117, 2.137], p=1.91e−06；Gemma K8 +0.767 [0.607, 0.950], p=1.91e−06</td><td>Both frozen primary head sets have seed-level matched-control effects.</td></tr>
    <tr><td>Retrieval</td><td>Broad-bank state geometry on confirmation seeds</td><td>exact/nearest-centroid accuracy; rank-3 capture; silhouette</td><td>accuracy 38%–54% / 38%–53%；centroid rank-3 0.968–0.995；silhouette −0.098…0.011</td><td>The retrieved aggregate has a low-rank mean count trajectory but noisy and unstable individual-state geometry.</td></tr>
    <tr><td>Retrieval</td><td>Aligned retrieval rank-3 removal vs equal-norm orthogonal, natural and restored</td><td>natural specificity / restoration mediation</td><td>Qwen L23 +0.333 / +0.265；Gemma L29 +0.525 / +0.527；later frozen layers approximately zero</td><td>The fitted retrieval subspace is causally used in a localized aggregation window, not as a persistent late counter.</td></tr>
    <tr><td>Retrieval</td><td>Donor source patch + downstream exact block</td><td>source gain / mediation</td><td>Qwen +{f(qwen_upstream_primary['early_effect']['mean'],4)} / +{f(qwen_upstream_primary['mediation']['mean'],4)}；Gemma +{f(gemma_candidate['source_donor_transport']['mean'],4)} / +{f(gemma_candidate['exact_residual_mediation']['mean'],4)}</td><td>Retrieval writes a donor-directed state into a downstream path.</td></tr>
    <tr><td>Patch</td><td>Prompt full-span donor→receiver patch</td><td>strict donor-gold hit</td><td>Qwen {pct(prompt_patching['Qwen3-8B']['pooled_average_patching_acc'])}；Gemma {pct(prompt_patching['Gemma4-E4B']['pooled_average_patching_acc'])}</td><td>Distributed prompt-span states contain sufficient upstream information, despite weak rank-3 necessity.</td></tr>
    <tr><td>Write</td><td>Full answer-state donor→receiver patch</td><td>strict donor-gold hit</td><td>Qwen 96.6%；Gemma 96.0%</td><td>Late answer-query state is sufficient to determine the count output.</td></tr>
    <tr><td>Write</td><td>Answer-query rank-3 removal</td><td>peak Δ absolute error</td><td>Qwen L28 +0.878；Gemma L32 +1.222</td><td>Late count-aligned state is direction-specifically necessary.</td></tr>
    <tr><td>Map</td><td>Discovery-only adjacent-layer rank-3 centroid maps</td><td>seed-CV centroid NRMSE / gauge-aligned bootstrap map error / full-operator cosine to next</td><td>Qwen L28→29 {f(selected_map_rows['Qwen3-8B']['cv_centroid_normalized_rmse'],4)} / {f(selected_map_rows['Qwen3-8B']['bootstrap_map_relative_frobenius_median'],4)} / {f(selected_map_rows['Qwen3-8B']['full_operator_cosine_to_next'],4)}；Gemma L36→37 {f(selected_map_rows['Gemma4-E4B']['cv_centroid_normalized_rmse'],4)} / {f(selected_map_rows['Gemma4-E4B']['bootstrap_map_relative_frobenius_median'],4)} / {f(selected_map_rows['Gemma4-E4B']['full_operator_cosine_to_next'],4)}</td><td>The selected late boundaries have predictive and reproducible local 3D centroid relations, with substantial but imperfect orientation continuity to the next boundary.</td></tr>
    <tr><td>Transport</td><td>Adjacent-layer aligned 1× vs actual-norm-matched orthogonal</td><td>target-chord propagation coefficient F</td><td>Qwen contrast +0.9417 [0.9127, 0.9670]；Gemma +0.9759 [0.9639, 0.9884]</td><td>An injected count-aligned change is selectively relayed across one block; this is local transport capacity, not by itself proof of natural-axis use.</td></tr>
    <tr><td>OV write</td><td>Qwen L28 H16/H19 natural pre-O injection/removal</td><td>expected-count slope / error specificity</td><td>+0.0640 / +0.0732</td><td>A localized natural OV transporter writes signed count content in Qwen.</td></tr>
    <tr><td>OV / residual write</td><td>Gemma L29H4 natural-OV tests + L37 residual mediation</td><td>carrier / injection / removal error / exact residual mediation</td><td>+0.1360 / +0.0612 / +0.0628 / +0.0864</td><td>L29H4 participates in count-relevant writing, while matched-head controls prevent a unique localized-writer claim; L37 is the confirmed distributed mediator.</td></tr>
    <tr><td>Integrated chain</td><td>Same-forward 11-arm source restoration × retrieval/late directional blocks</td><td>source / retrieval / late expected-count effects</td><td>Qwen +2.674 / +0.327 / +1.118；Gemma +2.670 / +0.521 / +1.215 counts；all ordered criteria PASS</td><td>The three stages form an ordered partial serial mediation in each model; negative interaction and remaining repair rule out an exhaustive unique path.</td></tr>
    <tr><td>Formation micro-circuit</td><td>Independent induction assay + canonical previous-successor edge removal vs matched ordinary edge</td><td>candidate-minus-control expected-error damage</td><td>Qwen −0.02193 [−0.03311,−0.01076]；Gemma −0.01207 [−0.02499,0.00127]</td><td>Induction-like heads exist in the synthetic assay, but the registered classical-induction edge specificity is not supported in canonical counting.</td></tr>
    <tr><td>Prompt-noise attribution</td><td>I×C×P factorial + selected outside-halo edge removal vs two matched controls</td><td>held-out full R² / candidate specificity gate</td><td>full R² −0.0221/−0.0893；candidate_exceeds_both_controls=false in both models</td><td>The registered factor model does not stably explain held-out scatter, and the selected halo edges are not specifically necessary beyond matched controls.</td></tr>
  </tbody></table></div>

  <h3>7.1 What is established—and what is not</h3>
  <ul>
    <li><span class="pill">Established</span> 两个位置都含 count geometry；prompt centroid trajectory 低维但单样本 noisy。</li>
    <li><span class="pill">Established</span> Qwen endpoint 三维轨迹比 Gemma 更有序、同向，但 absolute position 几乎与 running index 共线；去除 cubic position 后冻结前三维 readout 接近零。</li>
    <li><span class="pill">Established</span> 完整 needle span 的早层 state 可强力修复 corrupt behavior；endpoint-only restoration 近零。</li>
    <li><span class="pill">Established</span> 晚层相邻 answer-query boundaries 的局部三维 centroid maps 在 held-out seeds 上可预测、对 seed bootstrap 稳定；连续 ambient operators 的 cosine 较高但低于 1，表示方向更连续但并非固定不变。</li>
    <li><span class="pill">Established</span> broad-ranked answer-query heads、局部 retrieval rank-3、late answer residual state 与最终数字之间存在 matched-control causal effects；retrieval rank-3 effect 只在 Qwen L21–23 / Gemma L29 出现。</li>
    <li><span class="pill">Established</span> 同一 forward 的 nested intervention 在两模型均满足 source→retrieval→late→output 的三项有序判据；这是 partial serial mediation，而非把独立实验事后拼接。</li>
    <li><span class="pill">Falsified strong version</span> Synthetic relation-following head 的存在不足以建立 canonical classical-induction mechanism；冻结 previous-successor edge removal 未超过 attention/distance-matched ordinary-edge control。</li>
    <li><span class="pill">Falsified registered attribution</span> Identity/context/position factorial 的 held-out full model 为负 R²，selected outside-halo edges 也未同时超过两个 matched controls；因此不能把 prompt scatter 归因于这套简单受控分解。</li>
    <li><span class="pill">Not established</span> running-index direction与answer count direction是同一轴；事实上两者可近正交，因为位置、basis、局部 computation 与写入坐标系不同。</li>
    <li><span class="pill">Not established</span> 某一个 prompt token 或某一个 attention head 单独存储/计算完整整数。</li>
    <li><span class="pill">Scope boundary / closed</span> 不声称 endpoint trajectory 对 final count N 不变；当前 running-index capture 的 N 固定为 10，position control 说明 ordinal-position 解释不可忽略，因此不再为本文追加 cross-final-N experiment。</li>
    <li><span class="pill">Optional finer attribution</span> 早层 nonlinear/distributed code 如何被各 broad head 的 QK routing 与 V content 分别读取，仍可用 position-resolved path patching 细分；但 span-level restoration 已足以支持本文粒度的 distributed-evidence claim。</li>
  </ul>

  <h3>7.2 Reproducibility ledger</h3>
  <ol class="source-list">
    <li><code>v4_4_extension/geometry/*.csv</code>：rank、regression、clustering。</li>
    <li><code>v4_4_extension/counter_properties/*</code>：frozen 3-PC counter quantities、grouped cubic position control、跨层图与 PASS audit。</li>
    <li><code>v4_4_extension/all_token/{'{'}all_token_frozen_pca_projections.csv.gz,gated_curve_formula_tests.csv{'}'}</code>：endpoint/interior/ordinary/hard-negative formula controls。</li>
    <li><code>v4_4_causal_v2/baseline_by_count.csv</code>：按 gold count 的行为 accuracy / absolute error。</li>
    <li><code>v4_4_extension/classification/classification_all_*/answer_classifier_metrics.csv</code>：仅 answer-query seed-held-out classification；本报告读取 <code>logistic_l2</code> 与 <code>nearest_centroid</code> 两行系列。</li>
    <li><code>v4_4_extension/token_corruption/token_corruption_statistics.csv</code>。</li>
    <li><code>v4_4_extension/prompt_subspace_ablation/subspace_ablation_statistics.csv</code>。</li>
    <li><code>v4_4_extension/endpoint_attention_mask/*earlier_span_head_confirmation.csv</code>。</li>
    <li><code>v4_4/realistic_niah_v4_head_atlas.csv</code> 与 <code>v4_4_causal_v2/full_span_topk/{'{'}full_span_topk_membership,full_span_topk_primary_statistics{'}'}.csv</code>。</li>
    <li><code>v4_4/v4_4_answer_query_patching.csv</code> 与 <code>v4_4_causal_v2/correct_patching_pooled.csv</code>。</li>
    <li><code>v4_4_extension/layerwise_subspace/{'{'}layer_maps,map_causal_link,answer_query_removal,transport{'}'}/*.csv</code>：三维 map、gauge-aligned stability、removal 与单剂量 transport。</li>
    <li><code>v4_4_4/*analysis.json</code>：Qwen natural OV/read-write/upstream chain。</li>
    <li><code>v4_4_4/gemma/residual/k2/*analysis.json</code>：Gemma K2 residual path。</li>
    <li><code>v4_4_5_followup/campaign_summary.json</code>、<code>v4_4_5_followup/span_restoration/{{needle_minus_ordinary_specificity,full_minus_endpoint,layerwise_seed_statistics}}.*</code> 与 <code>plans/nonthinking-followup-experiment-log-20260813.md</code>：72,000-row dense restoration、逐层 seed-cluster statistics、23,400 answer-state rows、3,000 retrieval states、2,800 retrieval-subspace rows及 persistent-copy audits。</li>
    <li><code>v4_4_5_followup/span_restoration/attention_response_canonical.csv</code>：由 Filestream audited <code>analysis/span_restoration/broad_summary.csv</code>（SHA-256 <code>bd4c958f…b04d608a</code>）派生；覆盖 canonical 30 seeds×counts 1–10、Qwen L0–35 top-32 与 Gemma L0–41 top-8，并保存 needle/ordinary response 及两者 specificity。</li>
    <li><code>v4_4_5_followup/exp19/*/serial_summary.json</code>：每模型 1,100-row same-forward partial serial mediation 与 10,000-draw paired audit。</li>
    <li><code>v4_4_5_followup/exp22_v3/*/{{analysis_summary,canonical_registration,synthetic_audit}}.json</code>：独立 induction-like gate、300-row canonical matched-edge confirmation 与限定性 negative verdict。</li>
    <li><code>v4_4_5_followup/exp23_v2/*/{{analysis_summary,analysis_audit,outside_context_registration,complete,run_provenance}}.json</code>：240-row factorial、2,400 endpoint states、400-row outside-context panel、100 exact edge audits与双 control decision。</li>
    <li><code>v4_4_2/realistic_niah_v4_4_2_mode_geometry_attention_report.html</code>、<code>docs/realistic_niah_v4_4_2.md</code> 与 <code>scripts/analyze_realistic_niah_v4_4_2_counter_geometry.py</code>：Q21 opening-definition cue removal 的冻结 paired states、逐层 CKA/readout 与 intervention 边界。</li>
  </ol>
</section>

<section id="extension-audit">
  <h2>8. Non-thinking extension 问题审计：验证了多少，证伪了多少？</h2>
  <p class="lead">为避免把一个宽泛问题同时算作“回答”和“未回答”，这里把 <code>non-thinking extension.md</code> 的提问拆成 25 个可判定命题，并保留原始题号。<strong>已验证</strong>表示有直接实验支持限定后的正命题；<strong>已证伪</strong>只否定表中写出的强版本，不等于证明所有替代理论；<strong>部分回答</strong>表示证据已约束答案但命题本身不允许唯一识别；<strong>未完成</strong>保留为状态类别，但本轮计数为 0；<strong>已关闭</strong>表示当前论文主动不提出相应强主张，因此不再追加实验。原问题 19、22、23 的最终结果见 Appendix B–D；原问题 21 的完整 cue-removal 证据见 Appendix A；原问题 24 作为主动关闭的范围边界保留在表尾。</p>
  <div class="audit-summary" aria-label="Extension audit status counts">
    <div class="audit-card"><strong>{status_counts['verified']}</strong><span>已验证 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['falsified']}</strong><span>已证伪 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['partial']}</strong><span>部分回答 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['open']}</strong><span>未完成 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['closed']}</strong><span>已关闭 / 25</span></div>
  </div>
  <p>最重要的更新有四条：第一，prompt endpoint 确有可读的 counter-like ordering，但绝对位置足以解释冻结前三维的大部分信号；第二，endpoint 局部寄存器假说被否定，而 full-span distributed evidence 得到强因果支持；第三，broad-bank count subspace 的自然因果使用只出现在 Qwen L21–L23 / Gemma L29，之后转入不同的 late residual representation；第四，同一-forward experiment 19 正式闭合有序部分中介，而 experiments 22/23 分别否定 canonical classical-induction specificity 与简单 prompt-noise attribution package。<strong>目前总评：</strong>证据支持阶段性“形成—局部聚合—晚层执行”，反对一枚固定三维 counter、单一 induction edge registry 或一组简单 nuisance factors 穷尽机制。</p>
  <details class="collapsible-list"><summary>展开 25 项实验设置—结果—结论对照表</summary>
    <div class="table-wrap"><table class="extension-audit"><thead><tr><th>#</th><th>原问题 / 可判定命题</th><th>状态</th><th>实验设置与 control</th><th>具体结果</th><th>目前结论</th></tr></thead><tbody>{extension_audit_rows}</tbody></table></div>
  </details>
  <div class="experiment"><div class="experiment-label">Completed experiment 19 · audit PASS</div><div><h4>同一 forward 中的 source → retrieval → late answer → output 串联中介</h4><p>该实验已在 canonical confirmation panel（seeds 1254–1263 × counts 1–10，共 100 paired units/model）完成。每模型运行 11 arms、1,100 unique rows，并以 10,000 次 seed-unit bootstrap 估计区间；Qwen 固定 source/retrieval/late layers 为 L8/L23/L29，Gemma 为 L9/L29/L37。层、两个彼此独立的 rank-3 bases、normalization 与 sign 都在既有 discovery artifacts 上冻结，不按本实验结果重新选峰。Joint <span class="math">2×2</span> retrieval/late factorial 是正式 arm，因此“串联”不是只靠两个独立实验事后拼接。</p>
  <details class="collapsible-list"><summary>展开：实验 arms、读数、判据与数值例子</summary>
    <div class="table-wrap"><table><thead><tr><th>Arm</th><th>同一 forward 内的操作</th><th>用途</th></tr></thead><tbody>
      <tr><td>C / O</td><td>needle-corrupt；以及在同层恢复等 token 数、等长度 ordinary spans</td><td>corrupt reference 与 source-restoration matched control</td></tr>
      <tr><td>S</td><td>在 source layer 恢复 clean prompt 的完整 active needle spans</td><td>测量分布式 source repair</td></tr>
      <tr><td>S + R⊥ / S + R∥</td><td>先做 S，再在冻结 retrieval layer 删除等实际范数的正交 component / count-aligned component</td><td>检验 source repair 是否经局部 retrieval subspace 传递</td></tr>
      <tr><td>S + T⊥ / S + T∥</td><td>先做 S，再在冻结 late-answer layer 删除 matched orthogonal / count-aligned component</td><td>检验 retrieval 之后形成的 late state 是否把 repair 写向输出</td></tr>
      <tr><td>S + R<sub>a</sub> + T<sub>b</sub>，a,b∈{{⊥,∥}}</td><td>四个 joint arms：R⊥T⊥、R∥T⊥、R⊥T∥、R∥T∥；每一级均按本例 realized norm 匹配</td><td>检验两级效应是 additive、overlapping/occluding 还是 synergistic，并估计 joint block 后的 bypass；不要求两级 basis 相同</td></tr>
    </tbody></table></div>
    <div class="formula"><strong>行为读数。</strong>对 counts 1–10 的候选答案 softmax 定义 <span class="math">E[c]</span>，expected-count error 为 <span class="math">e(X)=|E[c]_X−N|</span>。Source repair 为 <span class="math">G<sub>S</sub>=e(O)−e(S)</span>；retrieval mediation 为 <span class="math">M<sub>R</sub>=e(S+R∥)−e(S+R⊥)</span>；late mediation 为 <span class="math">M<sub>T</sub>=e(S+T∥)−e(S+T⊥)</span>。三者均以 counts 为单位，并同时汇报 strict generated-count accuracy。<span class="example">例：gold N=8，ordinary control 的 E[c]=3（error 5），source restoration 后 E[c]=6（error 2），则 G<sub>S</sub>=3 counts。若 S+R⊥ 的 E[c]=5.8（error 2.2），S+R∥ 为 4.8（error 3.2），则 M<sub>R</sub>=1.0 count：删除 count-aligned retrieval component 比等范数正交删除多抹掉 1 count 的 repair。</span></div>
    <div class="formula"><strong>串联判据。</strong>除最终行为外，同时读取 frozen broad-bank count coordinate、needle attention mass / broad score 与 late-answer count coordinate。若（i）S 改变后续 broad routing 与 retrieval coordinate；（ii）R∥ 相对 R⊥ 同时削弱 late coordinate 和输出 repair；（iii）T∥ 相对 T⊥ 进一步伤害输出，但不反向改变更早的 retrieval readout，则建立有序的 partial serial mediation。Joint factorial 的 overlap contrast 定义为 <span class="math">I<sub>RT</sub>=[e(S+R∥+T∥)−e(S+R⊥+T∥)]−[e(S+R∥+T⊥)−e(S+R⊥+T⊥)]</span>：负值表示 late block 部分遮蔽 retrieval block 的附加损伤，零附近表示近似相加，正值表示协同；它只描述路径重叠，不证明唯一性。Joint aligned block 后的剩余 repair 为 <span class="math">G<sub>resid</sub>=e(O)−e(S+R∥+T∥)</span>；可报告未截断的 accounted fraction <span class="math">1−G<sub>resid</sub>/G<sub>S</sub></span>。<span class="example">沿用上例，若 joint block 后 E[c]=3.5（error 4.5），则 G<sub>resid</sub>=0.5，accounted fraction=1−0.5/3=0.833；其余 16.7% 可来自 bypass、冗余或测量误差。</span></div>
  </details><figure><h4 class="figure-title">图 8a · 实验 19 的三段正效应</h4>{exp19_chart}<figcaption>横轴是相对 matched control 的 expected-count effect（counts）。图中 source、retrieval、late 三段分别回答“完整 span 修复是否到达 retrieval”“count-aligned retrieval 是否影响后续 late state”“late count-aligned state 是否影响输出”；它们不是可直接求和的独立份额。</figcaption></figure><details class="collapsible-list"><summary>展开实验 19 的均值与 95% bootstrap CI</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Source repair</th><th>Retrieval mediation</th><th>Late mediation</th><th>Joint interaction</th><th>Remaining repair</th></tr></thead><tbody>{exp19_rows}</tbody></table></div></details><p><strong>结果与目前结论：</strong>三项 ordered criteria 在两个模型均通过；joint interaction 为 Qwen −0.382、Gemma −0.380，fully aligned block 后仍有 +1.477/+1.291 counts source repair。原问题 19 因此升级为<strong>支持同一 forward 的 ordered partial serial mediation</strong>。负 interaction 与 remaining repair 明确反对“两个冻结 mediator 穷尽唯一通路”的强版本。</p></div></div>
  <div class="claim boundary"><strong>审计结论。</strong>25 项中，12 项限定后的正命题已有直接支持，9 个过强机制版本被 matched-control 或 paired robustness experiments 否定，3 项已有实质约束但仍属部分回答，没有仍待运行的预注册机制实验，另有 1 项因当前论文不提出相应强主张而关闭。Q21 只证伪“opening definition cue 必要”，并未删除计数问题与输出指令。尤其不能把“position-confounded counter-like geometry”升级为抽象计数器，也不能把“partial serial mediation”升级为唯一通道。</div>
</section>

<section id="limitations">
  <h2>9. What remains：还差什么没做，以及哪些并非当前主张所必需</h2>
  <p class="lead">现有证据已完成本文中尺度机制链的预注册补全：non-thinking counting 依赖分布式 prompt evidence，经局部 broad retrieval/aggregation 进入晚层可执行 answer state；同一-forward 串联中介在两模型均通过，Qwen 与 Gemma 则采用不同粒度的写入实现。Q22 与 Q23 也已运行并给出限定性负结果。因此当前没有尚待运行、会改变中心机制结论的 GPU 实验；剩余工作主要属于 token-level 精细归因或跨任务外部有效性。</p>
  <div class="table-wrap"><table><thead><tr><th>Open question</th><th>Why it matters</th><th>Decisive experiment</th><th>Priority for the present paper</th></tr></thead><tbody>
    <tr><td>Attention endpoint peaks 的 QK address 与 V content 如何分工？</td><td>这会把 Stage II 从 span/head-bank 层级推进到 token-level circuit；attention map 本身不承担因果结论。</td><td>严格 token registration 后分别 patch Q/K routing、V content 与 pre-O z，并加入 span-interior 和 position-matched controls。</td><td><strong>低优先级。</strong>解析成本高；当前论文不需要声称唯一 token-level path。</td></tr>
    <tr><td>机制是否泛化到更长上下文、更多 count range 与非 city needles？</td><td>现有结论严格限定于冻结的 V4.4 panel、counts 1–10 与当前模板。</td><td>预注册 held-out templates、不同 needle semantics、不同 context lengths 与超出训练范围的 counts。</td><td><strong>投稿前最值得补的泛化实验之一，</strong>但不影响当前 panel 内的因果识别。</td></tr>
  </tbody></table></div>
  <div class="claim boundary"><strong>已证伪与主动关闭的范围。</strong>原问题 21 的强必要性版本已被证伪：只删除开头两句 task/record 定义后，浅层 running ordering 与 seed-held-out readout 几乎保留；但 counting query 与 numeric-output instruction 仍在，所以这不是“全部 task instructions 均非必要”的实验，完整证据见 Appendix A。原问题 24 才是主动关闭：当前报告只主张 position-confounded counter-like record，不主张 final-N invariant abstract counter。原问题 25 的固定 prompt rank-3→answer rank-3 直传强版本也已被证伪，且不是补全原问题 19 的必要条件。</div>
  <div class="claim"><strong>最终结论与优先级。</strong>中心机制链已经闭环，Q22/Q23 的注册强版本也已有负判定；继续租 GPU 不应再用于重复这三项。若资源仍可用，投稿前更高价值的下一步是跨模板、长度、needle semantics 与 count range 的预注册 generalization。QK/V address-content 拆分只在论文要升级为 token-level unique-circuit 主张时才值得做。</div>
</section>

<section id="appendix">
  <h2>Appendix · Q21、Q19、Q22、Q23 的完整实验定义与审计结果</h2>
  <p class="lead">本附录把主文不宜展开的 robustness、微电路和负结果完整保留。Appendix A–D 均为已经完成并通过 coverage/audit 的实验；其中 B 支持 ordered partial serial mediation，C 与 D 否定各自预注册的强版本，同时保留清楚的解释边界。</p>

  <h3>Appendix A · Q21：opening counting-definition cue 的必要性被证伪</h3>
  <div class="experiment"><div class="experiment-label">Completed · V4.4.2</div><div><h4>删除了什么，保留了什么</h4><p>paired intervention 只删除 prompt 开头两句定义：“需要数 passage 中的 city-score audit records”以及“record 的定义”。passage、全部 slots、计数问题、<code>Total:&lt;integer&gt;</code> numeric-output instruction 与 assistant formatting 均保持不变。正式 panel 使用 seeds 1234–1243；下面的 prompt running-index geometry 对每个模型使用 10 个 final-N=10 prompts，每个 prompt 读取第 1–10 个 needle endpoints，因此共有 100 对 cue-present/cue-absent endpoint states。V4.4.2 没有 discovery/confirmation split；ridge 在两种 cue 条件共同拟合的 shared six-PC basis 中使用固定 <span class="math">α=1</span>，并做 leave-one-seed-out prediction，不能把这 10 seeds 重新称为独立 confirmation。</p></div></div>
  <div class="formula"><strong>Centroid-topology linear CKA。</strong>在每个 layer，把十个 running-index centroids 排成矩阵 <span class="math">C<sup>+</sup></span>（cue present）与 <span class="math">C<sup>−</sup></span>（cue absent），按列中心化后形成 Gram matrices <span class="math">K<sup>+</sup>=C̃<sup>+</sup>C̃<sup>+T</sup></span>、<span class="math">K<sup>−</sup>=C̃<sup>−</sup>C̃<sup>−T</sup></span>，再计算 <span class="math">CKA=⟨K<sup>+</sup>,K<sup>−</sup>⟩<sub>F</sub>/(‖K<sup>+</sup>‖<sub>F</sub>‖K<sup>−</sup>‖<sub>F</sub>)</span>。它比较十个 count centroids 之间的关系是否保留，对全局旋转和统一缩放不敏感。<span class="example">例：若 cue removal 只把整条 centroid curve 旋转并放大两倍，两张 Gram matrix 只差统一比例，CKA=1；CKA 接近 0 才表示两种条件下的 centroid relations 不再对齐。</span></div>
  <div class="formula"><strong>Count η² 与 paired interaction η²。</strong><span class="math">η²<sub>count</sub>=SS<sub>between-count</sub>/SS<sub>total</sub></span>，表示完整 hidden-state variation 中由 running-index 分组解释的比例。interaction 先对每个 matched endpoint 求 cue displacement <span class="math">δ=h<sup>−</sup>−h<sup>+</sup></span>，再计算 <span class="math">δ</span> 的 count η²；因此它问“cue 造成的位移是否随 running index 系统变化”，不是行为 accuracy。<span class="example">例：若 displacement 的总平方能量为 100，其中 count-group means 占 48，则 paired interaction η²=0.48；它不表示 accuracy 改变了 48%。</span></div>
  <figure><h4 class="figure-title">图 A1 · 删除 opening definition cue 后，running-index geometry 随层变化</h4><div class="chart-pair"><div>{cue_cka_chart}</div><div>{cue_ridge_chart}</div></div><figcaption>左图横轴是 zero-based transformer layer，纵轴是同层 cue-present 与 cue-absent 十个 count centroids 的 linear CKA；纵轴从 0.94 起截断，用于放大小偏差，虚线 1 表示 centroid relations 完全一致。右图只显示预先用于 prompt geometry 的 Qwen L8 与 Gemma L9；横轴为 model/layer/condition，纵轴是在 pooled shared six-PC basis 中计算的 leave-one-seed-out ridge <span class="math">R²</span>（固定 <span class="math">α=1</span>）。两图共同问 low-dimensional ordering 是否在 cue removal 后仍保留，而不是 cue 是否对完整 residual 完全无影响。</figcaption></figure>
  <details class="collapsible-list"><summary>展开 Q21 代表层数值</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>Centroid CKA</th><th>Ridge R² present / absent</th><th>Count η² present / absent</th><th>Paired interaction η²</th></tr></thead><tbody>{cue_appendix_rows}</tbody></table></div></details>
  <div class="claim"><strong>Q21 的精确结论。</strong>Qwen L8 的 CKA=0.9995、ridge R² 0.845→0.840、count η² 0.645→0.633；Gemma L9 的 CKA=0.9999、ridge R² 0.343→0.355、count η² 0.440→0.433。由此可证伪“这两句 opening definitions 是形成有序 running geometry 的必要条件”。但 paired interaction η² 仍为 Qwen 0.484、Gemma 0.332，说明 cue 会以 count-dependent 方式调制完整 state。最重要的边界是：计数问题和输出指令没有删除，所以不能外推成“模型在没有任何 task instruction 时也会形成同一 geometry”。</div>

  <h3>Appendix B · Q19：同一-forward partial serial mediation</h3>
  <details class="collapsible-list"><summary>展开完整实验定义（已完成；audit PASS）</summary>
    <p><strong>目的。</strong>现有 full-span restoration、restoration→attention response、retrieval-subspace intervention、late answer intervention 与 write/mediation 分别支持相邻箭头。新实验不再寻找一枚跨层固定 basis，而是在同一个 seed-count forward 中检验“修复 source 后产生的收益，是否依次依赖冻结的 retrieval 与 late-answer components”。</p>
    <p><strong>Cohort 与冻结项。</strong>canonical confirmation seeds 1254–1263 × counts 1–10，共 100 paired units/model。Qwen 固定 source/retrieval/late layers 为 L8/L23/L29，Gemma 为 L9/L29/L37；所有 bases、head registries、normalization 与 sign 都从既有 discovery artifacts 读取并记录 SHA-256，不允许按本实验结果重新选 layer、rank 或 direction。</p>
    <div class="table-wrap"><table><thead><tr><th>Arm family</th><th>操作</th><th>识别对象</th></tr></thead><tbody>
      <tr><td>C / O / S</td><td>needle corruption；等 token-budget ordinary-span restoration；clean full-needle-span restoration</td><td>corrupt reference、matched source control 与 source repair</td></tr>
      <tr><td>S+R⊥ / S+R∥</td><td>S 后，在冻结 retrieval layer 删除 equal-realized-norm orthogonal / count-aligned component</td><td>retrieval-stage direction specificity</td></tr>
      <tr><td>S+T⊥ / S+T∥</td><td>S 后，在冻结 late-answer layer 删除 matched orthogonal / count-aligned component</td><td>late-stage direction specificity</td></tr>
      <tr><td>S+R<sub>a</sub>+T<sub>b</sub></td><td>四个 joint arms：R⊥T⊥、R∥T⊥、R⊥T∥、R∥T∥</td><td>retrieval 与 late block 的 additivity、occlusion/synergy 及剩余 bypass</td></tr>
    </tbody></table></div>
    <p>总计 11 个 arm conditions × 100 units = 1,100 condition-forwards/model，两个模型共 2,200；若实现共享 clean/corrupt cache，可减少计算但不能减少逻辑 coverage。每个 arm 保存 counts 1–10 candidate logits、strict generation、frozen broad mass/score、retrieval coordinate 与 late-answer coordinate。</p>
    <div class="formula"><strong>行为量与 interaction。</strong>令 <span class="math">e(X)=|E[c]<sub>X</sub>−N|</span>。source repair 为 <span class="math">G<sub>S</sub>=e(O)−e(S)</span>；retrieval/late mediation 分别为 <span class="math">M<sub>R</sub>=e(S+R∥)−e(S+R⊥)</span> 与 <span class="math">M<sub>T</sub>=e(S+T∥)−e(S+T⊥)</span>。joint interaction 为 <span class="math">I<sub>RT</sub>=[e(S+R∥+T∥)−e(S+R⊥+T∥)]−[e(S+R∥+T⊥)−e(S+R⊥+T⊥)]</span>：负值表示 late block 遮蔽一部分 retrieval damage，零附近表示近似相加，正值表示协同。<span class="example">例：若在 T⊥ 下 R∥ 相对 R⊥ 多造成 1.0 count error，而在 T∥ 下只多造成 0.3，则 I<sub>RT</sub>=0.3−1.0=−0.7，表示 late block 已遮蔽 retrieval block 的 0.7-count 附加作用；这不证明两者是唯一通道。</span></div>
    <p><strong>通过判据。</strong>顺序上应同时看到：（i）S 相对 O 改变随后 broad routing/retrieval coordinate；（ii）R∥ 相对 R⊥ 削弱 late coordinate 与行为 repair；（iii）T∥ 相对 T⊥ 损伤输出，但不能反向改变已经计算完成的早期 retrieval readout。通过后只能称为 <em>partial serial mediation</em>；失败可能来自 bypass、冗余或 basis mismatch，不能否定 raw prompt evidence 的既有因果性。</p>
  </details>
  <details class="collapsible-list"><summary>展开 Q19 精确结果（均值 [95% bootstrap CI]，单位 counts）</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Source repair</th><th>Retrieval mediation</th><th>Late mediation</th><th>Joint interaction</th><th>Remaining repair</th></tr></thead><tbody>{exp19_rows}</tbody></table></div></details>
  <div class="claim"><strong>Q19 目前结论。</strong>两模型都通过三项有序判据；更晚 intervention 对已经计算完的 retrieval/broad readout 的最大变化为 0。负 joint interaction 表明 late block 遮蔽一部分 retrieval damage，fully aligned block 后仍有正 remaining repair。因而支持的是<strong>有序、非穷尽的 partial serial mediation</strong>，不是线性相加、唯一通道或同一三维 basis 的跨层搬运。</div>

  <h3>Appendix C · Q22：是否存在 classical induction-head micro-circuit</h3>
  <details class="collapsible-list"><summary>展开完整实验定义（已完成；audit PASS）</summary>
    <p><strong>为什么现有结果不够。</strong>某个 head 在 needle endpoint 回看较早 spans，只能说明 earlier-span preference；classical induction 还要求它跟随“当前重复 identity → 上一次同 identity 后面的 successor”这一关系，而不是只跟随绝对位置、相对距离或通用 record marker。</p>
    <p><strong>两阶段冻结。</strong>先只用 canonical discovery seeds 1234–1253，按已有 endpoint→earlier-span preference 冻结 candidate heads。随后让这些同一 heads 接受一个独立的 standard induction assay：为每个模型从稳定 single-token pool 生成 30 个固定 base sequences，并各自构造四种完全 token/position-matched 版本。<code>repeated-consistent</code> 含重复 anchor→successor pairs；<code>unique-anchor</code> 消除 previous identity match；<code>successor-reassignment</code> 固定两个 earlier successor 的内容与位置，只交换它们前面的等长 anchor identities，使“当前 anchor 的 previous-match successor”从一个位置移动到另一个位置；<code>same-position ordinary-repeat</code> 保留相同重复/位置统计但打破 anchor→successor relation。该 assay 共 30×4=120 forwards/model，不用 confirmation NIAH outcomes 选 head。</p>
    <div class="formula"><strong>Relation-following score。</strong>对 candidate head <span class="math">h</span>，在当前 anchor query <span class="math">q_t</span> 上，定义 <span class="math">I<sub>h</sub>=mean α<sub>h</sub>(q<sub>t</sub>, successor(previous matching identity))−mean α<sub>h</sub>(q<sub>t</sub>, matched non-successor)</span>。<span class="example">例：若 head 对 identity-defined successor 的平均 attention mass 为 0.20，对同距离 control 为 0.05，则 I<sub>h</sub>=0.15。在 successor-reassignment 中，matching anchor 从 earlier position 1 换到 position 2，而两个 successor positions 均不移动；relation-following mass 应从 successor 1 转到 successor 2。若仍盯原位置，更像 positional routing。</span></div>
    <p><strong>Canonical causal confirmation。</strong>只保留同时满足 canonical earlier-span preference 与 synthetic induction score 的冻结 heads；再在 discovery data 中为每个 retained head 冻结 repeated record-template anchor、current-anchor query offset 与 previous-occurrence successor key offset。在 NIAH confirmation seeds 1254–1263 × counts 1–10 上运行三 arms：natural、candidate-edge removal、matched-control removal，共 300 condition rows/model。candidate arm 在真实 pre-O head slice 上减去每条冻结 natural edge 的 <span class="math">α(q,k)V(k)</span> contribution；control 在同 layer/head 中删除相同 edge 数、相同 key-distance bins 且 natural attention mass 匹配的 non-successor contribution。该操作保留其余 frozen forward，不重新归一化 attention logits，因此是 natural edge-contribution removal，不是 fully renormalized QK counterfactual。随后比较 frozen broad retrieval、correct margin 与最终 expected/strict count。只有 synthetic relation-following、unique-anchor collapse 与 canonical downstream matched-control effect 三者同时成立，才把“induction-like”升级为“classical induction-head mechanism”。</p>
  </details>
  <figure><h4 class="figure-title">图 C1 · Synthetic gate 通过，但 canonical matched-block gate 失败</h4>{exp22_chart}<figcaption>横轴是 counts 2–10 主分析中，冻结 previous-successor candidate-edge removal 相对同 layer/head、同距离、同 edge 数且 natural attention mass 匹配的 ordinary-edge removal 所增加的 expected-count absolute error。正值才支持 classical-induction edge specificity；圆点是 10 个 seed 的平均，横线是 10,000-draw seed bootstrap 95% CI。Qwen 均值和完整 CI 为负；Gemma CI 跨 0，均未满足正向 gate。</figcaption></figure>
  <details class="collapsible-list"><summary>展开 Q22 synthetic 与 canonical 数值</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Frozen head</th><th>Repeated relation</th><th>Reassignment following</th><th>Unique / ordinary absolute response</th><th>Canonical expected-error candidate−control</th><th>Decision</th></tr></thead><tbody>{exp22_rows}</tbody></table></div></details>
  <div class="claim boundary"><strong>Q22 目前结论。</strong>独立 synthetic assay 确实找到 induction-like relation-following heads，但把同一注册关系带回 canonical NIAH 后，candidate edge 并不比严格 matched ordinary edge 更必要。因此<strong>预注册的 classical induction-head specificity 不受支持</strong>。这只否定该 frozen current-anchor→previous-successor αV contribution 的特异必要性；它不否定 earlier-span routing、其他 head/path registry，也不是 fully renormalized QK deletion。</div>

  <h3>Appendix D · Q23：identity、context、position 与 outside-context synergy</h3>
  <details class="collapsible-list"><summary>展开完整实验定义（已完成；audit PASS）</summary>
    <p><strong>为什么需要这组补充。</strong><code>docs/realistic_niah_v4.md</code> 中冻结的旧 panel 已构成逐级放松的 robustness ladder：V4.1 固定 position/order/content，V4.2 放开 position，V4.3 再放开固定 fact set 的 order，V4.4 再放开 city-score content。它说明 geometry 不只存在于一个完全固定 prompt，但旧因素不是完整交叉操纵，不能分别估计 identity、context、position 的受控变形及交互。Q23 因而用 factorial 检验一个简单 held-out nuisance model，并用局部 edge removal 检验先前粗粒度 outside-mask 现象能否获得 matched-control specificity。</p>
    <p><strong>Phase A：2×2×2 paired factorial。</strong>在 Qwen L8、Gemma L9 固定读取 running-index states，不重新选层。三个二值因素分别是：（I）active records 的 city/score surface identities 保持原样或用 tokenizer-length-matched pool 随机替换；（C）各 record 周围的 ordinary context 保持原样或在相同 length/depth bins 内跨 slot 置换；（P）record 保持原位置或与 exact-token-length ordinary carrier 交换到预先冻结的 gap-jittered slots，同时保持 record order、总 prompt length 与 answer-query position。每个模型先独立做 tokenizer/span audit。使用全部 30 base seeds × 8 cells × final N=10，共 240 prompt-forwards/model、2,400 endpoint states/model；discovery seeds 1234–1253 只拟合 nuisance model，confirmation seeds 1254–1263 报告效应。</p>
    <div class="formula"><strong>Held-out incremental R²。</strong>先在冻结三维 basis 中减去每个 running index 的 discovery centroid，得到 within-count residual。以 seed 为 group 做 held-out multivariate regression，full model 含 I/C/P 的 main effects 与 interactions；因素 <span class="math">F</span> 的增量定义为 <span class="math">ΔR²<sub>F</sub>=R²(full)−R²(full without every term containing F)</span>。<span class="example">例：full model 在 held-out seeds 上解释 60% residual variance，删去 position 及其 interactions 后只解释 20%，则 ΔR²<sub>P</sub>=0.60−0.20=0.40。它表示这组受控 position manipulations 对 scatter 的增量预测力，不等于自然数据中“40% 神经元由位置产生”。</span></div>
    <p><strong>Phase B：targeted outside-context natural-edge removal。</strong>Discovery seeds 在候选 heads 中冻结一个 source head，并在每个 confirmation unit 内按 natural attention 排序 ordinary halo keys；最多保留 16 条，同时受每个 64-token distance bin 的 distinct non-halo control capacity 约束。Seeds 1254–1263 × counts 1–10 使用四 arms：natural、candidate halo-edge removal、exact-distance random control、同 bin attention-mass control，共 400 rows/model。三种 removal arms 的 edge 数严格相等且大于 0，每条 control 与 candidate 同 distance bin、key 不重用。干预在 answer-query source-head pre-O slice 减去 frozen natural <span class="math">αV</span> contribution；不重算归一化 QK。只有 candidate removal 在 expected error 上同时超过两个 controls，才支持注册的 specificity claim。</p>
    <p><strong>解释边界。</strong>Factorial 的 ΔR² 是受控 deformation 对 held-out residual 的增量预测力，不是自然方差份额；Phase B 识别一个冻结 source-head registry 的特异必要性，不是 outside-context token census 或 pathway-uniqueness test。Negative result 因而约束当前简单解释，但不能把 observational scatter 唯一分配给某个来源，也不能推出 distributed outside-context synergy 不存在。</p>
  </details>
  <figure><h4 class="figure-title">图 D1 · 三类受控变形没有形成稳定 held-out nuisance model</h4>{exp23_factor_chart}<figcaption>横轴是 confirmation seeds 上的 incremental ΔR²：完整 I/C/P 主效应与交互模型的 held-out R²，减去删除所有含该因素项后的 R²。正值表示该因素在这套受控 manipulation 下增加 held-out prediction；零线表示没有增量。Qwen position 为 +0.0175，Gemma identity 为 +0.0031，其余接近或低于 0；更关键的是两个 full-model held-out R² 本身为 −0.0221 与 −0.0893，说明整套模型不如用 confirmation mean 预测。故这些条形不能解释为自然 prompt noise 的 variance share。</figcaption></figure>
  <figure><h4 class="figure-title">图 D2 · Selected outside-halo edge removal 未超过两个 matched controls</h4>{exp23_specificity_chart}<figcaption>横轴是 candidate halo-edge removal 相对各 matched control 多造成的 expected-count absolute error，单位 counts；圆点为 10-seed 均值，横线为 10,000-draw seed bootstrap 95% CI。Distance-random control 严格匹配 layer/head、edge count 和每条 key 的 distance bin；attention-mass control 还在同一 distance bin 内匹配 natural pre-intervention attention。两模型四个 CI 全部跨 0，因此注册判据 <code>candidate_exceeds_both_controls</code> 均为 false。</figcaption></figure>
  <details class="collapsible-list"><summary>展开 Q23 精确结果</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Full held-out R²</th><th>ΔR² I / C / P</th><th>Frozen source head</th><th>Expected error vs distance control</th><th>Expected error vs attention-mass control</th><th>Decision</th></tr></thead><tbody>{exp23_rows}</tbody></table></div></details>
  <div class="claim boundary"><strong>Q23 目前结论。</strong>预注册的强解释包未通过：I/C/P manipulation 没有形成稳定的 held-out additive/interaction model，selected outside-halo edges 也没有超出两个 matched controls 的特异必要性。这个负结果不等于 identity、context 或 position 在自然数据中“没有作用”，也不等于所有 outside context 都无用；它只说明当前简单分解不能唯一解释 observational scatter，且冻结的单一 source-head αV halo registry 不是可确认的特异通路。</div>
</section>

</main>
</article>
<script>
const GEOMETRY={payload_json};
const COLORS=['#2563eb','#0891b2','#0f766e','#65a30d','#ca8a04','#d97706','#ea580c','#dc2626','#9333ea','#4f46e5'];

function collapseReportLists() {{
  const targets=[...document.querySelectorAll('.table-wrap, main ul, main ol')];
  for(const node of targets) {{
    if(node.closest('details.collapsible-list')) continue;
    const details=document.createElement('details');
    details.className='collapsible-list';
    const summary=document.createElement('summary');
    const isTable=node.classList.contains('table-wrap');
    const count=isTable ? node.querySelectorAll('tbody tr').length : node.children.length;
    summary.textContent=isTable ? `展开表格明细（${{count}} 行）` : `展开列表明细（${{count}} 项）`;
    node.parentNode.insertBefore(details,node);
    details.append(summary,node);
  }}
}}
collapseReportLists();

class PointCloud3D {{
  constructor(canvasId, modelSelectId, layerSelectId, site, defaults) {{
    this.canvas=document.getElementById(canvasId);
    this.modelSelect=document.getElementById(modelSelectId);
    this.layerSelect=document.getElementById(layerSelectId);
    this.site=site;
    this.defaults=defaults;
    this.ctx=this.canvas.getContext('2d');
    this.yaw=-0.7; this.pitch=0.35; this.dragging=false; this.last=null;
    this.modelSelect.addEventListener('change',()=>{{this.layerSelect.value='';this.populateLayers();this.draw();}});
    this.layerSelect.addEventListener('change',()=>this.draw());
    this.canvas.addEventListener('pointerdown',e=>{{this.dragging=true;this.last=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);}});
    this.canvas.addEventListener('pointermove',e=>{{if(!this.dragging)return;const dx=e.clientX-this.last[0],dy=e.clientY-this.last[1];this.yaw+=dx*.009;this.pitch=Math.max(-1.35,Math.min(1.35,this.pitch+dy*.009));this.last=[e.clientX,e.clientY];this.draw();}});
    this.canvas.addEventListener('pointerup',()=>{{this.dragging=false;}});
    this.canvas.addEventListener('pointercancel',()=>{{this.dragging=false;}});
    this.canvas.addEventListener('dblclick',()=>{{this.yaw=-0.7;this.pitch=.35;this.draw();}});
    new ResizeObserver(()=>this.draw()).observe(this.canvas);
    this.populateLayers();
    this.draw();
  }}
  populateLayers() {{
    const model=this.modelSelect.value;
    const layers=Object.keys(GEOMETRY[this.site][model]).map(Number).sort((a,b)=>a-b);
    const previousRaw=this.layerSelect.value;
    const previous=previousRaw===''?Number.NaN:Number(previousRaw);
    const preferred=Number.isFinite(previous)&&layers.includes(previous)?previous:this.defaults[model];
    this.layerSelect.replaceChildren(...layers.map(layer=>{{const option=document.createElement('option');option.value=String(layer);option.textContent=`L${{layer}}`;return option;}}));
    this.layerSelect.value=String(layers.includes(preferred)?preferred:layers[0]);
  }}
  rotate(p) {{
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const x=cy*p[0]+sy*p[2], z=-sy*p[0]+cy*p[2];
    return [x,cp*p[1]-sp*z,sp*p[1]+cp*z];
  }}
  draw() {{
    const model=this.modelSelect.value;
    const layer=this.layerSelect.value;
    const payload=GEOMETRY[this.site][model][layer];
    if(!payload)return;
    const rows=payload.rows;
    const rect=this.canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
    if(rect.width<20)return;
    this.canvas.width=Math.round(rect.width*dpr);this.canvas.height=Math.round(rect.height*dpr);
    const ctx=this.ctx;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
    const raw=rows.map(r=>[+r[6],+r[7],+r[8]]),mins=[0,1,2].map(k=>Math.min(...raw.map(p=>p[k]))),maxs=[0,1,2].map(k=>Math.max(...raw.map(p=>p[k])));
    const center=mins.map((v,k)=>(v+maxs[k])/2),span=Math.max(...maxs.map((v,k)=>v-mins[k]))||1;
    const norm=p=>p.map((v,k)=>(v-center[k])/span*2);
    const scale=Math.min(rect.width*.34,rect.height*.34),ox=rect.width*.50,oy=rect.height*.48;
    const project=p=>{{const r=this.rotate(p),persp=1/(1+r[2]*.13);return [ox+r[0]*scale*persp,oy-r[1]*scale*persp,r[2]];}};
    const axes=[[[0,0,0],[1.15,0,0],'PC1'],[[0,0,0],[0,1.15,0],'PC2'],[[0,0,0],[0,0,1.15],'PC3']];
    ctx.lineWidth=1.2;ctx.strokeStyle='#9aa5b4';ctx.fillStyle='#536074';ctx.font='12px system-ui';
    axes.forEach(a=>{{const p0=project(a[0]),p1=project(a[1]);ctx.beginPath();ctx.moveTo(p0[0],p0[1]);ctx.lineTo(p1[0],p1[1]);ctx.stroke();ctx.fillText(a[2],p1[0]+5,p1[1]-5);}});
    const points=rows.map(r=>{{const label=this.site==='prompt'?+r[5]:+r[5];const p=project(norm([+r[6],+r[7],+r[8]]));return {{p,label}};}}).sort((a,b)=>a.p[2]-b.p[2]);
    points.forEach(o=>{{ctx.globalAlpha=.42;ctx.fillStyle=COLORS[o.label-1];ctx.beginPath();ctx.arc(o.p[0],o.p[1],2.8,0,Math.PI*2);ctx.fill();}});ctx.globalAlpha=1;
    const grouped=new Map();rows.forEach(r=>{{const label=+r[5];if(!grouped.has(label))grouped.set(label,[]);grouped.get(label).push([+r[6],+r[7],+r[8]]);}});
    const centroids=[...grouped.entries()].sort((a,b)=>a[0]-b[0]).map(([label,ps])=>{{const mean=[0,1,2].map(k=>ps.reduce((s,p)=>s+p[k],0)/ps.length);return {{label,p:project(norm(mean))}};}});
    ctx.strokeStyle='#26364d';ctx.lineWidth=2;ctx.beginPath();centroids.forEach((o,i)=>{{if(i===0)ctx.moveTo(o.p[0],o.p[1]);else ctx.lineTo(o.p[0],o.p[1]);}});ctx.stroke();
    centroids.sort((a,b)=>a.p[2]-b.p[2]).forEach(o=>{{ctx.fillStyle=COLORS[o.label-1];ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(o.p[0],o.p[1],8,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font='600 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(o.label),o.p[0],o.p[1]+.5);}});
    ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.font='12px system-ui';
    const legendY=rect.height-24,step=Math.min(42,(rect.width-28)/10);for(let n=1;n<=10;n++){{const x=14+(n-1)*step;ctx.fillStyle=COLORS[n-1];ctx.beginPath();ctx.arc(x,legendY,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#4b5565';ctx.fillText(String(n),x+8,legendY+4);}}
    ctx.fillStyle='#4b5565';ctx.font='600 12px system-ui';ctx.fillText(`${{model}} · L${{payload.layer}} · ${{rows.length}} states`,14,22);
  }}
}}
new PointCloud3D('prompt-canvas','prompt-model','prompt-layer','prompt',{{'Qwen3-8B':8,'Gemma4-E4B':9}});
new PointCloud3D('answer-canvas','answer-model','answer-layer','answer',{{'Qwen3-8B':28,'Gemma4-E4B':37}});
</script>
</body>
</html>
"""

    # Keep the source template convenient to maintain while enforcing the
    # publication-facing, mechanism-first narrative in the rendered report.
    section_order = (
        "summary",
        "baseline",
        "representation",
        "formation",
        "retrieval",
        "write",
        "ov-write",
        "ledger",
        "extension-audit",
        "limitations",
        "appendix",
    )
    main_start = html_doc.index("<main>") + len("<main>")
    main_end = html_doc.index("</main>", main_start)
    main_html = html_doc[main_start:main_end]
    section_matches = list(
        re.finditer(r'<section id="([^"]+)">.*?</section>', main_html, re.DOTALL)
    )
    rendered_sections = {match.group(1): match.group(0) for match in section_matches}
    if tuple(rendered_sections) != (
        "baseline",
        "summary",
        "representation",
        "formation",
        "retrieval",
        "write",
        "ov-write",
        "ledger",
        "extension-audit",
        "limitations",
        "appendix",
    ):
        raise RuntimeError(
            "Unexpected report sections before mechanism-first reordering: "
            f"{tuple(rendered_sections)}"
        )
    reordered_main = "\n\n" + "\n\n".join(
        rendered_sections[section_id] for section_id in section_order
    ) + "\n"
    html_doc = html_doc[:main_start] + reordered_main + html_doc[main_end:]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "NiaH_Non-thinking_report.html",
    )
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
