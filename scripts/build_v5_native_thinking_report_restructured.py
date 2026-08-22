#!/usr/bin/env python3
"""Build the Native-thinking mechanism report.

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

import numpy as np

from build_v5_native_thinking_report_final import (
    answer_source_rerouting_svg,
    attention_example_switcher,
    head_map_svg,
    load_attention_examples,
    load_representation,
    load_token_ablation_evidence,
    token_source_ablation_svg,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
SHORT = {"Qwen3-8B": "Qwen", "Gemma4-E4B": "Gemma"}
COLORS = {"Qwen3-8B": "#0f766e", "Gemma4-E4B": "#7c3aed"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_all_layer_native_geometry(
    comparison_report: Path,
    running_defaults: Mapping[str, int],
    final_defaults: Mapping[str, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract all-layer native PCA3 coordinates from the audited geometry report.

    That report fits StandardScaler + PCA3 separately at every layer using only
    discovery states.  We retain confirmation rows only for the mechanism report.
    """

    prefix = "const DUAL="
    dual: dict[str, Any] | None = None
    with comparison_report.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(prefix):
                require(line.rstrip().endswith(";"), "Malformed DUAL geometry payload")
                dual = json.loads(line.rstrip()[len(prefix) : -1])
                break
    require(dual is not None, "Geometry comparison report lacks DUAL payload")

    result: dict[str, dict[str, dict[str, Any]]] = {
        "running": {},
        "final": {},
    }
    for model in MODELS:
        for endpoint, panel_name, defaults in (
            ("running", "running_native", running_defaults),
            ("final", "final_native", final_defaults),
        ):
            panel = dual[model]["panels"][panel_name]
            layer_payload: dict[str, Any] = {}
            discovery_rows = confirmation_rows = None
            for layer_text, block in panel["coordinates"].items():
                source_rows = list(block["points"])
                current_discovery = sum(row[0] == "discovery" for row in source_rows)
                current_confirmation = sum(row[0] == "confirmation" for row in source_rows)
                if discovery_rows is None:
                    discovery_rows = current_discovery
                    confirmation_rows = current_confirmation
                require(
                    (current_discovery, current_confirmation)
                    == (discovery_rows, confirmation_rows),
                    f"{model}/{endpoint}: layerwise state support changed",
                )
                confirmation = [row for row in source_rows if row[0] == "confirmation"]
                rows = [
                    [
                        int(row[1]),
                        int(row[2] if endpoint == "running" else row[6]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    ]
                    for row in confirmation
                ]
                require(rows, f"{model}/{endpoint}/L{layer_text}: no confirmation rows")
                layer_payload[str(int(layer_text))] = {
                    "evr": [float(value) for value in block["evr"]],
                    "rows": rows,
                }
            layers = sorted(int(layer) for layer in layer_payload)
            default_layer = int(defaults[model])
            require(
                default_layer in layers,
                f"{model}/{endpoint}: frozen display L{default_layer} unavailable",
            )
            result[endpoint][model] = {
                "default_layer": default_layer,
                "layers": layer_payload,
                "discovery_rows": int(discovery_rows or 0),
                "confirmation_rows": int(confirmation_rows or 0),
                "token_site": str(panel["token_site"]),
            }
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def inline_standalone_svg(path: Path) -> str:
    """Embed a generated SVG and make its namespace explicit for HTML delivery."""
    markup = path.read_text(encoding="utf-8")
    if not markup.lstrip().startswith("<svg"):
        raise ValueError(f"Expected SVG root in {path}")
    if 'xmlns="http://www.w3.org/2000/svg"' not in markup[:512]:
        markup = markup.replace(
            "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1
        )
    return markup


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


def load_p0_head_score_rows(
    atlas_root: Path, model: str, *, grammar: str = "all"
) -> list[dict[str, str]]:
    """Adapt the frozen P0 ranking CSV to the report's layer-by-head renderer."""

    rows = [
        row
        for row in read_csv(atlas_root / "p0_targeted_retrieval_head_scores.csv")
        if row["model_label"] == model and row["grammar"] == grammar
    ]
    require(rows, f"Missing P0 head scores for {model}/{grammar}")
    return [
        {
            **row,
            "discovery_rank": row["rank"],
            "discovery_selection_value": row["score"],
        }
        for row in rows
    ]


def load_layerwise_representation(
    causal_root: Path, dual_root: Path
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Load frozen held-out probe curves without performing layer selection."""

    running_rows = read_csv(causal_root / "site_layer_candidates.csv")
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    for model in MODELS:
        running = [
            row
            for row in running_rows
            if row["model_label"] == model and row["site_kind"] == "item_end"
        ]
        final = [
            row
            for row in read_csv(
                dual_root
                / model
                / "pca16_whiten"
                / "final_count_candidate_metrics.csv"
            )
            if row["model_label"] == model
            and row["mode"] == "native_thinking"
            and row["endpoint"] == "final_count"
        ]
        running.sort(key=lambda row: int(row["layer"]))
        final.sort(key=lambda row: int(row["layer"]))
        require(running and final, f"Missing layerwise representation rows for {model}")
        result[model] = {"running": running, "final": final}
    return result


def layerwise_representation_svg(
    curves: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
    selected: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    """Four-panel probe plot aligned with the Non-thinking report grammar."""

    width, height = 1040, 620
    colors = {"logistic": "#0f766e", "ncc": "#d97706"}
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="layer-probe-title layer-probe-desc">',
        '<title id="layer-probe-title">Layerwise Native-thinking count decodability</title>',
        '<desc id="layer-probe-desc">Four panels show held-out balanced accuracy by decoder layer for running item-end and final answer-query states.</desc>',
    ]
    for row_index, endpoint in enumerate(("running", "final")):
        for col_index, model in enumerate(MODELS):
            rows = curves[model][endpoint]
            ox, oy = 58 + col_index * 500, 48 + row_index * 292
            plot_w, plot_h = 416, 190
            max_layer = max(int(row["layer"]) for row in rows)

            def sx(layer: int) -> float:
                return ox + layer / max(max_layer, 1) * plot_w

            def sy(value: float) -> float:
                return oy + (1.0 - value) * plot_h

            title = (
                "running commit · item_end"
                if endpoint == "running"
                else "final count · answer query"
            )
            parts.append(
                f'<text x="{ox}" y="{oy-18}" class="heat-title">'
                f'{esc(SHORT[model])} · {esc(title)}</text>'
            )
            parts.append(
                f'<rect x="{ox}" y="{oy}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>'
            )
            for tick in (0.1, 0.4, 0.7, 1.0):
                y = sy(tick)
                parts.append(
                    f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+plot_w}" y2="{y:.1f}" class="grid"/>'
                )
                parts.append(
                    f'<text x="{ox-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>'
                )
            chance_y = sy(0.1)
            parts.append(
                f'<line x1="{ox}" y1="{chance_y:.1f}" x2="{ox+plot_w}" y2="{chance_y:.1f}" '
                'stroke="#667085" stroke-dasharray="5 4"/>'
            )
            for metric, key in (
                ("logistic", "confirmation_logistic_balanced_accuracy"),
                ("ncc", "confirmation_ncc_balanced_accuracy"),
            ):
                points = " ".join(
                    f'{sx(int(row["layer"])):.1f},{sy(float(row[key])):.1f}'
                    for row in rows
                )
                parts.append(
                    f'<polyline points="{points}" class="series-line" stroke="{colors[metric]}"/>'
                )
            selected_layer = int(selected[model][endpoint]["layer"])
            selected_value = float(
                selected[model][endpoint]["confirmation_logistic_balanced_accuracy"]
            )
            parts.append(
                f'<circle cx="{sx(selected_layer):.1f}" cy="{sy(selected_value):.1f}" r="6" '
                'fill="#fff" stroke="#0f766e" stroke-width="3"/>'
            )
            parts.append(
                f'<text x="{sx(selected_layer)+8:.1f}" y="{sy(selected_value)-8:.1f}" class="chart-value">'
                f'L{selected_layer} · {selected_value:.2f}</text>'
            )
            for layer_tick in sorted({0, max_layer // 2, max_layer}):
                parts.append(
                    f'<text x="{sx(layer_tick):.1f}" y="{oy+plot_h+19}" text-anchor="middle" class="tick">{layer_tick}</text>'
                )
            if col_index == 0:
                parts.append(
                    f'<text transform="translate({ox-42} {oy+plot_h/2}) rotate(-90)" '
                    'text-anchor="middle" class="axis-label">confirmation balanced accuracy</text>'
                )
            parts.append(
                f'<text x="{ox+plot_w/2}" y="{oy+plot_h+38}" text-anchor="middle" class="axis-label">zero-based post-block layer</text>'
            )
    parts.extend(
        [
            '<line x1="372" y1="603" x2="398" y2="603" stroke="#0f766e" stroke-width="3"/>',
            '<text x="405" y="607" class="legend-label">L2 logistic</text>',
            '<line x1="500" y1="603" x2="526" y2="603" stroke="#d97706" stroke-width="3"/>',
            '<text x="533" y="607" class="legend-label">nearest centroid</text>',
            '<line x1="678" y1="603" x2="704" y2="603" stroke="#667085" stroke-dasharray="5 4"/>',
            '<text x="711" y="607" class="legend-label">10-class chance</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def load_running_manifold_points(
    band_root: Path,
) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = {}
    for model in MODELS:
        rows = read_csv(band_root / model / "confirmation_points.csv")
        points = [
            {
                "seed": float(row["seed"]),
                "label": float(row["occurrence"]),
                "x": float(row["pc1"]),
                "y": float(row["pc2"]),
                "z": float(row["pc3"]),
            }
            for row in rows
        ]
        require(points, f"Missing running-manifold points for {model}")
        result[model] = points
    return result


def load_final_manifold_points(
    capture_root: Path,
    representation: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict[str, Any]]]:
    """Fit PCA on discovery answer-query states and project confirmation only."""

    result: dict[str, list[dict[str, float]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        root = capture_root / "final" / model
        index = read_jsonl(root / "capture_index.jsonl")
        layer = int(representation[model]["final"]["layer"])
        discovery_states: list[np.ndarray] = []
        discovery_labels: list[float] = []
        confirmation_states: list[np.ndarray] = []
        confirmation_labels: list[float] = []
        confirmation_seeds: list[float] = []
        for row in index:
            archive = np.load(root / str(row["states_path"]))
            layers = archive["layer_indices"].astype(int)
            matches = np.flatnonzero(layers == layer)
            require(len(matches) == 1, f"{model} final capture misses L{layer}")
            states = archive["site_states"]
            require(states.shape[0] == 1, f"{model} final capture has multiple answer sites")
            vector = states[0, int(matches[0])].astype(np.float32)
            label = float(row["gold_count"])
            if row["split"] == "discovery":
                discovery_states.append(vector)
                discovery_labels.append(label)
            elif row["split"] == "confirmation":
                confirmation_states.append(vector)
                confirmation_labels.append(label)
                confirmation_seeds.append(float(row["seed"]))
        require(
            len(discovery_states) == 200 and len(confirmation_states) == 100,
            f"{model} final PCA split changed",
        )
        train = np.stack(discovery_states)
        test = np.stack(confirmation_states)
        center = train.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(train - center, full_matrices=False)
        components = vt[:3]
        train_coords = (train - center) @ components.T
        test_coords = (test - center) @ components.T
        labels = np.asarray(discovery_labels, dtype=np.float32)
        for axis in range(3):
            corr = np.corrcoef(train_coords[:, axis], labels)[0, 1]
            if np.isfinite(corr) and corr < 0:
                test_coords[:, axis] *= -1
        result[model] = [
            {
                "seed": seed,
                "label": label,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
            }
            for seed, label, point in zip(
                confirmation_seeds, confirmation_labels, test_coords
            )
        ]
        variance_ratio = (_s[:3] ** 2) / max(float(np.sum(_s ** 2)), 1e-12)
        diagnostics[model] = {
            "discovery_rows": len(discovery_states),
            "confirmation_rows": len(confirmation_states),
            "explained_variance_ratio": [float(value) for value in variance_ratio],
            "explained_variance_ratio_sum": float(np.sum(variance_ratio)),
        }
    return result, diagnostics


def manifold_payload(
    running: Mapping[str, Sequence[Mapping[str, float]]],
    final: Mapping[str, Sequence[Mapping[str, float]]],
    running_layers: Mapping[str, int],
    final_layers: Mapping[str, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Serialize the two frozen display manifolds for the in-report 3D viewer."""

    payload: dict[str, dict[str, dict[str, Any]]] = {}
    for endpoint, source, layers in (
        ("running", running, running_layers),
        ("final", final, final_layers),
    ):
        payload[endpoint] = {}
        for model in MODELS:
            rows = [
                [
                    int(point["seed"]),
                    int(round(point["label"])),
                    round(float(point["x"]), 6),
                    round(float(point["y"]), 6),
                    round(float(point["z"]), 6),
                ]
                for point in source[model]
            ]
            require(rows, f"Missing {endpoint} 3D manifold points for {model}")
            payload[endpoint][model] = {
                "layer": int(layers[model]),
                "rows": rows,
            }
    return payload


def representation_manifold_svg(
    running: Mapping[str, Sequence[Mapping[str, float]]],
    final: Mapping[str, Sequence[Mapping[str, float]]],
) -> str:
    """Four-panel confirmation PCA plot with discovery-frozen axes."""

    palette = (
        "#0b4f6c", "#176b87", "#228b8d", "#3aa17e", "#66b56b",
        "#9abe55", "#d0bb42", "#e69b35", "#df6f32", "#c9413a",
    )
    width, height = 1040, 690
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="manifold-title manifold-desc">',
        '<title id="manifold-title">Native-thinking running and final count manifolds</title>',
        '<desc id="manifold-desc">Confirmation hidden states projected onto discovery-fitted first two principal components, colored by occurrence or final count.</desc>',
    ]
    panels = (("running", running), ("final", final))
    for row_index, (endpoint, source) in enumerate(panels):
        for col_index, model in enumerate(MODELS):
            points = list(source[model])
            ox, oy = 58 + col_index * 500, 48 + row_index * 292
            plot_w, plot_h = 416, 208
            xs = np.asarray([point["x"] for point in points])
            ys = np.asarray([point["y"] for point in points])
            x_pad = max(float(np.ptp(xs)) * 0.06, 1e-6)
            y_pad = max(float(np.ptp(ys)) * 0.06, 1e-6)
            x_min, x_max = float(xs.min() - x_pad), float(xs.max() + x_pad)
            y_min, y_max = float(ys.min() - y_pad), float(ys.max() + y_pad)

            def sx(value: float) -> float:
                return ox + (value - x_min) / max(x_max - x_min, 1e-9) * plot_w

            def sy(value: float) -> float:
                return oy + (y_max - value) / max(y_max - y_min, 1e-9) * plot_h

            layer_note = "display PCA" if endpoint == "running" else "answer-query PCA"
            parts.append(
                f'<text x="{ox}" y="{oy-18}" class="heat-title">{esc(SHORT[model])} · '
                f'{"running occurrence" if endpoint == "running" else "final gold count"} · {layer_note}</text>'
            )
            parts.append(
                f'<rect x="{ox}" y="{oy}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>'
            )
            for point in points:
                label = max(1, min(10, int(round(point["label"]))))
                parts.append(
                    f'<circle cx="{sx(point["x"]):.1f}" cy="{sy(point["y"]):.1f}" r="3" '
                    f'fill="{palette[label-1]}" opacity=".32"/>'
                )
            centroids: list[tuple[int, float, float]] = []
            for label in range(1, 11):
                group = [point for point in points if int(round(point["label"])) == label]
                if not group:
                    continue
                cx = float(np.mean([point["x"] for point in group]))
                cy = float(np.mean([point["y"] for point in group]))
                centroids.append((label, cx, cy))
            polyline = " ".join(f"{sx(cx):.1f},{sy(cy):.1f}" for _, cx, cy in centroids)
            parts.append(
                f'<polyline points="{polyline}" fill="none" stroke="#344054" stroke-width="1.8" opacity=".7"/>'
            )
            for label, cx, cy in centroids:
                parts.append(
                    f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="7" fill="{palette[label-1]}" '
                    'stroke="#fff" stroke-width="1.5"/>'
                )
                parts.append(
                    f'<text x="{sx(cx):.1f}" y="{sy(cy)+3.2:.1f}" text-anchor="middle" '
                    'font-size="7.5" font-weight="800" fill="#fff">'
                    f'{label}</text>'
                )
            parts.append(
                f'<text x="{ox+plot_w/2}" y="{oy+plot_h+30}" text-anchor="middle" class="axis-label">PC1 (arbitrary PCA units)</text>'
            )
            parts.append(
                f'<text transform="translate({ox-38} {oy+plot_h/2}) rotate(-90)" text-anchor="middle" '
                'class="axis-label">PC2 (arbitrary PCA units)</text>'
            )
    for label in range(1, 11):
        x = 245 + (label - 1) * 58
        parts.append(f'<circle cx="{x}" cy="656" r="6" fill="{palette[label-1]}"/>')
        parts.append(f'<text x="{x+10}" y="660" class="legend-label">{label}</text>')
    parts.append('<text x="190" y="660" text-anchor="end" class="legend-label">occurrence / count</text>')
    parts.append("</svg>")
    return "".join(parts)


def split_gemma_current_head_map_svg(rows: Sequence[Mapping[str, str]]) -> str:
    """Render Gemma's tall 42x8 atlas as two large, legible layer panels."""

    layers = max(int(row["layer"]) for row in rows) + 1
    heads = max(int(row["head"]) for row in rows) + 1
    require((layers, heads) == (42, 8), "Unexpected Gemma layer/head geometry")
    by_head = {
        (int(row["layer"]), int(row["head"])): row for row in rows
    }
    scores = [max(0.0, float(row["discovery_selection_value"])) for row in rows]
    cap = max(float(np.quantile(scores, 0.99)), 1e-12)

    def color(value: float) -> str:
        t = max(0.0, min(1.0, value)) ** 0.5
        low, high = (241, 245, 249), (15, 118, 110)
        rgb = tuple(round(a + (b - a) * t) for a, b in zip(low, high))
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    width, height = 1040, 485
    x_starts = (82, 495)
    layer_ranges = ((0, 21), (21, 42))
    y0, cw, ch = 72, 38, 16
    parts = [
        f'<svg class="head-map gemma-split-map" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Gemma4-E4B exact-P0 current Top-6 layer by head attention atlas">',
        '<title>Gemma4-E4B exact-P0 target-attention atlas, current Top-6</title>',
    ]
    for (layer_start, layer_end), x0 in zip(layer_ranges, x_starts):
        parts.append(
            f'<text x="{x0}" y="28" class="heat-title">Layers {layer_start}–{layer_end-1}</text>'
        )
        for head in range(heads):
            parts.append(
                f'<text x="{x0 + (head + .5) * cw:.1f}" y="58" '
                f'text-anchor="middle" class="heat-x">H{head}</text>'
            )
        for local_layer, layer in enumerate(range(layer_start, layer_end)):
            y = y0 + local_layer * ch
            parts.append(
                f'<text x="{x0-10}" y="{y+12}" text-anchor="end" '
                f'class="heat-row">L{layer}</text>'
            )
            for head in range(heads):
                row = by_head[(layer, head)]
                score = max(0.0, float(row["discovery_selection_value"]))
                rank = int(row["discovery_rank"])
                selected = rank <= 6
                parts.append(
                    f'<rect x="{x0 + head*cw}" y="{y}" width="{cw}" height="{ch}" '
                    f'fill="{color(min(score/cap, 1.0))}" '
                    f'stroke="{"#b42318" if selected else "#ffffff"}" '
                    f'stroke-width="{1.35 if selected else .35}">'
                    f'<title>L{layer}H{head}; seed-event mean target-record attention mass='
                    f'{score:.6g}; frozen rank={rank}; selected={str(selected).lower()}</title></rect>'
                )
        parts.append(
            f'<text x="{x0 + heads*cw/2}" y="{height-42}" text-anchor="middle" '
            'class="axis-label">Attention head h</text>'
        )
    legend_x, legend_y = 900, 86
    for step in range(40):
        t = step / 39
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y + (39-step)*6}" width="20" '
            f'height="6" fill="{color(t)}"/>'
        )
    parts.extend(
        [
            f'<text x="{legend_x+28}" y="{legend_y+10}" class="heat-x">≥99th pct.</text>',
            f'<text x="{legend_x+28}" y="{legend_y+236}" class="heat-x">0</text>',
            '<text transform="translate(23 240) rotate(-90)" text-anchor="middle" '
            'class="axis-label">Transformer layer ℓ</text>',
            '<rect x="884" y="372" width="18" height="14" fill="#fff" stroke="#b42318" stroke-width="1.5"/>',
            '<text x="912" y="384" class="heat-x">frozen Top-6</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def point_cloud_script(payload: Mapping[str, Any]) -> str:
    """Return a dependency-free interactive PC1-PC3 canvas viewer."""

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "const NATIVE_GEOMETRY=" + encoded + r""";
const COUNT_COLORS=['#0b4f6c','#176b87','#228b8d','#3aa17e','#66b56b','#9abe55','#d0bb42','#e69b35','#df6f32','#c9413a'];
const NATIVE_CLOUDS=[];
class NativePointCloud3D {
  constructor(canvasId, modelId, layerId, statsId, endpoint) {
    this.canvas=document.getElementById(canvasId);
    this.ctx=this.canvas.getContext('2d');
    this.model=document.getElementById(modelId);
    this.layer=document.getElementById(layerId);
    this.stats=document.getElementById(statsId);
    this.endpoint=endpoint;
    this.yaw=-0.58;
    this.pitch=0.34;
    this.dragging=false;
    this.last=null;
    this.model.addEventListener('change',()=>this.setModel());
    this.layer.addEventListener('change',()=>this.setLayer());
    this.canvas.addEventListener('pointerdown',(event)=>this.pointerDown(event));
    this.canvas.addEventListener('pointermove',(event)=>this.pointerMove(event));
    this.canvas.addEventListener('pointerup',(event)=>this.pointerUp(event));
    this.canvas.addEventListener('pointercancel',(event)=>this.pointerUp(event));
    this.canvas.addEventListener('dblclick',()=>this.resetView());
    new ResizeObserver(()=>this.resize()).observe(this.canvas);
    NATIVE_CLOUDS.push(this);
    this.setModel();
  }
  setModel() {
    this.modelData=NATIVE_GEOMETRY[this.endpoint][this.model.value];
    this.layer.innerHTML='';
    Object.keys(this.modelData.layers).map(Number).sort((a,b)=>a-b).forEach(layer=>{
      const option=document.createElement('option');
      option.value=String(layer);
      option.textContent='L'+layer+(layer===this.modelData.default_layer?' · frozen default':'');
      this.layer.appendChild(option);
    });
    this.layer.value=String(this.modelData.default_layer);
    this.setLayer();
  }
  setLayer() {
    this.data=this.modelData.layers[this.layer.value];
    this.rows=this.data.rows;
    this.prepare();
    const evr=100*this.data.evr.reduce((sum,value)=>sum+value,0);
    this.stats.textContent=this.modelData.token_site+' · L'+this.layer.value+' · '+this.modelData.discovery_rows+' discovery states fit StandardScaler/PCA3 · '+this.rows.length+' confirmation states shown · EVR₁₋₃ '+evr.toFixed(1)+'%';
    this.draw();
  }
  resetView() { NATIVE_CLOUDS.forEach(cloud=>{cloud.yaw=-0.58;cloud.pitch=0.34;cloud.draw();}); }
  pointerDown(event) {
    this.dragging=true; this.last=[event.clientX,event.clientY];
    this.canvas.setPointerCapture(event.pointerId);
  }
  pointerMove(event) {
    if(!this.dragging) return;
    const dx=event.clientX-this.last[0], dy=event.clientY-this.last[1];
    this.last=[event.clientX,event.clientY];
    NATIVE_CLOUDS.forEach(cloud=>{cloud.yaw+=dx*0.009;cloud.pitch=Math.max(-1.35,Math.min(1.35,cloud.pitch+dy*0.009));cloud.draw();});
  }
  pointerUp(event) {
    this.dragging=false; this.last=null;
    if(this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
  }
  resize() {
    const rect=this.canvas.getBoundingClientRect();
    const dpr=Math.min(window.devicePixelRatio||1,2);
    const width=Math.max(320,Math.round(rect.width));
    const height=Math.max(400,Math.round(rect.height));
    if(this.canvas.width!==Math.round(width*dpr)||this.canvas.height!==Math.round(height*dpr)) {
      this.canvas.width=Math.round(width*dpr); this.canvas.height=Math.round(height*dpr);
    }
    this.ctx.setTransform(dpr,0,0,dpr,0,0);
    this.cssWidth=width; this.cssHeight=height;
    this.draw();
  }
  prepare() {
    const coords=[0,1,2].map(axis=>this.rows.map(row=>row[axis+2]));
    this.centers=coords.map(values=>(Math.min(...values)+Math.max(...values))/2);
    this.scales=coords.map(values=>Math.max(Math.max(...values)-Math.min(...values),1e-8)/2);
    this.points=this.rows.map(row=>({seed:row[0],label:row[1],v:[0,1,2].map(axis=>(row[axis+2]-this.centers[axis])/this.scales[axis])}));
    this.centroids=[];
    for(let label=1;label<=10;label++) {
      const group=this.points.filter(point=>point.label===label);
      if(!group.length) continue;
      this.centroids.push({label:label,v:[0,1,2].map(axis=>group.reduce((sum,point)=>sum+point.v[axis],0)/group.length)});
    }
  }
  rotate(v) {
    const cy=Math.cos(this.yaw), sy=Math.sin(this.yaw), cp=Math.cos(this.pitch), sp=Math.sin(this.pitch);
    const x=cy*v[0]+sy*v[2], z=-sy*v[0]+cy*v[2];
    return [x,cp*v[1]-sp*z,sp*v[1]+cp*z];
  }
  project(v) {
    const r=this.rotate(v), depth=3.4-r[2]*0.34, scale=Math.min(this.cssWidth,510)*0.78/depth;
    return {x:this.cssWidth*0.5+r[0]*scale,y:this.cssHeight*0.45-r[1]*scale,z:r[2],s:scale};
  }
  line(a,b,color,width,dash) {
    const pa=this.project(a),pb=this.project(b),ctx=this.ctx;
    ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash||[]);
    ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();ctx.restore();
  }
  draw() {
    if(!this.ctx||!this.rows||!this.cssWidth) return;
    const ctx=this.ctx,w=this.cssWidth,h=this.cssHeight;
    ctx.clearRect(0,0,w,h);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,w,h);
    const axes=[[[0,0,0],[1.18,0,0],'PC1'],[[0,0,0],[0,1.18,0],'PC2'],[[0,0,0],[0,0,1.18],'PC3']];
    axes.forEach(axis=>{this.line(axis[0],axis[1],'#98a2b3',1.25,[]);const p=this.project(axis[1]);ctx.fillStyle='#475467';ctx.font='700 12px system-ui';ctx.fillText(axis[2],p.x+5,p.y-5);});
    const projected=this.points.map(point=>Object.assign({},point,this.project(point.v))).sort((a,b)=>a.z-b.z);
    projected.forEach(point=>{ctx.beginPath();ctx.arc(point.x,point.y,3.1,0,Math.PI*2);ctx.fillStyle=COUNT_COLORS[point.label-1]+'70';ctx.fill();});
    const centroids=this.centroids.map(point=>Object.assign({},point,this.project(point.v))).sort((a,b)=>a.label-b.label);
    ctx.save();ctx.strokeStyle='#344054';ctx.lineWidth=1.7;ctx.globalAlpha=.75;ctx.beginPath();
    centroids.forEach((point,index)=>{if(index===0)ctx.moveTo(point.x,point.y);else ctx.lineTo(point.x,point.y);});ctx.stroke();ctx.restore();
    centroids.slice().sort((a,b)=>a.z-b.z).forEach(point=>{ctx.beginPath();ctx.arc(point.x,point.y,10.5,0,Math.PI*2);ctx.fillStyle=COUNT_COLORS[point.label-1];ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.8;ctx.stroke();ctx.fillStyle='#fff';ctx.font='800 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(point.label),point.x,point.y+.4);});
    ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.font='12px system-ui';
    const legendWidth=Math.min(620,w-36), startX=(w-legendWidth)/2, step=legendWidth/10;
    for(let label=1;label<=10;label++){const x=startX+(label-.5)*step;ctx.fillStyle=COUNT_COLORS[label-1];ctx.beginPath();ctx.arc(x,h-24,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#475467';ctx.fillText(String(label),x+9,h-20);}
    ctx.fillStyle='#667085';ctx.font='11px system-ui';ctx.fillText(this.points.length+' confirmation states · drag to rotate · double-click to reset',16,22);
  }
}
new NativePointCloud3D('native-running-canvas','native-geometry-model','native-running-layer','native-running-stats','running');
new NativePointCloud3D('native-final-canvas','native-geometry-model','native-final-layer','native-final-stats','final');
document.getElementById('native-geometry-reset').addEventListener('click',()=>NATIVE_CLOUDS[0].resetView());
"""


def effect_small_multiples_svg(
    title: str,
    panels: Sequence[
        tuple[str, str, Sequence[tuple[str, Mapping[str, Any]]], float]
    ],
) -> str:
    """Draw independently scaled mean/95%-CI panels for heterogeneous estimands."""

    require(1 <= len(panels) <= 4, "Small-multiple figure supports one to four panels")
    width, height = 1040, 600
    panel_w, panel_h = 472, 238
    origins = ((58, 68), (550, 68), (58, 338), (550, 338))
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
    ]
    for panel_index, (panel_title, unit, rows, floor) in enumerate(panels):
        x0, y0 = origins[panel_index]
        values = [
            abs(float(row[key]))
            for _, row in rows
            for key in ("mean_effect", "ci_low", "ci_high")
        ]
        limit = max(max(values, default=0.0) * 1.16, float(floor))
        plot_left, plot_right = x0 + 130, x0 + panel_w - 18
        center = (plot_left + plot_right) / 2
        half = (plot_right - plot_left) / 2

        def sx(value: float) -> float:
            return center + max(-limit, min(limit, value)) / limit * half

        parts.extend(
            [
                f'<rect x="{x0}" y="{y0-30}" width="{panel_w}" height="{panel_h}" fill="#fbfcfe" stroke="#d0d5dd"/>',
                f'<text x="{x0+14}" y="{y0-7}" class="heat-title">{esc(panel_title)}</text>',
                f'<line x1="{center:.1f}" y1="{y0+12}" x2="{center:.1f}" y2="{y0+150}" stroke="#98a2b3" stroke-dasharray="4 4"/>',
            ]
        )
        for row_index, (label, row) in enumerate(rows):
            y = y0 + 35 + row_index * 42
            mean = float(row["mean_effect"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            gate = bool(row.get("gate_pass", row.get("positive_95pct_ci", low > 0)))
            point_color = "#0f766e" if gate else "#d97706"
            parts.extend(
                [
                    f'<text x="{plot_left-10}" y="{y+4}" text-anchor="end" class="chart-axis">{esc(label)}</text>',
                    f'<line x1="{sx(low):.1f}" y1="{y}" x2="{sx(high):.1f}" y2="{y}" stroke="#475467" stroke-width="2"/>',
                    f'<line x1="{sx(low):.1f}" y1="{y-5}" x2="{sx(low):.1f}" y2="{y+5}" stroke="#475467"/>',
                    f'<line x1="{sx(high):.1f}" y1="{y-5}" x2="{sx(high):.1f}" y2="{y+5}" stroke="#475467"/>',
                    f'<circle cx="{sx(mean):.1f}" cy="{y}" r="5" fill="{point_color}"><title>{esc(label)}: mean={mean:+.6g}; 95% CI [{low:+.6g}, {high:+.6g}]</title></circle>',
                    f'<text x="{plot_right}" y="{y-8}" text-anchor="end" class="chart-value">{mean:+.3g}</text>',
                ]
            )
        tick_y = y0 + 177
        parts.extend(
            [
                f'<text x="{plot_left}" y="{tick_y}" text-anchor="middle" class="chart-axis">−{limit:.3g}</text>',
                f'<text x="{center}" y="{tick_y}" text-anchor="middle" class="chart-axis">0</text>',
                f'<text x="{plot_right}" y="{tick_y}" text-anchor="middle" class="chart-axis">+{limit:.3g}</text>',
                f'<text x="{center}" y="{tick_y+20}" text-anchor="middle" class="axis-label">{esc(unit)} · panel-specific scale</text>',
            ]
        )
    parts.extend(
        [
            '<circle cx="390" cy="580" r="5" fill="#0f766e"/><text x="401" y="584" class="chart-axis">registered CI entirely supportive</text>',
            '<circle cx="650" cy="580" r="5" fill="#d97706"/><text x="661" y="584" class="chart-axis">gate not met / interval touches zero</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def failed_control_svg() -> str:
    """Explain why the first single-seed scrub was not an interpretable null."""

    return """<svg class="paper-chart" viewBox="0 0 1040 330" role="img" aria-label="V1 versus V2 context-scrub control coverage">
<title>Why the V1 single-seed scrub was a failed control</title>
<defs><marker id="audit-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#667085"/></marker></defs>
<text x="40" y="34" class="heat-title">V1 · parsed-item-only scrub</text><rect x="40" y="55" width="260" height="62" rx="5" fill="#eef2f6" stroke="#98a2b3"/><text x="170" y="82" text-anchor="middle" class="chain-title">Prompt records blanked</text><text x="170" y="103" text-anchor="middle" class="chain-sub">registered record spans</text><line x1="300" y1="86" x2="350" y2="86" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="355" y="55" width="260" height="62" rx="5" fill="#eef2f6" stroke="#98a2b3"/><text x="485" y="82" text-anchor="middle" class="chain-title">Parsed trace items blanked</text><text x="485" y="103" text-anchor="middle" class="chain-sub">parser-observed list only</text><line x1="615" y1="86" x2="665" y2="86" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="670" y="55" width="330" height="62" rx="5" fill="#fff3e0" stroke="#d97706" stroke-width="2"/><text x="835" y="82" text-anchor="middle" class="chain-title">Trace tail remained visible</text><text x="835" y="103" text-anchor="middle" class="chain-sub">possible final-count leakage → null uninterpretable</text>
<text x="40" y="181" class="heat-title">V2 · complete source scrub</text><rect x="40" y="202" width="260" height="62" rx="5" fill="#e8f5f2" stroke="#0f766e"/><text x="170" y="229" text-anchor="middle" class="chain-title">Prompt records blanked</text><text x="170" y="250" text-anchor="middle" class="chain-sub">same-length ordinary tokens</text><line x1="300" y1="233" x2="350" y2="233" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="355" y="202" width="260" height="62" rx="5" fill="#e8f5f2" stroke="#0f766e"/><text x="485" y="229" text-anchor="middle" class="chain-title">Entire trace context blanked</text><text x="485" y="250" text-anchor="middle" class="chain-sub">items + tail + residual text</text><line x1="615" y1="233" x2="665" y2="233" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="670" y="202" width="330" height="62" rx="5" fill="#e8f5f2" stroke="#0f766e" stroke-width="2"/><text x="835" y="229" text-anchor="middle" class="chain-title">Uninformative baseline verified</text><text x="835" y="250" text-anchor="middle" class="chain-sub">single-item restoration can now be interpreted</text>
<text x="520" y="307" text-anchor="middle" class="axis-label">Only V2 enters the report's descriptive single-seed conclusion</text></svg>"""


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
        ("Next query", "targeted-head routing", "strong", "qualified"),
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
            color = {"strong": "#0f766e", "qualified": "#0f766e", "conditional": "#7c3aed"}[status]
            label = {"strong": "confirmed", "qualified": "confirmed†", "conditional": "controlled only"}[status]
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
    layerwise_representation = load_layerwise_representation(
        args.representation_root, args.dual_endpoint_root
    )
    band_diagnostics = {
        model: read_json(args.band_diagnostic_root / model / "band_diagnostic.json")
        for model in MODELS
    }
    running_display_layers = {
        model: int(band_diagnostics[model]["layer"])
        for model in MODELS
    }
    final_display_layers = {
        model: int(representation[model]["final"]["layer"]) for model in MODELS
    }
    geometry_3d = load_all_layer_native_geometry(
        args.geometry_comparison_report,
        running_display_layers,
        final_display_layers,
    )
    token_evidence = {
        model: load_token_ablation_evidence(args.token_ablation_root / model, model)
        for model in MODELS
    }
    attention_examples = load_attention_examples(args.atlas_root)
    qwen_attention_sum_svg = inline_standalone_svg(
        args.atlas_root
        / "Qwen3-8B_adjacent_rank_after_city_Top128_p0_attention_sum.svg"
    )
    gemma_attention_sum_svg = inline_standalone_svg(
        args.atlas_root
        / "Gemma4-E4B_adjacent_rank_after_city_Top6_p0_attention_sum.svg"
    )
    appendix_e_svgs = {
        "qwen_same_head": inline_standalone_svg(
            args.atlas_root / "Qwen3-8B_same_unit_rank_before_city_p0_head_map.svg"
        ),
        "gemma_same_head": inline_standalone_svg(
            args.atlas_root / "Gemma4-E4B_same_unit_rank_before_city_p0_head_map.svg"
        ),
        "qwen_same_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Qwen3-8B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg"
        ),
        "gemma_same_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Gemma4-E4B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg"
        ),
        "qwen_bullet_head": inline_standalone_svg(
            args.atlas_root / "Qwen3-8B_structural_invariant_bullet_p0_head_map.svg"
        ),
        "gemma_bullet_head": inline_standalone_svg(
            args.atlas_root / "Gemma4-E4B_structural_invariant_bullet_p0_head_map.svg"
        ),
        "qwen_bullet_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Qwen3-8B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg"
        ),
        "gemma_bullet_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Gemma4-E4B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg"
        ),
    }
    for example in attention_examples:
        if example["key"] == "gemma_top8_aggregate":
            example["label"] = (
                "Gemma4-E4B · ranked-head aggregate "
                "(columns 1–6 = current Top-6; 7–8 = K8 diagnostic)"
            )
            example["evidence"] = "exact P0 · 20 discovery seeds"
    qwen_head_map = head_map_svg(
        load_p0_head_score_rows(args.atlas_root, "Qwen3-8B"),
        "Qwen3-8B exact-P0 target-attention atlas · current Top-128",
        top_outline=128,
        value_field="discovery_selection_value",
        value_label="seed-event mean target-record attention mass",
    )
    gemma_head_map = split_gemma_current_head_map_svg(
        load_p0_head_score_rows(args.atlas_root, "Gemma4-E4B")
    )
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
    gemma_query_mediation = {
        geometry: read_json(
            args.snapshot_root
            / f"gemma_query_mediation_{geometry}_discovery_claim_gates.json"
        )
        for geometry in ("endpoint", "suffix4", "suffix8")
    }
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
    for geometry, gates in gemma_query_mediation.items():
        require(gates.get("geometry") == geometry, f"Gemma mediation geometry mismatch: {geometry}")
        require(gates.get("phase") == "discovery", f"Gemma mediation phase mismatch: {geometry}")
        require(gates.get("seed_count") == 20, f"Gemma mediation seed drift: {geometry}")
        require(gates.get("selection_rank_used") is False, f"Gemma mediation rank leakage: {geometry}")
        require(gates.get("geometry_pass") is False, f"Gemma mediation null changed: {geometry}")

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
            "orthogonal_d2": effect(query_rows, "full_commit_targeted_attention_vs_orthogonal_distance_2"),
            "orthogonal_d3": effect(query_rows, "full_commit_targeted_attention_vs_orthogonal_distance_3"),
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
    query_distance_groups = [
        (
            "|donor−receiver| = 1",
            {model: float(query_effects[model]["orthogonal"]["mean_effect"]) for model in MODELS},
        ),
        (
            "|donor−receiver| = 2",
            {model: float(query_effects[model]["orthogonal_d2"]["mean_effect"]) for model in MODELS},
        ),
        (
            "|donor−receiver| = 3",
            {model: float(query_effects[model]["orthogonal_d3"]["mean_effect"]) for model in MODELS},
        ),
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
    terminal_groups = [
        (
            "marker-core restoration vs corrupt",
            {
                model: float(terminal_effects[model]["restoration"]["mean_effect"])
                for model in MODELS
            },
        ),
        (
            "marker-core vs equal-token ordinary state",
            {
                model: float(
                    terminal_effects[model]["matched_random_specificity"]["mean_effect"]
                )
                for model in MODELS
            },
        ),
    ]
    mediation_panels = []
    for panel_title, estimand, floor in (
        ("A · Full-state city-log-odds effect", "full_state_effect_intact", 1.0),
        ("B · Selected-mask interaction", "full_selected_mask_interaction", 0.1),
        ("C · Selected-head pre-O restoration", "full_head_output_restore", 0.1),
        ("D · Selected vs random specificity", "full_selected_vs_random_specificity", 0.1),
    ):
        mediation_panels.append(
            (
                panel_title,
                "donor−receiver query city log-odds",
                [
                    (geometry, effect(gemma_query_mediation[geometry]["estimands"], estimand))
                    for geometry in ("endpoint", "suffix4", "suffix8")
                ],
                floor,
            )
        )
    q_free_rows = q_free["confirmation"]["all_estimands"]
    q_free_panels = [
        (
            "A · Candidate-distribution recovery",
            "reduction in TV distance",
            [("clean carrier restore", effect(q_free_rows, "selected_clean_state_distribution_recovery"))],
            1e-12,
        ),
        (
            "B · Expected-count recovery",
            "expected count change",
            [("clean carrier restore", effect(q_free_rows, "selected_clean_state_expected_count_recovery"))],
            1e-12,
        ),
        (
            "C · Correct-count margin recovery",
            "log-probability margin",
            [
                (
                    "clean carrier restore",
                    next(
                        row
                        for row in q_free_rows
                        if row["estimand"] == "selected_clean_state_restoration"
                        and row["outcome"] == "correct_count_margin"
                    ),
                )
            ],
            0.6,
        ),
        (
            "D · Greedy exact-count recovery",
            "paired exact-count rate",
            [
                (
                    "clean carrier restore",
                    next(
                        row
                        for row in q_free_rows
                        if row["estimand"] == "selected_clean_state_restoration"
                        and row["outcome"] == "exact_count"
                    ),
                )
            ],
            0.1,
        ),
    ]

    custom_css = """
.report-note{max-width:920px;color:#475467}.status-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}.status-card{padding:18px;border:1px solid var(--line);background:#fbfcfe}.status-card h3{margin:0 0 8px}.status-card p{margin:6px 0;font-size:14px}.status-good{color:#075e58;font-weight:750}.status-open{color:#9a4b00;font-weight:750}.chain-figure{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fbfcfe}.chain-title{fill:#172033;font-size:13px;font-weight:750}.chain-sub{fill:#667085;font-size:11px}.chain-model{fill:#344054;font-size:12px;font-weight:750}.chain-status{font-size:11px;font-weight:750}.mini-model{font-size:11px;font-weight:800}.metric-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}.metric{padding:15px;border-top:3px solid var(--teal);background:#f8fafc}.metric strong,.metric span{display:block}.metric strong{font-size:22px}.metric span{color:#667085;font-size:12px}.negative-result{padding:17px 19px;border-left:4px solid var(--amber);background:#fff8eb}.audit-list{font-size:12px;color:#667085;overflow-wrap:anywhere}.compact-table td,.compact-table th{padding:7px 8px}.walkthrough-callout{display:grid;grid-template-columns:1fr 1fr;gap:14px}.walkthrough-callout>div{padding:15px;border:1px solid var(--line);background:#fbfcfe}
.definition-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:18px 0 28px}.definition{padding:15px 16px;border:1px solid var(--line);background:#fbfcfe}.definition dt{font-weight:800;color:#172033;margin-bottom:6px}.definition dd{margin:0;color:#475467;font-size:13px;line-height:1.62}.experiment-frame{margin:18px 0 26px;border:1px solid var(--line);background:#fff}.experiment-frame>div{padding:15px 18px;border-bottom:1px solid var(--line)}.experiment-frame>div:last-child{border-bottom:0}.experiment-label{display:inline-block;min-width:88px;color:#0f766e;font-size:11px;font-weight:850;letter-spacing:.07em;text-transform:uppercase}.formula{display:block;margin:9px 0 0;padding:15px 18px;background:#f5f8fb;border-left:3px solid #46758f;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;overflow-x:auto}.figure-primer{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;margin:16px 0 8px;background:var(--line);border:1px solid var(--line)}.figure-primer>div{padding:13px 15px;background:#f8fafc;font-size:12px;line-height:1.55}.figure-primer strong{display:block;margin-bottom:4px;color:#172033}.paper-chart{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}.three-d{margin:12px 0 0;border:1px solid var(--line);background:linear-gradient(#fbfcfe,#f5f8fb)}.three-d-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:13px 15px;border-bottom:1px solid var(--line)}.three-d-controls{display:flex;align-items:end;justify-content:flex-end;gap:10px;flex-wrap:wrap}.three-d-head label,.manifold-panel-head label{color:#475467;font-size:12px;font-weight:700}.three-d-head select,.three-d-head button,.manifold-panel-head select{display:block;margin-top:5px;padding:7px 9px;border:1px solid #b8c1cf;background:#fff;color:#172033;font:inherit}.three-d-head button{cursor:pointer}.manifold-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}.manifold-panel{min-width:0;padding:0;background:#fbfcfe;border:0}.manifold-panel-head{display:flex;align-items:end;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid var(--line)}.manifold-panel-head strong,.manifold-panel-head span{display:block}.manifold-panel-head span{margin-top:3px;color:#667085;font-size:12px}.three-d canvas{display:block;width:100%;height:455px;cursor:grab;touch-action:none}.three-d canvas:active{cursor:grabbing}.manifold-stats{min-height:45px;margin:0;padding:9px 13px;border-top:1px solid var(--line);color:#667085;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.attention-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.attention-pair figure{margin:0}.attention-pair img{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}.attention-atlas-stack{display:grid;grid-template-columns:1fr;gap:26px;margin-top:12px}.attention-atlas-stack figure{margin:0;padding:16px;border:1px solid var(--line);background:#fff}.attention-atlas-frame{width:100%;overflow-x:auto}.attention-atlas-frame .head-map{display:block;width:100%;min-width:900px;height:auto;margin:0 auto}.attention-switcher{margin:14px 0;padding:16px;border:1px solid var(--line);background:#fbfcfe}.attention-select{display:block;max-width:680px;color:#344054;font-size:12px;font-weight:750}.attention-select select{display:block;width:100%;margin-top:7px;padding:9px 11px;border:1px solid #b8c1cf;background:#fff;color:#172033;font:inherit}.attention-example-panel{margin-top:16px}.attention-example-svg{overflow-x:auto}.attention-example-svg svg{display:block;width:100%;min-width:900px;height:auto;margin:0 auto}.map-meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 9px;color:#667085;font-size:12px}.map-meta strong{color:#172033;font-size:14px}.head-map{display:block;width:100%;height:auto;background:#fff}.term-note{font-size:12px;color:#667085}.qualification{padding:16px 18px;border-left:4px solid #0f766e;background:#f0f9f7}.appendix-block{margin-top:22px}.appendix-block summary{cursor:pointer;font-weight:800}.section-conclusion{margin-top:22px;padding:17px 19px;background:#eef7f5;border-left:4px solid #0f766e}.section-conclusion strong{color:#075e58}
.attention-pair svg{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}
@media print{.attention-pair{display:block}.attention-pair figure{break-inside:avoid-page;margin:0 0 20px}.attention-pair figure .head-map{width:auto;max-width:100%;max-height:620px;margin:0 auto}.attention-atlas-stack figure{break-inside:avoid-page}.attention-atlas-frame .head-map{min-width:0}.three-d{break-inside:avoid-page}.three-d canvas{height:430px}.formula{white-space:normal}.attention-switcher{break-inside:avoid-page}}
@media(max-width:900px){.manifold-grid{grid-template-columns:1fr}}
@media(max-width:760px){.status-grid,.walkthrough-callout,.metric-strip,.definition-grid,.attention-pair,.figure-primer{grid-template-columns:1fr}.three-d-head{align-items:flex-start;flex-direction:column}.three-d-controls{justify-content:flex-start}.three-d canvas{height:430px}.chain-figure{min-width:850px}.chain-scroll{overflow-x:auto}}
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
                "Qwen strong; Gemma local specificity attenuated",
            ),
            (
                "marker-core → answer count margin",
                ci(terminal_effects["Qwen3-8B"]["restoration"]),
                ci(terminal_effects["Gemma4-E4B"]["restoration"]),
                "fixed-suffix controlled bridge",
            ),
        ),
    )
    query_distance_table = table(
        ("Distance |d|", "Qwen: full−orthogonal", "Gemma: full−orthogonal", "Registered role"),
        (
            (
                str(distance),
                ci(query_effects["Qwen3-8B"][key]),
                ci(query_effects["Gemma4-E4B"][key]),
                "primary" if distance == 1 else "secondary dose robustness",
            )
            for distance, key in ((1, "orthogonal"), (2, "orthogonal_d2"), (3, "orthogonal_d3"))
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
<title>Native-thinking 计数机制：因果链与表征报告</title><link rel="icon" href="data:,"><style>{css}</style></head>
<body><article class="page"><header><p class="eyebrow">Realistic CoT NiaH · Native-thinking mechanism</p>
<h1>Native-thinking 如何在 trace 中维持并输出 count</h1>
<p class="dek">本报告与 Non-thinking 冻结版使用同一证据语法：先定义对象与判据，再分别检验表征、局部检索、状态写入、循环传播与终端读取。</p>
<div class="meta"><span>Qwen3-8B · frozen Top-128</span><span>Gemma4-E4B · frozen Top-6</span><span>formal: 20 discovery / 10 confirmation</span><span>generated {esc(generated)}</span></div></header>
<nav><a href="#definitions">定义</a><a href="#summary">结论</a><a href="#design">设计</a><a href="#task">任务</a><a href="#representation">表征</a><a href="#retrieval">检索</a><a href="#write">写入与循环</a><a href="#answer">终端</a><a href="#walkthrough">单 seed</a><a href="#appendix">Appendix</a></nav>
<main>

<section id="definitions"><p class="eyebrow">00 · Definitions before claims</p><h2>先定义本文所有核心对象</h2>
<p class="lead">下面的名词在机制图之前统一定义。本文始终区分“信息可读”、“干预足够改变下游”与“自然运行必须使用”；三者不是同一命题。</p>
<dl class="definition-grid">
<div class="definition"><dt>Gold count <em>N</em> 与 occurrence <em>k</em></dt><dd><em>N</em>是 prompt 中真实 needle records 数，取 1–10。<em>k</em>是模型在 trace 中已完成的第 <em>k</em> 条记录。例如 <em>N</em>=6、<em>k</em>=3 表示总共应有 6 条，当前处理到第 3 条。</dd></div>
<div class="definition"><dt>Hidden state / residual stream</dt><dd>每一 token 在某个 transformer block 后的向量 <em>h</em><sub>ℓ,t</sub>。“state patch”是把 donor 的这个向量写到 receiver 的同层位置，而不改变可见 token。</dd></div>
<div class="definition"><dt>Query、attention head 与 head bank</dt><dd>Query 是当前 token 向先前 tokens 读取信息的位置。一枚 attention head 是一条局部读取通道；head bank 是事先冻结的多枚 heads 集合。Top-<em>K</em> 表示 bank 有 <em>K</em> 枚 heads，不表示单头必要。</dd></div>
<div class="definition"><dt>Raw attention mass 与 bank-summed mass</dt><dd>单头从 query <em>q</em> 指向 source span <em>R</em> 的 raw mass 是 <span class="math">A<sub>h</sub>(q,R)=Σ<sub>t∈R</sub>α<sub>h</sub>(q,t)</span>，范围 0–1。整 bank 的 mass 再对 heads 求和，因此可以大于 1；它不是概率，且不同 K 的绝对色值不可直接比较。</dd></div>
<div class="definition"><dt>Targeted retrieval</dt><dd>在一次 <em>k</em>→<em>k</em>+1 的 trace query 上，head bank 对 prompt 中第 <em>k</em>+1 条正确 city record 进行定向读取。它不是 answer-time 对全部 records 的 broad retrieval。</dd></div>
<div class="definition"><dt>Grammar carrier</dt><dd>检索之后、item commit 之前承载当前进度的 grammar-specific token states。Rank-after-city trace 使用显式 marker core；rank-before-city trace 使用 city-to-commit tail。</dd></div>
<div class="definition"><dt>Commit state (P0)</dt><dd>一条 trace item 完成时的 item-end post-block hidden state。它是“已经处理到 <em>k</em>”的提交位置，也是下一次 targeted query 的上游候选状态。</dd></div>
<div class="definition"><dt>Ablation、self control 与 matched control</dt><dd>Ablation 在指定 query 清零 selected heads 的 pre-O output。Self-patch 把 receiver 自己写回，控制 hook 本身。Layer-matched random 使用同层同数随机 heads。Orthogonal control 与 <em>donor−receiver 差向量在 frozen count 子空间中的投影</em>等范数，但其方向与该子空间正交；它并不与完整 donor−receiver 差向量等范数。</dd></div>
<div class="definition"><dt>Representation、probe 与 PCA manifold</dt><dd>Representation 问 hidden state 中是否可读出 count。Probe 是只在 discovery 拟合的简单分类器。PCA manifold 是 discovery-fitted 主成分上的可视化；confirmation 只被投影，不参与定轴。</dd></div>
<div class="definition"><dt>Effect、95% CI 与 distance <em>d</em></dt><dd>Effect 是同 seed、同 pair 内 treatment−control 的均值。95% CI 是 seed-level bootstrap 区间，作为稳定性记录。<em>d</em>=donor occurrence−receiver occurrence；正文主检验使用 |<em>d</em>|=1。</dd></div>
<div class="definition"><dt>Confirmed、directional 与 controlled only</dt><dd><strong>Confirmed</strong>：冻结设计在 20-seed discovery 后于 10-seed confirmation 复现直接因果对比。<strong>Directional</strong>：均值方向对，但 matched-control specificity 不稳定。<strong>Controlled only</strong>：只在 teacher-forced 或 fixed-suffix 环境成立，尚未推广到 free-running 全链。</dd></div>
</dl>
<div class="section-conclusion"><strong>定义层面的结论。</strong> 后文的“confirmed”不意味唯一回路；“可解码”也不自动意味模型必须使用该线性方向。</div></section>

<section id="summary"><p class="eyebrow">Conclusion first</p><h2>一条 recurrent counting pathway 已经接上；natural end-to-end sufficiency 仍未证明</h2>
<p class="lead">两模型都支持同一类局部因果链：targeted heads 检索下一条 city，改变 grammar-specific marker/tail carrier；carrier 写入 commit state；commit state 再改变下一次 targeted query。终端 marker state 在固定 suffix 的受控实验中能恢复 answer count margin，但把全部上下文抹掉后，仅恢复任一单 item 并不能让答案随 k 从 1 走到 10。</p>
<div class="figure-primer"><div><strong>图中画什么</strong>五个框是一次循环计数从检索到答案的候选阶段。</div><div><strong>怎么读</strong>每一列分别给 Qwen 与 Gemma 的最高证据级别；紫色表示仅在受控终端成立。</div><div><strong>简单例子</strong>完成第 3 条后，commit state 改变“下一次应读第 4 条”的 query routing。</div></div>
<div class="chain-scroll">{chain_svg()}</div>
<div class="status-grid"><div class="status-card"><h3>Qwen3-8B</h3><p class="status-good">recurrent loop：强 confirmation</p><p>Top-128 retrieval、carrier→commit、commit→next query 都有大效应。terminal marker 的局部受控 restoration 为正。</p><p class="status-open">仍开放：free-running answer count-margin 与全上下文擦除后的单点 sufficiency。</p></div>
<div class="status-card"><h3>Gemma4-E4B</h3><p class="status-good">recurrent direct edge：confirmed†</p><p>Top-6 retrieval、carrier→commit 成立；commit→next query 在 prospective confirmation 相对 self patch 为 +0.491。</p><p class="status-open">† 局部 |d|=1 的 orthogonal specificity 只有 +0.126，但 |d|=2/3 按 +0.175/+0.244 增强；这是已确认直接边的限制，不是强 selection-specific bottleneck。</p></div></div>
<div class="claim"><strong>允许的主张。</strong> Native-thinking trace 中存在一条可重复干预的 recurrent counting pathway；它不是已证明唯一或排他的 counting circuit。</div></section>

<section id="design"><p class="eyebrow">01 · Experimental contract</p><h2>设计与判据</h2>
<p class="lead">正式因果实验固定使用 discovery seeds 1234–1253 与 confirmation seeds 1254–1263；所有正式 pair plan 都是 outcome-blind，且不使用 selection_rank。单 seed walkthrough 只作 case study。</p>
<div class="reading-protocol"><div class="protocol-step"><span class="protocol-no">Discovery</span><h3>定位与冻结</h3><p>20 seeds。选择层、head bank、geometry 与 primary estimand。</p></div><div class="protocol-step"><span class="protocol-no">Confirmation</span><h3>独立复现</h3><p>10 seeds。设计不因 partial outcomes 改动。</p></div><div class="protocol-step"><span class="protocol-no">Claim scope</span><h3>效应优先</h3><p>正文同时给 mean effect 与 95% CI，但机制判断首先看效应方向、大小和控制组。</p></div></div>
<p class="report-note">Head bank：Qwen Top-128；Gemma Top-6。Gemma K=6 的 selected−random retrieval failure 比 K=8 更大，因此最新主线固定 K=6；旧 K=8 仅属于历史实验。</p></section>

<section id="task"><p class="eyebrow">02 · Task and unit of analysis</p><h2>任务设定：模型一边在 trace 中找 city，一边维持当前计数</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>先明确需要解释的行为单位：不仅是最后数字 <em>N</em>，还包括 trace 中每个 <em>k</em>→<em>k</em>+1 的局部检索。</div><div><span class="experiment-label">设定</span>每条 prompt 约 10k tokens，含 1–10 条真实 city records 与大量普通文本。Native-thinking 模型先生成 reasoning trace，再输出 total count。</div><div><span class="experiment-label">简单例子</span>Prompt 里有 6 条真实 records。Trace 已处理到 <em>k</em>=3 时，下一次 query 应读第 4 条 city；完成第 6 条后，answer query 应输出 6。</div><div><span class="experiment-label">分析单位</span>Representation 使用 seed–count 或 seed–occurrence state；causal patching 使用同 seed 内 donor–receiver pair，所有 treatment/control 在 pair 内配对。</div></div>
<div class="section-conclusion"><strong>本节结论。</strong> 要串起完整机制，必须同时解释“下一条读谁”和“已处理多少条”；只看 final exact accuracy 会混合多个故障源。</div></section>

<section id="representation"><p class="eyebrow">03 · Representation</p><h2>Trace commit 与 answer query 都含有可读的 count geometry</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>定位 count information 何时出现、在 trace 和 answer 两个 endpoint 上是否可分。这一步只找候选状态，不声称自然因果使用。</div><div><span class="experiment-label">设定</span>每个 discovery fold 内单独拟合 standardization、PCA-16 与 L2 logistic probe；nearest-centroid (NCC) 作为不同决策规则的控制。选定层冻结后只在 10 confirmation seeds 评估。</div><div><span class="experiment-label">计算方法</span><span class="formula">BalancedAccuracy = (1/10) Σ<sub>c=1</sub><sup>10</sup> TP<sub>c</sub> / (TP<sub>c</sub> + FN<sub>c</sub>)</span>十个 count 类等权，因此 chance=0.10，不被不同 occurrence 样本数影响。</div><div><span class="experiment-label">简单例子</span>如果第 4 条完成后的 hidden states 在 held-out seeds 上仍靠近“4”的 discovery centroid，而远离“3/5”的 centroid，则该层含有可读的 running count。</div></div>

<h3>3.1 可解码性如何随层变化</h3>
<div class="figure-primer"><div><strong>图中画什么</strong>上排是 item-end running count，下排是 answer-query final count。</div><div><strong>坐标怎么读</strong>横轴是 zero-based post-block layer；纵轴是 confirmation balanced accuracy；灰虚线 0.10 是 chance。</div><div><strong>圆点是什么</strong>空心圆是 discovery 冻结的报告层，不是根据 confirmation 再挑的最高点。</div></div>
<figure><h3 class="figure-title">图 1a · Native count representation 的逐层 held-out readout</h3>{layerwise_representation_svg(layerwise_representation, representation)}<figcaption>每个 panel 的横轴是 zero-based transformer layer；纵轴是 10-class confirmation balanced accuracy。绿线=L2 logistic，橙线=nearest centroid，灰虚线=0.10 chance。Running panel 固定使用 exact causal item-end site；final panel 使用 answer query。折线是描述性层进程，不把最高 confirmation 点当作新选择。</figcaption></figure>

<h3>3.2 Count clouds 在低维中长什么样</h3>
<p><strong>先定义三维 PCA 图。</strong> PC1、PC2、PC3 是 discovery hidden-state covariance 中方差最大的三个彼此正交方向；它们不是三个预设的“count 维度”，也不是三个神经元。对 discovery 矩阵 <em>H</em><sub>D</sub> 去均值后取前三个右奇异向量 <em>V</em><sub>3</sub>，confirmation state 只做固定投影：</p>
<span class="formula">μ<sub>D,j</sub>, σ<sub>D,j</sub> = discovery mean and standard deviation of hidden dimension j<br>X<sub>D,j</sub> = (H<sub>D,j</sub>−μ<sub>D,j</sub>)/σ<sub>D,j</sub>, &nbsp; V<sub>3</sub> = top-3 right singular vectors of X<sub>D</sub><br>z<sub>confirm</sub> = ((h<sub>confirm</sub>−μ<sub>D</sub>)/σ<sub>D</sub>)V<sub>3</sub> = (PC1, PC2, PC3)</span>
<p>这条计算规则阻止 confirmation 信息参与定轴。图中的小点是 confirmation states；带数字的大点是各 occurrence/count 的 confirmation centroid；折线仅连接 1→10 的 centroid，帮助观察几何顺序，不进入 probe 或因果统计。分类器仍使用 PCA-16，因此 3D 云比 probe 有意丢弃更多信息。</p>
<div class="figure-primer"><div><strong>如何交互</strong>拖动点云改变观察角度；双击或按 Reset 恢复视角；模型下拉框切换 Qwen/Gemma。</div><div><strong>Layer 选择</strong>下拉框只显示 discovery 冻结的 report layer。逐层比较已在图 1a 完成，不允许凭 confirmation 云形状重新挑层。</div><div><strong>坐标和颜色</strong>三轴是任意 PCA 单位；颜色与 centroid 数字均表示 1–10。不同模型分别归一显示，不能跨模型比较坐标尺度。</div></div>

<figure><h3 class="figure-title">图 1b · Trace 与 answer 的逐层 PC1–PC3 confirmation manifold</h3>
<div class="three-d manifold-comparison"><div class="three-d-head"><div><strong>Native hidden-state geometry · trace / answer side by side</strong><div class="term-note">拖动任一 panel 会同步旋转两侧，便于比较几何形状；两个 endpoint 仍使用各自独立的 discovery-fitted PCA basis。</div></div><div class="three-d-controls"><label>模型<select id="native-geometry-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label><button type="button" id="native-geometry-reset">Reset both views</button></div></div>
<div class="manifold-grid"><section class="manifold-panel"><div class="manifold-panel-head"><div><strong>Trace commit</strong><span>running occurrence <em>k</em></span></div><label>Layer<select id="native-running-layer" aria-label="Native running manifold layer"></select></label></div><canvas id="native-running-canvas" aria-label="Native-thinking item-end confirmation hidden states projected onto layer-specific discovery-fitted PC1 PC2 PC3. Drag to rotate.">浏览器需要 Canvas 支持。</canvas><p class="manifold-stats" id="native-running-stats"></p></section>
<section class="manifold-panel"><div class="manifold-panel-head"><div><strong>Answer query</strong><span>final gold count <em>N</em></span></div><label>Layer<select id="native-final-layer" aria-label="Native final manifold layer"></select></label></div><canvas id="native-final-canvas" aria-label="Native-thinking answer-query confirmation hidden states projected onto layer-specific discovery-fitted PC1 PC2 PC3. Drag to rotate.">浏览器需要 Canvas 支持。</canvas><p class="manifold-stats" id="native-final-stats"></p></section></div></div>
<figcaption>左 panel 取 trace item-end hidden states，颜色/数字是已完成 occurrence <em>k</em>；右 panel 取数字答案生成前的 <code>answer_query_v3</code>，颜色/数字是 gold count <em>N</em>。模型选择器同时切换两侧；两个 layer 选择器彼此独立，覆盖该模型的全部 post-block layers。每个 layer 都只用 discovery states 拟合 StandardScaler 与 PCA3，再显示 confirmation states；默认层分别为 running Qwen L{running_display_layers['Qwen3-8B']} / Gemma L{running_display_layers['Gemma4-E4B']}，answer Qwen L{final_display_layers['Qwen3-8B']} / Gemma L{final_display_layers['Gemma4-E4B']}。下方状态行实时报告 token site、样本数和前三 PC 的 discovery explained-variance ratio。</figcaption></figure>
<p><strong>如何严谨地使用 layer selector。</strong> 它用于检查 geometry 随深度如何演化，不参与任何 formal selection。默认层在 discovery 阶段冻结；图 1a 的 layerwise confirmation curves 是完整的数值汇总。用户手动切到另一层后看到的云只能作为描述性浏览，不能据此事后替换默认层或改变 causal experiment。</p>
<p><strong>替代解释与左右比较边界。</strong> Running endpoint 的 marker kind、boundary token、trace grammar 和 token identity 会共同改变 residual state，冻结诊断中也出现 grammar/token 相关双 band；因此左侧分簇不能单独证明“纯 count axis”。左右两侧使用不同 endpoint、不同 state 数和独立 PCA basis，只能比较有无有序结构，不能逐轴对齐 PC1 或比较 centroid 距离。更可靠的 evidence 是跨-seed PCA-16 readout 与后续配对 patching。</p>

<h3>3.3 冻结层的数值结果</h3>
<figure><h3 class="figure-title">图 1c · Frozen report layers 的 confirmation balanced accuracy</h3>{grouped_bars_svg('Frozen-decoder balanced accuracy', rep_groups, maximum=1.0)}<figcaption>Qwen running commit L18={float(representation['Qwen3-8B']['running']['confirmation_logistic_balanced_accuracy']):.3f}，answer query L26={float(representation['Qwen3-8B']['final']['confirmation_logistic_balanced_accuracy']):.3f}。Gemma running commit L16={float(representation['Gemma4-E4B']['running']['confirmation_logistic_balanced_accuracy']):.3f}，answer query L34={float(representation['Gemma4-E4B']['final']['confirmation_logistic_balanced_accuracy']):.3f}。横向条长是 balanced accuracy，从 0 到 1；同一行的两颜色分别是 Qwen 与 Gemma。数值来自 PCA-16 probe，而不是前三 PC 的视觉可分性。</figcaption></figure>
<p><strong>结果分析。</strong> Qwen 的 answer-query count 几乎完全可读，Gemma 的 running commit 反而比 answer query 更强。这说明两模型不必在相同层或同一维度上保留 count；但两者都有可供后续因果实验锁定的 state。三维图补充的是几何直觉，图 1a/1d 的 held-out 数值才是 representation 主证据。</p>
<div class="section-conclusion"><strong>Experiment 3 结论。</strong> Count information 同时存在于循环中的 commit state 与末端 answer state；本节只建立 representation，后续 ablation/patching 再判断这些状态是否被实际使用。</div></section>

<section id="retrieval"><p class="eyebrow">04 · Targeted retrieval</p><h2>下一条 city 由 model-specific targeted head banks 定向检索</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>确定“该读 prompt 中哪一条 city”是否由 discovery-localized heads 实际执行，而不只是 attention visualization。</div><div><span class="experiment-label">设定</span>在 exact next-city query 开始清零 selected bank 的 pre-O head slices，干预持续到首个可识别 city。每个 selected bank 与同层构成、同 K 的 random bank 配对。</div><div><span class="experiment-label">主终点</span><span class="formula">Failure(B) = 1[first generated semantic city ≠ registered next city]<br>Δ<sub>bank</sub> = mean Failure(selected) − mean Failure(layer-matched random)</span></div><div><span class="experiment-label">简单例子</span>当 trace 正从 3 走向 4，正确 target 是 prompt 第 4 条 Paris。如果只关 selected heads 后首 city 变成 Oslo，而关同数 random heads 仍是 Paris，该 pair 支持 selection-specific retrieval necessity。</div></div>

<h3>4.1 Heads 在哪里，attention 是否随 target 移动</h3>
<div class="figure-primer"><div><strong>左图</strong>横轴是 transition <em>k</em>→<em>k</em>+1，纵轴是 prompt record ordinal；颜色是 Top-K 合计 raw attention mass。</div><div><strong>正确图样</strong>亮带应随 <em>k</em>+1 沿对角线移动，而不是永远固定在第一条。</div><div><strong>证据边界</strong>这两张图是 descriptive localization；因果必要性来自下面 selected-vs-random ablation。</div></div>
<div class="attention-pair"><figure>{qwen_attention_sum_svg}<figcaption>图 2a · Qwen Top-128 在单条真实 trace 上的 bank-summed attention。横轴为 P0 transition，纵轴为真实 prompt record，颜色为 128 枚 heads 对 record span 的 raw attention mass 之和。跨模型不比较颜色绝对值。</figcaption></figure><figure>{gemma_attention_sum_svg}<figcaption>图 2b · Gemma Top-6 的同构图。横轴、纵轴和颜色定义与左图相同，但 bank 只有 6 枚 heads。对角 target pattern 说明少数 heads 可被多个 occurrence 重复使用。</figcaption></figure></div>

<h3>4.2 单头、整 bank 与跨-seed attention pattern</h3>
<div class="figure-primer"><div><strong>单轨迹 panels</strong>横轴是 P0 transition；纵轴是带真实 city 名的 prompt records；红框或红点标出正确 <em>k</em>+1 target。</div><div><strong>跨-seed panel</strong>横轴是 discovery-ranked head；纵轴是正确 target ordinal；颜色是先在 seed 内平均、再对 20 discovery seeds 等权平均的 raw target mass。</div><div><strong>怎么使用</strong>用下拉框切换 Qwen/Gemma、单头/整 bank/跨-seed 版本。热图展示 routing 形状，不替代因果 ablation。</div></div>
<figure><h3 class="figure-title">图 2c · 可切换的 Native targeted-retrieval attention maps</h3>{attention_example_switcher(attention_examples)}<figcaption>单头和 bank-summed panels：横轴=P0 transition <em>k</em>→<em>k</em>+1，纵轴=prompt record region，颜色=raw attention mass，红框/红点=正确 next record；Top-K 合计可能大于 1。Ordinal×head panels：横轴=ranked head，纵轴=target ordinal，颜色=20 discovery seeds 的 seed-equal target mass。Gemma aggregate 保留八列以复现历史 atlas，其中前六列构成当前 Top-6，后两列仅是 K8 diagnostic；正式因果结论始终使用 Top-6。</figcaption></figure>

<h3>4.3 Head bank 在 layer×head 空间中的位置</h3>
<div class="figure-primer"><div><strong>坐标</strong>横轴是 head index <em>h</em>，纵轴是 zero-based layer ℓ；每个格子是一枚 head。</div><div><strong>颜色与边框</strong>颜色越深表示 exact-P0 query 对正确 next record 的 discovery seed-event mean raw mass 越大；红框是当前 frozen Top-K membership。</div><div><strong>为何要画</strong>它显示 bank 是集中在少数层还是跨层分布，也防止把 Top-K 误写成一个单层模块。</div></div>
<p>Atlas 的每格先在一个 event 内求 query 对正确 target-record span 的 attention mass，再在同一 seed 内平均 events，最后对 20 个 discovery seeds 等权平均。这样，一条 trace 中 occurrence 较多的 seed 不会因为事件数更多而自动获得更大权重。显示时对每个模型自己的分数取 99th percentile 作为色标上限，并对归一值开平方以显示弱但非零的 heads；红框 membership 使用未截断的冻结 rank。</p>
<div class="attention-atlas-stack"><figure><h3 class="figure-title">图 2d · Qwen Top-128 layer×head atlas（全宽）</h3><div class="attention-atlas-frame">{qwen_head_map}</div><figcaption>横轴=H0–H31，纵轴=L0–L35；每个格子是一枚 Qwen attention head。颜色是 20 discovery seeds 的 seed-equal exact-P0 target-record raw mass，经 Qwen 内部 99th-percentile 截断后显示；红框=冻结 Top-128。全宽图显示 selected bank 跨多个中后层分布，因此“Top-128”是宽 bank-level necessity，不表示 128 枚 heads 各自同等必要，也不表示一个单层 module。</figcaption></figure><figure><h3 class="figure-title">图 2e · Gemma Top-6 layer×head atlas（L0–20 / L21–41 分栏放大）</h3><div class="attention-atlas-frame">{gemma_head_map}</div><figcaption>左右 panel 分别是 L0–L20 与 L21–L41；每个 panel 横轴=H0–H7，纵轴=逐层 Lℓ，红框仅标当前冻结 Top-6，而非历史 Top-8。颜色在 Gemma 内独立按同一规则缩放。分栏只是版式变换，未改变 layer/head 坐标、分数、rank 或 bank membership；因此可读性提高但科研对象不变。</figcaption></figure></div>
<p><strong>Atlas 的证据边界。</strong> 高 attention mass 是 localization 指标，不等于该 head 因果必要；红框本身也可能包含冗余 heads。真正的 selection specificity 来自下一节把 selected bank 与同层、同 K random bank 做配对 ablation。反过来，random bank 也可能偶然包含有用 heads，所以 selected−random 是比 selected−clean 更严格、通常也更小的效应。</p>

<h3>4.4 关闭 bank 后，下一条 city 是否失败</h3>
<div class="figure-primer"><div><strong>图中画什么</strong>不同冻结 K 下的 selected 与 random failure rate。</div><div><strong>坐标</strong>横轴=bank size K；纵轴=0–1 retrieval failure rate。绿线高于灰线才是 selection specificity。</div><div><strong>为何 K 不同</strong>Qwen 的 routing 分布得更宽，Gemma 的效应集中于窄 bank；K 不是跨架构同一容量单位。</div></div>
<figure><h3 class="figure-title">图 2f · Targeted-bank confirmation dose response</h3>{line_chart((('Qwen3-8B', q_dose), ('Gemma4-E4B', g_dose)))}<figcaption>横轴是各模型的 frozen bank size K；纵轴是首个 semantic city 的 failure rate。绿色=selected bank，灰色=layer-matched random。Qwen 在 K=128 的 selected/random 为 {float(q_primary['selected_failure_rate']):.3f}/{float(q_primary['random_failure_rate']):.3f}；Gemma 在 K=6 为 {float(g_primary['selected_failure_rate']):.3f}/{float(g_primary['random_failure_rate']):.3f}。</figcaption></figure>
{targeted_table}
<p><strong>结果分析。</strong> Qwen 需要宽 Top-128 才超过旁路冗余，Gemma 的少数 heads 已能强烈破坏 next-city retrieval。这个结果只支持 bank-level necessity；它不说 Top-128 中每枚 Qwen head 都同等重要。</p>
<div class="section-conclusion"><strong>Experiment 4 结论。</strong> Targeted retrieval 是两模型的强必要边：Qwen Top-128 的 selected−random failure={100*float(q_primary['selected_minus_random_failure_rate']):+.1f} pp；Gemma Top-6={100*float(g_primary['selected_minus_random_failure_rate']):+.1f} pp。</div></section>

<section id="write"><p class="eyebrow">05 · Write and recurrent propagation</p><h2>Targeted retrieval 写入 grammar carrier，commit state 再改变下一次 query</h2>
<h3>5.1 Targeted bank → grammar carrier → commit</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>把“已知 selected bank 负责检索”接到“count state 被写进 trace”，避免只有两个彼此无关的局部效应。</div><div><span class="experiment-label">设定</span>保持可见 trace tokens 不变，在注册 query 上 mask selected bank。然后将同一 clean forward 的 carrier hidden state 从 source layer 起累积恢复，并与相近深度、相同 token budget 的 ordinary state 比较。</div><div><span class="experiment-label">计算方法</span><span class="formula">Deformation = RMS(h<sup>masked</sup><sub>carrier</sub> − h<sup>clean</sup><sub>carrier</sub>)<br>Restoration = D(masked, clean) − D(clean-carrier-restored, clean)</span>正值分别表示 selected ablation 使 carrier 变形，以及恢复 carrier 确实把 commit 拉回 clean state。</div><div><span class="experiment-label">简单例子</span>在 rank-after-city trace 中，模型读到第 4 条 city 后写出“这是第 4 条”的 marker。如果关 selected heads 让 marker state 偏离，恢复该 marker 又救回 item-end commit，则两条边被串联。</div></div>
<div class="figure-primer"><div><strong>三行的含义</strong>分别是 carrier 变形、commit 恢复、相同位置相对 ordinary-state 的 specificity。</div><div><strong>坐标</strong>横条长是 confirmation mean effect，左侧行名定义各自量。</div><div><strong>限制</strong>三行量纲不同，只在同一行比较 Qwen/Gemma，不把三条长相加。</div></div>
<figure><h3 class="figure-title">图 3 · Targeted retrieval 对 carrier 的写入与 commit restoration</h3>{grouped_bars_svg('Confirmation mean effects', write_groups)}<figcaption>每行横轴都是注册 treatment−control 的 confirmation seed-level mean effect；绿=Qwen，紫=Gemma。第一行是 carrier RMS deformation，第二行是 commit distance recovery，第三行是正确 carrier restoration 减去等 token ordinary-state restoration。精确单位与 CI 见下表。</figcaption></figure>
{write_table}
<p><strong>结果分析。</strong> Qwen 的向量位移更大，但 Gemma 的三个配对对比也均为正。位置控制排除了“任意恢复同样多 token states 都会救回下游”。</p>
<div class="section-conclusion"><strong>Experiment 5.1 结论。</strong> 两模型都确认 targeted retrieval 会改变 grammar-specific carrier，且 clean carrier 对后续 commit state 具有位置特异的恢复作用。</div>

<h3>5.2 Commit state → next targeted query</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>检验 commit state 是否真的设定下一次 targeted routing，从而闭合 trace 内部的 recurrent loop。</div><div><span class="experiment-label">设定</span>在 source layer 将 donor occurrence 的完整 single-token commit hidden state patch 到 receiver commit 位置；下游 token 与 answer query 均不 patch。Self-patch 控制 hook。Count-subspace transplant 只写入 donor−receiver 差向量在 frozen count 子空间中的投影；orthogonal patch 使用与该投影等范数、但正交于 count 子空间的扰动。它们判断完整 state 是否超过低维 count 方向或一般同尺度扰动。</div><div><span class="experiment-label">主终点</span><span class="formula">Y = A<sub>bank</sub>(donor-successor record) − A<sub>bank</sub>(receiver-successor record)<br>Δ<sub>self</sub> = Y<sub>full donor</sub> − Y<sub>self</sub>; Δ<sub>orth</sub> = Y<sub>full donor</sub> − Y<sub>orthogonal</sub></span></div><div><span class="experiment-label">简单例子</span>把“已完成 5”的 donor commit 写到“已完成 4”的 receiver。如果下一 query 相对第 5 条的后继 record，改为更看第 6 条 donor-successor record，Y 变大。</div></div>
<div class="figure-primer"><div><strong>前两行</strong>同一 full commit 效应分别相对 self 和 orthogonal controls。</div><div><strong>第三行</strong>是之后 terminal local bridge，量纲不同，放在同图仅用于展示链上相邻已确认效应。</div><div><strong>† 的含义</strong>Gemma 对 self 的直接边已在 confirmation 复现；但 |d|=1 的 orthogonal specificity 较小。</div></div>
<figure><h3 class="figure-title">图 4a · Commit patch 对 next-query targeted routing 的 confirmation effect</h3>{grouped_bars_svg('Confirmation mean effects', query_groups)}<figcaption>横条是 confirmation treatment−control mean effect。前两行的单位是 frozen targeted bank 对 donor-successor 减 receiver-successor record 的 raw attention-mass difference；第三行是 terminal correct-count logit margin，不与前两行比较条长。</figcaption></figure>
{query_table}

<h3>5.3 “directional”是什么，是否必须再做一轮</h3>
<p>旧图把 Gemma 的 next-query 边标为 <code>directional</code>，是因为最严格的 |d|=1 <span class="term-note">full donor−orthogonal</span> 对比为 {ci(query_effects['Gemma4-E4B']['orthogonal'])}：均值为正，但 seed 稳定性不够。它不是“full commit 没有效果”；相对 self 的 prospective confirmation 是 {ci(query_effects['Gemma4-E4B']['self'])}。</p>
<figure><h3 class="figure-title">图 4b · Full commit 相对 orthogonal control 的 donor-distance robustness</h3>{grouped_bars_svg('Full commit − orthogonal targeted-attention effect', query_distance_groups)}<figcaption>纵向三行是 donor/receiver occurrence 距离 |d|=1,2,3；横条是 full-donor patch 相对 orthogonal patch 的 targeted-attention mean effect。Orthogonal 扰动与 count-projected donor−receiver change 等范数，而非与完整 donor−receiver change 等范数。|d|=1 是冻结 primary，|d|=2/3 是预注册 secondary dose robustness，不反向替换 primary。Gemma 从 +0.126 增到 +0.175/+0.244；Qwen 在所有距离都远大于 0。</figcaption></figure>
{query_distance_table}
<div class="qualification"><strong>状态判定。</strong> 本报告将 Gemma 该边改为 <strong>confirmed†</strong>：直接 full-state→next-query 效应已用 10 个 prospective confirmation seeds 复现，而且两个冻结 primary means 均为正；† 明确保留“local |d|=1 matched-control specificity 衰减”。不为改变图中颜色再开一个 outcome-driven 实验。</div>
<p><strong>结果分析。</strong> Qwen 的 recurrent loop 效应大且控制特异。Gemma 的 direct routing 效应较小，但随 donor distance 增强，更像分布式 full state 携带的序数信息，而不是单一低维 counter direction。</p>
<div class="section-conclusion"><strong>Experiment 5.2–5.3 结论。</strong> 两模型的 full commit state 都会因果地改变 next-query targeted routing。Qwen 可写成强 matched-control specificity；Gemma 应写成 confirmed direct edge with attenuated local specificity，不声称一条狭窄、排他的 head bottleneck。</div></section>

<section id="answer"><p class="eyebrow">06 · Terminal readout</p><h2>Answer query 自然依赖 trace source；terminal state 在 fixed-suffix 中能控制 count margin</h2>
<h3>6.1 Answer 到底在读 trace，还是回 prompt 重数</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>区分两种 answer-time 路径：（1）从已写好的 trace states 读 count；（2）在 answer query 再对 prompt records 做一次 broad retrieval。</div><div><span class="experiment-label">设定</span>用等长 blank tokens 分别替换 prompt records 或完整 trace items，其余上下文保持不变。两套 answer head banks 分别是 discovery 中对 prompt-record source 和 trace-item source 排名的 Top-32。</div><div><span class="experiment-label">主终点</span><span class="formula">Exact = 1[greedy parsed answer = gold N]<br>Trace necessity = Exact<sub>clean</sub> − Exact<sub>trace blank</sub></span></div><div><span class="experiment-label">简单例子</span>如果保留 trace 而擦除 prompt records 后仍答对 6，但保留 prompt 而擦除 trace 后答错，则此时 answer 主要使用 trace 已写入的状态。</div></div>
<div class="figure-primer"><div><strong>左图/上图</strong>比较 clean、prompt-record blank 和 full-trace blank 的 greedy exact accuracy。</div><div><strong>Head-bank 图</strong>展示 answer query 在 prompt-record 与 trace-context 两类 source 上的 attention composition。</div><div><strong>不能过度解释</strong>Source blank 证明内容必要性，不证明某个 Top-32 bank 是唯一 readout。</div></div>
<figure><h3 class="figure-title">图 5a · Token-source ablation 对局部检索与最终 count 的影响</h3>{token_source_ablation_svg(token_evidence)}<figcaption>左 panel 的横轴依次为 clean、早期一半 trace blank、较早累计 trace blank、最近 transition blank、完整 trace blank；纵轴是 next-city retrieval success rate。柱是 selected treatment，黑色短横线是 equal-token matched control。右 panel 的横轴为 clean、prompt records blank、full trace blank、prompt+trace blank；纵轴是 greedy exact-count accuracy。两 panel 均以百分比为单位，Qwen/Gemma 每个 confirmation condition 各 100 prompts。右 panel 中 Qwen clean 0.97→trace blank 0.01，Gemma 0.70→0.12；prompt-record blank 的损伤远小于 trace blank。</figcaption></figure>
<figure><h3 class="figure-title">图 5b · Answer-query head banks 的 source composition</h3><div class="figure-scroll">{answer_source_rerouting_svg(token_evidence)}</div><figcaption>每行是一个冻结 answer bank 在一种 token condition 下的 source composition。横向堆叠条在 prompt-record 与 trace-context 两个互斥 source groups 内归一；右侧 Σ 是两组 raw bank-summed mass 之和。颜色说明 attention 来源，不是 accuracy。</figcaption></figure>
<p><strong>结果分析。</strong> 两模型都可在 answer query 读 prompt，但当 trace 已存在时，最终 exact answer 对 trace content 的依赖更强。这与“count 随 trace stream 传递”一致，也允许 broad prompt retrieval 作为并行补充路径。</p>
<div class="section-conclusion"><strong>Experiment 6.1 结论。</strong> Trace 是 final answer 的主要自然信息源；prompt-broad retrieval 并未被排除，但“在 answer 时才从 prompt 重数”不足以解释 source-blank 对比。</div>

<h3>6.2 Terminal marker/tail state 是否能改变 answer</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>把“trace 内容必要”缩小到 grammar-aware terminal state→answer 的局部因果边。</div><div><span class="experiment-label">设定</span>根据 trace grammar 选择 marker core 或 city-to-commit tail；在固定 teacher-forced suffix 中恢复 clean hidden states，并与同 token budget 的 ordinary state patch 比较。</div><div><span class="experiment-label">计算方法</span><span class="formula">Margin(N) = log P(answer=N) − max<sub>j≠N</sub> log P(answer=j)<br>Restoration = Margin<sub>clean state restored</sub> − Margin<sub>corrupt</sub></span></div><div><span class="experiment-label">简单例子</span>Gold N=6。当 terminal marker 被破坏时，模型更偏向 5；恢复 clean marker 后 log P(6)−log P(5) 上升，而恢复同样多普通 states 不上升。</div></div>
<div class="figure-primer"><div><strong>第一行</strong>clean marker-core restoration 相对被破坏 baseline 的 correct-count margin 增量。</div><div><strong>第二行</strong>同一 restoration 相对 equal-token ordinary-state restoration；它控制“补回任意 state 都有帮助”。</div><div><strong>坐标</strong>横条单位是 log-probability margin；绿=Qwen，紫=Gemma。两行可直接比较方向，但不能与 accuracy 百分比相加。</div></div>
<figure><h3 class="figure-title">图 5c · Terminal grammar-state 对 correct-count margin 的受控恢复</h3>{grouped_bars_svg('Fixed-suffix terminal-state confirmation effects', terminal_groups)}<figcaption>第一行是 <code>marker_core_restore − uninformative</code>，第二行是 <code>marker_core_restore − marker_core_matched_random</code>；均为 10 confirmation seeds 的 seed-level mean effect。正值表示恢复 grammar-aware terminal state 比 corrupt 或等 token ordinary state 更有利于 gold count。该图只对应 fixed-suffix teacher-forced 条件，不能直接替代 Appendix C 的 free-running null。</figcaption></figure>
<p>在固定 suffix 的 grammar-span patch 中，marker-core clean-state restoration 对 correct-count margin 为 Qwen {ci(terminal_effects['Qwen3-8B']['restoration'])}、Gemma {ci(terminal_effects['Gemma4-E4B']['restoration'])}；matched-random specificity 也为正。</p>
<div class="negative-result"><strong>为何只标 controlled only。</strong> Qwen 的更自由 targeted-counter/count-margin recovery 未通过 strong gate；完整 distribution、expected count 与 greedy exact count 没有形成稳定恢复。因此本节只支持固定 suffix 的 local bridge，不支持单一 terminal state 在 free-running 中无条件接管答案。</div>
<div class="section-conclusion"><strong>Experiment 6.2 结论。</strong> Terminal grammar state 在受控上下文中会因果改变 correct-count margin；但 final answer 仍可同时使用其他 trace states、prompt retrieval 与后续自我校正，所以不主张“只依赖”这一路。</div></section>

<section id="walkthrough"><p class="eyebrow">07 · Non-thinking-style case study</p><h2>一个 outcome-blind seed 从 item 1 走到 item 10</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>复制 Non-thinking 报告里最直观的“一条轨迹走到底”展示，检查 Native 的单个 item/carrier state 在丧失其余上下文后是否足以指定答案 <em>k</em>。</div><div><span class="experiment-label">设定</span>用 identity hash outcome-blind 冻结一个 N=10 confirmation case。Prompt records 与完整 trace context 都被等长 ordinary text 擦除；随后只恢复第 <em>k</em> 个 full item、grammar carrier，或等 token ordinary state。Answer query 始终不 patch。</div><div><span class="experiment-label">计算方法</span><span class="formula">E[count | condition,k] = Σ<sub>n=1</sub><sup>10</sup> n · P(answer=n | condition,k)</span>横轴为被恢复的 occurrence <em>k</em>，理想充分状态应产生 y=<em>k</em> 对角线。</div><div><span class="experiment-label">简单例子</span>当只恢复第 4 个 item 时，若该 state 是独立 counter，则 expected count 应靠近 4；若仍靠近 scrub baseline，则它需要其他 trace dynamics 才能被读出。</div></div>
<figure><h3 class="figure-title">图 6 · Full-context scrub 后的 expected-count path</h3>{walkthrough_svg(walkthrough)}<figcaption>橙色虚线是理想路径 y=k。Qwen 三条恢复路径大多停在高 count 区；Gemma 大多停在低 count 区。两者都没有形成 1→10 的对角路径。</figcaption></figure>
{walkthrough_table}
<div class="walkthrough-callout"><div><strong>控制成功。</strong><p>Clean case 均输出 10；uninformative baseline 分别退化为 Qwen candidate 9、Gemma candidate 1，且 P(10) 分别仅 {walkthrough['Qwen3-8B']['baselines']['uninformative']['gold_count_probability']:.4f} / {walkthrough['Gemma4-E4B']['baselines']['uninformative']['gold_count_probability']:.4f}。</p></div><div><strong>Restoration 失败。</strong><p>单 item hidden state 不能在被擦除的下游 trace 动力学中独立决定答案。V1 只擦 parsed items，遗漏 trace tail 的答案泄露，作为 failed-control audit 保留，不进入结果。</p></div></div>
<div class="section-conclusion"><strong>Experiment 7 结论。</strong> 这是 descriptive null，不做群体推断。在该两个冻结 case 中，Native count 更像需要后续 trace 配合传播的循环状态，而不是一枚在全上下文被擦除后仍能直接指定答案的独立 token code。</div></section>

<section id="comparison"><p class="eyebrow">08 · Mechanism synthesis</p><h2>最终机制图景</h2>
<div class="mechanism"><div class="stage"><span class="stage-no">01</span><h3>Retrieve</h3><p>Targeted heads 读取下一条 prompt city。Qwen 使用宽 Top-128；Gemma 使用窄 Top-6。</p><span class="evidence">causal necessity</span></div><div class="stage"><span class="stage-no">02</span><h3>Write & commit</h3><p>检索改变 grammar-specific carrier，carrier 将进度提交到 residual stream。</p><span class="evidence">deform + restore</span></div><div class="stage"><span class="stage-no">03</span><h3>Loop & read</h3><p>commit 改变下一次 targeted query；终端 state 再被 answer-time readout 使用。</p><span class="evidence">recurrent + conditional terminal</span></div></div>
{table(('Model','Strongest connected path','Remaining confound'),((
    'Qwen','Top-128 retrieval → carrier → commit → next query','terminal readout 只在固定 suffix 局部成立；free-running recovery 与 single-seed walk 不成立'),(
    'Gemma','Top-6 retrieval → carrier → commit → next query','|d|=1 orthogonal specificity 衰减；terminal 仅 controlled bridge；single-seed walk 不成立')))}
<p><strong>整体分析。</strong> 两模型共享同一计算抽象，但实现宽度不同：Qwen 使用宽 retrieval bank 和大效应 full commit；Gemma 使用窄 retrieval bank，commit→query 效应较小且对 local control 更敏感。这不要求两模型共享同一组 heads 或同一线性 counter。</p>
<div class="section-conclusion"><strong>与 Non-thinking 的核心差异。</strong> Non-thinking 更接近 answer-time broad aggregation + late write；Native-thinking 的主链先在 trace 内部做 targeted retrieval 与 recurrent state update，到 terminal 才由 answer readout 使用。两种模式都可在 answer query 表征 count，但 count 的形成和传递路径不同。</div></section>

<section id="appendix"><p class="eyebrow">Appendix · Negative and secondary evidence</p><h2>不改变主链、但必须保留的结果</h2>
<details class="appendix-block" open><summary>Appendix A · Gemma next-query 边的历史 directional 标签</summary><p>旧标签由 |d|=1 的 strong-direct gate 未满足引起。本版不删除该事实，而是把状态改为 confirmed†：直接效应在 prospective confirmation 成立，但 local matched-control specificity 衰减。正文图 4b 给出完整 dose 图：|d|=1 是冻结 primary，|d|=2/3 只是预注册 robustness，不能事后替换 primary。</p><div class="section-conclusion"><strong>Appendix A 结论。</strong> “†”限制的是局部 specificity 强度，不是否定 full commit state 对 next query 的直接因果效应。</div></details>

<details class="appendix-block" open><summary>Appendix B · Gemma query-mediation geometry ladder：full state 有效，但狭窄 Top-6 output mediation 未闭合</summary><p><strong>实验目的。</strong> 检验 commit state 的作用是否必须经由冻结 Top-6 targeted heads 的 pre-O outputs 传到 donor-successor city log-odds。按 outcome-blind 顺序检查 endpoint、suffix4、suffix8；每个 geometry 只跑 20 discovery seeds，只有 geometry_pass 才允许 confirmation。</p><p><strong>四个 estimands。</strong> A 是 full donor state 相对 self patch 的 intact 效应；B 是 selected-mask 对该 full-state 效应的削弱；C 在 selected-mask 条件下只恢复 selected-head pre-O outputs，看能否把 city log-odds救回；D 比较 selected mask 与同层同 K random mask。完整中介 gate 要求 A/B/C 的 95% CI 下界都大于 0；D 是 specificity diagnostic。</p>
<figure><h3 class="figure-title">图 B1 · Gemma endpoint→suffix4→suffix8 query-mediation ladder</h3>{effect_small_multiples_svg('Gemma query-mediation discovery ladder', mediation_panels)}<figcaption>每个 panel 横轴都是 donor−receiver query city log-odds effect，但四个 panel 为了显示数量级而使用独立对称尺度；点=20 discovery seeds 的 mean，线=95% CI，绿=该单项 gate 支持，橙=未满足。三行依次为 endpoint、suffix4、suffix8。Full-state effect 在三种 geometry 都大；selected-mask interaction 在较宽 suffix 上为正；关键的 selected-head pre-O restoration 仍触零，selected-vs-random specificity 也不稳定。</figcaption></figure>
<p><strong>结果与替代解释。</strong> suffix8 的 full-state effect 为 {ci(effect(gemma_query_mediation['suffix8']['estimands'], 'full_state_effect_intact'))}，selected-mask interaction 为 {ci(effect(gemma_query_mediation['suffix8']['estimands'], 'full_selected_mask_interaction'))}，但 pre-O restoration 只有 {ci(effect(gemma_query_mediation['suffix8']['estimands'], 'full_head_output_restore'))}；random mask interaction 也可为正。这与“Top-6 参与 routing”相容，却不足以证明所有可用 full-state 信息都串行穿过这 6 枚 heads。可能原因包括 head-output patch 的几何/时序不完整、selected bank 与旁路协同，或 layer-matched random 中也包含一般 query-support heads。</p><div class="section-conclusion"><strong>Appendix B 结论。</strong> 三个 geometry 均未达到 geometry_pass，confirmation 未开启。因此该 null 只否定当前狭窄 pre-O restoration 中介操作，不否定正文已经确认的 full commit→next-query 直接边或 Top-6 retrieval necessity。</div></details>

<details class="appendix-block" open><summary>Appendix C · Qwen free-running targeted-counter null：局部 margin 与自然答案充分性不同</summary><p><strong>实验目的。</strong> 将 fixed-suffix 的 terminal restoration 推广到更自由的 answer generation：要求 clean carrier restoration 同时把 count-candidate distribution、expected count、correct-count margin 与 greedy exact answer向 clean reference 拉回。20 discovery/10 confirmation 与 bank/position controls 均冻结。</p>
<figure><h3 class="figure-title">图 C1 · Qwen free-running clean-carrier restoration 的四个 answer endpoints</h3>{effect_small_multiples_svg('Qwen free-running targeted-counter confirmation null', q_free_panels)}<figcaption>四个 panel 分别显示 candidate-distribution TV recovery、expected-count recovery、correct-count margin recovery 与 greedy exact-count recovery。每个 panel 都有自己的对称横轴，因为量纲不同；点=10 confirmation seeds mean，线=95% CI。前三个 distribution/expectation quantities 接近数值零，margin 为 −0.075 [−0.500, +0.287]，greedy exact recovery 为 0。独立尺度只用于看每个 endpoint 是否离开 0，不能跨 panel 比条长。</figcaption></figure>
<p><strong>为何不是矛盾。</strong> Fixed-suffix patch 保留了后续 token trajectory，只问某个 terminal grammar state 能否局部提高 gold-number margin；free-running 实验允许后续状态、多个 trace sources、prompt retrieval 与自我校正共同决定答案。某个 ordinary-position control 可产生 margin 变化，也不等于 clean carrier 把完整 distribution 恢复到了 clean。</p><div class="section-conclusion"><strong>Appendix C 结论。</strong> Qwen 的 local terminal bridge 保留在正文，但自然 free-running count recovery 不成立；因此报告使用 controlled only，而不把 marker state 写成 answer 的独立充分统计量。</div></details>

<details class="appendix-block" open><summary>Appendix D · Single-seed V1 failed control：为何必须重做完整 context scrub</summary><p>V1 只擦除 parser 识别的 trace items，遗漏 trace tail；如果 tail 已含 final-count clue，那么单 item restore 的“无效”或“有效”都不可解释。V2 同时擦除 prompt records 与完整 trace context，并先验证 uninformative baseline 已显著退化。</p><figure><h3 class="figure-title">图 D1 · V1 与 V2 的 scrub coverage</h3>{failed_control_svg()}<figcaption>上排橙框是 V1 的泄露通路：parser item 之外的 trace tail 仍可被 answer query 读取。下排是 V2：等长 ordinary tokens 覆盖 prompt records 与整个 trace source，再测试单 item/carrier restoration。图示的是 control topology，不是定量 effect。</figcaption></figure><div class="section-conclusion"><strong>Appendix D 结论。</strong> V1 仅作 implementation audit；正文图 6 与 single-seed descriptive null 只使用通过 baseline-degradation 检查的 V2。</div></details>

<details class="appendix-block" open><summary>Appendix E · 其他 grammar 的 attention-map 对应版本（含 8 张图）</summary><p>正文使用冻结主线 <code>adjacent_rank_after_city</code>。下面把 same-unit rank-before-city 与 structural-invariant bullet 的两类图都直接展开：layer×head atlas 显示 head 在模型中的位置；ordinal×head map 显示正确 target ordinal 在 discovery-ranked heads 上的 attention mass。每张图独立缩放，不能用颜色深浅跨模型或 grammar 比较 raw mass；其作用是检查主线对角 pattern 是否只来自一种表面写法。</p>
<div class="attention-atlas-stack"><figure><h3 class="figure-title">图 E1 · Qwen same-unit rank-before-city · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_same_head']}</div><figcaption>横轴=head，纵轴=layer；颜色=该 grammar 的 discovery exact-P0 target mass。它回答“哪些 heads 亮”，不显示 target ordinal。</figcaption></figure><figure><h3 class="figure-title">图 E2 · Gemma same-unit rank-before-city · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_same_head']}</div><figcaption>坐标与 E1 同构，但 Gemma 有 42×8 heads，色标独立；不能与 Qwen 色值作绝对比较。</figcaption></figure>
<figure><h3 class="figure-title">图 E3 · Qwen same-unit rank-before-city · ordinal×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_same_ordinal']}</div><figcaption>横轴=discovery-ranked head，纵轴=正确 target record ordinal；颜色=seed-equal raw target mass。它检查同一 head bank 是否跨 occurrence 重复路由。</figcaption></figure><figure><h3 class="figure-title">图 E4 · Gemma same-unit rank-before-city · ordinal×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_same_ordinal']}</div><figcaption>轴定义同 E3；Gemma 图的 head 数与色标独立。</figcaption></figure>
<figure><h3 class="figure-title">图 E5 · Qwen structural-invariant bullet · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_bullet_head']}</div><figcaption>无显式 rank 的 bullet grammar；横轴=head、纵轴=layer。保留该图用于检验 selected heads 是否只锁定显式数字 marker。</figcaption></figure><figure><h3 class="figure-title">图 E6 · Gemma structural-invariant bullet · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_bullet_head']}</div><figcaption>轴定义同 E5，色标在 Gemma panel 内独立。</figcaption></figure>
<figure><h3 class="figure-title">图 E7 · Qwen structural-invariant bullet · ordinal×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_bullet_ordinal']}</div><figcaption>横轴=ranked head，纵轴=target ordinal；颜色=20 discovery seeds 的 seed-equal target mass。</figcaption></figure><figure><h3 class="figure-title">图 E8 · Gemma structural-invariant bullet · ordinal×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_bullet_ordinal']}</div><figcaption>轴定义同 E7；只比较空间 pattern，不比较跨模型 raw magnitude。</figcaption></figure></div>
<div class="section-conclusion"><strong>Appendix E 结论。</strong> Target-following attention 并非只在正文那一种 rank-after-city 文法中可见；但这些 map 仍是 descriptive localization，正式因果主张继续由冻结 bank ablation 支撑。</div></details>
<div class="section-conclusion"><strong>Appendix 结论。</strong> 这些 null 缩小了主张：我们确认一条可干预 recurrent pathway，但不确认单头排他中介、单 state 全局充分，或所有 grammar 与距离上完全同质的 circuit。</div></section>

<section id="audit"><p class="eyebrow">09 · Boundaries and reproducibility</p><h2>边界、复现与底层文件</h2>
<ul><li>本报告证明一条 pathway，不证明唯一性、排他性或所有 grammar 共用完全相同的 heads。</li><li>CI 与 p-value 保留用于审计；正文的“强/弱”判断同时考虑 effect size、控制组和跨 phase 复现。</li><li>单 seed walkthrough 不进入 discovery/confirmation gate；V2 是在 V1 暴露 trace-tail 泄露后修正的 exploratory control。</li><li>Qwen 与 Gemma 的状态几何、bank 宽度和最后一条边不同，不强行合并成完全同构 circuit。</li></ul>
<details class="paper-appendix"><summary>底层报告与外部证据包</summary><div class="source-list"><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">P0 targeted-retrieval atlas</a><br><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a><br><a href="NiaH_Geometry_Comparison.html">Representation geometry</a><br><span>逐 seed、逐 arm、claim-gate 与运行审计文件保存在外部实验归档，不随 Git 仓库分发。报告中的聚合值与输入哈希已冻结；复算时通过构建器参数挂载对应 evidence bundle。</span></div></details>
<p class="audit">Generated UTC: {esc(generated)}<br>Schema: realistic_niah_v5_native_thinking_restructured_v3</p></section>

</main></article><script>
{point_cloud_script(geometry_3d)}
document.querySelectorAll('[data-attention-selector]').forEach(function(selector) {{
  var container = selector.closest('.attention-switcher').querySelector('[data-attention-container]');
  function showSelected() {{
    container.querySelectorAll('[data-attention-example]').forEach(function(panel) {{
      panel.style.display = panel.dataset.attentionExample === selector.value ? 'block' : 'none';
    }});
  }}
  selector.addEventListener('change', showSelected);
  showSelected();
}});
</script></body></html>"""

    # Keep secondary and null-result material available without expanding the
    # initial reading path. Readers can open each appendix independently.
    html_text = html_text.replace(
        '<details class="appendix-block" open>',
        '<details class="appendix-block">',
    )

    input_paths = [
        args.reference_report,
        args.qwen_targeted_analysis,
        args.gemma_targeted_analysis,
        args.representation_root / "site_selected.csv",
        args.representation_root / "site_layer_candidates.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_candidate_metrics.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_candidate_metrics.csv",
        *(args.band_diagnostic_root / model / "band_diagnostic.json" for model in MODELS),
        args.geometry_comparison_report,
        args.atlas_root / "p0_head_atlas_manifest.json",
        args.atlas_root / "p0_targeted_retrieval_head_scores.csv",
        args.atlas_root / "Qwen3-8B_adjacent_rank_after_city_Top128_p0_attention_sum.svg",
        args.atlas_root / "Gemma4-E4B_adjacent_rank_after_city_Top6_p0_attention_sum.svg",
        *(Path(example["path"]) for example in attention_examples),
        *(args.atlas_root / filename for filename in (
            "Qwen3-8B_same_unit_rank_before_city_p0_head_map.svg",
            "Gemma4-E4B_same_unit_rank_before_city_p0_head_map.svg",
            "Qwen3-8B_structural_invariant_bullet_p0_head_map.svg",
            "Gemma4-E4B_structural_invariant_bullet_p0_head_map.svg",
        )),
        *(args.snapshot_root / "targeted_counter_write_20260822" / model / "complete.json" for model in MODELS),
        *(args.snapshot_root / "commit_state_query_20260822" / model / "commit_to_query_complete.json" for model in MODELS),
        args.snapshot_root / "qwen_grammar_span_decomposition_complete.json",
        args.snapshot_root / "gemma_grammar_span_decomposition_complete.json",
        args.snapshot_root / "targeted_counter_20260822" / "Qwen3-8B" / "targeted_counter_complete.json",
        *(args.snapshot_root / "single_seed_walkthrough_20260822_v2" / model / "analysis" / "walkthrough_complete.json" for model in MODELS),
        *(
            args.snapshot_root / f"gemma_query_mediation_{geometry}_discovery_claim_gates.json"
            for geometry in ("endpoint", "suffix4", "suffix8")
        ),
        args.snapshot_root / "gemma_query_mediation_complete.json",
        *(args.atlas_root / filename for filename in (
            "Qwen3-8B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg",
            "Gemma4-E4B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg",
            "Qwen3-8B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg",
            "Gemma4-E4B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg",
        )),
        *(Path(path) for evidence in token_evidence.values() for path in evidence["input_files"]),
    ]
    manifest = {
        "schema_version": "realistic_niah_v5_native_thinking_restructured_v3",
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
            "gemma_commit_to_query_direct_effect_confirmed": True,
            "gemma_commit_to_query_local_specificity_qualified": True,
            "gemma_narrow_pre_o_query_mediation_confirmed": False,
            "qwen_free_running_terminal_restoration_confirmed": False,
            "all_layer_pca3_is_descriptive": True,
        },
        "derived_display_data_sha256": {
            "geometry_3d": hashlib.sha256(
                json.dumps(geometry_3d, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "inputs_sha256": {str(path): sha256(path) for path in input_paths},
    }
    return html_text, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, default=Path("reports/former_report/NiaH_Non-thinking_report_frozen.html"))
    parser.add_argument("--qwen-targeted-analysis", type=Path, default=Path("reports/v5_native_final_localizers/analysis/qwen_final_merged_dose_grid.json"))
    parser.add_argument("--gemma-targeted-analysis", type=Path, default=Path("reports/v5_native_hybrid_supplement/Gemma4-E4B/analysis_hybrid_supplement_registered_v1/hybrid_dose_grid_complete.json"))
    parser.add_argument("--representation-root", type=Path, default=Path("reports/v5_native_causal_aligned_geometry"))
    parser.add_argument("--dual-endpoint-root", type=Path, default=Path("reports/v5_dual_endpoint_geometry_full300"))
    parser.add_argument("--band-diagnostic-root", type=Path, default=Path("reports/native_geometry_band_diagnostic_full300"))
    parser.add_argument("--geometry-comparison-report", type=Path, default=Path("reports/NiaH_Geometry_Comparison.html"))
    parser.add_argument("--atlas-root", type=Path, default=Path("reports/v5_native_p0_head_atlas"))
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
