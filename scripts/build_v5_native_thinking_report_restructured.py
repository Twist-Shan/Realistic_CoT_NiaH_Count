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
    width = 1040
    max_rows = max(len(rows) for _, _, rows, _ in panels)
    four_row_two_panel = len(panels) <= 2 and max_rows >= 4
    height = 390 if four_row_two_panel else (330 if len(panels) <= 2 else 600)
    legend_y = height - 20
    panel_w = 472
    panel_h = 285 if four_row_two_panel else 238
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
        last_row_y = y0 + 35 + (len(rows) - 1) * 42
        zero_line_bottom = max(y0 + 150, last_row_y + 10)

        def sx(value: float) -> float:
            return center + max(-limit, min(limit, value)) / limit * half

        parts.extend(
            [
                f'<rect x="{x0}" y="{y0-30}" width="{panel_w}" height="{panel_h}" fill="#fbfcfe" stroke="#d0d5dd"/>',
                f'<text x="{x0+14}" y="{y0-7}" class="heat-title">{esc(panel_title)}</text>',
                f'<line x1="{center:.1f}" y1="{y0+12}" x2="{center:.1f}" y2="{zero_line_bottom}" stroke="#98a2b3" stroke-dasharray="4 4"/>',
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
        tick_y = max(y0 + 177, last_row_y + 44)
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
            f'<circle cx="390" cy="{legend_y}" r="5" fill="#0f766e"/><text x="401" y="{legend_y+4}" class="chart-axis">registered CI entirely supportive</text>',
            f'<circle cx="650" cy="{legend_y}" r="5" fill="#d97706"/><text x="661" y="{legend_y+4}" class="chart-axis">gate not met / interval touches zero</text>',
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


def internal_counter_restoration_svg(
    occurrence_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Plot confirmation early-stop effects by restored occurrence k."""

    width, height = 1040, 390
    origins = ((70, 62), (565, 62))
    panel_w, panel_h = 400, 230
    panels = (
        ("A · Target-count margin gain", "mean target-margin gain", "mean_target_margin_gain"),
        ("B · Exact early-stop accuracy gain", "patched − scrub accuracy", "exact_gain"),
    )
    normalized: dict[str, list[dict[str, float]]] = {}
    for model, rows in occurrence_rows.items():
        normalized[model] = [
            {
                "k": float(row["target_occurrence"]),
                "mean_target_margin_gain": float(row["mean_target_margin_gain"]),
                "exact_gain": float(row["patched_exact_accuracy"])
                - float(row["baseline_exact_accuracy"]),
            }
            for row in rows
        ]
        require(
            [int(row["k"]) for row in normalized[model]] == list(range(2, 10)),
            f"{model} internal-counter occurrence support changed",
        )
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="No-running-index full-item restoration by early-stop target">'
    ]
    for panel_index, (title, y_label, key) in enumerate(panels):
        x0, y0 = origins[panel_index]
        values = [0.0] + [row[key] for rows in normalized.values() for row in rows]
        low, high = min(values), max(values)
        pad = max((high - low) * 0.16, 0.04 if key == "exact_gain" else 0.15)
        low, high = low - pad, high + pad

        def sx(k: float) -> float:
            return x0 + (k - 2.0) / 7.0 * panel_w

        def sy(value: float) -> float:
            return y0 + panel_h - (value - low) / (high - low) * panel_h

        parts.extend(
            [
                f'<text x="{x0}" y="30" class="heat-title">{esc(title)}</text>',
                f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfcfe" stroke="#d0d5dd"/>',
                f'<line x1="{x0}" y1="{sy(0):.1f}" x2="{x0+panel_w}" y2="{sy(0):.1f}" stroke="#98a2b3" stroke-dasharray="4 4"/>',
            ]
        )
        for tick_index in range(5):
            value = low + tick_index * (high - low) / 4
            y = sy(value)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:+.2f}</text>')
        for k in range(2, 10):
            x = sx(k)
            parts.append(f'<text x="{x:.1f}" y="{y0+panel_h+20}" text-anchor="middle" class="tick">{k}</text>')
        for model in MODELS:
            color = COLORS[model]
            rows = normalized[model]
            points = " ".join(f"{sx(row['k']):.1f},{sy(row[key]):.1f}" for row in rows)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6"/>')
            for row in rows:
                parts.append(
                    f'<circle cx="{sx(row["k"]):.1f}" cy="{sy(row[key]):.1f}" r="4" fill="{color}">'
                    f'<title>{esc(SHORT[model])}, k={int(row["k"])}: {row[key]:+.4f}</title></circle>'
                )
        parts.append(f'<text x="{x0+panel_w/2}" y="{height-48}" text-anchor="middle" class="axis-label">restored occurrence k</text>')
        parts.append(f'<text transform="translate({x0-54} {y0+panel_h/2}) rotate(-90)" text-anchor="middle" class="axis-label">{esc(y_label)}</text>')
    parts.extend(
        [
            '<line x1="380" y1="365" x2="405" y2="365" stroke="#0f766e" stroke-width="3"/><text x="412" y="369" class="legend-label">Qwen</text>',
            '<line x1="550" y1="365" x2="575" y2="365" stroke="#7c3aed" stroke-width="3"/><text x="582" y="369" class="legend-label">Gemma</text>',
            "</svg>",
        ]
    )
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
    grammar_anchor_paths = {
        "Qwen3-8B": args.snapshot_root / "qwen_grammar_span_anchor_panel.jsonl",
        "Gemma4-E4B": args.snapshot_root / "gemma_grammar_span_anchor_panel.jsonl",
    }
    grammar_anchors = {
        model: read_jsonl(path) for model, path in grammar_anchor_paths.items()
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
    ncc_supplement = {
        model: read_json(
            args.ncc_supplement_root / model / "ncc_analysis" / "claim_gates.json"
        )
        for model in MODELS
    }
    ncc_layerwise_diagnostic = {
        model: read_json(
            args.ncc_supplement_root
            / model
            / "ncc_analysis"
            / "layerwise_timing_diagnostic.json"
        )
        for model in MODELS
    }
    gemma_ncc_layerwise_diagnostic = ncc_layerwise_diagnostic["Gemma4-E4B"]
    stratified_ncc = {
        model: {
            timing: read_json(
                args.stratified_ncc_root
                / model
                / timing
                / "analysis"
                / "claim_gates.json"
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    stratified_ncc_synthesis = read_json(
        args.stratified_ncc_root
        / "synthesis"
        / "stratified_ncc_synthesis.json"
    )
    stratified_ncc_inputs = {
        model: read_json(
            args.stratified_ncc_root / model / "stratified_ncc_input_manifest.json"
        )
        for model in MODELS
    }
    stratified_ncc_site_audits = {
        model: {
            timing: read_json(
                args.stratified_ncc_root / model / f"{timing}_site_audit.json"
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    targeted_logit_margin = {
        model: {
            timing: read_json(
                args.logit_margin_root
                / model
                / timing
                / "analysis"
                / "claim_gates.json"
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    targeted_logit_margin_synthesis = read_json(
        args.logit_margin_root
        / "synthesis"
        / "targeted_logit_margin_synthesis.json"
    )
    unnumbered_counter = {
        model: read_json(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "claim_gates.json"
        )
        for model in MODELS
    }
    unnumbered_occurrence = {
        model: read_csv(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "occurrence_metrics.csv"
        )
        for model in MODELS
    }

    require(q_analysis.get("status") == "PASS" and g_analysis.get("status") == "PASS", "Targeted analyses must PASS")
    require(float(g_primary["selected_minus_random_failure_rate"]) > float(selected_row(g_analysis, 8)["selected_minus_random_failure_rate"]), "Gemma K6 must remain the frozen primary over K8")
    for model in MODELS:
        ncc = ncc_supplement[model]
        require(ncc.get("status") == "PASS", f"{model} NCC analysis not sealed")
        require(ncc.get("discovery_seed_count") == 20, f"{model} NCC discovery seed drift")
        require(ncc.get("confirmation_seed_count") == 10, f"{model} NCC confirmation seed drift")
        require(ncc.get("outcome_blind") is True, f"{model} NCC not outcome blind")
        require(ncc.get("selection_rank_used") is False, f"{model} NCC uses selection rank")
        counter = unnumbered_counter[model]
        require(counter.get("status") == "PASS", f"{model} unnumbered counter not sealed")
        require(counter.get("phase") == "confirmation", f"{model} unnumbered counter phase drift")
        require(counter.get("seed_count") == 10, f"{model} unnumbered counter seed drift")
        require(
            counter.get("trace_panel_kind") == "teacher_forced_unnumbered_gold_bullets",
            f"{model} unnumbered panel provenance drift",
        )
        require(counter.get("natural_generation_claim_allowed") is False, f"{model} counter claim overreach")
        require(len(unnumbered_occurrence[model]) == 8, f"{model} counter occurrence support drift")
        for evidence_name, evidence in (("write", write[model]), ("commit-query", commit_query[model])):
            require(evidence.get("status") == "PASS", f"{model} {evidence_name} not PASS")
            require(evidence.get("discovery_seed_count") == 20, f"{model} {evidence_name} discovery seed drift")
            require(evidence.get("confirmation_seed_count") == 10, f"{model} {evidence_name} confirmation seed drift")
            require(evidence.get("outcome_blind") is True, f"{model} {evidence_name} not outcome blind")
            require(evidence.get("selection_rank_used") is False, f"{model} {evidence_name} uses selection_rank")
        anchors = grammar_anchors[model]
        require(len(anchors) == 30, f"{model} grammar anchor panel must contain 30 traces")
        require(
            {int(row["seed"]) for row in anchors} == set(range(1234, 1264)),
            f"{model} grammar anchor seed contract changed",
        )
        require(
            all(bool(row["grammar_span_outcome_blind"]) for row in anchors),
            f"{model} grammar anchor panel is not outcome blind",
        )
        require(
            not any(bool(row["grammar_span_selection_rank_used"]) for row in anchors),
            f"{model} grammar anchor panel uses selection rank",
        )
        for phase, seeds, expected_per_timing in (
            ("discovery", set(range(1234, 1254)), 10),
            ("confirmation", set(range(1254, 1264)), 5),
        ):
            active = [row for row in anchors if int(row["seed"]) in seeds]
            require(
                len(active) == len(seeds),
                f"{model} {phase} grammar anchor count changed",
            )
            for timing in ("rank_after_city", "rank_before_city"):
                require(
                    sum(row["grammar_span_timing_stratum"] == timing for row in active)
                    == expected_per_timing,
                    f"{model} {phase} grammar timing balance changed",
                )
        require(grammar_span[model].get("status") == "PASS", f"{model} grammar span not PASS")
        require(grammar_span[model].get("discovery_seed_count") == 20, f"{model} grammar span discovery drift")
        require(grammar_span[model].get("confirmation_seed_count") == 10, f"{model} grammar span confirmation drift")
        require(walkthrough[model].get("status") == "PASS", f"{model} walkthrough not PASS")
        require(walkthrough[model].get("case_study_not_inferential") is True, f"{model} walkthrough scope drift")
        require(walkthrough[model].get("case_selected_by_outcome") is False, f"{model} walkthrough selected by outcome")
        require(walkthrough[model].get("answer_query_patched") is False, f"{model} walkthrough patched answer query")
    for model in MODELS:
        diagnostic = ncc_layerwise_diagnostic[model]
        require(diagnostic.get("status") == "PASS", f"{model} NCC layerwise diagnostic not sealed")
        require(diagnostic.get("model_label") == model, f"{model} NCC diagnostic label drift")
        require(
            diagnostic.get("inferential_status") == "post_hoc_diagnostic_not_confirmatory",
            f"{model} NCC diagnostic must remain explicitly post-hoc",
        )
        require(
            diagnostic.get("discovery_seed_count") == 20
            and diagnostic.get("confirmation_seed_count") == 10,
            f"{model} NCC diagnostic seed contract changed",
        )
        require(
            diagnostic["phase_audit"]["discovery"]["final_timing_histogram"]
            == {"rank_after_city": 10, "rank_before_city": 10}
            and diagnostic["phase_audit"]["confirmation"]["final_timing_histogram"]
            == {"rank_after_city": 5, "rank_before_city": 5},
            f"{model} NCC timing balance changed",
        )
        require(
            diagnostic["frozen_result_reproduction"]["selected_layer_matches"] is True
            and diagnostic["frozen_result_reproduction"]["within_tolerance_0_002"] is True,
            f"{model} frozen NCC result did not reproduce",
        )
        require(
            diagnostic["interpretation"]["cross_model_effect_size_comparison_allowed"] is False,
            f"{model} NCC diagnostic overclaims cross-model comparability",
        )
    require(
        stratified_ncc_synthesis.get("status") == "PASS"
        and stratified_ncc_synthesis.get("raw_margins_pooled") is False
        and stratified_ncc_synthesis.get(
            "directional_specific_support_requires_valid_clean_readout"
        )
        is True,
        "Timing-stratified NCC synthesis contract changed",
    )
    expected_stratified_counts = {
        "Qwen3-8B": {
            "rank_after_city": (19, 10),
            "rank_before_city": (19, 9),
        },
        "Gemma4-E4B": {
            "rank_after_city": (18, 9),
            "rank_before_city": (19, 10),
        },
    }
    expected_capture_start = {"Qwen3-8B": 35, "Gemma4-E4B": 30}
    for model in MODELS:
        inputs = stratified_ncc_inputs[model]
        require(inputs.get("status") == "FROZEN", f"{model} stratified NCC inputs not frozen")
        require(
            inputs.get("confirmation_used_for_bank_selection") is False
            and inputs.get("panel_selection_uses_model_outcomes") is False,
            f"{model} stratified NCC selection leakage",
        )
        for timing in ("rank_after_city", "rank_before_city"):
            result = stratified_ncc[model][timing]
            expected_dev, expected_confirmation = expected_stratified_counts[model][timing]
            require(result.get("status") == "PASS", f"{model}/{timing} stratified NCC not sealed")
            require(
                result.get("model_label") == model
                and result.get("timing_branch") == timing,
                f"{model}/{timing} stratified NCC label drift",
            )
            require(
                result.get("development_seed_count") == expected_dev
                and result.get("confirmation_seed_count") == expected_confirmation,
                f"{model}/{timing} stratified NCC seed drift",
            )
            require(
                result.get("confirmation_used_for_fit_or_layer_selection") is False
                and result.get("outcome_blind") is True
                and result.get("selection_rank_used") is False,
                f"{model}/{timing} stratified NCC fitting contract changed",
            )
            require(
                result.get("capture_layer_rule")
                == "strictly_above_all_ablated_head_layers",
                f"{model}/{timing} stratified NCC causal reach changed",
            )
            require(
                int(inputs["banks"][timing]["capture_start_layer"])
                == expected_capture_start[model],
                f"{model}/{timing} capture start changed",
            )
            site_audit = stratified_ncc_site_audits[model][timing]
            require(
                site_audit.get("status") == "PASS"
                and site_audit.get("all_final_endpoints_downstream_of_targeted_query")
                is True,
                f"{model}/{timing} site audit failed",
            )
            if timing == "rank_after_city":
                require(
                    site_audit.get("city_to_rank_marker_tokens_excluded") is True,
                    f"{model} City→rank marker leakage",
                )
    require(
        targeted_logit_margin_synthesis.get("status") == "PASS"
        and targeted_logit_margin_synthesis.get("all_four_clean_readouts_valid")
        is True
        and targeted_logit_margin_synthesis.get(
            "all_four_clean_candidate_accuracies_one"
        )
        is True
        and targeted_logit_margin_synthesis.get(
            "any_branch_passes_both_interval_gates"
        )
        is False
        and targeted_logit_margin_synthesis.get(
            "gemma_both_branches_positive_in_discovery_and_confirmation"
        )
        is True
        and targeted_logit_margin_synthesis.get(
            "gemma_both_specificity_bootstrap_intervals_exclude_zero"
        )
        is True
        and targeted_logit_margin_synthesis.get(
            "gemma_selected_loss_intervals_exclude_zero"
        )
        is False
        and targeted_logit_margin_synthesis.get(
            "qwen_has_no_directional_specific_branch"
        )
        is True,
        "Direct count-output margin synthesis contract changed",
    )
    for model in MODELS:
        for timing in ("rank_after_city", "rank_before_city"):
            result = targeted_logit_margin[model][timing]
            expected_dev, expected_confirmation = expected_stratified_counts[model][
                timing
            ]
            endpoint = result["primary_endpoint_result"]
            require(
                result.get("status") == "PASS"
                and result.get("model_label") == model
                and result.get("timing_branch") == timing,
                f"{model}/{timing} direct margin label or status drift",
            )
            require(
                result.get("schema_version")
                == "realistic_niah_v5_targeted_counter_logit_margin_analysis_v1"
                and result.get("primary_endpoint")
                == "final_answer_sequence_margin"
                and result.get("candidate_answer_scoring")
                == "full_autoregressive_sequence_log_probability_1_to_10",
                f"{model}/{timing} direct margin endpoint changed",
            )
            require(
                result.get("conditions")
                == [
                    "clean",
                    "selected_mask",
                    "random_mask_r1",
                    "random_mask_r2",
                    "random_mask_r3",
                ],
                f"{model}/{timing} direct margin arms changed",
            )
            require(
                result.get("outcome_blind_panel") is True
                and result.get("selection_rank_used") is False
                and result.get("no_decoder_fit_or_layer_selection") is True
                and result.get("raw_ncc_centroids_used") is False
                and result.get("raw_margins_pooled_across_branches") is False,
                f"{model}/{timing} direct margin analysis contract changed",
            )
            require(
                result.get("margin_gate_registered_before_logit_outcome_inspection")
                is True
                and result.get("confirmation_used_for_registration") is False
                and result.get("confirmation_status")
                == "registered_existing_split_after_ncc_inspection",
                f"{model}/{timing} direct margin registration status changed",
            )
            require(
                endpoint["development"]["seed_count"] == expected_dev
                and endpoint["confirmation"]["seed_count"]
                == expected_confirmation,
                f"{model}/{timing} direct margin seed contract changed",
            )
            require(
                endpoint["readout_validity"]["pass"] is True
                and float(endpoint["confirmation"]["clean_accuracy"]) == 1.0
                and float(endpoint["confirmation"]["clean_mean_margin"]) > 0.0,
                f"{model}/{timing} direct output readout validity failed",
            )
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
                "selected_carrier_deformation_specificity",
                "selected_boundary_deformation",
                "selected_boundary_deformation_specificity",
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
    write_damage_panels = [
        (
            "A · Selected mask vs clean",
            "mean downstream carrier RMS distance",
            [
                (SHORT[model], write_effects[model]["selected_carrier_deformation"])
                for model in MODELS
            ],
            0.07,
        ),
        (
            "B · Selected mask vs 3× random masks",
            "additional carrier RMS distance",
            [
                (
                    SHORT[model],
                    write_effects[model]["selected_carrier_deformation_specificity"],
                )
                for model in MODELS
            ],
            0.03,
        ),
    ]
    ncc_primary = {
        model: effect(
            ncc_supplement[model]["all_estimands"],
            "selected_correct_centroid_margin_loss",
        )
        for model in MODELS
    }
    ncc_random = {
        model: effect(
            ncc_supplement[model]["all_estimands"],
            "random_mean_correct_centroid_margin_loss",
        )
        for model in MODELS
    }
    ncc_specificity = {
        model: effect(
            ncc_supplement[model]["all_estimands"],
            "selected_vs_random_margin_loss_specificity",
        )
        for model in MODELS
    }
    ncc_panels = [
        (
            "A · Selected mask corrupts count geometry",
            "loss of correct-centroid margin",
            [(SHORT[model], ncc_primary[model]) for model in MODELS],
            40.0,
        ),
        (
            "B · Selected effect beyond random masks",
            "selected-minus-random margin loss",
            [(SHORT[model], ncc_specificity[model]) for model in MODELS],
            40.0,
        ),
    ]
    ncc_condition = {
        model: {
            str(row["condition"]): row
            for row in ncc_supplement[model]["condition_metrics"]
        }
        for model in MODELS
    }
    stratified_primary = {
        model: {
            timing: stratified_ncc[model][timing]["primary_endpoint_result"]
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }

    def stratified_plot_effect(
        model: str, timing: str, estimand_key: str
    ) -> dict[str, Any]:
        endpoint = stratified_primary[model][timing]
        value = dict(endpoint[estimand_key])
        value["gate_pass"] = bool(
            endpoint["readout_validity"]["pass"]
            and float(value["ci_low"]) > 0.0
        )
        return value

    stratified_ncc_panels = [
        (
            "A · Selected mask vs clean",
            "selected loss / discovery OOF SD",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    stratified_plot_effect(
                        model, timing, "standardized_primary_estimand"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.01,
        ),
        (
            "B · Selected effect beyond random masks",
            "specificity / discovery OOF SD",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    stratified_plot_effect(
                        model, timing, "standardized_specificity_estimand"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.01,
        ),
    ]
    stratified_status_text = {
        "NO_DIRECTIONAL_SPECIFIC_EVIDENCE": "未见 directional-specific damage",
        "UNINTERPRETABLE_MARGIN_SHIFT_READOUT_VALIDITY_FAILURE": (
            "margin shift 为正，但 clean readout validity 失败"
        ),
        "VALID_READOUT_DIRECTIONAL_SPECIFIC_EVIDENCE": (
            "readout 有效；方向与 specificity 为正，但区间跨 0"
        ),
        "VALID_READOUT_INTERVAL_DIRECTIONAL_SPECIFIC_SUPPORT": (
            "readout 有效且两个区间均高于 0"
        ),
    }
    stratified_ncc_result_table = table(
        (
            "Model / timing",
            "Primary endpoint / layer",
            "Discovery / confirmation seeds",
            "OOF BA / clean confirmation exact",
            "Clean mean margin / validity",
            "Selected loss: raw / SD units [95% CI]",
            "Selected−random: raw / SD units [95% CI]",
            "Qualified result",
        ),
        (
            (
                f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                (
                    f"{stratified_ncc[model][timing]['primary_endpoint']} · "
                    f"L{int(stratified_primary[model][timing]['selected_layer'])}"
                ),
                (
                    f"{int(stratified_ncc[model][timing]['development_seed_count'])} / "
                    f"{int(stratified_ncc[model][timing]['confirmation_seed_count'])}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['selected_layer_discovery_metrics']['grouped_oof_ncc_balanced_accuracy']):.3f} / "
                    f"{float(next(row for row in stratified_primary[model][timing]['condition_metrics'] if row['condition'] == 'clean')['exact_accuracy']):.3f}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f} / "
                    f"{'PASS' if stratified_primary[model][timing]['readout_validity']['pass'] else 'FAIL'}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['raw_primary_estimand']['mean_effect']):+.2f} / "
                    f"{ci(stratified_primary[model][timing]['standardized_primary_estimand'], 4)}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['raw_specificity_estimand']['mean_effect']):+.2f} / "
                    f"{ci(stratified_primary[model][timing]['standardized_specificity_estimand'], 4)}"
                ),
                stratified_status_text[
                    stratified_primary[model][timing]["ncc_effect_status"]
                ],
            )
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
    )
    stratified_ncc_bank_table = table(
        (
            "Model",
            "City→rank bank",
            "Rank→city bank",
            "Bank overlap",
            "Causally reachable readout layers",
        ),
        (
            (
                SHORT[model],
                (
                    f"Top-{int(stratified_ncc_inputs[model]['banks']['rank_after_city']['bank_size'])}; "
                    f"{stratified_ncc_inputs[model]['banks']['rank_after_city']['source_grammar']}; P0"
                ),
                (
                    f"Top-{int(stratified_ncc_inputs[model]['banks']['rank_before_city']['bank_size'])}; "
                    f"{stratified_ncc_inputs[model]['banks']['rank_before_city']['source_grammar']}; "
                    f"{'post-marker' if model == 'Qwen3-8B' else 'P0'}"
                ),
                (
                    f"{int(stratified_ncc_inputs[model]['selected_bank_overlap']['head_count'])}/"
                    f"{int(stratified_ncc_inputs[model]['banks']['rank_after_city']['bank_size'])}"
                ),
                "L35" if model == "Qwen3-8B" else "L30–L41（discovery 内选层）",
            )
            for model in MODELS
        ),
    )
    direct_margin_primary = {
        model: {
            timing: targeted_logit_margin[model][timing]["primary_endpoint_result"]
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    direct_margin_local = {
        model: targeted_logit_margin[model]["rank_after_city"][
            "endpoint_results"
        ]["local_rank_adjacent_sequence_margin"]
        for model in MODELS
    }

    def direct_margin_plot_effect(
        model: str, timing: str, estimand_key: str
    ) -> dict[str, Any]:
        endpoint = direct_margin_primary[model][timing]
        value = dict(endpoint["confirmation"][estimand_key])
        value["gate_pass"] = bool(
            endpoint["readout_validity"]["pass"]
            and float(value["ci_low"]) > 0.0
        )
        return value

    direct_margin_panels = [
        (
            "A · Selected mask vs clean",
            "final count-margin loss",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    direct_margin_plot_effect(
                        model, timing, "selected_margin_loss"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.6,
        ),
        (
            "B · Selected effect beyond random masks",
            "selected−random count-margin loss",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    direct_margin_plot_effect(
                        model, timing, "selected_vs_random_specificity"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.6,
        ),
    ]
    direct_margin_status_text = {
        "NO_DIRECTIONAL_SPECIFIC_EVIDENCE": "无 specificity",
        "VALID_READOUT_DIRECTIONAL_SPECIFIC_EVIDENCE": "directional-specific†",
        "VALID_READOUT_INTERVAL_DIRECTIONAL_SPECIFIC_SUPPORT": (
            "interval-confirmed"
        ),
    }
    direct_margin_result_table = table(
        (
            "Model / timing",
            "n: dev / conf",
            "Clean M / acc.",
            "Dev loss / spec.",
            "Conf. loss [95% CI]",
            "Conf. spec. [95% CI]",
            "Gate",
        ),
        (
            (
                f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                (
                    f"{int(direct_margin_primary[model][timing]['development']['seed_count'])} / "
                    f"{int(direct_margin_primary[model][timing]['confirmation']['seed_count'])}"
                ),
                (
                    f"{float(direct_margin_primary[model][timing]['confirmation']['clean_mean_margin']):+.3f} / "
                    f"{float(direct_margin_primary[model][timing]['confirmation']['clean_accuracy']):.3f}"
                ),
                (
                    f"{float(direct_margin_primary[model][timing]['development']['selected_margin_loss']['mean_effect']):+.3f} / "
                    f"{float(direct_margin_primary[model][timing]['development']['selected_vs_random_specificity']['mean_effect']):+.3f}"
                ),
                ci(
                    direct_margin_primary[model][timing]["confirmation"][
                        "selected_margin_loss"
                    ]
                ),
                ci(
                    direct_margin_primary[model][timing]["confirmation"][
                        "selected_vs_random_specificity"
                    ]
                ),
                direct_margin_status_text[
                    direct_margin_primary[model][timing]["effect_status"]
                ],
            )
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
    )
    ncc_result_table = table(
        (
            "Model",
            "Frozen layer",
            "Discovery grouped-OOF NCC BA",
            "Confirmation exact: clean → selected",
            "Selected margin loss",
            "Random mean loss",
            "Selected−random",
        ),
        (
            (
                SHORT[model],
                f"L{int(ncc_supplement[model]['selected_layer'])}",
                f"{float(ncc_supplement[model]['selected_layer_discovery_metrics']['mean_timing_oof_ncc_balanced_accuracy']):.3f}",
                f"{float(ncc_condition[model]['clean']['exact_accuracy']):.2f} → {float(ncc_condition[model]['selected_mask']['exact_accuracy']):.2f}",
                ci(ncc_primary[model]),
                ci(ncc_random[model]),
                ci(ncc_specificity[model]),
            )
            for model in MODELS
        ),
    )
    ncc_diagnostic_by_model_layer_timing = {
        model: {
            (int(row["layer"]), str(row["timing"])): row
            for row in ncc_layerwise_diagnostic[model]["layer_timing_rows"]
        }
        for model in MODELS
    }
    qwen_ncc_diagnostic_by_layer_timing = ncc_diagnostic_by_model_layer_timing["Qwen3-8B"]
    gemma_ncc_diagnostic_by_layer_timing = ncc_diagnostic_by_model_layer_timing["Gemma4-E4B"]

    def count_histogram_text(model: str) -> str:
        histogram = ncc_layerwise_diagnostic[model]["phase_audit"]["confirmation"][
            "gold_count_histogram"
        ]
        return " · ".join(
            f"N={count}×{histogram[count]}" for count in sorted(histogram, key=int)
        )

    ncc_correspondence_table = table(
        ("Audit dimension", "Qwen3-8B", "Gemma4-E4B", "Cross-model reading"),
        (
            (
                "Offline reproduction",
                (
                    f"L{int(ncc_layerwise_diagnostic['Qwen3-8B']['frozen_result_reproduction']['selected_layer_recomputed'])}; "
                    f"max |Δ|={float(ncc_layerwise_diagnostic['Qwen3-8B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f}"
                ),
                (
                    f"L{int(ncc_layerwise_diagnostic['Gemma4-E4B']['frozen_result_reproduction']['selected_layer_recomputed'])}; "
                    f"max |Δ|={float(ncc_layerwise_diagnostic['Gemma4-E4B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f}"
                ),
                "PASS：各模型冻结结果可复现",
            ),
            (
                "Seed/timing contract",
                "20 discovery（10/10）· 10 confirmation（5/5）",
                "20 discovery（10/10）· 10 confirmation（5/5）",
                "协议与 timing 配额对齐",
            ),
            (
                "Confirmation gold N",
                count_histogram_text("Qwen3-8B"),
                count_histogram_text("Gemma4-E4B"),
                "不匹配；不是跨模型 paired cohort",
            ),
            (
                "Frozen targeted bank",
                (
                    f"Top-{int(ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['bank_size'])}; "
                    f"{ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['selection_target_grammar_class']}; "
                    f"{ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['selection_anchor_role']}"
                ),
                (
                    f"Top-{int(ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['bank_size'])}; "
                    f"{ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['selection_target_grammar_class']}; "
                    f"{ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['selection_anchor_role']}"
                ),
                "功能类比，但 bank 剂量与定位点不同",
            ),
            (
                "Frozen layer / causally active selected heads",
                (
                    f"L{int(ncc_layerwise_diagnostic['Qwen3-8B']['selection_audit']['original_pooled_timing_layer'])}; "
                    f"{int(ncc_layerwise_diagnostic['Qwen3-8B']['selection_audit']['original_pooled_timing_active_selected_heads'])}/"
                    f"{int(ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['bank_size'])}"
                ),
                (
                    f"L{int(ncc_layerwise_diagnostic['Gemma4-E4B']['selection_audit']['original_pooled_timing_layer'])}; "
                    f"{int(ncc_layerwise_diagnostic['Gemma4-E4B']['selection_audit']['original_pooled_timing_active_selected_heads'])}/"
                    f"{int(ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['bank_size'])}"
                ),
                "有效 intervention dose 不对齐",
            ),
            (
                "NCC coordinates",
                "每 timing 独立 standardize + PCA-16 + centroids",
                "每 timing 独立 standardize + PCA-16 + centroids",
                "方法相同；raw margins 不同尺度",
            ),
        ),
    )
    ncc_frozen_timing_table = table(
        (
            "Model",
            "Frozen layer",
            "Timing",
            "Discovery OOF BA",
            "Clean margin mean",
            "Selected loss",
            "Selected−random",
            "Positive seeds",
        ),
        (
            (
                SHORT[model],
                f"L{int(ncc_supplement[model]['selected_layer'])}",
                "City→rank" if timing == "rank_after_city" else "Rank→city",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['discovery_grouped_oof_balanced_accuracy']):.3f}",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['clean_margin_mean']):.1f}",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['selected_margin_loss_mean']):+.2f}",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['selected_vs_random_specificity_mean']):+.2f}",
                f"{int(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['selected_margin_loss_positive_n'])}/5",
            )
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
    )
    qwen_ncc_l23_after = qwen_ncc_diagnostic_by_layer_timing[(23, "rank_after_city")]
    qwen_ncc_l23_before = qwen_ncc_diagnostic_by_layer_timing[(23, "rank_before_city")]
    gemma_ncc_diagnostic_layers = (16, 18, 24, 27, 28, 30, 34)
    gemma_ncc_causal_reach = {
        16: "mask 尚未可达",
        18: "L17 heads 首次可达",
        24: "仅 L17 heads",
        27: "仅 L17 heads；discovery-best",
        28: "仅 L17 heads；相邻稳定层",
        30: "L17+L29 首次可达",
        34: "full Top-6；冻结 pooled primary",
    }
    for layer in gemma_ncc_diagnostic_layers:
        for timing in ("rank_after_city", "rank_before_city"):
            require(
                (layer, timing) in gemma_ncc_diagnostic_by_layer_timing,
                f"Missing Gemma NCC diagnostic L{layer} {timing}",
            )
    gemma_ncc_layerwise_table = table(
        (
            "Layer",
            "query-mask causal reach",
            "City→rank: OOF BA / selected loss",
            "Rank→city: OOF BA / selected loss",
            "Rank→city selected−random",
            "Rank→city positive seeds",
        ),
        (
            (
                f"L{layer}",
                gemma_ncc_causal_reach[layer],
                (
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_after_city')]['discovery_grouped_oof_balanced_accuracy']):.3f} / "
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_after_city')]['selected_margin_loss_mean']):+.2f}"
                ),
                (
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['discovery_grouped_oof_balanced_accuracy']):.3f} / "
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['selected_margin_loss_mean']):+.2f}"
                ),
                f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['selected_vs_random_specificity_mean']):+.2f}",
                f"{int(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['selected_margin_loss_positive_n'])}/5",
            )
            for layer in gemma_ncc_diagnostic_layers
        ),
    )
    gemma_ncc_l27_before = gemma_ncc_diagnostic_by_layer_timing[(27, "rank_before_city")]
    gemma_ncc_l28_before = gemma_ncc_diagnostic_by_layer_timing[(28, "rank_before_city")]
    gemma_ncc_l34_after = gemma_ncc_diagnostic_by_layer_timing[(34, "rank_after_city")]
    gemma_ncc_l34_before = gemma_ncc_diagnostic_by_layer_timing[(34, "rank_before_city")]
    counter_margin = {
        model: effect(unnumbered_counter[model]["primary_estimands"], "target_margin_gain")
        for model in MODELS
    }
    counter_exact = {
        model: effect(unnumbered_counter[model]["primary_estimands"], "exact_accuracy_gain")
        for model in MODELS
    }
    counter_mae = {
        model: effect(unnumbered_counter[model]["primary_estimands"], "mae_reduction")
        for model in MODELS
    }
    counter_result_table = table(
        (
            "Model",
            "Frozen source layer",
            "Scrub → patched exact",
            "Exact gain",
            "MAE reduction",
            "Target-margin gain",
            "Old-HTML magnitude gate",
        ),
        (
            (
                SHORT[model],
                f"L{int(unnumbered_counter[model]['selected_layer'])}",
                f"{float(unnumbered_counter[model]['selected_layer_metrics']['baseline_exact_accuracy']):.3f} → {float(unnumbered_counter[model]['selected_layer_metrics']['patched_exact_accuracy']):.3f}",
                ci(counter_exact[model]),
                ci(counter_mae[model]),
                ci(counter_margin[model]),
                "PASS" if unnumbered_counter[model]["old_html_internal_counter_magnitude_pass"] else "not met",
            )
            for model in MODELS
        ),
    )
    write_mediation_panels = [
        (
            "A · Damage reaches the later commit",
            "final-boundary RMS distance from clean",
            [
                (SHORT[model], write_effects[model]["selected_boundary_deformation"])
                for model in MODELS
            ],
            0.09,
        ),
        (
            "B · Clean carrier rescues that damage",
            "reduction in final-boundary RMS distance",
            [
                (SHORT[model], write_effects[model]["clean_carrier_restoration"])
                for model in MODELS
            ],
            0.06,
        ),
    ]
    write_recovery_fraction = {
        model: float(write_effects[model]["clean_carrier_restoration"]["mean_effect"])
        / float(write_effects[model]["selected_boundary_deformation"]["mean_effect"])
        for model in MODELS
    }
    query_groups = [
        ("full commit vs self patch", {model: float(query_effects[model]["self"]["mean_effect"]) for model in MODELS}),
        ("full commit vs orthogonal", {model: float(query_effects[model]["orthogonal"]["mean_effect"]) for model in MODELS}),
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
.report-note{max-width:920px;color:#475467}.status-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}.status-card{padding:18px;border:1px solid var(--line);background:#fbfcfe}.status-card h3{margin:0 0 8px}.status-card p{margin:6px 0;font-size:14px}.status-good{color:#075e58;font-weight:750}.status-open{color:#9a4b00;font-weight:750}.chain-figure{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fbfcfe}.chain-title{fill:#172033;font-size:13px;font-weight:750}.chain-sub{fill:#667085;font-size:11px}.chain-model{fill:#344054;font-size:12px;font-weight:750}.chain-status{font-size:11px;font-weight:750}.mini-model{font-size:11px;font-weight:800}.metric-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}.metric{padding:15px;border-top:3px solid var(--teal);background:#f8fafc}.metric strong,.metric span{display:block}.metric strong{font-size:22px}.metric span{color:#667085;font-size:12px}.negative-result{padding:17px 19px;border-left:4px solid var(--amber);background:#fff8eb}.audit-list{font-size:12px;color:#667085;overflow-wrap:anywhere}.compact-table td,.compact-table th{padding:7px 8px}.walkthrough-callout{display:grid;grid-template-columns:1fr 1fr;gap:14px}.walkthrough-callout>div{padding:15px;border:1px solid var(--line);background:#fbfcfe}.edge-roadmap{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;margin:18px 0 28px;border:1px solid var(--line);background:var(--line)}.edge-roadmap>div{padding:16px 17px;background:#fbfcfe}.edge-roadmap strong,.edge-roadmap span{display:block}.edge-roadmap span{margin-bottom:5px;color:#0f766e;font:800 11px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.condition-list{margin:14px 0 22px;border-top:1px solid var(--line)}.condition-row{display:grid;grid-template-columns:170px 1fr 1fr;gap:16px;padding:12px 4px;border-bottom:1px solid var(--line);font-size:13px;line-height:1.55}.condition-row strong{color:#172033}.condition-row span{color:#475467}.plain-language{margin:16px 0;padding:16px 18px;border:1px solid #b8d7d1;background:#f4fbf9}.plain-language strong{color:#075e58}.scope-compare{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}.scope-compare>div{padding:16px 18px;border-top:3px solid #98a2b3;background:#f8fafc}.scope-compare>div:first-child{border-top-color:#0f766e}.scope-compare h4{margin:0 0 7px}.scope-compare p{margin:6px 0;font-size:13px}.appendix-e-index{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:14px 0 20px}.appendix-e-index div{padding:12px 14px;border-left:3px solid #46758f;background:#f5f8fb;font-size:12px;line-height:1.55}.appendix-e-index strong{display:block;color:#172033}.appendix-e-figure{scroll-margin-top:24px}.appendix-e-figure .attention-atlas-frame>svg{display:block;width:100%;height:auto;margin:0 auto}.appendix-e-figure .attention-atlas-frame>.ordinal-map{min-width:1100px}.appendix-e-proof{margin:10px 0;color:#475467;font-size:12px}
.definition-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:18px 0 28px}.definition{padding:15px 16px;border:1px solid var(--line);background:#fbfcfe}.definition dt{font-weight:800;color:#172033;margin-bottom:6px}.definition dd{margin:0;color:#475467;font-size:13px;line-height:1.62}.experiment-frame{margin:18px 0 26px;border:1px solid var(--line);background:#fff}.experiment-frame>div{padding:15px 18px;border-bottom:1px solid var(--line)}.experiment-frame>div:last-child{border-bottom:0}.experiment-label{display:inline-block;min-width:88px;color:#0f766e;font-size:11px;font-weight:850;letter-spacing:.07em;text-transform:uppercase}.formula{display:block;margin:9px 0 0;padding:15px 18px;background:#f5f8fb;border-left:3px solid #46758f;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;overflow-x:auto}.figure-primer{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;margin:16px 0 8px;background:var(--line);border:1px solid var(--line)}.figure-primer>div{padding:13px 15px;background:#f8fafc;font-size:12px;line-height:1.55}.figure-primer strong{display:block;margin-bottom:4px;color:#172033}.paper-chart{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}.three-d{margin:12px 0 0;border:1px solid var(--line);background:linear-gradient(#fbfcfe,#f5f8fb)}.three-d-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:13px 15px;border-bottom:1px solid var(--line)}.three-d-controls{display:flex;align-items:end;justify-content:flex-end;gap:10px;flex-wrap:wrap}.three-d-head label,.manifold-panel-head label{color:#475467;font-size:12px;font-weight:700}.three-d-head select,.three-d-head button,.manifold-panel-head select{display:block;margin-top:5px;padding:7px 9px;border:1px solid #b8c1cf;background:#fff;color:#172033;font:inherit}.three-d-head button{cursor:pointer}.manifold-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}.manifold-panel{min-width:0;padding:0;background:#fbfcfe;border:0}.manifold-panel-head{display:flex;align-items:end;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid var(--line)}.manifold-panel-head strong,.manifold-panel-head span{display:block}.manifold-panel-head span{margin-top:3px;color:#667085;font-size:12px}.three-d canvas{display:block;width:100%;height:455px;cursor:grab;touch-action:none}.three-d canvas:active{cursor:grabbing}.manifold-stats{min-height:45px;margin:0;padding:9px 13px;border-top:1px solid var(--line);color:#667085;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.attention-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.attention-pair figure{margin:0}.attention-pair img{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}.attention-atlas-stack{display:grid;grid-template-columns:1fr;gap:26px;margin-top:12px}.attention-atlas-stack figure{margin:0;padding:16px;border:1px solid var(--line);background:#fff}.attention-atlas-frame{width:100%;overflow-x:auto}.attention-atlas-frame .head-map{display:block;width:100%;min-width:900px;height:auto;margin:0 auto}.attention-switcher{margin:14px 0;padding:16px;border:1px solid var(--line);background:#fbfcfe}.attention-select{display:block;max-width:680px;color:#344054;font-size:12px;font-weight:750}.attention-select select{display:block;width:100%;margin-top:7px;padding:9px 11px;border:1px solid #b8c1cf;background:#fff;color:#172033;font:inherit}.attention-example-panel{margin-top:16px}.attention-example-svg{overflow-x:auto}.attention-example-svg svg{display:block;width:100%;min-width:900px;height:auto;margin:0 auto}.map-meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 9px;color:#667085;font-size:12px}.map-meta strong{color:#172033;font-size:14px}.head-map{display:block;width:100%;height:auto;background:#fff}.term-note{font-size:12px;color:#667085}.qualification{padding:16px 18px;border-left:4px solid #0f766e;background:#f0f9f7}.appendix-block{margin-top:22px}.appendix-block summary{cursor:pointer;font-weight:800}.section-conclusion{margin-top:22px;padding:17px 19px;background:#eef7f5;border-left:4px solid #0f766e}.section-conclusion strong{color:#075e58}
.attention-pair svg{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}
@media print{.attention-pair{display:block}.attention-pair figure{break-inside:avoid-page;margin:0 0 20px}.attention-pair figure .head-map{width:auto;max-width:100%;max-height:620px;margin:0 auto}.attention-atlas-stack figure{break-inside:avoid-page}.attention-atlas-frame .head-map{min-width:0}.three-d{break-inside:avoid-page}.three-d canvas{height:430px}.formula{white-space:normal}.attention-switcher{break-inside:avoid-page}}
@media(max-width:900px){.manifold-grid{grid-template-columns:1fr}}
@media(max-width:760px){.status-grid,.walkthrough-callout,.metric-strip,.definition-grid,.attention-pair,.figure-primer,.edge-roadmap,.scope-compare,.appendix-e-index{grid-template-columns:1fr}.condition-row{grid-template-columns:1fr;gap:4px}.three-d-head{align-items:flex-start;flex-direction:column}.three-d-controls{justify-content:flex-start}.three-d canvas{height:430px}.chain-figure{min-width:850px}.chain-scroll{overflow-x:auto}}
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
        ("属于哪一步", "Qwen confirmation", "Gemma confirmation", "它单独回答什么"),
        (
            (
                "5.1 direct damage · selected mask → carrier",
                ci(write_effects["Qwen3-8B"]["selected_carrier_deformation"]),
                ci(write_effects["Gemma4-E4B"]["selected_carrier_deformation"]),
                "关掉 selected bank 后，query 之后的 carrier 是否离开 clean state",
            ),
            (
                "5.2 main rescue · clean carrier → commit",
                ci(write_effects["Qwen3-8B"]["clean_carrier_restoration"]),
                ci(write_effects["Gemma4-E4B"]["clean_carrier_restoration"]),
                "保持同一 head damage，只补回 clean carrier 能否把 commit 拉回 clean",
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
        ),
    )
    query_sample_table = table(
        (
            "Model / phase",
            "独立样本",
            "每个 seed 的 donor–receiver pairs",
            "正式 pair 数",
            "patch layer / readout bank",
        ),
        (
            (
                f"{SHORT[model]} · {phase}",
                f"{20 if phase == 'discovery' else 10} seeds × 1 trace/seed",
                "d = −3, −2, −1, +1, +2, +3（每个 signed offset 1 pair）",
                str((20 if phase == "discovery" else 10) * 6),
                (
                    "L19 / frozen Top-128"
                    if model == "Qwen3-8B"
                    else "L16 / frozen Top-6"
                ),
            )
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
    )
    query_arm_table = table(
        ("每个 donor–receiver pair 的 condition", "receiver commit 中写入什么", "该 arm 回答什么"),
        (
            ("Clean", "不调用 patch hook", "保留自然轨迹作审计参照；不进入正文两个主差分"),
            ("Self patch", "把 receiver 自己的完整 post-block vector 写回", "控制 hook、复制和写回本身"),
            ("Full donor patch", "把 donor commit 的完整 post-block vector 写到 receiver", "完整 commit state 是否足以重定向下一 query"),
            ("Count-subspace transplant", "只移植 donor−receiver delta 在 discovery-frozen count subspace 上的投影", "窄线性 count component 是否足够"),
            ("Norm-matched orthogonal patch", "写入与上述 count 投影等范数、但对该 subspace 正交的扰动", "同尺度的一般 state 扰动能否解释结果"),
        ),
    )
    def anchor_phase(model: str, phase: str) -> list[dict[str, Any]]:
        seeds = (
            set(range(1234, 1254))
            if phase == "discovery"
            else set(range(1254, 1264))
        )
        return [row for row in grammar_anchors[model] if int(row["seed"]) in seeds]

    def count_composition(rows: Sequence[Mapping[str, Any]]) -> str:
        counts = sorted({int(row["gold_count"]) for row in rows})
        return " · ".join(
            f"N={count} × {sum(int(row['gold_count']) == count for row in rows)}"
            for count in counts
        )

    write_sample_table = table(
        ("Model / phase", "独立 trace 样本", "实际 gold-count 组成", "grammar timing"),
        (
            (
                f"{SHORT[model]} · {phase}",
                f"{len(anchor_phase(model, phase))} seeds × 1 trace/seed",
                count_composition(anchor_phase(model, phase)),
                (
                    "city→rank 10；rank→city 10"
                    if phase == "discovery"
                    else "city→rank 5；rank→city 5"
                ),
            )
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
    )
    token_condition_table = table(
        ("图 5a panel / 横轴", "实际被置零的 token states", "为什么要放这个条件"),
        (
            ("左 · clean", "不置零", "给出自然 next-city retrieval 基线"),
            ("左 · early half", "当前 query 可见的已完成 items 中，最早 floor(k/2) 个 item 的 trace states", "检验较远的早期历史是否仍直接参与当前 routing"),
            ("左 · cumulative", "从 trace 起点到最近一个已完成 item 之前的全部 trace states；保留最近 transition", "把长期累计历史与最近一次局部更新分开"),
            ("左 · recent", "最近一个已完成 item 起点到当前 query 之前的 trace states；保留更早历史", "检验当前 query 是否主要由最近一次 transition 触发"),
            ("左 · full trace", "prompt 结束后到当前 query 之前的全部 trace states", "测量早期历史与最近 transition 合在一起的总必要性"),
            ("右 · clean", "不置零", "给出自然 greedy exact-count accuracy"),
            ("右 · prompt records", "只置零原 prompt 中 parser-registered needle / city records；保留其他 prompt 与完整 trace", "检验已有 trace 时，answer 是否仍必须回原始 records 重数"),
            ("右 · full trace", "置零 prompt 后生成的整段 trace；保留完整 prompt", "检验 answer 是否必须读取 trace 已写好的状态"),
            ("右 · full prompt + trace", "除 BOS/system boundary 与 answer query 外，置零完整 prompt 与完整 trace", "给出两类信息源同时移除的 floor / sanity control"),
        ),
    )
    terminal_sample_table = table(
        (
            "Model / phase",
            "独立 trace 样本",
            "实际 gold-count 组成",
            "grammar timing",
            "source layer / condition rows",
        ),
        (
            (
                f"{SHORT[model]} · {phase}",
                f"{len(anchor_phase(model, phase))} seeds × 1 trace/seed",
                count_composition(anchor_phase(model, phase)),
                (
                    "city→rank 10；rank→city 10"
                    if phase == "discovery"
                    else "city→rank 5；rank→city 5"
                ),
                (
                    f"L{19 if model == 'Qwen3-8B' else 16}→final block / "
                    f"{len(anchor_phase(model, phase)) * 12} rows"
                ),
            )
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
    )
    terminal_geometry_table = table(
        ("Geometry", "最后一个 item 内的 receiver positions", "它区分的解释"),
        (
            ("Full item", "parser-registered 最后一个完整 trace item", "可恢复上界；不用于选择最小 carrier"),
            ("Marker core", "显式 rank / count marker 的 token span", "数字/序数 marker state 本身是否承载 terminal count"),
            ("Retrieved city", "最后一次检索得到的 city token span", "排除仅由 city 词汇内容驱动 answer 的解释"),
            ("Grammar-timed tail", "city→rank grammar: marker 到 commit；rank→city grammar: city 到 commit", "允许两种 grammar 在不同时间完成 terminal update"),
            ("Boundary commit", "parser 注册的 item-end commit token span", "最局部的提交边界是否已足够"),
        ),
    )
    terminal_arm_table = table(
        ("每条 trace 的 arm 组", "数量", "具体操作", "用于什么比较"),
        (
            ("Clean", "1", "原始 prompt + 原始 trace，不 patch", "自然参考"),
            ("Uninformative", "1", "每个 parsed trace item 都换成等长 prompt-background tokens", "所有 restoration 的共同 damaged baseline"),
            ("Semantic restore", "5", "分别把五种 geometry 的 clean same-position hidden states 写回", "restore − uninformative"),
            ("Matched-random restore", "5", "在相同 receiver positions 写入等 token、近深度、非 terminal-item 的 clean states", "semantic restore − matched random specificity"),
        ),
    )
    write_arm_table = table(
        ("Forward arm", "每条 trace 的次数", "在 query 做什么", "在 carrier 做什么", "用于哪一节"),
        (
            ("Clean", "1", "不 mask", "不 patch", "5.1/5.2 的 clean state 与距离参照"),
            ("Selected mask", "1", "mask frozen Top-128 / Top-6", "不 patch", "5.1 主 treatment；5.2 damaged baseline"),
            ("Layer-matched random mask", "3", "三套不同随机 bank；逐层 head 数与 selected bank 相同", "不 patch", "5.1 secondary identity control"),
            ("Selected mask + clean-carrier clamp", "1", "保持 selected mask", "逐层写回同一位置的 clean carrier", "5.2 主 restoration"),
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
<div class="definition"><dt>Confirmed、confirmed† 与 controlled only</dt><dd><strong>Confirmed</strong>：冻结干预在 20-seed discovery 后，又在独立 10-seed confirmation 上复现。<strong>Confirmed†</strong>：直接因果边复现了，但某个更严格的局部控制较弱；† 后会逐项写明限制。<strong>Controlled only</strong>：在所有后续可见 tokens 都固定相同的受控比较中，patch 会改变答案分数；但让模型自由继续生成时，还没有证明这一个 state 单独足以改变最终答案。它不是“没有效果”，而是“效果适用范围较窄”。</dd></div>
</dl>
<div class="section-conclusion"><strong>定义层面的结论。</strong> 后文的“confirmed”不意味唯一回路；“可解码”也不自动意味模型必须使用该线性方向。</div></section>

<section id="summary"><p class="eyebrow">Conclusion first</p><h2>一条 recurrent counting pathway 已经接上；natural end-to-end sufficiency 仍未证明</h2>
<p class="lead">两模型都支持同一类局部因果链：targeted heads 检索下一条 city，改变 grammar-specific marker/tail carrier；carrier 写入 commit state；commit state 再改变下一次 targeted query。终端 grammar state 在固定 suffix 的受控实验中能恢复 answer count margin，但把全部上下文抹掉后，仅恢复任一单 item 并不能让答案随 k 从 1 走到 10。</p>
<div class="figure-primer"><div><strong>图中画什么</strong>五个框是一次循环计数从检索到答案的候选阶段。</div><div><strong>怎么读</strong>每一列分别给 Qwen 与 Gemma 的最高证据级别；紫色表示仅在受控终端成立。</div><div><strong>简单例子</strong>完成第 3 条后，commit state 改变“下一次应读第 4 条”的 query routing。</div></div>
<div class="chain-scroll">{chain_svg()}</div>
<div class="status-grid"><div class="status-card"><h3>Qwen3-8B</h3><p class="status-good">recurrent loop：强 confirmation</p><p>Top-128 retrieval、carrier→commit、commit→next query 都有大效应。terminal grammar state 的局部受控 restoration 为正。</p><p class="status-open">仍开放：free-running answer count-margin 与全上下文擦除后的单点 sufficiency。</p></div>
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

<section id="write"><p class="eyebrow">05 · Write and recurrent propagation</p><h2>把 trace 内部的三条边分开检验</h2>
<p class="lead">这一节不是一次复杂 patch 得出整条链，而是依次问三个较小的问题。每个问题的 treatment、control 和读数都不同：</p>
<div class="edge-roadmap"><div><span>5.1 · READ → CARRIER</span><strong>关掉 targeted heads 后，紧随检索出现的 grammar state 会不会变？</strong></div><div><span>5.2 · CARRIER → COMMIT</span><strong>在 heads 仍被关闭时，只把 clean carrier 补回来，item-end commit 会不会恢复？</strong></div><div><span>5.3 · COMMIT → NEXT READ</span><strong>把“已完成 k”的 commit 换成“已完成 k+1”，下一次 query 会不会改读再下一条 record？</strong></div></div>

<h3>5.1–5.2 共用什么样本与实验底座</h3>
<p class="lead">5.1 与 5.2 不是先后另找两批“有效样本”。它们读取同一个预冻结的 <code>teacher_forced_targeted_counter_write</code> 七臂实验：5.1 看 retrieval query 之后的 carrier，5.2 看更晚的 item-end commit。这样才能在同一条 trace、同一次局部 head damage 下检验中介链。</p>
<div class="experiment-frame"><div><span class="experiment-label">独立样本</span><strong>一个 seed 对应一条完整 trace，才算一个独立样本。</strong>每模型使用 discovery seeds 1234–1253 的 20 条 trace，以及完全不重叠的 confirmation seeds 1254–1263 的 10 条 trace。每个 seed outcome-blind 地冻结一条 T10000 benchmark instance；该 trace 必须含恰好 <em>N</em> 个 parser-registered items，并有可定位的最后一次 <em>N−1</em>→<em>N</em> retrieval。</div><div><span class="experiment-label">不是样本</span>同一 seed 的 7 个 forward arms 是配对条件，不是 7 个独立样本；多个 carrier tokens、hidden coordinates 和 layers 也不是额外样本。统计时先在每个 seed 内算 treatment−control，再让 20 或 10 个 seeds 等权。</div><div><span class="experiment-label">实际规模</span>每模型 30 条独立 traces × 7 arms = 210 condition rows：discovery 20×7=140，confirmation 10×7=70。两模型合计 60 条独立 traces、420 condition rows。</div><div><span class="experiment-label">如何选 trace</span>先在每个 phase 内强制 grammar timing 平衡，再在可行候选中优先更高 <em>N</em>、固定 grammar preference 与 request id；不读取实验 outcome，也不使用 <code>selection_rank</code>。因此 count 分布不是 1–10 均匀抽样，而是一个 grammar-balanced、偏高 count 的因果 panel。</div></div>
{write_sample_table}
<p class="term-note"><strong>表中两个 grammar timing：</strong><em>city→rank</em> 表示 trace 先写 city，之后才写“这是第六条”一类 rank/count marker；<em>rank→city</em> 表示 trace 先写“第六条”，随后检索并写 city。每模型 discovery 严格 10/10，confirmation 严格 5/5。由于本实验需要 <em>N−1</em>→<em>N</em> transition，<em>N</em>=1 不可能进入样本。</p>
<h4>同一条 trace 的正式主分析使用哪 6 个 forward</h4>
{write_arm_table}
<p class="term-note">“Pre-O mask”指在注册 query 上，把入选 attention heads 的 <span class="formula">Σ<sub>t</sub>A(q,t)V(t)</span> slice 在进入该层 output projection <em>W</em><sub>O</sub> 前清零。它不删除整层，不改 prompt，也不直接 patch logits。Qwen 使用先前冻结的 Top-128；Gemma 使用 Top-6。三套 random banks 在每一层都匹配 selected bank 的 head 数量。原冻结协议额外跑过一个后来被判定为不可解释的 control，因此运行审计仍是 7 arms / trace；该 arm 不进入正文主分析，原因仅在 Appendix D 留档。</p>

<h3>5.1 关闭 targeted bank 后，检索结果有没有写入 grammar carrier</h3>
<div class="experiment-frame"><div><span class="experiment-label">要检验的箭头</span><strong>Targeted retrieval query → grammar carrier state。</strong>Experiment 4 已证明 bank mask 会使自由生成的 next city 检索失败；这里进一步问，即使把可见 trace 固定不让它分叉，关闭同一 bank 是否仍会改变检索之后的内部 state。</div><div><span class="experiment-label">Token 固定</span>把这条完整 clean trace 作为 teacher-forced token sequence，所有 arms 输入完全相同。Head mask 只发生在最后一次注册的 <em>N−1</em>→<em>N</em> query；所以不同 arm 不会因为生成了不同 city 或不同句子而落到不同 token positions。</div><div><span class="experiment-label">Carrier 怎么定</span>按 grammar outcome-blind 地注册。对于 <em>city→rank</em>，检索到 city 后才出现的新进度信息是 rank-marker core，因此读取该 marker 的全部 tokens；对于 <em>rank→city</em>，rank 已经先出现，检索后的新信息从 retrieved-city 第一个 token 延续到 item-end commit，因此读取整个 city-to-commit tail。Carrier 可以是多个 tokens，不要求固定长度。</div><div><span class="experiment-label">读取哪些层</span>Qwen 在 post-block L19–L35 读取 carrier，共 17 层；Gemma 在 L16–L41 读取，共 26 层。Source layers L19/L16 与 frozen targeted-counter protocol 一起预注册，没有按本实验 effect 重选。</div><div><span class="experiment-label">每 seed 的读数</span><span class="formula">D<sup>a</sup><sub>carrier</sub> = (1/|L|) Σ<sub>ℓ∈L</sub> ||H<sup>a,ℓ</sup><sub>carrier</sub> − H<sup>clean,ℓ</sup><sub>carrier</sub>||<sub>F</sub> / √(m·d)<br>Primary = D<sup>selected</sup><sub>carrier</sub><br>Selected-vs-random = D<sup>selected</sup><sub>carrier</sub> − (1/3)Σ<sub>r=1</sub><sup>3</sup>D<sup>random-r</sup><sub>carrier</sub></span><em>m</em> 是该 trace 的 carrier token 数，<em>d</em> 是 hidden width。先对 token×hidden dimensions 做 RMS，再对已注册 layers 平均，最后才跨 seeds 平均。</div><div><span class="experiment-label">具体例子 A</span><em>City→rank，N=6：</em>固定 trace 中出现“... Paris received a score ... That is the sixth record.”。5→6 query 位于检索 Paris 之前；我们只 mask 该 query 的 targeted heads，仍强制喂入完全相同的 Paris 和后续文字。读数取 “sixth” 这一 marker core 的 hidden states。如果它偏离 clean，说明 retrieval computation 会影响随后写出的进度 carrier。</div><div><span class="experiment-label">具体例子 B</span><em>Rank→city，N=6：</em>固定 trace 类似“Sixth, Paris ...”。Rank “Sixth” 在 query 前已可见，因此不能把它当作检索之后才写入的结果；这里 carrier 改取 Paris 第一个 token 到该 item commit 的 tail。这样两个 grammar 都遵守同一原则：只测 query 之后出现的状态。</div></div>
<div class="condition-list"><div class="condition-row"><strong>Primary：selected vs clean</strong><span>同一 seed、同一 token sequence、同一 carrier positions</span><span>回答 frozen bank 是否会改变 carrier；clean distance 按定义为 0</span></div><div class="condition-row"><strong>Secondary：selected vs 3×random</strong><span>random bank 逐层匹配 head 数，先在 seed 内取三者平均</span><span>回答变化是否特别归因于 selected identity，而不只是“关一批 heads 都会扰动”</span></div></div>
<div class="figure-primer"><div><strong>Panel A · 直接 damage</strong>selected-mask carrier 离 clean carrier 多远；回答“关 bank 有没有传到 carrier”。</div><div><strong>Panel B · head-identity control</strong>selected 的距离减去三套 layer-matched random 的平均距离；回答“是否特别是这组 heads”。</div><div><strong>每个 panel 独立缩放</strong>横轴都是 RMS distance，但 panel 自己定标；点=均值，线=95% CI。</div></div>
<figure><h3 class="figure-title">图 3a · 关掉 targeted heads 后 carrier hidden state 如何变化</h3>{effect_small_multiples_svg('Experiment 5.1 · targeted-head mask and carrier-state deformation', write_damage_panels)}<figcaption>两个 panel 都只属于 Experiment 5.1，不是 5.2 的 restoration。A 是 selected-mask 相对 clean 的总形变；B 是 selected 相对三套 random masks 的附加形变。Qwen 的 A/B 都稳定为正；Gemma 的 A 为正，但 B 接近 0，因此 Gemma 的 state damage 存在、bank identity specificity 较弱。</figcaption></figure>
<p><strong>Confirmation 结果。</strong> Selected mask 的 carrier deformation 为 Qwen {ci(write_effects['Qwen3-8B']['selected_carrier_deformation'])}、Gemma {ci(write_effects['Gemma4-E4B']['selected_carrier_deformation'])}。Selected 相对三套 layer-matched random 的附加 deformation 为 Qwen {ci(write_effects['Qwen3-8B']['selected_carrier_deformation_specificity'])}、Gemma {ci(write_effects['Gemma4-E4B']['selected_carrier_deformation_specificity'])}。</p>
<p><strong>如何读结果。</strong> 两模型的 selected→clean 数值都为正，所以 targeted-bank intervention 确实传到了 query 之后的 carrier。Qwen 的 selected−random 也清楚为正；Gemma 的均值只有 +0.009，区间跨 0，因此 Gemma 只能支持“这 6 枚 heads 被关掉时 carrier 会改变”，不能支持“只有这 6 枚 heads 才能写 carrier”。</p>
<h4>5.1b NCC：carrier 不只是“变了”，是否朝错误 count centroid 移动</h4>
<div class="experiment-frame"><div><span class="experiment-label">为什么重新分支运行</span>旧 NCC 虽然在拟合时区分 City→rank 与 Rank→city，却把两种 timing 放进同一个 20/10 panel、用二者平均 OOF 共同选层，并在最后汇总两个不可直接比较的 raw squared-distance margins。更重要的是，旧 City→rank carrier 直接读取 teacher-forced rank/count marker，本身已经包含要解码的显式 count token。新实验把 row panel、head bank、endpoint、layer selection 和 confirmation 统计全部按 timing 分开。</div><div><span class="experiment-label">City→rank 主 endpoint</span>使用 discovery atlas 中与 <code>adjacent_rank_after_city</code> 匹配的 P0 Top-K bank；读出点改为 <code>pre_marker_state</code>，即 rank marker 第一个 token 之前的精确 post-block state。注册的 secondary endpoint 是以该位置结尾的四-token suffix。全量 tokenizer audit 已验证两者均位于 retrieved city 之后、rank marker 之前，不含任何 marker/count token。</div><div><span class="experiment-label">Rank→city 主 endpoint</span>保留原冻结 rank-before bank 与 query anchor：Qwen 为 adjacent-rank-before / post-marker Top-128，Gemma 为 same-unit-rank-before / P0 Top-6；carrier 仍是 retrieved city 第一个 token 到 item commit 的 tail。</div><div><span class="experiment-label">Bank 与因果层</span>每个 timing 使用自己的 selected bank 与三套 selected-excluded、逐层 head 数完全匹配的 random banks。query head 在 L<em>h</em> 的输出最早只能影响更晚 token 的 L<em>h</em>+1 state，因此只在完整 bank 最高层之后读出：Qwen 两支均为 L35；Gemma 在 L30–L41 内对每个 endpoint 单独用 discovery grouped-OOF BA 选层。</div><div><span class="experiment-label">样本与冻结</span>仍使用固定 discovery seeds 1234–1253 与 confirmation seeds 1254–1263，但每支采用该 timing 下的 maximal eligible panel，所以 cohort 不强行配对：Qwen City→rank 19/10、Rank→city 19/9；Gemma 18/9、19/10。panel 与 bank 均不读取 outcome；confirmation 不参与 PCA、centroid 或 layer selection。由于旧 NCC 的 confirmation 已经被看过，本轮准确标为 registered-split retrospective extension，而不是新的 pristine prospective confirmation。</div><div><span class="experiment-label">可比较效应</span>每个 endpoint 仍用 <span class="formula">loss=m(clean,c)−m(mask,c)</span>；raw margin 只在该模型×timing 坐标系内报告。跨分支汇报时再除以 discovery grouped-OOF correct-centroid margin 的标准差，得到 SD units。即便 intervention contrast 为正，若 clean confirmation 的平均 correct-centroid margin 不为正，则不能把它解释为“损伤了正确 count geometry”；这个 readout-validity 是看到初始 contrast 后增加的解释资格审计，不冒充预注册 gate。</div></div>
{stratified_ncc_bank_table}
{stratified_ncc_result_table}
<figure><h3 class="figure-title">图 3a-2 · Targeted-head mask 对 NCC count geometry 的影响</h3>{effect_small_multiples_svg('Experiment 5.1b · timing-stratified NCC carrier corruption', stratified_ncc_panels)}<figcaption>四行是模型×timing 的独立 primary estimand。Panel A 为 selected margin loss，Panel B 为 selected−三套 random mean；均以各自 discovery OOF margin SD 标准化。点=confirmation seed mean，线=95% seed-bootstrap CI。raw margins 不跨行合并。绿色需要同时满足有效 clean readout 与支持方向的完整区间；橙色表示区间触零、方向不支持或 readout validity 失败。</figcaption></figure>
<p><strong>City→rank：更好的位置已经找到，但结果仍是 null。</strong>Qwen 的 exact pre-marker clean decoder 在 discovery/confirmation 为 1.000/0.800；Gemma 为 0.560/0.556，说明该位置并非“早到尚无 count 信息”。然而 Qwen selected loss / specificity 为 {ci(stratified_primary['Qwen3-8B']['rank_after_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Qwen3-8B']['rank_after_city']['standardized_specificity_estimand'], 4)} SD，Gemma 为 {ci(stratified_primary['Gemma4-E4B']['rank_after_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Gemma4-E4B']['rank_after_city']['standardized_specificity_estimand'], 4)} SD，均为负方向且区间跨 0。四-token suffix secondary 在两模型也同样不支持 damage。因此旧 City→rank 负项不能再归因于“marker endpoint 泄露导致符号被冲掉”；在无 marker 的预测 state 上仍未观察到 selected-bank-specific NCC 损伤。</p>
<p><strong>Rank→city：两模型都出现正方向，但证据资格不同。</strong>Qwen raw loss / specificity 为 {float(stratified_primary['Qwen3-8B']['rank_before_city']['raw_primary_estimand']['mean_effect']):+.2f} / {float(stratified_primary['Qwen3-8B']['rank_before_city']['raw_specificity_estimand']['mean_effect']):+.2f}，标准化后为 {ci(stratified_primary['Qwen3-8B']['rank_before_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Qwen3-8B']['rank_before_city']['standardized_specificity_estimand'], 4)} SD；但 clean confirmation 仅 1/9 exact、mean correct-centroid margin={float(stratified_primary['Qwen3-8B']['rank_before_city']['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f}。因此 Qwen 的 clean confirmation readout 无效，该正 shift 不能解释成已确认的 correct-count damage。Gemma raw 为 {float(stratified_primary['Gemma4-E4B']['rank_before_city']['raw_primary_estimand']['mean_effect']):+.2f} / {float(stratified_primary['Gemma4-E4B']['rank_before_city']['raw_specificity_estimand']['mean_effect']):+.2f}，标准化后只有 {ci(stratified_primary['Gemma4-E4B']['rank_before_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Gemma4-E4B']['rank_before_city']['standardized_specificity_estimand'], 4)} SD；clean 4/10 exact、mean margin={float(stratified_primary['Gemma4-E4B']['rank_before_city']['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f}，readout 有效，但两个区间均跨 0，只是 directional-specific evidence，尚非 interval-confirmed result。</p>
<p><strong>Qwen/Gemma 对应性审计。</strong>两模型现在在设计上真正对应：各 timing 独立 panel、独立匹配 bank、独立 discovery 选层、完整 bank 均已因果可达、同一五臂 clean/selected/3×random factorial，并用同一 discovery-only SD 规则标准化。它们也在 City→rank 上一致为 null，在 Rank→city 上一致呈正 raw 方向。它们仍不是同剂量或 paired-cohort 的模型比较：Qwen/Gemma bank 分别为 128/6 heads，confirmation seed 集不同，且没有 model×mask interaction。值得注意的是，两个 timing bank 仍重合 Qwen {int(stratified_ncc_inputs['Qwen3-8B']['selected_bank_overlap']['head_count'])}/128、Gemma {int(stratified_ncc_inputs['Gemma4-E4B']['selected_bank_overlap']['head_count'])}/6；Gemma 的弱效应不能简单归因于“选了完全错误的一组 heads”。</p>
<h4>5.1c 直接 count-output margin：绕过 NCC centroid，检验最终答案分布</h4>
<div class="experiment-frame"><div><span class="experiment-label">为什么再加一个 margin</span>NCC 要先假设某层 hidden state 的欧氏 centroid geometry 是模型实际使用的 count code；clean decoder 若无效，或有效信息位于非线性/非欧氏方向，真实的输出变化也可能被 NCC 漏掉。直接 output margin 不拟合 PCA、decoder、centroid，也不再选择读出层，而是直接问局部 retrieval lesion 是否持续到最终 count distribution。</div><div><span class="experiment-label">Primary endpoint</span>在冻结 trace 的最终 <code>Total:</code> answer query 上，对候选答案字符串 1,…,10 计算完整 autoregressive sequence log probability：<span class="formula">M<sub>out</sub>=log p(y=c|prefix)−max<sub>j≠c</sub> log p(y=j|prefix)<br>Selected loss=M<sub>clean</sub>−M<sub>selected</sub><br>Specificity=Selected loss−mean<sub>r=1..3</sub>(Random-r loss)</span>正 margin 表示正确 count 的完整序列分数高于最强错误候选；正 loss 表示 intervention 削弱这一优势。</div><div><span class="experiment-label">干预仍然局部</span>五个 arms 与 timing-matched NCC 完全对应：clean、selected bank、三套逐层匹配 random banks。Mask 只在冻结的最后一次 <em>N−1</em>→<em>N</em> retrieval query 执行；随后 answer-candidate tokens 在没有 hook 的状态下计分。因此测到的是该精确 lesion 是否持久影响最终答案，不是直接在 answer token 上动手。</div><div><span class="experiment-label">三个判据</span>先要求 clean mean margin&gt;0 且 candidate accuracy&gt;chance；方向性证据再要求 selected loss&gt;0 且 specificity&gt;0；最强的 interval-confirmed 判据要求这两个 confirmation seed-bootstrap 95% CI 的下界都&gt;0。四个模型×timing 分支分别报告，不合并 raw margins。</div><div><span class="experiment-label">注册边界</span>endpoint、五臂 contrast 与 gate 都在查看本轮 logit-margin outcome 前冻结；但沿用的 confirmation split 已在 NCC 分析中看过，所以这是 <em>registered existing-split retrospective extension</em>，不是 pristine prospective confirmation。模型间也没有 model×mask interaction，不能把 Gemma 与 Qwen 的数值差直接当作已检验的模型差异。</div></div>
<p class="term-note"><strong>证据层级。</strong>5.1a 的 carrier RMS deformation 才是“局部 hidden state 是否被改变”的主终点；5.1c 是更下游的 secondary functional diagnostic，不是确认 state change 的必要条件。最终答案还会受后续层重算、冗余通路、answer-time retrieval、候选 tokenization 与输出校准影响，所以 output-margin null 不能否定 hidden-state change，output-margin positive 也不能把全部效应归因于一个局部 count code。</p>
{direct_margin_result_table}
<p class="term-note">† <em>Directional-specific</em> 表示 development/confirmation 中 loss 与 specificity 方向均为正，且 confirmation specificity CI&gt;0；两支 Gemma 的 selected-loss CI 仍跨 0，因此不是 two-gate interval confirmation。</p>
<figure><h3 class="figure-title">图 3a-3 · Retrieval-query mask 对最终 count-output margin 的影响</h3>{effect_small_multiples_svg('Experiment 5.1c · direct final count-output margin', direct_margin_panels)}<figcaption>四行仍是模型×timing 的独立 primary estimands。Panel A 是 clean−selected 的最终答案 sequence-log-prob margin loss；Panel B 再减去三套 random-mask losses 的均值。点=confirmation seed mean，线=95% seed-bootstrap CI。绿色只表示该单项 CI 完全高于 0；完整 interval-confirmed 结论要求同一行 A/B 两项同时为绿，本次没有任何一行满足。</figcaption></figure>
<p><strong>先看 readout 是否可信。</strong>四支 clean candidate accuracy 都是 1.000，confirmation clean mean margin 为 Qwen City→rank {float(direct_margin_primary['Qwen3-8B']['rank_after_city']['confirmation']['clean_mean_margin']):+.3f}、Rank→city {float(direct_margin_primary['Qwen3-8B']['rank_before_city']['confirmation']['clean_mean_margin']):+.3f}，Gemma 分别为 {float(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['clean_mean_margin']):+.3f} 与 {float(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['clean_mean_margin']):+.3f}。因此四个 primary 都通过 readout-validity gate；这里不存在 Qwen Rank→city NCC 那种 clean geometry 已经把正确 count 放在错误侧的问题。</p>
<p><strong>Qwen：直接答案读出有效，但没有 selected-bank-specific damage。</strong>City→rank 的 confirmation selected loss / specificity 为 {ci(direct_margin_primary['Qwen3-8B']['rank_after_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Qwen3-8B']['rank_after_city']['confirmation']['selected_vs_random_specificity'])}；Rank→city 为 {ci(direct_margin_primary['Qwen3-8B']['rank_before_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Qwen3-8B']['rank_before_city']['confirmation']['selected_vs_random_specificity'])}。两支 selected loss 均值虽略为正，但 confirmation specificity 都为负；Rank→city 的 discovery specificity 还是正值，到了 confirmation 反向，因而不能称为复现。</p>
<p><strong>Gemma：最终答案 margin 出现小的 selected-specific signal。</strong>City→rank 的 selected loss / specificity 为 {ci(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['selected_vs_random_specificity'])}；Rank→city 为 {ci(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['selected_vs_random_specificity'])}。两支在 discovery 与 confirmation 都保持 selected loss&gt;0、specificity&gt;0，而且 specificity 的 confirmation CI 均完全高于 0；但 selected-loss CI 都跨 0。因此可写成“跨 split 的 directional-specific evidence”，不能升级成“两道 interval gate 均确认”。按 clean margin 归一化，selected loss 约为 {100*float(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['selected_margin_loss']['mean_effect'])/float(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['clean_mean_margin']):.1f}% 与 {100*float(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['selected_margin_loss']['mean_effect'])/float(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['clean_mean_margin']):.1f}%，效应很小但方向与最终任务对齐。</p>
<p><strong>City→rank 的局部 marker secondary 没有提供更强结论。</strong>Qwen 的 clean local sequence margin={float(direct_margin_local['Qwen3-8B']['confirmation']['clean_mean_margin']):+.3f}、accuracy={float(direct_margin_local['Qwen3-8B']['confirmation']['clean_accuracy']):.3f}，readout validity 失败；marker 候选的表面 tokenization/长度差使其正 specificity 不可解释。Gemma 的 local readout 有效（clean margin={float(direct_margin_local['Gemma4-E4B']['confirmation']['clean_mean_margin']):+.3f}、accuracy={float(direct_margin_local['Gemma4-E4B']['confirmation']['clean_accuracy']):.3f}），但 selected loss / specificity 为 {ci(direct_margin_local['Gemma4-E4B']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_local['Gemma4-E4B']['confirmation']['selected_vs_random_specificity'])}，均不支持 damage。也就是说，Gemma 的阳性更明确地出现在最终答案 margin，而不是紧邻 City→rank marker 的这个特定表面读出；这不等于已经定位了中间的完整中介路径。</p>
<div class="qualification"><strong>NCC 与 output margin 并不矛盾。</strong>RMS norm 证明 hidden state 被 intervention 改变；NCC 问它是否沿一个 discovery-fitted centroid geometry 移动；output margin 则记录经过全部后续计算后的净答案效应。Gemma 的组合结果至少说明 state change 与弱 downstream signal 可以并存；但仅凭当前实验无法区分“NCC 没捕获到模型使用的方向”“count code 在后层被重新编码”与“其他通路补偿后只留下小净效应”，更不能据此宣称一个唯一 count code。</div>
<details class="paper-appendix"><summary>历史 frozen pooled NCC 与事后 layer×timing 诊断（仅用于复现旧 headline）</summary>
<div class="experiment-frame"><div><span class="experiment-label">为什么补 NCC</span>RMS distance 只回答 masked state 离 clean state 多远；它不知道移动方向是否与 running count 有关。Nearest-centroid classifier（NCC）把 discovery clean carrier 的 count geometry 冻结下来，再问 confirmation state 更靠近哪个 count centroid。</div><div><span class="experiment-label">训练样本</span>对 20 discovery seeds 的每个已完成 occurrence 都取 grammar carrier vector。City→rank 与 rank→city 分开拟合；每个 stratum 内先 standardize，再 PCA-16，最后计算每个 count 的 centroid。五折按 seed 分组，任何同一 seed 的 occurrence 都不会同时出现在 train/test。</div><div><span class="experiment-label">冻结的层选择</span>原协议在每层分别计算两个 grammar strata 的 grouped-OOF balanced accuracy，再取两者平均。Discovery 最大层被冻结：Qwen L{int(ncc_supplement['Qwen3-8B']['selected_layer'])}，Gemma L{int(ncc_supplement['Gemma4-E4B']['selected_layer'])}。10 confirmation seeds 没有参与原 PCA、centroid 或 layer selection；下面新增的逐层检查明确标为事后诊断，不能替换这项 frozen primary。</div><div><span class="experiment-label">Confirmation 条件</span>每条 confirmation trace 在同一 frozen carrier 上比较 clean、selected bank mask 与三套逐层匹配 head 数的 random masks。所有条件共享完全相同的 teacher-forced tokens 与 positions；区别只在注册 retrieval query 的 pre-O head output 是否被清零。</div><div><span class="experiment-label">Correct-centroid margin</span><span class="formula">m(z,c)=min<sub>j≠c</sub>||z−μ<sub>j</sub>||²−||z−μ<sub>c</sub>||²<br>Margin loss = m(clean,c)−m(mask,c)</span>正 margin 表示正确 centroid 比任何错误 centroid 更近；正 loss 表示 mask 把 state 推离正确 count geometry。Specificity 再减去三套 random-mask loss 的 seed 内平均。</div><div><span class="experiment-label">重要尺度限制</span>City→rank 与 rank→city 使用各自独立的 standardization、PCA 与 centroids，因此 raw squared-distance margins 不具有天然相同的尺度。原 pooled mean 保留用于复现冻结分析，但应把分层结果作为主要诊断，不能把 pooled 数值当作跨 timing 或跨模型的校准 effect size。</div></div>
<h4>冻结主分析：保留结果，但收窄解释</h4>
{ncc_result_table}
<figure><h3 class="figure-title">历史图 H5.1 · 原 pooled NCC margin</h3>{effect_small_multiples_svg('Historical Experiment 5.1b · pooled NCC carrier corruption', ncc_panels)}<figcaption>这是原冻结层与原 pooled estimand，只用于复现旧 headline。Panel A 是 clean→mask correct-centroid margin loss；Panel B 再减去三套 layer-matched random masks 的平均 loss。它不提供 Qwen–Gemma interaction test，也不校准两个 grammar-specific NCC 坐标系的 margin 尺度。</figcaption></figure>
<p><strong>冻结 NCC 结果。</strong> Qwen selected margin loss={ci(ncc_primary['Qwen3-8B'])}、selected−random={ci(ncc_specificity['Qwen3-8B'])}，只能记为未确认的方向性信号；hard NCC exact 保持 {float(ncc_condition['Qwen3-8B']['clean']['exact_accuracy']):.2f}→{float(ncc_condition['Qwen3-8B']['selected_mask']['exact_accuracy']):.2f}。Gemma pooled selected loss={ci(ncc_primary['Gemma4-E4B'])}、specificity={ci(ncc_specificity['Gemma4-E4B'])}，hard exact 同样保持 {float(ncc_condition['Gemma4-E4B']['clean']['exact_accuracy']):.2f}→{float(ncc_condition['Gemma4-E4B']['selected_mask']['exact_accuracy']):.2f}。因此 frozen primary 既没有确认 Gemma 的 late full-Top-6 NCC damage，也没有检验出“Gemma 比 Qwen 更弱”的模型差异。</p>

<h4>Qwen/Gemma 对应性审计：数据可复现，但不是同剂量、同 cohort 的模型比较</h4>
{ncc_correspondence_table}
<p><strong>先确认结果文件本身。</strong>我们从两模型原始 NPZ 独立重跑 frozen analyzer：Qwen 仍选择 L23，Gemma 仍选择 L34；selected loss、specificity 与 hard exact 等冻结汇总的最大绝对差分别只有 {float(ncc_layerwise_diagnostic['Qwen3-8B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f} 和 {float(ncc_layerwise_diagnostic['Gemma4-E4B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f}，来自不同平台 SVD/浮点舍入。条件顺序、teacher-forced token contract、20/10 split、10/10 与 5/5 timing 配额均一致。因此 shard、聚合代码和报告数字之间没有发现错位。</p>
<p class="term-note"><strong>Hook audit 字段说明。</strong>旧 NPZ 的 condition metadata 中 <code>head_hook_applications</code> 为空，是 capture caller 读取了旧字段名；底层 intervention helper 仍会强制每个注册 layer 恰好执行一次、否则直接报错，且逐层 state difference 只从因果可达层开始出现。该问题不改变已有 vectors 或统计，但削弱了旧 shard 的持久化可读审计；后续 capture 已同时保存 application counts 与 post-zero maximum。</p>
<p><strong>再确认“对应”到什么程度。</strong>两者使用同一 NCC pipeline 和同构 selected/random 条件，可以做各自模型内的因果诊断；但 Qwen confirmation 主要是 N=10，而 Gemma 的 N 更分散，两个 bank 的大小、定位 grammar/anchor 和层分布也不同。更关键的是，query head 在 L 的输出要到后续 carrier 的 L+1 才可达：Qwen frozen L23 实际只有 24/128 selected heads 已能影响读数，Gemma L34 则为 6/6。因此 raw margin loss 不能作为 calibrated Qwen-vs-Gemma effect size，当前也没有 model×mask interaction test。</p>

<h4>同一 frozen layer 内拆开 timing 后，两模型呈现相同的符号结构</h4>
{ncc_frozen_timing_table}
<p><strong>核心对应关系。</strong>两个 frozen banks 都是在 rank-before-city 家族上定位的。与 bank 匹配的 Rank→city stratum 中，Qwen L23 selected loss={float(qwen_ncc_l23_before['selected_margin_loss_mean']):+.2f}、specificity={float(qwen_ncc_l23_before['selected_vs_random_specificity_mean']):+.2f}，5/5 seeds 为正；Gemma L34 也保持正方向（{float(gemma_ncc_l34_before['selected_margin_loss_mean']):+.2f}、{float(gemma_ncc_l34_before['selected_vs_random_specificity_mean']):+.2f}），只是较弱。相反，非 bank-matched 的 City→rank stratum 在两模型都是负方向：Qwen {float(qwen_ncc_l23_after['selected_margin_loss_mean']):+.2f}，Gemma {float(gemma_ncc_l34_after['selected_margin_loss_mean']):+.2f}。所以 Qwen pooled +29.39 与 Gemma pooled −16.58 不是相反机制；它们都是“Rank→city 正、City→rank 负”，区别在于两个独立坐标系的哪一项数值占主导。</p>
<p><strong>统计边界。</strong>Qwen Rank→city 的 bootstrap interval 为 {ci(qwen_ncc_l23_before['selected_margin_loss_summary'])}，但 n=5 的精确 sign-flip <em>p</em>={float(qwen_ncc_l23_before['selected_margin_loss_summary']['p_value']):.3f}；Gemma L34 Rank→city 区间跨 0。因此这次拆分更正了机制方向的解释，却没有把任一模型升级成新的确认性结果。</p>

<h4>Gemma layer×timing 事后诊断：负的 pooled result 从哪里来</h4>
<div class="experiment-frame"><div><span class="experiment-label">重算对象</span>直接读取原 20 discovery + 10 confirmation NPZ，在每个 L16–L41、每个 grammar timing 上重新拟合同构 discovery NCC，并计算同一批 confirmation mask effects；没有重新运行模型，也没有改变任何 token 或 intervention。</div><div><span class="experiment-label">因果层窗口</span>Gemma Top-6 由 L17 三头与 L29 三头构成。query token 在 L17 的输出最早只能影响后续 carrier token 的 L18 state；L29 heads 最早从 L30 才可达。因此 L16–L17 是零效应 sanity check，L18–L29 主要隔离 L17 三头，L30+ 才包含 full Top-6 的传播。</div><div><span class="experiment-label">推断边界</span>这次诊断是在看到原 confirmation 结果后进行。L27 是按 discovery-only、bank-matched rank→city OOF 在 L17-only causal window 内选出的诊断层；L28 是相邻层稳定性检查。二者都不能升级为新的 confirmation gate，必须用新 seeds 复验。</div></div>
{gemma_ncc_layerwise_table}
<p><strong>层选择错配。</strong>Rank→city 的全层 discovery OOF 最大值其实在 L16（0.532），但该层位于 L17 heads 上游，mask 在这里按因果结构必须为 0。进入 L17-only 可达窗口后，discovery-only 最佳层是 L27（OOF={float(gemma_ncc_l27_before['discovery_grouped_oof_balanced_accuracy']):.3f}）；full Top-6 可达窗口内最佳层才是 L34，但其 rank→city OOF 只剩 {float(gemma_ncc_l34_before['discovery_grouped_oof_balanced_accuracy']):.3f}。与此同时，city→rank OOF 在 L{int(gemma_ncc_layerwise_diagnostic['timing_crossover']['rank_after_first_oof_at_least_0_95'])} 首次超过 0.95，并从 L{int(gemma_ncc_layerwise_diagnostic['timing_crossover']['rank_after_first_perfect_oof'])} 起达到 1.00；原平均-OOF rule 因而被晚层 city→rank decoder 主导，选到 L34。</p>
<p><strong>中层存在探索性 rank→city damage。</strong>在 discovery-only 诊断层 L27，selected loss={ci(gemma_ncc_l27_before['selected_margin_loss_summary'])}，selected−random={ci(gemma_ncc_l27_before['selected_vs_random_specificity_summary'])}，4/5 seeds 为正。相邻 L28 上，selected loss={ci(gemma_ncc_l28_before['selected_margin_loss_summary'])}，specificity={ci(gemma_ncc_l28_before['selected_vs_random_specificity_summary'])}，5/5 seeds 为正；但精确 sign-flip <em>p</em>={float(gemma_ncc_l28_before['selected_margin_loss_summary']['p_value']):.3f}，受 n=5 的离散分辨率限制，仍不构成确认性证据。该窗口位于 L29 heads 可传播之前，因此更具体地指向 L17 三个 retrieval heads，而不是完整 Top-6。</p>
<p><strong>为什么最终变成 −16.58。</strong>在 frozen L34，rank→city selected loss 仍是 {float(gemma_ncc_l34_before['selected_margin_loss_mean']):+.2f}、specificity {float(gemma_ncc_l34_before['selected_vs_random_specificity_mean']):+.2f}；city→rank 却是 selected loss {float(gemma_ncc_l34_after['selected_margin_loss_mean']):+.2f}、specificity {float(gemma_ncc_l34_after['selected_vs_random_specificity_mean']):+.2f}。后者的 clean margin 均值约 {float(gemma_ncc_l34_after['clean_margin_mean']):.0f}，前者约 {float(gemma_ncc_l34_before['clean_margin_mean']):.0f}。两个独立 NCC 坐标系被等权汇总后，高尺度、晚层、可完美解码的 city→rank 项把 pooled mean 拉成负值。这更像显式 marker/grammar readout 与 layer pooling 的组合，而不是“Gemma 没有 count state”。</p>
<div class="plain-language"><strong>最简单的读法：</strong>Gemma 的 frozen L34 pooled NCC 没有确认 full Top-6 会损伤 late count geometry；但原始逐层数据也不支持“Gemma 完全没有 NCC effect”。更精确的定位是：L17 targeted heads 对 L18–L28 的 rank→city geometry 有探索性的选择性损伤，随后该 geometry 在 L29 附近被重写、修复或换成晚层 marker-dominated representation。</div>
</details>
<div class="section-conclusion"><strong>Experiment 5.1 结论。</strong>主结论来自 RMS hidden-state distance：retrieval lesion 会改变 query 之后的 carrier；这个结论不依赖最终 count 是否翻转。修复 marker 泄露、timing pooling 与因果层可达性后，完整 frozen bank→late correct-count NCC chain 在两模型仍未确认。更下游的 final count-output margin 中，四支 clean readout 全部有效，Qwen 两支没有 selected-bank specificity；Gemma 两支出现小幅、跨 discovery/confirmation 同方向的 selected-specific reduction，且 specificity CI 高于 0，但 selected-loss CI 仍跨 0，所以没有任何分支通过两道 interval gate。它说明局部 state change 在 Gemma 的最终分布中留下弱净效应，不把该净效应等同于局部 state 的全部功能。原 pooled headline 的符号差异来自 timing/layer/scale mixture，只保留为历史复现。</div>

<h3>5.2 在同一 query damage 下，恢复 clean carrier 能否救回 item-end commit</h3>
<div class="experiment-frame"><div><span class="experiment-label">要检验的箭头</span><strong>Grammar carrier state → later commit state。</strong>5.1 只说明 carrier 会随 retrieval damage 改变，仍可能只是旁观者。5.2 保持 selected head mask 不变，只操纵 carrier，观察更晚的 commit 是否被救回。</div><div><span class="experiment-label">三个必要条件</span>Clean forward 提供 clean carrier 与 clean commit；selected-mask forward 提供 damaged baseline；selected-mask + clean-carrier clamp 是主 restoration。后两臂使用同一 query mask、visible tokens 和 carrier positions，唯一改变是否把 clean carrier hidden vectors 写回。</div><div><span class="experiment-label">Clamp 的位置和层</span>Qwen 在 L19–L34、Gemma 在 L16–L40，把 clean carrier vectors 写回相同 carrier token positions；即从 source layer 一直 clamp 到倒数第二个 block。最后一层不再 clamp，而是在 Qwen L35 / Gemma L41 读取 boundary commit。逐层 cumulative clamp 回答“在 head damage 已存在时，维持正确 carrier 是否足以让下游 commit 恢复”，不回答“只 patch 哪一层就足够”。</div><div><span class="experiment-label">Commit 怎么读</span>Boundary commit 是 parser/causal compiler 注册的 <code>post_update_commit_state</code> tokens，严格位于 terminal item 内、answer query 之前。只在最后一个 post-block layer 计算它到 clean boundary 的 full-vector RMS distance。</div><div><span class="experiment-label">每 seed 的读数</span><span class="formula">D<sup>a</sup><sub>commit</sub> = ||H<sup>a,last</sup><sub>boundary</sub> − H<sup>clean,last</sup><sub>boundary</sub>||<sub>F</sub> / √(b·d)<br>Clean-carrier recovery = D<sup>selected</sup><sub>commit</sub> − D<sup>selected+clean-carrier</sup><sub>commit</sub></span><em>b</em> 是 boundary token 数。正的 recovery 表示：在同一 selected-head damage 下，补回 clean carrier 使 commit 更接近 clean reference。</div><div><span class="experiment-label">完整例子</span>仍用 <em>city→rank, N=6</em>。先运行 clean arm，保存 “sixth” carrier 在各层的 vectors 和句末 commit。第二个 arm 在 5→6 query 关闭 selected heads，得到偏离 clean 的 commit。第三个 arm 保持同一 mask，但在每一层把 clean “sixth” vectors 写回 “sixth” positions。若 selected commit distance=0.80，restored distance=0.30，则 recovery=0.50。</div></div>
<div class="plain-language"><strong>图 3b 只回答两件事。</strong>A 先确认 selected-mask damage 确实到了 later commit；B 再检验在同一 damage 下补回 clean carrier 能否 rescue commit。A 是 damage sanity check，B 是 5.2 的核心因果对比。</div>
<div class="figure-primer"><div><strong>Panel A · damage sanity check</strong>selected mask 下的 commit 离 clean commit 多远；若为 0，restoration 就没有可救对象。</div><div><strong>Panel B · main causal rescue</strong>同一 selected mask 下，clean-carrier clamp 减少了多少 commit distance；这是 5.2 的核心数字。</div><div><strong>每个 panel 独立缩放</strong>点是 10 confirmation seeds 的 paired mean，线是 95% CI；两 panel 量纲相同，但为了可读性独立定标。</div></div>
<figure><h3 class="figure-title">图 3b · 同一 head damage 下，clean carrier 能否救回 later commit</h3>{effect_small_multiples_svg('Experiment 5.2 · carrier-mediated commit restoration', write_mediation_panels)}<figcaption>A 是 selected-mask commit damage；B 是 clean-carrier 相对该 damaged baseline 减少的 commit distance。两个对比都在相同 seed、相同 teacher-forced trace 和相同 query mask 内配对计算；图中只保留主因果问题必需的 damage 与 rescue。</figcaption></figure>
{write_table}
<p><strong>Confirmation 结果。</strong> Clean-carrier recovery 为 Qwen {ci(write_effects['Qwen3-8B']['clean_carrier_restoration'])}、Gemma {ci(write_effects['Gemma4-E4B']['clean_carrier_restoration'])}。正值表示 clean-carrier clamp 使 later commit 更接近同一 trace 的 clean commit。</p>
<p><strong>换成更直观的恢复比例。</strong>Selected mask 对 final commit 造成的平均 damage 是 Qwen {float(write_effects['Qwen3-8B']['selected_boundary_deformation']['mean_effect']):.3f}、Gemma {float(write_effects['Gemma4-E4B']['selected_boundary_deformation']['mean_effect']):.3f}；clean-carrier recovery 分别是 {float(write_effects['Qwen3-8B']['clean_carrier_restoration']['mean_effect']):.3f} 与 {float(write_effects['Gemma4-E4B']['clean_carrier_restoration']['mean_effect']):.3f}。用两个 confirmation mean 做描述性比值，约恢复原 damage 的 {100*write_recovery_fraction['Qwen3-8B']:.1f}% 和 {100*write_recovery_fraction['Gemma4-E4B']:.1f}%。这个比例是帮助读数的 effect-size summary，不是另外注册的 seed-level estimand，所以不为它单独报 CI。</p>
<div class="plain-language"><strong>用一句话概括 5.1–5.2：</strong>同一条固定 trace 上，先只破坏最后一次 targeted retrieval，随后看到 carrier 与 commit 偏离 clean；保持这次 query damage 不变，只补回正确 carrier，commit 又朝 clean 回来。因为 damage 和 mediator restoration 在同一 seed 内配对，这比“carrier 与 commit 相关”更接近一条串联因果边。</div>
<div class="section-conclusion"><strong>Experiment 5.2 结论。</strong> 在 10 条独立 confirmation traces/model 上，保持 targeted-head damage 不变时，cumulative clean-carrier clamp 能部分恢复最终 item-end commit。因此 carrier 不只是随 retrieval 一起变化的旁观 state；它对 later commit 具有受控的因果作用。证据针对当前 grammar-balanced final-transition panel，不声称单层、单 token 或所有 counts 上都同样充分。</div>

<h3>5.3 Commit state 是否决定下一次 targeted query 读向哪里</h3>
<p class="lead">5.1–5.2 已经说明一次 retrieval 会写出 carrier，再提交成 item-end state；5.3 换一个问题：<strong>这个 item-end state 会不会成为下一轮 retrieval 的控制输入？</strong> 若会，trace 中就形成 READ→WRITE→COMMIT→NEXT READ 的 recurrent loop。</p>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>只改“已完成 item <em>k</em>”处的 commit hidden state，随后观察紧接着的 targeted query 是否改读另一个 prompt record。下一 query 的 token、prompt records 和 targeted head bank 均不改。</div><div><span class="experiment-label">什么是一个样本</span>一个 seed 的完整 trace 是一个独立样本。每个 seed 内预先编译 6 个 same-trajectory donor–receiver pairs：signed offset <em>d</em>=−3,−2,−1,+1,+2,+3；这些 pairs 和五个 conditions 都是同一 seed 内的配对重复，不当作新的独立 seeds。</div><div><span class="experiment-label">角色</span>Receiver 是要被改写的 completed-item commit；donor 是同一 trace 中相隔 <em>d</em> 个 occurrences 的 commit。Qwen 在 L19、Gemma 在 L16 读取 donor 的 single-token post-block state并写到 receiver 的同层位置。正文 primary 只用 outcome-blind 冻结的 <em>d</em>=±1；±2/±3 是预注册 dose robustness。</div><div><span class="experiment-label">五个条件</span>Clean 保留自然轨迹；self patch 控制 hook；full donor patch 移植完整 commit vector；count-subspace transplant 只移植 frozen linear count component；orthogonal patch 提供同范数的非该 subspace 扰动。下表把每个 arm 的因果问题逐项列出。</div><div><span class="experiment-label">主终点</span><span class="formula">Y = Σ<sub>h∈bank</sub>A<sub>h</sub>(donor-successor record) − Σ<sub>h∈bank</sub>A<sub>h</sub>(receiver-successor record)<br>Direct effect = Y<sub>full donor</sub> − Y<sub>self</sub><br>Local specificity = Y<sub>full donor</sub> − Y<sub>orthogonal</sub></span>Y&gt;0 表示下一 query 的 bank-level attention 更偏向 donor 所对应的下一条 record。</div><div><span class="experiment-label">简单例子</span>Receiver 表示“刚完成第 4 条”，自然下一 query 应读 record 5；donor 表示“刚完成第 5 条”，其 successor 是 record 6。若只把 donor 的 commit hidden state 写到 receiver，下一 query 对 record 6 相对 record 5 的 attention 增加，就说明 commit 在控制下一轮 routing。可见文本中没有额外写入数字 5，也没有 patch query 本身。</div></div>
{query_sample_table}
<p class="term-note"><strong>实验规模。</strong> Discovery 每模型 20×6=120 pairs，confirmation 每模型 10×6=60 pairs；每 pair 注册 5 个 conditions。因此协议对应每模型 600 个 discovery condition rows 与 300 个 confirmation condition rows。正式统计先在相同 seed、相同 pair 内求 treatment−control，再对 seeds 等权。</p>
{query_arm_table}
<div class="figure-primer"><div><strong>第一行</strong>full donor−self：回答“完整 commit 换过去，下一 query 是否真的转向 donor successor”。这是直接边。</div><div><strong>第二行</strong>full donor−orthogonal：回答“完整 donor state 是否优于一个同等 count-component 范数、但不在 frozen count subspace 内的一般扰动”。这是更严格的局部 specificity。</div><div><strong>读数单位</strong>两行都是 frozen-bank-summed raw attention mass。Qwen bank 有 128 heads，Gemma 只有 6 heads，因此<strong>不能用条长绝对值跨模型比较机制强弱</strong>；应在各模型内比较 treatment 与 control。</div></div>
<figure><h3 class="figure-title">图 4a · Commit patch 是否把下一次 query 转向 donor-successor record</h3>{grouped_bars_svg('Confirmation mean effects', query_groups)}<figcaption>每个横条是 10 confirmation seeds 的 paired mean effect。Qwen full-vs-self / full-vs-orthogonal 分别为 {float(query_effects['Qwen3-8B']['self']['mean_effect']):+.3f} / {float(query_effects['Qwen3-8B']['orthogonal']['mean_effect']):+.3f}；Gemma 为 {float(query_effects['Gemma4-E4B']['self']['mean_effect']):+.3f} / {float(query_effects['Gemma4-E4B']['orthogonal']['mean_effect']):+.3f}。正值表示 full donor commit 更能让下一 query 偏向 donor 的下一条 record。</figcaption></figure>
{query_table}
<figure><h3 class="figure-title">图 4b · Full commit 相对 orthogonal control 的 donor-distance robustness</h3>{grouped_bars_svg('Full commit − orthogonal targeted-attention effect', query_distance_groups)}<figcaption>纵向三行是 |donor−receiver|=1,2,3；横条仍是 full-donor patch 相对 orthogonal patch 的 targeted-attention effect。|d|=1 是冻结 primary；|d|=2/3 是预注册 secondary dose robustness，不能事后代替 primary。Orthogonal 扰动只与 count-projected delta 等范数，不与完整 donor−receiver delta 等范数。</figcaption></figure>
{query_distance_table}
<div class="qualification"><strong>为什么 Gemma 标 confirmed†，而不是含糊的 directional。</strong> 最直接的 full donor−self 对比在独立 confirmation 上是 {ci(query_effects['Gemma4-E4B']['self'])}，所以“完整 commit 会改变下一 query”已经复现。† 只提醒更严格的 |d|=1 full−orthogonal 对比为 {ci(query_effects['Gemma4-E4B']['orthogonal'])}，均值为正但跨 seed 不够稳定。换句话说：<strong>直接边成立；当前窄线性 count control 下的排他 specificity 较弱。</strong></div>
<div class="scope-compare"><div><h4>Gemma 看起来较小，但不是主效应失败</h4><p>图的 outcome 对 bank heads 求和；Qwen 是 Top-128，Gemma 是 Top-6。Gemma 的 +0.491 与 Qwen 的 +4.749 不是同一容量尺度，不能直接说前者只有后者的十分之一。Gemma 的 full−self CI 全为正，直接 recurrent edge 已确认。</p></div><div><h4>真正偏弱的是 full−orthogonal</h4><p>Gemma 在 |d|=1 只有 +0.126，说明 orthogonal arm 自己也能推动一部分 routing。可能原因是 Gemma commit 的功能信息分布在 frozen count subspace 之外，或这个 6-head bank 对一般 query-state 扰动更敏感；这两种解释当前实验不能区分。|d|=3 时效应升到 {float(query_effects['Gemma4-E4B']['orthogonal_d3']['mean_effect']):+.3f}，与 state separation 增大后 signal/control 更易区分相容，但只是 secondary diagnostic。</p></div></div>
<p><strong>结果分析。</strong> Qwen 的 direct effect 与 local specificity 都大，支持强 recurrent routing。Gemma 的完整 commit 同样会重定向下一 query，但“这条作用只来自一个狭窄线性 counter direction”没有得到同等强度的支持。因而合理 claim 是 shared recurrent computation、不同 state geometry，而不是两模型都实现同一条线性 counter axis。</p>
<div class="section-conclusion"><strong>Experiment 5.3 结论。</strong> 两模型的 commit state 都会因果改变下一次 targeted query：Qwen 是强且控制特异的边；Gemma 是已复现的 direct edge，但 local matched-control specificity 有 † 限制。至此，targeted retrieval→carrier→commit→next targeted retrieval 的 trace 内循环已经接上。</div></section>

<section id="answer"><p class="eyebrow">06 · Terminal readout</p><h2>Answer query 自然依赖 trace source；terminal state 在 fixed-suffix 中能控制 count margin</h2>
<h3>6.1 Answer 到底在读 trace，还是回 prompt 重数</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>区分两种 answer-time 路径：（1）从已写好的 trace states 读 count；（2）在 answer query 再对 prompt records 做一次 broad retrieval。</div><div><span class="experiment-label">干预如何实现</span>“Blank”不是删除 token，也不是缩短序列：目标 token 的 embedding 与每一层 post-block residual state 都置零，绝对位置、序列长度和 query token 保持不变。左 panel 的 matched control 置零等数量的 ordinary prompt tokens。</div><div><span class="experiment-label">样本与加权</span>左侧 next-city localization：Qwen 10 confirmation seeds / 45 requests，Gemma 9 seeds / 30 requests；在 request 内配对、seed 内平均、最后 seeds 等权。右侧 answer source：每模型 10 confirmation seeds / 100 prompts per condition，同样 seed-equal。Gemma 左侧少一个 seed 是 frozen exact-P0 可用事件集合的审计结果，不用 outcome 补样本。</div><div><span class="experiment-label">两套 answer banks</span>Discovery seeds 1234–1253 分别按 answer query 对 trace-item source 和 prompt-record source 的 attention 排名，冻结各自 Top-32；confirmation 不重新选 heads。图 5a 的 exact generation 不依赖选择哪套 bank 来“读数”，两套 instrumentation 得到相同生成结果。</div><div><span class="experiment-label">主终点</span><span class="formula">Exact = 1[greedy parsed answer = gold N]<br>Trace necessity = Exact<sub>clean</sub> − Exact<sub>trace blank</sub></span></div><div><span class="experiment-label">简单例子</span>如果保留 trace 而擦除 prompt records 后仍答对 6，但保留 prompt 而擦除 trace 后答错，则此时 answer 主要使用 trace 已写入的状态；这不排除 prompt 在其他样本中提供并行补充。</div></div>
<div class="figure-primer"><div><strong>左图/上图</strong>比较 clean、prompt-record blank 和 full-trace blank 的 greedy exact accuracy。</div><div><strong>Head-bank 图</strong>展示 answer query 在 prompt-record 与 trace-context 两类 source 上的 attention composition。</div><div><strong>不能过度解释</strong>Source blank 证明内容必要性，不证明某个 Top-32 bank 是唯一 readout。</div></div>
<figure><h3 class="figure-title">图 5a · Token-source ablation 对局部检索与最终 count 的影响</h3>{token_source_ablation_svg(token_evidence)}<figcaption>左 panel 纵轴是 next-city retrieval success rate；柱是 trace-source treatment，黑色短横线是置零等数量 ordinary prompt tokens 的 matched control。横轴从左到右逐步区分远期历史、最近 transition 与全 trace 的作用。右 panel 纵轴是 greedy exact-count accuracy，横轴依次比较自然输入、只擦原 prompt records、只擦完整 trace、以及擦完整 prompt+trace。左侧样本为 Qwen 10 seeds/45 requests、Gemma 9 seeds/30 requests；右侧为两模型各 10 seeds/100 prompts per condition。右 panel 中 Qwen clean 0.97→trace blank 0.01，Gemma 0.70→0.12；prompt-record blank 的损伤远小于 trace blank。</figcaption></figure>
{token_condition_table}
<p class="term-note"><strong>横轴为何不是一条简单的“擦得越来越多”剂量轴。</strong> Early half、cumulative 与 recent 是三个有语义的 source partitions：前两者定位远期累计历史，recent 定位最后一次局部更新；full trace 才是它们的联合必要性。右侧 prompt-record blank 与 trace blank 则是两个互补信息源，不代表相同 token 数。因此应比较每个 condition 回答的机制问题，而不是只按 blank token 数解释柱高。</p>
<figure><h3 class="figure-title">图 5b · Answer-query head banks 的 source composition</h3><div class="figure-scroll">{answer_source_rerouting_svg(token_evidence)}</div><figcaption>每行是一个冻结 answer bank 在一种 token condition 下的 source composition。横向堆叠条在 prompt-record 与 trace-context 两个互斥 source groups 内归一；右侧 Σ 是两组 raw bank-summed mass 之和。颜色说明 attention 来源，不是 accuracy。</figcaption></figure>
<p><strong>结果分析。</strong> 两模型都可在 answer query 读 prompt，但当 trace 已存在时，最终 exact answer 对 trace content 的依赖更强。这与“count 随 trace stream 传递”一致，也允许 broad prompt retrieval 作为并行补充路径。</p>
<div class="section-conclusion"><strong>Experiment 6.1 结论。</strong> Trace 是 final answer 的主要自然信息源；prompt-broad retrieval 并未被排除，但“在 answer 时才从 prompt 重数”不足以解释 source-blank 对比。</div>

<h3>6.2 最后一条 trace 的 grammar state 能否推动正确答案</h3>
<p class="lead">6.1 证明“整段 trace content 对自然答案很重要”，但还没有定位最后是谁把 count 交给 answer query。6.2 因此构造一个受损 trace，再只恢复最后一个 item 中特定语义位置的 hidden states，检验 terminal grammar-state→answer-score 的局部因果边。</p>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>把 6.1 的 whole-trace necessity 缩小成一个可定位、可做 matched control 的 terminal bridge：最后一个 item 的哪类 state 能让正确 count candidate 的分数恢复？</div><div><span class="experiment-label">什么是一个样本</span>一条 outcome-blind 冻结的完整 trace/seed 是一个独立样本。每模型 discovery 20 seeds、confirmation 10 个完全不重叠 seeds；每 phase 恰好一半 city→rank、一半 rank→city。12 个 arms、多个 tokens 和多层 clamp 都是 seed 内配对条件，不增加独立样本量。</div><div><span class="experiment-label">损坏基线</span>对每个 parsed trace item，用 prompt 中不属于 needle records 的 background tokens 作等长替换；不重分词、不 padding、不移动 answer query。这样保留语法长度与绝对位置，但移除原 trace item 内容，得到 uninformative arm。</div><div><span class="experiment-label">五种位点</span>在最后一个 item 内预注册 full item、marker core、retrieved city、grammar-timed tail、boundary commit 五种位置。每种都从 clean run 捕获 same-position post-block states，再写入 uninformative run 的相同 positions。Discovery 仅用 mean restoration 冻结最大的非-full geometry；两模型都选到 marker core，随后在独立 10-seed confirmation 上复验。</div><div><span class="experiment-label">层与时序</span>Qwen 从 L19、Gemma 从 L16 开始，在每个后续 decoder block 的同一 terminal positions 重施 clean state（cumulative clamp）。Answer query hidden state从未直接 patch；它只能通过正常 attention 读取这些 terminal positions。三个正式对比 arms（uninformative、semantic restore、matched-random restore）的 visible token prefix 完全相同；clean arm只作自然参考，不进入这两个差分。</div><div><span class="experiment-label">两个正式差分</span><span class="formula">Restoration = Margin<sub>semantic restore</sub> − Margin<sub>uninformative</sub><br>Specificity = Margin<sub>semantic restore</sub> − Margin<sub>matched ordinary state</sub></span>第一个问“补回语义 state 能否救回分数”；第二个问“相同 token 预算、相近深度的一般 clean state 是否也能做到”。</div><div><span class="experiment-label">计分</span>对候选答案字符串 1,…,10 做 teacher-forced sequence scoring，记 S(j) 为候选 <em>j</em> 的完整 sequence log score：<span class="formula">Margin(N)=S(N)−max<sub>j≠N</sub>S(j), &nbsp; j∈{{1,…,10}}</span>Margin&gt;0 才表示 gold N 在十个候选中排名第一；这里的单位是 log-score margin，不是“恢复了几个 counts”或 accuracy 百分点。</div><div><span class="experiment-label">具体数字例子</span>Gold N=6。若 uninformative 的 margin=−1.2，marker restore 后=+0.8，则 restoration=+2.0；若 matched ordinary state 后仍为−0.9，则 specificity=+1.7。这个结果说明“正确 marker state”在固定 pre-answer context 中有功能，不等于单独一个 marker 就能让自由生成必然输出 6。</div></div>
{terminal_sample_table}
<p class="term-note"><strong>实际规模。</strong> 每条 trace 有 clean + uninformative + 5 semantic restores + 5 matched-random restores = 12 arms。因而每模型 discovery 是 20×12=240 condition rows，confirmation 是 10×12=120 rows；两模型合计 720 rows。统计仍先在 seed 内求差，再让 seeds 等权。</p>
{terminal_geometry_table}
{terminal_arm_table}
<div class="figure-primer"><div><strong>第一行</strong>正确 terminal state 相对 uninformative trace，让 gold-number margin 上升多少。</div><div><strong>第二行</strong>正确 terminal state 相对 equal-token ordinary-state patch，多提供多少恢复。</div><div><strong>坐标</strong>横轴单位是 log-probability margin，不是正确率；绿=Qwen，紫=Gemma。</div></div>
<figure><h3 class="figure-title">图 5c · Terminal grammar-state 对 correct-count margin 的受控恢复</h3>{grouped_bars_svg('Fixed-suffix terminal-state confirmation effects', terminal_groups)}<figcaption>两行均为 10 confirmation seeds 的 seed-level paired mean effect。Qwen restoration / ordinary-state specificity 为 {float(terminal_effects['Qwen3-8B']['restoration']['mean_effect']):+.3f} / {float(terminal_effects['Qwen3-8B']['matched_random_specificity']['mean_effect']):+.3f}；Gemma 为 {float(terminal_effects['Gemma4-E4B']['restoration']['mean_effect']):+.3f} / {float(terminal_effects['Gemma4-E4B']['matched_random_specificity']['mean_effect']):+.3f}。正值表示正确 grammar state 比受损基线或普通 state 更有利于 gold count。</figcaption></figure>
<p><strong>受控结果。</strong> Terminal-state restoration 为 Qwen {ci(terminal_effects['Qwen3-8B']['restoration'])}、Gemma {ci(terminal_effects['Gemma4-E4B']['restoration'])}；相对 ordinary-state control 的 specificity 分别为 {ci(terminal_effects['Qwen3-8B']['matched_random_specificity'])} 与 {ci(terminal_effects['Gemma4-E4B']['matched_random_specificity'])}。所以这条局部边在两模型中都有实质数值效应。</p>
<div class="scope-compare"><div><h4>本实验已经确认：controlled local effect</h4><p>三个正式差分 arms 使用相同 pre-answer visible tokens、相同 answer query 与相同 candidate scorer；唯一关键差别是 terminal semantic hidden state 是否写回。因此 paired margin 变化可归因于当前 state intervention。Clean arm只提供自然参照。</p><p><strong>对应标签：</strong>controlled only。</p></div><div><h4>本实验没有确认：free-running sufficiency</h4><p>更强命题是：只补这一处 state 后，让模型自由继续运行，它是否必然改变最终 greedy 数字。自由运行可同时读取其他 trace states、回 prompt、或在生成后续 token 时自我修正，所以局部 log-score 效应不自动推出全局输出改变。</p><p>Qwen 的 free-running extension 在 distribution、expected count、margin 和 exact answer 上没有稳定恢复，见 Appendix C。</p></div></div>
<div class="plain-language"><strong>为什么总览只写 controlled only？</strong> “Confirmed”若不加限定，读者很容易理解成“marker state 单独足以决定最终自然答案”。我们实际确认的是更窄也更严谨的命题：<strong>在一个位置与后续输入都受控的 counterfactual context 中，恢复正确 terminal marker state 会提高 gold count 的 candidate margin，而且优于等预算 ordinary-state patch。</strong> 这个局部因果结果可靠；只是它没有覆盖自由生成的全部旁路与自我修正，因此不升级成全局 sufficiency。</div>
<div class="section-conclusion"><strong>Experiment 6.2 结论。</strong> 两模型的 terminal grammar state 都能在严格配对、固定 answer prefix 的条件下显著推动 correct-count margin；但它尚未被证明在 free-running 中单独足以决定最终答案。因此主链最后一条边保留为 controlled local bridge。</div></section>

<section id="walkthrough"><p class="eyebrow">07 · Non-thinking-style case study</p><h2>一个 outcome-blind seed 从 item 1 走到 item 10</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>复制 Non-thinking 报告里最直观的“一条轨迹走到底”展示，检查 Native 的单个 item/carrier state 在丧失其余上下文后是否足以指定答案 <em>k</em>。</div><div><span class="experiment-label">设定</span>用 identity hash outcome-blind 冻结一个 N=10 confirmation case。Prompt records 与完整 trace context 都被等长 ordinary text 擦除；随后只恢复第 <em>k</em> 个 full item、grammar carrier，或等 token ordinary state。Answer query 始终不 patch。</div><div><span class="experiment-label">计算方法</span><span class="formula">E[count | condition,k] = Σ<sub>n=1</sub><sup>10</sup> n · P(answer=n | condition,k)</span>横轴为被恢复的 occurrence <em>k</em>，理想充分状态应产生 y=<em>k</em> 对角线。</div><div><span class="experiment-label">简单例子</span>当只恢复第 4 个 item 时，若该 state 是独立 counter，则 expected count 应靠近 4；若仍靠近 scrub baseline，则它需要其他 trace dynamics 才能被读出。</div></div>
<figure><h3 class="figure-title">图 6 · Full-context scrub 后的 expected-count path</h3>{walkthrough_svg(walkthrough)}<figcaption>橙色虚线是理想路径 y=k。Qwen 三条恢复路径大多停在高 count 区；Gemma 大多停在低 count 区。两者都没有形成 1→10 的对角路径。</figcaption></figure>
{walkthrough_table}
<div class="walkthrough-callout"><div><strong>控制成功。</strong><p>Clean case 均输出 10；uninformative baseline 分别退化为 Qwen candidate 9、Gemma candidate 1，且 P(10) 分别仅 {walkthrough['Qwen3-8B']['baselines']['uninformative']['gold_count_probability']:.4f} / {walkthrough['Gemma4-E4B']['baselines']['uninformative']['gold_count_probability']:.4f}。</p></div><div><strong>Restoration 失败。</strong><p>单 item hidden state 不能在被擦除的下游 trace 动力学中独立决定答案。V1 只擦 parsed items，遗漏 trace tail 的答案泄露，作为 failed-control audit 保留，不进入结果。</p></div></div>
<h3>7.2 无显式 running index 的 20/10-seed old-HTML restoration</h3>
<p class="lead">单 seed null 可能只是个例；同时，自然 trace 常直接写出“第六条”“count=6”等数字标签，使任何 state→count 结果都可能只是复制显式文本。这个补充实验因此复刻旧 HTML 的 full-span restore，但在一个没有 occurrence 序号的受控 trace grammar 上做正式 discovery/confirmation。</p>
<div class="experiment-frame"><div><span class="experiment-label">样本与 trace</span>每模型固定 30 个 N=10 benchmark prompts：seeds 1234–1253 用于 discovery，1254–1263 用于 confirmation。现有 300 条自然 archive 中没有一条通过严格格式审计；继续重试直到“碰巧合规”会按生成文本选样，因此我们 outcome-blind 地 teacher-force 完全相同的 grammar：每个 gold record 只写一行 <code>- City: score 61</code>。</div><div><span class="experiment-label">“无数字”具体指什么</span>这里严格排除的是<strong>计数数字</strong>：没有 item 1/2/3、first/second、running subtotal，也没有 answer 前的总数。城市的 score 数字仍保留，因为它是 needle 的证据内容而非 running count label。报告因此使用“no explicit running index”，不使用容易误解的“trace 完全没有 digit”。</div><div><span class="experiment-label">为什么叫受控反事实</span>Reasoning body 与最终 <code>Total: N</code> channel 被标准化并重新分词；prompt、native-thinking container、seed split 与 gold records 保持固定。它能检验模型在处理这种 trace 时是否形成可用 state，但不能估计模型自然生成这种 grammar 的概率，也不能冒充自然 generation confirmation。</div><div><span class="experiment-label">Uninformative receiver</span>把 prompt 中全部 10 个 needle records，以及 trace 中全部 10 个 bullet-item spans，都替换成 prompt 里同长度、非 needle 的 ordinary token windows。替换不 padding、不移动位置，sequence length 和 attention mask 完全不变；answer query 不 patch。这样 receiver 看不到任何 city/score needle content。</div><div><span class="experiment-label">Patch 几何</span>对每个 k=2,…,9，从 clean donor 捕获第 k 个完整 bullet span 的所有 token hidden states，再写到 receiver 的完全相同 positions；从 source layer 到最后一层逐层重施（cumulative clamp）。这与旧 HTML 的“clean needle-span hidden states → no-needle receiver same span”相同，不是单 token patch。</div><div><span class="experiment-label">Discovery 与 confirmation</span>Qwen 在 L18/22/26/30，Gemma 在 L16/20/24/28/32/36 扫描。按 seed-equal patched exact、再 target margin、再较晚层的冻结规则选择 Qwen L{int(unnumbered_counter['Qwen3-8B']['selected_layer'])} / Gemma L{int(unnumbered_counter['Gemma4-E4B']['selected_layer'])}；随后只在 10 个新 seeds 上跑这一层。</div><div><span class="experiment-label">实际规模</span>每个 seed 有 clean、fully-uninformative 两个 baseline；discovery 再有 8×4=32（Qwen）或 8×6=48（Gemma）个 patch arms，分别是 34/50 rows per seed。Confirmation 每 seed 只有 2+8=10 rows。两模型合计 1,880 condition rows，但独立样本仍只是每模型 20 discovery / 10 confirmation seeds。</div><div><span class="experiment-label">Early-stop 读数</span>在不自由生成后续解释的条件下，对答案 1,…,10 做完整 sequence log scoring。对 target k 定义 <span class="formula">Margin(k)=S(k)−max<sub>j≠k</sub>S(j)</span>；patch−scrub 是 target-margin gain。Hard exact 则要求 k 在十个候选中真正排名第一。</div><div><span class="experiment-label">简单例子</span>Gold trace 有 10 条，但只恢复第 4 个 item state。如果该 state 本身携带“目前完成 4 条”的计数信号，候选 4 的 margin 应相对 fully-uninformative receiver 上升；强 old-HTML sufficiency 还要求最后 argmax 直接变成 4。</div></div>
{counter_result_table}
<figure><h3 class="figure-title">图 6b · 无 running-index trace 的 full-item state 是否推动 early-stop k</h3>{internal_counter_restoration_svg(unnumbered_occurrence)}<figcaption>横轴是被恢复的 occurrence k=2…9。Panel A 是 confirmation 中 candidate k 的 patch−scrub mean margin gain；Panel B 是 hard exact accuracy gain。每个点先在对应 k 的 10 seeds 内平均。零线表示 patch 没有帮助；Qwen/Gemma 各自使用冻结 L18/L16。连续 margin 在大多数 k 上为正，但 exact gain 很小，因此曲线显示 count-aligned signal，而不是旧 HTML 式强 diagonal recovery。</figcaption></figure>
<p><strong>Confirmation 结果。</strong>Qwen 的 target-margin gain={ci(counter_margin['Qwen3-8B'])}，exact gain={ci(counter_exact['Qwen3-8B'])}，hard exact 从 {float(unnumbered_counter['Qwen3-8B']['selected_layer_metrics']['baseline_exact_accuracy']):.3f} 到 {float(unnumbered_counter['Qwen3-8B']['selected_layer_metrics']['patched_exact_accuracy']):.3f}；8/8 个 k 的 mean margin 都为正。Gemma margin gain={ci(counter_margin['Gemma4-E4B'])}，exact gain={ci(counter_exact['Gemma4-E4B'])}，hard exact 从 {float(unnumbered_counter['Gemma4-E4B']['selected_layer_metrics']['baseline_exact_accuracy']):.3f} 到 {float(unnumbered_counter['Gemma4-E4B']['selected_layer_metrics']['patched_exact_accuracy']):.3f}；7/8 个 k 为正。两模型都未达到预注册 old-HTML magnitude gate（patched exact≥0.50、gain≥0.25、mean margin&gt;0、至少 6/8 k 为正）。</p>
<div class="scope-compare"><div><h4>可以得到的结论</h4><p>在没有显式 running-index label 的受控 trace 中，第 k 个 full-item hidden state 含有跨 seed 可复现、与 k 对齐的可转移信号；把它写入无 needle receiver，会提高候选 k 的分数。这个结果排除了“只是在复制 item-number token”这一最直接混淆。</p></div><div><h4>仍不能得到的结论</h4><p>单个 item state 并不足以重建完整 counter：hard argmax 很少直接变成 k，而且该 trace 是 teacher-forced counterfactual，不是自然生成样本。更合理的机制解释是 count state 分布在 recurrent trace dynamics 中，单 span 携带其中一部分，但后续 propagation/readout 仍是必要条件。</p></div></div>
<div class="section-conclusion"><strong>Experiment 7 结论。</strong> 原单-seed natural-trace scrub 是 descriptive null；新的 20/10-seed no-running-index panel 给出两模型均可复现的 count-aligned margin gain，但强 old-HTML early-stop sufficiency 失败。因此最稳妥的 claim 是“存在受控可转移的 internal count signal”，而不是“一个 span 就是完整 internal counter”。</div></section>

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
<p><strong>为何不是矛盾。</strong> Fixed-suffix patch 保留了后续 token trajectory，只问某个 terminal grammar state 能否局部提高 gold-number margin；free-running 实验允许后续状态、多个 trace sources、prompt retrieval 与自我校正共同决定答案。某个 ordinary-position control 可产生 margin 变化，也不等于 clean carrier 把完整 distribution 恢复到了 clean。</p><div class="section-conclusion"><strong>Appendix C 结论。</strong> Qwen 的 local terminal bridge 保留在正文，但自然 free-running count recovery 不成立；因此报告使用 controlled only，而不把 terminal grammar state 写成 answer 的独立充分统计量。</div></details>

<details class="appendix-block" open><summary>Appendix D · 失败 control 审计：ordinary-state clamp 与 single-seed scrub</summary>
<h3>D.1 为什么 5.2 的 ordinary-state clamp 不进入主结果</h3><p>这个 arm 原想控制“只要在同样的 positions 写入同样数量的 hidden vectors，commit 就会自然恢复”。但结果显示，将语义不相容的非-item states 主动写入 carrier positions 会产生很大的额外 damage；它不是一个接近“什么都不做”的 inert placebo。所以 clean carrier 相对该 arm 的巨大优势主要表明“普通 state 会主动破坏”，不能校准 carrier restoration 的因果大小。本版因此删去其数值、图和正文 claim，只保留这条 protocol audit。</p><div class="section-conclusion"><strong>D.1 结论。</strong>5.2 的主证据只是 selected-mask damaged baseline 与 selected-mask + clean-carrier restoration 的配对差；ordinary-state arm 不提供有效 specificity 证据。</div>
<h3>D.2 为什么第一版 single-seed scrub 不能解释，第二版如何修正</h3>
<p><strong>这个 appendix 在审计什么。</strong> 正文图 6 想检验：当 prompt 和其余 trace 都不再提供 count clue 时，只恢复第 <em>k</em> 个 item/carrier state，是否足以让答案接近 <em>k</em>。要解释这个实验，首先必须确认“什么都不恢复”的 baseline 真的失去 count 信息。</p>
<div class="scope-compare"><div><h4>V1：只擦 parser 找到的 item spans</h4><p>Prompt record spans 被替换，parser 明确认出的 numbered/bulleted items 也被替换；但 parser item 之后、answer query 之前的 trace tail 没有全部覆盖。</p><p><strong>问题：</strong>tail 可能含“therefore the total is 10”一类可见 clue，也可能已有能被 answer query 读取的 final-count hidden state。于是 baseline 仍可能知道答案。</p></div><div><h4>V2：擦完整信息源</h4><p>Prompt records 与整个 trace source——包括 items、连接文本和 terminal tail——都用等长 ordinary tokens 替换；answer query 不 patch。</p><p><strong>先验检查：</strong>只有 uninformative baseline 的答案表现确实明显退化后，单 item/carrier restoration 才开始解释。</p></div></div>
<div class="experiment-frame"><div><span class="experiment-label">具体例子</span>Gold N=10，我们想测试“只恢复第 4 个 item 是否让答案走向 4”。如果 V1 遗留的 tail 仍写着“所以一共有 10 条”，模型输出 10 既可能来自 tail，也可能来自被恢复 item；即便恢复 item 4 后仍输出 10，也不能判定 item 4 没有 count 信息。V2 删除这条旁路后，baseline 先退化，再恢复 item 4，结果才可解释。</div><div><span class="experiment-label">等长替换的作用</span>替换后的 token budget 与位置拓扑保持一致，避免把“信息被删掉”与“序列突然变短、answer query 移位”混在一起。</div><div><span class="experiment-label">V2 仍然是什么</span>它仍只是 outcome-blind 单 seed case study，不是 20/10 seed 群体 confirmation；其作用是给 sufficiency 一个直观而严格的 sanity check。</div></div>
<figure><h3 class="figure-title">图 D1 · V1 与 V2 的 scrub coverage</h3>{failed_control_svg()}<figcaption>从左到右表示 answer query 可能读取的信息源。上排 V1 只覆盖 prompt records 与 parser-observed items，橙框 trace tail 仍可泄露 final count，因此 V1 的 restoration 正负结果都不可归因。下排 V2 用等长 ordinary tokens 覆盖 prompt records 和完整 trace context，并要求 uninformative baseline 先退化。图只表示 control topology，不表示 effect size。</figcaption></figure>
<div class="plain-language"><strong>最简单的判断规则：</strong>如果“什么都不恢复”的 baseline 还答得很好，就不能用该实验判断单个 state 是否充分。V1 没满足这条规则；V2 满足后才进入正文。</div>
<div class="section-conclusion"><strong>Appendix D 结论。</strong>两个 failed controls 都不进入主 claim：5.2 的 non-inert state clamp 不用于 specificity；single-seed V1 暴露 trace-tail leakage，正文图 6 只使用 V2。V2 的 null 可以写成“在这两个冻结 case 中，单 item/carrier 不能在完整 source scrub 后独立指定 count”，不能外推成“模型中不存在 count state”。</div></details>

<details class="appendix-block" open><summary id="appendix-e">Appendix E · 其他 grammar 的 attention maps（8 张 SVG 已内嵌在本 HTML）</summary>
<p>正文图 2 使用冻结主线 <code>adjacent_rank_after_city</code>。这里补两种不同表面结构，检查 target-following attention 是否只在正文那一种写法中出现。展开本 appendix 后，E1–E8 会逐张显示；每张图是独立 SVG，已直接写进 HTML，不依赖本地图片路径或 Git 中的实验目录。</p>
<div class="appendix-e-index"><div><strong>E1–E4 · same-unit rank-before-city</strong>序号在 city 前；Qwen/Gemma 各有一张 layer×head atlas 和一张 ordinal×ranked-head map。</div><div><strong>E5–E8 · structural-invariant bullet</strong>没有显式数字 rank；Qwen/Gemma 同样各有两种 map，用来检查 heads 是否只追踪数字 marker。</div></div>
<p class="appendix-e-proof"><strong>读图规则：</strong>layer×head 图回答“哪些层、哪些 heads 对正确 target 较亮”；ordinal×head 图回答“同一批 ranked heads 是否随 target ordinal 1→10 移动”。每张图有自己的色标，颜色深浅不能跨模型或 grammar 比 raw magnitude。Ordinal 图保留横向滚动以保证 head 标签可读。</p>
<div class="attention-atlas-stack appendix-e-gallery"><figure class="appendix-e-figure" id="figure-e1"><h3 class="figure-title">图 E1 · Qwen same-unit rank-before-city · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_same_head']}</div><figcaption>横轴=Qwen head index H0–H31，纵轴=layer L0–L35；每格颜色是该 grammar 在 exact-P0 query 对正确 next-record span 的 discovery seed-equal raw attention mass。白边格是高排名 heads。它定位 heads，但不显示 target ordinal。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e2"><h3 class="figure-title">图 E2 · Gemma same-unit rank-before-city · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_same_head']}</div><figcaption>横轴=Gemma head index H0–H7，纵轴=layer L0–L41；其余定义与 E1 相同。Gemma 使用独立色标，不能用色深与 Qwen 比绝对 attention mass。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e3"><h3 class="figure-title">图 E3 · Qwen same-unit rank-before-city · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_same_ordinal']}</div><figcaption>横轴=按 discovery score 排序的 Qwen heads，纵轴=正确 next record 的 ordinal；颜色=20 discovery seeds 的 seed-equal raw target mass。纵向多个 ordinal 上反复出现亮列，表示同一 head 可跨 occurrence 重复参与 routing。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e4"><h3 class="figure-title">图 E4 · Gemma same-unit rank-before-city · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_same_ordinal']}</div><figcaption>轴与 E3 相同，但列是 Gemma 的 ranked heads。只看同一图内亮列是否跨 ordinal 延续；不要把图宽或色深与 Qwen 当作机制容量比较。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e5"><h3 class="figure-title">图 E5 · Qwen structural-invariant bullet · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_bullet_head']}</div><figcaption>Bullet grammar 没有显式数字 rank。横轴=head、纵轴=layer、颜色=正确 target mass。仍有局部亮 heads，说明 localization 不完全依赖“第 k 条”的数字 token。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e6"><h3 class="figure-title">图 E6 · Gemma structural-invariant bullet · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_bullet_head']}</div><figcaption>轴定义与 E5 相同，使用 Gemma 自己的色标。该图是 grammar-robustness visualization，不改变 frozen Top-6 的正式因果 bank。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e7"><h3 class="figure-title">图 E7 · Qwen structural-invariant bullet · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_bullet_ordinal']}</div><figcaption>横轴=Qwen discovery-ranked heads，纵轴=target ordinal，颜色=seed-equal raw target mass。它检查没有数字 marker 时，heads 是否仍随正确 record ordinal 路由。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e8"><h3 class="figure-title">图 E8 · Gemma structural-invariant bullet · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_bullet_ordinal']}</div><figcaption>轴定义与 E7 相同；该图只支持 descriptive target-following pattern。正式因果必要性仍来自正文 frozen Top-6 selected-vs-random ablation。</figcaption></figure></div>
<div class="section-conclusion"><strong>Appendix E 结论。</strong> 八张对应图显示 target-following attention 不只存在于正文的 rank-after-city grammar，也可在 rank-before-city 与无显式数字的 bullet grammar 中定位。它们扩展的是可视化稳健性，不把 descriptive attention 升格为新的因果 confirmation。</div></details>
<div class="section-conclusion"><strong>Appendix 结论。</strong> 这些 null 缩小了主张：我们确认一条可干预 recurrent pathway，但不确认单头排他中介、单 state 全局充分，或所有 grammar 与距离上完全同质的 circuit。</div></section>

<section id="audit"><p class="eyebrow">09 · Boundaries and reproducibility</p><h2>边界、复现与底层文件</h2>
<ul><li>本报告证明一条 pathway，不证明唯一性、排他性或所有 grammar 共用完全相同的 heads。</li><li>CI 与 p-value 保留用于审计；正文的“强/弱”判断同时考虑 effect size、控制组和跨 phase 复现。</li><li>单 seed walkthrough 不进入 discovery/confirmation gate；V2 是在 V1 暴露 trace-tail 泄露后修正的 exploratory control。</li><li>Qwen 与 Gemma 的状态几何、bank 宽度和最后一条边不同，不强行合并成完全同构 circuit。</li></ul>
<details class="paper-appendix"><summary>底层报告与外部证据包</summary><div class="source-list"><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">P0 targeted-retrieval atlas</a><br><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a><br><a href="NiaH_Geometry_Comparison.html">Representation geometry</a><br><span>逐 seed、逐 arm、claim-gate 与运行审计文件保存在外部实验归档，不随 Git 仓库分发。报告中的聚合值与输入哈希已冻结；复算时通过构建器参数挂载对应 evidence bundle。</span></div></details>
<p class="audit">Generated UTC: {esc(generated)}<br>Schema: realistic_niah_v5_native_thinking_restructured_v7</p></section>

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
        *grammar_anchor_paths.values(),
        *(args.snapshot_root / "commit_state_query_20260822" / model / "commit_to_query_complete.json" for model in MODELS),
        args.snapshot_root / "qwen_grammar_span_decomposition_complete.json",
        args.snapshot_root / "gemma_grammar_span_decomposition_complete.json",
        args.snapshot_root / "targeted_counter_20260822" / "Qwen3-8B" / "targeted_counter_complete.json",
        *(args.snapshot_root / "single_seed_walkthrough_20260822_v2" / model / "analysis" / "walkthrough_complete.json" for model in MODELS),
        *(
            args.ncc_supplement_root / model / "ncc_analysis" / "claim_gates.json"
            for model in MODELS
        ),
        *(
            args.ncc_supplement_root
            / model
            / "ncc_analysis"
            / "layerwise_timing_diagnostic.json"
            for model in MODELS
        ),
        args.stratified_ncc_root
        / "synthesis"
        / "stratified_ncc_synthesis.json",
        *(
            args.stratified_ncc_root
            / model
            / timing
            / "analysis"
            / "claim_gates.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        *(
            args.stratified_ncc_root
            / model
            / "stratified_ncc_input_manifest.json"
            for model in MODELS
        ),
        *(
            args.stratified_ncc_root
            / model
            / f"{timing}_site_audit.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        args.logit_margin_root
        / "synthesis"
        / "targeted_logit_margin_synthesis.json",
        *(
            args.logit_margin_root
            / model
            / timing
            / "analysis"
            / "claim_gates.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        *(
            args.logit_margin_root
            / model
            / "targeted_logit_margin_complete.json"
            for model in MODELS
        ),
        *(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "claim_gates.json"
            for model in MODELS
        ),
        *(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "occurrence_metrics.csv"
            for model in MODELS
        ),
        *(args.ncc_supplement_root / model / "generation_manifest.json" for model in MODELS),
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
        "schema_version": "realistic_niah_v5_native_thinking_restructured_v7",
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
            "ncc_frozen_results_independently_reproduced": True,
            "ncc_timing_stratified_recapture_complete": True,
            "ncc_city_to_rank_marker_tokens_excluded": True,
            "ncc_bank_matched_timing_raw_direction_corresponds_across_models": True,
            "ncc_cross_model_effect_size_comparison_allowed": False,
            "ncc_model_by_mask_interaction_confirmed": False,
            "qwen_historical_ncc_pooled_directionally_supported": True,
            "qwen_stratified_rank_to_city_readout_validity_pass": False,
            "gemma_ncc_pooled_directionally_supported": False,
            "gemma_stratified_rank_to_city_directional_specific_only": True,
            "ncc_marker_free_city_to_rank_damage_supported": False,
            "ncc_readout_validity_gate_post_analysis": True,
            "qwen_full_bank_to_late_ncc_confirmed": False,
            "gemma_full_bank_to_late_ncc_confirmed": False,
            "gemma_l17_rank_before_midlayer_damage_exploratory": True,
            "direct_count_output_margin_complete": True,
            "direct_margin_all_clean_readouts_valid": True,
            "qwen_direct_margin_directional_specific_supported": False,
            "gemma_direct_margin_both_timings_directional_specific": True,
            "gemma_direct_margin_interval_confirmed": False,
            "direct_margin_confirmation_pristine_prospective": False,
            "direct_margin_model_by_mask_interaction_tested": False,
            "no_running_index_count_signal_confirmed": True,
            "no_running_index_single_span_strong_sufficiency": False,
            "no_running_index_panel_natural_generation": False,
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
    parser.add_argument(
        "--ncc-supplement-root",
        type=Path,
        default=Path("work_remote_snapshots/ncc_unnumbered_supplement_20260823"),
    )
    parser.add_argument(
        "--stratified-ncc-root",
        type=Path,
        default=Path("work_remote_snapshots/stratified_ncc_20260823"),
    )
    parser.add_argument(
        "--logit-margin-root",
        type=Path,
        default=Path("work_remote_snapshots/targeted_logit_margin_20260823_v2"),
    )
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
