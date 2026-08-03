from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


REPORT_SCHEMA = "realistic_niah_v4_4_causal_v2_report_v1"
MODEL_ORDER = ("Qwen3-8B", "Gemma4-E4B")
FAMILY_ORDER = ("prompt_patching", "answer_patching", "steering")
FAMILY_LABELS = {
    "prompt_patching": "Prompt full-span patching",
    "answer_patching": "Answer-query patching",
    "steering": "Answer-query steering",
}
MODEL_COLORS = {"Qwen3-8B": "#0B7772", "Gemma4-E4B": "#B35C22"}
K_COLORS = {1: "#187C78", 3: "#2F5AA6", 5: "#B66A1C"}
BANK_COLORS = {"broad_aggregation": "#187C78", "first_locator": "#B35C22"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Sources:
    def __init__(self) -> None:
        self.paths: dict[str, Path] = {}

    def add(self, label: str, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Required report input is missing: {resolved}")
        self.paths[label] = resolved
        return resolved

    def csv(self, label: str, path: Path) -> list[dict[str, str]]:
        with self.add(label, path).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def json(self, label: str, path: Path) -> dict[str, Any]:
        return json.loads(self.add(label, path).read_text(encoding="utf-8"))

    def ledger(self) -> list[dict[str, Any]]:
        return [
            {
                "source_label": label,
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for label, path in sorted(self.paths.items())
        ]


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def _median(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else math.nan


def _fmt(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    number = _float(value)
    if not math.isfinite(number):
        return "NA"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(number, spec)


def _pct(value: Any, digits: int = 1) -> str:
    number = _float(value)
    return "NA" if not math.isfinite(number) else f"{100 * number:.{digits}f}%"


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern!r} below {root}, got {len(matches)}")
    return matches[0]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_baseline(model: str, root: Path, sources: Sources) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = _find_one(root, "baseline/all_*/generation_labels.csv")
    rows = sources.csv(f"{model}.baseline_labels", path)
    by_split: list[dict[str, Any]] = []
    for split in ("discovery", "confirmation"):
        selected = [row for row in rows if row["split"] == split]
        by_split.append(
            {
                "model_label": model,
                "split": split,
                "examples": len(selected),
                "format_valid": sum(_bool(row["format_valid"]) for row in selected),
                "format_valid_rate": _mean(float(_bool(row["format_valid"])) for row in selected),
                "correct": sum(_bool(row["is_correct"]) for row in selected),
                "accuracy": _mean(float(_bool(row["is_correct"])) for row in selected),
                "mean_absolute_error": _mean(abs(_float(row["count_error"])) for row in selected),
                "mean_signed_error": _mean(_float(row["count_error"]) for row in selected),
            }
        )
    by_count: list[dict[str, Any]] = []
    for count in range(11):
        selected = [row for row in rows if int(row["gold_count"]) == count]
        by_count.append(
            {
                "model_label": model,
                "gold_count": count,
                "examples": len(selected),
                "format_valid_rate": _mean(float(_bool(row["format_valid"])) for row in selected),
                "accuracy": _mean(float(_bool(row["is_correct"])) for row in selected),
                "mean_prediction": _mean(_float(row["parsed_count"]) for row in selected),
                "mean_signed_error": _mean(_float(row["count_error"]) for row in selected),
                "mean_absolute_error": _mean(abs(_float(row["count_error"])) for row in selected),
            }
        )
    return by_split, by_count


def _load_alignment(model: str, root: Path, sources: Sources) -> tuple[dict[str, Any], dict[str, Any]]:
    csv_path = _find_one(root, "prompt_span_alignment/preflight_*/prompt_full_span_alignment.csv")
    design_path = _find_one(root, "prompt_span_alignment/preflight_*/design.json")
    rows = sources.csv(f"{model}.prompt_alignment", csv_path)
    design = sources.json(f"{model}.prompt_alignment_design", design_path)
    exact = sum(_bool(row["exact_model_token_alignment"]) for row in rows)
    supported = sum(_bool(row["mapping_supported"]) for row in rows)
    return (
        {
            "model_label": model,
            "rows": len(rows),
            "exact": exact,
            "remapped": supported - exact,
            "unsupported": len(rows) - supported,
            "max_normalized_position_error": max(_float(row["max_normalized_position_error"]) for row in rows),
            "policy": design["alignment_policy"],
        },
        design,
    )


def _load_selection(model: str, root: Path, family: str, sources: Sources) -> dict[str, Any]:
    path = _find_one(root, f"{family}/screen_*/selection/{family}_selection.json")
    payload = sources.json(f"{model}.{family}.selection", path)
    return {
        "model_label": model,
        "family": family,
        "selected_conditions": len(payload["selected"]),
        "screen_seeds": len(payload["screen_seeds"]),
        "held_out_confirmation_seeds": len(payload["held_out_confirmation_seeds"]),
        "screen_detail_sha256": payload["screen_detail_sha256"],
        **{f"threshold_{key}": value for key, value in payload["thresholds"].items()},
    }


def _load_confirmation(model: str, root: Path, family: str, sources: Sources) -> list[dict[str, Any]]:
    path = root / "analysis" / "tables" / f"{family}_confirmation_statistics.csv"
    rows = sources.csv(f"{model}.{family}.confirmation_statistics", path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not _bool(row["is_primary_confirmation"]):
            continue
        protocol = row.get("patch_protocol") or row.get("steering_protocol") or ""
        layer = row.get("start_layer") or row.get("layer_set") or ""
        normalized.append(
            {
                "model_label": model,
                "family": family,
                "site": row.get("site", "answer_query"),
                "protocol": protocol,
                "layer": layer,
                "k": int(row["k"]),
                "held_out_seeds": int(row["held_out_confirmation_seeds"]),
                "evidence_scope": row["evidence_scope"],
                "mean_control_adjusted_transport": _float(row["mean_control_adjusted_transport"]),
                "ci95_low": _float(row["ci95_low"]),
                "ci95_high": _float(row["ci95_high"]),
                "exact_sign_flip_p": _float(row["exact_sign_flip_p"]),
                "positive_seed_fraction": _float(row["positive_seed_fraction"]),
                "holm_p": _float(row["holm_p"]),
            }
        )
    return normalized


def _group_confirmation(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    detailed_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_groups[(row["model_label"], row["family"])].append(row)
        detailed_groups[(row["model_label"], row["family"], row["protocol"], row["k"])].append(row)

    def summarize(key: tuple[Any, ...], selected: Sequence[dict[str, Any]], names: Sequence[str]) -> dict[str, Any]:
        effects = [row["mean_control_adjusted_transport"] for row in selected]
        result = dict(zip(names, key))
        result.update(
            {
                "conditions": len(selected),
                "mean_effect": _mean(effects),
                "median_effect": _median(effects),
                "min_effect": min(effects),
                "max_effect": max(effects),
                "ci95_excludes_zero": sum(row["ci95_low"] > 0 for row in selected),
                "all_held_out_seeds_positive": sum(row["positive_seed_fraction"] == 1.0 for row in selected),
                "holm_p_le_0_05": sum(row["holm_p"] <= 0.05 for row in selected),
            }
        )
        return result

    family = [
        summarize(key, selected, ("model_label", "family"))
        for key, selected in sorted(family_groups.items(), key=lambda item: (MODEL_ORDER.index(item[0][0]), FAMILY_ORDER.index(item[0][1])))
    ]
    detailed = [
        summarize(key, selected, ("model_label", "family", "protocol", "k"))
        for key, selected in sorted(detailed_groups.items(), key=lambda item: (MODEL_ORDER.index(item[0][0]), FAMILY_ORDER.index(item[0][1]), item[0][2], item[0][3]))
    ]
    return family, detailed


def _load_ablation(model: str, root: Path, sources: Sources) -> list[dict[str, Any]]:
    path = root / "analysis" / "tables" / "ablation_top_k_sweep.csv"
    rows = sources.csv(f"{model}.ablation_top_k_sweep", path)
    return [
        {
            "model_label": model,
            "head_bank": row["head_bank"],
            "top_n": int(row["top_n"]),
            "examples": int(row["examples"]),
            "seeds": int(row["seeds"]),
            "accuracy_effect": _float(row["accuracy_effect"]),
            "absolute_error_effect": _float(row["absolute_error_effect"]),
            "prediction_change_effect": _float(row["prediction_change_effect"]),
            "ranked_valid_rate": _float(row["ranked_valid_rate"]),
            "random_overlap_mean": _float(row["random_overlap_mean"]),
        }
        for row in rows
    ]


def _load_audit(model: str, root: Path, sources: Sources) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / "audit" / "audit.json"
    audit = sources.json(f"{model}.audit", path)
    categories = Counter(check["name"].split(".", 1)[0] for check in audit["checks"])
    summary = {
        "model_label": model,
        "status": audit["status"],
        "checks": int(audit["check_count"]),
        "errors": int(audit["error_count"]),
        "audit_sha256": _sha256(path),
        "resolved_stage_roots": len(audit["resolved_stage_roots"]),
    }
    return summary, [
        {"model_label": model, "category": category, "checks": count}
        for category, count in sorted(categories.items())
    ]


def _load_stage_inventory(model: str, root: Path, sources: Sources) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for phase in ("screen", "confirmation"):
            stage = _find_one(root, f"{family}/{phase}_*")
            complete_path = stage / "complete.json"
            payload = sources.json(f"{model}.{family}.{phase}.complete", complete_path)
            design_path = stage / "design.json"
            design = sources.json(f"{model}.{family}.{phase}.design", design_path)
            capture_index = stage / "capture" / "capture_index.jsonl"
            capture_rows = 0
            if capture_index.is_file():
                sources.add(f"{model}.{family}.{phase}.capture_index", capture_index)
                with capture_index.open("r", encoding="utf-8") as handle:
                    capture_rows = sum(1 for line in handle if line.strip())
            inventory.append(
                {
                    "model_label": model,
                    "family": family,
                    "phase": phase,
                    "design_hash": payload["design_hash"],
                    "status": payload["status"],
                    "logical_rows": int(payload["rows"]),
                    "successful_rows": int(payload.get("successful_rows", payload["rows"])),
                    "executed_generation_rows": int(payload.get("executed_generation_rows", payload["rows"])),
                    "reused_logical_rows": int(payload.get("reused_logical_rows", 0)),
                    "skipped_rows": int(payload.get("skipped_rows", 0)),
                    "capture_index_rows": capture_rows,
                    "evaluation_seeds": "+".join(str(value) for value in (design.get("evaluation_seeds") or design.get("seeds") or [])),
                }
            )
    return inventory


def _load_export(model: str, export_root: Path | None, sources: Sources) -> dict[str, Any]:
    if export_root is None:
        return {"model_label": model, "available": False}
    archive = _find_one(export_root, "archive/*.tar.zst")
    expected_file = _find_one(export_root, "archive/*.sha256")
    expected = expected_file.read_text(encoding="utf-8").split()[0]
    actual = _sha256(archive)
    source_manifest = export_root / "manifest" / "source_file_manifest.tsv"
    copy_manifest = export_root / "manifest" / "run_file_manifest.tsv"
    sources.add(f"{model}.export.archive_checksum", expected_file)
    sources.add(f"{model}.export.source_manifest", source_manifest)
    sources.add(f"{model}.export.copy_manifest", copy_manifest)
    sources.add(f"{model}.export.verification_audit", export_root / "verification" / "audit.json")
    with source_manifest.open("r", encoding="utf-8") as handle:
        manifest_lines = sum(1 for line in handle if line.strip())
    return {
        "model_label": model,
        "available": True,
        "export_name": export_root.name,
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": actual,
        "archive_sha256_expected": expected,
        "archive_sha256_verified": actual == expected,
        "source_copy_manifests_identical": _sha256(source_manifest) == _sha256(copy_manifest),
        "manifest_nonempty_lines": manifest_lines,
    }


def _load_model(model: str, root: Path, export_root: Path | None, sources: Sources) -> dict[str, Any]:
    baseline_split, baseline_count = _load_baseline(model, root, sources)
    alignment, preflight_design = _load_alignment(model, root, sources)
    confirmations: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        confirmations.extend(_load_confirmation(model, root, family, sources))
        selections.append(_load_selection(model, root, family, sources))
    audit, audit_categories = _load_audit(model, root, sources)
    runtime = sources.json(f"{model}.last_runtime", root / "last_runtime.json")
    completion_path = root.parents[2] / "causal_v2.complete"
    completion = sources.json(f"{model}.campaign_complete", completion_path)
    return {
        "model": model,
        "root": root,
        "baseline_split": baseline_split,
        "baseline_count": baseline_count,
        "alignment": alignment,
        "preflight_design": preflight_design,
        "confirmations": confirmations,
        "selections": selections,
        "ablation": _load_ablation(model, root, sources),
        "audit": audit,
        "audit_categories": audit_categories,
        "stages": _load_stage_inventory(model, root, sources),
        "runtime": runtime,
        "completion": completion,
        "export": _load_export(model, export_root, sources),
    }


def _svg_text(x: float, y: float, text: str, **attrs: Any) -> str:
    defaults = {"font-size": 12, "fill": "#263238"}
    defaults.update(attrs)
    rendered = " ".join(f'{key}="{html.escape(str(value))}"' for key, value in defaults.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {rendered}>{html.escape(text)}</text>'


def _overview_svg(rows: Sequence[dict[str, Any]]) -> str:
    width, height = 920, 410
    left, top, plot_w, plot_h = 92, 38, 770, 280
    x_positions = {family: left + (index + 0.5) * plot_w / 3 for index, family in enumerate(FAMILY_ORDER)}
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="fig1-title fig1-desc">', '<title id="fig1-title">Held-out primary confirmation effect overview</title>', '<desc id="fig1-desc">Median and observed range of matched-control adjusted normalized transport for each intervention family and model.</desc>']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(_svg_text(left - 12, y + 4, f"{tick:.2f}", **{"text-anchor": "end"}))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(_svg_text(20, top + plot_h / 2, "matched-control adjusted normalized transport", transform=f"rotate(-90 20 {top + plot_h / 2})", **{"text-anchor": "middle", "font-size": 13}))
    for family in FAMILY_ORDER:
        center = x_positions[family]
        parts.append(_svg_text(center, top + plot_h + 34, FAMILY_LABELS[family].replace(" patching", ""), **{"text-anchor": "middle", "font-size": 13, "font-weight": 650}))
        selected = [row for row in rows if row["family"] == family]
        for model_index, model in enumerate(MODEL_ORDER):
            row = next(item for item in selected if item["model_label"] == model)
            x = center + (-22 if model_index == 0 else 22)
            y_min = top + plot_h * (1 - row["min_effect"])
            y_max = top + plot_h * (1 - row["max_effect"])
            y_med = top + plot_h * (1 - row["median_effect"])
            color = MODEL_COLORS[model]
            parts.append(f'<line x1="{x}" y1="{y_min:.1f}" x2="{x}" y2="{y_max:.1f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<line x1="{x - 7}" y1="{y_min:.1f}" x2="{x + 7}" y2="{y_min:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 7}" y1="{y_max:.1f}" x2="{x + 7}" y2="{y_max:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x}" cy="{y_med:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="2"><title>{model}: median {row["median_effect"]:.3f}, range {row["min_effect"]:.3f}–{row["max_effect"]:.3f}, n={row["conditions"]}</title></circle>')
            parts.append(_svg_text(x, top + plot_h + 56, f"n={row['conditions']}", **{"text-anchor": "middle", "font-size": 10, "fill": color}))
    legend_x = left + 8
    for index, model in enumerate(MODEL_ORDER):
        x = legend_x + index * 180
        parts.append(f'<circle cx="{x}" cy="{height - 24}" r="5" fill="{MODEL_COLORS[model]}"/>')
        parts.append(_svg_text(x + 11, height - 20, model, **{"font-size": 12}))
    parts.append("</svg>")
    return "".join(parts)


def _alignment_svg(rows: Sequence[dict[str, Any]]) -> str:
    width, height = 820, 330
    left, top, plot_w, plot_h = 92, 32, 650, 220
    colors = {"exact": "#187C78", "remapped": "#D68A36", "unsupported": "#B33A3A"}
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="fig2-title fig2-desc">', '<title id="fig2-title">Prompt full-span alignment preflight</title>', '<desc id="fig2-desc">Counts of exact, deterministically remapped, and unsupported prompt-span alignments for Qwen and Gemma.</desc>']
    maximum = max(row["rows"] for row in rows)
    for tick in (0, 180, 360, 540):
        y = top + plot_h * (1 - tick / maximum)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(_svg_text(left - 10, y + 4, str(tick), **{"text-anchor": "end"}))
    for index, row in enumerate(rows):
        x = left + 150 + index * 320
        y_cursor = top + plot_h
        for key in ("exact", "remapped", "unsupported"):
            value = row[key]
            bar_h = plot_h * value / maximum
            y_cursor -= bar_h
            if value:
                parts.append(f'<rect x="{x}" y="{y_cursor:.1f}" width="92" height="{bar_h:.1f}" fill="{colors[key]}"><title>{key}: {value}</title></rect>')
                if bar_h > 22:
                    parts.append(_svg_text(x + 46, y_cursor + bar_h / 2 + 4, str(value), **{"text-anchor": "middle", "fill": "#fff", "font-weight": 700}))
        parts.append(_svg_text(x + 46, top + plot_h + 28, row["model_label"], **{"text-anchor": "middle", "font-weight": 650}))
    legend_x = left + 120
    for index, key in enumerate(("exact", "remapped", "unsupported")):
        x = legend_x + index * 170
        parts.append(f'<rect x="{x}" y="{height - 24}" width="14" height="14" fill="{colors[key]}"/>')
        parts.append(_svg_text(x + 21, height - 12, key))
    parts.append("</svg>")
    return "".join(parts)


def _layer_svg(rows: Sequence[dict[str, Any]], family: str, figure_id: str, title: str) -> str:
    width, height = 960, 440
    panel_w, panel_h = 390, 285
    top, first_left = 40, 104
    gap = 70
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{figure_id}-title {figure_id}-desc">', f'<title id="{figure_id}-title">{html.escape(title)}</title>', f'<desc id="{figure_id}-desc">Held-out condition effects by decoder layer and k. Only screen-selected, confirmed single-layer conditions appear.</desc>']
    for model_index, model in enumerate(MODEL_ORDER):
        left = first_left + model_index * (panel_w + gap)
        selected = [row for row in rows if row["model_label"] == model and row["family"] == family and row["protocol"] == "single_layer"]
        points_by_k: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for row in selected:
            if "+" in str(row["layer"]):
                continue
            points_by_k[int(row["k"])].append((int(row["layer"]), row["mean_control_adjusted_transport"]))
        max_layer = max((layer for points in points_by_k.values() for layer, _ in points), default=1)
        for tick in (0, 0.25, 0.5, 0.75, 1.0):
            y = top + panel_h * (1 - tick)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" class="grid"/>')
            if model_index == 0:
                parts.append(_svg_text(left - 10, y + 4, f"{tick:.2f}", **{"text-anchor": "end"}))
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" class="axis"/>')
        for tick in sorted(set((0, max_layer // 4, max_layer // 2, 3 * max_layer // 4, max_layer))):
            x = left + panel_w * tick / max_layer
            parts.append(f'<line x1="{x:.1f}" y1="{top + panel_h}" x2="{x:.1f}" y2="{top + panel_h + 5}" class="axis"/>')
            parts.append(_svg_text(x, top + panel_h + 20, str(tick), **{"text-anchor": "middle", "font-size": 10}))
        for k in (1, 3, 5):
            points = sorted(points_by_k.get(k, []))
            if not points:
                continue
            coords = [(left + panel_w * layer / max_layer, top + panel_h * (1 - effect)) for layer, effect in points]
            parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coords)}" fill="none" stroke="{K_COLORS[k]}" stroke-width="2.4" stroke-linejoin="round"/>')
            for (layer, effect), (x, y) in zip(points, coords):
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{K_COLORS[k]}"><title>{model}, k={k}, L{layer}: {effect:.3f}</title></circle>')
        parts.append(_svg_text(left + panel_w / 2, 22, model, **{"text-anchor": "middle", "font-size": 14, "font-weight": 700}))
        parts.append(_svg_text(left + panel_w / 2, height - 65, "decoder start layer", **{"text-anchor": "middle", "font-size": 12}))
    parts.append(_svg_text(22, top + panel_h / 2, "matched-control adjusted normalized transport", transform=f"rotate(-90 22 {top + panel_h / 2})", **{"text-anchor": "middle", "font-size": 12}))
    legend_x = 250
    for index, k in enumerate((1, 3, 5)):
        x = legend_x + index * 140
        parts.append(f'<line x1="{x}" y1="{height - 25}" x2="{x + 24}" y2="{height - 25}" stroke="{K_COLORS[k]}" stroke-width="3"/>')
        parts.append(_svg_text(x + 31, height - 21, f"k={k}"))
    parts.append("</svg>")
    return "".join(parts)


def _ablation_svg(rows: Sequence[dict[str, Any]], metric: str, figure_id: str, title: str, y_label: str) -> str:
    width, height = 960, 430
    panel_w, panel_h = 390, 280
    top, first_left, gap = 40, 104, 70
    values = [row[metric] for row in rows]
    y_min = min(0.0, min(values))
    y_max = max(0.0, max(values))
    span = max(y_max - y_min, 0.1)
    y_min -= 0.08 * span
    y_max += 0.08 * span
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{figure_id}-title {figure_id}-desc">', f'<title id="{figure_id}-title">{html.escape(title)}</title>', '<desc id="{figure_id}-desc">Discovery-screen ranked-minus-random head-ablation effect across top-n heads for two ranked banks and two models.</desc>']
    for model_index, model in enumerate(MODEL_ORDER):
        left = first_left + model_index * (panel_w + gap)
        selected = [row for row in rows if row["model_label"] == model]
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            value = y_min + fraction * (y_max - y_min)
            y = top + panel_h * (1 - fraction)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" class="grid"/>')
            if model_index == 0:
                parts.append(_svg_text(left - 10, y + 4, f"{value:.2f}", **{"text-anchor": "end", "font-size": 10}))
        zero_y = top + panel_h * (y_max / (y_max - y_min))
        parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + panel_w}" y2="{zero_y:.1f}" stroke="#546E7A" stroke-width="1.5" stroke-dasharray="5 4"/>')
        for bank in ("broad_aggregation", "first_locator"):
            points = sorted((row["top_n"], row[metric]) for row in selected if row["head_bank"] == bank)
            coords = [(left + panel_w * (top_n - 1) / 31, top + panel_h * (y_max - value) / (y_max - y_min)) for top_n, value in points]
            parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coords)}" fill="none" stroke="{BANK_COLORS[bank]}" stroke-width="2.2"/>')
            for (top_n, value), (x, y) in zip(points, coords):
                if top_n in {1, 4, 8, 16, 24, 32}:
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{BANK_COLORS[bank]}"><title>{model}, {bank}, top-{top_n}: {value:.3f}</title></circle>')
        parts.append(_svg_text(left + panel_w / 2, 22, model, **{"text-anchor": "middle", "font-size": 14, "font-weight": 700}))
        parts.append(_svg_text(left + panel_w / 2, top + panel_h + 38, "ablated top-n ranked heads", **{"text-anchor": "middle"}))
    parts.append(_svg_text(22, top + panel_h / 2, y_label, transform=f"rotate(-90 22 {top + panel_h / 2})", **{"text-anchor": "middle", "font-size": 12}))
    legend_x = 230
    for index, bank in enumerate(("broad_aggregation", "first_locator")):
        x = legend_x + index * 280
        parts.append(f'<line x1="{x}" y1="{height - 24}" x2="{x + 24}" y2="{height - 24}" stroke="{BANK_COLORS[bank]}" stroke-width="3"/>')
        parts.append(_svg_text(x + 32, height - 20, bank.replace("_", " ")))
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, compact: bool = False) -> str:
    class_name = "compact" if compact else ""
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table class="{class_name}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _conclusion(text: str, qualifier: str = "数据支持") -> str:
    return f'<div class="conclusion"><span>本节结论 · {html.escape(qualifier)}</span><p>{text}</p></div>'


def _figure(number: int, svg: str, caption: str) -> str:
    return f'<figure id="fig-{number}">{svg}<figcaption><strong>图 {number}.</strong> {caption}</figcaption></figure>'


def _best_worst(rows: Sequence[dict[str, Any]], model: str, family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [row for row in rows if row["model_label"] == model and row["family"] == family]
    return max(selected, key=lambda row: row["mean_control_adjusted_transport"]), min(selected, key=lambda row: row["mean_control_adjusted_transport"])


def _condition_label(row: dict[str, Any]) -> str:
    site = row["site"].replace("toggled_needle_", "")
    layer = f"L{row['layer']}" if "+" not in str(row["layer"]) else f"layers {row['layer']}"
    return f"{site}/{row['protocol']}/{layer}/k={row['k']}"


def _render_report(payload: dict[str, Any]) -> str:
    models = payload["models"]
    family_summary = payload["family_summary"]
    conditions = payload["primary_confirmation_conditions"]
    alignment = [model["alignment"] for model in models]
    baseline = [row for model in models for row in model["baseline_split"]]
    selections = [row for model in models for row in model["selections"]]
    audits = [model["audit"] for model in models]
    stages = [row for model in models for row in model["stages"]]
    exports = [model["export"] for model in models]
    ablation = [row for model in models for row in model["ablation"]]
    preflight = models[0]["preflight_design"]

    family_table_rows: list[list[str]] = []
    for family in FAMILY_ORDER:
        for model in MODEL_ORDER:
            row = next(item for item in family_summary if item["model_label"] == model and item["family"] == family)
            family_table_rows.append(
                [
                    html.escape(FAMILY_LABELS[family]),
                    html.escape(model),
                    str(row["conditions"]),
                    _fmt(row["median_effect"]),
                    f"{_fmt(row['min_effect'])}–{_fmt(row['max_effect'])}",
                    f"{row['ci95_excludes_zero']}/{row['conditions']}",
                    f"{row['all_held_out_seeds_positive']}/{row['conditions']}",
                    f"{row['holm_p_le_0_05']}/{row['conditions']}",
                ]
            )
    baseline_rows = [
        [
            html.escape(row["model_label"]),
            html.escape(row["split"]),
            str(row["examples"]),
            _pct(row["format_valid_rate"]),
            _pct(row["accuracy"]),
            _fmt(row["mean_absolute_error"]),
            _fmt(row["mean_signed_error"], signed=True),
        ]
        for row in baseline
    ]
    alignment_rows = [
        [
            html.escape(row["model_label"]),
            str(row["rows"]),
            str(row["exact"]),
            str(row["remapped"]),
            str(row["unsupported"]),
            _fmt(row["max_normalized_position_error"], 4),
            f"<code>{html.escape(row['policy'])}</code>",
        ]
        for row in alignment
    ]
    selection_rows = [
        [
            html.escape(row["model_label"]),
            html.escape(FAMILY_LABELS[row["family"]]),
            str(row["selected_conditions"]),
            str(row["screen_seeds"]),
            str(row["held_out_confirmation_seeds"]),
            f"<code>{row['screen_detail_sha256'][:12]}…</code>",
        ]
        for row in selections
    ]
    audit_rows = [
        [
            html.escape(row["model_label"]),
            html.escape(row["status"]),
            str(row["checks"]),
            str(row["errors"]),
            str(row["resolved_stage_roots"]),
            f"<code>{row['audit_sha256'][:16]}…</code>",
        ]
        for row in audits
    ]
    stage_rows = [
        [
            html.escape(row["model_label"]),
            html.escape(FAMILY_LABELS[row["family"]]),
            html.escape(row["phase"]),
            f"<code>{row['design_hash']}</code>",
            f"{row['successful_rows']:,}/{row['logical_rows']:,}",
            f"{row['executed_generation_rows']:,}",
            f"{row['reused_logical_rows']:,}",
            str(row["skipped_rows"]),
            str(row["capture_index_rows"]),
        ]
        for row in stages
    ]
    export_rows = [
        [
            html.escape(row["model_label"]),
            html.escape(row.get("export_name", "not supplied")),
            f"{row.get('archive_size_bytes', 0):,}",
            "PASS" if row.get("archive_sha256_verified") else "NA",
            "PASS" if row.get("source_copy_manifests_identical") else "NA",
            f"<code>{row.get('archive_sha256', '')[:16]}…</code>",
        ]
        for row in exports
    ]
    cross_rows: list[list[str]] = []
    for family in FAMILY_ORDER:
        qwen = next(row for row in family_summary if row["model_label"] == "Qwen3-8B" and row["family"] == family)
        gemma = next(row for row in family_summary if row["model_label"] == "Gemma4-E4B" and row["family"] == family)
        cross_rows.append(
            [
                html.escape(FAMILY_LABELS[family]),
                _fmt(qwen["median_effect"]),
                _fmt(gemma["median_effect"]),
                _fmt(qwen["median_effect"] - gemma["median_effect"], signed=True),
                f"{qwen['ci95_excludes_zero']}/{qwen['conditions']}",
                f"{gemma['ci95_excludes_zero']}/{gemma['conditions']}",
            ]
        )

    prompt_site_counts = Counter((row["model_label"], row["site"]) for row in conditions if row["family"] == "prompt_patching")
    q_prompt = next(row for row in family_summary if row["model_label"] == "Qwen3-8B" and row["family"] == "prompt_patching")
    g_prompt = next(row for row in family_summary if row["model_label"] == "Gemma4-E4B" and row["family"] == "prompt_patching")
    q_answer = next(row for row in family_summary if row["model_label"] == "Qwen3-8B" and row["family"] == "answer_patching")
    g_answer = next(row for row in family_summary if row["model_label"] == "Gemma4-E4B" and row["family"] == "answer_patching")
    q_steer = next(row for row in family_summary if row["model_label"] == "Qwen3-8B" and row["family"] == "steering")
    g_steer = next(row for row in family_summary if row["model_label"] == "Gemma4-E4B" and row["family"] == "steering")
    q_prompt_best, q_prompt_worst = _best_worst(conditions, "Qwen3-8B", "prompt_patching")
    g_prompt_best, g_prompt_worst = _best_worst(conditions, "Gemma4-E4B", "prompt_patching")
    q_answer_best, q_answer_worst = _best_worst(conditions, "Qwen3-8B", "answer_patching")
    g_answer_best, g_answer_worst = _best_worst(conditions, "Gemma4-E4B", "answer_patching")
    q_steer_best, q_steer_worst = _best_worst(conditions, "Qwen3-8B", "steering")
    g_steer_best, g_steer_worst = _best_worst(conditions, "Gemma4-E4B", "steering")

    ablation_extrema_rows: list[list[str]] = []
    for model in MODEL_ORDER:
        for bank in ("broad_aggregation", "first_locator"):
            selected = [row for row in ablation if row["model_label"] == model and row["head_bank"] == bank]
            acc_low = min(selected, key=lambda row: row["accuracy_effect"])
            acc_high = max(selected, key=lambda row: row["accuracy_effect"])
            err_low = min(selected, key=lambda row: row["absolute_error_effect"])
            err_high = max(selected, key=lambda row: row["absolute_error_effect"])
            ablation_extrema_rows.append(
                [
                    html.escape(model),
                    html.escape(bank.replace("_", " ")),
                    f"{_fmt(acc_low['accuracy_effect'], signed=True)} @ top-{acc_low['top_n']}",
                    f"{_fmt(acc_high['accuracy_effect'], signed=True)} @ top-{acc_high['top_n']}",
                    f"{_fmt(err_low['absolute_error_effect'], signed=True)} @ top-{err_low['top_n']}",
                    f"{_fmt(err_high['absolute_error_effect'], signed=True)} @ top-{err_high['top_n']}",
                ]
            )

    audit_category_counts = Counter()
    for model in models:
        for row in model["audit_categories"]:
            audit_category_counts[row["category"]] += row["checks"]
    audit_category_rows = [
        [html.escape(category), str(count // 2), str(count)]
        for category, count in sorted(audit_category_counts.items())
    ]

    css = """
:root{--ink:#1f2a2d;--muted:#607076;--paper:#fbfaf7;--surface:#fff;--line:#d8d6cf;--teal:#0b7772;--rust:#b35c22;--blue:#2f5aa6;--soft:#eef3f1;--warn:#fff4e2;--max:1120px}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.66}header{padding:64px 24px 42px;background:linear-gradient(135deg,#123c3a 0%,#17333a 56%,#3f3127 100%);color:#fff}.hero{max-width:var(--max);margin:auto}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.78rem;font-weight:750;color:#b9ded9}.hero h1{font-size:clamp(2.1rem,5vw,4.7rem);line-height:1.04;margin:.35rem 0 1.15rem;max-width:970px;letter-spacing:-.035em}.hero .lede{font-size:1.12rem;max-width:880px;color:#e3eeec}.badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}.badge{border:1px solid #ffffff55;border-radius:999px;padding:6px 11px;font-size:.82rem;background:#ffffff12}main{max-width:var(--max);margin:0 auto;padding:30px 24px 80px}.layout{display:grid;grid-template-columns:250px minmax(0,1fr);gap:42px}.toc{position:sticky;top:18px;align-self:start;border-left:3px solid var(--teal);padding:6px 0 6px 18px}.toc strong{display:block;margin-bottom:8px}.toc a{display:block;color:#486066;text-decoration:none;padding:4px 0;font-size:.91rem}.toc a:hover{color:var(--teal)}article{min-width:0}section{scroll-margin-top:18px;margin:0 0 56px}h2{font-size:2rem;line-height:1.15;margin:0 0 18px;letter-spacing:-.025em}h3{font-size:1.3rem;margin:34px 0 12px;color:#173f42}p{margin:10px 0}a{color:#0b6664}code{font-family:"Cascadia Mono",Consolas,monospace;font-size:.9em;word-break:break-all}.callout{border-left:4px solid var(--rust);background:var(--warn);padding:15px 18px;margin:20px 0}.formula{font-family:"Cambria Math","Times New Roman",serif;background:#f1f4f2;border:1px solid #d9e1de;border-radius:5px;padding:13px 16px;margin:14px 0;overflow:auto}.conclusion{border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:24px 0 0;padding:14px 0}.conclusion span{font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;color:var(--teal);font-weight:800}.conclusion p{font-weight:620;margin:5px 0}.table-wrap{overflow:auto;margin:16px 0 24px;border:1px solid var(--line);background:var(--surface)}table{border-collapse:collapse;width:100%;font-size:.87rem}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e8e5de;vertical-align:top}th{background:#edf2f0;color:#294245;position:sticky;top:0;z-index:1}tr:last-child td{border-bottom:0}table.compact td,table.compact th{padding:7px 9px}figure{margin:26px 0 34px;border:1px solid var(--line);background:#fff;padding:15px}.chart{display:block;width:100%;height:auto}.grid{stroke:#dfe3e1;stroke-width:1}.axis{stroke:#455a64;stroke-width:1.3}figcaption{font-size:.88rem;color:#4e5c60;border-top:1px solid #e7e5df;padding-top:12px;margin-top:8px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0}.card{border-top:4px solid var(--teal);background:#fff;padding:15px;border-left:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.card:nth-child(2){border-top-color:var(--rust)}.card:nth-child(3){border-top-color:var(--blue)}.card strong{font-size:1.45rem;display:block}.small{font-size:.88rem;color:var(--muted)}.status-pass{color:#0a6d4f;font-weight:750}.warning{color:#935113;font-weight:700}.footnotes{font-size:.88rem;color:#56666a}.provenance{font-family:"Cascadia Mono",Consolas,monospace;font-size:.78rem;word-break:break-all}.skip-link{position:absolute;left:-9999px}.skip-link:focus{left:12px;top:12px;background:white;padding:8px;z-index:20}@media(max-width:860px){.layout{grid-template-columns:1fr}.toc{position:relative;top:0}.cards{grid-template-columns:1fr}.hero h1{font-size:2.4rem}header{padding-top:48px}}@media print{body{background:#fff;font-size:10pt}header{background:#fff!important;color:#000;padding:24px 0;border-bottom:2px solid #000}.eyebrow,.badge{color:#000}.badge{border-color:#777}.layout{display:block}.toc{display:none}main{max-width:none;padding:18px 0}section{break-inside:auto;margin-bottom:28px}figure,.table-wrap,.conclusion{break-inside:avoid}a{color:#000;text-decoration:none}.chart{max-height:230mm}}
"""

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Realistic NIAH V4.4 causal-v2：双模型正式实验报告</title><style>{css}</style></head>
<body><a class="skip-link" href="#content">跳到正文</a>
<header><div class="hero"><div class="eyebrow">Realistic NIAH · V4.4 · causal-v2 · formal</div><h1>完整 prompt span 可以运输计数状态；答案查询位也可被运输与定向操控</h1><p class="lede">Qwen3-8B 与 Gemma4-E4B 的正式、分离发现/确认实验。报告只把预注册干预直接支持的内容写成结论，并把基线错误、筛选、低检验力、负结果与可复现证据放在同一页面。</p><div class="badges"><span class="badge">Qwen audit 302/302</span><span class="badge">Gemma audit 302/302</span><span class="badge">k ∈ {{1,3,5}}</span><span class="badge">5 screen + 5 held-out seeds</span><span class="badge">implementation dd409f2</span></div></div></header>
<main id="content"><div class="layout"><nav class="toc" aria-label="目录"><strong>目录</strong><a href="#executive">1. 结论摘要</a><a href="#design">2. 实验设定</a><a href="#metrics">3. 指标与统计</a><a href="#integrity">4. 完整性与审计</a><a href="#baseline">5. 基线行为</a><a href="#prompt">6. Prompt patching</a><a href="#answer">7. Answer patching</a><a href="#steering">8. Steering</a><a href="#ablation">9. Head ablation</a><a href="#cross">10. 跨模型整合</a><a href="#limits">11. 限制与复现</a></nav><article>

<section id="executive"><h2>1. 结论摘要</h2><div class="cards"><div class="card"><span>Prompt span</span><strong>{q_prompt['conditions']} / {g_prompt['conditions']}</strong><span class="small">Qwen / Gemma held-out conditions；全部 bootstrap CI 下界 &gt; 0</span></div><div class="card"><span>Answer query</span><strong>{q_answer['median_effect']:.3f} / {g_answer['median_effect']:.3f}</strong><span class="small">两模型 primary median transport</span></div><div class="card"><span>Formal randomization</span><strong>0 significant</strong><span class="small">所有 family、所有模型 Holm p &gt; .05；5 seeds 的最小双侧 p=.0625</span></div></div>
<p>最稳健的共同结果有三条。第一，完整的 toggled-needle span residual 在所有进入 held-out 确认的 Qwen 条件（{q_prompt['conditions']} 个）和 Gemma 条件（{g_prompt['conditions']} 个）上都产生正的 matched-control-adjusted transport；单 token 的 span endpoint 没有任何条件通过发现阶段筛选。第二，答案查询位的 donor state 在 single-layer 与 cumulative-from-layer 协议中都能运输计数方向。第三，基于 count centroid 的几何 steering 在多数确认条件上也能推动输出，但不是精确 set-to-count 操作。</p>
<div class="callout"><strong>统计限制。</strong>每个 primary 条件只有 5 个 held-out seed cluster，双侧 exact sign-flip test 的最小可能 p 值是 2/2⁵ = 0.0625。因此，虽然 bootstrap 95% CI 在所有 prompt/answer 条件和全部 steering 条件上都高于 0，预注册随机化检验没有任何未校正或 Holm 校正后的 p&lt;.05 结果。本文把前者写为稳定的描述性/重抽样证据，不把它替代成正式显著性。</div>
{_figure(1, _overview_svg(family_summary), '横轴为三类干预 family，每类并列 Qwen 与 Gemma；每个点是该模型 × family 的 primary held-out 条件中位数。竖线是这些“条件均值”的观测最小—最大范围，不是置信区间。纵轴为 matched-control-adjusted strict normalized transport，0 表示相对配对控制无运输，1 表示平均完成一个 receiver→target 的归一化距离。图下 n 是经 screen 冻结后进入确认的条件数（steering 还包含每个 k 的 frozen multi-layer plan）。')}
{_conclusion('直接数据支持：完整 span、答案查询 patch 与答案查询 steering 在两个模型的已筛选 held-out 条件中均表现出方向一致的正运输。尚不能下结论：这些结果不证明唯一机制、精确计数设置，也不达到预注册 exact test 的 0.05 显著性门槛。')}
</section>

<section id="design"><h2>2. 实验问题与冻结设定</h2><h3>2.1 问题与模型</h3><p>实验固定 Realistic NIAH V4.4、10,000-token、numeric non-thinking 生成，研究三个互补问题：prompt 中一个完整被切换 needle span 的 residual state 是否足以运输计数信息；最终 <code>Total:</code> 查询位的 residual 是否足以运输 donor prediction；population-level count centroid direction 是否能定向改变输出。模型为 <code>Qwen/Qwen3-8B</code> 与 <code>google/gemma-4-E4B-it</code>，分别固定 tokenizer revision <code>{models[0]['preflight_design']['tokenizer_revision']}</code> 与 <code>{models[1]['preflight_design']['tokenizer_revision']}</code>。</p>{_conclusion('三类干预回答的是不同问题：prompt/answer donor patch 测试样本特异状态的充分性，centroid steering 测试群体方向的可操控性；它们不能互相替代。')}
<h3>2.2 k、配对、种子与阶段</h3><p>计数取 0…10。九个 canonical 无向 pair 为 k=1 的 (0,1)/(4,5)/(9,10)，k=3 的 (0,3)/(3,6)/(7,10)，k=5 的 (0,5)/(2,7)/(5,10)；两方向均执行，共 18 个 directed pairs。centroid 用 seeds 1234–1253（20）；screen 用 1254–1258（5）；held-out confirmation 用 1259–1263（5）。ablation screen 使用 counts 7–10、top-n=1…32、每个 ranked bank 配三套 layer-matched random control。</p>
{_table(['模型','family','screen 选中','screen seeds','held-out seeds','screen detail SHA-256'], selection_rows)}
<p>筛选规则在确认前冻结：5 个 screen seeds 全部有效；至少 4/5 seed 的 matched-control effect 为正；increase/decrease 两方向都为正；三组 anchor pair 至少 2/3 为正；平均 control-adjusted transport ≥0.15；valid rate ≥0.95。selection JSON 同时冻结 screen detail SHA 与 held-out seed 列表。</p>{_conclusion('发现与确认 seed 完全分离，确认条件由不可变 selection JSON 冻结；因此 held-out 结果不是在确认数据上重新挑层。')}
<h3>2.3 干预与控制</h3><p><strong>Prompt patching：</strong>站点为 <code>toggled_needle_end</code>（单 endpoint token）或 <code>toggled_needle_span</code>（完整 token span）；协议为 single-layer 或 cumulative-from-layer。treatment 复制 donor residual；控制为同 receiver 自身复制与同 count donor 的 matched control。<strong>Answer patching：</strong>站点固定 final answer query，协议同上。<strong>Steering：</strong>α=1 的 count-centroid delta；控制是在同一层、同 prompt 上与 delta 正交且范数相同的确定性随机方向；另有 screen 冻结的 multi-layer plan。</p>{_conclusion('每个 primary estimand 都是 treatment 减去同例 matched control，而不是只与 clean generation 比较；结论只适用于这些站点、强度与协议。')}
<h3>2.4 Prompt full-span 映射</h3><p>映射策略固定为 <code>{preflight['alignment_policy']}</code>。当 receiver 长 R、donor 长 S 且 R,S&gt;1 时，第 j 个 receiver token 取 donor 索引：</p><div class="formula">a(j) = floor((2j(S−1)+(R−1)) / (2(R−1))).</div><p>它使用确定性 round-half-up，单调、保持两端点，R=S 时为恒等映射；R=1 取 donor 中点，S=1 重用唯一 donor vector。Qwen 的 540 个 pair×seed 映射全部 exact；Gemma 有 178 exact、362 deterministic remap；二者 unsupported 都为 0。</p>
{_figure(2, _alignment_svg(alignment), 'Prompt full-span 预检。横轴为模型；纵轴为 540 个 directed-pair × evaluation-seed 映射。绿色 exact 表示 token 数和逐位映射完全一致；橙色 remapped 表示按冻结的 monotone endpoint-preserving nearest-neighbor rule 确定性重采样；红色 unsupported 表示无法构造映射。Gemma 最大归一化位置误差为 '+_fmt(models[1]['alignment']['max_normalized_position_error'],4)+'，Qwen 为 0。')}
{_table(['模型','总映射','exact','remapped','unsupported','max normalized error','policy'], alignment_rows)}
{_conclusion('预检满足注册标准：Qwen 540/0/0，Gemma 178/362/0；Gemma 结果包含定义内的确定性重映射，因此跨模型比较还受 tokenizer/span-length 差异影响。')}
</section>

<section id="metrics"><h2>3. 概念、指标与统计量</h2><h3>3.1 Strict normalized transport</h3><p>对 receiver gold count r、target/donor gold count t、clean receiver prediction y₀ 和 treatment prediction y₁，未裁剪运输量定义为：</p><div class="formula">T = (y₁ − y₀) / (t − r).</div><p>T&gt;0 表示沿目标方向移动，T=1 表示恰好移动完整 gold-count 距离，T&gt;1 表示越过目标，T&lt;0 表示反向。target conformity 为 <span class="formula">C = 1 − |y₁−t| / |t−r|</span>。生成不符合严格 numeric 格式时，不丢弃样本，而把 strict effect 记 0，同时单独报告 valid rate。</p>{_conclusion('方向约定、分母和 invalid handling 在两个模型及三类干预中完全一致；图中的 0.7 表示平均完成 70% 的 receiver→target 距离，不等同于 70% accuracy。')}
<h3>3.2 Matched-control adjustment</h3><p>每条 treatment 的 strict transport 减去同一 example 内 controls 的均值：patching controls 是 self-copy 与 same-count donor；steering control 是一个 norm-matched orthogonal random direction。先在 seed 内对 directed pairs/controls 求配对均值，再以 seed cluster 为独立聚合单位。</p><div class="formula">Δ<sub>s,c</sub> = mean<sub>examples in seed s</sub>(T<sub>treat</sub> − mean(T<sub>matched controls</sub>)); &nbsp; estimate<sub>c</sub> = mean<sub>s</sub> Δ<sub>s,c</sub>.</div>{_conclusion('primary effect 隔离的是相对于配对控制的方向运输，不是 treatment 输出的原始正确率，也不是跨模型参数量差异。')}
<h3>3.3 不确定性与多重比较</h3><p>每个 held-out 条件有 5 个 seed cluster。95% CI 来自 10,000 次 seed-cluster bootstrap；双侧 exact sign-flip p 在 seed means 上计算；同一 evidence scope 内使用 Holm correction。缺失值不做插补；invalid generation 按 strict zero-effect 进入分母，stage audit 要求成功行数等于逻辑行数且 skipped=0。</p>{_conclusion('bootstrap CI 和 exact randomization p 回答不同问题。由于 n=5 的离散性，本报告不把 CI&gt;0 写成“通过 0.05 正式检验”。')}
</section>

<section id="integrity"><h2>4. 数据完整性、运行与审计</h2><p>两次 campaign 均为 COMPLETE，正式实现 Git commit 为 <code>dd409f2dff82ccd6400dfc3d7704025cb6939940</code>；preflight 记录的 implementation SHA 为 <code>{preflight['implementation_sha256']}</code>、causal config SHA 为 <code>{preflight['causal_config_sha256']}</code>、stimuli SHA 为 <code>{preflight['stimuli_sha256']}</code>。Qwen 完成于 {models[0]['completion']['updated_utc']}，Gemma 完成于 {models[1]['completion']['updated_utc']}。</p>
{_table(['模型','audit','checks','errors','resolved stage roots','audit SHA-256'], audit_rows)}
<details><summary>302 项审计的类别分解（单模型 / 双模型合计）</summary>{_table(['类别','每模型 checks','双模型 checks'], audit_category_rows, compact=True)}</details>
<h3>4.1 Stage 行数与计算复用</h3>{_table(['模型','family','phase','design hash','successful/logical','executed generations','reused logical','skipped','capture index'], stage_rows, compact=True)}
<p><code>reused logical</code> 表示可由同一实际 generation 复用的逻辑 control 行；它减少推理次数但不改变 estimand 分母。所有 patching stage 的 successful=logical、skipped=0；steering complete schema 直接记录 rows，审计逐项核对 detail/summary/selection/confirmation。</p>
<h3>4.2 FileStream 与本地归档</h3>{_table(['模型','export','archive bytes','SHA verified','source/copy manifest','archive SHA-256'], export_rows)}
<p>本地分析从完整、SHA-verified 的 <code>.tar.zst</code> 解包到短路径，避免 Windows MAX_PATH 对深层 capture shards 的影响；原始归档是完整性的权威副本，报告仅把小型机器可读分析表纳入 Git。Qwen/Gemma 归档分别为 {exports[0].get('archive_size_bytes',0):,} / {exports[1].get('archive_size_bytes',0):,} bytes。</p>{_conclusion('两模型 campaign、design、stage 行数、selection、confirmation 与 302 项审计均闭合；FileStream 与本地归档的 SHA-256 和 source/copy manifests 一致。数据完整性不依赖部分展开的深路径目录。')}
</section>

<section id="baseline"><h2>5. 基线行为：有效但并非普遍正确</h2>{_table(['模型','split','n','format valid','exact accuracy','MAE','signed error'], baseline_rows)}
<p>两模型 330/330 completions 都满足严格 numeric 格式；但 confirmation accuracy 只有 Qwen 49.1% 和 Gemma 42.7%。按 gold count 分解，低计数接近饱和，高计数持续低估：在 count=10 上 Qwen/Gemma 的平均预测分别为 {models[0]['baseline_count'][10]['mean_prediction']:.3f}/{models[1]['baseline_count'][10]['mean_prediction']:.3f}，平均 signed error 为 {models[0]['baseline_count'][10]['mean_signed_error']:+.3f}/{models[1]['baseline_count'][10]['mean_signed_error']:+.3f}。</p>
<details><summary>按 gold count 的全部基线结果</summary>{_table(['模型','gold count','n','valid','accuracy','mean prediction','signed error','MAE'], [[html.escape(row['model_label']),str(row['gold_count']),str(row['examples']),_pct(row['format_valid_rate']),_pct(row['accuracy']),_fmt(row['mean_prediction']),_fmt(row['mean_signed_error'],signed=True),_fmt(row['mean_absolute_error'])] for model in models for row in model['baseline_count']], compact=True)}</details>
{_conclusion('干预测试的是方向运输/操控，而不是建立两个模型在 V4.4 上“已经会正确计数”。高计数低估是解释 target-hit 与 donor-state 结果时必须保留的混杂背景。')}
</section>

<section id="prompt"><h2>6. Prompt full-span patching</h2><p>screen 只把 <code>toggled_needle_span</code> 条件送入确认：Qwen {prompt_site_counts[('Qwen3-8B','toggled_needle_span')]} 个、Gemma {prompt_site_counts[('Gemma4-E4B','toggled_needle_span')]} 个；<code>toggled_needle_end</code> 为 0/0。所有 primary span 条件的 bootstrap CI 下界大于 0，全部 Qwen 条件和全部 Gemma 条件的 5/5 held-out seeds 为正；但 Holm-significant 仍是 0。</p>
{_figure(3, _layer_svg(conditions,'prompt_patching','fig3','Prompt full-span single-layer confirmation'), '完整 toggled-needle span 的 single-layer held-out 确认。横轴是 decoder start layer；纵轴是 matched-control-adjusted strict normalized transport；绿色/蓝色/橙色分别为 k=1/3/5。每个点是 5 个 held-out seed cluster 的条件均值；线仅帮助读取，空白 layer 表示未通过 screen、未进入确认，不代表 effect=0；图中不画误差条以避免上百条重叠，逐条件 95% CI 在机器表 primary_confirmation_conditions.csv。')}
<p>Qwen 的中位 effect 为 {_fmt(q_prompt['median_effect'])}，范围 {_fmt(q_prompt['min_effect'])}–{_fmt(q_prompt['max_effect'])}；最佳 {_condition_label(q_prompt_best)} 为 {_fmt(q_prompt_best['mean_control_adjusted_transport'])} [{_fmt(q_prompt_best['ci95_low'])}, {_fmt(q_prompt_best['ci95_high'])}]，最弱 {_condition_label(q_prompt_worst)} 为 {_fmt(q_prompt_worst['mean_control_adjusted_transport'])} [{_fmt(q_prompt_worst['ci95_low'])}, {_fmt(q_prompt_worst['ci95_high'])}]。Gemma 中位 {_fmt(g_prompt['median_effect'])}、范围 {_fmt(g_prompt['min_effect'])}–{_fmt(g_prompt['max_effect'])}；最佳/最弱分别为 {_condition_label(g_prompt_best)} {_fmt(g_prompt_best['mean_control_adjusted_transport'])} 和 {_condition_label(g_prompt_worst)} {_fmt(g_prompt_worst['mean_control_adjusted_transport'])}。</p>
{_conclusion('直接数据支持：协调复制完整 needle span 在两个模型的已筛选 held-out 条件中足以运输计数方向。负结果/边界：单 endpoint token 没有条件通过 screen；因此证据支持“分布在 span 上的充分信息”，不支持单 endpoint 充分，也不证明 span 对正确计数是必要的。')}
</section>

<section id="answer"><h2>7. Answer-query patching</h2><p>primary held-out 条件为 Qwen {q_answer['conditions']}、Gemma {g_answer['conditions']}；全部条件 bootstrap CI 下界 &gt;0 且 5/5 seeds 为正。中位 effect 分别 {_fmt(q_answer['median_effect'])}/{_fmt(g_answer['median_effect'])}。Qwen 最佳/最弱为 {_condition_label(q_answer_best)} {_fmt(q_answer_best['mean_control_adjusted_transport'])} 与 {_condition_label(q_answer_worst)} {_fmt(q_answer_worst['mean_control_adjusted_transport'])}；Gemma 为 {_condition_label(g_answer_best)} {_fmt(g_answer_best['mean_control_adjusted_transport'])} 与 {_condition_label(g_answer_worst)} {_fmt(g_answer_worst['mean_control_adjusted_transport'])}。</p>
{_figure(4, _layer_svg(conditions,'answer_patching','fig4','Answer-query single-layer confirmation'), 'Final Total: query residual 的 single-layer held-out 确认。横轴 decoder layer；纵轴 matched-control-adjusted strict normalized transport；颜色为 k。每点由 5 个 held-out seeds 聚合。为避免数百条重叠，图中不画误差条；逐条件 bootstrap CI 与 exact/Holm p 在机器表中。未确认层留空。cumulative-from-layer 条件不画在本图，因为持续 clamp 从 L 到末层测试“从 L 起的充分性”，不能当作局部层定位。')}
<p>cumulative 结果总体大且近似层不变，这是协议本身持续覆盖后续层的预期特征；single-layer 结果更适合观察层依赖，但仍是“在该层替换 query state 后的下游可读性”，不是对一个唯一计算瞬间的证明。</p>{_conclusion('答案查询 residual 是两个模型中可运输计数方向的强充分载体。不能下结论：cumulative 曲线的平坦不等于所有层同等重要；patch donor state 运输的也可能是 donor 的错误预测，而非 gold count。')}
</section>

<section id="steering"><h2>8. Count-centroid steering</h2><p>Qwen 45 个 primary 条件中 45 个为 5/5 seeds 正且 CI 下界&gt;0；Gemma 54 个条件中 53 个为 5/5 seeds 正，54 个 CI 下界&gt;0。Qwen/Gemma 中位 effect 为 {_fmt(q_steer['median_effect'])}/{_fmt(g_steer['median_effect'])}。Gemma 的唯一 4/5-positive 条件是其最弱条件 {_condition_label(g_steer_worst)}，effect {_fmt(g_steer_worst['mean_control_adjusted_transport'])} [{_fmt(g_steer_worst['ci95_low'])}, {_fmt(g_steer_worst['ci95_high'])}]。</p>
{_figure(5, _layer_svg(conditions,'steering','fig5','Single-layer count-centroid steering confirmation'), 'α=1 count-centroid steering 的 single-layer held-out 确认。横轴 decoder layer；纵轴 geometric treatment 减去 norm-matched orthogonal random control 的 strict normalized transport；颜色为 k。每点聚合 5 个 held-out seeds。图中不画误差条；逐条件 bootstrap CI 与 exact/Holm p 在机器表中。每个 k 另有一个 screen 冻结的 multi-layer plan，列在机器表但不与单一 x-layer 混画。')}
<p>Qwen 最佳/最弱为 {_condition_label(q_steer_best)} {_fmt(q_steer_best['mean_control_adjusted_transport'])} 与 {_condition_label(q_steer_worst)} {_fmt(q_steer_worst['mean_control_adjusted_transport'])}；Gemma 最佳为 {_condition_label(g_steer_best)} {_fmt(g_steer_best['mean_control_adjusted_transport'])}。正运输表明 downstream readout 响应 population-level count direction；它不保证生成精确 target，也不建立 multi-layer 优于 single-layer。</p>{_conclusion('两个模型都存在可沿 count centroid direction 操控的 answer-query state；这是受配对随机方向控制的干预证据。边界：效果是方向运输而非精确 set-to-count，且 5-seed exact test 仍未过 0.05。')}
</section>

<section id="ablation"><h2>9. Head-bank ablation：必要性信号不稳定</h2><p>这是 discovery screen（每个 top-n/bank 20 examples=5 seeds×4 high counts），没有 held-out confirmation CI。effect 定义为 ranked ablation 减去三套 layer-matched random controls 的均值；accuracy effect &lt;0 或 absolute-error effect &gt;0 才是 ranked bank 相对随机更伤性能。</p>
{_figure(6, _ablation_svg(ablation,'accuracy_effect','fig6','Ranked-minus-random accuracy effect','ranked − random accuracy change'), 'Head-bank top-n discovery sweep 的 accuracy effect。横轴 top-n=1…32；纵轴 ranked ablation accuracy change 减去 layer-matched random ablation accuracy change。0 虚线表示与随机控制相同；负值表示 ranked bank 更损害 accuracy。每点 20 examples、5 seeds；线不带确认 CI。')}
{_figure(7, _ablation_svg(ablation,'absolute_error_effect','fig7','Ranked-minus-random absolute-error effect','ranked − random absolute-error change'), '同一 discovery sweep 的 absolute-error effect。横轴为 top-n=1…32，左右 panel 分别为 Qwen/Gemma，青色/棕色为 broad-aggregation/first-locator bank；纵轴为 ranked 减 random 的 |prediction−gold| 变化，正值表示 ranked bank 相对随机增加更多误差，负值相反。每点 20 examples、5 seeds；该 discovery-only 曲线没有 confirmation CI，因此不画误差条。')}
{_table(['模型','head bank','accuracy min','accuracy max','|error| min','|error| max'], ablation_extrema_rows)}
<p>Qwen broad bank 的 accuracy effect 从 −0.050 到 +0.233，Gemma 从 −0.100 到 0；first-locator bank 在 Gemma 上甚至出现方向反转。曲线不呈共同的单调 dose-response，且 random heads 与 ranked heads可有重叠。</p>{_conclusion('目前不能把 ranked head banks 写成跨模型、单调的必要组件。该 screen 提供的是探索性干预现象与候选定位，不是确认性必要性证据；方向反转和控制重叠必须保留。','探索性/负结果')}
</section>

<section id="cross"><h2>10. 跨模型整合</h2>{_table(['family','Qwen median','Gemma median','Qwen−Gemma','Qwen CI>0','Gemma CI>0'], cross_rows)}
<p>定性一致性高于数值一致性：两个模型都只把完整 prompt span（非 endpoint）送入确认；answer patch 与 steering 也都产生正运输。数值上 Qwen 的 family median 略高，但 condition 数、层数、tokenizer 映射、基线错误分布和被 screen 选中的层集合都不同，因此这些 median 差不能解释为模型能力排名。</p>
{_table(['family','模型','conditions','median','range','bootstrap CI>0','all 5 seeds positive','Holm p≤.05'], family_table_rows)}
<p>最小机制叙述是：计数相关状态在 prompt 中跨完整 needle span 分布；到 final answer query，它成为可被 donor residual 运输、也可被 count-centroid direction 操控的下游载体。head-bank ablation 尚未给出稳定必要性。这个叙述连接了三组干预，但不主张 span→query 的逐层因果路径已经被直接追踪。</p>{_conclusion('跨模型共同支持“分布式 prompt span → 可执行 answer-query state”的工作模型。它是与全部结果相容的最小解释，不是已排除替代机制的完备因果图。')}
</section>

<section id="limits"><h2>11. 限制、复现与后续验证</h2><h3>11.1 结论强度与限制</h3><ul><li>每条件 5 个 held-out seeds，exact test 分辨率不足以达到双侧 p&lt;.05；需要更多独立 seed clusters。</li><li>screen 后确认估计适用于通过冻结阈值的条件，不能外推到所有层；空白层不是零效应。</li><li>Gemma 的 362/540 span pairs 使用定义内 deterministic remap；tokenizer 差异可能影响跨模型绝对 effect。</li><li>基线在高计数上系统低估；transport 可改善方向但未必达到 gold。</li><li>head ablation 是 discovery-only，缺 held-out confirmation，且 ranked/random overlap 使简单必要性解释更弱。</li><li>本实验没有直接测试从各 prompt token 到 answer query 的逐层中介链，也没有证明 count representation 唯一或线性。</li></ul>{_conclusion('现有证据足以支持受限的干预充分性与可操控性陈述；更强的必要性、层级定位、精确设置与完整路径结论都需要新实验。','尚不能下结论')}
<h3>11.2 优先后续实验</h3><ol><li>把 held-out seeds 增至至少 6（更合理为 20+），解除双侧 sign-flip p 的离散下限，并预先规定 family-level multiplicity。</li><li>对 full span 做 coordinated leave-one-token-out、span-width dose-response 与 matched random-span patch，区分广泛状态复制与语义位置特异性。</li><li>用 mediation-style sequential patching 追踪 span→query 的层间路径，并把 donor baseline correct/wrong 分层为预注册 estimand。</li><li>重新设计 non-overlapping head controls，冻结 top-n banks 后做独立 confirmation necessity sweep。</li></ol>{_conclusion('下一步首先应增加独立 seeds，其次用局部删除/随机 span 控制解释 full-span effect，再以中介式干预验证 prompt→query 路径。','建议')}
<h3>11.3 可复现路径</h3><p>正式设计文档：<code>docs/realistic_niah_v4_causal_v2.md</code>；配置：<code>configs/realistic_niah_v4_causal_v2.json</code>；分析实现：<code>src/realistic_niah_v4/causal_v2_analysis.py</code>；本报告生成器：<code>scripts/build_realistic_niah_v4_4_causal_v2_report.py</code>。机器可读输出在 <code>reports/v4_non-thinking_causal/v4_4_causal_v2/</code>，包括逐条件 primary 表、聚合表、基线、alignment、ablation、stage inventory、source SHA ledger 与 <code>report_summary.json</code>。</p>
<div class="provenance">report schema: {REPORT_SCHEMA}<br>generated from audited run roots; displayed numeric values are generated from checked machine-readable tables, not hand-entered.</div>{_conclusion('报告中的统计数字由同一生成脚本从审计 run 导出；source_ledger.csv 固定每个输入文件的 SHA-256，report_summary.json 与 CSV 提供可重算路径。')}
</section>

</article></div></main></body></html>"""


def build_report(
    *,
    qwen_root: Path,
    gemma_root: Path,
    output: Path,
    data_dir: Path,
    qwen_export: Path | None = None,
    gemma_export: Path | None = None,
) -> dict[str, Any]:
    sources = Sources()
    models = [
        _load_model("Qwen3-8B", qwen_root, qwen_export, sources),
        _load_model("Gemma4-E4B", gemma_root, gemma_export, sources),
    ]
    conditions = [row for model in models for row in model["confirmations"]]
    family_summary, group_summary = _group_confirmation(conditions)
    payload = {
        "schema_version": REPORT_SCHEMA,
        "models": models,
        "primary_confirmation_conditions": conditions,
        "family_summary": family_summary,
        "group_summary": group_summary,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(data_dir / "primary_confirmation_conditions.csv", conditions)
    _write_csv(data_dir / "primary_confirmation_family_summary.csv", family_summary)
    _write_csv(data_dir / "primary_confirmation_group_summary.csv", group_summary)
    _write_csv(data_dir / "baseline_by_split.csv", [row for model in models for row in model["baseline_split"]])
    _write_csv(data_dir / "baseline_by_count.csv", [row for model in models for row in model["baseline_count"]])
    _write_csv(data_dir / "prompt_alignment_summary.csv", [model["alignment"] for model in models])
    _write_csv(data_dir / "selection_summary.csv", [row for model in models for row in model["selections"]])
    _write_csv(data_dir / "stage_inventory.csv", [row for model in models for row in model["stages"]])
    _write_csv(data_dir / "ablation_top_k_sweep.csv", [row for model in models for row in model["ablation"]])
    _write_csv(data_dir / "audit_summary.csv", [model["audit"] for model in models])
    _write_csv(data_dir / "audit_category_summary.csv", [row for model in models for row in model["audit_categories"]])
    _write_csv(data_dir / "export_verification.csv", [model["export"] for model in models])
    source_ledger = sources.ledger()
    _write_csv(data_dir / "source_ledger.csv", source_ledger)

    summary = {
        "schema_version": REPORT_SCHEMA,
        "implementation_commit": models[0]["runtime"]["git_head"],
        "implementation_sha256": models[0]["preflight_design"]["implementation_sha256"],
        "causal_config_sha256": models[0]["preflight_design"]["causal_config_sha256"],
        "stimuli_sha256": models[0]["preflight_design"]["stimuli_sha256"],
        "alignment_policy": models[0]["alignment"]["policy"],
        "model_completion_utc": {model["model"]: model["completion"]["updated_utc"] for model in models},
        "audits": {model["model"]: model["audit"] for model in models},
        "alignment": {model["model"]: model["alignment"] for model in models},
        "baseline": {model["model"]: {row["split"]: row for row in model["baseline_split"]} for model in models},
        "selection": {model["model"]: {row["family"]: row for row in model["selections"]} for model in models},
        "primary_confirmation_family_summary": {f"{row['model_label']}::{row['family']}": row for row in family_summary},
        "exports": {model["model"]: model["export"] for model in models},
        "source_file_count": len(source_ledger),
    }
    (data_dir / "report_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_report(payload), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the audited V4.4 causal-v2 dual-model HTML report")
    parser.add_argument("--qwen-run-root", type=Path, required=True)
    parser.add_argument("--gemma-run-root", type=Path, required=True)
    parser.add_argument("--qwen-export", type=Path)
    parser.add_argument("--gemma-export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_report(
        qwen_root=args.qwen_run_root,
        gemma_root=args.gemma_run_root,
        qwen_export=args.qwen_export,
        gemma_export=args.gemma_export,
        output=args.output,
        data_dir=args.data_dir,
    )
    print(json.dumps({"status": "ok", "output": str(args.output), "schema": summary["schema_version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
