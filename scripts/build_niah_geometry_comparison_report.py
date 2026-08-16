#!/usr/bin/env python3
"""Build a self-contained three-column NIAH geometry comparison report.

The three displayed populations are deliberately distinct:

1. non-thinking prompt needle endpoints on the full registered seed panel;
2. native-thinking response item endpoints after one-to-one trace filtering;
3. native-thinking response item endpoints on the full registered seed panel,
   retaining every parser-observed ordinal from partial traces.

Position (1--10) is the geometry class.  Final-answer correctness is carried as
an independent trajectory-level display attribute and never changes the
position label or the primary aligned cohort.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    ModeDataset,
    load_native_thinking_capture,
    load_non_thinking_capture,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
PCA_DIMS = (32,)
EXPECTED_FULL_PANEL = {
    "discovery": list(range(1234, 1254)),
    "confirmation": list(range(1254, 1264)),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"empty CSV: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value))


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def pct(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def html_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _best(rows: list[dict[str, str]], field: str) -> dict[str, str]:
    return max(rows, key=lambda row: float(row[field]))


def _support_range(audit: Mapping[str, Any], mode: str) -> tuple[int, int]:
    values = [
        int(value)
        for value in audit["position_support"][mode]["confirmation"].values()
    ]
    return min(values), max(values)


def load_metric_comparison(
    aligned_root: Path, one_to_one_root: Path
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[str, int],
    list[Path],
]:
    """Read audited metrics for the three report columns."""

    comparison: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    aligned_peak_layer: dict[str, int] = {}
    inputs: list[Path] = []
    for model in MODELS:
        for pca_dim in PCA_DIMS:
            aligned_dir = aligned_root / model / f"pca{pca_dim}"
            complete_dir = one_to_one_root / model / f"pca{pca_dim}"
            aligned_audit_path = aligned_dir / "cross_mode_geometry_audit.json"
            complete_audit_path = complete_dir / "cross_mode_geometry_audit.json"
            aligned_global_path = aligned_dir / "global_covariance_geometry.csv"
            complete_global_path = complete_dir / "global_covariance_geometry.csv"
            aligned_audit = read_json(aligned_audit_path)
            complete_audit = read_json(complete_audit_path)
            aligned_rows = read_csv(aligned_global_path)
            complete_rows = read_csv(complete_global_path)

            require(aligned_audit["model_label"] == model, "aligned model mismatch")
            require(complete_audit["model_label"] == model, "one-to-one model mismatch")
            require(aligned_audit["native_cohort"] == "parser_hit", "aligned cohort mismatch")
            require(complete_audit["native_cohort"] == "one_to_one", "one-to-one cohort mismatch")
            require(
                aligned_audit["analysis_design"]
                == "fixed_registered_seed_panel_observed_positions",
                "aligned design mismatch",
            )
            require(
                complete_audit["analysis_design"]
                == "complete_trajectory_paired_sensitivity",
                "one-to-one design mismatch",
            )
            require(
                aligned_audit["registered_seed_panel"] == EXPECTED_FULL_PANEL,
                "aligned seed panel is not the registered 30-seed panel",
            )
            for audit in (aligned_audit, complete_audit):
                require(audit["evaluation_split"] == "confirmation only", "split leakage")
                require(audit["preprocessing_fit_split"] == "discovery only", "PCA leakage")
                require(audit["probe_fit_split"] == "discovery only", "probe leakage")
                require(audit["cluster_labels"] == list(range(1, 11)), "label mismatch")

            non_rows = [row for row in aligned_rows if row["mode"] == "non_thinking"]
            aligned_native_rows = [
                row for row in aligned_rows if row["mode"] == "native_thinking"
            ]
            complete_native_rows = [
                row for row in complete_rows if row["mode"] == "native_thinking"
            ]
            require(non_rows and aligned_native_rows and complete_native_rows, "missing mode rows")

            def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
                logistic = _best(rows, "logistic_balanced_accuracy")
                ncc = _best(rows, "ncc_balanced_accuracy")
                return {
                    "n_confirmation": int(logistic["n_confirmation"]),
                    "logistic": float(logistic["logistic_balanced_accuracy"]),
                    "logistic_layer": int(logistic["layer"]),
                    "ncc": float(ncc["ncc_balanced_accuracy"]),
                    "ncc_layer": int(ncc["layer"]),
                }

            comparison[model][pca_dim] = {
                "non_thinking": summarize(non_rows),
                "native_one_to_one": summarize(complete_native_rows),
                "native_aligned": summarize(aligned_native_rows),
                "non_support": _support_range(aligned_audit, "non_thinking"),
                "one_support": _support_range(complete_audit, "native_thinking"),
                "aligned_support": _support_range(aligned_audit, "native_thinking"),
                "one_seed_panel": complete_audit["registered_seed_panel"],
            }
            if pca_dim == 32:
                aligned_peak_layer[model] = comparison[model][pca_dim][
                    "native_aligned"
                ]["logistic_layer"]
            inputs.extend(
                [
                    aligned_audit_path,
                    aligned_global_path,
                    complete_audit_path,
                    complete_global_path,
                ]
            )
    return comparison, aligned_peak_layer, inputs


def non_thinking_outcomes(
    export_root: Path, model: str
) -> tuple[dict[tuple[str, int], dict[str, Any]], Path]:
    candidates = sorted(
        (
            export_root
            / model
            / "numeric"
            / "representation"
            / "analysis"
            / "outcomes"
        ).glob("shared_pca_span_end_layer_*_labeled.csv")
    )
    require(candidates, f"no non-thinking outcome table for {model}")
    path = candidates[0]
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(path):
        if row["design_variant"] != "v4.4":
            continue
        key = (row["split"], int(row["seed"]))
        value = {
            "exact_count": truth(row["is_correct"]),
            "parsed_count": int(row["parsed_count"]) if row["parsed_count"] else None,
            "count_error": int(row["count_error"]) if row["count_error"] else None,
        }
        if key in mapping:
            require(mapping[key] == value, f"inconsistent non-thinking outcome {model}/{key}")
        else:
            mapping[key] = value
    require(len(mapping) == 30, f"expected 30 non-thinking N10 outcomes for {model}")
    return mapping, path


def native_outcomes(
    capture_index: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(capture_index)
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if int(row.get("gold_count", -1)) != 10:
            continue
        key = (str(row["split"]), int(row["seed"]))
        mapping[key] = {
            "exact_count": bool(row.get("exact_count")),
            "parsed_count": row.get("parsed_count"),
            "count_error": (
                int(row["parsed_count"]) - 10
                if row.get("parsed_count") is not None
                else None
            ),
        }
    require(len(mapping) == 30, f"expected 30 native N10 outcomes in {capture_index}")
    return mapping, rows


def partial_trace_rows(
    capture_index: Path, index_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in index_rows:
        if int(row.get("gold_count", -1)) != 10 or bool(row.get("trace_one_to_one")):
            continue
        manifest_path = capture_index.parent / str(row["manifest_path"])
        manifest = read_json(manifest_path)
        item_sites = [
            site for site in manifest["site_rows"] if site.get("site_kind") == "item_end"
        ]
        parser = manifest["parser"]
        result.append(
            {
                "model": row["model_label"],
                "split": row["split"],
                "seed": int(row["seed"]),
                "observed": len(item_sites),
                "occurrences": [int(site["occurrence"]) for site in item_sites],
                "cities": [str(site.get("city")) for site in item_sites],
                "missing": [str(city) for city in parser.get("missing_gold_cities", [])],
                "parsed_count": row.get("parsed_count"),
                "exact_count": bool(row.get("exact_count")),
                "trace_category": row.get("trace_category"),
            }
        )
    return sorted(result, key=lambda row: (row["split"], row["seed"]))


def display_layers(dataset: ModeDataset, aligned_peak: int) -> list[int]:
    available = sorted(dataset.states_by_layer)
    last = max(available)
    landmarks = {
        0,
        round(last * 0.25),
        round(last * 0.50),
        round(last * 0.75),
        last,
        int(aligned_peak),
    }
    return sorted(layer for layer in landmarks if layer in dataset.states_by_layer)


def fit_display_coordinates(
    dataset: ModeDataset,
    layers: Iterable[int],
    outcomes: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = dataset.metadata.reset_index(drop=True)
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    require(discovery.sum() >= 3, f"{dataset.mode}: too few discovery rows")
    result: dict[str, Any] = {}
    for layer in layers:
        states = np.asarray(dataset.states_by_layer[int(layer)], dtype=np.float32)
        scaler = StandardScaler().fit(states[discovery])
        scaled_discovery = scaler.transform(states[discovery])
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(
            scaled_discovery
        )
        coordinates = pca.transform(scaler.transform(states))
        points = []
        for index, row in metadata.iterrows():
            split = str(row["split"])
            seed = int(row["seed"])
            outcome = outcomes[(split, seed)]
            points.append(
                [
                    split,
                    seed,
                    int(row["occurrence"]),
                    1 if outcome["exact_count"] else 0,
                    outcome.get("parsed_count"),
                    round(float(coordinates[index, 0]), 5),
                    round(float(coordinates[index, 1]), 5),
                    round(float(coordinates[index, 2]), 5),
                ]
            )
        result[str(layer)] = {
            "evr": [round(float(value), 6) for value in pca.explained_variance_ratio_],
            "points": points,
        }
    return result


def build_visual_data(
    export_root: Path,
    native_capture_root: Path,
    aligned_peak_layer: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    visual: dict[str, Any] = {}
    partials: list[dict[str, Any]] = []
    inputs: list[Path] = []
    for model in MODELS:
        non_index = (
            export_root
            / model
            / "numeric"
            / "representation"
            / "capture"
            / "capture_index.jsonl"
        )
        native_index = native_capture_root / model / "capture_index.jsonl"
        non_outcome, outcome_path = non_thinking_outcomes(export_root, model)
        native_outcome, native_rows = native_outcomes(native_index)
        partials.extend(partial_trace_rows(native_index, native_rows))

        non = load_non_thinking_capture(non_index, design_variant="v4.4", pooling="span_end")
        aligned = load_native_thinking_capture(
            native_index, site_kind="item_end", cohort="parser_hit"
        )
        one = load_native_thinking_capture(
            native_index, site_kind="item_end", cohort="one_to_one"
        )
        common_layers = sorted(
            set(non.states_by_layer) & set(aligned.states_by_layer) & set(one.states_by_layer)
        )
        probe = ModeDataset(
            mode="common",
            model_label=model,
            metadata=non.metadata,
            states_by_layer={layer: non.states_by_layer[layer] for layer in common_layers},
        )
        layers = display_layers(probe, aligned_peak_layer[model])
        visual[model] = {
            "layers": layers,
            "default_layer": int(aligned_peak_layer[model]),
            "panels": {
                "non_thinking": fit_display_coordinates(non, layers, non_outcome),
                "native_one_to_one": fit_display_coordinates(one, layers, native_outcome),
                "native_aligned": fit_display_coordinates(aligned, layers, native_outcome),
            },
        }
        inputs.extend([non_index, native_index, outcome_path])
        for provenance_path in (
            non_index.parent / "capture_manifest.json",
            native_index.parent / "export_audit.json",
        ):
            if provenance_path.is_file():
                inputs.append(provenance_path)
        del non, aligned, one, probe
        gc.collect()
    return visual, partials, inputs


def metric_table(comparison: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> str:
    rows = []
    for model in MODELS:
        for pca_dim in PCA_DIMS:
            item = comparison[model][pca_dim]
            non = item["non_thinking"]
            one = item["native_one_to_one"]
            aligned = item["native_aligned"]
            one_panel = item["one_seed_panel"]

            def cell(value: Mapping[str, Any], support: tuple[int, int], seeds: str) -> str:
                support_text = (
                    str(support[0])
                    if support[0] == support[1]
                    else f"{support[0]}–{support[1]}"
                )
                return (
                    f"<strong>logistic {pct(value['logistic'])} @ L{value['logistic_layer']}</strong>"
                    f"<br>NCC {pct(value['ncc'])} @ L{value['ncc_layer']}"
                    f"<br><span class=\"muted\">D/C seeds {esc(seeds)} · confirmation nₖ {support_text}</span>"
                )

            one_seeds = (
                f"{len(one_panel['discovery'])}/{len(one_panel['confirmation'])}"
            )
            rows.append(
                (
                    esc(model),
                    str(pca_dim),
                    cell(non, item["non_support"], "20/10"),
                    cell(one, item["one_support"], one_seeds),
                    cell(aligned, item["aligned_support"], "20/10"),
                )
            )
    return html_table(
        [
            "模型",
            "PCA",
            "Non-thinking · full panel",
            "Native · one-to-one",
            "Native · ordinal-aligned full panel",
        ],
        rows,
    )


def partial_table(partials: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for row in partials:
        rows.append(
            (
                esc(row["model"]),
                esc(row["split"]),
                str(row["seed"]),
                str(row["observed"]),
                esc(", ".join(map(str, row["occurrences"]))),
                esc(", ".join(row["cities"])),
                esc(str(row["parsed_count"])),
                "correct" if row["exact_count"] else "wrong",
                esc(row["trace_category"]),
            )
        )
    return html_table(
        [
            "模型",
            "split",
            "seed",
            "observed items",
            "ordinal labels",
            "parser-observed cities",
            "final count",
            "final outcome",
            "trace category",
        ],
        rows,
    )


def model_section(model: str, payload: Mapping[str, Any]) -> str:
    slug = "qwen" if model.startswith("Qwen") else "gemma"
    options = "".join(
        f'<option value="{layer}"{(" selected" if layer == payload["default_layer"] else "")}>L{layer}</option>'
        for layer in payload["layers"]
    )
    cards = []
    definitions = (
        (
            "non_thinking",
            "1 · Non-thinking",
            "Prompt 中第 k 个真实 needle 的 span-end；共享 30 seeds。",
        ),
        (
            "native_one_to_one",
            "2 · Native-thinking · one-to-one",
            "parser-observed city multiset 与 gold 严格相等；不按最终答案正确性筛选。",
        ),
        (
            "native_aligned",
            "3 · Native-thinking · ordinal-aligned",
            "共享 30 seeds；实际写出的第 k 项就是位置 k，允许后段缺失。",
        ),
    )
    for key, title, description in definitions:
        cards.append(
            f'<article class="geometry-card"><h3>{esc(title)}</h3>'
            f'<p>{esc(description)}</p>'
            f'<canvas id="{slug}-{key}" data-model="{esc(model)}" data-panel="{key}"></canvas>'
            f'<div class="panel-stats" id="{slug}-{key}-stats"></div></article>'
        )
    return f"""
<section id="{slug}">
  <div class="section-title"><div><div class="eyebrow">MODEL COMPARISON</div><h2>{esc(model)}</h2></div>
  <div class="controls"><label>Layer <select id="{slug}-layer">{options}</select></label>
  <label>Displayed panel <select id="{slug}-split"><option value="confirmation">confirmation only · 10 seeds / nominal 100</option><option value="all">all registered · 30 seeds / nominal 300</option></select></label>
  <label>Final outcome <select id="{slug}-outcome"><option value="all">all</option><option value="correct">correct</option><option value="wrong">wrong</option></select></label></div></div>
  <div class="geometry-grid">{''.join(cards)}</div>
</section>
"""


def build_html(
    comparison: Mapping[str, Mapping[int, Mapping[str, Any]]],
    visual: Mapping[str, Any],
    partials: list[dict[str, Any]],
) -> str:
    visual_json = json.dumps(visual, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    css = """
:root{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#626A74;--line:#C9C2B6;--indigo:#23165C;--violet:#6750E8;--teal:#00A88F;--yellow:#D6B52C}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}nav{position:sticky;top:0;z-index:5;display:flex;gap:18px;padding:10px 22px;background:rgba(243,238,228,.96);border-bottom:1px solid var(--line)}nav a{color:var(--indigo);font-size:13px;font-weight:750;text-decoration:none}main{max-width:1480px;margin:auto;padding:38px 28px 80px}header{max-width:1080px;border-bottom:2px solid var(--ink);padding-bottom:28px}.eyebrow{font:700 12px/1.2 Consolas,monospace;letter-spacing:.12em;color:var(--teal)}h1{font-size:44px;line-height:1.08;margin:10px 0 16px;letter-spacing:-.035em}h2{font-size:29px;margin:0}.lead{font-size:18px;color:#404852;max-width:92ch}section{padding:46px 0;border-bottom:1px solid var(--line)}.callout{max-width:1080px;background:var(--surface);border-left:4px solid var(--teal);padding:15px 19px;margin:20px 0}.warning{border-left-color:var(--yellow)}.definitions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}.definitions>div,.geometry-card{background:var(--surface);border:1px solid var(--line);padding:17px}.definitions h3,.geometry-card h3{color:var(--indigo);margin:0 0 8px;font-size:17px}.definitions p,.geometry-card p{font-size:13px;color:var(--muted);margin:0 0 12px}.section-title{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:18px}.controls{display:flex;gap:12px;flex-wrap:wrap}.controls label{font-size:12px;font-weight:700;color:var(--muted)}select{display:block;margin-top:4px;border:1px solid var(--line);background:var(--surface);padding:7px 28px 7px 9px;color:var(--ink)}.geometry-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.geometry-card canvas{display:block;width:100%;height:360px;background:#F8F4EC;border:1px solid #DDD5C9}.panel-stats{min-height:46px;margin-top:9px;color:var(--muted);font:12px/1.5 Consolas,monospace}.table-scroll{overflow:auto;background:var(--surface);border:1px solid var(--line);margin:16px 0 22px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #DED8CE}th{position:sticky;top:0;background:#ECE6DA;color:#303744}.muted{color:var(--muted);font-size:11px}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0;font-size:13px}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px;background:var(--violet)}.dot.correct{border:3px solid white;box-shadow:0 0 0 1px #49515B}.dot.wrong{border:2px solid #20242D}.small{font-size:13px;color:var(--muted);max-width:110ch}details{background:var(--surface);border:1px solid var(--line);margin:18px 0}summary{cursor:pointer;padding:12px 15px;font-weight:750;color:var(--indigo)}details .table-scroll{border:0;border-top:1px solid var(--line);margin:0}.provenance{font:11px/1.6 Consolas,monospace;color:var(--muted)}
@media(max-width:1050px){.geometry-grid,.definitions{grid-template-columns:1fr}.geometry-card canvas{height:390px}.section-title{align-items:flex-start;flex-direction:column}}@media(max-width:650px){main{padding:25px 13px 60px}h1{font-size:34px}.geometry-card canvas{height:330px}}
"""
    script = """
const DATA=__VISUAL_DATA__;
const COLORS=['#6750E8','#00A9D8','#00A88F','#2DBE77','#A7C957','#D6B52C','#F29E4C','#E76F51','#D94B86','#8E5DB7'];
function filtered(model,panel,layer,split,outcome){
  const block=DATA[model].panels[panel][String(layer)]; if(!block)return {points:[],evr:[]};
  return {evr:block.evr,points:block.points.filter(p=>(split==='all'||p[0]===split)&&(outcome==='all'||(outcome==='correct'?p[3]===1:p[3]===0)))};
}
function draw(canvas,model,panel,layer,split,outcome){
  const {points,evr}=filtered(model,panel,layer,split,outcome),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));
  const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);
  const pad={l:42,r:17,t:18,b:34};if(!points.length){c.fillStyle='#6A727D';c.font='14px Segoe UI';c.fillText('No states match this filter.',20,30);return;}
  const xs=points.map(p=>p[5]),ys=points.map(p=>p[6]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.09;xmax+=dx*.09;ymin-=dy*.09;ymax+=dy*.09;
  const sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  c.strokeStyle='#D9D2C7';c.lineWidth=1;if(xmin<=0&&xmax>=0){c.beginPath();c.moveTo(sx(0),pad.t);c.lineTo(sx(0),h-pad.b);c.stroke()}if(ymin<=0&&ymax>=0){c.beginPath();c.moveTo(pad.l,sy(0));c.lineTo(w-pad.r,sy(0));c.stroke()}
  const groups=new Map();for(const p of points){if(!groups.has(p[2]))groups.set(p[2],[]);groups.get(p[2]).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>[k,ps.reduce((s,p)=>s+p[5],0)/ps.length,ps.reduce((s,p)=>s+p[6],0)/ps.length,ps.length]);
  c.strokeStyle='#2C3440';c.lineWidth=2;c.beginPath();cent.forEach((p,i)=>i?c.lineTo(sx(p[1]),sy(p[2])):c.moveTo(sx(p[1]),sy(p[2])));c.stroke();
  for(const p of points){c.globalAlpha=.66;c.fillStyle=COLORS[p[2]-1];c.strokeStyle=p[3]===1?'#FFFDF8':'#20242D';c.lineWidth=p[3]===1?2.4:1.15;c.beginPath();c.arc(sx(p[5]),sy(p[6]),3.3,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const p of cent){c.fillStyle=COLORS[p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.3;c.beginPath();c.arc(sx(p[1]),sy(p[2]),6,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(p[0]),sx(p[1])+7,sy(p[2])-7)}
  c.fillStyle='#4F5863';c.font='11px Consolas';c.fillText('PC1',w-38,h-10);c.save();c.translate(13,41);c.rotate(-Math.PI/2);c.fillText('PC2',0,0);c.restore();
  const seeds=new Set(points.map(p=>p[0]+':'+p[1])).size,counts=cent.map(p=>p[3]),nominal=split==='confirmation'?100:(split==='all'?300:seeds*10);const stat=document.getElementById(canvas.id+'-stats');
  stat.textContent=`L${layer} · nominal ${nominal} · actual ${points.length} states · ${seeds} seeds · nₖ ${Math.min(...counts)}–${Math.max(...counts)} · EVR(PC1–3) ${(100*evr.reduce((a,b)=>a+b,0)).toFixed(1)}%`;
}
function redraw(model){const slug=model.startsWith('Qwen')?'qwen':'gemma',layer=+document.getElementById(slug+'-layer').value,split=document.getElementById(slug+'-split').value,outcome=document.getElementById(slug+'-outcome').value;for(const panel of ['non_thinking','native_one_to_one','native_aligned'])draw(document.getElementById(slug+'-'+panel),model,panel,layer,split,outcome)}
for(const model of Object.keys(DATA)){const slug=model.startsWith('Qwen')?'qwen':'gemma';for(const key of ['layer','split','outcome'])document.getElementById(slug+'-'+key).addEventListener('change',()=>redraw(model));redraw(model)}
let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>Object.keys(DATA).forEach(redraw),100)});
""".replace("__VISUAL_DATA__", visual_json)

    partial_confirmation = [row for row in partials if row["split"] == "confirmation"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Geometry Comparison</title><style>{css}</style></head>
<body><nav><a href="#design">口径</a><a href="#qwen">Qwen</a><a href="#gemma">Gemma</a><a href="#metrics">指标</a><a href="#partial">部分轨迹</a></nav><main>
<header><div class="eyebrow">REALISTIC NIAH · THREE-COHORT GEOMETRY</div><h1>NiaH Geometry Comparison</h1>
<p class="lead">同一份报告并列展示 non-thinking、经过 one-to-one 结构清洗的 native-thinking，以及使用共享 30 seeds、按实际出现 ordinal 对齐的 native-thinking。</p></header>
<section id="design"><h2>比较口径</h2><div class="definitions"><div><h3>1 · Non-thinking</h3><p>固定 V4.4 N=10 prompt；第 k 类是 prompt 中第 k 个真实 needle 的 span-end state。每个 seed 固定有十个位置。</p></div><div><h3>2 · Native · one-to-one</h3><p>第 k 类是 response 中第 k 个 item-end state。要求 parser-observed city multiset 与 gold 严格相等、无重复或遗漏；不筛最终答案正确性。这是 completion-conditioned sensitivity。</p></div><div><h3>3 · Native · ordinal-aligned</h3><p>同一套 30 seeds 上保留所有 parser-hit。模型实际写出的第 k 项标为 k；少写就少观测，不插值、不补齐。</p></div></div>
<div class="legend"><span><i class="dot"></i>填充颜色 = 位置 k</span><span><i class="dot correct"></i>白色粗边 = 最终答对</span><span><i class="dot wrong"></i>深色边 = 最终答错</span></div>
<div class="callout"><strong>标签分离：</strong>位置标签始终是 <code>occurrence=k</code>；<code>final exact_count</code> 只控制点的轮廓，不参与 PCA、probe class 或 aligned cohort 入选。</div>
<div class="callout"><strong>“少数了”的两种含义：</strong>non-thinking 即使最终输出 6 而不是 10，N=10 prompt 中十个真实 needle endpoints 仍全部存在，所以仍贡献十个位置 state；native-thinking 若 response 只实际写出六项，则只有六个 item-end states。二者都保留错误样本，但只有后者会产生 ragged position support。</div>
<div class="callout warning"><strong>站点语义边界：</strong>non-thinking 是 prompt needle endpoint，native-thinking 是 response item endpoint。三列比较的是“运行位置几何是否形成”，不是声称三个站点是同一个 token-level random variable。图可在 confirmation-only（10 seeds，nominal 100）与全注册 panel（30 seeds，nominal 300）之间切换；native 列始终另报实际可观测 state 数。</div></section>
{model_section('Qwen3-8B', visual['Qwen3-8B'])}
{model_section('Gemma4-E4B', visual['Gemma4-E4B'])}
<section id="metrics"><h2>Held-out 定量比较</h2><p class="small">所有 PCA/标准化/probe 只在 discovery 拟合，数值只在 confirmation 评价。表中跨层最大值是描述性 layer scan；one-to-one 与 full-panel 的 seed population 不同，不能把差值直接归因于清洗操作。</p>{metric_table(comparison)}</section>
<section id="partial"><h2>部分轨迹如何进入 aligned 列</h2><p>下面列出 confirmation 中所有非 one-to-one 轨迹。<code>ordinal labels</code> 正是进入第三列的 class；例如只有 <code>1,2</code> 就只贡献两个 state。最终答案可以仍然是 10，这不会虚构第 3–10 个 item-end state。</p><details open><summary>Confirmation partial trajectories · {len(partial_confirmation)} rows</summary>{partial_table(partial_confirmation)}</details>
<p class="small">这里展示的是本地 capture manifest 中可审计的 parser 结果。生成全文没有包含在本地 geometry export；接回 native-thinking filestream 后，可再把原始 trace 片段并入此节。</p></section>
<section><h2>解释优先级</h2><div class="callout"><strong>主结果：</strong>第三列（ordinal-aligned full panel）回答共享 seed panel 上的总体问题。第二列只作为敏感性分析，回答“条件于完整写出十项时，几何怎样”。若两列不同，首先解释为 trajectory-completion selection，而不是几何被“修复”。</div>
<p class="provenance">Report schema: niah_geometry_comparison_v1 · display PCA: discovery-fitted, independently fit per column · quantitative PCA: discovery-fitted 32-dimensional pipeline</p></section>
</main><script>{script}</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-thinking-export-root", type=Path, required=True)
    parser.add_argument("--native-capture-root", type=Path, required=True)
    parser.add_argument("--aligned-geometry-root", type=Path, required=True)
    parser.add_argument("--one-to-one-geometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    comparison, aligned_peak, metric_inputs = load_metric_comparison(
        args.aligned_geometry_root.resolve(), args.one_to_one_geometry_root.resolve()
    )
    visual, partials, visual_inputs = build_visual_data(
        args.non_thinking_export_root.resolve(),
        args.native_capture_root.resolve(),
        aligned_peak,
    )
    document = build_html(comparison, visual, partials)
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    all_inputs = sorted(set(metric_inputs + visual_inputs), key=str)
    manifest = {
        "schema_version": "niah_geometry_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "three_columns": [
            "non_thinking_full_panel",
            "native_thinking_one_to_one",
            "native_thinking_ordinal_aligned_full_panel",
        ],
        "position_label": "ordinal occurrence 1-10",
        "final_correctness_role": "display attribute only; never a geometry class or primary cohort filter",
        "inputs": {str(path): sha256(path) for path in all_inputs},
        "output": str(output),
        "output_sha256": sha256(output),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
