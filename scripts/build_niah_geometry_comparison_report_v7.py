#!/usr/bin/env python3
"""Build the compact all-count NiaH geometry comparison report.

The main text contains only the two registered endpoint comparisons.  Trace
format proportions and trajectory-band diagnostics are appendices; there is no
format-stratified site x layer winner search in the primary estimand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.augment_niah_geometry_comparison_report import (  # noqa: E402
    FULL_LEGACY_MARKERS,
    HYBRID_MARKERS,
    TRACE_CATEGORIES,
    hybrid_marker_summary,
    legacy_compatible_marker_summary,
    marker_definitions_html,
    scatter_svg,
    table,
    trace_category_summary,
)
from scripts.build_niah_geometry_comparison_report import (  # noqa: E402
    MODELS,
    build_dual_visual_data,
    dual_endpoint_section,
    esc,
    expected_trajectory_keys,
    load_dual_endpoint_results,
    read_csv,
    read_json,
    read_jsonl,
    sha256,
)


REPORT_SCHEMA_VERSION = "niah_geometry_comparison_v8_fixed_end_all_counts"


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _selected_by_mode(
    dual_results: Mapping[str, Mapping[str, Any]], model: str, endpoint: str
) -> dict[str, Mapping[str, Any]]:
    rows = dual_results[model][f"{endpoint}_selected"]
    selected = {str(row["mode"]): row for row in rows}
    expected = {"non_thinking", "native_thinking"}
    if set(selected) != expected:
        raise ValueError(
            f"Expected exactly {sorted(expected)} for {model}/{endpoint}; "
            f"got {sorted(selected)}"
        )
    return selected


def empirical_claims(
    dual_results: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render claims from the frozen confirmation rows, without hard-coded scores."""

    blocks = []
    for model in MODELS:
        endpoint_lines = []
        for endpoint, label in (
            ("running", "Running index"),
            ("final", "Final count"),
        ):
            rows = _selected_by_mode(dual_results, model, endpoint)
            non = rows["non_thinking"]
            native = rows["native_thinking"]
            non_log = float(non["confirmation_logistic_balanced_accuracy"])
            non_ncc = float(non["confirmation_ncc_balanced_accuracy"])
            non_snr = float(non["confirmation_class_balanced_snr_db"])
            native_log = float(native["confirmation_logistic_balanced_accuracy"])
            native_ncc = float(native["confirmation_ncc_balanced_accuracy"])
            native_snr = float(native["confirmation_class_balanced_snr_db"])
            probe_relation = (
                "两种 probe 都更高"
                if native_log > non_log and native_ncc > non_ncc
                else "两种 probe 并非一致更高"
            )
            snr_relation = "更高" if native_snr > non_snr else "更低"
            interpretation = (
                "因此这里同时支持更可解码和更高的类间/类内比。"
                if native_log > non_log
                and native_ncc > non_ncc
                and native_snr > non_snr
                else "因此最多 claim 更可解码，不能同时 claim 簇更紧。"
                if native_log > non_log and native_ncc > non_ncc
                else "因此不支持 native-thinking 在该 endpoint 上形成一致优势。"
            )
            endpoint_lines.append(
                f"<li><strong>{label}：</strong>native/non-thinking Logistic "
                f"{_pct(native_log)}/{_pct(non_log)}，NCC "
                f"{_pct(native_ncc)}/{_pct(non_ncc)}，SNR "
                f"{native_snr:.2f}/{non_snr:.2f} dB；{probe_relation}，"
                f"但 native SNR {snr_relation}。{interpretation}</li>"
            )
        blocks.append(
            f"<div><h3>{esc(model)}</h3><ul>{''.join(endpoint_lines)}</ul></div>"
        )
    return f"""
<section id="claims"><h2>Confirmation 100 的可支持结论</h2>
<div class="callout"><strong>总括：</strong>native-thinking 的优势首先应表述为 count variable 在各自 discovery-selected representation 中更可解码；“geometry 更紧”只在 probe 与 SNR 同向改善的比较中成立。以下数字全部来自冻结选择后的 N=1…10 confirmation panel，不由 300-view 的视觉效果决定。</div>
<div class="definitions two">{''.join(blocks)}</div>
</section>"""


def _cohort_table(
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    categories: Iterable[str],
) -> str:
    rows = []
    categories = tuple(categories)
    for model in MODELS:
        for split in ("all", "confirmation"):
            payload = summary[model][split]
            total = int(payload["total"])
            values = []
            for category in categories:
                count = int(payload["counts"].get(category, 0))
                values.append(
                    f"{count} <span class=\"muted\">({100*count/total:.1f}%)</span>"
                )
            rows.append((esc(model), esc(split), str(total), *values))
    return table(("模型", "cohort", "trajectories", *categories), rows)


def marker_appendix(parser_rows: list[dict[str, Any]]) -> str:
    for model in MODELS:
        model_rows = [
            row for row in parser_rows if str(row.get("model_label")) == model
        ]
        keys = {
            (str(row["split"]), int(row["seed"]), int(row["gold_count"]))
            for row in model_rows
        }
        if len(model_rows) != 300 or keys != expected_trajectory_keys():
            raise ValueError(
                f"{model} parser audit is not the registered 10 x 30 panel: "
                f"rows={len(model_rows)}, unique_keys={len(keys)}"
            )
    legacy = legacy_compatible_marker_summary(parser_rows)
    hybrid = hybrid_marker_summary(parser_rows)
    categories = trace_category_summary(parser_rows)
    return f"""
<section id="appendix-markers"><h2>Appendix A · Trace marker 与 completeness 构成</h2>
<p>比例单位始终是 trajectory，不是 hidden-state。<code>full</code> 的分母为每模型 300，<code>confirmation</code> 的分母为每模型 100；两者都覆盖 N=1…10。旧五类用于与早期 parser/图例兼容，<code>unresolved</code> 是审计状态，不是第六类 marker。</p>
<h3>Legacy-compatible 五类 marker</h3>
{_cohort_table(legacy, FULL_LEGACY_MARKERS)}
{marker_definitions_html()}
<details><summary>Current hybrid parser marker</summary>
<p class="small"><code>inline_count</code> 合并 Count:k、文字 cardinal/ordinal 等连续显式进度事件；<code>evidence_sequence</code> 是只有 city+score 表面顺序、没有显式 ordinal 的保守兜底。该分类用于混杂诊断，不改变固定的 <code>item_end</code> 主站点，也不参与 layer selector。</p>
{_cohort_table(hybrid, HYBRID_MARKERS)}</details>
<details><summary>Trace completeness category</summary>
{_cohort_table(categories, TRACE_CATEGORIES)}</details>
</section>"""


def _association(audit: Mapping[str, Any], column: str, *, full: bool) -> float:
    key = "full_panel_categorical_associations" if full else "categorical_associations"
    rows = {str(row["column"]): row for row in audit[key]}
    return float(rows[column]["nmi"])


def _band_verdict(audit: Mapping[str, Any]) -> str:
    marker = _association(audit, "marker_kind", full=False)
    seed = _association(audit, "seed", full=False)
    occurrence = _association(audit, "occurrence", full=False)
    purity = float(
        audit["trajectory_band_summary"]["mean_within_trajectory_band_purity"]
    )
    if purity >= 0.8 and max(marker, seed) > occurrence + 0.1:
        return (
            "分层主要随整条 trajectory 的 template/seed offset，而不是随 running k；"
            "它更像叠加在 ordinal manifold 上的 nuisance direction。"
        )
    if occurrence > max(marker, seed) + 0.1:
        return (
            "分层与 running k 的关联高于 marker/seed；不能把它简单归因于 trace template。"
        )
    return (
        "marker、seed 与 occurrence 的关联没有形成单一主导解释；当前只能报告多因素混合。"
    )


def _band_model_block(
    model: str,
    audit: Mapping[str, Any],
    all_points: list[dict[str, str]],
    confirmation_points: list[dict[str, str]],
) -> str:
    scope = audit["scope"]
    raw_full = audit["display_pca3"]["full_panel_two_band"]
    raw_confirmation = audit["display_pca3"]["confirmation_two_band"]
    centered = audit["within_trajectory_centered_pca3"]
    raw_metrics = audit["ordinal_decodability"]["raw"]
    centered_metrics = audit["ordinal_decodability"][
        "within_trajectory_centered_diagnostic"
    ]
    association_rows = []
    for column in ("marker_kind", "seed", "occurrence", "boundary_kind"):
        association_rows.append(
            (
                f"<code>{esc(column)}</code>",
                f"{_association(audit, column, full=True):.3f}",
                f"{_association(audit, column, full=False):.3f}",
            )
        )
    return f"""
<article class="appendix-model"><h3>{esc(model)}</h3>
<div class="callout"><strong>判读：</strong>{esc(_band_verdict(audit))}</div>
<p class="small">分析站点 <code>{esc(audit['site_kind'])}</code> @ L{int(audit['layer'])}；full {int(scope['full_trajectories'])} trajectories / {int(scope['full_states'])} states，confirmation {int(scope['confirmation_trajectories'])} trajectories / {int(scope['confirmation_states'])} states。PCA3 只用来自 discovery 200 trajectories 的 states 拟合；“上/下”名称依赖固定相机方向，K-means 本身不使用 count 标签。</p>
<h4>Full 300-source panel · all N=1…10</h4>
<div class="band-grid">{scatter_svg(all_points, centered=False)}{scatter_svg(all_points, centered=True)}</div>
<p class="small">raw two-band silhouette={float(raw_full['silhouette']):.3f}；cluster sizes upper/lower={int(raw_full['cluster_sizes']['upper'])}/{int(raw_full['cluster_sizes']['lower'])}。</p>
<h4>Confirmation 100-source panel · all N=1…10</h4>
<div class="band-grid">{scatter_svg(confirmation_points, centered=False)}{scatter_svg(confirmation_points, centered=True)}</div>
<p class="small">raw two-band silhouette={float(raw_confirmation['silhouette']):.3f}；cluster sizes upper/lower={int(raw_confirmation['cluster_sizes']['upper'])}/{int(raw_confirmation['cluster_sizes']['lower'])}。</p>
{table(['candidate nuisance','NMI · full','NMI · confirmation'], association_rows)}
<div class="definitions two"><div><h3>Trajectory centering diagnostic</h3><p>在原 hidden space 内逐 trajectory 减去自身 state mean 后，raw/centered band NMI={float(centered['raw_vs_centered_band_nmi']):.3f}。这个操作使用整条 trajectory，只是定位 nuisance offset，不能作为在线 estimator 或因果干预。</p></div><div><h3>Ordinal signal 是否保留</h3><p>confirmation Logistic {_pct(raw_metrics['confirmation_logistic_balanced_accuracy'])} → {_pct(centered_metrics['confirmation_logistic_balanced_accuracy'])}；NCC {_pct(raw_metrics['confirmation_ncc_balanced_accuracy'])} → {_pct(centered_metrics['confirmation_ncc_balanced_accuracy'])}；SNR {float(raw_metrics['confirmation_class_balanced_snr_db']):.2f} → {float(centered_metrics['confirmation_class_balanced_snr_db']):.2f} dB。</p></div></div>
</article>"""


def band_appendix(band_root: Path) -> tuple[str, list[Path]]:
    blocks = []
    inputs: list[Path] = []
    for model in MODELS:
        directory = band_root / model
        audit_path = directory / "band_diagnostic.json"
        all_points_path = directory / "all_points.csv"
        confirmation_points_path = directory / "confirmation_points.csv"
        audit = read_json(audit_path)
        if str(audit.get("model_label")) != model:
            raise ValueError(f"Band diagnostic model mismatch: {audit_path}")
        if sorted(map(int, audit["scope"]["gold_counts"])) != list(range(1, 11)):
            raise ValueError(f"Band diagnostic is not all-count: {audit_path}")
        if int(audit["scope"]["full_trajectories"]) != 300:
            raise ValueError(f"Band diagnostic is not full-300: {audit_path}")
        if int(audit["scope"]["discovery_trajectories"]) != 200:
            raise ValueError(f"Band diagnostic is not discovery-200: {audit_path}")
        if int(audit["scope"]["confirmation_trajectories"]) != 100:
            raise ValueError(f"Band diagnostic is not confirmation-100: {audit_path}")
        blocks.append(
            _band_model_block(
                model,
                audit,
                read_csv(all_points_path),
                read_csv(confirmation_points_path),
            )
        )
        inputs.extend((audit_path, all_points_path, confirmation_points_path))
    return f"""
<section id="appendix-bands"><h2>Appendix B · Native-thinking 的上下分层</h2>
<p>本 appendix 对 Qwen 与 Gemma 使用同一诊断：native running site 固定为 <code>item_end</code>，layer 由 discovery 选择；在 discovery-fitted PCA3 中分别查看 full 300-source panel 与 confirmation 100-source panel，再比较 marker/seed/occurrence 关联和逐 trajectory 去均值后的结构。分成两团本身不是“两个计数器”的证据。</p>
{''.join(blocks)}
</section>""", inputs


def _dual_script(dual_visual: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dual_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const DUAL=__DUAL__;
const COLORS=['#6750E8','#00A9D8','#00A88F','#2DBE77','#A7C957','#D6B52C','#F29E4C','#E76F51','#D94B86','#8E5DB7'];
const VIEWS={};
function slug(model){return model.startsWith('Qwen')?'qwen':'gemma'}
function rotate3(x,y,z,view){const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z;return [x1,cp*y-sp*z1,sp*y+cp*z1]}
function dualIds(model,panel){const base='dual-'+slug(model)+'-'+panel;return {base,layer:document.getElementById(base+'-layer'),split:document.getElementById(base+'-split'),canvas:document.getElementById(base)}}
function drawDual3D(model,panel){
  const ids=dualIds(model,panel),payload=DUAL[model].panels[panel],layer=+ids.layer.value,cohort=ids.split.value,block=payload.coordinates[String(layer)],points=block.points.filter(p=>cohort==='all'||p[0]==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,stat=document.getElementById(ids.base+'-stats');c.clearRect(0,0,w,h);
  if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No states in this cohort.',20,30);return}
  const view=VIEWS[canvas.id]||(VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){if(!groups.has(p[2]))groups.set(p[2],[]);groups.get(p[2]).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>[k,ps.reduce((s,p)=>s+p[3],0)/ps.length,ps.reduce((s,p)=>s+p[4],0)/ps.length,ps.reduce((s,p)=>s+p[5],0)/ps.length,ps.length]);
  const rotated=points.map(p=>({p,r:rotate3(p[3],p[4],p[5],view)})),rcent=cent.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)})),maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[3]),Math.abs(p[4]),Math.abs(p[5])]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const xy=rotated.map(o=>o.r).concat(axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:23,r:23,t:18,b:22},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.25;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#2C3440';c.globalAlpha=.8;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const p=o.p,depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.40+.48*depth;c.fillStyle=COLORS[p[2]-1];c.strokeStyle=p[0]==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=p[0]==='confirmation'?2.1:1;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.6+1.2*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){const p=o.p;c.fillStyle=COLORS[p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.3;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(p[0]),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[6])).size,seeds=new Set(points.map(p=>p[0]+':'+p[1])).size,support=cent.map(p=>p[4]),metric=payload.metrics[String(layer)],evr=100*block.evr.reduce((a,b)=>a+b,0),source=cohort==='all'?300:100;
  stat.textContent=`${payload.token_site} · L${layer} · source ${source} trajectories across N=1…10 · displayed ${trajectories} trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C-held-out Log/NCC ${(100*metric.confirmation_logistic).toFixed(1)}%/${(100*metric.confirmation_ncc).toFixed(1)}% · SNR ${metric.confirmation_snr_db.toFixed(2)} dB`;
}
function setup(model,panel){const ids=dualIds(model,panel);ids.layer.addEventListener('change',()=>drawDual3D(model,panel));ids.split.addEventListener('change',()=>drawDual3D(model,panel));let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=VIEWS[ids.canvas.id]||(VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawDual3D(model,panel)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);drawDual3D(model,panel)}
for(const model of Object.keys(DUAL))for(const panel of Object.keys(DUAL[model].panels))setup(model,panel);
let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>{for(const model of Object.keys(DUAL))for(const panel of Object.keys(DUAL[model].panels))drawDual3D(model,panel)},100)});
""".replace("__DUAL__", payload)


def build_html(
    *,
    dual_results: Mapping[str, Mapping[str, Any]],
    dual_visual: Mapping[str, Any],
    marker_html: str,
    band_html: str,
) -> str:
    css = """
:root{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#626A74;--line:#C9C2B6;--indigo:#23165C;--teal:#00A88F;--yellow:#D6B52C}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}nav{position:sticky;top:0;z-index:5;display:flex;gap:18px;padding:10px 22px;background:rgba(243,238,228,.96);border-bottom:1px solid var(--line)}nav a{color:var(--indigo);font-size:13px;font-weight:750;text-decoration:none}main{max-width:1480px;margin:auto;padding:38px 28px 80px}header{max-width:1080px;border-bottom:2px solid var(--ink);padding-bottom:28px}.eyebrow{font:700 12px/1.2 Consolas,monospace;letter-spacing:.12em;color:var(--teal)}h1{font-size:44px;line-height:1.08;margin:10px 0 16px;letter-spacing:-.035em}h2{font-size:29px;margin:0 0 12px}h4{color:var(--indigo)}.lead{font-size:18px;color:#404852;max-width:92ch}section{padding:46px 0;border-bottom:1px solid var(--line)}.callout{max-width:1120px;background:var(--surface);border-left:4px solid var(--teal);padding:15px 19px;margin:20px 0}.warning{border-left-color:var(--yellow)}.definitions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}.definitions.two{grid-template-columns:repeat(2,minmax(0,1fr))}.definitions>div,.geometry-card,.appendix-model{background:var(--surface);border:1px solid var(--line);padding:17px}.definitions h3,.geometry-card h3{color:var(--indigo);margin:0 0 8px;font-size:17px}.definitions p,.geometry-card p{font-size:13px;color:var(--muted);margin:0 0 12px}.controls{display:flex;gap:12px;flex-wrap:wrap}.controls label{font-size:12px;font-weight:700;color:var(--muted)}select{display:block;margin-top:4px;border:1px solid var(--line);background:var(--surface);padding:7px 28px 7px 9px;color:var(--ink)}.dual-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:28px}.geometry-card canvas{display:block;width:100%;height:390px;background:#F8F4EC;border:1px solid #DDD5C9;touch-action:none;cursor:grab}.geometry-card canvas:active{cursor:grabbing}.rotate-hint{margin-top:5px;color:#7A7270;font:10px/1.4 Consolas,monospace}.panel-stats{min-height:70px;margin-top:7px;color:var(--muted);font:12px/1.5 Consolas,monospace}.table-scroll{overflow:auto;background:var(--surface);border:1px solid var(--line);margin:16px 0 22px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #DED8CE}th{background:#ECE6DA;color:#303744}.muted,.small{color:var(--muted);font-size:12px}.band-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:18px 0}.band-figure{margin:0;background:#FFFDF8;border:1px solid var(--line);padding:12px}.band-figure h3{font-size:15px;color:var(--indigo);margin:0 0 8px}.band-figure svg{display:block;width:100%;height:auto}.appendix-model{margin:22px 0}.provenance{font:11px/1.6 Consolas,monospace;color:var(--muted)}details{background:var(--surface);border:1px solid var(--line);margin:18px 0}summary{cursor:pointer;padding:12px 15px;font-weight:750;color:var(--indigo)}@media(max-width:1000px){.dual-grid,.definitions,.definitions.two,.band-grid{grid-template-columns:1fr}}@media(max-width:650px){main{padding:25px 13px 60px}h1{font-size:34px}.geometry-card canvas{height:330px}}
"""
    script = _dual_script(dual_visual)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Geometry Comparison</title><style>{css}</style></head><body>
<nav><a href="#scope">口径</a><a href="#tokens">Token 提取</a><a href="#dual">主结果</a><a href="#claims">结论</a><a href="#analysis-role">分析取舍</a><a href="#appendix-markers">Marker appendix</a><a href="#appendix-bands">分层 appendix</a></nav><main>
<header><div class="eyebrow">REALISTIC NIAH · ALL-COUNT GEOMETRY</div><h1>NiaH Geometry Comparison</h1><p class="lead">Running index 与 final count 两组比较都覆盖 N=1…10，并可在完整 300 trajectories 与 confirmation 100 trajectories 之间切换。Running index 固定比较 prompt <code>span_end</code> 与 thinking-trace <code>item_end</code>；两个模式只各自选择最佳 decoder layer。</p></header>
<section id="scope"><h2>严格比较口径</h2><div class="definitions"><div><h3>Full 300</h3><p>10 个 gold N × 30 seeds。它是 descriptive geometry view；PCA3 仍只由 discovery 200 拟合，避免 confirmation 反向选显示 basis。</p></div><div><h3>Confirmation 100</h3><p>10 个 gold N × 10 held-out seeds。主表的 Logistic、nearest-centroid 与 SNR 都是 discovery-frozen 后在这里评价。</p></div><div><h3>Native running 的 ragged rule</h3><p>每条 trace 只贡献 parser 实际观察到的 1…M。数到 8 就贡献八个 states；不按 gold N 或最终 Total 补到 9/10。</p></div></div></section>
<section id="tokens"><h2>Token 提取与两个独立 filestream</h2><div class="definitions two"><div><h3>Running index · fixed end</h3><p>Non-thinking 固定读取 prompt 第 k 个 needle span 的最后一个 token：<code>hidden[0, end−1]</code>。Native-thinking 先在原始 response 字符串中定位第 k 个完整 city-count item，再用保存的原始 <code>output_token_ids</code> 做 exact-prefix 对齐，固定读取 <code>item_end</code>：query position = <code>prompt_token_count + prefix_token_count − 1</code>。两侧都只使用一个自然边界 token，主分析不再搜索其他 token sites。</p></div><div><h3>Final count · answer_query_v3</h3><p><code>answer_query_v3</code> 直接从最后一个 literal <code>Total: &lt;integer&gt;</code> 提取，边界停在数字首字符前；它已与 running parser 的 detected/miss 状态解耦。每条 trajectory 必须恰好一个该站点，并单独物化为 final-count capture 后再进入报告。</p></div></div></section>
{dual_endpoint_section(dual_results, dual_visual)}
{empirical_claims(dual_results)}
<section id="analysis-role"><h2>为什么删除 trace-format × site × layer 大段</h2><div class="callout"><strong>结论：</strong>不放在主报告。按 marker-format 再分别搜索 token site 与 layer，会在支持不均的小 strata 中产生大量 post-hoc choices；它主要回答 parser 表面格式，而不是 cross-mode representation。保留 marker 比例和 band 归因作为 appendix 混杂诊断。</div><p>主结果把 running token site 直接固定为 <code>span_end/item_end</code>，仅由 discovery grouped-CV 选择各自 layer；confirmation 100 只评价冻结层。删掉的是 site search 和重复的 every-layer/per-format held-out 曲线，不是删掉防止 selection bias 的 held-out 设计。</p></section>
<section id="snr"><h2>SNR 的读法</h2><p>在 discovery-fitted PCA16-whitened 空间中，SNR = 各 count centroid 围绕 class-balanced grand centroid 的平均平方距离 ÷ 各 count 内部的平均平方残差；报告 dB = 10 log<sub>10</sub>(SNR)。0 dB 表示类间 centroid energy 与类内 scatter 相当，越高表示类间/类内比越高。它与分类 accuracy 不等价：两种 probe 提高但 SNR 不提高时，只能 claim “更可解码”，不能 claim “所有簇更紧”。</p></section>
{marker_html}{band_html}
<section><h2>解释边界</h2><p>这些图和 probes 证明的是 within-task decodability/geometry，不单独证明离散计数器、逐步加一算法或因果使用。两个 mode 的 end token 语义和最佳层仍不同，因此比较的是两个单-token 完成边界上同一任务变量的可读性，而不是共享坐标系中的绝对距离。</p><p class="provenance">Report schema: {REPORT_SCHEMA_VERSION} · pooled 10 counts × 30 seeds · full/confirmation views: 300/100 trajectories · running sites fixed: span_end/item_end · layer selector: pooled discovery only · trace-format sweep: appendix-only diagnostic</p></section>
</main><script>{script}</script></body></html>"""


def build_report(
    *,
    non_thinking_export_root: Path,
    native_running_root: Path,
    native_final_root: Path,
    dual_endpoint_root: Path,
    parser_audit: Path,
    band_root: Path,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    dual_results, dual_inputs = load_dual_endpoint_results(
        dual_endpoint_root.resolve()
    )
    dual_visual, visual_inputs = build_dual_visual_data(
        non_thinking_export_root.resolve(),
        native_running_root.resolve(),
        native_final_root.resolve(),
        dual_results,
    )
    parser_rows = read_jsonl(parser_audit.resolve())
    band_html, band_inputs = band_appendix(band_root.resolve())
    document = build_html(
        dual_results=dual_results,
        dual_visual=dual_visual,
        marker_html=marker_appendix(parser_rows),
        band_html=band_html,
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    inputs = sorted(
        set(
            dual_inputs
            + visual_inputs
            + band_inputs
            + [parser_audit.resolve()]
        ),
        key=str,
    )
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {
            "full": "10 gold counts x 30 seeds = 300 trajectories",
            "confirmation": "10 gold counts x 10 seeds = 100 trajectories",
        },
        "running_state_rule": "parser-observed 1..M only; never pad to gold N or final Total",
        "answer_query_v3": "independent final-answer extraction; exactly one site per trajectory",
        "primary_analysis": "running sites fixed to span_end/item_end; pooled discovery-only layer selection; frozen confirmation evaluation",
        "trace_format_site_layer_sweep": "omitted from main; marker/band diagnostics moved to appendix",
        "inputs": {str(path): sha256(path) for path in inputs},
        "output": str(output),
        "output_sha256": sha256(output),
    }
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-thinking-export-root", type=Path, required=True)
    parser.add_argument("--native-running-root", type=Path, required=True)
    parser.add_argument("--native-final-root", type=Path, required=True)
    parser.add_argument("--dual-endpoint-root", type=Path, required=True)
    parser.add_argument("--parser-audit", type=Path, required=True)
    parser.add_argument("--band-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_report(
        non_thinking_export_root=args.non_thinking_export_root,
        native_running_root=args.native_running_root,
        native_final_root=args.native_final_root,
        dual_endpoint_root=args.dual_endpoint_root,
        parser_audit=args.parser_audit,
        band_root=args.band_root,
        output=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
