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


REPORT_SCHEMA_VERSION = "niah_geometry_comparison_v10_entity_domain_transfer"


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


def _accuracy_svg(
    model: str,
    dual_results: Mapping[str, Mapping[str, Any]],
) -> str:
    rows: list[tuple[str, float, float]] = []
    for endpoint, endpoint_label in (("running", "Running"), ("final", "Final")):
        selected = _selected_by_mode(dual_results, model, endpoint)
        for field, metric_label in (
            ("confirmation_logistic_balanced_accuracy", "Logistic"),
            ("confirmation_ncc_balanced_accuracy", "Nearest centroid"),
        ):
            rows.append(
                (
                    f"{endpoint_label} · {metric_label}",
                    float(selected["non_thinking"][field]),
                    float(selected["native_thinking"][field]),
                )
            )
    width, height = 700, 260
    x0, x1 = 170.0, 520.0
    scale = lambda value: x0 + float(value) * (x1 - x0)
    grid = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = scale(tick)
        grid.append(
            f'<line class="metric-gridline" x1="{x:.1f}" y1="31" '
            f'x2="{x:.1f}" y2="224"/><text class="metric-tick" '
            f'x="{x:.1f}" y="19" text-anchor="middle">{100*tick:.0f}%</text>'
        )
    marks = []
    for index, (label, non, native) in enumerate(rows):
        y = 56 + 47 * index
        non_x, native_x = scale(non), scale(native)
        marks.append(
            f'<text class="metric-label" x="8" y="{y+4}">{esc(label)}</text>'
            f'<line class="metric-link" x1="{non_x:.1f}" y1="{y}" '
            f'x2="{native_x:.1f}" y2="{y}"/>'
            f'<circle class="metric-dot metric-non" cx="{non_x:.1f}" cy="{y}" r="6"/>'
            f'<circle class="metric-dot metric-native" cx="{native_x:.1f}" cy="{y}" r="6"/>'
            f'<text class="metric-value" x="542" y="{y+4}">'
            f'{100*non:.1f} → {100*native:.1f}%</text>'
        )
    return (
        f'<figure class="metric-figure"><h3>{esc(model)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(model)} confirmation balanced accuracy, non-thinking versus native-thinking">'
        f'<title>{esc(model)} confirmation balanced accuracy</title>'
        f'<desc>Four held-out comparisons. Dark circles are non-thinking and teal circles are native-thinking.</desc>'
        + "".join(grid)
        + "".join(marks)
        + "</svg></figure>"
    )


def empirical_claims(
    dual_results: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render frozen Confirmation-100 probe evidence and a direct visual comparison."""

    verdicts = []
    charts = []
    for model in MODELS:
        wins = []
        deltas = []
        for endpoint in ("running", "final"):
            rows = _selected_by_mode(dual_results, model, endpoint)
            for field in (
                "confirmation_logistic_balanced_accuracy",
                "confirmation_ncc_balanced_accuracy",
            ):
                non = float(rows["non_thinking"][field])
                native = float(rows["native_thinking"][field])
                wins.append(native > non)
                deltas.append(native - non)
        verdict = (
            "四个 endpoint × probe 对比均为 native-thinking 更高"
            if all(wins)
            else f"四个对比中有 {sum(wins)}/4 个为 native-thinking 更高"
        )
        verdicts.append(
            f"<li><strong>{esc(model)}：</strong>{verdict}；balanced-accuracy "
            f"差值范围为 {100*min(deltas):+.1f}–{100*max(deltas):+.1f} 个百分点。</li>"
        )
        charts.append(_accuracy_svg(model, dual_results))
    return f"""
<section id="claims"><h2>Confirmation 100 的可支持结论</h2>
<div class="callout"><strong>结论：</strong>在两种 mode 各自由 discovery 选择最佳层、并冻结到相同的 N=1…10 held-out panel 后，native-thinking 的 count variable 更容易被 Logistic 与 nearest-centroid probe 解码。这里的主 claim 是 <em>more decodable</em>；SNR 是否支持“更紧”在下一节单独判断。</div>
<div class="metric-legend"><span><i class="legend-non"></i>non-thinking</span><span><i class="legend-native"></i>native-thinking</span><span>右侧数值：non → native</span></div>
<div class="metric-grid">{''.join(charts)}</div>
<ul>{''.join(verdicts)}</ul>
</section>"""


def _retained_label_text(value: Mapping[str, Any]) -> str:
    labels = list(map(int, value["retained_labels"]))
    if labels == list(range(1, 11)):
        return "k=1…10"
    if labels and labels == list(range(labels[0], labels[-1] + 1)):
        return f"k={labels[0]}…{labels[-1]}"
    return "k=" + ",".join(map(str, labels))


def _support_range(value: Mapping[str, Any]) -> str:
    retained = set(map(int, value["retained_labels"]))
    supports = [
        int(count)
        for label, count in value["support"].items()
        if int(label) in retained
    ]
    return f"nₖ={min(supports)}–{max(supports)}" if supports else "no retained k"


def _snr_svg(
    model: str,
    dual_results: Mapping[str, Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    domain: tuple[float, float],
) -> str:
    running = _selected_by_mode(dual_results, model, "running")
    final = _selected_by_mode(dual_results, model, "final")
    conditioned = audit["ordinal_decodability"]["band_conditioned_confirmation_snr"]
    upper = conditioned["per_band"]["upper"]
    lower = conditioned["per_band"]["lower"]
    rows: list[tuple[str, list[tuple[str, float]]]] = [
        (
            "Running · global",
            [
                ("non", float(running["non_thinking"]["confirmation_class_balanced_snr_db"])),
                ("native", float(running["native_thinking"]["confirmation_class_balanced_snr_db"])),
            ],
        ),
        ("Native upper · within-band", [("upper", float(upper["snr_db"]))]),
        ("Native lower · within-band", [("lower", float(lower["snr_db"]))]),
        (
            "Final · global",
            [
                ("non", float(final["non_thinking"]["confirmation_class_balanced_snr_db"])),
                ("native", float(final["native_thinking"]["confirmation_class_balanced_snr_db"])),
            ],
        ),
    ]
    width, height = 700, 260
    x0, x1 = 190.0, 520.0
    low, high = domain
    scale = lambda value: x0 + (float(value) - low) / (high - low) * (x1 - x0)
    ticks = list(range(math.ceil(low), math.floor(high) + 1, 2))
    if 0 not in ticks and low <= 0 <= high:
        ticks.append(0)
        ticks.sort()
    grid = []
    for tick in ticks:
        x = scale(tick)
        zero_class = " metric-zero" if tick == 0 else ""
        tick_label = "0" if tick == 0 else f"{tick:+d}"
        grid.append(
            f'<line class="metric-gridline{zero_class}" x1="{x:.1f}" y1="31" '
            f'x2="{x:.1f}" y2="224"/><text class="metric-tick" '
            f'x="{x:.1f}" y="19" text-anchor="middle">{tick_label}</text>'
        )
    marks = []
    for index, (label, values) in enumerate(rows):
        y = 56 + 47 * index
        marks.append(f'<text class="metric-label" x="8" y="{y+4}">{esc(label)}</text>')
        if len(values) == 2:
            marks.append(
                f'<line class="metric-link" x1="{scale(values[0][1]):.1f}" y1="{y}" '
                f'x2="{scale(values[1][1]):.1f}" y2="{y}"/>'
            )
        for kind, value in values:
            marks.append(
                f'<circle class="metric-dot snr-{kind}" cx="{scale(value):.1f}" '
                f'cy="{y}" r="6"/>'
            )
        summary = " / ".join(f"{kind} {value:+.2f}" for kind, value in values)
        marks.append(
            f'<text class="metric-value" x="538" y="{y+4}">{esc(summary)}</text>'
        )
    return (
        f'<figure class="metric-figure"><h3>{esc(model)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(model)} confirmation signal to noise ratio in decibels">'
        f'<title>{esc(model)} confirmation SNR</title>'
        f'<desc>Global non-thinking and native-thinking SNR plus native-thinking upper and lower discovery-fitted within-band SNR.</desc>'
        + "".join(grid)
        + "".join(marks)
        + '<text class="metric-axis-title" x="355" y="250" text-anchor="middle">SNR (dB)</text>'
        + "</svg></figure>"
    )


def snr_section(
    dual_results: Mapping[str, Mapping[str, Any]],
    band_audits: Mapping[str, Mapping[str, Any]],
) -> str:
    values: list[float] = []
    for model in MODELS:
        for endpoint in ("running", "final"):
            selected = _selected_by_mode(dual_results, model, endpoint)
            values.extend(
                float(selected[mode]["confirmation_class_balanced_snr_db"])
                for mode in ("non_thinking", "native_thinking")
            )
        conditioned = band_audits[model]["ordinal_decodability"][
            "band_conditioned_confirmation_snr"
        ]
        values.extend(
            float(conditioned["per_band"][name]["snr_db"])
            for name in ("upper", "lower")
            if math.isfinite(float(conditioned["per_band"][name]["snr_db"]))
        )
    domain = (float(math.floor(min(values)) - 1), float(math.ceil(max(values)) + 1))
    charts = [
        _snr_svg(model, dual_results, band_audits[model], domain=domain)
        for model in MODELS
    ]
    rows = []
    interpretations = []
    for model in MODELS:
        audit = band_audits[model]
        global_snr = float(
            audit["ordinal_decodability"]["raw"][
                "confirmation_class_balanced_snr_db"
            ]
        )
        conditioned = audit["ordinal_decodability"][
            "band_conditioned_confirmation_snr"
        ]
        upper = conditioned["per_band"]["upper"]
        lower = conditioned["per_band"]["lower"]
        macro = conditioned["macro_within_band"]
        complete = (
            upper["retained_labels"] == list(range(1, 11))
            and lower["retained_labels"] == list(range(1, 11))
        )
        macro_text = (
            f"{float(macro['snr_db']):+.2f} dB"
            if complete
            else "不汇总：两带的 k 范围不同"
        )
        rows.append(
            (
                esc(model),
                f"{global_snr:+.2f} dB",
                f"{float(upper['snr_db']):+.2f} dB<br><span class='small'>{esc(_retained_label_text(upper))}; {esc(_support_range(upper))}</span>",
                f"{float(lower['snr_db']):+.2f} dB<br><span class='small'>{esc(_retained_label_text(lower))}; {esc(_support_range(lower))}</span>",
                macro_text,
            )
        )
        if complete:
            direction = "上升" if float(macro["snr_db"]) > global_snr else "下降"
            interpretations.append(
                f"<li><strong>{esc(model)}：</strong>upper 与 lower 都覆盖 k=1…10；"
                f"从 global {global_snr:+.2f} dB 到 equal-band macro "
                f"{float(macro['snr_db']):+.2f} dB（{direction}）。这说明 band offset "
                "确实贡献了同一 k 的跨-template scatter，但该条件化结果仍是 post-hoc nuisance diagnostic。</li>"
            )
        else:
            interpretations.append(
                f"<li><strong>{esc(model)}：</strong>upper 只保留 {_retained_label_text(upper)}，"
                f"lower 保留 {_retained_label_text(lower)}。上下分层本身与 running position "
                "纠缠，因此两条 SNR 不是同一个 10-class estimand；不能用 macro 改写主结论。</li>"
            )
    return f"""
<section id="snr"><h2>SNR：global 与 band-conditioned 要回答两个不同问题</h2>
<div class="definitions two"><div><h3>Global SNR（主指标）</h3><p>在 discovery-fitted PCA16-whitened 空间中，对 confirmation 每个 k 求 centroid。Signal 是各 k centroid 到 class-balanced grand centroid 的平均平方距离；noise 是各 k 内残差平方距离的 class-balanced 平均。SNR<sub>dB</sub>=10 log<sub>10</sub>(signal/noise)。同一个 k 若落在不同 trace-template band，band offset 会进入 noise——这正是未条件化表示的总变异。</p></div><div><h3>Within-band SNR（混杂诊断）</h3><p>先只用 discovery PCA3 拟合两个 K-means band 并冻结，再把 confirmation 指派到 upper/lower。在每条 band 内以该 band 自己的 grand centroid 计算 signal，并以 (band,k) centroid 计算 residual；因此上下 band 的均值差既不算 signal，也不算 noise。每个 k 在该 band 至少需要 2 states，否则剔除并公开 support。</p></div></div>
<div class="metric-legend"><span><i class="legend-non"></i>non global</span><span><i class="legend-native"></i>native global</span><span><i class="legend-upper"></i>native upper</span><span><i class="legend-lower"></i>native lower</span></div>
<div class="metric-grid">{''.join(charts)}</div>
{table(['模型','Native running · global','Upper · conditional','Lower · conditional','Equal-band macro'], rows)}
<ul>{''.join(interpretations)}</ul>
<div class="callout warning"><strong>解释边界：</strong>条件化能回答“每条 band 内的 count geometry 是否清楚”，但不能取代跨 mode 的 primary global comparison，因为 non-thinking 没有经过同样的 hidden-state-derived 分带。Probe accuracy 与 SNR 也不等价：判别器可利用少数方向，而 SNR 把 PCA16 中的残差能量各向同性相加。</div>
</section>"""


def _capture_manifest_example(
    root: Path,
    model: str,
    site_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    index_path = root / model / "capture_index.jsonl"
    rows = read_jsonl(index_path)
    if not rows:
        raise ValueError(f"Empty capture index: {index_path}")
    row = sorted(rows, key=lambda value: (int(value["seed"]), int(value["gold_count"])))[0]
    manifest_path = index_path.parent / str(row["manifest_path"])
    manifest = read_json(manifest_path)
    sites = [
        value
        for value in manifest["site_rows"]
        if str(value.get("site_kind")) == site_kind
    ]
    if site_kind == "item_end":
        sites = [value for value in sites if int(value.get("occurrence") or 0) == 1]
    if len(sites) != 1:
        raise ValueError(
            f"Expected one {site_kind} example in {manifest_path}; got {len(sites)}"
        )
    return row, sites[0], index_path, manifest_path


def token_extraction_section(
    native_running_root: Path,
    native_final_root: Path,
) -> tuple[str, list[Path]]:
    running_row, running_site, running_index, running_manifest_path = (
        _capture_manifest_example(
            native_running_root, "Qwen3-8B", "item_end"
        )
    )
    final_row, final_site, final_index, final_manifest_path = (
        _capture_manifest_example(
            native_final_root, "Qwen3-8B", "answer_query_v3"
        )
    )
    if str(running_row["request_id"]) != str(final_row["request_id"]):
        raise ValueError("Running and final token examples do not refer to one trace")
    running_manifest = read_json(running_manifest_path)
    event = running_manifest["episode_parse"]["events"][0]
    prompt_tokens = int(running_manifest["prompt_token_count"])
    running_prefix = int(running_site["prefix_token_count"])
    running_endpoint = int(running_site["endpoint_token"])
    running_global = prompt_tokens + running_endpoint
    final_prefix = int(final_site["prefix_token_count"])
    final_endpoint = int(final_site["endpoint_token"])
    final_global = prompt_tokens + final_endpoint
    if running_endpoint != running_prefix - 1 or final_endpoint != final_prefix - 1:
        raise ValueError("Example endpoint is not the last token of the saved prefix")
    html = f"""
<section id="tokens"><h2>Token 提取：边界、索引与实际例子</h2>
<div class="callout"><strong>统一约定：</strong>所有 token index 都从 0 开始；区间 <code>[start,end)</code> 的 <code>end</code> 不包含在 span 内。报告读取的是所选 decoder block 输出中<strong>边界前最后一个 token</strong>的 post-block hidden state，始终是单 token endpoint，不做 span mean。</div>
<div class="token-flow">
<article><h3>1 · Non-thinking prompt：<code>span_end</code></h3><p>第 k 个 needle 在 prompt tokenizer 下已有精确 token span <code>[sₖ,eₖ)</code>。固定读取 <code>h<sup>(ℓ)</sup>[eₖ−1]</code>；既不取城市前一个 token，也不把整个 needle 平均。下面是索引示意，数字只用于说明 exclusive-end：</p><div class="token-strip"><span>…</span><span data-pos="8421">The</span><span data-pos="8422">Chicago</span><span data-pos="8423">entry</span><span data-pos="8424">scored</span><span data-pos="8425">72</span><span class="picked" data-pos="8426">.</span><b>eₖ=8427</b></div><p class="small">span=<code>[8421,8427)</code>，所以 endpoint=<code>8426</code>。实际 capture 直接由每条 stimulus 的 <code>needle_spans</code> 生成这些位置，并为每个 k 各保存一个 state。</p></article>
<article><h3>2 · Native running：<code>item_end</code></h3><p>Parser 先在原始 response 字符串中得到第 k 个完整 item 的字符边界，再对 <code>response[:char_end]</code> 做 exact-prefix token alignment；endpoint 是该 prefix 的最后一个 output token。</p><div class="boundary-example"><span>… {esc(event['city'])} … {esc(event['evidence_surface'])}</span><i></i><span>&lt;/think&gt; …</span></div><p>实际 Qwen 例（seed {int(running_row['seed'])}, gold N={int(running_row['gold_count'])}, k=1）：字符 item=<code>[{int(running_site['char_start'])},{int(running_site['char_end'])})</code>；prefix 有 {running_prefix} 个 output tokens，因此 output endpoint=<code>{running_endpoint}</code>。拼回 prompt 后，global hidden index=<code>{prompt_tokens}+{running_prefix}−1={running_global}</code>。本例边界跨 tokenizer piece，使用 <code>{esc(running_site['alignment_strategy'])}</code>，不是近邻猜测。</p></article>
<article><h3>3 · Native final：<code>answer_query_v3</code></h3><p>独立 parser 寻找最后一个 literal <code>Total: &lt;integer&gt;</code>，边界停在数字首字符之前。因此读取的是“模型即将写出最终数字”时的 prefix endpoint，而不是数字 token 本身。</p><div class="boundary-example"><span>… &lt;/think&gt; Total: </span><i></i><span class="answer-token">1</span></div><p>同一条 Qwen trace：字符 query=<code>[{int(final_site['char_start'])},{int(final_site['char_end'])})</code>；prefix={final_prefix} output tokens，output endpoint=<code>{final_endpoint}</code>，global hidden index=<code>{prompt_tokens}+{final_prefix}−1={final_global}</code>。Final filestream 只保留这一站点，每条 trajectory 恰好一个 state。</p></article>
<article><h3>为什么两侧可以比较</h3><p>Running 比较的是“第 k 个计数单元完成后”的单-token state：prompt needle 完成边界 <code>span_end</code> 对 thinking item 完成边界 <code>item_end</code>。Final 比较的是两侧各自即将输出总数的 query state。语义角色对齐，但 token 字面值、绝对层号与坐标系不要求相同；每种 mode 只在 discovery 内选自己的 layer。</p></article>
</div></section>"""
    return html, [
        running_index.resolve(),
        running_manifest_path.resolve(),
        final_index.resolve(),
        final_manifest_path.resolve(),
    ]


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
) -> str:
    slug = "qwen" if model.startswith("Qwen") else "gemma"
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
<p>固定分析站点为 <code>{esc(audit['site_kind'])}</code> @ L{int(audit['layer'])}。原始图问“上下分层是否存在”；centered 图先在原 hidden space 中逐 trajectory 减去自己的 state mean，再重新用 discovery 拟合 PCA3，问“去掉整条 trace 的 offset 后，原来的几何分层还剩多少”。两张图都可以拖动旋转；颜色可切换为 frozen band、running k 或 marker。</p>
<div class="controls band-controls"><label>Cohort<select id="band-{slug}-cohort"><option value="all">完整 300 trajectories</option><option value="confirmation">Confirmation 100</option></select></label><label>Color<select id="band-{slug}-color"><option value="band">Frozen upper/lower band</option><option value="occurrence">Running k</option><option value="marker">Trace marker</option></select></label></div>
<div class="band-grid"><figure class="band-figure"><h4>Raw · discovery-fitted PCA3</h4><canvas id="band-{slug}-raw" role="img" aria-label="{esc(model)} raw native-thinking geometry in three dimensions"></canvas><p class="rotate-hint">drag to rotate · band centers fitted on discovery only</p><p class="panel-stats" id="band-{slug}-raw-stats"></p></figure><figure class="band-figure"><h4>Trajectory-centered · discovery-fitted PCA3</h4><canvas id="band-{slug}-centered" role="img" aria-label="{esc(model)} trajectory-centered native-thinking geometry in three dimensions"></canvas><p class="rotate-hint">drag to rotate · colors keep the raw frozen-band identity</p><p class="panel-stats" id="band-{slug}-centered-stats"></p></figure></div>
<div class="band-dynamic-legend" id="band-{slug}-legend"></div>
<div class="definitions two"><div><h3>Full 300 view</h3><p>{int(scope['full_trajectories'])} trajectories / {int(scope['full_states'])} states；raw frozen-band silhouette={float(raw_full['silhouette']):.3f}，upper/lower={int(raw_full['cluster_sizes']['upper'])}/{int(raw_full['cluster_sizes']['lower'])}。这里包含 discovery 与 confirmation，只用于描述。</p></div><div><h3>Confirmation 100 view</h3><p>{int(scope['confirmation_trajectories'])} trajectories / {int(scope['confirmation_states'])} states；raw frozen-band silhouette={float(raw_confirmation['silhouette']):.3f}，upper/lower={int(raw_confirmation['cluster_sizes']['upper'])}/{int(raw_confirmation['cluster_sizes']['lower'])}。Band centers 和“upper”命名都已经由 discovery 冻结。</p></div></div>
{table(['candidate nuisance','NMI · full','NMI · confirmation'], association_rows)}
<div class="definitions two"><div><h3>Band 是否主要是 trace offset</h3><p>Confirmation 中 raw-band 与 centered-space 新 band 的 NMI={float(centered['raw_vs_centered_band_nmi']):.3f}。越接近 0，越说明原上下标签在去 trajectory mean 后不再对应同一分层；越接近 1，越说明分层不是一个简单的 trajectory-level translation。</p></div><div><h3>去 offset 后 count signal 是否还在</h3><p>confirmation Logistic {_pct(raw_metrics['confirmation_logistic_balanced_accuracy'])} → {_pct(centered_metrics['confirmation_logistic_balanced_accuracy'])}；NCC {_pct(raw_metrics['confirmation_ncc_balanced_accuracy'])} → {_pct(centered_metrics['confirmation_ncc_balanced_accuracy'])}；SNR {float(raw_metrics['confirmation_class_balanced_snr_db']):.2f} → {float(centered_metrics['confirmation_class_balanced_snr_db']):.2f} dB。Centering 使用整条 trajectory，只是混杂诊断，不是在线 estimator 或因果干预。</p></div></div>
</article>"""


def _compact_band_points(rows: list[dict[str, str]]) -> list[list[Any]]:
    return [
        [
            str(row["split"]),
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["occurrence"]),
            str(row["marker_kind"]),
            str(row["band"]),
            round(float(row["pc1"]), 5),
            round(float(row["pc2"]), 5),
            round(float(row["pc3"]), 5),
            round(float(row["centered_pc1"]), 5),
            round(float(row["centered_pc2"]), 5),
            round(float(row["centered_pc3"]), 5),
        ]
        for row in rows
    ]


def band_appendix(
    band_root: Path,
) -> tuple[str, list[Path], dict[str, Any], dict[str, Mapping[str, Any]]]:
    blocks = []
    inputs: list[Path] = []
    visual: dict[str, Any] = {}
    audits: dict[str, Mapping[str, Any]] = {}
    for model in MODELS:
        directory = band_root / model
        audit_path = directory / "band_diagnostic.json"
        all_points_path = directory / "all_points.csv"
        confirmation_points_path = directory / "confirmation_points.csv"
        audit = read_json(audit_path)
        if str(audit.get("schema_version")) != "native_geometry_band_diagnostic_v2_frozen_band":
            raise ValueError(f"Band diagnostic is not discovery-frozen v2: {audit_path}")
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
        all_points = read_csv(all_points_path)
        confirmation_points = read_csv(confirmation_points_path)
        filtered_confirmation = [
            row for row in all_points if str(row["split"]) == "confirmation"
        ]
        if len(filtered_confirmation) != len(confirmation_points):
            raise ValueError(f"Band point cohorts disagree: {directory}")
        blocks.append(_band_model_block(model, audit))
        audits[model] = audit
        visual[model] = {
            "layer": int(audit["layer"]),
            "site": str(audit["site_kind"]),
            "raw_evr": [
                round(float(value), 7)
                for value in audit["display_pca3"]["explained_variance_ratio"]
            ],
            "centered_evr": [
                round(float(value), 7)
                for value in audit["within_trajectory_centered_pca3"][
                    "explained_variance_ratio"
                ]
            ],
            "points": _compact_band_points(all_points),
        }
        inputs.extend((audit_path, all_points_path, confirmation_points_path))
    return f"""
<section id="appendix-bands"><h2>Appendix B · Native-thinking 的上下分层</h2>
<div class="definitions"><div><h3>Step 1 · 冻结分带</h3><p>StandardScaler 与 PCA3 只在 discovery states 上拟合；K-means 的两个中心也只在 discovery 拟合。再用冻结中心给 full/confirmation 指派 band。“upper/lower”只是固定初始相机下的显示名称，不是模型内生标签。</p></div><div><h3>Step 2 · 找分带在跟随什么</h3><p>分别计算 band 与 marker、seed、running k、boundary 的 NMI。NMI=0 表示在该样本中无离散关联，NMI=1 表示一方完全决定另一方；它不提供因果方向。</p></div><div><h3>Step 3 · 去 trajectory offset</h3><p>逐 trajectory 在原 hidden space 减去自己的 mean，再重新拟合 discovery PCA3。如果原 band 消失而 ordinal probe 保留，更符合“trace/template offset 叠加在 count geometry 上”。</p></div></div>
<div class="callout warning"><strong>不要把两团直接叫两个计数器：</strong>PCA3 只是总方差的低维显示，K-means 又强制给出两组。必须同时看 frozen confirmation silhouette、NMI、support、trajectory centering 与 within-band SNR；任何单张图都不足以识别机制。</div>
{''.join(blocks)}
</section>""", inputs, visual, audits


def _domain_dimension_svg(model: str, payload: Mapping[str, Any]) -> str:
    """Render PCA-dimension sensitivity for the frozen answer layer."""

    modes = ("non_thinking", "native_thinking")
    dimensions = [
        int(row["dimensions"])
        for row in payload[modes[0]]["metrics"]["dimension_sweep"]
    ]
    if dimensions != [1, 2, 4, 8, 16, 32]:
        raise ValueError(f"Unexpected domain-transfer PCA dimensions for {model}")
    width, height = 760, 330
    panels = (
        (
            "City confirmation",
            "city_logistic_balanced_accuracy",
            52.0,
            350.0,
        ),
        (
            "City → flower/animal",
            "cross_domain_logistic_balanced_accuracy",
            422.0,
            720.0,
        ),
    )
    y_top, y_bottom = 54.0, 268.0

    def sx(dimension: int, left: float, right: float) -> float:
        return left + math.log2(dimension) / 5.0 * (right - left)

    def sy(value: float) -> float:
        return y_bottom - float(value) * (y_bottom - y_top)

    elements = []
    for title, metric, left, right in panels:
        elements.append(
            f'<text class="metric-label" x="{(left+right)/2:.1f}" y="27" '
            f'text-anchor="middle">{esc(title)}</text>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = sy(tick)
            elements.append(
                f'<line class="metric-gridline" x1="{left:.1f}" y1="{y:.1f}" '
                f'x2="{right:.1f}" y2="{y:.1f}"/>'
            )
            if left == panels[0][2]:
                elements.append(
                    f'<text class="metric-tick" x="{left-8:.1f}" y="{y+4:.1f}" '
                    f'text-anchor="end">{100*tick:.0f}%</text>'
                )
        chance_y = sy(0.1)
        elements.append(
            f'<line class="domain-chance" x1="{left:.1f}" y1="{chance_y:.1f}" '
            f'x2="{right:.1f}" y2="{chance_y:.1f}"/>'
        )
        for dimension in dimensions:
            x = sx(dimension, left, right)
            elements.append(
                f'<text class="metric-tick" x="{x:.1f}" y="288" '
                f'text-anchor="middle">{dimension}</text>'
            )
        for mode, css_class in (
            ("non_thinking", "domain-line-non"),
            ("native_thinking", "domain-line-native"),
        ):
            rows = payload[mode]["metrics"]["dimension_sweep"]
            points = [
                (sx(int(row["dimensions"]), left, right), sy(float(row[metric])))
                for row in rows
            ]
            elements.append(
                f'<polyline class="domain-dim-line {css_class}" style="fill:none" points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                + '"/>'
            )
            for x, y in points:
                shape = (
                    f'<circle class="domain-dim-mark {css_class}" cx="{x:.1f}" '
                    f'cy="{y:.1f}" r="4.5"/>'
                    if mode == "non_thinking"
                    else f'<rect class="domain-dim-mark {css_class}" x="{x-4.5:.1f}" '
                    f'y="{y-4.5:.1f}" width="9" height="9"/>'
                )
                elements.append(shape)
    elements.append(
        '<text class="metric-axis-title" x="380" y="319" '
        'text-anchor="middle">PCA dimensions (fit on layer-selection seeds only)</text>'
    )
    return (
        f'<figure class="metric-figure domain-dim-figure"><h3>{esc(model)} · '
        f'Logistic dimension sweep</h3><svg viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{esc(model)} held-out count accuracy across PCA dimensions">'
        f'<title>{esc(model)} PCA dimension sensitivity</title><desc>Non-thinking '
        f'is the dark circle line and native-thinking is the teal square line. The '
        f'dashed reference is ten-class chance.</desc>{"".join(elements)}</svg></figure>'
    )


def _domain_verdict(model_payload: Mapping[str, Any]) -> str:
    non = model_payload["non_thinking"]["metrics"]
    native = model_payload["native_thinking"]["metrics"]
    transfer_fields = (
        "logistic_balanced_accuracy",
        "ncc_balanced_accuracy",
    )
    transfer_wins = [
        float(native["cross_domain_count_mean"][field])
        > float(non["cross_domain_count_mean"][field])
        for field in transfer_fields
    ]
    leakage_wins = [
        float(native["count_residual_domain_leakage"][field])
        < float(non["count_residual_domain_leakage"][field])
        for field in transfer_fields
    ]
    if all(transfer_wins) and all(leakage_wins):
        return (
            "两个 count probe 的跨实体迁移都更高，同时两个 residual-domain "
            "probe 都更低；这一组 exploratory 指标与 native-thinking 在 answer "
            "endpoint 更稳健地保留 count、并保留较少实体类别 nuisance 的解释一致。"
        )
    if any(transfer_wins) and any(leakage_wins):
        return (
            "证据是混合的：至少一个跨实体 count probe 与一个 residual-domain "
            "probe 有利于 native-thinking，但没有在两种 probe 上同时复现。"
        )
    return (
        "当前指标不支持“native-thinking 更过滤实体类别 nuisance”的简单结论；"
        "3-D 视觉分离不能替代 held-out transfer 与 residual-domain probe。"
    )


def _domain_metric_definitions() -> str:
    """Give one plain-language guide, with reproducibility details collapsed."""

    rows = (
        (
            "独立最佳层",
            "这个 model × mode 在哪一层最容易读出 count？",
            "只用 city discovery 200 按 seed 交叉验证；各模式各选各的层。",
            "仅确定读取位置",
        ),
        (
            "三域 pooled count BAcc",
            "在 city 上学到的 count 表征，放到三种实体上一共还能读多准？",
            "冻结后一起测试 city 100 + flower 100 + animal 100。",
            "越高越好；chance 10%",
        ),
        (
            "city → flower/animal count BAcc",
            "不看熟悉的 city，只换成新实体后还能读多准？",
            "分别测试 flower 100、animal 100，再平均两者。",
            "越高越好；chance 10%",
        ),
        (
            "count-residual domain BAcc",
            "先减掉每个 count 的平均表征后，还能不能猜出实体是 city、flower 还是 animal？",
            "在 confirmation 300 上按 seed 留一验证，预测三种实体域。",
            "越低越好；chance 33.3%",
        ),
    )
    return f"""
<h3>如何读下面两张表</h3>
<div class="callout"><strong>统一流程：</strong>只用原始 <code>city discovery 200</code> 选层并训练 count 读出器 → 全部冻结 → 再看 <code>confirmation</code>。两个 count accuracy 问“count 还读不读得出来”（越高越好）；最后一个问“去掉 count 后实体类型还剩多少”（越低越好）。</div>
{table(('表头', '它实际在问什么', '怎么算', '如何判读'), rows)}
<p class="small"><strong>Log / NCC：</strong>同一数据给两种简单读出器。Log 用线性分类边界；NCC 把点分给最近的类别中心。单元格始终按 <code>Log / NCC</code> 排列。<strong>BAcc</strong> 是各类别 recall 的平均；本实验类别平衡，所以数值也等于普通 accuracy。</p>
<details><summary>严格计算细节（复现时再看）</summary><div class="callout"><p>每个 probe 的 <code>StandardScaler</code> 和 whitened PCA 都只在相应训练折拟合，测试数据只做冻结 transform。Log 使用 logistic regression（<code>lbfgs</code>、L2/<code>C=1</code>），NCC 使用 PCA 空间中的欧氏最近 centroid；<code>BAcc=(1/C)Σ recall_c</code>。</p><p>选层时，city discovery 200 按 seed 做 5-fold grouped CV；每层分数是 Log 与 NCC 的跨折均值再取平均，同分取较早层。</p><p>Residual-domain 每折用 9 个 confirmation seeds 估计各 count centroid <code>μ_N</code>，对训练和 held-out seed 都计算 <code>r_i=h_i−μ_{{N_i}}</code>，再用 residual 预测 city/flower/animal，最后平均 10 折。它只移除训练折可估计的加性 count centroid，不保证移除非线性 domain information。</p></div></details>
"""


def _domain_model_block(model: str, payload: Mapping[str, Any]) -> str:
    slug = "qwen" if model.startswith("Qwen") else "gemma"
    metrics_rows = []
    audit_rows = []
    for mode, label in (
        ("non_thinking", "non-thinking"),
        ("native_thinking", "native-thinking"),
    ):
        mode_payload = payload[mode]
        metrics = mode_payload["metrics"]
        audit = mode_payload["audit"]
        overall = metrics["overall_count"]
        transfer = metrics["cross_domain_count_mean"]
        leakage = metrics["count_residual_domain_leakage"]
        metrics_rows.append(
            (
                label,
                f"L{int(mode_payload['selected_layer'])}",
                f"{_pct(overall['logistic_balanced_accuracy'])} / "
                f"{_pct(overall['ncc_balanced_accuracy'])}",
                f"{_pct(transfer['logistic_balanced_accuracy'])} / "
                f"{_pct(transfer['ncc_balanced_accuracy'])}",
                f"{_pct(leakage['logistic_balanced_accuracy'])} / "
                f"{_pct(leakage['ncc_balanced_accuracy'])}",
            )
        )
        exact_denominator = int(audit["transfer_exact_count_denominator"])
        exact_text = (
            f"{int(audit['transfer_exact_count_rows'])}/{exact_denominator}"
            if exact_denominator
            else "n/a"
        )
        audit_rows.append(
            (
                label,
                str(int(audit["transfer_answer_states"])),
                str(int(audit["transfer_running_states"])),
                exact_text,
                str(int(audit.get("transfer_generation_rescue_rows", 0))),
                esc(json.dumps(audit["trace_category_counts"], ensure_ascii=False)),
            )
        )
    return f"""
<article class="appendix-model"><h3>{esc(model)}</h3>
<div class="callout"><strong>判读：</strong>{esc(_domain_verdict(payload))}</div>
<div class="dual-grid domain-grid">
<figure class="geometry-card"><h3>Answer endpoint · non-thinking</h3><div class="controls"><label>Cohort<select id="domain-{slug}-non-cohort"><option value="all">全部 300</option><option value="city">city 100</option><option value="flower">flower 100</option><option value="animal">animal 100</option></select></label><label>Color<select id="domain-{slug}-non-color"><option value="count">gold count</option><option value="domain">entity domain</option></select></label></div><canvas id="domain-{slug}-non" role="img" aria-label="{esc(model)} non-thinking city flower animal answer geometry"></canvas><p class="rotate-hint">drag to rotate · circle city · triangle flower · square animal</p><p class="panel-stats" id="domain-{slug}-non-stats"></p></figure>
<figure class="geometry-card"><h3>Answer endpoint · native-thinking</h3><div class="controls"><label>Cohort<select id="domain-{slug}-native-cohort"><option value="all">全部 300</option><option value="city">city 100</option><option value="flower">flower 100</option><option value="animal">animal 100</option></select></label><label>Color<select id="domain-{slug}-native-color"><option value="count">gold count</option><option value="domain">entity domain</option></select></label></div><canvas id="domain-{slug}-native" role="img" aria-label="{esc(model)} native-thinking city flower animal answer geometry"></canvas><p class="rotate-hint">drag to rotate · each mode uses its own selected layer and PCA basis</p><p class="panel-stats" id="domain-{slug}-native-stats"></p></figure>
</div><div class="domain-legend"><span><i class="domain-city"></i>city</span><span><i class="domain-flower"></i>flower</span><span><i class="domain-animal"></i>animal</span><span>颜色默认表示 N=1…10</span></div>
{table(('mode', '独立最佳层', '三域 pooled count BAcc（PCA16；Log/NCC）', 'city → flower/animal count BAcc（PCA16；Log/NCC）', 'count-residual domain BAcc（PCA16；Log/NCC）↓'), metrics_rows)}
<details><summary>PCA dimension sweep（仅作敏感性诊断）</summary>{_domain_dimension_svg(model, payload)}<p class="small">保留逐维结果供检查，但不再跨维度汇总，也不把它用于上面的判读。</p></details>
<details><summary>Capture 完整性与已保存的 running index</summary><p class="small">Flower 与 animal 每种各 100 trajectories。Answer 图每条只用一个 answer-query state；running states 没有被丢弃：non-thinking 保存 prompt 中每个 active-record span end，native-thinking 保存 parser 实际观察到的每个 item end，允许少数、重复与 ragged path，并保存全部 decoder layers。每个 model × mode 另有顶层 <code>site_index.jsonl</code>：一行对应一个 site，直接记录 shard <code>states.npz</code>、<code>site_states</code> 的 <code>state_axis</code>、token endpoint、边界语义与层注册表，后续无需逐个扫描 shard manifest。若 greedy generation 达到旧 token ceiling，只有 censored row 会在保持 prompt 与 greedy rule 不变时提高 ceiling 重试；数量单列审计。</p>{table(('mode', 'transfer answer states', 'transfer running states', 'final exact / audited', 'ceiling rescues', 'native trace categories'), audit_rows)}</details>
</article>"""


def domain_transfer_appendix(
    analysis_root: Path,
) -> tuple[str, list[Path], dict[str, Any]]:
    payload_path = analysis_root / "report_payload.json"
    manifest_path = analysis_root / "analysis_manifest.json"
    layer_path = analysis_root / "layer_selection_sweep.csv"
    dimension_path = analysis_root / "pca_dimension_sweep.csv"
    capture_audit_path = analysis_root / "capture_audit.json"
    payload = read_json(payload_path)
    if str(payload.get("schema_version")) != "realistic_niah_domain_transfer_geometry_v2_city_discovery":
        raise ValueError(f"Unexpected domain-transfer payload: {payload_path}")
    models = payload.get("models", {})
    if set(models) != set(MODELS):
        raise ValueError(f"Domain-transfer payload has models {sorted(models)}")
    capture_audit = read_json(capture_audit_path)
    if str(capture_audit.get("schema_version")) != (
        "realistic_niah_domain_transfer_capture_audit_v1"
    ):
        raise ValueError(f"Unexpected domain-transfer capture audit: {capture_audit_path}")
    strict_audits = {
        (str(row["model_label"]), str(row["mode"])): row
        for row in capture_audit.get("audits", ())
    }
    expected_audits = {
        (model, mode)
        for model in MODELS
        for mode in ("non_thinking", "native_thinking")
    }
    if set(strict_audits) != expected_audits:
        raise ValueError("Domain-transfer capture audit is not the complete 2 x 2 grid")
    visual: dict[str, Any] = {}
    blocks = []
    for model in MODELS:
        model_payload = models[model]
        if set(model_payload) != {"non_thinking", "native_thinking"}:
            raise ValueError(f"Domain-transfer modes are incomplete for {model}")
        blocks.append(_domain_model_block(model, model_payload))
        visual[model] = {}
        for mode, short in (("non_thinking", "non"), ("native_thinking", "native")):
            value = model_payload[mode]
            strict = strict_audits[(model, mode)]
            if not (
                bool(strict.get("registered_panel_complete"))
                and bool(strict.get("all_states_finite"))
                and bool(strict.get("site_index_audited"))
                and int(strict.get("answer_states", -1))
                == int(value["audit"]["transfer_answer_states"])
                and int(strict.get("running_states", -1))
                == int(value["audit"]["transfer_running_states"])
            ):
                raise ValueError(f"Strict capture audit disagrees for {model}/{mode}")
            points = value["visualization"]["points"]
            if len(points) != 300:
                raise ValueError(f"Domain-transfer visualization is not 300 rows: {model}/{mode}")
            visual[model][short] = {
                "layer": int(value["selected_layer"]),
                "evr": [
                    round(float(component), 7)
                    for component in value["visualization"]["explained_variance_ratio"]
                ],
                "metrics": value["metrics"],
                "points": [
                    [
                        str(point["entity_domain"]),
                        int(point["seed"]),
                        int(point["gold_count"]),
                        str(point["analysis_split"]),
                        round(float(point["x"]), 5),
                        round(float(point["y"]), 5),
                        round(float(point["z"]), 5),
                    ]
                    for point in points
                ],
            }
    return f"""
<section id="appendix-domain-transfer"><h2>Appendix C · Entity-domain transfer：city → flower / animal</h2>
<div class="definitions"><div><h3>配对刺激</h3><p>同一套 V4.4 confirmation 10 counts × 10 seeds 被逐 cell 改写为 city、flower、animal；haystack、active-record score、N 与 seed 保持不变，只改实体词表及对应 prompt。每个 model × mode 共 300 answer trajectories；按要求保留错误输出，count 标签始终是 gold N。</p></div><div><h3>比较位置</h3><p>Non-thinking 取回答前 prompt-final colon；native-thinking 取完整 reasoning 后、数字答案前的 <code>answer_query_v3</code>。两个 mode 独立选层，比较的是各自最清楚的 answer endpoint，而不是同层绝对距离。</p></div><div><h3>3-D 图的基准</h3><p>每个 panel 的 StandardScaler/PCA3 只在该 mode 的 city discovery 200 上拟合，再原样 transform 三个 domain 的 confirmation 300。形状表示 domain、颜色表示 count；左右坐标轴不可直接比较绝对距离。</p></div></div>
<div class="callout warning"><strong>“过滤无关内容”的可检验含义：</strong>不是肉眼更像一条线，而是只在 city 上训练的 count probe 能迁移到 flower/animal，同时在减去 count centroid 后更难解码 city/flower/animal。两项没有同时成立时，不作强 claim。</div>
<details><summary>Flower / animal prompt 如何分别改写</summary><div class="definitions two"><div><h3>Non-thinking</h3><p><code>How many flower-score audit records are in the passage?</code>（animal 版本只把 flower 改为 animal。）随后仍是 <code>Do not explain, reason aloud, quote, or list any records.</code>，并强制整段 response 为 <code>Total:&lt;integer&gt;</code>。保存 prompt-final colon 的 hidden state。</p></div><div><h3>Native-thinking</h3><p>定义句分别写成 <code>A flower-score audit record names one flower…</code> 与 <code>An animal-score audit record names one animal…</code>；问题后要求 concise reasoning、不要 repeating/restarting，并以 <code>Total: &lt;integer&gt;</code> 结束。保存 thinking item ends 与最终数字前的 <code>answer_query_v3</code>。</p></div></div></details>
{_domain_metric_definitions()}
<p class="small"><strong>解释边界：</strong>所有 count 指标都使用 gold N，模型答错的 trajectories 也保留。Residual-domain 解码的是所有与实体域共变的信息，包括实体语义、prompt 词汇和名称 tokenization/长度；因此较低只能表述为“较少可线性解码的 domain-specific nuisance”，不能证明模型抹除了实体语义。本实验是 appendix exploratory evidence，不是主报告的预注册 confirmatory claim。</p>
{''.join(blocks)}
</section>""", [payload_path, manifest_path, layer_path, dimension_path, capture_audit_path], visual


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


def _band_script(band_visual: Mapping[str, Any]) -> str:
    payload = json.dumps(
        band_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const BAND=__BAND__;
const BAND_VIEWS={};
const BAND_COLORS={upper:'#00A88F',lower:'#6750E8'};
const MARKER_COLORS={inline_count:'#A7C957',indexed:'#00A88F',ordinal:'#D6B52C',bullet:'#00A9D8',audit_sentence:'#D94B86',completion_recap:'#6750E8',evidence_sequence:'#E76F51',unresolved:'#8A838E'};
function bandSlug(model){return model.startsWith('Qwen')?'qwen':'gemma'}
function bandIds(model,space){const s=bandSlug(model);return {cohort:document.getElementById('band-'+s+'-cohort'),color:document.getElementById('band-'+s+'-color'),canvas:document.getElementById('band-'+s+'-'+space),stats:document.getElementById('band-'+s+'-'+space+'-stats'),legend:document.getElementById('band-'+s+'-legend')}}
function bandColor(point,mode){if(mode==='band')return BAND_COLORS[point[5]];if(mode==='occurrence')return COLORS[Math.max(0,Math.min(9,point[3]-1))];return MARKER_COLORS[point[4]]||'#8A838E'}
function drawBandMark(c,x,y,r,point,mode){c.beginPath();if(mode==='band'&&point[5]==='lower')c.rect(x-r,y-r,2*r,2*r);else c.arc(x,y,r,0,Math.PI*2);c.fill();c.stroke()}
function renderBandLegend(model){const ids=bandIds(model,'raw'),mode=ids.color.value,points=BAND[model].points,values=mode==='band'?['upper','lower']:mode==='occurrence'?[1,2,3,4,5,6,7,8,9,10]:[...new Set(points.map(p=>p[4]))].sort();ids.legend.replaceChildren();for(const value of values){const span=document.createElement('span'),swatch=document.createElement('i'),label=document.createElement('b'),probe=mode==='band'?['',0,0,0,'',value]:mode==='occurrence'?['',0,0,value,'','upper']:['',0,0,0,value,'upper'];swatch.style.background=bandColor(probe,mode);if(mode==='band'&&value==='lower')swatch.className='square';label.textContent=mode==='occurrence'?'k='+value:String(value);span.append(swatch,label);ids.legend.append(span)}}
function drawBand3D(model,space){
  const ids=bandIds(model,space),payload=BAND[model],cohort=ids.cohort.value,mode=ids.color.value,start=space==='raw'?6:9,points=payload.points.filter(p=>cohort==='all'||p[0]==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length)return;
  const view=BAND_VIEWS[canvas.id]||(BAND_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),rotated=points.map(p=>({p,r:rotate3(p[start],p[start+1],p[start+2],view)})),maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[start]),Math.abs(p[start+1]),Math.abs(p[start+2])]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const centerGroups=new Map();for(const p of points){if(!centerGroups.has(p[5]))centerGroups.set(p[5],[]);centerGroups.get(p[5]).push(p)}const centers=[...centerGroups.entries()].map(([name,ps])=>({name,r:rotate3(ps.reduce((s,p)=>s+p[start],0)/ps.length,ps.reduce((s,p)=>s+p[start+1],0)/ps.length,ps.reduce((s,p)=>s+p[start+2],0)/ps.length,view)}));
  const xy=rotated.map(o=>o.r).concat(axes.map(a=>a[2]),centers.map(o=>o.r),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.38+.48*depth;c.fillStyle=bandColor(o.p,mode);c.strokeStyle=o.p[0]==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p[0]==='confirmation'?1.8:.65;drawBandMark(c,sx(o.r[0]),sy(o.r[1]),2.2+1.1*depth,o.p,mode)}
  c.globalAlpha=1;for(const center of centers){const x=sx(center.r[0]),y=sy(center.r[1]);c.fillStyle=BAND_COLORS[center.name];c.strokeStyle='#20242D';c.lineWidth=2;c.beginPath();if(center.name==='lower')c.rect(x-6,y-6,12,12);else c.arc(x,y,6,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='11px Consolas';c.fillText(center.name,x+9,y-7)}
  const trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[2])).size,upper=points.filter(p=>p[5]==='upper').length,lower=points.length-upper,evr=100*(space==='raw'?payload.raw_evr:payload.centered_evr).reduce((a,b)=>a+b,0);ids.stats.textContent=`${payload.site} · L${payload.layer} · ${cohort==='all'?'full 300-source':'confirmation 100-source'} · ${trajectories} trajectories / ${points.length} states · raw-band upper/lower ${upper}/${lower} · EVR3 ${evr.toFixed(1)}%`;
}
function setupBand(model){const raw=bandIds(model,'raw'),redraw=()=>{drawBand3D(model,'raw');drawBand3D(model,'centered');renderBandLegend(model)};raw.cohort.addEventListener('change',redraw);raw.color.addEventListener('change',redraw);for(const space of ['raw','centered']){const ids=bandIds(model,space);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=BAND_VIEWS[ids.canvas.id]||(BAND_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawBand3D(model,space)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop)}redraw()}
for(const model of Object.keys(BAND))setupBand(model);
window.addEventListener('resize',()=>{for(const model of Object.keys(BAND)){drawBand3D(model,'raw');drawBand3D(model,'centered')}});
""".replace("__BAND__", payload)


def _domain_script(domain_visual: Mapping[str, Any]) -> str:
    if not domain_visual:
        return ""
    payload = json.dumps(
        domain_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const DOMAIN_TRANSFER=__DOMAIN_TRANSFER__;
const DOMAIN_VIEWS={};
const DOMAIN_COLORS={city:'#20242D',flower:'#00A88F',animal:'#E76F51'};
function domainIds(model,mode){const s=model.startsWith('Qwen')?'qwen':'gemma',base='domain-'+s+'-'+mode;return {canvas:document.getElementById(base),cohort:document.getElementById(base+'-cohort'),color:document.getElementById(base+'-color'),stats:document.getElementById(base+'-stats')}}
function domainMark(c,domain,x,y,r){c.beginPath();if(domain==='animal'){c.rect(x-r,y-r,2*r,2*r)}else if(domain==='flower'){c.moveTo(x,y-r*1.18);c.lineTo(x+r*1.08,y+r*.88);c.lineTo(x-r*1.08,y+r*.88);c.closePath()}else{c.arc(x,y,r,0,Math.PI*2)}c.fill();c.stroke()}
function domainPointColor(point,mode){return mode==='domain'?DOMAIN_COLORS[point[0]]:COLORS[Math.max(0,Math.min(9,point[2]-1))]}
function drawDomain3D(model,mode){
  const ids=domainIds(model,mode),payload=DOMAIN_TRANSFER[model][mode],cohort=ids.cohort.value,colorMode=ids.color.value,points=payload.points.filter(p=>cohort==='all'||p[0]===cohort),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length)return;
  const view=DOMAIN_VIEWS[canvas.id]||(DOMAIN_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),rotated=points.map(p=>({p,r:rotate3(p[4],p[5],p[6],view)})),maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[4]),Math.abs(p[5]),Math.abs(p[6])]),1e-6),axisLen=maxAbs*.70,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const centerGroups=new Map();for(const p of points){if(!centerGroups.has(p[2]))centerGroups.set(p[2],[]);centerGroups.get(p[2]).push(p)}const centers=[...centerGroups.entries()].sort((a,b)=>a[0]-b[0]).map(([count,rows])=>[count,rows.reduce((s,p)=>s+p[4],0)/rows.length,rows.reduce((s,p)=>s+p[5],0)/rows.length,rows.reduce((s,p)=>s+p[6],0)/rows.length]);const rotatedCenters=centers.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)}));
  const xy=rotated.map(o=>o.r).concat(rotatedCenters.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  if(colorMode==='count'){c.strokeStyle='#4B5563';c.globalAlpha=.72;c.lineWidth=2;c.beginPath();rotatedCenters.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke()}
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.35+.52*depth;c.fillStyle=domainPointColor(o.p,colorMode);c.strokeStyle='#FFFDF8';c.lineWidth=1.8;domainMark(c,o.p[0],sx(o.r[0]),sy(o.r[1]),2.5+1.1*depth)}
  if(colorMode==='count'){c.globalAlpha=1;for(const o of rotatedCenters){c.fillStyle=COLORS[o.p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.5,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p[0]),sx(o.r[0])+7,sy(o.r[1])-6)}}
  c.globalAlpha=1;const evr=100*payload.evr.reduce((a,b)=>a+b,0),metric=payload.metrics.cross_domain_count_mean,leak=payload.metrics.count_residual_domain_leakage;ids.stats.textContent=`answer query · L${payload.layer} · ${points.length} confirmation trajectories · city-discovery PCA3 EVR ${evr.toFixed(1)}% · city→flower/animal Log/NCC ${(100*metric.logistic_balanced_accuracy).toFixed(1)}%/${(100*metric.ncc_balanced_accuracy).toFixed(1)}% · residual-domain Log/NCC ${(100*leak.logistic_balanced_accuracy).toFixed(1)}%/${(100*leak.ncc_balanced_accuracy).toFixed(1)}%`;
}
function setupDomain(model,mode){const ids=domainIds(model,mode),redraw=()=>drawDomain3D(model,mode);ids.cohort.addEventListener('change',redraw);ids.color.addEventListener('change',redraw);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=DOMAIN_VIEWS[ids.canvas.id]||(DOMAIN_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawDomain3D(model,mode)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);redraw()}
for(const model of Object.keys(DOMAIN_TRANSFER))for(const mode of ['non','native'])setupDomain(model,mode);
window.addEventListener('resize',()=>{for(const model of Object.keys(DOMAIN_TRANSFER))for(const mode of ['non','native'])drawDomain3D(model,mode)});
""".replace("__DOMAIN_TRANSFER__", payload)


def build_html(
    *,
    dual_results: Mapping[str, Mapping[str, Any]],
    dual_visual: Mapping[str, Any],
    token_html: str,
    marker_html: str,
    band_html: str,
    band_visual: Mapping[str, Any],
    band_audits: Mapping[str, Mapping[str, Any]],
    domain_html: str = "",
    domain_visual: Mapping[str, Any] | None = None,
) -> str:
    css = """
:root{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#626A74;--line:#C9C2B6;--indigo:#23165C;--teal:#00A88F;--yellow:#D6B52C}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;overflow-x:hidden;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}nav{position:sticky;top:0;z-index:5;display:flex;gap:18px;padding:10px 22px;background:rgba(243,238,228,.96);border-bottom:1px solid var(--line);overflow-x:auto}nav a{color:var(--indigo);font-size:13px;font-weight:750;text-decoration:none;white-space:nowrap}main{max-width:1480px;margin:auto;padding:38px 28px 80px}header{max-width:1080px;border-bottom:2px solid var(--ink);padding-bottom:28px}.eyebrow{font:700 12px/1.2 Consolas,monospace;letter-spacing:.12em;color:var(--teal)}h1{font-size:44px;line-height:1.08;margin:10px 0 16px;letter-spacing:-.035em}h2{font-size:29px;margin:0 0 12px}h4{color:var(--indigo)}.lead{font-size:18px;color:#404852;max-width:92ch}section{padding:46px 0;border-bottom:1px solid var(--line)}.callout{max-width:1120px;background:var(--surface);border-left:4px solid var(--teal);padding:15px 19px;margin:20px 0}.warning{border-left-color:var(--yellow)}.definitions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}.definitions.two{grid-template-columns:repeat(2,minmax(0,1fr))}.definitions>div,.geometry-card,.appendix-model{min-width:0;background:var(--surface);border:1px solid var(--line);padding:17px}.definitions h3,.geometry-card h3{color:var(--indigo);margin:0 0 8px;font-size:17px}.definitions p,.geometry-card p{font-size:13px;color:var(--muted);margin:0 0 12px}.controls{display:flex;gap:12px;flex-wrap:wrap}.controls label{font-size:12px;font-weight:700;color:var(--muted)}select{display:block;margin-top:4px;border:1px solid var(--line);background:var(--surface);padding:7px 28px 7px 9px;color:var(--ink)}.dual-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:28px}.geometry-card canvas{display:block;width:100%;height:390px;background:#F8F4EC;border:1px solid #DDD5C9;touch-action:none;cursor:grab}.geometry-card canvas:active{cursor:grabbing}.rotate-hint{margin-top:5px;color:#7A7270;font:10px/1.4 Consolas,monospace}.panel-stats{min-height:70px;margin-top:7px;color:var(--muted);font:12px/1.5 Consolas,monospace}.table-scroll{overflow:auto;background:var(--surface);border:1px solid var(--line);margin:16px 0 22px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #DED8CE}th{background:#ECE6DA;color:#303744}.muted,.small{color:var(--muted);font-size:12px}.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0 20px}.metric-figure{min-width:0;margin:0;background:var(--surface);border:1px solid var(--line);padding:13px}.metric-figure h3{margin:0;color:var(--indigo);font-size:17px}.metric-figure svg{display:block;width:100%;height:auto}.metric-gridline{stroke:#D9D2C7;stroke-width:1}.metric-zero{stroke:#756E68;stroke-width:1.5}.metric-tick,.metric-label,.metric-value,.metric-axis-title{fill:#303744;font:12px Consolas,monospace}.metric-tick{fill:var(--muted);font-size:11px}.metric-link{stroke:#8A838E;stroke-width:2}.metric-dot{stroke:#FFFDF8;stroke-width:2}.metric-non,.snr-non{fill:#20242D}.metric-native,.snr-native{fill:#00A88F}.snr-upper{fill:#E76F51}.snr-lower{fill:#6750E8}.metric-legend,.band-dynamic-legend{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:10px 0}.metric-legend span,.band-dynamic-legend span{display:inline-flex;align-items:center;gap:6px}.metric-legend i,.band-dynamic-legend i{display:inline-block;width:11px;height:11px;border-radius:50%;background:#8A838E}.metric-legend .legend-non{background:#20242D}.metric-legend .legend-native{background:#00A88F}.metric-legend .legend-upper{background:#E76F51}.metric-legend .legend-lower{background:#6750E8}.band-dynamic-legend i.square{border-radius:0}.band-dynamic-legend b{font-weight:500}.token-flow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}.token-flow article{min-width:0;background:var(--surface);border:1px solid var(--line);padding:17px}.token-flow h3{color:var(--indigo);margin:0 0 8px;font-size:17px}.token-flow p{font-size:13px;color:var(--muted)}.token-strip{display:flex;gap:5px;align-items:flex-start;flex-wrap:wrap;margin:17px 0 26px}.token-strip span{position:relative;background:#ECE6DA;padding:5px 7px;font:12px Consolas,monospace}.token-strip span[data-pos]::after{content:attr(data-pos);position:absolute;left:50%;top:100%;transform:translateX(-50%);font:9px Consolas,monospace;color:#7A7270}.token-strip .picked{background:#00A88F;color:#FFFDF8}.token-strip b{font:11px Consolas,monospace;color:var(--indigo);padding:5px}.boundary-example{display:flex;align-items:stretch;margin:15px 0;font:12px/1.5 Consolas,monospace}.boundary-example span{background:#ECE6DA;padding:8px 10px}.boundary-example i{display:block;width:4px;background:#E76F51}.boundary-example .answer-token{background:#D9F1EA}.band-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:18px 0}.band-figure{min-width:0;margin:0;background:#FFFDF8;border:1px solid var(--line);padding:12px}.band-figure h4{font-size:15px;color:var(--indigo);margin:0 0 8px}.band-figure canvas{display:block;width:100%;height:380px;background:#F8F4EC;border:1px solid #DDD5C9;touch-action:none;cursor:grab}.band-figure canvas:active{cursor:grabbing}.band-controls{margin-top:15px}.appendix-model{margin:22px 0}.domain-legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:-14px 0 18px}.domain-legend span{display:inline-flex;align-items:center;gap:6px}.domain-legend i{display:inline-block;width:11px;height:11px;background:#20242D}.domain-legend .domain-city{border-radius:50%}.domain-legend .domain-flower{background:#00A88F;clip-path:polygon(50% 0,100% 100%,0 100%)}.domain-legend .domain-animal{background:#E76F51}.domain-dim-figure{margin:18px 0}.domain-dim-line{fill:none;stroke-width:2.3}.domain-line-non{stroke:#20242D;fill:#20242D}.domain-line-native{stroke:#00A88F;fill:#00A88F}.domain-dim-mark{stroke:#FFFDF8;stroke-width:1.4}.domain-chance{stroke:#8A838E;stroke-width:1.3;stroke-dasharray:5 4}.provenance{font:11px/1.6 Consolas,monospace;color:var(--muted)}details{background:var(--surface);border:1px solid var(--line);margin:18px 0}summary{cursor:pointer;padding:12px 15px;font-weight:750;color:var(--indigo)}@media(max-width:1000px){.dual-grid,.definitions,.definitions.two,.metric-grid,.token-flow,.band-grid{grid-template-columns:1fr}}@media(max-width:650px){main{padding:25px 13px 60px}h1{font-size:34px}.geometry-card canvas,.band-figure canvas{height:330px}.metric-value{font-size:10px}}
"""
    script = (
        _dual_script(dual_visual)
        + _band_script(band_visual)
        + _domain_script(domain_visual or {})
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Geometry Comparison</title><style>{css}</style></head><body>
<nav><a href="#scope">口径</a><a href="#tokens">Token 提取</a><a href="#dual">主结果</a><a href="#claims">Confirmation 结论</a><a href="#snr">SNR</a><a href="#appendix-markers">Marker appendix</a><a href="#appendix-bands">分层 appendix</a>{'<a href="#appendix-domain-transfer">实体迁移 appendix</a>' if domain_html else ''}</nav><main>
<header><div class="eyebrow">REALISTIC NIAH · ALL-COUNT GEOMETRY</div><h1>NiaH Geometry Comparison</h1><p class="lead">Running index 与 final count 两组比较都覆盖 N=1…10，并可在完整 300 trajectories 与 confirmation 100 trajectories 之间切换。Running index 固定比较 prompt <code>span_end</code> 与 thinking-trace <code>item_end</code>；两个模式只各自选择最佳 decoder layer。</p></header>
<section id="scope"><h2>严格比较口径</h2><div class="definitions"><div><h3>Full 300</h3><p>10 个 gold N × 30 seeds。它是 descriptive geometry view；PCA3 仍只由 discovery 200 拟合，避免 confirmation 反向选显示 basis。</p></div><div><h3>Confirmation 100</h3><p>10 个 gold N × 10 held-out seeds。主表的 Logistic、nearest-centroid 与 SNR 都是 discovery-frozen 后在这里评价。</p></div><div><h3>Native running 的 ragged rule</h3><p>每条 trace 只贡献 parser 实际观察到的 1…M。数到 8 就贡献八个 states；不按 gold N 或最终 Total 补到 9/10。</p></div></div></section>
{token_html}
{dual_endpoint_section(dual_results, dual_visual)}
{empirical_claims(dual_results)}
{snr_section(dual_results, band_audits)}
{marker_html}{band_html}{domain_html}
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
    domain_transfer_root: Path | None = None,
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
    token_html, token_inputs = token_extraction_section(
        native_running_root.resolve(), native_final_root.resolve()
    )
    band_html, band_inputs, band_visual, band_audits = band_appendix(
        band_root.resolve()
    )
    domain_html = ""
    domain_inputs: list[Path] = []
    domain_visual: dict[str, Any] = {}
    if domain_transfer_root is not None:
        domain_html, domain_inputs, domain_visual = domain_transfer_appendix(
            domain_transfer_root.resolve()
        )
    document = build_html(
        dual_results=dual_results,
        dual_visual=dual_visual,
        token_html=token_html,
        marker_html=marker_appendix(parser_rows),
        band_html=band_html,
        band_visual=band_visual,
        band_audits=band_audits,
        domain_html=domain_html,
        domain_visual=domain_visual,
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    inputs = sorted(
        set(
            dual_inputs
            + visual_inputs
            + band_inputs
            + token_inputs
            + domain_inputs
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
        "native_band_snr": "bands and PCA16 frozen on discovery; per-band confirmation SNR requires at least two states per retained k",
        "entity_domain_transfer": (
            "city-discovery-200 layer selection/probe fitting and frozen "
            "city/flower/animal confirmation-100 evaluation; city-anchored PCA3; "
            "running endpoints archived at all layers"
            if domain_transfer_root is not None
            else "not included"
        ),
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
    parser.add_argument("--domain-transfer-root", type=Path)
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
        domain_transfer_root=args.domain_transfer_root,
        output=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
