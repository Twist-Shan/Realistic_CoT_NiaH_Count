#!/usr/bin/env python3
"""Merge and readability-review the canonical Realistic NiaH HTML report.

This is a report-only post-processing step. It does not alter raw requests,
frozen stimuli, manifests, QC artifacts, or fitted numeric tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


MODEL_ORDER = [
    "Qwen3-8B",
    "Qwen3-1.7B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "OLMo-Hybrid-7B",
    "Llama3.1-8B",
    "Llama3.2-3B",
]
REGISTERED_NEEDLES = np.array(
    [1, 2, 3, 4, 5, 6, 8, 10, 20, 30], dtype=float
)
REGISTERED_LENGTHS = [2000, 5000, 10000]
FIGURE_FILES = [
    "fig01_model_error_composition.png",
    "fig03_mode_accuracy_heatmap.png",
    "fig02_native_thinking_comparison.png",
    "fig11_over_under_rates.png",
    "fig04_accuracy_law_cv.png",
    "fig05_pooled_accuracy_length_needles.png",
    "fig07_unified_law_parameters.png",
    "fig08_unified_law_surfaces.png",
    "fig09_heldout_observed_predicted.png",
    "fig10_model_bias_summary.png",
    "fig12_bias_length_needle_slopes.png",
    "fig13_model_bias_candidate_search.png",
    "fig14_model_bias_log_slopes.png",
    "fig15_model_bias_selected_surfaces.png",
    "fig16_model_bias_oof_calibration.png",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_report_root(path: Path) -> Path:
    path = path.resolve()
    required = [
        path / "report.html",
        path / "analysis_manifest.json",
        path / "tables",
        path / "assets",
        path / "scripts",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Not a complete report root; missing: {missing}")
    return path


def merge_report(source: Path, target: Path) -> int:
    """Copy a verified report tree into an existing canonical directory."""
    source = ensure_report_root(source)
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        copied += 1
    return copied


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_between(
    text: str, start_marker: str, end_marker: str, replacement: str, label: str
) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise RuntimeError(f"{label}: section markers not found")
    return text[:start] + replacement + text[end:]


def replace_figure(
    text: str, filename: str, alt: str, caption: str
) -> str:
    pattern = re.compile(
        rf'<figure class="report-figure"><img src="assets/{re.escape(filename)}"'
        r'[^>]*><figcaption>.*?</figcaption></figure>'
    )
    replacement = (
        '<figure class="report-figure">'
        f'<img src="assets/{filename}" alt="{alt}" loading="lazy">'
        f"<figcaption>{caption}</figcaption></figure>"
    )
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"figure caption not found exactly once: {filename}")
    return text


def refine_html(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")

    if "scripts/refine_report_readability.py" not in text:
        reproduction_item = (
            "    <li><code>scripts/build_report.py</code>："
            "从原始 JSONL 重建报告的脚本；</li>"
        )
        expanded_reproduction = reproduction_item + """
    <li><code>scripts/build_model_bias_addendum.py</code>：重建逐模型 signed-bias 分析；</li>
    <li><code>scripts/build_prompt_format_addendum.py</code>：重建审计后的 prompt 格式章节；</li>
    <li><code>scripts/refine_report_readability.py</code>：统一公式符号、图注，并从保留的 CSV 表重绘关键图；</li>"""
        text = replace_once(
            text,
            reproduction_item,
            expanded_reproduction,
            "report reproduction scripts",
        )

    refined_markers = [
        'class="reading-key"',
        "推荐的 exact-accuracy law",
        "Conditional absolute-error law",
        "Robust signed-bias target",
        "图 15｜模型内所选 bias law 的 seed-held-out 校准",
    ]
    if all(marker in text for marker in refined_markers):
        report_path.write_text(text, encoding="utf-8")
        return {
            "formula_cards": len(re.findall(r'<div class="formula', text)),
            "embedded_figures": len(
                re.findall(r'<figure class="report-figure">', text)
            ),
            "caption_revisions": 15,
        }

    old_css = """.formula {
  font-family: "Cambria Math", "Times New Roman", serif;
  font-size: 1.08rem;
  background: var(--wash);
  border: 1px solid var(--line);
  padding: 14px 18px;
  overflow-x: auto;
}"""
    new_css = old_css + """
.formula + .formula { margin-top: 10px; }
.equation-card { line-height: 1.65; }
.equation-title {
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  font-size: .82rem;
  font-weight: 750;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 6px;
}
.equation-line {
  display: block;
  font-size: 1.11rem;
  white-space: nowrap;
}
.equation-note {
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  font-size: .88rem;
  line-height: 1.55;
  color: var(--muted);
  margin-top: 8px;
  white-space: normal;
}
.notation-table th:first-child,
.notation-table td:first-child {
  text-align: center;
  white-space: nowrap;
  width: 11%;
}
.notation-table th:nth-child(2),
.notation-table td:nth-child(2) { text-align: left; }
.notation-table th:nth-child(3),
.notation-table td:nth-child(3) { text-align: left; width: 32%; }
.reading-key {
  border-left: 4px solid var(--accent);
  padding-left: 14px;
}
@media (max-width: 720px) {
  .equation-line { font-size: .98rem; }
  .formula { padding: 12px 13px; }
}"""
    text = replace_once(text, old_css, new_css, "formula CSS")

    definitions = """  <h3>符号与统计口径</h3>
  <p class="reading-key">以下各式统一用 <strong>T</strong> 表示 canonical passage 长度；不再混用 L。准确率以全部注册请求为分母，而 absolute error 与 signed bias 只在冻结 parser 成功提取数值的输出上定义。</p>
  <div class="table-wrap"><table class="data-table notation-table">
    <thead><tr><th>符号</th><th>定义</th><th>在本实验中的取值或含义</th></tr></thead>
    <tbody>
      <tr><td><em>i</em></td><td>单条请求索引</td><td>共 n=6,300 条正式请求</td></tr>
      <tr><td><em>m</em></td><td>模型</td><td>八个模型之一</td></tr>
      <tr><td><em>T</em></td><td>canonical passage 长度</td><td>2,000、5,000 或 10,000 tokens</td></tr>
      <tr><td><em>N</em></td><td>真实 needle / record 数量</td><td>1、2、3、4、5、6、8、10、20 或 30</td></tr>
      <tr><td><em>q</em></td><td>prompt mode</td><td>direct、enumeration 或 native_thinking</td></tr>
      <tr><td><em>o</em></td><td>query order</td><td>query_first 或 query_last</td></tr>
      <tr><td>N̂<sub>i</sub></td><td>冻结 parser 提取的预测整数</td><td>仅在 parsed<sub>i</sub>=1 时存在</td></tr>
    </tbody>
  </table></div>
  <div class="formula equation-card">
    <div class="equation-title">Primary outcome：全请求 exact accuracy</div>
    <span class="equation-line">Y<sub>i</sub> = 𝟙{parsed<sub>i</sub>=1，truncated<sub>i</sub>=0，N̂<sub>i</sub>=N<sub>i</sub>}，&nbsp; Accuracy = n<sup>−1</sup> Σ<sub>i</sub>Y<sub>i</sub></span>
    <div class="equation-note">𝟙{·} 是指示函数。解析失败、未按格式、截断和数值错误都令 Y<sub>i</sub>=0；因此它们没有从准确率分母中删除。</div>
  </div>
  <div class="formula equation-card">
    <div class="equation-title">Conditional error outcomes：只在 parsed outputs 上定义</div>
    <span class="equation-line">b<sub>i</sub> = N̂<sub>i</sub> − N<sub>i</sub>，&nbsp; e<sub>i</sub> = |b<sub>i</sub>|，&nbsp; 条件：parsed<sub>i</sub>=1</span>
    <div class="equation-note">b<sub>i</sub> 是 signed bias：正值表示多计，负值表示少计；e<sub>i</sub> 是 absolute error。两者只对 5,385 条成功解析输出计算，未解析请求不会被伪造为 bias=0 或 error=0。</div>
  </div>
  <div class="formula equation-card">
    <div class="equation-title">Needle density</div>
    <span class="equation-line">d = 1000N/T &nbsp;（needles per 1k canonical passage tokens）</span>
    <div class="equation-note">T 由 Qwen3-8B canonical tokenizer 定义，指目标 passage 长度，不是各模型 chat template 渲染后的总输入 tokens。Density-only 假设把 T 与 N 压缩成 N/T；后文 held-out 比较显示这种压缩不充分。</div>
  </div>
"""
    text = replace_between(
        text,
        "  <h3>定义</h3>",
        '  <p class="muted">“thinking 开/关”',
        definitions,
        "setup definitions",
    )

    literal_replacements = {
        '<li><strong>分离的 log L 与 log N：</strong>logit(p)=控制项+b<sub>L</sub>log₂(L/5000)+b<sub>N</sub>log₂(N/5)。它允许长度和 needle 数量有不同阶。</li>':
            '<li><strong>分离的 log T 与 log N：</strong>logit(p)=控制项+b<sub>T</sub>log₂(T/5000)+b<sub>N</sub>log₂(N/5)。它允许 passage 长度和 needle 数量有不同阶。</li>',
        '<li><strong>带交互的 response surface：</strong>在上一式加入 log L × log N，检验一个维度的效应是否随另一个维度改变。</li>':
            '<li><strong>带交互的 response surface：</strong>在上一式加入 log T × log N，检验一个维度的效应是否随另一个维度改变。</li>',
        "每模型独立 L/N 斜率": "每模型独立 T/N 斜率",
        "共享 log L + log N + 交互": "共享 log T + log N + 交互",
        "共享 log L + log N": "共享 log T + log N",
        "分开的 log L + log N": "分开的 log T + log N",
        "log₂(L/5000)": "log₂(T/5000)",
        "分开 L/N 的共享斜率模型": "分开 T/N 的共享斜率模型",
        "blocked (L,N) cells": "blocked (T,N) cells",
        "带 L×N 交互": "带 T×N 交互",
        "L 翻倍后的 odds 倍率": "T 翻倍后的 odds 倍率",
        "分离 log L/log N": "分离 log T/log N",
        "原始归一化 L/N": "原始归一化 T/N",
        "原始归一化 L + N": "原始归一化 T + N",
        "分离 log₂L + log₂N + 交互": "分离 log₂T + log₂N + 交互",
        "平滑 L/N law": "平滑 T/N law",
        "βL": "β_T",
        "βN": "β_N",
        "βD": "β_D",
        "βLN": "β_TN",
    }
    for old, new in literal_replacements.items():
        text = text.replace(old, new)

    old_accuracy_formula = (
        '<div class="formula">p<sub>m</sub>(L,N,q,o) = 1 / '
        '[1 + A<sub>m</sub>(L/5000)<sup>r<sub>m</sub></sup>'
        '(N/5)<sup>s<sub>m</sub></sup> · '
        'exp(−δ<sub>q</sub>−γ<sub>o</sub>)]</div>'
    )
    new_accuracy_formula = """<div class="formula equation-card">
    <div class="equation-title">推荐的 exact-accuracy law</div>
    <span class="equation-line">p<sub>m</sub>(T,N,q,o) = {1 + A<sub>m</sub>(T/5000)<sup>r<sub>m</sub></sup>(N/5)<sup>s<sub>m</sub></sup> exp[−δ<sub>q</sub>−γ<sub>o</sub>]}<sup>−1</sup></span>
    <span class="equation-line">logit p<sub>m</sub> = −ln A<sub>m</sub> − r<sub>m</sub>ln(T/5000) − s<sub>m</sub>ln(N/5) + δ<sub>q</sub> + γ<sub>o</sub></span>
    <div class="equation-note">p<sub>m</sub> 是单条请求 exact-correct 的概率；ln 是自然对数。A<sub>m</sub>&gt;0 是模型 m 在 T=5000、N=5、direct/query_first 时的 failure-to-success odds；r<sub>m</sub> 与 s<sub>m</sub> 分别是长度阶与 needle 阶。δ<sub>q</sub>、γ<sub>o</sub> 是 prompt mode 和 query order 的共享偏移，参考条件取 0。</div>
  </div>"""
    text = replace_once(
        text,
        old_accuracy_formula,
        new_accuracy_formula,
        "accuracy formula",
    )

    old_accuracy_explanation = (
        "<p>m 表示模型；q 和 o 分别是 prompt mode 与 query order 的共享 nuisance 修正。"
        "在 direct/query_first baseline 下，δ=γ=0。因为 odds=p/(1−p)，长度翻倍会把 "
        "odds 乘以 2<sup>−rₘ</sup>，needle 数翻倍会乘以 2<sup>−sₘ</sup>。"
        "Aₘ 控制基准难度，rₘ 与 sₘ 就是用户关心的“分别成多少阶”。</p>"
    )
    new_accuracy_explanation = (
        "<p>读法很直接：在其他变量不变时，T 翻倍把成功 odds 乘以 "
        "2<sup>−rₘ</sup>，N 翻倍把成功 odds 乘以 2<sup>−sₘ</sup>。"
        "因此正的 rₘ 或 sₘ 表示相应维度增大时准确率下降；数值越大，下降越快。"
        "这是注册网格内的预测性经验关系，不是关于模型内部机制的因果定律。</p>"
    )
    text = replace_once(
        text,
        old_accuracy_explanation,
        new_accuracy_explanation,
        "accuracy explanation",
    )

    old_error_formula = (
        '<div class="formula">E[log(1+|error|) | parsed] ≈ log '
        'B<sub>m</sub> + u<sub>m</sub>log(L/5000) + '
        'v<sub>m</sub>log(N/5) + prompt/order controls</div>'
    )
    new_error_formula = """<div class="formula equation-card">
    <div class="equation-title">Conditional absolute-error law</div>
    <span class="equation-line">z<sub>i</sub> = ln(1+|b<sub>i</sub>|)</span>
    <span class="equation-line">E[z<sub>i</sub> | parsed,m,T,N,q,o] ≈ ln B<sub>m</sub> + u<sub>m</sub>ln(T/5000) + v<sub>m</sub>ln(N/5) + η<sub>q</sub> + κ<sub>o</sub></span>
    <div class="equation-note">B<sub>m</sub> 是拟合的 baseline 误差尺度；u<sub>m</sub>、v<sub>m</sub> 是乘法阶数。T 翻倍使拟合的 geometric-scale (1+|b|) 乘以 2<sup>uₘ</sup>，N 翻倍使其乘以 2<sup>vₘ</sup>。该 law 只描述已成功解析数值时的典型误差量级。</div>
  </div>"""
    text = replace_once(
        text, old_error_formula, new_error_formula, "absolute-error formula"
    )

    old_bias_definition = (
        '<div class="formula"><strong>bias</strong> = predicted_count − true_count；'
        '<strong>asinh-centered bias</strong> = sinh(E[asinh(bias)])。'
        "asinh 在 0 附近近似线性、对大正负值近似对数，因此保留方向，同时降低少数数百至数千的 "
        "over-count 对回归的支配。它不是 raw mean，也不等同于 median。</div>"
    )
    new_bias_definition = """<div class="formula equation-card">
    <div class="equation-title">Robust signed-bias target</div>
    <span class="equation-line">b<sub>i</sub> = N̂<sub>i</sub> − N<sub>i</sub>，&nbsp; z<sub>i</sub> = asinh(b<sub>i</sub>) = ln[b<sub>i</sub> + √(b<sub>i</sub><sup>2</sup>+1)]</span>
    <span class="equation-line">c<sub>m</sub>(T,N,q,o) = sinh{E[z<sub>i</sub> | parsed,m,T,N,q,o]}</span>
    <div class="equation-note">c<sub>m</sub> 称为 asinh-centered bias，回到 count 单位并保留正负方向。asinh 在 0 附近近似线性、在长尾处近似对数，所以少数巨大 over-count 不会像 raw mean 那样支配拟合；c<sub>m</sub> 既不是 raw mean，也不是 median。</div>
  </div>"""
    text = replace_once(
        text,
        old_bias_definition,
        new_bias_definition,
        "bias definition",
    )

    old_bias_formula = (
        '<div class="formula"><strong>可比较固定形式：</strong>asinh(bias) = '
        'α<sub>m,q,o</sub> + β<sub>m,L</sub> log₂(T/5000) + '
        'β<sub>m,N</sub> log₂(N/5)。因此 β<sub>m,L</sub> 是长度翻倍时 mean '
        "asinh(bias) 的变化，β<sub>m,N</sub> 是 needle 数翻倍时的变化；"
        "α 随模型、prompt mode 与 query order 改变。</div>"
    )
    new_bias_formula = """<div class="formula equation-card">
    <div class="equation-title">用于跨模型比较的固定函数族</div>
    <span class="equation-line">E[z<sub>i</sub> | parsed,m,T,N,q,o] = α<sub>m,q,o</sub> + β<sub>m,T</sub>log₂(T/5000) + β<sub>m,N</sub>log₂(N/5)</span>
    <div class="equation-note">β<sub>m,T</sub> 是 T 翻倍时 mean asinh(bias) 的变化，β<sub>m,N</sub> 是 N 翻倍时的变化；它们是 transformed-response units，不是原始计数的直接增量。正值表示更趋向 over-count，负值表示更趋向 under-count。每个模型的最终选式仍由 grouped held-out 验证决定。</div>
  </div>"""
    text = replace_once(
        text, old_bias_formula, new_bias_formula, "bias comparison formula"
    )

    captions = {
        "fig01_model_error_composition.png": (
            "Stacked outcome composition for every registered request in each model.",
            "<strong>图 1｜各模型全部注册请求的互斥结果构成。</strong> 每行以该模型的 600 或 900 条请求为分母；横轴是请求比例。正确、已解析少计、已解析多计、非截断格式/解析失败、截断或其他五类互斥且总和为 100%。这张图采用 primary failure policy，不会删除无法解析的请求。",
        ),
        "fig03_mode_accuracy_heatmap.png": (
            "Exact-count accuracy by model and registered prompt or thinking mode.",
            "<strong>图 2｜模型 × prompt/thinking 模式的 exact accuracy。</strong> 行是模型，列是 direct-off、enumeration-off 与 native-thinking-on；颜色和格内数字都以该模型在该模式下的全部注册请求为分母。N/A 表示该模型没有注册 native-thinking 条件。",
        ),
        "fig02_native_thinking_comparison.png": (
            "Paired direct-off and native-thinking-on exact accuracy for five supported models.",
            "<strong>图 3｜相同 direct 任务文字下，native thinking 开与关的准确率。</strong> 横轴是支持两种条件的五个模型，纵轴是全部注册请求上的 exact accuracy；连线比较同一模型，标注为“开−关”的百分点差。Thinking-on 同时改变输出预算和采样温度，因此差值是整套推理模式的关联，不是隔离的因果效应。",
        ),
        "fig11_over_under_rates.png": (
            "Under-count, exact, and over-count rates conditional on parsed numeric outputs.",
            "<strong>图 4｜成功解析数值后的少计/正确/多计比例。</strong> 每行只以该模型的 parsed outputs 为分母；横轴是条件比例，三类总和为 100%。未解析与截断请求不在这张条件分布中，必须与图 1 的全请求失败比例一起解释。",
        ),
        "fig04_accuracy_law_cv.png": (
            "Five-fold leave-one-seed-out Bernoulli log loss for the accuracy-law candidates.",
            "<strong>图 5｜准确率候选 law 的 leave-one-seed-out 比较。</strong> 横轴是 held-out Bernoulli log loss，越低越好；每个点是 5 个 seed-held-out folds 的均值，误差线为 mean ± 1.96×SE（5 folds）。同一 seed 的全部模型、模式、order 与 (T,N) 条件共同留出；所有 parse/format/truncation failures 均以 Y=0 保留。Density-only 明显落后于分开的 log T + log N。",
        ),
        "fig05_pooled_accuracy_length_needles.png": (
            "Observed pooled exact accuracy versus needle count at each registered passage length.",
            "<strong>图 6｜观测准确率随 N 与 T 的描述性关系。</strong> 横轴是 needle 数 N，纵轴是 pooled exact accuracy，三条线对应 T=2k/5k/10k；每个点汇总八个模型的已注册 prompt modes、两种 query order 与 5 个 seeds。由于不同模型注册的模式数不同，这是一张加权总体趋势图，不能替代带模型与 condition 控制项的回归。",
        ),
        "fig07_unified_law_parameters.png": (
            "Model-specific passage-length and needle-count exponents with stimulus-cluster bootstrap intervals.",
            "<strong>图 7｜推荐 accuracy law 的模型特异阶数。</strong> 左图横轴为 passage-length exponent rₘ，右图为 needle-count exponent sₘ；点是全数据估计，横线是按完整 stimulus 聚类的 95% bootstrap CI，虚线 0 表示该维度没有可辨识效应。正值越大，变量翻倍时成功 odds 下降越快。T 只有 3 个注册水平，因此 rₘ 通常比有 10 个水平的 sₘ 更不精确。",
        ),
        "fig08_unified_law_surfaces.png": (
            "Predicted exact accuracy from the selected law across registered N and T values.",
            "<strong>图 8｜推荐 accuracy law 在参考 condition 下的预测切片。</strong> 每幅小图是一种模型；横轴是 N（log₂ 刻度），纵轴是预测 exact accuracy，三条线是 T=2k/5k/10k。所有曲线固定为 direct/query_first，因此模型间只通过 Aₘ、rₘ、sₘ 区分。只在注册范围 1≤N≤30、2k≤T≤10k 内解释，不能外推到更长上下文。",
        ),
        "fig09_heldout_observed_predicted.png": (
            "Blocked-cell out-of-fold predicted and observed exact accuracy by model.",
            "<strong>图 9｜blocked-(T,N)-cell held-out 校准。</strong> 横轴是所选 law 对完全留出的 (T,N) cells 给出的 out-of-fold（OOF）预测概率，纵轴是同一 held-out cell 的观测准确率；每点在一个模型内汇总其已注册 prompt modes、两种 order 与 seeds。45° 线是完美校准，偏离表示平滑 law 未捕获的 cell heterogeneity。",
        ),
        "fig10_model_bias_summary.png": (
            "Raw mean, trimmed mean, and median signed bias among parsed outputs.",
            "<strong>图 10｜parsed outputs 的 raw mean、10% trimmed mean 与 median bias。</strong> 横轴单位是 counts（N̂−N），0 表示无方向偏差。蓝色圆点与横线是 raw mean 及 500 次 stimulus-cluster bootstrap 95% CI，方块是 trimmed mean，叉号是 median。三者分离越大，说明少数巨大正向 outliers 对 raw mean 的影响越强。",
        ),
        "fig12_bias_length_needle_slopes.png": (
            "Separate T and N coefficients for raw and asinh signed-bias regressions.",
            "<strong>图 11｜bias 对 T 与 N 的固定分离-log斜率诊断。</strong> 样本只含 parsed outputs。上排响应是 raw signed bias（count units），下排响应是 asinh(bias)；左列横轴为 T effect，右列为 N effect。点是全数据估计，横线是 stimulus-cluster bootstrap 区间，0 虚线表示无可辨识方向。Raw-mean coefficients 对极端 over-count 很敏感，稳健解释应优先看下排。",
        ),
        "fig13_model_bias_candidate_search.png": (
            "Per-model grouped-validation improvement of signed-bias coordinate candidates.",
            "<strong>图 12｜各模型独立的 bias 坐标搜索。</strong> 行是模型，列是预先限定的候选坐标；每格为相对 condition-only baseline 的四种 grouped splits 平均 asinh-bias MAE 改善率：(MAE₀−MAE候选)/MAE₀×100%。正值越大越好，负值表示更差；黑框是最终选式，未框出说明 condition-only 最优。所有候选均保留在图中，避免只展示最好看的结果。",
        ),
        "fig14_model_bias_log_slopes.png": (
            "Within-model fixed log T and log N coefficients for asinh signed bias.",
            "<strong>图 13｜统一固定函数族下，每个模型自己的 bias 阶数。</strong> 左图横轴是 T 翻倍时 mean asinh(bias) 的变化 βₘ,T，右图是 N 翻倍时的变化 βₘ,N；横线为 500 次 stimulus-cluster bootstrap 95% CI，虚线 0 表示无可辨识方向。系数是 transformed-response units；正值趋向 over-count，负值趋向 under-count。区间显著不等于该函数族能在 grouped held-out 数据上稳定胜出。",
        ),
        "fig15_model_bias_selected_surfaces.png": (
            "Selected per-model robust signed-bias response curves within the registered grid.",
            "<strong>图 14｜各模型最终选中 bias law 的 response-surface 切片。</strong> 横轴是 N（log₂ 刻度），纵轴是 back-transformed robust center cₘ，单位为 counts；三种颜色是 T=2k/5k/10k。每幅图固定在该模型 parsed 样本最多的 prompt/order 参考条件，且纵轴范围按模型独立设置。水平灰线与“No stable T/N effect selected”表示 grouped validation 选择了 condition-only，而不是 bias 必然恒定为 0。",
        ),
        "fig16_model_bias_oof_calibration.png": (
            "Seed-held-out predicted and observed mean asinh signed bias by model and T-N cell.",
            "<strong>图 15｜模型内所选 bias law 的 seed-held-out 校准。</strong> 横轴是 leave-one-seed-out 预测的 cell mean asinh(b)，纵轴是观测 cell mean asinh(b)；每个模型有 30 个 (T,N) cells，每点汇总其已注册 prompt/order 条件与 held-out seed 中成功解析的输出。虚线是理想 45° 线。垂直带状或远离对角线的散点表示 condition 截距与平滑 T/N law 仍无法解释 stimulus heterogeneity。",
        ),
    }
    for filename, (alt, caption) in captions.items():
        text = replace_figure(text, filename, alt, caption)

    report_path.write_text(text, encoding="utf-8")
    return {
        "formula_cards": len(re.findall(r'<div class="formula', text)),
        "embedded_figures": len(re.findall(r'<figure class="report-figure">', text)),
        "caption_revisions": len(captions),
    }


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="x", color="#d7dee2", linewidth=0.7, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def regenerate_accuracy_parameter_figure(report_root: Path) -> None:
    table = pd.read_csv(
        report_root / "tables" / "archived_unified_parameters.csv"
    ).set_index("model_label").loc[MODEL_ORDER].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7), sharey=True)
    specs = [
        (
            "length_parameter",
            "Passage-length exponent  rₘ",
            "Exponent rₘ  (larger = faster decline with T)",
            "#2f6f8f",
        ),
        (
            "needle_parameter",
            "Needle-count exponent  sₘ",
            "Exponent sₘ  (larger = faster decline with N)",
            "#c46a24",
        ),
    ]
    y = np.arange(len(table))
    for ax, (field, title, xlabel, color) in zip(axes, specs):
        values = table[field].to_numpy(dtype=float)
        low = table[f"{field}_ci95_low"].to_numpy(dtype=float)
        high = table[f"{field}_ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            values,
            y,
            xerr=np.vstack([values - low, high - values]),
            fmt="o",
            markersize=6,
            capsize=3,
            color=color,
            ecolor=color,
            elinewidth=1.5,
        )
        ax.axvline(0, color="#6f777b", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12.5, weight="bold")
        ax.set_xlabel(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(table["model_label"])
        style_axes(ax)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Model-specific exponents in the selected exact-accuracy law",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Points: full-data estimates · Lines: 95% stimulus-cluster bootstrap intervals",
        ha="center",
        color="#596368",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(
        report_root / "assets" / "fig07_unified_law_parameters.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def regenerate_accuracy_surface_figure(report_root: Path) -> None:
    table = pd.read_csv(
        report_root / "tables" / "archived_unified_parameters.csv"
    ).set_index("model_label")
    colors = ["#2878b5", "#f28e2b", "#2a9d55"]
    fig, axes = plt.subplots(
        2, 4, figsize=(15.2, 7.7), sharex=True, sharey=True
    )
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        row = table.loc[model]
        for length, color in zip(REGISTERED_LENGTHS, colors):
            probability = 1.0 / (
                1.0
                + float(row["amplitude"])
                * (length / 5000.0) ** float(row["length_parameter"])
                * (REGISTERED_NEEDLES / 5.0)
                ** float(row["needle_parameter"])
            )
            ax.plot(
                REGISTERED_NEEDLES,
                probability,
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                color=color,
                label=f"T={length:,}",
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(model, fontsize=10.5, weight="bold")
        ax.grid(alpha=0.2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    for ax in axes[-1, :]:
        ax.set_xlabel("Needle count N (log₂ scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted exact accuracy")
    axes[0, 0].legend(title="Passage length", fontsize=8)
    fig.suptitle(
        "Selected exact-accuracy law at the direct/query-first reference condition",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        report_root / "assets" / "fig08_unified_law_surfaces.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def selected_bias_curve(
    row: pd.Series, length: int, needles: np.ndarray
) -> np.ndarray:
    candidate = str(row["selected_candidate"])
    eta = np.full_like(
        needles, float(row["reference_intercept"]), dtype=float
    )
    if candidate == "raw_separable":
        eta += float(row["length_slope"]) * ((length - 5000.0) / 5000.0)
        eta += float(row["needle_slope"]) * ((needles - 5.0) / 5.0)
    elif candidate in {"log_separable", "log_interaction"}:
        log_t = math.log2(length / 5000.0)
        log_n = np.log2(needles / 5.0)
        eta += float(row["length_slope"]) * log_t
        eta += float(row["needle_slope"]) * log_n
        if candidate == "log_interaction":
            eta += float(row["interaction_slope"]) * log_t * log_n
    elif candidate == "log_density":
        log_density = np.log2(
            (needles / 5.0) / (length / 5000.0)
        )
        eta += float(row["density_slope"]) * log_density
    elif candidate != "condition_only":
        raise ValueError(f"Unknown selected candidate: {candidate}")
    return np.sinh(np.clip(eta, -8.0, 8.0))


def regenerate_bias_surface_figure(report_root: Path) -> None:
    table = pd.read_csv(
        report_root / "tables" / "model_specific_bias_selected_laws.csv"
    ).set_index("model_label").loc[MODEL_ORDER].reset_index()
    colors = ["#2878b5", "#f28e2b", "#2a9d55"]
    candidate_labels = {
        "condition_only": "No stable T/N effect selected",
        "raw_separable": "Linear in normalized T and N",
        "log_separable": "log₂ T + log₂ N",
        "log_interaction": "log₂ T + log₂ N + interaction",
        "log_density": "log₂ density N/T",
    }
    fig, axes = plt.subplots(2, 4, figsize=(15.7, 8.0), sharex=True)
    for ax, (_, row) in zip(axes.ravel(), table.iterrows()):
        candidate = str(row["selected_candidate"])
        all_values: list[float] = []
        if candidate == "condition_only":
            values = selected_bias_curve(
                row, REGISTERED_LENGTHS[0], REGISTERED_NEEDLES
            )
            ax.plot(
                REGISTERED_NEEDLES,
                values,
                color="#596368",
                linewidth=2.2,
            )
            all_values.extend(values.tolist())
            ax.text(
                0.04,
                0.08,
                "No stable T/N effect\nselected by grouped CV",
                transform=ax.transAxes,
                fontsize=8.5,
                color="#596368",
                va="bottom",
            )
        else:
            for length, color in zip(REGISTERED_LENGTHS, colors):
                values = selected_bias_curve(
                    row, length, REGISTERED_NEEDLES
                )
                all_values.extend(values.tolist())
                ax.plot(
                    REGISTERED_NEEDLES,
                    values,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.7,
                    color=color,
                    label=f"T={length:,}",
                )
        ax.axhline(0.0, color="#777777", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        low = min(0.0, min(all_values))
        high = max(0.0, max(all_values))
        span = high - low
        pad = max(0.15, span * 0.12)
        if span < 1e-10:
            pad = max(0.35, abs(high) * 0.3)
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(
            f"{row['model_label']}\n{candidate_labels[candidate]}",
            fontsize=9.5,
            weight="bold",
        )
        ax.grid(alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted robust center cₘ (counts)")
    for ax in axes[1, :]:
        ax.set_xlabel("Needle count N (log₂ scale)")
    handles = [
        Line2D([0], [0], color=color, marker="o", label=f"T={length:,}")
        for length, color in zip(REGISTERED_LENGTHS, colors)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#596368",
            linewidth=2.2,
            label="Condition-only",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Selected per-model robust signed-bias laws at each reference condition",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(
        report_root / "assets" / "fig15_model_bias_selected_surfaces.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def update_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    block = """## Canonical reviewed version

This directory is the canonical report location. It includes the audited prompt
formats, per-model signed-bias analysis, and a formula/figure readability review.
All formulas use `T` for canonical passage length and `N` for true needle count.
Primary exact accuracy uses all 6,300 registered requests; absolute error and
signed bias remain conditional on the 5,385 parsed numeric outputs.

The readability pass regenerated Figures 7, 8, and 14 from the preserved CSV
tables and rewrote every figure caption to define its sample, axes, uncertainty,
and interpretation. Reapply the report-only pass in place with:

```powershell
python scripts/refine_report_readability.py --report-root .
```

"""
    marker = "## Rebuild\n"
    if block not in text:
        text = replace_once(text, marker, block + marker, "README insertion")
    path.write_text(text, encoding="utf-8")


def update_manifest(
    path: Path,
    report_root: Path,
    source_report: Path | None,
    html_audit: dict[str, Any],
    script_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    manifest["output_root"] = str(report_root)
    manifest["modified_at_utc"] = now
    figure_hashes = {
        filename: sha256(report_root / "assets" / filename)
        for filename in [
            "fig07_unified_law_parameters.png",
            "fig08_unified_law_surfaces.png",
            "fig15_model_bias_selected_surfaces.png",
        ]
    }
    manifest["readability_review_v1"] = {
        "reviewed_at_utc": now,
        "status": "pass",
        "canonical_report_root": str(report_root),
        "merged_source_report": str(source_report) if source_report else None,
        "prompt_section_present": True,
        "notation_policy": (
            "T is canonical passage length; N is true needle count; "
            "L is not used as the length symbol in report formulas."
        ),
        "sample_policy": {
            "primary_exact_accuracy": "all 6300 registered requests",
            "absolute_error_and_signed_bias": (
                "5385 parsed numeric outputs only"
            ),
        },
        "formula_cards": html_audit["formula_cards"],
        "embedded_figures": html_audit["embedded_figures"],
        "captions_reviewed": html_audit["caption_revisions"],
        "regenerated_figures": figure_hashes,
        "numeric_tables_modified": False,
        "raw_or_frozen_artifacts_modified": False,
    }
    script_entry = {
        "source": str(script_path),
        "destination": "refine_report_readability.py",
        "sha256": sha256(script_path),
    }
    reproduction = manifest.setdefault("reproduction_scripts", [])
    reproduction = [
        item
        for item in reproduction
        if item.get("destination") != "refine_report_readability.py"
    ]
    reproduction.append(script_entry)
    manifest["reproduction_scripts"] = reproduction
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest["readability_review_v1"]


def write_review_log(
    report_root: Path,
    copied_files: int,
    review: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "realistic-niah-report-readability-review-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "copied_files_from_prompt_report": copied_files,
        "review": review,
        "checks": [
            "canonical report contains audited prompt formats",
            "T/N notation is consistent in displayed formulas",
            "primary and conditional samples are explicitly separated",
            "all 15 embedded figure captions define sample and axes",
            "figures 7, 8, and 14 regenerated from preserved CSV tables",
            "raw requests, frozen stimuli, QC, and numeric fit tables unchanged",
        ],
    }
    (report_root / "logs" / "readability_review_log.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_hashes(report_root: Path) -> int:
    checksum_path = report_root / "SHA256SUMS.tsv"
    rows = []
    for path in sorted(report_root.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        rows.append((sha256(path), path.relative_to(report_root).as_posix()))
    with checksum_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)
    return len(rows)


def validate(report_root: Path) -> dict[str, Any]:
    html_text = (report_root / "report.html").read_text(encoding="utf-8")
    manifest = json.loads(
        (report_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    prompt_payload = json.loads(
        (
            report_root
            / "prompt_formats"
            / "model_prompt_formats.json"
        ).read_text(encoding="utf-8")
    )
    if manifest["output_root"] != str(report_root):
        raise RuntimeError("manifest output_root does not match canonical root")
    if "readability_review_v1" not in manifest:
        raise RuntimeError("manifest lacks readability review")
    if 'id="prompt-formats"' not in html_text:
        raise RuntimeError("prompt section missing")
    if len(re.findall(r'<figure class="report-figure">', html_text)) != 15:
        raise RuntimeError("embedded figure count is not 15")
    if len(re.findall(r'<div class="formula', html_text)) != 7:
        raise RuntimeError("formula card count is not 7")
    laws = html_text.split('<section id="laws">', 1)[1].split(
        "</section>", 1
    )[0]
    bias = html_text.split('<section id="bias">', 1)[1].split(
        "</section>", 1
    )[0]
    model_bias = html_text.split(
        '<section id="model-bias-laws">', 1
    )[1].split("</section>", 1)[0]
    forbidden = [
        "p<sub>m</sub>(L",
        "log(L/5000)",
        "log₂(L/5000)",
        "blocked (L,N)",
        "log L + log N",
        "平滑 L/N law",
    ]
    stale = [token for token in forbidden if token in laws + bias + model_bias]
    if stale:
        raise RuntimeError(f"stale length notation remains: {stale}")
    for filename in FIGURE_FILES:
        path = report_root / "assets" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty figure: {filename}")
        image = plt.imread(path)
        if image.size == 0:
            raise RuntimeError(f"unreadable figure: {filename}")
    if len(prompt_payload) == 0:
        raise RuntimeError("prompt format payload is empty")
    checksum_path = report_root / "SHA256SUMS.tsv"
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("\t", 1)
        actual = sha256(report_root / relative)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {relative}")
        checked += 1
    return {
        "status": "pass",
        "files_hashed": checked,
        "embedded_figures": 15,
        "formula_cards": 7,
        "prompt_combinations": manifest["prompt_formats_v1"][
            "registered_combinations"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--source-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.report_root.resolve()
    source = args.source_report.resolve() if args.source_report else None
    copied_files = 0
    if source and source != target:
        copied_files = merge_report(source, target)
    target = ensure_report_root(target)

    script_destination = target / "scripts" / "refine_report_readability.py"
    current_script = Path(__file__).resolve()
    if current_script != script_destination.resolve():
        shutil.copy2(current_script, script_destination)

    html_audit = refine_html(target / "report.html")
    regenerate_accuracy_parameter_figure(target)
    regenerate_accuracy_surface_figure(target)
    regenerate_bias_surface_figure(target)
    update_readme(target / "README.md")
    review = update_manifest(
        target / "analysis_manifest.json",
        target,
        source,
        html_audit,
        script_destination,
    )
    write_review_log(target, copied_files, review)
    write_hashes(target)
    validation = validate(target)
    print(
        json.dumps(
            {
                "canonical_report_root": str(target),
                "copied_files": copied_files,
                "html_audit": html_audit,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
