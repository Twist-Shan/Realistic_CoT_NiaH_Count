from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


MODE_ORDER = [
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
]
MODE_LABEL = {
    "direct": "Direct",
    "enumeration_index": "Index",
    "enumeration_bullet": "Bullet",
    "native_thinking": "Native Thinking",
}
MODE_COLORS = {
    "direct": "#3366A5",
    "enumeration_index": "#D77826",
    "enumeration_bullet": "#2A8C6A",
    "native_thinking": "#7652A8",
}
MODEL_ORDER = [
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "DeepSeek-R1-0528-Qwen3-8B",
    "GLM-Z1-9B-0414",
    "GLM-4-9B-0414",
]
TARGET_MODELS = MODEL_ORDER[:-1]
QWEN_MODELS = MODEL_ORDER[:4]
ERROR_ORDER = [
    "wrong_count",
    "truncated",
    "parse_failure",
    "response_format_failure",
    "enumeration_format_failure",
    "other_failure",
]
ERROR_LABEL = {
    "wrong_count": "Wrong count",
    "truncated": "Truncated",
    "parse_failure": "No parsable Total",
    "response_format_failure": "Final-format failure",
    "enumeration_format_failure": "Enumeration-format failure",
    "other_failure": "Other",
}
ERROR_COLORS = {
    "wrong_count": "#C66A42",
    "truncated": "#A23B3B",
    "parse_failure": "#8156A7",
    "response_format_failure": "#B28A2E",
    "enumeration_format_failure": "#6E7C86",
    "other_failure": "#999999",
}
L_LEVELS = [2000, 3000, 5000, 10000, 20000]
N_LEVELS = [1, 2, 3, 4, 5, 6, 8, 10, 20, 30]


COMMON_PROMPT = """You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{PASSAGE}
</passage>"""

FORMAL_DIRECT = """How many city-score audit records are in the passage?
In the final answer, output exactly one line:
Total: <integer>"""

STRICT_DIRECT = """How many city-score audit records are in the passage?
Do not explain, reason aloud, quote, or list any records.
Your entire response must be exactly one line:
Total: <integer>"""

INDEX_PROMPT = """How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin the first item with "1. ", the second with "2. ", and continue with ordinary digits.
After each number, write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text."""

BULLET_PROMPT = """How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin each item with "-", then write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text."""

THINKING_PROMPT = """How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>"""


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    formula: str
    feature_names: tuple[str, ...]
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray]

    @property
    def k(self) -> int:
        return 1 + len(self.feature_names)


def cols(*arrays: np.ndarray) -> np.ndarray:
    return np.column_stack(arrays)


def candidate_registry() -> list[Candidate]:
    return [
        Candidate("constant", "constant", r"b=\alpha", (), lambda n, l: np.empty((len(n), 0))),
        Candidate("N", "N", r"b=\alpha+\beta_NN", ("N",), lambda n, l: cols(n)),
        Candidate("L", "L", r"b=\alpha+\beta_LL_k", ("Lk",), lambda n, l: cols(l)),
        Candidate(
            "logN",
            "log₂ N",
            r"b=\alpha+\beta_N\log_2N",
            ("log2N",),
            lambda n, l: cols(np.log2(n)),
        ),
        Candidate(
            "logL",
            "log₂ L",
            r"b=\alpha+\beta_L\log_2L_k",
            ("log2Lk",),
            lambda n, l: cols(np.log2(l)),
        ),
        Candidate(
            "density",
            "N/L",
            r"b=\alpha+\beta_DN/L_k",
            ("N/Lk",),
            lambda n, l: cols(n / l),
        ),
        Candidate(
            "inv_density",
            "L/N",
            r"b=\alpha+\beta_D L_k/N",
            ("Lk/N",),
            lambda n, l: cols(l / n),
        ),
        Candidate(
            "product",
            "NL",
            r"b=\alpha+\beta_PNL_k",
            ("NLk",),
            lambda n, l: cols(n * l),
        ),
        Candidate(
            "log_product",
            "log₂(NL)",
            r"b=\alpha+\beta_P\log_2(NL_k)",
            ("log2NLk",),
            lambda n, l: cols(np.log2(n * l)),
        ),
        Candidate(
            "N_L",
            "N + L",
            r"b=\alpha+\beta_NN+\beta_LL_k",
            ("N", "Lk"),
            lambda n, l: cols(n, l),
        ),
        Candidate(
            "N_logL",
            "N + log₂L",
            r"b=\alpha+\beta_NN+\beta_L\log_2L_k",
            ("N", "log2Lk"),
            lambda n, l: cols(n, np.log2(l)),
        ),
        Candidate(
            "logN_L",
            "log₂N + L",
            r"b=\alpha+\beta_N\log_2N+\beta_LL_k",
            ("log2N", "Lk"),
            lambda n, l: cols(np.log2(n), l),
        ),
        Candidate(
            "logN_logL",
            "log₂N + log₂L",
            r"b=\alpha+\beta_N\log_2N+\beta_L\log_2L_k",
            ("log2N", "log2Lk"),
            lambda n, l: cols(np.log2(n), np.log2(l)),
        ),
        Candidate(
            "N_density",
            "N + N/L",
            r"b=\alpha+\beta_NN+\beta_DN/L_k",
            ("N", "N/Lk"),
            lambda n, l: cols(n, n / l),
        ),
        Candidate(
            "L_density",
            "L + N/L",
            r"b=\alpha+\beta_LL_k+\beta_DN/L_k",
            ("Lk", "N/Lk"),
            lambda n, l: cols(l, n / l),
        ),
        Candidate(
            "N_L_density",
            "N + L + N/L",
            r"b=\alpha+\beta_NN+\beta_LL_k+\beta_DN/L_k",
            ("N", "Lk", "N/Lk"),
            lambda n, l: cols(n, l, n / l),
        ),
        Candidate(
            "N_L_interaction",
            "N + L + NL",
            r"b=\alpha+\beta_NN+\beta_LL_k+\beta_{NL}NL_k",
            ("N", "Lk", "NLk"),
            lambda n, l: cols(n, l, n * l),
        ),
        Candidate(
            "N_logL_interaction",
            "N + log₂L + Nlog₂L",
            r"b=\alpha+\beta_NN+\beta_L\log_2L_k+\beta_{NL}N\log_2L_k",
            ("N", "log2Lk", "Nlog2Lk"),
            lambda n, l: cols(n, np.log2(l), n * np.log2(l)),
        ),
        Candidate(
            "logN_logL_interaction",
            "log₂N + log₂L + interaction",
            r"b=\alpha+\beta_N\log_2N+\beta_L\log_2L_k+\beta_{NL}\log_2N\log_2L_k",
            ("log2N", "log2Lk", "log2Nlog2Lk"),
            lambda n, l: cols(
                np.log2(n), np.log2(l), np.log2(n) * np.log2(l)
            ),
        ),
        Candidate(
            "N_piece10_logL",
            "segmented N at 10 + log₂L",
            r"b=\alpha+\beta_NN+\beta_H(N-10)_++\beta_L\log_2L_k",
            ("N", "(N-10)+", "log2Lk"),
            lambda n, l: cols(n, np.maximum(n - 10.0, 0.0), np.log2(l)),
        ),
        Candidate(
            "logN_L_piece5",
            "log₂N + segmented L at 5k",
            r"b=\alpha+\beta_N\log_2N+\beta_LL_k+\beta_H(L_k-5)_+",
            ("log2N", "Lk", "(Lk-5)+"),
            lambda n, l: cols(np.log2(n), l, np.maximum(l - 5.0, 0.0)),
        ),
        Candidate(
            "N_piece10_L_piece5",
            "segmented N at 10 + segmented L at 5k",
            r"b=\alpha+\beta_NN+\beta_{NH}(N-10)_++\beta_LL_k+\beta_{LH}(L_k-5)_+",
            ("N", "(N-10)+", "Lk", "(Lk-5)+"),
            lambda n, l: cols(
                n,
                np.maximum(n - 10.0, 0.0),
                l,
                np.maximum(l - 5.0, 0.0),
            ),
        ),
        Candidate(
            "N2_L",
            "quadratic N + L",
            r"b=\alpha+\beta_1N+\beta_2N^2+\beta_LL_k",
            ("N", "N2", "Lk"),
            lambda n, l: cols(n, n * n, l),
        ),
        Candidate(
            "N_L2",
            "N + quadratic L",
            r"b=\alpha+\beta_NN+\beta_1L_k+\beta_2L_k^2",
            ("N", "Lk", "Lk2"),
            lambda n, l: cols(n, l, l * l),
        ),
        Candidate(
            "N2_L2",
            "quadratic N + quadratic L",
            r"b=\alpha+\beta_1N+\beta_2N^2+\beta_3L_k+\beta_4L_k^2",
            ("N", "N2", "Lk", "Lk2"),
            lambda n, l: cols(n, n * n, l, l * l),
        ),
    ]


def ensure_bool(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in data:
            continue
        if data[column].dtype == object:
            data[column] = data[column].map(
                {"True": True, "False": False, True: True, False: False}
            )
        data[column] = data[column].fillna(False).astype(bool)
    return data


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = stats.norm.ppf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return center - half, center + half


def bh_adjust(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    valid = np.flatnonzero(np.isfinite(arr))
    if not len(valid):
        return pd.Series(result, index=values.index)
    order = valid[np.argsort(arr[valid])]
    adjusted = np.empty(len(order))
    running = 1.0
    m = len(order)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = m - reverse_rank + 1
        running = min(running, arr[index] * m / rank)
        adjusted[m - reverse_rank] = running
    result[order] = adjusted
    return pd.Series(result, index=values.index)


def safe_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(prediction)
    y = y[valid]
    prediction = prediction[valid]
    denominator = np.sum((y - y.mean()) ** 2)
    if len(y) < 2 or denominator <= 1e-12:
        return math.nan
    return float(1 - np.sum((y - prediction) ** 2) / denominator)


def raw_features(candidate: Candidate, data: pd.DataFrame) -> np.ndarray:
    n = data["N"].to_numpy(dtype=float)
    lk = data["L"].to_numpy(dtype=float) / 1000.0
    transformed = candidate.transform(n, lk)
    return np.column_stack([np.ones(len(data)), transformed])


def fit_raw(
    candidate: Candidate, train: pd.DataFrame, target: str
) -> tuple[np.ndarray, np.ndarray]:
    x = raw_features(candidate, train)
    y = train[target].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta, x @ beta


def fit_predict_standardized(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> np.ndarray:
    n_train = train["N"].to_numpy(dtype=float)
    l_train = train["L"].to_numpy(dtype=float) / 1000.0
    n_test = test["N"].to_numpy(dtype=float)
    l_test = test["L"].to_numpy(dtype=float) / 1000.0
    a = candidate.transform(n_train, l_train)
    b = candidate.transform(n_test, l_test)
    if a.shape[1] == 0:
        return np.repeat(train[target].mean(), len(test))
    mean = a.mean(axis=0)
    scale = np.where(a.std(axis=0) == 0, 1.0, a.std(axis=0))
    x_train = np.column_stack([np.ones(len(train)), (a - mean) / scale])
    x_test = np.column_stack([np.ones(len(test)), (b - mean) / scale])
    beta = np.linalg.lstsq(
        x_train, train[target].to_numpy(dtype=float), rcond=None
    )[0]
    return x_test @ beta


def candidate_cv(
    group: pd.DataFrame,
    candidate: Candidate,
    target: str,
) -> dict[str, float | np.ndarray]:
    group = group.reset_index(drop=True)
    oof = np.full(len(group), np.nan)
    fold_mse: list[float] = []
    for seed in sorted(group["seed"].unique()):
        test = group[group["seed"] == seed]
        train = group[group["seed"] != seed]
        if train.empty or test.empty:
            continue
        prediction = fit_predict_standardized(candidate, train, test, target)
        oof[test.index] = prediction
        fold_mse.append(
            float(np.mean((test[target].to_numpy(dtype=float) - prediction) ** 2))
        )
    crossfit_cells = (
        group.assign(prediction=oof)
        .groupby(["N", "L"], as_index=False)
        .agg(observed=(target, "mean"), prediction=("prediction", "mean"))
    )
    cells = (
        group.groupby(["N", "L"], as_index=False)
        .agg(observed=(target, "mean"), parsed_n=(target, "size"))
        .rename(columns={"observed": target})
    )
    prediction = fit_predict_standardized(candidate, group, cells, target)

    def level_oof(column: str) -> float:
        holdout = np.full(len(group), np.nan)
        for level in sorted(group[column].unique()):
            test = group[group[column] == level]
            train = group[group[column] != level]
            if train.empty or test.empty:
                continue
            holdout[test.index] = fit_predict_standardized(
                candidate, train, test, target
            )
        level_cells = (
            group.assign(prediction=holdout)
            .groupby(["N", "L"], as_index=False)
            .agg(observed=(target, "mean"), prediction=("prediction", "mean"))
        )
        return safe_r2(
            level_cells["observed"].to_numpy(),
            level_cells["prediction"].to_numpy(),
        )

    return {
        "seed_cv_mse": float(np.mean(fold_mse)),
        "seed_cv_se": float(
            np.std(fold_mse, ddof=1) / math.sqrt(len(fold_mse))
            if len(fold_mse) > 1
            else math.nan
        ),
        "request_oof_r2": safe_r2(group[target].to_numpy(), oof),
        "cell_crossfit_r2": safe_r2(
            crossfit_cells["observed"].to_numpy(),
            crossfit_cells["prediction"].to_numpy(),
        ),
        "cell_fit_r2": safe_r2(cells[target].to_numpy(), prediction),
        "leave_N_out_r2": level_oof("N"),
        "leave_L_out_r2": level_oof("L"),
        "oof": oof,
    }


def overall_f_test(candidate: Candidate, cells: pd.DataFrame, target: str) -> float:
    if candidate.k <= 1 or len(cells) <= candidate.k:
        return math.nan
    x = raw_features(candidate, cells)
    y = cells[target].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    sse = float(residual @ residual)
    sst = float(((y - y.mean()) ** 2).sum())
    df_model = candidate.k - 1
    df_error = len(y) - candidate.k
    if sst <= 1e-12 or df_error <= 0:
        return math.nan
    ssr = max(0.0, sst - sse)
    f_value = (ssr / df_model) / max(sse / df_error, 1e-15)
    return float(stats.f.sf(f_value, df_model, df_error))


def bootstrap_coefficients(
    candidate: Candidate,
    group: pd.DataFrame,
    target: str,
    *,
    repetitions: int = 400,
    seed: int = 20260726,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seed_values = np.array(sorted(group["seed"].unique()))
    draws: list[np.ndarray] = []
    for _ in range(repetitions):
        sampled = rng.choice(seed_values, size=len(seed_values), replace=True)
        pieces = [group[group["seed"] == value] for value in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        cells = boot.groupby(["N", "L"], as_index=False)[target].mean()
        if len(cells) <= candidate.k:
            continue
        beta, _ = fit_raw(candidate, cells, target)
        draws.append(beta)
    return np.asarray(draws)


def format_number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "—"
    if abs(value) < 0.0005:
        return "0"
    return f"{value:.{digits}f}"


def numeric_formula(candidate: Candidate, beta: np.ndarray, target_symbol: str) -> str:
    terms = [f"{beta[0]:.3g}"]
    for value, name in zip(beta[1:], candidate.feature_names):
        sign = "+" if value >= 0 else "-"
        pretty_name = {
            "Lk": "L_k",
            "N2": "N^2",
            "Lk2": "L_k^2",
            "NLk": "NL_k",
            "log2N": r"\log_2N",
            "log2Lk": r"\log_2L_k",
            "log2NLk": r"\log_2(NL_k)",
            "N/Lk": r"N/L_k",
            "Lk/N": r"L_k/N",
            "Nlog2Lk": r"N\log_2L_k",
            "log2Nlog2Lk": r"\log_2N\log_2L_k",
            "(N-10)+": r"(N-10)_+",
            "(Lk-5)+": r"(L_k-5)_+",
        }.get(name, name)
        terms.append(f" {sign} {abs(value):.3g}{pretty_name}")
    return rf"{target_symbol}=" + "".join(terms)


def table_html(
    frame: pd.DataFrame,
    *,
    classes: str = "data-table",
    index: bool = False,
    escape: bool = True,
) -> str:
    return (
        '<div class="table-wrap">'
        + frame.to_html(
            index=index,
            classes=classes,
            border=0,
            escape=escape,
            na_rep="—",
        )
        + "</div>"
    )


CSS = """
:root {
  --ink:#18222e; --muted:#5c6b79; --line:#d8e0e7; --soft:#f4f7f9;
  --blue:#275d8c; --blue-soft:#eaf2f8; --green:#176f5b; --amber:#9a5a14;
  --red:#9d3434; --paper:#ffffff;
}
*{box-sizing:border-box}
body{margin:0;background:#eef2f5;color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif;line-height:1.62}
main{max-width:1180px;margin:0 auto;background:var(--paper);padding:48px 64px 80px;min-height:100vh}
h1{font-size:2.05rem;line-height:1.22;margin:0 0 10px;letter-spacing:-.02em}
h2{font-size:1.46rem;margin:48px 0 18px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:1.1rem;margin:28px 0 12px}
p{margin:10px 0}
.subtitle{color:var(--muted);font-size:1.05rem;margin-bottom:24px}
.meta{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 28px}
.pill{background:var(--soft);border:1px solid var(--line);border-radius:999px;padding:5px 10px;font-size:.86rem}
.toc{background:var(--soft);border-left:4px solid var(--blue);padding:14px 18px;margin:24px 0}
.toc a{color:var(--blue);text-decoration:none;margin-right:16px;white-space:nowrap}
.callout{background:var(--blue-soft);border-left:4px solid var(--blue);padding:14px 18px;margin:18px 0}
.warning{background:#fff6e9;border-left-color:var(--amber)}
.conclusion{background:#edf7f3;border-left:4px solid var(--green);padding:14px 18px;margin:20px 0 6px}
.conclusion strong{color:var(--green)}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.metric{border:1px solid var(--line);padding:13px 15px;border-radius:8px}
.metric .value{font-size:1.38rem;font-weight:650;color:var(--blue)}
.metric .label{font-size:.82rem;color:var(--muted)}
figure{margin:26px 0 30px}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:8px;background:white}
figcaption{font-size:.9rem;color:var(--muted);margin-top:8px}
.table-wrap{overflow-x:auto;margin:14px 0 20px}
table.dataframe{border-collapse:collapse;width:100%;font-size:.86rem}
table.dataframe th,table.dataframe td{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
table.dataframe th:first-child,table.dataframe td:first-child{text-align:left}
table.dataframe thead th{background:var(--soft);position:sticky;top:0;z-index:1}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;border:1px solid var(--line);border-radius:7px;padding:14px;font-size:.86rem;line-height:1.5}
code{font-family:"Cascadia Mono",Consolas,monospace}
details{border:1px solid var(--line);border-radius:7px;margin:10px 0;padding:10px 14px}
summary{cursor:pointer;font-weight:650}
.equation{overflow-x:auto;text-align:center;margin:16px 0;padding:10px}
.small{font-size:.88rem;color:var(--muted)}
.good{color:var(--green);font-weight:650}.mid{color:var(--amber);font-weight:650}.weak{color:var(--red);font-weight:650}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
@media(max-width:760px){main{padding:28px 18px}h1{font-size:1.65rem}.metric-grid{grid-template-columns:1fr 1fr}}
@media print{body{background:white}main{max-width:none;padding:20px}details{break-inside:avoid}figure{break-inside:avoid}}
"""


def page(title: str, subtitle: str, body: str, generated: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<script>
window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="subtitle">{html.escape(subtitle)}</div>
{body}
<footer>Generated {html.escape(generated)}. 所有百分比均由冻结请求逐条重新汇总；未改写原始输出或评分。</footer>
</main></body></html>"""


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#8896A3",
            "axes.grid": True,
            "grid.color": "#E4E9ED",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_old_gemma(v2_root: Path) -> tuple[pd.DataFrame, list[str]]:
    path = (
        v2_root
        / "shards"
        / "Gemma4-12B__direct"
        / "main"
        / "requests.jsonl"
    )
    rows: list[dict] = []
    examples: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            ev = item["evaluation"]
            if len(examples) < 2:
                examples.append(str(item.get("raw_output_text") or ""))
            rows.append(
                {
                    "registered_success": bool(ev.get("registered_success")),
                    "exact_count": bool(ev.get("exact_count")),
                    "parse_ok": ev.get("parse_status") == "ok",
                    "format_ok": bool(ev.get("response_format_compliant")),
                    "truncated": bool(ev.get("truncated")),
                    "N": item["num_needles"],
                    "L": item["target_passage_tokens"],
                }
            )
    return pd.DataFrame(rows), examples


def prompt_summaries(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, mode), group in data.groupby(["model", "mode"], sort=False):
        successes = int(group["registered_success"].sum())
        low, high = wilson_interval(successes, len(group))
        numeric = group["signed_error"].dropna()
        records.append(
            {
                "model": model,
                "mode": mode,
                "n": len(group),
                "success_rate": successes / len(group),
                "ci_low": low,
                "ci_high": high,
                "exact_rate": group["exact_count"].mean(),
                "parse_rate": group["parse_ok"].mean(),
                "format_rate": group["format_ok"].mean(),
                "truncation_rate": group["truncated"].mean(),
                "mean_bias_parsed": numeric.mean(),
                "mae_parsed": group["absolute_error"].mean(),
                "under_rate_parsed": (numeric < 0).mean(),
                "over_rate_parsed": (numeric > 0).mean(),
                "source_version": group["source_version"].iloc[0],
            }
        )
    result = pd.DataFrame(records)
    result["model"] = pd.Categorical(result["model"], MODEL_ORDER, ordered=True)
    result["mode"] = pd.Categorical(result["mode"], MODE_ORDER, ordered=True)
    return result.sort_values(["model", "mode"]).reset_index(drop=True)


def paired_mode_effects(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    comparisons = [
        ("direct", "enumeration_index"),
        ("direct", "enumeration_bullet"),
        ("direct", "native_thinking"),
        ("enumeration_bullet", "enumeration_index"),
        ("enumeration_index", "native_thinking"),
    ]
    for model, model_data in data.groupby("model"):
        for mode_a, mode_b in comparisons:
            a = model_data[model_data["mode"] == mode_a][
                ["stimulus_id", "registered_success"]
            ]
            b = model_data[model_data["mode"] == mode_b][
                ["stimulus_id", "registered_success"]
            ]
            if a.empty or b.empty:
                continue
            joined = a.merge(b, on="stimulus_id", suffixes=("_a", "_b"))
            if len(joined) != 500:
                continue
            sa = joined["registered_success_a"].astype(int)
            sb = joined["registered_success_b"].astype(int)
            a_only = int(((sa == 1) & (sb == 0)).sum())
            b_only = int(((sa == 0) & (sb == 1)).sum())
            discordant = a_only + b_only
            p_value = (
                stats.binomtest(min(a_only, b_only), discordant, 0.5).pvalue
                if discordant
                else 1.0
            )
            records.append(
                {
                    "model": model,
                    "mode_A": mode_a,
                    "mode_B": mode_b,
                    "success_A": sa.mean(),
                    "success_B": sb.mean(),
                    "difference_pp": 100 * (sb.mean() - sa.mean()),
                    "A_only": a_only,
                    "B_only": b_only,
                    "mcnemar_exact_p": p_value,
                }
            )
    result = pd.DataFrame(records)
    result["holm_p"] = np.minimum(
        1.0,
        result["mcnemar_exact_p"].rank(method="min")
        .rsub(len(result) + 1)
        .mul(result["mcnemar_exact_p"]),
    )
    return result


def plot_accuracy_heatmap(summary: pd.DataFrame, path: Path) -> None:
    matrix = (
        summary.pivot(index="model", columns="mode", values="success_rate")
        .reindex(index=MODEL_ORDER, columns=MODE_ORDER)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(MODE_ORDER)), [MODE_LABEL[m] for m in MODE_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{100*value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.58 else "#17212B",
                    fontsize=8.5,
                    fontweight="bold",
                )
            else:
                ax.text(column, row, "not run", ha="center", va="center", color="#777")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Registered success rate")
    ax.set_title("Current-prompt success by model and mode")
    ax.set_xlabel("Prompt mode")
    ax.set_ylabel("Model")
    save_figure(fig, path)


def plot_accuracy_by_axis(data: pd.DataFrame, axis: str, path: Path) -> None:
    levels = N_LEVELS if axis == "N" else L_LEVELS
    grouped = (
        data.groupby(["model", "mode", axis], as_index=False)["registered_success"]
        .mean()
        .rename(columns={"registered_success": "accuracy"})
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2), sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        subset = grouped[grouped["model"] == model]
        for mode in MODE_ORDER:
            line = subset[subset["mode"] == mode].sort_values(axis)
            if line.empty:
                continue
            x = line[axis].to_numpy()
            if axis == "L":
                x = x / 1000
            ax.plot(
                x,
                100 * line["accuracy"],
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                color=MODE_COLORS[mode],
                label=MODE_LABEL[mode],
            )
        ax.set_title(model)
        ax.set_ylim(-2, 102)
        ax.set_xticks(
            [1, 2, 3, 4, 5, 6, 8, 10, 20, 30]
            if axis == "N"
            else [2, 3, 5, 10, 20]
        )
        ax.set_xlabel("True needle count N" if axis == "N" else "Passage length L (k tokens)")
        ax.set_ylabel("Registered success (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        "Accuracy versus true needle count" if axis == "N" else "Accuracy versus passage length",
        y=1.015,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_failure_budget(data: pd.DataFrame, path: Path) -> None:
    failures = (
        data[data["error_category"] != "success"]
        .groupby(["model", "mode", "error_category"])
        .size()
        .unstack(fill_value=0)
    )
    index = pd.MultiIndex.from_tuples(
        [
            (model, mode)
            for model in MODEL_ORDER
            for mode in MODE_ORDER
            if ((data["model"] == model) & (data["mode"] == mode)).any()
        ],
        names=["model", "mode"],
    )
    failures = failures.reindex(index, fill_value=0)
    totals = data.groupby(["model", "mode"]).size().reindex(index)
    values = failures.div(totals, axis=0)
    labels = [f"{model} · {MODE_LABEL[mode]}" for model, mode in index]
    fig, ax = plt.subplots(figsize=(10.2, 9.2))
    left = np.zeros(len(values))
    for category in ERROR_ORDER:
        series = values[category].to_numpy() if category in values else np.zeros(len(values))
        ax.barh(
            np.arange(len(values)),
            100 * series,
            left=100 * left,
            color=ERROR_COLORS[category],
            label=ERROR_LABEL[category],
            height=0.72,
        )
        left += series
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of all 500 requests (%)")
    ax.set_title("Failure budget: mutually exclusive first failure")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    save_figure(fig, path)


def plot_gemma_direct_comparison(
    old: pd.DataFrame, strict: pd.DataFrame, path: Path
) -> None:
    metrics = ["registered_success", "exact_count", "parse_ok", "format_ok", "truncated"]
    labels = ["Success", "Exact count", "Parsed", "Strict format", "Truncated"]
    old_values = [old[m].mean() for m in metrics]
    strict_values = [strict[m].mean() for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.36
    ax.bar(x - width / 2, 100 * np.array(old_values), width, label="Original V2 Direct", color="#9D6670")
    ax.bar(x + width / 2, 100 * np.array(strict_values), width, label="Strict appendix Direct", color="#3D7FA6")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Gemma4-12B Direct: prompt-only intervention")
    ax.legend(frameon=False)
    for i, values in enumerate([old_values, strict_values]):
        offset = -width / 2 if i == 0 else width / 2
        for j, value in enumerate(values):
            ax.text(j + offset, 100 * value + 2, f"{100*value:.1f}", ha="center", fontsize=8)
    save_figure(fig, path)


def representative_errors(data: pd.DataFrame, old_examples: list[str]) -> pd.DataFrame:
    specifications = [
        (
            "Gemma4-12B (original)",
            "direct",
            "truncated",
            "The weak Direct instruction triggered an explicit scan/list; the 64-token budget ended before Total.",
            old_examples[0][:900],
        ),
        (
            "Gemma4-12B",
            "native_thinking",
            "response_format_failure",
            "The stored output exposed a <|channel>thought block. The frozen parser retained it in final_text, so an often-correct count failed the Total-only gate.",
            None,
        ),
        (
            "Qwen3-1.7B",
            "enumeration_index",
            "parse_failure",
            "The model listed records but omitted the required final Total line.",
            None,
        ),
        (
            "Qwen3-1.7B",
            "enumeration_index",
            "enumeration_format_failure",
            "After valid records it hallucinated/repeated placeholder-like items, inflating the total.",
            None,
        ),
        (
            "Qwen3-1.7B",
            "enumeration_bullet",
            "truncated",
            "The model entered a repeated-item loop until the 1,536-token limit.",
            None,
        ),
        (
            "Qwen3-4B",
            "direct",
            "truncated",
            "Despite non-thinking mode, the weak formal Direct wording elicited explanation and enumeration; 64 tokens were insufficient.",
            None,
        ),
        (
            "GLM-4-9B-0414",
            "enumeration_index",
            "wrong_count",
            "The syntax was valid, but one or more dispersed records were omitted.",
            None,
        ),
        (
            "DeepSeek-R1-0528-Qwen3-8B",
            "native_thinking",
            "truncated",
            "The reasoning trace began a long scan/list and did not reach the final line within 4,096 tokens.",
            None,
        ),
    ]
    rows = []
    for model, mode, category, mechanism, fixed_excerpt in specifications:
        excerpt = fixed_excerpt
        if excerpt is None:
            match = data[
                (data["model"] == model)
                & (data["mode"] == mode)
                & (data["error_category"] == category)
            ]
            excerpt = "" if match.empty else str(match.iloc[0]["raw_output_excerpt"])
        row = {
            "model": model,
            "mode": mode,
            "failure": category,
            "observed mechanism": mechanism,
            "output excerpt": excerpt.replace("\r", " ").strip(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_prompt_report(
    data: pd.DataFrame,
    v2_root: Path,
    output: Path,
    generated: str,
) -> dict[str, pd.DataFrame]:
    figures = output / "figures" / "prompt"
    tables = output / "tables" / "prompt"
    tables.mkdir(parents=True, exist_ok=True)
    summary = prompt_summaries(data)
    summary.to_csv(tables / "model_mode_summary.csv", index=False)
    by_n = (
        data.groupby(["model", "mode", "N"], as_index=False)
        .agg(n=("request_id", "size"), success_rate=("registered_success", "mean"))
    )
    by_l = (
        data.groupby(["model", "mode", "L"], as_index=False)
        .agg(n=("request_id", "size"), success_rate=("registered_success", "mean"))
    )
    by_n.to_csv(tables / "accuracy_by_N.csv", index=False)
    by_l.to_csv(tables / "accuracy_by_L.csv", index=False)
    failure_budget = (
        data.groupby(["model", "mode", "error_category"])
        .size()
        .rename("count")
        .reset_index()
    )
    failure_budget["rate"] = failure_budget["count"] / 500
    failure_budget.to_csv(tables / "failure_budget.csv", index=False)
    paired = paired_mode_effects(data)
    paired.to_csv(tables / "paired_mode_effects.csv", index=False)
    old_gemma, old_examples = load_old_gemma(v2_root)
    strict_gemma = data[
        (data["model"] == "Gemma4-12B") & (data["mode"] == "direct")
    ]
    examples = representative_errors(data, old_examples)
    examples.to_csv(tables / "representative_errors.csv", index=False)

    plot_accuracy_heatmap(summary, figures / "01_accuracy_heatmap.png")
    plot_accuracy_by_axis(data, "N", figures / "02_accuracy_by_N.png")
    plot_accuracy_by_axis(data, "L", figures / "03_accuracy_by_L.png")
    plot_failure_budget(data, figures / "04_failure_budget.png")
    plot_gemma_direct_comparison(
        old_gemma, strict_gemma, figures / "05_gemma_direct_prompt_intervention.png"
    )

    top_summary = summary.copy()
    top_summary["Model"] = top_summary["model"].astype(str)
    top_summary["Mode"] = top_summary["mode"].map(MODE_LABEL)
    top_summary["Success"] = top_summary["success_rate"].map(lambda x: f"{100*x:.1f}%")
    top_summary["95% CI"] = top_summary.apply(
        lambda r: f"{100*r.ci_low:.1f}–{100*r.ci_high:.1f}%", axis=1
    )
    top_summary["Exact"] = top_summary["exact_rate"].map(lambda x: f"{100*x:.1f}%")
    top_summary["Parsed"] = top_summary["parse_rate"].map(lambda x: f"{100*x:.1f}%")
    top_summary["Format"] = top_summary["format_rate"].map(lambda x: f"{100*x:.1f}%")
    top_summary["Truncated"] = top_summary["truncation_rate"].map(lambda x: f"{100*x:.1f}%")
    top_summary["Mean bias*"] = top_summary["mean_bias_parsed"].map(
        lambda x: format_number(x, 2)
    )
    display_summary = top_summary[
        ["Model", "Mode", "Success", "95% CI", "Exact", "Parsed", "Format", "Truncated", "Mean bias*"]
    ]

    best_mode_rows = []
    for model in MODEL_ORDER:
        subset = summary[summary["model"].astype(str) == model]
        if subset.empty:
            continue
        best = subset.loc[subset["success_rate"].idxmax()]
        best_mode_rows.append(
            {
                "Model": model,
                "Best current mode": MODE_LABEL[str(best["mode"])],
                "Success": f"{100*best['success_rate']:.1f}%",
            }
        )
    best_modes = pd.DataFrame(best_mode_rows)

    paired_show = paired[
        paired["model"].isin(QWEN_MODELS + ["Gemma4-E4B", "Gemma4-12B"])
    ].copy()
    paired_show["Comparison"] = paired_show.apply(
        lambda r: f"{MODE_LABEL[r.mode_B]} − {MODE_LABEL[r.mode_A]}", axis=1
    )
    paired_show["Δ success"] = paired_show["difference_pp"].map(lambda x: f"{x:+.1f} pp")
    paired_show["Exact paired p"] = paired_show["mcnemar_exact_p"].map(
        lambda x: f"{x:.2e}"
    )
    paired_show = paired_show[["model", "Comparison", "Δ success", "Exact paired p"]]
    paired_show.columns = ["Model", "Comparison", "Δ success", "Exact paired p"]

    error_rows = []
    for (model, mode), group in data.groupby(["model", "mode"]):
        failures = group[group["error_category"] != "success"]
        dominant = (
            failures["error_category"].value_counts().index[0]
            if len(failures)
            else "none"
        )
        numeric = group["signed_error"].dropna()
        error_rows.append(
            {
                "Model": model,
                "Mode": MODE_LABEL[mode],
                "Dominant failure": ERROR_LABEL.get(dominant, "None"),
                "Failure rate": f"{100*(1-group.registered_success.mean()):.1f}%",
                "Undercount among parsed": f"{100*(numeric<0).mean():.1f}%",
                "Overcount among parsed": f"{100*(numeric>0).mean():.1f}%",
                "MAE among parsed": f"{group.absolute_error.mean():.2f}",
            }
        )
    error_table = pd.DataFrame(error_rows)

    detail_blocks = []
    for model in MODEL_ORDER:
        model_n = (
            by_n[by_n["model"] == model]
            .pivot(index="mode", columns="N", values="success_rate")
            .reindex(MODE_ORDER)
        )
        model_l = (
            by_l[by_l["model"] == model]
            .pivot(index="mode", columns="L", values="success_rate")
            .reindex(MODE_ORDER)
        )
        if model_n.dropna(how="all").empty:
            continue
        model_n.index = [MODE_LABEL.get(x, x) for x in model_n.index]
        model_l.index = [MODE_LABEL.get(x, x) for x in model_l.index]
        model_n = model_n.map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
        model_l = model_l.map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
        model_l.columns = [f"{int(x/1000)}k" for x in model_l.columns]
        detail_blocks.append(
            f"""<details><summary>{html.escape(model)}：按 N 与 L 的完整准确率</summary>
            <h3>按真实 needle 数 N</h3>{table_html(model_n, index=True)}
            <h3>按 passage 长度 L</h3>{table_html(model_l, index=True)}
            </details>"""
        )

    example_blocks = []
    for _, row in examples.iterrows():
        example_blocks.append(
            f"""<details><summary>{html.escape(str(row['model']))} · {html.escape(MODE_LABEL.get(str(row['mode']), str(row['mode'])))} · {html.escape(str(row['failure']))}</summary>
            <p><strong>观测到的机制：</strong>{html.escape(str(row['observed mechanism']))}</p>
            <pre><code>{html.escape(str(row['output excerpt']))}</code></pre></details>"""
        )

    metrics = {
        "best_qwen32": float(
            summary[
                (summary.model.astype(str) == "Qwen3-32B")
                & (summary["mode"].astype(str) == "native_thinking")
            ].success_rate.iloc[0]
        ),
        "gemma12_native_success": float(
            summary[
                (summary.model.astype(str) == "Gemma4-12B")
                & (summary["mode"].astype(str) == "native_thinking")
            ].success_rate.iloc[0]
        ),
        "gemma12_native_exact": float(
            summary[
                (summary.model.astype(str) == "Gemma4-12B")
                & (summary["mode"].astype(str) == "native_thinking")
            ].exact_rate.iloc[0]
        ),
        "gemma12_strict": float(strict_gemma.registered_success.mean()),
        "qwen17_index": float(
            summary[
                (summary.model.astype(str) == "Qwen3-1.7B")
                & (summary["mode"].astype(str) == "enumeration_index")
            ].success_rate.iloc[0]
        ),
    }

    body = f"""
<div class="meta"><span class="pill">14,500 selected requests</span><span class="pill">29 model × mode cells</span><span class="pill">500 requests per cell</span><span class="pill">V2 + V2.1 current-prompt composite</span></div>
<nav class="toc"><a href="#scope">1. 口径</a><a href="#prompts">2. Prompt</a><a href="#accuracy">3. 正确率</a><a href="#difficulty">4. N 与 L</a><a href="#errors">5. 错误机制</a><a href="#gemma">6. Gemma intervention</a></nav>

<section id="scope"><h2>1. 实验设定与分析口径</h2>
<p>设计为 5 个 passage 长度 \(L\\in\\{{2,3,5,10,20\\}}\)k tokens × 10 个真实 needle 数 \(N\\in\\{{1,2,3,4,5,6,8,10,20,30\\}}\) × 10 seeds（1234–1243），每个 model × mode 单元 500 条。八个目标模型之外，GLM-4-9B-0414 是 GLM-Z1 的 matched non-thinking control，因此表中共有 9 个模型标签、29 个实际运行单元。</p>
<p>当前 prompt 口径不是把所有历史文件简单混合：Index/Bullet 取 V2.1 重跑；Gemma4-12B Direct 取严格 appendix；其余 Direct 和 Native Thinking 取正式 V2。原始 Gemma4-12B Direct 只在第 6 节作为受控 prompt 失败对照。成功严格定义为：计数正确、可解析、最终格式合规且未因长度截断。</p>
<div class="callout warning"><strong>实际 decoding：</strong>冻结结果显示 Direct/Index/Bullet 使用 temperature=0、max tokens 分别为 64/1536；Native Thinking 使用 Qwen/DeepSeek/GLM temperature=0.6、Gemma temperature=1.0，max tokens=4096。因此本报告比较的是“mode + 实际 decoding policy”的整体条件，不能把差异全部归因于一句 prompt 文本。</div>
<div class="conclusion"><strong>本节结论：</strong>主表有 14,500 条且每格平衡；但它是 current-prompt composite。Gemma4-12B Strict Direct 与其他模型旧 Direct 的横向比较必须谨慎，最可靠用途是模型内部和同 stimulus 的 mode 比较。</div></section>

<section id="prompts"><h2>2. 四种 Prompt 与输出约束</h2>
<h3>共用前半段</h3><pre><code>{html.escape(COMMON_PROMPT)}</code></pre>
<h3>Direct（正式 V2，除当前 Gemma4-12B appendix 外）</h3><pre><code>{html.escape(FORMAL_DIRECT)}</code></pre>
<h3>Gemma4-12B Strict Direct appendix</h3><pre><code>{html.escape(STRICT_DIRECT)}</code></pre>
<h3>Index（V2.1 replacement）</h3><pre><code>{html.escape(INDEX_PROMPT)}</code></pre>
<h3>Bullet（V2.1 replacement）</h3><pre><code>{html.escape(BULLET_PROMPT)}</code></pre>
<h3>Native Thinking</h3><pre><code>{html.escape(THINKING_PROMPT)}</code></pre>
<p>Direct/Native 的最终可见文本必须完整匹配 <code>Total: &lt;integer&gt;</code>。Index/Bullet 还要求每个非空行匹配指定列表语法，且 Total 必须位于最后一行。枚举内容与 gold 的 pair precision/recall 作为机制诊断，但注册成功的冻结定义并未额外要求 pair F1=1。</p>
<div class="conclusion"><strong>本节结论：</strong>新版枚举 prompt 已消除“复制 &lt;k&gt;/&lt;city&gt; 占位符”的歧义；Strict Direct 的核心干预是禁止在 64-token 输出预算内先解释或列举。</div></section>

<section id="accuracy"><h2>3. 模型 × Mode 的正确率</h2>
<div class="metric-grid"><div class="metric"><div class="value">{100*metrics['best_qwen32']:.1f}%</div><div class="label">Qwen3-32B Native Thinking</div></div><div class="metric"><div class="value">{100*metrics['gemma12_strict']:.1f}%</div><div class="label">Gemma4-12B Strict Direct</div></div><div class="metric"><div class="value">{100*metrics['qwen17_index']:.1f}%</div><div class="label">Qwen3-1.7B Index</div></div><div class="metric"><div class="value">{100*metrics['gemma12_native_exact']:.1f}% / {100*metrics['gemma12_native_success']:.1f}%</div><div class="label">Gemma4-12B Native exact / strict success</div></div></div>
<figure><img src="figures/prompt/01_accuracy_heatmap.png" alt="Accuracy heatmap"><figcaption><strong>图 1.</strong> 每格为 500 条请求的 registered success。横轴是 prompt mode，纵轴是模型；空格代表该组合未运行，不按 0 处理。</figcaption></figure>
{table_html(best_modes)}
<h3>完整指标</h3>{table_html(display_summary)}
<p class="small">* Mean bias 仅在成功解析出数值的请求上定义；它不是把 parse failure 当作零误差。</p>
<h3>同 stimulus 配对差异</h3>{table_html(paired_show)}
<p>Qwen 随规模增大，从 Direct 到显式/隐式过程的收益总体减小：1.7B 强烈依赖 Native Thinking；4B 的 Native 明显优于 Direct；8B 和 32B 的 Index 与 Native 接近饱和。Gemma 则更偏好 Index：E4B 和 12B 的 Index 分别达到 88.2% 与 95.0%。</p>
<div class="conclusion"><strong>本节结论：</strong>没有单一 mode 在每个模型上都同等占优；但 Index/Native 在中大型 Qwen 上接近 96–98%，Index 是两款 Gemma 的最佳严格输出模式。Qwen3-1.7B 是明显能力边界，prompt 变清楚仍不能阻止枚举退化。</div></section>

<section id="difficulty"><h2>4. Needle 数 N 与 Passage 长度 L</h2>
<figure><img src="figures/prompt/02_accuracy_by_N.png" alt="Accuracy versus N"><figcaption><strong>图 2.</strong> 横轴为真实 needle 数 \(N\)，纵轴为 registered success；每点平均五个长度和十个 seeds。线只连接离散实验水平，不表示连续插值 law。</figcaption></figure>
<figure><img src="figures/prompt/03_accuracy_by_L.png" alt="Accuracy versus L"><figcaption><strong>图 3.</strong> 横轴为 passage 长度 \(L\)（千 tokens），纵轴为 registered success；每点平均十个 N 水平和十个 seeds。</figcaption></figure>
{''.join(detail_blocks)}
<p>多数过程化模式在 N 和 L 增大时下降，但下降方式并不相同：Direct 常在高 N 发生计数偏差或触发隐式列举；Enumeration 在弱模型上会随列表变长出现漏项、幻觉或循环；Native Thinking 在较强 Qwen 上对设计范围最稳。</p>
<div class="conclusion"><strong>本节结论：</strong>N 与 L 都是有效难度轴，但 N 通常更直接控制需要维护的计数状态；L 主要增加搜索距离。二者在弱模型和长输出模式中表现出交互，而非一个简单的“needle density”即可解释全部正确率。</div></section>

<section id="errors"><h2>5. 具体错误原因</h2>
<figure><img src="figures/prompt/04_failure_budget.png" alt="Failure composition"><figcaption><strong>图 4.</strong> 每一横条以全部 500 请求为分母，按互斥优先级拆分失败：截断 → 无 Total → 格式失败 → 数值错误。横轴不是“失败内部比例”，因此条长直接等于总失败率。</figcaption></figure>
{table_html(error_table)}
<h3>代表性原始输出片段</h3>{''.join(example_blocks)}
<p>最主要的三类机制可以被区分：①检索/计数错误——输出格式正确但 Total 错；②输出控制失败——循环、冗长解释或未写 Total；③评估接口不匹配——内容中已有正确 Total，但冻结 parser 仍把额外 channel/text 判为格式错误。下面的因果表述只限于输出证据，不推断模型内部神经机制。</p>
<div class="conclusion"><strong>本节结论：</strong>低分不能统一解释为“不会数数”。Qwen3-1.7B 的枚举失败主要是生成退化，GLM-4 Direct 主要是系统性漏计，Gemma4-12B Native 的低 registered success 则主要是 thought-channel/严格格式不匹配。</div></section>

<section id="gemma"><h2>6. Gemma4-12B Direct 的受控 Prompt 干预</h2>
<figure><img src="figures/prompt/05_gemma_direct_prompt_intervention.png" alt="Gemma direct prompt comparison"><figcaption><strong>图 5.</strong> 同一 500-stimulus 设计中，旧 Direct 与 Strict Direct 的各门槛比例。Truncated 越低越好，其余四项越高越好。</figcaption></figure>
<p>旧 prompt 的 500/500 输出都在 64 tokens 截断，且没有可解析的 Total；典型输出先解释如何扫描，再开始编号列举。Strict Direct 保持数据、模型和 64-token预算不变，只明确禁止解释、公开推理、引用和列举：截断降至 0，解析率与格式率升至 100%，registered success 升至 {100*metrics['gemma12_strict']:.1f}%。剩余失败均为可解析但数值错误。</p>
<div class="conclusion"><strong>本节结论：</strong>Gemma4-12B 原始 Direct 的 0% 是 prompt × 输出预算造成的完全输出控制失败，不支持“模型完全听不懂任务”。严格 prompt 修复了输出通道，但其计数能力在该模式下仍只有约一半请求成功。</div></section>
"""
    report = page(
        "Realistic NiaH V2：Prompt、Mode 与计数正确率",
        "当前 intended prompt 组合的 14,500 条请求；含错误机制与 Gemma4-12B Direct 受控干预",
        body,
        generated,
    )
    (output / "01_prompt_accuracy_report.html").write_text(report, encoding="utf-8")
    return {
        "summary": summary,
        "paired": paired,
        "examples": examples,
        "failure_budget": failure_budget,
    }


def fit_all_bias_laws(
    data: pd.DataFrame,
    output: Path,
) -> dict[str, pd.DataFrame]:
    tables = output / "tables" / "bias"
    tables.mkdir(parents=True, exist_ok=True)
    registry = candidate_registry()
    candidates_rows: list[dict] = []
    oof_store: dict[tuple[str, str, str, str], np.ndarray] = {}
    parsed = data[data["signed_error"].notna()].copy()
    for (model, mode), group in parsed.groupby(["model", "mode"], sort=False):
        for target in ["signed_error", "absolute_error"]:
            for candidate in registry:
                cv = candidate_cv(group, candidate, target)
                candidates_rows.append(
                    {
                        "model": model,
                        "mode": mode,
                        "target": target,
                        "candidate": candidate.name,
                        "label": candidate.label,
                        "formula": candidate.formula,
                        "k": candidate.k,
                        "parsed_n": len(group),
                        "seed_cv_mse": cv["seed_cv_mse"],
                        "seed_cv_se": cv["seed_cv_se"],
                        "request_oof_r2": cv["request_oof_r2"],
                        "cell_crossfit_r2": cv["cell_crossfit_r2"],
                        "cell_fit_r2": cv["cell_fit_r2"],
                        "leave_N_out_r2": cv["leave_N_out_r2"],
                        "leave_L_out_r2": cv["leave_L_out_r2"],
                    }
                )
                oof_store[(model, mode, target, candidate.name)] = cv["oof"]
    candidate_frame = pd.DataFrame(candidates_rows)
    candidate_frame.to_csv(tables / "candidate_law_comparison.csv", index=False)

    selected_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    oof_rows: list[pd.DataFrame] = []
    cell_rows: list[pd.DataFrame] = []
    for (model, mode, target), candidates in candidate_frame.groupby(
        ["model", "mode", "target"], sort=False
    ):
        best_mse = candidates.loc[candidates["seed_cv_mse"].idxmin()]
        tolerance = best_mse["seed_cv_mse"] + (
            best_mse["seed_cv_se"] if np.isfinite(best_mse["seed_cv_se"]) else 0
        )
        eligible = candidates[candidates["seed_cv_mse"] <= tolerance].copy()
        eligible["selection_score"] = (
            eligible["cell_crossfit_r2"].fillna(-1.0)
            - 0.015 * (eligible["k"] - 2).clip(lower=0)
        )
        selected = eligible.sort_values(
            ["selection_score", "seed_cv_mse", "k"],
            ascending=[False, True, True],
        ).iloc[0]
        candidate = next(c for c in registry if c.name == selected["candidate"])
        group = parsed[(parsed["model"] == model) & (parsed["mode"] == mode)].copy()
        cells = (
            group.groupby(["N", "L"], as_index=False)
            .agg(
                target_mean=(target, "mean"),
                target_median=(target, "median"),
                parsed_n=(target, "size"),
                target_sd=(target, "std"),
            )
            .rename(columns={"target_mean": target})
        )
        beta, cell_prediction = fit_raw(candidate, cells, target)
        cells["prediction"] = cell_prediction
        p_value = overall_f_test(candidate, cells, target)
        draws = bootstrap_coefficients(candidate, group, target)
        ci_low = (
            np.quantile(draws, 0.025, axis=0)
            if len(draws)
            else np.full(candidate.k, np.nan)
        )
        ci_high = (
            np.quantile(draws, 0.975, axis=0)
            if len(draws)
            else np.full(candidate.k, np.nan)
        )
        mean_abs_cell = float(
            np.mean(np.abs(cells[target].to_numpy(dtype=float)))
        )
        max_abs_cell = float(np.max(np.abs(cells[target].to_numpy(dtype=float))))
        if target == "signed_error":
            if selected["cell_crossfit_r2"] >= 0.8:
                quality = "strong"
            elif selected["cell_crossfit_r2"] >= 0.5:
                quality = "moderate"
            else:
                quality = "weak"
        else:
            quality = (
                "strong"
                if selected["cell_crossfit_r2"] >= 0.8
                else "moderate"
                if selected["cell_crossfit_r2"] >= 0.5
                else "weak"
            )
        selected_rows.append(
            {
                **selected.to_dict(),
                "overall_f_p": p_value,
                "quality": quality,
                "mean_abs_cell_target": mean_abs_cell,
                "max_abs_cell_target": max_abs_cell,
                "numeric_formula": numeric_formula(
                    candidate,
                    beta,
                    r"\bar b" if target == "signed_error" else r"\overline{|b|}",
                ),
            }
        )
        names = ("intercept",) + candidate.feature_names
        for index, name in enumerate(names):
            coefficient_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "target": target,
                    "candidate": candidate.name,
                    "term": name,
                    "estimate": beta[index],
                    "bootstrap_ci_low": ci_low[index],
                    "bootstrap_ci_high": ci_high[index],
                }
            )
        oof = oof_store[(model, mode, target, candidate.name)]
        oof_frame = group[
            ["model", "mode", "request_id", "stimulus_id", "seed", "N", "L", target]
        ].copy()
        oof_frame["selected_candidate"] = candidate.name
        oof_frame["oof_prediction"] = oof
        oof_rows.append(oof_frame)
        cells.insert(0, "mode", mode)
        cells.insert(0, "model", model)
        cells.insert(2, "target", target)
        cells.insert(3, "selected_candidate", candidate.name)
        cell_rows.append(cells)

    selected_frame = pd.DataFrame(selected_rows)
    signed_mask = selected_frame["target"] == "signed_error"
    selected_frame.loc[signed_mask, "fdr_q"] = bh_adjust(
        selected_frame.loc[signed_mask, "overall_f_p"]
    )
    absolute_mask = selected_frame["target"] == "absolute_error"
    selected_frame.loc[absolute_mask, "fdr_q"] = bh_adjust(
        selected_frame.loc[absolute_mask, "overall_f_p"]
    )
    selected_frame.to_csv(tables / "selected_laws.csv", index=False)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(tables / "selected_coefficients.csv", index=False)
    oof_frame = pd.concat(oof_rows, ignore_index=True)
    oof_frame.to_csv(tables / "selected_oof_predictions.csv", index=False)
    cells_frame = pd.concat(cell_rows, ignore_index=True)
    cells_frame.to_csv(tables / "cell_targets_and_predictions.csv", index=False)
    return {
        "candidates": candidate_frame,
        "selected": selected_frame,
        "coefficients": coefficients,
        "oof": oof_frame,
        "cells": cells_frame,
    }


def fit_relative_fallbacks(
    data: pd.DataFrame,
    signed_selected: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """Search normalized targets only where the raw signed-bias law is weak."""
    working = data.copy()
    working["relative_signed_error"] = working["signed_error"] / working["N"]
    working["relative_absolute_error"] = working["absolute_error"] / working["N"]
    weak_keys = {
        (str(row.model), str(row["mode"]))
        for _, row in signed_selected[
            signed_selected["quality"] == "weak"
        ].iterrows()
    }
    registry = candidate_registry()
    rows: list[dict] = []
    for model, mode in sorted(weak_keys):
        group = working[
            (working.model == model)
            & (working["mode"] == mode)
            & working["signed_error"].notna()
        ].copy()
        raw_r2 = float(
            signed_selected[
                (signed_selected.model == model)
                & (signed_selected["mode"] == mode)
            ]["cell_crossfit_r2"].iloc[0]
        )
        for target in ["relative_signed_error", "relative_absolute_error"]:
            comparisons = []
            for candidate in registry:
                cv = candidate_cv(group, candidate, target)
                comparisons.append(
                    {
                        "candidate": candidate.name,
                        "label": candidate.label,
                        "formula": candidate.formula,
                        "k": candidate.k,
                        **{key: value for key, value in cv.items() if key != "oof"},
                    }
                )
            frame = pd.DataFrame(comparisons)
            best = frame.loc[frame.seed_cv_mse.idxmin()]
            tolerance = best.seed_cv_mse + (
                best.seed_cv_se if np.isfinite(best.seed_cv_se) else 0
            )
            eligible = frame[frame.seed_cv_mse <= tolerance].copy()
            eligible["selection_score"] = (
                eligible.cell_crossfit_r2.fillna(-1)
                - 0.015 * (eligible.k - 2).clip(lower=0)
            )
            selected = eligible.sort_values(
                ["selection_score", "seed_cv_mse", "k"],
                ascending=[False, True, True],
            ).iloc[0]
            candidate = candidate_by_name(str(selected.candidate))
            cells = group.groupby(["N", "L"], as_index=False)[target].mean()
            beta, _ = fit_raw(candidate, cells, target)
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "target": target,
                    "raw_signed_bias_cv_cell_r2": raw_r2,
                    **selected.to_dict(),
                    "numeric_formula": numeric_formula(
                        candidate,
                        beta,
                        r"\overline{b/N}"
                        if target == "relative_signed_error"
                        else r"\overline{|b|/N}",
                    ),
                    "improvement_over_raw_r2": (
                        float(selected.cell_crossfit_r2) - raw_r2
                        if target == "relative_signed_error"
                        else math.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(
        output / "tables" / "bias" / "relative_bias_fallbacks.csv", index=False
    )
    return result


def candidate_by_name(name: str) -> Candidate:
    return next(c for c in candidate_registry() if c.name == name)


def plot_model_bias(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    model: str,
    path: Path,
) -> None:
    modes = [m for m in MODE_ORDER if ((data.model == model) & (data["mode"] == m)).any()]
    fig, axes = plt.subplots(
        len(modes),
        2,
        figsize=(11.4, max(3.2, 2.75 * len(modes))),
        squeeze=False,
    )
    for row_index, mode in enumerate(modes):
        group = data[
            (data.model == model)
            & (data["mode"] == mode)
            & data["signed_error"].notna()
        ].copy()
        record = selected[
            (selected.model == model)
            & (selected["mode"] == mode)
            & (selected.target == "signed_error")
        ].iloc[0]
        candidate = candidate_by_name(record.candidate)
        cells = group.groupby(["N", "L"], as_index=False).signed_error.mean()
        beta, _ = fit_raw(candidate, cells, "signed_error")
        for column_index, axis_name in enumerate(["N", "L"]):
            ax = axes[row_index, column_index]
            marginal = (
                group.groupby(axis_name, as_index=False)
                .agg(
                    observed=("signed_error", "mean"),
                    se=("signed_error", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
                )
                .sort_values(axis_name)
            )
            x = marginal[axis_name].to_numpy(dtype=float)
            x_display = x if axis_name == "N" else x / 1000
            ax.errorbar(
                x_display,
                marginal["observed"],
                yerr=1.96 * marginal["se"],
                fmt="o",
                markersize=4,
                color=MODE_COLORS[mode],
                ecolor="#9AA6B1",
                elinewidth=1,
                capsize=2,
                label="Observed marginal mean ±95% CI",
            )
            if axis_name == "N":
                grid = np.linspace(min(N_LEVELS), max(N_LEVELS), 180)
                prediction_rows = pd.DataFrame(
                    [
                        {"N": value, "L": length}
                        for value in grid
                        for length in L_LEVELS
                    ]
                )
                prediction = (
                    pd.DataFrame(
                        {
                            "N": prediction_rows["N"],
                            "prediction": raw_features(candidate, prediction_rows)
                            @ beta,
                        }
                    )
                    .groupby("N", as_index=False)
                    .prediction.mean()
                )
                px = prediction["N"]
            else:
                grid = np.linspace(min(L_LEVELS), max(L_LEVELS), 180)
                prediction_rows = pd.DataFrame(
                    [{"N": needle, "L": value} for value in grid for needle in N_LEVELS]
                )
                prediction = (
                    pd.DataFrame(
                        {
                            "L": prediction_rows["L"],
                            "prediction": raw_features(candidate, prediction_rows)
                            @ beta,
                        }
                    )
                    .groupby("L", as_index=False)
                    .prediction.mean()
                )
                px = prediction["L"] / 1000
            ax.plot(
                px,
                prediction["prediction"],
                color="#17212B",
                linewidth=2,
                label="Selected-law marginal fit",
            )
            ax.axhline(0, color="#6D7882", linewidth=1, linestyle="--")
            ax.set_xlabel("True needle count N" if axis_name == "N" else "Passage length L (k tokens)")
            ax.set_ylabel("Mean signed bias (predicted − true)")
            ax.set_title(
                f"{MODE_LABEL[mode]} · {record['label']} · CV cell R²={record['cell_crossfit_r2']:.2f}"
            )
            if row_index == 0 and column_index == 0:
                ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{model}: bias response and selected low-complexity law", y=1.01, fontsize=13)
    fig.tight_layout()
    save_figure(fig, path)


def plot_bias_quality(selected: pd.DataFrame, path: Path) -> None:
    signed = selected[selected.target == "signed_error"].copy()
    matrix = (
        signed.pivot(index="model", columns="mode", values="cell_crossfit_r2")
        .reindex(index=MODEL_ORDER, columns=MODE_ORDER)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4), [MODE_LABEL[m] for m in MODE_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.35 or value > 0.72 else "#111", fontweight="bold", fontsize=8.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Seed-cross-fitted cell R²")
    ax.set_title("Goodness of fit for selected signed-bias laws")
    ax.set_xlabel("Prompt mode")
    ax.set_ylabel("Model")
    save_figure(fig, path)


def plot_relative_fallbacks(
    data: pd.DataFrame,
    fallback: pd.DataFrame,
    path: Path,
) -> None:
    selected = fallback[fallback.target == "relative_signed_error"].copy()
    if selected.empty:
        return
    working = data.copy()
    working["relative_signed_error"] = working["signed_error"] / working["N"]
    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(11.2, max(4.0, 2.65 * len(selected))),
        squeeze=False,
    )
    for row_index, (_, record) in enumerate(selected.iterrows()):
        group = working[
            (working.model == record.model)
            & (working["mode"] == record["mode"])
            & working["relative_signed_error"].notna()
        ].copy()
        candidate = candidate_by_name(str(record.candidate))
        cells = group.groupby(["N", "L"], as_index=False).relative_signed_error.mean()
        beta, _ = fit_raw(candidate, cells, "relative_signed_error")
        for column_index, axis_name in enumerate(["N", "L"]):
            ax = axes[row_index, column_index]
            marginal = (
                group.groupby(axis_name, as_index=False)
                .agg(
                    observed=("relative_signed_error", "mean"),
                    se=(
                        "relative_signed_error",
                        lambda x: x.std(ddof=1) / math.sqrt(len(x)),
                    ),
                )
                .sort_values(axis_name)
            )
            x = marginal[axis_name].to_numpy(dtype=float)
            x_display = x if axis_name == "N" else x / 1000
            ax.errorbar(
                x_display,
                marginal.observed,
                yerr=1.96 * marginal.se,
                fmt="o",
                markersize=4,
                color=MODE_COLORS[str(record["mode"])],
                ecolor="#9AA6B1",
                capsize=2,
            )
            if axis_name == "N":
                grid = np.linspace(min(N_LEVELS), max(N_LEVELS), 180)
                prediction_rows = pd.DataFrame(
                    [
                        {"N": value, "L": length}
                        for value in grid
                        for length in L_LEVELS
                    ]
                )
                prediction = (
                    pd.DataFrame(
                        {
                            "N": prediction_rows.N,
                            "prediction": raw_features(candidate, prediction_rows)
                            @ beta,
                        }
                    )
                    .groupby("N", as_index=False)
                    .prediction.mean()
                )
                px = prediction.N
            else:
                grid = np.linspace(min(L_LEVELS), max(L_LEVELS), 180)
                prediction_rows = pd.DataFrame(
                    [
                        {"N": needle, "L": value}
                        for value in grid
                        for needle in N_LEVELS
                    ]
                )
                prediction = (
                    pd.DataFrame(
                        {
                            "L": prediction_rows.L,
                            "prediction": raw_features(candidate, prediction_rows)
                            @ beta,
                        }
                    )
                    .groupby("L", as_index=False)
                    .prediction.mean()
                )
                px = prediction.L / 1000
            ax.plot(px, prediction.prediction, color="#17212B", linewidth=2)
            ax.axhline(0, color="#6D7882", linewidth=1, linestyle="--")
            ax.set_xlabel(
                "True needle count N"
                if axis_name == "N"
                else "Passage length L (k tokens)"
            )
            ax.set_ylabel("Mean relative bias (predicted − true) / N")
            ax.set_title(
                f"{record.model} · {MODE_LABEL[str(record['mode'])]} · "
                f"{record['label']} · CV R²={record.cell_crossfit_r2:.2f}"
            )
    fig.suptitle(
        "Relative-bias fallback for cells with weak raw signed-bias laws",
        y=1.005,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, path)


def common_bilinear_table(data: pd.DataFrame) -> pd.DataFrame:
    candidate = candidate_by_name("N_L_interaction")
    rows = []
    parsed = data[data.signed_error.notna()]
    for (model, mode), group in parsed.groupby(["model", "mode"]):
        cells_mean = group.groupby(["N", "L"], as_index=False).signed_error.mean()
        beta_mean, pred_mean = fit_raw(candidate, cells_mean, "signed_error")
        cells_median = group.groupby(["N", "L"], as_index=False).signed_error.median()
        beta_median, pred_median = fit_raw(candidate, cells_median, "signed_error")
        median_values = cells_median.signed_error.to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "mode": mode,
                "mean_bias_r2": safe_r2(
                    cells_mean.signed_error.to_numpy(dtype=float), pred_mean
                ),
                "median_bias_r2": safe_r2(median_values, pred_median),
                "median_flat_zero": bool(np.allclose(median_values, 0)),
                "mean_bias": group.signed_error.mean(),
                "mae": group.absolute_error.mean(),
                "beta_intercept": beta_mean[0],
                "beta_N": beta_mean[1],
                "beta_L": beta_mean[2],
                "beta_NL": beta_mean[3],
            }
        )
    return pd.DataFrame(rows)


def plot_native_commonality(common: pd.DataFrame, path: Path) -> None:
    native = common[
        (common["mode"] == "native_thinking") & common.model.isin(TARGET_MODELS)
    ].copy()
    fig, ax = plt.subplots(figsize=(9.3, 4.7))
    x = np.arange(len(native))
    mean_r2 = native["mean_bias_r2"].fillna(0).to_numpy()
    median_r2 = native["median_bias_r2"].fillna(0).to_numpy()
    width = 0.37
    ax.bar(x - width / 2, mean_r2, width, color="#396F9F", label="Mean signed bias")
    ax.bar(x + width / 2, median_r2, width, color="#3A8A70", label="Median signed bias")
    for i, flat in enumerate(native["median_flat_zero"]):
        if flat:
            ax.text(i + width / 2, 0.04, "flat 0", rotation=90, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, native.model, rotation=28, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Cell R² for common N + L + NL form")
    ax.set_title("Native Thinking: one shared functional form, model-specific coefficients")
    ax.legend(frameon=False)
    save_figure(fig, path)


def build_bias_report(
    data: pd.DataFrame,
    output: Path,
    generated: str,
) -> dict[str, pd.DataFrame]:
    figures = output / "figures" / "bias"
    tables = output / "tables" / "bias"
    fit = fit_all_bias_laws(data, output)
    selected = fit["selected"]
    signed_selected = selected[selected.target == "signed_error"].copy()
    fallback = fit_relative_fallbacks(data, signed_selected, output)
    common = common_bilinear_table(data)
    common.to_csv(tables / "common_bilinear_diagnostics.csv", index=False)
    plot_bias_quality(selected, figures / "01_selected_law_quality.png")
    for model in MODEL_ORDER:
        safe = (
            model.lower()
            .replace(".", "")
            .replace("/", "-")
            .replace(" ", "-")
        )
        plot_model_bias(data, selected, model, figures / f"model_{safe}.png")
    plot_native_commonality(common, figures / "02_native_commonality.png")
    plot_relative_fallbacks(data, fallback, figures / "03_relative_bias_fallbacks.png")

    signed = signed_selected
    signed["Model"] = signed["model"]
    signed["Mode"] = signed["mode"].map(MODE_LABEL)
    signed["Selected form"] = signed["label"]
    signed["CV cell R²"] = signed["cell_crossfit_r2"].map(lambda x: format_number(x, 3))
    signed["Request OOF R²"] = signed["request_oof_r2"].map(lambda x: format_number(x, 3))
    signed["Leave-N-out R²"] = signed["leave_N_out_r2"].map(lambda x: format_number(x, 3))
    signed["Leave-L-out R²"] = signed["leave_L_out_r2"].map(lambda x: format_number(x, 3))
    signed["FDR q"] = signed["fdr_q"].map(lambda x: f"{x:.2e}" if np.isfinite(x) else "—")
    signed["Fit class"] = signed["quality"].str.title()
    display_signed = signed[
        [
            "Model",
            "Mode",
            "Selected form",
            "CV cell R²",
            "Request OOF R²",
            "Leave-N-out R²",
            "Leave-L-out R²",
            "FDR q",
            "Fit class",
        ]
    ]
    quality_counts = signed["quality"].value_counts()
    strong = int(quality_counts.get("strong", 0))
    moderate = int(quality_counts.get("moderate", 0))
    weak = int(quality_counts.get("weak", 0))

    abs_selected = selected[selected.target == "absolute_error"].copy()
    abs_display = abs_selected[
        ["model", "mode", "label", "cell_crossfit_r2", "request_oof_r2", "quality"]
    ].copy()
    abs_display.columns = [
        "Model",
        "Mode",
        "Selected |bias| form",
        "CV cell R²",
        "Request OOF R²",
        "Fit class",
    ]
    abs_display["Mode"] = abs_display["Mode"].map(MODE_LABEL)
    abs_display["CV cell R²"] = abs_display["CV cell R²"].map(lambda x: format_number(x, 3))
    abs_display["Request OOF R²"] = abs_display["Request OOF R²"].map(lambda x: format_number(x, 3))

    fallback_display = fallback.copy()
    fallback_display["Model"] = fallback_display["model"]
    fallback_display["Mode"] = fallback_display["mode"].map(MODE_LABEL)
    fallback_display["Normalized target"] = fallback_display["target"].map(
        {
            "relative_signed_error": "mean b/N",
            "relative_absolute_error": "mean |b|/N",
        }
    )
    fallback_display["Selected form"] = fallback_display["label"]
    fallback_display["Raw-bias CV R²"] = fallback_display[
        "raw_signed_bias_cv_cell_r2"
    ].map(lambda x: format_number(x, 3))
    fallback_display["Normalized CV R²"] = fallback_display[
        "cell_crossfit_r2"
    ].map(lambda x: format_number(x, 3))
    fallback_display["Leave-N-out R²"] = fallback_display["leave_N_out_r2"].map(
        lambda x: format_number(x, 3)
    )
    fallback_display["Leave-L-out R²"] = fallback_display["leave_L_out_r2"].map(
        lambda x: format_number(x, 3)
    )
    fallback_display = fallback_display[
        [
            "Model",
            "Mode",
            "Normalized target",
            "Selected form",
            "Raw-bias CV R²",
            "Normalized CV R²",
            "Leave-N-out R²",
            "Leave-L-out R²",
        ]
    ]

    native = common[
        (common["mode"] == "native_thinking") & common.model.isin(TARGET_MODELS)
    ].copy()
    native_display = native[
        ["model", "mean_bias_r2", "median_bias_r2", "median_flat_zero", "mean_bias", "mae"]
    ].copy()
    native_display.columns = [
        "Model",
        "Mean-bias R²",
        "Median-bias R²",
        "Median flat zero",
        "Mean bias",
        "MAE",
    ]
    for column in ["Mean-bias R²", "Median-bias R²", "Mean bias", "MAE"]:
        native_display[column] = native_display[column].map(lambda x: format_number(x, 3))

    qwen_common = common[common.model.isin(QWEN_MODELS)].copy()
    qwen_common.to_csv(tables / "qwen_common_bilinear_by_mode.csv", index=False)

    law_details = []
    for _, row in signed.iterrows():
        coefficients = fit["coefficients"][
            (fit["coefficients"].model == row.model)
            & (fit["coefficients"]["mode"] == row["mode"])
            & (fit["coefficients"].target == "signed_error")
        ]
        coefficient_table = coefficients[
            ["term", "estimate", "bootstrap_ci_low", "bootstrap_ci_high"]
        ].copy()
        coefficient_table.columns = ["Term", "Estimate", "Bootstrap 2.5%", "Bootstrap 97.5%"]
        for column in ["Estimate", "Bootstrap 2.5%", "Bootstrap 97.5%"]:
            coefficient_table[column] = coefficient_table[column].map(
                lambda x: format_number(x, 4)
            )
        law_details.append(
            f"""<details><summary>{html.escape(str(row.model))} · {html.escape(MODE_LABEL[str(row['mode'])])} · CV cell R²={row.cell_crossfit_r2:.3f}</summary>
            <div class="equation">\\[{row.numeric_formula}\\]</div>
            <p>候选形式：\\({row.formula}\\)。\(L_k=L/1000\)。Seed-cross-fitted cell R²={row.cell_crossfit_r2:.3f}；request OOF R²={row.request_oof_r2:.3f}；leave-N-out R²={row.leave_N_out_r2:.3f}；leave-L-out R²={row.leave_L_out_r2:.3f}；探索性 FDR q={row.fdr_q:.2e}。</p>
            {table_html(coefficient_table)}
            </details>"""
        )

    model_sections = []
    for model in MODEL_ORDER:
        safe = model.lower().replace(".", "").replace("/", "-").replace(" ", "-")
        subset = signed[signed.model == model]
        if subset.empty:
            continue
        strong_modes = subset[subset.quality == "strong"]["mode"].map(MODE_LABEL).tolist()
        weak_modes = subset[subset.quality == "weak"]["mode"].map(MODE_LABEL).tolist()
        conclusion_parts = []
        if strong_modes:
            conclusion_parts.append("强拟合：" + "、".join(strong_modes))
        if weak_modes:
            conclusion_parts.append(
                "未发现稳定低维 mean-bias law：" + "、".join(weak_modes)
            )
        conclusion_text = "；".join(conclusion_parts) or "以中等强度关系为主"
        model_sections.append(
            f"""<h3>{html.escape(model)}</h3>
            <figure><img src="figures/bias/model_{safe}.png" alt="{html.escape(model)} bias fits"><figcaption><strong>{html.escape(model)}。</strong> 每行对应一个已运行 mode。左列横轴为 \(N\)，右列横轴为 \(L\)（千 tokens）；纵轴统一为解析样本的 mean signed bias。散点是对另一设计轴和 seeds 边际平均后的观测值，误差棒是 95% CI；实线是所选二维 law 在另一轴取实验分布平均后的边际曲线。</figcaption></figure>
            <div class="conclusion"><strong>{html.escape(model)} 结论：</strong>{html.escape(conclusion_text)}。低 R² 不自动等于性能差：若 bias 几乎恒为 0，目标方差过小也会使 R² 不稳定。</div>"""
        )

    body = f"""
<div class="meta"><span class="pill">29 separate fits</span><span class="pill">Primary target: signed bias</span><span class="pill">Finite low-complexity search</span><span class="pill">Leave-one-seed cross-fitting</span></div>
<nav class="toc"><a href="#definitions">1. 定义</a><a href="#search">2. 搜索与验证</a><a href="#results">3. 29 个结果</a><a href="#models">4. 曲线</a><a href="#universal">5. 普适性</a><a href="#limits">6. 边界</a></nav>

<section id="definitions"><h2>1. Bias 与分析目标</h2>
<p>对成功解析出整数的请求 \(i\)，定义 signed bias 与 absolute deviation：</p>
<div class="equation">\\[b_i=\\widehat N_i-N_i,\\qquad a_i=|b_i|.\\]</div>
<p>\(b_i&lt;0\) 表示漏计，\(b_i&gt;0\) 表示过计。主拟合目标是在固定模型、mode、\(N\)、\(L\) 后对十个 seeds 求条件均值：</p>
<div class="equation">\\[\\bar b_{{N,L}}=\\frac{{1}}{{m_{{N,L}}}}\\sum_{{i:\\,(N_i,L_i)=(N,L),\\,\\mathrm{{parsed}}}} b_i.\\]</div>
<p>其中 \(m_{{N,L}}\\le 10\)。Parse failure 没有数值 \(\widehat N\)，因此 bias 在数学上未定义；报告既不把它设为 0，也不删除其存在，而是在第一份报告中单列 parse coverage。Absolute-deviation law \(\overline{{|b|}}_{{N,L}}\) 是次级诊断，用于识别正负误差相互抵消的情形。</p>
<div class="conclusion"><strong>本节结论：</strong>本报告估计的是“成功解析条件下的计数偏差”，不是包含无答案请求的总体效用；总体成功率必须与第一份报告合读。</div></section>

<section id="search"><h2>2. 有边界的候选 Law 搜索</h2>
<p>为保持横纵轴和解释简单，候选变量只来自 \(N\)、\(L_k=L/1000\)、\(\log_2N\)、\(\log_2L_k\)、\(N/L_k\)、\(L_k/N\)、\(NL_k\)，以及低阶加法、一个交互项、一个预先固定断点（\(N=10\) 或 \(L_k=5\)）和至多二次项。没有逐点多项式、模型 ID 特征或事后删点。</p>
<p>每个模型 × mode 单独搜索。每次留出一个 seed，使用其余九个 seeds 拟合并预测留出 seed；候选必须落在最低 seed-CV MSE 的 one-standard-error 集合内，再最大化“cell cross-fitted R² − 每个额外参数 0.015”的简洁性分数。报告同时给出 request-level OOF R²、leave-one-N-level-out 与 leave-one-L-level-out R²。F-test/FDR 仅作为探索性描述，因为公式经过选择，不能替代预注册确认实验。</p>
<div class="callout warning"><strong>关于“拟合优度要足够高”：</strong>本分析不会通过高阶插值强行让 29 格都得到高 R²。门槛定义为：CV cell R²≥0.8 强、0.5–0.8 中等、&lt;0.5 弱。当前得到 {strong} 个强、{moderate} 个中等、{weak} 个弱 mean-bias law；弱结果被保留为有效负结论。</div>
<div class="conclusion"><strong>本节结论：</strong>选模依赖 held-out seed，而不是训练集曲线漂亮程度；N/L 外推指标单独报告，防止把同网格内复现误称为外推 law。</div></section>

<section id="results"><h2>3. 29 个 Signed-bias Law</h2>
<figure><img src="figures/bias/01_selected_law_quality.png" alt="Selected bias law quality"><figcaption><strong>图 1.</strong> 横轴为 mode，纵轴为模型；每格是所选 signed-bias law 的 seed-cross-fitted cell R²。颜色和数值衡量 50 个 \((N,L)\) 条件均值是否被简单曲面解释，而不是单条请求能否被准确预测。</figcaption></figure>
{table_html(display_signed)}
<details><summary>展开全部 29 条数值公式与 bootstrap 区间</summary>{''.join(law_details)}</details>
<h3>Absolute deviation 的补充拟合</h3>{table_html(abs_display)}
<p>当 signed bias R² 很低但 \(|b|\) R² 较高时，主要机制不是“没有难度关系”，而是随着 N/L 变难，误差幅度增加但方向在不同 seeds 间正负抵消。Qwen3-8B Direct、Qwen3-4B Bullet/Index 是典型例子。</p>
<h3>Raw bias 较弱时的 b/N fallback</h3>
<p>按用户建议，只对 raw signed-bias CV R²&lt;0.5 的单元追加相对偏差 \(r=b/N\) 与相对绝对偏差 \(|b|/N\) 搜索。该步骤不改变 raw-bias 主结果。</p>
{table_html(fallback_display)}
<figure><img src="figures/bias/03_relative_bias_fallbacks.png" alt="Relative bias fallback"><figcaption><strong>图 2.</strong> 仅展示 raw signed-bias law 较弱的单元。横轴分别为 \(N\) 与 \(L\)，纵轴是 mean relative bias \(b/N\)；散点和 95% CI 为观测边际均值，实线为归一化目标的所选二维 law。</figcaption></figure>
<p>归一化最有帮助的是 GLM-4 Index：CV cell R² 从约 0.33 提升到约 0.59，说明其误差更接近“相对比例偏差”而非固定整数偏差。Qwen3-4B Index 提升到约 0.49，仍只到边界水平；其余弱单元没有因除以 N 而变成可靠 law。</p>
<div class="conclusion"><strong>本节结论：</strong>系统性漏计/过计的单元通常能得到强二维 law；接近无偏或误差方向随机的单元不应强行拟合 signed bias。此时 \(|b|\) 常比 \(b\) 稳定，而 \(b/N\) 只对少数比例型偏差单元有效。</div></section>

<section id="models"><h2>4. 各模型、各 Mode 的散点与拟合曲线</h2>
{''.join(model_sections)}
</section>

<section id="universal"><h2>5. 是否存在普适形式？</h2>
<p>最清晰的共同结构出现在 Native Thinking。固定同一双线性形式、只允许每个模型拥有不同参数：</p>
<div class="equation">\\[\\bar b_m(N,L)=\\alpha_m+\\beta_{{N,m}}N+\\beta_{{L,m}}L_k+\\beta_{{NL,m}}NL_k.\\]</div>
<figure><img src="figures/bias/02_native_commonality.png" alt="Native thinking commonality"><figcaption><strong>图 3.</strong> 八个目标模型的 Native Thinking 均使用同一 \(N+L+NL\) 形式。蓝柱拟合 cell mean bias；绿柱拟合 cell median bias。标为 “flat 0” 表示 50 个条件的 median bias 全部为 0，此时 R² 无定义，而不是拟合失败。</figcaption></figure>
{table_html(native_display)}
<p>四款 Qwen 的 mean-bias cell R² 分别约为 0.943、0.845、0.841、0.708；Qwen3-8B 与 32B 的 50 个条件 median bias 全部为 0。扩展到八个目标模型时，median 形式在所有非平坦模型上约为 0.774–0.931，DeepSeek 的 mean bias 是唯一明显例外，但其 median bias 仍可由同一形式解释（约 0.83）。</p>
<p>Direct、Index、Bullet 没有同样强的跨四款 Qwen 非平凡 law。更合理的共同描述是能力分区：1.7B 枚举出现正向爆炸偏差；4B/8B/32B 的 Index 逐渐进入近零 median-bias 区；Direct 的误差方向跨规模不稳定。</p>
<div class="conclusion"><strong>本节结论：</strong>可以支持的普适结果是“Native Thinking 的条件中心偏差具有 \(N+L+NL\) 双线性形状，参数随模型变化”；它至少覆盖四款 Qwen，并在 robust median 上扩展到全部八个目标模型。不能支持一个对所有 mode 都共享相同参数或相同误差方向的 law。</div></section>

<section id="limits"><h2>6. 适用范围与不能推出的结论</h2>
<p>这些 law 只在 \(N=1\)–30、\(L=2\)k–20k、当前城市-分数模板和固定 decoding 下成立；leave-level-out 结果显示，某些高 cell R² law 对未见 N 或 L 水平的外推会显著变差。公式是经验响应面，不证明模型内部真的执行乘法、密度估计或任何特定算法。</p>
<p>此外，Gemma4-12B Direct 使用 Strict appendix，Native Thinking 的大量严格格式失败仍可解析出数字；因此 bias law 与 registered success 描述的是不同层面。若要做确认性研究，应冻结 \(N+L+NL\) Native law，在新 seeds、不同 haystack 文体和更宽 N/L 范围上复验。</p>
<div class="conclusion"><strong>本节结论：</strong>目前最可靠的是设计范围内的条件均值/中位数规律，而不是无限外推的 scaling law；下一轮最有信息量的实验是对 Native 双线性形状做独立确认。</div></section>
"""
    report = page(
        "Realistic NiaH V2：Bias 与 N、L 的经验 Law",
        "29 个模型 × mode 独立拟合；散点、边际曲线、交叉验证与跨模型共同结构",
        body,
        generated,
    )
    (output / "02_bias_law_report.html").write_text(report, encoding="utf-8")
    return {**fit, "common": common, "fallback": fallback}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(output: Path, input_csv: Path, args: argparse.Namespace) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "analysis_manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    manifest = {
        "analysis": "realistic_niah_v2_prompt_accuracy_and_bias_laws",
        "generated_at": args.generated,
        "input_compact_csv": str(input_csv.resolve()),
        "input_rows": int(pd.read_csv(input_csv, usecols=["request_id"]).shape[0]),
        "selection": {
            "cells": 29,
            "requests_per_cell": 500,
            "enumeration": "V2.1 replacement",
            "gemma4_12b_direct": "V2.1 strict appendix substituted in the current-prompt 29-cell composite",
            "remaining_modes": "V2 formal",
        },
        "method": {
            "bias": "predicted_count - gold_count, parsed requests only",
            "primary_cell_target": "mean signed bias over seeds",
            "candidate_search": "finite low-complexity registry in build_v2_reports.py",
            "selection": "leave-one-seed CV; one-SE eligibility; cross-fitted cell R2 minus 0.015 per extra parameter",
            "bootstrap": "400 seed-block resamples",
        },
        "files": files,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksums = [
        f"{item['sha256']}\t{item['bytes']}\t{item['path']}" for item in files
    ]
    (output / "SHA256SUMS.tsv").write_text(
        "sha256\tbytes\tpath\n" + "\n".join(checksums) + "\n", encoding="utf-8"
    )


def write_readme(output: Path) -> None:
    text = """# Realistic NiaH V2 analysis reports

This directory contains two complementary reports:

- `01_prompt_accuracy_report.html`: prompt/mode accuracy and observed failure mechanisms.
- `02_bias_law_report.html`: model-by-mode signed-bias laws and cross-model commonality.

The current-prompt composite contains exactly 29 cells × 500 requests. V2.1
enumeration replacements supersede the old enumeration rows. The strict
Gemma4-12B Direct appendix supplies the current Direct cell for numerical-bias
analysis; its original V2 Direct run remains a prompt-failure comparator only.

Rebuild:

```powershell
python scripts/build_v2_reports.py --input tables/request_level_compact.csv --output .
```

All formulas are exploratory empirical response surfaces. See the reports for
the exact success, bias, cross-validation, and extrapolation definitions.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path(
            r"C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting"
            r"\Realistic_CoT_NiaH_Count\exports"
            r"\Realistic_CoT_NiaH_Count_20260726_v2"
        ),
    )
    parser.add_argument("--generated", default="2026-07-26")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    data = ensure_bool(
        data,
        [
            "parse_ok",
            "exact_count",
            "format_ok",
            "enumeration_format_ok",
            "registered_success",
            "truncated",
        ],
    )
    if len(data) != 14_500:
        raise ValueError(f"Expected 14,500 rows, got {len(data):,}")
    cell_counts = data.groupby(["model", "mode"]).size()
    if len(cell_counts) != 29 or not (cell_counts == 500).all():
        raise ValueError(f"Expected 29 balanced cells, got {cell_counts.to_dict()}")
    output_tables = output / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, output_tables / "request_level_compact.csv")
    scripts = output / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), scripts / Path(__file__).name)
    set_plot_style()
    build_prompt_report(data, args.v2_root, output, args.generated)
    build_bias_report(data, output, args.generated)
    write_readme(output)
    build_manifest(output, args.input, args)
    print(output / "01_prompt_accuracy_report.html")
    print(output / "02_bias_law_report.html")


if __name__ == "__main__":
    main()
