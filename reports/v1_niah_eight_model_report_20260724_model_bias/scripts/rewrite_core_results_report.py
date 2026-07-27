"""Rewrite the canonical Realistic NiaH report around its core empirical questions.

The report-only workflow reads the preserved 6,300-request CSV and existing
fit outputs. It does not alter request-level data, frozen prompts, fit tables,
or original figures. New summary tables and figures are written beside the
existing artifacts, and the root ``report.html`` is replaced atomically.

Narrative order:
1. exact accuracy split by query placement;
2. within-model prompt/reasoning-mode comparisons;
3. low-accuracy failure mechanisms;
4. retained empirical-law results and reproducibility links.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


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
MODEL_EXPECTED = {
    "Qwen3-8B": 900,
    "Qwen3-1.7B": 900,
    "Qwen3-32B": 900,
    "Gemma4-E4B": 900,
    "Gemma4-12B": 900,
    "OLMo-Hybrid-7B": 600,
    "Llama3.1-8B": 600,
    "Llama3.2-3B": 600,
}
MODE_ORDER = ["direct", "enumeration", "native_thinking"]
MODE_LABEL = {
    "direct": "Direct",
    "enumeration": "Enumeration",
    "native_thinking": "Native thinking",
}
ORDER_ORDER = ["query_first", "query_last"]
ORDER_LABEL = {"query_first": "Query first", "query_last": "Query last"}
ERROR_ORDER = ["correct", "undercount", "overcount", "parse_failure", "truncation"]
ERROR_LABEL = {
    "correct": "Correct",
    "undercount": "Undercount",
    "overcount": "Overcount",
    "parse_failure": "Format / parse",
    "truncation": "Truncation",
}
ERROR_COLORS = {
    "correct": "#2f7d67",
    "undercount": "#5072a7",
    "overcount": "#d07443",
    "parse_failure": "#a2618b",
    "truncation": "#7b7f8c",
}
DIAGNOSTIC_MODELS = [
    "Qwen3-1.7B",
    "Gemma4-12B",
    "OLMo-Hybrid-7B",
    "Llama3.1-8B",
    "Llama3.2-3B",
]
SEED = 20260725
BOOTSTRAP_REPLICATES = 5000
REPORT_MARKER = "CORE_RESULTS_REPORT_V3"
QUERY_ANOMALY_THRESHOLD = 0.10
MODE_ANOMALY_THRESHOLD = 0.20
INTERFACE_FAILURE_THRESHOLD = 0.25
AGGREGATION_FAILURE_THRESHOLD = 0.20

DIRECT_TASK_BLOCK = """The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

How many city-score audit records are in the passage?
In the final answer, output exactly one line in this form:
Total: <integer>"""

ENUMERATION_TASK_BLOCK = """The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

Find every city-score audit record in the passage. In passage order, output
one record per line as:
<k>. <city>: <score>
where k starts at 1 and increases by 1.
Then output one final line:
Total: <integer>
Do not include any other text."""

WRAPPER_DESCRIPTIONS = {
    "Qwen3-8B": (
        "<|im_start|>user … <|im_end|> → <|im_start|>assistant",
        "No injected system message. With thinking off, the assistant prefix "
        "contains an empty <think>…</think> block; native thinking leaves it open.",
    ),
    "Qwen3-1.7B": (
        "<|im_start|>user … <|im_end|> → <|im_start|>assistant",
        "No injected system message. With thinking off, the assistant prefix "
        "contains an empty <think>…</think> block; native thinking leaves it open.",
    ),
    "Qwen3-32B": (
        "<|im_start|>user … <|im_end|> → <|im_start|>assistant",
        "No injected system message. With thinking off, the assistant prefix "
        "contains an empty <think>…</think> block; native thinking leaves it open.",
    ),
    "Gemma4-E4B": (
        "<bos><|turn>user … <turn|> → <|turn>model",
        "Official Gemma template; native thinking adds its thought-channel "
        "template behavior.",
    ),
    "Gemma4-12B": (
        "<bos><|turn>user … <turn|> → <|turn>model",
        "Official Gemma template; the frozen direct rendering opens the "
        "thought channel after the model marker.",
    ),
    "OLMo-Hybrid-7B": (
        "ChatML system → user → assistant",
        "Tokenizer injects a function-calling system message stating that no "
        "functions are available.",
    ),
    "Llama3.1-8B": (
        "<|begin_of_text|> + system/user/assistant headers",
        "Tokenizer injects knowledge date and frozen Today Date: 26 Jul 2024.",
    ),
    "Llama3.2-3B": (
        "<|begin_of_text|> + system/user/assistant headers",
        "Tokenizer injects knowledge date and frozen Today Date: 24 Jul 2026.",
    ),
}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def save_figure_atomic(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "rewrite_core_results_report.py"},
    )
    plt.close(fig)
    atomic_write_bytes(path, buffer.getvalue())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refresh_checksums(base: Path, manifest: Path) -> None:
    records: list[str] = []
    for path in sorted(base.rglob("*"), key=lambda p: str(p.relative_to(base)).lower()):
        if not path.is_file() or path.resolve() == manifest.resolve():
            continue
        records.append(f"{sha256_file(path)}\t{path.relative_to(base)}")
    atomic_write_text(manifest, "\n".join(records) + "\n")


def configure_plot_style() -> None:
    preferred = ["Microsoft YaHei", "Segoe UI", "Arial", "DejaVu Sans"]
    chosen = "DejaVu Sans"
    for name in preferred:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen, "DejaVu Sans"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def validate_requests(data: pd.DataFrame) -> None:
    required = {
        "request_id",
        "stimulus_id",
        "model_label",
        "prompt_mode",
        "query_order",
        "exact_correct",
        "parse_success",
        "format_failure",
        "truncated",
        "error_category",
        "target_passage_tokens",
        "num_needles",
        "gold_count",
        "predicted_count",
        "missing_pairs_n",
        "hallucinated_pairs_n",
        "duplicate_listed_pairs_n",
        "listed_records_n",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"request table is missing columns: {missing}")
    if len(data) != 6300:
        raise ValueError(f"expected 6300 requests, found {len(data)}")
    if data["request_id"].nunique() != 6300:
        raise ValueError("request_id values are not unique")
    counts = data.groupby("model_label").size().to_dict()
    if counts != MODEL_EXPECTED:
        raise ValueError(f"unexpected per-model counts: {counts}")
    if set(data["model_label"]) != set(MODEL_ORDER):
        raise ValueError("model set does not match the frozen eight-model registry")
    if set(data["query_order"]) != set(ORDER_ORDER):
        raise ValueError("query order set is incomplete")
    expected_modes = {
        model: (set(MODE_ORDER) if MODEL_EXPECTED[model] == 900 else {"direct", "enumeration"})
        for model in MODEL_ORDER
    }
    actual_modes = data.groupby("model_label")["prompt_mode"].agg(lambda x: set(x)).to_dict()
    if actual_modes != expected_modes:
        raise ValueError(f"unexpected mode registry: {actual_modes}")
    if not set(data["error_category"]).issubset(set(ERROR_ORDER)):
        raise ValueError(f"unknown error categories: {set(data['error_category']) - set(ERROR_ORDER)}")
    for columns in (["model_label", "query_order"], ["model_label", "prompt_mode", "query_order"]):
        grouped = data.groupby(columns).size()
        if (grouped <= 0).any():
            raise ValueError(f"empty registered group for {columns}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def cluster_bootstrap_interval(
    pairs: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    cluster_values = pairs.groupby(group_column, sort=False)[value_column].mean().to_numpy(dtype=float)
    if len(cluster_values) == 0:
        return (math.nan, math.nan)
    draws = rng.choice(cluster_values, size=(replicates, len(cluster_values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def query_order_pair_frame(data: pd.DataFrame, model: str, mode: str | None = None) -> pd.DataFrame:
    subset = data[data["model_label"].eq(model)].copy()
    index = ["prompt_mode", "stimulus_id"]
    if mode is not None:
        subset = subset[subset["prompt_mode"].eq(mode)].copy()
        index = ["stimulus_id"]
    pivot = (
        subset.pivot(index=index, columns="query_order", values="exact_correct")
        .dropna(subset=ORDER_ORDER)
        .reset_index()
    )
    pivot["difference_last_minus_first"] = pivot["query_last"] - pivot["query_first"]
    return pivot


def mode_pair_frame(data: pd.DataFrame, model: str, comparison: str) -> pd.DataFrame:
    subset = data[
        data["model_label"].eq(model) & data["prompt_mode"].isin(["direct", comparison])
    ].copy()
    pivot = (
        subset.pivot(
            index=["query_order", "stimulus_id"],
            columns="prompt_mode",
            values="exact_correct",
        )
        .dropna(subset=["direct", comparison])
        .reset_index()
    )
    pivot["difference_vs_direct"] = pivot[comparison] - pivot["direct"]
    return pivot


def summarize_accuracy(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    model_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = data[data["model_label"].eq(model)]
        pairs = query_order_pair_frame(data, model)
        ci_low, ci_high = cluster_bootstrap_interval(
            pairs,
            group_column="stimulus_id",
            value_column="difference_last_minus_first",
            rng=rng,
        )
        first = subset[subset["query_order"].eq("query_first")]
        last = subset[subset["query_order"].eq("query_last")]
        successes = int(subset["exact_correct"].sum())
        overall_low, overall_high = wilson_interval(successes, len(subset))
        model_rows.append(
            {
                "model_label": model,
                "requests": len(subset),
                "exact_correct": successes,
                "overall_accuracy": successes / len(subset),
                "overall_ci95_low": overall_low,
                "overall_ci95_high": overall_high,
                "query_first_requests": len(first),
                "query_first_accuracy": first["exact_correct"].mean(),
                "query_last_requests": len(last),
                "query_last_accuracy": last["exact_correct"].mean(),
                "delta_query_last_minus_first": pairs["difference_last_minus_first"].mean(),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "paired_units": len(pairs),
                "first_only_correct": int(
                    ((pairs["query_first"] == 1) & (pairs["query_last"] == 0)).sum()
                ),
                "last_only_correct": int(
                    ((pairs["query_first"] == 0) & (pairs["query_last"] == 1)).sum()
                ),
            }
        )
    model_query = pd.DataFrame(model_rows)

    mode_query_rows: list[dict[str, Any]] = []
    order_effect_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for mode in MODE_ORDER:
            subset = data[
                data["model_label"].eq(model) & data["prompt_mode"].eq(mode)
            ].copy()
            if subset.empty:
                continue
            pairs = query_order_pair_frame(data, model, mode)
            ci_low, ci_high = cluster_bootstrap_interval(
                pairs,
                group_column="stimulus_id",
                value_column="difference_last_minus_first",
                rng=rng,
            )
            row: dict[str, Any] = {
                "model_label": model,
                "prompt_mode": mode,
                "requests": len(subset),
                "exact_correct": int(subset["exact_correct"].sum()),
                "accuracy": subset["exact_correct"].mean(),
                "parse_success_rate": subset["parse_success"].mean(),
                "format_failure_rate": subset["format_failure"].mean(),
                "truncation_rate": subset["truncated"].mean(),
            }
            for order in ORDER_ORDER:
                order_data = subset[subset["query_order"].eq(order)]
                row[f"{order}_requests"] = len(order_data)
                row[f"{order}_accuracy"] = order_data["exact_correct"].mean()
                row[f"{order}_parse_success_rate"] = order_data["parse_success"].mean()
                row[f"{order}_format_failure_rate"] = order_data["format_failure"].mean()
                row[f"{order}_truncation_rate"] = order_data["truncated"].mean()
            mode_query_rows.append(row)
            order_effect_rows.append(
                {
                    "model_label": model,
                    "prompt_mode": mode,
                    "paired_stimuli": len(pairs),
                    "query_first_accuracy": pairs["query_first"].mean(),
                    "query_last_accuracy": pairs["query_last"].mean(),
                    "delta_query_last_minus_first": pairs[
                        "difference_last_minus_first"
                    ].mean(),
                    "delta_ci95_low": ci_low,
                    "delta_ci95_high": ci_high,
                    "first_only_correct": int(
                        ((pairs["query_first"] == 1) & (pairs["query_last"] == 0)).sum()
                    ),
                    "last_only_correct": int(
                        ((pairs["query_first"] == 0) & (pairs["query_last"] == 1)).sum()
                    ),
                }
            )
    mode_query = pd.DataFrame(mode_query_rows)
    order_effects = pd.DataFrame(order_effect_rows)

    mode_effect_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        registered = set(data.loc[data["model_label"].eq(model), "prompt_mode"])
        for comparison in ["enumeration", "native_thinking"]:
            if comparison not in registered:
                continue
            pairs = mode_pair_frame(data, model, comparison)
            ci_low, ci_high = cluster_bootstrap_interval(
                pairs,
                group_column="stimulus_id",
                value_column="difference_vs_direct",
                rng=rng,
            )
            row = {
                "model_label": model,
                "comparison_mode": comparison,
                "paired_requests": len(pairs),
                "direct_accuracy": pairs["direct"].mean(),
                "comparison_accuracy": pairs[comparison].mean(),
                "delta_vs_direct": pairs["difference_vs_direct"].mean(),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
            }
            for order in ORDER_ORDER:
                order_pairs = pairs[pairs["query_order"].eq(order)]
                row[f"{order}_direct_accuracy"] = order_pairs["direct"].mean()
                row[f"{order}_comparison_accuracy"] = order_pairs[comparison].mean()
                row[f"{order}_delta_vs_direct"] = order_pairs[
                    "difference_vs_direct"
                ].mean()
            mode_effect_rows.append(row)
    mode_effects = pd.DataFrame(mode_effect_rows)

    return {
        "model_query": model_query,
        "mode_query": mode_query,
        "order_effects": order_effects,
        "mode_effects": mode_effects,
    }


def summarize_failures(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (model, mode, order), subset in data.groupby(
        ["model_label", "prompt_mode", "query_order"], sort=False
    ):
        counts = subset["error_category"].value_counts()
        row: dict[str, Any] = {
            "model_label": model,
            "prompt_mode": mode,
            "query_order": order,
            "requests": len(subset),
        }
        for category in ERROR_ORDER:
            count = int(counts.get(category, 0))
            row[f"{category}_count"] = count
            row[f"{category}_rate"] = count / len(subset)
        rows.append(row)
    failure_budget = pd.DataFrame(rows)

    model_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = data[data["model_label"].eq(model)]
        counts = subset["error_category"].value_counts()
        parsed = int(subset["parse_success"].sum())
        row = {
            "model_label": model,
            "requests": len(subset),
            "overall_accuracy": subset["exact_correct"].mean(),
            "parse_success_rate": subset["parse_success"].mean(),
            "exact_given_parsed": subset["exact_correct"].sum() / parsed if parsed else math.nan,
        }
        for category in ERROR_ORDER:
            row[f"{category}_rate"] = counts.get(category, 0) / len(subset)
        model_rows.append(row)
    model_diagnostics = pd.DataFrame(model_rows)

    enum = data[data["prompt_mode"].eq("enumeration")].copy()
    enum["complete_retrieval_wrong_total"] = (
        enum["error_category"].eq("overcount")
        & enum["missing_pairs_n"].fillna(999).eq(0)
        & enum["hallucinated_pairs_n"].fillna(999).eq(0)
        & enum["duplicate_listed_pairs_n"].fillna(999).eq(0)
        & enum["listed_records_n"].eq(enum["gold_count"])
    )
    enum_rows: list[dict[str, Any]] = []
    for (model, order), subset in enum.groupby(["model_label", "query_order"], sort=False):
        overcount = subset["error_category"].eq("overcount")
        complete = subset["complete_retrieval_wrong_total"]
        enum_rows.append(
            {
                "model_label": model,
                "query_order": order,
                "enumeration_requests": len(subset),
                "enumeration_accuracy": subset["exact_correct"].mean(),
                "overcount_count": int(overcount.sum()),
                "overcount_rate": overcount.mean(),
                "complete_retrieval_wrong_total_count": int(complete.sum()),
                "complete_retrieval_wrong_total_rate": complete.mean(),
                "share_of_overcounts_with_complete_retrieval": (
                    complete.sum() / overcount.sum() if overcount.sum() else math.nan
                ),
            }
        )
    enum_aggregation = pd.DataFrame(enum_rows)

    low = model_diagnostics[
        model_diagnostics["model_label"].isin(DIAGNOSTIC_MODELS)
    ].copy()
    diagnoses = {
        "Qwen3-1.7B": (
            "Parsing is intact; errors are primarily numeric under/over-counting. "
            "Native thinking materially improves accuracy."
        ),
        "Gemma4-12B": (
            "A mode-specific generation-budget collapse: direct and part of native "
            "thinking truncate, while parsed numeric outputs are usually exact."
        ),
        "OLMo-Hybrid-7B": (
            "Parsing is intact; direct counting errors and enumeration score-summing "
            "dominate."
        ),
        "Llama3.1-8B": (
            "Mixed failure: direct undercounting plus enumeration format/parse "
            "failures."
        ),
        "Llama3.2-3B": (
            "Both interface and counting fail: enumeration repetition/format "
            "failures coexist with direct over/under-counting."
        ),
    }
    low["diagnosis"] = low["model_label"].map(diagnoses)

    return {
        "failure_budget": failure_budget,
        "model_diagnostics": model_diagnostics,
        "low_diagnostics": low,
        "enum_aggregation": enum_aggregation,
    }


def load_raw_enumeration_evidence(
    archive: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Stream the eight main request files from the preserved export archive.

    The report CSV intentionally omits full outputs and the listed score values.
    This read-only pass recovers only the fields needed to distinguish record
    retrieval from final aggregation and to test the score-summing signature.
    """

    listing = subprocess.run(
        ["tar", "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    members = sorted(
        line.strip()
        for line in listing.stdout.splitlines()
        if line.strip().endswith("/main/requests.jsonl")
        and "/six_models_formal_20260723T194300Z/" in line
    )
    if len(members) != len(MODEL_ORDER):
        raise ValueError(
            f"expected {len(MODEL_ORDER)} main request members in {archive}, "
            f"found {len(members)}"
        )

    rows: list[dict[str, Any]] = []
    for member in members:
        process = subprocess.Popen(
            ["tar", "-xOf", str(archive), member],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        assert process.stdout is not None
        for line in process.stdout:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("prompt_mode") != "enumeration":
                continue
            evaluation = record["evaluation"]
            listed = evaluation.get("listed_records") or []
            score_sum = sum(int(item["score"]) for item in listed)
            predicted = evaluation.get("predicted_count")
            complete_retrieval = (
                len(evaluation.get("missing_pairs") or []) == 0
                and len(evaluation.get("hallucinated_pairs") or []) == 0
                and int(evaluation.get("duplicate_listed_pairs") or 0) == 0
                and len(listed) == int(record["gold_count"])
            )
            wrong_total_after_complete = (
                complete_retrieval and not bool(evaluation["exact_count"])
            )
            predicted_equals_score_sum = (
                predicted is not None and int(predicted) == score_sum
            )
            relative_distance_to_score_sum = math.nan
            if (
                wrong_total_after_complete
                and predicted is not None
                and score_sum > 0
            ):
                relative_distance_to_score_sum = (
                    abs(float(predicted) - score_sum) / score_sum
                )
            output_lines = [
                value.strip()
                for value in str(record.get("raw_output_text") or "").splitlines()
                if value.strip()
            ]
            rows.append(
                {
                    "request_id": record["request_id"],
                    "stimulus_id": record["stimulus_id"],
                    "model_label": record["model_label"],
                    "query_order": record["query_order"],
                    "target_passage_tokens": int(record["target_passage_tokens"]),
                    "num_needles": int(record["num_needles"]),
                    "gold_count": int(record["gold_count"]),
                    "predicted_count": (
                        float(predicted) if predicted is not None else math.nan
                    ),
                    "listed_records_n": len(listed),
                    "listed_score_sum": score_sum,
                    "exact_correct": int(bool(evaluation["exact_count"])),
                    "complete_retrieval": int(complete_retrieval),
                    "wrong_total_after_complete": int(
                        wrong_total_after_complete
                    ),
                    "wrong_total_equals_score_sum": int(
                        wrong_total_after_complete
                        and predicted_equals_score_sum
                    ),
                    "relative_distance_to_score_sum": (
                        relative_distance_to_score_sum
                    ),
                    "output_excerpt": "\n".join(output_lines[-3:]),
                }
            )
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"tar failed for {member} with exit code {return_code}: {stderr}"
            )

    raw = pd.DataFrame(rows)
    expected_rows = len(MODEL_ORDER) * 300
    if len(raw) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} raw enumeration rows, found {len(raw)}"
        )

    summary_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for order in ORDER_ORDER:
            subset = raw[
                raw["model_label"].eq(model) & raw["query_order"].eq(order)
            ]
            wrong_complete = subset[
                subset["wrong_total_after_complete"].eq(1)
                & subset["predicted_count"].notna()
            ]
            correlation = math.nan
            if (
                len(wrong_complete) >= 2
                and wrong_complete["predicted_count"].nunique() > 1
                and wrong_complete["listed_score_sum"].nunique() > 1
            ):
                correlation = float(
                    wrong_complete[
                        ["predicted_count", "listed_score_sum"]
                    ].corr().iloc[0, 1]
                )
            distances = wrong_complete[
                "relative_distance_to_score_sum"
            ].dropna()
            summary_rows.append(
                {
                    "model_label": model,
                    "query_order": order,
                    "enumeration_requests": len(subset),
                    "exact_count_accuracy": subset["exact_correct"].mean(),
                    "complete_retrieval_count": int(
                        subset["complete_retrieval"].sum()
                    ),
                    "complete_retrieval_rate": subset[
                        "complete_retrieval"
                    ].mean(),
                    "complete_retrieval_wrong_total_count": int(
                        subset["wrong_total_after_complete"].sum()
                    ),
                    "complete_retrieval_wrong_total_rate": subset[
                        "wrong_total_after_complete"
                    ].mean(),
                    "wrong_total_equals_score_sum_count": int(
                        subset["wrong_total_equals_score_sum"].sum()
                    ),
                    "predicted_vs_score_sum_correlation": correlation,
                    "median_relative_distance_to_score_sum": (
                        float(distances.median()) if len(distances) else math.nan
                    ),
                    "wrong_totals_within_10pct_of_score_sum_count": int(
                        distances.le(0.10).sum()
                    ),
                }
            )

    examples: dict[str, dict[str, Any]] = {}
    indexed = raw.set_index(["model_label", "stimulus_id", "query_order"])
    for model in ["Qwen3-8B", "Qwen3-32B"]:
        candidates = raw[
            raw["model_label"].eq(model)
            & raw["query_order"].eq("query_first")
            & raw["wrong_total_after_complete"].eq(1)
            & raw["wrong_total_equals_score_sum"].eq(1)
        ]
        if candidates.empty:
            continue
        first = candidates.sort_values(
            ["target_passage_tokens", "num_needles"], ascending=[False, True]
        ).iloc[0]
        last = indexed.loc[
            (model, first["stimulus_id"], "query_last")
        ]
        examples[model] = {
            "stimulus_id": first["stimulus_id"],
            "query_first_excerpt": first["output_excerpt"],
            "query_last_excerpt": last["output_excerpt"],
            "gold_count": int(first["gold_count"]),
            "query_first_predicted": int(first["predicted_count"]),
            "query_last_predicted": (
                int(last["predicted_count"])
                if pd.notna(last["predicted_count"])
                else None
            ),
            "listed_score_sum": int(first["listed_score_sum"]),
        }
    return pd.DataFrame(summary_rows), examples


def exact_mcnemar_p(first_only_correct: int, last_only_correct: int) -> float:
    discordant = int(first_only_correct + last_only_correct)
    if discordant == 0:
        return 1.0
    tail = min(first_only_correct, last_only_correct)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def summarize_anomaly_flags(
    *,
    order_effects: pd.DataFrame,
    mode_effects: pd.DataFrame,
    failure_budget: pd.DataFrame,
    enum_fidelity: pd.DataFrame,
) -> pd.DataFrame:
    """Apply frozen, interpretable rules so anomaly selection is exhaustive."""

    rows: list[dict[str, Any]] = []
    for _, row in order_effects.iterrows():
        effect = float(row["delta_query_last_minus_first"])
        if abs(effect) < QUERY_ANOMALY_THRESHOLD:
            continue
        p_value = exact_mcnemar_p(
            int(row["first_only_correct"]), int(row["last_only_correct"])
        )
        rows.append(
            {
                "flag_type": "query_order_shift",
                "model_label": row["model_label"],
                "prompt_mode": row["prompt_mode"],
                "query_order": "paired",
                "effect_or_rate": effect,
                "absolute_threshold": QUERY_ANOMALY_THRESHOLD,
                "paired_units": int(row["paired_stimuli"]),
                "exact_p_value": p_value,
                "evidence": (
                    f"query_last − query_first = {effect * 100:+.1f} pp; "
                    f"discordant first-only/last-only = "
                    f"{int(row['first_only_correct'])}/"
                    f"{int(row['last_only_correct'])}"
                ),
            }
        )
    for _, row in mode_effects.iterrows():
        effect = float(row["delta_vs_direct"])
        if abs(effect) < MODE_ANOMALY_THRESHOLD:
            continue
        rows.append(
            {
                "flag_type": "mode_shift_vs_direct",
                "model_label": row["model_label"],
                "prompt_mode": row["comparison_mode"],
                "query_order": "paired",
                "effect_or_rate": effect,
                "absolute_threshold": MODE_ANOMALY_THRESHOLD,
                "paired_units": int(row["paired_requests"]),
                "exact_p_value": math.nan,
                "evidence": (
                    f"{row['comparison_mode']} − direct = "
                    f"{effect * 100:+.1f} pp"
                ),
            }
        )
    for _, row in failure_budget.iterrows():
        interface_rate = float(
            row["parse_failure_rate"] + row["truncation_rate"]
        )
        if interface_rate < INTERFACE_FAILURE_THRESHOLD:
            continue
        rows.append(
            {
                "flag_type": "interface_failure",
                "model_label": row["model_label"],
                "prompt_mode": row["prompt_mode"],
                "query_order": row["query_order"],
                "effect_or_rate": interface_rate,
                "absolute_threshold": INTERFACE_FAILURE_THRESHOLD,
                "paired_units": int(row["requests"]),
                "exact_p_value": math.nan,
                "evidence": (
                    f"format/parse + truncation = {interface_rate * 100:.1f}% "
                    f"({int(row['parse_failure_count'])} + "
                    f"{int(row['truncation_count'])} of "
                    f"{int(row['requests'])})"
                ),
            }
        )
    for _, row in enum_fidelity.iterrows():
        rate = float(row["complete_retrieval_wrong_total_rate"])
        if rate < AGGREGATION_FAILURE_THRESHOLD:
            continue
        rows.append(
            {
                "flag_type": "aggregation_after_complete_retrieval",
                "model_label": row["model_label"],
                "prompt_mode": "enumeration",
                "query_order": row["query_order"],
                "effect_or_rate": rate,
                "absolute_threshold": AGGREGATION_FAILURE_THRESHOLD,
                "paired_units": int(row["enumeration_requests"]),
                "exact_p_value": math.nan,
                "evidence": (
                    f"complete record set but wrong Total = "
                    f"{int(row['complete_retrieval_wrong_total_count'])}/"
                    f"{int(row['enumeration_requests'])} "
                    f"({rate * 100:.1f}%)"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["flag_type", "model_label", "prompt_mode", "query_order"]
    )


def summarize_anomaly_cases(
    *,
    mode_query: pd.DataFrame,
    failure_budget: pd.DataFrame,
    enum_fidelity: pd.DataFrame,
) -> pd.DataFrame:
    """Create the compact interpretation layer shown in the Results section."""

    mode = mode_query.set_index(["model_label", "prompt_mode"])
    failures = failure_budget.set_index(
        ["model_label", "prompt_mode", "query_order"]
    )
    fidelity = enum_fidelity.set_index(["model_label", "query_order"])

    def mq(model: str, prompt_mode: str) -> pd.Series:
        return mode.loc[(model, prompt_mode)]

    def fb(model: str, prompt_mode: str, order: str) -> pd.Series:
        return failures.loc[(model, prompt_mode, order)]

    def ef(model: str, order: str) -> pd.Series:
        return fidelity.loc[(model, order)]

    q8_enum = mq("Qwen3-8B", "enumeration")
    q8_first = ef("Qwen3-8B", "query_first")
    q32_enum = mq("Qwen3-32B", "enumeration")
    q32_first = ef("Qwen3-32B", "query_first")
    q17_direct = mq("Qwen3-1.7B", "direct")
    q17_enum = mq("Qwen3-1.7B", "enumeration")
    q17_native = mq("Qwen3-1.7B", "native_thinking")
    q32_direct = mq("Qwen3-32B", "direct")
    q32_native = mq("Qwen3-32B", "native_thinking")
    ge_direct = mq("Gemma4-E4B", "direct")
    ge_enum = mq("Gemma4-E4B", "enumeration")
    ge_native = mq("Gemma4-E4B", "native_thinking")
    g12_direct = mq("Gemma4-12B", "direct")
    g12_enum = mq("Gemma4-12B", "enumeration")
    g12_native = mq("Gemma4-12B", "native_thinking")
    olmo_direct = mq("OLMo-Hybrid-7B", "direct")
    olmo_enum = mq("OLMo-Hybrid-7B", "enumeration")
    llama31_direct = mq("Llama3.1-8B", "direct")
    llama31_enum = mq("Llama3.1-8B", "enumeration")
    llama32_direct = mq("Llama3.2-3B", "direct")
    llama32_enum = mq("Llama3.2-3B", "enumeration")

    rows = [
        {
            "anomaly_id": "A1",
            "models": "Qwen3-8B",
            "scope": "enumeration × query order",
            "observation": (
                f"Exact count jumps from {pct(q8_enum['query_first_accuracy'])} "
                f"to {pct(q8_enum['query_last_accuracy'])}."
            ),
            "evidence": (
                f"With query first, {int(q8_first['complete_retrieval_count'])}/150 "
                f"lists are complete, but "
                f"{int(q8_first['complete_retrieval_wrong_total_count'])} of them "
                f"still have the wrong Total. Predicted Total versus score sum has "
                f"r={q8_first['predicted_vs_score_sum_correlation']:.3f}; median "
                f"relative distance is "
                f"{q8_first['median_relative_distance_to_score_sum'] * 100:.1f}%, "
                f"and {int(q8_first['wrong_totals_within_10pct_of_score_sum_count'])}/"
                f"{int(q8_first['complete_retrieval_wrong_total_count'])} are "
                "within 10%."
            ),
            "interpretation": (
                "The main failure is operator selection at aggregation: Total is "
                "read as a sum of numeric scores rather than the number of records. "
                "Moving the instruction next to generation restores the count rule."
            ),
            "caveat": (
                "The 95.3% headline scores only the final count. Full-list fidelity "
                f"is {pct(ef('Qwen3-8B', 'query_last')['complete_retrieval_rate'])} "
                "with query last, so it is not a 95.3% fully-correct enumeration rate."
            ),
        },
        {
            "anomaly_id": "A2",
            "models": "Qwen3-32B",
            "scope": "enumeration × query order",
            "observation": (
                f"Exact count rises from {pct(q32_enum['query_first_accuracy'])} "
                f"to {pct(q32_enum['query_last_accuracy'])}."
            ),
            "evidence": (
                f"All {int(q32_first['complete_retrieval_wrong_total_count'])} "
                "query-first complete-list errors occur after retrieval; "
                f"{int(q32_first['wrong_total_equals_score_sum_count'])} equal the "
                "listed score sum exactly, "
                f"{int(q32_first['wrong_totals_within_10pct_of_score_sum_count'])} "
                f"are within 10%, and corr(Total, score sum)="
                f"{q32_first['predicted_vs_score_sum_correlation']:.3f}."
            ),
            "interpretation": (
                "The same aggregation ambiguity appears at 32B, so it is not unique "
                "to the 8B checkpoint. Native thinking stays near ceiling, showing "
                "that the failure is execution-path specific."
            ),
            "caveat": (
                "Shared behavior across two Qwen sizes supports a family-level "
                "prompt interaction, not a universal architectural law."
            ),
        },
        {
            "anomaly_id": "A3",
            "models": "Qwen3-8B",
            "scope": "direct signed-error reversal",
            "observation": (
                "Direct accuracy is 42.7% for both query orders, although the error "
                "direction nearly reverses."
            ),
            "evidence": (
                f"Query first has {int(fb('Qwen3-8B','direct','query_first')['undercount_count'])} "
                "undercounts and "
                f"{int(fb('Qwen3-8B','direct','query_first')['overcount_count'])} "
                "overcount; query last has "
                f"{int(fb('Qwen3-8B','direct','query_last')['undercount_count'])} "
                "undercounts and "
                f"{int(fb('Qwen3-8B','direct','query_last')['overcount_count'])} "
                "overcounts."
            ),
            "interpretation": (
                "Query placement changes the sign of numerical bias even when exact "
                "accuracy is unchanged. Accuracy alone therefore hides a major "
                "mechanism change."
            ),
            "caveat": (
                "This identifies an output-bias shift; it does not isolate the "
                "internal attention or memory process causing it."
            ),
        },
        {
            "anomaly_id": "A4",
            "models": "Qwen3-1.7B",
            "scope": "mode × query-order sign reversal",
            "observation": (
                f"Query last helps direct ({pct(q17_direct['query_first_accuracy'])} → "
                f"{pct(q17_direct['query_last_accuracy'])}) and enumeration "
                f"({pct(q17_enum['query_first_accuracy'])} → "
                f"{pct(q17_enum['query_last_accuracy'])}), but hurts native thinking "
                f"({pct(q17_native['query_first_accuracy'])} → "
                f"{pct(q17_native['query_last_accuracy'])})."
            ),
            "evidence": (
                "Direct query first produces 126 undercounts; native thinking query "
                "first instead reaches 102/150 exact."
            ),
            "interpretation": (
                "Native thinking is not a uniform accuracy offset: it changes how "
                "the model uses query position and reduces the strong query-first "
                "undercount mechanism."
            ),
            "caveat": (
                "The comparison is within one model and frozen decoding; it should "
                "not be generalized to untested Qwen templates."
            ),
        },
        {
            "anomaly_id": "A5",
            "models": "Qwen3-32B",
            "scope": "direct/enumeration order reversal",
            "observation": (
                f"Direct prefers query first ({pct(q32_direct['query_first_accuracy'])} "
                f"vs {pct(q32_direct['query_last_accuracy'])}), enumeration strongly "
                f"prefers query last, and native thinking is stable "
                f"({pct(q32_native['query_first_accuracy'])} vs "
                f"{pct(q32_native['query_last_accuracy'])})."
            ),
            "evidence": (
                "The order effect changes from −16.0 pp in direct to +48.0 pp in "
                "enumeration while native thinking differs by only −1.3 pp."
            ),
            "interpretation": (
                "There is no model-wide best query order. The output protocol selects "
                "different counting/aggregation paths with different position "
                "sensitivity."
            ),
            "caveat": (
                "Order and instruction-to-generation distance change together in "
                "this design, so a rule-repetition ablation is needed for causality."
            ),
        },
        {
            "anomaly_id": "A6",
            "models": "Gemma4-E4B",
            "scope": "query-first advantage only for explicit reasoning",
            "observation": (
                f"Enumeration/native thinking fall from "
                f"{pct(ge_enum['query_first_accuracy'])}/"
                f"{pct(ge_native['query_first_accuracy'])} to "
                f"{pct(ge_enum['query_last_accuracy'])}/"
                f"{pct(ge_native['query_last_accuracy'])}, while direct improves "
                f"from {pct(ge_direct['query_first_accuracy'])} to "
                f"{pct(ge_direct['query_last_accuracy'])}."
            ),
            "evidence": (
                "Enumeration undercounts rise from 1 to 34 and native-thinking "
                "undercounts from 4 to 30 when the query moves last."
            ),
            "interpretation": (
                "For explicit extraction/reasoning, knowing the task before reading "
                "the passage appears to help Gemma retain the full candidate set; "
                "the terse direct path has a different bottleneck."
            ),
            "caveat": (
                "This is the best-supported behavioral interpretation, not direct "
                "evidence about Gemma's internal memory state."
            ),
        },
        {
            "anomaly_id": "A7",
            "models": "Gemma4-12B",
            "scope": "mode-specific generation collapse",
            "observation": (
                f"Direct accuracy is {pct(g12_direct['accuracy'])}, whereas "
                f"enumeration reaches {pct(g12_enum['accuracy'])}; native thinking "
                f"is intermediate at {pct(g12_native['accuracy'])}."
            ),
            "evidence": (
                "Direct truncates on 293/300 requests, including 150/150 query-last "
                "requests. Native-thinking truncation rises from 38/150 query first "
                "to 78/150 query last; enumeration query first is 150/150 exact."
            ),
            "interpretation": (
                "This is a generation-budget/channel-path failure rather than a "
                "general inability to count. The same model succeeds when routed "
                "through the enumeration output path."
            ),
            "caveat": (
                "Its direct/native scores are conditional on the frozen generation "
                "budget and should not be treated as architecture-only capability."
            ),
        },
        {
            "anomaly_id": "A8",
            "models": "OLMo-Hybrid-7B",
            "scope": "enumeration underperforms direct",
            "observation": (
                f"Explicit enumeration lowers accuracy from "
                f"{pct(olmo_direct['accuracy'])} to {pct(olmo_enum['accuracy'])}."
            ),
            "evidence": (
                "Parsing remains high, but enumeration produces 101 overcounts; "
                f"{int(ef('OLMo-Hybrid-7B','query_first')['complete_retrieval_wrong_total_count'] + ef('OLMo-Hybrid-7B','query_last')['complete_retrieval_wrong_total_count'])} "
                "requests retrieve the complete record set and still give a wrong "
                "Total."
            ),
            "interpretation": (
                "Listing records adds an aggregation stage that can fail; explicit "
                "intermediate output is not automatically beneficial."
            ),
            "caveat": (
                "OLMo native thinking was not registered, so this does not compare "
                "all possible reasoning interfaces."
            ),
        },
        {
            "anomaly_id": "A9",
            "models": "Llama3.1-8B",
            "scope": "mixed counting and interface failure",
            "observation": (
                f"Query last improves direct from "
                f"{pct(llama31_direct['query_first_accuracy'])} to "
                f"{pct(llama31_direct['query_last_accuracy'])}, but enumeration "
                f"remains only {pct(llama31_enum['accuracy'])} overall."
            ),
            "evidence": (
                "Direct query first has 121 undercounts. Enumeration has 159/300 "
                "non-truncation format/parse failures (92 query first, 67 query last)."
            ),
            "interpretation": (
                "Two mechanisms coexist: position-sensitive undercounting in direct "
                "answers and failure to obey the multi-line enumeration contract."
            ),
            "caveat": (
                "Combining these failures into one 'counting error' would overstate "
                "the numerical component."
            ),
        },
        {
            "anomaly_id": "A10",
            "models": "Llama3.2-3B",
            "scope": "mode-specific query-last collapse",
            "observation": (
                f"Query last raises direct accuracy from "
                f"{pct(llama32_direct['query_first_accuracy'])} to "
                f"{pct(llama32_direct['query_last_accuracy'])}, yet enumeration "
                f"falls from {pct(llama32_enum['query_first_accuracy'])} to "
                f"{pct(llama32_enum['query_last_accuracy'])}."
            ),
            "evidence": (
                "Enumeration query last has 123 format/parse failures plus 5 "
                "truncations out of 150; query first also mixes 67 parse failures, "
                "58 overcounts, and 7 truncations."
            ),
            "interpretation": (
                "The short direct contract benefits from a recent query, while the "
                "long enumeration contract collapses at the interface/format stage. "
                "This is not a simple query-position effect."
            ),
            "caveat": (
                "The 2.0% exact score should be read with the 85.3% format-failure "
                "rate, not as pure retrieval inability."
            ),
        },
    ]
    return pd.DataFrame(rows)


def plot_query_order(model_query: pd.DataFrame, output: Path) -> None:
    ordered = model_query.sort_values("overall_accuracy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(ordered))
    for index, row in ordered.iterrows():
        ax.plot(
            [row["query_first_accuracy"], row["query_last_accuracy"]],
            [index, index],
            color="#c8cdd4",
            linewidth=2.2,
            zorder=1,
        )
    ax.scatter(
        ordered["query_first_accuracy"],
        y,
        s=65,
        color="#315f8c",
        label="Query first",
        zorder=3,
    )
    ax.scatter(
        ordered["query_last_accuracy"],
        y,
        s=70,
        marker="D",
        color="#c56635",
        label="Query last",
        zorder=3,
    )
    for index, row in ordered.iterrows():
        ax.text(
            row["query_first_accuracy"] - 0.012,
            index + 0.15,
            f"{row['query_first_accuracy']:.1%}",
            ha="right",
            va="bottom",
            color="#315f8c",
            fontsize=8.5,
        )
        ax.text(
            row["query_last_accuracy"] + 0.012,
            index - 0.15,
            f"{row['query_last_accuracy']:.1%}",
            ha="left",
            va="top",
            color="#a74e25",
            fontsize=8.5,
        )
    ax.set_yticks(y, ordered["model_label"])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Exact accuracy (all registered prompt modes pooled)")
    ax.set_title("Exact accuracy changes with query placement")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color="#e5e8ec", linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure_atomic(fig, output)


def plot_mode_query_heatmap(mode_query: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.4, 13.2), constrained_layout=True)
    axes_flat = axes.ravel()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "accuracy", ["#f4ece4", "#f0c796", "#8fb8ad", "#276b5d"]
    )
    image = None
    for ax, model in zip(axes_flat, MODEL_ORDER):
        matrix = np.full((3, 2), np.nan)
        subset = mode_query[mode_query["model_label"].eq(model)].set_index("prompt_mode")
        for row_index, mode in enumerate(MODE_ORDER):
            if mode in subset.index:
                matrix[row_index, 0] = subset.loc[mode, "query_first_accuracy"]
                matrix[row_index, 1] = subset.loc[mode, "query_last_accuracy"]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1], ["Query first", "Query last"])
        ax.set_yticks([0, 1, 2], ["Direct", "Enumeration", "Native thinking"])
        ax.set_title(model, loc="left", fontweight="bold")
        for row_index in range(3):
            for column_index in range(2):
                value = matrix[row_index, column_index]
                text = "not run" if np.isnan(value) else f"{value:.1%}"
                color = "#7a7f87" if np.isnan(value) else ("white" if value >= 0.68 else "#1e262b")
                ax.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold" if not np.isnan(value) else "normal",
                    color=color,
                )
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes_flat.tolist(), shrink=0.55, pad=0.02)
        colorbar.set_label("Exact accuracy")
        colorbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Mode × query-placement accuracy: the interaction is model-specific", y=1.02)
    save_figure_atomic(fig, output)


def plot_low_failure_budget(failure_budget: pd.DataFrame, output: Path) -> None:
    registered_rows: list[tuple[str, str]] = []
    for model in DIAGNOSTIC_MODELS:
        for mode in MODE_ORDER:
            if (
                (failure_budget["model_label"].eq(model))
                & (failure_budget["prompt_mode"].eq(mode))
            ).any():
                registered_rows.append((model, mode))
    labels = [f"{model} · {MODE_LABEL[mode]}" for model, mode in registered_rows]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 9.2), sharey=True, constrained_layout=True)
    for ax, order in zip(axes, ORDER_ORDER):
        left = np.zeros(len(registered_rows))
        for category in ERROR_ORDER:
            values: list[float] = []
            for model, mode in registered_rows:
                match = failure_budget[
                    failure_budget["model_label"].eq(model)
                    & failure_budget["prompt_mode"].eq(mode)
                    & failure_budget["query_order"].eq(order)
                ]
                values.append(float(match.iloc[0][f"{category}_rate"]))
            ax.barh(
                np.arange(len(registered_rows)),
                values,
                left=left,
                color=ERROR_COLORS[category],
                label=ERROR_LABEL[category],
                height=0.72,
            )
            left += np.asarray(values)
        ax.set_xlim(0, 1)
        ax.set_title(ORDER_LABEL[order], fontweight="bold")
        ax.set_xlabel("Share of all registered requests")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", color="#e6e8eb", linewidth=0.7)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("Why the low-accuracy and mode-collapse cases fail", y=1.02)
    save_figure_atomic(fig, output)


def plot_enumeration_aggregation(enum_aggregation: pd.DataFrame, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = enum_aggregation[enum_aggregation["model_label"].eq(model)]
        total = subset["enumeration_requests"].sum()
        rows.append(
            {
                "model_label": model,
                "overcount_rate": subset["overcount_count"].sum() / total,
                "complete_rate": subset["complete_retrieval_wrong_total_count"].sum() / total,
            }
        )
    frame = pd.DataFrame(rows).sort_values("overcount_rate", ascending=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10.4, 6.3))
    ax.barh(
        y,
        frame["overcount_rate"],
        height=0.58,
        color="#d99467",
        label="All enumeration overcounts",
    )
    ax.barh(
        y,
        frame["complete_rate"],
        height=0.30,
        color="#8a3e2c",
        label="Complete retrieval, wrong final total",
    )
    for index, row in frame.reset_index(drop=True).iterrows():
        if row["complete_rate"] > 0:
            ax.text(
                row["complete_rate"] + 0.008,
                index,
                f"{row['complete_rate']:.1%}",
                va="center",
                fontsize=8.5,
                color="#6e2b20",
            )
    ax.set_yticks(y, frame["model_label"])
    ax.set_xlim(0, max(0.56, frame["overcount_rate"].max() + 0.06))
    ax.set_xlabel("Share of enumeration requests")
    ax.set_title("A major enumeration failure is aggregation, not retrieval")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color="#e6e8eb", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure_atomic(fig, output)


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def pp(value: Any, digits: int = 1, signed: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    number = float(value) * 100
    return f"{number:+.{digits}f} pp" if signed else f"{number:.{digits}f} pp"


def decimal(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def table_html(headers: list[str], rows: Iterable[list[Any]], *, classes: str = "") -> str:
    head = "".join(f"<th>{esc(label)}</th>" for label in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{value}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table class="{esc(classes)}">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def anomaly_flag_summary(flags: pd.DataFrame) -> str:
    definitions = [
        (
            "query_order_shift",
            f"|query-last − query-first| ≥ {QUERY_ANOMALY_THRESHOLD * 100:.0f} pp",
        ),
        (
            "mode_shift_vs_direct",
            f"|mode − direct| ≥ {MODE_ANOMALY_THRESHOLD * 100:.0f} pp",
        ),
        (
            "interface_failure",
            f"format/parse + truncation ≥ {INTERFACE_FAILURE_THRESHOLD * 100:.0f}%",
        ),
        (
            "aggregation_after_complete_retrieval",
            "complete record set but wrong Total ≥ "
            f"{AGGREGATION_FAILURE_THRESHOLD * 100:.0f}%",
        ),
    ]
    rows = []
    for flag_type, rule in definitions:
        subset = flags[flags["flag_type"].eq(flag_type)]
        examples = "; ".join(
            f"{row['model_label']} / {MODE_LABEL.get(row['prompt_mode'], row['prompt_mode'])}"
            + (
                f" / {ORDER_LABEL.get(row['query_order'], row['query_order'])}"
                if row["query_order"] != "paired"
                else ""
            )
            for _, row in subset.head(4).iterrows()
        )
        if len(subset) > 4:
            examples += f"; +{len(subset) - 4} more"
        rows.append(
            [
                esc(flag_type.replace("_", " ")),
                esc(rule),
                str(len(subset)),
                esc(examples or "None"),
            ]
        )
    return table_html(
        ["Flag family", "Frozen rule", "Flagged conditions", "Examples"],
        rows,
        classes="compact",
    )


def qwen_enumeration_fidelity_table(enum_fidelity: pd.DataFrame) -> str:
    rows = []
    subset = enum_fidelity[
        enum_fidelity["model_label"].isin(["Qwen3-8B", "Qwen3-32B"])
    ]
    for model in ["Qwen3-8B", "Qwen3-32B"]:
        for order in ORDER_ORDER:
            row = subset[
                subset["model_label"].eq(model)
                & subset["query_order"].eq(order)
            ].iloc[0]
            rows.append(
                [
                    f"<strong>{esc(model)}</strong>",
                    esc(ORDER_LABEL[order]),
                    pct(row["exact_count_accuracy"]),
                    (
                        f"{int(row['complete_retrieval_count'])}/"
                        f"{int(row['enumeration_requests'])} "
                        f"({pct(row['complete_retrieval_rate'])})"
                    ),
                    str(int(row["complete_retrieval_wrong_total_count"])),
                    str(int(row["wrong_total_equals_score_sum_count"])),
                    decimal(row["predicted_vs_score_sum_correlation"]),
                    pct(row["median_relative_distance_to_score_sum"]),
                ]
            )
    return table_html(
        [
            "Model",
            "Query order",
            "Final-count accuracy",
            "Complete record set",
            "Complete set, wrong Total",
            "Wrong Total = score sum",
            "corr(Total, score sum)",
            "Median distance to score sum",
        ],
        rows,
        classes="compact numeric",
    )


def anomaly_case_cards(
    cases: pd.DataFrame,
    examples: dict[str, dict[str, Any]],
) -> str:
    cards: list[str] = []
    for _, row in cases.iterrows():
        example_html = ""
        if row["anomaly_id"] == "A1" and "Qwen3-8B" in examples:
            example = examples["Qwen3-8B"]
            example_html = f"""
            <div class="paired-example">
              <div>
                <strong>Same stimulus · query first</strong>
                <pre>{esc(example['query_first_excerpt'])}</pre>
              </div>
              <div>
                <strong>Same stimulus · query last</strong>
                <pre>{esc(example['query_last_excerpt'])}</pre>
              </div>
            </div>
            """
        cards.append(
            f"""
            <details class="anomaly-case">
              <summary>
                <span class="anomaly-id">{esc(row['anomaly_id'])}</span>
                <span>{esc(row['models'])} · {esc(row['observation'])}</span>
              </summary>
              <div>
                <p><strong>Scope.</strong> {esc(row['scope'])}</p>
                <p><strong>Measured evidence.</strong> {esc(row['evidence'])}</p>
                <p><strong>Best-supported explanation.</strong> {esc(row['interpretation'])}</p>
                <p class="caveat"><strong>Limit.</strong> {esc(row['caveat'])}</p>
                {example_html}
              </div>
            </details>
            """
        )
    return '<div class="anomaly-stack">' + "".join(cards) + "</div>"


def model_accuracy_table(model_query: pd.DataFrame) -> str:
    frame = model_query.set_index("model_label")
    rows = []
    for model in sorted(MODEL_ORDER, key=lambda item: frame.loc[item, "overall_accuracy"], reverse=True):
        row = frame.loc[model]
        ci = f"[{pp(row['delta_ci95_low'])}, {pp(row['delta_ci95_high'])}]"
        rows.append(
            [
                f"<strong>{esc(model)}</strong>",
                f"{int(row['exact_correct'])}/{int(row['requests'])}",
                pct(row["overall_accuracy"]),
                pct(row["query_first_accuracy"]),
                pct(row["query_last_accuracy"]),
                pp(row["delta_query_last_minus_first"]),
                esc(ci),
            ]
        )
    return table_html(
        [
            "Model",
            "Exact / n",
            "Overall",
            "Query first",
            "Query last",
            "Δ last − first",
            "Paired cluster-bootstrap 95% CI",
        ],
        rows,
        classes="numeric",
    )


def mode_effect_lookup(mode_effects: pd.DataFrame, model: str, mode: str) -> pd.Series | None:
    match = mode_effects[
        mode_effects["model_label"].eq(model)
        & mode_effects["comparison_mode"].eq(mode)
    ]
    return None if match.empty else match.iloc[0]


def model_note(model: str, mode_query: pd.DataFrame, mode_effects: pd.DataFrame) -> str:
    rows = mode_query[mode_query["model_label"].eq(model)].set_index("prompt_mode")
    direct = rows.loc["direct"]
    enumeration = rows.loc["enumeration"]
    native = rows.loc["native_thinking"] if "native_thinking" in rows.index else None
    if model == "Qwen3-8B":
        return (
            f"Native thinking is both strongest and stable ({pct(native['accuracy'])}). "
            f"Enumeration is highly order-sensitive: {pct(enumeration['query_first_accuracy'])} "
            f"with query first versus {pct(enumeration['query_last_accuracy'])} with query last. "
            "The low query-first enumeration score is dominated by summing the record scores "
            "instead of counting records."
        )
    if model == "Qwen3-1.7B":
        return (
            f"Native thinking raises accuracy from {pct(direct['accuracy'])} to "
            f"{pct(native['accuracy'])}. Query last helps direct and enumeration, but native "
            "thinking is better with query first. The remaining error is mainly numeric "
            "under/over-counting rather than parsing."
        )
    if model == "Qwen3-32B":
        return (
            f"Native thinking is near ceiling ({pct(native['accuracy'])}) for both orders. "
            f"Enumeration again depends on placement ({pct(enumeration['query_first_accuracy'])} "
            f"vs {pct(enumeration['query_last_accuracy'])}); all 72 query-first enumeration "
            "overcounts retained the complete gold record list and then produced the wrong total."
        )
    if model == "Gemma4-E4B":
        return (
            f"Enumeration ({pct(enumeration['accuracy'])}) and native thinking "
            f"({pct(native['accuracy'])}) are effectively tied and far above direct "
            f"({pct(direct['accuracy'])}). Unlike Qwen enumeration, query first is better; "
            "query-last errors are mainly undercounts."
        )
    if model == "Gemma4-12B":
        return (
            f"Enumeration is the reliable mode ({pct(enumeration['accuracy'])}; "
            f"{pct(enumeration['query_first_accuracy'])} with query first). Direct collapses "
            f"to {pct(direct['accuracy'])} because {pct(direct['truncation_rate'])} of requests "
            "hit the generation limit. This is a mode/template-output failure, not evidence "
            "that parsed numeric counts are generally poor."
        )
    if model == "OLMo-Hybrid-7B":
        return (
            f"Direct ({pct(direct['accuracy'])}) is better than enumeration "
            f"({pct(enumeration['accuracy'])}); query first is modestly better in both modes. "
            "Native thinking was not registered. Parsing is almost complete, so the bottleneck "
            "is numeric counting and enumeration aggregation."
        )
    if model == "Llama3.1-8B":
        return (
            f"Both modes are weak (direct {pct(direct['accuracy'])}, enumeration "
            f"{pct(enumeration['accuracy'])}), and query last helps both. Direct query-first "
            "failures are mostly undercounts; enumeration additionally has a high format/parse "
            "failure rate. Native thinking was not registered."
        )
    if model == "Llama3.2-3B":
        return (
            f"Direct query last reaches {pct(direct['query_last_accuracy'])}, but query first is "
            f"only {pct(direct['query_first_accuracy'])}. Enumeration is unreliable "
            f"({pct(enumeration['accuracy'])}), especially query last, where formatting fails "
            f"on {pct(enumeration['query_last_format_failure_rate'])}. Repetition, hallucinated "
            "records, and score-summing all occur. Native thinking was not registered."
        )
    raise KeyError(model)


def model_cards(mode_query: pd.DataFrame, mode_effects: pd.DataFrame) -> str:
    output: list[str] = []
    for model in MODEL_ORDER:
        subset = mode_query[mode_query["model_label"].eq(model)].set_index("prompt_mode")
        rows: list[list[Any]] = []
        for mode in MODE_ORDER:
            if mode not in subset.index:
                rows.append(
                    [
                        f"<strong>{esc(MODE_LABEL[mode])}</strong>",
                        "—",
                        "—",
                        "—",
                        "—",
                        "Not registered",
                    ]
                )
                continue
            row = subset.loc[mode]
            effect = mode_effect_lookup(mode_effects, model, mode)
            effect_text = "reference" if mode == "direct" else pp(effect["delta_vs_direct"])
            rows.append(
                [
                    f"<strong>{esc(MODE_LABEL[mode])}</strong>",
                    pct(row["query_first_accuracy"]),
                    pct(row["query_last_accuracy"]),
                    pct(row["accuracy"]),
                    pp(row["query_last_accuracy"] - row["query_first_accuracy"]),
                    esc(effect_text),
                ]
            )
        output.append(
            f"""
            <article class="model-detail" id="model-{esc(model.lower().replace('.', '-'))}">
              <h3>{esc(model)}</h3>
              {table_html(
                  ["Mode", "Query first", "Query last", "Overall", "Δ last − first", "Δ vs direct"],
                  rows,
                  classes="compact numeric",
              )}
              <p>{esc(model_note(model, mode_query, mode_effects))}</p>
            </article>
            """
        )
    return "\n".join(output)


def low_diagnostic_table(low: pd.DataFrame) -> str:
    frame = low.set_index("model_label")
    rows = []
    for model in DIAGNOSTIC_MODELS:
        row = frame.loc[model]
        rows.append(
            [
                f"<strong>{esc(model)}</strong>",
                pct(row["overall_accuracy"]),
                pct(row["parse_success_rate"]),
                pct(row["exact_given_parsed"]),
                pct(row["undercount_rate"]),
                pct(row["overcount_rate"]),
                pct(row["parse_failure_rate"]),
                pct(row["truncation_rate"]),
            ]
        )
    return table_html(
        [
            "Model",
            "Exact accuracy",
            "Parse success",
            "Exact | parsed",
            "Undercount",
            "Overcount",
            "Format / parse",
            "Truncation",
        ],
        rows,
        classes="numeric",
    )


def enumeration_mechanism_table(enum: pd.DataFrame) -> str:
    rows: list[list[Any]] = []
    for model in MODEL_ORDER:
        subset = enum[enum["model_label"].eq(model)]
        first = subset[subset["query_order"].eq("query_first")].iloc[0]
        last = subset[subset["query_order"].eq("query_last")].iloc[0]
        total_requests = int(subset["enumeration_requests"].sum())
        overcounts = int(subset["overcount_count"].sum())
        complete = int(subset["complete_retrieval_wrong_total_count"].sum())
        rows.append(
            [
                f"<strong>{esc(model)}</strong>",
                pct(first["enumeration_accuracy"]),
                pct(last["enumeration_accuracy"]),
                f"{overcounts}/{total_requests} ({pct(overcounts / total_requests)})",
                f"{complete}/{total_requests} ({pct(complete / total_requests)})",
                pct(complete / overcounts) if overcounts else "—",
            ]
        )
    return table_html(
        [
            "Model",
            "Enum. acc. first",
            "Enum. acc. last",
            "All overcounts",
            "Complete retrieval, wrong total",
            "Share of overcounts",
        ],
        rows,
        classes="numeric",
    )


def accuracy_fit_table(fits: pd.DataFrame) -> str:
    labels = {
        "model_specific_length_needles_slopes": "Model-specific length / needle slopes",
        "log_length_needles_interaction_model_fe": "Shared log L/N + interaction; model intercepts",
        "log_length_needles_model_fe": "Shared log L/N; model intercepts",
        "density_model_fe": "Density-only; model intercepts",
        "controls_only": "Mode/order controls only",
    }
    rows = []
    for _, row in fits.sort_values("local_cv_log_loss_mean").iterrows():
        rows.append(
            [
                esc(labels.get(row["candidate"], row["candidate"])),
                str(int(row["n_parameters"])),
                decimal(row["local_cv_log_loss_mean"]),
                decimal(row["local_cv_brier"]),
                pct(row["local_cv_ece"]),
            ]
        )
    return table_html(
        ["Accuracy candidate", "Parameters", "Grouped-CV log loss", "Brier", "ECE"],
        rows,
        classes="numeric",
    )


def bias_fit_table(bias: pd.DataFrame) -> str:
    frame = bias.set_index("model_label")
    rows = []
    for model in MODEL_ORDER:
        row = frame.loc[model]
        rows.append(
            [
                f"<strong>{esc(model)}</strong>",
                esc(row["selected_candidate_cn"]),
                pct(row["four_scheme_gain_pct"] / 100.0),
                esc(row["evidence_cn"]),
            ]
        )
    return table_html(
        ["Model", "Selected signed-bias law", "CV gain vs condition-only", "Evidence"],
        rows,
        classes="numeric",
    )


def prompt_protocol_table(prompt_summary: pd.DataFrame) -> str:
    frame = prompt_summary.set_index("model_label")
    rows = []
    for model in MODEL_ORDER:
        row = frame.loc[model]
        wrapper, behavior = WRAPPER_DESCRIPTIONS[model]
        rows.append(
            [
                f"<strong>{esc(model)}</strong>",
                esc(row["registered_prompt_modes"]),
                "Yes" if bool(row["native_thinking_supported"]) else "No",
                esc(wrapper),
                esc(behavior),
            ]
        )
    return table_html(
        [
            "Model",
            "Registered modes",
            "Native thinking",
            "Tokenizer-rendered wrapper",
            "Injected/template behavior",
        ],
        rows,
        classes="compact",
    )


def prompt_and_decoding_protocol() -> str:
    return f"""
    <h3>Query placement：唯一改变的是 task block 的位置</h3>
    <div class="prompt-grid">
      <div>
        <h4>Query-first 外层结构</h4>
        <pre>&lt;TASK BLOCK&gt;

&lt;passage&gt;
{{context}}
&lt;/passage&gt;</pre>
      </div>
      <div>
        <h4>Query-last 外层结构</h4>
        <pre>&lt;passage&gt;
{{context}}
&lt;/passage&gt;

&lt;TASK BLOCK&gt;</pre>
      </div>
    </div>
    <p>System/template 文本、passage 内容、分隔符和输出约束保持不变；query last
    的 passage 前不出现寻找 city-score records 的 task cue。</p>

    <h3>Direct 与 native thinking 使用的完整 task block</h3>
    <pre>{esc(DIRECT_TASK_BLOCK)}</pre>
    <p>两者 messages 完全相同。Direct 关闭模型官方 thinking 开关；
    native thinking 只开启该开关，并将 raw reasoning 与 final answer 分开保存。</p>
    <p>对 Qwen3，direct 与 enumeration 使用
    <code>enable_thinking=false</code>，tokenizer-rendered assistant prefix 中预闭合一个
    空的 <code>&lt;think&gt;…&lt;/think&gt;</code> 区块；native thinking 使用
    <code>enable_thinking=true</code>，不预闭合该区块。其他 checkpoint 的精确 wrapper
    与注入行为列在下表，并由逐条件 prompt SHA256 快照审计。</p>

    <h3>Enumeration 的完整 task block</h3>
    <pre>{esc(ENUMERATION_TASK_BLOCK)}</pre>
    <p>Enumeration 关闭 native thinking；评估器同时解析编号清单与最后的
    <code>Total</code>。Primary exact accuracy 只检查 final count，完整清单 fidelity
    在异常诊断中另算。</p>

    <h3>冻结解码设置</h3>
    {table_html(
        [
            "Mode / family",
            "Sampling",
            "Thinking",
            "max new tokens",
            "temperature",
            "top-p",
            "top-k",
            "min-p",
        ],
        [
            [
                "Direct / all models",
                "Greedy",
                "Off where supported",
                "64",
                "0",
                "1.0",
                "−1",
                "0",
            ],
            [
                "Enumeration / all models",
                "Greedy",
                "Off where supported",
                "1,536",
                "0",
                "1.0",
                "−1",
                "0",
            ],
            [
                "Native thinking / Qwen3",
                "Sampled",
                "On",
                "4,096",
                "0.6",
                "0.95",
                "20",
                "0",
            ],
            [
                "Native thinking / Gemma4",
                "Sampled",
                "On",
                "4,096",
                "1.0",
                "0.95",
                "64",
                "template default",
            ],
        ],
        classes="compact numeric",
    )}
    <p>Native-thinking generation seed 由 stimulus seed 确定并保存。所有 first-pass
    outputs 均保留；<code>finish_reason=length</code> 记为 truncation，不重写 prompt、
    不放宽 parser、不选择性重跑。</p>
    """


def method_conclusion(method: str, conclusion: str) -> str:
    return f"""
    <div class="method-grid">
      <div class="method-box">
        <span>计算方法</span>
        <p>{method}</p>
      </div>
      <div class="conclusion-box">
        <span>目前结论</span>
        <p>{conclusion}</p>
      </div>
    </div>
    """


def math_accuracy_definition() -> str:
    return """
    <div class="math-equation">
      <div class="equation-title">Primary outcome: exact correctness over every registered request</div>
      <div class="math-scroll">
        <math display="block" aria-label="Exact accuracy definition">
          <mrow>
            <msub><mi>Y</mi><mi>i</mi></msub><mo>=</mo>
            <mn>1</mn><mo>{</mo>
            <msub><mi>parsed</mi><mi>i</mi></msub><mo>=</mo><mn>1</mn><mo>,</mo>
            <msub><mi>truncated</mi><mi>i</mi></msub><mo>=</mo><mn>0</mn><mo>,</mo>
            <msub><mover><mi>N</mi><mo>^</mo></mover><mi>i</mi></msub>
            <mo>=</mo><msub><mi>N</mi><mi>i</mi></msub><mo>}</mo><mo>,</mo>
            <mspace width="1em"/>
            <mi>Accuracy</mi><mo>=</mo>
            <mfrac><mrow><munderover><mo>∑</mo><mi>i</mi><mi>n</mi></munderover>
            <msub><mi>Y</mi><mi>i</mi></msub></mrow><mi>n</mi></mfrac>
          </mrow>
        </math>
      </div>
      <p class="equation-note">Parse failure, wrong format, and truncation remain failures; no request is removed from the denominator.</p>
    </div>
    """


def math_accuracy_law() -> str:
    return """
    <div class="math-equation">
      <div class="equation-title">Retained exact-accuracy response surface</div>
      <div class="math-scroll">
        <math display="block" aria-label="Model-specific accuracy law">
          <mrow>
            <mi>logit</mi><mo>(</mo>
            <msub><mi>p</mi><mi>m</mi></msub><mo>(</mo><mi>T</mi><mo>,</mo><mi>N</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>o</mi><mo>)</mo><mo>)</mo>
            <mo>=</mo>
            <msub><mi>α</mi><mrow><mi>m</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>o</mi></mrow></msub>
            <mo>+</mo>
            <msub><mi>β</mi><mrow><mi>m</mi><mo>,</mo><mi>T</mi></mrow></msub>
            <msub><mi>log</mi><mn>2</mn></msub><mo>(</mo><mfrac><mi>T</mi><mn>5000</mn></mfrac><mo>)</mo>
            <mo>+</mo>
            <msub><mi>β</mi><mrow><mi>m</mi><mo>,</mo><mi>N</mi></mrow></msub>
            <msub><mi>log</mi><mn>2</mn></msub><mo>(</mo><mfrac><mi>N</mi><mn>5</mn></mfrac><mo>)</mo>
          </mrow>
        </math>
      </div>
      <p class="equation-note">The large mode × query-order differences are retained in the condition intercept; length and needle slopes are allowed to differ by model.</p>
    </div>
    """


def math_bias_law() -> str:
    return """
    <div class="math-equation">
      <div class="equation-title">Model-wise signed-bias family retained from the prior analysis</div>
      <div class="math-scroll">
        <math display="block" aria-label="Signed bias response surface">
          <mrow>
            <mi>E</mi><mo>[</mo><mi>asinh</mi><mo>(</mo>
            <mover><mi>N</mi><mo>^</mo></mover><mo>−</mo><mi>N</mi><mo>)</mo>
            <mo>|</mo><mi>parsed</mi><mo>,</mo><mi>m</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>o</mi><mo>,</mo><mi>T</mi><mo>,</mo><mi>N</mi><mo>]</mo>
            <mo>=</mo>
            <msub><mi>α</mi><mrow><mi>m</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>o</mi></mrow></msub>
            <mo>+</mo><msub><mi>f</mi><mi>m</mi></msub><mo>(</mo><mi>T</mi><mo>,</mo><mi>N</mi><mo>)</mo>
          </mrow>
        </math>
      </div>
      <p class="equation-note">Bias is prediction minus truth and is defined only for parsed numeric outputs. The selected function fₘ differs by model; unsupported models retain condition-only baselines.</p>
    </div>
    """


def build_html(
    *,
    model_query: pd.DataFrame,
    mode_query: pd.DataFrame,
    mode_effects: pd.DataFrame,
    anomaly_flags: pd.DataFrame,
    anomaly_cases: pd.DataFrame,
    enum_fidelity: pd.DataFrame,
    anomaly_examples: dict[str, dict[str, Any]],
    low_diagnostics: pd.DataFrame,
    enum_aggregation: pd.DataFrame,
    accuracy_fits: pd.DataFrame,
    bias_fits: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    generated_at: str,
) -> str:
    q8_enum = mode_query[
        mode_query["model_label"].eq("Qwen3-8B")
        & mode_query["prompt_mode"].eq("enumeration")
    ].iloc[0]
    q32_enum = mode_query[
        mode_query["model_label"].eq("Qwen3-32B")
        & mode_query["prompt_mode"].eq("enumeration")
    ].iloc[0]
    ge_enum = mode_query[
        mode_query["model_label"].eq("Gemma4-E4B")
        & mode_query["prompt_mode"].eq("enumeration")
    ].iloc[0]
    llama2 = mode_query[
        mode_query["model_label"].eq("Llama3.2-3B")
        & mode_query["prompt_mode"].eq("enumeration")
    ].iloc[0]
    diagnostics = low_diagnostics.set_index("model_label")
    q17 = diagnostics.loc["Qwen3-1.7B"]
    gemma12 = diagnostics.loc["Gemma4-12B"]
    llama31 = diagnostics.loc["Llama3.1-8B"]
    llama32 = diagnostics.loc["Llama3.2-3B"]

    style = r"""
    :root {
      --ink: #1d262b;
      --muted: #667078;
      --line: #dfe3e6;
      --soft: #f5f7f7;
      --accent: #245f56;
      --accent-soft: #e9f2ef;
      --warm: #9d4e2f;
      --warm-soft: #f8eee8;
      --blue: #315f8c;
      --page: #ffffff;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      font-size: 16px;
      line-height: 1.72;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: linear-gradient(125deg, #f4f8f7, #ffffff 65%);
    }
    .header-inner, main, .nav-inner {
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
    }
    .header-inner { padding: 56px 0 42px; }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--accent);
      font-size: .78rem;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 {
      max-width: 900px;
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2rem, 4.5vw, 3.5rem);
      line-height: 1.12;
      letter-spacing: -.025em;
    }
    .subtitle {
      max-width: 850px;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 1.08rem;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 22px;
      margin-top: 24px;
      color: var(--muted);
      font-size: .88rem;
    }
    nav {
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.96);
      backdrop-filter: blur(8px);
    }
    .nav-inner {
      display: flex;
      gap: 22px;
      overflow-x: auto;
      padding: 12px 0;
      scrollbar-width: thin;
    }
    nav a {
      flex: 0 0 auto;
      color: #435159;
      font-size: .86rem;
      text-decoration: none;
    }
    nav a:hover { color: var(--accent); }
    main { padding: 34px 0 90px; }
    section {
      padding: 38px 0 48px;
      border-top: 1px solid var(--line);
      scroll-margin-top: 55px;
    }
    section:first-child { border-top: 0; padding-top: 18px; }
    h2 {
      margin: 0 0 14px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(1.55rem, 3vw, 2.15rem);
      line-height: 1.25;
    }
    h3 {
      margin: 0 0 12px;
      font-size: 1.05rem;
      line-height: 1.4;
    }
    p { margin: 10px 0 16px; }
    .lead { max-width: 900px; color: var(--muted); font-size: 1.02rem; }
    h4 { margin: 0 0 8px; font-size: .92rem; color: #344b50; }
    .prompt-grid, .method-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin: 20px 0;
    }
    .prompt-grid > div {
      min-width: 0;
      padding: 15px 17px;
      border: 1px solid var(--line);
      background: #fafbfb;
    }
    pre {
      margin: 10px 0;
      padding: 14px 16px;
      overflow-x: auto;
      border: 1px solid #d9dfe0;
      border-left: 4px solid var(--accent);
      background: #f7f9f9;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: .82rem;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .method-box, .conclusion-box {
      padding: 15px 18px;
      border-left: 4px solid var(--blue);
      background: #edf3f8;
    }
    .conclusion-box {
      border-left-color: var(--accent);
      background: var(--accent-soft);
    }
    .method-box span, .conclusion-box span {
      display: block;
      color: var(--blue);
      font-size: .75rem;
      font-weight: 850;
      letter-spacing: .1em;
      text-transform: uppercase;
    }
    .conclusion-box span { color: var(--accent); }
    .method-box p, .conclusion-box p {
      margin: 5px 0 0;
      font-size: .91rem;
    }
    .answer-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0 34px;
      margin: 28px 0 6px;
      border-top: 2px solid var(--ink);
      border-bottom: 1px solid var(--line);
    }
    .answer {
      padding: 20px 0;
      border-bottom: 1px solid var(--line);
    }
    .answer:nth-last-child(-n+2) { border-bottom: 0; }
    .answer .number {
      color: var(--accent);
      font-family: Georgia, serif;
      font-size: 1.45rem;
      font-weight: 700;
    }
    .answer strong { display: block; margin: 4px 0 4px; }
    .answer p { margin: 0; color: var(--muted); font-size: .93rem; }
    .callout {
      margin: 24px 0;
      padding: 18px 22px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
    }
    .callout.warning {
      border-left-color: var(--warm);
      background: var(--warm-soft);
    }
    .callout p:last-child { margin-bottom: 0; }
    .table-wrap {
      width: 100%;
      margin: 22px 0 14px;
      overflow-x: auto;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }
    table {
      width: 100%;
      min-width: 760px;
      border-collapse: collapse;
      font-size: .88rem;
      line-height: 1.45;
    }
    table.compact { min-width: 660px; }
    th, td {
      padding: 11px 12px;
      border-bottom: 1px solid #e9ecee;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: #4d5960;
      background: #f6f7f7;
      font-size: .76rem;
      font-weight: 800;
      letter-spacing: .025em;
      text-transform: uppercase;
    }
    tbody tr:last-child td { border-bottom: 0; }
    .numeric td:not(:first-child), .numeric th:not(:first-child) {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    figure {
      width: min(100%, 1080px);
      margin: 30px auto 38px;
    }
    figure img {
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      background: #fff;
    }
    figcaption {
      margin-top: 10px;
      color: var(--muted);
      font-size: .86rem;
      line-height: 1.58;
    }
    .model-detail {
      padding: 26px 0 22px;
      border-top: 1px solid var(--line);
    }
    .model-detail:first-child { border-top: 0; }
    .model-detail .table-wrap { margin: 12px 0 10px; }
    .model-detail p { max-width: 980px; color: #4e5960; }
    .diagnosis-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0 34px;
      margin-top: 24px;
    }
    .diagnosis {
      padding: 18px 0;
      border-top: 1px solid var(--line);
    }
    .diagnosis p { margin: 5px 0 0; color: var(--muted); }
    .anomaly-stack { margin: 20px 0 26px; }
    details.anomaly-case { margin: 0; }
    details.anomaly-case summary {
      display: flex;
      align-items: baseline;
      gap: 12px;
      padding: 15px 0;
    }
    .anomaly-id {
      flex: 0 0 auto;
      color: var(--warm);
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.05rem;
      font-weight: 800;
    }
    details.anomaly-case p {
      max-width: 980px;
      margin: 7px 0;
    }
    details.anomaly-case .caveat { color: var(--muted); }
    .paired-example {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .paired-example > div {
      min-width: 0;
      padding: 13px 15px;
      border: 1px solid var(--line);
      background: var(--soft);
    }
    pre {
      margin: 8px 0 0;
      overflow-x: auto;
      white-space: pre-wrap;
      color: #263238;
      font: .84rem/1.55 Consolas, "SFMono-Regular", monospace;
    }
    .math-equation {
      width: min(100%, 1040px);
      margin: 28px auto;
      padding: 0 8px;
      counter-increment: equation;
    }
    main { counter-reset: equation; }
    .equation-title {
      max-width: 920px;
      margin: 0 auto 8px;
      font-size: .88rem;
      font-weight: 700;
    }
    .math-scroll {
      display: grid;
      grid-template-columns: minmax(max-content, 1fr) 3rem;
      align-items: center;
      width: 100%;
      overflow-x: auto;
      padding: 7px 0;
    }
    .math-scroll::after {
      content: "(" counter(equation) ")";
      grid-column: 2;
      justify-self: end;
      color: var(--muted);
      font-family: "Times New Roman", serif;
    }
    math[display="block"] {
      grid-column: 1;
      justify-self: center;
      width: max-content;
      margin: 0 1rem;
      font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif;
      font-size: 1.26rem;
    }
    .equation-note {
      max-width: 920px;
      margin: 7px auto 0;
      color: var(--muted);
      font-size: .88rem;
    }
    details {
      margin: 18px 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }
    summary {
      cursor: pointer;
      padding: 13px 0;
      font-weight: 700;
    }
    details > div { padding: 0 0 18px; }
    .links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 18px;
    }
    .links a {
      color: var(--accent);
      text-decoration-thickness: 1px;
      text-underline-offset: 3px;
    }
    code {
      padding: .12em .32em;
      border-radius: 3px;
      background: #f0f2f2;
      font-size: .9em;
    }
    footer {
      padding: 28px 0 45px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: .82rem;
    }
    @media (max-width: 780px) {
      .header-inner, main, .nav-inner { width: min(100% - 24px, 1180px); }
      .header-inner { padding: 38px 0 30px; }
      .answer-grid, .diagnosis-list, .paired-example, .prompt-grid, .method-grid { grid-template-columns: 1fr; }
      .answer:nth-last-child(-n+2) { border-bottom: 1px solid var(--line); }
      .answer:last-child { border-bottom: 0; }
      section { padding: 30px 0 38px; }
      .math-scroll { grid-template-columns: minmax(max-content, 1fr) 2.4rem; }
      math[display="block"] { font-size: 1.05rem; margin: 0 .6rem; }
    }
    @media print {
      nav { display: none; }
      body { font-size: 11pt; }
      section { break-inside: auto; }
      figure, .model-detail, .math-equation { break-inside: avoid; }
    }
    """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Realistic CoT NiaH Count：准确率、Query 顺序与推理模式</title>
  <meta name="description" content="Eight-model Realistic NiaH exact-accuracy report split by query placement and reasoning mode.">
  <style>{style}</style>
</head>
<body>
<!-- {REPORT_MARKER} -->
<header>
  <div class="header-inner">
    <p class="eyebrow">Realistic CoT NiaH Count · Core results</p>
    <h1>准确率、Query 顺序与推理模式</h1>
    <p class="subtitle">重写后的主报告先回答三个问题：八个模型到底有多准；query 放在 passage 前后会怎样；每个模型内部的 direct、enumeration 与 native thinking 有什么差别。低准确率的原因随后单独诊断，empirical-law 拟合保留在最后。</p>
    <div class="meta">
      <span>6,300 requests</span><span>8 models</span><span>3 passage lengths</span>
      <span>10 needle counts</span><span>5 seeds</span><span>Generated {esc(generated_at)}</span>
    </div>
  </div>
</header>
<nav>
  <div class="nav-inner">
    <a href="#answers">结论</a>
    <a href="#setup">实验设定</a>
    <a href="#query-order">准确率与 query 顺序</a>
    <a href="#modes">逐模型模式比较</a>
    <a href="#anomalies">异常现象与机制</a>
    <a href="#low-accuracy">低准确率原因</a>
    <a href="#laws">Empirical law</a>
    <a href="#reproducibility">复现与完整数据</a>
  </div>
</nav>
<main>
  <section id="answers">
    <h2>先给答案</h2>
    <p class="lead">下面四点是主结果。所有百分比均为 exact accuracy；parse failure、格式失败和截断都保留为失败。</p>
    <div class="answer-grid">
      <div class="answer">
        <span class="number">01</span>
        <strong>总体最高的是 Qwen3-32B，最低的是 Llama3.2-3B。</strong>
        <p>总体准确率从 Qwen3-32B 的 76.2%、Gemma4-E4B 的 69.9%、Qwen3-8B 的 68.0%，下降到 Llama3.2-3B 的 13.7%。但总分不能解释 query-order 反转。</p>
      </div>
      <div class="answer">
        <span class="number">02</span>
        <strong>Query last 并不普遍更好；效应由模型和模式共同决定。</strong>
        <p>Qwen enumeration 强烈偏好 query last（8B: {pct(q8_enum['query_first_accuracy'])} → {pct(q8_enum['query_last_accuracy'])}; 32B: {pct(q32_enum['query_first_accuracy'])} → {pct(q32_enum['query_last_accuracy'])}），Gemma4-E4B enumeration 则相反（{pct(ge_enum['query_first_accuracy'])} → {pct(ge_enum['query_last_accuracy'])}）。</p>
      </div>
      <div class="answer">
        <span class="number">03</span>
        <strong>Native thinking 对五个支持它的模型全部有明显收益。</strong>
        <p>相对 direct 的总体提升为 +36.7 至 +56.0 pp。OLMo 与两个 Llama 在本实验中没有注册 native thinking，不能把缺失值解释为 0。</p>
      </div>
      <div class="answer">
        <span class="number">04</span>
        <strong>低准确率并非单一原因。</strong>
        <p>Qwen3-1.7B 与 OLMo 主要是可解析的数值计数错误；Llama3.1 混合了 direct undercount 与 enumeration 格式失败；Llama3.2 同时存在格式、重复、幻觉和计数错误；Gemma4-12B 则是严重的模式特定截断。</p>
      </div>
    </div>
  </section>

  <section id="setup">
    <h2>实验设定与读表规则</h2>
    <p class="lead">每个已注册模式包含 3 个 canonical passage 长度（2k、5k、10k tokens）× 10 个真实 needle 数量（1、2、3、4、5、6、8、10、20、30）× 5 seeds × 2 个 query orders，共 300 个请求。下面把数据生成、prompt 与解码口径全部写出，避免把模板差异误当成模型能力差异。</p>
    <ul>
      <li><strong>Master stimuli：</strong>150 条 = 3 个 T × 10 个 N × 5 seeds；同一 raw-text passage 在所有模型、模式与 query order 中复用。</li>
      <li><strong>长度 T：</strong>使用 canonical tokenizer <code>Qwen/Qwen3-8B</code>，在插入全部 needles 后、加入 task/chat template 前计算。固定 T 时增加 N 会相应缩短 filler，因此不会机械地同时增加总 passage 长度。</li>
      <li><strong>Needle density：</strong><code>ρ = N / (T/1000) = 1000N/T</code>；每条 needle 是唯一 city-score audit record，单个 stimulus 内 city 与 score 都不重复。</li>
      <li><strong>Query first</strong>：任务问题/输出约束位于 passage 之前；<strong>query last</strong>：位于 passage 之后。</li>
      <li><strong>Direct</strong>：直接给出总数；<strong>enumeration</strong>：列出识别到的记录并给出总数；<strong>native thinking</strong>：使用该模型官方 chat template 的 thinking 开关。</li>
      <li>Qwen3 与 Gemma4 共 5 个模型运行三种模式，各 900 条；OLMo 与两个 Llama 只注册 direct + enumeration，各 600 条。</li>
      <li><strong>运行环境：</strong>Lambda Cloud 单卡 NVIDIA H100 PCIe 80 GB，BF16、vLLM 0.25.1、Transformers 5.14.1；仓库 commit <code>090d9838…1c9</code>，stimuli SHA256 <code>374dc935…a5f1</code>。</li>
      <li>Primary metric 的分母始终包含所有请求；只有 signed bias/absolute error 才条件于成功解析的数值输出。</li>
    </ul>
    {math_accuracy_definition()}
    {prompt_and_decoding_protocol()}
    <details>
      <summary>查看八个 checkpoint 的 tokenizer-rendered wrapper 摘要</summary>
      <div>
        {prompt_protocol_table(prompt_summary)}
        <p>完整 42 组 model × mode × query-order 示例与 SHA256 保留在
        <a href="tables/model_prompt_format_examples.csv">prompt-format CSV</a> 和
        <a href="prompt_formats/model_prompt_formats.json">prompt-format JSON</a>。</p>
      </div>
    </details>
    {method_conclusion(
        "正式网格由 150 条 master stimuli 与已注册模型/模式/order 做笛卡尔展开；结构 QC 核对 6,300 个唯一 request_id、T/N/seed 完整网格、context hash、stimuli SHA256、manifest 与 parser 字段。",
        "分析中的 T、N、ρ、mode 与 query order 都是冻结实验变量；prompt、解码和 parser 没有根据结果事后修改。不同模型 tokenizer 下的实际 passage/input tokens另存，但跨模型主图统一使用 canonical T。",
    )}
  </section>

  <section id="query-order">
    <h2>1. 先拆开 Query first 与 Query last</h2>
    <p class="lead">表中的差值为 <code>query_last − query_first</code>。95% 区间按 stimulus 聚类 bootstrap；同一 stimulus 在不同模式中的结果一起重采样，避免把重复条件当作完全独立样本。</p>
    {model_accuracy_table(model_query)}
    <figure>
      <img src="assets/core01_query_order_accuracy.png" alt="Eight-model dumbbell plot comparing exact accuracy for query first and query last.">
      <figcaption><strong>Figure 1.</strong> 每个模型把所有已注册模式合并后，query first 与 query last 的 exact accuracy。横轴是全部请求准确率，连线只表示同一模型两种 query placement 的差异。该图适合看总体方向，但会掩盖 Figure 2 中的 mode-specific 反转。</figcaption>
    </figure>
    <div class="callout warning">
      <strong>最重要的交互：</strong>
      Qwen3-8B 和 Qwen3-32B 的 enumeration 在 query first 时大量把“分数之和”当作“记录数量”，query last 几乎消除了该错误；Gemma enumeration 则在 query first 时检索更完整。Llama3.2-3B 的 enumeration 是另一种模式：query last 的格式失败达到 {pct(llama2['query_last_format_failure_rate'])}，因此把所有模型平均成一个 query-order 主效应会产生误导。
    </div>
    <figure>
      <img src="assets/core02_mode_query_accuracy.png" alt="Faceted heatmap of exact accuracy by model, prompt mode, and query placement.">
      <figcaption><strong>Figure 2.</strong> 八个面板分别对应一个模型；行是 direct、enumeration、native thinking，列是 query first/query last，单元格为全部长度、needle 数量和 seeds 上的 exact accuracy。灰色 “not run” 表示模式未注册，不是失败。</figcaption>
    </figure>
    {method_conclusion(
        "总体表按模型合并该模型已注册 modes；mode-specific 表保持 model×mode。差值定义为 query_last−query_first，并按完全相同 stimulus 配对；95% CI 用 5,000 次 stimulus-cluster bootstrap，异常条目另报告 exact McNemar 检验。",
        "不存在全模型共享的 query-last 主效应。Qwen enumeration 从 query last 获益很大；Gemma4-E4B enumeration 相反；部分 direct/native 条件还有方向反转。因此后续机制与回归必须保留 mode×order 条件。",
    )}
  </section>

  <section id="modes">
    <h2>2. 再在每个模型内部比较三种模式</h2>
    <p class="lead">每个表先保留 query-order 拆分，再给出模式总体准确率。<code>Δ vs direct</code> 使用完全相同 stimulus 和 query order 的配对请求；因此它比跨模型比较更直接地回答“这种推理模式对该模型是否有帮助”。</p>
    {model_cards(mode_query, mode_effects)}
    {method_conclusion(
        "在每个模型内部，以 stimulus_id×query_order 将 enumeration/native-thinking 与 direct 一一配对；报告 comparison−direct 的平均 exact-correct 差值、cluster-bootstrap 95% CI 与 discordant-pair McNemar p-value。未注册 native thinking 的模型保持缺失。",
        "五个支持 native thinking 的模型都得到明显准确率提升；enumeration 则高度依赖模型与 query placement。模式收益不能从一个模型外推到另一模型，也不能把未运行的 native thinking 当作 0。",
    )}
  </section>

  <section id="anomalies">
    <h2>3. 异常现象：先定位发生在哪一步，再解释原因</h2>
    <p class="lead">为避免只挑最显眼的例子，本节先冻结四条筛选规则，再逐项核对原始请求。规则覆盖大幅 query-order 跳变、模式相对 direct 的大幅变化、接口/生成失败，以及“已经完整检索却给错 Total”的 aggregation failure。完整命中清单保存在 CSV；下面十个案例将同一机制的重复命中合并解释。</p>
    {anomaly_flag_summary(anomaly_flags)}
    <p>这些规则共标记 {len(anomaly_flags)} 个 model × mode × order/contrast 条件。它们是诊断入口，不是新的显著性阈值；query-order 案例另外保留成对 discordance 与 exact McNemar p-value。<a href="tables/core_anomaly_flags.csv">下载完整 anomaly flags</a>。</p>

    <h3>先澄清 Qwen enumeration 的“准确率跳变”到底测到了什么</h3>
    <p>Primary exact accuracy 只检查最终计数是否等于真实 <em>N</em>；完整 enumeration 还要求城市—分数记录无缺失、无幻觉、无重复。因此，“final-count accuracy”与“complete record-set fidelity”必须同时报告。下表中的相关系数和相对距离只在“记录已完整但 Total 错误”的请求上计算。</p>
    {qwen_enumeration_fidelity_table(enum_fidelity)}
    <div class="callout warning">
      <strong>最关键的结论：</strong>
      Qwen3-8B query-first enumeration 的低分不是检索崩溃：146/150 个请求找全记录，98 个却在最后的 Total 上失败。预测 Total 与所列 scores 之和高度相关，因此最符合数据的解释是 aggregation operator 选错；而 95.3% 的 query-last 分数也不等于 95.3% 的完整清单正确率。
    </div>

    <h3>逐项机制诊断</h3>
    <p>每个条目严格区分 measured evidence、best-supported explanation 与尚未被当前实验隔离的因果机制。展开条目即可查看。</p>
    {anomaly_case_cards(anomaly_cases, anomaly_examples)}
    <div class="links">
      <a href="tables/core_anomaly_case_summary.csv">下载十个异常案例摘要</a>
      <a href="tables/core_enumeration_fidelity.csv">下载 enumeration fidelity / arithmetic audit</a>
    </div>
    {method_conclusion(
        "先按四条冻结规则穷举异常：|query-order 差|≥20 pp、|mode−direct|≥20 pp、格式/解析/截断率≥20%，或“完整检索但 Total 错误”率≥10%。再回到 request-level 输出与 enumeration 记录集合逐项核对；query-order 差异同时报告配对 McNemar 检验。这里的规则是诊断阈值，不是新增的显著性门槛。",
        f"四类规则共标记 {len(anomaly_flags)} 个条件，并归并成 10 个可解释案例。Qwen3-8B enumeration 的 32.0%→95.3% 主要不是 retrieval 突然改善，而是 query-first 下大量把 score 的和当作记录数；更一般地，异常多来自 prompt placement、输出通道与 counting/aggregation operator 的交互，不能只用模型规模解释。",
    )}
  </section>

  <section id="low-accuracy">
    <h2>4. 为什么几个模型或模式的准确率很低？</h2>
    <p class="lead">这里不把所有错误都称为“不会数数”。先用 parse success 把接口/输出失败和已解析的数值错误分开，再看 undercount、overcount 与 truncation。</p>
    {low_diagnostic_table(low_diagnostics)}
    <p><strong>Exact | parsed</strong> 是成功得到数值预测后，预测恰好正确的比例；它不替代 primary accuracy，只用于识别失败机制。例如 Gemma4-12B 的总体准确率只有 {pct(gemma12['overall_accuracy'])}，但 <em>exact | parsed</em> 达到 {pct(gemma12['exact_given_parsed'])}，说明其主要问题是输出没有完成，而非已给出的数字普遍错误。</p>
    <figure>
      <img src="assets/core03_low_accuracy_failure_budget.png" alt="Stacked failure budgets for the low-accuracy and mode-collapse models, split by query placement.">
      <figcaption><strong>Figure 3.</strong> 每个横条的分母是相应 model × mode × query-order 的全部 150 个请求。绿色为 exact correct；其余依次是 undercount、overcount、非截断格式/解析失败与 truncation。左右面板让 query placement 导致的错误类型变化直接可见。</figcaption>
    </figure>
    <div class="diagnosis-list">
      <div class="diagnosis">
        <h3>Qwen3-1.7B：主要是数值计数，不是 parser</h3>
        <p>Parse success 为 {pct(q17['parse_success_rate'])}，但 exact | parsed 只有 {pct(q17['exact_given_parsed'])}。Direct query first 150 条中有 126 条 undercount；native thinking 把总体准确率从 25.0% 提高到 61.7%。</p>
      </div>
      <div class="diagnosis">
        <h3>Gemma4-12B：generation-budget / channel collapse</h3>
        <p>Direct 的 300 条中 293 条截断，native thinking 也有 116 条截断；enumeration 却达到 90.0%，且 query first 为 100%。因此该异常是模式与模板输出路径特定的。</p>
      </div>
      <div class="diagnosis">
        <h3>OLMo-Hybrid-7B：解析正常，聚合和计数错误</h3>
        <p>Parse success 为 97.2%，但 undercount 与 overcount 各约四分之一。Enumeration 的 101 个 overcount 中，38 个已经完整找对记录，却仍输出错误总数。</p>
      </div>
      <div class="diagnosis">
        <h3>Llama3.1-8B：direct undercount + enumeration 格式失败</h3>
        <p>Direct 的 51.3% 是 undercount；enumeration 的 53.0% 是非截断格式/解析失败。Query last 将总体准确率从 21.3% 提高到 39.3%，但没有消除两种机制。</p>
      </div>
      <div class="diagnosis">
        <h3>Llama3.2-3B：接口与 counting 同时失效</h3>
        <p>总体 parse success 只有 {pct(llama32['parse_success_rate'])}，而 exact | parsed 也只有 {pct(llama32['exact_given_parsed'])}。Enumeration 出现重复记录、幻觉记录、分数求和与缺少合规 final total；direct 则同时 overcount 和 undercount。</p>
      </div>
      <div class="diagnosis">
        <h3>低准确率不是“模型越小越差”的单变量结论</h3>
        <p>同一模型在 query placement 和模式之间可以跨越几十个百分点。当前数据支持的是 prompt/template 与 counting mechanism 的交互，不支持把差异简单归因于参数量或架构。</p>
      </div>
    </div>

    <h3 style="margin-top:34px">Enumeration 的关键错误：找对记录，但把分数相加</h3>
    <p>“Complete retrieval, wrong total” 要求 missing=0、hallucinated=0、duplicate=0 且列出的记录数等于真实 N；在这个严格条件下仍然 overcount，说明错误发生在最终 aggregation/answer 阶段。</p>
    {enumeration_mechanism_table(enum_aggregation)}
    <figure>
      <img src="assets/core04_enumeration_aggregation.png" alt="Enumeration overcount rates and complete-retrieval wrong-total rates by model.">
      <figcaption><strong>Figure 4.</strong> 横轴是全部 enumeration 请求中的比例。浅色表示所有 overcount，深色表示已经完整、无幻觉、无重复地检索到 N 条记录，却仍给出错误 final total。Qwen3-8B 与 Qwen3-32B 的深浅条几乎相同，直接指向“把 numeric scores 求和”这一 aggregation bug。</figcaption>
    </figure>
    {method_conclusion(
        "对每条请求依次判定 exact correct、可解析 undercount、可解析 overcount、非截断格式/解析失败与 truncation；五类互斥且覆盖全部请求。Exact | parsed 只在成功解析出数值的请求中计算，用于区分“接口没有完成”与“给出数字但数错”，不替代全请求 primary accuracy。Enumeration 另核对 missing、hallucinated、duplicate 与 listed-record 数。",
        "极低准确率不是同一种失败：Qwen3-1.7B 与 OLMo 以可解析计数偏差为主；Gemma4-12B 主要是 direct/native generation-budget 截断；Llama3.1 混合 direct undercount 与 enumeration 格式失败；Llama3.2 同时存在接口失败和计数失败。因而后续 law 必须分模型、分 mode，不能把所有错误压成一个统一噪声项。",
    )}
  </section>

  <section id="laws">
    <h2>5. Empirical-law 拟合保留，但不再打断主结果</h2>
    <p class="lead">前面的拆分说明为什么拟合必须保留 model × mode × query-order 条件项。以下只保留经过 held-out 验证的核心结论；完整候选、参数、fold metrics、OOF predictions 与诊断图仍在原 analysis 目录中。</p>
    {math_accuracy_law()}
    {accuracy_fit_table(accuracy_fits)}
    <p>准确率的最佳候选允许每个模型拥有自己的 length 与 needle slopes；grouped-CV log loss 为 0.490、Brier 为 0.162、ECE 为 1.7%。这优于只含模式/query-order 控制项的 0.563 log loss，但不意味着所有模型共享同一阶数。</p>
    <figure>
      <img src="analysis/counting_mechanism_law_v1/figures/exact_candidate_cv.png" alt="Held-out comparison of exact-accuracy empirical-law candidates.">
      <figcaption><strong>Figure 5.</strong> 原 empirical-law 分析保留的 held-out candidate comparison。指标来自按 stimulus/cell 隔离的验证；模型选择不使用训练集拟合优度。</figcaption>
    </figure>

    {math_bias_law()}
    {bias_fit_table(bias_fits)}
    <p>分模型 signed-bias law 仅在 Qwen3-1.7B、Gemma4-E4B、Llama3.1-8B 与 Llama3.2-3B 上得到稳定支持；其余模型的 L/N 变换未能可靠优于 mode/query-order condition-only baseline。Qwen 三个规模可以共享候选函数族和验证协议，但共享斜率没有 held-out 支持。</p>
    <figure>
      <img src="assets/fig15_model_bias_selected_surfaces.png" alt="Observed and predicted signed-bias surfaces for the selected per-model laws.">
      <figcaption><strong>Figure 6.</strong> 分模型 signed-bias surfaces。横纵坐标分别为 canonical passage length T 与真实 needle 数量 N；颜色为 parsed outputs 上的 asinh signed bias。该图是机制诊断，不改变全请求 exact-accuracy 结论。</figcaption>
    </figure>
    <div class="links">
      <a href="analysis/counting_mechanism_law_v1/report.html">打开完整 counting-mechanism / empirical-law 报告</a>
      <a href="../qwen_query_last_empirical_law_20260725/report.html">打开独立 Qwen query-last 分层拟合报告</a>
      <a href="tables/model_specific_bias_selected_laws.csv">下载分模型 bias-law 选择表</a>
      <a href="tables/local_regression_candidate_comparison.csv">下载 accuracy candidate comparison</a>
    </div>
    {method_conclusion(
        "Accuracy 使用全请求 Bernoulli exact-correct；bias 定义为 predicted_count−N，并仅在成功解析数值的请求上拟合 asinh(bias)，绝对误差拟合 log1p(|bias|)。所有候选均按 stimulus/cell 分组做 held-out 验证，模型选择看 OOF log loss/Brier 或 OOF RMSE，而不是训练集 R²；原分析保留全部候选与失败口径。",
        "八模型 pooled law 只能给出中等强度预测，且没有证据支持所有架构共享同一组 N、T 阶数。新增加的 Qwen 专报固定 query last，并在 3 个规模×3 种 mode 内分别估计同一参数化函数族；其中 Qwen direct 的误差结构最稳定，高准确率 enumeration/native strata 则受天花板效应限制。该结论比强行拟合统一跨模型斜率更可靠。",
    )}
  </section>

  <section id="reproducibility">
    <h2>复现、审计与完整数据</h2>
    <p>本次重写只生成新的汇总表、四张主结果图和新的 root HTML；没有改动 6,300 条 request-level 数据、prompt 快照、原始拟合参数、OOF predictions 或任何 frozen experiment artifact。</p>
    <div class="links">
      <a href="tables/request_level_report.csv">6,300-row request-level CSV</a>
      <a href="tables/core_accuracy_by_model_query.csv">Model × query-order accuracy</a>
      <a href="tables/core_accuracy_by_model_mode_query.csv">Model × mode × query-order accuracy</a>
      <a href="tables/core_query_order_effects_paired.csv">Paired query-order effects</a>
      <a href="tables/core_mode_effects_paired.csv">Paired mode effects</a>
      <a href="tables/core_failure_budget.csv">Failure budgets</a>
      <a href="tables/core_enumeration_aggregation.csv">Enumeration aggregation audit</a>
      <a href="tables/core_enumeration_fidelity.csv">Enumeration fidelity / arithmetic audit</a>
      <a href="tables/core_anomaly_flags.csv">Exhaustive anomaly flags</a>
      <a href="tables/core_anomaly_case_summary.csv">Interpreted anomaly cases</a>
      <a href="scripts/rewrite_core_results_report.py">Rebuild script</a>
      <a href="scripts/audit_core_results_report.py">Static and numerical audit script</a>
      <a href="analysis/full_report_before_core_rewrite_20260725.html">Archived pre-rewrite full HTML</a>
      <a href="SHA256SUMS.tsv">SHA256 manifest</a>
    </div>
    {method_conclusion(
        "重建脚本从冻结的 6,300-row request table 与只读 raw export 重新计算汇总、异常和图；静态审计再次核对 request_id 唯一性、逐模型/逐 mode 行数、关键准确率、enumeration arithmetic、图片可解析性、本地链接与 SHA256 manifest。",
        "报告层的改写没有删除请求、重跑 parser、替换 prompt 或覆盖原始拟合。所有用于本文结论的汇总表、prompt 快照、构建脚本与审计脚本均随报告保存，可从 request level 复算。",
    )}
  </section>
</main>
<footer>
  <div class="header-inner" style="padding:0">
    {REPORT_MARKER} · Primary sample n=6,300 · Signed bias conditional on parsed numeric outputs · No post-hoc request deletion.
  </div>
</footer>
</body>
</html>
"""


def update_manifest(
    root: Path,
    generated_paths: list[Path],
    model_query: pd.DataFrame,
    mode_query: pd.DataFrame,
    anomaly_flags: pd.DataFrame,
    anomaly_cases: pd.DataFrame,
    export_archive: Path,
) -> None:
    path = root / "analysis_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, Any]] = {}
    for artifact in generated_paths:
        rel = str(artifact.relative_to(root))
        artifacts[rel] = {"sha256": sha256_file(artifact), "bytes": artifact.stat().st_size}
    manifest.pop("core_results_report_v2", None)
    manifest["core_results_report_v3"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_requests": 6300,
        "n_unique_request_ids": 6300,
        "primary_estimand": (
            "exact correctness over every registered request; parse failure, "
            "format failure, and truncation retained as failures"
        ),
        "narrative_order": [
            "frozen prompt formats and experimental protocol",
            "query-placement accuracy",
            "within-model prompt/reasoning-mode comparison",
            "exhaustive anomaly flags and mechanism diagnosis",
            "low-accuracy failure mechanism",
            "retained empirical-law fits",
        ],
        "section_contract": (
            "each major scientific section states its calculation method and "
            "the conclusion currently supported by the data"
        ),
        "query_effect_definition": "query_last accuracy minus query_first accuracy",
        "query_effect_interval": (
            f"{BOOTSTRAP_REPLICATES}-replicate stimulus-cluster bootstrap; seed {SEED}"
        ),
        "mode_effect_definition": "paired comparison-mode accuracy minus direct accuracy",
        "native_thinking_missing_policy": (
            "OLMo-Hybrid-7B, Llama3.1-8B, and Llama3.2-3B were not registered "
            "for native thinking; missing is not treated as failure"
        ),
        "low_accuracy_threshold": "overall exact accuracy below 0.50",
        "mode_collapse_exception": (
            "Gemma4-12B included because direct accuracy is 0.0233 with 0.9767 truncation"
        ),
        "anomaly_rules": {
            "absolute_query_order_shift": QUERY_ANOMALY_THRESHOLD,
            "absolute_mode_shift_vs_direct": MODE_ANOMALY_THRESHOLD,
            "interface_failure_rate": INTERFACE_FAILURE_THRESHOLD,
            "complete_retrieval_wrong_total_rate": AGGREGATION_FAILURE_THRESHOLD,
        },
        "anomaly_flag_rows": len(anomaly_flags),
        "interpreted_anomaly_cases": len(anomaly_cases),
        "source_request_table": {
            "path": "tables\\request_level_report.csv",
            "sha256": sha256_file(root / "tables" / "request_level_report.csv"),
            "bytes": (root / "tables" / "request_level_report.csv").stat().st_size,
        },
        "source_export_archive": {
            "path": str(export_archive),
            "sha256": sha256_file(export_archive),
            "bytes": export_archive.stat().st_size,
            "access": "read-only raw enumeration arithmetic audit",
        },
        "headline_model_accuracy": {
            row["model_label"]: float(row["overall_accuracy"])
            for _, row in model_query.iterrows()
        },
        "registered_mode_rows": len(mode_query),
        "artifacts": artifacts,
        "raw_or_frozen_artifacts_modified": False,
    }
    manifest["modified_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def update_readme(root: Path) -> None:
    path = root / "README.md"
    original = path.read_text(encoding="utf-8")
    marker = "<!-- CORE_RESULTS_REPORT_V3 -->"
    legacy_marker = "<!-- CORE_RESULTS_REPORT_V2 -->"
    block = f"""
{marker}
## Core-results narrative rewrite (v3)

The canonical `report.html` is organized in the scientific order used for
interpretation: frozen prompt formats and concrete experimental settings,
query-first/query-last exact accuracy, within-model prompt/reasoning modes,
rule-based anomaly detection and mechanism diagnosis, low-accuracy failure
mechanisms, and finally the retained empirical-law results. Every major
scientific section explicitly states its calculation method and the conclusion
currently supported by the data.

Rebuild this report-only layer with:

```powershell
python scripts/rewrite_core_results_report.py --report-root "<canonical report directory>"
```

The script reads the preserved 6,300-row request table plus the local export
archive (read-only, for enumeration score-summing diagnostics), validates the
frozen model/mode registry, writes auditable summary CSVs and figures, and
refreshes the root SHA256 manifest. It does not modify request-level data,
prompt snapshots, raw outputs, or fitted parameters.

Audit the generated report with:

```powershell
python scripts/audit_core_results_report.py --report-root "<canonical report directory>"
```
"""
    for existing_marker in (marker, legacy_marker):
        if existing_marker in original:
            original = original.split(existing_marker, 1)[0].rstrip()
    atomic_write_text(path, original + "\n\n" + block.strip() + "\n")


def main() -> None:
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--export-archive",
        type=Path,
        help=(
            "Preserved .tar.zst export used read-only for enumeration arithmetic "
            "diagnostics. Defaults to the single archive under <project>/exports."
        ),
    )
    args = parser.parse_args()
    root = args.report_root.resolve()
    if args.export_archive is not None:
        export_archive = args.export_archive.resolve()
    else:
        project_root = root.parents[1]
        candidates = sorted((project_root / "exports").rglob("*.tar.zst"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "expected exactly one export archive under "
                f"{project_root / 'exports'}, found {len(candidates)}"
            )
        export_archive = candidates[0].resolve()
    tables = root / "tables"
    assets = root / "assets"
    scripts = root / "scripts"
    analysis = root / "analysis" / "counting_mechanism_law_v1"

    required_paths = [
        tables / "request_level_report.csv",
        tables / "local_regression_candidate_comparison.csv",
        tables / "model_specific_bias_selected_laws.csv",
        tables / "model_prompt_format_summary.csv",
        analysis / "report.html",
        analysis / "figures" / "exact_candidate_cv.png",
        assets / "fig15_model_bias_selected_surfaces.png",
        root / "analysis_manifest.json",
        export_archive,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing report inputs: {missing}")

    data = pd.read_csv(tables / "request_level_report.csv")
    validate_requests(data)
    accuracy = summarize_accuracy(data)
    failures = summarize_failures(data)
    enum_fidelity, anomaly_examples = load_raw_enumeration_evidence(
        export_archive
    )
    anomaly_flags = summarize_anomaly_flags(
        order_effects=accuracy["order_effects"],
        mode_effects=accuracy["mode_effects"],
        failure_budget=failures["failure_budget"],
        enum_fidelity=enum_fidelity,
    )
    anomaly_cases = summarize_anomaly_cases(
        mode_query=accuracy["mode_query"],
        failure_budget=failures["failure_budget"],
        enum_fidelity=enum_fidelity,
    )
    accuracy_fits = pd.read_csv(tables / "local_regression_candidate_comparison.csv")
    bias_fits = pd.read_csv(tables / "model_specific_bias_selected_laws.csv")
    prompt_summary = pd.read_csv(tables / "model_prompt_format_summary.csv")

    generated_tables = {
        "core_accuracy_by_model_query.csv": accuracy["model_query"],
        "core_accuracy_by_model_mode_query.csv": accuracy["mode_query"],
        "core_query_order_effects_paired.csv": accuracy["order_effects"],
        "core_mode_effects_paired.csv": accuracy["mode_effects"],
        "core_failure_budget.csv": failures["failure_budget"],
        "core_low_accuracy_diagnostics.csv": failures["low_diagnostics"],
        "core_enumeration_aggregation.csv": failures["enum_aggregation"],
        "core_enumeration_fidelity.csv": enum_fidelity,
        "core_anomaly_flags.csv": anomaly_flags,
        "core_anomaly_case_summary.csv": anomaly_cases,
    }
    for name, frame in generated_tables.items():
        write_csv_atomic(frame, tables / name)

    configure_plot_style()
    plot_query_order(accuracy["model_query"], assets / "core01_query_order_accuracy.png")
    plot_mode_query_heatmap(
        accuracy["mode_query"], assets / "core02_mode_query_accuracy.png"
    )
    plot_low_failure_budget(
        failures["failure_budget"], assets / "core03_low_accuracy_failure_budget.png"
    )
    plot_enumeration_aggregation(
        failures["enum_aggregation"], assets / "core04_enumeration_aggregation.png"
    )

    archive = root / "analysis" / "full_report_before_core_rewrite_20260725.html"
    if not archive.exists():
        shutil.copy2(root / "report.html", archive)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = build_html(
        model_query=accuracy["model_query"],
        mode_query=accuracy["mode_query"],
        mode_effects=accuracy["mode_effects"],
        anomaly_flags=anomaly_flags,
        anomaly_cases=anomaly_cases,
        enum_fidelity=enum_fidelity,
        anomaly_examples=anomaly_examples,
        low_diagnostics=failures["low_diagnostics"],
        enum_aggregation=failures["enum_aggregation"],
        accuracy_fits=accuracy_fits,
        bias_fits=bias_fits,
        prompt_summary=prompt_summary,
        generated_at=generated_at,
    )
    atomic_write_text(root / "report.html", report)

    script_destination = scripts / "rewrite_core_results_report.py"
    if Path(__file__).resolve() != script_destination.resolve():
        shutil.copy2(Path(__file__).resolve(), script_destination)
    audit_destination = scripts / "audit_core_results_report.py"
    audit_candidates = [
        Path(__file__).resolve().with_name("main_report_audit.py"),
        Path(__file__).resolve().with_name("audit_core_results_report.py"),
    ]
    for audit_source in audit_candidates:
        if audit_source.is_file():
            if audit_source.resolve() != audit_destination.resolve():
                shutil.copy2(audit_source, audit_destination)
            break

    generated_paths = [
        *(tables / name for name in generated_tables),
        assets / "core01_query_order_accuracy.png",
        assets / "core02_mode_query_accuracy.png",
        assets / "core03_low_accuracy_failure_budget.png",
        assets / "core04_enumeration_aggregation.png",
        root / "report.html",
        script_destination,
        archive,
    ]
    if audit_destination.is_file():
        generated_paths.append(audit_destination)
    update_manifest(
        root,
        generated_paths,
        accuracy["model_query"],
        accuracy["mode_query"],
        anomaly_flags,
        anomaly_cases,
        export_archive,
    )
    update_readme(root)
    refresh_checksums(root, root / "SHA256SUMS.tsv")

    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "status": "PASS",
                "report": str(root / "report.html"),
                "request_rows": len(data),
                "unique_request_ids": data["request_id"].nunique(),
                "model_rows": len(accuracy["model_query"]),
                "mode_rows": len(accuracy["mode_query"]),
                "anomaly_flag_rows": len(anomaly_flags),
                "anomaly_cases": len(anomaly_cases),
                "generated_tables": len(generated_tables),
                "generated_figures": 4,
                "elapsed_seconds": round(elapsed, 3),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
