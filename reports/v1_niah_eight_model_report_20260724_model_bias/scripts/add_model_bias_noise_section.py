#!/usr/bin/env python3
"""Add a model-wise signed-bias/noise analysis to the canonical HTML report.

The primary scientific estimand remains exact correctness over all requests.
Signed bias is a secondary, conditional estimand defined only for successfully
parsed numeric predictions.  This script:

1. summarizes bias tails separately for every model and prompt mode;
2. compares shared and model-specific Qwen bias surfaces under grouped CV;
3. writes reproducible CSV tables and two diagnostic figures;
4. inserts an idempotent, self-contained section into the canonical report;
5. records provenance in the report manifest and refreshes checksums.

It never edits the original request records or any frozen experiment artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


MODELS = [
    "Qwen3-8B",
    "Qwen3-1.7B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "OLMo-Hybrid-7B",
    "Llama3.1-8B",
    "Llama3.2-3B",
]
QWEN_MODELS = ["Qwen3-1.7B", "Qwen3-8B", "Qwen3-32B"]
PROMPT_MODES = ["direct", "enumeration", "native_thinking"]
MODE_LABELS = {
    "direct": "Direct / nonthinking",
    "enumeration": "Enumeration",
    "native_thinking": "Native thinking / CoT",
}
RANDOM_SEED = 20260725
BOOTSTRAP_REPLICATES = 500

SECTION_START = "<!-- MODEL_BIAS_NOISE_V1_START -->"
SECTION_END = "<!-- MODEL_BIAS_NOISE_V1_END -->"
STYLE_START = "/* MODEL_BIAS_NOISE_V1_START */"
STYLE_END = "/* MODEL_BIAS_NOISE_V1_END */"
README_MARKER = "<!-- MODEL_BIAS_NOISE_V1 -->"

CANDIDATES = [
    "condition_only",
    "shared_log_length_only",
    "shared_log_needles_only",
    "shared_log_density",
    "shared_log_separable",
    "shared_log_interaction",
    "shared_raw_separable",
    "model_specific_log",
    "model_specific_log_interaction",
    "model_specific_raw",
]
CANDIDATE_LABELS = {
    "condition_only": "Condition only",
    "shared_log_length_only": "Shared log L",
    "shared_log_needles_only": "Shared log N",
    "shared_log_density": "Shared log density",
    "shared_log_separable": "Shared log L + log N",
    "shared_log_interaction": "Shared log L + log N + interaction",
    "shared_raw_separable": "Shared raw L + N",
    "model_specific_log": "Model-specific log L + log N",
    "model_specific_log_interaction": "Model-specific log + interaction",
    "model_specific_raw": "Model-specific raw L + N",
}
CANDIDATE_COMPLEXITY = {
    "condition_only": 0,
    "shared_log_length_only": 1,
    "shared_log_needles_only": 1,
    "shared_log_density": 1,
    "shared_log_separable": 2,
    "shared_raw_separable": 2,
    "shared_log_interaction": 3,
    "model_specific_log": 6,
    "model_specific_raw": 6,
    "model_specific_log_interaction": 9,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_checksums(base: Path, manifest_path: Path) -> None:
    records: list[str] = []
    for path in sorted(base.rglob("*"), key=lambda p: str(p.relative_to(base)).lower()):
        if not path.is_file() or path.resolve() == manifest_path.resolve():
            continue
        records.append(f"{sha256(path)}\t{path.relative_to(base)}")
    manifest_path.write_text(
        "\n".join(records) + "\n", encoding="utf-8", newline="\n"
    )


def trimmed_mean(values: np.ndarray, proportion: float = 0.10) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    if len(array) == 0:
        return float("nan")
    trim = int(math.floor(len(array) * proportion))
    if 2 * trim >= len(array):
        return float(np.mean(array))
    return float(np.mean(array[trim : len(array) - trim]))


def tail_share(values: np.ndarray, proportion: float) -> float:
    absolute = np.abs(np.asarray(values, dtype=float))
    total = float(np.sum(absolute))
    if len(absolute) == 0 or total == 0:
        return 0.0
    count = max(1, int(math.ceil(len(absolute) * proportion)))
    return float(np.sum(np.sort(absolute)[-count:]) / total)


def summarize_bias(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame))
    parsed = frame.loc[
        (frame["parse_success"].astype(int) == 1)
        & frame["signed_error"].notna()
        & frame["predicted_count"].notna()
    ].copy()
    bias = parsed["signed_error"].to_numpy(dtype=float)
    absolute = np.abs(bias)
    if len(parsed):
        mean_bias = float(np.mean(bias))
        trim_bias = trimmed_mean(bias)
        median_bias = float(np.median(bias))
        asinh_center = float(np.sinh(np.mean(np.arcsinh(bias))))
        over_rate = float(np.mean(bias > 0))
        under_rate = float(np.mean(bias < 0))
        exact_parsed = float(np.mean(bias == 0))
        mae = float(np.mean(absolute))
        p95 = float(np.quantile(absolute, 0.95))
        maximum = float(np.max(absolute))
        share1 = tail_share(bias, 0.01)
        share5 = tail_share(bias, 0.05)
    else:
        mean_bias = trim_bias = median_bias = asinh_center = float("nan")
        over_rate = under_rate = exact_parsed = float("nan")
        mae = p95 = maximum = share1 = share5 = float("nan")

    return {
        "requests": total,
        "parsed_requests": int(len(parsed)),
        "parse_rate": float(len(parsed) / total) if total else float("nan"),
        "exact_rate_all": float(frame["exact_correct"].astype(float).mean())
        if total
        else float("nan"),
        "format_failure_rate": float(frame["format_failure"].astype(float).mean())
        if total
        else float("nan"),
        "truncation_rate": float(frame["truncated"].astype(float).mean())
        if total
        else float("nan"),
        "exact_rate_parsed": exact_parsed,
        "mean_bias": mean_bias,
        "trimmed_mean_bias_10pct": trim_bias,
        "median_bias": median_bias,
        "asinh_center_bias": asinh_center,
        "tail_shift": mean_bias - trim_bias,
        "over_rate_parsed": over_rate,
        "under_rate_parsed": under_rate,
        "mean_absolute_error": mae,
        "p95_absolute_error": p95,
        "max_absolute_error": maximum,
        "top1pct_absolute_error_share": share1,
        "top5pct_absolute_error_share": share5,
    }


def make_summary_tables(requests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_records: list[dict[str, Any]] = []
    for model in MODELS:
        record = {"model_label": model}
        record.update(summarize_bias(requests.loc[requests["model_label"] == model]))
        overall_records.append(record)

    mode_records: list[dict[str, Any]] = []
    for model in MODELS:
        model_frame = requests.loc[requests["model_label"] == model]
        for mode in PROMPT_MODES:
            subset = model_frame.loc[model_frame["prompt_mode"] == mode]
            if subset.empty:
                continue
            record = {
                "model_label": model,
                "prompt_mode": mode,
                "prompt_mode_label": MODE_LABELS[mode],
            }
            record.update(summarize_bias(subset))
            mode_records.append(record)

    return pd.DataFrame(overall_records), pd.DataFrame(mode_records)


def add_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["condition"] = (
        out["model_label"].astype(str)
        + "|"
        + out["prompt_mode"].astype(str)
        + "|"
        + out["query_order"].astype(str)
    )
    out["x_log_length"] = np.log2(
        out["target_passage_tokens"].astype(float) / 5000.0
    )
    out["x_log_needles"] = np.log2(out["num_needles"].astype(float) / 5.0)
    out["x_log_density"] = out["x_log_needles"] - out["x_log_length"]
    out["x_raw_length"] = (
        out["target_passage_tokens"].astype(float) - 5000.0
    ) / 5000.0
    out["x_raw_needles"] = (out["num_needles"].astype(float) - 5.0) / 5.0
    out["asinh_bias"] = np.arcsinh(out["signed_error"].astype(float))
    return out


def condition_levels(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["condition"].astype(str).unique())


def build_design(
    frame: pd.DataFrame,
    candidate: str,
    levels: list[str],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    conditions = frame["condition"].astype(str).to_numpy()
    models = frame["model_label"].astype(str).to_numpy()
    for level in levels:
        columns.append((conditions == level).astype(float))
        names.append(f"intercept[{level}]")

    log_l = frame["x_log_length"].to_numpy(dtype=float)
    log_n = frame["x_log_needles"].to_numpy(dtype=float)
    raw_l = frame["x_raw_length"].to_numpy(dtype=float)
    raw_n = frame["x_raw_needles"].to_numpy(dtype=float)

    if candidate == "condition_only":
        pass
    elif candidate == "shared_log_length_only":
        columns.append(log_l)
        names.append("shared_log2_length")
    elif candidate == "shared_log_needles_only":
        columns.append(log_n)
        names.append("shared_log2_needles")
    elif candidate == "shared_log_density":
        columns.append(log_n - log_l)
        names.append("shared_log2_density")
    elif candidate == "shared_log_separable":
        columns.extend([log_l, log_n])
        names.extend(["shared_log2_length", "shared_log2_needles"])
    elif candidate == "shared_log_interaction":
        columns.extend([log_l, log_n, log_l * log_n])
        names.extend(
            [
                "shared_log2_length",
                "shared_log2_needles",
                "shared_log2_length_x_needles",
            ]
        )
    elif candidate == "shared_raw_separable":
        columns.extend([raw_l, raw_n])
        names.extend(["shared_raw_length", "shared_raw_needles"])
    elif candidate in {
        "model_specific_log",
        "model_specific_log_interaction",
        "model_specific_raw",
    }:
        for model in QWEN_MODELS:
            indicator = (models == model).astype(float)
            if candidate == "model_specific_raw":
                columns.extend([indicator * raw_l, indicator * raw_n])
                names.extend(
                    [
                        f"raw_length[{model}]",
                        f"raw_needles[{model}]",
                    ]
                )
            else:
                columns.extend([indicator * log_l, indicator * log_n])
                names.extend(
                    [
                        f"log2_length[{model}]",
                        f"log2_needles[{model}]",
                    ]
                )
                if candidate == "model_specific_log_interaction":
                    columns.append(indicator * log_l * log_n)
                    names.append(f"log2_length_x_needles[{model}]")
    else:
        raise KeyError(candidate)

    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite design for {candidate}")
    return matrix, names


def fit_ols(matrix: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(matrix, outcome, rcond=None)
    if not np.isfinite(coefficients).all():
        raise ValueError("Non-finite OLS coefficients")
    return coefficients


def fold_assignments(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    lengths = sorted(int(v) for v in frame["target_passage_tokens"].unique())
    needles = sorted(int(v) for v in frame["num_needles"].unique())
    length_index = {value: index for index, value in enumerate(lengths)}
    needle_index = {value: index for index, value in enumerate(needles)}
    cell_fold = np.array(
        [
            (
                2 * length_index[int(length)]
                + needle_index[int(needle)]
            )
            % 5
            for length, needle in zip(
                frame["target_passage_tokens"],
                frame["num_needles"],
            )
        ],
        dtype=int,
    )
    return {
        "seed": frame["seed"].to_numpy(),
        "length": frame["target_passage_tokens"].to_numpy(),
        "needle_level": frame["num_needles"].to_numpy(),
        "blocked_cell": cell_fold,
    }


def r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def evaluate_qwen_candidates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    levels = condition_levels(frame)
    fold_maps = fold_assignments(frame)
    outcome = frame["asinh_bias"].to_numpy(dtype=float)
    fold_records: list[dict[str, Any]] = []
    scheme_records: list[dict[str, Any]] = []
    oof_cell_records: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        for scheme, assignments in fold_maps.items():
            predictions = np.full(len(frame), np.nan, dtype=float)
            for fold in sorted(np.unique(assignments), key=lambda value: str(value)):
                test_mask = assignments == fold
                train_mask = ~test_mask
                train = frame.loc[train_mask]
                test = frame.loc[test_mask]
                x_train, _ = build_design(train, candidate, levels)
                x_test, _ = build_design(test, candidate, levels)
                beta = fit_ols(x_train, outcome[train_mask])
                fold_prediction = x_test @ beta
                predictions[test_mask] = fold_prediction
                residual = outcome[test_mask] - fold_prediction
                fold_records.append(
                    {
                        "candidate": candidate,
                        "candidate_label": CANDIDATE_LABELS[candidate],
                        "scheme": scheme,
                        "fold": str(fold),
                        "train_n": int(train_mask.sum()),
                        "test_n": int(test_mask.sum()),
                        "asinh_mae": float(np.mean(np.abs(residual))),
                        "asinh_rmse": float(np.sqrt(np.mean(residual**2))),
                    }
                )
            if not np.isfinite(predictions).all():
                raise ValueError(f"Incomplete OOF predictions for {candidate}/{scheme}")

            eval_frame = frame[
                [
                    "model_label",
                    "prompt_mode",
                    "query_order",
                    "target_passage_tokens",
                    "num_needles",
                ]
            ].copy()
            eval_frame["observed"] = outcome
            eval_frame["predicted"] = predictions
            cells = (
                eval_frame.groupby(
                    [
                        "model_label",
                        "prompt_mode",
                        "query_order",
                        "target_passage_tokens",
                        "num_needles",
                    ],
                    observed=True,
                )[["observed", "predicted"]]
                .mean()
                .reset_index()
            )
            request_residual = outcome - predictions
            cell_residual = (
                cells["observed"].to_numpy(dtype=float)
                - cells["predicted"].to_numpy(dtype=float)
            )
            scheme_records.append(
                {
                    "candidate": candidate,
                    "candidate_label": CANDIDATE_LABELS[candidate],
                    "scheme": scheme,
                    "request_asinh_mae": float(
                        np.mean(np.abs(request_residual))
                    ),
                    "request_asinh_rmse": float(
                        np.sqrt(np.mean(request_residual**2))
                    ),
                    "cell_asinh_mae": float(np.mean(np.abs(cell_residual))),
                    "cell_asinh_rmse": float(np.sqrt(np.mean(cell_residual**2))),
                    "cell_r2": r2_score(
                        cells["observed"].to_numpy(dtype=float),
                        cells["predicted"].to_numpy(dtype=float),
                    ),
                    "cells": int(len(cells)),
                }
            )
            for record in cells.to_dict(orient="records"):
                record.update({"candidate": candidate, "scheme": scheme})
                oof_cell_records.append(record)

    fold_metrics = pd.DataFrame(fold_records)
    scheme_metrics = pd.DataFrame(scheme_records)
    oof_cells = pd.DataFrame(oof_cell_records)

    comparisons: list[dict[str, Any]] = []
    baseline = scheme_metrics.loc[
        scheme_metrics["candidate"] == "condition_only"
    ].set_index("scheme")
    for candidate in CANDIDATES:
        subset = scheme_metrics.loc[scheme_metrics["candidate"] == candidate].copy()
        subset = subset.set_index("scheme").loc[list(fold_maps)]
        base = baseline.loc[list(fold_maps)]
        score = float(subset["request_asinh_mae"].mean())
        base_score = float(base["request_asinh_mae"].mean())
        gain = float(100.0 * (base_score - score) / base_score)
        improvements = (
            base["request_asinh_mae"].to_numpy(dtype=float)
            - subset["request_asinh_mae"].to_numpy(dtype=float)
        )
        comparisons.append(
            {
                "candidate": candidate,
                "candidate_label": CANDIDATE_LABELS[candidate],
                "complexity": CANDIDATE_COMPLEXITY[candidate],
                "mean_grouped_request_asinh_mae": score,
                "condition_only_mean_grouped_asinh_mae": base_score,
                "gain_vs_condition_only_pct": gain,
                "schemes_better_than_condition_only": int(
                    np.sum(improvements > 1e-12)
                ),
                "seed_request_asinh_mae": float(
                    subset.loc["seed", "request_asinh_mae"]
                ),
                "seed_cell_asinh_mae": float(
                    subset.loc["seed", "cell_asinh_mae"]
                ),
                "seed_cell_r2": float(subset.loc["seed", "cell_r2"]),
                "length_holdout_mae": float(
                    subset.loc["length", "request_asinh_mae"]
                ),
                "needle_holdout_mae": float(
                    subset.loc["needle_level", "request_asinh_mae"]
                ),
                "blocked_cell_mae": float(
                    subset.loc["blocked_cell", "request_asinh_mae"]
                ),
            }
        )

    comparison = pd.DataFrame(comparisons).sort_values(
        ["mean_grouped_request_asinh_mae", "complexity", "candidate"]
    )
    return comparison, fold_metrics, scheme_metrics, oof_cells


def bootstrap_shared_qwen_parameters(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    levels = condition_levels(frame)
    full_x, names = build_design(frame, "shared_log_separable", levels)
    outcome = frame["asinh_bias"].to_numpy(dtype=float)
    full_beta = fit_ols(full_x, outcome)
    target_names = ["shared_log2_length", "shared_log2_needles"]
    target_indices = [names.index(name) for name in target_names]

    rng = np.random.default_rng(RANDOM_SEED)
    clusters = sorted(frame["stimulus_id"].astype(str).unique())
    cluster_frames = {
        cluster: frame.loc[frame["stimulus_id"].astype(str) == cluster]
        for cluster in clusters
    }
    draws = np.full((BOOTSTRAP_REPLICATES, len(target_indices)), np.nan)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        bootstrap = pd.concat(
            [cluster_frames[str(cluster)] for cluster in sampled],
            ignore_index=True,
        )
        x_bootstrap, names_bootstrap = build_design(
            bootstrap, "shared_log_separable", levels
        )
        beta = fit_ols(
            x_bootstrap,
            bootstrap["asinh_bias"].to_numpy(dtype=float),
        )
        for position, name in enumerate(target_names):
            draws[replicate, position] = beta[names_bootstrap.index(name)]

    rows: list[dict[str, Any]] = []
    display = {
        "shared_log2_length": "shared_log2_length",
        "shared_log2_needles": "shared_log2_needles",
    }
    for position, name in enumerate(target_names):
        rows.append(
            {
                "scope": "Qwen shared",
                "parameter": display[name],
                "estimate": float(full_beta[target_indices[position]]),
                "ci95_low": float(np.quantile(draws[:, position], 0.025)),
                "ci95_high": float(np.quantile(draws[:, position], 0.975)),
                "bootstrap_unit": "stimulus_id",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    return pd.DataFrame(rows)


def qwen_parameter_table(
    fixed_parameters: pd.DataFrame,
    shared_parameters: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in QWEN_MODELS:
        record = fixed_parameters.loc[
            fixed_parameters["model_label"] == model
        ].iloc[0]
        rows.extend(
            [
                {
                    "scope": model,
                    "parameter": "log2_length",
                    "estimate": float(record["length_slope"]),
                    "ci95_low": float(record["length_slope_ci95_low"]),
                    "ci95_high": float(record["length_slope_ci95_high"]),
                    "bootstrap_unit": "stimulus_id",
                    "bootstrap_replicates": 500,
                },
                {
                    "scope": model,
                    "parameter": "log2_needles",
                    "estimate": float(record["needle_slope"]),
                    "ci95_low": float(record["needle_slope_ci95_low"]),
                    "ci95_high": float(record["needle_slope_ci95_high"]),
                    "bootstrap_unit": "stimulus_id",
                    "bootstrap_replicates": 500,
                },
            ]
        )
    return pd.concat(
        [pd.DataFrame(rows), shared_parameters],
        ignore_index=True,
    )


def save_bias_tail_figure(
    overall: pd.DataFrame,
    by_mode: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 7.3))

    ordered = overall.set_index("model_label").loc[MODELS].reset_index()
    y = np.arange(len(ordered))
    raw = ordered["mean_bias"].to_numpy(dtype=float)
    trimmed = ordered["trimmed_mean_bias_10pct"].to_numpy(dtype=float)
    for index, (left, right) in enumerate(zip(raw, trimmed)):
        axes[0].plot(
            [left, right],
            [index, index],
            color="#8da0a6",
            linewidth=1.8,
            zorder=1,
        )
    axes[0].scatter(
        raw,
        y,
        color="#c65d3b",
        s=55,
        label="Raw mean bias",
        zorder=3,
    )
    axes[0].scatter(
        trimmed,
        y,
        color="#1f6f78",
        s=55,
        marker="D",
        label="10% trimmed mean",
        zorder=3,
    )
    axes[0].axvline(0, color="#6d777b", linestyle="--", linewidth=1)
    axes[0].set_xscale("symlog", linthresh=1.0)
    axes[0].set_yticks(y, ordered["model_label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Signed bias in counts (symmetric-log scale)")
    axes[0].set_title("Tail pull: raw mean versus robust mean", weight="bold")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="x", color="#e1e6e8", linewidth=0.7)

    pivot = (
        by_mode.pivot(
            index="model_label",
            columns="prompt_mode",
            values="top1pct_absolute_error_share",
        )
        .reindex(index=MODELS, columns=PROMPT_MODES)
        .astype(float)
        * 100.0
    )
    cmap = LinearSegmentedColormap.from_list(
        "tail_share",
        ["#f3f7f6", "#c4dcd8", "#5a9b94", "#c65d3b"],
    )
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    image = axes[1].imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    axes[1].set_xticks(
        np.arange(len(PROMPT_MODES)),
        ["Direct", "Enumeration", "Native thinking"],
        rotation=18,
        ha="right",
    )
    axes[1].set_yticks(np.arange(len(MODELS)), MODELS)
    axes[1].set_title(
        "Share of total |bias| carried by largest 1%",
        weight="bold",
    )
    for row_index in range(len(MODELS)):
        for column_index in range(len(PROMPT_MODES)):
            value = pivot.iloc[row_index, column_index]
            if np.isfinite(value):
                color = "white" if value >= 58 else "#17353a"
                axes[1].text(
                    column_index,
                    row_index,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    weight="bold",
                )
            else:
                axes[1].text(
                    column_index,
                    row_index,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#7d888c",
                )
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.045, pad=0.03)
    colorbar.set_label("Top-1% share of total absolute bias (%)")
    fig.suptitle(
        "Model-wise signed-bias tails among parsed numeric predictions",
        fontsize=15.5,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_qwen_figure(
    comparison: pd.DataFrame,
    parameters: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.8))

    plot_comparison = comparison.copy().sort_values(
        ["gain_vs_condition_only_pct", "complexity"]
    )
    y = np.arange(len(plot_comparison))
    colors = [
        "#9ca6aa"
        if candidate == "condition_only"
        else "#c65d3b"
        if candidate.startswith("model_specific")
        else "#1f6f78"
        for candidate in plot_comparison["candidate"]
    ]
    axes[0].barh(
        y,
        plot_comparison["gain_vs_condition_only_pct"],
        color=colors,
        alpha=0.92,
    )
    axes[0].axvline(0, color="#6d777b", linewidth=1, linestyle="--")
    axes[0].set_yticks(y, plot_comparison["candidate_label"])
    axes[0].set_xlabel("Grouped-CV MAE improvement vs condition-only (%)")
    axes[0].set_title("Shared versus model-specific surfaces", weight="bold")
    axes[0].grid(axis="x", color="#e1e6e8", linewidth=0.7)
    for index, value in enumerate(
        plot_comparison["gain_vs_condition_only_pct"].to_numpy(dtype=float)
    ):
        axes[0].text(
            value + (0.12 if value < 0 else -0.12),
            index,
            f"{value:+.1f}%",
            va="center",
            ha="left" if value < 0 else "right",
            fontsize=8.5,
            color="white" if value < -0.7 else "#17353a",
            weight="bold",
        )

    scopes = QWEN_MODELS + ["Qwen shared"]
    y_positions = np.arange(len(scopes))
    axes[1].axvline(0, color="#6d777b", linewidth=1, linestyle="--")
    parameter_style = {
        "log2_length": ("#1f6f78", "o", "Length slope"),
        "shared_log2_length": ("#1f6f78", "o", "Length slope"),
        "log2_needles": ("#c65d3b", "s", "Needle-count slope"),
        "shared_log2_needles": ("#c65d3b", "s", "Needle-count slope"),
    }
    used_labels: set[str] = set()
    for scope_index, scope in enumerate(scopes):
        subset = parameters.loc[parameters["scope"] == scope]
        for _, row in subset.iterrows():
            color, marker, label = parameter_style[str(row["parameter"])]
            label_for_plot = label if label not in used_labels else None
            used_labels.add(label)
            estimate = float(row["estimate"])
            lower = float(row["ci95_low"])
            upper = float(row["ci95_high"])
            offset = -0.10 if "length" in str(row["parameter"]) else 0.10
            axes[1].errorbar(
                estimate,
                scope_index + offset,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=3,
                markersize=6.5,
                label=label_for_plot,
            )
    axes[1].set_yticks(y_positions, scopes)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Change in mean asinh(bias) when L or N doubles")
    axes[1].set_title("Fixed log-coordinate slopes and 95% CI", weight="bold")
    axes[1].grid(axis="x", color="#e1e6e8", linewidth=0.7)
    axes[1].legend(frameon=False, loc="lower left")

    fig.suptitle(
        "Qwen family: shared slopes are not supported by grouped validation",
        fontsize=15.5,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def percentage(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{100.0 * value:.{digits}f}%"


def number(value: float, digits: int = 2, sign: bool = False) -> str:
    if not np.isfinite(value):
        return "—"
    if sign:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def table_html(frame: pd.DataFrame, columns: list[str], rename: dict[str, str]) -> str:
    table = frame.loc[:, columns].rename(columns=rename)
    return table.to_html(
        index=False,
        classes=["data-table", "bias-noise-table"],
        border=0,
        escape=True,
        na_rep="—",
    )


def overall_display(overall: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    merged = overall.merge(
        selected[
            [
                "model_label",
                "selected_candidate",
                "nested_seed_gain_pct",
                "evidence",
            ]
        ],
        on="model_label",
        how="left",
        validate="one_to_one",
    )
    merged["parsed"] = merged.apply(
        lambda row: f"{int(row['parsed_requests'])}/{int(row['requests'])}", axis=1
    )
    merged["parse_pct"] = merged["parse_rate"].map(percentage)
    merged["exact_pct"] = merged["exact_rate_all"].map(percentage)
    merged["mean_b"] = merged["mean_bias"].map(lambda value: number(value, 2, True))
    merged["trim_b"] = merged["trimmed_mean_bias_10pct"].map(
        lambda value: number(value, 2, True)
    )
    merged["median_b"] = merged["median_bias"].map(
        lambda value: number(value, 1, True)
    )
    merged["over_pct"] = merged["over_rate_parsed"].map(percentage)
    merged["under_pct"] = merged["under_rate_parsed"].map(percentage)
    merged["p95_abs"] = merged["p95_absolute_error"].map(
        lambda value: number(value, 1)
    )
    merged["max_abs"] = merged["max_absolute_error"].map(
        lambda value: number(value, 0)
    )
    merged["top1_pct"] = merged["top1pct_absolute_error_share"].map(percentage)
    merged["law_status"] = merged["evidence"].map(
        {
            "supported": "支持模型内 N/L law",
            "weak_support": "弱支持",
            "not_supported": "未发现稳定 N/L law",
        }
    )
    return merged


def mode_display(by_mode: pd.DataFrame) -> pd.DataFrame:
    display = by_mode.copy()
    display["n"] = display["requests"].astype(int).astype(str)
    display["parsed_pct"] = display["parse_rate"].map(percentage)
    display["exact_pct"] = display["exact_rate_all"].map(percentage)
    display["mean_b"] = display["mean_bias"].map(lambda value: number(value, 2, True))
    display["trim_b"] = display["trimmed_mean_bias_10pct"].map(
        lambda value: number(value, 2, True)
    )
    display["median_b"] = display["median_bias"].map(
        lambda value: number(value, 1, True)
    )
    display["over_pct"] = display["over_rate_parsed"].map(percentage)
    display["under_pct"] = display["under_rate_parsed"].map(percentage)
    display["p95_abs"] = display["p95_absolute_error"].map(
        lambda value: number(value, 1)
    )
    display["top1_pct"] = display["top1pct_absolute_error_share"].map(percentage)
    return display


def candidate_display(comparison: pd.DataFrame) -> pd.DataFrame:
    display = comparison.copy()
    display["mae"] = display["mean_grouped_request_asinh_mae"].map(
        lambda value: number(value, 3)
    )
    display["gain"] = display["gain_vs_condition_only_pct"].map(
        lambda value: f"{value:+.1f}%"
    )
    display["wins"] = display["schemes_better_than_condition_only"].map(
        lambda value: f"{int(value)}/4"
    )
    display["seed_r2"] = display["seed_cell_r2"].map(
        lambda value: number(value, 3)
    )
    return display


def per_model_cards(
    overall: pd.DataFrame,
    selected: pd.DataFrame,
) -> str:
    overall_index = overall.set_index("model_label")
    selected_index = selected.set_index("model_label")
    interpretations = {
        "Qwen3-8B": (
            "grouped validation 选择 condition-only：当前网格内没有可复现的平滑 L/N bias 面。"
            "原始均值的正偏主要来自少数巨大 over-count；典型输出更接近零偏。"
        ),
        "Qwen3-1.7B": (
            "支持模型内 raw-separable law：L 增大把典型 bias 推向更正，N 增大则推向更负。"
            "因此它不是简单的“总是多算”，而是带明显负向 N 漂移并叠加右尾。"
        ),
        "Qwen3-32B": (
            "没有稳定胜过 condition-only 的 L/N 曲面。高 exact-among-parsed 与强正 raw mean 并存，"
            "说明数值噪声主要是稀有、极大的 over-count，而不是连续漂移。"
        ),
        "Gemma4-E4B": (
            "支持 log L、log N 与交互项；更长 L、更大 N 都把 robust bias 推向 under-count，"
            "且二者交互会加强这一趋势。少数极端 over-count 仍可把 raw mean 拉正。"
        ),
        "Gemma4-12B": (
            "parsed 数值的 bias 几乎集中在 0，未发现稳定 L/N law。该模型的主要失败机制在"
            " parse/truncation 门槛，而不是成功解析后的连续计数偏差。"
        ),
        "OLMo-Hybrid-7B": (
            "未发现稳定 L/N bias 面；over 与 under 都很多，并有很长的正尾。"
            "更合适的描述是宽而双侧的条件噪声，而非单一方向的平滑 drift。"
        ),
        "Llama3.1-8B": (
            "支持模型内 raw-separable law：L 与 N 增大都把典型 bias 推向 under-count。"
            "raw mean 仍可为正，是因为少数 over-count 的量级远大于常见负误差。"
        ),
        "Llama3.2-3B": (
            "支持最强的负向 L 漂移和显著负向 N 漂移。低负荷时有大量 over-count，"
            "负荷升高后向 under-count 移动，表现为异质状态混合而不是同方差噪声。"
        ),
    }

    cards: list[str] = ['<div class="bias-model-grid">']
    for model in MODELS:
        summary = overall_index.loc[model]
        law = selected_index.loc[model]
        supported = str(law["evidence"]) == "supported"
        status_class = "supported" if supported else "not-supported"
        status_text = "支持模型内 law" if supported else "未发现稳定 N/L law"
        if supported and str(law["selected_candidate"]) == "raw_separable":
            law_line = (
                "选中斜率："
                f"β<sub>L</sub>={number(float(law['length_slope']), 3, True)}，"
                f"β<sub>N</sub>={number(float(law['needle_slope']), 3, True)} "
                "（raw-normalized coordinates）。"
            )
        elif supported and str(law["selected_candidate"]) == "log_interaction":
            law_line = (
                "选中斜率："
                f"β<sub>L</sub>={number(float(law['length_slope']), 3, True)}，"
                f"β<sub>N</sub>={number(float(law['needle_slope']), 3, True)}，"
                f"β<sub>LN</sub>={number(float(law['interaction_slope']), 3, True)}。"
            )
        else:
            law_line = (
                "候选 L/N 坐标未在 grouped held-out 数据中稳定优于 condition-only。"
            )
        cards.append(
            f"""
<article class="bias-model-card">
  <div class="bias-card-head">
    <h4>{html.escape(model)}</h4>
    <span class="bias-status {status_class}">{status_text}</span>
  </div>
  <dl class="bias-metrics">
    <div><dt>parsed</dt><dd>{int(summary['parsed_requests'])}/{int(summary['requests'])} ({percentage(float(summary['parse_rate']))})</dd></div>
    <div><dt>raw mean bias</dt><dd>{number(float(summary['mean_bias']), 2, True)}</dd></div>
    <div><dt>10% trimmed</dt><dd>{number(float(summary['trimmed_mean_bias_10pct']), 2, True)}</dd></div>
    <div><dt>over / under</dt><dd>{percentage(float(summary['over_rate_parsed']))} / {percentage(float(summary['under_rate_parsed']))}</dd></div>
    <div><dt>top-1% |bias| share</dt><dd>{percentage(float(summary['top1pct_absolute_error_share']))}</dd></div>
    <div><dt>nested gain</dt><dd>{float(law['nested_seed_gain_pct']):+.1f}%</dd></div>
  </dl>
  <p>{law_line}</p>
  <p>{interpretations[model]}</p>
</article>"""
        )
    cards.append("</div>")
    return "\n".join(cards)


def tail_math_card() -> str:
    return r"""
<div class="formula equation-card math-equation">
  <div class="equation-title">Bias-tail diagnostics（仅在 parsed numeric outputs 上定义）</div>
  <div class="math-scroll">
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="\mathrm{TailShare}_{1\%}=\frac{\sum_{i\in\mathrm{largest}\ 1\%\ |b|}|b_i|}{\sum_i|b_i|}">
      <semantics>
        <mrow>
          <msub><mi mathvariant="normal">TailShare</mi><mrow><mn>1</mn><mo>%</mo></mrow></msub>
          <mo>=</mo>
          <mfrac>
            <mrow><munder><mo>∑</mo><mrow><mi>i</mi><mo>∈</mo><mtext>largest 1% by |b|</mtext></mrow></munder><mo>|</mo><msub><mi>b</mi><mi>i</mi></msub><mo>|</mo></mrow>
            <mrow><munder><mo>∑</mo><mi>i</mi></munder><mo>|</mo><msub><mi>b</mi><mi>i</mi></msub><mo>|</mo></mrow>
          </mfrac>
        </mrow>
        <annotation encoding="application/x-tex">\mathrm{TailShare}_{1\%}=\frac{\sum_{i\in\mathrm{largest}\ 1\%\ |b|}|b_i|}{\sum_i|b_i|}</annotation>
      </semantics>
    </math>
  </div>
  <div class="math-scroll">
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="\mathrm{TailShift}=\overline b-\operatorname{trimmean}_{10\%}(b)">
      <semantics>
        <mrow>
          <mi mathvariant="normal">TailShift</mi>
          <mo>=</mo>
          <mover><mi>b</mi><mo>¯</mo></mover>
          <mo>−</mo>
          <msub><mi mathvariant="normal">trimmean</mi><mrow><mn>10</mn><mo>%</mo></mrow></msub>
          <mrow><mo>(</mo><mi>b</mi><mo>)</mo></mrow>
        </mrow>
        <annotation encoding="application/x-tex">\mathrm{TailShift}=\overline b-\operatorname{trimmean}_{10\%}(b)</annotation>
      </semantics>
    </math>
  </div>
  <div class="equation-note">TailShare 衡量绝对偏差总量是否被极少数输出支配；TailShift 衡量 raw mean 被双侧各裁掉 10% 后移动了多少。两者都保留极端错误，不用于删点或改写 primary accuracy。</div>
</div>"""


def qwen_math_card() -> str:
    return r"""
<div class="formula equation-card math-equation">
  <div class="equation-title">Qwen 家族：共享函数形式与共享参数是两个不同假设</div>
  <div class="math-scroll">
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="\mathbb E[\operatorname{arsinh}(b_i)\mid m,q,o,L,N]=\alpha_{m,q,o}+\beta_L\log_2(L/5000)+\beta_N\log_2(N/5)">
      <semantics>
        <mrow>
          <mi mathvariant="double-struck">E</mi><mrow><mo>[</mo><mrow><mi mathvariant="normal">arsinh</mi><mrow><mo>(</mo><msub><mi>b</mi><mi>i</mi></msub><mo>)</mo></mrow><mo>|</mo><mi>m</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>o</mi><mo>,</mo><mi>L</mi><mo>,</mo><mi>N</mi></mrow><mo>]</mo></mrow>
          <mo>=</mo><msub><mi>α</mi><mrow><mi>m</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>o</mi></mrow></msub>
          <mo>+</mo><msub><mi>β</mi><mi>L</mi></msub><msub><mi mathvariant="normal">log</mi><mn>2</mn></msub><mrow><mo>(</mo><mfrac><mi>L</mi><mn>5000</mn></mfrac><mo>)</mo></mrow>
          <mo>+</mo><msub><mi>β</mi><mi>N</mi></msub><msub><mi mathvariant="normal">log</mi><mn>2</mn></msub><mrow><mo>(</mo><mfrac><mi>N</mi><mn>5</mn></mfrac><mo>)</mo></mrow>
        </mrow>
        <annotation encoding="application/x-tex">\mathbb E[\operatorname{arsinh}(b_i)\mid m,q,o,L,N]=\alpha_{m,q,o}+\beta_L\log_2(L/5000)+\beta_N\log_2(N/5)</annotation>
      </semantics>
    </math>
  </div>
  <div class="math-scroll">
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="\mathbb E[\operatorname{arsinh}(b_i)\mid\cdots]=\alpha_{m,q,o}+\beta_{m,L}\log_2(L/5000)+\beta_{m,N}\log_2(N/5)">
      <semantics>
        <mrow>
          <mi mathvariant="double-struck">E</mi><mrow><mo>[</mo><mrow><mi mathvariant="normal">arsinh</mi><mrow><mo>(</mo><msub><mi>b</mi><mi>i</mi></msub><mo>)</mo></mrow><mo>|</mo><mo>…</mo></mrow><mo>]</mo></mrow>
          <mo>=</mo><msub><mi>α</mi><mrow><mi>m</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>o</mi></mrow></msub>
          <mo>+</mo><msub><mi>β</mi><mrow><mi>m</mi><mo>,</mo><mi>L</mi></mrow></msub><msub><mi mathvariant="normal">log</mi><mn>2</mn></msub><mrow><mo>(</mo><mfrac><mi>L</mi><mn>5000</mn></mfrac><mo>)</mo></mrow>
          <mo>+</mo><msub><mi>β</mi><mrow><mi>m</mi><mo>,</mo><mi>N</mi></mrow></msub><msub><mi mathvariant="normal">log</mi><mn>2</mn></msub><mrow><mo>(</mo><mfrac><mi>N</mi><mn>5</mn></mfrac><mo>)</mo></mrow>
        </mrow>
        <annotation encoding="application/x-tex">\mathbb E[\operatorname{arsinh}(b_i)\mid\cdots]=\alpha_{m,q,o}+\beta_{m,L}\log_2(L/5000)+\beta_{m,N}\log_2(N/5)</annotation>
      </semantics>
    </math>
  </div>
  <div class="equation-note">第一式要求 Qwen 三个参数量共享 L/N 斜率；第二式只共享函数族，让每个规模拥有自己的斜率。两者都保留 model × prompt mode × query order 截距，并用相同 grouped held-out splits 比较。</div>
</div>"""


def qwen_conclusion(
    comparison: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    indexed = comparison.set_index("candidate")
    shared_candidates = [
        candidate for candidate in CANDIDATES if candidate.startswith("shared_")
    ]
    specific_candidates = [
        candidate
        for candidate in CANDIDATES
        if candidate.startswith("model_specific")
    ]
    best_shared_name = min(
        shared_candidates,
        key=lambda name: float(
            indexed.loc[name, "mean_grouped_request_asinh_mae"]
        ),
    )
    best_specific_name = min(
        specific_candidates,
        key=lambda name: float(
            indexed.loc[name, "mean_grouped_request_asinh_mae"]
        ),
    )
    best_shared = indexed.loc[best_shared_name]
    best_specific = indexed.loc[best_specific_name]
    baseline = indexed.loc["condition_only"]
    shared_supported = (
        float(best_shared["gain_vs_condition_only_pct"]) >= 2.0
        and int(best_shared["schemes_better_than_condition_only"]) >= 3
    )
    specific_gain_over_shared = 100.0 * (
        float(best_shared["mean_grouped_request_asinh_mae"])
        - float(best_specific["mean_grouped_request_asinh_mae"])
    ) / float(best_shared["mean_grouped_request_asinh_mae"])

    per_model = parameters.loc[
        parameters["scope"].isin(QWEN_MODELS)
    ].copy()
    length_rows = per_model.loc[per_model["parameter"] == "log2_length"]
    needle_rows = per_model.loc[per_model["parameter"] == "log2_needles"]
    positive_length = int((length_rows["ci95_low"] > 0).sum())
    negative_length = int((length_rows["ci95_high"] < 0).sum())
    positive_needles = int((needle_rows["ci95_low"] > 0).sum())
    negative_needles = int((needle_rows["ci95_high"] < 0).sum())

    shared_text = (
        f"最好的共享斜率候选是 <strong>{html.escape(CANDIDATE_LABELS[best_shared_name])}</strong>："
        f"四类 grouped split 平均 asinh-bias MAE 为 {float(best_shared['mean_grouped_request_asinh_mae']):.3f}，"
        f"相对 condition-only（{float(baseline['mean_grouped_request_asinh_mae']):.3f}）改善 "
        f"{float(best_shared['gain_vs_condition_only_pct']):+.1f}%，"
        f"在 {int(best_shared['schemes_better_than_condition_only'])}/4 类切分中胜出。"
    )
    if shared_supported:
        shared_judgment = (
            "按预设的 ≥2% 且至少 3/4 切分胜出规则，它对一个有限的共享趋势有支持；"
            "但这不意味着每个规模的 L 与 N 斜率相同。"
        )
    else:
        shared_judgment = (
            "它没有达到共享斜率 law 的预设支持门槛；因此只能说 Qwen 可以共用候选函数族，"
            "不能把同一组 L/N 参数当成已发现的家族定律。"
        )
    heterogeneity = (
        f"允许模型特异斜率后，最佳候选为 <strong>{html.escape(CANDIDATE_LABELS[best_specific_name])}</strong>，"
        f"相对最佳共享式再改善 {specific_gain_over_shared:+.1f}%。"
    )
    direction = (
        f"固定 log 坐标下，长度斜率有 {positive_length}/3 个模型显著为正、"
        f"{negative_length}/3 显著为负；needle 斜率有 {positive_needles}/3 显著为正、"
        f"{negative_needles}/3 显著为负。尤其 N 的方向在 Qwen3-1.7B 与 Qwen3-32B 之间相反，"
        "所以不能用单一 needle-count 阶数概括三个规模。"
    )
    conclusion = (
        f"<p>{shared_text}{shared_judgment}</p>"
        f"<p>{heterogeneity}{direction}</p>"
        "<div class=\"callout\"><strong>Qwen 结论：</strong>"
        "目前最稳妥的共性是“使用同一类带 condition 截距的 response-surface 进行检验”；"
        "真正可解释的斜率仍应留在各模型内部。Qwen3-1.7B 的模型内规律可复现，"
        "Qwen3-8B 与 Qwen3-32B 则主要表现为条件差异与右尾噪声。"
        "</div>"
    )
    metadata = {
        "best_shared_candidate": best_shared_name,
        "best_shared_gain_vs_condition_only_pct": float(
            best_shared["gain_vs_condition_only_pct"]
        ),
        "best_shared_schemes_better": int(
            best_shared["schemes_better_than_condition_only"]
        ),
        "shared_supported_by_preregistered_rule": bool(shared_supported),
        "best_model_specific_candidate": best_specific_name,
        "model_specific_gain_over_best_shared_pct": float(
            specific_gain_over_shared
        ),
        "significant_positive_length_models": positive_length,
        "significant_negative_length_models": negative_length,
        "significant_positive_needle_models": positive_needles,
        "significant_negative_needle_models": negative_needles,
    }
    return conclusion, metadata


def build_section(
    overall: pd.DataFrame,
    by_mode: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    overall_table = overall_display(overall, selected)
    prompt_table = mode_display(by_mode)
    comparison_table = candidate_display(comparison)
    qwen_text, qwen_metadata = qwen_conclusion(comparison, parameters)

    summary_html = table_html(
        overall_table,
        [
            "model_label",
            "parsed",
            "parse_pct",
            "exact_pct",
            "mean_b",
            "trim_b",
            "median_b",
            "over_pct",
            "under_pct",
            "p95_abs",
            "max_abs",
            "top1_pct",
            "law_status",
        ],
        {
            "model_label": "模型",
            "parsed": "parsed n / total",
            "parse_pct": "parse rate",
            "exact_pct": "exact（all）",
            "mean_b": "mean bias",
            "trim_b": "10% trimmed",
            "median_b": "median",
            "over_pct": "over / parsed",
            "under_pct": "under / parsed",
            "p95_abs": "p95 |bias|",
            "max_abs": "max |bias|",
            "top1_pct": "top-1% share",
            "law_status": "模型内结论",
        },
    )
    mode_html = table_html(
        prompt_table,
        [
            "model_label",
            "prompt_mode_label",
            "n",
            "parsed_pct",
            "exact_pct",
            "mean_b",
            "trim_b",
            "median_b",
            "over_pct",
            "under_pct",
            "p95_abs",
            "top1_pct",
        ],
        {
            "model_label": "模型",
            "prompt_mode_label": "prompt mode",
            "n": "requests",
            "parsed_pct": "parse rate",
            "exact_pct": "exact（all）",
            "mean_b": "mean bias",
            "trim_b": "10% trimmed",
            "median_b": "median",
            "over_pct": "over / parsed",
            "under_pct": "under / parsed",
            "p95_abs": "p95 |bias|",
            "top1_pct": "top-1% share",
        },
    )
    candidate_html = table_html(
        comparison_table,
        [
            "candidate_label",
            "mae",
            "gain",
            "wins",
            "seed_r2",
        ],
        {
            "candidate_label": "Qwen 候选式",
            "mae": "四类 grouped-CV MAE",
            "gain": "相对 condition-only",
            "wins": "切分胜出",
            "seed_r2": "seed-held-out cell R²",
        },
    )

    section = f"""
{SECTION_START}
<div id="model-bias-noise-v1" class="bias-noise-analysis">
  <h3>模型内 bias law 与 noise mechanism：逐模型解释</h3>
  <div class="callout">
    <strong>本节的判断层级。</strong>
    第一优先级是每个模型内部是否存在可复现的 <em>N,L → signed bias</em> 规律；
    不要求不同架构共享斜率。第二优先级才是 Qwen3-1.7B/8B/32B 是否能共享方向、阶数或函数形式。
    所有模型仍完整保留 parse failure、格式失败和 truncation 作为 primary exact-accuracy 的失败；
    下述 bias 统计只条件于成功解析出的数值预测，不能代替总体准确率。
  </div>

  {tail_math_card()}

  <p>
    <strong>读表原则：</strong>raw mean bias 对科学结论是必要的，因为极端 over-count 也是有效错误；
    但它必须与 median、10% trimmed mean 和 TailShare 并列。若 raw mean 很正、trimmed mean 接近零或为负，
    则“平均多算”主要是右尾事件，而不是典型预测的平滑正偏。
  </p>

  <h4>八个模型的模型内结论</h4>
  {per_model_cards(overall, selected)}

  <h4>总体 bias 与长尾汇总</h4>
  <div class="table-wrap">{summary_html}</div>
  <p class="table-note">
    exact（all）以全部请求为分母；其余 bias、over/under、p95 和 tail-share 只以 parsed numeric outputs 为分母。
    “支持模型内 law”沿用前节的 nested seed MAE 与四类 grouped split 规则，不因本节的描述性统计而改变。
  </p>

  <figure class="report-figure bias-wide-figure">
    <img src="assets/fig17_model_bias_noise_tail.png" alt="Per-model raw versus trimmed signed bias and top-one-percent absolute-bias share by prompt mode." loading="lazy">
    <figcaption>
      <strong>图 16｜每个模型的 bias 长尾。</strong>
      左图横轴是 count 单位的 signed bias，采用 symmetric-log 刻度：圆点为 raw mean，菱形为双侧各裁 10% 后的 trimmed mean；
      两点距离越大，均值越被少数极端输出拉动。右图每格是该模型与 prompt mode 下，
      绝对 bias 最大的 1% parsed 输出占全部 |bias| 总量的百分比；颜色越深，数值噪声越集中于稀有尾部。
      当一个条件仅有极少数非零误差时，该占比可以达到 100%，需与下表的 exact rate、p95 和分母一起阅读。
      空白表示该模型没有该 prompt mode。图中未删除任何请求。
    </figcaption>
  </figure>

  <h4>按 prompt mode 分解：Direct、Enumeration 与 Native thinking</h4>
  <div class="table-wrap">{mode_html}</div>
  <p class="table-note">
    该表将 query-first/query-last 合并，只用于识别 prompt mode 的噪声形态；
    正式回归仍为每个 model × prompt mode × query order 保留独立截距，避免把 query order 差异误认成 N/L 斜率。
  </p>

  <h4>Qwen 三个参数量：能共享什么，不能共享什么</h4>
  {qwen_math_card()}
  {qwen_text}
  <div class="table-wrap">{candidate_html}</div>
  <p class="table-note">
    MAE 目标为 parsed 输出的 asinh(signed bias)。四类 held-out 切分分别留出 seed、完整 L 水平、
    完整 N 水平和成组 (L,N) cells；每个候选保留 model × prompt mode × query order 截距。
    共享候选强制三个 Qwen 使用相同斜率，model-specific 候选只共享函数族。
  </p>
  <figure class="report-figure bias-wide-figure">
    <img src="assets/fig18_qwen_bias_commonality.png" alt="Grouped cross-validation comparison of shared and model-specific Qwen bias surfaces and per-size log-coordinate slopes." loading="lazy">
    <figcaption>
      <strong>图 17｜Qwen 家族的共享性检验。</strong>
      左图横轴是相对 condition-only 的四类 grouped-CV asinh-bias MAE 改善率；正值越大越好，
      青色是共享斜率候选，橙色是模型特异斜率候选。右图横轴是在固定 log 坐标下，
      L 或 N 翻倍时 mean asinh(bias) 的变化；点为估计，横线为 stimulus-cluster bootstrap 95% CI，
      正值趋向 over-count、负值趋向 under-count。“Qwen shared”是强制共享斜率的 pooled estimate。
    </figcaption>
  </figure>

  <div class="callout">
    <strong>对 counting mechanism 的含义。</strong>
    “模型内部有规律”与“跨模型统一参数”是不同命题。当前证据更支持：
    使用统一的可解释函数族和同一套 held-out 检验协议，但让每个模型拥有自己的方向与斜率；
    对 condition-only 胜出的模型，则把结论写成“噪声/尾部机制未被当前 N,L 网格平滑解释”，而不是伪造一个阶数。
  </div>
</div>
{SECTION_END}"""
    return section, qwen_metadata


MODEL_BIAS_STYLE = r"""
/* MODEL_BIAS_NOISE_V1_START */
#model-bias-noise-v1 {
  min-width: 0;
  margin-top: 34px;
  padding-top: 10px;
  border-top: 2px solid rgba(33, 96, 103, .16);
}
#model-bias-noise-v1 .bias-model-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin: 18px 0 28px;
}
#model-bias-noise-v1 .bias-model-card {
  min-width: 0;
  padding: 16px 17px 15px;
  border: 1px solid var(--line);
  border-radius: 11px;
  background: linear-gradient(145deg, rgba(255,255,255,.88), rgba(243,247,246,.72));
  box-shadow: 0 3px 12px rgba(26, 42, 50, .045);
}
#model-bias-noise-v1 .bias-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
#model-bias-noise-v1 .bias-card-head h4 {
  margin: 0;
}
#model-bias-noise-v1 .bias-status {
  flex: 0 0 auto;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: .74rem;
  font-weight: 750;
  line-height: 1.2;
}
#model-bias-noise-v1 .bias-status.supported {
  color: #15534f;
  background: #d8ece8;
  border: 1px solid #9cc9c2;
}
#model-bias-noise-v1 .bias-status.not-supported {
  color: #6d4a22;
  background: #f4e7d5;
  border: 1px solid #dfc39f;
}
#model-bias-noise-v1 .bias-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px 13px;
  margin: 0 0 11px;
}
#model-bias-noise-v1 .bias-metrics div {
  min-width: 0;
  padding-bottom: 5px;
  border-bottom: 1px solid rgba(123, 143, 149, .17);
}
#model-bias-noise-v1 .bias-metrics dt {
  color: var(--muted);
  font-size: .74rem;
  font-weight: 700;
}
#model-bias-noise-v1 .bias-metrics dd {
  margin: 2px 0 0;
  font-variant-numeric: tabular-nums;
  font-weight: 720;
}
#model-bias-noise-v1 .bias-model-card p:last-child {
  margin-bottom: 0;
}
#model-bias-noise-v1 .table-wrap {
  display: block;
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
}
#model-bias-noise-v1 .table-wrap table {
  width: max-content;
  min-width: 100%;
}
#model-bias-noise-v1 .bias-noise-table {
  font-variant-numeric: tabular-nums;
}
#model-bias-noise-v1 .bias-wide-figure {
  width: min(100%, 1120px);
  max-width: 1120px;
  margin: 28px auto 36px;
}
#model-bias-noise-v1 .bias-wide-figure img {
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  border: 1px solid var(--line);
  background: #fff;
}
@media (max-width: 800px) {
  #model-bias-noise-v1 .bias-model-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 520px) {
  #model-bias-noise-v1 .bias-card-head {
    display: block;
  }
  #model-bias-noise-v1 .bias-status {
    display: inline-block;
    margin-top: 7px;
  }
  #model-bias-noise-v1 .bias-metrics {
    grid-template-columns: 1fr;
  }
}
/* MODEL_BIAS_NOISE_V1_END */
"""


def inject_style(text: str) -> str:
    if STYLE_START in text and STYLE_END in text:
        pattern = re.compile(
            re.escape(STYLE_START) + r".*?" + re.escape(STYLE_END),
            flags=re.DOTALL,
        )
        return pattern.sub(lambda _: MODEL_BIAS_STYLE.strip(), text, count=1)
    if "</style>" not in text:
        raise ValueError("Report has no closing </style> tag")
    return text.replace("</style>", MODEL_BIAS_STYLE + "\n</style>", 1)


def inject_section(text: str, section: str) -> str:
    if SECTION_START in text and SECTION_END in text:
        pattern = re.compile(
            re.escape(SECTION_START) + r".*?" + re.escape(SECTION_END),
            flags=re.DOTALL,
        )
        return pattern.sub(lambda _: section, text, count=1)
    section_start = text.find('<section id="model-bias-laws">')
    if section_start < 0:
        raise ValueError("Could not find model-bias-laws section")
    section_end = text.find("</section>", section_start)
    if section_end < 0:
        raise ValueError("Could not find closing model-bias-laws section")
    return text[:section_end] + section + "\n" + text[section_end:]


def update_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    addition = """

<!-- MODEL_BIAS_NOISE_V1 -->
## Model-wise bias/noise analysis

The canonical report includes a model-wise signed-bias and tail-noise section,
plus a Qwen shared-versus-model-specific slope comparison. Rebuild it with:

```powershell
python scripts/add_model_bias_noise_section.py --report-root "<report directory>"
```

Bias tables are conditional on parsed numeric outputs; primary exact accuracy
continues to count parse failure, format failure, and truncation as failures.
"""
    if README_MARKER not in text:
        path.write_text(
            text.rstrip() + addition + "\n",
            encoding="utf-8",
            newline="\n",
        )


def update_manifest(
    path: Path,
    report_root: Path,
    source_paths: list[Path],
    artifact_paths: list[Path],
    qwen_metadata: dict[str, Any],
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = {
        str(source.relative_to(report_root)): {
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        }
        for source in source_paths
    }
    artifacts = {
        str(artifact.relative_to(report_root)): {
            "sha256": sha256(artifact),
            "bytes": artifact.stat().st_size,
        }
        for artifact in artifact_paths
    }
    manifest["model_bias_noise_v1"] = {
        "generated_at_utc": utc_now(),
        "primary_estimand": (
            "exact correctness over all requests; parse/format/truncation retained "
            "as failures"
        ),
        "secondary_estimand": (
            "signed bias conditional on a parsed numeric prediction"
        ),
        "bias_definition": "predicted_count - gold_count",
        "robust_target": "asinh(signed_bias)",
        "tail_share_definition": (
            "sum absolute bias among largest ceil(1% * parsed_n) outputs divided "
            "by total absolute bias"
        ),
        "trimmed_mean_definition": (
            "arithmetic mean after removing floor(10% * n) values from each tail"
        ),
        "qwen_validation_schemes": [
            "leave-one-seed-out",
            "leave-one-length-level-out",
            "leave-one-needle-level-out",
            "five-fold blocked (L,N) cells",
        ],
        "qwen_candidate_grid": CANDIDATES,
        "qwen_conclusion": qwen_metadata,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "stimulus_id",
        "sources": sources,
        "artifacts": artifacts,
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_inputs(
    requests: pd.DataFrame,
    selected: pd.DataFrame,
    fixed: pd.DataFrame,
) -> None:
    required_request_columns = {
        "request_id",
        "stimulus_id",
        "seed",
        "model_label",
        "target_passage_tokens",
        "num_needles",
        "prompt_mode",
        "query_order",
        "exact_correct",
        "parse_success",
        "format_failure",
        "truncated",
        "predicted_count",
        "signed_error",
    }
    missing = required_request_columns - set(requests.columns)
    if missing:
        raise ValueError(f"Missing request columns: {sorted(missing)}")
    if len(requests) != 6300:
        raise ValueError(f"Expected 6300 requests, found {len(requests)}")
    if requests["request_id"].nunique() != 6300:
        raise ValueError("request_id is not unique")
    if set(requests["model_label"].unique()) != set(MODELS):
        raise ValueError("Unexpected model set")
    if set(selected["model_label"]) != set(MODELS):
        raise ValueError("Selected-law table does not contain all models")
    if set(fixed["model_label"]) != set(MODELS):
        raise ValueError("Fixed-parameter table does not contain all models")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.report_root.resolve()
    tables = root / "tables"
    assets = root / "assets"
    scripts = root / "scripts"
    report_path = root / "report.html"
    manifest_path = root / "analysis_manifest.json"
    readme_path = root / "README.md"

    source_request = tables / "request_level_report.csv"
    source_selected = tables / "model_specific_bias_selected_laws.csv"
    source_fixed = tables / "model_specific_bias_fixed_log_parameters.csv"
    requests = pd.read_csv(source_request)
    selected = pd.read_csv(source_selected)
    fixed = pd.read_csv(source_fixed)
    validate_inputs(requests, selected, fixed)

    overall, by_mode = make_summary_tables(requests)
    parsed_qwen = requests.loc[
        requests["model_label"].isin(QWEN_MODELS)
        & (requests["parse_success"].astype(int) == 1)
        & requests["signed_error"].notna()
        & requests["predicted_count"].notna()
    ].copy()
    parsed_qwen = add_coordinates(parsed_qwen).reset_index(drop=True)
    comparison, fold_metrics, scheme_metrics, oof_cells = (
        evaluate_qwen_candidates(parsed_qwen)
    )
    shared_parameters = bootstrap_shared_qwen_parameters(parsed_qwen)
    qwen_parameters = qwen_parameter_table(fixed, shared_parameters)

    output_tables = {
        "model_bias_noise_model_summary.csv": overall,
        "model_bias_noise_by_mode.csv": by_mode,
        "qwen_bias_family_candidate_comparison.csv": comparison,
        "qwen_bias_family_fold_metrics.csv": fold_metrics,
        "qwen_bias_family_scheme_metrics.csv": scheme_metrics,
        "qwen_bias_family_oof_cells.csv": oof_cells,
        "qwen_bias_family_parameters.csv": qwen_parameters,
    }
    artifact_paths: list[Path] = []
    for name, frame in output_tables.items():
        output = tables / name
        frame.to_csv(output, index=False, encoding="utf-8")
        artifact_paths.append(output)

    tail_figure = assets / "fig17_model_bias_noise_tail.png"
    qwen_figure = assets / "fig18_qwen_bias_commonality.png"
    save_bias_tail_figure(overall, by_mode, tail_figure)
    save_qwen_figure(comparison, qwen_parameters, qwen_figure)
    artifact_paths.extend([tail_figure, qwen_figure])

    section, qwen_metadata = build_section(
        overall,
        by_mode,
        selected,
        comparison,
        qwen_parameters,
    )
    original_report = report_path.read_text(encoding="utf-8")
    revised_report = inject_style(original_report)
    revised_report = inject_section(revised_report, section)
    tmp_report = report_path.with_suffix(".html.tmp")
    tmp_report.write_text(
        revised_report,
        encoding="utf-8",
        newline="\n",
    )
    tmp_report.replace(report_path)

    update_readme(readme_path)
    script_path = scripts / "add_model_bias_noise_section.py"
    if script_path.exists():
        artifact_paths.append(script_path)
    update_manifest(
        manifest_path,
        root,
        [source_request, source_selected, source_fixed],
        artifact_paths,
        qwen_metadata,
    )
    refresh_checksums(root, root / "SHA256SUMS.tsv")

    result = {
        "status": "PASS",
        "request_rows": int(len(requests)),
        "parsed_qwen_rows": int(len(parsed_qwen)),
        "model_summary_rows": int(len(overall)),
        "mode_summary_rows": int(len(by_mode)),
        "qwen_candidates": int(len(comparison)),
        "qwen_conclusion": qwen_metadata,
        "artifacts": [str(path.relative_to(root)) for path in artifact_paths],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
