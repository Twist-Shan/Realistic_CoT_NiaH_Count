#!/usr/bin/env python3
"""Fit independent empirical signed-bias laws for each Realistic NiaH model.

The script copies an existing verified report package into a new directory,
fits each model in isolation, writes reproducible tables/figures, inserts a
Chinese addendum into the HTML report, updates the manifest, and regenerates
artifact checksums.

Primary accuracy is not changed. Signed bias is conditional on a numeric output
having been parsed, exactly as in the parent report.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
PROMPT_ORDER = ["direct", "enumeration", "native_thinking"]
QUERY_ORDER = ["query_first", "query_last"]
RANDOM_SEED = 20260724
BOOTSTRAP_REPLICATES = 500

CANDIDATES = [
    "condition_only",
    "log_density",
    "log_separable",
    "log_interaction",
    "raw_separable",
]
CANDIDATE_LABELS = {
    "condition_only": "Condition only",
    "log_density": "Log density",
    "log_separable": "Separate log L + log N",
    "log_interaction": "Log L + log N + interaction",
    "raw_separable": "Raw normalized L + N",
}
CANDIDATE_LABELS_CN = {
    "condition_only": "仅模式/query-order 截距",
    "log_density": "log density",
    "log_separable": "分离 log₂L + log₂N",
    "log_interaction": "分离 log₂L + log₂N + 交互",
    "raw_separable": "原始归一化 L + N",
}
CANDIDATE_COMPLEXITY = {
    "condition_only": 0,
    "log_density": 1,
    "log_separable": 2,
    "raw_separable": 2,
    "log_interaction": 3,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_levels(frame: pd.DataFrame) -> list[str]:
    present = set(frame["condition"].astype(str))
    ordered = [
        f"{prompt}|{order}"
        for prompt in PROMPT_ORDER
        for order in QUERY_ORDER
        if f"{prompt}|{order}" in present
    ]
    if set(ordered) != present:
        ordered.extend(sorted(present - set(ordered)))
    return ordered


def add_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["condition"] = (
        out["prompt_mode"].astype(str)
        + "|"
        + out["query_order"].astype(str)
    )
    out["x_log_length"] = np.log2(
        out["target_passage_tokens"].astype(float) / 5000.0
    )
    out["x_log_needles"] = np.log2(
        out["num_needles"].astype(float) / 5.0
    )
    out["x_log_density"] = (
        out["x_log_needles"] - out["x_log_length"]
    )
    out["x_raw_length"] = (
        out["target_passage_tokens"].astype(float) - 5000.0
    ) / 5000.0
    out["x_raw_needles"] = (
        out["num_needles"].astype(float) - 5.0
    ) / 5.0
    out["asinh_bias"] = np.arcsinh(out["signed_error"].astype(float))
    out["capped_bias"] = np.clip(
        out["signed_error"].astype(float), -30.0, 30.0
    )
    return out


def build_design(
    frame: pd.DataFrame,
    candidate: str,
    levels: list[str],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    condition = frame["condition"].astype(str).to_numpy()
    for level in levels:
        columns.append((condition == level).astype(float))
        names.append(f"intercept[{level}]")

    if candidate == "condition_only":
        pass
    elif candidate == "log_density":
        columns.append(frame["x_log_density"].to_numpy(dtype=float))
        names.append("log2_density")
    elif candidate in {"log_separable", "log_interaction"}:
        x_length = frame["x_log_length"].to_numpy(dtype=float)
        x_needles = frame["x_log_needles"].to_numpy(dtype=float)
        columns.extend([x_length, x_needles])
        names.extend(["log2_length", "log2_needles"])
        if candidate == "log_interaction":
            columns.append(x_length * x_needles)
            names.append("log2_length_x_needles")
    elif candidate == "raw_separable":
        columns.extend(
            [
                frame["x_raw_length"].to_numpy(dtype=float),
                frame["x_raw_needles"].to_numpy(dtype=float),
            ]
        )
        names.extend(["raw_length", "raw_needles"])
    else:
        raise KeyError(candidate)

    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite design for {candidate}")
    return matrix, names


def fit_ols(matrix: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(matrix, outcome, rcond=None)
    if not np.isfinite(beta).all():
        raise ValueError("Non-finite OLS coefficients")
    return beta


def outcome_values(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "asinh_bias":
        return frame["asinh_bias"].to_numpy(dtype=float)
    if target == "capped_bias":
        return frame["capped_bias"].to_numpy(dtype=float)
    if target == "raw_bias":
        return frame["signed_error"].to_numpy(dtype=float)
    raise KeyError(target)


def cell_blocks(frame: pd.DataFrame) -> np.ndarray:
    cells = sorted(
        {
            (int(length), int(needles))
            for length, needles in zip(
                frame["target_passage_tokens"],
                frame["num_needles"],
            )
        }
    )
    mapping = {cell: index % 5 for index, cell in enumerate(cells)}
    return np.array(
        [
            mapping[(int(length), int(needles))]
            for length, needles in zip(
                frame["target_passage_tokens"],
                frame["num_needles"],
            )
        ],
        dtype=int,
    )


def fold_labels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "seed": frame["seed"].to_numpy(),
        "needle_level": frame["num_needles"].to_numpy(),
        "length_level": frame["target_passage_tokens"].to_numpy(),
        "cell_block": cell_blocks(frame),
    }


def cross_validated_predictions(
    frame: pd.DataFrame,
    candidate: str,
    target: str,
    scheme_labels: np.ndarray,
    levels: list[str],
) -> np.ndarray:
    outcome = outcome_values(frame, target)
    predictions = np.full(len(frame), np.nan, dtype=float)
    for label in pd.unique(scheme_labels):
        test = scheme_labels == label
        train = ~test
        train_matrix, _ = build_design(frame.loc[train], candidate, levels)
        test_matrix, _ = build_design(frame.loc[test], candidate, levels)
        beta = fit_ols(train_matrix, outcome[train])
        predictions[test] = test_matrix @ beta
    if not np.isfinite(predictions).all():
        raise RuntimeError(
            f"Incomplete OOF prediction for {candidate}, {target}"
        )
    return predictions


def target_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    target: str,
) -> dict[str, float]:
    residual = observed - predicted
    result = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_residual": float(np.mean(residual)),
    }
    if target == "asinh_bias":
        raw_observed = np.sinh(np.clip(observed, -10.0, 10.0))
        raw_predicted = np.sinh(np.clip(predicted, -10.0, 10.0))
        raw_residual = raw_observed - raw_predicted
        result["raw_scale_mae"] = float(
            np.mean(np.abs(raw_residual))
        )
        result["raw_scale_rmse"] = float(
            np.sqrt(np.mean(raw_residual**2))
        )
    return result


def evaluate_candidates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    seed_predictions: dict[tuple[str, str], np.ndarray] = {}
    levels = condition_levels(frame)
    schemes = fold_labels(frame)
    for target in ["asinh_bias", "capped_bias", "raw_bias"]:
        observed = outcome_values(frame, target)
        selection_metric = "rmse" if target == "raw_bias" else "mae"
        for candidate in CANDIDATES:
            row: dict[str, Any] = {
                "target": target,
                "candidate": candidate,
                "candidate_label": CANDIDATE_LABELS[candidate],
                "selection_metric": selection_metric,
                "n_parameters": len(
                    build_design(frame, candidate, levels)[1]
                ),
            }
            scheme_scores = []
            for scheme, labels in schemes.items():
                prediction = cross_validated_predictions(
                    frame,
                    candidate,
                    target,
                    labels,
                    levels,
                )
                metrics = target_metrics(observed, prediction, target)
                for key, value in metrics.items():
                    row[f"{scheme}_{key}"] = value
                scheme_scores.append(metrics[selection_metric])
                if scheme == "seed" and target == "asinh_bias":
                    seed_predictions[(target, candidate)] = prediction
            row["selection_score"] = float(np.mean(scheme_scores))
            rows.append(row)
    return pd.DataFrame(rows), seed_predictions


def choose_candidate(
    comparison: pd.DataFrame,
    target: str,
) -> pd.Series:
    subset = comparison.loc[comparison["target"] == target].copy()
    minimum = float(subset["selection_score"].min())
    tolerance = max(1e-6, minimum * 0.002)
    eligible = subset.loc[
        subset["selection_score"] <= minimum + tolerance
    ].copy()
    eligible["complexity"] = eligible["candidate"].map(
        CANDIDATE_COMPLEXITY
    )
    return eligible.sort_values(
        ["complexity", "selection_score", "candidate"]
    ).iloc[0]


def nested_seed_validation(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    levels = condition_levels(frame)
    outcome = frame["asinh_bias"].to_numpy(dtype=float)
    outer_seed = frame["seed"].to_numpy()
    predictions = np.full(len(frame), np.nan, dtype=float)
    rows = []
    for held_seed in sorted(pd.unique(outer_seed)):
        outer_test = outer_seed == held_seed
        outer_train = ~outer_test
        training = frame.loc[outer_train].copy()
        training_levels = levels
        inner_seed = training["seed"].to_numpy()
        inner_observed = training["asinh_bias"].to_numpy(dtype=float)
        candidate_scores: list[tuple[float, int, str]] = []
        for candidate in CANDIDATES:
            inner_prediction = cross_validated_predictions(
                training,
                candidate,
                "asinh_bias",
                inner_seed,
                training_levels,
            )
            mae = float(
                np.mean(np.abs(inner_observed - inner_prediction))
            )
            candidate_scores.append(
                (mae, CANDIDATE_COMPLEXITY[candidate], candidate)
            )
        candidate_scores.sort()
        best_inner = candidate_scores[0][2]
        train_matrix, _ = build_design(
            frame.loc[outer_train], best_inner, levels
        )
        test_matrix, _ = build_design(
            frame.loc[outer_test], best_inner, levels
        )
        beta = fit_ols(train_matrix, outcome[outer_train])
        predictions[outer_test] = test_matrix @ beta
        fold_metrics = target_metrics(
            outcome[outer_test],
            predictions[outer_test],
            "asinh_bias",
        )
        rows.append(
            {
                "held_out_seed": int(held_seed),
                "selected_candidate": best_inner,
                **fold_metrics,
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete nested seed OOF predictions")
    return pd.DataFrame(rows), predictions


def bootstrap_fit(
    frame: pd.DataFrame,
    candidate: str,
    levels: list[str],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    matrix, names = build_design(frame, candidate, levels)
    outcome = frame["asinh_bias"].to_numpy(dtype=float)
    beta = fit_ols(matrix, outcome)
    cluster_values = sorted(frame["stimulus_id"].astype(str).unique())
    cluster_indices = {
        cluster: np.flatnonzero(
            frame["stimulus_id"].astype(str).to_numpy() == cluster
        )
        for cluster in cluster_values
    }
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(
            cluster_values, size=len(cluster_values), replace=True
        )
        indices = np.concatenate(
            [cluster_indices[str(cluster)] for cluster in sampled]
        )
        draw_beta = fit_ols(matrix[indices], outcome[indices])
        draws.append(draw_beta)
    bootstrap = np.asarray(draws, dtype=float)
    return beta, names, bootstrap


def reference_condition(frame: pd.DataFrame, levels: list[str]) -> str:
    counts = frame["condition"].value_counts()
    return sorted(
        levels,
        key=lambda level: (
            -int(counts.get(level, 0)),
            levels.index(level),
        ),
    )[0]


def coefficient_value(
    beta: np.ndarray,
    names: list[str],
    bootstrap: np.ndarray,
    name: str,
) -> tuple[float, float, float]:
    if name not in names:
        return float("nan"), float("nan"), float("nan")
    index = names.index(name)
    return (
        float(beta[index]),
        float(np.percentile(bootstrap[:, index], 2.5)),
        float(np.percentile(bootstrap[:, index], 97.5)),
    )


def equation_text(candidate: str) -> str:
    if candidate == "condition_only":
        return "asinh(b)=α(q,o)"
    if candidate == "log_density":
        return "asinh(b)=α(q,o)+βD·log₂[(N/T)/(5/5000)]"
    if candidate == "log_separable":
        return (
            "asinh(b)=α(q,o)+βL·log₂(T/5000)"
            "+βN·log₂(N/5)"
        )
    if candidate == "log_interaction":
        return (
            "asinh(b)=α(q,o)+βL·log₂(T/5000)"
            "+βN·log₂(N/5)+βLN·log₂(T/5000)log₂(N/5)"
        )
    if candidate == "raw_separable":
        return (
            "asinh(b)=α(q,o)+βL·(T−5000)/5000"
            "+βN·(N−5)/5"
        )
    raise KeyError(candidate)


def parameter_rows(
    frame: pd.DataFrame,
    candidate: str,
    beta: np.ndarray,
    names: list[str],
    bootstrap: np.ndarray,
) -> dict[str, Any]:
    levels = condition_levels(frame)
    reference = reference_condition(frame, levels)
    result: dict[str, Any] = {
        "reference_condition": reference,
        "reference_condition_parsed_n": int(
            (frame["condition"] == reference).sum()
        ),
        "equation": equation_text(candidate),
    }
    parameter_names = {
        "reference_intercept": f"intercept[{reference}]",
        "length_slope": (
            "log2_length"
            if candidate in {"log_separable", "log_interaction"}
            else "raw_length"
            if candidate == "raw_separable"
            else ""
        ),
        "needle_slope": (
            "log2_needles"
            if candidate in {"log_separable", "log_interaction"}
            else "raw_needles"
            if candidate == "raw_separable"
            else ""
        ),
        "density_slope": (
            "log2_density" if candidate == "log_density" else ""
        ),
        "interaction_slope": (
            "log2_length_x_needles"
            if candidate == "log_interaction"
            else ""
        ),
    }
    for output_name, coefficient_name in parameter_names.items():
        if not coefficient_name:
            result[output_name] = float("nan")
            result[f"{output_name}_ci95_low"] = float("nan")
            result[f"{output_name}_ci95_high"] = float("nan")
            continue
        value, low, high = coefficient_value(
            beta, names, bootstrap, coefficient_name
        )
        result[output_name] = value
        result[f"{output_name}_ci95_low"] = low
        result[f"{output_name}_ci95_high"] = high
    return result


def evidence_label(
    selected_candidate: str,
    nested_gain_pct: float,
    scheme_wins: int,
) -> tuple[str, str]:
    if (
        selected_candidate != "condition_only"
        and nested_gain_pct >= 2.0
        and scheme_wins >= 3
    ):
        return "supported", "有支持"
    if (
        selected_candidate != "condition_only"
        and nested_gain_pct > 0.0
        and scheme_wins >= 2
    ):
        return "weak", "弱支持"
    return "not_supported", "未发现稳定 law"


def analyze_models(
    parsed: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, Any]],
]:
    comparison_parts = []
    selected_rows = []
    fixed_rows = []
    nested_parts = []
    calibration_parts = []
    model_fits: dict[str, dict[str, Any]] = {}

    for model_index, model in enumerate(MODELS):
        frame = parsed.loc[parsed["model_label"] == model].copy()
        frame = frame.reset_index(drop=True)
        levels = condition_levels(frame)
        comparison, seed_predictions = evaluate_candidates(frame)
        comparison.insert(0, "model_label", model)
        comparison_parts.append(comparison)

        asinh_subset = comparison.loc[
            comparison["target"] == "asinh_bias"
        ].copy()
        selected = choose_candidate(comparison, "asinh_bias")
        selected_name = str(selected["candidate"])
        baseline = asinh_subset.loc[
            asinh_subset["candidate"] == "condition_only"
        ].iloc[0]
        scheme_wins = sum(
            float(selected[f"{scheme}_mae"])
            < float(baseline[f"{scheme}_mae"])
            for scheme in [
                "seed",
                "needle_level",
                "length_level",
                "cell_block",
            ]
        )

        nested_folds, nested_prediction = nested_seed_validation(frame)
        nested_folds.insert(0, "model_label", model)
        nested_parts.append(nested_folds)
        nested_mae = float(
            np.mean(
                np.abs(
                    frame["asinh_bias"].to_numpy(dtype=float)
                    - nested_prediction
                )
            )
        )
        baseline_seed_prediction = seed_predictions[
            ("asinh_bias", "condition_only")
        ]
        baseline_nested_mae = float(
            np.mean(
                np.abs(
                    frame["asinh_bias"].to_numpy(dtype=float)
                    - baseline_seed_prediction
                )
            )
        )
        nested_gain_pct = 100.0 * (
            baseline_nested_mae - nested_mae
        ) / baseline_nested_mae
        evidence, evidence_cn = evidence_label(
            selected_name, nested_gain_pct, scheme_wins
        )

        rng = np.random.default_rng(RANDOM_SEED + model_index)
        selected_beta, selected_names, selected_bootstrap = bootstrap_fit(
            frame,
            selected_name,
            levels,
            BOOTSTRAP_REPLICATES,
            rng,
        )
        selected_parameters = parameter_rows(
            frame,
            selected_name,
            selected_beta,
            selected_names,
            selected_bootstrap,
        )

        fixed_rng = np.random.default_rng(
            RANDOM_SEED + 100 + model_index
        )
        fixed_beta, fixed_names, fixed_bootstrap = bootstrap_fit(
            frame,
            "log_separable",
            levels,
            BOOTSTRAP_REPLICATES,
            fixed_rng,
        )
        fixed_parameters = parameter_rows(
            frame,
            "log_separable",
            fixed_beta,
            fixed_names,
            fixed_bootstrap,
        )

        raw_subset = comparison.loc[
            comparison["target"] == "raw_bias"
        ].copy()
        raw_selected = choose_candidate(comparison, "raw_bias")
        raw_baseline = raw_subset.loc[
            raw_subset["candidate"] == "condition_only"
        ].iloc[0]
        raw_gain_pct = 100.0 * (
            float(raw_baseline["selection_score"])
            - float(raw_selected["selection_score"])
        ) / float(raw_baseline["selection_score"])

        selected_rows.append(
            {
                "model_label": model,
                "parsed_requests": len(frame),
                "selected_candidate": selected_name,
                "selected_candidate_cn": CANDIDATE_LABELS_CN[
                    selected_name
                ],
                "selection_score_asinh_mae": float(
                    selected["selection_score"]
                ),
                "condition_only_score_asinh_mae": float(
                    baseline["selection_score"]
                ),
                "four_scheme_gain_pct": 100.0
                * (
                    float(baseline["selection_score"])
                    - float(selected["selection_score"])
                )
                / float(baseline["selection_score"]),
                "schemes_better_than_condition_only": scheme_wins,
                "nested_seed_mae": nested_mae,
                "nested_seed_condition_only_mae": baseline_nested_mae,
                "nested_seed_gain_pct": nested_gain_pct,
                "nested_selected_candidate_counts": json.dumps(
                    Counter(nested_folds["selected_candidate"]),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "evidence": evidence,
                "evidence_cn": evidence_cn,
                "raw_mean_selected_candidate": str(
                    raw_selected["candidate"]
                ),
                "raw_mean_selected_rmse": float(
                    raw_selected["selection_score"]
                ),
                "raw_mean_condition_only_rmse": float(
                    raw_baseline["selection_score"]
                ),
                "raw_mean_gain_pct": raw_gain_pct,
                **selected_parameters,
            }
        )
        fixed_rows.append(
            {
                "model_label": model,
                "parsed_requests": len(frame),
                **fixed_parameters,
            }
        )

        selected_seed_prediction = seed_predictions[
            ("asinh_bias", selected_name)
        ]
        calibration = frame[
            [
                "target_passage_tokens",
                "num_needles",
                "prompt_mode",
                "query_order",
            ]
        ].copy()
        calibration["observed_asinh_bias"] = frame[
            "asinh_bias"
        ].to_numpy(dtype=float)
        calibration["oof_predicted_asinh_bias"] = (
            selected_seed_prediction
        )
        calibration["model_label"] = model
        calibration_parts.append(calibration)

        model_fits[model] = {
            "frame": frame,
            "levels": levels,
            "selected_candidate": selected_name,
            "selected_beta": selected_beta,
            "selected_names": selected_names,
            "selected_parameters": selected_parameters,
            "fixed_beta": fixed_beta,
            "fixed_names": fixed_names,
        }

    comparison_all = pd.concat(comparison_parts, ignore_index=True)
    selected_all = pd.DataFrame(selected_rows)
    fixed_all = pd.DataFrame(fixed_rows)
    nested_all = pd.concat(nested_parts, ignore_index=True)
    calibration_all = pd.concat(calibration_parts, ignore_index=True)
    return (
        comparison_all,
        selected_all,
        fixed_all,
        nested_all,
        calibration_all,
        model_fits,
    )


def make_candidate_figure(
    comparison: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
) -> None:
    candidates = [
        candidate
        for candidate in CANDIDATES
        if candidate != "condition_only"
    ]
    matrix = np.zeros((len(MODELS), len(candidates)), dtype=float)
    for row_index, model in enumerate(MODELS):
        subset = comparison.loc[
            (comparison["model_label"] == model)
            & (comparison["target"] == "asinh_bias")
        ].set_index("candidate")
        baseline = float(
            subset.loc["condition_only", "selection_score"]
        )
        for column_index, candidate in enumerate(candidates):
            score = float(subset.loc[candidate, "selection_score"])
            matrix[row_index, column_index] = (
                100.0 * (baseline - score) / baseline
            )

    limit = max(5.0, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    image = ax.imshow(
        matrix,
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(
        [CANDIDATE_LABELS[candidate] for candidate in candidates],
        rotation=18,
        ha="right",
    )
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for row_index, model in enumerate(MODELS):
        chosen = str(
            selected.loc[
                selected["model_label"] == model,
                "selected_candidate",
            ].iloc[0]
        )
        for column_index, candidate in enumerate(candidates):
            value = matrix[row_index, column_index]
            color = "white" if abs(value) > limit * 0.52 else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
            if candidate == chosen:
                ax.add_patch(
                    Rectangle(
                        (column_index - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.2,
                    )
                )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        "Four-scheme asinh-bias MAE improvement vs condition-only"
    )
    ax.set_title(
        "Independent per-model coordinate search for typical signed bias"
    )
    ax.set_xlabel("Candidate coordinate law")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_slope_figure(fixed: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0), sharey=True)
    y = np.arange(len(MODELS))
    for ax, prefix, title in [
        (
            axes[0],
            "length_slope",
            "Length effect per doubling",
        ),
        (
            axes[1],
            "needle_slope",
            "Needle-count effect per doubling",
        ),
    ]:
        values = fixed.set_index("model_label").loc[MODELS, prefix]
        low = (
            fixed.set_index("model_label")
            .loc[MODELS, f"{prefix}_ci95_low"]
            .to_numpy(dtype=float)
        )
        high = (
            fixed.set_index("model_label")
            .loc[MODELS, f"{prefix}_ci95_high"]
            .to_numpy(dtype=float)
        )
        center = values.to_numpy(dtype=float)
        ax.errorbar(
            center,
            y,
            xerr=np.vstack([center - low, high - center]),
            fmt="o",
            capsize=3,
            color="#2a6f97",
        )
        ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Change in mean asinh(bias)")
        ax.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(MODELS)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Separate log-length and log-needle slopes fitted within each model"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_calibration_figure(
    calibration: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    grouped = (
        calibration.groupby(
            [
                "model_label",
                "target_passage_tokens",
                "num_needles",
            ],
            as_index=False,
        )
        .agg(
            observed_mean_asinh_bias=(
                "observed_asinh_bias",
                "mean",
            ),
            oof_predicted_mean_asinh_bias=(
                "oof_predicted_asinh_bias",
                "mean",
            ),
            parsed_n=("observed_asinh_bias", "size"),
        )
    )
    fig, axes = plt.subplots(
        2, 4, figsize=(15.0, 7.2), sharex=False, sharey=False
    )
    for ax, model in zip(axes.ravel(), MODELS):
        part = grouped.loc[grouped["model_label"] == model]
        x = part["oof_predicted_mean_asinh_bias"].to_numpy(dtype=float)
        y = part["observed_mean_asinh_bias"].to_numpy(dtype=float)
        ax.scatter(x, y, s=25, alpha=0.75, color="#3a86a8")
        finite = np.concatenate([x, y])
        low = float(np.min(finite))
        high = float(np.max(finite))
        padding = max(0.15, (high - low) * 0.08)
        low -= padding
        high += padding
        ax.plot([low, high], [low, high], "--", color="#777777")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(model)
        ax.grid(alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed cell mean asinh(bias)")
    for ax in axes[1, :]:
        ax.set_xlabel("Seed-OOF predicted mean asinh(bias)")
    fig.suptitle(
        "Held-out calibration of each model's selected signed-bias law"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return grouped


def make_surface_figure(
    model_fits: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    grid_needles = np.array(
        [1, 2, 3, 4, 5, 6, 8, 10, 20, 30], dtype=int
    )
    lengths = [2000, 5000, 10000]
    colors = ["#2878b5", "#f28e2b", "#2a9d55"]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), sharex=True)
    for ax, model in zip(axes.ravel(), MODELS):
        fitted = model_fits[model]
        frame = fitted["frame"]
        reference = fitted["selected_parameters"][
            "reference_condition"
        ]
        prompt_mode, query_order = reference.split("|", 1)
        candidate = fitted["selected_candidate"]
        for length, color in zip(lengths, colors):
            grid = pd.DataFrame(
                {
                    "prompt_mode": prompt_mode,
                    "query_order": query_order,
                    "target_passage_tokens": length,
                    "num_needles": grid_needles,
                }
            )
            grid["signed_error"] = 0.0
            grid["stimulus_id"] = "grid"
            grid["seed"] = 0
            grid = add_coordinates(grid)
            matrix, _ = build_design(
                grid, candidate, fitted["levels"]
            )
            predicted_asinh = matrix @ fitted["selected_beta"]
            centered_bias = np.sinh(np.clip(predicted_asinh, -8.0, 8.0))
            ax.plot(
                grid_needles,
                centered_bias,
                marker="o",
                markersize=3,
                color=color,
                label=f"T={length:,}",
            )
        ax.axhline(0.0, color="#777777", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(
            f"{model}\n{CANDIDATE_LABELS[candidate]}",
            fontsize=10,
        )
        ax.grid(alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted asinh-centered bias")
    for ax in axes[1, :]:
        ax.set_xlabel("Needle count N (log₂ scale)")
    axes[0, 0].legend(title="Passage length", fontsize=8)
    fig.suptitle(
        "Selected per-model laws at each model's reference prompt/order condition"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_percent(value: Any, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}%"


def format_float(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def interval_text(row: pd.Series, prefix: str) -> str:
    value = row.get(prefix)
    low = row.get(f"{prefix}_ci95_low")
    high = row.get(f"{prefix}_ci95_high")
    if pd.isna(value):
        return "—"
    return (
        f"{float(value):+.3f} "
        f"[{float(low):+.3f}, {float(high):+.3f}]"
    )


def selection_table_html(selected: pd.DataFrame) -> str:
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td>{int(row['parsed_requests']):,}</td>"
            f"<td>{html.escape(str(row['selected_candidate_cn']))}</td>"
            f"<td>{html.escape(str(row['evidence_cn']))}</td>"
            f"<td>{format_float(row['nested_seed_mae'])}</td>"
            f"<td>{format_percent(row['nested_seed_gain_pct'])}</td>"
            f"<td>{int(row['schemes_better_than_condition_only'])}/4</td>"
            f"<td>{format_percent(row['raw_mean_gain_pct'])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="data-table">'
        "<thead><tr>"
        "<th>模型</th><th>parsed n</th><th>选中的稳健坐标</th>"
        "<th>证据判定</th><th>nested seed MAE</th>"
        "<th>相对 condition-only 改善</th>"
        "<th>跨切分胜出</th><th>raw-mean RMSE 最佳改善</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def parameter_table_html(fixed: pd.DataFrame) -> str:
    rows = []
    for _, row in fixed.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td>{html.escape(str(row['reference_condition']))}</td>"
            f"<td>{interval_text(row, 'reference_intercept')}</td>"
            f"<td>{interval_text(row, 'length_slope')}</td>"
            f"<td>{interval_text(row, 'needle_slope')}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="data-table">'
        "<thead><tr>"
        "<th>模型</th><th>参考 condition</th>"
        "<th>α：参考点 asinh bias [95% CI]</th>"
        "<th>βL：T 翻倍变化 [95% CI]</th>"
        "<th>βN：N 翻倍变化 [95% CI]</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def selected_equation_table_html(selected: pd.DataFrame) -> str:
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td>{html.escape(str(row['reference_condition']))}</td>"
            f"<td><code>{html.escape(str(row['equation']))}</code></td>"
            f"<td>{interval_text(row, 'reference_intercept')}</td>"
            f"<td>{interval_text(row, 'length_slope')}</td>"
            f"<td>{interval_text(row, 'needle_slope')}</td>"
            f"<td>{interval_text(row, 'density_slope')}</td>"
            f"<td>{interval_text(row, 'interaction_slope')}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="data-table">'
        "<thead><tr>"
        "<th>模型</th><th>参考 condition</th><th>选中公式</th>"
        "<th>α [95% CI]</th><th>βL [95% CI]</th>"
        "<th>βN [95% CI]</th><th>βD [95% CI]</th>"
        "<th>βLN [95% CI]</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build_addendum_html(
    selected: pd.DataFrame,
    fixed: pd.DataFrame,
) -> str:
    supported = selected.loc[
        selected["evidence"] == "supported", "model_label"
    ].tolist()
    weak = selected.loc[
        selected["evidence"] == "weak", "model_label"
    ].tolist()
    unsupported = selected.loc[
        selected["evidence"] == "not_supported", "model_label"
    ].tolist()

    def names(values: list[str]) -> str:
        return "、".join(values) if values else "无"

    return f"""
<section id="model-bias-laws">
  <h2>每个模型各自的 bias empirical law</h2>
  <p>这一节把八个模型完全拆开拟合；任何模型都不借用其他模型的斜率或截距。样本仍限于成功解析出数值的 5,385 条输出，未解析、格式失败和截断继续在 primary accuracy 中计为失败，并且不会被伪造为 bias=0。</p>

  <h3>目标与可解释的固定形式</h3>
  <div class="formula"><strong>bias</strong> = predicted_count − true_count；<strong>asinh-centered bias</strong> = sinh(E[asinh(bias)])。asinh 在 0 附近近似线性、对大正负值近似对数，因此保留方向，同时降低少数数百至数千的 over-count 对回归的支配。它不是 raw mean，也不等同于 median。</div>
  <div class="formula"><strong>可比较固定形式：</strong>asinh(bias) = α<sub>m,q,o</sub> + β<sub>m,L</sub> log₂(T/5000) + β<sub>m,N</sub> log₂(N/5)。因此 β<sub>m,L</sub> 是长度翻倍时 mean asinh(bias) 的变化，β<sub>m,N</sub> 是 needle 数翻倍时的变化；α 随模型、prompt mode 与 query order 改变。</div>
  <p>每个模型还独立比较 condition-only、density-only、分离 log L/log N、带交互 log surface、以及原始归一化 L/N。最终坐标使用留一 seed、留一 N、留一 T 与 blocked cell 四套验证；另做外层留一 seed、内层重新选坐标的 nested 验证，避免只报告挑选后的训练拟合。</p>

  <h3>是否真的存在可复现的模型内 law</h3>
  {selection_table_html(selected)}
  <p class="table-note">“有支持”要求：nested seed MAE 相对 condition-only 至少改善 2%，且在四种完整切分中至少 3 种胜出；“弱支持”表示改善为正且至少 2/4 切分胜出。raw-mean 列使用原始 count-unit bias 的 OOF RMSE，专门检查极端 outlier 下的字面平均偏差能否预测。</p>
  <figure class="report-figure"><img src="assets/fig13_model_bias_candidate_search.png" alt="Heatmap of per-model cross-validated MAE improvements for four signed-bias coordinate laws." loading="lazy"><figcaption><strong>图 12｜各模型独立坐标搜索。</strong> 行为模型，列为候选坐标；格内数字是相对仅含 prompt/order 截距的四切分平均 asinh-bias MAE 改善百分比，正值越大越好、负值表示更差。黑框标出最终候选；未框出说明 condition-only 本身最优。</figcaption></figure>

  <p><strong>证据汇总：</strong>有支持：{names(supported)}；弱支持：{names(weak)}；未发现稳定 law：{names(unsupported)}。这里的“未发现”不是 bias 恒定，而是当前 3 个长度水平、10 个 N 水平和 5 个 seed 不足以让候选 response surface 在 held-out 数据上稳定优于 condition-only。</p>

  <h3>统一函数族下的模型特异阶数</h3>
  <p>为了直接回答“长度和 needle 数分别是什么阶”，无论模型最终选中哪个坐标，都额外强制拟合同一个分离 log 形式。下表与图中的系数完全由该模型自己的 parsed outputs 得到；区间按完整 stimulus 聚类 bootstrap 500 次。</p>
  <p class="table-note">这里的 bootstrap 区间是在“固定采用分离 log 形式”的前提下计算，未对候选公式选择作事后校正。因此，即使某个 β 的区间不跨 0，只要该模型在 grouped held-out 验证中仍由 condition-only 胜出，就只能解释为条件相关性，不能称为具有稳定预测力的 empirical law。</p>
  {parameter_table_html(fixed)}
  <figure class="report-figure"><img src="assets/fig14_model_bias_log_slopes.png" alt="Model-specific log-length and log-needle coefficients for asinh signed bias with clustered bootstrap intervals." loading="lazy"><figcaption><strong>图 13｜每模型单独拟合的长度与 needle bias 斜率。</strong> 左图横轴为 T 翻倍时 mean asinh(bias) 的变化，右图为 N 翻倍时的变化；横线是 stimulus-cluster bootstrap 95% CI，虚线 0 表示没有可辨识方向。系数为正表示该维度增大时更偏向 over-count，负值表示更偏向 under-count。</figcaption></figure>

  <h3>每个模型最终选中的方程</h3>
  {selected_equation_table_html(selected)}
  <p class="table-note">α 使用该模型 parsed 样本最多的 prompt/order condition 作为参考；其他 condition 有各自截距但未在宽表重复。βD 只用于 density law，βLN 只用于交互 law。所有公式只在 2k≤T≤10k、1≤N≤30 的注册范围内解释。</p>
  <figure class="report-figure"><img src="assets/fig15_model_bias_selected_surfaces.png" alt="Selected per-model signed-bias response curves versus needle count at three passage lengths." loading="lazy"><figcaption><strong>图 14｜各模型选中 law 的 response-surface 切片。</strong> 横轴是 N（log₂ 刻度），纵轴是预测 asinh-centered bias，采用对称 log 刻度以同时显示正负与长尾；三条线为 T=2k/5k/10k。每幅图使用该模型 parsed 数据最多的 prompt/order condition，0 线表示没有方向偏差。</figcaption></figure>
  <figure class="report-figure"><img src="assets/fig16_model_bias_oof_calibration.png" alt="Observed versus seed-held-out predicted mean asinh signed bias for each model." loading="lazy"><figcaption><strong>图 15｜模型内 bias law 的 held-out 校准。</strong> 横轴为 leave-one-seed-out 预测的 cell mean asinh(bias)，纵轴为观测 cell mean asinh(bias)；每点汇总一个模型的 (T,N) cell，虚线是理想 45° 线。散点远离对角线说明 prompt/order 截距与平滑 L/N law 仍不能解释全部 stimulus heterogeneity。</figcaption></figure>

  <div class="callout"><strong>结论。</strong> 可以为每个模型写出带参数的 signed-bias response surface，但“可写出”不等于“held-out 可预测”。稳健 asinh 目标比 raw mean 更适合作为经验 law；raw mean 往往被少数 enumeration 巨大多计主导。模型内 law 应作为典型错误方向的机制诊断，不能替代把所有 parse/truncation 计为失败的 exact-accuracy law。</div>
</section>
"""


def update_report_html(
    report_path: Path,
    addendum: str,
) -> None:
    text = report_path.read_text(encoding="utf-8")
    if 'id="model-bias-laws"' in text:
        raise RuntimeError("Model-bias addendum already present")
    nav_anchor = '    <a href="#limits">限制</a>'
    if nav_anchor not in text:
        raise RuntimeError("Navigation insertion anchor missing")
    text = text.replace(
        nav_anchor,
        '    <a href="#model-bias-laws">模型 bias laws</a>\n'
        + nav_anchor,
        1,
    )
    section_anchor = '<section id="limits">'
    if section_anchor not in text:
        raise RuntimeError("Section insertion anchor missing")
    text = text.replace(
        section_anchor,
        addendum + "\n" + section_anchor,
        1,
    )
    report_path.write_text(text, encoding="utf-8")


def update_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    addition = """

## Independent per-model signed-bias laws

This report version adds a model-by-model signed-bias analysis. It compares a
fixed, interpretable log-length/log-needle law with a bounded coordinate search,
uses four grouped validation schemes plus nested leave-one-seed-out selection,
and reports 500-replicate stimulus-cluster bootstrap intervals.

Reproduce only this addendum with:

```powershell
python scripts/build_model_bias_addendum.py `
  --base-report <verified-base-report-directory> `
  --output-dir <new-empty-output-directory>
```
"""
    path.write_text(text.rstrip() + addition, encoding="utf-8")


def update_manifest(
    path: Path,
    source_csv: Path,
    selected: pd.DataFrame,
    output_dir: Path,
    script_source: Path,
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_root"] = str(output_dir)
    manifest["modified_at_utc"] = utc_now()
    reproduction_scripts = manifest.setdefault("reproduction_scripts", [])
    reproduction_scripts = [
        item
        for item in reproduction_scripts
        if item.get("destination") != script_source.name
    ]
    reproduction_scripts.append(
        {
            "source": str(script_source.resolve()),
            "destination": script_source.name,
            "sha256": sha256(script_source),
        }
    )
    manifest["reproduction_scripts"] = reproduction_scripts
    manifest["model_specific_bias_laws_v1"] = {
        "created_at_utc": utc_now(),
        "definition": "signed_error = predicted_count - true_count",
        "sample": "parsed numeric outputs only; primary accuracy unchanged",
        "robust_target": "asinh(signed_error)",
        "asinh_centered_bias": "sinh(mean(asinh(signed_error)))",
        "coordinate_candidates": CANDIDATES,
        "validation": [
            "leave-one-seed-out",
            "leave-one-needle-level-out",
            "leave-one-length-level-out",
            "five blocked length/needle-cell folds",
            "nested leave-one-seed-out coordinate selection",
        ],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "source_request_table": str(source_csv),
        "source_request_table_sha256": sha256(source_csv),
        "supported_models": selected.loc[
            selected["evidence"] == "supported", "model_label"
        ].tolist(),
        "weak_models": selected.loc[
            selected["evidence"] == "weak", "model_label"
        ].tolist(),
        "unsupported_models": selected.loc[
            selected["evidence"] == "not_supported", "model_label"
        ].tolist(),
        "tables": [
            "model_specific_bias_candidate_comparison.csv",
            "model_specific_bias_selected_laws.csv",
            "model_specific_bias_fixed_log_parameters.csv",
            "model_specific_bias_nested_seed_folds.csv",
            "model_specific_bias_oof_cells.csv",
        ],
        "figures": [
            "fig13_model_bias_candidate_search.png",
            "fig14_model_bias_log_slopes.png",
            "fig15_model_bias_selected_surfaces.png",
            "fig16_model_bias_oof_calibration.png",
        ],
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def regenerate_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS.tsv"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{sha256(path)}\t{path.relative_to(root).as_posix()}"
        for path in files
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def main() -> None:
    started_at_utc = utc_now()
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    base_report = args.base_report.resolve()
    output_dir = args.output_dir.resolve()
    if not base_report.is_dir():
        raise FileNotFoundError(base_report)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    required = [
        base_report / "report.html",
        base_report / "README.md",
        base_report / "analysis_manifest.json",
        base_report / "tables" / "request_level_report.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    shutil.copytree(base_report, output_dir)
    tables = output_dir / "tables"
    assets = output_dir / "assets"
    scripts = output_dir / "scripts"
    source_csv = base_report / "tables" / "request_level_report.csv"

    requests = pd.read_csv(source_csv, low_memory=False)
    if len(requests) != 6300:
        raise ValueError(f"Expected 6300 requests, got {len(requests)}")
    parsed = requests.loc[requests["parse_success"] == 1].copy()
    parsed["signed_error"] = pd.to_numeric(
        parsed["signed_error"], errors="raise"
    )
    if len(parsed) != 5385:
        raise ValueError(f"Expected 5385 parsed outputs, got {len(parsed)}")
    parsed = add_coordinates(parsed)

    (
        comparison,
        selected,
        fixed,
        nested,
        calibration,
        model_fits,
    ) = analyze_models(parsed)

    comparison.to_csv(
        tables / "model_specific_bias_candidate_comparison.csv",
        index=False,
    )
    selected.to_csv(
        tables / "model_specific_bias_selected_laws.csv",
        index=False,
    )
    fixed.to_csv(
        tables / "model_specific_bias_fixed_log_parameters.csv",
        index=False,
    )
    nested.to_csv(
        tables / "model_specific_bias_nested_seed_folds.csv",
        index=False,
    )

    make_candidate_figure(
        comparison,
        selected,
        assets / "fig13_model_bias_candidate_search.png",
    )
    make_slope_figure(
        fixed,
        assets / "fig14_model_bias_log_slopes.png",
    )
    make_surface_figure(
        model_fits,
        assets / "fig15_model_bias_selected_surfaces.png",
    )
    grouped_calibration = make_calibration_figure(
        calibration,
        assets / "fig16_model_bias_oof_calibration.png",
    )
    grouped_calibration.to_csv(
        tables / "model_specific_bias_oof_cells.csv",
        index=False,
    )

    shutil.copy2(Path(__file__), scripts / Path(__file__).name)
    addendum = build_addendum_html(selected, fixed)
    update_report_html(output_dir / "report.html", addendum)
    update_readme(output_dir / "README.md")
    update_manifest(
        output_dir / "analysis_manifest.json",
        source_csv,
        selected,
        output_dir,
        Path(__file__),
    )
    build_log = {
        "status": "complete",
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "base_report": str(base_report),
        "output_dir": str(output_dir),
        "source_request_table": str(source_csv),
        "source_request_table_sha256": sha256(source_csv),
        "requests": len(requests),
        "parsed_outputs": len(parsed),
        "models": len(MODELS),
        "bootstrap_replicates_per_model_and_form": BOOTSTRAP_REPLICATES,
        "supported_models": selected.loc[
            selected["evidence"] == "supported", "model_label"
        ].tolist(),
        "weak_models": selected.loc[
            selected["evidence"] == "weak", "model_label"
        ].tolist(),
        "unsupported_models": selected.loc[
            selected["evidence"] == "not_supported", "model_label"
        ].tolist(),
    }
    (output_dir / "logs" / "model_bias_build_log.json").write_text(
        json.dumps(build_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_entries = regenerate_checksums(output_dir)

    summary = {
        "status": "complete",
        "output_dir": str(output_dir),
        "parsed_outputs": len(parsed),
        "supported_models": selected.loc[
            selected["evidence"] == "supported", "model_label"
        ].tolist(),
        "weak_models": selected.loc[
            selected["evidence"] == "weak", "model_label"
        ].tolist(),
        "unsupported_models": selected.loc[
            selected["evidence"] == "not_supported", "model_label"
        ].tolist(),
        "checksum_entries": checksum_entries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
