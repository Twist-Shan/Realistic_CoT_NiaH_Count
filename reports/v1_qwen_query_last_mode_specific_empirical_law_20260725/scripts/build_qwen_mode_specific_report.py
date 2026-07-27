"""Build a mode-specific Qwen query-last empirical-law report.

This analysis is intentionally narrower than the eight-model report:

* only Qwen3-1.7B, Qwen3-8B, and Qwen3-32B;
* only query_last;
* each model x prompt-mode stratum is fitted independently;
* direct, enumeration, and native-thinking never share a forced formula;
* a bounded expanded candidate grid is selected independently in each stratum;
* reported goodness-of-fit uses nested grouped validation, not training fit;
* the original fixed log-separable law remains a transparent reference.

The source request-level CSV and frozen prompt artifacts are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPORT_MARKER = "QWEN_QUERY_LAST_MODE_SPECIFIC_V2"
RNG_SEED = 20260725
BOOTSTRAP_REPLICATES = 500
RIDGE_LOGISTIC = 1e-2
RIDGE_OLS = 1e-8
EPS = 1e-8

MODELS = ["Qwen3-1.7B", "Qwen3-8B", "Qwen3-32B"]
MODES = ["direct", "enumeration", "native_thinking"]
MODE_LABEL = {
    "direct": "Direct",
    "enumeration": "Enumeration",
    "native_thinking": "Native thinking",
}
TARGET_LENGTHS = [2000, 5000, 10000]
NEEDLE_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 20, 30]
SEEDS = [1234, 1235, 1236, 1237, 1238]

MODEL_METADATA = {
    "Qwen3-1.7B": {
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "batch": 16,
    },
    "Qwen3-8B": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "batch": 8,
    },
    "Qwen3-32B": {
        "model_id": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "batch": 1,
    },
}

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


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    features: tuple[str, ...]
    complexity: int
    interpretation: str


CANDIDATES = [
    Candidate(
        "intercept_only",
        "Constant",
        (),
        0,
        "No T/N dependence.",
    ),
    Candidate(
        "log_density",
        "Log density",
        ("log_density",),
        1,
        "Depends only on ρ = 1000N/T.",
    ),
    Candidate(
        "log_burden",
        "Log burden",
        ("log_burden",),
        1,
        "Depends on the product of relative length and needle count.",
    ),
    Candidate(
        "log_separable",
        "Separate log T + log N",
        ("log_length", "log_needles"),
        2,
        "Power-order terms for T and N with no interaction.",
    ),
    Candidate(
        "semi_log_needles",
        "Log T + linear N",
        ("log_length", "needle_linear"),
        2,
        "Power in T and exponential-in-N odds or transformed error.",
    ),
    Candidate(
        "piecewise_log_needles",
        "Log T + piecewise log N",
        ("log_length", "log_needles", "needle_hinge"),
        3,
        "Allows the N slope to change above N=5.",
    ),
    Candidate(
        "log_interaction",
        "Log T × log N",
        ("log_length", "log_needles", "log_interaction"),
        3,
        "Allows the apparent order in one variable to depend on the other.",
    ),
]
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}

# Expanded, still finite, mode-specific search grid.  These candidates are
# evaluated independently for every model x mode x target stratum.  The grid is
# intentionally fixed here before fitting so that an attractive plot cannot
# trigger an unrecorded post-hoc formula.
FLEX_CANDIDATES = [
    Candidate("constant", "Constant", (), 0, "No T/N dependence."),
    Candidate(
        "log_needles_only",
        "Log N only",
        ("log_needles",),
        1,
        "A single power/exponent term in needle count.",
    ),
    Candidate(
        "linear_needles_only",
        "Linear N only",
        ("needle_linear",),
        1,
        "An exponential-in-N accuracy odds or linear transformed error.",
    ),
    Candidate(
        "log_length_only",
        "Log T only",
        ("log_length",),
        1,
        "A single power/exponent term in passage length.",
    ),
    Candidate(
        "log_density",
        "Log density",
        ("log_density",),
        1,
        "Depends only on rho = 1000N/T.",
    ),
    Candidate(
        "log_burden",
        "Log burden",
        ("log_burden",),
        1,
        "Depends on (T/5000)(N/5).",
    ),
    Candidate(
        "log_separable",
        "Log T + log N",
        ("log_length", "log_needles"),
        2,
        "Separate T and N orders.",
    ),
    Candidate(
        "semi_log_needles",
        "Log T + linear N",
        ("log_length", "needle_linear"),
        2,
        "Power in T with exponential-in-N response.",
    ),
    Candidate(
        "quadratic_log_needles",
        "Log T + curved log N",
        ("log_length", "log_needles", "log_needles_sq"),
        3,
        "Allows smooth curvature along N.",
    ),
    Candidate(
        "piecewise_log_needles",
        "Log T + piecewise log N",
        ("log_length", "log_needles", "needle_hinge"),
        3,
        "Allows the N slope to change above N=5.",
    ),
    Candidate(
        "log_interaction",
        "Log T x log N",
        ("log_length", "log_needles", "log_interaction"),
        3,
        "Lets the apparent N order depend on T.",
    ),
    Candidate(
        "quadratic_surface",
        "Quadratic log surface",
        (
            "log_length",
            "log_needles",
            "log_length_sq",
            "log_needles_sq",
            "log_interaction",
        ),
        5,
        "A smooth second-order surface in log T and log N.",
    ),
    Candidate(
        "length_factor_log_n",
        "T-level intercepts + log N",
        ("t_is_2000", "t_is_10000", "log_needles"),
        3,
        "Allows each observed T level its own intercept.",
    ),
    Candidate(
        "length_factor_curved_n",
        "T-level intercepts + curved log N",
        ("t_is_2000", "t_is_10000", "log_needles", "log_needles_sq"),
        4,
        "Observed T-level intercepts plus smooth N curvature.",
    ),
    Candidate(
        "length_specific_log_n",
        "T-specific log-N slopes",
        (
            "t_is_2000",
            "t_is_10000",
            "log_needles",
            "t2000_log_needles",
            "t10000_log_needles",
        ),
        5,
        "A separate log-N slope at each observed T level.",
    ),
    Candidate(
        "length_specific_piecewise_n",
        "T-specific piecewise N",
        (
            "t_is_2000",
            "t_is_10000",
            "log_needles",
            "needle_hinge",
            "t2000_log_needles",
            "t10000_log_needles",
        ),
        6,
        "T-specific N slopes plus a shared high-N hinge.",
    ),
]
FLEX_CANDIDATE_BY_NAME = {
    candidate.name: candidate for candidate in FLEX_CANDIDATES
}

TARGETS = {
    "exact": {
        "label": "Exact correctness",
        "kind": "binary",
        "column": "exact_correct",
        "filter": lambda frame: np.ones(len(frame), dtype=bool),
        "transform": lambda frame: frame["exact_correct"].to_numpy(float),
        "loss": "log_loss",
    },
    "bias": {
        "label": "Signed bias",
        "kind": "continuous",
        "column": "signed_error",
        "filter": lambda frame: frame["parse_success"].eq(1)
        & frame["signed_error"].notna(),
        "transform": lambda frame: np.arcsinh(
            frame["signed_error"].to_numpy(float)
        ),
        "loss": "mae",
    },
    "absolute_error": {
        "label": "Absolute error",
        "kind": "continuous",
        "column": "absolute_error",
        "filter": lambda frame: frame["parse_success"].eq(1)
        & frame["absolute_error"].notna(),
        "transform": lambda frame: np.log1p(
            frame["absolute_error"].to_numpy(float)
        ),
        "loss": "mae",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def decimal(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):+.{digits}f}"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "qwen_query_last_builder.py"},
    )
    plt.close(fig)
    path.write_bytes(buffer.getvalue())


def add_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["T"] = pd.to_numeric(out["target_passage_tokens"], errors="raise")
    out["N"] = pd.to_numeric(out["num_needles"], errors="raise")
    out["log_length"] = np.log(out["T"].to_numpy(float) / 5000.0)
    out["log_needles"] = np.log(out["N"].to_numpy(float) / 5.0)
    out["density_per_1k_recomputed"] = (
        1000.0 * out["N"].to_numpy(float) / out["T"].to_numpy(float)
    )
    out["log_density"] = np.log(
        out["density_per_1k_recomputed"].to_numpy(float)
    )
    out["log_burden"] = out["log_length"] + out["log_needles"]
    out["needle_linear"] = (out["N"].to_numpy(float) - 5.0) / 5.0
    out["needle_hinge"] = np.maximum(out["log_needles"].to_numpy(float), 0.0)
    out["log_interaction"] = out["log_length"] * out["log_needles"]
    out["log_length_sq"] = out["log_length"] ** 2
    out["log_needles_sq"] = out["log_needles"] ** 2
    out["t_is_2000"] = out["T"].eq(2000).astype(float)
    out["t_is_10000"] = out["T"].eq(10000).astype(float)
    out["t2000_log_needles"] = out["t_is_2000"] * out["log_needles"]
    out["t10000_log_needles"] = out["t_is_10000"] * out["log_needles"]
    out["cell_id"] = (
        "T"
        + out["T"].astype(int).astype(str)
        + "_N"
        + out["N"].astype(int).astype(str)
    )
    return out


def validate_and_filter(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "request_id",
        "stimulus_id",
        "seed",
        "model_label",
        "model_id",
        "target_passage_tokens",
        "model_passage_tokens",
        "model_input_tokens",
        "num_needles",
        "density_per_1k",
        "prompt_mode",
        "thinking_enabled",
        "query_order",
        "exact_correct",
        "parse_success",
        "format_failure",
        "truncated",
        "gold_count",
        "predicted_count",
        "absolute_error",
        "signed_error",
    }
    full = pd.read_csv(source)
    missing = sorted(required - set(full.columns))
    if missing:
        raise ValueError(f"Missing request-level columns: {missing}")
    if len(full) != 6300 or full["request_id"].nunique() != 6300:
        raise ValueError("Source request table is not the audited 6,300-row table")
    subset = full[
        full["model_label"].isin(MODELS) & full["query_order"].eq("query_last")
    ].copy()
    if len(subset) != 1350 or subset["request_id"].nunique() != 1350:
        raise ValueError("Expected 1,350 unique Qwen query-last requests")
    counts = subset.groupby(["model_label", "prompt_mode"]).size().to_dict()
    expected = {(model, mode): 150 for model in MODELS for mode in MODES}
    if counts != expected:
        raise ValueError(f"Unexpected stratum counts: {counts}")
    for (model, mode), part in subset.groupby(
        ["model_label", "prompt_mode"], sort=False
    ):
        if sorted(part["target_passage_tokens"].unique()) != TARGET_LENGTHS:
            raise ValueError(f"Length grid mismatch for {model}/{mode}")
        if sorted(part["num_needles"].unique()) != NEEDLE_COUNTS:
            raise ValueError(f"Needle grid mismatch for {model}/{mode}")
        if sorted(part["seed"].unique()) != SEEDS:
            raise ValueError(f"Seed grid mismatch for {model}/{mode}")
        grid = part.groupby(
            ["target_passage_tokens", "num_needles", "seed"]
        ).size()
        if len(grid) != 150 or not grid.eq(1).all():
            raise ValueError(f"Grid duplication/missing cell for {model}/{mode}")
    subset = add_coordinates(subset)
    if not np.allclose(
        subset["density_per_1k"].to_numpy(float),
        subset["density_per_1k_recomputed"].to_numpy(float),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Stored density does not equal 1000*N/T")
    return full, subset


def candidate_matrix(
    frame: pd.DataFrame, candidate_name: str
) -> tuple[np.ndarray, list[str]]:
    candidate = CANDIDATE_BY_NAME[candidate_name]
    columns = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    for feature in candidate.features:
        columns.append(frame[feature].to_numpy(float))
        names.append(feature)
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite design matrix for {candidate_name}")
    return matrix, names


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic(
    matrix: np.ndarray,
    outcome: np.ndarray,
    ridge: float = RIDGE_LOGISTIC,
    max_iterations: int = 100,
) -> np.ndarray:
    if matrix.ndim != 2 or len(matrix) != len(outcome):
        raise ValueError("Logistic matrix/outcome shape mismatch")
    prevalence = (float(outcome.sum()) + 0.5) / (len(outcome) + 1.0)
    beta = np.zeros(matrix.shape[1], dtype=float)
    beta[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.full(matrix.shape[1], ridge, dtype=float)
    penalty[0] = ridge * 0.1
    for _ in range(max_iterations):
        probability = np.clip(sigmoid(matrix @ beta), EPS, 1.0 - EPS)
        gradient = matrix.T @ (probability - outcome) + penalty * beta
        weights = probability * (1.0 - probability)
        hessian = matrix.T @ (weights[:, None] * matrix)
        hessian.flat[:: hessian.shape[0] + 1] += penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        old_objective = logistic_objective(
            matrix, outcome, beta, penalty=penalty
        )
        scale = 1.0
        updated = beta - step
        while (
            logistic_objective(matrix, outcome, updated, penalty=penalty)
            > old_objective
            and scale > 1e-6
        ):
            scale *= 0.5
            updated = beta - scale * step
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    if not np.isfinite(beta).all():
        raise RuntimeError("Non-finite logistic coefficients")
    return beta


def logistic_objective(
    matrix: np.ndarray,
    outcome: np.ndarray,
    beta: np.ndarray,
    *,
    penalty: np.ndarray,
) -> float:
    linear = np.clip(matrix @ beta, -35.0, 35.0)
    negative_log_likelihood = np.logaddexp(0.0, linear).sum() - (
        outcome * linear
    ).sum()
    return float(negative_log_likelihood + 0.5 * np.sum(penalty * beta**2))


def fit_ols(
    matrix: np.ndarray, outcome: np.ndarray, ridge: float = RIDGE_OLS
) -> np.ndarray:
    penalty = np.eye(matrix.shape[1], dtype=float) * ridge
    penalty[0, 0] = ridge * 0.1
    return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ outcome)


def predict(
    kind: str, matrix: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    linear = matrix @ coefficients
    return sigmoid(linear) if kind == "binary" else linear


def binary_metrics(outcome: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, EPS, 1.0 - EPS)
    log_loss = -np.mean(
        outcome * np.log(prediction) + (1.0 - outcome) * np.log(1.0 - prediction)
    )
    return {
        "log_loss": float(log_loss),
        "brier": float(np.mean((outcome - prediction) ** 2)),
    }


def continuous_metrics(
    outcome: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    residual = outcome - prediction
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def fold_masks(frame: pd.DataFrame, scheme: str) -> list[tuple[str, np.ndarray]]:
    if scheme == "seed":
        return [
            (f"seed={seed}", frame["seed"].to_numpy() == seed)
            for seed in SEEDS
        ]
    if scheme == "needle":
        return [
            (f"N={needle}", frame["N"].to_numpy() == needle)
            for needle in NEEDLE_COUNTS
        ]
    if scheme == "length":
        return [
            (f"T={length}", frame["T"].to_numpy() == length)
            for length in TARGET_LENGTHS
        ]
    raise KeyError(scheme)


def evaluate_candidate(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    kind: str,
    candidate_name: str,
    scheme: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    matrix, _ = candidate_matrix(frame, candidate_name)
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for label, test in fold_masks(frame, scheme):
        train = ~test
        if not test.any() or not train.any():
            raise RuntimeError(f"Empty fold {scheme}/{label}")
        coefficients = (
            fit_logistic(matrix[train], outcome[train])
            if kind == "binary"
            else fit_ols(matrix[train], outcome[train])
        )
        oof[test] = predict(kind, matrix[test], coefficients)
        metrics = (
            binary_metrics(outcome[test], oof[test])
            if kind == "binary"
            else continuous_metrics(outcome[test], oof[test])
        )
        fold_rows.append({"fold": label, **metrics})
    if not np.isfinite(oof).all():
        raise RuntimeError(f"Incomplete OOF prediction for {candidate_name}/{scheme}")
    metrics = (
        binary_metrics(outcome, oof)
        if kind == "binary"
        else continuous_metrics(outcome, oof)
    )
    metric_name = "log_loss" if kind == "binary" else "mae"
    fold_values = np.array([row[metric_name] for row in fold_rows], dtype=float)
    metrics[f"{metric_name}_fold_se"] = (
        float(np.std(fold_values, ddof=1) / np.sqrt(len(fold_values)))
        if len(fold_values) > 1
        else 0.0
    )
    prediction_frame = frame[
        ["request_id", "stimulus_id", "seed", "T", "N", "cell_id"]
    ].copy()
    prediction_frame["observed"] = outcome
    prediction_frame["oof_prediction"] = oof
    prediction_frame["scheme"] = scheme
    prediction_frame["candidate"] = candidate_name
    return metrics, prediction_frame


def fit_full(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    kind: str,
    candidate_name: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    matrix, names = candidate_matrix(frame, candidate_name)
    coefficients = (
        fit_logistic(matrix, outcome)
        if kind == "binary"
        else fit_ols(matrix, outcome)
    )
    return coefficients, names, predict(kind, matrix, coefficients)


def nested_model_test(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    kind: str,
) -> tuple[float, float, float]:
    """Return statistic, unadjusted p-value, and full-model fit score."""
    full_matrix, _ = candidate_matrix(frame, "log_separable")
    null_matrix, _ = candidate_matrix(frame, "intercept_only")
    if kind == "binary":
        full_beta = fit_logistic(full_matrix, outcome)
        null_beta = fit_logistic(null_matrix, outcome)
        full_p = np.clip(sigmoid(full_matrix @ full_beta), EPS, 1.0 - EPS)
        null_p = np.clip(sigmoid(null_matrix @ null_beta), EPS, 1.0 - EPS)
        full_ll = float(
            np.sum(
                outcome * np.log(full_p)
                + (1.0 - outcome) * np.log(1.0 - full_p)
            )
        )
        null_ll = float(
            np.sum(
                outcome * np.log(null_p)
                + (1.0 - outcome) * np.log(1.0 - null_p)
            )
        )
        statistic = max(0.0, 2.0 * (full_ll - null_ll))
        p_value = float(stats.chi2.sf(statistic, df=2))
        score = float(binary_metrics(outcome, full_p)["log_loss"])
        return statistic, p_value, score
    full_beta = fit_ols(full_matrix, outcome)
    null_beta = fit_ols(null_matrix, outcome)
    full_residual = outcome - full_matrix @ full_beta
    null_residual = outcome - null_matrix @ null_beta
    full_sse = float(np.sum(full_residual**2))
    null_sse = float(np.sum(null_residual**2))
    df_num = full_matrix.shape[1] - null_matrix.shape[1]
    df_den = len(outcome) - full_matrix.shape[1]
    numerator = max(0.0, (null_sse - full_sse) / max(df_num, 1))
    denominator = full_sse / max(df_den, 1)
    statistic = numerator / max(denominator, EPS)
    p_value = float(stats.f.sf(statistic, df_num, df_den))
    score = float(continuous_metrics(outcome, full_matrix @ full_beta)["mae"])
    return statistic, p_value, score


def bootstrap_coefficients(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    kind: str,
    candidate_name: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    matrix, names = candidate_matrix(frame, candidate_name)
    cluster_indices = [
        np.flatnonzero(frame["cell_id"].to_numpy() == cell)
        for cell in sorted(frame["cell_id"].unique())
    ]
    draws: list[np.ndarray] = []
    for _ in range(replicates):
        chosen = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
        index = np.concatenate([cluster_indices[position] for position in chosen])
        coefficients = (
            fit_logistic(matrix[index], outcome[index])
            if kind == "binary"
            else fit_ols(matrix[index], outcome[index])
        )
        if np.isfinite(coefficients).all():
            draws.append(coefficients)
    if len(draws) < int(0.95 * replicates):
        raise RuntimeError(
            f"Only {len(draws)}/{replicates} valid bootstrap fits for "
            f"{candidate_name}"
        )
    return np.asarray(draws, dtype=float), names


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    total = len(values)
    for reverse_rank in range(total - 1, -1, -1):
        original_index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[original_index] * total / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted


def interval_excludes_zero(low: float, high: float) -> bool:
    return bool((low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0))


def r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= EPS:
        return float("nan")
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def one_se_selection(
    comparison: pd.DataFrame, target: str, model: str, mode: str
) -> str:
    part = comparison[
        comparison["target"].eq(target)
        & comparison["model_label"].eq(model)
        & comparison["prompt_mode"].eq(mode)
        & comparison["scheme"].eq("seed")
    ].copy()
    metric = "log_loss" if TARGETS[target]["kind"] == "binary" else "mae"
    best = part.sort_values([metric, "complexity", "candidate"]).iloc[0]
    cutoff = float(best[metric] + best[f"{metric}_fold_se"])
    eligible = part[part[metric].le(cutoff)].sort_values(
        ["complexity", metric, "candidate"]
    )
    return str(eligible.iloc[0]["candidate"])


def selected_formula(candidate_name: str, target: str) -> str:
    left = (
        "logit p"
        if target == "exact"
        else "E[asinh(b) | parsed]"
        if target == "bias"
        else "E[log(1+|b|) | parsed]"
    )
    terms = {
        "intercept_only": "α",
        "log_density": "α + βρ ln(ρ/1)",
        "log_burden": "α + βB ln[(T/5000)(N/5)]",
        "log_separable": "α + βT ln(T/5000) + βN ln(N/5)",
        "semi_log_needles": "α + βT ln(T/5000) + γN (N−5)/5",
        "piecewise_log_needles": (
            "α + βT ln(T/5000) + βN ln(N/5) "
            "+ βH max[0, ln(N/5)]"
        ),
        "log_interaction": (
            "α + βT ln(T/5000) + βN ln(N/5) "
            "+ βTN ln(T/5000)ln(N/5)"
        ),
    }
    return f"{left} = {terms[candidate_name]}"


def analyze(
    subset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []

    for model in MODELS:
        for mode in MODES:
            stratum = subset[
                subset["model_label"].eq(model)
                & subset["prompt_mode"].eq(mode)
            ].copy()
            parsed = stratum[stratum["parse_success"].eq(1)].copy()
            summary_rows.append(
                {
                    "model_label": model,
                    "prompt_mode": mode,
                    "requests": len(stratum),
                    "exact_correct": int(stratum["exact_correct"].sum()),
                    "exact_accuracy": float(stratum["exact_correct"].mean()),
                    "parsed_requests": int(stratum["parse_success"].sum()),
                    "parse_success_rate": float(stratum["parse_success"].mean()),
                    "format_failures": int(stratum["format_failure"].sum()),
                    "truncations": int(stratum["truncated"].sum()),
                    "mean_signed_error_parsed": float(parsed["signed_error"].mean()),
                    "median_signed_error_parsed": float(
                        parsed["signed_error"].median()
                    ),
                    "mean_absolute_error_parsed": float(
                        parsed["absolute_error"].mean()
                    ),
                }
            )
            cells = (
                stratum.groupby(["T", "N"], as_index=False)
                .agg(
                    requests=("request_id", "size"),
                    exact_correct=("exact_correct", "sum"),
                    exact_accuracy=("exact_correct", "mean"),
                    parsed_requests=("parse_success", "sum"),
                    parse_success_rate=("parse_success", "mean"),
                    mean_signed_error_parsed=("signed_error", "mean"),
                    median_signed_error_parsed=("signed_error", "median"),
                    mean_absolute_error_parsed=("absolute_error", "mean"),
                )
                .assign(model_label=model, prompt_mode=mode)
            )
            cell_rows.extend(cells.to_dict("records"))

            for target_name, target_spec in TARGETS.items():
                target_frame = stratum.loc[target_spec["filter"](stratum)].copy()
                outcome = target_spec["transform"](target_frame)
                kind = str(target_spec["kind"])
                for candidate in CANDIDATES:
                    for scheme in ("seed", "needle", "length"):
                        metrics, predictions = evaluate_candidate(
                            target_frame,
                            outcome,
                            kind=kind,
                            candidate_name=candidate.name,
                            scheme=scheme,
                        )
                        comparison_rows.append(
                            {
                                "target": target_name,
                                "model_label": model,
                                "prompt_mode": mode,
                                "candidate": candidate.name,
                                "candidate_label": candidate.label,
                                "complexity": candidate.complexity,
                                "scheme": scheme,
                                "n": len(target_frame),
                                **metrics,
                            }
                        )
                        for fold_label, test in fold_masks(target_frame, scheme):
                            fold_outcome = predictions.loc[
                                test, "observed"
                            ].to_numpy(float)
                            fold_prediction = predictions.loc[
                                test, "oof_prediction"
                            ].to_numpy(float)
                            fold_metric = (
                                binary_metrics(fold_outcome, fold_prediction)
                                if kind == "binary"
                                else continuous_metrics(
                                    fold_outcome, fold_prediction
                                )
                            )
                            fold_rows.append(
                                {
                                    "target": target_name,
                                    "model_label": model,
                                    "prompt_mode": mode,
                                    "candidate": candidate.name,
                                    "scheme": scheme,
                                    "fold": fold_label,
                                    "n_test": int(np.sum(test)),
                                    **fold_metric,
                                }
                            )

                fixed_metrics: dict[str, float] = {}
                fixed_oof_seed: pd.DataFrame | None = None
                for scheme in ("seed", "needle", "length"):
                    law_metrics, law_predictions = evaluate_candidate(
                        target_frame,
                        outcome,
                        kind=kind,
                        candidate_name="log_separable",
                        scheme=scheme,
                    )
                    null_metrics, _ = evaluate_candidate(
                        target_frame,
                        outcome,
                        kind=kind,
                        candidate_name="intercept_only",
                        scheme=scheme,
                    )
                    primary_metric = (
                        "log_loss" if kind == "binary" else "mae"
                    )
                    fixed_metrics[f"{scheme}_{primary_metric}"] = law_metrics[
                        primary_metric
                    ]
                    fixed_metrics[f"{scheme}_{primary_metric}_null"] = null_metrics[
                        primary_metric
                    ]
                    fixed_metrics[f"{scheme}_gain_pct"] = 100.0 * (
                        null_metrics[primary_metric] - law_metrics[primary_metric]
                    ) / max(null_metrics[primary_metric], EPS)
                    if kind == "binary":
                        fixed_metrics[f"{scheme}_brier"] = law_metrics["brier"]
                    if scheme == "seed":
                        fixed_oof_seed = law_predictions

                coefficients, coefficient_names, full_prediction = fit_full(
                    target_frame,
                    outcome,
                    kind=kind,
                    candidate_name="log_separable",
                )
                rng_offset = (
                    MODELS.index(model) * 100
                    + MODES.index(mode) * 10
                    + list(TARGETS).index(target_name)
                )
                bootstrap, bootstrap_names = bootstrap_coefficients(
                    target_frame,
                    outcome,
                    kind=kind,
                    candidate_name="log_separable",
                    replicates=BOOTSTRAP_REPLICATES,
                    rng=np.random.default_rng(RNG_SEED + rng_offset),
                )
                if coefficient_names != bootstrap_names:
                    raise RuntimeError("Bootstrap coefficient-name mismatch")
                coefficient_map = dict(zip(coefficient_names, coefficients))
                interval_map = {
                    name: (
                        float(np.percentile(bootstrap[:, position], 2.5)),
                        float(np.percentile(bootstrap[:, position], 97.5)),
                    )
                    for position, name in enumerate(coefficient_names)
                }
                for draw_index, draw in enumerate(bootstrap):
                    for name, value in zip(coefficient_names, draw):
                        bootstrap_rows.append(
                            {
                                "target": target_name,
                                "model_label": model,
                                "prompt_mode": mode,
                                "replicate": draw_index,
                                "coefficient": name,
                                "value": float(value),
                            }
                        )
                statistic, p_value, in_sample_score = nested_model_test(
                    target_frame, outcome, kind=kind
                )
                if fixed_oof_seed is None:
                    raise RuntimeError("Missing seed OOF predictions")
                aggregated = (
                    fixed_oof_seed.groupby(["T", "N"], as_index=False)
                    .agg(
                        observed=("observed", "mean"),
                        predicted=("oof_prediction", "mean"),
                    )
                )
                cell_r2 = r2_score(
                    aggregated["observed"].to_numpy(float),
                    aggregated["predicted"].to_numpy(float),
                )
                fixed_row = {
                    "target": target_name,
                    "target_label": target_spec["label"],
                    "model_label": model,
                    "prompt_mode": mode,
                    "n": len(target_frame),
                    "event_count": (
                        int(outcome.sum()) if kind == "binary" else np.nan
                    ),
                    "failure_count": (
                        int(len(outcome) - outcome.sum())
                        if kind == "binary"
                        else np.nan
                    ),
                    "intercept": coefficient_map["intercept"],
                    "intercept_ci95_low": interval_map["intercept"][0],
                    "intercept_ci95_high": interval_map["intercept"][1],
                    "length_order": coefficient_map["log_length"],
                    "length_order_ci95_low": interval_map["log_length"][0],
                    "length_order_ci95_high": interval_map["log_length"][1],
                    "needle_order": coefficient_map["log_needles"],
                    "needle_order_ci95_low": interval_map["log_needles"][0],
                    "needle_order_ci95_high": interval_map["log_needles"][1],
                    "length_doubling_factor": float(
                        2.0 ** coefficient_map["log_length"]
                    ),
                    "needle_doubling_factor": float(
                        2.0 ** coefficient_map["log_needles"]
                    ),
                    "global_test_statistic": statistic,
                    "global_p_value": p_value,
                    "in_sample_score": in_sample_score,
                    "seed_oof_cell_r2": cell_r2,
                    **fixed_metrics,
                }
                fixed_rows.append(fixed_row)
                oof = fixed_oof_seed.copy()
                oof.insert(0, "target", target_name)
                oof.insert(1, "model_label", model)
                oof.insert(2, "prompt_mode", mode)
                oof_rows.append(oof)

    summary = pd.DataFrame(summary_rows)
    cells = pd.DataFrame(cell_rows)
    comparison = pd.DataFrame(comparison_rows)
    folds = pd.DataFrame(fold_rows)
    fixed = pd.DataFrame(fixed_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    fixed["global_q_value"] = np.nan
    for target_name in TARGETS:
        index = fixed.index[fixed["target"].eq(target_name)]
        fixed.loc[index, "global_q_value"] = bh_adjust(
            fixed.loc[index, "global_p_value"].to_numpy(float)
        )

    evidence: list[str] = []
    for _, row in fixed.iterrows():
        slope_supported = interval_excludes_zero(
            row["length_order_ci95_low"], row["length_order_ci95_high"]
        ) or interval_excludes_zero(
            row["needle_order_ci95_low"], row["needle_order_ci95_high"]
        )
        gain = float(row["seed_gain_pct"])
        q_value = float(row["global_q_value"])
        if row["target"] == "exact" and int(row["failure_count"]) < 10:
            label = "ceiling_limited"
        elif q_value < 0.05 and gain > 2.0 and slope_supported:
            label = "supported"
        elif (
            (q_value < 0.10 and gain > 0 and slope_supported)
            or (gain > 5.0 and slope_supported)
        ):
            label = "suggestive"
        else:
            label = "not_supported"
        evidence.append(label)
    fixed["evidence"] = evidence

    for target_name in TARGETS:
        for model in MODELS:
            for mode in MODES:
                selected_name = one_se_selection(
                    comparison, target_name, model, mode
                )
                stratum = subset[
                    subset["model_label"].eq(model)
                    & subset["prompt_mode"].eq(mode)
                ].copy()
                target_spec = TARGETS[target_name]
                target_frame = stratum.loc[target_spec["filter"](stratum)].copy()
                outcome = target_spec["transform"](target_frame)
                kind = str(target_spec["kind"])
                coefficients, names, _ = fit_full(
                    target_frame,
                    outcome,
                    kind=kind,
                    candidate_name=selected_name,
                )
                selected_cv = comparison[
                    comparison["target"].eq(target_name)
                    & comparison["model_label"].eq(model)
                    & comparison["prompt_mode"].eq(mode)
                    & comparison["candidate"].eq(selected_name)
                    & comparison["scheme"].eq("seed")
                ].iloc[0]
                null_cv = comparison[
                    comparison["target"].eq(target_name)
                    & comparison["model_label"].eq(model)
                    & comparison["prompt_mode"].eq(mode)
                    & comparison["candidate"].eq("intercept_only")
                    & comparison["scheme"].eq("seed")
                ].iloc[0]
                metric = "log_loss" if kind == "binary" else "mae"
                selected_rows.append(
                    {
                        "target": target_name,
                        "model_label": model,
                        "prompt_mode": mode,
                        "n": len(target_frame),
                        "selected_candidate": selected_name,
                        "selected_candidate_label": CANDIDATE_BY_NAME[
                            selected_name
                        ].label,
                        "formula": selected_formula(selected_name, target_name),
                        "coefficient_names": json.dumps(names),
                        "coefficients": json.dumps(
                            [float(value) for value in coefficients]
                        ),
                        f"seed_oof_{metric}": float(selected_cv[metric]),
                        f"seed_oof_{metric}_null": float(null_cv[metric]),
                        "seed_gain_pct": 100.0
                        * (float(null_cv[metric]) - float(selected_cv[metric]))
                        / max(float(null_cv[metric]), EPS),
                    }
                )

    selected = pd.DataFrame(selected_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    return (
        summary,
        cells,
        comparison,
        folds,
        fixed,
        selected,
        oof,
        bootstrap_frame,
    )


@dataclass(frozen=True)
class FlexSpec:
    """A complete mode-specific model specification."""

    name: str
    candidate: str
    estimator: str
    transform: str
    ridge: float
    complexity: int
    preference_rank: int
    label: str


FLEX_TERMS = {
    "constant": "alpha",
    "log_needles_only": "alpha + beta_N ln(N/5)",
    "linear_needles_only": "alpha + gamma_N (N-5)/5",
    "log_length_only": "alpha + beta_T ln(T/5000)",
    "log_density": "alpha + beta_rho ln(rho/1)",
    "log_burden": "alpha + beta_B ln[(T/5000)(N/5)]",
    "log_separable": "alpha + beta_T ln(T/5000) + beta_N ln(N/5)",
    "semi_log_needles": "alpha + beta_T ln(T/5000) + gamma_N (N-5)/5",
    "quadratic_log_needles": (
        "alpha + beta_T ln(T/5000) + beta_N ln(N/5) "
        "+ beta_N2 ln(N/5)^2"
    ),
    "piecewise_log_needles": (
        "alpha + beta_T ln(T/5000) + beta_N ln(N/5) "
        "+ beta_H max[0,ln(N/5)]"
    ),
    "log_interaction": (
        "alpha + beta_T ln(T/5000) + beta_N ln(N/5) "
        "+ beta_TN ln(T/5000)ln(N/5)"
    ),
    "quadratic_surface": (
        "alpha + beta_T ln(T/5000) + beta_N ln(N/5) "
        "+ beta_T2 ln(T/5000)^2 + beta_N2 ln(N/5)^2 "
        "+ beta_TN ln(T/5000)ln(N/5)"
    ),
    "length_factor_log_n": "alpha_T + beta_N ln(N/5)",
    "length_factor_curved_n": (
        "alpha_T + beta_N ln(N/5) + beta_N2 ln(N/5)^2"
    ),
    "length_specific_log_n": "alpha_T + beta_N,T ln(N/5)",
    "length_specific_piecewise_n": (
        "alpha_T + beta_N,T ln(N/5) + beta_H max[0,ln(N/5)]"
    ),
}


def build_flex_specs(target: str) -> list[FlexSpec]:
    specs: list[FlexSpec] = []
    if target == "exact":
        methods = [
            ("logistic", "identity", 0.3, 0, "ridge=0.3"),
            ("logistic", "identity", 0.03, 1, "ridge=0.03"),
            ("logistic", "identity", 0.001, 2, "ridge=0.001"),
        ]
    elif target == "bias":
        methods = [
            ("ridge", "asinh", 1e-3, 0, "asinh + ridge"),
            ("huber", "asinh", 1e-2, 1, "asinh + Huber"),
            ("huber", "identity", 1e-2, 2, "raw bias + Huber"),
        ]
    elif target == "absolute_error":
        methods = [
            ("ridge", "log1p", 1e-3, 0, "log1p + ridge"),
            ("huber", "log1p", 1e-2, 1, "log1p + Huber"),
            ("huber", "sqrt", 1e-2, 2, "sqrt + Huber"),
        ]
    else:  # pragma: no cover - registry guard
        raise KeyError(target)
    for candidate in FLEX_CANDIDATES:
        for estimator, transform, ridge, rank, method_label in methods:
            name = f"{candidate.name}__{transform}__{estimator}__{ridge:g}"
            specs.append(
                FlexSpec(
                    name=name,
                    candidate=candidate.name,
                    estimator=estimator,
                    transform=transform,
                    ridge=ridge,
                    complexity=candidate.complexity,
                    preference_rank=rank,
                    label=f"{candidate.label}; {method_label}",
                )
            )
    return specs


def flex_matrix(
    frame: pd.DataFrame, candidate_name: str
) -> tuple[np.ndarray, list[str]]:
    candidate = FLEX_CANDIDATE_BY_NAME[candidate_name]
    columns = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    for feature in candidate.features:
        columns.append(frame[feature].to_numpy(float))
        names.append(feature)
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite flexible matrix for {candidate_name}")
    return matrix, names


def transform_outcome(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "identity":
        return values
    if transform == "asinh":
        return np.arcsinh(values)
    if transform == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    if transform == "sqrt":
        return np.sqrt(np.maximum(values, 0.0))
    raise KeyError(transform)


def inverse_outcome(values: np.ndarray, transform: str, target: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "identity":
        output = values
    elif transform == "asinh":
        output = np.sinh(np.clip(values, -12.0, 12.0))
    elif transform == "log1p":
        output = np.expm1(np.clip(values, 0.0, 12.0))
    elif transform == "sqrt":
        output = np.maximum(values, 0.0) ** 2
    else:  # pragma: no cover - registry guard
        raise KeyError(transform)
    if target == "absolute_error":
        output = np.maximum(output, 0.0)
    return output


def fit_huber(
    matrix: np.ndarray,
    outcome: np.ndarray,
    *,
    ridge: float,
    delta: float = 1.345,
    max_iterations: int = 100,
) -> np.ndarray:
    beta = fit_ols(matrix, outcome, ridge=max(ridge, 1e-8))
    penalty = np.eye(matrix.shape[1], dtype=float) * ridge
    penalty[0, 0] = ridge * 0.1
    for _ in range(max_iterations):
        residual = outcome - matrix @ beta
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        scale = max(1.4826 * mad, 1e-6)
        standardized = np.abs(residual) / (delta * scale)
        weights = np.ones_like(standardized)
        mask = standardized > 1.0
        weights[mask] = 1.0 / standardized[mask]
        lhs = matrix.T @ (weights[:, None] * matrix) + penalty
        rhs = matrix.T @ (weights * outcome)
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.pinv(lhs) @ rhs
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    return beta


def fit_flex_spec(
    frame: pd.DataFrame,
    raw_outcome: np.ndarray,
    *,
    target: str,
    spec: FlexSpec,
) -> tuple[np.ndarray, list[str]]:
    matrix, names = flex_matrix(frame, spec.candidate)
    if target == "exact":
        coefficients = fit_logistic(
            matrix, raw_outcome, ridge=spec.ridge
        )
    else:
        transformed = transform_outcome(raw_outcome, spec.transform)
        if spec.estimator == "huber":
            coefficients = fit_huber(
                matrix, transformed, ridge=spec.ridge
            )
        else:
            coefficients = fit_ols(
                matrix, transformed, ridge=spec.ridge
            )
    return coefficients, names


def predict_flex_spec(
    frame: pd.DataFrame,
    coefficients: np.ndarray,
    *,
    target: str,
    spec: FlexSpec,
) -> np.ndarray:
    matrix, _ = flex_matrix(frame, spec.candidate)
    linear = matrix @ coefficients
    if target == "exact":
        return np.clip(sigmoid(linear), EPS, 1.0 - EPS)
    return inverse_outcome(linear, spec.transform, target)


def raw_loss_vector(
    observed: np.ndarray, predicted: np.ndarray, target: str
) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if target == "exact":
        predicted = np.clip(predicted, EPS, 1.0 - EPS)
        return -(
            observed * np.log(predicted)
            + (1.0 - observed) * np.log(1.0 - predicted)
        )
    return np.abs(observed - predicted)


def baseline_prediction(train_outcome: np.ndarray, target: str) -> float:
    if target == "exact":
        return (float(train_outcome.sum()) + 0.5) / (
            len(train_outcome) + 1.0
        )
    return float(np.median(train_outcome))


def target_frame_and_outcome(
    stratum: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if target == "exact":
        frame = stratum.copy()
        outcome = frame["exact_correct"].to_numpy(float)
    elif target == "bias":
        frame = stratum[
            stratum["parse_success"].eq(1)
            & stratum["signed_error"].notna()
        ].copy()
        outcome = frame["signed_error"].to_numpy(float)
    elif target == "absolute_error":
        frame = stratum[
            stratum["parse_success"].eq(1)
            & stratum["absolute_error"].notna()
        ].copy()
        outcome = frame["absolute_error"].to_numpy(float)
    else:  # pragma: no cover - registry guard
        raise KeyError(target)
    return frame.reset_index(drop=True), outcome


def evaluate_flex_cv(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    target: str,
    spec: FlexSpec,
    fold_column: str = "seed",
) -> tuple[float, float, np.ndarray, list[float]]:
    levels = sorted(frame[fold_column].unique())
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_losses: list[float] = []
    for level in levels:
        test = frame[fold_column].to_numpy() == level
        train = ~test
        coefficients, _ = fit_flex_spec(
            frame.loc[train],
            outcome[train],
            target=target,
            spec=spec,
        )
        oof[test] = predict_flex_spec(
            frame.loc[test],
            coefficients,
            target=target,
            spec=spec,
        )
        fold_losses.append(
            float(np.mean(raw_loss_vector(outcome[test], oof[test], target)))
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"Incomplete flexible OOF predictions: {spec.name}")
    pooled = float(np.mean(raw_loss_vector(outcome, oof, target)))
    fold_se = (
        float(np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses)))
        if len(fold_losses) > 1
        else 0.0
    )
    return pooled, fold_se, oof, fold_losses


def choose_one_se(
    rows: list[dict[str, Any]], spec_lookup: dict[str, FlexSpec]
) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["loss"]),
            spec_lookup[str(row["spec"])].complexity,
            spec_lookup[str(row["spec"])].preference_rank,
            str(row["spec"]),
        ),
    )
    best = ordered[0]
    cutoff = float(best["loss"]) + float(best["fold_se"])
    eligible = [row for row in rows if float(row["loss"]) <= cutoff]
    selected = sorted(
        eligible,
        key=lambda row: (
            spec_lookup[str(row["spec"])].complexity,
            spec_lookup[str(row["spec"])].preference_rank,
            float(row["loss"]),
            str(row["spec"]),
        ),
    )[0]
    return str(selected["spec"])


def choose_best(
    rows: list[dict[str, Any]], spec_lookup: dict[str, FlexSpec]
) -> str:
    """Choose the lowest-loss candidate, with deterministic tie-breaking."""
    selected = sorted(
        rows,
        key=lambda row: (
            float(row["loss"]),
            spec_lookup[str(row["spec"])].complexity,
            spec_lookup[str(row["spec"])].preference_rank,
            str(row["spec"]),
        ),
    )[0]
    return str(selected["spec"])


def nested_seed_predictions(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    target: str,
    specs: list[FlexSpec],
    selection_rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selection_rule not in {"one_se", "best"}:
        raise ValueError(f"Unknown selection rule: {selection_rule}")
    spec_lookup = {spec.name: spec for spec in specs}
    prediction_rows: list[pd.DataFrame] = []
    choice_rows: list[dict[str, Any]] = []
    for outer_seed in sorted(frame["seed"].unique()):
        test = frame["seed"].to_numpy() == outer_seed
        train = ~test
        inner = frame.loc[train].reset_index(drop=True)
        inner_outcome = outcome[train]
        score_rows: list[dict[str, Any]] = []
        for spec in specs:
            loss, fold_se, _, _ = evaluate_flex_cv(
                inner,
                inner_outcome,
                target=target,
                spec=spec,
                fold_column="seed",
            )
            score_rows.append(
                {"spec": spec.name, "loss": loss, "fold_se": fold_se}
            )
        selected_name = (
            choose_one_se(score_rows, spec_lookup)
            if selection_rule == "one_se"
            else choose_best(score_rows, spec_lookup)
        )
        selected = spec_lookup[selected_name]
        coefficients, _ = fit_flex_spec(
            frame.loc[train],
            outcome[train],
            target=target,
            spec=selected,
        )
        prediction = predict_flex_spec(
            frame.loc[test],
            coefficients,
            target=target,
            spec=selected,
        )
        baseline = baseline_prediction(outcome[train], target)
        fold_frame = frame.loc[
            test,
            ["request_id", "stimulus_id", "seed", "T", "N", "cell_id"],
        ].copy()
        fold_frame["observed"] = outcome[test]
        fold_frame["nested_prediction"] = prediction
        fold_frame["nested_baseline"] = baseline
        fold_frame["outer_seed"] = outer_seed
        fold_frame["selection_rule"] = selection_rule
        fold_frame["outer_selected_spec"] = selected_name
        prediction_rows.append(fold_frame)
        selected_score = next(
            row for row in score_rows if row["spec"] == selected_name
        )
        best_score = min(float(row["loss"]) for row in score_rows)
        choice_rows.append(
            {
                "outer_seed": int(outer_seed),
                "selection_rule": selection_rule,
                "selected_spec": selected_name,
                "selected_label": selected.label,
                "inner_loss": float(selected_score["loss"]),
                "inner_fold_se": float(selected_score["fold_se"]),
                "best_inner_loss": best_score,
            }
        )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(
        choice_rows
    )


def fixed_grouped_predictions(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    target: str,
    spec: FlexSpec,
    fold_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(frame), np.nan, dtype=float)
    baselines = np.full(len(frame), np.nan, dtype=float)
    for level in sorted(frame[fold_column].unique()):
        test = frame[fold_column].to_numpy() == level
        train = ~test
        coefficients, _ = fit_flex_spec(
            frame.loc[train],
            outcome[train],
            target=target,
            spec=spec,
        )
        predictions[test] = predict_flex_spec(
            frame.loc[test],
            coefficients,
            target=target,
            spec=spec,
        )
        baselines[test] = baseline_prediction(outcome[train], target)
    return predictions, baselines


def expected_calibration_error(
    observed: np.ndarray, predicted: np.ndarray, bins: int = 6
) -> float:
    order = np.argsort(predicted)
    groups = np.array_split(order, bins)
    total = len(observed)
    value = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        value += (
            len(group)
            / total
            * abs(float(np.mean(observed[group]) - np.mean(predicted[group])))
        )
    return float(value)


def calibration_line(
    observed: np.ndarray, predicted: np.ndarray, target: str
) -> tuple[float, float]:
    if target == "exact":
        score = np.log(
            np.clip(predicted, EPS, 1.0 - EPS)
            / np.clip(1.0 - predicted, EPS, 1.0 - EPS)
        )
        matrix = np.column_stack([np.ones(len(score)), score])
        coefficients = fit_logistic(
            matrix, observed, ridge=1e-3
        )
        return float(coefficients[0]), float(coefficients[1])
    matrix = np.column_stack([np.ones(len(predicted)), predicted])
    coefficients = fit_ols(matrix, observed, ridge=1e-8)
    return float(coefficients[0]), float(coefficients[1])


def cell_r2_from_predictions(predictions: pd.DataFrame) -> float:
    cells = (
        predictions.groupby(["T", "N"], as_index=False)
        .agg(
            observed=("observed", "mean"),
            predicted=("nested_prediction", "mean"),
        )
    )
    return r2_score(
        cells["observed"].to_numpy(float),
        cells["predicted"].to_numpy(float),
    )


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if (
        len(observed) < 3
        or float(np.nanstd(observed)) <= EPS
        or float(np.nanstd(predicted)) <= EPS
    ):
        return np.nan
    return float(stats.spearmanr(observed, predicted).statistic)


def clustered_gain_interval(
    predictions: pd.DataFrame,
    *,
    target: str,
    rng: np.random.Generator,
    replicates: int = 2000,
) -> tuple[float, float, float, float]:
    observed = predictions["observed"].to_numpy(float)
    model_loss = raw_loss_vector(
        observed, predictions["nested_prediction"].to_numpy(float), target
    )
    baseline_loss = raw_loss_vector(
        observed, predictions["nested_baseline"].to_numpy(float), target
    )
    values = pd.DataFrame(
        {
            "cell_id": predictions["cell_id"].to_numpy(),
            "gain": baseline_loss - model_loss,
        }
    ).groupby("cell_id", as_index=False)["gain"].mean()
    cell_gain = values["gain"].to_numpy(float)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, len(cell_gain), size=len(cell_gain))
        draws[index] = float(np.mean(cell_gain[chosen]))
    estimate = float(np.mean(cell_gain))
    low, high = np.percentile(draws, [2.5, 97.5])
    one_sided = (1.0 + float(np.sum(draws <= 0.0))) / (replicates + 1.0)
    return estimate, float(low), float(high), min(1.0, one_sided)


def nested_goodness(
    nested: pd.DataFrame,
    *,
    target: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    observed = nested["observed"].to_numpy(float)
    predicted = nested["nested_prediction"].to_numpy(float)
    baseline = nested["nested_baseline"].to_numpy(float)
    loss = float(np.mean(raw_loss_vector(observed, predicted, target)))
    baseline_loss = float(
        np.mean(raw_loss_vector(observed, baseline, target))
    )
    gain_pct = 100.0 * (baseline_loss - loss) / max(baseline_loss, EPS)
    calibration_intercept, calibration_slope = calibration_line(
        observed, predicted, target
    )
    gain, gain_low, gain_high, gain_p = clustered_gain_interval(
        nested, target=target, rng=rng
    )
    if target == "exact":
        brier = float(np.mean((observed - predicted) ** 2))
        ece = expected_calibration_error(observed, predicted)
        rmse = float(np.sqrt(brier))
    else:
        residual = observed - predicted
        rmse = float(np.sqrt(np.mean(residual**2)))
        brier = np.nan
        ece = np.nan
    return {
        "oof_loss": loss,
        "baseline_loss": baseline_loss,
        "gain_pct": gain_pct,
        "rmse": rmse,
        "brier": brier,
        "ece": ece,
        "request_r2": r2_score(observed, predicted),
        "cell_r2": cell_r2_from_predictions(nested),
        "spearman": safe_spearman(observed, predicted),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "cell_cluster_gain": gain,
        "cell_cluster_gain_ci95_low": gain_low,
        "cell_cluster_gain_ci95_high": gain_high,
        "cell_cluster_gain_p": gain_p,
    }


def flex_spec_formula(target: str, spec: FlexSpec) -> str:
    left = (
        "logit p"
        if target == "exact"
        else "E[asinh(b)|parsed]"
        if target == "bias" and spec.transform == "asinh"
        else "E[b|parsed]"
        if target == "bias"
        else "E[log(1+|b|)|parsed]"
        if spec.transform == "log1p"
        else "E[sqrt(|b|)|parsed]"
        if spec.transform == "sqrt"
        else "E[|b||parsed]"
    )
    return f"{left} = {FLEX_TERMS[spec.candidate]}"


def bootstrap_selected_coefficients(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    *,
    target: str,
    spec: FlexSpec,
    rng: np.random.Generator,
    replicates: int = 500,
) -> tuple[pd.DataFrame, list[str]]:
    _, names = flex_matrix(frame, spec.candidate)
    cell_values = frame["cell_id"].to_numpy()
    clusters = [
        np.flatnonzero(cell_values == cell)
        for cell in sorted(frame["cell_id"].unique())
    ]
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        index = np.concatenate([clusters[position] for position in chosen])
        coefficients, fit_names = fit_flex_spec(
            frame.iloc[index].reset_index(drop=True),
            outcome[index],
            target=target,
            spec=spec,
        )
        if fit_names != names:
            raise RuntimeError("Flexible bootstrap name mismatch")
        for name, value in zip(names, coefficients):
            rows.append(
                {
                    "replicate": replicate,
                    "coefficient": name,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows), names


def analyze_mode_specific(
    subset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    registry_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    choice_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[pd.DataFrame] = []
    goodness_rows: list[dict[str, Any]] = []

    for target in ("exact", "bias", "absolute_error"):
        for spec in build_flex_specs(target):
            candidate = FLEX_CANDIDATE_BY_NAME[spec.candidate]
            registry_rows.append(
                {
                    "target": target,
                    "spec": spec.name,
                    "label": spec.label,
                    "candidate": spec.candidate,
                    "features": json.dumps(candidate.features),
                    "estimator": spec.estimator,
                    "transform": spec.transform,
                    "ridge": spec.ridge,
                    "complexity": spec.complexity,
                    "interpretation": candidate.interpretation,
                }
            )

    for model_index, model in enumerate(MODELS):
        for mode_index, mode in enumerate(MODES):
            stratum = subset[
                subset["model_label"].eq(model)
                & subset["prompt_mode"].eq(mode)
            ].copy()
            for target_index, target in enumerate(
                ("exact", "bias", "absolute_error")
            ):
                frame, outcome = target_frame_and_outcome(stratum, target)
                specs = build_flex_specs(target)
                spec_lookup = {spec.name: spec for spec in specs}
                score_rows: list[dict[str, Any]] = []
                spec_oof: dict[str, np.ndarray] = {}
                for spec in specs:
                    loss, fold_se, oof, fold_losses = evaluate_flex_cv(
                        frame,
                        outcome,
                        target=target,
                        spec=spec,
                        fold_column="seed",
                    )
                    row = {
                        "target": target,
                        "model_label": model,
                        "prompt_mode": mode,
                        "spec": spec.name,
                        "label": spec.label,
                        "candidate": spec.candidate,
                        "estimator": spec.estimator,
                        "transform": spec.transform,
                        "complexity": spec.complexity,
                        "n": len(frame),
                        "seed_oof_loss": loss,
                        "seed_fold_se": fold_se,
                        "seed_fold_losses": json.dumps(fold_losses),
                    }
                    score_rows.append(
                        {
                            "spec": spec.name,
                            "loss": loss,
                            "fold_se": fold_se,
                        }
                    )
                    comparison_rows.append(row)
                    spec_oof[spec.name] = oof
                selected_name = choose_one_se(score_rows, spec_lookup)
                best_name = choose_best(score_rows, spec_lookup)
                selected_spec = spec_lookup[selected_name]
                best_spec = spec_lookup[best_name]
                coefficients, coefficient_names = fit_flex_spec(
                    frame, outcome, target=target, spec=selected_spec
                )
                best_coefficients, best_coefficient_names = fit_flex_spec(
                    frame, outcome, target=target, spec=best_spec
                )

                nested_parsimonious, choices_parsimonious = (
                    nested_seed_predictions(
                        frame,
                        outcome,
                        target=target,
                        specs=specs,
                        selection_rule="one_se",
                    )
                )
                nested_best, choices_best = nested_seed_predictions(
                    frame,
                    outcome,
                    target=target,
                    specs=specs,
                    selection_rule="best",
                )
                for nested in (nested_parsimonious, nested_best):
                    nested.insert(0, "target", target)
                    nested.insert(1, "model_label", model)
                    nested.insert(2, "prompt_mode", mode)
                    prediction_rows.append(nested)
                for choices in (choices_parsimonious, choices_best):
                    choices.insert(0, "target", target)
                    choices.insert(1, "model_label", model)
                    choices.insert(2, "prompt_mode", mode)
                    choice_rows.append(choices)

                parsimonious_goodness = nested_goodness(
                    nested_parsimonious,
                    target=target,
                    rng=np.random.default_rng(
                        RNG_SEED
                        + model_index * 100
                        + mode_index * 10
                        + target_index
                    ),
                )
                best_goodness = nested_goodness(
                    nested_best,
                    target=target,
                    rng=np.random.default_rng(
                        RNG_SEED
                        + 10000
                        + model_index * 100
                        + mode_index * 10
                        + target_index
                    ),
                )

                grouped_metrics: dict[str, float] = {}
                for rule_name, spec in (
                    ("parsimonious", selected_spec),
                    ("best", best_spec),
                ):
                    needle_prediction, needle_baseline = (
                        fixed_grouped_predictions(
                            frame,
                            outcome,
                            target=target,
                            spec=spec,
                            fold_column="N",
                        )
                    )
                    length_prediction, length_baseline = (
                        fixed_grouped_predictions(
                            frame,
                            outcome,
                            target=target,
                            spec=spec,
                            fold_column="T",
                        )
                    )
                    needle_loss = float(
                        np.mean(
                            raw_loss_vector(
                                outcome, needle_prediction, target
                            )
                        )
                    )
                    needle_null = float(
                        np.mean(
                            raw_loss_vector(
                                outcome, needle_baseline, target
                            )
                        )
                    )
                    length_loss = float(
                        np.mean(
                            raw_loss_vector(
                                outcome, length_prediction, target
                            )
                        )
                    )
                    length_null = float(
                        np.mean(
                            raw_loss_vector(
                                outcome, length_baseline, target
                            )
                        )
                    )
                    grouped_metrics.update(
                        {
                            f"{rule_name}_leave_n_loss": needle_loss,
                            f"{rule_name}_leave_n_baseline_loss": needle_null,
                            f"{rule_name}_leave_n_gain_pct": 100.0
                            * (needle_null - needle_loss)
                            / max(needle_null, EPS),
                            f"{rule_name}_leave_t_loss": length_loss,
                            f"{rule_name}_leave_t_baseline_loss": length_null,
                            f"{rule_name}_leave_t_gain_pct": 100.0
                            * (length_null - length_loss)
                            / max(length_null, EPS),
                        }
                    )

                stability = float(
                    choices_parsimonious["selected_spec"]
                    .eq(selected_name)
                    .mean()
                )
                best_stability = float(
                    choices_best["selected_spec"].eq(best_name).mean()
                )
                selected_rows.append(
                    {
                        "target": target,
                        "model_label": model,
                        "prompt_mode": mode,
                        "n": len(frame),
                        "selected_spec": selected_name,
                        "selected_label": selected_spec.label,
                        "candidate": selected_spec.candidate,
                        "estimator": selected_spec.estimator,
                        "transform": selected_spec.transform,
                        "ridge": selected_spec.ridge,
                        "complexity": selected_spec.complexity,
                        "formula": flex_spec_formula(
                            target, selected_spec
                        ),
                        "coefficient_names": json.dumps(
                            coefficient_names
                        ),
                        "coefficients": json.dumps(
                            [float(value) for value in coefficients]
                        ),
                        "outer_choice_stability": stability,
                        "best_spec": best_name,
                        "best_label": best_spec.label,
                        "best_candidate": best_spec.candidate,
                        "best_estimator": best_spec.estimator,
                        "best_transform": best_spec.transform,
                        "best_ridge": best_spec.ridge,
                        "best_complexity": best_spec.complexity,
                        "best_formula": flex_spec_formula(
                            target, best_spec
                        ),
                        "best_coefficient_names": json.dumps(
                            best_coefficient_names
                        ),
                        "best_coefficients": json.dumps(
                            [
                                float(value)
                                for value in best_coefficients
                            ]
                        ),
                        "best_outer_choice_stability": best_stability,
                    }
                )
                goodness_row: dict[str, Any] = {
                    "target": target,
                    "model_label": model,
                    "prompt_mode": mode,
                    "n": len(frame),
                    "event_count": (
                        int(outcome.sum()) if target == "exact" else np.nan
                    ),
                    "failure_count": (
                        int(len(outcome) - outcome.sum())
                        if target == "exact"
                        else np.nan
                    ),
                    "selected_spec": selected_name,
                    "selected_label": selected_spec.label,
                    "best_spec": best_name,
                    "best_label": best_spec.label,
                    "outer_choice_stability": stability,
                    "best_outer_choice_stability": best_stability,
                }
                goodness_row.update(
                    {
                        f"parsimonious_{key}": value
                        for key, value in parsimonious_goodness.items()
                    }
                )
                goodness_row.update(
                    {
                        f"best_{key}": value
                        for key, value in best_goodness.items()
                    }
                )
                goodness_row.update(grouped_metrics)
                goodness_rows.append(
                    goodness_row
                )

                for rule_index, (rule_name, spec) in enumerate(
                    (
                        ("one_se", selected_spec),
                        ("best", best_spec),
                    )
                ):
                    bootstrap, _ = bootstrap_selected_coefficients(
                        frame,
                        outcome,
                        target=target,
                        spec=spec,
                        rng=np.random.default_rng(
                            RNG_SEED
                            + 1000
                            + rule_index * 10000
                            + model_index * 100
                            + mode_index * 10
                            + target_index
                        ),
                    )
                    bootstrap.insert(0, "target", target)
                    bootstrap.insert(1, "model_label", model)
                    bootstrap.insert(2, "prompt_mode", mode)
                    bootstrap.insert(3, "selection_rule", rule_name)
                    bootstrap.insert(4, "selected_spec", spec.name)
                    bootstrap_rows.append(bootstrap)

    goodness = pd.DataFrame(goodness_rows)
    for rule in ("parsimonious", "best"):
        q_column = f"{rule}_cell_cluster_gain_q"
        p_column = f"{rule}_cell_cluster_gain_p"
        goodness[q_column] = np.nan
        for target in goodness["target"].unique():
            index = goodness.index[goodness["target"].eq(target)]
            goodness.loc[index, q_column] = bh_adjust(
                goodness.loc[index, p_column].to_numpy(float)
            )

        evidence: list[str] = []
        for row in goodness.to_dict("records"):
            if row["target"] == "exact" and int(row["failure_count"]) < 10:
                label = "ceiling_limited"
            elif (
                float(row[q_column]) < 0.05
                and float(row[f"{rule}_gain_pct"]) >= 10.0
                and float(row[f"{rule}_cell_r2"]) >= 0.40
            ):
                label = (
                    "strong_generalizing"
                    if float(row[f"{rule}_leave_n_gain_pct"]) > 0.0
                    and float(row[f"{rule}_leave_t_gain_pct"]) > 0.0
                    else "strong_within_grid"
                )
            elif (
                float(row[q_column]) < 0.10
                and float(row[f"{rule}_gain_pct"]) >= 5.0
                and float(row[f"{rule}_cell_r2"]) >= 0.20
            ):
                label = "moderate"
            elif (
                float(row[f"{rule}_gain_pct"]) > 0.0
                and float(row[f"{rule}_cell_cluster_gain_ci95_high"]) > 0.0
            ):
                label = "weak"
            else:
                label = "not_supported"
            evidence.append(label)
        goodness[f"{rule}_evidence"] = evidence

    # Stable aliases keep downstream readers explicit about which pipeline is
    # the interpretability-first sensitivity analysis.
    alias_map = {
        "nested_oof_loss": "parsimonious_oof_loss",
        "nested_baseline_loss": "parsimonious_baseline_loss",
        "nested_gain_pct": "parsimonious_gain_pct",
        "nested_rmse": "parsimonious_rmse",
        "nested_brier": "parsimonious_brier",
        "nested_ece": "parsimonious_ece",
        "nested_request_r2": "parsimonious_request_r2",
        "nested_cell_r2": "parsimonious_cell_r2",
        "nested_spearman": "parsimonious_spearman",
        "calibration_intercept": "parsimonious_calibration_intercept",
        "calibration_slope": "parsimonious_calibration_slope",
        "leave_n_loss": "parsimonious_leave_n_loss",
        "leave_n_baseline_loss": "parsimonious_leave_n_baseline_loss",
        "leave_n_gain_pct": "parsimonious_leave_n_gain_pct",
        "leave_t_loss": "parsimonious_leave_t_loss",
        "leave_t_baseline_loss": "parsimonious_leave_t_baseline_loss",
        "leave_t_gain_pct": "parsimonious_leave_t_gain_pct",
        "cell_cluster_gain": "parsimonious_cell_cluster_gain",
        "cell_cluster_gain_ci95_low": (
            "parsimonious_cell_cluster_gain_ci95_low"
        ),
        "cell_cluster_gain_ci95_high": (
            "parsimonious_cell_cluster_gain_ci95_high"
        ),
        "cell_cluster_gain_p": "parsimonious_cell_cluster_gain_p",
        "cell_cluster_gain_q": "parsimonious_cell_cluster_gain_q",
        "evidence": "parsimonious_evidence",
    }
    for alias, source in alias_map.items():
        goodness[alias] = goodness[source]
    return (
        pd.DataFrame(registry_rows),
        pd.DataFrame(comparison_rows),
        pd.DataFrame(selected_rows),
        pd.concat(choice_rows, ignore_index=True),
        pd.concat(prediction_rows, ignore_index=True),
        pd.concat(bootstrap_rows, ignore_index=True),
        goodness,
    )


def load_prompt_settings(report_root: Path) -> pd.DataFrame:
    source = report_root / "tables" / "model_prompt_format_examples.csv"
    prompts = pd.read_csv(source)
    subset = prompts[
        prompts["model_label"].isin(MODELS)
        & prompts["query_order"].eq("query_last")
    ].copy()
    if len(subset) != 9:
        raise ValueError("Expected nine Qwen query-last prompt-format rows")
    subset["model_label"] = pd.Categorical(
        subset["model_label"], categories=MODELS, ordered=True
    )
    subset["prompt_mode"] = pd.Categorical(
        subset["prompt_mode"], categories=MODES, ordered=True
    )
    subset = subset.sort_values(["model_label", "prompt_mode"]).reset_index(
        drop=True
    )
    keep = [
        "model_label",
        "model_id",
        "prompt_mode",
        "thinking_enabled",
        "query_order",
        "request_count",
        "message_roles",
        "tokenize",
        "add_generation_prompt",
        "enable_thinking_argument",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "sample_request_id",
        "sample_full_rendered_prompt_sha256",
        "user_message_redacted",
        "rendered_prompt_redacted",
    ]
    return subset[keep].copy()


def make_accuracy_heatmaps(cells: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(
        len(MODELS),
        len(MODES),
        figsize=(16.2, 11.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row_index, model in enumerate(MODELS):
        for column_index, mode in enumerate(MODES):
            ax = axes[row_index, column_index]
            part = cells[
                cells["model_label"].eq(model)
                & cells["prompt_mode"].eq(mode)
            ].pivot(index="T", columns="N", values="exact_accuracy")
            part = part.reindex(index=TARGET_LENGTHS, columns=NEEDLE_COUNTS)
            matrix = part.to_numpy(float)
            image = ax.imshow(
                matrix,
                vmin=0,
                vmax=1,
                cmap="viridis",
                aspect="auto",
                interpolation="nearest",
            )
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    value = matrix[y, x]
                    color = "white" if value < 0.55 else "#162229"
                    ax.text(
                        x,
                        y,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        color=color,
                    )
            if row_index == 0:
                ax.set_title(MODE_LABEL[mode], fontsize=12, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{model}\nT (tokens)", fontsize=10)
            if row_index == len(MODELS) - 1:
                ax.set_xlabel("True needle count N")
            ax.set_xticks(range(len(NEEDLE_COUNTS)))
            ax.set_xticklabels(NEEDLE_COUNTS, fontsize=8)
            ax.set_yticks(range(len(TARGET_LENGTHS)))
            ax.set_yticklabels(["2k", "5k", "10k"], fontsize=9)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            shrink=0.78,
            pad=0.02,
            label="Exact accuracy across 5 seeds",
        )
    fig.suptitle(
        "Qwen query-last exact accuracy by model, mode, length, and needle count",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, path)


def make_coefficient_forest(
    fixed: pd.DataFrame,
    target: str,
    path: Path,
    *,
    title: str,
    xlabels: tuple[str, str],
) -> None:
    part = fixed[fixed["target"].eq(target)].copy()
    labels = [
        f"{model} · {MODE_LABEL[mode]}"
        for model in MODELS
        for mode in MODES
    ]
    part["stratum"] = [
        f"{row.model_label} · {MODE_LABEL[row.prompt_mode]}"
        for row in part.itertuples()
    ]
    part = part.set_index("stratum").reindex(labels)
    y = np.arange(len(labels))
    colors = {
        "supported": "#237a57",
        "suggestive": "#d17b34",
        "ceiling_limited": "#756bb1",
        "not_supported": "#6f777d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), sharey=True)
    for ax, stem, xlabel in zip(
        axes, ("length_order", "needle_order"), xlabels
    ):
        for position, (_, row) in enumerate(part.iterrows()):
            color = colors.get(str(row["evidence"]), "#6f777d")
            value = float(row[stem])
            low = float(row[f"{stem}_ci95_low"])
            high = float(row[f"{stem}_ci95_high"])
            ax.errorbar(
                value,
                position,
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=6,
            )
        ax.axvline(0, color="#30363b", lw=1, ls="--")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#d9dedf", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Intervals: 500 cell-cluster bootstrap replicates. "
        "Green=supported, orange=suggestive, purple=ceiling-limited.",
        ha="center",
        fontsize=9,
        color="#59636a",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_figure(fig, path)


def make_accuracy_calibration(
    oof: pd.DataFrame, fixed: pd.DataFrame, path: Path
) -> None:
    exact = oof[oof["target"].eq("exact")].copy()
    metrics = fixed[fixed["target"].eq("exact")].set_index(
        ["model_label", "prompt_mode"]
    )
    fig, axes = plt.subplots(
        len(MODELS),
        len(MODES),
        figsize=(13.2, 11.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row_index, model in enumerate(MODELS):
        for column_index, mode in enumerate(MODES):
            ax = axes[row_index, column_index]
            part = exact[
                exact["model_label"].eq(model)
                & exact["prompt_mode"].eq(mode)
            ]
            cells = (
                part.groupby(["T", "N"], as_index=False)
                .agg(
                    observed=("observed", "mean"),
                    predicted=("oof_prediction", "mean"),
                )
            )
            ax.scatter(
                cells["predicted"],
                cells["observed"],
                c=np.log10(cells["N"]),
                cmap="plasma",
                s=31,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.45,
            )
            ax.plot([0, 1], [0, 1], color="#667078", ls="--", lw=1)
            row = metrics.loc[(model, mode)]
            ax.text(
                0.04,
                0.94,
                f"OOF cell R²={decimal(row['seed_oof_cell_r2'], 2)}\n"
                f"log loss={decimal(row['seed_log_loss'])}",
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
            )
            if row_index == 0:
                ax.set_title(MODE_LABEL[mode], fontsize=11, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{model}\nObserved cell accuracy")
            if row_index == len(MODELS) - 1:
                ax.set_xlabel("Leave-one-seed-out predicted probability")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(color="#e4e8e9", lw=0.6)
    fig.suptitle(
        "Held-out calibration of the fixed log-separable accuracy law",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, path)


def make_bias_surfaces(fixed: pd.DataFrame, path: Path) -> None:
    part = fixed[fixed["target"].eq("bias")].set_index(
        ["model_label", "prompt_mode"]
    )
    n_grid = np.geomspace(1, 30, 120)
    length_colors = {2000: "#3d6fa6", 5000: "#2f7d67", 10000: "#bc653f"}
    fig, axes = plt.subplots(
        len(MODELS),
        len(MODES),
        figsize=(13.6, 10.6),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, model in enumerate(MODELS):
        for column_index, mode in enumerate(MODES):
            ax = axes[row_index, column_index]
            row = part.loc[(model, mode)]
            for length in TARGET_LENGTHS:
                linear = (
                    float(row["intercept"])
                    + float(row["length_order"]) * np.log(length / 5000.0)
                    + float(row["needle_order"]) * np.log(n_grid / 5.0)
                )
                centered_bias = np.sinh(np.clip(linear, -7, 7))
                ax.plot(
                    n_grid,
                    centered_bias,
                    color=length_colors[length],
                    lw=2,
                    label=f"T={length // 1000}k",
                )
            ax.axhline(0, color="#30363b", ls="--", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("symlog", linthresh=0.5)
            ax.set_xticks([1, 2, 4, 8, 16, 30])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.grid(color="#e4e8e9", lw=0.6)
            if row_index == 0:
                ax.set_title(MODE_LABEL[mode], fontsize=11, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{model}\nPredicted centered bias")
            if row_index == len(MODELS) - 1:
                ax.set_xlabel("True needle count N (log₂ scale)")
            if row_index == 0 and column_index == 2:
                ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Fixed-law signed-bias surfaces (parsed numeric outputs only)",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, path)


def make_candidate_gain_heatmap(
    selected: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.6), sharey=True)
    labels = [
        f"{model} · {MODE_LABEL[mode]}"
        for model in MODELS
        for mode in MODES
    ]
    for ax, target in zip(axes, TARGETS):
        part = selected[selected["target"].eq(target)].copy()
        part["stratum"] = [
            f"{row.model_label} · {MODE_LABEL[row.prompt_mode]}"
            for row in part.itertuples()
        ]
        part = part.set_index("stratum").reindex(labels)
        values = part["seed_gain_pct"].to_numpy(float)[:, None]
        vmax = max(10.0, float(np.nanmax(np.abs(values))))
        image = ax.imshow(
            values,
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        for position, (_, row) in enumerate(part.iterrows()):
            ax.text(
                0,
                position,
                f"{row['selected_candidate_label']}\n{row['seed_gain_pct']:+.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="#172025",
            )
        ax.set_title(TARGETS[target]["label"], fontsize=11, fontweight="bold")
        ax.set_xticks([])
        if ax is axes[0]:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8.5)
        else:
            ax.set_yticks(range(len(labels)))
        fig.colorbar(image, ax=ax, shrink=0.65, label="Seed-OOF gain vs constant (%)")
    fig.suptitle(
        "One-standard-error candidate selection and held-out gain",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path)


def stratum_label(model: str, mode: str) -> str:
    return f"{model} / {MODE_LABEL[mode]}"


def make_mode_specific_accuracy_gof(
    goodness: pd.DataFrame, path: Path
) -> None:
    frame = goodness[goodness["target"].eq("exact")].copy()
    frame["stratum"] = [
        stratum_label(model, mode)
        for model, mode in zip(frame["model_label"], frame["prompt_mode"])
    ]
    frame["order"] = [
        MODELS.index(model) * len(MODES) + MODES.index(mode)
        for model, mode in zip(frame["model_label"], frame["prompt_mode"])
    ]
    frame = frame.sort_values("order")
    labels = frame["stratum"].tolist()
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    width = 0.35
    for axis, metric, title, xlabel in (
        (
            axes[0],
            "gain_pct",
            "Nested held-out improvement",
            "Log-loss reduction vs fold-trained constant (%)",
        ),
        (
            axes[1],
            "cell_r2",
            "Cell-level predictive fit",
            "Nested OOF cell R²",
        ),
    ):
        one = frame[f"parsimonious_{metric}"].to_numpy(float)
        best = frame[f"best_{metric}"].to_numpy(float)
        axis.barh(
            y + width / 2,
            one,
            height=width,
            color="#8ab6ac",
            label="One-SE parsimonious",
        )
        axis.barh(
            y - width / 2,
            best,
            height=width,
            color="#315f8c",
            label="Best predictive",
        )
        axis.axvline(0.0, color="#56656a", linewidth=0.9)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#e3e8e8", linewidth=0.7)
        axis.set_yticks(y)
        axis.invert_yaxis()
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "Qwen query-last accuracy: post-selection goodness of fit",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path)


def make_mode_specific_accuracy_scatter(
    predictions: pd.DataFrame, goodness: pd.DataFrame, path: Path
) -> None:
    source = predictions[
        predictions["target"].eq("exact")
        & predictions["selection_rule"].eq("best")
    ]
    gof = goodness[goodness["target"].eq("exact")].set_index(
        ["model_label", "prompt_mode"]
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.6, 12.4), sharex=True, sharey=True)
    for row_index, model in enumerate(MODELS):
        for column_index, mode in enumerate(MODES):
            axis = axes[row_index, column_index]
            frame = source[
                source["model_label"].eq(model)
                & source["prompt_mode"].eq(mode)
            ]
            cells = (
                frame.groupby(["T", "N"], as_index=False)
                .agg(
                    observed=("observed", "mean"),
                    predicted=("nested_prediction", "mean"),
                )
            )
            for length, color in zip(
                TARGET_LENGTHS, ("#277da1", "#2a9d8f", "#e07a5f")
            ):
                part = cells[cells["T"].eq(length)]
                axis.scatter(
                    part["predicted"],
                    part["observed"],
                    s=33,
                    color=color,
                    alpha=0.86,
                    edgecolor="white",
                    linewidth=0.45,
                    label=f"T={length // 1000}k",
                )
            axis.plot([0, 1], [0, 1], "--", color="#6f7c80", linewidth=0.9)
            record = gof.loc[(model, mode)]
            axis.text(
                0.03,
                0.95,
                (
                    f"gain={record['best_gain_pct']:.1f}%\n"
                    f"cell R²={record['best_cell_r2']:.2f}"
                ),
                transform=axis.transAxes,
                va="top",
                fontsize=8.4,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.84,
                    "edgecolor": "#d5dddd",
                },
            )
            axis.set_title(
                f"{model} · {MODE_LABEL[mode]}", fontsize=10, fontweight="bold"
            )
            axis.grid(color="#edf0f0", linewidth=0.65)
            axis.set_xlim(-0.03, 1.03)
            axis.set_ylim(-0.03, 1.03)
            if row_index == 2:
                axis.set_xlabel("Nested held-out predicted accuracy")
            if column_index == 0:
                axis.set_ylabel("Observed held-out accuracy")
    axes[0, 2].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle(
        "Observed versus nested held-out cell accuracy",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)


def make_mode_specific_accuracy_response(
    predictions: pd.DataFrame, path: Path
) -> None:
    source = predictions[
        predictions["target"].eq("exact")
        & predictions["selection_rule"].eq("best")
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 12.2), sharex=True, sharey=True)
    colors = ("#277da1", "#2a9d8f", "#e07a5f")
    for row_index, model in enumerate(MODELS):
        for column_index, mode in enumerate(MODES):
            axis = axes[row_index, column_index]
            frame = source[
                source["model_label"].eq(model)
                & source["prompt_mode"].eq(mode)
            ]
            cells = (
                frame.groupby(["T", "N"], as_index=False)
                .agg(
                    observed=("observed", "mean"),
                    predicted=("nested_prediction", "mean"),
                )
            )
            for length, color in zip(TARGET_LENGTHS, colors):
                part = cells[cells["T"].eq(length)].sort_values("N")
                axis.plot(
                    part["N"],
                    part["predicted"],
                    color=color,
                    linewidth=1.9,
                    label=f"fit T={length // 1000}k",
                )
                axis.scatter(
                    part["N"],
                    part["observed"],
                    color=color,
                    s=24,
                    marker="o",
                    edgecolor="white",
                    linewidth=0.4,
                )
            axis.set_xscale("log")
            axis.set_xticks([1, 2, 5, 10, 20, 30])
            axis.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter()
            )
            axis.set_ylim(-0.03, 1.03)
            axis.grid(color="#edf0f0", linewidth=0.65)
            axis.set_title(
                f"{model} · {MODE_LABEL[mode]}", fontsize=10, fontweight="bold"
            )
            if row_index == 2:
                axis.set_xlabel("Needle count N (log scale)")
            if column_index == 0:
                axis.set_ylabel("Exact accuracy")
    axes[0, 2].legend(loc="lower left", frameon=False, fontsize=7.5)
    fig.suptitle(
        "Mode-specific accuracy response along N at each passage length",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)


def make_mode_specific_error_gof(
    goodness: pd.DataFrame, path: Path
) -> None:
    labels = [
        stratum_label(model, mode) for model in MODELS for mode in MODES
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0), sharey=True)
    y = np.arange(len(labels))
    for row_index, target in enumerate(("bias", "absolute_error")):
        frame = goodness[goodness["target"].eq(target)].copy()
        frame["order"] = [
            MODELS.index(model) * len(MODES) + MODES.index(mode)
            for model, mode in zip(frame["model_label"], frame["prompt_mode"])
        ]
        frame = frame.sort_values("order")
        for column_index, metric in enumerate(("gain_pct", "cell_r2")):
            axis = axes[row_index, column_index]
            axis.barh(
                y + 0.17,
                frame[f"parsimonious_{metric}"],
                height=0.34,
                color="#8ab6ac",
                label="One-SE parsimonious",
            )
            axis.barh(
                y - 0.17,
                frame[f"best_{metric}"],
                height=0.34,
                color="#315f8c",
                label="Best predictive",
            )
            axis.axvline(0.0, color="#56656a", linewidth=0.9)
            axis.grid(axis="x", color="#e3e8e8", linewidth=0.7)
            axis.set_yticks(y)
            axis.invert_yaxis()
            if row_index == 0:
                axis.set_title(
                    "Nested MAE reduction (%)"
                    if metric == "gain_pct"
                    else "Nested OOF cell R²",
                    fontweight="bold",
                )
            axis.set_xlabel(
                "Signed bias"
                if target == "bias"
                else "Absolute count error"
            )
    axes[0, 0].set_yticklabels(labels, fontsize=8.7)
    axes[1, 0].set_yticklabels(labels, fontsize=8.7)
    axes[0, 1].tick_params(axis="y", labelleft=False)
    axes[1, 1].tick_params(axis="y", labelleft=False)
    axes[0, 1].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "Qwen query-last error laws: post-selection goodness of fit",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)


def html_table(
    headers: list[str], rows: list[list[str]], *, css_class: str = ""
) -> str:
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="{esc(css_class)}">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


EVIDENCE_LABELS = {
    "strong_generalizing": "Strong + cross-N/T",
    "strong_within_grid": "Strong in-grid; extrapolation warning",
    "moderate": "Moderate",
    "weak": "Weak",
    "ceiling_limited": "Ceiling-limited",
    "not_supported": "Not supported",
}


def clean_decimal(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "—"


def clean_signed(value: Any, digits: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:+.{digits}f}" if np.isfinite(numeric) else "—"


FEATURE_DISPLAY = {
    "log_length": "ln(T/5000)",
    "log_needles": "ln(N/5)",
    "needle_linear": "(N-5)/5",
    "log_density": "ln(1000N/T)",
    "log_burden": "ln[(T/5000)(N/5)]",
    "needle_hinge": "max[0, ln(N/5)]",
    "log_interaction": "ln(T/5000) ln(N/5)",
    "log_length_sq": "ln(T/5000)^2",
    "log_needles_sq": "ln(N/5)^2",
    "t_is_2000": "I(T=2000)",
    "t_is_10000": "I(T=10000)",
    "t2000_log_needles": "I(T=2000) ln(N/5)",
    "t10000_log_needles": "I(T=10000) ln(N/5)",
}


def formula_lhs(target: str, transform: str) -> str:
    if target == "exact":
        return "logit(p)"
    if target == "bias" and transform == "asinh":
        return "E[asinh(b) | parsed]"
    if target == "bias":
        return "E[b | parsed]"
    if transform == "log1p":
        return "E[ln(1+|b|) | parsed]"
    if transform == "sqrt":
        return "E[sqrt(|b|) | parsed]"
    return "E[|b| | parsed]"


def inline_numeric_formula(
    row: pd.Series, *, target: str, rule: str
) -> str:
    prefix = "" if rule == "one_se" else "best_"
    transform = str(row[f"{prefix}transform"])
    names = json.loads(str(row[f"{prefix}coefficient_names"]))
    values = json.loads(str(row[f"{prefix}coefficients"]))
    lhs = formula_lhs(target, transform)
    pieces = [
        f'<mtext>{esc(lhs)}</mtext><mo>=</mo><mn>{float(values[0]):.3f}</mn>'
    ]
    for name, value in zip(names[1:], values[1:]):
        sign = "+" if float(value) >= 0 else "−"
        pieces.append(
            f"<mo>{sign}</mo><mn>{abs(float(value)):.3f}</mn>"
            f"<mo>×</mo><mtext>{esc(FEATURE_DISPLAY.get(name, name))}</mtext>"
        )
    aria = (
        f"{lhs} = "
        + " ".join(
            [f"{float(values[0]):.3f}"]
            + [
                f"{float(value):+.3f}*{FEATURE_DISPLAY.get(name, name)}"
                for name, value in zip(names[1:], values[1:])
            ]
        )
    )
    return (
        f'<div class="formula-scroll"><math class="inline-formula" '
        f'aria-label="{esc(aria)}"><mrow>{"".join(pieces)}</mrow></math></div>'
    )


def mode_specific_gof_table(
    goodness: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    target: str,
    rule: str = "best",
) -> str:
    gof = goodness[goodness["target"].eq(target)].set_index(
        ["model_label", "prompt_mode"]
    )
    laws = selected[selected["target"].eq(target)].set_index(
        ["model_label", "prompt_mode"]
    )
    rows: list[list[str]] = []
    prefix = f"{rule}_"
    for model in MODELS:
        for mode in MODES:
            metric = gof.loc[(model, mode)]
            law = laws.loc[(model, mode)]
            label_column = "selected_label" if rule == "one_se" else "best_label"
            evidence = EVIDENCE_LABELS.get(
                str(metric[f"{prefix}evidence"]),
                str(metric[f"{prefix}evidence"]),
            )
            common = [
                f"<strong>{esc(model)}</strong>",
                esc(MODE_LABEL[mode]),
                str(int(metric["n"])),
                esc(law[label_column]),
                inline_numeric_formula(law, target=target, rule=rule),
                (
                    f"{clean_decimal(metric[f'{prefix}oof_loss'])} / "
                    f"{clean_decimal(metric[f'{prefix}baseline_loss'])}"
                ),
                clean_signed(metric[f"{prefix}gain_pct"], 1) + "%",
            ]
            if target == "exact":
                tail = [
                    clean_decimal(metric[f"{prefix}brier"]),
                    clean_decimal(metric[f"{prefix}ece"]),
                    clean_decimal(metric[f"{prefix}cell_r2"]),
                    clean_decimal(metric[f"{prefix}calibration_slope"], 2),
                ]
            else:
                tail = [
                    clean_decimal(metric[f"{prefix}rmse"]),
                    clean_decimal(metric[f"{prefix}cell_r2"]),
                    clean_decimal(metric[f"{prefix}spearman"], 2),
                    "—",
                ]
            rows.append(
                common
                + tail
                + [
                    clean_signed(metric[f"{prefix}leave_n_gain_pct"], 1) + "%",
                    clean_signed(metric[f"{prefix}leave_t_gain_pct"], 1) + "%",
                    clean_decimal(metric[f"{prefix}cell_cluster_gain_q"], 3),
                    esc(evidence),
                ]
            )
    return html_table(
        [
            "Model",
            "Mode",
            "n",
            "Selected form",
            "Full-data fitted equation",
            "Nested loss / null",
            "Gain",
            "Brier" if target == "exact" else "RMSE",
            "ECE" if target == "exact" else "Cell R²",
            "Cell R²" if target == "exact" else "Spearman ρ",
            "Cal. slope" if target == "exact" else "—",
            "Leave-N gain",
            "Leave-T gain",
            "FDR q",
            "Evidence",
        ],
        rows,
        css_class="dense numeric gof-table",
    )


def selection_sensitivity_table(
    goodness: pd.DataFrame, selected: pd.DataFrame, target: str
) -> str:
    gof = goodness[goodness["target"].eq(target)].set_index(
        ["model_label", "prompt_mode"]
    )
    laws = selected[selected["target"].eq(target)].set_index(
        ["model_label", "prompt_mode"]
    )
    rows: list[list[str]] = []
    for model in MODELS:
        for mode in MODES:
            metric = gof.loc[(model, mode)]
            law = laws.loc[(model, mode)]
            rows.append(
                [
                    f"<strong>{esc(model)}</strong>",
                    esc(MODE_LABEL[mode]),
                    esc(law["selected_label"]),
                    esc(law["best_label"]),
                    (
                        f"{clean_signed(metric['parsimonious_gain_pct'], 1)}% / "
                        f"{clean_signed(metric['best_gain_pct'], 1)}%"
                    ),
                    (
                        f"{clean_decimal(metric['parsimonious_cell_r2'])} / "
                        f"{clean_decimal(metric['best_cell_r2'])}"
                    ),
                    (
                        f"{clean_decimal(metric['outer_choice_stability'], 2)} / "
                        f"{clean_decimal(metric['best_outer_choice_stability'], 2)}"
                    ),
                ]
            )
    return html_table(
        [
            "Model",
            "Mode",
            "One-SE form",
            "Best-predictive form",
            "Nested gain: one-SE / best",
            "Cell R²: one-SE / best",
            "Outer-choice stability: one-SE / best",
        ],
        rows,
        css_class="dense numeric",
    )


def summarize_flex_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    return (
        bootstrap.groupby(
            [
                "target",
                "model_label",
                "prompt_mode",
                "selection_rule",
                "selected_spec",
                "coefficient",
            ],
            as_index=False,
        )
        .agg(
            bootstrap_median=("value", "median"),
            ci95_low=("value", lambda values: np.quantile(values, 0.025)),
            ci95_high=("value", lambda values: np.quantile(values, 0.975)),
            bootstrap_sd=("value", "std"),
            replicates=("replicate", "nunique"),
        )
    )


def overall_summary_table(summary: pd.DataFrame) -> str:
    frame = summary.set_index(["model_label", "prompt_mode"])
    rows: list[list[str]] = []
    for model in MODELS:
        for mode in MODES:
            row = frame.loc[(model, mode)]
            rows.append(
                [
                    f"<strong>{esc(model)}</strong>",
                    esc(MODE_LABEL[mode]),
                    f"{int(row['exact_correct'])}/150",
                    pct(row["exact_accuracy"]),
                    f"{int(row['parsed_requests'])}/150",
                    pct(row["parse_success_rate"]),
                    signed(row["mean_signed_error_parsed"], 2),
                    decimal(row["mean_absolute_error_parsed"], 2),
                ]
            )
    return html_table(
        [
            "Model",
            "Mode",
            "Exact",
            "Accuracy",
            "Parsed",
            "Parse rate",
            "Mean bias",
            "MAE",
        ],
        rows,
        css_class="numeric",
    )


def prompt_settings_table(prompts: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for row in prompts.itertuples():
        rows.append(
            [
                f"<strong>{esc(row.model_label)}</strong>",
                esc(MODE_LABEL[str(row.prompt_mode)]),
                "on" if bool(row.thinking_enabled) else "off",
                str(int(row.max_tokens)),
                decimal(row.temperature, 1),
                decimal(row.top_p, 2),
                str(int(row.top_k)),
                esc(str(row.sample_full_rendered_prompt_sha256)),
            ]
        )
    return html_table(
        [
            "Model",
            "Mode",
            "Thinking",
            "Max output",
            "Temp.",
            "top-p",
            "top-k",
            "Rendered prompt SHA256 (sample)",
        ],
        rows,
        css_class="compact hash-table",
    )


def fixed_accuracy_table(fixed: pd.DataFrame) -> str:
    part = fixed[fixed["target"].eq("exact")].set_index(
        ["model_label", "prompt_mode"]
    )
    rows: list[list[str]] = []
    evidence_label = {
        "supported": "有支持",
        "suggestive": "方向性",
        "ceiling_limited": "天花板受限",
        "not_supported": "未支持",
    }
    for model in MODELS:
        for mode in MODES:
            row = part.loc[(model, mode)]
            rows.append(
                [
                    f"<strong>{esc(model)}</strong>",
                    esc(MODE_LABEL[mode]),
                    pct(row["event_count"] / row["n"]),
                    f"{signed(row['length_order'])} "
                    f"[{signed(row['length_order_ci95_low'])}, "
                    f"{signed(row['length_order_ci95_high'])}]",
                    decimal(row["length_doubling_factor"], 2) + "×",
                    f"{signed(row['needle_order'])} "
                    f"[{signed(row['needle_order_ci95_low'])}, "
                    f"{signed(row['needle_order_ci95_high'])}]",
                    decimal(row["needle_doubling_factor"], 2) + "×",
                    decimal(row["seed_log_loss"]),
                    signed(row["seed_gain_pct"], 1) + "%",
                    signed(row["needle_gain_pct"], 1) + "%",
                    signed(row["length_gain_pct"], 1) + "%",
                    decimal(row["global_q_value"], 3),
                    esc(evidence_label[str(row["evidence"])]),
                ]
            )
    return html_table(
        [
            "Model",
            "Mode",
            "Accuracy",
            "βT [95% CI]",
            "Odds / T×2",
            "βN [95% CI]",
            "Odds / N×2",
            "Seed-OOF log loss",
            "Seed gain",
            "Held-out N gain",
            "Held-out T gain",
            "FDR q",
            "Evidence",
        ],
        rows,
        css_class="numeric dense",
    )


def continuous_fixed_table(fixed: pd.DataFrame, target: str) -> str:
    part = fixed[fixed["target"].eq(target)].set_index(
        ["model_label", "prompt_mode"]
    )
    evidence_label = {
        "supported": "有支持",
        "suggestive": "方向性",
        "ceiling_limited": "天花板受限",
        "not_supported": "未支持",
    }
    rows: list[list[str]] = []
    for model in MODELS:
        for mode in MODES:
            row = part.loc[(model, mode)]
            rows.append(
                [
                    f"<strong>{esc(model)}</strong>",
                    esc(MODE_LABEL[mode]),
                    str(int(row["n"])),
                    f"{signed(row['length_order'])} "
                    f"[{signed(row['length_order_ci95_low'])}, "
                    f"{signed(row['length_order_ci95_high'])}]",
                    f"{signed(row['needle_order'])} "
                    f"[{signed(row['needle_order_ci95_low'])}, "
                    f"{signed(row['needle_order_ci95_high'])}]",
                    decimal(row["seed_mae"]),
                    signed(row["seed_gain_pct"], 1) + "%",
                    decimal(row["global_q_value"], 3),
                    esc(evidence_label[str(row["evidence"])]),
                ]
            )
    return html_table(
        [
            "Model",
            "Mode",
            "Parsed n",
            "βT [95% CI]",
            "βN [95% CI]",
            "Seed-OOF MAE",
            "Gain vs constant",
            "FDR q",
            "Evidence",
        ],
        rows,
        css_class="numeric dense",
    )


def selected_table(selected: pd.DataFrame, target: str) -> str:
    part = selected[selected["target"].eq(target)].set_index(
        ["model_label", "prompt_mode"]
    )
    rows: list[list[str]] = []
    metric = "seed_oof_log_loss" if target == "exact" else "seed_oof_mae"
    for model in MODELS:
        for mode in MODES:
            row = part.loc[(model, mode)]
            rows.append(
                [
                    f"<strong>{esc(model)}</strong>",
                    esc(MODE_LABEL[mode]),
                    esc(row["selected_candidate_label"]),
                    f"<code>{esc(row['formula'])}</code>",
                    decimal(row[metric]),
                    signed(row["seed_gain_pct"], 1) + "%",
                ]
            )
    return html_table(
        [
            "Model",
            "Mode",
            "One-SE selected form",
            "Formula",
            "Seed-OOF loss",
            "Gain vs constant",
        ],
        rows,
        css_class="dense",
    )


def support_summary(fixed: pd.DataFrame, target: str) -> str:
    part = fixed[fixed["target"].eq(target)]
    chunks = []
    labels = [
        ("supported", "有支持"),
        ("suggestive", "方向性"),
        ("ceiling_limited", "天花板受限"),
        ("not_supported", "未支持"),
    ]
    for key, label in labels:
        names = [
            f"{row.model_label}/{MODE_LABEL[row.prompt_mode]}"
            for row in part[part["evidence"].eq(key)].itertuples()
        ]
        if names:
            chunks.append(f"<strong>{label}</strong>：{esc('；'.join(names))}")
    return "；".join(chunks) + "。"


def empirical_headlines(
    summary: pd.DataFrame, fixed: pd.DataFrame, selected: pd.DataFrame
) -> list[str]:
    exact = fixed[fixed["target"].eq("exact")].set_index(
        ["model_label", "prompt_mode"]
    )
    result: list[str] = []
    for model in MODELS:
        direct = exact.loc[(model, "direct")]
        enum = exact.loc[(model, "enumeration")]
        thinking = exact.loc[(model, "native_thinking")]
        if model == "Qwen3-1.7B":
            result.append(
                "Qwen3-1.7B 的主要规律来自 N：direct 与 enumeration "
                f"的 βN 分别为 {direct['needle_order']:+.2f} 和 "
                f"{enum['needle_order']:+.2f}；native thinking 的平均准确率更高，"
                "但非单调 cell 波动仍使简单 law 的 held-out 解释力有限。"
            )
        elif model == "Qwen3-8B":
            result.append(
                "Qwen3-8B direct 随 N 增大明显恶化；enumeration 与 native "
                "thinking 已接近天花板，因此当前样本只能给出“误差事件很少”的结论，"
                "不能稳定估计细小斜率。"
            )
        else:
            result.append(
                "Qwen3-32B direct 仍表现出随 N 增大的准确率下降；enumeration "
                "和 native thinking 分别只有极少数错误，属于 ceiling-limited，"
                "高准确率不等于已证明 βT=βN=0。"
            )
    return result


def math_exact_law() -> str:
    return """
    <div class="equation-card">
      <div class="equation-title">固定主模型：准确率 odds 的分离幂律</div>
      <div class="math-scroll">
        <math display="block" aria-label="Exact accuracy log-separable law">
          <mrow>
            <mi>logit</mi><mo>(</mo><msub><mi>p</mi><mrow><mi>m</mi><mo>,</mo><mi>q</mi></mrow></msub><mo>)</mo>
            <mo>=</mo><msub><mi>α</mi><mrow><mi>m</mi><mo>,</mo><mi>q</mi></mrow></msub>
            <mo>+</mo><msub><mi>β</mi><mrow><mi>T</mi><mo>,</mo><mi>m</mi><mo>,</mo><mi>q</mi></mrow></msub>
            <mi>ln</mi><mo>(</mo><mfrac><mi>T</mi><mn>5000</mn></mfrac><mo>)</mo>
            <mo>+</mo><msub><mi>β</mi><mrow><mi>N</mi><mo>,</mo><mi>m</mi><mo>,</mo><mi>q</mi></mrow></msub>
            <mi>ln</mi><mo>(</mo><mfrac><mi>N</mi><mn>5</mn></mfrac><mo>)</mo>
          </mrow>
        </math>
      </div>
      <p>其中 m 是 Qwen 模型，q 是 mode。于是 odds ∝ (T/5000)<sup>βT</sup>(N/5)<sup>βN</sup>；
      T 或 N 翻倍时 odds 分别乘以 2<sup>βT</sup> 或 2<sup>βN</sup>。</p>
    </div>
    """


def math_bias_law() -> str:
    return """
    <div class="equation-card">
      <div class="equation-title">Bias 与 absolute error 的固定坐标</div>
      <div class="math-scroll">
        <math display="block" aria-label="Bias and absolute-error laws">
          <mtable columnalign="right left" rowspacing="0.65em">
            <mtr>
              <mtd><mi>b</mi><mo>=</mo><mover><mi>N</mi><mo>^</mo></mover><mo>−</mo><mi>N</mi><mo>,</mo></mtd>
              <mtd><mi>E</mi><mo>[</mo><mi>asinh</mi><mo>(</mo><mi>b</mi><mo>)</mo><mo>|</mo><mi>parsed</mi><mo>]</mo>
              <mo>=</mo><mi>α</mi><mo>+</mo><msub><mi>β</mi><mi>T</mi></msub><mi>ln</mi><mo>(</mo><mi>T</mi><mo>/</mo><mn>5000</mn><mo>)</mo>
              <mo>+</mo><msub><mi>β</mi><mi>N</mi></msub><mi>ln</mi><mo>(</mo><mi>N</mi><mo>/</mo><mn>5</mn><mo>)</mo></mtd>
            </mtr>
            <mtr>
              <mtd><mi>a</mi><mo>=</mo><mo>|</mo><mi>b</mi><mo>|</mo><mo>,</mo></mtd>
              <mtd><mi>E</mi><mo>[</mo><mi>ln</mi><mo>(</mo><mn>1</mn><mo>+</mo><mi>a</mi><mo>)</mo><mo>|</mo><mi>parsed</mi><mo>]</mo>
              <mo>=</mo><mi>α</mi><mo>+</mo><msub><mi>β</mi><mi>T</mi></msub><mi>ln</mi><mo>(</mo><mi>T</mi><mo>/</mo><mn>5000</mn><mo>)</mo>
              <mo>+</mo><msub><mi>β</mi><mi>N</mi></msub><mi>ln</mi><mo>(</mo><mi>N</mi><mo>/</mo><mn>5</mn><mo>)</mo></mtd>
            </mtr>
          </mtable>
        </math>
      </div>
      <p>asinh 在 0 附近近似线性、在尾部近似带符号的 log；log1p absolute error 则让指数化后的系数可以解释典型误差幅度的幂阶。</p>
    </div>
    """


def method_conclusion(method: str, conclusion: str) -> str:
    return f"""
    <div class="method-grid">
      <div class="method-box"><span>计算方法</span><p>{method}</p></div>
      <div class="conclusion-box"><span>目前结论</span><p>{conclusion}</p></div>
    </div>
    """


def build_html(
    *,
    summary: pd.DataFrame,
    cells: pd.DataFrame,
    fixed: pd.DataFrame,
    selected: pd.DataFrame,
    prompts: pd.DataFrame,
    source_report_root: Path,
    source_report_href: str,
    generated_at: str,
) -> str:
    headlines = empirical_headlines(summary, fixed, selected)
    style = r"""
    :root {
      --ink: #172126;
      --muted: #606c73;
      --line: #dce2e4;
      --paper: #ffffff;
      --soft: #f5f7f7;
      --teal: #1e6a5d;
      --teal-soft: #e8f3f0;
      --blue: #315f8c;
      --blue-soft: #edf3f9;
      --amber: #9a5a22;
      --amber-soft: #fbf2e8;
      --red: #9c433a;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      font-size: 16px;
      line-height: 1.72;
    }
    header {
      border-bottom: 1px solid var(--line);
      background:
        radial-gradient(circle at 82% 18%, rgba(30,106,93,.13), transparent 28%),
        linear-gradient(130deg, #f4f8f7, #fff 66%);
    }
    .shell { width: min(1180px, calc(100% - 40px)); margin: 0 auto; }
    .hero { padding: 58px 0 44px; }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--teal);
      text-transform: uppercase;
      letter-spacing: .12em;
      font-weight: 800;
      font-size: .78rem;
    }
    h1 { margin: 0; max-width: 980px; font-size: clamp(2rem, 5vw, 3.35rem); line-height: 1.12; letter-spacing: -.035em; }
    .subtitle { max-width: 930px; margin: 18px 0 0; color: var(--muted); font-size: 1.08rem; }
    .meta { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 24px; }
    .meta span { border: 1px solid #cfd8d8; border-radius: 999px; padding: 5px 11px; background: rgba(255,255,255,.78); font-size: .82rem; }
    nav {
      position: sticky; top: 0; z-index: 20;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.94);
      backdrop-filter: blur(10px);
    }
    nav .shell { display: flex; gap: 21px; overflow-x: auto; padding: 11px 0; white-space: nowrap; }
    nav a { color: #34464b; font-size: .86rem; text-decoration: none; font-weight: 650; }
    nav a:hover { color: var(--teal); }
    section { padding: 44px 0 52px; border-bottom: 1px solid var(--line); }
    h2 { margin: 0 0 13px; font-size: clamp(1.55rem, 3vw, 2.15rem); line-height: 1.22; letter-spacing: -.02em; }
    h3 { margin: 31px 0 10px; font-size: 1.18rem; line-height: 1.35; }
    p { margin: 10px 0; }
    .lead { max-width: 980px; color: #415056; font-size: 1.02rem; }
    .headline-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin: 25px 0 3px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .headline { padding: 20px 22px; border-right: 1px solid var(--line); }
    .headline:last-child { border-right: 0; }
    .headline .num { display: block; color: var(--teal); font-size: .78rem; font-weight: 800; letter-spacing: .12em; }
    .headline p { margin-bottom: 0; color: #405057; font-size: .93rem; }
    .method-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 22px 0; }
    .method-box, .conclusion-box { border-left: 4px solid var(--blue); background: var(--blue-soft); padding: 15px 18px; }
    .conclusion-box { border-left-color: var(--teal); background: var(--teal-soft); }
    .method-box span, .conclusion-box span { display: block; font-size: .76rem; font-weight: 850; letter-spacing: .11em; text-transform: uppercase; color: var(--blue); }
    .conclusion-box span { color: var(--teal); }
    .method-box p, .conclusion-box p { margin: 5px 0 0; font-size: .92rem; }
    .table-wrap { width: 100%; overflow-x: auto; margin: 18px 0 25px; border: 1px solid var(--line); border-radius: 7px; }
    table { width: 100%; border-collapse: collapse; min-width: 800px; background: white; font-size: .87rem; }
    th { position: sticky; top: 0; background: #f1f4f4; color: #344248; text-align: left; font-size: .76rem; letter-spacing: .025em; }
    th, td { padding: 10px 11px; border-bottom: 1px solid #e6eaeb; vertical-align: top; }
    tbody tr:last-child td { border-bottom: 0; }
    tbody tr:hover { background: #fafcfc; }
    .numeric td:not(:first-child) { font-variant-numeric: tabular-nums; }
    .dense { font-size: .81rem; }
    .compact { font-size: .8rem; }
    .hash-table td:last-child { max-width: 250px; word-break: break-all; font-family: Consolas, monospace; font-size: .7rem; }
    figure { margin: 28px 0 35px; }
    figure img { display: block; width: 100%; height: auto; border: 1px solid var(--line); background: white; }
    figcaption { max-width: 1020px; margin-top: 10px; color: var(--muted); font-size: .88rem; }
    .equation-card { margin: 20px 0; border: 1px solid #ccd6d7; background: #fbfcfc; padding: 15px 18px 13px; }
    .equation-title { text-align: center; font-weight: 800; color: #2c4147; font-size: .88rem; }
    .math-scroll { overflow-x: auto; padding: 13px 0 6px; }
    math[display="block"] { width: max-content; margin: 0 auto; font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif; font-size: 1.24rem; }
    .equation-card > p { max-width: 960px; margin: 8px auto 2px; color: var(--muted); font-size: .88rem; text-align: center; }
    pre { margin: 12px 0; padding: 15px 17px; overflow-x: auto; border: 1px solid #d8dedf; border-left: 4px solid var(--teal); background: #f7f9f9; font-family: "Cascadia Mono", Consolas, monospace; font-size: .82rem; line-height: 1.55; white-space: pre-wrap; }
    code { padding: .1em .3em; background: #eef1f1; border-radius: 3px; font-size: .9em; }
    details { margin: 18px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    summary { cursor: pointer; padding: 13px 0; font-weight: 750; color: #30474d; }
    details > div { padding: 0 0 18px; }
    .callout { margin: 22px 0; border: 1px solid #cddbd7; background: var(--teal-soft); padding: 15px 18px; }
    .callout.warning { border-color: #ead6bb; background: var(--amber-soft); }
    .callout p:last-child { margin-bottom: 0; }
    .definitions { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 18px 0; }
    .definition { border-top: 3px solid #95aaa6; background: var(--soft); padding: 14px 16px; }
    .definition strong { color: #28474a; }
    .links { display: flex; flex-wrap: wrap; gap: 9px 17px; margin-top: 18px; }
    .links a { color: var(--teal); text-decoration-thickness: 1px; text-underline-offset: 3px; }
    footer { padding: 28px 0 45px; color: var(--muted); font-size: .82rem; }
    @media (max-width: 820px) {
      .shell { width: min(100% - 24px, 1180px); }
      .hero { padding: 40px 0 32px; }
      .headline-grid, .method-grid, .definitions { grid-template-columns: 1fr; }
      .headline { border-right: 0; border-bottom: 1px solid var(--line); }
      .headline:last-child { border-bottom: 0; }
      section { padding: 34px 0 40px; }
      math[display="block"] { font-size: 1.05rem; }
    }
    @media print {
      nav { display: none; }
      body { font-size: 11pt; }
      figure, .equation-card, .method-grid { break-inside: avoid; }
    }
    """
    headline_html = "".join(
        f'<div class="headline"><span class="num">0{index}</span>'
        f"<p>{esc(text)}</p></div>"
        for index, text in enumerate(headlines, start=1)
    )
    prompt_example = prompts[
        prompts["model_label"].astype(str).eq("Qwen3-8B")
        & prompts["prompt_mode"].astype(str).eq("direct")
    ].iloc[0]
    rendered_direct = str(prompt_example["rendered_prompt_redacted"])
    rendered_thinking = str(
        prompts[
            prompts["model_label"].astype(str).eq("Qwen3-8B")
            & prompts["prompt_mode"].astype(str).eq("native_thinking")
        ].iloc[0]["rendered_prompt_redacted"]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Query-last Counting Empirical Law</title>
  <meta name="description" content="Independent query-last empirical-law fits for three Qwen3 models and three prompt modes.">
  <style>{style}</style>
</head>
<body>
<!-- {REPORT_MARKER} -->
<header>
  <div class="shell hero">
    <p class="eyebrow">Realistic CoT NiaH Count · Qwen-only analysis</p>
    <h1>固定 Query Last 后，Qwen 的 counting empirical law</h1>
    <p class="subtitle">三种 Qwen3 规模 × direct / enumeration / native thinking 共 9 个独立 strata。每个 stratum 都拥有自己的截距、长度阶数与 needle 阶数；不再让 query order、其他模型家族或共享斜率混入拟合。</p>
    <div class="meta">
      <span>1,350 query-last requests</span><span>3 Qwen models</span><span>3 modes</span>
      <span>30 T×N cells / stratum</span><span>5 seeds / cell</span><span>Generated {esc(generated_at)}</span>
    </div>
  </div>
</header>
<nav><div class="shell">
  <a href="#summary">结论</a><a href="#setup">实验与 Prompt</a><a href="#definitions">计算定义</a>
  <a href="#observed">观测准确率</a><a href="#accuracy-law">Accuracy law</a>
  <a href="#error-laws">Bias / error law</a><a href="#search">候选搜索</a><a href="#limits">边界与复现</a>
</div></nav>
<main class="shell">
  <section id="summary">
    <h2>先给结论</h2>
    <p class="lead">固定 query last 后，最清楚的关系出现在 direct counting：needle 数增加会系统性降低准确率。Enumeration 和 native thinking 在 Qwen3-8B/32B 上接近满分，反而缺少估计斜率所需的错误事件；“没有显著斜率”在这里主要是天花板限制，不是证明长度或 N 完全无效。</p>
    <div class="headline-grid">{headline_html}</div>
    {method_conclusion(
        "结论只来自 9 个模型×mode 独立拟合；支持等级同时要求固定式的 FDR-adjusted 全局检验、leave-one-seed-out 改善和 cell-cluster bootstrap 斜率区间，不以训练集 R² 单独判定。",
        support_summary(fixed, "exact"),
    )}
  </section>

  <section id="setup">
    <h2>1. 实验设置与 Prompt 格式</h2>
    <p class="lead">这是正式 6,300-request 实验的严格子集，不重新生成任何输出。筛选条件只有 <code>model ∈ {{Qwen3-1.7B, Qwen3-8B, Qwen3-32B}}</code> 与 <code>query_order = query_last</code>。</p>
    <div class="definitions">
      <div class="definition"><strong>网格。</strong> T ∈ {{2,000, 5,000, 10,000}}；N ∈ {{1,2,3,4,5,6,8,10,20,30}}；seed ∈ {{1234,…,1238}}。每个模型×mode 为 3×10×5=150 条。</div>
      <div class="definition"><strong>长度。</strong> T 是插入全部 needles 后、加入 task/chat template 前，用 canonical tokenizer <code>Qwen/Qwen3-8B</code> 计算的 passage tokens。</div>
      <div class="definition"><strong>配对。</strong> 同一 master stimulus 的 passage 在模型与 mode 间复用；固定 T 时增加 N 会相应缩短 filler，因此 N 与总长度不会机械地一起增长。</div>
      <div class="definition"><strong>运行。</strong> Lambda Cloud 单卡 NVIDIA H100 PCIe 80 GB；BF16、vLLM 0.25.1、max model length 16,384、tensor parallel 1；batch size 随模型为 16/8/1。</div>
    </div>
    <h3>Query-last 外层结构</h3>
    <pre>&lt;passage&gt;
{{context}}
&lt;/passage&gt;

&lt;TASK BLOCK&gt;</pre>
    <p>Passage 之前没有寻找 city-score records 的 task cue；这是与 query first 唯一的用户消息层位置差异。</p>
    <h3>Direct 与 native thinking 使用的 task block</h3>
    <pre>{esc(DIRECT_TASK_BLOCK)}</pre>
    <p>两者的 user message 完全相同。Direct 调用 Qwen 官方 chat template 时设置 <code>enable_thinking=false</code>；native thinking 设置 <code>enable_thinking=true</code>。</p>
    <h3>Enumeration 使用的 task block</h3>
    <pre>{esc(ENUMERATION_TASK_BLOCK)}</pre>
    <p>Enumeration 关闭 thinking；parser 同时核对编号清单与最后的 <code>Total</code>。Primary exact accuracy 只要求最终数值正确，完整清单 fidelity 另行记录，二者不可混同。</p>
    <h3>Qwen tokenizer-rendered wrapper</h3>
    <details>
      <summary>查看 thinking 关闭时的完整 redacted rendered prompt</summary>
      <div><pre>{esc(rendered_direct)}</pre></div>
    </details>
    <details>
      <summary>查看 native thinking 开启时的完整 redacted rendered prompt</summary>
      <div><pre>{esc(rendered_thinking)}</pre></div>
    </details>
    <p>Direct/enumeration 的 assistant 起始处由模板预置空的 <code>&lt;think&gt;…&lt;/think&gt;</code>；native thinking 不预置闭合块。这里只把 passage body 替换为 <code>[PASSAGE OMITTED]</code>，SHA256 来自未删改的完整 rendered prompt。</p>
    {prompt_settings_table(prompts)}
    {method_conclusion(
        "Prompt 内容和 rendered wrapper 从冻结 requests.jsonl 审计产物读取；三种模型各 9 个 query-last 组合逐行核对生成参数和 sample rendered-prompt SHA256。",
        "三个 Qwen checkpoint 的用户消息结构一致；mode 差异来自 enumeration task block 或官方 thinking 开关及对应解码预算，而不是另写了隐藏任务提示。",
    )}
  </section>

  <section id="definitions">
    <h2>2. 每一段分析使用什么计算</h2>
    <div class="definitions">
      <div class="definition"><strong>Exact correctness。</strong> 只有成功解析、未截断且预测整数等于 N 时 Y=1；parse failure、格式失败和 truncation 均为 0，分母固定 150。</div>
      <div class="definition"><strong>Signed bias。</strong> b=预测数−N；b&lt;0 是 undercount，b&gt;0 是 overcount。只在 parsed numeric outputs 上定义。</div>
      <div class="definition"><strong>Robust bias target。</strong> z=asinh(b)。它在 0 附近近似 b，在大尾部近似带符号 log，避免少量巨大 enumeration overcount 支配回归。</div>
      <div class="definition"><strong>Absolute-error target。</strong> u=log(1+|b|)。它忽略方向、保留误差幅度；分离 log-law 下的 β 可转成典型误差的幂阶。</div>
      <div class="definition"><strong>Density。</strong> ρ=1000N/T（每 1k canonical tokens 的 needles）。Density-only 候选只使用 ln(ρ/1)。</div>
      <div class="definition"><strong>Held-out validation。</strong> Primary 是 leave-one-seed-out；另报告 leave-one-N-level 和 leave-one-T-level，以区分网格内泛化与跨水平外推。</div>
    </div>
    {math_exact_law()}
    {math_bias_law()}
    {method_conclusion(
        "固定 log-separable 式是预先指定的 inferential target；7 个有限候选只作 sensitivity search。系数区间用 500 次完整 (T,N) cell 聚类 bootstrap；9 个 strata 的全局检验按 target 内 Benjamini–Hochberg FDR 校正。",
        "Exact accuracy、bias 和 absolute error 回答三个不同问题：能否完全数对、错误往哪边偏、错误幅度多大。后两者不能把未解析样本伪装成 bias=0。",
    )}
  </section>

  <section id="observed">
    <h2>3. 先看观测数据，不让回归替代结果</h2>
    {overall_summary_table(summary)}
    <figure>
      <img src="figures/fig01_accuracy_cells.png" alt="Nine heatmaps of query-last exact accuracy by Qwen model, mode, passage length, and needle count.">
      <figcaption><strong>图 1｜每个模型×mode 的 30 个观测 cell。</strong> 横轴是真实 needle 数 N，纵轴是 canonical passage 长度 T；每格是 5 个 seeds 的 exact accuracy，色标固定 0–100%。该图显示 direct 的 N 梯度，也显示 8B/32B enumeration 和 native thinking 的天花板。</figcaption>
    </figure>
    {method_conclusion(
        "每格直接计算 5 个冻结请求的平均 Y；总体表对每个 150-request stratum 计算 exact/parse rate，并在 parsed 子集上计算 mean signed bias 与 MAE。没有平滑、删点或模型借力。",
        "Qwen3-1.7B 三种模式仍有大量错误；Qwen3-8B/32B 的 direct 约为中等准确率，而 query-last enumeration/native thinking 达到 95.3%–98.7%。高分模式的 bias 和 MAE 很小，但完整 enumeration fidelity 仍是另一个指标。",
    )}
  </section>

  <section id="accuracy-law">
    <h2>4. 固定式 Accuracy empirical law</h2>
    <p class="lead">下表的 βT、βN 是每个模型×mode 自己的参数。负数表示该变量增加时准确率 odds 下降；“Odds / ×2”把 log 系数转为长度或 N 翻倍时的赔率倍数。</p>
    {fixed_accuracy_table(fixed)}
    <figure>
      <img src="figures/fig02_accuracy_coefficients.png" alt="Bootstrap intervals for length and needle orders in nine independent accuracy laws.">
      <figcaption><strong>图 2｜固定分离式中的长度阶数与 needle 阶数。</strong> 左图横轴是 βT，右图是 βN；点为全数据拟合，横线为 500 次 cell-cluster bootstrap 95% interval，虚线 0 表示没有可辨识方向。绿色只有在 FDR、held-out gain 和区间方向同时通过时才标为“有支持”。</figcaption>
    </figure>
    <figure>
      <img src="figures/fig03_accuracy_calibration.png" alt="Observed versus leave-one-seed-out predicted cell accuracy for nine strata.">
      <figcaption><strong>图 3｜固定式的 leave-one-seed-out 校准。</strong> 横轴是 held-out seed 的预测概率在 (T,N) cell 内的平均，纵轴是观测 cell accuracy；每点一个 cell，虚线为理想 45°。OOF cell R² 可为负，表示简单平滑 law 甚至不如 cell 均值基线；它不会被训练集拟合优度掩盖。</figcaption>
    </figure>
    <div class="callout">
      <strong>Direct accuracy 的 Qwen 共性。</strong>
      <p>三个规模的 βN 均为明显负值；1.7B 与 32B 的 one-SE 候选进一步选择
      <code>log burden = ln[(T/5000)(N/5)]</code>，8B 则保留 separate log T/N。
      这说明 direct counting 更接近“长度与记录数量共同增加计算负担”，而不是仅由 density
      <code>N/T</code> 决定。8B 的 βT 区间跨 0，提示其主要可识别梯度是 N。</p>
    </div>
    {method_conclusion(
        "每个 stratum 对 150 个 Bernoulli Y 独立拟合 weak-ridge logistic law；ridge=0.01 仅用于完全/近完全分离时保持有限估计。Primary predictive check 是五折 leave-one-seed-out log loss，相对同折 constant model 报告 gain；少于 10 个 failure events 的 stratum 无论 p/q 多小都标为 ceiling-limited。",
        support_summary(fixed, "exact"),
    )}
  </section>

  <section id="error-laws">
    <h2>5. Bias 与绝对误差的独立 laws</h2>
    <p class="lead">这些表只使用 parsed numeric outputs，因此 <code>n</code> 可能小于 150。Bias 的正负用于机制解释；absolute error 更适合问“典型误差幅度随 T/N 成多少阶”。</p>
    <h3>5.1 Signed bias：asinh(b)</h3>
    {continuous_fixed_table(fixed, "bias")}
    <figure>
      <img src="figures/fig04_bias_coefficients.png" alt="Bootstrap intervals for length and needle coefficients of asinh signed bias.">
      <figcaption><strong>图 4｜Signed-bias 斜率。</strong> 横轴分别为 βT 与 βN；正值表示变量增加时更偏 overcount，负值表示更偏 undercount。区间和支持颜色与图 2 相同，但样本只包括 parsed outputs。</figcaption>
    </figure>
    <figure>
      <img src="figures/fig05_bias_surfaces.png" alt="Predicted asinh-centered signed-bias curves versus N at three passage lengths.">
      <figcaption><strong>图 5｜固定式的 bias response-surface 切片。</strong> 横轴 N 使用 log₂ 刻度；纵轴为 sinh(E[asinh(b)])，可理解为对长尾稳健的 centered bias，symlog 轴同时展示 under/over-count。三条线是 T=2k/5k/10k；它们是模型预测，不是逐 cell 原始均值。</figcaption>
    </figure>
    <div class="callout">
      <strong>Qwen 家族内最清楚的共性出现在 direct bias。</strong>
      <p>One-SE 搜索对 1.7B、8B、32B 三个 direct strata 都选择
      <code>asinh(b)=α+βT ln(T/5000)+βN ln(N/5)+βTN ln(T/5000)ln(N/5)</code>。
      8B 与 32B 的 βN 分别约为 0.779/0.773，βTN 约为 0.656/0.488；
      因此 N 引起的 over-count 倾向会在更长 passage 中增强。1.7B 的 βTN 更大但 βN 接近 0，
      表示它在短、长 context 下甚至可能改变 N→bias 的方向。该共性来自 held-out candidate
      selection，但仍属于本数据集内的经验规律，尚未由新数据复现。</p>
    </div>
    <h3>5.2 Absolute error：log(1+|b|)</h3>
    {continuous_fixed_table(fixed, "absolute_error")}
    <figure>
      <img src="figures/fig06_absolute_error_coefficients.png" alt="Bootstrap intervals for length and needle exponents of log absolute error.">
      <figcaption><strong>图 6｜典型 absolute-error 的长度/needle 阶数。</strong> 因目标是 log(1+|b|)，βT、βN 可直接读成 <em>1+典型绝对误差</em> 对 T、N 的近似幂指数；0 线表示没有稳定的乘法尺度变化。</figcaption>
    </figure>
    {method_conclusion(
        "在 parsed 子集上分别以 asinh(b) 和 log1p(|b|) 做 OLS；使用与 accuracy 相同的固定坐标、cell bootstrap、leave-one-seed-out MAE 和 target 内 FDR。",
        f"Bias：{support_summary(fixed, 'bias')} Absolute error：{support_summary(fixed, 'absolute_error')}",
    )}
  </section>

  <section id="search">
    <h2>6. 有边界的坐标/目标搜索</h2>
    <p class="lead">为了检验“也许不是分离幂律”，每个 stratum 比较 constant、density、burden、separate log T/N、semi-log N、piecewise log N 与 log interaction。选择使用 seed-OOF 的 one-standard-error rule：只要更简单模型落在最佳模型一个标准误以内，就选择更简单者。</p>
    <figure>
      <img src="figures/fig07_candidate_gain.png" alt="Selected candidate and seed-held-out gain relative to the constant model.">
      <figcaption><strong>图 7｜有限候选搜索。</strong> 每格写出 one-SE 选择的函数族和相对 constant 的 seed-OOF loss 改善；绿色为改善、红色为恶化。搜索结果完整保留，不只展示最佳曲线。</figcaption>
    </figure>
    <details><summary>Exact accuracy 的候选选择表</summary><div>{selected_table(selected, "exact")}</div></details>
    <details><summary>Signed bias 的候选选择表</summary><div>{selected_table(selected, "bias")}</div></details>
    <details><summary>Absolute error 的候选选择表</summary><div>{selected_table(selected, "absolute_error")}</div></details>
    {method_conclusion(
        "每个候选在 leave-one-seed、leave-one-N-level、leave-one-T-level 三套分组验证中全部运行；one-SE rule 只使用 seed folds 选型，另外两套作为外推敏感性，不允许事后删点或无限增加函数。",
        "没有一个复杂坐标在 9 个 strata 上普遍胜出。Direct 更常支持 N-dependent 形式；高准确率 enumeration/native-thinking 多由 constant 或简单式胜出。当前证据支持“统一函数族、参数随模型与 mode 变化”，不支持共享一个数值斜率。",
    )}
  </section>

  <section id="limits">
    <h2>7. 能说什么，不能说什么；以及如何复现</h2>
    <div class="callout">
      <strong>最稳妥的 counting-mechanism 结论。</strong>
      <p>固定 query last 后，Qwen direct counting 的主要 difficulty axis 是 N，而不是单一 density；更大的 Qwen 并没有消除 direct 的 N 梯度，但 enumeration/native thinking 在 8B/32B 上把最终 count 推至天花板。此时可识别的是“错误概率很低”，不是精确的零斜率。</p>
    </div>
    <div class="callout warning">
      <strong>限制。</strong>
      <p>每个 stratum 只有 30 个 (T,N) cells、每格 5 seeds；T 只有 3 个水平。Near-ceiling strata 的 failure events 太少，bootstrap 区间和 q-value 都会不稳定。Bias/absolute-error 条件于 parsed outputs，不能描述格式失败样本的潜在数值误差；本报告是观察性 response surface，不是架构因果结论。</p>
    </div>
    <div class="links">
      <a href="tables/qwen_query_last_requests.csv">1,350 request rows</a>
      <a href="tables/accuracy_cells.csv">270 observed cells</a>
      <a href="tables/fixed_log_separable_laws.csv">Fixed-law parameters</a>
      <a href="tables/candidate_comparison.csv">All candidate metrics</a>
      <a href="tables/candidate_fold_metrics.csv">All fold metrics</a>
      <a href="tables/selected_laws.csv">One-SE selections</a>
      <a href="tables/fixed_law_oof_predictions.csv">OOF predictions</a>
      <a href="tables/fixed_law_bootstrap_draws.csv">Bootstrap draws</a>
      <a href="tables/qwen_query_last_prompt_settings.csv">Prompt/settings audit</a>
      <a href="analysis_manifest.json">Analysis manifest</a>
      <a href="scripts/build_qwen_query_last_report.py">Rebuild script</a>
      <a href="scripts/audit_qwen_query_last_report.py">Audit script</a>
      <a href="{esc(source_report_href)}">Return to eight-model report</a>
      <a href="SHA256SUMS.tsv">SHA256 manifest</a>
    </div>
    {method_conclusion(
        "Audit checks the 1,350-row subset, 9×150 strata, complete 3×10×5 grids, prompt rows, table schemas, MathML/image/link integrity, checksum manifest, and recomputation of headline accuracies from the request subset.",
        "所有结果可由本目录脚本从 canonical eight-model request table重建；原始 6,300 条请求、prompt 快照和实验产物均未修改。",
    )}
  </section>
</main>
<footer><div class="shell">{REPORT_MARKER} · query_last only · no request deletion · generated {esc(generated_at)}</div></footer>
</body>
</html>
"""


def method_conclusion_v2(method: str, conclusion: str) -> str:
    return f"""
    <div class="method-grid">
      <div class="method-box"><span>测评方法</span><p>{method}</p></div>
      <div class="conclusion-box"><span>当前结论</span><p>{conclusion}</p></div>
    </div>
    """


def build_mode_specific_html(
    *,
    summary: pd.DataFrame,
    cells: pd.DataFrame,
    selected: pd.DataFrame,
    goodness: pd.DataFrame,
    prompts: pd.DataFrame,
    source_report_root: Path,
    source_report_href: str,
    generated_at: str,
) -> str:
    exact = goodness[goodness["target"].eq("exact")]
    bias = goodness[goodness["target"].eq("bias")]
    strong_exact = [
        stratum_label(row.model_label, row.prompt_mode)
        for row in exact[
            exact["best_evidence"].isin(
                ["strong_generalizing", "strong_within_grid"]
            )
        ].itertuples()
    ]
    ceiling_exact = [
        stratum_label(row.model_label, row.prompt_mode)
        for row in exact[exact["best_evidence"].eq("ceiling_limited")].itertuples()
    ]
    strong_bias = [
        stratum_label(row.model_label, row.prompt_mode)
        for row in bias[
            bias["best_evidence"].isin(
                ["strong_generalizing", "strong_within_grid"]
            )
        ].itertuples()
    ]
    prompt_direct = str(
        prompts[
            prompts["model_label"].astype(str).eq("Qwen3-8B")
            & prompts["prompt_mode"].astype(str).eq("direct")
        ].iloc[0]["rendered_prompt_redacted"]
    )
    prompt_thinking = str(
        prompts[
            prompts["model_label"].astype(str).eq("Qwen3-8B")
            & prompts["prompt_mode"].astype(str).eq("native_thinking")
        ].iloc[0]["rendered_prompt_redacted"]
    )
    style = r"""
    :root {
      --ink:#172126; --muted:#5d696f; --line:#dce3e4; --paper:#fff;
      --soft:#f5f8f8; --teal:#1f6b5d; --teal-soft:#e9f4f1;
      --blue:#315f8c; --blue-soft:#edf3f9; --amber:#9b5b22;
      --amber-soft:#fcf2e7; --red:#934038;
    }
    *{box-sizing:border-box} html{scroll-behavior:smooth}
    body{margin:0;color:var(--ink);background:var(--paper);
      font-family:Inter,"Segoe UI","Microsoft YaHei",Arial,sans-serif;
      font-size:16px;line-height:1.72}
    .shell{width:min(1220px,calc(100% - 40px));margin:0 auto}
    header{border-bottom:1px solid var(--line);
      background:radial-gradient(circle at 82% 12%,rgba(31,107,93,.14),transparent 29%),
      linear-gradient(130deg,#f3f8f7,#fff 67%)}
    .hero{padding:58px 0 44px}.eyebrow{margin:0 0 9px;color:var(--teal);
      text-transform:uppercase;letter-spacing:.13em;font-weight:800;font-size:.78rem}
    h1{margin:0;max-width:1050px;font-size:clamp(2rem,5vw,3.35rem);
      line-height:1.1;letter-spacing:-.035em}
    .subtitle{max-width:980px;margin:18px 0 0;color:var(--muted);font-size:1.08rem}
    .meta{display:flex;flex-wrap:wrap;gap:9px;margin-top:23px}
    .meta span{border:1px solid #ced9d8;border-radius:999px;padding:5px 11px;
      background:rgba(255,255,255,.8);font-size:.82rem}
    nav{position:sticky;top:0;z-index:30;border-bottom:1px solid var(--line);
      background:rgba(255,255,255,.95);backdrop-filter:blur(10px)}
    nav .shell{display:flex;gap:20px;overflow-x:auto;padding:11px 0;white-space:nowrap}
    nav a{color:#34464b;font-size:.86rem;text-decoration:none;font-weight:670}
    nav a:hover{color:var(--teal)}
    section{padding:44px 0 52px;border-bottom:1px solid var(--line)}
    h2{margin:0 0 13px;font-size:clamp(1.55rem,3vw,2.12rem);
      line-height:1.22;letter-spacing:-.02em}
    h3{margin:31px 0 10px;font-size:1.18rem}.lead{max-width:1020px;color:#415056}
    .headline-grid{display:grid;grid-template-columns:repeat(3,1fr);
      border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:25px 0}
    .headline{padding:19px 22px;border-right:1px solid var(--line)}
    .headline:last-child{border-right:0}.headline b{display:block;color:var(--teal);
      font-size:.76rem;letter-spacing:.12em;text-transform:uppercase}
    .method-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:22px 0}
    .method-box,.conclusion-box{border-left:4px solid var(--blue);
      background:var(--blue-soft);padding:15px 18px}
    .conclusion-box{border-left-color:var(--teal);background:var(--teal-soft)}
    .method-box span,.conclusion-box span{display:block;font-size:.76rem;
      font-weight:850;letter-spacing:.1em;color:var(--blue)}
    .conclusion-box span{color:var(--teal)}
    .method-box p,.conclusion-box p{margin:5px 0 0;font-size:.92rem}
    .definitions{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:18px 0}
    .definition{border-top:3px solid #91aaa5;background:var(--soft);padding:14px 16px}
    .definition strong{color:#24494a}
    .table-wrap{width:100%;overflow:auto;margin:18px 0 26px;
      border:1px solid var(--line);border-radius:7px}
    table{width:100%;border-collapse:collapse;min-width:860px;background:#fff;font-size:.86rem}
    th{position:sticky;top:0;background:#f0f4f4;color:#344248;font-size:.75rem;
      letter-spacing:.02em;text-align:left}
    th,td{padding:10px 11px;border-bottom:1px solid #e6eaeb;vertical-align:top}
    tbody tr:last-child td{border-bottom:0}.dense{font-size:.79rem}
    .gof-table{min-width:1850px}.numeric td{font-variant-numeric:tabular-nums}
    figure{margin:28px 0 36px}figure img{display:block;width:100%;height:auto;
      border:1px solid var(--line);background:#fff}
    figcaption{max-width:1060px;margin-top:10px;color:var(--muted);font-size:.88rem}
    .equation-card{margin:20px 0;border:1px solid #cbd7d7;background:#fbfcfc;
      padding:16px 18px}.equation-title{text-align:center;font-weight:800;color:#294349}
    .math-scroll,.formula-scroll{overflow-x:auto;padding:12px 0 5px}
    math{font-family:"Cambria Math","STIX Two Math","Times New Roman",serif}
    math[display="block"]{width:max-content;margin:0 auto;font-size:1.25rem}
    .inline-formula{font-size:.93rem;white-space:nowrap}.formula-scroll{min-width:350px}
    pre{padding:15px 17px;overflow:auto;border:1px solid #d8dfe0;
      border-left:4px solid var(--teal);background:#f7f9f9;
      font-family:"Cascadia Mono",Consolas,monospace;font-size:.81rem;line-height:1.55}
    code{padding:.1em .3em;background:#edf1f1;border-radius:3px}
    details{margin:18px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
    summary{cursor:pointer;padding:13px 0;font-weight:750;color:#30474d}
    details>div{padding-bottom:18px}.callout{margin:22px 0;padding:15px 18px;
      border:1px solid #cddbd7;background:var(--teal-soft)}
    .callout.warning{border-color:#ead4b8;background:var(--amber-soft)}
    .metric-list{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}
    .metric{padding:13px 15px;background:var(--soft);border:1px solid var(--line)}
    .metric b{display:block;color:#2d4d50}.links{display:flex;flex-wrap:wrap;
      gap:9px 17px;margin-top:18px}.links a{color:var(--teal);text-underline-offset:3px}
    footer{padding:28px 0 44px;color:var(--muted);font-size:.82rem}
    @media(max-width:760px){
      .shell{width:min(100% - 24px,1220px)}.hero{padding:39px 0 31px}
      .headline-grid,.method-grid,.definitions,.metric-list{grid-template-columns:1fr}
      .headline{border-right:0;border-bottom:1px solid var(--line)}
      section{padding:34px 0 40px}math[display="block"]{font-size:1.03rem}
    }
    @media print{nav{display:none}figure,.equation-card,.method-grid{break-inside:avoid}}
    """
    generic_law = """
    <div class="equation-card">
      <div class="equation-title">每个模型 × mode 独立选择的统一函数框架</div>
      <div class="math-scroll">
        <math display="block" aria-label="General mode-specific empirical law">
          <mrow>
            <msub><mi>g</mi><mi>r</mi></msub><mo>(</mo><mi>Y</mi><mo>)</mo>
            <mo>=</mo><msub><mi>α</mi><mi>r</mi></msub>
            <mo>+</mo><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>J</mi><mi>r</mi></msub></munderover>
            <msub><mi>β</mi><mrow><mi>r</mi><mi>j</mi></mrow></msub>
            <msub><mi>φ</mi><mrow><mi>r</mi><mi>j</mi></mrow></msub>
            <mo>(</mo><mi>T</mi><mo>,</mo><mi>N</mi><mo>)</mo>
          </mrow>
        </math>
      </div>
      <p>r 表示一个具体的 Qwen checkpoint × mode × target。函数族相同，但
      变换 g、坐标项 φ、项数 J 与参数 α/β 都允许随 r 改变；因此没有强迫
      direct、enumeration、native thinking 使用同一条曲线。</p>
    </div>
    """
    target_math = """
    <div class="equation-card">
      <div class="equation-title">三个互不替代的测评目标</div>
      <div class="math-scroll">
        <math display="block" aria-label="Exact correctness, signed bias, and absolute error">
          <mtable columnalign="right left" rowspacing=".65em">
            <mtr><mtd><mi>Y</mi><mo>=</mo><mn>1</mn></mtd>
              <mtd><mtext>iff parsed, untruncated, and predicted count equals N</mtext></mtd></mtr>
            <mtr><mtd><mi>b</mi><mo>=</mo><mover><mi>N</mi><mo>^</mo></mover><mo>−</mo><mi>N</mi></mtd>
              <mtd><mtext>signed bias on parsed numeric outputs</mtext></mtd></mtr>
            <mtr><mtd><mi>a</mi><mo>=</mo><mo>|</mo><mi>b</mi><mo>|</mo></mtd>
              <mtd><mtext>absolute count error on parsed numeric outputs</mtext></mtd></mtr>
          </mtable>
        </math>
      </div>
    </div>
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Query-last Mode-specific Empirical Laws</title>
  <meta name="description" content="Nested held-out empirical-law analysis for each Qwen model and prompting mode.">
  <style>{style}</style>
</head>
<body>
<!-- {REPORT_MARKER} -->
<header><div class="shell hero">
  <p class="eyebrow">Realistic CoT NiaH Count · Qwen-only report v2</p>
  <h1>Query-last 下，Qwen 三种 mode 各自寻找 empirical law</h1>
  <p class="subtitle">3 个 Qwen checkpoint × 3 种 mode × 3 个目标分别选型。报告把“最好预测”
  与“一标准误最简模型”并列展示，并且只用嵌套 held-out 结果评价拟合优度。</p>
  <div class="meta"><span>1,350 requests</span><span>9 independent strata</span>
  <span>48 frozen candidates / target</span><span>Nested leave-one-seed-out</span>
  <span>Query last only</span><span>Generated {esc(generated_at)}</span></div>
</div></header>
<nav><div class="shell">
  <a href="#summary">结论</a><a href="#setup">实验与 Prompt</a><a href="#evaluation">测评方法</a>
  <a href="#observed">观测结果</a><a href="#accuracy">Accuracy laws</a>
  <a href="#bias">Bias / error laws</a><a href="#search">搜索与稳健性</a><a href="#reproduce">复现</a>
</div></nav>
<main class="shell">
<section id="summary">
  <h2>结论先行</h2>
  <div class="headline-grid">
    <div class="headline"><b>Accuracy</b><p>有强 held-out 支持的 strata：
      {esc("；".join(strong_exact))}。</p></div>
    <div class="headline"><b>Ceiling</b><p>错误事件不足 10 个、无法稳定估斜率：
      {esc("；".join(ceiling_exact))}。</p></div>
    <div class="headline"><b>Signed bias</b><p>有强支持的 bias laws：
      {esc("；".join(strong_bias)) or "无"}。</p></div>
  </div>
  <p class="lead">最稳定的 counting-mechanism 信号仍然来自 direct：准确率随 N 增大而显著下降；
  Qwen3-8B/32B 的 direct signed bias 与 absolute error 也有可预测的 T–N response surface。
  Qwen3-1.7B enumeration 的 bias 被少量巨大 overcount 主导，扩展搜索后仍不能在 held-out
  数据上优于常数基线，因此正式结论是“当前样本未发现稳定统一 law”，而不是继续试到训练集 R² 好看为止。</p>
  {method_conclusion_v2(
      "所有 headline 使用外层留一 seed 的 post-selection 预测；每个外层折内重新执行候选选择。"
      "显著性使用完整 (T,N) cell 聚类 bootstrap，并在同一 target 的 9 个 strata 内做 BH-FDR。",
      "三种 mode 可以且确实选择了不同函数。Direct 的规律最可识别；8B/32B 的 enumeration/native-thinking "
      "主要受 ceiling 限制；1.7B enumeration 的数值误差主要受不稳定长尾限制。"
  )}
</section>

<section id="setup">
  <h2>1. 实验设置与 Prompt 格式</h2>
  <p class="lead">本报告是正式 6,300-request 实验的只读子集：
  <code>model ∈ {{Qwen3-1.7B, Qwen3-8B, Qwen3-32B}}</code> 且
  <code>query_order=query_last</code>。没有重跑模型、删除请求或修改 parser。</p>
  <div class="definitions">
    <div class="definition"><strong>设计网格。</strong> T∈{{2,000,5,000,10,000}}；
      N∈{{1,2,3,4,5,6,8,10,20,30}}；5 seeds。每个模型×mode 为 150 请求。</div>
    <div class="definition"><strong>长度 T。</strong> 插入 needles 后、添加 task/chat template 前，
      用 canonical tokenizer <code>Qwen/Qwen3-8B</code> 计算的 passage tokens。</div>
    <div class="definition"><strong>配对。</strong> master stimulus 在模型和 mode 间复用；
      固定 T 时增加 N 会缩短 filler，因此 N 与 passage 长度不是机械共线。</div>
    <div class="definition"><strong>运行。</strong> 单张 NVIDIA H100 PCIe 80 GB；BF16；
      vLLM 0.25.1；max model length 16,384；batch size 依模型为 16/8/1。</div>
  </div>
  <h3>Query-last 外层结构</h3>
  <pre>&lt;passage&gt;
[PASSAGE]
&lt;/passage&gt;

[TASK BLOCK]</pre>
  <h3>Direct 与 native thinking 的 task block</h3>
  <pre>{esc(DIRECT_TASK_BLOCK)}</pre>
  <h3>Enumeration 的 task block</h3>
  <pre>{esc(ENUMERATION_TASK_BLOCK)}</pre>
  <p>Direct 与 native thinking 的 user message 相同；差异只在 Qwen 官方 chat template 的
  <code>enable_thinking</code> 和 decoding budget。Enumeration 使用要求列举全部记录的独立 task block。</p>
  <details><summary>查看 Qwen3-8B direct 的 redacted rendered prompt</summary>
    <div><pre>{esc(prompt_direct)}</pre></div></details>
  <details><summary>查看 Qwen3-8B native-thinking 的 redacted rendered prompt</summary>
    <div><pre>{esc(prompt_thinking)}</pre></div></details>
  {prompt_settings_table(prompts)}
  {method_conclusion_v2(
      "Prompt、decoding 参数与 rendered-prompt SHA256 均从冻结 request 审计表读取；每个 Qwen×mode "
      "组合逐行核对。passage body 仅在展示时替换为 [PASSAGE OMITTED]。",
      "三个 checkpoint 的消息结构一致；mode 差异来自 task block 或官方 thinking 开关，不是隐含的额外提示。"
  )}
</section>

<section id="evaluation">
  <h2>2. 拟合目标、候选函数与拟合优度</h2>
  {target_math}
  {generic_law}
  <div class="definitions">
    <div class="definition"><strong>候选坐标。</strong> 原始/线性 N、ln N、ln T、density N/T、
      burden TN、二次项、T×N interaction、N=5 hinge、T-level intercept 与 T-specific N slopes。</div>
    <div class="definition"><strong>候选估计器。</strong> Accuracy 为 3 个 ridge 强度的 logistic；
      bias 为 asinh-ridge、asinh-Huber、raw-bias Huber；absolute error 为 log1p/sqrt 与 ridge/Huber。</div>
    <div class="definition"><strong>Best predictive。</strong> 内层 CV loss 最小的候选；
      用于回答“在冻结搜索空间内能达到多好的 held-out 预测”。</div>
    <div class="definition"><strong>One-SE parsimonious。</strong> 与内层最优相差不超过一折标准误时，
      选择项数更少者；用于检查结论是否依赖复杂曲面。</div>
  </div>
  <div class="metric-list">
    <div class="metric"><b>Nested loss / null</b>外层 held-out 请求上的 log loss（accuracy）或
      raw-scale MAE（bias/error），与同外层训练集拟合的常数基线比较。</div>
    <div class="metric"><b>Gain</b><code>100×(L_null−L_model)/L_null</code>。正值为改善；
      负值表示选型后仍不如常数。</div>
    <div class="metric"><b>Cell R²</b>先在每个 (T,N) cell 内平均 nested OOF 预测与观测，再算 R²；
      它回答 response surface 能解释多少 cell-level 变化。</div>
    <div class="metric"><b>Brier / ECE</b>Accuracy 的概率误差与 6 个等频 bins 的 expected calibration error；
      越小越好。</div>
    <div class="metric"><b>Leave-N / Leave-T</b>固定全数据所选公式，分别留出整个 N level 或 T level；
      检查跨坐标水平外推，而非只在相同网格内插值。</div>
    <div class="metric"><b>Evidence</b>Strong 要求 FDR q&lt;.05、nested gain≥10%、cell R²≥.40；
      若 leave-N 或 leave-T 为负，则标记为 in-grid warning。错误不足 10 个一律 ceiling-limited。</div>
  </div>
  {method_conclusion_v2(
      "外层 5 folds 每次留出一个 seed；内层仅用其余 4 seeds 再做留一 seed 选择。这样报告的 "
      "nested 指标没有让 held-out seed 参与公式选择。48 个候选在分析前写入脚本并全部保留。",
      "“理想拟合”按 held-out loss、校准、cell R²、跨 N/T 泛化和 bootstrap 稳定性共同定义；"
      "训练集 R² 或单张漂亮曲线不能单独构成证据。"
  )}
</section>

<section id="observed">
  <h2>3. 先看观测准确率</h2>
  {overall_summary_table(summary)}
  <figure><img src="figures/fig01_accuracy_cells.png"
    alt="Observed exact accuracy heatmaps for nine Qwen model-mode strata.">
    <figcaption><strong>图 1｜30 个设计 cell 的原始 exact accuracy。</strong>
    横轴是真实 needle 数 N，纵轴是 passage 长度 T；每格是 5 seeds 的均值，颜色范围固定为 0–100%。
    该图不含回归平滑。Direct 的 N 梯度清楚；8B/32B enumeration 与 native thinking 接近天花板。</figcaption></figure>
  {method_conclusion_v2(
      "Exact accuracy 分母固定为每 stratum 150；parse failure、格式错误和 truncation 全部计 0。"
      "Bias/MAE 只在成功解析数值的请求上计算，不能把未解析请求伪装成 bias=0。",
      "1.7B 三种 mode 仍有大量错误；8B/32B direct 约为中等准确率，而 enumeration/native thinking "
      "达到 95.3%–98.7%，因此后两类的斜率估计受低事件数限制。"
  )}
</section>

<section id="accuracy">
  <h2>4. 每个模型 × mode 的 Accuracy law</h2>
  <p class="lead">下表是 best-predictive pipeline。每一行都允许选择不同坐标、interaction 与 ridge；
  “Full-data fitted equation”给出在全部 150 请求上重拟合后的参数，但拟合优度完全来自 nested held-out 预测。</p>
  {mode_specific_gof_table(goodness, selected, target="exact", rule="best")}
  <figure><img src="figures/fig02_mode_specific_accuracy_gof.png"
    alt="Nested log-loss gain and cell R-squared for Qwen accuracy laws.">
    <figcaption><strong>图 2｜拟合优度总览。</strong> 左轴是相对 fold-trained constant 的 nested log-loss
    改善百分比；右轴是 nested OOF cell R²。浅绿为 one-SE 简洁模型，蓝色为 best predictive。
    负值意味着模型在严格 held-out 上比常数更差。</figcaption></figure>
  <figure><img src="figures/fig03_mode_specific_accuracy_scatter.png"
    alt="Observed versus nested held-out predicted cell accuracy.">
    <figcaption><strong>图 3｜观测与 nested held-out 预测。</strong> 横轴为 held-out 预测 accuracy，
    纵轴为 held-out 观测 accuracy；每点一个 (T,N) cell，颜色表示 T。虚线是理想 45°。
    面板内同时标注 best pipeline 的 log-loss gain 与 cell R²。</figcaption></figure>
  <figure><img src="figures/fig04_mode_specific_accuracy_response.png"
    alt="Observed and fitted accuracy response versus N at three passage lengths.">
    <figcaption><strong>图 4｜N response 与 T 切片。</strong> 横轴 N 为 log 刻度；点为观测 cell accuracy，
    线为 best pipeline 的 nested OOF cell prediction；三种颜色对应 T=2k/5k/10k。
    该图直接显示各 mode 可采用不同曲率，而不是强制共享斜率。</figcaption></figure>
  <details><summary>查看 one-SE 与 best-predictive 的选型敏感性</summary>
    <div>{selection_sensitivity_table(goodness, selected, "exact")}</div></details>
  {method_conclusion_v2(
      "Accuracy 对每个请求拟合 Bernoulli logistic family；主 loss 为 log loss，另报告 Brier、ECE、"
      "calibration slope、cell R²、leave-N 和 leave-T。低于 10 个 failure 的 strata 不作斜率显著性宣称。",
      "1.7B 三种 mode 与 8B/32B direct 都存在可复现 N-dependent response；8B/32B 的 "
      "enumeration/native-thinking 因错误过少而 ceiling-limited，即使复杂式偶尔改善也不能解释为稳定阶数。"
  )}
</section>

<section id="bias">
  <h2>5. Signed bias 与 absolute error 的独立 laws</h2>
  <p class="lead">Signed bias <code>b=predicted_count−N</code> 保留方向；
  absolute error <code>|b|</code>只描述幅度。二者分别搜索变换和估计器，不能互相替代。</p>
  <h3>5.1 Signed bias</h3>
  {mode_specific_gof_table(goodness, selected, target="bias", rule="best")}
  <h3>5.2 Absolute count error</h3>
  {mode_specific_gof_table(goodness, selected, target="absolute_error", rule="best")}
  <figure><img src="figures/fig05_mode_specific_error_gof.png"
    alt="Nested MAE gain and cell R-squared for signed-bias and absolute-error laws.">
    <figcaption><strong>图 5｜误差 law 的 held-out 拟合优度。</strong> 上排为 signed bias，下排为
    absolute error；左列是 raw-scale nested MAE 相对常数的改善，右列是 cell R²。
    浅绿/蓝色分别为 one-SE/best pipeline。对几乎全部正确的 strata，R² 可能未定义，此时显示为空。</figcaption></figure>
  <details><summary>Signed bias：one-SE 与 best-predictive 的敏感性</summary>
    <div>{selection_sensitivity_table(goodness, selected, "bias")}</div></details>
  <details><summary>Absolute error：one-SE 与 best-predictive 的敏感性</summary>
    <div>{selection_sensitivity_table(goodness, selected, "absolute_error")}</div></details>
  <div class="callout"><strong>机制解释。</strong>
    <p>8B/32B direct 的 bias 与 absolute error 均能解释大部分 cell-level 变化；
    但 8B direct bias 的 leave-T gain 为负，所以它是“当前三个 T 水平内强、跨新 T 不稳”。
    1.7B native thinking 同时在 bias 与 absolute error 上得到更稳定的 T/N 曲面。
    1.7B direct 的 signed bias 在 expanded search 后仍未可靠超过常数，但 absolute error 有强支持：
    这表示错误幅度有规律，方向却被 seed-level 正负波动抵消。</p></div>
  <div class="callout warning"><strong>为什么 1.7B enumeration 没有漂亮 bias law？</strong>
    <p>该 stratum 的少量巨大 enumeration overcount 造成长尾；asinh、raw-Huber、不同坐标与交互都已比较，
    但 best pipeline 的 nested MAE 仍不优于 fold-trained constant，cell R² 也为负。
    继续增加自由度只会提高训练拟合，不会提供可复现机制，因此正式保留“not supported”。</p></div>
  {method_conclusion_v2(
      "Bias 候选比较 asinh-ridge、asinh-Huber 与 raw-bias Huber；absolute error 比较 log1p/sqrt "
      "目标与 ridge/Huber。选型仍嵌套在外层 seed 之内，最终 loss 全部逆变换回原始 count 单位计算 MAE。",
      "可识别规律集中在 direct（尤其 8B/32B）和 1.7B native-thinking。高准确率 modes 的错误事件太少；"
      "1.7B enumeration 则是长尾太强。两种情形都不应靠事后删点制造显著性。"
  )}
</section>

<section id="search">
  <h2>6. 有边界的持续搜索与稳健性判断</h2>
  <p class="lead">本轮不是只试一个 log-linear 式：每个 target 有 16 个坐标结构 × 3 个估计器/
  超参数，共 48 个候选；9 strata × 3 targets 共比较 1,296 个完整模型，并在每个外层 fold
  内重新选择。所有候选、fold loss、outer choices、nested predictions 与 bootstrap draws 都保存。</p>
  <div class="definitions">
    <div class="definition"><strong>为什么不无限搜索。</strong> 候选空间在看结果前冻结；否则反复换坐标直到
      某个 p&lt;.05 会放大选择偏差。</div>
    <div class="definition"><strong>为什么同时报告两条 pipeline。</strong> Best 检查预测上限；
      one-SE 检查复杂曲面是否真的必要。两者都用 nested OOF，差异本身是模型不确定性。</div>
    <div class="definition"><strong>参数稳定性。</strong> 完整 (T,N) cells 聚类 bootstrap 500 次；
      coefficient intervals 单独保存在表中。</div>
    <div class="definition"><strong>外推边界。</strong> T 只有三个水平；T-level factor/interaction
      可以描述当前网格，但不能作为连续外推的普适 scaling law。</div>
  </div>
  {method_conclusion_v2(
      "模型优劣按 nested held-out loss 为主，校准、cell R²、leave-N/T、bootstrap/FDR 与简洁性共同审核。"
      "所有失败或不收敛尝试均保留在 candidate tables，不只保存最好看的图。",
      "现有数据支持若干模型内、mode 内的 empirical laws，但不支持所有 mode 的单一公式；"
      "对 ceiling 或长尾 strata，“未发现可靠统一拟合”本身就是有效、可复现的结果。"
  )}
</section>

<section id="reproduce">
  <h2>7. 复现、产物与适用范围</h2>
  <div class="callout warning"><strong>适用范围。</strong>
    <p>这些 laws 仅覆盖 query-last、T=2k/5k/10k、N=1–30、当前 prompt/parser/decoding 与三个 Qwen3
    checkpoint。它们是 observational response surfaces，不证明架构因果；在新 T、新 N、query-first
    或其他 tokenizer 上使用前必须重新验证。</p></div>
  <div class="links">
    <a href="tables/qwen_query_last_requests.csv">1,350 requests</a>
    <a href="tables/accuracy_cells.csv">Observed cells</a>
    <a href="tables/flex_candidate_registry.csv">Frozen candidate registry</a>
    <a href="tables/flex_candidate_comparison.csv">All candidate CV metrics</a>
    <a href="tables/flex_selected_laws.csv">Selected laws and coefficients</a>
    <a href="tables/flex_outer_choices.csv">Outer-fold selections</a>
    <a href="tables/flex_nested_predictions.csv">Nested OOF predictions</a>
    <a href="tables/flex_goodness_of_fit.csv">Goodness-of-fit table</a>
    <a href="tables/flex_selected_coefficient_intervals.csv">Bootstrap coefficient intervals</a>
    <a href="tables/qwen_query_last_prompt_settings.csv">Prompt/settings audit</a>
    <a href="analysis_manifest.json">Analysis manifest</a>
    <a href="scripts/build_qwen_mode_specific_report.py">Rebuild script</a>
    <a href="scripts/audit_qwen_mode_specific_report.py">Audit script</a>
    <a href="{esc(source_report_href)}">Eight-model source report</a>
    <a href="SHA256SUMS.tsv">SHA256 manifest</a>
  </div>
  {method_conclusion_v2(
      "Audit 将重新计算 1,350-row filter、9×150 strata、完整 3×10×5 网格、headline accuracy、"
      "nested prediction coverage、table schemas、JSON/CSV/HTML 可解析性、图片/链接/MathML 与 SHA256。",
      "原始 6,300 requests、prompt snapshots 与冻结实验产物未被修改；本目录可从 canonical "
      "eight-model request table 独立重建。"
  )}
</section>
</main>
<footer><div class="shell">{REPORT_MARKER} · query_last only · no request deletion · generated {esc(generated_at)}</div></footer>
</body>
</html>
"""


def write_readme(output_dir: Path, source_report_root: Path) -> None:
    content = f"""# Qwen query-last mode-specific empirical-law report v2

Open `report.html` for the self-contained narrative.

## Scope

- Qwen3-1.7B, Qwen3-8B, and Qwen3-32B only
- query order fixed to `query_last`
- direct, enumeration, and native thinking fitted independently
- 1,350 requests total; 150 requests per model x mode stratum
- 48 frozen candidates per target and stratum
- nested leave-one-seed-out evaluation after candidate selection
- best-predictive and one-standard-error pipelines reported side by side
- no request deletion or post-hoc parser changes

Each model x mode x target is allowed to select a different bounded response
surface in T and N. Exact correctness includes parse/format/truncation failures
as failures. Signed bias and absolute error remain conditional on parsed numeric
outputs.

## Rebuild

Use the same environment as the eight-model report (NumPy, pandas, SciPy,
Matplotlib):

```powershell
python scripts/build_qwen_mode_specific_report.py `
  --source-report-root "{source_report_root}" `
  --output-dir <new-empty-output-directory>
```

The builder refuses to write into a non-empty directory.

## Audit

```powershell
python scripts/audit_qwen_mode_specific_report.py --report-root .
```

The audit recomputes the 1,350-row filter and headline accuracy values, verifies
the complete 3 x 10 x 5 grid in every stratum and dual nested-prediction
coverage, checks tables/images/links/MathML, and validates `SHA256SUMS.tsv`.
"""
    write_text(output_dir / "README.md", content)


def write_checksums(output_dir: Path) -> int:
    rows: list[tuple[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.tsv":
            continue
        relative = path.relative_to(output_dir).as_posix()
        rows.append((digest(path), relative))
    with (output_dir / "SHA256SUMS.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source_report_root = args.source_report_root.resolve()
    output_dir = args.output_dir.resolve()
    request_source = source_report_root / "tables" / "request_level_report.csv"
    prompt_source = (
        source_report_root / "tables" / "model_prompt_format_examples.csv"
    )
    if not request_source.is_file():
        raise FileNotFoundError(request_source)
    if not prompt_source.is_file():
        raise FileNotFoundError(prompt_source)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tables", "figures", "scripts", "logs"):
        (output_dir / name).mkdir(exist_ok=True)

    full, subset = validate_and_filter(request_source)
    prompts = load_prompt_settings(source_report_root)
    (
        summary,
        cells,
        comparison,
        folds,
        fixed,
        selected,
        oof,
        bootstrap,
    ) = analyze(subset)
    (
        flex_registry,
        flex_comparison,
        flex_selected,
        flex_outer_choices,
        flex_nested_predictions,
        flex_bootstrap,
        flex_goodness,
    ) = analyze_mode_specific(subset)
    flex_coefficient_intervals = summarize_flex_bootstrap(flex_bootstrap)

    table_paths = {
        "qwen_query_last_requests.csv": subset,
        "model_mode_summary.csv": summary,
        "accuracy_cells.csv": cells,
        "candidate_comparison.csv": comparison,
        "candidate_fold_metrics.csv": folds,
        "fixed_log_separable_laws.csv": fixed,
        "selected_laws.csv": selected,
        "fixed_law_oof_predictions.csv": oof,
        "fixed_law_bootstrap_draws.csv": bootstrap,
        "qwen_query_last_prompt_settings.csv": prompts,
        "flex_candidate_registry.csv": flex_registry,
        "flex_candidate_comparison.csv": flex_comparison,
        "flex_selected_laws.csv": flex_selected,
        "flex_outer_choices.csv": flex_outer_choices,
        "flex_nested_predictions.csv": flex_nested_predictions,
        "flex_selected_bootstrap_draws.csv": flex_bootstrap,
        "flex_selected_coefficient_intervals.csv": (
            flex_coefficient_intervals
        ),
        "flex_goodness_of_fit.csv": flex_goodness,
    }
    for filename, frame in table_paths.items():
        write_csv(frame, output_dir / "tables" / filename)

    make_accuracy_heatmaps(cells, output_dir / "figures" / "fig01_accuracy_cells.png")
    make_coefficient_forest(
        fixed,
        "exact",
        output_dir / "figures" / "fig02_accuracy_coefficients.png",
        title="Fixed-law orders for exact accuracy odds",
        xlabels=(
            "βT: order of passage length in accuracy odds",
            "βN: order of needle count in accuracy odds",
        ),
    )
    make_accuracy_calibration(
        oof, fixed, output_dir / "figures" / "fig03_accuracy_calibration.png"
    )
    make_coefficient_forest(
        fixed,
        "bias",
        output_dir / "figures" / "fig04_bias_coefficients.png",
        title="Fixed-law coefficients for asinh signed bias",
        xlabels=(
            "βT: change per ln(T/5000)",
            "βN: change per ln(N/5)",
        ),
    )
    make_bias_surfaces(
        fixed, output_dir / "figures" / "fig05_bias_surfaces.png"
    )
    make_coefficient_forest(
        fixed,
        "absolute_error",
        output_dir / "figures" / "fig06_absolute_error_coefficients.png",
        title="Fixed-law exponents for log(1 + absolute error)",
        xlabels=(
            "βT: approximate T exponent of 1+|bias|",
            "βN: approximate N exponent of 1+|bias|",
        ),
    )
    make_candidate_gain_heatmap(
        selected, output_dir / "figures" / "fig07_candidate_gain.png"
    )
    make_mode_specific_accuracy_gof(
        flex_goodness,
        output_dir / "figures" / "fig02_mode_specific_accuracy_gof.png",
    )
    make_mode_specific_accuracy_scatter(
        flex_nested_predictions,
        flex_goodness,
        output_dir / "figures" / "fig03_mode_specific_accuracy_scatter.png",
    )
    make_mode_specific_accuracy_response(
        flex_nested_predictions,
        output_dir / "figures" / "fig04_mode_specific_accuracy_response.png",
    )
    make_mode_specific_error_gof(
        flex_goodness,
        output_dir / "figures" / "fig05_mode_specific_error_gof.png",
    )
    for legacy_name in (
        "fig02_accuracy_coefficients.png",
        "fig03_accuracy_calibration.png",
        "fig04_bias_coefficients.png",
        "fig05_bias_surfaces.png",
        "fig06_absolute_error_coefficients.png",
        "fig07_candidate_gain.png",
    ):
        (output_dir / "figures" / legacy_name).unlink(missing_ok=True)

    generated_at = utc_now()
    report_html = build_mode_specific_html(
        summary=summary,
        cells=cells,
        selected=flex_selected,
        goodness=flex_goodness,
        prompts=prompts,
        source_report_root=source_report_root,
        source_report_href=Path(
            os.path.relpath(source_report_root / "report.html", output_dir)
        ).as_posix(),
        generated_at=generated_at,
    )
    write_text(output_dir / "report.html", report_html)
    write_readme(output_dir, source_report_root)

    build_destination = (
        output_dir / "scripts" / "build_qwen_mode_specific_report.py"
    )
    shutil.copy2(Path(__file__).resolve(), build_destination)
    audit_source = Path(__file__).resolve().with_name(
        "qwen_mode_specific_audit.py"
    )
    if audit_source.is_file():
        shutil.copy2(
            audit_source,
            output_dir / "scripts" / "audit_qwen_mode_specific_report.py",
        )

    elapsed = time.perf_counter() - started
    fixed_summary = {}
    for target in TARGETS:
        target_frame = fixed[fixed["target"].eq(target)]
        fixed_summary[target] = {
            evidence: int(target_frame["evidence"].eq(evidence).sum())
            for evidence in (
                "supported",
                "suggestive",
                "ceiling_limited",
                "not_supported",
            )
        }
    manifest = {
        "schema_version": "qwen_query_last_mode_specific_empirical_law_v2",
        "created_at_utc": generated_at,
        "scope": {
            "models": MODELS,
            "prompt_modes": MODES,
            "query_order": "query_last",
            "requests": len(subset),
            "requests_per_stratum": 150,
            "target_passage_tokens": TARGET_LENGTHS,
            "needle_counts": NEEDLE_COUNTS,
            "seeds": SEEDS,
        },
        "source": {
            "eight_model_report_root": str(source_report_root),
            "request_table": {
                "path": str(request_source),
                "sha256": digest(request_source),
                "bytes": request_source.stat().st_size,
                "rows": len(full),
            },
            "prompt_table": {
                "path": str(prompt_source),
                "sha256": digest(prompt_source),
                "bytes": prompt_source.stat().st_size,
                "rows": len(pd.read_csv(prompt_source)),
            },
            "stimuli_sha256": "374dc935bf4c1403f705bb8b95ce686e5063647c83c609501e6f668e2331a5f1",
            "git_commit": "090d983819f06234cb135f6c499bf82e9a6de1c9",
        },
        "experiment": {
            "canonical_tokenizer": "Qwen/Qwen3-8B",
            "length_definition": (
                "post-insertion passage tokens before task/chat template"
            ),
            "density_definition": "rho = 1000*N/T",
            "hardware": "single NVIDIA H100 PCIe 81559 MiB",
            "engine": {
                "dtype": "bfloat16",
                "vllm": "0.25.1",
                "transformers": "5.14.1",
                "max_model_len": 16384,
                "tensor_parallel_size": 1,
                "model_batch_sizes": {
                    model: metadata["batch"]
                    for model, metadata in MODEL_METADATA.items()
                },
            },
            "model_metadata": MODEL_METADATA,
            "prompt": {
                "message_roles": ["user"],
                "query_last_template": (
                    "<passage>\\n[PASSAGE]\\n</passage>\\n\\n[TASK BLOCK]"
                ),
                "direct_and_native_task_block": DIRECT_TASK_BLOCK,
                "enumeration_task_block": ENUMERATION_TASK_BLOCK,
            },
            "decoding": {
                "direct": {
                    "enable_thinking": False,
                    "do_sample": False,
                    "max_new_tokens": 64,
                },
                "enumeration": {
                    "enable_thinking": False,
                    "do_sample": False,
                    "max_new_tokens": 1536,
                },
                "native_thinking": {
                    "enable_thinking": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "max_new_tokens": 4096,
                },
            },
        },
        "estimands": {
            "primary_exact": (
                "all requests; parse/format/truncation failures retained as zero"
            ),
            "signed_bias": (
                "predicted_count - N, conditional on parsed numeric output"
            ),
            "robust_bias_target": "asinh(signed_bias)",
            "absolute_error_target": "log1p(abs(signed_bias))",
        },
        "fixed_law": {
            "formula": (
                "g(y)=alpha+beta_T*ln(T/5000)+beta_N*ln(N/5)"
            ),
            "accuracy_link": "logistic",
            "continuous_link": "identity",
            "ridge_logistic": RIDGE_LOGISTIC,
            "ridge_ols": RIDGE_OLS,
            "bootstrap": {
                "clusters": "complete (T,N) cells",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": RNG_SEED,
            },
            "validation": [
                "leave-one-seed-out",
                "leave-one-N-level-out",
                "leave-one-T-level-out",
            ],
            "multiple_testing": "Benjamini-Hochberg within each target across 9 strata",
            "evidence_counts": fixed_summary,
        },
        "candidate_grid": [
            {
                "name": candidate.name,
                "label": candidate.label,
                "features": candidate.features,
                "complexity": candidate.complexity,
                "interpretation": candidate.interpretation,
            }
            for candidate in CANDIDATES
        ],
        "mode_specific_candidate_grid": [
            {
                "name": candidate.name,
                "label": candidate.label,
                "features": candidate.features,
                "complexity": candidate.complexity,
                "interpretation": candidate.interpretation,
            }
            for candidate in FLEX_CANDIDATES
        ],
        "mode_specific_estimators": {
            target: [
                {
                    "name": spec.name,
                    "estimator": spec.estimator,
                    "transform": spec.transform,
                    "ridge": spec.ridge,
                }
                for spec in build_flex_specs(target)
                if spec.candidate == "constant"
            ]
            for target in ("exact", "bias", "absolute_error")
        },
        "selection": {
            "primary_predictive": (
                "minimum inner leave-one-seed-out loss, repeated inside "
                "each outer held-out-seed fold"
            ),
            "parsimonious_sensitivity": (
                "one-standard-error rule on inner leave-one-seed-out loss"
            ),
            "outer_evaluation": "five-fold leave-one-seed-out",
            "candidate_count_per_target": len(build_flex_specs("exact")),
        },
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            "tables": list(table_paths),
            "figures": [
                "fig01_accuracy_cells.png",
                "fig02_mode_specific_accuracy_gof.png",
                "fig03_mode_specific_accuracy_scatter.png",
                "fig04_mode_specific_accuracy_response.png",
                "fig05_mode_specific_error_gof.png",
            ],
        },
        "elapsed_seconds": elapsed,
        "raw_or_frozen_artifacts_modified": False,
    }
    # scipy's public version lives on the package, not scipy.stats.
    import scipy

    manifest["software_versions"]["scipy"] = scipy.__version__
    write_text(
        output_dir / "analysis_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    write_text(
        output_dir / "logs" / "build_log.json",
        json.dumps(
            {
                "status": "PASS",
                "created_at_utc": generated_at,
                "elapsed_seconds": elapsed,
                "request_rows": len(subset),
                "strata": 9,
                "bootstrap_rows": len(bootstrap),
                "flex_candidate_rows": len(flex_comparison),
                "flex_nested_prediction_rows": len(
                    flex_nested_predictions
                ),
                "flex_bootstrap_rows": len(flex_bootstrap),
                "table_rows": {
                    filename: len(frame) for filename, frame in table_paths.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    checksum_entries = write_checksums(output_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "request_rows": len(subset),
                "strata": 9,
                "fixed_law_rows": len(fixed),
                "candidate_rows": len(comparison),
                "bootstrap_rows": len(bootstrap),
                "flex_candidate_rows": len(flex_comparison),
                "flex_nested_prediction_rows": len(
                    flex_nested_predictions
                ),
                "flex_bootstrap_rows": len(flex_bootstrap),
                "figures": 5,
                "checksum_entries": checksum_entries,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
