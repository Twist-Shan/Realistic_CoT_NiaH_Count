#!/usr/bin/env python3
"""Fit length/needle empirical laws without using model parameter scale."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit


RUN = Path(
    "/lambda/nfs/Twist-CoT-Count-Multi-Model/runs/"
    "realistic_niah_v1/six_models_formal_20260723T194300Z"
)
SOURCE_ANALYSIS = RUN / "analysis" / "empirical_law_v1"
SOURCE = SOURCE_ANALYSIS / "tables" / "request_level.csv"
SOURCE_INTEGRITY = SOURCE_ANALYSIS / "artifact_integrity.json"
SOURCE_MANIFEST = SOURCE_ANALYSIS / "analysis_manifest.json"
OUT = RUN / "analysis" / "empirical_law_no_model_size_v1"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
METRICS = OUT / "metrics"
PLAN = OUT / "analysis_plan.md"
STATE = OUT / "state.json"
LOG = OUT / "run.log"
INTEGRITY = OUT / "artifact_integrity.json"
MANIFEST = OUT / "analysis_manifest.json"

EXPECTED_FILESYSTEM_ID = "c8d6df94b8504c14a4ba5e05e3119723"
EXPECTED_TOTAL = 6300
RANDOM_SEED = 20260724
BOOTSTRAP_REPLICATES = 200
RIDGE = 1e-4

MODEL_LEVELS = [
    "Qwen3-8B",
    "Qwen3-1.7B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "OLMo-Hybrid-7B",
    "Llama3.1-8B",
    "Llama3.2-3B",
]
CATEGORY_LEVELS = {
    "model_label": MODEL_LEVELS,
    "prompt_mode": ["direct", "enumeration", "native_thinking"],
    "query_order": ["query_first", "query_last"],
    "length_cat": ["2000", "5000", "10000"],
    "needles_cat": ["1", "2", "3", "4", "5", "6", "8", "10", "20", "30"],
}

PLAN_TEXT = """# Empirical law without model parameter scale

## Scope and estimands

This is a separate analysis. It does not modify the experiment, the original
request JSONL files, or `analysis/empirical_law_v1/`.

No parameter-count or model-scale variable is permitted in any design matrix.
Two views are reported:

1. pooled descriptive fits containing only length and needle coordinates;
2. shared-slope fits using model identity as a categorical nuisance intercept,
   together with prompt mode and query order. Model identity is not converted
   into, ordered by, or otherwise represented as model size.

The primary target is request-level exact correctness over all 6,300 requests.
Parse failures, format failures, and truncations remain incorrect. The
secondary error target is absolute count error conditional on a successfully
parsed numeric prediction; its parse coverage is reported beside every result
and it cannot replace the primary target.

## Frozen coordinate comparisons

- `log_length = log2(target_passage_tokens / 5000)`
- `log_needles = log2(num_needles / 5)`
- `log_density = log2(needles per 1k passage tokens)`
- a low-order `log_length * log_needles` interaction
- categorical length and needle levels as a finite flexible check
- model-specific length/needle slopes as a heterogeneity diagnostic

Density-only fits test whether density is sufficient. Length+density and
length+needle fits are both retained even though their log-linear coordinate
spans are algebraically equivalent; this makes the density interpretation
explicit.

## Validation and selection

- five leave-one-seed-out folds, retaining complete same-seed stimuli together;
- held-out log loss, Brier score, calibration and ECE for exact correctness;
- held-out MAE/RMSE for absolute error;
- 200 bootstrap replicates clustered by complete `stimulus_id`;
- per-model slope diagnostics;
- all candidates and failures are retained, with no point deletion.

The parsimonious shared-slope length+needle law is the interpretive target.
Predictive candidate comparison is reported separately. A more complex
candidate is not preferred solely for a small in-sample improvement.
"""


@dataclass(frozen=True)
class Candidate:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...] = ()
    description: str = ""


CANDIDATES = [
    Candidate(
        "pooled_log_length_needles",
        ("log_length", "log_needles"),
        (),
        description="Pooled descriptive length and needle slopes",
    ),
    Candidate(
        "pooled_log_density",
        ("log_density",),
        (),
        description="Pooled descriptive density-only slope",
    ),
    Candidate(
        "controls_only",
        (),
        ("model_label", "prompt_mode", "query_order"),
        description="Categorical nuisance baselines only",
    ),
    Candidate(
        "log_length_needles_model_fe",
        ("log_length", "log_needles"),
        ("model_label", "prompt_mode", "query_order"),
        description="Shared length/needle slopes with categorical model intercepts",
    ),
    Candidate(
        "density_model_fe",
        ("log_density",),
        ("model_label", "prompt_mode", "query_order"),
        description="Density-only slope with categorical model intercepts",
    ),
    Candidate(
        "log_length_density_model_fe",
        ("log_length", "log_density"),
        ("model_label", "prompt_mode", "query_order"),
        description="Length plus density coordinates with model intercepts",
    ),
    Candidate(
        "log_length_needles_interaction_model_fe",
        ("log_length", "log_needles", "log_length_x_log_needles"),
        ("model_label", "prompt_mode", "query_order"),
        description="Shared log-linear slopes plus one interaction",
    ),
    Candidate(
        "categorical_length_needles_model_fe",
        (),
        (
            "model_label",
            "prompt_mode",
            "query_order",
            "length_cat",
            "needles_cat",
        ),
        description="Flexible additive categorical response surface",
    ),
    Candidate(
        "model_specific_length_needles_slopes",
        ("log_length", "log_needles"),
        ("model_label", "prompt_mode", "query_order"),
        interactions=(
            ("log_length", "model_label"),
            ("log_needles", "model_label"),
        ),
        description="Model-specific slope heterogeneity diagnostic",
    ),
]

INTERPRETIVE = "log_length_needles_model_fe"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(message: str) -> None:
    stamp = utc_now()
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{message}\n")


def set_state(phase: str, **extra: Any) -> None:
    write_json(
        STATE,
        {
            "status": phase,
            "phase": phase,
            "updated_at_utc": utc_now(),
            "model_size_used": False,
            **extra,
        },
    )


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    prior_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    prior_integrity = json.loads(SOURCE_INTEGRITY.read_text(encoding="utf-8"))
    if prior_manifest.get("filesystem_id") != EXPECTED_FILESYSTEM_ID:
        raise ValueError("Prior analysis filesystem ID mismatch")
    indexed = {
        item["path"]: item for item in prior_integrity.get("files", [])
    }
    source_key = "tables/request_level.csv"
    if source_key not in indexed:
        raise ValueError("Prior integrity file does not index request_level.csv")
    item = indexed[source_key]
    if SOURCE.stat().st_size != item["bytes"] or sha256(SOURCE) != item["sha256"]:
        raise ValueError("Verified source table no longer matches its SHA256")

    frame = pd.read_csv(SOURCE)
    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} rows, found {len(frame)}")
    if frame["request_id"].nunique() != EXPECTED_TOTAL:
        raise ValueError("Request IDs are missing or duplicated")
    if frame["exact_correct"].isna().any():
        raise ValueError("Missing exact correctness target")
    if set(frame["seed"].astype(int)) != {1234, 1235, 1236, 1237, 1238}:
        raise ValueError("Unexpected seed set")
    if not set(frame["exact_correct"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("Exact correctness is not binary")

    frame["length_cat"] = frame["target_passage_tokens"].astype(int).astype(str)
    frame["needles_cat"] = frame["num_needles"].astype(int).astype(str)
    frame["log_length"] = np.log2(
        frame["target_passage_tokens"].astype(float) / 5000.0
    )
    frame["log_needles"] = np.log2(frame["num_needles"].astype(float) / 5.0)
    frame["log_density"] = np.log2(frame["density_per_1k"].astype(float))
    frame["log_length_x_log_needles"] = (
        frame["log_length"] * frame["log_needles"]
    )
    frame["absolute_error"] = pd.to_numeric(
        frame["absolute_error"], errors="coerce"
    )
    frame["normalized_absolute_error"] = pd.to_numeric(
        frame["normalized_absolute_error"], errors="coerce"
    )
    parsed = frame["parse_success"].astype(int).eq(1)
    if frame.loc[parsed, "absolute_error"].isna().any():
        raise ValueError("Parsed rows contain missing absolute error")
    if frame.loc[~parsed, "absolute_error"].notna().any():
        raise ValueError("Unparsed rows unexpectedly contain absolute error")

    source_meta = {
        "path": str(SOURCE),
        "bytes": SOURCE.stat().st_size,
        "sha256": sha256(SOURCE),
        "prior_analysis_manifest": str(SOURCE_MANIFEST),
        "prior_analysis_manifest_sha256": sha256(SOURCE_MANIFEST),
        "prior_artifact_integrity": str(SOURCE_INTEGRITY),
        "prior_artifact_integrity_sha256": sha256(SOURCE_INTEGRITY),
    }
    return frame, source_meta


def build_design(
    frame: pd.DataFrame, candidate: Candidate
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=float)]
    names = ["Intercept"]
    for name in candidate.numeric:
        if "model_scale" in name or "params" in name:
            raise ValueError("Model-scale feature is forbidden")
        columns.append(frame[name].to_numpy(dtype=float))
        names.append(name)
    for category in candidate.categorical:
        values = frame[category].astype(str).to_numpy()
        levels = CATEGORY_LEVELS[category]
        for level in levels[1:]:
            indicator = (values == level).astype(float)
            columns.append(indicator)
            names.append(f"{category}[{level}]")
    for numeric, category in candidate.interactions:
        values = frame[category].astype(str).to_numpy()
        numeric_values = frame[numeric].to_numpy(dtype=float)
        for level in CATEGORY_LEVELS[category][1:]:
            columns.append(numeric_values * (values == level).astype(float))
            names.append(f"{numeric}:{category}[{level}]")
    matrix = np.column_stack(columns)
    keep = [0] + [
        index
        for index in range(1, matrix.shape[1])
        if np.nanmax(matrix[:, index]) != np.nanmin(matrix[:, index])
    ]
    matrix = matrix[:, keep]
    names = [names[index] for index in keep]
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite design for {candidate.name}")
    if any("model_scale" in name or "params" in name for name in names):
        raise AssertionError("Model size entered a design matrix")
    return matrix, names


def fit_logistic(
    matrix: np.ndarray, outcome: np.ndarray, ridge: float = RIDGE
) -> dict[str, Any]:
    y = outcome.astype(float)
    prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    initial = np.zeros(matrix.shape[1], dtype=float)
    initial[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.ones(matrix.shape[1], dtype=float)
    penalty[0] = 0.0

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = matrix @ beta
        loss = float(np.logaddexp(0.0, linear).sum() - y @ linear)
        loss += 0.5 * ridge * float(np.sum(penalty * beta**2))
        gradient = matrix.T @ (expit(linear) - y)
        gradient += ridge * penalty * beta
        return loss, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-11},
    )
    beta = result.x
    probability = np.clip(expit(matrix @ beta), 1e-8, 1 - 1e-8)
    weights = probability * (1.0 - probability)
    hessian = matrix.T @ (weights[:, None] * matrix)
    hessian += ridge * np.diag(penalty)
    covariance = np.linalg.pinv(hessian)
    log_likelihood = float(
        np.sum(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
    )
    return {
        "beta": beta,
        "probability": probability,
        "covariance": covariance,
        "converged": bool(result.success),
        "message": str(result.message),
        "log_likelihood": log_likelihood,
    }


def fit_ols(matrix: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    beta, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
    prediction = matrix @ beta
    residual = outcome - prediction
    degrees = max(len(outcome) - int(rank), 1)
    sigma2 = float(residual @ residual / degrees)
    covariance = sigma2 * np.linalg.pinv(matrix.T @ matrix)
    return {
        "beta": beta,
        "prediction": prediction,
        "covariance": covariance,
        "rank": int(rank),
        "converged": True,
    }


def binary_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = outcome.astype(float)
    p = np.clip(probability.astype(float), 1e-8, 1 - 1e-8)
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((y - p) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    assignments = np.clip(np.digitize(p, bins, right=True) - 1, 0, 9)
    ece = 0.0
    for index in range(10):
        mask = assignments == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    logits = np.log(p / (1.0 - p))
    calibration_matrix = np.column_stack([np.ones(len(p)), logits])
    calibration = fit_logistic(calibration_matrix, y, ridge=1e-8)
    return {
        "log_loss": log_loss,
        "brier": brier,
        "ece": float(ece),
        "calibration_intercept": float(calibration["beta"][0]),
        "calibration_slope": float(calibration["beta"][1]),
    }


def evaluate_binary_candidates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    y = frame["exact_correct"].to_numpy(dtype=float)
    seeds = sorted(int(value) for value in frame["seed"].unique())
    comparisons = []
    folds = []
    oof_by_candidate: dict[str, np.ndarray] = {}
    full_fits: dict[str, Any] = {}
    for candidate in CANDIDATES:
        matrix, names = build_design(frame, candidate)
        oof = np.full(len(frame), np.nan, dtype=float)
        fold_losses = []
        convergence = True
        for seed in seeds:
            test = frame["seed"].astype(int).to_numpy() == seed
            train = ~test
            fitted = fit_logistic(matrix[train], y[train])
            convergence = convergence and fitted["converged"]
            predicted = np.clip(
                expit(matrix[test] @ fitted["beta"]), 1e-8, 1 - 1e-8
            )
            oof[test] = predicted
            metrics = binary_metrics(y[test], predicted)
            fold_losses.append(metrics["log_loss"])
            folds.append(
                {
                    "candidate": candidate.name,
                    "held_out_seed": seed,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "converged": fitted["converged"],
                    **metrics,
                }
            )
        if np.isnan(oof).any():
            raise RuntimeError(f"Missing OOF predictions for {candidate.name}")
        metrics = binary_metrics(y, oof)
        full = fit_logistic(matrix, y)
        full_fits[candidate.name] = {
            "fit": full,
            "matrix": matrix,
            "feature_names": names,
        }
        n_parameters = matrix.shape[1]
        aic = 2 * n_parameters - 2 * full["log_likelihood"]
        bic = math.log(len(y)) * n_parameters - 2 * full["log_likelihood"]
        comparisons.append(
            {
                "candidate": candidate.name,
                "description": candidate.description,
                "n_parameters": n_parameters,
                "converged_all_folds": convergence,
                "cv_log_loss_mean": float(np.mean(fold_losses)),
                "cv_log_loss_se": float(
                    np.std(fold_losses, ddof=1) / math.sqrt(len(fold_losses))
                ),
                "cv_brier": metrics["brier"],
                "cv_ece": metrics["ece"],
                "cv_calibration_intercept": metrics["calibration_intercept"],
                "cv_calibration_slope": metrics["calibration_slope"],
                "full_log_likelihood": full["log_likelihood"],
                "aic": aic,
                "bic": bic,
            }
        )
        oof_by_candidate[candidate.name] = oof
        log(
            f"binary {candidate.name}: cv log loss "
            f"{metrics['log_loss']:.6f}, converged={convergence}"
        )
    return (
        pd.DataFrame(comparisons).sort_values("cv_log_loss_mean"),
        pd.DataFrame(folds),
        oof_by_candidate,
        full_fits,
    )


def transform_error_target(
    frame: pd.DataFrame, target: str
) -> tuple[np.ndarray, str]:
    absolute = frame["absolute_error"].to_numpy(dtype=float)
    if target == "absolute_error":
        return absolute, "raw absolute count error"
    if target == "log1p_absolute_error":
        return np.log1p(absolute), "log1p absolute count error"
    if target == "normalized_absolute_error":
        return frame["normalized_absolute_error"].to_numpy(dtype=float), (
            "absolute count error divided by true needle count"
        )
    raise KeyError(target)


def error_to_absolute_units(
    prediction: np.ndarray, frame: pd.DataFrame, target: str
) -> np.ndarray:
    if target == "absolute_error":
        return np.maximum(prediction, 0.0)
    if target == "log1p_absolute_error":
        return np.maximum(np.expm1(prediction), 0.0)
    if target == "normalized_absolute_error":
        return np.maximum(prediction, 0.0) * frame["num_needles"].to_numpy(
            dtype=float
        )
    raise KeyError(target)


def evaluate_error_candidates(
    parsed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seeds = sorted(int(value) for value in parsed["seed"].unique())
    targets = [
        "absolute_error",
        "log1p_absolute_error",
        "normalized_absolute_error",
    ]
    comparisons = []
    folds = []
    full_fits: dict[str, Any] = {}
    for target in targets:
        outcome, target_description = transform_error_target(parsed, target)
        absolute_observed = parsed["absolute_error"].to_numpy(dtype=float)
        for candidate in CANDIDATES:
            matrix, names = build_design(parsed, candidate)
            oof = np.full(len(parsed), np.nan, dtype=float)
            fold_fit_mae = []
            fold_abs_mae = []
            for seed in seeds:
                test = parsed["seed"].astype(int).to_numpy() == seed
                train = ~test
                fitted = fit_ols(matrix[train], outcome[train])
                predicted = matrix[test] @ fitted["beta"]
                oof[test] = predicted
                predicted_abs = error_to_absolute_units(
                    predicted, parsed.loc[test], target
                )
                fit_mae = float(np.mean(np.abs(outcome[test] - predicted)))
                abs_mae = float(
                    np.mean(np.abs(absolute_observed[test] - predicted_abs))
                )
                fold_fit_mae.append(fit_mae)
                fold_abs_mae.append(abs_mae)
                folds.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "held_out_seed": seed,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "fit_scale_mae": fit_mae,
                        "absolute_unit_mae": abs_mae,
                        "absolute_unit_rmse": float(
                            np.sqrt(
                                np.mean(
                                    (absolute_observed[test] - predicted_abs) ** 2
                                )
                            )
                        ),
                    }
                )
            predicted_abs = error_to_absolute_units(oof, parsed, target)
            full = fit_ols(matrix, outcome)
            full_fits[f"{target}::{candidate.name}"] = {
                "fit": full,
                "matrix": matrix,
                "feature_names": names,
            }
            comparisons.append(
                {
                    "target": target,
                    "target_description": target_description,
                    "candidate": candidate.name,
                    "description": candidate.description,
                    "n_parameters": matrix.shape[1],
                    "n_parsed": len(parsed),
                    "parse_coverage": len(parsed) / EXPECTED_TOTAL,
                    "cv_fit_scale_mae": float(np.mean(np.abs(outcome - oof))),
                    "cv_fit_scale_rmse": float(
                        np.sqrt(np.mean((outcome - oof) ** 2))
                    ),
                    "cv_absolute_unit_mae": float(
                        np.mean(np.abs(absolute_observed - predicted_abs))
                    ),
                    "cv_absolute_unit_mae_se": float(
                        np.std(fold_abs_mae, ddof=1) / math.sqrt(len(fold_abs_mae))
                    ),
                    "cv_absolute_unit_rmse": float(
                        np.sqrt(np.mean((absolute_observed - predicted_abs) ** 2))
                    ),
                    "cv_absolute_unit_median_ae": float(
                        np.median(np.abs(absolute_observed - predicted_abs))
                    ),
                }
            )
            log(
                f"error {target} {candidate.name}: "
                f"absolute-unit MAE {comparisons[-1]['cv_absolute_unit_mae']:.6f}"
            )
    return (
        pd.DataFrame(comparisons).sort_values(
            ["target", "cv_absolute_unit_mae"]
        ),
        pd.DataFrame(folds),
        full_fits,
    )


def coefficient_table(
    names: list[str], fit: dict[str, Any], model_type: str
) -> pd.DataFrame:
    beta = fit["beta"]
    standard_error = np.sqrt(np.maximum(np.diag(fit["covariance"]), 0.0))
    table = pd.DataFrame(
        {
            "feature": names,
            "estimate": beta,
            "standard_error": standard_error,
            "ci95_low_hessian": beta - 1.96 * standard_error,
            "ci95_high_hessian": beta + 1.96 * standard_error,
            "model_type": model_type,
        }
    )
    if model_type == "logistic":
        table["odds_ratio"] = np.exp(table["estimate"])
    return table


def clustered_bootstrap(
    frame: pd.DataFrame,
    candidate: Candidate,
    outcome: np.ndarray,
    kind: str,
) -> pd.DataFrame:
    matrix, names = build_design(frame, candidate)
    clusters = frame["stimulus_id"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    indices_by_cluster = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters
    }
    rng = np.random.default_rng(RANDOM_SEED)
    estimates = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(
            unique_clusters, size=len(unique_clusters), replace=True
        )
        indices = np.concatenate(
            [indices_by_cluster[cluster] for cluster in sampled]
        )
        if kind == "logistic":
            fitted = fit_logistic(matrix[indices], outcome[indices])
            if not fitted["converged"]:
                continue
        elif kind == "ols":
            fitted = fit_ols(matrix[indices], outcome[indices])
        else:
            raise KeyError(kind)
        estimates.append(fitted["beta"])
    if len(estimates) < int(0.9 * BOOTSTRAP_REPLICATES):
        raise RuntimeError(f"Too few converged {kind} bootstrap replicates")
    values = np.vstack(estimates)
    return pd.DataFrame(
        {
            "feature": names,
            "bootstrap_mean": values.mean(axis=0),
            "bootstrap_ci95_low": np.percentile(values, 2.5, axis=0),
            "bootstrap_ci95_high": np.percentile(values, 97.5, axis=0),
            "bootstrap_replicates": len(estimates),
        }
    )


def make_aggregate_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def aggregate(grouped: Any) -> pd.DataFrame:
        records = []
        for keys, part in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            parsed = part[part["parse_success"].astype(int).eq(1)]
            record = {
                "requests": len(part),
                "accuracy": float(part["exact_correct"].mean()),
                "parse_success_rate": float(part["parse_success"].mean()),
                "truncation_rate": float(part["truncated"].mean()),
                "parsed_requests": len(parsed),
                "mean_absolute_error_parsed": (
                    float(parsed["absolute_error"].mean()) if len(parsed) else None
                ),
                "median_absolute_error_parsed": (
                    float(parsed["absolute_error"].median()) if len(parsed) else None
                ),
            }
            for name, value in zip(grouped.grouper.names, keys):
                record[name] = value
            records.append(record)
        return pd.DataFrame(records)

    return {
        "pooled_length_needles": aggregate(
            frame.groupby(
                ["target_passage_tokens", "num_needles"], sort=True, dropna=False
            )
        ),
        "model_length_needles": aggregate(
            frame.groupby(
                ["model_label", "target_passage_tokens", "num_needles"],
                sort=True,
                dropna=False,
            )
        ),
        "pooled_density": aggregate(
            frame.groupby(["density_per_1k"], sort=True, dropna=False)
        ),
        "model_density": aggregate(
            frame.groupby(
                ["model_label", "density_per_1k"], sort=True, dropna=False
            )
        ),
    }


def per_model_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    candidate = Candidate(
        "within_model",
        ("log_length", "log_needles"),
        ("prompt_mode", "query_order"),
    )
    rows = []
    for model in MODEL_LEVELS:
        part = frame[frame["model_label"] == model].copy()
        matrix, names = build_design(part, candidate)
        fitted = fit_logistic(
            matrix, part["exact_correct"].to_numpy(dtype=float)
        )
        coefficients = coefficient_table(names, fitted, "logistic")
        for feature in ["log_length", "log_needles"]:
            row = coefficients[coefficients["feature"] == feature].iloc[0]
            rows.append(
                {
                    "model_label": model,
                    "requests": len(part),
                    "observed_accuracy": float(part["exact_correct"].mean()),
                    "feature": feature,
                    "estimate": float(row["estimate"]),
                    "standard_error": float(row["standard_error"]),
                    "ci95_low": float(row["ci95_low_hessian"]),
                    "ci95_high": float(row["ci95_high_hessian"]),
                    "odds_ratio_per_doubling": float(row["odds_ratio"]),
                    "converged": fitted["converged"],
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    frame: pd.DataFrame,
    aggregates: dict[str, pd.DataFrame],
    binary_comparison: pd.DataFrame,
    error_comparison: pd.DataFrame,
    oof: np.ndarray,
) -> None:
    plt.style.use("default")

    ordered = binary_comparison.sort_values("cv_log_loss_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ordered["candidate"], ordered["cv_log_loss_mean"], color="#4472C4")
    ax.set_xlabel("Leave-one-seed-out log loss (lower is better)")
    ax.set_title("Accuracy candidates without model parameter scale")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_candidate_cv.png", dpi=180)
    plt.close(fig)

    pooled = aggregates["pooled_length_needles"].copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for length, part in pooled.groupby("target_passage_tokens"):
        part = part.sort_values("num_needles")
        ax.plot(
            part["num_needles"],
            part["accuracy"],
            marker="o",
            label=f"{int(length)} tokens",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 20, 30])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Needle count")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Pooled accuracy by length and needle count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_length_needles.png", dpi=180)
    plt.close(fig)

    matrix = pooled.pivot(
        index="target_passage_tokens", columns="num_needles", values="accuracy"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(int(value)) for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(int(value)) for value in matrix.index])
    ax.set_xlabel("Needle count")
    ax.set_ylabel("Passage tokens")
    ax.set_title("Pooled exact accuracy response surface")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_length_needles_heatmap.png", dpi=180)
    plt.close(fig)

    model_density = aggregates["model_density"].copy()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODEL_LEVELS):
        part = model_density[model_density["model_label"] == model].sort_values(
            "density_per_1k"
        )
        ax.plot(part["density_per_1k"], part["accuracy"], marker="o", markersize=3)
        ax.set_xscale("log", base=2)
        ax.set_title(model)
        ax.set_ylim(0, 1)
    for ax in axes[-1, :]:
        ax.set_xlabel("Needles per 1k tokens")
    for ax in axes[:, 0]:
        ax.set_ylabel("Exact accuracy")
    fig.suptitle("Accuracy versus needle density, shown separately by model")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_vs_density_by_model.png", dpi=180)
    plt.close(fig)

    parsed_pooled = pooled[pooled["parsed_requests"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for length, part in parsed_pooled.groupby("target_passage_tokens"):
        part = part.sort_values("num_needles")
        ax.plot(
            part["num_needles"],
            part["mean_absolute_error_parsed"],
            marker="o",
            label=f"{int(length)} tokens",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Needle count")
    ax.set_ylabel("Mean absolute count error (parsed outputs)")
    ax.set_title("Conditional absolute error by length and needle count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "absolute_error_by_length_needles.png", dpi=180)
    plt.close(fig)

    error_plot = error_comparison[
        error_comparison["target"] == "log1p_absolute_error"
    ].sort_values("cv_absolute_unit_mae")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        error_plot["candidate"],
        error_plot["cv_absolute_unit_mae"],
        color="#70AD47",
    )
    ax.set_xlabel("Held-out MAE in count units (lower is better)")
    ax.set_title("Absolute-error candidates without model parameter scale")
    fig.tight_layout()
    fig.savefig(FIGURES / "absolute_error_candidate_cv.png", dpi=180)
    plt.close(fig)

    calibration = pd.DataFrame(
        {
            "observed": frame["exact_correct"].astype(float),
            "predicted": oof,
        }
    )
    calibration["bin"] = pd.qcut(
        calibration["predicted"], q=10, duplicates="drop"
    )
    curve = calibration.groupby("bin", observed=False).agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        n=("observed", "size"),
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(curve["predicted"], curve["observed"], marker="o", color="#C55A11")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted accuracy")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration: shared length/needle law")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_calibration.png", dpi=180)
    plt.close(fig)


def render_readme(
    summary: dict[str, Any],
    accuracy_coefficients: pd.DataFrame,
    error_coefficients: pd.DataFrame,
) -> str:
    relation = summary["accuracy"]["interpretive_relation"]
    error = summary["absolute_error"]
    coefficient_lines = []
    for feature in ["log_length", "log_needles"]:
        row = accuracy_coefficients[
            accuracy_coefficients["feature"] == feature
        ].iloc[0]
        coefficient_lines.append(
            f"- `{feature}`: {row['estimate']:.4f}, clustered-bootstrap 95% CI "
            f"[{row['bootstrap_ci95_low']:.4f}, "
            f"{row['bootstrap_ci95_high']:.4f}], odds ratio "
            f"{math.exp(row['estimate']):.3f} per doubling."
        )
    error_lines = []
    for feature in ["log_length", "log_needles"]:
        row = error_coefficients[error_coefficients["feature"] == feature].iloc[0]
        error_lines.append(
            f"- `{feature}`: {row['estimate']:.4f}, clustered-bootstrap 95% CI "
            f"[{row['bootstrap_ci95_low']:.4f}, "
            f"{row['bootstrap_ci95_high']:.4f}] on log1p absolute error."
        )
    return f"""# Length/needle empirical law without model size

## Result

No parameter-count or model-size feature was used. Model identity appears only
as a categorical nuisance intercept in the shared-slope analysis, and fully
pooled as well as per-model summaries are also provided.

The primary accuracy law is:

`logit(P(exact correct)) = nuisance intercepts + b_L log2(L/5000) + b_N log2(N/5)`

{chr(10).join(coefficient_lines)}

Held-out log loss is `{relation['cv_log_loss']:.6f}` versus
`{relation['controls_only_cv_log_loss']:.6f}` for categorical nuisance
intercepts alone. The density-only law has held-out log loss
`{relation['density_only_cv_log_loss']:.6f}`.

Length+density and length+needle coordinates span the same two-dimensional
log-linear surface. Density alone imposes a stronger constraint and is tested
separately rather than assumed.

## Conditional absolute error

Absolute count error is defined only for successfully parsed numeric outputs.
Coverage is `{error['parsed_requests']}/{EXPECTED_TOTAL}`
(`{error['parse_coverage']:.2%}`). Parse/format/truncation failures remain
failures in the primary accuracy analysis and are not silently dropped from it.

{chr(10).join(error_lines)}

The best held-out absolute-error configuration is
`{error['best_configuration']}` with count-unit MAE
`{error['best_cv_absolute_unit_mae']:.6f}`.

## Interpretation limits

These are associational response surfaces over the registered 2K--10K token,
1--30 needle grid. Model identity, prompt mode and query order are not fully
crossed, so coefficients are not causal effects. Density alone cannot in
general distinguish changing needle count from changing length. Absolute-error
fits are conditional diagnostics and should be read together with parse
coverage and exact accuracy.

## Reproduction

Run `empirical_law_no_size.py` on the verified
`analysis/empirical_law_v1/tables/request_level.csv`. See
`analysis_plan.md`, `analysis_manifest.json`, `metrics/summary.json`, and
`artifact_integrity.json` for the frozen design, source SHA256, conclusions,
and output hashes.
"""


def main() -> None:
    for directory in (OUT, TABLES, FIGURES, METRICS):
        directory.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_text(encoding="utf-8") != PLAN_TEXT:
        raise RuntimeError("Existing analysis plan differs from frozen plan")
    PLAN.write_text(PLAN_TEXT, encoding="utf-8")
    LOG.write_text("", encoding="utf-8")
    set_state("running")
    log("no-model-size analysis started")

    frame, source_meta = load_source()
    log("source loaded and SHA256 verified")
    if "model_scale_b" not in frame.columns:
        log("source does not contain model scale column; still enforcing feature ban")

    output_columns = [
        "request_id",
        "stimulus_id",
        "seed",
        "model_label",
        "target_passage_tokens",
        "num_needles",
        "density_per_1k",
        "prompt_mode",
        "query_order",
        "exact_correct",
        "parse_success",
        "format_failure",
        "truncated",
        "parse_status",
        "absolute_error",
        "normalized_absolute_error",
        "log_length",
        "log_needles",
        "log_density",
    ]

    binary_comparison, binary_folds, oof_by_candidate, binary_fits = (
        evaluate_binary_candidates(frame)
    )
    binary_comparison.to_csv(
        TABLES / "accuracy_candidate_comparison.csv", index=False
    )
    binary_folds.to_csv(TABLES / "accuracy_fold_metrics.csv", index=False)

    interpretive_candidate = next(
        candidate for candidate in CANDIDATES if candidate.name == INTERPRETIVE
    )
    interpretive_full = binary_fits[INTERPRETIVE]
    accuracy_coefficients = coefficient_table(
        interpretive_full["feature_names"],
        interpretive_full["fit"],
        "logistic",
    )
    accuracy_bootstrap = clustered_bootstrap(
        frame,
        interpretive_candidate,
        frame["exact_correct"].to_numpy(dtype=float),
        "logistic",
    )
    accuracy_coefficients = accuracy_coefficients.merge(
        accuracy_bootstrap, on="feature", how="left"
    )
    accuracy_coefficients.to_csv(
        TABLES / "accuracy_interpretive_coefficients.csv", index=False
    )

    frame["no_size_oof_probability"] = oof_by_candidate[INTERPRETIVE]
    output_columns.append("no_size_oof_probability")
    frame[output_columns].to_csv(TABLES / "request_level_no_size.csv", index=False)

    parsed = frame[frame["parse_success"].astype(int).eq(1)].copy()
    error_comparison, error_folds, error_fits = evaluate_error_candidates(parsed)
    error_comparison.to_csv(
        TABLES / "absolute_error_candidate_comparison.csv", index=False
    )
    error_folds.to_csv(TABLES / "absolute_error_fold_metrics.csv", index=False)

    error_key = f"log1p_absolute_error::{INTERPRETIVE}"
    error_full = error_fits[error_key]
    error_coefficients = coefficient_table(
        error_full["feature_names"], error_full["fit"], "ols_log1p_error"
    )
    log1p_error = np.log1p(parsed["absolute_error"].to_numpy(dtype=float))
    error_bootstrap = clustered_bootstrap(
        parsed,
        interpretive_candidate,
        log1p_error,
        "ols",
    )
    error_coefficients = error_coefficients.merge(
        error_bootstrap, on="feature", how="left"
    )
    error_coefficients.to_csv(
        TABLES / "absolute_error_interpretive_coefficients.csv", index=False
    )

    aggregates = make_aggregate_tables(frame)
    for name, table in aggregates.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)
    slopes = per_model_slopes(frame)
    slopes.to_csv(TABLES / "per_model_accuracy_slopes.csv", index=False)

    binary_lookup = binary_comparison.set_index("candidate")
    interpretive_metrics = binary_lookup.loc[INTERPRETIVE]
    density_metrics = binary_lookup.loc["density_model_fe"]
    length_density_metrics = binary_lookup.loc["log_length_density_model_fe"]
    controls_metrics = binary_lookup.loc["controls_only"]
    best_binary = binary_comparison.iloc[0]
    best_error = error_comparison.sort_values("cv_absolute_unit_mae").iloc[0]

    length_row = accuracy_coefficients[
        accuracy_coefficients["feature"] == "log_length"
    ].iloc[0]
    needles_row = accuracy_coefficients[
        accuracy_coefficients["feature"] == "log_needles"
    ].iloc[0]
    density_sufficiency_gap = float(
        density_metrics["cv_log_loss_mean"]
        - interpretive_metrics["cv_log_loss_mean"]
    )

    summary = {
        "status": "complete",
        "model_size_used": False,
        "requests": len(frame),
        "unique_request_ids": int(frame["request_id"].nunique()),
        "accuracy": {
            "target": "exact_correct",
            "failures_retained_as_incorrect": True,
            "observed_accuracy": float(frame["exact_correct"].mean()),
            "interpretive_candidate": INTERPRETIVE,
            "interpretive_formula": (
                "logit(P(exact_correct)) = categorical model/prompt/order "
                "intercepts + b_L log2(L/5000) + b_N log2(N/5)"
            ),
            "interpretive_relation": {
                "length_coefficient": float(length_row["estimate"]),
                "length_bootstrap_ci95": [
                    float(length_row["bootstrap_ci95_low"]),
                    float(length_row["bootstrap_ci95_high"]),
                ],
                "length_odds_ratio_per_doubling": float(
                    math.exp(length_row["estimate"])
                ),
                "needles_coefficient": float(needles_row["estimate"]),
                "needles_bootstrap_ci95": [
                    float(needles_row["bootstrap_ci95_low"]),
                    float(needles_row["bootstrap_ci95_high"]),
                ],
                "needles_odds_ratio_per_doubling": float(
                    math.exp(needles_row["estimate"])
                ),
                "cv_log_loss": float(interpretive_metrics["cv_log_loss_mean"]),
                "cv_brier": float(interpretive_metrics["cv_brier"]),
                "cv_ece": float(interpretive_metrics["cv_ece"]),
                "cv_calibration_slope": float(
                    interpretive_metrics["cv_calibration_slope"]
                ),
                "controls_only_cv_log_loss": float(
                    controls_metrics["cv_log_loss_mean"]
                ),
                "density_only_cv_log_loss": float(
                    density_metrics["cv_log_loss_mean"]
                ),
                "length_density_cv_log_loss": float(
                    length_density_metrics["cv_log_loss_mean"]
                ),
                "density_only_log_loss_gap_vs_length_needles": (
                    density_sufficiency_gap
                ),
            },
            "best_predictive_candidate": str(best_binary["candidate"]),
            "best_predictive_cv_log_loss": float(
                best_binary["cv_log_loss_mean"]
            ),
        },
        "absolute_error": {
            "status": "secondary_conditional_diagnostic",
            "parsed_requests": len(parsed),
            "parse_coverage": len(parsed) / len(frame),
            "unparsed_requests": len(frame) - len(parsed),
            "interpretive_target": "log1p_absolute_error",
            "interpretive_candidate": INTERPRETIVE,
            "best_configuration": (
                f"{best_error['target']}::{best_error['candidate']}"
            ),
            "best_cv_absolute_unit_mae": float(
                best_error["cv_absolute_unit_mae"]
            ),
            "best_cv_absolute_unit_rmse": float(
                best_error["cv_absolute_unit_rmse"]
            ),
        },
        "coordinates": {
            "log_length": "log2(target_passage_tokens / 5000)",
            "log_needles": "log2(num_needles / 5)",
            "log_density": "log2(needles per 1k target passage tokens)",
            "identity": (
                "log_density = log_needles - log_length plus a fixed centering "
                "constant"
            ),
        },
        "scope": {
            "target_passage_tokens": sorted(
                int(value) for value in frame["target_passage_tokens"].unique()
            ),
            "num_needles": sorted(
                int(value) for value in frame["num_needles"].unique()
            ),
            "models": MODEL_LEVELS,
            "seeds": sorted(int(value) for value in frame["seed"].unique()),
        },
    }
    write_json(METRICS / "summary.json", summary)
    write_json(
        METRICS / "accuracy_law.json",
        {
            "model_size_used": False,
            "candidate": INTERPRETIVE,
            "formula": summary["accuracy"]["interpretive_formula"],
            "coefficients": accuracy_coefficients.to_dict(orient="records"),
            "held_out_metrics": summary["accuracy"]["interpretive_relation"],
        },
    )
    write_json(
        METRICS / "absolute_error_law.json",
        {
            "model_size_used": False,
            "conditional_on_parse_success": True,
            "parse_coverage": len(parsed) / len(frame),
            "target": "log1p_absolute_error",
            "candidate": INTERPRETIVE,
            "coefficients": error_coefficients.to_dict(orient="records"),
            "best_configuration": summary["absolute_error"]["best_configuration"],
        },
    )

    make_figures(
        frame,
        aggregates,
        binary_comparison,
        error_comparison,
        oof_by_candidate[INTERPRETIVE],
    )
    (OUT / "README.md").write_text(
        render_readme(summary, accuracy_coefficients, error_coefficients),
        encoding="utf-8",
    )

    set_state(
        "complete",
        interpretive_candidate=INTERPRETIVE,
        best_predictive_candidate=str(best_binary["candidate"]),
        best_error_configuration=summary["absolute_error"]["best_configuration"],
    )
    log("no-model-size analysis computational stages complete")

    excluded = {
        MANIFEST.name,
        INTEGRITY.name,
        "launcher_stdout.log",
        "launcher_stderr.log",
        "postrun_verification.json",
    }
    outputs = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            outputs.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(
        INTEGRITY,
        {
            "created_at_utc": utc_now(),
            "analysis_root": str(OUT),
            "files": outputs,
        },
    )
    manifest = {
        "schema_version": "realistic_niah_empirical_law_no_model_size_v1",
        "created_at_utc": utc_now(),
        "analysis_root": str(OUT),
        "run_root": str(RUN),
        "filesystem_id": EXPECTED_FILESYSTEM_ID,
        "source": source_meta,
        "source_rows": len(frame),
        "source_request_ids_unique": int(frame["request_id"].nunique()),
        "model_size_used": False,
        "model_identity_use": (
            "categorical nuisance intercept and heterogeneity diagnostics only"
        ),
        "primary_target": "exact_correct over all requests",
        "secondary_target": (
            "absolute count error conditional on parsed numeric output"
        ),
        "failure_handling": (
            "parse failure, format failure and truncation retained as incorrect "
            "for primary accuracy"
        ),
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "split": "five leave-one-seed-out folds",
        "candidates": [
            {
                "name": candidate.name,
                "numeric": candidate.numeric,
                "categorical": candidate.categorical,
                "interactions": candidate.interactions,
                "description": candidate.description,
            }
            for candidate in CANDIDATES
        ],
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifact_integrity_path": str(INTEGRITY.relative_to(OUT)),
        "artifact_integrity_sha256": sha256(INTEGRITY),
        "output_file_count_excluding_manifest_integrity_logs_verification": len(
            outputs
        ),
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_root": str(OUT),
                "model_size_used": False,
                "requests": len(frame),
                "parsed_for_absolute_error": len(parsed),
                "interpretive_cv_log_loss": float(
                    interpretive_metrics["cv_log_loss_mean"]
                ),
                "density_only_cv_log_loss": float(
                    density_metrics["cv_log_loss_mean"]
                ),
                "best_accuracy_candidate": str(best_binary["candidate"]),
                "best_error_configuration": summary["absolute_error"][
                    "best_configuration"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
