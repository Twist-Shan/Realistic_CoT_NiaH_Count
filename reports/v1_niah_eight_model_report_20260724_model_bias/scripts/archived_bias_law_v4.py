#!/usr/bin/env python3
"""Analyze signed counting bias versus length and needle count."""

from __future__ import annotations

import hashlib
import json
import math
import platform
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
from scipy import stats
from scipy import linalg as scipy_linalg


RUN = Path(
    "/lambda/nfs/Twist-CoT-Count-Multi-Model/runs/"
    "realistic_niah_v1/six_models_formal_20260723T194300Z"
)
SOURCE_ANALYSIS = RUN / "analysis" / "empirical_law_v1"
SOURCE = SOURCE_ANALYSIS / "tables" / "request_level.csv"
SOURCE_INTEGRITY = SOURCE_ANALYSIS / "artifact_integrity.json"
SOURCE_MANIFEST = SOURCE_ANALYSIS / "analysis_manifest.json"
OUT = RUN / "analysis" / "bias_law_v4"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
METRICS = OUT / "metrics"
PLAN = OUT / "analysis_plan.md"
STATE = OUT / "state.json"
LOG = OUT / "run.log"
INTEGRITY = OUT / "artifact_integrity.json"
MANIFEST = OUT / "analysis_manifest.json"

EXPECTED_FILESYSTEM_ID = "c8d6df94b8504c14a4ba5e05e3119723"
EXPECTED_ROWS = 6300
RANDOM_SEED = 20260724
BOOTSTRAP_REPLICATES = 500
L0 = 5000.0
N0 = 5.0
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

PLAN_TEXT = """# Signed-bias analysis

## Version note

This is v4. The aborted v1 diagnostic forced the primary selector to exclude
the density candidate and used a zero-collapsing Huber MAD on integer errors.
Those choices were invalid for the observed point mass at zero. V2 then
exposed a broken default SVD least-squares driver in the Lambda numerical
stack: its returned residual exceeded the zero-coefficient residual, violating
the defining least-squares bound. V3 added residual-checked cross-product/QR
solves, but Huber's changing weighted systems still failed the same bound in
some folds. V4 preserves all aborted versions, removes iterative Huber fitting,
uses residual-checked cross-product/QR solves, and replaces Huber with a
frozen symmetric count-scale cap at +/-30 (the maximum true needle count).
This revision is explicitly post-diagnostic rather than presented as
preregistered.

## Definition and sample

`signed_error = predicted_count - true_num_needles`.

Positive values are over-counting and negative values are under-counting.
Numeric bias is defined only when a numeric prediction was parsed. All 915
unparsed outputs remain failures in the primary accuracy analysis and are
reported as missing-by-definition here; they are not assigned zero bias.

## Estimands

1. Conditional mean signed bias in count units, estimated by OLS.
2. Robust count-scale bias after symmetric clipping to `[-30, 30]`.
3. Relative signed bias `signed_error / num_needles`.
4. Sign-preserving compressed bias `asinh(signed_error)`.
5. Descriptive median, over-count and under-count rates.

The mean is the literal statistical bias but is sensitive to extreme numeric
outputs. The capped, median and asinh analyses distinguish systematic typical
bias from a small number of very large over-counts.

## Frozen coordinate candidates

Within each model, with separate intercepts for every observed
model/prompt/order condition:

- log length plus log needle count;
- raw normalized length plus raw normalized needle count;
- log needle density alone;
- log length plus log needle count plus their low-order interaction.

No model-size variable is used.

## Validation and uncertainty

- leave-one-seed-out;
- leave-one-needle-level-out;
- leave-one-length-level-out;
- five blocked length/needle-cell folds;
- 500 bootstrap replicates clustered by complete stimulus ID.

Raw and capped conditional means are selected by held-out RMSE. Relative and
asinh diagnostic targets are selected by held-out MAE on their own target
scale. Family-wise significance across the eight model-specific tests uses
Holm adjustment. The separate log-length/log-needle candidate is always
retained as a diagnostic even if another coordinate validates better.

All candidate results are retained. Bias laws are restricted to parsed outputs
and to the registered 2K--10K token, 1--30 needle grid.
"""


@dataclass(frozen=True)
class Candidate:
    name: str
    coordinate: str
    interaction: bool = False


CANDIDATES = [
    Candidate("log_length_needles", "log"),
    Candidate("raw_length_needles", "raw"),
    Candidate("log_density_only", "density"),
    Candidate("log_length_needles_interaction", "log", interaction=True),
]


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
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()}\t{message}\n")


def set_state(phase: str, **extra: Any) -> None:
    write_json(
        STATE,
        {
            "status": phase,
            "phase": phase,
            "model_size_used": False,
            "updated_at_utc": utc_now(),
            **extra,
        },
    )


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    prior_integrity = json.loads(SOURCE_INTEGRITY.read_text(encoding="utf-8"))
    if prior_manifest.get("filesystem_id") != EXPECTED_FILESYSTEM_ID:
        raise ValueError("Source analysis filesystem ID mismatch")
    indexed = {item["path"]: item for item in prior_integrity["files"]}
    key = "tables/request_level.csv"
    item = indexed.get(key)
    if item is None:
        raise ValueError("Source table is not integrity-indexed")
    if SOURCE.stat().st_size != item["bytes"] or sha256(SOURCE) != item["sha256"]:
        raise ValueError("Source table SHA mismatch")
    frame = pd.read_csv(SOURCE)
    if len(frame) != EXPECTED_ROWS or frame["request_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("Source rows or request IDs are invalid")
    parsed = frame[frame["parse_success"].astype(int).eq(1)].copy()
    parsed["signed_error"] = pd.to_numeric(
        parsed["signed_error"], errors="raise"
    )
    parsed["predicted_count"] = pd.to_numeric(
        parsed["predicted_count"], errors="raise"
    )
    definition_error = (
        parsed["predicted_count"]
        - parsed["num_needles"].astype(float)
        - parsed["signed_error"]
    ).abs()
    if float(definition_error.max()) > 1e-9:
        raise ValueError("signed_error is not predicted_count - num_needles")
    parsed["relative_signed_error"] = (
        parsed["signed_error"] / parsed["num_needles"].astype(float)
    )
    parsed["asinh_signed_error"] = np.arcsinh(parsed["signed_error"])
    parsed["capped_signed_error"] = np.clip(
        parsed["signed_error"].to_numpy(dtype=float), -30.0, 30.0
    )
    parsed["ln_length"] = np.log(
        parsed["target_passage_tokens"].astype(float) / L0
    )
    parsed["ln_needles"] = np.log(
        parsed["num_needles"].astype(float) / N0
    )
    parsed["ln_density"] = np.log(
        parsed["density_per_1k"].astype(float)
    )
    parsed["raw_length"] = (
        parsed["target_passage_tokens"].astype(float) / L0 - 1.0
    )
    parsed["raw_needles"] = (
        parsed["num_needles"].astype(float) / N0 - 1.0
    )
    parsed["condition"] = (
        parsed["model_label"].astype(str)
        + "|"
        + parsed["prompt_mode"].astype(str)
        + "|"
        + parsed["query_order"].astype(str)
    )
    conditions = sorted(parsed["condition"].unique())
    parsed.attrs["conditions"] = conditions
    return frame, parsed, {
        "path": str(SOURCE),
        "bytes": SOURCE.stat().st_size,
        "sha256": sha256(SOURCE),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_integrity": str(SOURCE_INTEGRITY),
        "source_integrity_sha256": sha256(SOURCE_INTEGRITY),
    }


def coordinate_values(
    frame: pd.DataFrame, candidate: Candidate
) -> tuple[np.ndarray, np.ndarray | None]:
    if candidate.coordinate == "log":
        return (
            frame["ln_length"].to_numpy(dtype=float),
            frame["ln_needles"].to_numpy(dtype=float),
        )
    if candidate.coordinate == "raw":
        return (
            frame["raw_length"].to_numpy(dtype=float),
            frame["raw_needles"].to_numpy(dtype=float),
        )
    if candidate.coordinate == "density":
        return frame["ln_density"].to_numpy(dtype=float), None
    raise KeyError(candidate.coordinate)


def build_design(
    frame: pd.DataFrame,
    candidate: Candidate,
    conditions: list[str],
) -> tuple[np.ndarray, list[str]]:
    first, second = coordinate_values(frame, candidate)
    model_values = frame["model_label"].astype(str).to_numpy()
    condition_values = frame["condition"].astype(str).to_numpy()
    columns: list[np.ndarray] = []
    names: list[str] = []
    for condition in conditions:
        columns.append((condition_values == condition).astype(float))
        names.append(f"intercept[{condition}]")
    if candidate.coordinate == "density":
        for model in MODELS:
            indicator = (model_values == model).astype(float)
            columns.append(indicator * first)
            names.append(f"density[{model}]")
    else:
        assert second is not None
        for model in MODELS:
            indicator = (model_values == model).astype(float)
            columns.append(indicator * first)
            names.append(f"length[{model}]")
        for model in MODELS:
            indicator = (model_values == model).astype(float)
            columns.append(indicator * second)
            names.append(f"needles[{model}]")
        if candidate.interaction:
            for model in MODELS:
                indicator = (model_values == model).astype(float)
                columns.append(indicator * first * second)
                names.append(f"interaction[{model}]")
    matrix = np.column_stack(columns)
    keep = [
        index
        for index in range(matrix.shape[1])
        if np.nanmax(matrix[:, index]) != np.nanmin(matrix[:, index])
        or np.nanmax(matrix[:, index]) != 0.0
    ]
    matrix = matrix[:, keep]
    names = [names[index] for index in keep]
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite design: {candidate.name}")
    return matrix, names


def solve_least_squares(
    matrix: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve least squares without the broken default SVD driver.

    The Lambda image's NumPy/SciPy default GELSD path was empirically observed
    to return a residual norm larger than the zero-coefficient residual on
    this well-conditioned design.  The cross-product system is well
    conditioned here; QR/GELSY is retained as a fallback.
    """
    if weights is None:
        normal = matrix.T @ matrix
        right = matrix.T @ outcome
    else:
        normal = matrix.T @ (weights[:, None] * matrix)
        right = matrix.T @ (weights * outcome)
    def weighted_sum_squares(value: np.ndarray) -> float:
        residual = outcome - matrix @ value
        if weights is None:
            return float(residual @ residual)
        return float((weights * residual) @ residual)

    if weights is None:
        zero_sum_squares = float(outcome @ outcome)
    else:
        zero_sum_squares = float((weights * outcome) @ outcome)
    use_fallback = False
    try:
        beta = np.linalg.solve(normal, right)
        candidate_sum_squares = weighted_sum_squares(beta)
        use_fallback = (
            not np.isfinite(candidate_sum_squares)
            or candidate_sum_squares
            > zero_sum_squares * (1.0 + 1e-8) + 1e-8
        )
    except np.linalg.LinAlgError:
        use_fallback = True
    if use_fallback:
        if weights is None:
            weighted_matrix = matrix
            weighted_outcome = outcome
        else:
            root = np.sqrt(weights)
            weighted_matrix = matrix * root[:, None]
            weighted_outcome = outcome * root
        beta = scipy_linalg.lstsq(
            weighted_matrix,
            weighted_outcome,
            lapack_driver="gelsy",
        )[0]
        fallback_sum_squares = weighted_sum_squares(beta)
        if (
            not np.isfinite(fallback_sum_squares)
            or fallback_sum_squares
            > zero_sum_squares * (1.0 + 1e-8) + 1e-8
        ):
            raise RuntimeError(
                "Both cross-product and GELSY least-squares solvers "
                "failed the residual-bound check"
            )
    return beta, normal


def fit_ols(matrix: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    beta, normal = solve_least_squares(matrix, outcome)
    prediction = matrix @ beta
    residual = outcome - prediction
    residual_sum_squares = float(residual @ residual)
    zero_sum_squares = float(outcome @ outcome)
    if residual_sum_squares > zero_sum_squares * (1.0 + 1e-8) + 1e-8:
        raise RuntimeError(
            "Least-squares residual bound failed: "
            f"{residual_sum_squares} > {zero_sum_squares}"
        )
    degrees = max(len(outcome) - matrix.shape[1], 1)
    try:
        inverse_normal = np.linalg.solve(
            normal, np.eye(normal.shape[0])
        )
    except np.linalg.LinAlgError:
        inverse_normal = scipy_linalg.pinvh(normal)
    covariance = residual_sum_squares / degrees * inverse_normal
    return {
        "beta": beta,
        "prediction": prediction,
        "covariance": covariance,
        "converged": True,
    }


def fit_huber(
    matrix: np.ndarray,
    outcome: np.ndarray,
    tuning: float = 1.345,
) -> dict[str, Any]:
    beta, _ = solve_least_squares(matrix, outcome)
    converged = False
    scale = 1.0
    weights = np.ones(len(outcome))
    for _ in range(300):
        residual = outcome - matrix @ beta
        centered = residual - np.median(residual)
        # Bias lives on an integer-count lattice and often has a point mass at
        # zero.  A raw MAD can therefore collapse to zero even with meaningful
        # non-zero errors.  One count is a frozen, interpretable scale floor.
        scale = max(
            1.4826 * float(np.median(np.abs(centered))),
            1.0,
        )
        threshold = tuning * scale
        absolute = np.abs(residual)
        weights = np.ones(len(outcome))
        mask = absolute > threshold
        weights[mask] = threshold / np.maximum(absolute[mask], 1e-12)
        updated, _ = solve_least_squares(
            matrix, outcome, weights=weights
        )
        if float(np.linalg.norm(updated - beta)) < 1e-6 * (
            1.0 + float(np.linalg.norm(beta))
        ):
            beta = updated
            converged = True
            break
        beta = updated
    prediction = matrix @ beta
    weighted_normal = matrix.T @ (weights[:, None] * matrix)
    try:
        bread = np.linalg.solve(
            weighted_normal, np.eye(weighted_normal.shape[0])
        )
    except np.linalg.LinAlgError:
        bread = scipy_linalg.pinvh(weighted_normal)
    residual = outcome - prediction
    meat = matrix.T @ (
        (weights * residual**2)[:, None] * matrix
    )
    covariance = bread @ meat @ bread
    return {
        "beta": beta,
        "prediction": prediction,
        "covariance": covariance,
        "converged": converged,
        "scale": scale,
    }


def validation_schemes(
    frame: pd.DataFrame,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    schemes: dict[str, list[tuple[str, np.ndarray]]] = {}
    schemes["seed"] = [
        (
            str(value),
            frame["seed"].astype(int).to_numpy() == int(value),
        )
        for value in sorted(frame["seed"].unique())
    ]
    lengths = sorted(
        int(value) for value in frame["target_passage_tokens"].unique()
    )
    needles = sorted(
        int(value) for value in frame["num_needles"].unique()
    )
    schemes["length_level"] = [
        (
            str(value),
            frame["target_passage_tokens"].astype(int).to_numpy() == value,
        )
        for value in lengths
    ]
    schemes["needle_level"] = [
        (
            str(value),
            frame["num_needles"].astype(int).to_numpy() == value,
        )
        for value in needles
    ]
    length_index = {value: index for index, value in enumerate(lengths)}
    needle_index = {value: index for index, value in enumerate(needles)}
    block = np.array(
        [
            (
                length_index[int(length)] * len(needles)
                + needle_index[int(needle)]
            )
            % 5
            for length, needle in zip(
                frame["target_passage_tokens"], frame["num_needles"]
            )
        ]
    )
    schemes["cell_block"] = [
        (str(index), block == index) for index in range(5)
    ]
    return schemes


def target_values(
    frame: pd.DataFrame, target: str
) -> np.ndarray:
    if target == "raw_signed_error":
        return frame["signed_error"].to_numpy(dtype=float)
    if target == "capped_signed_error":
        return frame["capped_signed_error"].to_numpy(dtype=float)
    if target == "relative_signed_error":
        return frame["relative_signed_error"].to_numpy(dtype=float)
    if target == "asinh_signed_error":
        return frame["asinh_signed_error"].to_numpy(dtype=float)
    raise KeyError(target)


def to_raw_bias(
    prediction: np.ndarray, frame: pd.DataFrame, target: str
) -> np.ndarray:
    if target == "raw_signed_error":
        return prediction
    if target == "capped_signed_error":
        return prediction
    if target == "relative_signed_error":
        return prediction * frame["num_needles"].to_numpy(dtype=float)
    if target == "asinh_signed_error":
        return np.sinh(np.clip(prediction, -20.0, 20.0))
    raise KeyError(target)


def evaluate_models(
    parsed: pd.DataFrame,
    conditions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    configurations = [
        ("ols", "raw_signed_error"),
        ("ols", "capped_signed_error"),
        ("ols", "relative_signed_error"),
        ("ols", "asinh_signed_error"),
    ]
    schemes = validation_schemes(parsed)
    observed_raw = parsed["signed_error"].to_numpy(dtype=float)
    comparisons = []
    fold_rows = []
    full_fits: dict[str, Any] = {}
    for method, target in configurations:
        outcome = target_values(parsed, target)
        for candidate in CANDIDATES:
            matrix, names = build_design(parsed, candidate, conditions)
            scheme_values = {}
            all_converged = True
            for scheme, folds in schemes.items():
                oof = np.full(len(parsed), np.nan)
                for label, test in folds:
                    train = ~test
                    if method == "ols":
                        fitted = fit_ols(matrix[train], outcome[train])
                    else:
                        fitted = fit_huber(matrix[train], outcome[train])
                    all_converged = (
                        all_converged and fitted["converged"]
                    )
                    prediction = matrix[test] @ fitted["beta"]
                    oof[test] = prediction
                    raw_prediction = to_raw_bias(
                        prediction, parsed.loc[test], target
                    )
                    residual = observed_raw[test] - raw_prediction
                    fold_rows.append(
                        {
                            "method": method,
                            "target": target,
                            "candidate": candidate.name,
                            "scheme": scheme,
                            "held_out": label,
                            "n_train": int(train.sum()),
                            "n_test": int(test.sum()),
                            "target_mae": float(
                                np.mean(
                                    np.abs(outcome[test] - prediction)
                                )
                            ),
                            "target_rmse": float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            outcome[test] - prediction
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            "raw_bias_mae": float(
                                np.mean(np.abs(residual))
                            ),
                            "raw_bias_rmse": float(
                                np.sqrt(np.mean(residual**2))
                            ),
                            "raw_mean_residual": float(
                                np.mean(residual)
                            ),
                        }
                    )
                raw_prediction = to_raw_bias(oof, parsed, target)
                residual = observed_raw - raw_prediction
                scheme_values[scheme] = {
                    "target_mae": float(
                        np.mean(np.abs(outcome - oof))
                    ),
                    "target_rmse": float(
                        np.sqrt(np.mean((outcome - oof) ** 2))
                    ),
                    "raw_bias_mae": float(np.mean(np.abs(residual))),
                    "raw_bias_rmse": float(
                        np.sqrt(np.mean(residual**2))
                    ),
                    "raw_mean_residual": float(np.mean(residual)),
                }
            if method == "ols":
                full = fit_ols(matrix, outcome)
            else:
                full = fit_huber(matrix, outcome)
            key = f"{method}::{target}::{candidate.name}"
            full_fits[key] = {
                "fit": full,
                "matrix": matrix,
                "names": names,
                "candidate": candidate,
                "method": method,
                "target": target,
            }
            if method == "ols" and target in {
                "raw_signed_error",
                "capped_signed_error",
            }:
                selection_metric = "raw_bias_rmse"
                if target == "capped_signed_error":
                    selection_metric = "target_rmse"
            else:
                # Relative and asinh targets are diagnostic estimands.  Their
                # inverse transforms are not conditional-mean estimators on
                # the raw count scale, so select them on their own scale.
                selection_metric = "target_mae"
            selection_score = float(
                np.mean(
                    [
                        scheme_values[scheme][selection_metric]
                        for scheme in [
                            "seed",
                            "needle_level",
                            "length_level",
                            "cell_block",
                        ]
                    ]
                )
            )
            row: dict[str, Any] = {
                "method": method,
                "target": target,
                "candidate": candidate.name,
                "coordinate": candidate.coordinate,
                "interaction": candidate.interaction,
                "n_parameters": matrix.shape[1],
                "all_converged": all_converged,
                "selection_metric": selection_metric,
                "selection_score": selection_score,
            }
            for scheme, values in scheme_values.items():
                for metric, value in values.items():
                    row[f"{scheme}_{metric}"] = value
            comparisons.append(row)
            log(
                f"{key}: {selection_metric} selection={selection_score:.6f}"
            )
    return (
        pd.DataFrame(comparisons).sort_values(
            ["method", "target", "selection_score"]
        ),
        pd.DataFrame(fold_rows),
        full_fits,
    )


def cluster_indices(
    frame: pd.DataFrame, replicates: int
) -> list[np.ndarray]:
    clusters = frame["stimulus_id"].astype(str).to_numpy()
    unique = np.unique(clusters)
    by_cluster = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique
    }
    rng = np.random.default_rng(RANDOM_SEED)
    return [
        np.concatenate(
            [
                by_cluster[value]
                for value in rng.choice(
                    unique, size=len(unique), replace=True
                )
            ]
        )
        for _ in range(replicates)
    ]


def bootstrap_fit(
    parsed: pd.DataFrame,
    fitted: dict[str, Any],
    indices: list[np.ndarray],
) -> np.ndarray:
    matrix = fitted["matrix"]
    outcome = target_values(parsed, fitted["target"])
    values = []
    for sample in indices:
        if fitted["method"] == "ols":
            result = fit_ols(matrix[sample], outcome[sample])
        else:
            result = fit_huber(matrix[sample], outcome[sample])
        if result["converged"]:
            values.append(result["beta"])
    if len(values) < int(0.9 * len(indices)):
        raise RuntimeError("Too few converged bias bootstrap fits")
    return np.vstack(values)


def bootstrap_two_sided_p(values: np.ndarray) -> float:
    """Finite-replicate, two-sided sign p-value with a plus-one correction."""
    replicate_count = len(values)
    lower = (float(np.sum(values <= 0)) + 1.0) / (
        replicate_count + 1.0
    )
    upper = (float(np.sum(values >= 0)) + 1.0) / (
        replicate_count + 1.0
    )
    return min(1.0, 2.0 * min(lower, upper))


def holm_adjust(values: np.ndarray) -> np.ndarray:
    """Holm family-wise-error adjustment, preserving the input order."""
    raw = np.asarray(values, dtype=float)
    adjusted = np.full(len(raw), np.nan)
    valid = np.flatnonzero(np.isfinite(raw))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(raw[valid])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * raw[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def slope_parameter_table(
    fitted: dict[str, Any], bootstrap: np.ndarray
) -> pd.DataFrame:
    names = fitted["names"]
    mapping = {name: index for index, name in enumerate(names)}
    beta = fitted["fit"]["beta"]
    rows = []
    for model in MODELS:
        direct_key = f"{model}|direct|query_first"
        intercept_name = f"intercept[{direct_key}]"
        row: dict[str, Any] = {
            "model_label": model,
            "method": fitted["method"],
            "target": fitted["target"],
            "candidate": fitted["candidate"].name,
            "coordinate": fitted["candidate"].coordinate,
            "bootstrap_replicates": len(bootstrap),
        }
        if intercept_name in mapping:
            index = mapping[intercept_name]
            values = bootstrap[:, index]
            row["baseline_direct_query_first"] = float(beta[index])
            row["baseline_ci95_low"] = float(
                np.percentile(values, 2.5)
            )
            row["baseline_ci95_high"] = float(
                np.percentile(values, 97.5)
            )
        for feature, label in [
            ("length", "length_slope"),
            ("needles", "needle_slope"),
            ("density", "density_slope"),
        ]:
            name = f"{feature}[{model}]"
            if name not in mapping:
                continue
            index = mapping[name]
            values = bootstrap[:, index]
            estimate = float(beta[index])
            row[label] = estimate
            row[f"{label}_ci95_low"] = float(
                np.percentile(values, 2.5)
            )
            row[f"{label}_ci95_high"] = float(
                np.percentile(values, 97.5)
            )
            row[f"{label}_bootstrap_p_two_sided"] = (
                bootstrap_two_sided_p(values)
            )
            if fitted["candidate"].coordinate == "log":
                row[f"{label}_effect_per_doubling"] = (
                    estimate * math.log(2.0)
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    p_columns = [
        column
        for column in table.columns
        if column.endswith("_bootstrap_p_two_sided")
    ]
    for column in p_columns:
        table[f"{column}_holm"] = holm_adjust(
            table[column].to_numpy(dtype=float)
        )
    return table


def descriptive_bootstrap(
    parsed: pd.DataFrame, indices: list[np.ndarray]
) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        model_mask = (
            parsed["model_label"].astype(str).to_numpy() == model
        )
        observed = parsed.loc[model_mask, "signed_error"].to_numpy(
            dtype=float
        )
        point = {
            "mean_bias": float(np.mean(observed)),
            "median_bias": float(np.median(observed)),
            "trimmed_mean_10pct": float(
                stats.trim_mean(observed, 0.1)
            ),
            "overcount_rate": float(np.mean(observed > 0)),
            "undercount_rate": float(np.mean(observed < 0)),
            "exact_rate_among_parsed": float(np.mean(observed == 0)),
        }
        draws: dict[str, list[float]] = {
            key: [] for key in point
        }
        for sample in indices:
            selected = sample[model_mask[sample]]
            values = parsed.iloc[selected]["signed_error"].to_numpy(
                dtype=float
            )
            if not len(values):
                continue
            draws["mean_bias"].append(float(np.mean(values)))
            draws["median_bias"].append(float(np.median(values)))
            draws["trimmed_mean_10pct"].append(
                float(stats.trim_mean(values, 0.1))
            )
            draws["overcount_rate"].append(
                float(np.mean(values > 0))
            )
            draws["undercount_rate"].append(
                float(np.mean(values < 0))
            )
            draws["exact_rate_among_parsed"].append(
                float(np.mean(values == 0))
            )
        row: dict[str, Any] = {
            "model_label": model,
            "parsed_requests": len(observed),
            "minimum_signed_error": float(np.min(observed)),
            "maximum_signed_error": float(np.max(observed)),
            **point,
        }
        for key, values in draws.items():
            array = np.asarray(values)
            row[f"{key}_ci95_low"] = float(
                np.percentile(array, 2.5)
            )
            row[f"{key}_ci95_high"] = float(
                np.percentile(array, 97.5)
            )
            if key in {"mean_bias", "median_bias", "trimmed_mean_10pct"}:
                row[f"{key}_bootstrap_p_two_sided"] = (
                    bootstrap_two_sided_p(array)
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    for key in ["mean_bias", "median_bias", "trimmed_mean_10pct"]:
        column = f"{key}_bootstrap_p_two_sided"
        table[f"{column}_holm"] = holm_adjust(
            table[column].to_numpy(dtype=float)
        )
    return table


def aggregate_tables(parsed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def aggregate(columns: list[str]) -> pd.DataFrame:
        records = []
        for keys, part in parsed.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = part["signed_error"].to_numpy(dtype=float)
            record: dict[str, Any] = {
                "parsed_requests": len(values),
                "mean_bias": float(np.mean(values)),
                "median_bias": float(np.median(values)),
                "trimmed_mean_10pct": float(
                    stats.trim_mean(values, 0.1)
                ),
                "mean_relative_bias": float(
                    part["relative_signed_error"].mean()
                ),
                "overcount_rate": float(np.mean(values > 0)),
                "undercount_rate": float(np.mean(values < 0)),
                "exact_rate_among_parsed": float(np.mean(values == 0)),
            }
            for column, value in zip(columns, keys):
                record[column] = value
            records.append(record)
        return pd.DataFrame(records)

    return {
        "pooled_length_needles_bias": aggregate(
            ["target_passage_tokens", "num_needles"]
        ),
        "model_length_needles_bias": aggregate(
            ["model_label", "target_passage_tokens", "num_needles"]
        ),
        "model_density_bias": aggregate(
            ["model_label", "density_per_1k"]
        ),
    }


def make_figures(
    parsed: pd.DataFrame,
    descriptive: pd.DataFrame,
    comparison: pd.DataFrame,
    ols_parameters: pd.DataFrame,
    robust_parameters: pd.DataFrame,
    separate_raw_parameters: pd.DataFrame,
    separate_asinh_parameters: pd.DataFrame,
    aggregates: dict[str, pd.DataFrame],
) -> None:
    plt.style.use("default")
    y = np.arange(len(descriptive))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lower = descriptive["mean_bias"] - descriptive["mean_bias_ci95_low"]
    upper = descriptive["mean_bias_ci95_high"] - descriptive["mean_bias"]
    ax.errorbar(
        descriptive["mean_bias"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        label="Mean bias",
    )
    ax.scatter(
        descriptive["trimmed_mean_10pct"],
        y,
        marker="s",
        label="10% trimmed mean",
    )
    ax.scatter(
        descriptive["median_bias"],
        y,
        marker="x",
        label="Median",
    )
    ax.axvline(0, linestyle="--", color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(descriptive["model_label"])
    ax.set_xlabel("Signed count bias (prediction - truth)")
    ax.set_title("Mean bias is strongly affected by rare positive outliers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "model_bias_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.35
    ax.barh(
        y - width / 2,
        descriptive["overcount_rate"],
        width,
        label="Over-count",
        color="#C55A11",
    )
    ax.barh(
        y + width / 2,
        descriptive["undercount_rate"],
        width,
        label="Under-count",
        color="#4472C4",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(descriptive["model_label"])
    ax.set_xlabel("Rate among parsed outputs")
    ax.set_title("Bias direction by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "over_under_rates.png", dpi=180)
    plt.close(fig)

    raw = comparison[
        comparison["target"].isin(
            ["raw_signed_error", "capped_signed_error"]
        )
    ].copy()
    labels = raw["target"] + "::" + raw["candidate"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, raw["selection_score"], color="#70AD47")
    ax.set_xlabel("Blocked validation RMSE on each count-scale target")
    ax.set_title("Raw versus capped signed-bias coordinate comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "bias_candidate_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, table, title in [
        (axes[0], ols_parameters, "OLS conditional mean bias"),
        (
            axes[1],
            robust_parameters,
            "Capped [-30, 30] conditional mean bias",
        ),
    ]:
        available = [
            field
            for field in ["density_slope", "length_slope", "needle_slope"]
            if field in table.columns
        ]
        y_local = np.arange(len(table), dtype=float)
        offsets = np.linspace(-0.15, 0.15, max(len(available), 1))
        for offset, field in zip(offsets, available):
            lower = table[field] - table[f"{field}_ci95_low"]
            upper = table[f"{field}_ci95_high"] - table[field]
            ax.errorbar(
                table[field],
                y_local + offset,
                xerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=3,
                label=field.replace("_", " "),
            )
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_yticks(y_local)
        ax.set_yticklabels(table["model_label"])
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_bias_slopes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row_index, (table, title) in enumerate(
        [
            (separate_raw_parameters, "Raw mean signed bias"),
            (separate_asinh_parameters, "asinh signed bias"),
        ]
    ):
        for column_index, field in enumerate(
            ["length_slope", "needle_slope"]
        ):
            ax = axes[row_index, column_index]
            y_local = np.arange(len(table))
            lower = table[field] - table[f"{field}_ci95_low"]
            upper = table[f"{field}_ci95_high"] - table[field]
            ax.errorbar(
                table[field],
                y_local,
                xerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=3,
            )
            ax.axvline(0, linestyle="--", color="gray")
            ax.set_yticks(y_local)
            ax.set_yticklabels(table["model_label"])
            ax.set_title(f"{title}: {field.replace('_', ' ')}")
    fig.suptitle(
        "Diagnostic separate log-length and log-needle effects "
        "(full-data estimates)"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "separate_length_needle_slopes.png",
        dpi=180,
    )
    plt.close(fig)

    pooled = aggregates["pooled_length_needles_bias"]
    for metric, filename, title in [
        ("mean_bias", "mean_bias_heatmap.png", "Pooled mean signed bias"),
        (
            "median_bias",
            "median_bias_heatmap.png",
            "Pooled median signed bias",
        ),
    ]:
        matrix = pooled.pivot(
            index="target_passage_tokens",
            columns="num_needles",
            values=metric,
        ).sort_index()
        fig, ax = plt.subplots(figsize=(10, 3.8))
        maximum = float(np.nanmax(np.abs(matrix.to_numpy())))
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="coolwarm",
            vmin=-maximum,
            vmax=maximum,
        )
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([str(int(value)) for value in matrix.columns])
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([str(int(value)) for value in matrix.index])
        ax.set_xlabel("Needle count")
        ax.set_ylabel("Passage tokens")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="Prediction - truth")
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    model_table = aggregates["model_length_needles_bias"]
    for ax, model in zip(axes.ravel(), MODELS):
        part = model_table[model_table["model_label"] == model]
        for length, curve in part.groupby("target_passage_tokens"):
            curve = curve.sort_values("num_needles")
            ax.plot(
                curve["num_needles"],
                curve["trimmed_mean_10pct"],
                marker="o",
                markersize=3,
                label=str(int(length)),
            )
        ax.axhline(0, linestyle="--", color="gray")
        ax.set_xscale("log", base=2)
        ax.set_title(model)
    for ax in axes[-1, :]:
        ax.set_xlabel("Needle count")
    for ax in axes[:, 0]:
        ax.set_ylabel("10% trimmed signed bias")
    axes[0, 0].legend(title="Tokens")
    fig.suptitle("Robust signed bias across length and needle count")
    fig.tight_layout()
    fig.savefig(FIGURES / "robust_bias_curves_by_model.png", dpi=180)
    plt.close(fig)

    density_table = aggregates["model_density_bias"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    for ax, model in zip(axes.ravel(), MODELS):
        part = density_table[
            density_table["model_label"] == model
        ].sort_values("density_per_1k")
        ax.plot(
            part["density_per_1k"],
            part["mean_bias"],
            marker="o",
            markersize=3,
            label="mean",
        )
        ax.plot(
            part["density_per_1k"],
            part["trimmed_mean_10pct"],
            marker="s",
            markersize=3,
            label="10% trimmed",
        )
        ax.axhline(0, linestyle="--", color="gray")
        ax.set_xscale("log", base=2)
        ax.set_title(model)
    for ax in axes[-1, :]:
        ax.set_xlabel("Needles per 1,000 tokens")
    for ax in axes[:, 0]:
        ax.set_ylabel("Signed bias")
    axes[0, 0].legend()
    fig.suptitle("Signed bias versus needle density")
    fig.tight_layout()
    fig.savefig(FIGURES / "bias_vs_density_by_model.png", dpi=180)
    plt.close(fig)


def main() -> None:
    for directory in (OUT, TABLES, FIGURES, METRICS):
        directory.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_text(encoding="utf-8") != PLAN_TEXT:
        raise RuntimeError("Existing plan differs from frozen plan")
    PLAN.write_text(PLAN_TEXT, encoding="utf-8")
    LOG.write_text("", encoding="utf-8")
    set_state("running")
    full, parsed, source_meta = load_source()
    conditions = list(parsed.attrs["conditions"])
    log(
        f"source verified; parsed={len(parsed)}, "
        f"unparsed={len(full)-len(parsed)}"
    )
    diagnostic_matrix, _ = build_design(
        parsed, CANDIDATES[0], conditions
    )
    diagnostic_outcome = target_values(
        parsed, "asinh_signed_error"
    )
    safe_beta, _ = solve_least_squares(
        diagnostic_matrix, diagnostic_outcome
    )
    default_beta = np.linalg.lstsq(
        diagnostic_matrix, diagnostic_outcome, rcond=None
    )[0]
    zero_rmse = float(
        np.sqrt(np.mean(diagnostic_outcome**2))
    )
    safe_rmse = float(
        np.sqrt(
            np.mean(
                (
                    diagnostic_outcome
                    - diagnostic_matrix @ safe_beta
                )
                ** 2
            )
        )
    )
    default_rmse = float(
        np.sqrt(
            np.mean(
                (
                    diagnostic_outcome
                    - diagnostic_matrix @ default_beta
                )
                ** 2
            )
        )
    )
    if safe_rmse > zero_rmse * (1.0 + 1e-8) + 1e-8:
        raise RuntimeError("Safe solver failed residual-bound smoke test")
    write_json(
        METRICS / "numerical_solver_diagnostic.json",
        {
            "design": "asinh_signed_error::log_length_needles",
            "rows": len(diagnostic_outcome),
            "columns": diagnostic_matrix.shape[1],
            "zero_coefficient_rmse": zero_rmse,
            "cross_product_solver_rmse": safe_rmse,
            "default_numpy_lstsq_rmse": default_rmse,
            "default_driver_residual_bound_passed": (
                default_rmse <= zero_rmse * (1.0 + 1e-8) + 1e-8
            ),
            "selected_solver": (
                "cross_product_solve_with_scipy_gelsy_fallback"
            ),
        },
    )
    log(
        "solver smoke test: "
        f"zero={zero_rmse:.6f}, safe={safe_rmse:.6f}, "
        f"default={default_rmse:.6f}"
    )

    comparison, fold_metrics, fits = evaluate_models(
        parsed, conditions
    )
    comparison.to_csv(TABLES / "bias_candidate_comparison.csv", index=False)
    fold_metrics.to_csv(TABLES / "bias_fold_metrics.csv", index=False)

    def select(method: str, target: str) -> tuple[str, dict[str, Any]]:
        subset = comparison[
            (comparison["method"] == method)
            & (comparison["target"] == target)
        ].sort_values("selection_score")
        name = str(subset.iloc[0]["candidate"])
        key = f"{method}::{target}::{name}"
        return name, fits[key]

    ols_name, ols_fit = select("ols", "raw_signed_error")
    robust_name, robust_fit = select(
        "ols", "capped_signed_error"
    )
    relative_name, relative_fit = select(
        "ols", "relative_signed_error"
    )
    asinh_name, asinh_fit = select("ols", "asinh_signed_error")
    separate_raw_fit = fits[
        "ols::raw_signed_error::log_length_needles"
    ]
    separate_asinh_fit = fits[
        "ols::asinh_signed_error::log_length_needles"
    ]

    indices = cluster_indices(parsed, BOOTSTRAP_REPLICATES)
    ols_bootstrap = bootstrap_fit(parsed, ols_fit, indices)
    robust_bootstrap = bootstrap_fit(parsed, robust_fit, indices)
    relative_bootstrap = bootstrap_fit(parsed, relative_fit, indices)
    asinh_bootstrap = bootstrap_fit(parsed, asinh_fit, indices)
    separate_raw_bootstrap = bootstrap_fit(
        parsed, separate_raw_fit, indices
    )
    separate_asinh_bootstrap = bootstrap_fit(
        parsed, separate_asinh_fit, indices
    )

    ols_parameters = slope_parameter_table(
        ols_fit, ols_bootstrap
    )
    robust_parameters = slope_parameter_table(
        robust_fit, robust_bootstrap
    )
    relative_parameters = slope_parameter_table(
        relative_fit, relative_bootstrap
    )
    asinh_parameters = slope_parameter_table(
        asinh_fit, asinh_bootstrap
    )
    separate_raw_parameters = slope_parameter_table(
        separate_raw_fit, separate_raw_bootstrap
    )
    separate_asinh_parameters = slope_parameter_table(
        separate_asinh_fit, separate_asinh_bootstrap
    )
    ols_parameters.to_csv(
        TABLES / "raw_mean_bias_parameters.csv", index=False
    )
    robust_parameters.to_csv(
        TABLES / "robust_capped_bias_parameters.csv", index=False
    )
    relative_parameters.to_csv(
        TABLES / "relative_bias_parameters.csv", index=False
    )
    asinh_parameters.to_csv(
        TABLES / "asinh_bias_parameters.csv", index=False
    )
    separate_raw_parameters.to_csv(
        TABLES / "separate_log_length_needle_raw_parameters.csv",
        index=False,
    )
    separate_asinh_parameters.to_csv(
        TABLES / "separate_log_length_needle_asinh_parameters.csv",
        index=False,
    )

    descriptive = descriptive_bootstrap(parsed, indices)
    descriptive.to_csv(
        TABLES / "model_bias_descriptive.csv", index=False
    )
    aggregates = aggregate_tables(parsed)
    for name, table in aggregates.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)

    make_figures(
        parsed,
        descriptive,
        comparison,
        ols_parameters,
        robust_parameters,
        separate_raw_parameters,
        separate_asinh_parameters,
        aggregates,
    )

    def comparison_row(
        method: str, target: str, candidate: str
    ) -> pd.Series:
        return comparison[
            (comparison["method"] == method)
            & (comparison["target"] == target)
            & (comparison["candidate"] == candidate)
        ].iloc[0]

    ols_row = comparison_row("ols", "raw_signed_error", ols_name)
    robust_row = comparison_row(
        "ols", "capped_signed_error", robust_name
    )
    relative_row = comparison_row(
        "ols", "relative_signed_error", relative_name
    )
    asinh_row = comparison_row(
        "ols", "asinh_signed_error", asinh_name
    )
    def significant_models(
        table: pd.DataFrame, estimand: str, adjusted: bool = True
    ) -> list[str]:
        column = f"{estimand}_bootstrap_p_two_sided"
        if adjusted:
            column = f"{column}_holm"
        if column not in table.columns:
            return []
        return table.loc[
            table[column].astype(float) < 0.05,
            "model_label",
        ].tolist()

    def formula_for(candidate: str, target: str) -> str:
        prefix = (
            f"E[{target}|parsed,m,c,o] = B_mco + "
        )
        if candidate == "log_density_only":
            return prefix + "d_m*ln((N/L)/(5/5000))"
        if candidate == "log_length_needles":
            return prefix + "u_m*ln(L/5000) + v_m*ln(N/5)"
        if candidate == "raw_length_needles":
            return prefix + "u_m*(L/5000-1) + v_m*(N/5-1)"
        return (
            prefix
            + "u_m*ln(L/5000) + v_m*ln(N/5) "
            "+ w_m*ln(L/5000)*ln(N/5)"
        )

    robust_effect_fields = [
        field
        for field in ["density_slope", "length_slope", "needle_slope"]
        if field in robust_parameters.columns
    ]
    robust_significance = {
        field: significant_models(robust_parameters, field)
        for field in robust_effect_fields
    }

    summary = {
        "status": "complete",
        "definition": "signed_error = predicted_count - true_num_needles",
        "model_size_used": False,
        "all_requests": len(full),
        "parsed_requests": len(parsed),
        "parse_coverage": len(parsed) / len(full),
        "unparsed_bias_missing_by_definition": len(full) - len(parsed),
        "overall": {
            "mean_bias": float(parsed["signed_error"].mean()),
            "median_bias": float(parsed["signed_error"].median()),
            "trimmed_mean_10pct": float(
                stats.trim_mean(
                    parsed["signed_error"].to_numpy(dtype=float), 0.1
                )
            ),
            "overcount_rate": float(
                np.mean(parsed["signed_error"] > 0)
            ),
            "undercount_rate": float(
                np.mean(parsed["signed_error"] < 0)
            ),
        },
        "raw_mean_bias_law": {
            "selected_candidate": ols_name,
            "formula": formula_for(ols_name, "bias"),
            "selection_score": float(ols_row["selection_score"]),
            "selection_metric": str(ols_row["selection_metric"]),
        },
        "robust_capped_bias_law": {
            "selected_candidate": robust_name,
            "formula": formula_for(robust_name, "capped-bias"),
            "cap": [-30, 30],
            "selection_score": float(robust_row["selection_score"]),
            "selection_metric": str(robust_row["selection_metric"]),
        },
        "relative_bias_law": {
            "selected_candidate": relative_name,
            "selection_score": float(
                relative_row["selection_score"]
            ),
        },
        "asinh_bias_law": {
            "selected_candidate": asinh_name,
            "selection_score": float(asinh_row["selection_score"]),
        },
        "bootstrap_significance": {
            "holm_alpha": 0.05,
            "models_with_nonzero_mean_bias_holm": significant_models(
                descriptive, "mean_bias"
            ),
            "models_with_nonzero_trimmed_mean_holm": significant_models(
                descriptive, "trimmed_mean_10pct"
            ),
            "selected_capped_effects_holm": robust_significance,
            "separate_asinh_length_effect_holm": significant_models(
                separate_asinh_parameters, "length_slope"
            ),
            "separate_asinh_needle_effect_holm": significant_models(
                separate_asinh_parameters, "needle_slope"
            ),
        },
        "separate_length_needle_diagnostic": {
            "coordinate": "ln(L/5000), ln(N/5)",
            "raw_mean_cv_selection_score": float(
                comparison_row(
                    "ols",
                    "raw_signed_error",
                    "log_length_needles",
                )["selection_score"]
            ),
            "asinh_cv_selection_score": float(
                comparison_row(
                    "ols",
                    "asinh_signed_error",
                    "log_length_needles",
                )["selection_score"]
            ),
            "warning": (
                "Retained for separate length/needle interpretation, but "
                "not promoted when blocked validation favors density."
            ),
        },
        "interpretation_warning": (
            "Mean signed bias is dominated by rare very large positive "
            "predictions; median, trimmed mean, capped and direction rates "
            "must be reported together."
        ),
    }
    write_json(METRICS / "summary.json", summary)
    write_json(
        METRICS / "raw_mean_bias_law.json",
        {
            **summary["raw_mean_bias_law"],
            "parameters": ols_parameters.to_dict(orient="records"),
        },
    )
    write_json(
        METRICS / "robust_bias_law.json",
        {
            **summary["robust_capped_bias_law"],
            "parameters": robust_parameters.to_dict(orient="records"),
        },
    )
    write_json(
        METRICS / "model_bias_descriptive.json",
        {
            "definition": summary["definition"],
            "models": descriptive.to_dict(orient="records"),
        },
    )

    model_lines = []
    for _, row in descriptive.iterrows():
        model_lines.append(
            f"- {row['model_label']}: mean={row['mean_bias']:.3f}, "
            f"95% CI [{row['mean_bias_ci95_low']:.3f}, "
            f"{row['mean_bias_ci95_high']:.3f}], "
            f"median={row['median_bias']:.3f}, "
            f"trimmed mean={row['trimmed_mean_10pct']:.3f}"
        )
    (OUT / "README.md").write_text(
        f"""# Signed counting bias

`bias = predicted_count - true_num_needles`.

Bias is defined for {len(parsed)}/{len(full)} parsed outputs
({len(parsed)/len(full):.2%}). The remaining {len(full)-len(parsed)} outputs
remain failures and receive no fabricated bias value.

## Descriptive bias

{chr(10).join(model_lines)}

The pooled mean bias is `{summary['overall']['mean_bias']:.3f}`, while the
pooled median is `{summary['overall']['median_bias']:.3f}`. This gap is caused
by rare extreme over-counts, so the raw mean alone is not representative.

## Fitted relations

- Conditional mean bias: `{ols_name}`
- Capped [-30, 30] robust bias: `{robust_name}`
- Relative bias: `{relative_name}`
- Sign-preserving asinh bias: `{asinh_name}`

See the parameter CSV files for clustered-bootstrap intervals and
per-doubling effects. The `*_holm` columns control family-wise error across
the eight model-specific tests. Separate log-length/log-needle fits are
retained as diagnostics even when blocked validation selects density. All
coordinate candidates and validation folds are retained.
""",
        encoding="utf-8",
    )

    set_state(
        "complete",
        selected_raw_mean_candidate=ols_name,
        selected_robust_candidate=robust_name,
    )
    log("bias analysis complete")

    excluded = {
        MANIFEST.name,
        INTEGRITY.name,
        "launcher_stdout.log",
        "launcher_stderr.log",
        "run.stdout.log",
        "run.stderr.log",
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
    write_json(
        MANIFEST,
        {
            "schema_version": "realistic_niah_signed_bias_law_v4",
            "created_at_utc": utc_now(),
            "analysis_root": str(OUT),
            "run_root": str(RUN),
            "filesystem_id": EXPECTED_FILESYSTEM_ID,
            "source": source_meta,
            "source_rows": len(full),
            "source_request_ids_unique": int(
                full["request_id"].nunique()
            ),
            "parsed_rows": len(parsed),
            "model_size_used": False,
            "bias_definition": summary["definition"],
            "robust_bias_cap": [-30, 30],
            "unparsed_handling": (
                "bias missing by definition; retained as failures in "
                "primary accuracy analysis"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "random_seed": RANDOM_SEED,
            "least_squares_solver": (
                "cross_product_solve_with_scipy_gelsy_fallback"
            ),
            "numerical_solver_diagnostic": (
                "metrics/numerical_solver_diagnostic.json"
            ),
            "validation_schemes": [
                "leave-one-seed-out",
                "leave-one-needle-level-out",
                "leave-one-length-level-out",
                "five blocked length-needle cell folds",
            ],
            "candidates": [
                {
                    "name": candidate.name,
                    "coordinate": candidate.coordinate,
                    "interaction": candidate.interaction,
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
            "artifact_integrity_sha256": sha256(INTEGRITY),
            "indexed_output_count": len(outputs),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_root": str(OUT),
                "parsed_requests": len(parsed),
                "mean_bias": summary["overall"]["mean_bias"],
                "median_bias": summary["overall"]["median_bias"],
                "selected_raw_mean_candidate": ols_name,
                "selected_robust_candidate": robust_name,
                "model_size_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
