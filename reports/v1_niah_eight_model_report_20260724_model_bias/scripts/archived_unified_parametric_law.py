#!/usr/bin/env python3
"""Select one parametric law family with model-specific parameters."""

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
from scipy.optimize import minimize
from scipy.special import expit


RUN = Path(
    "/lambda/nfs/Twist-CoT-Count-Multi-Model/runs/"
    "realistic_niah_v1/six_models_formal_20260723T194300Z"
)
SOURCE_ANALYSIS = RUN / "analysis" / "empirical_law_no_model_size_v1"
SOURCE = SOURCE_ANALYSIS / "tables" / "request_level_no_size.csv"
SOURCE_INTEGRITY = SOURCE_ANALYSIS / "artifact_integrity.json"
SOURCE_MANIFEST = SOURCE_ANALYSIS / "analysis_manifest.json"
OUT = RUN / "analysis" / "unified_parametric_law_v1"
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
BOOTSTRAP_REPLICATES = 200
RIDGE = 1e-4
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
PROMPTS = ["direct", "enumeration", "native_thinking"]
ORDERS = ["query_first", "query_last"]

PLAN_TEXT = """# Unified parametric law with model-specific parameters

## Scientific target

Select one bounded probability law `p_m(L, N)` shared by all eight models.
Each model receives its own baseline, length parameter and needle parameter.
No model-size or parameter-count variable is used. Prompt mode and query order
enter only as shared categorical nuisance modifiers.

The primary target is exact correctness over all 6,300 requests. Parse,
format and truncation failures remain incorrect.

## Frozen separable law families

For every model `m`, `x=L/5000` and `z=N/5`:

1. Hill power:
   `p = 1 / (1 + A_m x^r_m z^s_m)`
2. log-log stretched power:
   `p = exp(-A_m x^r_m z^s_m)`
3. complementary-log-log inverse-power:
   `p = 1 - exp(-B_m x^(-r_m) z^(-s_m))`
4. Hill exponential:
   `p = 1 / (1 + A_m exp(q_Lm(x-1) + q_Nm(z-1)))`
5. log-log stretched exponential in raw coordinates
6. complementary-log-log exponential in raw coordinates

A shared-order Hill model and a model-specific Hill interaction model are
diagnostics, not eligible to replace the selected separable model-specific
law.

## Validation

Every candidate is evaluated by:

- leave-one-seed-out validation;
- leave-one-needle-level-out validation;
- leave-one-length-level-out validation;
- five blocked `(length, needle)` cell folds.

The selection score is the equal-weight mean of the four complete OOF log
losses. This explicitly tests unseen coordinate levels rather than validating
only new seeds. The six separable model-specific candidates have the same
parameter count and are compared directly.

Uncertainty uses 200 bootstrap replicates clustered by complete stimulus ID.
The secondary parsed-output law compares power versus raw-coordinate
exponential forms for `log(1 + absolute count error)`. It remains conditional
on parse success and cannot replace the all-request accuracy law.

## Limits

Only three length levels are observed, so length functional-form
discrimination is intrinsically weaker than needle discrimination over ten
levels. All conclusions are restricted to 2K--10K tokens and 1--30 needles.
"""


@dataclass(frozen=True)
class Candidate:
    name: str
    link: str
    coordinate: str
    model_specific_slopes: bool = True
    interaction: bool = False
    eligible: bool = True
    formula: str = ""


CANDIDATES = [
    Candidate(
        "hill_power_model_specific",
        "logistic",
        "log",
        formula="p=1/(1+A_m*(L/L0)^r_m*(N/N0)^s_m)",
    ),
    Candidate(
        "loglog_power_model_specific",
        "loglog",
        "log",
        formula="p=exp(-A_m*(L/L0)^r_m*(N/N0)^s_m)",
    ),
    Candidate(
        "cloglog_power_model_specific",
        "cloglog",
        "log",
        formula="p=1-exp(-B_m*(L/L0)^(-r_m)*(N/N0)^(-s_m))",
    ),
    Candidate(
        "hill_exponential_model_specific",
        "logistic",
        "raw",
        formula="p=1/(1+A_m*exp(q_Lm*(L/L0-1)+q_Nm*(N/N0-1)))",
    ),
    Candidate(
        "loglog_exponential_model_specific",
        "loglog",
        "raw",
        formula="p=exp(-A_m*exp(q_Lm*(L/L0-1)+q_Nm*(N/N0-1)))",
    ),
    Candidate(
        "cloglog_exponential_model_specific",
        "cloglog",
        "raw",
        formula="p=1-exp(-B_m*exp(-q_Lm*(L/L0-1)-q_Nm*(N/N0-1)))",
    ),
    Candidate(
        "hill_power_shared_orders",
        "logistic",
        "log",
        model_specific_slopes=False,
        eligible=False,
        formula="p=1/(1+A_m*(L/L0)^r*(N/N0)^s)",
    ),
    Candidate(
        "hill_power_model_specific_interaction",
        "logistic",
        "log",
        interaction=True,
        eligible=False,
        formula=(
            "p=1/(1+A_m*(L/L0)^r_m*(N/N0)^s_m*"
            "exp(t_m*log(L/L0)*log(N/N0)))"
        ),
    ),
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


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    prior_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    prior_integrity = json.loads(SOURCE_INTEGRITY.read_text(encoding="utf-8"))
    if prior_manifest.get("filesystem_id") != EXPECTED_FILESYSTEM_ID:
        raise ValueError("Source analysis filesystem ID mismatch")
    indexed = {item["path"]: item for item in prior_integrity["files"]}
    key = "tables/request_level_no_size.csv"
    if key not in indexed:
        raise ValueError("Source request table is not integrity-indexed")
    item = indexed[key]
    if SOURCE.stat().st_size != item["bytes"] or sha256(SOURCE) != item["sha256"]:
        raise ValueError("Source request table SHA256 mismatch")
    frame = pd.read_csv(SOURCE)
    if len(frame) != EXPECTED_ROWS or frame["request_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("Source rows or request IDs are invalid")
    if "model_scale_b" in frame.columns:
        raise ValueError("No-model-size source unexpectedly contains model scale")
    frame["model_label"] = pd.Categorical(
        frame["model_label"], categories=MODELS, ordered=True
    )
    frame["prompt_mode"] = pd.Categorical(
        frame["prompt_mode"], categories=PROMPTS, ordered=True
    )
    frame["query_order"] = pd.Categorical(
        frame["query_order"], categories=ORDERS, ordered=True
    )
    frame["ln_length"] = np.log(
        frame["target_passage_tokens"].astype(float) / L0
    )
    frame["ln_needles"] = np.log(frame["num_needles"].astype(float) / N0)
    frame["raw_length"] = (
        frame["target_passage_tokens"].astype(float) / L0 - 1.0
    )
    frame["raw_needles"] = frame["num_needles"].astype(float) / N0 - 1.0
    frame["absolute_error"] = pd.to_numeric(
        frame["absolute_error"], errors="coerce"
    )
    parsed = frame["parse_success"].astype(int).eq(1)
    if frame.loc[parsed, "absolute_error"].isna().any():
        raise ValueError("Parsed rows contain missing absolute error")
    return frame, {
        "path": str(SOURCE),
        "bytes": SOURCE.stat().st_size,
        "sha256": sha256(SOURCE),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_integrity": str(SOURCE_INTEGRITY),
        "source_integrity_sha256": sha256(SOURCE_INTEGRITY),
    }


def coordinate_values(
    frame: pd.DataFrame, coordinate: str
) -> tuple[np.ndarray, np.ndarray]:
    if coordinate == "log":
        return (
            frame["ln_length"].to_numpy(dtype=float),
            frame["ln_needles"].to_numpy(dtype=float),
        )
    if coordinate == "raw":
        return (
            frame["raw_length"].to_numpy(dtype=float),
            frame["raw_needles"].to_numpy(dtype=float),
        )
    raise KeyError(coordinate)


def build_design(
    frame: pd.DataFrame, candidate: Candidate
) -> tuple[np.ndarray, list[str], np.ndarray]:
    length, needles = coordinate_values(frame, candidate.coordinate)
    model_values = frame["model_label"].astype(str).to_numpy()
    columns: list[np.ndarray] = []
    names: list[str] = []
    unpenalized: list[bool] = []
    for model in MODELS:
        indicator = (model_values == model).astype(float)
        columns.append(indicator)
        names.append(f"intercept[{model}]")
        unpenalized.append(True)
    if candidate.model_specific_slopes:
        for model in MODELS:
            indicator = (model_values == model).astype(float)
            columns.append(indicator * length)
            names.append(f"length[{model}]")
            unpenalized.append(False)
        for model in MODELS:
            indicator = (model_values == model).astype(float)
            columns.append(indicator * needles)
            names.append(f"needles[{model}]")
            unpenalized.append(False)
        if candidate.interaction:
            for model in MODELS:
                indicator = (model_values == model).astype(float)
                columns.append(indicator * length * needles)
                names.append(f"interaction[{model}]")
                unpenalized.append(False)
    else:
        columns.extend([length, needles])
        names.extend(["length[shared]", "needles[shared]"])
        unpenalized.extend([False, False])
    prompt_values = frame["prompt_mode"].astype(str).to_numpy()
    for prompt in PROMPTS[1:]:
        columns.append((prompt_values == prompt).astype(float))
        names.append(f"prompt[{prompt}]")
        unpenalized.append(False)
    order_values = frame["query_order"].astype(str).to_numpy()
    columns.append((order_values == ORDERS[1]).astype(float))
    names.append(f"order[{ORDERS[1]}]")
    unpenalized.append(False)
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite design: {candidate.name}")
    if any("size" in name or "param" in name for name in names):
        raise AssertionError("Forbidden model-size feature")
    penalty = np.where(np.array(unpenalized), 0.0, 1.0)
    return matrix, names, penalty


def inverse_link(
    linear: np.ndarray, link: str
) -> tuple[np.ndarray, np.ndarray]:
    if link == "logistic":
        probability = expit(linear)
        derivative = probability * (1.0 - probability)
    elif link == "loglog":
        clipped = np.clip(linear, -25.0, 6.0)
        transformed = np.exp(clipped)
        probability = np.exp(-transformed)
        derivative = -transformed * probability
    elif link == "cloglog":
        clipped = np.clip(linear, -25.0, 6.0)
        transformed = np.exp(clipped)
        survival = np.exp(-transformed)
        probability = 1.0 - survival
        derivative = transformed * survival
    else:
        raise KeyError(link)
    return np.clip(probability, 1e-9, 1 - 1e-9), derivative


def link_transform(probability: float, link: str) -> float:
    p = float(np.clip(probability, 1e-5, 1 - 1e-5))
    if link == "logistic":
        return math.log(p / (1.0 - p))
    if link == "loglog":
        return math.log(-math.log(p))
    if link == "cloglog":
        return math.log(-math.log(1.0 - p))
    raise KeyError(link)


def fit_binary(
    matrix: np.ndarray,
    outcome: np.ndarray,
    penalty: np.ndarray,
    link: str,
    model_values: np.ndarray,
) -> dict[str, Any]:
    y = outcome.astype(float)
    initial = np.zeros(matrix.shape[1], dtype=float)
    for index, model in enumerate(MODELS):
        mask = model_values == model
        if mask.any():
            initial[index] = link_transform(float(y[mask].mean()), link)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        probability, derivative = inverse_link(matrix @ beta, link)
        loss = float(
            -np.sum(
                y * np.log(probability)
                + (1.0 - y) * np.log(1.0 - probability)
            )
        )
        loss += 0.5 * RIDGE * float(np.sum(penalty * beta**2))
        gradient_eta = (
            (probability - y)
            / (probability * (1.0 - probability))
            * derivative
        )
        gradient = matrix.T @ gradient_eta + RIDGE * penalty * beta
        return loss, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-7},
    )
    probability, derivative = inverse_link(matrix @ result.x, link)
    fisher_weight = derivative**2 / (
        probability * (1.0 - probability)
    )
    hessian = matrix.T @ (fisher_weight[:, None] * matrix)
    hessian += RIDGE * np.diag(penalty)
    covariance = np.linalg.pinv(hessian)
    log_likelihood = float(
        np.sum(
            y * np.log(probability)
            + (1.0 - y) * np.log(1.0 - probability)
        )
    )
    return {
        "beta": result.x,
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
    covariance = (
        float(residual @ residual / degrees)
        * np.linalg.pinv(matrix.T @ matrix)
    )
    return {
        "beta": beta,
        "prediction": prediction,
        "covariance": covariance,
        "converged": True,
    }


def calibration_fit(
    outcome: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    p = np.clip(probability, 1e-8, 1 - 1e-8)
    x = np.column_stack(
        [np.ones(len(p)), np.log(p / (1.0 - p))]
    )
    y = outcome.astype(float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        fitted = np.clip(expit(x @ beta), 1e-9, 1 - 1e-9)
        loss = float(
            -np.sum(y * np.log(fitted) + (1.0 - y) * np.log(1.0 - fitted))
        )
        return loss, x.T @ (fitted - y)

    result = minimize(
        lambda beta: objective(beta)[0],
        np.array([0.0, 1.0]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
    )
    return float(result.x[0]), float(result.x[1])


def binary_metrics(
    outcome: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    y = outcome.astype(float)
    p = np.clip(probability.astype(float), 1e-8, 1 - 1e-8)
    log_loss = float(
        -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    )
    brier = float(np.mean((y - p) ** 2))
    assignments = np.clip(
        np.digitize(p, np.linspace(0, 1, 11), right=True) - 1, 0, 9
    )
    ece = 0.0
    for index in range(10):
        mask = assignments == index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(y[mask].mean() - p[mask].mean())
            )
    intercept, slope = calibration_fit(y, p)
    return {
        "log_loss": log_loss,
        "brier": brier,
        "ece": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def validation_schemes(
    frame: pd.DataFrame,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    schemes: dict[str, list[tuple[str, np.ndarray]]] = {}
    seeds = sorted(int(value) for value in frame["seed"].unique())
    schemes["seed"] = [
        (
            str(seed),
            frame["seed"].astype(int).to_numpy() == seed,
        )
        for seed in seeds
    ]
    needle_levels = sorted(
        int(value) for value in frame["num_needles"].unique()
    )
    schemes["needle_level"] = [
        (
            str(level),
            frame["num_needles"].astype(int).to_numpy() == level,
        )
        for level in needle_levels
    ]
    length_levels = sorted(
        int(value) for value in frame["target_passage_tokens"].unique()
    )
    schemes["length_level"] = [
        (
            str(level),
            frame["target_passage_tokens"].astype(int).to_numpy() == level,
        )
        for level in length_levels
    ]
    length_index = {
        value: index for index, value in enumerate(length_levels)
    }
    needle_index = {
        value: index for index, value in enumerate(needle_levels)
    }
    block = np.array(
        [
            (
                length_index[int(length)] * len(needle_levels)
                + needle_index[int(needles)]
            )
            % 5
            for length, needles in zip(
                frame["target_passage_tokens"], frame["num_needles"]
            )
        ]
    )
    schemes["cell_block"] = [
        (str(index), block == index) for index in range(5)
    ]
    return schemes


def evaluate_candidates(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    y = frame["exact_correct"].to_numpy(dtype=float)
    model_values = frame["model_label"].astype(str).to_numpy()
    schemes = validation_schemes(frame)
    comparisons = []
    fold_rows = []
    oof_predictions: dict[str, dict[str, np.ndarray]] = {}
    full_fits: dict[str, Any] = {}
    for candidate in CANDIDATES:
        matrix, names, penalty = build_design(frame, candidate)
        scheme_metrics: dict[str, dict[str, float]] = {}
        candidate_oof: dict[str, np.ndarray] = {}
        all_converged = True
        for scheme, folds in schemes.items():
            oof = np.full(len(frame), np.nan)
            for label, test in folds:
                train = ~test
                fitted = fit_binary(
                    matrix[train],
                    y[train],
                    penalty,
                    candidate.link,
                    model_values[train],
                )
                all_converged = all_converged and fitted["converged"]
                predicted, _ = inverse_link(
                    matrix[test] @ fitted["beta"], candidate.link
                )
                oof[test] = predicted
                fold_rows.append(
                    {
                        "candidate": candidate.name,
                        "scheme": scheme,
                        "held_out": label,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "converged": fitted["converged"],
                        **binary_metrics(y[test], predicted),
                    }
                )
            if np.isnan(oof).any():
                raise RuntimeError(
                    f"Incomplete OOF predictions: {candidate.name}, {scheme}"
                )
            scheme_metrics[scheme] = binary_metrics(y, oof)
            candidate_oof[scheme] = oof
        full = fit_binary(
            matrix, y, penalty, candidate.link, model_values
        )
        all_converged = all_converged and full["converged"]
        full_fits[candidate.name] = {
            "fit": full,
            "matrix": matrix,
            "names": names,
            "penalty": penalty,
            "candidate": candidate,
        }
        selection_score = float(
            np.mean(
                [
                    scheme_metrics[name]["log_loss"]
                    for name in [
                        "seed",
                        "needle_level",
                        "length_level",
                        "cell_block",
                    ]
                ]
            )
        )
        row: dict[str, Any] = {
            "candidate": candidate.name,
            "formula": candidate.formula,
            "link": candidate.link,
            "coordinate": candidate.coordinate,
            "eligible_separable_model_specific": candidate.eligible,
            "n_parameters": matrix.shape[1],
            "all_converged": all_converged,
            "selection_score": selection_score,
            "full_log_likelihood": full["log_likelihood"],
            "aic": 2 * matrix.shape[1] - 2 * full["log_likelihood"],
            "bic": math.log(len(frame)) * matrix.shape[1]
            - 2 * full["log_likelihood"],
        }
        for scheme, metrics in scheme_metrics.items():
            for metric, value in metrics.items():
                row[f"{scheme}_{metric}"] = value
        comparisons.append(row)
        oof_predictions[candidate.name] = candidate_oof
        log(
            f"{candidate.name}: selection={selection_score:.6f}, "
            f"converged={all_converged}"
        )
    return (
        pd.DataFrame(comparisons).sort_values("selection_score"),
        pd.DataFrame(fold_rows),
        oof_predictions,
        full_fits,
    )


def clustered_bootstrap_binary(
    frame: pd.DataFrame, fitted: dict[str, Any]
) -> np.ndarray:
    candidate: Candidate = fitted["candidate"]
    matrix = fitted["matrix"]
    penalty = fitted["penalty"]
    y = frame["exact_correct"].to_numpy(dtype=float)
    models = frame["model_label"].astype(str).to_numpy()
    clusters = frame["stimulus_id"].astype(str).to_numpy()
    unique = np.unique(clusters)
    by_cluster = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique
    }
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[value] for value in sampled])
        result = fit_binary(
            matrix[indices],
            y[indices],
            penalty,
            candidate.link,
            models[indices],
        )
        if result["converged"]:
            values.append(result["beta"])
    if len(values) < 180:
        raise RuntimeError("Too few converged binary bootstrap replicates")
    return np.vstack(values)


def canonical_parameters(
    beta: np.ndarray,
    candidate: Candidate,
    names: list[str],
    model: str,
) -> dict[str, float]:
    mapping = {name: index for index, name in enumerate(names)}
    intercept = float(beta[mapping[f"intercept[{model}]"]])
    if candidate.model_specific_slopes:
        length = float(beta[mapping[f"length[{model}]"]])
        needles = float(beta[mapping[f"needles[{model}]"]])
    else:
        length = float(beta[mapping["length[shared]"]])
        needles = float(beta[mapping["needles[shared]"]])
    if candidate.coordinate == "log":
        parameter_type = "power_order"
        if candidate.link == "logistic":
            amplitude = math.exp(-intercept)
            length_parameter = -length
            needle_parameter = -needles
        elif candidate.link == "loglog":
            amplitude = math.exp(intercept)
            length_parameter = length
            needle_parameter = needles
        else:
            amplitude = math.exp(intercept)
            length_parameter = -length
            needle_parameter = -needles
    else:
        parameter_type = "raw_coordinate_rate"
        if candidate.link == "logistic":
            amplitude = math.exp(-intercept)
            length_parameter = -length
            needle_parameter = -needles
        elif candidate.link == "loglog":
            amplitude = math.exp(intercept)
            length_parameter = length
            needle_parameter = needles
        else:
            amplitude = math.exp(intercept)
            length_parameter = -length
            needle_parameter = -needles
    baseline_probability, _ = inverse_link(
        np.array([intercept]), candidate.link
    )
    return {
        "intercept": intercept,
        "baseline_probability_L0_N0_direct_query_first": float(
            baseline_probability[0]
        ),
        "amplitude": amplitude,
        "length_parameter": length_parameter,
        "needle_parameter": needle_parameter,
        "parameter_type": parameter_type,
    }


def parameter_table(
    selected: dict[str, Any], bootstrap: np.ndarray
) -> pd.DataFrame:
    candidate: Candidate = selected["candidate"]
    beta = selected["fit"]["beta"]
    names = selected["names"]
    rows = []
    for model in MODELS:
        point = canonical_parameters(beta, candidate, names, model)
        draws = [
            canonical_parameters(draw, candidate, names, model)
            for draw in bootstrap
        ]
        row: dict[str, Any] = {
            "model_label": model,
            "selected_candidate": candidate.name,
            "selected_formula": candidate.formula,
            **point,
            "bootstrap_replicates": len(draws),
        }
        for field in [
            "baseline_probability_L0_N0_direct_query_first",
            "amplitude",
            "length_parameter",
            "needle_parameter",
        ]:
            values = np.array([draw[field] for draw in draws], dtype=float)
            row[f"{field}_ci95_low"] = float(np.percentile(values, 2.5))
            row[f"{field}_ci95_high"] = float(np.percentile(values, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def nuisance_table(
    selected: dict[str, Any], bootstrap: np.ndarray
) -> pd.DataFrame:
    names = selected["names"]
    beta = selected["fit"]["beta"]
    rows = []
    for feature in [
        "prompt[enumeration]",
        "prompt[native_thinking]",
        "order[query_last]",
    ]:
        index = names.index(feature)
        values = bootstrap[:, index]
        rows.append(
            {
                "feature": feature,
                "estimate_on_selected_link": float(beta[index]),
                "bootstrap_ci95_low": float(np.percentile(values, 2.5)),
                "bootstrap_ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_replicates": len(values),
            }
        )
    return pd.DataFrame(rows)


def per_model_validation(
    frame: pd.DataFrame,
    selected_name: str,
    oof: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    y = frame["exact_correct"].to_numpy(dtype=float)
    for model in MODELS:
        mask = frame["model_label"].astype(str).to_numpy() == model
        for scheme, probability in oof[selected_name].items():
            rows.append(
                {
                    "model_label": model,
                    "scheme": scheme,
                    "requests": int(mask.sum()),
                    "observed_accuracy": float(y[mask].mean()),
                    **binary_metrics(y[mask], probability[mask]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_error_laws(
    parsed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    definitions = [
        (
            "power_model_specific",
            Candidate("error_power", "ols", "log"),
        ),
        (
            "exponential_model_specific",
            Candidate("error_exponential", "ols", "raw"),
        ),
        (
            "power_shared_orders",
            Candidate(
                "error_power_shared",
                "ols",
                "log",
                model_specific_slopes=False,
                eligible=False,
            ),
        ),
        (
            "power_model_specific_interaction",
            Candidate(
                "error_power_interaction",
                "ols",
                "log",
                interaction=True,
                eligible=False,
            ),
        ),
    ]
    outcome = np.log1p(parsed["absolute_error"].to_numpy(dtype=float))
    absolute = parsed["absolute_error"].to_numpy(dtype=float)
    schemes = validation_schemes(parsed)
    comparisons = []
    fold_rows = []
    full_fits: dict[str, Any] = {}
    for name, candidate in definitions:
        matrix, feature_names, _ = build_design(parsed, candidate)
        scheme_values = {}
        for scheme, folds in schemes.items():
            oof = np.full(len(parsed), np.nan)
            for label, test in folds:
                train = ~test
                fitted = fit_ols(matrix[train], outcome[train])
                predicted = matrix[test] @ fitted["beta"]
                oof[test] = predicted
                predicted_abs = np.maximum(np.expm1(predicted), 0.0)
                fold_rows.append(
                    {
                        "candidate": name,
                        "scheme": scheme,
                        "held_out": label,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "log1p_mae": float(
                            np.mean(np.abs(outcome[test] - predicted))
                        ),
                        "absolute_unit_mae": float(
                            np.mean(np.abs(absolute[test] - predicted_abs))
                        ),
                    }
                )
            predicted_abs = np.maximum(np.expm1(oof), 0.0)
            scheme_values[scheme] = {
                "log1p_mae": float(np.mean(np.abs(outcome - oof))),
                "absolute_unit_mae": float(
                    np.mean(np.abs(absolute - predicted_abs))
                ),
                "absolute_unit_rmse": float(
                    np.sqrt(np.mean((absolute - predicted_abs) ** 2))
                ),
            }
        full = fit_ols(matrix, outcome)
        full_fits[name] = {
            "fit": full,
            "matrix": matrix,
            "names": feature_names,
            "candidate": candidate,
        }
        selection = float(
            np.mean(
                [
                    scheme_values[scheme]["log1p_mae"]
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
            "candidate": name,
            "coordinate": candidate.coordinate,
            "n_parameters": matrix.shape[1],
            "eligible_model_specific_separable": name
            in {"power_model_specific", "exponential_model_specific"},
            "selection_score_log1p_mae": selection,
        }
        for scheme, values in scheme_values.items():
            for metric, value in values.items():
                row[f"{scheme}_{metric}"] = value
        comparisons.append(row)
    return (
        pd.DataFrame(comparisons).sort_values("selection_score_log1p_mae"),
        pd.DataFrame(fold_rows),
        full_fits,
    )


def bootstrap_error(
    parsed: pd.DataFrame, selected: dict[str, Any]
) -> np.ndarray:
    matrix = selected["matrix"]
    outcome = np.log1p(parsed["absolute_error"].to_numpy(dtype=float))
    clusters = parsed["stimulus_id"].astype(str).to_numpy()
    unique = np.unique(clusters)
    by_cluster = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique
    }
    rng = np.random.default_rng(RANDOM_SEED + 1)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[value] for value in sampled])
        values.append(fit_ols(matrix[indices], outcome[indices])["beta"])
    return np.vstack(values)


def error_parameter_table(
    selected: dict[str, Any], bootstrap: np.ndarray
) -> pd.DataFrame:
    candidate: Candidate = selected["candidate"]
    names = selected["names"]
    mapping = {name: index for index, name in enumerate(names)}
    beta = selected["fit"]["beta"]
    rows = []
    for model in MODELS:
        intercept_index = mapping[f"intercept[{model}]"]
        length_index = mapping[f"length[{model}]"]
        needles_index = mapping[f"needles[{model}]"]
        if candidate.coordinate == "log":
            kind = "power_order"
        else:
            kind = "raw_coordinate_rate"
        values = {
            "amplitude_B": math.exp(float(beta[intercept_index])),
            "length_parameter": float(beta[length_index]),
            "needle_parameter": float(beta[needles_index]),
        }
        row: dict[str, Any] = {
            "model_label": model,
            "selected_error_coordinate": candidate.coordinate,
            "parameter_type": kind,
            **values,
            "bootstrap_replicates": len(bootstrap),
        }
        transformed = {
            "amplitude_B": np.exp(bootstrap[:, intercept_index]),
            "length_parameter": bootstrap[:, length_index],
            "needle_parameter": bootstrap[:, needles_index],
        }
        for field, draws in transformed.items():
            row[f"{field}_ci95_low"] = float(np.percentile(draws, 2.5))
            row[f"{field}_ci95_high"] = float(np.percentile(draws, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(
    frame: pd.DataFrame,
    comparison: pd.DataFrame,
    parameters: pd.DataFrame,
    selected_name: str,
    oof: dict[str, dict[str, np.ndarray]],
    selected: dict[str, Any],
    error_parameters: pd.DataFrame,
) -> None:
    plt.style.use("default")
    ordered = comparison.sort_values("selection_score")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [
        "#4472C4" if eligible else "#A5A5A5"
        for eligible in ordered["eligible_separable_model_specific"]
    ]
    ax.barh(ordered["candidate"], ordered["selection_score"], color=colors)
    ax.set_xlabel("Mean blocked/coordinate/seed OOF log loss")
    ax.set_title("Unified probability-law functional-form comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "functional_form_selection.png", dpi=180)
    plt.close(fig)

    schemes = ["seed", "needle_level", "length_level", "cell_block"]
    x = np.arange(len(ordered))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for index, scheme in enumerate(schemes):
        ax.bar(
            x + (index - 1.5) * width,
            ordered[f"{scheme}_log_loss"],
            width,
            label=scheme,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["candidate"], rotation=35, ha="right")
    ax.set_ylabel("OOF log loss")
    ax.set_title("Validation by unseen seeds, levels and cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_scheme_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    y = np.arange(len(parameters))
    for ax, field, title in [
        (axes[0], "length_parameter", "Length order/rate"),
        (axes[1], "needle_parameter", "Needle order/rate"),
    ]:
        lower = parameters[field] - parameters[f"{field}_ci95_low"]
        upper = parameters[f"{field}_ci95_high"] - parameters[field]
        ax.errorbar(
            parameters[field],
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
        )
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_yticks(y)
        ax.set_yticklabels(parameters["model_label"])
        ax.set_title(title)
        ax.set_xlabel(parameters["parameter_type"].iloc[0])
    fig.suptitle(f"Selected unified law parameters: {selected_name}")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_law_parameters.png", dpi=180)
    plt.close(fig)

    candidate: Candidate = selected["candidate"]
    lengths = sorted(
        int(value) for value in frame["target_passage_tokens"].unique()
    )
    needles = sorted(int(value) for value in frame["num_needles"].unique())
    synthetic_rows = []
    for model in MODELS:
        for length in lengths:
            for needle in needles:
                synthetic_rows.append(
                    {
                        "model_label": model,
                        "prompt_mode": "direct",
                        "query_order": "query_first",
                        "target_passage_tokens": length,
                        "num_needles": needle,
                        "ln_length": math.log(length / L0),
                        "ln_needles": math.log(needle / N0),
                        "raw_length": length / L0 - 1.0,
                        "raw_needles": needle / N0 - 1.0,
                    }
                )
    synthetic = pd.DataFrame(synthetic_rows)
    matrix, _, _ = build_design(synthetic, candidate)
    synthetic["probability"], _ = inverse_link(
        matrix @ selected["fit"]["beta"], candidate.link
    )
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODELS):
        part = synthetic[synthetic["model_label"] == model]
        for length, curve in part.groupby("target_passage_tokens"):
            curve = curve.sort_values("num_needles")
            ax.plot(
                curve["num_needles"],
                curve["probability"],
                marker="o",
                markersize=3,
                label=f"{int(length)}",
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
        ax.set_title(model)
    for ax in axes[-1, :]:
        ax.set_xlabel("Needle count")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted exact accuracy")
    axes[0, 0].legend(title="Tokens")
    fig.suptitle("Selected law at direct/query-first baseline")
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_law_surfaces.png", dpi=180)
    plt.close(fig)

    cell_oof = oof[selected_name]["cell_block"]
    plot_frame = frame.copy()
    plot_frame["oof_probability"] = cell_oof
    grouped = (
        plot_frame.groupby(
            ["model_label", "target_passage_tokens", "num_needles"],
            observed=False,
        )
        .agg(
            observed=("exact_correct", "mean"),
            predicted=("oof_probability", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODELS):
        part = grouped[grouped["model_label"].astype(str) == model]
        ax.scatter(part["predicted"], part["observed"], alpha=0.75, s=22)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(model)
    for ax in axes[-1, :]:
        ax.set_xlabel("OOF predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed")
    fig.suptitle("Held-out cell calibration by model")
    fig.tight_layout()
    fig.savefig(FIGURES / "heldout_cell_observed_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    y = np.arange(len(error_parameters))
    for ax, field, title in [
        (axes[0], "length_parameter", "Absolute-error length parameter"),
        (axes[1], "needle_parameter", "Absolute-error needle parameter"),
    ]:
        lower = (
            error_parameters[field]
            - error_parameters[f"{field}_ci95_low"]
        )
        upper = (
            error_parameters[f"{field}_ci95_high"]
            - error_parameters[field]
        )
        ax.errorbar(
            error_parameters[field],
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
            color="#70AD47",
        )
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_yticks(y)
        ax.set_yticklabels(error_parameters["model_label"])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES / "absolute_error_parameters.png", dpi=180)
    plt.close(fig)


def law_description(candidate: Candidate) -> str:
    if candidate.coordinate == "log":
        if candidate.link == "logistic":
            return (
                "p_m(L,N)=1/[1+A_m(L/5000)^{r_m}(N/5)^{s_m}]"
            )
        if candidate.link == "loglog":
            return "p_m(L,N)=exp[-A_m(L/5000)^{r_m}(N/5)^{s_m}]"
        return (
            "p_m(L,N)=1-exp[-B_m(L/5000)^{-r_m}(N/5)^{-s_m}]"
        )
    if candidate.link == "logistic":
        return (
            "p_m(L,N)=1/[1+A_m exp(q_Lm(L/5000-1)+"
            "q_Nm(N/5-1))]"
        )
    if candidate.link == "loglog":
        return (
            "p_m(L,N)=exp[-A_m exp(q_Lm(L/5000-1)+"
            "q_Nm(N/5-1))]"
        )
    return (
        "p_m(L,N)=1-exp[-B_m exp(-q_Lm(L/5000-1)-"
        "q_Nm(N/5-1))]"
    )


def main() -> None:
    for directory in (OUT, TABLES, FIGURES, METRICS):
        directory.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_text(encoding="utf-8") != PLAN_TEXT:
        raise RuntimeError("Existing plan differs from frozen plan")
    PLAN.write_text(PLAN_TEXT, encoding="utf-8")
    LOG.write_text("", encoding="utf-8")
    set_state("running")
    frame, source_meta = load_source()
    log("source SHA and row structure verified")

    comparison, fold_metrics, oof, full_fits = evaluate_candidates(frame)
    comparison.to_csv(TABLES / "functional_form_comparison.csv", index=False)
    fold_metrics.to_csv(TABLES / "functional_form_fold_metrics.csv", index=False)
    eligible = comparison[
        comparison["eligible_separable_model_specific"].astype(bool)
    ].sort_values("selection_score")
    selected_name = str(eligible.iloc[0]["candidate"])
    selected = full_fits[selected_name]
    selected_candidate: Candidate = selected["candidate"]
    bootstrap = clustered_bootstrap_binary(frame, selected)
    parameters = parameter_table(selected, bootstrap)
    parameters.to_csv(TABLES / "model_law_parameters.csv", index=False)
    nuisance = nuisance_table(selected, bootstrap)
    nuisance.to_csv(TABLES / "shared_nuisance_parameters.csv", index=False)
    validation = per_model_validation(frame, selected_name, oof)
    validation.to_csv(TABLES / "per_model_validation.csv", index=False)

    coefficient_rows = []
    beta = selected["fit"]["beta"]
    standard_error = np.sqrt(
        np.maximum(np.diag(selected["fit"]["covariance"]), 0.0)
    )
    for index, name in enumerate(selected["names"]):
        coefficient_rows.append(
            {
                "feature": name,
                "estimate": float(beta[index]),
                "standard_error": float(standard_error[index]),
                "bootstrap_ci95_low": float(
                    np.percentile(bootstrap[:, index], 2.5)
                ),
                "bootstrap_ci95_high": float(
                    np.percentile(bootstrap[:, index], 97.5)
                ),
                "bootstrap_replicates": len(bootstrap),
            }
        )
    pd.DataFrame(coefficient_rows).to_csv(
        TABLES / "selected_link_coefficients.csv", index=False
    )

    parsed = frame[frame["parse_success"].astype(int).eq(1)].copy()
    error_comparison, error_folds, error_fits = evaluate_error_laws(parsed)
    error_comparison.to_csv(
        TABLES / "absolute_error_form_comparison.csv", index=False
    )
    error_folds.to_csv(
        TABLES / "absolute_error_fold_metrics.csv", index=False
    )
    eligible_error = error_comparison[
        error_comparison["eligible_model_specific_separable"].astype(bool)
    ].sort_values("selection_score_log1p_mae")
    selected_error_name = str(eligible_error.iloc[0]["candidate"])
    selected_error = error_fits[selected_error_name]
    error_bootstrap = bootstrap_error(parsed, selected_error)
    error_parameters = error_parameter_table(
        selected_error, error_bootstrap
    )
    error_parameters.to_csv(
        TABLES / "absolute_error_law_parameters.csv", index=False
    )

    make_figures(
        frame,
        comparison,
        parameters,
        selected_name,
        oof,
        selected,
        error_parameters,
    )

    selected_row = comparison.set_index("candidate").loc[selected_name]
    runner_up = eligible.iloc[1]
    interaction_row = comparison.set_index("candidate").loc[
        "hill_power_model_specific_interaction"
    ]
    shared_row = comparison.set_index("candidate").loc[
        "hill_power_shared_orders"
    ]
    error_row = error_comparison.set_index("candidate").loc[
        selected_error_name
    ]
    summary = {
        "status": "complete",
        "model_size_used": False,
        "requests": len(frame),
        "selected_unified_law": {
            "candidate": selected_name,
            "formula": law_description(selected_candidate),
            "link": selected_candidate.link,
            "coordinate": selected_candidate.coordinate,
            "selection_score": float(selected_row["selection_score"]),
            "seed_oof_log_loss": float(selected_row["seed_log_loss"]),
            "needle_level_oof_log_loss": float(
                selected_row["needle_level_log_loss"]
            ),
            "length_level_oof_log_loss": float(
                selected_row["length_level_log_loss"]
            ),
            "cell_block_oof_log_loss": float(
                selected_row["cell_block_log_loss"]
            ),
            "runner_up": str(runner_up["candidate"]),
            "runner_up_selection_score": float(
                runner_up["selection_score"]
            ),
            "model_specific_parameters": True,
            "separable_length_and_needles": True,
        },
        "diagnostics": {
            "shared_order_hill_selection_score": float(
                shared_row["selection_score"]
            ),
            "interaction_hill_selection_score": float(
                interaction_row["selection_score"]
            ),
            "length_levels": sorted(
                int(value)
                for value in frame["target_passage_tokens"].unique()
            ),
            "needle_levels": sorted(
                int(value) for value in frame["num_needles"].unique()
            ),
            "length_form_identification": (
                "weak-to-moderate because only three length levels exist"
            ),
        },
        "absolute_error_secondary": {
            "conditional_on_parse_success": True,
            "parsed_requests": len(parsed),
            "parse_coverage": len(parsed) / len(frame),
            "selected_candidate": selected_error_name,
            "coordinate": selected_error["candidate"].coordinate,
            "selection_score_log1p_mae": float(
                error_row["selection_score_log1p_mae"]
            ),
            "seed_absolute_unit_mae": float(
                error_row["seed_absolute_unit_mae"]
            ),
            "needle_level_absolute_unit_mae": float(
                error_row["needle_level_absolute_unit_mae"]
            ),
            "length_level_absolute_unit_mae": float(
                error_row["length_level_absolute_unit_mae"]
            ),
            "cell_block_absolute_unit_mae": float(
                error_row["cell_block_absolute_unit_mae"]
            ),
        },
    }
    write_json(METRICS / "summary.json", summary)
    write_json(
        METRICS / "selected_law.json",
        {
            **summary["selected_unified_law"],
            "parameters": parameters.to_dict(orient="records"),
            "shared_nuisance_parameters": nuisance.to_dict(orient="records"),
        },
    )
    write_json(
        METRICS / "absolute_error_law.json",
        {
            **summary["absolute_error_secondary"],
            "parameters": error_parameters.to_dict(orient="records"),
        },
    )

    parameter_lines = []
    for _, row in parameters.iterrows():
        parameter_lines.append(
            f"- {row['model_label']}: A={row['amplitude']:.4g}, "
            f"length={row['length_parameter']:.3f}, "
            f"needles={row['needle_parameter']:.3f}"
        )
    (OUT / "README.md").write_text(
        f"""# Unified parametric law

## Selected family

`{law_description(selected_candidate)}`

The same bounded functional form is used for every model. Each model has its
own amplitude, length parameter and needle parameter. No model-size variable
is used.

Selection score: `{selected_row['selection_score']:.6f}`.
The score is the equal-weight mean of seed, unseen-needle, unseen-length and
held-out-cell OOF log loss.

## Model parameters

{chr(10).join(parameter_lines)}

See `tables/model_law_parameters.csv` for clustered-bootstrap intervals.

## Secondary error law

`log(1+absolute error)` uses the selected `{selected_error_name}` coordinate
family over the {len(parsed)} parsed outputs. Parse failures remain incorrect
in the primary probability law.

## Limitations

There are ten needle levels but only three length levels. Needle order/form
selection is materially better identified. Length-law extrapolation outside
2K--10K tokens is unsupported without additional length levels.
""",
        encoding="utf-8",
    )

    set_state(
        "complete",
        selected_candidate=selected_name,
        selected_error_candidate=selected_error_name,
    )
    log("unified parametric law analysis complete")

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
    write_json(
        MANIFEST,
        {
            "schema_version": "realistic_niah_unified_parametric_law_v1",
            "created_at_utc": utc_now(),
            "analysis_root": str(OUT),
            "run_root": str(RUN),
            "filesystem_id": EXPECTED_FILESYSTEM_ID,
            "source": source_meta,
            "source_rows": len(frame),
            "source_request_ids_unique": int(frame["request_id"].nunique()),
            "model_size_used": False,
            "selected_candidate": selected_name,
            "selected_formula": law_description(selected_candidate),
            "selected_error_candidate": selected_error_name,
            "validation_schemes": [
                "leave-one-seed-out",
                "leave-one-needle-level-out",
                "leave-one-length-level-out",
                "five blocked length-needle cell folds",
            ],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "random_seed": RANDOM_SEED,
            "candidates": [
                {
                    "name": candidate.name,
                    "link": candidate.link,
                    "coordinate": candidate.coordinate,
                    "model_specific_slopes": candidate.model_specific_slopes,
                    "interaction": candidate.interaction,
                    "eligible": candidate.eligible,
                    "formula": candidate.formula,
                }
                for candidate in CANDIDATES
            ],
            "failure_handling": (
                "all parse, format and truncation failures remain incorrect"
            ),
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
                "selected_candidate": selected_name,
                "selected_formula": law_description(selected_candidate),
                "selection_score": float(selected_row["selection_score"]),
                "selected_error_candidate": selected_error_name,
                "model_size_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
