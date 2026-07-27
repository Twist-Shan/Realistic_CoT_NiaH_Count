from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


L0 = 5000.0
N0 = 5.0
RIDGE = 1e-5
EPS = 1e-9
MODE_LABELS = {
    "direct": "Nonthinking / direct",
    "enumeration": "Enumeration",
    "native_thinking": "CoT / native thinking",
}
TARGET_LABELS = {
    "exact_correct": "Exact correctness (all requests)",
    "parse_success": "Parse success",
    "truncated": "Truncation",
    "exact_given_parsed": "Exact correctness | parsed",
    "within_one": "|count error| ≤ 1 | parsed",
    "undercount": "Undercount | parsed",
    "overcount": "Overcount | parsed",
    "all_pairs_found": "All gold pairs retrieved",
    "no_hallucinations": "No hallucinated pairs",
    "no_duplicates": "No duplicate listed pairs",
    "list_length_exact": "Listed-record count = N",
    "listed_total_consistent": "Reported total = list length | available",
    "numeric_list_consistent": "Parsed count = list length | parsed",
    "pair_retrieval": "Per-needle retrieval (binomial)",
    "log1p_absolute_error": "log(1 + absolute count error) | parsed",
    "log1p_relative_error": "log(1 + relative absolute error) | parsed",
    "signed_relative_error": "Signed error / N | parsed",
    "asinh_signed_error": "asinh(signed count error) | parsed",
    "signed_log_count_ratio": "log((predicted+0.5)/(N+0.5)) | parsed",
    "log1p_output_tokens": "log(1 + output tokens)",
    "missing_fraction": "Missing-pair fraction",
    "log1p_hallucination_rate": "log(1 + hallucinated pairs / N)",
    "log1p_duplicate_rate": "log(1 + duplicate pairs / N)",
    "listed_record_ratio": "Listed records / N",
    "pair_recall_continuous": "Pair recall (continuous diagnostic)",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    complexity: int
    description: str


CANDIDATES = [
    Candidate("intercept_only", (), 0, "Model and query-order baselines only"),
    Candidate("log_density", ("log_density",), 1, "log(N/L) density coordinate"),
    Candidate("log_burden", ("log_burden",), 1, "log(N×L) burden coordinate"),
    Candidate("power_separable", ("log_L", "log_N"), 2, "separate log L and log N orders"),
    Candidate(
        "power_interaction",
        ("log_L", "log_N", "log_interaction"),
        3,
        "power surface plus log L × log N",
    ),
    Candidate(
        "power_quadratic",
        ("log_L", "log_N", "log_L2", "log_N2", "log_interaction"),
        5,
        "quadratic response surface in log coordinates",
    ),
    Candidate(
        "piecewise_power",
        ("log_L", "log_N", "hinge_L", "hinge_N"),
        4,
        "piecewise log surface (L=5000, N=8 knots)",
    ),
    Candidate(
        "exponential_separable",
        ("raw_L", "raw_N"),
        2,
        "separate raw/scaled L and N exponential orders",
    ),
    Candidate(
        "exponential_interaction",
        ("raw_L", "raw_N", "raw_interaction"),
        3,
        "raw/scaled L,N surface plus interaction",
    ),
    Candidate(
        "root_separable",
        ("sqrt_L", "sqrt_N"),
        2,
        "separate square-root L and N coordinates",
    ),
    Candidate(
        "inverse_separable",
        ("inv_L", "inv_N"),
        2,
        "separate inverse L and N coordinates",
    ),
    Candidate(
        "logL_rawN",
        ("log_L", "raw_N"),
        2,
        "log length plus raw/scaled needle count",
    ),
    Candidate(
        "rawL_logN",
        ("raw_L", "log_N"),
        2,
        "raw/scaled length plus log needle count",
    ),
]


@dataclass
class TargetFrame:
    name: str
    frame: pd.DataFrame
    successes: np.ndarray | None = None
    trials: np.ndarray | None = None
    values: np.ndarray | None = None
    definition: str = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["L"] = data["target_passage_tokens"].astype(float)
    data["N"] = data["num_needles"].astype(float)
    data["log_L"] = np.log(data["L"] / L0)
    data["log_N"] = np.log(data["N"] / N0)
    data["log_density"] = data["log_N"] - data["log_L"]
    data["log_burden"] = data["log_N"] + data["log_L"]
    data["log_interaction"] = data["log_L"] * data["log_N"]
    data["log_L2"] = data["log_L"] ** 2
    data["log_N2"] = data["log_N"] ** 2
    data["hinge_L"] = np.maximum(data["log_L"], 0.0)
    data["hinge_N"] = np.maximum(data["log_N"] - math.log(8.0 / N0), 0.0)
    data["raw_L"] = data["L"] / L0 - 1.0
    data["raw_N"] = data["N"] / N0 - 1.0
    data["raw_interaction"] = data["raw_L"] * data["raw_N"]
    data["sqrt_L"] = np.sqrt(data["L"] / L0) - 1.0
    data["sqrt_N"] = np.sqrt(data["N"] / N0) - 1.0
    data["inv_L"] = L0 / data["L"] - 1.0
    data["inv_N"] = N0 / data["N"] - 1.0
    data["query_last"] = (data["query_order"] == "query_last").astype(float)
    return data


def mode_models(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["model_label"].unique().tolist())


def build_design(
    frame: pd.DataFrame, candidate: Candidate, models: list[str] | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    if models is None:
        models = mode_models(frame)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for model in models:
        indicator = (frame["model_label"].to_numpy() == model).astype(float)
        columns.append(indicator)
        names.append(f"{model}::intercept")
        columns.append(indicator * frame["query_last"].to_numpy(float))
        names.append(f"{model}::query_last")
        for feature in candidate.features:
            columns.append(indicator * frame[feature].to_numpy(float))
            names.append(f"{model}::{feature}")
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite matrix for {candidate.name}")
    return matrix, names, models


def inverse_link(eta: np.ndarray, link: str) -> tuple[np.ndarray, np.ndarray]:
    if link == "logistic":
        clipped = np.clip(eta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-clipped))
        derivative = p * (1.0 - p)
    elif link == "survival_loglog":
        clipped = np.clip(eta, -20.0, 4.0)
        hazard = np.exp(clipped)
        p = np.exp(-hazard)
        derivative = -hazard * p
    else:
        raise ValueError(link)
    return np.clip(p, EPS, 1.0 - EPS), derivative


def fit_binomial(
    matrix: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    link: str,
    ridge: float = RIDGE,
    max_iter: int = 30,
) -> dict[str, Any]:
    x = np.asarray(matrix, float)
    k = np.asarray(successes, float)
    w = np.asarray(trials, float)
    y = np.divide(k, w, out=np.zeros_like(k), where=w > 0)
    if link == "survival_loglog":
        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            raw_eta = x @ beta
            eta = np.clip(raw_eta, -20.0, 4.0)
            hazard = np.exp(eta)
            p = np.clip(np.exp(-hazard), EPS, 1.0 - EPS)
            value = float(
                -np.sum(k * np.log(p) + (w - k) * np.log(1.0 - p))
                + 0.5 * ridge * np.dot(beta, beta)
            )
            eta_gradient = hazard * (k - w * p) / (1.0 - p)
            eta_gradient *= ((raw_eta > -20.0) & (raw_eta < 4.0)).astype(float)
            gradient = x.T @ eta_gradient + ridge * beta
            return value, gradient

        result = scipy.optimize.minimize(
            objective,
            np.zeros(x.shape[1], dtype=float),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-6},
        )
        return {
            "beta": np.asarray(result.x, float),
            "converged": bool(result.success),
            "iterations": int(result.nit),
            "last_change": float(np.max(np.abs(result.jac))),
        }
    beta = np.zeros(x.shape[1], dtype=float)
    converged = False
    last_change = math.inf
    for iteration in range(max_iter):
        eta = x @ beta
        p, derivative = inverse_link(eta, link)
        derivative = np.where(
            np.abs(derivative) < 1e-8,
            np.where(derivative < 0, -1e-8, 1e-8),
            derivative,
        )
        variance = np.clip(p * (1.0 - p), 1e-8, None)
        weights = np.clip(w * derivative**2 / variance, 1e-8, 1e8)
        pseudo = eta + (y - p) / derivative
        xtw = x.T * weights
        lhs = xtw @ x + ridge * np.eye(x.shape[1])
        rhs = xtw @ pseudo
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        # Damping prevents survival-link oscillation near probability boundaries.
        if link == "survival_loglog":
            updated = 0.65 * updated + 0.35 * beta
        last_change = float(np.max(np.abs(updated - beta)))
        beta = updated
        if last_change < 1e-6:
            converged = True
            break
    return {
        "beta": beta,
        "converged": converged,
        "iterations": iteration + 1,
        "last_change": last_change,
    }


def predict_binomial(matrix: np.ndarray, beta: np.ndarray, link: str) -> np.ndarray:
    return inverse_link(matrix @ beta, link)[0]


def binomial_metrics(
    successes: np.ndarray, trials: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    k = np.asarray(successes, float)
    w = np.asarray(trials, float)
    p = np.clip(np.asarray(prediction, float), EPS, 1.0 - EPS)
    total = float(w.sum())
    frac = np.divide(k, w, out=np.zeros_like(k), where=w > 0)
    log_loss = float(
        -np.sum(k * np.log(p) + (w - k) * np.log(1.0 - p)) / total
    )
    brier = float(np.sum(w * (frac - p) ** 2) / total)
    return {"log_loss": log_loss, "brier": brier}


def continuous_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y, float)
    pred = np.asarray(prediction, float)
    residual = truth - pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else math.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


def fit_ols(matrix: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    x = np.asarray(matrix, float)
    target = np.asarray(y, float)
    lhs = x.T @ x + ridge * np.eye(x.shape[1])
    rhs = x.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def cell_fold_ids(frame: pd.DataFrame) -> np.ndarray:
    lengths = sorted(frame["target_passage_tokens"].unique().tolist())
    needles = sorted(frame["num_needles"].unique().tolist())
    lmap = {value: index for index, value in enumerate(lengths)}
    nmap = {value: index for index, value in enumerate(needles)}
    return np.array(
        [
            (2 * lmap[length] + 3 * nmap[needle]) % 5
            for length, needle in zip(
                frame["target_passage_tokens"], frame["num_needles"]
            )
        ],
        dtype=int,
    )


def fold_definitions(frame: pd.DataFrame, scheme: str) -> list[tuple[str, np.ndarray]]:
    if scheme == "seed":
        return [
            (f"seed={seed}", frame["seed"].to_numpy() == seed)
            for seed in sorted(frame["seed"].unique())
        ]
    if scheme == "cell":
        ids = cell_fold_ids(frame)
        return [(f"cell_fold={fold}", ids == fold) for fold in range(5)]
    if scheme == "length":
        return [
            (f"L={length}", frame["target_passage_tokens"].to_numpy() == length)
            for length in sorted(frame["target_passage_tokens"].unique())
        ]
    if scheme == "needle":
        return [
            (f"N={needle}", frame["num_needles"].to_numpy() == needle)
            for needle in sorted(frame["num_needles"].unique())
        ]
    raise ValueError(scheme)


def selected_by_one_se(comparison: pd.DataFrame, score_col: str) -> pd.Series:
    eligible = comparison[np.isfinite(comparison[score_col])].copy()
    convergence_column = next(
        (
            name
            for name in ("converged_all_folds", "converged")
            if name in eligible.columns
        ),
        None,
    )
    if convergence_column is not None and eligible[convergence_column].any():
        eligible = eligible[eligible[convergence_column]].copy()
    if eligible.empty:
        raise RuntimeError("No eligible candidate")
    best = eligible.sort_values(score_col).iloc[0]
    threshold = float(best[score_col] + max(float(best["selection_se"]), 0.0))
    within = eligible[eligible[score_col] <= threshold].copy()
    return within.sort_values(
        ["complexity", score_col, "candidate", "link"], ascending=True
    ).iloc[0]


def make_binary_targets(frame: pd.DataFrame, mode: str) -> dict[str, TargetFrame]:
    targets: dict[str, TargetFrame] = {}

    def add(
        name: str,
        mask: np.ndarray,
        successes: np.ndarray,
        trials: np.ndarray | None = None,
        definition: str = "",
    ) -> None:
        part = frame.loc[mask].copy()
        k = np.asarray(successes)[mask].astype(float)
        w = (
            np.ones(len(part), dtype=float)
            if trials is None
            else np.asarray(trials)[mask].astype(float)
        )
        targets[name] = TargetFrame(
            name=name, frame=part, successes=k, trials=w, definition=definition
        )

    all_mask = np.ones(len(frame), dtype=bool)
    parsed = frame["parse_success"].to_numpy() == 1
    add(
        "exact_correct",
        all_mask,
        frame["exact_correct"].to_numpy(),
        definition="Frozen exact correctness; every parse/format/truncation failure is 0.",
    )
    add("parse_success", all_mask, frame["parse_success"].to_numpy())
    add("truncated", all_mask, frame["truncated"].to_numpy())
    add(
        "exact_given_parsed",
        parsed,
        frame["exact_correct"].to_numpy(),
        definition="Exact numeric count among successfully parsed outputs only.",
    )
    add(
        "within_one",
        parsed,
        (frame["absolute_error"].fillna(np.inf).to_numpy() <= 1).astype(int),
    )
    add(
        "undercount",
        parsed,
        (frame["signed_error"].fillna(0).to_numpy() < 0).astype(int),
    )
    add(
        "overcount",
        parsed,
        (frame["signed_error"].fillna(0).to_numpy() > 0).astype(int),
    )
    if mode == "enumeration":
        add(
            "all_pairs_found",
            all_mask,
            (frame["missing_pairs_n"].to_numpy() == 0).astype(int),
        )
        add(
            "no_hallucinations",
            all_mask,
            (frame["hallucinated_pairs_n"].to_numpy() == 0).astype(int),
        )
        add(
            "no_duplicates",
            all_mask,
            (frame["duplicate_listed_pairs_n"].to_numpy() == 0).astype(int),
        )
        add(
            "list_length_exact",
            all_mask,
            (frame["listed_records_n"].to_numpy() == frame["N"].to_numpy()).astype(
                int
            ),
        )
        listed_raw = frame["listed_total_matches_length"]
        listed_available = listed_raw.notna().to_numpy()
        listed_bool = (
            listed_raw.astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
        )
        add(
            "listed_total_consistent",
            listed_available,
            listed_bool.to_numpy(),
        )
        numeric_consistent = (
            np.isfinite(frame["predicted_count"].to_numpy(float))
            & (
                np.abs(
                    frame["predicted_count"].fillna(-99999).to_numpy(float)
                    - frame["listed_records_n"].to_numpy(float)
                )
                < 1e-9
            )
        ).astype(int)
        add("numeric_list_consistent", parsed, numeric_consistent)
        retrieved = np.clip(
            frame["N"].to_numpy(float)
            - frame["missing_pairs_n"].to_numpy(float),
            0,
            frame["N"].to_numpy(float),
        )
        add(
            "pair_retrieval",
            all_mask,
            retrieved,
            trials=frame["N"].to_numpy(float),
            definition="Grouped binomial: successes=N−missing_pairs, trials=N.",
        )
    return targets


def make_continuous_targets(frame: pd.DataFrame, mode: str) -> dict[str, TargetFrame]:
    targets: dict[str, TargetFrame] = {}

    def add(
        name: str, mask: np.ndarray, values: np.ndarray, definition: str = ""
    ) -> None:
        part = frame.loc[mask].copy()
        targets[name] = TargetFrame(
            name=name,
            frame=part,
            values=np.asarray(values)[mask].astype(float),
            definition=definition,
        )

    parsed = (
        (frame["parse_success"].to_numpy() == 1)
        & np.isfinite(frame["predicted_count"].to_numpy(float))
    )
    all_mask = np.ones(len(frame), dtype=bool)
    add(
        "log1p_absolute_error",
        parsed,
        np.log1p(frame["absolute_error"].fillna(0).to_numpy(float)),
    )
    add(
        "log1p_relative_error",
        parsed,
        np.log1p(frame["normalized_absolute_error"].fillna(0).to_numpy(float)),
    )
    add(
        "signed_relative_error",
        parsed,
        frame["signed_error"].fillna(0).to_numpy(float) / frame["N"].to_numpy(float),
    )
    add(
        "asinh_signed_error",
        parsed,
        np.arcsinh(frame["signed_error"].fillna(0).to_numpy(float)),
    )
    nonnegative = parsed & (frame["predicted_count"].fillna(-1).to_numpy() >= 0)
    add(
        "signed_log_count_ratio",
        nonnegative,
        np.log(
            (frame["predicted_count"].fillna(0).to_numpy(float) + 0.5)
            / (frame["N"].to_numpy(float) + 0.5)
        ),
    )
    add(
        "log1p_output_tokens",
        all_mask,
        np.log1p(frame["output_tokens"].to_numpy(float)),
    )
    if mode == "enumeration":
        add(
            "missing_fraction",
            all_mask,
            frame["missing_pairs_n"].to_numpy(float) / frame["N"].to_numpy(float),
        )
        add(
            "log1p_hallucination_rate",
            all_mask,
            np.log1p(
                frame["hallucinated_pairs_n"].to_numpy(float)
                / frame["N"].to_numpy(float)
            ),
        )
        add(
            "log1p_duplicate_rate",
            all_mask,
            np.log1p(
                frame["duplicate_listed_pairs_n"].to_numpy(float)
                / frame["N"].to_numpy(float)
            ),
        )
        add(
            "listed_record_ratio",
            all_mask,
            frame["listed_records_n"].to_numpy(float) / frame["N"].to_numpy(float),
        )
        add(
            "pair_recall_continuous",
            all_mask,
            frame["pair_recall"].fillna(0).to_numpy(float),
        )
    return targets


def evaluate_binary_candidate(
    target: TargetFrame,
    candidate: Candidate,
    link: str,
    schemes: tuple[str, ...] = ("seed", "cell"),
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    frame = target.frame
    matrix, _, _ = build_design(frame, candidate)
    fold_rows: list[dict[str, Any]] = []
    oof_by_scheme: dict[str, np.ndarray] = {}
    scheme_metrics: dict[str, float] = {}
    all_converged = True
    combined_fold_losses: list[float] = []
    for scheme in schemes:
        oof = np.full(len(frame), np.nan)
        for fold_label, test in fold_definitions(frame, scheme):
            train = ~test
            fit = fit_binomial(
                matrix[train],
                target.successes[train],
                target.trials[train],
                link,
            )
            prediction = predict_binomial(matrix[test], fit["beta"], link)
            oof[test] = prediction
            metrics = binomial_metrics(
                target.successes[test], target.trials[test], prediction
            )
            all_converged = all_converged and bool(fit["converged"])
            combined_fold_losses.append(metrics["log_loss"])
            fold_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold_label,
                    "candidate": candidate.name,
                    "link": link,
                    "n_rows": int(test.sum()),
                    "n_trials": float(target.trials[test].sum()),
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                    "converged": bool(fit["converged"]),
                    "iterations": int(fit["iterations"]),
                }
            )
        if np.isnan(oof).any():
            raise RuntimeError(f"Missing OOF values: {scheme}/{candidate.name}/{link}")
        metrics = binomial_metrics(target.successes, target.trials, oof)
        scheme_metrics[f"{scheme}_log_loss"] = metrics["log_loss"]
        scheme_metrics[f"{scheme}_brier"] = metrics["brier"]
        oof_by_scheme[scheme] = oof
    score = float(
        np.mean([scheme_metrics[f"{scheme}_log_loss"] for scheme in schemes])
    )
    se = (
        float(np.std(combined_fold_losses, ddof=1) / math.sqrt(len(combined_fold_losses)))
        if len(combined_fold_losses) > 1
        else 0.0
    )
    row = {
        "candidate": candidate.name,
        "link": link,
        "complexity": candidate.complexity,
        "description": candidate.description,
        "selection_score": score,
        "selection_se": se,
        "converged_all_folds": all_converged,
        **scheme_metrics,
    }
    return row, fold_rows, oof_by_scheme


def evaluate_continuous_candidate(
    target: TargetFrame,
    candidate: Candidate,
    schemes: tuple[str, ...] = ("seed", "cell"),
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    frame = target.frame
    y = target.values
    matrix, _, _ = build_design(frame, candidate)
    scale = float(np.std(y))
    if scale < 1e-8:
        scale = 1.0
    fold_rows: list[dict[str, Any]] = []
    oof_by_scheme: dict[str, np.ndarray] = {}
    scheme_metrics: dict[str, float] = {}
    combined_nrmse: list[float] = []
    for scheme in schemes:
        oof = np.full(len(frame), np.nan)
        for fold_label, test in fold_definitions(frame, scheme):
            train = ~test
            beta = fit_ols(matrix[train], y[train])
            prediction = matrix[test] @ beta
            oof[test] = prediction
            metrics = continuous_metrics(y[test], prediction)
            nrmse = metrics["rmse"] / scale
            combined_nrmse.append(nrmse)
            fold_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold_label,
                    "candidate": candidate.name,
                    "link": "identity",
                    "n_rows": int(test.sum()),
                    "rmse": metrics["rmse"],
                    "nrmse": nrmse,
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                }
            )
        if np.isnan(oof).any():
            raise RuntimeError(f"Missing OOF values: {scheme}/{candidate.name}")
        metrics = continuous_metrics(y, oof)
        scheme_metrics[f"{scheme}_rmse"] = metrics["rmse"]
        scheme_metrics[f"{scheme}_nrmse"] = metrics["rmse"] / scale
        scheme_metrics[f"{scheme}_mae"] = metrics["mae"]
        scheme_metrics[f"{scheme}_r2"] = metrics["r2"]
        oof_by_scheme[scheme] = oof
    score = float(np.mean([scheme_metrics[f"{s}_nrmse"] for s in schemes]))
    se = (
        float(np.std(combined_nrmse, ddof=1) / math.sqrt(len(combined_nrmse)))
        if len(combined_nrmse) > 1
        else 0.0
    )
    row = {
        "candidate": candidate.name,
        "link": "identity",
        "complexity": candidate.complexity,
        "description": candidate.description,
        "selection_score": score,
        "selection_se": se,
        **scheme_metrics,
    }
    return row, fold_rows, oof_by_scheme


def selected_candidate(name: str) -> Candidate:
    return next(candidate for candidate in CANDIDATES if candidate.name == name)


def cell_level_binary_metrics(
    frame: pd.DataFrame,
    successes: np.ndarray,
    trials: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    temp = frame[
        ["model_label", "target_passage_tokens", "num_needles"]
    ].copy()
    temp["successes"] = successes
    temp["trials"] = trials
    temp["pred_numerator"] = prediction * trials
    cells = (
        temp.groupby(
            ["model_label", "target_passage_tokens", "num_needles"], as_index=False
        )
        .agg(
            successes=("successes", "sum"),
            trials=("trials", "sum"),
            pred_numerator=("pred_numerator", "sum"),
        )
        .assign(
            observed=lambda x: x["successes"] / x["trials"],
            predicted=lambda x: x["pred_numerator"] / x["trials"],
        )
    )
    metrics = continuous_metrics(
        cells["observed"].to_numpy(), cells["predicted"].to_numpy()
    )
    return {
        "cell_r2": metrics["r2"],
        "cell_rmse": metrics["rmse"],
        "cell_mae": metrics["mae"],
        "n_cells": int(len(cells)),
    }


def fit_selected_binary(
    mode: str,
    target: TargetFrame,
    selected: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    candidate = selected_candidate(str(selected["candidate"]))
    link = str(selected["link"])
    matrix, names, _ = build_design(target.frame, candidate)
    full_fit = fit_binomial(matrix, target.successes, target.trials, link)
    coefficients = pd.DataFrame(
        {
            "mode": mode,
            "target": target.name,
            "candidate": candidate.name,
            "link": link,
            "term": names,
            "estimate": full_fit["beta"],
        }
    )
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "target": target.name,
        "candidate": candidate.name,
        "link": link,
        "n_rows": int(len(target.frame)),
        "n_trials": float(target.trials.sum()),
        "converged_full": bool(full_fit["converged"]),
    }
    oofs: dict[str, np.ndarray] = {}
    for scheme in ("seed", "cell", "length", "needle"):
        oof = np.full(len(target.frame), np.nan)
        fold_losses: list[float] = []
        for _, test in fold_definitions(target.frame, scheme):
            train = ~test
            fit = fit_binomial(
                matrix[train],
                target.successes[train],
                target.trials[train],
                link,
            )
            oof[test] = predict_binomial(matrix[test], fit["beta"], link)
            fold_losses.append(
                binomial_metrics(
                    target.successes[test], target.trials[test], oof[test]
                )["log_loss"]
            )
        metrics = binomial_metrics(target.successes, target.trials, oof)
        cell_metrics = cell_level_binary_metrics(
            target.frame, target.successes, target.trials, oof
        )
        diagnostics.update(
            {
                f"{scheme}_log_loss": metrics["log_loss"],
                f"{scheme}_brier": metrics["brier"],
                f"{scheme}_cell_r2": cell_metrics["cell_r2"],
                f"{scheme}_cell_rmse": cell_metrics["cell_rmse"],
                f"{scheme}_fold_log_loss_sd": float(np.std(fold_losses, ddof=1))
                if len(fold_losses) > 1
                else 0.0,
            }
        )
        oofs[scheme] = oof
    return coefficients, diagnostics, oofs


def fit_selected_continuous(
    mode: str,
    target: TargetFrame,
    selected: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    candidate = selected_candidate(str(selected["candidate"]))
    matrix, names, _ = build_design(target.frame, candidate)
    beta = fit_ols(matrix, target.values)
    coefficients = pd.DataFrame(
        {
            "mode": mode,
            "target": target.name,
            "candidate": candidate.name,
            "link": "identity",
            "term": names,
            "estimate": beta,
        }
    )
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "target": target.name,
        "candidate": candidate.name,
        "link": "identity",
        "n_rows": int(len(target.frame)),
    }
    oofs: dict[str, np.ndarray] = {}
    for scheme in ("seed", "cell", "length", "needle"):
        oof = np.full(len(target.frame), np.nan)
        for _, test in fold_definitions(target.frame, scheme):
            train = ~test
            fold_beta = fit_ols(matrix[train], target.values[train])
            oof[test] = matrix[test] @ fold_beta
        metrics = continuous_metrics(target.values, oof)
        diagnostics.update(
            {
                f"{scheme}_rmse": metrics["rmse"],
                f"{scheme}_mae": metrics["mae"],
                f"{scheme}_r2": metrics["r2"],
            }
        )
        oofs[scheme] = oof
    return coefficients, diagnostics, oofs


def nested_primary_exact(
    mode: str, target: TargetFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    frame = target.frame
    outer_oof = np.full(len(frame), np.nan)
    selection_rows: list[dict[str, Any]] = []
    for outer_label, outer_test in fold_definitions(frame, "cell"):
        outer_train = ~outer_test
        inner_rows: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            matrix, _, _ = build_design(frame, candidate)
            for link in ("logistic", "survival_loglog"):
                losses: list[float] = []
                converged = True
                for inner_label, inner_test_all in fold_definitions(frame, "seed"):
                    inner_test = inner_test_all & outer_train
                    inner_train = outer_train & ~inner_test_all
                    fit = fit_binomial(
                        matrix[inner_train],
                        target.successes[inner_train],
                        target.trials[inner_train],
                        link,
                    )
                    prediction = predict_binomial(
                        matrix[inner_test], fit["beta"], link
                    )
                    losses.append(
                        binomial_metrics(
                            target.successes[inner_test],
                            target.trials[inner_test],
                            prediction,
                        )["log_loss"]
                    )
                    converged = converged and bool(fit["converged"])
                inner_rows.append(
                    {
                        "candidate": candidate.name,
                        "link": link,
                        "complexity": candidate.complexity,
                        "selection_score": float(np.mean(losses)),
                        "selection_se": float(
                            np.std(losses, ddof=1) / math.sqrt(len(losses))
                        ),
                        "converged": converged,
                    }
                )
        chosen = selected_by_one_se(pd.DataFrame(inner_rows), "selection_score")
        chosen_candidate = selected_candidate(str(chosen["candidate"]))
        matrix, _, _ = build_design(frame, chosen_candidate)
        fit = fit_binomial(
            matrix[outer_train],
            target.successes[outer_train],
            target.trials[outer_train],
            str(chosen["link"]),
        )
        prediction = predict_binomial(
            matrix[outer_test], fit["beta"], str(chosen["link"])
        )
        outer_oof[outer_test] = prediction
        fold_metrics = binomial_metrics(
            target.successes[outer_test], target.trials[outer_test], prediction
        )
        selection_rows.append(
            {
                "mode": mode,
                "outer_fold": outer_label,
                "selected_candidate": chosen["candidate"],
                "selected_link": chosen["link"],
                "inner_log_loss": chosen["selection_score"],
                "outer_log_loss": fold_metrics["log_loss"],
                "outer_brier": fold_metrics["brier"],
                "n_test": int(outer_test.sum()),
            }
        )
    metrics = binomial_metrics(target.successes, target.trials, outer_oof)
    metrics.update(
        cell_level_binary_metrics(
            frame, target.successes, target.trials, outer_oof
        )
    )
    pred_frame = frame[
        [
            "request_id",
            "stimulus_id",
            "seed",
            "model_label",
            "target_passage_tokens",
            "num_needles",
            "query_order",
            "exact_correct",
        ]
    ].copy()
    pred_frame["mode"] = mode
    pred_frame["nested_cell_oof_probability"] = outer_oof
    return pd.DataFrame(selection_rows), pred_frame, metrics


def candidate_formula(candidate: str, link: str) -> str:
    if candidate == "power_separable":
        linear = "a_m + r_m log(L/5000) + s_m log(N/5) + o_m I(query-last)"
    elif candidate == "log_density":
        linear = "a_m + d_m log[(N/5)/(L/5000)] + o_m I(query-last)"
    elif candidate == "log_burden":
        linear = "a_m + b_m log[(N/5)(L/5000)] + o_m I(query-last)"
    elif candidate == "power_interaction":
        linear = (
            "a_m + r_m log(L/5000) + s_m log(N/5) "
            "+ i_m log(L/5000)log(N/5) + o_m I(query-last)"
        )
    elif candidate == "power_quadratic":
        linear = "model-specific quadratic surface in log(L/5000), log(N/5)"
    elif candidate == "piecewise_power":
        linear = "model-specific piecewise log surface with L=5000 and N=8 knots"
    elif candidate == "intercept_only":
        linear = "a_m + o_m I(query-last)"
    else:
        linear = f"model-specific {candidate} surface plus query-order nuisance"
    if link == "logistic":
        return f"logit(p) = {linear}"
    if link == "survival_loglog":
        return f"p = exp[-exp({linear})]"
    return f"E[target] = {linear}"


def extract_model_orders(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mode, target, candidate, link), group in coefficients.groupby(
        ["mode", "target", "candidate", "link"], dropna=False
    ):
        for model in sorted(
            {str(term).split("::", 1)[0] for term in group["term"]}
        ):
            values = {
                str(row.term).split("::", 1)[1]: float(row.estimate)
                for row in group.itertuples()
                if str(row.term).startswith(model + "::")
            }
            rows.append(
                {
                    "mode": mode,
                    "target": target,
                    "model_label": model,
                    "candidate": candidate,
                    "link": link,
                    "intercept": values.get("intercept", math.nan),
                    "query_last": values.get("query_last", math.nan),
                    "length_order_log": values.get("log_L", math.nan),
                    "needle_order_log": values.get("log_N", math.nan),
                    "log_interaction": values.get("log_interaction", math.nan),
                    "density_order": values.get("log_density", math.nan),
                    "burden_order": values.get("log_burden", math.nan),
                    "raw_length_order": values.get("raw_L", math.nan),
                    "raw_needle_order": values.get("raw_N", math.nan),
                }
            )
    return pd.DataFrame(rows)


def enumeration_compounding(
    targets: dict[str, TargetFrame],
    selected_oofs: dict[tuple[str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retrieval = targets["pair_retrieval"]
    all_found = targets["all_pairs_found"]
    exact = targets["exact_correct"]
    rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for scheme in ("seed", "cell"):
        q = np.clip(
            selected_oofs[("enumeration", "pair_retrieval", scheme)], EPS, 1 - EPS
        )
        compounded = q ** retrieval.frame["N"].to_numpy(float)
        for target_name, target in (
            ("all_pairs_found", all_found),
            ("exact_correct", exact),
        ):
            # Enumeration targets share identical row ordering and complete rows.
            metrics = binomial_metrics(
                target.successes, target.trials, compounded
            )
            cell = cell_level_binary_metrics(
                target.frame, target.successes, target.trials, compounded
            )
            rows.append(
                {
                    "scheme": scheme,
                    "prediction": "q_hat^N",
                    "target": target_name,
                    **metrics,
                    **cell,
                }
            )
        temp = retrieval.frame[
            [
                "model_label",
                "target_passage_tokens",
                "num_needles",
                "query_order",
            ]
        ].copy()
        temp["q_hat"] = q
        temp["q_hat_power_N"] = compounded
        temp["all_pairs_found"] = all_found.successes
        temp["exact_correct"] = exact.successes
        temp["scheme"] = scheme
        cell_rows.append(temp)
    return pd.DataFrame(rows), pd.concat(cell_rows, ignore_index=True)


def save_exact_candidate_figure(
    binary_comparison: pd.DataFrame, path: Path
) -> None:
    modes = list(MODE_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)
    for ax, mode in zip(axes, modes):
        part = binary_comparison[
            (binary_comparison["mode"] == mode)
            & (binary_comparison["target"] == "exact_correct")
        ].nsmallest(12, "selection_score")
        labels = part["candidate"] + " / " + part["link"].str.replace(
            "survival_loglog", "survival", regex=False
        )
        order = np.arange(len(part))
        ax.barh(order, part["selection_score"], color="#3a6ea5")
        ax.set_yticks(order, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Mean grouped held-out log loss\n(lower is better)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Exact-count candidate search by prompt mode", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_oof_calibration_figure(
    primary_predictions: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, mode in zip(axes, MODE_LABELS):
        part = primary_predictions[primary_predictions["mode"] == mode].copy()
        cells = (
            part.groupby(
                ["model_label", "target_passage_tokens", "num_needles"],
                as_index=False,
            )
            .agg(
                observed=("exact_correct", "mean"),
                predicted=("nested_cell_oof_probability", "mean"),
            )
        )
        for model, group in cells.groupby("model_label"):
            ax.scatter(
                group["predicted"],
                group["observed"],
                s=18,
                alpha=0.72,
                label=model,
            )
        ax.plot([0, 1], [0, 1], "--", color="black", lw=1)
        metrics = continuous_metrics(
            cells["observed"].to_numpy(), cells["predicted"].to_numpy()
        )
        ax.text(
            0.03,
            0.97,
            f"held-out cell R²={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("Nested blocked-cell predicted probability")
        ax.set_ylabel("Observed cell accuracy")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Primary exact-count law: fully held-out length–needle cells", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_order_heatmap(orders: pd.DataFrame, path: Path) -> None:
    primary = orders[orders["target"] == "exact_correct"].copy()
    models = sorted(primary["model_label"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.5), constrained_layout=True)
    for ax, term, title in (
        (axes[0], "length_order_log", "Coefficient on log(L/5000)"),
        (axes[1], "needle_order_log", "Coefficient on log(N/5)"),
    ):
        matrix = (
            primary.pivot(index="mode", columns="model_label", values=term)
            .reindex(index=list(MODE_LABELS), columns=models)
            .to_numpy(float)
        )
        finite = np.abs(matrix[np.isfinite(matrix)])
        vmax = max(float(np.quantile(finite, 0.95)) if len(finite) else 1.0, 0.2)
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
        ax.set_yticks(
            range(len(MODE_LABELS)),
            [MODE_LABELS[mode] for mode in MODE_LABELS],
        )
        ax.set_title(title)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    "—" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(
        "Selected exact-law log-coordinate orders (blank if another coordinate won)",
        fontsize=14,
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_mechanism_performance_figure(
    selected_diagnostics: pd.DataFrame, path: Path
) -> None:
    binary = selected_diagnostics[
        selected_diagnostics["kind"] == "binary"
    ].copy()
    display_targets = [
        "exact_correct",
        "parse_success",
        "truncated",
        "exact_given_parsed",
        "within_one",
        "pair_retrieval",
        "all_pairs_found",
        "list_length_exact",
    ]
    modes = list(MODE_LABELS)
    matrix = np.full((len(display_targets), len(modes)), np.nan)
    for row, target in enumerate(display_targets):
        for col, mode in enumerate(modes):
            part = binary[
                (binary["mode"] == mode) & (binary["target"] == target)
            ]
            if not part.empty:
                matrix[row, col] = float(part.iloc[0]["cell_cell_r2"])
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    finite = matrix[np.isfinite(matrix)]
    vmin = min(float(finite.min()) if len(finite) else 0.0, 0.0)
    image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(modes)), [MODE_LABELS[m] for m in modes], rotation=15)
    ax.set_yticks(
        range(len(display_targets)),
        [TARGET_LABELS.get(target, target) for target in display_targets],
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                "—" if not np.isfinite(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if np.isfinite(value) and value < 0.55 else "black",
                fontsize=9,
            )
    ax.set_title("Blocked-cell OOF R² of selected mechanism outcomes")
    fig.colorbar(image, ax=ax, label="R² across model × L × N cell means")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_enumeration_compounding_figure(
    compounding_cells: pd.DataFrame, path: Path
) -> None:
    data = compounding_cells[compounding_cells["scheme"] == "cell"].copy()
    cells = (
        data.groupby(
            ["model_label", "target_passage_tokens", "num_needles"], as_index=False
        )
        .agg(
            q_hat_power_N=("q_hat_power_N", "mean"),
            all_pairs_found=("all_pairs_found", "mean"),
            exact_correct=("exact_correct", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, target, title in (
        (axes[0], "all_pairs_found", "All gold pairs retrieved"),
        (axes[1], "exact_correct", "Exact reported count"),
    ):
        for model, part in cells.groupby("model_label"):
            ax.scatter(
                part["q_hat_power_N"],
                part[target],
                s=20,
                alpha=0.75,
                label=model,
            )
        ax.plot([0, 1], [0, 1], "--", color="black", lw=1)
        metrics = continuous_metrics(
            cells[target].to_numpy(), cells["q_hat_power_N"].to_numpy()
        )
        ax.text(
            0.03,
            0.97,
            f"cell R²={metrics['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted q(L,N)^N")
        ax.set_ylabel("Observed cell rate")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Enumeration retrieval-compounding test (blocked-cell predictions)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_continuous_target_figure(
    diagnostics: pd.DataFrame, path: Path
) -> None:
    part = diagnostics[
        (diagnostics["kind"] == "continuous")
        & diagnostics["target"].isin(
            [
                "log1p_absolute_error",
                "log1p_relative_error",
                "signed_relative_error",
                "asinh_signed_error",
                "log1p_output_tokens",
                "missing_fraction",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(12.5, 6))
    pivot = part.pivot(index="target", columns="mode", values="cell_r2")
    pivot = pivot.reindex(
        index=[
            "log1p_absolute_error",
            "log1p_relative_error",
            "signed_relative_error",
            "asinh_signed_error",
            "log1p_output_tokens",
            "missing_fraction",
        ],
        columns=list(MODE_LABELS),
    )
    x = np.arange(len(pivot.index))
    width = 0.24
    for index, mode in enumerate(MODE_LABELS):
        values = pivot[mode].to_numpy(float)
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=MODE_LABELS[mode],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(
        x,
        [TARGET_LABELS.get(target, target) for target in pivot.index],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Blocked-cell OOF request-level R²")
    ax.set_title("Predictability of transformed error and resource targets")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def dataframe_html(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    data = frame if columns is None else frame[columns]
    return data.to_html(index=False, border=0, classes="data-table", na_rep="—")


def build_html(
    out: Path,
    selected_laws: pd.DataFrame,
    nested_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    orders: pd.DataFrame,
    compounding: pd.DataFrame,
    source_sha: str,
) -> str:
    exact = selected_laws[
        (selected_laws["kind"] == "binary")
        & (selected_laws["target"] == "exact_correct")
    ].copy()
    exact["mode"] = exact["mode"].map(MODE_LABELS)
    exact["formula"] = [
        candidate_formula(candidate, link)
        for candidate, link in zip(exact["candidate"], exact["link"])
    ]
    exact_table = dataframe_html(
        exact.round(4),
        [
            "mode",
            "candidate",
            "link",
            "formula",
            "seed_log_loss",
            "cell_log_loss",
            "cell_cell_r2",
            "length_cell_r2",
            "needle_cell_r2",
        ],
    )
    nested = nested_metrics.copy()
    nested["mode"] = nested["mode"].map(MODE_LABELS)
    nested_table = dataframe_html(nested.round(4))
    key_targets = diagnostics[
        diagnostics["target"].isin(
            [
                "parse_success",
                "truncated",
                "exact_given_parsed",
                "within_one",
                "pair_retrieval",
                "all_pairs_found",
                "list_length_exact",
                "log1p_absolute_error",
                "signed_relative_error",
                "missing_fraction",
            ]
        )
    ].copy()
    key_targets["mode"] = key_targets["mode"].map(MODE_LABELS)
    key_targets["target"] = key_targets["target"].map(TARGET_LABELS)
    key_table = dataframe_html(
        key_targets.round(4),
        [
            "mode",
            "target",
            "kind",
            "candidate",
            "link",
            "n_rows",
            "cell_log_loss",
            "cell_brier",
            "cell_cell_r2",
            "cell_r2",
            "length_cell_r2",
            "needle_cell_r2",
        ],
    )
    primary_orders = orders[orders["target"] == "exact_correct"].copy()
    primary_orders["mode"] = primary_orders["mode"].map(MODE_LABELS)
    order_table = dataframe_html(
        primary_orders.round(4),
        [
            "mode",
            "model_label",
            "candidate",
            "link",
            "length_order_log",
            "needle_order_log",
            "log_interaction",
            "density_order",
            "burden_order",
        ],
    )
    comp_table = dataframe_html(compounding.round(4))
    timestamp = datetime.now(timezone.utc).isoformat()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Counting-mechanism empirical law</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#596875; --blue:#245b8a; --line:#d9e0e6; --panel:#f7f9fb; }}
body {{ margin:0; font-family: Inter, "Segoe UI", "Microsoft YaHei", sans-serif; color:var(--ink); line-height:1.62; background:white; }}
main {{ max-width:1180px; margin:0 auto; padding:40px 34px 80px; }}
h1 {{ font-size:2.15rem; line-height:1.2; margin:0 0 8px; }}
h2 {{ margin-top:42px; padding-bottom:8px; border-bottom:2px solid var(--line); }}
h3 {{ margin-top:28px; }}
p,li {{ max-width:92ch; }}
.meta {{ color:var(--muted); margin-bottom:28px; }}
.callout {{ background:#edf5fb; border-left:5px solid var(--blue); padding:16px 20px; margin:18px 0; }}
.warning {{ background:#fff6df; border-left-color:#c67a00; }}
.formula {{ font-family:"Cambria Math","STIX Two Math",serif; font-size:1.05rem; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 18px; margin:12px 0; overflow-x:auto; }}
figure {{ margin:28px 0 38px; }}
figure img {{ display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:8px; background:white; }}
figcaption {{ color:var(--muted); font-size:.94rem; margin-top:9px; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin:14px 0 26px; }}
.data-table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
.data-table th {{ position:sticky; top:0; background:#eaf0f5; text-align:left; }}
.data-table th,.data-table td {{ border-bottom:1px solid var(--line); padding:7px 9px; vertical-align:top; }}
code {{ background:#f0f3f5; border-radius:4px; padding:1px 4px; }}
small {{ color:var(--muted); }}
</style>
</head>
<body><main>
<h1>Counting mechanism 的 empirical-law 搜索</h1>
<p class="meta">按 nonthinking、enumeration、CoT 分层；不使用模型大小；生成于 {html.escape(timestamp)}。源请求级表 SHA256：<code>{source_sha}</code>。</p>

<div class="callout">
<strong>如何读这份分析：</strong>每种 prompt mode 使用同一组候选函数族，但每个模型拥有自己的参数。选择依据是按 seed 与完整 (L,N) cell 分组的 held-out 指标；primary exact law 另外使用“外层 cell、内层 seed”的 nested 验证。任何高训练拟合都不会单独构成结论。
</div>

<h2>1. 机制假说与定义</h2>
<p>令 <strong>L</strong> 为目标 haystack token 长度，<strong>N</strong> 为 needle 数。基础坐标为 <code>l=log(L/5000)</code> 与 <code>n=log(N/5)</code>。所有 exact correctness 都把解析失败、未按格式和截断保留为失败；数值误差只在成功解析的输出上定义并明确报告分母。</p>
<div class="formula"><strong>Failure-hazard / survival law</strong><br>
p<sub>m</sub>(L,N,o) = exp{{−exp[a<sub>m</sub> + r<sub>m</sub> log(L/5000) + s<sub>m</sub> log(N/5) + o<sub>m</sub>I(query-last)]}}.
</div>
<div class="formula"><strong>Logistic / Hill law</strong><br>
logit p<sub>m</sub>(L,N,o) = a<sub>m</sub> + r<sub>m</sub> log(L/5000) + s<sub>m</sub> log(N/5) + o<sub>m</sub>I(query-last).
</div>
<div class="formula"><strong>Enumeration retrieval compounding</strong><br>
logit q<sub>m</sub>(L,N,o) = a<sub>m</sub> + r<sub>m</sub>log(L/5000) + s<sub>m</sub>log(N/5) + o<sub>m</sub>I(query-last), &nbsp;
P(all N gold pairs found) ≈ q<sub>m</sub><sup>N</sup>.
</div>
<p>这里的 <em>q</em> 是逐 needle 的 pair-recall 概率：每条枚举输出以 <code>N−missing_pairs</code> 为成功数、<code>N</code> 为试验数。<code>q^N</code> 是“各 needle 检索近似独立”时全部找齐的复合概率，因此它比只回归最终准确率更接近可检验的 counting/retrieval mechanism。</p>

<h2>2. Primary exact-count law</h2>
<div class="table-wrap">{exact_table}</div>
<figure><img src="figures/exact_candidate_cv.png" alt="Exact-count candidate CV">
<figcaption>图 1｜每种 mode 的候选函数族。横轴是 seed OOF 与 blocked-cell OOF log loss 的均值，越低越好。图中仅显示每种 mode 最好的 12 个候选；完整搜索表保存在 CSV。</figcaption></figure>
<figure><img src="figures/nested_exact_cell_calibration.png" alt="Nested exact calibration">
<figcaption>图 2｜严格 nested 验证：外层整块留出 (L,N) cell，内层仅在训练 cell 上用 leave-one-seed-out 选择函数族。每点是一个 model × L × N 单元；横轴为完全 held-out 的预测准确率，纵轴为实际准确率。R² 只描述 cell 均值，不等于 request-level R²。</figcaption></figure>
<div class="table-wrap">{nested_table}</div>

<h2>3. L 与 N 的阶数</h2>
<p>当被选函数族包含 log L 与 log N 时，表中的两个系数就是相应 link 尺度上的局部幂阶。不同模型可以有不同斜率；空值表示该 mode 选中了 density、burden、raw、piecewise 或其他坐标，不能硬解释成独立幂阶。</p>
<figure><img src="figures/exact_selected_orders.png" alt="Selected exact orders">
<figcaption>图 3｜exact law 中 log(L/5000) 与 log(N/5) 的模型参数。颜色与数字均是 link 尺度上的系数；survival-link 中正系数意味着 failure hazard 随变量增加，logistic-link 中负系数意味着准确率下降。</figcaption></figure>
<div class="table-wrap">{order_table}</div>

<h2>4. 机制拆解：哪里容易拟合，哪里不容易</h2>
<figure><img src="figures/mechanism_target_cell_r2.png" alt="Mechanism target cell R2">
<figcaption>图 4｜各机制指标的 blocked-cell OOF R²。行是不同目标，列是 prompt mode；每个值在 model × L × N 的 held-out 单元均值上计算。高值说明 L、N 与 query order 的响应面能跨 cell 复现该指标；负值说明还不如单纯预测总体均值。</figcaption></figure>
<div class="table-wrap">{key_table}</div>

<h2>5. Enumeration：逐 needle 检索能否复合成“全部找齐”</h2>
<figure><img src="figures/enumeration_retrieval_compounding.png" alt="Enumeration compounding">
<figcaption>图 5｜先在 blocked cell 上预测单个 needle 的检索概率 q，再计算 q^N。左图与“全部 gold pairs 找齐”比较，是独立检索假说的直接检验；右图与最终 numeric exact count 比较，检验检索完整性是否足以解释最后的计数答案。两者不应混为一谈。</figcaption></figure>
<div class="table-wrap">{comp_table}</div>

<h2>6. Error 与资源量的响应面</h2>
<figure><img src="figures/continuous_target_r2.png" alt="Continuous target R2">
<figcaption>图 6｜误差和输出 token 等连续目标的 blocked-cell OOF request-level R²。误差只在成功解析的请求上定义；<code>log1p</code> 与 <code>asinh</code> 变换用于保留所有极端值同时降低少数异常大输出的支配。</figcaption></figure>

<h2>7. 结论边界</h2>
<ul>
<li>“统一 law”在这里指<strong>统一的函数族</strong>，不是统一参数；每个模型都允许不同的 a、r、s 和 query-order nuisance 参数。</li>
<li>三种 mode 分开选择，避免把 nonthinking、enumeration 与 CoT 的不同失败机制压成一个斜率。</li>
<li>pair retrieval、parse、truncation 与 numeric count 是统计关联的机制指标，不构成神经网络内部因果机制的直接证明。</li>
<li>实验只覆盖 L∈{{2000,5000,10000}}、N∈{{1,2,3,4,5,6,8,10,20,30}}；任何区间外外推都需要新实验验证。</li>
</ul>
<p><small>完整候选、fold 指标、参数、OOF 预测、分析计划、manifest 与复现脚本均在本目录中。</small></p>
</main></body></html>"""


def inject_into_main_report(main_report: Path, section_html: str) -> None:
    start = "<!-- COUNTING_MECHANISM_LAW_V1_START -->"
    end = "<!-- COUNTING_MECHANISM_LAW_V1_END -->"
    source = main_report.read_text(encoding="utf-8")
    body_start = section_html.index("<main>") + len("<main>")
    body_end = section_html.rindex("</main>")
    body = section_html[body_start:body_end]
    # Fix paths because the canonical report sits two directories above this addendum.
    body = body.replace('src="figures/', 'src="analysis/counting_mechanism_law_v1/figures/')
    fragment = (
        f"\n{start}\n<section id=\"counting-mechanism-law-v1\">\n"
        f"{body}\n</section>\n{end}\n"
    )
    if start in source and end in source:
        left = source.split(start, 1)[0]
        right = source.split(end, 1)[1]
        updated = left + fragment + right
    else:
        marker = "</main>"
        if marker not in source:
            marker = "</body>"
        if marker not in source:
            raise RuntimeError("Cannot find insertion marker in canonical report")
        updated = source.replace(marker, fragment + marker, 1)
    main_report.write_text(updated, encoding="utf-8")


def write_manifest(
    out: Path,
    source: Path,
    selected_laws: pd.DataFrame,
    files: list[Path],
) -> None:
    manifest = {
        "analysis": "counting_mechanism_law_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "rows": 6300,
        },
        "invariants": {
            "modes_analyzed_separately": list(MODE_LABELS),
            "model_size_predictor": False,
            "failures_remain_exact_failures": True,
            "point_deletion": False,
            "primary_selection": "nested outer blocked-cell / inner leave-one-seed-out",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "numerical_settings": {
            "irls_max_iterations": 30,
            "irls_coefficient_tolerance": 1e-6,
            "ridge_penalty": RIDGE,
            "recommended_blas_threads": 1,
            "survival_link_optimizer": "L-BFGS-B with analytic gradient",
            "survival_link_max_iterations": 250,
        },
        "candidate_grid": [
            {
                "name": candidate.name,
                "features": list(candidate.features),
                "complexity": candidate.complexity,
                "description": candidate.description,
            }
            for candidate in CANDIDATES
        ],
        "selected_laws": selected_laws.to_dict(orient="records"),
        "files": [
            {
                "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
            if path.is_file()
        ],
    }
    (out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-main-report", action="store_true")
    args = parser.parse_args()

    report_root = args.report_root.resolve()
    out = args.output.resolve()
    tables = out / "tables"
    figures = out / "figures"
    scripts = out / "scripts"
    logs = out / "logs"
    for directory in (out, tables, figures, scripts, logs):
        directory.mkdir(parents=True, exist_ok=True)
    source = report_root / "tables" / "request_level_report.csv"
    frame = enrich(pd.read_csv(source))
    if len(frame) != 6300 or frame["request_id"].duplicated().any():
        raise RuntimeError("Source integrity check failed")

    binary_comparison_rows: list[dict[str, Any]] = []
    binary_fold_rows: list[dict[str, Any]] = []
    continuous_comparison_rows: list[dict[str, Any]] = []
    continuous_fold_rows: list[dict[str, Any]] = []
    selected_law_rows: list[dict[str, Any]] = []
    coefficient_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    selected_oofs: dict[tuple[str, str, str], np.ndarray] = {}
    target_registry: dict[tuple[str, str], TargetFrame] = {}
    nested_selection_frames: list[pd.DataFrame] = []
    nested_prediction_frames: list[pd.DataFrame] = []
    nested_metric_rows: list[dict[str, Any]] = []

    log_lines = [
        f"start={datetime.now(timezone.utc).isoformat()}",
        f"source={source}",
        f"source_sha256={sha256(source)}",
    ]

    for mode in MODE_LABELS:
        mode_frame = frame[frame["prompt_mode"] == mode].reset_index(drop=True)
        binary_targets = make_binary_targets(mode_frame, mode)
        continuous_targets = make_continuous_targets(mode_frame, mode)
        for name, target in binary_targets.items():
            target_registry[(mode, name)] = target
            comparison: list[dict[str, Any]] = []
            folds: list[dict[str, Any]] = []
            for candidate in CANDIDATES:
                for link in ("logistic", "survival_loglog"):
                    row, fold_rows, _ = evaluate_binary_candidate(
                        target, candidate, link
                    )
                    row.update({"mode": mode, "target": name, "kind": "binary"})
                    for fold in fold_rows:
                        fold.update({"mode": mode, "target": name, "kind": "binary"})
                    comparison.append(row)
                    folds.extend(fold_rows)
            comparison_frame = pd.DataFrame(comparison)
            chosen = selected_by_one_se(comparison_frame, "selection_score")
            coefficients, diagnostics, oofs = fit_selected_binary(
                mode, target, chosen
            )
            coefficients["kind"] = "binary"
            coefficient_frames.append(coefficients)
            diagnostics["kind"] = "binary"
            diagnostic_rows.append(diagnostics)
            for scheme, values in oofs.items():
                selected_oofs[(mode, name, scheme)] = values
            selected_law_rows.append(
                {
                    "mode": mode,
                    "target": name,
                    "kind": "binary",
                    "candidate": chosen["candidate"],
                    "link": chosen["link"],
                    "formula": candidate_formula(
                        str(chosen["candidate"]), str(chosen["link"])
                    ),
                    "n_rows": len(target.frame),
                    "n_trials": float(target.trials.sum()),
                    **{
                        key: diagnostics[key]
                        for key in diagnostics
                        if key
                        not in {
                            "mode",
                            "target",
                            "candidate",
                            "link",
                            "n_rows",
                            "n_trials",
                            "kind",
                        }
                    },
                }
            )
            binary_comparison_rows.extend(comparison)
            binary_fold_rows.extend(folds)
            log_lines.append(
                f"selected binary {mode}/{name}: {chosen['candidate']}/{chosen['link']}"
            )
        for name, target in continuous_targets.items():
            target_registry[(mode, name)] = target
            comparison = []
            folds = []
            for candidate in CANDIDATES:
                row, fold_rows, _ = evaluate_continuous_candidate(target, candidate)
                row.update({"mode": mode, "target": name, "kind": "continuous"})
                for fold in fold_rows:
                    fold.update(
                        {"mode": mode, "target": name, "kind": "continuous"}
                    )
                comparison.append(row)
                folds.extend(fold_rows)
            comparison_frame = pd.DataFrame(comparison)
            chosen = selected_by_one_se(comparison_frame, "selection_score")
            coefficients, diagnostics, oofs = fit_selected_continuous(
                mode, target, chosen
            )
            coefficients["kind"] = "continuous"
            coefficient_frames.append(coefficients)
            diagnostics["kind"] = "continuous"
            diagnostic_rows.append(diagnostics)
            for scheme, values in oofs.items():
                selected_oofs[(mode, name, scheme)] = values
            selected_law_rows.append(
                {
                    "mode": mode,
                    "target": name,
                    "kind": "continuous",
                    "candidate": chosen["candidate"],
                    "link": "identity",
                    "formula": candidate_formula(
                        str(chosen["candidate"]), "identity"
                    ),
                    "n_rows": len(target.frame),
                    "n_trials": math.nan,
                    **{
                        key: diagnostics[key]
                        for key in diagnostics
                        if key
                        not in {
                            "mode",
                            "target",
                            "candidate",
                            "link",
                            "n_rows",
                            "kind",
                        }
                    },
                }
            )
            continuous_comparison_rows.extend(comparison)
            continuous_fold_rows.extend(folds)
            log_lines.append(
                f"selected continuous {mode}/{name}: {chosen['candidate']}"
            )

        nested_selection, nested_predictions, nested_metrics = nested_primary_exact(
            mode, binary_targets["exact_correct"]
        )
        nested_selection_frames.append(nested_selection)
        nested_prediction_frames.append(nested_predictions)
        nested_metric_rows.append({"mode": mode, **nested_metrics})

    binary_comparison = pd.DataFrame(binary_comparison_rows)
    binary_folds = pd.DataFrame(binary_fold_rows)
    continuous_comparison = pd.DataFrame(continuous_comparison_rows)
    continuous_folds = pd.DataFrame(continuous_fold_rows)
    selected_laws = pd.DataFrame(selected_law_rows)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    orders = extract_model_orders(coefficients)
    nested_selection = pd.concat(nested_selection_frames, ignore_index=True)
    primary_predictions = pd.concat(nested_prediction_frames, ignore_index=True)
    nested_metrics = pd.DataFrame(nested_metric_rows)

    enum_targets = {
        name: target
        for (mode, name), target in target_registry.items()
        if mode == "enumeration"
    }
    compounding, compounding_cells = enumeration_compounding(
        enum_targets, selected_oofs
    )

    binary_comparison.to_csv(tables / "binary_candidate_comparison.csv", index=False)
    binary_folds.to_csv(tables / "binary_fold_metrics.csv", index=False)
    continuous_comparison.to_csv(
        tables / "continuous_candidate_comparison.csv", index=False
    )
    continuous_folds.to_csv(tables / "continuous_fold_metrics.csv", index=False)
    selected_laws.to_csv(tables / "selected_laws.csv", index=False)
    coefficients.to_csv(tables / "selected_coefficients.csv", index=False)
    diagnostics.to_csv(tables / "selected_law_diagnostics.csv", index=False)
    orders.to_csv(tables / "model_parameter_orders.csv", index=False)
    nested_selection.to_csv(tables / "nested_primary_selection.csv", index=False)
    primary_predictions.to_csv(
        tables / "nested_primary_oof_predictions.csv", index=False
    )
    nested_metrics.to_csv(tables / "nested_primary_metrics.csv", index=False)
    compounding.to_csv(tables / "enumeration_compounding_metrics.csv", index=False)
    compounding_cells.to_csv(
        tables / "enumeration_compounding_oof.csv", index=False
    )

    primary_oof_rows: list[pd.DataFrame] = []
    for mode in MODE_LABELS:
        target = target_registry[(mode, "exact_correct")]
        temp = target.frame[
            [
                "request_id",
                "stimulus_id",
                "model_label",
                "seed",
                "target_passage_tokens",
                "num_needles",
                "query_order",
                "exact_correct",
            ]
        ].copy()
        temp["mode"] = mode
        temp["selected_seed_oof"] = selected_oofs[(mode, "exact_correct", "seed")]
        temp["selected_cell_oof"] = selected_oofs[(mode, "exact_correct", "cell")]
        primary_oof_rows.append(temp)
    pd.concat(primary_oof_rows, ignore_index=True).to_csv(
        tables / "selected_exact_oof_predictions.csv", index=False
    )

    save_exact_candidate_figure(
        binary_comparison, figures / "exact_candidate_cv.png"
    )
    save_oof_calibration_figure(
        primary_predictions, figures / "nested_exact_cell_calibration.png"
    )
    save_order_heatmap(orders, figures / "exact_selected_orders.png")
    save_mechanism_performance_figure(
        diagnostics, figures / "mechanism_target_cell_r2.png"
    )
    save_enumeration_compounding_figure(
        compounding_cells, figures / "enumeration_retrieval_compounding.png"
    )
    save_continuous_target_figure(
        diagnostics, figures / "continuous_target_r2.png"
    )

    report_html = build_html(
        out,
        selected_laws,
        nested_metrics,
        diagnostics,
        orders,
        compounding,
        sha256(source),
    )
    (out / "report.html").write_text(report_html, encoding="utf-8")
    if args.inject_main_report:
        inject_into_main_report(report_root / "report.html", report_html)

    readme = f"""# Counting-mechanism empirical law v1

This directory contains the bounded, grouped-validation search requested for
nonthinking/direct, enumeration, and native-thinking/CoT modes.

- Standalone report: `report.html`
- Frozen search plan: `analysis_plan.md`
- Reproduction script: `scripts/run_counting_mechanism_law.py`
- Complete candidate and fold results: `tables/`
- Figures: `figures/`

Source SHA256: `{sha256(source)}`

Run:

```powershell
& "<python-with-numpy-pandas-scipy-matplotlib>" `
  "scripts/run_counting_mechanism_law.py" `
  --report-root "{report_root}" `
  --output "{out}" `
  --inject-main-report
```
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": sha256(source),
                "selected_primary_laws": selected_laws[
                    selected_laws["target"] == "exact_correct"
                ].to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (logs / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    generated = [
        path
        for path in out.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    ]
    write_manifest(out, source, selected_laws, generated)
    generated = [path for path in out.rglob("*") if path.is_file()]
    checksum_rows = [
        f"{sha256(path)}\t{path.relative_to(out)}"
        for path in sorted(generated)
        if path.name != "SHA256SUMS.tsv"
    ]
    (out / "SHA256SUMS.tsv").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(out),
                "files": len([p for p in out.rglob('*') if p.is_file()]),
                "selected_primary": selected_laws[
                    selected_laws["target"] == "exact_correct"
                ][["mode", "candidate", "link", "cell_log_loss", "cell_cell_r2"]]
                .round(6)
                .to_dict(orient="records"),
                "nested_primary": nested_metrics.round(6).to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
