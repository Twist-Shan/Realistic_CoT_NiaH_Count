from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
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

import run_counting_mechanism_law as base


EPS = 1e-9
RIDGE = 1e-5


@dataclass(frozen=True)
class CompoundingCandidate:
    name: str
    complexity: int
    description: str


COMPOUNDING_CANDIDATES = [
    CompoundingCandidate("independence", 0, "N_eff=N"),
    CompoundingCandidate("model_scale", 1, "N_eff=kappa_m N"),
    CompoundingCandidate("shared_scale_order", 2, "N_eff=kappa N^tau"),
    CompoundingCandidate(
        "model_scale_shared_order", 2, "N_eff=kappa_m N^tau"
    ),
    CompoundingCandidate(
        "model_scale_order", 3, "N_eff=kappa_m N^tau_m"
    ),
    CompoundingCandidate(
        "model_scale_order_query",
        4,
        "N_eff=kappa_m N^tau_m exp(o_m I(query-last))",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_from_selected(
    selected: pd.DataFrame, mode: str, target: str
) -> tuple[base.Candidate, str]:
    row = selected[
        (selected["mode"] == mode) & (selected["target"] == target)
    ].iloc[0]
    return base.selected_candidate(str(row["candidate"])), str(row["link"])


def build_compounding_design(
    frame: pd.DataFrame,
    candidate: CompoundingCandidate,
    models: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    log_n = np.log(frame["N"].to_numpy(float))
    query_last = frame["query_last"].to_numpy(float)
    columns: list[np.ndarray] = []
    names: list[str] = []
    if candidate.name == "independence":
        return np.zeros((len(frame), 0)), [], log_n
    if candidate.name == "shared_scale_order":
        columns = [np.ones(len(frame)), log_n]
        names = ["shared::log_kappa", "shared::tau"]
        return np.column_stack(columns), names, np.zeros(len(frame))
    for model in models:
        indicator = (frame["model_label"].to_numpy() == model).astype(float)
        columns.append(indicator)
        names.append(f"{model}::log_kappa")
        if candidate.name in {
            "model_scale_order",
            "model_scale_order_query",
        }:
            columns.append(indicator * log_n)
            names.append(f"{model}::tau")
        if candidate.name == "model_scale_order_query":
            columns.append(indicator * query_last)
            names.append(f"{model}::query_last")
    if candidate.name == "model_scale_shared_order":
        columns.append(log_n)
        names.append("shared::tau")
    offset = log_n if candidate.name == "model_scale" else np.zeros(len(frame))
    return np.column_stack(columns), names, offset


def predict_compounding(
    q: np.ndarray,
    matrix: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[1] == 0:
        log_neff = offset
    else:
        log_neff = np.clip(offset + matrix @ beta, -12.0, 12.0)
    neff = np.exp(log_neff)
    log_p = neff * np.log(np.clip(q, EPS, 1.0 - EPS))
    p = np.exp(np.clip(log_p, -30.0, -EPS))
    return np.clip(p, EPS, 1.0 - EPS), neff


def fit_compounding(
    frame: pd.DataFrame,
    q: np.ndarray,
    y: np.ndarray,
    candidate: CompoundingCandidate,
    models: list[str],
) -> dict[str, Any]:
    matrix, names, offset = build_compounding_design(frame, candidate, models)
    if matrix.shape[1] == 0:
        prediction, neff = predict_compounding(
            q, matrix, np.zeros(0), np.log(frame["N"].to_numpy(float))
        )
        return {
            "beta": np.zeros(0),
            "names": names,
            "success": True,
            "prediction": prediction,
            "neff": neff,
        }

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        raw_log_neff = offset + matrix @ beta
        log_neff = np.clip(raw_log_neff, -12.0, 12.0)
        neff = np.exp(log_neff)
        log_q = np.log(np.clip(q, EPS, 1.0 - EPS))
        log_p = neff * log_q
        p = np.clip(np.exp(np.clip(log_p, -30.0, -EPS)), EPS, 1.0 - EPS)
        value = float(
            -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            + 0.5 * RIDGE * np.dot(beta, beta)
        )
        eta_gradient = (p - y) * log_p / (1.0 - p)
        eta_gradient *= (
            (raw_log_neff > -12.0) & (raw_log_neff < 12.0)
        ).astype(float)
        gradient = matrix.T @ eta_gradient + RIDGE * beta
        return value, gradient

    result = scipy.optimize.minimize(
        objective,
        np.zeros(matrix.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-6},
    )
    prediction, neff = predict_compounding(q, matrix, result.x, offset)
    return {
        "beta": np.asarray(result.x, float),
        "names": names,
        "success": bool(result.success),
        "prediction": prediction,
        "neff": neff,
        "iterations": int(result.nit),
    }


def crossfit_retrieval_q(
    frame: pd.DataFrame,
    successes: np.ndarray,
    trials: np.ndarray,
    candidate: base.Candidate,
    link: str,
    eligible: np.ndarray,
) -> np.ndarray:
    matrix, _, _ = base.build_design(frame, candidate)
    q = np.full(len(frame), np.nan)
    for seed in sorted(frame["seed"].unique()):
        test = eligible & (frame["seed"].to_numpy() == seed)
        train = eligible & ~test
        fit = base.fit_binomial(
            matrix[train], successes[train], trials[train], link
        )
        q[test] = base.predict_binomial(matrix[test], fit["beta"], link)
    if np.isnan(q[eligible]).any():
        raise RuntimeError("Missing cross-fitted q")
    return q


def effective_retrieval_search(
    enum: pd.DataFrame,
    retrieval_target: base.TargetFrame,
    all_found_target: base.TargetFrame,
    retrieval_candidate: base.Candidate,
    retrieval_link: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    models = base.mode_models(enum)
    matrix_q, _, _ = base.build_design(enum, retrieval_candidate)
    outer_folds = base.fold_definitions(enum, "cell")
    candidate_oofs = {
        candidate.name: np.full(len(enum), np.nan)
        for candidate in COMPOUNDING_CANDIDATES
    }
    candidate_neff = {
        candidate.name: np.full(len(enum), np.nan)
        for candidate in COMPOUNDING_CANDIDATES
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_label, test in outer_folds:
        train = ~test
        q_train = crossfit_retrieval_q(
            enum,
            retrieval_target.successes,
            retrieval_target.trials,
            retrieval_candidate,
            retrieval_link,
            train,
        )
        fit_q = base.fit_binomial(
            matrix_q[train],
            retrieval_target.successes[train],
            retrieval_target.trials[train],
            retrieval_link,
        )
        q_test = base.predict_binomial(
            matrix_q[test], fit_q["beta"], retrieval_link
        )
        for candidate in COMPOUNDING_CANDIDATES:
            fit = fit_compounding(
                enum.loc[train].reset_index(drop=True),
                q_train[train],
                all_found_target.successes[train],
                candidate,
                models,
            )
            test_matrix, _, test_offset = build_compounding_design(
                enum.loc[test].reset_index(drop=True), candidate, models
            )
            if candidate.name == "independence":
                test_offset = np.log(enum.loc[test, "N"].to_numpy(float))
            prediction, neff = predict_compounding(
                q_test, test_matrix, fit["beta"], test_offset
            )
            candidate_oofs[candidate.name][test] = prediction
            candidate_neff[candidate.name][test] = neff
            metrics = base.binomial_metrics(
                all_found_target.successes[test],
                all_found_target.trials[test],
                prediction,
            )
            fold_rows.append(
                {
                    "candidate": candidate.name,
                    "fold": fold_label,
                    "n_test": int(test.sum()),
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                    "converged": bool(fit["success"]),
                }
            )
    comparison_rows: list[dict[str, Any]] = []
    for candidate in COMPOUNDING_CANDIDATES:
        prediction = candidate_oofs[candidate.name]
        metrics = base.binomial_metrics(
            all_found_target.successes, all_found_target.trials, prediction
        )
        cell = base.cell_level_binary_metrics(
            enum,
            all_found_target.successes,
            all_found_target.trials,
            prediction,
        )
        fold_losses = [
            row["log_loss"]
            for row in fold_rows
            if row["candidate"] == candidate.name
        ]
        comparison_rows.append(
            {
                "candidate": candidate.name,
                "link": "compounding",
                "complexity": candidate.complexity,
                "description": candidate.description,
                "selection_score": metrics["log_loss"],
                "selection_se": float(
                    np.std(fold_losses, ddof=1) / math.sqrt(len(fold_losses))
                ),
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                **cell,
                "converged_all_folds": all(
                    row["converged"]
                    for row in fold_rows
                    if row["candidate"] == candidate.name
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    chosen = base.selected_by_one_se(comparison, "selection_score")
    chosen_name = str(chosen["candidate"])
    oof = enum[
        [
            "request_id",
            "stimulus_id",
            "model_label",
            "seed",
            "target_passage_tokens",
            "num_needles",
            "query_order",
        ]
    ].copy()
    oof["observed_all_pairs_found"] = all_found_target.successes
    oof["predicted_probability"] = candidate_oofs[chosen_name]
    oof["predicted_neff"] = candidate_neff[chosen_name]
    oof["selected_candidate"] = chosen_name

    all_rows = np.ones(len(enum), dtype=bool)
    q_seed = crossfit_retrieval_q(
        enum,
        retrieval_target.successes,
        retrieval_target.trials,
        retrieval_candidate,
        retrieval_link,
        all_rows,
    )
    selected_candidate = next(
        candidate
        for candidate in COMPOUNDING_CANDIDATES
        if candidate.name == chosen_name
    )
    final_fit = fit_compounding(
        enum,
        q_seed,
        all_found_target.successes,
        selected_candidate,
        models,
    )
    parameter_rows: list[dict[str, Any]] = []
    values = dict(zip(final_fit["names"], final_fit["beta"]))
    for model in models:
        if chosen_name == "independence":
            log_kappa, tau, order_effect = 0.0, 1.0, 0.0
        elif chosen_name == "shared_scale_order":
            log_kappa = values["shared::log_kappa"]
            tau = values["shared::tau"]
            order_effect = 0.0
        else:
            log_kappa = values.get(f"{model}::log_kappa", 0.0)
            if chosen_name == "model_scale":
                tau = 1.0
            elif chosen_name == "model_scale_shared_order":
                tau = values["shared::tau"]
            else:
                tau = values.get(f"{model}::tau", 1.0)
            order_effect = values.get(f"{model}::query_last", 0.0)
        kappa = math.exp(float(log_kappa))
        parameter_rows.append(
            {
                "model_label": model,
                "selected_candidate": chosen_name,
                "kappa": kappa,
                "tau": float(tau),
                "query_last_log_neff_effect": float(order_effect),
                "neff_at_N1_query_first": kappa,
                "neff_at_N5_query_first": kappa * 5.0 ** float(tau),
                "neff_at_N30_query_first": kappa * 30.0 ** float(tau),
                "neff_over_N_at_N30": kappa * 30.0 ** (float(tau) - 1.0),
            }
        )
    return (
        comparison,
        pd.DataFrame(fold_rows),
        oof,
        pd.DataFrame(parameter_rows),
        q_seed,
    )


def extract_compounding_parameters(
    fit: dict[str, Any],
    candidate: CompoundingCandidate,
    models: list[str],
) -> list[dict[str, float | str]]:
    values = dict(zip(fit["names"], fit["beta"]))
    rows: list[dict[str, float | str]] = []
    for model in models:
        if candidate.name == "independence":
            log_kappa, tau, order_effect = 0.0, 1.0, 0.0
        elif candidate.name == "shared_scale_order":
            log_kappa = float(values["shared::log_kappa"])
            tau = float(values["shared::tau"])
            order_effect = 0.0
        else:
            log_kappa = float(values.get(f"{model}::log_kappa", 0.0))
            if candidate.name == "model_scale":
                tau = 1.0
            elif candidate.name == "model_scale_shared_order":
                tau = float(values["shared::tau"])
            else:
                tau = float(values.get(f"{model}::tau", 1.0))
            order_effect = float(values.get(f"{model}::query_last", 0.0))
        rows.append(
            {
                "model_label": model,
                "log_kappa": log_kappa,
                "kappa": math.exp(log_kappa),
                "tau": tau,
                "query_last_log_neff_effect": order_effect,
            }
        )
    return rows


def compounding_parameter_stability(
    enum: pd.DataFrame,
    q_seed: np.ndarray,
    all_found: np.ndarray,
    selected_name: str,
    bootstrap_draws: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate = next(
        item for item in COMPOUNDING_CANDIDATES if item.name == selected_name
    )
    models = base.mode_models(enum)
    seeds = np.array(sorted(enum["seed"].unique()))
    rng = np.random.default_rng(20260725)
    draw_rows: list[dict[str, Any]] = []
    for draw in range(bootstrap_draws):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(enum["seed"].to_numpy() == seed) for seed in sampled]
        )
        fit = fit_compounding(
            enum.iloc[indices].reset_index(drop=True),
            q_seed[indices],
            all_found[indices],
            candidate,
            models,
        )
        for row in extract_compounding_parameters(fit, candidate, models):
            draw_rows.append(
                {
                    "draw": draw,
                    "sampled_seeds": ",".join(map(str, sampled.tolist())),
                    "converged": bool(fit["success"]),
                    **row,
                }
            )
    draws = pd.DataFrame(draw_rows)
    summary = (
        draws[draws["converged"]]
        .groupby("model_label", as_index=False)
        .agg(
            bootstrap_draws=("draw", "nunique"),
            kappa_median=("kappa", "median"),
            kappa_ci95_low=("kappa", lambda x: float(np.quantile(x, 0.025))),
            kappa_ci95_high=("kappa", lambda x: float(np.quantile(x, 0.975))),
            tau_median=("tau", "median"),
            tau_ci95_low=("tau", lambda x: float(np.quantile(x, 0.025))),
            tau_ci95_high=("tau", lambda x: float(np.quantile(x, 0.975))),
        )
    )
    leave_rows: list[dict[str, Any]] = []
    for scheme in ("seed", "length", "needle"):
        for fold_label, test in base.fold_definitions(enum, scheme):
            train = ~test
            fit = fit_compounding(
                enum.loc[train].reset_index(drop=True),
                q_seed[train],
                all_found[train],
                candidate,
                models,
            )
            for row in extract_compounding_parameters(fit, candidate, models):
                leave_rows.append(
                    {
                        "scheme": scheme,
                        "held_out": fold_label,
                        "converged": bool(fit["success"]),
                        **row,
                    }
                )
    return draws, summary, pd.DataFrame(leave_rows)


def fit_binary_component_to_full_test(
    full: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    fit_mask: np.ndarray,
    y: np.ndarray,
    candidate: base.Candidate,
    link: str,
) -> np.ndarray:
    matrix, _, _ = base.build_design(full, candidate)
    actual_train = train & fit_mask
    fit = base.fit_binomial(
        matrix[actual_train],
        y[actual_train],
        np.ones(actual_train.sum()),
        link,
    )
    return base.predict_binomial(matrix[test], fit["beta"], link)


def exact_hurdle_decomposition(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    nested_primary = pd.read_csv(
        selected.attrs["out"] / "tables" / "nested_primary_oof_predictions.csv"
    )
    for mode in base.MODE_LABELS:
        full = frame[frame["prompt_mode"] == mode].reset_index(drop=True)
        parse_candidate, parse_link = candidate_from_selected(
            selected, mode, "parse_success"
        )
        conditional_candidate, conditional_link = candidate_from_selected(
            selected, mode, "exact_given_parsed"
        )
        parse_y = full["parse_success"].to_numpy(float)
        exact_y = full["exact_correct"].to_numpy(float)
        parsed = parse_y == 1
        oof = np.full(len(full), np.nan)
        p_parse_oof = np.full(len(full), np.nan)
        p_cond_oof = np.full(len(full), np.nan)
        for _, test in base.fold_definitions(full, "cell"):
            train = ~test
            p_parse = fit_binary_component_to_full_test(
                full,
                train,
                test,
                np.ones(len(full), dtype=bool),
                parse_y,
                parse_candidate,
                parse_link,
            )
            p_conditional = fit_binary_component_to_full_test(
                full,
                train,
                test,
                parsed,
                exact_y,
                conditional_candidate,
                conditional_link,
            )
            p_parse_oof[test] = p_parse
            p_cond_oof[test] = p_conditional
            oof[test] = p_parse * p_conditional
        metrics = base.binomial_metrics(exact_y, np.ones(len(full)), oof)
        cell = base.cell_level_binary_metrics(
            full, exact_y, np.ones(len(full)), oof
        )
        direct_part = nested_primary[nested_primary["mode"] == mode]
        direct_metrics = base.binomial_metrics(
            direct_part["exact_correct"].to_numpy(float),
            np.ones(len(direct_part)),
            direct_part["nested_cell_oof_probability"].to_numpy(float),
        )
        direct_cell = base.cell_level_binary_metrics(
            direct_part,
            direct_part["exact_correct"].to_numpy(float),
            np.ones(len(direct_part)),
            direct_part["nested_cell_oof_probability"].to_numpy(float),
        )
        metric_rows.extend(
            [
                {
                    "mode": mode,
                    "method": "parse_times_conditional_exact",
                    **metrics,
                    **cell,
                },
                {
                    "mode": mode,
                    "method": "stage1_nested_direct_surface",
                    **direct_metrics,
                    **direct_cell,
                },
            ]
        )
        part = full[
            [
                "request_id",
                "model_label",
                "seed",
                "target_passage_tokens",
                "num_needles",
                "query_order",
                "exact_correct",
            ]
        ].copy()
        part["mode"] = mode
        part["p_parse_oof"] = p_parse_oof
        part["p_exact_given_parse_oof"] = p_cond_oof
        part["p_exact_product_oof"] = oof
        prediction_rows.append(part)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def conditional_magnitude_search(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], base.Candidate],
]:
    comparison_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[pd.DataFrame] = []
    selected_candidates: dict[tuple[str, str], base.Candidate] = {}
    for mode in base.MODE_LABELS:
        parsed = frame[
            (frame["prompt_mode"] == mode) & (frame["parse_success"] == 1)
        ].reset_index(drop=True)
        signed = parsed["signed_error"].to_numpy(float)
        rel_mag = np.abs(signed) / parsed["N"].to_numpy(float)
        for direction, mask in (
            ("under", signed < 0),
            ("over", signed > 0),
        ):
            target_frame = base.TargetFrame(
                name=f"{direction}_log1p_relative_magnitude",
                frame=parsed.loc[mask].reset_index(drop=True),
                values=np.log1p(rel_mag[mask]),
                definition=f"log(1+|error|/N) conditional on {direction}count",
            )
            rows: list[dict[str, Any]] = []
            folds: list[dict[str, Any]] = []
            for candidate in base.CANDIDATES:
                row, fold, _ = base.evaluate_continuous_candidate(
                    target_frame, candidate
                )
                row.update(
                    {
                        "mode": mode,
                        "target": target_frame.name,
                        "direction": direction,
                    }
                )
                for item in fold:
                    item.update(
                        {
                            "mode": mode,
                            "target": target_frame.name,
                            "direction": direction,
                        }
                    )
                rows.append(row)
                folds.extend(fold)
            comparison = pd.DataFrame(rows)
            chosen = base.selected_by_one_se(comparison, "selection_score")
            candidate = base.selected_candidate(str(chosen["candidate"]))
            for row in rows:
                row["selected_one_se"] = bool(
                    row["candidate"] == chosen["candidate"]
                )
            selected_candidates[(mode, direction)] = candidate
            matrix, names, _ = base.build_design(target_frame.frame, candidate)
            beta = base.fit_ols(matrix, target_frame.values)
            coefficient_rows.append(
                pd.DataFrame(
                    {
                        "mode": mode,
                        "direction": direction,
                        "target": target_frame.name,
                        "candidate": candidate.name,
                        "term": names,
                        "estimate": beta,
                    }
                )
            )
            comparison_rows.extend(rows)
            fold_rows.extend(folds)
    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(fold_rows),
        pd.concat(coefficient_rows, ignore_index=True),
        selected_candidates,
    )


def smearing_by_model(
    frame: pd.DataFrame,
    residual: np.ndarray,
) -> dict[str, float]:
    temp = pd.DataFrame(
        {
            "model_label": frame["model_label"].to_numpy(),
            "factor": np.exp(np.clip(residual, -10, 10)),
        }
    )
    result = temp.groupby("model_label")["factor"].mean().to_dict()
    overall = float(temp["factor"].mean())
    result["_fallback"] = overall
    return result


def hurdle_oof_for_mode(
    parsed: pd.DataFrame,
    selected: pd.DataFrame,
    mode: str,
    magnitude_candidates: dict[tuple[str, str], base.Candidate],
    baseline: bool = False,
) -> pd.DataFrame:
    signed = parsed["signed_error"].to_numpy(float)
    rel = signed / parsed["N"].to_numpy(float)
    abs_rel = np.abs(rel)
    under_y = (signed < 0).astype(float)
    over_y = (signed > 0).astype(float)
    if baseline:
        under_candidate = over_candidate = base.selected_candidate("intercept_only")
        under_link = over_link = "logistic"
        mag_under_candidate = mag_over_candidate = base.selected_candidate(
            "intercept_only"
        )
    else:
        under_candidate, under_link = candidate_from_selected(
            selected, mode, "undercount"
        )
        over_candidate, over_link = candidate_from_selected(
            selected, mode, "overcount"
        )
        mag_under_candidate = magnitude_candidates[(mode, "under")]
        mag_over_candidate = magnitude_candidates[(mode, "over")]

    x_under, _, _ = base.build_design(parsed, under_candidate)
    x_over, _, _ = base.build_design(parsed, over_candidate)
    x_mag_under, _, _ = base.build_design(parsed, mag_under_candidate)
    x_mag_over, _, _ = base.build_design(parsed, mag_over_candidate)
    predicted_bias = np.full(len(parsed), np.nan)
    predicted_abs = np.full(len(parsed), np.nan)
    p_under_oof = np.full(len(parsed), np.nan)
    p_over_oof = np.full(len(parsed), np.nan)
    mag_under_oof = np.full(len(parsed), np.nan)
    mag_over_oof = np.full(len(parsed), np.nan)
    for _, test in base.fold_definitions(parsed, "cell"):
        train = ~test
        fit_under = base.fit_binomial(
            x_under[train], under_y[train], np.ones(train.sum()), under_link
        )
        fit_over = base.fit_binomial(
            x_over[train], over_y[train], np.ones(train.sum()), over_link
        )
        p_under = base.predict_binomial(
            x_under[test], fit_under["beta"], under_link
        )
        p_over = base.predict_binomial(x_over[test], fit_over["beta"], over_link)

        train_under = train & (under_y == 1)
        train_over = train & (over_y == 1)
        y_mag_under = np.log1p(abs_rel[train_under])
        y_mag_over = np.log1p(abs_rel[train_over])
        beta_mag_under = base.fit_ols(
            x_mag_under[train_under], y_mag_under
        )
        beta_mag_over = base.fit_ols(x_mag_over[train_over], y_mag_over)
        fitted_under = x_mag_under[train_under] @ beta_mag_under
        fitted_over = x_mag_over[train_over] @ beta_mag_over
        smear_under = smearing_by_model(
            parsed.loc[train_under].reset_index(drop=True),
            y_mag_under - fitted_under,
        )
        smear_over = smearing_by_model(
            parsed.loc[train_over].reset_index(drop=True),
            y_mag_over - fitted_over,
        )
        log_pred_under = x_mag_under[test] @ beta_mag_under
        log_pred_over = x_mag_over[test] @ beta_mag_over
        test_models = parsed.loc[test, "model_label"].tolist()
        mag_under = np.array(
            [
                max(
                    math.exp(value)
                    * smear_under.get(model, smear_under["_fallback"])
                    - 1.0,
                    0.0,
                )
                for value, model in zip(log_pred_under, test_models)
            ]
        )
        mag_over = np.array(
            [
                max(
                    math.exp(value)
                    * smear_over.get(model, smear_over["_fallback"])
                    - 1.0,
                    0.0,
                )
                for value, model in zip(log_pred_over, test_models)
            ]
        )
        p_under_oof[test] = p_under
        p_over_oof[test] = p_over
        mag_under_oof[test] = mag_under
        mag_over_oof[test] = mag_over
        predicted_bias[test] = p_over * mag_over - p_under * mag_under
        predicted_abs[test] = p_over * mag_over + p_under * mag_under
    result = parsed[
        [
            "request_id",
            "model_label",
            "seed",
            "target_passage_tokens",
            "num_needles",
            "query_order",
        ]
    ].copy()
    result["mode"] = mode
    result["method"] = "model_only_hurdle" if baseline else "selected_hurdle"
    result["observed_signed_relative_error"] = rel
    result["observed_relative_absolute_error"] = abs_rel
    result["p_under_oof"] = p_under_oof
    result["p_over_oof"] = p_over_oof
    result["under_magnitude_oof"] = mag_under_oof
    result["over_magnitude_oof"] = mag_over_oof
    result["predicted_signed_relative_bias"] = predicted_bias
    result["predicted_relative_absolute_error"] = predicted_abs
    return result


def single_surface_bias_oof(
    parsed: pd.DataFrame,
    selected: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    candidate, _ = candidate_from_selected(selected, mode, "signed_relative_error")
    matrix, _, _ = base.build_design(parsed, candidate)
    y = parsed["signed_error"].to_numpy(float) / parsed["N"].to_numpy(float)
    oof = np.full(len(parsed), np.nan)
    for _, test in base.fold_definitions(parsed, "cell"):
        train = ~test
        beta = base.fit_ols(matrix[train], y[train])
        oof[test] = matrix[test] @ beta
    result = parsed[
        [
            "request_id",
            "model_label",
            "seed",
            "target_passage_tokens",
            "num_needles",
            "query_order",
        ]
    ].copy()
    result["mode"] = mode
    result["method"] = "single_signed_surface"
    result["observed_signed_relative_error"] = y
    result["observed_relative_absolute_error"] = (
        parsed["absolute_error"].to_numpy(float) / parsed["N"].to_numpy(float)
    )
    result["predicted_signed_relative_bias"] = oof
    result["predicted_relative_absolute_error"] = np.nan
    return result


def aggregate_hurdle_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_label", "target_passage_tokens", "num_needles"]
    for (mode, method), part in predictions.groupby(["mode", "method"]):
        cells = (
            part.groupby(group_cols, as_index=False)
            .agg(
                observed_bias=("observed_signed_relative_error", "mean"),
                predicted_bias=("predicted_signed_relative_bias", "mean"),
                observed_abs=("observed_relative_absolute_error", "mean"),
                predicted_abs=("predicted_relative_absolute_error", "mean"),
            )
        )
        bias_metrics = base.continuous_metrics(
            cells["observed_bias"].to_numpy(),
            cells["predicted_bias"].to_numpy(),
        )
        rows.append(
            {
                "mode": mode,
                "method": method,
                "target": "cell_mean_signed_relative_bias",
                **bias_metrics,
                "n_cells": len(cells),
            }
        )
        valid_abs = np.isfinite(cells["predicted_abs"].to_numpy())
        if valid_abs.any():
            abs_metrics = base.continuous_metrics(
                cells.loc[valid_abs, "observed_abs"].to_numpy(),
                cells.loc[valid_abs, "predicted_abs"].to_numpy(),
            )
            rows.append(
                {
                    "mode": mode,
                    "method": method,
                    "target": "cell_mean_relative_absolute_error",
                    **abs_metrics,
                    "n_cells": int(valid_abs.sum()),
                }
            )
    return pd.DataFrame(rows)


def cluster_robust_power_orders(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate = base.selected_candidate("power_separable")
    for mode in base.MODE_LABELS:
        mode_frame = frame[frame["prompt_mode"] == mode].reset_index(drop=True)
        targets = base.make_binary_targets(mode_frame, mode)
        target_names = ["exact_correct"]
        if mode == "enumeration":
            target_names.extend(["pair_retrieval", "all_pairs_found"])
        for target_name in target_names:
            target = targets[target_name]
            matrix, names, _ = base.build_design(target.frame, candidate)
            fit = base.fit_binomial(
                matrix,
                target.successes,
                target.trials,
                "logistic",
            )
            beta = fit["beta"]
            p = base.predict_binomial(matrix, beta, "logistic")
            weights = target.trials * p * (1.0 - p)
            hessian = (matrix.T * weights) @ matrix + RIDGE * np.eye(
                matrix.shape[1]
            )
            bread = np.linalg.pinv(hessian)
            score_residual = target.successes - target.trials * p
            meat = np.zeros_like(hessian)
            clusters = target.frame["seed"].to_numpy()
            unique_clusters = np.unique(clusters)
            for cluster in unique_clusters:
                mask = clusters == cluster
                score = matrix[mask].T @ score_residual[mask]
                meat += np.outer(score, score)
            n = len(target.frame)
            q = matrix.shape[1]
            correction = (
                len(unique_clusters)
                / max(len(unique_clusters) - 1, 1)
                * (n - 1)
                / max(n - q, 1)
            )
            covariance = correction * bread @ meat @ bread
            se = np.sqrt(np.clip(np.diag(covariance), 0, None))
            for index, name in enumerate(names):
                if not (name.endswith("::log_L") or name.endswith("::log_N")):
                    continue
                model, term = name.split("::", 1)
                estimate = float(beta[index])
                standard_error = float(se[index])
                rows.append(
                    {
                        "mode": mode,
                        "target": target_name,
                        "model_label": model,
                        "term": term,
                        "estimate": estimate,
                        "cluster_seed_se": standard_error,
                        "ci95_low": estimate - 1.96 * standard_error,
                        "ci95_high": estimate + 1.96 * standard_error,
                        "link": "logistic",
                        "formula": "logit(p)=a_m+r_m log(L/5000)+s_m log(N/5)+order nuisance",
                    }
                )
    return pd.DataFrame(rows)


def save_compounding_figures(
    comparison: pd.DataFrame,
    oof: pd.DataFrame,
    figures: Path,
) -> None:
    ordered = comparison.sort_values("log_loss", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered["log_loss"], color="#386c9f")
    axes[0].set_yticks(y, ordered["candidate"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Blocked-cell OOF log loss")
    axes[0].set_title("Effective-N compounding candidates")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(y, ordered["cell_r2"], color="#6a9f58")
    axes[1].set_yticks(y, ordered["candidate"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Held-out cell R²")
    axes[1].set_title("All-pairs-found cell prediction")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "stage2_compounding_candidates.png", dpi=190)
    plt.close(fig)

    cells = (
        oof.groupby(
            ["model_label", "target_passage_tokens", "num_needles"],
            as_index=False,
        )
        .agg(
            observed=("observed_all_pairs_found", "mean"),
            predicted=("predicted_probability", "mean"),
            neff=("predicted_neff", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, part in cells.groupby("model_label"):
        axes[0].scatter(
            part["predicted"], part["observed"], s=22, alpha=0.75, label=model
        )
        axes[1].plot(
            part["num_needles"],
            part["neff"],
            "o",
            ms=3,
            alpha=0.6,
            label=model,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="black")
    metrics = base.continuous_metrics(
        cells["observed"].to_numpy(), cells["predicted"].to_numpy()
    )
    axes[0].text(
        0.03,
        0.97,
        f"cell R²={metrics['r2']:.3f}",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axes[0].set_xlabel("Blocked-cell predicted P(all pairs found)")
    axes[0].set_ylabel("Observed cell rate")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Selected effective-N law")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].plot([1, 30], [1, 30], "--", color="black", label="N_eff=N")
    axes[1].set_xlabel("Observed needle count N (log scale)")
    axes[1].set_ylabel("Predicted effective N_eff (log scale)")
    axes[1].set_title("Effective independent needle count")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "stage2_compounding_calibration.png", dpi=190)
    plt.close(fig)


def save_hurdle_figure(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    figures: Path,
) -> None:
    selected = predictions[predictions["method"] == "selected_hurdle"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, mode in zip(axes, base.MODE_LABELS):
        part = selected[selected["mode"] == mode]
        cells = (
            part.groupby(
                ["model_label", "target_passage_tokens", "num_needles"],
                as_index=False,
            )
            .agg(
                observed=("observed_signed_relative_error", "mean"),
                predicted=("predicted_signed_relative_bias", "mean"),
            )
        )
        for model, group in cells.groupby("model_label"):
            ax.scatter(
                group["predicted"],
                group["observed"],
                s=20,
                alpha=0.75,
                label=model,
            )
        limits = np.nanquantile(
            np.concatenate([cells["observed"], cells["predicted"]]), [0.01, 0.99]
        )
        span = max(limits[1] - limits[0], 0.2)
        lo, hi = limits[0] - 0.08 * span, limits[1] + 0.08 * span
        ax.plot([lo, hi], [lo, hi], "--", color="black")
        row = metrics[
            (metrics["mode"] == mode)
            & (metrics["method"] == "selected_hurdle")
            & (metrics["target"] == "cell_mean_signed_relative_bias")
        ].iloc[0]
        ax.text(
            0.03,
            0.97,
            f"cell R²={row.r2:.3f}\nRMSE={row.rmse:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(base.MODE_LABELS[mode])
        ax.set_xlabel("Hurdle-predicted mean signed bias/N")
        ax.set_ylabel("Observed cell mean signed bias/N")
        ax.grid(alpha=0.2)
    fig.suptitle("Two-part under/over counting law on held-out cells")
    fig.tight_layout()
    fig.savefig(figures / "stage2_hurdle_bias_calibration.png", dpi=190)
    plt.close(fig)


def save_exact_decomposition_figure(
    metrics: pd.DataFrame, figures: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    modes = list(base.MODE_LABELS)
    methods = ["stage1_nested_direct_surface", "parse_times_conditional_exact"]
    labels = ["Direct exact surface", "P(parse) × P(exact|parse)"]
    x = np.arange(len(modes))
    width = 0.34
    for index, (method, label) in enumerate(zip(methods, labels)):
        part = metrics[metrics["method"] == method].set_index("mode").reindex(modes)
        axes[0].bar(
            x + (index - 0.5) * width,
            part["log_loss"],
            width,
            label=label,
        )
        axes[1].bar(
            x + (index - 0.5) * width,
            part["cell_r2"],
            width,
            label=label,
        )
    for ax in axes:
        ax.set_xticks(x, [base.MODE_LABELS[m] for m in modes], rotation=15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Blocked-cell OOF log loss (lower better)")
    axes[1].set_ylabel("Held-out cell R² (higher better)")
    axes[0].set_title("Request-level probability")
    axes[1].set_title("Cell response surface")
    axes[1].legend(fontsize=8)
    fig.suptitle("Exact-count hurdle decomposition")
    fig.tight_layout()
    fig.savefig(figures / "stage2_exact_decomposition.png", dpi=190)
    plt.close(fig)


def save_power_order_figure(orders: pd.DataFrame, figures: Path) -> None:
    exact = orders[orders["target"] == "exact_correct"].copy()
    modes = list(base.MODE_LABELS)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    for row_index, mode in enumerate(modes):
        part = exact[exact["mode"] == mode]
        models = part["model_label"].drop_duplicates().tolist()
        for col_index, term in enumerate(("log_L", "log_N")):
            ax = axes[row_index, col_index]
            sub = part[part["term"] == term].set_index("model_label").reindex(models)
            x = np.arange(len(models))
            estimate = sub["estimate"].to_numpy(float)
            lower = estimate - sub["ci95_low"].to_numpy(float)
            upper = sub["ci95_high"].to_numpy(float) - estimate
            ax.errorbar(
                x,
                estimate,
                yerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=3,
                color="#2f6690",
            )
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x, models, rotation=28, ha="right", fontsize=8)
            ax.set_ylabel("Logistic-link coefficient")
            ax.set_title(
                f"{base.MODE_LABELS[mode]} — "
                + ("length order r_m" if term == "log_L" else "needle order s_m")
            )
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Reference separable-power exact law (seed-clustered 95% intervals)"
    )
    fig.tight_layout()
    fig.savefig(figures / "stage2_reference_power_orders.png", dpi=190)
    plt.close(fig)


def table_html(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    data = frame if columns is None else frame[columns]
    return data.to_html(
        index=False, border=0, classes="data-table", na_rep="—", float_format="%.4f"
    )


def build_stage2_fragment(
    comp_comparison: pd.DataFrame,
    comp_parameters: pd.DataFrame,
    hurdle_metrics: pd.DataFrame,
    exact_metrics: pd.DataFrame,
    magnitude_comparison: pd.DataFrame,
    power_orders: pd.DataFrame,
) -> str:
    selected_comp = base.selected_by_one_se(comp_comparison, "selection_score")
    selected_magnitude = magnitude_comparison[
        magnitude_comparison["selected_one_se"]
    ].copy()
    exact_orders = power_orders[power_orders["target"] == "exact_correct"].copy()
    exact_orders["mode"] = exact_orders["mode"].map(base.MODE_LABELS)
    exact_orders["term"] = exact_orders["term"].map(
        {"log_L": "length order r_m", "log_N": "needle order s_m"}
    )
    tau_median = float(comp_parameters.iloc[0]["tau_median"])
    tau_low = float(comp_parameters.iloc[0]["tau_ci95_low"])
    tau_high = float(comp_parameters.iloc[0]["tau_ci95_high"])
    return f"""
<!-- COUNTING_MECHANISM_STAGE2_START -->
<section id="counting-mechanism-stage2">
<h2>8. 第二阶段：从相关检索与正负误差混合中提炼机制</h2>
<div class="callout">
第一阶段的两个失败诊断被保留：独立 needle 假说 <code>q^N</code> 对 all-pairs-found 的 held-out cell R² 较低；单一 signed-bias 曲面也因欠计与多计抵消而不稳定。第二阶段在看到这些失败后单独冻结，只比较 <code>N_eff=κN^τ</code> 与 under/over hurdle 两个有机制含义的小型候选族。
</div>

<h3>8.1 Enumeration 的有效独立 needle 数</h3>
<div class="formula"><strong>Correlated-retrieval law</strong><br>
P(all pairs found | L,N,m,o) = q<sub>m</sub>(L,N,o)<sup>N<sub>eff,m</sub>(N,o)</sup>,<br>
N<sub>eff,m</sub> = κ<sub>m</sub>N<sup>τ<sub>m</sub></sup>exp[o<sub>m</sub>I(query-last)].
</div>
<p>选择结果：<strong>{html.escape(str(selected_comp['candidate']))}</strong>；blocked-cell OOF log loss={selected_comp['log_loss']:.4f}，cell R²={selected_comp['cell_r2']:.4f}。共享 needle 阶数的 seed-cluster bootstrap 中位数为 <strong>τ={tau_median:.3f}</strong>（95% percentile interval {tau_low:.3f}–{tau_high:.3f}）。若 N_eff/N&lt;1，表示一次共同难度会让多个 needle 的成功/失败正相关，因此“有效独立试验数”少于实际 needle 数；它是经验相关性参数，不是网络内部单元数。</p>
<div class="table-wrap">{table_html(comp_comparison.sort_values('log_loss'))}</div>
<figure><img src="figures/stage2_compounding_candidates.png" alt="Effective N candidates">
<figcaption>图 7｜有效 N 候选的完全 blocked-cell OOF 比较。左轴为 request-level log loss，右轴为 model×L×N cell 均值 R²；independence 行就是第一阶段保留的 q^N 基线。</figcaption></figure>
<figure><img src="figures/stage2_compounding_calibration.png" alt="Effective N calibration">
<figcaption>图 8｜左：所选 effective-N law 的 held-out cell 校准；右：实际 N 与估计 N_eff，双轴均为 log。虚线 N_eff=N 代表独立 needle；偏离虚线量化相关检索/共同难度。</figcaption></figure>
<div class="table-wrap">{table_html(comp_parameters)}</div>

<h3>8.2 Bias 不是单一曲面，而是 hurdle mixture</h3>
<div class="formula"><strong>Two-part bias law</strong><br>
E[(N̂−N)/N] = P(over)·E[|N̂−N|/N | over] − P(under)·E[|N̂−N|/N | under].
</div>
<p>概率项使用 grouped logistic law；幅度项拟合 <code>log(1+|error|/N)</code> 并以 training-only Duan smearing 反变换。这样不会让符号抵消掩盖 L、N 对错误频率和错误幅度的不同作用。</p>
<div class="table-wrap">{table_html(hurdle_metrics.sort_values(['mode','target','method']))}</div>
<figure><img src="figures/stage2_hurdle_bias_calibration.png" alt="Hurdle bias calibration">
<figcaption>图 9｜每点是 held-out model×L×N cell。横轴为 under/over hurdle 组合出的平均 signed bias/N，纵轴为观测 cell 均值；各面板分别对应 nonthinking、enumeration、CoT。</figcaption></figure>
<p>条件幅度候选（每个 mode×方向最低 held-out score 的行）：</p>
<div class="table-wrap">{table_html(selected_magnitude[['mode','direction','candidate','seed_r2','cell_r2','seed_nrmse','cell_nrmse']])}</div>

<h3>8.3 Exact accuracy 的 parse × counting 分解</h3>
<div class="formula">P(exact) = P(parse) × P(exact | parse).</div>
<div class="table-wrap">{table_html(exact_metrics.sort_values(['mode','method']))}</div>
<figure><img src="figures/stage2_exact_decomposition.png" alt="Exact decomposition">
<figcaption>图 10｜蓝/橙柱比较直接 exact surface 与 parse×conditional-count hurdle。在 log loss 上越低越好，在 cell R² 上越高越好；若乘积分解不改善，说明直接 surface 已更有效地吸收了组件相关性。</figcaption></figure>

<h3>8.4 可比较的 L、N 阶数参考</h3>
<p>即使 one-standard-error 规则选择了 density、burden、root 或 piecewise 坐标，我们仍额外报告同一个 separable-power logistic 参考式，以便跨模型比较 r_m 与 s_m；它不是为了替换 held-out 最佳 law。置信区间按 5 个 seed 聚类，因此只应视为有限簇的稳健性诊断。</p>
<div class="formula">logit p_m = a_m + r_m log(L/5000) + s_m log(N/5) + query-order nuisance.</div>
<figure><img src="figures/stage2_reference_power_orders.png" alt="Reference power orders">
<figcaption>图 11｜exact correctness 的参考幂阶与 seed-clustered 95% 区间。负 r_m/s_m 表示相应变量增大时准确率下降；区间跨 0 表示当前数据不足以稳定确定该独立阶数。</figcaption></figure>
<div class="table-wrap">{table_html(exact_orders[['mode','model_label','term','estimate','cluster_seed_se','ci95_low','ci95_high']])}</div>

<h3>8.5 第二阶段的解释边界</h3>
<ul>
<li>Stage 2 是由预先保留的 failure diagnostics 触发的探索性细化；它与 stage-1 预注册比较分开记录。</li>
<li>高 cell R² 说明响应面在当前 L、N 网格内可复现，不意味着能外推到更长 context 或更大的 N。</li>
<li>N_eff 是相关性/异质性的经验摘要；hurdle law 是输出行为分解。二者都不是对隐藏状态内部算法的直接因果识别。</li>
</ul>
</section>
<!-- COUNTING_MECHANISM_STAGE2_END -->
"""


def inject_fragment(path: Path, fragment: str, main_report: bool) -> None:
    source = path.read_text(encoding="utf-8")
    start = "<!-- COUNTING_MECHANISM_STAGE2_START -->"
    end = "<!-- COUNTING_MECHANISM_STAGE2_END -->"
    adjusted = fragment
    if main_report:
        adjusted = adjusted.replace(
            'src="figures/',
            'src="analysis/counting_mechanism_law_v1/figures/',
        )
    if start in source and end in source:
        left = source.split(start, 1)[0]
        right = source.split(end, 1)[1]
        source = left + adjusted + right
    else:
        marker = "</main>" if "</main>" in source else "</body>"
        source = source.replace(marker, adjusted + marker, 1)
    path.write_text(source, encoding="utf-8")


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
    logs = out / "logs"
    for directory in (tables, figures, logs):
        directory.mkdir(parents=True, exist_ok=True)

    frame = base.enrich(
        pd.read_csv(report_root / "tables" / "request_level_report.csv")
    )
    selected = pd.read_csv(tables / "selected_laws.csv")
    selected.attrs["out"] = out

    enum = frame[frame["prompt_mode"] == "enumeration"].reset_index(drop=True)
    enum_targets = base.make_binary_targets(enum, "enumeration")
    retrieval_candidate, retrieval_link = candidate_from_selected(
        selected, "enumeration", "pair_retrieval"
    )
    (
        comp_comparison,
        comp_folds,
        comp_oof,
        comp_parameters,
        q_seed,
    ) = effective_retrieval_search(
        enum,
        enum_targets["pair_retrieval"],
        enum_targets["all_pairs_found"],
        retrieval_candidate,
        retrieval_link,
    )
    selected_comp = base.selected_by_one_se(comp_comparison, "selection_score")
    (
        comp_bootstrap,
        comp_bootstrap_summary,
        comp_leave_level,
    ) = compounding_parameter_stability(
        enum,
        q_seed,
        enum_targets["all_pairs_found"].successes,
        str(selected_comp["candidate"]),
    )
    comp_parameters = comp_parameters.merge(
        comp_bootstrap_summary, on="model_label", how="left"
    )

    exact_metrics, exact_oof = exact_hurdle_decomposition(frame, selected)
    (
        magnitude_comparison,
        magnitude_folds,
        magnitude_coefficients,
        magnitude_candidates,
    ) = conditional_magnitude_search(frame)

    hurdle_predictions: list[pd.DataFrame] = []
    for mode in base.MODE_LABELS:
        parsed = frame[
            (frame["prompt_mode"] == mode) & (frame["parse_success"] == 1)
        ].reset_index(drop=True)
        hurdle_predictions.append(
            hurdle_oof_for_mode(
                parsed, selected, mode, magnitude_candidates, baseline=False
            )
        )
        hurdle_predictions.append(
            hurdle_oof_for_mode(
                parsed, selected, mode, magnitude_candidates, baseline=True
            )
        )
        hurdle_predictions.append(single_surface_bias_oof(parsed, selected, mode))
    hurdle_oof = pd.concat(hurdle_predictions, ignore_index=True)
    hurdle_metrics = aggregate_hurdle_metrics(hurdle_oof)
    power_orders = cluster_robust_power_orders(frame)

    comp_comparison.to_csv(
        tables / "stage2_compounding_candidate_comparison.csv", index=False
    )
    comp_folds.to_csv(tables / "stage2_compounding_fold_metrics.csv", index=False)
    comp_oof.to_csv(tables / "stage2_compounding_oof.csv", index=False)
    comp_parameters.to_csv(
        tables / "stage2_compounding_parameters.csv", index=False
    )
    comp_bootstrap.to_csv(
        tables / "stage2_compounding_bootstrap_draws.csv", index=False
    )
    comp_leave_level.to_csv(
        tables / "stage2_compounding_leave_level_parameters.csv", index=False
    )
    exact_metrics.to_csv(
        tables / "stage2_exact_decomposition_metrics.csv", index=False
    )
    exact_oof.to_csv(
        tables / "stage2_exact_decomposition_oof.csv", index=False
    )
    magnitude_comparison.to_csv(
        tables / "stage2_magnitude_candidate_comparison.csv", index=False
    )
    magnitude_folds.to_csv(
        tables / "stage2_magnitude_fold_metrics.csv", index=False
    )
    magnitude_coefficients.to_csv(
        tables / "stage2_magnitude_coefficients.csv", index=False
    )
    hurdle_oof.to_csv(tables / "stage2_hurdle_oof.csv", index=False)
    hurdle_metrics.to_csv(tables / "stage2_hurdle_metrics.csv", index=False)
    power_orders.to_csv(tables / "stage2_reference_power_orders.csv", index=False)

    save_compounding_figures(comp_comparison, comp_oof, figures)
    save_hurdle_figure(hurdle_oof, hurdle_metrics, figures)
    save_exact_decomposition_figure(exact_metrics, figures)
    save_power_order_figure(power_orders, figures)

    fragment = build_stage2_fragment(
        comp_comparison,
        comp_parameters,
        hurdle_metrics,
        exact_metrics,
        magnitude_comparison,
        power_orders,
    )
    (out / "stage2_fragment.html").write_text(fragment, encoding="utf-8")
    inject_fragment(out / "report.html", fragment, main_report=False)
    if args.inject_main_report:
        inject_fragment(report_root / "report.html", fragment, main_report=True)

    stage2_manifest = {
        "analysis": "counting_mechanism_law_v1_stage2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trigger": {
            "independent_q_power_n_cell_r2": float(
                pd.read_csv(
                    tables / "enumeration_compounding_metrics.csv"
                )
                .query("scheme == 'cell' and target == 'all_pairs_found'")
                .iloc[0]["cell_r2"]
            ),
            "signed_bias_single_surface_was_weak": True,
        },
        "selected_compounding": selected_comp.to_dict(),
        "compounding_stability": {
            "bootstrap_unit": "seed cluster",
            "bootstrap_draws": 400,
            "bootstrap_random_seed": 20260725,
            "leave_level_schemes": ["seed", "length", "needle"],
        },
        "hurdle_metrics": hurdle_metrics.to_dict(orient="records"),
        "exact_decomposition_metrics": exact_metrics.to_dict(orient="records"),
        "source_sha256": sha256(
            report_root / "tables" / "request_level_report.csv"
        ),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (out / "stage2_manifest.json").write_text(
        json.dumps(stage2_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (logs / "stage2_run.log").write_text(
        "\n".join(
            [
                f"completed_at_utc={datetime.now(timezone.utc).isoformat()}",
                f"selected_compounding={selected_comp['candidate']}",
                f"selected_compounding_cell_r2={selected_comp['cell_r2']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    files = [
        path
        for path in out.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.tsv"
    ]
    (out / "SHA256SUMS.tsv").write_text(
        "\n".join(
            f"{sha256(path)}\t{path.relative_to(out)}" for path in sorted(files)
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_compounding": selected_comp.to_dict(),
                "hurdle_metrics": hurdle_metrics.round(5).to_dict(
                    orient="records"
                ),
                "exact_decomposition": exact_metrics.round(5).to_dict(
                    orient="records"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
