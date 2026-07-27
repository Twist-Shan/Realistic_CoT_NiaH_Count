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
from matplotlib.lines import Line2D
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
LAW_MODE_FAMILIES = {
    "Direct": ("direct",),
    "Enumerate": ("enumeration_index", "enumeration_bullet"),
}
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
            "lnN",
            "ln N",
            r"b=\alpha+\beta_N\ln N",
            ("lnN",),
            lambda n, l: cols(np.log(n)),
        ),
        Candidate(
            "lnL",
            "ln L",
            r"b=\alpha+\beta_L\ln L_k",
            ("lnLk",),
            lambda n, l: cols(np.log(l)),
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
            "ln_product",
            "ln(NL)",
            r"b=\alpha+\beta_P\ln(NL_k)",
            ("lnNLk",),
            lambda n, l: cols(np.log(n * l)),
        ),
        Candidate(
            "ln_density",
            "ln(N/L)",
            r"b=\alpha+\beta_R\ln(N/L_k)",
            ("lnN/Lk",),
            lambda n, l: cols(np.log(n / l)),
        ),
        Candidate(
            "N_L",
            "N + L",
            r"b=\alpha+\beta_NN+\beta_LL_k",
            ("N", "Lk"),
            lambda n, l: cols(n, l),
        ),
        Candidate(
            "N_lnL",
            "N + ln L",
            r"b=\alpha+\beta_NN+\beta_L\ln L_k",
            ("N", "lnLk"),
            lambda n, l: cols(n, np.log(l)),
        ),
        Candidate(
            "lnN_L",
            "ln N + L",
            r"b=\alpha+\beta_N\ln N+\beta_LL_k",
            ("lnN", "Lk"),
            lambda n, l: cols(np.log(n), l),
        ),
        Candidate(
            "lnN_lnL",
            "ln N + ln L",
            r"b=\alpha+\beta_N\ln N+\beta_L\ln L_k",
            ("lnN", "lnLk"),
            lambda n, l: cols(np.log(n), np.log(l)),
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
            "N_lnL_interaction",
            "N + ln L + N ln L",
            r"b=\alpha+\beta_NN+\beta_L\ln L_k+\beta_{NL}N\ln L_k",
            ("N", "lnLk", "NlnLk"),
            lambda n, l: cols(n, np.log(l), n * np.log(l)),
        ),
        Candidate(
            "lnN_lnL_interaction",
            "ln N + ln L + interaction",
            r"b=\alpha+\beta_N\ln N+\beta_L\ln L_k+\beta_{NL}\ln N\ln L_k",
            ("lnN", "lnLk", "lnNlnLk"),
            lambda n, l: cols(np.log(n), np.log(l), np.log(n) * np.log(l)),
        ),
        Candidate(
            "N_piece10_lnL",
            "segmented N at 10 + ln L",
            r"b=\alpha+\beta_NN+\beta_H(N-10)_++\beta_L\ln L_k",
            ("N", "(N-10)+", "lnLk"),
            lambda n, l: cols(n, np.maximum(n - 10.0, 0.0), np.log(l)),
        ),
        Candidate(
            "lnN_L_piece5",
            "ln N + segmented L at 5k",
            r"b=\alpha+\beta_N\ln N+\beta_LL_k+\beta_H(L_k-5)_+",
            ("lnN", "Lk", "(Lk-5)+"),
            lambda n, l: cols(np.log(n), l, np.maximum(l - 5.0, 0.0)),
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


def regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    fitted_parameters: int | None = None,
) -> dict[str, float]:
    """Return scale, fit, and Gaussian information-criterion diagnostics."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction)
    y = y[valid]
    prediction = prediction[valid]
    if not len(y):
        return {
            "n": 0,
            "r2": math.nan,
            "adjusted_r2": math.nan,
            "mse": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "median_ae": math.nan,
            "nrmse_sd": math.nan,
            "sse_deviance": math.nan,
            "log_likelihood": math.nan,
            "aic": math.nan,
            "aicc": math.nan,
            "bic": math.nan,
        }
    residual = y - prediction
    squared = residual * residual
    sse = float(squared.sum())
    mse = float(squared.mean())
    rmse = math.sqrt(mse)
    mae = float(np.abs(residual).mean())
    median_ae = float(np.median(np.abs(residual)))
    target_sd = float(np.std(y, ddof=1)) if len(y) > 1 else math.nan
    nrmse_sd = (
        rmse / target_sd
        if np.isfinite(target_sd) and target_sd > 1e-12
        else math.nan
    )
    r2 = safe_r2(y, prediction)
    adjusted_r2 = math.nan
    log_likelihood = math.nan
    aic = math.nan
    aicc = math.nan
    bic = math.nan
    if fitted_parameters is not None:
        n = len(y)
        regression_k = int(fitted_parameters)
        if np.isfinite(r2) and n > regression_k:
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - regression_k)
        # Gaussian OLS likelihood. K includes the residual-variance parameter.
        likelihood_k = regression_k + 1
        sigma2_mle = max(sse / n, np.finfo(float).tiny)
        log_likelihood = float(
            -0.5 * n * (math.log(2 * math.pi) + 1 + math.log(sigma2_mle))
        )
        aic = float(2 * likelihood_k - 2 * log_likelihood)
        if n > likelihood_k + 1:
            aicc = float(
                aic
                + 2
                * likelihood_k
                * (likelihood_k + 1)
                / (n - likelihood_k - 1)
            )
        bic = float(math.log(n) * likelihood_k - 2 * log_likelihood)
    return {
        "n": int(len(y)),
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_ae": median_ae,
        "nrmse_sd": nrmse_sd,
        "sse_deviance": sse,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
    }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if (
        len(left) < 3
        or np.ptp(left) <= 1e-12
        or np.ptp(right) <= 1e-12
    ):
        return math.nan
    return float(stats.spearmanr(left, right).statistic)


def max_abs_finite(*values: float) -> float:
    finite = [abs(float(value)) for value in values if np.isfinite(value)]
    return max(finite) if finite else math.nan


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
    fold_mae: list[float] = []
    for seed in sorted(group["seed"].unique()):
        test = group[group["seed"] == seed]
        train = group[group["seed"] != seed]
        if train.empty or test.empty:
            continue
        prediction = fit_predict_standardized(candidate, train, test, target)
        oof[test.index] = prediction
        residual = test[target].to_numpy(dtype=float) - prediction
        fold_mse.append(
            float(np.mean(residual**2))
        )
        fold_mae.append(float(np.mean(np.abs(residual))))
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

    def level_oof(column: str) -> dict[str, float]:
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
        return regression_metrics(
            level_cells["observed"].to_numpy(),
            level_cells["prediction"].to_numpy(),
        )

    request_metrics = regression_metrics(group[target].to_numpy(), oof)
    crossfit_cell_metrics = regression_metrics(
        crossfit_cells["observed"].to_numpy(),
        crossfit_cells["prediction"].to_numpy(),
    )
    fit_cell_metrics = regression_metrics(
        cells[target].to_numpy(),
        prediction,
        fitted_parameters=candidate.k,
    )
    leave_n = level_oof("N")
    leave_l = level_oof("L")
    fit_residual = cells[target].to_numpy(dtype=float) - prediction
    return {
        "seed_cv_mse": float(np.mean(fold_mse)),
        "seed_cv_rmse": math.sqrt(float(np.mean(fold_mse))),
        "seed_cv_mae": float(np.mean(fold_mae)),
        "seed_cv_se": float(
            np.std(fold_mse, ddof=1) / math.sqrt(len(fold_mse))
            if len(fold_mse) > 1
            else math.nan
        ),
        "request_oof_r2": request_metrics["r2"],
        "request_oof_rmse": request_metrics["rmse"],
        "request_oof_mae": request_metrics["mae"],
        "request_oof_median_ae": request_metrics["median_ae"],
        "request_oof_nrmse_sd": request_metrics["nrmse_sd"],
        "cell_crossfit_r2": crossfit_cell_metrics["r2"],
        "cell_crossfit_rmse": crossfit_cell_metrics["rmse"],
        "cell_crossfit_mae": crossfit_cell_metrics["mae"],
        "cell_crossfit_median_ae": crossfit_cell_metrics["median_ae"],
        "cell_crossfit_nrmse_sd": crossfit_cell_metrics["nrmse_sd"],
        "cell_fit_r2": fit_cell_metrics["r2"],
        "cell_adjusted_r2": fit_cell_metrics["adjusted_r2"],
        "cell_fit_rmse": fit_cell_metrics["rmse"],
        "cell_fit_mae": fit_cell_metrics["mae"],
        "cell_fit_median_ae": fit_cell_metrics["median_ae"],
        "cell_fit_nrmse_sd": fit_cell_metrics["nrmse_sd"],
        "cell_sse_deviance": fit_cell_metrics["sse_deviance"],
        "cell_log_likelihood": fit_cell_metrics["log_likelihood"],
        "cell_aic": fit_cell_metrics["aic"],
        "cell_aicc": fit_cell_metrics["aicc"],
        "cell_bic": fit_cell_metrics["bic"],
        "leave_N_out_r2": leave_n["r2"],
        "leave_N_out_rmse": leave_n["rmse"],
        "leave_N_out_mae": leave_n["mae"],
        "leave_L_out_r2": leave_l["r2"],
        "leave_L_out_rmse": leave_l["rmse"],
        "leave_L_out_mae": leave_l["mae"],
        "residual_spearman_N": safe_spearman(
            fit_residual,
            cells["N"].to_numpy(dtype=float),
        ),
        "residual_spearman_L": safe_spearman(
            fit_residual,
            cells["L"].to_numpy(dtype=float),
        ),
        "abs_residual_spearman_fitted": safe_spearman(
            np.abs(fit_residual),
            prediction,
        ),
        "max_abs_cell_residual": float(np.max(np.abs(fit_residual))),
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
            "lnN": r"\ln N",
            "lnLk": r"\ln L_k",
            "lnNLk": r"\ln(NL_k)",
            "lnN/Lk": r"\ln(N/L_k)",
            "N/Lk": r"N/L_k",
            "Lk/N": r"L_k/N",
            "NlnLk": r"N\ln L_k",
            "lnNlnLk": r"\ln N\ln L_k",
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


def fold_html(summary: str, content: str, *, open_by_default: bool = False) -> str:
    open_attribute = " open" if open_by_default else ""
    return (
        f'<details class="report-fold"{open_attribute}>'
        f"<summary>{html.escape(summary)}</summary>"
        f'<div class="fold-content">{content}</div>'
        "</details>"
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
.evidence{background:#f2f7fb;border-left:4px solid #4a7fa5;padding:13px 17px;margin:14px 0}
.hypothesis{background:#fff8ed;border-left:4px solid var(--amber);padding:13px 17px;margin:14px 0}
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
.report-fold{margin:16px 0;padding:12px 16px}
.report-fold>summary{color:var(--blue)}
.fold-content{margin-top:12px}
.fold-content>figure:first-child,.fold-content>.table-wrap:first-child{margin-top:4px}
.source-list li{margin:7px 0}
.case-meta{color:var(--muted);font-size:.88rem;margin:5px 0 10px}
.equation{overflow-x:auto;text-align:center;margin:16px 0;padding:10px}
.small{font-size:.88rem;color:var(--muted)}
.good{color:var(--green);font-weight:650}.mid{color:var(--amber);font-weight:650}.weak{color:var(--red);font-weight:650}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
@media(max-width:760px){main{padding:28px 18px}h1{font-size:1.65rem}.metric-grid{grid-template-columns:1fr 1fr}}
@media print{body{background:white}main{max-width:none;padding:20px}details{break-inside:avoid}figure{break-inside:avoid}}
"""


def page(title: str, subtitle: str, body: str, generated: str) -> str:
    # Report bodies are assembled with raw f-strings so TeX commands such as
    # ``\alpha`` cannot become Python control characters.  Static TeX fragments
    # therefore use doubled backslashes in the Python source; collapse those
    # once at the HTML boundary so MathJax receives normal ``\(...\)`` and
    # ``\[...\]`` delimiters and single-backslash commands.
    while "\\\\" in body:
        body = body.replace("\\\\", "\\")
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
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.8), sharey=True)
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
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Accuracy versus true needle count" if axis == "N" else "Accuracy versus passage length",
        y=0.992,
        fontsize=13,
    )
    fig.subplots_adjust(
        top=0.895,
        bottom=0.065,
        left=0.07,
        right=0.985,
        hspace=0.42,
        wspace=0.25,
    )
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
    fig, ax = plt.subplots(figsize=(10.5, 9.8))
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
    fig.suptitle("Failure budget: mutually exclusive first failure", y=0.992, fontsize=13)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.30, right=0.985)
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
            "The thought channel was separated, but the model's final channel still repeated prose or a record list before Total; the strict one-line gate therefore failed, often despite a correct count.",
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


def load_prompt_failure_audit(
    audit_dir: Path, destination_tables: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    required = {
        "request_level": audit_dir / "request_level_failure_audit.csv",
        "failure_types": audit_dir / "failure_type_summary.csv",
        "cell_diagnostics": audit_dir / "cell_mechanism_diagnostics.csv",
        "paired_index_bullet": audit_dir / "paired_index_bullet.csv",
        "examples": audit_dir / "phenomenon_examples.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prompt-failure audit files: " + ", ".join(missing))
    destination_tables.mkdir(parents=True, exist_ok=True)
    for key, path in required.items():
        if key != "examples":
            shutil.copy2(path, destination_tables / path.name)
    shutil.copy2(required["examples"], destination_tables / required["examples"].name)
    return (
        pd.read_csv(required["failure_types"]),
        pd.read_csv(required["cell_diagnostics"]),
        pd.read_csv(required["paired_index_bullet"]),
        json.loads(required["examples"].read_text(encoding="utf-8")),
    )


def percent(value: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(value) else f"{100 * value:.{digits}f}%"


def build_prompt_report(
    data: pd.DataFrame,
    v2_root: Path,
    failure_audit_dir: Path,
    output: Path,
    generated: str,
) -> dict[str, pd.DataFrame]:
    figures = output / "figures" / "prompt"
    tables = output / "tables" / "prompt"
    tables.mkdir(parents=True, exist_ok=True)
    failure_types, cell_diagnostics, index_bullet, audit_examples = load_prompt_failure_audit(
        failure_audit_dir, tables
    )
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

    failure_counts = (
        data[data["error_category"] != "success"]
        .groupby(["model", "mode", "error_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ERROR_ORDER:
        if column not in failure_counts:
            failure_counts[column] = 0
    gate_table = cell_diagnostics.merge(failure_counts, on=["model", "mode"], how="left")
    gate_table["format_failure_n"] = (
        gate_table["response_format_failure"] + gate_table["enumeration_format_failure"]
    )
    gate_table = gate_table[
        (gate_table["truncated_n"] > 0)
        | (gate_table["parse_failure"] > 0)
        | (gate_table["format_failure_n"] > 0)
    ].copy()
    gate_display = pd.DataFrame(
        {
            "Model": gate_table["model"],
            "Mode": gate_table["mode"].map(MODE_LABEL),
            "Truncated": gate_table["truncated_n"].astype(int),
            "Truncated without Total": gate_table["truncated_without_Total_n"].astype(int),
            "No parsable Total": gate_table["parse_failure"].astype(int),
            "Strict-format failure": gate_table["format_failure_n"].astype(int),
            "Exact but format-failed": gate_table["exact_but_format_fail_n"].astype(int),
            "Median output tokens": gate_table["median_output_tokens"].round(1),
            "P90 output tokens": gate_table["p90_output_tokens"].round(1),
        }
    )

    concrete = failure_types[
        failure_types["primary_failure"].isin(
            [
                "truncated",
                "parse_failure",
                "response_format_failure",
                "enumeration_format_failure",
            ]
        )
        & (failure_types["count"] >= 2)
    ].copy()
    concrete = concrete.sort_values(["count", "model"], ascending=[False, True])
    concrete_display = pd.DataFrame(
        {
            "Model": concrete["model"],
            "Mode": concrete["mode"].map(MODE_LABEL),
            "Primary gate": concrete["primary_failure"].map(ERROR_LABEL),
            "Concrete output pattern": concrete["failure_subtype"],
            "n / 500": concrete["count"].astype(int),
        }
    )

    index_bullet_display = index_bullet.copy()
    index_bullet_display = pd.DataFrame(
        {
            "Model": index_bullet_display["model"],
            "Index": index_bullet_display["index_success_rate"].map(percent),
            "Bullet": index_bullet_display["bullet_success_rate"].map(percent),
            "Index − Bullet": index_bullet_display["index_minus_bullet_pp"].map(
                lambda x: f"{x:+.1f} pp"
            ),
            "Index-only wins": index_bullet_display["index_only_success"].astype(int),
            "Bullet-only wins": index_bullet_display["bullet_only_success"].astype(int),
            "Index Total=list": index_bullet_display["index_total_matches_list_rate"].map(percent),
            "Bullet Total=list": index_bullet_display["bullet_total_matches_list_rate"].map(percent),
            "Paired exact p": index_bullet_display["mcnemar_exact_p"].map(
                lambda x: f"{x:.2e}"
            ),
        }
    )

    native_models = [
        "Qwen3-8B",
        "Gemma4-E4B",
        "Gemma4-12B",
        "DeepSeek-R1-0528-Qwen3-8B",
        "GLM-Z1-9B-0414",
    ]
    native_diag = cell_diagnostics[
        (cell_diagnostics["model"].isin(native_models))
        & (cell_diagnostics["mode"] == "native_thinking")
    ].copy()
    native_diag_display = pd.DataFrame(
        {
            "Model": native_diag["model"],
            "Strict success": native_diag["success_rate"].map(percent),
            "Exact count": native_diag["exact_rate"].map(percent),
            "Strict format": native_diag["format_rate"].map(percent),
            "Truncated": native_diag["truncated_n"].astype(int),
            "Median tokens": native_diag["median_output_tokens"].round(1),
            "P90 tokens": native_diag["p90_output_tokens"].round(1),
            "P99 tokens": native_diag["p99_output_tokens"].round(1),
            "Restart signal": native_diag["restart_flag_n"].astype(int),
        }
    )

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

    case_titles = {
        "original_gemma_direct_trunc": "Gemma4-12B 旧 Direct：先解释/列举，64 tokens 前没有 Total",
        "qwen17_bullet_loop": "Qwen3-1.7B Bullet：重复相同记录直到 1,536-token 截断",
        "qwen17_index_no_total": "Qwen3-1.7B Index：列表完成但遗漏 Total",
        "gemma12_native_verbose_final": "Gemma4-12B Native：计数正确，但 final channel 又列了一遍记录",
        "gemma12_native_trunc": "Gemma4-12B Native：扫描无关文本并陷入重复",
        "deepseek_native_trunc": "DeepSeek-R1：多次扫描/重启后耗尽 4,096 tokens",
        "deepseek_wrong_count": "DeepSeek-R1：推理完成但计数漂移",
        "glmz1_native_trunc": "GLM-Z1：长 reasoning 未在预算内到达 Total",
        "gemma12_bullet_wrong": "Gemma4-12B Bullet：pair 基本找全但 Total 漏计",
        "gemma12_index_success": "Gemma4-12B Index：序号与 Total 形成一致的外部计数",
    }
    example_blocks = []
    for row in audit_examples:
        title = case_titles.get(
            str(row.get("key")),
            f"{row.get('model')} · {MODE_LABEL.get(str(row.get('mode')), row.get('mode'))}",
        )
        signals = ", ".join(map(str, row.get("overthinking_signals") or [])) or "none"
        example_blocks.append(
            f"""<details><summary>{html.escape(title)}</summary>
            <div class="case-meta">request_id={html.escape(str(row.get('request_id')))}；N={int(row.get('N'))}；L={int(row.get('L')):,}；output tokens={int(row.get('output_tokens'))}；exact={html.escape(str(row.get('exact_count')))}；format={html.escape(str(row.get('format_ok')))}；signals={html.escape(signals)}</div>
            <p><strong>审计分类：</strong>{html.escape(str(row.get('failure_subtype')))}</p>
            <pre><code>{html.escape(str(row.get('raw_excerpt')))}</code></pre></details>"""
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
    def cell_metric(model: str, mode: str, column: str) -> float:
        return float(
            summary[
                (summary.model.astype(str) == model)
                & (summary["mode"].astype(str) == mode)
            ][column].iloc[0]
        )

    metrics.update(
        {
            "qwen8_native": cell_metric("Qwen3-8B", "native_thinking", "success_rate"),
            "deepseek_native": cell_metric(
                "DeepSeek-R1-0528-Qwen3-8B", "native_thinking", "success_rate"
            ),
            "glmz1_native": cell_metric("GLM-Z1-9B-0414", "native_thinking", "success_rate"),
            "glm4_index": cell_metric("GLM-4-9B-0414", "enumeration_index", "success_rate"),
            "glm4_bullet": cell_metric("GLM-4-9B-0414", "enumeration_bullet", "success_rate"),
            "gemmae4_native": cell_metric("Gemma4-E4B", "native_thinking", "success_rate"),
            "gemmae4_index": cell_metric("Gemma4-E4B", "enumeration_index", "success_rate"),
            "gemma12_index": cell_metric("Gemma4-12B", "enumeration_index", "success_rate"),
            "gemma12_bullet": cell_metric("Gemma4-12B", "enumeration_bullet", "success_rate"),
        }
    )

    body = rf"""
<div class="meta"><span class="pill">14,500 selected requests</span><span class="pill">29 model × mode cells</span><span class="pill">500 requests per cell</span><span class="pill">V2 + V2.1 current-prompt composite</span></div>
<nav class="toc"><a href="#scope">1. 口径</a><a href="#prompts">2. Prompt</a><a href="#accuracy">3. 正确率</a><a href="#difficulty">4. N 与 L</a><a href="#failures">5. 截断/格式</a><a href="#mechanisms">6. 四个现象</a><a href="#gemma">7. Gemma Direct</a><a href="#sources">8. 证据边界</a></nav>

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
<details class="report-fold"><summary>展开图 1：全部模型 × mode 的正确率热图</summary><div class="fold-content">
<figure><img src="figures/prompt/01_accuracy_heatmap.png" alt="Accuracy heatmap"><figcaption><strong>图 1.</strong> 每格为 500 条请求的 registered success。横轴是 prompt mode，纵轴是模型；空格代表该组合未运行，不按 0 处理。</figcaption></figure>
</div></details>
<details class="report-fold"><summary>展开每个模型最佳 mode 的简表</summary><div class="fold-content">{table_html(best_modes)}</div></details>
<h3>完整指标</h3>
<details class="report-fold"><summary>展开 29 个模型 × mode 单元的完整指标表</summary><div class="fold-content">{table_html(display_summary)}</div></details>
<p class="small">* Mean bias 仅在成功解析出数值的请求上定义；它不是把 parse failure 当作零误差。</p>
<h3>同 stimulus 配对差异</h3>
<details class="report-fold"><summary>展开所有同 stimulus 配对差异</summary><div class="fold-content">{table_html(paired_show)}</div></details>
<p>Qwen 随规模增大，从 Direct 到显式/隐式过程的收益总体减小：1.7B 强烈依赖 Native Thinking；4B 的 Native 明显优于 Direct；8B 和 32B 的 Index 与 Native 接近饱和。Gemma 则更偏好 Index：E4B 和 12B 的 Index 分别达到 88.2% 与 95.0%。</p>
<div class="conclusion"><strong>本节结论：</strong>没有单一 mode 在每个模型上都同等占优；但 Index/Native 在中大型 Qwen 上接近 96–98%，Index 是两款 Gemma 的最佳严格输出模式。Qwen3-1.7B 是明显能力边界，prompt 变清楚仍不能阻止枚举退化。</div></section>

<section id="difficulty"><h2>4. Needle 数 N 与 Passage 长度 L</h2>
<details class="report-fold"><summary>展开图 2：正确率随 needle 数 N 的变化</summary><div class="fold-content">
<figure><img src="figures/prompt/02_accuracy_by_N.png" alt="Accuracy versus N"><figcaption><strong>图 2.</strong> 横轴为真实 needle 数 \(N\)，纵轴为 registered success；每点平均五个长度和十个 seeds。线只连接离散实验水平，不表示连续插值 law。</figcaption></figure>
</div></details>
<details class="report-fold"><summary>展开图 3：正确率随 passage 长度 L 的变化</summary><div class="fold-content">
<figure><img src="figures/prompt/03_accuracy_by_L.png" alt="Accuracy versus L"><figcaption><strong>图 3.</strong> 横轴为 passage 长度 \(L\)（千 tokens），纵轴为 registered success；每点平均十个 N 水平和十个 seeds。</figcaption></figure>
</div></details>
{''.join(detail_blocks)}
<p>多数过程化模式在 N 和 L 增大时下降，但下降方式并不相同：Direct 常在高 N 发生计数偏差或触发隐式列举；Enumeration 在弱模型上会随列表变长出现漏项、幻觉或循环；Native Thinking 在较强 Qwen 上对设计范围最稳。</p>
<div class="conclusion"><strong>本节结论：</strong>N 与 L 都是有效难度轴，但 N 通常更直接控制需要维护的计数状态；L 主要增加搜索距离。二者在弱模型和长输出模式中表现出交互，而非一个简单的“needle density”即可解释全部正确率。</div></section>

<section id="failures"><h2>5. 截断与不按格式：到底发生了什么</h2>
<details class="report-fold"><summary>展开图 4：各单元的互斥失败构成</summary><div class="fold-content">
<figure><img src="figures/prompt/04_failure_budget.png" alt="Failure composition"><figcaption><strong>图 4.</strong> 每一横条以全部 500 请求为分母，按互斥优先级拆分失败：截断 → 无可解析 Total → 严格格式失败 → 数值错误。标题与图例已移到独立顶部区域；横轴是占全部请求的比例，而不是失败内部比例。</figcaption></figure>
</div></details>
<p>审计使用冻结输出逐条分类，并保持互斥优先级不变。这里的“格式失败”不等于 parser 完全看不懂：只要最终可见文本不是规定的一行 <code>Total: integer</code>，或枚举行不满足指定语法，即使数字正确也仍失败。</p>
<h3>5.1 各单元的输出门槛损失</h3>
<details class="report-fold"><summary>展开存在截断、缺失 Total 或格式失败的单元表</summary><div class="fold-content">{table_html(gate_display)}</div></details>
<p class="small">表只显示至少有一次截断、无 Total 或严格格式失败的单元。完整逐请求审计与全部小频数类型保存在 <code>tables/prompt/request_level_failure_audit.csv</code> 和 <code>failure_type_summary.csv</code>。</p>
<h3>5.2 具体输出形态</h3>
<details class="report-fold"><summary>展开全部具体失败形态及频数</summary><div class="fold-content">{table_html(concrete_display)}</div></details>
<div class="evidence"><strong>直接观察。</strong>旧 Gemma4-12B Direct 的 500/500 都先解释或编号列举，并在 64 tokens 前结束；当前主表中的 Strict Direct 已把该问题降到 0。Gemma4-E4B Direct 仍有 170/500 次“先列举、后 Total”而截断。Qwen3-4B Direct 的 52 次截断中，45 次已有显式列表、7 次为解释/扫描；Qwen3-8B 的 14 次对应 3 次显式列表和 11 次解释/扫描。</div>
<div class="evidence"><strong>长列表退化。</strong>Qwen3-1.7B Bullet 的 125 次截断中，114 次是重复相同城市记录的循环，11 次是未完成长列表；Index 只有 12 次截断，却另有 97 次“列表写完但没写 Total”。Native Thinking 的截断以重扫/重复为主：DeepSeek-R1 为 22 次（16 重启、2 重复、4 长扫描），GLM-Z1 为 41 次（13 重复、28 长扫描），Gemma4-12B 为 8 次（4 重启、2 重复、2 长扫描）。这些请求均保留为失败，没有剔除。</div>
<div class="evidence"><strong>格式失败的主要形态。</strong>Gemma4-12B Native 有 315 次严格格式失败：216 次在 final channel 完整或部分重列记录，99 次加入解释性 prose；其中 297 次数字本身正确。因此 90.8% exact count 与 31.4% strict success 可以同时成立。枚举模式还有三类清晰形态：①列表存在但省略 Total（例如 Qwen3-1.7B Index 97 次、GLM-4 Bullet 25 次）；②bullet 标记和 <code>city: score</code> 被拆成两行；③输出完整句子、占位符、问号或不存在的“记录”，从而不满足严格枚举语法。</div>
<h3>5.3 代表性冻结输出</h3>{''.join(example_blocks)}
<h3>5.4 数值错误方向</h3>
<details class="report-fold"><summary>展开 29 个单元的漏计、过计与 MAE 表</summary><div class="fold-content">{table_html(error_table)}</div></details>
<div class="conclusion"><strong>本节结论：</strong>低准确率至少包含三种不同机制：检索/计数偏差、生成长度失控、以及最终呈现格式失控。Gemma4-12B Native 的主损失是“正确数字 + 冗长 final”；DeepSeek/GLM-Z1 的主额外风险是长 reasoning 与 4,096-token 预算冲突；Qwen3-1.7B 枚举还存在明显的循环和占位符续写。</div></section>

<section id="mechanisms"><h2>6. 四个跨模式现象：结果、案例与解释</h2>
<div class="callout warning"><strong>因果边界：</strong>以下把冻结输出统计称为“证据”，把训练/架构联系称为“解释假说”。本实验没有消融预训练语料、RL 配方或网络组件，因此不能单凭这些数据证明内部因果机制。</div>

<h3>6.1 为什么 Bullet 通常明显低于 Index？</h3>
<details class="report-fold"><summary>展开 Index–Bullet 的同 stimulus 配对检验表</summary><div class="fold-content">{table_html(index_bullet_display)}</div></details>
<div class="evidence"><strong>结果与案例。</strong>7 个成对模型中有 6 个是 Index 更高，差值从 6.6 到 20.4 pp，配对 McNemar 检验均显著；唯一反例是 Qwen3-1.7B（Bullet 38.0%，Index 24.8%）。关键并非 Bullet 找不到城市：Gemma4-12B 的 mean pair recall 为 99.11%，Index 为 99.37%，但 success 相差 20.4 pp；Qwen3-32B 的 Bullet pair recall 甚至略高，却只有 85.8% success。差异集中在“Total 是否等于列表长度”：例如 Gemma4-12B 为 Index 100.0% 对 Bullet 79.6%，Qwen3-32B 为 99.2% 对 85.4%。</div>
<div class="hypothesis"><strong>最简解释。</strong>序号同时承担了检索记录和维护计数状态两项功能：第 <em>k</em> 项的前缀本身就是外部计数器，最后一个序号可以直接复制为 Total。Bullet 只分隔项目，模型仍需在生成完毕后重新计算有多少行；于是“pair 找全但 Total 不等于列表长度”更常见。这是与输出证据一致的功能性解释，不等于已经定位到某个神经回路。</div>
<div class="hypothesis"><strong>为什么 1.7B 反例？</strong>较弱 Qwen3-1.7B 会把数字序列当成继续生成的强模式：出现 <code>(missing)</code>、<code>?</code>、占位符和从 10 续写到 100 的退化，同时 97 次写完列表后遗漏 Total。对它而言，序号的外部计数收益被“数字续写/格式控制失败”抵消；Bullet 虽有 125 次循环截断，仍比 Index 多 66 个净成功样本。</div>
<div class="conclusion"><strong>本小节结论：</strong>对中大型模型，Index 的优势主要是把计数状态显式绑定到列表位置，而不是提升 needle 检索；对 1.7B，序号本身会触发生成退化，因此不能把“Index 总是更好”当作普适规律。</div>

<h3>6.2 为什么 Gemma Thinking 不如 Enumeration，尤其 12B 很低？</h3>
<details class="report-fold"><summary>展开两款 Gemma Native 的长度、格式与重启诊断表</summary><div class="fold-content">{table_html(native_diag_display[native_diag_display['Model'].isin(['Gemma4-E4B','Gemma4-12B'])])}</div></details>
<div class="evidence"><strong>结果与案例。</strong>Gemma4-E4B：Index 88.2%，Bullet 79.4%，Native 79.2%；Native exact count 实为 83.2%，有 22 个格式失败和 1 个截断。Gemma4-12B：Index 95.0%，Bullet 74.6%，Native strict success 31.4%，但 Native exact count 高达 90.8%。在 500 条 Native 中，315 条 final channel 在 Total 前增加解释/列表，297 条数字仍正确；另有 8 条因重复扫描达到 4,096 tokens。其 Native 输出中位数 674.5 tokens、P90 1,996.9，而 Index 中位数仅 50.5。</div>
<div class="hypothesis"><strong>解释。</strong>Gemma 4 的官方接口把 thinking 与 final 分成专用 channel，模型也经过“给出有帮助的完整最终答复”的 post-training。我们的 prompt 要求 final 只有一行，但 12B 经常在 final channel 再总结/列出证据；这不是 parser 泄漏 thought channel，而是模型真实生成了冗长 final。较大的 12B 比 E4B 更稳定地找对记录，却更稳定地违反极窄的 final contract，因此严格准确率反而更低。Native reasoning 还会多次重扫 passage（12B restart signal 144/500），增加长度与漂移风险；Index 则把任务约束成短、确定、可核验的表面程序。</div>
<div class="conclusion"><strong>本小节结论：</strong>Gemma4-12B Thinking 的 31.4% 主要不是“不会数”，而是 final-format adherence 只有 35.4%；若只看 exact count，它接近 Index（90.8% 对 95.0%）。E4B 的差距更像计数与输出控制共同造成，不能全部归因于格式。</div>

<h3>6.3 为什么 DeepSeek-R1 Thinking 与 Qwen3-8B 差距大？</h3>
<details class="report-fold"><summary>展开 Qwen3-8B 与 DeepSeek-R1 的 Native 诊断表</summary><div class="fold-content">{table_html(native_diag_display[native_diag_display['Model'].isin(['Qwen3-8B','DeepSeek-R1-0528-Qwen3-8B'])])}</div></details>
<div class="evidence"><strong>结果与案例。</strong>同一 500 stimuli 上，Qwen3-8B Native 为 95.6%，DeepSeek-R1-0528-Qwen3-8B 为 67.2%，相差 28.4 pp；配对结果为 145 条 Qwen-only success、3 条 DeepSeek-only success、333 条两者都成功。差距随难度放大：N=30 时 84% 对 24%，L=20k 时 86% 对 47%。DeepSeek 有 22 次截断、142 次数值错，中位输出 397.5 tokens、P90 1,908.9、P99 已到 4,096；Qwen 对应 1 次截断、21 次数值错、P90 837.3。restart signal 为 116 对 23。</div>
<div class="hypothesis"><strong>解释。</strong>DeepSeek 模型卡说明该模型把 DeepSeek-R1-0528 的长链推理蒸馏到 Qwen3-8B Base，并强调更深 reasoning；它不是“原 Qwen3-8B 加同一句 prompt”，而是后训练行为已改变。城市记录计数主要是长上下文检索与单调累加，不需要数学证明。R1 风格的反复核验、重启和长 CoT 在此任务上增加了重复计数与遗漏机会；官方卡建议的最长生成预算远高于本实验 4,096，因此其长推理先验又与冻结预算冲突。由于两者骨干规模接近而行为差异巨大，当前证据更支持“post-training/推理策略不匹配”，而不是参数量不足。</div>
<div class="conclusion"><strong>本小节结论：</strong>DeepSeek 的主要问题是长、反复的 reasoning 对串行检索任务产生负迁移；差距同时来自更多截断和更多非截断计数漂移，不能只用 max tokens 解释。</div>

<h3>6.4 为什么 GLM Thinking 看起来不如 Enumeration？</h3>
<details class="report-fold"><summary>展开 GLM-Z1 Native 的长度、格式与重启诊断表</summary><div class="fold-content">{table_html(native_diag_display[native_diag_display['Model'].isin(['GLM-Z1-9B-0414'])])}</div></details>
<div class="evidence"><strong>先校正现象。</strong>GLM-Z1 Native 为 68.4%，只比 matched GLM-4 Index 的 70.4% 低 2.0 pp，却比 Bullet 的 62.4% 高 6.0 pp，所以“Thinking 一定不如 Enumeration”并不成立。差异主要出现在长 passage：L=2k 时 Z1/Index 为 96%/95%，L=20k 时为 39%/48%。Z1 有 41 次截断（8.2%），Index 只有 1 次；Z1 在未截断子集中的 success 是 342/459=74.5%（仅作描述，不能当作随机删失后的无偏估计）。</div>
<div class="hypothesis"><strong>解释。</strong>GLM-Z1 官方说明其 reasoning 版本通过 cold start 与扩展 RL 强化数学、代码、逻辑，并由 chat template 强制进入 <code>&lt;think&gt;</code>；模型卡建议的生成上限是 30,000 tokens。本实验固定 4,096，使其 P90 已达 3,053.4、P99 达上限。Index 的短表面程序避免了 reasoning 展开，因此在长 passage 上更稳。另一个限制是 Z1 与 GLM-4 是 matched family control 而非同一权重的开/关 thinking 消融，2 pp 差异不能纯归因于“思考”本身。</div>
<div class="conclusion"><strong>本小节结论：</strong>GLM-Z1 并非整体弱于枚举；它与 Index 接近、优于 Bullet。当前较清晰的劣势是 forced thinking × 长上下文 × 4,096-token 预算造成的尾部截断。</div></section>

<section id="gemma"><h2>7. Gemma4-12B Direct 的受控 Prompt 干预</h2>
<details class="report-fold"><summary>展开图 5：旧 Direct 与 Strict Direct 的门槛对照</summary><div class="fold-content">
<figure><img src="figures/prompt/05_gemma_direct_prompt_intervention.png" alt="Gemma direct prompt comparison"><figcaption><strong>图 5.</strong> 同一 500-stimulus 设计中，旧 Direct 与 Strict Direct 的各门槛比例。Truncated 越低越好，其余四项越高越好。</figcaption></figure>
</div></details>
<p>旧 prompt 的 500/500 输出都在 64 tokens 截断，且没有可解析的 Total；典型输出先解释如何扫描，再开始编号列举。Strict Direct 保持数据、模型和 64-token预算不变，只明确禁止解释、公开推理、引用和列举：截断降至 0，解析率与格式率升至 100%，registered success 升至 {100*metrics['gemma12_strict']:.1f}%。剩余失败均为可解析但数值错误。</p>
<div class="conclusion"><strong>本节结论：</strong>Gemma4-12B 原始 Direct 的 0% 是 prompt × 输出预算造成的完全输出控制失败，不支持“模型完全听不懂任务”。严格 prompt 修复了输出通道，但其计数能力在该模式下仍只有约一半请求成功。</div></section>

<section id="sources"><h2>8. 外部证据与可复核边界</h2>
<p>模型训练/接口信息只取官方模型卡，用于约束解释范围；输出统计与案例全部来自本实验冻结 JSONL。外部材料不能替代本实验的行为证据。</p>
<ul class="source-list">
<li><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B official model card</a>：说明 thinking/non-thinking 切换、Qwen3 架构摘要及 thinking 推荐采样参数。</li>
<li><a href="https://huggingface.co/google/gemma-4-12B-it/blob/main/README.md">Gemma 4 official README</a>：说明 <code>enable_thinking</code>、thought/final channel 解析，以及 temperature=1、top-p=0.95、top-k=64 的推荐配置。</li>
<li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B">DeepSeek-R1-0528-Qwen3-8B official model card</a>：说明从 DeepSeek-R1-0528 向 Qwen3-8B Base 蒸馏 reasoning，以及更长的建议生成预算。</li>
<li><a href="https://huggingface.co/zai-org/GLM-Z1-9B-0414/blob/main/README.md">GLM-Z1-9B-0414 official README</a>：说明 cold start + extended RL 的 reasoning 后训练、强制 thinking 模板与 30,000-token 建议上限。</li>
</ul>
<div class="conclusion"><strong>本节结论：</strong>“序号充当外部计数器”“深推理后训练对串行检索产生负迁移”是与数据一致的机制解释，不是神经层面的已证因果。可确认事实是输出长度、重启、格式形态、配对成功差及其随 N/L 的变化。</div></section>
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
    metric_keys = [
        "seed_cv_mse",
        "seed_cv_rmse",
        "seed_cv_mae",
        "seed_cv_se",
        "request_oof_r2",
        "request_oof_rmse",
        "request_oof_mae",
        "request_oof_median_ae",
        "request_oof_nrmse_sd",
        "cell_crossfit_r2",
        "cell_crossfit_rmse",
        "cell_crossfit_mae",
        "cell_crossfit_median_ae",
        "cell_crossfit_nrmse_sd",
        "cell_fit_r2",
        "cell_adjusted_r2",
        "cell_fit_rmse",
        "cell_fit_mae",
        "cell_fit_median_ae",
        "cell_fit_nrmse_sd",
        "cell_sse_deviance",
        "cell_log_likelihood",
        "cell_aic",
        "cell_aicc",
        "cell_bic",
        "leave_N_out_r2",
        "leave_N_out_rmse",
        "leave_N_out_mae",
        "leave_L_out_r2",
        "leave_L_out_rmse",
        "leave_L_out_mae",
        "residual_spearman_N",
        "residual_spearman_L",
        "abs_residual_spearman_fitted",
        "max_abs_cell_residual",
    ]
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
                        **{key: cv[key] for key in metric_keys},
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
        slope_sign_stability = []
        if len(draws) and candidate.k > 1:
            for index in range(1, candidate.k):
                if abs(beta[index]) <= 1e-12:
                    continue
                slope_sign_stability.append(
                    float(
                        np.mean(
                            np.sign(draws[:, index])
                            == np.sign(beta[index])
                        )
                    )
                )
        minimum_sign_stability = (
            min(slope_sign_stability) if slope_sign_stability else math.nan
        )
        slopes_ci_excluding_zero = int(
            np.sum(
                (ci_low[1:] > 0) | (ci_high[1:] < 0)
            )
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
                "bootstrap_valid_reps": len(draws),
                "minimum_slope_sign_stability": minimum_sign_stability,
                "slopes_ci_excluding_zero": slopes_ci_excluding_zero,
                "slope_terms": max(candidate.k - 1, 0),
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
        oof_frame["oof_residual"] = oof_frame[target] - oof
        oof_frame["oof_absolute_residual"] = np.abs(oof_frame["oof_residual"])
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


def fit_relative_log_ratio_laws(
    data: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """Fit b/N = alpha + beta ln(N/L_k) in every model-by-mode cell."""
    working = data.copy()
    working["relative_signed_error"] = working["signed_error"] / working["N"]
    candidate = candidate_by_name("ln_density")
    rows: list[dict] = []
    for (model, mode), group in working[
        working["relative_signed_error"].notna()
    ].groupby(["model", "mode"]):
        group = group.copy()
        cv = candidate_cv(group, candidate, "relative_signed_error")
        cells = group.groupby(["N", "L"], as_index=False).relative_signed_error.mean()
        beta, _ = fit_raw(candidate, cells, "relative_signed_error")
        draws = bootstrap_coefficients(
            candidate, group, "relative_signed_error", repetitions=400
        )
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
        cell_r2 = float(cv["cell_crossfit_r2"])
        quality = (
            "strong"
            if cell_r2 >= 0.8
            else "moderate"
            if cell_r2 >= 0.5
            else "weak"
        )
        rows.append(
            {
                "model": model,
                "mode": mode,
                "candidate": candidate.name,
                "label": candidate.label,
                "formula": r"\overline{b/N}=\alpha+\beta\ln(N/L_k)",
                **{key: value for key, value in cv.items() if key != "oof"},
                "intercept": float(beta[0]),
                "slope": float(beta[1]),
                "intercept_ci_low": float(ci_low[0]),
                "intercept_ci_high": float(ci_high[0]),
                "slope_ci_low": float(ci_low[1]),
                "slope_ci_high": float(ci_high[1]),
                "overall_f_p": overall_f_test(
                    candidate, cells, "relative_signed_error"
                ),
                "quality": quality,
                "numeric_formula": numeric_formula(
                    candidate, beta, r"\overline{b/N}"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_q"] = bh_adjust(result["overall_f_p"])
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    mode_rank = {name: index for index, name in enumerate(MODE_ORDER)}
    result["_model_rank"] = result["model"].map(model_rank)
    result["_mode_rank"] = result["mode"].map(mode_rank)
    result = (
        result.sort_values(["_model_rank", "_mode_rank"])
        .drop(columns=["_model_rank", "_mode_rank"])
        .reset_index(drop=True)
    )
    result.to_csv(
        output / "tables" / "bias" / "relative_log_ratio_laws.csv",
        index=False,
    )
    return result


def plot_model_bias(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    relative_log_ratio: pd.DataFrame,
    model: str,
    path: Path,
) -> None:
    modes = [m for m in MODE_ORDER if ((data.model == model) & (data["mode"] == m)).any()]
    fig, axes = plt.subplots(
        len(modes),
        3,
        figsize=(16.8, max(3.2, 2.75 * len(modes))),
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
        relative_ax = axes[row_index, 2]
        relative_group = group.assign(
            relative_signed_error=group["signed_error"] / group["N"],
            ln_density=np.log(group["N"] / (group["L"] / 1000.0)),
        )
        relative_cells = (
            relative_group.groupby(["N", "L"], as_index=False)
            .agg(
                observed=("relative_signed_error", "mean"),
                ln_density=("ln_density", "first"),
            )
            .sort_values("ln_density")
        )
        relative_record = relative_log_ratio[
            (relative_log_ratio.model == model)
            & (relative_log_ratio["mode"] == mode)
        ].iloc[0]
        relative_ax.scatter(
            relative_cells["ln_density"],
            relative_cells["observed"],
            s=22,
            alpha=0.78,
            color=MODE_COLORS[mode],
            label="Observed (N, L) cell mean",
        )
        ratio_grid = np.linspace(
            relative_cells["ln_density"].min(),
            relative_cells["ln_density"].max(),
            220,
        )
        ratio_prediction = (
            float(relative_record.intercept)
            + float(relative_record.slope) * ratio_grid
        )
        relative_ax.plot(
            ratio_grid,
            ratio_prediction,
            color="#17212B",
            linewidth=2,
            label="Prespecified log-linear fit",
        )
        relative_ax.axhline(0, color="#6D7882", linewidth=1, linestyle="--")
        relative_ax.set_xlabel(r"Log needle density $\ln(N/L_k)$")
        relative_ax.set_ylabel(r"Mean relative bias $b/N$")
        relative_ax.set_title(
            f"{MODE_LABEL[mode]} · b/N ~ ln(N/Lk) · CV cell R²="
            f"{relative_record.cell_crossfit_r2:.2f}"
        )
        if row_index == 0:
            relative_ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{model}: bias response and selected low-complexity law", y=0.992, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
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


def plot_relative_log_ratio_quality(
    relative_log_ratio: pd.DataFrame,
    path: Path,
) -> None:
    matrix = (
        relative_log_ratio.pivot(
            index="model", columns="mode", values="cell_crossfit_r2"
        )
        .reindex(index=MODEL_ORDER, columns=MODE_ORDER)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    ax.set_xticks(range(4), [MODE_LABEL[m] for m in MODE_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.55 else "#111",
                    fontweight="bold",
                    fontsize=8.5,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Seed-cross-fitted cell R²")
    ax.set_title(r"Fixed log-linear law: mean $b/N$ versus $\ln(N/L_k)$")
    ax.set_xlabel("Prompt mode")
    ax.set_ylabel("Model")
    fig.tight_layout()
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
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
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


def fit_universal_mode_laws(
    data: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    output: Path,
) -> dict[str, pd.DataFrame]:
    """Find one shared functional form per mode family with cell-specific parameters."""
    tables = output / "tables" / "bias"
    candidate_rows: list[pd.DataFrame] = []
    for family, modes in LAW_MODE_FAMILIES.items():
        subset = candidate_frame[
            candidate_frame["mode"].isin(modes)
            & candidate_frame["target"].isin(["signed_error", "absolute_error"])
        ].copy()
        subset["family"] = family
        candidate_rows.append(subset)
    family_candidates = pd.concat(candidate_rows, ignore_index=True)

    summary = (
        family_candidates.groupby(
            ["family", "target", "candidate", "label", "formula", "k"],
            as_index=False,
        )
        .agg(
            cells=("model", "size"),
            moderate_cells=(
                "cell_crossfit_r2",
                lambda values: int((values >= 0.5).sum()),
            ),
            strong_cells=(
                "cell_crossfit_r2",
                lambda values: int((values >= 0.8).sum()),
            ),
            median_cell_cv_r2=("cell_crossfit_r2", "median"),
            mean_cell_cv_r2=("cell_crossfit_r2", "mean"),
            q25_cell_cv_r2=(
                "cell_crossfit_r2",
                lambda values: float(values.quantile(0.25)),
            ),
            min_cell_cv_r2=("cell_crossfit_r2", "min"),
            median_cell_cv_rmse=("cell_crossfit_rmse", "median"),
            median_cell_cv_mae=("cell_crossfit_mae", "median"),
            median_cell_cv_median_ae=("cell_crossfit_median_ae", "median"),
            median_cell_cv_nrmse_sd=("cell_crossfit_nrmse_sd", "median"),
            median_request_oof_r2=("request_oof_r2", "median"),
            median_request_oof_rmse=("request_oof_rmse", "median"),
            median_request_oof_mae=("request_oof_mae", "median"),
            median_leave_N_out_r2=("leave_N_out_r2", "median"),
            median_leave_L_out_r2=("leave_L_out_r2", "median"),
            min_leave_N_out_r2=("leave_N_out_r2", "min"),
            min_leave_L_out_r2=("leave_L_out_r2", "min"),
        )
    )
    summary["moderate_fraction"] = summary["moderate_cells"] / summary["cells"]
    summary["strong_fraction"] = summary["strong_cells"] / summary["cells"]
    summary["selected"] = False

    selected_records: list[pd.Series] = []
    for (family, target), group in summary.groupby(["family", "target"]):
        # Robust, pre-specified hierarchy: maximize broad coverage first. Treat
        # median CV R² values within 0.02 as practically tied, then prefer the
        # larger lower quartile. Remaining ties favor extrapolation and parsimony.
        eligible = group[
            group["moderate_cells"] == group["moderate_cells"].max()
        ].copy()
        best_median = float(eligible["median_cell_cv_r2"].max())
        eligible = eligible[
            eligible["median_cell_cv_r2"] >= best_median - 0.02
        ].copy()
        eligible["extrapolation_floor"] = eligible[
            ["median_leave_N_out_r2", "median_leave_L_out_r2"]
        ].min(axis=1)
        selected = eligible.sort_values(
            [
                "q25_cell_cv_r2",
                "extrapolation_floor",
                "median_cell_cv_r2",
                "k",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]
        summary.loc[selected.name, "selected"] = True
        selected_records.append(selected)
    selected_summary = pd.DataFrame(selected_records).drop(
        columns=["extrapolation_floor"],
        errors="ignore",
    )

    parsed = data[data["signed_error"].notna()].copy()
    per_cell_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for _, selected in selected_summary.iterrows():
        family = str(selected.family)
        target = str(selected.target)
        candidate = candidate_by_name(str(selected.candidate))
        modes = LAW_MODE_FAMILIES[family]
        family_data = parsed[parsed["mode"].isin(modes)]
        for (model, mode), group in family_data.groupby(["model", "mode"]):
            group = group.copy()
            cv = candidate_cv(group, candidate, target)
            cells = group.groupby(["N", "L"], as_index=False)[target].mean()
            beta, _ = fit_raw(candidate, cells, target)
            draws = bootstrap_coefficients(
                candidate,
                group,
                target,
                repetitions=400,
            )
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
            slope_stability = []
            for index in range(1, candidate.k):
                stability = (
                    float(
                        np.mean(
                            np.sign(draws[:, index])
                            == np.sign(beta[index])
                        )
                    )
                    if len(draws) and abs(beta[index]) > 1e-12
                    else math.nan
                )
                if np.isfinite(stability):
                    slope_stability.append(stability)
            matched = family_candidates[
                (family_candidates["family"] == family)
                & (family_candidates["target"] == target)
                & (family_candidates["candidate"] == candidate.name)
                & (family_candidates["model"] == model)
                & (family_candidates["mode"] == mode)
            ].iloc[0]
            per_cell_rows.append(
                {
                    **matched.to_dict(),
                    "bootstrap_valid_reps": len(draws),
                    "minimum_slope_sign_stability": (
                        min(slope_stability) if slope_stability else math.nan
                    ),
                    "slopes_ci_excluding_zero": int(
                        np.sum((ci_low[1:] > 0) | (ci_high[1:] < 0))
                    ),
                    "slope_terms": candidate.k - 1,
                    "numeric_formula": numeric_formula(
                        candidate,
                        beta,
                        r"\bar b"
                        if target == "signed_error"
                        else r"\overline{|b|}",
                    ),
                }
            )
            names = ("intercept",) + candidate.feature_names
            for index, name in enumerate(names):
                coefficient_rows.append(
                    {
                        "family": family,
                        "model": model,
                        "mode": mode,
                        "target": target,
                        "candidate": candidate.name,
                        "term": name,
                        "estimate": beta[index],
                        "bootstrap_ci_low": ci_low[index],
                        "bootstrap_ci_high": ci_high[index],
                        "bootstrap_sign_stability": (
                            float(
                                np.mean(
                                    np.sign(draws[:, index])
                                    == np.sign(beta[index])
                                )
                            )
                            if len(draws) and abs(beta[index]) > 1e-12
                            else math.nan
                        ),
                    }
                )
            cell_predictions = (
                group.assign(oof_prediction=cv["oof"])
                .groupby(["N", "L"], as_index=False)
                .agg(
                    observed=(target, "mean"),
                    prediction=("oof_prediction", "mean"),
                    parsed_n=(target, "size"),
                )
            )
            cell_predictions.insert(0, "target", target)
            cell_predictions.insert(0, "mode", mode)
            cell_predictions.insert(0, "model", model)
            cell_predictions.insert(0, "family", family)
            cell_predictions["candidate"] = candidate.name
            cell_predictions["residual"] = (
                cell_predictions["observed"]
                - cell_predictions["prediction"]
            )
            prediction_rows.append(cell_predictions)

    per_cell = pd.DataFrame(per_cell_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    stability = (
        per_cell.groupby(["family", "target"], as_index=False)
        .agg(
            median_minimum_slope_sign_stability=(
                "minimum_slope_sign_stability",
                "median",
            ),
            min_minimum_slope_sign_stability=(
                "minimum_slope_sign_stability",
                "min",
            ),
        )
    )
    selected_summary = selected_summary.merge(
        stability,
        on=["family", "target"],
        how="left",
    )

    summary.to_csv(tables / "universal_mode_candidate_comparison.csv", index=False)
    selected_summary.to_csv(tables / "universal_mode_selected_laws.csv", index=False)
    per_cell.to_csv(tables / "universal_mode_per_cell_metrics.csv", index=False)
    coefficients.to_csv(tables / "universal_mode_coefficients.csv", index=False)
    predictions.to_csv(tables / "universal_mode_cell_predictions.csv", index=False)
    return {
        "candidate_summary": summary,
        "selected": selected_summary,
        "per_cell": per_cell,
        "coefficients": coefficients,
        "predictions": predictions,
    }


def plot_universal_mode_quality(
    per_cell: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 8.8),
        gridspec_kw={"width_ratios": [0.85, 1.25]},
    )
    target_order = ["signed_error", "absolute_error"]
    target_labels = ["Signed bias", "Absolute deviation"]
    image = None
    for ax, family in zip(axes, LAW_MODE_FAMILIES):
        subset = per_cell[per_cell["family"] == family].copy()
        subset["cell_label"] = subset.apply(
            lambda row: (
                str(row.model)
                if family == "Direct"
                else f"{row.model} · {MODE_LABEL[str(row['mode'])]}"
            ),
            axis=1,
        )
        labels = list(dict.fromkeys(subset["cell_label"].tolist()))
        matrix = (
            subset.pivot(
                index="cell_label",
                columns="target",
                values="cell_crossfit_r2",
            )
            .reindex(index=labels, columns=target_order)
        )
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            cmap="RdYlGn",
            vmin=-0.2,
            vmax=1.0,
        )
        ax.set_xticks(range(len(target_order)), target_labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{family}: shared form, cell-specific parameters")
        for row in range(len(labels)):
            for column in range(len(target_order)):
                value = matrix.iloc[row, column]
                if np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    colorbar_axis = fig.add_axes([0.92, 0.15, 0.018, 0.68])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Seed-cross-fitted cell R²")
    fig.suptitle(
        "Direct and Enumerate: coverage of the selected universal functional forms",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.08, top=0.92, wspace=0.62)
    save_figure(fig, path)


def plot_universal_mode_predictions(
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2), squeeze=False)
    colors = {
        model: plt.get_cmap("tab10")(index % 10)
        for index, model in enumerate(MODEL_ORDER)
    }
    markers = {
        "direct": "o",
        "enumeration_index": "s",
        "enumeration_bullet": "^",
    }
    for row, family in enumerate(["Direct", "Enumerate"]):
        for column, target in enumerate(["signed_error", "absolute_error"]):
            ax = axes[row, column]
            subset = predictions[
                (predictions["family"] == family)
                & (predictions["target"] == target)
            ]
            for (model, mode), group in subset.groupby(["model", "mode"]):
                ax.scatter(
                    group["observed"],
                    group["prediction"],
                    s=18,
                    alpha=0.55,
                    color=colors[str(model)],
                    marker=markers[str(mode)],
                    edgecolors="none",
                )
            finite = subset[["observed", "prediction"]].to_numpy(dtype=float)
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))
            padding = max((high - low) * 0.05, 0.25)
            ax.plot(
                [low - padding, high + padding],
                [low - padding, high + padding],
                color="#555555",
                linewidth=1,
                linestyle="--",
            )
            ax.set_xlim(low - padding, high + padding)
            ax.set_ylim(low - padding, high + padding)
            record = selected[
                (selected["family"] == family)
                & (selected["target"] == target)
            ].iloc[0]
            ax.set_title(
                f"{family} · {'signed bias' if target == 'signed_error' else '|bias|'}\n"
                f"{record['label']} · median CV R²={record.median_cell_cv_r2:.2f}"
            )
            ax.set_xlabel("Observed condition mean")
            ax.set_ylabel("Seed-cross-fitted prediction")
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=colors[model],
            label=model,
            markersize=6,
        )
        for model in MODEL_ORDER
        if model in set(predictions["model"])
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[mode],
            linestyle="",
            color="#555555",
            label=MODE_LABEL[mode],
            markersize=6,
        )
        for mode in ["enumeration_index", "enumeration_bullet"]
    ]
    fig.legend(
        handles=model_handles + mode_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Observed versus held-out-seed predictions for the universal forms",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    save_figure(fig, path)


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
    relative_log_ratio = fit_relative_log_ratio_laws(data, output)
    common = common_bilinear_table(data)
    universal = fit_universal_mode_laws(data, fit["candidates"], output)
    common.to_csv(tables / "common_bilinear_diagnostics.csv", index=False)
    plot_bias_quality(selected, figures / "01_selected_law_quality.png")
    plot_relative_log_ratio_quality(
        relative_log_ratio,
        figures / "04_relative_log_ratio_quality.png",
    )
    for model in MODEL_ORDER:
        safe = (
            model.lower()
            .replace(".", "")
            .replace("/", "-")
            .replace(" ", "-")
        )
        plot_model_bias(
            data,
            selected,
            relative_log_ratio,
            model,
            figures / f"model_{safe}.png",
        )
    plot_native_commonality(common, figures / "02_native_commonality.png")
    plot_relative_fallbacks(data, fallback, figures / "03_relative_bias_fallbacks.png")
    plot_universal_mode_quality(
        universal["per_cell"],
        figures / "05_universal_mode_quality.png",
    )
    plot_universal_mode_predictions(
        universal["predictions"],
        universal["selected"],
        figures / "06_universal_mode_predictions.png",
    )

    signed = signed_selected
    signed["Model"] = signed["model"]
    signed["Mode"] = signed["mode"].map(MODE_LABEL)
    signed["Selected form"] = signed["label"]
    signed["CV cell R²"] = signed["cell_crossfit_r2"].map(lambda x: format_number(x, 3))
    signed["CV cell RMSE"] = signed["cell_crossfit_rmse"].map(
        lambda x: format_number(x, 3)
    )
    signed["CV cell MAE"] = signed["cell_crossfit_mae"].map(
        lambda x: format_number(x, 3)
    )
    signed["CV cell median AE"] = signed["cell_crossfit_median_ae"].map(
        lambda x: format_number(x, 3)
    )
    signed["CV cell NRMSE/SD"] = signed["cell_crossfit_nrmse_sd"].map(
        lambda x: format_number(x, 3)
    )
    signed["Request OOF R²"] = signed["request_oof_r2"].map(lambda x: format_number(x, 3))
    signed["Request OOF RMSE"] = signed["request_oof_rmse"].map(
        lambda x: format_number(x, 3)
    )
    signed["Request OOF MAE"] = signed["request_oof_mae"].map(
        lambda x: format_number(x, 3)
    )
    signed["Leave-N-out R²"] = signed["leave_N_out_r2"].map(lambda x: format_number(x, 3))
    signed["Leave-L-out R²"] = signed["leave_L_out_r2"].map(lambda x: format_number(x, 3))
    signed["Adjusted fit R²"] = signed["cell_adjusted_r2"].map(
        lambda x: format_number(x, 3)
    )
    signed["Cell AICc"] = signed["cell_aicc"].map(lambda x: format_number(x, 2))
    signed["Cell BIC"] = signed["cell_bic"].map(lambda x: format_number(x, 2))
    signed["Min slope sign stability"] = signed[
        "minimum_slope_sign_stability"
    ].map(lambda x: format_number(x, 3))
    signed["Slopes CI≠0"] = signed.apply(
        lambda row: f"{int(row.slopes_ci_excluding_zero)}/{int(row.slope_terms)}",
        axis=1,
    )
    signed["Residual max |ρ(N,L)|"] = signed.apply(
        lambda row: format_number(
            max_abs_finite(
                row.residual_spearman_N,
                row.residual_spearman_L,
            ),
            3,
        ),
        axis=1,
    )
    signed["|residual|–fitted ρ"] = signed[
        "abs_residual_spearman_fitted"
    ].map(lambda x: format_number(x, 3))
    signed["FDR q"] = signed["fdr_q"].map(lambda x: f"{x:.2e}" if np.isfinite(x) else "—")
    signed["Fit class"] = signed["quality"].str.title()
    display_signed = signed[
        [
            "Model",
            "Mode",
            "Selected form",
            "CV cell R²",
            "CV cell RMSE",
            "CV cell MAE",
            "CV cell median AE",
            "CV cell NRMSE/SD",
            "Request OOF R²",
            "Request OOF RMSE",
            "Request OOF MAE",
            "Leave-N-out R²",
            "Leave-L-out R²",
            "Fit class",
        ]
    ]
    diagnostic_signed = signed[
        [
            "Model",
            "Mode",
            "Selected form",
            "Adjusted fit R²",
            "Cell AICc",
            "Cell BIC",
            "Min slope sign stability",
            "Slopes CI≠0",
            "Residual max |ρ(N,L)|",
            "|residual|–fitted ρ",
            "FDR q",
        ]
    ]
    quality_counts = signed["quality"].value_counts()
    strong = int(quality_counts.get("strong", 0))
    moderate = int(quality_counts.get("moderate", 0))
    weak = int(quality_counts.get("weak", 0))

    abs_selected = selected[selected.target == "absolute_error"].copy()
    abs_display = abs_selected[
        [
            "model",
            "mode",
            "label",
            "cell_crossfit_r2",
            "cell_crossfit_rmse",
            "cell_crossfit_mae",
            "cell_crossfit_median_ae",
            "cell_crossfit_nrmse_sd",
            "request_oof_r2",
            "leave_N_out_r2",
            "leave_L_out_r2",
            "cell_adjusted_r2",
            "cell_aicc",
            "cell_bic",
            "minimum_slope_sign_stability",
            "quality",
        ]
    ].copy()
    abs_display.columns = [
        "Model",
        "Mode",
        "Selected |bias| form",
        "CV cell R²",
        "CV cell RMSE",
        "CV cell MAE",
        "CV cell median AE",
        "CV cell NRMSE/SD",
        "Request OOF R²",
        "Leave-N-out R²",
        "Leave-L-out R²",
        "Adjusted fit R²",
        "Cell AICc",
        "Cell BIC",
        "Min slope sign stability",
        "Fit class",
    ]
    abs_display["Mode"] = abs_display["Mode"].map(MODE_LABEL)
    for column in [
        "CV cell R²",
        "CV cell RMSE",
        "CV cell MAE",
        "CV cell median AE",
        "CV cell NRMSE/SD",
        "Request OOF R²",
        "Leave-N-out R²",
        "Leave-L-out R²",
        "Adjusted fit R²",
        "Min slope sign stability",
    ]:
        abs_display[column] = abs_display[column].map(
            lambda x: format_number(x, 3)
        )
    for column in ["Cell AICc", "Cell BIC"]:
        abs_display[column] = abs_display[column].map(
            lambda x: format_number(x, 2)
        )

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

    log_ratio_display = relative_log_ratio.copy()
    log_ratio_display["Model"] = log_ratio_display["model"]
    log_ratio_display["Mode"] = log_ratio_display["mode"].map(MODE_LABEL)
    log_ratio_display["Formula"] = log_ratio_display["numeric_formula"].map(
        lambda value: f"\\({value}\\)"
    )
    log_ratio_display["CV cell R²"] = log_ratio_display[
        "cell_crossfit_r2"
    ].map(lambda value: format_number(value, 3))
    log_ratio_display["Request OOF R²"] = log_ratio_display[
        "request_oof_r2"
    ].map(lambda value: format_number(value, 3))
    log_ratio_display["Leave-N-out R²"] = log_ratio_display[
        "leave_N_out_r2"
    ].map(lambda value: format_number(value, 3))
    log_ratio_display["Leave-L-out R²"] = log_ratio_display[
        "leave_L_out_r2"
    ].map(lambda value: format_number(value, 3))
    log_ratio_display["Slope 95% CI"] = log_ratio_display.apply(
        lambda row: (
            f"{row.slope:.3g} "
            f"[{row.slope_ci_low:.3g}, {row.slope_ci_high:.3g}]"
        ),
        axis=1,
    )
    log_ratio_display["FDR q"] = log_ratio_display["fdr_q"].map(
        lambda value: f"{value:.2e}" if np.isfinite(value) else "—"
    )
    log_ratio_display["Fit class"] = log_ratio_display["quality"].str.title()
    log_ratio_display = log_ratio_display[
        [
            "Model",
            "Mode",
            "Formula",
            "CV cell R²",
            "Request OOF R²",
            "Leave-N-out R²",
            "Leave-L-out R²",
            "Slope 95% CI",
            "FDR q",
            "Fit class",
        ]
    ]
    log_ratio_quality_counts = relative_log_ratio["quality"].value_counts()
    log_ratio_strong = int(log_ratio_quality_counts.get("strong", 0))
    log_ratio_moderate = int(log_ratio_quality_counts.get("moderate", 0))
    log_ratio_weak = int(log_ratio_quality_counts.get("weak", 0))

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

    universal_selected = universal["selected"].copy()
    universal_selected["Family"] = universal_selected["family"]
    universal_selected["Target"] = universal_selected["target"].map(
        {
            "signed_error": "Mean signed bias",
            "absolute_error": "Mean absolute deviation",
        }
    )
    universal_selected["Shared form"] = universal_selected["label"]
    universal_selected["Coverage R²≥0.5"] = universal_selected.apply(
        lambda row: f"{int(row.moderate_cells)}/{int(row.cells)}",
        axis=1,
    )
    universal_selected["Strong R²≥0.8"] = universal_selected.apply(
        lambda row: f"{int(row.strong_cells)}/{int(row.cells)}",
        axis=1,
    )
    universal_selected["Median CV R²"] = universal_selected[
        "median_cell_cv_r2"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Q25 CV R²"] = universal_selected[
        "q25_cell_cv_r2"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Minimum CV R²"] = universal_selected[
        "min_cell_cv_r2"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Median CV RMSE"] = universal_selected[
        "median_cell_cv_rmse"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Median CV MAE"] = universal_selected[
        "median_cell_cv_mae"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Median CV median AE"] = universal_selected[
        "median_cell_cv_median_ae"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Median CV NRMSE/SD"] = universal_selected[
        "median_cell_cv_nrmse_sd"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Median leave-N R²"] = universal_selected[
        "median_leave_N_out_r2"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Median leave-L R²"] = universal_selected[
        "median_leave_L_out_r2"
    ].map(lambda value: format_number(value, 3))
    universal_selected["Median min sign stability"] = universal_selected[
        "median_minimum_slope_sign_stability"
    ].map(lambda value: format_number(value, 3))
    universal_selected_display = universal_selected[
        [
            "Family",
            "Target",
            "Shared form",
            "Coverage R²≥0.5",
            "Strong R²≥0.8",
            "Median CV R²",
            "Q25 CV R²",
            "Minimum CV R²",
            "Median CV RMSE",
            "Median CV MAE",
            "Median CV median AE",
            "Median CV NRMSE/SD",
            "Median leave-N R²",
            "Median leave-L R²",
            "Median min sign stability",
        ]
    ]

    universal_per_cell = universal["per_cell"].copy()
    universal_per_cell["Family"] = universal_per_cell["family"]
    universal_per_cell["Model"] = universal_per_cell["model"]
    universal_per_cell["Mode"] = universal_per_cell["mode"].map(MODE_LABEL)
    universal_per_cell["Target"] = universal_per_cell["target"].map(
        {
            "signed_error": "Signed bias",
            "absolute_error": "Absolute deviation",
        }
    )
    universal_per_cell["CV R²"] = universal_per_cell[
        "cell_crossfit_r2"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["CV RMSE"] = universal_per_cell[
        "cell_crossfit_rmse"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["CV MAE"] = universal_per_cell[
        "cell_crossfit_mae"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["CV median AE"] = universal_per_cell[
        "cell_crossfit_median_ae"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["CV NRMSE/SD"] = universal_per_cell[
        "cell_crossfit_nrmse_sd"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["Leave-N R²"] = universal_per_cell[
        "leave_N_out_r2"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["Leave-L R²"] = universal_per_cell[
        "leave_L_out_r2"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["Adjusted fit R²"] = universal_per_cell[
        "cell_adjusted_r2"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell["AICc"] = universal_per_cell["cell_aicc"].map(
        lambda value: format_number(value, 2)
    )
    universal_per_cell["BIC"] = universal_per_cell["cell_bic"].map(
        lambda value: format_number(value, 2)
    )
    universal_per_cell["Min sign stability"] = universal_per_cell[
        "minimum_slope_sign_stability"
    ].map(lambda value: format_number(value, 3))
    universal_per_cell_display = universal_per_cell[
        [
            "Family",
            "Model",
            "Mode",
            "Target",
            "CV R²",
            "CV RMSE",
            "CV MAE",
            "CV median AE",
            "CV NRMSE/SD",
            "Leave-N R²",
            "Leave-L R²",
            "Adjusted fit R²",
            "AICc",
            "BIC",
            "Min sign stability",
        ]
    ]

    universal_comparison = universal["candidate_summary"].copy()
    universal_comparison = universal_comparison.sort_values(
        ["family", "target", "moderate_cells", "median_cell_cv_r2"],
        ascending=[True, True, False, False],
    )
    universal_top = universal_comparison.groupby(
        ["family", "target"],
        as_index=False,
        group_keys=False,
    ).head(6).copy()
    universal_top["Family"] = universal_top["family"]
    universal_top["Target"] = universal_top["target"].map(
        {
            "signed_error": "Signed bias",
            "absolute_error": "Absolute deviation",
        }
    )
    universal_top["Candidate"] = universal_top["label"]
    universal_top["Coverage"] = universal_top.apply(
        lambda row: f"{int(row.moderate_cells)}/{int(row.cells)}",
        axis=1,
    )
    for source, destination in [
        ("median_cell_cv_r2", "Median CV R²"),
        ("q25_cell_cv_r2", "Q25 CV R²"),
        ("min_cell_cv_r2", "Minimum CV R²"),
        ("median_leave_N_out_r2", "Median leave-N R²"),
        ("median_leave_L_out_r2", "Median leave-L R²"),
    ]:
        universal_top[destination] = universal_top[source].map(
            lambda value: format_number(value, 3)
        )
    universal_top_display = universal_top[
        [
            "Family",
            "Target",
            "Candidate",
            "Coverage",
            "Median CV R²",
            "Q25 CV R²",
            "Minimum CV R²",
            "Median leave-N R²",
            "Median leave-L R²",
        ]
    ]

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
            rf"""<details><summary>{html.escape(str(row.model))} · {html.escape(MODE_LABEL[str(row['mode'])])} · CV cell R²={row.cell_crossfit_r2:.3f}</summary>
            <div class="equation">\\[{row.numeric_formula}\\]</div>
            <p>候选形式：\\({row.formula}\\)。\(L_k=L/1000\)。Seed-cross-fitted cell R²={row.cell_crossfit_r2:.3f}，RMSE={row.cell_crossfit_rmse:.3f}，MAE={row.cell_crossfit_mae:.3f}，median AE={row.cell_crossfit_median_ae:.3f}，NRMSE/SD={row.cell_crossfit_nrmse_sd:.3f}；request OOF R²={row.request_oof_r2:.3f}；leave-N-out R²={row.leave_N_out_r2:.3f}；leave-L-out R²={row.leave_L_out_r2:.3f}。训练内 adjusted R²={row.cell_adjusted_r2:.3f}，AICc={row.cell_aicc:.2f}，BIC={row.cell_bic:.2f}；最弱斜率符号稳定率={row.minimum_slope_sign_stability:.3f}；探索性 FDR q={row.fdr_q:.2e}。</p>
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
            rf"""<details class="report-fold"><summary>{html.escape(model)}：展开各 mode 散点与拟合曲线</summary>
            <div class="fold-content">
            <figure><img src="figures/bias/model_{safe}.png" alt="{html.escape(model)} bias fits"><figcaption><strong>{html.escape(model)}。</strong> 每行对应一个已运行 mode。左列横轴为 \(N\)，中列横轴为 \(L\)（千 tokens），纵轴均为解析样本的 mean signed bias；散点是对另一设计轴和 seeds 边际平均后的观测值，误差棒是 95% CI，实线是所选二维 law 的边际曲线。右列横轴为 \(\ln(N/L_k)\)，纵轴为 mean relative bias \(b/N\)；每个散点是一组 \((N,L)\) 的 seed 均值，实线固定为预先指定的 log-linear law。</figcaption></figure>
            <div class="conclusion"><strong>{html.escape(model)} 结论：</strong>{html.escape(conclusion_text)}。低 R² 不自动等于性能差：若 bias 几乎恒为 0，目标方差过小也会使 R² 不稳定。</div>"""
            "</div></details>"
        )

    direct_signed = universal["selected"][
        (universal["selected"]["family"] == "Direct")
        & (universal["selected"]["target"] == "signed_error")
    ].iloc[0]
    direct_absolute = universal["selected"][
        (universal["selected"]["family"] == "Direct")
        & (universal["selected"]["target"] == "absolute_error")
    ].iloc[0]
    enumerate_signed = universal["selected"][
        (universal["selected"]["family"] == "Enumerate")
        & (universal["selected"]["target"] == "signed_error")
    ].iloc[0]
    enumerate_absolute = universal["selected"][
        (universal["selected"]["family"] == "Enumerate")
        & (universal["selected"]["target"] == "absolute_error")
    ].iloc[0]

    body = rf"""
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
<p>为保持横纵轴和解释简单，候选变量只来自 \(N\)、\(L_k=L/1000\)、自然对数 \(\ln N\)、\(\ln L_k\)、\(N/L_k\)、\(\ln(N/L_k)\)、\(L_k/N\)、\(NL_k\)，以及低阶加法、一个交互项、一个预先固定断点（\(N=10\) 或 \(L_k=5\)）和至多二次项。没有逐点多项式、模型 ID 特征或事后删点。固定底数的对数在带自由斜率的线性模型中只会重标系数，因此本版统一使用自然对数，不重复保留等价的 \(\log_2\) 候选。</p>
<p>每个模型 × mode 单独搜索。每次留出一个 seed，使用其余九个 seeds 拟合并预测留出 seed；候选必须落在最低 seed-CV MSE 的 one-standard-error 集合内，再最大化“cell cross-fitted R² − 每个额外参数 0.015”的简洁性分数。报告同时给出 request-level OOF R²、leave-one-N-level-out 与 leave-one-L-level-out R²。F-test/FDR 仅作为探索性描述，因为公式经过选择，不能替代预注册确认实验。</p>
<p>除 \(R^2\) 外，本版加入四组拟合优度量。对 held-out 预测残差 \(e_i=y_i-\hat y_i\)，报告：</p>
<div class="equation">\\[
\\mathrm{{RMSE}}=\\sqrt{{n^{{-1}}\\sum_i e_i^2}},\\qquad
\\mathrm{{MAE}}=n^{{-1}}\\sum_i|e_i|,\\qquad
\\mathrm{{MedAE}}=\\mathrm{{median}}(|e_i|),\\qquad
\\mathrm{{NRMSE}}_{{SD}}=\\mathrm{{RMSE}}/s_y.
\\]</div>
<p>RMSE 强调少数大误差，MAE 给出平均相差多少个 needle，MedAE 对长尾更稳健，NRMSE/SD 用目标标准差无量纲化；当目标方差近零时 NRMSE 与 R² 都可能不稳定。训练内同时报告 adjusted R²、Gaussian cell-mean likelihood、SSE/deviance、AIC、AICc 与 BIC；AICc/BIC 只允许在<strong>同一模型 × mode、同一响应和同一批样本</strong>的候选间比较，不能跨 bias 与 bias/N 或跨模型直接排序。</p>
<p>稳定性由 seed-block bootstrap 衡量：表中“最弱斜率符号稳定率”是所有非截距系数中，bootstrap 与点估计保持同号比例的最小值；并记录有多少斜率的 95% 区间排除 0。残差诊断报告残差与 \(N,L\) 的最大绝对 Spearman \(|\\rho|\)，以及 \(|e|\) 与 fitted value 的 Spearman \(\rho\)；前者较大提示遗漏曲率，后者较大提示异方差。显著性/FDR 不等同于拟合优度。</p>
<div class="callout warning"><strong>关于“拟合优度要足够高”：</strong>本分析不会通过高阶插值强行让 29 格都得到高 R²。门槛定义为：CV cell R²≥0.8 强、0.5–0.8 中等、&lt;0.5 弱。当前得到 {strong} 个强、{moderate} 个中等、{weak} 个弱 mean-bias law；弱结果被保留为有效负结论。</div>
<div class="conclusion"><strong>本节结论：</strong>选模依赖 held-out seed，而不是训练集曲线漂亮程度；最终判断同时考虑 CV R²、绝对误差、leave-\(N/L\)-out、复杂度、bootstrap 稳定性和残差结构，防止把同网格内复现误称为外推 law。</div></section>

<section id="results"><h2>3. 29 个 Signed-bias Law</h2>
<details class="report-fold"><summary>展开图 1：29 个所选 signed-bias law 的拟合优度</summary><div class="fold-content">
<figure><img src="figures/bias/01_selected_law_quality.png" alt="Selected bias law quality"><figcaption><strong>图 1.</strong> 横轴为 mode，纵轴为模型；每格是所选 signed-bias law 的 seed-cross-fitted cell R²。颜色和数值衡量 50 个 \((N,L)\) 条件均值是否被简单曲面解释，而不是单条请求能否被准确预测。</figcaption></figure>
</div></details>
<details class="report-fold"><summary>展开 29 个所选 signed-bias law 的样本外预测指标</summary><div class="fold-content">{table_html(display_signed)}</div></details>
<details class="report-fold"><summary>展开 29 个所选 signed-bias law 的复杂度、稳定性与残差诊断</summary><div class="fold-content">{table_html(diagnostic_signed)}</div></details>
<details><summary>展开全部 29 条数值公式与 bootstrap 区间</summary>{''.join(law_details)}</details>
<h3>Absolute deviation 的补充拟合</h3>
<details class="report-fold"><summary>展开 29 个 absolute-deviation law 的拟合表</summary><div class="fold-content">{table_html(abs_display)}</div></details>
<p>当 signed bias R² 很低但 \(|b|\) R² 较高时，主要机制不是“没有难度关系”，而是随着 N/L 变难，误差幅度增加但方向在不同 seeds 间正负抵消。Qwen3-8B Direct、Qwen3-4B Bullet/Index 是典型例子。</p>
<h3>Raw bias 较弱时的 b/N fallback</h3>
<p>按用户建议，只对 raw signed-bias CV R²&lt;0.5 的单元追加相对偏差 \(r=b/N\) 与相对绝对偏差 \(|b|/N\) 搜索。该步骤不改变 raw-bias 主结果。</p>
<details class="report-fold"><summary>展开弱 raw-bias 单元的归一化候选比较表</summary><div class="fold-content">{table_html(fallback_display)}</div></details>
<details class="report-fold"><summary>展开图 2：弱 raw-bias 单元的 b/N 备选曲线</summary><div class="fold-content">
<figure><img src="figures/bias/03_relative_bias_fallbacks.png" alt="Relative bias fallback"><figcaption><strong>图 2.</strong> 仅展示 raw signed-bias law 较弱的单元。横轴分别为 \(N\) 与 \(L\)，纵轴是 mean relative bias \(b/N\)；散点和 95% CI 为观测边际均值，实线为归一化目标的所选二维 law。</figcaption></figure>
</div></details>
<p>归一化最有帮助的是 GLM-4 Index：CV cell R² 从约 0.33 提升到约 0.59，说明其误差更接近“相对比例偏差”而非固定整数偏差。Qwen3-4B Index 提升到约 0.49，仍只到边界水平；其余弱单元没有因除以 N 而变成可靠 law。</p>
<h3>预先指定的 log-linear density law</h3>
<p>额外对全部 29 个模型 × mode 单元固定检验同一个两参数形式：</p>
<div class="equation">\\[\\overline{{b/N}}=\\alpha+\\beta\\ln(N/L_k),\\qquad L_k=L/1000.\\]</div>
<p>该检验不参与“挑选最好曲线”：每个单元都使用完全相同的响应、横轴和验证方式。它得到 {log_ratio_strong} 个强、{log_ratio_moderate} 个中等、{log_ratio_weak} 个弱拟合；斜率 bootstrap 区间、seed-CV、request OOF 以及 leave-N/L-out 指标全部保留。</p>
<details class="report-fold"><summary>展开图 3：固定 b/N–ln(N/L) law 的 29 单元拟合优度</summary><div class="fold-content">
<figure><img src="figures/bias/04_relative_log_ratio_quality.png" alt="Relative log-ratio law quality"><figcaption><strong>图 3.</strong> 每格固定使用 \(\overline{{b/N}}=\alpha+\beta\ln(N/L_k)\)，颜色与数字为 seed-cross-fitted cell R²；负值表示该直线在留出 seed 上比直接使用条件均值基线更差。</figcaption></figure>
</div></details>
<details class="report-fold"><summary>展开固定 log-linear density law 的 29 单元参数与验证表</summary><div class="fold-content">{table_html(log_ratio_display, escape=False)}</div></details>
<p>每个模型的折叠图把这条固定 law 放在第三列，因此可以同时比较“自由选择的 raw-bias 二维曲面”和“统一横轴的相对偏差直线”。若某一格 R² 很低，报告保留该负结果，不通过更换坐标继续追逐拟合。</p>
<div class="conclusion"><strong>本节结论：</strong>系统性漏计/过计的单元通常能得到强二维 law；接近无偏或误差方向随机的单元不应强行拟合 signed bias。自然对数已进入完整候选集，而统一的 \(b/N\)–\(\ln(N/L_k)\) 关系作为独立、不可事后更换的简洁假说单列报告。</div></section>

<section id="models"><h2>4. 各模型、各 Mode 的散点与拟合曲线</h2>
{''.join(model_sections)}
</section>

<section id="universal"><h2>5. 是否存在普适形式？</h2>
<h3>5.1 “普适”的操作定义与选模规则</h3>
<p>这里的“普适 law”不是强迫所有模型共享数值系数，而是要求同一 mode family 使用相同特征映射 \(g(N,L)\)，同时允许每个模型（Enumerate 还允许 Index/Bullet）拥有独立参数。Direct 覆盖 7 个模型单元；Enumerate 覆盖同一 7 个模型的 Index 与 Bullet，共 14 个单元。Signed bias 与 absolute deviation 分别选择，不能用后者掩盖前者。</p>
<p>共同形式按预先固定的稳健次序选择：先最大化 CV cell R²≥0.5 的覆盖单元数；将 median CV R² 相差不超过 0.02 的候选视为实际并列；再最大化 CV R² 的下四分位数，随后比较 leave-\(N/L\)-out 与参数数目。这个规则避免由单个极端模型或平均值主导选模。</p>
<details class="report-fold"><summary>展开 Direct/Enumerate 普适候选的前六名比较</summary><div class="fold-content">{table_html(universal_top_display)}</div></details>

<h3>5.2 Direct：误差幅度有普适 law，signed bias 为近普适</h3>
<p>Direct 的 signed bias 与 absolute deviation 都选中同一自然对数交互形式：</p>
<div class="equation">\\[
g_{{D,m}}(N,L_k)
=\\alpha_{{D,m}}+\\beta_{{N,D,m}}N
+\\beta_{{L,D,m}}\\ln L_k
+\\beta_{{N\\ln L,D,m}}N\\ln L_k,
\\qquad L_k=L/1000.
\\]</div>
<p>对 signed bias，形式为 \(\bar b_{{D,m}}=g_{{D,m}}\)：覆盖 {int(direct_signed.moderate_cells)}/{int(direct_signed.cells)} 个模型达到 CV R²≥0.5，median CV R²={direct_signed.median_cell_cv_r2:.3f}，下四分位数={direct_signed.q25_cell_cv_r2:.3f}；Qwen3-8B Direct 是弱拟合例外，因此不能称为“七个模型无例外”的 signed-bias law。对 absolute deviation，另用一套模型参数拟合 \(\overline{{|b|}}_{{D,m}}=g^{{(|b|)}}_{{D,m}}\)：7/7 达到中等以上、{int(direct_absolute.strong_cells)}/7 达到强拟合，median CV R²={direct_absolute.median_cell_cv_r2:.3f}，最低仍为 {direct_absolute.min_cell_cv_r2:.3f}。</p>
<div class="conclusion"><strong>Direct 结论：</strong>当前数据支持一个跨七模型的<strong>误差幅度</strong> law：\(\overline{{|b|}}\) 随 \(N\)、\(\ln L\) 及 \(N\ln L\) 变化，参数因模型而异。Signed bias 的同形 law 对 6/7 有效，但 Qwen3-8B 的误差方向在 seeds 间不稳定；因此“困难度如何放大误差”比“必然漏计还是过计”更普适。</div>

<h3>5.3 Enumerate：最佳共同形状为双线性，但 Index 存在明确例外</h3>
<p>把 Index 与 Bullet 视为同一 Enumerate family、但保留模型 × 格式各自参数后，signed bias 和 absolute deviation 都选择：</p>
<div class="equation">\\[
g_{{E,m,e}}(N,L_k)
=\\alpha_{{E,m,e}}+\\beta_{{N,E,m,e}}N
+\\beta_{{L,E,m,e}}L_k
+\\beta_{{NL,E,m,e}}NL_k,
\\qquad e\\in\\{{\\mathrm{{Index}},\\mathrm{{Bullet}}\\}}.
\\]</div>
<p>Signed bias 覆盖 {int(enumerate_signed.moderate_cells)}/{int(enumerate_signed.cells)} 个单元达到 CV R²≥0.5，median={enumerate_signed.median_cell_cv_r2:.3f}；absolute deviation 覆盖 {int(enumerate_absolute.moderate_cells)}/{int(enumerate_absolute.cells)}，median={enumerate_absolute.median_cell_cv_r2:.3f}。Bullet 的 absolute-deviation 版本在 7/7 模型达到中等以上；主要例外集中在 Index：Qwen3-32B Index 与 GLM-4 Index 的 absolute-deviation CV R² 分别约 0.10 与 0.39，signed bias 还包括 Qwen3-4B Index，以及 Qwen3-4B Bullet 的方向性例外。</p>
<div class="conclusion"><strong>Enumerate 结论：</strong>\(N+L+NL\) 是 14 个枚举单元中证据最强的共同形状，但它是“多数适用、存在可复现例外”的 law，而不是无条件普适定律。Bullet 的误差幅度关系明显更一致；Index 的近零偏差、偶发爆炸式过计与格式行为使单一平滑曲面难以同时解释所有模型。</div>

<details class="report-fold"><summary>展开图 5：Direct/Enumerate 普适形式的逐单元 CV R²</summary><div class="fold-content">
<figure><img src="figures/bias/05_universal_mode_quality.png" alt="Universal mode law quality"><figcaption><strong>图 5.</strong> 左面板是 7 个 Direct 模型，右面板是 14 个 Enumerate 模型 × 格式；两列分别为 signed bias 与 absolute deviation。每格固定使用该 family 选出的同一函数形式，但系数独立拟合；颜色与数字是 seed-cross-fitted condition-mean R²。图直接显示普适覆盖与例外，而不是只报告平均值。</figcaption></figure>
</div></details>
<details class="report-fold"><summary>展开图 6：普适形式的观测值与 held-out-seed 预测</summary><div class="fold-content">
<figure><img src="figures/bias/06_universal_mode_predictions.png" alt="Universal mode observed versus predicted"><figcaption><strong>图 6.</strong> 横轴是每个 \((N,L)\) 条件的观测均值，纵轴是 leave-one-seed 产生的交叉拟合预测；虚线是理想的 \(y=x\)。每种颜色表示模型；Enumerate 中方形为 Index、三角形为 Bullet。点靠近虚线说明同一函数形状在该条件上具有样本外解释力。</figcaption></figure>
</div></details>
<details class="report-fold"><summary>展开 Direct/Enumerate 所选普适 law 的汇总指标</summary><div class="fold-content">{table_html(universal_selected_display)}</div></details>
<details class="report-fold"><summary>展开 Direct/Enumerate 所选普适 law 的全部 42 个单元诊断</summary><div class="fold-content">{table_html(universal_per_cell_display)}</div></details>

<h3>5.4 Native Thinking：先前发现的共同双线性结构</h3>
<p>Native Thinking 最清晰的共同结构仍是固定同一双线性形式、只允许每个模型拥有不同参数：</p>
<div class="equation">\\[\\bar b_m(N,L)=\\alpha_m+\\beta_{{N,m}}N+\\beta_{{L,m}}L_k+\\beta_{{NL,m}}NL_k.\\]</div>
<details class="report-fold"><summary>展开图 7：Native Thinking 的共同双线性结构</summary><div class="fold-content">
<figure><img src="figures/bias/02_native_commonality.png" alt="Native thinking commonality"><figcaption><strong>图 7.</strong> 八个目标模型的 Native Thinking 均使用同一 \(N+L+NL\) 形式。蓝柱拟合 cell mean bias；绿柱拟合 cell median bias。标为 “flat 0” 表示 50 个条件的 median bias 全部为 0，此时 R² 无定义，而不是拟合失败。</figcaption></figure>
</div></details>
<details class="report-fold"><summary>展开 Native Thinking 共同结构的完整诊断表</summary><div class="fold-content">{table_html(native_display)}</div></details>
<p>四款 Qwen 的 mean-bias cell R² 分别约为 0.943、0.845、0.841、0.708；Qwen3-8B 与 32B 的 50 个条件 median bias 全部为 0。扩展到八个目标模型时，median 形式在所有非平坦模型上约为 0.774–0.931，DeepSeek 的 mean bias 是唯一明显例外，但其 median bias 仍可由同一形式解释（约 0.83）。</p>
<div class="conclusion"><strong>普适性总论：</strong>目前最强的三条共同结构是：Direct absolute deviation 的 \(N+\ln L+N\ln L\)；Enumerate（尤其 Bullet）absolute deviation 的 \(N+L+NL\)；Native Thinking 条件中心的 \(N+L+NL\)。参数必须随模型与 mode 改变，且 Enumerate Index 保留明确例外，因此不能支持一个对所有模式、所有模型共享相同参数或相同误差方向的单一定律。</div></section>

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
    return {
        **fit,
        "common": common,
        "fallback": fallback,
        "relative_log_ratio": relative_log_ratio,
        "universal_mode_laws": universal,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(output: Path, input_csv: Path, args: argparse.Namespace) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {
            "analysis_manifest.json",
            "SHA256SUMS.tsv",
        }:
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
        "prompt_failure_audit_dir": str(args.failure_audit_dir.resolve()),
        "input_rows": int(pd.read_csv(input_csv, usecols=["request_id"]).shape[0]),
        "selection": {
            "cells": 29,
            "requests_per_cell": 500,
            "enumeration": "V2.1 replacement",
            "gemma4_12b_direct": "V2.1 strict appendix substituted in the current-prompt 29-cell composite",
            "remaining_modes": "V2 formal",
        },
        "method": {
            "prompt_failure_taxonomy": "frozen-output audit; mutually exclusive priority truncated > parse > strict format > wrong count",
            "bias": "predicted_count - gold_count, parsed requests only",
            "primary_cell_target": "mean signed bias over seeds",
            "candidate_search": "finite low-complexity registry with natural-log N/L terms in build_v2_reports.py",
            "fixed_relative_log_law": "mean(bias/N) = alpha + beta ln(N/(L/1000)), evaluated in all 29 cells",
            "selection": "leave-one-seed CV; one-SE eligibility; cross-fitted cell R2 minus 0.015 per extra parameter",
            "goodness_of_fit": "cross-fitted R2/RMSE/MAE/median-AE/NRMSE-SD; request OOF metrics; leave-N/L-out; adjusted R2; Gaussian cell AIC/AICc/BIC/log-likelihood/SSE-deviance; residual Spearman diagnostics",
            "coefficient_stability": "minimum non-intercept bootstrap sign agreement and count of 95% coefficient intervals excluding zero",
            "universal_mode_laws": "one common feature map per Direct or Enumerate family with model/mode-specific coefficients; maximize R2>=0.5 coverage, treat median CV R2 within 0.02 as tied, then maximize lower-quartile CV R2, extrapolation, and parsimony",
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

- `01_prompt_accuracy_report.html`: prompt/mode accuracy and observed failure mechanisms, with long tables and figures collapsed by default.
- `02_bias_law_report.html`: model-by-mode signed-bias laws, natural-log candidates, the fixed `bias/N ~ ln(N/Lk)` diagnostic, expanded goodness-of-fit metrics, and Direct/Enumerate/Native common functional forms.

The current-prompt composite contains exactly 29 cells × 500 requests. V2.1
enumeration replacements supersede the old enumeration rows. The strict
Gemma4-12B Direct appendix supplies the current Direct cell for numerical-bias
analysis; its original V2 Direct run remains a prompt-failure comparator only.

Rebuild:

```powershell
python scripts/build_v2_reports.py --input tables/request_level_compact.csv --failure-audit-dir tables/prompt --output .
```

All formulas are exploratory empirical response surfaces. The candidate tables
include held-out R²/RMSE/MAE/median-AE/NRMSE, leave-N/L-out metrics, information
criteria, bootstrap coefficient stability, and residual diagnostics. See the
reports for the exact success, bias, validation, and comparison definitions.
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
    parser.add_argument(
        "--failure-audit-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prompt_failure_audit",
    )
    parser.add_argument("--generated", default="2026-07-27")
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
    compact_copy = output_tables / "request_level_compact.csv"
    if args.input.resolve() != compact_copy.resolve():
        shutil.copy2(args.input, compact_copy)
    scripts = output / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), scripts / Path(__file__).name)
    for support_name in ["prompt_failure_audit.py", "eda_v2.py"]:
        support = Path(__file__).resolve().parent / support_name
        if support.is_file():
            shutil.copy2(support, scripts / support.name)
    set_plot_style()
    build_prompt_report(data, args.v2_root, args.failure_audit_dir, output, args.generated)
    build_bias_report(data, output, args.generated)
    write_readme(output)
    build_manifest(output, args.input, args)
    print(output / "01_prompt_accuracy_report.html")
    print(output / "02_bias_law_report.html")


if __name__ == "__main__":
    main()
