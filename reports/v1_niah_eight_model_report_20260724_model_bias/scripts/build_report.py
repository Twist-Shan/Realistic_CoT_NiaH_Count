#!/usr/bin/env python3
"""Build a reproducible HTML report for the eight-model Realistic NiaH run.

The script reads the immutable request JSONL files, independently reconstructs
all descriptive tables, replays the primary leave-one-seed-out regressions, and
then incorporates the hash-verified blocked-validation/bootstrap artifacts from
the archived empirical-law analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit


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
EXPECTED_ROWS = {
    "Qwen3-1.7B": 900,
    "Qwen3-8B": 900,
    "Qwen3-32B": 900,
    "Gemma4-E4B": 900,
    "Gemma4-12B": 900,
    "OLMo-Hybrid-7B": 600,
    "Llama3.1-8B": 600,
    "Llama3.2-3B": 600,
}
PROMPT_MODES = ["direct", "enumeration", "native_thinking"]
PROMPT_LABELS = {
    "direct": "直接计数（thinking 关）",
    "enumeration": "逐条枚举（thinking 关）",
    "native_thinking": "直接计数（原生 thinking 开）",
}
PROMPT_SHORT = {
    "direct": "Direct / off",
    "enumeration": "Enumeration / off",
    "native_thinking": "Native thinking / on",
}
LENGTHS = [2000, 5000, 10000]
NEEDLES = [1, 2, 3, 4, 5, 6, 8, 10, 20, 30]
SEEDS = [1234, 1235, 1236, 1237, 1238]
ORDERS = ["query_first", "query_last"]
EXPECTED_STIMULI_SHA = (
    "374dc935bf4c1403f705bb8b95ce686e5063647c83c609501e6f668e2331a5f1"
)
EXPECTED_COMMIT = "090d983819f06234cb135f6c499bf82e9a6de1c9"
RANDOM_SEED = 20260724
RIDGE = 1e-4

ERROR_ORDER = [
    "correct",
    "undercount",
    "overcount",
    "parse_failure",
    "truncation",
    "other_numeric",
]
ERROR_LABELS = {
    "correct": "正确",
    "undercount": "已解析但少计",
    "overcount": "已解析但多计",
    "parse_failure": "格式/解析失败",
    "truncation": "输出截断",
    "other_numeric": "其他数值不一致",
}
ERROR_COLORS = {
    "correct": "#2a9d8f",
    "undercount": "#e9c46a",
    "overcount": "#f4a261",
    "parse_failure": "#e76f51",
    "truncation": "#6d597a",
    "other_numeric": "#8d99ae",
}
ERROR_LABELS_EN = {
    "correct": "Correct",
    "undercount": "Parsed under-count",
    "overcount": "Parsed over-count",
    "parse_failure": "Parse / format failure",
    "truncation": "Truncated",
    "other_numeric": "Other numeric mismatch",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def output_excerpt(text: str, limit: int = 420) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    half = (limit - 5) // 2
    return normalized[:half] + " … " + normalized[-half:]


def classify_error(
    *,
    exact: bool,
    truncated: bool,
    parse_status: str,
    signed_error: float,
) -> str:
    if exact:
        return "correct"
    if truncated:
        return "truncation"
    if parse_status != "ok":
        return "parse_failure"
    if math.isfinite(signed_error):
        if signed_error < 0:
            return "undercount"
        if signed_error > 0:
            return "overcount"
    return "other_numeric"


def load_requests(run_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for model in MODELS:
        path = run_root / model / "main" / "requests.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing main result: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                ev = obj.get("evaluation") or {}
                exact = bool(ev.get("exact_count", False))
                truncated = bool(ev.get("truncated", False))
                parse_status = str(ev.get("parse_status") or "parse_fail")
                signed_error = safe_float(ev.get("signed_error"))
                category = classify_error(
                    exact=exact,
                    truncated=truncated,
                    parse_status=parse_status,
                    signed_error=signed_error,
                )
                missing_pairs = ev.get("missing_pairs") or []
                hallucinated_pairs = ev.get("hallucinated_pairs") or []
                listed_records = ev.get("listed_records") or []
                rows.append(
                    {
                        "request_id": obj["request_id"],
                        "stimulus_id": obj["stimulus_id"],
                        "seed": int(obj["seed"]),
                        "model_label": obj["model_label"],
                        "model_id": obj["model_id"],
                        "target_passage_tokens": int(
                            obj["target_passage_tokens"]
                        ),
                        "model_passage_tokens": int(
                            obj.get("model_passage_tokens")
                            or obj["target_passage_tokens"]
                        ),
                        "model_input_tokens": int(
                            obj.get("model_input_tokens") or 0
                        ),
                        "num_needles": int(obj["num_needles"]),
                        "density_per_1k": (
                            1000.0
                            * float(obj["num_needles"])
                            / float(obj["target_passage_tokens"])
                        ),
                        "prompt_mode": obj["prompt_mode"],
                        "thinking_enabled": int(
                            obj["prompt_mode"] == "native_thinking"
                        ),
                        "query_order": obj["query_order"],
                        "exact_correct": int(exact),
                        "parse_success": int(parse_status == "ok"),
                        "format_failure": int(parse_status != "ok"),
                        "truncated": int(truncated),
                        "parse_status": parse_status,
                        "finish_reason": str(
                            ev.get("finish_reason")
                            or obj.get("finish_reason")
                            or ""
                        ),
                        "gold_count": int(
                            obj.get("gold_count") or obj["num_needles"]
                        ),
                        "predicted_count": safe_float(
                            ev.get("predicted_count")
                        ),
                        "absolute_error": safe_float(ev.get("absolute_error")),
                        "normalized_absolute_error": safe_float(
                            ev.get("normalized_absolute_error")
                        ),
                        "signed_error": signed_error,
                        "output_tokens": int(obj.get("output_tokens") or 0),
                        "error_category": category,
                        "missing_pairs_n": len(missing_pairs),
                        "hallucinated_pairs_n": len(hallucinated_pairs),
                        "duplicate_listed_pairs_n": int(
                            ev.get("duplicate_listed_pairs") or 0
                        ),
                        "listed_records_n": len(listed_records),
                        "listed_total_matches_length": (
                            ev.get("listed_total_matches_length")
                        ),
                        "pair_precision": safe_float(ev.get("pair_precision")),
                        "pair_recall": safe_float(ev.get("pair_recall")),
                        "pair_f1": safe_float(ev.get("pair_f1")),
                        "raw_output_excerpt": output_excerpt(
                            obj.get("raw_output_text") or ""
                        ),
                        "source_file": str(path),
                        "source_line": line_no,
                    }
                )
                count += 1
        sources.append(
            {
                "model": model,
                "path": str(path),
                "rows": count,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, sources


def validate_frame(frame: pd.DataFrame, run_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    if len(frame) != 6300:
        failures.append(f"expected 6300 rows, found {len(frame)}")
    if frame["request_id"].nunique() != len(frame):
        failures.append("duplicate request_id detected")
    counts = frame["model_label"].value_counts().to_dict()
    for model, expected in EXPECTED_ROWS.items():
        if counts.get(model) != expected:
            failures.append(
                f"{model}: expected {expected}, found {counts.get(model)}"
            )
    if sorted(frame["target_passage_tokens"].unique().tolist()) != LENGTHS:
        failures.append("unexpected passage length grid")
    if sorted(frame["num_needles"].unique().tolist()) != NEEDLES:
        failures.append("unexpected needle-count grid")
    if sorted(frame["seed"].unique().tolist()) != SEEDS:
        failures.append("unexpected seed grid")
    if sorted(frame["query_order"].unique().tolist()) != ORDERS:
        failures.append("unexpected query-order grid")
    if (frame["exact_correct"] > frame["parse_success"]).any():
        failures.append("exact-correct row without parsed output")
    if (
        frame.loc[frame["parse_success"] == 1, "predicted_count"].isna().any()
    ):
        failures.append("parsed row without numeric prediction")

    dataset = (
        run_root.parent / "dataset_candidate_n6_offset" / "stimuli.jsonl"
    )
    dataset_sha = sha256(dataset) if dataset.is_file() else None
    if dataset_sha != EXPECTED_STIMULI_SHA:
        failures.append(
            f"stimuli SHA mismatch: expected {EXPECTED_STIMULI_SHA}, "
            f"found {dataset_sha}"
        )

    manifests: list[dict[str, Any]] = []
    for model in MODELS:
        manifest_path = run_root / model / "main" / "run_manifest.json"
        qc_path = run_root / model / "main" / "qc_report.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        manifests.append(
            {
                "model": model,
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "qc_path": str(qc_path),
                "qc_sha256": sha256(qc_path),
                "completed": manifest.get("completed_requests"),
                "expected": manifest.get("expected_requests"),
                "qc_status": qc.get("status"),
            }
        )

    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "requests": len(frame),
        "unique_request_ids": int(frame["request_id"].nunique()),
        "unique_stimuli": int(frame["stimulus_id"].nunique()),
        "dataset_path": str(dataset),
        "stimuli_sha256": dataset_sha,
        "expected_git_commit": EXPECTED_COMMIT,
        "model_rows": counts,
        "manifests": manifests,
    }
    if failures:
        raise RuntimeError("Input validation failed: " + "; ".join(failures))
    return result


def aggregate_tables(
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    model_summary = (
        frame.groupby("model_label", sort=False)
        .agg(
            requests=("request_id", "size"),
            exact_correct=("exact_correct", "sum"),
            accuracy=("exact_correct", "mean"),
            parsed_outputs=("parse_success", "sum"),
            parse_success_rate=("parse_success", "mean"),
            format_failure_rate=("format_failure", "mean"),
            truncation_rate=("truncated", "mean"),
            mean_absolute_error_parsed=("absolute_error", "mean"),
            median_absolute_error_parsed=("absolute_error", "median"),
            mean_signed_bias_parsed=("signed_error", "mean"),
            median_signed_bias_parsed=("signed_error", "median"),
        )
        .reindex(MODELS)
        .reset_index()
    )
    model_summary["accuracy_rank"] = (
        model_summary["accuracy"].rank(method="min", ascending=False).astype(int)
    )

    mode_summary = (
        frame.groupby(["model_label", "prompt_mode"], sort=False)
        .agg(
            requests=("request_id", "size"),
            exact_correct=("exact_correct", "sum"),
            accuracy=("exact_correct", "mean"),
            parsed_outputs=("parse_success", "sum"),
            parse_success_rate=("parse_success", "mean"),
            format_failure_rate=("format_failure", "mean"),
            truncation_rate=("truncated", "mean"),
            mean_absolute_error_parsed=("absolute_error", "mean"),
            median_absolute_error_parsed=("absolute_error", "median"),
            mean_signed_bias_parsed=("signed_error", "mean"),
            median_signed_bias_parsed=("signed_error", "median"),
        )
        .reset_index()
    )

    cell_accuracy = (
        frame.groupby(
            [
                "model_label",
                "prompt_mode",
                "target_passage_tokens",
                "num_needles",
            ],
            sort=False,
        )
        .agg(
            requests=("request_id", "size"),
            exact_correct=("exact_correct", "sum"),
            accuracy=("exact_correct", "mean"),
            parse_success_rate=("parse_success", "mean"),
            truncation_rate=("truncated", "mean"),
        )
        .reset_index()
    )

    error_categories = (
        frame.groupby(["model_label", "prompt_mode", "error_category"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        frame.groupby(["model_label", "prompt_mode"])
        .size()
        .rename("requests")
        .reset_index()
    )
    error_categories = error_categories.merge(
        totals, on=["model_label", "prompt_mode"], how="left"
    )
    error_categories["rate"] = (
        error_categories["count"] / error_categories["requests"]
    )

    enumeration = frame[
        (frame["prompt_mode"] == "enumeration")
        & (frame["exact_correct"] == 0)
    ].copy()
    enum_rows: list[dict[str, Any]] = []
    for model in MODELS:
        group = enumeration[enumeration["model_label"] == model]
        if group.empty:
            continue
        enum_rows.append(
            {
                "model_label": model,
                "wrong_enumeration_requests": len(group),
                "truncation": int((group["error_category"] == "truncation").sum()),
                "parse_failure_nontruncated": int(
                    (group["error_category"] == "parse_failure").sum()
                ),
                "parsed_undercount": int(
                    (group["error_category"] == "undercount").sum()
                ),
                "parsed_overcount": int(
                    (group["error_category"] == "overcount").sum()
                ),
                "missing_gold_pair_flag": int(
                    (group["missing_pairs_n"] > 0).sum()
                ),
                "hallucinated_pair_flag": int(
                    (group["hallucinated_pairs_n"] > 0).sum()
                ),
                "duplicate_pair_flag": int(
                    (group["duplicate_listed_pairs_n"] > 0).sum()
                ),
                "listed_total_mismatch_flag": int(
                    (group["listed_total_matches_length"] == False).sum()  # noqa: E712
                ),
            }
        )
    enumeration_mechanisms = pd.DataFrame(enum_rows)

    return {
        "model_summary": model_summary,
        "mode_summary": mode_summary,
        "cell_accuracy": cell_accuracy,
        "error_categories": error_categories,
        "enumeration_mechanisms": enumeration_mechanisms,
    }


def paired_thinking_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    capable = [
        model
        for model in MODELS
        if "native_thinking"
        in set(frame.loc[frame["model_label"] == model, "prompt_mode"])
    ]
    for model in capable:
        sub = frame[
            (frame["model_label"] == model)
            & (frame["prompt_mode"].isin(["direct", "native_thinking"]))
        ]
        pivot = sub.pivot(
            index=["stimulus_id", "query_order"],
            columns="prompt_mode",
            values="exact_correct",
        ).dropna()
        direct = float(pivot["direct"].mean())
        thinking = float(pivot["native_thinking"].mean())
        delta = thinking - direct
        clusters = np.array(
            sorted(pivot.index.get_level_values("stimulus_id").unique())
        )
        boot: list[float] = []
        for _ in range(500):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            values: list[float] = []
            for stimulus_id in sampled:
                block = pivot.loc[stimulus_id]
                values.extend(
                    (
                        block["native_thinking"] - block["direct"]
                    ).to_numpy(dtype=float)
                )
            boot.append(float(np.mean(values)))
        rows.append(
            {
                "model_label": model,
                "paired_requests_per_mode": len(pivot),
                "direct_accuracy": direct,
                "native_thinking_accuracy": thinking,
                "difference": delta,
                "difference_percentage_points": 100.0 * delta,
                "cluster_bootstrap_ci95_low": float(
                    np.quantile(boot, 0.025)
                ),
                "cluster_bootstrap_ci95_high": float(
                    np.quantile(boot, 0.975)
                ),
                "bootstrap_replicates": 500,
            }
        )
    return pd.DataFrame(rows)


def select_error_examples(frame: pd.DataFrame) -> pd.DataFrame:
    examples: list[pd.Series] = []
    for model in MODELS:
        sub = frame[frame["model_label"] == model]
        for category in [
            "truncation",
            "parse_failure",
            "undercount",
            "overcount",
        ]:
            group = sub[sub["error_category"] == category]
            if group.empty:
                continue
            if category in {"undercount", "overcount"}:
                chosen = group.sort_values(
                    ["absolute_error", "request_id"], ascending=[False, True]
                ).iloc[0]
            else:
                chosen = group.sort_values(
                    ["output_tokens", "request_id"], ascending=[False, True]
                ).iloc[0]
            examples.append(chosen)
    if not examples:
        return frame.iloc[0:0].copy()
    columns = [
        "model_label",
        "error_category",
        "request_id",
        "prompt_mode",
        "query_order",
        "target_passage_tokens",
        "num_needles",
        "gold_count",
        "predicted_count",
        "signed_error",
        "finish_reason",
        "output_tokens",
        "raw_output_excerpt",
    ]
    return pd.DataFrame(examples)[columns].reset_index(drop=True)


def base_design(frame: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
    arrays = [np.ones(len(frame), dtype=float)]
    names = ["Intercept"]
    for model in MODELS[1:]:
        arrays.append((frame["model_label"].to_numpy() == model).astype(float))
        names.append(f"model_label[{model}]")
    for mode in PROMPT_MODES[1:]:
        arrays.append((frame["prompt_mode"].to_numpy() == mode).astype(float))
        names.append(f"prompt_mode[{mode}]")
    arrays.append(
        (frame["query_order"].to_numpy() == "query_last").astype(float)
    )
    names.append("query_order[query_last]")
    return arrays, names


def design_matrix(
    frame: pd.DataFrame, candidate: str
) -> tuple[np.ndarray, list[str]]:
    arrays, names = base_design(frame)
    log_length = np.log2(
        frame["target_passage_tokens"].to_numpy(dtype=float) / 5000.0
    )
    log_needles = np.log2(frame["num_needles"].to_numpy(dtype=float) / 5.0)
    log_density = np.log2(frame["density_per_1k"].to_numpy(dtype=float))
    if candidate == "controls_only":
        pass
    elif candidate == "density_model_fe":
        arrays.append(log_density)
        names.append("log_density")
    elif candidate == "log_length_needles_model_fe":
        arrays.extend([log_length, log_needles])
        names.extend(["log_length", "log_needles"])
    elif candidate == "log_length_needles_interaction_model_fe":
        arrays.extend([log_length, log_needles, log_length * log_needles])
        names.extend(
            ["log_length", "log_needles", "log_length:log_needles"]
        )
    elif candidate == "model_specific_length_needles_slopes":
        arrays.extend([log_length, log_needles])
        names.extend(["log_length", "log_needles"])
        for model in MODELS[1:]:
            indicator = (
                frame["model_label"].to_numpy() == model
            ).astype(float)
            arrays.extend([indicator * log_length, indicator * log_needles])
            names.extend(
                [
                    f"log_length:model_label[{model}]",
                    f"log_needles:model_label[{model}]",
                ]
            )
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    return np.column_stack(arrays), names


def fit_logistic(
    x: np.ndarray, y: np.ndarray, ridge: float = RIDGE
) -> tuple[np.ndarray, bool]:
    n_features = x.shape[1]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        penalty = 0.5 * ridge * float(np.dot(beta[1:], beta[1:]))
        probability = expit(eta)
        gradient = x.T @ (probability - y)
        gradient[1:] += ridge * beta[1:]
        return loss + penalty, gradient

    result = minimize(
        objective,
        np.zeros(n_features, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    return result.x, bool(result.success)


def log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_metrics(
    y: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    score = np.log(p / (1 - p))
    x = np.column_stack([np.ones(len(score)), score])
    beta, converged = fit_logistic(x, y, ridge=0.0)
    if not converged:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def expected_calibration_error(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (
                probability <= edges[index + 1]
            )
        else:
            mask = (probability >= edges[index]) & (
                probability < edges[index + 1]
            )
        if mask.any():
            total += float(mask.mean()) * abs(
                float(y[mask].mean()) - float(probability[mask].mean())
            )
    return total


def replay_regressions(
    frame: pd.DataFrame, archived_comparison: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = [
        "controls_only",
        "density_model_fe",
        "log_length_needles_model_fe",
        "log_length_needles_interaction_model_fe",
        "model_specific_length_needles_slopes",
    ]
    y = frame["exact_correct"].to_numpy(dtype=float)
    candidate_rows: list[dict[str, Any]] = []
    oof_by_candidate: dict[str, np.ndarray] = {}
    for candidate in candidates:
        x, names = design_matrix(frame, candidate)
        oof = np.full(len(frame), np.nan)
        fold_losses: list[float] = []
        converged_all = True
        for seed in SEEDS:
            test = frame["seed"].to_numpy() == seed
            train = ~test
            beta, converged = fit_logistic(x[train], y[train])
            converged_all = converged_all and converged
            oof[test] = expit(x[test] @ beta)
            fold_losses.append(log_loss(y[test], oof[test]))
        if np.isnan(oof).any():
            raise RuntimeError(f"OOF predictions incomplete for {candidate}")
        beta_full, converged = fit_logistic(x, y)
        converged_all = converged_all and converged
        full_probability = expit(x @ beta_full)
        full_log_likelihood = -len(frame) * log_loss(y, full_probability)
        intercept, slope = calibration_metrics(y, oof)
        row = {
            "candidate": candidate,
            "n_parameters": x.shape[1],
            "converged_all_folds": converged_all,
            "local_cv_log_loss_mean": log_loss(y, oof),
            "local_cv_log_loss_se": float(
                np.std(fold_losses, ddof=1) / math.sqrt(len(fold_losses))
            ),
            "local_cv_brier": float(np.mean((oof - y) ** 2)),
            "local_cv_ece": expected_calibration_error(y, oof),
            "local_cv_calibration_intercept": intercept,
            "local_cv_calibration_slope": slope,
            "local_full_log_likelihood": full_log_likelihood,
            "local_aic": 2 * x.shape[1] - 2 * full_log_likelihood,
            "local_bic": (
                math.log(len(frame)) * x.shape[1] - 2 * full_log_likelihood
            ),
        }
        archived = archived_comparison[
            archived_comparison["candidate"] == candidate
        ]
        if len(archived) == 1:
            archived_row = archived.iloc[0]
            row["archived_cv_log_loss_mean"] = float(
                archived_row["cv_log_loss_mean"]
            )
            row["absolute_log_loss_difference"] = abs(
                row["local_cv_log_loss_mean"]
                - row["archived_cv_log_loss_mean"]
            )
        candidate_rows.append(row)
        oof_by_candidate[candidate] = oof

    local_comparison = pd.DataFrame(candidate_rows).sort_values(
        "local_cv_log_loss_mean"
    )
    max_difference = float(
        local_comparison["absolute_log_loss_difference"].dropna().max()
    )
    if max_difference > 2e-4:
        raise RuntimeError(
            "Local replay differs materially from archived regression: "
            f"max |Δ log loss|={max_difference}"
        )

    slope_rows: list[dict[str, Any]] = []
    for model in MODELS:
        sub = frame[frame["model_label"] == model].copy()
        y_model = sub["exact_correct"].to_numpy(dtype=float)
        arrays = [np.ones(len(sub), dtype=float)]
        names = ["Intercept"]
        for mode in PROMPT_MODES[1:]:
            if mode in set(sub["prompt_mode"]):
                arrays.append(
                    (sub["prompt_mode"].to_numpy() == mode).astype(float)
                )
                names.append(f"prompt_mode[{mode}]")
        arrays.append(
            (sub["query_order"].to_numpy() == "query_last").astype(float)
        )
        names.append("query_order[query_last]")
        log_length = np.log2(
            sub["target_passage_tokens"].to_numpy(dtype=float) / 5000.0
        )
        log_needles = np.log2(
            sub["num_needles"].to_numpy(dtype=float) / 5.0
        )
        arrays.extend([log_length, log_needles])
        names.extend(["log_length", "log_needles"])
        x = np.column_stack(arrays)
        beta, converged = fit_logistic(x, y_model)
        p = expit(x @ beta)
        weights = p * (1 - p)
        hessian = x.T @ (x * weights[:, None])
        hessian[1:, 1:] += np.eye(x.shape[1] - 1) * RIDGE
        covariance = np.linalg.pinv(hessian)
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        for feature in ["log_length", "log_needles"]:
            index = names.index(feature)
            estimate = float(beta[index])
            standard_error = float(standard_errors[index])
            slope_rows.append(
                {
                    "model_label": model,
                    "feature": feature,
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "ci95_low_hessian": estimate - 1.96 * standard_error,
                    "ci95_high_hessian": estimate + 1.96 * standard_error,
                    "odds_ratio_per_doubling": math.exp(estimate),
                    "converged": converged,
                }
            )
    local_slopes = pd.DataFrame(slope_rows)
    replay_summary = {
        "max_absolute_log_loss_difference_vs_archived": max_difference,
        "candidate_count": len(local_comparison),
        "folds": "five leave-one-seed-out folds",
        "ridge": RIDGE,
        "status": "pass",
    }
    return local_comparison, local_slopes, replay_summary


def percent(value: float, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def number(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def html_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, str]],
    *,
    table_class: str = "data-table",
) -> str:
    parts = [f'<div class="table-wrap"><table class="{table_class}"><thead><tr>']
    for _, label, _ in columns:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in frame.iterrows():
        parts.append("<tr>")
        for key, _, fmt in columns:
            value = row.get(key)
            if fmt == "pct":
                rendered = percent(value)
            elif fmt == "pp":
                rendered = "—" if pd.isna(value) else f"{float(value):+.1f} pp"
            elif fmt == "int":
                rendered = "—" if pd.isna(value) else f"{int(value):,}"
            elif fmt == "float2":
                rendered = number(value, 2)
            elif fmt == "float3":
                rendered = number(value, 3)
            elif fmt == "float4":
                rendered = number(value, 4)
            else:
                rendered = "" if pd.isna(value) else str(value)
            parts.append(f"<td>{html.escape(rendered)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def accuracy_color(value: float) -> str:
    value = min(1.0, max(0.0, float(value)))
    low = np.array([252, 236, 230], dtype=float)
    mid = np.array([251, 243, 200], dtype=float)
    high = np.array([211, 240, 226], dtype=float)
    if value <= 0.5:
        rgb = low + (mid - low) * (value / 0.5)
    else:
        rgb = mid + (high - mid) * ((value - 0.5) / 0.5)
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def accuracy_grid(
    cells: pd.DataFrame, model: str, mode: str
) -> str:
    sub = cells[
        (cells["model_label"] == model) & (cells["prompt_mode"] == mode)
    ]
    if sub.empty:
        return '<p class="muted">该模型未运行此模式。</p>'
    lookup = {
        (int(row.num_needles), int(row.target_passage_tokens)): row
        for row in sub.itertuples()
    }
    parts = [
        '<div class="table-wrap"><table class="accuracy-grid">',
        "<thead><tr><th>Needles N</th>",
    ]
    for length in LENGTHS:
        parts.append(f"<th>T={length:,} tokens</th>")
    parts.append("</tr></thead><tbody>")
    for needle in NEEDLES:
        parts.append(f"<tr><th>{needle}</th>")
        for length in LENGTHS:
            row = lookup[(needle, length)]
            value = float(row.accuracy)
            parts.append(
                '<td style="background:'
                + accuracy_color(value)
                + '">'
                + f"<strong>{percent(value)}</strong>"
                + f"<span>{int(row.exact_correct)}/{int(row.requests)}</span>"
                + "</td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def mode_error_table(
    error_categories: pd.DataFrame, model: str
) -> str:
    rows: list[dict[str, Any]] = []
    for mode in PROMPT_MODES:
        sub = error_categories[
            (error_categories["model_label"] == model)
            & (error_categories["prompt_mode"] == mode)
        ]
        if sub.empty:
            continue
        lookup = {
            row.error_category: (int(row.count), float(row.rate))
            for row in sub.itertuples()
        }
        item: dict[str, Any] = {
            "mode": PROMPT_LABELS[mode],
            "requests": int(sub["requests"].iloc[0]),
        }
        for category in ERROR_ORDER:
            count, rate = lookup.get(category, (0, 0.0))
            item[f"{category}_text"] = f"{count} ({percent(rate)})"
        rows.append(item)
    frame = pd.DataFrame(rows)
    columns = [("mode", "模式", "str"), ("requests", "n", "int")]
    for category in ERROR_ORDER[:-1]:
        columns.append(
            (f"{category}_text", ERROR_LABELS[category], "str")
        )
    return html_table(frame, columns)


def mode_summary_for_model(mode_summary: pd.DataFrame, model: str) -> str:
    sub = mode_summary[mode_summary["model_label"] == model].copy()
    sub["mode_label"] = sub["prompt_mode"].map(PROMPT_LABELS)
    return html_table(
        sub,
        [
            ("mode_label", "模式", "str"),
            ("requests", "n", "int"),
            ("accuracy", "准确率", "pct"),
            ("parse_success_rate", "解析成功率", "pct"),
            ("truncation_rate", "截断率", "pct"),
            ("median_absolute_error_parsed", "解析后 |误差| 中位数", "float2"),
            ("median_signed_bias_parsed", "解析后 bias 中位数", "float2"),
        ],
    )


def model_narrative(
    model: str, model_summary: pd.DataFrame, mode_summary: pd.DataFrame
) -> str:
    overall = model_summary[model_summary["model_label"] == model].iloc[0]
    modes = mode_summary[mode_summary["model_label"] == model]
    best = modes.sort_values("accuracy", ascending=False).iloc[0]
    worst = modes.sort_values("accuracy", ascending=True).iloc[0]
    dominant_candidates = {
        "truncation": float(overall["truncation_rate"]),
        "format_failure": float(overall["format_failure_rate"]),
    }
    dominant = max(dominant_candidates, key=dominant_candidates.get)
    dominant_text = (
        "截断"
        if dominant == "truncation"
        else "没有形成可解析的规定格式"
    )
    return (
        f"总体准确率 <strong>{percent(overall['accuracy'])}</strong>"
        f"（第 {int(overall['accuracy_rank'])}/8），解析成功率 "
        f"{percent(overall['parse_success_rate'])}。最佳模式是"
        f"“{html.escape(PROMPT_LABELS[best['prompt_mode']])}”"
        f"（{percent(best['accuracy'])}），最低模式是"
        f"“{html.escape(PROMPT_LABELS[worst['prompt_mode']])}”"
        f"（{percent(worst['accuracy'])}）。主要非数值失败来源是"
        f"{dominant_text}；在能解析出数值的请求中，绝对误差中位数为 "
        f"{number(overall['median_absolute_error_parsed'])}，"
        f"signed bias 中位数为 "
        f"{number(overall['median_signed_bias_parsed'])}。"
    )


def make_figures(
    frame: pd.DataFrame,
    aggregates: dict[str, pd.DataFrame],
    thinking: pd.DataFrame,
    archived_accuracy_candidates: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    model_summary = aggregates["model_summary"].sort_values(
        "accuracy", ascending=True
    )
    ordered = model_summary["model_label"].tolist()
    composition = (
        frame.groupby(["model_label", "error_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(ordered)
    )
    composition = composition.div(composition.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    left = np.zeros(len(composition))
    for category in ERROR_ORDER:
        values = (
            composition[category].to_numpy()
            if category in composition
            else np.zeros(len(composition))
        )
        ax.barh(
            composition.index,
            values,
            left=left,
            label=ERROR_LABELS_EN[category],
            color=ERROR_COLORS[category],
            edgecolor="white",
            linewidth=0.4,
        )
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of all registered requests")
    ax.set_title("Exclusive outcome composition by model")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output / "fig01_model_error_composition.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    x = np.arange(len(thinking))
    ax.plot(
        x,
        thinking["direct_accuracy"],
        "o-",
        label="Direct / thinking off",
        color="#457b9d",
    )
    ax.plot(
        x,
        thinking["native_thinking_accuracy"],
        "s-",
        label="Native thinking on",
        color="#e76f51",
    )
    for index, row in thinking.reset_index(drop=True).iterrows():
        ax.annotate(
            f"{row['difference_percentage_points']:+.1f} pp",
            (index, row["native_thinking_accuracy"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x, thinking["model_label"], rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Exact-count accuracy")
    ax.set_title("Direct task: native thinking on versus off")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "fig02_native_thinking_comparison.png", bbox_inches="tight")
    plt.close(fig)

    mode_matrix = (
        aggregates["mode_summary"]
        .pivot(index="model_label", columns="prompt_mode", values="accuracy")
        .reindex(index=MODELS, columns=PROMPT_MODES)
    )
    fig, ax = plt.subplots(figsize=(8.1, 5.8))
    image = ax.imshow(
        mode_matrix.to_numpy(dtype=float),
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        aspect="auto",
    )
    ax.set_yticks(np.arange(len(MODELS)), MODELS)
    ax.set_xticks(
        np.arange(len(PROMPT_MODES)),
        ["Direct\noff", "Enumeration\noff", "Native thinking\non"],
    )
    for i in range(len(MODELS)):
        for j in range(len(PROMPT_MODES)):
            value = mode_matrix.iloc[i, j]
            text = "N/A" if pd.isna(value) else f"{100 * value:.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    ax.set_title("Accuracy by model and prompt/thinking mode")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label("Exact-count accuracy")
    colorbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    fig.tight_layout()
    fig.savefig(output / "fig03_mode_accuracy_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    candidate_plot = archived_accuracy_candidates[
        archived_accuracy_candidates["candidate"].isin(
            [
                "controls_only",
                "density_model_fe",
                "log_length_needles_model_fe",
                "log_length_needles_interaction_model_fe",
                "model_specific_length_needles_slopes",
            ]
        )
    ].sort_values("cv_log_loss_mean", ascending=False)
    labels = {
        "controls_only": "Controls only",
        "density_model_fe": "Density only",
        "log_length_needles_model_fe": "Shared log L + log N",
        "log_length_needles_interaction_model_fe": "Shared + interaction",
        "model_specific_length_needles_slopes": "Model-specific slopes",
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y_pos = np.arange(len(candidate_plot))
    ax.errorbar(
        candidate_plot["cv_log_loss_mean"],
        y_pos,
        xerr=1.96 * candidate_plot["cv_log_loss_se"],
        fmt="o",
        color="#264653",
        ecolor="#8d99ae",
        capsize=3,
    )
    ax.set_yticks(
        y_pos,
        [labels[value] for value in candidate_plot["candidate"]],
    )
    ax.set_xlabel("Leave-one-seed-out log loss (lower is better)")
    ax.set_title("Accuracy-law candidate comparison")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output / "fig04_accuracy_law_cv.png", bbox_inches="tight")
    plt.close(fig)


def copy_archived_assets(analysis_root: Path, assets: Path) -> list[dict[str, Any]]:
    selected = {
        (
            "empirical_law_no_model_size_v1/figures/"
            "accuracy_by_length_needles.png"
        ): "fig05_pooled_accuracy_length_needles.png",
        (
            "unified_parametric_law_v1/figures/"
            "functional_form_selection.png"
        ): "fig06_functional_form_selection.png",
        (
            "unified_parametric_law_v1/figures/"
            "model_law_parameters.png"
        ): "fig07_unified_law_parameters.png",
        (
            "unified_parametric_law_v1/figures/"
            "selected_law_surfaces.png"
        ): "fig08_unified_law_surfaces.png",
        (
            "unified_parametric_law_v1/figures/"
            "heldout_cell_observed_vs_predicted.png"
        ): "fig09_heldout_observed_predicted.png",
        (
            "bias_law_v4/figures/model_bias_summary.png"
        ): "fig10_model_bias_summary.png",
        (
            "bias_law_v4/figures/over_under_rates.png"
        ): "fig11_over_under_rates.png",
        (
            "bias_law_v4/figures/separate_length_needle_slopes.png"
        ): "fig12_bias_length_needle_slopes.png",
    }
    copied: list[dict[str, Any]] = []
    for source_relative, destination_name in selected.items():
        source = analysis_root / source_relative
        destination = assets / destination_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "source_sha256": sha256(source),
                "destination": destination.name,
                "destination_sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    return copied


def figure_block(path: str, alt: str, caption: str) -> str:
    return (
        '<figure class="report-figure">'
        f'<img src="assets/{html.escape(path)}" alt="{html.escape(alt)}" '
        'loading="lazy">'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def build_model_sections(
    aggregates: dict[str, pd.DataFrame],
) -> str:
    model_summary = aggregates["model_summary"]
    mode_summary = aggregates["mode_summary"]
    cells = aggregates["cell_accuracy"]
    errors = aggregates["error_categories"]
    sections: list[str] = []
    for model in MODELS:
        modes = [
            mode
            for mode in PROMPT_MODES
            if not cells[
                (cells["model_label"] == model)
                & (cells["prompt_mode"] == mode)
            ].empty
        ]
        grids: list[str] = []
        for mode in modes:
            grids.append(
                '<details class="accuracy-detail" open>'
                f"<summary>{html.escape(PROMPT_LABELS[mode])}</summary>"
                "<p class=\"table-note\">每格合并 5 个 seed × 2 种 query order，"
                "因此 n=10；括号式分子/分母显示正确请求数。</p>"
                + accuracy_grid(cells, model, mode)
                + "</details>"
            )
        sections.append(
            f'<section class="model-section" id="model-{model.lower().replace(".", "-")}">'
            f"<h3>{html.escape(model)}</h3>"
            f"<p>{model_narrative(model, model_summary, mode_summary)}</p>"
            "<h4>模式汇总</h4>"
            + mode_summary_for_model(mode_summary, model)
            + "<h4>按 N、T 与 thinking/提示模式分层的准确率</h4>"
            + "".join(grids)
            + "<h4>互斥错误类型</h4>"
            + mode_error_table(errors, model)
            + "</section>"
        )
    return "".join(sections)


def model_summary_html(model_summary: pd.DataFrame) -> str:
    frame = model_summary.sort_values("accuracy", ascending=False).copy()
    return html_table(
        frame,
        [
            ("accuracy_rank", "排名", "int"),
            ("model_label", "模型", "str"),
            ("requests", "n", "int"),
            ("accuracy", "准确率", "pct"),
            ("parse_success_rate", "解析成功率", "pct"),
            ("truncation_rate", "截断率", "pct"),
            ("median_absolute_error_parsed", "解析后 |误差| 中位数", "float2"),
            ("mean_signed_bias_parsed", "解析后平均 bias", "float2"),
            ("median_signed_bias_parsed", "解析后 bias 中位数", "float2"),
        ],
    )


def overall_mode_html(mode_summary: pd.DataFrame) -> str:
    frame = mode_summary.copy()
    frame["mode_label"] = frame["prompt_mode"].map(PROMPT_LABELS)
    model_order = {model: index for index, model in enumerate(MODELS)}
    mode_order = {mode: index for index, mode in enumerate(PROMPT_MODES)}
    frame["_model_order"] = frame["model_label"].map(model_order)
    frame["_mode_order"] = frame["prompt_mode"].map(mode_order)
    frame = frame.sort_values(["_model_order", "_mode_order"])
    return html_table(
        frame,
        [
            ("model_label", "模型", "str"),
            ("mode_label", "模式", "str"),
            ("requests", "n", "int"),
            ("accuracy", "准确率", "pct"),
            ("parse_success_rate", "解析成功率", "pct"),
            ("truncation_rate", "截断率", "pct"),
            ("mean_absolute_error_parsed", "解析后平均 |误差|", "float2"),
            ("median_absolute_error_parsed", "解析后 |误差| 中位数", "float2"),
        ],
    )


def thinking_html(thinking: pd.DataFrame) -> str:
    frame = thinking.copy()
    frame["ci_text"] = frame.apply(
        lambda row: (
            f"[{100 * row['cluster_bootstrap_ci95_low']:+.1f}, "
            f"{100 * row['cluster_bootstrap_ci95_high']:+.1f}] pp"
        ),
        axis=1,
    )
    return html_table(
        frame,
        [
            ("model_label", "模型", "str"),
            ("paired_requests_per_mode", "每模式配对 n", "int"),
            ("direct_accuracy", "Direct / thinking 关", "pct"),
            ("native_thinking_accuracy", "Native thinking 开", "pct"),
            ("difference_percentage_points", "差值", "pp"),
            ("ci_text", "按 stimulus 聚类 bootstrap 95% CI", "str"),
        ],
    )


def enumeration_html(enumeration: pd.DataFrame) -> str:
    return html_table(
        enumeration,
        [
            ("model_label", "模型", "str"),
            ("wrong_enumeration_requests", "枚举模式错误 n", "int"),
            ("truncation", "截断", "int"),
            ("parse_failure_nontruncated", "非截断解析失败", "int"),
            ("parsed_undercount", "已解析少计", "int"),
            ("parsed_overcount", "已解析多计", "int"),
            ("missing_gold_pair_flag", "含漏掉 gold pair", "int"),
            ("hallucinated_pair_flag", "含幻觉 pair", "int"),
            ("duplicate_pair_flag", "含重复 pair", "int"),
            ("listed_total_mismatch_flag", "Total 与列表长度不符", "int"),
        ],
    )


def regression_replay_html(local: pd.DataFrame) -> str:
    labels = {
        "controls_only": "仅模型/模式/query-order 控制项",
        "density_model_fe": "仅 log density",
        "log_length_needles_model_fe": "共享 log L + log N",
        "log_length_needles_interaction_model_fe": "共享 log L + log N + 交互",
        "model_specific_length_needles_slopes": "每模型独立 L/N 斜率",
    }
    frame = local.copy()
    frame["candidate_label"] = frame["candidate"].map(labels)
    return html_table(
        frame,
        [
            ("candidate_label", "候选式", "str"),
            ("n_parameters", "参数数", "int"),
            ("local_cv_log_loss_mean", "本地 OOF log loss", "float4"),
            ("archived_cv_log_loss_mean", "归档值", "float4"),
            ("absolute_log_loss_difference", "|本地−归档|", "float4"),
            ("local_cv_brier", "OOF Brier", "float4"),
            ("local_cv_ece", "OOF ECE", "float4"),
            ("local_cv_calibration_slope", "校准斜率", "float3"),
        ],
    )


def shared_coefficients_html(coefficients: pd.DataFrame) -> str:
    frame = coefficients[
        coefficients["feature"].isin(["log_length", "log_needles"])
    ].copy()
    frame["feature_label"] = frame["feature"].map(
        {
            "log_length": "log₂(L/5000)",
            "log_needles": "log₂(N/5)",
        }
    )
    frame["bootstrap_ci"] = frame.apply(
        lambda row: (
            f"[{row['bootstrap_ci95_low']:.4f}, "
            f"{row['bootstrap_ci95_high']:.4f}]"
        ),
        axis=1,
    )
    return html_table(
        frame,
        [
            ("feature_label", "项", "str"),
            ("estimate", "系数", "float4"),
            ("bootstrap_ci", "stimulus-cluster bootstrap 95% CI", "str"),
            ("odds_ratio", "每翻倍 odds ratio", "float3"),
        ],
    )


def unified_parameters_html(parameters: pd.DataFrame) -> str:
    frame = parameters.copy()
    frame["baseline_ci"] = frame.apply(
        lambda row: (
            f"[{row['baseline_probability_L0_N0_direct_query_first_ci95_low']:.3f}, "
            f"{row['baseline_probability_L0_N0_direct_query_first_ci95_high']:.3f}]"
        ),
        axis=1,
    )
    frame["length_ci"] = frame.apply(
        lambda row: (
            f"[{row['length_parameter_ci95_low']:.3f}, "
            f"{row['length_parameter_ci95_high']:.3f}]"
        ),
        axis=1,
    )
    frame["needle_ci"] = frame.apply(
        lambda row: (
            f"[{row['needle_parameter_ci95_low']:.3f}, "
            f"{row['needle_parameter_ci95_high']:.3f}]"
        ),
        axis=1,
    )
    frame["odds_length_doubling"] = 2.0 ** (-frame["length_parameter"])
    frame["odds_needle_doubling"] = 2.0 ** (-frame["needle_parameter"])
    return html_table(
        frame,
        [
            ("model_label", "模型", "str"),
            (
                "baseline_probability_L0_N0_direct_query_first",
                "p(5000,5) baseline",
                "pct",
            ),
            ("baseline_ci", "baseline 95% CI", "str"),
            ("amplitude", "Aₘ", "float3"),
            ("length_parameter", "rₘ（长度阶）", "float3"),
            ("length_ci", "rₘ 95% CI", "str"),
            ("needle_parameter", "sₘ（needle 阶）", "float3"),
            ("needle_ci", "sₘ 95% CI", "str"),
            ("odds_length_doubling", "L 翻倍后的 odds 倍率", "float3"),
            ("odds_needle_doubling", "N 翻倍后的 odds 倍率", "float3"),
        ],
    )


def absolute_error_parameters_html(parameters: pd.DataFrame) -> str:
    frame = parameters.copy()
    frame["length_ci"] = frame.apply(
        lambda row: (
            f"[{row['length_parameter_ci95_low']:.3f}, "
            f"{row['length_parameter_ci95_high']:.3f}]"
        ),
        axis=1,
    )
    frame["needle_ci"] = frame.apply(
        lambda row: (
            f"[{row['needle_parameter_ci95_low']:.3f}, "
            f"{row['needle_parameter_ci95_high']:.3f}]"
        ),
        axis=1,
    )
    return html_table(
        frame,
        [
            ("model_label", "模型", "str"),
            ("amplitude_B", "Bₘ", "float3"),
            ("length_parameter", "uₘ（长度阶）", "float3"),
            ("length_ci", "uₘ 95% CI", "str"),
            ("needle_parameter", "vₘ（needle 阶）", "float3"),
            ("needle_ci", "vₘ 95% CI", "str"),
        ],
    )


def error_examples_html(examples: pd.DataFrame) -> str:
    parts = [
        '<div class="table-wrap"><table class="data-table examples"><thead><tr>',
    ]
    headers = [
        "模型",
        "类型",
        "模式",
        "T / N",
        "gold → prediction",
        "request_id",
        "输出节选",
    ]
    for label in headers:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in examples.itertuples():
        prediction = (
            "未解析"
            if pd.isna(row.predicted_count)
            else f"{row.predicted_count:g}"
        )
        parts.append("<tr>")
        parts.append(f"<td>{html.escape(row.model_label)}</td>")
        parts.append(
            f"<td>{html.escape(ERROR_LABELS[row.error_category])}</td>"
        )
        parts.append(
            f"<td>{html.escape(PROMPT_LABELS[row.prompt_mode])}</td>"
        )
        parts.append(
            f"<td>{int(row.target_passage_tokens):,} / "
            f"{int(row.num_needles)}</td>"
        )
        parts.append(
            f"<td>{int(row.gold_count)} → {html.escape(prediction)}</td>"
        )
        parts.append(f"<td><code>{html.escape(row.request_id)}</code></td>")
        parts.append(
            f"<td class=\"excerpt\">{html.escape(row.raw_output_excerpt)}</td>"
        )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def source_integrity_html(audit: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    rows = pd.DataFrame(sources)
    table = html_table(
        rows,
        [
            ("model", "模型", "str"),
            ("rows", "行数", "int"),
            ("bytes", "字节", "int"),
            ("sha256", "SHA256", "str"),
        ],
    )
    return (
        "<p>输入审计状态：<strong>"
        + html.escape(audit["status"].upper())
        + "</strong>；请求数 "
        + f"{audit['requests']:,}，唯一 request_id "
        + f"{audit['unique_request_ids']:,}，唯一 stimulus_id "
        + f"{audit['unique_stimuli']:,}。固定 stimuli SHA256："
        + f"<code>{html.escape(audit['stimuli_sha256'])}</code>；"
        + "冻结代码 commit："
        + f"<code>{EXPECTED_COMMIT}</code>。</p>"
        + table
    )


def build_html_report(
    *,
    output: Path,
    frame: pd.DataFrame,
    sources: list[dict[str, Any]],
    audit: dict[str, Any],
    aggregates: dict[str, pd.DataFrame],
    thinking: pd.DataFrame,
    examples: pd.DataFrame,
    local_regression: pd.DataFrame,
    replay_summary: dict[str, Any],
    archived: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
) -> str:
    total_accuracy = float(frame["exact_correct"].mean())
    parse_rate = float(frame["parse_success"].mean())
    trunc_rate = float(frame["truncated"].mean())
    format_failure_rate = float(frame["format_failure"].mean())
    parsed_n = int(frame["parse_success"].sum())
    model_summary = aggregates["model_summary"]
    mode_summary = aggregates["mode_summary"]
    best_model = model_summary.sort_values("accuracy", ascending=False).iloc[0]

    shared = archived["shared_coefficients"]
    shared_length = shared[shared["feature"] == "log_length"].iloc[0]
    shared_needles = shared[shared["feature"] == "log_needles"].iloc[0]
    unified_summary = metrics["unified_summary"]
    no_size_summary = metrics["no_size_summary"]

    regression_table = regression_replay_html(local_regression)
    model_sections = build_model_sections(aggregates)
    generated = utc_now()

    css = """
:root {
  --ink: #182026;
  --muted: #5f6b73;
  --paper: #ffffff;
  --wash: #f5f7f8;
  --line: #d7dde1;
  --accent: #264653;
  --accent-2: #2a9d8f;
  --warn: #b55d2d;
  --max: 1180px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  line-height: 1.66;
}
header {
  background: linear-gradient(135deg, #18313b 0%, #264653 56%, #2a6670 100%);
  color: white;
  padding: 58px 24px 46px;
}
.header-inner, main, .footer-inner { max-width: var(--max); margin: 0 auto; }
h1 { margin: 0 0 12px; font-size: clamp(2rem, 5vw, 3.5rem); line-height: 1.12; }
.subtitle { max-width: 900px; font-size: 1.08rem; opacity: .9; }
.meta { margin-top: 22px; font-size: .9rem; opacity: .78; }
nav {
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,.96);
  position: sticky;
  top: 0;
  z-index: 20;
}
nav .nav-inner {
  max-width: var(--max);
  margin: 0 auto;
  padding: 10px 20px;
  display: flex;
  gap: 18px;
  overflow-x: auto;
  white-space: nowrap;
}
nav a { color: var(--accent); text-decoration: none; font-size: .9rem; }
main { padding: 36px 22px 72px; }
section { margin: 0 0 58px; scroll-margin-top: 72px; }
h2 { font-size: 1.75rem; margin: 0 0 18px; color: var(--accent); }
h3 { font-size: 1.35rem; margin: 28px 0 12px; }
h4 { font-size: 1.05rem; margin: 22px 0 10px; }
p { max-width: 94ch; }
.lead { font-size: 1.08rem; }
.muted, .table-note { color: var(--muted); font-size: .91rem; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin: 24px 0 30px;
}
.kpi {
  border: 1px solid var(--line);
  border-top: 4px solid var(--accent-2);
  padding: 18px 18px 16px;
  background: var(--paper);
}
.kpi .value { font-size: 1.9rem; font-weight: 650; line-height: 1.1; }
.kpi .label { margin-top: 7px; color: var(--muted); font-size: .9rem; }
.callout {
  border-left: 4px solid var(--accent-2);
  background: #f0f7f5;
  padding: 16px 20px;
  margin: 20px 0;
}
.callout.warn { border-left-color: var(--warn); background: #fff7f2; }
.formula {
  font-family: "Cambria Math", "Times New Roman", serif;
  font-size: 1.08rem;
  background: var(--wash);
  border: 1px solid var(--line);
  padding: 14px 18px;
  overflow-x: auto;
}
.table-wrap { overflow-x: auto; margin: 12px 0 22px; }
table { width: 100%; border-collapse: collapse; font-size: .88rem; }
caption { text-align: left; font-weight: 600; margin-bottom: 8px; }
th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: right; vertical-align: top; }
th { background: var(--wash); font-weight: 650; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
.data-table td:nth-child(1), .data-table td:nth-child(2) { text-align: left; }
.accuracy-grid { min-width: 560px; }
.accuracy-grid th, .accuracy-grid td { text-align: center; }
.accuracy-grid td strong { display: block; font-size: .95rem; }
.accuracy-grid td span { color: var(--muted); font-size: .78rem; }
.examples { min-width: 1050px; }
.examples .excerpt { min-width: 320px; text-align: left; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: .83em; overflow-wrap: anywhere; }
.report-figure { margin: 30px auto 38px; max-width: 1080px; }
.report-figure img { width: 100%; height: auto; border: 1px solid var(--line); background: white; }
figcaption { color: var(--muted); font-size: .9rem; margin-top: 10px; }
.model-section {
  border-top: 2px solid var(--line);
  padding-top: 6px;
  margin-bottom: 48px;
}
details.accuracy-detail {
  border: 1px solid var(--line);
  margin: 12px 0;
  padding: 0 14px 10px;
}
summary { cursor: pointer; font-weight: 650; padding: 12px 0; color: var(--accent); }
.two-col {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 24px;
}
ul.tight li { margin-bottom: 7px; }
footer { background: var(--wash); border-top: 1px solid var(--line); padding: 28px 20px; color: var(--muted); }
@media (max-width: 760px) {
  .two-col { grid-template-columns: 1fr; }
  header { padding-top: 38px; }
  th, td { padding: 7px 8px; }
}
@media print {
  nav { display: none; }
  body { font-size: 10.5pt; }
  header { background: white; color: black; padding: 20px 0; }
  main { padding: 0; }
  section { break-inside: avoid; }
  details { break-inside: avoid; }
  details > * { display: block !important; }
}
"""

    head = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Realistic CoT NiaH Count：八模型实验结果与 empirical law</title>
  <style>{css}</style>
</head>
<body>
<header>
  <div class="header-inner">
    <h1>Realistic CoT NiaH Count</h1>
    <div class="subtitle">八个模型的完整分层结果、错误归因与 empirical-law 回归报告</div>
    <div class="meta">6,300 个正式请求 · 冻结实验 commit {EXPECTED_COMMIT[:12]} · 报告生成于 {html.escape(generated)}</div>
  </div>
</header>
<nav aria-label="报告目录">
  <div class="nav-inner">
    <a href="#summary">摘要</a>
    <a href="#setup">实验设定</a>
    <a href="#overall">总体结果</a>
    <a href="#models">逐模型明细</a>
    <a href="#errors">错误分析</a>
    <a href="#laws">Empirical law</a>
    <a href="#bias">绝对误差与 bias</a>
    <a href="#limits">限制</a>
    <a href="#repro">复现与完整性</a>
  </div>
</nav>
<main>
"""

    summary_section = f"""
<section id="summary">
  <h2>执行摘要</h2>
  <p class="lead">本报告从八个 <code>main/requests.jsonl</code> 重新读取全部请求，保留解析失败、格式失败和截断为准确率失败；没有删点、补值或选择性排除模型。分层统计在本地独立重算，主要 logit 回归也在本地重放，并与归档回归逐项核对。</p>
  <div class="kpi-grid">
    <div class="kpi"><div class="value">6,300</div><div class="label">正式请求</div></div>
    <div class="kpi"><div class="value">{percent(total_accuracy)}</div><div class="label">总体 exact-count 准确率（3,246/6,300）</div></div>
    <div class="kpi"><div class="value">{percent(parse_rate)}</div><div class="label">可解析数值输出（{parsed_n:,}/6,300）</div></div>
    <div class="kpi"><div class="value">{percent(trunc_rate)}</div><div class="label">输出截断率</div></div>
    <div class="kpi"><div class="value">{html.escape(str(best_model['model_label']))}</div><div class="label">总体最佳模型：{percent(best_model['accuracy'])}</div></div>
  </div>
  <div class="callout">
    <strong>核心结论。</strong> 准确率不能只用 needle density <em>N/T</em> 概括：分开的长度项与 needle 数量项明显优于 density-only。最简洁且验证表现最好的统一形式是每个模型具有不同参数的 Hill/power law；needle 阶在八个模型上均为正且更稳定，长度阶因为只有 3 个长度水平而不如 needle 阶精确。
  </div>
  <ul class="tight">
    <li><strong>模型表现：</strong>Qwen3-32B 总体最高（76.2%），其后为 Gemma4-E4B（69.9%）和 Qwen3-8B（68.0%）。</li>
    <li><strong>thinking：</strong>对支持原生 thinking 的 Qwen/Gemma，native-thinking 模式相对 direct 均提高准确率；但同时改变了解码预算/温度，因此应解释为“完整模式差异”，不能单独归因为隐藏推理开关。</li>
    <li><strong>主要回归：</strong>共享斜率模型中，长度翻倍的 odds ratio 为 {float(shared_length['odds_ratio']):.3f}，needle 数翻倍为 {float(shared_needles['odds_ratio']):.3f}；两者的 stimulus-cluster bootstrap 95% CI 均不跨 0。</li>
    <li><strong>错误机制：</strong>Gemma4-12B 的 direct 模式主要被 97.7% 截断拖累；Llama3.2-3B 的 enumeration 模式既大量无法解析，也出现极端多计。枚举模式的平均绝对误差常被少数巨大 over-count 主导，因此报告同时给出中位数和 signed-bias 稳健统计。</li>
  </ul>
</section>
"""

    setup_section = f"""
<section id="setup">
  <h2>实验设定与统计口径</h2>
  <div class="two-col">
    <div>
      <h3>注册网格</h3>
      <ul class="tight">
        <li>Haystack 长度 <strong>T</strong>：2,000、5,000、10,000 个 canonical passage tokens。</li>
        <li>Needle 数量 <strong>N</strong>：1、2、3、4、5、6、8、10、20、30。</li>
        <li>Seed：1234–1238，共 5 个；因此有 3×10×5=150 个 master stimuli。</li>
        <li>Query order：任务在 passage 前（query_first）或后（query_last）。</li>
        <li>模型：Qwen3 三种规模、Gemma4 两种、OLMo-Hybrid-7B、Llama3.1-8B、Llama3.2-3B。</li>
      </ul>
    </div>
    <div>
      <h3>提示与 thinking</h3>
      <ul class="tight">
        <li><strong>direct：</strong>thinking 关闭，只要求最终一行 <code>Total: &lt;integer&gt;</code>；max tokens=64，temperature=0。</li>
        <li><strong>enumeration：</strong>thinking 关闭，要求逐条列出 city-score record 再给 Total；max tokens=1,536，temperature=0。</li>
        <li><strong>native_thinking：</strong>与 direct 使用相同任务文字，但 chat template 打开原生 thinking；max tokens=4,096。Qwen temperature=0.6，Gemma temperature=1.0。</li>
        <li>Llama/OLMo 不支持本实验定义下的 native thinking，因此每个模型为 600 请求；Qwen/Gemma 各 900 请求。</li>
      </ul>
    </div>
  </div>
  <h3>定义</h3>
  <div class="formula"><strong>Exact accuracy</strong> = 1[predicted_count = N 且最终答案可按冻结 parser 解析]。解析失败、未按格式、以及截断均记为 0。</div>
  <div class="formula"><strong>Signed bias</strong> = predicted_count − N；正值表示多计，负值表示少计。<strong>Absolute error</strong> = |predicted_count − N|。这两个量只在成功解析出数值的 {parsed_n:,} 条输出上定义；未解析请求不会被伪造为 0。</div>
  <div class="formula"><strong>Needle density</strong> d = 1000N/T，单位为 needles per 1k canonical passage tokens。T 是由 Qwen3-8B canonical tokenizer 定义的目标 passage 长度，而不是每个模型最终输入总 token 数。</div>
  <p class="muted">“thinking 开/关”在本报告中严格由 <code>prompt_mode == native_thinking</code> 定义。Enumeration 虽然要求显式列举，但 chat template 的原生 thinking 仍关闭；因此它单独列为第三种提示模式，避免把输出格式变化与 thinking 混在一起。</p>
</section>
"""

    overall_section = f"""
<section id="overall">
  <h2>总体结果</h2>
  <h3>八模型汇总</h3>
  {model_summary_html(model_summary)}
  <p class="table-note">准确率分母包含全部注册请求；绝对误差与 bias 列只对成功解析出的数值输出计算。平均 bias 对极端多计非常敏感，必须与中位数共同阅读。</p>
  {figure_block(
      "fig01_model_error_composition.png",
      "Eight horizontal stacked bars showing correct, undercount, overcount, parse failure, and truncation shares for each model.",
      "<strong>图 1｜模型的互斥结果构成。</strong> 横轴是该模型全部注册请求的比例；每条请求只进入一个颜色区间：正确、已解析少计、已解析多计、非截断格式/解析失败、截断或其他。颜色总和为 100%。",
  )}
  {figure_block(
      "fig03_mode_accuracy_heatmap.png",
      "Heatmap of exact-count accuracy for each model and prompt/thinking mode.",
      "<strong>图 2｜模型 × 提示/thinking 模式准确率。</strong> 行为模型，列为 direct-off、enumeration-off 和 native-thinking-on；颜色与格内数字均表示 exact-count accuracy。N/A 表示该模型未注册该模式。",
  )}
  <h3>逐模式汇总</h3>
  {overall_mode_html(mode_summary)}
  <h3>原生 thinking 的配对比较</h3>
  <p>这里仅在支持 native thinking 的五个模型上，将相同 stimulus、相同 query order 的 direct 与 native-thinking 请求配对。置信区间按完整 stimulus 聚类 bootstrap；它衡量完整模式的预测差异，不是纯 thinking 因果效应。</p>
  {thinking_html(thinking)}
  {figure_block(
      "fig02_native_thinking_comparison.png",
      "Paired accuracy comparison between direct thinking-off and native-thinking-on modes for five capable models.",
      "<strong>图 3｜相同 direct 任务文字下，native thinking 开与关的准确率。</strong> 横轴为支持该模式的模型，纵轴为 exact-count accuracy；标注为开 − 关的百分点差。原生 thinking 同时使用更长输出预算和不同采样温度，故不能把差值只归因于内部推理。",
  )}
</section>
"""

    models_section = f"""
<section id="models">
  <h2>逐模型、逐 N/T/thinking 的准确率</h2>
  <p>下面每个模型先给总体与模式摘要，再给完整 10×3 准确率网格。每格固定合并 5 个 seed 与 2 个 query order，因此 n=10。绿色越深表示准确率越高，红色越深表示越低；具体分子/分母始终写在格内。</p>
  {model_sections}
</section>
"""

    errors_section = f"""
<section id="errors">
  <h2>错误情况与原因</h2>
  <h3>互斥一级分类</h3>
  <ol>
    <li><strong>截断：</strong>生成达到长度限制或被标记 truncated；即便出现部分列表，只要没有正确最终结果仍计失败。</li>
    <li><strong>格式/解析失败：</strong>非截断，但冻结 parser 没有抽取到规定的最终整数，通常是缺少 <code>Total:</code>、输出额外格式或生成了无法解析的文本。</li>
    <li><strong>少计/多计：</strong>成功解析整数，但分别小于/大于真实 N。</li>
    <li><strong>正确：</strong>解析出的 Total 与真实 N 完全一致。</li>
  </ol>
  <div class="callout warn"><strong>为什么平均绝对误差会很大：</strong>多个 enumeration 模式虽然大多数请求正确或误差很小，但少数输出把无关编号/重复内容当成计数并产生数百乃至更大的 Total。于是均值被长尾支配，而中位数仍可能为 0。报告不会删除这些点，因为它们是真实模型失败。</div>
  <h3>枚举模式的可观测机制标签</h3>
  <p>冻结 evaluator 对 enumeration 输出还能检测 gold pair 是否遗漏、是否出现幻觉 pair、是否重复列出 pair，以及列表长度是否与 Total 一致。下表只统计 enumeration 中的错误请求；后四列是可重叠的机制标签，因此不能相加当作总数。</p>
  {enumeration_html(aggregates['enumeration_mechanisms'])}
  {figure_block(
      "fig11_over_under_rates.png",
      "Under-count, exact, and over-count rates by model among parsed outputs.",
      "<strong>图 4｜成功解析后的少计/正确/多计比例。</strong> 横轴为解析输出中的比例，行是模型；未解析与截断请求不在该条件分布中，因此必须与图 1 的失败比例一起解释。",
  )}
  <h3>代表性原始错误</h3>
  <p>每个模型最多保留一条截断、一条非截断解析失败、一条最大少计和一条最大多计作为审计样例。输出仅显示首尾压缩节选；完整原文仍在原始 JSONL 中。</p>
  <details><summary>展开错误样例表</summary>{error_examples_html(examples)}</details>
</section>
"""

    laws_section = f"""
<section id="laws">
  <h2>Empirical law：猜想、验证与拟合</h2>
  <h3>候选猜想</h3>
  <ol>
    <li><strong>Density-only：</strong>logit(p) 只依赖 log(N/T)。这是最强的“由密度完全决定”假设。</li>
    <li><strong>分离的 log L 与 log N：</strong>logit(p)=控制项+b<sub>L</sub>log₂(L/5000)+b<sub>N</sub>log₂(N/5)。它允许长度和 needle 数量有不同阶。</li>
    <li><strong>带交互的 response surface：</strong>在上一式加入 log L × log N，检验一个维度的效应是否随另一个维度改变。</li>
    <li><strong>统一形式、每模型不同参数：</strong>八个模型共享同一 Hill/power 函数族，但 A、长度阶 r 和 needle 阶 s 各不相同。</li>
    <li><strong>原始坐标 exponential 与其他 link：</strong>作为有边界的替代函数族，与 power/Hill 一起用 held-out 数据比较。</li>
  </ol>
  <h3>本地回归重放</h3>
  <p>本地从 6,300 条原始请求重新构建设计矩阵，执行 5 个 leave-one-seed-out folds。同 seed 的所有模型、模式、query order 和 N/T 条件共同进入一个 held-out fold，以避免同源 stimulus 泄漏。本地 log loss 与归档结果最大绝对差为 <strong>{replay_summary['max_absolute_log_loss_difference_vs_archived']:.2e}</strong>，通过数值复现阈值。</p>
  {regression_table}
  {figure_block(
      "fig04_accuracy_law_cv.png",
      "Held-out log loss with standard error for five accuracy-law candidates.",
      "<strong>图 5｜准确率候选 law 的 leave-one-seed-out 比较。</strong> 横轴为 held-out Bernoulli log loss（越低越好），误差线为 fold mean 的约 95% 区间。Density-only 明显落后于分开的 log L + log N；每模型斜率进一步改善预测。",
  )}
  <h3>共享长度/needle 阶</h3>
  <p>不使用模型大小，只把模型 identity、prompt mode 和 query order 当作 categorical nuisance control。对数坐标系下，系数直接描述自变量翻倍时 log-odds 的变化。</p>
  {shared_coefficients_html(archived['shared_coefficients'])}
  <p>两个 cluster-bootstrap 区间都不跨 0，因此在注册网格内，长度与 needle 数量各自的负向关联都有统计支持。Density-only 的 OOF log loss 为 {float(no_size_summary['accuracy']['interpretive_relation']['density_only_cv_log_loss']):.4f}，而分开 L/N 的共享斜率模型为 {float(no_size_summary['accuracy']['interpretive_relation']['cv_log_loss']):.4f}；所以 <em>N/T</em> 不能充分折叠两维。</p>
  {figure_block(
      "fig05_pooled_accuracy_length_needles.png",
      "Pooled exact accuracy versus needle count for passage lengths 2000, 5000, and 10000 tokens.",
      "<strong>图 6｜观测准确率随 N 与 T 的关系。</strong> 横轴是 needle 数 N，纵轴是 pooled exact accuracy，三条线分别为 2k、5k、10k canonical passage tokens；点汇总所有模型、模式、query order 与 seed。该图只描述总体趋势，不能替代带模型/模式控制项的回归。",
  )}
  <h3>最终推荐的统一参数 law</h3>
  <div class="formula">p<sub>m</sub>(L,N,q,o) = 1 / [1 + A<sub>m</sub>(L/5000)<sup>r<sub>m</sub></sup>(N/5)<sup>s<sub>m</sub></sup> · exp(−δ<sub>q</sub>−γ<sub>o</sub>)]</div>
  <p>m 表示模型；q 和 o 分别是 prompt mode 与 query order 的共享 nuisance 修正。在 direct/query_first baseline 下，δ=γ=0。因为 odds=p/(1−p)，长度翻倍会把 odds 乘以 2<sup>−rₘ</sup>，needle 数翻倍会乘以 2<sup>−sₘ</sup>。Aₘ 控制基准难度，rₘ 与 sₘ 就是用户关心的“分别成多少阶”。</p>
  <p>模型选择使用四套完整 OOF：留一 seed、留一 needle level、留一 length level、以及 blocked (L,N) cells；selection score 是四种 log loss 的等权平均。选中的 Hill/power model-specific law 得分 <strong>{float(unified_summary['selected_unified_law']['selection_score']):.6f}</strong>。带 L×N 交互的上限模型只改善约 0.00034，却增加 8 个参数且 BIC 更差，因此保留更简洁的可分离形式。</p>
  {unified_parameters_html(archived['unified_parameters'])}
  {figure_block(
      "fig07_unified_law_parameters.png",
      "Model-specific length and needle power orders with clustered bootstrap confidence intervals.",
      "<strong>图 7｜统一 Hill/power law 的模型特异参数。</strong> 左图横轴为长度阶 rₘ，右图为 needle 阶 sₘ；点是全数据估计，横线是按完整 stimulus 聚类的 95% bootstrap CI，虚线 0 表示无相应维度效应。只有 3 个长度水平，因此 rₘ 的区间普遍更宽。",
  )}
  {figure_block(
      "fig08_unified_law_surfaces.png",
      "Predicted exact accuracy curves versus needle count at three passage lengths for all eight models.",
      "<strong>图 8｜推荐 law 在 direct/query_first baseline 下的预测曲面切片。</strong> 每个小图是一种模型；横轴为 N（log₂ 刻度），纵轴为预测 exact accuracy，三条线对应 T=2k/5k/10k。图只在实验观测范围 1≤N≤30、2k≤T≤10k 内解释。",
  )}
  {figure_block(
      "fig09_heldout_observed_predicted.png",
      "Held-out observed versus predicted cell accuracy for the selected unified law.",
      "<strong>图 9｜blocked-cell held-out 观测值与预测值。</strong> 横轴为 law 给出的 OOF 预测概率，纵轴为相应 held-out cell 的观测准确率；45° 线表示完美校准。偏离反映模型没有捕获的 cell heterogeneity。",
  )}
</section>
"""

    bias_section = f"""
<section id="bias">
  <h2>绝对误差与 signed bias</h2>
  <h3>条件绝对误差 law</h3>
  <p>绝对误差只在 {parsed_n:,}/{len(frame):,}（{percent(parse_rate)}）条成功解析输出上拟合。选中的 log/power 坐标写为：</p>
  <div class="formula">E[log(1+|error|) | parsed] ≈ log B<sub>m</sub> + u<sub>m</sub>log(L/5000) + v<sub>m</sub>log(N/5) + prompt/order controls</div>
  <p>这不是全请求准确率 law，而是“已经成功给出数值时，错多少”的条件诊断。未解析与截断仍在 primary accuracy 中算失败，不能用这一节掩盖。</p>
  {absolute_error_parameters_html(archived['absolute_error_parameters'])}
  <p>needle 阶 vₘ 对多数模型为正且区间更稳定；长度阶 uₘ 在 Qwen3-32B、Llama3.2-3B 等模型上跨 0。由于少量 enumeration 极端 over-count，count-unit held-out MAE 仍约 36，而 log1p 目标更能刻画典型量级。</p>
  <h3>Signed bias</h3>
  <p>原始平均 bias 是字面上的统计偏差，但被少数巨大正向异常值严重拉高。总体 parsed-output mean bias 为 {number(frame['signed_error'].mean())}，中位数为 {number(frame['signed_error'].median())}。因此报告同时给出原始均值、10% trimmed mean、中位数、[-30,30] capped bias 与 asinh(bias)；bias-law v4 用 500 次 stimulus-cluster bootstrap，并对八个模型的检验做 Holm 调整。</p>
  {figure_block(
      "fig10_model_bias_summary.png",
      "Mean signed bias with confidence interval, 10 percent trimmed mean, and median for each model.",
      "<strong>图 10｜平均 bias、10% trimmed mean 与中位数。</strong> 横轴单位是计数（prediction − truth），0 为无方向偏差。蓝色圆点及区间是原始平均 bias 与 95% CI，方块为 trimmed mean，叉号为中位数。三者差异直接显示极端正向 outlier 的影响。",
  )}
  {figure_block(
      "fig12_bias_length_needle_slopes.png",
      "Separate log-length and log-needle slopes for raw and asinh signed bias by model.",
      "<strong>图 11｜bias 对长度与 needle 数量的独立斜率诊断。</strong> 上排纵轴目标为原始 signed bias（count units），下排为 asinh(bias)；左列横轴是 log-length slope，右列是 log-needle slope。点为全数据估计，横线为 cluster-bootstrap 区间，0 虚线表示没有方向关联。原始均值斜率对极端 outlier 很敏感，优先结合下排稳健结果。",
  )}
  <div class="callout warn"><strong>结论：</strong>可以得到 bias 作为纵坐标的结果，但不存在一个对所有模型都稳定的统一 signed-bias 方向。典型输出的中位 bias 多接近 0；部分模型/枚举模式出现少数巨大多计，使 raw mean 显著为正。因而 bias law 更适合作为错误机制诊断，而不是替代准确率 law。</div>
</section>
"""

    limitations_section = """
<section id="limits">
  <h2>解释边界与限制</h2>
  <ul class="tight">
    <li><strong>长度水平只有 3 个：</strong>虽然 needle 有 10 个水平，T 只有 2k/5k/10k；因此“长度是幂律还是其他平滑函数”的辨识力有限，任何 10k 以外外推都不可靠。</li>
    <li><strong>模式没有完全交叉：</strong>Llama/OLMo 没有 native-thinking 条件；模型 family、tokenizer、architecture 与 thinking availability 也没有正交化。</li>
    <li><strong>thinking 比较含解码差异：</strong>native-thinking 同时更改 max tokens、temperature、top-p/top-k。观察差值是整个推理模式的关联，非 isolated causal effect。</li>
    <li><strong>误差量级是条件结果：</strong>absolute error 和 signed bias 只在 parsed outputs 上定义；模型若经常不产出可解析答案，条件误差看起来可能很小，却不代表总体性能好。</li>
    <li><strong>均值被长尾支配：</strong>极端 over-count 是有效失败，不能删掉；但报告 raw mean 时必须并列中位数、trimmed/capped/asinh 稳健版本。</li>
    <li><strong>预测而非因果：</strong>所有 law 都是注册网格内的 empirical response surface，不能推断模型内部因果机制。</li>
  </ul>
</section>
"""

    repro_section = f"""
<section id="repro">
  <h2>复现、文件与完整性</h2>
  <p>报告目录同时包含：</p>
  <ul class="tight">
    <li><code>report.html</code>：本页面；</li>
    <li><code>tables/</code>：逐请求 compact table、模型/模式/单元格准确率、错误分类、thinking 配对比较、本地回归重放与归档参数表；</li>
    <li><code>assets/</code>：报告中全部图；</li>
    <li><code>scripts/build_report.py</code>：从原始 JSONL 重建报告的脚本；</li>
    <li><code>analysis_manifest.json</code> 与 <code>SHA256SUMS.tsv</code>：来源、软件版本与产物校验。</li>
  </ul>
  <p>本地回归使用 Python {platform.python_version()}、NumPy {np.__version__}、pandas {pd.__version__}、SciPy {scipy.__version__}、Matplotlib {matplotlib.__version__}。所有输入 request JSONL 与冻结 stimuli 在生成报告前重新计算 SHA256。</p>
  <h3>输入审计</h3>
  {source_integrity_html(audit, sources)}
</section>
"""

    footer = f"""
</main>
<footer><div class="footer-inner">Realistic_CoT_NiaH_Count · 本地、离线、可审计报告 · 生成时间 {html.escape(generated)}</div></footer>
</body>
</html>
"""
    report = (
        head
        + summary_section
        + setup_section
        + overall_section
        + models_section
        + errors_section
        + laws_section
        + bias_section
        + limitations_section
        + repro_section
        + footer
    )
    destination = output / "report.html"
    destination.write_text(report, encoding="utf-8")
    return report


def copy_reproduction_sources(
    analysis_root: Path, output_scripts: Path, current_script: Path
) -> list[dict[str, Any]]:
    sources = {
        (
            analysis_root
            / "empirical_law_no_model_size_v1"
            / "empirical_law_no_size.py"
        ): "archived_empirical_law_no_size.py",
        (
            analysis_root
            / "unified_parametric_law_v1"
            / "unified_parametric_law.py"
        ): "archived_unified_parametric_law.py",
        (
            analysis_root / "bias_law_v4" / "bias_law.py"
        ): "archived_bias_law_v4.py",
        current_script: "build_report.py",
    }
    copied: list[dict[str, Any]] = []
    for source, destination_name in sources.items():
        destination = output_scripts / destination_name
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "destination": destination.name,
                "sha256": sha256(destination),
            }
        )
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Path to six_models_formal_20260723T194300Z",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report directory: {output}"
        )
    assets = output / "assets"
    tables = output / "tables"
    scripts = output / "scripts"
    logs = output / "logs"
    for directory in [output, assets, tables, scripts, logs]:
        directory.mkdir(parents=True, exist_ok=True)

    frame, sources = load_requests(run_root)
    audit = validate_frame(frame, run_root)
    aggregates = aggregate_tables(frame)
    thinking = paired_thinking_comparison(frame)
    examples = select_error_examples(frame)

    analysis_root = run_root / "analysis"
    archived = {
        "accuracy_candidates": pd.read_csv(
            analysis_root
            / "empirical_law_no_model_size_v1"
            / "tables"
            / "accuracy_candidate_comparison.csv"
        ),
        "shared_coefficients": pd.read_csv(
            analysis_root
            / "empirical_law_no_model_size_v1"
            / "tables"
            / "accuracy_interpretive_coefficients.csv"
        ),
        "per_model_accuracy_slopes": pd.read_csv(
            analysis_root
            / "empirical_law_no_model_size_v1"
            / "tables"
            / "per_model_accuracy_slopes.csv"
        ),
        "functional_forms": pd.read_csv(
            analysis_root
            / "unified_parametric_law_v1"
            / "tables"
            / "functional_form_comparison.csv"
        ),
        "unified_parameters": pd.read_csv(
            analysis_root
            / "unified_parametric_law_v1"
            / "tables"
            / "model_law_parameters.csv"
        ),
        "absolute_error_parameters": pd.read_csv(
            analysis_root
            / "unified_parametric_law_v1"
            / "tables"
            / "absolute_error_law_parameters.csv"
        ),
        "bias_candidates": pd.read_csv(
            analysis_root
            / "bias_law_v4"
            / "tables"
            / "bias_candidate_comparison.csv"
        ),
        "bias_descriptive": pd.read_csv(
            analysis_root
            / "bias_law_v4"
            / "tables"
            / "model_bias_descriptive.csv"
        ),
    }
    metrics = {
        "no_size_summary": json.loads(
            (
                analysis_root
                / "empirical_law_no_model_size_v1"
                / "metrics"
                / "summary.json"
            ).read_text(encoding="utf-8")
        ),
        "unified_summary": json.loads(
            (
                analysis_root
                / "unified_parametric_law_v1"
                / "metrics"
                / "summary.json"
            ).read_text(encoding="utf-8")
        ),
        "bias_summary": json.loads(
            (
                analysis_root / "bias_law_v4" / "metrics" / "summary.json"
            ).read_text(encoding="utf-8")
        ),
    }

    local_regression, local_slopes, replay_summary = replay_regressions(
        frame, archived["accuracy_candidates"]
    )

    frame.drop(columns=["raw_output_excerpt"]).to_csv(
        tables / "request_level_report.csv", index=False
    )
    for name, data in aggregates.items():
        data.to_csv(tables / f"{name}.csv", index=False)
    thinking.to_csv(tables / "thinking_paired_comparison.csv", index=False)
    examples.to_csv(tables / "representative_error_examples.csv", index=False)
    local_regression.to_csv(
        tables / "local_regression_candidate_comparison.csv", index=False
    )
    local_slopes.to_csv(
        tables / "local_per_model_accuracy_slopes.csv", index=False
    )
    for name, data in archived.items():
        data.to_csv(tables / f"archived_{name}.csv", index=False)

    make_figures(
        frame,
        aggregates,
        thinking,
        archived["accuracy_candidates"],
        assets,
    )
    copied_assets = copy_archived_assets(analysis_root, assets)
    copied_scripts = copy_reproduction_sources(
        analysis_root, scripts, Path(__file__).resolve()
    )
    build_html_report(
        output=output,
        frame=frame,
        sources=sources,
        audit=audit,
        aggregates=aggregates,
        thinking=thinking,
        examples=examples,
        local_regression=local_regression,
        replay_summary=replay_summary,
        archived=archived,
        metrics=metrics,
    )

    readme = f"""# Realistic NiaH eight-model HTML report

Open `report.html` in a browser. The report is offline; all images use relative
paths under `assets/`.

## Rebuild

Use a Python environment with NumPy, pandas, SciPy and Matplotlib:

```powershell
python scripts/build_report.py `
  --run-root <extracted>\\runs\\realistic_niah_v1\\six_models_formal_20260723T194300Z `
  --output-dir <new-empty-output-directory>
```

The builder refuses to overwrite an existing output directory. It validates
6,300 unique request IDs, expected per-model row counts, the complete N/T/seed
grid, and stimuli SHA256 `{EXPECTED_STIMULI_SHA}` before generating any result.

Primary accuracy retains parse failure, wrong format and truncation as failures.
Absolute error and signed bias are explicitly conditional on parsed outputs.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    elapsed = time.time() - started
    log_payload = {
        "started_at_utc": datetime.fromtimestamp(
            time.time() - elapsed, tz=timezone.utc
        ).isoformat(),
        "finished_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "input_audit": audit,
        "regression_replay": replay_summary,
    }
    write_json(logs / "build_log.json", log_payload)

    manifest = {
        "schema_version": "realistic-niah-eight-model-html-report-v1",
        "created_at_utc": utc_now(),
        "run_root": str(run_root),
        "output_root": str(output),
        "input_audit": audit,
        "request_sources": sources,
        "failure_policy": (
            "parse failure, wrong format and truncation retained as incorrect"
        ),
        "conditional_error_policy": (
            "absolute error and signed bias defined only for parsed outputs"
        ),
        "regression_replay": replay_summary,
        "archived_assets": copied_assets,
        "reproduction_scripts": copied_scripts,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "elapsed_seconds": elapsed,
    }
    write_json(output / "analysis_manifest.json", manifest)

    checksum_lines: list[str] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.tsv":
            checksum_lines.append(
                f"{sha256(path)}\t{path.relative_to(output).as_posix()}"
            )
    (output / "SHA256SUMS.tsv").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(output / "report.html"),
                "requests": len(frame),
                "accuracy": float(frame["exact_correct"].mean()),
                "parsed_outputs": int(frame["parse_success"].sum()),
                "regression_replay_max_abs_log_loss_difference": (
                    replay_summary[
                        "max_absolute_log_loss_difference_vs_archived"
                    ]
                ),
                "files": len(checksum_lines) + 1,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
