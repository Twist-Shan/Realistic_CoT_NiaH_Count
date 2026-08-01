from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


MODELS = ("Qwen3-8B", "Gemma4-E4B")
POOLINGS = ("span_end", "span_mean")
VARIANTS = ("v4.1", "v4.2", "v4.3", "v4.4")
VARIANT_DESCRIPTIONS = {
    "v4.1": "position, city-score order, and city-score content fixed",
    "v4.2": "position released; order and content fixed",
    "v4.3": "position and city-score order released; content fixed",
    "v4.4": "position, order, and city-score content all released",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _number(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    prefix = "+" if signed and numeric > 0 else ""
    return f"{prefix}{numeric:.{digits}f}"


def _image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _primary_layers(model_root: Path) -> dict[str, int]:
    payload = _read_json(
        model_root / "representation" / "analysis" / "representation_summary.json"
    )
    return {
        str(pooling): int(layer)
        for pooling, layer in payload["primary_layer_selection"]["layers"].items()
    }


def _n10_labels(model_root: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], pd.DataFrame]:
    labels = pd.read_csv(model_root / "behavior" / "capture" / "generation_labels.csv")
    labels = labels[labels["gold_count"].astype(int) == 10].copy()
    if labels.duplicated(["design_variant", "seed"]).any():
        raise ValueError("N=10 labels are not unique by variant and seed")
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in labels.to_dict("records"):
        outcome = str(row.get("outcome_group", "wrong"))
        if _bool(row.get("is_correct")):
            outcome = "correct"
        elif not _bool(row.get("format_valid")):
            outcome = "invalid"
        else:
            outcome = "wrong"
        lookup[(str(row["design_variant"]), int(row["seed"]))] = {
            "outcome": outcome,
            "parsed_count": (
                None if pd.isna(row.get("parsed_count")) else int(row["parsed_count"])
            ),
            "count_error": (
                None if pd.isna(row.get("count_error")) else int(row["count_error"])
            ),
        }
    return lookup, labels


def _load_projection(
    model_root: Path,
    *,
    model: str,
    pooling: str,
    layer: int,
    labels: dict[tuple[str, int], dict[str, Any]],
    components: int = 6,
) -> dict[str, Any]:
    capture_root = model_root / "representation" / "capture"
    records = _read_jsonl(capture_root / "capture_index.jsonl")
    tensors: dict[str, list[tuple[int, str, np.ndarray]]] = {variant: [] for variant in VARIANTS}
    for record in records:
        variant = str(record["design_variant"])
        if variant not in tensors:
            continue
        shard = capture_root / str(record["shard_path"])
        with np.load(shard, allow_pickle=False) as payload:
            layer_indices = np.asarray(payload["layer_indices"], dtype=int)
            match = np.flatnonzero(layer_indices == int(layer))
            if len(match) != 1:
                raise RuntimeError(f"{model}/{pooling}: layer {layer} absent in {shard}")
            states = np.asarray(payload[pooling][int(match[0])], dtype=np.float32)
        if states.shape[0] != 10:
            raise RuntimeError(f"Expected ten occurrence states, got {states.shape}")
        tensors[variant].append((int(record["seed"]), str(record["split"]), states))
    for variant in VARIANTS:
        tensors[variant].sort(key=lambda item: item[0])
        if len(tensors[variant]) != 30:
            raise RuntimeError(f"{model}/{pooling}/{variant}: expected 30 seed captures")

    reference = np.stack(
        [states for _seed, split, states in tensors["v4.1"] if split == "discovery"],
        axis=0,
    )
    fit = reference.reshape(-1, reference.shape[-1])
    pca = PCA(n_components=int(components), svd_solver="randomized", random_state=0)
    pca.fit(fit)

    rows: list[list[Any]] = []
    for variant in VARIANTS:
        for seed, split, states in tensors[variant]:
            projected = pca.transform(states)
            label = labels.get((variant, seed))
            if label is None:
                raise RuntimeError(f"Missing final-output label for {model}/{variant}/seed{seed}")
            for count_index, point in enumerate(projected, start=1):
                rows.append(
                    [
                        variant,
                        int(seed),
                        split,
                        label["outcome"],
                        label["parsed_count"],
                        label["count_error"],
                        int(count_index),
                        *[round(float(value), 6) for value in point],
                    ]
                )
    return {
        "model": model,
        "pooling": pooling,
        "layer": int(layer),
        "fit_variant": "v4.1",
        "fit_split": "discovery",
        "explained_variance_ratio": [
            round(float(value), 8) for value in pca.explained_variance_ratio_
        ],
        "rows": rows,
    }


def _metric_rows(run_root: Path, primary: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        analysis = run_root / model / "numeric" / "representation" / "analysis"
        metrics = pd.read_csv(analysis / "representation_layer_metrics.csv")
        for pooling in POOLINGS:
            layer = primary[model][pooling]
            selected = metrics[
                (metrics["pooling"] == pooling) & (metrics["layer"].astype(int) == layer)
            ]
            for row in selected.to_dict("records"):
                rows.append(row)
    return rows


def _behavior_rows(labels_by_model: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        for (variant, split), frame in labels.groupby(["design_variant", "split"], sort=True):
            parsed = pd.to_numeric(frame["parsed_count"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "variant": str(variant),
                    "split": str(split),
                    "n": int(len(frame)),
                    "correct": int(frame["is_correct"].map(_bool).sum()),
                    "accuracy": float(frame["is_correct"].map(_bool).mean()),
                    "mean_prediction": float(parsed.mean()),
                    "mae": float(pd.to_numeric(frame["count_error"], errors="coerce").abs().mean()),
                }
            )
    return rows


def _sensitivity_rows(run_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            run_root
            / model
            / "numeric"
            / "representation"
            / "analysis"
            / "seed_sensitivity_paired_bootstrap.csv"
        )
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            result.append({"model": model, **row})
    return result


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _seed_bootstrap(
    values: np.ndarray,
    *,
    label: str,
    iterations: int = 20_000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    estimate = float(values.mean())
    rng = np.random.default_rng(_stable_seed(label))
    sampled = values[
        rng.integers(0, values.size, size=(int(iterations), values.size))
    ].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return estimate, float(low), float(high)


def _exact_sign_flip_p(values: np.ndarray) -> float:
    """Two-sided exact sign-flip p-value for one seed-level paired contrast."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    if values.size > 20:
        raise ValueError("Exact sign-flip enumeration is capped at 20 clusters")
    observed = abs(float(values.mean()))
    assignments = np.arange(1 << values.size, dtype=np.uint64)[:, None]
    bits = (assignments >> np.arange(values.size, dtype=np.uint64)) & 1
    signs = bits.astype(float) * 2.0 - 1.0
    permuted = np.abs(signs @ values / values.size)
    return float(np.mean(permuted >= observed - 1e-12))


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    finite = [index for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda index: p_values[index])
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _p_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    if numeric < 0.001:
        return "&lt;0.001"
    return f"{numeric:.3f}"


def _span_end_undercount_frame(run_root: Path, model: str) -> pd.DataFrame:
    path = (
        run_root
        / model
        / "numeric"
        / "attention"
        / "analysis"
        / "tables"
        / "omission_diagnostics.csv"
    )
    frame = pd.read_csv(path)
    frame = frame[
        (frame["split"] == "confirmation")
        & (frame["pooling"] == "span_end")
        & (pd.to_numeric(frame["omission_count"], errors="coerce") > 0)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No confirmation span-end undercounts in {path}")
    selected_counts = pd.to_numeric(
        frame["selected_head_count"], errors="raise"
    ).astype(int)
    if not (selected_counts == 8).all():
        raise RuntimeError(f"Expected an eight-head discovery ensemble in {path}")
    frame["count"] = pd.to_numeric(frame["count"], errors="raise").astype(int)
    frame["omission_count"] = pd.to_numeric(
        frame["omission_count"], errors="raise"
    ).astype(int)
    invalid_k = (frame["omission_count"] <= 0) | (
        frame["omission_count"] > frame["count"]
    )
    if invalid_k.any():
        raise RuntimeError(f"Invalid undercount magnitude in {path}")
    frame["overlap"] = pd.to_numeric(
        frame["bottom_k_tail_overlap_fraction"], errors="raise"
    )
    frame["overlap_count"] = pd.to_numeric(
        frame["bottom_k_tail_overlap"], errors="raise"
    ).astype(int)
    frame["chance"] = frame["omission_count"] / frame["count"]
    frame["delta"] = frame["overlap"] - frame["chance"]
    frame["exact"] = (
        frame["overlap_count"] == frame["omission_count"]
    ).astype(float)
    frame["exact_chance"] = [
        1.0 / math.comb(int(count), int(k))
        for count, k in zip(frame["count"], frame["omission_count"])
    ]
    frame["exact_delta"] = frame["exact"] - frame["exact_chance"]
    frame["tail_prefix_ratio"] = pd.to_numeric(
        frame["undercount_tail_to_prefix_ratio"], errors="coerce"
    )
    return frame


def _span_end_alignment_rows(run_root: Path) -> list[dict[str, Any]]:
    """Variant-level confirmation tail alignment with seed-cluster inference."""
    results: list[dict[str, Any]] = []
    for model in MODELS:
        frame = _span_end_undercount_frame(run_root, model)

        for variant in VARIANTS:
            selected = frame[frame["design_variant"] == variant].copy()
            if selected.empty:
                raise RuntimeError(f"No {model}/{variant} span-end undercounts")
            by_seed = selected.groupby("seed", sort=True)[
                [
                    "overlap",
                    "chance",
                    "delta",
                    "exact",
                    "exact_chance",
                    "exact_delta",
                    "tail_prefix_ratio",
                ]
            ].mean()
            delta_est, delta_low, delta_high = _seed_bootstrap(
                by_seed["delta"].to_numpy(),
                label=f"tail-delta|{model}|{variant}",
            )
            exact_est, exact_low, exact_high = _seed_bootstrap(
                by_seed["exact_delta"].to_numpy(),
                label=f"tail-exact-delta|{model}|{variant}",
            )
            results.append(
                {
                    "model": model,
                    "variant": variant,
                    "prompts": int(len(selected)),
                    "seeds": int(by_seed.shape[0]),
                    "mean_k": float(selected["omission_count"].mean()),
                    "overlap": float(by_seed["overlap"].mean()),
                    "chance": float(by_seed["chance"].mean()),
                    "delta": delta_est,
                    "delta_low": delta_low,
                    "delta_high": delta_high,
                    "p_raw": _exact_sign_flip_p(by_seed["delta"].to_numpy()),
                    "exact": float(by_seed["exact"].mean()),
                    "exact_chance": float(by_seed["exact_chance"].mean()),
                    "exact_delta": exact_est,
                    "exact_delta_low": exact_low,
                    "exact_delta_high": exact_high,
                    "tail_prefix_ratio": float(by_seed["tail_prefix_ratio"].mean()),
                }
            )

    adjusted = _holm_adjust([float(row["p_raw"]) for row in results])
    for row, value in zip(results, adjusted):
        row["p_holm"] = value
    return results


def _span_end_pooled_rows(run_root: Path) -> list[dict[str, Any]]:
    """Equal-variant-weight model-level summary; every seed contributes once."""
    results: list[dict[str, Any]] = []
    metrics = [
        "overlap",
        "chance",
        "delta",
        "exact",
        "exact_chance",
        "exact_delta",
        "tail_prefix_ratio",
    ]
    for model in MODELS:
        frame = _span_end_undercount_frame(run_root, model)
        seed_variant = frame.groupby(["seed", "design_variant"], sort=True)[
            metrics
        ].mean()
        variant_counts = seed_variant.reset_index().groupby("seed")[
            "design_variant"
        ].nunique()
        if not (variant_counts == len(VARIANTS)).all():
            raise RuntimeError(f"Incomplete pooled span-end variants for {model}")
        by_seed = seed_variant.groupby("seed", sort=True)[metrics].mean()
        delta_est, delta_low, delta_high = _seed_bootstrap(
            by_seed["delta"].to_numpy(),
            label=f"tail-pooled-delta|{model}",
        )
        exact_est, exact_low, exact_high = _seed_bootstrap(
            by_seed["exact_delta"].to_numpy(),
            label=f"tail-pooled-exact-delta|{model}",
        )
        results.append(
            {
                "model": model,
                "seeds": int(len(by_seed)),
                "overlap": float(by_seed["overlap"].mean()),
                "chance": float(by_seed["chance"].mean()),
                "delta": delta_est,
                "delta_low": delta_low,
                "delta_high": delta_high,
                "p_raw": _exact_sign_flip_p(by_seed["delta"].to_numpy()),
                "exact": float(by_seed["exact"].mean()),
                "exact_chance": float(by_seed["exact_chance"].mean()),
                "exact_delta": exact_est,
                "exact_delta_low": exact_low,
                "exact_delta_high": exact_high,
                "tail_prefix_ratio": float(by_seed["tail_prefix_ratio"].mean()),
            }
        )
    adjusted = _holm_adjust([float(row["p_raw"]) for row in results])
    for row, value in zip(results, adjusted):
        row["p_holm"] = value
    return results


def _span_end_nested_rows(run_root: Path) -> list[dict[str, Any]]:
    """Exact new-needle diagnostic on undercount-ending nested transitions."""
    results: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            run_root
            / model
            / "numeric"
            / "attention"
            / "analysis"
            / "tables"
            / "nested_increment_diagnostics.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[
            (frame["split"] == "confirmation")
            & (frame["pooling"] == "span_end")
            & (pd.to_numeric(frame["count"], errors="coerce") >= 2)
            & (pd.to_numeric(frame["omission_count"], errors="coerce") > 0)
            & frame["increment_status"].isin(
                ["failed_to_increment", "registered_plus_one"]
            )
        ].copy()
        frame["omission_count"] = pd.to_numeric(
            frame["omission_count"], errors="raise"
        ).astype(int)
        frame["new_needle_low_attention_rank"] = pd.to_numeric(
            frame["new_needle_low_attention_rank"], errors="raise"
        ).astype(int)
        frame["new_needle_normalized_share"] = pd.to_numeric(
            frame["new_needle_normalized_share"], errors="raise"
        )
        frame["new_in_bottom_k"] = (
            frame["new_needle_low_attention_rank"] <= frame["omission_count"]
        ).astype(float)
        status_names = {
            "failed_to_increment": "failed",
            "registered_plus_one": "registered",
        }
        frame["status"] = frame["increment_status"].map(status_names)

        block_status = frame.groupby(
            ["seed", "design_variant", "status"], sort=True
        )[
            ["new_in_bottom_k", "new_needle_normalized_share"]
        ].mean()
        wide_bottom = block_status["new_in_bottom_k"].unstack("status").dropna()
        wide_share = block_status["new_needle_normalized_share"].unstack(
            "status"
        ).dropna()
        if not {"failed", "registered"}.issubset(wide_bottom.columns):
            raise RuntimeError(f"Missing paired nested statuses in {path}")
        common_blocks = wide_bottom.index.intersection(wide_share.index)
        wide_bottom = wide_bottom.loc[common_blocks]
        wide_share = wide_share.loc[common_blocks]
        paired_mask = frame.set_index(["seed", "design_variant"]).index.isin(
            common_blocks
        )
        paired_frame = frame.loc[paired_mask]
        seed_bottom = wide_bottom.groupby(level="seed").mean()
        seed_share = wide_share.groupby(level="seed").mean()
        bottom_difference = (
            wide_bottom["failed"] - wide_bottom["registered"]
        ).groupby(level="seed").mean().to_numpy()
        share_difference = (
            wide_share["registered"] - wide_share["failed"]
        ).groupby(level="seed").mean().to_numpy()
        bottom_est, bottom_low, bottom_high = _seed_bootstrap(
            bottom_difference,
            label=f"nested-bottom-difference|{model}",
        )
        share_est, share_low, share_high = _seed_bootstrap(
            share_difference,
            label=f"nested-share-difference|{model}",
        )
        failed_rate, failed_low, failed_high = _seed_bootstrap(
            seed_bottom["failed"].to_numpy(),
            label=f"nested-failed-rate|{model}",
        )
        registered_rate, registered_low, registered_high = _seed_bootstrap(
            seed_bottom["registered"].to_numpy(),
            label=f"nested-registered-rate|{model}",
        )
        results.append(
            {
                "model": model,
                "paired_seeds": int(seed_bottom.shape[0]),
                "paired_blocks": int(len(common_blocks)),
                "failed_n": int((paired_frame["status"] == "failed").sum()),
                "registered_n": int((paired_frame["status"] == "registered").sum()),
                "failed_bottom": failed_rate,
                "failed_bottom_low": failed_low,
                "failed_bottom_high": failed_high,
                "registered_bottom": registered_rate,
                "registered_bottom_low": registered_low,
                "registered_bottom_high": registered_high,
                "bottom_difference": bottom_est,
                "bottom_difference_low": bottom_low,
                "bottom_difference_high": bottom_high,
                "bottom_p_raw": _exact_sign_flip_p(bottom_difference),
                "failed_share": float(seed_share["failed"].mean()),
                "registered_share": float(seed_share["registered"].mean()),
                "share_difference": share_est,
                "share_difference_low": share_low,
                "share_difference_high": share_high,
                "share_p_raw": _exact_sign_flip_p(share_difference),
            }
        )

    for field in ("bottom", "share"):
        adjusted = _holm_adjust([float(row[f"{field}_p_raw"]) for row in results])
        for row, value in zip(results, adjusted):
            row[f"{field}_p_holm"] = value
    return results


def _table_metric_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td><code>{html.escape(str(row['pooling']))}</code></td>"
            f"<td>L{int(row['layer'])}</td>"
            f"<td>{html.escape(str(row['design_variant']))}</td>"
            f"<td>{_number(row['confirmation_r2'])}</td>"
            f"<td>{_number(row['confirmation_mae'])}</td>"
            f"<td>{_number(row['noise_to_signal_ratio'])}</td>"
            f"<td>{_number(row['discovery_confirmation_linear_cka'])}</td>"
            f"<td>{_number(row['discovery_confirmation_distance_correlation'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_behavior_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['model'])}</td>"
            f"<td>{html.escape(row['variant'])}</td>"
            f"<td>{html.escape(row['split'])}</td>"
            f"<td>{row['correct']}/{row['n']}</td>"
            f"<td>{_number(row['accuracy'], 2)}</td>"
            f"<td>{_number(row['mean_prediction'], 2)}</td>"
            f"<td>{_number(row['mae'], 2)}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_sensitivity_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td><code>{html.escape(str(row['pooling']))}</code></td>"
            f"<td>L{int(row['primary_layer'])}</td>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{html.escape(str(row['left_variant']))} → {html.escape(str(row['right_variant']))}</td>"
            f"<td>{_number(row['delta_mean'], signed=True)}</td>"
            f"<td>[{_number(row['ci95_low'])}, {_number(row['ci95_high'])}]</td>"
            f"<td>{'yes' if _bool(row['increase_ci_excludes_zero']) else 'no'}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_span_end_alignment_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{int(row['prompts'])} / {int(row['seeds'])}</td>"
            f"<td>{_number(row['mean_k'], 2)}</td>"
            f"<td>{_number(row['overlap'])}</td>"
            f"<td>{_number(row['chance'])}</td>"
            f"<td>{_number(row['delta'], signed=True)} "
            f"[{_number(row['delta_low'])}, {_number(row['delta_high'])}]</td>"
            f"<td>{_p_value(row['p_holm'])}</td>"
            f"<td>{_number(row['exact'])} / {_number(row['exact_chance'])}</td>"
            f"<td>{_number(row['exact_delta'], signed=True)} "
            f"[{_number(row['exact_delta_low'])}, {_number(row['exact_delta_high'])}]</td>"
            f"<td>{_number(row['tail_prefix_ratio'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_span_end_pooled_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{int(row['seeds'])}</td>"
            f"<td>{_number(row['overlap'])} / {_number(row['chance'])}</td>"
            f"<td>{_number(row['delta'], signed=True)} "
            f"[{_number(row['delta_low'])}, {_number(row['delta_high'])}]</td>"
            f"<td>{_p_value(row['p_holm'])}</td>"
            f"<td>{_number(row['exact'])} / {_number(row['exact_chance'])}</td>"
            f"<td>{_number(row['exact_delta'], signed=True)} "
            f"[{_number(row['exact_delta_low'])}, {_number(row['exact_delta_high'])}]</td>"
            f"<td>{_number(row['tail_prefix_ratio'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_span_end_nested_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{int(row['failed_n'])} / {int(row['registered_n'])}</td>"
            f"<td>{int(row['paired_blocks'])} / {int(row['paired_seeds'])}</td>"
            f"<td>{_number(row['failed_bottom'])} / "
            f"{_number(row['registered_bottom'])}</td>"
            f"<td>{_number(row['bottom_difference'], signed=True)} "
            f"[{_number(row['bottom_difference_low'])}, "
            f"{_number(row['bottom_difference_high'])}]</td>"
            f"<td>{_p_value(row['bottom_p_holm'])}</td>"
            f"<td>{_number(row['failed_share'])} / "
            f"{_number(row['registered_share'])}</td>"
            f"<td>{_number(row['share_difference'], signed=True)} "
            f"[{_number(row['share_difference_low'])}, "
            f"{_number(row['share_difference_high'])}]</td>"
            f"<td>{_p_value(row['share_p_holm'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _span_end_alignment_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1040, 390
    panel_lefts = [70, 570]
    panel_width = 390
    x_max = 0.70

    def x_position(value: float, panel_left: float) -> float:
        return panel_left + max(0.0, min(x_max, value)) / x_max * panel_width

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="tail-plot-title tail-plot-desc">',
        '<title id="tail-plot-title">Span-end tail and bottom-k overlap versus chance</title>',
        '<desc id="tail-plot-desc">Observed overlap is above the hypergeometric chance baseline in most model and variant panels.</desc>',
        '<rect width="1040" height="390" fill="#fffdf8"/>',
        '<g font-family="Inter,system-ui,sans-serif" fill="#172128">',
    ]
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        parts.append(
            f'<text x="{left}" y="34" font-size="17" font-weight="700">'
            f'{html.escape(model)}</text>'
        )
        for tick in np.arange(0.0, x_max + 0.001, 0.1):
            x = x_position(float(tick), left)
            parts.append(
                f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="315" '
                'stroke="#e5dfd3" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="338" text-anchor="middle" '
                f'font-size="11" fill="#66727a">{tick:.1f}</text>'
            )
        model_rows = [row for row in rows if row["model"] == model]
        for row_index, row in enumerate(model_rows):
            y = 88 + row_index * 58
            chance_x = x_position(float(row["chance"]), left)
            observed_x = x_position(float(row["overlap"]), left)
            ci_low_x = x_position(
                float(row["chance"]) + float(row["delta_low"]), left
            )
            ci_high_x = x_position(
                float(row["chance"]) + float(row["delta_high"]), left
            )
            parts.extend(
                [
                    f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" '
                    f'font-size="12" font-weight="650">{html.escape(str(row["variant"]))}</text>',
                    f'<line x1="{chance_x:.1f}" y1="{y}" x2="{observed_x:.1f}" '
                    f'y2="{y}" stroke="#9aa5a8" stroke-width="2"/>',
                    f'<line x1="{ci_low_x:.1f}" y1="{y}" x2="{ci_high_x:.1f}" '
                    f'y2="{y}" stroke="#2e5d72" stroke-width="5" stroke-linecap="round" opacity=".42"/>',
                    f'<path d="M {chance_x:.1f} {y - 6} L {chance_x + 6:.1f} {y} '
                    f'L {chance_x:.1f} {y + 6} L {chance_x - 6:.1f} {y} Z" fill="#b08430"/>',
                    f'<circle cx="{observed_x:.1f}" cy="{y}" r="6" fill="#2e5d72" stroke="#fffdf8" stroke-width="1.5"/>',
                    f'<text x="{left + panel_width + 9}" y="{y + 4}" font-size="11" fill="#3d4c53">'
                    f'Δ {_number(row["delta"], signed=True)}</text>',
                ]
            )
        parts.append(
            f'<text x="{left + panel_width / 2:.1f}" y="365" text-anchor="middle" '
            'font-size="12" fill="#66727a">tail overlap fraction</text>'
        )
    parts.extend(
        [
            '<path d="M 761 28 L 767 34 L 761 40 L 755 34 Z" fill="#b08430"/>',
            '<text x="774" y="38" font-size="11" fill="#66727a">hypergeometric chance</text>',
            '<circle cx="910" cy="34" r="6" fill="#2e5d72"/>',
            '<text x="920" y="38" font-size="11" fill="#66727a">observed (95% seed CI)</text>',
            '</g></svg>',
        ]
    )
    return "".join(parts)


def _span_end_nested_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 920, 290
    left, plot_width, x_max = 180, 570, 0.70

    def x_position(value: float) -> float:
        return left + max(0.0, min(x_max, value)) / x_max * plot_width

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="nested-plot-title nested-plot-desc">',
        '<title id="nested-plot-title">New needle bottom-k attention risk by increment status</title>',
        '<desc id="nested-plot-desc">The newly introduced needle is more often in the lowest-attention set when the output fails to increment.</desc>',
        '<rect width="920" height="290" fill="#fffdf8"/>',
        '<g font-family="Inter,system-ui,sans-serif" fill="#172128">',
    ]
    for tick in np.arange(0.0, x_max + 0.001, 0.1):
        x = x_position(float(tick))
        parts.append(
            f'<line x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="222" '
            'stroke="#e5dfd3" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="244" text-anchor="middle" font-size="11" '
            f'fill="#66727a">{tick:.1f}</text>'
        )
    for row_index, row in enumerate(rows):
        center_y = 92 + row_index * 88
        parts.append(
            f'<text x="{left - 18}" y="{center_y + 4}" text-anchor="end" '
            f'font-size="14" font-weight="700">{html.escape(str(row["model"]))}</text>'
        )
        for status, color, offset in (
            ("failed", "#a0443e", -12),
            ("registered", "#47705e", 12),
        ):
            value = float(row[f"{status}_bottom"])
            low = float(row[f"{status}_bottom_low"])
            high = float(row[f"{status}_bottom_high"])
            y = center_y + offset
            parts.extend(
                [
                    f'<line x1="{x_position(low):.1f}" y1="{y}" '
                    f'x2="{x_position(high):.1f}" y2="{y}" stroke="{color}" '
                    'stroke-width="4" stroke-linecap="round" opacity=".42"/>',
                    f'<circle cx="{x_position(value):.1f}" cy="{y}" r="6" '
                    f'fill="{color}" stroke="#fffdf8" stroke-width="1.5"/>',
                ]
            )
        parts.append(
            f'<text x="770" y="{center_y + 4}" font-size="11" fill="#3d4c53">'
            f'RD {_number(row["bottom_difference"], signed=True)} '
            f'[{_number(row["bottom_difference_low"])}, {_number(row["bottom_difference_high"])}]'
            '</text>'
        )
    parts.extend(
        [
            '<circle cx="590" cy="25" r="6" fill="#a0443e"/><text x="601" y="29" font-size="11" fill="#66727a">failed to increment</text>',
            '<circle cx="720" cy="25" r="6" fill="#47705e"/><text x="731" y="29" font-size="11" fill="#66727a">registered +1</text>',
            '<text x="465" y="273" text-anchor="middle" font-size="12" fill="#66727a">P(new needle is in current bottom-k attention set)</text>',
            '</g></svg>',
        ]
    )
    return "".join(parts)


def _variant_list(values: list[str]) -> str:
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _span_end_conclusion_html(
    pooled_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    nested_rows: list[dict[str, Any]],
) -> str:
    cards: list[str] = []
    for model in MODELS:
        selected = [row for row in alignment_rows if row["model"] == model]
        pooled = next(row for row in pooled_rows if row["model"] == model)
        positive_ci = [
            str(row["variant"]) for row in selected if float(row["delta_low"]) > 0
        ]
        holm = [
            str(row["variant"]) for row in selected if float(row["p_holm"]) < 0.05
        ]
        cards.append(
            '<div class="note"><strong>'
            + html.escape(model)
            + " tail alignment.</strong><p>The equal-variant pooled contrast is "
            + _number(pooled["delta"], signed=True)
            + " ["
            + _number(pooled["delta_low"])
            + ", "
            + _number(pooled["delta_high"])
            + "], Holm p="
            + _p_value(pooled["p_holm"])
            + ". All four variant point estimates are above chance. "
            + "The 95% seed-cluster interval excludes zero for "
            + html.escape(_variant_list(positive_ci))
            + "; the exact sign-flip test remains below 0.05 after Holm correction for "
            + html.escape(_variant_list(holm))
            + ".</p></div>"
        )
    nested_sentences: list[str] = []
    for row in nested_rows:
        nested_sentences.append(
            f"{html.escape(str(row['model']))}: risk difference "
            f"{_number(row['bottom_difference'], signed=True)} "
            f"[{_number(row['bottom_difference_low'])}, "
            f"{_number(row['bottom_difference_high'])}], "
            f"Holm p={_p_value(row['bottom_p_holm'])}"
        )
    cards.append(
        '<div class="note"><strong>Exact new-needle check.</strong><p>'
        + "; ".join(nested_sentences)
        + ". Positive values mean the newly added needle is more often in the "
        + "bottom-k attention set when the output fails to increment.</p></div>"
    )
    return "".join(cards)


def _static_figure_html(run_root: Path) -> str:
    cards: list[str] = []
    for model in MODELS:
        analysis = run_root / model / "numeric" / "representation" / "analysis"
        summary = _read_json(analysis / "representation_summary.json")
        layers = summary["primary_layer_selection"]["layers"]
        for pooling in POOLINGS:
            layer = int(layers[pooling])
            source = analysis / "figures" / f"shared_pca_{pooling}_layer_{layer}.png"
            outcome = (
                analysis
                / "outcomes"
                / "figures"
                / f"shared_pca_{pooling}_layer_{layer}_by_outcome.png"
            )
            cards.append(
                "<article class=\"figure-card\">"
                f"<h3>{html.escape(model)} · {html.escape(pooling)} · L{layer}</h3>"
                f"<img loading=\"lazy\" src=\"{_image_data_uri(source)}\" "
                f"alt=\"Shared PCA for {html.escape(model)} {html.escape(pooling)}\">"
                "<p>All seeds. PCA basis fit on v4.1 discovery.</p>"
                "</article>"
            )
            if outcome.exists():
                cards.append(
                    "<article class=\"figure-card\">"
                    f"<h3>{html.escape(model)} · {html.escape(pooling)} · output strata</h3>"
                    f"<img loading=\"lazy\" src=\"{_image_data_uri(outcome)}\" "
                    f"alt=\"Outcome-stratified PCA for {html.escape(model)} {html.escape(pooling)}\">"
                    "<p>Confirmation seeds labeled by the actual greedy N=10 output.</p>"
                    "</article>"
                )
    return "\n".join(cards)


REPORT_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Realistic NIAH V4 Representation Report</title>
<style>
:root { --ink:#172128; --muted:#66727a; --paper:#f7f4ed; --card:#fffdf8; --line:#d8d2c6; --blue:#2e5d72; --red:#a0443e; --gold:#b08430; --green:#47705e; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--paper); font:15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding:58px max(24px, calc((100vw - 1180px)/2)); background:#18282f; color:#f8f4e9; border-bottom:5px solid var(--gold); }
header .eyebrow { color:#d9bc78; text-transform:uppercase; letter-spacing:.13em; font-size:12px; font-weight:700; }
h1 { max-width:900px; margin:10px 0 14px; font:700 clamp(34px,5vw,62px)/1.02 Georgia,serif; letter-spacing:-.025em; }
header p { max-width:860px; color:#d7e0e2; font-size:17px; }
.meta { display:flex; flex-wrap:wrap; gap:9px; margin-top:24px; }
.pill { border:1px solid #557079; border-radius:999px; padding:6px 11px; color:#e9efef; font-size:12px; }
nav { position:sticky; top:0; z-index:20; display:flex; gap:20px; overflow:auto; padding:11px max(24px, calc((100vw - 1180px)/2)); background:rgba(247,244,237,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(9px); }
nav a { color:var(--blue); text-decoration:none; white-space:nowrap; font-weight:650; }
main { max-width:1180px; margin:auto; padding:32px 24px 80px; }
section { margin:0 0 46px; scroll-margin-top:58px; }
h2 { margin:0 0 10px; font:700 30px/1.15 Georgia,serif; }
h3 { margin:0 0 8px; font-size:16px; }
.lede { max-width:900px; color:#3d4c53; font-size:16px; }
.callout { margin:18px 0; padding:16px 18px; border-left:4px solid var(--gold); background:#efe9dc; }
.grid4 { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:22px 0; }
.step { min-height:145px; padding:17px; background:var(--card); border:1px solid var(--line); border-top:4px solid var(--blue); }
.step strong { display:block; font:700 23px Georgia,serif; color:var(--blue); }
.step small { color:var(--muted); }
.table-wrap { overflow:auto; border:1px solid var(--line); background:var(--card); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { padding:9px 10px; text-align:right; border-bottom:1px solid #e7e2d9; white-space:nowrap; }
th { position:sticky; top:0; background:#eee9de; color:#39474d; font-size:11px; text-transform:uppercase; letter-spacing:.045em; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
tr:last-child td { border-bottom:0; }
code { color:#31596a; background:#e9eff0; padding:2px 5px; border-radius:3px; }
.viz-shell { margin-top:20px; padding:18px; background:#122229; color:#eaf0ef; border:1px solid #33474e; box-shadow:0 18px 38px rgba(24,40,47,.16); }
.controls { display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)); gap:10px; margin-bottom:12px; }
label { display:flex; flex-direction:column; gap:4px; color:#b9c7ca; font-size:11px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
select,button { width:100%; border:1px solid #4d6269; background:#1d333b; color:#f3f5f3; border-radius:3px; padding:8px 9px; font:inherit; }
button { cursor:pointer; font-weight:700; }
button:hover { background:#294650; }
.canvas-wrap { position:relative; min-height:610px; background:#0b171c; border:1px solid #31444b; }
#counter3d { display:block; width:100%; height:610px; cursor:grab; }
#counter3d.dragging { cursor:grabbing; }
#tooltip { position:absolute; display:none; pointer-events:none; max-width:260px; padding:8px 10px; border:1px solid #6f858b; background:rgba(8,19,24,.94); color:#f5f3ea; font-size:12px; box-shadow:0 8px 18px rgba(0,0,0,.32); }
.viz-foot { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:12px; color:#bfcacc; font-size:12px; }
#geometry-stats { color:#f0d79b; text-align:right; }
.legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:9px; color:#c6d1d3; font-size:12px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }
.figures { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
.figure-card { background:var(--card); border:1px solid var(--line); padding:13px; }
.figure-card img { display:block; width:100%; height:auto; background:white; }
.figure-card p { margin:7px 0 0; color:var(--muted); font-size:12px; }
.stat-grid { display:grid; grid-template-columns:1fr; gap:16px; margin:18px 0; }
.stat-figure { margin:0; padding:14px; background:var(--card); border:1px solid var(--line); }
.stat-svg { display:block; width:100%; height:auto; }
.stat-figure figcaption { margin:8px 6px 2px; color:var(--muted); font-size:12px; }
.formula { margin:16px 0; padding:14px 17px; background:#edf1ef; border:1px solid #ccd8d3; font-family:Georgia,serif; font-size:16px; }
.method-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:16px 0; }
.method-strip div { padding:12px; background:var(--card); border:1px solid var(--line); font-size:12px; }
.method-strip strong { display:block; color:var(--blue); font-size:13px; }
.notes { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
.note { padding:17px; background:var(--card); border:1px solid var(--line); }
.note strong { color:var(--blue); }
footer { padding:24px; color:#6d7475; text-align:center; border-top:1px solid var(--line); }
@media (max-width:900px) { .grid4,.notes,.method-strip { grid-template-columns:repeat(2,1fr); } .controls { grid-template-columns:repeat(3,1fr); } .figures { grid-template-columns:1fr; } }
@media (max-width:560px) { .grid4,.notes,.method-strip,.viz-foot { grid-template-columns:1fr; } .controls { grid-template-columns:repeat(2,1fr); } #counter3d { height:500px; } .canvas-wrap { min-height:500px; } }
</style>
</head>
<body>
<header>
  <div class="eyebrow">Realistic NIAH · non-thinking · V4.1–V4.4</div>
  <h1>Where does the running count live?</h1>
  <p>A v10-style representation analysis of needle-occurrence hidden states. The report separates <code>span_end</code> from <code>span_mean</code>, fits every PCA basis only on v4.1 discovery seeds, and then stress-tests the same coordinate system as position, order, and content are released.</p>
  <div class="meta">
    <span class="pill">Qwen3-8B + Gemma4-E4B</span><span class="pill">length ≈ 10,000 tokens</span><span class="pill">needle index 1–10</span><span class="pill">30 seeds / variant</span><span class="pill">commit @@COMMIT@@</span>
  </div>
</header>
<nav><a href="#design">Design</a><a href="#metrics">Metrics</a><a href="#counter">3D counter</a><a href="#span-end-attention">Undercount attention</a><a href="#sensitivity">Seed sensitivity</a><a href="#outcomes">Output strata</a><a href="#figures">2D panels</a><a href="#limits">Limits</a></nav>
<main>
<section id="design">
  <h2>Controlled relaxation ladder</h2>
  <p class="lede">The unit of representation analysis is one N=10 prompt. Its ten occurrence vectors are ordered by the nested needle index. Discovery uses seeds 1234–1253; confirmation uses 1254–1263. Final-output labels always come from actual greedy generation, never candidate probability.</p>
  <div class="grid4">
    <div class="step"><strong>V4.1</strong><small>all fixed</small><p>Position, city-score order, and city-score content fixed across seeds.</p></div>
    <div class="step"><strong>V4.2</strong><small>position released</small><p>Needle positions vary; order and semantic content remain fixed.</p></div>
    <div class="step"><strong>V4.3</strong><small>order released</small><p>Positions and city-score order vary; content remains fixed.</p></div>
    <div class="step"><strong>V4.4</strong><small>content released</small><p>Position, order, and city-score content all vary across seeds.</p></div>
  </div>
  <div class="callout"><strong>Interpretation rule.</strong> A clean v4.1 curve can be an identity/order/position code. Evidence for a content-independent counter requires geometry and count decoding to persist into v4.3 and especially v4.4.</div>
</section>

<section id="metrics">
  <h2>Primary-layer confirmation metrics</h2>
  <p class="lede">Each pooling’s layer is selected once, using maximum grouped-seed ridge cross-validation R² on v4.1 discovery only. No confirmation label enters layer selection. Lower MAE and noise/signal are better; higher R², linear CKA, and distance correlation are better.</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>layer</th><th>variant</th><th>confirm R²</th><th>confirm MAE</th><th>noise / signal</th><th>linear CKA</th><th>distance corr.</th></tr></thead><tbody>@@METRIC_ROWS@@</tbody></table></div>
</section>

<section id="counter">
  <h2>Interactive 3D count manifold</h2>
  <p class="lede">This is the requested switchable 3D counter view. Drag to rotate, use the mouse wheel to zoom, and switch any displayed axis among PC1–PC6. Faint dots are individual seed × occurrence states; lines are split-specific centroids from index 1 to 10. Colors encode the occurrence index, not correctness.</p>
  <div class="viz-shell">
    <div class="controls">
      <label>Model<select id="model-select"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
      <label>Pooling<select id="pooling-select"><option value="span_end">span-end</option><option value="span_mean">span-mean</option></select></label>
      <label>Variant<select id="variant-select"><option>v4.1</option><option>v4.2</option><option>v4.3</option><option>v4.4</option></select></label>
      <label>Split<select id="split-select"><option value="all">all</option><option value="discovery">discovery</option><option value="confirmation">confirmation</option></select></label>
      <label>Final output<select id="outcome-select"><option value="all">all</option><option value="correct">correct</option><option value="wrong">wrong</option><option value="invalid">invalid</option></select></label>
      <label>View<button id="reset-view" type="button">reset rotation</button></label>
      <label>X axis<select id="x-axis"></select></label>
      <label>Y axis<select id="y-axis"></select></label>
      <label>Z axis<select id="z-axis"></select></label>
      <label>Points<select id="points-select"><option value="all">all seed points</option><option value="confirmation">confirmation only</option><option value="centroids">centroids only</option></select></label>
      <label>Scale<select id="scale-select"><option value="metric">equal metric scale</option><option value="normalized">normalize each axis</option></select></label>
      <label>Preset<select id="axis-preset"><option value="0,1,2">PC1 / PC2 / PC3</option><option value="0,2,3">PC1 / PC3 / PC4</option><option value="1,2,3">PC2 / PC3 / PC4</option><option value="3,4,5">PC4 / PC5 / PC6</option></select></label>
    </div>
    <div class="canvas-wrap"><canvas id="counter3d" aria-label="Interactive 3D PCA counter trajectory"></canvas><div id="tooltip"></div></div>
    <div class="viz-foot"><div id="pca-stats"></div><div id="geometry-stats"></div></div>
    <div class="legend" id="count-legend"></div>
  </div>
  <div class="callout"><strong>Coordinate comparability.</strong> All four variants share a PCA basis within a fixed model × pooling panel. Bases are fitted separately across panels, so absolute PC coordinates should not be compared between Qwen/Gemma or span-end/span-mean.</div>
</section>

<section id="span-end-attention">
  <h2>Span-end attention–undercount alignment</h2>
  <p class="lede">This analysis is intentionally restricted to <code>span_end</code>. It asks whether the answer-query attention ensemble assigns unusually low mass to the evidence implied by an undercount. All estimates use confirmation seeds only; head ranking is frozen from discovery data within each model × variant, and the top eight discovery-ranked broad heads are averaged after each head is normalized to mean occurrence share 1.</p>
  <div class="method-strip">
    <div><strong>Behavior label</strong>Actual greedy integer output <em>N̂</em>; no sequence probability or candidate score enters the label. Unparseable and non-undercount outputs are outside this estimand.</div>
    <div><strong>Held-out unit</strong>Ten confirmation seeds (1254–1263); prompts within a seed remain in the same resampling cluster.</div>
    <div><strong>Uncertainty</strong>20,000 percentile bootstrap resamples of seed-level means.</div>
    <div><strong>Testing</strong>Two-sided exact seed sign-flip tests under the symmetry null; Holm families are the two pooled model tests and, separately, the eight variant panels.</div>
  </div>

  <h3>1. Behavior-implied omitted tail versus the lowest-attention occurrences</h3>
  <p>For a gold count <em>N</em> and an undercount <em>N̂</em>, let <em>k = N − N̂</em>. The behavior-implied omitted tail is <em>T<sub>k</sub> = {N−k+1,…,N}</em>; the attention-implied candidate set <em>B<sub>k</sub></em> contains the <em>k</em> occurrence endpoints with the lowest ensemble attention. The primary score is their overlap fraction.</p>
  <div class="formula"><em>S = |B<sub>k</sub> ∩ T<sub>k</sub>| / k</em>. If <em>B<sub>k</sub></em> were a uniformly random <em>k</em>-subset, <em>E[S] = k/N</em> and <em>P(B<sub>k</sub> = T<sub>k</sub>) = 1 / C(N,k)</em>.</div>
  <p class="lede">The estimand is the seed-equal mean of <em>S − k/N</em>. To avoid selecting the most favorable relaxation, the cross-variant aggregate gives each variant equal weight inside each seed; the variant panels expose heterogeneity. This analysis was not preregistered, so its intervals and tests are inferential audits rather than confirmatory claims. “Exact / random” reports the observed exact-set-match rate and its combinatorial chance baseline. Tail/prefix is the mean normalized attention on the behavior-implied tail divided by the retained prefix; values below 1 indicate suppressed tail attention.</p>
  <h3>Cross-variant aggregate</h3>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>seeds</th><th>overlap / random</th><th>Δ [95% seed CI]</th><th>Holm p</th><th>exact / random</th><th>exact Δ [95% CI]</th><th>tail / prefix</th></tr></thead><tbody>@@SPAN_END_POOLED_ROWS@@</tbody></table></div>
  <h3>Variant-level heterogeneity</h3>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>variant</th><th>prompts / seeds</th><th>mean k</th><th>overlap</th><th>random k/N</th><th>Δ [95% seed CI]</th><th>Holm p</th><th>exact / random</th><th>exact Δ [95% CI]</th><th>tail / prefix</th></tr></thead><tbody>@@SPAN_END_ALIGNMENT_ROWS@@</tbody></table></div>
  <div class="stat-grid">
    <figure class="stat-figure">@@SPAN_END_ALIGNMENT_SVG@@<figcaption>Circle: observed seed-equal overlap. Diamond: hypergeometric chance. The translucent interval is the 95% seed-cluster interval for the observed-minus-chance contrast, translated onto the overlap axis.</figcaption></figure>
  </div>

  <h3>2. Exact newly added needle in nested N−1 → N pairs</h3>
  <p>The tail analysis identifies an omitted <em>set</em> only under an ordering assumption. The nested design supplies a sharper check: the newly introduced Nth occurrence is known exactly. We retain transitions ending in an undercount for both groups, then compare (i) whether that new endpoint falls inside the current bottom-<em>k</em> attention set and (ii) its normalized attention share, where 1 is uniform over occurrences. “Failed” means the greedy output did not increase; “registered” means it increased by exactly one. Statuses are first matched within seed × variant; matched variant contrasts are then averaged inside seed. Holm correction is applied across the two models separately for each nested estimand.</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>failed / registered n</th><th>paired blocks / seeds</th><th>bottom-k failed / reg.</th><th>risk Δ F−R [95% CI]</th><th>Holm p</th><th>share failed / reg.</th><th>share Δ R−F [95% CI]</th><th>Holm p</th></tr></thead><tbody>@@SPAN_END_NESTED_ROWS@@</tbody></table></div>
  <div class="stat-grid">
    <figure class="stat-figure">@@SPAN_END_NESTED_SVG@@<figcaption>Both transition groups end in a behavioral undercount. Intervals around each point resample complete seeds; the reported risk-difference interval is paired at seed level.</figcaption></figure>
  </div>
  <div class="notes">@@SPAN_END_CONCLUSION@@</div>
  <div class="callout"><strong>Inference boundary.</strong> The tail set is behavior-implied, not an observed record of which occurrences the model “forgot.” The nested increment diagnostic identifies the added occurrence exactly, but attention remains correlational. These results motivate—and do not replace—the registered ablation and endpoint residual-patching tests.</div>
</section>

<section id="sensitivity">
  <h2>Where seed sensitivity first becomes detectable</h2>
  <p class="lede">Adjacent relaxations are compared with paired confirmation seeds. “Positive” means the 95% seed-bootstrap interval for worsening excludes zero. This controls the seed pairing but remains a descriptive representation result.</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>layer</th><th>metric</th><th>step</th><th>Δ mean</th><th>95% CI</th><th>positive</th></tr></thead><tbody>@@SENSITIVITY_ROWS@@</tbody></table></div>
</section>

<section id="outcomes">
  <h2>Actual greedy N=10 outcomes</h2>
  <p class="lede">Every occurrence vector in one trajectory inherits that prompt’s final greedy output label. This is the requested behavior-based labeling. It is not based on token probability and it does not relabel individual occurrences.</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>variant</th><th>split</th><th>correct / n</th><th>accuracy</th><th>mean prediction</th><th>MAE</th></tr></thead><tbody>@@BEHAVIOR_ROWS@@</tbody></table></div>
  <div class="callout"><strong>Severe class imbalance.</strong> Qwen has no correct N=10 confirmation trajectory in any variant; Gemma has one in v4.1 and none in v4.2–v4.4. Therefore the correct/wrong switch is an audit view, not a powered group comparison. The causal studies should stratify by the baseline outcome over the full count grid, where both correct and wrong examples are available.</div>
</section>

<section id="figures">
  <h2>Static 2D audit panels</h2>
  <p class="lede">These are the original analysis artifacts embedded verbatim for reproducibility. The first panel in each pair shows all seeds; the second applies actual final-output strata.</p>
  <div class="figures">@@STATIC_FIGURES@@</div>
</section>

<section id="limits">
  <h2>What this report can and cannot establish</h2>
  <div class="notes">
    <div class="note"><strong>Representation ≠ mechanism.</strong><p>PCA, ridge decoding, and centroid geometry show availability of count-related information. They do not show that generation reads or needs it.</p></div>
    <div class="note"><strong>Span-end vs span-mean.</strong><p>Span-end asks whether a localized terminal state carries the running index. Span-mean asks whether information is distributed across the full semantic needle span. Their PCA bases and primary layers are intentionally separate.</p></div>
    <div class="note"><strong>Causal next step.</strong><p>Registered head ablation, answer-query head-output patching, exact needle-end/full-span residual patching, and geometric steering test necessity and transport on held-out confirmation seeds.</p></div>
  </div>
</section>
</main>
<footer>Generated @@GENERATED@@ · source run <code>@@RUN_NAME@@</code> · commit <code>@@COMMIT@@</code></footer>
<script>
const REP_DATA = @@REP_DATA@@;
const COLORS = ['#482878','#3e4989','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725','#ffb000'];
const canvas = document.getElementById('counter3d');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const controls = {
  model: document.getElementById('model-select'), pooling: document.getElementById('pooling-select'),
  variant: document.getElementById('variant-select'), split: document.getElementById('split-select'),
  outcome: document.getElementById('outcome-select'), points: document.getElementById('points-select'),
  scale: document.getElementById('scale-select'), x: document.getElementById('x-axis'),
  y: document.getElementById('y-axis'), z: document.getElementById('z-axis'),
  preset: document.getElementById('axis-preset')
};
for (const select of [controls.x, controls.y, controls.z]) {
  for (let i=0;i<6;i++) { const o=document.createElement('option'); o.value=i; o.textContent=`PC${i+1}`; select.appendChild(o); }
}
controls.x.value='0'; controls.y.value='1'; controls.z.value='2';
let yaw=-0.72, pitch=0.44, zoom=1.0, dragging=false, lastX=0, lastY=0, projectedPoints=[];

function activeData() { return REP_DATA[`${controls.model.value}|${controls.pooling.value}`]; }
function filteredRows() {
  const data=activeData(); if (!data) return [];
  return data.rows.filter(r => r[0]===controls.variant.value && (controls.split.value==='all'||r[2]===controls.split.value) && (controls.outcome.value==='all'||r[3]===controls.outcome.value));
}
function resizeCanvas() {
  const rect=canvas.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.max(1,Math.round(rect.width*dpr)); canvas.height=Math.max(1,Math.round(rect.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0); draw();
}
function statsFor(rows, axes) {
  if (!rows.length) return null;
  const vals=axes.map(a=>rows.map(r=>r[7+a]));
  const mins=vals.map(v=>Math.min(...v)), maxs=vals.map(v=>Math.max(...v));
  const centers=mins.map((m,i)=>(m+maxs[i])/2), ranges=mins.map((m,i)=>Math.max(maxs[i]-m,1e-8));
  return {mins,maxs,centers,ranges};
}
function makeTransform(rows, axes, width, height) {
  const s=statsFor(rows,axes); if (!s) return null;
  const perAxis=controls.scale.value==='normalized';
  const common=Math.max(...s.ranges); const scales=s.ranges.map(r=>perAxis?1/r:1/common);
  const radius=Math.min(width,height)*0.36*zoom;
  return p=>{
    let x=(p[0]-s.centers[0])*scales[0]*2, y=(p[1]-s.centers[1])*scales[1]*2, z=(p[2]-s.centers[2])*scales[2]*2;
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const x1=cy*x+sy*z, z1=-sy*x+cy*z, y1=cp*y-sp*z1, z2=sp*y+cp*z1;
    return {x:width/2+x1*radius,y:height/2-y1*radius,z:z2,raw:p};
  };
}
function centroids(rows) {
  const groups=new Map();
  for (const r of rows) { const key=r[2]; if (!groups.has(key)) groups.set(key,new Map()); const byCount=groups.get(key); if (!byCount.has(r[6])) byCount.set(r[6],[]); byCount.get(r[6]).push(r); }
  const result=[];
  for (const [split,byCount] of groups.entries()) {
    const path=[];
    for (let count=1;count<=10;count++) { const rs=byCount.get(count)||[]; if (!rs.length) continue; const p=[]; for(let pc=0;pc<6;pc++) p.push(rs.reduce((a,r)=>a+r[7+pc],0)/rs.length); path.push({count,p,n:rs.length}); }
    result.push({split,path});
  }
  return result;
}
function geometryText(paths, axes) {
  if (!paths.length) return 'No centroid path for this filter.';
  return paths.map(group=>{
    const p=group.path.map(d=>axes.map(a=>d.p[a])); if(p.length<2) return `${group.split}: insufficient points`;
    const steps=[]; for(let i=1;i<p.length;i++) steps.push(Math.hypot(...p[i].map((v,j)=>v-p[i-1][j])));
    const mean=steps.reduce((a,b)=>a+b,0)/steps.length; const sd=Math.sqrt(steps.reduce((a,b)=>a+(b-mean)**2,0)/steps.length); const chord=Math.hypot(...p[p.length-1].map((v,j)=>v-p[0][j]));
    const path=steps.reduce((a,b)=>a+b,0); return `${group.split}: step CV ${(sd/Math.max(mean,1e-9)).toFixed(2)} · path/chord ${(path/Math.max(chord,1e-9)).toFixed(2)}`;
  }).join('<br>');
}
function drawAxes(transform, stats, axes, width, height) {
  const origin=[stats.mins[0],stats.mins[1],stats.mins[2]], ends=[[stats.maxs[0],origin[1],origin[2]],[origin[0],stats.maxs[1],origin[2]],[origin[0],origin[1],stats.maxs[2]]];
  const o=transform(origin); ctx.lineWidth=1; ctx.font='11px system-ui';
  ends.forEach((end,i)=>{ const e=transform(end); ctx.strokeStyle=['#d5a659','#78a6b7','#b889a5'][i]; ctx.beginPath();ctx.moveTo(o.x,o.y);ctx.lineTo(e.x,e.y);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(`PC${axes[i]+1}`,e.x+4,e.y-4); });
}
function draw() {
  const rect=canvas.getBoundingClientRect(), width=rect.width, height=rect.height;
  ctx.clearRect(0,0,width,height); ctx.fillStyle='#0b171c'; ctx.fillRect(0,0,width,height);
  const rows=filteredRows(), axes=[+controls.x.value,+controls.y.value,+controls.z.value];
  const data=activeData(); document.getElementById('pca-stats').innerHTML=data?`<strong>${data.model} · ${data.pooling} · L${data.layer}</strong><br>PCA fit: v4.1 discovery · EVR ${data.explained_variance_ratio.slice(0,6).map((v,i)=>`PC${i+1} ${(100*v).toFixed(1)}%`).join(' · ')}`:'';
  const stats=statsFor(rows,axes), transform=makeTransform(rows,axes,width,height); projectedPoints=[];
  if (!rows.length || !stats || !transform) { ctx.fillStyle='#d7c9a8';ctx.font='16px system-ui';ctx.textAlign='center';ctx.fillText('No trajectories match this filter.',width/2,height/2);document.getElementById('geometry-stats').textContent='No data';return; }
  drawAxes(transform,stats,axes,width,height);
  const paths=centroids(rows); const pointMode=controls.points.value;
  for (const group of paths) {
    const pts=group.path.map(d=>({...d,q:transform(axes.map(a=>d.p[a]))}));
    ctx.strokeStyle=group.split==='confirmation'?'#f4efe1':'#799199'; ctx.lineWidth=group.split==='confirmation'?2.5:1.5; ctx.setLineDash(group.split==='confirmation'?[]:[6,5]);
    ctx.beginPath(); pts.forEach((d,i)=>i?ctx.lineTo(d.q.x,d.q.y):ctx.moveTo(d.q.x,d.q.y));ctx.stroke();ctx.setLineDash([]);
    for (const d of pts) { ctx.fillStyle=COLORS[d.count-1];ctx.strokeStyle='#071115';ctx.lineWidth=1;ctx.beginPath();ctx.arc(d.q.x,d.q.y,5.6,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#f7f0dd';ctx.font='10px system-ui';ctx.fillText(String(d.count),d.q.x+7,d.q.y-6); }
  }
  if (pointMode!=='centroids') {
    let pointRows=rows; if(pointMode==='confirmation') pointRows=rows.filter(r=>r[2]==='confirmation');
    const pts=pointRows.map(r=>({r,q:transform(axes.map(a=>r[7+a]))})).sort((a,b)=>a.q.z-b.q.z);
    for(const item of pts){const r=item.r,q=item.q;ctx.globalAlpha=r[2]==='confirmation'?.56:.18;ctx.fillStyle=COLORS[r[6]-1];ctx.strokeStyle=r[3]==='correct'?'#ffffff':(r[3]==='invalid'?'#ef7d71':'#071115');ctx.lineWidth=r[3]==='correct'?1.8:.7;ctx.beginPath();ctx.arc(q.x,q.y,r[2]==='confirmation'?3.0:2.2,0,Math.PI*2);ctx.fill();ctx.stroke();projectedPoints.push({x:q.x,y:q.y,r});} ctx.globalAlpha=1;
  }
  ctx.fillStyle='#9fb0b4';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText(`${rows.length} occurrence points · ${new Set(rows.map(r=>r[1])).size} seeds`,12,height-12);
  document.getElementById('geometry-stats').innerHTML=geometryText(paths,axes);
}
function reset(){yaw=-0.72;pitch=.44;zoom=1;draw();}
Object.values(controls).forEach(el=>el.addEventListener('change',draw));
controls.preset.addEventListener('change',()=>{const a=controls.preset.value.split(',');controls.x.value=a[0];controls.y.value=a[1];controls.z.value=a[2];draw();});
document.getElementById('reset-view').addEventListener('click',reset);
canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(dragging){yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY;draw();return;} const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=null,dist=Infinity;for(const p of projectedPoints){const d=(p.x-x)**2+(p.y-y)**2;if(d<dist){dist=d;best=p;}}if(best&&dist<80){const r=best.r;tooltip.style.display='block';tooltip.style.left=`${Math.min(rect.width-250,x+14)}px`;tooltip.style.top=`${Math.max(8,y-10)}px`;tooltip.innerHTML=`<strong>${r[0]} · seed ${r[1]} · index ${r[6]}</strong><br>${r[2]} · output ${r[3]} · predicted ${r[4]??'invalid'} · error ${r[5]??'—'}`;}else tooltip.style.display='none';});
canvas.addEventListener('pointerup',()=>{dragging=false;canvas.classList.remove('dragging');}); canvas.addEventListener('pointercancel',()=>{dragging=false;canvas.classList.remove('dragging');}); canvas.addEventListener('mouseleave',()=>{tooltip.style.display='none';});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.45,Math.min(2.8,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
document.getElementById('count-legend').innerHTML=COLORS.map((c,i)=>`<span><i style="background:${c}"></i>${i+1}</span>`).join('');
new ResizeObserver(resizeCanvas).observe(canvas); resizeCanvas();
</script>
</body>
</html>'''


def build_report(run_root: Path, output: Path, repo_root: Path) -> None:
    run_root = run_root.resolve()
    primary: dict[str, dict[str, int]] = {}
    labels_lookup: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    labels_frames: dict[str, pd.DataFrame] = {}
    projections: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_root = run_root / model / "numeric"
        primary[model] = _primary_layers(model_root)
        labels_lookup[model], labels_frames[model] = _n10_labels(model_root)
        for pooling in POOLINGS:
            key = f"{model}|{pooling}"
            projections[key] = _load_projection(
                model_root,
                model=model,
                pooling=pooling,
                layer=primary[model][pooling],
                labels=labels_lookup[model],
            )

    metric_rows = _metric_rows(run_root, primary)
    behavior_rows = _behavior_rows(labels_frames)
    sensitivity_rows = _sensitivity_rows(run_root)
    span_end_alignment_rows = _span_end_alignment_rows(run_root)
    span_end_pooled_rows = _span_end_pooled_rows(run_root)
    span_end_nested_rows = _span_end_nested_rows(run_root)
    commit = _git_commit(repo_root)
    replacements = {
        "@@COMMIT@@": html.escape(commit[:12]),
        "@@GENERATED@@": html.escape(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ),
        "@@RUN_NAME@@": html.escape(run_root.name),
        "@@METRIC_ROWS@@": _table_metric_html(metric_rows),
        "@@BEHAVIOR_ROWS@@": _table_behavior_html(behavior_rows),
        "@@SENSITIVITY_ROWS@@": _table_sensitivity_html(sensitivity_rows),
        "@@SPAN_END_ALIGNMENT_ROWS@@": _table_span_end_alignment_html(
            span_end_alignment_rows
        ),
        "@@SPAN_END_POOLED_ROWS@@": _table_span_end_pooled_html(
            span_end_pooled_rows
        ),
        "@@SPAN_END_ALIGNMENT_SVG@@": _span_end_alignment_svg(
            span_end_alignment_rows
        ),
        "@@SPAN_END_NESTED_ROWS@@": _table_span_end_nested_html(
            span_end_nested_rows
        ),
        "@@SPAN_END_NESTED_SVG@@": _span_end_nested_svg(span_end_nested_rows),
        "@@SPAN_END_CONCLUSION@@": _span_end_conclusion_html(
            span_end_pooled_rows, span_end_alignment_rows, span_end_nested_rows
        ),
        "@@STATIC_FIGURES@@": _static_figure_html(run_root),
        "@@REP_DATA@@": json.dumps(
            projections, ensure_ascii=False, separators=(",", ":")
        ),
    }
    rendered = REPORT_TEMPLATE
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "projection_panels": len(projections),
                "projection_rows": sum(len(item["rows"]) for item in projections.values()),
                "primary_layers": primary,
                "span_end_pooled": span_end_pooled_rows,
                "span_end_alignment": span_end_alignment_rows,
                "span_end_nested": span_end_nested_rows,
                "commit": commit,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained V4 representation HTML report."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    build_report(args.run_root, args.output, args.repo_root)


if __name__ == "__main__":
    main()
