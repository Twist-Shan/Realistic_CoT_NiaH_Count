from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from realistic_niah_v4.causal_audit import (  # noqa: E402
    audit_screen_8h,
    find_screen_designs,
)
from realistic_niah_v4.answer_query_patching import (  # noqa: E402
    audit_answer_query_patching,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
POOLINGS = ("span_end", "span_mean")
VARIANTS = ("v4.1", "v4.2", "v4.3", "v4.4")
VARIANT_DESCRIPTIONS = {
    "v4.1": "position, city-score order, and city-score content fixed",
    "v4.2": "position released; order and content fixed",
    "v4.3": "position and city-score order released; content fixed",
    "v4.4": "position, order, and city-score content all released",
}

# Aurora is the project-wide plotting system for V4 and later reports.  The
# chromatic anchors come from the user-provided Aurora reference; intermediate
# count colors are restrained blends of those anchors so that the ordered
# 1--10 trajectory remains legible without introducing a second palette.
AURORA = {
    "midnight_indigo": "#23165C",
    "polar_violet": "#6750E8",
    "ice_cyan": "#00C2FF",
    "aurora_yellow": "#F6E36A",
    "aurora_teal": "#00D4B4",
    "aurora_green": "#39E58C",
    "polar_magenta": "#C04DFF",
    "sunset_pink": "#FF5FA2",
    "night_black": "#161923",
    "snow_white": "#F8FBFF",
    "frost_gray": "#8190A5",
    "warm_brown": "#765347",
}
MODEL_COLORS = {
    "Qwen3-8B": AURORA["polar_violet"],
    "Gemma4-E4B": AURORA["aurora_teal"],
}
VARIANT_COLORS = {
    "v4.1": AURORA["midnight_indigo"],
    "v4.2": AURORA["polar_violet"],
    "v4.3": AURORA["ice_cyan"],
    "v4.4": AURORA["sunset_pink"],
}
POOLING_COLORS = {
    "span_end": AURORA["ice_cyan"],
    "span_mean": AURORA["sunset_pink"],
}
COUNT_COLORS = (
    "#23165C",
    "#4430A2",
    "#6750E8",
    "#9950F4",
    "#C04DFF",
    "#FF5FA2",
    "#F6E36A",
    "#39E58C",
    "#00D4B4",
    "#00C2FF",
)
PHENOTYPE_COLORS = {
    "global_endpoint_aggregator": AURORA["aurora_green"],
    "partition_local_endpoint_aggregator": AURORA["ice_cyan"],
    "occurrence_endpoint_selector": AURORA["sunset_pink"],
    "other": AURORA["frost_gray"],
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


def _n10_labels(
    model_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], pd.DataFrame]:
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
    tensors: dict[str, list[tuple[int, str, np.ndarray]]] = {
        variant: [] for variant in VARIANTS
    }
    for record in records:
        variant = str(record["design_variant"])
        if variant not in tensors:
            continue
        shard = capture_root / str(record["shard_path"])
        with np.load(shard, allow_pickle=False) as payload:
            layer_indices = np.asarray(payload["layer_indices"], dtype=int)
            match = np.flatnonzero(layer_indices == int(layer))
            if len(match) != 1:
                raise RuntimeError(
                    f"{model}/{pooling}: layer {layer} absent in {shard}"
                )
            states = np.asarray(payload[pooling][int(match[0])], dtype=np.float32)
        if states.shape[0] != 10:
            raise RuntimeError(f"Expected ten occurrence states, got {states.shape}")
        tensors[variant].append((int(record["seed"]), str(record["split"]), states))
    for variant in VARIANTS:
        tensors[variant].sort(key=lambda item: item[0])
        if len(tensors[variant]) != 30:
            raise RuntimeError(
                f"{model}/{pooling}/{variant}: expected 30 seed captures"
            )

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
                raise RuntimeError(
                    f"Missing final-output label for {model}/{variant}/seed{seed}"
                )
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


def _metric_rows(
    run_root: Path, primary: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        analysis = run_root / model / "numeric" / "representation" / "analysis"
        metrics = pd.read_csv(analysis / "representation_layer_metrics.csv")
        for pooling in POOLINGS:
            layer = primary[model][pooling]
            selected = metrics[
                (metrics["pooling"] == pooling)
                & (metrics["layer"].astype(int) == layer)
            ]
            for row in selected.to_dict("records"):
                rows.append(row)
    return rows


def _behavior_rows(labels_by_model: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        for (variant, split), frame in labels.groupby(
            ["design_variant", "split"], sort=True
        ):
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
                    "mae": float(
                        pd.to_numeric(frame["count_error"], errors="coerce")
                        .abs()
                        .mean()
                    ),
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
    frame["exact"] = (frame["overlap_count"] == frame["omission_count"]).astype(float)
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
        variant_counts = (
            seed_variant.reset_index().groupby("seed")["design_variant"].nunique()
        )
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

        block_status = frame.groupby(["seed", "design_variant", "status"], sort=True)[
            ["new_in_bottom_k", "new_needle_normalized_share"]
        ].mean()
        wide_bottom = block_status["new_in_bottom_k"].unstack("status").dropna()
        wide_share = (
            block_status["new_needle_normalized_share"].unstack("status").dropna()
        )
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
            (wide_bottom["failed"] - wide_bottom["registered"])
            .groupby(level="seed")
            .mean()
            .to_numpy()
        )
        share_difference = (
            (wide_share["registered"] - wide_share["failed"])
            .groupby(level="seed")
            .mean()
            .to_numpy()
        )
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


def _representation_r2_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1080, 420
    panel_lefts = (84, 604)
    plot_width, top, bottom = 390, 72, 322
    y_min, y_max = -0.25, 1.05

    def x_position(variant: str, left: float) -> float:
        return left + VARIANTS.index(variant) / (len(VARIANTS) - 1) * plot_width

    def y_position(value: float) -> float:
        bounded = max(y_min, min(y_max, float(value)))
        return bottom - (bounded - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="representation-r2-title representation-r2-desc">',
        '<title id="representation-r2-title">Held-out ridge count decoding across the V4 relaxation ladder</title>',
        '<desc id="representation-r2-desc">Span-end decoding remains positive through v4.4, while span-mean decoding collapses after city-score order is released, especially in Gemma.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for index, pooling in enumerate(POOLINGS):
        x = 392 + index * 170
        color = POOLING_COLORS[pooling]
        dash = "" if pooling == "span_end" else ' stroke-dasharray="7 5"'
        parts.extend(
            [
                f'<line x1="{x}" y1="28" x2="{x+28}" y2="28" stroke="{color}" stroke-width="4"{dash}/>',
                f'<circle cx="{x+14}" cy="28" r="4" fill="{color}"/>',
                f'<text x="{x+36}" y="32" font-size="12">{pooling.replace("_", "-")}</text>',
            ]
        )
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        parts.append(
            f'<text x="{left}" y="54" font-size="17" font-weight="700">{html.escape(model)}</text>'
        )
        for tick in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_position(tick)
            line_color = AURORA["warm_brown"] if tick == 0 else AURORA["frost_gray"]
            line_width = 1.5 if tick == 0 else 1
            opacity = 0.65 if tick == 0 else 0.28
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                    f'stroke="{line_color}" stroke-width="{line_width}" opacity="{opacity}"/>',
                    f'<text x="{left-11}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                    f'fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
                ]
            )
            if tick == 0:
                parts.append(
                    f'<text x="{left+plot_width-4}" y="{y-6:.1f}" text-anchor="end" '
                    f'font-size="10" fill="{AURORA["warm_brown"]}">R²=0</text>'
                )
        for variant in VARIANTS:
            x = x_position(variant, left)
            parts.append(
                f'<text x="{x:.1f}" y="344" text-anchor="middle" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{variant}</text>'
            )
        for pooling in POOLINGS:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model_label"] == model and row["pooling"] == pooling
                ],
                key=lambda row: VARIANTS.index(str(row["design_variant"])),
            )
            points = [
                (
                    x_position(str(row["design_variant"]), left),
                    y_position(float(row["confirmation_r2"])),
                )
                for row in selected
            ]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
                for index, (x, y) in enumerate(points)
            )
            color = POOLING_COLORS[pooling]
            dash = "" if pooling == "span_end" else ' stroke-dasharray="7 5"'
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" '
                    f'stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>'
                )
        parts.extend(
            [
                f'<text x="{left + plot_width/2:.1f}" y="388" text-anchor="middle" font-size="12">controlled relaxation panel</text>',
                f'<text transform="translate({left-58} {(top+bottom)/2:.1f}) rotate(-90)" '
                'text-anchor="middle" font-size="12">confirmation R²</text>',
            ]
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _representation_conclusion_html(
    rows: list[dict[str, Any]], sensitivity_rows: list[dict[str, Any]]
) -> str:
    summaries: list[str] = []
    for model in MODELS:
        end = next(
            row
            for row in rows
            if row["model_label"] == model
            and row["pooling"] == "span_end"
            and row["design_variant"] == "v4.4"
        )
        mean = next(
            row
            for row in rows
            if row["model_label"] == model
            and row["pooling"] == "span_mean"
            and row["design_variant"] == "v4.4"
        )
        summaries.append(
            f"{html.escape(model)} 在 v4.4 的 span-end R²={_number(end['confirmation_r2'])}，"
            f"span-mean R²={_number(mean['confirmation_r2'])}"
        )
    first_noise: list[str] = []
    for model in MODELS:
        selected = [
            row
            for row in sensitivity_rows
            if row["model"] == model
            and row["pooling"] == "span_end"
            and row["metric"] == "curve_residual_to_signal"
            and _bool(row["increase_ci_excludes_zero"])
        ]
        if selected:
            row = selected[0]
            first_noise.append(
                f"{html.escape(model)} 在 {html.escape(str(row['left_variant']))}→{html.escape(str(row['right_variant']))}"
            )
    return (
        '<div class="section-conclusion"><span>本节结论</span><p>'
        + "；".join(summaries)
        + "。因此 needle 末端一直保留可解码的 count-related signal，但它不是低噪声、等间距的标量计数器；span-mean 在释放 city-score 顺序后明显崩溃，说明其早期高分主要依赖固定记录结构。span-end 的 seed-noise 首次显著上升分别出现在 "
        + "、".join(first_noise)
        + "。这些证据证明信息可用性，不证明生成必然读取该方向。</p></div>"
    )


def _attention_top_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            run_root
            / model
            / "numeric"
            / "attention"
            / "analysis"
            / "tables"
            / "discovery_head_summary.csv"
        )
        frame = pd.read_csv(path)
        rank = pd.to_numeric(frame["candidate_rank"], errors="coerce")
        selected = frame[rank == 1].copy()
        expected = len(VARIANTS) * len(POOLINGS)
        if len(selected) != expected:
            raise RuntimeError(f"Expected {expected} rank-1 rows in {path}")
        for row in selected.to_dict("records"):
            rows.append(
                {
                    "model": model,
                    "variant": str(row["design_variant"]),
                    "pooling": str(row["pooling"]),
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "coverage": float(row["pool_coverage"]),
                    "effective_number": float(row["pool_effective_number"]),
                    "primary": float(row["pool_primary"]),
                    "total_mass": float(row["pool_sum"]),
                    "enrichment": float(row["pool_enrichment"]),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            MODELS.index(str(row["model"])),
            POOLINGS.index(str(row["pooling"])),
            VARIANTS.index(str(row["variant"])),
        ),
    )


def _qwen_partition_summary(run_root: Path) -> dict[str, Any]:
    root = (
        run_root
        / "Qwen3-8B"
        / "numeric"
        / "attention"
        / "analysis"
        / "partitioning"
    )
    counts = pd.read_csv(root / "all_candidate_phenotype_counts.csv")
    bank = pd.read_csv(root / "phenotype_bank_coverage.csv")
    by_split = pd.read_csv(root / "all_candidate_head_phenotypes_by_split.csv")
    assessment = _read_json(root / "partition_hypothesis_assessment.json")
    manifest = _read_json(root / "partition_analysis_manifest.json")

    key_phenotypes = tuple(PHENOTYPE_COLORS)[:-1]
    count_lookup = {
        (str(row.design_variant), str(row.phenotype)): int(row.heads)
        for row in counts.itertuples()
    }
    total_lookup = {
        variant: int(sum(count_lookup[(variant, phenotype)] for phenotype in counts[counts["design_variant"] == variant]["phenotype"].astype(str)))
        for variant in VARIANTS
    }
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for phenotype in key_phenotypes:
            selected = bank[
                (bank["design_variant"] == variant)
                & (bank["phenotype"] == phenotype)
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Missing Qwen partition bank {variant}/{phenotype}")
            item = selected.iloc[0]
            rows.append(
                {
                    "variant": variant,
                    "phenotype": phenotype,
                    "heads": count_lookup[(variant, phenotype)],
                    "equal_effective_number": float(
                        item["equal_head_profile_effective_number"]
                    ),
                    "raw_effective_number": float(
                        item["raw_attention_ensemble_effective_number"]
                    ),
                    "mean_bank_mass": float(item["mean_summed_bank_endpoint_mass"]),
                }
            )

    global_rows = by_split[
        by_split["phenotype"] == "global_endpoint_aggregator"
    ].copy()
    global_rows["cell"] = (
        global_rows["design_variant"].astype(str)
        + "|"
        + global_rows["split"].astype(str)
    )
    stable = (
        global_rows.groupby(["layer", "head"])["cell"]
        .nunique()
        .loc[lambda values: values == len(VARIANTS) * 2]
        .index.tolist()
    )
    stable_labels = [f"L{int(layer)}H{int(head)}" for layer, head in stable]
    return {
        "rows": rows,
        "counts": count_lookup,
        "totals": total_lookup,
        "stable_global_heads": stable_labels,
        "assessment": assessment,
        "candidate_counts": {
            str(key): int(value)
            for key, value in manifest["candidate_counts"].items()
        },
    }


def _table_attention_top_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td><code>{html.escape(str(row['pooling']))}</code></td>"
            f"<td>L{int(row['layer'])}H{int(row['head'])}</td>"
            f"<td>{_number(row['total_mass'], 6)}</td>"
            f"<td>{_number(row['coverage'])}</td>"
            f"<td>{_number(row['effective_number'], 2)}</td>"
            f"<td>{_number(row['primary'], 6)}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_partition_bank_html(rows: list[dict[str, Any]]) -> str:
    labels = {
        "global_endpoint_aggregator": "global endpoint aggregator",
        "partition_local_endpoint_aggregator": "partition-local aggregator",
        "occurrence_endpoint_selector": "occurrence selector",
    }
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{html.escape(labels[str(row['phenotype'])])}</td>"
            f"<td>{int(row['heads'])}</td>"
            f"<td>{_number(row['equal_effective_number'], 2)}</td>"
            f"<td>{_number(row['raw_effective_number'], 2)}</td>"
            f"<td>{_number(row['mean_bank_mass'], 5)}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _attention_breadth_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1080, 430
    panel_lefts = (80, 600)
    plot_width, top, bottom = 400, 74, 330
    bar_width = 27

    def y_position(value: float) -> float:
        return bottom - max(0.0, min(10.0, float(value))) / 10.0 * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="breadth-title breadth-desc">',
        '<title id="breadth-title">Effective number of needles covered by each discovery rank-1 attention head</title>',
        '<desc id="breadth-desc">Qwen span-end rank-1 attention is a selector covering one occurrence, whereas Qwen span-mean and both Gemma poolings are broader.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for index, pooling in enumerate(POOLINGS):
        x = 404 + index * 170
        color = POOLING_COLORS[pooling]
        parts.extend(
            [
                f'<rect x="{x}" y="19" width="15" height="15" fill="{color}"/>',
                f'<text x="{x+23}" y="32" font-size="12">{pooling.replace("_", "-")}</text>',
            ]
        )
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        parts.append(
            f'<text x="{left}" y="54" font-size="17" font-weight="700">{html.escape(model)}</text>'
        )
        for tick in range(0, 11, 2):
            y = y_position(tick)
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                    f'stroke="{AURORA["frost_gray"]}" opacity=".27"/>',
                    f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                    f'fill="{AURORA["frost_gray"]}">{tick}</text>',
                ]
            )
        group_width = plot_width / len(VARIANTS)
        for variant_index, variant in enumerate(VARIANTS):
            center = left + (variant_index + 0.5) * group_width
            parts.append(
                f'<text x="{center:.1f}" y="352" text-anchor="middle" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{variant}</text>'
            )
            for pooling_index, pooling in enumerate(POOLINGS):
                row = next(
                    item
                    for item in rows
                    if item["model"] == model
                    and item["variant"] == variant
                    and item["pooling"] == pooling
                )
                x = center + (pooling_index - 0.5) * 34 - bar_width / 2
                y = y_position(row["effective_number"])
                color = POOLING_COLORS[pooling]
                parts.extend(
                    [
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bottom-y:.1f}" '
                        f'fill="{color}" opacity=".88"/>',
                        f'<text x="{x+bar_width/2:.1f}" y="{y-7:.1f}" text-anchor="middle" '
                        f'font-size="10" fill="{AURORA["night_black"]}">{float(row["effective_number"]):.1f}</text>',
                    ]
                )
        parts.extend(
            [
                f'<text x="{left+plot_width/2:.1f}" y="399" text-anchor="middle" font-size="12">V4 panel</text>',
                f'<text transform="translate({left-50} {(top+bottom)/2:.1f}) rotate(-90)" '
                'text-anchor="middle" font-size="12">effective number N_eff (max 10)</text>',
            ]
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _partition_phenotype_svg(summary: dict[str, Any]) -> str:
    width, height = 900, 430
    left, top, bottom, plot_width = 105, 70, 335, 690
    totals = summary["totals"]
    y_max = max(totals.values()) * 1.05
    order = (
        "global_endpoint_aggregator",
        "partition_local_endpoint_aggregator",
        "occurrence_endpoint_selector",
        "other",
    )
    labels = {
        "global_endpoint_aggregator": "global aggregator",
        "partition_local_endpoint_aggregator": "partition-local",
        "occurrence_endpoint_selector": "selector",
        "other": "other phenotypes",
    }

    def y_position(value: float) -> float:
        return bottom - float(value) / y_max * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="phenotype-title phenotype-desc">',
        '<title id="phenotype-title">Qwen span-end attention head phenotypes</title>',
        '<desc id="phenotype-desc">Many global aggregators coexist with a larger selector population; partition-local heads become rarer after v4.1.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    legend_x = 125
    for index, phenotype in enumerate(order):
        x = legend_x + index * 180
        color = PHENOTYPE_COLORS[phenotype]
        parts.extend(
            [
                f'<rect x="{x}" y="22" width="14" height="14" fill="{color}"/>',
                f'<text x="{x+21}" y="34" font-size="11">{labels[phenotype]}</text>',
            ]
        )
    for tick in range(0, 251, 50):
        y = y_position(tick)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                f'stroke="{AURORA["frost_gray"]}" opacity=".27"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{tick}</text>',
            ]
        )
    group_width = plot_width / len(VARIANTS)
    for index, variant in enumerate(VARIANTS):
        x = left + index * group_width + 38
        bar_width = group_width - 76
        cumulative = 0
        key_total = 0
        for phenotype in order[:-1]:
            key_total += int(summary["counts"][(variant, phenotype)])
        values = {
            phenotype: int(summary["counts"][(variant, phenotype)])
            for phenotype in order[:-1]
        }
        values["other"] = int(totals[variant]) - key_total
        for phenotype in order:
            value = values[phenotype]
            y_top = y_position(cumulative + value)
            y_bottom = y_position(cumulative)
            parts.append(
                f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" '
                f'height="{y_bottom-y_top:.1f}" fill="{PHENOTYPE_COLORS[phenotype]}"/>'
            )
            if value >= 18:
                parts.append(
                    f'<text x="{x+bar_width/2:.1f}" y="{(y_top+y_bottom)/2+4:.1f}" '
                    f'text-anchor="middle" font-size="11" font-weight="700" '
                    f'fill="{AURORA["night_black"]}">{value}</text>'
                )
            cumulative += value
        parts.extend(
            [
                f'<text x="{x+bar_width/2:.1f}" y="{y_position(totals[variant])-8:.1f}" '
                f'text-anchor="middle" font-size="11">n={totals[variant]}</text>',
                f'<text x="{x+bar_width/2:.1f}" y="360" text-anchor="middle" font-size="12">{variant}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left+plot_width/2:.1f}" y="407" text-anchor="middle" font-size="12">V4 panel</text>',
            f'<text transform="translate(35 {(top+bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12">discovery-eligible head count</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _attention_conclusion_html(
    top_rows: list[dict[str, Any]], partition: dict[str, Any]
) -> str:
    qwen_end = [
        row
        for row in top_rows
        if row["model"] == "Qwen3-8B" and row["pooling"] == "span_end"
    ]
    qwen_mean = [
        row
        for row in top_rows
        if row["model"] == "Qwen3-8B" and row["pooling"] == "span_mean"
    ]
    gemma_end = [
        row
        for row in top_rows
        if row["model"] == "Gemma4-E4B" and row["pooling"] == "span_end"
    ]
    assessment = partition["assessment"]["assessments"]
    first_share = float(np.mean([row["endpoint_first_occurrence_share"] for row in assessment]))
    global_counts = [
        partition["counts"][(variant, "global_endpoint_aggregator")]
        for variant in VARIANTS
    ]
    local_counts = [
        partition["counts"][(variant, "partition_local_endpoint_aggregator")]
        for variant in VARIANTS
    ]
    return (
        '<div class="section-conclusion"><span>本节结论</span><p>'
        f"Qwen 的 rank-1 span-end head 在四个 panel 的平均 N_eff={np.mean([row['effective_number'] for row in qwen_end]):.2f}，"
        f"约 {100*first_share:.1f}% 的 endpoint share 都落在第一个 occurrence；它是 selector，不是 broad aggregator。"
        f"相反，Qwen span-mean rank-1 的平均 N_eff={np.mean([row['effective_number'] for row in qwen_mean]):.2f}，Gemma span-end rank-1 的平均 N_eff={np.mean([row['effective_number'] for row in gemma_end]):.2f}。"
        f"Qwen 全候选分析仍找到每个 panel {min(global_counts)}–{max(global_counts)} 个 global aggregators，且有 {len(partition['stable_global_heads'])} 个在全部 panel×split cells 中保持该 phenotype；partition-local heads 从 v4.1 的 {local_counts[0]} 个降到其余 panel 的 {min(local_counts[1:])} 个。"
        "因此 broad aggregation 是多 head 分布式机制，不能用最高排名的单个 head 代表；固定、seed-invariant 的分区电路目前证据不足。</p></div>"
    )


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
        f'<rect width="1040" height="390" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        parts.append(
            f'<text x="{left}" y="34" font-size="17" font-weight="700">'
            f"{html.escape(model)}</text>"
        )
        for tick in np.arange(0.0, x_max + 0.001, 0.1):
            x = x_position(float(tick), left)
            parts.append(
                f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="315" '
                f'stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="338" text-anchor="middle" '
                f'font-size="11" fill="{AURORA["frost_gray"]}">{tick:.1f}</text>'
            )
        model_rows = [row for row in rows if row["model"] == model]
        for row_index, row in enumerate(model_rows):
            y = 88 + row_index * 58
            chance_x = x_position(float(row["chance"]), left)
            observed_x = x_position(float(row["overlap"]), left)
            ci_low_x = x_position(float(row["chance"]) + float(row["delta_low"]), left)
            ci_high_x = x_position(
                float(row["chance"]) + float(row["delta_high"]), left
            )
            parts.extend(
                [
                    f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" '
                    f'font-size="12" font-weight="650">{html.escape(str(row["variant"]))}</text>',
                    f'<line x1="{chance_x:.1f}" y1="{y}" x2="{observed_x:.1f}" '
                    f'y2="{y}" stroke="{AURORA["frost_gray"]}" stroke-width="2"/>',
                    f'<line x1="{ci_low_x:.1f}" y1="{y}" x2="{ci_high_x:.1f}" '
                    f'y2="{y}" stroke="{MODEL_COLORS[model]}" stroke-width="5" stroke-linecap="round" opacity=".42"/>',
                    f'<path d="M {chance_x:.1f} {y - 6} L {chance_x + 6:.1f} {y} '
                    f'L {chance_x:.1f} {y + 6} L {chance_x - 6:.1f} {y} Z" fill="{AURORA["aurora_yellow"]}"/>',
                    f'<circle cx="{observed_x:.1f}" cy="{y}" r="6" fill="{MODEL_COLORS[model]}" stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>',
                    f'<text x="{left + panel_width + 9}" y="{y + 4}" font-size="11" fill="{AURORA["night_black"]}">'
                    f'Δ {_number(row["delta"], signed=True)}</text>',
                ]
            )
        parts.append(
            f'<text x="{left + panel_width / 2:.1f}" y="365" text-anchor="middle" '
            f'font-size="12" fill="{AURORA["night_black"]}">tail overlap fraction</text>'
        )
    parts.extend(
        [
            f'<path d="M 761 28 L 767 34 L 761 40 L 755 34 Z" fill="{AURORA["aurora_yellow"]}"/>',
            f'<text x="774" y="38" font-size="11" fill="{AURORA["frost_gray"]}">hypergeometric chance</text>',
            f'<circle cx="910" cy="34" r="6" fill="{AURORA["polar_violet"]}"/>',
            f'<text x="920" y="38" font-size="11" fill="{AURORA["frost_gray"]}">observed (95% seed CI)</text>',
            "</g></svg>",
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
        f'<rect width="920" height="290" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for tick in np.arange(0.0, x_max + 0.001, 0.1):
        x = x_position(float(tick))
        parts.append(
            f'<line x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="222" '
            f'stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="244" text-anchor="middle" font-size="11" '
            f'fill="{AURORA["frost_gray"]}">{tick:.1f}</text>'
        )
    for row_index, row in enumerate(rows):
        center_y = 92 + row_index * 88
        parts.append(
            f'<text x="{left - 18}" y="{center_y + 4}" text-anchor="end" '
            f'font-size="14" font-weight="700">{html.escape(str(row["model"]))}</text>'
        )
        for status, color, offset in (
            ("failed", AURORA["sunset_pink"], -12),
            ("registered", AURORA["aurora_green"], 12),
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
                    f'fill="{color}" stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>',
                ]
            )
        parts.append(
            f'<text x="770" y="{center_y + 4}" font-size="11" fill="{AURORA["night_black"]}">'
            f'RD {_number(row["bottom_difference"], signed=True)} '
            f'[{_number(row["bottom_difference_low"])}, {_number(row["bottom_difference_high"])}]'
            "</text>"
        )
    parts.extend(
        [
            f'<circle cx="590" cy="25" r="6" fill="{AURORA["sunset_pink"]}"/><text x="601" y="29" font-size="11" fill="{AURORA["frost_gray"]}">failed to increment</text>',
            f'<circle cx="720" cy="25" r="6" fill="{AURORA["aurora_green"]}"/><text x="731" y="29" font-size="11" fill="{AURORA["frost_gray"]}">registered +1</text>',
            f'<text x="465" y="273" text-anchor="middle" font-size="12" fill="{AURORA["night_black"]}">P(new needle is in current bottom-k attention set)</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _variant_list(values: list[str]) -> str:
    if not values:
        return "无"
    if len(values) == 1:
        return values[0]
    return "、".join(values)


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
        holm = [str(row["variant"]) for row in selected if float(row["p_holm"]) < 0.05]
        cards.append(
            '<div class="note"><strong>'
            + html.escape(model)
            + " 尾部对齐。</strong><p>四个 panel 等权 pooling 后，tail-alignment contrast 为 "
            + _number(pooled["delta"], signed=True)
            + " ["
            + _number(pooled["delta_low"])
            + ", "
            + _number(pooled["delta_high"])
            + "]，Holm p="
            + _p_value(pooled["p_holm"])
            + "。四个 panel 的点估计都高于机会水平；95% seed-cluster 区间排除 0 的 panel 为 "
            + html.escape(_variant_list(positive_ci))
            + "；exact sign-flip 检验经 Holm 校正后仍小于 0.05 的 panel 为 "
            + html.escape(_variant_list(holm))
            + "。</p></div>"
        )
    nested_sentences: list[str] = []
    for row in nested_rows:
        nested_sentences.append(
            f"{html.escape(str(row['model']))}：risk difference "
            f"{_number(row['bottom_difference'], signed=True)} "
            f"[{_number(row['bottom_difference_low'])}, "
            f"{_number(row['bottom_difference_high'])}], "
            f"Holm p={_p_value(row['bottom_p_holm'])}"
        )
    cards.append(
        '<div class="note"><strong>新增 needle 的 exact 配对检查。</strong><p>'
        + "；".join(nested_sentences)
        + "。正值表示：当输出没有随 gold count 增加时，新加入的 needle 更常落入 "
        + "bottom-k attention set。</p></div>"
    )
    return "".join(cards)


def _causal_frames(
    run_root: Path,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, Any]],
]:
    designs = find_screen_designs(run_root)
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    paths: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        frames[model] = {}
        paths[model] = {}
        for stage in ("ablation", "patching", "steering"):
            selected = designs[model][stage]
            frames[model][stage] = pd.read_csv(
                selected.root / "detail.csv.gz", compression="gzip"
            )
            paths[model][stage] = selected
        frames[model]["geometry"] = pd.read_csv(
            designs[model]["steering"].root / "centroid_geometry_summary.csv"
        )
    return frames, paths


def _answer_query_frames(
    run_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    analysis_root = run_root / "analysis" / "answer_query_patching_dense_v1"
    required = (
        "layer_summary",
        "pair_summary",
        "variant_summary",
        "outcome_summary",
        "stratum_summary",
        "invalid_rows",
    )
    frames: dict[str, pd.DataFrame] = {}
    for name in required:
        path = analysis_root / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing answer-query analysis table {path}; run "
                "scripts/analyze_realistic_niah_v4_answer_query_patching.py first"
            )
        frames[name] = pd.read_csv(path)
    return frames, audit_answer_query_patching(run_root)


def _answer_query_final_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    selected: list[pd.DataFrame] = []
    for model in MODELS:
        model_rows = frame[frame["model"] == model]
        selected.append(model_rows[model_rows["layer"] == model_rows["layer"].max()])
    return pd.concat(selected, ignore_index=True).to_dict("records")


def _answer_query_onset_rows(
    layer_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        selected = layer_frame[layer_frame["model"] == model].sort_values("layer")
        candidates = selected[
            (selected["layer"] > selected["layer"].min())
            & (selected["eligible_donor_adoption_rate"] >= 0.5)
            & (selected["eligible_donor_adoption_vs_layer0_p_holm"] < 0.05)
        ]
        if candidates.empty:
            raise RuntimeError(f"No significant answer-query transport onset for {model}")
        rows.append(candidates.iloc[0].to_dict())
    return rows


def _all_generation_labels(model_root: Path) -> pd.DataFrame:
    """Load the complete 4 panels x 30 seeds x 10 counts behavior grid."""
    path = model_root / "behavior" / "capture" / "generation_labels.csv"
    labels = pd.read_csv(path)
    expected_rows = len(VARIANTS) * 30 * 10
    if len(labels) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} behavior rows in {path}")
    if labels.duplicated(["design_variant", "seed", "gold_count"]).any():
        raise RuntimeError(f"Behavior rows are not unique in {path}")
    labels = labels.copy()
    for field in ("gold_count", "parsed_count", "count_error"):
        labels[field] = pd.to_numeric(labels[field], errors="coerce")
    labels["is_correct_bool"] = labels["is_correct"].map(_bool)
    labels["format_valid_bool"] = labels["format_valid"].map(_bool)
    return labels


def _behavior_panel_rows(
    labels_by_model: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        selected = labels[labels["split"] == "confirmation"].copy()
        for variant, frame in selected.groupby("design_variant", sort=True):
            error = pd.to_numeric(frame["count_error"], errors="coerce")
            prediction = pd.to_numeric(frame["parsed_count"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "variant": str(variant),
                    "n": int(len(frame)),
                    "correct": int(frame["is_correct_bool"].sum()),
                    "accuracy": float(frame["is_correct_bool"].mean()),
                    "format_valid": float(frame["format_valid_bool"].mean()),
                    "mean_prediction": float(prediction.mean()),
                    "mae": float(error.abs().mean()),
                    "undercount_rate": float((error < 0).mean()),
                    "overcount_rate": float((error > 0).mean()),
                }
            )
    return rows


def _behavior_count_rows(
    labels_by_model: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        selected = labels[labels["split"] == "confirmation"].copy()
        for (variant, count), frame in selected.groupby(
            ["design_variant", "gold_count"], sort=True
        ):
            error = pd.to_numeric(frame["count_error"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "variant": str(variant),
                    "count": int(count),
                    "n": int(len(frame)),
                    "accuracy": float(frame["is_correct_bool"].mean()),
                    "undercount_rate": float((error < 0).mean()),
                    "mean_prediction": float(
                        pd.to_numeric(frame["parsed_count"], errors="coerce").mean()
                    ),
                }
            )
    return rows


def _behavior_count_pooled_rows(
    labels_by_model: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        selected = labels[labels["split"] == "confirmation"].copy()
        for count, frame in selected.groupby("gold_count", sort=True):
            error = pd.to_numeric(frame["count_error"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "count": int(count),
                    "n": int(len(frame)),
                    "correct": int(frame["is_correct_bool"].sum()),
                    "accuracy": float(frame["is_correct_bool"].mean()),
                    "mean_prediction": float(
                        pd.to_numeric(frame["parsed_count"], errors="coerce").mean()
                    ),
                    "undercount_rate": float((error < 0).mean()),
                }
            )
    return rows


def _table_behavior_panel_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{int(row['correct'])}/{int(row['n'])}</td>"
            f"<td>{100 * float(row['accuracy']):.1f}%</td>"
            f"<td>{_number(row['mean_prediction'], 2)}</td>"
            f"<td>{_number(row['mae'], 2)}</td>"
            f"<td>{100 * float(row['undercount_rate']):.1f}%</td>"
            f"<td>{100 * float(row['format_valid']):.1f}%</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_behavior_count_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{int(row['count'])}</td>"
            f"<td>{int(row['correct'])}/{int(row['n'])}</td>"
            f"<td>{100 * float(row['accuracy']):.1f}%</td>"
            f"<td>{_number(row['mean_prediction'], 2)}</td>"
            f"<td>{100 * float(row['undercount_rate']):.1f}%</td>"
            "</tr>"
        )
    return "".join(rendered)


def _behavior_accuracy_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1080, 450
    panel_lefts = (74, 604)
    plot_width, top, bottom = 390, 76, 350

    def x_position(count: int, left: float) -> float:
        return left + (int(count) - 1) / 9 * plot_width

    def y_position(value: float) -> float:
        return bottom - max(0.0, min(1.0, float(value))) * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="behavior-plot-title behavior-plot-desc">',
        '<title id="behavior-plot-title">Confirmation accuracy by true count and V4 panel</title>',
        '<desc id="behavior-plot-desc">Accuracy falls sharply at medium and high counts in both models; each line is one controlled-relaxation panel.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    legend_x = 292
    for index, variant in enumerate(VARIANTS):
        x = legend_x + index * 125
        color = VARIANT_COLORS[variant]
        parts.extend(
            [
                f'<line x1="{x}" y1="30" x2="{x+24}" y2="30" stroke="{color}" stroke-width="4"/>',
                f'<circle cx="{x+12}" cy="30" r="4" fill="{color}"/>',
                f'<text x="{x+31}" y="34" font-size="12">{variant}</text>',
            ]
        )
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        high_count_x = x_position(5, left) - 18
        parts.append(
            f'<rect x="{high_count_x:.1f}" y="{top}" width="{left + plot_width - high_count_x:.1f}" '
            f'height="{bottom-top}" fill="{AURORA["aurora_yellow"]}" opacity=".12"/>'
        )
        parts.append(
            f'<text x="{left}" y="58" font-size="17" font-weight="700">{html.escape(model)}</text>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_position(tick)
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                    f'stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>',
                    f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                    f'fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
                ]
            )
        for count in range(1, 11):
            x = x_position(count, left)
            parts.append(
                f'<text x="{x:.1f}" y="371" text-anchor="middle" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{count}</text>'
            )
        for variant in VARIANTS:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == model and row["variant"] == variant
                ],
                key=lambda row: int(row["count"]),
            )
            points = [
                (x_position(int(row["count"]), left), y_position(row["accuracy"]))
                for row in selected
            ]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
                for index, (x, y) in enumerate(points)
            )
            color = VARIANT_COLORS[variant]
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{color}" '
                    f'stroke="{AURORA["snow_white"]}" stroke-width="1.2"/>'
                )
        parts.extend(
            [
                f'<text x="{left + plot_width/2:.1f}" y="414" text-anchor="middle" font-size="12">true needle count N</text>',
                f'<text transform="translate({left-53} {(top+bottom)/2:.1f}) rotate(-90)" '
                'text-anchor="middle" font-size="12">greedy exact-match accuracy</text>',
            ]
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _behavior_conclusion_html(
    panel_rows: list[dict[str, Any]], pooled_rows: list[dict[str, Any]]
) -> str:
    sentences: list[str] = []
    for model in MODELS:
        model_panels = [row for row in panel_rows if row["model"] == model]
        model_counts = [row for row in pooled_rows if row["model"] == model]
        first_below_half = next(
            int(row["count"])
            for row in sorted(model_counts, key=lambda item: int(item["count"]))
            if float(row["accuracy"]) < 0.5
        )
        sentences.append(
            f"{html.escape(model)} 的 confirmation panel accuracy 为 "
            + "/".join(f"{100*float(row['accuracy']):.0f}%" for row in model_panels)
            + f"（v4.1→v4.4），首次低于 50% 出现在 N={first_below_half}"
        )
    return (
        '<div class="section-conclusion"><span>本节结论</span><p>'
        + "；".join(sentences)
        + "。主要行为边界由 count 大小而不是 V4 panel 决定；高 count 的错误几乎都是 undercount，因此后续机制分析应解释为什么证据未被完整聚合，而不能只比较总体 accuracy。</p></div>"
    )


def _paired_seed_contrast(
    frame: pd.DataFrame,
    *,
    metric: str,
    condition_column: str,
    treatment: str,
    control: str,
    identity_columns: list[str],
    label: str,
) -> dict[str, Any]:
    pivot = frame.pivot(
        index=identity_columns,
        columns=condition_column,
        values=metric,
    )
    missing = sorted({treatment, control} - set(pivot.columns))
    if missing or pivot[[treatment, control]].isna().any().any():
        raise RuntimeError(f"{label}: incomplete paired conditions {missing}")
    differences = (pivot[treatment] - pivot[control]).rename("difference").reset_index()
    seed_values = differences.groupby("seed", sort=True)["difference"].mean().to_numpy()
    if len(seed_values) != 10:
        raise RuntimeError(f"{label}: expected ten paired confirmation seeds")
    estimate, low, high = _seed_bootstrap(seed_values, label=label)
    return {
        "estimate": estimate,
        "low": low,
        "high": high,
        "p_raw": _exact_sign_flip_p(seed_values),
        "seed_values": seed_values,
    }


def _one_sample_seed_estimate(
    frame: pd.DataFrame,
    *,
    metric: str,
    label: str,
) -> dict[str, Any]:
    seed_values = frame.groupby("seed", sort=True)[metric].mean().to_numpy()
    if len(seed_values) != 10:
        raise RuntimeError(f"{label}: expected ten confirmation seeds")
    estimate, low, high = _seed_bootstrap(seed_values, label=label)
    return {
        "estimate": estimate,
        "low": low,
        "high": high,
        "p_raw": _exact_sign_flip_p(seed_values),
        "seed_values": seed_values,
    }


def _causal_ablation_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        detail = frames[model]["ablation"].copy()
        detail["prediction_changed_numeric"] = detail["prediction_changed"].astype(
            float
        )
        for top_n in (4, 8):
            selected = detail[detail["top_n"].astype(int) == top_n].copy()
            identity = ["design_variant", "seed", "stimulus_id", "top_n"]
            changed = _paired_seed_contrast(
                selected,
                metric="prediction_changed_numeric",
                condition_column="condition",
                treatment="ranked",
                control="layer_matched_random",
                identity_columns=identity,
                label=f"causal-ablation-changed-{model}-top{top_n}",
            )
            count_shift = _paired_seed_contrast(
                selected,
                metric="generated_count_shift",
                condition_column="condition",
                treatment="ranked",
                control="layer_matched_random",
                identity_columns=identity,
                label=f"causal-ablation-shift-{model}-top{top_n}",
            )
            error = _paired_seed_contrast(
                selected,
                metric="absolute_error_delta",
                condition_column="condition",
                treatment="ranked",
                control="layer_matched_random",
                identity_columns=identity,
                label=f"causal-ablation-error-{model}-top{top_n}",
            )
            ranked = selected[selected["condition"] == "ranked"]
            control = selected[selected["condition"] == "layer_matched_random"]
            baseline = selected.drop_duplicates("stimulus_id")
            rows.append(
                {
                    "model": model,
                    "top_n": top_n,
                    "prompts": int(baseline["stimulus_id"].nunique()),
                    "baseline_correct": int(
                        baseline["baseline_is_correct"].astype(bool).sum()
                    ),
                    "ranked_changed": float(
                        ranked["prediction_changed_numeric"].mean()
                    ),
                    "random_changed": float(
                        control["prediction_changed_numeric"].mean()
                    ),
                    "changed_difference": changed["estimate"],
                    "changed_difference_low": changed["low"],
                    "changed_difference_high": changed["high"],
                    "ranked_count_shift": float(ranked["generated_count_shift"].mean()),
                    "random_count_shift": float(
                        control["generated_count_shift"].mean()
                    ),
                    "count_shift_difference": count_shift["estimate"],
                    "count_shift_difference_low": count_shift["low"],
                    "count_shift_difference_high": count_shift["high"],
                    "count_shift_p_raw": count_shift["p_raw"],
                    "error_difference": error["estimate"],
                    "error_difference_low": error["low"],
                    "error_difference_high": error["high"],
                }
            )
    adjusted = _holm_adjust([float(row["count_shift_p_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["count_shift_p_holm"] = p_holm
    return rows


def _causal_patching_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        detail = frames[model]["patching"].copy()
        detail["prediction_changed_numeric"] = detail["prediction_changed"].astype(
            float
        )
        detail["moved_numeric"] = detail["moved_toward_donor_gold"].astype(float)
        detail["direction_aligned_shift"] = pd.to_numeric(
            detail["generated_count_shift"]
        ) * np.sign(pd.to_numeric(detail["gold_count_offset"]))
        for layer in sorted(pd.to_numeric(detail["start_layer"]).astype(int).unique()):
            selected = detail[pd.to_numeric(detail["start_layer"]).astype(int) == layer]
            aligned = _one_sample_seed_estimate(
                selected,
                metric="direction_aligned_shift",
                label=f"causal-patching-aligned-{model}-L{layer}",
            )
            moved = _one_sample_seed_estimate(
                selected,
                metric="moved_numeric",
                label=f"causal-patching-moved-{model}-L{layer}",
            )
            insertion = selected[selected["direction"] == "needle_insertion"]
            removal = selected[selected["direction"] == "needle_removal"]
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "rows": int(len(selected)),
                    "changed_rate": float(
                        selected["prediction_changed_numeric"].mean()
                    ),
                    "moved_rate": moved["estimate"],
                    "moved_rate_low": moved["low"],
                    "moved_rate_high": moved["high"],
                    "insertion_shift": float(insertion["generated_count_shift"].mean()),
                    "removal_shift": float(removal["generated_count_shift"].mean()),
                    "aligned_shift": aligned["estimate"],
                    "aligned_shift_low": aligned["low"],
                    "aligned_shift_high": aligned["high"],
                    "aligned_shift_p_raw": aligned["p_raw"],
                }
            )
    adjusted = _holm_adjust([float(row["aligned_shift_p_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["aligned_shift_p_holm"] = p_holm
    return rows


def _causal_steering_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        detail = frames[model]["steering"].copy()
        detail["prediction_changed_numeric"] = detail["prediction_changed"].astype(
            float
        )
        detail["moved_numeric"] = detail["moved_toward_path_count"].astype(float)
        detail["target_hit_numeric"] = detail["nearest_path_count_hit"].astype(float)
        detail["direction_aligned_shift"] = pd.to_numeric(
            detail["generated_count_shift"]
        ) * np.sign(pd.to_numeric(detail["intended_count_shift"]))
        for layer in sorted(pd.to_numeric(detail["layer"]).astype(int).unique()):
            selected = detail[pd.to_numeric(detail["layer"]).astype(int) == layer]
            identity = [
                "design_variant",
                "seed",
                "receiver_stimulus_id",
                "target_stimulus_id",
                "layer",
                "steering_method",
                "alpha",
            ]
            moved = _paired_seed_contrast(
                selected,
                metric="moved_numeric",
                condition_column="condition",
                treatment="geometric",
                control="orthogonal_norm_matched_random",
                identity_columns=identity,
                label=f"causal-steering-moved-{model}-L{layer}",
            )
            aligned = _paired_seed_contrast(
                selected,
                metric="direction_aligned_shift",
                condition_column="condition",
                treatment="geometric",
                control="orthogonal_norm_matched_random",
                identity_columns=identity,
                label=f"causal-steering-aligned-{model}-L{layer}",
            )
            geometric = selected[selected["condition"] == "geometric"]
            control = selected[
                selected["condition"] == "orthogonal_norm_matched_random"
            ]
            baseline = selected.drop_duplicates("receiver_stimulus_id")
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "pairs_per_condition": int(len(geometric)),
                    "baseline_correct": int(
                        baseline["baseline_is_correct"].astype(bool).sum()
                    ),
                    "geometric_changed": float(
                        geometric["prediction_changed_numeric"].mean()
                    ),
                    "random_changed": float(
                        control["prediction_changed_numeric"].mean()
                    ),
                    "geometric_moved": float(geometric["moved_numeric"].mean()),
                    "random_moved": float(control["moved_numeric"].mean()),
                    "moved_difference": moved["estimate"],
                    "moved_difference_low": moved["low"],
                    "moved_difference_high": moved["high"],
                    "geometric_target_hit": float(
                        geometric["target_hit_numeric"].mean()
                    ),
                    "random_target_hit": float(control["target_hit_numeric"].mean()),
                    "geometric_aligned_shift": float(
                        geometric["direction_aligned_shift"].mean()
                    ),
                    "random_aligned_shift": float(
                        control["direction_aligned_shift"].mean()
                    ),
                    "aligned_difference": aligned["estimate"],
                    "aligned_difference_low": aligned["low"],
                    "aligned_difference_high": aligned["high"],
                    "aligned_difference_p_raw": aligned["p_raw"],
                }
            )
    adjusted = _holm_adjust([float(row["aligned_difference_p_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["aligned_difference_p_holm"] = p_holm
    return rows


def _causal_geometry_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        geometry = frames[model]["geometry"].copy()
        for layer, selected in geometry.groupby("layer", sort=True):
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "variants": int(selected["design_variant"].nunique()),
                    "projection_correlation_mean": float(
                        selected["endpoint_projection_count_correlation"].mean()
                    ),
                    "projection_correlation_min": float(
                        selected["endpoint_projection_count_correlation"].min()
                    ),
                    "monotone_fraction_min": float(
                        selected["endpoint_projection_monotone_fraction"].min()
                    ),
                    "step_cv_mean": float(selected["adjacent_step_cv"].mean()),
                    "successive_cosine_mean": float(
                        selected["mean_successive_step_cosine"].mean()
                    ),
                    "tortuosity_mean": float(selected["path_tortuosity"].mean()),
                }
            )
    return rows


def _table_answer_query_layer_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{int(row['layer'])}</td>"
            f"<td>{int(row['rows'])} / 10</td>"
            f"<td>{100*float(row['patched_valid_rate']):.2f}%</td>"
            f"<td>{int(row['eligible_donor_prediction_rows'])}</td>"
            f"<td>{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}] "
            f"(n={int(row['eligible_donor_prediction_rows'])}, "
            f"{int(row['eligible_donor_adoption_rate_seed_clusters'])} seeds)</td>"
            f"<td>{100*float(row['changed_rate']):.1f}%</td>"
            f"<td>{100*float(row['moved_toward_donor_gold_rate']):.1f}%</td>"
            f"<td>{100*float(row['follows_donor_prediction_rate']):.1f}%</td>"
            f"<td>{_number(row['mean_direction_aligned_shift'], signed=True)} "
            f"[{_number(row['mean_direction_aligned_shift_ci95_low'], signed=True)}, "
            f"{_number(row['mean_direction_aligned_shift_ci95_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['eligible_donor_adoption_vs_layer0_p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_answer_query_variant_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{int(row['layer'])}</td>"
            f"<td>{html.escape(str(row['design_variant']))}</td>"
            f"<td>{int(row['rows'])} / {int(row['seed_clusters'])}</td>"
            f"<td>{100*float(row['patched_valid_rate']):.1f}%</td>"
            f"<td>{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}] "
            f"(n={int(row['eligible_donor_prediction_rows'])}, "
            f"{int(row['eligible_donor_adoption_rate_seed_clusters'])} seeds)</td>"
            f"<td>{_number(row['mean_direction_aligned_shift'], signed=True)} "
            f"[{_number(row['mean_direction_aligned_shift_ci95_low'], signed=True)}, "
            f"{_number(row['mean_direction_aligned_shift_ci95_high'], signed=True)}]</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_answer_query_pair_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{int(row['layer'])}</td>"
            f"<td>{int(row['receiver_count'])}→{int(row['donor_count'])}</td>"
            f"<td>{int(row['rows'])} / {int(row['seed_clusters'])}</td>"
            f"<td>{100*float(row['patched_valid_rate']):.1f}%</td>"
            f"<td>{int(row['eligible_donor_prediction_rows'])} / "
            f"{int(row['eligible_donor_adoption_rate_seed_clusters'])} seeds</td>"
            f"<td>{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}]</td>"
            f"<td>{100*float(row['follows_donor_prediction_rate']):.1f}%</td>"
            f"<td>{_number(row['mean_direction_aligned_shift'], signed=True)} "
            f"[{_number(row['mean_direction_aligned_shift_ci95_low'], signed=True)}, "
            f"{_number(row['mean_direction_aligned_shift_ci95_high'], signed=True)}]</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_answer_query_audit_html(audit: dict[str, Any]) -> str:
    rendered: list[str] = []
    for model in MODELS:
        row = audit["models"][model]
        rendered.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{row['shards']} / {row['detail_rows']}</td>"
            f"<td>{row['successful_rows']} / {row['skipped_rows']}</td>"
            f"<td>{row['patched_valid_rows']} / {row['patched_invalid_rows']}</td>"
            f"<td>{row['eligible_donor_prediction_rows']}</td>"
            "<td>verified</td>"
            "</tr>"
        )
    return "".join(rendered)


def _answer_query_invalid_html(invalid: pd.DataFrame) -> str:
    if invalid.empty:
        return (
            '<div class="callout"><strong>严格格式审计。</strong> '
            "所有 patched continuation 都是 1–10 内的合法整数。</div>"
        )
    models = ", ".join(sorted(invalid["model"].astype(str).unique()))
    layers = ", ".join(
        f"L{int(value)}" for value in sorted(pd.to_numeric(invalid["start_layer"]).unique())
    )
    first = invalid.iloc[0]
    return (
        '<div class="callout"><strong>严格格式审计：'
        f"{len(invalid)} 个非法输出。</strong>它们全部来自 {html.escape(models)}、"
        f"{html.escape(str(first['design_variant']))}、seed {int(first['seed'])}、"
        f"receiver {int(first['receiver_count'])} ← donor {int(first['donor_count'])}，"
        f"位于 {layers}。receiver baseline 为 <code>"
        f"{html.escape(str(first['receiver_baseline_completion_text_raw']))}</code> "
        f"（token IDs <code>{html.escape(str(first['receiver_baseline_generated_token_ids']))}</code>）；"
        f"donor baseline 为 <code>{html.escape(str(first['donor_baseline_completion_text_raw']))}</code> "
        f"（IDs <code>{html.escape(str(first['donor_baseline_generated_token_ids']))}</code>）；"
        f"patch 后生成 <code>{html.escape(str(first['patched_completion_text_raw']))}</code> "
        f"（IDs <code>{html.escape(str(first['patched_generated_token_ids']))}</code>）。"
        "这支持“前缀被 transport，但随后发生未 patch 的自回归续写错误”：answer-query state "
        "决定首位数字，而下一生成步已超出 single-position patch。所有 eligible invalid rows "
        "均按 donor-adoption failure 计入主分析，没有被删除。</div>"
    )


def _table_causal_ablation_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>top-{row['top_n']}</td>"
            f"<td>{row['prompts']} ({row['baseline_correct']} correct)</td>"
            f"<td>{100*row['ranked_changed']:.1f}% / {100*row['random_changed']:.1f}%</td>"
            f"<td>{100*row['changed_difference']:+.1f} pp "
            f"[{100*row['changed_difference_low']:+.1f}, {100*row['changed_difference_high']:+.1f}]</td>"
            f"<td>{_number(row['ranked_count_shift'], signed=True)} / "
            f"{_number(row['random_count_shift'], signed=True)}</td>"
            f"<td>{_number(row['count_shift_difference'], signed=True)} "
            f"[{_number(row['count_shift_difference_low'], signed=True)}, "
            f"{_number(row['count_shift_difference_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['count_shift_p_holm'])}</td>"
            f"<td>{_number(row['error_difference'], signed=True)} "
            f"[{_number(row['error_difference_low'], signed=True)}, "
            f"{_number(row['error_difference_high'], signed=True)}]</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_causal_patching_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{row['layer']}</td>"
            f"<td>{row['rows']} / 10</td><td>{100*row['changed_rate']:.1f}%</td>"
            f"<td>{100*row['moved_rate']:.1f}% "
            f"[{100*row['moved_rate_low']:.1f}, {100*row['moved_rate_high']:.1f}]</td>"
            f"<td>{_number(row['insertion_shift'], signed=True)}</td>"
            f"<td>{_number(row['removal_shift'], signed=True)}</td>"
            f"<td>{_number(row['aligned_shift'], signed=True)} "
            f"[{_number(row['aligned_shift_low'], signed=True)}, "
            f"{_number(row['aligned_shift_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['aligned_shift_p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_causal_steering_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{row['layer']}</td>"
            f"<td>{row['pairs_per_condition']} / 10</td>"
            f"<td>{100*row['geometric_changed']:.1f}% / {100*row['random_changed']:.1f}%</td>"
            f"<td>{100*row['geometric_moved']:.1f}% / {100*row['random_moved']:.1f}%</td>"
            f"<td>{100*row['moved_difference']:+.1f} pp "
            f"[{100*row['moved_difference_low']:+.1f}, {100*row['moved_difference_high']:+.1f}]</td>"
            f"<td>{100*row['geometric_target_hit']:.1f}% / "
            f"{100*row['random_target_hit']:.1f}%</td>"
            f"<td>{_number(row['geometric_aligned_shift'], signed=True)} / "
            f"{_number(row['random_aligned_shift'], signed=True)}</td>"
            f"<td>{_number(row['aligned_difference'], signed=True)} "
            f"[{_number(row['aligned_difference_low'], signed=True)}, "
            f"{_number(row['aligned_difference_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['aligned_difference_p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_causal_geometry_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{row['layer']}</td>"
            f"<td>{row['variants']}</td>"
            f"<td>{_number(row['projection_correlation_mean'])} "
            f"(min {_number(row['projection_correlation_min'])})</td>"
            f"<td>{_number(row['monotone_fraction_min'])}</td>"
            f"<td>{_number(row['step_cv_mean'])}</td>"
            f"<td>{_number(row['successive_cosine_mean'])}</td>"
            f"<td>{_number(row['tortuosity_mean'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_causal_audit_html(audit: dict[str, Any]) -> str:
    rendered: list[str] = []
    for model in MODELS:
        stages = audit["models"][model]
        ablation = stages["ablation"]
        patching = stages["patching"]
        steering = stages["steering"]
        rendered.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{ablation['shards']} / {ablation['detail_rows']}</td>"
            f"<td>{patching['shards']} / {patching['detail_rows']} "
            f"({patching['skipped_rows']} skipped)</td>"
            f"<td>{steering['discovery']['npz_shards']} / "
            f"{steering['shards']} / {steering['detail_rows']}</td>"
            "<td>verified</td>"
            "</tr>"
        )
    return "".join(rendered)


def _forest_svg(
    rows: list[dict[str, Any]],
    *,
    estimate_key: str,
    low_key: str,
    high_key: str,
    title: str,
    axis_label: str,
    label: Any,
) -> str:
    width = 980
    left = 255
    right = 110
    top = 48
    row_height = 38
    height = top + row_height * len(rows) + 60
    lows = [float(row[low_key]) for row in rows]
    highs = [float(row[high_key]) for row in rows]
    minimum = min([0.0, *lows])
    maximum = max([0.0, *highs])
    span = max(maximum - minimum, 1e-6)
    minimum -= 0.08 * span
    maximum += 0.08 * span

    def x_position(value: float) -> float:
        return left + (float(value) - minimum) / (maximum - minimum) * (
            width - left - right
        )

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        '<g font-family="Aptos,Segoe UI,system-ui,sans-serif">',
        f'<text x="{left}" y="22" font-size="14" font-weight="700" fill="{AURORA["night_black"]}">'
        f"{html.escape(title)}</text>",
    ]
    ticks = np.linspace(minimum, maximum, 6)
    for tick in ticks:
        x = x_position(float(tick))
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" '
                f'y2="{height-42}" stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>',
                f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" '
                f'font-size="10" fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
            ]
        )
    zero_x = x_position(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top-14}" x2="{zero_x:.1f}" '
        f'y2="{height-40}" stroke="{AURORA["warm_brown"]}" stroke-width="1.6"/>'
    )
    colors = MODEL_COLORS
    for index, row in enumerate(rows):
        y = top + index * row_height
        estimate = float(row[estimate_key])
        low = float(row[low_key])
        high = float(row[high_key])
        color = colors.get(str(row.get("model")), AURORA["aurora_green"])
        parts.extend(
            [
                f'<text x="{left-12}" y="{y+4}" text-anchor="end" font-size="11" '
                f'fill="{AURORA["night_black"]}">{html.escape(str(label(row)))}</text>',
                f'<line x1="{x_position(low):.1f}" y1="{y}" '
                f'x2="{x_position(high):.1f}" y2="{y}" stroke="{color}" '
                'stroke-width="5" stroke-linecap="round" opacity=".38"/>',
                f'<circle cx="{x_position(estimate):.1f}" cy="{y}" r="6" '
                f'fill="{color}" stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>',
                f'<text x="{width-right+12}" y="{y+4}" font-size="10" fill="{AURORA["frost_gray"]}">'
                f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]</text>",
            ]
        )
    parts.extend(
        [
            f'<text x="{(left + width-right)/2:.1f}" y="{height-5}" text-anchor="middle" '
            f'font-size="11" fill="{AURORA["night_black"]}">{html.escape(axis_label)}</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _causal_conclusion_html(
    ablation_rows: list[dict[str, Any]],
    patching_rows: list[dict[str, Any]],
    steering_rows: list[dict[str, Any]],
    answer_query_layer_rows: list[dict[str, Any]],
) -> str:
    cards: list[str] = []
    for model in MODELS:
        ablation = next(
            row for row in ablation_rows if row["model"] == model and row["top_n"] == 8
        )
        cards.append(
            '<div class="note"><strong>'
            + html.escape(model)
            + " broad-head 必要性。</strong><p>Top-8 ranked ablation 改变 "
            + f"{100*ablation['ranked_changed']:.1f}% 的输出；"
            + f"layer-matched random heads 仅改变 {100*ablation['random_changed']:.1f}%。"
            + "配对 count-shift contrast 为 "
            + _number(ablation["count_shift_difference"], signed=True)
            + " ["
            + _number(ablation["count_shift_difference_low"], signed=True)
            + ", "
            + _number(ablation["count_shift_difference_high"], signed=True)
            + "]，Holm p="
            + _p_value(ablation["count_shift_p_holm"])
            + "。负值表示 ablate ranked heads 后 undercount 更强。</p></div>"
        )
    steering_sentences: list[str] = []
    for model in MODELS:
        selected = [row for row in steering_rows if row["model"] == model]
        final = max(selected, key=lambda row: int(row["layer"]))
        steering_sentences.append(
            f"{model} L{final['layer']}：aligned geometric-minus-random shift "
            f"{_number(final['aligned_difference'], signed=True)} "
            f"[{_number(final['aligned_difference_low'], signed=True)}, "
            f"{_number(final['aligned_difference_high'], signed=True)}], "
            f"Holm p={_p_value(final['aligned_difference_p_holm'])}"
        )
    max_moved = max(float(row["moved_rate"]) for row in patching_rows)
    cards.append(
        '<div class="note"><strong>Transport 与 manipulability 的分离。</strong><p>'
        + "；".join(steering_sentences)
        + f"。相反，exact needle-end residual patching 在所有测试层中最多只有 "
        f"{100*max_moved:.1f}% 的 rows 朝 donor gold 移动。这说明晚层 answer-query "
        "count geometry 可以被因果操纵，但被测试的单个 endpoint 不是充分 transport channel。</p></div>"
    )
    onset_sentences: list[str] = []
    final_sentences: list[str] = []
    for row in _answer_query_onset_rows(pd.DataFrame(answer_query_layer_rows)):
        onset_sentences.append(
            f"{row['model']} L{int(row['layer'])}："
            f"{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}]"
        )
    for row in _answer_query_final_rows(pd.DataFrame(answer_query_layer_rows)):
        final_sentences.append(
            f"{row['model']} L{int(row['layer'])}："
            f"{100*float(row['eligible_donor_adoption_rate']):.2f}%"
        )
    cards.append(
        '<div class="note"><strong>Exact query-state transport。</strong><p>'
        + "首次出现显著且 ≥50% 的 donor-prediction adoption 位于 "
        + "；".join(onset_sentences)
        + "。最终 block 的保守 eligible-row adoption rates 为 "
        + "；".join(final_sentences)
        + "。因此，晚层 answer-query state 是模型已计算 prediction 的高度充分载体，"
        "与 needle-end transplant 的近零结果形成鲜明对照。</p></div>"
    )
    return "".join(cards)


def _projection_2d_svg(projection: dict[str, Any]) -> str:
    """Aurora PC1/PC2 audit panels on one shared scale per model/pooling."""
    width, height = 960, 720
    panel_width, panel_height = 370, 245
    panel_positions = ((90, 70), (535, 70), (90, 385), (535, 385))
    rows = projection["rows"]
    x_values = np.asarray([float(row[7]) for row in rows], dtype=float)
    y_values = np.asarray([float(row[8]) for row in rows], dtype=float)
    x_low, x_high = np.quantile(x_values, [0.005, 0.995])
    y_low, y_high = np.quantile(y_values, [0.005, 0.995])
    x_margin = max(1e-6, float(x_high - x_low) * 0.08)
    y_margin = max(1e-6, float(y_high - y_low) * 0.08)
    x_low, x_high = float(x_low - x_margin), float(x_high + x_margin)
    y_low, y_high = float(y_low - y_margin), float(y_high + y_margin)

    def project(row: list[Any], left: float, top: float) -> tuple[float, float]:
        x = left + (float(row[7]) - x_low) / (x_high - x_low) * panel_width
        y = top + panel_height - (float(row[8]) - y_low) / (y_high - y_low) * panel_height
        return x, y

    parts = [
        f'<svg class="projection-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="PC1 and PC2 projections for {html.escape(str(projection["model"]))} {html.escape(str(projection["pooling"]))}">',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for panel_index, variant in enumerate(VARIANTS):
        left, top = panel_positions[panel_index]
        selected = [row for row in rows if row[0] == variant]
        parts.extend(
            [
                f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" '
                f'fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}" stroke-opacity=".42"/>',
                f'<text x="{left}" y="{top-16}" font-size="15" font-weight="700">{variant}</text>',
            ]
        )
        for fraction in (0.25, 0.5, 0.75):
            x = left + fraction * panel_width
            y = top + fraction * panel_height
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+panel_height}" '
                    f'stroke="{AURORA["frost_gray"]}" opacity=".16"/>',
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" '
                    f'stroke="{AURORA["frost_gray"]}" opacity=".16"/>',
                ]
            )
        for row in sorted(selected, key=lambda item: item[2] == "confirmation"):
            x, y = project(row, left, top)
            opacity = 0.50 if row[2] == "confirmation" else 0.16
            radius = 2.4 if row[2] == "confirmation" else 1.7
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{COUNT_COLORS[int(row[6])-1]}" opacity="{opacity}"/>'
            )
        for split, dash, opacity in (
            ("discovery", ' stroke-dasharray="6 5"', 0.60),
            ("confirmation", "", 0.95),
        ):
            split_rows = [row for row in selected if row[2] == split]
            centroid_rows: list[list[Any]] = []
            for count in range(1, 11):
                group = [row for row in split_rows if int(row[6]) == count]
                if not group:
                    continue
                centroid = group[0].copy()
                centroid[7] = float(np.mean([float(row[7]) for row in group]))
                centroid[8] = float(np.mean([float(row[8]) for row in group]))
                centroid_rows.append(centroid)
            points = [project(row, left, top) for row in centroid_rows]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{AURORA["night_black"]}" '
                f'stroke-width="{2.4 if split == "confirmation" else 1.5}" opacity="{opacity}"{dash}/>'
            )
            for count, (x, y) in enumerate(points, start=1):
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{COUNT_COLORS[count-1]}" '
                    f'stroke="{AURORA["night_black"]}" stroke-width=".8" opacity="{opacity}"/>'
                )
        parts.extend(
            [
                f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height+28}" text-anchor="middle" font-size="11">PC1 score</text>',
                f'<text transform="translate({left-43} {top+panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="11">PC2 score</text>',
                f'<text x="{left}" y="{top+panel_height+49}" font-size="10" fill="{AURORA["frost_gray"]}">shared axes: PC1 [{x_low:.2f}, {x_high:.2f}], PC2 [{y_low:.2f}, {y_high:.2f}]</text>',
            ]
        )
    legend_y = 688
    for count in range(1, 11):
        x = 135 + (count - 1) * 72
        parts.extend(
            [
                f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{COUNT_COLORS[count-1]}"/>',
                f'<text x="{x+10}" y="{legend_y+4}" font-size="10">{count}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="80" y="{legend_y+4}" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">index</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _static_figure_html(projections: dict[str, dict[str, Any]]) -> str:
    cards: list[str] = []
    for model in MODELS:
        for pooling in POOLINGS:
            projection = projections[f"{model}|{pooling}"]
            evr = projection["explained_variance_ratio"]
            cards.append(
                '<article class="figure-card">'
                f'<div class="figure-kicker">{html.escape(model)} · {html.escape(pooling.replace("_", "-"))} · L{int(projection["layer"])}</div>'
                f"{_projection_2d_svg(projection)}"
                '<p class="figure-caption"><strong>图：共享 PC1–PC2 轨迹。</strong>'
                "横轴和纵轴分别是以 v4.1 discovery occurrence states 拟合的 PC1、PC2 score；四个 panel 使用同一坐标范围。"
                "淡点是单个 seed×occurrence，节点和连线是 index 1→10 的 split centroid；虚线为 discovery，实线为 confirmation。"
                f"PC1/PC2 分别解释 {100*float(evr[0]):.1f}%/{100*float(evr[1]):.1f}% 的 v4.1 discovery variance。PCA 符号本身任意，应该比较顺序、间距和跨 seed 散布，而不是正负号。</p>"
                "</article>"
            )
    return "\n".join(cards)


REPORT_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Realistic NIAH V4：从表征到因果机制</title>
<style>
:root {
  --midnight:#23165C; --violet:#6750E8; --cyan:#00C2FF; --yellow:#F6E36A;
  --teal:#00D4B4; --green:#39E58C; --magenta:#C04DFF; --pink:#FF5FA2;
  --ink:#161923; --paper:#F8FBFF; --muted:#66758A; --frost:#8190A5;
  --brown:#765347; --line:rgba(129,144,165,.34); --surface:#FFFFFF;
  --soft-violet:rgba(103,80,232,.08); --soft-cyan:rgba(0,194,255,.09);
  --soft-yellow:rgba(246,227,106,.18); --soft-green:rgba(57,229,140,.10);
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--paper); font:15px/1.68 "Aptos","Segoe UI Variable Text","Segoe UI",system-ui,sans-serif; }
header { position:relative; overflow:hidden; padding:62px max(24px,calc((100vw - 1240px)/2)) 56px; color:var(--paper); background:var(--midnight); border-bottom:4px solid var(--cyan); }
header::after { content:""; position:absolute; inset:-45% -10% auto 40%; height:430px; pointer-events:none; opacity:.34; transform:rotate(-8deg); background:linear-gradient(115deg,transparent 8%,var(--violet) 35%,var(--magenta) 54%,var(--teal) 72%,transparent 88%); filter:blur(42px); }
header>* { position:relative; z-index:1; }
header .eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.15em; font-size:12px; font-weight:750; }
h1 { max-width:930px; margin:11px 0 15px; font:760 clamp(34px,5vw,61px)/1.04 "Aptos Display","Segoe UI Variable Display","Segoe UI",sans-serif; letter-spacing:-.035em; }
header p { max-width:920px; margin:0; color:rgba(248,251,255,.84); font-size:17px; }
.meta { display:flex; flex-wrap:wrap; gap:9px; margin-top:26px; }
.pill { border:1px solid rgba(0,194,255,.48); padding:6px 11px; color:var(--paper); font:12px/1.3 "Cascadia Mono","SFMono-Regular",Consolas,monospace; background:rgba(22,25,35,.24); }
nav { position:sticky; top:0; z-index:20; display:flex; gap:21px; overflow:auto; padding:11px max(24px,calc((100vw - 1240px)/2)); background:rgba(248,251,255,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }
nav a { color:var(--midnight); text-decoration:none; white-space:nowrap; font-weight:720; font-size:13px; }
nav a:hover { color:var(--violet); }
main { max-width:1240px; margin:auto; padding:38px 24px 88px; }
section { margin:0 0 64px; padding-top:4px; scroll-margin-top:62px; }
.section-kicker { display:block; margin-bottom:8px; color:var(--violet); font:740 11px/1.2 "Cascadia Mono",Consolas,monospace; letter-spacing:.13em; text-transform:uppercase; }
h2 { max-width:980px; margin:0 0 11px; font:760 clamp(27px,3.2vw,38px)/1.13 "Aptos Display","Segoe UI Variable Display","Segoe UI",sans-serif; letter-spacing:-.025em; }
h3 { margin:28px 0 9px; font-size:18px; line-height:1.3; }
h4 { margin:22px 0 7px; font-size:14px; color:var(--midnight); }
p { max-width:980px; }
.lede { max-width:980px; color:#39445A; font-size:16px; }
.callout { margin:20px 0; padding:16px 18px; border-left:4px solid var(--yellow); background:var(--soft-yellow); }
.callout strong { color:var(--midnight); }
.grid4 { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin:24px 0; border:1px solid var(--line); background:var(--line); }
.step { min-height:164px; padding:18px; background:var(--surface); border-top:4px solid var(--violet); }
.step:nth-child(2) { border-top-color:var(--cyan); } .step:nth-child(3) { border-top-color:var(--teal); } .step:nth-child(4) { border-top-color:var(--pink); }
.step strong { display:block; color:var(--midnight); font:780 23px/1.15 "Aptos Display","Segoe UI",sans-serif; }
.step small { color:var(--muted); font-family:"Cascadia Mono",Consolas,monospace; }
.table-wrap { overflow:auto; border:1px solid var(--line); background:var(--surface); }
table { width:100%; border-collapse:collapse; font-size:12.5px; font-variant-numeric:tabular-nums; }
caption { padding:10px 12px; text-align:left; color:var(--muted); }
th,td { padding:9px 10px; text-align:right; border-bottom:1px solid rgba(129,144,165,.19); white-space:nowrap; }
th { position:sticky; top:0; background:#EEF3FA; color:var(--midnight); font:720 10.5px/1.3 "Cascadia Mono",Consolas,monospace; letter-spacing:.035em; text-transform:uppercase; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
tbody tr:hover { background:var(--soft-cyan); }
tr:last-child td { border-bottom:0; }
code { color:var(--midnight); background:rgba(103,80,232,.09); padding:2px 5px; border:1px solid rgba(103,80,232,.15); font-family:"Cascadia Mono",Consolas,monospace; }
details { margin:14px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
summary { cursor:pointer; padding:11px 2px; color:var(--midnight); font-weight:720; }
summary:hover { color:var(--violet); }
.viz-shell { margin-top:22px; padding:18px; color:var(--paper); background:var(--midnight); border:1px solid var(--violet); box-shadow:0 22px 46px rgba(35,22,92,.14); }
.controls { display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)); gap:10px; margin-bottom:12px; }
label { display:flex; flex-direction:column; gap:4px; color:rgba(248,251,255,.72); font:720 10.5px/1.3 "Cascadia Mono",Consolas,monospace; letter-spacing:.04em; text-transform:uppercase; }
select,button { width:100%; border:1px solid rgba(0,194,255,.42); background:rgba(22,25,35,.54); color:var(--paper); padding:8px 9px; font:inherit; }
button { cursor:pointer; font-weight:720; transition:transform .16s ease,background .16s ease; }
button:hover { background:rgba(103,80,232,.42); } button:active { transform:translateY(1px); }
.canvas-wrap { position:relative; min-height:610px; background:#120D31; border:1px solid rgba(0,194,255,.32); }
#counter3d { display:block; width:100%; height:610px; cursor:grab; }
#counter3d.dragging { cursor:grabbing; }
#tooltip { position:absolute; display:none; pointer-events:none; max-width:280px; padding:9px 11px; border:1px solid var(--cyan); background:rgba(22,25,35,.95); color:var(--paper); font-size:12px; box-shadow:0 10px 24px rgba(22,25,35,.35); }
.viz-foot { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:12px; color:rgba(248,251,255,.72); font-size:12px; }
#geometry-stats { color:var(--yellow); text-align:right; }
.legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; color:rgba(248,251,255,.76); font-size:12px; }
.legend i { display:inline-block; width:10px; height:10px; margin-right:5px; }
.figures { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.figure-card { padding:13px; background:var(--surface); border:1px solid var(--line); }
.figure-kicker { margin:2px 4px 10px; color:var(--midnight); font:720 12px/1.3 "Cascadia Mono",Consolas,monospace; }
.projection-svg,.stat-svg { display:block; width:100%; height:auto; background:var(--paper); }
.figure-caption,.stat-figure figcaption { margin:10px 5px 3px; color:var(--muted); font-size:12px; line-height:1.55; }
.figure-caption strong,.stat-figure figcaption strong { color:var(--midnight); }
.stat-grid { display:grid; grid-template-columns:1fr; gap:16px; margin:19px 0; }
.stat-figure { margin:0; padding:14px; background:var(--surface); border:1px solid var(--line); }
.formula { margin:16px 0; padding:15px 17px; background:var(--soft-violet); border-left:3px solid var(--violet); font:15px/1.65 "Cascadia Mono",Consolas,monospace; overflow:auto; }
.method-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin:17px 0; border:1px solid var(--line); background:var(--line); }
.method-strip div { padding:13px; background:var(--surface); font-size:12px; }
.method-strip strong { display:block; margin-bottom:4px; color:var(--midnight); font-size:13px; }
.metric-defs { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:20px 0; }
.definition { padding:15px 17px; border-top:3px solid var(--cyan); background:var(--surface); }
.definition:nth-child(2n) { border-top-color:var(--teal); }
.definition strong { color:var(--midnight); }
.definition p { margin:5px 0 0; color:var(--muted); font-size:12.5px; }
.notes { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
.note { padding:17px; background:var(--surface); border-top:3px solid var(--cyan); }
.note:nth-child(2) { border-top-color:var(--teal); } .note:nth-child(3) { border-top-color:var(--pink); }
.note strong { color:var(--midnight); }
.section-conclusion { margin:24px 0 0; padding:17px 19px 18px; color:var(--paper); background:var(--midnight); border-left:5px solid var(--green); }
.section-conclusion span { display:block; margin-bottom:5px; color:var(--green); font:760 11px/1.2 "Cascadia Mono",Consolas,monospace; letter-spacing:.12em; text-transform:uppercase; }
.section-conclusion p { margin:0; color:rgba(248,251,255,.91); }
.mechanism-flow { display:grid; grid-template-columns:1fr 36px 1fr 36px 1fr 36px 1fr; align-items:stretch; margin:25px 0; }
.flow-node { padding:17px; background:var(--surface); border-top:4px solid var(--violet); }
.flow-node:nth-of-type(2) { border-color:var(--cyan); } .flow-node:nth-of-type(3) { border-color:var(--teal); } .flow-node:nth-of-type(4) { border-color:var(--pink); }
.flow-node b { display:block; color:var(--midnight); }
.flow-node small { color:var(--muted); }
.flow-arrow { display:grid; place-items:center; color:var(--violet); font-size:24px; }
.evidence-ledger { display:grid; grid-template-columns:1.2fr 1fr; gap:1px; margin:20px 0; border:1px solid var(--line); background:var(--line); }
.ledger-row { display:contents; }
.ledger-row>div { padding:13px 15px; background:var(--surface); }
.evidence-tag { display:inline-block; margin-right:8px; padding:2px 7px; color:var(--midnight); background:var(--soft-yellow); border:1px solid rgba(118,83,71,.25); font:700 10px/1.5 "Cascadia Mono",Consolas,monospace; text-transform:uppercase; }
.next-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.next-item { padding:18px 0; border-top:2px solid var(--violet); }
.next-item:nth-child(2n) { border-top-color:var(--teal); }
.next-item strong { display:block; color:var(--midnight); }
footer { padding:25px; color:var(--muted); text-align:center; border-top:1px solid var(--line); font-size:12px; }
@media (max-width:960px) { .grid4,.notes,.method-strip,.metric-defs,.next-grid { grid-template-columns:repeat(2,1fr); } .controls { grid-template-columns:repeat(3,1fr); } .figures { grid-template-columns:1fr; } .mechanism-flow { grid-template-columns:1fr; gap:0; } .flow-arrow { transform:rotate(90deg); min-height:34px; } }
@media (max-width:600px) { main { padding-inline:16px; } header { padding-inline:18px; } .grid4,.notes,.method-strip,.metric-defs,.next-grid,.viz-foot,.evidence-ledger { grid-template-columns:1fr; } .ledger-row { display:block; } .controls { grid-template-columns:repeat(2,1fr); } #counter3d { height:500px; } .canvas-wrap { min-height:500px; } }
@media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } button { transition:none; } }
</style>
</head>
<body>
<header>
  <div class="eyebrow">Realistic NIAH · Non-thinking · V4.1–V4.4</div>
  <h1>从可解码的 count signal，到生成真正使用的因果状态</h1>
  <p>本报告把完整 V4 结果组织成一条逐级收紧的证据链：先定位 prompt-reading representation，再检查 answer-query attention 如何聚合 needles，最后用 ablation、residual patching 与 geometric steering 区分“信息存在”“机制必要”“状态充分”和“方向可操纵”四种不同主张。</p>
  <div class="meta">
    <span class="pill">Qwen3-8B + Gemma4-E4B</span><span class="pill">10,000 canonical passage tokens</span><span class="pill">numeric counts 1–10</span><span class="pill">30 paired seeds / panel</span><span class="pill">commit @@COMMIT@@</span>
  </div>
</header>
<nav><a href="#overview">结论总览</a><a href="#design">实验设定</a><a href="#definitions">指标定义</a><a href="#behavior">Behavior</a><a href="#metrics">Representation</a><a href="#attention-heads">Attention</a><a href="#span-end-attention">错误诊断</a><a href="#causal">Causal effects</a><a href="#synthesis">机制综合</a><a href="#limits">缺口与下一步</a></nav>
<main>
<section id="overview">
  <span class="section-kicker">01 · Executive synthesis</span>
  <h2>当前最小机制：分布式 evidence aggregation，在后层写入可执行的 answer-query count state</h2>
  <p class="lede">最符合全部结果的工作模型不是“每个 needle 末尾独立保存一个可直接搬运的整数”，也不是“一个最强 head 均匀数完所有 needles”。更窄、也更可检验的模型是：多个 attention heads 在 <code>Total:</code> query 聚合 occurrence evidence；该聚合对最终 count magnitude 因果必要；随后在模型后段形成一个能够决定首个答案 token 的 query residual state。</p>
  <div class="mechanism-flow" aria-label="V4 mechanism summary">
    <div class="flow-node"><b>Needle-local states</b><small>span-end 中 count index 可解码，但单 endpoint state 跨 prompt patch 不足以搬运 count。</small></div><div class="flow-arrow">→</div>
    <div class="flow-node"><b>Distributed head bank</b><small>多个 broad / selector / local heads 并存；ranked top-8 bank ablation 比 layer-matched random 更强地导致 undercount。</small></div><div class="flow-arrow">→</div>
    <div class="flow-node"><b>Late answer-query state</b><small>Qwen L26、Gemma L31 开始出现强 donor-prediction transport；末层对合法 eligible rows 达到 100%。</small></div><div class="flow-arrow">→</div>
    <div class="flow-node"><b>Greedy numeric output</b><small>count-related geometry可被 steering，但 exact target hit 仍低；多 token 的“10”还需要后续自回归计算。</small></div>
  </div>
  <div class="evidence-ledger">
    <div class="ledger-row"><div><span class="evidence-tag">Descriptive</span>Needle-end count information persists</div><div>v4.4 confirmation R²：Qwen 0.866；Gemma 0.916。</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Correlational</span>Undercount aligns with low span-end attention</div><div>Omitted-tail overlap exceeds its combinatorial baseline; nested failures more often place the new needle in bottom-k.</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Necessary</span>Discovery-ranked head bank preserves count</div><div>Top-8 ranked-minus-random count shift：Qwen −0.331；Gemma −2.156。</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Sufficient</span>Late query state carries computed prediction</div><div>Final-layer valid eligible rows全部复制 donor prediction；不是 donor gold，也不保证多 token realization。</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Manipulable</span>Late query geometry moves output</div><div>Geometric-minus-random aligned shift：Qwen L26 +0.958；Gemma L31 +1.388。</div></div>
  </div>
  <div class="section-conclusion"><span>本节结论</span><p>目前可以主张“broad evidence aggregation → late executable query state”这条 bank-level 机制链；还不能主张存在唯一 scalar counter、单一 broad head、固定 partition circuit，或能够精确设定任意 target count 的线性控制方向。</p></div>
</section>
<section id="design">
  <span class="section-kicker">02 · Experimental design</span>
  <h2>实验设定：用四级 controlled relaxation 定位 seed sensitivity 的来源</h2>
  <p class="lede">两个模型都以 non-thinking mode 直接回答数字。每个 panel 包含 30 个 paired seeds × 10 个 gold counts；seed 1234–1253 仅用于 discovery、模型/层/head/方向选择，seed 1254–1263 仅用于 confirmation。所有 correctness、wrong/undercount 与 causal effect 标签都来自 <code>Total:</code> 后完整 deterministic greedy continuation；数字 10 按完整多-token sequence 解析，不使用 first-token probability。</p>
  <div class="grid4">
    <div class="step"><strong>V4.1</strong><small>all fixed</small><p>Needle position、city-score 顺序与具体内容跨 seed 固定，只改变 count 和 haystack。</p></div>
    <div class="step"><strong>V4.2</strong><small>release position</small><p>释放 needle position；city-score 顺序与内容仍固定。</p></div>
    <div class="step"><strong>V4.3</strong><small>release order</small><p>同时释放 position 与 city-score 顺序；内容仍固定。</p></div>
    <div class="step"><strong>V4.4</strong><small>release content</small><p>position、顺序、city-score 内容全部跨 seed 变化。</p></div>
  </div>
  <div class="method-strip">
    <div><strong>Stimulus</strong>10,000 canonical passage tokens；gold count N∈{1,…,10}；同一 family 采用 nested N−1→N construction，新增 occurrence 可精确定位。</div>
    <div><strong>Prompt-reading capture</strong>每个 active needle 保存两种 state：最后 token 的 <code>span_end</code>，以及整个 needle span 的 tokenwise mean <code>span_mean</code>。</div>
    <div><strong>Answer-query capture</strong>在 prompt-final <code>Total:</code> query 保存 hidden state 与每层每 head 对原 prompt key positions 的原始 attention row。</div>
    <div><strong>Models and output</strong>Qwen3-8B 与 Gemma4-E4B；greedy、numeric-only、最多 16 new tokens；所有 2,400 个 baseline answers 均 format-valid。</div>
  </div>
  <h3>完成性与原始数据审计</h3>
  <div class="table-wrap"><table><thead><tr><th>artifact / model</th><th>Qwen3-8B</th><th>Gemma4-E4B</th><th>用于什么</th></tr></thead><tbody>
    <tr><td>behavior rows</td><td>1,200</td><td>1,200</td><td>完整 greedy output label</td></tr>
    <tr><td>representation capture shards</td><td>120</td><td>120</td><td>span-end / span-mean hidden states</td></tr>
    <tr><td>raw answer-query attention tensors</td><td>1,200</td><td>1,200</td><td>head ranking、omission、partitioning</td></tr>
    <tr><td>raw attention bytes</td><td>28.36 GB</td><td>1.78 GB</td><td>保留可复算的 query rows</td></tr>
    <tr><td>causal detail rows / model</td><td>5,360</td><td>5,360</td><td>640 ablation + 720 endpoint patch + 2,560 query patch + 1,440 steering</td></tr>
  </tbody></table></div>
  <div class="callout"><strong>解释规则。</strong>v4.1 的干净曲线仍可能只是固定 identity、位置或记录顺序编码。只有 signal 在 v4.3/v4.4 仍可跨 seed 解码，才支持相对 content-independent 的 count representation；即便如此，representation 仍不等于机制。</div>
  <div class="section-conclusion"><span>本节结论</span><p>V4.1→V4.4 是逐项释放 nuisance factors 的 paired ladder，而不是四个无关数据集。后续所有 discovery selection 与 confirmation inference 严格分离；实验数据、raw attention、hidden states 和 causal detail rows 均完整，因此报告中的差异可以解释为控制项释放与干预效应，而不是样本缺失。</p></div>
</section>

<section id="definitions">
  <span class="section-kicker">03 · Definitions and estimands</span>
  <h2>新概念与计算方法：先定义“看见了什么”，再讨论“是否因果”</h2>
  <div class="metric-defs">
    <div class="definition"><strong>Ridge count probe 与 held-out R²</strong><p>对 occurrence state <em>h</em> 拟合 <em>ŷ=wᵀz(h)+b</em>。α 由 discovery seeds 的 5-fold GroupKFold 选择；primary layer 仅由 v4.1 discovery CV-R² 选择。每个 panel 的 probe weights 在该 panel discovery 上拟合，再在同 panel confirmation 上评估。R²=1−Σ(y−ŷ)²/Σ(y−ȳ)²；MAE=mean|y−ŷ|。</p></div>
    <div class="definition"><strong>Noise / signal ratio</strong><p>discovery count centroid 为 μ₁,…,μ₁₀，grand mean 为 μ̄。signal RMS=[meanᵢ||μᵢ−μ̄||²]¹ᐟ²；confirmation noise RMS=[meanⱼ||hⱼ−μ<sub>yⱼ</sub>||²]¹ᐟ²；ratio=noise RMS/signal RMS，越小表示跨 seed 散点相对 count separation 越紧。</p></div>
    <div class="definition"><strong>Geometry stability</strong><p>Linear CKA 比较 discovery 与 confirmation 的 10×10 centered centroid Gram geometry；distance correlation 是两套 centroid pairwise Euclidean distances 的 Pearson correlation。二者越接近 1，跨 split 相对几何越稳定，但不要求坐标轴方向相同。</p></div>
    <div class="definition"><strong>Attention mass、coverage 与 N<sub>eff</sub></strong><p>对第 i 个 needle span 的 attention mass 记作 mᵢ，pᵢ=mᵢ/Σm。entropy coverage=exp[−Σpᵢlog pᵢ]/N；effective number N<sub>eff</sub>=N×coverage。Primary score=(Σmᵢ)×coverage，因此同时要求总 mass 高且在 needles 间分布广。</p></div>
    <div class="definition"><strong>Head-bank effective number</strong><p>对 occurrence profile q 计算 inverse-participation N<sub>eff</sub>=1/Σqᵢ²。Equal-head 版本先把每个 head profile 归一化后等权相加；raw-attention 版本保留真实 attention magnitude。前者高、后者低表示“覆盖潜力存在，但被少数高-mass selector 压过”。</p></div>
    <div class="definition"><strong>Causal effect labels</strong><p><em>Changed</em>：patch/ablation 后 parsed count 与 receiver baseline 不同。<em>Moved</em>：到 donor gold/target 的距离严格变小。<em>Aligned shift</em>=(patched−receiver)×sign(target−receiver)。Query patch 的 primary outcome 是在 receiver/donor baseline predictions 不同的 eligible rows 中，patched prediction 是否等于 donor prediction。</p></div>
  </div>
  <div class="method-strip">
    <div><strong>Uncertainty</strong>先在每个 confirmation seed 内平均 panels/pairs，再对 10 个完整 seeds 做 20,000 次 percentile bootstrap。</div>
    <div><strong>Testing</strong>paired seed contrast 使用 two-sided exact sign-flip test；每个 intervention family 内用 Holm correction。</div>
    <div><strong>Evidence hierarchy</strong>PCA/probe=descriptive availability；attention-error alignment=correlation；ablation=necessity；patching=sufficiency；steering=manipulability。</div>
    <div><strong>Index convention</strong>layer/head 均为 zero-based；<code>L29H3</code> 表示 layer 29、head 3。</div>
  </div>
  <div class="section-conclusion"><span>本节结论</span><p>报告不会把高 R²、漂亮 PCA 或高 attention mass 直接称为“模型真的在用的 counter”。只有 matched-control ablation、跨 prompt patching 和 held-out steering 能把描述性表示推进到不同层级的因果结论。</p></div>
</section>

<section id="behavior">
  <span class="section-kicker">04 · Behavioral boundary</span>
  <h2>Behavior：主要失败边界随 count 增大出现，而不是由某一个 V4 panel 单独触发</h2>
  <p class="lede">图中横轴是真实 needle count N，纵轴是完整 greedy numeric sequence 的 exact-match accuracy；每条线对应一个 V4 panel，每个点包含 10 个 confirmation seeds。黄色背景从 N=5 开始，仅用于帮助观察中高 count 区间，不参与统计。</p>
  <div class="stat-grid"><figure class="stat-figure">@@BEHAVIOR_ACCURACY_SVG@@<figcaption><strong>图 1 · Confirmation accuracy by count。</strong>横轴：gold count 1–10；纵轴：parsed full-sequence exact-match accuracy（0–1）。颜色：V4.1–V4.4 controlled panels。两模型在小 count 上接近饱和，在 count 4–6 附近进入快速下降区；9–10 几乎全部 undercount。</figcaption></figure></div>
  <h3>Panel-level confirmation summary</h3>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>correct / 100</th><th>accuracy</th><th>mean prediction</th><th>MAE</th><th>undercount</th><th>format valid</th></tr></thead><tbody>@@BEHAVIOR_PANEL_ROWS@@</tbody></table></div>
  <details><summary>展开：跨 panel pooling 后每个 count 的完整数值</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>gold N</th><th>correct / 40</th><th>accuracy</th><th>mean prediction</th><th>undercount</th></tr></thead><tbody>@@BEHAVIOR_COUNT_ROWS@@</tbody></table></div></details>
  @@BEHAVIOR_CONCLUSION@@
</section>

<section id="metrics">
  <span class="section-kicker">05 · Representation I</span>
  <h2>Prompt-reading representation：span-end signal 跨 v4.4 仍可解码，span-mean 则依赖固定记录结构</h2>
  <p class="lede">每个 model×pooling 只用 v4.1 discovery 的 grouped-seed CV-R² 选择 primary layer：Qwen span-end L1、span-mean L0；Gemma span-end L22、span-mean L0。选层之后，每个 V4 panel 的 ridge weights 与 α 只在该 panel discovery seeds 上拟合，再在同 panel confirmation seeds 上评估；confirmation 从不参与选层、调 α 或拟合。</p>
  <div class="stat-grid"><figure class="stat-figure">@@REPRESENTATION_R2_SVG@@<figcaption><strong>图 2 · Held-out count decoding。</strong>横轴：V4 controlled relaxation；纵轴：confirmation R²。实线/圆点是 span-end，虚线/圆点是 span-mean。R²=0 水平线表示不优于用 confirmation mean 预测。span-end 随控制释放逐渐变差但仍保持强正值；span-mean 在 v4.3 释放 city-score order 后明显崩溃。</figcaption></figure></div>
  <details open><summary>Primary-layer confirmation metrics</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>layer</th><th>panel</th><th>confirm R²</th><th>confirm MAE</th><th>noise / signal</th><th>linear CKA</th><th>distance corr.</th></tr></thead><tbody>@@METRIC_ROWS@@</tbody></table></div></details>
  <details><summary>Paired confirmation-seed sensitivity：相邻 relaxation 在哪里首次显著变差</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>layer</th><th>metric</th><th>step</th><th>Δ mean</th><th>95% seed CI</th><th>CI &gt; 0</th></tr></thead><tbody>@@SENSITIVITY_ROWS@@</tbody></table></div></details>
  @@REPRESENTATION_CONCLUSION@@
</section>

<section id="counter">
  <span class="section-kicker">06 · Representation II</span>
  <h2>Count manifold：centroid 顺序存在，但散点与相邻步长并不支持“干净等距 scalar counter”</h2>
  <p class="lede">下面的 3D view 在每个 model×pooling 的 primary layer 上，用 v4.1 discovery 的全部 occurrence states 拟合 PC1–PC6，并把同一 basis 应用于 v4.1–v4.4。拖动旋转、滚轮缩放，可自由切换三条 PCA axes。淡点是单 seed×occurrence state，彩色节点/连线是 split-specific index 1→10 centroids；颜色编码 occurrence index，不编码 correctness。</p>
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
  <div class="formula">step CV = std(||μᵢ₊₁−μᵢ||) / mean(||μᵢ₊₁−μᵢ||)；path/chord = Σ||μᵢ₊₁−μᵢ|| / ||μ₁₀−μ₁||。前者衡量相邻 count 步长是否等距，后者衡量 centroid path 是否接近一条直线；理想等距直线计数轴应同时接近 0 与 1。</div>
  <div class="callout"><strong>坐标可比性。</strong>同一 model×pooling 内四个 panel 共享 PCA basis；不同模型或不同 pooling 分别拟合，因此 PC 坐标绝对值不可跨 panel group 直接比较。PCA component 的正负号没有语义。</div>
  <h3>Aurora PC1–PC2 audit panels</h3>
  <p>以下四张图与 3D view 使用相同隐藏状态与 v4.1 discovery basis，但固定展示 PC1–PC2，便于比较跨 seed 散点宽度。它们替代旧配色 PNG 作为主报告图；原始 CSV/PNG 仍保留在 run artifact 中。</p>
  <div class="figures">@@STATIC_FIGURES@@</div>
  <details><summary>N=10 trajectory 的实际 greedy outcome strata</summary><p class="lede">一条 N=10 trajectory 的十个 occurrence vectors 共同继承该 prompt 的最终输出标签；不是按单 occurrence 重新分类。Qwen confirmation 在四个 panel 都没有正确 N=10 trajectory；Gemma 只有 v4.1 的 1 条，因此 correct/wrong 几何只能作 audit，不能作有 power 的组间比较。</p><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>split</th><th>correct / n</th><th>accuracy</th><th>mean prediction</th><th>MAE</th></tr></thead><tbody>@@BEHAVIOR_ROWS@@</tbody></table></div></details>
  <div class="section-conclusion"><span>本节结论</span><p>可切换 PCA 中的 centroid trajectory 证明 count-related geometry 具有低维可视结构，但 individual seed scatter、step CV 与 path/chord 显示它既不完全等距，也不总是笔直。PCA 只能说明 representation 的组织方式；是否进入生成读出，需要后面的 attention 与 causal intervention。</p></div>
</section>

<section id="attention-heads">
  <span class="section-kicker">07 · Descriptive attention mechanism</span>
  <h2>Answer-query attention：最高排名 head 不等于完整机制，broad aggregation 分散在一个多-head bank 中</h2>
  <p class="lede">所有 head ranking 只使用 discovery prompts。对每个 head，<code>span_end</code> 把每个 needle 的最后 token 当作 evidence location；<code>span_mean</code> 汇总完整 needle span。Rank primary score 同时奖励总 needle mass 与 entropy coverage。图中纵轴 N<sub>eff</sub> 是 rank-1 head 在 N=10 时等效覆盖的 needles 数量；10 表示均匀覆盖，1 表示几乎只选择一个 occurrence。</p>
  <div class="stat-grid"><figure class="stat-figure">@@ATTENTION_BREADTH_SVG@@<figcaption><strong>图 3 · Rank-1 head breadth。</strong>横轴：V4 panel；纵轴：rank-1 discovery head 的 N<sub>eff</sub>（0–10）。每个 bar 上的数值是 effective number。Qwen span-end 的最高排名 head 只覆盖约一个 endpoint；Qwen span-mean 与 Gemma 两种 pooling 更接近 5 个 occurrences 的 broad distribution。</figcaption></figure></div>
  <details><summary>展开：每个 model×panel×pooling 的 rank-1 head 与指标</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>pooling</th><th>rank-1 head</th><th>total mass</th><th>coverage</th><th>N_eff</th><th>primary</th></tr></thead><tbody>@@ATTENTION_TOP_ROWS@@</tbody></table></div></details>

  <h3>Qwen span-end 全候选 phenotype：用户提出的“分区内 aggregation”假设</h3>
  <p>因为 Qwen L29H3 的 rank 很高但 N<sub>eff</sub>≈1，我们没有只看 top-1/top-8，而是对每个 panel 的全部 discovery-eligible candidates（212/226/226/225 heads）计算 endpoint breadth、winner occurrence、query-depth quartile gating 和 full-span breadth，并用以下 post-hoc descriptive rules 分类：</p>
  <div class="metric-defs">
    <div class="definition"><strong>Global endpoint aggregator</strong><p>endpoint N<sub>eff</sub>≥6，且任何单 occurrence 的 mean normalized share≤0.25。</p></div>
    <div class="definition"><strong>Partition-local endpoint aggregator</strong><p>不是 global；winning depth quartile 内至少含 2 个 needles；local effective fraction≥0.8；整个 query row 至少 50% mass 落在同一 depth quartile。</p></div>
    <div class="definition"><strong>Occurrence endpoint selector</strong><p>endpoint N<sub>eff</sub>≤2，且至少 80% examples 选择相同 occurrence。</p></div>
    <div class="definition"><strong>证据边界</strong><p>这些 phenotype 是行为描述，不是模块标签。一个 head 可有 broad span-mean profile，却在 endpoint 上是 selector；attention profile 也没有包含 value vector 与 output projection。</p></div>
  </div>
  <div class="stat-grid"><figure class="stat-figure">@@PARTITION_PHENOTYPE_SVG@@<figcaption><strong>图 4 · Qwen discovery-eligible span-end head taxonomy。</strong>横轴：V4 panel；纵轴：候选 head 数量。堆叠颜色显示 global aggregator、partition-local aggregator、occurrence selector 与其他 phenotypes。v4.1 的 partition-local 数量较多，但释放 position 后明显减少；global aggregator bank 在四个 panel 都存在。</figcaption></figure></div>
  <details open><summary>Phenotype bank coverage：等权覆盖潜力与 raw attention 实际权重</summary><div class="table-wrap"><table><thead><tr><th>panel</th><th>phenotype</th><th>heads</th><th>equal-head N_eff</th><th>raw-mass N_eff</th><th>mean summed endpoint mass</th></tr></thead><tbody>@@PARTITION_BANK_ROWS@@</tbody></table></div></details>
  <div class="callout"><strong>跨 panel×split 稳定的 Qwen global aggregators（13 个）：</strong><code>@@STABLE_GLOBAL_HEADS@@</code>。其中 L6H12 的 endpoint mass 较高；L13H16、L17H22 也是下一轮 bank-specific ablation 的优先候选。L29H3 则在四个 panel 都把约 99% endpoint share 给 occurrence 1；position 变化后它跟随“最早 occurrence”，而不是固定绝对 depth bin。</div>
  @@ATTENTION_CONCLUSION@@
</section>

<section id="span-end-attention">
  <span class="section-kicker">08 · Error-linked attention</span>
  <h2>Undercount 诊断：错误输出是否漏注意了行为上“少算”的同一批 needles？</h2>
  <p class="lede">主分析按用户指定只报告 <code>span_end</code>。它检验 answer-query top-8 discovery-ranked ensemble 是否给 undercount 所隐含的 omitted evidence 更低 attention。所有估计只使用 confirmation seeds；每个 head 先归一化到 occurrence mean share=1 后再等权平均，避免高-mass selector 完全淹没 broad heads。</p>
  <div class="method-strip">
    <div><strong>Behavior label</strong>完整 greedy integer output <em>N̂</em>；sequence probability 与 candidate score 均不参与。无法解析或非-undercount rows 不属于该 estimand。</div>
    <div><strong>Held-out unit</strong>10 个 confirmation seeds（1254–1263）；同一 seed 内全部 prompts 保留在同一 resampling cluster。</div>
    <div><strong>Uncertainty</strong>对 seed-level means 做 20,000 次 percentile bootstrap，报告 95% interval。</div>
    <div><strong>Testing</strong>two-sided exact seed sign-flip；两个 pooled model tests 为一个 Holm family，八个 panel-level tests 为另一个。</div>
  </div>

  <h3>8.1 Behavior-implied omitted tail 与 lowest-attention occurrences</h3>
  <p>对 gold count <em>N</em> 和 undercount <em>N̂</em>，令 <em>k=N−N̂</em>。行为上“少算”的尾部集合为 <em>T<sub>k</sub>={N−k+1,…,N}</em>；attention-implied 集合 <em>B<sub>k</sub></em> 是 ensemble attention 最低的 k 个 occurrence endpoints。主分数是两个集合的 overlap fraction。</p>
  <div class="formula"><em>S=|B<sub>k</sub>∩T<sub>k</sub>|/k</em>。若 <em>B<sub>k</sub></em> 是均匀随机的 k-subset，则 <em>E[S]=k/N</em>，而 exact set match 的 chance 为 <em>P(B<sub>k</sub>=T<sub>k</sub>)=1/C(N,k)</em>。</div>
  <p class="lede">Primary estimand 是 seed-equal mean 的 <em>S−k/N</em>。Cross-panel aggregate 先在每个 seed 内给四个 panel 等权，再跨 seed 推断，避免选择最有利的 relaxation。<em>Tail/prefix</em> 是 omitted tail 的 mean normalized attention 除以 retained prefix；小于 1 表示 tail evidence 被相对抑制。该 omission 分析为 post-hoc inferential audit，并非 preregistered confirmatory test。</p>
  <h3>Cross-panel aggregate</h3>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>seeds</th><th>overlap / chance</th><th>Δ [95% seed CI]</th><th>Holm p</th><th>exact / chance</th><th>exact Δ [95% CI]</th><th>tail / prefix</th></tr></thead><tbody>@@SPAN_END_POOLED_ROWS@@</tbody></table></div>
  <details><summary>Panel-level heterogeneity</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>prompts / seeds</th><th>mean k</th><th>overlap</th><th>chance k/N</th><th>Δ [95% seed CI]</th><th>Holm p</th><th>exact / chance</th><th>exact Δ [95% CI]</th><th>tail / prefix</th></tr></thead><tbody>@@SPAN_END_ALIGNMENT_ROWS@@</tbody></table></div></details>
  <div class="stat-grid">
    <figure class="stat-figure">@@SPAN_END_ALIGNMENT_SVG@@<figcaption><strong>图 5 · Omitted-tail overlap。</strong>横轴：attention bottom-k 与 behavior-implied omitted tail 的 overlap fraction。圆点：seed-equal observed overlap；菱形：同 k/N 下的 hypergeometric chance；半透明线：observed−chance 的 95% seed-cluster CI 平移回 overlap axis。圆点在菱形右侧表示低 attention occurrences 更像模型行为上少算的尾部。</figcaption></figure>
  </div>

  <h3>8.2 Nested N−1→N 中精确新增的 needle</h3>
  <p>Tail 分析把“少算”解释为遗漏后 k 个 occurrences，仍依赖顺序假设。Nested construction 提供更强的 paired check：从 N−1 到 N 新增的 occurrence 精确已知。我们只比较两组最终都仍 undercount 的 transitions：(i) 新 endpoint 是否落在当前 bottom-k attention set；(ii) 新 endpoint 的 normalized share，其中 1 表示在所有 occurrences 间均匀。<em>Failed</em> 表示 output 未增加；<em>registered</em> 表示恰好 +1。先在 seed×panel 内配对，再在 seed 内平均 panels。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>failed / registered n</th><th>paired blocks / seeds</th><th>bottom-k failed / reg.</th><th>risk Δ F−R [95% CI]</th><th>Holm p</th><th>share failed / reg.</th><th>share Δ R−F [95% CI]</th><th>Holm p</th></tr></thead><tbody>@@SPAN_END_NESTED_ROWS@@</tbody></table></div>
  <div class="stat-grid">
    <figure class="stat-figure">@@SPAN_END_NESTED_SVG@@<figcaption><strong>图 6 · Newly added needle in bottom-k。</strong>横轴：新增 endpoint 进入当前 bottom-k attention set 的概率；纵向分组：模型；粉色/绿色分别为 failed-to-increment 与 registered +1。横线是完整 seed bootstrap 95% CI；右侧 RD=failed−registered 的 paired risk difference。两组最后都 undercount，因而差异不是简单的 correct/wrong 对比。</figcaption></figure>
  </div>
  <div class="notes">@@SPAN_END_CONCLUSION@@</div>
  <div class="callout"><strong>推断边界。</strong>Tail set 是由 behavior 推断出的“可能遗漏集合”，不是模型内部忘记项的直接记录；nested check 虽然精确知道新增 occurrence，但 attention 与 output 仍是相关。它们为 causal ablation/patching 提供靶点，不能替代干预。</div>
  <div class="section-conclusion"><span>本节结论</span><p>在 confirmation undercounts 中，span-end ensemble 的最低-attention occurrences 与行为上少算的尾部显著对齐；当 nested pair 未注册新增 occurrence 时，该新 endpoint 更常进入 bottom-k、normalized share 也更低。最合理的描述是“evidence omission 与 undercount 同步出现”，尚不能单凭 attention map 宣称 omission 导致错误。</p></div>
</section>

<section id="causal">
  <span class="section-kicker">09 · Causal interventions</span>
  <h2>从 representation 到 causal effect：分别检验 head necessity、state transport 与 geometry manipulability</h2>
  <p class="lede">完成的 <code>screen_8h_v1</code> 与 <code>answer_query_dense_v1</code> 保留两个模型、四个 V4 panels 和全部 10 个 confirmation seeds，只缩小 intervention grid。每一行都在 intervention 后执行完整 deterministic greedy generation；correctness、changed、moved 与 aligned shift 都来自最终 parsed continuation，不使用 candidate probability。</p>
  <div class="method-strip">
    <div><strong>Head necessity</strong>仅在 answer-query row ablate discovery-ranked span-end top-4/top-8 heads；counts 7–10；control 是相同 layers、相同数量的 random heads。</div>
    <div><strong>Endpoint transport</strong>5↔6、7↔8、9↔10 nested pairs；把精确 toggled needle-end residual 从 donor 复制到 receiver；从三个 matched depths 起 cumulative patch。</div>
    <div><strong>Exact query transport</strong>5↔6、7↔8、9↔10、5↔10；只复制 prompt-final <code>Total:</code> query 的单个 residual vector；8 个 single-layer sites。</div>
    <div><strong>Query manipulability</strong>7↔8、9↔10、5↔10；用 discovery count centroids 的 α=1 delta steering query residual；control 为同 norm 的 orthogonal random direction。</div>
    <div><strong>Inference unit</strong>point estimate 等权 prompts；seed 内先平均 panels/pairs，再 bootstrap 10 个 paired seeds；primary tests 为 exact sign-flip + family-wise Holm。</div>
  </div>
  <div class="notes">@@CAUSAL_CONCLUSION@@</div>

  <h3>9.1 Discovery-ranked head-bank ablation：被选中的 aggregation bank 是否因果必要？</h3>
  <p>Primary contrast 是 ranked minus layer-matched random。生成 count shift 定义为 ablated prediction−baseline prediction，因此负 contrast 表示删掉 ranked bank 比删掉同层 random heads 造成更强 undercount。由于高 count baseline-correct prompts 极少，主表按 preregistered screen estimand 合并 correct/wrong；原始 summary/control tables 仍保留 outcome strata。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>set</th><th>prompts (correct)</th><th>changed ranked / random</th><th>Δ changed [95% CI]</th><th>count shift ranked / random</th><th>Δ count shift [95% CI]</th><th>Holm p</th><th>Δ MAE [95% CI]</th></tr></thead><tbody>@@CAUSAL_ABLATION_ROWS@@</tbody></table></div>
  <div class="stat-grid"><figure class="stat-figure">@@CAUSAL_ABLATION_SVG@@<figcaption><strong>图 7 · Ranked head-bank ablation。</strong>横轴：paired mean count shift 的 ranked−random contrast；纵轴：model 与 top-k bank。圆点是 seed-equal estimate，横线是 95% seed-bootstrap CI，棕色竖线为零效应。负值表示 ranked bank 被删后额外 undercount；同层 random control 排除了“只是删了若干 heads”的解释。</figcaption></figure></div>
  <div class="section-conclusion"><span>当前结论 · Necessity</span><p>Top-8 bank 在两个模型都产生稳定负 count-shift contrast（Qwen −0.331；Gemma −2.156，Holm p=0.0078），因此所选 span-end head bank 对保持 output magnitude 因果必要。该结论是 bank-level，不证明 bank 中每个 head 单独 broad、不能区分 aggregator 与 selector 的个体贡献，也不等价于精确算术求和。</p></div>

  <h3>9.2 Exact needle-end residual patching：可解码 endpoint 是否是可运输的 count carrier？</h3>
  <p>该实验按用户指定只 patch <code>span_end</code>，不是 span mean。Insertion 把 N−1 receiver 的新增 donor endpoint state 贴入，removal 做反向；positive aligned shift 要求 insertion 增大 output、removal 减小 output。<em>Moved</em> 要求到 donor gold 的距离严格缩短，因此 receiver baseline 已等于 donor gold 时不会被误计为 transport success。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>start</th><th>rows / seeds</th><th>changed</th><th>moved [95% CI]</th><th>insertion shift</th><th>removal shift</th><th>aligned shift [95% CI]</th><th>Holm p</th></tr></thead><tbody>@@CAUSAL_PATCHING_ROWS@@</tbody></table></div>
  <div class="stat-grid"><figure class="stat-figure">@@CAUSAL_PATCHING_SVG@@<figcaption><strong>图 8 · Exact endpoint transport。</strong>横轴：mean direction-aligned generated-count shift；纵轴：model 与 cumulative start layer。圆点/横线为 seed estimate/95% CI；零表示复制该单 endpoint state 没有沿 donor count 方向移动输出。这里检验的是明确 directional null，因此没有把 random patch 作为 primary contrast。</figcaption></figure></div>
  <div class="section-conclusion"><span>当前结论 · Endpoint insufficiency</span><p>所有 tested depths 的 aligned-shift CI 都包含 0，严格 moved rate 最高仅 2.1%。因此单个 toggled needle-end state 虽然高度 count-decodable，却不是跨 prompt 可直接运输的充分 count state。仍可能需要 position-specific routing、完整 needle token sequence、多个 endpoints 的协调状态，或由 query-side attention 重新聚合。</p></div>

  <h3>9.3 Exact answer-query residual patching：聚合后的 query state 是否足以搬运模型已经算出的 prediction？</h3>
  <p>每一行在某一层把 donor prompt-final <code>Total:</code> query 的一个 residual vector 精确复制到 receiver，然后执行完整 greedy generation。Primary estimand 是 receiver 与 donor baseline predictions 不同时，patched output 是否采用 <em>donor model prediction</em>。这与 donor-gold accuracy 有意分离：一个完美 transport 可以忠实复制 donor 已经算错的数字。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>rows / seeds</th><th>valid</th><th>eligible n</th><th>adopts donor prediction [95% CI]</th><th>changed (valid)</th><th>moved to donor gold (valid)</th><th>matches donor prediction (all valid)</th><th>aligned shift (valid) [95% CI]</th><th>adoption vs L0 Holm p</th></tr></thead><tbody>@@ANSWER_QUERY_LAYER_ROWS@@</tbody></table></div>
  <div class="stat-grid"><figure class="stat-figure">@@ANSWER_QUERY_ADOPTION_SVG@@<figcaption><strong>图 9 · Layerwise donor-prediction transport。</strong>横轴：receiver/donor baseline predictions 不同的 eligible rows 中，patched output 等于 donor prediction 的比例；纵轴：model/layer。严格 invalid continuations 留在分母并记作 failure。横线为 10 个完整 confirmation seeds 的 95% bootstrap CI；Holm p 比较每个 later layer 与同模型 L0。</figcaption></figure></div>
  <h4>Final-layer robustness by V4 panel</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>panel</th><th>rows / seeds</th><th>valid</th><th>eligible adoption [95% CI]</th><th>aligned shift [95% CI]</th></tr></thead><tbody>@@ANSWER_QUERY_VARIANT_ROWS@@</tbody></table></div>
  <h4>Final-layer transport by directed count pair</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>receiver→donor</th><th>rows / seeds</th><th>valid</th><th>eligible n</th><th>eligible adoption [95% CI]</th><th>follows donor prediction (valid)</th><th>aligned shift [95% CI]</th></tr></thead><tbody>@@ANSWER_QUERY_PAIR_ROWS@@</tbody></table></div>
  @@ANSWER_QUERY_INVALID@@
  <div class="section-conclusion"><span>当前结论 · Query-state sufficiency</span><p>Transport 在 Qwen L18→L26 与 Gemma L20→L31 之间突然开启；末层所有合法 eligible rows 都等于 donor prediction。把 Gemma 五个 <code>11</code> strict-invalid rows 作为 failure 后，保守 adoption 仍为 Qwen 100%、Gemma 99.58%。这证明 late query state 对“模型已经算出的 prediction”高度充分，但不证明它是干净 scalar counter、donor prediction 正确，或一个 query patch 足以决定多 token answer 的全部后续 tokens。</p></div>

  <h3>9.4 Answer-query geometric steering：count geometry 是否与生成 readout 对齐？</h3>
  <p>α=1 时，geometric vector 是 discovery centroid 从 receiver count 指向 target count 的差；control 是与之正交且 norm 完全相同的 random vector。Primary effect 是沿 target direction 的 generated-count shift，按 prompt/pair 与 control 配对。<em>Moved</em> 与 <em>target hit</em> 都基于最终 parsed count，而不是 token probability。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>pairs / seeds</th><th>changed geom. / random</th><th>moved geom. / random</th><th>Δ moved [95% CI]</th><th>target hit geom. / random</th><th>aligned shift geom. / random</th><th>Δ aligned [95% CI]</th><th>Holm p</th></tr></thead><tbody>@@CAUSAL_STEERING_ROWS@@</tbody></table></div>
  <div class="stat-grid"><figure class="stat-figure">@@CAUSAL_STEERING_SVG@@<figcaption><strong>图 10 · Geometry versus norm-matched random steering。</strong>横轴：paired aligned-count-shift 的 geometric−random contrast；纵轴：model/layer。正值表示 discovery-fit centroid delta 比等 norm 正交方向更有效地把 held-out output 推向 target。只有 Qwen L26 与 Gemma L31 在 family-wise correction 后明确为正。</figcaption></figure></div>
  <div class="section-conclusion"><span>当前结论 · Manipulability</span><p>后层 query geometry 确实进入生成 readout：Qwen L26 的 geometric−random aligned shift 为 +0.958，Gemma L31 为 +1.388（均 Holm p=0.0117）。但 exact target hit 只有 Qwen 8.75%、Gemma 7.5%，所以当前只支持“方向可操纵”，不支持“α=1 可精确设置目标整数”。</p></div>

  <h3>9.5 为什么 early decoding 很强，steering 却可能无效？</h3>
  <p>下表诊断 steering 使用的 10 个 discovery centroids。Endpoint correlation 是 count 与 centroid 在 1→10 chord 上投影的相关；monotonicity 检查该投影是否随 count 单调。Step CV 高表示相邻 count 步长不等；path/chord 高表示曲线弯折。一个方向可以高度可解码，却不一定是后续 readout 使用的 causal coordinate。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>variants</th><th>endpoint corr. mean (min)</th><th>minimum monotone fraction</th><th>mean step CV</th><th>mean successive-step cosine</th><th>mean tortuosity</th></tr></thead><tbody>@@CAUSAL_GEOMETRY_ROWS@@</tbody></table></div>

  <div class="section-conclusion"><span>当前结论 · Availability versus usage</span><p>Early/middle layers 的 centroid path 可接近单调、R² 很高，却在 steering 下近乎 inert；late path 更弯、更不等距，反而强烈改变 output。信息“可被线性读出”与模型“实际沿该方向生成”必须分开验证。</p></div>

  <h3>9.6 Artifact and label audit</h3>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>ablation shards / rows</th><th>patch families / rows</th><th>steering discovery / families / rows</th><th>greedy-label alignment</th></tr></thead><tbody>@@CAUSAL_AUDIT_ROWS@@</tbody></table></div>
  <div class="callout"><strong>Audit result。</strong>所有 expected shards、detail/summary/control tables 均存在；每个 patch row 成功；discovery NPZ shapes 与 finite values 已核验；causal baseline 与保存的 greedy behavior labels 完全一致；patched correctness 由最终 parsed continuation 重算；logs 中无 Traceback、OOM 或 FAILED marker。</div>
  <h4>Answer-query dense-patching audit</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>family shards / rows</th><th>successful / skipped</th><th>valid / invalid</th><th>eligible donor-prediction rows</th><th>greedy-label alignment</th></tr></thead><tbody>@@ANSWER_QUERY_AUDIT_ROWS@@</tbody></table></div>
  <div class="section-conclusion"><span>本节结论</span><p>四类 causal result 形成一致的定位：selected head bank 对 count magnitude 必要；单 needle endpoint 不足以跨 prompt 搬运 count；late <code>Total:</code> query residual 对 donor prediction 近乎充分；同一位置的 count geometry 还能方向性 steering。当前证据最支持“分布式聚合后在 query 侧形成 executable state”，而非 needle-local scalar transport。</p></div>
</section>

<section id="synthesis">
  <span class="section-kicker">10 · Mechanistic synthesis</span>
  <h2>综合机制：哪些解释被支持，哪些解释已经不够，哪些仍无法区分？</h2>
  <div class="evidence-ledger">
    <div class="ledger-row"><div><strong>H1 · Needle-end 存有独立、可直接运输的 running count</strong></div><div><span class="evidence-tag">Not sufficient</span>span-end probe 很强，但 exact endpoint patch 全部接近 null；单 endpoint transport 解释不足。</div></div>
    <div class="ledger-row"><div><strong>H2 · 一个最高排名 broad head 统一汇总所有 needles</strong></div><div><span class="evidence-tag">Rejected for Qwen</span>Qwen L29H3 是 first-occurrence selector；真正 broad coverage 分散在 24–35 个 global heads。</div></div>
    <div class="ledger-row"><div><strong>H3 · 多-head answer-query aggregation 对 count magnitude 必要</strong></div><div><span class="evidence-tag">Supported</span>ranked top-8 bank ablation 相对 layer-matched random 在两个模型都额外造成 undercount。</div></div>
    <div class="ledger-row"><div><strong>H4 · Late answer-query residual 携带模型已完成的 count decision</strong></div><div><span class="evidence-tag">Strongly supported</span>single-layer donor patch 在 late layers 近确定性复制 donor prediction，并跨全部 V4 panels/pairs 稳健。</div></div>
    <div class="ledger-row"><div><strong>H5 · Late state 是精确、等距、单维 scalar counter</strong></div><div><span class="evidence-tag">Not established</span>path 弯曲、步长不等；steering exact hit 低；Gemma “11”说明多-token realization 仍依赖后续 computation。</div></div>
    <div class="ledger-row"><div><strong>H6 · Qwen 使用固定的 partition-local aggregation circuit</strong></div><div><span class="evidence-tag">Open</span>存在 local heads，但 phenotype 跨 split/panel 稳定性弱；global bank 更稳定。</div></div>
  </div>
  <div class="section-conclusion"><span>本节结论</span><p>目前最小且不超出证据的机制是：prompt-reading 阶段在 needle spans 保留 count-related local states；多个 answer-query heads 以不同 breadth/selection profiles 聚合这些证据；该 bank 对避免 undercount 必要；聚合结果在后层 query residual 中变成可直接驱动首个 numeric decision 的 executable state。该链条仍缺少从具体 V/O head contributions 到 query residual 的逐层写入分解。</p></div>
</section>

<section id="limits">
  <span class="section-kicker">11 · Limits and next discriminating tests</span>
  <h2>目前仍缺什么：下一轮应优先做能区分机制的实验，而不是简单扩大同一 grid</h2>
  <div class="next-grid">
    <div class="next-item"><strong>1. Stable-global bank vs selector bank 的因果分离</strong><p>分别 ablate 13 个跨 panel×split 稳定 global aggregators、L29H3-like selectors、partition-local heads，并使用多组 layer-matched random controls；增加 leave-one-head-out 与 cumulative dose curve，判断必要性来自 broad heads 还是 rank bank 中的混合 phenotype。</p></div>
    <div class="next-item"><strong>2. 从 attention weight 到写入 residual 的路径分解</strong><p>对 priority heads 保存/重构 V、head output 与 O-projection contribution，在 <code>Total:</code> query 做 direct logit/readout alignment 和 sublayer causal tracing，定位 aggregation evidence 在哪一层被 MLP/residual 转成 count decision。</p></div>
    <div class="next-item"><strong>3. Full-needle 与 coordinated multi-endpoint patch</strong><p>单 endpoint null 不能排除分布式 source state。下一步用 exact tokenwise full-needle patch、position-aligned patch，以及多个新增 endpoints 的 coordinated patch；仍避免大规模 head-output patching。</p></div>
    <div class="next-item"><strong>4. Steering dose response 与 target-setting</strong><p>对 α∈{−2,…,2}、adjacent/non-local pairs、chord/polyline/local tangent 做 held-out sweep；测 monotonic response、overshoot、exact target hit 与 off-manifold norm，区分方向可操纵和精确可控制。</p></div>
    <div class="next-item"><strong>5. Gemma 的完整 phenotype/partition replication</strong><p>当前 exhaustive partition taxonomy 只对 Qwen 运行；Gemma top span-end 已较 broad，但尚不知道其 broad coverage 是少数 heads 还是稳定 bank。复现实验后才能比较两个模型是否使用同一种 aggregation architecture。</p></div>
    <div class="next-item"><strong>6. Error-correction 与 thinking-mode generalization</strong><p>高 count correct baselines 太少，correct/wrong causal strata power 不足。可通过调节 length/count 难度获得 matched correct/wrong prompts，再检验 patch 是否纠错；最后扩展到 thinking mode，比较 query state 与 CoT progress state 是否分离。</p></div>
  </div>
  <div class="callout"><strong>不要过度外推。</strong>本轮是 targeted causal screen：selected pairs、三个 steering depths、α=1、一个 matched random replicate，以及八个 query-patch layers。它支持上述具体因果主张，但不替代更大、预注册且多 control replicates 的 full sweep。</div>
  <div class="section-conclusion"><span>本节结论</span><p>优先级最高的是“stable global aggregator bank 的 phenotype-specific ablation”与“head V/O contribution → query residual 的写入路径”，因为它们直接补上当前机制链中唯一缺失的 causal edge；扩大 PCA 或重复更多同类 attention heatmaps 的信息增益较低。</p></div>
</section>

<section id="reproducibility">
  <span class="section-kicker">12 · Reproducibility</span>
  <h2>复现、归档与报告 provenance</h2>
  <p>Source run：<code>@@RUN_NAME@@</code>。本地与 Lambda filesystem 均保留完整 run；最终 answer-query bundle 的 SHA-256 为 <code>93776fdea92a07e358d52594969a7ab0d97ad9ef9107ed543d4a7daaa6567920</code>。报告由保存的 behavior labels、representation NPZ、raw answer-query attention rows、causal detail/summary/control tables 重新生成，不依赖服务器内存状态。</p>
  <div class="formula">PYTHONPATH=src python scripts/build_realistic_niah_v4_representation_report.py --run-root &lt;run-root&gt; --output reports/realistic_niah_v4_representation_report.html --repo-root .</div>
  <p>视觉系统固定为 Aurora：Midnight Indigo <code>#23165C</code>、Polar Violet <code>#6750E8</code>、Ice Cyan <code>#00C2FF</code>、Aurora Yellow <code>#F6E36A</code>、Aurora Teal <code>#00D4B4</code>、Aurora Green <code>#39E58C</code>、Polar Magenta <code>#C04DFF</code>、Sunset Pink <code>#FF5FA2</code>；正文/背景/网格分别使用 Night Black、Snow White、Frost Gray。后续 V4+ plots 应复用这一 palette 和语义映射。</p>
  <div class="section-conclusion"><span>本节结论</span><p>所有图表的数值来源、坐标含义、计算公式、selection split 与 inference unit 都在报告中显式记录；HTML 为 self-contained artifact，可离线打开并复查交互式 3D geometry。</p></div>
</section>
</main>
<footer>生成时间 @@GENERATED@@ · source run <code>@@RUN_NAME@@</code> · commit <code>@@COMMIT@@</code> · Aurora report system</footer>
<script>
const REP_DATA = @@REP_DATA@@;
const COLORS = ['#23165C','#4430A2','#6750E8','#9950F4','#C04DFF','#FF5FA2','#F6E36A','#39E58C','#00D4B4','#00C2FF'];
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
  ends.forEach((end,i)=>{ const e=transform(end); ctx.strokeStyle=['#00C2FF','#39E58C','#FF5FA2'][i]; ctx.beginPath();ctx.moveTo(o.x,o.y);ctx.lineTo(e.x,e.y);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(`PC${axes[i]+1}`,e.x+4,e.y-4); });
}
function draw() {
  const rect=canvas.getBoundingClientRect(), width=rect.width, height=rect.height;
  ctx.clearRect(0,0,width,height); ctx.fillStyle='#120D31'; ctx.fillRect(0,0,width,height);
  const rows=filteredRows(), axes=[+controls.x.value,+controls.y.value,+controls.z.value];
  const data=activeData(); document.getElementById('pca-stats').innerHTML=data?`<strong>${data.model} · ${data.pooling} · L${data.layer}</strong><br>PCA fit: v4.1 discovery · EVR ${data.explained_variance_ratio.slice(0,6).map((v,i)=>`PC${i+1} ${(100*v).toFixed(1)}%`).join(' · ')}`:'';
  const stats=statsFor(rows,axes), transform=makeTransform(rows,axes,width,height); projectedPoints=[];
  if (!rows.length || !stats || !transform) { ctx.fillStyle='#F6E36A';ctx.font='16px system-ui';ctx.textAlign='center';ctx.fillText('No trajectories match this filter.',width/2,height/2);document.getElementById('geometry-stats').textContent='No data';return; }
  drawAxes(transform,stats,axes,width,height);
  const paths=centroids(rows); const pointMode=controls.points.value;
  for (const group of paths) {
    const pts=group.path.map(d=>({...d,q:transform(axes.map(a=>d.p[a]))}));
    ctx.strokeStyle=group.split==='confirmation'?'#F8FBFF':'#8190A5'; ctx.lineWidth=group.split==='confirmation'?2.5:1.5; ctx.setLineDash(group.split==='confirmation'?[]:[6,5]);
    ctx.beginPath(); pts.forEach((d,i)=>i?ctx.lineTo(d.q.x,d.q.y):ctx.moveTo(d.q.x,d.q.y));ctx.stroke();ctx.setLineDash([]);
    for (const d of pts) { ctx.fillStyle=COLORS[d.count-1];ctx.strokeStyle='#161923';ctx.lineWidth=1;ctx.beginPath();ctx.arc(d.q.x,d.q.y,5.6,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#F8FBFF';ctx.font='10px system-ui';ctx.fillText(String(d.count),d.q.x+7,d.q.y-6); }
  }
  if (pointMode!=='centroids') {
    let pointRows=rows; if(pointMode==='confirmation') pointRows=rows.filter(r=>r[2]==='confirmation');
    const pts=pointRows.map(r=>({r,q:transform(axes.map(a=>r[7+a]))})).sort((a,b)=>a.q.z-b.q.z);
    for(const item of pts){const r=item.r,q=item.q;ctx.globalAlpha=r[2]==='confirmation'?.56:.18;ctx.fillStyle=COLORS[r[6]-1];ctx.strokeStyle=r[3]==='correct'?'#F8FBFF':(r[3]==='invalid'?'#FF5FA2':'#161923');ctx.lineWidth=r[3]==='correct'?1.8:.7;ctx.beginPath();ctx.arc(q.x,q.y,r[2]==='confirmation'?3.0:2.2,0,Math.PI*2);ctx.fill();ctx.stroke();projectedPoints.push({x:q.x,y:q.y,r});} ctx.globalAlpha=1;
  }
  ctx.fillStyle='#8190A5';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText(`${rows.length} occurrence points · ${new Set(rows.map(r=>r[1])).size} seeds`,12,height-12);
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
</html>"""


def build_report(run_root: Path, output: Path, repo_root: Path) -> None:
    run_root = run_root.resolve()
    primary: dict[str, dict[str, int]] = {}
    labels_lookup: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    labels_frames: dict[str, pd.DataFrame] = {}
    all_labels_frames: dict[str, pd.DataFrame] = {}
    projections: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_root = run_root / model / "numeric"
        primary[model] = _primary_layers(model_root)
        labels_lookup[model], labels_frames[model] = _n10_labels(model_root)
        all_labels_frames[model] = _all_generation_labels(model_root)
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
    behavior_panel_rows = _behavior_panel_rows(all_labels_frames)
    behavior_count_rows = _behavior_count_rows(all_labels_frames)
    behavior_count_pooled_rows = _behavior_count_pooled_rows(all_labels_frames)
    sensitivity_rows = _sensitivity_rows(run_root)
    attention_top_rows = _attention_top_rows(run_root)
    partition_summary = _qwen_partition_summary(run_root)
    span_end_alignment_rows = _span_end_alignment_rows(run_root)
    span_end_pooled_rows = _span_end_pooled_rows(run_root)
    span_end_nested_rows = _span_end_nested_rows(run_root)
    causal_audit = audit_screen_8h(run_root)
    causal_frames, _causal_paths = _causal_frames(run_root)
    causal_ablation_rows = _causal_ablation_rows(causal_frames)
    causal_patching_rows = _causal_patching_rows(causal_frames)
    causal_steering_rows = _causal_steering_rows(causal_frames)
    causal_geometry_rows = _causal_geometry_rows(causal_frames)
    answer_query_frames, answer_query_audit = _answer_query_frames(run_root)
    answer_query_layer_rows = answer_query_frames["layer_summary"].to_dict("records")
    answer_query_variant_rows = _answer_query_final_rows(
        answer_query_frames["variant_summary"]
    )
    answer_query_pair_rows = _answer_query_final_rows(
        answer_query_frames["pair_summary"]
    )
    commit = _git_commit(repo_root)
    replacements = {
        "@@COMMIT@@": html.escape(commit[:12]),
        "@@GENERATED@@": html.escape(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ),
        "@@RUN_NAME@@": html.escape(run_root.name),
        "@@METRIC_ROWS@@": _table_metric_html(metric_rows),
        "@@BEHAVIOR_ROWS@@": _table_behavior_html(behavior_rows),
        "@@BEHAVIOR_ACCURACY_SVG@@": _behavior_accuracy_svg(behavior_count_rows),
        "@@BEHAVIOR_PANEL_ROWS@@": _table_behavior_panel_html(
            behavior_panel_rows
        ),
        "@@BEHAVIOR_COUNT_ROWS@@": _table_behavior_count_html(
            behavior_count_pooled_rows
        ),
        "@@BEHAVIOR_CONCLUSION@@": _behavior_conclusion_html(
            behavior_panel_rows, behavior_count_pooled_rows
        ),
        "@@REPRESENTATION_R2_SVG@@": _representation_r2_svg(metric_rows),
        "@@REPRESENTATION_CONCLUSION@@": _representation_conclusion_html(
            metric_rows, sensitivity_rows
        ),
        "@@SENSITIVITY_ROWS@@": _table_sensitivity_html(sensitivity_rows),
        "@@ATTENTION_BREADTH_SVG@@": _attention_breadth_svg(attention_top_rows),
        "@@ATTENTION_TOP_ROWS@@": _table_attention_top_html(attention_top_rows),
        "@@PARTITION_PHENOTYPE_SVG@@": _partition_phenotype_svg(
            partition_summary
        ),
        "@@PARTITION_BANK_ROWS@@": _table_partition_bank_html(
            partition_summary["rows"]
        ),
        "@@STABLE_GLOBAL_HEADS@@": html.escape(
            ", ".join(partition_summary["stable_global_heads"])
        ),
        "@@ATTENTION_CONCLUSION@@": _attention_conclusion_html(
            attention_top_rows, partition_summary
        ),
        "@@SPAN_END_ALIGNMENT_ROWS@@": _table_span_end_alignment_html(
            span_end_alignment_rows
        ),
        "@@SPAN_END_POOLED_ROWS@@": _table_span_end_pooled_html(span_end_pooled_rows),
        "@@SPAN_END_ALIGNMENT_SVG@@": _span_end_alignment_svg(span_end_alignment_rows),
        "@@SPAN_END_NESTED_ROWS@@": _table_span_end_nested_html(span_end_nested_rows),
        "@@SPAN_END_NESTED_SVG@@": _span_end_nested_svg(span_end_nested_rows),
        "@@SPAN_END_CONCLUSION@@": _span_end_conclusion_html(
            span_end_pooled_rows, span_end_alignment_rows, span_end_nested_rows
        ),
        "@@CAUSAL_ABLATION_ROWS@@": _table_causal_ablation_html(causal_ablation_rows),
        "@@CAUSAL_PATCHING_ROWS@@": _table_causal_patching_html(causal_patching_rows),
        "@@CAUSAL_STEERING_ROWS@@": _table_causal_steering_html(causal_steering_rows),
        "@@CAUSAL_GEOMETRY_ROWS@@": _table_causal_geometry_html(causal_geometry_rows),
        "@@CAUSAL_AUDIT_ROWS@@": _table_causal_audit_html(causal_audit),
        "@@ANSWER_QUERY_LAYER_ROWS@@": _table_answer_query_layer_html(
            answer_query_layer_rows
        ),
        "@@ANSWER_QUERY_VARIANT_ROWS@@": _table_answer_query_variant_html(
            answer_query_variant_rows
        ),
        "@@ANSWER_QUERY_PAIR_ROWS@@": _table_answer_query_pair_html(
            answer_query_pair_rows
        ),
        "@@ANSWER_QUERY_INVALID@@": _answer_query_invalid_html(
            answer_query_frames["invalid_rows"]
        ),
        "@@ANSWER_QUERY_AUDIT_ROWS@@": _table_answer_query_audit_html(
            answer_query_audit
        ),
        "@@CAUSAL_ABLATION_SVG@@": _forest_svg(
            causal_ablation_rows,
            estimate_key="count_shift_difference",
            low_key="count_shift_difference_low",
            high_key="count_shift_difference_high",
            title="Discovery-ranked broad-head ablation versus layer-matched random",
            axis_label="paired mean count shift: ranked minus random (negative = stronger undercount)",
            label=lambda row: f"{row['model']} top-{row['top_n']}",
        ),
        "@@CAUSAL_PATCHING_SVG@@": _forest_svg(
            causal_patching_rows,
            estimate_key="aligned_shift",
            low_key="aligned_shift_low",
            high_key="aligned_shift_high",
            title="Exact needle-end residual transport",
            axis_label="mean direction-aligned generated-count shift",
            label=lambda row: f"{row['model']} L{row['layer']}",
        ),
        "@@CAUSAL_STEERING_SVG@@": _forest_svg(
            causal_steering_rows,
            estimate_key="aligned_difference",
            low_key="aligned_difference_low",
            high_key="aligned_difference_high",
            title="Centroid-delta steering versus norm-matched orthogonal random",
            axis_label="paired direction-aligned count shift: geometric minus random",
            label=lambda row: f"{row['model']} L{row['layer']}",
        ),
        "@@ANSWER_QUERY_ADOPTION_SVG@@": _forest_svg(
            answer_query_layer_rows,
            estimate_key="eligible_donor_adoption_rate",
            low_key="eligible_donor_adoption_rate_ci95_low",
            high_key="eligible_donor_adoption_rate_ci95_high",
            title="Exact answer-query donor-prediction transport",
            axis_label="eligible rows adopting donor baseline prediction",
            label=lambda row: f"{row['model']} L{int(row['layer'])}",
        ),
        "@@CAUSAL_CONCLUSION@@": _causal_conclusion_html(
            causal_ablation_rows,
            causal_patching_rows,
            causal_steering_rows,
            answer_query_layer_rows,
        ),
        "@@STATIC_FIGURES@@": _static_figure_html(projections),
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
                "projection_rows": sum(
                    len(item["rows"]) for item in projections.values()
                ),
                "primary_layers": primary,
                "behavior_confirmation_rows": sum(
                    int(row["n"]) for row in behavior_panel_rows
                ),
                "attention_top_rows": len(attention_top_rows),
                "qwen_stable_global_heads": len(
                    partition_summary["stable_global_heads"]
                ),
                "causal_audit_validated": bool(causal_audit["validated"]),
                "causal_summary_rows": {
                    "ablation": len(causal_ablation_rows),
                    "endpoint_patching": len(causal_patching_rows),
                    "steering": len(causal_steering_rows),
                    "centroid_geometry": len(causal_geometry_rows),
                },
                "answer_query_audit_validated": bool(answer_query_audit["validated"]),
                "answer_query_summary_rows": {
                    "layers": len(answer_query_layer_rows),
                    "final_variants": len(answer_query_variant_rows),
                    "final_pairs": len(answer_query_pair_rows),
                },
                "answer_query_invalid_rows": int(
                    len(answer_query_frames["invalid_rows"])
                ),
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
