from __future__ import annotations

"""Audit and analyze follow-up 23 without pooling models."""

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


TERMS = ("intercept", "I", "C", "P", "IC", "IP", "CP", "ICP")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def design_row(identity: bool, context: bool, position: bool) -> np.ndarray:
    i, c, p = float(identity), float(context), float(position)
    return np.asarray([1.0, i, c, p, i * c, i * p, c * p, i * c * p], dtype=float)


def multi_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("R2 arrays must be matching [sample,response] matrices")
    denominator = float(np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2))
    if denominator <= 0:
        raise ValueError("Confirmation responses have zero variance")
    return 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
    return test_x @ coefficients


def factorial_analysis(model_root: Path, rows: Sequence[dict]) -> dict:
    discovery = [row for row in rows if row["split"] == "discovery"]
    confirmation = [row for row in rows if row["split"] == "confirmation"]
    if len(discovery) != 160 or len(confirmation) != 80:
        raise RuntimeError("Factorial split must contain 160 discovery and 80 confirmation forwards")
    base = [row for row in discovery if row["cell"] == "000"]
    if len(base) != 20:
        raise RuntimeError("Discovery base cell must contain 20 seeds")
    base_states = np.stack(
        [np.load(model_root / row["state_path"])["states"].astype(np.float64) for row in base]
    )
    if base_states.shape[1] != 10:
        raise RuntimeError("Factorial base states do not contain ten endpoints")
    centroids = base_states.mean(axis=0)
    centered_centroids = centroids - centroids.mean(axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered_centroids, full_matrices=False)
    components = vh[:3]
    centroid_capture = float(np.sum(singular[:3] ** 2) / np.sum(singular**2))

    def materialize(selected: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for row in selected:
            states = np.load(model_root / row["state_path"])["states"].astype(np.float64)
            if states.shape != centroids.shape:
                raise RuntimeError("A factorial state array has the wrong shape")
            coordinates = (states - centroids) @ components.T
            x = design_row(row["identity"], row["context"], row["position"])
            xs.extend([x] * 10)
            ys.extend(coordinates)
        return np.asarray(xs), np.asarray(ys)

    train_x, train_y = materialize(discovery)
    test_x, test_y = materialize(confirmation)
    full_prediction = fit_predict(train_x, train_y, test_x)
    full_r2 = multi_r2(test_y, full_prediction)
    factor_terms = {
        "identity": {"I", "IC", "IP", "ICP"},
        "context": {"C", "IC", "CP", "ICP"},
        "position": {"P", "IP", "CP", "ICP"},
    }
    delta_r2 = {}
    reduced_r2 = {}
    for factor, removed in factor_terms.items():
        keep = [index for index, term in enumerate(TERMS) if term not in removed]
        prediction = fit_predict(train_x[:, keep], train_y, test_x[:, keep])
        score = multi_r2(test_y, prediction)
        reduced_r2[factor] = score
        delta_r2[factor] = full_r2 - score
    np.savez_compressed(
        model_root / "factorial_frozen_geometry.npz",
        centroids=centroids.astype(np.float32),
        components=components.astype(np.float32),
    )
    return {
        "discovery_forwards": len(discovery),
        "confirmation_forwards": len(confirmation),
        "endpoint_states_per_forward": 10,
        "centroid_rank3_capture": centroid_capture,
        "confirmation_full_model_r2": full_r2,
        "confirmation_reduced_r2": reduced_r2,
        "confirmation_incremental_delta_r2": delta_r2,
        "definition": "Delta R2_F is held-out full-model R2 minus held-out R2 after dropping every term containing factor F.",
    }


def per_seed_contrast(
    rows: Sequence[dict],
    *,
    candidate: str,
    control: str,
    value: Callable[[dict], float],
) -> np.ndarray:
    units: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        units.setdefault((int(row["seed"]), int(row["gold_count"])), {})[str(row["arm"])] = row
    per_seed: dict[int, list[float]] = {}
    required = {"natural", "candidate_halo_edge_block", "distance_random_control", "attention_mass_control"}
    for (seed, _count), group in units.items():
        if set(group) != required:
            raise RuntimeError("An outside-context unit lacks one frozen arm")
        per_seed.setdefault(seed, []).append(value(group[candidate]) - value(group[control]))
    return np.asarray([np.mean(per_seed[seed]) for seed in sorted(per_seed)], dtype=float)


def bootstrap(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(int(seed))
    estimates = np.mean(rng.choice(values, size=(int(draws), len(values)), replace=True), axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "positive_seed_fraction": float(np.mean(values > 0)),
        "seed_count": int(len(values)),
    }


def outside_analysis(rows: Sequence[dict], *, draws: int) -> dict:
    if len(rows) != 400:
        raise RuntimeError(f"Outside-context stage expected 400 rows, got {len(rows)}")
    keys = {(row["model_label"], int(row["seed"]), int(row["gold_count"]), row["arm"]) for row in rows}
    if len(keys) != 400:
        raise RuntimeError("Outside-context rows contain duplicate keys")
    error = lambda row: abs(float(row["expected_count"]) - int(row["gold_count"]))
    broad_damage = lambda row: -float(row["retrieval_bank_broad_score_mean"])
    margin_damage = lambda row: -float(row["correct_count_margin"])
    metrics = {}
    counter = 0
    for control in ("distance_random_control", "attention_mass_control"):
        label = control.replace("_control", "")
        for name, function in (
            ("expected_error", error),
            ("broad_score_damage", broad_damage),
            ("correct_margin_damage", margin_damage),
        ):
            counter += 1
            values = per_seed_contrast(
                rows,
                candidate="candidate_halo_edge_block",
                control=control,
                value=function,
            )
            metrics[f"{name}_candidate_minus_{label}"] = bootstrap(
                values, draws=draws, seed=4452300 + counter
            )
    expected_gates = [
        metrics["expected_error_candidate_minus_distance_random"]["ci95_low"] > 0,
        metrics["expected_error_candidate_minus_attention_mass"]["ci95_low"] > 0,
    ]
    return {
        "rows": len(rows),
        "unique_keys": len(keys),
        "metrics": metrics,
        "candidate_exceeds_both_controls": bool(all(expected_gates)),
        "definition": "Positive expected-error specificity means candidate halo-edge removal harms count more than the stated matched control.",
    }


def analyze_model(root: Path, model: str, *, draws: int) -> dict:
    model_root = root / model
    completion = json.loads((model_root / "complete.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError(f"{model} completion marker is not complete")
    factorial_rows = load_jsonl(model_root / "factorial_rows.jsonl")
    outside_rows = load_jsonl(model_root / "outside_context_rows.jsonl")
    factorial_keys = {(int(row["seed"]), row["cell"]) for row in factorial_rows}
    if len(factorial_rows) != 240 or len(factorial_keys) != 240:
        raise RuntimeError("Factorial stage expected 240 unique rows")
    result = {
        "model": model,
        "status": "PASS",
        "factorial": factorial_analysis(model_root, factorial_rows),
        "outside_context": outside_analysis(outside_rows, draws=draws),
        "boundary": "The factorial estimates controlled deformation under token-level matched interventions; it is not a census or unique decomposition of natural prompt noise.",
    }
    (model_root / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (model_root / "analysis_audit.json").write_text(json.dumps({"status": "PASS", "factorial_rows": 240, "outside_context_rows": 400}, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    summaries = [analyze_model(args.run_root.resolve(), model, draws=args.bootstrap_draws) for model in args.models]
    combined = {"schema_version": "realistic_niah_v4_4_5_noise_factorial_analysis_v1", "status": "PASS", "models": summaries, "models_pooled": False}
    output = args.run_root.resolve() / "analysis_summary.json"
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
