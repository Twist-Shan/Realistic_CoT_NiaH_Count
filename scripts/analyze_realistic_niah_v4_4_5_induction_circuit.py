from __future__ import annotations

"""Audit and analyze follow-up 22 without pooling models."""

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_by_seed(rows: Sequence[dict], value: Callable[[dict], float]) -> np.ndarray:
    arms: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        arms.setdefault((int(row["seed"]), int(row["gold_count"])), {})[str(row["arm"])] = row
    per_seed: dict[int, list[float]] = {}
    for (seed, _count), group in arms.items():
        if set(group) != {"natural", "candidate_edge_block", "mass_distance_control"}:
            raise RuntimeError("A canonical unit lacks one frozen arm")
        per_seed.setdefault(seed, []).append(
            value(group["candidate_edge_block"]) - value(group["mass_distance_control"])
        )
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


def analyze_model(root: Path, model: str, *, draws: int) -> dict:
    model_root = root / model
    completion = json.loads((model_root / "complete.json").read_text(encoding="utf-8"))
    synthetic = json.loads((model_root / "synthetic_audit.json").read_text(encoding="utf-8"))
    if completion["status"] == "complete_no_retained_head":
        result = {
            "model": model,
            "status": "PASS",
            "scientific_decision": "not_supported",
            "failed_stage": "synthetic_relation_gate",
            "synthetic": synthetic,
            "canonical_rows": 0,
        }
        (model_root / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (model_root / "analysis_audit.json").write_text(json.dumps({"status": "PASS", "canonical_expected": False}, indent=2) + "\n", encoding="utf-8")
        return result
    if completion.get("status") != "complete":
        raise RuntimeError(
            f"Unexpected completion status for {model}: {completion.get('status')}"
        )
    rows = load_jsonl(model_root / "detail.jsonl")
    keys = {(row["model_label"], int(row["seed"]), int(row["gold_count"]), row["arm"]) for row in rows}
    if len(rows) != 300 or len(keys) != 300:
        raise RuntimeError(f"{model} expected 300 canonical rows/keys, got {len(rows)}/{len(keys)}")
    if any(not np.isfinite(float(row["expected_count"])) for row in rows):
        raise RuntimeError("Canonical expected count contains nonfinite values")
    expected_error = paired_by_seed(rows, lambda row: abs(float(row["expected_count"]) - int(row["gold_count"])))
    strict_error = paired_by_seed(rows, lambda row: float(row["strict_absolute_error"]))
    broad_damage = paired_by_seed(rows, lambda row: -float(row["retrieval_bank_broad_score_mean"]))
    margin_damage = paired_by_seed(rows, lambda row: -float(row["correct_count_margin"]))
    metrics = {
        "expected_absolute_error_candidate_minus_control": bootstrap(expected_error, draws=draws, seed=4452201),
        "strict_absolute_error_candidate_minus_control": bootstrap(strict_error, draws=draws, seed=4452202),
        "broad_score_damage_candidate_minus_control": bootstrap(broad_damage, draws=draws, seed=4452203),
        "correct_margin_damage_candidate_minus_control": bootstrap(margin_damage, draws=draws, seed=4452204),
    }
    synthetic_pass = bool(synthetic.get("retained_heads"))
    behavior_pass = metrics["expected_absolute_error_candidate_minus_control"]["ci95_low"] > 0
    result = {
        "model": model,
        "status": "PASS",
        "scientific_decision": "supported" if synthetic_pass and behavior_pass else "not_supported",
        "synthetic_relation_gate": synthetic_pass,
        "canonical_matched_block_gate": behavior_pass,
        "canonical_rows": len(rows),
        "paired_seed_count": len(expected_error),
        "metrics": metrics,
        "boundary": "Supports or rejects the classical induction label for the frozen primary head; it does not alter the established distributed span-evidence mechanism.",
    }
    (model_root / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (model_root / "analysis_audit.json").write_text(json.dumps({"status": "PASS", "rows": len(rows), "unique_keys": len(keys), "paired_seeds": len(expected_error)}, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    summaries = [analyze_model(args.run_root.resolve(), model, draws=args.bootstrap_draws) for model in args.models]
    combined = {"schema_version": "realistic_niah_v4_4_5_induction_analysis_v1", "status": "PASS", "models": summaries, "models_pooled": False}
    output = args.run_root.resolve() / "analysis_summary.json"
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
