from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CONTROL_MODE = "native_thinking_control"
TREATMENT_MODE = "native_thinking"

METRIC_DIRECTIONS = {
    "registered_success": "higher_is_better",
    "exact_count": "higher_is_better",
    "truncated": "lower_is_better",
    "output_tokens": "lower_is_better",
    "reasoning_enumeration_restart_count": "lower_is_better",
    "reasoning_duplicate_record_mentions": "lower_is_better",
    "reasoning_duplicate_lines": "lower_is_better",
    "overthinking_flag": "lower_is_better",
}


def load_request_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _metric_value(row: dict[str, Any], metric: str) -> float:
    evaluation = row.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Missing evaluation in {row.get('request_id')}")
    value = (
        row.get("output_tokens")
        if metric == "output_tokens"
        else evaluation.get(metric)
    )
    if value is None:
        raise ValueError(
            f"Missing metric {metric!r} in {row.get('request_id')}"
        )
    return float(value)


def summarize_overthinking_smoke(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    models: set[str] = set()
    stimuli_by_model: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        mode = str(row.get("prompt_mode"))
        if mode not in {CONTROL_MODE, TREATMENT_MODE}:
            continue
        model = str(row["model_label"])
        stimulus = str(row["stimulus_id"])
        key = (model, stimulus, mode)
        if key in indexed:
            raise ValueError(f"Duplicate smoke result for {key}")
        indexed[key] = row
        models.add(model)
        stimuli_by_model[model].add(stimulus)

    if not indexed:
        raise ValueError("No registered smoke-control rows were found")

    summaries: dict[str, Any] = {}
    for model in sorted(models):
        stimuli = sorted(stimuli_by_model[model])
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for stimulus in stimuli:
            control = indexed.get((model, stimulus, CONTROL_MODE))
            treatment = indexed.get((model, stimulus, TREATMENT_MODE))
            if control is None or treatment is None:
                raise ValueError(
                    f"Incomplete paired smoke result for {model}/{stimulus}"
                )
            pairs.append((control, treatment))

        metric_summaries: dict[str, Any] = {}
        for metric, direction in METRIC_DIRECTIONS.items():
            control_values = [
                _metric_value(control, metric)
                for control, _ in pairs
            ]
            treatment_values = [
                _metric_value(treatment, metric)
                for _, treatment in pairs
            ]
            deltas = [
                treatment - control
                for control, treatment in zip(
                    control_values,
                    treatment_values,
                )
            ]
            improvement_multiplier = 1 if direction == "higher_is_better" else -1
            oriented = [delta * improvement_multiplier for delta in deltas]
            metric_summaries[metric] = {
                "direction": direction,
                "control_mean": sum(control_values) / len(control_values),
                "treatment_mean": sum(treatment_values) / len(treatment_values),
                "treatment_minus_control": sum(deltas) / len(deltas),
                "improved_pairs": sum(value > 0 for value in oriented),
                "worsened_pairs": sum(value < 0 for value in oriented),
                "tied_pairs": sum(value == 0 for value in oriented),
            }
        summaries[model] = {
            "paired_stimuli": len(pairs),
            "metrics": metric_summaries,
        }

    return {
        "schema_version": "realistic_niah_overthinking_smoke_summary_v2",
        "control_mode": CONTROL_MODE,
        "treatment_mode": TREATMENT_MODE,
        "models": summaries,
    }


def summarize_guarded_smoke(
    rows: Iterable[dict[str, Any]],
    *,
    expected_models: Iterable[str] | None = None,
    expected_requests_per_model: int | None = None,
) -> dict[str, Any]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if str(row.get("prompt_mode")) != TREATMENT_MODE:
            continue
        model = str(row["model_label"])
        stimulus = str(row["stimulus_id"])
        key = (model, stimulus)
        if key in indexed:
            raise ValueError(f"Duplicate guarded smoke result for {key}")
        indexed[key] = row
        by_model[model].append(row)

    if not indexed:
        raise ValueError("No guarded native-thinking smoke rows were found")

    summaries: dict[str, Any] = {}
    for model in sorted(by_model):
        model_rows = by_model[model]
        evaluations = [row["evaluation"] for row in model_rows]
        output_tokens = [int(row["output_tokens"]) for row in model_rows]
        truncations = sum(
            bool(evaluation["truncated"]) for evaluation in evaluations
        )
        summaries[model] = {
            "requests": len(model_rows),
            "truncations": truncations,
            "zero_truncations": truncations == 0,
            "parse_failures": sum(
                evaluation["parse_status"] == "parse_fail"
                for evaluation in evaluations
            ),
            "format_failures": sum(
                not bool(evaluation["response_format_compliant"])
                for evaluation in evaluations
            ),
            "registered_accuracy": sum(
                bool(evaluation["registered_success"])
                for evaluation in evaluations
            )
            / len(evaluations),
            "exact_count_accuracy": sum(
                bool(evaluation["exact_count"])
                for evaluation in evaluations
            )
            / len(evaluations),
            "overthinking_flags": sum(
                bool(evaluation["overthinking_flag"])
                for evaluation in evaluations
            ),
            "mean_output_tokens": sum(output_tokens) / len(output_tokens),
            "max_output_tokens": max(output_tokens),
        }

    observed_models = set(by_model)
    expected = set(expected_models or observed_models)
    missing_models = sorted(expected - observed_models)
    unexpected_models = sorted(observed_models - expected)
    request_count_mismatches = {
        model: summary["requests"]
        for model, summary in summaries.items()
        if expected_requests_per_model is not None
        and summary["requests"] != expected_requests_per_model
    }
    total_truncations = sum(
        summary["truncations"] for summary in summaries.values()
    )
    gate_passed = (
        total_truncations == 0
        and not missing_models
        and not unexpected_models
        and not request_count_mismatches
    )
    return {
        "schema_version": "realistic_niah_guarded_smoke_summary_v2",
        "prompt_mode": TREATMENT_MODE,
        "gate": {
            "requirement": "zero_truncations",
            "passed": gate_passed,
            "total_truncations": total_truncations,
            "missing_models": missing_models,
            "unexpected_models": unexpected_models,
            "request_count_mismatches": request_count_mismatches,
        },
        "models": summaries,
    }
