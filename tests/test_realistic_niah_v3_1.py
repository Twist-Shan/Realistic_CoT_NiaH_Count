from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from realistic_niah_v3_1.analysis import (
    accuracy_condition_table,
    add_derived_predictors,
    behavior_tables,
    bias_condition_table,
    symmetric_trimmed_mean,
)
from realistic_niah_v3_1.cot_style import (
    build_blinded_annotation_samples,
    classify_counting_style,
    classify_request_table,
    evaluate_style_annotations,
)
from realistic_niah_v3_1.laws import (
    CANDIDATES,
    CANDIDATE_BY_NAME,
    cross_validate_candidate,
    design_matrix,
    fit_law,
    nested_held_axis_validation,
    nested_seed_validation,
    probability_distribution_diagnostics,
)
from realistic_niah_v3_1.sharding import formal_shard_plan
from realistic_niah_v3_1.spec import (
    EXPECTED_REQUESTS,
    EXPECTED_SHARDS,
    EXPECTED_STIMULI,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    SEEDS,
    V31_FREEZE_PROTOCOL,
    V31_RUN_PROTOCOL,
)
from realistic_niah_v3_1.stimuli import default_freeze_spec


def _synthetic_requests() -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(("model-a", "model-b")):
        for seed in range(30):
            for count in (1, 2, 4):
                for length in (1_000, 2_000, 5_000):
                    correct = (seed + count + model_index) % 4 != 0
                    predicted = count if correct else count + 1
                    rows.append(
                        {
                            "request_id": f"{model}/{seed}/{count}/{length}",
                            "model_label": model,
                            "comparison_slot": model,
                            "prompt_mode": "direct",
                            "seed": seed,
                            "N": count,
                            "L": length,
                            "predicted_count": predicted,
                            "parse_success": True,
                            "exact_count": correct,
                            "format_compliant": True,
                            "strict_registered_success": correct,
                            "truncated": False,
                            "signed_deviation": float(predicted - count),
                            "absolute_deviation": float(abs(predicted - count)),
                            "reasoning_text": "",
                            "final_text": f"Total: {predicted}",
                            "raw_output_text": f"Total: {predicted}",
                            "separate_reasoning_text": "",
                        }
                    )
    return add_derived_predictors(pd.DataFrame(rows))


def test_registered_grid_and_request_accounting() -> None:
    assert PASSAGE_LENGTHS == (
        1_000,
        2_000,
        3_000,
        5_000,
        8_000,
        10_000,
        15_000,
        20_000,
    )
    assert NEEDLE_COUNTS == tuple(range(1, 11)) + (12, 15, 18, 20)
    assert SEEDS == tuple(range(1234, 1264))
    assert EXPECTED_STIMULI == 3_360
    assert EXPECTED_SHARDS == 48
    assert EXPECTED_REQUESTS == 161_280
    assert V31_RUN_PROTOCOL.request_id_namespace == "v3.1"
    assert V31_FREEZE_PROTOCOL.stimulus_id_prefix == "V31_"
    freeze = default_freeze_spec()
    assert freeze.passage_lengths == PASSAGE_LENGTHS
    assert freeze.seeds == SEEDS


def test_config_matches_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "realistic_niah_v3_1.json").read_text(encoding="utf-8")
    )
    assert tuple(config["target_passage_tokens"]) == PASSAGE_LENGTHS
    assert tuple(config["needle_counts"]) == NEEDLE_COUNTS
    assert tuple(config["seeds"]) == SEEDS
    assert config["expected_stimuli"] == EXPECTED_STIMULI
    assert config["expected_requests_total"] == EXPECTED_REQUESTS
    assert config["outcomes"]["primary_bias"]["tail_fraction_each_side"] == 0.1
    assert (
        config["empirical_law"]["accuracy_observation_models"][
            "distributional_confirmatory"
        ]["overdispersed_distribution"]
        == "Beta-Binomial"
    )


def test_shard_plan_is_complete_and_namespaced() -> None:
    plan = formal_shard_plan()
    assert plan["expected_shards"] == 48
    assert plan["expected_stimuli_per_shard"] == 3_360
    assert plan["expected_requests"] == 161_280
    assert plan["request_id_namespace"] == "v3.1"
    assert len(plan["tasks"]) == 48
    assert {task["prompt_mode"] for task in plan["tasks"]} == {
        "direct",
        "enumeration_index",
        "enumeration_bullet",
        "native_thinking",
    }


def test_symmetric_trim_removes_exactly_three_each_tail_at_30() -> None:
    values = [-10_000, -1_000, -100] + [2.0] * 24 + [100, 1_000, 10_000]
    assert symmetric_trimmed_mean(values) == 2.0


def test_accuracy_decomposition_and_bias_coverage() -> None:
    requests = _synthetic_requests()
    mask = (
        (requests["comparison_slot"] == "model-a")
        & (requests["N"] == 1)
        & (requests["L"] == 1_000)
    )
    parse_fail_indices = requests.loc[mask].index[:11]
    requests.loc[parse_fail_indices, "parse_success"] = False
    requests.loc[parse_fail_indices, "exact_count"] = False
    requests.loc[parse_fail_indices, "predicted_count"] = float("nan")
    requests.loc[parse_fail_indices, "signed_deviation"] = float("nan")
    requests.loc[parse_fail_indices, "absolute_deviation"] = float("nan")
    accuracy = accuracy_condition_table(requests)
    row = accuracy.loc[
        (accuracy["comparison_slot"] == "model-a")
        & (accuracy["N"] == 1)
        & (accuracy["L"] == 1_000)
    ].iloc[0]
    assert row["n_total"] == 30
    assert row["n_parseable"] == 19
    bias = bias_condition_table(requests)
    bias_row = bias.loc[
        (bias["comparison_slot"] == "model-a") & (bias["N"] == 1) & (bias["L"] == 1_000)
    ].iloc[0]
    assert not bool(bias_row["bias_law_eligible"])
    assert bias_row["bias_coverage_status"] == "insufficient_conditional_bias_coverage"
    summary, accuracy_cells, bias_cells, outcomes = behavior_tables(requests)
    assert not summary.empty
    assert len(accuracy_cells) == len(bias_cells) == 18
    assert set(outcomes["outcome_class"]).issuperset(
        {"parse_failure", "strict_success"}
    )


def test_candidate_registry_is_bounded_and_hierarchical() -> None:
    assert len(CANDIDATES) == 13
    interactions = [candidate for candidate in CANDIDATES if candidate.parent]
    assert len(interactions) == 4
    for candidate in interactions:
        parent = CANDIDATE_BY_NAME[candidate.parent]
        assert candidate.interaction_feature is not None
        assert set(parent.features).issubset(candidate.features)


def test_model_specific_slopes_and_all_observation_models_fit() -> None:
    requests = _synthetic_requests()
    candidate = CANDIDATE_BY_NAME["log_additive"]
    cells = accuracy_condition_table(requests)
    fit = fit_law(cells, candidate, "binomial")
    matrix, names = design_matrix(cells, candidate, fit.levels, fit.scaler)
    assert matrix.shape[1] == 2 * (1 + len(candidate.features))
    assert "beta[model-a]:logN" in names
    assert "beta[model-b]:logL" in names
    for outcome_model in ("bernoulli", "binomial", "beta_binomial", "bias"):
        result, coefficients, fitted = cross_validate_candidate(
            requests,
            prompt_mode="direct",
            candidate=candidate,
            outcome_model=outcome_model,
        )
        assert result["converged"]
        assert result["cv_primary_loss_mean"] >= 0
        assert not coefficients.empty
        assert fitted.levels == ("model-a", "model-b")


def test_probability_distribution_diagnostics_cover_all_cells() -> None:
    requests = _synthetic_requests()
    cells = accuracy_condition_table(requests)
    fit = fit_law(cells, CANDIDATE_BY_NAME["log_additive"], "beta_binomial")
    detail, calibration, summary = probability_distribution_diagnostics(fit, cells)
    assert len(detail) == len(cells)
    assert not calibration.empty
    assert summary["cells"] == len(cells)
    assert {"pi50_covered", "pi80_covered", "pi95_covered"}.issubset(detail.columns)


def test_nested_validation_selects_structure_without_outer_data() -> None:
    requests = _synthetic_requests()
    held_seed = nested_seed_validation(
        requests,
        outcome_models=("bernoulli",),
        interaction_bootstrap_replicates=0,
    )
    assert len(held_seed) == 5
    assert held_seed["held_seeds"].str.len().gt(0).all()
    held_n = nested_held_axis_validation(
        requests,
        axis="N",
        outcome_models=("bernoulli",),
        interaction_bootstrap_replicates=0,
    )
    assert set(held_n["held_value"]) == {1, 2, 4}
    assert set(held_n["validation_kind"]) == {
        "boundary_extrapolation",
        "interpolation",
    }


def test_cot_style_rules_and_blinded_sampling() -> None:
    indexed = classify_counting_style(
        reasoning_text="1. Madison: 4\n2. Boston: 8\nSo 4 + 8 = 12",
        final_text="Total: 2",
    )
    assert indexed.index_enumeration
    assert indexed.arithmetic_grouping
    assert indexed.dominant_style == "index_enumeration"
    bullet = classify_counting_style(
        reasoning_text="- Madison: 4\n- Boston: 8",
        final_text="Total: 2",
    )
    assert bullet.bullet_enumeration
    words = classify_counting_style(
        reasoning_text="First Madison is present; second Boston is present.",
        final_text="Total: 2",
    )
    assert words.word_enumeration
    answer = classify_counting_style(reasoning_text="", final_text="Total: 2")
    assert answer.answer_only and answer.observability == "final_only"

    requests = _synthetic_requests().head(100).copy()
    requests.loc[requests.index[:10], "reasoning_text"] = "1. A\n2. B"
    styles = classify_request_table(requests)
    random_sample, challenge = build_blinded_annotation_samples(
        requests,
        styles,
        random_size=40,
        challenge_size=10,
    )
    forbidden = {"N", "exact_count", "signed_deviation", "model_label"}
    assert forbidden.isdisjoint(random_sample.columns)
    assert forbidden.isdisjoint(challenge.columns)
    assert len(random_sample) == 40
    assert len(challenge) == 10

    annotated = random_sample.copy()
    automated = styles.loc[styles["request_id"].isin(annotated["request_id"])]
    automated_lookup = automated.set_index("request_id")
    annotated["human_dominant_style"] = annotated["request_id"].map(
        automated_lookup["dominant_style"]
    )
    for style in (
        "index_enumeration",
        "bullet_enumeration",
        "word_enumeration",
        "running_tally",
        "arithmetic_grouping",
        "scan_or_retrieval_summary",
    ):
        annotated[f"human_{style}"] = annotated["request_id"].map(
            automated_lookup[style]
        )
    validation, per_label, confusion = evaluate_style_annotations(
        annotated,
        automated,
    )
    assert validation["confirmatory_automated_reporting_allowed"]
    assert len(per_label) == 6
    assert not confusion.empty
