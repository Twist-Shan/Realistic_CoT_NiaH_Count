from __future__ import annotations

import math

import pandas as pd

from realistic_niah_v4.correct_interventions import (
    eligible_directed_pairs,
    select_sequential_supplement,
    summarize_ablation_n_diagnostics,
    summarize_ablation_population,
    summarize_average_patching_accuracy,
)


def _label(
    model: str,
    seed: int,
    count: int,
    *,
    correct: bool,
) -> dict[str, object]:
    return {
        "stimulus_id": f"{model}-s{seed}-n{count}",
        "model_label": model,
        "seed": seed,
        "gold_count": count,
        "outcome_group": "correct" if correct else "wrong",
        "format_valid": True,
    }


def _pair_rows(model: str, k: int, seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for receiver, donor, direction in (
            (0, k, "increase"),
            (k, 0, "decrease"),
        ):
            rows.append(
                {
                    "model_label": model,
                    "seed": seed,
                    "receiver_count": receiver,
                    "donor_count": donor,
                    "receiver_stimulus_id": f"old-s{seed}-n{receiver}",
                    "donor_stimulus_id": f"old-s{seed}-n{donor}",
                    "k": k,
                    "target_direction": direction,
                }
            )
    return rows


def test_sequential_supplement_records_shortage_and_added_seeds() -> None:
    model = "Qwen3-8B"
    existing_pairs = pd.DataFrame(
        _pair_rows(model, 1, [1, 2, 3, 4, 5])
        + _pair_rows(model, 3, [1, 2, 3, 4])
        + _pair_rows(model, 5, [1, 2])
    )
    existing_ablation = pd.DataFrame(
        [_label(model, seed, 7, correct=True) for seed in (10, 11, 12)]
    )
    correctness = {
        20: {0, 3, 5, 7},
        21: {0, 5, 8},
        22: {0, 5, 10},
        23: {0, 5, 9},
    }
    candidate = pd.DataFrame(
        [
            _label(model, seed, count, correct=count in correctness[seed])
            for seed in correctness
            for count in range(11)
        ]
    )

    manifest, fresh_pairs, fresh_ablation = select_sequential_supplement(
        existing_pairs=existing_pairs,
        existing_ablation_baselines=existing_ablation,
        candidate_baselines=candidate,
        reserve_seeds=(20, 21, 22, 23),
        patch_cluster_target=5,
        ablation_cluster_target=5,
        ablation_counts=(7, 8, 9, 10),
    )

    assert manifest["selection_status"] == "complete"
    assert manifest["scanned_supplement_seeds"] == [20, 21, 22]
    assert manifest["unused_reserve_seeds"] == [23]
    initial = {
        (row["k"], row["target_direction"]): row["missing_seed_clusters"]
        for row in manifest["patching"]["initial_support"]
    }
    assert initial[(1, "increase")] == 0
    assert initial[(3, "increase")] == 1
    assert initial[(5, "increase")] == 3
    assert manifest["correct_only_ablation"]["initial_missing_seed_clusters"] == 2
    assert set(fresh_pairs["seed"]) == {20, 21, 22}
    assert set(fresh_ablation["seed"]) == {20, 21}
    assert manifest["correct_only_ablation"]["unselected_fresh_eligible_seeds"] == [22]
    assert manifest["correct_only_ablation"]["shared_discovery_seed_prefix"] == [20, 21]
    assert (
        manifest["correct_only_ablation"]
        ["shared_discovery_expected_all_example_stimuli"]
        == 8
    )


def test_eligible_pairs_require_both_clean_answers() -> None:
    labels = pd.DataFrame(
        [
            _label("Gemma4-E4B", 30, count, correct=count in {0, 1, 3})
            for count in range(11)
        ]
    )
    pairs = eligible_directed_pairs(labels)
    assert set(zip(pairs["receiver_count"], pairs["donor_count"])) == {
        (0, 1),
        (1, 0),
        (0, 3),
        (3, 0),
    }


def test_shared_ablation_prefix_keeps_incorrect_seed_clusters() -> None:
    model = "Gemma4-E4B"
    existing_pairs = pd.DataFrame(
        _pair_rows(model, 1, [1, 2, 3, 4, 5])
        + _pair_rows(model, 3, [1, 2, 3, 4, 5])
        + _pair_rows(model, 5, [1, 2, 3, 4, 5])
    )
    existing_ablation = pd.DataFrame([_label(model, 10, 1, correct=False)])
    correct = {20: set(), 21: {1}, 22: {2}}
    candidate = pd.DataFrame(
        [
            _label(model, seed, count, correct=count in correct[seed])
            for seed in correct
            for count in range(11)
        ]
    )

    manifest, _fresh_pairs, fresh_ablation = select_sequential_supplement(
        existing_pairs=existing_pairs,
        existing_ablation_baselines=existing_ablation,
        candidate_baselines=candidate,
        reserve_seeds=(20, 21, 22),
        patch_cluster_target=5,
        ablation_cluster_target=2,
    )

    assert set(fresh_ablation["seed"]) == {21, 22}
    assert manifest["correct_only_ablation"]["counts"] == [1, 2, 3, 4, 5]
    assert manifest["correct_only_ablation"]["shared_discovery_seed_prefix"] == [
        20,
        21,
        22,
    ]
    assert (
        manifest["correct_only_ablation"]
        ["shared_discovery_expected_all_example_stimuli"]
        == 15
    )


def test_average_patching_accuracy_is_donor_target_hit() -> None:
    detail = pd.DataFrame(
        [
            {
                "model_label": "Qwen3-8B",
                "family": "answer_patching",
                "site": "answer_query",
                "patch_protocol": "single_layer",
                "start_layer": 4,
                "k": 3,
                "target_direction": "increase",
                "condition": "donor_transport",
                "seed": seed,
                "receiver_count": 0,
                "donor_count": 3,
                "strict_target_hit": hit,
                "patched_is_correct": not hit,
                "patched_format_valid": True,
                "generated_count_shift": 3 if hit else 1,
                "status": "ok",
            }
            for seed, hit in ((1, True), (2, False), (3, True))
        ]
    )
    summary = summarize_average_patching_accuracy(
        detail,
        group_columns=(
            "model_label",
            "family",
            "site",
            "patch_protocol",
            "start_layer",
            "k",
            "target_direction",
        ),
        bootstrap_repetitions=200,
    )
    row = summary.iloc[0]
    assert math.isclose(row["average_patching_acc"], 2 / 3)
    assert row["patching_acc_successes"] == 2
    assert row["patching_acc_denominator"] == 3
    assert row["average_post_patch_receiver_acc"] == 1 / 3


def test_average_patching_accuracy_point_estimate_is_pair_weighted() -> None:
    detail = pd.DataFrame(
        [
            {
                "model_label": "Qwen3-8B",
                "k": 3,
                "target_direction": "increase",
                "condition": "donor_transport",
                "seed": seed,
                "receiver_count": receiver,
                "donor_count": receiver + 3,
                "strict_target_hit": hit,
                "patched_is_correct": not hit,
                "patched_format_valid": True,
                "generated_count_shift": 3 if hit else 0,
                "status": "ok",
            }
            for seed, receiver, hit in (
                (1, 0, True),
                (1, 1, True),
                (1, 2, True),
                (2, 0, False),
            )
        ]
    )
    summary = summarize_average_patching_accuracy(
        detail,
        group_columns=("model_label", "k", "target_direction"),
        bootstrap_repetitions=200,
    ).iloc[0]
    assert summary["seed_clusters"] == 2
    assert summary["patching_acc_successes"] == 3
    assert summary["patching_acc_denominator"] == 4
    assert summary["average_patching_acc"] == 3 / 4


def test_ablation_populations_separate_signed_shift_and_failure_rate() -> None:
    rows: list[dict[str, object]] = []
    for seed, baseline_correct, shift, patched_correct in (
        (1, True, -2, False),
        (2, False, 1, False),
    ):
        base = {
            "model_label": "Gemma4-E4B",
            "stimulus_id": f"s{seed}",
            "seed": seed,
            "gold_count": 8,
            "head_bank": "broad_aggregation",
            "top_n": 6,
            "baseline_is_correct": baseline_correct,
            "baseline_format_valid": True,
        }
        rows.append(
            {
                **base,
                "condition": "ranked",
                "patched_is_correct": patched_correct,
                "patched_format_valid": True,
                "accuracy_delta": int(patched_correct) - int(baseline_correct),
                "absolute_error_delta": 2,
                "generated_count_shift": shift,
                "prediction_changed": True,
            }
        )
        for replicate in range(3):
            rows.append(
                {
                    **base,
                    "condition": "layer_matched_random",
                    "random_replicate": replicate,
                    "patched_is_correct": baseline_correct,
                    "patched_format_valid": True,
                    "accuracy_delta": 0,
                    "absolute_error_delta": 0,
                    "generated_count_shift": 0,
                    "prediction_changed": False,
                }
            )
    detail = pd.DataFrame(rows)
    overall = summarize_ablation_population(
        detail, population="all_examples_signed", bootstrap_repetitions=200
    ).iloc[0]
    correct = summarize_ablation_population(
        detail, population="clean_correct_only", bootstrap_repetitions=200
    ).iloc[0]
    assert overall["examples"] == 2
    assert overall["mean_signed_count_shift_valid"] == -0.5
    assert math.isnan(overall["correct_to_wrong_rate"])
    assert correct["examples"] == 1
    assert correct["mean_signed_count_shift_valid"] == -2
    assert correct["correct_to_wrong_rate"] == 1


def test_ablation_n_diagnostics_use_population_specific_primary_endpoint() -> None:
    rows: list[dict[str, object]] = []
    for seed in (1, 2):
        for top_n, ranked_shift, ranked_correct in (
            (1, -1, True),
            (2, -3, False),
        ):
            base = {
                "model_label": "Qwen3-8B",
                "stimulus_id": f"s{seed}",
                "seed": seed,
                "gold_count": 8,
                "head_bank": "broad_aggregation",
                "top_n": top_n,
                "baseline_is_correct": True,
                "baseline_format_valid": True,
            }
            rows.append(
                {
                    **base,
                    "condition": "ranked",
                    "patched_is_correct": ranked_correct,
                    "patched_format_valid": True,
                    "accuracy_delta": int(ranked_correct) - 1,
                    "absolute_error_delta": abs(ranked_shift),
                    "generated_count_shift": ranked_shift,
                    "prediction_changed": ranked_shift != 0,
                    "ranked_random_head_overlap": top_n,
                }
            )
            for replicate in range(3):
                rows.append(
                    {
                        **base,
                        "condition": "layer_matched_random",
                        "random_replicate": replicate,
                        "patched_is_correct": True,
                        "patched_format_valid": True,
                        "accuracy_delta": 0,
                        "absolute_error_delta": 0,
                        "generated_count_shift": 0,
                        "prediction_changed": False,
                        "ranked_random_head_overlap": 0,
                    }
                )
    detail = pd.DataFrame(rows)
    overall = summarize_ablation_n_diagnostics(
        detail,
        population="all_examples_signed",
        bootstrap_repetitions=200,
    )
    correct = summarize_ablation_n_diagnostics(
        detail,
        population="clean_correct_only",
        bootstrap_repetitions=200,
    )
    assert set(overall["top_n"]) == {1, 2}
    assert set(overall["selection_status"]) == {"discovery_only_unfrozen"}
    assert overall.loc[overall["top_n"].eq(2), "primary_effect"].iloc[0] == 3
    assert (
        overall.loc[overall["top_n"].eq(2), "primary_metric"].iloc[0]
        == "ranked_minus_random_absolute_count_shift"
    )
    assert correct.loc[correct["top_n"].eq(2), "primary_effect"].iloc[0] == 1
    assert (
        correct.loc[correct["top_n"].eq(2), "primary_metric"].iloc[0]
        == "ranked_minus_random_correct_to_wrong"
    )
    assert (
        correct.loc[correct["top_n"].eq(2), "primary_rank_within_model_bank"].iloc[0]
        == 1
    )
