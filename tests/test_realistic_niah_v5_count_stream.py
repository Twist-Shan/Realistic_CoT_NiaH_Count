from __future__ import annotations

import json
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_realistic_niah_v5_count_stream import (
    _cohort_exclusion_reason,
    _validate_trace_patch_basis_manifest,
    build_parser,
)
from realistic_niah_v5.count_stream import (
    AnswerSourceRegistry,
    NativeCountMechanismSpec,
    _registered_ordinary_corruption_banks,
    answer_source_mask,
    build_answer_broad_head_plan,
    build_sparse_trace_patch_sample_plan,
    build_trace_patch_pair_plan,
    deterministic_control_basis,
    fit_count_stream_basis,
    mechanism_decision_ledger,
    rank_answer_broad_heads,
    select_answer_broad_bank_size,
    source_attention_metrics,
    stream_state_retention_metrics,
    summarize_linear_contrasts,
    trace_patch_condition_states,
    valid_trace_patch_receivers,
)


def _registry() -> AnswerSourceRegistry:
    value = AnswerSourceRegistry(
        request_id="request-1",
        answer_site_id="answer_query_v3",
        sequence_length=40,
        prompt_token_count=25,
        query_position=39,
        prompt_records=((2, 5), (7, 9)),
        trace_context=((25, 39),),
        trace_items=((26, 30), (31, 35)),
        trace_other=((25, 26), (30, 31), (35, 39)),
        trace_markers=((26, 27), (31, 32)),
        trace_nonmarkers=((27, 30), (32, 35)),
        earlier_trace_items=((26, 30),),
        terminal_trace_item=((31, 35),),
    )
    value.validate()
    return value


def test_source_registry_rejects_nonpartitioned_markers() -> None:
    value = _registry()
    invalid = AnswerSourceRegistry(
        **{
            **value.__dict__,
            "trace_nonmarkers": ((17, 20), (23, 25)),
        }
    )
    with pytest.raises(ValueError, match="do not cover"):
        invalid.validate()


def test_answer_source_masks_have_exact_registered_budgets() -> None:
    registry = _registry()
    clean, clean_audit = answer_source_mask(registry, condition="clean")
    trace, trace_audit = answer_source_mask(registry, condition="block_trace_items")
    trace_context, trace_context_audit = answer_source_mask(
        registry, condition="block_trace_context"
    )
    prompt, prompt_audit = answer_source_mask(
        registry, condition="block_prompt_records"
    )
    both, both_audit = answer_source_mask(registry, condition="block_trace_and_prompt")
    trace_control, trace_control_audit = answer_source_mask(
        registry, condition="block_trace_items_matched_control"
    )
    context_control, context_control_audit = answer_source_mask(
        registry, condition="block_trace_context_matched_control"
    )
    marker_control, marker_control_audit = answer_source_mask(
        registry, condition="block_trace_markers_matched_control"
    )
    assert int(clean.sum()) == 40
    assert clean_audit["blocked_token_count"] == 0
    assert trace_audit["blocked_token_count"] == 8
    assert trace_context_audit["blocked_token_count"] == 14
    assert prompt_audit["blocked_token_count"] == 5
    assert both_audit["blocked_token_count"] == 19
    assert trace_control_audit["blocked_token_count"] == 8
    assert marker_control_audit["blocked_token_count"] == 2
    assert context_control_audit["blocked_token_count"] == 14
    assert int(trace_control.sum()) == int(trace.sum())
    assert (
        trace_control_audit["blocked_positions_sha256"]
        != trace_audit["blocked_positions_sha256"]
    )
    assert int(marker_control[:, registry.query_position]) == 1
    assert int(prompt[:, registry.query_position]) == 1
    assert int(both[:, registry.query_position]) == 1


def test_marker_mask_is_explicitly_not_applicable_for_unmarked_trace() -> None:
    value = _registry()
    unmarked = AnswerSourceRegistry(
        **{
            **value.__dict__,
            "trace_markers": (),
            "trace_nonmarkers": value.trace_items,
        }
    )
    unmarked.validate()
    with pytest.raises(ValueError, match="not applicable"):
        answer_source_mask(unmarked, condition="block_trace_markers")


def test_ordinary_corruption_banks_exclude_active_prompt_records() -> None:
    registry = AnswerSourceRegistry(
        request_id="request-banks",
        answer_site_id="answer_query_v3",
        sequence_length=70,
        prompt_token_count=50,
        query_position=69,
        prompt_records=((10, 14), (20, 24)),
        trace_context=((50, 69),),
        trace_items=((52, 56), (58, 62)),
        trace_other=((50, 52), (56, 58), (62, 69)),
        trace_markers=((52, 53), (58, 59)),
        trace_nonmarkers=((53, 56), (59, 62)),
        earlier_trace_items=((52, 56),),
        terminal_trace_item=((58, 62),),
    )
    registry.validate()
    banks = _registered_ordinary_corruption_banks(registry)
    prompt_records = set(registry.positions("prompt_records"))
    flattened = [
        {position for start, end in bank for position in range(start, end)}
        for bank in banks
    ]
    assert all(len(values) == 8 for values in flattened)
    assert all(not values & prompt_records for values in flattened)
    assert not flattened[0] & flattened[1]
    assert not flattened[0] & flattened[2]
    assert not flattened[1] & flattened[2]


def test_source_attention_metrics_separate_mass_and_breadth() -> None:
    attention = torch.tensor(
        [
            [0.2, 0.2, 0.1, 0.1, 0.2, 0.2],
            [0.4, 0.4, 0.1, 0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    metrics = source_attention_metrics(
        attention,
        key_start=0,
        spans=((0, 2), (4, 6)),
    )
    assert metrics[0]["mass"] == pytest.approx(0.8)
    assert metrics[0]["coverage"] == pytest.approx(1.0)
    assert metrics[0]["broad_score"] == pytest.approx(0.8)
    assert metrics[1]["mass"] == pytest.approx(0.8)
    assert metrics[1]["coverage"] == pytest.approx(0.5)
    assert metrics[1]["broad_score"] == pytest.approx(0.4)


def _capture_frame() -> pd.DataFrame:
    rows = []
    for seed in (1, 2):
        for request in ("a", "b"):
            for layer in (0, 1):
                for head in range(4):
                    score = 0.01
                    if (layer, head) == (0, 0):
                        score = 1.0
                    elif (layer, head) == (1, 0):
                        score = 0.9
                    rows.append(
                        {
                            "request_id": f"{seed}-{request}",
                            "model_label": "Qwen3-8B",
                            "seed": seed,
                            "layer": layer,
                            "head": head,
                            "layer_head_count": 4,
                            "trace_items_broad_score": score,
                        }
                    )
    return pd.DataFrame(rows)


def test_broad_head_plan_freezes_treatment_before_matched_controls() -> None:
    ranking = rank_answer_broad_heads(
        _capture_frame(),
        source_group="trace_items",
        development_seeds=(1, 2),
        model_label="Qwen3-8B",
    )
    assert list(
        ranking.head(2)[["layer", "head"]].itertuples(index=False, name=None)
    ) == [(0, 0), (1, 0)]
    plan = build_answer_broad_head_plan(
        ranking,
        bank_size=2,
        random_controls=3,
        random_seed=17,
    )
    selected = json.loads(
        plan.loc[plan["condition"].eq("selected_bank"), "heads"].iloc[0]
    )
    assert selected == [[0, 0], [1, 0]]
    random_rows = plan.loc[plan["condition"].eq("layer_matched_random")]
    assert len(random_rows) == 3
    assert random_rows["bank_sha256"].nunique() == 3
    for raw in random_rows["heads"]:
        control = json.loads(raw)
        assert len(control) == 2
        assert sorted(layer for layer, _head in control) == [0, 1]
        assert not {tuple(value) for value in control} & {(0, 0), (1, 0)}


def test_control_basis_is_orthonormal_and_deterministic() -> None:
    rng = np.random.default_rng(4)
    basis, _ = np.linalg.qr(rng.standard_normal((12, 3)))
    left = deterministic_control_basis(basis, seed=9)
    right = deterministic_control_basis(basis, seed=9)
    np.testing.assert_allclose(left, right)
    np.testing.assert_allclose(left.T @ left, np.eye(3), atol=2e-5)
    np.testing.assert_allclose(basis.T @ left, np.zeros((3, 3)), atol=2e-5)


def test_count_stream_basis_uses_label_centroids() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(1, 5), 6)
    states = rng.normal(scale=0.05, size=(len(labels), 10))
    states[:, 0] += labels
    center, basis, control = fit_count_stream_basis(
        states.astype(np.float32), labels, rank=2, seed=3
    )
    assert center.shape == (10,)
    assert basis.shape == (10, 2)
    assert control.shape == (10, 2)
    np.testing.assert_allclose(basis.T @ control, 0.0, atol=2e-5)


def test_trace_patch_pair_plan_is_directed_and_strictly_intermediate() -> None:
    specifications = [
        {
            "from_occurrence": occurrence,
            "anchor_equivalence_id": f"item-end-{occurrence}",
            "target_city": f"city-{occurrence + 1}",
        }
        for occurrence in range(1, 5)
    ]
    plan = build_trace_patch_pair_plan(specifications, gold_count=5)
    assert [
        (value["donor_occurrence"], value["receiver_occurrence"]) for value in plan
    ] == [(1, 2), (3, 2), (2, 3), (4, 3), (3, 4)]
    assert {value["donor_direction"] for value in plan} == {
        "past_to_later_receiver",
        "future_to_earlier_receiver",
    }
    assert all(1 < value["receiver_occurrence"] < 5 for value in plan)
    with pytest.raises(ValueError, match="strictly intermediate"):
        build_trace_patch_pair_plan(
            specifications,
            gold_count=5,
            receiver_occurrences=(1,),
        )


def test_sparse_trace_patch_plan_freezes_330_local_plus_20_terminal_pairs() -> None:
    rows = [
        {
            "request_id": f"request-{seed}-{count}",
            "model_label": "Qwen3-8B",
            "seed": seed,
            "gold_count": count,
            # Deliberately different outcomes: sampling must not inspect them.
            "prediction": (seed + count) % 10,
            "exact_count": bool((seed + count) % 2),
        }
        for seed in range(100, 130)
        for count in range(1, 11)
    ]
    plan = build_sparse_trace_patch_sample_plan(
        rows,
        model_label="Qwen3-8B",
        seeds_per_cell=10,
        sampling_seed=17,
    )
    repeated = build_sparse_trace_patch_sample_plan(
        [{**row, "prediction": -999, "exact_count": False} for row in rows],
        model_label="Qwen3-8B",
        seeds_per_cell=10,
        sampling_seed=17,
    )
    assert list(plan["pair_sha256"]) == list(repeated["pair_sha256"])
    assert len(plan) == 350
    assert int(plan["panel_kind"].eq("local").sum()) == 330
    assert int(plan["panel_kind"].eq("terminal").sum()) == 20
    cell_sizes = plan.groupby(
        ["panel_kind", "gold_count", "donor_offset"]
    ).size()
    assert len(cell_sizes) == 35
    assert set(cell_sizes) == {10}
    assert plan.groupby(
        ["panel_kind", "gold_count", "donor_offset"]
    )["seed"].nunique().eq(10).all()
    expected_min_count = {-5: 7, -3: 5, -1: 3, 1: 4, 3: 6, 5: 8}
    local = plan.loc[plan["panel_kind"].eq("local")]
    for offset, minimum in expected_min_count.items():
        assert sorted(local.loc[local["donor_offset"].eq(offset), "gold_count"].unique()) == list(
            range(minimum, 11)
        )
    for _cell, frame in local.groupby(
        ["gold_count", "donor_offset"], sort=True
    ):
        frequencies = frame["receiver_occurrence"].value_counts()
        assert int(frequencies.max() - frequencies.min()) <= 1
    terminal = plan.loc[plan["panel_kind"].eq("terminal")]
    assert set(
        terminal[["donor_occurrence", "receiver_occurrence"]].itertuples(
            index=False, name=None
        )
    ) == {(1, 2), (2, 1)}
    assert not terminal["local_next_city_outcome_registered"].any()


def test_valid_trace_patch_receiver_ranges_are_signed() -> None:
    assert valid_trace_patch_receivers(3, -1) == (2,)
    assert valid_trace_patch_receivers(3, 1) == ()
    assert valid_trace_patch_receivers(5, -3) == (4,)
    assert valid_trace_patch_receivers(5, 3) == ()
    assert valid_trace_patch_receivers(8, 5) == (2,)


def test_trace_patch_conditions_isolate_progress_and_match_control_norm() -> None:
    receiver = np.zeros(4, dtype=np.float32)
    donor = np.asarray([2.0, 3.0, 0.0, 0.0], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    states, audit = trace_patch_condition_states(
        receiver,
        donor,
        basis,
        random_seed=19,
    )
    np.testing.assert_allclose(states["self_patch"], receiver)
    np.testing.assert_allclose(states["full_donor_patch"], donor)
    np.testing.assert_allclose(
        states["progress_projected_patch"],
        np.asarray([2.0, 0.0, 0.0, 0.0]),
    )
    orthogonal = np.asarray(states["norm_matched_orthogonal_patch"])
    assert orthogonal[0] == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(orthogonal) == pytest.approx(2.0, rel=1e-5)
    assert audit["progress_projected_delta_norm"] == pytest.approx(2.0)
    assert audit["orthogonal_control_delta_norm"] == pytest.approx(2.0)


def test_stream_retention_metric_uses_later_item_bases_not_answer_query() -> None:
    clean = np.zeros((2, 3, 4), dtype=np.float32)
    aligned = clean.copy()
    aligned[0, :2, 0] = 2.0
    aligned[1, :2, 1] = 1.0
    aligned[:, 2, 3] = 5.0
    orthogonal = clean.copy()
    orthogonal[0, :2, 2] = 2.0
    orthogonal[1, :2, 3] = 1.0
    bases = {
        4: np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32),
        7: np.asarray([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32),
    }
    aligned_metrics = stream_state_retention_metrics(
        clean,
        aligned,
        readout_layers=(4, 7),
        readout_positions=(5, 8, 10),
        query_position=10,
        count_bases=bases,
    )
    orthogonal_metrics = stream_state_retention_metrics(
        clean,
        orthogonal,
        readout_layers=(4, 7),
        readout_positions=(5, 8, 10),
        query_position=10,
        count_bases=bases,
    )
    assert aligned_metrics["downstream_item_readout_count"] == 4
    assert aligned_metrics[
        "downstream_item_progress_subspace_displacement_rms"
    ] == pytest.approx(np.sqrt(2.5))
    assert orthogonal_metrics[
        "downstream_item_progress_subspace_displacement_rms"
    ] == pytest.approx(0.0)
    assert aligned_metrics["answer_query_projected_into_item_basis"] is False
    assert aligned_metrics["answer_query_full_state_displacement_rms"] == 5.0


def test_mechanism_spec_does_not_relabel_seen_seeds_as_confirmation() -> None:
    spec = NativeCountMechanismSpec()
    spec.validate()
    assert spec.seed_role(1258) == "development"
    assert not spec.formal_inference_eligible
    assert NativeCountMechanismSpec.from_mapping(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="disjoint"):
        NativeCountMechanismSpec(
            status="frozen_confirmation",
            development_seeds=(1, 2),
            confirmation_seeds=(2, 3),
        ).validate()


def test_broad_discovery_roles_and_odd_even_panels_are_disjoint() -> None:
    spec = NativeCountMechanismSpec()
    spec.validate()
    assert spec.broad_phase(1234) == "ranking_discovery"
    assert spec.broad_phase(1244) == "k_selection_discovery"
    assert spec.broad_counts_for_seed(1234, phase="ranking_discovery") == tuple(
        range(1, 11)
    )
    assert spec.broad_counts_for_seed(1244, phase="k_selection_discovery") == (
        1,
        3,
        5,
        7,
        9,
    )
    assert spec.broad_counts_for_seed(1263, phase="k_selection_discovery") == (
        2,
        4,
        6,
        8,
        10,
    )


def _broad_k_trials(selected_shifts: dict[int, float]) -> pd.DataFrame:
    rows = []
    for bank_size, shift in selected_shifts.items():
        for seed in (1, 2, 3, 4):
            for request_index in (0, 1):
                request_id = f"{seed}-{request_index}"
                common = {
                    "experiment_id": "answer_broad_head_ablation",
                    "model_label": "Qwen3-8B",
                    "source_group": "trace_items",
                    "bank_size": bank_size,
                    "request_id": request_id,
                    "seed": seed,
                    "status": "ok",
                }
                rows.append(
                    {
                        **common,
                        "condition": "clean",
                        "repeat": 0,
                        "expected_count": 5.0,
                        "correct_count_margin": 2.0,
                    }
                )
                rows.append(
                    {
                        **common,
                        "condition": "selected_bank",
                        "repeat": 0,
                        "expected_count": 5.0 + shift,
                        "correct_count_margin": 2.0 - shift,
                    }
                )
                for repeat in (1, 2, 3):
                    rows.append(
                        {
                            **common,
                            "condition": "layer_matched_random",
                            "repeat": repeat,
                            "expected_count": 5.1,
                            "correct_count_margin": 1.9,
                        }
                    )
    return pd.DataFrame(rows)


def test_broad_k_selection_uses_smallest_one_se_positive_bank() -> None:
    curve, seed_effects, decision = select_answer_broad_bank_size(
        _broad_k_trials({1: 0.2, 2: 1.0, 4: 0.8}),
        model_label="Qwen3-8B",
        source_group="trace_items",
        expected_seeds=(1, 2, 3, 4),
        expected_bank_sizes=(1, 2, 4),
        expected_requests_per_seed=2,
        expected_random_controls=3,
        boundary_extension_bank_size=8,
        bootstrap_samples=100,
        random_seed=3,
    )
    assert list(curve["bank_size"]) == [1, 2, 4]
    assert len(seed_effects) == 12
    assert decision["status"] == "frozen_for_confirmation"
    assert decision["selected_bank_size"] == 2
    assert decision["confirmation_outcomes_used"] is False


def test_broad_k_selection_requires_boundary_extension_when_curve_is_rising() -> None:
    _curve, _seed_effects, decision = select_answer_broad_bank_size(
        _broad_k_trials({1: 0.2, 2: 0.6, 4: 1.1}),
        model_label="Qwen3-8B",
        source_group="trace_items",
        expected_seeds=(1, 2, 3, 4),
        expected_bank_sizes=(1, 2, 4),
        expected_requests_per_seed=2,
        expected_random_controls=3,
        boundary_extension_bank_size=8,
        bootstrap_samples=100,
        random_seed=4,
    )
    assert decision["status"] == "requires_boundary_extension"
    assert decision["selected_bank_size"] is None
    assert decision["required_next_bank_size"] == 8


def test_native_cohort_filter_keeps_correctness_out_of_primary_selection() -> None:
    row = {
        "trace_parse": {
            "exact_count": False,
            "parser": {"detected": True, "trace_one_to_one": True},
        }
    }
    assert _cohort_exclusion_reason(row, "parser_hit") is None
    assert _cohort_exclusion_reason(row, "one_to_one") is None
    assert (
        _cohort_exclusion_reason(row, "one_to_one_correct") == "final_count_incorrect"
    )


def test_trace_patch_cli_registers_full_control_panel() -> None:
    args = build_parser().parse_args(
        [
            "trace-patch",
            "--model",
            "Qwen3-8B",
            "--generations",
            "generations.jsonl",
            "--pair-plan",
            "pair-plan.csv",
            "--basis",
            "basis.npz",
            "--layer",
            "20",
            "--output",
            "trace-patch-output",
        ]
    )
    assert args.command == "trace-patch"
    assert args.max_new_tokens == 48
    assert args.conditions == [
        "clean",
        "self_patch",
        "full_donor_patch",
        "progress_projected_patch",
        "norm_matched_orthogonal_patch",
    ]


def test_trace_patch_basis_manifest_freezes_site_and_label(tmp_path) -> None:
    artifact = tmp_path / "basis.npz"
    np.savez_compressed(artifact, basis_L1=np.eye(2, dtype=np.float32))
    manifest = artifact.with_suffix(".json")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = {
        "artifact_sha256": digest,
        "site_kind": "item_end",
        "label": "occurrence",
        "confirmation_used_for_fit": False,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path, loaded = _validate_trace_patch_basis_manifest(artifact)
    assert manifest_path == manifest
    assert loaded["label"] == "occurrence"

    payload["label"] = "gold_count"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="occurrence label"):
        _validate_trace_patch_basis_manifest(artifact)


def test_registered_contrasts_are_request_first_and_seed_equal() -> None:
    rows = []
    values = {
        1: {
            "clean": 5.0,
            "block_trace_context": 2.0,
            "block_trace_context_matched_control": 4.0,
        },
        2: {
            "clean": 7.0,
            "block_trace_context": 6.0,
            "block_trace_context_matched_control": 7.0,
        },
    }
    for seed, conditions in values.items():
        for condition, value in conditions.items():
            rows.append(
                {
                    "experiment_id": "answer_source_mask_factorial",
                    "condition": condition,
                    "model_label": "Qwen3-8B",
                    "request_id": f"r{seed}",
                    "seed": seed,
                    "correct_count_margin": value,
                }
            )
    summary, seed_effects = summarize_linear_contrasts(
        pd.DataFrame(rows),
        experiment_id="answer_source_mask_factorial",
        outcome="correct_count_margin",
        bootstrap_samples=200,
    )
    trace = summary.loc[summary["contrast"].eq("trace_damage")].iloc[0]
    specificity = summary.loc[summary["contrast"].eq("trace_source_specificity")].iloc[
        0
    ]
    assert trace["mean_effect"] == pytest.approx(2.0)
    assert specificity["mean_effect"] == pytest.approx(1.5)
    assert np.isfinite(trace["p_value_holm_across_models"])
    assert len(seed_effects.loc[seed_effects["contrast"].eq("trace_damage")]) == 2


def test_restoration_specificity_is_a_difference_of_repairs() -> None:
    rows = []
    values = {
        "trace_token_corrupt": 0.0,
        "trace_corrupt_full_span_restore": 3.0,
        "ordinary_token_corrupt": 1.0,
        "ordinary_corrupt_ordinary_state_restore": 2.0,
    }
    for seed in (1, 2):
        for condition, value in values.items():
            rows.append(
                {
                    "experiment_id": "trace_source_restoration",
                    "condition": condition,
                    "model_label": "Qwen3-8B",
                    "request_id": f"restoration-{seed}",
                    "seed": seed,
                    "correct_count_margin": value,
                }
            )
    summary, _seed_effects = summarize_linear_contrasts(
        pd.DataFrame(rows),
        experiment_id="trace_source_restoration",
        outcome="correct_count_margin",
        bootstrap_samples=100,
    )
    specificity = summary.loc[
        summary["contrast"].eq("full_span_vs_ordinary_repair_specificity")
    ].iloc[0]
    assert specificity["mean_effect"] == pytest.approx(2.0)


def test_decision_ledger_requires_every_component_gate() -> None:
    summary = pd.DataFrame(
        [
            {
                "experiment_id": "stream_state_retention",
                "contrast": "aligned_vs_orthogonal_specificity",
                "model_label": "Qwen3-8B",
                "mean_effect": 1.0,
                "ci_low": 0.2,
            }
        ]
    )
    ledger = mechanism_decision_ledger(summary)
    stream = ledger.loc[ledger["claim"].eq("stream_written_state")].iloc[0]
    assert stream["status"] == "not_established"
    assert "full_donor_vs_self_transport" in stream["missing_gates"]


def test_decision_ledger_requires_forward_trace_transport() -> None:
    rows = []
    for contrast in (
        "full_donor_vs_self_transport",
        "projected_vs_orthogonal_transport",
    ):
        rows.append(
            {
                "experiment_id": "trace_intermediate_state_patching",
                "contrast": contrast,
                "donor_direction": "past_to_later_receiver",
                "model_label": "Qwen3-8B",
                "mean_effect": 1.0,
                "ci_low": 0.2,
            }
        )
    ledger = mechanism_decision_ledger(pd.DataFrame(rows))
    stream = ledger.loc[ledger["claim"].eq("stream_written_state")].iloc[0]
    assert stream["status"] == "passes_registered_gate"
    assert "past_to_later_receiver" in stream["observed_gates"]

    future = pd.DataFrame(rows).assign(donor_direction="future_to_earlier_receiver")
    future_ledger = mechanism_decision_ledger(future)
    future_stream = future_ledger.loc[
        future_ledger["claim"].eq("stream_written_state")
    ].iloc[0]
    assert future_stream["status"] == "not_established"


def test_decision_ledger_keeps_trace_and_prompt_broad_banks_separate() -> None:
    rows = [
        {
            "experiment_id": "answer_source_mask_factorial",
            "contrast": "trace_source_specificity",
            "model_label": "Qwen3-8B",
            "mean_effect": 1.0,
            "ci_low": 0.2,
        },
        {
            "experiment_id": "answer_source_mask_factorial",
            "contrast": "prompt_source_specificity",
            "model_label": "Qwen3-8B",
            "mean_effect": 1.0,
            "ci_low": 0.2,
        },
        {
            "experiment_id": "answer_broad_head_ablation",
            "contrast": "selected_vs_layer_matched_random",
            "source_group": "trace_items",
            "model_label": "Qwen3-8B",
            "mean_effect": 0.8,
            "ci_low": 0.1,
        },
        {
            "experiment_id": "answer_broad_head_ablation",
            "contrast": "selected_vs_layer_matched_random",
            "source_group": "prompt_records",
            "model_label": "Qwen3-8B",
            "mean_effect": -0.4,
            "ci_low": -0.9,
        },
    ]
    ledger = mechanism_decision_ledger(pd.DataFrame(rows))
    trace = ledger.loc[ledger["claim"].eq("answer_time_trace_retrieval")].iloc[0]
    prompt = ledger.loc[ledger["claim"].eq("answer_time_prompt_retrieval")].iloc[0]
    assert trace["status"] == "passes_registered_gate"
    assert prompt["status"] == "not_established"
    assert "trace_items" in trace["observed_gates"]
    assert "prompt_records" not in trace["observed_gates"]


def test_decision_ledger_rejects_ambiguous_duplicate_gate_rows() -> None:
    rows = [
        {
            "experiment_id": "answer_source_mask_factorial",
            "contrast": "trace_source_specificity",
            "model_label": "Qwen3-8B",
            "mean_effect": 1.0,
            "ci_low": 0.2,
        },
    ]
    for bank_size in (8, 16):
        rows.append(
            {
                "experiment_id": "answer_broad_head_ablation",
                "contrast": "selected_vs_layer_matched_random",
                "source_group": "trace_items",
                "bank_size": bank_size,
                "model_label": "Qwen3-8B",
                "mean_effect": 0.8,
                "ci_low": 0.1,
            }
        )
    ledger = mechanism_decision_ledger(pd.DataFrame(rows))
    trace = ledger.loc[ledger["claim"].eq("answer_time_trace_retrieval")].iloc[0]
    assert trace["status"] == "not_established"
    assert "selected_vs_layer_matched_random" in trace["ambiguous_gates"]


def test_analyze_command_writes_registered_outputs(tmp_path) -> None:
    mechanism_config = tmp_path / "mechanism.json"
    mechanism_config.write_text(
        json.dumps(NativeCountMechanismSpec(bootstrap_samples=100).to_dict()),
        encoding="utf-8",
    )
    trials = tmp_path / "trials.jsonl"
    rows = []
    for seed in (1, 2):
        for condition, value in {
            "clean": 2.0,
            "block_trace_context": 0.5,
            "block_trace_context_matched_control": 1.5,
        }.items():
            rows.append(
                {
                    "experiment_id": "answer_source_mask_factorial",
                    "condition": condition,
                    "model_label": "Qwen3-8B",
                    "request_id": f"request-{seed}",
                    "seed": seed,
                    "correct_count_margin": value,
                    "status": "ok",
                }
            )
        for source_group, conditions in {
            "trace_items": {
                "clean": 2.0,
                "selected_bank": 0.0,
                "layer_matched_random": 1.5,
            },
            "prompt_records": {
                "clean": 2.0,
                "selected_bank": 1.0,
                "layer_matched_random": 1.5,
            },
        }.items():
            for condition, value in conditions.items():
                rows.append(
                    {
                        "experiment_id": "answer_broad_head_ablation",
                        "condition": condition,
                        "source_group": source_group,
                        "bank_size": 8,
                        "model_label": "Qwen3-8B",
                        "request_id": f"request-{seed}",
                        "seed": seed,
                        "correct_count_margin": value,
                        "status": "ok",
                    }
                )
    trials.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    args = build_parser().parse_args(
        [
            "analyze",
            "--mechanism-config",
            str(mechanism_config),
            "--trials",
            str(trials),
            "--output",
            str(output),
        ]
    )
    args.func(args)
    assert (output / "estimands.csv").is_file()
    assert (output / "seed_effects.csv").is_file()
    assert (output / "mechanism_decision_ledger.csv").is_file()
    ledger = pd.read_csv(output / "mechanism_decision_ledger.csv")
    assert set(ledger["claim_scope"]) == {"development_only_no_confirmatory_claim"}
    estimands = pd.read_csv(output / "estimands.csv")
    broad = estimands.loc[estimands["experiment_id"].eq("answer_broad_head_ablation")]
    assert set(broad["source_group"]) == {"trace_items", "prompt_records"}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "analyze"
    assert manifest["formal_inference_eligible"] is False


def test_analyze_trace_patch_separates_temporal_directions(tmp_path) -> None:
    mechanism_config = tmp_path / "mechanism.json"
    mechanism_config.write_text(
        json.dumps(NativeCountMechanismSpec(bootstrap_samples=100).to_dict()),
        encoding="utf-8",
    )
    trials = tmp_path / "trace-patch.jsonl"
    rows = []
    for seed in range(1, 9):
        for direction in (
            "past_to_later_receiver",
            "future_to_earlier_receiver",
        ):
            values = {
                "clean": 0.0,
                "self_patch": 0.0,
                "full_donor_patch": 2.0,
                "progress_projected_patch": 1.5,
                "norm_matched_orthogonal_patch": 0.0,
            }
            for condition, value in values.items():
                rows.append(
                    {
                        "experiment_id": "trace_intermediate_state_patching",
                        "condition": condition,
                        "donor_direction": direction,
                        "model_label": "Qwen3-8B",
                        "request_id": f"trace-{seed}",
                        "seed": seed,
                        "donor_vs_receiver_city_log_odds": value,
                        "status": "ok",
                    }
                )
    trials.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    args = build_parser().parse_args(
        [
            "analyze",
            "--mechanism-config",
            str(mechanism_config),
            "--trials",
            str(trials),
            "--experiment-ids",
            "trace_intermediate_state_patching",
            "--outcome",
            "donor_vs_receiver_city_log_odds",
            "--output",
            str(output),
        ]
    )
    args.func(args)
    estimands = pd.read_csv(output / "estimands.csv")
    assert set(estimands["donor_direction"]) == {
        "past_to_later_receiver",
        "future_to_earlier_receiver",
    }
    ledger = pd.read_csv(output / "mechanism_decision_ledger.csv")
    stream = ledger.loc[ledger["claim"].eq("stream_written_state")].iloc[0]
    assert stream["status"] == "passes_registered_gate"
