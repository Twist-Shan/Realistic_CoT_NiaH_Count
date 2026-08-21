from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v5.causal import (
    bootstrap_seed_mean_ci,
    build_causal_plan,
    control_feasible_ranked_bank,
    fit_centroid_subspace,
    global_random_controls,
    layer_matched_random_controls,
    mechanism_continuations,
    paired_seed_effects,
    query_context_mask,
    rank_mechanism_heads,
    rank_pooled_source_specific_heads,
    ranked_bank_with_layer_profile,
    rank_retrieval_heads,
    sign_flip_pvalue,
    strict_ranked_bank,
)
from realistic_niah_v5.encoding import (
    build_native_causal_encoding,
    build_native_trace_encoding,
)
from realistic_niah_v5.generation import render_native_prompt
from realistic_niah_v5.parsing import (
    PARSER_FILE_SHA256,
    PARSER_IMPLEMENTATION,
    PARSER_UPSTREAM_COMMIT,
    parse_and_align_record,
    parse_trace_record,
)
from realistic_niah_v5.representation import (
    analyze_representation,
    curve_metrics,
    noise_decomposition,
    rank_metrics,
)
from realistic_niah_v5.spec import V5Config
from realistic_niah_v4.spec import V4ModelSpec


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        assert not add_special_tokens
        result = {
            "input_ids": [ord(value) for value in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) for value in text]

    def decode(
        self,
        values: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(int(value)) for value in values)

    def apply_chat_template(self, messages, **_kwargs) -> str:
        return "".join(str(message["content"]) for message in messages) + "<eos>"


def _row(family: str = "qwen3") -> dict[str, object]:
    if family == "qwen3":
        raw = (
            "<think>\nI will enumerate.\n"
            "1. Chicago received a score.\n"
            "2. Baku received a score.\n"
            "Therefore there are two.\n</think>\nTotal: 2"
        )
        label = "Qwen3-8B"
    else:
        raw = (
            "<|channel>thought\nI will enumerate.\n"
            "* Chicago received a score.\n"
            "• Baku received a score.\n"
            "Therefore there are two.\n<channel|>\nTotal: 2"
        )
        label = "Gemma4-E4B"
    prompt = "Chicago received a score of 72.\nBaku received a score of 98.\n"
    chicago_end = prompt.index("\n") + 1
    return {
        "request_id": f"test-{family}",
        "model_label": label,
        "model_family": family,
        "raw_output_text": raw,
        "output_token_ids": [ord(value) for value in raw],
        "input_ids": [ord(value) for value in prompt],
        "attention_mask": [1] * len(prompt),
        "prompt_record_spans": [
            {
                "slot_index": 1,
                "city": "Chicago",
                "score": 72,
                "start": 0,
                "end": chicago_end,
            },
            {
                "slot_index": 2,
                "city": "Baku",
                "score": 98,
                "start": chicago_end,
                "end": len(prompt),
            },
        ],
        "gold_records": [
            {"city": "Chicago", "score": 72},
            {"city": "Baku", "score": 98},
        ],
        "seed": 1254,
        "split": "confirmation",
    }


@pytest.mark.parametrize("family", ["qwen3", "gemma4"])
def test_hybrid_oracle_parser_and_registered_sites(family: str) -> None:
    result = parse_and_align_record(_row(family), CharacterTokenizer())
    assert result["parser"]["trace_category"] == "one_to_one"
    assert result["parser"]["trace_one_to_one"] is True
    assert result["parsed_count"] == 2
    assert result["exact_count"] is True
    assert result["parser_implementation"] == PARSER_IMPLEMENTATION
    assert result["parser_upstream_commit"] == PARSER_UPSTREAM_COMMIT
    assert set(PARSER_FILE_SHA256) == {
        "city_list_termination.py",
        "first_list_cutoff.py",
        "gold_city_cutoff.py",
        "hybrid_trace_parser.py",
        "parsing.py",
    }
    assert result["alignment_summary"]["eligible"] == len(result["token_sites"])
    item_sites = [
        site for site in result["token_sites"] if site["site_kind"] == "item_end"
    ]
    assert [site["occurrence"] for site in item_sites] == [1, 2]
    assert all(site["primary"] for site in item_sites)
    assert all(
        site["alignment_strategy"] == "literal_baseline_token_prefix"
        for site in item_sites
    )
    assert any(site["site_id"] == "answer_query" for site in result["token_sites"])
    assert any(
        site["site_id"] == "answer_query_v3" for site in result["token_sites"]
    )


def test_parser_version_is_provenance_not_a_fake_rule_id() -> None:
    parsed = parse_trace_record(_row())
    assert "parser_rule_id" not in parsed
    assert "v26_v44" not in str(parsed)


def test_answer_query_candidates_include_chat_termination() -> None:
    encoding = build_native_trace_encoding(
        _row(),
        CharacterTokenizer(),
        site_id="answer_query",
        candidate_counts=tuple(range(1, 11)),
    )
    answers = dict(encoding.count_candidate_answer_token_ids)
    scored = dict(encoding.count_candidate_token_ids)
    assert [span.city for span in encoding.prompt_record_spans] == [
        "Chicago",
        "Baku",
    ]
    assert tuple(chr(value) for value in answers[2]) == ("2",)
    assert "".join(chr(value) for value in scored[2]).endswith("<eos>")
    assert len(scored[2]) > len(answers[2])


def test_native_generation_registers_exact_prompt_record_spans() -> None:
    passage = "Chicago received a score of 72. Noise. Baku received a score of 98."
    records = []
    for slot, (city, score) in enumerate((("Chicago", 72), ("Baku", 98)), start=1):
        start = passage.index(city)
        end = passage.index(".", start) + 1
        records.append(
            {
                "slot_index": slot,
                "city": city,
                "score": score,
                "char_start": start,
                "char_end": end,
            }
        )
    stimulus = {
        "stimulus_id": "span-test",
        "design_variant": "v4.4",
        "seed": 1234,
        "split": "discovery",
        "gold_count": 2,
        "passage": passage,
        "gold_pairs": records,
        "active_needle_spans": records,
    }
    model_spec = V4ModelSpec(
        label="Qwen3-8B",
        model_id="test",
        revision="test",
        family="qwen3",
        loader_class="test",
    )
    prompt = render_native_prompt(
        stimulus, tokenizer=CharacterTokenizer(), model_spec=model_spec
    )
    assert [row["city"] for row in prompt.prompt_record_spans] == [
        "Chicago",
        "Baku",
    ]
    for row in prompt.prompt_record_spans:
        text = prompt.rendered_prompt[row["start"] : row["end"]]
        assert row["city"] in text


def test_sparse_context_controls_have_exactly_matched_key_budgets() -> None:
    encoding = build_native_trace_encoding(
        _row(), CharacterTokenizer(), site_id="answer_query"
    )
    trace = query_context_mask(encoding, condition="trace_only")
    control = query_context_mask(encoding, condition="matched_nontrace_only")
    assert int(trace.sum()) == int(control.sum())
    assert int(trace[0, encoding.query_position]) == 1
    assert int(control[0, encoding.query_position]) == 1


def test_parser_boundaries_define_fixed_city_transition_anchors() -> None:
    localized, localized_excluded = mechanism_continuations(
        _row(), CharacterTokenizer(), mechanism="retrieval_anchor_localization"
    )
    progress, progress_excluded = mechanism_continuations(
        _row(), CharacterTokenizer(), mechanism="progress_transition"
    )
    assert localized
    assert all(row["target_city"] == "Baku" for row in localized)
    assert len({row["query_output_token_index"] for row in localized}) == len(
        localized
    )
    assert len(
        {
            (
                row["target_output_token_start"],
                row["target_output_token_end"],
                tuple(row["target_token_ids"]),
            )
            for row in localized
        }
    ) == 1
    roles = {role for row in localized for role in row["anchor_roles"]}
    assert {"p0_item_end", "city_pre_d1"}.issubset(roles)
    assert all(row["status"] == "not_applicable" for row in localized_excluded)
    specification = localized[0]
    encoding = build_native_causal_encoding(
        _row(),
        CharacterTokenizer(),
        query_output_token_index=specification["query_output_token_index"],
        sequence_output_token_end=specification["target_output_token_end"],
        selected_site=specification,
    )
    assert encoding.query_position == (
        encoding.prompt_token_count + specification["query_output_token_index"]
    )
    assert encoding.sequence_length == (
        encoding.prompt_token_count + specification["target_output_token_end"]
    )
    assert not progress_excluded
    assert len(progress) == 1
    assert progress[0]["target_city"] == "Baku"
    assert "p0_item_end" in progress[0]["anchor_roles"]
    with pytest.raises(ValueError, match="legacy targeted_retrieval"):
        mechanism_continuations(
            _row(), CharacterTokenizer(), mechanism="targeted_retrieval"
        )


def test_v5_config_freezes_sites_and_disjoint_seed_splits() -> None:
    config = V5Config()
    config.validate()
    assert config.primary_trace_site == "item_end"
    assert config.causal_head_mechanisms == ("retrieval_anchor_localization",)
    assert config.causal_head_selection_metric.startswith("seed_first")
    assert set(config.all_seeds).issubset(config.causal_development_seeds)
    assert not config.causal_confirmation_seeds
    assert set(config.discovery_seeds).isdisjoint(config.confirmation_seeds)
    with pytest.raises(ValueError, match="primary site"):
        V5Config(primary_trace_site="city_end").validate()


def test_geometry_metrics_recover_a_low_dimensional_ordered_curve() -> None:
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(1, 6), 20)
    direction = rng.normal(size=12)
    direction /= np.linalg.norm(direction)
    states = labels[:, None] * direction[None, :] + rng.normal(
        scale=0.02, size=(len(labels), 12)
    )
    ranks = rank_metrics(states, labels)
    curve = curve_metrics(states, labels)
    noise = noise_decomposition(states, labels)
    assert ranks["centroid_rank_1_fraction"] > 0.99
    # Repeated integer pair distances induce ties; a near-linear noisy curve
    # therefore has a Spearman ceiling below one under average tie ranks.
    assert curve["label_distance_spearman"] > 0.90
    assert noise["label_centroid_fraction"] > 0.99
    assert noise["decomposition_identity_error"] < 1e-8


def test_discovery_head_ranking_and_layer_matched_controls() -> None:
    rows = []
    for layer in (0, 1):
        for head in range(4):
            for seed in range(3):
                rows.append(
                    {
                        "model_label": "Qwen3-8B",
                        "split": "discovery",
                        "mechanism": "targeted_retrieval",
                        "gold_count": 10,
                        "trace_one_to_one": True,
                        "layer": layer,
                        "head": head,
                        "target_needle_raw_mass": (
                            1.0 - 0.1 * layer - 0.01 * head
                        ),
                        "target_needle_relative_mass": 0.8,
                        "target_needle_top1": True,
                        "seed": seed,
                    }
                )
    rows.append(
        {
            "model_label": "Qwen3-8B",
            "split": "discovery",
            "mechanism": "targeted_retrieval",
            "gold_count": 10,
            "trace_one_to_one": False,
            "layer": 0,
            "head": 3,
            "target_needle_raw_mass": 100.0,
            "target_needle_relative_mass": 1.0,
            "target_needle_top1": True,
            "seed": 98,
        }
    )
    rows.append(
        {
            "model_label": "Qwen3-8B",
            "split": "discovery",
            "mechanism": "progress_transition",
            "gold_count": 1,
            "trace_one_to_one": True,
            "layer": 0,
            "head": 3,
            "target_needle_raw_mass": 100.0,
            "target_needle_relative_mass": 1.0,
            "target_needle_top1": True,
            "seed": 99,
        }
    )
    ranking = rank_retrieval_heads(pd.DataFrame(rows))
    assert tuple(ranking.iloc[0][["layer", "head"]].astype(int)) == (0, 0)
    progress = rank_mechanism_heads(
        pd.DataFrame(rows), mechanism="progress_transition"
    )
    assert tuple(progress.iloc[0][["layer", "head"]].astype(int)) == (0, 3)
    controls = layer_matched_random_controls(
        ranking,
        [(0, 0), (1, 0)],
        repeats=3,
        seed_text="registered",
    )
    assert len(controls) == 3
    assert len({tuple(bank) for bank in controls}) == 3
    assert all([layer for layer, _head in bank] == [0, 1] for bank in controls)
    assert all((0, 0) not in bank and (1, 0) not in bank for bank in controls)


def test_ranked_bank_preserves_capacity_for_exact_layer_controls() -> None:
    ranking = pd.DataFrame(
        [
            {
                "layer": layer,
                "head": head,
                "discovery_rank": 1 + layer * 4 + head,
            }
            for layer in (0, 1)
            for head in range(4)
        ]
    )
    selected = control_feasible_ranked_bank(ranking, bank_size=3)
    assert selected == [(0, 0), (0, 1), (1, 0)]
    controls = layer_matched_random_controls(
        ranking, selected, repeats=2, seed_text="capacity"
    )
    assert all(len(bank) == 3 for bank in controls)


def test_local_pooled_ranking_is_seed_equal_and_event_proportional() -> None:
    rows = []
    values = {
        (1234, 0): [1.0, 1.0, 10.0],
        (1235, 0): [0.0],
        (1234, 1): [1.0, 1.0, 1.0],
        (1235, 1): [1.0],
    }
    for (seed, head), event_values in values.items():
        for event_index, value in enumerate(event_values):
            rows.append(
                {
                    "model_label": "Qwen3-8B",
                    "request_id": f"request-{seed}",
                    "seed": seed,
                    "anchor_role": "p0_item_end",
                    "primary_anchor_eligible": False,
                    "local_anchor_eligible": True,
                    "event_specific": True,
                    "status": "ok",
                    "layer": 0,
                    "head": head,
                    "target_source_attention_mass": value,
                    "event_index": event_index,
                }
            )
    ranking = rank_pooled_source_specific_heads(
        pd.DataFrame(rows),
        anchor_role="p0_item_end",
        selection_metric="target_source_attention_mass",
        selection_eligibility_scope="local",
        selection_aggregation="seed_event_mean",
    )
    scores = {
        int(row.head): float(row.discovery_selection_value)
        for row in ranking.itertuples(index=False)
    }
    assert scores[0] == pytest.approx(2.0)
    assert scores[1] == pytest.approx(1.0)
    assert tuple(ranking.iloc[0][["layer", "head"]].astype(int)) == (0, 0)
    assert set(ranking["selection_eligibility_scope"]) == {"local"}
    assert set(ranking["selection_aggregation"]) == {"seed_event_mean"}


def test_global_random_controls_keep_literal_treatment_and_same_k() -> None:
    ranking = pd.DataFrame(
        [
            {
                "layer": layer,
                "head": head,
                "discovery_rank": 1 + layer * 8 + head,
            }
            for layer in range(3)
            for head in range(8)
        ]
    )
    selected = strict_ranked_bank(ranking, bank_size=7)
    controls = global_random_controls(
        ranking,
        selected,
        repeats=3,
        seed_text="global-same-k",
    )
    assert len(controls) == 3
    assert len({tuple(bank) for bank in controls}) == 3
    assert all(len(bank) == len(selected) for bank in controls)
    assert all(set(selected).isdisjoint(bank) for bank in controls)


def test_strict_ranked_bank_is_not_rewritten_for_controls() -> None:
    ranking = pd.DataFrame(
        [
            {"layer": 0, "head": head, "discovery_rank": head + 1}
            for head in range(8)
        ]
        + [
            {"layer": 1, "head": head, "discovery_rank": head + 9}
            for head in range(8)
        ]
    )
    assert strict_ranked_bank(ranking, bank_size=5) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    ]
    with pytest.raises(ValueError, match="Not enough non-selected heads"):
        layer_matched_random_controls(
            ranking,
            strict_ranked_bank(ranking, bank_size=5),
            repeats=1,
            seed_text="treatment-is-frozen",
        )


def test_ranked_bank_can_match_a_reference_layer_profile() -> None:
    ranking = pd.DataFrame(
        [
            {
                "layer": layer,
                "head": head,
                "discovery_rank": rank,
            }
            for rank, (layer, head) in enumerate(
                [(0, 3), (1, 2), (0, 1), (2, 0), (1, 0), (2, 1)],
                start=1,
            )
        ]
    )
    selected = ranked_bank_with_layer_profile(
        ranking, layer_profile={0: 1, 1: 2, 2: 1}
    )
    assert selected == [(0, 3), (1, 2), (2, 0), (1, 0)]
    assert pd.Series([layer for layer, _head in selected]).value_counts().to_dict() == {
        0: 1,
        1: 2,
        2: 1,
    }
    with pytest.raises(ValueError, match="for quota"):
        ranked_bank_with_layer_profile(ranking, layer_profile={2: 3})


def test_ranked_bank_preserves_three_distinct_exact_controls() -> None:
    ranking = pd.DataFrame(
        [
            {
                "layer": layer,
                "head": head,
                "discovery_rank": 1 + layer * 8 + head,
            }
            for layer in (0, 1, 2)
            for head in range(8)
        ]
    )
    selected = control_feasible_ranked_bank(
        ranking,
        bank_size=8,
        control_repeats=3,
    )
    selected_counts = pd.Series(
        [layer for layer, _head in selected]
    ).value_counts()
    assert selected_counts.max() <= 3
    controls = layer_matched_random_controls(
        ranking,
        selected,
        repeats=3,
        seed_text="three-distinct-controls",
    )
    selected_set = set(selected)
    assert len(controls) == 3
    assert len({tuple(bank) for bank in controls}) == 3
    for bank in controls:
        control_counts = pd.Series(
            [layer for layer, _head in bank]
        ).value_counts()
        assert control_counts.to_dict() == selected_counts.to_dict()
        assert selected_set.isdisjoint(bank)


def test_causal_plan_crossfits_source_specific_anchor_pooled_banks(
    tmp_path,
) -> None:
    write_rows = [
            {
                "model_label": "Qwen3-8B",
                "request_id": f"r{seed}-{anchor}",
                "seed": seed,
                "anchor_role": anchor,
                "event_specific": True,
                "primary_anchor_eligible": True,
                "status": "ok",
                "layer": 0,
                "head": head,
                "source_specific_ov_write_norm": 10.0 - head,
                "source_attention_mass": 0.1 + 0.1 * head,
                "target_source_attention_mass": 0.1 + 0.1 * head,
                "target_source_relative_attention_mass": 0.1 + 0.2 * head,
                "target_minus_max_wrong_source_attention_mass": -0.3 + 0.2 * head,
                "grammar_pair": (
                    "adjacent_rank_after_city -> same_unit_rank_before_city"
                ),
            }
            for seed in range(1234, 1264)
            for anchor in ("p0_item_end", "city_pre_d1")
            for head in range(4)
        ]
    writes = pd.DataFrame(write_rows)
    ranking = rank_pooled_source_specific_heads(writes)
    assert tuple(ranking.iloc[0][["layer", "head"]].astype(int)) == (0, 0)
    source = tmp_path / "source_writes.csv"
    writes.to_csv(source, index=False)
    config = V5Config(
        causal_primary_bank_size=2,
        causal_crossfit_folds=2,
        causal_random_controls=1,
    )
    paths = build_causal_plan(
        source,
        tmp_path / "plan",
        config=config,
        anchor_role="p0_item_end",
    )
    plan = pd.read_csv(paths["plan"])
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert len(plan) == 4
    assert set(plan["mechanism"]) == {"retrieval_anchor_localization"}
    assert set(plan["condition"]) == {"selected_bank", "layer_matched_random"}
    assert set(plan["bank_size"]) == {2}
    assert set(plan["fold"]) == {0, 1}
    assert audit["confirmation_used_for_selection"] is False
    assert audit["mechanism"]["target"] == (
        "identical next-city token span at every anchor"
    )
    sweep_paths = build_causal_plan(
        source,
        tmp_path / "plan_k1",
        config=config,
        bank_size=1,
        anchor_role="p0_item_end",
        minimum_layer=0,
        maximum_layer=0,
    )
    sweep_plan = pd.read_csv(sweep_paths["plan"])
    sweep_audit = json.loads(sweep_paths["audit"].read_text(encoding="utf-8"))
    assert set(sweep_plan["bank_size"]) == {1}
    assert sweep_audit["registered_bank_size"] == 1
    assert sweep_audit["selection_anchor_role"] == "p0_item_end"
    assert sweep_audit["representation_guided_minimum_layer"] == 0
    assert sweep_audit["representation_guided_maximum_layer"] == 0
    smoke_source = tmp_path / "source_writes_smoke.csv"
    writes.loc[writes["seed"].isin([1234, 1235])].to_csv(
        smoke_source, index=False
    )
    smoke_paths = build_causal_plan(
        smoke_source,
        tmp_path / "plan_smoke",
        config=config,
        anchor_role="p0_item_end",
        allow_incomplete_development_smoke=True,
    )
    smoke_audit = json.loads(
        smoke_paths["audit"].read_text(encoding="utf-8")
    )
    assert smoke_audit["formal_inference_eligible"] is False
    assert smoke_audit["source_seed_coverage"]["Qwen3-8B"] == [1234, 1235]

    selected_only_paths = build_causal_plan(
        source,
        tmp_path / "plan_selected_only",
        config=config,
        bank_size=3,
        anchor_role="p0_item_end",
        include_random_controls=False,
    )
    selected_only_plan = pd.read_csv(selected_only_paths["plan"])
    selected_only_audit = json.loads(
        selected_only_paths["audit"].read_text(encoding="utf-8")
    )
    assert set(selected_only_plan["condition"]) == {"selected_bank"}
    assert selected_only_audit["random_controls_included"] is False
    assert selected_only_audit["formal_inference_eligible"] is False

    global_paths = build_causal_plan(
        source,
        tmp_path / "plan_global_random",
        config=config,
        bank_size=2,
        anchor_role="p0_item_end",
        random_control_matching="global",
    )
    global_plan = pd.read_csv(global_paths["plan"])
    global_audit = json.loads(
        global_paths["audit"].read_text(encoding="utf-8")
    )
    assert set(global_plan["condition"]) == {"selected_bank", "global_random"}
    assert set(global_plan["random_control_matching"]) == {"global"}
    assert global_audit["random_control_matching"] == "global"

    confirmation_config = V5Config(
        discovery_seeds=tuple(range(1234, 1264)),
        confirmation_seeds=(1336, 1337),
        causal_development_seeds=tuple(range(1234, 1264)),
        causal_confirmation_seeds=(1336, 1337),
        causal_primary_bank_size=2,
        causal_crossfit_folds=2,
        causal_random_controls=1,
    )
    confirmation_paths = build_causal_plan(
        source,
        tmp_path / "plan_confirmation",
        config=confirmation_config,
        anchor_role="p0_item_end",
        confirmation_plan=True,
    )
    confirmation_plan = pd.read_csv(confirmation_paths["plan"])
    confirmation_audit = json.loads(
        confirmation_paths["audit"].read_text(encoding="utf-8")
    )
    assert len(confirmation_plan) == 2
    assert set(confirmation_plan["fold"]) == {0}
    assert {
        tuple(json.loads(value))
        for value in confirmation_plan["validation_seeds"]
    } == {(1336, 1337)}
    assert confirmation_audit["confirmation_plan"] is True
    assert confirmation_audit["formal_inference_eligible"] is True

    full_panel_paths = build_causal_plan(
        smoke_source,
        tmp_path / "plan_full_panel",
        config=confirmation_config,
        anchor_role="p0_item_end",
        full_panel_plan=True,
    )
    full_panel_plan = pd.read_csv(full_panel_paths["plan"])
    full_panel_audit = json.loads(
        full_panel_paths["audit"].read_text(encoding="utf-8")
    )
    assert len(full_panel_plan) == 2
    assert set(full_panel_plan["fold"]) == {0}
    assert {
        tuple(json.loads(value))
        for value in full_panel_plan["validation_seeds"]
    } == {tuple(range(1234, 1264)) + (1336, 1337)}
    assert full_panel_audit["full_panel_plan"] is True
    assert full_panel_audit["formal_inference_eligible"] is False
    assert full_panel_audit["registered_confirmation_subcohort_eligible"] is True


def test_same_site_attention_metric_and_target_grammar_choose_the_bank() -> None:
    rows = []
    for seed in (1234, 1235):
        for grammar in ("grammar_a", "grammar_b"):
            for head in range(4):
                rows.append(
                    {
                        "model_label": "Gemma4-E4B",
                        "request_id": f"r{seed}-{grammar}",
                        "seed": seed,
                        "anchor_role": "city_pre_d1",
                        "anchor_roles": ["city_pre_d1"],
                        "event_specific": True,
                        "primary_anchor_eligible": True,
                        "status": "ok",
                        "grammar_pair": f"source -> {grammar}",
                        "target_retrieval_surface_variant": (
                            "rank_before_city_compact"
                            if grammar == "grammar_a"
                            else "rank_before_city_record_clause"
                        ),
                        "layer": 0,
                        "head": head,
                        # OV favors H0, while target attention at the same
                        # query favors H1 for grammar_a and H2 for grammar_b.
                        "source_specific_ov_write_norm": 10.0 - head,
                        "source_attention_mass": (
                            0.9
                            if head == (1 if grammar == "grammar_a" else 2)
                            else 0.1
                        ),
                        "target_source_attention_mass": (
                            0.9
                            if head == (1 if grammar == "grammar_a" else 2)
                            else 0.1
                        ),
                        "target_source_relative_attention_mass": (
                            0.8
                            if head == (1 if grammar == "grammar_a" else 2)
                            else 0.05
                        ),
                        "target_minus_max_wrong_source_attention_mass": (
                            0.6
                            if head == (1 if grammar == "grammar_a" else 2)
                            else -0.3
                        ),
                    }
                )
    writes = pd.DataFrame(rows)
    ov = rank_pooled_source_specific_heads(
        writes,
        anchor_role="city_pre_d1",
        selection_metric="source_specific_ov_write_norm",
        target_grammar_class="grammar_a",
    )
    attention_a = rank_pooled_source_specific_heads(
        writes,
        anchor_role="city_pre_d1",
        selection_metric="target_source_relative_attention_mass",
        target_grammar_class="grammar_a",
        target_retrieval_surface_variant="rank_before_city_compact",
    )
    attention_b = rank_pooled_source_specific_heads(
        writes,
        anchor_role="city_pre_d1",
        selection_metric="target_source_relative_attention_mass",
        target_grammar_class="grammar_b",
    )
    assert int(ov.iloc[0]["head"]) == 0
    assert int(attention_a.iloc[0]["head"]) == 1
    assert int(attention_b.iloc[0]["head"]) == 2
    assert set(attention_a["selection_anchor_role"]) == {"city_pre_d1"}
    assert set(attention_a["selection_target_grammar_class"]) == {
        "grammar_a"
    }
    assert set(
        attention_a["selection_target_retrieval_surface_variant"]
    ) == {"rank_before_city_compact"}


def test_subspace_and_seed_level_statistics() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(1, 5), 8)
    states = np.zeros((len(labels), 9), dtype=float)
    states[:, 0] = labels
    states[:, 1:] = rng.normal(scale=0.05, size=(len(labels), 8))
    center, basis = fit_centroid_subspace(states, labels, rank=3)
    assert center.shape == (9,)
    assert basis.shape == (9, 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)
    effects = np.array([0.2, 0.4, 0.6, 0.8])
    ci = bootstrap_seed_mean_ci(effects, samples=1000, seed=4)
    assert ci["mean_effect"] == pytest.approx(0.5)
    assert 0.0 < sign_flip_pvalue(effects) <= 1.0


def test_causal_query_rows_are_averaged_before_seed_inference() -> None:
    trials = pd.DataFrame(
        [
            {"model_label": "Qwen3-8B", "seed": 1, "request_id": "r1", "anchor_equivalence_id": "a", "condition": "ranked", "y": 0.0},
            {"model_label": "Qwen3-8B", "seed": 1, "request_id": "r1", "anchor_equivalence_id": "b", "condition": "ranked", "y": 2.0},
            {"model_label": "Qwen3-8B", "seed": 1, "request_id": "r1", "anchor_equivalence_id": "a", "condition": "random", "y": 2.0},
            {"model_label": "Qwen3-8B", "seed": 1, "request_id": "r1", "anchor_equivalence_id": "b", "condition": "random", "y": 4.0},
        ]
    )
    effects = paired_seed_effects(
        trials, treatment="ranked", control="random", outcome="y"
    )
    assert effects.loc[0, "mean_effect"] == pytest.approx(-2.0)


def test_capture_shards_feed_end_to_end_representation_analysis(tmp_path) -> None:
    rng = np.random.default_rng(11)
    index_rows = []
    for row_index, (split, seed) in enumerate(
        [("discovery", value) for value in range(4)]
        + [("confirmation", value) for value in range(4, 8)]
    ):
        request_id = f"r{row_index}"
        shard = tmp_path / "shards" / request_id
        shard.mkdir(parents=True)
        states = np.stack(
            [
                np.asarray([occurrence, occurrence**2, 0.0, 1.0])
                + rng.normal(scale=0.01, size=4)
                for occurrence in (1, 2, 3)
            ]
        )[:, None, :]
        np.savez(
            shard / "states.npz",
            layer_indices=np.asarray([0]),
            site_states=states.astype(np.float32),
        )
        manifest = {
            "request_id": request_id,
            "stimulus_id": request_id,
            "model_label": "Qwen3-8B",
            "model_family": "qwen3",
            "seed": seed,
            "split": split,
            "gold_count": 10,
            "parsed_count": 10,
            "exact_count": True,
            "parser": {
                "detected": True,
                "trace_one_to_one": True,
                "trace_category": "one_to_one",
            },
            "site_rows": [
                {
                    "site_id": f"item_end:{occurrence}",
                    "site_kind": "item_end",
                    "occurrence": occurrence,
                    "alignment_strategy": "literal_baseline_token_prefix",
                }
                for occurrence in (1, 2, 3)
            ],
        }
        (shard / "capture_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        index_rows.append(
            {
                "request_id": request_id,
                "manifest_path": f"shards/{request_id}/capture_manifest.json",
                "states_path": f"shards/{request_id}/states.npz",
            }
        )
    index = tmp_path / "capture_index.jsonl"
    index.write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8"
    )
    paths = analyze_representation(index, tmp_path / "analysis", config=V5Config())
    summary = pd.read_csv(paths["summary"])
    regression = pd.read_csv(paths["regression"])
    assert set(summary["cohort"]) == {
        "parser_hit",
        "one_to_one",
        "one_to_one_correct",
    }
    assert set(regression["probe"]) == {"ridge", "knn_5"}
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["primary_site"] == "item_end"
    assert audit["groups_completed"] == 3
