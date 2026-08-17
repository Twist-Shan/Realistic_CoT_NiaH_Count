from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v5.causal import (
    bootstrap_seed_mean_ci,
    build_causal_plan,
    fit_centroid_subspace,
    layer_matched_random_controls,
    mechanism_continuations,
    paired_seed_effects,
    query_context_mask,
    rank_mechanism_heads,
    rank_retrieval_heads,
    sign_flip_pvalue,
)
from realistic_niah_v5.encoding import build_native_trace_encoding
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
    return {
        "request_id": f"test-{family}",
        "model_label": label,
        "model_family": family,
        "raw_output_text": raw,
        "output_token_ids": [ord(value) for value in raw],
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "prompt_record_spans": [
            {"slot_index": 1, "city": "Chicago", "score": 72, "start": 0, "end": 1},
            {"slot_index": 2, "city": "Baku", "score": 98, "start": 1, "end": 2},
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


def test_parser_boundaries_define_targeted_and_progress_causal_continuations() -> None:
    targeted, targeted_excluded = mechanism_continuations(
        _row(), CharacterTokenizer(), mechanism="targeted_retrieval"
    )
    progress, progress_excluded = mechanism_continuations(
        _row(), CharacterTokenizer(), mechanism="progress_transition"
    )
    assert not targeted_excluded
    assert [(row["query_site_id"], row["target_city"]) for row in targeted] == [
        ("marker_end:1", "Chicago"),
        ("marker_end:2", "Baku"),
    ]
    assert not progress_excluded
    assert [(row["transition_phase"], row["target_city"]) for row in progress] == [
        ("continue", "Baku"),
        ("stop", None),
    ]


def test_v5_config_freezes_sites_and_disjoint_seed_splits() -> None:
    config = V5Config()
    config.validate()
    assert config.primary_trace_site == "item_end"
    assert config.causal_head_mechanisms == (
        "targeted_retrieval",
        "progress_transition",
    )
    assert "broad" not in config.causal_head_selection_metric
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
    assert all([layer for layer, _head in bank] == [0, 1] for bank in controls)
    assert all((0, 0) not in bank and (1, 0) not in bank for bank in controls)


def test_causal_plan_audits_unmatched_banks_instead_of_scheduling_them(
    tmp_path,
) -> None:
    attention_rows = [
            {
                "model_label": "Qwen3-8B",
                "split": "discovery",
                "mechanism": mechanism,
                "gold_count": 10,
                "trace_one_to_one": True,
                "layer": 0,
                "head": head,
                "target_needle_raw_mass": 1.0 - head / 10,
                "target_needle_relative_mass": 0.8,
                "target_needle_top1": True,
            }
            for mechanism in ("targeted_retrieval", "progress_transition")
            for head in range(4)
        ]
    attention = pd.DataFrame(attention_rows)
    source = tmp_path / "attention.csv"
    attention.to_csv(source, index=False)
    config = V5Config(causal_head_bank_sizes=(4,), causal_random_controls=1)
    paths = build_causal_plan(source, tmp_path / "plan", config=config)
    plan = pd.read_csv(paths["plan"])
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert plan.empty
    assert {row["mechanism"] for row in audit["skipped_banks"]} == {
        "targeted_retrieval",
        "progress_transition",
    }
    assert audit["skipped_banks"][0]["bank_size"] == 4
    assert "Not enough non-selected heads" in audit["skipped_banks"][0]["reason"]


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
            {"model_label": "Qwen3-8B", "seed": 1, "condition": "ranked", "y": 0.0},
            {"model_label": "Qwen3-8B", "seed": 1, "condition": "ranked", "y": 2.0},
            {"model_label": "Qwen3-8B", "seed": 1, "condition": "random", "y": 2.0},
            {"model_label": "Qwen3-8B", "seed": 1, "condition": "random", "y": 4.0},
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
