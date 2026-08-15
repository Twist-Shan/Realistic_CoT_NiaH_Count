from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_5.followup_edges import (
    derive_factorial_encoding,
    distance_bin,
    natural_edge_delta,
    repeated_anchor_candidates,
    select_attention_mass_control,
    select_deterministic_random_control,
)


def encoding_fixture() -> PromptEncoding:
    spans = (
        TokenSpan(0, 20, 24, True, "needle", 4, 4),
        TokenSpan(1, 50, 54, True, "needle", 4, 4),
    )
    ids = list(range(80))
    ids[20:24] = [7, 100, 9, 10]
    ids[50:54] = [7, 101, 9, 11]
    return PromptEncoding(
        stimulus_id="fixture",
        design_variant="v4.4",
        seed=1,
        split="test",
        count=2,
        model_label="fixture",
        answer_format="numeric",
        text="fixture",
        generation_prompt="fixture",
        input_ids=tuple(ids),
        attention_mask=tuple([1] * len(ids)),
        query_position=79,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


def test_repeated_anchor_registration_requires_unique_in_span_successor():
    candidates = repeated_anchor_candidates(encoding_fixture())
    assert candidates[7] == ((20, 21), (50, 51))
    assert candidates[9] == ((22, 23), (52, 53))
    assert 10 not in candidates


def test_mass_and_random_controls_stay_in_distance_bin():
    row = torch.linspace(0.0, 1.0, 80)
    mass_key, audit = select_attention_mass_control(
        row,
        query=79,
        target_key=50,
        allowed=range(0, 79),
        excluded={20, 21, 22, 23, 50, 51, 52, 53},
        bin_width=16,
    )
    random_key, random_audit = select_deterministic_random_control(
        query=79,
        target_key=50,
        allowed=range(0, 79),
        excluded={20, 21, 22, 23, 50, 51, 52, 53},
        bin_width=16,
        label="fixture",
    )
    assert audit["exact_distance_bin"]
    assert random_audit["exact_distance_bin"]
    assert distance_bin(79, mass_key, width=16) == distance_bin(79, 50, width=16)
    assert distance_bin(79, random_key, width=16) == distance_bin(79, 50, width=16)


def test_natural_edge_delta_obeys_gqa_mapping():
    # Two KV heads, width two; query heads 0/1 map KV0 and 2/3 map KV1.
    values = torch.tensor(
        [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]
    )
    row = torch.tensor([0.25, 0.75])
    delta, audit = natural_edge_delta(
        row,
        values,
        keys=[1],
        key_start=0,
        query_head=3,
        query_heads=4,
        head_dim=2,
    )
    assert torch.allclose(delta, torch.tensor([-22.5, -30.0]))
    assert audit["edge_attention_mass"] == pytest.approx(0.75)


def test_factorial_derivation_preserves_length_query_and_moves_spans():
    base = encoding_fixture()
    replacements = ([7, 200, 9, 12], [7, 201, 9, 13])
    derived, audit = derive_factorial_encoding(
        base,
        identity_replacements=replacements,
        identity=True,
        context=True,
        position=True,
        context_width=2,
    )
    assert derived.sequence_length == base.sequence_length
    assert derived.query_position == base.query_position
    assert derived.input_ids[base.query_position] == base.input_ids[base.query_position]
    assert [span.model_token_length for span in derived.needle_spans] == [4, 4]
    assert [span.start for span in derived.needle_spans] != [20, 50]
    assert audit["identity"] and audit["context"] and audit["position"]
