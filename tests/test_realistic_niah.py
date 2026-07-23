from __future__ import annotations

from dataset_generation.dynamic_niah import TokenizerAdapter
from realistic_niah.parsing import (
    evaluate_generation,
    split_reasoning_and_final,
)
from realistic_niah.prompts import build_messages
from realistic_niah.spec import (
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    SEEDS,
    SMOKE_NEEDLE_COUNTS,
)
from realistic_niah.stimuli import (
    FreezeSpec,
    audit_frozen_grid,
    freeze_grid,
    freeze_stimulus,
)


def test_registered_grid_replaces_zero_with_six() -> None:
    assert 0 not in NEEDLE_COUNTS
    assert 6 in NEEDLE_COUNTS
    assert NEEDLE_COUNTS == (1, 2, 3, 4, 5, 6, 8, 10, 20, 30)
    assert len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS) == 150
    assert SMOKE_NEEDLE_COUNTS == (5, 6, 30)


def test_query_order_only_moves_the_task_block() -> None:
    passage = "Alpha passage."
    direct_first = build_messages(
        passage,
        prompt_mode="direct",
        query_order="query_first",
    )
    native_first = build_messages(
        passage,
        prompt_mode="native_thinking",
        query_order="query_first",
    )
    query_last = build_messages(
        passage,
        prompt_mode="direct",
        query_order="query_last",
    )

    assert direct_first == native_first
    assert direct_first[0]["content"].index("How many") < direct_first[0][
        "content"
    ].index("<passage>")
    assert query_last[0]["content"].index("<passage>") < query_last[0][
        "content"
    ].index("How many")
    assert "city-score" not in query_last[0]["content"].split("<passage>", 1)[0]


def test_enumeration_parser_handles_six_records() -> None:
    gold = [
        {"city": f"City {index}", "score": 50 + index}
        for index in range(1, 7)
    ]
    output = "\n".join(
        [
            *(f"{index}. City {index}: {50 + index}" for index in range(1, 7)),
            "Total: 6",
        ]
    )
    result = evaluate_generation(
        output,
        prompt_mode="enumeration",
        gold_pairs=gold,
        finish_reason="stop",
    )

    assert result["predicted_count"] == 6
    assert result["exact_count"] is True
    assert result["pair_recall"] == 1.0
    assert result["listed_total_matches_length"] is True


def test_qwen_reasoning_and_final_are_separated() -> None:
    reasoning, final = split_reasoning_and_final(
        "<think>\nI found six records.\n</think>\n\nTotal: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_qwen_prompt_supplied_opening_think_token_is_supported() -> None:
    reasoning, final = split_reasoning_and_final(
        "I found six records.\n</think>\n\nTotal: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_gemma_reasoning_and_final_are_separated() -> None:
    reasoning, final = split_reasoning_and_final(
        "<|channel>thought\nI found six records.<channel|>Total: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_gemma_prompt_supplied_opening_thought_channel_is_supported() -> None:
    reasoning, final = split_reasoning_and_final(
        "I found six records.<channel|>Total: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_backend_parsed_native_response_is_treated_as_final() -> None:
    reasoning, final = split_reasoning_and_final(
        "Total: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == ""
    assert final == "Total: 6"


def test_fixed_post_insertion_length_with_simple_tokenizer() -> None:
    tokenizer = TokenizerAdapter("simple")
    spec = FreezeSpec(
        passage_lengths=(180,),
        needle_counts=(6,),
        seeds=(1234,),
        canonical_tokenizer="simple",
        max_search_attempts=12,
        max_window_retries=2,
        minimum_filler_tokens=40,
    )

    row = freeze_stimulus(
        target_passage_tokens=180,
        num_needles=6,
        seed=1234,
        tokenizer=tokenizer,
        spec=spec,
    )

    assert row["canonical_passage_tokens"] == 180
    assert row["gold_count"] == 6
    assert len(row["needles"]) == 6
    assert row["length_search"]["post_insertion_truncation"] is False


def test_frozen_grid_passes_independent_audit(tmp_path) -> None:
    spec = FreezeSpec(
        passage_lengths=(180,),
        needle_counts=(6,),
        seeds=(1234,),
        canonical_tokenizer="simple",
        max_search_attempts=12,
        max_window_retries=2,
        minimum_filler_tokens=40,
    )
    paths = freeze_grid(
        output_dir=tmp_path,
        spec=spec,
        require_huggingface_tokenizer=False,
    )

    report = audit_frozen_grid(
        stimuli_path=paths["stimuli"],
        manifest_path=paths["manifest"],
        require_huggingface_tokenizer=False,
    )

    assert report["passed"] is True
    assert report["rows_checked"] == 1
    assert report["needle_counts"] == [6]
