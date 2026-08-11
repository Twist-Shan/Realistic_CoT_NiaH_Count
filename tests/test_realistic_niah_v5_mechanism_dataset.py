from __future__ import annotations

from realistic_niah_v5.mechanism_dataset import (
    audit_paired_records,
    build_non_thinking_user_text,
    paired_record,
)


def _stimulus() -> dict:
    text = "Excerpt: City record."
    passage = f"before {text} after"
    start = passage.index(text)
    import hashlib

    return {
        "schema_version": "realistic_niah_v4_stimulus_v1",
        "protocol_version": "realistic_niah_v4_nonthinking_v3",
        "stimulus_id": "V4_4_T10000_N1_seed1234",
        "design_variant": "v4.4",
        "seed": 1234,
        "split": "discovery",
        "target_passage_tokens": 10000,
        "canonical_passage_tokens": 10000,
        "canonical_tokenizer": "Qwen/Qwen3-8B",
        "canonical_tokenizer_revision": "revision",
        "num_needles": 1,
        "gold_count": 1,
        "passage": passage,
        "passage_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        "gold_pairs": [{"slot_index": 1, "city": "Oslo", "score": 7}],
        "slots": [{"slot_index": 1, "active": True}],
        "active_needle_spans": [
            {
                "slot_index": 1,
                "needle_id": "N1",
                "text": text,
                "city": "Oslo",
                "score": 7,
                "canonical_span_start": 8,
                "canonical_span_end": 11,
                "canonical_token_length": 3,
                "char_start": start,
                "char_end": start + len(text),
            }
        ],
        "hard_negative_spans": [],
        "design": {"family_key": "unit"},
        "haystack": {"source": "unit"},
    }


def test_paired_modes_share_exact_v44_backbone() -> None:
    stimulus = _stimulus()
    non = paired_record(stimulus, mode="non_thinking")
    native = paired_record(stimulus, mode="native_thinking")
    audit = audit_paired_records([non], [native])
    assert audit["passed"] is True
    assert non["backbone_sha256"] == native["backbone_sha256"]
    assert non["passage"] == native["passage"]
    assert non["active_needle_spans"] == native["active_needle_spans"]
    assert non["chat_template_thinking_enabled"] is False
    assert native["chat_template_thinking_enabled"] is True
    assert non["expected_final_line"] == "Total:1"
    assert native["expected_final_line"] == "Total: 1"


def test_non_thinking_prompt_is_exact_v4_numeric_layout() -> None:
    prompt = build_non_thinking_user_text("PASSAGE")
    assert prompt.count("PASSAGE") == 1
    assert prompt.endswith("Total:<integer>")
    assert "with no space after the colon" in prompt

