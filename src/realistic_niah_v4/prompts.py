from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from realistic_niah.prompts import (
    build_messages,
    query_block,
    render_generation_prompt,
)
from realistic_niah_v3.spec import resolve_model_spec as resolve_v3_model_spec

from .spec import V4Config, V4ModelSpec


V4_DIRECT_QUERY_BLOCK = """\
How many city-score audit records are in the passage?
Do not explain, reason aloud, quote, or list any records.
Write the count as exactly one lowercase English number word from one through ten.
Your entire response must be exactly one line:
Total: <number word>"""


@dataclass(frozen=True)
class TokenSpan:
    slot_index: int
    start: int
    end: int
    active: bool
    kind: str
    canonical_length: int
    model_token_length: int


@dataclass(frozen=True)
class PromptEncoding:
    stimulus_id: str
    design_variant: str
    seed: int
    split: str
    count: int
    model_label: str
    text: str
    generation_prompt: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    query_position: int
    slot_spans: tuple[TokenSpan, ...]
    needle_spans: tuple[TokenSpan, ...]
    hard_negative_spans: tuple[TokenSpan, ...]
    count_candidate_token_ids: tuple[tuple[int, int], ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    def candidate_token_id(self, count: int) -> int:
        mapping = dict(self.count_candidate_token_ids)
        if int(count) not in mapping:
            raise KeyError(f"No count-token candidate for {count}")
        return int(mapping[int(count)])


def _flat_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("V4 prompt encoding requires batch size one")
        value = value[0]
    return list(value)


def _encode_text_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError("V4 requires an exact fast-tokenizer offset mapping")
    input_ids = [int(value) for value in _flat_list(encoded["input_ids"])]
    offsets = [
        (int(start), int(end)) for start, end in _flat_list(encoded["offset_mapping"])
    ]
    if len(input_ids) != len(offsets):
        raise RuntimeError("Input IDs and offset mapping have different lengths")
    if "attention_mask" in encoded:
        attention_mask = [int(value) for value in _flat_list(encoded["attention_mask"])]
    else:
        attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, offsets


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(value) for value in _flat_list(encoded["input_ids"])]


def _span_for_char_interval(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < int(char_end) and end > int(char_start)
    ]
    if not indices:
        raise RuntimeError(
            f"No model tokens overlap character span [{char_start}, {char_end})"
        )
    return indices[0], indices[-1] + 1


def _passage_char_start(model_text: str, passage: str) -> int:
    start = model_text.find(passage)
    if start < 0:
        raise RuntimeError("Rendered model prompt does not contain the passage")
    if model_text.find(passage, start + 1) >= 0:
        raise RuntimeError("Rendered model prompt contains the passage more than once")
    return start


def _count_candidate_ids(
    tokenizer: Any,
    model_text: str,
    counts: tuple[int, ...],
    candidate_words: tuple[str, ...],
    *,
    require_single_token: bool,
) -> tuple[tuple[int, int], ...]:
    if len(counts) != len(candidate_words):
        raise ValueError("Count/candidate-word length mismatch")
    prefix_ids = _encode_ids(tokenizer, model_text)
    result: list[tuple[int, int]] = []
    seen: set[int] = set()
    for count, word in zip(counts, candidate_words):
        full_ids = _encode_ids(tokenizer, model_text + f" {word}")
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                "Count continuation retokenized the answer prefix; change the "
                "registered answer_prefix before running causal metrics"
            )
        continuation = full_ids[len(prefix_ids) :]
        if not continuation:
            raise RuntimeError(f"Count {count} produced an empty continuation")
        if require_single_token and len(continuation) != 1:
            raise RuntimeError(
                f"Count {count} is not a single-token continuation: {continuation}"
            )
        first = int(continuation[0])
        if first in seen:
            raise RuntimeError(
                "Two registered counts share the same first continuation token"
            )
        seen.add(first)
        result.append((int(count), first))
    return tuple(result)


def render_v4_prompt(
    stimulus: dict[str, Any],
    *,
    tokenizer: Any,
    model_spec: V4ModelSpec,
    config: V4Config,
) -> PromptEncoding:
    config.validate()
    if int(stimulus["target_passage_tokens"]) != config.target_passage_tokens:
        raise ValueError("Stimulus passage length does not match V4 config")
    if str(stimulus["protocol_version"]) != config.protocol_version:
        raise ValueError("Stimulus protocol does not match V4 config")

    registered = resolve_v3_model_spec(model_spec.label)
    if registered.model_id != model_spec.model_id:
        raise RuntimeError("V3/V4 model registry disagreement")
    messages = build_messages(
        str(stimulus["passage"]),
        prompt_mode=config.prompt_mode,
    )
    registered_query = query_block(config.prompt_mode)
    if len(messages) != 1 or not str(messages[0]["content"]).endswith(registered_query):
        raise RuntimeError("Unexpected base prompt layout for registered V4")
    messages[0]["content"] = (
        str(messages[0]["content"])[: -len(registered_query)] + V4_DIRECT_QUERY_BLOCK
    )
    generation_prompt = render_generation_prompt(
        tokenizer,
        messages,
        model_spec=registered,
        prompt_mode=config.prompt_mode,
    )
    model_text = generation_prompt + config.answer_prefix
    input_ids, attention_mask, offsets = _encode_text_with_offsets(
        tokenizer, model_text
    )
    if not input_ids:
        raise RuntimeError("Rendered V4 prompt produced no tokens")

    suffix_start = len(generation_prompt)
    query_start, query_end = _span_for_char_interval(
        offsets,
        suffix_start,
        len(model_text),
    )
    query_position = query_end - 1
    if query_position != len(input_ids) - 1:
        raise RuntimeError("The registered answer query must be the last prompt token")

    passage = str(stimulus["passage"])
    passage_start = _passage_char_start(model_text, passage)
    slot_spans: list[TokenSpan] = []
    for slot in stimulus["slots"]:
        start, end = _span_for_char_interval(
            offsets,
            passage_start + int(slot["char_start"]),
            passage_start + int(slot["char_end"]),
        )
        slot_spans.append(
            TokenSpan(
                slot_index=int(slot["slot_index"]),
                start=start,
                end=end,
                active=bool(slot["active"]),
                kind=str(slot["content_kind"]),
                canonical_length=int(slot["canonical_token_length"]),
                model_token_length=end - start,
            )
        )
    hard_negative_spans: list[TokenSpan] = []
    for negative in stimulus["hard_negative_spans"]:
        start, end = _span_for_char_interval(
            offsets,
            passage_start + int(negative["char_start"]),
            passage_start + int(negative["char_end"]),
        )
        hard_negative_spans.append(
            TokenSpan(
                slot_index=int(negative["slot_index"]),
                start=start,
                end=end,
                active=False,
                kind="hard_negative",
                canonical_length=int(negative["canonical_token_length"]),
                model_token_length=end - start,
            )
        )
    if any(span.end > query_position for span in slot_spans):
        raise RuntimeError("A V4 slot lies after the answer query")
    if len(slot_spans) != max(config.needle_counts):
        raise RuntimeError("Rendered V4 prompt has the wrong number of slots")
    if [span.active for span in slot_spans] != [
        index < int(stimulus["gold_count"]) for index in range(len(slot_spans))
    ]:
        raise RuntimeError("Rendered V4 prompt violates nested active slots")

    candidates = _count_candidate_ids(
        tokenizer,
        model_text,
        config.needle_counts,
        config.count_candidate_words,
        require_single_token=config.require_single_token_count_candidates,
    )
    return PromptEncoding(
        stimulus_id=str(stimulus["stimulus_id"]),
        design_variant=str(stimulus["design_variant"]),
        seed=int(stimulus["seed"]),
        split=str(stimulus["split"]),
        count=int(stimulus["gold_count"]),
        model_label=model_spec.label,
        text=model_text,
        generation_prompt=generation_prompt,
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        query_position=query_position,
        slot_spans=tuple(slot_spans),
        needle_spans=tuple(span for span in slot_spans if span.active),
        hard_negative_spans=tuple(hard_negative_spans),
        count_candidate_token_ids=candidates,
    )
