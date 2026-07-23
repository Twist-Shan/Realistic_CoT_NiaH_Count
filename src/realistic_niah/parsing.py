from __future__ import annotations

import re
from typing import Any

TOTAL_RE = re.compile(
    r"(?im)^\s*Total\s*:\s*(-?\d+)\s*"
    r"(?:<\|im_end\|>|<turn\|>)?\s*$"
)
ENUMERATION_RE = re.compile(
    r"(?m)^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+)\s*$"
)
QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"
GEMMA_THINK_OPEN = "<|channel>thought\n"
GEMMA_THINK_CLOSE = "<channel|>"


def split_reasoning_and_final(
    raw_text: str,
    *,
    prompt_mode: str,
) -> tuple[str, str]:
    if prompt_mode != "native_thinking":
        return "", raw_text.strip()

    if QWEN_THINK_OPEN in raw_text:
        after_open = raw_text.split(QWEN_THINK_OPEN, 1)[1]
        if QWEN_THINK_CLOSE not in after_open:
            return after_open.strip(), ""
        reasoning, final = after_open.rsplit(QWEN_THINK_CLOSE, 1)
        return reasoning.strip(), final.strip()
    if QWEN_THINK_CLOSE in raw_text:
        # Qwen's chat template normally places the opening <think> token in
        # the prompt. Offline generation therefore returns only the reasoning
        # body, the closing token, and the final answer.
        reasoning, final = raw_text.rsplit(QWEN_THINK_CLOSE, 1)
        return reasoning.strip(), final.strip()

    if GEMMA_THINK_OPEN in raw_text:
        after_open = raw_text.split(GEMMA_THINK_OPEN, 1)[1]
        if GEMMA_THINK_CLOSE not in after_open:
            return after_open.strip(), ""
        reasoning, final = after_open.split(GEMMA_THINK_CLOSE, 1)
        return reasoning.strip(), final.strip()
    if GEMMA_THINK_CLOSE in raw_text:
        reasoning, final = raw_text.split(GEMMA_THINK_CLOSE, 1)
        return reasoning.strip(), final.strip()

    # Some inference backends return an already-parsed final response with
    # the reasoning channel removed. Treat that as a final answer rather than
    # incorrectly classifying it as an unterminated thought.
    return "", raw_text.strip()


def parse_total(final_text: str) -> int | None:
    matches = TOTAL_RE.findall(final_text)
    return int(matches[-1]) if matches else None


def parse_indexed_records(final_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, city, score in ENUMERATION_RE.findall(final_text):
        records.append(
            {
                "index": int(index),
                "city": city.strip(),
                "score": int(score),
            }
        )
    return records


def evaluate_generation(
    raw_text: str,
    *,
    prompt_mode: str,
    gold_pairs: list[dict[str, Any]],
    finish_reason: str | None,
) -> dict[str, Any]:
    reasoning, final = split_reasoning_and_final(
        raw_text,
        prompt_mode=prompt_mode,
    )
    predicted_count = parse_total(final)
    listed_records = (
        parse_indexed_records(final) if prompt_mode == "enumeration" else []
    )
    gold = [(str(pair["city"]), int(pair["score"])) for pair in gold_pairs]
    predicted = [
        (str(pair["city"]), int(pair["score"])) for pair in listed_records
    ]
    gold_set = set(gold)
    predicted_set = set(predicted)
    true_positive = len(gold_set & predicted_set)
    precision = true_positive / len(predicted_set) if predicted_set else None
    recall = true_positive / len(gold_set) if gold_set else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    return {
        "reasoning_text": reasoning,
        "final_text": final,
        "predicted_count": predicted_count,
        "parse_status": "ok" if predicted_count is not None else "parse_fail",
        "exact_count": predicted_count == len(gold),
        "signed_error": (
            predicted_count - len(gold) if predicted_count is not None else None
        ),
        "absolute_error": (
            abs(predicted_count - len(gold))
            if predicted_count is not None
            else None
        ),
        "normalized_absolute_error": (
            abs(predicted_count - len(gold)) / max(1, len(gold))
            if predicted_count is not None
            else None
        ),
        "listed_records": listed_records,
        "listed_total_matches_length": (
            predicted_count == len(listed_records)
            if prompt_mode == "enumeration" and predicted_count is not None
            else None
        ),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "duplicate_listed_pairs": len(predicted) - len(predicted_set),
        "hallucinated_pairs": sorted(predicted_set - gold_set),
        "missing_pairs": sorted(gold_set - predicted_set),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
    }
