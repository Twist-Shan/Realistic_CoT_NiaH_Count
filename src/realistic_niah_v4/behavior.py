from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .prompts import PromptEncoding


def count_logit_metrics(
    logits: torch.Tensor | np.ndarray,
    encoding: PromptEncoding,
) -> dict[str, Any]:
    """Reduce one answer-query vocabulary vector to the registered count outcomes."""

    values = (
        logits.detach().float().cpu().numpy()
        if isinstance(logits, torch.Tensor)
        else np.asarray(logits, dtype=float)
    )
    if values.ndim != 1:
        raise ValueError("count_logit_metrics expects one vocabulary vector")
    candidates: list[tuple[int, int]] = []
    for count, token_ids in encoding.count_candidate_answer_token_ids:
        if len(token_ids) != 1:
            raise ValueError(
                "count_logit_metrics is only valid for distinct single-token "
                "answers; use joint sequence log-probabilities for numeric V4"
            )
        candidates.append((int(count), int(token_ids[0])))
    candidates.sort()
    if len({token_id for _, token_id in candidates}) != len(candidates):
        raise ValueError(
            "count_logit_metrics cannot distinguish candidates that share an "
            "initial token; use joint sequence log-probabilities"
        )
    counts = np.asarray([count for count, _ in candidates], dtype=float)
    token_ids = np.asarray([token_id for _, token_id in candidates], dtype=int)
    if int(token_ids.max()) >= len(values):
        raise ValueError("A count candidate token is outside the vocabulary")
    candidate_logits = values[token_ids].astype(float)
    shifted = candidate_logits - float(candidate_logits.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    correct_index = int(np.flatnonzero(counts == encoding.count)[0])
    other = np.delete(candidate_logits, correct_index)
    return {
        "gold_count": int(encoding.count),
        "predicted_count_among_candidates": int(counts[int(candidate_logits.argmax())]),
        "correct_count_logit": float(candidate_logits[correct_index]),
        "correct_count_margin": float(candidate_logits[correct_index] - other.max()),
        "correct_count_probability": float(probabilities[correct_index]),
        "expected_count": float(np.sum(probabilities * counts)),
        "candidate_counts": ",".join(str(int(value)) for value in counts),
        "candidate_logits": ",".join(
            f"{float(value):.9g}" for value in candidate_logits
        ),
        "candidate_probabilities": ",".join(
            f"{float(value):.9g}" for value in probabilities
        ),
    }
