"""Reproducible multi-model Realistic NIAH counting experiments."""

from .spec import (
    FORMAL_PROMPT_MODES,
    MODEL_SPECS,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    QUERY_LAYOUT,
    SEEDS,
    SMOKE_NEEDLE_COUNTS,
    SMOKE_PASSAGE_LENGTHS,
    ModelSpec,
)

__all__ = [
    "MODEL_SPECS",
    "FORMAL_PROMPT_MODES",
    "NEEDLE_COUNTS",
    "PASSAGE_LENGTHS",
    "QUERY_LAYOUT",
    "SEEDS",
    "SMOKE_NEEDLE_COUNTS",
    "SMOKE_PASSAGE_LENGTHS",
    "ModelSpec",
]
