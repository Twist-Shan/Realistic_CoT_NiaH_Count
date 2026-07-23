from __future__ import annotations

from dataclasses import dataclass

CANONICAL_TOKENIZER = "Qwen/Qwen3-8B"
PASSAGE_LENGTHS = (2_000, 5_000, 10_000)
NEEDLE_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 20, 30)
SEEDS = (1234, 1235, 1236, 1237, 1238)
QUERY_ORDERS = ("query_first", "query_last")
THREE_PROMPT_MODES = ("direct", "enumeration", "native_thinking")
TWO_PROMPT_MODES = ("direct", "enumeration")

SMOKE_PASSAGE_LENGTHS = (2_000, 10_000)
SMOKE_NEEDLE_COUNTS = (5, 6, 30)
SMOKE_SEEDS = (1234,)

DECODING_CONTROL_PASSAGE_LENGTHS = (2_000, 10_000)
DECODING_CONTROL_NEEDLE_COUNTS = (5, 6, 20, 30)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    family: str
    native_thinking: bool
    prompt_modes: tuple[str, ...]


MODEL_SPECS = {
    spec.label: spec
    for spec in (
        ModelSpec(
            "Qwen3-1.7B",
            "Qwen/Qwen3-1.7B",
            "qwen3",
            True,
            THREE_PROMPT_MODES,
        ),
        ModelSpec(
            "Qwen3-8B",
            "Qwen/Qwen3-8B",
            "qwen3",
            True,
            THREE_PROMPT_MODES,
        ),
        ModelSpec(
            "Qwen3-32B",
            "Qwen/Qwen3-32B",
            "qwen3",
            True,
            THREE_PROMPT_MODES,
        ),
        ModelSpec(
            "Gemma4-E4B",
            "google/gemma-4-E4B-it",
            "gemma4",
            True,
            THREE_PROMPT_MODES,
        ),
        ModelSpec(
            "Gemma4-12B",
            "google/gemma-4-12B-it",
            "gemma4",
            True,
            THREE_PROMPT_MODES,
        ),
        ModelSpec(
            "Llama3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "llama",
            False,
            TWO_PROMPT_MODES,
        ),
        ModelSpec(
            "Llama3.2-3B",
            "meta-llama/Llama-3.2-3B-Instruct",
            "llama",
            False,
            TWO_PROMPT_MODES,
        ),
        ModelSpec(
            "OLMo-Hybrid-7B",
            "allenai/Olmo-Hybrid-Instruct-DPO-7B",
            "olmo_hybrid",
            False,
            TWO_PROMPT_MODES,
        ),
    )
}


def validate_experiment_spec() -> None:
    if 0 in NEEDLE_COUNTS:
        raise ValueError("N=0 is not part of the registered experiment")
    if len(NEEDLE_COUNTS) != 10:
        raise ValueError("The main grid must contain exactly 10 needle counts")
    if len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS) != 150:
        raise ValueError("The master grid must contain exactly 150 stimuli")


validate_experiment_spec()
