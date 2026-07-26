from __future__ import annotations

from dataclasses import dataclass

CANONICAL_TOKENIZER = "Qwen/Qwen3-8B"
PASSAGE_LENGTHS = (2_000, 3_000, 5_000, 10_000, 20_000)
NEEDLE_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 20, 30)
SEEDS = tuple(range(1234, 1244))
QUERY_LAYOUT = "cue_before_query_after"
FORMAL_PROMPT_MODES = (
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
)
NONTHINKING_PROMPT_MODES = FORMAL_PROMPT_MODES[:-1]
SMOKE_PROMPT_MODES = ("native_thinking_control", "native_thinking")
THINKING_PROMPT_MODES = frozenset(SMOKE_PROMPT_MODES)
ENUMERATION_PROMPT_MODES = frozenset(
    ("enumeration_index", "enumeration_bullet")
)

SMOKE_PASSAGE_LENGTHS = (2_000, 20_000)
SMOKE_NEEDLE_COUNTS = (6, 20, 30)
SMOKE_SEEDS = (1234, 1235)

DECODING_CONTROL_PASSAGE_LENGTHS = (2_000, 20_000)
DECODING_CONTROL_NEEDLE_COUNTS = (5, 6, 20, 30)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    family: str
    native_thinking: bool
    prompt_modes: tuple[str, ...]
    reasoning_policy: str = "switchable"


PRIMARY_MODEL_LABELS = (
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "DeepSeek-R1-0528-Qwen3-8B",
    "GLM-Z1-9B-0414",
)
MATCHED_CONTROL_MODEL_LABELS = ("GLM-4-9B-0414",)
MATCHED_NONTHINKING_CONTROLS = {
    "DeepSeek-R1-0528-Qwen3-8B": "Qwen3-8B",
    "GLM-Z1-9B-0414": "GLM-4-9B-0414",
}


MODEL_SPECS = {
    spec.label: spec
    for spec in (
        ModelSpec(
            "Qwen3-1.7B",
            "Qwen/Qwen3-1.7B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
        ),
        ModelSpec(
            "Qwen3-4B",
            "Qwen/Qwen3-4B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
        ),
        ModelSpec(
            "Qwen3-8B",
            "Qwen/Qwen3-8B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
        ),
        ModelSpec(
            "Qwen3-32B",
            "Qwen/Qwen3-32B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
        ),
        ModelSpec(
            "Gemma4-E4B",
            "google/gemma-4-E4B-it",
            "gemma4",
            True,
            FORMAL_PROMPT_MODES,
        ),
        ModelSpec(
            "Gemma4-12B",
            "google/gemma-4-12B-it",
            "gemma4",
            True,
            FORMAL_PROMPT_MODES,
        ),
        ModelSpec(
            "DeepSeek-R1-0528-Qwen3-8B",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "deepseek_r1_qwen3",
            True,
            FORMAL_PROMPT_MODES,
            "always_on",
        ),
        ModelSpec(
            "GLM-Z1-9B-0414",
            "zai-org/GLM-Z1-9B-0414",
            "glm_z1",
            True,
            FORMAL_PROMPT_MODES,
            "always_on",
        ),
        ModelSpec(
            "GLM-4-9B-0414",
            "zai-org/GLM-4-9B-0414",
            "glm4",
            False,
            NONTHINKING_PROMPT_MODES,
            "off_only",
        ),
    )
}


def validate_experiment_spec() -> None:
    if 0 in NEEDLE_COUNTS:
        raise ValueError("N=0 is not part of the registered experiment")
    if len(NEEDLE_COUNTS) != 10:
        raise ValueError("The main grid must contain exactly 10 needle counts")
    if len(PASSAGE_LENGTHS) != 5:
        raise ValueError("The V2 grid must contain exactly five passage lengths")
    if len(SEEDS) != 10:
        raise ValueError("The V2 grid must contain exactly ten seeds")
    if len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS) != 500:
        raise ValueError("The V2 master grid must contain exactly 500 stimuli")
    if len(PRIMARY_MODEL_LABELS) != 8:
        raise ValueError("The V2 primary panel must contain exactly eight models")
    if set(PRIMARY_MODEL_LABELS) | set(MATCHED_CONTROL_MODEL_LABELS) != set(
        MODEL_SPECS
    ):
        raise ValueError("Primary and matched-control labels must cover the registry")
    if any(
        target not in MODEL_SPECS or control not in MODEL_SPECS
        for target, control in MATCHED_NONTHINKING_CONTROLS.items()
    ):
        raise ValueError("Matched non-thinking controls must be registered")
    invalid_policies = {
        spec.reasoning_policy
        for spec in MODEL_SPECS.values()
        if spec.reasoning_policy
        not in {"switchable", "always_on", "off_only"}
    }
    if invalid_policies:
        raise ValueError(f"Invalid reasoning policies: {invalid_policies}")


validate_experiment_spec()
