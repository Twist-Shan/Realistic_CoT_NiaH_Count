from __future__ import annotations

from typing import Any

from .spec import MODEL_SPECS, ModelSpec, QUERY_ORDERS

DIRECT_TASK_BLOCK = """\
The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

How many city-score audit records are in the passage?
In the final answer, output exactly one line in this form:
Total: <integer>"""

ENUMERATION_TASK_BLOCK = """\
The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

Find every city-score audit record in the passage. In passage order, output
one record per line as:
<k>. <city>: <score>
where k starts at 1 and increases by 1.
Then output one final line:
Total: <integer>
Do not include any other text."""


def task_block(prompt_mode: str) -> str:
    if prompt_mode in {"direct", "native_thinking"}:
        return DIRECT_TASK_BLOCK
    if prompt_mode == "enumeration":
        return ENUMERATION_TASK_BLOCK
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def build_messages(
    passage: str,
    *,
    prompt_mode: str,
    query_order: str,
) -> list[dict[str, str]]:
    if query_order not in QUERY_ORDERS:
        raise ValueError(f"Unsupported query_order: {query_order}")
    task = task_block(prompt_mode)
    passage_block = f"<passage>\n{passage}\n</passage>"
    content = (
        f"{task}\n\n{passage_block}"
        if query_order == "query_first"
        else f"{passage_block}\n\n{task}"
    )
    return [{"role": "user", "content": content}]


def thinking_enabled(prompt_mode: str) -> bool:
    return prompt_mode == "native_thinking"


def resolve_model_spec(model: str) -> ModelSpec:
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]
    for spec in MODEL_SPECS.values():
        if spec.model_id == model:
            return spec
    raise ValueError(f"Unknown registered model: {model}")


def render_generation_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    model_spec: ModelSpec,
    prompt_mode: str,
) -> str:
    if prompt_mode not in model_spec.prompt_modes:
        raise ValueError(
            f"{model_spec.label} does not support prompt mode {prompt_mode!r}"
        )
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_spec.family in {"qwen3", "gemma4"}:
        kwargs["enable_thinking"] = thinking_enabled(prompt_mode)
    return tokenizer.apply_chat_template(messages, **kwargs)
