from __future__ import annotations

from realistic_niah.prompts import (
    build_messages,
    reasoning_expected,
    render_generation_prompt,
)
from realistic_niah.spec import MODEL_SPECS

COMMON = """\
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
PASSAGE
</passage>

"""

EXPECTED_SUFFIXES = {
    "direct": """\
How many city-score audit records are in the passage?
In the final answer, output exactly one line:
Total: <integer>""",
    "enumeration_index": """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order, as:
<k>. <city>: <score>
Then report the number listed:
Total: <integer>
Do not include any other text.""",
    "enumeration_bullet": """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order, as:
- <city>: <score>
Then report the number listed:
Total: <integer>
Do not include any other text.""",
    "native_thinking": """\
How many city-score audit records are in the passage?
Reason concisely. Do not restart or repeat a completed enumeration.
Once you determine the count, output exactly one final line:
Total: <integer>""",
}


def test_all_four_formal_prompts_match_registered_snapshots() -> None:
    for mode, suffix in EXPECTED_SUFFIXES.items():
        messages = build_messages("PASSAGE", prompt_mode=mode)

        assert messages == [
            {
                "role": "user",
                "content": COMMON + suffix,
            }
        ]


def test_smoke_control_only_removes_the_native_thinking_guard() -> None:
    control = build_messages(
        "PASSAGE",
        prompt_mode="native_thinking_control",
    )[0]["content"]
    direct = build_messages("PASSAGE", prompt_mode="direct")[0]["content"]
    treatment = build_messages(
        "PASSAGE",
        prompt_mode="native_thinking",
    )[0]["content"]

    assert control == direct
    assert control != treatment
    assert "Do not restart or repeat" not in control
    assert "Do not restart or repeat" in treatment


def test_smoke_control_and_treatment_both_enable_template_thinking() -> None:
    class RecordingTokenizer:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def apply_chat_template(
            self,
            messages: list[dict],
            **kwargs: object,
        ) -> str:
            self.calls.append(kwargs)
            return "rendered"

    tokenizer = RecordingTokenizer()
    model = MODEL_SPECS["Qwen3-4B"]
    messages = [{"role": "user", "content": "test"}]

    for mode in (
        "direct",
        "native_thinking_control",
        "native_thinking",
    ):
        render_generation_prompt(
            tokenizer,
            messages,
            model_spec=model,
            prompt_mode=mode,
        )

    assert [call["enable_thinking"] for call in tokenizer.calls] == [
        False,
        True,
        True,
    ]


def test_always_on_reasoning_models_use_their_native_templates_unchanged() -> None:
    class RecordingTokenizer:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def apply_chat_template(
            self,
            messages: list[dict],
            **kwargs: object,
        ) -> str:
            self.calls.append(kwargs)
            return "rendered"

    messages = [{"role": "user", "content": "test"}]
    for label in (
        "DeepSeek-R1-0528-Qwen3-8B",
        "GLM-Z1-9B-0414",
    ):
        tokenizer = RecordingTokenizer()
        model = MODEL_SPECS[label]
        for mode in (
            "direct",
            "native_thinking_control",
            "native_thinking",
        ):
            render_generation_prompt(
                tokenizer,
                messages,
                model_spec=model,
                prompt_mode=mode,
            )
            assert reasoning_expected(model, mode) is True

        assert all(
            "enable_thinking" not in call
            for call in tokenizer.calls
        )
