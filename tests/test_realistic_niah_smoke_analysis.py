from __future__ import annotations

from realistic_niah.smoke_analysis import summarize_overthinking_smoke


def _row(
    *,
    stimulus: str,
    mode: str,
    exact: bool,
    truncated: bool,
    output_tokens: int,
    restarts: int,
) -> dict:
    return {
        "request_id": f"Qwen3-4B/{mode}/{stimulus}",
        "model_label": "Qwen3-4B",
        "stimulus_id": stimulus,
        "prompt_mode": mode,
        "output_tokens": output_tokens,
        "evaluation": {
            "registered_success": exact and not truncated,
            "exact_count": exact,
            "truncated": truncated,
            "reasoning_enumeration_restart_count": restarts,
            "reasoning_duplicate_record_mentions": restarts,
            "reasoning_duplicate_lines": restarts,
            "overthinking_flag": truncated or restarts > 0,
        },
    }


def test_smoke_summary_uses_paired_control_and_treatment() -> None:
    rows = [
        _row(
            stimulus="s1",
            mode="native_thinking_control",
            exact=False,
            truncated=True,
            output_tokens=4096,
            restarts=2,
        ),
        _row(
            stimulus="s1",
            mode="native_thinking",
            exact=True,
            truncated=False,
            output_tokens=700,
            restarts=0,
        ),
    ]

    summary = summarize_overthinking_smoke(rows)
    model = summary["models"]["Qwen3-4B"]

    assert model["paired_stimuli"] == 1
    assert model["metrics"]["registered_success"]["improved_pairs"] == 1
    assert model["metrics"]["exact_count"]["improved_pairs"] == 1
    assert model["metrics"]["truncated"]["improved_pairs"] == 1
    assert model["metrics"]["output_tokens"]["treatment_minus_control"] == -3396


def test_smoke_summary_rejects_incomplete_pairs() -> None:
    rows = [
        _row(
            stimulus="s1",
            mode="native_thinking",
            exact=True,
            truncated=False,
            output_tokens=700,
            restarts=0,
        )
    ]

    try:
        summarize_overthinking_smoke(rows)
    except ValueError as error:
        assert "Incomplete paired smoke result" in str(error)
    else:
        raise AssertionError("Expected incomplete smoke pair to fail")
