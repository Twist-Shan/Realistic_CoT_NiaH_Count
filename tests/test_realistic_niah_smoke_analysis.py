from __future__ import annotations

from realistic_niah.smoke_analysis import (
    summarize_guarded_smoke,
    summarize_overthinking_smoke,
)


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


def _guarded_row(
    *,
    model: str,
    stimulus: str,
    truncated: bool = False,
) -> dict:
    return {
        "request_id": f"{model}/native_thinking/{stimulus}",
        "model_label": model,
        "stimulus_id": stimulus,
        "prompt_mode": "native_thinking",
        "output_tokens": 800 if not truncated else 4096,
        "evaluation": {
            "registered_success": not truncated,
            "exact_count": not truncated,
            "truncated": truncated,
            "parse_status": "ok" if not truncated else "parse_fail",
            "response_format_compliant": not truncated,
            "overthinking_flag": truncated,
        },
    }


def test_guarded_smoke_gate_passes_for_complete_zero_truncation_results() -> None:
    models = ("Qwen3-8B", "Gemma4-12B")
    rows = [
        _guarded_row(model=model, stimulus=stimulus)
        for model in models
        for stimulus in ("s1", "s2")
    ]

    summary = summarize_guarded_smoke(
        rows,
        expected_models=models,
        expected_requests_per_model=2,
    )

    assert summary["gate"]["passed"] is True
    assert summary["gate"]["total_truncations"] == 0
    assert summary["models"]["Qwen3-8B"]["zero_truncations"] is True


def test_guarded_smoke_gate_fails_on_truncation_or_missing_model() -> None:
    rows = [
        _guarded_row(
            model="Qwen3-8B",
            stimulus="s1",
            truncated=True,
        )
    ]

    summary = summarize_guarded_smoke(
        rows,
        expected_models=("Qwen3-8B", "Gemma4-12B"),
        expected_requests_per_model=1,
    )

    assert summary["gate"]["passed"] is False
    assert summary["gate"]["total_truncations"] == 1
    assert summary["gate"]["missing_models"] == ["Gemma4-12B"]
