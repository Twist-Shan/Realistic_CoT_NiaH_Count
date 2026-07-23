from __future__ import annotations

from pathlib import Path

from realistic_niah.drive_sync import build_run_archive
from realistic_niah.runner import (
    _batched,
    _sampling_params_kwargs,
    build_requests,
    decoding_config,
)
from realistic_niah.spec import MODEL_SPECS


def _stimulus(index: int) -> dict:
    return {
        "stimulus_id": f"T2000_N6_seed{index}",
        "passage": f"passage {index}",
        "seed": index,
    }


def test_qwen_smoke_builds_36_unique_requests() -> None:
    requests = build_requests(
        [_stimulus(index) for index in range(6)],
        model_spec=MODEL_SPECS["Qwen3-8B"],
    )

    assert len(requests) == 36
    assert len({request["request_id"] for request in requests}) == 36


def test_llama_uses_only_four_conditions_per_stimulus() -> None:
    requests = build_requests(
        [_stimulus(1234)],
        model_spec=MODEL_SPECS["Llama3.1-8B"],
    )

    assert len(requests) == 4
    assert {request["prompt_mode"] for request in requests} == {
        "direct",
        "enumeration",
    }


def test_registered_decoding_budgets() -> None:
    qwen = MODEL_SPECS["Qwen3-8B"]

    assert decoding_config(qwen, "direct").max_tokens == 64
    assert decoding_config(qwen, "enumeration").max_tokens == 1536
    thinking = decoding_config(qwen, "native_thinking")
    assert thinking.max_tokens == 4096
    assert thinking.temperature == 0.6
    assert _sampling_params_kwargs(thinking, seed=1234)[
        "skip_special_tokens"
    ] is False


def test_request_batches_checkpoint_at_bounded_size() -> None:
    rows = [{"index": index} for index in range(7)]

    batches = list(_batched(rows, 3))

    assert [len(batch) for batch in batches] == [3, 3, 1]
    assert [item["index"] for batch in batches for item in batch] == list(range(7))


def test_archive_is_reproducibly_verifiable(tmp_path: Path) -> None:
    source = tmp_path / "run"
    source.mkdir()
    (source / "requests.jsonl").write_text('{"ok": true}\n', encoding="utf-8")

    metadata = build_run_archive(source, tmp_path / "archives" / "run.tar.gz")

    assert Path(metadata.archive_path).exists()
    assert metadata.size_bytes > 0
    assert len(metadata.sha256) == 64
    assert len(metadata.md5) == 32
