from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from realistic_niah.prompts import build_messages, query_block
from realistic_niah_v4.prompts import V4_NUMERIC_QUERY_BLOCK

from .generation import build_v5_user_text


DATASET_SCHEMA_VERSION = "realistic_niah_count_mechanism_dataset_v1"
DATASET_PROTOCOL_VERSION = "v4_4_shared_backbone_non_native_pair_v1"
EXPECTED_SOURCE_STIMULI_SHA256 = (
    "da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9da5078f1b340aea440b2dcb51caa3c135873fb2daa9f53462fa4086b0b162ef"
)
MODES = ("non_thinking", "native_thinking")
SPLITS = ("discovery", "confirmation")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(dict(row), ensure_ascii=True, sort_keys=True).encode("utf-8")
        + b"\n"
        for row in rows
    )
    _write_bytes(path, payload)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def build_non_thinking_user_text(passage: str) -> str:
    messages = build_messages(str(passage), prompt_mode="direct")
    registered = query_block("direct")
    if len(messages) != 1 or not messages[0]["content"].endswith(registered):
        raise RuntimeError("Registered direct prompt layout changed")
    return messages[0]["content"][: -len(registered)] + V4_NUMERIC_QUERY_BLOCK


def _backbone_sha256(stimulus: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(dict(stimulus)))


def _validate_stimulus(row: Mapping[str, Any]) -> None:
    if row.get("design_variant") != "v4.4":
        raise ValueError("Mechanism dataset accepts V4.4 rows only")
    passage = str(row["passage"])
    if _sha256_bytes(passage.encode("utf-8")) != row["passage_sha256"]:
        raise ValueError(f"Passage SHA mismatch for {row.get('stimulus_id')}")
    count = int(row["gold_count"])
    spans = list(row["active_needle_spans"])
    gold = list(row["gold_pairs"])
    if len(spans) != count or len(gold) != count:
        raise ValueError(f"Needle/gold count mismatch for {row.get('stimulus_id')}")
    for span, pair in zip(spans, gold):
        start, end = int(span["char_start"]), int(span["char_end"])
        if not 0 <= start < end <= len(passage):
            raise ValueError(f"Invalid needle char span for {row.get('stimulus_id')}")
        if passage[start:end] != str(span["text"]):
            raise ValueError(f"Needle span text mismatch for {row.get('stimulus_id')}")
        if int(span["slot_index"]) != int(pair["slot_index"]):
            raise ValueError(f"Needle/gold slot mismatch for {row.get('stimulus_id')}")
        if str(span["city"]) != str(pair["city"]):
            raise ValueError(f"Needle/gold city mismatch for {row.get('stimulus_id')}")
        if int(span["score"]) != int(pair["score"]):
            raise ValueError(f"Needle/gold score mismatch for {row.get('stimulus_id')}")


def paired_record(stimulus: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown mechanism dataset mode: {mode}")
    _validate_stimulus(stimulus)
    source = dict(stimulus)
    source_schema = str(source.pop("schema_version"))
    source_protocol = str(source.pop("protocol_version"))
    passage = str(source["passage"])
    if mode == "non_thinking":
        user_text = build_non_thinking_user_text(passage)
        enable_thinking = False
        assistant_prefix = "Total:"
        expected_final_line = f"Total:{int(source['gold_count'])}"
        runtime_note = (
            "Apply the registered model chat template with native thinking disabled, "
            "then append assistant_prefix before scoring/generation."
        )
    else:
        user_text = build_v5_user_text(passage)
        enable_thinking = True
        assistant_prefix = ""
        expected_final_line = f"Total: {int(source['gold_count'])}"
        runtime_note = (
            "Apply the registered model chat template with native thinking enabled; "
            "do not inject a synthetic reasoning scaffold."
        )
    backbone = {
        "schema_version": source_schema,
        "protocol_version": source_protocol,
        **source,
    }
    pair_id = str(source["stimulus_id"])
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "dataset_mode": mode,
        "row_id": f"{mode}:{pair_id}",
        "pair_id": pair_id,
        "backbone_sha256": _backbone_sha256(backbone),
        "source_schema_version": source_schema,
        "source_protocol_version": source_protocol,
        "native_thinking": enable_thinking,
        "chat_template_thinking_enabled": enable_thinking,
        "user_text": user_text,
        "user_text_sha256": _sha256_bytes(user_text.encode("utf-8")),
        "messages": [{"role": "user", "content": user_text}],
        "assistant_prefix": assistant_prefix,
        "expected_final_line": expected_final_line,
        "runtime_note": runtime_note,
        **source,
    }


def audit_paired_records(
    non_thinking: Sequence[Mapping[str, Any]],
    native_thinking: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(non_thinking) != len(native_thinking):
        errors.append("mode row counts differ")
    pairs = []
    for left, right in zip(non_thinking, native_thinking):
        pair_id = str(left.get("pair_id"))
        if pair_id != str(right.get("pair_id")):
            errors.append(f"pair order/id mismatch at {pair_id}")
            continue
        if left.get("backbone_sha256") != right.get("backbone_sha256"):
            errors.append(f"backbone mismatch for {pair_id}")
        for key in (
            "passage",
            "passage_sha256",
            "gold_count",
            "gold_pairs",
            "slots",
            "active_needle_spans",
            "hard_negative_spans",
            "design",
            "seed",
            "split",
        ):
            if left.get(key) != right.get(key):
                errors.append(f"{key} differs for {pair_id}")
        passage = str(left.get("passage", ""))
        if str(left.get("user_text", "")).count(passage) != 1:
            errors.append(f"non-thinking prompt does not contain one passage: {pair_id}")
        if str(right.get("user_text", "")).count(passage) != 1:
            errors.append(f"native prompt does not contain one passage: {pair_id}")
        pairs.append((pair_id, str(left.get("backbone_sha256"))))
    return {
        "passed": not errors,
        "errors": errors,
        "paired_rows": len(pairs),
        "pair_backbone_sha256": _sha256_bytes(_canonical_json(pairs)),
        "all_passages_identical_between_modes": not any(
            "passage differs" in error for error in errors
        ),
        "all_needle_spans_identical_between_modes": not any(
            "active_needle_spans differs" in error for error in errors
        ),
        "all_gold_pairs_identical_between_modes": not any(
            "gold_pairs differs" in error for error in errors
        ),
    }


def _dataset_card(source_stimuli_sha256: str) -> str:
    return f"""---
pretty_name: Realistic NIAH Count Mechanism Analysis
task_categories:
- text-generation
language:
- en
size_categories:
- n<1K
configs:
- config_name: non_thinking
  default: true
  data_files:
  - split: discovery
    path: data/non_thinking/discovery.jsonl
  - split: confirmation
    path: data/non_thinking/confirmation.jsonl
- config_name: native_thinking
  data_files:
  - split: discovery
    path: data/native_thinking/discovery.jsonl
  - split: confirmation
    path: data/native_thinking/confirmation.jsonl
---

# Realistic NIAH count mechanism analysis

This release freezes the exact V4.4 stimulus backbone used by the non-thinking
mechanism experiments and exposes two paired prompt configurations:
`non_thinking` and `native_thinking`. Each configuration contains the same 300
V4.4 stimuli: 200 discovery rows (seeds 1234-1253) and 100 held-out
confirmation rows (seeds 1254-1263), with counts 1-10 balanced within each
seed.

The two configurations differ only in the task prompt and runtime thinking
control. Join them on `pair_id`. Equality of `passage`, `passage_sha256`,
`gold_pairs`, `slots`, `active_needle_spans`, `hard_negative_spans`, and
`design` is checked byte-canonically through `backbone_sha256` and recorded in
`AUDIT.json`.

## Frozen source

- Source protocol: `realistic_niah_v4_nonthinking_v3`
- Design variant: `v4.4`
- Canonical passage length: 10,000 Qwen3-8B tokenizer tokens
- Full source `stimuli.jsonl` SHA-256: `{source_stimuli_sha256}`
- Original full-grid audit: passed (1,200 rows across V4.1-V4.4)

The uploaded data contains only the 300 V4.4 rows. `source/` retains the
original audit and a V4.4-only excerpt of the source manifest.

This repository contains inputs only: no model generations, parsed traces,
hidden states, attention tables, representation outputs, causal results, or
mechanism result schemas are included.

## Prompt/runtime contract

- `non_thinking`: use `user_text` as the only user message, disable native
  thinking in the registered chat template, then append `assistant_prefix`
  (`Total:`) before scoring or generation.
- `native_thinking`: use the paired `user_text` as the only user message,
  enable the model's native thinking chat-template control, and do not inject
  a synthetic reasoning scaffold.

Rendered token IDs are intentionally not stored because they depend on the
model family and pinned tokenizer revision. Exact prompt-token needle spans
must be reconstructed from each row's half-open passage character spans after
applying the registered model chat template.

"""


def build_mechanism_dataset(
    *,
    source_dataset_dir: str | Path,
    output_dir: str | Path,
    repository_head: str | None = None,
) -> dict[str, Any]:
    source = Path(source_dataset_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stimuli_path = source / "stimuli.jsonl"
    manifest_path = source / "manifest.json"
    audit_path = source / "audit.json"
    stimuli_sha = sha256_file(stimuli_path)
    manifest_sha = sha256_file(manifest_path)
    if stimuli_sha != EXPECTED_SOURCE_STIMULI_SHA256:
        raise ValueError(
            f"Unexpected source stimuli SHA-256: {stimuli_sha}; "
            f"expected {EXPECTED_SOURCE_STIMULI_SHA256}"
        )
    if manifest_sha != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            f"Unexpected source manifest SHA-256: {manifest_sha}; "
            f"expected {EXPECTED_SOURCE_MANIFEST_SHA256}"
        )
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if source_audit.get("passed") is not True:
        raise ValueError("The original V4 source audit did not pass")
    stimuli = [
        row for row in load_jsonl(stimuli_path) if row.get("design_variant") == "v4.4"
    ]
    if len(stimuli) != 300:
        raise ValueError(f"Expected 300 frozen V4.4 rows, found {len(stimuli)}")
    if len({str(row["stimulus_id"]) for row in stimuli}) != len(stimuli):
        raise ValueError("V4.4 stimulus IDs are not unique")
    expected_seeds = set(range(1234, 1264))
    expected_counts = set(range(1, 11))
    if {int(row["seed"]) for row in stimuli} != expected_seeds:
        raise ValueError("V4.4 seed grid is not 1234-1263")
    if {int(row["gold_count"]) for row in stimuli} != expected_counts:
        raise ValueError("V4.4 count grid is not 1-10")
    cells = Counter((int(row["seed"]), int(row["gold_count"])) for row in stimuli)
    if set(cells.values()) != {1} or len(cells) != 300:
        raise ValueError("V4.4 seed/count grid is incomplete or duplicated")

    records = {
        mode: [paired_record(row, mode=mode) for row in stimuli] for mode in MODES
    }
    pair_audit = audit_paired_records(
        records["non_thinking"], records["native_thinking"]
    )
    if not pair_audit["passed"]:
        raise RuntimeError("Paired export audit failed: " + "; ".join(pair_audit["errors"]))

    data_paths: dict[str, Path] = {}
    for mode in MODES:
        for split in SPLITS:
            path = output / "data" / mode / f"{split}.jsonl"
            rows = [row for row in records[mode] if row["split"] == split]
            expected = 200 if split == "discovery" else 100
            if len(rows) != expected:
                raise ValueError(f"Expected {expected} rows for {mode}/{split}")
            _write_jsonl(path, rows)
            data_paths[f"{mode}/{split}"] = path

    source_excerpt = {
        **{key: value for key, value in source_manifest.items() if key != "families"},
        "families": [
            item
            for item in source_manifest["families"]
            if item.get("design_variant") == "v4.4"
        ],
        "source_stimuli_sha256": stimuli_sha,
        "source_manifest_sha256": manifest_sha,
        "exported_design_variant": "v4.4",
        "exported_rows": 300,
    }
    _write_json(output / "source" / "manifest_v4_4_excerpt.json", source_excerpt)
    _write_json(output / "source" / "original_audit.json", source_audit)
    _write_bytes(output / "README.md", _dataset_card(stimuli_sha).encode("utf-8"))

    file_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "realistic_niah_count_mechanism_manifest_v1",
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "repository_head": repository_head,
        "source": {
            "artifact": "exports/run_20260731_v4_numeric_presentation_v3/dataset",
            "stimuli_sha256": stimuli_sha,
            "manifest_sha256": manifest_sha,
            "original_rows": int(source_manifest["rows"]),
            "original_audit_passed": True,
            "exported_design_variant": "v4.4",
        },
        "configs": {
            mode: {
                "rows": len(records[mode]),
                "discovery_rows": sum(row["split"] == "discovery" for row in records[mode]),
                "confirmation_rows": sum(
                    row["split"] == "confirmation" for row in records[mode]
                ),
                "seeds": list(range(1234, 1264)),
                "counts": list(range(1, 11)),
            }
            for mode in MODES
        },
        "pair_audit": pair_audit,
        "files": file_hashes,
    }
    _write_json(output / "DATASET_MANIFEST.json", manifest)
    audit = {
        "schema_version": "realistic_niah_count_mechanism_audit_v1",
        "passed": True,
        "source_full_grid_audit_passed": True,
        "source_stimuli_sha256_verified": True,
        "source_manifest_sha256_verified": True,
        "v4_4_rows": 300,
        "non_thinking_rows": 300,
        "native_thinking_rows": 300,
        "discovery_rows_per_config": 200,
        "confirmation_rows_per_config": 100,
        **pair_audit,
    }
    _write_json(output / "AUDIT.json", audit)
    final_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checksum = "".join(f"{digest}  {name}\n" for name, digest in final_hashes.items())
    _write_bytes(output / "SHA256SUMS", checksum.encode("utf-8"))
    return {
        "output_dir": str(output),
        "manifest": str(output / "DATASET_MANIFEST.json"),
        "audit": audit,
        "data_files": {key: str(value) for key, value in data_paths.items()},
    }
