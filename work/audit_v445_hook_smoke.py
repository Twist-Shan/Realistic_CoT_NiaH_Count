from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


IGNORED_DETAIL_FIELDS = {
    "elapsed_seconds",
    "max_cuda_allocated_bytes",
    "max_cuda_reserved_bytes",
    "patch_hook_applications",
    "state_path",
    "strict_generation_reused_prefill",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def detail_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["condition"]), int(row["patch_layer"])


def compare_tree(left: Any, right: Any, path: str = "root") -> float:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"{path}: tensor/type mismatch")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{path}: tensor metadata mismatch")
        if left.numel() == 0:
            return 0.0
        delta = float((left.float() - right.float()).abs().max())
        if delta != 0.0:
            raise AssertionError(f"{path}: tensor max_abs_delta={delta}")
        return delta
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"{path}: mapping keys mismatch")
        return max(
            (compare_tree(left[key], right[key], f"{path}.{key}") for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"{path}: sequence mismatch")
        return max(
            (compare_tree(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right))),
            default=0.0,
        )
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args()

    old_rows = {detail_key(row): row for row in read_jsonl(args.old / "detail.jsonl")}
    new_rows = {detail_key(row): row for row in read_jsonl(args.new / "detail.jsonl")}
    if set(old_rows) != set(new_rows):
        raise AssertionError("detail keys differ")

    summaries: list[dict[str, Any]] = []
    max_state_delta = 0.0
    for key in sorted(old_rows):
        old_row = old_rows[key]
        new_row = new_rows[key]
        old_science = {k: v for k, v in old_row.items() if k not in IGNORED_DETAIL_FIELDS}
        new_science = {k: v for k, v in new_row.items() if k not in IGNORED_DETAIL_FIELDS}
        if old_science != new_science:
            raise AssertionError(f"{key}: scientific detail fields differ")
        if int(key[1]) >= 0:
            if int(old_row["patch_hook_applications"]) != 3:
                raise AssertionError(f"{key}: old hook count is not 3")
            if int(new_row["patch_hook_applications"]) != 2:
                raise AssertionError(f"{key}: new hook count is not 2")
        if bool(old_row["strict_generation_reused_prefill"]):
            raise AssertionError(f"{key}: old row unexpectedly reused prefill")
        if not bool(new_row["strict_generation_reused_prefill"]):
            raise AssertionError(f"{key}: new row did not reuse prefill")

        old_state = torch.load(args.old / old_row["state_path"], map_location="cpu", weights_only=False)
        new_state = torch.load(args.new / new_row["state_path"], map_location="cpu", weights_only=False)
        max_state_delta = max(max_state_delta, compare_tree(old_state, new_state, str(key)))
        old_elapsed = float(old_row["elapsed_seconds"])
        new_elapsed = float(new_row["elapsed_seconds"])
        summaries.append(
            {
                "condition": key[0],
                "layer": key[1],
                "old_seconds": old_elapsed,
                "new_seconds": new_elapsed,
                "time_reduction_fraction": 1.0 - new_elapsed / old_elapsed,
                "old_peak_allocated_bytes": int(old_row["max_cuda_allocated_bytes"]),
                "new_peak_allocated_bytes": int(new_row["max_cuda_allocated_bytes"]),
                "allocated_increase_bytes": int(new_row["max_cuda_allocated_bytes"])
                - int(old_row["max_cuda_allocated_bytes"]),
            }
        )

    old_broad = read_jsonl(args.old / "broad_metrics.jsonl")
    new_broad = read_jsonl(args.new / "broad_metrics.jsonl")
    old_broad_science = [
        {key: value for key, value in row.items() if key != "patch_hook_applications"}
        for row in old_broad
    ]
    new_broad_science = [
        {key: value for key, value in row.items() if key != "patch_hook_applications"}
        for row in new_broad
    ]
    if old_broad_science != new_broad_science:
        raise AssertionError("broad-metrics rows differ")

    print(
        json.dumps(
            {
                "status": "pass",
                "detail_rows": len(old_rows),
                "broad_rows": len(old_broad),
                "max_state_tensor_abs_delta": max_state_delta,
                "conditions": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
