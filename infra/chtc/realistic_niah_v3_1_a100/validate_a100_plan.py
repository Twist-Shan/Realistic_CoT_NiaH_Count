from __future__ import annotations

import csv
import sys
from pathlib import Path


SWITCHABLE = {
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "Gemma4-26B-A4B",
    "Gemma4-31B",
    "Nemotron-Nano-v2-9B",
    "Nemotron-3-Nano-4B",
}
CONTROL = {"GLM-4-9B-0414", "Ministral-3-Instruct-8B"}
REASONING = {"GLM-Z1-9B-0414", "Ministral-3-Reasoning-8B"}
FORMAL_MODES = (
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
)
CONTROL_MODES = FORMAL_MODES[:3]
REASONING_MODES = ("native_thinking",)
MODELS_80GB = {
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma4-12B",
    "Gemma4-26B-A4B",
    "Gemma4-31B",
    "Nemotron-Nano-v2-9B",
}
FIELDS = (
    "group_id",
    "tier",
    "gpu_memory_min_mb",
    "gpu_memory_max_mb",
    "job_memory_mb",
    "job_disk_mb",
    "worker0",
    "worker1",
    "worker2",
    "worker3",
)


def expected_tasks() -> set[tuple[str, str]]:
    tasks = {(model, mode) for model in SWITCHABLE for mode in FORMAL_MODES}
    tasks |= {(model, mode) for model in CONTROL for mode in CONTROL_MODES}
    tasks |= {(model, mode) for model in REASONING for mode in REASONING_MODES}
    return tasks


def main(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t", fieldnames=FIELDS))
    if len(rows) != 6:
        raise AssertionError(f"expected 6 four-GPU groups, got {len(rows)}")
    if len({row["group_id"] for row in rows}) != 6:
        raise AssertionError("group_id values must be unique")
    tiers = [row["tier"] for row in rows]
    if tiers.count("80gb") != 2 or tiers.count("40gb") != 4:
        raise AssertionError(f"expected 2x80GB and 4x40GB groups, got {tiers}")

    observed: list[tuple[str, str]] = []
    workers = 0
    for row in rows:
        if int(row["gpu_memory_min_mb"]) >= int(row["gpu_memory_max_mb"]):
            raise AssertionError(f"invalid memory interval in {row['group_id']}")
        for index in range(4):
            spec = row[f"worker{index}"]
            model, separator, modes_csv = spec.partition("|")
            if not separator or not model or not modes_csv:
                raise AssertionError(f"invalid worker spec: {spec!r}")
            should_be_80 = model in MODELS_80GB
            if should_be_80 != (row["tier"] == "80gb"):
                raise AssertionError(
                    f"tier mismatch for {model}: {row['tier']} in {row['group_id']}"
                )
            modes = modes_csv.split("+")
            if len(modes) != len(set(modes)):
                raise AssertionError(f"duplicate modes in {spec!r}")
            observed.extend((model, mode) for mode in modes)
            workers += 1

    if workers != 24:
        raise AssertionError(f"expected 24 GPU workers, got {workers}")
    duplicates = {task for task in observed if observed.count(task) > 1}
    if duplicates:
        raise AssertionError(f"duplicate logical shards: {sorted(duplicates)}")
    expected = expected_tasks()
    actual = set(observed)
    if actual != expected:
        raise AssertionError(
            f"task coverage mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    if len(actual) != 48:
        raise AssertionError(f"expected 48 logical shards, got {len(actual)}")
    print("validated: 6 jobs, 24 A100 workers, 48 logical shards exactly once")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "chtc_v31_a100_assignments.tsv"))
