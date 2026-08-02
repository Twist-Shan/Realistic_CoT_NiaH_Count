from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realistic_niah_v3.scheduler import (
    allocate_pending_tasks,
    validate_resource_plan,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _visible_gpu_ids() -> tuple[int, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ids = tuple(
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip()
    )
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("Could not determine unique visible NVIDIA GPUs")
    return ids


def _marker_ids(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.tsv") if path.stat().st_size}


def _write_state(
    path: Path,
    *,
    status: str,
    visible_gpu_ids: tuple[int, ...],
    tasks: list[dict[str, Any]],
    running: dict[str, dict[str, Any]],
    completed: set[str],
    failed: set[str],
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "realistic_niah_v3_scheduler_state_v1",
            "updated_at_utc": _utc_now(),
            "hostname": socket.gethostname(),
            "scheduler_pid": os.getpid(),
            "status": status,
            "visible_gpu_ids": list(visible_gpu_ids),
            "task_count": len(tasks),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "pending_count": len(
                {str(task["task_id"]) for task in tasks}
                - completed
                - failed
                - set(running)
            ),
            "running": {
                task_id: {
                    key: value
                    for key, value in record.items()
                    if key != "process"
                }
                for task_id, record in sorted(running.items())
            },
            "completed_task_ids": sorted(completed),
            "failed_task_ids": sorted(failed),
        },
    )


def schedule(
    *,
    run_root: Path,
    repo_root: Path,
    max_gpus: int,
    poll_seconds: float,
) -> None:
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    plan_path = run_root / "orchestration" / "formal_shards.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = list(plan["tasks"])
    validate_resource_plan(tasks, maximum_supported_gpus=8)
    task_by_id = {str(task["task_id"]): task for task in tasks}
    all_task_ids = set(task_by_id)

    detected = _visible_gpu_ids()
    if not 1 <= max_gpus <= min(len(detected), 8):
        raise ValueError(
            "max_gpus must be within 1..min(visible NVIDIA GPUs, 8)"
        )
    visible = detected[:max_gpus]
    largest_task = max(int(task["gpus_required"]) for task in tasks)
    if max_gpus < largest_task:
        raise RuntimeError(
            f"This plan contains a {largest_task}-GPU task but only "
            f"{max_gpus} GPU(s) were assigned"
        )

    state_root = run_root / "orchestration" / "shard_state"
    completed_root = state_root / "completed"
    failed_root = state_root / "failed"
    completed_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "orchestration" / "scheduler_state.json"
    runner = repo_root / "scripts" / "run_realistic_niah_v3_worker.sh"
    if not runner.is_file():
        raise FileNotFoundError(runner)

    running: dict[str, dict[str, Any]] = {}
    scheduler_failed = False
    while True:
        completed = _marker_ids(completed_root)
        failed = _marker_ids(failed_root)
        unexpected = (completed | failed) - all_task_ids
        if unexpected:
            raise RuntimeError(
                f"Unexpected V3 task markers: {sorted(unexpected)}"
            )

        for task_id, record in list(running.items()):
            process = record["process"]
            return_code = process.poll()
            if return_code is None:
                continue
            del running[task_id]
            if return_code != 0:
                scheduler_failed = True
                print(
                    f"Task {task_id} exited with code {return_code}; "
                    "no new tasks will be launched.",
                    file=sys.stderr,
                    flush=True,
                )
            elif task_id not in _marker_ids(completed_root):
                scheduler_failed = True
                print(
                    f"Task {task_id} exited without a completion marker.",
                    file=sys.stderr,
                    flush=True,
                )

        completed = _marker_ids(completed_root)
        failed = _marker_ids(failed_root)
        if failed:
            scheduler_failed = True

        if not scheduler_failed:
            pending = [
                task
                for task in tasks
                if str(task["task_id"])
                not in completed | failed | set(running)
            ]
            busy = {
                int(gpu)
                for record in running.values()
                for gpu in record["gpu_ids"]
            }
            allocations = allocate_pending_tasks(
                pending,
                visible_gpu_ids=visible,
                busy_gpu_ids=busy,
            )
            for allocation in allocations:
                gpu_csv = ",".join(str(gpu) for gpu in allocation.gpu_ids)
                command = [
                    "bash",
                    str(runner),
                    str(run_root),
                    gpu_csv,
                    allocation.task_id,
                ]
                process = subprocess.Popen(command, cwd=repo_root)
                running[allocation.task_id] = {
                    "process": process,
                    "pid": process.pid,
                    "gpu_ids": list(allocation.gpu_ids),
                    "started_at_utc": _utc_now(),
                }
                print(
                    f"Launched {allocation.task_id} on GPU(s) {gpu_csv}",
                    flush=True,
                )

        completed = _marker_ids(completed_root)
        failed = _marker_ids(failed_root)
        if len(completed) == len(tasks) and not running and not failed:
            _write_state(
                state_path,
                status="completed",
                visible_gpu_ids=visible,
                tasks=tasks,
                running=running,
                completed=completed,
                failed=failed,
            )
            print(f"All {len(tasks)} V3 tasks completed.", flush=True)
            return
        if scheduler_failed and not running:
            _write_state(
                state_path,
                status="failed",
                visible_gpu_ids=visible,
                tasks=tasks,
                running=running,
                completed=completed,
                failed=failed,
            )
            raise RuntimeError(
                "The V3 scheduler stopped after a task failure; existing "
                "outputs and failure markers were preserved"
            )

        _write_state(
            state_path,
            status="draining_after_failure" if scheduler_failed else "running",
            visible_gpu_ids=visible,
            tasks=tasks,
            running=running,
            completed=completed,
            failed=failed,
        )
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resource-aware scheduler for up to eight H100 GPUs."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    schedule(
        run_root=Path(args.run_root),
        repo_root=Path(args.repo_root),
        max_gpus=args.max_gpus,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
