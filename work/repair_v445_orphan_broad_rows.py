from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["seed"]),
        int(row["gold_count"]),
        str(row["condition"]),
        int(row["patch_layer"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--expected-heads", type=int, required=True)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    detail_rows = read_jsonl(args.detail)
    broad_rows = read_jsonl(args.broad)
    detail_keys = {group_key(row) for row in detail_rows}
    group_counts = Counter(group_key(row) for row in broad_rows)
    orphan_rows = [row for row in broad_rows if group_key(row) not in detail_keys]
    known_group_violations = {
        str(key): count
        for key, count in group_counts.items()
        if key in detail_keys and count != int(args.expected_heads)
    }
    report = {
        "detail_rows": len(detail_rows),
        "broad_rows_before": len(broad_rows),
        "orphan_rows": len(orphan_rows),
        "orphan_keys": sorted({str(group_key(row)) for row in orphan_rows}),
        "known_group_size_violations": known_group_violations,
        "apply_requested": bool(args.apply),
    }
    if known_group_violations:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("Refusing repair: a detail-backed broad group has wrong size")
    if not args.apply:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if not orphan_rows:
        raise SystemExit("Refusing repair: no orphan broad rows were found")
    if args.backup is None:
        raise SystemExit("--backup is required with --apply")
    if args.backup.exists():
        raise SystemExit(f"Refusing to overwrite backup: {args.backup}")
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.broad, args.backup)
    kept = [row for row in broad_rows if group_key(row) in detail_keys]
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{args.broad.name}.", suffix=".tmp", dir=args.broad.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, args.broad)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    report["broad_rows_after"] = len(kept)
    report["backup"] = str(args.backup)
    report["status"] = "repaired"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
