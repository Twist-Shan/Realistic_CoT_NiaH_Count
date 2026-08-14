from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["run_root"])
    reports = []
    passed = True
    for shard in manifest["shards"]:
        label = str(shard["label"])
        model = str(shard["model"])
        path = root / "shards" / label / model / "detail.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        keys = [
            (
                int(row["seed"]),
                int(row["gold_count"]),
                str(row["condition"]),
                int(row["patch_layer"]),
            )
            for row in rows
        ]
        expected = {
            (seed, count, condition, layer)
            for seed in map(int, shard["seeds"])
            for count in range(1, 11)
            for condition, layer in (
                [("clean", -1), ("needle_corrupt", -1), ("ordinary_corrupt", -1)]
                + [
                    (f"restore_{kind}", layer)
                    for layer in map(int, shard["layers"])
                    for kind in ("needle_endpoint", "needle_full", "ordinary_full")
                ]
            )
        }
        latest = rows[-1]
        shard_pass = len(keys) == len(set(keys)) and set(keys).issubset(expected)
        passed = passed and shard_pass
        reports.append(
            {
                "label": label,
                "rows": len(rows),
                "unique_keys": len(set(keys)),
                "invalid_keys": len(set(keys) - expected),
                "latest_key": list(keys[-1]),
                "latest_reused_prefill": latest.get("strict_generation_reused_prefill"),
                "latest_hook_applications": latest.get("patch_hook_applications"),
                "status": "PASS" if shard_pass else "FAIL",
            }
        )
    result = {"status": "PASS" if passed else "FAIL", "shards": reports}
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
