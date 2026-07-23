from __future__ import annotations

import argparse
import json

from realistic_niah.drive_sync import archive_and_sync_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive one completed run and verify its Google Drive copy."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument(
        "--remote-dir",
        default="gdrive:Realistic_CoT_NiaH_Count/realistic_niah_v1",
    )
    parser.add_argument("--rclone-binary", default="rclone")
    args = parser.parse_args()

    result = archive_and_sync_run(
        source_dir=args.source_dir,
        archive_dir=args.archive_dir,
        remote_dir=args.remote_dir,
        rclone_binary=args.rclone_binary,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
