"""Build a self-contained source snapshot plus frozen inputs; no checkpoints."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from protocol import sha256, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def bundle(frozen_roots: list[Path], output: Path):
    frozen_roots = [p.resolve() for p in frozen_roots]
    if output.exists():
        raise FileExistsError(output)
    paths = set((REPO / "src").rglob("*.py"))
    paths.update(p for p in ROOT.rglob("*") if p.is_file() and not
                 any(x in p.relative_to(ROOT).parts for x in ("runs", "bundles", "__pycache__", ".pytest_cache")))
    for frozen in frozen_roots:
        paths.update(p for p in frozen.rglob("*") if p.is_file())
    for p in REPO.glob("requirements*.txt"):
        paths.add(p)
    paths.add(REPO / "AGENTS.md")
    # Verify the entire legacy source snapshot agrees with each frozen run.
    for frozen in frozen_roots:
        expected = json.loads((frozen / "legacy_source_hashes.json").read_text(encoding="utf-8"))
        for rel, digest in expected.items():
            if sha256(REPO / rel.replace("\\", "/")) != digest:
                raise ValueError(f"Legacy source changed after freeze: {rel}")
    manifest = {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted(paths)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for p in sorted(paths):
            archive.add(p, arcname="repo_snapshot/" + p.relative_to(REPO).as_posix(), recursive=False)
    # Verify every archived regular file directly from the finished archive.
    import hashlib
    with tarfile.open(output, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if len(members) != len(manifest):
            raise ValueError("Archive file count mismatch")
        for member in members:
            name = member.name.removeprefix("repo_snapshot/")
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != manifest[name]:
                raise ValueError("Archive content hash mismatch")
    record = {"status": "PASS", "archive": str(output.resolve()), "archive_sha256": sha256(output),
              "bytes": output.stat().st_size, "file_count": len(manifest), "files": manifest,
              "checkpoints_included": False, "remote_upload_completed": False}
    write_json(output.with_suffix(output.suffix + ".manifest.json"), record)
    print(json.dumps({k: v for k, v in record.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle(args.frozen, args.output)
