#!/usr/bin/env python3
"""Read-only integrity checks for the prompt-format report package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {".html", ".md", ".json", ".csv", ".tsv", ".py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def clean_passage(value: str) -> bool:
    matches = re.findall(
        r"<passage>\r?\n(.*?)\r?\n</passage>",
        value,
        flags=re.DOTALL,
    )
    return matches == ["[PASSAGE OMITTED]"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--source-script", type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()

    checksum_rows = []
    for line in (root / "SHA256SUMS.tsv").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, relative = line.split("\t", 1)
        checksum_rows.append((expected, relative))
    checksum_failures = [
        relative
        for expected, relative in checksum_rows
        if not (root / relative).is_file()
        or sha256(root / relative) != expected
    ]

    json_files = list(root.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))

    html_text = (root / "report.html").read_text(encoding="utf-8")
    image_refs = re.findall(r'<img[^>]+src="([^"]+)"', html_text)
    missing_images = [
        relative for relative in image_refs if not (root / relative).is_file()
    ]
    replacement_char_files = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and "\ufffd" in path.read_text(encoding="utf-8")
    ]

    summary_csv = root / "tables" / "model_prompt_format_summary.csv"
    examples_csv = root / "tables" / "model_prompt_format_examples.csv"
    prompt_json = (
        root / "prompt_formats" / "model_prompt_formats.json"
    )
    structured = json.loads(prompt_json.read_text(encoding="utf-8"))
    examples = structured["examples"]
    redaction_failures = [
        item["sample_request_id"]
        for item in examples
        if not clean_passage(item["user_message_redacted"])
        or not clean_passage(item["rendered_prompt_redacted"])
    ]

    manifest = json.loads(
        (root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    copied_script = (
        root / "scripts" / "build_prompt_format_addendum.py"
    )
    copied_script_matches = (
        args.source_script is None
        or sha256(copied_script) == sha256(args.source_script.resolve())
    )
    result = {
        "status": "pass",
        "checksum_entries": len(checksum_rows),
        "checksum_failures": checksum_failures,
        "json_files": len(json_files),
        "html_image_refs": len(image_refs),
        "missing_images": missing_images,
        "replacement_char_files": replacement_char_files,
        "section_present": 'id="prompt-formats"' in html_text,
        "nav_present": 'href="#prompt-formats"' in html_text,
        "html_combination_details": html_text.count(
            'class="prompt-example-combination"'
        ),
        "summary_rows": csv_rows(summary_csv),
        "example_rows": csv_rows(examples_csv),
        "json_examples": len(examples),
        "redaction_failures": redaction_failures,
        "output_root_matches": (
            Path(manifest["output_root"]).resolve() == root
        ),
        "manifest_prompt_status": manifest["prompt_formats_v1"][
            "source_audit_status"
        ],
        "build_log_present": (
            root / "logs" / "prompt_format_build_log.json"
        ).is_file(),
        "copied_script_matches": copied_script_matches,
    }
    failures = [
        bool(checksum_failures),
        bool(missing_images),
        bool(replacement_char_files),
        not result["section_present"],
        not result["nav_present"],
        result["html_combination_details"] != 42,
        result["summary_rows"] != 8,
        result["example_rows"] != 42,
        result["json_examples"] != 42,
        bool(redaction_failures),
        not result["output_root_matches"],
        result["manifest_prompt_status"] != "pass",
        not result["build_log_present"],
        not copied_script_matches,
    ]
    if any(failures):
        result["status"] = "fail"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
