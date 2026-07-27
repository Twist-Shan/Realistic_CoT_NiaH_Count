"""Static and numerical audit for the rewritten core-results HTML report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
from PIL import Image


MODEL_EXPECTED = {
    "Qwen3-8B": 900,
    "Qwen3-1.7B": 900,
    "Qwen3-32B": 900,
    "Gemma4-E4B": 900,
    "Gemma4-12B": 900,
    "OLMo-Hybrid-7B": 600,
    "Llama3.1-8B": 600,
    "Llama3.2-3B": 600,
}
EXPECTED_TABLE_ROWS = {
    "core_accuracy_by_model_query.csv": 8,
    "core_accuracy_by_model_mode_query.csv": 21,
    "core_query_order_effects_paired.csv": 21,
    "core_mode_effects_paired.csv": 13,
    "core_failure_budget.csv": 42,
    "core_low_accuracy_diagnostics.csv": 5,
    "core_enumeration_aggregation.csv": 16,
    "core_enumeration_fidelity.csv": 16,
    "core_anomaly_flags.csv": 42,
    "core_anomaly_case_summary.csv": 10,
}


class ArtifactParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.math_count = 0
        self.method_boxes = 0
        self.conclusion_boxes = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "img" and values.get("src"):
            self.images.append(str(values["src"]))
        if tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))
        if tag == "math":
            self.math_count += 1
        classes = str(values.get("class", "")).split()
        self.method_boxes += int("method-box" in classes)
        self.conclusion_boxes += int("conclusion-box" in classes)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolve_local(root: Path, value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or value.startswith("#"):
        return None
    path_text = unquote(parsed.path)
    if not path_text:
        return None
    return (root / path_text.replace("/", str(Path("/")))).resolve()


def audit_manifest(root: Path) -> list[str]:
    failures: list[str] = []
    manifest = root / "SHA256SUMS.tsv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2:
                failures.append(f"malformed checksum row: {row}")
                continue
            expected, relative = row
            path = root / relative
            if not path.is_file():
                failures.append(f"missing checksum target: {relative}")
            elif digest(path) != expected:
                failures.append(f"checksum mismatch: {relative}")
    return failures


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_root.resolve()
    report_path = root / "report.html"
    html_text = report_path.read_text(encoding="utf-8")
    document = ArtifactParser()
    document.feed(html_text)

    failures: list[str] = []
    if html_text.count("CORE_RESULTS_REPORT_V3") != 2:
        failures.append("core-results marker count is not exactly two (body marker + footer)")
    expected_sections = {
        "answers",
        "setup",
        "query-order",
        "modes",
        "anomalies",
        "low-accuracy",
        "laws",
        "reproducibility",
    }
    if not expected_sections.issubset(document.ids):
        failures.append(f"missing section ids: {sorted(expected_sections - document.ids)}")
    if document.math_count != 3:
        failures.append(f"expected 3 MathML equations, found {document.math_count}")
    if len(document.images) != 6:
        failures.append(f"expected 6 embedded images, found {len(document.images)}")
    if document.method_boxes < 7 or document.conclusion_boxes < 7:
        failures.append("major scientific sections are missing method/conclusion boxes")
    for phrase in (
        "Direct 与 native thinking 使用的完整 task block",
        "Query-first 外层结构",
        "enable_thinking=false",
        "计算方法",
        "目前结论",
        "逐项机制诊断",
        "aggregation operator",
        "146/150",
        "r=0.916",
        "A10",
    ):
        if phrase not in html_text:
            failures.append(f"missing anomaly-report content: {phrase}")

    broken_links: list[str] = []
    image_failures: list[str] = []
    image_dimensions: dict[str, tuple[int, int]] = {}
    for source in document.images:
        path = resolve_local(root, source)
        if path is None or not path.is_file():
            image_failures.append(source)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_dimensions[source] = image.size
                if image.width < 900 or image.height < 450:
                    image_failures.append(f"{source}: suspicious dimensions {image.size}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            image_failures.append(f"{source}: {exc}")
    for link in document.links:
        parsed = urlsplit(link)
        if link.startswith("#"):
            if link[1:] not in document.ids:
                broken_links.append(link)
            continue
        path = resolve_local(root, link)
        if path is not None and not path.exists():
            broken_links.append(link)
    if image_failures:
        failures.append(f"image failures: {image_failures}")
    if broken_links:
        failures.append(f"broken local links: {broken_links}")

    requests = pd.read_csv(root / "tables" / "request_level_report.csv")
    if len(requests) != 6300 or requests["request_id"].nunique() != 6300:
        failures.append("request-level row or uniqueness check failed")
    if requests.groupby("model_label").size().to_dict() != MODEL_EXPECTED:
        failures.append("per-model request counts changed")

    table_counts: dict[str, int] = {}
    for name, expected in EXPECTED_TABLE_ROWS.items():
        frame = pd.read_csv(root / "tables" / name)
        table_counts[name] = len(frame)
        if len(frame) != expected:
            failures.append(f"{name}: expected {expected} rows, found {len(frame)}")

    model_query = pd.read_csv(root / "tables" / "core_accuracy_by_model_query.csv")
    grouped_model = requests.groupby("model_label", sort=False)["exact_correct"].agg(["sum", "size", "mean"])
    for _, row in model_query.iterrows():
        source = grouped_model.loc[row["model_label"]]
        if int(row["exact_correct"]) != int(source["sum"]):
            failures.append(f"exact count mismatch for {row['model_label']}")
        if int(row["requests"]) != int(source["size"]):
            failures.append(f"request count mismatch for {row['model_label']}")
        if not close(row["overall_accuracy"], source["mean"]):
            failures.append(f"overall accuracy mismatch for {row['model_label']}")
        for order in ("query_first", "query_last"):
            value = requests[
                requests["model_label"].eq(row["model_label"])
                & requests["query_order"].eq(order)
            ]["exact_correct"].mean()
            if not close(row[f"{order}_accuracy"], value):
                failures.append(f"{order} accuracy mismatch for {row['model_label']}")

    mode_query = pd.read_csv(root / "tables" / "core_accuracy_by_model_mode_query.csv")
    for _, row in mode_query.iterrows():
        subset = requests[
            requests["model_label"].eq(row["model_label"])
            & requests["prompt_mode"].eq(row["prompt_mode"])
        ]
        if not close(row["accuracy"], subset["exact_correct"].mean()):
            failures.append(
                f"mode accuracy mismatch for {row['model_label']} / {row['prompt_mode']}"
            )
        for order in ("query_first", "query_last"):
            order_subset = subset[subset["query_order"].eq(order)]
            if not close(row[f"{order}_accuracy"], order_subset["exact_correct"].mean()):
                failures.append(
                    f"mode-order mismatch for {row['model_label']} / "
                    f"{row['prompt_mode']} / {order}"
                )

    enum_fidelity = pd.read_csv(
        root / "tables" / "core_enumeration_fidelity.csv"
    )
    enumeration = requests[requests["prompt_mode"].eq("enumeration")].copy()
    enumeration["complete_retrieval"] = (
        enumeration["missing_pairs_n"].fillna(999).eq(0)
        & enumeration["hallucinated_pairs_n"].fillna(999).eq(0)
        & enumeration["duplicate_listed_pairs_n"].fillna(999).eq(0)
        & enumeration["listed_records_n"].eq(enumeration["gold_count"])
    )
    enumeration["wrong_total_after_complete"] = (
        enumeration["complete_retrieval"]
        & enumeration["exact_correct"].eq(0)
    )
    for _, row in enum_fidelity.iterrows():
        subset = enumeration[
            enumeration["model_label"].eq(row["model_label"])
            & enumeration["query_order"].eq(row["query_order"])
        ]
        if not close(
            row["exact_count_accuracy"], subset["exact_correct"].mean()
        ):
            failures.append(
                f"enumeration exact-count mismatch for "
                f"{row['model_label']} / {row['query_order']}"
            )
        if int(row["complete_retrieval_count"]) != int(
            subset["complete_retrieval"].sum()
        ):
            failures.append(
                f"enumeration complete-retrieval mismatch for "
                f"{row['model_label']} / {row['query_order']}"
            )
        if int(row["complete_retrieval_wrong_total_count"]) != int(
            subset["wrong_total_after_complete"].sum()
        ):
            failures.append(
                f"enumeration wrong-total mismatch for "
                f"{row['model_label']} / {row['query_order']}"
            )

    anomaly_flags = pd.read_csv(root / "tables" / "core_anomaly_flags.csv")
    expected_flag_types = {
        "query_order_shift",
        "mode_shift_vs_direct",
        "interface_failure",
        "aggregation_after_complete_retrieval",
    }
    if set(anomaly_flags["flag_type"]) != expected_flag_types:
        failures.append("anomaly flag families are incomplete")
    anomaly_cases = pd.read_csv(
        root / "tables" / "core_anomaly_case_summary.csv"
    )
    if set(anomaly_cases["anomaly_id"]) != {
        f"A{index}" for index in range(1, 11)
    }:
        failures.append("anomaly case IDs are incomplete")

    analysis_manifest = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
    core = analysis_manifest.get("core_results_report_v3")
    if not isinstance(core, dict):
        failures.append("core_results_report_v3 manifest entry is missing")
    else:
        if core.get("n_requests") != 6300 or core.get("registered_mode_rows") != 21:
            failures.append("core manifest counts are wrong")
        if core.get("anomaly_flag_rows") != 42 or core.get(
            "interpreted_anomaly_cases"
        ) != 10:
            failures.append("core manifest anomaly counts are wrong")
        for relative, metadata in core.get("artifacts", {}).items():
            path = root / relative
            if not path.is_file() or digest(path) != metadata.get("sha256"):
                failures.append(f"core manifest artifact mismatch: {relative}")

    checksum_failures = audit_manifest(root)
    if checksum_failures:
        failures.extend(checksum_failures)

    result = {
        "status": "PASS" if not failures else "FAIL",
        "request_rows": len(requests),
        "unique_request_ids": requests["request_id"].nunique(),
        "core_table_counts": table_counts,
        "html_bytes": report_path.stat().st_size,
        "embedded_images": len(document.images),
        "mathml_equations": document.math_count,
        "method_boxes": document.method_boxes,
        "conclusion_boxes": document.conclusion_boxes,
        "image_dimensions": image_dimensions,
        "broken_links": broken_links,
        "checksum_entries": sum(1 for _ in (root / "SHA256SUMS.tsv").open("r", encoding="utf-8")),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
