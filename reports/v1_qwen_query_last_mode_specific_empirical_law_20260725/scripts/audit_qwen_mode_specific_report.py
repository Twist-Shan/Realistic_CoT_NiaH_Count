"""Audit the Qwen query-last mode-specific empirical-law report v2."""

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


MODELS = ["Qwen3-1.7B", "Qwen3-8B", "Qwen3-32B"]
MODES = ["direct", "enumeration", "native_thinking"]
TARGETS = {"exact", "bias", "absolute_error"}
RULES = {"one_se", "best"}
EXPECTED_ROWS = {
    "qwen_query_last_requests.csv": 1350,
    "model_mode_summary.csv": 9,
    "accuracy_cells.csv": 270,
    "candidate_comparison.csv": 567,
    "candidate_fold_metrics.csv": 3402,
    "fixed_log_separable_laws.csv": 27,
    "selected_laws.csv": 27,
    "fixed_law_oof_predictions.csv": 4014,
    "fixed_law_bootstrap_draws.csv": 40500,
    "qwen_query_last_prompt_settings.csv": 9,
    "flex_candidate_registry.csv": 144,
    "flex_candidate_comparison.csv": 1296,
    "flex_selected_laws.csv": 27,
    "flex_outer_choices.csv": 270,
    "flex_nested_predictions.csv": 8028,
    "flex_goodness_of_fit.csv": 27,
}


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.images: list[str] = []
        self.links: list[str] = []
        self.math_count = 0
        self.method_boxes = 0
        self.conclusion_boxes = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_local(root: Path, value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or value.startswith("#"):
        return None
    path_text = unquote(parsed.path)
    return (root / Path(path_text)).resolve() if path_text else None


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tolerance
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_root.resolve()
    failures: list[str] = []

    report_path = root / "report.html"
    text = report_path.read_text(encoding="utf-8")
    document = ReportParser()
    document.feed(text)
    if text.count("QWEN_QUERY_LAST_MODE_SPECIFIC_V2") != 2:
        failures.append("report marker count is not two")
    expected_sections = {
        "summary",
        "setup",
        "evaluation",
        "observed",
        "accuracy",
        "bias",
        "search",
        "reproduce",
    }
    if not expected_sections.issubset(document.ids):
        failures.append(
            f"missing sections: {sorted(expected_sections - document.ids)}"
        )
    if len(document.images) != 5:
        failures.append(f"expected 5 images, found {len(document.images)}")
    if document.math_count != 29:
        failures.append(
            f"expected 29 MathML blocks, found {document.math_count}"
        )
    if document.method_boxes < 8 or document.conclusion_boxes < 8:
        failures.append("major sections lack method/conclusion boxes")
    for phrase in (
        "Query-last 外层结构",
        "enable_thinking",
        "Nested loss / null",
        "One-SE parsimonious",
        "Best predictive",
        "ceiling-limited",
        "leave-one-seed-out",
        "FDR",
        "未发现稳定统一 law",
    ):
        if phrase not in text:
            failures.append(f"missing required narrative phrase: {phrase}")

    image_dimensions: dict[str, tuple[int, int]] = {}
    for source in document.images:
        path = resolve_local(root, source)
        if path is None or not path.is_file():
            failures.append(f"missing image: {source}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_dimensions[source] = image.size
                if image.width < 900 or image.height < 450:
                    failures.append(
                        f"suspicious image dimensions {source}: {image.size}"
                    )
        except Exception as exc:
            failures.append(f"invalid image {source}: {exc}")

    broken_links: list[str] = []
    for value in document.links:
        if value.startswith("#"):
            if value[1:] not in document.ids:
                broken_links.append(value)
            continue
        path = resolve_local(root, value)
        if path is not None and not path.exists():
            broken_links.append(value)
    if broken_links:
        failures.append(f"broken local links: {broken_links}")

    table_counts: dict[str, int] = {}
    frames: dict[str, pd.DataFrame] = {}
    for filename, expected in EXPECTED_ROWS.items():
        path = root / "tables" / filename
        if not path.is_file():
            failures.append(f"missing table: {filename}")
            continue
        frame = pd.read_csv(path)
        frames[filename] = frame
        table_counts[filename] = len(frame)
        if len(frame) != expected:
            failures.append(
                f"{filename}: expected {expected} rows, found {len(frame)}"
            )

    for filename in (
        "flex_selected_bootstrap_draws.csv",
        "flex_selected_coefficient_intervals.csv",
    ):
        path = root / "tables" / filename
        if not path.is_file():
            failures.append(f"missing table: {filename}")
            continue
        frame = pd.read_csv(path)
        frames[filename] = frame
        table_counts[filename] = len(frame)
        if frame.empty:
            failures.append(f"empty table: {filename}")

    requests = frames.get("qwen_query_last_requests.csv")
    if requests is not None:
        if requests["request_id"].nunique() != 1350:
            failures.append("request IDs are not unique")
        if set(requests["model_label"]) != set(MODELS):
            failures.append("request model registry mismatch")
        if set(requests["prompt_mode"]) != set(MODES):
            failures.append("request prompt-mode registry mismatch")
        if set(requests["query_order"]) != {"query_last"}:
            failures.append("request subset is not query_last only")
        counts = requests.groupby(["model_label", "prompt_mode"]).size()
        if len(counts) != 9 or not counts.eq(150).all():
            failures.append("model x mode strata are not all 150 requests")
        grid = requests.groupby(
            [
                "model_label",
                "prompt_mode",
                "target_passage_tokens",
                "num_needles",
                "seed",
            ]
        ).size()
        if len(grid) != 1350 or not grid.eq(1).all():
            failures.append("3 x 10 x 5 request grids are incomplete")
        density = (
            1000
            * requests["num_needles"].to_numpy(float)
            / requests["target_passage_tokens"].to_numpy(float)
        )
        if not all(
            close(left, right)
            for left, right in zip(density, requests["density_per_1k"])
        ):
            failures.append("density recomputation mismatch")

    summary = frames.get("model_mode_summary.csv")
    if requests is not None and summary is not None:
        recomputed = (
            requests.groupby(["model_label", "prompt_mode"], as_index=False)
            .agg(
                requests=("request_id", "size"),
                exact_correct=("exact_correct", "sum"),
                exact_accuracy=("exact_correct", "mean"),
                parsed_requests=("parse_success", "sum"),
                parse_success_rate=("parse_success", "mean"),
            )
            .set_index(["model_label", "prompt_mode"])
        )
        saved = summary.set_index(["model_label", "prompt_mode"])
        for key, row in recomputed.iterrows():
            for column in (
                "requests",
                "exact_correct",
                "exact_accuracy",
                "parsed_requests",
                "parse_success_rate",
            ):
                if not close(row[column], saved.loc[key, column]):
                    failures.append(f"summary mismatch: {key}/{column}")
        expected_accuracy = {
            ("Qwen3-1.7B", "direct"): 52 / 150,
            ("Qwen3-1.7B", "enumeration"): 50 / 150,
            ("Qwen3-1.7B", "native_thinking"): 83 / 150,
            ("Qwen3-8B", "direct"): 64 / 150,
            ("Qwen3-8B", "enumeration"): 143 / 150,
            ("Qwen3-8B", "native_thinking"): 147 / 150,
            ("Qwen3-32B", "direct"): 72 / 150,
            ("Qwen3-32B", "enumeration"): 148 / 150,
            ("Qwen3-32B", "native_thinking"): 146 / 150,
        }
        for key, expected in expected_accuracy.items():
            actual = float(saved.loc[key, "exact_accuracy"])
            if not close(actual, expected):
                failures.append(
                    f"headline accuracy mismatch {key}: {actual} vs {expected}"
                )

    registry = frames.get("flex_candidate_registry.csv")
    if registry is not None:
        counts = registry.groupby("target")["spec"].nunique()
        if set(counts.index) != TARGETS or not counts.eq(48).all():
            failures.append("flex registry is not 48 specs per target")

    comparison = frames.get("flex_candidate_comparison.csv")
    if comparison is not None:
        counts = comparison.groupby(
            ["target", "model_label", "prompt_mode"]
        )["spec"].nunique()
        if len(counts) != 27 or not counts.eq(48).all():
            failures.append("candidate comparison coverage is incomplete")
        if not comparison["seed_oof_loss"].map(math.isfinite).all():
            failures.append("candidate comparison has non-finite losses")

    selected = frames.get("flex_selected_laws.csv")
    if selected is not None:
        if set(selected["target"]) != TARGETS:
            failures.append("selected-law target registry mismatch")
        for row in selected.itertuples():
            names = json.loads(row.coefficient_names)
            values = json.loads(row.coefficients)
            best_names = json.loads(row.best_coefficient_names)
            best_values = json.loads(row.best_coefficients)
            if len(names) != len(values) or len(best_names) != len(best_values):
                failures.append(
                    f"coefficient length mismatch: {row.model_label}/{row.prompt_mode}/{row.target}"
                )

    outer = frames.get("flex_outer_choices.csv")
    if outer is not None:
        if set(outer["selection_rule"]) != RULES:
            failures.append("outer-choice selection rules mismatch")
        counts = outer.groupby(
            [
                "target",
                "model_label",
                "prompt_mode",
                "selection_rule",
            ]
        )["outer_seed"].nunique()
        if len(counts) != 54 or not counts.eq(5).all():
            failures.append("outer-choice coverage is not 54 x 5")

    nested = frames.get("flex_nested_predictions.csv")
    if nested is not None:
        if set(nested["selection_rule"]) != RULES:
            failures.append("nested-prediction selection rules mismatch")
        if nested[
            ["observed", "nested_prediction", "nested_baseline"]
        ].isna().any().any():
            failures.append("nested predictions contain missing values")
        exact_nested = nested[nested["target"].eq("exact")]
        if len(exact_nested) != 2700:
            failures.append("exact nested prediction row count is not 2700")
        exact_counts = exact_nested.groupby("request_id").size()
        if (
            len(exact_counts) != 1350
            or not exact_counts.eq(2).all()
        ):
            failures.append("exact requests do not have both nested pipelines")
        if not exact_nested["nested_prediction"].between(0, 1).all():
            failures.append("exact nested probabilities outside [0,1]")

    goodness = frames.get("flex_goodness_of_fit.csv")
    if goodness is not None:
        if set(goodness["target"]) != TARGETS:
            failures.append("goodness target registry mismatch")
        for rule in ("parsimonious", "best"):
            if not goodness[f"{rule}_cell_cluster_gain_q"].between(0, 1).all():
                failures.append(f"{rule} q-values outside [0,1]")
            allowed = {
                "strong_generalizing",
                "strong_within_grid",
                "moderate",
                "weak",
                "ceiling_limited",
                "not_supported",
            }
            if not set(goodness[f"{rule}_evidence"]).issubset(allowed):
                failures.append(f"unknown {rule} evidence label")
        exact_gof = goodness[goodness["target"].eq("exact")]
        expected_ceiling = exact_gof["failure_count"].lt(10)
        if not exact_gof.loc[
            expected_ceiling, "best_evidence"
        ].eq("ceiling_limited").all():
            failures.append("low-event exact strata are not ceiling-limited")

    bootstrap = frames.get("flex_selected_bootstrap_draws.csv")
    if bootstrap is not None:
        counts = bootstrap.groupby(
            [
                "target",
                "model_label",
                "prompt_mode",
                "selection_rule",
                "coefficient",
            ]
        )["replicate"].nunique()
        if counts.empty or not counts.eq(500).all():
            failures.append("flex bootstrap coverage is not 500 per coefficient")

    intervals = frames.get("flex_selected_coefficient_intervals.csv")
    if intervals is not None:
        if not intervals["replicates"].eq(500).all():
            failures.append("coefficient interval replicate counts mismatch")
        if (intervals["ci95_low"] > intervals["ci95_high"]).any():
            failures.append("coefficient interval bounds are reversed")

    prompts = frames.get("qwen_query_last_prompt_settings.csv")
    if prompts is not None:
        if set(prompts["query_order"]) != {"query_last"}:
            failures.append("prompt table is not query_last only")
        if prompts["sample_full_rendered_prompt_sha256"].str.len().ne(64).any():
            failures.append("prompt SHA256 lengths are invalid")
        settings = {
            "direct": (64, 0.0, "false"),
            "enumeration": (1536, 0.0, "false"),
            "native_thinking": (4096, 0.6, "true"),
        }
        for mode, (tokens, temperature, thinking) in settings.items():
            part = prompts[prompts["prompt_mode"].eq(mode)]
            if (
                not part["max_tokens"].eq(tokens).all()
                or not part["temperature"].eq(temperature).all()
                or not part["enable_thinking_argument"]
                .astype(str)
                .str.lower()
                .eq(thinking)
                .all()
            ):
                failures.append(f"prompt decoding mismatch for {mode}")

    manifest_path = root / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "qwen_query_last_mode_specific_empirical_law_v2"
    ):
        failures.append("analysis manifest schema mismatch")
    if manifest.get("scope", {}).get("requests") != 1350:
        failures.append("analysis manifest request count mismatch")
    if (
        manifest.get("selection", {}).get("candidate_count_per_target")
        != 48
    ):
        failures.append("manifest candidate count mismatch")
    if manifest.get("raw_or_frozen_artifacts_modified") is not False:
        failures.append("manifest does not preserve frozen artifacts")

    checksum_path = root / "SHA256SUMS.tsv"
    checksum_entries = 0
    with checksum_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            checksum_entries += 1
            if len(row) != 2:
                failures.append(f"malformed checksum row: {row}")
                continue
            expected, relative = row
            path = root / relative
            if not path.is_file():
                failures.append(f"missing checksum target: {relative}")
            elif digest(path) != expected:
                failures.append(f"checksum mismatch: {relative}")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "report_bytes": report_path.stat().st_size,
        "request_rows": 0 if requests is None else len(requests),
        "table_rows": table_counts,
        "images": image_dimensions,
        "mathml_blocks": document.math_count,
        "method_boxes": document.method_boxes,
        "conclusion_boxes": document.conclusion_boxes,
        "checksum_entries": checksum_entries,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
