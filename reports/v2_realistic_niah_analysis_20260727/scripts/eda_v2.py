from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REALISTIC_ROOT = Path(
    r"C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting"
    r"\Realistic_CoT_NiaH_Count"
)
V2_ROOT = REALISTIC_ROOT / "exports" / "Realistic_CoT_NiaH_Count_20260726_v2"
V21_ROOT = (
    REALISTIC_ROOT
    / "exports"
    / "Realistic_CoT_NiaH_Count_20260726_v2_1_prompt_revision"
)
OUT = Path(__file__).resolve().parent / "eda_requests.csv"

BASE_MODELS = [
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
]
MODES = [
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
]


def source_registry() -> list[dict[str, str | Path]]:
    sources: list[dict[str, str | Path]] = []
    for model in BASE_MODELS:
        for mode in MODES:
            if mode.startswith("enumeration_"):
                path = (
                    V21_ROOT
                    / "shards"
                    / f"{model}__{mode}"
                    / "main"
                    / "requests.jsonl"
                )
                version = "v2.1_replacement"
            elif model == "Gemma4-12B" and mode == "direct":
                path = (
                    V21_ROOT
                    / "appendix"
                    / "Gemma4-12B"
                    / "direct_strict"
                    / "main"
                    / "requests.jsonl"
                )
                version = "v2.1_strict_direct"
            else:
                path = (
                    V2_ROOT
                    / "shards"
                    / f"{model}__{mode}"
                    / "main"
                    / "requests.jsonl"
                )
                version = "v2_formal"
            sources.append(
                {"model": model, "mode": mode, "version": version, "path": path}
            )

    sources.extend(
        [
            {
                "model": "DeepSeek-R1-0528-Qwen3-8B",
                "mode": "native_thinking",
                "version": "v2_formal",
                "path": V2_ROOT
                / "shards"
                / "DeepSeek-R1-0528-Qwen3-8B__native_thinking"
                / "main"
                / "requests.jsonl",
            },
            {
                "model": "GLM-Z1-9B-0414",
                "mode": "native_thinking",
                "version": "v2_formal",
                "path": V2_ROOT
                / "shards"
                / "GLM-Z1-9B-0414__native_thinking"
                / "main"
                / "requests.jsonl",
            },
        ]
    )
    for mode in ["direct", "enumeration_index", "enumeration_bullet"]:
        if mode.startswith("enumeration_"):
            path = (
                V21_ROOT
                / "shards"
                / f"GLM-4-9B-0414__{mode}"
                / "main"
                / "requests.jsonl"
            )
            version = "v2.1_replacement"
        else:
            path = (
                V2_ROOT
                / "shards"
                / f"GLM-4-9B-0414__{mode}"
                / "main"
                / "requests.jsonl"
            )
            version = "v2_formal"
        sources.append(
            {
                "model": "GLM-4-9B-0414",
                "mode": mode,
                "version": version,
                "path": path,
            }
        )
    return sources


def classify_error(ev: dict, mode: str) -> str:
    if ev.get("registered_success"):
        return "success"
    if ev.get("truncated") or ev.get("finish_reason") == "length":
        return "truncated"
    if ev.get("parse_status") != "ok":
        return "parse_failure"
    if not ev.get("response_format_compliant", False):
        if mode.startswith("enumeration_"):
            return "enumeration_format_failure"
        return "response_format_failure"
    if not ev.get("exact_count", False):
        return "wrong_count"
    return "other_failure"


def main() -> None:
    registry = source_registry()
    assert len(registry) == 29
    fields = [
        "model",
        "mode",
        "source_version",
        "request_id",
        "stimulus_id",
        "seed",
        "L",
        "N",
        "predicted_count",
        "signed_error",
        "absolute_error",
        "normalized_absolute_error",
        "parse_ok",
        "exact_count",
        "format_ok",
        "enumeration_format_ok",
        "registered_success",
        "truncated",
        "finish_reason",
        "output_tokens",
        "pair_precision",
        "pair_recall",
        "pair_f1",
        "listed_total_matches_length",
        "duplicate_listed_pairs",
        "missing_pairs_count",
        "hallucinated_pairs_count",
        "error_category",
        "final_text_excerpt",
        "raw_output_excerpt",
    ]
    counts: Counter[tuple[str, str]] = Counter()
    category_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in registry:
            path = Path(source["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as rows:
                for line in rows:
                    item = json.loads(line)
                    ev = item["evaluation"]
                    model = str(source["model"])
                    mode = str(source["mode"])
                    key = (model, mode)
                    counts[key] += 1
                    category = classify_error(ev, mode)
                    category_counts[key][category] += 1
                    final_text = str(ev.get("final_text") or "").replace("\r", " ")
                    raw_text = str(item.get("raw_output_text") or "").replace("\r", " ")
                    writer.writerow(
                        {
                            "model": model,
                            "mode": mode,
                            "source_version": source["version"],
                            "request_id": item["request_id"],
                            "stimulus_id": item["stimulus_id"],
                            "seed": item["seed"],
                            "L": item["target_passage_tokens"],
                            "N": item["num_needles"],
                            "predicted_count": ev.get("predicted_count"),
                            "signed_error": ev.get("signed_error"),
                            "absolute_error": ev.get("absolute_error"),
                            "normalized_absolute_error": ev.get(
                                "normalized_absolute_error"
                            ),
                            "parse_ok": ev.get("parse_status") == "ok",
                            "exact_count": ev.get("exact_count"),
                            "format_ok": ev.get("response_format_compliant"),
                            "enumeration_format_ok": ev.get(
                                "enumeration_format_compliant"
                            ),
                            "registered_success": ev.get("registered_success"),
                            "truncated": ev.get("truncated"),
                            "finish_reason": ev.get("finish_reason"),
                            "output_tokens": ev.get("output_tokens"),
                            "pair_precision": ev.get("pair_precision"),
                            "pair_recall": ev.get("pair_recall"),
                            "pair_f1": ev.get("pair_f1"),
                            "listed_total_matches_length": ev.get(
                                "listed_total_matches_length"
                            ),
                            "duplicate_listed_pairs": ev.get(
                                "duplicate_listed_pairs"
                            ),
                            "missing_pairs_count": len(ev.get("missing_pairs") or []),
                            "hallucinated_pairs_count": len(
                                ev.get("hallucinated_pairs") or []
                            ),
                            "error_category": category,
                            "final_text_excerpt": final_text[:500],
                            "raw_output_excerpt": raw_text[:1200],
                        }
                    )
    for key in sorted(counts):
        print(key, counts[key], dict(category_counts[key]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
