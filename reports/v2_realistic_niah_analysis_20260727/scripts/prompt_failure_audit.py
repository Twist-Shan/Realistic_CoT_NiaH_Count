from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from eda_v2 import V2_ROOT, classify_error, source_registry


OUT = Path(__file__).resolve().parent / "prompt_failure_audit"


def compact_excerpt(text: str, head: int = 900, tail: int = 500) -> str:
    text = text.replace("\r", "").strip()
    if len(text) <= head + tail + 40:
        return text
    return text[:head].rstrip() + "\n\n[... middle omitted ...]\n\n" + text[-tail:].lstrip()


def has_total(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*Total\s*:\s*[-+]?\d+\s*$", text))


def has_explicit_list(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*(?:\d+[.)]|[-*])\s+[^\n]+", text))


def repeated_line_signal(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 16]
    if not lines:
        return False
    return len(lines) - len(set(lines)) >= 3


def truncation_subtype(item: dict, mode: str) -> str:
    ev = item["evaluation"]
    raw = str(item.get("raw_output_text") or "")
    total = has_total(raw)
    if mode == "direct":
        if total:
            return "Direct: Total present but output still hit budget"
        if has_explicit_list(raw):
            return "Direct: explicit list before Total; budget exhausted"
        return "Direct: explanation/scan before Total; budget exhausted"
    if mode.startswith("enumeration_"):
        repeat = (
            repeated_line_signal(raw)
            or int(ev.get("reasoning_duplicate_lines") or 0) >= 3
            or int(ev.get("reasoning_duplicate_record_mentions") or 0) >= 3
            or int(ev.get("duplicate_listed_pairs") or 0) >= 2
        )
        if repeat:
            return "Enumeration: repeated-item/list loop"
        if total:
            return "Enumeration: Total present but output still hit budget"
        return "Enumeration: unfinished long list before Total"
    signals = set(ev.get("overthinking_signals") or [])
    if int(ev.get("reasoning_enumeration_restart_count") or 0) > 0 or "enumeration_restart" in signals:
        return "Thinking: restart/rescan before final answer"
    if (
        int(ev.get("reasoning_duplicate_lines") or 0) >= 3
        or int(ev.get("reasoning_duplicate_record_mentions") or 0) >= 3
        or repeated_line_signal(raw)
    ):
        return "Thinking: repetition loop before final answer"
    if total:
        return "Thinking: final Total present but generation hit budget"
    return "Thinking: long scan/reasoning ended before Total"


def format_subtype(item: dict, mode: str, category: str) -> str:
    ev = item["evaluation"]
    raw = str(item.get("raw_output_text") or "")
    final = str(ev.get("final_text") or "")
    if category == "parse_failure":
        if mode.startswith("enumeration_") and has_explicit_list(raw):
            return "Listed records but omitted parsable Total"
        if mode == "native_thinking" and raw.lstrip().startswith("<|channel>thought"):
            return "Thought channel ended without parsable final Total"
        return "No parsable Total line"
    if category == "response_format_failure":
        if mode == "native_thinking" and raw.lstrip().startswith("<|channel>thought"):
            if re.fullmatch(r"\s*Total\s*:\s*[-+]?\d+\s*", final):
                return "Native final is one Total line; surrounding channel text failed strict gate"
            if has_explicit_list(final):
                return "Native final channel repeats/lists records before Total"
            if has_total(final):
                return "Native final channel adds prose before Total"
            return "Native thought/final channel has no compliant one-line Total"
        if has_total(raw):
            return "Extra prose or lines around an otherwise parsable Total"
        return "Non-Total response shape"
    if category == "enumeration_format_failure":
        status = str(ev.get("enumeration_format_status") or "unspecified")
        if status == "ok":
            if re.search(r"(?im)^\s*-\s*$", raw):
                return "Enumeration structure: bullet marker split from city-score line"
            if re.search(r"(?i)(missing|not provided|placeholder|\?\s*$)", raw):
                return "Enumeration structure: placeholder/non-record lines"
            if ev.get("listed_total_matches_length") is False:
                return "Enumeration structure: listed length disagrees with Total"
            return "Enumeration structure: extra/nonconforming line despite parsable list"
        status = status.replace("_", " ")
        return f"Enumeration structure: {status}"
    return category.replace("_", " ")


def load_rows() -> tuple[pd.DataFrame, list[dict]]:
    records: list[dict] = []
    full: list[dict] = []
    for source in source_registry():
        model = str(source["model"])
        mode = str(source["mode"])
        with Path(source["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                ev = item["evaluation"]
                category = classify_error(ev, mode)
                raw = str(item.get("raw_output_text") or "")
                subtype = "success"
                if category == "truncated":
                    subtype = truncation_subtype(item, mode)
                elif category in {
                    "parse_failure",
                    "response_format_failure",
                    "enumeration_format_failure",
                }:
                    subtype = format_subtype(item, mode, category)
                elif category == "wrong_count":
                    signed = ev.get("signed_error")
                    subtype = "Wrong count: undercount" if signed is not None and signed < 0 else "Wrong count: overcount"
                    if signed == 0:
                        subtype = "Correct count but another registered gate failed"
                record = {
                    "model": model,
                    "mode": mode,
                    "source_version": source["version"],
                    "request_id": item["request_id"],
                    "stimulus_id": item["stimulus_id"],
                    "N": item["num_needles"],
                    "L": item["target_passage_tokens"],
                    "primary_failure": category,
                    "failure_subtype": subtype,
                    "registered_success": bool(ev.get("registered_success")),
                    "exact_count": bool(ev.get("exact_count")),
                    "parse_ok": ev.get("parse_status") == "ok",
                    "format_ok": bool(ev.get("response_format_compliant")),
                    "enum_format_ok": bool(ev.get("enumeration_format_compliant")),
                    "truncated": bool(ev.get("truncated")),
                    "has_total": has_total(raw),
                    "output_tokens": int(ev.get("output_tokens") or item.get("output_tokens") or 0),
                    "reasoning_characters": int(ev.get("reasoning_characters") or 0),
                    "overthinking_flag": bool(ev.get("overthinking_flag")),
                    "restart_count": int(ev.get("reasoning_enumeration_restart_count") or 0),
                    "duplicate_reasoning_lines": int(ev.get("reasoning_duplicate_lines") or 0),
                    "duplicate_record_mentions": int(ev.get("reasoning_duplicate_record_mentions") or 0),
                    "pair_precision": ev.get("pair_precision"),
                    "pair_recall": ev.get("pair_recall"),
                    "pair_f1": ev.get("pair_f1"),
                    "listed_total_matches_length": ev.get("listed_total_matches_length"),
                    "duplicate_listed_pairs": int(ev.get("duplicate_listed_pairs") or 0),
                    "missing_pairs_count": len(ev.get("missing_pairs") or []),
                    "hallucinated_pairs_count": len(ev.get("hallucinated_pairs") or []),
                    "signed_error": ev.get("signed_error"),
                    "format_status": ev.get("enumeration_format_status"),
                    "raw_excerpt": compact_excerpt(raw),
                }
                records.append(record)
                full.append({"source": source, "item": item, "record": record})
    return pd.DataFrame(records), full


def cell_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, mode), group in data.groupby(["model", "mode"], sort=False):
        enum = group if mode.startswith("enumeration_") else group.iloc[0:0]
        rows.append(
            {
                "model": model,
                "mode": mode,
                "success_rate": group.registered_success.mean(),
                "exact_rate": group.exact_count.mean(),
                "format_rate": group.format_ok.mean(),
                "truncation_rate": group.truncated.mean(),
                "truncated_n": int(group.truncated.sum()),
                "truncated_without_Total_n": int((group.truncated & ~group.has_total).sum()),
                "exact_but_format_fail_n": int((group.exact_count & ~group.format_ok).sum()),
                "median_output_tokens": float(group.output_tokens.median()),
                "p90_output_tokens": float(group.output_tokens.quantile(0.9)),
                "p99_output_tokens": float(group.output_tokens.quantile(0.99)),
                "overthinking_flag_n": int(group.overthinking_flag.sum()),
                "restart_flag_n": int((group.restart_count > 0).sum()),
                "mean_pair_recall": float(enum.pair_recall.mean()) if len(enum) else np.nan,
                "mean_pair_precision": float(enum.pair_precision.mean()) if len(enum) else np.nan,
                "list_total_matches_length_rate": float(enum.listed_total_matches_length.fillna(False).mean()) if len(enum) else np.nan,
                "any_missing_pair_rate": float((enum.missing_pairs_count > 0).mean()) if len(enum) else np.nan,
                "any_hallucinated_pair_rate": float((enum.hallucinated_pairs_count > 0).mean()) if len(enum) else np.nan,
                "any_duplicate_pair_rate": float((enum.duplicate_listed_pairs > 0).mean()) if len(enum) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def exact_mcnemar(index_only: int, bullet_only: int) -> float:
    discordant = index_only + bullet_only
    if discordant == 0:
        return 1.0
    return float(2 * stats.binom.cdf(min(index_only, bullet_only), discordant, 0.5))


def paired_index_bullet(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in data.groupby("model"):
        index = group[group["mode"] == "enumeration_index"]
        bullet = group[group["mode"] == "enumeration_bullet"]
        if index.empty or bullet.empty:
            continue
        joined = index.merge(bullet, on="stimulus_id", suffixes=("_index", "_bullet"))
        si = joined.registered_success_index.astype(bool)
        sb = joined.registered_success_bullet.astype(bool)
        index_only = int((si & ~sb).sum())
        bullet_only = int((~si & sb).sum())
        rows.append(
            {
                "model": model,
                "n_pairs": len(joined),
                "index_success_rate": si.mean(),
                "bullet_success_rate": sb.mean(),
                "index_minus_bullet_pp": 100 * (si.mean() - sb.mean()),
                "index_only_success": index_only,
                "bullet_only_success": bullet_only,
                "both_success": int((si & sb).sum()),
                "neither_success": int((~si & ~sb).sum()),
                "mcnemar_exact_p": min(1.0, exact_mcnemar(index_only, bullet_only)),
                "index_mean_pair_recall": joined.pair_recall_index.mean(),
                "bullet_mean_pair_recall": joined.pair_recall_bullet.mean(),
                "index_mean_pair_precision": joined.pair_precision_index.mean(),
                "bullet_mean_pair_precision": joined.pair_precision_bullet.mean(),
                "index_total_matches_list_rate": joined.listed_total_matches_length_index.fillna(False).mean(),
                "bullet_total_matches_list_rate": joined.listed_total_matches_length_bullet.fillna(False).mean(),
                "index_truncation_rate": joined.truncated_index.mean(),
                "bullet_truncation_rate": joined.truncated_bullet.mean(),
            }
        )
    return pd.DataFrame(rows)


def select_examples(full: list[dict]) -> list[dict]:
    specs = [
        ("original_gemma_direct_trunc", "Gemma4-12B", "direct", "truncated", None),
        ("qwen17_bullet_loop", "Qwen3-1.7B", "enumeration_bullet", "truncated", "repeated-item/list loop"),
        ("qwen17_index_no_total", "Qwen3-1.7B", "enumeration_index", "parse_failure", "omitted parsable Total"),
        ("gemma12_native_verbose_final", "Gemma4-12B", "native_thinking", "response_format_failure", "final channel"),
        ("gemma12_native_trunc", "Gemma4-12B", "native_thinking", "truncated", None),
        ("deepseek_native_trunc", "DeepSeek-R1-0528-Qwen3-8B", "native_thinking", "truncated", None),
        ("deepseek_wrong_count", "DeepSeek-R1-0528-Qwen3-8B", "native_thinking", "wrong_count", None),
        ("glmz1_native_trunc", "GLM-Z1-9B-0414", "native_thinking", "truncated", None),
        ("gemma12_bullet_wrong", "Gemma4-12B", "enumeration_bullet", "wrong_count", None),
        ("gemma12_index_success", "Gemma4-12B", "enumeration_index", "success", None),
    ]
    chosen: list[dict] = []
    for key, model, mode, category, subtype_contains in specs:
        candidates = [
            x for x in full
            if x["record"]["model"] == model
            and x["record"]["mode"] == mode
            and x["record"]["primary_failure"] == category
            and (subtype_contains is None or subtype_contains.lower() in x["record"]["failure_subtype"].lower())
        ]
        if not candidates:
            continue
        # Prefer a concise representative unless the phenomenon is truncation, where the tail matters.
        candidates.sort(key=lambda x: len(str(x["item"].get("raw_output_text") or "")))
        chosen_item = candidates[len(candidates) // 2]
        record = chosen_item["record"]
        ev = chosen_item["item"]["evaluation"]
        chosen.append(
            {
                "key": key,
                "model": model,
                "mode": mode,
                "failure": category,
                "failure_subtype": record["failure_subtype"],
                "request_id": record["request_id"],
                "N": record["N"],
                "L": record["L"],
                "predicted_count": ev.get("predicted_count"),
                "exact_count": bool(ev.get("exact_count")),
                "format_ok": bool(ev.get("response_format_compliant")),
                "truncated": bool(ev.get("truncated")),
                "output_tokens": record["output_tokens"],
                "overthinking_signals": ev.get("overthinking_signals") or [],
                "raw_excerpt": compact_excerpt(str(chosen_item["item"].get("raw_output_text") or ""), 1200, 800),
            }
        )

    old_path = V2_ROOT / "shards" / "Gemma4-12B__direct" / "main" / "requests.jsonl"
    old_candidates = []
    with old_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            old_candidates.append(item)
    old_candidates.sort(key=lambda x: len(str(x.get("raw_output_text") or "")))
    item = old_candidates[len(old_candidates) // 2]
    ev = item["evaluation"]
    chosen.insert(
        0,
        {
            "key": "original_gemma_direct_trunc",
            "model": "Gemma4-12B (original V2 Direct)",
            "mode": "direct",
            "failure": "truncated",
            "failure_subtype": "Direct: explanation/list before Total; 64-token budget exhausted",
            "request_id": item["request_id"],
            "N": item["num_needles"],
            "L": item["target_passage_tokens"],
            "predicted_count": ev.get("predicted_count"),
            "exact_count": bool(ev.get("exact_count")),
            "format_ok": bool(ev.get("response_format_compliant")),
            "truncated": bool(ev.get("truncated")),
            "output_tokens": int(ev.get("output_tokens") or 0),
            "overthinking_signals": ev.get("overthinking_signals") or [],
            "raw_excerpt": compact_excerpt(str(item.get("raw_output_text") or ""), 1200, 400),
        },
    )
    # Drop the placeholder main-composite spec because old Direct is deliberately external.
    seen = set()
    unique = []
    for row in chosen:
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        unique.append(row)
    return unique


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, full = load_rows()
    if len(data) != 14_500:
        raise ValueError(f"expected 14,500 rows, got {len(data)}")
    data.to_csv(OUT / "request_level_failure_audit.csv", index=False)
    summary = (
        data[data.primary_failure != "success"]
        .groupby(["model", "mode", "primary_failure", "failure_subtype"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    summary["rate_of_500"] = summary["count"] / 500
    summary.to_csv(OUT / "failure_type_summary.csv", index=False)
    cell_diagnostics(data).to_csv(OUT / "cell_mechanism_diagnostics.csv", index=False)
    paired_index_bullet(data).to_csv(OUT / "paired_index_bullet.csv", index=False)
    examples = select_examples(full)
    (OUT / "phenomenon_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nPAIRED INDEX-BULLET")
    print(paired_index_bullet(data).to_string(index=False))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
