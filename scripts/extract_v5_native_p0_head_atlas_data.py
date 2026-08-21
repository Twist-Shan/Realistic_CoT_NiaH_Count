#!/usr/bin/env python3
"""Extract exact P0 head maps and per-needle attention examples.

The grammar-specific head maps are read from the frozen causal-plan rankings.
The example matrices are recomputed at the same exact ``p0_item_end`` query
token used for ranking and ablation, so every cell is a raw attention mass to
one registered prompt-record span.  This script is intended to run on the
model host and writes a compact JSON artifact for the local HTML builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    load_registered_model,
    position_attention_outputs,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.causal import mechanism_continuations  # noqa: E402
from realistic_niah_v5.encoding import build_native_causal_encoding  # noqa: E402
from realistic_niah_v5.pipeline import (  # noqa: E402
    read_jsonl,
    registered_records,
)
from realistic_niah_v5.spec import V5Config  # noqa: E402


def _float_or_none(value: Any) -> float | None:
    text = str(value).strip()
    return float(text) if text else None


def _parse_head(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Head examples must be GRAMMAR:LAYER:HEAD"
        )
    grammar, layer, head = parts
    return grammar, int(layer), int(head)


def _grammar_from_pair(value: Any) -> str:
    return str(value).rsplit(" -> ", 1)[-1]


def _ranking_bundle(plan_root: Path, plan_k: int) -> dict[str, Any]:
    suffix = f"_p0_local_seed_event_k{int(plan_k)}_fullpanel_v1"
    rankings: dict[str, Any] = {}
    for plan_dir in sorted(plan_root.glob(f"causal_plan_*{suffix}")):
        grammar = plan_dir.name.removeprefix("causal_plan_").removesuffix(suffix)
        path = plan_dir / "crossfit_source_specific_head_ranking.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rankings[grammar] = {
            "plan_k": int(plan_k),
            "n_seeds": max((int(float(row["n_seeds"])) for row in rows), default=0),
            "rows": [
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "score": float(row["discovery_selection_value"]),
                    "rank": int(float(row["discovery_rank"])),
                    "n_seeds": int(float(row["n_seeds"])),
                    "mean_anchor_roles_per_seed": float(
                        row["mean_anchor_roles_per_seed"]
                    ),
                    "relative_attention_mass": _float_or_none(
                        row.get(
                            "discovery_target_source_relative_attention_mass",
                            "nan",
                        )
                    ),
                    "target_top1_rate": _float_or_none(
                        row.get("discovery_target_source_attention_top1", "nan")
                    ),
                }
                for row in rows
            ],
        }
    if not rankings:
        raise FileNotFoundError(
            f"No grammar-specific P0 K{plan_k} rankings under {plan_root}"
        )
    return rankings


def _eligible_tasks(
    rows: list[dict[str, Any]], tokenizer: Any
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    by_grammar: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        specifications, _excluded = mechanism_continuations(
            row,
            tokenizer,
            mechanism="retrieval_anchor_localization",
        )
        for specification in specifications:
            if "p0_item_end" not in {
                str(value) for value in specification.get("anchor_roles", [])
            }:
                continue
            if not bool(specification.get("event_specific")):
                continue
            if not bool(specification.get("local_anchor_eligible")):
                continue
            by_grammar[_grammar_from_pair(specification.get("grammar_pair"))].append(
                (row, specification)
            )
    return by_grammar


def _representative_request(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Choose deterministically: most events, then largest N, then lowest seed."""

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, specification in tasks:
        request_id = str(row.get("request_id", row.get("stimulus_id")))
        grouped[request_id].append((row, specification))
    if not grouped:
        raise ValueError("No eligible P0 tasks for requested grammar")

    def key(item: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> tuple[Any, ...]:
        request_id, values = item
        row = values[0][0]
        count = int(row.get("gold_count", len(row.get("gold_records", []))))
        seed = int(row["seed"])
        return (-len(values), -count, seed, request_id)

    _request_id, selected = min(grouped.items(), key=key)
    return sorted(
        selected,
        key=lambda value: (
            int(value[1]["from_occurrence"]),
            int(value[1]["query_output_token_index"]),
        ),
    )


def _record_masses(
    attention: Any,
    *,
    key_start: int,
    prompt_record_spans: Any,
) -> tuple[list[dict[str, Any]], float]:
    key_end = int(key_start) + int(attention.shape[-1])
    records: list[dict[str, Any]] = []
    needle_total = 0.0
    for source_index, span in enumerate(
        sorted(prompt_record_spans, key=lambda value: (int(value.start), int(value.end))),
        start=1,
    ):
        overlap_start = max(int(span.start), int(key_start))
        overlap_end = min(int(span.end), key_end)
        mass = (
            float(
                attention[
                    overlap_start - int(key_start) : overlap_end - int(key_start)
                ].sum().item()
            )
            if overlap_end > overlap_start
            else 0.0
        )
        needle_total += mass
        records.append(
            {
                "source_index": int(source_index),
                "city": str(span.city),
                "token_start": int(span.start),
                "token_end": int(span.end),
                "visible_token_count": max(0, overlap_end - overlap_start),
                "mass": mass,
            }
        )
    total_mass = float(attention.sum().item())
    return records, max(0.0, total_mass - needle_total)


def _capture_examples(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    tasks_by_grammar: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    examples: list[tuple[str, int, int]],
    max_queries: int,
) -> list[dict[str, Any]]:
    requested: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for grammar, layer, head in examples:
        if (layer, head) not in requested[grammar]:
            requested[grammar].append((layer, head))

    output: list[dict[str, Any]] = []
    for grammar, heads in requested.items():
        selected = _representative_request(tasks_by_grammar.get(grammar, []))
        if len(selected) > max_queries:
            indices = [
                round(index * (len(selected) - 1) / (max_queries - 1))
                for index in range(max_queries)
            ] if max_queries > 1 else [0]
            selected = [selected[index] for index in dict.fromkeys(indices)]
        row = selected[0][0]
        panels = {
            (layer, head): {
                "grammar": grammar,
                "layer": int(layer),
                "head": int(head),
                "request_id": str(row.get("request_id", row.get("stimulus_id"))),
                "seed": int(row["seed"]),
                "gold_count": int(
                    row.get("gold_count", len(row.get("gold_records", [])))
                ),
                "query_site": "p0_item_end",
                "events": [],
            }
            for layer, head in heads
        }
        for task_index, (_row, specification) in enumerate(selected, start=1):
            encoding = build_native_causal_encoding(
                _row,
                tokenizer,
                query_output_token_index=int(specification["query_output_token_index"]),
                sequence_output_token_end=int(specification["target_output_token_end"]),
                selected_site=specification,
            )
            rows, key_starts, _logits = position_attention_outputs(
                model,
                adapter,
                encoding,
                int(encoding.query_position),
            )
            for layer, head in heads:
                attention = rows[layer][head]
                record_rows, context_mass = _record_masses(
                    attention,
                    key_start=int(key_starts[layer]),
                    prompt_record_spans=encoding.prompt_record_spans,
                )
                target_city = str(specification["target_city"])
                for record in record_rows:
                    record["is_target"] = (
                        str(record["city"]).casefold() == target_city.casefold()
                    )
                panels[(layer, head)]["events"].append(
                    {
                        "event_index": int(task_index),
                        "from_occurrence": int(specification["from_occurrence"]),
                        "to_occurrence": int(specification["to_occurrence"]),
                        "query_output_token_index": int(
                            specification["query_output_token_index"]
                        ),
                        "query_full_sequence_token": int(encoding.query_position),
                        "query_token_text": str(specification.get("token_text", "")),
                        "target_city": target_city,
                        "records": record_rows,
                        "non_needle_context_mass": float(context_mass),
                        "attention_total_mass": float(attention.sum().item()),
                    }
                )
        output.extend(panels.values())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--plan-k", type=int, required=True)
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--head-example", action="append", type=_parse_head, default=[])
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.head_example:
        raise ValueError("At least one --head-example is required")
    if args.max_queries < 1:
        raise ValueError("--max-queries must be positive")

    config = V5Config.load(args.config)
    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    development = set(config.causal_development_seeds)
    generation_rows = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label=args.model
        )
        if int(row["seed"]) in development
    ]
    tasks_by_grammar = _eligible_tasks(generation_rows, tokenizer)
    bundle = {
        "schema_version": "realistic_niah_v5_p0_head_atlas_v1",
        "model_label": args.model,
        "query_site": "p0_item_end",
        "selection_split": "discovery",
        "selection_aggregation": "seed_event_mean",
        "selection_metric": "target_source_attention_mass",
        "development_seeds": sorted(development),
        "rankings": _ranking_bundle(args.plan_root, args.plan_k),
        "examples": _capture_examples(
            model,
            tokenizer,
            adapter,
            tasks_by_grammar=tasks_by_grammar,
            examples=args.head_example,
            max_queries=int(args.max_queries),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
