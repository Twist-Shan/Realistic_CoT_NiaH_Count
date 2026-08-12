#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.pipeline import (
    parse_records,
    read_jsonl,
    registered_records,
    write_jsonl,
)
from realistic_niah_v5.spec import V5Config


DEFAULT_CONFIG = ROOT / "configs" / "realistic_niah_v5.json"


def _config(args: argparse.Namespace) -> V5Config:
    return V5Config.load(args.config)


def _model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    spec = resolve_model_spec(args.model)
    return load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )


def command_parse(args: argparse.Namespace) -> None:
    rows = parse_records(read_jsonl(args.input), include_input=not args.compact)
    count = write_jsonl(args.output, rows)
    print(f"[v5 parse] wrote {count} rows to {args.output}")


def command_generate(args: argparse.Namespace) -> None:
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5.generation import generate_native_trace, render_native_prompt

    config = _config(args)
    spec = resolve_model_spec(args.model)
    model, tokenizer, _adapter = _model(args)
    stimuli = registered_records(
        read_jsonl(args.stimuli), config, model_label=args.model
    )
    output_rows = []
    for index, stimulus in enumerate(stimuli, start=1):
        prompt = render_native_prompt(stimulus, tokenizer=tokenizer, model_spec=spec)
        output_rows.append(
            generate_native_trace(
                model,
                tokenizer,
                prompt,
                decoding=config.decoding,
                sampling_seed=int(stimulus["seed"]),
            )
        )
        print(
            f"[v5 generate] {index}/{len(stimuli)} {prompt.stimulus_id} "
            f"parser={output_rows[-1]['trace_parse']['parser']['trace_category']}",
            flush=True,
        )
    count = write_jsonl(args.output, output_rows)
    print(f"[v5 generate] wrote {count} rows to {args.output}")


def command_capture(args: argparse.Namespace) -> None:
    from realistic_niah_v5.capture import capture_trace_shards

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    raw_rows = read_jsonl(args.generations)
    rows = (
        [
            row
            for row in raw_rows
            if str(row.get("model_label", row.get("model"))) == args.model
        ]
        if args.allow_unregistered
        else registered_records(raw_rows, config, model_label=args.model)
    )
    index = capture_trace_shards(
        model,
        adapter,
        tokenizer,
        rows,
        config=config,
        output_dir=args.output,
        layers=args.layers,
        overwrite=args.overwrite,
        site_ids=args.site_ids,
    )
    print(f"[v5 capture] index: {index}")


def command_attention(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.capture import capture_trace_attention_metrics

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    frames = []
    for index, row in enumerate(rows, start=1):
        frames.append(
            capture_trace_attention_metrics(
                model,
                adapter,
                tokenizer,
                row,
                config=config,
                mechanisms=tuple(args.mechanisms),
            )
        )
        print(f"[v5 attention] {index}/{len(rows)}", flush=True)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"[v5 attention] wrote {len(output)} rows to {args.output}")


def command_attention_pre_city(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.pre_city import (
        capture_pre_city_attention_metrics,
        write_pre_city_audit,
    )

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _parser_cohort_rows(rows, "one_to_one")
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    exclusion_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.csv.gz"
        exclusion_path = shard_root / f"{index:04d}__{safe_id}.exclusions.jsonl"
        if args.overwrite or not shard_path.exists():
            frame, row_exclusions = capture_pre_city_attention_metrics(
                model,
                adapter,
                tokenizer,
                row,
                depths=tuple(args.depths),
                include_anchor=not args.no_anchor,
            )
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(shard_path)
            write_jsonl(exclusion_path, row_exclusions)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        exclusion_paths.append(exclusion_path)
        print(
            f"[v5 attention-pre-city] {index}/{len(rows)} {action} {request_id}",
            flush=True,
        )
    frames = [pd.read_csv(path) for path in shard_paths]
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    exclusions = []
    for path in exclusion_paths:
        if path.exists():
            exclusions.extend(read_jsonl(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    exclusion_path = args.output.with_suffix(".exclusions.jsonl")
    write_jsonl(exclusion_path, exclusions)
    audit_path = args.output.with_suffix(".audit.json")
    write_pre_city_audit(output, exclusions, audit_path)
    print(
        f"[v5 attention-pre-city] wrote rows={len(output)} "
        f"exclusions={len(exclusions)} output={args.output}",
        flush=True,
    )


def command_attention_answer_query(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.capture import (
        capture_answer_query_attention_metrics,
    )

    model, tokenizer, adapter = _model(args)
    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    rows = _parser_cohort_rows(rows, args.cohort)
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.csv.gz"
        if args.overwrite or not shard_path.exists():
            frame = capture_answer_query_attention_metrics(
                model,
                adapter,
                tokenizer,
                row,
                site_id=args.site_id,
            )
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(shard_path)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 attention-answer-query] {index}/{len(rows)} "
            f"{action} {request_id}",
            flush=True,
        )
    frames = [pd.read_csv(path) for path in shard_paths]
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    audit = {
        "schema_version": "realistic_niah_v5_answer_query_extension_v1",
        "site_id": args.site_id,
        "query_definition": (
            "last literal Total: prefix before the first numeric answer token"
        ),
        "cohort": args.cohort,
        "split": args.split,
        "requests": len(rows),
        "rows": len(output),
        "restartable_shards": True,
        "confirmation_used_for_selection": False,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v5 attention-answer-query] wrote requests={len(rows)} "
        f"rows={len(output)} output={args.output}",
        flush=True,
    )


def command_representation(args: argparse.Namespace) -> None:
    from realistic_niah_v5.representation import analyze_representation

    paths = analyze_representation(
        args.capture_index,
        args.output,
        config=_config(args),
        cohorts=args.cohorts,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import build_causal_plan

    paths = build_causal_plan(args.attention, args.output, config=_config(args))
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_answer_query_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import build_answer_query_causal_plan

    paths = build_answer_query_causal_plan(
        args.attention,
        args.output,
        config=_config(args),
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_pre_city_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.pre_city import build_pre_city_causal_plan

    paths = build_pre_city_causal_plan(
        args.attention,
        args.output,
        config=_config(args),
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_response_reference_attention(args: argparse.Namespace) -> None:
    """Attach model-specific response types to exact reused pre-city d1 maps."""

    import pandas as pd

    from realistic_niah_v5.response_reference import (
        RESPONSE_REFERENCE_SCHEMA_VERSION,
        attach_response_reference_types,
    )

    generations = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    attention = pd.read_csv(args.pre_city_attention)
    attention = attention.loc[
        attention["model_label"].astype(str).eq(args.model)
    ].copy()
    enriched, exclusions = attach_response_reference_types(attention, generations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    write_jsonl(args.output.with_suffix(".exclusions.jsonl"), exclusions)
    counts = (
        enriched[["request_id", "occurrence", "split", "response_type"]]
        .drop_duplicates()
        .groupby(["split", "response_type"])
        .size()
        .to_dict()
    )
    audit = {
        "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
        "model_label": args.model,
        "source_attention": str(args.pre_city_attention.resolve()),
        "source_query_variants": [
            "pre_city_d1",
            "pre_city_d2",
            "pre_city_anchor",
        ],
        "query_position_reuse_audit": "PASS_IDENTICAL_SOURCE_PRE_CITY",
        "requests_in_generation_source": len(generations),
        "attention_rows": len(enriched),
        "registered_query_counts": {
            f"{split}:{response_type}": int(value)
            for (split, response_type), value in sorted(counts.items())
        },
        "exclusions": len(exclusions),
        "confirmation_used_for_selection": False,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_response_reference_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.response_reference import (
        build_response_reference_causal_plan,
    )

    paths = build_response_reference_causal_plan(
        args.attention, args.output, config=_config(args)
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def _parse_heads(value: str) -> list[tuple[int, int]]:
    parsed = json.loads(value)
    return [(int(layer), int(head)) for layer, head in parsed]


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    return [row for row in rows if str(row.get("split")) == split]


def _parser_cohort_rows(
    rows: list[dict[str, Any]], cohort: str
) -> list[dict[str, Any]]:
    from realistic_niah_v5.parsing import parse_trace_record

    selected = []
    for row in rows:
        parsed = parse_trace_record(row)
        parser = parsed["parser"]
        keep = bool(parser.get("detected"))
        if cohort in {"one_to_one", "one_to_one_correct"}:
            keep &= bool(parser.get("trace_one_to_one"))
        if cohort == "one_to_one_correct":
            keep &= bool(parsed.get("exact_count"))
        if keep:
            selected.append(row)
    return selected


def _row_site_ids(
    row: dict[str, Any], *, site_kind: str | None, site_id: str | None
) -> list[str]:
    if site_id is not None:
        return [site_id]
    if site_kind is None:
        raise ValueError("Either site_kind or site_id is required")
    from realistic_niah_v5.parsing import parse_trace_record

    parsed = parse_trace_record(row)
    return [
        str(site["site_id"])
        for site in parsed["char_sites"]
        if str(site["site_kind"]) == site_kind
    ]


def _safe_stem(*values: Any) -> str:
    readable = "__".join(str(value) for value in values)
    digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:12]
    return f"trial_{digest}"


def _atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _claim_restartable_task(
    shard_path: Path,
    *,
    worker_id: str,
    stale_seconds: float,
) -> Path | None:
    if shard_path.exists():
        return None
    claim = shard_path.with_suffix(shard_path.suffix + ".claim")
    if claim.exists():
        try:
            age = time.time() - claim.stat().st_mtime
            owner_dead = False
            try:
                owner = json.loads(claim.read_text(encoding="utf-8"))
                if str(owner.get("hostname")) == socket.gethostname():
                    owner_pid = int(owner["pid"])
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        owner_dead = True
                    except PermissionError:
                        owner_dead = False
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                owner_dead = False
            if (
                (owner_dead or age > float(stale_seconds))
                and not shard_path.exists()
            ):
                claim.unlink()
        except FileNotFoundError:
            pass
    try:
        with claim.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "worker_id": worker_id,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "claimed_unix": time.time(),
                    "target": str(shard_path),
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError:
        return None
    if shard_path.exists():
        claim.unlink(missing_ok=True)
        return None
    return claim


def _load_layer_bases(path: Path) -> dict[int, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        result = {
            int(key.removeprefix("basis_L")): np.asarray(archive[key], dtype=np.float32)
            for key in archive.files
            if key.startswith("basis_L")
        }
    if not result:
        raise ValueError(f"No basis_L<layer> arrays found in {path}")
    return result


def _load_layer_directions(path: Path) -> dict[int, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        result: dict[int, Any] = {}
        for key in archive.files:
            if key.startswith("direction_L"):
                result[int(key.removeprefix("direction_L"))] = np.asarray(
                    archive[key], dtype=np.float32
                )
            elif key.startswith("basis_L"):
                basis = np.asarray(archive[key], dtype=np.float32)
                if basis.ndim != 2 or basis.shape[1] < 1:
                    raise ValueError(f"Invalid direction basis {key} in {path}")
                result[int(key.removeprefix("basis_L"))] = basis[:, 0]
    if not result:
        raise ValueError(f"No direction_L<layer> or basis_L<layer> arrays in {path}")
    return result


def command_causal_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.causal import run_mechanism_head_ablation_trials

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    if not args.include_discovery:
        rows = [row for row in rows if row["split"] == "confirmation"]
    rows = _parser_cohort_rows(rows, args.cohort)
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].eq(args.model)].reset_index(drop=True)
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected)]
    output_rows = []
    for row_index, row in enumerate(rows, start=1):
        for mechanism in sorted(plan["mechanism"].astype(str).unique()):
            output_rows.extend(
                run_mechanism_head_ablation_trials(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    mechanism=mechanism,
                    heads=[],
                    condition="clean",
                    boundary_policy=args.boundary_policy,
                )
            )
        for plan_row in plan.itertuples(index=True):
            results = run_mechanism_head_ablation_trials(
                model,
                tokenizer,
                adapter,
                row,
                mechanism=str(plan_row.mechanism),
                heads=_parse_heads(plan_row.heads),
                condition=str(plan_row.condition),
                boundary_policy=args.boundary_policy,
            )
            for result in results:
                result["plan_row"] = int(plan_row.Index)
                result["repeat"] = int(plan_row.repeat)
                # Preserve the exact semantic needle-span attention evidence
                # for the discovery-frozen bank directly in every causal head
                # trial row.  These fields describe the bank selection, not a
                # post-intervention attention remeasurement.
                result["target_needle_raw_mass"] = float(
                    plan_row.target_needle_raw_mass
                )
                result["target_needle_relative_mass"] = float(
                    plan_row.target_needle_relative_mass
                )
                result["relative_mass_defined_heads"] = int(
                    plan_row.relative_mass_defined_heads
                )
                result["attention_mass_split"] = str(
                    plan_row.attention_mass_split
                )
                result["attention_mass_aggregation"] = str(
                    plan_row.attention_mass_aggregation
                )
            output_rows.extend(results)
        print(f"[v5 causal-heads] {row_index}/{len(rows)}", flush=True)
    write_jsonl(args.output, output_rows)


def command_causal_pre_city_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.pre_city import run_pre_city_head_ablation_trials

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    raw_rows = read_jsonl(args.generations)
    rows = (
        [
            row
            for row in raw_rows
            if str(row.get("model_label", row.get("model"))) == args.model
        ]
        if args.allow_unregistered
        else registered_records(raw_rows, config, model_label=args.model)
    )
    split = "all" if args.include_discovery else args.split
    rows = _split_rows(rows, split)
    rows = _parser_cohort_rows(rows, args.cohort)
    if args.gold_count is not None:
        from realistic_niah_v5.parsing import parse_trace_record

        rows = [
            row
            for row in rows
            if int(parse_trace_record(row)["gold_count"]) == int(args.gold_count)
        ]
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].eq(args.model)].reset_index(drop=True)
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected)]
    variants = set(plan["query_variant"].astype(str))
    if variants != {args.query_variant}:
        raise ValueError(
            f"Selected plan rows have variants={sorted(variants)}; "
            f"expected only {args.query_variant}"
        )
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for row_index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", row_index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{row_index:05d}__{safe_id}.jsonl"
        if args.overwrite or not shard_path.exists():
            trial_rows = run_pre_city_head_ablation_trials(
                model,
                tokenizer,
                adapter,
                row,
                query_variant=args.query_variant,
                heads=[],
                condition="clean",
            )
            for plan_row in plan.itertuples(index=True):
                results = run_pre_city_head_ablation_trials(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    query_variant=args.query_variant,
                    heads=_parse_heads(plan_row.heads),
                    condition=str(plan_row.condition),
                )
                for result in results:
                    result["plan_row"] = int(plan_row.Index)
                    result["repeat"] = int(plan_row.repeat)
                    result["mechanism"] = str(plan_row.mechanism)
                    for field in (
                        "target_needle_raw_mass",
                        "target_needle_relative_mass",
                        "confirmation_target_needle_raw_mass",
                        "confirmation_target_needle_relative_mass",
                    ):
                        result[field] = float(getattr(plan_row, field))
                    for field in (
                        "relative_mass_defined_heads",
                        "raw_mass_defined_heads",
                        "confirmation_raw_mass_defined_heads",
                        "confirmation_relative_mass_defined_heads",
                    ):
                        result[field] = int(getattr(plan_row, field))
                    result["attention_mass_split"] = str(
                        plan_row.attention_mass_split
                    )
                    result["confirmation_mass_split"] = str(
                        plan_row.confirmation_mass_split
                    )
                    result["attention_mass_aggregation"] = str(
                        plan_row.attention_mass_aggregation
                    )
                trial_rows.extend(results)
            _atomic_write_jsonl(shard_path, trial_rows)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 causal-pre-city-heads] {row_index}/{len(rows)} "
            f"{action} request={request_id} variant={args.query_variant}",
            flush=True,
        )
    output_rows = []
    for shard_path in shard_paths:
        output_rows.extend(read_jsonl(shard_path))
    _atomic_write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-pre-city-heads] requests={len(rows)} "
        f"rows={len(output_rows)} output={args.output}",
        flush=True,
    )


def command_causal_pre_city_all_sites(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.pre_city import (
        run_all_site_pre_city_damage_trial,
    )

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    raw_rows = read_jsonl(args.generations)
    if args.allow_unregistered:
        rows = [
            row
            for row in raw_rows
            if str(row.get("model_label", row.get("model"))) == args.model
        ]
    else:
        rows = registered_records(raw_rows, config, model_label=args.model)
    if not args.include_discovery:
        rows = [row for row in rows if row["split"] == "confirmation"]
    rows = _parser_cohort_rows(rows, args.cohort)
    if args.gold_count is not None:
        from realistic_niah_v5.parsing import gold_records

        rows = [
            row
            for row in rows
            if len(gold_records(row)) == int(args.gold_count)
        ]
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].eq(args.model)].reset_index(drop=True)
    if args.plan_rows:
        selected_rows = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected_rows)]
    else:
        plan = plan.loc[
            plan["query_variant"].astype(str).eq(args.query_variant)
        ]
    variants = set(plan["query_variant"].astype(str))
    if variants != {args.query_variant}:
        raise ValueError(
            f"Selected plan rows have variants={sorted(variants)}; "
            f"expected only {args.query_variant}"
        )
    worker_id = str(args.worker_id or f"{socket.gethostname()}-{os.getpid()}")
    shard_root = args.output.parent / f"{args.output.stem}_shards_v2"
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        request_id = str(
            row.get("request_id", row.get("stimulus_id", row_index))
        )
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        row_root = shard_root / f"{row_index:04d}__{safe_id}"
        tasks.append(
            {
                "task_id": f"{row_index:04d}:clean",
                "row_index": row_index,
                "request_id": request_id,
                "row": row,
                "plan": None,
                "shard": row_root / "clean.jsonl",
            }
        )
        for plan_index, plan_row in plan.iterrows():
            tasks.append(
                {
                    "task_id": f"{row_index:04d}:plan:{int(plan_index):04d}",
                    "row_index": row_index,
                    "request_id": request_id,
                    "row": row,
                    "plan": {**plan_row.to_dict(), "plan_row": int(plan_index)},
                    "shard": row_root / f"plan_{int(plan_index):04d}.jsonl",
                }
            )
    if args.overwrite:
        if worker_id != "primary":
            raise ValueError("Only worker_id=primary may initialize --overwrite")
        for task in tasks:
            Path(task["shard"]).unlink(missing_ok=True)
            Path(task["shard"]).with_suffix(".jsonl.claim").unlink(
                missing_ok=True
            )

    deadline = time.time() + float(args.worker_wait_timeout)
    completed_here = 0
    while True:
        missing = [task for task in tasks if not Path(task["shard"]).exists()]
        if not missing:
            break
        claimed_any = False
        for task in missing:
            shard_path = Path(task["shard"])
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            claim = _claim_restartable_task(
                shard_path,
                worker_id=worker_id,
                stale_seconds=float(args.claim_stale_seconds),
            )
            if claim is None:
                continue
            claimed_any = True
            try:
                plan_row = task["plan"]
                if plan_row is None:
                    result = run_all_site_pre_city_damage_trial(
                        model,
                        tokenizer,
                        adapter,
                        task["row"],
                        query_variant=args.query_variant,
                        heads=[],
                        condition="clean",
                    )
                else:
                    result = run_all_site_pre_city_damage_trial(
                        model,
                        tokenizer,
                        adapter,
                        task["row"],
                        query_variant=args.query_variant,
                        heads=_parse_heads(str(plan_row["heads"])),
                        condition=str(plan_row["condition"]),
                    )
                    result["plan_row"] = int(plan_row["plan_row"])
                    result["repeat"] = int(plan_row["repeat"])
                    for field in (
                        "target_needle_raw_mass",
                        "target_needle_relative_mass",
                        "confirmation_target_needle_raw_mass",
                        "confirmation_target_needle_relative_mass",
                    ):
                        result[field] = float(plan_row[field])
                    result["attention_mass_split"] = str(
                        plan_row["attention_mass_split"]
                    )
                    result["confirmation_mass_split"] = str(
                        plan_row["confirmation_mass_split"]
                    )
                    result["attention_mass_aggregation"] = str(
                        plan_row["attention_mass_aggregation"]
                    )
                result["restartable_task_id"] = str(task["task_id"])
                result["execution_worker_id"] = worker_id
                _atomic_write_jsonl(shard_path, [result])
                completed_here += 1
                print(
                    "[v5 causal-pre-city-all-sites] "
                    f"worker={worker_id} completed={completed_here} "
                    f"task={task['task_id']} request={task['request_id']} "
                    f"variant={args.query_variant}",
                    flush=True,
                )
            finally:
                claim.unlink(missing_ok=True)
        if all(Path(task["shard"]).exists() for task in tasks):
            break
        if time.time() > deadline:
            remaining = sum(
                not Path(task["shard"]).exists() for task in tasks
            )
            raise TimeoutError(
                f"Timed out waiting for {remaining}/{len(tasks)} shared tasks"
            )
        if not claimed_any:
            time.sleep(float(args.worker_poll_seconds))

    output_rows = []
    for task in tasks:
        shard_rows = read_jsonl(Path(task["shard"]))
        if len(shard_rows) != 1:
            raise ValueError(f"Task shard must contain one row: {task['shard']}")
        output_rows.extend(shard_rows)
    _atomic_write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-pre-city-all-sites] wrote requests={len(rows)} "
        f"rows={len(output_rows)} worker={worker_id} "
        f"completed_here={completed_here} output={args.output}",
        flush=True,
    )


def command_causal_marker_needle_patch(args: argparse.Namespace) -> None:
    from realistic_niah_v5.pre_city import (
        pre_city_token_queries,
        run_marker_needle_patch_trials,
    )

    model, tokenizer, adapter = _model(args)
    generations: dict[str, dict[str, Any]] = {}
    for path in args.generations:
        for row in read_jsonl(path):
            if str(row.get("model_label", row.get("model"))) != args.model:
                continue
            request_id = str(row.get("request_id", row.get("stimulus_id")))
            if request_id in generations:
                prior = generations[request_id]
                if int(prior["seed"]) != int(row["seed"]):
                    raise ValueError(f"Conflicting generation row: {request_id}")
                continue
            generations[request_id] = row
    pairs = [
        row
        for row in read_jsonl(args.pairs)
        if str(row["model_label"]) == args.model
        and (args.split == "all" or str(row["split"]) == args.split)
    ]
    if args.max_pairs is not None:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        raise ValueError("No marker-needle patch pairs matched the request")
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for pair_index, pair in enumerate(pairs, start=1):
        full_id = str(pair["full_request_id"])
        counterfactual_id = str(pair["counterfactual_request_id"])
        if full_id not in generations or counterfactual_id not in generations:
            raise ValueError(
                "Marker pair rows missing from generations: "
                f"{full_id}/{counterfactual_id}"
            )
        for variant in args.query_variants:
            for layer in args.layers:
                safe_pair = str(pair["pair_id"]).replace("/", "__").replace("\\", "__")
                shard = (
                    shard_root
                    / f"{pair_index:05d}__{safe_pair}__{variant}__L{int(layer):03d}.jsonl"
                )
                tasks.append((pair, variant, int(layer), shard))
    if args.overwrite:
        for _pair, _variant, _layer, shard in tasks:
            shard.unlink(missing_ok=True)
    completed = 0
    alias_cache: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for task_index, (pair, variant, layer, shard) in enumerate(tasks, start=1):
        full_id = str(pair["full_request_id"])
        query_candidates, _query_exclusions = pre_city_token_queries(
            generations[full_id], tokenizer
        )
        query_matches = [
            query
            for query in query_candidates
            if query.query_variant == variant
            and int(query.occurrence) == int(pair["occurrence"])
        ]
        if len(query_matches) != 1:
            raise ValueError(
                f"Cannot resolve marker alias for {pair['pair_id']}/{variant}"
            )
        alias_key = (
            str(pair["pair_id"]),
            int(layer),
            int(query_matches[0].query_output_token_count),
        )
        if shard.exists():
            rows = read_jsonl(shard)
            if len(rows) != 6:
                raise ValueError(f"Incomplete marker patch shard: {shard}")
            alias_cache.setdefault(alias_key, rows)
            action = "reused"
        elif alias_key in alias_cache:
            source_rows = alias_cache[alias_key]
            source_variant = str(source_rows[0]["query_variant"])
            rows = []
            for source in source_rows:
                row = dict(source)
                row["query_variant"] = variant
                row["query_alias_forward_reused"] = True
                row["query_alias_reused_from_variant"] = source_variant
                rows.append(row)
            _atomic_write_jsonl(shard, rows)
            completed += 1
            action = f"alias_reused_from_{source_variant}"
        else:
            counterfactual_id = str(pair["counterfactual_request_id"])
            rows = run_marker_needle_patch_trials(
                model,
                tokenizer,
                adapter,
                generations[full_id],
                generations[counterfactual_id],
                query_variant=variant,
                occurrence=int(pair["occurrence"]),
                layer=layer,
            )
            for row in rows:
                row["pair_id"] = str(pair["pair_id"])
                row["pair_eligibility"] = str(pair["pair_eligibility"])
                row["full_exact_count"] = bool(pair["full_exact_count"])
                row["counterfactual_exact_count"] = bool(
                    pair["counterfactual_exact_count"]
                )
                row["query_alias_forward_reused"] = False
                row["query_alias_reused_from_variant"] = None
            _atomic_write_jsonl(shard, rows)
            alias_cache[alias_key] = rows
            completed += 1
            action = "captured"
        print(
            f"[v5 causal-marker-needle-patch] {task_index}/{len(tasks)} "
            f"{action} variant={variant} L{layer} pair={pair['pair_id']}",
            flush=True,
        )
    output_rows = []
    for _pair, _variant, _layer, shard in tasks:
        shard_rows = read_jsonl(shard)
        if len(shard_rows) != 6:
            raise ValueError(f"Marker patch shard must contain six rows: {shard}")
        output_rows.extend(shard_rows)
    _atomic_write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-marker-needle-patch] pairs={len(pairs)} "
        f"tasks={len(tasks)} rows={len(output_rows)} completed_here={completed} "
        f"output={args.output}",
        flush=True,
    )


def command_causal_answer_query_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.causal import run_answer_query_head_ablation_trial

    model, tokenizer, adapter = _model(args)
    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    rows = _parser_cohort_rows(rows, args.cohort)
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].astype(str).eq(args.model)].reset_index(
        drop=True
    )
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected)]
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.jsonl"
        if args.overwrite or not shard_path.exists():
            trial_rows = [
                run_answer_query_head_ablation_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    heads=[],
                    condition="clean",
                    site_id=args.site_id,
                )
            ]
            for plan_row in plan.itertuples(index=True):
                result = run_answer_query_head_ablation_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    heads=_parse_heads(plan_row.heads),
                    condition=str(plan_row.condition),
                    site_id=args.site_id,
                )
                result["plan_row"] = int(plan_row.Index)
                result["mechanism"] = str(plan_row.mechanism)
                result["repeat"] = int(plan_row.repeat)
                for field in (
                    "target_needle_raw_mass",
                    "target_needle_relative_mass",
                    "selected_aggregation_raw_mass",
                    "selected_aggregation_relative_mass",
                    "confirmation_target_needle_raw_mass",
                    "confirmation_target_needle_relative_mass",
                    "confirmation_selected_aggregation_raw_mass",
                    "confirmation_selected_aggregation_relative_mass",
                    "selected_aggregation_broad_score",
                    "selected_aggregation_broad_coverage",
                    "confirmation_selected_aggregation_broad_score",
                    "confirmation_selected_aggregation_broad_coverage",
                ):
                    value = getattr(plan_row, field)
                    if value is None or (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        continue
                    result[field] = float(value)
                for field in (
                    "selected_head_count",
                    "prompt_bank_size",
                    "trace_bank_size",
                    "prompt_trace_head_overlap",
                    "prompt_aggregation_raw_mass",
                    "prompt_aggregation_relative_mass",
                    "trace_aggregation_raw_mass",
                    "trace_aggregation_relative_mass",
                    "prompt_aggregation_broad_score",
                    "prompt_aggregation_broad_coverage",
                    "trace_aggregation_broad_score",
                    "trace_aggregation_broad_coverage",
                    "confirmation_prompt_aggregation_raw_mass",
                    "confirmation_prompt_aggregation_relative_mass",
                    "confirmation_trace_aggregation_raw_mass",
                    "confirmation_trace_aggregation_relative_mass",
                    "confirmation_prompt_aggregation_broad_score",
                    "confirmation_prompt_aggregation_broad_coverage",
                    "confirmation_trace_aggregation_broad_score",
                    "confirmation_trace_aggregation_broad_coverage",
                ):
                    if hasattr(plan_row, field):
                        value = getattr(plan_row, field)
                        # Prompt-only and trace-only plan rows intentionally
                        # leave the joint-bank audit columns empty.  Pandas
                        # represents those optional CSV cells as NaN; omit
                        # them from the heterogeneous JSONL row instead of
                        # attempting int(NaN) or emitting non-standard NaN.
                        if value is None or (
                            isinstance(value, float) and math.isnan(value)
                        ):
                            continue
                        result[field] = (
                            int(value)
                            if field in {
                                "selected_head_count",
                                "prompt_bank_size",
                                "trace_bank_size",
                                "prompt_trace_head_overlap",
                            }
                            else float(value)
                        )
                result["attention_mass_split"] = str(
                    plan_row.attention_mass_split
                )
                result["confirmation_mass_split"] = str(
                    plan_row.confirmation_mass_split
                )
                result["attention_mass_aggregation"] = str(
                    plan_row.attention_mass_aggregation
                )
                result["selected_aggregation_metric"] = str(
                    plan_row.selected_aggregation_metric
                )
                trial_rows.append(result)
            write_jsonl(shard_path, trial_rows)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 causal-answer-query-heads] {index}/{len(rows)} "
            f"{action} {request_id}",
            flush=True,
        )
    output_rows = []
    for path in shard_paths:
        output_rows.extend(read_jsonl(path))
    write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-answer-query-heads] wrote requests={len(rows)} "
        f"rows={len(output_rows)} output={args.output}",
        flush=True,
    )


def command_causal_response_reference_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.response_reference import (
        run_response_reference_head_ablation_trials,
    )

    model, tokenizer, adapter = _model(args)
    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    rows = _parser_cohort_rows(rows, args.cohort)
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].astype(str).eq(args.model)].reset_index(
        drop=True
    )
    if args.response_types:
        allowed = {str(value) for value in args.response_types}
        plan = plan.loc[plan["response_type"].astype(str).isin(allowed)]
    if args.position_variants:
        allowed_positions = {str(value) for value in args.position_variants}
        plan = plan.loc[
            plan["position_variant"].astype(str).isin(allowed_positions)
        ]
    if args.bank_scopes:
        allowed_scopes = {str(value) for value in args.bank_scopes}
        plan = plan.loc[plan["bank_scope"].astype(str).isin(allowed_scopes)]
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected)]
    if plan.empty:
        raise ValueError("No response-reference plan rows were selected")

    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.jsonl"
        if args.overwrite or not shard_path.exists():
            trial_rows: list[dict[str, Any]] = []
            clean_strata = sorted(
                {
                    (str(plan_row.response_type), str(plan_row.position_variant))
                    for plan_row in plan.itertuples(index=False)
                }
            )
            for response_type, position_variant in clean_strata:
                trial_rows.extend(
                    run_response_reference_head_ablation_trials(
                        model,
                        tokenizer,
                        adapter,
                        row,
                        response_type=response_type,
                        position_variant=position_variant,
                        heads=[],
                        condition="clean",
                    )
                )
            for plan_row in plan.itertuples(index=True):
                result_rows = run_response_reference_head_ablation_trials(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    response_type=str(plan_row.response_type),
                    position_variant=str(plan_row.position_variant),
                    heads=_parse_heads(plan_row.heads),
                    condition=str(plan_row.condition),
                )
                for result in result_rows:
                    result["plan_row"] = int(plan_row.Index)
                    result["repeat"] = int(plan_row.repeat)
                    result["bank_scope"] = str(plan_row.bank_scope)
                    for field in (
                        "target_needle_raw_mass",
                        "target_needle_relative_mass",
                        "discovery_target_needle_raw_mass",
                        "discovery_target_needle_relative_mass",
                        "confirmation_target_needle_raw_mass",
                        "confirmation_target_needle_relative_mass",
                    ):
                        if hasattr(plan_row, field):
                            result[field] = float(getattr(plan_row, field))
                trial_rows.extend(result_rows)
            write_jsonl(shard_path, trial_rows)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 causal-response-reference] {index}/{len(rows)} "
            f"{action} {request_id}",
            flush=True,
        )
    output_rows = []
    for path in shard_paths:
        output_rows.extend(read_jsonl(path))
    write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-response-reference] requests={len(rows)} "
        f"rows={len(output_rows)} output={args.output}",
        flush=True,
    )


def command_causal_tokens(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import run_token_corruption_trial

    config = _config(args)
    model, tokenizer, _adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    if not args.include_discovery:
        rows = [row for row in rows if row["split"] == "confirmation"]
    output_rows = []
    for index, row in enumerate(rows, start=1):
        output_rows.extend(
            run_token_corruption_trial(
                model,
                tokenizer,
                row,
                config=config,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print(f"[v5 causal-tokens] {index}/{len(rows)}", flush=True)
    write_jsonl(args.output, output_rows)


def command_causal_context(args: argparse.Namespace) -> None:
    import numpy as np

    from realistic_niah_v5.causal import run_query_context_mask_trial

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _split_rows(rows, args.split)
    output_dir = Path(args.output)
    state_dir = output_dir / "states"
    index_rows = []
    for row_index, row in enumerate(rows, start=1):
        site_ids = _row_site_ids(
            row, site_kind=args.site_kind, site_id=args.site_id
        )
        for site_id in site_ids:
            for condition in args.conditions:
                result = run_query_context_mask_trial(
                    model,
                    adapter,
                    tokenizer,
                    row,
                    site_id=site_id,
                    condition=condition,
                    layers=args.layers,
                )
                states = np.asarray(result.pop("states"), dtype=np.float32)
                layers = np.asarray(result["layers"], dtype=np.int64)
                stem = _safe_stem(result["request_id"], site_id, condition)
                state_path = state_dir / f"{stem}.npz"
                _atomic_npz(state_path, states=states, layers=layers)
                result["states_path"] = str(state_path.relative_to(output_dir))
                index_rows.append(result)
        print(f"[v5 causal-context] {row_index}/{len(rows)}", flush=True)
    index_path = output_dir / "context_index.jsonl"
    write_jsonl(index_path, index_rows)
    print(f"[v5 causal-context] wrote {len(index_rows)} trials to {index_path}")


def command_causal_writes(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    from realistic_niah_v5.causal import capture_natural_head_writes
    from realistic_niah_v5.encoding import build_native_trace_encoding

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _split_rows(rows, args.split)
    directions = (
        _load_layer_directions(args.directions) if args.directions else None
    )
    frames = []
    vector_index = []
    output = Path(args.output)
    for row_index, row in enumerate(rows, start=1):
        encoding = build_native_trace_encoding(
            row,
            tokenizer,
            site_id=args.site_id,
            candidate_counts=config.candidate_counts,
        )
        frame, writes = capture_natural_head_writes(
            model,
            adapter,
            encoding,
            layers=args.layers,
            count_direction=directions,
        )
        frames.append(frame)
        if args.save_vectors:
            ordered = sorted(writes)
            stem = _safe_stem(encoding.request_id, args.site_id, "natural_writes")
            vector_path = output.parent / f"{output.stem}_vectors" / f"{stem}.npz"
            _atomic_npz(
                vector_path,
                layers=np.asarray([layer for layer, _head in ordered], dtype=np.int64),
                heads=np.asarray([head for _layer, head in ordered], dtype=np.int64),
                writes=np.stack([writes[key].numpy() for key in ordered]),
            )
            vector_index.append(
                {
                    "request_id": encoding.request_id,
                    "site_id": args.site_id,
                    "vectors_path": str(vector_path.resolve()),
                }
            )
        print(f"[v5 causal-writes] {row_index}/{len(rows)}", flush=True)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    if args.save_vectors:
        write_jsonl(output.with_suffix(".vectors.jsonl"), vector_index)
    print(f"[v5 causal-writes] wrote {len(result)} rows to {output}")


def command_causal_subspace_fit(args: argparse.Namespace) -> None:
    import numpy as np

    from realistic_niah_v5.causal import fit_centroid_subspace
    from realistic_niah_v5.representation import cohort_mask, load_capture_dataset

    config = _config(args)
    dataset = load_capture_dataset(
        args.capture_index, site_kinds=[args.site_kind]
    )
    metadata = dataset.metadata
    mask = cohort_mask(metadata, args.cohort)
    mask &= metadata["split"].astype(str).eq("discovery").to_numpy()
    if args.site_id:
        mask &= metadata["site_id"].astype(str).eq(args.site_id).to_numpy()
    if (
        args.site_kind not in {"answer_query", "answer_query_v2", "answer_query_v3"}
        and config.representation_n10_only
    ):
        mask &= metadata["gold_count"].astype(int).eq(10).to_numpy()
    available_layers = sorted(
        int(value) for value in metadata.loc[mask, "layer"].unique()
    )
    layers = available_layers if args.layers is None else sorted(set(args.layers))
    missing = sorted(set(layers) - set(available_layers))
    if missing:
        raise ValueError(f"Subspace fit layers are unavailable: {missing}")
    arrays: dict[str, Any] = {}
    fit_rows = []
    for layer in layers:
        layer_mask = mask & metadata["layer"].astype(int).eq(layer).to_numpy()
        states = dataset.states[layer_mask]
        label_column = (
            "gold_count"
            if args.site_kind in {"answer_query", "answer_query_v2", "answer_query_v3"}
            else "occurrence"
        )
        labels = metadata.loc[layer_mask, label_column].to_numpy(dtype=int)
        center, basis = fit_centroid_subspace(states, labels, rank=args.rank)
        arrays[f"center_L{layer}"] = center
        arrays[f"basis_L{layer}"] = basis
        fit_rows.append(
            {
                "layer": int(layer),
                "observations": int(len(states)),
                "labels": sorted(int(value) for value in np.unique(labels)),
                "effective_rank": int(basis.shape[1]),
            }
        )
    _atomic_npz(args.output, **arrays)
    audit = {
        "schema_version": "realistic_niah_v5_subspace_fit_v1",
        "capture_index": str(args.capture_index.resolve()),
        "selection_split": "discovery",
        "confirmation_used_for_fit": False,
        "cohort": args.cohort,
        "site_kind": args.site_kind,
        "site_id": args.site_id,
        "label": (
            "gold_count"
            if args.site_kind in {"answer_query", "answer_query_v2", "answer_query_v3"}
            else "occurrence"
        ),
        "basis_orientation": (
            "each component is signed toward increasing label when its linear "
            "label association is nonzero"
        ),
        "requested_rank": int(args.rank),
        "fits": fit_rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[v5 causal-subspace-fit] wrote {len(fit_rows)} bases to {args.output}")


def command_causal_subspace(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import run_subspace_ablation

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _split_rows(rows, args.split)
    bases = _load_layer_bases(args.basis)
    import numpy as np

    with np.load(args.basis, allow_pickle=False) as archive:
        centers = {
            int(key.removeprefix("center_L")): np.asarray(
                archive[key], dtype=np.float32
            )
            for key in archive.files
            if key.startswith("center_L")
        }
    layers = sorted(bases) if args.layers is None else sorted(set(args.layers))
    if sorted(set(layers) - (set(bases) & set(centers))):
        raise ValueError("Every requested layer needs both center and basis arrays")
    output_rows = []
    for row_index, row in enumerate(rows, start=1):
        site_ids = _row_site_ids(
            row, site_kind=args.site_kind, site_id=args.site_id
        )
        for site_id in site_ids:
            for layer in layers:
                for dose in (0.0, *args.doses):
                    result = run_subspace_ablation(
                        model,
                        tokenizer,
                        adapter,
                        row,
                        site_id=site_id,
                        layer=layer,
                        center=centers[layer],
                        basis=bases[layer],
                        dose=dose,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if dose == 0.0:
                        result["condition"] = "clean_dose_zero"
                    output_rows.append(result)
        print(f"[v5 causal-subspace] {row_index}/{len(rows)}", flush=True)
    write_jsonl(args.output, output_rows)
    print(f"[v5 causal-subspace] wrote {len(output_rows)} trials to {args.output}")


def command_causal_patch(args: argparse.Namespace) -> None:
    from realistic_niah_v4.modeling import generate_with_residual_interventions
    from realistic_niah_v5.causal import (
        capture_site_state,
        run_projected_patch_trials_from_states,
    )
    from realistic_niah_v5.parsing import parse_trace_record

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    row_by_id = {
        str(row.get("request_id", row.get("stimulus_id"))): row for row in rows
    }
    bases = _load_layer_bases(args.basis)
    output_rows = []
    pairs = read_jsonl(args.pairs)
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    if args.restartable:
        shard_root.mkdir(parents=True, exist_ok=True)
    state_cache: dict[tuple[str, str, int], tuple[Any, Any]] = {}
    self_patch_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def cached_state(
        request_id: str,
        site_id: str,
        layer: int,
    ) -> tuple[Any, Any]:
        key = (request_id, site_id, int(layer))
        if key not in state_cache:
            state_cache[key] = capture_site_state(
                model,
                adapter,
                tokenizer,
                row_by_id[request_id],
                site_id=site_id,
                layer=int(layer),
            )
        return state_cache[key]

    for pair_index, pair in enumerate(pairs, start=1):
        shard_path = shard_root / f"{pair_index:05d}.jsonl"
        if args.restartable and shard_path.exists() and not args.overwrite:
            output_rows.extend(read_jsonl(shard_path))
            print(
                f"[v5 causal-patch] {pair_index}/{len(pairs)} reused",
                flush=True,
            )
            continue
        receiver_id = str(pair["receiver_request_id"])
        donor_id = str(pair["donor_request_id"])
        if not bool(pair.get("receiver_exact_count")) or not bool(
            pair.get("donor_exact_count")
        ):
            raise ValueError(
                "Answer execution is correct-only: both receiver and donor "
                f"must be clean-correct ({receiver_id}/{donor_id})"
            )
        if receiver_id not in row_by_id or donor_id not in row_by_id:
            raise KeyError(f"Unknown receiver/donor pair: {receiver_id}/{donor_id}")
        receiver_parsed = parse_trace_record(row_by_id[receiver_id])
        donor_parsed = parse_trace_record(row_by_id[donor_id])
        if (
            not bool(receiver_parsed["parser"].get("trace_one_to_one"))
            or not bool(donor_parsed["parser"].get("trace_one_to_one"))
            or not bool(receiver_parsed.get("exact_count"))
            or not bool(donor_parsed.get("exact_count"))
        ):
            raise ValueError(
                "Answer execution runtime revalidation failed: receiver and "
                "donor must both be strict one-to-one and baseline-final-answer "
                f"correct ({receiver_id}/{donor_id})"
            )
        if int(receiver_parsed["gold_count"]) != int(pair["receiver_count"]):
            raise ValueError(f"Receiver count mismatch for {receiver_id}")
        if int(donor_parsed["gold_count"]) != int(pair["donor_count"]):
            raise ValueError(f"Donor count mismatch for {donor_id}")
        layer = int(pair.get("layer", args.layer))
        if layer not in bases:
            raise ValueError(f"Patch basis has no layer {layer}")
        receiver_site = str(pair.get("receiver_site_id", args.receiver_site_id))
        donor_site = str(pair.get("donor_site_id", args.donor_site_id))
        receiver, receiver_state = cached_state(
            receiver_id,
            receiver_site,
            layer,
        )
        donor, donor_state = cached_state(
            donor_id,
            donor_site,
            layer,
        )
        self_key = (receiver_id, receiver_site, int(layer))
        if self_key not in self_patch_cache:
            self_patch_cache[self_key] = generate_with_residual_interventions(
                model,
                tokenizer,
                adapter,
                receiver,
                {int(layer): ([receiver.query_position], receiver_state)},
                max_new_tokens=args.max_new_tokens,
            )
        trial_rows = run_projected_patch_trials_from_states(
            model,
            tokenizer,
            adapter,
            receiver,
            receiver_state,
            donor,
            donor_state,
            receiver_site_id=receiver_site,
            donor_site_id=donor_site,
            layer=layer,
            basis=bases[layer],
            max_new_tokens=args.max_new_tokens,
            self_patch_result=self_patch_cache[self_key],
        )
        for result in trial_rows:
            result["pair_id"] = pair.get("pair_id", pair_index)
            result["donor_role"] = pair.get("donor_role", "registered")
            for field in (
                "pair_direction",
                "receiver_exact_count",
                "donor_exact_count",
                "pair_eligibility",
            ):
                if field in pair:
                    result[field] = pair[field]
        if args.restartable:
            write_jsonl(shard_path, trial_rows)
        output_rows.extend(trial_rows)
        print(f"[v5 causal-patch] {pair_index}/{len(pairs)}", flush=True)
    write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-patch] wrote {len(output_rows)} trials to {args.output}; "
        f"cached_states={len(state_cache)} cached_self_patches={len(self_patch_cache)}"
    )


def command_causal_analyze(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import analyze_paired_causal_results

    result = analyze_paired_causal_results(
        args.trials,
        args.output,
        treatment=args.treatment,
        control=args.control,
        outcome=args.outcome,
        config=_config(args),
        mechanism=args.mechanism,
        bank_size=args.bank_size,
        transition_phase=args.transition_phase,
    )
    print(result.to_string(index=False))


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")


def _add_split(parser: argparse.ArgumentParser, *, default: str = "confirmation") -> None:
    parser.add_argument(
        "--split",
        choices=["discovery", "confirmation", "all"],
        default=default,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V5 native-thinking representation and causal pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse = subparsers.add_parser("parse")
    parse.add_argument("--input", type=Path, required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--compact", action="store_true")
    parse.set_defaults(func=command_parse)

    generate = subparsers.add_parser("generate")
    _add_config(generate)
    _add_model(generate)
    generate.add_argument("--stimuli", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(func=command_generate)

    capture = subparsers.add_parser("capture")
    _add_config(capture)
    _add_model(capture)
    capture.add_argument("--generations", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--layers", type=int, nargs="+")
    capture.add_argument("--overwrite", action="store_true")
    capture.add_argument(
        "--site-ids",
        nargs="+",
        help=(
            "Explicit registered site IDs for an isolated extension capture; "
            "omitting this preserves the frozen main-site configuration."
        ),
    )
    capture.add_argument(
        "--allow-unregistered",
        action="store_true",
        help=(
            "Allow isolated supplement rows outside the primary 300-seed "
            "registry; rows are still model-filtered and never merged in place."
        ),
    )
    capture.set_defaults(func=command_capture)

    attention = subparsers.add_parser("attention")
    _add_config(attention)
    _add_model(attention)
    attention.add_argument("--generations", type=Path, required=True)
    attention.add_argument("--output", type=Path, required=True)
    attention.add_argument(
        "--mechanisms",
        nargs="+",
        choices=["targeted_retrieval", "progress_transition"],
        default=["targeted_retrieval", "progress_transition"],
    )
    attention.set_defaults(func=command_attention)

    attention_pre_city = subparsers.add_parser("attention-pre-city")
    _add_config(attention_pre_city)
    _add_model(attention_pre_city)
    attention_pre_city.add_argument("--generations", type=Path, required=True)
    attention_pre_city.add_argument("--output", type=Path, required=True)
    attention_pre_city.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    attention_pre_city.add_argument("--no-anchor", action="store_true")
    attention_pre_city.add_argument(
        "--split", choices=["all", "discovery", "confirmation"], default="all"
    )
    attention_pre_city.add_argument("--max-rows", type=int)
    attention_pre_city.add_argument("--overwrite", action="store_true")
    attention_pre_city.set_defaults(func=command_attention_pre_city)

    response_reference_attention = subparsers.add_parser(
        "response-reference-attention"
    )
    response_reference_attention.add_argument("--model", required=True)
    response_reference_attention.add_argument(
        "--generations", type=Path, required=True
    )
    response_reference_attention.add_argument(
        "--pre-city-attention", type=Path, required=True
    )
    response_reference_attention.add_argument("--output", type=Path, required=True)
    response_reference_attention.set_defaults(
        func=command_response_reference_attention
    )

    attention_answer_query = subparsers.add_parser("attention-answer-query")
    _add_model(attention_answer_query)
    attention_answer_query.add_argument(
        "--generations", type=Path, required=True
    )
    attention_answer_query.add_argument("--output", type=Path, required=True)
    attention_answer_query.add_argument(
        "--site-id", default="answer_query_v3"
    )
    attention_answer_query.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    attention_answer_query.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="all",
    )
    attention_answer_query.add_argument("--max-rows", type=int)
    attention_answer_query.add_argument("--overwrite", action="store_true")
    attention_answer_query.set_defaults(func=command_attention_answer_query)

    representation = subparsers.add_parser("representation")
    _add_config(representation)
    representation.add_argument("--capture-index", type=Path, required=True)
    representation.add_argument("--output", type=Path, required=True)
    representation.add_argument(
        "--cohorts",
        nargs="+",
        default=["parser_hit", "one_to_one", "one_to_one_correct"],
    )
    representation.set_defaults(func=command_representation)

    causal_plan = subparsers.add_parser("causal-plan")
    _add_config(causal_plan)
    causal_plan.add_argument("--attention", type=Path, required=True)
    causal_plan.add_argument("--output", type=Path, required=True)
    causal_plan.set_defaults(func=command_causal_plan)

    answer_query_plan = subparsers.add_parser("answer-query-causal-plan")
    _add_config(answer_query_plan)
    answer_query_plan.add_argument("--attention", type=Path, required=True)
    answer_query_plan.add_argument("--output", type=Path, required=True)
    answer_query_plan.set_defaults(func=command_answer_query_causal_plan)

    pre_city_causal_plan = subparsers.add_parser("pre-city-causal-plan")
    _add_config(pre_city_causal_plan)
    pre_city_causal_plan.add_argument("--attention", type=Path, required=True)
    pre_city_causal_plan.add_argument("--output", type=Path, required=True)
    pre_city_causal_plan.set_defaults(func=command_pre_city_causal_plan)

    response_reference_plan = subparsers.add_parser(
        "response-reference-causal-plan"
    )
    _add_config(response_reference_plan)
    response_reference_plan.add_argument("--attention", type=Path, required=True)
    response_reference_plan.add_argument("--output", type=Path, required=True)
    response_reference_plan.set_defaults(
        func=command_response_reference_causal_plan
    )

    causal_heads = subparsers.add_parser("causal-heads")
    _add_config(causal_heads)
    _add_model(causal_heads)
    causal_heads.add_argument("--generations", type=Path, required=True)
    causal_heads.add_argument("--plan", type=Path, required=True)
    causal_heads.add_argument("--output", type=Path, required=True)
    causal_heads.add_argument("--plan-rows", type=int, nargs="+")
    causal_heads.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    causal_heads.add_argument("--include-discovery", action="store_true")
    causal_heads.add_argument(
        "--boundary-policy",
        choices=["strict_registered", "item_end_fallback_v2"],
        default="strict_registered",
        help=(
            "Registered target-boundary policy. item_end_fallback_v2 preserves "
            "the primary boundary and deterministically uses item_end only when "
            "the primary boundary is not literal-token aligned."
        ),
    )
    causal_heads.set_defaults(func=command_causal_heads)

    causal_pre_city_heads = subparsers.add_parser("causal-pre-city-heads")
    _add_config(causal_pre_city_heads)
    _add_model(causal_pre_city_heads)
    causal_pre_city_heads.add_argument("--generations", type=Path, required=True)
    causal_pre_city_heads.add_argument("--plan", type=Path, required=True)
    causal_pre_city_heads.add_argument("--output", type=Path, required=True)
    causal_pre_city_heads.add_argument("--plan-rows", type=int, nargs="+")
    causal_pre_city_heads.add_argument(
        "--query-variant",
        choices=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
        required=True,
    )
    causal_pre_city_heads.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    causal_pre_city_heads.add_argument("--include-discovery", action="store_true")
    causal_pre_city_heads.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="confirmation",
    )
    causal_pre_city_heads.add_argument("--gold-count", type=int)
    causal_pre_city_heads.add_argument("--max-rows", type=int)
    causal_pre_city_heads.add_argument("--allow-unregistered", action="store_true")
    causal_pre_city_heads.add_argument("--overwrite", action="store_true")
    causal_pre_city_heads.set_defaults(func=command_causal_pre_city_heads)

    causal_pre_city_all_sites = subparsers.add_parser(
        "causal-pre-city-all-sites"
    )
    _add_config(causal_pre_city_all_sites)
    _add_model(causal_pre_city_all_sites)
    causal_pre_city_all_sites.add_argument(
        "--generations", type=Path, required=True
    )
    causal_pre_city_all_sites.add_argument("--plan", type=Path, required=True)
    causal_pre_city_all_sites.add_argument(
        "--output", type=Path, required=True
    )
    causal_pre_city_all_sites.add_argument("--plan-rows", type=int, nargs="+")
    causal_pre_city_all_sites.add_argument(
        "--query-variant",
        choices=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
        required=True,
    )
    causal_pre_city_all_sites.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    causal_pre_city_all_sites.add_argument(
        "--include-discovery", action="store_true"
    )
    causal_pre_city_all_sites.add_argument("--max-rows", type=int)
    causal_pre_city_all_sites.add_argument(
        "--gold-count",
        type=int,
        choices=range(1, 11),
        help="Optional N filter used for exact all-site smoke/audit runs.",
    )
    causal_pre_city_all_sites.add_argument("--overwrite", action="store_true")
    causal_pre_city_all_sites.add_argument(
        "--worker-id",
        default="primary",
        help="Audit label for a cooperative shared-shard worker.",
    )
    causal_pre_city_all_sites.add_argument(
        "--worker-poll-seconds", type=float, default=5.0
    )
    causal_pre_city_all_sites.add_argument(
        "--worker-wait-timeout", type=float, default=86400.0
    )
    causal_pre_city_all_sites.add_argument(
        "--claim-stale-seconds", type=float, default=21600.0
    )
    causal_pre_city_all_sites.add_argument(
        "--allow-unregistered",
        action="store_true",
        help=(
            "Allow isolated supplement seeds outside the primary registry; "
            "outputs remain in a separate caller-provided path."
        ),
    )
    causal_pre_city_all_sites.set_defaults(
        func=command_causal_pre_city_all_sites
    )

    marker_needle_patch = subparsers.add_parser(
        "causal-marker-needle-patch"
    )
    _add_model(marker_needle_patch)
    marker_needle_patch.add_argument(
        "--generations", type=Path, nargs="+", required=True
    )
    marker_needle_patch.add_argument("--pairs", type=Path, required=True)
    marker_needle_patch.add_argument("--output", type=Path, required=True)
    marker_needle_patch.add_argument(
        "--query-variants",
        nargs="+",
        choices=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
        default=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
    )
    marker_needle_patch.add_argument("--layers", type=int, nargs="+", required=True)
    marker_needle_patch.add_argument(
        "--split", choices=["all", "discovery", "confirmation"], default="all"
    )
    marker_needle_patch.add_argument("--max-pairs", type=int)
    marker_needle_patch.add_argument("--overwrite", action="store_true")
    marker_needle_patch.set_defaults(func=command_causal_marker_needle_patch)

    causal_answer_query_heads = subparsers.add_parser(
        "causal-answer-query-heads"
    )
    _add_model(causal_answer_query_heads)
    causal_answer_query_heads.add_argument(
        "--generations", type=Path, required=True
    )
    causal_answer_query_heads.add_argument("--plan", type=Path, required=True)
    causal_answer_query_heads.add_argument(
        "--output", type=Path, required=True
    )
    causal_answer_query_heads.add_argument(
        "--site-id", default="answer_query_v3"
    )
    causal_answer_query_heads.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    causal_answer_query_heads.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="confirmation",
    )
    causal_answer_query_heads.add_argument("--plan-rows", type=int, nargs="+")
    causal_answer_query_heads.add_argument("--max-rows", type=int)
    causal_answer_query_heads.add_argument("--overwrite", action="store_true")
    causal_answer_query_heads.set_defaults(
        func=command_causal_answer_query_heads
    )

    causal_response_reference = subparsers.add_parser(
        "causal-response-reference-heads"
    )
    _add_model(causal_response_reference)
    causal_response_reference.add_argument(
        "--generations", type=Path, required=True
    )
    causal_response_reference.add_argument("--plan", type=Path, required=True)
    causal_response_reference.add_argument("--output", type=Path, required=True)
    causal_response_reference.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one_correct",
    )
    causal_response_reference.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="confirmation",
    )
    causal_response_reference.add_argument(
        "--response-types", nargs="+", choices=list(("bare_or_list", "record_template", "semantic_cue"))
    )
    causal_response_reference.add_argument(
        "--position-variants",
        nargs="+",
        choices=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
    )
    causal_response_reference.add_argument(
        "--bank-scopes",
        nargs="+",
        choices=["unified_consensus", "response_type_and_position_specific"],
    )
    causal_response_reference.add_argument("--plan-rows", type=int, nargs="+")
    causal_response_reference.add_argument("--max-rows", type=int)
    causal_response_reference.add_argument("--overwrite", action="store_true")
    causal_response_reference.set_defaults(
        func=command_causal_response_reference_heads
    )

    causal_tokens = subparsers.add_parser("causal-tokens")
    _add_config(causal_tokens)
    _add_model(causal_tokens)
    causal_tokens.add_argument("--generations", type=Path, required=True)
    causal_tokens.add_argument("--output", type=Path, required=True)
    causal_tokens.add_argument("--max-new-tokens", type=int, default=16)
    causal_tokens.add_argument("--include-discovery", action="store_true")
    causal_tokens.set_defaults(func=command_causal_tokens)

    causal_context = subparsers.add_parser("causal-context")
    _add_config(causal_context)
    _add_model(causal_context)
    _add_split(causal_context)
    causal_context.add_argument("--generations", type=Path, required=True)
    causal_context.add_argument("--output", type=Path, required=True)
    site_group = causal_context.add_mutually_exclusive_group(required=True)
    site_group.add_argument("--site-kind")
    site_group.add_argument("--site-id")
    causal_context.add_argument(
        "--conditions",
        nargs="+",
        choices=["clean", "trace_only", "matched_nontrace_only"],
        default=["clean", "trace_only", "matched_nontrace_only"],
    )
    causal_context.add_argument("--layers", type=int, nargs="+")
    causal_context.set_defaults(func=command_causal_context)

    causal_writes = subparsers.add_parser("causal-writes")
    _add_config(causal_writes)
    _add_model(causal_writes)
    _add_split(causal_writes, default="all")
    causal_writes.add_argument("--generations", type=Path, required=True)
    causal_writes.add_argument("--output", type=Path, required=True)
    causal_writes.add_argument("--site-id", default="answer_query")
    causal_writes.add_argument("--layers", type=int, nargs="+")
    causal_writes.add_argument("--directions", type=Path)
    causal_writes.add_argument("--save-vectors", action="store_true")
    causal_writes.set_defaults(func=command_causal_writes)

    subspace_fit = subparsers.add_parser("causal-subspace-fit")
    _add_config(subspace_fit)
    subspace_fit.add_argument("--capture-index", type=Path, required=True)
    subspace_fit.add_argument("--output", type=Path, required=True)
    subspace_fit.add_argument("--site-kind", required=True)
    subspace_fit.add_argument("--site-id")
    subspace_fit.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    subspace_fit.add_argument("--layers", type=int, nargs="+")
    subspace_fit.add_argument("--rank", type=int, default=3)
    subspace_fit.set_defaults(func=command_causal_subspace_fit)

    causal_subspace = subparsers.add_parser("causal-subspace")
    _add_config(causal_subspace)
    _add_model(causal_subspace)
    _add_split(causal_subspace)
    causal_subspace.add_argument("--generations", type=Path, required=True)
    causal_subspace.add_argument("--basis", type=Path, required=True)
    causal_subspace.add_argument("--output", type=Path, required=True)
    subspace_site = causal_subspace.add_mutually_exclusive_group(required=True)
    subspace_site.add_argument("--site-kind")
    subspace_site.add_argument("--site-id")
    causal_subspace.add_argument("--layers", type=int, nargs="+")
    causal_subspace.add_argument("--doses", type=float, nargs="+", default=[1.0])
    causal_subspace.add_argument("--max-new-tokens", type=int, default=16)
    causal_subspace.set_defaults(func=command_causal_subspace)

    causal_patch = subparsers.add_parser("causal-patch")
    _add_config(causal_patch)
    _add_model(causal_patch)
    causal_patch.add_argument("--generations", type=Path, required=True)
    causal_patch.add_argument("--pairs", type=Path, required=True)
    causal_patch.add_argument("--basis", type=Path, required=True)
    causal_patch.add_argument("--output", type=Path, required=True)
    causal_patch.add_argument("--layer", type=int, required=True)
    causal_patch.add_argument("--receiver-site-id", default="answer_query")
    causal_patch.add_argument("--donor-site-id", default="answer_query")
    causal_patch.add_argument("--max-new-tokens", type=int, default=16)
    causal_patch.add_argument("--restartable", action="store_true")
    causal_patch.add_argument("--overwrite", action="store_true")
    causal_patch.set_defaults(func=command_causal_patch)

    causal_analyze = subparsers.add_parser("causal-analyze")
    _add_config(causal_analyze)
    causal_analyze.add_argument("--trials", type=Path, required=True)
    causal_analyze.add_argument("--output", type=Path, required=True)
    causal_analyze.add_argument("--treatment", required=True)
    causal_analyze.add_argument("--control", required=True)
    causal_analyze.add_argument("--outcome", default="absolute_error")
    causal_analyze.add_argument(
        "--mechanism",
        choices=["targeted_retrieval", "progress_transition"],
    )
    causal_analyze.add_argument("--bank-size", type=int)
    causal_analyze.add_argument(
        "--transition-phase", choices=["retrieve", "continue", "stop"]
    )
    causal_analyze.set_defaults(func=command_causal_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
