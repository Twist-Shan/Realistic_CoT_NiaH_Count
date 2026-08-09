#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
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
            )
            for result in results:
                result["plan_row"] = int(plan_row.Index)
                result["repeat"] = int(plan_row.repeat)
            output_rows.extend(results)
        print(f"[v5 causal-heads] {row_index}/{len(rows)}", flush=True)
    write_jsonl(args.output, output_rows)


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
    if args.site_kind != "answer_query" and config.representation_n10_only:
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
        label_column = "gold_count" if args.site_kind == "answer_query" else "occurrence"
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
        "label": "gold_count" if args.site_kind == "answer_query" else "occurrence",
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
    from realistic_niah_v5.causal import run_projected_patch_trials

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
    for pair_index, pair in enumerate(pairs, start=1):
        receiver_id = str(pair["receiver_request_id"])
        donor_id = str(pair["donor_request_id"])
        if receiver_id not in row_by_id or donor_id not in row_by_id:
            raise KeyError(f"Unknown receiver/donor pair: {receiver_id}/{donor_id}")
        layer = int(pair.get("layer", args.layer))
        if layer not in bases:
            raise ValueError(f"Patch basis has no layer {layer}")
        receiver_site = str(pair.get("receiver_site_id", args.receiver_site_id))
        donor_site = str(pair.get("donor_site_id", args.donor_site_id))
        trial_rows = run_projected_patch_trials(
            model,
            tokenizer,
            adapter,
            row_by_id[receiver_id],
            row_by_id[donor_id],
            receiver_site_id=receiver_site,
            donor_site_id=donor_site,
            layer=layer,
            basis=bases[layer],
            max_new_tokens=args.max_new_tokens,
        )
        for result in trial_rows:
            result["pair_id"] = pair.get("pair_id", pair_index)
            result["donor_role"] = pair.get("donor_role", "registered")
        output_rows.extend(trial_rows)
        print(f"[v5 causal-patch] {pair_index}/{len(pairs)}", flush=True)
    write_jsonl(args.output, output_rows)
    print(f"[v5 causal-patch] wrote {len(output_rows)} trials to {args.output}")


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
    causal_heads.set_defaults(func=command_causal_heads)

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
