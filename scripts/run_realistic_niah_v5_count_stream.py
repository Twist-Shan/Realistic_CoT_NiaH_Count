#!/usr/bin/env python3
"""Run the native-thinking count-stream mechanism experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_SCHEMA_VERSION,
    REGISTERED_HIGHER_IS_BETTER_OUTCOMES,
    REGISTERED_MASK_CONDITIONS,
    REGISTERED_RESTORATION_CONDITIONS,
    REGISTERED_STREAM_CONDITIONS,
    REGISTERED_TRACE_PATCH_CONDITIONS,
    NativeCountMechanismSpec,
    build_answer_broad_head_plan,
    build_sparse_trace_patch_sample_plan,
    capture_answer_source_attention,
    count_stream_cohort_mask,
    fit_count_stream_basis,
    load_count_stream_capture_dataset,
    mechanism_decision_ledger,
    rank_answer_broad_heads,
    run_answer_broad_head_trial,
    run_answer_source_mask_trial,
    run_stream_state_trial,
    run_trace_intermediate_patch_trials,
    run_trace_terminal_patch_trials,
    run_trace_restoration_trials,
    stream_state_retention_metrics,
    select_answer_broad_bank_size,
    summarize_linear_contrasts,
)
from realistic_niah_v5.pipeline import (  # noqa: E402
    read_jsonl,
    registered_records,
)
from realistic_niah_v5.parsing import (  # noqa: E402
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    raw_output_text,
)
from realistic_niah_v5.spec import V5Config  # noqa: E402


DEFAULT_V5_CONFIG = ROOT / "configs" / "realistic_niah_v5.json"
DEFAULT_MECHANISM_CONFIG = (
    ROOT / "configs" / "realistic_niah_v5_native_count_stream_dev.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_stem(*values: Any) -> str:
    text = "__".join(str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    clean = "".join(character if character.isalnum() else "_" for character in text)
    return f"{clean[:96].strip('_')}__{digest}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _runtime_manifest(
    args: argparse.Namespace,
    *,
    mechanism: NativeCountMechanismSpec,
    started: float,
    completed_shards: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except Exception:  # pragma: no cover - environment audit only
        torch_version = None
        cuda = False
        devices = []
    return {
        "schema_version": COUNT_STREAM_SCHEMA_VERSION,
        "command": str(args.command),
        "argv": sys.argv,
        "mechanism_config": str(args.mechanism_config.resolve()),
        "mechanism_config_sha256": _sha256(args.mechanism_config),
        "v5_config": (
            str(args.v5_config.resolve()) if hasattr(args, "v5_config") else None
        ),
        "v5_config_sha256": (
            _sha256(args.v5_config) if hasattr(args, "v5_config") else None
        ),
        "mechanism_spec": mechanism.to_dict(),
        "formal_inference_eligible": mechanism.formal_inference_eligible,
        "completed_shards": int(completed_shards),
        "elapsed_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda,
        "devices": devices,
        "git": _git_state(),
        "cohort": getattr(args, "cohort", None),
        "cohort_audit": getattr(args, "cohort_audit", None),
        **(extra or {}),
    }


def _model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    model_spec = resolve_model_spec(args.model)
    return load_registered_model(
        model_spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )


def _spec(args: argparse.Namespace) -> NativeCountMechanismSpec:
    return NativeCountMechanismSpec.load(args.mechanism_config)


def _cohort_exclusion_reason(row: dict[str, Any], cohort: str) -> str | None:
    trace_parse = row.get("trace_parse")
    if not isinstance(trace_parse, dict):
        return "missing_trace_parse"
    archived_parser = trace_parse.get("parser")
    if not isinstance(archived_parser, dict):
        return "missing_trace_parse"
    if isinstance(row.get("raw_output_text"), str):
        parser = find_trace_count_sequence(
            raw_output_text(row),
            model_family=infer_model_family(row),
            gold_records=gold_records(row),
        )
        detected = bool(parser.detected)
        trace_one_to_one = bool(parser.trace_one_to_one)
    else:
        # Unit fixtures and pre-V5 legacy rows may retain only the archived
        # parser payload. Production V5 rows always take the active-parser
        # branch so cohort selection and causal site compilation cannot drift.
        detected = bool(archived_parser.get("detected"))
        trace_one_to_one = bool(archived_parser.get("trace_one_to_one"))
    if not detected:
        return "parser_miss"
    if cohort == "parser_hit":
        return None
    if not trace_one_to_one:
        return "not_one_to_one"
    if cohort == "one_to_one":
        return None
    if cohort == "one_to_one_correct":
        return None if bool(trace_parse.get("exact_count")) else "final_count_incorrect"
    raise ValueError(f"Unknown native-thinking cohort: {cohort}")


def _registered_rows(
    args: argparse.Namespace,
    mechanism: NativeCountMechanismSpec,
) -> list[dict[str, Any]]:
    config = V5Config.load(args.v5_config)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    selected = []
    exclusion_counts: dict[str, int] = {}
    row_panel = str(getattr(args, "row_panel", "all"))
    broad_phase_by_panel = {
        "broad_ranking": "ranking_discovery",
        "broad_k_selection": "k_selection_discovery",
        "broad_confirmation": "confirmation",
    }
    broad_phase = broad_phase_by_panel.get(row_panel)
    if broad_phase in {"ranking_discovery", "k_selection_discovery"} and args.seed_role != "development":
        raise ValueError(f"{row_panel} requires --seed-role development")
    if broad_phase == "confirmation" and args.seed_role != "confirmation":
        raise ValueError("broad_confirmation requires --seed-role confirmation")
    for row in rows:
        role = mechanism.seed_role(int(row["seed"]))
        if role != args.seed_role:
            continue
        if broad_phase is not None:
            if mechanism.broad_phase(int(row["seed"])) != broad_phase:
                continue
            allowed_counts = mechanism.broad_counts_for_seed(
                int(row["seed"]), phase=broad_phase
            )
            if int(row.get("gold_count", 0)) not in set(allowed_counts):
                continue
        exclusion = _cohort_exclusion_reason(row, args.cohort)
        if exclusion is not None:
            exclusion_counts[exclusion] = exclusion_counts.get(exclusion, 0) + 1
            continue
        active = dict(row)
        active["mechanism_split"] = role
        active["mechanism_cohort"] = args.cohort
        selected.append(active)
    if args.seed_role == "confirmation" and not mechanism.formal_inference_eligible:
        raise ValueError(
            "The mechanism config has no frozen fresh-confirmation registry"
        )
    selected.sort(key=lambda row: (int(row["seed"]), int(row.get("gold_count", 0))))
    if getattr(args, "limit", None) is not None:
        if int(args.limit) < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: int(args.limit)]
    args.cohort_audit = {
        "cohort": args.cohort,
        "parser_source": "active_parser_over_frozen_output",
        "eligible_rows": len(selected),
        "exclusion_counts": exclusion_counts,
        "limit": getattr(args, "limit", None),
        "row_panel": row_panel,
        "broad_phase": broad_phase,
    }
    if not selected:
        raise ValueError(f"No {args.seed_role} rows remain for {args.model}")
    return selected


def _prepare_shards(output: Path, *, resume: bool, suffix: str) -> Path:
    shard_dir = output / "shards"
    existing = list(shard_dir.glob(f"*.{suffix}")) if shard_dir.exists() else []
    if existing and not resume:
        raise FileExistsError(
            f"Existing shards found in {shard_dir}; resume or choose a new output"
        )
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir


def command_capture_broad(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    if args.seed_role != "development" or args.row_panel != "broad_ranking":
        raise ValueError(
            "Broad attention ranking capture requires the frozen "
            "development/broad_ranking panel"
        )
    rows = _registered_rows(args, mechanism)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="csv")
    completed = skipped = 0
    registry_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(row["request_id"], mechanism.answer_site_id)
        shard = shard_dir / f"{stem}.csv"
        registry_path = shard_dir / f"{stem}.registry.json"
        if args.resume and shard.exists() and registry_path.exists():
            skipped += 1
            continue
        frame, registry = capture_answer_source_attention(
            model,
            tokenizer,
            adapter,
            row,
            answer_site_id=mechanism.answer_site_id,
            layers=args.layers,
        )
        frame["mechanism_split"] = args.seed_role
        _atomic_csv(shard, frame)
        _atomic_json(registry_path, registry)
        registry_rows.append(registry)
        completed += 1
        print(f"[count-stream capture-broad] {index}/{len(rows)}", flush=True)
    manifest = _runtime_manifest(
        args,
        mechanism=mechanism,
        started=started,
        completed_shards=len(list(shard_dir.glob("*.csv"))),
        extra={
            "newly_completed": completed,
            "resume_skipped": skipped,
            "selection_prohibited": args.seed_role != "development",
            "capture_role": args.seed_role,
            "registry_rows_written_this_run": len(registry_rows),
        },
    )
    _atomic_json(output / "manifest.json", manifest)


def _read_capture_frames(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted((path / "shards").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No broad-capture CSV files found under {path}")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def command_plan_broad(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    captures = _read_capture_frames(args.captures)
    coverage = captures.loc[
        captures["model_label"].astype(str).eq(str(args.model))
        & pd.to_numeric(captures["seed"], errors="coerce").isin(
            mechanism.broad_ranking_seeds
        )
    ][["request_id", "seed", "gold_count"]].drop_duplicates()
    observed_pairs = set(
        coverage[["seed", "gold_count"]].itertuples(index=False, name=None)
    )
    expected_pairs = {
        (int(seed), int(count))
        for seed in mechanism.broad_ranking_seeds
        for count in mechanism.candidate_counts
    }
    if observed_pairs != expected_pairs or len(coverage) != len(expected_pairs):
        raise ValueError(
            "Broad ranking capture must contain exactly one request for every "
            "frozen ranking seed x count cell"
        )
    ranking = rank_answer_broad_heads(
        captures,
        source_group=args.source_group,
        development_seeds=mechanism.broad_ranking_seeds,
        model_label=args.model,
    )
    sizes = (
        tuple(args.bank_sizes) if args.bank_sizes else mechanism.development_bank_sizes
    )
    plans = [
        build_answer_broad_head_plan(
            ranking,
            bank_size=size,
            random_controls=mechanism.random_controls,
            random_seed=args.random_seed,
            allow_selected_random_overlap=(
                mechanism.random_control_overlap_policy
                == "nonthinking_allow_treatment_overlap"
            ),
        )
        for size in sizes
    ]
    plan = pd.concat(plans, ignore_index=True)
    output = Path(args.output)
    _atomic_csv(output / "answer_broad_head_ranking.csv", ranking)
    _atomic_csv(output / "answer_broad_head_plan.csv", plan)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "captures": str(args.captures.resolve()),
                "source_group": args.source_group,
                "bank_sizes": list(sizes),
                "selection_role": "development_only",
                "ranking_seed_role": "ranking_discovery",
                "ranking_seeds": list(mechanism.broad_ranking_seeds),
                "confirmation_used_for_selection": False,
                "plan_sha256": _sha256(output / "answer_broad_head_plan.csv"),
            },
        ),
    )


def command_plan_trace_patch(args: argparse.Namespace) -> None:
    """Freeze the 330-local-plus-20-terminal outcome-blind pair panel."""

    started = time.perf_counter()
    mechanism = _spec(args)
    if args.seed_role != "development":
        raise ValueError("Trace-patch pair sampling is frozen on development rows")
    rows = _registered_rows(args, mechanism)
    plan = build_sparse_trace_patch_sample_plan(
        rows,
        model_label=args.model,
        donor_offsets=mechanism.trace_patch_donor_offsets,
        seeds_per_cell=mechanism.trace_patch_seeds_per_cell,
        sampling_seed=mechanism.trace_patch_sampling_seed,
        include_count2_terminal_panel=(
            mechanism.trace_patch_include_count2_terminal_panel
        ),
        candidate_counts=mechanism.candidate_counts,
    )
    output = Path(args.output)
    plan_path = output / "trace_patch_pair_plan.csv"
    _atomic_csv(plan_path, plan)
    cell_counts = (
        plan.groupby(
            ["panel_kind", "gold_count", "donor_offset"], as_index=False
        )
        .agg(
            pair_count=("pair_sha256", "nunique"),
            seed_count=("seed", "nunique"),
            receiver_count=("receiver_occurrence", "nunique"),
        )
        .sort_values(["panel_kind", "gold_count", "donor_offset"])
    )
    _atomic_csv(output / "trace_patch_cell_counts.csv", cell_counts)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "selection_policy": "outcome_blind_registry_identity_hash",
                "selection_input_fields": (
                    "sampling_seed,model_label,panel_kind,gold_count,"
                    "donor_offset,seed,request_id"
                ),
                "pair_count": int(len(plan)),
                "local_pair_count": int(plan["panel_kind"].eq("local").sum()),
                "terminal_pair_count": int(
                    plan["panel_kind"].eq("terminal").sum()
                ),
                "cell_count": int(len(cell_counts)),
                "pair_plan": str(plan_path.resolve()),
                "pair_plan_sha256": _sha256(plan_path),
            },
        ),
    )


def command_source_mask(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(row["request_id"], "answer_source_mask")
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        trial_rows = []
        for condition in args.conditions:
            try:
                result = run_answer_source_mask_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    condition=condition,
                    answer_site_id=mechanism.answer_site_id,
                    run_greedy=not args.skip_greedy,
                    max_new_tokens=args.max_new_tokens,
                )
                result["status"] = "ok"
            except ValueError as exc:
                if "not applicable" not in str(exc).lower():
                    raise
                result = {
                    "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                    "experiment_id": "answer_source_mask_factorial",
                    "condition": condition,
                    "status": "not_applicable",
                    "exclusion_reason": str(exc),
                    "request_id": row["request_id"],
                    "model_label": args.model,
                    "seed": int(row["seed"]),
                }
            result["mechanism_split"] = args.seed_role
            trial_rows.append(result)
        _atomic_jsonl(shard, trial_rows)
        completed += 1
        print(f"[count-stream source-mask] {index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={"newly_completed": completed, "resume_skipped": skipped},
        ),
    )


def _load_head_plan(path: Path, *, model: str, bank_size: int) -> pd.DataFrame:
    plan = pd.read_csv(path)
    selected = plan.loc[
        plan["model_label"].eq(model)
        & pd.to_numeric(plan["bank_size"], errors="coerce").eq(int(bank_size))
    ].copy()
    if selected.empty:
        raise ValueError(f"No {model}/K{bank_size} answer broad-head plan")
    required = {"clean", "selected_bank", "layer_matched_random"}
    if not required <= set(selected["condition"].astype(str)):
        raise ValueError("Head plan is missing registered causal arms")
    return selected


def command_select_broad_k(args: argparse.Namespace) -> None:
    """Freeze one model/source K without opening any confirmation outcome."""

    started = time.perf_counter()
    mechanism = _spec(args)
    rows = [row for file in _trial_files(args.trials) for row in read_jsonl(file)]
    curve, seed_effects, decision = select_answer_broad_bank_size(
        pd.DataFrame(rows),
        model_label=args.model,
        source_group=args.source_group,
        expected_seeds=mechanism.broad_k_selection_seeds,
        expected_bank_sizes=mechanism.development_bank_sizes,
        expected_requests_per_seed=mechanism.broad_panel_counts_per_seed,
        expected_random_controls=mechanism.random_controls,
        boundary_extension_bank_size=mechanism.boundary_extension_bank_size,
        bootstrap_samples=mechanism.bootstrap_samples,
        random_seed=args.random_seed,
    )
    output = Path(args.output)
    _atomic_csv(output / "k_discovery_curve.csv", curve)
    _atomic_csv(output / "k_discovery_seed_effects.csv", seed_effects)
    _atomic_json(output / "k_selection_decision.json", decision)
    frozen_plan_path: Path | None = None
    selected_k = decision.get("selected_bank_size")
    if decision["status"] == "frozen_for_confirmation" and selected_k is not None:
        plan = _load_head_plan(
            args.plan, model=args.model, bank_size=int(selected_k)
        ).copy()
        if set(plan["source_group"].astype(str)) != {str(args.source_group)}:
            raise ValueError("K decision and head plan use different source groups")
        plan["ranking_split"] = plan.get("selection_split", "ranking_discovery")
        plan["selection_split"] = "k_selection_discovery"
        plan["k_selection_status"] = "frozen_for_confirmation"
        plan["k_selection_decision_sha256"] = str(decision["decision_sha256"])
        plan["confirmation_locked"] = True
        frozen_plan_path = output / "frozen_answer_broad_head_plan.csv"
        _atomic_csv(frozen_plan_path, plan)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "model_label": args.model,
                "source_group": args.source_group,
                "selection_phase": "k_selection_discovery",
                "selection_seeds": list(mechanism.broad_k_selection_seeds),
                "confirmation_outcomes_used": False,
                "decision": decision,
                "frozen_plan": (
                    str(frozen_plan_path.resolve()) if frozen_plan_path else None
                ),
                "frozen_plan_sha256": (
                    _sha256(frozen_plan_path) if frozen_plan_path else None
                ),
            },
        ),
    )


def command_broad_heads(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    expected_panel = (
        "broad_confirmation"
        if args.seed_role == "confirmation"
        else "broad_k_selection"
    )
    if args.row_panel != expected_panel:
        raise ValueError(
            f"Broad-head {args.seed_role} runs require --row-panel "
            f"{expected_panel}"
        )
    rows = _registered_rows(args, mechanism)
    bank_sizes = tuple(
        int(value)
        for value in (
            args.bank_sizes
            if args.bank_sizes is not None
            else (args.bank_size,)
        )
    )
    if not bank_sizes or len(set(bank_sizes)) != len(bank_sizes):
        raise ValueError("Broad-head bank sizes must be nonempty and unique")
    if args.seed_role == "confirmation" and len(bank_sizes) != 1:
        raise ValueError("Confirmation opens exactly one discovery-frozen K")
    plan = pd.concat(
        [
            _load_head_plan(args.plan, model=args.model, bank_size=bank_size)
            for bank_size in bank_sizes
        ],
        ignore_index=True,
    )
    if args.seed_role == "confirmation":
        required_columns = {
            "k_selection_status",
            "k_selection_decision_sha256",
            "confirmation_locked",
        }
        missing_columns = sorted(required_columns - set(plan.columns))
        if missing_columns:
            raise ValueError(
                "Confirmation requires a discovery-frozen K plan; missing "
                f"{missing_columns}"
            )
        if set(plan["k_selection_status"].astype(str)) != {
            "frozen_for_confirmation"
        } or not plan["confirmation_locked"].astype(bool).all():
            raise ValueError("The answer broad-head plan is not confirmation-locked")
    source_groups = set(plan["source_group"].astype(str))
    if len(source_groups) != 1:
        raise ValueError("One answer-head run cannot mix source groups")
    source_group = next(iter(source_groups))
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(
            row["request_id"], source_group, "K", *sorted(bank_sizes)
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = []
        for plan_row in plan.itertuples(index=False):
            heads = [
                tuple(int(item) for item in pair) for pair in json.loads(plan_row.heads)
            ]
            result = run_answer_broad_head_trial(
                model,
                tokenizer,
                adapter,
                row,
                heads=heads,
                condition=str(plan_row.condition),
                source_group=source_group,
                answer_site_id=mechanism.answer_site_id,
                run_greedy=not args.skip_greedy,
                max_new_tokens=args.max_new_tokens,
            )
            result.update(
                {
                    "status": "ok",
                    "repeat": int(plan_row.repeat),
                    "bank_size": int(plan_row.bank_size),
                    "realized_ablated_head_count": len(heads),
                    "planned_bank_sha256": str(plan_row.bank_sha256),
                    "selection_metric": str(plan_row.selection_metric),
                    "mechanism_split": args.seed_role,
                    "plan_sha256": _sha256(args.plan),
                }
            )
            results.append(result)
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[count-stream broad-heads] {index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "plan": str(args.plan.resolve()),
                "plan_sha256": _sha256(args.plan),
                "source_group": source_group,
                "bank_sizes": list(bank_sizes),
            },
        ),
    )


def command_fit_basis(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    metadata, states = load_count_stream_capture_dataset(
        args.capture_index,
        site_kinds=[args.site_kind],
    )
    mask = count_stream_cohort_mask(metadata, args.cohort)
    mask &= metadata["seed"].astype(int).isin(mechanism.development_seeds).to_numpy()
    if args.site_id:
        mask &= metadata["site_id"].astype(str).eq(args.site_id).to_numpy()
    available_layers = sorted(
        int(value) for value in metadata.loc[mask, "layer"].unique()
    )
    layers = available_layers if args.layers is None else sorted(set(args.layers))
    missing = sorted(set(layers) - set(available_layers))
    if missing:
        raise ValueError(f"Basis layers are unavailable: {missing}")
    arrays: dict[str, np.ndarray] = {}
    fit_rows = []
    for layer in layers:
        layer_mask = mask & metadata["layer"].astype(int).eq(layer).to_numpy()
        layer_states = np.asarray(states[layer_mask], dtype=np.float32)
        labels = metadata.loc[layer_mask, args.label].to_numpy(dtype=int)
        observed_seeds = sorted(
            int(value) for value in metadata.loc[layer_mask, "seed"].unique()
        )
        if len(np.unique(labels)) < 2:
            raise ValueError(f"L{layer} has fewer than two {args.label} classes")
        center, basis, control = fit_count_stream_basis(
            layer_states,
            labels,
            rank=args.rank,
            seed=args.random_seed + int(layer),
        )
        arrays[f"center_L{layer}"] = center
        arrays[f"basis_L{layer}"] = basis
        arrays[f"control_basis_L{layer}"] = control
        fit_rows.append(
            {
                "layer": int(layer),
                "observations": int(len(layer_states)),
                "labels": sorted(int(value) for value in np.unique(labels)),
                "development_seeds": observed_seeds,
                "effective_rank": int(basis.shape[1]),
                "basis_control_max_abs_dot": float(np.max(np.abs(basis.T @ control))),
            }
        )
    _atomic_npz(args.output, **arrays)
    audit = _runtime_manifest(
        args,
        mechanism=mechanism,
        started=started,
        completed_shards=0,
        extra={
            "capture_index": str(args.capture_index.resolve()),
            "capture_index_sha256": _sha256(args.capture_index),
            "site_kind": args.site_kind,
            "site_id": args.site_id,
            "cohort": args.cohort,
            "label": args.label,
            "selection_role": "development_only",
            "confirmation_used_for_fit": False,
            "fits": fit_rows,
            "artifact_sha256": _sha256(args.output),
        },
    )
    _atomic_json(args.output.with_suffix(".json"), audit)


def _load_basis(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = (
            f"center_L{layer}",
            f"basis_L{layer}",
            f"control_basis_L{layer}",
        )
        missing = [key for key in keys if key not in archive.files]
        if missing:
            raise ValueError(f"Basis artifact is missing {missing}")
        return tuple(np.asarray(archive[key], dtype=np.float32) for key in keys)  # type: ignore[return-value]


def _validate_trace_patch_basis_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Trace patching requires the fit-basis sidecar manifest: "
            f"{manifest_path}"
        )
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Basis sidecar manifest must contain one JSON object")
    if str(value.get("artifact_sha256")) != _sha256(path):
        raise ValueError("Basis artifact hash disagrees with its sidecar manifest")
    if str(value.get("site_kind")) != "item_end":
        raise ValueError("Trace patch basis must be fitted at item_end sites")
    if str(value.get("label")) != "occurrence":
        raise ValueError("Trace patch basis must use the occurrence label")
    if value.get("confirmation_used_for_fit") is not False:
        raise ValueError("Trace patch basis selection must exclude confirmation data")
    return manifest_path, value


def _load_readout_bases(path: Path, layers: Iterable[int]) -> dict[int, np.ndarray]:
    selected = tuple(sorted({int(value) for value in layers}))
    with np.load(path, allow_pickle=False) as archive:
        missing = [
            int(layer)
            for layer in selected
            if f"basis_L{int(layer)}" not in archive.files
        ]
        if missing:
            raise ValueError(
                "Basis artifact lacks downstream item-end readout bases for "
                f"layers {missing}; refit with every source/readout layer"
            )
        return {
            int(layer): np.asarray(archive[f"basis_L{int(layer)}"], dtype=np.float32)
            for layer in selected
        }


def _validated_trace_patch_plan(
    path: Path,
    *,
    mechanism: NativeCountMechanismSpec,
    rows: list[dict[str, Any]],
    model_label: str,
) -> pd.DataFrame:
    """Reconstruct and verify the frozen sampling plan before GPU work."""

    plan = pd.read_csv(path)
    needed = {
        "schema_version",
        "model_label",
        "panel_kind",
        "request_id",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "donor_direction",
        "selection_cell_id",
        "selection_rank",
        "sampling_seed",
        "seeds_per_cell",
        "pair_sha256",
    }
    missing = sorted(needed - set(plan.columns))
    if missing:
        raise ValueError(f"Trace-patch pair plan is missing {missing}")
    plan = plan.loc[plan["model_label"].astype(str).eq(str(model_label))].copy()
    if plan.empty:
        raise ValueError(f"Pair plan has no rows for {model_label}")
    for column in (
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "selection_rank",
        "sampling_seed",
        "seeds_per_cell",
    ):
        plan[column] = pd.to_numeric(plan[column], errors="raise").astype(int)
    if set(plan["schema_version"].astype(str)) != {COUNT_STREAM_SCHEMA_VERSION}:
        raise ValueError("Pair plan uses a different count-stream schema")
    if plan["pair_sha256"].astype(str).duplicated().any():
        raise ValueError("Pair plan contains duplicate pair hashes")
    expected = build_sparse_trace_patch_sample_plan(
        rows,
        model_label=model_label,
        donor_offsets=mechanism.trace_patch_donor_offsets,
        seeds_per_cell=mechanism.trace_patch_seeds_per_cell,
        sampling_seed=mechanism.trace_patch_sampling_seed,
        include_count2_terminal_panel=(
            mechanism.trace_patch_include_count2_terminal_panel
        ),
        candidate_counts=mechanism.candidate_counts,
    )
    identity_columns = [
        "panel_kind",
        "selection_cell_id",
        "selection_rank",
        "request_id",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "donor_direction",
        "sampling_seed",
        "seeds_per_cell",
        "pair_sha256",
    ]
    left = plan[identity_columns].sort_values("pair_sha256").reset_index(drop=True)
    right = expected[identity_columns].sort_values("pair_sha256").reset_index(
        drop=True
    )
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            "Pair plan does not match the outcome-blind plan reconstructed from "
            "the active registry/config"
        ) from exc
    return plan.sort_values(
        ["panel_kind", "gold_count", "donor_offset", "selection_rank"]
    ).reset_index(drop=True)


def command_stream_state(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    center, basis, _control = _load_basis(args.basis, args.source_layer)
    active_readout_layers = tuple(
        sorted({int(value) for value in args.readout_layers})
    ) or (int(args.source_layer) + 1,)
    readout_bases = _load_readout_bases(args.basis, active_readout_layers)
    if "clean" not in args.conditions:
        raise ValueError(
            "Stream-state runs require the clean arm for downstream retention metrics"
        )
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Stream-state conditions must be unique")
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    state_dir = output / "states"
    completed = skipped = 0
    for row_index, row in enumerate(rows, start=1):
        occurrences: list[int | None]
        if args.scope == "all":
            occurrences = [None]
        elif args.occurrences:
            occurrences = [int(value) for value in args.occurrences]
        else:
            count = int(row.get("gold_count", len(row.get("gold_records", []))))
            occurrences = list(range(1, count + 1))
        for occurrence in occurrences:
            stem = _safe_stem(
                row["request_id"], "stream", args.scope, occurrence, args.source_layer
            )
            shard = shard_dir / f"{stem}.jsonl"
            if args.resume and shard.exists():
                skipped += 1
                continue
            results = []
            states_by_condition: dict[str, np.ndarray] = {}
            for condition in args.conditions:
                result = run_stream_state_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    condition=condition,
                    source_layer=args.source_layer,
                    center=center,
                    basis=basis,
                    scope=args.scope,
                    occurrence=occurrence,
                    readout_layers=active_readout_layers,
                    answer_site_id=mechanism.answer_site_id,
                    random_seed=args.random_seed
                    + int(row["seed"])
                    + (0 if occurrence is None else int(occurrence)),
                    run_greedy=not args.skip_greedy,
                    max_new_tokens=args.max_new_tokens,
                )
                states = np.asarray(result.pop("readout_states"), dtype=np.float32)
                states_by_condition[str(condition)] = states
                state_path = state_dir / f"{stem}__{condition}.npz"
                _atomic_npz(
                    state_path,
                    states=states,
                    layers=np.asarray(result["readout_layers"], dtype=np.int64),
                    positions=np.asarray(result["readout_positions"], dtype=np.int64),
                )
                result.update(
                    {
                        "status": "ok",
                        "mechanism_split": args.seed_role,
                        "basis": str(args.basis.resolve()),
                        "basis_sha256": _sha256(args.basis),
                        "readout_states_path": str(state_path.relative_to(output)),
                    }
                )
                results.append(result)
            clean_states = states_by_condition["clean"]
            for result in results:
                condition = str(result["condition"])
                result.update(
                    stream_state_retention_metrics(
                        clean_states,
                        states_by_condition[condition],
                        readout_layers=result["readout_layers"],
                        readout_positions=result["readout_positions"],
                        query_position=max(result["readout_positions"]),
                        count_bases=readout_bases,
                    )
                )
            _atomic_jsonl(shard, results)
            completed += 1
        print(f"[count-stream stream-state] {row_index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "source_layer": int(args.source_layer),
                "source_scope": args.scope,
                "readout_layers": list(active_readout_layers),
                "readout_basis_site_kind": "item_end",
            },
        ),
    )


def command_trace_patch(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    pair_plan = _validated_trace_patch_plan(
        args.pair_plan,
        mechanism=mechanism,
        rows=rows,
        model_label=args.model,
    )
    planned_request_ids = set(pair_plan["request_id"].astype(str))
    rows = [row for row in rows if str(row["request_id"]) in planned_request_ids]
    row_by_request = {str(row["request_id"]): row for row in rows}
    missing_requests = sorted(planned_request_ids - set(row_by_request))
    if missing_requests:
        raise ValueError(
            f"Pair plan references {len(missing_requests)} absent registry rows"
        )
    basis_manifest_path, basis_manifest = _validate_trace_patch_basis_manifest(
        args.basis
    )
    fitted_mechanism = basis_manifest.get("mechanism_spec")
    if not isinstance(fitted_mechanism, dict):
        raise ValueError("Basis sidecar has no resolved mechanism specification")
    if tuple(int(value) for value in fitted_mechanism.get("development_seeds", ())) != (
        mechanism.development_seeds
    ):
        raise ValueError(
            "Trace patch basis and active mechanism config use different "
            "development seeds"
        )
    _center, basis, _control = _load_basis(args.basis, args.layer)
    active_readout_layers = tuple(
        sorted({int(value) for value in args.readout_layers})
    ) or (int(args.layer) + 1,)
    readout_bases = _load_readout_bases(args.basis, active_readout_layers)
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Trace patch conditions must be unique")
    if not {"clean", "self_patch"} <= set(args.conditions):
        raise ValueError("Trace patch runs require clean and self_patch controls")
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for pair_index, pair in enumerate(pair_plan.itertuples(index=False), start=1):
        row = row_by_request[str(pair.request_id)]
        receiver = int(pair.receiver_occurrence)
        donor = int(pair.donor_occurrence)
        panel_kind = str(pair.panel_kind)
        stem = _safe_stem(
            row["request_id"], panel_kind, receiver, donor, args.layer
        )
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        common_kwargs = {
            "receiver_occurrence": receiver,
            "donor_occurrence": donor,
            "layer": args.layer,
            "basis": basis,
            "readout_layers": active_readout_layers,
            "readout_bases": readout_bases,
            "conditions": args.conditions,
            "answer_site_id": mechanism.answer_site_id,
            "random_seed": (
                int(args.random_seed)
                + int(row["seed"]) * 101
                + receiver * 17
                + donor
            ),
            "run_greedy": not args.skip_greedy,
            "max_new_tokens": args.max_new_tokens,
        }
        if panel_kind == "local":
            results = run_trace_intermediate_patch_trials(
                model,
                tokenizer,
                adapter,
                row,
                **common_kwargs,
            )
        elif panel_kind == "terminal":
            results = run_trace_terminal_patch_trials(
                model,
                tokenizer,
                adapter,
                row,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown trace-patch panel kind: {panel_kind}")
        for result in results:
            result.update(
                {
                    "mechanism_split": args.seed_role,
                    "panel_kind": panel_kind,
                    "selection_cell_id": str(pair.selection_cell_id),
                    "selection_rank": int(pair.selection_rank),
                    "pair_sha256": str(pair.pair_sha256),
                    "pair_plan": str(args.pair_plan.resolve()),
                    "pair_plan_sha256": _sha256(args.pair_plan),
                    "basis": str(args.basis.resolve()),
                    "basis_sha256": _sha256(args.basis),
                    "basis_manifest_sha256": _sha256(basis_manifest_path),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        print(
            f"[count-stream trace-patch] {pair_index}/{len(pair_plan)}",
            flush=True,
        )
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "planned_pair_count": int(len(pair_plan)),
                "local_pair_count": int(
                    pair_plan["panel_kind"].eq("local").sum()
                ),
                "terminal_pair_count": int(
                    pair_plan["panel_kind"].eq("terminal").sum()
                ),
                "pair_plan": str(args.pair_plan.resolve()),
                "pair_plan_sha256": _sha256(args.pair_plan),
                "basis": str(args.basis.resolve()),
                "basis_sha256": _sha256(args.basis),
                "basis_manifest": str(basis_manifest_path.resolve()),
                "basis_manifest_sha256": _sha256(basis_manifest_path),
                "basis_fit_label": str(basis_manifest["label"]),
                "basis_fit_site_kind": str(basis_manifest["site_kind"]),
                "patch_layer": int(args.layer),
                "readout_layers": list(active_readout_layers),
                "donor_offsets": list(mechanism.trace_patch_donor_offsets),
                "conditions": list(args.conditions),
                "primary_temporal_direction": mechanism.trace_patch_primary_direction,
                "primary_outcome": mechanism.trace_patch_primary_outcome,
                "future_to_past_role": "counterfactual_representational_sensitivity",
            },
        ),
    )


def command_restoration(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = _registered_rows(args, mechanism)
    model, tokenizer, adapter = _model(args)
    output = Path(args.output)
    shard_dir = _prepare_shards(output, resume=args.resume, suffix="jsonl")
    completed = skipped = 0
    for index, row in enumerate(rows, start=1):
        stem = _safe_stem(row["request_id"], "restoration", args.layer)
        shard = shard_dir / f"{stem}.jsonl"
        if args.resume and shard.exists():
            skipped += 1
            continue
        results = run_trace_restoration_trials(
            model,
            tokenizer,
            adapter,
            row,
            layer=args.layer,
            conditions=args.conditions,
            answer_site_id=mechanism.answer_site_id,
            run_greedy=not args.skip_greedy,
            max_new_tokens=args.max_new_tokens,
        )
        for result in results:
            result["mechanism_split"] = args.seed_role
        _atomic_jsonl(shard, results)
        completed += 1
        print(f"[count-stream restoration] {index}/{len(rows)}", flush=True)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            extra={
                "newly_completed": completed,
                "resume_skipped": skipped,
                "restoration_layer": int(args.layer),
            },
        ),
    )


def _trial_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif (path / "shards").is_dir():
            files.extend(sorted((path / "shards").glob("*.jsonl")))
        else:
            files.extend(sorted(path.rglob("*.jsonl")))
    if not files:
        raise FileNotFoundError("No JSONL trial shards were found")
    return files


def command_analyze(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    mechanism = _spec(args)
    rows = [row for file in _trial_files(args.trials) for row in read_jsonl(file)]
    trials = pd.DataFrame(rows)
    experiments = (
        args.experiment_ids
        if args.experiment_ids
        else sorted(trials["experiment_id"].dropna().astype(str).unique())
    )
    summaries = []
    effects = []
    for experiment_id in experiments:
        strata = list(args.strata)
        if (
            experiment_id == "answer_broad_head_ablation"
            and "source_group" in trials.columns
            and "source_group" not in strata
        ):
            strata.append("source_group")
        if (
            experiment_id == "answer_broad_head_ablation"
            and "bank_size" in trials.columns
            and trials.loc[
                trials["experiment_id"].eq(experiment_id), "bank_size"
            ].nunique(dropna=True)
            > 1
            and "bank_size" not in strata
        ):
            # Multiple K values are development curves, not one registered
            # confirmation estimand.  Keep them separate so the decision
            # ledger marks an unfrozen source gate as ambiguous.
            strata.append("bank_size")
        if (
            experiment_id
            in {
                "trace_intermediate_state_patching",
                "trace_terminal_state_patching",
            }
            and "donor_direction" in trials.columns
            and "donor_direction" not in strata
        ):
            strata.append("donor_direction")
        summary, seed_effects = summarize_linear_contrasts(
            trials,
            experiment_id=experiment_id,
            outcome=args.outcome,
            bootstrap_samples=mechanism.bootstrap_samples,
            random_seed=args.random_seed,
            stratum_columns=strata,
        )
        summaries.append(summary)
        effects.append(seed_effects)
    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_effects = pd.concat(effects, ignore_index=True)
    ledger = mechanism_decision_ledger(combined_summary)
    ledger["design_status"] = mechanism.status
    ledger["formal_inference_eligible"] = mechanism.formal_inference_eligible
    ledger["claim_scope"] = (
        "fresh_confirmation"
        if mechanism.formal_inference_eligible
        else "development_only_no_confirmatory_claim"
    )
    output = Path(args.output)
    _atomic_csv(output / "estimands.csv", combined_summary)
    _atomic_csv(output / "seed_effects.csv", combined_effects)
    _atomic_csv(output / "mechanism_decision_ledger.csv", ledger)
    trial_files = _trial_files(args.trials)
    _atomic_json(
        output / "manifest.json",
        _runtime_manifest(
            args,
            mechanism=mechanism,
            started=started,
            completed_shards=0,
            extra={
                "trial_files": [str(path.resolve()) for path in trial_files],
                "experiment_ids": list(experiments),
                "outcome": args.outcome,
                "estimands_sha256": _sha256(output / "estimands.csv"),
                "claim_policy": (
                    "Decision ledger reports registered component gates only; "
                    "it never upgrades them to a unique-circuit or scalar-counter claim."
                ),
            },
        ),
    )


def _add_configs(parser: argparse.ArgumentParser, *, include_v5: bool = True) -> None:
    parser.add_argument(
        "--mechanism-config", type=Path, default=DEFAULT_MECHANISM_CONFIG
    )
    if include_v5:
        parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")


def _add_rows(parser: argparse.ArgumentParser) -> None:
    _add_configs(parser)
    _add_model(parser)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--seed-role",
        choices=["development", "confirmation"],
        default="development",
    )
    parser.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
        help=(
            "Native trace cohort; one_to_one is the registered default, while "
            "one_to_one_correct is secondary robustness only."
        ),
    )
    parser.add_argument(
        "--row-panel",
        choices=[
            "all",
            "broad_ranking",
            "broad_k_selection",
            "broad_confirmation",
            "trace_patch",
        ],
        default="all",
        help="Frozen seed/count sampling panel; all is for legacy/dev diagnostics.",
    )
    parser.add_argument("--limit", type=int)


def _add_output_resume(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)


def _add_behavior(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Native-thinking count-stream versus final broad-retrieval mechanism"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture-broad",
        help="Capture answer-query attention to prompt/trace source partitions.",
    )
    _add_rows(capture)
    _add_output_resume(capture)
    capture.add_argument("--layers", type=int, nargs="+")
    capture.set_defaults(
        func=command_capture_broad,
        cohort="parser_hit",
        row_panel="broad_ranking",
        seed_role="development",
    )

    plan = subparsers.add_parser(
        "plan-broad",
        help="Freeze development-ranked broad banks and exact matched controls.",
    )
    _add_configs(plan, include_v5=False)
    plan.add_argument("--captures", type=Path, required=True)
    plan.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    plan.add_argument(
        "--source-group",
        required=True,
        choices=[
            "prompt_records",
            "trace_context",
            "trace_items",
            "trace_other",
            "trace_markers",
            "trace_nonmarkers",
            "earlier_trace_items",
            "terminal_trace_item",
        ],
    )
    plan.add_argument("--bank-sizes", type=int, nargs="+")
    plan.add_argument("--random-seed", type=int, default=0)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(func=command_plan_broad)

    trace_plan = subparsers.add_parser(
        "plan-trace-patch",
        help="Freeze the outcome-blind sparse directed trace-patch pair panel.",
    )
    _add_rows(trace_plan)
    trace_plan.add_argument("--output", type=Path, required=True)
    trace_plan.set_defaults(
        func=command_plan_trace_patch,
        row_panel="trace_patch",
        seed_role="development",
    )

    select_k = subparsers.add_parser(
        "select-broad-k",
        help="Select and freeze one model/source K from development dose outcomes.",
    )
    _add_configs(select_k, include_v5=False)
    select_k.add_argument("--trials", type=Path, nargs="+", required=True)
    select_k.add_argument("--plan", type=Path, required=True)
    select_k.add_argument(
        "--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"]
    )
    select_k.add_argument(
        "--source-group", required=True, choices=["trace_items", "prompt_records"]
    )
    select_k.add_argument("--random-seed", type=int, default=0)
    select_k.add_argument("--output", type=Path, required=True)
    select_k.set_defaults(func=command_select_broad_k)

    source_mask = subparsers.add_parser(
        "source-mask",
        help="Run the final-query prompt/trace source factorial.",
    )
    _add_rows(source_mask)
    _add_output_resume(source_mask)
    _add_behavior(source_mask)
    source_mask.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_MASK_CONDITIONS),
        default=list(REGISTERED_MASK_CONDITIONS),
    )
    source_mask.set_defaults(func=command_source_mask)

    broad_heads = subparsers.add_parser(
        "broad-heads",
        help="Ablate a frozen answer-query broad bank and matched random banks.",
    )
    _add_rows(broad_heads)
    _add_output_resume(broad_heads)
    _add_behavior(broad_heads)
    broad_heads.add_argument("--plan", type=Path, required=True)
    broad_size_group = broad_heads.add_mutually_exclusive_group(required=True)
    broad_size_group.add_argument("--bank-size", type=int)
    broad_size_group.add_argument("--bank-sizes", type=int, nargs="+")
    broad_heads.set_defaults(
        func=command_broad_heads,
        cohort="parser_hit",
        row_panel="broad_k_selection",
    )

    fit_basis = subparsers.add_parser(
        "fit-basis",
        help="Fit a development-only progress/count basis plus orthogonal control.",
    )
    _add_configs(fit_basis, include_v5=False)
    fit_basis.add_argument("--capture-index", type=Path, required=True)
    fit_basis.add_argument("--site-kind", required=True)
    fit_basis.add_argument("--site-id")
    fit_basis.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    fit_basis.add_argument(
        "--label", choices=["occurrence", "gold_count"], required=True
    )
    fit_basis.add_argument("--layers", type=int, nargs="+")
    fit_basis.add_argument("--rank", type=int, default=3)
    fit_basis.add_argument("--random-seed", type=int, default=0)
    fit_basis.add_argument("--output", type=Path, required=True)
    fit_basis.set_defaults(func=command_fit_basis)

    stream = subparsers.add_parser(
        "stream-state",
        help="Remove running-index state at item endpoints and follow the suffix.",
    )
    _add_rows(stream)
    _add_output_resume(stream)
    _add_behavior(stream)
    stream.add_argument("--basis", type=Path, required=True)
    stream.add_argument("--source-layer", type=int, required=True)
    stream.add_argument("--readout-layers", type=int, nargs="+", default=[])
    stream.add_argument(
        "--scope", choices=["occurrence", "prefix", "all"], default="occurrence"
    )
    stream.add_argument("--occurrences", type=int, nargs="+")
    stream.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_STREAM_CONDITIONS),
        default=list(REGISTERED_STREAM_CONDITIONS),
    )
    stream.add_argument("--random-seed", type=int, default=0)
    stream.set_defaults(func=command_stream_state)

    trace_patch = subparsers.add_parser(
        "trace-patch",
        help=("Patch directed donor states between intermediate trace item endpoints."),
    )
    _add_rows(trace_patch)
    _add_output_resume(trace_patch)
    _add_behavior(trace_patch)
    trace_patch.add_argument("--pair-plan", type=Path, required=True)
    trace_patch.add_argument("--basis", type=Path, required=True)
    trace_patch.add_argument("--layer", type=int, required=True)
    trace_patch.add_argument("--readout-layers", type=int, nargs="+", default=[])
    trace_patch.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_TRACE_PATCH_CONDITIONS),
        default=list(REGISTERED_TRACE_PATCH_CONDITIONS),
    )
    trace_patch.add_argument("--random-seed", type=int, default=0)
    trace_patch.set_defaults(
        func=command_trace_patch,
        max_new_tokens=48,
        cohort="one_to_one",
        row_panel="trace_patch",
    )

    restoration = subparsers.add_parser(
        "restoration",
        help="Corrupt trace tokens and restore clean span/endpoint/marker states.",
    )
    _add_rows(restoration)
    _add_output_resume(restoration)
    _add_behavior(restoration)
    restoration.add_argument("--layer", type=int, required=True)
    restoration.add_argument(
        "--conditions",
        nargs="+",
        choices=list(REGISTERED_RESTORATION_CONDITIONS),
        default=list(REGISTERED_RESTORATION_CONDITIONS),
    )
    restoration.set_defaults(func=command_restoration)

    analyze = subparsers.add_parser(
        "analyze",
        help="Compute request-first, seed-equal registered contrasts and gates.",
    )
    _add_configs(analyze, include_v5=False)
    analyze.add_argument("--trials", type=Path, nargs="+", required=True)
    analyze.add_argument(
        "--experiment-ids",
        nargs="+",
        choices=[
            "answer_source_mask_factorial",
            "answer_broad_head_ablation",
            "trace_intermediate_state_patching",
            "trace_terminal_state_patching",
            "stream_state_retention",
            "trace_source_restoration",
        ],
    )
    analyze.add_argument(
        "--outcome",
        choices=list(REGISTERED_HIGHER_IS_BETTER_OUTCOMES),
        default="correct_count_margin",
        help=(
            "Registered higher-is-better endpoint. Error-valued columns are "
            "excluded to preserve the positive damage/repair orientation."
        ),
    )
    analyze.add_argument(
        "--strata",
        nargs="+",
        default=[],
        choices=[
            "source_group",
            "source_scope",
            "source_occurrences",
            "source_layer",
            "layer",
            "bank_size",
            "donor_direction",
            "receiver_is_visible_marker_token",
        ],
    )
    analyze.add_argument("--random-seed", type=int, default=0)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
