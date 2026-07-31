from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from .attention import (
    analyze_attention_table,
    capture_attention_shards,
    load_attention_shards,
    load_head_ranking,
)
from .causal import (
    compare_head_ablation_to_random,
    run_head_ablation_experiment,
    run_hidden_patching_experiment,
    summarize_head_ablation,
    summarize_hidden_patching,
)
from .modeling import (
    DecoderAdapter,
    capture_post_block_states,
    capture_span_states,
    load_registered_model,
    query_attention_outputs,
    run_last_logits,
    run_with_head_ablation,
    run_with_residual_patch,
)
from .prompts import PromptEncoding, render_v4_prompt
from .representation import (
    analyze_representation_captures,
    capture_representation_shards,
)
from .spec import V4Config, resolve_fractional_layers, resolve_model_spec
from .stimuli import load_stimuli


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        row = {
            "event": str(event),
            "unix_time": time.time(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"[v4] {event} "
            + " ".join(f"{key}={value}" for key, value in payload.items()),
            flush=True,
        )

    @contextmanager
    def timer(self, name: str, **payload: Any) -> Iterator[None]:
        start = time.perf_counter()
        self.write(f"{name}_start", **payload)
        try:
            yield
        except Exception as exc:
            self.write(
                f"{name}_failed",
                elapsed_seconds=time.perf_counter() - start,
                error_type=type(exc).__name__,
                error=str(exc),
                **payload,
            )
            raise
        self.write(
            f"{name}_complete",
            elapsed_seconds=time.perf_counter() - start,
            **payload,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status_short": status,
    }


def write_runtime_provenance(
    *,
    output_dir: str | Path,
    config_path: str | Path,
    stimuli_path: str | Path,
    model_label: str,
    answer_format: str,
    repo_root: str | Path,
) -> Path:
    import scipy
    import sklearn
    import transformers

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_spec = resolve_model_spec(model_label)
    payload = {
        "schema_version": "realistic_niah_v4_runtime_provenance_v1",
        "model": {
            "label": model_spec.label,
            "model_id": model_spec.model_id,
            "revision": model_spec.revision,
            "loader_class": model_spec.loader_class,
        },
        "answer_format": str(answer_format),
        "inputs": {
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": _sha256_file(Path(config_path)),
            "stimuli_path": str(Path(stimuli_path).resolve()),
            "stimuli_sha256": _sha256_file(Path(stimuli_path)),
        },
        "runtime": {
            "command": list(sys.argv),
            "working_directory": str(Path.cwd()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "git": _git_state(repo_root),
    }
    path = output / "runtime_provenance.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_config = V4Config.from_json(config_path)
    (output / "config.resolved.json").write_text(
        json.dumps(
            resolved_config.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def select_stimuli(
    stimuli_path: str | Path,
    *,
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    rows = load_stimuli(stimuli_path)
    variant_set = None if variants is None else {str(value) for value in variants}
    seed_set = None if seeds is None else {int(value) for value in seeds}
    count_set = None if counts is None else {int(value) for value in counts}
    selected = [
        row
        for row in rows
        if (
            (variant_set is None or str(row["design_variant"]) in variant_set)
            and (seed_set is None or int(row["seed"]) in seed_set)
            and (count_set is None or int(row["gold_count"]) in count_set)
            and (split is None or str(row["split"]) == str(split))
        )
    ]
    if not selected:
        raise ValueError("V4 stimulus filters selected no rows")
    return selected


def _write_csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _write_shard_index(
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_csv_gzip_shards(
    index_rows: Sequence[dict[str, Any]],
    *,
    root: Path,
) -> pd.DataFrame:
    if not index_rows:
        raise ValueError("No V4 causal shards were indexed")
    return pd.concat(
        [
            pd.read_csv(
                root / str(row["shard_path"]),
                compression="gzip",
            )
            for row in index_rows
        ],
        ignore_index=True,
    )


def render_encodings(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    model_label: str,
    config: V4Config,
    answer_format: str,
) -> Iterator[PromptEncoding]:
    model_spec = resolve_model_spec(model_label)
    for row in rows:
        yield render_v4_prompt(
            row,
            tokenizer=tokenizer,
            model_spec=model_spec,
            config=config,
            answer_format=answer_format,
        )


def preflight_report(
    *,
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    representative_rows: Sequence[dict[str, Any]],
    model_label: str,
    config: V4Config,
    answer_format: str,
    forward_smoke: bool = False,
) -> dict[str, Any]:
    encodings = list(
        render_encodings(
            representative_rows,
            tokenizer=tokenizer,
            model_label=model_label,
            config=config,
            answer_format=answer_format,
        )
    )
    sequence_lengths = [encoding.sequence_length for encoding in encodings]
    text_config = getattr(model.config, "text_config", model.config)
    max_positions = getattr(text_config, "max_position_embeddings", None)
    if max_positions is not None and max(sequence_lengths) > int(max_positions):
        raise RuntimeError(
            f"Rendered prompt length {max(sequence_lengths)} exceeds "
            f"max_position_embeddings={max_positions}"
        )
    forward = None
    if forward_smoke:
        primary = encodings[0]
        logits = run_last_logits(model, primary)
        first_candidate_ids = sorted(
            {
                int(token_ids[0])
                for _, token_ids in primary.count_candidate_answer_token_ids
            }
        )
        attention_rows, key_starts, attention_logits = query_attention_outputs(
            model,
            adapter,
            primary,
        )
        smoke_layers = sorted({0, adapter.num_layers - 1})
        first_span = primary.needle_spans[0]
        captured = capture_span_states(
            model,
            adapter,
            primary,
            spans=[first_span],
            layers=smoke_layers,
        )
        ablated_logits = run_with_head_ablation(
            model,
            adapter,
            primary,
            [(0, 0)],
            scope="answer_query",
        )
        donor_positions = [primary.query_position] + list(
            range(first_span.start, first_span.end)
        )
        _donor_logits, donor_states = capture_post_block_states(
            model,
            adapter,
            primary,
            donor_positions,
            layers=smoke_layers,
        )
        first_layer = smoke_layers[0]
        last_layer = smoke_layers[-1]
        query_patched = run_with_residual_patch(
            model,
            adapter,
            primary,
            layer=last_layer,
            receiver_positions=[primary.query_position],
            donor_states=donor_states[last_layer][0:1] + 0.01,
        )
        end_patched = run_with_residual_patch(
            model,
            adapter,
            primary,
            layer=first_layer,
            receiver_positions=[first_span.end - 1],
            donor_states=donor_states[first_layer][-1:] + 0.01,
        )
        span_patched = run_with_residual_patch(
            model,
            adapter,
            primary,
            layer=first_layer,
            receiver_positions=list(range(first_span.start, first_span.end)),
            donor_states=donor_states[first_layer][1:] + 0.01,
        )
        smoke_logits = {
            "baseline": logits,
            "head_ablation": ablated_logits,
            "query_patch": query_patched,
            "needle_end_patch": end_patched,
            "needle_span_patch": span_patched,
        }
        if not all(
            bool(torch.isfinite(values[first_candidate_ids]).all())
            for values in smoke_logits.values()
        ):
            raise RuntimeError("Preflight produced a non-finite first-token logit")
        forward = {
            "vocabulary_size": int(logits.numel()),
            "finite_candidate_logits": True,
            "query_attention_shapes": [
                [int(value) for value in row.shape] for row in attention_rows
            ],
            "query_attention_key_starts": [int(value) for value in key_starts],
            "query_attention_row_sum_range": [
                float(min(row.sum(dim=-1).min().item() for row in attention_rows)),
                float(max(row.sum(dim=-1).max().item() for row in attention_rows)),
            ],
            "query_cache_vs_full_candidate_logit_max_abs_delta": float(
                torch.max(
                    torch.abs(
                        attention_logits[first_candidate_ids]
                        - logits[first_candidate_ids]
                    )
                ).item()
            ),
            "span_capture_shapes": {
                key: [int(value) for value in tensor.shape]
                for key, tensor in captured.items()
            },
            "smoke_layers": smoke_layers,
            "head_ablation_max_abs_logit_delta": float(
                torch.max(torch.abs(ablated_logits - logits)).item()
            ),
            "query_patch_max_abs_logit_delta": float(
                torch.max(torch.abs(query_patched - logits)).item()
            ),
            "needle_end_patch_max_abs_logit_delta": float(
                torch.max(torch.abs(end_patched - logits)).item()
            ),
            "needle_span_patch_max_abs_logit_delta": float(
                torch.max(torch.abs(span_patched - logits)).item()
            ),
        }
    return {
        "schema_version": "realistic_niah_v4_preflight_v1",
        "model_label": model_label,
        "answer_format": str(answer_format),
        "decoder_layer_container": adapter.layer_container_name,
        "num_layers": adapter.num_layers,
        "num_heads_by_layer": list(adapter.num_heads),
        "head_dims_by_layer": list(adapter.head_dims),
        "layer_types": list(adapter.layer_types),
        "rendered_sequence_lengths": sequence_lengths,
        "max_position_embeddings": (
            None if max_positions is None else int(max_positions)
        ),
        "count_candidate_texts": {
            str(count): text for count, text in encodings[0].count_candidate_texts
        },
        "count_candidate_answer_token_ids": {
            str(count): list(token_ids)
            for count, token_ids in encodings[0].count_candidate_answer_token_ids
        },
        "count_candidate_scored_token_ids": {
            str(count): list(token_ids)
            for count, token_ids in encodings[0].count_candidate_token_ids
        },
        "query_is_last_token": all(
            encoding.query_position == encoding.sequence_length - 1
            for encoding in encodings
        ),
        "forward_smoke": forward,
    }


def run_model_stage(
    *,
    stage: str,
    stimuli_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_label: str,
    answer_format: str,
    cache_dir: str | Path | None = None,
    device_map: str = "auto",
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    overwrite: bool = False,
    forward_smoke: bool = False,
) -> dict[str, Any]:
    """Run one GPU-facing V4 stage for one registered model."""

    allowed = {
        "preflight",
        "representation-capture",
        "attention",
        "ablation",
        "patching",
    }
    if stage not in allowed:
        raise ValueError(f"Unknown model stage {stage!r}; choose from {allowed}")
    config = V4Config.from_json(config_path)
    answer_format = str(answer_format)
    if answer_format not in config.answer_formats:
        raise ValueError(f"Unregistered V4 answer format: {answer_format}")
    model_spec = resolve_model_spec(model_label)
    model_output = Path(output_dir) / model_spec.label / answer_format
    logger = EventLogger(model_output / "events.jsonl")
    with logger.timer("model_load", model=model_spec.label):
        model, tokenizer, adapter = load_registered_model(
            model_spec,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=config.model_torch_dtype,
            attention_backend=config.attention_prefix_backend,
        )
    selected = select_stimuli(
        stimuli_path,
        variants=variants,
        seeds=seeds,
        counts=counts,
    )
    representative = []
    for variant in config.design_variants:
        match = next(
            (
                row
                for row in selected
                if row["design_variant"] == variant
                and int(row["gold_count"]) == max(config.needle_counts)
            ),
            None,
        )
        if match is not None:
            representative.append(match)
    if not representative:
        representative = [selected[0]]
    with logger.timer("preflight", model=model_spec.label):
        report = preflight_report(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            representative_rows=representative,
            model_label=model_spec.label,
            config=config,
            answer_format=answer_format,
            forward_smoke=forward_smoke,
        )
        preflight_path = model_output / "preflight.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if stage == "preflight":
        return {"preflight": str(preflight_path)}

    if stage == "representation-capture":
        representation_rows = [
            row
            for row in selected
            if int(row["gold_count"]) == config.representation_count
        ]
        with logger.timer(
            "representation_capture",
            rows=len(representation_rows),
        ):
            index_path = capture_representation_shards(
                model,
                adapter,
                render_encodings(
                    representation_rows,
                    tokenizer=tokenizer,
                    model_label=model_spec.label,
                    config=config,
                    answer_format=answer_format,
                ),
                output_dir=model_output / "representation" / "capture",
                save_dtype=config.hidden_save_dtype,
                overwrite=overwrite,
            )
        return {
            "preflight": str(preflight_path),
            "capture_index": str(index_path),
        }

    if stage == "attention":
        with logger.timer("attention_capture", rows=len(selected)):
            attention_index = capture_attention_shards(
                model,
                adapter,
                render_encodings(
                    selected,
                    tokenizer=tokenizer,
                    model_label=model_spec.label,
                    config=config,
                    answer_format=answer_format,
                ),
                output_dir=model_output / "attention" / "capture",
                save_raw_rows=config.save_raw_attention_rows,
                save_dtype=config.attention_save_dtype,
                overwrite=overwrite,
            )
        with logger.timer("attention_analyze"):
            detail = load_attention_shards(attention_index)
            outputs = analyze_attention_table(
                detail,
                output_dir=model_output / "attention",
                metric=config.attention_primary_metric,
            )
        return {
            "preflight": str(preflight_path),
            "capture_index": str(attention_index),
            **{
                key: (value if isinstance(value, dict) else str(value))
                for key, value in outputs.items()
                if key != "rankings"
            },
        }

    confirmation_rows = [row for row in selected if row["split"] == "confirmation"]
    confirmation_encodings = list(
        render_encodings(
            confirmation_rows,
            tokenizer=tokenizer,
            model_label=model_spec.label,
            config=config,
            answer_format=answer_format,
        )
    )
    if stage == "ablation":
        observed_variants = sorted(
            {encoding.design_variant for encoding in confirmation_encodings}
        )
        rankings = {
            variant: load_head_ranking(
                model_output
                / "attention"
                / f"head_ranking_{variant.replace('.', '_')}.json"
            )
            for variant in observed_variants
        }
        capture_root = model_output / "causal" / "head_ablation_capture"
        index_rows: list[dict[str, Any]] = []
        with logger.timer("head_ablation", rows=len(confirmation_encodings)):
            for encoding in confirmation_encodings:
                relative = (
                    Path("shards")
                    / encoding.design_variant
                    / f"{encoding.stimulus_id}.csv.gz"
                )
                shard = capture_root / relative
                if shard.exists() and not overwrite:
                    frame = pd.read_csv(shard, compression="gzip")
                    if frame.empty or set(frame["stimulus_id"]) != {
                        encoding.stimulus_id
                    }:
                        raise RuntimeError(f"Invalid head-ablation shard: {shard}")
                else:
                    frame = run_head_ablation_experiment(
                        model,
                        adapter,
                        [encoding],
                        rankings=rankings,
                        top_ns=config.ablation_top_ns,
                        random_replicates=config.ablation_random_replicates,
                        scope=config.ablation_scope,
                    )
                    _write_csv_gzip_atomic(frame, shard)
                index_rows.append(
                    {
                        "stimulus_id": encoding.stimulus_id,
                        "design_variant": encoding.design_variant,
                        "seed": int(encoding.seed),
                        "count": int(encoding.count),
                        "rows": len(frame),
                        "shard_path": relative.as_posix(),
                    }
                )
        index_path = capture_root / "head_ablation_capture_index.jsonl"
        _write_shard_index(index_rows, index_path)
        detail = _load_csv_gzip_shards(index_rows, root=capture_root)
        causal_output = model_output / "causal"
        causal_output.mkdir(parents=True, exist_ok=True)
        detail_path = causal_output / "head_ablation_detail.csv"
        summary_path = causal_output / "head_ablation_summary.csv"
        comparison_path = causal_output / "head_ablation_broad_vs_random.csv"
        detail.to_csv(detail_path, index=False)
        summarize_head_ablation(detail).to_csv(summary_path, index=False)
        compare_head_ablation_to_random(detail).to_csv(comparison_path, index=False)
        return {
            "preflight": str(preflight_path),
            "capture_index": str(index_path),
            "detail": str(detail_path),
            "summary": str(summary_path),
            "broad_vs_random": str(comparison_path),
        }

    selected_layers = resolve_fractional_layers(
        adapter.num_layers, config.causal_layer_fractions
    )
    patch_capture_root = model_output / "causal" / "hidden_patching_capture"
    patch_index_rows: list[dict[str, Any]] = []
    grouped_encodings: dict[tuple[str, int], list[PromptEncoding]] = {}
    for encoding in confirmation_encodings:
        grouped_encodings.setdefault(
            (encoding.design_variant, int(encoding.seed)), []
        ).append(encoding)
    with logger.timer(
        "hidden_patching",
        rows=len(confirmation_encodings),
        layers=list(selected_layers),
    ):
        for (variant, seed), family_encodings in sorted(grouped_encodings.items()):
            relative = (
                Path("shards")
                / variant
                / f"{variant.replace('.', '_')}_seed{seed}.csv.gz"
            )
            shard = patch_capture_root / relative
            if shard.exists() and not overwrite:
                frame = pd.read_csv(shard, compression="gzip")
                if (
                    frame.empty
                    or set(frame["design_variant"]) != {variant}
                    or set(frame["seed"].astype(int)) != {seed}
                ):
                    raise RuntimeError(f"Invalid hidden-patching shard: {shard}")
            else:
                frame = run_hidden_patching_experiment(
                    model,
                    adapter,
                    family_encodings,
                    layers=selected_layers,
                    count_pairs=config.patch_count_pairs,
                    sites=config.patch_sites,
                )
                _write_csv_gzip_atomic(frame, shard)
            patch_index_rows.append(
                {
                    "design_variant": variant,
                    "seed": int(seed),
                    "rows": len(frame),
                    "shard_path": relative.as_posix(),
                }
            )
    patch_index_path = patch_capture_root / "hidden_patching_capture_index.jsonl"
    _write_shard_index(patch_index_rows, patch_index_path)
    detail = _load_csv_gzip_shards(patch_index_rows, root=patch_capture_root)
    causal_output = model_output / "causal"
    causal_output.mkdir(parents=True, exist_ok=True)
    detail_path = causal_output / "hidden_patching_detail.csv"
    summary_path = causal_output / "hidden_patching_summary.csv"
    detail.to_csv(detail_path, index=False)
    summarize_hidden_patching(detail).to_csv(summary_path, index=False)
    return {
        "preflight": str(preflight_path),
        "capture_index": str(patch_index_path),
        "detail": str(detail_path),
        "summary": str(summary_path),
    }


def run_representation_analysis(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    model_label: str,
    answer_format: str,
) -> dict[str, str]:
    config = V4Config.from_json(config_path)
    model_spec = resolve_model_spec(model_label)
    answer_format = str(answer_format)
    if answer_format not in config.answer_formats:
        raise ValueError(f"Unregistered V4 answer format: {answer_format}")
    model_output = Path(output_dir) / model_spec.label / answer_format
    outputs = analyze_representation_captures(
        capture_index_path=(
            model_output / "representation" / "capture" / "capture_index.jsonl"
        ),
        output_dir=model_output / "representation" / "analysis",
        config=config,
    )
    return {key: str(value) for key, value in outputs.items()}
