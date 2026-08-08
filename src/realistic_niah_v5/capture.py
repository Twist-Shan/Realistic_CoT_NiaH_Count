from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.modeling import (
    DecoderAdapter,
    capture_post_block_states,
    capture_span_states,
    position_attention_outputs,
)
from realistic_niah_v4.prompts import TokenSpan
from realistic_niah_v3.city_list_termination import (
    find_first_terminated_gold_city_list,
)

from .encoding import build_native_trace_encoding
from .parsing import (
    TraceTokenSite,
    align_trace_sites,
    gold_records,
    infer_model_family,
    output_token_ids,
    parse_trace_record,
    prompt_token_ids,
    raw_output_text,
    trace_char_sites,
)
from .spec import V5Config


CAPTURE_SCHEMA_VERSION = "realistic_niah_v5_trace_capture_v1"
ATTENTION_SCHEMA_VERSION = "realistic_niah_v5_mechanism_attention_v2"


@dataclass(frozen=True)
class _ModelInput:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _selected_sites(
    token_sites: Sequence[TraceTokenSite], config: V5Config
) -> list[TraceTokenSite]:
    registered = set(config.registered_sites)
    result = [
        site
        for site in token_sites
        if site.char_site.site_kind in registered and site.alignment_eligible
    ]
    return sorted(
        result,
        key=lambda site: (
            site.char_site.char_end,
            site.char_site.site_kind,
            site.char_site.site_id,
        ),
    )


def _full_input(row: Mapping[str, Any]) -> _ModelInput:
    prompt = prompt_token_ids(row)
    output = output_token_ids(row)
    mask = row.get("attention_mask")
    prompt_mask = (
        tuple(int(value) for value in mask)
        if mask is not None
        else (1,) * len(prompt)
    )
    if len(prompt_mask) != len(prompt):
        raise ValueError("Prompt attention-mask length mismatch")
    return _ModelInput(
        input_ids=prompt + output,
        attention_mask=prompt_mask + (1,) * len(output),
    )


def _literal_item_spans(
    token_sites: Sequence[TraceTokenSite], prompt_count: int
) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for site in token_sites:
        if site.char_site.site_kind != "item_end":
            continue
        start = site.literal_token_start
        end = site.literal_token_end
        if start is None or end is None or end <= start:
            continue
        occurrence = int(site.char_site.occurrence or 0)
        spans.append(
            TokenSpan(
                slot_index=occurrence,
                start=prompt_count + int(start),
                end=prompt_count + int(end),
                active=True,
                kind="native_trace_item",
                canonical_length=end - start,
                model_token_length=end - start,
            )
        )
    return spans


@torch.inference_mode()
def capture_trace_record(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    config: V5Config,
    output_dir: str | Path,
    layers: Iterable[int] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config.validate()
    output = Path(output_dir)
    manifest_path = output / "capture_manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    family = infer_model_family(row)
    raw = raw_output_text(row)
    parser = find_first_terminated_gold_city_list(
        raw,
        model_family=family,
        gold_records=gold_records(row),
    )
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw, parser),
    )
    sites = _selected_sites(token_sites, config)
    if not sites:
        raise ValueError("No aligned registered V5 trace sites")
    started = time.perf_counter()
    full = _full_input(row)
    prompt_count = len(prompt_token_ids(row))
    literal_indices = [
        index
        for index, site in enumerate(sites)
        if site.alignment_strategy == "literal_baseline_token_prefix"
    ]
    state_by_site: dict[int, torch.Tensor] = {}
    layer_indices: torch.Tensor | None = None
    if literal_indices:
        positions = [
            prompt_count + int(sites[index].prefix_token_count or 0) - 1
            for index in literal_indices
        ]
        _logits, captured = capture_post_block_states(
            model,
            adapter,
            full,
            positions,
            layers=layers,
        )
        ordered_layers = tuple(sorted(captured))
        layer_indices = torch.tensor(ordered_layers, dtype=torch.long)
        stacked = torch.stack([captured[layer] for layer in ordered_layers], dim=0)
        for position_axis, site_index in enumerate(literal_indices):
            state_by_site[site_index] = stacked[:, position_axis, :]
    for site_index, site in enumerate(sites):
        if site_index in state_by_site:
            continue
        encoding = build_native_trace_encoding(
            row,
            tokenizer,
            site_id=site.char_site.site_id,
            candidate_counts=config.candidate_counts,
        )
        _logits, captured = capture_post_block_states(
            model,
            adapter,
            encoding,
            [encoding.query_position],
            layers=layers,
        )
        ordered_layers = tuple(sorted(captured))
        current_layers = torch.tensor(ordered_layers, dtype=torch.long)
        if layer_indices is None:
            layer_indices = current_layers
        elif not torch.equal(layer_indices, current_layers):
            raise RuntimeError("V5 site captures disagree on decoder layers")
        state_by_site[site_index] = torch.stack(
            [captured[layer][0] for layer in ordered_layers], dim=0
        )
    if layer_indices is None or len(state_by_site) != len(sites):
        raise RuntimeError("V5 did not capture every registered trace site")
    states = torch.stack(
        [state_by_site[index] for index in range(len(sites))], dim=0
    )
    numpy_dtype = np.float16 if config.hidden_save_dtype == "float16" else np.float32
    arrays: dict[str, np.ndarray] = {
        "layer_indices": layer_indices.numpy(),
        "site_states": states.numpy().astype(numpy_dtype, copy=False),
    }
    item_spans = _literal_item_spans(token_sites, prompt_count)
    span_site_ids: list[str] = []
    if item_spans:
        pooled = capture_span_states(
            model,
            adapter,
            full,
            item_spans,
            layers=layer_indices.tolist(),
        )
        arrays["item_span_mean"] = (
            pooled["span_mean"].permute(1, 0, 2).numpy().astype(numpy_dtype, copy=False)
        )
        arrays["item_span_end_literal"] = (
            pooled["span_end"].permute(1, 0, 2).numpy().astype(numpy_dtype, copy=False)
        )
        span_site_ids = [f"item_span:{span.slot_index}" for span in item_spans]
    _save_npz(output / "states.npz", **arrays)
    parsed = parse_trace_record(row)
    manifest = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "stimulus_id": row.get("stimulus_id"),
        "request_id": row.get("request_id"),
        "model_label": row.get("model_label"),
        "model_family": family,
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold_records(row)),
        "parsed_count": parsed["parsed_count"],
        "exact_count": parsed["exact_count"],
        "parser_implementation": parsed["parser_implementation"],
        "parser_file_sha256": parsed["parser_file_sha256"],
        "parser": parsed["parser"],
        "prompt_token_count": prompt_count,
        "output_token_count": len(output_token_ids(row)),
        "layers": layer_indices.tolist(),
        "site_rows": [site.to_dict() for site in sites],
        "span_site_ids": span_site_ids,
        "states_file": "states.npz",
        "site_states_shape": list(states.shape),
        "save_dtype": config.hidden_save_dtype,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _mass(row: torch.Tensor, *, key_start: int, spans: Sequence[Any]) -> float:
    total = 0.0
    for span in spans:
        left = max(int(span.start), int(key_start)) - int(key_start)
        right = min(int(span.end), int(key_start) + int(row.shape[-1])) - int(key_start)
        if right > left:
            total += float(row[left:right].sum().item())
    return total


@torch.inference_mode()
def capture_trace_attention_metrics(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    config: V5Config,
    mechanisms: Sequence[str] = (
        "targeted_retrieval",
        "progress_transition",
    ),
) -> pd.DataFrame:
    """Capture native-thinking head scores aligned to semantic targets.

    ``targeted_retrieval`` queries the parser marker boundary for item k and
    scores attention to the matching prompt city-record span.  The
    ``progress_transition`` query is the accepted item-k endpoint and scores
    attention to the prompt span of the next city that the trace actually
    enumerates.  The latter is defined only for nonterminal k.
    """

    allowed = {"targeted_retrieval", "progress_transition"}
    unknown = sorted(set(mechanisms) - allowed)
    if unknown:
        raise ValueError(f"Unknown V5 attention mechanisms: {unknown}")
    parsed = parse_trace_record(row)
    if not bool(parsed["parser"]["detected"]):
        return pd.DataFrame()
    raw = raw_output_text(row)
    parser = find_first_terminated_gold_city_list(
        raw,
        model_family=parsed["model_family"],
        gold_records=gold_records(row),
    )
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw, parser),
    )
    by_id = {site.char_site.site_id: site for site in token_sites}
    trace_cities = [str(value) for value in parser.item_gold_cities]
    queries: list[tuple[str, Any, str]] = []
    if "targeted_retrieval" in mechanisms:
        for occurrence, city in enumerate(trace_cities, start=1):
            site = by_id.get(f"marker_end:{occurrence}")
            if site is not None and site.alignment_eligible:
                queries.append(("targeted_retrieval", site, city))
    if "progress_transition" in mechanisms:
        for occurrence, target_city in enumerate(trace_cities[1:], start=1):
            site = by_id.get(f"item_end:{occurrence}")
            if site is not None and site.alignment_eligible:
                queries.append(("progress_transition", site, target_city))
    rows: list[dict[str, Any]] = []
    for mechanism, site, target_city in queries:
        encoding = build_native_trace_encoding(
            row,
            tokenizer,
            site_id=site.char_site.site_id,
            candidate_counts=config.candidate_counts,
        )
        if not encoding.prompt_record_spans:
            raise ValueError(
                "Mechanism-specific attention requires prompt_record_spans; "
                "regenerate V5 rows from frozen V4.4 stimuli"
            )
        prompt_by_city = {
            span.city.casefold(): span for span in encoding.prompt_record_spans
        }
        target_span = prompt_by_city.get(target_city.casefold())
        if target_span is None:
            raise RuntimeError(f"No prompt record span for target city {target_city}")
        attention_rows, key_starts, _logits = position_attention_outputs(
            model,
            adapter,
            encoding,
            encoding.query_position,
        )
        occurrence = int(site.char_site.occurrence or 0)
        for layer, (attention, key_start) in enumerate(
            zip(attention_rows, key_starts)
        ):
            for head in range(attention.shape[0]):
                head_row = attention[head]
                record_masses = [
                    _mass(head_row, key_start=key_start, spans=[span])
                    for span in encoding.prompt_record_spans
                ]
                prompt_records_mass = float(sum(record_masses))
                target_mass = _mass(
                    head_row, key_start=key_start, spans=[target_span]
                )
                trace_mass = _mass(
                    head_row, key_start=key_start, spans=encoding.needle_spans
                )
                prompt_right = min(
                    encoding.prompt_token_count - key_start,
                    int(head_row.shape[-1]),
                )
                prompt_mass = (
                    float(head_row[: max(0, prompt_right)].sum().item())
                    if prompt_right > 0
                    else 0.0
                )
                rows.append(
                    {
                        "schema_version": ATTENTION_SCHEMA_VERSION,
                        "request_id": encoding.request_id,
                        "stimulus_id": encoding.stimulus_id,
                        "model_label": encoding.model_label,
                        "seed": encoding.seed,
                        "split": encoding.split,
                        "gold_count": encoding.count,
                        "trace_one_to_one": bool(
                            parsed["parser"].get("trace_one_to_one")
                        ),
                        "trace_category": parsed["parser"].get("trace_category"),
                        "final_exact_count": bool(parsed.get("exact_count")),
                        "mechanism": mechanism,
                        "site_id": site.char_site.site_id,
                        "site_kind": site.char_site.site_kind,
                        "occurrence": occurrence,
                        "query_city": site.char_site.city,
                        "target_city": target_city,
                        "target_slot_index": int(target_span.slot_index),
                        "layer": layer,
                        "head": head,
                        "key_start": key_start,
                        "target_prompt_record_mass": target_mass,
                        "prompt_records_total_mass": prompt_records_mass,
                        "target_within_records_fraction": (
                            target_mass / prompt_records_mass
                            if prompt_records_mass > 0
                            else 0.0
                        ),
                        "target_record_top1": bool(
                            prompt_records_mass > 0
                            and record_masses
                            and int(np.argmax(record_masses))
                            == list(encoding.prompt_record_spans).index(target_span)
                        ),
                        "trace_item_mass": trace_mass,
                        "full_prompt_mass": prompt_mass,
                        "other_generated_mass": max(
                            0.0, 1.0 - trace_mass - prompt_mass
                        ),
                        "row_sum": float(head_row.sum().item()),
                    }
                )
    return pd.DataFrame(rows)


def capture_trace_shards(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    records: Iterable[Mapping[str, Any]],
    *,
    config: V5Config,
    output_dir: str | Path,
    layers: Iterable[int] | None = None,
    overwrite: bool = False,
) -> Path:
    output = Path(output_dir)
    index_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        request_id = str(row.get("request_id", row.get("stimulus_id", row_index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        relative = Path("shards") / safe_id
        manifest = capture_trace_record(
            model,
            adapter,
            tokenizer,
            row,
            config=config,
            output_dir=output / relative,
            layers=layers,
            overwrite=overwrite,
        )
        index_rows.append(
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "row_index": row_index,
                "request_id": request_id,
                "stimulus_id": manifest["stimulus_id"],
                "model_label": manifest["model_label"],
                "seed": manifest["seed"],
                "split": manifest["split"],
                "gold_count": manifest["gold_count"],
                "parsed_count": manifest["parsed_count"],
                "exact_count": manifest["exact_count"],
                "trace_one_to_one": manifest["parser"]["trace_one_to_one"],
                "trace_category": manifest["parser"]["trace_category"],
                "marker_kind": manifest["parser"]["marker_kind"],
                "manifest_path": (relative / "capture_manifest.json").as_posix(),
                "states_path": (relative / "states.npz").as_posix(),
            }
        )
        print(
            f"[v5 capture] {row_index + 1} {request_id} "
            f"sites={len(manifest['site_rows'])} elapsed={manifest['elapsed_seconds']:.2f}s",
            flush=True,
        )
    if not index_rows:
        raise ValueError("No V5 records were supplied for capture")
    index = output / "capture_index.jsonl"
    _atomic_jsonl(index, index_rows)
    _atomic_json(
        output / "capture_manifest.json",
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "rows": len(index_rows),
            "parser_implementation": (
                "realistic_niah_v3.find_first_terminated_gold_city_list"
            ),
            "primary_trace_site": config.primary_trace_site,
            "registered_sites": list(config.registered_sites),
            "restartable_shards": True,
            "full_sequence_hidden_states_materialized": False,
        },
    )
    return index
