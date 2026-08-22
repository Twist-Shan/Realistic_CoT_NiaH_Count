"""Terminal item-token -> full-span state -> count-answer mediation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from realistic_niah_v4.modeling import DecoderAdapter, capture_post_block_states

from .count_stream import (
    COUNT_STREAM_SCHEMA_VERSION,
    NativeTraceEncoding,
    _full_state_patch_layers,
    _prefill_with_layerwise_state_replacements,
    _score_and_generate_prefill,
    _sha256_json,
    build_answer_source_registry,
    build_html_aligned_uninformative_trace_encoding,
)


REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS = (
    "clean",
    "clean_terminal_token_ablation",
    "uninformative",
    "terminal_token_restore",
    "terminal_marker_token_restore",
    "terminal_nonmarker_token_restore",
    "uninformative_terminal_state_restore",
    "terminal_token_restore_state_occluded",
)

REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS = (
    "clean",
    "terminal_token_ablation",
    "terminal_marker_token_ablation",
    "terminal_nonmarker_token_ablation",
    "ablated_terminal_state_restore",
    "clean_terminal_state_occluded",
)


def _replace_positions(
    receiver: NativeTraceEncoding,
    donor: NativeTraceEncoding,
    positions: Sequence[int],
) -> NativeTraceEncoding:
    active = list(receiver.input_ids)
    donor_ids = tuple(int(value) for value in donor.input_ids)
    normalized = tuple(sorted({int(value) for value in positions}))
    if not normalized or normalized[0] < 0 or normalized[-1] >= len(active):
        raise ValueError("Terminal token-state replacement positions are invalid")
    for position in normalized:
        active[position] = donor_ids[position]
    return replace(receiver, input_ids=tuple(active))


def _restore_or_identity(
    receiver: NativeTraceEncoding,
    donor: NativeTraceEncoding,
    positions: Sequence[int],
) -> NativeTraceEncoding:
    """Restore a diagnostic partition, using identity for an empty partition."""

    normalized = tuple(sorted({int(value) for value in positions}))
    if not normalized:
        return receiver
    return _replace_positions(receiver, donor, normalized)


@torch.inference_mode()
def run_terminal_token_state_bridge_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Test whether terminal item tokens write the count-relevant terminal state."""

    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    uninformative, control_audit = build_html_aligned_uninformative_trace_encoding(
        clean,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    terminal_positions = tuple(registry.positions("terminal_trace_item"))
    terminal_set = set(terminal_positions)
    marker_positions = tuple(
        position
        for position in registry.positions("trace_markers")
        if position in terminal_set
    )
    nonmarker_positions = tuple(
        position
        for position in registry.positions("trace_nonmarkers")
        if position in terminal_set
    )
    if not terminal_positions:
        raise ValueError("Terminal token-state bridge has an empty terminal item")
    if set(marker_positions) & set(nonmarker_positions):
        raise RuntimeError("Terminal marker/nonmarker partitions overlap")
    if set(marker_positions) | set(nonmarker_positions) != terminal_set:
        raise RuntimeError("Terminal marker/nonmarker partitions do not cover the item")

    terminal_restore = _replace_positions(
        uninformative, clean, terminal_positions
    )
    marker_restore = _restore_or_identity(uninformative, clean, marker_positions)
    nonmarker_restore = _restore_or_identity(
        uninformative, clean, nonmarker_positions
    )
    clean_terminal_ablation = _replace_positions(
        clean, uninformative, terminal_positions
    )
    patch_layers = _full_state_patch_layers(
        source_layer=int(layer),
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    _unused, uninformative_capture = capture_post_block_states(
        model,
        adapter,
        uninformative,
        terminal_positions,
        layers=patch_layers,
    )
    _unused, restored_capture = capture_post_block_states(
        model,
        adapter,
        terminal_restore,
        terminal_positions,
        layers=patch_layers,
    )
    uninformative_states = {
        patch_layer: uninformative_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    restored_states = {
        patch_layer: restored_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    state_delta_norms = {
        str(patch_layer): float(
            torch.linalg.vector_norm(
                restored_states[patch_layer] - uninformative_states[patch_layer]
            )
        )
        for patch_layer in patch_layers
    }
    conditions: dict[
        str,
        tuple[NativeTraceEncoding, Mapping[int, torch.Tensor] | None],
    ] = {
        "clean": (clean, None),
        "clean_terminal_token_ablation": (clean_terminal_ablation, None),
        "uninformative": (uninformative, None),
        "terminal_token_restore": (terminal_restore, None),
        "terminal_marker_token_restore": (marker_restore, None),
        "terminal_nonmarker_token_restore": (nonmarker_restore, None),
        "uninformative_terminal_state_restore": (
            uninformative,
            restored_states,
        ),
        "terminal_token_restore_state_occluded": (
            terminal_restore,
            uninformative_states,
        ),
    }
    rows: list[dict[str, Any]] = []
    for condition in REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS:
        encoding, replacements = conditions[condition]
        prefill, _captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                encoding,
                positions=terminal_positions,
                replacements=replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(registry.query_position,),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=int(max_new_tokens),
        )
        rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "terminal_token_to_fullspan_state_bridge",
                "condition": condition,
                "status": "ok",
                "request_id": clean.request_id,
                "model_label": clean.model_label,
                "seed": clean.seed,
                "dataset_split": clean.split,
                "gold_count": clean.count,
                "answer_site_id": answer_site_id,
                "target_occurrence": len(registry.trace_items),
                "target_is_terminal": True,
                "patch_geometry": "terminal_full_item_span_same_position",
                "patch_layer_mode": "cumulative_clamp",
                "layer": int(layer),
                "patch_layers": list(patch_layers),
                "patch_token_count": len(terminal_positions),
                "terminal_positions_sha256": _sha256_json(terminal_positions),
                "terminal_marker_token_count": len(marker_positions),
                "terminal_nonmarker_token_count": len(nonmarker_positions),
                "terminal_marker_partition_empty": not bool(marker_positions),
                "terminal_nonmarker_partition_empty": not bool(nonmarker_positions),
                "terminal_marker_positions_sha256": _sha256_json(marker_positions),
                "terminal_nonmarker_positions_sha256": _sha256_json(
                    nonmarker_positions
                ),
                "terminal_token_state_delta_norm_by_layer": state_delta_norms,
                "patch_hook_applications": {
                    str(key): value for key, value in sorted(applications.items())
                },
                "patch_realized_fro_norm_by_layer": {
                    str(key): value for key, value in sorted(realized_norms.items())
                },
                "input_token_restoration_uses_outcome": False,
                "causal_claim_scope": (
                    "terminal_item_token_content_to_fullspan_hidden_state_to_answer"
                ),
                "registry_sha256": registry.to_dict()["registry_sha256"],
                **control_audit,
                **outcomes,
            }
        )
    return rows


@torch.inference_mode()
def run_local_terminal_token_state_bridge_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    layer: int,
    random_seed: int,
    answer_site_id: str = "answer_query_v3",
    run_greedy: bool = True,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Mediate a local terminal-item ablation through its full-span state."""

    clean, registry = build_answer_source_registry(
        row, tokenizer, answer_site_id=answer_site_id
    )
    background, control_audit = build_html_aligned_uninformative_trace_encoding(
        clean,
        registry,
        tokenizer,
        random_seed=int(random_seed),
    )
    terminal_positions = tuple(registry.positions("terminal_trace_item"))
    terminal_set = set(terminal_positions)
    marker_positions = tuple(
        position
        for position in registry.positions("trace_markers")
        if position in terminal_set
    )
    nonmarker_positions = tuple(
        position
        for position in registry.positions("trace_nonmarkers")
        if position in terminal_set
    )
    if not terminal_positions:
        raise ValueError("Local terminal token-state bridge has an empty item")
    if set(marker_positions) & set(nonmarker_positions):
        raise RuntimeError("Local terminal marker/nonmarker partitions overlap")
    if set(marker_positions) | set(nonmarker_positions) != terminal_set:
        raise RuntimeError(
            "Local terminal marker/nonmarker partitions do not cover the item"
        )
    terminal_ablation = _replace_positions(
        clean, background, terminal_positions
    )
    marker_ablation = _restore_or_identity(clean, background, marker_positions)
    nonmarker_ablation = _restore_or_identity(
        clean, background, nonmarker_positions
    )
    patch_layers = _full_state_patch_layers(
        source_layer=int(layer),
        num_layers=int(adapter.num_layers),
        layer_mode="cumulative_clamp",
    )
    _unused, clean_capture = capture_post_block_states(
        model,
        adapter,
        clean,
        terminal_positions,
        layers=patch_layers,
    )
    _unused, ablated_capture = capture_post_block_states(
        model,
        adapter,
        terminal_ablation,
        terminal_positions,
        layers=patch_layers,
    )
    clean_states = {
        patch_layer: clean_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    ablated_states = {
        patch_layer: ablated_capture[patch_layer].clone()
        for patch_layer in patch_layers
    }
    state_delta_norms = {
        str(patch_layer): float(
            torch.linalg.vector_norm(
                clean_states[patch_layer] - ablated_states[patch_layer]
            )
        )
        for patch_layer in patch_layers
    }
    conditions: dict[
        str,
        tuple[NativeTraceEncoding, Mapping[int, torch.Tensor] | None],
    ] = {
        "clean": (clean, None),
        "terminal_token_ablation": (terminal_ablation, None),
        "terminal_marker_token_ablation": (marker_ablation, None),
        "terminal_nonmarker_token_ablation": (nonmarker_ablation, None),
        "ablated_terminal_state_restore": (terminal_ablation, clean_states),
        "clean_terminal_state_occluded": (clean, ablated_states),
    }
    rows: list[dict[str, Any]] = []
    for condition in REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS:
        encoding, replacements = conditions[condition]
        prefill, _captures, applications, realized_norms = (
            _prefill_with_layerwise_state_replacements(
                model,
                adapter,
                encoding,
                positions=terminal_positions,
                replacements=replacements,
                readout_layers=(int(adapter.num_layers) - 1,),
                readout_positions=(registry.query_position,),
            )
        )
        outcomes = _score_and_generate_prefill(
            model,
            tokenizer,
            encoding,
            prefill,
            run_greedy=run_greedy,
            max_new_tokens=int(max_new_tokens),
        )
        rows.append(
            {
                "schema_version": COUNT_STREAM_SCHEMA_VERSION,
                "experiment_id": "local_terminal_token_to_fullspan_state_bridge",
                "condition": condition,
                "status": "ok",
                "request_id": clean.request_id,
                "model_label": clean.model_label,
                "seed": clean.seed,
                "dataset_split": clean.split,
                "gold_count": clean.count,
                "answer_site_id": answer_site_id,
                "target_occurrence": len(registry.trace_items),
                "target_is_terminal": True,
                "earlier_trace_tokens_remain_clean": True,
                "patch_geometry": "terminal_full_item_span_same_position",
                "patch_layer_mode": "cumulative_clamp",
                "layer": int(layer),
                "patch_layers": list(patch_layers),
                "patch_token_count": len(terminal_positions),
                "terminal_positions_sha256": _sha256_json(terminal_positions),
                "terminal_marker_token_count": len(marker_positions),
                "terminal_nonmarker_token_count": len(nonmarker_positions),
                "terminal_marker_partition_empty": not bool(marker_positions),
                "terminal_nonmarker_partition_empty": not bool(nonmarker_positions),
                "terminal_token_state_delta_norm_by_layer": state_delta_norms,
                "patch_hook_applications": {
                    str(key): value for key, value in sorted(applications.items())
                },
                "patch_realized_fro_norm_by_layer": {
                    str(key): value for key, value in sorted(realized_norms.items())
                },
                "terminal_token_ablation_uses_outcome": False,
                "causal_claim_scope": (
                    "local_terminal_item_token_content_to_fullspan_state_to_answer"
                ),
                "registry_sha256": registry.to_dict()["registry_sha256"],
                **control_audit,
                **outcomes,
            }
        )
    return rows
