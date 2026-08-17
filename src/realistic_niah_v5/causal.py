from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah.parsing import parse_total
from realistic_niah_v4.counter_channel_interventions import removal_transform
from realistic_niah_v4.counter_channel_interventions import (
    run_counter_subspace_conditions,
)
from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _tensor_from_output,
    capture_post_block_states,
    generate_answer_completion,
    generate_with_residual_interventions,
    generate_with_residual_transforms,
)
from .encoding import NativeTraceEncoding, build_native_trace_encoding
from .parsing import (
    align_trace_sites,
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    output_token_ids,
    prompt_token_ids,
    parse_trace_record,
    raw_output_text,
    trace_char_sites,
)
from .spec import V5Config


CAUSAL_SCHEMA_VERSION = "realistic_niah_v5_causal_v2"


@dataclass(frozen=True)
class CausalExperiment:
    experiment_id: str
    report_role: str
    intervention: str
    primary_endpoint: str
    required_controls: tuple[str, ...]
    implementation: str


V5_CAUSAL_LEDGER: tuple[CausalExperiment, ...] = (
    CausalExperiment(
        "earlier_span_attention",
        "Step 1: evidence for accumulation across earlier trace items",
        "restrict item-end query context to prior accepted-item spans",
        "downstream residual displacement and attention mass",
        ("clean", "length/depth-matched generated-text context"),
        "native query-context mask runner",
    ),
    CausalExperiment(
        "trace_token_corruption",
        "Step 1: token-level necessity of enumerated trace items",
        "replace trace-item tokens with equal-length non-trace tokens",
        "count accuracy, signed error, answer-query displacement",
        ("clean", "equal-token-count non-trace corruption"),
        "corrupt_trace_tokens",
    ),
    CausalExperiment(
        "trace_subspace_ablation",
        "Step 1/3: necessity of the discovery-frozen running-index subspace",
        "remove count-subspace component at trace endpoints",
        "count accuracy and answer-query count-axis coefficient",
        ("clean", "equal-norm orthogonal residual removal"),
        "run_subspace_ablation",
    ),
    CausalExperiment(
        "targeted_retrieval_head_ablation",
        "Step 2a: necessity of k-to-record targeted retrieval heads",
        "zero discovery-ranked head outputs at parser marker-end queries",
        "matching city continuation likelihood and exact next-token accuracy",
        ("clean", "layer-matched random heads", "K dose response"),
        "run_mechanism_head_ablation_trials",
    ),
    CausalExperiment(
        "progress_transition_head_ablation",
        "Step 2b: necessity of progress-transition heads",
        "zero discovery-ranked head outputs at accepted item-end queries",
        "next-item/stop continuation likelihood and exact next-token accuracy",
        ("clean", "layer-matched random heads", "K dose response"),
        "run_mechanism_head_ablation_trials",
    ),
    CausalExperiment(
        "answer_query_patch",
        "Step 3: sufficiency of answer-query count state",
        "patch matched donor state into receiver answer-query residual",
        "transport toward donor count",
        ("self patch", "mismatched donor", "orthogonal norm-matched delta"),
        "run_residual_patch_trial",
    ),
    CausalExperiment(
        "trace_endpoint_patch",
        "Step 1 -> Step 3 causal transport",
        "patch discovery-frozen trace endpoint state/subspace",
        "answer-query representation and count transport",
        ("self patch", "orthogonal norm-matched delta"),
        "run_residual_patch_trial/run_counter_subspace_conditions",
    ),
    CausalExperiment(
        "qwen_natural_ov_write",
        "Step 2 optional delayed OV rewriting (Qwen)",
        "trace natural attention-weighted OV contribution",
        "count-axis write and candidate-score change",
        ("selected heads", "layer-matched controls", "read/write partition"),
        "capture_natural_head_writes",
    ),
    CausalExperiment(
        "gemma_distributed_residual_write",
        "Step 2 optional distributed residual write (Gemma)",
        "trace/patch the registered cross-layer residual contribution",
        "count-axis write and candidate-score change",
        ("equal-norm orthogonal delta", "site-set specificity"),
        "capture_natural_head_writes with cross-layer aggregation",
    ),
)


def causal_ledger_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **experiment.__dict__,
                "required_controls": ";".join(experiment.required_controls),
            }
            for experiment in V5_CAUSAL_LEDGER
        ]
    )


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def rank_mechanism_heads(
    attention: pd.DataFrame,
    *,
    mechanism: str,
    split: str = "discovery",
) -> pd.DataFrame:
    """Freeze heads by query-weighted semantic target mass on discovery."""

    needed = {
        "model_label",
        "split",
        "mechanism",
        "gold_count",
        "trace_one_to_one",
        "layer",
        "head",
        "target_needle_raw_mass",
        "target_needle_relative_mass",
        "target_needle_top1",
    }
    missing = sorted(needed - set(attention.columns))
    if missing:
        raise ValueError(f"Attention table is missing {missing}")
    if mechanism not in {"targeted_retrieval", "progress_transition"}:
        raise ValueError(f"Unknown head mechanism: {mechanism}")
    mask = attention["split"].astype(str).str.lower().eq(split.lower())
    mask &= attention["mechanism"].astype(str).eq(mechanism)
    one_to_one = attention["trace_one_to_one"]
    if one_to_one.dtype == bool:
        mask &= one_to_one
    else:
        mask &= one_to_one.astype(str).str.lower().isin({"true", "1"})
    selected = attention.loc[mask].copy()
    if selected.empty:
        raise ValueError(
            f"No attention rows matched {split}/{mechanism}"
        )
    grouped = (
        selected.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            discovery_target_raw_mass=("target_needle_raw_mass", "mean"),
            discovery_target_relative_mass=("target_needle_relative_mass", "mean"),
            discovery_target_top1=("target_needle_top1", "mean"),
            n_queries=("target_needle_raw_mass", "size"),
        )
        .sort_values(
            ["model_label", "discovery_target_raw_mass", "layer", "head"],
            ascending=[True, False, True, True],
        )
    )
    grouped["discovery_rank"] = (
        grouped.groupby("model_label").cumcount() + 1
    )
    grouped["mechanism"] = mechanism
    grouped["query_site_kind"] = (
        "marker_end" if mechanism == "targeted_retrieval" else "item_end"
    )
    grouped["selection_metric"] = "query_weighted_mean_target_needle_raw_mass"
    grouped["selection_split"] = split
    grouped["selection_cohort"] = "one_to_one"
    grouped["selection_counts"] = "1-10"
    return grouped.reset_index(drop=True)


def rank_retrieval_heads(
    attention: pd.DataFrame, *, split: str = "discovery"
) -> pd.DataFrame:
    """Compatibility name for the registered targeted-retrieval ranking."""

    return rank_mechanism_heads(
        attention, mechanism="targeted_retrieval", split=split
    )


def rank_answer_query_heads(
    attention: pd.DataFrame,
    *,
    mechanism: str = "answer_prompt_aggregation",
    split: str = "discovery",
) -> pd.DataFrame:
    """Freeze prompt- and trace-aggregation answer-query heads separately.

    Confirmation rows are retained only for later descriptive bank-mass
    evaluation; they never influence ordering or membership.
    """

    mechanisms = {
        "answer_prompt_aggregation": (
            "target_needle_raw_mass",
            "target_needle_relative_mass",
            "prompt_broad_score",
            "prompt_broad_coverage",
            "exact_prompt_needle_span_broad_score",
        ),
        "answer_trace_aggregation": (
            "trace_item_raw_mass",
            "trace_item_relative_mass",
            "trace_broad_score",
            "trace_broad_coverage",
            "registered_trace_item_span_broad_score",
        ),
    }
    if mechanism not in mechanisms:
        raise ValueError(f"Unsupported answer-query mechanism: {mechanism}")
    (
        selection_raw,
        selection_relative,
        selection_broad,
        selection_coverage,
        selection_name,
    ) = mechanisms[mechanism]
    needed = {
        "model_label",
        "split",
        "trace_one_to_one",
        "layer",
        "head",
        "target_needle_raw_mass",
        "target_needle_relative_mass",
        "trace_item_raw_mass",
        "trace_item_relative_mass",
        "prompt_broad_score",
        "prompt_broad_coverage",
        "trace_broad_score",
        "trace_broad_coverage",
    }
    missing = sorted(needed - set(attention.columns))
    if missing:
        raise ValueError(f"Answer-query attention table is missing {missing}")
    selected = attention.copy()
    mask = selected["split"].astype(str).str.lower().eq(split.lower())
    one_to_one = selected["trace_one_to_one"]
    if one_to_one.dtype == bool:
        mask &= one_to_one
    else:
        mask &= one_to_one.astype(str).str.lower().isin({"true", "1"})
    selected = selected.loc[mask]
    if selected.empty:
        raise ValueError(f"No answer-query attention rows matched {split}")
    grouped = (
        selected.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            discovery_target_raw_mass=("target_needle_raw_mass", "mean"),
            discovery_target_relative_mass=(
                "target_needle_relative_mass", "mean"
            ),
            discovery_trace_raw_mass=("trace_item_raw_mass", "mean"),
            discovery_trace_relative_mass=("trace_item_relative_mass", "mean"),
            discovery_prompt_broad_score=("prompt_broad_score", "mean"),
            discovery_prompt_broad_coverage=("prompt_broad_coverage", "mean"),
            discovery_trace_broad_score=("trace_broad_score", "mean"),
            discovery_trace_broad_coverage=("trace_broad_coverage", "mean"),
            n_queries=("target_needle_raw_mass", "size"),
        )
    )
    grouped["discovery_selection_raw_mass"] = grouped[
        "discovery_target_raw_mass"
        if selection_raw == "target_needle_raw_mass"
        else "discovery_trace_raw_mass"
    ]
    grouped["discovery_selection_relative_mass"] = grouped[
        "discovery_target_relative_mass"
        if selection_relative == "target_needle_relative_mass"
        else "discovery_trace_relative_mass"
    ]
    grouped["discovery_selection_broad_score"] = grouped[
        "discovery_prompt_broad_score"
        if selection_broad == "prompt_broad_score"
        else "discovery_trace_broad_score"
    ]
    grouped["discovery_selection_broad_coverage"] = grouped[
        "discovery_prompt_broad_coverage"
        if selection_coverage == "prompt_broad_coverage"
        else "discovery_trace_broad_coverage"
    ]
    grouped = grouped.sort_values(
        [
            "model_label",
            "discovery_selection_broad_score",
            "discovery_selection_raw_mass",
            "layer",
            "head",
        ],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    grouped["discovery_rank"] = grouped.groupby("model_label").cumcount() + 1
    grouped["mechanism"] = mechanism
    grouped["query_site_kind"] = "answer_query_v3"
    grouped["selection_metric"] = f"query_weighted_mean_{selection_name}"
    grouped["selection_split"] = split
    grouped["selection_cohort"] = "one_to_one"
    return grouped


def layer_matched_random_controls(
    head_ranking: pd.DataFrame,
    selected_heads: Sequence[tuple[int, int]],
    *,
    repeats: int,
    seed_text: str,
) -> list[list[tuple[int, int]]]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    available = {
        (int(row.layer), int(row.head))
        for row in head_ranking.itertuples(index=False)
    }
    selected = [(int(layer), int(head)) for layer, head in selected_heads]
    by_layer: dict[int, list[int]] = {}
    for layer, head in available - set(selected):
        by_layer.setdefault(layer, []).append(head)
    target_counts: dict[int, int] = {}
    for layer, _head in selected:
        target_counts[layer] = target_counts.get(layer, 0) + 1
    rng = np.random.default_rng(_stable_seed(seed_text))
    controls: list[list[tuple[int, int]]] = []
    for _repeat in range(repeats):
        bank: list[tuple[int, int]] = []
        for layer, count in sorted(target_counts.items()):
            candidates = np.asarray(sorted(by_layer.get(layer, [])), dtype=int)
            if len(candidates) < count:
                raise ValueError(
                    f"Not enough non-selected heads for layer-matched L{layer} control"
                )
            chosen = rng.choice(candidates, size=count, replace=False)
            bank.extend((layer, int(head)) for head in chosen)
        controls.append(sorted(bank))
    return controls


def _control_constructible_ranked_heads(
    head_ranking: pd.DataFrame,
    ordered_heads: Sequence[tuple[int, int]],
    *,
    bank_size: int,
) -> list[tuple[int, int]]:
    """Greedily preserve discovery rank while keeping exact controls feasible.

    A layer-matched control must draw distinct heads from the complement of the
    ranked bank.  Consequently no ranked bank may occupy more than half of the
    available heads in any layer.  This constraint is architectural (not a
    post-hoc outcome filter), so it belongs in the frozen discovery plan.
    """

    available_by_layer: dict[int, int] = {}
    for row in head_ranking.itertuples(index=False):
        layer = int(row.layer)
        available_by_layer[layer] = available_by_layer.get(layer, 0) + 1
    layer_counts: dict[int, int] = {}
    selected: list[tuple[int, int]] = []
    for raw_layer, raw_head in ordered_heads:
        layer, head = int(raw_layer), int(raw_head)
        capacity = available_by_layer.get(layer, 0) // 2
        if layer_counts.get(layer, 0) >= capacity:
            continue
        selected.append((layer, head))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if len(selected) == int(bank_size):
            return selected
    raise ValueError(
        f"Cannot construct ranked bank K={bank_size} under exact-control "
        "per-layer capacity"
    )


def _control_constructible_joint_ranked_heads(
    head_ranking: pd.DataFrame,
    prompt_ordered: Sequence[tuple[int, int]],
    trace_ordered: Sequence[tuple[int, int]],
    *,
    bank_size: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Select equal-weight prompt/trace banks whose union admits controls."""

    available_by_layer: dict[int, int] = {}
    for row in head_ranking.itertuples(index=False):
        layer = int(row.layer)
        available_by_layer[layer] = available_by_layer.get(layer, 0) + 1
    prompt: list[tuple[int, int]] = []
    trace: list[tuple[int, int]] = []
    union: set[tuple[int, int]] = set()
    layer_counts: dict[int, int] = {}

    def add_next(
        ordered: Sequence[tuple[int, int]],
        selected: list[tuple[int, int]],
    ) -> bool:
        selected_set = set(selected)
        for raw_layer, raw_head in ordered:
            head = (int(raw_layer), int(raw_head))
            if head in selected_set:
                continue
            layer = head[0]
            capacity = available_by_layer.get(layer, 0) // 2
            if head not in union and layer_counts.get(layer, 0) >= capacity:
                continue
            selected.append(head)
            if head not in union:
                union.add(head)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            return True
        return False

    # Alternating additions give the two independently ranked mechanisms equal
    # weight in the joint bank while allowing a shared head to count for both.
    while len(prompt) < int(bank_size) or len(trace) < int(bank_size):
        progressed = False
        if len(prompt) < int(bank_size):
            progressed = add_next(prompt_ordered, prompt) or progressed
        if len(trace) < int(bank_size):
            progressed = add_next(trace_ordered, trace) or progressed
        if not progressed:
            raise ValueError(
                f"Cannot construct joint ranked banks K={bank_size} under "
                "exact-control per-layer capacity"
            )
    return prompt, trace


def _bank_attention_mass(
    head_ranking: pd.DataFrame,
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Summarize the frozen discovery exact-span mass for one head bank."""

    lookup = {
        (int(row.layer), int(row.head)): (
            float(row.discovery_target_raw_mass),
            float(row.discovery_target_relative_mass),
        )
        for row in head_ranking.itertuples(index=False)
    }
    values = [lookup[(int(layer), int(head))] for layer, head in heads]
    raw = np.asarray([value[0] for value in values], dtype=float)
    relative = np.asarray([value[1] for value in values], dtype=float)
    finite_relative = relative[np.isfinite(relative)]
    return {
        "target_needle_raw_mass": float(raw.mean()),
        "target_needle_relative_mass": (
            float(finite_relative.mean()) if len(finite_relative) else float("nan")
        ),
        "relative_mass_defined_heads": int(len(finite_relative)),
        "attention_mass_split": "discovery",
        "attention_mass_aggregation": (
            "mean_across_head_query_weighted_exact_span_scores"
        ),
    }


def _bank_answer_selection_mass(
    head_ranking: pd.DataFrame,
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Summarize the mechanism-specific frozen answer-query ranking mass."""

    lookup = {
        (int(row.layer), int(row.head)): (
            float(row.discovery_selection_raw_mass),
            float(row.discovery_selection_relative_mass),
            float(row.discovery_selection_broad_score),
            float(row.discovery_selection_broad_coverage),
        )
        for row in head_ranking.itertuples(index=False)
    }
    values = [lookup[(int(layer), int(head))] for layer, head in heads]
    raw = np.asarray([value[0] for value in values], dtype=float)
    relative = np.asarray([value[1] for value in values], dtype=float)
    broad = np.asarray([value[2] for value in values], dtype=float)
    coverage = np.asarray([value[3] for value in values], dtype=float)
    finite_relative = relative[np.isfinite(relative)]
    return {
        "selected_aggregation_raw_mass": float(raw.mean()),
        "selected_aggregation_relative_mass": (
            float(finite_relative.mean()) if len(finite_relative) else float("nan")
        ),
        "selected_aggregation_relative_defined_heads": int(len(finite_relative)),
        "selected_aggregation_broad_score": float(np.nanmean(broad)),
        "selected_aggregation_broad_coverage": float(np.nanmean(coverage)),
        "selected_aggregation_metric": str(
            head_ranking["selection_metric"].iloc[0]
        ),
    }


def build_causal_plan(
    attention_csv: str | Path,
    output_dir: str | Path,
    *,
    config: V5Config,
) -> dict[str, Path]:
    config.validate()
    attention = pd.read_csv(attention_csv)
    ranking = pd.concat(
        [
            rank_mechanism_heads(attention, mechanism=mechanism)
            for mechanism in config.causal_head_mechanisms
        ],
        ignore_index=True,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    skipped_banks: list[dict[str, Any]] = []
    for (model_label, mechanism), model_frame in ranking.groupby(
        ["model_label", "mechanism"], sort=True
    ):
        ordered = [
            (int(row.layer), int(row.head))
            for row in model_frame.sort_values("discovery_rank").itertuples()
        ]
        for bank_size in config.causal_head_bank_sizes:
            if bank_size > len(ordered):
                continue
            chosen = ordered[:bank_size]
            # The registered ranked treatment is scientifically meaningful at
            # every available K even when a disjoint, exactly layer-matched
            # random bank is combinatorially impossible.  Do not silently
            # discard the treatment together with an unavailable control.
            plan_rows.append(
                {
                    "model_label": model_label,
                    "mechanism": mechanism,
                    "query_site_kind": str(
                        model_frame["query_site_kind"].iloc[0]
                    ),
                    "experiment_id": f"{mechanism}_head_ablation",
                    "condition": f"{mechanism}_ranked",
                    "bank_size": bank_size,
                    "repeat": 0,
                    "heads": json.dumps(chosen),
                    **_bank_attention_mass(model_frame, chosen),
                }
            )
            try:
                controls = layer_matched_random_controls(
                    model_frame,
                    chosen,
                    repeats=config.causal_random_controls,
                    seed_text=f"v5:{model_label}:{mechanism}:K{bank_size}",
                )
            except ValueError as error:
                skipped_banks.append(
                    {
                        "model_label": str(model_label),
                        "mechanism": str(mechanism),
                        "bank_size": int(bank_size),
                        "ranked_treatment_included": True,
                        "control_status": (
                            "not_constructible_disjoint_exact_layer_match"
                        ),
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    {
                        "model_label": model_label,
                        "mechanism": mechanism,
                        "query_site_kind": str(
                            model_frame["query_site_kind"].iloc[0]
                        ),
                        "experiment_id": f"{mechanism}_head_ablation",
                        "condition": "layer_matched_random",
                        "bank_size": bank_size,
                        "repeat": repeat,
                        "heads": json.dumps(control),
                        **_bank_attention_mass(model_frame, control),
                    }
                )
    paths = {
        "ranking": output / "discovery_head_ranking.csv",
        "plan": output / "causal_plan.csv",
        "ledger": output / "causal_ledger.csv",
        "audit": output / "causal_plan_audit.json",
    }
    ranking.to_csv(paths["ranking"], index=False)
    pd.DataFrame(
        plan_rows,
        columns=[
            "model_label",
            "mechanism",
            "query_site_kind",
            "experiment_id",
            "condition",
            "bank_size",
            "repeat",
            "heads",
            "target_needle_raw_mass",
            "target_needle_relative_mass",
            "relative_mass_defined_heads",
            "attention_mass_split",
            "attention_mass_aggregation",
        ],
    ).to_csv(paths["plan"], index=False)
    causal_ledger_frame().to_csv(paths["ledger"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "selection_split": "discovery",
                "selection_cohort": "one_to_one",
                "selection_counts": list(config.counts),
                "confirmation_used_for_selection": False,
                "selection_metric": (
                    "query-weighted mean raw attention mass to the semantic "
                    "target prompt-record span"
                ),
                "mechanisms": {
                    "targeted_retrieval": {
                        "query": "marker_end:k",
                        "target": "prompt record matching accepted city k",
                    },
                    "progress_transition": {
                        "query": "item_end:k",
                        "target": "prompt record matching accepted city k+1",
                    },
                },
                "primary_ablation_scope": "mechanism query position only",
                "registered_bank_sizes": list(config.causal_head_bank_sizes),
                "ranked_treatment_policy": (
                    "include every registered K with enough ranked heads"
                ),
                "random_control": (
                    "disjoint random heads with the same selected-head count "
                    "in every layer; audited as structurally unavailable "
                    "rather than replaced by overlapping controls"
                ),
                "skipped_banks": skipped_banks,
                "attention_source": str(Path(attention_csv).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def build_answer_query_causal_plan(
    attention_csv: str | Path,
    output_dir: str | Path,
    *,
    config: V5Config,
) -> dict[str, Path]:
    """Build discovery-frozen answer-query banks and exact controls."""

    config.validate()
    attention = pd.read_csv(attention_csv)
    answer_mechanisms = (
        "answer_prompt_aggregation",
        "answer_trace_aggregation",
    )
    ranking = pd.concat(
        [
            rank_answer_query_heads(
                attention, mechanism=mechanism, split="discovery"
            )
            for mechanism in answer_mechanisms
        ],
        ignore_index=True,
    )
    confirmation = attention.loc[
        attention["split"].astype(str).str.lower().eq("confirmation")
    ].copy()
    confirmation_ranking = (
        confirmation.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            confirmation_target_raw_mass=("target_needle_raw_mass", "mean"),
            confirmation_target_relative_mass=(
                "target_needle_relative_mass", "mean"
            ),
            confirmation_trace_raw_mass=("trace_item_raw_mass", "mean"),
            confirmation_trace_relative_mass=("trace_item_relative_mass", "mean"),
            confirmation_prompt_broad_score=("prompt_broad_score", "mean"),
            confirmation_prompt_broad_coverage=(
                "prompt_broad_coverage",
                "mean",
            ),
            confirmation_trace_broad_score=("trace_broad_score", "mean"),
            confirmation_trace_broad_coverage=(
                "trace_broad_coverage",
                "mean",
            ),
            confirmation_queries=("target_needle_raw_mass", "size"),
        )
        if not confirmation.empty
        else pd.DataFrame()
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    skipped_banks: list[dict[str, Any]] = []
    for (model_label, mechanism), model_frame in ranking.groupby(
        ["model_label", "mechanism"], sort=True
    ):
        ordered = [
            (int(row.layer), int(row.head))
            for row in model_frame.sort_values("discovery_rank").itertuples()
        ]
        confirmation_lookup: dict[
            tuple[int, int], tuple[float, float, float, float, float, float, float, float]
        ] = {}
        if not confirmation_ranking.empty:
            model_confirmation = confirmation_ranking.loc[
                confirmation_ranking["model_label"].astype(str).eq(
                    str(model_label)
                )
            ]
            confirmation_lookup = {
                (int(row.layer), int(row.head)): (
                    float(row.confirmation_target_raw_mass),
                    float(row.confirmation_target_relative_mass),
                    float(row.confirmation_trace_raw_mass),
                    float(row.confirmation_trace_relative_mass),
                    float(row.confirmation_prompt_broad_score),
                    float(row.confirmation_prompt_broad_coverage),
                    float(row.confirmation_trace_broad_score),
                    float(row.confirmation_trace_broad_coverage),
                )
                for row in model_confirmation.itertuples(index=False)
            }

        def bank_row(
            heads: Sequence[tuple[int, int]],
            *,
            condition: str,
            bank_size: int,
            repeat: int,
        ) -> dict[str, Any]:
            discovery_mass = _bank_attention_mass(model_frame, heads)
            confirmation_values = [
                confirmation_lookup.get(
                    (int(layer), int(head)),
                    (np.nan,) * 8,
                )
                for layer, head in heads
            ]
            confirmation_raw = np.asarray(
                [value[0] for value in confirmation_values], dtype=float
            )
            confirmation_relative = np.asarray(
                [value[1] for value in confirmation_values], dtype=float
            )
            confirmation_trace_raw = np.asarray(
                [value[2] for value in confirmation_values], dtype=float
            )
            confirmation_trace_relative = np.asarray(
                [value[3] for value in confirmation_values], dtype=float
            )
            confirmation_prompt_broad = np.asarray(
                [value[4] for value in confirmation_values], dtype=float
            )
            confirmation_prompt_coverage = np.asarray(
                [value[5] for value in confirmation_values], dtype=float
            )
            confirmation_trace_broad = np.asarray(
                [value[6] for value in confirmation_values], dtype=float
            )
            confirmation_trace_coverage = np.asarray(
                [value[7] for value in confirmation_values], dtype=float
            )
            selection_confirmation_raw = (
                confirmation_raw
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_raw
            )
            selection_confirmation_relative = (
                confirmation_relative
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_relative
            )
            selection_confirmation_broad = (
                confirmation_prompt_broad
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_broad
            )
            selection_confirmation_coverage = (
                confirmation_prompt_coverage
                if mechanism == "answer_prompt_aggregation"
                else confirmation_trace_coverage
            )
            return {
                "model_label": model_label,
                "mechanism": mechanism,
                "query_site_kind": "answer_query_v3",
                "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
                "condition": condition,
                "bank_size": int(bank_size),
                "repeat": int(repeat),
                "heads": json.dumps(heads),
                **discovery_mass,
                **_bank_answer_selection_mass(model_frame, heads),
                "confirmation_target_needle_raw_mass": (
                    float(np.nanmean(confirmation_raw))
                    if np.isfinite(confirmation_raw).any()
                    else np.nan
                ),
                "confirmation_target_needle_relative_mass": (
                    float(np.nanmean(confirmation_relative))
                    if np.isfinite(confirmation_relative).any()
                    else np.nan
                ),
                "confirmation_mass_split": "confirmation_descriptive_only",
                "confirmation_selected_aggregation_raw_mass": (
                    float(np.nanmean(selection_confirmation_raw))
                    if np.isfinite(selection_confirmation_raw).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_relative_mass": (
                    float(np.nanmean(selection_confirmation_relative))
                    if np.isfinite(selection_confirmation_relative).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_broad_score": (
                    float(np.nanmean(selection_confirmation_broad))
                    if np.isfinite(selection_confirmation_broad).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_broad_coverage": (
                    float(np.nanmean(selection_confirmation_coverage))
                    if np.isfinite(selection_confirmation_coverage).any()
                    else np.nan
                ),
            }

        for bank_size in config.causal_head_bank_sizes:
            if bank_size > len(ordered):
                continue
            chosen = _control_constructible_ranked_heads(
                model_frame,
                ordered,
                bank_size=int(bank_size),
            )
            plan_rows.append(
                bank_row(
                    chosen,
                    condition=f"{mechanism}_ranked",
                    bank_size=bank_size,
                    repeat=0,
                )
            )
            try:
                controls = layer_matched_random_controls(
                    model_frame,
                    chosen,
                    repeats=config.causal_random_controls,
                    seed_text=(
                        f"v5-answer-query:{model_label}:{mechanism}:K{bank_size}"
                    ),
                )
            except ValueError as error:
                skipped_banks.append(
                    {
                        "model_label": str(model_label),
                        "mechanism": str(mechanism),
                        "bank_size": int(bank_size),
                        "ranked_treatment_included": True,
                        "control_status": (
                            "not_constructible_disjoint_exact_layer_match"
                        ),
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    bank_row(
                        control,
                        condition=f"{mechanism}_layer_matched_random",
                        bank_size=bank_size,
                        repeat=repeat,
                    )
                )

    # The answer readout has two potentially redundant routes: direct prompt
    # aggregation and aggregation over the generated reasoning trace.  Report
    # their joint intervention as a third frozen treatment rather than asking
    # readers to infer it from two independent runs.
    for model_label, model_ranking in ranking.groupby("model_label", sort=True):
        mechanism_frames = {
            str(mechanism): frame.sort_values("discovery_rank").reset_index(
                drop=True
            )
            for mechanism, frame in model_ranking.groupby("mechanism", sort=True)
        }
        if set(mechanism_frames) != set(answer_mechanisms):
            raise ValueError(
                f"Answer aggregation mechanisms are incomplete for {model_label}: "
                f"{sorted(mechanism_frames)}"
            )
        prompt_frame = mechanism_frames["answer_prompt_aggregation"]
        trace_frame = mechanism_frames["answer_trace_aggregation"]
        prompt_ordered = [
            (int(row.layer), int(row.head))
            for row in prompt_frame.itertuples(index=False)
        ]
        trace_ordered = [
            (int(row.layer), int(row.head))
            for row in trace_frame.itertuples(index=False)
        ]
        confirmation_lookup = {}
        if not confirmation_ranking.empty:
            active_confirmation = confirmation_ranking.loc[
                confirmation_ranking["model_label"].astype(str).eq(
                    str(model_label)
                )
            ]
            confirmation_lookup = {
                (int(row.layer), int(row.head)): (
                    float(row.confirmation_target_raw_mass),
                    float(row.confirmation_target_relative_mass),
                    float(row.confirmation_trace_raw_mass),
                    float(row.confirmation_trace_relative_mass),
                    float(row.confirmation_prompt_broad_score),
                    float(row.confirmation_prompt_broad_coverage),
                    float(row.confirmation_trace_broad_score),
                    float(row.confirmation_trace_broad_coverage),
                )
                for row in active_confirmation.itertuples(index=False)
            }

        def joint_row(
            active_heads: Sequence[tuple[int, int]],
            *,
            condition: str,
            bank_size: int,
            repeat: int,
            prompt_heads: Sequence[tuple[int, int]],
            trace_heads: Sequence[tuple[int, int]],
        ) -> dict[str, Any]:
            generic_mass = _bank_attention_mass(prompt_frame, active_heads)
            prompt_mass = _bank_answer_selection_mass(
                prompt_frame, active_heads
            )
            trace_mass = _bank_answer_selection_mass(trace_frame, active_heads)
            confirmation_values = [
                confirmation_lookup.get(
                    (int(layer), int(head)),
                    (np.nan,) * 8,
                )
                for layer, head in active_heads
            ]
            confirmation_array = np.asarray(confirmation_values, dtype=float)
            prompt_set = set(prompt_heads)
            trace_set = set(trace_heads)
            return {
                "model_label": model_label,
                "mechanism": "answer_prompt_and_trace_aggregation",
                "query_site_kind": "answer_query_v3",
                "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
                "condition": condition,
                "bank_size": int(bank_size),
                "repeat": int(repeat),
                "heads": json.dumps(list(active_heads)),
                "selected_head_count": int(len(active_heads)),
                "prompt_bank_size": int(len(prompt_heads)),
                "trace_bank_size": int(len(trace_heads)),
                "prompt_trace_head_overlap": int(len(prompt_set & trace_set)),
                "prompt_bank_heads": json.dumps(list(prompt_heads)),
                "trace_bank_heads": json.dumps(list(trace_heads)),
                **generic_mass,
                "selected_aggregation_raw_mass": np.nan,
                "selected_aggregation_relative_mass": np.nan,
                "selected_aggregation_broad_score": np.nan,
                "selected_aggregation_broad_coverage": np.nan,
                "selected_aggregation_relative_defined_heads": 0,
                "selected_aggregation_metric": (
                    "joint_prompt_and_trace_reported_separately"
                ),
                "prompt_aggregation_raw_mass": prompt_mass[
                    "selected_aggregation_raw_mass"
                ],
                "prompt_aggregation_relative_mass": prompt_mass[
                    "selected_aggregation_relative_mass"
                ],
                "prompt_aggregation_broad_score": prompt_mass[
                    "selected_aggregation_broad_score"
                ],
                "prompt_aggregation_broad_coverage": prompt_mass[
                    "selected_aggregation_broad_coverage"
                ],
                "trace_aggregation_raw_mass": trace_mass[
                    "selected_aggregation_raw_mass"
                ],
                "trace_aggregation_relative_mass": trace_mass[
                    "selected_aggregation_relative_mass"
                ],
                "trace_aggregation_broad_score": trace_mass[
                    "selected_aggregation_broad_score"
                ],
                "trace_aggregation_broad_coverage": trace_mass[
                    "selected_aggregation_broad_coverage"
                ],
                "confirmation_target_needle_raw_mass": (
                    float(np.nanmean(confirmation_array[:, 0]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 0]).any()
                    else np.nan
                ),
                "confirmation_target_needle_relative_mass": (
                    float(np.nanmean(confirmation_array[:, 1]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 1]).any()
                    else np.nan
                ),
                "confirmation_selected_aggregation_raw_mass": np.nan,
                "confirmation_selected_aggregation_relative_mass": np.nan,
                "confirmation_selected_aggregation_broad_score": np.nan,
                "confirmation_selected_aggregation_broad_coverage": np.nan,
                "confirmation_prompt_aggregation_raw_mass": (
                    float(np.nanmean(confirmation_array[:, 0]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 0]).any()
                    else np.nan
                ),
                "confirmation_prompt_aggregation_relative_mass": (
                    float(np.nanmean(confirmation_array[:, 1]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 1]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_raw_mass": (
                    float(np.nanmean(confirmation_array[:, 2]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 2]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_relative_mass": (
                    float(np.nanmean(confirmation_array[:, 3]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 3]).any()
                    else np.nan
                ),
                "confirmation_prompt_aggregation_broad_score": (
                    float(np.nanmean(confirmation_array[:, 4]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 4]).any()
                    else np.nan
                ),
                "confirmation_prompt_aggregation_broad_coverage": (
                    float(np.nanmean(confirmation_array[:, 5]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 5]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_broad_score": (
                    float(np.nanmean(confirmation_array[:, 6]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 6]).any()
                    else np.nan
                ),
                "confirmation_trace_aggregation_broad_coverage": (
                    float(np.nanmean(confirmation_array[:, 7]))
                    if confirmation_array.size
                    and np.isfinite(confirmation_array[:, 7]).any()
                    else np.nan
                ),
                "confirmation_mass_split": "confirmation_descriptive_only",
            }

        for bank_size in config.causal_head_bank_sizes:
            if bank_size > len(prompt_ordered) or bank_size > len(trace_ordered):
                continue
            prompt_heads, trace_heads = _control_constructible_joint_ranked_heads(
                prompt_frame,
                prompt_ordered,
                trace_ordered,
                bank_size=int(bank_size),
            )
            joint_heads = list(dict.fromkeys([*prompt_heads, *trace_heads]))
            plan_rows.append(
                joint_row(
                    joint_heads,
                    condition="answer_prompt_and_trace_aggregation_ranked",
                    bank_size=int(bank_size),
                    repeat=0,
                    prompt_heads=prompt_heads,
                    trace_heads=trace_heads,
                )
            )
            try:
                controls = layer_matched_random_controls(
                    prompt_frame,
                    joint_heads,
                    repeats=config.causal_random_controls,
                    seed_text=(
                        f"v5-answer-query:{model_label}:joint:K{bank_size}"
                    ),
                )
            except ValueError as error:
                skipped_banks.append(
                    {
                        "model_label": str(model_label),
                        "mechanism": "answer_prompt_and_trace_aggregation",
                        "bank_size": int(bank_size),
                        "ranked_treatment_included": True,
                        "control_status": (
                            "not_constructible_disjoint_exact_layer_match"
                        ),
                        "reason": str(error),
                    }
                )
                continue
            for repeat, control in enumerate(controls, start=1):
                plan_rows.append(
                    joint_row(
                        control,
                        condition=(
                            "answer_prompt_and_trace_aggregation_"
                            "layer_matched_random"
                        ),
                        bank_size=int(bank_size),
                        repeat=repeat,
                        prompt_heads=prompt_heads,
                        trace_heads=trace_heads,
                    )
                )
    paths = {
        "ranking": output / "discovery_answer_query_head_ranking.csv",
        "plan": output / "answer_query_causal_plan.csv",
        "audit": output / "answer_query_causal_plan_audit.json",
    }
    ranking.to_csv(paths["ranking"], index=False)
    pd.DataFrame(plan_rows).to_csv(paths["plan"], index=False)
    paths["audit"].write_text(
        json.dumps(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
                "site_id": "answer_query_v3",
                "query_definition": (
                    "literal baseline token immediately before the first "
                    "numeric answer token"
                ),
                "mechanisms": [
                    *answer_mechanisms,
                    "answer_prompt_and_trace_aggregation",
                ],
                "reported_behavioral_conditions": [
                    "clean",
                    "answer_prompt_aggregation_ranked",
                    "answer_trace_aggregation_ranked",
                    "answer_prompt_and_trace_aggregation_ranked",
                ],
                "selection_split": "discovery",
                "confirmation_used_for_selection": False,
                "selection_cohort": "one_to_one",
                "ranked_treatment_policy": (
                    "preserve discovery rank subject to per-layer occupancy "
                    "<= floor(available_heads/2), guaranteeing a disjoint "
                    "exact layer-matched control bank"
                ),
                "joint_ranked_treatment_policy": (
                    "equal-weight alternating greedy selection over the "
                    "independent prompt and trace discovery rankings under "
                    "the same control-constructability constraint"
                ),
                "selection_metrics": {
                    "answer_prompt_aggregation": (
                        "mean(prompt exact-span total mass * normalized "
                        "effective-span coverage), identical to non-thinking "
                        "broad_primary"
                    ),
                    "answer_trace_aggregation": (
                        "mean(trace registered-item total mass * normalized "
                        "effective-span coverage), identical broad-primary "
                        "definition applied independently to trace items"
                    ),
                },
                "relative_mass_denominator": "all_prompt_attention_mass",
                "registered_bank_sizes": list(config.causal_head_bank_sizes),
                "random_control": (
                    "disjoint exact layer-matched random heads; structural "
                    "unavailability is reported, never silently substituted"
                ),
                "skipped_banks": skipped_banks,
                "attention_source": str(Path(attention_csv).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def completion_metrics(result: Mapping[str, Any], *, gold_count: int) -> dict[str, Any]:
    full = str(result.get("full_answer_text", ""))
    prediction = parse_total(full)
    return {
        "prediction": prediction,
        "exact_count": prediction == int(gold_count),
        "signed_error": prediction - int(gold_count) if prediction is not None else None,
        "absolute_error": (
            abs(prediction - int(gold_count)) if prediction is not None else None
        ),
        "completion_text_raw": result.get("completion_text_raw"),
        "generated_token_count": result.get("generated_token_count"),
        "generation_truncated": result.get("generation_truncated"),
    }


@torch.inference_mode()
def capture_natural_head_writes(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    *,
    layers: Iterable[int] | None = None,
    count_direction: (
        np.ndarray
        | torch.Tensor
        | Mapping[int, np.ndarray | torch.Tensor]
        | None
    ) = None,
) -> tuple[pd.DataFrame, dict[tuple[int, int], torch.Tensor]]:
    """Decompose the natural attention output at one registered query.

    The input to each layer's output projection is the concatenated,
    attention-weighted head output. For a linear output projection, evaluating
    one head slice at a time and subtracting the zero-input output gives its
    exact natural residual write, including GQA/global-layer differences
    already resolved by ``DecoderAdapter``. This implementation is native to
    V5 and has no V4.4.2+ experiment dependency.
    """

    active_layers = (
        tuple(range(int(adapter.num_layers)))
        if layers is None
        else tuple(sorted({int(layer) for layer in layers}))
    )
    if not active_layers or any(
        layer < 0 or layer >= int(adapter.num_layers) for layer in active_layers
    ):
        raise ValueError("Invalid natural-write layers")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in active_layers:

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
        ) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention output projection received no tensor")
            value = args[0]
            if value.ndim != 3:
                raise RuntimeError("Expected [batch,time,heads*head_dim] before o_proj")
            captured[layer] = value[0, int(encoding.query_position)].detach()

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    try:
        input_ids, attention_mask = _encoding_tensors(model, encoding)
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **_bounded_logits_kwargs(model),
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(active_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Natural-write capture missed layers {missing}")
    def normalized_direction(
        value: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        direction = torch.as_tensor(value, dtype=torch.float32)
        if direction.ndim != 1:
            raise ValueError("count_direction must be one residual vector")
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise ValueError("count_direction must be finite and nonzero")
        return direction / norm

    directions: dict[int, torch.Tensor] = {}
    if isinstance(count_direction, Mapping):
        directions = {
            int(layer): normalized_direction(value)
            for layer, value in count_direction.items()
        }
        missing_directions = sorted(set(active_layers) - set(directions))
        if missing_directions:
            raise ValueError(
                f"count_direction mapping is missing layers {missing_directions}"
            )
    elif count_direction is not None:
        shared_direction = normalized_direction(count_direction)
        directions = {layer: shared_direction for layer in active_layers}
    rows: list[dict[str, Any]] = []
    writes: dict[tuple[int, int], torch.Tensor] = {}
    for layer in active_layers:
        direction = directions.get(layer)
        projection = adapter.output_projections[layer]
        aggregate = captured[layer]
        heads = int(adapter.num_heads[layer])
        head_dim = int(adapter.head_dims[layer])
        if aggregate.numel() != heads * head_dim:
            raise RuntimeError(
                f"L{layer} o_proj input width disagrees with decoder adapter"
            )
        zero = torch.zeros_like(aggregate)
        zero_output = projection(zero.reshape(1, 1, -1))[0, 0]
        full_delta = projection(aggregate.reshape(1, 1, -1))[0, 0] - zero_output
        summed = torch.zeros_like(full_delta)
        layer_writes: list[torch.Tensor] = []
        for head in range(heads):
            left = head * head_dim
            right = left + head_dim
            isolated = zero.clone()
            isolated[left:right] = aggregate[left:right]
            write = (
                projection(isolated.reshape(1, 1, -1))[0, 0] - zero_output
            ).detach().float().cpu()
            writes[(int(layer), int(head))] = write
            layer_writes.append(write)
            summed = summed + write.to(device=summed.device, dtype=summed.dtype)
        reconstruction_error = float(
            torch.linalg.vector_norm(summed - full_delta).detach().float().cpu()
        )
        for head, write in enumerate(layer_writes):
            norm = float(torch.linalg.vector_norm(write))
            if direction is None:
                coefficient = np.nan
                cosine = np.nan
            else:
                active_direction = direction.to(write)
                coefficient = float(torch.dot(write, active_direction))
                cosine = coefficient / norm if norm > 0 else np.nan
            rows.append(
                {
                    "schema_version": CAUSAL_SCHEMA_VERSION,
                    "request_id": encoding.request_id,
                    "model_label": encoding.model_label,
                    "seed": encoding.seed,
                    "split": encoding.split,
                    "gold_count": encoding.count,
                    "site_id": encoding.selected_site.get("site_id"),
                    "layer": int(layer),
                    "head": int(head),
                    "layer_type": adapter.layer_types[layer],
                    "write_norm": norm,
                    "count_direction_coefficient": coefficient,
                    "count_direction_cosine": cosine,
                    "layer_reconstruction_error": reconstruction_error,
                }
            )
    return pd.DataFrame(rows), writes


def mechanism_continuations(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    mechanism: str,
    boundary_policy: str = "strict_registered",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if boundary_policy not in {"strict_registered", "item_end_fallback_v2"}:
        raise ValueError(f"Unknown causal boundary policy: {boundary_policy}")
    family = infer_model_family(row)
    parser = find_trace_count_sequence(
        raw_output_text(row),
        model_family=family,
        gold_records=gold_records(row),
    )
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw_output_text(row),
        baseline_output_token_ids=output_token_ids(row),
        sites=trace_char_sites(raw_output_text(row), parser),
    )
    by_id = {site.char_site.site_id: site for site in token_sites}
    cities = [str(value) for value in parser.item_gold_cities]
    specifications: list[
        tuple[str, tuple[str, ...], int, str, str | None]
    ] = []
    if mechanism == "targeted_retrieval":
        specifications = [
            (
                f"marker_end:{k}",
                (
                    (f"city_end:{k}", f"item_end:{k}")
                    if boundary_policy == "item_end_fallback_v2"
                    else (f"city_end:{k}",)
                ),
                k,
                "retrieve",
                city,
            )
            for k, city in enumerate(cities, start=1)
        ]
    elif mechanism == "progress_transition":
        specifications = [
            (
                f"item_end:{k}",
                (
                    (f"marker_end:{k + 1}", f"item_end:{k + 1}")
                    if boundary_policy == "item_end_fallback_v2"
                    else (f"marker_end:{k + 1}",)
                ),
                k,
                "continue",
                cities[k],
            )
            for k in range(1, len(cities))
        ]
        if cities and "answer_query" in by_id:
            specifications.append(
                (
                    f"item_end:{len(cities)}",
                    ("answer_query",),
                    len(cities),
                    "stop",
                    None,
                )
            )
    else:
        raise ValueError(f"Unknown head mechanism: {mechanism}")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for query_id, target_ids, occurrence, phase, target_city in specifications:
        query = by_id.get(query_id)
        primary_target_id = target_ids[0]
        reason = None
        if query is None:
            reason = "missing_registered_boundary"
        elif not query.alignment_eligible:
            reason = "ineligible_text_exact_alignment"
        elif query.alignment_strategy != "literal_baseline_token_prefix":
            reason = "local_causal_trial_requires_literal_baseline_alignment"
        elif query.prefix_token_count is None:
            reason = "empty_or_reversed_target_continuation"

        candidate_audit: list[dict[str, str]] = []
        selected_target_id: str | None = None
        target = None
        if reason is None:
            for target_id in target_ids:
                candidate = by_id.get(target_id)
                candidate_reason = None
                if candidate is None:
                    candidate_reason = "missing_registered_boundary"
                elif not candidate.alignment_eligible:
                    candidate_reason = "ineligible_text_exact_alignment"
                elif candidate.alignment_strategy != "literal_baseline_token_prefix":
                    candidate_reason = (
                        "local_causal_trial_requires_literal_baseline_alignment"
                    )
                elif (
                    candidate.prefix_token_count is None
                    or int(candidate.prefix_token_count)
                    <= int(query.prefix_token_count)
                ):
                    candidate_reason = "empty_or_reversed_target_continuation"
                candidate_audit.append(
                    {
                        "site_id": target_id,
                        "status": candidate_reason or "selected",
                    }
                )
                if candidate_reason is None:
                    selected_target_id = target_id
                    target = candidate
                    break
            if selected_target_id is None:
                if boundary_policy == "strict_registered":
                    reason = candidate_audit[0]["status"]
                else:
                    reason = "no_eligible_registered_target_boundary"
        payload = {
            "mechanism": mechanism,
            "query_site_id": query_id,
            "target_site_id": selected_target_id or primary_target_id,
            "primary_target_site_id": primary_target_id,
            "target_site_candidates": list(target_ids),
            "target_boundary_policy": boundary_policy,
            "target_boundary_variant": (
                "item_end_fallback"
                if selected_target_id is not None
                and selected_target_id != primary_target_id
                else "primary"
            ),
            "target_site_fallback": bool(
                selected_target_id is not None
                and selected_target_id != primary_target_id
            ),
            "target_candidate_audit": candidate_audit,
            "occurrence": int(occurrence),
            "transition_phase": phase,
            "target_city": target_city,
        }
        if reason is not None:
            excluded.append({**payload, "status": reason})
            continue
        eligible.append(
            {
                **payload,
                "query_output_token_count": int(query.prefix_token_count),
                "target_output_token_count": int(target.prefix_token_count),
            }
        )
    return eligible, excluded


@torch.inference_mode()
def _local_head_ablation_logits(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_position: int,
    target_token_count: int,
) -> torch.Tensor:
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3 or hook_position >= int(value.shape[1]):
                raise RuntimeError("Mechanism query is outside the prefill tensor")
            patched = value.clone()
            for head in layer_heads:
                left = head * head_dim
                patched[:, hook_position, left : left + head_dim] = 0
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    keep = int(target_token_count) + 1
    if _accepts_keyword(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = keep
    try:
        output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Mechanism trial returned no sequence logits")
    if int(logits.shape[1]) == int(encoding.sequence_length):
        selected = logits[0, hook_position : hook_position + target_token_count]
    elif int(logits.shape[1]) == keep:
        selected = logits[0, :target_token_count]
    else:
        raise RuntimeError(
            f"Unexpected kept-logit length {logits.shape[1]} (expected {keep})"
        )
    return selected.detach().float().cpu()


@torch.inference_mode()
def _scheduled_head_ablation_logits(
    model: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_positions: Sequence[int],
    score_positions: Sequence[int],
) -> torch.Tensor:
    """Jointly ablate one bank at many baseline positions in one prefill.

    ``score_positions`` are autoregressive predictor positions (the token at
    position ``p`` predicts the target token at ``p + 1``).  Returning only
    those vocabulary rows avoids copying the full trace-by-vocabulary tensor
    to CPU while preserving exact teacher-forced downstream scoring.
    """

    scheduled = tuple(sorted({int(value) for value in hook_positions}))
    scored = tuple(int(value) for value in score_positions)
    if not scheduled:
        raise ValueError("Scheduled head ablation needs at least one query")
    if not scored:
        raise ValueError("Scheduled head ablation needs at least one endpoint")
    sequence_length = int(encoding.sequence_length)
    if any(value < 0 or value >= sequence_length for value in scheduled):
        raise ValueError("A scheduled query is outside the teacher-forced trace")
    if any(value < 0 or value >= sequence_length for value in scored):
        raise ValueError("A score position is outside the teacher-forced trace")
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3 or int(value.shape[1]) != sequence_length:
                raise RuntimeError(
                    "Scheduled damage requires one full teacher-forced prefill"
                )
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * head_dim
                right = left + head_dim
                patched[:, scheduled, left:right] = 0
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Scheduled damage returned no sequence logits")
    if int(logits.shape[1]) != sequence_length:
        raise RuntimeError("Scheduled damage did not return full-sequence logits")
    position_tensor = torch.as_tensor(scored, dtype=torch.long, device=logits.device)
    return logits[0].index_select(0, position_tensor).detach().float().cpu()


@torch.inference_mode()
def generate_with_head_ablation_at_positions(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encoding: NativeTraceEncoding,
    heads: Sequence[tuple[int, int]],
    *,
    hook_positions: Sequence[int],
    max_new_tokens: int = 16,
    score_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Greedily decode after damaging registered positions in the prefill.

    Generation is the primary behavioral count endpoint.  Hooks apply during
    the initial teacher-forced prefill only; the patched KV/residual history is
    then carried through ordinary greedy decoding.
    """

    scheduled = tuple(sorted({int(value) for value in hook_positions}))
    if not scheduled:
        raise ValueError("Head-ablation generation needs at least one position")
    if any(value < 0 or value >= int(encoding.sequence_length) for value in scheduled):
        raise ValueError("A generation hook position is outside the input prefix")
    by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in heads:
        layer = int(raw_layer)
        head = int(raw_head)
        if not 0 <= layer < int(adapter.num_layers):
            raise ValueError(f"Invalid ablation layer: {layer}")
        if not 0 <= head < int(adapter.num_heads[layer]):
            raise ValueError(f"Invalid head L{layer}H{head}")
        by_layer.setdefault(layer, []).append(head)
    applied: dict[int, set[int]] = {layer: set() for layer in by_layer}
    handles = []
    for layer, layer_heads in by_layer.items():
        head_dim = int(adapter.head_dims[layer])

        def hook(
            _module: Any,
            args: tuple[Any, ...],
            *,
            layer: int = layer,
            layer_heads: tuple[int, ...] = tuple(sorted(set(layer_heads))),
            head_dim: int = head_dim,
        ) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("Attention projection received no head tensor")
            value = args[0]
            if value.ndim != 3:
                raise RuntimeError("Expected [batch,time,heads*head_dim]")
            active = [position for position in scheduled if position < value.shape[1]]
            if not active:
                return args
            patched = value.clone()
            for head in layer_heads:
                left = int(head) * head_dim
                right = left + head_dim
                patched[:, active, left:right] = 0
            applied[layer].update(active)
            return (patched, *args[1:])

        handles.append(
            adapter.output_projections[layer].register_forward_pre_hook(hook)
        )
    scored = tuple(int(value) for value in (score_positions or ()))
    if any(
        value < 0 or value >= int(encoding.sequence_length)
        for value in scored
    ):
        raise ValueError("A generation score position is outside the input prefix")
    captured_prefill_logits: list[torch.Tensor] = []
    capture_handle = None
    if scored:
        sequence_length = int(encoding.sequence_length)

        def capture_prefill_logits(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
        ) -> None:
            logits = getattr(output, "logits", None)
            if (
                captured_prefill_logits
                or not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or int(logits.shape[0]) != 1
                or int(logits.shape[1]) != sequence_length
            ):
                return
            positions = torch.as_tensor(
                scored,
                dtype=torch.long,
                device=logits.device,
            )
            captured_prefill_logits.append(
                logits[0]
                .index_select(0, positions)
                .detach()
                .float()
                .cpu()
            )

        capture_handle = model.register_forward_hook(capture_prefill_logits)
    try:
        if not scored:
            result = generate_answer_completion(
                model,
                tokenizer,
                encoding,
                max_new_tokens=max_new_tokens,
            )
        else:
            # A single generate() prefill now supplies both the frozen-trace
            # diagnostic rows and the KV cache used for actual greedy answer
            # generation.  ``logits_to_keep=0`` is required on architectures
            # that otherwise retain only the final prefill logit.
            if int(max_new_tokens) < 1:
                raise ValueError("max_new_tokens must be positive")
            input_ids, attention_mask = _encoding_tensors(model, encoding)
            generation_config = getattr(model, "generation_config", None)
            eos_value = (
                getattr(generation_config, "eos_token_id", None)
                if generation_config is not None
                else None
            )
            if eos_value is None:
                eos_value = getattr(tokenizer, "eos_token_id", None)
            if eos_value is None:
                eos_ids: list[int] = []
            elif isinstance(eos_value, (tuple, list, set)):
                eos_ids = [int(value) for value in eos_value]
            else:
                eos_ids = [int(eos_value)]
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None and eos_ids:
                pad_token_id = eos_ids[0]
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "do_sample": False,
                "max_new_tokens": int(max_new_tokens),
                "use_cache": True,
            }
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = int(pad_token_id)
            if _accepts_keyword(model, "logits_to_keep"):
                generation_kwargs["logits_to_keep"] = 0
            generated = model.generate(**generation_kwargs)
            if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
                sequences = getattr(generated, "sequences", None)
                if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                    raise RuntimeError(
                        "Model.generate did not return [batch, time] sequences"
                    )
                generated = sequences
            if generated.shape[0] != 1 or generated.shape[1] < input_ids.shape[1]:
                raise RuntimeError("Unexpected generated sequence shape")
            continuation = [
                int(value)
                for value in generated[
                    0, input_ids.shape[1] :
                ].detach().cpu().tolist()
            ]
            if not continuation:
                raise RuntimeError("Greedy V5 generation returned an empty continuation")
            eos_set = set(eos_ids)
            stopped_on_eos = bool(eos_set and continuation[-1] in eos_set)
            raw_text = tokenizer.decode(
                continuation,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            clean_text = tokenizer.decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            result = {
                "generated_token_ids": continuation,
                "generated_token_count": len(continuation),
                "generation_eos_token_ids": eos_ids,
                "stopped_on_eos": stopped_on_eos,
                "generation_truncated": bool(
                    len(continuation) >= int(max_new_tokens)
                    and not stopped_on_eos
                ),
                "completion_text_raw": str(raw_text),
                "completion_text": str(clean_text),
                "full_answer_text": "Total:" + str(clean_text),
            }
    finally:
        if capture_handle is not None:
            capture_handle.remove()
        for handle in handles:
            handle.remove()
    missed = {
        int(layer): sorted(set(scheduled) - positions)
        for layer, positions in applied.items()
        if set(scheduled) - positions
    }
    if missed:
        raise RuntimeError(f"Generation head hooks missed positions: {missed}")
    if scored and len(captured_prefill_logits) != 1:
        raise RuntimeError(
            "Fused generation did not capture exactly one full prefill logit tensor"
        )
    combined = {
        **result,
        "head_ablation_prefill_positions": list(scheduled),
        "head_ablation_prefill_position_count": len(scheduled),
        "head_ablation_layers": sorted(by_layer),
        "head_ablation_hook_audit": "PASS",
    }
    if scored:
        combined["prefill_selected_logits"] = captured_prefill_logits[0]
        combined["prefill_reuse_audit"] = "PASS_SINGLE_PREFILL"
    return combined


def _continuation_metrics(
    logits: torch.Tensor, target_ids: Sequence[int]
) -> dict[str, Any]:
    targets = torch.as_tensor(tuple(target_ids), dtype=torch.long)
    if logits.ndim != 2 or len(logits) != len(targets) or not len(targets):
        raise ValueError("Continuation logits/targets disagree")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    token_logp = log_probabilities[
        torch.arange(len(targets), dtype=torch.long), targets
    ]
    predictions = logits.argmax(dim=-1)
    first_target = int(targets[0])
    competing = logits[0].clone()
    competing[first_target] = -torch.inf
    top_competing_logit = float(competing.max())
    target_first_logit = float(logits[0, first_target])
    first_rank = int(
        1
        + torch.count_nonzero(
            logits[0] > logits[0, int(targets[0])]
        ).item()
    )
    return {
        "target_sequence_log_probability": float(token_logp.sum()),
        "target_mean_token_log_probability": float(token_logp.mean()),
        "target_sequence_teacher_forced_exact": bool(
            torch.equal(predictions, targets)
        ),
        "target_first_token_exact": bool(predictions[0] == targets[0]),
        "target_first_token_rank": first_rank,
        "target_first_token_logit": target_first_logit,
        "top_non_target_first_token_logit": top_competing_logit,
        "target_first_token_logit_margin": (
            target_first_logit - top_competing_logit
        ),
    }


@torch.inference_mode()
def run_mechanism_head_ablation_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    mechanism: str,
    heads: Sequence[tuple[int, int]],
    condition: str,
    boundary_policy: str = "strict_registered",
) -> list[dict[str, Any]]:
    """Run position-local targeted/progress teacher-forced necessity tests."""

    specifications, excluded = mechanism_continuations(
        row,
        tokenizer,
        mechanism=mechanism,
        boundary_policy=boundary_policy,
    )
    output: list[dict[str, Any]] = []
    for exclusion in excluded:
        output.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": f"{mechanism}_head_ablation",
                "condition": condition,
                "request_id": row.get("request_id", row.get("stimulus_id")),
                "model_label": row.get("model_label"),
                "seed": row.get("seed"),
                "split": row.get("split"),
                "gold_count": len(gold_records(row)),
                "heads": [[int(layer), int(head)] for layer, head in heads],
                **exclusion,
            }
        )
    baseline_ids = output_token_ids(row)
    prompt_count = len(prompt_token_ids(row))
    for specification in specifications:
        target_encoding = build_native_trace_encoding(
            row,
            tokenizer,
            site_id=str(specification["target_site_id"]),
            candidate_counts=(),
        )
        query_output_count = int(specification["query_output_token_count"])
        target_output_count = int(specification["target_output_token_count"])
        target_ids = baseline_ids[query_output_count:target_output_count]
        hook_position = prompt_count + query_output_count - 1
        logits = _local_head_ablation_logits(
            model,
            adapter,
            target_encoding,
            heads,
            hook_position=hook_position,
            target_token_count=len(target_ids),
        )
        output.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": f"{mechanism}_head_ablation",
                "condition": condition,
                "request_id": target_encoding.request_id,
                "model_label": target_encoding.model_label,
                "seed": target_encoding.seed,
                "split": target_encoding.split,
                "gold_count": target_encoding.count,
                "heads": [[int(layer), int(head)] for layer, head in heads],
                "bank_size": len(heads),
                "status": "ok",
                "target_token_count": len(target_ids),
                "target_text": tokenizer.decode(
                    list(target_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                **specification,
                **_continuation_metrics(logits, target_ids),
            }
        )
    return output


@torch.inference_mode()
def run_answer_query_head_ablation_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    heads: Sequence[tuple[int, int]],
    condition: str,
    site_id: str = "answer_query_v3",
) -> dict[str, Any]:
    """Test one frozen head bank at the true final answer-query token."""

    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id=site_id,
        candidate_counts=tuple(range(1, 11)),
    )
    answer_ids = dict(encoding.count_candidate_answer_token_ids)[encoding.count]
    # ``_local_head_ablation_logits`` expects the teacher-forced target tokens
    # to be present after the query position.  Without this extension,
    # ``logits_to_keep=target_count+1`` returns the query's predecessor as the
    # first kept position for a prefix-only answer encoding, silently scoring
    # an off-by-one logit.  Keep ``query_position`` fixed at the literal token
    # immediately before the numeric answer and append only the frozen gold
    # continuation used for scoring.
    scoring_encoding = replace(
        encoding,
        input_ids=encoding.input_ids + tuple(int(value) for value in answer_ids),
        attention_mask=encoding.attention_mask + (1,) * len(answer_ids),
    )
    logits = _local_head_ablation_logits(
        model,
        adapter,
        scoring_encoding,
        heads,
        hook_position=encoding.query_position,
        target_token_count=len(answer_ids),
    )
    metrics = _continuation_metrics(logits, answer_ids)
    generation = generate_with_head_ablation_at_positions(
        model,
        tokenizer,
        adapter,
        encoding,
        heads,
        hook_positions=[int(encoding.query_position)],
        max_new_tokens=16,
    )
    behavior = completion_metrics(generation, gold_count=encoding.count)
    probabilities = torch.softmax(logits[0], dim=-1)
    first_id = int(answer_ids[0])
    candidate_first_ids = {
        int(count): int(ids[0])
        for count, ids in encoding.count_candidate_answer_token_ids
    }
    candidate_logits = {
        count: float(logits[0, token_id])
        for count, token_id in candidate_first_ids.items()
    }
    competing_counts = {
        count: value
        for count, value in candidate_logits.items()
        if int(candidate_first_ids[count]) != first_id
    }
    best_candidate_logit = max(candidate_logits.values())
    candidate_argmax_counts = sorted(
        count
        for count, value in candidate_logits.items()
        if value == best_candidate_logit
    )
    target_token_alias_counts = sorted(
        count
        for count, token_id in candidate_first_ids.items()
        if int(token_id) == first_id
    )
    candidate_margin = (
        float(candidate_logits[encoding.count])
        - float(max(competing_counts.values()))
        if competing_counts
        else float("nan")
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "answer_query_head_ablation_v5_broad_factorial",
        "condition": condition,
        "request_id": encoding.request_id,
        "stimulus_id": encoding.stimulus_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "baseline_exact_count": bool(parse_trace_record(row)["exact_count"]),
        "site_id": site_id,
        "query_position": int(encoding.query_position),
        "query_definition": (
            "literal baseline token immediately before the first numeric "
            "answer token"
        ),
        "heads": [[int(layer), int(head)] for layer, head in heads],
        "bank_size": int(len(heads)),
        "status": "ok",
        "target_token_count": int(len(answer_ids)),
        "target_text": tokenizer.decode(
            list(answer_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "target_first_token_probability": float(probabilities[first_id]),
        "behavioral_endpoint": "greedy_generated_numeric_answer",
        "prediction": behavior["prediction"],
        "exact_count": behavior["exact_count"],
        "signed_error": behavior["signed_error"],
        "absolute_error": behavior["absolute_error"],
        "completion_text_raw": behavior["completion_text_raw"],
        "generated_token_count": behavior["generated_token_count"],
        "generation_truncated": behavior["generation_truncated"],
        "head_ablation_hook_audit": generation[
            "head_ablation_hook_audit"
        ],
        "answer_candidate_argmax_counts": candidate_argmax_counts,
        "answer_candidate_argmax_contains_gold": bool(
            int(encoding.count) in candidate_argmax_counts
        ),
        "answer_gold_first_token_alias_counts": target_token_alias_counts,
        "answer_gold_first_token_identifies_unique_count": bool(
            len(target_token_alias_counts) == 1
        ),
        "answer_candidate_first_token_logits": candidate_logits,
        "answer_gold_vs_best_distinct_count_token_logit_margin": (
            candidate_margin
        ),
        **metrics,
    }


@torch.inference_mode()
def run_projected_patch_trials(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    receiver_row: Mapping[str, Any],
    donor_row: Mapping[str, Any],
    *,
    receiver_site_id: str,
    donor_site_id: str,
    layer: int,
    basis: np.ndarray | torch.Tensor,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    """Run a subspace donor patch with preregistered matched controls."""

    receiver, receiver_state = capture_site_state(
        model,
        adapter,
        tokenizer,
        receiver_row,
        site_id=receiver_site_id,
        layer=layer,
    )
    donor, donor_state = capture_site_state(
        model,
        adapter,
        tokenizer,
        donor_row,
        site_id=donor_site_id,
        layer=layer,
    )
    return run_projected_patch_trials_from_states(
        model,
        tokenizer,
        adapter,
        receiver,
        receiver_state,
        donor,
        donor_state,
        receiver_site_id=receiver_site_id,
        donor_site_id=donor_site_id,
        layer=layer,
        basis=basis,
        max_new_tokens=max_new_tokens,
    )


@torch.inference_mode()
def run_projected_patch_trials_from_states(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    receiver: NativeTraceEncoding,
    receiver_state: torch.Tensor,
    donor: NativeTraceEncoding,
    donor_state: torch.Tensor,
    *,
    receiver_site_id: str,
    donor_site_id: str,
    layer: int,
    basis: np.ndarray | torch.Tensor,
    max_new_tokens: int = 16,
    self_patch_result: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run registered patch conditions from reusable captured site states."""

    if receiver.model_label != donor.model_label:
        raise ValueError("Receiver and donor must use the same registered model")
    if receiver.count == donor.count:
        raise ValueError(
            "Projected donor trials require different receiver/donor counts; "
            "self patch is run internally as a control"
        )
    basis_tensor = torch.as_tensor(basis, dtype=torch.float32)
    if basis_tensor.ndim != 2 or basis_tensor.shape[0] != receiver_state.numel():
        raise ValueError("Patch basis must have shape [hidden, rank]")
    reused_self_patch = self_patch_result is not None
    clean = (
        dict(self_patch_result)
        if self_patch_result is not None
        else generate_with_residual_interventions(
            model,
            tokenizer,
            adapter,
            receiver,
            {int(layer): ([receiver.query_position], receiver_state)},
            max_new_tokens=max_new_tokens,
        )
    )
    full_donor = generate_with_residual_interventions(
        model,
        tokenizer,
        adapter,
        receiver,
        {int(layer): ([receiver.query_position], donor_state)},
        max_new_tokens=max_new_tokens,
    )
    conditions = run_counter_subspace_conditions(
        model,
        tokenizer,
        adapter,
        receiver,
        source_layer=int(layer),
        source_positions=[receiver.query_position],
        receiver_source_state=receiver_state,
        donor_source_state=donor_state,
        source_basis=basis_tensor,
        random_seed=_stable_seed(
            f"v5-patch:{receiver.request_id}:{donor.request_id}:L{layer}"
        )
        % (2**31 - 1),
        max_new_tokens=max_new_tokens,
    )
    audit = dict(conditions.pop("_audit"))
    result_rows: list[dict[str, Any]] = []
    for condition, result in (
        ("self_patch", clean),
        ("full_donor_patch", full_donor),
        ("projected_donor_patch", conditions["projected_patch"]),
        ("orthogonal_norm_matched", conditions["orthogonal_norm_matched"]),
    ):
        result_rows.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": "answer_query_patch",
                "condition": condition,
                "request_id": receiver.request_id,
                "donor_request_id": donor.request_id,
                "model_label": receiver.model_label,
                "seed": receiver.seed,
                "split": receiver.split,
                "gold_count": receiver.count,
                "donor_count": donor.count,
                "receiver_site_id": receiver_site_id,
                "donor_site_id": donor_site_id,
                "layer": int(layer),
                "rank": int(basis_tensor.shape[1]),
                "captured_state_cache_audit": "PASS_REUSED_SITE_STATE",
                "self_patch_cache_reused": bool(reused_self_patch),
                **audit,
                **completion_metrics(result, gold_count=receiver.count),
            }
        )
    return result_rows


@torch.inference_mode()
def capture_site_state(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    site_id: str,
    layer: int,
) -> tuple[NativeTraceEncoding, torch.Tensor]:
    encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        [encoding.query_position],
        layers=[layer],
    )
    return encoding, captured[int(layer)][0]


@torch.inference_mode()
def run_residual_patch_trial(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    receiver_row: Mapping[str, Any],
    donor_row: Mapping[str, Any],
    *,
    receiver_site_id: str,
    donor_site_id: str,
    layer: int,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    receiver = build_native_trace_encoding(
        receiver_row, tokenizer, site_id=receiver_site_id
    )
    donor, donor_state = capture_site_state(
        model,
        adapter,
        tokenizer,
        donor_row,
        site_id=donor_site_id,
        layer=layer,
    )
    result = generate_with_residual_interventions(
        model,
        tokenizer,
        adapter,
        receiver,
        {int(layer): ([receiver.query_position], donor_state)},
        max_new_tokens=max_new_tokens,
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "answer_query_patch",
        "condition": "donor_patch",
        "request_id": receiver.request_id,
        "donor_request_id": donor.request_id,
        "model_label": receiver.model_label,
        "seed": receiver.seed,
        "split": receiver.split,
        "gold_count": receiver.count,
        "donor_count": donor.count,
        "receiver_site_id": receiver_site_id,
        "donor_site_id": donor_site_id,
        "layer": int(layer),
        **completion_metrics(result, gold_count=receiver.count),
    }


def fit_centroid_subspace(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if states.ndim != 2 or len(states) != len(labels):
        raise ValueError("states/labels must be [observations, hidden]/[observations]")
    if int(rank) < 1:
        raise ValueError("rank must be positive")
    values = np.unique(labels)
    if len(values) < 2:
        raise ValueError("At least two labels are needed to fit a count subspace")
    centroids = np.stack([states[labels == value].mean(axis=0) for value in values])
    center = centroids.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centroids - center, full_matrices=False)
    effective_rank = min(int(rank), len(values) - 1, states.shape[1])
    basis = vh[:effective_rank].T
    centered_values = values.astype(float) - float(np.mean(values))
    scores = (centroids - center) @ basis
    for component in range(effective_rank):
        orientation = float(np.dot(centered_values, scores[:, component]))
        if orientation < 0:
            basis[:, component] *= -1
        elif abs(orientation) <= 1e-12:
            anchor = int(np.argmax(np.abs(basis[:, component])))
            if basis[anchor, component] < 0:
                basis[:, component] *= -1
    return center.astype(np.float32), basis.astype(np.float32)


@torch.inference_mode()
def run_subspace_ablation(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    row: Mapping[str, Any],
    *,
    site_id: str,
    layer: int,
    center: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
    dose: float = 1.0,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    center_tensor = torch.as_tensor(center, dtype=torch.float32)
    basis_tensor = torch.as_tensor(basis, dtype=torch.float32)
    result = generate_with_residual_transforms(
        model,
        tokenizer,
        adapter,
        encoding,
        {
            int(layer): (
                [encoding.query_position],
                removal_transform(center_tensor, basis_tensor, dose=dose),
            )
        },
        max_new_tokens=max_new_tokens,
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "trace_subspace_ablation",
        "condition": "count_subspace_removal",
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "site_id": site_id,
        "layer": int(layer),
        "rank": int(basis_tensor.shape[1]),
        "dose": float(dose),
        **completion_metrics(result, gold_count=encoding.count),
    }


def _ordinary_segments(
    encoding: NativeTraceEncoding,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    spans = sorted(encoding.needle_spans, key=lambda span: span.start)
    if not spans:
        raise ValueError("Token corruption needs visible trace-item spans")
    lengths = [int(span.end) - int(span.start) for span in spans]
    forbidden = {
        position
        for span in spans
        for position in range(int(span.start), int(span.end))
    }
    used = set(forbidden)
    start = 1
    end = int(encoding.prompt_token_count)

    def allocate(length: int) -> tuple[int, int]:
        for candidate in range(start, end - length + 1):
            positions = set(range(candidate, candidate + length))
            if not positions.intersection(used):
                used.update(positions)
                return candidate, candidate + length
        raise RuntimeError("Could not allocate equal-length prompt control segment")

    sources = [allocate(length) for length in lengths]
    controls = [allocate(length) for length in lengths]
    control_sources = [allocate(length) for length in lengths]
    return sources, controls, control_sources


def corrupt_trace_tokens(
    encoding: NativeTraceEncoding,
) -> tuple[NativeTraceEncoding, NativeTraceEncoding, dict[str, Any]]:
    """Return trace corruption and equal-budget ordinary-token control."""

    sources, controls, control_sources = _ordinary_segments(encoding)
    clean = list(encoding.input_ids)
    trace_ids = clean.copy()
    control_ids = clean.copy()
    trace_changed = 0
    control_changed = 0
    for span, source in zip(encoding.needle_spans, sources):
        target = (int(span.start), int(span.end))
        replacement = clean[source[0] : source[1]]
        before = trace_ids[target[0] : target[1]]
        trace_ids[target[0] : target[1]] = replacement
        trace_changed += sum(left != right for left, right in zip(before, replacement))
    for target, source in zip(controls, control_sources):
        replacement = clean[source[0] : source[1]]
        before = control_ids[target[0] : target[1]]
        control_ids[target[0] : target[1]] = replacement
        control_changed += sum(left != right for left, right in zip(before, replacement))
    return (
        replace(encoding, input_ids=tuple(trace_ids)),
        replace(encoding, input_ids=tuple(control_ids)),
        {
            "token_budget": sum(
                int(span.end) - int(span.start) for span in encoding.needle_spans
            ),
            "trace_changed_tokens": trace_changed,
            "control_changed_tokens": control_changed,
            "trace_sources": sources,
            "control_targets": controls,
            "control_sources": control_sources,
        },
    )


def _matched_nontrace_positions(
    encoding: NativeTraceEncoding,
    *,
    token_budget: int,
) -> list[int]:
    forbidden = {
        position
        for span in encoding.needle_spans
        for position in range(int(span.start), int(span.end))
    }
    candidates = [
        position
        for position in range(1, int(encoding.query_position))
        if position not in forbidden
    ]
    if len(candidates) < token_budget:
        raise RuntimeError("Not enough non-trace keys for a matched context mask")
    # Evenly cover the prompt/trace depth instead of taking one local window.
    indices = np.linspace(0, len(candidates) - 1, token_budget, dtype=int)
    return [candidates[index] for index in indices]


def query_context_mask(
    encoding: NativeTraceEncoding,
    *,
    condition: str,
) -> torch.Tensor:
    query = int(encoding.query_position)
    mask = torch.zeros((1, query + 1), dtype=torch.long)
    if condition == "clean":
        mask[:] = 1
    elif condition == "trace_only":
        for span in encoding.needle_spans:
            mask[:, int(span.start) : min(int(span.end), query + 1)] = 1
        mask[:, query] = 1
    elif condition == "matched_nontrace_only":
        trace_positions = {
            position
            for span in encoding.needle_spans
            for position in range(
                int(span.start), min(int(span.end), query + 1)
            )
        }
        trace_positions.add(query)
        # The query itself is always available in both sparse conditions.
        # Match the number of *other* visible keys exactly.
        budget = len(trace_positions - {query})
        if budget < 1:
            raise RuntimeError("No earlier trace keys exist for a matched mask")
        positions = _matched_nontrace_positions(encoding, token_budget=budget)
        mask[:, positions] = 1
        mask[:, query] = 1
    else:
        raise ValueError(f"Unknown query context condition: {condition}")
    if int(mask.sum()) < 1:
        raise RuntimeError("Query context mask is empty")
    return mask


@torch.inference_mode()
def run_query_context_mask_trial(
    model: Any,
    adapter: DecoderAdapter,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    site_id: str,
    condition: str,
    layers: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Capture item-end/answer-query states under a key-context intervention."""

    encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    query = int(encoding.query_position)
    if query < 1:
        raise ValueError("A context-mask trial needs a non-initial query")
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefix_output = model(
        input_ids=input_ids[:, :query],
        attention_mask=attention_mask[:, :query],
        use_cache=True,
        output_attentions=False,
        **_bounded_logits_kwargs(model),
    )
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Context-mask prefix forward returned no KV cache")
    active_layers = (
        tuple(range(int(adapter.num_layers)))
        if layers is None
        else tuple(sorted({int(layer) for layer in layers}))
    )
    if not active_layers or any(
        layer < 0 or layer >= int(adapter.num_layers) for layer in active_layers
    ):
        raise ValueError("Invalid context-mask capture layers")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in active_layers:

        def hook(_module: Any, _args: tuple[Any, ...], output: Any, *, layer: int = layer):
            hidden = _tensor_from_output(output)
            captured[layer] = hidden[0, -1].detach().float().cpu()

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query : query + 1],
        "attention_mask": query_context_mask(
            encoding, condition=condition
        ).to(input_ids.device),
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor(
            [[query]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor(
            [query], dtype=torch.long, device=input_ids.device
        )
    shared = getattr(prefix_output, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared
    try:
        model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(active_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Context-mask trial missed layers {missing}")
    states = np.stack([captured[layer].numpy() for layer in active_layers])
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "experiment_id": "earlier_span_attention",
        "condition": condition,
        "request_id": encoding.request_id,
        "model_label": encoding.model_label,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "site_id": site_id,
        "occurrence": encoding.selected_site.get("occurrence"),
        "layers": list(active_layers),
        "states": states,
        "allowed_key_count": int(
            query_context_mask(encoding, condition=condition).sum().item()
        ),
    }


@torch.inference_mode()
def run_token_corruption_trial(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    config: V5Config,
    max_new_tokens: int = 16,
) -> list[dict[str, Any]]:
    encoding = build_native_trace_encoding(
        row,
        tokenizer,
        site_id="answer_query",
        candidate_counts=config.candidate_counts,
    )
    corrupted, control, audit = corrupt_trace_tokens(encoding)
    rows: list[dict[str, Any]] = []
    for condition, active in (
        ("clean", encoding),
        ("trace_corrupt", corrupted),
        ("ordinary_control", control),
    ):
        result = generate_answer_completion(
            model, tokenizer, active, max_new_tokens=max_new_tokens
        )
        rows.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "experiment_id": "trace_token_corruption",
                "condition": condition,
                "request_id": encoding.request_id,
                "model_label": encoding.model_label,
                "seed": encoding.seed,
                "split": encoding.split,
                "gold_count": encoding.count,
                **audit,
                **completion_metrics(result, gold_count=encoding.count),
            }
        )
    return rows


def paired_seed_effects(
    trials: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    outcome: str,
) -> pd.DataFrame:
    required = {"model_label", "seed", "condition", outcome}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Causal trials are missing {missing}")
    selected = trials.loc[trials["condition"].isin([treatment, control])].copy()
    selected[outcome] = pd.to_numeric(selected[outcome], errors="coerce")
    # Query-level effects are averaged within seed and condition first. This
    # preserves the registered query weighting while preventing thousands of
    # within-seed trace rows from masquerading as independent samples.
    seed_conditions = (
        selected.groupby(["model_label", "seed", "condition"], as_index=False)
        .agg(outcome_mean=(outcome, "mean"), n_query_rows=(outcome, "count"))
    )
    pivot = seed_conditions.pivot_table(
        index=["model_label", "seed"],
        columns="condition",
        values="outcome_mean",
        aggfunc="first",
    )
    if treatment not in pivot or control not in pivot:
        raise ValueError("Treatment/control pairing is incomplete")
    pivot = pivot.dropna(subset=[treatment, control]).reset_index()
    pivot["effect"] = pivot[treatment] - pivot[control]
    return pivot[["model_label", "seed", "effect"]].rename(
        columns={"effect": "mean_effect"}
    )


def bootstrap_seed_mean_ci(
    seed_effects: Sequence[float],
    *,
    samples: int,
    seed: int = 5,
) -> dict[str, float]:
    values = np.asarray(seed_effects, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Bootstrap needs finite seed effects")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(samples), len(values)), replace=True).mean(axis=1)
    return {
        "mean_effect": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_seeds": int(len(values)),
    }


def sign_flip_pvalue(seed_effects: Sequence[float]) -> float:
    values = np.asarray(seed_effects, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        raise ValueError("Sign-flip test needs finite seed effects")
    observed = abs(float(values.mean()))
    if n <= 20:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        permuted = np.abs((signs * values).mean(axis=1))
    else:
        rng = np.random.default_rng(544)
        signs = rng.choice((-1, 1), size=(200_000, n))
        permuted = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))


def analyze_paired_causal_results(
    trials_csv: str | Path,
    output_csv: str | Path,
    *,
    treatment: str,
    control: str,
    outcome: str,
    config: V5Config,
    mechanism: str | None = None,
    bank_size: int | None = None,
    transition_phase: str | None = None,
) -> pd.DataFrame:
    source = Path(trials_csv)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8") as handle:
            trials = pd.DataFrame(
                json.loads(line) for line in handle if line.strip()
            )
    else:
        trials = pd.read_csv(source)
    if "status" in trials:
        trials = trials.loc[trials["status"].astype(str).eq("ok")]
    filters = {
        "mechanism": mechanism,
        "bank_size": bank_size,
        "transition_phase": transition_phase,
    }
    for column, value in filters.items():
        if value is None:
            if column in trials and trials[column].dropna().astype(str).nunique() > 1:
                raise ValueError(
                    f"Causal analysis must select one {column}; found multiple"
                )
            continue
        if column not in trials:
            raise ValueError(f"Causal trials have no filter column {column}")
        if column == "bank_size":
            # JSONL exclusion rows omit bank_size, so pandas promotes the
            # otherwise integral column to float (for example 4 -> 4.0).
            # Numeric comparison avoids silently dropping every treatment.
            value_mask = pd.to_numeric(
                trials[column], errors="coerce"
            ).eq(float(value))
            # Clean head-ablation rows are evaluated once per mechanism and
            # therefore have K=0.  Retain that shared baseline when selecting
            # a registered treatment K; otherwise treatment-vs-clean analyses
            # are structurally impossible and only random-control contrasts
            # survive the filter.
            clean_mask = trials["condition"].astype(str).eq("clean")
            value_mask |= clean_mask
        else:
            value_mask = trials[column].astype(str).eq(str(value))
        trials = trials.loc[value_mask]
    if trials.empty:
        raise ValueError("No causal trial rows remain after registered filters")
    seed_frame = paired_seed_effects(
        trials,
        treatment=treatment,
        control=control,
        outcome=outcome,
    )
    rows: list[dict[str, Any]] = []
    for model_label, frame in seed_frame.groupby("model_label"):
        statistics = bootstrap_seed_mean_ci(
            frame["mean_effect"], samples=config.bootstrap_samples
        )
        rows.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "model_label": model_label,
                "treatment": treatment,
                "control": control,
                "outcome": outcome,
                **{key: value for key, value in filters.items() if value is not None},
                **statistics,
                "sign_flip_pvalue": sign_flip_pvalue(frame["mean_effect"]),
                "unit_of_inference": "seed",
            }
        )
    result = pd.DataFrame(rows)
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    return result
