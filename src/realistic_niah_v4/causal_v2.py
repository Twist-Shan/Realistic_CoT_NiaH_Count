from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .attention import Head


CAUSAL_V2_SCHEMA_VERSION = "realistic_niah_v4_causal_v2_design_v1"
CAUSAL_V2_VARIANT = "v4.4"
CAUSAL_V2_VALID_COUNTS = tuple(range(0, 11))
CAUSAL_V2_K_VALUES = (1, 2, 3, 4, 5)
CAUSAL_V2_CANONICAL_PAIRS = (
    (0, 1),
    (4, 5),
    (9, 10),
    (0, 2),
    (4, 6),
    (8, 10),
    (0, 3),
    (3, 6),
    (7, 10),
    (0, 4),
    (3, 7),
    (6, 10),
    (0, 5),
    (2, 7),
    (5, 10),
)
CAUSAL_V2_CENTROID_FIT_SEEDS = tuple(range(1234, 1254))
CAUSAL_V2_SCREEN_SEEDS = tuple(range(1254, 1259))
CAUSAL_V2_CONFIRMATION_SEEDS = tuple(range(1259, 1264))
CAUSAL_V2_ABLATION_COUNTS = (7, 8, 9, 10)
CAUSAL_V2_ABLATION_TOP_NS = tuple(range(1, 33))
CAUSAL_V2_PATCH_SITES = (
    "answer_query",
    "toggled_needle_end",
    "toggled_needle_span",
)
CAUSAL_V2_PATCH_PROTOCOLS = ("single_layer", "cumulative_from_layer")


def _directed_pairs(
    canonical_pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        directed
        for lower, upper in canonical_pairs
        for directed in ((int(lower), int(upper)), (int(upper), int(lower)))
    )


CAUSAL_V2_DIRECTED_PAIRS = _directed_pairs(CAUSAL_V2_CANONICAL_PAIRS)


@dataclass(frozen=True)
class CausalV2Design:
    """Frozen executable registry for the V4.4 causal-v2 study.

    The original V4 behavior/representation registry remains unchanged.  This
    extension adds an auxiliary zero-count member to the already nested V4.4
    families and reserves five confirmation seeds until discovery selection is
    written to an immutable manifest.
    """

    schema_version: str = CAUSAL_V2_SCHEMA_VERSION
    variant: str = CAUSAL_V2_VARIANT
    valid_counts: tuple[int, ...] = CAUSAL_V2_VALID_COUNTS
    k_values: tuple[int, ...] = CAUSAL_V2_K_VALUES
    canonical_pairs: tuple[tuple[int, int], ...] = CAUSAL_V2_CANONICAL_PAIRS
    centroid_fit_seeds: tuple[int, ...] = CAUSAL_V2_CENTROID_FIT_SEEDS
    screen_seeds: tuple[int, ...] = CAUSAL_V2_SCREEN_SEEDS
    confirmation_seeds: tuple[int, ...] = CAUSAL_V2_CONFIRMATION_SEEDS
    ablation_counts: tuple[int, ...] = CAUSAL_V2_ABLATION_COUNTS
    ablation_top_ns: tuple[int, ...] = CAUSAL_V2_ABLATION_TOP_NS
    ablation_random_replicates: int = 3
    ablation_scope: str = "answer_query"
    ablation_random_baseline: str = (
        "all_heads_in_matched_layers_without_replacement_overlap_allowed"
    )
    patch_sites: tuple[str, ...] = CAUSAL_V2_PATCH_SITES
    patch_protocols: tuple[str, ...] = CAUSAL_V2_PATCH_PROTOCOLS
    prompt_full_span_alignment: str = "exact_model_token_length_required"
    answer_multi_layer_protocol: str = "cumulative_clamp_L_to_final"
    transport_metric_version: str = (
        "signed_baseline_output_shift_over_semantic_target_shift_v1"
    )
    invalid_policy: str = "strict_zero_effect_plus_separate_invalid_rate"
    steering_alpha: float = 1.0
    steering_random_replicates: int = 1
    stability_min_positive_seeds: int = 4
    stability_min_anchor_pairs: int = 2
    stability_min_control_adjusted_transport: float = 0.15
    stability_min_valid_rate: float = 0.95
    confirmation_primary_evidence_scope: str = "held_out_only"
    confirmation_secondary_evidence_scope: str = "screen_plus_held_out"
    multiple_testing_correction: str = "holm_within_evidence_scope"
    bootstrap_repetitions: int = 10_000

    @property
    def directed_pairs(self) -> tuple[tuple[int, int], ...]:
        return _directed_pairs(self.canonical_pairs)

    @property
    def required_counts(self) -> tuple[int, ...]:
        return tuple(sorted({value for pair in self.canonical_pairs for value in pair}))

    def pairs_for_k(
        self, k: int, *, directed: bool = True
    ) -> tuple[tuple[int, int], ...]:
        k = int(k)
        pairs = tuple(
            pair for pair in self.canonical_pairs if int(pair[1]) - int(pair[0]) == k
        )
        if not pairs:
            raise KeyError(f"No registered causal-v2 pairs for k={k}")
        return _directed_pairs(pairs) if directed else pairs

    def validate(self) -> None:
        if self.schema_version != CAUSAL_V2_SCHEMA_VERSION:
            raise ValueError("Unexpected causal-v2 schema version")
        if self.variant != "v4.4":
            raise ValueError("Causal-v2 is intentionally restricted to v4.4")
        if self.valid_counts != tuple(range(0, 11)):
            raise ValueError("Causal-v2 valid counts must be exactly 0 through 10")
        if self.k_values != (1, 2, 3, 4, 5):
            raise ValueError("Causal-v2 k values must be exactly 1 through 5")
        if len(set(self.canonical_pairs)) != len(self.canonical_pairs):
            raise ValueError("Causal-v2 canonical pairs must be unique")
        if len(self.canonical_pairs) != 15:
            raise ValueError("Causal-v2 requires three anchor pairs for every k")
        for k in self.k_values:
            pairs = self.pairs_for_k(k, directed=False)
            if len(pairs) != 3:
                raise ValueError(f"Causal-v2 k={k} requires three anchors")
            if pairs[0][0] != 0 or pairs[-1][1] != 10:
                raise ValueError(f"Causal-v2 k={k} must cover low and high anchors")
        seed_sets = [
            set(self.centroid_fit_seeds),
            set(self.screen_seeds),
            set(self.confirmation_seeds),
        ]
        if any(not values for values in seed_sets):
            raise ValueError("All causal-v2 seed partitions must be nonempty")
        if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Causal-v2 seed partitions must be disjoint")
        if len(self.screen_seeds) != 5 or len(self.confirmation_seeds) != 5:
            raise ValueError(
                "Causal-v2 screen and confirmation each require five seeds"
            )
        if self.ablation_top_ns != tuple(range(1, 33)):
            raise ValueError("Causal-v2 ablation must sweep every top-k from 1 to 32")
        if self.ablation_random_replicates < 1:
            raise ValueError("Ablation needs at least one matched-random replicate")
        if self.ablation_scope != "answer_query":
            raise ValueError("Causal-v2 ablation is answer-query-only")
        if self.ablation_random_baseline != (
            "all_heads_in_matched_layers_without_replacement_overlap_allowed"
        ):
            raise ValueError("Unexpected causal-v2 random-ablation baseline")
        if self.patch_sites != CAUSAL_V2_PATCH_SITES:
            raise ValueError("Causal-v2 patch sites are not the frozen registry")
        if self.patch_protocols != CAUSAL_V2_PATCH_PROTOCOLS:
            raise ValueError("Causal-v2 patch protocols are not the frozen registry")
        if self.prompt_full_span_alignment != "exact_model_token_length_required":
            raise ValueError("Unexpected prompt full-span alignment policy")
        if self.answer_multi_layer_protocol != "cumulative_clamp_L_to_final":
            raise ValueError("Unexpected answer multi-layer patch definition")
        if self.transport_metric_version != (
            "signed_baseline_output_shift_over_semantic_target_shift_v1"
        ):
            raise ValueError("Unexpected causal-v2 transport metric")
        if self.invalid_policy != "strict_zero_effect_plus_separate_invalid_rate":
            raise ValueError("Unexpected causal-v2 invalid-output policy")
        if not 0.0 < float(self.steering_alpha):
            raise ValueError("Steering alpha must be positive")
        if self.steering_random_replicates < 1:
            raise ValueError("Steering needs a norm-matched random control")
        if not 1 <= self.stability_min_positive_seeds <= len(self.screen_seeds):
            raise ValueError("Invalid positive-seed stability threshold")
        if not 1 <= self.stability_min_anchor_pairs <= 3:
            raise ValueError("Invalid anchor-pair stability threshold")
        if not 0.0 <= self.stability_min_valid_rate <= 1.0:
            raise ValueError("Invalid stability valid-rate threshold")
        if self.confirmation_primary_evidence_scope != "held_out_only":
            raise ValueError("Primary causal-v2 confirmation must be held-out-only")
        if self.confirmation_secondary_evidence_scope != "screen_plus_held_out":
            raise ValueError("Unexpected secondary causal-v2 evidence scope")
        if self.multiple_testing_correction != "holm_within_evidence_scope":
            raise ValueError("Unexpected causal-v2 multiple-testing correction")
        if self.bootstrap_repetitions < 100:
            raise ValueError("Causal-v2 bootstrap must use at least 100 repetitions")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CausalV2Design":
        values = dict(payload)
        tuple_fields = {
            "valid_counts",
            "k_values",
            "centroid_fit_seeds",
            "screen_seeds",
            "confirmation_seeds",
            "ablation_counts",
            "ablation_top_ns",
            "patch_sites",
            "patch_protocols",
        }
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        if "canonical_pairs" in values:
            values["canonical_pairs"] = tuple(
                tuple(int(item) for item in pair) for pair in values["canonical_pairs"]
            )
        design = cls(**values)
        design.validate()
        return design

    @classmethod
    def from_json(cls, path: str | Path) -> "CausalV2Design":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_transport_metrics(
    *,
    baseline_prediction: int | None,
    intervened_prediction: int | None,
    receiver_count: int,
    target_count: int,
) -> dict[str, Any]:
    """Return the preregistered signed transport and target-conformity metrics.

    ``normalized_transport`` measures the generated change relative to the
    semantic count displacement.  ``target_conformity`` instead measures final
    distance to the target.  Invalid generations remain missing in the raw
    metrics and are assigned zero in the strict failure-aware counterparts.
    Values are deliberately not clipped: negative transport is a reversal and
    conformity below zero means the intervention missed by more than the full
    receiver-to-target gap.
    """

    receiver_count = int(receiver_count)
    target_count = int(target_count)
    expected_shift = target_count - receiver_count
    if expected_shift == 0:
        raise ValueError(
            "Transport metrics require distinct receiver and target counts"
        )
    valid = baseline_prediction is not None and intervened_prediction is not None
    if valid:
        generated_shift = int(intervened_prediction) - int(baseline_prediction)
        transport = float(generated_shift) / float(expected_shift)
        conformity = 1.0 - (
            abs(int(intervened_prediction) - target_count) / abs(expected_shift)
        )
        baseline_target_distance = abs(int(baseline_prediction) - target_count)
        target_distance_reduction = (
            float(baseline_target_distance)
            - abs(int(intervened_prediction) - target_count)
        ) / abs(expected_shift)
    else:
        generated_shift = math.nan
        transport = math.nan
        conformity = math.nan
        target_distance_reduction = math.nan
    return {
        "k": abs(expected_shift),
        "target_direction": "increase" if expected_shift > 0 else "decrease",
        "expected_count_shift": expected_shift,
        "normalized_transport": transport,
        "target_conformity": conformity,
        "target_distance_reduction_normalized": target_distance_reduction,
        "strict_normalized_transport": transport if valid else 0.0,
        "strict_target_conformity": conformity if valid else 0.0,
        "strict_target_hit": bool(
            intervened_prediction is not None
            and int(intervened_prediction) == target_count
        ),
        "transport_numeric_valid": bool(valid),
        "transport_generated_shift": generated_shift,
    }


def _json_float_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError("Needle span masses must be a finite nonempty vector")
    return result


def head_phenotype_scores(
    attention_detail: pd.DataFrame,
    *,
    variant: str = CAUSAL_V2_VARIANT,
    split: str = "discovery",
) -> pd.DataFrame:
    """Score full-span broad aggregation and first-needle localization heads."""

    required = {
        "model_label",
        "design_variant",
        "split",
        "seed",
        "count",
        "layer",
        "head",
        "broad_mass",
        "broad_coverage",
        "broad_primary",
        "needle_span_masses",
    }
    missing = sorted(required - set(attention_detail.columns))
    if missing:
        raise ValueError(f"Attention detail is missing phenotype columns: {missing}")
    frame = attention_detail[
        attention_detail["design_variant"].astype(str).eq(str(variant))
        & attention_detail["split"].astype(str).eq(str(split))
    ].copy()
    if frame.empty:
        raise ValueError(f"No {variant}/{split} attention rows for phenotype ranking")
    first_mass: list[float] = []
    other_mean: list[float] = []
    locator_score: list[float] = []
    for value in frame["needle_span_masses"]:
        masses = _json_float_array(value)
        first = float(masses[0])
        others = float(masses[1:].mean()) if len(masses) > 1 else math.nan
        first_mass.append(first)
        other_mean.append(others)
        locator_score.append(first - others if np.isfinite(others) else math.nan)
    frame["first_needle_span_mass"] = first_mass
    frame["other_needle_span_mass_mean"] = other_mean
    frame["first_locator_score"] = locator_score
    group_columns = ["model_label", "design_variant", "layer", "head"]
    summary = (
        frame.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            discovery_rows=("seed", "size"),
            discovery_seeds=("seed", "nunique"),
            discovery_counts=("count", "nunique"),
            broad_span_mass=("broad_mass", "mean"),
            broad_uniformity=("broad_coverage", "mean"),
            broad_aggregation_score=("broad_primary", "mean"),
            first_needle_span_mass=("first_needle_span_mass", "mean"),
            other_needle_span_mass_mean=("other_needle_span_mass_mean", "mean"),
            first_locator_rows=("first_locator_score", "count"),
            first_locator_score=("first_locator_score", "mean"),
        )
        .sort_values(["model_label", "layer", "head"])
        .reset_index(drop=True)
    )
    return summary


def rank_head_phenotypes(
    scores: pd.DataFrame,
    *,
    top_n: int = 32,
) -> dict[str, list[Head]]:
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    models = sorted(str(value) for value in scores["model_label"].unique())
    if len(models) != 1:
        raise ValueError("Head phenotype ranking expects exactly one model")
    result: dict[str, list[Head]] = {}
    definitions = {
        "broad_aggregation": (
            ["broad_aggregation_score", "broad_span_mass", "layer", "head"],
            [False, False, True, True],
        ),
        "first_locator": (
            ["first_locator_score", "first_needle_span_mass", "layer", "head"],
            [False, False, True, True],
        ),
    }
    for name, (columns, ascending) in definitions.items():
        ranked = scores.dropna(subset=[columns[0]]).sort_values(
            columns, ascending=ascending
        )
        if len(ranked) < int(top_n):
            raise ValueError(
                f"Only {len(ranked)} heads available for {name} top-{top_n}"
            )
        result[name] = [
            (int(row.layer), int(row.head))
            for row in ranked.head(int(top_n)).itertuples(index=False)
        ]
    return result


def write_head_phenotype_registry(
    scores: pd.DataFrame,
    *,
    output_dir: str | Path,
    top_n: int = 32,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rankings = rank_head_phenotypes(scores, top_n=top_n)
    scores_path = output / "head_phenotype_scores.csv"
    registry_path = output / "head_phenotype_rankings.json"
    scores_temporary = scores_path.with_suffix(scores_path.suffix + ".tmp")
    scores.to_csv(scores_temporary, index=False)
    scores_temporary.replace(scores_path)
    payload = {
        "schema_version": "realistic_niah_v4_causal_v2_head_rankings_v1",
        "selection_split": "discovery",
        "design_variant": CAUSAL_V2_VARIANT,
        "model_label": str(scores["model_label"].iloc[0]),
        "mass_definition": "sum of answer-query attention over every token in each active needle span",
        "broad_aggregation_definition": "mean(broad_mass * exp(entropy(per-needle mass))/needle_count)",
        "first_locator_definition": "mean(first needle span mass - mean(other needle span masses))",
        "rankings": {
            name: [
                {"rank": rank, "layer": layer, "head": head}
                for rank, (layer, head) in enumerate(heads, start=1)
            ]
            for name, heads in rankings.items()
        },
    }
    registry_temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    registry_temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registry_temporary.replace(registry_path)
    return {"scores": scores_path, "rankings": registry_path}


def load_head_phenotype_registry(path: str | Path) -> dict[str, list[Head]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("selection_split") != "discovery":
        raise ValueError("Causal-v2 head rankings must be discovery-only")
    if payload.get("design_variant") != CAUSAL_V2_VARIANT:
        raise ValueError("Causal-v2 head rankings must be for v4.4")
    rankings = payload.get("rankings")
    if not isinstance(rankings, dict):
        raise ValueError("Head phenotype registry has no rankings")
    result: dict[str, list[Head]] = {}
    for name in ("broad_aggregation", "first_locator"):
        rows = rankings.get(name)
        if not isinstance(rows, list) or len(rows) < 32:
            raise ValueError(f"Head phenotype registry lacks {name} top-32")
        result[name] = [(int(row["layer"]), int(row["head"])) for row in rows]
    return result


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _condition_columns(detail: pd.DataFrame, family: str) -> tuple[list[str], str, str]:
    if family in {"prompt_patching", "answer_patching"}:
        columns = ["model_label", "site", "patch_protocol", "start_layer", "k"]
        return columns, "donor_transport", "condition"
    if family == "steering":
        columns = ["model_label", "steering_protocol", "layer_set", "k"]
        return columns, "geometric", "condition"
    raise KeyError(f"Unknown causal-v2 stability family: {family}")


def stable_layer_k_conditions(
    detail: pd.DataFrame,
    *,
    family: str,
    design: CausalV2Design | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen five-seed stability screen and writeable manifest data."""

    design = design or CausalV2Design()
    design.validate()
    condition_columns, treatment_name, condition_column = _condition_columns(
        detail, family
    )
    required = {
        *condition_columns,
        condition_column,
        "seed",
        "receiver_count",
        "target_direction",
        "strict_normalized_transport",
        "patched_format_valid",
        "transport_numeric_valid",
    }
    if family in {"prompt_patching", "answer_patching"}:
        required.add("donor_count")
    else:
        required.add("target_count")
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"{family} screen is missing columns: {missing}")
    work = detail.copy()
    if "status" in work.columns:
        failed = work[work["status"].astype(str).ne("ok")]
        if not failed.empty:
            reasons = (
                failed.get("skip_reason", pd.Series("unknown", index=failed.index))
                .astype(str)
                .value_counts()
                .to_dict()
            )
            raise ValueError(
                f"{family} stability cannot select from skipped interventions: "
                f"{reasons}"
            )
        work = work[work["status"].astype(str).eq("ok")].copy()
    target_count_column = "donor_count" if "donor_count" in work else "target_count"
    work["anchor_pair"] = work.apply(
        lambda row: f"{min(int(row['receiver_count']), int(row[target_count_column]))}:"
        f"{max(int(row['receiver_count']), int(row[target_count_column]))}",
        axis=1,
    )
    identity = [
        *condition_columns,
        "seed",
        "receiver_count",
        target_count_column,
        "target_direction",
        "anchor_pair",
    ]
    treatment = work[work[condition_column].astype(str).eq(treatment_name)].copy()
    controls = work[~work[condition_column].astype(str).eq(treatment_name)].copy()
    if treatment.empty or controls.empty:
        raise ValueError(f"{family} stability requires treatment and matched controls")
    control_mean = controls.groupby(identity, as_index=False, dropna=False).agg(
        control_transport=("strict_normalized_transport", "mean"),
        control_valid_rate=("transport_numeric_valid", "mean"),
    )
    paired = treatment.merge(
        control_mean, on=identity, how="inner", validate="many_to_one"
    )
    if len(paired) != len(treatment):
        raise ValueError(f"{family} screen has incomplete matched controls")
    paired["control_adjusted_transport"] = pd.to_numeric(
        paired["strict_normalized_transport"], errors="raise"
    ) - pd.to_numeric(paired["control_transport"], errors="raise")
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(condition_columns, sort=True, dropna=False):
        metadata = {
            name: _json_scalar(value) for name, value in zip(condition_columns, keys)
        }
        seed_means = frame.groupby("seed")["control_adjusted_transport"].mean()
        direction_means = frame.groupby("target_direction")[
            "control_adjusted_transport"
        ].mean()
        anchor_means = frame.groupby("anchor_pair")["control_adjusted_transport"].mean()
        valid_rate = float(frame["transport_numeric_valid"].astype(bool).mean())
        patched_valid_rate = float(frame["patched_format_valid"].astype(bool).mean())
        positive_seeds = int((seed_means > 0).sum())
        positive_anchors = int((anchor_means > 0).sum())
        both_directions_positive = bool(
            {"increase", "decrease"}.issubset(set(direction_means.index))
            and (direction_means.loc[["increase", "decrease"]] > 0).all()
        )
        mean_effect = float(seed_means.mean())
        stable = bool(
            len(seed_means) == len(design.screen_seeds)
            and positive_seeds >= design.stability_min_positive_seeds
            and positive_anchors >= design.stability_min_anchor_pairs
            and both_directions_positive
            and mean_effect >= design.stability_min_control_adjusted_transport
            and valid_rate >= design.stability_min_valid_rate
        )
        row = {
            **metadata,
            "screen_seeds": int(len(seed_means)),
            "positive_screen_seeds": positive_seeds,
            "positive_anchor_pairs": positive_anchors,
            "both_directions_positive": both_directions_positive,
            "mean_control_adjusted_transport": mean_effect,
            "transport_numeric_valid_rate": valid_rate,
            "patched_valid_rate": patched_valid_rate,
            "stable": stable,
        }
        rows.append(row)
        if stable:
            selected.append(row)
    scores = pd.DataFrame(rows).sort_values(condition_columns).reset_index(drop=True)
    manifest = {
        "schema_version": "realistic_niah_v4_causal_v2_stability_selection_v1",
        "family": family,
        "selection_split": "screen",
        "screen_seeds": list(design.screen_seeds),
        "held_out_confirmation_seeds": list(design.confirmation_seeds),
        "thresholds": {
            "minimum_positive_seeds": design.stability_min_positive_seeds,
            "minimum_positive_anchor_pairs": design.stability_min_anchor_pairs,
            "minimum_control_adjusted_transport": design.stability_min_control_adjusted_transport,
            "minimum_valid_rate": design.stability_min_valid_rate,
            "both_directions_must_be_positive": True,
        },
        "confirmation_analysis": {
            "primary_evidence_scope": design.confirmation_primary_evidence_scope,
            "secondary_evidence_scope": design.confirmation_secondary_evidence_scope,
            "multiple_testing_correction": design.multiple_testing_correction,
        },
        "selected": selected,
        "selected_condition_count": len(selected),
    }
    return scores, manifest


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    observed = abs(float(values.mean()))
    if len(values) <= 16:
        masks = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = np.where(bits == 0, -1.0, 1.0)
        distribution = np.abs((signs * values[None, :]).mean(axis=1))
        return float(np.mean(distribution >= observed - 1e-15))
    else:
        rng = np.random.default_rng(_stable_seed(f"sign-flip:{values.tolist()}"))
        signs = rng.choice((-1.0, 1.0), size=(100_000, len(values)))
        distribution = np.abs((signs * values[None, :]).mean(axis=1))
    return float(
        (np.sum(distribution >= observed - 1e-15) + 1) / (len(distribution) + 1)
    )


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.where(np.isfinite(values))[0]
    if len(finite_indices) == 0:
        return result.tolist()
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, float(values[index]) * (m - rank))
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def confirmation_statistics(
    detail: pd.DataFrame,
    *,
    family: str,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    """Estimate confirmed treatment-minus-control transport by seed cluster."""

    condition_columns, treatment_name, condition_column = _condition_columns(
        detail, family
    )
    work = detail.copy()
    if "status" in work.columns:
        failed = work[work["status"].astype(str).ne("ok")]
        if not failed.empty:
            raise ValueError("Confirmation detail contains skipped interventions")
        work = work[work["status"].astype(str).eq("ok")].copy()
    target_count_column = "donor_count" if "donor_count" in detail else "target_count"
    identity = [
        *condition_columns,
        "seed",
        "receiver_count",
        target_count_column,
        "target_direction",
    ]
    if "evidence_split" in work.columns:
        identity.append("evidence_split")
    treatment = work[work[condition_column].astype(str).eq(treatment_name)].copy()
    controls = work[~work[condition_column].astype(str).eq(treatment_name)].copy()
    control = controls.groupby(identity, as_index=False, dropna=False).agg(
        control_transport=("strict_normalized_transport", "mean")
    )
    paired = treatment.merge(control, on=identity, how="inner", validate="many_to_one")
    if len(paired) != len(treatment):
        raise ValueError("Confirmation detail has incomplete matched controls")
    paired["effect"] = pd.to_numeric(
        paired["strict_normalized_transport"], errors="raise"
    ) - pd.to_numeric(paired["control_transport"], errors="raise")
    rows: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(condition_columns, sort=True, dropna=False):
        scopes: list[tuple[str, pd.DataFrame]] = []
        if "evidence_split" in frame.columns:
            held_out = frame[
                frame["evidence_split"].astype(str).eq("held_out_confirmation")
            ]
            if not held_out.empty:
                scopes.append(("held_out_only", held_out))
            screen = frame[frame["evidence_split"].astype(str).eq("screen")]
            if not screen.empty and not held_out.empty:
                scopes.append(("screen_plus_held_out", frame))
        else:
            scopes.append(("all_supplied_seeds", frame))
        for evidence_scope, evidence in scopes:
            seed_means = evidence.groupby("seed")["effect"].mean().to_numpy(dtype=float)
            if len(seed_means) == 0:
                continue
            rng = np.random.default_rng(
                _stable_seed(
                    "confirmation:"
                    + ":".join(str(value) for value in keys)
                    + f":{evidence_scope}"
                )
            )
            indices = rng.integers(
                0,
                len(seed_means),
                size=(int(bootstrap_repetitions), len(seed_means)),
            )
            distribution = seed_means[indices].mean(axis=1)
            low, high = np.quantile(distribution, [0.025, 0.975])
            split_counts: dict[str, int] = {}
            if "evidence_split" in evidence.columns:
                split_counts = {
                    str(split): int(split_frame["seed"].nunique())
                    for split, split_frame in evidence.groupby("evidence_split")
                }
            rows.append(
                {
                    **dict(zip(condition_columns, keys)),
                    "seeds": int(len(seed_means)),
                    "screen_seeds": int(split_counts.get("screen", 0)),
                    "held_out_confirmation_seeds": int(
                        split_counts.get("held_out_confirmation", 0)
                    ),
                    "evidence_scope": evidence_scope,
                    "is_primary_confirmation": evidence_scope == "held_out_only",
                    "mean_control_adjusted_transport": float(seed_means.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "exact_sign_flip_p": _exact_sign_flip_p(seed_means),
                    "positive_seed_fraction": float((seed_means > 0).mean()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_p"] = math.nan
    for _scope, indices in result.groupby("evidence_scope", sort=False).groups.items():
        result.loc[indices, "holm_p"] = _holm_adjust(
            result.loc[indices, "exact_sign_flip_p"].tolist()
        )
    return result.sort_values([*condition_columns, "evidence_scope"]).reset_index(
        drop=True
    )


CausalV2Design().validate()
