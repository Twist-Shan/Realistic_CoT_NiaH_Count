from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "realistic_niah_v5_config_v1"
PRIMARY_TRACE_SITE = "item_end"
REGISTERED_TRACE_SITES = (
    "marker_end",
    "city_end",
    "item_end",
    "post_boundary",
    "list_cut",
    "answer_query",
)
REGISTERED_COHORTS = (
    "parser_hit",
    "one_to_one",
    "one_to_one_correct",
)
REGISTERED_CAUSAL_HEAD_MECHANISMS = (
    "targeted_retrieval",
    "progress_transition",
)


@dataclass(frozen=True)
class DecodingSpec:
    max_new_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


@dataclass(frozen=True)
class V5Config:
    schema_version: str = SCHEMA_VERSION
    design_variant: str = "v4.4"
    prompt_mode: str = "native_thinking"
    model_labels: tuple[str, ...] = ("Qwen3-8B", "Gemma4-E4B")
    counts: tuple[int, ...] = tuple(range(1, 11))
    discovery_seeds: tuple[int, ...] = tuple(range(1234, 1254))
    confirmation_seeds: tuple[int, ...] = tuple(range(1254, 1264))
    primary_trace_site: str = PRIMARY_TRACE_SITE
    sensitivity_trace_sites: tuple[str, ...] = (
        "marker_end",
        "city_end",
        "post_boundary",
        "list_cut",
        "answer_query",
    )
    primary_parser_cohort: str = "one_to_one"
    representation_n10_only: bool = True
    hidden_save_dtype: str = "float16"
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    bootstrap_samples: int = 5000
    causal_head_mechanisms: tuple[str, ...] = REGISTERED_CAUSAL_HEAD_MECHANISMS
    causal_head_selection_metric: str = (
        "query_weighted_mean_target_prompt_record_mass"
    )
    causal_head_bank_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    causal_random_controls: int = 3
    candidate_counts: tuple[int, ...] = tuple(range(1, 11))
    decoding: DecodingSpec = field(default_factory=DecodingSpec)

    @property
    def all_seeds(self) -> tuple[int, ...]:
        return self.discovery_seeds + self.confirmation_seeds

    @property
    def registered_sites(self) -> tuple[str, ...]:
        return (self.primary_trace_site,) + tuple(self.sensitivity_trace_sites)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported V5 config schema: {self.schema_version}")
        if self.design_variant != "v4.4":
            raise ValueError("V5 mechanism replication is fixed to V4.4 stimuli")
        if self.prompt_mode != "native_thinking":
            raise ValueError("V5 is registered only for native_thinking")
        if not self.model_labels or len(set(self.model_labels)) != len(self.model_labels):
            raise ValueError("model_labels must be non-empty and unique")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("V5 counts must be exactly 1 through 10")
        if not self.discovery_seeds or not self.confirmation_seeds:
            raise ValueError("Both discovery and confirmation seeds are required")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("Discovery and confirmation seeds must be disjoint")
        if self.primary_trace_site != PRIMARY_TRACE_SITE:
            raise ValueError(
                "The registered primary site is item_end; change requires a new schema"
            )
        sites = self.registered_sites
        if len(set(sites)) != len(sites):
            raise ValueError("Trace sites must be unique")
        unknown = sorted(set(sites) - set(REGISTERED_TRACE_SITES))
        if unknown:
            raise ValueError(f"Unknown trace sites: {unknown}")
        if self.primary_parser_cohort not in REGISTERED_COHORTS:
            raise ValueError("Unknown primary parser cohort")
        if self.hidden_save_dtype not in {"float16", "float32"}:
            raise ValueError("hidden_save_dtype must be float16 or float32")
        if not self.ridge_alphas or any(value <= 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.causal_head_mechanisms != REGISTERED_CAUSAL_HEAD_MECHANISMS:
            raise ValueError(
                "V5 causal heads must be targeted_retrieval and progress_transition"
            )
        if self.causal_head_selection_metric != (
            "query_weighted_mean_target_prompt_record_mass"
        ):
            raise ValueError("V5 causal head selection cannot use broad aggregation")
        if tuple(sorted(set(self.causal_head_bank_sizes))) != self.causal_head_bank_sizes:
            raise ValueError("causal_head_bank_sizes must be unique and increasing")
        if min(self.causal_head_bank_sizes) < 1:
            raise ValueError("causal_head_bank_sizes must be positive")
        if self.causal_random_controls < 1:
            raise ValueError("At least one random causal control is required")
        if len(set(self.candidate_counts)) != len(self.candidate_counts):
            raise ValueError("candidate_counts must be unique")
        if self.candidate_counts != tuple(range(1, 11)):
            raise ValueError("V5 causal candidate counts must be exactly 1 through 10")
        self.decoding.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "V5Config":
        payload = dict(value)
        decoding = DecodingSpec(**dict(payload.pop("decoding", {})))
        tuple_fields = {
            "model_labels",
            "counts",
            "discovery_seeds",
            "confirmation_seeds",
            "sensitivity_trace_sites",
            "ridge_alphas",
            "causal_head_bank_sizes",
            "causal_head_mechanisms",
            "candidate_counts",
        }
        for name in tuple_fields:
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(decoding=decoding, **payload)
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "V5Config":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
