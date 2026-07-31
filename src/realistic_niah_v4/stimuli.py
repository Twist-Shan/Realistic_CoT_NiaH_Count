from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dataset_generation.dynamic_niah import TokenizerAdapter
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    generate_dynamic_niah_dataset_v2,
)
from realistic_niah.stimuli import CITY_SCORE_RE

from .spec import (
    DESIGN_VARIANT_CONTROLS,
    MANIFEST_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    STIMULUS_SCHEMA_VERSION,
    V4Config,
)


@dataclass(frozen=True)
class ControlledFreezeSpec:
    """Inputs that are external to the preregistered V4 analysis config."""

    config: V4Config = V4Config()
    haystack_dir: str = "data/haystacks/paul_graham"
    haystack_source_mode: str = "multi_file_no_repeat"
    entities_path: str = "data/entities/cities.csv"
    fact_templates_path: str = "data/templates/niah_fact_single_template.txt"
    tokenizer_cache_dir: str | None = None
    minimum_filler_tokens: int = 128
    hard_negative_gap_tokens: int = 8

    def validate(self) -> None:
        self.config.validate()
        if self.minimum_filler_tokens <= 0:
            raise ValueError("minimum_filler_tokens must be positive")
        if self.hard_negative_gap_tokens < 0:
            raise ValueError("hard_negative_gap_tokens must be non-negative")
        if self.haystack_source_mode != "multi_file_no_repeat":
            raise ValueError("Registered V4 requires multi_file_no_repeat haystacks")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(f"{label}:{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _catalog_fingerprint(needles: Sequence[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            _sha256_text(str(item["decoded_text"])),
            str(item["record"]["city"]),
            int(item["record"]["score"]),
            int(item["token_length"]),
        )
        for item in needles
    )


def _catalog_metadata(
    needles: Sequence[dict[str, Any]],
    *,
    permutation: Sequence[int],
) -> list[dict[str, Any]]:
    if len(needles) != len(permutation):
        raise ValueError("Needle/permutation length mismatch")
    output: list[dict[str, Any]] = []
    for slot_index, (item, source_index) in enumerate(
        zip(needles, permutation), start=1
    ):
        text = str(item["decoded_text"])
        output.append(
            {
                "slot_index": slot_index,
                "source_catalog_index": int(source_index) + 1,
                "needle_id": f"N{slot_index}",
                "source_needle_id": str(item["needle_id"]),
                "text": text,
                "city": str(item["record"]["city"]),
                "score": int(item["record"]["score"]),
                "canonical_token_length": int(item["token_length"]),
                "text_sha256": _sha256_text(text),
            }
        )
    return output


def _generator_config(
    *,
    freeze_spec: ControlledFreezeSpec,
    target_filler_tokens: int,
    content_seed: int,
    seed: int,
    insertion_positions: Sequence[int],
) -> DynamicNiahV2Config:
    config = freeze_spec.config
    slot_count = max(config.needle_counts)
    if len(insertion_positions) != slot_count:
        raise ValueError("Expected one insertion position per V4 slot")
    return DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name=config.canonical_tokenizer,
        num_examples=1,
        target_haystack_tokens=int(target_filler_tokens),
        num_needles=slot_count,
        insertion_positions=tuple(int(value) for value in insertion_positions),
        randomize_needle_insertion=False,
        sentence_level_insertion=False,
        word_level_insertion=False,
        prompt_style="vanilla_no_cue",
        global_random_seed=int(seed),
        haystack_seed=int(seed),
        haystack_source_mode=freeze_spec.haystack_source_mode,
        needle_seed=int(content_seed),
        control_switch=[True] * slot_count,
        haystack_dir=freeze_spec.haystack_dir,
        entities_path=freeze_spec.entities_path,
        fact_templates_path=freeze_spec.fact_templates_path,
    )


def _scaffold(
    *,
    tokenizer: TokenizerAdapter,
    freeze_spec: ControlledFreezeSpec,
    target_filler_tokens: int,
    content_seed: int,
    seed: int,
    insertion_positions: Sequence[int],
) -> dict[str, Any]:
    row = generate_dynamic_niah_dataset_v2(
        _generator_config(
            freeze_spec=freeze_spec,
            target_filler_tokens=target_filler_tokens,
            content_seed=content_seed,
            seed=seed,
            insertion_positions=insertion_positions,
        ),
        tokenizer_adapter=tokenizer,
    )[0]
    slot_count = max(freeze_spec.config.needle_counts)
    if len(row["needles"]) != slot_count:
        raise RuntimeError("V4 scaffold returned the wrong number of needles")
    if not all(bool(item["is_control"]) for item in row["needles"]):
        raise RuntimeError("V4 scaffold must use controls in every slot")
    for item in row["needles"]:
        if len(item["tokens"]) != len(item["inserted_tokens"]):
            raise RuntimeError("V4 scaffold produced a non-length-matched control")
    return row


def _pilot_needles(
    *,
    tokenizer: TokenizerAdapter,
    freeze_spec: ControlledFreezeSpec,
    content_seed: int,
    seed: int,
) -> list[dict[str, Any]]:
    slot_count = max(freeze_spec.config.needle_counts)
    pilot_filler = max(
        freeze_spec.minimum_filler_tokens,
        min(1_024, freeze_spec.config.target_passage_tokens // 2),
    )
    row = _scaffold(
        tokenizer=tokenizer,
        freeze_spec=freeze_spec,
        target_filler_tokens=pilot_filler,
        content_seed=content_seed,
        seed=seed,
        insertion_positions=(0,) * slot_count,
    )
    return list(row["needles"])


def _fixed_final_starts(
    *,
    target_tokens: int,
    depths: Iterable[float],
) -> tuple[int, ...]:
    return tuple(int(round(float(depth) * int(target_tokens))) for depth in depths)


def _randomized_final_starts(
    *,
    target_tokens: int,
    slot_count: int,
    seed: int,
    minimum_separation: int,
) -> tuple[int, ...]:
    """Sample one seed-paired position schedule shared by v4.2-v4.4."""

    lower = int(round(0.05 * int(target_tokens)))
    # Reserve one separation unit as a width-independent right-edge guard.
    upper = int(round(0.95 * int(target_tokens))) - int(minimum_separation)
    if upper <= lower:
        raise ValueError("Target is too short for randomized V4 positions")
    rng = random.Random(_stable_seed("v4_position_schedule", seed))
    population = range(lower, upper + 1)
    if len(population) < slot_count:
        raise ValueError("Not enough candidate V4 positions")
    for _ in range(20_000):
        starts = tuple(sorted(rng.sample(population, k=int(slot_count))))
        if all(
            right - left >= int(minimum_separation)
            for left, right in zip(starts, starts[1:])
        ):
            return starts
    raise RuntimeError("Failed to sample a feasible V4 position schedule")


def _content_permutation(slot_count: int, seed: int) -> tuple[int, ...]:
    indices = list(range(int(slot_count)))
    rng = random.Random(_stable_seed("v4_content_order", seed))
    rng.shuffle(indices)
    if indices == list(range(int(slot_count))):
        indices = indices[1:] + indices[:1]
    return tuple(indices)


def _content_seed(
    *,
    variant: str,
    seed: int,
    fixed_seed: int,
) -> int:
    if DESIGN_VARIANT_CONTROLS[variant]["city_score_content_fixed_across_seeds"]:
        return int(fixed_seed)
    return _stable_seed("v4_city_score_content", seed)


def _design_for_family(
    *,
    variant: str,
    seed: int,
    config: V4Config,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    controls = DESIGN_VARIANT_CONTROLS[variant]
    slot_count = max(config.needle_counts)
    content_seed = _content_seed(
        variant=variant,
        seed=seed,
        fixed_seed=config.fixed_needle_seed,
    )
    if controls["positions_fixed_across_seeds"]:
        starts = _fixed_final_starts(
            target_tokens=config.target_passage_tokens,
            depths=config.fixed_slot_depths,
        )
    else:
        starts = _randomized_final_starts(
            target_tokens=config.target_passage_tokens,
            slot_count=slot_count,
            seed=seed,
            minimum_separation=config.randomized_position_min_separation_tokens,
        )
    if controls["city_score_order_fixed_across_seeds"]:
        permutation = tuple(range(slot_count))
    else:
        permutation = _content_permutation(slot_count, seed)
    return content_seed, starts, permutation


def _filler_positions(
    *,
    final_starts: Sequence[int],
    widths: Sequence[int],
    target_tokens: int,
    minimum_separation: int,
) -> tuple[int, tuple[int, ...]]:
    if len(final_starts) != len(widths):
        raise ValueError("Final-start/width length mismatch")
    if any(int(width) <= 0 for width in widths):
        raise ValueError("Every V4 slot width must be positive")
    if any(int(width) >= int(minimum_separation) for width in widths):
        raise ValueError(
            "A V4 city-score span is wider than the preregistered "
            "position-separation guard"
        )
    filler_tokens = int(target_tokens) - sum(int(width) for width in widths)
    cumulative = 0
    positions: list[int] = []
    prior_end = -1
    for start, width in zip(final_starts, widths):
        start = int(start)
        width = int(width)
        if start <= prior_end:
            raise ValueError("V4 final slot spans overlap")
        if start < 0 or start + width > int(target_tokens):
            raise ValueError("A V4 final slot span lies outside the passage")
        positions.append(start - cumulative)
        cumulative += width
        prior_end = start + width - 1
    if any(position < 0 or position > filler_tokens for position in positions):
        raise ValueError("A V4 filler-coordinate position is out of range")
    if any(right < left for left, right in zip(positions, positions[1:])):
        raise ValueError("V4 filler-coordinate positions are not monotone")
    return filler_tokens, tuple(positions)


def _assemble_tokens(
    *,
    base_tokens: Sequence[Any],
    ordered_needles: Sequence[dict[str, Any]],
    active_count: int,
    filler_positions: Sequence[int],
) -> tuple[list[Any], list[dict[str, Any]]]:
    if len(ordered_needles) != len(filler_positions):
        raise ValueError("Needle/position length mismatch")
    out = list(base_tokens)
    shift = 0
    realized: list[dict[str, Any]] = []
    for slot_index, (position, item) in enumerate(
        zip(filler_positions, ordered_needles), start=1
    ):
        active = slot_index <= int(active_count)
        inserted = list(item["tokens"] if active else item["inserted_tokens"])
        width = int(item["token_length"])
        if len(inserted) != width:
            raise RuntimeError("V4 active/control slot width changed")
        final_start = int(position) + shift
        out[final_start:final_start] = inserted
        realized.append(
            {
                "slot_index": slot_index,
                "final_start": final_start,
                "final_end": final_start + width,
                "active": active,
                "inserted_tokens": inserted,
                "control": item.get("control"),
            }
        )
        shift += width
    return out, realized


def _token_char_span(
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    if not 0 <= start < end <= len(offsets):
        raise ValueError(f"Invalid token span [{start}, {end})")
    nonempty = [
        (left, right) for left, right in offsets[start:end] if int(right) > int(left)
    ]
    if not nonempty:
        raise ValueError(f"Token span [{start}, {end}) has no character extent")
    return int(nonempty[0][0]), int(nonempty[-1][1])


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _hard_negative_spans(
    *,
    passage: str,
    offsets: Sequence[tuple[int, int]],
    slot_spans: Sequence[tuple[int, int]],
    gap: int,
) -> list[dict[str, Any]]:
    occupied = list(slot_spans)
    selected: list[tuple[int, int]] = []
    output: list[dict[str, Any]] = []
    sequence_length = len(offsets)
    for slot_index, (slot_start, slot_end) in enumerate(slot_spans, start=1):
        width = slot_end - slot_start
        preferred = max(0, slot_start - int(gap) - width)
        candidates = list(range(preferred, -1, -1)) + list(
            range(preferred + 1, sequence_length - width + 1)
        )
        chosen: tuple[int, int] | None = None
        for start in candidates:
            candidate = (int(start), int(start) + width)
            if any(_overlaps(candidate, interval) for interval in occupied):
                continue
            if any(_overlaps(candidate, interval) for interval in selected):
                continue
            char_start, char_end = _token_char_span(offsets, candidate[0], candidate[1])
            if CITY_SCORE_RE.search(passage[char_start:char_end]):
                continue
            chosen = candidate
            break
        if chosen is None:
            raise RuntimeError(f"Unable to find a hard negative for slot {slot_index}")
        selected.append(chosen)
        char_start, char_end = _token_char_span(offsets, chosen[0], chosen[1])
        output.append(
            {
                "slot_index": slot_index,
                "canonical_span_start": chosen[0],
                "canonical_span_end": chosen[1],
                "canonical_token_length": width,
                "char_start": char_start,
                "char_end": char_end,
                "text": passage[char_start:char_end],
            }
        )
    return output


def _compact_stimulus(
    *,
    tokenizer: TokenizerAdapter,
    freeze_spec: ControlledFreezeSpec,
    variant: str,
    seed: int,
    active_count: int,
    catalog: Sequence[dict[str, Any]],
    ordered_needles: Sequence[dict[str, Any]],
    base_tokens: Sequence[Any],
    haystack_metadata: dict[str, Any],
    final_starts: Sequence[int],
    filler_positions: Sequence[int],
    permutation: Sequence[int],
    content_seed: int,
) -> dict[str, Any]:
    config = freeze_spec.config
    final_tokens, realized = _assemble_tokens(
        base_tokens=base_tokens,
        ordered_needles=ordered_needles,
        active_count=active_count,
        filler_positions=filler_positions,
    )
    passage = tokenizer.decode(final_tokens)
    actual_ids, offsets = tokenizer.encode_with_offsets(passage)
    if list(actual_ids) != list(final_tokens):
        mismatch = next(
            (
                index
                for index, (expected, actual) in enumerate(
                    zip(final_tokens, actual_ids)
                )
                if expected != actual
            ),
            min(len(final_tokens), len(actual_ids)),
        )
        window_start = max(0, mismatch - 4)
        window_end = mismatch + 5
        raise RuntimeError(
            "Canonical decode/re-encode changed token identities; refusing "
            "approximate V4 alignment. "
            f"variant={variant}, seed={seed}, count={active_count}, "
            f"first_mismatch={mismatch}, expected_length={len(final_tokens)}, "
            f"actual_length={len(actual_ids)}, "
            f"expected_ids={list(final_tokens[window_start:window_end])}, "
            f"actual_ids={list(actual_ids[window_start:window_end])}, "
            f"expected_window={tokenizer.decode(list(final_tokens[window_start:window_end]))!r}, "
            f"actual_window={tokenizer.decode(list(actual_ids[window_start:window_end]))!r}"
        )
    if len(actual_ids) != config.target_passage_tokens:
        raise RuntimeError(
            f"V4 passage length mismatch: {len(actual_ids)} != "
            f"{config.target_passage_tokens}"
        )
    slots: list[dict[str, Any]] = []
    token_spans: list[tuple[int, int]] = []
    for registered, insertion in zip(catalog, realized):
        start = int(insertion["final_start"])
        end = int(insertion["final_end"])
        slot_index = int(registered["slot_index"])
        if start != int(final_starts[slot_index - 1]):
            raise RuntimeError(f"V4 slot {slot_index} moved during assembly")
        char_start, char_end = _token_char_span(offsets, start, end)
        active = bool(insertion["active"])
        slots.append(
            {
                **registered,
                "active": active,
                "content_kind": ("needle" if active else "length_matched_control"),
                "canonical_span_start": start,
                "canonical_span_end": end,
                "char_start": char_start,
                "char_end": char_end,
                "content_text": passage[char_start:char_end],
                "control_source": insertion.get("control"),
            }
        )
        token_spans.append((start, end))

    expected_gold = [
        (str(item["city"]), int(item["score"])) for item in catalog[: int(active_count)]
    ]
    observed = Counter(
        (match.group("city"), int(match.group("score")))
        for match in CITY_SCORE_RE.finditer(passage)
    )
    if observed != Counter(expected_gold):
        raise RuntimeError(
            "V4 haystack contamination or slot corruption: "
            f"observed={observed}, expected={Counter(expected_gold)}"
        )
    hard_negatives = _hard_negative_spans(
        passage=passage,
        offsets=offsets,
        slot_spans=token_spans,
        gap=freeze_spec.hard_negative_gap_tokens,
    )
    split = "discovery" if int(seed) in set(config.discovery_seeds) else "confirmation"
    variant_slug = variant.replace(".", "_").upper()
    controls = DESIGN_VARIANT_CONTROLS[variant]
    return {
        "schema_version": STIMULUS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "stimulus_id": (
            f"{variant_slug}_T{config.target_passage_tokens}_N{active_count}_seed{seed}"
        ),
        "design_variant": variant,
        "seed": int(seed),
        "split": split,
        "target_passage_tokens": int(config.target_passage_tokens),
        "canonical_passage_tokens": len(actual_ids),
        "canonical_tokenizer": config.canonical_tokenizer,
        "canonical_tokenizer_revision": config.canonical_tokenizer_revision,
        "num_needles": int(active_count),
        "gold_count": int(active_count),
        "passage": passage,
        "passage_sha256": _sha256_text(passage),
        "gold_pairs": [
            {
                "slot_index": index,
                "city": city,
                "score": score,
            }
            for index, (city, score) in enumerate(expected_gold, start=1)
        ],
        "slots": slots,
        "active_needle_spans": [
            {
                key: slot[key]
                for key in (
                    "slot_index",
                    "needle_id",
                    "text",
                    "city",
                    "score",
                    "canonical_span_start",
                    "canonical_span_end",
                    "canonical_token_length",
                    "char_start",
                    "char_end",
                )
            }
            for slot in slots
            if bool(slot["active"])
        ],
        "hard_negative_spans": hard_negatives,
        "design": {
            "family_key": (f"{variant}_T{config.target_passage_tokens}_seed{seed}"),
            "nested_active_slots": list(range(1, int(active_count) + 1)),
            "slot_final_starts": [int(value) for value in final_starts],
            "content_permutation_zero_based": [int(value) for value in permutation],
            "content_seed": int(content_seed),
            "inactive_replacement": ("canonical-token-length-matched_haystack"),
            "canonical_token_alignment_exact": True,
            **controls,
        },
        "haystack": {
            key: value
            for key, value in haystack_metadata.items()
            if key != "base_tokens"
        },
    }


def build_controlled_family(
    *,
    variant: str,
    seed: int,
    tokenizer: TokenizerAdapter,
    freeze_spec: ControlledFreezeSpec,
    fixed_needles: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the nested N=1..10 family for one design variant and seed."""

    freeze_spec.validate()
    config = freeze_spec.config
    if variant not in config.design_variants:
        raise ValueError(f"Unknown registered V4 design variant: {variant}")
    if int(seed) not in set(config.seeds):
        raise ValueError(f"Seed {seed} is not registered in this V4 config")
    content_seed, final_starts, permutation = _design_for_family(
        variant=variant,
        seed=seed,
        config=config,
    )
    content_fixed = DESIGN_VARIANT_CONTROLS[variant][
        "city_score_content_fixed_across_seeds"
    ]
    if content_fixed and fixed_needles is not None:
        pilot = list(fixed_needles)
    else:
        pilot = _pilot_needles(
            tokenizer=tokenizer,
            freeze_spec=freeze_spec,
            content_seed=content_seed,
            seed=seed,
        )
    pilot_fingerprint = _catalog_fingerprint(pilot)
    widths_unordered = [int(item["token_length"]) for item in pilot]
    ordered_widths = [widths_unordered[index] for index in permutation]
    filler_tokens, filler_positions = _filler_positions(
        final_starts=final_starts,
        widths=ordered_widths,
        target_tokens=config.target_passage_tokens,
        minimum_separation=config.randomized_position_min_separation_tokens,
    )
    if filler_tokens < freeze_spec.minimum_filler_tokens:
        raise ValueError("V4 leaves too few filler tokens")

    raw = _scaffold(
        tokenizer=tokenizer,
        freeze_spec=freeze_spec,
        target_filler_tokens=filler_tokens,
        content_seed=content_seed,
        seed=seed,
        # The generator's output context is discarded; these positions merely
        # need to be valid while it samples a paired base and slot controls.
        insertion_positions=tuple(range(len(pilot))),
    )
    raw_needles = list(raw["needles"])
    if _catalog_fingerprint(raw_needles) != pilot_fingerprint:
        raise RuntimeError("V4 catalog changed between pilot and exact scaffold")
    ordered_needles = [raw_needles[index] for index in permutation]
    catalog = _catalog_metadata(
        ordered_needles,
        permutation=permutation,
    )
    stimuli = [
        _compact_stimulus(
            tokenizer=tokenizer,
            freeze_spec=freeze_spec,
            variant=variant,
            seed=seed,
            active_count=int(count),
            catalog=catalog,
            ordered_needles=ordered_needles,
            base_tokens=raw["haystack"]["base_tokens"],
            haystack_metadata=raw["haystack"],
            final_starts=final_starts,
            filler_positions=filler_positions,
            permutation=permutation,
            content_seed=content_seed,
        )
        for count in config.needle_counts
    ]
    metadata = {
        "design_variant": variant,
        "seed": int(seed),
        "content_seed": int(content_seed),
        "content_permutation_zero_based": list(permutation),
        "slot_final_starts": list(final_starts),
        "canonical_slot_widths": ordered_widths,
        "catalog": catalog,
        "catalog_set_fingerprint": sorted(item["text_sha256"] for item in catalog),
        "catalog_order_fingerprint": [item["text_sha256"] for item in catalog],
        "haystack_seed": int(seed),
        **DESIGN_VARIANT_CONTROLS[variant],
    }
    return stimuli, metadata


def _audit_family_contract(
    families: Sequence[dict[str, Any]],
    *,
    config: V4Config,
) -> list[str]:
    errors: list[str] = []
    by_variant: dict[str, list[dict[str, Any]]] = {
        variant: [family for family in families if family["design_variant"] == variant]
        for variant in config.design_variants
    }
    for variant, variant_families in by_variant.items():
        if len(variant_families) != len(config.seeds):
            errors.append(f"{variant}: wrong number of seed families")
            continue
        controls = DESIGN_VARIANT_CONTROLS[variant]
        positions = {tuple(family["slot_final_starts"]) for family in variant_families}
        catalog_sets = {
            tuple(family["catalog_set_fingerprint"]) for family in variant_families
        }
        order_schedules = {
            tuple(family["content_permutation_zero_based"])
            for family in variant_families
        }
        if controls["positions_fixed_across_seeds"] != (len(positions) == 1):
            errors.append(f"{variant}: position-control contract failed")
        if controls["city_score_content_fixed_across_seeds"] != (
            len(catalog_sets) == 1
        ):
            errors.append(f"{variant}: content-control contract failed")
        if controls["city_score_order_fixed_across_seeds"] != (
            len(order_schedules) == 1
        ):
            errors.append(f"{variant}: order-control contract failed")

    # Adjacent panels share nuisance realizations wherever possible.
    keyed = {
        (family["design_variant"], int(family["seed"])): family for family in families
    }
    for seed in config.seeds:
        random_starts = {
            tuple(keyed[(variant, int(seed))]["slot_final_starts"])
            for variant in ("v4.2", "v4.3", "v4.4")
        }
        if len(random_starts) != 1:
            errors.append(f"seed={seed}: v4.2-v4.4 position schedules are not paired")
        random_orders = {
            tuple(keyed[(variant, int(seed))]["content_permutation_zero_based"])
            for variant in ("v4.3", "v4.4")
        }
        if len(random_orders) != 1:
            errors.append(f"seed={seed}: v4.3-v4.4 order schedules are not paired")
    return errors


def freeze_v4_grid(
    *,
    output_dir: str | Path,
    freeze_spec: ControlledFreezeSpec | None = None,
    require_huggingface_tokenizer: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    resolved = freeze_spec or ControlledFreezeSpec()
    resolved.validate()
    config = resolved.config
    output = Path(output_dir)
    stimuli_path = output / "stimuli.jsonl"
    if stimuli_path.exists() and not overwrite:
        raise FileExistsError(f"{stimuli_path} exists; pass overwrite=True explicitly")
    tokenizer = TokenizerAdapter(
        config.canonical_tokenizer,
        revision=config.canonical_tokenizer_revision,
        cache_dir=resolved.tokenizer_cache_dir,
    )
    if require_huggingface_tokenizer and tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Canonical V4 tokenizer failed to load; refusing the simple "
            f"fallback: {tokenizer.load_error}"
        )
    fixed_needles = _pilot_needles(
        tokenizer=tokenizer,
        freeze_spec=resolved,
        content_seed=config.fixed_needle_seed,
        seed=int(config.seeds[0]),
    )
    rows: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    for variant in config.design_variants:
        for seed in config.seeds:
            family_rows, family_metadata = build_controlled_family(
                variant=variant,
                seed=int(seed),
                tokenizer=tokenizer,
                freeze_spec=resolved,
                fixed_needles=fixed_needles,
            )
            rows.extend(family_rows)
            families.append(family_metadata)
            print(
                f"[v4 freeze] variant={variant} seed={seed} rows={len(rows)}",
                flush=True,
            )
    expected = (
        len(config.design_variants) * len(config.seeds) * len(config.needle_counts)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} V4 stimuli, generated {len(rows)}")
    ids = [str(row["stimulus_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate V4 stimulus IDs")
    contract_errors = _audit_family_contract(families, config=config)
    if contract_errors:
        raise RuntimeError(
            "V4 family-control audit failed:\n- " + "\n- ".join(contract_errors)
        )

    jsonl = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )
    fixed_catalog = _catalog_metadata(
        fixed_needles,
        permutation=tuple(range(len(fixed_needles))),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "config": config.to_dict(),
        "freeze_spec": {
            key: value for key, value in asdict(resolved).items() if key != "config"
        },
        "rows": len(rows),
        "cells": len(config.design_variants) * len(config.needle_counts),
        "rows_per_cell": len(config.seeds),
        "fixed_needle_catalog": fixed_catalog,
        "families": families,
        "design_variants": DESIGN_VARIANT_CONTROLS,
        "paired_randomization": {
            "v4.2_to_v4.4_share_positions_within_seed": True,
            "v4.3_to_v4.4_share_order_permutation_within_seed": True,
            "all_variants_share_haystack_seed_within_seed": True,
        },
        "interpretation": {
            "v4.1": (
                "Stability of a fully index/identity/position-aligned "
                "trajectory; it is not by itself evidence for an abstract counter."
            ),
            "v4.2": "Tests invariance to absolute needle position.",
            "v4.3": (
                "Tests invariance to position and ordering for one fixed "
                "city-score set."
            ),
            "v4.4": (
                "Tests invariance to position, ordering, and city-score identity."
            ),
        },
    }
    paths = {
        "stimuli": stimuli_path,
        "manifest": output / "manifest.json",
        "cell_counts": output / "cell_counts.json",
        "sha256": output / "SHA256SUMS",
    }
    counts = Counter(
        (str(row["design_variant"]), int(row["num_needles"])) for row in rows
    )
    _atomic_write(paths["stimuli"], jsonl)
    _atomic_write(paths["manifest"], _json_bytes(manifest))
    _atomic_write(
        paths["cell_counts"],
        _json_bytes(
            [
                {
                    "design_variant": variant,
                    "num_needles": count,
                    "rows": counts[(variant, count)],
                }
                for variant in config.design_variants
                for count in config.needle_counts
            ]
        ),
    )
    checksum_lines = []
    for name in ("stimuli", "manifest", "cell_counts"):
        payload = paths[name].read_bytes()
        checksum_lines.append(
            f"{hashlib.sha256(payload).hexdigest()}  {paths[name].name}"
        )
    _atomic_write(
        paths["sha256"],
        ("\n".join(checksum_lines) + "\n").encode("utf-8"),
    )
    return paths


def load_stimuli(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_v4_grid(
    *,
    stimuli_path: str | Path,
    manifest_path: str | Path | None = None,
    cache_dir: str | None = None,
    require_huggingface_tokenizer: bool = True,
) -> dict[str, Any]:
    stimuli_file = Path(stimuli_path)
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else stimuli_file.with_name("manifest.json")
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    config = V4Config.from_mapping(manifest["config"])
    tokenizer = TokenizerAdapter(
        config.canonical_tokenizer,
        revision=config.canonical_tokenizer_revision,
        cache_dir=cache_dir,
    )
    if require_huggingface_tokenizer and tokenizer.backend != "huggingface":
        raise RuntimeError(
            f"Canonical V4 tokenizer failed during audit: {tokenizer.load_error}"
        )
    rows = load_stimuli(stimuli_file)
    errors: list[str] = []
    cells: dict[tuple[str, int], set[int]] = {}
    ids: set[str] = set()
    family_by_key = {
        (str(item["design_variant"]), int(item["seed"])): item
        for item in manifest["families"]
    }
    for row in rows:
        identifier = str(row.get("stimulus_id", "<missing>"))
        try:
            if identifier in ids:
                raise ValueError("duplicate stimulus ID")
            ids.add(identifier)
            if row.get("schema_version") != STIMULUS_SCHEMA_VERSION:
                raise ValueError("stimulus schema mismatch")
            if row.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("protocol mismatch")
            variant = str(row["design_variant"])
            if variant not in config.design_variants:
                raise ValueError("unregistered design variant")
            passage = str(row["passage"])
            token_ids = tokenizer.encode(passage)
            if len(token_ids) != config.target_passage_tokens:
                raise ValueError("canonical passage length mismatch")
            if row["passage_sha256"] != _sha256_text(passage):
                raise ValueError("passage SHA256 mismatch")
            count = int(row["num_needles"])
            seed = int(row["seed"])
            cells.setdefault((variant, count), set()).add(seed)
            family = family_by_key[(variant, seed)]
            slots = row["slots"]
            if len(slots) != max(config.needle_counts):
                raise ValueError("slot count mismatch")
            if tuple(int(slot["canonical_span_start"]) for slot in slots) != tuple(
                family["slot_final_starts"]
            ):
                raise ValueError("slot position mismatch")
            if [bool(slot["active"]) for slot in slots] != [
                index < count for index in range(len(slots))
            ]:
                raise ValueError("nested active-slot policy mismatch")
            expected = Counter(
                (str(item["city"]), int(item["score"]))
                for item in family["catalog"][:count]
            )
            observed = Counter(
                (match.group("city"), int(match.group("score")))
                for match in CITY_SCORE_RE.finditer(passage)
            )
            if observed != expected:
                raise ValueError("city-score contamination or corruption")
            if len(row["hard_negative_spans"]) != len(slots):
                raise ValueError("hard-negative count mismatch")
            for slot, negative in zip(slots, row["hard_negative_spans"]):
                if int(slot["canonical_token_length"]) != int(
                    negative["canonical_token_length"]
                ):
                    raise ValueError("hard-negative length mismatch")
        except Exception as exc:
            errors.append(f"{identifier}: {type(exc).__name__}: {exc}")

    for variant in config.design_variants:
        for count in config.needle_counts:
            if cells.get((variant, int(count)), set()) != set(config.seeds):
                errors.append(f"seed mismatch for {variant}, N={count}")
    expected_rows = (
        len(config.design_variants) * len(config.seeds) * len(config.needle_counts)
    )
    if len(rows) != expected_rows:
        errors.append(f"row count mismatch: {len(rows)} != {expected_rows}")
    errors.extend(_audit_family_contract(manifest["families"], config=config))

    checksum_file = stimuli_file.with_name("SHA256SUMS")
    checksum_status: dict[str, bool] = {}
    if not checksum_file.exists():
        errors.append("SHA256SUMS is missing")
    else:
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, filename = line.split(maxsplit=1)
            candidate = checksum_file.parent / filename.strip()
            passed = (
                candidate.exists()
                and hashlib.sha256(candidate.read_bytes()).hexdigest() == digest
            )
            checksum_status[filename.strip()] = passed
            if not passed:
                errors.append(f"checksum mismatch: {filename.strip()}")
    return {
        "schema_version": "realistic_niah_v4_audit_v1",
        "protocol_version": PROTOCOL_VERSION,
        "passed": not errors,
        "rows_checked": len(rows),
        "cells_checked": sorted(f"{variant}:N{count}" for variant, count in cells),
        "checksum_status": checksum_status,
        "errors": errors,
    }
