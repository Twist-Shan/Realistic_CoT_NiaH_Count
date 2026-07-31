from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from realistic_niah_v4.attention import (
    _validate_raw_attention_shard,
    _write_raw_attention_shard,
    broad_attention_metrics,
)
from realistic_niah_v4.causal import count_logit_metrics
from realistic_niah_v4.modeling import (
    _attention_tensor,
    capture_post_block_states,
    capture_span_states,
    discover_decoder_adapter,
    query_attention_rows,
    run_last_logits,
    run_with_head_ablation,
    run_with_residual_patch,
)
from realistic_niah_v4.prompts import (
    PromptEncoding,
    TokenSpan,
    render_v4_prompt,
)
from realistic_niah_v4.representation import (
    analyze_representation_captures,
)
from realistic_niah_v4.spec import (
    DESIGN_VARIANTS,
    MODEL_SPECS,
    V4Config,
)
from realistic_niah_v4.stimuli import (
    ControlledFreezeSpec,
    audit_v4_grid,
    freeze_v4_grid,
    load_stimuli,
)


def _small_config() -> V4Config:
    config = V4Config(
        target_passage_tokens=300,
        needle_counts=(1, 2, 3),
        seeds=(11, 12, 13, 14),
        discovery_seeds=(11, 12),
        confirmation_seeds=(13, 14),
        canonical_tokenizer="simple",
        canonical_tokenizer_revision="test-only",
        fixed_slot_depths=(0.2, 0.5, 0.8),
        randomized_position_min_separation_tokens=20,
        representation_count=3,
        patch_count_pairs=((1, 2), (2, 3)),
    )
    config.validate()
    return config


@pytest.fixture()
def small_grid(tmp_path: Path) -> tuple[V4Config, Path]:
    haystack = tmp_path / "haystack"
    haystack.mkdir()
    for index in range(4):
        text = " ".join(f"essay{index}_token{token % 97}" for token in range(1_200))
        (haystack / f"essay_{index}.txt").write_text(text, encoding="utf-8")
    entities = tmp_path / "cities.csv"
    entities.write_text(
        "entity,region,category\n"
        + "".join(
            f"City{index},Region{index % 3},Category{index % 2}\n"
            for index in range(30)
        ),
        encoding="utf-8",
    )
    template = tmp_path / "template.txt"
    template.write_text(
        "In the {year} city score audit, {entity} received a score of {score}.\n",
        encoding="utf-8",
    )
    config = _small_config()
    output = tmp_path / "frozen"
    freeze_v4_grid(
        output_dir=output,
        freeze_spec=ControlledFreezeSpec(
            config=config,
            haystack_dir=str(haystack),
            entities_path=str(entities),
            fact_templates_path=str(template),
            minimum_filler_tokens=32,
            hard_negative_gap_tokens=2,
        ),
        require_huggingface_tokenizer=False,
    )
    return config, output


def test_registered_v4_grid_accounting() -> None:
    config = V4Config()
    assert config.design_variants == DESIGN_VARIANTS
    assert len(config.seeds) == 30
    assert len(config.discovery_seeds) == 20
    assert len(config.confirmation_seeds) == 10
    assert (
        len(config.design_variants) * len(config.seeds) * len(config.needle_counts)
        == 1_200
    )
    assert set(config.model_labels) == {"Qwen3-8B", "Gemma4-E4B"}


def test_four_panel_freeze_contract_and_audit(
    small_grid: tuple[V4Config, Path],
) -> None:
    config, output = small_grid
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    families = manifest["families"]
    assert manifest["rows"] == 4 * 4 * 3

    def selected(variant: str) -> list[dict[str, object]]:
        return [row for row in families if row["design_variant"] == variant]

    v41 = selected("v4.1")
    assert len({tuple(row["slot_final_starts"]) for row in v41}) == 1
    assert len({tuple(row["catalog_order_fingerprint"]) for row in v41}) == 1

    v42 = selected("v4.2")
    assert len({tuple(row["slot_final_starts"]) for row in v42}) > 1
    assert len({tuple(row["catalog_order_fingerprint"]) for row in v42}) == 1

    v43 = selected("v4.3")
    assert len({tuple(row["slot_final_starts"]) for row in v43}) > 1
    assert len({tuple(row["catalog_set_fingerprint"]) for row in v43}) == 1
    assert len({tuple(row["catalog_order_fingerprint"]) for row in v43}) > 1

    v44 = selected("v4.4")
    assert len({tuple(row["catalog_set_fingerprint"]) for row in v44}) > 1

    keyed = {(row["design_variant"], row["seed"]): row for row in families}
    for seed in config.seeds:
        assert (
            keyed[("v4.2", seed)]["slot_final_starts"]
            == keyed[("v4.3", seed)]["slot_final_starts"]
            == keyed[("v4.4", seed)]["slot_final_starts"]
        )
        assert (
            keyed[("v4.3", seed)]["content_permutation_zero_based"]
            == keyed[("v4.4", seed)]["content_permutation_zero_based"]
        )

    rows = load_stimuli(output / "stimuli.jsonl")
    family_rows = [
        row for row in rows if row["design_variant"] == "v4.3" and row["seed"] == 13
    ]
    assert [row["gold_count"] for row in family_rows] == [1, 2, 3]
    assert all(row["canonical_passage_tokens"] == 300 for row in family_rows)
    starts = {
        tuple(slot["canonical_span_start"] for slot in row["slots"])
        for row in family_rows
    }
    assert len(starts) == 1
    for row in family_rows:
        assert [slot["active"] for slot in row["slots"]] == [
            index < row["gold_count"] for index in range(3)
        ]
        assert row["passage"].count("Excerpt:") == 3
        assert row["passage"].count("End excerpt.") == 3
    assert all(
        family["nested_token_identity_outside_toggled_slot"] is True
        for family in families
    )
    for lower, higher in zip(family_rows, family_rows[1:]):
        toggled = higher["slots"][higher["gold_count"] - 1]
        start = int(toggled["canonical_span_start"])
        end = int(toggled["canonical_span_end"])
        lower_tokens = lower["passage"].split()
        higher_tokens = higher["passage"].split()
        assert lower_tokens[:start] == higher_tokens[:start]
        assert lower_tokens[end:] == higher_tokens[end:]

    audit = audit_v4_grid(
        stimuli_path=output / "stimuli.jsonl",
        manifest_path=output / "manifest.json",
        require_huggingface_tokenizer=False,
    )
    assert audit["passed"], audit["errors"]


def test_broad_metric_rewards_uniform_span_coverage() -> None:
    needles = tuple(
        TokenSpan(index, index * 3, index * 3 + 1, True, "needle", 1, 1)
        for index in range(1, 4)
    )
    negatives = tuple(
        TokenSpan(index, 20 + index, 21 + index, False, "negative", 1, 1)
        for index in range(1, 4)
    )
    uniform = np.zeros(32)
    concentrated = np.zeros(32)
    uniform[[3, 6, 9]] = 0.2
    concentrated[[3, 6, 9]] = [0.58, 0.01, 0.01]
    uniform[0] = 0.4
    concentrated[0] = 0.4
    broad = broad_attention_metrics(uniform, needles, negatives)
    narrow = broad_attention_metrics(concentrated, needles, negatives)
    assert broad["broad_mass"] == pytest.approx(narrow["broad_mass"])
    assert broad["broad_coverage"] == pytest.approx(1.0)
    assert broad["broad_primary"] > narrow["broad_primary"]
    assert broad["broad_contrast"] > 0


class WhitespaceFastTokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self.vocabulary:
            self.vocabulary[token] = len(self.vocabulary) + 1
        return self.vocabulary[token]

    def apply_chat_template(self, messages, **_kwargs) -> str:
        return "\n".join(str(message["content"]) for message in messages) + (
            "\n<assistant>"
        )

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        matches = list(re.finditer(r"\S+", text))
        result = {
            "input_ids": [self._id(match.group(0)) for match in matches],
            "attention_mask": [1] * len(matches),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]
        return result


def test_prompt_maps_spans_and_single_token_counts(
    small_grid: tuple[V4Config, Path],
) -> None:
    config, output = small_grid
    row = next(
        item
        for item in load_stimuli(output / "stimuli.jsonl")
        if item["design_variant"] == "v4.1" and item["gold_count"] == 3
    )
    encoding = render_v4_prompt(
        row,
        tokenizer=WhitespaceFastTokenizer(),
        model_spec=MODEL_SPECS["Qwen3-8B"],
        config=config,
    )
    assert encoding.design_variant == "v4.1"
    assert encoding.query_position == encoding.sequence_length - 1
    assert len(encoding.needle_spans) == 3
    assert len(encoding.hard_negative_spans) == 3
    assert len(dict(encoding.count_candidate_token_ids)) == 3
    assert all(span.start < span.end for span in encoding.needle_spans)


class ToyAttention(nn.Module):
    def __init__(self, hidden_size: int = 4, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_type = "global"
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(hidden)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.self_attn(hidden) + torch.tanh(self.mlp(hidden))


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=4,
            num_attention_heads=2,
        )
        self.embed = nn.Embedding(40, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock(), ToyBlock()])
        self.lm_head = nn.Linear(4, 40, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        logits_to_keep=None,
        **_kwargs,
    ):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep) :]
        return SimpleNamespace(logits=logits)


def _toy_encoding(input_ids: tuple[int, ...]) -> PromptEncoding:
    spans = (
        TokenSpan(1, 1, 2, True, "needle", 1, 1),
        TokenSpan(2, 2, 3, True, "needle", 1, 1),
    )
    return PromptEncoding(
        stimulus_id="toy",
        design_variant="v4.1",
        seed=1,
        split="confirmation",
        count=2,
        model_label="toy",
        text="toy",
        generation_prompt="toy",
        input_ids=input_ids,
        attention_mask=(1,) * len(input_ids),
        query_position=len(input_ids) - 1,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_token_ids=((1, 10), (2, 11), (3, 12)),
    )


def test_decoder_hooks_ablation_and_residual_patch() -> None:
    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    assert adapter.num_layers == 2
    encoding = _toy_encoding((1, 2, 3, 4))
    captured = capture_span_states(model, adapter, encoding)
    assert captured["span_end"].shape == (2, 2, 4)
    assert captured["span_mean"].shape == (2, 2, 4)

    baseline = run_last_logits(model, encoding)
    ablated = run_with_head_ablation(
        model,
        adapter,
        encoding,
        [(0, 0)],
        scope="answer_query",
    )
    assert not torch.allclose(baseline, ablated)

    donor = _toy_encoding((5, 6, 7, 8))
    donor_states = capture_span_states(
        model,
        adapter,
        donor,
        spans=[TokenSpan(1, 3, 4, True, "query", 1, 1)],
        layers=[0],
    )["span_end"][0]
    patched = run_with_residual_patch(
        model,
        adapter,
        encoding,
        layer=0,
        receiver_positions=[encoding.query_position],
        donor_states=donor_states,
    )
    assert patched.shape == baseline.shape
    assert not torch.allclose(baseline, patched)

    metrics = count_logit_metrics(baseline, encoding)
    assert {
        metrics["predicted_count_among_candidates"],
        metrics["gold_count"],
    }.issubset({1, 2, 3})
    assert 0.0 <= metrics["correct_count_probability"] <= 1.0


def test_raw_answer_query_attention_round_trip(tmp_path: Path) -> None:
    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    encoding = _toy_encoding((1, 2, 3, 4))
    raw_rows = [
        torch.softmax(
            torch.arange(
                adapter.num_heads[layer] * encoding.sequence_length,
                dtype=torch.float32,
            ).reshape(adapter.num_heads[layer], encoding.sequence_length),
            dim=-1,
        )
        for layer in range(adapter.num_layers)
    ]
    shard = tmp_path / "raw_attention.npz"
    _write_raw_attention_shard(
        shard,
        attention_rows=raw_rows,
        key_starts=[0] * adapter.num_layers,
        adapter=adapter,
        encoding=encoding,
        save_dtype="float16",
    )
    _validate_raw_attention_shard(
        shard,
        adapter=adapter,
        encoding=encoding,
    )
    with np.load(shard, allow_pickle=False) as saved:
        assert saved["layer_000"].dtype == np.float16
        assert saved["layer_000"].shape == (
            adapter.num_heads[0],
            encoding.sequence_length,
        )


def test_attention_tensor_accepts_single_block_query() -> None:
    tensor = torch.zeros(1, 2, 1, 1, 7)
    assert _attention_tensor(tensor).shape == (1, 2, 1, 7)
    with pytest.raises(RuntimeError, match="more than one query cell"):
        _attention_tensor(torch.zeros(1, 2, 2, 1, 7))


def test_tiny_transformers_architectures_support_v4_hooks() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "AutoModelForMultimodalLM"):
        pytest.skip("Installed Transformers predates the Gemma 4 API")
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4AudioConfig,
            Gemma4Config,
            Gemma4TextConfig,
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )
    except ImportError:
        pytest.skip("Installed Transformers lacks a registered V4 architecture")

    spans = tuple(
        TokenSpan(index, position, position + 1, True, "needle", 1, 1)
        for index, position in enumerate((2, 5, 8), start=1)
    )
    negatives = tuple(
        TokenSpan(index, position, position + 1, False, "negative", 1, 1)
        for index, position in enumerate((3, 6, 9), start=1)
    )

    def encoding(label: str) -> PromptEncoding:
        return PromptEncoding(
            stimulus_id=f"tiny_{label}",
            design_variant="v4.1",
            seed=1,
            split="confirmation",
            count=3,
            model_label=label,
            text="",
            generation_prompt="",
            input_ids=tuple(range(4, 16)),
            attention_mask=(1,) * 12,
            query_position=11,
            slot_spans=spans,
            needle_spans=spans,
            hard_negative_spans=negatives,
            count_candidate_token_ids=tuple(
                (count, 20 + count) for count in range(1, 11)
            ),
        )

    models = [
        (
            "qwen",
            Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=4,
                    max_position_embeddings=128,
                    layer_types=["full_attention", "full_attention"],
                )
            ),
        ),
        (
            "gemma4",
            Gemma4ForConditionalGeneration(
                Gemma4Config(
                    text_config=Gemma4TextConfig(
                        vocab_size=64,
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=4,
                        global_head_dim=4,
                        num_global_key_value_heads=2,
                        max_position_embeddings=128,
                        sliding_window=8,
                        layer_types=["sliding_attention", "full_attention"],
                        vocab_size_per_layer_input=64,
                        hidden_size_per_layer_input=0,
                        num_kv_shared_layers=0,
                    ),
                    vision_config=Gemma4VisionConfig(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=4,
                        head_dim=4,
                        max_position_embeddings=128,
                        patch_size=4,
                        position_embedding_size=64,
                    ),
                    audio_config=Gemma4AudioConfig(
                        hidden_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        subsampling_conv_channels=(4, 4),
                        output_proj_dims=16,
                    ),
                    boi_token_id=50,
                    eoi_token_id=51,
                    image_token_id=52,
                    video_token_id=53,
                    boa_token_id=54,
                    eoa_token_index=55,
                    audio_token_id=56,
                )
            ),
        ),
    ]
    for label, model in models:
        model.eval()
        adapter = discover_decoder_adapter(model)
        if label == "gemma4":
            assert adapter.layer_container_name == "model.language_model.layers"
        item = encoding(label)
        baseline = run_last_logits(model, item)
        captured = capture_span_states(model, adapter, item)
        rows, key_starts = query_attention_rows(model, adapter, item)
        ablated = run_with_head_ablation(
            model, adapter, item, [(0, 0)], scope="answer_query"
        )
        _, states = capture_post_block_states(
            model,
            adapter,
            item,
            [item.query_position],
            layers=[0],
        )
        patched = run_with_residual_patch(
            model,
            adapter,
            item,
            layer=0,
            receiver_positions=[item.query_position],
            donor_states=states[0] + 0.1,
        )
        assert captured["span_end"].shape == (2, 3, 16)
        expected_attention_shapes = (
            [(4, 8), (4, 12)] if label == "gemma4" else [(4, 12), (4, 12)]
        )
        assert [tuple(row.shape) for row in rows] == expected_attention_shapes
        assert key_starts == ([4, 0] if label == "gemma4" else [0, 0])
        assert not torch.allclose(baseline, ablated)
        assert not torch.allclose(baseline, patched)


def test_synthetic_representation_analysis_recovers_count_curve(
    tmp_path: Path,
) -> None:
    config = V4Config(
        seeds=(1, 2, 3, 4, 5, 6, 7, 8),
        discovery_seeds=(1, 2, 3, 4, 5, 6),
        confirmation_seeds=(7, 8),
    )
    config.validate()
    capture = tmp_path / "capture"
    index_rows = []
    direction = np.asarray([1.0, -0.5, 0.25, 0.1, -0.2, 0.4])
    noise_scale = {
        "v4.1": 0.01,
        "v4.2": 0.03,
        "v4.3": 0.12,
        "v4.4": 0.35,
    }
    for variant_index, variant in enumerate(config.design_variants):
        for seed in config.seeds:
            rng = np.random.default_rng(1000 * variant_index + seed)
            array = np.empty((2, 10, 6), dtype=np.float32)
            array[0] = rng.normal(size=(10, 6))
            for count in range(1, 11):
                array[1, count - 1] = float(count) * direction + rng.normal(
                    scale=noise_scale[variant], size=direction.shape
                )
            relative = Path("shards") / variant / f"{variant}_{seed}.npz"
            shard = capture / relative
            shard.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                shard,
                layer_indices=np.asarray([0, 1]),
                span_end=array,
                span_mean=array + 0.001,
            )
            index_rows.append(
                {
                    "schema_version": "realistic_niah_v4_capture_v1",
                    "stimulus_id": f"{variant}_{seed}",
                    "design_variant": variant,
                    "model_label": "synthetic",
                    "seed": seed,
                    "split": (
                        "discovery"
                        if seed in config.discovery_seeds
                        else "confirmation"
                    ),
                    "count": 10,
                    "sequence_length": 100,
                    "query_position": 99,
                    "poolings": ["span_end", "span_mean"],
                    "array_shape": [2, 10, 6],
                    "save_dtype": "float32",
                    "shard_path": relative.as_posix(),
                }
            )
    index_path = capture / "capture_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    outputs = analyze_representation_captures(
        capture_index_path=index_path,
        output_dir=tmp_path / "analysis",
        config=config,
    )
    metrics = np.genfromtxt(
        outputs["layer_metrics"],
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    matching = [
        row
        for row in metrics
        if row["design_variant"] == "v4.1"
        and row["pooling"] == "span_end"
        and int(row["layer"]) == 1
    ]
    assert len(matching) == 1
    assert float(matching[0]["confirmation_r2"]) > 0.95
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["primary_layer_selection"]["layers"]["span_end"] == 1
    assert (
        tmp_path / "analysis" / "figures" / "shared_pca_span_end_layer_1.png"
    ).exists()
