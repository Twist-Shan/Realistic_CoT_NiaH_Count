# Realistic NIAH V4.4.2: trace hidden states and attention

V4.4.2 is an isolated, append-only extension of the frozen V4.4 panel. It
changes neither stimuli nor the numeric task. It studies hidden-state and
attention changes in a `2 models × 2 modes × 2 prompt variants` grid.

## Registered grid

Models: `Qwen3-8B`, `Gemma4-E4B`.

Modes: `nonthinking`, `native_thinking`. The user message is identical across
the two modes. Only the model's native chat-template thinking flag changes.

Prompt variants:

- `cue_present` begins with the two frozen definition sentences:

  ```text
  You will need to count all city-score audit records in the passage below.
  A city-score audit record names one city and gives that city's numeric score.
  ```

- `cue_absent` deletes exactly those two sentences. The passage, tags, question,
  numeric output instruction, and assistant formatting are unchanged.

The common question is:

```text
How many city-score audit records are in the passage?
Do not explain, reason aloud, quote, or list any records.
Write the count using ordinary decimal digits, with no space after the colon.
Your entire response must be exactly one line:
Total:<integer>
```

All four cells per model are fresh, independent runs. The H100 worker does not
need access to the earlier V4.4 filestream.

The formal panel uses 10 fixed seeds (`1234`–`1243`) and all 10 counts. There
is no discovery/confirmation analysis split in V4.4.2; the old `split` field is
retained only as frozen-stimulus provenance. Each condition bucket therefore
contains 100 examples. All eight buckets are newly run (800 examples total).

## Capture contract

Native generation uses the V3 settings and at most 4096 new tokens:

- Qwen: temperature 0.6, top-p 0.95, top-k 20;
- Gemma: temperature 1.0, top-p 0.95, top-k 64.

The saved residual stream contains trace tokens and the final-answer region,
not prompt-token hidden states. The comparison query is always the final token
of `Total:` (the state that predicts the first answer digit).

Raw attention rows are not stored. A teacher-forced replay saves:

- normalized Q only for trace/final query positions;
- normalized K for the full key context, including prompt K;
- model-produced RoPE tables and exact layer metadata;
- post-block residuals only for trace/final positions.

Gemma shared-KV layers reference the last non-sharing producer of the same
attention type. This is exact architectural sharing, not cross-condition
deduplication. Attention is reconstructed offline in float32 with causal/local
masks, GQA mapping, scaling, and logit soft-capping recorded from the module.

## Primary comparisons

1. `cue_present_mode_effect`: native thinking minus freshly run non-thinking
   at the final `Total:` query under the cue-present prompt.
2. `native_cue_effect`: cue absent minus cue present within native thinking,
   including trace trajectories and trace attention maps.
3. `cue_absent_mode_effect`: native thinking minus non-thinking with the cue
   removed, at the final `Total:` query.

Trace maps use 128 normalized query-time bins and 128 trace key-time bins.
Region summaries cover cue, needle spans, needle endpoints, passage, question,
other prompt, trace, and final answer.

## Runbook

Install the pinned environment:

```bash
python -m pip install -r requirements-mechanistic-v4_4_2.txt
```

Before a full run, perform one end-to-end pilot for each model:

```bash
export PYTHONPATH=src
python scripts/run_realistic_niah_v4_4_2.py --stage generate-capture \
  --stimuli "$STIMULI" --config configs/realistic_niah_v4_4_2.json \
  --output-dir "$RUN_ROOT" --model Qwen3-8B --mode native_thinking \
  --prompt-variant cue_present --seeds 1234 --counts 1 --cache-dir "$HF_CACHE"

python scripts/run_realistic_niah_v4_4_2.py --stage analyze-existing \
  --config configs/realistic_niah_v4_4_2.json --output-dir "$RUN_ROOT" \
  --model Qwen3-8B --mode native_thinking --prompt-variant cue_present \
  --analysis-device cpu
```

Repeat the pilot for Gemma. Check `generation.json` boundary status, capture
shapes, `attention_summary.pt`, disk use, and peak GPU memory. Then launch the
full grid with `scripts/launch_realistic_niah_v4_4_2.sh`.

The launcher uses `generate-capture` so generation and teacher replay share one
model load. `analyze-existing` never loads a model, but the full run defaults to
CUDA because reconstructing every trace QK map on CPU would be slow. Set
`V442_ANALYSIS_DEVICE=cpu` only when GPU time is unavailable. All shards are
restartable unless `--overwrite` is explicitly used.

At an 11k-token prompt and the worst-case 4096-token trace, the uncompressed
float16 capture estimate is about 3.55 GiB per Qwen native shard and 2.70 GiB
per Gemma native shard. Across 200 native plus 200 non-thinking shards per
model, the two-model upper bound is about 1.6 TiB. This is a ceiling,
not an expected total: run the two pilots and extrapolate from observed trace
lengths before allocating storage. Keep the output on persistent/network
storage and do not rely on an ephemeral container disk.

## Output layout

```text
RUN_ROOT/
  runtime_provenance.json
  conditions/<model>/<prompt>/<mode>/<split>/<stimulus>/
    generation.json
    capture/
      capture_manifest.json
      layer_*_hidden.pt
      layer_*_q_norm.pt
      kv_source_*_k_norm.pt
      layer_*_rope.pt
      attention_summary.pt
      attention_head_summary.csv.gz
  filestream_index.jsonl
  filestream_manifest.json
  analysis/tables/*.csv.gz
  analysis/realistic_niah_v4_4_2_report.md
```

`events.jsonl` and per-command logs provide restart/audit evidence. Large
artifacts should remain under `RUN_ROOT`, outside Git.
