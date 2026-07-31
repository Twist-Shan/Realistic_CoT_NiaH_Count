# Realistic NIAH V4: non-thinking counting mechanisms

V4 is a preregistered-style mechanistic extension of the Realistic NIAH
counting task. It asks two descriptive questions and two causal questions:

1. Does the post-block residual stream at successive needle spans contain a
   stable, low-noise count/index trajectory?
2. At the answer-query token, are there attention heads whose mass is both
   concentrated on needle spans and broadly distributed across those spans?
3. Are discovery-ranked broad heads necessary for count prediction relative
   to layer-matched random heads?
4. Can localized residual-stream patching move a receiver prompt toward a
   paired donor count?

V4 is non-thinking only. It does not treat generated chain of thought as a
measurement of the internal counting algorithm.

## Registered grid

| Dimension | Registered value |
| --- | --- |
| Models | `Qwen/Qwen3-8B`, `google/gemma-4-E4B-it` |
| Passage length | 10,000 canonical Qwen3-8B tokens |
| Counts | 1 through 10 |
| Seeds | 1234 through 1263 (30) |
| Discovery split | 1234 through 1253 (20) |
| Confirmation split | 1254 through 1263 (10) |
| Prompt | direct, non-thinking |
| Answer query | last token of a teacher-forced `Total:` prefix |
| Representation sites | needle-span end; mean over the full needle span |

The exact model and tokenizer revisions are immutable SHAs in
`src/realistic_niah_v4/spec.py`.

## Four cumulative control panels

Every `(panel, seed)` is one nested family. It has one base haystack, one set
of ten slot positions, one ordered ten-item catalog, and one length-matched
control for each slot. The N=1 through N=10 rows activate the first N slots
and replace all later slots with their paired controls. Thus count changes do
not change canonical passage length or shift later slots.

| Panel | Position across seeds | City-score order across seeds | City-score content across seeds | Main question |
| --- | --- | --- | --- | --- |
| v4.1 | fixed | fixed | fixed | Is an entirely aligned trajectory stable at all? |
| v4.2 | varied | fixed | fixed | Does it survive absolute-position variation? |
| v4.3 | varied | varied | fixed set | Does it survive permutation of the same facts? |
| v4.4 | varied | varied | varied | Does it generalize to new city-score facts? |

The relaxations are paired rather than sampled independently:

- v4.2, v4.3, and v4.4 use the same position schedule for a given seed;
- v4.3 and v4.4 use the same catalog permutation for a given seed;
- all four panels use the same haystack seed for a given seed.

Varying city-score content can change the number of canonical tokens occupied
by the ten facts. To preserve the exact 10,000-token total, v4.4 can therefore
use a slightly different prefix length of the same seed-specific haystack.
This consequence is recorded in the manifest and should not be described as
perfect filler identity across v4.3 and v4.4.

The freeze audit checks all registered control claims from realized data; it
does not infer them from seed settings alone. It also verifies exact
decode/re-encode identity, length, nesting, city-score contamination,
length-matched hard negatives, unique IDs, cell counts, and file checksums.

## Representation analysis

The primary prompt-reading analysis uses only the N=10 row of each family.
Forward hooks retain post-block states at ten needle spans and discard all
other token states. For every layer, seed, and panel, two arrays are saved:

- `span_end`: the final token of each needle span;
- `span_mean`: the mean over every model token in the span.

The implementation never requests full-sequence hidden states. Restartable
per-prompt shards are stored in float16 by default; reductions and statistical
analyses are done in float32 or float64.

For each panel, pooling, and layer:

1. a ridge probe is selected by grouped-seed cross-validation on the 20
   discovery seeds;
2. R², MAE, and Pearson correlation are measured once on the 10 confirmation
   seeds;
3. discovery index centroids define the mean trajectory;
4. held-out distance to the matching discovery centroid quantifies scatter
   noise;
5. signal RMS, noise-to-signal ratio, adjacent-step regularity,
   path-to-chord ratio, linear CKA, and pairwise-distance reproducibility
   quantify geometry.

For each pooling, the primary layer is the layer with the highest v4.1
discovery grouped-seed CV R². A shared three-component PCA basis is fit only
on v4.1 discovery examples at that layer; the first two components are
displayed. All four panels and both splits are then projected into those same
axes. The resulting four-panel plot shows raw scatter plus discovery and
confirmation mean curves.

The first seed-sensitive relaxation is reported separately for probe MAE and
curve-residual RMS normalized by the discovery curve's signal RMS. The
decision rule is the earliest adjacent step whose increase has a positive 95%
paired seed-bootstrap interval. With only ten confirmation seeds, intervals
may be wide; a null call is not evidence of invariance.

### Interpretation boundary

In v4.1, occurrence index, content identity, and absolute position are
one-to-one. A clean curve there establishes only a stable aligned trajectory.
Position invariance requires v4.2; deconfounding fixed content from index
requires v4.3; evidence for content-general counting requires v4.4. A probe
establishes decodability, not causal use.

## Answer-query attention

Let `a(t)` be one head's answer-query attention row, and let `S_i` be active
needle span `i`. The registered per-span mass is

```text
m_i = sum_{t in S_i} a(t)
```

For N active spans, define

```text
total mass = sum_i m_i
p_i = m_i / total mass
coverage = exp(-sum_i p_i log p_i) / N
broad primary = total mass * coverage
```

Coverage is one when mass is uniform across spans and approaches `1/N` when
one span dominates. The analysis also saves length-normalized coverage,
coefficient of variation, effective number of attended spans, and per-token
contrast against ten length-matched hard-negative spans.

Discovery heads are ranked by mean `broad primary` over N=2 through N=10,
with positive hard-negative contrast preferred. N=1 is excluded from ranking
because its coverage is identically one. Confirmation seeds are used for
locked visualization and causal tests, never head selection.

The 10k-token prefix is evaluated once with an efficient KV cache. Only the
single final answer-query token is evaluated with eager attention. The code
therefore materializes an `heads × 1 × key_length` row per layer, never a
10,000 × 10,000 attention matrix. Local-attention layers are scored over
their realized key window and record that window's absolute start.

For every prompt, the complete answer-query row from every head is retained
as an uncompressed float16 NPZ shard. Each layer is a separate array because
Gemma 4 local and global layers have different key-axis lengths. The shard
also records absolute key starts, layer types, query position, and sequence
length. Candidate-count logits and probabilities from the same cached query
forward are saved in the metric shard and summarized as a behavioral sanity
check. Full-sequence attention matrices and full Q/K/V tensors are not
materialized.

## Causal tests

### Head ablation

For each panel, the top 1, 2, 4, and 8 discovery-ranked heads are zeroed at
the input to the attention output projection. The primary intervention
affects only the answer-query position. Each set is compared with three
deterministic, layer-matched random-head sets on confirmation seeds. Saved
outcomes include the correct-count logit margin, candidate-only accuracy,
and expected count over the registered 1–10 candidate tokens.

Ablation can demonstrate necessity but can also introduce distribution shift.
The layer-matched controls and top-N dose response are required for
interpretation.

### Residual-stream patching

For paired counts `(1→2, 3→4, 5→6, 7→8, 9→10)`, donor post-block states are
patched into the lower-count receiver at five preregistered relative depths.
Three sites are tested:

1. the answer-query state;
2. the end token of the newly activated needle slot;
3. every token in that slot when donor and receiver model-token lengths are
   equal.

Full-span patches with unequal model-token lengths are explicitly recorded as
skipped. Primary outcomes are the change in
`logit(donor count) - logit(receiver count)`, expected-count shift, and
recovery fraction relative to the paired donor baseline.

## Workflow

Create a dedicated environment. V4 uses Transformers directly, not vLLM,
because it needs hooks, KV-cache access, and eager attention rows.

```bash
python3 -m venv /path/to/venvs/realistic-niah-v4
. /path/to/venvs/realistic-niah-v4/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-mechanistic-v4.txt
export PYTHONPATH=src
```

Freeze the full 1,200-row grid:

```bash
RUN_ROOT=/path/to/runs/realistic_niah_v4/run_YYYYMMDD
HF_CACHE=/path/to/hf-cache

PYTHONPATH=src python scripts/freeze_realistic_niah_v4.py \
  --config configs/realistic_niah_v4.json \
  --output-dir "${RUN_ROOT}/dataset" \
  --cache-dir "${HF_CACHE}"
```

Run each model separately. Separate stages are recommended because every
large stage is restartable and has its own failure boundary.

```bash
for MODEL in Qwen3-8B Gemma4-E4B; do
  PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
    --stage preflight \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}" \
    --cache-dir "${HF_CACHE}" \
    --forward-smoke

  PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
    --stage representation-capture \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}" \
    --cache-dir "${HF_CACHE}"

  PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
    --stage representation-analyze \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}"

  PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
    --stage attention \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}" \
    --cache-dir "${HF_CACHE}"

  PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
    --stage ablation \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}" \
    --cache-dir "${HF_CACHE}"

  PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
    --stage patching \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}" \
    --cache-dir "${HF_CACHE}"
done
```

Use `--variants`, `--seeds`, and `--counts` only for smoke tests or explicitly
labelled partial runs. Full representation analysis requires all four panels
and all 30 seeds. Ablation requires previously generated discovery rankings.

## Outputs

```text
RUN_ROOT/
├── dataset/
│   ├── stimuli.jsonl
│   ├── manifest.json
│   ├── cell_counts.json
│   ├── SHA256SUMS
│   └── audit.json
├── Qwen3-8B/
│   ├── runtime_provenance.json
│   ├── events.jsonl
│   ├── preflight.json
│   ├── representation/
│   │   ├── capture/shards/
│   │   └── analysis/
│   ├── attention/
│   └── causal/
└── Gemma4-E4B/
    └── ...
```

Hidden-state shards, result tables, figures, model caches, and run directories
should remain outside Git. `events.jsonl` records model load and stage
durations. `runtime_provenance.json` records exact inputs, package versions,
CUDA visibility, model revision, and Git state.

Within each model's `attention/` directory, `capture/shards/` contains
per-head metrics and `capture/raw_shards/` contains the complete float16
answer-query rows. `answer_query_behavior.csv` and
`answer_query_behavior_by_count.csv` contain the prompt-level behavioral
sanity checks derived from the same query forwards.

## Validation status required before formal inference

CPU tests cover configuration, four-panel randomization contracts, exact
nesting, prompt spans, broad-attention metrics, adapter hooks, causal metric
accounting, and synthetic representation recovery. Before a formal run, both
registered checkpoints still require one GPU preflight that confirms:

- model loading with the pinned Transformers build;
- single-token count continuations;
- finite count logits at 10k context;
- query-only eager attention output shape for every layer;
- agreement diagnostics between the cached eager query and full SDPA logits;
- pre-output-projection head layout;
- answer-query head ablation;
- answer-query, needle-end, and full-span residual patches.

Passing that smoke test establishes implementation compatibility, not
scientific validity.
