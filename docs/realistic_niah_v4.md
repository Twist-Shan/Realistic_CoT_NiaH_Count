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
| Answer vocabulary | decimal strings `1` through `10` |
| Representation sites | needle-span end; mean over the full needle span |

The exact model and tokenizer revisions are immutable SHAs in
`src/realistic_niah_v4/spec.py`.

The prompt requests ordinary decimal digits. With both registered tokenizers,
`1` through `9` are one answer token while `10` is two and shares its first
token with `1`. Therefore V4 does not use a ten-way softmax at one position to
label behavior. It greedily generates the actual continuation after the
already-present `Total:` prefix and applies a strict parser: after removal of
special tokens, the continuation must be exactly one in-range decimal integer
apart from surrounding whitespace. Verbose, truncated, or otherwise malformed
outputs are retained as `invalid`, separate from valid wrong counts.

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

After deterministic behavior generation finishes, the N=10 prompt label is
joined back to every saved span-end and span-mean trajectory by panel and
seed. Supplemental confirmation plots mark complete PCA trajectories by the
actual correct/wrong/invalid output, and per-seed probe/curve residual tables
carry the same labels. This is a descriptive association; the label is not
used to refit the PCA basis or choose the primary layer.

### Answer-query PCA sensitivity

The completed `answer_query_all_layers_v1` capture saves the full residual at
the prompt-final `Total:` query before the first answer token is generated.
It stores every zero-based post-block layer: Qwen L0--L35 and Gemma L0--L41.
For each model the grid is four V4 panels times 20 discovery seeds times ten
gold counts, or 800 restartable float16 NPZ shards. It does not interpolate
layers and does not retain full-sequence hidden states. At each saved layer the
report fits two six-component PCA bases on V4.1 discovery prompts:

- `all`: all 200 rows (20 seeds times 10 counts);
- `correct_only`: only rows whose complete greedy continuation is strictly
  correct.

Both bases then project the same 800 saved discovery states (four panels times
20 seeds times 10 counts). The interactive `correct`/`wrong`/`invalid` switch
filters displayed points after projection and therefore never refits PCA. Fit
cohort EVR is reported only as an in-cohort diagnostic. The directly
comparable sensitivity quantity is common-set variance capture, evaluated on
all 200 V4.1 rows:

```text
capture_k(B) = sum_{j=1..k} Var(X_all B_j) / sum_d Var(X_all,d)
```

Here `B_j` is PCA direction `j` from either fit and `X_all` is the common
all-row V4.1 evaluation matrix. Count centroids and within-count seed RMS are
also computed on this same common set. `seed noise / count signal` is the RMS
distance from each row to its count centroid divided by the RMS distance of
the ten count centroids from their grand centroid. Finally, centroid-distance
correlation compares the 45 pairwise distances among count centroids under a
candidate basis with those under the all-row basis. Per-count correct-only
support is always shown because a missing high-count class makes that fit a
sensitivity check rather than a replacement primary analysis.

The all-layer overview reports four discovery-only quantities per layer:

1. seed-grouped CV R2 for a Ridge count probe using PC1--PC3;
2. cumulative PC1--PC3 explained-variance ratio;
3. the fraction of full-space count-centroid signal retained by PC1--PC3;
4. seed compactness `C = 1 / (1 + leave-one-seed-out noise / count signal)`.

The `probe-optimal` answer layer maximizes the first quantity. The
`manifold-display` layer first must lie within 0.02 of that optimum and then
maximizes `M3 = EVR3 * count-signal capture * C`. These are descriptive,
discovery-only layer-selection rules, not causal tests.

### Joint prompt/answer coordinates

Prompt-reading and answer-query trajectories are never overlaid after fitting
separate PCA bases. For the same model, V4 panel, discovery seed, layer, and
count `k`, the paired states are:

- the state after occurrence `k` in that seed's N=10 prompt; and
- the prompt-final `Total:` query state in that seed's N=`k` prompt.

A joint PCA is fit to the concatenated V4.1 state vectors. The raw view retains
the fixed token-role offset. The role-centered sensitivity view first removes
the separate V4.1 grand mean for prompt and answer states, then fits one common
basis. Linear CKA, the correlation of the 45 pairwise centroid distances, and
successive-step cosine are also computed in full residual space so the
comparison does not depend on PCA axis signs or rotations. Similar geometry is
compatible with a shared count organization but does not establish literal
state transport; that claim requires the donor-patching experiment.

### Interpretation boundary

In v4.1, occurrence index, content identity, and absolute position are
one-to-one. A clean curve there establishes only a stable aligned trajectory.
Position invariance requires v4.2; deconfounding fixed content from index
requires v4.3; evidence for content-general counting requires v4.4. A probe
establishes decodability, not causal use.

## Prompt-counter write-side attention

This analysis asks whether the attention row available when the model has just
read the `n`th realistic record becomes more diffuse as `n` grows, and whether
that dispersion covaries with hidden-counter noise. The query site is always
the final token of the current needle. Two separate key reductions are kept:

```text
endpoint mass m_j = attention to the final key token of historical needle j
span mass     m_j = sum of attention over every key token in needle span j
```

Thus `span mass` is literal probability mass in the complete record, not its
per-token mean and not another query position. A span-mean hidden state averages
multiple token states and has no unique native attention row. Its sensitivity
analysis therefore pairs span-mean hidden noise with the same endpoint-query
row pooled over full needle spans; the report labels this explicitly as a
key-pooling sensitivity.

For one head and occurrence `n`, let `a` be the normalized attention row over
all `K` visible prompt keys and let `p_j = m_j / sum_{r<=n} m_r`. The saved
dispersion statistics are:

```text
row effective tokens   = exp(-sum_t a_t log a_t)
row effective fraction = row effective tokens / K
needle effective count = exp(-sum_{j<=n} p_j log p_j)
relative coverage      = needle effective count / n
```

Absolute effective counts can grow mechanically as more tokens or needles
become visible. `row effective fraction` and `relative coverage` remove those
two denominators and are the primary relative-dispersion checks. Discovery
seeds freeze the top eight heads per model, panel, layer, and key pooling using
mean `(total needle mass * relative coverage)` over occurrences 2--10. The
confirmation analysis also repeats every statistic after averaging all heads
in the layer, so a trend induced only by selecting broad heads is visible.

Hidden-counter noise is calculated in full residual space, separately for
span-end and span-mean states:

```text
noise(s,n) = ||h(s,n) - discovery_centroid(n)||_2
             / RMS_n(discovery_centroid(n) - grand_centroid)
```

Within every confirmation seed, each quantity is regressed on
`x=(n-1)/9`; the reported slope is therefore the estimated complete change
from N=1 to N=10, not the per-needle increment. To avoid a spurious correlation
caused by both variables sharing an N trend, attention and hidden noise are
separately demeaned within each occurrence before Pearson correlation. The ten
confirmation seeds are the inference clusters for bootstrap intervals. This
is an observational association and cannot by itself show that attention
dispersion causes representation noise.

Storage is bounded by evaluating the prefix once with KV cache and requesting
one eager query row at each needle endpoint. Compact per-head metrics and a
10-by-10 occurrence profile are saved; no full 10k-by-10k attention matrix or
full Q/K/V tensor is materialized.

### Completed-run interpretation boundary

In the completed numeric run, absolute effective token/needle counts usually
increase with `n`, while relative needle coverage decreases in every
model/pooling pair. Qwen hidden-counter noise is flat and Gemma noise decreases
in several panels; no pair has a significantly positive N=1-to-10 noise slope.
The data therefore reject a universal observational chain from larger `n` to
more diffuse retrieval to a noisier hidden counter. The restricted positive
same-occurrence association for Gemma span-end remains descriptive and requires
a write-site intervention before any causal wording is licensed. Exact values
and all-head controls are reported in
[`realistic_niah_v4_numeric_results_20260731.md`](realistic_niah_v4_numeric_results_20260731.md).

## Answer-query attention

Let `a(t)` be one head's answer-query attention row, and let `S_i` be active
needle span `i`, with model-token length `L_i`. Three occurrence-level
reductions are analyzed and must not be conflated:

```text
span-end:  m_i = a(last token of S_i)
span-sum:  m_i = sum_{t in S_i} a(t)
span-mean: m_i = span-sum / L_i
```

Span-end asks whether the query reads the same token sites whose hidden states
formed the cleanest occurrence-index trajectories. Span-sum is the literal
fraction of the query row assigned anywhere inside the complete realistic
records (when needle spans do not overlap). Span-mean is per-token attention
density and controls tokenizer-dependent record length; its sum across records
is not a query-row mass.

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

Discovery heads are ranked separately for each reduction by mean `broad
primary` over N=2 through N=10.
Eligibility requires that the layer can see every needle and matched hard
negative on the complete discovery grid, mean needle-minus-negative density is
positive, and needle density exceeds the head's prompt-wide baseline. There is
no fallback to negative-contrast heads. N=1 is excluded because its coverage
is identically one. Confirmation seeds are used for locked correct/wrong
comparisons, never head selection. A discovery-seed bootstrap additionally
reports each candidate's top-k selection frequency and rank variability, while
cross-panel and cross-pooling top-k Jaccard scores quantify rank stability. For
`span_sum`, mass, coverage, primary score, occurrence profiles, and omission
diagnostics use literal span sums, but the matched-negative contrast and
enrichment eligibility gate reuses `span_mean` density so longer records do not
win merely because they contain more tokens.

The 10k-token prefix is evaluated once with an efficient KV cache. Only the
single final answer-query token is evaluated with eager attention. The code
therefore materializes an `heads × 1 × key_length` row per layer, never a
10,000 × 10,000 attention matrix. Local-attention layers are scored over
their realized key window and record that window's absolute start.

For every prompt, the complete answer-query row from every head is retained
as an uncompressed float16 NPZ shard. Each layer is a separate array because
Gemma 4 local and global layers have different key-axis lengths. The shard
also records absolute key starts, layer types, query position, and sequence
length. The raw capture is outcome-agnostic. A separate greedy pass saves
actual continuations, token IDs, strict correct/wrong/invalid labels, signed
count error, and decoding provenance. `attention-analyze` joins those labels
by `stimulus_id`, compares discovery-selected heads on confirmation seeds, and
uses a seed-cluster bootstrap for count-adjusted wrong-minus-correct effects.
For an undercount of `k`, the bottom-`k` occurrences in the selected-head
ensemble are reported only as *attention-implied missed candidates*: a scalar
answer does not reveal which particular record was internally omitted.
The nested family gives a second, more targeted diagnostic: N-1 to N toggles
exactly the Nth slot into a needle, so the analysis compares attention to this
new occurrence when the greedy prediction does versus does not increment by
one. This tests a missed-new-evidence hypothesis without claiming that
attention alone establishes the internal cause of an error.
Full-sequence attention matrices and full Q/K/V tensors are not materialized.

## Causal tests

All registered causal outcomes use the actual deterministic continuation after
the same final `Total:` query. The strict parser records the generated integer,
correct/wrong/invalid label, signed count shift, absolute-error change, target
hit, and raw generated token IDs. This handles multi-token `10` without a
single-token approximation. Candidate probability and first-token logits are
not causal outcome labels.

Discovery seeds select heads and fit steering centroids. All intervention
effect estimates use confirmation seeds and are stratified by the receiver's
baseline greedy outcome. Every expensive stage writes restartable per-example
or per-family shards before producing aggregate tables.

### Head ablation

For each panel and each `span_end`/`span_mean` broad-head ranking, the top 1,
2, 4, and 8 discovery-ranked pre-`o_proj` head slices are zeroed. Two scopes
are registered: the one-shot final prompt query and the global head across
prompt prefill plus decoding. Each set is compared with three deterministic,
same-layer random-head sets on confirmation seeds. The top-N dose response and
paired seed-cluster bootstrap are saved separately for baseline-correct and
baseline-wrong prompts.

Ablation can demonstrate necessity but can also introduce distribution shift.
The layer-matched controls and top-N dose response are required for
interpretation.

### Residual-stream patching, removal, and restoration

Adjacent same-seed prompts differ at one known nested slot. A higher-count
donor supplies an active needle for insertion/restoration; a lower-count donor
supplies the matched inactive hard-negative state for removal/ablation. Three
sites are registered:

1. the final answer-query residual, patched at one layer;
2. the toggled needle's final token;
3. the complete toggled token sequence, copied token by token.

Needle-end and full-span interventions are evaluated both at one layer and by
clamping the matched donor states at every layer from a selected start layer
through the final block. Full-span patches require equal donor/receiver model
token lengths; mismatches are retained as explicit skipped rows. `span_mean`
is a descriptive representation summary and is not broadcast back into token
states.

### Geometric steering

At five registered decoder depths, discovery-only centroids
`mu[variant, layer, count]` are fit from the post-block residual at the final
answer query. Confirmation receivers are tested with four V10-style maps:

- full centroid transplant: `h' = mu_target`;
- residual-preserving delta: `h' = h + mu_target - mu_receiver`;
- straight centroid chord interpolation;
- adjacent-centroid polyline interpolation normalized by arc length.

The latter two use `alpha = 0.25, 0.5, 0.75, 1`. Alongside adjacent pairs,
registered non-adjacent pairs test whether the representation follows a
smooth count curve rather than a local binary boundary. Each geometric
perturbation has an orthogonal, norm-matched random-direction control.
Centroid step norms, successive-step cosine, path tortuosity, monotonicity,
greedy count-transport slope, target/path hit rate, and paired geometric-minus-
random effects are saved.

The completed `steering_v2` follow-up uses a narrower but inferentially clean
two-stage design. Four discovery seeds screen 15 single/multi-layer and alpha
plans per model; a worst-panel robust score locks one plan for each protocol.
Ten disjoint confirmation seeds then evaluate only those locks over all four
panels and six directed pairs. Qwen locks L26 and L9+18+26; Gemma locks L31
and L10+20+31, all at alpha 1. The held-out geometric-minus-random aligned
effects are +1.000/+0.992 for Qwen and +1.371/+1.387 for Gemma
(single/multi; all Holm p=0.0039), with every panel CI above zero. Single and
multi are nearly identical and exact target-hit gains remain 5.4--7.5
percentage points, so the result supports stable directional manipulability,
not precise target setting or a multi-layer advantage.

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
    --stage behavior \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}" \
    --cache-dir "${HF_CACHE}" \
    --generation-max-new-tokens 16

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
    --stage attention-analyze \
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

  PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
    --stage geometric-steering \
    --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
    --output-dir "${RUN_ROOT}" \
    --model "${MODEL}" \
    --cache-dir "${HF_CACHE}"
done
```

Use `--variants`, `--seeds`, and `--counts` only for smoke tests or explicitly
labelled partial runs. Pair-based patching stages automatically load all ten
counts for each selected confirmation seed. Geometric steering also loads the
complete discovery split for centroid fitting, even when confirmation seeds
are filtered. Full representation analysis requires all four panels and all
30 seeds. Head ablation requires completed discovery broad-head rankings.

### Budgeted `screen_8h_v1` causal campaign

`scripts/launch_realistic_niah_v4_causal_campaign.sh` runs a targeted screen
whose resolved settings are isolated from the full design by the existing
design hash. It preserves all four panels and all ten confirmation seeds while
using the following reduced estimands:

- ablation: counts 7--10, `span_end`, answer-query scope, top-4/top-8 broad
  heads, and one deterministic layer-matched random set;
- residual transport: pairs 5--6, 7--8, and 9--10 in both directions, the
  exact toggled needle-end token, cumulative-from-layer clamping, and three
  matched depths (25%, 50%, and 75%);
- steering: adjacent pairs 7--8 and 9--10 plus non-local 5--10 in both
  directions, residual-preserving centroid delta, the same three depths, and
  one orthogonal norm-matched random direction.

This is a targeted mechanistic screen rather than a fully powered sweep over
every count, site, method, and layer. It schedules 2,800 intervention
generations per model (5,600 total), compared with 99,600 per model in the
full retained causal design. Discovery-fit head rankings and centroids remain
strictly separated from confirmation evaluation.

For an explicit GPU smoke, causal designs can be narrowed without editing the
registered JSON. For example:

```bash
PYTHONPATH=src python scripts/run_realistic_niah_v4.py \
  --stage geometric-steering \
  --stimuli "${RUN_ROOT}/dataset/stimuli.jsonl" \
  --output-dir "${RUN_ROOT}" \
  --model Qwen3-8B \
  --cache-dir "${HF_CACHE}" \
  --variants v4.1 --seeds 1254 \
  --causal-layers 0 \
  --steering-count-pairs 1:2 \
  --steering-methods centroid_delta \
  --steering-alphas 1 \
  --steering-random-replicates 1
```

Related overrides are `--causal-top-ns`,
`--causal-random-replicates`, `--causal-count-pairs`,
`--ablation-scopes`, `--ablation-poolings`, `--residual-patch-sites`, and
`--residual-patch-protocols`. Because the resolved settings are hashed into
the design directory, screen, smoke, and full-design shards cannot collide.

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
answer-query rows. `behavior/capture/generation_labels.csv` contains the
actual greedy outputs and strict labels. `attention/analysis/` contains
the original restartable span-end/span-mean pooling shards, discovery
rankings, held-out correct/wrong effects, occurrence-level omission
diagnostics, and figures. For the completed numeric run, the non-destructive
`attention/analysis_span_sum_v3/` sibling adds literal span-sum shards,
end-versus-mean/sum alignment, and three-pooling outcome/omission tables while
preserving the original analysis tree.

Within `causal/`, each intervention family has a versioned directory. Each
resolved CLI/config selection is isolated under `design_<12-char SHA>/`, whose
`design.json` records the exact layers, pairs, heads, controls, seeds, and
generation bound. This prevents a partial smoke shard from being silently
reused in a formal design. Every design directory has a restartable `capture/`
tree, `detail.csv.gz`, and `summary.csv`:

- `generation_head_ablation_v1/` also saves the paired broad-vs-random table;
- `generation_residual_patching_v1/` retains successful and explicitly
  skipped full-span rows;
- `geometric_steering_v1/` contains discovery query-state shards,
  `centroids.npz`, centroid-geometry tables, confirmation generations, and the
  paired geometric-vs-random table.
- `geometric_steering_v2/` contains separate discovery-screen and held-out
  confirmation designs, `selection.json`, `plan_scores.csv`, locked-plan
  detail/summary tables, and strict row-level generations for both protocols.

Every causal detail row includes the strict greedy completion, generated token
IDs, baseline label, intervention label, and
`behavior_metric=strict_greedy_complete_numeric_generation`.

## Validation status required before formal inference

CPU tests cover configuration, four-panel randomization contracts, exact
nesting, prompt spans, broad-attention metrics, actual-generation intervention
hooks, multi-token `10`, centroid chord/polyline geometry, causal metric
accounting, and synthetic representation recovery. Before a formal run, both
registered checkpoints still require one GPU preflight that confirms:

- model loading with the pinned Transformers build;
- exact numeric continuation boundaries, including the two-token `10`;
- finite count logits at 10k context;
- query-only eager attention output shape for every layer;
- agreement diagnostics between the cached eager query and full SDPA logits;
- pre-output-projection head layout;
- answer-query and global head ablation under generation;
- answer-query, needle-end, and full-span residual generation patches;
- one discovery-centroid and confirmation steering smoke family.

Passing that smoke test establishes implementation compatibility, not
scientific validity.
