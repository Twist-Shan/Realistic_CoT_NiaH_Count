# Realistic NIAH V4.4 causal-v2 protocol

This document specifies the expanded V4.4-only causal study requested after
the cross-panel representation analysis. It remains the executable protocol;
completed estimates and interpretation are now maintained in the standalone
[V4.4 causal-v2 dual-model report](../reports/realistic_niah_v4_4_causal_v2_report.html)
and its
[audit companion](realistic_niah_v4_4_causal_v2_report.md).

## Completion status

The formal Qwen3-8B and Gemma4-E4B `k={1,3,5}` campaigns completed on
2026-08-03 at implementation commit
`dd409f2dff82ccd6400dfc3d7704025cb6939940`. Each passed the strict audit with
302/302 checks and zero errors. Their separate FileStream exports and local
archives have matching SHA-256 checksums and byte-identical source/copy
manifests. Do not infer effect sizes from this protocol; use the generated
machine-readable tables and report above.

## Scope and causal questions

The protocol fixes the observational panel to **v4.4**, because the existing
V4.1–V4.4 analysis found that the prompt-side needle-end representation remains
stable after varying position, city-score order, and city-score identity. It
then asks three different questions:

1. **Head-bank ablation (necessity contrast):** does removing the top-n
   discovery-ranked broad-aggregation or first-needle-locator heads at the
   answer query hurt behavior more than removing a random bank with the same
   per-layer allocation?
2. **Residual patching (sample-specific sufficiency):** when a donor and
   receiver differ by k needles, does copying the donor state at prompt needle
   sites or at the final answer query move the receiver output by the intended
   amount?
3. **Geometric steering (population-direction manipulability):** does adding
   the full-dimensional count-centroid displacement at the answer query move
   output in the intended ±k direction more than an orthogonal norm-matched
   direction?

These estimands are not interchangeable. A positive steering effect does not
show that a head bank is necessary; a late cumulative answer patch shows
state sufficiency but does not localize where the count was computed.

## Frozen data grid

The executable registry is
[`configs/realistic_niah_v4_causal_v2.json`](../configs/realistic_niah_v4_causal_v2.json)
and the typed validator is
[`src/realistic_niah_v4/causal_v2.py`](../src/realistic_niah_v4/causal_v2.py).

- Model checkpoints: `Qwen3-8B` and `Gemma4-E4B`, using the immutable V4
  revisions.
- Prompt mode: numeric, non-thinking, 10,000 canonical passage tokens.
- Panel: v4.4 only.
- Counts: N=0,…,10. N=1,…,10 must reproduce the original frozen V4.4 rows
  exactly; only N=0 is new.
- Centroid-fit seeds: 1234–1253.
- Screen seeds: 1254–1258.
- Held-out confirmation seeds: 1259–1263.

The N=0 extension preserves the same ten length-matched nested slots. The
freezer verifies regenerated N=1,…,10 passage hashes, slot schedules, content
permutations, and active spans against the original V4 file before committing
the new grid.

The three registered unordered anchor pairs for each k are:

| k | low anchor | middle anchor | high anchor |
|---:|---:|---:|---:|
| 1 | 0↔1 | 4↔5 | 9↔10 |
| 3 | 0↔3 | 3↔6 | 7↔10 |
| 5 | 0↔5 | 2↔7 | 5↔10 |

The registered distances are therefore `k∈{1,3,5}`: small, medium, and large
semantic changes. Every pair is run in both directions, giving 18 directed
pairs. This prevents an apparent effect from being driven only by low counts,
high counts, or one direction while reducing the dense sweep without reducing
the five screen seeds used for any retained condition.

## Outcome and normalization

Let `r` be the receiver gold count, `t` the semantic donor/target count,
`y0` the receiver's actual unmodified greedy prediction, and `y1` the
intervened greedy prediction. The primary sample-level transport is

```text
normalized transport T = (y1 - y0) / (t - r).
```

Thus `T=1` means the generated change exactly equals the intended semantic
change, `T=0` means no output movement, `T<0` means movement in the wrong
direction, and `T>1` is overshoot. It is deliberately not clipped.

A complementary final-target score is

```text
target conformity C = 1 - |y1 - t| / |t - r|.
```

`C=1` is an exact target hit; `C<0` means the patched output misses the target
by more than the complete receiver-to-target gap. Because baseline models can
already be wrong, both T and C are reported.

Correctness and numeric validity use the complete deterministic greedy
continuation, including every token of multi-token `10`. A continuation that
is not exactly one registered integer in 0,…,10 is invalid. Raw T/C are then
undefined; the strict failure-aware versions contribute zero, and invalid
rate is reported separately. Invalid rows are never silently dropped.

## 1. Answer-query head-bank ablation

### Discovery-only head definitions

For one answer-query attention row and active needle spans `S1,…,Sn`, the mass
on needle i is the literal full-span sum

```text
m_i = sum_{token j in S_i} attention(query, j).
```

Total needle mass is `M=sum_i m_i`. Uniformity is the normalized effective
number of attended needles,

```text
U = exp(-sum_i p_i log p_i) / n,  where p_i = m_i / M.
```

The broad-aggregation score is `M×U`, averaged over v4.4 discovery prompts.
It rewards both large total mass on all needle tokens and an even allocation
across needles. The first-locator score is

```text
m_1 - mean(m_2,…,m_n),
```

again averaged over discovery prompts with at least two needles. Neither rank
uses held-out screen or confirmation outcomes.

### Intervention and random control

Only the final `Total:` answer-query token is intervened on. There is no global
ablation. For each phenotype, head-bank size n is swept through every integer
from 1 to 32. This top-n index is separate from semantic count distance
`k∈{1,3,5}` used by patching and steering.
The selected pre-`o_proj` head-output slices are zeroed only for that query.

Each ranked bank is compared with three deterministic random banks. A random
bank is sampled without replacement from all heads in the same layers and has
exactly the same number of heads per layer as the ranked bank. Sampling from
the complete layer population is unbiased; it may overlap the ranked bank by
chance, and the overlap count is stored in every row. Forbidding the entire
top-32 candidate union would make an exact layer match impossible when ranked
heads cluster in one layer.

The primary curve is ranked-minus-random change in strict greedy accuracy and
absolute count error. This is a bank-level dose-response contrast, not a claim
that any single head is necessary.

## 2. Residual patching

All patches copy **complete hidden vectors**, never a scalar count probe and
never a span mean broadcast across tokens.

### Prompt needle-site patching

For an `r→t` nested pair, the changed slots are exactly
`min(r,t)+1,…,max(r,t)`, so every intervention patches k slots.

- `toggled_needle_end`: copy the last model-token residual of each changed
  slot to the corresponding receiver endpoint.
- `toggled_needle_span`: copy every token-state vector in every changed slot
  to the corresponding receiver span under the frozen mapping below. This is
  a coordinated multi-token patch, not `mean(span)`.

Let a receiver span contain R model tokens and its donor span contain S model
tokens. For R,S>1, receiver position j receives the complete donor vector at

```text
a(j) = floor((2 j (S - 1) + (R - 1)) / (2 (R - 1))),  j=0,...,R-1.
```

This is nearest-neighbor matching in normalized within-span position with
deterministic round-half-up ties. It is monotone, preserves the first and last
token, and is exactly the identity map when R=S. If R=1, the sole receiver
token takes the donor midpoint; if S=1, that sole donor vector is reused at
all receiver positions. No vectors are averaged, interpolated, or mixed. When
S>R, some interior donor vectors can be dropped; when R>S, some donor vectors
can be reused. These counts, the explicit source-index map, the absolute
length difference, and maximum normalized-position error are stored per slot.

The tokenizer-only preflight covers all ten screen/confirmation seeds and all
18 directed pairs for each model before baseline extension or intervention
generation. It separately reports exact rows and remapped rows and fails only
if a registered mapping is unsupported. Thus Qwen's equal-length spans reduce
to the original positionwise copy, while Gemma's small tokenizer-dependent
length differences remain visible rather than being silently dropped.

Both sites have:

- `single_layer`: replace the chosen state(s) only after decoder block L;
- `cumulative_from_layer`: clamp clean donor state(s) after every block from L
  through the final block.

The matched controls are clean self-patching at the same positions and layers.

### Answer-query patching

At the final prompt token in `Total:`, the donor's complete answer-query
residual replaces the receiver state. Every decoder layer is swept under:

- a one-layer replacement at L;
- the registered multi-layer definition, a cumulative donor clamp from L
  through the final block.

No other answer multi-layer definition is part of causal-v2. Controls are
self-patching and a same-count state from another seed. Single-layer results
carry the localization interpretation. Late cumulative results are sufficiency
tests and must not be described as locating the computation at L.

## 3. Geometric steering

For each decoder layer, full-dimensional answer-query centroids are fit on the
20 centroid-fit seeds for N=0,…,10. For receiver r and target t, steering uses

```text
h'_L = h_L + (mu_{L,t} - mu_{L,r}).
```

This moves the complete residual by the empirical count-centroid displacement;
it is not a donor-state patch and not a centroid transplant `h'=mu_t`.
Every layer and all registered ±k pairs are screened. The matched control is
an orthogonal random vector with exactly the same norm at the same layer.

Stable singleton layer/k conditions receive five held-out seeds. For each k
with at least two stable layers, one frozen multi-layer plan applies the same
layer-specific centroid displacement simultaneously at all stable layers. The
multi plan is selected after the screen and therefore is reported as a
five-held-out-seed estimate, not a ten-seed screen-plus-confirmation estimate.

## Stability screen and confirmation

Selection is applied independently to exact `site × protocol × layer × k`
patch conditions, or `layer × k` steering conditions. A condition passes only
if all of the following hold on seeds 1254–1258:

- all five screen seeds are present and at least four have positive
  treatment-minus-control transport;
- mean effect is positive in both increase and decrease directions;
- at least two of the three anchor pairs have positive mean effect;
- overall mean control-adjusted transport is at least 0.15;
- treatment transport-valid rate is at least 95%, meaning both the clean
  receiver prediction `y0` and intervened prediction `y1` are legal integers.

The immutable selection JSON is written before confirmation. Every passing
exact condition, not only the best one, receives seeds 1259–1263. The primary
confirmation estimate uses these five held-out seeds only. A separately
labelled ten-seed screen-plus-held-out estimate is retained as a secondary
precision summary and must not be described as independent confirmation.
Tables use a seed-cluster bootstrap confidence interval, an exact seed-level
sign-flip test, and Holm correction within evidence scope. Failed screen
conditions cannot enter final confirmation statistics.

## Formal request accounting

Let D be the decoder-block count (36 for Qwen, 42 for Gemma).

| Stage | Rows per model before confirmation |
|---|---:|
| Answer-query ablation | `5 seeds × 4 counts × 2 banks × 32 top-n sizes × 4 conditions = 5,120` |
| Prompt patch screen | `5 × 18 pairs × D × 2 sites × 2 protocols × 2 conditions = 720D` |
| Answer patch screen | `5 × 18 × D × 1 site × 2 protocols × 3 conditions = 540D` |
| Steering screen | `5 × 18 × D × 2 conditions = 180D` |
| Centroid captures | `20 seeds × 11 counts = 220` NPZ shards |

The pre-confirmation intervention totals are therefore 56,960 Qwen rows and
65,600 Gemma rows. Confirmation cost depends on how many exact layer/k
conditions pass the frozen screen. This is substantially larger than the old
8-hour causal screen; the launcher is restartable at one seed/pair or
seed/count shard and never overwrites completed hash-matched shards.

### Logical rows versus executed 10k-token generations

The table above is the inferential table size, not the number of distinct GPU
prefills. Causal-v2 preserves every logical control row but does not recompute
mathematically identical controls:

- `self_patch` replaces a receiver state with the same receiver state. The
  runner executes this hook at early, middle, and final sentinel layers for
  every site/protocol on one maximum-k pair. It aborts if any greedy token
  differs from the registered baseline. After that identity preflight, the
  remaining dense self rows reuse the exact registered baseline completion.
- At `answer_query`, a `same_count_seed` intervention is determined by
  receiver prompt, same-count source prompt, protocol, and layer. It does not
  depend on the nominal donor or k. The runner executes each such intervention
  once, caches the completion, and recomputes the pair-specific normalized
  transport after reuse. Counting repeated donor labels as new GPU trials
  would be pseudoreplication.
- Donor-transport rows are never synthesized or dropped: every registered
  seed, direction, anchor, k, layer, site, and protocol still receives its own
  real intervention generation.

Consequently, prompt patching executes `360D` donor generations per model
rather than `720D` GPU generations, plus 12 identity-preflight generations.
Answer patching executes `180D` donor generations plus at most `100D` unique
same-count controls, plus 6 identity-preflight generations, rather than `540D`.
The logical detail tables retain their original `720D` and `540D` sizes and
record `generation_executed` plus `generation_reuse_mode` for every row. On the
same A100-SXM4-80GB used by the earlier V4 runs, the measured-throughput
estimate is about 29--31 raw GPU-hours before confirmation; model loading,
captures, I/O, and auditing give a practical budget of approximately 32--38
hours on one GPU. Confirmation adds work only for exact retained conditions
that pass the frozen stability rule.

## Run, analyze, and audit

The one-command formal launcher takes an existing V4 run root, the GPU Python,
HF cache, original V4 stimuli, and a full-span attention source. `AUTO` resolves
each model's existing `numeric/attention/capture/attention_capture_index.jsonl`;
a capture directory, index JSONL, or consolidated raw-metric CSV is also valid.
The commonly used `pooling_head_detail.csv.gz` is not a valid source because it
does not retain the per-needle `needle_span_masses` needed for first-locator
ranking.

```bash
scripts/launch_realistic_niah_v4_causal_v2.sh \
  /path/to/run_20260731_v4_numeric_presentation_v3 \
  /path/to/venv/bin/python \
  /path/to/hf-cache \
  /path/to/run/dataset/stimuli.jsonl \
  AUTO
```

Every stage has a content-addressed design directory. The hash includes the
stimulus, configs, frozen implementation sources, model, split, pairs,
conditions, and controls. Per-shard SHA256 values and a completion marker are
written atomically.

After the formal stages, build all-layer tables and Aurora figures with
[`scripts/analyze_realistic_niah_v4_causal_v2.py`](../scripts/analyze_realistic_niah_v4_causal_v2.py).
The analysis writes the top-1…32 ablation curve and direction-separated
layer×k heatmaps for endpoint patching, full-span patching, answer patching,
and steering.

The final strict audit is:

```bash
PYTHONPATH=src python scripts/audit_realistic_niah_v4_causal_v2.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --model Qwen3-8B \
  --stimuli /path/to/run/dataset/causal_v2/stimuli_v4_4_causal_v2.jsonl \
  --causal-config configs/realistic_niah_v4_causal_v2.json \
  --require-confirmation
```

It checks unique formal design roots, design and implementation hashes, shard
checksums, exact count/pair/seed grids, answer-query-only ablation, top-1…32
coverage, random layer matching, all decoder layers, k changed slots,
single/cumulative semantics, registered full-span mapping diagnostics, strict
numeric metrics, selection hashes, and the five held-out seeds for every
selected condition.
