# Realistic NIAH V4 numeric non-thinking results

This note records the completed V4 run
`run_20260731_v4_numeric_presentation_v3`. The descriptive representation and
attention analyses are followed by the targeted `screen_8h_v1` causal
campaign and the denser `answer_query_dense_v1` residual-patching follow-up.
Both are complete, but smaller than the fully registered causal grid. Detailed
intervention estimands, seed-cluster intervals, matched controls, and audit
results are in
[`realistic_niah_v4_causal_screen_20260801.md`](realistic_niah_v4_causal_screen_20260801.md).

After the cross-panel needle-end robustness check, the focused synthesis fixes
the panel to V4.4. Its standalone interactive report is
[`realistic_niah_v4_4_mechanism_report.html`](../reports/realistic_niah_v4_4_mechanism_report.html),
with an explicit treatment/control inventory and V4.4-only estimates in
[`realistic_niah_v4_4_mechanism_report.md`](realistic_niah_v4_4_mechanism_report.md).

## Registered design

- Models: `Qwen3-8B` and `Gemma4-E4B`.
- Context length: 10,000 canonical passage tokens.
- Counts: decimal `1` through `10`.
- Panels: v4.1 fixed; v4.2 position-relaxed; v4.3 city-score-order-relaxed;
  v4.4 city-score-content-relaxed.
- Seeds: 30 paired seeds per panel and count. Seeds 1234--1253 are discovery;
  seeds 1254--1263 are confirmation.
- Behavior label: the parsed complete deterministic greedy continuation after
  the prompt-final `Total:` query. This includes the full multi-token sequence
  for `10`; no first-token candidate probability is used.

## Completion audit

| Artifact per model | Qwen3-8B | Gemma4-E4B |
| --- | ---: | ---: |
| Behavior rows | 1,200 | 1,200 |
| Unique stimulus IDs | 1,200 | 1,200 |
| Representation capture shards | 120 | 120 |
| Raw answer-query attention tensors | 1,200 | 1,200 |
| Non-empty attention-analysis tables | 12 | 12 |
| Non-empty attention figures | 23 | 23 |
| Non-empty labeled representation outputs | 7 | 7 |
| Raw attention-capture bytes | 28,362,547,387 | 1,783,768,868 |

All 54 absolute artifact references found in the five manifests for each model
resolved to existing, non-empty files. The complete remote run is about 33 GB;
Qwen occupies about 30 GB and Gemma about 2.6 GB. The apparent size difference
comes primarily from the registered query-row storage shape, not from a
difference in the number of examples.

A subsequent Qwen-only partition analysis added 16 tables/manifests and seven
figures under `Qwen3-8B/numeric/attention/analysis/partitioning`. It reads the
same saved answer-query rows and does not change the registered capture.

## Behavior

Every generated answer was format-valid. The failures below are therefore
numeric errors rather than parser or formatting failures.

| Model | v4.1 confirmation | v4.2 confirmation | v4.3 confirmation | v4.4 confirmation | All 1,200 rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 35% | 45% | 48% | 44% | 45.92% |
| Gemma4-E4B | 43% | 37% | 38% | 37% | 37.58% |

The dominant behavioral transition is count-dependent, not panel-dependent.
On the 400 confirmation examples per model, accuracy by true count is:

| Count | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 92.5% | 100% | 82.5% | 95% | 35% | 17.5% | 0% | 7.5% | 0% | 0% |
| Gemma4-E4B | 100% | 100% | 100% | 45% | 25% | 10% | 2.5% | 2.5% | 0% | 2.5% |

At counts 9 and 10, Qwen undercounts every confirmation example. Gemma
undercounts every count-9 example and 97.5% of count-10 examples. Thus overall
accuracy alone hides a sharp capacity/failure boundary around counts 4--6.

## Prompt-reading representations

Layers were selected only from v4.1 discovery data. The registered
`probe-optimal` layer maximizes grouped-seed full-space Ridge CV R2: Qwen L1
(`span_end`) and L0 (`span_mean`), and Gemma L22 (`span_end`) and L0
(`span_mean`). The separately selected `manifold-display` layer must remain
within 0.02 of the best full-space CV R2 and then maximizes
`M3 = EVR3 * PC1--3 count-signal capture * seed compactness`. This second rule
selects a three-dimensional view that is faithful to the count signal rather
than merely choosing the best probe. The same frozen protocol was then
evaluated on confirmation seeds and progressively relaxed panels.

| Model / pooling | Probe-optimal | Manifold-display | Display-layer PCA3 CV R2 | EVR3 | PC1--3 count-signal capture | Seed compactness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen span-end | L1 | L8 | 0.835 | 0.725 | 0.962 | 0.555 |
| Qwen span-mean | L0 | L35 | 0.891 | 0.883 | 0.976 | 0.654 |
| Gemma span-end | L22 | L9 | 0.369 | 0.610 | 0.926 | 0.426 |
| Gemma span-mean | L0 | L41 | 0.844 | 0.898 | 0.984 | 0.692 |

The probe and display layers are therefore deliberately different in all four
cases. In particular, a high-dimensional count probe can peak early even when
the cleanest three-dimensional manifold is late.

### Confirmation metrics at the registered primary layers

| Model / pooling | Metric | v4.1 | v4.2 | v4.3 | v4.4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen span-end L1 | R2 | 0.989 | 0.926 | 0.874 | 0.866 |
| Qwen span-end L1 | MAE | 0.216 | 0.608 | 0.792 | 0.822 |
| Qwen span-mean L0 | R2 | 0.999 | 0.999 | 0.691 | 0.596 |
| Qwen span-mean L0 | MAE | 0.046 | 0.065 | 1.264 | 1.489 |
| Gemma span-end L22 | R2 | 0.954 | 0.914 | 0.890 | 0.916 |
| Gemma span-end L22 | MAE | 0.450 | 0.671 | 0.756 | 0.661 |
| Gemma span-mean L0 | R2 | 0.997 | 0.997 | -0.168 | -0.177 |
| Gemma span-mean L0 | MAE | 0.118 | 0.130 | 2.576 | 2.611 |

Paired confirmation-seed bootstraps show that span-end probe MAE first worsens
at v4.2 in both models. Span-end curve residual relative to signal first has a
strictly positive paired 95% interval at v4.2 for Qwen and v4.3 for Gemma.
Span-mean count decoding remains nearly perfect through v4.2 but collapses at
v4.3, especially for Gemma.

The defensible interpretation is therefore narrower than "a clean scalar
counter." A count-decodable signal persists at the needle end through v4.4,
but it becomes less accurate and seed-sensitive as controls are relaxed. The
span-mean result is strongly dependent on the fixed city-score organization and
does not survive the v4.3 relaxation. PCA alone should not be treated as causal
or as proof of an abstract counter.

### Interactive representation report

[`reports/realistic_niah_v4_representation_report.html`](../reports/realistic_niah_v4_representation_report.html)
is the self-contained, v10-style V4 result report. Its argument runs from the
full confirmation behavior boundary to representation geometry, attention-head
phenotypes, omission diagnostics, and causal necessity/sufficiency tests. It
adds Aurora-native 2D figures, explicit metric formulas, axis/estimand captions,
and a current-conclusion block after every section. The interactive 3D count
manifold keeps individual seed points and split-specific 1--10 centroid paths,
and allows switching model, span-end/span-mean pooling, v4.1--v4.4,
discovery/confirmation, actual greedy output strata, and any displayed axes
among PC1--PC6. A separate all-layer answer-query 3D view uses Qwen L0--L35 and
Gemma L0--L41 at the prompt-final `Total:` query, switches
`correct`/`wrong`/`invalid` final greedy outcomes, and compares an all-V4.1 PCA
basis with a correct-only sensitivity basis. At every selected layer, both
bases project the same 800 saved states per model. The report compares
fit-cohort EVR, common all-V4.1 variance capture, count-centroid trajectory
geometry, and within-count seed scatter; it reports per-count correct-only
support because high-count correct rows can be absent.

Each model/pooling basis is fit only on v4.1 discovery states at that pooling's
registered primary layer, then reused across all four variants. Bases are not
shared across models or poolings. Because the representation capture consists
of N=10 trajectories, correct/wrong coloring uses the actual greedy N=10
output for the whole trajectory. This stratum is severely imbalanced: Qwen has
no correct N=10 confirmation trajectory; Gemma has one in v4.1 and none in
v4.2--v4.4. The outcome switch is therefore an audit, not a powered group
comparison.

### Answer-query all-layer selection and PCA fit-cohort sensitivity

The restartable answer-query capture contains every zero-based post-block
layer: Qwen L0--L35 and Gemma L0--L41. Each layer contains all 800 rows
(4 panels x 20 discovery seeds x 10 counts). Layer selection uses only the 200
V4.1 discovery rows. The answer-query `manifold-display` layers are Qwen L29
and Gemma L37; the independently selected PCA3 `probe-optimal` layers are Qwen
L35 and Gemma L39.

At each layer, `all` fits on all 200 V4.1 discovery rows; `correct_only` fits on
the strictly correct subset (89 Qwen rows and 76 Gemma rows). Both bases project
the same 800 saved states, and every geometry comparison below is evaluated on
the common set of all 200 V4.1 rows. Thus fit-cohort EVR and common-set capture
are deliberately different quantities.

| Model / display layer | Fit EVR PC1--3, all / correct-only | Common-set capture PC1--3, all / correct-only | Seed-noise / count-signal, all / correct-only | Correct-only centroid-distance correlation to all-fit | Correct-only support per count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen L29 | 0.769 / 0.798 | 0.769 / 0.713 | 0.318 / 0.320 | 0.977 | 0--20 |
| Gemma L37 | 0.843 / 0.880 | 0.843 / 0.824 | 0.268 / 0.260 | 0.999 | 1--20 |

The late answer-query count-centroid geometry is therefore not an artifact of
fitting wrong rows. Correct-only remains a sensitivity analysis rather than the
primary basis: Qwen has zero correct fit support for one count, while Gemma's
least-supported count has only one row.

### Joint prompt-reading and answer-query geometry

Prompt and answer states are overlaid only after fitting one common basis to
paired V4.1 states: prompt occurrence `k` from an N=10 trajectory is paired
with the answer-query state from the N=`k` prompt at the same model, seed,
panel, and layer. The primary sensitivity view removes the separate prompt and
answer grand means before fitting, so a fixed token-role offset cannot create
an apparently shared count curve. Full-space linear CKA, the correlation of the
45 inter-count centroid distances, and the cosine between adjacent count steps
do not depend on PCA axis signs or rotations.

| Model / pooling / answer display layer | Role-centered linear CKA | Centroid-distance correlation | Adjacent-step cosine | Answer/prompt count-signal scale | Raw role-offset / count-signal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen span-end L29 | 0.798 | 0.761 | -0.004 | 0.995 | 3.383 |
| Qwen span-mean L29 | 0.848 | 0.829 | 0.042 | 1.835 | 3.847 |
| Gemma span-end L37 | 0.798 | 0.839 | 0.017 | 1.494 | 2.290 |
| Gemma span-mean L37 | 0.851 | 0.901 | 0.032 | 3.078 | 2.259 |

After role centering, prompt and answer trajectories share a strong global
count organization, especially under span-mean pooling. Their adjacent steps
are nevertheless nearly orthogonal, and the raw role offset is 2.26--3.85
times the count-signal scale. The licensed conclusion is therefore "related
global geometry," not "the same counter state is copied from prompt to
answer."

### Prompt-counter write-side attention dispersion and hidden noise

At each current needle end, the saved query row is pooled in two ways over
historical needles: `needle_end` keeps one key token per record, while
`needle_span_sum` sums literal attention over every token in each record. For a
normalized full row `a`, `row N_eff = exp(H(a))`; for normalized occurrence
masses `p_j`, `needle N_eff = exp(H(p))` and `relative coverage = needle
N_eff/n`. Hidden noise is the full-space distance to the matching discovery
count centroid, normalized by discovery count-signal RMS. Every number below is
the median across v4.1--v4.4 of the confirmation-seed regression's complete
N=1-to-10 change. Correlations first remove the occurrence-`n` mean from both
variables and use seed-cluster bootstrap intervals.

| Model / hidden pooling / display layer | Row N_eff change | Row effective-fraction change | Needle N_eff change | Relative-coverage change | Hidden-noise change | Same-n coverage--noise r | Panels with positive / negative noise CI | Panels with positive r CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen span-end L8 | +51.00 | -0.0016 | +3.33 | -0.449 | +0.012 | +0.432 | 0 / 0 | 1 / 4 |
| Qwen span-mean L35 | +116.45 | -0.0323 | +0.74 | -0.668 | -0.024 | +0.097 | 0 / 0 | 0 / 4 |
| Gemma span-end L9 | +0.80 | +0.0016 | +0.01 | -0.687 | -0.250 | +0.209 | 0 / 3 | 3 / 4 |
| Gemma span-mean L41 | +784.38 | +0.0283 | +6.56 | -0.232 | -0.064 | +0.110 | 0 / 2 | 1 / 4 |

The positive absolute effective-count changes mean that attention is spread
over more tokens or needles as the prompt grows, but in every pair the number
of effectively covered needles grows much more slowly than `n`, so relative
coverage falls. The all-head controls preserve this conclusion (Qwen
span-end/span-mean relative-coverage medians -0.611/-0.663; Gemma is identical
to its eight-head bank at these layers). Crucially, no model/pooling has a
significantly positive hidden-noise slope: Qwen is flat and Gemma becomes less
noisy in several panels. A limited positive same-`n` coverage--noise
association survives in Gemma span-end, but it is not the proposed across-`n`
chain and is not causal. The completed data therefore do **not** support the
general claim that larger `n` makes retrieval more diffuse and thereby makes
the counter noisier.

For `span_mean` hidden states, there is no single native attention row because
the hidden state averages several query positions. The analysis explicitly
pairs span-mean hidden noise with the same needle-end query row and full-span
key pooling as a sensitivity check; it must not be read as attention emitted by
a hypothetical mean query.

All V4 and later visualizations use the Aurora palette registered in the report
builder and repository README. Count colors are ordered blends of the supplied
Aurora anchors; model, panel, pooling, control, and zero-reference colors keep
fixed meanings across figures.

## Answer-query attention

The expanded analysis keeps three non-interchangeable occurrence reductions:
`span_end` is one final-token weight, `span_sum` is the literal attention sum
over every token in the complete needle span, and `span_mean=span_sum/L_i` is
per-token density. Only the sum over `span_sum` occurrences is the fraction of
the query row assigned to all needle-span tokens. Discovery-only rankings are
stable under 500 seed bootstraps, but the reductions can expose different
head banks. Span-sum uses literal mass for breadth and omission diagnostics;
its matched-negative eligibility gate remains length-normalized.

- Qwen `span_end`: L29H3 is rank 1 in all four panels with top-8 selection
  frequency 1.0. Its coverage is only about 0.22 and its effective number is
  about 1.04, so it concentrates on roughly one needle rather than aggregating
  uniformly across all needles.
- Qwen `span_mean`: L27H18 is rank 1 in v4.1, v4.3, and v4.4; L28H19 is rank 1
  in v4.2. Coverage is about 0.79--0.86 and the effective number is about
  4.8--5.2, which is much closer to broad aggregation.
- Qwen `span_sum`: the same rank-1 identities are recovered; literal total
  needle-span mass is 0.64--0.70, coverage is 0.79--0.86, and effective number
  is 4.77--5.17.
- Gemma `span_end`: the top head is L5H2 in v4.1 and L29H6 in v4.2--v4.4.
  Coverage is about 0.85--0.89 and the effective number is about 5.0--5.3.
- Gemma `span_mean`: L35H2 is rank 1 in every panel. Coverage is about
  0.76--0.80 and the effective number is about 4.6--4.8.
- Gemma `span_sum`: L35H2 is again rank 1 in every panel; literal needle-span
  mass is 0.41--0.43, coverage is 0.76--0.80, and effective number is
  4.58--4.83.

Span-mean and span-sum select the same top-8 head set in every model and panel,
which is expected because these realistic record spans have nearly equal token
lengths. Endpoint rankings are not interchangeable with full-span rankings.
For Qwen, end-versus-sum Spearman correlation is only 0.624--0.626 and their
top-8 sets have zero overlap in every panel. For Gemma the correlation is
0.772--0.797 and top-8 overlap ranges from 2/8 to 5/8. The full-span result
therefore supports a distinct record-reading bank, especially in Qwen, rather
than treating the endpoint as a lossless proxy for all needle tokens.

### Qwen span-end: full candidate bank and positional partitioning

The rank-1 result does not characterize the complete head population. We
therefore evaluated every discovery-eligible Qwen `span_end` candidate (212 in
v4.1, 226 in v4.2/v4.3, and 225 in v4.4) on the exact saved N=10 attention
rows. Each head was measured at every needle endpoint, over the full needle
span, and over 20 equal-width query-depth bins. The following post-hoc rules
are descriptive rather than causal or preregistered:

- a global endpoint aggregator has endpoint effective number at least 6 and no
  single occurrence with mean normalized share above 0.25;
- a partition-local endpoint aggregator is not global, has at least two
  needles in its winning depth quartile, local effective fraction at least
  0.8, and at least half of its full-row mass in one depth quartile; and
- an occurrence selector has endpoint effective number at most 2 and the same
  winning occurrence in at least 80% of examples.

| All-30-seed result | v4.1 | v4.2 | v4.3 | v4.4 |
| --- | ---: | ---: | ---: | ---: |
| Global endpoint-aggregator heads | 35 | 24 | 24 | 24 |
| Global-bank raw effective number / 10 | 9.59 | 9.06 | 9.14 | 9.04 |
| Partition-local endpoint-aggregator heads | 16 | 5 | 5 | 5 |
| Partition-local-bank raw effective number / 10 | 6.58 | 7.10 | 8.24 | 7.31 |
| Occurrence endpoint selectors | 63 | 69 | 66 | 69 |
| Selector-bank raw effective number / 10 | 1.20 | 1.18 | 1.20 | 1.20 |

The central result is therefore that broad aggregation is distributed across
many heads, even though the highest-ranked head is not itself a broad
aggregator. On confirmation seeds alone, the global class contains 34, 32, 33,
and 31 heads and its raw effective number is 9.63, 9.18, 9.21, and 9.11. Thirteen
heads remain global aggregators in every one of the eight panel-by-split cells:
L6H12, L8H19, L9H19, L9H27, L10H22, L13H16, L15H4, L15H8, L16H15,
L17H22, L18H22, L19H15, and L22H16. L6H12 has the largest mean endpoint
mass among this stable set; L13H16 and L17H22 are also high-mass candidates
for controlled ablation. Layer and head indices are zero-based throughout.

Partition-local specialization is present but less stable than global
aggregation. In the all-seed summaries, L8H16 is prefix-local in v4.1, v4.2,
and v4.4; L13H17 is prefix-local in v4.2--v4.4; and L14H15 is suffix-local in
v4.2--v4.4. Exact local-phenotype identity agrees between discovery and
confirmation for 12 heads in v4.1 but only 1, 0, and 2 heads in v4.2--v4.4.
Thus the evidence for a stable global aggregation bank is strong, whereas a
fixed seed-invariant partition circuit remains only a hypothesis.

L29H3 remains a first-occurrence endpoint selector: its first endpoint receives
about 99.1--99.3% of within-endpoint mass in every panel. When positions vary,
its winning absolute depth bin is stable in only 46.7% of seeds, so it tracks
the earliest occurrence rather than a fixed absolute bin. Its `span_mean`
effective number is 5.30--5.70, showing broader attention inside record spans,
but not uniform aggregation at their endpoint tokens. All heads classified as
strong occurrence selectors also choose occurrence 1; there is no matching
family that cleanly assigns one selector to each later occurrence.

Finally, an eight-head complement-greedy bank reaches effective number
9.46--9.61 when every head profile is normalized to equal weight, but only
3.24--3.62 when raw attention magnitudes are preserved. The full unfiltered
candidate bank is lower still at 2.81--3.19 because the selector class carries
more total endpoint mass and overwhelms the aggregator class. These sums are
attention diagnostics, not a reconstruction of the model computation: heads
have different value vectors and output-projection slices. The causal follow-up
must therefore ablate the stable aggregator bank separately from
L29H3-like selectors and use layer-matched random controls.

Most count-adjusted wrong-minus-correct bootstrap intervals include zero. Of
384 model-specific comparisons (four panels times eight discovery-ranked heads
times four metrics times three poolings), 56 Qwen and 43 Gemma intervals
exclude zero before any multiplicity correction. Broken down by pooling, the
negative/positive counts are Qwen end 14/5, mean 9/9, sum 11/8; Gemma end 3/0,
mean 5/15, sum 4/16. These are sparse, uncorrected outcome associations, so
aggregate needle mass alone does not reliably separate correct from wrong
answers.

## Omission and nested-increment diagnostics

These diagnostics are descriptive correlations on confirmation examples, not
interventions.

- For Qwen undercounts, `span_end` attention to the omitted tail relative to
  the retained prefix averages 0.28--0.37 across panels; the low-attention
  bottom-k set overlaps the omitted tail by 0.38--0.58. The analogous
  `span_mean` tail ratio is 0.79--0.84. Pooled across panels, `span_sum` gives
  tail/prefix 0.807 and bottom-k overlap 0.346, close to span-mean (0.816 and
  0.339) but much less omission-aligned than span-end (0.336 and 0.460).
- For Gemma undercounts, the `span_end` tail ratio is 0.68--0.93 and bottom-k
  overlap is 0.29--0.46. The `span_mean` tail ratio is usually above 1, with
  only 0.11--0.16 bottom-k overlap. Pooled `span_sum` is likewise 1.161 with
  bottom-k overlap 0.139, versus endpoint 0.779 and 0.383.
- When a nested N to N+1 pair registers the increment, attention assigned to
  the newly added needle is higher than when it fails to increment: Qwen
  `span_end` 0.474 versus 0.379, `span_mean` 1.099 versus 0.808, and `span_sum`
  1.096 versus 0.802; Gemma `span_end` 0.824 versus 0.692, `span_mean` 1.324
  versus 0.983, and `span_sum` 1.324 versus 0.983.

The omission hypothesis is consequently most visible for Qwen `span_end`, and
the nested-pair association appears in both models. Neither establishes that
low attention causes the undercount. That question is reserved for the
registered broad-head ablation and needle-state patching sweeps with
layer-matched random controls.

## Causal screen result

The targeted screen and exact-query follow-up completed for both models with
no skipped intervention rows: 640 ablation rows, 720 exact needle-end patch
rows, 800 discovery query states, 1,440 steering rows, and 2,560 exact
answer-query patch rows per model.

- **Mixed ranked-bank ablation is positive.** Relative to layer-matched random
  heads,
  top-8 ablation shifts Qwen counts by -0.331 [95% seed CI -0.413, -0.256]
  and Gemma counts by -2.156 [-2.356, -1.969]; both Holm-adjusted exact
  sign-flip p-values are 0.0078.
- **Exact needle-end transport is null.** Across all tested depths, at most
  2.1% of rows move strictly toward the donor gold, and every direction-aligned
  shift interval includes zero after family-wise correction.
- **Exact late answer-query transport is near-deterministic.** Conditional on
  receiver and donor baseline predictions differing, donor-prediction adoption
  jumps from 1.4% at Qwen L18 to 59.2% [52.0, 66.1] at L26 and reaches 100%
  at L35. Gemma jumps from 0.8% at L20 to 86.5% [82.8, 90.0] at L31 and
  reaches 99.58% [98.75, 100] at L41 when strict-invalid outputs count as
  failures. Every valid eligible final-layer row copies the donor prediction.
- **Late answer-query steering is positive.** Geometric-minus-random aligned
  count shift is +0.958 [+0.808, +1.096] at Qwen L26 and +1.388 [+1.283,
  +1.488] at Gemma L31; both Holm-adjusted p-values are 0.0117. Exact target
  hits remain low, so this is directional manipulability rather than precise
  count setting.

The combined result distinguishes four properties: count information is
decodable at needle endpoints, a discovery-ranked mixed span-end head bank has
a necessary contribution to preserving output magnitude, the exact late query
state is
sufficient to transport the model's computed prediction, and query-state
geometry is directionally steerable. The single toggled endpoint is
nevertheless not sufficient to transport a nested count change across prompts.

Five Gemma rows emit strict-invalid `11<turn|>` rather than an integer in
1--10. They are all the same v4.1 seed-1263, 5←10 family at layers
31/35/38/40/41. The donor baseline `10` has token IDs
`[236770, 236771, 106]`; the patch emits `[236770, 236770, 106]`. This is
consistent with the query patch transferring the first digit `1` while the
next, unpatched autoregressive step generates `1` rather than `0`; it is not
truncation or a failed hook.

## Discovery-locked steering v2

The first steering screen located late readout-sensitive layers but used its
confirmation set for that localization. The v2 follow-up therefore made plan
selection and final inference disjoint. For each model, 15 plans crossed three
single layers, two multi-layer sets, and alpha in {0.25, 0.5, 1}. Four
discovery seeds (1234--1237) selected one single and one multi plan using the
minimum effect across the four V4 panels minus twice the invalid rate. The
locked plans were then evaluated without modification on ten held-out seeds
(1254--1263), six directed count pairs, both protocols, both intervention
arms, and all panels: 960 strict-greedy rows per model.

| Model | Protocol | Locked plan | Held-out aligned effect [95% seed CI] | Moved-toward-target effect [95% CI] | Exact target-hit effect | Holm p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-8B | single | L26, alpha=1 | +1.000 [+0.867, +1.133] | +36.7 pp [+33.3, +40.0] | +7.5 pp | 0.0039 |
| Qwen3-8B | multi | L9+18+26, alpha=1 | +0.992 [+0.838, +1.142] | +37.1 pp [+33.3, +40.8] | +7.5 pp | 0.0039 |
| Gemma4-E4B | single | L31, alpha=1 | +1.371 [+1.275, +1.463] | +48.8 pp [+44.2, +53.3] | +5.8 pp | 0.0039 |
| Gemma4-E4B | multi | L10+20+31, alpha=1 | +1.387 [+1.300, +1.467] | +50.0 pp [+47.1, +53.3] | +5.4 pp | 0.0039 |

Every model/protocol/panel estimate is positive and every panel-level 95% CI
also remains above zero. Single and multi are nearly identical, however:
multi minus single is -0.008 count unit for Qwen and +0.017 for Gemma, and the
design did not randomize a direct protocol contrast. Thus v2 establishes
held-out directional manipulability across panels, but does not establish a
multi-layer advantage or exact target setting. All 7,680 screen plus
confirmation rows are strict-format-valid; no output was dropped. The three
machine-readable summaries are
`reports/realistic_niah_v4_steering_v2_selection.csv`,
`reports/realistic_niah_v4_steering_v2_confirmation.csv`, and
`reports/realistic_niah_v4_steering_v2_panels.csv`.

## Reproducibility and remaining causal scope

The run root is `run_20260731_v4_numeric_presentation_v3`. Each model directory
contains behavior labels, representation captures, raw answer-query attention,
analysis manifests, tables, and figures. Recompute literal full-span attention
mass from the saved rows without rerunning the models or overwriting the
original two-pooling analysis with:

```bash
PYTHONPATH=src python scripts/backfill_realistic_niah_v4_span_sum.py \
  --run-root <run-root> --models Qwen3-8B,Gemma4-E4B
```

This writes the restartable `analysis_span_sum_v3` sibling. Reproduce the Qwen
partition analysis from the saved answer-query rows with:

```bash
PYTHONPATH=src python scripts/analyze_realistic_niah_v4_partitioning.py \
  --stimuli <run-root>/dataset/stimuli.jsonl \
  --run-root <run-root> --model Qwen3-8B --answer-format numeric \
  --count 10 --top-k 8 --partitions 4 --depth-bins 20 \
  --bootstrap-repetitions 10000
```

The command defaults to all discovery-eligible candidates; `--top-k-only`
restricts it to the registered diagnostic subset. The causal implementation
uses actual complete greedy numeric generation and supports:

1. answer-query and global broad-head ablation;
2. answer-query, needle-end, and exact tokenwise full-needle residual patching,
   with single-layer and cumulative-from-layer protocols; and
3. discovery-fit centroid transplant/delta, chord/polyline geometric steering,
   and norm-matched random controls.

Audit the downloaded causal result with:

```bash
PYTHONPATH=src python scripts/audit_realistic_niah_v4_causal.py \
  --run-root <run-root> \
  --output <run-root>/causal_screen_8h_audit.json
```

Audit and reproduce the exact answer-query analysis with:

```bash
PYTHONPATH=src python scripts/analyze_realistic_niah_v4_answer_query_patching.py \
  --run-root <run-root> \
  --output-dir <run-root>/analysis/answer_query_patching_dense_v1 \
  --bootstrap-repetitions 20000
```

The targeted screen supports the causal claims summarized above; descriptive
PCA, probe, and attention results remain non-causal. The unrun remainder is the
larger condition grid, not these completed interventions.
