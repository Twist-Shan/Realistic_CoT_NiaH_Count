# Realistic NIAH V4 numeric non-thinking results

This note records the completed V4 run
`run_20260731_v4_numeric_presentation_v3`. The descriptive representation and
attention analyses are followed by the targeted `screen_8h_v1` causal
campaign and the denser `answer_query_dense_v1` residual-patching follow-up.
Both are complete, but smaller than the fully registered causal grid. Detailed
intervention estimands, seed-cluster intervals, matched controls, and audit
results are in
[`realistic_niah_v4_causal_screen_20260801.md`](realistic_niah_v4_causal_screen_20260801.md).

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

Layers were selected only from v4.1 discovery data by maximum grouped-seed
ridge cross-validation R2. The selected layers were Qwen L1 (`span_end`) and L0
(`span_mean`), and Gemma L22 (`span_end`) and L0 (`span_mean`). The same fitted
protocol was then evaluated on confirmation seeds and progressively relaxed
panels.

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
among PC1--PC6.

Each model/pooling basis is fit only on v4.1 discovery states at that pooling's
registered primary layer, then reused across all four variants. Bases are not
shared across models or poolings. Because the representation capture consists
of N=10 trajectories, correct/wrong coloring uses the actual greedy N=10
output for the whole trajectory. This stratum is severely imbalanced: Qwen has
no correct N=10 confirmation trajectory; Gemma has one in v4.1 and none in
v4.2--v4.4. The outcome switch is therefore an audit, not a powered group
comparison.

All V4 and later visualizations use the Aurora palette registered in the report
builder and repository README. Count colors are ordered blends of the supplied
Aurora anchors; model, panel, pooling, control, and zero-reference colors keep
fixed meanings across figures.

## Answer-query attention

Discovery-only rankings are stable under 500 seed bootstraps, but `span_end`
and `span_mean` expose different mechanisms.

- Qwen `span_end`: L29H3 is rank 1 in all four panels with top-8 selection
  frequency 1.0. Its coverage is only about 0.22 and its effective number is
  about 1.04, so it concentrates on roughly one needle rather than aggregating
  uniformly across all needles.
- Qwen `span_mean`: L27H18 is rank 1 in v4.1, v4.3, and v4.4; L28H19 is rank 1
  in v4.2. Coverage is about 0.79--0.86 and the effective number is about
  4.8--5.2, which is much closer to broad aggregation.
- Gemma `span_end`: the top head is L5H2 in v4.1 and L29H6 in v4.2--v4.4.
  Coverage is about 0.85--0.89 and the effective number is about 5.0--5.3.
- Gemma `span_mean`: L35H2 is rank 1 in every panel. Coverage is about
  0.76--0.80 and the effective number is about 4.6--4.8.

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

Most count-adjusted wrong-minus-correct bootstrap intervals include zero:
only 36 of 256 Qwen comparisons and 24 of 256 Gemma comparisons exclude zero
before any multiplicity correction. Aggregate needle mass alone therefore does
not reliably separate correct from wrong answers.

## Omission and nested-increment diagnostics

These diagnostics are descriptive correlations on confirmation examples, not
interventions.

- For Qwen undercounts, `span_end` attention to the omitted tail relative to
  the retained prefix averages 0.28--0.37 across panels; the low-attention
  bottom-k set overlaps the omitted tail by 0.38--0.58. The analogous
  `span_mean` tail ratio is 0.79--0.84.
- For Gemma undercounts, the `span_end` tail ratio is 0.68--0.93 and bottom-k
  overlap is 0.29--0.46. The `span_mean` tail ratio is usually above 1, with
  only 0.11--0.16 bottom-k overlap.
- When a nested N to N+1 pair registers the increment, attention assigned to
  the newly added needle is higher than when it fails to increment: Qwen
  `span_end` 0.474 versus 0.379 and `span_mean` 1.099 versus 0.808; Gemma
  `span_end` 0.824 versus 0.692 and `span_mean` 1.324 versus 0.983.

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

## Reproducibility and remaining causal scope

The run root is `run_20260731_v4_numeric_presentation_v3`. Each model directory
contains behavior labels, representation captures, raw answer-query attention,
analysis manifests, tables, and figures. Reproduce the Qwen partition analysis
from the saved answer-query rows with:

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
