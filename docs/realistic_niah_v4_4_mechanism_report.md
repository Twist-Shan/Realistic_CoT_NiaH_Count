# Realistic NIAH V4.4 non-thinking mechanism report

This note is the compact audit companion to
[`reports/realistic_niah_v4_4_mechanism_report.html`](../reports/realistic_niah_v4_4_mechanism_report.html).
The HTML is the primary artifact: it contains the all-layer interactive prompt
and answer-query counter views, the endpoint/full-span attention atlas, the
causal forest plots, captions, and collapsed result tables.

The completed evidence summarized here predates the larger ±k causal-v2
follow-up. That follow-up is now implemented but has no results yet; its frozen
design, exact patch definitions, request accounting, and audit are in
[`realistic_niah_v4_causal_v2.md`](realistic_niah_v4_causal_v2.md). Do not mix
causal-v2 protocol intent with the completed estimates below.

## Why the report fixes V4.4

The four-panel analysis established that prompt-side `needle-end` count
geometry remains visible while position, city-score order, and city-score
content are successively relaxed from V4.1 to V4.4. The follow-up report
therefore fixes the stimulus panel to V4.4 and keeps `needle-end` as the primary
prompt read site.

The display coordinates are deliberately not re-selected on V4.4. Prompt and
answer-query PCA layers and bases remain frozen from V4.1 discovery data, and
V4.4 states are projected into those coordinates. This makes the V4.4 plots a
cross-panel generalization check rather than a V4.4 refit. All causal summary
tables in the standalone report are filtered to V4.4 before aggregation.

## Evidence order

1. Prompt-reading hidden states: the post-block residual at every needle's
   final token, with every captured layer available in the 3D selector.
2. Answer-query hidden states: the post-block residual at the final `Total:`
   query, with all layers, actual greedy outcome filters, and all-row versus
   correct-only PCA sensitivity.
3. Joint geometry: prompt endpoint and answer-query count centroids in a
   role-centered shared coordinate system.
4. Attention representation: all saved answer-query layer/head rows, switchable
   between endpoint-key mass and literal full-needle-span mass.
5. Causal tests: head-bank ablation, exact needle-end and answer-query residual
   transport, then single- and multi-layer geometric steering.

## What each causal experiment compares

| Experiment | Paired unit / receiver | Intervention | Control or baseline | Primary estimand |
|---|---|---|---|---|
| Ranked head-bank ablation | The same V4.4 count-7--10 prompt | At the final `Total:` query, zero the discovery-ranked span-end top-4 or top-8 pre-`o_proj` head slices | Zero the same number of deterministic random heads with the same per-layer allocation; both are also compared with the unmodified clean prompt | `(ranked - clean) - (random - clean)` generated-count shift, plus changed rate and MAE |
| Exact needle-end residual patch | Same-model, same-seed V4.4 nested 5↔6, 7↔8, or 9↔10 pair, differing at a known toggled slot | Copy the donor's post-block residual at the toggled needle's final token to the receiver's corresponding token and clamp it from the selected start layer through the final layer | The clean receiver. This screen has no matched-random residual control | Direction-aligned generated-count shift and movement toward donor gold |
| Exact answer-query residual patch | Same-model, same-seed V4.4 directed 5↔6, 7↔8, 9↔10, or 5↔10 pair | At one layer, replace only the receiver's final `Total:` query residual with the donor query residual, then continue full greedy generation | The clean receiver; L0 is also the cross-layer early reference | Among rows where clean donor and receiver predictions differ, whether the patched output adopts the donor prediction |
| Steering v1 | The same V4.4 confirmation receiver and target count | At one layer, use the full-dimensional residual-preserving update `h' = h + α(μ_target - μ_receiver)`, with `α=1` | An orthogonal random direction with the same norm at the same prompt and layer | Geometric-minus-random direction-aligned count shift, movement, and exact-target effects |
| Steering v2 | Held-out V4.4 confirmation receiver; plans locked on disjoint discovery seeds | Apply the locked full-dimensional centroid delta at one layer or a locked multi-layer set | Per-layer norm-matched orthogonal random directions; invalid output is a failure | Held-out V4.4 geometric-minus-random aligned, moved, and exact-target effects |

These are three different causal questions. Ablation tests a bank-level
necessity contrast. Exact donor-state patching tests whether a sample-specific
state is sufficient to transport information. Centroid-delta steering tests
whether a population-level count direction is used by the downstream readout.
They should not be treated as interchangeable positive or negative controls.

## V4.4 results

### Attention-bank ablation

- Qwen clean accuracy is 100% on the 40 selected V4.4 high-count prompts.
  Top-4 ranked-minus-random count shift is `-0.425` with 95% CI
  `[-0.575, -0.275]`; top-8 is `+0.025 [-0.075, +0.150]`. The lack of a
  monotone top-k effect argues against a simple “more selected heads means more
  necessity” account.
- Gemma clean accuracy is 0% on this same high-count slice. Its top-4 and top-8
  ranked-minus-random shifts are `-2.025` and `-2.625`, respectively. Those
  effects show that the ranked bank causally changes count magnitude, but this
  subset cannot establish that the bank is necessary for *correct* counting.
- The completed screen uses mixed discovery-ranked banks. It does not prove a
  single head is necessary and is not a complete phenotype-pure top-1→top-k
  dose-response scan.

### Exact needle-end patching

At Qwen start layers 9, 18, and 26 and Gemma start layers 10, 20, and 31, all
V4.4 direction-aligned effects are essentially zero and every Holm-adjusted
`p=1`. A single toggled needle-end residual is therefore not sufficient to
transport one increment under the tested cumulative-clamp protocol. This does
not test a complete needle span, several coordinated token states, or a
matched-random residual control.

### Exact answer-query patching

- Qwen donor-prediction adoption is 0% through L18, 53.3% at L26, 98.3% at
  L29, and 100% at L35.
- Gemma adoption is 0% through L20, 87.5% at L31, 98.8% at L35, and 100% from
  L38 through L41.
- The denominator contains 48 eligible Qwen and 56 eligible Gemma V4.4 rows per
  layer. Every patched V4.4 output is a valid 1--10 numeric continuation.

This is strong late-layer state sufficiency for transporting the donor's
already-computed *prediction*. It is not necessarily transport of the donor's
gold count when the donor baseline is wrong.

### Geometric steering

In v1, the early and middle layers are approximately null. The late-layer
geometric-minus-random aligned effects are `+0.917 [0.733, 1.100]` at Qwen L26
and `+1.400 [1.217, 1.583]` at Gemma L31, both Holm `p=0.0117`.

V2 locks plans on four discovery seeds across all panels and tests them on ten
disjoint seeds. On the held-out V4.4 panel:

- Qwen L26 single and L9+18+26 multi effects are `+1.000` and `+0.950`.
- Gemma L31 single and L10+20+31 multi effects are `+1.383` and `+1.367`.
- All four 95% confidence intervals exclude zero, but multi-layer steering does
  not consistently exceed single-layer steering. Exact-target improvements are
  only 3.3--6.7 percentage points.

The licensed conclusion is stable directional control of the late
answer-query state, not exact count setting and not a multi-layer advantage.

## Current mechanism and missing tests

The smallest account consistent with all completed V4.4 evidence is a
distributed retrieval-and-readout mechanism: multiple attention heads retrieve
needle information; prompt-side endpoints contain decodable count/index
structure but one endpoint state is not sufficient by itself; a late
`Total:`-query state becomes an executable carrier from which the LM readout
generates the number.

The following experiments are not completed evidence and are not claimed in
the report:

- head-output patching;
- full-needle-span or coordinated multi-token residual patching;
- a phenotype-pure top-1→top-k head-bank ablation curve;
- a V4.4-only preregistered discovery/confirmation rerun of every causal test;
- precise set-to-count steering.

The phenotype-pure top-1…32 sweep, coordinated full-span patches, all-layer
single/cumulative answer patches, and ±k steering are registered in causal-v2.
They remain missing *evidence* until the formal causal-v2 run, held-out
confirmation, and strict audit finish.

## Rebuild

```bash
PYTHONPATH=scripts:src python scripts/build_realistic_niah_v4_4_report.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --output reports/realistic_niah_v4_4_mechanism_report.html \
  --repo-root .
```

Machine-readable V4.4 tables are written to
`reports/v4_non-thinking_causal/v4_4/`.
