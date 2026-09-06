# Native-thinking cross-model sample-alignment contract (2026-08-29)

## Scientific estimand

The report compares causal and representational measurements between Qwen3-8B
and Gemma4-E4B.  A cross-model contrast is admissible in the main text only
when both models are evaluated on the same experimental units.  Equal sample
counts alone are not sufficient.

## Hard alignment contract

For every cross-model experiment, the two model tables must have identical
multisets of the following keys:

1. base key: `(phase, seed, gold_count)`;
2. transition assays: add `(from_occurrence, to_occurrence)`;
3. donor/recipient patching: add the registered donor and recipient count or
   offset;
4. layer sweeps: compare architecture-normalized layer fractions, while the
   non-layer sample keys remain identical;
5. condition grids: every aligned sample must have the same condition and
   repeat multiplicity in both models.

Aggregation uses the same hierarchy and weights in both models.  Model-specific
layer numbers, head identities, tokenizer spans, and frozen bank widths are
architecture parameters, not sample identities, and may differ.  Grammar
surface is recorded per model; a claim conditional on a grammar stratum needs a
separate common-support check for that stratum.

Selection must be outcome blind.  Intervention outcomes, effect sizes, and
`selection_rank` are forbidden when building a shared panel.  Existing
model-specific outputs are immutable historical runs; a mismatched historical
run may appear only in an explicitly labelled audit appendix and cannot support
a cross-model claim.

Single-model natural experiments are not cross-model comparisons.  In
particular, Qwen natural no-index and Gemma prompt-conditioned/simulative
no-index remain separate evidence tiers and are not forced into a synthetic
paired cohort.

## Shared transition panel

The canonical transition panel is the intersection of the two complete frozen
target-transition registries on
`(seed, gold_count, from_occurrence, to_occurrence)`.  It contains exactly one
terminal transition per fixed seed (20 discovery seeds 1234--1253 and 10
confirmation seeds 1254--1263).  The outcome-blind deterministic rule is:

1. require `to_occurrence == gold_count`;
2. maximize the common `gold_count` within seed;
3. break ties lexicographically by `(from_occurrence, to_occurrence)`.

The builder emits a shared-key file, one model-specific anchor file, and a
cryptographic manifest.  All new targeted-retrieval, counter-write,
commit/query, grammar-span, and next-city blank runs must consume these frozen
files.

## Audit and remediation table

| Experiment family | Current status | Required action |
|---|---|---|
| Running representation geometry | corrected to exact model-common support: 164 trajectories / 834 `(phase,seed,N,k)` states per model | use common-support result; old 239-vs-206 trajectory analysis is historical only |
| Final answer-query geometry | aligned full grid: 30 seeds × 10 counts = 300 trajectories/model | retain |
| Answer-source token blank | aligned: same 10 seeds × 10 counts × 5 conditions for each frozen bank | retain |
| Explicit-index progress control | new controlled panel aligned on seeds 1234–1263, N=10, ordered records, and identical numbered-list body | rerun discovery layer sweep and frozen-layer confirmation; old disjoint-seed run is historical only |
| Unnumbered controlled panel | aligned at discovery after projecting away architecture-specific layer grids; exact 10-seed × `k=2..9` confirmation | retain |
| Terminal relay partial mediation | raw plans differ; exact 86-pair/10-seed common support exists | make common-support result primary |
| Timing-stratified NCC | new exact same-timing panels contain 13 rank-after and 15 rank-before transitions/model | rerun; mismatched table becomes historical |
| Direct answer logit margin | uses the same exact timing panels as NCC | rerun; mismatched table becomes historical |
| Targeted retrieval dose | new exact 20+10 routed-transition panel frozen | rerun and compare equal sample keys at every dose |
| Targeted counter write | new exact routed-transition panel frozen | rerun |
| Commit state to targeted query | new exact ±1 plans: 40 discovery + 20 confirmation cells/model | rerun |
| Grammar-span decomposition | new real same-timing common support: 19 discovery + 9 confirmation seeds overall | rerun; do not impute the two absent timing-matched seeds |
| Next-city trace blank | new exact 30-transition panel frozen | rerun; old 45-vs-30 request result is historical only |
| Answer-query layer sweep | new exact 40 directed confirmation pairs/model | rerun on architecture-normalized layer grids |
| Qwen natural no-index | single-model natural experiment | retain; explicitly non-comparative |
| Gemma prompt-conditioned no-index | single-model auxiliary experiment | retain in appendix; explicitly non-comparative |

## Output policy

New outputs live under `work/v5_native_sample_aligned_20260829/`.  Every run
saves its resolved plan, sample-key digest, per-condition multiplicity, timer,
and environment manifest.  Before a result enters the report, an audit must
assert exact key equality and fail closed on missing/duplicate cells.

The current input-level audit is
`work/v5_native_sample_aligned_20260829/alignment_audit.json`.  It passes for
the routed and grammar panels, aligned generation views, answer-query and
native-loop plans, common-support representation, controlled explicit-index
and unnumbered panels, answer-source blanking, and terminal-relay common
support.  GPU-dependent output alignment will be audited again after both
model suites complete.
