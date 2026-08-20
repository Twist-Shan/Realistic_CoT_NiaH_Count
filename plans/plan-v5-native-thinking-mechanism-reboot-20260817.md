# V5 Native-Thinking Mechanism Reboot

**Status:** causal parser/compiler and development-smoke runner implemented; formal fresh-confirmation launch remains unfrozen
**Date:** 2026-08-17
**Proposed experiment namespace:** `realistic_niah_v5_native_mechanism_reboot_v1`

## 1. Why the V5 mechanism experiment should be reorganized

The current evidence is asymmetric:

- the non-thinking report has a nearly complete within-task causal chain;
- the native-thinking representation analysis is strong and well audited;
- the current native-thinking mechanism report has a confirmed retrieval result for
  Gemma, a weak/null result for Qwen under the same retrieval definition, a strong
  final answer-state transport result for both models, and a null broad-head result at
  the final answer query;
- the missing scientific object is the computation *inside the generated trace* that
  links one retrieved item to the next item or to stopping.

Therefore, the reboot must not center another answer-query broad-head search. The
primary native-thinking chain should be:

```text
exact prompt record R(c_k)
    -> response item span I_k
    -> item-end progress state S_k
    -> retrieve c_(k+1) or STOP
    -> terminal answer state A_N
    -> final numeric count N
```

The trace is allowed to function as external working memory. The plan does not assume
that the same basis, head set, or token site is preserved across these stages.

## 2. Audit of the evidence that motivates this plan

### 2.1 Non-thinking report

The current report is preferable as the manuscript-facing version; the frozen report
should remain the immutable audit snapshot. The current report preserves the frozen
scientific content but reorganizes it from four detailed chains into the clearer
three-step narrative `Form -> Retrieve -> Consolidate`, moves architecture-specific
writers and negative experiments to appendices, and clarifies the nearby
length-matched ordinary-span control and full-document attention visualization.

Within the registered two-model Realistic NIAH setting, the non-thinking analysis is
already unusually comprehensive:

| Stage | Strongest evidence already present | Correct claim level |
| --- | --- | --- |
| Form | active-needle corruption, state deformation/retention, dense full-span restoration | distributed span evidence is causally used |
| Retrieve | discovery-frozen broad heads, layer-matched ablation, post-O geometry, aligned-vs-orthogonal removal | a broad retrieval/aggregation pathway is naturally used |
| Consolidate | full answer-state donor patch and count-aligned removal | the late answer state is executable and directionally necessary |
| Integrated chain | same-forward partial serial mediation | the three stages form a supported partial chain, with bypasses |
| Architecture | Qwen local OV evidence and Gemma distributed residual evidence | implementations differ by model |

This is comprehensive **within scope**, not an exhaustive circuit proof. It still does
not establish a unique path, token-level address/content decomposition, an abstract
position-invariant counter, or generalization across models, lengths, templates,
tasks, and larger counts. Dense patches transport more than a scalar count, and the
negative interaction plus residual source repair in the integrated experiment already
shows that the registered mediators do not exhaust the computation.

### 2.2 Native-thinking representation

The geometry comparison has the right primary protocol:

- fixed semantic sites: `span_end` for non-thinking and `item_end` for native running
  states; registered answer-query sites for final count;
- 20 discovery seeds and 10 held-out confirmation seeds;
- discovery-only scaling, PCA, probe fitting, and layer selection;
- parser-observed ragged native trajectories without padding to gold N or final
  `Total`;
- Logistic, nearest-centroid, and SNR reported separately;
- marker/band analysis and entity-domain transfer kept as diagnostics/appendices.

The frozen-layer confirmation results support **more decodable** native states:

| Model | Endpoint | Non-thinking Log/NCC | Native Log/NCC | Native-minus-non-thinking interpretation |
| --- | --- | ---: | ---: | --- |
| Qwen3-8B | running | 53.0% / 46.0% | 68.1% / 58.9% | more decodable; not globally tighter |
| Qwen3-8B | final | 66.0% / 67.0% | 100.0% / 99.0% | much more decodable and higher SNR |
| Gemma4-E4B | running | 43.0% / 39.0% | 78.4% / 69.3% | more decodable and higher SNR |
| Gemma4-E4B | final | 55.0% / 58.0% | 71.0% / 70.0% | more decodable; SNR is not higher |

The mechanism plan must retain four limitations:

1. `span_end` and `item_end` are role-aligned, not token-identical or coordinate-aligned.
2. Qwen native running SNR is -5.25 dB versus -4.49 dB for non-thinking despite
   higher probe accuracy; "more decodable" is not equivalent to "less noisy".
3. In the confirmation traces, the hybrid parser classifies explicit `inline_count`
   structure in 73% of Qwen and 91% of Gemma trajectories. A running-index probe may
   read literal/ordinal or formatting information rather than a history-dependent
   count state.
4. Native missingness is outcome- and trace-dependent. Gemma confirmation contains
   only 64% one-to-one trajectories and 30% partial-unique trajectories, so later-k
   geometry is conditional on the model reaching a parser-observable item.

Representation is therefore a strong localization prior, but not evidence that
`item_end` is a causal counter.

## 3. What is retained, retired, and newly required

| Existing component | Reboot status | Reason |
| --- | --- | --- |
| Gold-assisted hybrid parser and exact token alignment | retain | required experimental alignment instrument |
| Full-panel geometry comparison | retain as the representation source of truth | newer and stricter than the geometry block in the current native mechanism report |
| Gemma pre-reference targeted-retrieval result | retain and replicate on fresh confirmation | current strongest retrieval result |
| Qwen raw-mass targeted-retrieval result | retain as a locked null/weak result | do not chase a favorable K or site on used confirmation data |
| Full answer-state patch | retain | establishes final-state sufficiency |
| Final answer-query broad aggregation sweep | retire from the primary chain | the registered ranked interventions changed no final count; the site is probably downstream of trace aggregation |
| Marker-site patch sweep | retire from the primary chain | multiple variants were null after correction and do not identify a trace update |
| New representation layer/site searches | do not run | primary sites and discovery-selected layers are already fixed |
| Trace source, progress transition, stop, and mediation tests | newly primary | these are the missing causal arrows |

All seeds and outcomes already inspected in V5 must be treated as **development data**
for any new hypothesis. They cannot regain confirmatory status merely because a new
script is written.

## 4. Claim ladder and terminology

The report must use the following hierarchy:

1. **Decodable running index:** a probe predicts k.
2. **Trace-organized evidence:** trace-token intervention changes the item-end state in
   a count-specific way.
3. **Causally active progress state:** a matched intervention at `item_end:k` changes
   the next retrieval/stop decision.
4. **Transportable progress state:** a signed state edit moves behavior toward a
   preregistered k+delta target more than matched controls.
5. **Partial mediated chain:** source repair is reduced by a downstream progress/answer
   mediator intervention in the same forward computation.

The phrase **internal counter** is prohibited unless all of the following gates pass:

- the hidden-state effect remains after controlling or intervening on visible
  index/ordinal tokens, marker kind, and token position;
- signed k-to-k+delta state edits cause sign-consistent changes in continuation or
  stopping behavior;
- the effect replicates in a marker-neutral or marker-corrupted condition;
- a source -> progress state -> answer state -> output mediation chain passes;
- the update generalizes over multiple k values rather than being a single donor-label
  lookup.

If these gates do not all pass, use **history-dependent progress state** or
**counter-compatible trace state**.

## 5. Data, split, and cohort contract

### 5.1 Development and fresh confirmation

- Models and immutable revisions remain Qwen3-8B and Gemma4-E4B from the V4/V5 model
  registry.
- Counts remain 1 through 10 on the frozen Realistic NIAH construction.
- Existing seeds 1234--1263 and all previously run supplemental seeds are development
  only for this reboot.
- Before formal launch, create a machine-readable registry of **20 new seed values**
  that do not appear in any local or remote generation, capture, plan, or report
  manifest. Exact values must be frozen only after this non-overlap audit.
- The new confirmation grid is 10 counts x 20 seeds = 200 trajectories per model.

Twenty new seeds are recommended because occurrence-level effects are averaged to the
seed before inference, parser constructibility reduces support, and the current
10-seed geometry panel is too small to support additional model/format stratification.

### 5.2 Cohorts

Use distinct cohorts for distinct estimands:

- `parser_hit`: representation and missingness audit only;
- `one_to_one`: discovery selection and teacher-forced local causal trials;
- `one_to_one` with a constructible clean continuation: primary local continuation
  inference;
- `one_to_one_correct`: secondary actual-greedy correct-to-wrong damage and final-count
  adoption;
- marker-neutral / marker-corrupted subsets: deconfounding robustness, never used to
  select the primary layer or bank.

Final correctness must not be used to select heads or fit a progress-state basis.
Every exclusion must carry a single explicit reason. Parser misses, invalid outputs,
truncations, and partial traces remain in the denominator of the audit even when they
are ineligible for a particular local intervention.

## 6. Site contract: frozen semantic roles, provisional retrieval anchor

| Site | Semantic role | Primary use |
| --- | --- | --- |
| `pre_reference_d1:k` / current `city_pre_d1:k` | output token immediately before the first city-target token | locked legacy retrieval replication and baseline arm in the new development smoke |
| full `item_span:k` tensor | all tokens of accepted trace item k | source corruption/restoration unit |
| `item_end:k` | final token of accepted item k | progress-state and next-item/stop intervention |
| terminal `item_end:M` | final accepted item boundary | terminal stop and answer-state formation |
| `answer_query_v3` | final literal token before numeric answer | final-state readout/mediation |
| `post_boundary:k` | fixed sensitivity only | boundary-token robustness |

The semantic roles above are frozen. The exact output-side retrieval anchor is the one
explicit exception: `city_pre_d1` remains the current registry baseline, but a
grammar-conditioned mapping may be selected by the development-only smoke below. No
retrieval anchor may be promoted after examining fresh confirmation outcomes, and no
other site may be moved.

### 6.1 Development-only grammar-conditioned retrieval-anchor smoke

**Status (updated 2026-08-18): compiler implemented; GPU smoke not yet run.** The
grammar-aware registry now materializes every candidate below, records explicit N/A
outcomes, deduplicates roles that resolve to one model token, and keeps
`retrieve_query_state = city_pre_d1` only as a compatibility alias. The 600-trajectory
local rebuild passed every audit gate with zero compile failures. This does not freeze
a winning anchor or authorize fresh confirmation.

The smoke must register the following candidate hidden-state anchors with exact
character/token boundaries. “Post” means the residual state at the final token of the
named delimiter, i.e. after that delimiter has been consumed for next-token prediction.

| Candidate anchor | Exact semantic boundary | Intended diagnostic |
| --- | --- | --- |
| `city_pre_d1` | token immediately before the first token overlapping the selected city occurrence | current local baseline; asks whether the city is already retrievable immediately before emission |
| `unit_pre_d1` | token immediately before the first token of the selected city-bearing semantic unit, including any opening parenthesis, bullet, rank label, or leading connective | asks whether retrieval is prepared at local unit entry rather than only next to the city |
| `post_open_delimiter` | final token overlapping an opening `(` or an invariant structural bullet, when present | distinguishes a boundary/opening-delimiter state from the later label-conditioned city-pre state |
| `record_clause_pre_d1` | token immediately before the literal clause `In the 2024 city score audit` inside the selected unit | tests retrieval at canonical source-record clause entry, before the city name appears |
| `block_pre_d1` | token immediately before the enclosing list/recap/evidence block | exploratory block-initiation diagnostic; not automatically an event-specific retrieval query |

Not every anchor is meaningful for every grammar. The preregistered smoke matrix is:

| Grammar family | Candidate anchors to materialize and compare |
| --- | --- |
| `adjacent_rank_after_city` | `city_pre_d1`, city-bearing `unit_pre_d1`; add `record_clause_pre_d1` when the canonical clause is present |
| `adjacent_rank_before_city` | `unit_pre_d1` before the rank unit versus `city_pre_d1` after the rank; add delimiter/clause anchors when present |
| `same_unit_rank_after_city` | `unit_pre_d1` before `(`, `post_open_delimiter`, and `city_pre_d1` after `City:`; add clause anchor when present |
| `same_unit_rank_before_city` | `unit_pre_d1` before `(`, `post_open_delimiter`, and `city_pre_d1` after `Record k:` or ordinal text |
| `structural_explicit_rank_before_city` | `unit_pre_d1` before the structural sequence/count marker, `city_pre_d1` after it, and clause/block entry where resolvable |
| `structural_invariant_bullet` | `unit_pre_d1` before the bullet, `post_open_delimiter` at the bullet, and `city_pre_d1` after it |
| `structural_unmarked` | `unit_pre_d1`, `city_pre_d1`, and `record_clause_pre_d1` for canonical source-record sentences |
| `evidence_sequence_unranked` | per-mention `unit_pre_d1`/`city_pre_d1`, plus `block_pre_d1` as a separate exploratory sequence-initiation diagnostic |

Smoke constraints:

- use development rows only and report results separately by grammar and model;
- freeze the layer, head bank/K, intervention strength, donor/source definition, and
  target metric across candidate anchors so the smoke searches position only;
- compare anchors within the same event with equal intervention budget and preserve all
  null/negative arms;
- log unresolvable or token-fused boundaries as explicit `not_applicable`/alignment
  outcomes rather than silently falling back to `city_pre_d1`;
- freeze and hash the grammar-to-anchor mapping before opening any new confirmation
  result;
- retain the historical `pre_reference_d1` result as a locked replication arm even if a
  different anchor is selected for the new trace-retrieval experiment.

### 6.2 Implemented development-smoke pipeline

The obsolete `marker_end:k -> city_end:k` causal CLI path has been removed. The new
pipeline has three resumable stages:

1. `causal-source-writes` captures
   `W_O^h sum_(j in prompt record of city k+1) A_h(q,j)V_h(j)` at each deduplicated
   transition anchor. Each anchor is an atomic JSONL shard.
2. `causal-plan` pools write norms request-first, anchor-role-first, then seed-first;
   builds five cross-fit folds; and freezes one K=8 bank plus three nonoverlapping,
   exactly layer-matched random banks. The selected bank is capped at half the heads in
   any layer so its own exact controls always remain constructible.
3. `causal-heads` applies the fold-specific bank at one anchor token. All text between
   the anchor and target city remains teacher-forced, but only the identical frozen
   city-token span is scored. Clean, selected-bank, and random-control results are
   atomic anchor-condition shards.

All existing seeds 1234--1263 are causal development data. Primary pooling excludes
the non-event-specific `block_pre_d1`; `--include-secondary` and
`--include-block-pre` are explicit smoke-only expansions. `--limit` chooses a
deterministic diversity-oriented subset that prioritizes uncovered semantic roles,
grammars, cohorts, and seeds rather than simply taking the first traces.

Example first-card smoke for Qwen (repeat with the Gemma generation file and model):

```bash
python scripts/run_realistic_niah_v5.py causal-source-writes \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --output work/v5_native_mechanism_reboot/Qwen3-8B/source_writes_smoke \
  --limit 24

python scripts/run_realistic_niah_v5.py causal-plan \
  --source-writes work/v5_native_mechanism_reboot/Qwen3-8B/source_writes_smoke \
  --output work/v5_native_mechanism_reboot/Qwen3-8B/causal_plan_smoke \
  --development-smoke

python scripts/run_realistic_niah_v5.py causal-heads \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --plan work/v5_native_mechanism_reboot/Qwen3-8B/causal_plan_smoke/retrieval_anchor_bank_plan.csv \
  --output work/v5_native_mechanism_reboot/Qwen3-8B/head_ablation_smoke \
  --limit 24 --include-secondary
```

Reissuing either GPU command resumes from existing complete shards. `--no-resume`
refuses to touch a nonempty shard directory rather than overwriting it.
The smoke plan is stamped `formal_inference_eligible=false`; omit
`--development-smoke` after a full 30-seed source-write capture.

## 7. Experiment R0: separate visible markers from hidden progress

This experiment is required before interpreting item-end geometry mechanistically.

### R0a. Incremental nuisance audit

On development only, fit three grouped-by-seed predictors of k:

1. nuisance-only: absolute token position, prefix length, item length, marker kind,
   visible numeric/ordinal token identity, and registered trace-template/category
   features;
2. hidden-state-only at the frozen `item_end` layer;
3. nuisance plus hidden state.

Report held-out incremental delta balanced accuracy and delta log loss from (1) to
(3), plus cross-marker train/test transfer. This is descriptive and cannot establish
causal use.

### R0b. Paired marker-token intervention

For each constructible teacher-forced prefix through `item_end:k`, create:

- clean prefix;
- explicit rank/count marker corruption;
- equal-token-count non-marker trace-token corruption;
- full current-item corruption;
- all-prior-item corruption.

Token count, answer-query position, prompt, cited city identities, and the suffix after
the edited region must be identical. Replacements use a pre-audited token-length pool;
they must not silently retokenize the remaining prefix.

Primary outcomes are next-continuation sequence log probability and distance from the
development-frozen correct-k centroid. Actual greedy next-item identity and final count
are secondary behavioral outcomes.

**R0 gate:** a history-dependent-state interpretation requires the all-prior-item or
full-item intervention to exceed marker-only and equal-budget controls on fresh
confirmation. If marker-only corruption explains the full effect, the primary result
is an externalized ordinal/list mechanism, not an internal counter.

## 8. Experiment C1: trace-span source and dense restoration

This is the native analogue of non-thinking full-span source restoration.

Start from a prefix in which accepted trace items 1..k are corrupted. At one layer,
apply one of the following patches:

- clean full states for all corrupted item spans;
- clean `item_end` states only;
- clean marker-token states only;
- equal-token-budget ordinary-state patch;
- no patch.

Run a layerwise scan on development. Freeze a reuse window and one representative
layer by a deterministic rule before the fresh confirmation run. The rule should use
the smallest contiguous layer interval whose full-span-minus-ordinary repair reaches
at least 80% of the development peak; it must not use confirmation.

Primary local outcome:

```text
repair_source = logp(target continuation | full-span restore)
              - logp(target continuation | ordinary-state patch)
```

Secondary outcomes are actual-greedy next citation, stop/continue decision, final
count error, and downstream `item_end`/`answer_query_v3` displacement.

**C1 gate:** full-span restoration must beat ordinary-state and endpoint-only patches.
If endpoint-only is equally effective, the causal source unit can be narrowed; if all
restoration arms are null, do not proceed to a source-mediation claim.

## 9. Experiment C2: targeted retrieval and progress transition

### C2a. Locked targeted-retrieval replication

Keep the existing `pre_reference_d1` bank and K policy fixed. Evaluate it once on the
new confirmation registry. Gemma is a replication target; Qwen is a locked weak/null
target. Do not search a new Qwen K on confirmation.

This locked replication remains at `city_pre_d1`. Separately, before confirmation, run
the grammar-conditioned retrieval-anchor smoke in Section 6.1 on development only.
That smoke may freeze a different output-side anchor for the *new* trace-retrieval
analysis, but it does not redefine or replace the historical replication result.

A preregistered Qwen sensitivity may zero the same frozen bank over one fixed four-token
window immediately preceding the citation. This is one joint intervention, not four
separate position tests.

### C2b. Progress-transition mechanism

For nonterminal k, query at `item_end:k` and target the exact prompt record matching the
actually accepted next city c_(k+1). For terminal k=M, target the registered stop-to-
`answer_query_v3` continuation. Preserve the observed response order; do not substitute
passage order.

Use two discovery rankings:

1. the existing raw target-span attention-mass bank as a locked replication/diagnostic;
2. one new primary source-specific OV-write bank, ranked by

```text
mean || W_O^h sum_(j in R(target)) A_h(q,j) V_h(j) ||_2
```

over development one-to-one rows. This score tests whether a head both routes to and
writes content from the semantic target span. Freeze one K per model by a deterministic
cumulative-score rule, with exact layer-matched disjoint random controls. K dose curves
may be shown for development but are not separate confirmation hypotheses.

Primary effects are ranked-minus-random target-continuation log probability for
`continue` and stop-continuation log probability for `stop`, reported separately.
Actual-greedy exact next citation/stop is the behavioral validation endpoint.

## 10. Experiment C3: signed progress-state editing

Fit a discovery-only `item_end` progress basis after residualizing the registered
nuisance features from R0. The primary edit is a centroid chord from k to k+delta,
with delta in {-1,+1}; |delta|=2 is sensitivity only.

At a fixed `item_end:k`, compare:

- aligned k-to-k+delta edit;
- equal-realized-norm orthogonal edit at the same layer and token;
- visible-marker-direction edit with the same norm;
- self/zero edit;
- full donor patch as a sensitivity analysis.

The visible prefix, including the already emitted marker k, is held fixed. Outcomes:

- adoption of the preregistered next marker/rank when the trace format exposes one;
- next cited city identity;
- continue-versus-stop decision;
- number of subsequently generated accepted items;
- final `Total` and absolute count error.

**C3 gate:** aligned edits must be sign-consistent across k, exceed both orthogonal and
marker-direction controls, and reproduce in marker-neutral or marker-corrupted rows.
Only this gate permits the phrase "causally active progress state".

## 11. Experiment C4: terminal consolidation and serial mediation

The existing full answer-state patch establishes final-state sufficiency but does not
identify how the trace creates that state. Run an integrated teacher-forced prefix
experiment with these registered arms:

1. clean;
2. trace-source corruption;
3. source corruption + full trace-span restoration;
4. arm 3 + progress-mediator aligned removal;
5. arm 3 + equal-norm progress orthogonal removal;
6. arm 3 + late answer-state aligned removal;
7. arm 3 + equal-norm late orthogonal removal;
8. arm 3 + both aligned removals.

Read progress-state geometry before the late intervention, answer-state geometry after
the late intervention, target-number log probability, and actual greedy final count.
The teacher-forced design is primary because it keeps every registered site and suffix
identical. After freezing one representative intervention, repeat it from `item_end:k`
with unconstrained greedy continuation as a behavioral validation; divergent or invalid
traces are outcomes and must not be filtered away.

Define, within seed:

```text
source_repair = outcome(restore) - outcome(corrupt)
progress_mediation = outcome(restore + progress_orthogonal)
                     - outcome(restore + progress_aligned)
late_mediation = outcome(restore + late_orthogonal)
                 - outcome(restore + late_aligned)
```

For error outcomes, orient signs so positive always means greater registered damage or
repair as named. The report must also show the joint interaction and residual repair;
it must not force mediated proportions when interventions interact or repair is near
zero.

**C4 gate:** source repair, progress mediation, and late mediation must all have the
registered positive direction, and a later intervention must have exactly zero effect
on earlier saved readouts. Passing supports an ordered **partial** serial mediation
chain, never a unique circuit.

## 12. Optional architecture localization

Run this only after C1--C4 pass for a model. Use the existing natural-head-write
capture and cross-layer residual tools to ask whether the supported progress/answer
mediators are written by a localized OV set or a distributed residual path. This is an
appendix objective and must not be used to rescue a failed functional chain.

## 13. Statistical protocol

- Independent unit: seed.
- Average all eligible occurrences within request, then counts/conditions within seed,
  before inference. A long N=10 trace must not contribute ten independent samples.
- Uncertainty: 10,000-draw seed-cluster bootstrap.
- Test: two-sided exact/randomized seed sign-flip on the seed effects.
- Multiplicity: for each experiment and primary endpoint, Holm correction spans the two
  models. Continue and stop are separate registered families. Development K/layer
  curves are not confirmation hypotheses.
- Local primary endpoint: complete target-continuation log probability.
- Local behavioral validation: parsed actual-greedy exact citation or stop.
- Final primary endpoint: actual-greedy absolute count error, retaining invalid outputs
  conservatively.
- Report effect sizes and intervals even when p-values are non-significant.
- Do not pool Qwen and Gemma into one mechanism claim unless both pass the same gate.

## 14. Stopping rules

1. If R0 shows that marker-only corruption accounts for the effect and aligned state
   editing fails, stop the internal-progress-state program and report an externalized
   ordinal/list mechanism.
2. If C1 full-span restoration is null, do not run the integrated source mediation.
3. If C2 progress transition is null for a model under both frozen banks, do not search
   confirmation positions, K values, or heads to obtain significance.
4. If C3 edits do not exceed marker-direction and orthogonal controls, retain
   decodability only.
5. Do not repeat the final answer-query broad-head sweep; its existing null is part of
   the evidence ledger.
6. Do not begin writer localization until the functional chain passes.

## 15. Implementation map

The repository already contains reusable primitives in
`src/realistic_niah_v5/causal.py` for:

- trace-token corruption and equal-budget control;
- targeted-retrieval and progress-transition continuations;
- position-local head ablation;
- item/answer subspace fitting and removal;
- residual donor patching;
- query-context masking;
- natural per-head writes;
- seed-level paired inference.

Implementation should extend these primitives rather than create an independent
pipeline. Expected changes after this draft is approved:

- add a new frozen config, not overwrite `configs/realistic_niah_v5.json`;
- add exact marker-token and prior-item span registries to the parser/capture audit;
- add dense multi-token trace-state restoration and source-specific OV decomposition;
- add integrated multi-site mediation trials;
- expose resumable CLI stages through `scripts/run_realistic_niah_v5.py`;
- add CPU/mocked tests for span budgets, token alignment, intervention position,
  realized-norm controls, no-future-read guarantees, seed aggregation, and resume;
- write every run under a new timestamped/experiment-ID directory with config, command,
  model revision, git state, timing, checksums, and completion audit.

No existing report or result directory should be overwritten. The new report should be
generated under a versioned directory until all audits pass.

## 16. Proposed report structure

1. Frozen parser, cohorts, and representation facts (link to the geometry comparison;
   do not duplicate stale geometry tables).
2. Retrieve: prompt record -> current citation.
3. Update: full trace item -> item-end progress state.
4. Transition: progress state -> next citation or stop.
5. Consolidate: terminal trace state -> executable answer state.
6. Integrated partial mediation.
7. Model-specific differences and registered nulls.
8. Claim ledger, exclusions, hashes, and complete negative-result appendix.

This structure mirrors the successful non-thinking `Form -> Retrieve -> Consolidate`
logic while respecting that native thinking performs repeated retrieval and state
updates inside an autoregressively generated trace.

## 17. Recommended decisions before freezing

Recommended defaults are:

- 20 genuinely unseen confirmation seeds;
- `city_pre_d1` as the locked retrieval-replication site, while the new
  grammar-conditioned trace-retrieval anchor remains provisional until the Section 6.1
  development smoke is run and hashed;
- `item_end` as the only primary progress site;
- source-specific OV-write norm as the new progress-transition ranking, with raw mass
  retained as a locked diagnostic;
- one deterministic K per model, chosen on development by cumulative source-specific
  write score;
- target-sequence log probability as the local primary endpoint and actual greedy
  behavior as validation;
- no architecture-specific search until the functional chain passes.

The seed registry, K-selection rule, exact replacement-token pool, and primary
confirmation arm count must be materialized and hashed before any formal GPU result is
opened.
