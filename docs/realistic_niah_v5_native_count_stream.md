# V5 native-thinking count mechanism: stream state versus final retrieval

## Scientific question

Targeted retrieval of the next needle is already handled by the V5 causal
pipeline. This extension tests the next unresolved fork:

1. **Stream-state hypothesis:** count/progress information is written at
   successive `item_end:k` states and causally persists through the generated
   trace.
2. **Answer-time retrieval hypothesis:** the final `answer_query_v3` computes
   the number by directly retrieving distributed evidence from trace items
   and/or prompt records.

These hypotheses are not mutually exclusive. A hybrid model can write a
progress state during the trace and still reread trace or prompt evidence at
the final query. The code therefore estimates the components separately and
does not force a binary verdict.

The implementation mirrors the non-thinking answer-time logic, but uses a
trace-local transport test for the stream question:

```text
Trace stream                                      Answer time
item_end:d --patch--> item_end:r                  independent broad banks
  |                    |                          for trace items / prompt records
  +-> donor-vs-receiver next-city odds             |
  +-> free next-record adoption                    +-> selected-vs-layer-matched ablation
  +-> later item_end / final-answer readouts
```

## Entry points

- CLI: `scripts/run_realistic_niah_v5_count_stream.py`
- Reusable implementation: `src/realistic_niah_v5/count_stream.py`
- Development registry:
  `configs/realistic_niah_v5_native_count_stream_dev.json`
- Frozen fresh-confirmation mechanism registry:
  `configs/realistic_niah_v5_native_count_stream_confirmation_v1.json`
- V5 dataset registry containing native-unseen seeds 1316--1335:
  `configs/realistic_niah_v5_native_count_stream_registry_v1.json`
- CPU tests: `tests/test_realistic_niah_v5_count_stream.py`

All new runs should use a new directory under
`work/v5_native_count_stream/<experiment-id>/<model>/`. Commands write atomic
per-request shards, a resolved manifest, elapsed time, config hashes, runtime
versions, device information, and git state. Reissuing a GPU command resumes
complete shards; `--no-resume` refuses a nonempty shard directory.

Broad answer-time commands use `parser_hit`: both models have all 100 frozen
ranking and 100 K-selection requests, whereas applying `one_to_one` would
drop the Gemma K-selection panel from 100 to 75. Trace patching and progress
basis fitting use `one_to_one`, because their occurrence endpoints must align
exactly. `one_to_one_correct` remains a secondary, correctness-conditioned
robustness cohort and never selects a bank, K, pair, layer, or basis.
Manifests record aggregate exclusion reasons.

## Registered experiments

### S1. Intermediate trace-to-trace patching (primary)

Use signed donor offsets `d-r in {-5,-3,-1,+1,+3,+5}`. Both endpoints must
own a local next-city transition, which yields these registered count ranges:

| Offset | Valid local counts |
| --- | --- |
| -1 | 3--10 |
| +1 | 4--10 |
| -3 | 5--10 |
| +3 | 6--10 |
| -5 | 7--10 |
| +5 | 8--10 |

Within each of the resulting 33 `count x signed-offset` cells, choose 10
unique seeds outcome-blind from registry identity fields and assign one valid
receiver per selected trace with cyclic balancing. This gives 330 directed
local pairs per model. Count 2 contributes a separate 10-seed panel in each
direction (20 pairs) between occurrences 1 and 2. Because one endpoint is
terminal, this panel has final-answer outcomes only and never fabricates a
local next-city outcome. Total: 350 pairs and 1,750 condition rows per model.

The patch layer is frozen from the pre-existing discovery-only `item_end`
selection: Qwen L18 and Gemma L16; the immediate downstream readouts are L19
and L17. At that layer compare:

- `clean`;
- executed `self_patch`;
- `full_donor_patch`, matching the full-state restoration logic in the former
  non-thinking HTML;
- `progress_projected_patch`, which transports only the development-fitted
  occurrence/progress component of `h_d-h_r`;
- `norm_matched_orthogonal_patch`, with exactly the same source-delta norm as
  the projected patch and zero projection into that basis.

The local branch teacher-forces only through `item_end:r`. It then scores two
continuations at the same receiver grammar slot: the receiver's natural next
city and the city that normally follows the donor. The primary local endpoint
is

```text
log P(donor-successor path) - log P(receiver-successor path).
```

Free continuation records which city appears first. A separate full-prefix
branch keeps the later trace suffix fixed and measures whether the patch
reaches subsequent `item_end` progress coordinates or the final count.

Only `d<r` is a natural forward-stream causal test. `d>r` patches a state that
was computed using future context back into an earlier receiver; it is retained
as a symmetric representational sensitivity test, but cannot establish natural
forward propagation. The decision ledger therefore gates the stream claim on
`past_to_later_receiver` only. Full-state transport is content-confounded by
city/wording/position and must agree with projected-vs-orthogonal transport.
Every row also audits whether donor or receiver endpoints overlap an explicit
visible count marker.

### S2. Stream-state removal/retention (secondary)

Fit a development-only running-index basis at `item_end` and compare:

- `clean`;
- `aligned_running_state_removal`;
- `norm_matched_orthogonal_removal`.

The edit can target one occurrence, the prefix through k, or every item
endpoint. All later trace tokens remain teacher-forced and unchanged. Readouts
are restricted to positions later than every edited position and to decoder
layers strictly after the intervention layer. Each requested readout layer
must also have its own development-fitted `item_end` basis in the basis
artifact. The runner reports both final-answer behavior and displacement of
later item endpoints in those layer-specific progress bases. It deliberately
does not project the answer query into an item-end basis, because that would
assume cross-site basis identity. The primary specificity effect is:

```text
outcome(orthogonal removal) - outcome(aligned removal)
```

A positive effect shows causal use of the fitted progress component; it does
not by itself show that the component is marker-invariant or a scalar counter.
Every row records whether an edited `item_end` token overlaps an explicit
marker. A marker-overlapping effect is reported as marker-confounded and must
be read together with the marker/non-marker source masks and restoration arms;
it cannot by itself establish a hidden counter.
For the direct persistence readout, analyze
`downstream_item_progress_subspace_retention_score`; positive
orthogonal-minus-aligned specificity means the source edit propagated into
later trace-item progress coordinates. `scope=all` has no later item endpoint,
so it supplies final-answer behavior but no intermediate persistence value.

### S3. Trace source corruption and restoration (secondary)

At the final answer prefix, replace every parser-accepted trace-item token with
an equal-length ordinary-context token bank. At a frozen layer compare:

- no restoration;
- clean full trace-span restoration;
- clean `item_end` restoration;
- clean visible-marker restoration;
- equal-budget ordinary-state patch;
- a separate ordinary-token corruption/restoration arm.

The decisive source contrast mirrors the non-thinking analysis and is a
difference of repairs:

```text
(trace full-span restore - trace corrupt)
  - (ordinary-span restore - ordinary corrupt)
```

The three ordinary token banks are length matched, mutually disjoint, and
exclude every active prompt-record span. The same-receiver ordinary-state
patch remains a placebo diagnostic, but it is not the registered source gate.
Endpoint and marker arms determine whether the causal carrier can be narrowed.
An unmarked trace records marker restoration as explicit `not_applicable`.

### A0. Final-query source factorial (optional localization)

The prefix through the token immediately before `answer_query_v3` is always
computed cleanly. Only the final query and every numeric-answer token receive
one of these key masks:

- block the entire generated reasoning context before the answer query
  (primary trace estimand);
- decompose that context into accepted trace items versus all other reasoning
  tokens, so summary sentences cannot escape the intervention;
- block all active prompt records;
- block both;
- block terminal item or earlier items separately;
- block visible marker tokens or non-marker item tokens separately;
- exact-token-budget depth-matched controls for trace, prompt, and markers.

The 2x2 prompt/trace arms estimate trace damage, prompt damage, joint damage,
and interaction. Source-specific effects compare a semantic mask with its
matched non-source mask. This remains an optional localization/sensitivity
analysis; it is not required by the answer-time decision gate because the
requested answer-time estimand is the non-thinking-style broad-head ablation.

`block_trace_context` is the primary trace arm. `block_trace_items` alone is
only a localization arm because it intentionally leaves non-item summary text
available.

### A1. Final-query broad-head necessity (primary)

On development data, capture natural attention from `answer_query_v3` to eight
fixed source partitions: prompt records, the complete reasoning context,
accepted trace items, non-item reasoning tokens, visible markers, non-marker
item tokens, earlier items, and the terminal item. For every head:

```text
broad_score = total_source_mass * normalized_entropy_coverage
```

Selection is request-first and seed-equal. A global top-K treatment bank is
frozen before controls are constructed. Every random bank has exactly the same
number of heads in every selected layer. To match the non-thinking experiment,
each random bank samples without replacement internally but is allowed to
overlap the selected bank; all three bank identities are fixed in the plan.

The causal edit zeros those pre-O head slices only while the final query is
computed. Candidate count sequence log score/margin is primary; strict greedy
count is behavioral validation. The selected-versus-layer-matched-random
contrast tests source-ranked bank specificity.

Trace-item and prompt-record banks are separate experiments: each is ranked
independently under its own fixed broad score and each receives its own
selected-versus-layer-matched-random ablation. They must never be pooled into a
single "answer-time retrieval" estimate.

The three-stage split freezes a selection rule rather than assuming K=8:

1. **D1 attention discovery:** seeds 1234--1243, every count 1--10 (100
   requests/model), rank heads only by natural `mass x coverage`.
2. **D2 behavioral K selection:** seeds 1244--1263, five counts per seed via a
   pre-fixed odd/even assignment (100 requests/model). Evaluate nested
   `K={1,2,4,8,16,32}`. For each model/source, choose the smallest eligible K
   within one seed-level SE of the largest positive selected-vs-random expected
   count shift; correct-count-margin damage must also be positive. If K32 is
   the rising maximum, run a discovery-only K64 boundary extension before
   freezing. If no K is positive, freeze "no bank" for that source.
3. **C1 fresh confirmation:** native-unseen seeds 1316--1335 with the same
   five-count odd/even panel (100 requests/model). Open only the single frozen
   model/source K; never rerank heads, inspect the K grid, or reselect K.

The old native seeds 1234--1263 are all development data even where an older
V5 config once called 1254--1263 confirmation.

## Development workflow

The examples use Qwen paths; repeat them for Gemma. The CLI enforces the D1/D2
row panels and rejects a confirmation run unless its plan contains a frozen K
decision hash.

```bash
python scripts/run_realistic_niah_v5_count_stream.py capture-broad \
  --mechanism-config configs/realistic_niah_v5_native_count_stream_dev.json \
  --v5-config configs/realistic_niah_v5_native_count_stream_registry_v1.json \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --cohort parser_hit --row-panel broad_ranking \
  --output work/v5_native_count_stream/dev/Qwen3-8B/broad_capture

python scripts/run_realistic_niah_v5_count_stream.py plan-broad \
  --model Qwen3-8B \
  --captures work/v5_native_count_stream/dev/Qwen3-8B/broad_capture \
  --source-group trace_items \
  --output work/v5_native_count_stream/dev/Qwen3-8B/broad_plan_trace

python scripts/run_realistic_niah_v5_count_stream.py plan-broad \
  --model Qwen3-8B \
  --captures work/v5_native_count_stream/dev/Qwen3-8B/broad_capture \
  --source-group prompt_records \
  --output work/v5_native_count_stream/dev/Qwen3-8B/broad_plan_prompt

python scripts/run_realistic_niah_v5_count_stream.py broad-heads \
  --mechanism-config configs/realistic_niah_v5_native_count_stream_dev.json \
  --v5-config configs/realistic_niah_v5_native_count_stream_registry_v1.json \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --plan work/v5_native_count_stream/dev/Qwen3-8B/broad_plan_trace/answer_broad_head_plan.csv \
  --cohort parser_hit --row-panel broad_k_selection \
  --bank-sizes 1 2 4 8 16 32 --skip-greedy \
  --output work/v5_native_count_stream/dev/Qwen3-8B/broad_heads_trace_Kgrid

python scripts/run_realistic_niah_v5_count_stream.py broad-heads \
  --mechanism-config configs/realistic_niah_v5_native_count_stream_dev.json \
  --v5-config configs/realistic_niah_v5_native_count_stream_registry_v1.json \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --plan work/v5_native_count_stream/dev/Qwen3-8B/broad_plan_prompt/answer_broad_head_plan.csv \
  --cohort parser_hit --row-panel broad_k_selection \
  --bank-sizes 1 2 4 8 16 32 --skip-greedy \
  --output work/v5_native_count_stream/dev/Qwen3-8B/broad_heads_prompt_Kgrid

python scripts/run_realistic_niah_v5_count_stream.py select-broad-k \
  --mechanism-config configs/realistic_niah_v5_native_count_stream_dev.json \
  --model Qwen3-8B --source-group trace_items \
  --trials work/v5_native_count_stream/dev/Qwen3-8B/broad_heads_trace_Kgrid \
  --plan work/v5_native_count_stream/dev/Qwen3-8B/broad_plan_trace/answer_broad_head_plan.csv \
  --output work/v5_native_count_stream/dev/Qwen3-8B/k_selection_trace

python scripts/run_realistic_niah_v5_count_stream.py select-broad-k \
  --mechanism-config configs/realistic_niah_v5_native_count_stream_dev.json \
  --model Qwen3-8B --source-group prompt_records \
  --trials work/v5_native_count_stream/dev/Qwen3-8B/broad_heads_prompt_Kgrid \
  --plan work/v5_native_count_stream/dev/Qwen3-8B/broad_plan_prompt/answer_broad_head_plan.csv \
  --output work/v5_native_count_stream/dev/Qwen3-8B/k_selection_prompt
```

If the decision JSON says `requires_boundary_extension`, rebuild the plan with
`--bank-sizes 1 2 4 8 16 32 64`, run only K64 on the same D2 rows, and pass
both trial directories back to `select-broad-k`. The supervisor below performs
this branch automatically.

Fit the progress basis from an existing V5 capture index. The artifact must
contain the source layer and every later readout layer. `trace-patch` also
requires the adjacent `fit-basis` JSON sidecar, verifies its artifact hash, and
rejects any basis not fitted at `item_end` with `label=occurrence`. It also
requires the sidecar's development seeds to equal the active mechanism
registry. Then run the primary directed trace patch panel:

```bash
python scripts/run_realistic_niah_v5_count_stream.py plan-trace-patch \
  --mechanism-config configs/realistic_niah_v5_native_count_stream_dev.json \
  --v5-config configs/realistic_niah_v5_native_count_stream_registry_v1.json \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --cohort one_to_one \
  --output work/v5_native_count_stream/dev/Qwen3-8B/trace_pair_plan

python scripts/run_realistic_niah_v5_count_stream.py fit-basis \
  --capture-index work/v5_geometry_full_panel/running/Qwen3-8B/capture_index.jsonl \
  --site-kind item_end --label occurrence --cohort one_to_one \
  --layers 18 19 --rank 3 --random-seed 20260820 \
  --output work/v5_native_count_stream/dev/Qwen3-8B/running_basis.npz

python scripts/run_realistic_niah_v5_count_stream.py trace-patch \
  --mechanism-config configs/realistic_niah_v5_native_count_stream_dev.json \
  --v5-config configs/realistic_niah_v5_native_count_stream_registry_v1.json \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --pair-plan work/v5_native_count_stream/dev/Qwen3-8B/trace_pair_plan/trace_patch_pair_plan.csv \
  --basis work/v5_native_count_stream/dev/Qwen3-8B/running_basis.npz \
  --layer 18 --readout-layers 19 \
  --skip-greedy \
  --output work/v5_native_count_stream/dev/Qwen3-8B/trace_patch_L18
```

For Gemma use source/readout layers 16/17. `trace-patch` reconstructs the
outcome-blind plan from the active registry and refuses any changed, missing,
or extra pair before it loads the model.

The removal and corruption/restoration panels are secondary robustness
analyses:

```bash

python scripts/run_realistic_niah_v5_count_stream.py stream-state \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --basis work/v5_native_count_stream/dev/Qwen3-8B/running_basis.npz \
  --source-layer 20 --readout-layers 21 24 28 \
  --scope occurrence \
  --output work/v5_native_count_stream/dev/Qwen3-8B/stream_L20

python scripts/run_realistic_niah_v5_count_stream.py restoration \
  --model Qwen3-8B \
  --generations work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl \
  --layer 20 \
  --output work/v5_native_count_stream/dev/Qwen3-8B/restoration_L20
```

Analyze one or several completed experiment directories:

```bash
python scripts/run_realistic_niah_v5_count_stream.py analyze \
  --trials \
    work/v5_native_count_stream/dev/Qwen3-8B/broad_heads_trace_Kgrid \
    work/v5_native_count_stream/dev/Qwen3-8B/broad_heads_prompt_Kgrid \
  --outcome correct_count_margin \
  --output work/v5_native_count_stream/dev/Qwen3-8B/answer_broad_analysis

python scripts/run_realistic_niah_v5_count_stream.py analyze \
  --trials work/v5_native_count_stream/dev/Qwen3-8B/trace_patch_L18 \
  --experiment-ids trace_intermediate_state_patching \
  --outcome donor_vs_receiver_city_log_odds \
  --output work/v5_native_count_stream/dev/Qwen3-8B/trace_transport_analysis

python scripts/run_realistic_niah_v5_count_stream.py analyze \
  --trials work/v5_native_count_stream/dev/Qwen3-8B/stream_L20 \
  --experiment-ids stream_state_retention \
  --outcome downstream_item_progress_subspace_retention_score \
  --strata source_scope source_layer \
  --output work/v5_native_count_stream/dev/Qwen3-8B/stream_retention_analysis
```

Inference averages repeated arms/occurrences within request, then requests
within seed. Confidence intervals use a 10,000-draw seed bootstrap and p-values
use a two-sided seed sign-flip test. `analyze --outcome` accepts only registered
higher-is-better endpoints, including donor-vs-receiver local path/city odds,
free-generation donor-city adoption, candidate margin/probability,
negative expected-count error (`expected_count_utility`), conservative strict
greedy utility, and the downstream progress-retention score. Raw error columns
are intentionally rejected so a positive damage/repair contrast cannot silently
reverse meaning.

## Dual-H100 stage-1 launch

The restartable supervisor performs CPU pair planning and basis fitting,
captures D1 attention, builds the two independent source plans, evaluates the
six-K D2 grid, automatically runs K64 only if required, freezes both source
decisions, and finally runs all 350 trace pairs. Assign one model per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash \
  scripts/supervise_realistic_niah_v5_native_count_stream_stage1.sh \
  Qwen3-8B > qwen-count-stream-launch.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash \
  scripts/supervise_realistic_niah_v5_native_count_stream_stage1.sh \
  Gemma4-E4B > gemma-count-stream-launch.log 2>&1 &
```

`stage1_complete.json` is written only after 100 D1 capture shards, the frozen
K decision(s), 350 trace-patch shards, and the 1,750 trace condition rows pass
the supervisor audits. Re-running the same command resumes atomic shards.

## Confirmation gate and interpretation

All seeds 1234--1263 have already informed V5 design choices and remain
development. The fresh-confirmation registries are now frozen at native-unseen
seeds 1316--1335. Confirmation generation may begin only after stage 1 has
written the K decision; answer-time confirmation uses five counts per seed and
only each source's `frozen_answer_broad_head_plan.csv`. A source with
`no_positive_discovery_bank` receives no confirmation ablation and is reported
as a development-frozen null branch. The runner rejects a confirmation plan
without the K-selection decision hash and lock columns.

The decision ledger keeps three component claims separate:

| Component | Required positive contrasts |
| --- | --- |
| Stream-written state | past-donor full patch vs self patch; past-donor projected patch vs norm-matched orthogonal patch |
| Answer-time trace retrieval | trace-item-ranked selected broad bank vs layer-matched random bank |
| Answer-time prompt retrieval | independently prompt-record-ranked selected broad bank vs layer-matched random bank |

Passing more than one row supports a hybrid mechanism. Failing one component
does not prove the other is unique. The terms **internal counter** and **unique
circuit** remain disallowed without marker-neutral signed state transport,
multi-k generalization, and an ordered source-to-output mediation chain. A
successful future-to-past patch is evidence that occurrence states are
interchangeable under intervention, not evidence that the model naturally had
access to a future state.
