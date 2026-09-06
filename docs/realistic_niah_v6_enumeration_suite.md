# Realistic NIAH V6: index/bullet full-suite replication

## Scope and invariants

V6 is an isolated replication of the Native-thinking report's complete
analysis path, represented internally by 20 auditable mechanism frames, on the
two structured enumeration prompts originally specified in `plan.tex`:
`enumeration_index` and `enumeration_bullet`. V5 source code and V5 artifacts
remain frozen. V6 reuses the audited V5 numerical kernels through a
process-local adapter that replaces only prompt/container-sensitive encoding,
enumeration parsing, provenance schema labels, and cohort selection.
Where a frozen script writes a literal V5 internal-kernel schema string, that
source-family label is intentionally retained; the adjacent
`v6_adapter_manifest.json` is the authoritative V6 protocol provenance.

The controlled factor is the enumeration grammar. Everything else is frozen:

- V4.4 stimuli from `work/nonthinking_report_filestream_stage3/stimuli.jsonl`;
- source SHA256
  `da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090`;
- counts 1--10;
- discovery seeds 1234--1253 and confirmation seeds 1254--1263;
- Qwen3-8B and Gemma4-E4B at their registered revisions;
- greedy decoding, the report-matched 4096-token ceiling, and thinking disabled;
- all decoder layers for representation/patch curves and all heads for head
  localization before discovery-only selection;
- the same self, random/layer-matched, orthogonal/norm-matched, top-K, span,
  single-layer, and cumulative-layer controls as the Native-thinking suite;
- the same endpoint-specific populations as Native-thinking: the original
  source-identical all-sample panel for direct grammar/representation contrasts,
  and the resolved strict-format cohort for cellwise causal replication;
- seed as the independent unit, with within-seed contrasts followed by equal
  seed weighting.

Greedy decoding removes sampling variation, but low-level GPU kernels may
still be nondeterministic on some driver/CUDA combinations; every run logs the
checkpoint revision, command, and elapsed time, while supervisor-run phases
also log the driver/GPU inventory, so such differences remain auditable.

The strict formal cohort requires the requested marker grammar, exact
city-score pairs in passage order, contiguous ordinary indices for index mode,
a matching `Total`, no extra text, and a forward one-to-one parser alignment.
Gold count and final `Total` never construct or pad a trace sequence.
Every downstream mechanism runner also rejects generation inputs whose V6
schema, protocol, mode, thinking-off flags, decoding contract, config hash, or
registered checkpoint metadata do not match the active run.

The machine-readable alignment contract is
`configs/realistic_niah_v6_native_analysis_alignment_v1.json`. It forbids a
generic ten-site representation scan from serving as primary evidence and
forbids a direct grammar-effect claim on source-mismatched replacement rows.
The already-frozen discovery representation artifact remains byte-stable as
historical provenance only.

The original 20 discovery and 10 confirmation seeds remain the all-sample
panel. For the strict formal cohort, a generation/runtime failure or a fresh
V6 strict-parser failure triggers deterministic seed replacement. Replacement
is resolved independently for each `model × prompt mode × split × count` cell:
successful original rows remain in their original analysis slots, while each
failed slot receives the lowest strictly eligible seed from its frozen,
split-specific amendment reserve pool. The row keeps its true source seed; the
original seed is stored separately as `analysis_slot_seed`, so replacements
never create false within-seed pairing. No hidden state, attention score, head rank,
source-write magnitude, intervention result, or causal effect may participate
in replacement. Every original failure, failed reserve attempt, selected
replacement, and source-to-slot mapping is written to an immutable ledger.
Pool exhaustion fails closed.

One registered subdesign needs a stricter replacement unit. Broad-head K
selection and its held-out confirmation first aggregate exactly five requests
within each seed, then bootstrap across seeds. A count-wise replacement cannot
be silently inserted into that cluster: doing so would manufacture a
within-seed trajectory that no model run produced. The separately frozen
`configs/realistic_niah_v6_coherent_broad_replacement_policy.json` therefore
uses a whole-panel rule. If any of a broad slot's five registered original
cells fails, all five are supplied by the lowest unused reserve seed that
fresh-strict-passes all five counts. Successful original cells displaced only
to preserve source-seed coherence are listed in `coherent_mapping.jsonl`.
Broad K/confirmation statistics continue to group on the true source seed,
and two slots may not share a source seed. Pool exhaustion fails closed.

The preceding attention-only broad ranking contains no behavioral or
intervention outcome. It retains the cell-resolved 10-by-10 fixed panel and
weights the ten original analysis slots equally; its manifest labels this as
`request_first_then_analysis_slot_equal_attention_only`. It is not reported as
a seed-level effect estimate. This distinction avoids both silent sample loss
and the false claim that a cell replacement shares the original source seed.

This is a prospective, user-authorized protocol amendment made after the
original model generations (and possibly some foundation/source-write
artifacts) already existed. It is not described as part of the original V6
preregistration. The reserve seed order was frozen before any reserve model
output, and the replacement code cannot read the pre-existing foundation or
source-write artifacts. The observed original failure rate may inform only the
capacity of the reserve pool, never candidate order or eligibility.

The frozen amendment policy is
`configs/realistic_niah_v6_replacement_policy.json`: discovery candidates are
1264--1363 and confirmation candidates are 1364--1413. Reserve stimuli use the
same deterministic V4.4 builder. Construction must first regenerate seed 1234
and seed 1254 and exactly match all ten frozen source rows. Confirmation reserve
stimuli may be constructed before discovery finishes because they contain no
model outcomes; confirmation generations remain behind the ordinary discovery
freeze. Once the strict quota is resolved, sparse trace-state panels again use
their registered ten seeds per count/offset cell rather than an adaptive
shortfall.

### Post-exhaustion reserve amendment 1 (2026-08-29)

The original discovery reserve failed closed during the
`Gemma4-E4B × enumeration_bullet × native_loop_discovery` coherent-trajectory
resolution. Eighteen original analysis slots required replacement, but only
fourteen seeds in the complete frozen 1264--1363 prefix passed the fresh strict
parser at every registered count 2--10. No hidden state, attention, head rank,
source-write value, intervention result, or causal effect was opened for this
decision. In accordance with the user's explicit authorization to add and
document seeds for failed samples, amendment 1 freezes discovery extension
seeds 1414--1513 in ascending order before generating any output for those
extension seeds. Seeds 1364--1413 remain exclusively assigned to confirmation;
the confirmation cohort, K choices, required counts, eligibility rule, and
statistical identity are unchanged. The capacity (100 additional candidates)
uses only the observed strict-parser pass/fail rate; it does not change
candidate order or the first-eligible stopping rule.

The machine-readable policies are
`configs/realistic_niah_v6_replacement_policy_amendment1.json` and
`configs/realistic_niah_v6_coherent_native_loop_replacement_policy_amendment1.json`.
The base policy SHA-256, exhaustion trigger, 14/18 resolved state, four-slot
shortfall, extension order, unchanged confirmation pool, and outcome firewall
are embedded in those files. The amended reserve stimulus manifest must bind
to the amendment-policy hash and must repeat the two exact V4.4 anchor
regeneration checks before the resumed model run.

Legacy count-stream kernels originally used one `seed` field for both frozen
panel membership and statistical identity. The V6 adapter separates these
roles without changing the numerical kernel: `analysis_slot_seed` determines
only which pre-existing discovery/confirmation and broad-panel cell a row
fills, whereas every trial, aggregation input, and result continues to use the
true source `seed`. The adapter records `seed_aliasing: false` in its cohort
audit. Experiments that require a coherent multi-count trajectory never splice
independently replaced cells into an original seed. Following the user's
explicit failure-top-up authorization, the native-loop panel has its own
auditable whole-trajectory amendment. If any required count 2--10 fails for an
original native-loop slot, the complete nine-count trajectory is replaced by
the lowest unused role-specific reserve seed that fresh-strict-passes every
required count. The original slot remains `analysis_slot_seed`; all trials and
inference use the one real reserve `source_seed`. Successful original counts
displaced solely for coherence are listed explicitly. This amendment is
frozen in
`configs/realistic_niah_v6_coherent_native_loop_replacement_policy.json` and is
reported as a post-failure user-authorized amendment, not retroactively as
part of the initial preregistration.

The count-basis fit uses the same separation. Membership in the discovery
block is checked with `analysis_slot_seed`, but every captured observation
keeps its true source request and seed. Its NPZ sidecar records both the 20
analysis slots and the actually observed source seeds. Broad-head trial rows
also carry both identities; the K selector refuses a panel unless each of its
ten analysis slots maps to exactly one distinct true source seed across the
registered five counts.

The specialized-kernel wrapper installs this same slot-membership adapter into
the V5 count-stream helper imported by targeted counter-write, stratified NCC,
direct count-margin, count-geometry, and terminal-bridge runners. Thus an
anchor registry containing a reserve source request still resolves all fixed
analysis slots, while every specialized output and seed-level aggregation uses
the true source seed. The frozen V6 timing panels label the discovery split
`discovery`, whereas the inherited stratified-NCC and direct-margin CLIs call
the same split `development`. For those two kernels only, the wrapper writes a
process-local panel view that translates that one role string. It verifies the
complete slot-to-source mapping before dispatch, changes no other field, reads
no intervention outcome, and records both panel hashes in the adapter manifest.

Legacy causal kernels also use `causal_development_seeds` and
`causal_confirmation_seeds` as hard membership filters. When a resolved cohort
registry is supplied, the V6 causal wrapper constructs a process-local config
view that adds every selected replacement's true source seed to the matching
filter. It does not edit the frozen config, `discovery_seeds`,
`confirmation_seeds`, or any generation row. Seed-based cross-fitting and
aggregation therefore retain the true source seed as the statistical identity,
while `analysis_slot_seed` remains only the fixed-quota panel identity. Each
causal adapter manifest records the original and effective membership sets,
the registry hash, added source seeds, and `seed_aliasing: false`. Even
`causal-plan`, which has no `--generations` argument, must receive the same
`--cohort-registry` so this membership contract cannot silently revert. A
confirmation causal command uses its confirmation registry as
`--cohort-registry` and also passes the already frozen discovery registry as
`--causal-membership-registry`; the process-local view then contains the true
source identities from both disjoint roles without modifying either registry.
For V5 causal commands whose frozen bank plan contains canonical
`validation_seeds`, the wrapper also creates a derived confirmation-only CSV
view: a replacement source seed is added to the same validation fold as its
`analysis_slot_seed`. No head, score, condition, training seed, or discovery
artifact is changed, and one source seed mapping to multiple folds is fatal.
The adapter manifest hashes both the original frozen plan and this routing-only
view and records that no confirmation intervention outcome was read.

V6 source-specific writes are an atomic `shards/*.jsonl` directory. The V5
source-write planner can consume that directory, but its public dispatcher
historically attempted a CSV-header probe first. The V6 causal wrapper installs
a process-local directory dispatcher at that boundary; ordinary file inputs
still use the unchanged V5 dispatch path. If a downstream phase fails after a
source bank is complete, reuse is allowed only after
`audit_realistic_niah_v6_completed_source_write_resume.py` verifies the exact
model/mode/anchor, adapter and cohort hashes, generation-view identity, positive
task quota, and one nonempty shard per completed task. Such reuse is logged as
infrastructure recovery, not a failed sample, and performs neither deletion nor
model-output recomputation.

## Code map

- `src/realistic_niah_v6/spec.py`: frozen protocol and decoding specification.
- `src/realistic_niah_v6/generation.py`: exact shared prompt rendering,
  thinking-disabled chat container, greedy generation, timing, and hashes.
- `src/realistic_niah_v6/parsing.py`: strict index/bullet parser and registered
  token sites.
- `src/realistic_niah_v6/encoding.py`: enumeration-specific answer termination
  and causal encodings.
- `src/realistic_niah_v6/kernel.py`: process-local V6 adapters around frozen V5
  numerical kernels.
- `src/realistic_niah_v6/count_stream.py`: outcome-blind balanced-panel adapter
  for strict-cohort trace-patch shortfalls.
- `src/realistic_niah_v6/replacement.py`: reserve-pool validation, fresh-parser
  eligibility, deterministic slot mapping, and resolved-cohort materialization.
- `src/realistic_niah_v6/suite.py`: one-to-one registry for the report's 20
  experiment frames and the discovery/confirmation freeze contract.
- `src/realistic_niah_v6/native_aligned_representation.py`: the two primary
  Native-thinking representation endpoints, exact four-cell identity checks,
  discovery-only layer selection, and held-out confirmation evaluation.
- `src/realistic_niah_v6/completion.py`: four-cell, 20-frame fail-closed
  completion audit, including all ordinary/coherent seed attempt ledgers and
  infrastructure-only recovery events.
- `src/realistic_niah_v6/reporting.py`: self-contained four-cell HTML result
  report with source hashes and the complete failure/replacement appendix.
- `scripts/run_realistic_niah_v6.py`: preflight, restartable generation,
  capture, attention, legacy discovery representation provenance, audits, and
  freezing.
- `scripts/analyze_realistic_niah_v6_native_aligned_representation.py`: shared
  CPU analysis over all four original all-sample capture indexes.
- `scripts/run_realistic_niah_v6_count_stream.py`: all count-stream, loop,
  restoration, and factorial subcommands under V6.
- `scripts/run_realistic_niah_v6_causal.py`: source-write ranking, frozen-bank
  planning, head necessity, activation/subspace patching, and causal analyses.
- `scripts/run_realistic_niah_v6_kernel.py`: allow-listed specialized report
  kernels under V6.
- `scripts/supervise_realistic_niah_v6_enumeration.sh`: restartable GPU
  foundation phases.
- `scripts/build_realistic_niah_v6_replacement_seed_pool.py`: exact V4.4
  reserve stimulus construction with two anchor-regeneration checks.
- `scripts/run_realistic_niah_v6_replacement_generation.py`: adaptive
  format-only generation until every strict cell reaches its fixed quota.
- `scripts/run_realistic_niah_v6_broad_panel_replacement.py`: whole-seed
  five-count supplementation for broad K selection/confirmation and
  nine-count supplementation for native-loop trajectories.
- `scripts/analyze_realistic_niah_v6_native_loop.py`: preserves the frozen V5
  estimands while validating and aggregating coherent true-source trajectories.
- `scripts/freeze_realistic_niah_v6_confirmation.py`: hashes every discovery
  choice, selected bank/layer, coherent registry, and negative result before
  any confirmation kernel may run.
- `scripts/analyze_realistic_niah_v6_specialized_confirmation.py`: paired
  discovery/confirmation analysis for counter write, timing-stratified NCC,
  direct count margin, count geometry, terminal bridge, and token ablations.
- `scripts/audit_realistic_niah_v6_completed_source_write_resume.py`:
  fail-closed validation before a completed source-write shard bank may be
  reused after a downstream infrastructure failure.
- `scripts/audit_realistic_niah_v6_suite_completion.py` and
  `scripts/build_realistic_niah_v6_enumeration_report.py`: final machine-readable
  audit and self-contained HTML deliverable.

The resolved mode configs are:

- `configs/realistic_niah_v6_enumeration_index.json`
- `configs/realistic_niah_v6_enumeration_bullet.json`
- `configs/realistic_niah_v6_enumeration_index_count_stream_dev.json`
- `configs/realistic_niah_v6_enumeration_bullet_count_stream_dev.json`
- `configs/realistic_niah_v6_replacement_policy.json`
- `configs/realistic_niah_v6_coherent_broad_replacement_policy.json`
- `configs/realistic_niah_v6_coherent_native_loop_replacement_policy.json`
- `configs/realistic_niah_v6_targeted_retrieval_report_contract.json`

## CPU preflight

Use the project environment (or another environment satisfying
`requirements.txt`):

```bash
export PYTHONPATH="$PWD/src"
.venv/bin/python -m pytest -q tests/test_realistic_niah_v6.py

for mode in index bullet; do
  .venv/bin/python scripts/run_realistic_niah_v6.py preflight \
    --config "configs/realistic_niah_v6_enumeration_${mode}.json" \
    --output "work/v6_preflight_${mode}.json"
  .venv/bin/python scripts/run_realistic_niah_v6.py suite-audit \
    --config "configs/realistic_niah_v6_enumeration_${mode}.json" \
    --output "work/v6_suite_audit_${mode}.json"
done
```

The preflight must report 300 unique cells per mode: 200 discovery and 100
confirmation. A hash mismatch, duplicate/missing cell, span mismatch, or absent
experiment entrypoint is fatal.

## GPU foundation runs

The supervisor accepts one mode, one model, and one phase. Set the Python,
model cache, run root, and visible GPU explicitly on a rented host:

```bash
export V6_ROOT=/path/to/Realistic_CoT_NiaH_Count
export V6_PYTHON=/path/to/venv/bin/python
export V6_CACHE=/path/to/huggingface-cache
export V6_RUN_ROOT=/path/to/runs/enumeration_index/Qwen3-8B
export V6_CUDA_VISIBLE_DEVICES=0

bash scripts/supervise_realistic_niah_v6_enumeration.sh \
  index Qwen3-8B discovery-generate
bash scripts/supervise_realistic_niah_v6_enumeration.sh \
  index Qwen3-8B discovery-foundation
```

Run the same two phases for `(index, Gemma4-E4B)`, `(bullet, Qwen3-8B)`, and
`(bullet, Gemma4-E4B)`. Generation writes one atomic shard per stimulus and
merges compatible shards into `generations.jsonl`; rerunning resumes safely.
Capture uses the existing shard/reuse contract. Logs, elapsed times, model and
config hashes, Git state, exclusions, and resolved settings are saved with the
artifacts.

Build the common reserve stimulus pool once, then resolve and rebuild the
formal foundation for each mode/model. The original all-sample capture is not
replaced:

```bash
.venv/bin/python scripts/build_realistic_niah_v6_replacement_seed_pool.py \
  --v6-config configs/realistic_niah_v6_enumeration_index.json \
  --replacement-policy configs/realistic_niah_v6_replacement_policy.json \
  --source-stimuli work/nonthinking_report_filestream_stage3/stimuli.jsonl \
  --cache-dir /path/to/huggingface-cache \
  --output /path/to/runs/replacement_seed_pool

export V6_REPLACEMENT_POOL=/path/to/runs/replacement_seed_pool
bash scripts/supervise_realistic_niah_v6_enumeration.sh \
  index Qwen3-8B discovery-supplement
bash scripts/supervise_realistic_niah_v6_enumeration.sh \
  index Qwen3-8B discovery-foundation-resolved
```

The formal registry is
`replacement/discovery/selected_cells.jsonl`; its sibling
`replacement_mapping.jsonl`, `attempt_ledger.jsonl`, and `manifest.json`
provide the complete audit trail.

The foundation phase captures formal and all-sample views separately. After
confirmation is generated, `confirmation-foundation` rebuilds the capture
indexes over both seed blocks but does not run a per-cell generic
representation sweep. Once all four original all-sample indexes exist, the
final-audit queue runs one shared Native-aligned analysis:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  .venv/bin/python \
  scripts/analyze_realistic_niah_v6_native_aligned_representation.py \
  --run-root work/realistic_niah_v6
```

Its running endpoint is `item_end` on exact common
`(split, seed, gold_count, occurrence)` support across all four cells. Its
final endpoint is `answer_query_v3` on the exact common 300-trajectory grid.
Each model×grammar cell selects its layer independently using only five-fold
grouped discovery CV with StandardScaler + whitened PCA16; confirmation is
evaluated at every layer but never enters selection.

## Report-aligned causal kernels

Print the machine-readable 20-frame registry:

```bash
.venv/bin/python scripts/run_realistic_niah_v6.py print-suite \
  --output work/realistic_niah_v6_suite.json
```

Count-stream experiments retain the complete V5 subcommand grammar. Replace
the old runner/config with the V6 wrapper/config; all remaining arguments are
unchanged. For example:

```bash
.venv/bin/python scripts/run_realistic_niah_v6_count_stream.py trace-patch \
  --v6-config configs/realistic_niah_v6_enumeration_bullet.json \
  --cohort-registry /path/to/replacement/discovery/selected_cells.jsonl \
  --mechanism-config configs/realistic_niah_v6_enumeration_bullet_count_stream_dev.json \
  --model Qwen3-8B \
  --generations /path/to/bullet/Qwen3-8B/generations.jsonl \
  ...the same frozen V5 trace-patch arguments...
```

The late V5 trace-patch and native-loop numerical kernels require the
grammar-aware `progress_transition` record (`from_occurrence`, fixed target
token span, and grammar pair), although the public V5 dispatcher still routes
that name through its earlier marker-only compiler. V6 corrects only this
process-local dispatch to the already existing frozen grammar-aware compiler.
Patch states, layers, model hooks, controls, and estimands are unchanged. A
resume after this structural dispatch failure reuses complete discovery
broad-K decisions, retains any completed shards, and records an infrastructure
recovery rather than a model-sample failure or seed replacement.

The sparse trace-patch plan also contains two registered endpoint types. Local
pairs have a successor city and use `donor_vs_receiver_city_log_odds`; the
count-2 terminal panel is explicitly `terminal_panel_answer_only`, has no
successor-city outcome, and uses `correct_count_margin`, matching the
Native-thinking final-answer score. The inherited V5 analysis CLI accepts only
one shared `--outcome`, so V6 routes these two experiment IDs to their frozen
endpoints and then concatenates the unchanged V5 contrast summaries. This
routing is determined from experiment topology and registry flags, not effect
magnitudes, and is identical in discovery and confirmation. A resume after the
shared-outcome analyzer rejects the answer-only terminal rows reuses every
completed model trial, runs analysis only, and is recorded as infrastructure
recovery rather than a sample failure or seed replacement.

The specialized frames use an allow-listed kernel adapter. It materializes a
mode-pure strict cohort, injects the V6 config where required, installs the V6
encoding/schema adapters, and writes an adapter provenance manifest next to
the output:

```bash
.venv/bin/python scripts/run_realistic_niah_v6_kernel.py \
  --target targeted-counter-ncc \
  --v6-config configs/realistic_niah_v6_enumeration_index.json \
  --cohort-registry /path/to/replacement/discovery/selected_cells.jsonl \
  --phase discovery -- \
  --model Qwen3-8B \
  --generations /path/to/index/Qwen3-8B/generations.jsonl \
  ...the same frozen V5 NCC arguments...
```

Supported specialized targets are listed by
`scripts/run_realistic_niah_v6_kernel.py --help`. The specialized causal
replications use the resolved strict formal cohort. `--include-nonstrict` is
reserved for a specifically registered endpoint and is not a universal second
analysis view.

The general causal ladder retains the V5 causal subcommand grammar. V6 forbids
an implicit cross-model K default. The targeted-retrieval subprotocol reruns
the exact final-report grids: Qwen `32,64,80,96,112,128` and Gemma
`1,2,4,6,8`. Qwen K32--K112 and every Gemma dose use three layer-matched random
banks; the cited Qwen merged-grid source artifact registers three
`global_random` banks at K128. The prose legend in the Native-thinking HTML is
less specific at that endpoint, so V6 records both the HTML hash and the
machine-readable merged-grid hash and follows the latter's actual condition.

The frozen base mode configs remain byte-stable because their hashes already
appear in completed foundation artifacts; they contain an earlier
overinclusive Qwen K125 entry. K125 failed a structural preflight before any
targeted behavior was produced because an exact layer-matched control could not
be constructed. This is neither a model-sample failure nor a seed-replacement
event. The narrower, outcome-blind targeted contract supersedes only the dose
grid/control-family fields and excludes K125.

The report-reference endpoints are Qwen K128 and Gemma K6, but they are audit
references rather than forced downstream choices. For each prompt mode and
model independently, V6 computes the seed-equal discovery contrast
`selected failure - mean(three registered-random failures)` at every dose,
maximizes that primary estimand, and breaks an exact tie toward smaller K. Thus
K96 or K112 is used if it wins for a Qwen mode. The chosen K and random-control
family are then frozen for every downstream discovery kernel and for
confirmation; confirmation cannot reselect K, and a negative discovery result
is retained.

Once the discovery dose rule freezes a K, every targeted follow-up kernel must
pass that mode/model `selection.json` through the V6 `--bank-selection`
adapter. The adapter verifies the exact selected full-panel source-plan hash
and installs process-local K and condition-label compatibility across all
legacy consumers. The discovery-selected treatment heads are immutable. For a
layer-matched plan, the complete source plan is copied byte-for-byte. For the
Qwen K128 `global_random` endpoint only, a downstream structural builder checks
whether a random head reaches or exceeds `max(selected layer)+1`, which would
leave no legal post-intervention NCC capture layer. It deterministically
replaces only those unreachable random-control heads from the globally sampled
capture-reachable complement; K, selected heads, control family, source layer,
hooks, and numerical interventions do not change. This construction reads no
behavior, specialized, or confirmation outcome and is recorded as structural
control adaptation, not a sample failure or seed replacement. The
report-reference plan remains the frozen selection audit, and all V5 source
files stay unchanged.

```bash
.venv/bin/python scripts/run_realistic_niah_v6_causal.py \
  --v6-config configs/realistic_niah_v6_enumeration_index.json \
  --model-label Qwen3-8B \
  --phase discovery \
  --cohort-registry /path/to/replacement/discovery/selected_cells.jsonl -- \
  causal-plan --source-writes /path/to/source_writes \
  --bank-size 128 --selection-aggregation seed_event_mean \
  --random-control-matching global --output /path/to/k128_plan
```

## Discovery/confirmation firewall

Confirmation generation and confirmation count-stream runs are deliberately
locked. Initialize a per-mode, per-model discovery ledger:

```bash
.venv/bin/python scripts/run_realistic_niah_v6.py init-discovery-ledger \
  --config configs/realistic_niah_v6_enumeration_index.json \
  --model Qwen3-8B \
  --output /path/to/run/discovery_ledger.json
```

After all discovery choices are made, fill every registered `choice` and
`artifact_paths`; set each cell to `FROZEN` or `NEGATIVE_FROZEN`, retain null
results, set the top-level status to `DISCOVERY_FROZEN`, and freeze the exact
artifacts:

```bash
.venv/bin/python scripts/run_realistic_niah_v6.py freeze-confirmation \
  --config configs/realistic_niah_v6_enumeration_index.json \
  --model Qwen3-8B \
  --discovery-ledger /path/to/run/discovery_ledger.json \
  --artifacts /path/to/run/frozen_bank.json /path/to/run/frozen_layers.json \
  --output /path/to/run/confirmation_freeze.json \
  --mechanism-config configs/realistic_niah_v6_enumeration_index_count_stream_dev.json \
  --frozen-mechanism-output /path/to/run/count_stream_frozen_confirmation.json
```

Then run confirmation:

```bash
export V6_CONFIRMATION_FREEZE=/path/to/run/confirmation_freeze.json
bash scripts/supervise_realistic_niah_v6_enumeration.sh \
  index Qwen3-8B confirmation-generate
bash scripts/supervise_realistic_niah_v6_enumeration.sh \
  index Qwen3-8B confirmation-supplement
bash scripts/supervise_realistic_niah_v6_enumeration.sh \
  index Qwen3-8B confirmation-foundation-resolved
```

`confirmation-supplement` applies the same frozen-amendment rule behind the
discovery freeze and writes `replacement/confirmation/selected_cells.jsonl`.
The resolved foundation then materializes the discovery and confirmation
registries together for the formal 300-cell capture index. The all-sample
capture remains the untouched original 20+10 seed panel, while confirmation
attention and answer-query outputs use only the resolved 10-slot-per-count
confirmation registry.

The discovery capture and legacy representation under `capture/formal/` and
`representation/formal/` remain immutable because their hashes are already in
the discovery freeze. Held-out capture indexes are written to
`capture/confirmation_formal/`; the untouched original panel is written to
`capture/confirmation_all_sample/`. Primary representation results are written
only after all four cells exist, under each cell's
`representation/native_aligned/` plus the shared
`native_aligned_representation/` root. No new generic
`representation/confirmation_*` scan is required. The
confirmation-foundation supervisor still validates every frozen discovery hash
before and after held-out capture/attention writes.

Pass the same freeze to confirmation kernel wrappers. The runner verifies the
mode, model, seed blocks, freeze hash, `confirmation_outcomes_read=false`, and
every frozen discovery artifact hash before dispatch.

## Full-suite completion audit

After both model queues finish both modes, run the fail-closed aggregate audit:

```bash
.venv/bin/python scripts/audit_realistic_niah_v6_suite_completion.py \
  --run-root /path/to/runs/v6_enumeration_replication_20260828 \
  --output /path/to/runs/v6_enumeration_replication_20260828/final_audit
```

The audit requires all four model-by-mode cells, every discovery and
confirmation completion manifest, all 20 report-frame evidence routes, and
the hash-locked discovery freeze. It independently checks the ordinary
cell-level replacement ledgers, the five-count broad-K coherent panels, and
the count-2-through-10 native-loop coherent panels. Its JSON report lists
every failed original cell, every failed reserve attempt, every selected
reserve seed, and every otherwise-successful original count displaced only to
keep a true-source trajectory coherent. Infrastructure marker recovery is
reported separately and is never counted as a sample failure. Canonical reserve
attempt rows use `candidate_kind=replacement`; the auditor also accepts the
early wording `reserve`, so failed reserve samples cannot silently disappear
from the aggregate count. Coherent source-identity skips are listed separately
from runtime/parser failures. The audit fails
closed on a missing file, hash mismatch, reused true-source seed within a
registered panel, incomplete fixed-slot quota, seed alias, or silent sample
exclusion.

Only after that audit returns `PASS_FULL_V6_ENUMERATION_SUITE`, build and
structurally validate the 20-frame × four-cell report:

```bash
.venv/bin/python scripts/build_realistic_niah_v6_enumeration_report.py \
  --run-root /path/to/runs/v6_enumeration_replication_20260828 \
  --completion-audit /path/to/runs/v6_enumeration_replication_20260828/final_audit/suite_completion_audit.json \
  --output /path/to/runs/v6_enumeration_replication_20260828/final_report/NiaH_V6_Index_Bullet_Replication_report.html
```

The validator requires frames 1--20 in order, exactly 80 frame-specific
model×mode result cards, unique HTML IDs, no external script, and no external
URL. The embedded JSON summary and every displayed source artifact carry
SHA-256 provenance.

The frozen coherent native-loop policy SHA-256 is
`e9f7e2cd88a6eba7f97342bc9fdfdf14cc9dacc6fbdd14b985105e67a8c571fa`.

## User-added answer/trace extension

The original 20-frame suite remains unchanged. A separately labelled
registered existing-split extension adds causal full-state answer-query
patching and the terminal trace-to-answer partial-mediation factorial for both
prompt modes and both models. (The observational `answer_query_v3` geometry is
now already part of the primary Native-aligned representation path.) Its frozen
contract is `configs/realistic_niah_v6_answer_trace_extension_v1.json`.

The extension first resolves a dedicated confirmation registry that is
true-source coherent across every count 1--10. If any original row in a slot
fails runtime or fresh strict parsing, the complete slot trajectory is moved to
the lowest available, unused confirmation reserve seed. This decision reads no
hidden state, attention, head rank, patch result, relay result, or effect size.
Scientific gate failure never triggers seed replacement.

The queue waits for both primary confirmation queues and the exploratory index
item-end sensitivity before using GPU0:

```bash
bash scripts/queue_realistic_niah_v6_answer_trace_extension.sh
```

It writes per-cell artifacts under
`causal/answer_trace_extension_v1/` and a self-contained four-cell supplement
under `answer_trace_extension_report/`. The supplement reports answer donor
adoption and terminal-relay gates while explicitly prohibiting complete
mediation, exclusive answer-query mediation, and memoryless `+1` claims.

## Validation status

CPU validation covers both exact prompt contracts, thinking-disabled chat
rendering, strict and negative parser cases, token-site alignment, answer
termination scoring, the frozen 300-cell stimulus panel, all 20 experiment
frames, both count-stream configs, the confirmation firewall, and the
fail-closed generation provenance and no-V5-source-mutation invariants. GPU
numerical equivalence cannot be asserted
until the four discovery jobs and their confirmation phases are run on the
registered model checkpoints.
