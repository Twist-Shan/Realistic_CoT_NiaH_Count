# Realistic NIAH V5: native-thinking representation and causal analysis

## Scope and scientific translation

V5 reproduces the evidence chain in
`reports/NiaH_Non-thinking_report.html` on native-thinking traces.
It uses the frozen V4.4 stimuli and seed split, but it does **not** use V4.4.2
or later native-thinking experiments as data, code dependencies, or sources of
discovery choices. Only the pre-existing V4 generic decoder hooks are reused.

The central translation is:

| V4.4 non-thinking construct | V5 native-thinking construct |
| --- | --- |
| Prompt needle-end state | Parser-accepted trace `item_end` state |
| Earlier needle spans | Earlier accepted city-item spans |
| Prompt needle target | Exact frozen prompt city-record span |
| Answer-query state after `Total:` | Native trace `answer_query` boundary |
| Correct prompt cohort | Parser-hit / one-to-one / one-to-one-and-final-correct, reported separately |

`item_end` is the registered primary trace site. `marker_end`, `city_end`,
`post_boundary`, `list_cut`, and `answer_query` are fixed sensitivity sites.
They must not be searched and then selectively reported using confirmation
results. The parser's `cut_char` means that the last accepted city item is
visible; it does not prove that a later textual `(Count: N)` statement is the
model's unique internal commitment.

## Parser identity

V5 directly vendors the three frozen algorithm files from
[`TheWayLost/niah-parser`](https://github.com/TheWayLost/niah-parser) at commit
`8ebf6b7af4770d8c91e6540d474505e23ad57c8c`. The parser is intentionally
gold-assisted: registered gold cities are prior information used to identify
city-list items. That makes it an experimental alignment instrument, not a
deployable blind parser.

`v26_v44_native_9000` is retained only as the historical archive/report label.
It is not invented as a parser rule ID. Runtime artifacts identify the actual
implementation by repository, commit, entry point, and SHA-256 values recorded
in `provenance/NIAH_PARSER_V5.json`.

The primary `one_to_one` cohort requires the accepted city multiset to equal
the gold city multiset. Reverse and other permutations remain valid. Parser
hit, one-to-one, and final answer correctness are distinct variables.

## Native-thinking head taxonomy

The causal head taxonomy follows the CoT side of the Synthetic V10 report,
not the non-thinking broad-aggregation score. Let `R(c)` be the exact frozen
prompt token span for city record `c`, recovered from V4.4
`active_needle_spans` and the native model tokenizer.

- `targeted_retrieval`: query at parser boundary `marker_end:k`; target is
  `R(c_k)`, where `c_k` is the city accepted as trace item k. The discovery
  score is the query-weighted mean of
  `sum(attention[marker_end:k, R(c_k)])`.
- `progress_transition`: query at `item_end:k`; target is `R(c_{k+1})` for
  nonterminal k. The discovery score is the query-weighted mean of
  `sum(attention[item_end:k, R(c_{k+1})])`. The terminal k is retained only
  as a separate stop-phase causal outcome and does not enter this attention
  ranking because no next prompt record exists.

Every result table must retain `target_needle_raw_mass`,
`all_active_needles_raw_mass`, and `target_needle_relative_mass`, where the
relative mass is the target raw mass divided by the summed raw mass over all
active frozen needle spans at that query. A zero denominator is reported as
undefined (`NaN`), not silently converted to zero. Raw target mass is the only
ranking variable; relative mass and target top-1 are required diagnostics.
The two families are ranked and intervened
on independently because retrieval of the current record and transition to
the next progress state may share heads without sharing the same top-K set.

## Representation battery

`src/realistic_niah_v5/representation.py` runs the following layer/site/cohort
battery:

- stable/effective rank, 90% and 95% energy rank, centroid rank-1/rank-3 share;
- running-index ridge and 5-nearest-neighbor regression;
- kNN, logistic, linear SVM, LDA, and random-forest classification;
- cosine silhouette, Calinski-Harabasz, Davies-Bouldin, and k-means ARI;
- adjacent step norm/cosine, curvature, and label-distance correlation;
- an exact descriptive decomposition into label-centroid signal and
  within-label residual.

For trace sites, the primary representation slice is `N=10`; its label is the
accepted-item occurrence 1 through 10. Answer-query analysis uses gold count
1 through 10. Probe hyperparameters are fitted on discovery seeds 1234-1253
and evaluated once on confirmation seeds 1254-1263. The decomposition is
descriptive and must not be described as identifying a causal source of noise.

## Causal ledger

`src/realistic_niah_v5/causal.py` registers and implements the report-facing
tests:

1. item-end query context masks: clean, trace-only, and matched non-trace-only;
2. trace token corruption with an equal-token-budget ordinary-token control;
3. discovery-frozen count-subspace removal;
4. `targeted_retrieval` head analysis: at parser `marker_end:k`, rank by raw
   attention mass to the prompt record matching accepted city k, then run
   position-local K-dose ablation with layer-matched random banks;
5. `progress_transition` head analysis (the successor-like family in the
   synthetic report): at `item_end:k`, rank by raw mass to the prompt record
   matching the next accepted city k+1, then test the next-item or stop
   continuation locally with the same controls;
6. answer-query and trace-endpoint donor/receiver residual patching;
7. Qwen natural OV write decomposition at the registered answer query;
8. the same exact per-head decomposition aggregated across Gemma layers to
   test distributed residual writing.

Both write analyses use V5's `capture_natural_head_writes`: it slices the
natural attention-weighted head aggregate immediately before each `o_proj`,
maps each slice separately, subtracts projection bias, and audits that summed
head writes reconstruct the full attention write. It does not import any
V4.4.2+ experiment module.

The two head families are frozen independently. Selection uses the
query-weighted mean raw mass to the corresponding semantic prompt-record
target over discovery counts 1-10 in the `one_to_one` cohort, matching the
Synthetic V10 definition while respecting the native parser audit.
It does not use broad aggregation, answer-query-to-trace mass, target
fraction, top-1 rate, or confirmation outcomes as the ranking score.
The head necessity test zeroes selected pre-`o_proj` slices only at the
registered semantic query. Its primary continuous endpoint is the
teacher-forced log probability of the complete target continuation; exact
sequence, first-token accuracy, and first-token rank are secondary endpoints.
Targeted continuations run from `marker_end:k` through `city_end:k`.
Progress continuations run from `item_end:k` through `marker_end:k+1`; the
terminal item is separately scored through `answer_query` as the stop phase.

Statistical tests average all registered query rows within seed and condition
before inference. Layer-matched random repeats are therefore controls, not
independent samples. Bootstrap intervals and sign-flip tests use seed as the
inference unit; mechanism, K, and transition phase must be selected rather
than pooled.

## Frozen shared geometry panel and causal extensions

The exact V4.4 input backbone used by V4 non-thinking is frozen at
[`twistshan/realistic-niah-count-mechanism-analysis`](https://huggingface.co/datasets/twistshan/realistic-niah-count-mechanism-analysis).
Dataset schema v2 exposes one default `geometry_shared` configuration with 200
discovery rows and 100 confirmation rows. Each `pair_id` appears once; passages,
gold records, slots, active needle spans, hard negatives, and design metadata
are therefore not duplicated across modes. `contracts/MODE_CONTRACTS.json`
stores the registered `non_thinking` and `native_thinking` prompt/runtime
contracts once, while each geometry row stores the corresponding rendered
prompt hashes and expected final lines in `mode_views`.

The mode contrast jointly changes the registered prompt contract and the
chat-template thinking control; it is not interpreted as a flag-only
intervention. Mode-specific causal datasets are registered separately in
`causal/REGISTRY.json` and stored under
`data/causal/<mode>/<experiment_id>/<split>.jsonl`. They may add seeds
independently and must declare any shared paired confirmation subset used for
cross-mode causal contrasts. Adding a causal extension must not change
`geometry_shared`.

The local reproducible builder is
`scripts/build_realistic_niah_mechanism_dataset.py`. Its source full-grid
stimulus SHA-256 is
`da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090`,
and its shared V4.4 backbone audit SHA-256 is
`f70a62c0a9cb10d4f80f950566aa321f84fb96df67479b2f60c424f6c300238e`.
Until causal extensions are explicitly registered, the Hub repository contains
inputs and provenance only—no generations, hidden states, attention tables,
representation outputs, or causal results.

## Pipeline

The single entry point is `scripts/run_realistic_niah_v5.py` and the frozen
configuration is `configs/realistic_niah_v5.json`.

```powershell
# 1. Parse an existing native-thinking generation archive.
python scripts/run_realistic_niah_v5.py parse `
  --input work/v5/generations.jsonl `
  --output work/v5/parsed.jsonl

# 2. Generate native traces from frozen V4.4 stimuli (one model per GPU job).
python scripts/run_realistic_niah_v5.py generate `
  --model Qwen3-8B `
  --stimuli work/v4/stimuli.jsonl `
  --output work/v5/qwen/generations.jsonl

# 3. Capture registered sites and all decoder layers.
python scripts/run_realistic_niah_v5.py capture `
  --model Qwen3-8B `
  --generations work/v5/qwen/generations.jsonl `
  --output work/v5/qwen/capture

# 4. Fit discovery probes and evaluate confirmation geometry.
python scripts/run_realistic_niah_v5.py representation `
  --capture-index work/v5/qwen/capture/capture_index.jsonl `
  --output work/v5/qwen/representation

# 5. Capture attention and freeze causal head banks.
python scripts/run_realistic_niah_v5.py attention `
  --model Qwen3-8B `
  --generations work/v5/qwen/generations.jsonl `
  --output work/v5/qwen/attention.csv
python scripts/run_realistic_niah_v5.py causal-plan `
  --attention work/v5/qwen/attention.csv `
  --output work/v5/qwen/causal_plan

# 6. Run confirmation head ablation and token-corruption trials.
python scripts/run_realistic_niah_v5.py causal-heads `
  --model Qwen3-8B `
  --generations work/v5/qwen/generations.jsonl `
  --plan work/v5/qwen/causal_plan/causal_plan.csv `
  --output work/v5/qwen/head_trials.jsonl
python scripts/run_realistic_niah_v5.py causal-analyze `
  --trials work/v5/qwen/head_trials.jsonl `
  --mechanism targeted_retrieval `
  --bank-size 4 `
  --transition-phase retrieve `
  --treatment targeted_retrieval_ranked `
  --control layer_matched_random `
  --outcome target_sequence_log_probability `
  --output work/v5/qwen/targeted_k4_inference.csv
python scripts/run_realistic_niah_v5.py causal-tokens `
  --model Qwen3-8B `
  --generations work/v5/qwen/generations.jsonl `
  --output work/v5/qwen/token_trials.jsonl

# 7. Test whether earlier accepted items are needed at item endpoints.
python scripts/run_realistic_niah_v5.py causal-context `
  --model Qwen3-8B `
  --generations work/v5/qwen/generations.jsonl `
  --site-kind item_end `
  --output work/v5/qwen/context_trials

# 8. Fit count bases on discovery captures only, then ablate on confirmation.
python scripts/run_realistic_niah_v5.py causal-subspace-fit `
  --capture-index work/v5/qwen/capture/capture_index.jsonl `
  --site-kind answer_query `
  --cohort one_to_one `
  --rank 3 `
  --output work/v5/qwen/answer_query_basis.npz
python scripts/run_realistic_niah_v5.py causal-subspace `
  --model Qwen3-8B `
  --generations work/v5/qwen/generations.jsonl `
  --basis work/v5/qwen/answer_query_basis.npz `
  --site-id answer_query `
  --doses 0.25 0.5 1.0 `
  --output work/v5/qwen/subspace_trials.jsonl

# 9. Decompose natural per-head residual writes. The fitted basis supplies
#    one discovery-frozen direction per layer; --save-vectors is optional.
python scripts/run_realistic_niah_v5.py causal-writes `
  --model Qwen3-8B `
  --generations work/v5/qwen/generations.jsonl `
  --directions work/v5/qwen/answer_query_basis.npz `
  --output work/v5/qwen/natural_writes.csv
```

Projected donor patching is batch-driven. `causal-patch --pairs PAIRS.jsonl`
expects `receiver_request_id` and `donor_request_id`; each row may override
`layer`, `receiver_site_id`, `donor_site_id`, `pair_id`, and `donor_role`.
The command emits a self-patch, a discovery-subspace donor patch, and an
equal-norm orthogonal control for every registered pair. Mismatched-donor
pairs must be listed explicitly in the frozen pair file rather than selected
after looking at confirmation outcomes.

`causal-context` writes an index plus one compact state array per
request/site/condition. `causal-subspace-fit` records in its JSON audit that it
uses discovery rows only. If a requested head-bank size cannot support a
same-layer random control (for example because all heads in that layer were
selected), `causal-plan` skips that K and records the exact reason in
`causal_plan_audit.json`; it never schedules an uncontrolled treatment arm.

Run the same stages for `Gemma4-E4B`. GPU jobs are restartable at the capture
shard level. Generation output currently commits after the model loop, so a
production multi-hour run should use an external job supervisor and a
model/seed shard per invocation.

## Reproducibility and limitations

- Formal model/tokenizer revisions remain those in
  `realistic_niah_v4.spec.MODEL_SPECS`.
- Exact character-to-token alignment is mandatory. A boundary that crosses a
  tokenizer unit is represented by text-exact suffix retokenization, with the
  shared baseline prefix and retokenized suffix recorded.
- Full-sequence hidden states are never materialized. Only registered site and
  item-span states are saved.
- The local test command should use the pinned `numpy<2` environment from
  `requirements-mechanistic-v4.txt`; mixed NumPy ABI installations are not a
  valid test environment.
- Passing CPU/unit tests does not substitute for the two-model GPU smoke test,
  exact attention-cache audit, or formal confirmation run.
