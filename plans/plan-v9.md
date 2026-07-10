# Plan

### General instructions
Principles: Read the plan carefully.
- You are encouraged to raise clarification questions about typos, ambiguities, uncertainties, etc. Ask me if you need more information for making decisions.
- Ask to correct obvious typos (with minimal changes) in this document before you code.
- Remember to update the `README.md` after coding
- If I ask you to include the Q&A interaction, summarize our interaction and add your summary under a subsection titled "#### Clarification Q&A (YYYY-MM-DD-time)" right below the requested section
- Don't modify the code block in the notebook that contains the `drive.mount()` command, unless I explicitly tell you to do so.

## Summary of previous plans

Earlier plans built the Dynamic NIAH v2 dataset generator, tokenizer-aware insertion metadata, response scoring, hidden-state extraction, token-level ridge/classifier probes, contrastive-success directions, counterfactual count directions, and steering evaluation. The current counting-feature notebook is the main Colab workflow: it generates or restores a dataset, scores model responses, optionally filters to successful examples through `FILTER_EXAMPLE`, tokenizes prompts, locates matching needle spans, builds token-level count labels, extracts hidden states, fits ridge/classifier probes, optionally computes feature vectors, and can run steering.

Recent plans added exact marker and literal-count variants. `match_count` can use city-score records or exact marker needles. `literal_count` generates a UID-like literal with configurable `UID_TOKEN_LENGTH`, inserts it at word boundaries, verifies the realized token length, and now names the exact literal in the prompt instruction. The most recent runs show that ridge vectors are often useful as probes but disappointing as steering vectors, motivating additional probing diagnostics before more steering experiments.

## Goal: expanded counting probes

Add diagnostic probe modes that reuse the existing counting-feature workflow and I/O. The goal is to test where count information is linearly available:

1. at individual matching occurrences, as a running occurrence index;
2. in a per-example pooled representation of all matching occurrence spans;
3. at the final prompt token that drives answer generation.

These modes should apply to count-like Dynamic NIAH tasks, not only `[dolphin]` marker runs. For `match_count`, `literal_count`, and `count_avg`, the primary target is `gold_answer["count"]`. For `count_avg`, use only the count target, not the average-value target. Ignore `argmax` for this plan.

This is diagnostic first. Save learned ridge vectors for the new probe modes so they can be inspected or reused later, but label them as diagnostic and not yet validated steering directions.

#### Clarification Q&A (2026-06-18-2141)

- Ignore `argmax` for now. Add a brief inline notebook comment near `COUNTING_PROBE_MODE` explaining that the new probes target count-like tasks and do not support `argmax`.
- Include `count_avg` by default, using only `gold_answer["count"]` as the probe target.
- Save ridge vectors for the new probe modes for possible later steering.
- Support both span-last and span-mean needle pooling through separate `COUNTING_PROBE_MODE` values.
- Non-`direct` probe modes replace the existing direct ridge workflow for that run.
- Keep `FILTER_EXAMPLE=True` as the default.
- Add `NUM_MAX_NEEDLES = None`. If it is `None`, the existing fixed-`NUM_NEEDLES` pipeline is unchanged. If it is a positive integer, each generated example uniformly samples a target needle count from `1..NUM_MAX_NEEDLES` and inserts that many needles, overriding `NUM_NEEDLES` only for per-example generation.

## Configuration

Add these notebook/config variables. Each newly introduced global variable in `notebooks/counting_feature_analysis.ipynb` should have a brief inline comment.

```python
COUNTING_PROBE_MODE = "direct"  # Probe workflow; "direct" preserves the existing token-level ridge pipeline.
NUM_MAX_NEEDLES = None          # None keeps fixed NUM_NEEDLES; positive int samples 1..NUM_MAX_NEEDLES per example.
RUN_COUNTING_PROBE_BASELINE = True  # Fit a non-needle positional baseline for non-direct probe modes.
COUNTING_PROBE_BASELINE_MIN_DISTANCE = 5  # Minimum token distance from any needle span for fake baseline positions.
```

Allowed `COUNTING_PROBE_MODE` values:

```python
"direct"                         # current token-level ridge/classifier workflow exactly
"mean_across_examples"           # occurrence-index prototypes and occurrence-index ridge probe
"mean_across_needles_span_last"  # per-example mean of matching span-last vectors -> gold count
"mean_across_needles_span_mean"  # per-example mean of matching span-mean vectors -> gold count
"final_token"               # final prompt token representation -> gold count
"all_diagnostics"                # run all non-direct probe modes
```

Add an inline comment near `COUNTING_PROBE_MODE` that `argmax` is intentionally unsupported in this first implementation. Rename the current implementation string `mean_final_token` to `final_token`; the final-token probe uses one final-token hidden state per example and does not perform any mean operation. Keep `COUNTING_FEATURE_CALC_METHOD` as the higher-level feature-vector/steering selector with existing values:

```python
"ridge" | "contrastive-success" | "counterfactual"
```

The new probe modes should only run with `COUNTING_FEATURE_CALC_METHOD="ridge"`. They should not alter contrastive-success or counterfactual direction calculation.

`direct` is required for backward compatibility. With `COUNTING_PROBE_MODE="direct"` and `COUNTING_FEATURE_CALC_METHOD="ridge"`, the notebook should produce the same files and behavior as the current ridge branch: `target_count_y_t.pt`, `ridge_probe_layer_*.pt`, `ridge_layer_*_eval.json`, plots, `probe_summary.csv`, and classifier outputs when enabled.

Filtering should reuse the existing `FILTER_EXAMPLE` switch. Do not introduce a second filtering variable unless necessary. Metadata must always record whether rows came from successful-only or all-example filtering.

`NUM_MAX_NEEDLES` belongs to dataset generation, not analysis. When `NUM_MAX_NEEDLES is None`, keep every existing command-line argument, resolved config field, and output naming behavior intact. When `NUM_MAX_NEEDLES` is a positive integer, data generation should:

1. sample `target_num_needles_i ~ Uniform({1, ..., NUM_MAX_NEEDLES})` independently for each example;
2. insert exactly `target_num_needles_i` needles for that example;
3. override the fixed `NUM_NEEDLES` value only for per-example generation;
4. save the sampled per-example target count in the generated JSONL and resolved run config;
5. make `gold_answer["count"]` equal to the sampled count for count-like tasks;
6. include `num_max_needles_{NUM_MAX_NEEDLES}` in run naming so variable-count runs are auditably distinct from fixed-count runs.

The CLI generator should expose this as `--num-max-needles`. Config-file and notebook values should resolve consistently with the existing "CLI overrides config" convention.

Update:

- `src/counting/feature_analysis.py` defaults and validation;
- the Dynamic NIAH v2 generation config/dataclass and CLI;
- `configs/counting_analysis.json` config and notes;
- `notebooks/counting_feature_analysis.ipynb` `CONFIG_OVERRIDES`, printed config, metadata, and ridge/probe-analysis blocks;
- tests that check config validation, generation behavior, and notebook source.

## Shared data model

Reuse existing `TokenizedCountingExample` objects from `tokenize_counting_examples`. Do not invent a second span locator. Matching spans should come from:

- `example.needle_segments`;
- `example.matching_needle_ids`;
- `_matching_segments(...)`.

This keeps city-score `match_count`, marker `match_count`, `literal_count`, and `count_avg` aligned with existing span logic. Count-like task labels should be resolved through a helper:

```python
count_probe_target_for_row(row) -> int | None
```

Rules:

- if `row["gold_answer"]["count"]` exists, use it;
- otherwise, return `None` and skip unless an explicit auxiliary-target option is later approved;
- verify `len(matching_spans) == gold_count` for tasks where the count is the number of matching inserted records;
- fail loudly on invalid spans, empty included sets, NaNs, shape mismatches, or missing hidden-state files.

Use `example.sequence_length - 1` for the final prompt token. Existing hidden-state tensors are padded `[n_examples, max_len, hidden_dim]`, and `TokenizedCountingExample.sequence_length` already gives the unpadded length. Do not rely on a global shared sequence length.

Save a shared audit table for non-direct modes:

```text
tables/counting_features/probe_diagnostics/{mode}/probe_examples.csv
```

Columns:

```text
local_example_index,row_id,gold_count,num_matching_spans,span_starts,span_ends,
final_pos,model_exact_match,included,exclude_reason
```

## Mode 1: `mean_across_examples`

Scientific question: do matching occurrence representations form a structured trajectory for first, second, third, ... occurrences?

For each layer and included example, take one vector per matching span:

```python
z_i_j = hidden[layer][i, span_end_j - 1]
```

Then compute occurrence-index prototypes:

```python
mu_j = mean_i z_i_j
```

where examples contribute to `mu_j` only if they have at least `j` matching occurrences. This remains meaningful for old fixed-count runs and becomes stronger for new `NUM_MAX_NEEDLES` variable-count runs.

Always train a linear ridge probe from individual occurrence vectors to occurrence index `j`. Split by example, not by occurrence row. This probe predicts occurrence index, not total count. Save ridge vectors and metadata for possible later steering or geometry inspection.

Outputs:

```text
tensors/counting_features/probe_diagnostics/mean_across_examples/layer_{layer}_prototypes.pt
tensors/counting_features/probe_diagnostics/mean_across_examples/ridge_probe_layer_{layer}.pt
tables/counting_features/probe_diagnostics/mean_across_examples/prototype_metadata_layer_{layer}.csv
tables/counting_features/probe_diagnostics/mean_across_examples/prototype_geometry_layer_{layer}.csv
tables/counting_features/probe_diagnostics/mean_across_examples/predictions_layer_{layer}.csv
tables/counting_features/probe_diagnostics/mean_across_examples/ridge_metrics.csv
tables/counting_features/probe_diagnostics/mean_across_examples/summary.csv
```

Prototype metadata should include layer, occurrence index, number of contributing examples, and contributing count distribution. Geometry should include prototype norm, adjacent cosine, adjacent delta norm, and optional linearity metrics.

## Mode 2: `mean_across_needles_span_last`

Scientific question: is the total count linearly decodable from the pooled representations of the things being counted?

For each example and layer:

```python
x_i = mean_j hidden[layer][i, span_end_j - 1]
y_i = gold_count_i
```

Fit an example-level ridge regression using the existing `train_test_split_indices`, `fit_ridge_probe`, `evaluate_ridge_probe`, or a thin wrapper around the same standardization logic. The split unit is the example, not tokens.

Outputs:

```text
tensors/counting_features/probe_diagnostics/mean_across_needles_span_last/layer_{layer}_X.pt
tensors/counting_features/probe_diagnostics/mean_across_needles_span_last/y.pt
tensors/counting_features/probe_diagnostics/mean_across_needles_span_last/ridge_probe_layer_{layer}.pt
tables/counting_features/probe_diagnostics/mean_across_needles_span_last/predictions_layer_{layer}.csv
tables/counting_features/probe_diagnostics/mean_across_needles_span_last/ridge_metrics.csv
tables/counting_features/probe_diagnostics/mean_across_needles_span_last/summary.csv
```

Metrics should include train/test R2, MAE, RMSE, rounded count accuracy, alpha, layer, number of train/test examples, and filter status.

## Mode 3: `mean_across_needles_span_mean`

Scientific question: does averaging across all tokens inside each matching needle span give a better total-count representation than using only the span-last token?

For each example and layer:

```python
span_vec_i_j = mean_t hidden[layer][i, t] for t in range(span_start_j, span_end_j)
x_i = mean_j span_vec_i_j
y_i = gold_count_i
```

Use the same example-level ridge procedure, metrics, predictions, and vector saving as `mean_across_needles_span_last`, under:

```text
tensors/counting_features/probe_diagnostics/mean_across_needles_span_mean/
tables/counting_features/probe_diagnostics/mean_across_needles_span_mean/
figures/counting_features/probe_diagnostics/mean_across_needles_span_mean/
```

## Mode 4: `final_token`

Scientific question: has count information been routed to the answer-generating position?

For each example and layer:

```python
final_pos_i = example.sequence_length - 1
x_i = hidden[layer][i, final_pos_i]
y_i = gold_count_i
```

Train and save the same example-level ridge metrics, predictions, and vectors as the needle-pooled modes, under:

```text
tensors/counting_features/probe_diagnostics/final_token/
tables/counting_features/probe_diagnostics/final_token/
figures/counting_features/probe_diagnostics/final_token/
```

Interpretation:

- needle-pooled modes high, `final_token` low: local count/occurrence information exists but is not routed to the answer position.
- needle-pooled modes low, `final_token` high: the model consolidates count information away from occurrence spans.
- both high but generation wrong: possible output/readout or decoding bottleneck.
- both low: count is not linearly available in these tested representations, or the diagnostic setup is underpowered.

## Notebook integration

Integrate inside `notebooks/counting_feature_analysis.ipynb` after rows are filtered, examples are tokenized, train/test split is created, and hidden states are available. The safest implementation is:

1. Keep the existing ridge branch unchanged when `COUNTING_PROBE_MODE="direct"`.
2. For any non-`direct` mode, skip the direct token-level ridge branch for that run and execute only the selected probe workflow.
3. If `COUNTING_PROBE_MODE="all_diagnostics"`, run `mean_across_examples`, `mean_across_needles_span_last`, `mean_across_needles_span_mean`, and `final_token`.
4. If a single diagnostic mode is selected, run only that mode.
5. Print a compact per-layer table and save complete metadata.

Do not change the `drive.mount()` cell.

The diagnostic block should reuse existing hidden-state tensors when possible. If direct ridge is skipped but diagnostics are requested, extract the same `hidden_layer_{layer}.pt` tensors into `tensors/counting_features/` so cache behavior remains compatible.

## Visualization

Reuse the existing direct-mode ridge visualization style where possible instead of inventing a separate plotting stack. In `direct` mode, the current pipeline already saves line-fit and 2D projection plots such as:

```text
figures/counting_features/ridge_layer_{layer}_train_line.png
figures/counting_features/ridge_layer_{layer}_test_line.png
figures/counting_features/ridge_layer_{layer}_train_2d.png
figures/counting_features/ridge_layer_{layer}_test_2d.png
```

For each non-direct diagnostic probe mode, add analogous compact ridge visualizations under:

```text
figures/counting_features/probe_diagnostics/{mode}/
```

Minimum plots:

- prediction-vs-target scatter for train and test examples/occurrences;
- residual-vs-target scatter for test examples/occurrences;
- optional 2D projection plot using the same PCA/probe projection helper style as direct mode when the local data shape makes it straightforward.

Plot titles should include mode, layer, split, test R2, rounded accuracy when available, and the positional baseline R2 when the baseline is enabled. For example:

```text
final_token layer 24 test | R2=0.972 | rounded acc=0.96 | baseline R2: 0.31
```

Do not make separate baseline plots in the first implementation. Use the baseline only as an annotation and as saved metadata/table columns.

## Positional baseline

Add a baseline to test whether the probes are decoding true needle-count information or mostly position information correlated with count. Since count labels often increase as token position increases, a ridge probe could appear successful if hidden states encode position rather than count.

For every included example, sample fake positions outside all matching and non-matching needle spans:

```python
num_fake_positions_i = NUM_MAX_NEEDLES if NUM_MAX_NEEDLES is not None else NUM_NEEDLES
```

Sampling rules:

- candidate positions must be within the unpadded prompt length;
- candidate positions must be outside every needle span in `example.needle_segments`, not only matching spans;
- prefer candidates at least `COUNTING_PROBE_BASELINE_MIN_DISTANCE` tokens away from every needle span boundary and interior token;
- sample without replacement when enough candidates exist;
- if there are fewer candidates than requested, issue a clear warning, record the shortage in metadata, and sample all available candidates rather than silently sampling near needles;
- use a deterministic seed derived from `SPLIT_SEED`, layer, example index, and mode.

Then run the same representation recipe as the active `COUNTING_PROBE_MODE`, replacing actual needle-derived vectors with fake-position vectors:

- `mean_across_examples`: use fake positions as fake occurrences and predict fake occurrence index `1..num_fake_positions_i`;
- `mean_across_needles_span_last`: mean hidden states over the sampled fake positions and predict `gold_count_i`;
- `mean_across_needles_span_mean`: because fake positions are points rather than spans, use the same fake-position mean as the baseline for this mode and record `baseline_representation="fake_position_mean"`;
- `final_token`: keep the actual final-token probe unchanged, but compare against a single-position baseline. For each example, choose one sampled fake position deterministically from that example's fake-position set and use its hidden state to predict `gold_count_i`.

The baseline must use the same train/test split, ridge alpha, standardization setting, and metrics as the actual probe. Save baseline artifacts under the same mode directory:

```text
tables/counting_features/probe_diagnostics/{mode}/baseline_ridge_metrics.csv
tables/counting_features/probe_diagnostics/{mode}/baseline_predictions_layer_{layer}.csv
tensors/counting_features/probe_diagnostics/{mode}/baseline_ridge_probe_layer_{layer}.pt
```

Metadata should include the fake positions per example, number of candidates, number requested, number sampled, minimum observed distance to any needle span, and whether the distance constraint had to be relaxed. The actual probe summary should include baseline test metrics as columns, especially:

```text
baseline_test_r2
baseline_test_mae
baseline_test_rounded_count_accuracy
```

Interpretation:

- high actual R2 and low baseline R2: evidence for needle/count-specific information;
- high actual R2 and high baseline R2: position or global progress may explain much of the result;
- low actual R2 and low baseline R2: tested representation does not linearly expose the target;
- low actual R2 and high baseline R2: suspicious setup; inspect targets, split, and sampled positions.

## Variable-count generation

Most current run directories use a fixed count within a dataset, for example all examples have 3 needles or all examples have 8 needles. This affects interpretation:

- token-level `direct` ridge can still use within-prompt count variation through `target_count_y_t`;
- `mean_across_examples` remains meaningful within a fixed-count dataset;
- `mean_across_needles_*` and `final_token` cannot learn a cross-example total-count regressor from a single fixed-count dataset because `y_i` is constant.

`NUM_MAX_NEEDLES` is the first-class fix. It allows a single generated run to contain examples with different target counts while preserving the old behavior when the value is `None`.

Implementation notes:

- Sampling should happen before per-example needle construction so all downstream metadata, prompt text, scoring, and `gold_answer` agree.
- The sampled count should determine the number of matching needles inserted for `match_count`, `literal_count`, and the count component of `count_avg`.
- Position handling should stay consistent with the current generator. If a positions list is shorter or longer than the sampled count, define a deterministic policy and record it. Prefer reusing the first sampled-count positions when enough positions are provided; if there are fewer positions than sampled needles, fail loudly rather than silently recycling unless an existing generator helper already defines safe behavior.
- Run metadata should record both `NUM_NEEDLES` and `NUM_MAX_NEEDLES`, plus the empirical count distribution.
- The notebook should still warn and skip example-level ridge fitting if the included examples have fewer than two unique `gold_count` values, because users may intentionally run fixed-count data.

## I/O and metadata

All new outputs should live under:

```text
tables/counting_features/probe_diagnostics/{mode}/
tensors/counting_features/probe_diagnostics/{mode}/
figures/counting_features/probe_diagnostics/{mode}/
```

Do not overwrite existing direct ridge outputs. Save:

```text
tables/counting_features/probe_diagnostics/{mode}/diagnostic_metadata.json
tables/counting_features/probe_diagnostics/{mode}/summary.csv
```

Metadata should include:

- `COUNTING_PROBE_MODE`;
- `COUNTING_FEATURE_CALC_METHOD`;
- `FILTER_EXAMPLE`;
- `NUM_NEEDLES`;
- `NUM_MAX_NEEDLES`;
- `RUN_COUNTING_PROBE_BASELINE`;
- `COUNTING_PROBE_BASELINE_MIN_DISTANCE`;
- number of generated/scored/successful/failed/used examples;
- task type and counting needle kind;
- count distribution;
- baseline fake-position sampling summary when enabled;
- layers;
- split seed and test fraction;
- hidden-state tensor paths;
- skipped examples and reasons.

## Validation

Add focused tests for:

- config validation accepts the new `COUNTING_PROBE_MODE` values and rejects unknown modes;
- `direct` mode preserves existing ridge behavior and expected output filenames;
- `NUM_MAX_NEEDLES=None` preserves fixed-count generation and existing run naming;
- `NUM_MAX_NEEDLES=k` samples per-example counts in `1..k`, overrides fixed `NUM_NEEDLES` only for generation, and saves correct `gold_answer["count"]`;
- `FILTER_EXAMPLE=True` and `False` are both recorded and respected by diagnostics;
- matching span collection works for city-score `match_count`, marker `match_count`, `literal_count`, and `count_avg`;
- final-token position uses `example.sequence_length - 1`;
- `mean_across_examples` prototypes and occurrence-index ridge outputs save expected shapes and metadata;
- non-needle baseline samples positions outside all needle spans and prefers positions at least 5 tokens away;
- baseline metrics are saved and copied into the actual probe summaries as `baseline_*` columns;
- diagnostic visualizations are saved under `figures/counting_features/probe_diagnostics/{mode}/` and include `baseline R2: {value}` in titles when enabled;
- example-level probes skip or warn on a fixed-count single dataset;
- variable-count or synthetic multi-count examples can recover a known linear count signal;
- notebook source includes the new settings and inline comments without touching the Drive mount cell.

After coding, run the smallest relevant checks, likely:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_counting_feature_analysis.py tests/test_counting_feature_notebook.py tests/test_dynamic_niah_v2_controls.py
```

Update `README.md` to explain `NUM_MAX_NEEDLES` and `COUNTING_PROBE_MODE`, especially that `direct` is the existing pipeline and the other modes are diagnostics rather than validated steering vectors.

# Revision

## Rename `mean_across_examples` to `occurrence_index_probe`

The current name `mean_across_examples` is misleading because the ridge regression is not fit only on averaged vectors. Rename this probe mode to:

```python
COUNTING_PROBE_MODE = "occurrence_index_probe"
```

Keep a backward-compatible alias from `"mean_across_examples"` to `"occurrence_index_probe"` so old notebooks, configs, and run-reading helpers fail less abruptly. All newly written outputs should use the canonical folder name:

```text
tables/counting_features/probe_diagnostics/occurrence_index_probe/
tensors/counting_features/probe_diagnostics/occurrence_index_probe/
figures/counting_features/probe_diagnostics/occurrence_index_probe/
```

`COUNTING_PROBE_MODE="all_diagnostics"` should now run:

```text
occurrence_index_probe
mean_across_needles_span_last
mean_across_needles_span_mean
final_token
```

Update config validation, notebook validation, tests, README text, and plan references accordingly.

## Preserve the current computation flow

Do not change the substantive computation for this mode yet. The intended behavior remains:

1. Build one training row per actual matching needle occurrence.
2. Use the hidden state at that occurrence position as the feature vector.
3. Use the occurrence index as the target: first matching needle -> `1`, second -> `2`, etc.
4. Fit ridge regression on individual occurrence rows, with train/test split inherited from the parent example.
5. Separately compute occurrence-index prototypes by averaging occurrence vectors across examples.

The prototype computation should remain saved because it is useful diagnostic geometry, but it should be described clearly as separate from ridge fitting.

Definition of prototype:

```text
prototype_k = the mean hidden vector over all occurrence rows whose occurrence index is k
```

For example, `prototype_3` is the average hidden vector for third matching needles across all examples that have at least three matching needles. Prototype files summarize whether occurrence-index representations form a structured progression in hidden-state space. They are not the training set for the ridge regression.

Keep saving:

```text
layer_{layer}_prototypes.pt
prototype_metadata_layer_{layer}.csv
prototype_geometry_layer_{layer}.csv
predictions_layer_{layer}.csv
ridge_metrics.csv
summary.csv
```

## Add notebook explanation for probe modes

Add a short explanatory text block near the `COUNTING_PROBE_MODE` config cell in `notebooks/counting_feature_analysis.ipynb`. The explanation should be concise and oriented toward interpretation, not implementation trivia.

Suggested text:

```text
Probe mode guide:

- final_token: per-example probe. Ask whether the final prompt-token hidden state linearly predicts the total number of needles in the example.
- mean_across_needles_span_last: per-example probe. Ask whether the average of the last token from each matching needle span linearly predicts the total count.
- mean_across_needles_span_mean: per-example probe. Ask whether the average hidden representation over matching needle spans linearly predicts the total count.
- occurrence_index_probe: occurrence-level probe. Ask whether an individual needle occurrence hidden state linearly predicts which occurrence number it is, e.g. first, second, third. This is not a total-count probe.

Prototype note:
- For occurrence_index_probe, the notebook also saves prototypes. A prototype is the mean hidden vector for all occurrences with the same occurrence index across examples. These prototypes are diagnostics for representation geometry; ridge regression is still fit on individual occurrence rows.
```

This text should make the central distinction explicit:

- the three per-example modes ask: "Can this representation tell the total count in the example?"
- `occurrence_index_probe` asks: "Can this representation tell which occurrence number this needle is?"

## Update baseline wording

Update the fake-position baseline description for the renamed mode:

- `occurrence_index_probe`: use sampled fake positions as fake occurrences and predict fake occurrence index `1..num_fake_positions_i`;
- preserve the existing baseline logic for the three per-example total-count modes.

The baseline interpretation should also reflect the renamed target:

- a high `occurrence_index_probe` actual R2 with high baseline R2 may indicate that occurrence index is partly recoverable from position/progress through the prompt;
- high per-example total-count probe R2 with low baseline R2 is stronger evidence for count-specific information.

## Add cross-mode ridge-vector comparison

When `COUNTING_PROBE_MODE="all_diagnostics"`, the run produces one ridge vector per diagnostic mode per layer. Add a cross-mode comparison step after all diagnostic probes finish. For each layer, load the four learned ridge vectors:

```text
occurrence_index_probe
mean_across_needles_span_last
mean_across_needles_span_mean
final_token
```

Compute pairwise cosine similarity among the ridge vectors and save both a table and heatmap:

```text
tables/counting_features/probe_diagnostics/ridge_vector_similarity/layer_{layer}_cosine_similarity.csv
figures/counting_features/probe_diagnostics/ridge_vector_similarity/layer_{layer}_cosine_similarity_heatmap.png
```

The heatmap should be 4-by-4 for `all_diagnostics`, with rows and columns ordered consistently as above. Use the actual learned direction vector from the saved ridge probe. If ridge standardization is enabled, document whether the comparison uses the raw saved coefficient vector or an input-space-adjusted vector. Prefer an input-space-adjusted vector when available so cosine similarity is interpretable in the original hidden-state coordinates.

Interpretation notes:

- high cosine similarity between two modes suggests their learned linear directions point through similar hidden-state dimensions;
- low or negative cosine similarity with similar R2 means different directions can predict related targets from different feature constructions;
- `occurrence_index_probe` should be interpreted separately because its target is occurrence index, not total example count;
- cosine similarity compares the ridge vectors, while prediction plots compare each ridge vector's scalar predictions against that mode's target. Good prediction plots do not by themselves imply similar vectors.

Also consider two optional comparison diagnostics:

- **cross-application prediction:** apply one mode's ridge vector to another mode's feature matrix when dimensions and semantics allow it, especially among the three per-example total-count modes;
- **projection correlation:** for the same held-out examples, correlate scalar projections from two per-example modes to see whether the probes order examples similarly even when vectors differ.

## Validation updates

Add or update tests for:

- `COUNTING_PROBE_MODE="occurrence_index_probe"` is accepted;
- `COUNTING_PROBE_MODE="mean_across_examples"` normalizes to `"occurrence_index_probe"` as a compatibility alias;
- `all_diagnostics` writes canonical `occurrence_index_probe` folders;
- prototype artifacts are still saved for the renamed mode;
- `all_diagnostics` writes ridge-vector cosine similarity CSV tables and heatmap figures for each analyzed layer;
- notebook source contains the short probe-mode guide and prototype definition.

## Revision: evaluation of counting probe

When `FILTER_EXAMPLE=True`, the current diagnostic probes train and test only on successful examples. This is useful for asking whether solved examples have clean count representations, but it does not test whether a success-derived ridge direction generalizes to the harder examples that the model answered incorrectly.

Add two additional evaluation sets after each diagnostic ridge probe is fit on the filtered training split. Do not add a new notebook global variable for this first implementation; the extra evaluations should run automatically when all of the following are true:

- `COUNTING_FEATURE_CALC_METHOD == "ridge"`;
- `COUNTING_PROBE_MODE` is a non-`direct` diagnostic mode or `"all_diagnostics"`;
- `FILTER_EXAMPLE=True`;
- scored predictions are available, so successful and failed examples can be separated.

The existing filtered-train / filtered-test evaluation should remain unchanged. The new evaluations are additional diagnostics:

```text
filtered_train -> unfiltered_all
filtered_train -> failed_only
```

### Scientific questions

The three evaluation views should be interpreted as:

- `filtered_train -> filtered_test`: among examples the model solved, is count information linearly decodable?
- `filtered_train -> unfiltered_all`: does the ridge direction learned from solved examples generalize to the full population?
- `filtered_train -> failed_only`: do failed examples contain the same count direction, or are their representations noisier/different?

If filtered-test performance is high but failed-only performance is low, that supports the hypothesis that failures have noisier or shifted count representations. If failed-only performance is still strong, the model may encode the count but fail during readout/generation.

### Data construction

Keep the existing filtered training split as the only training source. Reuse the trained ridge probe, standardization statistics, target definition, and feature construction for each mode/layer. Do not refit on unfiltered or failed-only examples.

For evaluation, build feature rows from:

- `unfiltered_all`: every otherwise valid generated/scored example, regardless of `exact_match`;
- `failed_only`: otherwise valid examples whose scored prediction has `exact_match=False`.

Use the same mode-specific feature recipe:

- `final_token`: one final-token feature row per example, target is total gold count;
- `mean_across_needles_span_last`: one pooled span-last feature row per example, target is total gold count;
- `mean_across_needles_span_mean`: one pooled span-mean feature row per example, target is total gold count;
- `occurrence_index_probe`: one feature row per matching occurrence, target is occurrence index.

For `occurrence_index_probe`, `failed_only` means all occurrence rows belonging to failed examples. The evaluation unit in the prediction table is still an occurrence row, but metadata should also report the number of parent examples.

Run hard validity checks exactly as in the main diagnostic probe. If an unfiltered or failed example has invalid spans, missing count target, or shape mismatch, exclude it with an explicit `exclude_reason` in an audit table. Do not silently drop rows.

### Metrics

For each mode, layer, and extra evaluation set, save:

- number of parent examples;
- number of evaluation rows;
- target distribution;
- R2;
- MAE;
- RMSE;
- rounded count/occurrence accuracy;
- Pearson correlation when defined;
- whether the evaluation set is `unfiltered_all` or `failed_only`;
- the training source, always `filtered_train`.

Use the same fitted ridge probe and the same scaler/standardization parameters learned on filtered training examples. If an evaluation target has fewer than two unique values, report R2 as undefined/NaN with a clear status field rather than failing the whole notebook.

### Outputs

Save the extra evaluations under each diagnostic mode directory:

```text
tables/counting_features/probe_diagnostics/{mode}/extra_eval_metrics.csv
tables/counting_features/probe_diagnostics/{mode}/extra_eval_predictions_{eval_set}_layer_{layer}.csv
tables/counting_features/probe_diagnostics/{mode}/extra_eval_examples_{eval_set}.csv
figures/counting_features/probe_diagnostics/{mode}/extra_eval_unified_layer_{layer}_prediction_scatter.png
figures/counting_features/probe_diagnostics/{mode}/extra_eval_unified_layer_{layer}_residual_scatter.png
```

It is fine to save more metric/statistic tables than listed here if they help interpret the extra evaluations. Avoid generating many separate plot files for each evaluation setup unless a plot is genuinely not readable as a unified figure.

`extra_eval_predictions_*` should include at least:

```text
eval_set,layer,row_id,local_example_index,occurrence_index_if_any,
gold_target,predicted_value,rounded_prediction,residual,model_exact_match
```

For per-example modes, `occurrence_index_if_any` can be empty. For `occurrence_index_probe`, it should contain the target occurrence index.

`extra_eval_examples_*` should include one row per parent example with:

```text
eval_set,row_id,local_example_index,model_exact_match,gold_count,
num_matching_spans,span_starts,span_ends,include_status,exclude_reason
```

The existing `summary.csv` or `ridge_metrics.csv` may include compact extra-eval columns if convenient, but the required complete output is `extra_eval_metrics.csv`.

### Plots

Use unified plots rather than one separate plot per testing setup. For each mode and layer, generate one prediction-vs-target scatter plot and one residual-vs-target scatter plot that include:

- the original filtered test rows, if available;
- the extra unfiltered-all evaluation rows;
- the failed-only rows, either as their own overlay or as the failed subset of unfiltered-all.

Use visual encoding to make success/failure status immediately visible:

- successful test examples: one color/marker, for example blue circles;
- failed test examples: a contrasting color/marker, for example red triangles;
- filtered training rows are optional in the unified plot; if shown, make them faint or use hollow markers so they do not obscure test rows.

For `occurrence_index_probe`, the points are occurrence rows, but marker color/status should come from the parent example's `model_exact_match`. For the three per-example total-count modes, each point is one parent example.

The unified plot should make the train/eval direction explicit in the title and legend, for example:

```text
final_token layer 24 | trained on successful examples | all eval R2=0.72 | failed-only R2=0.41
```

Plot annotations should include at least the filtered-test, unfiltered-all, and failed-only R2/MAE when defined. If this makes the title too long, put the metrics in a small text box inside the plot. The legend should explain marker/color mapping for successful and failed examples.

Save unified plots under:

```text
figures/counting_features/probe_diagnostics/{mode}/extra_eval_unified_layer_{layer}_prediction_scatter.png
figures/counting_features/probe_diagnostics/{mode}/extra_eval_unified_layer_{layer}_residual_scatter.png
```

For `failed_only`, also save a compact count-bin summary table when the target is total count:

```text
tables/counting_features/probe_diagnostics/{mode}/failed_only_count_bin_metrics_layer_{layer}.csv
```

Suggested bins:

```text
1-3, 1-6, 1-10, 11-20, 21-30
```

Only write bins with at least one example. Columns should include number of rows, mean gold target, mean prediction, mean residual, MAE, and rounded accuracy. For max-12 runs, bins above 12 will naturally be absent or empty.

### Baseline interaction

Do not train new fake-position baselines for these extra evaluations in the first implementation. The existing baseline metrics remain tied to the original filtered train/test split.

If the saved actual ridge probe generalizes poorly to failed-only examples, that should be interpreted before invoking the fake-position baseline. The primary comparison for this revision is:

```text
filtered_train -> filtered_test
vs.
filtered_train -> unfiltered_all
vs.
filtered_train -> failed_only
```

### Metadata

Update each mode's `diagnostic_metadata.json` to include:

- whether extra evaluations were attempted;
- skipped reason if not attempted;
- number of successful, failed, and all examples available;
- number of valid rows for `unfiltered_all` and `failed_only`;
- paths to the extra-evaluation metric, prediction, example, and unified figure files.

Also update the top-level split/filter summary if convenient with an `extra_eval_summary` field, but avoid changing existing keys in a way that would break old result-reading scripts.

### Validation updates

Add focused tests for:

- when `FILTER_EXAMPLE=True`, diagnostic probes still train on successful examples only;
- extra evaluation applies the already fitted ridge probe to unfiltered examples without refitting;
- `failed_only` contains only examples with `exact_match=False`;
- per-example modes and `occurrence_index_probe` save the required extra-evaluation metrics and predictions;
- unified extra-evaluation plots are written when valid extra-evaluation rows exist;
- failed examples are visually distinguishable from successful examples in the unified plot data or plotting metadata;
- failed-only count-bin metrics are written when failed examples exist;
- no extra evaluation is run, or a clear skipped status is saved, when `FILTER_EXAMPLE=False` or scored predictions are unavailable.

# New probe visualization

Add sequence-wide projection plots for diagnostic ridge probes. The current diagnostic visualizations project only the targeted hidden states used by each probe, such as final-token states, pooled needle-span states, or occurrence-level needle states. The new visualization should instead use the learned ridge vector for a mode/layer and project hidden states from every token position in selected examples. This will show how the ridge direction behaves across the whole prompt, not only at the probe's training/evaluation positions.

## Goal

For each diagnostic mode and layer, visualize:

- the scalar projection of every token hidden state onto the learned ridge direction;
- how this projection changes with token position through the sequence;
- which projected points belong to needle spans;
- whether needle-span projections sit on a distinct trend relative to non-needle positions;
- whether failed examples show different projection-vs-position structure from successful examples.

This should help distinguish count-specific signal from broader position/progress signal.

## Scope

Apply this only to non-`direct` diagnostic probe modes:

```text
occurrence_index_probe
mean_across_needles_span_last
mean_across_needles_span_mean
final_token
all_diagnostics
```

Use the already fitted ridge probe for each mode/layer. Do not refit anything for this visualization. If ridge standardization is enabled, use the same projection convention as the existing prediction code: apply the saved standardization parameters before taking the ridge projection, so plotted values match the probe's scalar prediction space.

## Example selection

To avoid unreadable plots, do not plot all 300 examples in one figure. Add a small deterministic selection policy:

- include a few successful examples and a few failed examples when scored predictions are available;
- prefer examples spanning low, medium, and high gold counts;
- for `FILTER_EXAMPLE=True`, include examples from the filtered successful set and, when extra-evaluation data are available, include failed examples from the unfiltered/failed-only evaluation set;
- use a deterministic seed, preferably `SPLIT_SEED`, so reruns are comparable;
- record selected row IDs, gold counts, exact-match status, and sequence lengths in metadata.

A reasonable first default is exactly this target when examples are available: up to 6 examples per mode/layer, organized as low/medium/high count buckets with two examples per bucket:

```text
low count:    prefer 1 successful + 1 failed example
medium count: prefer 1 successful + 1 failed example
high count:   prefer 1 successful + 1 failed example
```

Within each count bucket, prefer one successful test/evaluation example and one failed test/evaluation example when both exist. If one status is unavailable, use the available status rather than leaving the bucket empty, but record the substitution in metadata. If a bucket itself is unavailable, skip it and record the reason. For variable-count runs, define buckets from the realized gold-count range for that run. For example, max-12 runs can use low/medium/high ranges such as `1-3`, `4-8`, and `9-12`; max-30 runs can use `1-10`, `11-20`, and `21-30`.

## Plot design

For each selected example, make a plot with:

- x-axis: token position, either absolute token index or normalized position `position / sequence_length`;
- y-axis: ridge projection value;
- non-needle positions: scatter or thin line colored by a single continuous gradient according to token position;
- color bar: indicates token position, preferably normalized from 0 to 1 so examples of different lengths remain comparable;
- needle-span positions: overlay with a clearly different color and marker, not part of the gradient, so they are visually separable;
- matching needle spans and non-matching needle spans should be distinguishable if both exist. For example:
  - matching needle tokens: red diamonds or red circles;
  - non-matching needle tokens: orange x markers;
  - non-needle tokens: blue/viridis gradient by position.

Use a single-color-family gradient for non-needle positions, such as light-to-dark blue. Avoid using a rainbow palette. The color bar should be labeled:

```text
normalized token position
```

Plot titles should include:

```text
mode, layer, row_id, gold count, model exact-match status, sequence length
```

Example title:

```text
final_token L24 | row dynamic_niah_v2_17 | gold=12 | exact=False | seq_len=2074
```

## Outputs

Save plots in a dedicated subfolder under each diagnostic mode's figure directory, so the main mode folder does not become crowded:

```text
figures/counting_features/probe_diagnostics/{mode}/sequence_projection/sequence_projection_layer_{layer}_example_{local_or_dataset_index}.png
```

Save a compact metadata table:

```text
tables/counting_features/probe_diagnostics/{mode}/sequence_projection/sequence_projection_examples_layer_{layer}.csv
```

Suggested columns:

```text
layer,mode,row_id,local_example_index,dataset_index,gold_count,
model_exact_match,sequence_length,num_matching_spans,num_all_needle_spans,
selected_bucket,selected_reason,figure_path
```

Optionally save the raw plotted values for later analysis:

```text
tables/counting_features/probe_diagnostics/{mode}/sequence_projection/sequence_projection_values_layer_{layer}.csv
```

Suggested columns:

```text
layer,mode,row_id,local_example_index,token_position,normalized_position,
projection,is_matching_needle_token,is_any_needle_token,needle_id_if_any
```

The raw values table may become large, so keep it optional or write it only for the selected plotted examples.

## Interpretation

The plots should make the following patterns easy to inspect:

- If non-needle positions form a smooth monotonic trend with position, then the ridge direction may partially encode position or prompt progress.
- If needle positions jump away from the non-needle trend, that is stronger evidence of needle/count-specific signal.
- If matching needle positions align with increasing projection values across occurrence order, that supports occurrence/count encoding.
- If failed examples show needle projections compressed below the expected range, that would match the observed high-count underprediction.
- If final-token ridge projections are distinct only near the end of the sequence, this may indicate answer-position consolidation rather than local needle-span encoding.

## Integration details

Implement this after diagnostic ridge probes are fit and after any extra-evaluation hidden-state tensors are available. The visualization should reuse:

- saved ridge probes under `tensors/counting_features/probe_diagnostics/{mode}/ridge_probe_layer_{layer}.pt`;
- hidden-state tensors from the filtered diagnostic set;
- extra-evaluation hidden-state tensors when `FILTER_EXAMPLE=True` and failed examples are being plotted;
- existing tokenized examples and needle-span metadata.

Do not require a new global config parameter for the first implementation. It is acceptable to always generate these plots for non-direct diagnostic modes, as long as the number of selected examples is small and deterministic.

If plotting all modes/layers becomes too slow or creates too many files, add a later config knob such as:

```python
RUN_SEQUENCE_PROJECTION_PLOTS = True
MAX_SEQUENCE_PROJECTION_EXAMPLES = 6
```

but do not add these unless needed.

## Validation updates

Add focused tests for:

- sequence projection uses the saved ridge probe without refitting;
- standardized probes apply the saved feature mean/scale before projection;
- non-needle token points carry normalized position values in `[0, 1]`;
- matching needle tokens are marked separately from non-needle tokens in the raw values table;
- sequence projection example metadata and figure files are written for at least one diagnostic mode/layer;
- notebook source contains the sequence-projection call without touching the Drive mount cell.
