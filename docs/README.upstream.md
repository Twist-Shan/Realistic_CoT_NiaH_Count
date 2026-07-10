# Representation Analysis of LLMs for Dynamic NIAH Tasks

This repository builds tokenizer-aware Dynamic NIAH datasets and analysis workflows for probing how LLM representations encode inserted evidence. The active notebooks generate controlled needle-in-a-haystack examples, collect model responses, save hidden states and Q/K-cache statistics, and run token- and representation-level ablations/restores. Outputs are reproducible run folders with JSONL data, resolved configs, metrics, tensors, figures, critical-token tables, and optional Colab-friendly zip exports.

## Quick start: running notebooks

The main workflow is notebook-first. Run notebooks from the repo root (or set `NIAH_REPO_DIR` to the cloned repo in Colab), install `requirements.txt`, and keep `PYTHONPATH=src` so notebook cells can import `dataset_generation`, `counting`, and `single_example`. The counting-feature notebook resolves the repo from the current working directory by default, so avoid hard-coded Google Drive paths when moving between Colab repo versions. The expected high-level dependency order is:

```text
configs/ + data/haystacks/ + src/ + scripts/
        ↓
Dynamic NIAH JSONL rows in generate_data/dynamic_niah_v2.jsonl
        ↓
model responses and metrics in tables/
        ↓
hidden-state tensors/figures and Q/K outlier tables
        ↓
token ablation, representation ablation, and representation restore outputs
```

`notebooks/qk_qwen3_capture_smoke_test_colab.ipynb` is intentionally omitted below because it is a narrow Q/K capture smoke test rather than a main analysis notebook.

### `notebooks/analysis_hidden_states_v4.ipynb`

Use this for a full multi-example Dynamic NIAH run. It prepares Paul Graham haystacks, optionally generates the dataset, optionally generates model answers and metrics, runs hidden-state analysis over configured layers, inspects saved tensors/figures, runs Q/K-cache outlier analysis on uncontrolled prompts, extracts needle-sensitive tokens, and exports one consolidated zip to Drive.

Expected artifacts include `generate_data/dynamic_niah_v2.jsonl`, `generate_data/config.used.json`, `run_metadata.json`, `analyze_hidden_states_config.json`, `logs.txt`, hidden-state `.pt` files under `tensors/`, plots under `figures/`, predictions/metrics under `tables/`, Q/K-cache metadata under `tensors/qk_cache/`, and outlier/needle-sensitivity tables such as `tables/massive_tokens_all.csv`, `tables/attention_sinks_topk.csv`, `tables/needle_attention_mass.csv`, and `tables/needle_sensitive_tokens.json`.

### `notebooks/single-example-v2.ipynb`

Use this for deep inspection of one selected generated row, or set `ALL_EXAMPLES=True` for a slower loop over a dataset. It resolves `DATASET_JSONL_PATH` directly or via `DATASET_RUN_NAME` under `data/niah-example`, writes scratch outputs under `RUN_ROOT`, and then zips the parent run directory to `RESULTS_PATH`. The notebook loads one row, saves a compatibility one-row `generate_data/dynamic_niah_v2.jsonl`, captures hidden states and Q/K outliers, and can run token-level ablation, representation-level ablation, and representation-restore experiments.

Expected artifacts include per-example hidden-state tensors and input figures, Q/K-cache files, `single_example_notebook_manifest.json`, one-row generated-data copies, `tables/ablation/*` for token ablation, `tensors/ablation_representation/hidden_state_distribution_stats.pt` plus representation-ablation tables, representation-restore outputs, and a final zip export.

### `notebooks/counting_analysis.ipynb`

Use this for Dynamic NIAH counting variants (`match_count` and `literal_count`) and optional thinking-mode/CoT analysis. It prepares haystacks, generates a counting dataset, optionally runs answer generation and metrics, runs hidden-state analysis, then can run the same token, representation-ablation, and representation-restore stack over selected example IDs. When `USE_THINKING=True` and `ANALYZE_REASONING_TOKENS=True`, it saves prompt-plus-reasoning extended inputs and aligns downstream analysis to the final-answer span.

Expected artifacts include counting JSONL rows in `generate_data/dynamic_niah_v2.jsonl`, predictions and aggregate metrics in `tables/`, hidden-state tensors and plots, optional CoT metadata in `tables/cot_info.json` plus `tables/input_generate_cot_{example_id}.txt`, per-example ablation folders under `ablation_examples/example_id_*`, and a final `{RUN_NAME}.zip` export.

### `notebooks/counting_feature_analysis.ipynb`

Use this for counting-feature extraction on Dynamic NIAH counting variants (`match_count` and `literal_count`). It keeps the dataset-generation and response-scoring setup from `counting_analysis.ipynb`, then filters to successful model responses, tokenizes the uncontrolled prompts, locates matching needle spans, builds `target_count_y_t.pt`, saves layer-wise hidden states as padded `hidden_layer_{layer}.pt` tensors, and trains/plots ridge-regression and optional linear-classification probes. Classification is disabled for `TARGET_COUNT_TYPE = "interpolation"` because those labels are fractional.

Set `FILTER_EXAMPLE = False` in the counting-feature workflow to include all otherwise valid generated examples in feature calculation instead of only successful model-response examples. The default is `True`, preserving the historical successful-example-only behavior. For non-direct counting probe diagnostics with `FILTER_EXAMPLE=True`, probes still train on successful examples, and the notebook now also evaluates those fitted probes on all examples and failed-only examples, saving extra metrics plus unified success/failure scatter plots under each `probe_diagnostics/{mode}/` folder. Each non-direct diagnostic mode also saves sequence-wide ridge projection plots and per-token CSVs under `probe_diagnostics/{mode}/sequence_projection/`, using symlog y-scaling, position-gradient coloring for non-needle tokens, and separate markers for needle-span tokens.

The notebook defaults to the current counterfactual steering workflow rather than smoke-test mode: `SMOKE_TEST=False`, `PROMPT_STYLE="vanilla"`, sentence-level randomized insertion, the single city-score template, `COUNTING_FEATURE_CALC_METHOD="counterfactual"`, `FEATURE_CALC_POS="needle-last"`, and `STEERING_POSITION_MODE="needle_span"`. Config values are resolved in this order: Python defaults in `src/counting/feature_analysis.py`, then `configs/counting_analysis.json`, then the notebook `CONFIG_OVERRIDES` dictionary. The JSON file uses a valid `{"config": ..., "notes": ...}` shape, where only `config` affects behavior and `notes` documents common settings. The notebook keeps frequently adjusted values such as `LAYERS`, `PROMPT_STYLE`, `FEATURE_CALC_POS`, `STEERING_POSITION_MODE`, `STEERING_COEFF`, `NUM_EXAMPLES`, and `STEERING_TEST_EVAL` visible in `CONFIG_OVERRIDES`; those notebook values win over the JSON file. The default run uses `NUM_EXAMPLES=100`, layers `[16, 20, 24, 28]`, and bounded token caps of `50_000` per layer. Set `NUM_MAX_NEEDLES = None` to keep the existing fixed-`NUM_NEEDLES` generation behavior. Set it to a positive integer to sample each example's target count uniformly from `1..NUM_MAX_NEEDLES`; this overrides `NUM_NEEDLES` per generated example, saves the sampled count in row controls and `gold_answer["count"]`, and appends `num_max_needles_{K}` to run/cache names. Set `COUNTING_FEATURE_CALC_METHOD = "ridge"` for the original ridge/classification probe workflow, `"contrastive-success"` to compute raw successful-minus-unsuccessful directions from a balanced scored pool, or `"counterfactual"` to compute raw original-successful-minus-counterfactual-successful directions after removing one non-final needle slot. `COUNTING_PROBE_MODE = "direct"` preserves the existing token-level ridge pipeline exactly. With `COUNTING_FEATURE_CALC_METHOD="ridge"`, non-direct probe modes replace the direct ridge workflow for that run and save diagnostic outputs under `tables|tensors|figures/counting_features/probe_diagnostics/{mode}/`: `occurrence_index_probe`, `mean_across_needles_span_last`, `mean_across_needles_span_mean`, `final_token`, or `all_diagnostics`. These diagnostic ridge vectors are saved for inspection and possible later steering, but they are not yet validated steering directions. Non-direct diagnostics can also fit a fake-position baseline controlled by `RUN_COUNTING_PROBE_BASELINE` and `COUNTING_PROBE_BASELINE_MIN_DISTANCE`; baseline metrics are written to `baseline_ridge_metrics.csv`, included as `baseline_*` summary columns, and the actual-probe visualizations include `baseline R2: ...` in their titles. With `COUNTING_PROBE_MODE="all_diagnostics"`, all four diagnostic modes each get their own tables, tensors, and prediction/residual plots; the run also writes cross-mode ridge-vector cosine similarity tables and heatmaps under `probe_diagnostics/ridge_vector_similarity/`. `FEATURE_CALC_POS = "last"` uses the final prompt token and `"needle-last"` uses the last token inside the last matching needle span; `COUNTERFACTUAL_REMOVED_NEEDLE_INDEX` only affects the counterfactual method. Ridge probe outputs are written under `tables/counting_features/`, `tensors/counting_features/`, and `figures/counting_features/`; contrastive success vectors are written under `tensors/counting_features/contrastive_success/{FEATURE_CALC_POS}/`; counterfactual count vectors are written under `tensors/counting_features/counterfactual/{FEATURE_CALC_POS}/`, with summaries under the matching `tables/counting_features/...` folders. The main deliverable is one archive at `results/counting_features/{RUN_NAME}/{RUN_NAME}.zip`; unzipping it yields subfolders such as `tensors/`, `tables/`, `figures/`, and `generate_data/` directly. Optional reusable counting-feature caches are disabled by default and, when enabled, are stored by setting under `results/counting_feature_cache/{SETTING_NAME}/counting_features/` so matching reruns can restore feature artifacts and skip refitting. The notebook can also run counting-feature steering from the vectors produced by `COUNTING_FEATURE_CALC_METHOD`. Set `STEERING_POSITION_MODE = "last_token"` to estimate the last-token projection scale and apply `beta * sigma_l * v_l` during greedy decoding. Set `STEERING_POSITION_MODE = "needle_span"` to estimate `sigma_l` over actual matching needle-span positions, steer one inserted needle span at a time, and save detailed/aggregate outputs under the corresponding steering folder for the selected feature method.

Set `STEERING_TEST_EVAL = True` to add a held-out-style steering evaluation after the normal steering setup is available. The notebook generates fresh randomized counting datasets for gold counts `1..NUM_MAX_NEEDLES_STEERING_EVAL`, with `NUM_EXAMPLES_STEERING_EVAL` examples per count, then runs the active steering configuration on those datasets. Use `STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION` to control token-randomized insertion spacing; under `SENTENCE_LEVEL_INSERTION=True`, actual placement uses sentence-ending sites instead. Outputs are saved under `generate_data/steering_test_eval/`, `tables/counting_features/steering_test_eval/`, and `tensors/counting_features/steering_test_eval/`. The compact `steering_test_eval_summary.csv` has one row per layer, intervention target, and steering coefficient; each dataset column stores `(no_steering_acc, steering_acc, no_steering_avg_counts, steering_avg_counts)`. The detailed `steering_test_eval_predictions.csv` keeps one row per example and steering setting for later rescue/damage analyses. The notebook also writes stage-level runtime logs to `tables/timing_summary.json` and `tables/timing_summary.csv`; each row records a stage name, UTC start/end timestamps, elapsed seconds, status, error metadata, and CUDA memory snapshots when a GPU is available. Set `USE_KV_CACHE_FOR_NONTHINKG=False` to pass `use_cache=False` during non-thinking response generation, reducing memory for short JSON answers while preserving the default cached generation behavior when the setting is `True`. After the optional response-generation block succeeds, the notebook writes an early checkpoint archive at `{RUN_DIR}/{RUN_NAME}_response_generation_checkpoint.zip` containing the generated dataset, resolved config, scored predictions, aggregate metrics, run metadata, and timing summaries. This preserves the slow generation/scoring outputs even if later hidden-state or probe stages fail.

Set `RUN_NEEDLE_SENSITIVITY=True` to run the optional same-length needle-removal sensitivity analysis. For each included example with at least `NUM_REMOVAL` matching needles, the notebook samples needles, replaces each selected span with same-length non-needle tokens, and measures normalized hidden-state changes at `TARGET_SENSITIVITY_POSITION="last-token"` for `LAYERS`. The implementation streams one example or perturbation batch at a time and saves compact target-position vectors plus scalar metrics, not full perturbed sequence hidden states. Outputs are written under `tables|tensors|figures/counting_features/needle_sensitivity/`, including per-removal metrics, per-example summaries, `mean_sensitivity_layer_{layer}.pt`, and dedicated timing/memory summaries for diagnosing long-context runs.

Set `RUN_COUNTING_FEATURE_CALC = False` to skip Block 8 feature-vector calculation when matching feature-vector files already exist. This does not disable steering by itself; if `RUN_STEERING=True` or `STEERING_TEST_EVAL=True`, those later blocks still require the selected ridge, contrastive-success, or counterfactual vector files under `tensors/counting_features/`.

To regenerate only the sequence-projection PNGs from saved CSV data, run `python scripts/regenerate_sequence_projection_symlog.py /path/to/run_dir`. This rewrites `figures/counting_features/probe_diagnostics/{mode}/sequence_projection/*.png` with symlog y-scaling without rerunning response generation, hidden-state extraction, or probe fitting; add `--no-overwrite` to write `*_symlog.png` companions instead.

To compare saved counting feature vectors across run folders, run `PYTHONPATH=src python scripts/analyze_counting_feature_vectors.py` from the repo root. It scans the parent `NIAH_repo_and_local_runs` directory, loads ridge, contrastive-success, and counterfactual feature vectors when present, and writes `steering_run_analysis/counting_feature_vector_metadata.json` plus `steering_run_analysis/counting_feature_vector_cosine_heatmap.png` under the parent folder.

### `notebooks/cot_analysis.ipynb`

Use this for the newer full-sequence CoT analysis. Unlike the older `counting_analysis.ipynb` CoT path, which saves `prompt + reasoning before final answer`, this workflow saves `prompt + the entire generated output` for both non-thinking and thinking modes. The notebook compares modes on identical Dynamic NIAH rows, caches response/metric files under both the run folder and `data/niah-example`, then analyzes a shared subset of up to `MAX_ANALYSIS_EXAMPLES` examples balanced by success/failure and needle count when possible.

The notebook is a thin launcher around `scripts/run_cot_analysis.py`. Core reusable logic lives in `src/counting/cot_analysis.py`. Important controls include `THINKING_MODES`, `MAX_NEW_TOKENS_NONTHINKING`, `MAX_NEW_TOKENS_THINKING`, `LAYERS`, `K`, `OUTLIER_RATIO_THRESHOLD`, `MAX_ANALYSIS_EXAMPLES`, and `CAPTURE_ATTN_IMPLEMENTATION`. Before selecting the expensive analysis subset, the notebook verifies that inserted needle token spans can be located in the final model prompt; examples with missing needle spans are excluded from `MAX_ANALYSIS_EXAMPLES`, with a warning if fewer eligible examples remain.

Naming convention: `RUN_NAME` identifies one concrete execution and is allowed to be timestamped. Its artifacts live under the current run directory, usually `/content/{RUN_NAME}` in Colab. `SETTING_NAME` is a stable, timestamp-free cache key derived from the data/model/task settings. Its reusable artifacts live under `data/niah-example/{SETTING_NAME}`. Do not save timestamped run names under `data/niah-example`; that folder is for reusable setting-level caches that later runs can pick up.

High-level workflow:
1. Resolve config and paths, then print `RUN_NAME`, current run directory, `SETTING_NAME`, and the reusable data cache directory.
2. Generate or restore the Dynamic NIAH dataset for the stable setting.
3. For each requested mode in `THINKING_MODES`, restore cached responses when metadata matches; otherwise load the model, generate responses, score metrics, and save the reusable response cache.
4. Copy the dataset and per-mode response/metric files into the current run folder for auditability.
5. Select common examples across modes, capture selected hidden states, run full-sequence CoT/QK analyses, write mode-specific tables and figures, clean temporary tensor-heavy files, then archive the run.

I/O control: reusable dataset files are `data/niah-example/{SETTING_NAME}/dynamic_niah_v2.jsonl` and `data/niah-example/{SETTING_NAME}/config.used.json`. Reusable response files are stored per mode under `data/niah-example/{SETTING_NAME}/responses/{mode}/`, including `predictions.jsonl`, `metrics.json`, and `response_cache_metadata.json`. Response caches are reused only when metadata such as mode, setting name, model, tokenizer, generation limits, and sampling settings match. Current-run copies are written under `{RUN_DIR}/responses/{mode}/`, while analysis artifacts are mode-local under `{RUN_DIR}/modes/{mode}/tables/`, `{RUN_DIR}/modes/{mode}/figures/`, and `{RUN_DIR}/modes/{mode}/tensors/`. Temporary Q/K cache files are allowed during analysis, but cleanup removes Q/K caches, attention-stat tensor folders, and oversized `.pt` files before the final archive is written unless an explicit preservation setting says otherwise. CoT attention plots are pattern-level: for each selected example, layer, and needle/outlier/prompt pattern, the plot splits prompt and generated query positions into 1-by-2 panels, shades prompt needle spans, highlights the target pattern, and draws dashed vertical lines for detected thinking/final-answer boundaries when available. `OUTLIER_RATIO_THRESHOLD` filters weak attention-sink and massive-activation tokens out of the plotting pattern list without changing the saved top-`K` outlier tables. The always-included `prompt_span` and `prompt_span_no_first` patterns help diagnose total attention to the prompt during generation.

The same workflow can be launched without the notebook:

```bash
PYTHONPATH=src python scripts/run_cot_analysis.py \
  --override '{"NUM_EXAMPLES": 10, "MAX_ANALYSIS_EXAMPLES": 2, "THINKING_MODES": ["nonthinking", "thinking"]}'
```

## Generate dynamic NIAH

The minimal script workflow below creates the haystack corpus and runs a small hidden-state experiment using the core Dynamic NIAH pipeline. It uses one control needle (`true false false`) and two examples to keep the run short; increase `--num-examples` for the full experiment.

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python scripts/gather_paul_graham_essays_v2.py --out-dir data/haystacks/paul_graham
PYTHONPATH=src python scripts/analyze_hidden_states.py \
  --config configs/niah_dynamic.json \
  --model Qwen/Qwen3-8B \
  --num-examples 2 \
  --num-needles 3 \
  --positions 100 200 400 \
  --prompt-style easier \
  --layers 4 8 12 \
  --control_switch true false false \
  --save_data true
```

Expected report: a new `results/run_.../` directory with `logs.txt`, `run_metadata.json`, `analyze_hidden_states_config.json`, generated rows in `generate_data/dynamic_niah_v2.jsonl`, per-layer tensors in `tensors/`, and comparison/PCA plots in `figures/`. To add model answer metrics, run `scripts/gen_responses.py` with the same config/model/run settings; it writes `tables/predictions.jsonl` and `tables/metrics.json`, preserving `gold_answer` and `expected_answer` as the original all-needle answer used for scoring, while also reporting `control_gold_answer` for analysis.

## Recent updates (PR summary)

- Added timestamped experiment run directories under `results/`.
- Added run-local artifact folders: `figures/`, `tensors/`, and `generate_data/`.
- Added `logs.txt` tee logging so script output is persisted while still printing to the terminal.
- Changed `DynamicNiahV2Config.output_dir` and `DynamicNiahV2Config.data_save_path` defaults to `null` / `None`.
- Preserved optional user-specified output paths as **additional copies** instead of replacing the canonical run directory.
- Added explicit **control-switch support** in v2 generation, allowing each needle slot to be toggled between a real needle and a same-length control segment sampled from haystack text.
- Added **control-aware outputs** in each generated row:
  - `control_gold_answer`
  - `control_relevant_records`
  - per-needle `is_control` and `control` metadata
  - a `controls` block with seed provenance and run controls
- Improved **configuration normalization** (`insertion_positions`, `needle_seeds`, `control_switch`) when loading JSON configs.
- Added `--target-haystack-tokens`, `--num-needles`, and `--positions` overrides to `scripts/gen_responses.py` and `scripts/analyze_hidden_states.py` so notebook configuration variables stay aligned without hand-editing JSON.
- Updated hidden-state analysis integration so prompt/message construction follows config-driven settings (`prompt_style`, `thinking_mode`, tokenizer).
- Added representation-level ablation support for the single-example workflow, including `configs/ablation-representation.json`, package exports under `single_example`, Colab settings, hidden-state distribution profiling, filtered and all-position critical-token patterns, manual layer-hook generation, and representation-ablation result tables.
- Added representation-level restore support for the single-example workflow, including `configs/ablation-representation-restore.json`, `RESTORE_DATASET_RUN_NAME`, per-setting corrupted needle replacements, per-needle `needle_tail_*` patterns, clean-hidden-state restoration hooks, restore result tables, and cleanup of restore `.pt` intermediates plus any `.pt` artifact larger than 200 MB before zipping.

## Haystack sync: full Paul Graham corpus

Use the v2 gather script to sync Paul Graham essays into the haystack directory and keep only files that are useful for long-context slicing.

```bash
python scripts/gather_paul_graham_essays_v2.py --out-dir data/haystacks/paul_graham
```

What this supports for `plans/plan-v2.md`:
- a large pool of raw `.txt` haystacks,
- filtering away short files (the generator expects substantial source texts),
- provenance-oriented haystack preparation before token-level window sampling.

## Dynamic NIAH v2 (`plans/plan-v2.md`)

### Core behavior

The v2 generator (`scripts/generate_dynamic_niah_v2.py`) builds examples by:
1. Loading a tokenizer (default: `Qwen/Qwen3-8B`).
2. Sampling one Paul Graham source file and extracting a token window with length `target_haystack_tokens`; if the selected source is too short, the normalized haystack text is repeated until a full target-length window can be sampled.
3. Generating multiple needle facts from city/entity templates with deterministic score assignment, or repeated exact marker needles such as `[dolphin]` for marker-count runs.
4. Optionally replacing selected needles with same-length control segments from haystack text.
5. Inserting resulting token sequences at user-specified token positions, or using a text-level insertion mode. With `sentence_level_insertion=true`, non-null needle slots are randomly inserted at sentence-ending delimiters (`.`, `!`, `?`). With `word_level_insertion=true`, non-null slots are randomly inserted at whitespace boundaries between non-whitespace text chunks, without breaking existing words. If there are fewer sentence-ending delimiters or word-boundary sites than active needles, the text-level insertion mode samples sites with replacement, so multiple needles may intentionally appear back to back at the same boundary.
6. Emitting JSONL rows with:
   - tokenized and decoded inserted content,
   - insertion metadata and verified context token spans where available,
   - relevant/control-relevant records,
   - query/messages,
   - standard and control gold answers,
   - `schema_version` matching `dataset.schema.json`.

### Supported NIAH tasks

Dynamic NIAH v2 currently supports four task types through `task_type` / `--task-type`:

- `argmax`: each inserted needle is a city fact with an integer score. The model must identify the city with the highest score and return `{"city":"<city>","score":<int>}`. Gold answers break ties deterministically by score, then city name, although generated v2 scores are sampled uniquely for each example.
- `count_avg` (also accepted as `count_average` or `count_average_score`): the model must aggregate all rated cities in the example and return `{"count":<int>,"average_score":<float>}`. Evaluation requires the exact count and accepts the average score within a small numeric tolerance.
- `match_count`: by default, each inserted needle is a city-score fact, but the model only has to report how many cities received a score. The counting-feature notebook defaults to `data/templates/niah_fact_single_template.txt`, whose single city-score wording is aligned with the query "How many cities received a score?" The response schema is `{"count":<int>}` and the gold answer is computed from inserted `relevant_records`.
- `match_count` with `counting_needle_kind="marker"`: each active needle is the exact same marker text, defaulting to `[dolphin]`. For `prompt_style="vanilla"`, the instruction says that the exact marker is inserted in the following text, and the query asks how many exact copies of `[dolphin]` appear. For `prompt_style="vanilla_no_cue"`, the pre-context marker cue is removed from the instruction while the final query remains explicit. This variant is intended to isolate counting from city names, scores, and audit wording.
- `literal_count`: each example generates one alphanumeric UID/canary string and inserts exact copies for each non-null insertion position. The desired tokenizer length of this UID is controlled by `uid_token_length` / `UID_TOKEN_LENGTH`, defaulting to 4. Word-level insertion places UIDs at whitespace boundaries between words and verifies that the realized context span still has the requested token length. The query asks how many exact copies of the quoted UID appear, the response schema is `{"count":<int>}`, and generation validates the literal text occurrence count in the uncontrolled context.

When `control_switch` replaces one or more inserted needles with same-length haystack text, `gold_answer` still describes the inserted original needles, while `control_gold_answer` describes only inserted non-control needles. Downstream response evaluation scores model outputs against the original all-needle `gold_answer` for all tasks; prediction reports also include `control_gold_answer` for analysis of the non-control subset.

Prompt construction is split into three parts: an instruction block, a task-specific query, and the context/haystack with inserted needles. The instruction block always contains the JSON-only response constraint; for city-score tasks it also says that information about cities is inserted within the following text, while `literal_count` and marker-count runs get analogous task-specific instructions. For `prompt_style="easier"`, the order is instruction, query, then context. For `prompt_style="vanilla"` and `prompt_style="vanilla_no_cue"`, the order is instruction, context, then query. The memorization sentence belongs to the instruction block, not the query or context; `vanilla_no_cue` intentionally omits that memorization sentence and the reasoning ban.


### Counting Colab ablation workflow

`notebooks/counting_analysis.ipynb` now runs the full counting-task workflow from one generated dataset path. The notebook creates a compact run name such as `run_YYYYMMDD_HHMMSS_Qwen3-8B_match_count_easier_1000_needles_100_200_400`, writes the generated JSONL to `/content/{RUN_NAME}/generate_data/dynamic_niah_v2.jsonl`, and points response generation, hidden-state analysis, and optional ablation analysis back to that same generated dataset path.

Ablation analysis is controlled by `SELECT_EXAMPLE_ID`, `RUN_ABLATION`, `RUN_REPRESENTATION_ABLATION`, and `RUN_REPRESENTATION_RESTORE`. `SELECT_EXAMPLE_ID` must be a subset of `list(range(NUM_EXAMPLES // 2))`; for example, `SELECT_EXAMPLE_ID = [0, 1, 2]` writes separate outputs under `/content/{RUN_NAME}/ablation_examples/example_id_0`, `/content/{RUN_NAME}/ablation_examples/example_id_1`, and `/content/{RUN_NAME}/ablation_examples/example_id_2`. The three ablation levels reuse `configs/ablation.json`, `configs/ablation-representation.json`, and `configs/ablation-representation-restore.json`, while preserving the one-controlled-needle hidden-state workflow.

All three ablation levels use the same critical-token pattern family by default: six capped score-ranked patterns (`massive_activation`, `attention_sink`, and `needle_sensitive`, each in a filtered version that excludes needle and boundary positions and an `_all` version that ranks all in-range positions), plus one uncapped `needle_span_{i}` pattern and one K-token `needle_tail_{i}` pattern per needle. For `num_needles = N`, each example therefore has `6 + 2 * N` critical-token patterns before layer or k sweeps are expanded.


### CoT / thinking-mode counting analysis

`notebooks/counting_analysis.ipynb` supports an opt-in CoT analysis mode for Qwen-style thinking runs. Set `USE_THINKING = True` and `ANALYZE_REASONING_TOKENS = True` to make Block 5 generate reasoning with greedy decoding, save the extended input (`prompt + reasoning before the final JSON answer`) to `tensors/inputs_cot_{example_id}.pt`, and write per-example token-boundary metadata to `tables/cot_info.json`. CoT runs append `_cot` to the compact run name.

The final-answer boundary detector first searches after Qwen's `</think>` marker, then falls back to schema-aware JSON parsing for the counting response (`{"count": ...}`). If generation reaches `max_new_tokens` without EOS, the notebook/script prints a visible warning and records the condition in `cot_info.json`; decoded prompt/reasoning/final-answer text is kept in separate `tables/input_generate_cot_{example_id}.txt` files rather than embedded in the JSON metadata.

When CoT analysis is enabled, hidden-state, Q/K outlier, token-level ablation, representation ablation, and representation-restore paths use the extended input instead of the prompt-only input. The controlled extended input reuses the same generated reasoning suffix after the controlled prompt to keep the continuation aligned. Hidden-state input figures draw cyan dashed vertical lines at the end of the original prompt and at the end of the extended input, marking the reasoning span. Ablation scoring starts from the end of the extended input and uses `max_new_tokens_for_cot` (default `64`) for shorter final-answer generations.

### Run the generator

Default behavior creates a timestamped run directory under `results/` and writes generated data under that run.

```bash
PYTHONPATH=src python scripts/generate_dynamic_niah_v2.py \
  --config configs/niah_dynamic.json \
  --task-type argmax \
  --tokenizer Qwen/Qwen3-8B \
  --num-examples 100 \
  --target-haystack-tokens 1000 \
  --num-needles 3 \
  --positions 100 200 400 \
  --prompt-style easier
```

Example canonical output layout:

```text
results/
└── run_YYYYMMDD_HHMMSS_Qwen_Qwen3-8B_task-argmax_prompt-easier_len-1000_needles-3/
    ├── figures/
    ├── tensors/
    ├── generate_data/
    │   ├── dynamic_niah_v2.jsonl
    │   ├── config.used.json
    │   └── dataset.schema.json
    ├── logs.txt
    └── run_metadata.json
```

### Optional additional output copies

`output_dir` and `data_save_path` default to `null` in `configs/niah_dynamic.json` and to `None` in `DynamicNiahV2Config`. When either path is specified, the script still writes the canonical run under `results/`, then saves an **additional copy** to the specified path and prints a message explaining that copy action.

Examples:

```bash
# Canonical run under results/ plus an additional folder copy.
PYTHONPATH=src python scripts/generate_dynamic_niah_v2.py \
  --config configs/niah_dynamic.json \
  --output-dir generated/dynamic_niah_copy
```

```bash
# Canonical run under results/ plus an additional JSONL copy.
PYTHONPATH=src python scripts/generate_dynamic_niah_v2.py \
  --config configs/niah_dynamic.json \
  --data-save-path generated/dynamic_niah_copy/custom.jsonl
```

Use `--run-dir` or `run_dir` only when you want to choose the canonical run directory itself. Use `--results-root` or `results_root` to place automatically named runs somewhere other than `results/`.

## Configuration logic (how config works)

The active repo uses **Dynamic NIAH v2 runtime config** (`DynamicNiahV2Config` in `src/dataset_generation/dynamic_niah_v2.py`):

- Used directly by `scripts/generate_dynamic_niah_v2.py` and `scripts/analyze_hidden_states.py`.
- Loaded from JSON with type normalization and then overridden by CLI flags.

### Resolution order (effective config)

For `scripts/generate_dynamic_niah_v2.py`, effective values are resolved in this order:

1. Dataclass defaults in `DynamicNiahV2Config`.
2. JSON config file passed by `--config` (if any).
3. CLI overrides (if provided).

So CLI always wins over file values, and file values win over dataclass defaults.

### Output and run-directory config fields

Important output fields in `DynamicNiahV2Config`:

- `results_root`: root directory for automatically named run folders. Default: `"results"`.
- `run_dir`: optional canonical run directory. Default: `None` / JSON `null`.
- `run_name`: optional canonical run folder name under `results_root`. Default: `None` / JSON `null`.
- `output_dir`: optional extra output directory copy. Default: `None` / JSON `null`.
- `data_save_path`: optional extra JSONL dataset copy. Default: `None` / JSON `null`.

When `run_dir` is not set, scripts create run folders with this pattern:

```text
run_{date}_{time}_{model}_{params}
```

The params currently include `task`, `prompt`, `len`, and `needles` so that the folder name captures important run settings.

### JSON normalization rules

When loading a JSON config file via `load_config_file(...)`:

- `insertion_positions`: list → tuple of ints or `None` values (`null` in JSON)
- `needle_seeds`: string-key dict → int-key dict (`{"0": 11}` becomes `{0: 11}`)
- `control_switch`: list elements coerced to booleans, including common string values such as `"true"` and `"false"`
- `fact_templates_path`: path to the city-score fact templates file. The repository keeps the original mixed-template file at `data/templates/niah_fact_templates.txt` and the uniform counting template at `data/templates/niah_fact_single_template.txt`.

This keeps runtime types aligned with `DynamicNiahV2Config` expectations.

### Validation and invariants

The v2 pipeline enforces these constraints at runtime:

- `len(insertion_positions) == num_needles`; `null` positions still generate aligned needles but skip insertion into the haystack.
- non-null insertion positions must be within `[0, target_haystack_length]`.
- when `sentence_level_insertion=true`, numeric insertion positions are ignored for placement and only their null/non-null status is used; non-null slots are randomly placed at sentence ends and recorded in `controls.insertion_positions` as realized context-token starts.
- when `word_level_insertion=true`, numeric insertion positions are ignored for placement and only their null/non-null status is used; non-null slots are randomly placed at whitespace boundaries between non-whitespace text chunks and recorded in `controls.insertion_positions` as realized context-token starts.
- `sentence_level_insertion` and `word_level_insertion` are mutually exclusive.
- marker-count rows verify each inserted marker copy by character span and token span; repeated identical markers are kept as independent realized insertions.
- city-score tasks require `num_needles <= number_of_entities` and `num_needles <= 51` (unique score pool).
- if `control_switch` is set, `len(control_switch) == num_needles`; when overriding one from a notebook or CLI, also override the other and provide exactly `num_needles` insertion-position entries, using `null` for skipped needles when needed.
- control replacement segments must match original needle token length.
- hidden-state analysis keeps the existing invariant that exactly one `control_switch` value is `true`; for those runs, that controlled needle should have a non-null insertion position.

If sampling fails repeatedly, generation aborts with a guardrail error. Short haystack sources are no longer skipped solely because they are below `target_haystack_tokens`; they are repeated first and the repeat metadata is saved under the row's `haystack` field.

### Seed logic and reproducibility

There are four seed scopes:

- `global_random_seed`: base seed for everything when a narrower seed is not provided.
- `haystack_seed`: optional override for haystack file/window sampling.
- `needle_seed`: optional scalar override for all task needle generation.
- `needle_seeds`: optional advanced per-needle overrides, taking precedence over `needle_seed` for listed needle indices.

Derivation behavior:

- Haystack seed per example:
  - default: `global_random_seed * 1_000_003 + ex_idx`
  - override: `haystack_seed + ex_idx`
- Needle seed per needle/example:
  - per-index override path: `needle_seeds[i] + ex_idx`
  - scalar override path: `needle_seed + ex_idx * 1_009 + i`
  - default path: `global_random_seed * 10_000_019 + ex_idx * 1_009 + i`
- Control segment seed:
  - `global_random_seed * 100_000_007 + ex_idx * 10_007 + needle_idx * 101 + 53`

This design gives deterministic generation with stable reproducibility while still varying examples by index.

### CLI overrides for notebook runs

Both `scripts/gen_responses.py` and `scripts/analyze_hidden_states.py` accept the dataset-shape overrides below in addition to values loaded from `--config`:

```bash
--target-haystack-tokens 1000 --num-needles 5 --positions null 200 400 600 800
```

Use these whenever the notebook configuration cell changes `TARGET_HAYSTACK_TOKENS`, `NUM_NEEDLES`, or `INSERTION_POSITIONS`. Use `null`/`None` in `--positions` to generate a needle but skip insertion at that aligned index. The `--control_switch` list must contain exactly one boolean per needle for hidden-state analysis, for example `--control_switch true false false false false` for five needles.

For sentence-boundary insertion runs, set `SENTENCE_LEVEL_INSERTION=True` in `notebooks/counting_feature_analysis.ipynb` or pass `--sentence-level-insertion` to `scripts/generate_dynamic_niah_v2.py`. In this mode, the generated config records `sentence_level_insertion=true`, and counting run/cache names include `sent_insrt` to avoid reusing older token-position datasets. Sentence-boundary candidates are conservative: periods inside URL/path/domain-like text, common abbreviations, and numeric fragments are filtered out before random insertion sites are sampled. If a haystack chunk has no conservative sentence-ending candidates, the generator prints a warning and falls back to word-boundary insertion for that example, recording `sentence_delimiter_filter_fallback="word_boundary"` in insertion metadata. Regenerate datasets when validating this behavior, since old caches keep their original delimiter choices.

For word-boundary insertion runs, set `WORD_LEVEL_INSERTION=True` in `notebooks/counting_feature_analysis.ipynb` or pass `--word-level-insertion` to `scripts/generate_dynamic_niah_v2.py`. This mode is mutually exclusive with sentence-level insertion. It places needles at whitespace boundaries between existing words, records character and token spans, and includes `word_insrt` in counting run/cache names.

For simpler exact-marker counting, set `COUNTING_NEEDLE_KIND="marker"` and optionally adjust `MARKER_TEXT` in `notebooks/counting_feature_analysis.ipynb`, or pass `--counting-needle-kind marker --marker-text '[dolphin]'` to `scripts/generate_dynamic_niah_v2.py`. Existing city-score behavior remains the default with `COUNTING_NEEDLE_KIND="city_score"`.

For current `match_count` counting-feature runs, the notebook also sets `FACT_TEMPLATES_PATH` to `data/templates/niah_fact_single_template.txt` and passes it to dataset generation with `--fact-templates-path`. The resolved `fact_templates_path` is saved in `config.used.json` and checked during reusable dataset-cache validation.

### Control-switch behavior

`control_switch` is a boolean list aligned to needle index:

- `False`: insert the real generated needle fact.
- `True`: insert a same-length control token segment from haystack text.

Important implications:

- `gold_answer` is computed from all original generated needles.
- `control_gold_answer` is computed only from **non-control** needles.
- If all needles are controls, `control_gold_answer.has_answer` is `False`.
- Response prediction reports keep `gold_answer` as the original dataset answer, set `expected_answer` to that same all-needle answer for scoring, and include `control_gold_answer` only as auxiliary control-subset metadata. Model-response generation therefore builds its prompt directly from the all-needle `uncontrolled_context` when controls are enabled; the controlled `messages` prompt is retained for control-condition analysis.

### Prompt and message construction

Prompt construction is config-driven:

- `prompt_style="easier"` → easier prompt builder
- `prompt_style="vanilla"` → vanilla prompt builder
- `prompt_style="vanilla_no_cue"` → vanilla ordering with only the JSON-format instruction before context
- `thinking_mode` toggles thinking-aware formatting paths

Controlled-condition messages are emitted in each row under `messages`. Rows also include `uncontrolled_context`, which restores any controlled spans back to their original needle facts, and `uncontrolled_messages` as a saved all-needle prompt for inspection; response evaluation rebuilds the model-facing prompt from `uncontrolled_context` so `count_avg` examples with one control still present all three rated cities to the model.

### Output artifacts

Each generator run writes canonical artifacts to a run folder:

- JSONL dataset at `results/run_.../generate_data/dynamic_niah_v2.jsonl`
- effective run config at `results/run_.../generate_data/config.used.json`
- generated-row schema copy at `results/run_.../generate_data/dataset.schema.json`
- run metadata at `results/run_.../run_metadata.json`
- terminal output at `results/run_.../logs.txt`

If `output_dir` or `data_save_path` is specified, additional copies are written there and the script prints the extra-copy path.


### Run config JSON files

- `generate_data/config.used.json`: the effective Dynamic NIAH dataset-generation config for the saved JSONL rows, after JSON loading and CLI/notebook overrides.
- `run_metadata.json`: run-level metadata with the model name, run directory, compact run-name parameters, and resolved Dynamic NIAH config.
- `analyze_hidden_states_config.json`: hidden-state-analysis settings for the run, including model, layers, control switch, artifact directories, and resolved Dynamic NIAH config.
- `tensors/qk_cache/qk_cache_metadata.json`: consolidated Q/K-cache metadata for notebook outlier-analysis runs, including per-example prompts, token strings, analysis specs, tokenizer/model details, sequence lengths, selected layers, special tokens, and cache file names.
- `tensors/qk_cache/input_{i}/`: per-example Q/K tensor cache directories. Notebook outlier-analysis runs keep these directories focused on `.pt` tensors, while standalone capture-script runs may still write self-contained metadata files inside a single cache directory for debugging.
- `qk_outlier_analysis_config.json`: notebook-level Q/K outlier-analysis settings such as selected layers/heads, uncontrolled-prompt-only mode, massive-token thresholds, critical-token windows, and output subdirectories.

### Dataset row schema

`dataset.schema.json` describes each JSONL row emitted by `scripts/generate_dynamic_niah_v2.py`. The writer copies this schema into every generated-data folder next to `dynamic_niah_v2.jsonl`, and each row carries `schema_version: "dynamic_niah_v2_dataset_v1"`. The schema documents top-level fields such as tokenizer metadata, haystack token windows, generated needles, realized insertions, query/messages, gold answers, control-aware answers, relevant records, and run controls.

Legacy v1 specs and the old general-generator config were removed during repository cleanup because the active notebooks and scripts use the Dynamic NIAH v2 JSONL output described here.

## Example config

See `configs/niah_dynamic.json` for a concrete baseline configuration.

Notable defaults in that config include:
- `results_root: "results"`
- `run_dir: null`
- `run_name: null`
- `output_dir: null`
- `data_save_path: null`
- `save_data: false`

## Hidden-state analysis

`scripts/analyze_hidden_states.py` creates the same canonical run structure. Hidden-state tensors and plots are separated by artifact type:

```text
results/run_.../
├── figures/
│   ├── inputs_0.png
│   └── inputs_1.png
├── tensors/
│   ├── inputs_0.pt
│   ├── inputs_1.pt
│   ├── hidden_inputs_0.pt
│   └── hidden_inputs_1.pt
├── generate_data/
│   └── dynamic_niah_v2.jsonl  # when save_data=True
├── analyze_hidden_states_config.json
├── logs.txt
└── run_metadata.json
```

Updates supported by the analysis script:

- `generate_control_dataset_with_logging(cfg)` reads save behavior from config (`cfg.save_data`) rather than a separate function argument.
- `save_data` is part of `DynamicNiahV2Config` and can be set in `configs/niah_dynamic.json`.
- `scripts/analyze_hidden_states.py` supports `--save_data true|false` to override config.
- `scripts/analyze_hidden_states.py` supports `--prompt-style {vanilla,easier}` to override `prompt_style` from config.
- When `save_data=True`, generated rows are persisted to the run's `generate_data/dynamic_niah_v2.jsonl`; if that canonical dataset already exists, hidden-state analysis loads it instead of regenerating rows.
- `scripts/analyze_hidden_states.py` writes `analyze_hidden_states_config.json` to the run root.
- PCA visualization reserves the earliest `--pca-test-count` hidden-state records as test examples and fits on later records; when omitted, `--pca-test-count` defaults to `num_examples // 2`.
- Hidden-state input construction follows config-driven behavior (`prompt_style`, `thinking_mode`, tokenizer from `cfg`).
- If `output_dir` is specified, analysis figures/tensors/config are copied to that extra directory.
- If `data_save_path` is specified, generated rows are copied to that extra JSONL path.
