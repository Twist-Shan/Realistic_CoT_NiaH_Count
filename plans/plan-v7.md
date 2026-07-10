# Plan

### General instructions
Principles: Read the plan carefully. 
- You are encouraged to raise clarification questions about typos, ambiguities, uncertainties, etc. Ask me if you need more information for making decisions. 
- Ask to correct obvious typos (with minimal changes) in this document before you code.
- Remember to update the `README.md` after coding
- If I ask you to include the Q&A interaction, summarize our interaction and add your summary under a subsection titled "#### Clarification Q&A (YYYY-MM-DD-time)" right below the requested section

## Summary of existing plans

The earlier plans progressively built the current NIAH dataset-generation and counting-analysis workflow. `plan-v2.md` introduced tokenizer-aware dynamic NIAH generation: collect Paul Graham essay haystacks, tokenize a sampled haystack window to a target length, generate city-rating needles, insert needles at requested token positions, and save enough metadata to recover prompts, needles, token spans, and gold answers.

`plan-v4.md` extended the dataset-generation pipeline to counting tasks, especially `match_count` and `literal_count`. For `match_count`, the gold answer became the number of inserted matching city-score needles. This plan also added support for `None` insertion positions, revised run naming to include insertion-position patterns, and emphasized checking token-boundary behavior for literal needle matching.

`plan-v5.md` built the counting-feature notebook on top of these generated datasets. It added reusable generation/loading, response scoring, successful-example filtering, hidden-state extraction, and saved artifacts for feature analysis. It also introduced randomized needle insertion, per-row insertion metadata, and needle-span-aware steering, making actual located needle spans more important than nominal insertion positions.

`plan-v6.md` explored alternative counting-feature directions and broader steering evaluation. It reused the same dataset-generation backbone, added original/counterfactual counting datasets, required careful saving of original and counterfactual metadata, and generated fresh steering-evaluation datasets with varying numbers of needles. These experiments revealed that dataset construction details, especially whether inserted needles are clearly separated from surrounding haystack text, can strongly affect baseline task accuracy and downstream steering interpretation.

## Revising NIAH dataset generation

This plan revises the NIAH dataset construction so that counting examples are easier to audit and less vulnerable to accidental boundary artifacts between haystack text and inserted needles. You are likely going to mainly revise `src/dataset_generation/dynamic_niah_v2.py`, `src/dataset_generation/gen_responses.py`, `scripts/generate_dynamic_niah_v2.py`, and the notebook `notebooks/counting_feature_analysis.ipynb`.

1. When sampling haystack text, the generator should always produce enough source text to reach `TARGET_HAYSTACK_TOKENS`. If the selected haystack file is too short after tokenization, repeat or extend the haystack text until there is a tokenized sequence long enough to sample a window of length `TARGET_HAYSTACK_TOKENS`. The generator should avoid silently producing examples with fewer haystack tokens than requested.

2. Introduce a new notebook-level global variable:

```python
SENTENCE_LEVEL_INSERTION = True
```

If `SENTENCE_LEVEL_INSERTION = False`, keep the existing needle-insertion strategy unchanged.

If `SENTENCE_LEVEL_INSERTION = True`, print a clear message saying that needles will be inserted randomly at sentence ends. In this mode, ignore the fixed insertion behavior described by `INSERTION_POSITIONS` and choose insertion sites only after sentence-ending delimiters: `.`, `!`, and `?`. The inserted needle should begin after a whitespace following the delimiter, so the needle is not glued to the preceding haystack word or punctuation.

3. After constructing the full haystack-with-needles text, tokenize the final text and save the usual dataset metadata. The metadata should include the inserted needle strings, insertion information, and especially the token spans of each inserted needle in the final tokenized prompt/input. The code should locate and verify the needle spans after tokenization, rather than assuming that text-level insertion locations automatically imply correct token-level spans.

4. Rewrite the instruction and query text so that the instruction is shared across tasks, but the question is task-specific. The canonical task names in the current code are `argmax`, `count_avg`, `match_count`, and `literal_count`; `literal_match` is not the current task name.

For `match_count`, use:

```text
Return ONLY one JSON object on a single line with schema {"count":0}. No extra text. Do NOT explain or include reasoning.

Query:
One or more cities and their ratings are provided in a survey or audit. Make sure to memorize them. You need to answer the following question: How many cities receive a score? Respond as JSON with key count.
```

For the other NIAH tasks, keep the same instruction template but substitute the appropriate response schema and task question:

- `argmax`: ask which city has the highest score, and request JSON with keys `city` and `score`.
- `count_avg`: ask how many cities are rated and what their average score is, and request JSON with keys `count` and `average_score`.
- `literal_count`: ask how many exact copies of the generated literal/canary string appear in the context, and request JSON with key `count`.

The goal of this prompt revision is to make the expected output format explicit while avoiding a hard-coded `match_count` question for tasks whose gold answers and response schemas are different.

### Revision

The first sentence-level insertion run suggests that the boundary problem is fixed, but the counting task remains brittle because the city-score needles use two different surface forms. In particular, the model often appears to count only the two needles written as `travel preference audit ... satisfaction score was ...` and ignore the one written as `Survey note ... score recorded ...`. To remove this avoidable ambiguity, revise the dataset construction so that each city-score needle uses one uniform template.

1. Create a new template file under `data/templates` named `niah_fact_single_template.txt`. This file should be based on `data/templates/niah_fact_templates.txt`, but contain only one city-score template. Prefer a direct wording that matches the counting query, for example:

```text
In the 2024 city score audit, {entity} received a score of {score}.
```

Keep the template simple and repetitive on purpose. The goal is not linguistic variety; the goal is to make every inserted city-score record equally eligible for the counting task.

2. Redirect the relevant dataset-generation workflow to use `data/templates/niah_fact_single_template.txt` for the current counting experiments. The cleanest implementation is probably to change the notebook/config value for `FACT_TEMPLATES_PATH` or the equivalent config field, rather than hard-coding the new template path deep inside the generator. If the existing config plumbing does not expose this path in the notebook, add a notebook-level variable for it and pass it through to `DynamicNiahV2Config`.

3. Keep the old `data/templates/niah_fact_templates.txt` file available for backward compatibility and older experiments. The new single-template file should be an opt-in or explicitly configured template source, not a destructive replacement.

4. Update the `match_count` query so it matches the single-template wording. For example:

```text
One or more city score audit records are provided in the context. Make sure to memorize them. You need to answer the following question: How many cities received a score? Respond as JSON with key count.
```

The intended alignment is:

- Template: `{entity} received a score of {score}.`
- Query: `How many cities received a score?`
- Gold answer: the number of inserted non-null city-score records.

5. Save enough metadata to audit which template file was used. In particular, `config.used.json`, run metadata, and dataset-cache validation should record the resolved `fact_templates_path`, so runs using the two-template file cannot be silently reused for runs using the single-template file.

### Revision 2

Revise the prompt structure so that the general instruction contains both the output-format constraint and the high-level reminder that city information is hidden inside the context. The query should then contain only the task-specific question and response-key reminder. The haystack-with-inserted-needles construction should remain unchanged.

The current workflow has three layers:

1. `generate_dynamic_niah_dataset_v2(...)` builds `context`, which is the haystack text after inserting the generated needles.
2. `build_task_query(...)` builds the task-specific query string.
3. `build_messages_easier(...)` or `build_messages_vanilla(...)` combines instruction, query, and context into the user chat message. For `prompt_style = "easier"`, the order is currently instruction, then query, then context.

Keep that high-level structure, but change the text allocation:

1. The instruction should start with the existing JSON-format instruction:

```text
Return ONLY one JSON object on a single line with schema {schema}. No extra text.
```

When thinking mode is off, keep the existing extra sentence:

```text
Do NOT explain or include reasoning.
```

Then add:

```text
Some information about cities are inserted within the following text. Make sure to memorize them.
```

So for `match_count` with thinking mode off, the instruction should become approximately:

```text
Return ONLY one JSON object on a single line with schema {"count":0}. No extra text. Do NOT explain or include reasoning. Some information about cities are inserted within the following text. Make sure to memorize them.
```

2. The `match_count` query should remove the memorization sentence and use wording aligned with the single city-score template:

```text
One or more city score audit records are provided in the context. You need to answer the following question: How many cities received a score? Respond as JSON with key count.
```

3. The haystack-with-inserted-needles should remain unchanged. In particular, do not add the memorization sentence inside the haystack/context text, and do not change sentence-level insertion behavior or needle-span metadata for this revision.

4. Make the revision work for all four current Dynamic NIAH v2 tasks:

- `match_count`: use schema `{"count":0}` and ask how many cities received a score.
- `argmax`: use schema `{"city":"","score":0}` and ask which city has the highest score.
- `count_avg`: use schema `{"count":0,"average_score":0.0}` and ask how many cities are rated and what their average score is.
- `literal_count`: use schema `{"count":0}` and ask how many exact copies of the generated literal/canary appear in the context.

5. Be careful with `literal_count`. The instruction sentence about city information is mainly written for city-score tasks. To avoid confusing literal-count runs, implement the instruction preface as task-aware:

- For `argmax`, `count_avg`, and `match_count`, add the city-information sentence.
- For `literal_count`, add an analogous literal-specific sentence, for example:

```text
Some exact literal strings are inserted within the following text. Make sure to memorize them.
```

This keeps the common prompt structure while avoiding a city-specific instruction for non-city literals.

7. Avoid duplicate city names within a single city-score example. For `match_count`, `argmax`, and `count_avg`, sample inserted city entities without replacement within each generated example, so the same city cannot appear twice with different scores in the same context. This removes an ambiguity in the question wording: `How many cities received a score?` should match the number of inserted city-score records because each inserted record refers to a distinct city. If the entity pool is ever too small to provide the required number of distinct cities, the generator should raise a clear error rather than silently reusing an entity. This distinct-entity constraint does not apply to `literal_count`, whose repeated literal/canary behavior is task-specific.

8. Implementation notes:

- The main code change should be in `src/dataset_generation/niah_prompt_utils.py`, because that file owns the instruction/query/context composition.
- `src/dataset_generation/response_eval.py` should keep task-specific query generation, but remove duplicated memorization wording from the query strings.
- `generate_dynamic_niah_dataset_v2(...)` should not need changes for context construction, except for any metadata updates needed to audit the prompt wording.
- Address both prompt styles explicitly. For `PROMPT_STYLE = "easier"`, keep the order as instruction, query, then context. For `PROMPT_STYLE = "vanilla"`, keep the existing order as instruction, context, then query. In both cases, the added memorization sentence belongs only in the instruction block, not inside the query or context.
- Add tests that inspect both `build_messages_easier(...)` and `build_messages_vanilla(...)`, verifying that the memorization sentence appears once, appears before the context, and does not get duplicated inside the task query.
- Add or update tests for `build_messages_easier`, `build_messages_vanilla`, and `build_task_query` so the intended split between instruction and query is explicit.
- Update `README.md` if the prompt construction description changes.

## Timing logs and default counting-feature config

The recent notebook runs show that it is difficult to reconstruct runtime after the fact. Some run folders have `logs.txt`, but those logs are not timestamped by notebook stage, and some notebook-only runs do not produce a `logs.txt` file at all. Add explicit timing instrumentation to the counting-feature workflow, and promote the current preferred notebook override values into the default config so future runs require fewer manual overrides.

1. Add explicit timing logs to `notebooks/counting_feature_analysis.ipynb`. The notebook should record wall-clock elapsed seconds for major stages and save the result under the run directory, preferably as:

```text
tables/timing_summary.json
tables/timing_summary.csv
```

The timing record should include at least:

- config resolution and cache validation;
- dataset generation / dataset-cache restore;
- baseline response generation;
- target-count and needle-mask construction;
- hidden-state extraction or hidden-state cache restore;
- feature-vector calculation, including separate labels for `ridge`, `contrastive-success`, and `counterfactual`;
- regular steering sweep;
- `STEERING_TEST_EVAL` dataset generation;
- `STEERING_TEST_EVAL` baseline and steered generation;
- result archiving / zip creation.

Each timing row should include a stage name, start timestamp, end timestamp, elapsed seconds, and a short status field such as `completed`, `skipped`, or `failed`. The notebook should still print concise timing messages during execution, but the saved JSON/CSV should be the source of truth for later analysis.

2. Add a small reusable timing helper rather than scattering ad hoc `time.time()` calls throughout the notebook. A lightweight context manager in `src/counting/feature_analysis.py` would be enough. The helper should make it easy for notebook cells to write:

```python
with timing.stage("hidden_state_extraction"):
    ...
```

At the end of the notebook, save the timing records next to the other tables. If an exception occurs inside a timed block, record the failure status before re-raising the error.

3. Change the default counting-feature config in `src/counting/feature_analysis.py` to match the current preferred experiment setup. The goal is that the notebook can keep `CONFIG_OVERRIDES` short, using overrides only for genuinely run-specific choices.

Recommended default changes:

```python
SMOKE_TEST = False
NUM_EXAMPLES = 100
RANDOMIZE_NEEDLE_INSERTION = True
SENTENCE_LEVEL_INSERTION = True
LAYERS = [16, 20, 24, 28]
PROMPT_STYLE = "vanilla"
SAVE_GENERATED_DATA = True
TARGET_COUNT_TYPE = "interpolation"
MAX_TRAIN_TOKENS_PER_LAYER = 50_000
MAX_EVAL_TOKENS_PER_LAYER = 50_000
COUNTING_FEATURE_CALC_METHOD = "counterfactual"
FEATURE_CALC_POS = "needle-last"
STEERING_POSITION_MODE = "needle_span"
STEERING_COEFF = [-2, -1, -0.5, 0.5, 1, 2, 3, 4, 6]
NUM_MAX_NEEDLES_STEERING_EVAL = 5
NUM_EXAMPLES_STEERING_EVAL = 20
STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION = 20
```

Keep `FACT_TEMPLATES_PATH = "data/templates/niah_fact_single_template.txt"` as the default. Keep `USE_THINKING = False`, `RUN_GENERATION_EVAL = True`, `RUN_STEERING = True`, `MAX_NUM_STEERING_EXAMPLES = 10`, and `MAX_NEW_TOKEN_STEERING = 20`.

4. Decide carefully whether `STEERING_TEST_EVAL` should become a default. Because this block can multiply runtime substantially, the safer default is:

```python
STEERING_TEST_EVAL = False
```

Then the user can explicitly turn it on only for serious runs. If the intended workflow is now always to run held-out steering evaluation, change the default to `True`, but document that notebook runtime will increase.

5. After changing defaults, simplify `CONFIG_OVERRIDES` in `notebooks/counting_feature_analysis.ipynb`. Leave a short example override block showing only values that are commonly changed between runs, such as `NUM_EXAMPLES`, `STEERING_TEST_EVAL`, `NUM_EXAMPLES_STEERING_EVAL`, `STEERING_COEFF`, or `USER_RUN_NAME`. The resolved config should still be printed and saved exactly as before.

6. Add a secondary JSON config layer for counting-feature analysis. Create:

```text
configs/counting_analysis.json
```

This file should contain the main parameters that were removed from the longer notebook `CONFIG_OVERRIDES` block, so the settings remain easy to inspect and edit without making the notebook visually noisy. The merge order should be:

```text
script / Python defaults < configs/counting_analysis.json < notebook CONFIG_OVERRIDES
```

In other words, values in `configs/counting_analysis.json` can set or override the defaults in `src/counting/feature_analysis.py`, and values in the notebook `CONFIG_OVERRIDES` can override both the JSON file and the Python defaults. This preserves the notebook as the highest-priority place for one-off run changes while providing a stable, documented config file for ordinary experiment settings.

Because strict JSON does not support comments, use a JSON-compatible documentation pattern rather than invalid `//` or `#` comments. For example, include a top-level metadata/comments object such as `_comments`, and have the loader ignore keys that are only documentation. The actual config values can live in a top-level `config` object, or the loader can accept a flat JSON object while dropping keys whose names start with `_`. The important point is that `configs/counting_analysis.json` should remain valid JSON and should be readable by standard JSON parsers.

The initial JSON file should include the current preferred counting-feature settings, including the common run controls and steering settings:

```text
TASK_TYPE
SMOKE_TEST
NUM_EXAMPLES
RANDOMIZE_NEEDLE_INSERTION
SENTENCE_LEVEL_INSERTION
LAYERS
PROMPT_STYLE
SAVE_GENERATED_DATA
TARGET_COUNT_TYPE
MAX_TRAIN_TOKENS_PER_LAYER
MAX_EVAL_TOKENS_PER_LAYER
COUNTING_FEATURE_CALC_METHOD
FEATURE_CALC_POS
STEERING_POSITION_MODE
STEERING_COEFF
STEERING_TEST_EVAL
NUM_MAX_NEEDLES_STEERING_EVAL
NUM_EXAMPLES_STEERING_EVAL
STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION
```

Add concise comments for settings that are easy to misunderstand, especially `COUNTING_FEATURE_CALC_METHOD`, `FEATURE_CALC_POS`, `STEERING_POSITION_MODE`, `STEERING_TEST_EVAL`, `NUM_EXAMPLES`, `NUM_EXAMPLES_STEERING_EVAL`, and `STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION`. The comments should explain the scientific or runtime consequence of changing the value, not just restate the variable name.

Implementation should add a small loader function, probably in `src/counting/feature_analysis.py`, that reads `configs/counting_analysis.json`, removes comment/metadata fields, and passes the remaining values through the same config validation path used by notebook overrides. Unknown real config keys should still fail loudly rather than being silently ignored. The notebook should print the config file path, show the resolved config after merging, and save enough metadata to know whether a JSON config file was used.

7. Update tests to cover the new defaults, JSON config layer, and timing outputs:

- Update config-default tests in `tests/test_counting_feature_analysis.py`.
- Add a unit test that verifies the precedence order: Python defaults are overridden by `configs/counting_analysis.json`, and notebook-style overrides win over the JSON file.
- Add a unit test that JSON comment/metadata fields are ignored while unknown real config keys still raise an error.
- Add a small unit test for the timing helper, including success and failure cases if the helper lives in `src/counting/feature_analysis.py`.
- Add or update notebook-structure tests so the notebook contains the timing save paths and still records `STEERING_TEST_EVAL` outputs.

8. Update `README.md` to mention the timing summary files, `configs/counting_analysis.json`, and the new default experiment posture: counterfactual feature calculation, `needle-last` feature extraction, `needle_span` steering, vanilla prompt style, sentence-level insertion, and single-template city-score needles.

## Working-directory paths and sentence-delimiter replacement insertion

Two practical issues need to be fixed before the next round of runs. First, the notebook should not save outputs into an old Google Drive repo path when it is executed from a newer repo checkout. Second, sentence-level insertion should support small haystacks with many needles by allowing multiple needles to be inserted after the same sentence-ending delimiter.

1. Remove hard-coded repo paths from `notebooks/counting_feature_analysis.ipynb`.

The notebook currently contains an old Google Drive path to a previous repo version. This is risky because a notebook launched from `dataset-generation-main-v13` can still save outputs under `dataset-generation-main-v12`. The notebook should instead derive the repo directory from the active working directory:

```python
REPO_DIR = Path.cwd().resolve()
```

The notebook may still support an explicit user override if needed, but the default should be the current working directory, not a fixed `/content/drive/.../dataset-generation-main-v12` path. If the notebook changes directories, it should print the resolved `REPO_DIR` and verify that expected repo files exist, such as `src/`, `notebooks/counting_feature_analysis.ipynb`, and `configs/counting_analysis.json`.

2. Apply the same path policy to `scripts/run_counting_feature_analysis.py`.

The script should continue to use the working directory as the repo root unless the user explicitly passes a repo or run-root argument. Library code should not hard-code Google Drive, Colab, or local-machine paths. Any generated run path should be derived from the resolved config and the current working directory / `RUN_ROOT`, then saved in run metadata so the output location is auditable.

3. Make result paths consistent with the active run.

When writing `metrics.json`, `counting_feature_run_metadata.json`, timing summaries, steering summaries, and zip archives, the saved paths should point to the actual current run directory. Reused caches are acceptable, but cached absolute paths from earlier Colab sessions or older repo versions should not become the apparent output path for a new run. If cached artifacts are reused, metadata should distinguish:

```text
current_run_dir
cache_source_path
artifact_output_path
```

This makes it clear whether an artifact was recomputed, copied from cache, or restored from an earlier run.

4. Revise sentence-level insertion when delimiters are fewer than needles.

For `SENTENCE_LEVEL_INSERTION = True`, collect all sentence-ending delimiter offsets from the sampled haystack using `.`, `?`, and `!` as candidate insertion sites. If the number of candidate delimiters is at least the number of active needles, keep sampling without replacement as before.

If the number of candidate delimiters is smaller than the number of active needles, do not raise an error. Instead, print a clear warning and sample insertion sites with replacement. This allows multiple needles to be inserted after the same sentence boundary.

5. Define the ordering for multiple needles at the same delimiter.

When sampling with replacement assigns more than one needle to the same delimiter, the generated text may contain back-to-back needles after that sentence. Preserve the natural needle order by sorting insertions by:

```text
sentence delimiter offset, then needle ordinal
```

For example, if three needles are assigned to two sentence endings, the output may look like:

```text
[haystack sentence 1.] [needle 1] [needle 2] [haystack sentence 2.] [needle 3]
```

Each inserted needle should still be separated by whitespace so the final text is readable and token-span recovery is stable.

6. Preserve and verify metadata for repeated delimiter insertions.

Even when multiple needles are inserted after the same delimiter, each needle must have its own metadata entry, including:

- needle id / ordinal;
- inserted text;
- sentence delimiter character offset;
- whether insertion-site sampling used replacement;
- text slice offsets in the final haystack-with-needles string;
- verified token span in the final tokenized prompt/input.

The token-span verification logic should continue to locate each inserted needle independently. The implementation should fail loudly if any repeated-delimiter insertion causes an ambiguous or unverifiable span.

7. Keep token-based insertion behavior unchanged.

This replacement-sampling change applies only when `SENTENCE_LEVEL_INSERTION = True`. If sentence-level insertion is false, the existing fixed-position and token-randomized insertion behavior should remain unchanged, including the existing range and separation checks.

8. Update tests and documentation.

Add focused tests for:

- no hard-coded `dataset-generation-main-v12` or other repo-version path in the notebook;
- notebook/script path resolution from the working directory;
- sentence-level insertion with fewer delimiters than needles;
- repeated delimiter insertion producing back-to-back needles with verified distinct spans;
- ordinary sentence-level insertion still sampling without replacement when enough delimiters exist.

Update `README.md` or the relevant usage notes to explain that sentence-level insertion can now sample with replacement for small haystacks, so many needles may be inserted after the same sentence boundary by design.
