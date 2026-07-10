# Plan

### General instructions
Principles: Read the plan carefully. 
- You are encouraged to raise clarification questions about typos, ambiguities, uncertainties, etc. Ask me if you need more information for making decisions. 
- Ask to correct obvious typos (with minimal changes) in this document before you code.
- Remember to update the `README.md` after coding
- If I ask you to include the Q&A interaction, summarize our interaction and add your summary under a subsection titled "#### Clarification Q&A (YYYY-MM-DD-time)" right below the requested section
- Don't modify the code block in the notebook that contains the `drive.mount()` command, unless I explicitly tell you to do so.

## Summary of previous dataset generation and analysis

The current Dynamic NIAH pipeline generates tokenizer-aware needle-in-a-haystack datasets from Paul Graham essay text. Earlier plans built support for fixed token insertion, randomized insertion, sentence-level insertion after delimiters, repeated haystack sampling to reach the target context length, and metadata-rich JSONL rows that preserve inserted needles, realized token spans, prompt messages, gold answers, and scoring outputs.

The counting-feature workflow then uses these generated datasets to evaluate Qwen3-8B on counting-style tasks. The recent `match_count` experiments inserted city-score records, asked the model to return JSON with a count, filtered successful examples, extracted hidden states, and fit or computed counting directions using ridge, contrastive-success, and counterfactual methods. The most promising current feature-vector method is the counterfactual count direction, usually calculated at `FEATURE_CALC_POS="needle-last"` and steered with `STEERING_POSITION_MODE="needle_span"`.

The ablation and steering analyses show several recurring patterns. Ridge vectors are useful as probes but poor as steering vectors. Counterfactual steering can modestly improve selected held-out settings, but it is target-sensitive and does not yet robustly solve larger counts. Main findings from `main_findings/main-findings-061626.md` indicate that Qwen3-8B counts small numbers of inserted records reliably but becomes unreliable for larger counts. Shortening the haystack from around 1000 tokens to 500 or 200 tokens does not remove this large-count failure, suggesting that the bottleneck is not just long-context retrieval; the task may stress cardinality tracking or maintaining a stable set of many retrieved records.

The dataset construction itself remains a major experimental factor. The move from varied city-score templates to a single template improved interpretability, and sentence-level insertion fixed boundary artifacts. However, the current city-score task still contains unnecessary semantic baggage: city names, survey/audit wording, and scores. To isolate counting more cleanly, the next dataset revision should use a simpler exact marker that carries almost no semantic content beyond being a repeated item to count.

## Generating even simpler NIAH tasks

This plan introduces a simpler marker-counting NIAH variant. Instead of inserting city-score facts such as survey records, the generator should insert the exact marker:

```text
[dolphin]
```

The goal is to test whether the model can count repeated exact markers in context without needing to parse city names, scores, or audit language. The gold answer calculation remains similar to the previous `match_count` setup: the answer is the number of active inserted matching needles in the final context.

1. Add support for a simple exact-marker counting task.

The generator should be able to create examples where each active needle is exactly the same marker string:

```text
[dolphin]
```

The marker should be inserted into the haystack with whitespace before and after it. This applies to fixed token insertion, randomized token insertion, sentence-level insertion, and the new word-level insertion mode described below. The final context should contain readable boundaries such as:

```text
... haystack text. [dolphin] More haystack text ...
```

When multiple needles are inserted after the same sentence delimiter because sentence-level insertion samples with replacement, the result may contain back-to-back markers, but each marker should still be separated by whitespace:

```text
... haystack sentence. [dolphin] [dolphin] next sentence ...
```

2. Add word-level insertion.

Add a new insertion mode called word-level insertion. This mode should insert `[dolphin]` at randomly selected whitespace boundaries in the haystack text, without breaking existing words. For example, if the haystack contains:

```text
... word1 word2 ...
```

then an inserted marker may produce:

```text
... word1 [dolphin] word2 ...
```

The candidate insertion sites should be text-level whitespace runs whose immediate left and right sides contain non-whitespace characters. In other words, insertion should occur between existing word-like chunks, not in the middle of a token or inside a word. The implementation should not assume that tokenizer token boundaries line up cleanly with word boundaries; the text should be modified first, then the inserted marker spans should be recovered and verified after tokenization.

If the number of active needles is larger than the number of available word-boundary candidates, word-level insertion may sample insertion sites with replacement, following the recent sentence-level insertion behavior. This allows short haystacks to contain many markers. When the same word boundary is selected more than once, the final context may contain back-to-back markers:

```text
... word1 [dolphin] [dolphin] word2 ...
```

Each marker must still have whitespace around it and an independent span entry.

3. Re-check sentence-level insertion and word-level insertion around tokenization.

The existing sentence-level insertion mode inserts needles after delimiters such as `.`, `?`, and `!`, and now allows sampling delimiter sites with replacement when there are fewer delimiters than active needles. For `[dolphin]`, this behavior should continue to produce text like:

```text
... sentence. [dolphin] next sentence ...
```

or, with replacement:

```text
... sentence. [dolphin] [dolphin] next sentence ...
```

Both sentence-level and word-level insertion are text-level operations, so tokenization can introduce subtle issues. The implementation should explicitly verify, for every inserted copy, that:

- the character span points to exactly `[dolphin]`;
- the token span is non-empty and maps to the intended context region;
- repeated identical markers are not collapsed into a single occurrence;
- markers inserted at the same sentence or word boundary remain independently recoverable;
- the realized token spans used by `FEATURE_CALC_POS="needle-last"` correspond to the inserted marker, not to surrounding whitespace or punctuation.

If exact token-span recovery fails, the generator should raise an error rather than silently writing ambiguous metadata. This is especially important because `[dolphin]` may tokenize differently depending on surrounding whitespace, and because insertion after delimiters or inside whitespace runs may shift character offsets after earlier insertions.

4. Preserve span and metadata guarantees.

Even though every inserted marker has the same text, each inserted copy should still have its own metadata entry. The dataset should save:

- needle id / ordinal;
- inserted text, always `[dolphin]`;
- requested and realized insertion information;
- character span in the final context;
- verified token span in the final tokenized prompt/input;
- whether sentence-level insertion used replacement for that example;
- whether word-level insertion was used, how many word-boundary candidates existed, and whether word-boundary sampling used replacement.

The span-location code must not collapse repeated identical markers into one ambiguous occurrence. It should verify each inserted marker independently and fail loudly if an inserted marker span cannot be recovered.

5. Keep the gold-answer logic count-based.

For this exact-marker task, `gold_answer` should be:

```json
{"count": k}
```

where `k` is the number of active inserted marker copies. If some insertion positions are `None`, those slots should not contribute to the count. If a counterfactual dataset removes one active marker slot, the counterfactual gold count should decrease by one, following the same logic used by the current counterfactual count direction.

6. Add a prompt style for the simple marker task under `vanilla`.

For `PROMPT_STYLE = "vanilla"` and the simple marker task, the instruction before the context should be:

```text
Return ONLY one JSON object on a single line with schema {"count":0}. No extra text. The exact marker "[dolphin]" is inserted one or more times within the following text. Make sure to memorize them.
```

After the context, the query should be:

```text
How many times does the exact marker "[dolphin]" appear in the context? Count only exact copies of "[dolphin]". Respond as JSON with key count.
```

The earlier draft wrote `"{dolphin}"` in one place. That was a typo; the intended exact marker throughout this plan is `[dolphin]`.

7. Add a new prompt style `vanilla_no_cue`.

Add a new prompt style:

```python
PROMPT_STYLE = "vanilla_no_cue"
```

For the simple marker task, `vanilla_no_cue` should remove the marker-specific cue from the instruction. The instruction should become:

```text
Return ONLY one JSON object on a single line with schema {"count":0}. No extra text.
```

The context should remain unchanged. The query after the context should still ask the model to count the exact marker:

```text
How many times does the exact marker "[dolphin]" appear in the context? Count only exact copies of "[dolphin]". Respond as JSON with key count.
```

This style tests whether the pre-context cue helps the model track marker occurrences, while keeping the final question explicit enough for unambiguous scoring.

8. Keep older city-score tasks available.

Do not remove the existing `match_count`, `literal_count`, `argmax`, or `count_avg` behavior. The simple marker task should be added as a new controlled variant or a clearly named configuration path. Existing city-score experiments, templates, and run folders should remain reproducible.

Possible implementation choices include:

- adding a new task type such as `marker_count`;
- adding a new template/needle mode for `match_count`;
- or adding a config variable such as `COUNTING_NEEDLE_KIND = "city_score" | "marker"`.

Before coding, choose the option that best fits the existing generator and scoring code with the least ambiguity.

9. Update the counting-feature workflow to support the simple marker task.

The counting-feature notebook and script should work with the simple marker task using the same high-level pipeline:

- generate or load the marker-count dataset;
- run response generation and scoring;
- filter successful examples;
- locate inserted marker spans;
- compute ridge, contrastive-success, or counterfactual directions;
- optionally run regular steering and held-out steering eval.

The current `FEATURE_CALC_POS="needle-last"` logic should remain meaningful: it should refer to the final token inside the final inserted `[dolphin]` marker span.

10. Add focused validation.

Add tests or smoke checks that verify:

- each generated example contains exactly the expected number of `[dolphin]` markers;
- there is whitespace before and after inserted markers when text context permits;
- repeated identical markers have distinct verified spans;
- sentence-level insertion after delimiters still preserves exact marker text and independent spans, including replacement cases;
- word-level insertion places markers only at whitespace boundaries between non-whitespace text, never inside an existing word;
- word-level insertion with replacement can create back-to-back markers without losing span recoverability;
- `vanilla` and `vanilla_no_cue` render different instructions but the same exact-marker query;
- `gold_answer["count"]` equals the number of active inserted markers;
- counterfactual marker datasets reduce the count by one when one active marker slot is removed;
- old city-score tasks still pass existing tests.

11. Document the new task.

Update `README.md` after implementation to explain the simple marker-count task, the new `vanilla_no_cue` prompt style, the available insertion modes, and how this task differs from the city-score `match_count` setup. The documentation should make clear that this task is intended to isolate counting from semantic parsing of city-score records.

## Revising literal_matching task

This revision should slightly modify the existing `literal_matching` task. The goal is to keep the current `literal_matching` strategy of generating literal UID-style needles, while making the desired UID token length configurable and reducing boundary artifacts from tokenization. Do not implement these changes until this plan is approved.

1. Add a notebook-level `UID_token_length` setting.

In `notebooks/counting_feature_analysis.ipynb`, introduce a global configuration variable:

```python
UID_token_length = 4
```

The default should remain `4`, matching the current `literal_matching` setting. This variable should control the desired tokenizer length of each generated UID/literal needle used by `literal_matching`.

The existing strategy for generating the `literal_matching` literal needles should be preserved, but the UID-generation logic should be generalized so that the desired token length can be changed. If `UID_token_length` changes, the generator should choose or construct UID strings whose tokenizer encoding has exactly the requested length under the configured tokenizer.

2. Verify UID token length at generation time.

Before writing the dataset, the generator should explicitly tokenize each generated `literal_matching` UID/literal needle and verify:

- the tokenized UID length equals `UID_token_length`;
- the UID text is non-empty and stable under the configured tokenizer;
- each realized inserted needle copy has the intended UID text;
- each realized inserted needle copy records the expected inserted token length;
- any mismatch raises an error rather than silently generating ambiguous data.

This check should happen for the actual tokenizer used in the run, not from a hard-coded assumption about the UID string format.

3. Insert literal_matching needles at word-boundary whitespace sites.

To avoid boundary effects from tokenization, `literal_matching` UID needles should be inserted at text positions between two words, using the same broad idea as the existing random word-level insertion strategy. If the haystack contains:

```text
... word1 word2 ...
```

and the insertion site is the whitespace between `word1` and `word2`, the final text should become:

```text
... word1 UID word2 ...
```

Here `UID` means the generated literal needle for the `literal_matching` task. The UID should not be inserted inside an existing word, inside punctuation-token fragments, or at arbitrary tokenizer boundaries. The insertion candidates should be text-level whitespace runs whose immediate left and right sides are non-whitespace text.

This is similar to the current random word insertion mode, but the inserted payload is the generated `literal_matching` UID. The implementation should preserve readable text boundaries around the UID.

4. Re-verify token length after insertion.

Because tokenization can change when a UID is adjacent to spaces or neighboring text, the generator must double-check the realized UID in context. For each inserted `literal_matching` needle copy, verify that:

- the character span in the final context points exactly to the intended UID text, excluding surrounding spaces;
- token-span recovery identifies the intended inserted UID occurrence;
- the recovered context-local token span has length exactly `UID_token_length`;
- repeated or similar UID needles remain independently recoverable;
- back-to-back UID insertions, if sampling with replacement creates them, still have distinct spans and the desired token length.

If inserting at a candidate word boundary causes a UID to tokenize to a different length than `UID_token_length`, that candidate should be rejected or the example generation should fail loudly. The generator should not keep a row where any realized UID length differs from the requested length.

5. Preserve compatibility with current defaults and metadata.

With the default `UID_token_length = 4`, the intended behavior should reproduce the current `literal_matching` setting as closely as possible, except with stronger word-boundary placement and token-length validation. Existing metadata fields such as needle id, literal UID text, `token_length`, `inserted_tokens`, `context_span_start`, `context_span_end`, `token_span_verified`, and `realized_insertions` should remain available for downstream counting-feature analysis.

The counting-feature workflow should continue to treat `FEATURE_CALC_POS="needle-last"` as the final token of the verified `literal_matching` UID span. If downstream prompt-level span localization includes adjacent whitespace because of chat-template tokenization, that should be documented separately from the dataset-level inserted UID length. The dataset-level guarantee should remain that each `literal_matching` UID insertion has exactly `UID_token_length` tokens.

6. Add validation before running larger experiments.

Before large Colab runs, add or run focused checks for:

- `UID_token_length = 4` with the current default `literal_matching` UID generation;
- generated examples containing exactly the requested number of `literal_matching` needles;
- every inserted UID having dataset-level token length `UID_token_length`;
- word-boundary insertion not splitting words;
- repeated or similar UIDs remaining independently recoverable;
- existing non-`literal_matching` tasks remaining compatible where applicable.

## revising counting feature calculation

The current counting-feature calculation appears to rely on filtering the generated/scored dataset down to examples where the model response was successful before fitting probes or computing feature directions. In particular, the notebook uses the scored prediction file to select successful rows before tokenizing examples and building the token-level counting target. If this reading is wrong, first verify the actual dataflow in `notebooks/counting_feature_analysis.ipynb` and the helper functions under `src/counting/feature_analysis.py` before coding.

This filtering should become optional while preserving the existing behavior by default.

1. Add a notebook-level filtering switch.

In `notebooks/counting_feature_analysis.ipynb`, introduce a global configuration variable:

```python
FILTER_EXAMPLE = True
```

The default should be `True`, so existing experiments remain unchanged. With this default, the notebook should continue to compute counting features only from examples that pass the current success filter.

2. Allow counting-feature calculation on all examples.

When `FILTER_EXAMPLE = False`, the counting-feature calculation should skip the success-only filtering step and use all generated examples for the relevant feature calculation stage. This should apply to the rows used for tokenization, target construction, hidden-state extraction, probe fitting, and feature-vector calculation, unless a later step has a separate scientifically necessary filter such as missing data or invalid spans.

The implementation should distinguish between:

- response correctness filtering, controlled by `FILTER_EXAMPLE`;
- hard validity checks, which should still run regardless of `FILTER_EXAMPLE`;
- missing scores, malformed rows, failed tokenization, span-location failures, NaNs, or empty result sets, which should still raise errors rather than being silently ignored.

3. Preserve audit metadata.

Run metadata should record whether filtering was enabled. The saved `counting_feature_run_metadata.json` and any split/filter summary should make clear:

- the configured value of `FILTER_EXAMPLE`;
- the total number of generated rows;
- the number of scored rows;
- the number of successful and unsuccessful model responses;
- the number of rows actually used for counting-feature calculation;
- whether unsuccessful examples were included because filtering was disabled.

This is important because probe metrics from successful-only examples and all-example datasets answer different scientific questions.

4. Keep train/test splitting reproducible.

When `FILTER_EXAMPLE = True`, train/test splitting should remain exactly as it is now. When `FILTER_EXAMPLE = False`, the same split seed and test fraction should be applied to the larger all-example set. The split summary should be saved as before so the run can be audited or reproduced.

5. Validation.

Add focused checks that verify:

- the default `FILTER_EXAMPLE = True` reproduces the current successful-example-only behavior;
- `FILTER_EXAMPLE = False` includes unsuccessful but otherwise valid examples in the feature-calculation dataset;
- metadata records both the success counts and the number of rows actually used;
- span validation and target construction still fail loudly on invalid examples;
- no existing counting-feature tests change behavior under the default setting.
