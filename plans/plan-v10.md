# Plan

### General instructions
Principles: Read the plan carefully.
- You are encouraged to raise clarification questions about typos, ambiguities, uncertainties, etc. Ask me if you need more information for making decisions.
- Ask to correct obvious typos (with minimal changes) in this document before you code.
- Remember to update the `README.md` after coding
- If I ask you to include the Q&A interaction, summarize our interaction and add your summary under a subsection titled "#### Clarification Q&A (YYYY-MM-DD-time)" right below the requested section
- Don't modify the code block in the notebook that contains the `drive.mount()` command, unless I explicitly tell you to do so.

## Summary of implemented plans

The earlier Dynamic NIAH plans built the current tokenizer-aware dataset pipeline. The generator samples Paul Graham haystack text to a target token length, inserts needles at fixed, randomized, sentence-level, or word-level positions, verifies realized token spans after tokenization, and saves metadata-rich JSONL rows with prompts, inserted records, span information, gold answers, controls, seeds, and run configuration. Recent dataset revisions made the counting tasks easier to audit by using a single city-score template, adding exact marker and literal-count variants, naming the exact literal in `literal_count` prompts, supporting configurable `UID_TOKEN_LENGTH`, and inserting literal markers at word boundaries while verifying the realized token length.

The counting-feature notebook became the main Colab workflow for generation, response scoring, example filtering, hidden-state extraction, feature calculation, steering, and archiving. It supports optional `FILTER_EXAMPLE`, response-generation checkpoint zips, CUDA memory/timing logs, `USE_KV_CACHE_FOR_NONTHINKG`, and large-run cleanup. It now keeps notebooks as thin orchestrators while reusable logic lives in `src/counting/feature_analysis.py` and dataset-generation modules.

`plan-v6` added feature-vector alternatives to the original ridge counting direction: contrastive-success mean differences, counterfactual count directions, comparison scripts for saved feature vectors, and held-out-style steering evaluation datasets. The main empirical lesson was that ridge probes often decode count well but are weak or damaging steering directions, so a linearly decodable direction is not automatically a causal control direction.

`plan-v9` expanded diagnostic probing. `COUNTING_PROBE_MODE="direct"` preserves the original token-level ridge/classifier pipeline, while non-direct modes fit diagnostic ridge probes for `occurrence_index_probe`, `mean_across_needles_span_last`, `mean_across_needles_span_mean`, and `final_token`; `all_diagnostics` runs all four. The workflow supports `NUM_MAX_NEEDLES` variable-count generation, fake-position baselines, cross-mode ridge-vector cosine similarity heatmaps, optional filtered-train to unfiltered/failed-only evaluation, and unified success/failure diagnostic plots. For sequence-wide diagnostics, the current code projects every token hidden state onto each fitted ridge probe for selected examples, saves raw projection CSVs, and plots projections with needle tokens highlighted and symlog y-scaling. A separate script can regenerate these symlog sequence-projection plots from saved CSVs without rerunning experiments.

# Needle-sensitivity analysis

## Goal

Add a perturbation-based sensitivity analysis that asks how much selected hidden states change when individual needles are removed from an otherwise identical prompt. Instead of only asking whether count information is linearly decodable, this analysis asks whether the representation at a target position is sensitive to the presence of specific needle spans.

For each selected example, randomly choose `NUM_REMOVAL = 3` matching needles, replace each chosen needle span with same-length haystack tokens, rerun the model forward pass on each perturbed sequence, and compare hidden states at `TARGET_SENSITIVITY_POSITION`. The default target is the final prompt token:

```python
TARGET_SENSITIVITY_POSITION = "last-token"
```

The primary scientific question is:

```text
How much does the final-token hidden state change when one counted needle is removed, while sequence length and most token positions are preserved?
```

This should help distinguish three possibilities:

- the target hidden state is highly sensitive to individual needles, suggesting the final representation integrates specific evidence;
- the target hidden state is mostly insensitive to needle removal, suggesting count failures may be readout/generation problems or that the measured layer/position is not where evidence is integrated;
- sensitivity differs for successful vs failed examples, high vs low counts, or short vs long contexts.

## Configuration

Add notebook/config variables with brief inline comments:

```python
RUN_NEEDLE_SENSITIVITY = False  # Run same-length single-needle removal sensitivity analysis.
NUM_REMOVAL = 3                 # Number of matching needles to remove per analyzed example.
TARGET_SENSITIVITY_POSITION = "last-token"  # Hidden-state position measured after each removal.
NEEDLE_SENSITIVITY_SEED = 42    # Seed for selecting removed needles and replacement haystack segments.
MAX_SENSITIVITY_EXAMPLES = None # Optional cap on analyzed examples; None uses the selected analysis set.
```

Default `RUN_NEEDLE_SENSITIVITY=False` keeps the existing pipeline intact. When enabled, this analysis should reuse the same model, tokenizer, rows, tokenized examples, `LAYERS`, and output directories already used by `counting_feature_analysis.ipynb`.

For the first implementation, support:

```text
TARGET_SENSITIVITY_POSITION = "last-token"
```

Plan for future accepted values, but do not implement unless useful:

```text
"removed-span-last"       # hidden state at the last token of the removed/replaced span
"all-matching-span-last"  # hidden states at all matching span-last positions
"needle-span-mean"        # average over matching needle-span positions
```

## Example and needle selection

Use the existing tokenized example representation and span metadata. Do not introduce a second span locator. Matching needles should come from the same logic as counting probes:

- `TokenizedCountingExample.needle_segments`;
- `TokenizedCountingExample.matching_needle_ids`;
- `_matching_segments(...)`.

For each candidate example:

1. Resolve `gold_count` from `gold_answer["count"]`.
2. Locate matching needle spans.
3. Require `len(matching_spans) >= NUM_REMOVAL`.
4. If fewer matching spans are available, skip the example with `exclude_reason="fewer_than_num_removal_needles"`.
5. Sample `NUM_REMOVAL` distinct matching spans without replacement using `NEEDLE_SENSITIVITY_SEED`.
6. Record the selected needle ids, occurrence indices, span starts/ends, span lengths, and replacement segment locations.

The analysis should be deterministic for a fixed run directory and seed. Use a per-example RNG derived from the global sensitivity seed and stable row id or local example index so that adding/removing unrelated examples does not reshuffle every example.

## Same-length replacement construction

For each selected needle span, create one perturbed input sequence by replacing exactly the tokens in that span with a randomly selected same-length non-needle haystack segment. This produces `NUM_REMOVAL` perturbed sequences per original example, all with the same token length as the original input.

Prefer token-level replacement rather than text reconstruction:

```text
perturbed_input_ids = original_input_ids.copy()
perturbed_input_ids[needle_start:needle_end] = replacement_token_ids
```

This avoids accidental retokenization changes and guarantees the sequence length is unchanged. The replacement segment must:

- have the same number of tokens as the removed needle span;
- come from tokens outside all needle spans;
- avoid special/control tokens when such boundaries are known;
- preferably come from the haystack/context portion rather than instruction or query text;
- not overlap the removed span or any other needle span;
- be sampled from a contiguous token segment when possible.

If the code can reliably recover a context/haystack token range from existing metadata, sample replacements only inside that range and outside all needle spans. If the current metadata does not expose a clean context range, use all non-needle prompt tokens as the first implementation fallback, but record:

```text
replacement_source = "non_needle_prompt_tokens"
```

If no same-length replacement segment exists for a span, skip that removal with an explicit `exclude_reason`, not silent fallback to a shorter or padded segment. If too few removals remain after this filtering, keep the valid removals but record `num_valid_removals`.

## Hidden-state extraction

For each included example:

1. Run one forward pass for the original input if the needed hidden states are not already available in memory.
2. Run a batched forward pass for the `NUM_REMOVAL` perturbed inputs when memory allows.
3. Request only the selected `LAYERS` if the model wrapper supports layer-selective extraction. If Hugging Face still returns all hidden states, immediately move selected layer/position tensors to CPU and delete unneeded outputs.
4. Use `model.eval()` and `torch.no_grad()` or `torch.inference_mode()`.
5. Disable KV cache for forward-only hidden-state extraction.
6. Log CUDA memory before/after the sensitivity block and after cleanup.

Long-context runs can easily run out of GPU memory, so this block should be written as a streaming extraction rather than a bulk hidden-state dump. Do not accumulate full sequence hidden states for all original and perturbed examples. For each example or small perturbation batch, immediately slice out only the requested `LAYERS` and `TARGET_SENSITIVITY_POSITION`, move those small tensors to CPU, and delete the model outputs before processing the next batch. Call the existing cleanup helper or explicitly run `del outputs`, `gc.collect()`, and `torch.cuda.empty_cache()` after each example or batch when CUDA is available. If a batch of `NUM_REMOVAL` perturbed sequences is too large for long contexts, fall back to one perturbed sequence at a time and record that fallback in metadata.

Track running time and memory at multiple levels:

- total sensitivity block;
- example selection and perturbation construction;
- original forward passes;
- perturbed forward passes;
- metric calculation and serialization;
- per-layer plotting.

For CUDA runs, record allocated/reserved memory before and after each major substage, and record peak memory when available. These logs are part of the diagnostic output, not optional console-only messages, because early experiments may be used to tune batch size, context length, and `NUM_REMOVAL`.

For `TARGET_SENSITIVITY_POSITION="last-token"`, use:

```python
position = example.sequence_length - 1
```

The position is unchanged across removals because token-level replacement preserves sequence length.

Normalize both original and perturbed hidden states before computing sensitivity metrics:

```python
h0_norm = h0 / (||h0||_2 + eps)
hr_norm = hr / (||hr||_2 + eps)
```

Save raw norms as metadata before normalization so norm changes can still be inspected.

## Sensitivity metrics

For each example, layer, and target position, let:

```text
h0 = normalized original hidden state
h_r = normalized hidden state after removal r
delta_r = h0 - h_r
R = number of valid removals
```

Compute at least:

```text
mean_sensitivity_vector = h0 - mean_r(h_r)
mean_sensitivity_norm   = ||mean_sensitivity_vector||
dist_sensitivity        = sqrt(mean_r ||h0 - h_r||^2)
mean_cosine_drop        = mean_r (1 - cosine(h0, h_r))
max_removal_distance    = max_r ||h0 - h_r||
min_removal_distance    = min_r ||h0 - h_r||
removal_distance_std    = std_r ||h0 - h_r||
```

Because hidden states are normalized, `dist_sensitivity` and `mean_cosine_drop` are closely related but both are useful for readability. `dist_sensitivity` is a geometric distance; `mean_cosine_drop` is an angular change.

Also save per-removal rows with:

```text
removal_index, removed_needle_id, removed_occurrence_index,
removed_span_start, removed_span_end, replacement_start, replacement_end,
original_norm, removed_norm, l2_distance, cosine_similarity, cosine_drop
```

Proposed optional metrics:

- **directional consistency**:

```text
directional_consistency = ||mean_r delta_r|| / mean_r ||delta_r||
```

This is near 1 if all removals move the hidden state in a similar direction, and near 0 if each removed needle perturbs the representation differently.

- **sensitivity per removed occurrence index**: summarize distance by whether the removed needle was early, middle, or late in the matching-occurrence sequence.
- **count-probe projection change**: if diagnostic ridge probes exist for the same layer, project `h0` and each `h_r` onto a chosen ridge vector and report scalar projection drops. This should be optional in the first implementation because it introduces dependencies on which probe artifacts exist.
- **success/failure contrast**: compare sensitivity distributions between examples with `exact_match=True` and `exact_match=False`.
- **count-binned sensitivity**: summarize by gold-count bins such as `1-3`, `4-8`, `9-12`, `13-20`, `21-30`.

## Outputs

Save outputs under a dedicated folder:

```text
tables/counting_features/needle_sensitivity/
tensors/counting_features/needle_sensitivity/
figures/counting_features/needle_sensitivity/
```

Required tables:

```text
tables/counting_features/needle_sensitivity/sensitivity_examples.csv
tables/counting_features/needle_sensitivity/sensitivity_removals_layer_{layer}.csv
tables/counting_features/needle_sensitivity/sensitivity_summary_layer_{layer}.csv
tables/counting_features/needle_sensitivity/sensitivity_metadata.json
```

Suggested `sensitivity_examples.csv` columns:

```text
row_id,local_example_index,gold_count,model_exact_match,
num_matching_spans,num_requested_removals,num_valid_removals,
selected_needle_ids,selected_occurrence_indices,
included,exclude_reason
```

Suggested `sensitivity_removals_layer_{layer}.csv` columns:

```text
layer,row_id,local_example_index,gold_count,model_exact_match,
target_position,removal_index,removed_needle_id,removed_occurrence_index,
removed_span_start,removed_span_end,replacement_start,replacement_end,
original_norm,perturbed_norm,l2_distance,cosine_similarity,cosine_drop
```

Suggested `sensitivity_summary_layer_{layer}.csv` columns:

```text
layer,row_id,local_example_index,gold_count,model_exact_match,
target_position,num_valid_removals,mean_sensitivity_norm,dist_sensitivity,
mean_cosine_drop,max_removal_distance,min_removal_distance,
removal_distance_std,directional_consistency
```

Save tensors only when they are not too large. Preferred tensor payload:

```text
tensors/counting_features/needle_sensitivity/mean_sensitivity_layer_{layer}.pt
```

with a dictionary containing:

```python
{
    "layer": layer,
    "row_ids": [...],
    "local_example_indices": [...],
    "mean_sensitivity_vectors": tensor[num_examples, hidden_dim],
    "dist_sensitivity": tensor[num_examples],
    "target_position": TARGET_SENSITIVITY_POSITION,
    "normalized": True,
}
```

Do not save all full perturbed hidden states by default; that can become large quickly and can make long-context runs fail after successful computation. Save only compact target-position tensors or derived vectors. If a future debugging mode needs raw perturbed hidden states, gate it behind a separate explicit config variable and write a prominent warning about file size and memory pressure.

Required runtime/memory logs:

```text
tables/counting_features/needle_sensitivity/sensitivity_timing_summary.csv
tables/counting_features/needle_sensitivity/sensitivity_timing_summary.json
tables/counting_features/needle_sensitivity/sensitivity_memory_summary.csv
```

The timing rows should include stage name, UTC start/end timestamps, elapsed seconds, status, and any error/skip reason. Memory rows should include stage name, device name when available, CUDA allocated/reserved/peak bytes when available, and CPU fallback markers when CUDA is unavailable.

## Figures

Add compact diagnostic figures:

```text
figures/counting_features/needle_sensitivity/dist_sensitivity_by_count_layer_{layer}.png
figures/counting_features/needle_sensitivity/cosine_drop_by_count_layer_{layer}.png
figures/counting_features/needle_sensitivity/success_failure_sensitivity_layer_{layer}.png
```

Recommended plots:

- scatter or box plot of `dist_sensitivity` vs `gold_count`, colored/marked by `model_exact_match`;
- scatter or box plot of `mean_cosine_drop` vs `gold_count`;
- success vs failure distribution plot for `dist_sensitivity`;
- optional per-removal plot of removal distance vs removed occurrence index.

These plots should be summaries, not one figure per example by default. If example-level visualizations are later needed, add a separate cap such as `MAX_SENSITIVITY_EXAMPLE_PLOTS`.

## Notebook integration

Add the sensitivity block after tokenization/span location and after the model is loaded, but keep it separate from existing ridge/counterfactual/steering blocks. It should be possible to run:

```python
RUN_COUNTING_FEATURE_CALC = False
RUN_NEEDLE_SENSITIVITY = True
```

as long as the notebook has the dataset, tokenized examples, model, tokenizer, and selected layers available.

The block should print:

- number of candidate examples;
- number included/excluded;
- distribution of exclusion reasons;
- number of perturbed sequences evaluated;
- output paths;
- memory/timing summary.

It should also write timing entries through the existing timing-log mechanism.

## Interpretation

Possible outcomes:

- High final-token sensitivity, especially in successful examples, suggests the final-token representation integrates individual needles.
- Low final-token sensitivity despite good answer accuracy suggests either robust distributed integration, insensitivity at the measured layer, or that removal of one needle does not strongly perturb the normalized final state.
- Higher sensitivity in failed examples could mean representations are unstable/noisy.
- Lower sensitivity in failed examples could mean the model did not integrate needles into the final-token state.
- Sensitivity increasing with gold count might indicate accumulated evidence load; sensitivity decreasing with count might indicate individual needles become diluted in long/high-count prompts.

This analysis should not be interpreted as causal proof of counting by itself. Replacing a needle with haystack tokens changes both semantic content and local token identity, while preserving length. The result measures hidden-state sensitivity to a controlled content replacement, not an isolated abstract count variable.

## Validation

Add CPU-compatible tests with a tiny fake model or deterministic hidden-state stub where possible:

- examples with fewer than `NUM_REMOVAL` matching spans are excluded with the expected reason;
- replacement spans have exactly the same token length as removed spans;
- replacement spans do not overlap any needle span;
- perturbed sequences have the same length as originals;
- target position for `"last-token"` equals `sequence_length - 1`;
- normalized hidden states have unit norm within tolerance when norm is nonzero;
- summary metrics match hand-computed values for small synthetic hidden states;
- output CSVs and tensor files are written with expected columns/keys;
- notebook source contains the new config variables and block without modifying the `drive.mount()` cell.

## Clarification questions before coding

1. Should `RUN_NEEDLE_SENSITIVITY` analyze all otherwise valid examples by default, or only the same filtered examples selected by `FILTER_EXAMPLE`?
2. For replacement sampling, is token-level replacement acceptable, or do you specifically want text-level replacement followed by retokenization even though that may change sequence length?
3. Should replacement segments be sampled only from the original haystack/context body if available, or is any non-needle prompt token acceptable for the first implementation?
4. Do you want sensitivity computed for failed examples automatically when `FILTER_EXAMPLE=True`, similar to the extra probe evaluation setup?
5. Should we save mean sensitivity vectors for possible later steering/comparison, or only scalar sensitivity metrics for diagnostics?
