# Skill: Statistical Analysis and Experimental Diagnosis

## Purpose

Use this skill when analyzing outputs from computational experiments, especially LLM experiments. The goal is not only to compute summaries, but also to diagnose whether figures, tables, metrics, logs, and saved artifacts make sense according to the repo workflow.

This skill applies to:

* inspecting plots and figures,
* checking result tables,
* validating metrics,
* comparing experimental conditions,
* diagnosing suspicious trends,
* checking run logs and configurations,
* identifying possible bugs in data generation, inference, evaluation, aggregation, or plotting.

Prioritize correctness, reproducibility, and scientific interpretation over producing a polished-looking result quickly.

## First step: locate the source of the result

Before interpreting a figure or table, identify how it was produced.

Check:

1. the script or notebook that generated it,
2. the input data or result files,
3. the config file or CLI arguments,
4. the model, dataset, prompt style, seed, and run directory,
5. the aggregation logic,
6. the plotting or table-generation code.

Do not judge a plot only by appearance. Trace it back to the underlying data and code when possible.

## Basic data checks

Before making claims from a table or plot, inspect the underlying data.

Check:

* number of rows and columns,
* column names and types,
* missing values,
* duplicate rows,
* unique values of grouping variables,
* number of examples per condition,
* number of seeds per condition,
* expected value ranges,
* whether denominators are correct,
* whether percentages are represented as `0.42` or `42`,
* whether rows are filtered unintentionally,
* whether failed runs or empty outputs are included or silently dropped.

If the data is aggregated, verify the aggregation key. For LLM experiments, common keys include:

* model,
* dataset,
* prompt style,
* split,
* seed,
* context length,
* needle position,
* layer,
* head,
* token position,
* decoding setting,
* checkpoint,
* intervention type.

## Plausibility checks for LLM experiments

For LLM inference and evaluation results, check whether the result is plausible given the experimental setup.

Look for:

* accuracy values outside `[0, 1]` or percentages outside `[0, 100]`,
* exact 0 or exact 1 accuracy when that is unlikely,
* identical metrics across different seeds, models, layers, or conditions,
* missing baseline rows,
* unexpectedly small sample sizes,
* sudden performance jumps caused by changed filtering,
* train/test leakage,
* evaluation on the wrong split,
* accidentally evaluating only successful generations,
* mismatched prompt/answer boundaries,
* decoded outputs being scored incorrectly,
* tokenizer mismatch,
* left-padding/right-padding bugs,
* incorrect attention masks,
* off-by-one token positions,
* hidden-state layer indexing mistakes,
* dtype or device changes affecting numerical results.

For generation tasks, inspect a small sample of raw model outputs alongside parsed answers and scores. Verify that the parser is neither too strict nor too permissive.

## Diagnosis of figures

When inspecting a figure, check both the visual appearance and the underlying data.

Ask:

1. Does the axis label match the plotted variable?
2. Are units clear?
3. Are conditions ordered correctly?
4. Are colors, legends, and labels consistent?
5. Are there missing conditions?
6. Are error bars or confidence intervals computed over the correct unit?
7. Is the y-axis range appropriate?
8. Are log scales used intentionally?
9. Are repeated runs aggregated correctly?
10. Does the plot match the expected trend from the experimental design?

Common figure-specific issues:

* plotting percentages as fractions or vice versa,
* averaging over the wrong dimension,
* using token index where character index was intended,
* sorting layer labels lexicographically, e.g. `1, 10, 11, 2`,
* mixing model families or checkpoints,
* hiding failed runs by dropping NaNs,
* plotting smoothed curves without showing raw variability,
* using one seed but presenting the result as stable,
* saving an old figure after changing the code.

If a figure looks surprising, do not immediately treat it as a discovery. First search for pipeline, aggregation, or plotting bugs.

## Diagnosis of tables

When inspecting a table, verify:

* each row corresponds to the intended experimental unit,
* metrics are computed from the right subset,
* means, standard errors, and confidence intervals use the correct denominator,
* columns have clear names and units,
* rows are sorted in a meaningful way,
* no important baseline or control condition is missing,
* values are consistent with logs and raw result files.

For comparison tables, check whether differences are practically meaningful, not only numerically nonzero. When appropriate, report uncertainty across seeds, prompts, examples, or runs.

## Statistical interpretation

Separate descriptive summaries from inferential claims.

Acceptable descriptive claims:

* “Model A has higher average accuracy than Model B in this run.”
* “Accuracy decreases as context length increases in this plot.”
* “Layer 18 shows the largest observed intervention effect.”

Be careful with stronger claims:

* “Model A is better.”
* “The representation causes the behavior.”
* “The method generalizes.”
* “The difference is statistically significant.”
* “The trend proves the hypothesis.”

Use causal language only when the experimental design supports it, such as controlled interventions, ablations, or randomized comparisons.

When estimating uncertainty, identify the unit of variation:

* examples,
* prompts,
* seeds,
* model checkpoints,
* datasets,
* layers,
* heads,
* runs.

Do not treat non-independent observations as independent. For example, token positions from the same prompt, layers from the same model run, or multiple metrics from the same generation may be correlated.

## Debugging suspicious results

If results look wrong, diagnose from upstream to downstream:

1. config parsing,
2. data generation,
3. model/tokenizer loading,
4. prompt construction,
5. inference/generation,
6. output parsing,
7. metric computation,
8. aggregation,
9. plotting/table generation,
10. file saving and run-directory selection.

Prefer adding small checks near the point where the bug could occur. Examples:

* print a few prompts and decoded outputs,
* check tensor shapes,
* check token positions,
* check attention-mask shapes,
* check number of examples after filtering,
* save a small debug table,
* assert expected value ranges,
* assert non-empty outputs,
* compare one hand-computed metric against the code output.

Do not fix a suspicious result by changing the plot alone unless the bug is purely visual.

## Required outputs for a diagnosis task

When asked to diagnose results, provide:

1. a short summary of what was inspected,
2. whether the result appears plausible,
3. suspicious patterns or possible bugs,
4. concrete checks performed,
5. recommended next debugging steps,
6. files or functions likely involved,
7. whether any scientific conclusion is currently supported.

If changes are made to the repo, summarize:

* files changed,
* checks or tests run,
* remaining validation needed,
* whether GPU/Colab reruns are required.

## When to modify code

Modify code only after identifying a likely issue or adding useful validation. Prefer small changes such as:

* adding assertions,
* adding a smoke test,
* logging key metadata,
* saving intermediate debug tables,
* fixing aggregation keys,
* correcting axis labels or units,
* making plotting code read from explicit run directories.

Avoid broad rewrites during diagnosis unless explicitly requested.

## Red flags

Treat the following as red flags requiring investigation:

* empty tables or figures with no data,
* NaNs silently dropped,
* unusually perfect performance,
* identical results across different conditions,
* missing baseline or control conditions,
* sudden changes after a refactor,
* output files generated from stale cached results,
* plots that cannot be traced to a config or run directory,
* metrics computed from fewer examples than expected,
* hidden-state or attention results without clear layer/token indexing,
* README claims inconsistent with the actual scripts.

## Final response style

Be explicit and conservative. Say “the result is plausible given these checks” rather than “the result is correct” unless the full pipeline has been validated.

When evidence is incomplete, say what is missing.
