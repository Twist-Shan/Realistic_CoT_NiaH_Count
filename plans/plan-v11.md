# CoT for NIAH tasks

### General instructions
Principles: Read the plan carefully.
- You are encouraged to raise clarification questions about typos, ambiguities, uncertainties, etc. Ask me if you need more information for making decisions.
- Ask to correct obvious typos (with minimal changes) in this document before you code.
- Remember to update the `README.md` after coding
- If I ask you to include the Q&A interaction, summarize our interaction and add your summary under a subsection titled "#### Clarification Q&A (YYYY-MM-DD-time)" right below the requested section
  - When responding to your questions, I will use bullet points and mostly likely use [My Answer] to indicate that it is my answer to your question.
- Don't modify the code block in the notebook that contains the `drive.mount()` command, unless I explicitly tell you to do so.
- You should read the section `### My initial draft` and then expand on my ideas in new sections. Don't modify the text in `### My initial draft` and above it.

### My initial draft

The main idea is to lay out a plan for implementing CoT analysis. Previously, the experiments focused mostly on non-CoT (or nonthinking) mode for NIAH tasks. This plan is to expand analyses to cover CoT.
- In a new section `## Summary of implemented plans`, summarize the previous relevant plans and experiments. It is useful to recap QK-cache, the analysis of outlier tokens (attention sinks, massive activation tokens), and attention analysis. Focus less on linear probes and causal/intervention analysis.
- Create a new notebook `notebooks/cot_analysis.ipynb`. This new notebook should inherit and slightly modify existing code blocks 1--5, 10, and 11 from `counting_feature_analysis.ipynb`. You shouldn't change the first code block where I mount the Googld drive. You should consider simplify the second code block `Global configuration` because I don't need many config params, since the focus of this plan is different from the existing notebook. NOTE: Ask clarification questions if you don't know which config params to include in the second code block.
- Generation of the NIAH dataset should follow the most recent implementations, which support multiple tasks. I think you can reuse most existing code.
- You should also reuse most code for response generation and metrics. Add a new accepted value `USE_THINKING = None` in the notebook (is it a good practice?), so that the model will implement both modes. In other words, if `USE_THINKING = None`, the notebook will first generate responses and scoring metrics under `USE_THINKING = False` and then do the same under `USE_THINKING = True` separately. 
  - Check the max_new_tokens variables carefully in the pipeline. Explicitly introduce at least two max_new_tokens for both thinking mode and nonthinking mode as overriding variables in the notebook. 
  - Also, introduce the temperature config param as an overriding param in the notebook. The user can then easily adjust the values that control model generation. 
  - Check that `LAYERS` is in the config param, and as before, it is a global variable in the notebook. 
  - As before, save the generated model outputs and scoring metrics under the folder `data/niah-example`.
  - The I/O logic and zipping procedure should be similar to `counting_feature_analysis.ipynb`.
- Identify outlier tokens. Once the notebook runs initial response generation and metrics (or finds existing files), we now determine the outlier tokens. Here I define outlier tokens as attention sink tokens and massive activation tokens (please check earlier plans and implementation; raise questions if you have any). The goal is to find, for each layer in `LAYERS`, the outlier tokens from the entire sequence, including both the prompt and model's generation. This is different from earlier implementation where only the prompt is used. 
  - Check existing code carefully and decide what scripts can be reused or need modification. I believe that hidden states calculation and Q/K-cache are necessary for finding attention sinks in the long context.
  - Keep track of the runnig time and CPU & GPU memory use for this part. Be sure to free up memory for unused variable. It's important to avoid OOM issue.
  - Introduce a global variable `K` in the notebook. As before, `K` will be the most salient outlier tokens after we rank the tokens by the scores. Check and use existing logic for deciding top-K attention sinks and massive activation tokens.
- Attention analysis. Define a pattern as a needle span (as before, this means the the consecutive token positions for a needle), or one of the K outlier tokens; check and compare existing calculation. For a given layer in `LAYERS` and each pattern, calculate the average attention that flow into at later positions. You may use the earlier attention ratio metric relative to the uniform baseline.
- Making plots. For each layer and each pattern, generate a 1-by-2 subplots. The left subplot shows the attention ratio metric at prompt token positions. The right subplot shows the attention ratio metric at model generation token positions. As before, overlay the left subplot with shaded regions that show needle spans. 
  - I can't remember the exact scripts but in earlier implementations, the pipeline generated files `figures/inputs_*.png` which have the shaded regions.
  - You may also use the style and legends of the plots `figures/inputs_*.png` in earlier implementations.

#### Revision 1

- Plotting. I also want to draw vertical dashed lines to indicate the positions of special tokens, such as thinking tokens, and final_answer tokens. It may be partially implemented from earlier code. You can revisit earlier implementation and see what can be reused.

#### Revision 2

- Additional patterns. I want to include two patterns, besides outlier tokens and needle spans. The first pattern is very straightfoward: simply the entire prompt span. In other words, this first pattern includes all tokens from the first prompt token to the last prompt token. The second pattern is similar to the first pattern, but excludes the first token (since the first token is similar to a BOS token which has no good interpretations). You need to run similar calculation, analyses, and plots to study the two additional patterns.
- Reduce outlier-token patterns. Currently, for each attention sink and massive activation token, there is a pattern, and a plot is generated. I want to introduce a global variable in the notebook `outlier_ratio_threshold = 5` (capitalize it) such that tokens with top-K scores but `ratio` being less than `outlier_ratio_threshold` are not considered outlier tokens. This will remove weak outliers from the list of patterns, and significantly reduce the number of plots being generated.
- Revise the title in the generated plots. For each outlier-token, the current title writes `pattern=massive_activation_rank_9`, which is uninformative. Change it to `pattern=<token>` where `<token>` shows the token in text. This will tell readers what outlier token actually is.
- I plan to set `NUM_MAX_NEEDLES = 10` and expect the generated dataset to contain random number of needles up to 10. When selecting `MAX_ANALYSIS_EXAMPLES` number of examples for analyses, you need to balance the successful / failed examples, and also low / medium / high number of needles. Of course, you probably need rounding or sometimes failed examples may not exist, but at least you need to try to do that whenever possible, and print a message after your selection of the examples for analyses.


#### Revision 3

- Revise needle insertion. The current needle insertion approach is too permissive, even for `SENTENCE_LEVEL_INSERTION = True`. It assumes that `.`, `?`, `!` are all valid delimiters. However, this can create unnatural tokenization artifacts, because urls or abbreviates may contain periods, but they are not actual periods.
  - You should review the haystack texts and identify cases where `.` does not represent the usual period (end of a sentence).
  - Then, write functions to filter out invalid delimiters.
  - Keep other parts of the dataset creation logic the same under `SENTENCE_LEVEL_INSERTION = True`. I expect that needle insertion to be more conservative when selecting candidate delimiters for insertion.

## Summary of implemented plans

Previous plans established a reproducible NIAH workflow centered on `notebooks/counting_feature_analysis.ipynb`, with reusable logic moved into `src/` and notebook cells acting mostly as orchestration. The current pipeline can generate dynamic NIAH datasets for multiple task types, generate model responses, score those responses, cache/reuse existing artifacts, zip successful response-generation outputs, and save resolved run configs next to outputs.

The implemented counting-feature work added variable per-example needle counts through `NUM_MAX_NEEDLES`, optional example filtering through `FILTER_EXAMPLE`, multiple ridge-probe modes, baseline probes from non-needle positions, unified probe diagnostics, cosine-similarity analysis between probe vectors, and sequence-projection plots. These are useful background but are not the main focus of this CoT plan.

The more relevant earlier work for this plan is the Q/K-cache, outlier-token, and attention-analysis pipeline. Earlier scripts already contain logic for capturing Q/K states, ranking attention-sink tokens, identifying massive-activation tokens from hidden states, overlaying needle spans on input-position plots, and computing attention ratios relative to a uniform causal-attention baseline. The new CoT analysis should reuse these ideas, but update the token scope from prompt-only to prompt plus generated tokens.

## Scientific goal

The new experiment asks whether CoT / thinking mode changes where the model routes attention and where outlier representations appear during NIAH tasks. The primary comparison is between non-thinking generation and thinking generation on the same NIAH examples, with separate response files, metrics, outlier-token tables, and attention plots for each mode.

The core objects of analysis are:

- Needle spans in the prompt.
- Top-`K` attention-sink tokens in the full prompt-plus-generation sequence.
- Top-`K` massive-activation tokens in the full prompt-plus-generation sequence.
- Attention flowing into each pattern from later prompt positions and from generated-token positions.

The default task for initial runs should be `match_count`, but the implementation should support all currently supported NIAH task types when their prompt generation, response parsing, and metrics are compatible. If any task has incompatible response-format or span-metadata logic, raise that as a question before coding.

The main outputs should let us compare:

- Accuracy and predicted-count behavior under thinking vs non-thinking.
- Whether outlier tokens concentrate in the prompt, the generated reasoning, or the final answer.
- Whether generated CoT tokens attend to needles, attention sinks, or massive-activation tokens differently from prompt tokens.
- Whether attention-to-needle patterns differ between successful and failed examples.

## Proposed notebook structure

Create `notebooks/cot_analysis.ipynb` as a thin orchestration notebook. It should borrow the setup, data-generation, response-generation, metrics, logging, and zipping structure from `counting_feature_analysis.ipynb`, but omit counting-feature probe sections unless they are needed only as helper utilities.

The notebook should have these stages:

1. Environment setup and Drive mount.
   - Keep the existing first Drive-mount block unchanged.

2. Global configuration.
   - Use a smaller config surface than `counting_feature_analysis.ipynb`.
   - Keep only task-generation, model-generation, CoT-analysis, plotting, and I/O controls.

3. Resolve run config and paths.
   - Build one run root.
   - Build mode-specific subdirectories for non-thinking and thinking outputs.
   - Save the resolved config.

4. Dataset generation or reuse.
   - Use the latest dynamic NIAH generation logic.
   - Dataset generation should be mode-independent: thinking and non-thinking must use identical examples so the two modes are directly comparable.

5. Response generation and scoring.
   - Use the notebook global `THINKING_MODES`, for example `["nonthinking"]`, `["thinking"]`, or `["nonthinking", "thinking"]`.
   - Preserve script-level `USE_THINKING` behavior for backward compatibility, but hide that detail inside wrapper code called by the notebook.
   - Save predictions and metrics separately by mode to avoid overwriting.
   - Save response and metric caches both under the run folder and under `data/niah-example` once generation/scoring succeeds.

6. Full-sequence token assembly.
   - For each scored example, reconstruct token ids for the full sequence:
     `prompt tokens + generated response tokens`.
   - Save prompt length, generation length, prompt/generation boundary, and needle spans in full-sequence coordinates.
   - Detect generated-output boundaries when available, including thinking delimiters and final-answer starts, so plots can draw vertical dashed lines at these locations.

7. Outlier-token analysis.
   - For each mode, selected example, and layer in `LAYERS`, identify attention sinks and massive-activation tokens over the full sequence.
   - Track token text, token index, segment (`prompt` or `generation`), score, rank, whether the token overlaps a needle span, and whether it is a special token.

8. Attention-pattern analysis.
   - Define patterns as needle spans, top-`K` attention sinks, and top-`K` massive-activation tokens.
   - For each layer and pattern, compute attention mass from later positions into the pattern.
   - Report the attention-ratio metric relative to a uniform causal baseline.

9. Plotting.
   - For each mode, example, layer, and pattern, create a 1-by-2 plot:
     left subplot for prompt query positions, right subplot for generated-token query positions.
   - Shade needle spans in the prompt subplot.
   - Draw vertical dashed lines for detected special positions, including thinking markers and final-answer tokens when the output format allows detection.
   - Use existing `figures/inputs_*.png` style where practical.

10. Zip and summarize.
   - Zip mode-specific predictions, metrics, outlier tables, attention summaries, and figures after successful completion.
   - Save timing and CPU/GPU memory logs for each major stage.

## Proposed global variables

The notebook should list these globals in the configuration block, with short inline comments. Defaults should preserve a small, reproducible, easy-to-debug run.

- `MODEL_NAME`: Hugging Face model id or local model path used for generation and analysis.
- `TOKENIZER_NAME`: tokenizer id/path; defaults to `MODEL_NAME` unless explicitly set.
- `TASK_TYPE`: NIAH task type, for example `match_count`, `literal_count`, or other currently supported task values.
- `PROMPT_STYLE`: prompt template style, preserving the existing `vanilla` default unless overridden.
- `COUNTING_NEEDLE_KIND`: controls the kind of needle generated for counting tasks, such as `city_score` or `marker`.
- `MARKER_TEXT`: literal marker text used only when the selected task/needle kind requires it.
- `UID_TOKEN_LENGTH`: target token length for literal UID markers when literal-marker generation is active.
- `FACT_TEMPLATES_PATH`: path to the NIAH fact template file.
- `NUM_EXAMPLES`: number of examples to generate or load.
- `TARGET_HAYSTACK_TOKENS`: target prompt haystack length.
- `NUM_NEEDLES`: fixed needle count when `NUM_MAX_NEEDLES is None`.
- `NUM_MAX_NEEDLES`: if positive, sample each example's needle count uniformly from `1..NUM_MAX_NEEDLES`; if `None`, use `NUM_NEEDLES`.
- `INSERTION_POSITIONS`: requested insertion-position schedule passed to the data generator.
- `RANDOMIZE_NEEDLE_INSERTION`: whether to randomize insertion locations.
- `RANDOMIZE_NEEDLE_SEED`: seed for randomized insertion locations.
- `WORD_LEVEL_INSERTION`: whether to use word-boundary insertion.
- `GLOBAL_RANDOM_SEED`: global seed recorded in the run config.
- `HAYSTACK_SEED`: seed for haystack generation.
- `NEEDLE_SEED`: seed for needle content generation.
- `THINKING_MODES`: list of modes to run in the notebook; accepted values inside the list are `"nonthinking"` and `"thinking"`. Examples: `["nonthinking"]`, `["thinking"]`, or `["nonthinking", "thinking"]`. This replaces `USE_THINKING` in the notebook for readability while scripts may still use `USE_THINKING` internally for backward compatibility.
- `MAX_NEW_TOKENS_NONTHINKING`: generation budget for non-thinking mode.
- `MAX_NEW_TOKENS_THINKING`: generation budget for thinking mode, usually larger.
- `GENERATION_TEMPERATURE`: generation temperature for both modes unless mode-specific overrides are later added; default should be deterministic.
- `GENERATION_DO_SAMPLE`: whether generation samples; default should stay deterministic unless the experiment requires sampling.
- `GENERATION_TOP_P`: optional nucleus-sampling parameter, used only when sampling is enabled.
- `USE_KV_CACHE_FOR_NONTHINKING`: whether non-thinking generation uses `use_cache`; default should preserve current behavior.
- `USE_KV_CACHE_FOR_THINKING`: whether thinking generation uses `use_cache`; default should preserve current behavior.
- `LAYERS`: list of model layers to analyze for hidden states, Q/K cache, outlier tokens, and attention patterns.
- `K`: number of top attention-sink tokens and top massive-activation tokens to keep per layer, example, and mode.
- `MAX_ANALYSIS_EXAMPLES`: cap on examples used for expensive outlier and attention analysis; default `10`. Response generation and metrics can still run on all examples.
- `ANALYSIS_EXAMPLE_SELECTION`: how to choose examples for expensive analysis, for example `first`, `random`, or `balanced_success_failure`; default should prefer a mix of successful and failed examples when available.
- `RUN_DATASET_GENERATION`: whether to generate/reuse the NIAH dataset stage.
- `RUN_RESPONSE_GENERATION`: whether to generate/reuse model responses and metrics.
- `RUN_OUTLIER_ANALYSIS`: whether to run full-sequence attention-sink and massive-activation analysis.
- `RUN_ATTENTION_ANALYSIS`: whether to compute attention-flow metrics into patterns.
- `RUN_PLOTTING`: whether to generate diagnostic plots.
- `OUTLIER_TOKEN_SCOPE`: default `prompt_and_generation`; included to make the full-sequence scope explicit in saved configs.
- `ATTENTION_HEAD_AGG`: how to aggregate heads for summary plots. For consistency with earlier Q/K outlier figures, default should be `max_across_heads` for received-attention-ratio plots, while per-head tables are still saved.
- `SAVE_PER_HEAD_ATTENTION_TABLES`: whether to save per-head attention-ratio tables; default `True`.
- `SAVE_QK_CACHE`: whether to save Q/K cache tensors to disk; default `False`. Q/K-derived statistics should be computed and retained, but raw Q/K caches should not be included in normal outputs or zipped archives.
- `SAVE_FULL_HIDDEN_STATES`: whether to save full hidden-state tensors. Default should be `False`; save derived tables unless debugging requires tensors.
- `ANALYSIS_DTYPE`: dtype for analysis forward passes, matching the model dtype unless fp32 reductions are needed.
- `PLOT_SPECIAL_TOKEN_LINES`: whether to draw vertical dashed lines for detected generated-output markers; default `True`.
- `THINKING_MARKER_STRINGS`: marker strings used to detect thinking boundaries when present, for example model-specific thinking delimiters.
- `FINAL_ANSWER_MARKER_STRINGS`: marker strings or prefixes used to detect final-answer starts when present.
- `RUN_ROOT`: run output directory.
- `DATA_CACHE_ROOT`: folder for reusable dataset/response cache, expected to include `data/niah-example`.
- `FORCE_REGENERATE_DATASET`: force dataset regeneration even if matching cached files exist.
- `FORCE_REGENERATE_RESPONSES`: force response regeneration even if matching mode-specific predictions and metrics exist in both the run folder and `data/niah-example`.
- `ZIP_RESULTS_AFTER_RESPONSE_STAGE`: zip response-generation and scoring outputs immediately after successful scoring.
- `ZIP_RESULTS_AFTER_ANALYSIS_STAGE`: zip outlier/attention/plot outputs after successful analysis.
- `LOG_GPU_MEMORY`: whether to record GPU memory snapshots around major stages.
- `LOG_CPU_MEMORY`: whether to record CPU memory snapshots around major stages.

Implementation note: keep the existing lower-level `USE_THINKING` path for scripts that already expect booleans. The new notebook should normalize `THINKING_MODES` into per-mode script calls, mapping `"nonthinking"` to `USE_THINKING=False` and `"thinking"` to `USE_THINKING=True`.

## I/O plan

The dataset should be cached once per data-generation config. Thinking and non-thinking modes should always share the same dataset rows. If any existing generator flag would change the dataset content for thinking mode, the notebook wrapper should avoid using that flag during dataset generation and apply thinking-mode differences only during response generation.

Recommended output layout:

```text
RUN_ROOT/
  config.json
  logs/
    timing_summary.csv
    memory_summary.csv
  generate_data/
    dynamic_niah_v2.jsonl
  responses/
    nonthinking/
      predictions.jsonl
      metrics.json
      scoring_summary.csv
    thinking/
      predictions.jsonl
      metrics.json
      scoring_summary.csv
  tables/
    cot_outliers/
      nonthinking/
      thinking/
    cot_attention/
      nonthinking/
      thinking/
  figures/
    cot_attention/
      nonthinking/
      thinking/
  archives/
```

Mode-specific directories are important because running both modes otherwise risks overwriting predictions, metrics, and figures from the first mode.

Response and metric caches should also be copied or written under `data/niah-example` after successful generation/scoring. This protects long response-generation work if the notebook later crashes during outlier or attention analysis. The cache key must include generation-relevant settings, including mode, model, tokenizer, prompt style, task type, dataset config, max-new-token budget, temperature, and sampling controls.

The notebook should save enough metadata to reconstruct full-sequence coordinates:

- example id,
- prompt token count,
- generated token count,
- full sequence token count,
- needle spans in prompt coordinates,
- needle spans in full-sequence coordinates,
- response mode,
- correctness / parsed prediction from metrics,
- detected thinking-boundary positions,
- detected final-answer-token positions when available.

## Outlier-token analysis details

Attention sinks and massive-activation tokens should be computed on the full prompt-plus-generation sequence. The previous implementation should be reused where possible, but it likely needs a wrapper or extension because earlier logic focused on prompt/input tokens.

For each mode, example, and layer:

1. Build full-sequence token ids.
2. Run the model in evaluation mode with no gradients.
3. Compute massive-activation scores from hidden states at that layer.
4. Compute attention-sink scores using Q/K-derived attention received from later positions.
5. Rank tokens by each score and keep top `K`.
6. Save compact tables rather than full tensors by default.

Top-`K` outlier selection should be per example and per layer, because outlier tokens vary across examples. After per-example tables are saved, also save aggregate summaries across the selected examples, such as how often top outliers fall in prompt vs generation, special-token vs ordinary-token rates, and overlap with needle spans.

Suggested saved columns:

- `mode`,
- `example_id`,
- `layer`,
- `outlier_type`,
- `rank`,
- `token_index`,
- `token_text`,
- `segment`,
- `score`,
- `in_needle_span`,
- `needle_index`,
- `is_special_token`,
- `is_generation_token`,
- `gold_count`,
- `model_prediction`,
- `is_correct`,
- `thinking_boundary_type`,
- `final_answer_boundary_type`.

For memory safety, process examples one at a time and free tensors after each example/layer. Avoid saving full hidden states, Q/K tensors, or full attention matrices unless the corresponding debug flags are enabled. The default plan is to avoid saving raw Q/K caches to disk; implementation may compute Q/K-derived statistics in memory or use temporary files that are deleted before zipping.

## Attention-pattern analysis details

Define each pattern as a set of key-token positions:

- `needle_span_i`: all token positions belonging to one inserted needle.
- `attention_sink_rank_j`: one top-ranked attention-sink token.
- `massive_activation_rank_j`: one top-ranked massive-activation token.

For each query position later than a pattern, calculate attention mass into the pattern:

```text
attention_mass(t, pattern) = sum attention[t, key_position] over key_position in pattern
```

Then normalize against a uniform causal baseline:

```text
uniform_baseline(t, pattern) = pattern_length / number_of_available_key_positions_at_t
attention_ratio(t, pattern) = attention_mass(t, pattern) / uniform_baseline(t, pattern)
```

Save per-head per-position attention ratios so the analysis can inspect head-level variation. For plots, do not generate per-head figures by default. Earlier Q/K outlier figures saved per-head attention-stat tensors and plotted the maximum `received_uniform_ratio` across heads for each token position. The new CoT attention plots should follow that convention initially: keep per-head tables, but plot `max_across_heads` unless a different aggregation is requested later.

The prompt/generation split matters:

- Prompt subplot: query positions inside the prompt.
- Generation subplot: query positions inside the model output.
- Needle shading appears only on the prompt subplot because needles are inserted in the prompt.
- Generated-token x-axis should be relative to generation start, while saved tables should preserve full-sequence indices.
- In thinking mode, if the output format exposes thinking delimiters and/or a final-answer marker, those positions should be saved in the table and drawn as vertical dashed lines.

## Plotting plan

For each selected example, mode, layer, and pattern type, create 1-by-2 plots:

- Left: attention ratio over prompt query positions.
- Right: attention ratio over generated query positions.
- Needle spans shaded on the prompt side.
- Pattern token/span highlighted if it lies in the plotted region.
- Vertical dashed lines for detected thinking markers and final-answer starts.
- Title includes mode, layer, example id, gold answer, model prediction, correctness, and pattern name.

Recommended figure folders:

```text
figures/cot_attention/{mode}/layer_{layer}/example_{example_id}/
```

To avoid too many figures, the notebook should use `MAX_ANALYSIS_EXAMPLES = 10` by default and support `ANALYSIS_EXAMPLE_SELECTION`. A useful default is to choose a small set containing both successful and failed examples when available.

## Runtime and memory logging

Every major stage should be wrapped in timing and memory logging:

- dataset generation/reuse,
- response generation,
- response scoring,
- token assembly,
- outlier analysis,
- attention analysis,
- plotting,
- zipping.

For GPU memory, record at least:

- allocated memory,
- reserved memory,
- max allocated memory since last reset,
- device name.

For CPU memory, record process RSS if `psutil` is available. Missing optional logging dependencies should not silently fail; the notebook should print a clear warning and continue with GPU/time logs.

The analysis code should call `torch.cuda.empty_cache()` after large temporary tensors are deleted. For long-context experiments, the default should be to save derived CSV/JSON outputs rather than `.pt` tensors.

## Validation plan

Before large GPU runs, add or run small CPU-compatible checks where possible:

- Config normalization for `THINKING_MODES`, including `["nonthinking"]`, `["thinking"]`, and `["nonthinking", "thinking"]`.
- Path construction for mode-specific response outputs.
- Response cache lookup under both the run folder and `data/niah-example`.
- Full-sequence coordinate construction from prompt and generated tokens.
- Thinking/final-answer boundary detection on small synthetic strings.
- Attention-ratio calculation on a tiny synthetic attention matrix.
- Per-head attention-table serialization and max-across-head plotting aggregation.
- Plot generation from saved CSV tables without model execution.
- Zip creation after response-generation stage.

GPU validation should be treated separately:

- one tiny model/example smoke run,
- one short-context Qwen run with `MAX_ANALYSIS_EXAMPLES = 1`,
- then the intended long-context run.

## Clarification decisions incorporated

- Use `THINKING_MODES` in the notebook for readability, while preserving script-level `USE_THINKING` for backward compatibility.
- Always compare thinking and non-thinking on identical dataset rows.
- Default task is `match_count`, but implementation should aim to support all current NIAH task types and raise questions if a task-specific incompatibility appears.
- Use `MAX_ANALYSIS_EXAMPLES = 10` by default for expensive outlier and attention analysis.
- Select top-`K` outliers per example and per layer, then add aggregate summaries.
- Save per-head attention tables but not per-head plots.
- For plotting, follow earlier Q/K outlier behavior by plotting max received-attention ratio across heads unless changed later.
- Do not save raw Q/K caches to disk in normal outputs; keep zipped artifacts small.
- Draw vertical dashed lines for thinking markers and final-answer tokens when the model output format allows detection.
- Save response and metric caches under both the run folder and `data/niah-example`.
- Use deterministic generation by default.

## Clarification questions

1. Should `USE_THINKING = None` stay as the user-facing notebook value, or would you prefer an explicit global like `THINKING_MODES = ["nonthinking", "thinking"]`?
  - [My Answer]: I think that it is better to use `THINKING_MODES = ["nonthinking", "thinking"]` due to improved readability. (My understanding is that the list format covers 3 scenarios easily). Use `THINKING_MODES` in the new notebook and hide `USE_THINKING` into the scripts (For backward compatibility, I don't think it is good to delete this variable.)
2. Should thinking and non-thinking modes share the exact same generated dataset rows? I recommend sharing the dataset unless the current generator's `--thinking-mode` flag changes prompt content in a way you specifically want to compare.
  - [My Answer]: Yes, always use the same examples. In future analysis, I will need to compare the two modes. 
3. Which NIAH task types should the first implementation support and test? I assume all currently supported task types, but the minimum smoke tests may focus on `match_count`.
  - [My Answer]: the default NIAH task is `match_count`, but you should try all supported task types. If there is a logic issues for some NIAH tasks, raise a question before writing the code.
4. For expensive outlier and attention analysis, should the notebook analyze all examples by default, or should the default be a small selected subset through `MAX_ANALYSIS_EXAMPLES`?
  - [My Answer]: Use `MAX_ANALYSIS_EXAMPLES = 10` by default.
5. Should top-`K` outlier tokens be selected separately for each example and layer, or aggregated globally across examples for each layer? I recommend per-example top-`K` first, with aggregate summaries saved afterward.
  - [My Answer]: outlier tokens often vary across examples, so please implement per-example top-`K` calculation at this moment. Include an aggregate summaries.
6. For attention analysis, is averaging across heads enough for the first version, or do you want optional per-head tables/plots?
  - [My Answer]: Please generate per-head tables, but not per-head plots. Check the approach of aggregating heads in earlier experiments. (If I remember it correctly, you used max-scoring heads for plotting.) Explicitly summarize that info in this plan. I'll check that part later.
7. Do you want Q/K caches saved to disk, or should the default be derived statistics only? Saving Q/K cache is useful for debugging but risky for long-context memory and storage.
  - [My Answer]: Don't save the Q/K caches to disk. I want to keep a minimal zipped file for analysis.
8. In thinking mode, should generation-side plots treat the entire generated text as one segment, or should we try to split reasoning tokens from final-answer tokens if the model output format allows it?
  - [My Answer]: Yes, if the model output format allows, use a vertial dashed line to indicate the final-answer tokens.
9. Should response and metrics caches live only under the run folder, or also under `data/niah-example` for reuse across runs?
  - [My Answer]: It should live under both the run folder and `data/niah-example`. I believe that the pipeline saves the response and metrics caches under `data/niah-example` once the generation and scoring are finished. Then, it proceeds with further analysis and zip the results at the end of the notebook. Saving caches to `data/niah-example` is useful because generation and scoring the NIAH examples may take a lot of time, so it will protect against the notebook crashing in the middle of the analysis.
10. What default generation temperature do you want for thinking mode? I recommend deterministic generation initially so attention/outlier comparisons are easier to audit.
  - [My Answer]: I agree.

## Small draft typos to correct later

The initial draft has a few obvious typos: `Googld`, `runnig`, and `the the`. I left the draft unchanged, following the instruction not to modify that section.


## Additional questions before coding

1. For raw Q/K values, is it acceptable to write temporary Q/K cache files during analysis if they are deleted before zipping and not kept as run artifacts? The existing reusable Q/K code is disk-oriented, so this would reduce implementation risk. If you mean "never write raw Q/K to disk at all", the implementation should use a new streaming/in-memory path.
  - [My Answer]: Yes you can use temporary Q/K cache files. You can write Q/K to disk. Just clean up / remove them before zipping.
2. For final-answer boundaries in thinking mode, should the first version use a simple heuristic such as "final answer starts after the thinking end marker" when the model emits a thinking delimiter? Or do you want task-specific answer-prefix detection as well?
  - [My Answer]: Yes, use a simple heuristic. For now, Qwen models use thinking tokens, so you can check how these tokens are encoded and locate their positions.
3. When selecting `MAX_ANALYSIS_EXAMPLES = 10`, should selection happen independently for each mode, or should the notebook use the same example ids across modes? I recommend the same example ids across modes for cleaner thinking vs non-thinking comparison.
  - [My Answer]: Use the same example ids across modes. If possible, balance the successful / failed examples, and different number of needles.
4. Should the expensive outlier/attention analysis include examples whose generated response cannot be parsed or scored, or should it restrict to examples with valid metric rows? I recommend keeping parse failures eligible and recording `is_correct` / `parse_success` metadata, because failures may be scientifically interesting.
  - [My Answer]: I agree.
5. For supported task types beyond `match_count`, should the first implementation fail fast when required span metadata is missing, or should it skip only the incompatible examples with a warning? I recommend failing fast during initial development so task-specific metadata issues are visible.
  - [My Answer]: I agree.
   

## Concrete plan for Revision 2

### Scientific purpose

Revision 2 should make the CoT attention analysis easier to interpret by adding broad prompt-level reference patterns and reducing weak outlier-token plots. The new outputs should help distinguish two possibilities:

- generated tokens attend broadly to the prompt;
- generated tokens attend only to a few prompt outliers, while most prompt tokens receive little mass.

This should preserve the existing pattern-level attention-ratio calculation:

```text
attention_ratio(t, pattern) = attention_mass(t, pattern) / uniform_baseline(t, pattern)
```

but expand the pattern list and make the outlier-token list stricter.

### New notebook global

Add one global configuration variable to `notebooks/cot_analysis.ipynb` and `COT_ANALYSIS_CONFIG_DEFAULTS`:

- `OUTLIER_RATIO_THRESHOLD = 5.0`: minimum outlier score required for a top-`K` attention-sink or massive-activation token to become an attention pattern and receive a plot. For attention sinks, use `received_uniform_ratio`. For massive activations, use `norm_ratio_to_median`. Keep the top-`K` tables unchanged; this threshold only controls which outlier rows are promoted into attention-pattern plots.

The notebook config block should include a short inline comment explaining that this reduces weak outlier-token patterns and the number of generated figures.

### Pattern definitions

Keep existing patterns:

- `needle_span_i`: all tokens in one inserted needle span.
- `attention_sink_rank_j`: one top-ranked attention-sink token, after threshold filtering.
- `massive_activation_rank_j`: one top-ranked massive-activation token, after threshold filtering.

Add two prompt-level patterns for every selected example/layer:

- `prompt_span`: all prompt token positions from `0` through `prompt_tokens - 1`.
- `prompt_span_no_first`: all prompt token positions from `1` through `prompt_tokens - 1`.

If `prompt_tokens <= 1`, skip `prompt_span_no_first` with a visible warning. These prompt patterns are not outlier patterns and should not be filtered by `OUTLIER_RATIO_THRESHOLD`.

### Outlier filtering details

Modify the current pattern-construction helper rather than the upstream outlier detection. The upstream files should still save the full top-`K` attention-sink and massive-activation tables for auditability.

When constructing attention patterns:

1. Load `tables/attention_sinks_topk.csv` and `tables/massive_tokens_all.csv`.
2. Restrict rows to the current `example_id` and `layer`.
3. Deduplicate by token position, keeping the strongest row for that position.
4. Sort by the existing score field.
5. Keep only rows with score `>= OUTLIER_RATIO_THRESHOLD`.
6. Use the surviving rows as `attention_sink_rank_j` and `massive_activation_rank_j` patterns.

The resulting ranks should be recomputed after filtering, so `attention_sink_rank_1` means the strongest surviving attention-sink pattern, not necessarily the original top-`K` rank 1 if duplicates or filtering intervene. Save the original score and token text in the pattern table.

### Plot titles and filenames

Revise titles for single-token outlier patterns:

- Instead of `pattern=massive_activation_rank_9`, display `pattern=<token text>` when token text is available.
- Also include the stable pattern name in smaller metadata if practical, for example `massive_activation_rank_9`.
- For prompt-level patterns, use `pattern=prompt_span` and `pattern=prompt_span_no_first`.
- For needle patterns, keep the needle-span name and include token span positions.

File names should remain safe and stable:

```text
figures/cot_attention/layer_{layer}/example_{example_id}/{pattern_name}.png
```

Do not use raw token text as the filename because special tokens, whitespace, slashes, or Unicode can make unsafe paths. The CSV should preserve the readable token text.

### Attention tables

Continue writing one pattern table per `(mode, example, layer)`:

```text
tables/cot_attention/patterns/example_{example_id}_layer_{layer}.csv
```

Add or preserve these columns for every row:

- `pattern_name`
- `pattern_display`
- `pattern_type`
- `pattern_rank`
- `pattern_start`
- `pattern_end`
- `pattern_length`
- `pattern_score`
- `pattern_token`
- `position`
- `segment`
- `relative_generation_position`
- `max_attention_mass`
- `uniform_baseline`
- `max_attention_ratio`
- `best_head`
- `head_count`
- `boundary_type`

For `prompt_span` and `prompt_span_no_first`, `pattern_score` and `pattern_token` can be empty. Their baseline will be large because the pattern contains many tokens; this is expected and makes them useful for studying total prompt attention.

### Plot behavior

Keep the existing 1-by-2 plot structure:

- left subplot: prompt query positions;
- right subplot: generated query positions;
- orange/red shaded regions: needle spans in the prompt;
- green highlighted region: the current pattern when it lies in the plotted region;
- dashed vertical lines: prompt/generation boundary, thinking boundary, and final-answer boundary when available.

For `prompt_span`, the green highlighted region covers the full prompt. For `prompt_span_no_first`, it covers prompt positions `1..prompt_tokens - 1`. This will overlap with orange needle shading; use a green alpha low enough that the needle spans remain visible.

The y-axis can stay symlog because outlier ratios can be large. The title should make clear whether the plot shows an outlier token, a needle span, or a prompt-level pattern.

### Example selection

Revise `select_analysis_example_ids` so it explicitly tries to balance:

- success vs failure across available modes;
- low, medium, and high gold counts / needle counts.

Implementation detail:

1. Compute each example's count from `gold_answer["count"]` when available, otherwise from inserted non-control needles.
2. Assign count bins:
   - low: lower third of observed counts;
   - medium: middle third;
   - high: upper third.
   If there are too few unique counts, degrade gracefully to the available bins.
3. Compute success state using mode metrics. A simple first version can treat an example as successful if all requested modes are correct, failed if at least one requested mode is incorrect, and unknown if scoring metadata is missing.
4. Round-robin sample from `(count_bin, success_state)` buckets until `MAX_ANALYSIS_EXAMPLES` is reached.
5. Print a selection summary showing selected ids, gold counts, bins, per-mode correctness, and bucket counts before/after selection.

This keeps the same selected ids across thinking and non-thinking modes while making the selection more informative for `NUM_MAX_NEEDLES = 10`.

### I/O and reuse behavior

The new filtering threshold and prompt-level patterns affect only downstream attention-pattern analysis and plotting. They should not change dataset generation, response generation, response metrics, or response-cache reuse.

Existing response caches should continue to be reused. If full-sequence tensors are missing from an older cache, the current repair path should rebuild them from `model_output_text` and deterministic prompts before hidden-state/QK analysis.

### Validation plan for Revision 2

Add or update CPU-compatible tests for:

- `OUTLIER_RATIO_THRESHOLD` filtering: rows below threshold do not become attention patterns, while prompt-level and needle patterns remain.
- Prompt-level pattern construction: `prompt_span` and `prompt_span_no_first` have the expected positions and lengths.
- Plot title/display behavior: outlier-token plots use readable token text in the title while filenames use safe pattern names.
- Example selection balancing: synthetic examples with several counts and success/failure states produce a mixed selection when enough examples exist, and degrade gracefully when some buckets are empty.
- Existing response-cache reuse remains unaffected by the new analysis-only config.

GPU validation should use a small run first, for example `MAX_ANALYSIS_EXAMPLES = 1`, one layer, and small `K`, before the intended `NUM_MAX_NEEDLES = 10` run.


## Concrete plan for Revision 3

### Scientific purpose

Revision 3 should make sentence-level needle insertion less likely to introduce unnatural text artifacts. The current implementation treats every `.`, `?`, and `!` as a valid insertion delimiter. In practice, this includes periods inside URLs, filenames, domains, abbreviations, and fragments such as `e. g.`. These are not true sentence boundaries and can create odd contexts like:

```text
https://github. In the 2024 city score audit, ... com/...
```

The goal is not to redesign data generation. The goal is to keep the existing `SENTENCE_LEVEL_INSERTION=True` pipeline, but make candidate delimiter selection more conservative before random sampling occurs.

### Current implementation to revise

The main edit point is `src/dataset_generation/dynamic_niah_v2.py`.

Current flow:

1. `_insert_at_sentence_ends(...)` calls `_sentence_end_offsets(base_text)`.
2. `_sentence_end_offsets(text)` returns every regex match of `[.!?]`.
3. `_insert_at_text_offsets(...)` samples from those offsets and inserts each needle.
4. `_format_text_insertion(...)` handles whitespace around the inserted text.
5. `_verified_text_insertion_metadata(...)` verifies character and token spans.

Revision 3 should preserve steps 3--5. Only the sentence delimiter candidate finder should become stricter.

### Proposed delimiter filtering

Replace the permissive `_sentence_end_offsets(text)` implementation with a conservative scanner that returns offsets only after plausible sentence-ending delimiters.

A delimiter should be rejected if any of the following hold:

- It appears inside a URL, domain, email-like string, or path-like filename.
  - Examples to reject: `github.`, `paulgraham.com`, `diff.txt`, `rss.txt`, `http://...`, `https://...`.
- It is part of a common abbreviation or abbreviation-like sequence.
  - Examples to reject: `e. g.`, `i. e.`, `Mr.`, `Dr.`, `Prof.`, `vs.`, `etc.`, `Fig.`, `No.`, `St.`.
- The next non-whitespace character is lowercase, unless the delimiter is `?` or `!` and the surrounding context strongly looks like normal prose.
  - This catches many URL/domain splits and abbreviation fragments.
- The characters immediately around the delimiter look like a decimal number, version number, or enumerated numeric fragment.
  - Examples to reject: `3.14`, `v1.2`, `2024.06`.
- The delimiter is not followed by whitespace, a quote/closing bracket plus whitespace, or the end of the text.
  - This avoids accepting periods embedded inside tokens.

A delimiter should usually be accepted when:

- It is `.`, `?`, or `!`.
- The previous non-whitespace character is not suspicious for URL/path/number contexts.
- The following context looks like a new sentence boundary: whitespace then uppercase, quote, opening bracket, or end of text.

Implementation should favor false negatives over false positives. If a delimiter is ambiguous, skip it. The generator can already sample with replacement if too few candidates remain.

If a haystack chunk has zero conservative sentence-ending delimiters after filtering, do not crash the run. Fall back to word-boundary insertion for that example, print a warning, and add metadata such as `sentence_delimiter_filter_fallback="word_boundary"` plus the fallback candidate count. This fallback is less ideal than true sentence-boundary insertion, but it is preferable to either inserting inside URL/path punctuation or losing the entire experiment.

### Helper functions

Add small private helpers in `dynamic_niah_v2.py`, for example:

- `_looks_like_url_or_path_context(text, delimiter_index) -> bool`
- `_looks_like_abbreviation_context(text, delimiter_index) -> bool`
- `_looks_like_numeric_fragment(text, delimiter_index) -> bool`
- `_is_valid_sentence_end(text, delimiter_index) -> bool`

Keep the helpers simple and auditable. Prefer local-window regex checks around the delimiter rather than a heavy sentence tokenizer dependency. Do not add a new package for sentence segmentation.

The scanner can still be regex-based:

```python
for match in re.finditer(r"[.!?]", text):
    delimiter_index = match.start()
    if _is_valid_sentence_end(text, delimiter_index):
        offsets.append(match.end())
```

### Metadata and logging

Keep existing metadata keys:

- `sentence_delimiter_offset`
- `sentence_delimiter_candidate_count`
- `sentence_delimiter_sampled_with_replacement`
- `text_insertion_candidate_count`

No run-name or cache-key change is strictly required if `SENTENCE_LEVEL_INSERTION=True` semantics are intentionally updated. However, the change affects generated datasets, so existing cached datasets under `data/niah-example` should not be silently reused when evaluating the new behavior. For the first validation run, set `FORCE_REGENERATE_DATASET=True` or remove the old matching cache.

Optionally add one metadata field if it is useful and low-risk:

- `sentence_delimiter_filter_version`: a short string such as `conservative_v1`.

This field would make generated rows easier to audit, but it should not be necessary for downstream analysis.

### Tests

Update `tests/test_dynamic_niah_v2_controls.py` with focused unit tests for `_sentence_end_offsets`.

Test cases should include:

- Normal prose is preserved:
  - `Alpha ends here. Beta starts here! Gamma asks why? Delta continues.`
  - Expected: offsets after all true sentence delimiters.
- URLs/domains/filenames are rejected:
  - `Visit https://github.com/foo/bar.txt for details. Then continue.`
  - Expected: no offsets inside `https://`, `github.com`, or `bar.txt`; offsets only after real sentences.
- Abbreviation-like fragments are rejected:
  - `This is e. g. a fragment. Then a real sentence.`
  - Expected: reject `e.` and `g.` if they split the abbreviation; accept true sentence endings.
- Decimal/version fragments are rejected:
  - `Version 1.2.3 is installed. Next sentence.`
  - Expected: reject numeric-internal periods; accept sentence endings.
- Insertion still produces clean whitespace and verified spans:
  - Use `_insert_at_sentence_ends(...)` or `generate_dynamic_niah_v2(...)` on a short synthetic haystack containing both bad and good delimiters.
  - Assert inserted needles are placed only after valid sentence endings.
  - Assert there is whitespace before and after inserted text.
  - Assert `token_span_verified=True`.

Also update any existing tests that assumed every `.`, `?`, or `!` is accepted. Those tests should assert conservative sentence-boundary behavior instead.

### Validation on real haystack text

Before coding, the inspection of the downloaded run showed the main bad cases:

- URL/domain/file extensions such as `github. com`, `paulgraham. com`, `diff. txt`, `rss. txt`.
- Abbreviation-like fragments such as `e. g.`.

After coding, run a small local analysis over representative haystack text, if available, to print counts such as:

- total raw `[.!?]` delimiters,
- total accepted conservative delimiters,
- number rejected by URL/path rule,
- number rejected by abbreviation rule,
- number rejected by numeric rule,
- number rejected by following-context rule.

This can be a test helper or a temporary debugging print during validation; it does not need to be saved as a run artifact unless we later want a formal delimiter audit table.

### I/O and cache behavior

This change affects only dataset generation under `SENTENCE_LEVEL_INSERTION=True`. It should not affect:

- word-level insertion,
- token-position insertion,
- response parsing/scoring,
- CoT attention analysis,
- plotting,
- response-cache lookup once a dataset has already been generated.

Because the generated text can change, any experiment meant to use the new conservative insertion should regenerate the dataset. Reusing an old dataset would keep the old permissive delimiter choices.

### Analysis-example eligibility after token-span checks

During code block 6 of `notebooks/cot_analysis.ipynb`, the full-sequence construction / hidden-analysis preparation may emit warnings such as:

```text
[hidden-analysis] warning: could not find inserted token sequence for N* in final model input
```

These warnings mean that at least one inserted needle could not be reliably located in the final model input token sequence. Such examples should be excluded from the pool used by `select_analysis_example_ids(...)` / `MAX_ANALYSIS_EXAMPLES`, because downstream shaded regions, needle-span attention metrics, and pattern definitions would be unreliable.

Implementation details:

1. Record a per-example eligibility flag during full-sequence preparation, for example `needle_spans_verified_for_analysis`.
2. If any inserted needle in an example triggers the warning above, set that example's flag to `False`.
3. When selecting examples for expensive outlier/attention/plot analyses, filter out examples with `needle_spans_verified_for_analysis=False`.
4. Print a concise warning listing excluded example ids and the failed needle ids.
5. If fewer than `MAX_ANALYSIS_EXAMPLES` eligible examples remain, continue with the eligible examples and print:

```text
WARNING: only {n_eligible} eligible examples remain after needle-span verification; fewer than MAX_ANALYSIS_EXAMPLES={MAX_ANALYSIS_EXAMPLES} will be analyzed.
```

This filtering should affect only the expensive analysis subset. It should not remove examples from dataset generation, response generation, response metrics, or cached prediction files.

### Validation commands

Run CPU-compatible checks after implementation:

```bash
python3 -m py_compile src/dataset_generation/dynamic_niah_v2.py scripts/generate_dynamic_niah_v2.py
python3 -m pytest tests/test_dynamic_niah_v2_controls.py -q
```

If the full test file is slow, first run only the sentence-insertion tests, then the full file before finishing.
