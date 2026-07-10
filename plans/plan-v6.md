# Plan

### General instructions
Principles: Read the plan carefully. 
- You are encouraged to raise clarification questions about typos, ambiguities, uncertainties, etc. Ask me if you need more information for making decisions. 
- Ask to correct obvious typos (with minimal changes) in this document before you code.
- Remember to update the `README.md` after coding
- If I ask you to include the Q&A interaction, summarize our interaction and add your summary under a subsection titled "#### Clarification Q&A (YYYY-MM-DD-time)" right below the requested section

## Improving counting feature calculation

The current counting-feature workflow finds linear directions in hidden-state space that are predictive of the running number of matching needles. The previous steering experiments suggest that this feature is decodable, but may not be a good causal steering direction. This plan explores two alternative ways to calculate feature vectors that may be more aligned with the model's actual counting behavior.

## Summary of counting feature and previous steering attempts

The earlier counting-feature analysis builds on the `match_count` and `literal_count` NIAH tasks. For each example, the workflow identifies matching needle spans, constructs a token-level target count `y_t`, and fits linear probes from hidden states to the running count. The main regression probe learns a vector `u` such that `h_t^T u` approximates the number of matching needles before or around token position `t`, depending on `TARGET_COUNT_TYPE`.

The workflow originally used only successful examples to fit the counting feature, with the intuition that correct runs are more likely to contain the intended counting computation. Probe training uses token subsampling to keep the flattened hidden-state matrix manageable, while preserving needle-span positions when possible. The resulting ridge probes can achieve good predictive performance, indicating that count information is linearly decodable from hidden states.

Previous steering attempts used the ridge counting direction as an intervention vector. The first steering strategy added the direction at the current last-token position during decoding. A later revision added `STEERING_POSITION_MODE`, where `last_token` keeps the original pipeline and `needle_span` applies the same direction to every token inside one actual matching needle span. In the needle-span mode, the intervention is repeated during every generation forward pass.

Pilot results with aggressive coefficients showed that the intervention can change model outputs, but mostly damages originally correct answers and rarely rescues incorrect answers. This suggests that the ridge count vector may be a readout or correlate of counting rather than a writeable causal direction for controlling the model's answer.

## Contrastive success direction

The contrastive success direction aims to isolate what distinguishes successful counting runs from failed runs, rather than what predicts the count value itself. The central idea is to compare the last-token hidden states of successful and unsuccessful examples, and use the raw difference in group means as a new candidate counting-related feature direction.

This should be implemented as a new optional block in `notebooks/counting_feature_analysis.ipynb`, rather than starting a new notebook. The existing notebook already handles dataset generation/loading, response scoring, hidden-state extraction, output paths, config saving, and archiving. The contrastive success direction should therefore reuse the initial pipeline and add another feature-calculation method.

Introduce a new global variable in the notebook:

```python
COUNTING_FEATURE_CALC_METHOD = "ridge"  # "ridge" or "contrastive-success"
```

When `COUNTING_FEATURE_CALC_METHOD = "ridge"`, the notebook should run the existing ridge-probe counting-feature workflow.

When `COUNTING_FEATURE_CALC_METHOD = "contrastive-success"`, the notebook should skip the ridge-probe fitting workflow and calculate one contrastive success direction for each layer in `LAYERS`.

This variable should control which counting-feature calculation method is used. The steering block should then use the feature vectors produced by the selected method, rather than silently falling back to ridge vectors.

The procedure is the following.

1. Use the scored examples from the current run. The examples should come from the entire available pool, not only the earlier steering subset. In the current workflow this pool may contain up to `NUM_EXAMPLES = 500`, but the code should not hard-code 500.
2. Split examples into successful and unsuccessful examples using the existing exact-match score.
3. Select an equal number of successful and unsuccessful examples. Use as many balanced pairs as possible:

```text
n = min(num_successful, num_unsuccessful)
```

Then select `n` successful examples and `n` unsuccessful examples.
4. Print the number of successful and unsuccessful examples available, and the number selected for each group.
5. If either group has fewer than 10 examples, print a visible warning message. Do not silently continue as if the estimate were reliable.
6. For each selected example, use the hidden state at the last prompt token position. Do not use all token positions or needle-span positions for this first version.
7. For each layer `l`, compute the raw mean-difference direction:

```text
v_success = mean(h | correct) - mean(h | incorrect)
```

where `h` is the last-token hidden state at layer `l`.
8. Save both the raw direction and a unit-normalized direction. Also save the raw vector norm, group means, selected example indices, and summary metadata.

The expected saved artifacts should be systematic. A possible structure is:

```text
{run_dir}/tensors/counting_features/contrastive_success/
  contrastive_success_layer_{layer}.pt

{run_dir}/tables/counting_features/contrastive_success/
  contrastive_success_summary.csv
  contrastive_success_metadata.json
```

The `.pt` file for each layer should contain enough information to make later steering unambiguous, for example:

```python
{
    "layer": layer,
    "direction": unit_direction,
    "raw_direction": raw_direction,
    "raw_norm": raw_norm,
    "successful_mean": successful_mean,
    "unsuccessful_mean": unsuccessful_mean,
    "position": "last_token",
    "method": "raw_mean_difference",
}
```

The summary table should report, for each layer, at least the raw vector norm, number of selected successful examples, number of selected unsuccessful examples, and simple projection statistics for both groups. A useful sanity check is whether the projection onto the contrastive direction separates the selected successful and unsuccessful examples.

This direction can later be evaluated by steering. The steering question is whether adding the contrastive success direction improves failed examples without substantially damaging successful examples. This should be treated as a separate steering experiment after the direction calculation is working and saved clearly.

For steering, use the same method selector by default. The existing `RUN_STEERING` variable should continue to control whether any steering experiment is run. When `RUN_STEERING = True`, the notebook should inspect `COUNTING_FEATURE_CALC_METHOD`:
- If `COUNTING_FEATURE_CALC_METHOD = "ridge"`, use the existing ridge counting vectors, preserving the current `STEERING_POSITION_MODE` behavior.
- If `COUNTING_FEATURE_CALC_METHOD = "contrastive-success"`, use the saved contrastive success vectors instead of ridge vectors.

The goal is to avoid confusing a contrastive-feature run with an old ridge-steering run. If the selected method's feature vector files are missing, the notebook should raise a clear error rather than silently switching methods.

Important implementation choices to decide:
- Use the last-token position for contrastive hidden states.
- Use raw mean differences for the first version.
- Use an equal number of successful and unsuccessful examples.
- Take as many balanced examples as possible from the entire scored pool.
- Print a warning if either group has fewer than 10 examples.
- Compute one direction per layer.
- Add `COUNTING_FEATURE_CALC_METHOD` so the notebook can calculate either ridge vectors or contrastive success vectors.
- When steering is enabled, use the vector type produced by `COUNTING_FEATURE_CALC_METHOD`.

### Revision

Add another option for where the contrastive success hidden states are taken from. The motivation is that the final prompt token may already mix many computations, while the last token position inside the last matching needle span may better capture the representation immediately after processing the final relevant evidence.

Introduce a new global variable in the notebook:

```python
FEATURE_CALC_POS = "last"  # "last" or "needle-last"
```

This variable is relevant for feature-calculation methods that need a selected hidden-state position, including `COUNTING_FEATURE_CALC_METHOD = "contrastive-success"` and the counterfactual method described below.

The behavior should be:

1. If `FEATURE_CALC_POS = "last"`, use the existing contrastive-success behavior: for each selected example, take the hidden state at the last prompt token position.
2. If `FEATURE_CALC_POS = "needle-last"`, use the last token position inside the last matching inserted needle span. For example, if the task has three matching inserted needles, locate the third matching needle span, then use the hidden state at `span_end - 1`.
3. For randomized insertion runs, do not assume fixed insertion positions. Use per-row metadata and actual located needle spans.
4. If an example does not have a usable matching inserted needle span, skip that example for the contrastive direction and record the skipped row id and reason in metadata. After skipping, rebalance successful and unsuccessful examples if needed.
5. Print the selected position mode, number of skipped examples, and final number of successful and unsuccessful examples used.
6. Save `FEATURE_CALC_POS` in the `.pt` vector file, summary CSV, and metadata JSON. The `.pt` field previously called `"position"` should be `"last"` or `"needle-last"` rather than only `"last_token"`.

The raw mean-difference calculation should otherwise remain the same:

```text
v_success = mean(h | correct) - mean(h | incorrect)
```

where `h` is taken from the position specified by `FEATURE_CALC_POS`.

## Counterfactual count direction

The counterfactual count direction aims to estimate the hidden-state change caused by removing one matching needle while keeping the rest of the run configuration and random seeds as fixed as possible. The goal is to compare examples with gold count `k` against matched counterfactual examples with gold count `k - 1`, then use the mean hidden-state difference as a candidate counting feature vector.

Extend the method selector:

```python
COUNTING_FEATURE_CALC_METHOD = "ridge"  # "ridge", "contrastive-success", or "counterfactual"
```

When `COUNTING_FEATURE_CALC_METHOD = "counterfactual"`, run the new counterfactual feature-calculation pipeline below. Otherwise, keep the existing ridge or contrastive-success pipeline.

Assumption for the first implementation:
- `INSERTION_POSITIONS` should contain no `None` values. If any value is `None`, print a clear error message and stop. This keeps the original run at count `k` and the counterfactual run at count `k - 1` with an explicit single removed needle.
- The removed needle should not be the last slot in `INSERTION_POSITIONS`. In the first version, set one non-last insertion position to `None` for the counterfactual dataset. This avoids making `FEATURE_CALC_POS = "needle-last"` ambiguous by removing the final matching needle.

Introduce a new global variable in the notebook:

```python
COUNTERFACTUAL_REMOVED_NEEDLE_INDEX = 0  # non-last insertion slot to set to None for counterfactual runs
```

This variable is only used when `COUNTING_FEATURE_CALC_METHOD = "counterfactual"`. If `COUNTING_FEATURE_CALC_METHOD` is `"ridge"` or `"contrastive-success"`, print a warning message that `COUNTERFACTUAL_REMOVED_NEEDLE_INDEX` is ignored.

The procedure is the following.

1. Generate or load the original dataset using the existing notebook pipeline. This dataset uses the current `INSERTION_POSITIONS` or, if `RANDOMIZE_NEEDLE_INSERTION = True`, the existing randomized insertion logic. Since no insertion position is `None`, the original dataset has gold count `k`.
2. Score the original dataset using the existing response-generation/evaluation workflow.
3. Filter to successful original examples. Only successful original examples should be used to estimate `mean(h | gold_answer = k)`, because these are more likely to contain a valid count-`k` computation.
4. For the successful original examples, use `FEATURE_CALC_POS` to choose hidden-state positions:
   - If `FEATURE_CALC_POS = "last"`, use the last prompt token.
   - If `FEATURE_CALC_POS = "needle-last"`, locate the last matching inserted needle span and use `span_end - 1`.
5. For each layer in `LAYERS`, calculate:

```text
mean_k = mean(h | original successful examples, gold_answer = k)
```

6. Generate a counterfactual dataset under the same model, task, prompt style, random seeds, haystack settings, needle settings, and number of examples, but with the insertion slot specified by `COUNTERFACTUAL_REMOVED_NEEDLE_INDEX` set to `None`. For example, if `INSERTION_POSITIONS = [100, 200, 400]` and `COUNTERFACTUAL_REMOVED_NEEDLE_INDEX = 0`, the counterfactual pattern is `[None, 200, 400]`. The removed index must be a valid non-last index.
7. Save the counterfactual dataset under the run directory with clear names and metadata, so it can be audited separately from the original dataset. Save the original dataset only if the existing notebook pipeline has not already saved it.
8. Score the counterfactual dataset as well. The counterfactual gold count should be `k - 1`.
9. Filter to successful counterfactual examples. Only successful counterfactual examples should be used to estimate `mean(h | gold_answer = k - 1)`.
10. For the successful counterfactual examples, use the same `FEATURE_CALC_POS` rule to choose hidden-state positions.
11. For each layer in `LAYERS`, calculate:

```text
mean_k_minus_1 = mean(h | counterfactual successful examples, gold_answer = k - 1)
```

12. Define the counterfactual count direction as the raw mean difference:

```text
v_count = mean_k - mean_k_minus_1
```

Save both the raw direction and a unit-normalized direction, along with raw vector norm, group means, selected original example ids, selected counterfactual example ids, and all relevant metadata. The original and counterfactual means should use all successful examples from their respective datasets independently; do not restrict to examples where both the original and counterfactual versions are successful.

The expected saved artifacts should be systematic. A possible structure is:

```text
{run_dir}/generate_data/
  dynamic_niah_v2.jsonl
  dynamic_niah_v2_counterfactual.jsonl
  config.used.json
  config.counterfactual.used.json

{run_dir}/tensors/counting_features/counterfactual/{FEATURE_CALC_POS}/
  counterfactual_count_layer_{layer}.pt
  hidden_original_layer_{layer}.pt
  hidden_counterfactual_layer_{layer}.pt

{run_dir}/tables/counting_features/counterfactual/{FEATURE_CALC_POS}/
  counterfactual_count_summary.csv
  counterfactual_count_metadata.json
```

The `.pt` file for each layer should contain enough information to make later steering unambiguous, for example:

```python
{
    "layer": layer,
    "direction": unit_direction,
    "raw_direction": raw_direction,
    "raw_norm": raw_norm,
    "mean_k": mean_k,
    "mean_k_minus_1": mean_k_minus_1,
    "position": FEATURE_CALC_POS,
    "method": "counterfactual_count_difference",
    "original_gold_count": k,
    "counterfactual_gold_count": k - 1,
}
```

Important implementation choices to decide:
- Use `COUNTERFACTUAL_REMOVED_NEEDLE_INDEX = 0` by default, with an inline notebook comment explaining that it must be a non-last insertion slot.
- Use all successful original examples for `mean_k` and all successful counterfactual examples for `mean_k_minus_1` independently. Do not require paired success.
- For now, keep ridge and contrastive-success steering on the existing steering pipeline. Do not use the counterfactual idea for those methods.
- Name counterfactual datasets and caches by appending `_counterfactual` to the ordinary dataset/cache names.

## Comparing counting feature vectors

After running several steering experiments, compare the actual counting feature vectors saved by different run folders. The goal is to see whether different feature-calculation choices produce similar directions in hidden-state space, or whether the successful steering behavior depends on substantially different vectors.

Implement this as a simple Python script under `scripts/`, for example:

```text
scripts/analyze_counting_feature_vectors.py
```

The script should be run from the repository root. It should look in the parent folder `NIAH_repo_and_local_runs`, locate run folders that contain saved counting feature vectors, and ignore earlier run folders that do not contain counting-feature experiments.

The discovery logic should be broad enough to find the vector files we have used so far, including:

```text
{run_dir}/tensors/counting_features/counterfactual/{FEATURE_CALC_POS}/counterfactual_count_layer_{layer}.pt
{run_dir}/tensors/counting_features/contrastive_success/{FEATURE_CALC_POS}/contrastive_success_layer_{layer}.pt
{run_dir}/tensors/counting_features/ridge_probe_layer_{layer}.pt
```

For the first implementation, it is acceptable to compare all discovered vectors in a single table. Each vector entry should record enough metadata to identify it unambiguously:

```text
run_name
run_created_time or parsed run timestamp
method
position
layer
norm
```

For counterfactual and contrastive-success vectors, load the saved `.pt` file and use the saved unit `"direction"` field for cosine similarity. Use the saved `"raw_norm"` field for the reported norm when available. If `"raw_norm"` is missing, compute the norm of `"raw_direction"` if present, otherwise compute the norm of the vector being compared.

For ridge vectors, load the probe and convert it to the same hidden-state direction used by steering. If the probe was trained on standardized features, convert the coefficient from standardized space back to hidden-state space by dividing by `feature_scale`, following the existing steering logic. Report the hidden-space norm before unit normalization.

The script should calculate:

1. The norm of each discovered counting feature vector.
2. The pairwise cosine similarity between every pair of discovered vectors.

The output folder should be created under the parent folder:

```text
NIAH_repo_and_local_runs/steering_run_analysis/
```

The script should write one JSON file, not JSONL, for metadata and norms. The run entries should be ordered by run time from latest to earliest. If multiple vectors come from the same run, preserve that run ordering and then order vectors by method, position, and layer. A possible JSON schema is:

```json
{
  "created_at_utc": "...",
  "analysis_root": ".../NIAH_repo_and_local_runs/steering_run_analysis",
  "num_vectors": 0,
  "vectors": [
    {
      "index": 1,
      "run_name": "run_YYYYMMDD_HHMMSS_...",
      "run_timestamp": "YYYYMMDD_HHMMSS",
      "method": "counterfactual",
      "position": "needle-last",
      "layer": 20,
      "vector_path": "...",
      "norm": 0.0
    }
  ]
}
```

The script should also write a heatmap figure for cosine similarity. The x-axis and y-axis labels should be simple integer labels:

```text
1, 2, 3, ...
```

These integer labels must match the `"index"` values and the order of vector entries in the JSON file. Include a colorbar. The plot title can state that the heatmap shows pairwise cosine similarity between saved counting feature vectors.

Expected output files:

```text
NIAH_repo_and_local_runs/steering_run_analysis/
  counting_feature_vector_metadata.json
  counting_feature_vector_cosine_heatmap.png
```

The script should fail loudly if it finds no vectors, if a vector has non-finite values, or if two vectors selected for comparison have different hidden dimensions. It should print a short summary showing how many run folders were scanned, how many vector files were found, and where the JSON and heatmap were saved.

## New test examples and steering eval

The current steering evaluation mostly reuses examples from the feature-calculation run or from a small intervention subset. To better test whether a steering direction is useful, add a separate evaluation block that generates fresh test datasets with different gold counts and measures whether steering moves predictions in the intended direction across these datasets.

This should be implemented by revising reusable code under `src/counting/` where appropriate, and by adding a new optional code block in `notebooks/counting_feature_analysis.ipynb`. The notebook should remain the main experiment driver, but reusable dataset generation, steering, metric calculation, and result serialization should live in Python functions rather than being embedded only in notebook cells.

Introduce the following global variables in the notebook config:

```python
STEERING_TEST_EVAL = True
NUM_MAX_NEEDLES_STEERING_EVAL = 5
NUM_EXAMPLES_STEERING_EVAL = 10
```

If `STEERING_TEST_EVAL = False`, the notebook should skip the new test-evaluation block entirely. If `STEERING_TEST_EVAL = True`, the notebook should run the additional evaluation after the selected counting feature vectors have been calculated and after the normal steering setup is available.

Let:

```text
K = NUM_MAX_NEEDLES_STEERING_EVAL
```

Generate `K` new NIAH evaluation datasets. Dataset `k` should contain `NUM_EXAMPLES_STEERING_EVAL` examples, and each example should contain exactly `k` matching needles in the haystack, for every:

```text
k in {1, 2, ..., K}
```

The dataset-generation logic should follow the same conventions as the existing dataset-generation block in `counting_feature_analysis.ipynb`, including the same task family, prompt construction, haystack settings, needle text, model/tokenizer formatting, and saved metadata where applicable. For these steering-evaluation datasets, always use randomized needle insertion, effectively treating `RANDOMIZE_NEEDLE_INSERTION = True` for this eval block even if the main experiment used another insertion mode.

For each of the `K` test datasets:

1. Generate model responses without steering.
2. Extract predicted counts using the same parsing and scoring logic as the main experiment.
3. Save the no-steering predictions and metrics.

Then run the selected steering pipeline on the same test datasets. The steering vector should come from the current `COUNTING_FEATURE_CALC_METHOD`, and the intervention should respect the active notebook configuration. In particular, the new eval block should not assume a fixed value for `FEATURE_CALC_POS` or `STEERING_POSITION_MODE`. If `STEERING_POSITION_MODE = "last_token"`, the eval will naturally have fewer intervention targets. If `STEERING_POSITION_MODE = "needle_span"`, the eval should produce rows for the configured needle-span interventions. The steering behavior should otherwise reuse the existing steering implementation, including the same vector normalization, coefficient handling, prompt-boundary handling, and needle-span intervention logic.

Save the relevant artifacts in the same result archive as the rest of the counting-feature experiment. At minimum, save:

```text
tables/counting_features/steering_test_eval/
  steering_test_eval_summary.csv
  steering_test_eval_predictions.csv
  steering_test_eval_metadata.json

tensors/counting_features/steering_test_eval/
  copied or referenced steering vectors used for the evaluation, if not already saved elsewhere
```

The notebook should continue to zip all relevant output folders at the end, so these new evaluation files are included in the final `{run_name}.zip`.

Create one key summary CSV for quick inspection. Each row should correspond to one steering setting, including at least:

```text
layer
needle_span or intervention_target
steering_coeff
```

The summary should also include `K` dataset-level columns, one for each evaluation dataset:

```text
Dataset 1 Before/After steering
Dataset 2 Before/After steering
...
Dataset K Before/After steering
```

Add a header row or leading explanation row at the start of the CSV that states the meaning of the compact tuple used in these dataset columns:

```text
(no_steering_acc, steering_acc, no_steering_avg_counts, steering_avg_counts)
```

Each value in this 4-tuple should be an average or aggregate computed across the `NUM_EXAMPLES_STEERING_EVAL` examples in dataset `k`:
- `no_steering_acc`: exact-match accuracy before steering.
- `steering_acc`: exact-match accuracy after steering.
- `no_steering_avg_counts`: average extracted predicted count before steering.
- `steering_avg_counts`: average extracted predicted count after steering.

For example, the column for dataset `k` should contain a tuple such as:

```text
(0.20, 0.50, 2.10, 2.80)
```

where the four values are computed only from the test examples in that dataset. The dataset identity and target gold count should be clear from the column name or accompanying metadata.

The more detailed predictions CSV should keep one row per example and steering setting, so later analysis can compute rescue rates, damage rates, exact-match accuracy, mean predicted count, and count-shift metrics without reparsing the compact summary cells.

Important implementation choices:
- This new eval block should not replace the existing steering evaluation; it is an additional held-out-style check.
- The eval datasets should be newly generated and should always use randomized needle insertion.
- The feature vector calculation method and steering method should remain controlled by the existing variables.
- The output format should stay consistent with the current `results/counting_features/{run_name}/{run_name}.zip` structure.
- The code should fail clearly if steering is requested but the selected feature vectors are missing.
