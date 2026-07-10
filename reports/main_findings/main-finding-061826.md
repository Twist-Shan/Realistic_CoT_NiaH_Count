# Main finding 061826: ridge-vector diagnostics for Dynamic NIAH counting

## Paths to key directories

- Local repo: [`../NIAH/`](../NIAH/)
- Run folders: [`../run_results/`](../run_results/)
- Probe implementation plan: [`../NIAH/plans/plan-v9.md`](../NIAH/plans/plan-v9.md)
- Result-gathering instructions: [`../NIAH/skills/gather_run_results.md`](../NIAH/skills/gather_run_results.md)

## Auto summary of experiments

[Auto summary of experiments] The recent Dynamic NIAH runs test whether Qwen3-8B represents the number of matching needles in a linearly decodable way. The newer variable-count setup samples a target count per example with `NUM_MAX_NEEDLES`, inserts that many matching records, extracts hidden states at selected layers, and fits ridge probes from several token/needle pooling sites. The main question is whether the model encodes count information even when its generated answer is wrong, and whether the resulting ridge vectors are meaningful directions rather than artifacts of position or dataset construction.

Key hypotheses:

- Count information may be linearly decodable from hidden states even when final answer accuracy is modest.
- The final prompt token, span-pooled needle states, and occurrence-level states may encode different count-related quantities.
- Fake-position baselines should be weak if the probes are capturing needle/count information rather than generic position.
- Ridge vectors that work as probes should not automatically be treated as validated steering directions.

## Runs reviewed

| Short label | Run | Setup | Main files |
|---|---|---|---|
| 034815 | [`run_20260619_034815`](../run_results/run_20260619_034815_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/) | Qwen3-8B, `match_count`, 100 examples, 1000-token haystack, `NUM_MAX_NEEDLES=6`, no fake-position baseline yet | [`metrics.json`](../run_results/run_20260619_034815_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/metrics.json), probe summaries under [`probe_diagnostics`](../run_results/run_20260619_034815_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/) |
| 043634 | [`run_20260619_043634`](../run_results/run_20260619_043634_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/) | Qwen3-8B, `match_count`, 200 examples, 1000-token haystack, `NUM_MAX_NEEDLES=6`, fake-position baseline enabled | [`metrics.json`](../run_results/run_20260619_043634_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/metrics.json), [`final_token/summary.csv`](../run_results/run_20260619_043634_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/final_token/summary.csv) |
| 064633 | [`run_20260619_064633`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/) | Qwen3-8B, `match_count`, 300 examples, 2000-token haystack, `NUM_MAX_NEEDLES=12`, fake-position baseline and ridge-vector similarity enabled | [`metrics.json`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/metrics.json), [`ridge_vector_similarity`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/ridge_vector_similarity/) |

Note on naming: the earliest diagnostic run used the old names `mean_final_token` and `mean_across_examples`. These correspond to the later `final_token` and occurrence-index-probe logic, although the final canonical name is `occurrence_index_probe`.

## Main findings

**Finding 1, new: behavioral counting accuracy is high for small counts but degrades sharply as the target count grows.**

- In the `NUM_MAX_NEEDLES=6` runs, exact answer accuracy was similar across sample sizes: 78/100 in 034815 and 154/200 in 043634.
  - Evidence: [`034815/tables/metrics.json`](../run_results/run_20260619_034815_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/metrics.json), [`043634/tables/metrics.json`](../run_results/run_20260619_043634_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/metrics.json).
- In 043634, counts 1 and 2 were perfect, count 3 was nearly perfect, and accuracy fell for counts 4-6: 4 was 24/30, 5 was 21/35, and 6 was 11/36.
  - Evidence: [`043634/tables/predictions.jsonl`](../run_results/run_20260619_043634_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/predictions.jsonl).
- In the harder `NUM_MAX_NEEDLES=12` run, exact accuracy fell to 134/300. Counts 1 and 2 were still perfect, but counts 7-12 were poor: 7 was 2/25, 8 was 6/20, 9 was 0/30, 10 was 6/14, 11 was 3/28, and 12 was 1/27.
  - Evidence: [`064633/tables/metrics.json`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/metrics.json), [`064633/tables/predictions.jsonl`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/predictions.jsonl).

**Finding 2, new: ridge probes decode count strongly even when generated answers are much less accurate.**

- In 043634, the `final_token` probe reached test R2 0.985 at layer 16 and 0.983 at layer 24, with rounded-count accuracies 0.98 and 1.00.
  - Evidence: [`043634/final_token/summary.csv`](../run_results/run_20260619_043634_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/final_token/summary.csv).
- In 064633, generation exact-match accuracy was only 0.447, but `final_token` still reached test R2 0.988 at layer 16 and 0.990 at layer 24.
  - Evidence: [`064633/final_token/summary.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/final_token/summary.csv).
- Span-pooled probes were also strong. In 064633, `mean_across_needles_span_mean` reached test R2 0.971 at layer 16 and 0.976 at layer 24; `mean_across_needles_span_last` reached 0.947 and 0.934.
  - Evidence: [`064633/mean_across_needles_span_mean/summary.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/mean_across_needles_span_mean/summary.csv), [`064633/mean_across_needles_span_last/summary.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/mean_across_needles_span_last/summary.csv).

**Finding 3, new: the fake-position baseline is weak, so the high probe R2 is not explained by generic positional information alone.**

- In 043634, fake-position baseline R2 values were negative for `final_token`, `mean_across_needles_span_last`, and `mean_across_needles_span_mean`; for example, `final_token` baseline test R2 was -0.173 at layer 16 and -0.290 at layer 24.
  - Evidence: [`043634/final_token/baseline_ridge_metrics.csv`](../run_results/run_20260619_043634_Qwen3-8B_match_count_vanilla_1000_needles_0_0_0_0_0_0_num_max_needles_6_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/final_token/baseline_ridge_metrics.csv).
- In 064633, fake-position baselines were negative across all four modes. For example, `final_token` baseline R2 was -0.198 at layer 16 and -0.398 at layer 24; `occurrence_index_probe` baseline R2 was -0.304 and -0.262.
  - Evidence: [`064633/final_token/baseline_ridge_metrics.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/final_token/baseline_ridge_metrics.csv), [`064633/occurrence_index_probe/baseline_ridge_metrics.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/occurrence_index_probe/baseline_ridge_metrics.csv).
- The latest run reports no baseline shortage: requested 12 fake positions per example, with minimum distance 5 from needle spans, and all 300 examples contributed baseline rows.
  - Evidence: baseline rows in [`064633` probe summaries](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/).

**Finding 4, new: `final_token` is the strongest total-count readout among the tested modes, while span-mean is stronger than span-last.**

- In 064633, `final_token` had the best test R2 and rounded-count accuracy: layer 24 R2 0.990 and rounded accuracy 0.853.
  - Evidence: [`064633/final_token/summary.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/final_token/summary.csv).
- In both 043634 and 064633, span-mean outperformed span-last. In 064633, span-mean layer 24 R2 was 0.976 with rounded accuracy 0.680, while span-last layer 24 R2 was 0.934 with rounded accuracy 0.387.
  - Evidence: [`064633/mean_across_needles_span_mean/summary.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/mean_across_needles_span_mean/summary.csv), [`064633/mean_across_needles_span_last/summary.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/mean_across_needles_span_last/summary.csv).
- Interpretation: the answer-generating position appears to contain a strong linearly readable total-count signal. The lower rounded accuracies in the harder 12-count run show that high R2 can still allow enough numeric error to miss exact integer counts.

**Finding 5, new: `occurrence_index_probe` asks a different question from the other three modes.**

- `occurrence_index_probe` predicts the ordinal index of each matching occurrence, not the total count for the example. The other three diagnostic modes ask whether a per-example representation predicts the final gold count.
  - Implementation reference: [`plan-v9.md`](../NIAH/plans/plan-v9.md).
- In 064633, `occurrence_index_probe` reached test R2 0.939 at layer 16 and 0.908 at layer 24, showing that occurrence-level states carry a strong ordinal signal.
  - Evidence: [`064633/occurrence_index_probe/summary.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/occurrence_index_probe/summary.csv).
- The saved prototypes should be interpreted as diagnostic averages of occurrence vectors by occurrence index, not as the training rows themselves. They are useful for inspecting geometric trajectories of first, second, third, etc. occurrence states.
  - Evidence: [`064633/occurrence_index_probe/prototype_geometry_layer_16.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/occurrence_index_probe/prototype_geometry_layer_16.csv), [`064633/occurrence_index_probe/prototype_geometry_layer_24.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/occurrence_index_probe/prototype_geometry_layer_24.csv).

**Finding 6, new: the four ridge vectors are substantially different despite all giving strong probe performance.**

- In 064633 layer 16, cosine similarity between `final_token` and the other three ridge vectors was near zero: -0.010 with `occurrence_index_probe`, 0.016 with span-last, and 0.022 with span-mean.
  - Evidence: [`064633/ridge_vector_similarity/layer_16_cosine_similarity.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/ridge_vector_similarity/layer_16_cosine_similarity.csv).
- In layer 24, `final_token` was again nearly orthogonal to the other modes: -0.022 with `occurrence_index_probe`, -0.019 with span-last, and -0.010 with span-mean.
  - Evidence: [`064633/ridge_vector_similarity/layer_24_cosine_similarity.csv`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/tables/counting_features/probe_diagnostics/ridge_vector_similarity/layer_24_cosine_similarity.csv).
- The span modes were only moderately aligned: span-last vs span-mean cosine was 0.380 at layer 16 and 0.261 at layer 24. `occurrence_index_probe` was more aligned with span-last than with final-token, but still far from identical.
  - Evidence: same cosine-similarity files above; heatmaps are saved under [`064633/figures/counting_features/probe_diagnostics/ridge_vector_similarity/`](../run_results/run_20260619_064633_Qwen3-8B_match_count_vanilla_2000_needles_0_0_0_0_0_0_0_0_0_0_0_0_num_max_needles_12_rand_insrt_seed_42_word_insrt_tmpl_niah_fact_single_template/figures/counting_features/probe_diagnostics/ridge_vector_similarity/).
- Interpretation: high linear decodability does not imply a single shared count direction. The model may expose count through different local coordinate systems at occurrence spans, pooled needle states, and the final answer position.

**Finding 7, new: these ridge vectors are strong probes, but they are not yet validated causal steering directions.**

- Older steering summaries already suggested that direct ridge-vector steering was weak or inert in prior runs, even when ridge probes were statistically meaningful.
  - Evidence: prior summary in [`main-findings-061626.md`](main-findings-061626.md).
- The current diagnostic runs improve the probing setup with variable counts, baselines, and vector-similarity analysis, but they do not by themselves show that adding these vectors to model activations causally improves counting.
- Practical interpretation: the latest results justify more careful steering experiments, especially around `final_token` and span-mean vectors, but should not be cited as steering success.

## Objections and limitations

- These runs are all Qwen3-8B `match_count` runs with vanilla prompting and word-level insertion. The findings should not yet be generalized to `literal_count`, `count_avg`, other models, or other prompt styles.
- The first two diagnostic runs used earlier mode names. This report maps `mean_final_token` to `final_token` and treats `mean_across_examples` as the predecessor of `occurrence_index_probe`, but the cleanest canonical evidence is the 064633 run.
- Ridge regression is high-dimensional and regularized. High R2 shows linear decodability under this split, not necessarily a mechanistic scalar counter.
- `occurrence_index_probe` has more rows for smaller occurrence indices because examples with small total counts do not contribute later occurrences. This is scientifically meaningful for occurrence trajectories but should not be mixed with total-count probes.
- The probes were run with `FILTER_EXAMPLE=False`, so they include both correct and incorrect generations. That is appropriate for representation analysis, but it differs from a success-only probe.
- The latest diagnostic runs focus on layers 16 and 24. More layers are needed before making a layer-localization claim.
- The behavioral output is greedy short-form generation with `max_new_tokens=64`; the runs do not test chain-of-thought, tool use, or alternate decoding.

## To-do list

- Repeat the diagnostic suite across more layers, for example 4, 8, 12, 16, 20, 24, and 28.
- Add cross-mode transfer tests: train a probe/vector on one mode and evaluate projection quality on hidden states from another mode.
- Run the same `all_diagnostics` setup for `literal_count` and `count_avg`, with `NUM_MAX_NEEDLES` enabled.
- Compare `FILTER_EXAMPLE=True` vs `FILTER_EXAMPLE=False` to separate representation quality on successful vs failed examples.
- Test multiple random seeds for `NUM_MAX_NEEDLES`, haystack generation, and train/test split.
- Run the fake-position baseline for additional layers and with stricter distance thresholds to stress-test positional confounds.
- Evaluate whether `final_token` and span-mean vectors can causally steer answer accuracy, separately from older direct ridge steering vectors.
- Add model comparisons, starting with another Qwen size or a different model family, to see whether final-token count decodability is model-specific.
