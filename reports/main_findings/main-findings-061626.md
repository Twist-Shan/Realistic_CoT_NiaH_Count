# Main findings of NIAH

**Paths to key directories**

Assume your current path is the parent directory, which contains the local repo [`NIAH/`](../NIAH/).

- Local repo: [`NIAH/`](../NIAH/)
- Run folders: subfolders named as [`run_results/run_*/`](../run_results/)
- Instructions: [`NIAH/skills/gather_run_results.md`](../NIAH/skills/gather_run_results.md)
- Repo overview and plan: [`NIAH/README.md`](../NIAH/README.md), [`NIAH/plans/plan-v7.md`](../NIAH/plans/plan-v7.md)

## Auto summary of experiments

[Auto summary of experiments] The recent counting-feature experiments use Qwen3-8B on Dynamic NIAH `match_count`, where city-score facts are inserted into Paul Graham haystack text and the model must answer with `{"count": k}`. The latest workflow uses sentence-level insertion, a single uniform city-score template, a `vanilla` prompt, counterfactual feature-vector extraction at the last token of the final needle span, and steering applied inside needle spans. The main experimental question is whether a hidden-state direction estimated from original count-3 examples versus counterfactual count-2 examples can causally increase the model's counted answer on harder held-out count datasets.

Key hypotheses:

- A ridge probe can decode count-related information, but its vector is not a reliable causal steering direction.
- Counting features are more causally useful when extracted at the last token inside the last needle span than at the final prompt token.
- Steering works better when applied inside actual needle spans than only at the final prompt token.
- Counterfactual count directions are more promising than ridge directions, but they need held-out evaluation because small selected steering sets can overstate the effect.
- Dataset/prompt construction is a major confound: sentence-level insertion, uniform fact templates, prompt style, cache reuse, and duplicate city names all affect interpretation.

## Runs Reviewed

This report focuses on the runs after the single-template dataset revision:

- `020028`: easier prompt, ridge, needle-span steering, 100 examples.
- `023323`: vanilla prompt, counterfactual `needle-last`, needle-span steering, 200 examples.
- `031329`: easier prompt, counterfactual `needle-last`, needle-span steering, 200 examples.
- `035744`: vanilla prompt, ridge, prompt last-token steering, 100 examples.
- `041801`: vanilla prompt, ridge, needle-span steering, 100 examples.
- `054922`: vanilla prompt, counterfactual `needle-last`, needle-span steering, 200 examples, with `STEERING_TEST_EVAL=True`.
- `171336`: vanilla prompt, counterfactual `needle-last`, needle-span steering, 200 examples, with `STEERING_TEST_EVAL=True`, target haystack length 1000, and held-out eval datasets for counts 1-10.
- `011105`: vanilla prompt, counterfactual `needle-last`, needle-span steering, 200 examples, target haystack length 500, and held-out eval datasets for counts 1-10.
- `041923`: vanilla prompt, counterfactual `needle-last`, needle-span steering, 200 examples, target haystack length 200, and held-out eval datasets for counts 1-10.

## Main conclusions

**Finding 1**: The ridge vector is not good as a steering vector.

- Ridge probes can decode some count-related structure, but steering with the ridge vector was inert in the two clean vanilla ridge runs.
  - In `035744`, ridge with `FEATURE_CALC_POS=last` and `STEERING_POSITION_MODE=last_token` had successful accuracy unchanged at `1.0` and unsuccessful accuracy unchanged at `0.0` across the reported sweep.
    - See [`run_results/run_20260616_035744_.../tables/counting_features/steering/steering_summary.csv`](../run_results/run_20260616_035744_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering/steering_summary.csv).
  - In `041801`, ridge with `FEATURE_CALC_POS=needle-last` and `STEERING_POSITION_MODE=needle_span` was also inert: unsuccessful aggregate accuracy stayed `0.0`, and successful aggregate accuracy stayed `1.0`.
    - See [`run_results/run_20260616_041801_.../tables/counting_features/steering_needle_span/steering_summary.csv`](../run_results/run_20260616_041801_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_needle_span/steering_summary.csv).
- The ridge runs do not show the rescue pattern later seen with counterfactual steering.
  - `035744` and `041801` both had good baseline task accuracy, `96/100 = 0.96`, so the steering failure is not simply because the task prompt failed.
    - See each run's [`tables/metrics.json`](../run_results/run_20260616_041801_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/metrics.json).

**Finding 2**: For extracting and using steering vectors, the last token in the last needle is better than the prompt last token; steering works best at needle span, not prompt last token.

- The strongest current evidence is indirect but consistent: last-token ridge steering did not move predictions, while needle-span counterfactual steering did.
  - `035744` used prompt last-token ridge steering and showed no rescue.
    - See [`035744/tables/counting_features/steering/steering_summary.csv`](../run_results/run_20260616_035744_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering/steering_summary.csv).
  - `023323` and `054922` used `COUNTING_FEATURE_CALC_METHOD=counterfactual`, `FEATURE_CALC_POS=needle-last`, and `STEERING_POSITION_MODE=needle_span`; both produced nontrivial rescue effects on selected unsuccessful examples.
    - See [`023323/tables/counting_features/steering_counterfactual_needle_span/steering_summary.csv`](../run_results/run_20260616_023323_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_counterfactual_needle_span/steering_summary.csv).
    - See [`054922/tables/counting_features/steering_counterfactual_needle_span/steering_summary.csv`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_counterfactual_needle_span/steering_summary.csv).
- The causal target also appears span-specific: in held-out steering eval, the best layer/beta depends on which needle span is steered.
  - Best average improvement by needle span in `054922`:
    - `needle_1`: layer `24`, beta `2.0`, average accuracy `0.80 -> 0.81`.
    - `needle_2`: layer `20`, beta `6.0`, average accuracy `0.75 -> 0.825`.
    - `needle_3`: layer `16`, beta `2.0`, average accuracy `0.667 -> 0.717`.
    - `needle_4`: layer `16`, beta `2.0`, average accuracy `0.50 -> 0.60`.
    - `needle_5`: layer `16`, beta `3.0`, accuracy `0.40 -> 0.55`.
    - See [`054922/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv).

**Finding 3**: Counterfactual-based steering works moderately well, although this result needs further experimental evidence.

- Counterfactual directions separate original count-3 successful examples from counterfactual count-2 successful examples with a sizable margin.
  - In `023323` and `054922`, the counterfactual direction used `194` successful original examples and `200` successful counterfactual examples.
  - Layer-wise raw norms and margin-over-pooled-std in `054922`:
    - layer `16`: norm `10.34`, margin/std `1.85`
    - layer `20`: norm `22.50`, margin/std `1.88`
    - layer `24`: norm `30.38`, margin/std `1.77`
    - layer `28`: norm `43.04`, margin/std `1.48`
    - See [`054922/tables/counting_features/counterfactual/needle-last/counterfactual_count_summary.csv`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/counterfactual/needle-last/counterfactual_count_summary.csv).
- On selected failed examples, counterfactual steering can rescue many interventions.
  - In `023323`, best aggregate unsuccessful rescue was layer `24`, beta `2.0`: accuracy `0.0 -> 0.60`, or `9/15` interventions correct. Successful examples remained intact for several layer/beta choices.
  - In `054922`, best selected-example rescue was layer `24`, beta `3.0`: accuracy `0.0 -> 0.667`, or `10/15` interventions correct.
    - See [`054922/tables/counting_features/steering_counterfactual_needle_span/steering_summary.csv`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_counterfactual_needle_span/steering_summary.csv).
- Held-out eval confirms that counterfactual steering can help harder counts, but the effect is not uniformly positive.
  - In `054922`, unsteered held-out accuracy was perfect for counts 1-3, then fell for counts 4 and 5:
    - Dataset 1: `1.00`, average predicted count `1.0`
    - Dataset 2: `1.00`, average predicted count `2.0`
    - Dataset 3: `1.00`, average predicted count `3.0`
    - Dataset 4: `0.60`, average predicted count `3.7`
    - Dataset 5: `0.40`, average predicted count `4.8`
  - Best Dataset 4 steering setting was layer `24`, target `needle_4`, beta `4.0`: accuracy `0.60 -> 0.95`, average count `3.7 -> 4.05`.
  - Best Dataset 5 steering setting was layer `16`, target `needle_4`, beta `6.0`: accuracy `0.40 -> 0.65`, average count `4.8 -> 5.25`.
    - See [`054922/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv).

**Finding 4, new**: Prompt style matters a lot even after sentence insertion and single-template facts.

- `vanilla` prompt style was much stronger than `easier` in the post-template runs.
  - `020028`, `easier`, ridge, 100 examples: baseline accuracy `50/100 = 0.50`.
  - `031329`, `easier`, counterfactual, 200 examples: baseline accuracy `133/200 = 0.665`.
  - `035744` and `041801`, `vanilla`, ridge, 100 examples: baseline accuracy `96/100 = 0.96`.
  - `023323` and `054922`, `vanilla`, counterfactual, 200 examples: baseline accuracy `194/200 = 0.97`.
    - See the corresponding `tables/metrics.json` files under each run folder.
- The counterfactual count-2 task was easier than the original count-3 task in the vanilla counterfactual runs.
  - `023323` and `054922` both had counterfactual accuracy `200/200 = 1.0`.
    - See [`054922/tables/metrics_counterfactual.json`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/metrics_counterfactual.json).

**Finding 5, new**: The model's unsteered counting ability appears near-perfect for counts 1-3 but degrades for counts 4-5.

- The held-out steering-eval datasets in `054922` are cleanly stratified by count and contain no duplicate city names.
  - Dataset 1 has 20 examples with one inserted record.
  - Dataset 2 has 20 examples with two inserted records.
  - Dataset 3 has 20 examples with three inserted records.
  - Dataset 4 has 20 examples with four inserted records.
  - Dataset 5 has 20 examples with five inserted records.
    - See [`054922/generate_data/steering_test_eval/`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/generate_data/steering_test_eval/).
- Baseline held-out predictions show a clear undercounting pattern at higher counts.
  - Dataset 4: `12/20` correct; `7/20` predicted 3 instead of 4; `1/20` predicted 5.
  - Dataset 5: `8/20` correct; `9/20` predicted 4 instead of 5; `2/20` predicted 6; `1/20` predicted 8.
    - See [`054922/tables/counting_features/steering_test_eval/steering_test_eval_predictions.csv`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_predictions.csv).

**Finding 6, new**: In the broader count-1-to-count-10 eval, Qwen3-8B is reliable for small counts but not for larger numbers of inserted needles; counterfactual steering gives small-to-moderate gains for selected settings, but the effect is not uniformly consistent.

- The `171336` run is the cleanest current stress test for larger counts because it uses the post-template vanilla prompt, sentence-level insertion, counterfactual `needle-last` feature extraction, needle-span steering, and `STEERING_TEST_EVAL=True` with 10 examples for each count from 1 to 10.
  - See [`171336/tables/counting_feature_run_metadata.json`](../run_results/run_20260616_171336_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_feature_run_metadata.json).
  - See [`171336/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv`](../run_results/run_20260616_171336_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv).
- The unsteered model is perfect for counts 1-3, mostly correct for count 4, and then unreliable for counts 5-10.
  - Baseline accuracy by held-out dataset in `171336`: Dataset 1 `1.0`, Dataset 2 `1.0`, Dataset 3 `1.0`, Dataset 4 `0.8`, Dataset 5 `0.4`, Dataset 6 `0.1`, Dataset 7 `0.1`, Dataset 8 `0.4`, Dataset 9 `0.0`, Dataset 10 `0.3`.
  - The raw baseline prediction distributions show undercounting and instability at higher counts: Dataset 6 mostly predicts `5`; Dataset 7 often predicts `6`; Dataset 9 never predicts the correct count `9`.
    - See [`171336/tables/counting_features/steering_test_eval/steering_test_eval_predictions.csv`](../run_results/run_20260616_171336_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_predictions.csv).
- The best fixed-beta steering settings over the harder datasets 5-10 improve average accuracy, but the effect size is modest.
  - Best fair setting over datasets 5-10: layer `16`, `needle_2`, beta `2.0`, average accuracy `0.217 -> 0.317`, average change `+0.100`.
  - Next best fair setting: layer `16`, `needle_4`, beta `4.0`, average accuracy `0.217 -> 0.300`, average change `+0.083`.
  - Layer `24`, `needle_1`, beta `4.0` and layer `24`, `needle_5`, beta `4.0` both improved average accuracy `0.217 -> 0.283`, average change `+0.067`.
- Per-dataset best settings show that steering can rescue some harder-count examples, but not all of them.
  - Dataset 4: best setting reaches `0.8 -> 1.0`.
  - Dataset 5: best setting reaches `0.4 -> 0.6`.
  - Dataset 6: best setting reaches `0.1 -> 0.4`.
  - Dataset 7: best setting reaches `0.1 -> 0.3`.
  - Dataset 8: no clear improvement beyond baseline `0.4`.
  - Dataset 9: no setting rescues accuracy above `0.0` in this sweep.
  - Dataset 10: best setting reaches `0.3 -> 0.5`.
- The most defensible summary is therefore two-part: Qwen3-8B does not reliably count larger numbers of needles in this setup, and the current counterfactual needle-span steering strategy produces reproducible positive gains for selected settings, but the gains are small-to-moderate and sensitive to layer, beta, and target needle.

**Finding 7, new**: Shorter haystacks do not remove the large-count counting failure.

- Two follow-up runs shortened the target haystack length while keeping the same core setup: vanilla prompt, sentence-level insertion, single city-score template, counterfactual `needle-last` feature extraction, and needle-span steering.
  - `011105` used `TARGET_HAYSTACK_TOKENS=500`.
    - See [`011105/tables/counting_feature_run_metadata.json`](../run_results/run_20260617_011105_Qwen3-8B_match_count_vanilla_500_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_feature_run_metadata.json).
  - `041923` used `TARGET_HAYSTACK_TOKENS=200`.
    - See [`041923/tables/counting_feature_run_metadata.json`](../run_results/run_20260617_041923_Qwen3-8B_match_count_vanilla_200_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_feature_run_metadata.json).
- Both short-context runs stayed nearly perfect on the original three-needle task, so the prompt and basic dataset construction were working.
  - `011105`: original accuracy `199/200 = 0.995`; counterfactual accuracy `200/200 = 1.0`.
  - `041923`: original accuracy `199/200 = 0.995`; counterfactual accuracy `200/200 = 1.0`.
    - See [`011105/tables/metrics.json`](../run_results/run_20260617_011105_Qwen3-8B_match_count_vanilla_500_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/metrics.json) and [`041923/tables/metrics.json`](../run_results/run_20260617_041923_Qwen3-8B_match_count_vanilla_200_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/metrics.json).
- The held-out count eval still shows a large-count failure. Accuracy remains perfect for counts 1-3, then becomes unstable for larger counts.

| Dataset count | 1000-token run `171336` | 500-token run `011105` | 200-token run `041923` |
|---:|---:|---:|---:|
| 1 | 1.0 | 1.0 | 1.0 |
| 2 | 1.0 | 1.0 | 1.0 |
| 3 | 1.0 | 1.0 | 1.0 |
| 4 | 0.8 | 0.7 | 0.6 |
| 5 | 0.4 | 0.6 | 0.9 |
| 6 | 0.1 | 0.5 | 0.6 |
| 7 | 0.1 | 0.4 | 0.3 |
| 8 | 0.4 | 0.2 | 0.6 |
| 9 | 0.0 | 0.0 | 0.0 |
| 10 | 0.3 | 0.3 | 0.2 |

- The shorter contexts improve some mid-count cases, especially count 5 and parts of counts 6-8, but they do not produce reliable large-count counting. Dataset 9 remains `0.0` across all three context lengths, and Dataset 10 remains weak at `0.2-0.3`.
  - See [`011105/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv`](../run_results/run_20260617_011105_Qwen3-8B_match_count_vanilla_500_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv) and [`041923/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv`](../run_results/run_20260617_041923_Qwen3-8B_match_count_vanilla_200_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv).
- This suggests that the main bottleneck is not merely long-context retrieval. If context length were the dominant problem, reducing the haystack from 1000 to 500 or 200 tokens should have produced a broad recovery on counts 7-10. Instead, the model remains brittle when many city-score records must be enumerated. The failure looks more like a counting/cardinality-tracking limitation, or a difficulty maintaining a stable set of many retrieved records, than a pure inability to locate needles in a long context.
- The 200-token run also introduced replacement sentence-boundary insertion in the highest-count eval datasets, because there were sometimes fewer sentence delimiters than active needles. This is useful for making small-haystack stress tests possible, but it may create dense/back-to-back records that can change the error mode. For example, the 200-token run overcounted strongly at counts 9-10, with average predicted count `10.4` for Dataset 9 and `13.1` for Dataset 10.

**Finding 8, new**: Short-context steering results are similar in spirit to the 1000-token/2000-token cases: selected settings help modestly, but no single fixed intervention robustly solves counts 5-10.

- Using one fixed `(layer, needle, beta)` over Datasets 5-10, the best 500-token setting was layer `16`, `needle_2`, beta `2.0`: average accuracy `0.333 -> 0.400`, change `+0.067`.

| Dataset | acc before | acc after | change |
|---:|---:|---:|---:|
| 5 | 0.600 | 0.700 | +0.100 |
| 6 | 0.500 | 0.300 | -0.200 |
| 7 | 0.400 | 0.300 | -0.100 |
| 8 | 0.200 | 0.600 | +0.400 |
| 9 | 0.000 | 0.100 | +0.100 |
| 10 | 0.300 | 0.400 | +0.100 |

- Using one fixed `(layer, needle, beta)` over Datasets 5-10, the best 200-token setting was layer `16`, `needle_4`, beta `2.0`: average accuracy `0.433 -> 0.467`, change `+0.033`.

| Dataset | acc before | acc after | change |
|---:|---:|---:|---:|
| 5 | 0.900 | 0.900 | +0.000 |
| 6 | 0.600 | 0.800 | +0.200 |
| 7 | 0.300 | 0.300 | +0.000 |
| 8 | 0.600 | 0.600 | +0.000 |
| 9 | 0.000 | 0.000 | +0.000 |
| 10 | 0.200 | 0.200 | +0.000 |

- These tables reinforce the earlier steering conclusion: the counterfactual direction is not inert, but the gains are selective. The 500-token run has a clearer positive signal, especially on Dataset 8, while the 200-token best fixed setting mostly helps only Dataset 6. The method is therefore better described as a modest, target-sensitive intervention rather than a robust large-count counting fix.

## Objections and caveats

- **Objection 1**: Several post-template main datasets still contain duplicate city names because the notebook appears to have reused cached generated data from before the duplicate-city fix.
  - Duplicate inserted-city rows found:
    - `020028`: 3 duplicate rows in 100 original examples.
    - `023323`: 5 duplicate rows in 200 original examples; 4 duplicate rows in the counterfactual dataset.
    - `031329`: 5 duplicate rows in 200 original examples; 4 duplicate rows in the counterfactual dataset.
    - `035744` and `041801`: 3 duplicate rows in 100 original examples.
    - `054922`: 5 duplicate rows in 200 original examples; 4 duplicate rows in the counterfactual dataset.
  - Example duplicate pattern: Vientiane or Harbin appears twice in the same example with two different scores.
  - This does not affect the `054922` held-out steering-eval datasets; those count 1-5 datasets were checked and had no duplicate inserted city names.
- **Objection 2**: The strongest counterfactual steering result from regular steering uses only a small selected set: 5 successful and 5 unsuccessful examples, with 15 interventions per group for count-3 data.
  - The held-out `STEERING_TEST_EVAL` is more informative and shows both improvements and large damages depending on layer, beta, and target needle.
- **Objection 3**: Counterfactual steering is target-sensitive and can be destructive.
  - In `054922`, layer `16`, target `needle_2`, beta `6.0` damaged Dataset 2 from `1.00` to `0.20`.
  - Layer `20`, target `needle_3`, beta `-1.0` damaged Dataset 3 from `1.00` to `0.10`.
  - See [`054922/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv`](../run_results/run_20260616_054922_Qwen3-8B_match_count_vanilla_1000_needles_100_200_400_rand_insrt_seed_42_sent_insrt_tmpl_niah_fact_single_template/tables/counting_features/steering_test_eval/steering_test_eval_summary.csv).
- **Objection 4**: Some runs share cached baseline metrics pointing to earlier `/content/run_20260616_023323...` paths. This is probably expected from dataset/cache reuse, but it makes run provenance harder to read.
  - Future reports should rely on resolved metadata and local copied files, not only absolute paths saved from Colab.
- **Objection 5**: The phrase "consistent improvement" is too strong for the current steering results unless it is qualified.
  - The `171336` harder-count eval shows positive average gains for selected fixed settings, but some target needles are inert, some settings damage easy datasets, and Dataset 9 is not rescued by this sweep.
  - A better wording is: counterfactual needle-span steering gives small-to-moderate, directionally positive gains for selected layer/needle/beta settings, but it is not yet a robust general counting intervention.

## To-do list

- Regenerate the main original and counterfactual datasets after clearing/rebuilding the reusable dataset cache, so the duplicate-city fix is truly reflected in feature-vector training runs.
- Repeat `STEERING_TEST_EVAL=True` with a clean, no-duplicate counterfactual feature vector.
- Increase held-out steering-eval examples per count beyond 20 once the settings are stable; 20 examples is enough for signal hunting, but not enough for precise estimates.
- Add a compact automatic analysis script for `steering_test_eval_summary.csv` that reports:
  - baseline accuracy by dataset count,
  - best layer/beta by dataset,
  - best layer/beta by needle span across applicable datasets,
  - worst damaging settings.
- Consider a more systematic intervention policy instead of one global layer/beta:
  - count-4 examples seemed to prefer layer `24`, target `needle_4`;
  - count-5 examples often improved with layer `16`, later needle targets.
- Continue treating ridge as a decoding/probing baseline, not as the main causal steering vector, unless a new ridge-derived intervention is proposed.
