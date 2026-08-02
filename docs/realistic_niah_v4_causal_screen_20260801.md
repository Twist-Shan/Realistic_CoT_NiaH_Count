# Realistic NIAH V4 causal screen

This note records the completed `screen_8h_v1` causal campaign and the
`answer_query_dense_v1` follow-up for
`run_20260731_v4_numeric_presentation_v3`. Both retained both models, all four
V4 panels, and all ten held-out confirmation seeds while restricting the
intervention grid to diagnostic conditions. The initial screen completed on
2026-08-01 at 08:47:15 UTC in 3 h 05 min, including preflight; the exact
answer-query follow-up completed at 16:35:59 UTC.

## Scientific question and estimands

The four stages test different claims and should not be conflated.

1. **Mixed-bank necessity:** does ablating the discovery-ranked `span_end`
   head bank at the answer-query row change the final greedy count more than
   ablating the same number of layer-matched random heads? The ranking was not
   phenotype-pure, so this estimand does not isolate broad heads.
2. **Endpoint transport:** is the exact hidden state at the toggled needle-end
   token sufficient to transport a one-count change when copied from a nested
   donor prompt into a receiver prompt?
3. **Exact query-state transport:** is the one residual vector at the
   prompt-final `Total:` query sufficient to transfer a donor prompt's
   already-computed prediction at a given layer?
4. **Query-state manipulability:** does a discovery-fit count-centroid delta at
   the answer-query residual move held-out greedy outputs toward a target count
   more than an orthogonal vector with the same norm?

All behavior fields use the complete deterministic greedy continuation after
the prompt-final `Total:` query. Count `10` is parsed from its full generated
token sequence. No candidate probability, first-token softmax, or teacher
forcing defines correctness or intervention success.

Point estimates weight intervention rows equally. Uncertainty intervals first
average variants and count pairs within each confirmation seed, then perform a
20,000-repetition percentile bootstrap over the ten paired seeds. The primary
two-sided tests are exact sign-flip tests over those ten seed-level contrasts;
Holm correction is applied separately to the four ablation contrasts, six
needle-end patching contrasts, and six steering contrasts. The dense
answer-query primary analysis compares each of seven later layers with L0 by
paired seed, with Holm correction within model; panel, directed-pair,
direction, and baseline-outcome tables are robustness/descriptive strata.

## Screen design and completion audit

| Stage | Per-model design | Qwen design | Gemma design |
| --- | --- | --- | --- |
| Mixed ranked-bank ablation | counts 7--10; `span_end`; answer-query; top-4/top-8; one layer-matched random set | `design_2ccb6d6eee0f` | `design_5a1dcb9fa083` |
| Residual patching | 5↔6, 7↔8, 9↔10; exact toggled needle end; cumulative from three relative depths | `design_1702318319c8` | `design_f9c90bcbf650` |
| Exact answer-query patching | 5↔6, 7↔8, 9↔10, 5↔10; prompt-final query state; eight single-layer sites | `design_e67fc867381a` | `design_823e7a6638be` |
| Geometric steering | 7↔8, 9↔10, 5↔10; centroid delta; α=1; three relative depths; one norm-matched orthogonal control | `design_cf77fd4452c2` | `design_28163399d9ee` |

| Artifact per model | Qwen3-8B | Gemma4-E4B |
| --- | ---: | ---: |
| Ablation prompt shards / detail rows | 160 / 640 | 160 / 640 |
| Patching family shards / detail rows | 40 / 720 | 40 / 720 |
| Successful / skipped patch rows | 720 / 0 | 720 / 0 |
| Answer-query family shards / detail rows | 40 / 2,560 | 40 / 2,560 |
| Successful / skipped answer-query rows | 2,560 / 0 | 2,560 / 0 |
| Valid / strict-invalid answer-query outputs | 2,560 / 0 | 2,555 / 5 |
| Steering discovery NPZ | 800 | 800 |
| Steering confirmation families / detail rows | 40 / 1,440 | 40 / 1,440 |
| Query-state discovery shape | 3 × 4,096 | 3 × 2,560 |

The audit checks every indexed shard, consolidated detail row, intervention
identity, required summary/control table, finite discovery state, and expected
layer shape. Causal baseline labels match the saved behavior table exactly;
patched `correct`, `wrong`, and `invalid` labels agree with the parsed final
continuation and gold count. No `Traceback`, OOM, or `FAILED` marker occurs in
the screen logs.

The earlier interrupted Qwen full-grid ablation remains preserved as
`design_b46a127ab7ad`. It is not mixed into the screen estimates or report.

## 1. The discovery-ranked mixed head bank has a necessary contribution

The primary contrast is ranked minus layer-matched random. Negative generated
count shift means stronger undercount after ablating ranked heads.

| Model | Set | Changed ranked / random | Δ changed [95% seed CI] | Count shift ranked / random | Δ count shift [95% seed CI] | Holm p | Δ MAE [95% seed CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | top-4 | 8.8% / 6.2% | +2.5 pp [-1.9, +7.5] | -0.138 / +0.094 | **-0.231 [-0.300, -0.156]** | 0.0078 | +0.219 [+0.144, +0.294] |
| Qwen3-8B | top-8 | 23.1% / 10.0% | **+13.1 pp [+6.9, +18.8]** | -0.363 / -0.031 | **-0.331 [-0.413, -0.256]** | 0.0078 | +0.331 [+0.256, +0.413] |
| Gemma4-E4B | top-4 | 88.1% / 76.9% | **+11.3 pp [+2.5, +20.0]** | -1.125 / -0.181 | **-0.944 [-1.081, -0.788]** | 0.0078 | +0.944 [+0.788, +1.081] |
| Gemma4-E4B | top-8 | 97.5% / 45.0% | **+52.5 pp [+43.1, +61.9]** | -1.706 / +0.450 | **-2.156 [-2.356, -1.969]** | 0.0078 | +2.156 [+1.975, +2.356] |

The selected span-end bank therefore contributes causally to preserving count
magnitude in both models. The result is strongest for top-8 ablation and is
not explained by merely ablating the same number of heads at the same layers.
It does **not** establish that every selected head is individually necessary,
that every selected head is broad, that all broad heads were found, or that
the bank implements an exact arithmetic sum. Qwen's ranking includes
selector/partition-like heads as well as broad heads, so the appropriate claim
is necessity of the tested mixed bank, not phenotype-specific necessity.

Only 3 of the 160 high-count receiver prompts are baseline-correct for each
model. The saved `summary.csv` and matched-control table retain correct/wrong
strata, but the correct subset is not powered for a separate causal claim. The
table above reports the prespecified pooled screen estimand.

## 2. Exact needle-end residual transport is null

A positive direction-aligned shift means insertion patches increase the
generated count and removal patches decrease it. `Moved toward donor` requires
a strict reduction in distance to the donor gold; this avoids counting a
receiver that already equals the donor count as a patch success.

| Model | Start layer | Rows | Changed | Moved toward donor [95% seed CI] | Direction-aligned count shift [95% seed CI] | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 9 | 240 | 3.8% | 2.1% [0.8, 3.3] | -0.004 [-0.042, +0.033] | 1.000 |
| Qwen3-8B | 18 | 240 | 1.7% | 1.2% [0.0, 2.9] | +0.008 [-0.008, +0.029] | 1.000 |
| Qwen3-8B | 26 | 240 | 0.8% | 0.8% [0.0, 2.5] | -0.008 [-0.025, 0.000] | 1.000 |
| Gemma4-E4B | 10 | 240 | 2.1% | 1.2% [0.0, 2.5] | +0.004 [-0.008, +0.017] | 1.000 |
| Gemma4-E4B | 20 | 240 | 1.7% | 1.2% [0.0, 2.5] | -0.008 [-0.025, +0.008] | 1.000 |
| Gemma4-E4B | 31 | 240 | 0.0% | 0.0% [0.0, 0.0] | 0.000 [0.000, 0.000] | 1.000 |

The exact endpoint state is therefore not sufficient to transport the nested
count change under cumulative patching. This null is compatible with strong
span-end decoding: a variable can be decodable at one token without being in
a transportable basis or being the sole state read by downstream generation.
Possible remaining mechanisms include position-specific routing, multiple
tokens within the record, coordinated multi-needle state, or a query-side
aggregation that cannot be recreated by one donor endpoint.

## 3. Late answer-query state transports the computed prediction

This follow-up copies exactly one donor residual vector into the receiver at
the prompt-final `Total:` query and one layer, then runs complete greedy
generation. The primary estimand is **donor-prediction adoption conditional on
receiver and donor baseline predictions differing**. This is the appropriate
transport target: donor-gold accuracy can be low simply because the donor
model was already wrong. Strict-invalid continuations remain in the eligible
denominator as failures.

| Model | Layer | Valid | Eligible rows | Adopts donor prediction [95% seed CI] | Direction-aligned shift [95% seed CI] | Adoption vs L0 Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 0 | 100% | 188 | 1.4% [0.0, 2.8] | +0.009 [-0.016, +0.038] | 1.0000 |
| Qwen3-8B | 9 | 100% | 188 | 1.4% [0.0, 3.2] | +0.003 [-0.019, +0.025] | 1.0000 |
| Qwen3-8B | 18 | 100% | 188 | 1.4% [0.0, 3.6] | +0.022 [0.000, +0.044] | 1.0000 |
| Qwen3-8B | 26 | 100% | 188 | **59.2% [52.0, 66.1]** | **+0.744 [+0.628, +0.856]** | **0.0137** |
| Qwen3-8B | 29 | 100% | 188 | 96.2% [91.7, 100.0] | +1.059 [+0.869, +1.250] | 0.0137 |
| Qwen3-8B | 32 | 100% | 188 | 97.5% [95.4, 99.5] | +1.081 [+0.884, +1.275] | 0.0137 |
| Qwen3-8B | 34 | 100% | 188 | 98.0% [96.0, 100.0] | +1.100 [+0.888, +1.313] | 0.0137 |
| Qwen3-8B | 35 | 100% | 188 | **100.0% [100.0, 100.0]** | +1.094 [+0.894, +1.294] | 0.0137 |
| Gemma4-E4B | 0 | 100% | 230 | 0.4% [0.0, 1.3] | +0.003 [0.000, +0.009] | 1.0000 |
| Gemma4-E4B | 10 | 100% | 230 | 0.4% [0.0, 1.2] | -0.006 [-0.016, 0.000] | 1.0000 |
| Gemma4-E4B | 20 | 100% | 230 | 0.8% [0.0, 2.0] | +0.003 [-0.016, +0.028] | 1.0000 |
| Gemma4-E4B | 31 | 99.69% | 230 | **86.5% [82.8, 90.0]** | **+1.192 [+1.100, +1.283]** | **0.0137** |
| Gemma4-E4B | 35 | 99.69% | 230 | 96.2% [94.4, 97.9] | +1.254 [+1.163, +1.345] | 0.0137 |
| Gemma4-E4B | 38 | 99.69% | 230 | 98.8% [97.1, 100.0] | +1.286 [+1.197, +1.370] | 0.0137 |
| Gemma4-E4B | 40 | 99.69% | 230 | 99.2% [97.9, 100.0] | +1.282 [+1.197, +1.366] | 0.0137 |
| Gemma4-E4B | 41 | 99.69% | 230 | **99.6% [98.8, 100.0]** | +1.276 [+1.188, +1.360] | 0.0137 |

Transport therefore turns on abruptly between Qwen L18 and L26 and between
Gemma L20 and L31. At the final layer, every valid eligible continuation
equals the donor prediction. The conservative eligible adoption rate is 100%
for Qwen and 99.58% for Gemma because invalid rows are failures. The result is
stable across v4.1--v4.4 and all eight directed pairs; final-layer panel and
pair estimates are included in the HTML and saved CSV tables. It also holds in
both baseline-correct and baseline-wrong strata (Qwen: 100% in both; Gemma:
96% among eligible baseline-correct rows versus 100% among baseline-wrong
rows), although the correct strata contain only 9 Qwen and 5 Gemma seeds.

### Why five Gemma outputs are `11`

All five strict-invalid rows are the same v4.1 seed-1263 family, receiver
count 5 ← donor count 10, at L31/L35/L38/L40/L41. The exact continuations are:

- receiver baseline: `5<turn|>`, token IDs `[236810, 106]`;
- donor baseline: `10<turn|>`, token IDs `[236770, 236771, 106]`;
- patched: `11<turn|>`, token IDs `[236770, 236770, 106]`.

This is not truncation or a failed hook. The single query-state patch transfers
the donor's first digit token, but the next autoregressive step is no longer
patched and emits another `1` instead of `0`. It is therefore a revealing
boundary condition: the prompt-final state is sufficient to set the first
generated decision late in the network, while multi-token realization still
depends on subsequent unpatched computation.

## 4. Late answer-query geometry is causally manipulable

For α=1 centroid-delta steering, the direction-aligned effect is compared
within prompt and target pair against an orthogonal norm-matched control.

| Model | Layer | Changed geometric / random | Moved geometric / random | Δ moved [95% seed CI] | Aligned shift geometric / random | Δ aligned [95% seed CI] | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 9 | 2.9% / 2.9% | 2.5% / 1.7% | +0.8 pp [0.0, +2.1] | +0.008 / -0.008 | +0.017 [-0.004, +0.042] | 0.750 |
| Qwen3-8B | 18 | 2.9% / 2.5% | 2.5% / 1.7% | +0.8 pp [-0.8, +2.9] | +0.046 / 0.000 | +0.046 [+0.008, +0.088] | 0.375 |
| Qwen3-8B | 26 | **55.8% / 7.1%** | **39.6% / 4.2%** | **+35.4 pp [+32.1, +38.8]** | **+0.992 / +0.033** | **+0.958 [+0.808, +1.096]** | **0.0117** |
| Gemma4-E4B | 10 | 1.7% / 2.5% | 1.2% / 1.7% | -0.4 pp [-1.7, +0.8] | -0.017 / -0.017 | 0.000 [-0.017, +0.017] | 1.000 |
| Gemma4-E4B | 20 | 3.3% / 17.9% | 2.9% / 8.3% | -5.4 pp [-10.0, -0.8] | +0.025 / -0.033 | +0.058 [+0.021, +0.100] | 0.0938 |
| Gemma4-E4B | 31 | **71.2% / 8.3%** | **53.3% / 2.9%** | **+50.4 pp [+45.4, +55.4]** | **+1.363 / -0.025** | **+1.388 [+1.283, +1.488]** | **0.0117** |

The positive causal result is localized late: Qwen L26 and Gemma L31 move
held-out greedy counts by about one and 1.4 count units, respectively, in the
intended direction beyond the norm-matched control. Earlier layers do not
survive family-wise correction.

This is directional control, not exact set-to-count control. At the final
tested layer, exact target/path hit rates are 8.75% versus 1.25% for Qwen and
7.5% versus 1.67% for Gemma. The centroid direction is therefore used by the
readout, but α=1 does not reliably land on the target integer.

## 5. Discovery-locked single- and multi-layer steering

The initial screen above localized useful layers on its confirmation set. A
separate v2 experiment removed that selection reuse. Four discovery seeds
(1234--1237) screened 15 plans per model: three singleton layers, two
multi-layer sets, and alpha in {0.25, 0.5, 1}. Each protocol locked exactly one
plan by the worst-panel strict aligned effect minus twice the invalid rate.
The lock was then evaluated on ten disjoint seeds (1254--1263), all four V4
panels, six directed count pairs, both protocols, and a norm-matched
orthogonal random arm. This yields 2,880 screen and 960 confirmation rows per
model.

| Model | Protocol | Locked plan | Held-out geometric-random aligned effect [95% seed CI] | Moved effect [95% CI] | Target-hit effect | Holm p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-8B | single | L26, alpha=1 | +1.000 [+0.867, +1.133] | +36.7 pp [+33.3, +40.0] | +7.5 pp | 0.0039 |
| Qwen3-8B | multi | L9+18+26, alpha=1 | +0.992 [+0.838, +1.142] | +37.1 pp [+33.3, +40.8] | +7.5 pp | 0.0039 |
| Gemma4-E4B | single | L31, alpha=1 | +1.371 [+1.275, +1.463] | +48.8 pp [+44.2, +53.3] | +5.8 pp | 0.0039 |
| Gemma4-E4B | multi | L10+20+31, alpha=1 | +1.387 [+1.300, +1.467] | +50.0 pp [+47.1, +53.3] | +5.4 pp | 0.0039 |

All four locked plans reproduce, and all sixteen model × protocol × panel
intervals remain above zero. Multi is not detectably better: its descriptive
advantage over single is -0.008 count unit for Qwen and +0.017 for Gemma, with
no randomized protocol-difference test. Exact target-hit gains also remain
only 5.4--7.5 percentage points. The rigorous conclusion is therefore stable
late directional manipulability across held-out seeds and panels, not precise
target setting and not a multi-layer advantage. No v2 row is strict-invalid;
the conservative invalid-as-zero rule was nevertheless applied by design.

## 6. Geometry explains why decoding and steering can diverge

All 24 model × variant × layer discovery paths are monotone along their 1→10
endpoint chord, but the paths are not straight or equally spaced.

| Model | Layer | Mean endpoint count correlation (minimum panel) | Mean adjacent-step CV | Mean successive-step cosine | Mean path/chord |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 9 | 0.988 (0.985) | 0.307 | 0.395 | 1.429 |
| Qwen3-8B | 18 | 0.993 (0.991) | 0.222 | 0.181 | 1.667 |
| Qwen3-8B | 26 | 0.820 (0.812) | 1.227 | 0.596 | 2.044 |
| Gemma4-E4B | 10 | 0.997 (0.996) | 0.143 | -0.096 | 2.155 |
| Gemma4-E4B | 20 | 0.845 (0.838) | 1.307 | 0.847 | 1.169 |
| Gemma4-E4B | 31 | 0.863 (0.852) | 1.107 | 0.538 | 1.860 |

The early/middle path can be highly count-decodable yet causally inert when
steered. Conversely, the late path is noisier and more uneven but aligns with
the generation readout strongly enough to move output. The combined evidence
supports a distinction between **information availability** and **readout
usage**.

## Reproduction and artifact locations

Run the profile-specific audit after downloading the causal subtree:

```bash
PYTHONPATH=src python scripts/audit_realistic_niah_v4_causal.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --output /path/to/run/causal_screen_8h_audit.json
```

Audit and reproduce the dense answer-query statistics with:

```bash
PYTHONPATH=src python scripts/analyze_realistic_niah_v4_answer_query_patching.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --output-dir /path/to/run/analysis/answer_query_patching_dense_v1 \
  --bootstrap-repetitions 20000
```

Rebuild the self-contained representation-plus-causal report with:

```bash
PYTHONPATH=src python scripts/build_realistic_niah_v4_representation_report.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --output reports/realistic_niah_v4_representation_report.html \
  --repo-root .
```

The rebuilt HTML is the unified V4 mechanism report rather than a causal-only
appendix. It explicitly connects the count-dependent behavior failure, the
span-end representation and attention evidence, the Qwen head-bank taxonomy,
paired omission diagnostics, and four causal claims: bank-level necessity,
single-endpoint insufficiency, late query-state sufficiency, and directional
geometric manipulability. Each result block reports its estimand, axes,
seed-level uncertainty, current conclusion, and remaining alternative
explanations. Figures follow the registered Aurora palette in `README.md` and
`scripts/build_realistic_niah_v4_representation_report.py`.

The downloaded bundle is
`exports/run_20260731_v4_numeric_presentation_v3/causal_screen_8h_bundle.tar.gz`
with SHA-256
`a4acdaaa342e85eb3d20c16d2732bf85d71ea5432905c21dc33facfa30ca8955`.
The extracted `numeric/causal` directories contain raw restartable shards,
design manifests, consolidated details, outcome-stratified summaries, matched
control tables, discovery centroids, and centroid geometry tables. The bundle
is an uncommitted run artifact; code, documentation, and the compact HTML
report are version controlled.

The final answer-query follow-up is separately archived as
`answer_query_patching_dense_v1_final_bundle.tar.gz` (115 entries, 201,744
bytes, SHA-256
`93776fdea92a07e358d52594969a7ab0d97ad9ef9107ed543d4a7daaa6567920`).
It includes `layer_summary.csv`,
`variant_summary.csv`, `pair_summary.csv`, `outcome_summary.csv`, the complete
model × layer × direction × pair × panel × baseline-outcome
`stratum_summary.csv`, `invalid_rows.csv`, and the strict audit JSON.

## Limits and next discriminating tests

- This is a targeted causal screen, not the fully powered registered sweep.
  It uses selected count pairs, three depths, α=1, and one matched random
  replicate.
- Head ablation is bank-level. A leave-one-head-out or stable-global-bank
  design is needed to separate broad aggregators from Qwen selector/partition
  heads.
- Endpoint patching tests one exact token state. A small tokenwise full-needle
  patch or coordinated multi-endpoint patch remains a distinct intervention;
  head-state patching was intentionally removed from scope.
- Exact answer-query patching establishes late-state sufficiency for the
  model's prediction, not correctness of that prediction, a uniquely scalar
  representation, or sufficiency for every later token in a multi-token
  answer. The Gemma `11` rows demonstrate the last limitation directly.
- Steering establishes late readout-aligned manipulability, but the low exact
  target-hit rate motivates an α dose response and local versus non-local pair
  comparison before claiming precise count control.
- Correct high-count baselines are rare. Correct/wrong attention results remain
  reported separately, while causal outcome strata should be enlarged only if
  a follow-up specifically targets error correction rather than the pooled
  mechanism.
