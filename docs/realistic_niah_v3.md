# Realistic CoT NIAH Count V3

## 1. Scope and provenance

V3 implements only the **Behavior comparison** and **Empirical law** arms of
`pipeline_CoT_counting.pdf`. Mechanistic representations, attention heads,
probes, activation patching, causal intervention, and causal mediation are
deliberately excluded. Those questions require a separately designed
repository and should not be inferred from V3 behavior.

The implementation source document reviewed for this version was:

- local filename: `pipeline_CoT_counting.pdf`;
- 6 pages;
- SHA256:
  `9cb68b4f33a4242294230e8092da2082899b493e43df535a032725241e8ebadb`.

All six rendered pages were inspected in addition to text extraction. The V2
prompt, tokenizer, haystack, parser, vLLM, checkpoint, resume, and audit
components are reused where their semantics are unchanged. V3 uses separate
schemas, request IDs, dataset IDs, configuration, shard plans, and reports.

## 2. Registered stimulus grid

The frozen Cartesian grid is:

- final passage length including needles:
  `L = 2k, 3k, 5k, 8k, 10k, 15k, 20k` canonical tokens;
- true city-score record count:
  `N = 1,2,3,4,5,6,7,8,9,10,12,15,18,20`;
- paired seeds: `1234..1243`.

This produces `7 × 14 × 10 = 980` shared stimuli. Every checkpoint/mode sees
the same stimulus IDs.

Final passage length is measured with the frozen
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` tokenizer.
The freeze command rejects a different revision rather than resolving a
moving branch.

The Paul Graham corpus remains `multi_file_no_repeat`: a filler source is not
repeated inside one passage. Final length is solved after inserting the
records, so the task template and output budget are not counted in `L`.

### 2.1 Registered insertion depth

Each needle start must lie in the inclusive 5%–95% interval of the **final
passage character coordinate**:

```text
normalized_depth = needle_char_start / final_passage_characters
```

The generator first filters sentence-boundary candidates conservatively,
allowing for all characters that the insertions may add. It then recomputes
the final spans and validates every realized depth. No out-of-range sample is
silently dropped after generation.

V2 retains the default 0%–100% interval. The new depth controls are opt-in and
therefore do not change V2 generation behavior.

## 3. Frozen prompts

All modes use the same prefix:

```text
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{PASSAGE}
</passage>
```

### Direct

```text
How many city-score audit records are in the passage?
Do not explain, reason aloud, quote, or list any records.
Your entire response must be exactly one line:
Total: <integer>
```

### Index enumeration

```text
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin the first item with "1. ", the second with "2. ", and continue with ordinary digits.
After each number, write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text.
```

### Bullet enumeration

```text
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin each item with "-", then write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text.
```

### Native thinking

```text
How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>
```

Prompt snapshots remain centralized in `realistic_niah.prompts`; V3 does not
maintain a second, drifting copy.

## 4. Model panel and comparison units

### 4.1 Same-checkpoint, switchable thinking

The following ten checkpoints receive all four modes:

- Qwen3: 4B, 8B, 14B, 32B;
- Gemma 4: E4B, 12B, 26B-A4B, 31B;
- NVIDIA Nemotron Nano v2 9B;
- NVIDIA Nemotron 3 Nano 4B BF16.

Direct, index, and bullet disable native thinking with the model's registered
chat-template control. Native thinking enables it. These comparisons change
the mode while holding checkpoint weights fixed.

### 4.2 Matched but different checkpoints

Two conceptual behavior slots use separate checkpoints:

- `GLM-4-9B-0414` supplies Direct/Index/Bullet and
  `GLM-Z1-9B-0414` supplies Native Thinking;
- `Ministral-3-8B-Instruct-2512` supplies Direct/Index/Bullet and
  `Ministral-3-8B-Reasoning-2512` supplies Native Thinking.

These are **not** same-weight causal comparisons. The Ministral Instruct and
Reasoning checkpoints also differ in published weight precision/packaging, so
precision and post-training are confounded.

V3 therefore records both:

- 14 raw checkpoint labels; and
- 12 behavior-comparison slots.

The immutable model IDs and revisions are in
`src/realistic_niah_v3/spec.py` and `configs/realistic_niah_v3.json`.

## 5. Request accounting

One checkpoint × mode combination is one resumable shard:

- 10 switchable checkpoints × 4 modes = 40 shards;
- GLM control/reasoning pair = 3 + 1 shards;
- Ministral control/reasoning pair = 3 + 1 shards.

Thus V3 contains 48 shards, each with 980 requests, for
`48 × 980 = 47,040` requests. Request IDs begin with the `v3/` namespace.

The resource-aware scheduler supports 2 through 8 visible GPUs. It freezes
resource requirements inside the shard plan, then greedily packs the highest
priority pending tasks while allowing one-GPU tasks to backfill unused GPUs.
Gemma4-31B reserves two GPUs with tensor parallel size 2; Qwen3-32B uses one
H100 with `gpu_memory_utilization=0.92`; all remaining shards use one GPU.
GPU assignments never overlap.

Each task runner:

1. atomically claims one shard;
2. writes request-ID checkpoints after every bounded vLLM batch;
3. archives failed attempts without deleting completed rows;
4. safely resumes only missing request IDs;
5. never changes prompts, parser, decoding, model revision, or stimuli.

## 6. Outcomes

The PDF phrase “correct as long as we can parse the result” is ambiguous:
parseability alone cannot make a numerically wrong answer correct. V3 freezes
the following explicit definitions.

### 6.1 Primary accuracy

```text
parseable_exact_accuracy =
    1[predicted integer exists and predicted integer == N]
```

Every request remains in the denominator. A parse failure is incorrect; a
parseable but wrong integer is also incorrect.

### 6.2 Diagnostics

- parse rate;
- strict registered accuracy: exact count, correct mode-specific format, and
  no length truncation;
- format compliance;
- truncation rate;
- enumeration record precision/recall already supplied by the V2 parser;
- exclusive failure class: truncation, parse failure, undercount, overcount,
  format-only failure, or strict success.

Prompt modes are also compared on their shared stimulus IDs. For each pair,
V3 reports the exact-accuracy risk difference \(p_B-p_A\), a 95% interval
from resampling the ten seeds as clusters, discordant-pair counts, an exact
McNemar p-value, and a Holm-adjusted p-value across the six mode contrasts
within a behavior slot. The GLM and Ministral native-thinking contrasts remain
checkpoint-confounded even though their stimuli are paired.

### 6.3 Native-thinking visible counting style

Native-thinking traces are assigned one mutually exclusive, preregistered
observable-text class:

- indexed list;
- bullet list;
- mixed indexed and bullet structure;
- word enumeration using two or more distinct ordinal words;
- inline tally or arithmetic;
- prose reasoning without a registered list/tally marker;
- no visible reasoning.

The precedence is mixed structure, index, bullet, ordinal-word enumeration,
inline tally/arithmetic, prose, then empty. Classification describes the visible trace only; it is not
evidence that the same device is the model's latent counting mechanism. The
report gives each class's share and exact-count accuracy overall and by N/L,
plus deterministic HTML excerpts. The companion JSONL retains the complete
reasoning and final text for each selected example.

### 6.4 Bias and dispersion

For a successfully parsed response:

```text
signed deviation d = predicted_count - N
absolute deviation = |d|
```

No deviation is imputed for a parse failure. Every bias table must therefore
be interpreted with its parse rate. Condition summaries include:

- mean signed deviation;
- mean absolute deviation;
- median signed deviation;
- 10% trimmed signed mean;
- sample variance
  `s² = sum((d_i - mean(d))²) / (n - 1)`.

## 7. Empirical-law search

Within each prompt mode, V3 fits one shared functional form:

```text
response_m(N,L) = alpha_m + f_mode(N,L)
```

`alpha_m` is a behavior-slot fixed intercept. All `N,L` coefficients are
shared across slots, so this is not a post-hoc separate formula per model.
Accuracy is fit as a discrete probability distribution over the number of
correct seeds in each behavior-slot × N × L cell.

The bounded candidate grid uses:

- `N`;
- `L_k = L/1000`;
- `ln(N)`;
- `ln(L_k)`;
- density `N/L_k`;
- one hierarchical linear interaction `N * L_k`;
- one hierarchical log interaction `ln(N) * ln(L_k)`.

An interaction candidate always includes both parent main effects. No
higher-order polynomial or interaction is searched.

Every response surface is crossed with four explicit accuracy families:

- Binomial with logit link;
- Binomial with probit link;
- Binomial with complementary-log-log link;
- Beta-Binomial with logit mean and one fitted concentration parameter.

The Beta-Binomial tests whether seed-to-seed outcomes show more dispersion
than a Binomial law permits. It is not treated as automatically preferable.
Accuracy probability-distribution results and continuous deviation regressions
are written to separate comparison tables as well as the complete combined
table; distributional assumptions are never transferred to the bias targets.

### 7.1 Validation and selection

Five-fold grouped cross-validation holds out complete seeds. A held-out seed
is absent from training for every `N`, `L`, checkpoint, and mode, preventing
same-stimulus leakage.

- accuracy: held-out predictive negative log density of cell success counts,
  Bernoulli mean-probability log loss, Brier score, and deviance explained
  relative to a per-model training prevalence;
- continuous deviation targets: held-out condition-level R², MAE, and RMSE;
- all fits: convergence status, coefficients, standard errors, p-values, and
  95% intervals.

A fixed \(10^{-8}\) numerical ridge is applied to non-intercept link-scale
coefficients solely to keep separated folds finite. It is not searched or
tuned. Wald intervals and interaction p-values should therefore be read as
large-sample diagnostics rather than exact finite-sample tests.

After selection, the fitted discrete CDF produces fixed-seed Dunn--Smyth
randomized quantile residuals. Their Q--Q plots compare residual quantiles to
standard-normal quantiles. Q--Q correlation R-squared, Shapiro--Wilk, and
Cramer--von Mises are reported as distribution-adequacy diagnostics. They do
not select the model, and they are not a conventional Q--Q "significance
test"; held-out predictive scores remain primary.

A one-standard-error-style rule chooses the simplest near-best candidate.
Interaction candidates are ineligible when their interaction coefficient has
`p >= 0.05`. All attempted models remain in
`candidate_comparison.csv`; weak or negative held-out results are reported as
“no reliable unified law” rather than hidden.

## 8. Reproducible workflow

Create a dedicated V3 inference environment on the Linux GPU host:

```bash
python3 -m venv /home/ubuntu/venvs/realistic-niah-vllm
/home/ubuntu/venvs/realistic-niah-vllm/bin/python -m pip install \
  -r requirements-inference-v3.txt
```

V3 pins `transformers==5.14.1`, `vllm==0.25.1`, and
`mistral-common>=1.8.6,<2`. V3 does not contain OLMo 3. Do not use the
historical `requirements-inference.txt` environment for V3: its older
Transformers pin exists only for the OLMo 3 extension and does not recognize
the registered Gemma 4 Unified checkpoints.

Create an analysis environment separately from inference:

```bash
python -m venv .venv-analysis
. .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
```

Gather the full content-deduplicated RULER Paul Graham corpus as in V2, then:

```bash
PYTHONPATH=src python scripts/freeze_realistic_niah_v3.py \
  --output-dir /path/to/run/dataset \
  --haystack-dir /path/to/run/source_corpus \
  --haystack-corpus-manifest /path/to/run/source_corpus/corpus_manifest.json \
  --cache-dir /path/to/hf-cache
```

The freeze command also writes `dataset/audit_report.json` and stops if the
980-row grid, exact lengths, 5%–95% depths, uniqueness, contamination audit,
or checksums fail.

On a Linux GPU host:

```bash
bash scripts/launch_realistic_niah_v3.sh /path/to/run 8
```

The second argument is the maximum visible GPU count (2--8). The launcher
starts one scheduler tmux and one finalizer tmux. The scheduler records every
live task-to-GPU allocation in `orchestration/scheduler_state.json`; a shard
failure stops new launches while allowing already running shards to finish.
All outputs and request-ID checkpoints are preserved for an explicit safe
resume.

After all 48 completion markers, the finalizer performs a lossless merge and
requires 47,040 rows and 47,040 unique request IDs. It independently checks
the exact 980 registered stimulus IDs, canonical tokenizer revision, final
lengths, 5%–95% depths, all checkpoint revisions, the frozen shard-plan hash,
the preparation Git commit, every shard manifest, and every canonical
manifest/QC file.

Generate the two registered reports only after the final audit passes:

```bash
PYTHONPATH=src python scripts/analyze_realistic_niah_v3.py \
  --run-root /path/to/run \
  --output-dir /path/to/run/analysis/v3_behavior_empirical_law
```

Outputs include request- and condition-level tables, paired mode-comparison
statistics, native-thinking counting-style tables/examples, the full
response-surface × probability-family comparison, selected coefficients,
observed/fitted model panels, randomized-residual Q--Q plots and tests, two
HTML reports, an analysis plan, a resumable state file, and a SHA256 manifest.

## 9. Interpretation boundary

V3 can support statements such as:

- mode A has higher paired behavioral accuracy than mode B on this grid;
- a mode shows systematic overcount or undercount;
- a compact shared response surface predicts held-out seeds.

V3 cannot support statements such as:

- a visible chain of thought is the internal counting algorithm;
- a particular representation or attention head causes counting;
- prompting caused a checkpoint to acquire a mechanism;
- the selected law extrapolates outside `N=1..20` and `L=2k..20k`.

Those claims belong to the future causal/mechanistic repository.
