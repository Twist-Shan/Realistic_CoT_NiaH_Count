# Realistic CoT NIAH Count V3 formal plan

## Scope

V3 covers behavior comparison and empirical-law analysis only. Mechanistic,
representation, attention, intervention, and causal work is intentionally
excluded for a later repository.

## Frozen experiment

- Passage length: 2k, 3k, 5k, 8k, 10k, 15k, 20k canonical tokens.
- True records: 1--10, 12, 15, 18, 20.
- Seeds: 1234--1243, yielding 980 shared stimuli.
- Prompt modes: Direct, Index enumeration, Bullet enumeration, Native
  thinking.
- Models/checkpoints and immutable revisions are registered in
  `src/realistic_niah_v3/spec.py`.
- 48 model×mode shards, 980 requests each, 47,040 requests total.

Primary accuracy is one exactly when a parsed integer equals N. Parse failure,
wrong count, format failure, and truncation remain in the denominator.

## Native-thinking counting-style analysis

Classify the observable reasoning text into exactly one frozen category:
indexed list, bullet list, mixed structured list, ordinal-word enumeration,
inline tally/arithmetic, prose reasoning, or no visible reasoning. Report
frequency and exact accuracy overall and by N/L, with HTML excerpts and a
complete example JSONL. Do not interpret visible style as
proof of a latent mechanism.

## Empirical laws

For each mode, all comparison slots share N/L slopes and receive separate
fixed intercepts. The finite response-surface grid uses N, L/1000, ln N,
ln(L/1000), density, and at most one hierarchical first-order interaction.

Accuracy jointly searches:

1. Binomial-logit;
2. Binomial-probit;
3. Binomial-complementary-log-log;
4. Beta-Binomial-logit with one concentration parameter.

Five-fold GroupKFold holds out complete seeds. Select accuracy by held-out
predictive negative log density with Brier score as a secondary criterion.
Diagnose the selected distribution with fixed-seed Dunn--Smyth randomized
quantile residual Q--Q plots, Q--Q correlation R-squared, Shapiro--Wilk, and
Cramer--von Mises. These diagnostics do not replace held-out selection.

Bias and dispersion targets remain mean/median/trimmed signed deviation, mean
absolute deviation, and sample variance among successfully parsed outputs;
every table must be interpreted alongside parse rate.

Accuracy distribution fits and continuous deviation regressions are retained
as two separate comparison tables. They share predictor names but not a
likelihood or goodness-of-fit criterion.

## Eight-H100 execution

The formal plan freezes per-shard GPU count, tensor parallel size, request
batch size, sequence concurrency, memory utilization, and model length.
Gemma4-31B uses 2 GPUs with TP=2. Qwen3-32B uses one H100 at memory
utilization 0.92. A central resource-aware greedy/backfill scheduler supports
2--8 GPUs and never overlaps allocations. A task failure stops new launches,
preserves every request-ID checkpoint and log, and lets already-running tasks
drain before exiting.

## Completion criteria

Only merge after all 48 shard markers exist and none failed. The final audit
must verify 47,040 rows and unique request IDs, exact frozen stimuli, model
revisions, Git commit, shard-plan hash, manifests, QC reports, and Filestream
mount provenance. Analysis begins only after that audit passes.
