# Realistic CoT NIAH Count V3.1 Preregistration

## 1. Registration status and scope

This document preregisters the V3.1 behavior-comparison and empirical-law
experiment before any V3.1 outcomes are collected or inspected. V3.1 is a
prospective extension of V3, informed by the design and failure modes observed
in earlier pilot versions. It retains the V3 stimuli, prompts, model panel,
inference settings, parsing principles, and audit requirements except where a
change is stated explicitly below.

The confirmatory scope is:

1. compare all four registered prompt modes;
2. estimate parsed exact-count accuracy and its parsing/format decomposition;
3. estimate a robust conditional signed-count bias;
4. fit prompt-mode-specific empirical laws with a common functional form and
   model-specific intercepts and slopes;
5. describe observable counting styles in model outputs.

Mechanistic representations, attention heads, probes, activation patching,
causal intervention, causal mediation, and a separate mechanism dataset are
out of scope. No behavioral association in V3.1 will be described as evidence
for an internal causal mechanism.

The machine-readable companion is
`configs/realistic_niah_v3_1.json`. Counting-style definitions are frozen in
`docs/realistic_niah_v3_1_cot_style_codebook.md`.

Any change after registration must be added to a dated amendment log before
the affected analysis is run. An amended or unregistered analysis must be
labelled exploratory and must not replace the preregistered result.

## 2. Changes from V3

V3.1 makes the following registered changes:

- increase paired seeds from 10 (`1234..1243`) to 30 (`1234..1263`);
- extend the V3 passage-length grid downward by adding `L=1000` canonical
  tokens;
- retain Direct, Index Enumeration, Bullet Enumeration, and Native Thinking
  as confirmatory prompt modes;
- make the 10% symmetric trimmed signed mean the primary bias estimand;
- decompose parsed accuracy into parseability, numerical correctness, format
  compliance, and strict correctness;
- allow model-specific slopes under one shared candidate structure within
  each prompt mode and outcome;
- require interaction hierarchy, predictive improvement, and a multiplicity-
  corrected significance check before retaining an interaction;
- add held-seed, held-count, held-length, leave-one-model-out, and paired-seed
  bootstrap validation;
- add a preregistered, outcome-blind counting-style taxonomy.

All other V3 design choices remain unchanged unless contradicted here.

## 3. Registered design

### 3.1 Stimulus grid

The full Cartesian grid is:

- final passage length including needles:
  `L = 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000` canonical tokens;
- true city-score record count:
  `N = 1,2,3,4,5,6,7,8,9,10,12,15,18,20`;
- paired seeds: the 30 integers `1234..1263`.

This produces `8 x 14 x 30 = 3,360` shared stimuli. Every applicable
checkpoint and prompt mode receives the same stimulus IDs. The seed is a
paired sampling unit across all `N`, `L`, model, and mode conditions; it is
also the cluster used for resampling.

V3 insertion-depth, final-length, tokenizer, no-repeat haystack, contamination,
checkpoint-revision, and prompt-snapshot audits remain mandatory. Needle starts
remain inside the inclusive 5%-95% interval in final-passage character
coordinates.

### 3.2 Prompt modes and model panel

All four V3 modes are retained as confirmatory conditions:

1. `direct`;
2. `enumeration_index`;
3. `enumeration_bullet`;
4. `native_thinking`.

The frozen prompt text and the 12 behavior-comparison slots/14 raw checkpoints
are inherited unchanged from V3. The GLM and Ministral Direct-vs-Native
comparisons use different matched checkpoints and remain checkpoint-confounded;
they are not interpreted as same-weight causal effects of enabling thinking.

With 48 checkpoint-mode shards, each containing 3,360 requests, the planned
total is `48 x 3,360 = 161,280` requests.

### 3.3 Decoding, completion, and missingness

Decoding settings and mode-specific output budgets are inherited from V3 and
must be frozen in the run manifest. A request stopped by the output-token limit
is retained and flagged as truncated. Runtime failures are retried under the
same frozen configuration; they are not silently converted to model errors.
The confirmatory analysis starts only after the final audit accounts for every
registered request as completed or as a documented irrecoverable runtime
failure. A checkpoint substitution requires a preregistration amendment.

## 4. Registered outcomes

Let `n_total` be the number of registered requests in an aggregation cell,
`n_parseable` the number for which the frozen parser returns an integer, and
`n_correct_parsed` the number for which that integer equals the true `N`.

### 4.1 Primary accuracy

The primary accuracy is

```text
parsed_exact_accuracy = n_correct_parsed / n_total
```

Equivalently, the request-level outcome is

```text
1[an integer is parsed and parsed integer == N].
```

Every completed request is in the denominator. A parse failure is incorrect,
and a parseable but numerically wrong answer is also incorrect.

Every accuracy table and figure must report the following counts and rates so
that parsed correctness is not conflated with format following:

- `n_total`;
- `n_parseable` and `parse_rate = n_parseable / n_total`;
- `n_correct_parsed` and the primary `parsed_exact_accuracy`;
- `conditional_numeric_accuracy = n_correct_parsed / n_parseable`, when
  `n_parseable > 0`;
- `n_format_compliant` and `format_compliance_rate`;
- `n_correct_and_format_compliant`;
- `n_truncated` and `truncation_rate`;
- `strict_accuracy`, requiring a correct count, registered mode-specific
  format compliance, and no truncation.

Parseability and format compliance are overlapping diagnostics, not mutually
exclusive categories. The report must additionally provide one exclusive
failure category per request: runtime failure, truncation, parse failure,
undercount, overcount, format-only failure, or strict success.

The parser, fallback order, and mode-specific format checks must be frozen and
hashed before outcomes are joined to analysis tables. Manual correction of a
parsed answer is prohibited in the confirmatory analysis.

### 4.2 Primary bias

For each successfully parsed response define signed deviation

```text
d_i = predicted_count_i - N.
```

Bias is conditional on successful parsing. No deviation is imputed for a
parse failure. Parse rate must accompany every bias result.

Within each `behavior_slot x prompt_mode x N x L` cell, sort the `m`
successfully parsed deviations as
`d_(1) <= ... <= d_(m)`. Let `k = floor(0.10 m)`. The primary cell estimand is

```text
trimmed_signed_bias_10 =
    sum(d_(k+1), ..., d_(m-k)) / (m - 2k).
```

Thus a fully parseable 30-seed cell removes exactly the three lowest and three
highest deviations and averages the remaining 24. This order-statistic rule,
rather than an interpolated sample-quantile rule, resolves percentile and tie
ambiguities. When tied values cross a trim boundary, deleting any tied members
gives the same retained-value mean.

A cell is eligible for the confirmatory conditional-bias law only when
`m >= 20`. Below this threshold its robust bias is still shown descriptively,
but the cell is marked `insufficient_conditional_bias_coverage` and excluded
from primary bias-law fitting. Its requests remain in all accuracy, parse-rate,
and failure analyses.

The following secondary outcomes retain the information deliberately
downweighted by the primary robust estimand:

- arithmetic mean signed deviation;
- median signed deviation;
- mean and median absolute deviation;
- 90th and 95th percentiles of absolute deviation;
- maximum absolute deviation;
- undercount and overcount rates among parseable responses;
- the fraction of total absolute error contributed by the largest 5% of
  absolute errors.

Extreme outputs are never deleted from the raw data. They affect primary
accuracy and all tail-risk diagnostics; trimming is used only to estimate the
central signed-bias law.

## 5. Empirical-law models

### 5.1 Analysis unit, links, and coefficient structure

Laws are selected separately for each prompt mode and outcome. The primary
accuracy law uses request-level Bernoulli likelihood with a logistic link. A
second confirmatory version fits cell accuracy counts with explicit Binomial
and Beta-Binomial observation distributions as specified in Section 5.3.
Conditional bias uses the eligible cell-level 10% trimmed signed bias with an
identity link.

For prompt mode `p`, behavior slot `m`, and registered predictors `x_j(N,L)`,
the general form is

```text
g(E[Y | m,p,N,L]) = alpha_(m,p) + sum_j beta_(m,p,j) x_j(N,L).
```

The candidate term structure is shared across behavior slots within a prompt
mode, but every behavior slot may have its own intercept and slope for every
included term. The primary model is therefore not restricted to a shared
`N` or `L` slope and is not selected independently for each model.

Use `L_k = L/1000`, `logN = ln(N)`, and `logL = ln(L_k)`. Natural logarithms
are used. Predictors are standardized using training-fold means and standard
deviations for numerical fitting; reported effects are transformed back to
the registered units.

### 5.2 Bounded candidate registry

Only the following confirmatory structures are searched, in the stated
tie-breaking order:

1. intercept only;
2. `N`;
3. `L_k`;
4. `logN`;
5. `logL`;
6. `N + L_k`;
7. `logN + logL`;
8. `N + logL`;
9. `logN + L_k`;
10. `N + L_k + N:L_k`;
11. `logN + logL + logN:logL`;
12. `N + logL + N:logL`;
13. `logN + L_k + logN:L_k`.

Interactions are only between a representation of count and a representation
of length. An interaction always includes both parent main effects. Terms such
as `N:logN`, `L_k:logL`, higher-order polynomials, and post-hoc formulas are
not part of the confirmatory search.

### 5.3 Probability-distribution version of the accuracy law

In addition to the primary request-level Bernoulli-logistic law, V3.1 fits a
registered distributional version to the count of exact parsed successes in
each `behavior_slot x prompt_mode x N x L` cell. Let

```text
Y_(m,p,N,L) = n_correct_parsed
T_(m,p,N,L) = n_total.
```

The fixed-denominator Binomial reference model is

```text
Y ~ Binomial(T, mu),
logit(mu) = alpha_(m,p) + sum_j beta_(m,p,j) x_j(N,L).
```

The overdispersed probability model is

```text
Y ~ BetaBinomial(T, a=mu*kappa_(m,p), b=(1-mu)*kappa_(m,p)),
logit(mu) = alpha_(m,p) + sum_j beta_(m,p,j) x_j(N,L).
```

Here `kappa_(m,p) > 0` is a behavior-slot- and prompt-mode-specific
concentration parameter that is constant over `N` and `L`; the implied
intra-cell overdispersion is `rho_(m,p) = 1/(kappa_(m,p)+1)`. This model is
used instead of ordinary beta regression because the observations are integer
success counts with a known denominator and may equal 0 or 100% accuracy.

Both distributions use the same bounded candidate registry, model-specific
slopes, hierarchical interaction rules, and validation splits as the primary
law. The Binomial model is always reported as the nested no-overdispersion
reference; the Beta-Binomial model is not selected merely because it has
better in-sample likelihood.

For this distributional version, the primary selection/comparison metric is
held-out negative log predictive density. Also report held-out Brier score,
calibration by predicted-probability bin, randomized probability-integral-
transform diagnostics, and empirical coverage of 50%, 80%, and 95% predictive
intervals for the cell success count. Report `kappa`, `rho`, and paired-seed
bootstrap intervals. If the Beta-Binomial fit reaches its Binomial boundary or
fails to improve held-out log score, report that the data provide no reliable
evidence of extra-Binomial dispersion.

Interaction eligibility is assessed separately within the primary
Bernoulli-logistic and distributional versions. A headline interaction claim
is based on the primary Bernoulli-logistic law; agreement or disagreement with
the Beta-Binomial version is reported as a robustness result rather than used
to choose the more favorable conclusion.

### 5.4 Interaction eligibility and significance

An interaction structure is eligible for final selection only if all of the
following hold:

1. both parent main effects are present;
2. its primary held-seed loss is lower than that of its matched additive
   parent model;
3. the joint null that all model-specific interaction slopes equal zero is
   rejected by a paired-seed cluster bootstrap test;
4. the joint interaction p-value remains below 0.05 after Holm correction
   across the four registered interaction structures within the same
   `outcome x prompt_mode` family.

The joint test determines whether the shared structure may contain the
interaction. For interpretation, every model-specific interaction coefficient
must also be reported with a paired-seed bootstrap 95% interval and a
Holm-adjusted p-value across behavior slots. A jointly supported interaction
does not imply that every individual model has a significant interaction.

An interaction that improves in-sample fit but fails any eligibility condition
is reported in the candidate table but cannot be selected as the confirmatory
law.

### 5.5 Selection metrics

Candidate selection is nested inside the relevant training split.

- Primary Bernoulli-logistic accuracy-law loss: held-out log loss. Brier score,
  calibration slope/intercept, and deviance explained are secondary.
- Distributional accuracy-law loss: held-out negative log predictive density.
  Brier score, probability calibration, randomized probability-integral-
  transform diagnostics, and predictive-interval coverage are secondary.
- Conditional-bias primary loss: held-out MAE of the cell robust-bias target.
  RMSE and held-out `R^2` are secondary and cannot override MAE selection.

Among eligible candidates, use a one-standard-error rule and select the model
with the fewest terms whose primary loss is within one standard error of the
best candidate. Remaining ties follow the candidate order in Section 5.2.
If no non-intercept candidate improves reliably over the intercept-only model,
the result is reported as `no reliable empirical law`.

## 6. Validation design

All preprocessing parameters must be learned on the training portion of a
split. No held-out observation may influence standardization, candidate
selection, interaction testing, or coefficient estimation for its fold.

### 6.1 Held-seed generalization

- Accuracy: five-fold grouped cross-validation, with six complete seed IDs in
  each test fold.
- Conditional bias: three-fold grouped cross-validation, with ten complete
  seed IDs in each test fold. The cell target is recomputed separately in
  train and test data; a ten-seed test cell trims one observation from each
  tail when all ten responses parse.

For split-specific bias summaries, the parse-coverage threshold scales with
the number of seeds available in that split: at least `ceil(2s/3)` parseable
responses are required when a split contains `s` seeds. This gives 7 of 10 in
a bias test fold, 14 of 20 in its training complement, and recovers the
registered 20-of-30 threshold in the complete data.

Seeds are assigned by sorting the 30 registered seed IDs and distributing
them round-robin over folds. A seed is absent from training for every `N`,
`L`, model, and mode when it is held out.

### 6.2 Held-count validation

Perform leave-one-`N`-level-out validation over all 14 registered count levels.
Report pooled metrics, per-level metrics, and distinguish boundary extrapolation
(`N=1` and `N=20`) from interpolation at interior levels.

### 6.3 Held-length validation

Perform leave-one-`L`-level-out validation over all eight registered passage
lengths. Report pooled metrics, per-level metrics, and distinguish boundary
extrapolation (`L=1000` and `L=20000`) from interpolation at interior levels.

### 6.4 Leave-one-model-out stability

Repeat functional-form selection after removing each of the 12 behavior slots
in turn. Because the primary law permits model-specific slopes, this analysis
tests candidate-structure stability and does not claim zero-calibration
prediction of the omitted model's coefficients. Report:

- selected structure in each leave-one-model-out replicate;
- frequency with which the full-data structure is recovered;
- whether interaction eligibility changes;
- changes in the remaining models' coefficient signs and ranks.

Any optional hierarchical new-model prediction is exploratory unless added by
a prospective amendment.

### 6.5 Paired-seed cluster bootstrap

Use 2,000 bootstrap replicates. Each replicate samples the 30 seed IDs with
replacement and carries every selected seed's observations across all `N`,
`L`, models, and modes. Recompute cell summaries and refit the law in each
replicate.

For the fixed selected structure, report percentile 95% coefficient intervals,
sign stability, and interaction tests. Separately report full reselection
stability as candidate-selection frequencies. To avoid an unregistered nested
bootstrap-inside-bootstrap test, the reselection stage conditions on the set of
interactions that passed the full-data held-loss and Holm-adjusted joint test;
it does not retest interaction eligibility inside every bootstrap draw.
Interaction eligibility stability is reported separately through coefficient
sign/interval stability and the leave-one-model-out eligibility checks. Failed
numerical fits remain in the bootstrap audit table and are not silently
replaced.

## 7. Prompt-mode comparisons

All six pairwise contrasts among the four prompt modes are confirmatory. They
use shared stimulus IDs within a behavior slot. Report paired risk differences,
95% seed-cluster bootstrap intervals, discordant-pair counts, exact McNemar
p-values, and Holm-adjusted p-values across the six contrasts within each
behavior slot. Bias contrasts use paired cell robust-bias differences and
paired-seed bootstrap intervals.

The report must keep same-checkpoint comparisons separate from the checkpoint-
confounded GLM and Ministral comparisons.

## 8. Counting-style analysis

Observable style is coded from the generated completion, not inferred from
accuracy or from an unobserved internal state. The preregistered codebook
includes, at minimum:

- numeric index enumeration;
- bullet enumeration;
- word/prose enumeration;
- running tally;
- arithmetic grouping;
- scan/retrieval summary;
- answer only/no visible reasoning;
- repetition or restart;
- mixed;
- other/unclassifiable.

Styles are represented as non-exclusive request-level flags plus one exclusive
dominant-style label. Failure/process flags such as repetition, restart,
truncation, and template leakage are separate from the core strategy labels.
The detailed operational rules, observability field, annotation protocol, and
validation thresholds are in the V3.1 codebook.

The style classifier and human validation sample are frozen before style
labels are joined to `N`, correctness, or bias outcomes. Style frequencies by
model, prompt mode, `N`, and `L` are confirmatory descriptive results.
Accuracy/bias conditional on observed style is associational and must not be
interpreted causally because style is a post-treatment model behavior.

## 9. Reporting and reproducibility requirements

The V3.1 report must include:

1. the full request-accounting and audit table;
2. accuracy with parsing and format-compliance decomposition;
3. robust central bias together with non-robust mean and tail-risk diagnostics;
4. Bernoulli-logistic, Binomial, and Beta-Binomial accuracy-law results with
   probability calibration and predictive-distribution diagnostics;
5. every attempted candidate, including failed or ineligible interactions;
6. held-seed, held-count, held-length, and leave-one-model-out results;
7. bootstrap coefficient, sign, interaction, and selection stability;
8. counting-style prevalence and classifier validation results;
9. same-checkpoint and checkpoint-confounded results in separate panels;
10. machine-readable tables, run/config/parser/code hashes, and SHA256 manifests.

No claim of extrapolation beyond `N=1..20` or `L=1000..20000` is confirmatory.
No interaction is described as present unless it passes the registered joint
and multiplicity-corrected test. Missing, weak, negative, or unstable results
are retained in the report.
