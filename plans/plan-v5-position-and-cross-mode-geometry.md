# V5 registered extensions: first retrieval and position-wise geometry

This plan is frozen before inspecting V5 attention or cross-mode geometry
results.  It extends, but does not replace, the registered V5 representation
and causal analyses.

## E1. First targeted retrieval versus later retrieval

### Data and selection

- Cohort: native-thinking `trace_one_to_one` rows.
- Head selection: discovery only.
- Exact attention contract: `target_needle_raw_mass` and
  `target_needle_relative_mass`, both computed on the frozen full prompt-record
  span for the semantic target city.
- Two discovery rankings are frozen independently:
  - `targeted_first`: occurrence `k=1`;
  - `targeted_later`: occurrences `k>=2`.
- For the first/later rankings, restrict both regions to requests with `N>=2`
  and first average later positions within request, so an N=10 request does not
  receive nine times the selection weight of an N=2 request.
- Registered bank sizes: 1, 2, 4, 8, 16, 32.

### Confirmation estimand

For each request and selected head bank, compute

```text
mass(k=1) - mean_{k>=2} mass(k).
```

Average first within heads, then within request, then within seed.  Confidence
intervals use a seed-cluster bootstrap.  Report the estimand separately for raw
and relative target-needle mass.  Do not pool the two quantities.

Also report discovery head-score Spearman correlation, top-K identity overlap,
Jaccard overlap, and Jensen-Shannon distance between top-K layer histograms.

## E2. Targeted-first versus the V4.4 first-locator bank

The first-locator bank is not redefined on V5.  It is loaded from the frozen
V4.4 non-thinking discovery registry, whose score is

```text
first needle span mass - mean(other needle span masses).
```

At native-thinking occurrence `k=1`, compare that bank with the
discovery-ranked `targeted_first` bank.  Primary inference is the paired
confirmation request/seed contrast in exact raw and relative target-needle
mass.  Wasserstein and KS distances over individual head-query values are
descriptive only because those rows are not independent.  Head identity and
layer-distribution overlap are reported separately.

This comparison distinguishes two hypotheses:

1. the first retrieval reuses a fixed first-occurrence locator family;
2. the first retrieval uses the same semantic targeted-retrieval family as
   later enumeration positions.

It does not by itself establish causality.  The registered V5 head ablations
remain the causal test.

## E3. Does position-wise representation quality deteriorate?

### Pairing and labels

- Non-thinking: V4.4 N=10 prompt `span_end` running-index states.
- Native thinking: V5 N=10 `item_end` enumeration states, primary
  `one_to_one` cohort.
- Pair the two modes on complete discovery and confirmation seed trajectories.
- Each running/enumeration index `k in {1,...,10}` is one class.
- Fit all scaling, PCA, centroids, and classifiers on discovery only.  Evaluate
  on confirmation only.
- PCA dimension is registered as 32, with 16 and 64 as sensitivity analyses.

### Class-specific quality curves

For every model, mode, layer, and class `k`, report:

- multinomial-logistic precision, recall, and F1;
- nearest-class-centroid (NCC) recall;
- cosine silhouette;
- within-class covariance trace;
- class NC1 ratio `tr(Sigma_k) / ||mu_k-mu_G||^2`;
- nearest-centroid squared separation and its within-trace Fisher ratio;
- minimum and mean pairwise Bhattacharyya distance using OAS-shrunk class
  covariances;
- class contribution to NC2 simplex-ETF cosine deviation.

For every metric, register whether higher or lower means better and transform
it to a common `quality_value`.  Position degradation is summarized by:

- quality slope per index;
- Spearman correlation between `k` and quality;
- late (`k=8..10`) minus early (`k=1..3`) quality.

A negative slope or negative late-minus-early contrast means degradation.

### Global covariance and neural-collapse diagnostics

For each model, mode, and layer, report:

- Papyan--Han--Donoho NC1,
  `tr(Sigma_W Sigma_B^dagger) / C`;
- NC2 ETF Gram relative error and centered-centroid norm CV;
- NCC confirmation accuracy;
- logistic-probe/NCC disagreement as an **NC4-like diagnostic**, not strict
  NC4, because there is no native terminal classifier for intermediate index
  labels;
- regularized Pillai trace;
- regularized Lawley--Hotelling trace;
- regularized Wilks lambda and log separation.

The MANOVA generalized eigenvalues use discovery-fitted PCA and an explicit
trace-scaled ridge on pooled within-class covariance.  Class-specific
Bhattacharyya distances use OAS covariance shrinkage.  This is necessary
because hidden dimension is much larger than the per-class confirmation sample
size.

### Cross-mode claim

For each layer/index/metric, report

```text
native-thinking quality - non-thinking quality.
```

For curve-level estimands, use a paired delete-one-confirmation-seed jackknife.
Positive values mean native thinking is better after applying the registered
quality direction.  Results must be reported layerwise; any selected-layer
headline must use a layer fixed from discovery, not the best confirmation
layer.

## References

- Papyan, Han, and Donoho, *Prevalence of Neural Collapse during the Terminal
  Phase of Deep Learning Training*, PNAS (2020), arXiv:2008.08186.
- Ji, Lu, Zhang, Deng, and Su, *An Unconstrained Layer-Peeled Perspective on
  Neural Collapse*, ICLR (2022), arXiv:2110.02796.
- The user-supplied cluster-metric note motivating covariance-aware MANOVA
  statistics and Bhattacharyya distance.
