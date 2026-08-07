# V4.4 counter-channel supplement

## Scientific questions

1. Is count represented in a stable residual subspace across layers, and how
   does prompt/trace running-index geometry align with answer-query geometry?
2. How accurately can common classifiers decode count from answer-query hidden
   states under seed-held-out evaluation?
3. Which task, position, trace-form, attention, representation and behavioral
   factors explain prompt-versus-trace conditional variance?
4. Does a projected prompt-side donor patch causally reach the answer, and is
   that effect mediated by a later count subspace?

The observational claim is *persistent encoding*.  The stronger phrase
*counter channel carries/mediates information* is reserved for projected
patch/removal/serial-mediation results with norm-matched controls.

## CPU/GPU allocation

| Stage | Resource | Reason |
|---|---|---|
| Absolute-deviation tables and report | CPU | Uses embedded/generated labels only |
| Layer-manifest validation and NPZ loading | CPU | I/O and shape checks |
| Centroid axes, rank-1/2/3 subspaces, principal angles | CPU | NumPy SVD on one layer at a time |
| Cross-layer/cross-role ridge decoding | CPU | Seed-grouped scikit-learn |
| Logistic, SVM, kNN, centroid, LDA, NB and tree classifiers | CPU | No model forward; parallelize by layer |
| Conditional variance and noise-factor models | CPU | Tabular regression and grouped CV |
| Missing all-layer hidden/QK capture | GPU | Requires model forward |
| Projected patch, removal and serial mediation | GPU | Requires intervention during prefill/generation |
| Optional repeated native traces | GPU | Needed to separate sampling from document variance |

Raw tensors stay in the isolated FileStream run.  CPU scripts consume a layer
manifest whose shards contain `states [rows,hidden]`, `count`, and `seed`.
For prompt/trace roles, `count` is the externally aligned running index; for
the answer-query role it is the final gold count. The manifest adapter for the
9,000-row store is intentionally deferred until its real shard schema is
mounted, so it will validate rather than guess event/token semantics.

## Commands after FileStream is mounted

```bash
python scripts/analyze_realistic_niah_v4_4_counter_subspaces.py \
  --manifest "$RUN/counter_channel/layer_manifest.json" \
  --output "$RUN/counter_channel/analysis/subspaces_discovery" --rank 3 \
  --fit-splits discovery

# Outcome-conditioned sensitivity only; never replace the balanced main fit.
python scripts/analyze_realistic_niah_v4_4_counter_subspaces.py \
  --manifest "$RUN/counter_channel/layer_manifest.json" \
  --output "$RUN/counter_channel/analysis/subspaces_correct_only" --rank 3 \
  --fit-splits discovery --correct-only

python scripts/analyze_realistic_niah_v4_4_answer_classification.py \
  --manifest "$RUN/counter_channel/layer_manifest.json" \
  --output "$RUN/counter_channel/analysis/classification" --n-jobs 16

python scripts/analyze_realistic_niah_v4_4_counter_noise.py \
  --manifest "$RUN/counter_channel/layer_manifest.json" \
  --factor-config configs/realistic_niah_v4_4_noise_factors.json \
  --covariates "$RUN/counter_channel/noise_covariates.csv.gz" \
  --output "$RUN/counter_channel/analysis/noise" --n-jobs 16

python scripts/run_realistic_niah_v4_4_counter_subspace_interventions.py \
  --v4-config configs/realistic_niah_v4.json \
  --stimuli "$RUN/stimuli.jsonl" \
  --jobs "$RUN/counter_channel/intervention_jobs.json" \
  --output "$RUN/counter_channel/gpu_interventions" \
  --cache-dir "$HF_HOME"

python scripts/analyze_realistic_niah_v4_4_counter_subspace_interventions.py \
  --run-root "$RUN/counter_channel/gpu_interventions" \
  --output "$RUN/counter_channel/analysis/interventions"
```

The first server pass should create/validate the manifest and inspect trace-event
coverage before launching interventions.  Trace events must be externally
defined count updates; hidden-state decoding must not be used to choose the
events that are later claimed to contain count.

## Classification protocol

- Outer split: grouped by seed/document, never random rows.
- Primary labels: gold count 1--10.
- Secondary error audit: whether the probe predicts gold or the model's actual
  output on wrong samples.
- Screen common algorithms at frozen landmark layers; run a full layer sweep
  for the fast linear/local methods listed in the campaign config.
- Fit scaling and PCA inside each training fold.
- Report accuracy, balanced accuracy, count MAE and confusion matrices.
- Report seed-cluster bootstrap intervals; do not treat 9,000 token events as
  9,000 independent experimental units.

## Noise protocol

For each held-out seed and count, form the residual relative to a centroid
estimated without that seed.  Decompose squared residual energy into count-
subspace and orthogonal components.  Compare all factors against group-ablation
models using held-out-seed R-squared reduction.  Feature importance is
predictive attribution, not a causal effect.

`correct`, `absolute_deviation`, and `signed_error` are post-outcome diagnostic
factors. They may explain where noise concentrates but may not be described as
upstream causes. Position/spacing, task structure, trace form, attention
routing and pre-outcome representation dynamics form the primary explanatory
factor families.

Native trace sampling variance cannot be separated from document variance with
one trace per prompt.  If the 9,000-row store lacks sampling replicates, run a
small N=10 repeated-trace supplement after strict event alignment is audited.
