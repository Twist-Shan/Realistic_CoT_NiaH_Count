# V4.4 non-thinking mechanism extension: code map

This directory contains the report builder and the additional analyses requested in
`reports/non-thinking extension.md`. Raw activations remain on FileStream. The report
only consumes audited aggregate CSV/JSON files.

## GPU capture / intervention scripts

- `capture_v44_all_token_controls.py`: capture frozen-PCA projections for needle
  endpoints, needle interiors, matched hard negatives, and ordinary passage tokens.
- `run_v44_endpoint_attention_mask.py`: compare clean endpoint queries with
  needle-only and equal-key-budget ordinary-only attention masks; also save Qwen
  earlier-span attention rows.
- `run_v44_token_corruption.py`: replace every active needle span by equal-length
  ordinary tokens and compare with an equal-token-budget ordinary-passage mutation.
- `run_v44_prompt_subspace_ablation.py`: remove the discovery-frozen rank-3
  count-centroid subspace from all active prompt endpoints and compare with an
  equal-Frobenius-norm orthogonal removal.

## CPU analysis scripts

- `analyze_v44_extension_geometry.py`: stable/numeric rank, rank-3 capture,
  held-out regression, clustering, and balanced count/seed/interaction variance.
- `analyze_v44_prompt_classification_fast.py`: six fixed classifiers with
  seed-grouped cross-validation; no per-layer algorithm selection.
- `analyze_v44_all_token_controls.py`: category-wise decoding and endpoint-gated
  empirical formula tests.
- `analyze_v44_endpoint_attention_mask.py`: seed-level effects, exact sign-flip,
  50,000-replicate bootstrap intervals, Holm correction, and earlier-span head
  confirmation.
- `analyze_v44_token_corruption.py`: behavior and downstream answer-geometry damage,
  with needle-minus-ordinary specificity.
- `analyze_v44_prompt_subspace_ablation.py`: behavior and downstream answer-geometry
  specificity for actual/centroid rank-3 removal versus norm-matched orthogonal
  removal.

## Orchestration and report

- `launch_extension_geometry.sh`, `launch_prompt_classifiers_fast.sh`,
  `launch_all_token_capture.sh`, and `launch_extension_gpu_followups.sh` record the
  exact staged commands used on the A10 server.
- `build_realistic_niah_v4_4_integrated_report.py` reads the canonical V4.4 report,
  all prior causal analyses, and the extension aggregates; it emits one self-contained
  HTML file.
- `extension_question_matrix.md` maps every question from the extension memo to its
  experiment, estimand, and report section.

## Registered new-data split

- Token corruption: seeds 1254--1263, counts 1--10.
- Set-wide prompt-subspace ablation: seeds 1254--1263, counts 2--10.
- Qwen/Gemma axes and bases are frozen from disjoint discovery rows in the packed
  V4 data before these confirmation interventions are evaluated.

Layer/head indices are zero-based. Invalid numeric completions remain in the behavior
denominator; they receive accuracy 0 and absolute error 10 in the registered extension
analyses.
