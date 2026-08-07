# Realistic NIAH V3.1 Implementation Guide

## Status

V3.1 is implemented as a separate package, `realistic_niah_v3_1`, so its
3,360-stimulus grid, schemas, IDs, audits, and empirical-law definitions cannot
silently change V3. The V3 prompt text, model registry, immutable revisions,
and generic inference engine are reused.

The implementation provides:

- frozen-grid generation and independent audit;
- an immutable 48-shard plan and resumable GPU workers;
- final provenance audit and lossless canonical merge;
- parsed-accuracy/format decomposition;
- 10% symmetric trimmed cell bias with the 20-of-30 coverage rule;
- model-specific slopes for all 13 preregistered candidate structures;
- Bernoulli-logistic, Binomial, and Beta-Binomial accuracy laws;
- bootstrap/Holm interaction eligibility;
- nested held-seed, held-`N`, and held-`L` validation;
- leave-one-model-out structure selection;
- deterministic CoT style classification and blinded human-validation files.

## Freeze the dataset

From the repository root, with `PYTHONPATH=src`:

```bash
python scripts/freeze_realistic_niah_v3_1.py \
  --output-dir /path/to/run/dataset \
  --haystack-dir /path/to/source_corpus \
  --haystack-corpus-manifest /path/to/source_corpus/corpus_manifest.json \
  --cache-dir /path/to/hf-cache
```

This command refuses any grid other than eight passage lengths, fourteen
counts, and seeds 1234-1263. The audit requires exactly 3,360 stimulus rows.

## Prepare and launch inference

The formal launcher requires a clean Git worktree and a run root below
`runs/realistic_niah_v3_1/`:

```bash
bash scripts/launch_realistic_niah_v3_1.sh \
  /path/to/runs/realistic_niah_v3_1/RUN_NAME 8
```

It writes an immutable shard plan, launches one worker per requested GPU, and
starts a finalizer. The finalizer merges only after all 48 shards complete and
requires exactly 161,280 unique request IDs.

Individual stages can also be run with:

```bash
python scripts/prepare_realistic_niah_v3_1.py --run-root RUN_ROOT
python scripts/run_realistic_niah_v3_1.py --help
python scripts/merge_realistic_niah_v3_1_shards.py --run-root RUN_ROOT
```

## Run the confirmatory analysis

The full registered analysis uses 2,000 paired-seed bootstrap replicates at
the full-data, nested-validation, coefficient, and leave-one-model-out stages:

```bash
python scripts/analyze_realistic_niah_v3_1.py \
  --run-root RUN_ROOT \
  --run-lomo \
  --run-bootstrap-reselection
```

Important outputs include:

- `model_mode_summary.csv`: parsed accuracy, parse rate, format rate, and
  strict accuracy;
- `bias_cells.csv`: robust central bias, raw mean/median, coverage, and tail
  diagnostics;
- `candidate_comparison.csv`, `selected_laws.csv`, and `coefficients.csv`;
- `interaction_tests.csv`: held-seed improvement, cluster-bootstrap joint
  tests, Holm correction, and model-specific intervals;
- `nested_held_seed_validation.csv`, `held_N_validation.csv`, and
  `held_L_validation.csv`;
- Binomial/Beta-Binomial calibration, randomized-PIT, and predictive-interval
  tables;
- CoT style request/summary tables and blinded annotation samples.

The exact nested bootstrap analysis is intentionally expensive. For code
development only, bootstrap counts may be lowered and LOMO omitted. Such an
output is a smoke test and is not a confirmatory V3.1 result.

## Validate CoT style coding

After two blinded annotators and adjudication fill the `human_*` columns in
the exported random annotation sample, run:

```bash
python scripts/validate_realistic_niah_v3_1_cot_style.py \
  --annotations ANNOTATED_RANDOM_SAMPLE.csv \
  --automated-styles cot_style_request_level.csv.gz \
  --output-dir STYLE_VALIDATION_OUTPUT
```

The command exits nonzero unless weighted Cohen's kappa is at least 0.75 and
weighted macro-F1 over supported multilabel styles is at least 0.80.

## Development checks

On Windows in this workspace, the compatible environment is the repository's
parent `.venv`:

```powershell
$env:PYTHONPATH='src'
& '..\.venv\Scripts\python.exe' -m pytest -q `
  tests\test_realistic_niah_v3.py `
  tests\test_realistic_niah_v3_1.py
```
