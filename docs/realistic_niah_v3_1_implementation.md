# Realistic NIAH V3.1 Implementation Guide

## Status

V3.1 is implemented as a separate package, `realistic_niah_v3_1`, so its
3,360-stimulus grid, schemas, IDs, audits, and empirical-law definitions cannot
silently change V3. The V3 prompt text, model registry, immutable revisions,
and generic inference engine are reused.

The implementation provides:

- frozen-grid generation and independent audit;
- an immutable 48-logical-shard plan executed in 14 resumable model bundles;
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

It writes immutable logical-shard and physical-bundle plans, launches one
worker per requested GPU, and starts a finalizer. Each checkpoint is loaded
once and all of its registered modes run against that loaded engine. The
finalizer still merges only after all 48 logical shards complete and requires
exactly 161,280 unique request IDs.

For Purdue Anvil, use the separate Slurm adapter in
[`infra/anvil/realistic_niah_v3_1/`](../infra/anvil/realistic_niah_v3_1/README.md).
Its default interface launches eight independent one-H100 workers across two
four-GPU H100 nodes while preserving the same physical-bundle plan, atomic
checkpoint resume, clean-worktree requirement, and canonical final audit.

Individual stages can also be run with:

```bash
python scripts/prepare_realistic_niah_v3_1.py --run-root RUN_ROOT
python scripts/run_realistic_niah_v3_1.py --help
python scripts/run_realistic_niah_v3_1_model_bundle.py --help
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

The SciPy CPU backend is the confirmatory reference. A CUDA-enabled Torch
environment can run the same float64 likelihoods with:

```bash
python scripts/analyze_realistic_niah_v3_1.py \
  --run-root RUN_ROOT \
  --fit-backend torch \
  --analysis-device cuda \
  --run-lomo \
  --run-bootstrap-reselection
```

The output manifest records the backend and device. CUDA is an acceleration
backend, not a different statistical specification; SciPy/Torch parity checks
must pass before confirmatory use.

### Recommended compute split

- Freeze the 3,360 shared stimuli on the local CPU. This stage does not run an
  LLM and does not benefit materially from renting an accelerator.
- Use H100-class GPUs only for vLLM inference. The scheduler retains 48 logical
  model-mode shards for auditing but executes 14 physical model bundles, which
  avoids 34 redundant checkpoint loads. Atomic per-batch parts provide safe
  resume without repeatedly rewriting the growing canonical JSONL file.
- Run parsing, aggregation, table construction, plotting, and ordinary
  confirmatory SciPy fits on a local CPU or an inexpensive CPU machine.
- Use Torch/CUDA selectively when request-level Bernoulli fits and repeated
  likelihood optimization dominate analysis time. Small aggregated bias and
  Beta-Binomial problems may be faster on CPU because GPU launch and transfer
  overhead can exceed their arithmetic cost. An H100 should not be kept rented
  solely for post-processing without a timing pilot.

The CUDA path requires a CUDA-enabled PyTorch build; the Windows development
environment used to freeze the data has a CPU-only build. Therefore every new
GPU image must first pass the included SciPy/Torch parity test and a short
timing pilot before the 2,000-replicate confirmatory analysis is launched.

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
