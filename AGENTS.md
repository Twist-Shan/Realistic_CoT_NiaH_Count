# AGENTS.md

## Purpose

This repository supports LLM and statistical research, including inference, finetuning, dataset generation, representation analysis, hidden-state/attention analysis, and statistical evaluation.

Prioritize correctness, reproducibility, and auditability over speed or cleverness. Prefer a small, tested workflow that can be rerun over a large notebook or script that only works once.

## Working principles

- Make the scientific goal explicit before coding: prediction target, estimand, comparison, dataset split, metric, intervention, or hypothesis.
- Follow the existing repo structure and conventions. Do not introduce a new architecture unless it clearly simplifies the project.
- Put reusable logic in Python modules, usually under `src/` or the existing package directory.
- Use notebooks only for exploration, diagnostics, and reporting. Notebooks should call tested functions rather than contain core logic.
- Prefer simple, readable implementations with checks over clever abstractions.
- Never silently ignore failed imports, failed tests, missing files, shape mismatches, NaNs, empty result sets, or suspiciously small outputs.
- Do not print, save, or expose secrets such as API keys, Hugging Face tokens, W&B keys, or private dataset credentials.

## Repository structure

Use the existing project layout if one already exists. If starting from scratch, prefer a structure like:

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── plans/
├── skills/
├── configs/
├── data/
│   ├── raw/          # immutable inputs; do not edit in place
│   ├── processed/    # generated data; should be reproducible
│   └── samples/      # tiny synthetic/anonymized test data
├── notebooks/        # exploration and reports only
├── scripts/          # CLI entry points and one-off runners
├── src/              # reusable package code
├── tests/            # unit, integration, and smoke tests
└── outputs/          # generated artifacts; normally git-ignored
```

Most human edits are in the folder `plans`. Skill files are in the folder `skills`. Run outputs should be saved in explicit run directories, for example:

```text
outputs/run_YYYYMMDD_HHMMSS_model_task/
  config.json
  logs.txt
  figures/
  tables/
  tensors/
  metrics/
```

Do not commit large outputs, checkpoints, cached models, hidden-state dumps, attention matrices, or generated artifacts unless explicitly requested.

For future coding, respect the naming convention, high-level workflow, and I/O controls documented in `README.md`, especially the separation between timestamped run folders and stable reusable caches.

## Colab and Google Drive workflow

Large GPU experiments may run in Colab or another cloud GPU service. Keep Colab notebooks as thin launchers that mount Drive, clone or pull the repo, install dependencies, and run scripts.

Avoid writing many small files directly to Google Drive during runtime. Prefer writing locally under `/content/`, then compressing results into a single `.zip` or `.tar.gz` file before copying to Drive.

Avoid hiding multiple steps into a big subprocess block in Colab notebook, since intermediate print/log info won't be displayed, which makes debugging hard.

Do not hard-code Google Drive, Colab, or local machine paths inside library code. Pass paths such as `data_dir`, `cache_dir`, `output_dir`, and `model_cache_dir` through configs or CLI arguments.

## Environment and dependencies

Use the dependency convention already present in the repo. If none exists, document the setup in `README.md`.

Preferred install commands, depending on the repo:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

or:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When a dependency is missing:

1. Decide whether it is required for the current task.
2. If required, add it to the appropriate dependency file or document the install command.
3. If optional, skip only the tests that genuinely require it using `pytest.importorskip` or an explicit pytest marker.
4. Report dependency problems separately from code or test failures.

## Planning before coding

For nontrivial tasks, first state a short plan. Include assumptions, affected files, validation strategy, and any risk to experimental correctness.

If the task is ambiguous, make the smallest reasonable assumption and state it. Ask for approval only when the ambiguity could lead to destructive or scientifically misleading changes.

If the task may generate many files, large files, or expensive GPU runs, describe the proposed output structure and what should remain uncommitted.

If validation is incomplete, say so explicitly. Do not imply that code is scientifically validated just because it runs.

If I ask you to read a plan file, be sure to read the "General instructions" part at the start of the plan.

## LLM inference and experiment code

When writing LLM inference or evaluation code:

- Separate model loading, tokenizer loading, prompt construction, generation, decoding, metric computation, and serialization.
- Expose key settings through config files or CLI arguments: model name/path, tokenizer name/path, cache directory, device, dtype, batch size, max tokens, decoding parameters, seed, and output directory.
- Detect available devices, but do not assume GPU. Support CPU-compatible smoke tests.
- Use `model.eval()` for inference.
- Be explicit about padding side, attention masks, causal masks, prompt boundaries, and answer boundaries.
- Be careful with dtype conversions. For example, use `.float()` when reductions or numerical comparisons should be performed in fp32.
- Avoid storing full hidden states, attention matrices, or logits for large runs unless required. Prefer streaming summaries or selective saving.

## Configuration

Prefer a typed configuration object, such as a dataclass, when the experiment has multiple settings.

Config behavior should be explicit:

- Load defaults from the config file when one is provided.
- Let CLI arguments override config-file values.
- Save the resolved config next to outputs, preferably as `config.json`.
- Include all important hyperparameters, paths, seeds, and I/O controls in the saved config.

## Reproducibility and logging

For experiments and analyses, save enough information to reproduce or audit the run:

- command used
- config file path and resolved config
- random seeds
- package versions
- hardware/device information when relevant
- model/tokenizer identifiers and cache paths
- output directory
- major runtime durations

Use deterministic settings where reasonable, but do not hide that some GPU operations may remain nondeterministic.

When using `subprocess.run`, preserve useful debugging information. Prefer capturing stdout/stderr, reporting return codes, and saving logs when running long jobs.

Always add a timer to record the running time of major code blocks or components in a pipeline. Save the running time info in a logging file.

## Testing and validation

Before finishing a change, run the smallest relevant checks available in the repo. Examples:

```bash
python -m compileall src
python -m pytest
python -m src.run_experiment --config configs/smoke.yaml
```

If these exact commands do not exist, inspect the repo and use the closest available smoke test. If no smoke test exists and the task changes core behavior, add a small one when reasonable.

For GPU-only changes, still add or run CPU-compatible checks for imports, configs, data loading, tensor shapes, and small mock inputs when possible. Clearly state what still needs Colab/cloud GPU validation.

## README.md expectations

Keep `README.md` useful for a new reader. When a change affects usage, update the README.

A good README should include:

1. a short project summary,
2. quick-start commands,
3. repository structure,
4. environment setup,
5. configuration explanation,
6. how to run core scripts,
7. where outputs are saved,
8. how to interpret main results, if applicable.

Do not over-expand the README with obsolete experiments or long internal notes. Move outdated details to `docs/` or an archive when appropriate.

## Final response expectations

When finishing a task, summarize:

1. what changed,
2. which files were modified,
3. what tests or checks were run,
4. what still requires GPU or larger-scale validation,
5. any risks, limitations, or assumptions.
