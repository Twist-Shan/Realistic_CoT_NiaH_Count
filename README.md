# Realistic CoT NIAH Count

Reproducible experiments for studying exact counting in long
needle-in-a-haystack (NIAH) passages. The repository contains two related,
but scientifically distinct, research tracks:

1. **Realistic NIAH behavior experiments** compare direct answers,
   enumeration, and native reasoning across registered model checkpoints.
2. **Dynamic NIAH mechanism experiments** generate tokenizer-aware controlled
   examples for hidden-state, Q/K attention, probing, steering, ablation, and
   activation-restoration analyses.

The current registered behavior protocol is **Realistic NIAH V3**. It covers
behavior comparison and empirical-law search only; it does not make
mechanistic or causal claims. Mechanistic questions belong to the Dynamic
NIAH analysis stack and require separate interventions and evidence.

The repository was consolidated non-destructively from
`NIAH_repo_and_local_runs_001` and `NIAH_repo_and_local_runs_002`. The original
instructions are preserved in
[`docs/README.upstream.md`](docs/README.upstream.md), and the exact merge
mapping is recorded in
[`provenance/MERGE_MANIFEST.md`](provenance/MERGE_MANIFEST.md).

Remote: <https://github.com/Twist-Shan/Realistic_CoT_NiaH_Count.git>

## Start here

| Goal | Primary entry point | Detailed specification |
| --- | --- | --- |
| Run or audit Realistic NIAH V3 | `scripts/freeze_realistic_niah_v3.py`, `scripts/launch_realistic_niah_v3.sh` | [`docs/realistic_niah_v3.md`](docs/realistic_niah_v3.md) |
| Inspect the executable V3 registry | `src/realistic_niah_v3/spec.py`, `configs/realistic_niah_v3.json` | `tests/test_realistic_niah_v3.py` |
| Reproduce the completed V2 protocol | `src/realistic_niah/spec.py`, `src/realistic_niah/sharding.py` | [`docs/realistic_niah_v2.md`](docs/realistic_niah_v2.md) |
| Run a V2 model extension | `src/realistic_niah/*_extension.py`, matching launch scripts | [`docs/realistic_niah_olmo3_extension.md`](docs/realistic_niah_olmo3_extension.md), [`docs/realistic_niah_reasoning_models_extension.md`](docs/realistic_niah_reasoning_models_extension.md) |
| Generate controlled Dynamic NIAH data | `scripts/generate_dynamic_niah_v2.py` | [`docs/README.upstream.md`](docs/README.upstream.md) |
| Analyze representations or causal interventions | `notebooks/`, `src/counting/`, `src/single_example/` | [`docs/research-plan.md`](docs/research-plan.md) |
| Classify attention-head candidates | `notebooks/attention_head_taxonomy.ipynb`, `scripts/classify_attention_heads.py` | [`docs/head_taxonomy.md`](docs/head_taxonomy.md) |

For executable request accounting, model support, prompt text, and immutable
checkpoint revisions, treat the Python registries, JSON configs, and tests as
the source of truth. Narrative documents explain the scientific design and
historical decisions.

## Repository architecture

```text
.
├── configs/                    # Registered experiment and analysis configs
├── data/                       # Small source corpora, entities, and templates
├── docs/                       # Protocols, research plans, and upstream docs
├── notebooks/                  # Thin exploratory and Colab launchers
├── plans/                      # Historical and active research plans
├── provenance/                 # Consolidation and source provenance
├── reports/                    # Curated reports and reusable report assets
├── scripts/                    # CLI, launch, worker, merge, audit, and sync tools
├── src/
│   ├── realistic_niah/         # Shared V2/V3 prompt, parser, runner, and freeze logic
│   ├── realistic_niah_v3/      # V3 registry, shards, audit, analysis, and reporting
│   ├── dataset_generation/     # Dynamic NIAH generation and hidden-state/QK analysis
│   ├── counting/               # Probes, CoT analysis, steering, and evaluation
│   └── single_example/         # Token/representation ablation and restoration
├── tests/                      # Unit, integration, snapshot, and smoke tests
├── dataset.schema.json         # Dynamic NIAH v2 JSONL row schema
├── requirements.txt            # Core Dynamic NIAH and development environment
├── requirements-inference.txt  # Historical V2/OLMo-compatible inference pins
├── requirements-inference-v3.txt
│                               # V3 inference pins
└── requirements-analysis.txt   # CPU V3 analysis environment
```

The main dependency direction is:

```text
data + configs
    │
    ├── Dynamic NIAH generator
    │      └── responses → hidden states/QK → probes/steering/ablation
    │
    └── frozen Realistic NIAH stimuli
           └── model × mode shards → merge/audit → behavior/law reports
```

Reusable logic belongs in `src/`. Scripts should remain command-line
orchestration layers, and notebooks should call tested functions rather than
become the only implementation of an analysis.

## Environments

The repository is imported with `PYTHONPATH=src`; it is not installed as a
single packaged project.

### Core development and Dynamic NIAH

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest
```

Bash:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest
```

### Registered V3 inference

Use a separate Linux GPU environment:

```bash
python3 -m venv /path/to/venvs/realistic-niah-v3
/path/to/venvs/realistic-niah-v3/bin/python -m pip install \
  -r requirements-inference-v3.txt
```

V3 currently pins `transformers==5.14.1` and `vllm==0.25.1`. Do not run V3
from the historical `requirements-inference.txt` environment: that file keeps
the older Transformers pin needed by the completed OLMo 3 extension.

### V3 analysis

```bash
python3 -m venv .venv-analysis
. .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
```

Inference and analysis are intentionally separate: the former is a
model-serving environment, while the latter is a smaller CPU statistical
environment.

## Registered Realistic NIAH V3

V3 freezes one shared Cartesian stimulus grid:

| Dimension | Registered values |
| --- | --- |
| Final passage length | `2k, 3k, 5k, 8k, 10k, 15k, 20k` canonical tokens |
| True record count | `1,2,3,4,5,6,7,8,9,10,12,15,18,20` |
| Paired seed | `1234..1243` |
| Shared stimuli | `7 × 14 × 10 = 980` |
| Needle depth | Inclusive 5%–95% of final-passage character position |
| Prompt modes | Direct, indexed enumeration, bullet enumeration, native thinking |

Ten switchable Qwen, Gemma, and Nemotron checkpoints receive all four modes.
The GLM and Ministral behavior slots use separate control and reasoning
checkpoints. Consequently V3 records 14 raw checkpoints and 12 behavior
comparison slots. The full plan contains:

```text
40 switchable-checkpoint shards
+ 4 GLM control/reasoning shards
+ 4 Ministral control/reasoning shards
= 48 shards × 980 requests = 47,040 requests
```

All checkpoints and tokenizer revisions are immutable Git SHAs. Request IDs
use the `v3/` namespace so V3 rows cannot be silently mixed with V2 outputs.

### Scientific outcomes

The primary accuracy is:

```text
1[predicted integer is parseable and equals the true count]
```

Every request remains in the denominator. Parse failures, wrong integers,
format failures, and truncations are retained rather than filtered.
Additional outputs include strict registered accuracy, parse and format rates,
enumeration precision/recall, signed deviation, absolute deviation, and
exclusive failure classes.

Mode comparisons use shared stimulus IDs, seed-clustered intervals, exact
McNemar tests, and Holm adjustment. Empirical-law candidates are evaluated
with five-fold cross-validation grouped by seed; every attempted formula is
retained even when no reliable unified law is found.

V3 supports behavioral statements about accuracy, bias, dispersion, and
held-out predictive response surfaces. It does **not** establish that visible
chain of thought is the internal algorithm, identify a causal attention head,
or justify extrapolation beyond the registered grid.

### End-to-end V3 workflow

Choose a run root under a `runs/realistic_niah_v3/` directory:

```bash
RUN_ROOT=/path/to/runs/realistic_niah_v3/run_YYYYMMDD
HF_CACHE=/path/to/hf-cache
```

Build and freeze the content-deduplicated Paul Graham corpus:

```bash
PYTHONPATH=src python scripts/sync_paul_graham_full_corpus.py \
  --output-dir "${RUN_ROOT}/source_corpus"

PYTHONPATH=src python scripts/freeze_realistic_niah_v3.py \
  --output-dir "${RUN_ROOT}/dataset" \
  --haystack-dir "${RUN_ROOT}/source_corpus" \
  --haystack-corpus-manifest \
    "${RUN_ROOT}/source_corpus/corpus_manifest.json" \
  --cache-dir "${HF_CACHE}"
```

The freeze command writes `stimuli.jsonl`, a manifest, contamination/cell
audits, checksums, and `audit_report.json`. It fails rather than silently
accepting a wrong grid, tokenizer revision, passage length, insertion depth,
duplicate source, or contaminated filler.

Launch the resumable GPU plan:

```bash
export REALISTIC_NIAH_REPO_ROOT="$PWD"
export REALISTIC_NIAH_PYTHON=/path/to/venvs/realistic-niah-v3/bin/python
bash scripts/launch_realistic_niah_v3.sh "${RUN_ROOT}" 8
```

The launcher validates the environment and clean Git state, prepares the
frozen shard plan, starts one worker per requested GPU, and starts a finalizer.
Workers atomically claim shards, checkpoint `requests.jsonl` after bounded
vLLM batches, preserve failed attempts, and resume only missing request IDs.
The finalizer merges only after all 48 completion markers exist and the full
47,040-row audit passes.

Generate the registered reports after the final audit:

```bash
PYTHONPATH=src python scripts/analyze_realistic_niah_v3.py \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/analysis/v3_behavior_empirical_law"
```

A completed run has the following high-level layout:

```text
RUN_ROOT/
├── source_corpus/
├── dataset/
│   ├── stimuli.jsonl
│   ├── manifest.json
│   └── audit_report.json
├── orchestration/
│   ├── formal_shards.json
│   ├── prepare_audit.json
│   ├── shard_state/
│   └── final_shard_audit.json
├── shards/<task_id>/main/
│   ├── requests.jsonl
│   └── run_manifest.json
├── models/<checkpoint>/main/
├── matched_controls/<checkpoint>/main/
├── matched_reasoning/<checkpoint>/main/
└── analysis/v3_behavior_empirical_law/
```

See [`docs/realistic_niah_v3.md`](docs/realistic_niah_v3.md) for the exact
prompts, estimands, candidate formula grid, checkpoint revisions, audits, and
interpretation boundary.

## Registered Realistic NIAH V2 and extensions

V2 remains a frozen, independently auditable protocol. Its shared grid has
five passage lengths, ten count values, and ten paired seeds, for 500 stimuli.
The executable plan is:

| Component | Mode shards | Requests | Notes |
| --- | ---: | ---: | --- |
| Six switchable primary models | 24 | 12,000 | Four modes per checkpoint |
| DeepSeek-R1-Qwen3 and GLM-Z1 | 2 | 1,000 | Registered native-thinking mode only |
| GLM-4 matched control | 3 | 1,500 | Direct/index/bullet |
| **V2 formal total** | **29** | **14,500** | 500 requests per shard |

This accounting is enforced by `src/realistic_niah/sharding.py`,
`configs/realistic_niah_main.json`, and the test suite.

Independently versioned additions reuse the frozen 500-stimulus V2 dataset
without modifying the completed formal panel:

| Addition | Shards | Requests | Documentation |
| --- | ---: | ---: | --- |
| V2.1 prompt-revision rerun | 15 | 7,500 | `src/realistic_niah/prompt_revision_v2_1.py` |
| OLMo 3 7B extension | 4 | 2,000 | [`docs/realistic_niah_olmo3_extension.md`](docs/realistic_niah_olmo3_extension.md) |
| Native-reasoning model extension | 20 | 10,000 | [`docs/realistic_niah_reasoning_models_extension.md`](docs/realistic_niah_reasoning_models_extension.md) |

V2 and its extensions reuse the shared prompt renderer, parser, vLLM runner,
atomic checkpointing, and model-specific thinking controls in
`src/realistic_niah/`. A change to those shared modules can affect multiple
protocols and therefore requires targeted regression tests.

Before any formal V2 launch, run the registered guarded-CoT smoke test and
require zero truncations. Production outputs and model caches belong outside
the Git checkout. Completed runs can be archived and verified with
`scripts/sync_run_to_gdrive.py`.

## Dynamic NIAH and mechanism analysis

Dynamic NIAH is the controlled experimental stack used for representation and
causal analyses. It is related to, but not interchangeable with, the frozen
Realistic NIAH behavior datasets.

The generator supports:

- tokenizer-aware exact-length haystacks;
- city-score, marker-count, literal-count, and related task variants;
- fixed, randomized token, sentence-boundary, or word-boundary insertion;
- aligned control replacements;
- deterministic seed derivation;
- saved token/character spans and a versioned JSONL schema.

Generate a baseline dataset:

```bash
PYTHONPATH=src python scripts/generate_dynamic_niah_v2.py \
  --config configs/niah_dynamic.json \
  --task-type match_count \
  --tokenizer Qwen/Qwen3-8B \
  --num-examples 100 \
  --target-haystack-tokens 1000 \
  --num-needles 3 \
  --positions 100 200 400 \
  --prompt-style vanilla
```

The main analysis layers are:

1. `scripts/gen_responses.py` for generation and response metrics;
2. `scripts/analyze_hidden_states.py` for controlled hidden-state comparisons;
3. `dataset_generation/qk_hook_attention/` for Q/K reconstruction, attention
   outliers, and head taxonomy;
4. `counting/feature_analysis.py` for ridge/classification probes,
   contrastive directions, and counterfactual count directions;
5. `counting/steering.py` for last-token and needle-span interventions;
6. `single_example/` for token ablation, representation ablation, and
   activation restoration;
7. `counting/cot_analysis.py` for full-sequence thinking/non-thinking
   comparisons.

The primary notebooks are:

- `analysis_hidden_states_v4.ipynb`;
- `single-example-v2.ipynb`;
- `counting_analysis.ipynb`;
- `counting_feature_analysis.ipynb`;
- `cot_analysis.ipynb`;
- `attention_head_taxonomy.ipynb`.

Detailed configuration precedence, row fields, cache rules, and notebook
outputs are documented in
[`docs/README.upstream.md`](docs/README.upstream.md).

## Data contracts and reproducibility

The repository treats experiment identity and provenance as part of the data:

- immutable model and tokenizer revisions;
- stable stimulus and request IDs;
- resolved configs saved beside outputs;
- Git commit and dirty-state checks for formal runs;
- package and hardware snapshots;
- atomic JSON/JSONL writes and resumable request checkpoints;
- SHA256 manifests for frozen datasets, merged outputs, and reports;
- explicit parse, format, and truncation failures;
- timing information for major pipeline stages.

Do not edit raw request rows, frozen stimuli, manifests, or completed release
directories in place. Derive a new version or analysis directory instead.

The generic Dynamic NIAH row contract is
[`dataset.schema.json`](dataset.schema.json). Realistic NIAH datasets and
request rows use separate versioned schemas embedded in their manifests.

## Testing

Run the complete CPU-compatible suite:

```bash
PYTHONPATH=src python -m pytest
```

Useful targeted checks:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_realistic_niah.py \
  tests/test_realistic_niah_runner.py \
  tests/test_realistic_niah_sharding.py

PYTHONPATH=src python -m pytest tests/test_realistic_niah_v3.py

PYTHONPATH=src python -m pytest \
  tests/test_dynamic_niah_v2_controls.py \
  tests/test_hidden_state_analysis.py \
  tests/test_counting_feature_analysis.py

python -m compileall src scripts
```

CPU tests validate configs, schemas, prompts, parsers, request accounting,
resume behavior, analysis statistics, and report rendering. They do not
replace the registered GPU smoke tests or full formal-run audits.

## Outputs, archives, and Git hygiene

Generated results, model caches, checkpoints, hidden-state dumps, attention
matrices, and release bundles should normally remain outside the Git checkout.
The repository ignores local `results/`, `outputs/`, `runs/`, `exports/`,
`artifacts/`, `archive/`, and common model/archive formats.

Local retained material has the following roles:

- `artifacts/`: experiment runs retained from the source bundles;
- `archive/`: environments, source metadata, and read-only provenance;
- `exports/`: local/Drive release bundles;
- `versions/`: immutable historical releases;
- `reports/`: curated, reviewable summaries and reusable analysis products.

Do not delete a large artifact merely because it is ignored. Verify its
manifest, checksums, canonical request rows, configs, and report dependencies
before moving or removing it.

## Interpretation standard

Evidence should be described at the level supported by the analysis:

- behavioral accuracy and paired comparisons establish performance
  differences on the registered grid;
- probes establish decodability, not causal use;
- attention patterns identify candidate routing, not transported content;
- ablation establishes a form of necessity but may introduce broad damage;
- activation patching, restoration, and localized steering provide stronger
  evidence about a proposed mechanism;
- mechanistic or causal conclusions require multiple controls and should not
  be inferred from the V3 behavior reports alone.

Correctness, reproducibility, and auditability take priority over maximizing
the number of runs or hiding negative results.
