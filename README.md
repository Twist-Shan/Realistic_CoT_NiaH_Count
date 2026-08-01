# Realistic CoT NIAH Count

Reproducible experiments for studying exact counting in long
needle-in-a-haystack (NIAH) passages. The repository contains three related,
but scientifically distinct, research tracks:

1. **Realistic NIAH behavior experiments** compare direct answers,
   enumeration, and native reasoning across registered model checkpoints.
2. **Realistic NIAH V4 mechanisms** test non-thinking representations,
   answer-query attention, head ablation, and activation patching.
3. **Dynamic NIAH mechanism experiments** generate tokenizer-aware controlled
   examples for hidden-state, Q/K attention, probing, steering, ablation, and
   activation-restoration analyses.

The current registered behavior protocol is **Realistic NIAH V3**. The
registered mechanistic extension is **Realistic NIAH V4**, a two-model,
non-thinking analysis at 10k tokens. V4 separates four progressively relaxed
control panels (v4.1-v4.4), prompt-reading representations, answer-query
attention, and causal interventions. V3 remains behavior-only.

The completed V4 result separates where count information is visible from
what generation actually uses. A discovery-ranked span-end head bank is
causally necessary; one needle-end state is not sufficient for count
transport; late answer-query geometry is steerable; and exact answer-query
residual patching transfers the donor model prediction on 100% of eligible
Qwen final-layer rows and 99.58% of Gemma rows when strict-invalid outputs are
conservatively counted as failures.

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
| Run the Realistic NIAH V4 mechanism study | `scripts/freeze_realistic_niah_v4.py`, `scripts/run_realistic_niah_v4.py` | [`docs/realistic_niah_v4.md`](docs/realistic_niah_v4.md) |
| Inspect the completed V4 numeric non-thinking run | `run_20260731_v4_numeric_presentation_v3` | [`docs/realistic_niah_v4_numeric_results_20260731.md`](docs/realistic_niah_v4_numeric_results_20260731.md) |
| Open or rebuild the V4 representation + causal report | [`reports/realistic_niah_v4_representation_report.html`](reports/realistic_niah_v4_representation_report.html), `scripts/build_realistic_niah_v4_representation_report.py` | [`docs/realistic_niah_v4_causal_screen_20260801.md`](docs/realistic_niah_v4_causal_screen_20260801.md) |
| Audit the completed V4 causal screen | `scripts/audit_realistic_niah_v4_causal.py` | [`docs/realistic_niah_v4_causal_screen_20260801.md#screen-design-and-completion-audit`](docs/realistic_niah_v4_causal_screen_20260801.md#screen-design-and-completion-audit) |
| Audit and analyze exact answer-query transport | `scripts/analyze_realistic_niah_v4_answer_query_patching.py` | [`docs/realistic_niah_v4_causal_screen_20260801.md#3-late-answer-query-state-transports-the-computed-prediction`](docs/realistic_niah_v4_causal_screen_20260801.md#3-late-answer-query-state-transports-the-computed-prediction) |
| Analyze all Qwen span-end candidates and multi-head coverage | `scripts/analyze_realistic_niah_v4_partitioning.py` | [`docs/realistic_niah_v4_numeric_results_20260731.md`](docs/realistic_niah_v4_numeric_results_20260731.md#qwen-span-end-full-candidate-bank-and-positional-partitioning) |
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
│   ├── realistic_niah_v4/      # V4 freeze, capture, attention, causal, and audit logic
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

`src/realistic_niah_v4/` is the registered V4 package. It keeps controlled
stimulus freezing, exact prompt-span mapping, selective hooks, representation
statistics, attention scoring, and causal interventions separate. V4 outputs
belong under an external `runs/realistic_niah_v4/` root.

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

### Registered V4 mechanisms

V4 uses Transformers directly rather than vLLM because it requires selective
hidden-state hooks, query-row attention, head ablation, and activation
patching. Formal attention runs retain complete float16 answer-query rows
for every layer and head, together with separately generated greedy answers;
they never materialize a 10k-by-10k matrix or full-sequence Q/K/V:

```bash
python3 -m venv /path/to/venvs/realistic-niah-v4
. /path/to/venvs/realistic-niah-v4/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-mechanistic-v4.txt
export PYTHONPATH=src
```

The registered grid has two models (`Qwen3-8B`, `Gemma4-E4B`), count 1-10,
10,000 canonical passage tokens, and 30 seeds. Its four cumulative panels are:

V4 requests decimal answers `Total:1` through `Total:10`. The answer-query
state and attention row are measured at the final token of the rendered
`Total:` prefix. Correct/wrong labels come from a separate deterministic
greedy generation pass over the identical prompt, not a one-position
candidate softmax. This matters because decimal `10` is two tokens in both
registered tokenizers and shares its first token with `1`.

| Panel | Position | City-score order | City-score content |
| --- | --- | --- | --- |
| v4.1 | fixed | fixed | fixed |
| v4.2 | varied | fixed | fixed |
| v4.3 | varied | varied | fixed set |
| v4.4 | varied | varied | varied |

Freeze and audit the 1,200 shared stimuli before any model run:

```bash
PYTHONPATH=src python scripts/freeze_realistic_niah_v4.py \
  --config configs/realistic_niah_v4.json \
  --output-dir /path/to/runs/realistic_niah_v4/run_YYYYMMDD/dataset \
  --cache-dir /path/to/hf-cache
```

Then invoke `scripts/run_realistic_niah_v4.py` once per model and stage:
`preflight`, `behavior`, `representation-capture`, `representation-analyze`,
`attention`, and `attention-analyze`. Causal stages are `ablation`,
`patching` and `geometric-steering`. The `attention` stage
only captures restartable raw query rows; `attention-analyze` joins strict
greedy labels and runs span-end/span-mean broad-head, correct/wrong, and
omission-candidate diagnostics on CPU. Causal stages likewise score the actual
complete greedy numeric continuation, including multi-token `10`; they do not
use a first-token candidate softmax. Residual patching copies either the
needle-end state or the complete equal-length token-state sequence—never a
broadcast span mean. Geometric count centroids are fit on discovery seeds and
evaluated on held-out confirmation seeds with norm-matched random controls. See
[`docs/realistic_niah_v4.md`](docs/realistic_niah_v4.md) for exact estimands,
discovery/confirmation separation, formulas, commands, outputs, and
interpretation limits.

The default causal launcher runs the explicitly labelled `screen_8h_v1`
profile. It retains all four variants and all ten confirmation seeds, but
targets high-count failures: span-end answer-query top-4/top-8 head ablation on
counts 7--10, cumulative needle-end residual transport on pairs 5--6, 7--8,
and 9--10, and centroid-delta steering on 7--8, 9--10, and the non-local 5--10
pair at three matched relative depths. Every intervention keeps one
layer-matched or norm-matched random control. The reduced design is a causal
screen, not a replacement for a fully powered all-condition sweep.

Audit the completed `screen_8h_v1` causal subtree with:

```bash
PYTHONPATH=src python scripts/audit_realistic_niah_v4_causal.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --output /path/to/run/causal_screen_8h_audit.json
```

The completed `answer_query_dense_v1` follow-up uses eight single-layer
answer-query sites per model, all four panels, all ten confirmation seeds, and
directed pairs 5↔6, 7↔8, 9↔10, and 5↔10. Launch it restartably with
`scripts/launch_realistic_niah_v4_answer_query_patching.sh`; audit and analyze
the 5,120 strict-greedy rows with:

```bash
PYTHONPATH=src python scripts/analyze_realistic_niah_v4_answer_query_patching.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --output-dir /path/to/run/analysis/answer_query_patching_dense_v1 \
  --bootstrap-repetitions 20000
```

Build the self-contained V4 representation and causal report from a downloaded
run with:

```bash
PYTHONPATH=src python scripts/build_realistic_niah_v4_representation_report.py \
  --run-root /path/to/run_20260731_v4_numeric_presentation_v3 \
  --output reports/realistic_niah_v4_representation_report.html \
  --repo-root .
```

The report is the unified V4 result artifact and contains exactly five main
evidence blocks: (1) behavior, (2) prompt-reading and answer-query counter
representations, (3) answer-query attention representations, (4) head-bank
ablation, and (5) geometry steering. Experimental design, definitions,
mechanistic synthesis, limitations, and provenance are explicit preamble or
appendix material rather than extra result blocks. Every major subsection ends
with the strongest conclusion licensed by that evidence and the boundary that
remains untested.

The counter block includes an all-layer discovery sweep. It distinguishes the
`probe-optimal` layer, selected by grouped-seed full-space CV R², from the
`manifold-display` layer, selected only after a decodability gate using 3-PC
explained variance, count-centroid signal capture, and leave-one-seed-out
compactness. The Aurora 3D prompt counter exposes every captured post-block
layer, PC1--PC6, split, panel, and actual greedy outcome; a separate figure
shows answer-query counter geometry. Exact needle-end and exact answer-query
donor-state patching remain in this block because they test whether the two
representations are transportable.

The attention block now recomputes raw N=10 discovery profiles for both models.
It contains an all-head layer×head atlas, frozen rules for global broad,
partition-local broad, first-needle locator, other targeted retrieval,
span-mean-only breadth, and mixed candidates, a count-adjusted correct/wrong
comparison, and both omitted-tail and exact nested-new-needle diagnostics.
Machine-readable outputs are written beside the HTML as
`realistic_niah_v4_head_atlas.csv`,
`realistic_niah_v4_head_phenotypes.csv`, and
`realistic_niah_v4_attention_outcome_effects.csv`.

The two causal blocks report seed-cluster intervals for matched head-bank
ablation and centroid-delta steering. The report explicitly distinguishes
sample-wise full donor-state replacement (`h'=h_d`), the unrun centroid
transplant (`h'=mu_target`), and the completed full-dimensional centroid delta
(`h'=h+mu_target-mu_receiver`); the last establishes directional
manipulability, not full-state sufficiency. All five Gemma strict-invalid `11`
continuations remain in the conservative denominator rather than being
silently dropped. See
[`docs/realistic_niah_v4_causal_screen_20260801.md`](docs/realistic_niah_v4_causal_screen_20260801.md)
for the complete estimands, results, audit, and limitations.

V4 and later plots use the registered **Aurora** visual system: Midnight Indigo
`#23165C`, Polar Violet `#6750E8`, Ice Cyan `#00C2FF`, Aurora Yellow
`#F6E36A`, Aurora Teal `#00D4B4`, Aurora Green `#39E58C`, Polar Magenta
`#C04DFF`, Sunset Pink `#FF5FA2`, Night Black `#161923`, Snow White
`#F8FBFF`, Frost Gray `#8190A5`, and Warm Brown `#765347`. The report builder
is the source of truth for semantic color mappings and ordered count blends.

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

PYTHONPATH=src python -m pytest tests/test_realistic_niah_v4.py

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
