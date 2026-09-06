# Anvil adapter: Realistic NIAH V3.3 long-context holdout

## Scope

This adapter evaluates Gemma4-31B and Qwen3-32B at immutable registered
revisions on the nine new 25k-100k passage lengths. It reuses the V3.1 N grid,
30 paired seeds, passage construction, direct prompt, native-thinking prompt,
parser, and decoding rules. Its protocol namespace and run root are separate
from V3.1 and V3.2, so the new requests cannot be mistaken for earlier results.

The formal request count is 7,560 per model and 15,120 total. Each model has a
two-request maximum-context preflight under `preflight/`; those four requests
are excluded from the formal total.

## Hardware layout

Each model has an independent formal job requesting one Anvil H100 node for 36
hours:

```text
GPUs                  4 x H100 80GB
Slurm tasks           2
GPUs per task         2
vLLM tensor parallel  2 per task
CPUs per task         24
memory                480G per node
```

Both workers in a model job are launched in one `srun` step. This avoids the
nested concurrent step contention that previously produced `Requested nodes
are busy`. Worker 0 runs 30k/60k/70k/100k; worker 1 runs
25k/40k/50k/80k/90k. Atomic batch parts support resume after a timeout or
preemption.

Observed V3.1 Gemma throughput projects approximately 19-27 hours after two-way
parallelization, before model-loading and filesystem overhead. Qwen runtime is
not inferred from Gemma throughput. The 36-hour request is a scheduling budget,
not a runtime guarantee.

## 1. Freeze and seal the dataset

Run this once in a sufficiently large CPU session. Do not run a long freeze on
an Anvil login node. The output is expected to be roughly one gigabyte and
should remain outside Git.

```bash
RUN_ROOT="$PROJECT/runs/realistic_niah_v3_3_long_context/run_YYYYMMDD"
REPO="$PROJECT/niah/repo-COMMIT"
mkdir -p "$RUN_ROOT/dataset"

cd "$REPO"
PYTHONPATH=src "$PROJECT/envs/$USER/niah-v31/bin/python" \
  scripts/freeze_realistic_niah_v3_3_long_context.py \
  --output-dir "$RUN_ROOT/dataset" \
  --haystack-dir "$REPO/data/haystacks/paul_graham" \
  --haystack-corpus-manifest \
    "$REPO/data/haystacks/paul_graham/corpus_manifest.json" \
  --cache-dir "$PROJECT/hf-cache"
```

The command performs a full tokenizer-level audit and then writes
`dataset_seal.json`. Record the printed `seal_sha256`. Formal submission
requires that exact digest and verifies every sealed file before allocating the
model runtime.

Before formal submission, also run the registered prompt audit (the supplied
`v3_3_long_context_freeze.slurm` job does both steps). It checks all 15,120
model-mode request IDs and exact one-message prompt contracts, then renders and
tokenizes all 420 maximum-length stimuli in both modes for each checkpoint. The
four model-mode maxima must each fit within `max_model_len=131072`. It saves the
four exact 100k/N=20/seed=1234 rendered prompts separately for inspection.

## 2. Warm the model cache

The following immutable assets must already be readable in the shared cache:

```text
google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475
Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137
Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218
```

Do not launch a model's two TP=2 workers while its checkpoint download is still
in progress. Qwen3-32B uses the official YaRN long-context configuration with
factor 4.0 and original maximum position embeddings 32,768; the override is
part of manifest compatibility.

## 3. Dry-run and submit

Use a clean repository frozen at an exact commit:

```bash
CODE_COMMIT="FULL_40_CHARACTER_COMMIT"
DATASET_SEAL_SHA256="FULL_64_CHARACTER_SHA256"

bash infra/anvil/realistic_niah_v3_3_long_context/submit_anvil.sh \
  "$RUN_ROOT" \
  --expected-commit "$CODE_COMMIT" \
  --dataset-seal-sha256 "$DATASET_SEAL_SHA256" \
  --dry-run

bash infra/anvil/realistic_niah_v3_3_long_context/submit_anvil.sh \
  "$RUN_ROOT" \
  --expected-commit "$CODE_COMMIT" \
  --dataset-seal-sha256 "$DATASET_SEAL_SHA256"
```

The command submits two independent jobs by default. Pass `--model Gemma4-31B`
or `--model Qwen3-32B` to submit only one group, including a checkpoint resume.

Each job refuses a dirty repository, a different commit, a changed dataset
seal, a non-H100 device, fewer than four GPUs, or an incompatible result
manifest. If a code fix changes the commit, resume requires the prior commit in
`--resume-from-commits`; the manifest schema, model, request IDs, stimuli,
engine, prompts, and checkpoint strategy must still match exactly.

## 4. Runtime gates and outputs

Worker 0 in each model job first evaluates the 100k/N=20/seed=1234 cell in
direct and native-thinking modes. Worker 1 waits for that model's atomic
preflight marker. Expected log gates include:

```text
V33_LONG_CONTEXT_TP2_PREFLIGHT
V33_LONG_CONTEXT_100K_PREFLIGHT_OK
```

Inspect:

```bash
squeue -j JOB_ID -o '%.18i %.12T %.10M %.6D %R'
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList
tail -f "$RUN_ROOT/orchestration/slurm/rniah-v33-lc-"*.out
tail -f "$RUN_ROOT/orchestration/worker_logs/Gemma4-31B-worker-"*.out
tail -f "$RUN_ROOT/orchestration/worker_logs/Qwen3-32B-worker-"*.out
```

The only completion criterion is:

```text
results/final_audit.json: passed=true
results/final_audit.json: requests=15120
results/final_audit.json: unique_request_ids=15120
results/_SUCCESS.json exists and matches the final audit
```

Inference completion does not by itself validate the frozen V3.2 law. The
confirmatory analysis must score the preregistered frozen predictions before
any exploratory refit uses the new observations.
