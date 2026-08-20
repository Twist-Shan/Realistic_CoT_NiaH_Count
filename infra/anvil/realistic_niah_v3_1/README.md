# Anvil Slurm adapter for Realistic NIAH V3.1

## Purpose

This adapter runs the preregistered V3.1 empirical-law inference campaign on
Purdue Anvil without changing the frozen grid, model revisions, generation
settings, or output schema. It changes only scheduling and device binding.

The default launch uses eight independent one-H100 workers. Anvil H100 nodes
provide four GPUs, so an eight-worker launch requests two nodes and places four
Slurm tasks on each node. This is bundle-level parallelism: each worker loads
one model at a time with tensor parallel size 1. It is not an eight-way
tensor-parallel model launch.

## Required layout

The defaults follow the Anvil project guide:

```text
$PROJECT/niah/                                  repository
$PROJECT/envs/$USER/niah-v31/bin/python         inference Python
$PROJECT/hf-cache/                              durable model cache
$PROJECT/runs/realistic_niah_v3_1/RUN_NAME/     durable run root
```

The run root must already contain the audited frozen dataset:

```text
dataset/stimuli.jsonl
dataset/manifest.json
dataset/audit_report.json
dataset/source_revision.json
```

The repository must be clean. The environment must contain compatible pinned
builds of PyTorch, Transformers, and vLLM. Warm the Hugging Face cache before a
formal launch so eight workers do not download large checkpoints concurrently.
The Slurm adapter disables vLLM's optional FlashInfer sampler and uses the
native sampler, avoiding a runtime JIT dependency on Anvil's system `nvcc`.

The generic transfer example in the Anvil account guide excludes `.git`; that
example cannot be used unchanged for this formal workflow. Clone the repository
on Anvil, or transfer it with `.git` intact, because preparation records the
exact commit and refuses a dirty or non-Git source tree.

Install the registered V3 inference environment rather than the historical
OLMo-compatible inference environment:

```bash
module load modtree/gpu
module load conda

ENV_DIR="$PROJECT/envs/$USER/niah-v31"
conda create --prefix "$ENV_DIR" python=3.11 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
python -m pip install -r "$PROJECT/niah/requirements-inference-v3.txt"
```

That requirements file is the repository's authoritative inference pin; it
currently selects the V3-compatible Transformers and vLLM versions.

## Eight-GPU submission

From the repository root on an Anvil login node:

```bash
RUN_ROOT="$PROJECT/runs/realistic_niah_v3_1/run_YYYYMMDD"
CODE_COMMIT="9cecb34853c04457e285b42d65ccec6a2b0fcb24"

bash infra/anvil/realistic_niah_v3_1/submit_anvil.sh \
  "$RUN_ROOT" \
  --workers 8 \
  --expected-commit "$CODE_COMMIT"
```

Resolved defaults:

```text
account            mth260088-ai
partition          ai
constraint         H100
workers            8
nodes              2
tasks per node     4
GPUs per task      1
CPUs per worker    12
memory per node    480G
wall time          48:00:00
```

Confirm the account and partition with `mybalance`, `showpartitions`, and
`sfeatures` before the first submission. To inspect the exact command without
submitting it, add `--dry-run`.

## Pilot and overrides

A one-GPU pilot uses the same interface and therefore exercises the same Slurm
binding path as the formal run:

```bash
bash infra/anvil/realistic_niah_v3_1/submit_anvil.sh \
  "$RUN_ROOT" \
  --workers 1 \
  --time 02:00:00 \
  --mem-per-node 120G \
  --expected-commit "$CODE_COMMIT"
```

The interface accepts 1-12 workers. At most four tasks are placed on one H100
node. Common site or environment overrides can be supplied as flags or
environment variables; run `submit_anvil.sh --help` for the full list.

## Scheduling and resume semantics

Each Slurm task is bound to exactly one GPU and enters the existing atomic
bundle-claim loop. Eight workers dynamically consume the 14 physical model
bundles, preserving checkpoint-part resume and avoiding redundant checkpoint
loads within a bundle.

Claims record the Slurm job ID and worker ID. A worker never steals a claim
belonging to a currently queued or running Slurm job, including a worker on a
different node. After an interrupted job has left the queue, resubmitting the
same run root archives its stale incomplete claims and resumes from the saved
batch parts. Do not submit two campaigns against the same run root.

After all workers exit successfully, the batch script runs the canonical merge
and requires 48 completed logical shards and 161,280 unique request IDs. Slurm
stdout, bundle logs, worker state, and a job provenance snapshot are stored
below `RUN_ROOT/orchestration/`.

## Monitoring

```bash
squeue -u "$USER"
scontrol show job JOB_ID
wait_time -j JOB_ID
tail -f "$RUN_ROOT/orchestration/slurm/rniah-v31-"*.out
```

After completion:

```bash
jobinfo JOB_ID
seff JOB_ID
jobsu JOB_ID
mybalance
```

Inference may exceed a single 48-hour allocation. In that case, submit the same
command with the same run root; completed bundles are skipped and incomplete
bundles resume from atomic batch-part checkpoints.
