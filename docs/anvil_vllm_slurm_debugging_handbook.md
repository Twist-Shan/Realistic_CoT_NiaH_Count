# Debugging vLLM Inference on Purdue Anvil with Slurm

> Last updated: 2026-08-24
>
> Intended audience: researchers and coding agents maintaining multi-GPU vLLM inference jobs on Purdue Anvil.
>
> Validated environment: Python 3.11, Transformers 5.14.1, vLLM 0.25.1, and CUDA Toolkit 12.8.0 on H100 nodes.
>
> Important: version-specific conclusions in this document must be revalidated after changing vLLM, PyTorch, CUDA, the model revision, or the cluster environment.

## How to use this document

This file serves two purposes:

1. It explains the failures we actually encountered, so a human researcher can understand what went wrong and why.
2. It gives a coding agent an operational contract for diagnosing and repairing a similar pipeline without repeatedly introducing new bugs.

If you are handing this work to Codex, attach this file and use the copy-paste prompt in [Codex handoff prompt](#codex-handoff-prompt). The agent should still inspect your repository and live Slurm logs; this document is a debugging guide, not proof that your pipeline has exactly the same problem.

## Executive summary

The most important lessons are:

1. **PyTorch seeing the GPU does not prove that the vLLM environment is correct.** vLLM may separately invoke `nvcc`, Ninja, JIT extensions, FlashInfer, and dynamically linked CUDA libraries.
2. **A correct environment in the batch shell does not prove that an `srun` worker has the same environment.** Restricted Slurm exports can silently omit variables added after submission.
3. **Tensor-parallel size must equal the number of GPUs actually bound to the worker.** Verify this inside the worker using `CUDA_VISIBLE_DEVICES`; do not infer it from the total job allocation.
4. **Do not solve a structurally insufficient memory budget by pushing `gpu_memory_utilization` arbitrarily close to 1.0.** Startup memory fluctuates. Use tensor parallelism when a model has no stable single-GPU margin.
5. **Successful weight loading and KV-cache initialization are not sufficient smoke tests.** A job is not healthy until at least one real request completes.
6. **Checkpoint compatibility must be semantic and strict.** Controlled normalization is appropriate for newly added fields whose historical value is unambiguously the default. It is not appropriate for model revisions, prompts, request IDs, engine settings, or other experimental invariants.
7. **A running batch allocation does not imply that all internal steps are running.** Inspect `squeue -s`; an allocation may hold four or eight GPUs while only one TP=2 step is active.
8. **Simpler Slurm layouts are usually safer.** Separate jobs with atomic work claims are easier to verify than a complicated mixture of TP=2 and TP=1 steps inside one allocation.

## Non-negotiable invariants

Before debugging, identify the experiment's frozen invariants. For our formal run, these included the dataset, exact model revisions, prompts, ordered request IDs, maximum context length, decoding settings, and total request count.

A repair may change infrastructure or runtime plumbing, for example:

- shared script paths;
- CUDA module and environment initialization;
- `srun --export` contents;
- GPU binding and tensor-parallel scheduling;
- checkpoint locking and atomicity;
- narrowly justified vLLM stability flags;
- diagnostics, preflight checks, and tests.

A repair must not silently change scientific design, for example:

- model ID or revision;
- prompt text or chat template;
- dataset, stimuli, seeds, or request IDs;
- `max_model_len` or context layout;
- decoding semantics;
- output interpretation;
- completed results that have already passed integrity checks.

If the only apparent fix changes a scientific invariant, stop and ask the researcher instead of applying it automatically.

## Incident history and verified fixes

| Job | Symptom | Root cause | Fix and lesson |
| --- | --- | --- | --- |
| `20052052` | Preparation failed immediately | `prepare_realistic_niah_v3_1.py` did not import `PROTOCOL_VERSION` | Add the explicit import and run a real preparation smoke test before formal submission. Relevant fix: `3c5d2cf`. |
| `20058020` | Multi-node worker failed with exit code 127 | The worker entry point resolved to `/var/spool/slurm/job.../run_slurm_task.sh`, which was not visible from the other compute node | Launch workers from an absolute path on the shared GPFS repository. Relevant fix: `1657fb9`. |
| `20066166` | FlashInfer JIT reported unsupported `compute_90a` | JIT found CUDA 11.2.2 `nvcc`, which cannot compile H100 `sm_90a` code | Pin CUDA 12.8, prepend its `bin` directory to `PATH`, and compile a real `compute_90a` kernel during preflight. Relevant fixes: `d50a96e`, `62346ff`. |
| `20074871` | JIT-generated `sampling.so` could not load `libcudart.so.12` | Compilation used the right toolkit, but the dynamic linker could not find the CUDA 12 runtime | Add `$CUDA_HOME/lib64` to `LD_LIBRARY_PATH` and test `ctypes.CDLL("libcudart.so.12")`. Relevant fix: `f7b0bd7`. |
| `20075769` | Parent provenance showed CUDA 12.8, but workers still could not load CUDART | Restricted `sbatch`/`srun` exports did not propagate CUDA variables added at runtime | Explicitly include `CUDA_HOME`, `CUDA_PATH`, `LD_LIBRARY_PATH`, and vLLM-related variables in every `srun --export`. Repeat the CUDART test in each worker. Relevant fix: `549c235`. |
| `20077277` | Gemma4-31B lacked enough KV cache at `max_model_len=32768` | Its weights left insufficient single-GPU memory for the required 32k context | Increasing utilization to 0.995 was tested but was not a robust solution. |
| `20081937` | vLLM startup rejected the GPU: about 78.66 GiB was free but utilization 0.995 required about 78.78 GiB | Startup memory fluctuated below an extremely tight threshold | Move Gemma4-31B to TP=2 and return its per-GPU utilization target to 0.92. Keep other models TP=1. Relevant fix: `d4ba491`. |
| `20085073` | TP=2 weight loading and KV-cache initialization succeeded, but CUDA Graph capture caused `illegal memory access` | The graph-capture or custom all-reduce path was unstable for this Gemma/vLLM/CUDA combination | For Gemma only, use `enforce_eager=True` and `disable_custom_all_reduce=True`, falling back to eager execution and NCCL. Relevant fix: `2f465c2`. |
| `20091125` | General workers refused to resume because results were considered incompatible | Historical manifests lacked newly introduced Boolean fields and recorded an older Git commit | Normalize only explicitly approved additive default fields; require an allowlist of full Git SHAs for cross-commit resume; keep all scientific fields strict. Relevant fixes: `7d66bb9`, `939410e`. |
| Mixed four/eight-GPU layouts | The batch job held the requested node, but additional `srun` steps reported `Requested nodes are busy`; four allocated GPUs sometimes ran only one TP=2 step | Step-level CPU, memory, exclusivity, or GPU TRES declarations did not permit the intended concurrent layout | Verify internal steps using `squeue -s` and a resource-binding smoke test. Prefer separate Slurm jobs when possible. |

### Interpreting common Slurm outcomes

- **Exit code 127:** normally means an executable or script was not found. Check absolute paths, the shared filesystem, and `PATH`.
- **Exit code 143:** the process received `SIGTERM`. This is often secondary: one worker failed first, then `--kill-on-bad-exit=1` or the parent script terminated the rest. Find the earliest actual exception.
- **`DependencyNeverSatisfied`:** an upstream job did not complete in the state required by `afterok`. Inspect the dependency and upstream terminal state; this is not inherently a vLLM error.
- **`PENDING (Priority)` or `PENDING (Resources)`:** normal queueing, not a software failure.
- **`Requested nodes are busy`:** often a job-step problem inside an allocation. The batch job owns the node, but a new step cannot acquire the CPU, memory, or GPU TRES it declared.

## A disciplined debugging workflow

Follow this order. It prevents later symptoms from obscuring earlier causes.

### 1. Freeze and record the state

Before editing anything, record:

```bash
git rev-parse HEAD
git status --short
scontrol show job "$JOB_ID"
sacct -j "$JOB_ID" -X \
  -o JobID,JobName,State,Start,End,Elapsed,ExitCode,NodeList
```

Preserve the original logs and manifests. Do not delete claims, checkpoints, or output directories to make an error disappear.

### 2. Find the first real failure

Search the main log and worker logs in chronological order:

```bash
rg -n -i \
  'traceback|runtimeerror|error:|oom|out of memory|illegal memory|libcudart|compute_90a|unsupported gpu|worker failure' \
  "$RUN_ROOT/orchestration/slurm" \
  "$RUN_ROOT/orchestration/logs"
```

Do not stop at the final exit code. The root cause is usually the earliest specific error.

### 3. Separate job state from step state

```bash
squeue -j "$JOB_ID"
squeue -s -j "$JOB_ID"
scontrol show job "$JOB_ID"
sacct -j "$JOB_ID" -X \
  -o JobID,State,Start,End,Elapsed,ExitCode,NodeList
```

The job can be `RUNNING` while an intended worker step never started. Confirm the number of live steps and their resource bindings.

### 4. Verify the CUDA toolchain on a compute node

Do not use a login-node result as proof of the compute-node environment.

```bash
command -v nvcc
nvcc --version
command -v ninja
test -e "$CUDA_HOME/lib64/libcudart.so.12"

printf '__global__ void h100_preflight(){}\n' \
  | "$CUDA_HOME/bin/nvcc" -x cu \
      -gencode=arch=compute_90a,code=sm_90a \
      -c -o /dev/null -

"$PYTHON_BIN" -c \
  'import ctypes; ctypes.CDLL("libcudart.so.12"); print("CUDART_LINK_OK")'
```

`nvcc --version` alone is not sufficient. A real `compute_90a` compilation detects an older compiler that merely happens to be first on `PATH`.

### 5. Verify the environment again inside every worker

Print and validate at worker startup:

```bash
printf 'host=%s procid=%s localid=%s visible=%s cuda_home=%s\n' \
  "$(hostname)" \
  "${SLURM_PROCID:-}" \
  "${SLURM_LOCALID:-}" \
  "${CUDA_VISIBLE_DEVICES:-}" \
  "${CUDA_HOME:-}"

test -n "${CUDA_VISIBLE_DEVICES:-}"
test -n "${CUDA_HOME:-}"
test -n "${LD_LIBRARY_PATH:-}"
test -e "$CUDA_HOME/lib64/libcudart.so.12"
"$PYTHON_BIN" -c 'import ctypes; ctypes.CDLL("libcudart.so.12")'
```

### 6. Prove GPU binding equals tensor parallelism

For TP=2:

```bash
srun --exact --exclusive \
  --nodes=1 --ntasks=1 --cpus-per-task=24 \
  --gpus-per-task=2 --gpu-bind=per_task:2 \
  ...
```

Inside the worker, validate all three quantities:

```text
number of CUDA_VISIBLE_DEVICES
= Slurm GPUs per task
= vLLM tensor_parallel_size
```

### 7. Require a real inference smoke test

A smoke test should demonstrate all of the following:

- the model revision loaded from the expected cache;
- vLLM used the expected engine configuration;
- the intended number of GPUs joined the TP world;
- KV-cache initialization succeeded;
- at least one real, representative request completed;
- an output part was written and passed readback validation.

## Correct CUDA and module initialization

### Module initialization order

Anvil's module bootstrap can read optional variables that are initially unset. A safe order is:

```bash
#!/usr/bin/env bash
set -eo pipefail
source /etc/profile.d/modules.sh
set -u

module --force purge
module load modtree/gpu
module load conda
```

Do not enable `set -u` before sourcing the module initialization script.

### Pin the toolkit explicitly

```bash
PYTHON_BIN="$PROJECT/envs/$USER/niah-v31/bin/python"
ENV_BIN="$(dirname "$PYTHON_BIN")"

export CUDA_HOME=/apps/anvilgpu/external/apps/cuda-toolkit/12.8.0
export CUDA_PATH="$CUDA_HOME"
export PATH="$ENV_BIN:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
```

These variables solve different problems:

- `PATH` selects the actual `nvcc`, Ninja, and environment executables;
- `CUDA_HOME` and `CUDA_PATH` guide extension build systems;
- `LD_LIBRARY_PATH` lets the dynamic linker resolve `libcudart.so.12` when loading generated `.so` files.

### Export runtime variables explicitly to `srun`

```bash
runtime_exports="HOME,USER,PATH,SHELL,CUDA_HOME,CUDA_PATH,LD_LIBRARY_PATH,VLLM_USE_FLASHINFER_SAMPLER,REALISTIC_NIAH_REPO_ROOT,REALISTIC_NIAH_PYTHON,REALISTIC_NIAH_HF_CACHE"

srun ... \
  --export="$runtime_exports,REALISTIC_NIAH_GPUS_PER_TASK=2,REALISTIC_NIAH_TENSOR_PARALLEL_SIZE=2" \
  bash "$SHARED_REPO/infra/anvil/.../run_slurm_task.sh" "$RUN_ROOT"
```

Do not assume that `export CUDA_HOME=...` in the batch shell will pass through a restricted `--export` list.

## Slurm layout rules

### Worker scripts must live on shared storage

Do not launch a cross-node worker from a node-local Slurm spool path:

```bash
# Wrong for a multi-node step
srun bash "/var/spool/slurm/job${SLURM_JOB_ID}/run_slurm_task.sh"
```

Use an absolute path on shared project storage:

```bash
TASK_SCRIPT="$PROJECT/niah/infra/anvil/realistic_niah_v3_1/run_slurm_task.sh"
test -r "$TASK_SCRIPT"
srun ... bash "$TASK_SCRIPT" "$RUN_ROOT"
```

The repository, run root, and model cache must remain visible from every participating node.

### Smoke-test concurrent steps before relying on a mixed layout

If a four-GPU node is intended to run one TP=2 step and two TP=1 steps concurrently, first run a short allocation that occupies the declared resources and verify:

```bash
squeue -s -j "$SLURM_JOB_ID"
scontrol show step "$SLURM_JOB_ID.STEP_ID"
nvidia-smi
```

All three steps must be simultaneously `RUNNING`, each must see the intended GPUs, and their CPU and memory declarations must fit within the allocation.

For production, two independent jobs are often safer:

- a model-specific TP=2 job;
- a general TP=1 worker-pool job;
- a shared run root with atomic bundle claims;
- a locked, exactly-once finalizer after all groups succeed.

This is easier to reason about and less likely to waste GPUs than a complex internal step scheduler.

## A stable Gemma4-31B configuration from this incident

The following configuration was validated for this specific frozen environment:

```python
EngineConfig(
    tensor_parallel_size=2,
    request_batch_size=1,
    max_num_seqs=1,
    max_model_len=32768,
    gpu_memory_utilization=0.92,
    enforce_eager=True,
    disable_custom_all_reduce=True,
    dtype="bfloat16",
    enable_prefix_caching=True,
)
```

Why these settings were used:

- TP=2 provides stable memory headroom for the required 32k context;
- `enforce_eager=True` avoids the CUDA Graph capture path that caused an illegal memory access;
- `disable_custom_all_reduce=True` uses NCCL instead of vLLM's custom all-reduce path;
- utilization 0.92 leaves startup margin instead of relying on a fragile 0.995 threshold;
- `request_batch_size=1` and `max_num_seqs=1` limit peak scheduling pressure.

Validate that the installed vLLM version supports the intended arguments:

```python
import dataclasses
from importlib.metadata import version
from vllm.engine.arg_utils import EngineArgs

assert version("vllm") == "0.25.1"
fields = {field.name for field in dataclasses.fields(EngineArgs)}
required = {
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "enforce_eager",
    "disable_custom_all_reduce",
}
assert required <= fields
print("VLLM_TP2_PREFLIGHT_OK")
```

Do not copy this configuration blindly to another vLLM version. Parameter names, defaults, kernels, and graph-capture behavior may differ.

## Checkpointing, work claims, and manifests

### What the current checkpoint protocol guarantees

Each completed batch is written to a separate part:

```text
SHARD/main/request_parts/batch_<request-id-digest>.jsonl
```

The intended protocol is:

1. write a temporary file;
2. atomically replace it with the final part;
3. on restart, read the canonical output and all parts;
4. deduplicate by `request_id`;
5. generate only missing requests;
6. perform one canonical merge;
7. read back and verify the merged file before cleaning parts.

This supports safe restart of one worker. It does **not** automatically permit two ordinary workers to generate the same shard concurrently. Concurrent workers on the same shard may produce duplicate IDs, conflicting outputs, competing manifest updates, and racing canonical merges.

To parallelize within one shard safely, first implement deterministic request partitioning, isolated output namespaces, and an audited merge. Do not simply launch a second worker against the same shard.

### Atomic bundle claims

Workers should claim whole bundles through an atomic operation such as:

```text
mkdir claims/BUNDLE_ID
```

The claim should record at least:

```text
bundle_id, worker_id, pid, hostname, scheduler_job_id,
claimed_at_utc, attempt_id
```

Do not steal a claim when:

- its Slurm job is still visible in `squeue`;
- its local PID is alive;
- an old cross-node claim cannot be proven stale.

Archive a stale attempt and resume only after its owner is demonstrably terminal. Do not use `rm -rf claims/...` as a routine recovery method.

### Semantic manifest compatibility

Strictly compare:

- schema and protocol versions;
- model label, model ID, and exact revision;
- query layout and prompt mode;
- stimuli SHA256 and ordered request-ID SHA256;
- full engine configuration and model-specific overrides;
- prompt-payload storage mode;
- checkpoint strategy;
- clean Git state.

In this incident, only two later-added Boolean fields could be normalized for historical manifests:

```python
{
    "enforce_eager": False,
    "disable_custom_all_reduce": False,
}
```

This was safe only because the old code's behavior was unambiguously equivalent to those defaults. Do not generalize this into a permissive manifest comparison.

Cross-commit resume should require an explicit allowlist of audited full SHAs:

```bash
export REALISTIC_NIAH_RESUME_FROM_COMMITS="OLD_FULL_SHA[:ANOTHER_FULL_SHA]"
```

Do not accept short SHAs, branch names, arbitrary historical commits, dirty worktrees, or unknown manifests.

### Exactly-once finalization

Each group should atomically write its completion marker. The last successful group should acquire an exclusive lock:

```bash
flock -x FINALIZE_LOCK_FD
```

The finalizer may run only after every required group has passed. A filtered prepass must never run the global finalizer.

Slurm `COMPLETED` is not the scientific completion condition. The final audit must verify expected shards and exact request identity, for example:

```json
{
  "passed": true,
  "requests": 161280,
  "unique_request_ids": 161280
}
```

## Submission checklist

### Repository and experiment state

- [ ] The worker repository is a complete Git clone, not a copy without `.git`.
- [ ] It is detached at the intended full 40-character commit.
- [ ] `git status --short` is empty.
- [ ] Frozen data files pass their checksum and schema audits.
- [ ] Every model uses an exact revision and the required cache is available.
- [ ] The submission does not alter prompts, request IDs, seeds, decoding, context length, or other scientific invariants.

### Static and local tests

```bash
bash -n infra/anvil/realistic_niah_v3_1/*.sh
bash -n scripts/run_realistic_niah_v3_1_worker.sh
python -m pytest tests/test_realistic_niah_v3_1_anvil_reliability.py
python -m pytest tests/test_realistic_niah_v3_1.py
```

Also run, as applicable:

- submission `--dry-run`;
- preparation smoke test;
- real-manifest compatibility audit;
- TP=1 GPU-binding smoke test;
- TP=2 GPU-binding and first-real-request smoke test;
- multi-step resource-binding smoke test if using a mixed allocation.

### Compute-node preflight

- [ ] `nvcc` resolves to the pinned CUDA 12.8 path.
- [ ] A real `compute_90a` compilation succeeds.
- [ ] `libcudart.so.12` exists and loads through `ctypes.CDLL`.
- [ ] Ninja resolves from the frozen environment.
- [ ] Python, PyTorch, Transformers, vLLM, and related package versions are recorded.
- [ ] For each worker: visible GPU count = GPUs per task = tensor-parallel size.
- [ ] Provenance records `scontrol show job`, module state, CUDA tools, package versions, and `nvidia-smi`.

## Minimal monitoring set

```bash
squeue -j "$JOB_ID"
squeue -s -j "$JOB_ID"
sacct -j "$JOB_ID" -X \
  -o JobID,State,Elapsed,ExitCode,NodeList
tail -n 200 "$RUN_ROOT/orchestration/slurm/"*"$JOB_ID"*.out
```

Monitor all of the following:

- worker logs continue to receive new timestamps;
- request-part counts increase;
- completed-bundle markers appear exactly once;
- no new `Traceback`, `RuntimeError`, OOM, illegal memory access, CUDART, `compute_90a`, or worker-failure messages appear;
- every allocated GPU is attached to the intended live step;
- the final request and shard audits remain consistent.

Estimate throughput and ETA from a recent rolling window:

```text
throughput = change in completed requests / elapsed hours
ETA = remaining requests / throughput
```

Do not estimate completion from requested wall time, model parameter count, or the batch job's elapsed percentage.

## Unsafe shortcuts to avoid

- Do not run a large-model or JIT test directly on a login node.
- Do not trust the system-default `nvcc`.
- Do not set only `CUDA_HOME` while omitting `LD_LIBRARY_PATH`.
- Do not verify only the batch environment and skip worker-level validation.
- Do not launch cross-node workers from `/var/spool/slurm/...`.
- Do not push `gpu_memory_utilization` toward 1.0 as the default response to insufficient KV cache.
- Do not change TP=2 to TP=4 without revalidating the engine config and manifest.
- Do not launch two ordinary workers against the same shard.
- Do not delete manifests, claims, or checkpoints to bypass compatibility errors.
- Do not weaken a manifest comparison to a few selected fields.
- Do not run the global finalizer from a filtered or partial job.
- Do not claim that all GPUs are healthy because the parent job is `RUNNING`.
- Do not treat exit code 143 as the first/root failure.
- Do not cancel a healthy job merely because another group failed.

## Reference implementation in this repository

The final hardened implementation for this incident is at commit:

```text
939410edde9885fea5c791e90fdb632254bd327c
```

Relevant files:

- `infra/anvil/realistic_niah_v3_1/run_slurm_task.sh`: worker-level CUDA, GPU/TP, and vLLM preflight;
- `infra/anvil/realistic_niah_v3_1/v3_1_split_inference.slurm`: CUDA provenance, explicit `srun --export`, and split-group steps;
- `infra/anvil/realistic_niah_v3_1/submit_anvil_split.sh`: independent four-GPU submissions and pre-submission audit;
- `src/realistic_niah_v3_1/engine.py`: frozen per-model engine configuration;
- `src/realistic_niah_v3_1/resume.py`: manifest signature and cross-commit allowlist;
- `src/realistic_niah/runner.py`: atomic part checkpoints, canonical merge, and compatibility checks;
- `scripts/run_realistic_niah_v3_1_worker.sh`: bundle claims, failed-attempt archival, and resume behavior;
- `scripts/finalize_realistic_niah_v3_1_split_group.sh`: group markers, `flock`, and exactly-once finalization;
- `tests/test_realistic_niah_v3_1_anvil_reliability.py`: Slurm/CUDA reliability regression tests.

Copy the invariants and validation mechanisms, not a job-specific `sbatch` command. Paths, allocations, versions, and experiment designs must be adapted deliberately.

## Codex handoff prompt

The text below can be sent directly to Codex together with this file. Replace the bracketed placeholders before sending it.

```text
You are maintaining a multi-GPU vLLM inference pipeline on Purdue Anvil under Slurm.

Read the attached `anvil_vllm_slurm_debugging_handbook.md` completely before taking action. Treat it as a verified incident guide, not as proof that my pipeline has the same root cause.

Repository: [ABSOLUTE_REPOSITORY_PATH]
Run root: [ABSOLUTE_RUN_ROOT]
Current job IDs: [JOB_IDS]
Expected Git commit: [FULL_40_CHARACTER_SHA]
Expected models and exact revisions: [MODEL_REVISION_LIST_OR_MANIFEST_PATH]
Scientific invariants: [DATASET, PROMPT, REQUEST-ID, CONTEXT, DECODING, AND OUTPUT INVARIANTS]
Expected final audit: [AUDIT_PATH_AND_EXACT_EXPECTED_COUNTS]

Your objective is to diagnose the earliest concrete failure, make the smallest infrastructure/runtime repair, test it proportionately, and preserve scientifically compatible checkpoints.

Operating rules:

1. Begin with read-only inspection. Record `git status`, the exact commit, `squeue`, `squeue -s`, `sacct`, `scontrol show job`, the main log, worker logs, manifests, claims, and checkpoint progress.
2. Identify the earliest specific error. Do not infer the root cause from a final exit code such as 143.
3. Distinguish the batch allocation from its internal job steps. Verify how many workers are actually running and which GPUs each worker sees.
4. Verify CUDA independently at four layers: driver/PyTorch visibility, actual `nvcc`, real H100 `compute_90a` compilation, and runtime loading of `libcudart.so.12`.
5. Repeat the critical CUDA and GPU-binding checks inside each `srun` worker. Confirm visible GPU count = Slurm GPUs per task = vLLM tensor-parallel size.
6. Inspect restricted `--export` lists. Do not assume variables set in the batch shell reach each worker.
7. Require a representative real request to complete before declaring a vLLM configuration healthy.
8. Preserve all scientific invariants. Do not change model revisions, prompts, ordered request IDs, datasets, seeds, context length, decoding semantics, or validated outputs unless I explicitly authorize it.
9. Preserve user work. Do not overwrite a dirty worktree. Use a clean detached clone when a running job still references the old repository.
10. Preserve checkpoints and claims. Do not delete manifests, request parts, claims, or outputs to bypass an error. Prove a claim is stale before recovering it.
11. Resume across commits only when the full old SHA is explicitly allowlisted and every scientific manifest field remains compatible. Normalize only additive historical fields whose old behavior is provably identical to the new default.
12. Prefer simple Slurm layouts. If a mixed TP=2/TP=1 allocation is proposed, first prove simultaneous step scheduling with a short resource-binding smoke test. Otherwise use independent jobs with atomic bundle claims.
13. Run shell syntax checks, targeted unit/reliability tests, a submission dry run, compute-node CUDA preflight, and a first-real-request smoke test before a full resubmission.
14. Do not cancel a running job, delete data, or change allocation/account settings without explicit authorization. If permissions, quota, allocation, model license, or authentication blocks progress, stop and report it.
15. Do not report the experiment complete based only on Slurm `COMPLETED`. Read the final audit and verify its exact expected shard, request, and unique-request counts.

For each update, report:

- job and step states, queue reason, nodes, start time, and elapsed time;
- completed bundles, shards, and unique requests;
- earliest error and evidence for the root cause;
- exact files and lines changed;
- tests and preflights run, with results;
- the replacement job ID and frozen commit, if a resubmission was necessary;
- remaining risks or facts that are still unverified.

Do not make speculative broad changes. If evidence does not distinguish between two causes, run the smallest read-only or smoke-test experiment that separates them and report the uncertainty explicitly.
```

## Suggested first reply from Codex

A good initial response from the coding agent should resemble:

```text
I will first inspect the repository state, Slurm job and step states, the earliest worker error, CUDA provenance, manifests, and checkpoint progress without changing anything. I will then state the evidence-backed root cause and propose the smallest compatible fix. I will not alter the frozen experiment design, delete checkpoints, cancel jobs, or resubmit until the requested authorization and safety conditions are satisfied.
```

If the agent immediately proposes deleting checkpoints, weakening manifest validation, increasing GPU utilization to nearly 1.0, or resubmitting without locating the earliest error, stop it and ask for an evidence-backed diagnosis first.
