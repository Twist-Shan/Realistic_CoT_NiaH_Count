# Realistic NIAH V3.1 — CHTC 24×A100 backend

This package is an additional UW–Madison CHTC backend. It does not replace or
modify the existing eight-H100 submission, and it does not import, invoke, or
share state with the Lambda backend.

## Allocation

The DAG submits six independent four-GPU jobs:

- two jobs require full, non-MIG A100 80GB devices (8 GPUs total);
- four jobs require full, non-MIG A100 40GB devices (16 GPUs total);
- one CPU child job starts only after all GPU processes finish and merges the
  48 logical shards into one audited 161,280-request result.

The 80GB tier contains Qwen3-14B/32B, Gemma4-12B/26B/31B, and
Nemotron-Nano-v2-9B. Nemotron 9B is in this tier because its frozen engine
profile uses float32 Mamba state. The remaining 4–9B checkpoints use the 40GB
tier with conservative request concurrency.

`chtc_v31_a100_assignments.tsv` is the authoritative allocation plan. It
contains exactly 24 GPU workers and covers every registered `(model, mode)`
shard exactly once. Splitting selected four-mode model bundles raises physical
model loads from 14 to 24 in exchange for using all 24 GPUs concurrently; it
does not change stimuli, revisions, prompt construction, request IDs, or the
final merge audit.

Within the HTCondor queue table, `+` separates prompt modes. The job wrapper
converts it to the comma-separated CLI representation after HTCondor has parsed
the row; commas cannot be stored directly because `queue ... from` treats them
as field separators.

The queue columns carrying ordinary RAM and scratch-disk quantities deliberately
use the `job_` prefix. Naming them `request_*` would make HTCondor synthesize
additional custom-resource requirements instead of only setting the standard
`RequestMemory` and `RequestDisk` attributes.

## Frozen inputs (not stored in Git)

Place these two files beside the submit files before submission:

- `realistic_v31.bundle`, containing commit
  `0afd1adcf2e4b9033805323b9e396220c29afb9e` on
  `codex/realistic-consolidation`;
- `realistic_v31_dataset.tar.gz`, SHA-256
  `780a1539ce3a89a880e593ab0bc45939239b9394a369b070a5cdf289d01afe9b`.

The GPU job validates the assigned count, A100 model, non-MIG status, per-GPU
memory interval, frozen commit, dataset audit, runtime versions, free disk,
and each completed shard manifest. The merge job requires all six successful
partial archives and reruns the canonical V3.1 final-shard audit.

## Validate and submit

```bash
python3 validate_a100_plan.py chtc_v31_a100_assignments.tsv
mkdir -p logs
condor_submit_dag chtc_v31_a100.dag
```

Do not use `condor_submit chtc_v31_a100.sub` directly for the formal run: the
DAG dependency is what guarantees that merging starts only after all six GPU
processes have transferred their partial archives.
