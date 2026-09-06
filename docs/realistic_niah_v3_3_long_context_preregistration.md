# Realistic NIAH V3.3: 32B long-context holdout

## Experimental purpose

This extension measures whether the empirical laws selected in V3.2 retain
predictive value outside the original 1k-20k passage-length range. It evaluates
two immutable checkpoints under the existing direct (non-thinking) and
native-thinking prompt modes:

- `google/gemma-4-31B-it` at revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`;
- `Qwen/Qwen3-32B` at revision
  `9216db5781bf21249d130ec9da846c4624c16137`.

The primary analysis treats 25k-100k as an unopened holdout. The V3.2 law
structure and coefficients must be recorded before the holdout outputs are
examined. Any coefficient refit that incorporates these new observations is
exploratory.

**Current conclusion:** this document fixes a prospective extrapolation test;
it contains no empirical result from the new length range.

## Registered grid and request accounting

The passage lengths are

\[
L\in\{25,30,40,50,60,70,80,90,100\}\text{k tokens}.
\]

The needle counts and paired seed clusters are inherited unchanged from V3.1:

\[
N\in\{1,2,3,4,5,6,7,8,9,10,12,15,18,20\},\qquad
s\in\{1234,\ldots,1263\}.
\]

There are $9 \times 14 \times 30=3{,}780$ frozen stimuli. Each stimulus is
evaluated in two prompt modes for each model, producing 7,560 requests per
model and exactly 15,120 formal requests.
Passage length includes inserted facts. Needles are placed between 5% and 95%
of final passage characters and the haystack uses the registered
`multi_file_no_repeat` policy.

**Current conclusion:** the measurement protocol changes only the
passage-length range and checkpoint subset; the N grid, paired seeds, passage
construction, prompt wording, query layout, parser, and outcome definitions
remain fixed. Qwen additionally requires the separately registered YaRN
runtime transformation described below because the new range exceeds its
native context window.

## Engine and decoding settings

Both checkpoints run in bfloat16 with tensor parallelism 2. The vLLM engine uses
`max_model_len=131072`, `gpu_memory_utilization=0.92`, one active sequence,
one request per batch, eager execution, disabled custom all-reduce, and prefix
caching. The immutable Gemma4-31B configuration reports 262,144 maximum
position embeddings; 131,072 is within that checkpoint configuration.

The immutable Qwen3-32B configuration reports 40,960 maximum position
embeddings and does not contain a RoPE-scaling block. The official model card
states that the checkpoint is native to 32,768 tokens and was validated to
131,072 tokens with YaRN. The registered Qwen runtime therefore adds exactly
`rope_type=yarn`, factor 4.0, and
`original_max_position_embeddings=32768`. This override is saved in every run
manifest and is part of strict resume compatibility.

Source records: [Qwen3-32B official model card](https://huggingface.co/Qwen/Qwen3-32B)
and [immutable revision config](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json).

Direct decoding uses at most 64 output tokens and deterministic temperature
zero for both checkpoints. Gemma native-thinking decoding uses at most 4,096
output tokens with temperature 1.0, top-p 0.95, and top-k 64. Qwen
native-thinking decoding uses at most 4,096 output tokens with temperature 0.6,
top-p 0.95, and top-k 20. These settings match the corresponding V3.1 runs.

Before formal generation for each checkpoint, one 100k-token, N=20, seed=1234
stimulus is evaluated in both modes. This preflight tests rendered input length,
KV-cache allocation, TP=2 execution, and checkpoint writing at the maximum
registered context. The four preflight outputs are stored outside the formal
result directories and are never included in the 15,120-request audit.

**Current conclusion:** the 131,072-token engine limit provides headroom for
the 100k canonical passage, chat-template overhead, and the 4,096-token
native-thinking output budget. Successful preflight remains a required runtime
condition rather than evidence that every formal request will complete.

## Parallel execution and recovery

Each model receives an independent Anvil job on one node with four H100 80GB
GPUs. A single Slurm step starts two tasks, assigns two GPUs to each task, and
each task hosts one independent TP=2 vLLM runtime. The fixed length partition
within each model job is:

- worker 0: 30k, 60k, 70k, 100k;
- worker 1: 25k, 40k, 50k, 80k, 90k.

The partition is disjoint and covers all nine lengths. It was chosen using
V3.1 wall-time observations to balance projected runtime rather than request
count. Each worker writes atomic request-batch parts in a separate directory.
Interrupted jobs resume only from an exactly compatible manifest; a different
Git commit requires an explicit allowlist.

After both workers in a model job finish, that group writes synchronous worker
completion markers. The last model group to finish obtains a filesystem lock
and runs the finalizer. It checks the protocol, both model revisions, engine
settings and Qwen YaRN override, length assignments, two prompt modes, all 30
seeds in every model-length-count-mode cell, unique request IDs, 7,560 requests
per model, and the exact total of 15,120. Only then is a combined formal result
and completion marker written.

**Current conclusion:** the execution design prevents two workers from claiming
the same formal request and prevents a preflight or partial shard from being
reported as the completed holdout.

## Interpretation boundary

The primary result will be predictive performance of the frozen V3.2 laws on
new context lengths. Agreement supports extrapolation for the measured
checkpoint, prompt pair, task construction, and 25k-100k range. Disagreement
identifies a failure of that extrapolation. Neither outcome establishes a
universal law of language-model counting. Qwen results also include the effect
of the preregistered YaRN context extension, so a difference between Qwen and
Gemma cannot be attributed solely to model family.

**Current conclusion:** V3.3 is a targeted two-checkpoint out-of-range
validation study, not a new law-selection campaign.
