# Realistic NIAH counting protocol V2

## Scientific question

V2 measures how exact counting changes with passage length, true needle
count, model, and prompting mechanism. The primary outcome is registered
accuracy: the count must be exact, the response must satisfy the registered
format, and generation must not truncate. Parse failures, format failures,
and truncations therefore remain failures; no response is removed after
generation. Raw exact-count accuracy is retained as a secondary diagnostic.

Before the formal panel, V2 uses a small guarded-CoT smoke run to verify that
the concise anti-repetition instruction does not exhaust the registered
4,096-token generation budget. The smoke gate is zero truncations; accuracy
and overthinking indicators remain diagnostics rather than launch gates.

## Registered model panel

| Label | Hugging Face model ID | Reasoning policy |
|---|---|---|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | switchable |
| Qwen3-4B | `Qwen/Qwen3-4B` | switchable |
| Qwen3-8B | `Qwen/Qwen3-8B` | switchable |
| Qwen3-32B | `Qwen/Qwen3-32B` | switchable |
| Gemma4-E4B | `google/gemma-4-E4B-it` | switchable |
| Gemma4-12B | `google/gemma-4-12B-it` | switchable |
| DeepSeek-R1-0528-Qwen3-8B | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | always on |
| GLM-Z1-9B-0414 | `zai-org/GLM-Z1-9B-0414` | always on |

These are the eight primary models. `GLM-4-9B-0414`
(`zai-org/GLM-4-9B-0414`) is additionally registered as an off-only matched
control for GLM-Z1; it is outside the primary-panel request count.
`Qwen3-8B` is the architecture-matched non-thinking comparison for the
DeepSeek checkpoint. These comparisons isolate post-training more closely
than unrelated-model comparisons, but they are not claimed to be pure
single-variable causal interventions.

All eight models are reasoning-capable. Qwen and Gemma expose a template
switch, so V2 disables template thinking for direct/index/bullet modes and
enables it for native-thinking modes. DeepSeek-R1-0528-Qwen3-8B and
GLM-Z1-9B-0414 do not expose an equivalent off switch: the official
DeepSeek tokenizer has no `enable_thinking` argument, and GLM's official
`chat_template.jinja` injects `<think>` on generation. Their native templates
are therefore used unchanged in every mode. For those two models, “direct,”
“indexed enumeration,” and “bullet enumeration” name the visible answer
instruction, not a non-thinking condition.

The registration follows the official
[DeepSeek-R1-0528-Qwen3-8B model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
and
[GLM-Z1-9B-0414 model card](https://huggingface.co/zai-org/GLM-Z1-9B-0414),
including their published sampling guidance.

No Llama or OLMo model is part of the V2 panel.

## Frozen stimulus grid and request accounting

- Passage lengths: `2,000`, `3,000`, `5,000`, `10,000`, `20,000`
  canonical-tokenizer tokens after insertion.
- True needle counts: `1,2,3,4,5,6,8,10,20,30`.
- Seeds: `1234..1243`.
- Shared stimuli: `5 × 10 × 10 = 500`.
- Formal prompt modes: 4.
- Generations per model: `500 × 4 = 2,000`.
- Complete eight-model panel: `8 × 2,000 = 16,000`.

Here “500” denotes the shared stimulus matrix. It does not include the
four prompt-mode expansions.

The canonical tokenizer remains `Qwen/Qwen3-8B`. The run records both
canonical passage length and each model tokenizer's realized passage/input
length.

## Haystack corpus and length construction

V2 builds its filler from the complete 218-URL Paul Graham source list
distributed by
[NVIDIA/RULER](https://github.com/NVIDIA/RULER/tree/main/scripts/data/synthetic/json).
`scripts/sync_paul_graham_full_corpus.py` downloads the public essay pages
and repository text files, extracts visible article text, excludes sources
below 5 KiB, removes exact content duplicates by SHA256, and writes a
per-URL index, corpus manifest, and file checksums. The frozen-dataset
manifest records the corpus-manifest SHA256.

For each seed, content-unique essays are deterministically shuffled and
concatenated once. Passage lengths use nested prefixes of that shuffled
corpus: a seed's 2K filler is the beginning of its 10K filler, which is the
beginning of its 20K filler, modulo the small adjustment required to keep
post-insertion length exact. No essay or composite corpus is repeated. If
the deduplicated source is too short, generation stops with an error.

This follows the important part of common NIAH practice while tightening
its auditability. The
[Greg Kamradt NIAH implementation](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
concatenates all text files in `PaulGrahamEssays` before trimming to the
requested length; it only loops the whole collection if that collection is
still too short. RULER downloads the GitHub essays plus the Paul Graham
pages into one large essay corpus and slices the requested prefix; its code
only repeats after the requested length exceeds that entire corpus. Thus a
normal 10K condition should not be made by repeating one short essay.

The first V2 smoke artifact predating this rule selected one short essay
per row and repeated it 6--11 times for every 20K stimulus. That artifact
remains preserved as a diagnostic of the old prompt but is not valid
evidence for a clean length effect and is not reused by the revised smoke
or formal experiment.

## Prompt layout

The only registered layout is `cue_before_query_after`: an identical
counting cue precedes the passage and the mode-specific question follows
the passage. There is no query-first/query-last factor in V2.

### Direct

```text
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{passage}
</passage>

How many city-score audit records are in the passage?
In the final answer, output exactly one line:
Total: <integer>
```

### Indexed enumeration

```text
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{passage}
</passage>

How many city-score audit records are in the passage?
List each occurrence once, in passage order, as:
<k>. <city>: <score>
Then report the number listed:
Total: <integer>
Do not include any other text.
```

### Bullet enumeration

```text
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{passage}
</passage>

How many city-score audit records are in the passage?
List each occurrence once, in passage order, as:
- <city>: <score>
Then report the number listed:
Total: <integer>
Do not include any other text.
```

### Native thinking with anti-repetition guard

```text
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{passage}
</passage>

How many city-score audit records are in the passage?
Count in one left-to-right pass using only a running integer tally.
Do not output city names, scores, a list, copied passage text, explanations,
checking, or a second scan. Keep any reasoning under 100 words. Then
immediately output exactly one final line:
Total: <integer>
```

## Parsing and evaluation

`Total: <integer>` is the registered count parser. Indexed and bullet
enumeration have separate strict-format parsers. A second marker-tolerant
parser extracts city-score pairs from either registered list marker, so
format compliance and semantic retrieval quality are reported separately.
Using the wrong marker is a format failure but does not erase measurable
pair precision/recall.

The primary per-response fields are:

- registered success, raw exact count, signed error, absolute error, and
  normalized absolute error;
- parse status and truncation;
- strict enumeration format status;
- listed-total consistency;
- city-score pair precision, recall, F1, duplicates, hallucinations, and
  omissions.

## Guarded-CoT truncation smoke test

The smoke grid is:

- Models: Qwen3-8B, Gemma4-12B, DeepSeek-R1-0528-Qwen3-8B,
  and GLM-Z1-9B-0414.
- Passage lengths: `2K,20K`.
- Needle counts: `6,20,30`.
- Seeds: `2234,2235`, disjoint from the formal seeds `1234..1243`.
- Twelve stimuli and twelve guarded `native_thinking` requests per model,
  for 48 total.

The only primary launch requirement is zero truncations over all 48 requests.
Per-model registered accuracy, raw exact-count accuracy, parse and format
failures, output-token count, numbered-enumeration restarts, duplicate
city-score mentions, duplicate reasoning lines, and the operational
overthinking flag are reported for diagnosis. No smoke response is removed.

For Qwen and Gemma, the official template thinking switch is enabled.
DeepSeek and GLM-Z1 retain their official always-on reasoning templates.
The formal run should start only after the 48-row completeness check, the
zero-truncation gate, and inspection of any parse, formatting, or repetition
failure examples.

## Decoding controls

- Switchable-model direct: greedy, 64 output tokens.
- Switchable-model indexed/bullet enumeration: greedy, 1,536 output tokens.
- Native thinking (including the guarded smoke): 4,096 output tokens.
- Qwen thinking: temperature `0.6`, top-p `0.95`, top-k `20`.
- Gemma thinking: temperature `1.0`, top-p `0.95`, top-k `64`.
- DeepSeek-R1-0528-Qwen3-8B: always-on reasoning in all four modes,
  4,096 output tokens, temperature `0.6`, top-p `0.95`, and no top-k
  restriction.
- GLM-Z1-9B-0414: always-on reasoning in all four modes, 4,096 output
  tokens, temperature `0.6`, top-p `0.95`, and top-k `40`.

The 4,096-token cap is deliberately common to all reasoning conditions so
overrun remains measurable. It is lower than GLM's maximum-length guidance;
truncation is retained as failure rather than silently extending selected
runs.

V2 raises the runner's default model context limit to 32,768 tokens so a
20K passage plus the 4,096-token thinking budget fits. A run must still
check every rendered prompt against the selected model's effective context
limit before generation.

## Version isolation

V2 request IDs include model, prompt mode, fixed query layout, and stimulus
ID. V2 rows and manifests use V2 schema identifiers. The runner refuses to
resume into a directory whose schema, model, stimulus hash, request-ID hash,
engine configuration, or Git commit differs.
