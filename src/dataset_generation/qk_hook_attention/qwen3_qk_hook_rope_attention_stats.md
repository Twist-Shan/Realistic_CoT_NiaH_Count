# Qwen3-8B: Q/K Hooking + Offline Attention Statistics

Target model: `Qwen/Qwen3-8B` text-only causal LM.  
Goal: run a normal Hugging Face forward pass, ideally with `attn_implementation="flash_attention_2"`, save selected-layer Q/K projection outputs to disk, and compute attention statistics offline without materializing the full `[layers, heads, T, T]` tensor.

This is a prototype-oriented implementation. It is meant to be easy to inspect and modify, not a highly optimized Triton/FlashAttention replacement.

---

## 1. Summary of prior art and what this implementation changes

### Hugging Face `output_attentions=True`

The standard Hugging Face route is to call the model with `output_attentions=True`. This is convenient but not suitable for long-context analysis because it returns dense attention maps with shape roughly `[batch, heads, query_pos, key_pos]`. For Qwen3-8B, one 4K-token layer already costs about 1 GiB in bf16/fp16 for the full attention matrix. Hugging Face issues and docs also make clear that optimized attention backends such as FlashAttention generally do not expose attention weights in the same way as eager attention; current warnings tell users to switch to eager attention to capture attention outputs.

### BertViz

BertViz is a mature visualization tool for attention maps. It loads Hugging Face models with `output_attentions=True` for head/model views. Its neuron view needs query/key vectors and therefore uses custom model variants for some older architectures. This is useful for visualization, but it is not designed for Qwen3 long-context FlashAttention runs where the dense attention matrix is too expensive.

### TransformerLens

TransformerLens exposes hooks and activation caches for mechanistic interpretability. It can cache internal activations, including Q/K/pattern-like tensors, and has newer bridging support for Hugging Face models. It is excellent for exploratory interpretability, but for this particular use case I would avoid caching everything through a general-purpose activation cache. The goal here is narrower: selected Qwen3 layers, selected heads/statistics, and a normal HF FlashAttention forward.

### FlashAttention

FlashAttention is explicitly designed to compute exact attention while reducing memory traffic by avoiding reads/writes of the large `N x N` attention matrix to high-bandwidth memory. This is exactly why it is good for inference and bad for “please return the attention matrix” workflows. The practical compromise is: keep FlashAttention for the model forward, but save enough intermediate data, here raw Q/K projections, to reconstruct selected attention statistics offline.

### Hugging Face `AttentionInterface`

Hugging Face now supports registering custom attention functions through `AttentionInterface`. That could be used to compute statistics in the forward pass. However, custom attention backends require careful mask handling, and they are more invasive than projection hooks. Since you already decided that Q/K hooks are the best overall compromise, this implementation does not patch model source or register a custom attention function.

### Attention sinks and long-context statistics

Metrics such as BOS/special-token attention mass and “attention received by a span from later tokens” are motivated by long-context findings such as attention sinks, where early tokens can receive consistently high attention even when not semantically central. This implementation therefore treats sink/special-token mass, named span mass, local-window mass, entropy, and top-k attended positions as first-class statistics.

---

## 2. Qwen3-8B details that matter

The released `Qwen/Qwen3-8B` config has these relevant properties:

```text
num_hidden_layers      = 36
num_attention_heads    = 32
num_key_value_heads    = 8
head_dim               = 128
hidden_size            = 4096
rms_norm_eps           = 1e-6
rope_theta             = 1000000
use_sliding_window     = false
sliding_window         = null
torch_dtype            = bfloat16
```

The attention computation in Hugging Face Qwen3 is not just:

```text
q_proj/k_proj -> RoPE -> attention
```

It is:

```text
hidden_states
  -> q_proj / k_proj
  -> reshape to [..., heads, head_dim]
  -> q_norm / k_norm over head_dim
  -> RoPE
  -> GQA mapping: 32 Q heads share 8 KV heads
  -> causal mask
  -> softmax(q @ k.T / sqrt(head_dim))
```

Therefore, if you hook `q_proj` and `k_proj`, the offline reconstruction must apply Qwen3 `q_norm`/`k_norm` before RoPE. The capture script below saves the tiny norm weights per captured layer, so the analysis script does **not** need to load the 8B model again.

---

## 3. Code overview: what is added or changed

No Hugging Face model file needs to be copied or edited.

You add two scripts:

```text
capture_qk_qwen3.py
  - loads Qwen/Qwen3-8B with Hugging Face
  - uses attn_implementation="flash_attention_2" by default
  - registers forward hooks only on selected layers' q_proj/k_proj modules
  - runs one inference/prefill forward pass with output_attentions=False
  - saves raw q_proj/k_proj tensors, input_ids, position_ids, attention_mask,
    Qwen3 q_norm/k_norm weights, config metadata, and an editable analysis spec

analyze_qk_qwen3.py
  - loads saved raw Q/K for one layer
  - reconstructs one attention head:
      raw q/k -> q_norm/k_norm -> default Qwen3 RoPE -> GQA KV-head mapping
  - computes statistics blockwise:
      critical-token rows, span mass, top-k, entropy, local-window mass,
      and average attention received by a window from later tokens
```

The scripts assume batch size 1 and a full prompt/prefill forward pass. That matches your current long-context NIAH-style analysis use case.

---

## 4. Installation assumptions

Recommended environment:

```bash
pip install "transformers>=4.51.0" accelerate safetensors
pip install flash-attn --no-build-isolation  # if your CUDA/GPU setup supports it
```

For a machine without FlashAttention installed, set:

```bash
--attn-implementation sdpa
```

or:

```bash
--attn-implementation eager
```

But for production capture, use FlashAttention if available.

---

## 5. Quickstart

### 5.1 Optional marker file

For NIAH-style prompts, create a marker file to auto-detect needle spans:

```json
{
  "needle_1": "The special passkey is 72941.",
  "needle_2": "The secret city is Valparaiso."
}
```

Save it as `markers.json`. The capture script will tokenize each marker, find exact token subsequences in the final model input, and create `analysis_spec.json` with named spans and after-needle query positions.

Exact token subsequence matching can fail if whitespace or chat templating changes the tokenization. If so, manually edit `analysis_spec.json`.

### 5.2 Capture selected layers

```bash
python capture_qk_qwen3.py   --model-name Qwen/Qwen3-8B   --prompt-file prompt.txt   --markers-json markers.json   --out-dir qk_cache_run1   --layers 0,8,16,24,35   --attn-implementation flash_attention_2   --model-dtype bf16   --save-dtype bf16
```

For chat-template mode:

```bash
python capture_qk_qwen3.py   --model-name Qwen/Qwen3-8B   --prompt-file prompt.txt   --chat-template   --enable-thinking   --markers-json markers.json   --out-dir qk_cache_chat_run1   --layers 0,8,16,24,35
```

The cache directory will look like:

```text
qk_cache_run1/
  metadata.json
  analysis_spec.json
  prompt.txt
  model_text.txt
  input_ids.pt
  attention_mask.pt
  position_ids.pt
  tokens.json
  layer_00_q_raw.pt
  layer_00_k_raw.pt
  layer_00_qk_norms.pt
  layer_08_q_raw.pt
  layer_08_k_raw.pt
  layer_08_qk_norms.pt
  ...
```

### 5.3 Analyze one layer/head

```bash
python analyze_qk_qwen3.py   --cache-dir qk_cache_run1   --layer 24   --head 7   --topk 32   --key-block-size 8192   --query-block-size 64   --out-json stats_layer24_head07.json
```

This outputs a JSON file containing:

```text
selected_query_rows:
  - entropy at last token / special tokens / after-needle positions
  - attention mass to named spans
  - local-window mass
  - top-k attended positions with token ids/token strings

window_received:
  - average attention mass received by each marker/span from later tokens
```

To also save full attention rows for the selected critical query positions:

```bash
python analyze_qk_qwen3.py   --cache-dir qk_cache_run1   --layer 24   --head 7   --save-full-rows
```

This saves:

```text
attention_rows_layer_24_head_07.pt
```

with shape `[num_selected_queries, seq_len]`. Use this only for a small number of query positions.

---

## 6. Analysis spec format

The capture script writes a starter `analysis_spec.json`. You can edit it manually.

Example:

```json
{
  "spans": {
    "bos": [[0, 1]],
    "needle_1": [[1200, 1224]],
    "needle_2": [[2600, 2626]],
    "instruction": [[0, 180]],
    "special_tokens": [[0, 1], [13, 14]]
  },
  "query_positions": {
    "last": [-1],
    "after_needle_1": [1224],
    "after_needle_2": [2626],
    "probe_positions": [1024, 2048, 3072]
  },
  "window_received": [
    {
      "name": "needle_1_received_from_later_tokens",
      "span_name": "needle_1",
      "query_start": 1224,
      "query_end": null
    },
    {
      "name": "instruction_received_from_last_512_tokens",
      "span_name": "instruction",
      "query_start": -512,
      "query_end": null
    }
  ],
  "local_windows": [32, 128, 512]
}
```

Conventions:

```text
spans use [start, end) token intervals.
query position -1 means last token.
window_received query_end null means seq_len.
window_received query_end -1 means include the last token.
```

---

## 7. Memory notes for Qwen3-8B

For Qwen3-8B, per layer:

```text
q_raw shape = [1, T, 32 * 128] = [1, T, 4096]
k_raw shape = [1, T,  8 * 128] = [1, T, 1024]
```

With bf16/fp16:

| Sequence length | Q per layer | K per layer | Q+K per layer | 5 selected layers |
|---:|---:|---:|---:|---:|
| 4,096 | 32 MiB | 8 MiB | 40 MiB | 200 MiB |
| 32,768 | 256 MiB | 64 MiB | 320 MiB | 1.56 GiB |

For comparison, the dense attention matrix for one 4K-token layer is:

```text
32 heads * 4096 * 4096 * 2 bytes = 1 GiB per layer
```

At 32K tokens, one dense attention layer is about 64 GiB. That is why this approach saves Q/K and computes only targeted statistics offline.

---

## 8. Script A: `capture_qk_qwen3.py`

Save the following as `capture_qk_qwen3.py`.

```python
#!/usr/bin/env python3
"""
Capture raw q_proj/k_proj outputs for selected layers of Qwen/Qwen3-8B while
running the normal Hugging Face forward pass, e.g. with FlashAttention-2.

This script saves pre-Qwen3-q_norm, pre-RoPE Q/K projection outputs. The
companion analysis script reconstructs the exact per-head attention logits by
applying Qwen3 q_norm/k_norm, RoPE, GQA head mapping, and causal masks offline.

Target: text-only Qwen3 causal LMs, especially Qwen/Qwen3-8B.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "same": None,
}


def parse_layers(spec: str) -> List[int]:
    """Parse layer specs such as '0,8,16,24,35' or '0:36:4'."""
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError(f"Bad slice in --layers: {part!r}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            layers.extend(range(start, stop, step))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Provide only one of --prompt or --prompt-file.")
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide --prompt, --prompt-file, or pipe text on stdin.")


def find_all_subsequences(haystack: List[int], needle: List[int]) -> List[Tuple[int, int]]:
    if not needle:
        return []
    spans: List[Tuple[int, int]] = []
    n = len(needle)
    # Naive scan is fine for prompt construction/debugging. For huge prompts
    # and many markers, replace with KMP/Aho-Corasick.
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            spans.append((i, i + n))
    return spans


def load_markers(path: str | None) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("markers JSON must be an object: {name: marker_text}")
    return {str(k): str(v) for k, v in data.items()}


def build_analysis_spec(
    *,
    input_ids: List[int],
    tokenizer,
    markers: Dict[str, str],
) -> Dict:
    """Create a starter analysis spec with BOS/special spans and marker spans."""
    seq_len = len(input_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_positions = [i for i, tok_id in enumerate(input_ids) if tok_id in special_ids]

    spans: Dict[str, List[List[int]]] = {
        "bos": [[0, 1]],
    }
    if special_positions:
        spans["special_tokens"] = [[p, p + 1] for p in special_positions]

    query_positions: Dict[str, List[int]] = {
        "last": [seq_len - 1],
    }
    if special_positions:
        query_positions["special_tokens"] = special_positions

    window_received = []

    for name, marker_text in markers.items():
        candidate_texts = [marker_text]
        if marker_text and not marker_text.startswith(" "):
            candidate_texts.append(" " + marker_text)
        if marker_text.startswith(" "):
            candidate_texts.append(marker_text.lstrip())

        found: List[Tuple[int, int]] = []
        seen = set()
        for text in candidate_texts:
            marker_ids = tokenizer(text, add_special_tokens=False).input_ids
            for span in find_all_subsequences(input_ids, marker_ids):
                if span not in seen:
                    found.append(span)
                    seen.add(span)

        found = sorted(found)
        if found:
            spans[name] = [[s, e] for s, e in found]
            after_positions = [e for _, e in found if e < seq_len]
            if after_positions:
                query_positions[f"after_{name}"] = after_positions
            # Default "received by later tokens" metric for this marker.
            later_start = max(e for _, e in found)
            if later_start < seq_len:
                window_received.append(
                    {
                        "name": f"{name}_received_from_later_tokens",
                        "span_name": name,
                        "query_start": later_start,
                        "query_end": None,
                    }
                )
        else:
            print(
                f"[warn] Marker {name!r} was not found as an exact token subsequence. "
                "You can still edit analysis_spec.json manually.",
                file=sys.stderr,
            )

    return {
        "spans": spans,
        "query_positions": query_positions,
        "window_received": window_received,
        "local_windows": [32, 128, 512],
    }


def prepare_text(tokenizer, prompt: str, args: argparse.Namespace) -> str:
    if not args.chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if args.enable_thinking is not None:
        kwargs["enable_thinking"] = args.enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some tokenizer versions may not accept enable_thinking.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def install_qk_hooks(
    *,
    model,
    target_layers: Iterable[int],
    out_dir: Path,
    save_dtype: torch.dtype | None,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles = []

    def make_hook(layer_idx: int, kind: str):
        def hook(_module, _inputs, output):
            # output shape:
            #   q_proj: [batch, seq_len, num_attention_heads * head_dim]
            #   k_proj: [batch, seq_len, num_key_value_heads * head_dim]
            tensor = output.detach()
            if save_dtype is not None:
                tensor = tensor.to(dtype=save_dtype)
            tensor_cpu = tensor.cpu()
            path = out_dir / f"layer_{layer_idx:02d}_{kind}_raw.pt"
            torch.save(tensor_cpu, path)
            print(
                f"[hook] saved layer={layer_idx} {kind}_raw "
                f"shape={tuple(tensor_cpu.shape)} dtype={tensor_cpu.dtype} -> {path}",
                flush=True,
            )
            del tensor_cpu

        return hook

    for layer_idx in target_layers:
        attn = model.model.layers[layer_idx].self_attn
        handles.append(attn.q_proj.register_forward_hook(make_hook(layer_idx, "q")))
        handles.append(attn.k_proj.register_forward_hook(make_hook(layer_idx, "k")))
    return handles


def save_layer_norms(model, target_layers: Iterable[int], out_dir: Path) -> None:
    for layer_idx in target_layers:
        attn = model.model.layers[layer_idx].self_attn
        payload = {
            "q_norm_weight": attn.q_norm.weight.detach().cpu().float(),
            "k_norm_weight": attn.k_norm.weight.detach().cpu().float(),
            "q_norm_eps": float(attn.q_norm.variance_epsilon),
            "k_norm_eps": float(attn.k_norm.variance_epsilon),
        }
        torch.save(payload, out_dir / f"layer_{layer_idx:02d}_qk_norms.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture selected-layer Q/K projection outputs for Qwen3.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", default="0,8,16,24,35", help="e.g. '0,8,16,24,35' or '0:36:4'")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--markers-json", default=None, help="Optional JSON: {name: exact_marker_text}")
    parser.add_argument("--chat-template", action="store_true", help="Wrap prompt with tokenizer.apply_chat_template.")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Passed to Qwen3 chat template if supported. Omit for tokenizer default.",
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model-dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--save-dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-token-strings", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = parse_layers(args.layers)

    prompt = read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    text = prepare_text(tokenizer, prompt, args)
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    if encoded["input_ids"].shape[0] != 1:
        raise ValueError("This simple implementation assumes batch size 1.")

    input_ids_cpu = encoded["input_ids"].cpu()
    attention_mask_cpu = encoded.get("attention_mask", torch.ones_like(input_ids_cpu)).cpu()
    seq_len = int(input_ids_cpu.shape[1])
    position_ids_cpu = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    markers = load_markers(args.markers_json)
    analysis_spec = build_analysis_spec(
        input_ids=input_ids_cpu[0].tolist(),
        tokenizer=tokenizer,
        markers=markers,
    )

    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "model_text.txt").write_text(text, encoding="utf-8")
    torch.save(input_ids_cpu, out_dir / "input_ids.pt")
    torch.save(attention_mask_cpu, out_dir / "attention_mask.pt")
    torch.save(position_ids_cpu, out_dir / "position_ids.pt")
    (out_dir / "analysis_spec.json").write_text(json.dumps(analysis_spec, indent=2), encoding="utf-8")

    if args.save_token_strings:
        tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu[0].tolist())
        (out_dir / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False, indent=2), encoding="utf-8")

    model_dtype = DTYPE_MAP[args.model_dtype]
    save_dtype = DTYPE_MAP[args.save_dtype]

    print(f"[load] {args.model_name} with attn_implementation={args.attn_implementation}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    bad_layers = [i for i in target_layers if i < 0 or i >= num_layers]
    if bad_layers:
        raise ValueError(f"Invalid target layers {bad_layers}; model has {num_layers} layers.")

    save_layer_norms(model, target_layers, out_dir)

    input_device = model.get_input_embeddings().weight.device
    model_inputs = {
        "input_ids": input_ids_cpu.to(input_device),
        "attention_mask": attention_mask_cpu.to(input_device),
        "position_ids": position_ids_cpu.to(input_device),
    }

    metadata = {
        "model_name": args.model_name,
        "created_unix_time": time.time(),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attn_implementation_requested": args.attn_implementation,
        "attn_implementation_model_config": getattr(model.config, "_attn_implementation", None),
        "model_dtype": str(model_dtype),
        "save_dtype": str(save_dtype),
        "target_layers": target_layers,
        "seq_len": seq_len,
        "batch_size": 1,
        "model_config": model.config.to_dict(),
        "special_token_ids": getattr(tokenizer, "all_special_ids", []),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "files": {
            "input_ids": "input_ids.pt",
            "attention_mask": "attention_mask.pt",
            "position_ids": "position_ids.pt",
            "analysis_spec": "analysis_spec.json",
            "tokens": "tokens.json" if args.save_token_strings else None,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    handles = install_qk_hooks(model=model, target_layers=target_layers, out_dir=out_dir, save_dtype=save_dtype)
    try:
        with torch.inference_mode():
            # Do not request attentions. Keep the normal optimized attention path.
            # logits_to_keep=1 reduces final LM-head output memory in recent Transformers.
            _ = model(
                **model_inputs,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()

    print(f"[done] Q/K cache written to {out_dir}")


if __name__ == "__main__":
    main()

```

---

## 9. Script B: `analyze_qk_qwen3.py`

Save the following as `analyze_qk_qwen3.py`.

```python
#!/usr/bin/env python3
"""
Offline attention statistics from Qwen3 q_proj/k_proj caches.

Loads raw q_proj/k_proj outputs captured by capture_qk_qwen3.py, reconstructs
Qwen3's per-head post-q_norm/post-k_norm/post-RoPE Q/K, and computes attention
statistics for one layer and one query head without materializing the full
[heads, seq, seq] attention tensor.

Target: Qwen/Qwen3-8B text-only causal LM, batch size 1.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


COMPUTE_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tensor(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    return obj


def resolve_index(idx: Optional[int], T: int, *, end_exclusive: bool = False) -> int:
    """Resolve negative indices. For range ends, -1 means include the last token."""
    if idx is None:
        return T if end_exclusive else T - 1
    idx = int(idx)
    if idx < 0:
        return T + idx + (1 if end_exclusive else 0)
    return idx


def clamp_interval(start: int, end: int, T: int) -> Tuple[int, int]:
    start = max(0, min(T, int(start)))
    end = max(0, min(T, int(end)))
    if end < start:
        end = start
    return start, end


def normalize_intervals(raw, T: int) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    if raw is None:
        return intervals
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(isinstance(x, int) for x in raw):
        raw = [raw]
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Bad interval {item!r}; expected [start, end].")
        s, e = clamp_interval(int(item[0]), int(item[1]), T)
        if e > s:
            intervals.append((s, e))
    return intervals


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qwen3_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Mirrors Qwen3RMSNorm over the last dimension.
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * torch.rsqrt(variance + eps)
    return (y * weight.float()).to(dtype=x.dtype)


def get_rope_theta(config: Dict) -> float:
    rope_parameters = config.get("rope_parameters") or {}
    if "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    if "rope_theta" in config:
        return float(config["rope_theta"])
    return 1_000_000.0


def assert_default_rope(config: Dict) -> None:
    """
    This implementation covers the released Qwen/Qwen3-8B default RoPE.
    If you enable YaRN / dynamic scaling for 131k contexts, use the exact HF
    rotary embedding or extend this function to match modeling_rope_utils.
    """
    rope_parameters = config.get("rope_parameters") or {}
    rope_type = rope_parameters.get("rope_type", "default")
    rope_scaling = config.get("rope_scaling", None)
    if rope_type not in (None, "default"):
        raise NotImplementedError(f"Only default Qwen3 RoPE is implemented here; got rope_type={rope_type!r}")
    if rope_scaling not in (None, "null"):
        raise NotImplementedError(
            f"Only Qwen3 default RoPE is implemented here; got rope_scaling={rope_scaling!r}. "
            "For YaRN/long-context scaling, compute cos/sin with the exact Hugging Face rotary_emb."
        )


def qwen3_default_rope_cos_sin(
    *,
    position_ids: torch.Tensor,  # [T]
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    pos = position_ids.to(device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def reconstruct_single_head_qk(
    *,
    cache_dir: Path,
    layer: int,
    head: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Return q_head, k_head with shape [T, D] for one Q head.

    For Qwen3 GQA, multiple query heads share one KV head. The corresponding
    KV head is head // num_key_value_groups, matching Hugging Face repeat_kv.
    """
    meta = load_json(cache_dir / "metadata.json")
    config = meta["model_config"]
    assert_default_rope(config)

    Hq = int(config["num_attention_heads"])
    Hkv = int(config["num_key_value_heads"])
    D = int(config.get("head_dim") or int(config["hidden_size"]) // Hq)
    T = int(meta["seq_len"])
    if not (0 <= head < Hq):
        raise ValueError(f"head must be in [0, {Hq}); got {head}")
    n_rep = Hq // Hkv
    kv_head = head // n_rep

    q_raw = load_tensor(cache_dir / f"layer_{layer:02d}_q_raw.pt")
    k_raw = load_tensor(cache_dir / f"layer_{layer:02d}_k_raw.pt")
    if q_raw.shape[0] != 1 or k_raw.shape[0] != 1:
        raise ValueError("This simple implementation assumes batch size 1.")
    if int(q_raw.shape[1]) != T or int(k_raw.shape[1]) != T:
        raise ValueError("Q/K cache sequence length does not match metadata.")

    q_head_raw = q_raw.view(1, T, Hq, D)[0, :, head, :].to(device=device, dtype=compute_dtype)
    k_head_raw = k_raw.view(1, T, Hkv, D)[0, :, kv_head, :].to(device=device, dtype=compute_dtype)
    del q_raw, k_raw

    norms = torch.load(cache_dir / f"layer_{layer:02d}_qk_norms.pt", map_location="cpu")
    q_weight = norms["q_norm_weight"].to(device=device, dtype=compute_dtype)
    k_weight = norms["k_norm_weight"].to(device=device, dtype=compute_dtype)
    q = qwen3_rmsnorm(q_head_raw, q_weight, float(norms["q_norm_eps"]))
    k = qwen3_rmsnorm(k_head_raw, k_weight, float(norms["k_norm_eps"]))
    del q_head_raw, k_head_raw

    position_ids = load_tensor(cache_dir / "position_ids.pt")[0].to(device=device)
    cos, sin = qwen3_default_rope_cos_sin(
        position_ids=position_ids,
        head_dim=D,
        rope_theta=get_rope_theta(config),
        dtype=compute_dtype,
        device=device,
    )
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    info = {
        "seq_len": T,
        "head_dim": D,
        "num_attention_heads": Hq,
        "num_key_value_heads": Hkv,
        "num_key_value_groups": n_rep,
        "kv_head": kv_head,
        "scaling": D ** -0.5,
        "config": config,
        "metadata": meta,
    }
    return q, k, info


def block_span_mask(intervals: Sequence[Tuple[int, int]], ks: int, ke: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(ke - ks, dtype=torch.bool, device=device)
    for s, e in intervals:
        lo = max(s, ks)
        hi = min(e, ke)
        if hi > lo:
            mask[lo - ks : hi - ks] = True
    return mask


@torch.no_grad()
def compute_query_block_stats(
    *,
    q_block: torch.Tensor,  # [M, D]
    k: torch.Tensor,  # [T, D]
    query_positions: torch.Tensor,  # [M], absolute token indices
    spans: Dict[str, List[Tuple[int, int]]],
    key_padding_mask: Optional[torch.Tensor],  # [T] bool or None
    scaling: float,
    key_block_size: int,
    topk: int = 0,
    local_windows: Sequence[int] = (),
    return_rows: bool = False,
) -> Dict:
    device = q_block.device
    T = k.shape[0]
    M = q_block.shape[0]
    query_positions = query_positions.to(device=device, dtype=torch.long)

    row_max = torch.full((M,), -torch.inf, dtype=torch.float32, device=device)
    if topk > 0:
        k_eff = min(int(topk), T)
        top_vals = torch.full((M, k_eff), -torch.inf, dtype=torch.float32, device=device)
        top_idx = torch.full((M, k_eff), -1, dtype=torch.long, device=device)
    else:
        top_vals = top_idx = None

    # Pass 1: stable row max and top-k logits.
    for ks in range(0, T, key_block_size):
        ke = min(ks + key_block_size, T)
        key_positions = torch.arange(ks, ke, device=device, dtype=torch.long)
        scores = (q_block.float() @ k[ks:ke].float().T) * float(scaling)  # [M, Bk]
        allowed = key_positions[None, :] <= query_positions[:, None]
        if key_padding_mask is not None:
            allowed = allowed & key_padding_mask[ks:ke][None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        row_max = torch.maximum(row_max, scores.max(dim=-1).values)

        if topk > 0:
            candidate_vals = torch.cat([top_vals, scores], dim=-1)
            candidate_idx = torch.cat(
                [top_idx, key_positions[None, :].expand(M, -1)],
                dim=-1,
            )
            new_vals, gather_pos = torch.topk(candidate_vals, k=top_vals.shape[1], dim=-1)
            top_vals = new_vals
            top_idx = candidate_idx.gather(dim=-1, index=gather_pos)

    denom = torch.zeros((M,), dtype=torch.float32, device=device)
    weighted_score_sum = torch.zeros((M,), dtype=torch.float32, device=device)
    span_numer = {name: torch.zeros((M,), dtype=torch.float32, device=device) for name in spans}
    local_numer = {int(w): torch.zeros((M,), dtype=torch.float32, device=device) for w in local_windows}
    rows = torch.empty((M, T), dtype=torch.float32, device="cpu") if return_rows else None

    # Pass 2: denominator and statistics.
    for ks in range(0, T, key_block_size):
        ke = min(ks + key_block_size, T)
        key_positions = torch.arange(ks, ke, device=device, dtype=torch.long)
        scores = (q_block.float() @ k[ks:ke].float().T) * float(scaling)
        allowed = key_positions[None, :] <= query_positions[:, None]
        if key_padding_mask is not None:
            allowed = allowed & key_padding_mask[ks:ke][None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        safe_scores = scores.masked_fill(~allowed, 0.0)
        weights = torch.exp(scores - row_max[:, None]).masked_fill(~allowed, 0.0)

        denom += weights.sum(dim=-1)
        weighted_score_sum += (weights * safe_scores).sum(dim=-1)

        for name, intervals in spans.items():
            smask = block_span_mask(intervals, ks, ke, device)
            if smask.any():
                span_numer[name] += weights[:, smask].sum(dim=-1)

        for w in local_numer:
            local_start = torch.clamp(query_positions - int(w) + 1, min=0)
            lmask = (key_positions[None, :] >= local_start[:, None]) & (
                key_positions[None, :] <= query_positions[:, None]
            )
            local_numer[w] += weights.masked_fill(~lmask, 0.0).sum(dim=-1)

        if rows is not None:
            probs = weights / denom.clamp_min(1e-30)[:, None]  # temporary denominator so far is wrong here
            # We cannot fill rows until final denom is known, so write unnormalized
            # weights now and normalize after the loop.
            rows[:, ks:ke] = weights.cpu()

    entropy = row_max + torch.log(denom.clamp_min(1e-30)) - (weighted_score_sum / denom.clamp_min(1e-30))
    span_mass = {name: (num / denom.clamp_min(1e-30)).detach().cpu() for name, num in span_numer.items()}
    local_mass = {str(w): (num / denom.clamp_min(1e-30)).detach().cpu() for w, num in local_numer.items()}

    out = {
        "entropy": entropy.detach().cpu(),
        "span_mass": span_mass,
        "local_mass": local_mass,
    }

    if topk > 0:
        top_probs = torch.exp(top_vals - row_max[:, None]) / denom.clamp_min(1e-30)[:, None]
        out["topk_values"] = top_vals.detach().cpu()
        out["topk_indices"] = top_idx.detach().cpu()
        out["topk_probs"] = top_probs.detach().cpu()

    if rows is not None:
        rows = rows / denom.detach().cpu().clamp_min(1e-30)[:, None]
        out["rows"] = rows

    return out


def normalize_query_positions(spec: Dict, T: int, metadata: Dict) -> Tuple[Dict[str, List[int]], List[int]]:
    raw = spec.get("query_positions") or {}
    if not raw:
        raw = {"last": [T - 1]}
    normalized: Dict[str, List[int]] = {}
    all_positions = set()
    for label, vals in raw.items():
        if vals == "last":
            vals = [T - 1]
        elif vals == "auto_special":
            input_ids = load_tensor(Path(metadata["cache_dir"]) / "input_ids.pt")[0].tolist()
            special_ids = set(metadata.get("special_token_ids") or [])
            vals = [i for i, tok in enumerate(input_ids) if tok in special_ids]
        elif isinstance(vals, int):
            vals = [vals]
        elif vals is None:
            vals = []
        positions = []
        for v in vals:
            p = resolve_index(int(v), T)
            if 0 <= p < T:
                positions.append(p)
                all_positions.add(p)
        normalized[str(label)] = sorted(set(positions))
    return normalized, sorted(all_positions)


def normalize_spans(spec: Dict, T: int) -> Dict[str, List[Tuple[int, int]]]:
    raw_spans = spec.get("spans") or {"bos": [[0, 1]]}
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for name, raw in raw_spans.items():
        intervals = normalize_intervals(raw, T)
        if intervals:
            spans[str(name)] = intervals
    return spans


def token_repr(tokens: Optional[List[str]], input_ids: List[int], pos: int) -> Dict:
    out = {"position": int(pos), "token_id": int(input_ids[pos])}
    if tokens is not None and 0 <= pos < len(tokens):
        out["token"] = tokens[pos]
    return out


def summarize_selected_rows(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: Dict,
    cache_dir: Path,
    spec: Dict,
    layer: int,
    head: int,
    key_block_size: int,
    topk: int,
    save_full_rows: bool,
) -> Tuple[Dict, Optional[Path]]:
    T = int(info["seq_len"])
    metadata = info["metadata"]
    metadata = dict(metadata)
    metadata["cache_dir"] = str(cache_dir)
    query_by_label, all_q_positions = normalize_query_positions(spec, T, metadata)
    spans = normalize_spans(spec, T)
    local_windows = [int(w) for w in (spec.get("local_windows") or [])]

    attention_mask_path = cache_dir / "attention_mask.pt"
    if attention_mask_path.exists():
        key_padding_mask = load_tensor(attention_mask_path)[0].to(device=q.device).bool()
    else:
        key_padding_mask = None

    input_ids = load_tensor(cache_dir / "input_ids.pt")[0].tolist()
    tokens_path = cache_dir / "tokens.json"
    tokens = json.loads(tokens_path.read_text(encoding="utf-8")) if tokens_path.exists() else None

    if not all_q_positions:
        return {"query_positions": query_by_label, "rows": {}}, None

    qpos_tensor = torch.tensor(all_q_positions, dtype=torch.long, device=q.device)
    q_block = q[qpos_tensor]
    stats = compute_query_block_stats(
        q_block=q_block,
        k=k,
        query_positions=qpos_tensor,
        spans=spans,
        key_padding_mask=key_padding_mask,
        scaling=float(info["scaling"]),
        key_block_size=key_block_size,
        topk=topk,
        local_windows=local_windows,
        return_rows=save_full_rows,
    )

    pos_to_row = {p: i for i, p in enumerate(all_q_positions)}
    rows_out: Dict[str, Dict] = {}
    for pos in all_q_positions:
        i = pos_to_row[pos]
        labels = [label for label, positions in query_by_label.items() if pos in positions]
        item = token_repr(tokens, input_ids, pos)
        item["labels"] = labels
        item["entropy_nats"] = float(stats["entropy"][i].item())
        item["span_mass"] = {name: float(vals[i].item()) for name, vals in stats["span_mass"].items()}
        item["local_window_mass"] = {name: float(vals[i].item()) for name, vals in stats["local_mass"].items()}
        if topk > 0:
            top_items = []
            vals = stats["topk_values"][i].tolist()
            idxs = stats["topk_indices"][i].tolist()
            probs = stats["topk_probs"][i].tolist()
            for logit, idx, prob in zip(vals, idxs, probs):
                if idx < 0 or not math.isfinite(float(logit)):
                    continue
                tok = token_repr(tokens, input_ids, int(idx))
                tok["logit"] = float(logit)
                tok["prob"] = float(prob)
                top_items.append(tok)
            item["topk"] = top_items
        rows_out[str(pos)] = item

    row_path = None
    if save_full_rows and "rows" in stats:
        row_path = cache_dir / f"attention_rows_layer_{layer:02d}_head_{head:02d}.pt"
        torch.save({"query_positions": all_q_positions, "rows": stats["rows"]}, row_path)

    return {
        "query_positions_by_label": query_by_label,
        "rows": rows_out,
    }, row_path


@torch.no_grad()
def average_window_received(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: Dict,
    cache_dir: Path,
    spec: Dict,
    key_block_size: int,
    query_block_size: int,
) -> Dict[str, Dict]:
    T = int(info["seq_len"])
    spans = normalize_spans(spec, T)
    attention_mask_path = cache_dir / "attention_mask.pt"
    if attention_mask_path.exists():
        key_padding_mask = load_tensor(attention_mask_path)[0].to(device=q.device).bool()
    else:
        key_padding_mask = None

    results: Dict[str, Dict] = {}
    entries = spec.get("window_received") or []
    for entry in entries:
        name = str(entry.get("name") or entry.get("span_name") or "window")
        if "span_name" in entry:
            span_name = str(entry["span_name"])
            if span_name not in spans:
                raise ValueError(f"window_received entry refers to unknown span_name={span_name!r}")
            intervals = spans[span_name]
        else:
            intervals = normalize_intervals(entry.get("span"), T)
        if not intervals:
            continue

        default_start = max(e for _, e in intervals)
        q_start = resolve_index(entry.get("query_start", default_start), T, end_exclusive=False)
        q_end = resolve_index(entry.get("query_end", None), T, end_exclusive=True)
        q_start, q_end = clamp_interval(q_start, q_end, T)
        if q_end <= q_start:
            results[name] = {
                "intervals": intervals,
                "query_range": [q_start, q_end],
                "num_queries": 0,
                "mean_mass": None,
                "sum_mass": 0.0,
            }
            continue

        q_positions_all = torch.arange(q_start, q_end, dtype=torch.long, device=q.device)
        if key_padding_mask is not None:
            q_positions_all = q_positions_all[key_padding_mask[q_positions_all]]

        total_mass = 0.0
        total_entropy = 0.0
        count = 0
        for qb in range(0, int(q_positions_all.numel()), query_block_size):
            qpos = q_positions_all[qb : qb + query_block_size]
            if qpos.numel() == 0:
                continue
            stats = compute_query_block_stats(
                q_block=q[qpos],
                k=k,
                query_positions=qpos,
                spans={"target": intervals},
                key_padding_mask=key_padding_mask,
                scaling=float(info["scaling"]),
                key_block_size=key_block_size,
                topk=0,
                local_windows=(),
                return_rows=False,
            )
            masses = stats["span_mass"]["target"]
            entropy = stats["entropy"]
            total_mass += float(masses.sum().item())
            total_entropy += float(entropy.sum().item())
            count += int(masses.numel())

        results[name] = {
            "intervals": [[s, e] for s, e in intervals],
            "query_range": [q_start, q_end],
            "num_queries": count,
            "mean_mass": (total_mass / count) if count else None,
            "sum_mass": total_mass,
            "mean_entropy_nats": (total_entropy / count) if count else None,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Qwen3 Q/K cache for one layer/head.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--spec-json", default=None, help="Defaults to cache_dir/analysis_spec.json")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", default="fp32", choices=list(COMPUTE_DTYPES.keys()))
    parser.add_argument("--key-block-size", type=int, default=8192)
    parser.add_argument("--query-block-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--save-full-rows", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    spec_path = Path(args.spec_json) if args.spec_json is not None else cache_dir / "analysis_spec.json"
    spec = load_json(spec_path) if spec_path.exists() else {"query_positions": {"last": [-1]}, "spans": {"bos": [[0, 1]]}}

    device = torch.device(args.device)
    compute_dtype = COMPUTE_DTYPES[args.compute_dtype]
    q, k, info = reconstruct_single_head_qk(
        cache_dir=cache_dir,
        layer=args.layer,
        head=args.head,
        device=device,
        compute_dtype=compute_dtype,
    )

    selected_rows, row_path = summarize_selected_rows(
        q=q,
        k=k,
        info=info,
        cache_dir=cache_dir,
        spec=spec,
        layer=args.layer,
        head=args.head,
        key_block_size=args.key_block_size,
        topk=args.topk,
        save_full_rows=args.save_full_rows,
    )

    received = average_window_received(
        q=q,
        k=k,
        info=info,
        cache_dir=cache_dir,
        spec=spec,
        key_block_size=args.key_block_size,
        query_block_size=args.query_block_size,
    )

    result = {
        "cache_dir": str(cache_dir),
        "layer": args.layer,
        "head": args.head,
        "kv_head": info["kv_head"],
        "num_key_value_groups": info["num_key_value_groups"],
        "seq_len": info["seq_len"],
        "head_dim": info["head_dim"],
        "selected_query_rows": selected_rows,
        "window_received": received,
        "full_rows_path": str(row_path) if row_path is not None else None,
    }

    out_path = Path(args.out_json) if args.out_json is not None else cache_dir / f"stats_layer_{args.layer:02d}_head_{args.head:02d}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()

```

---

## 10. Validation against Hugging Face eager attention

Before trusting a new model/version, run a short sequence test, for example `T <= 256`.

1. Capture one layer using eager attention:

```bash
python capture_qk_qwen3.py   --model-name Qwen/Qwen3-8B   --prompt "A short test prompt for attention validation."   --out-dir qk_cache_validate   --layers 0   --attn-implementation eager   --save-dtype fp32
```

2. Run the analyzer and save full rows:

```bash
python analyze_qk_qwen3.py   --cache-dir qk_cache_validate   --layer 0   --head 0   --save-full-rows   --out-json validate_stats.json
```

3. Separately run Hugging Face with `output_attentions=True` and compare the same rows:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"
cache_dir = "qk_cache_validate"
layer = 0
head = 0

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = open(f"{cache_dir}/model_text.txt", encoding="utf-8").read()
inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to("cuda")
position_ids = torch.arange(inputs["input_ids"].shape[1], device="cuda").unsqueeze(0)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
with torch.inference_mode():
    outputs = model(
        **inputs,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
        logits_to_keep=1,
    )

hf_attn = outputs.attentions[layer][0, head].detach().cpu()  # [T, T]
row_pack = torch.load(f"{cache_dir}/attention_rows_layer_00_head_00.pt")
rows = row_pack["rows"]
qpos = row_pack["query_positions"]

for i, q in enumerate(qpos):
    diff = (rows[i] - hf_attn[q]).abs()
    print(q, diff.max().item(), diff.mean().item())
```

If the difference is large, check in this order:

```text
1. Did you use the same prompt text and tokenization?
2. Did you pass the same position_ids?
3. Did you include q_norm/k_norm?
4. Did you use the correct GQA mapping: kv_head = q_head // 4 for Qwen3-8B?
5. Did you use default Qwen3 RoPE, not Llama-style assumptions with a different theta?
6. Are you accidentally comparing fp32 eager against bf16 saved tensors?
```

---

## 11. Limitations and extensions

### Default RoPE only

The analyzer implements the released `Qwen/Qwen3-8B` default RoPE. It deliberately raises an error if `rope_scaling` or non-default `rope_parameters` appear. If you enable YaRN or other long-context RoPE scaling, compute `cos/sin` using the exact Hugging Face `Qwen3RotaryEmbedding` / `modeling_rope_utils` path or extend `qwen3_default_rope_cos_sin`.

### Batch size 1

The scripts assume one sequence. For multiple sequences, save batch-indexed files and run the analysis per batch item.

### Prefill/full forward only

The capture script is for one full forward pass. During autoregressive generation, hooks fire once per decode step and the current filenames would be overwritten. For generation analysis, add a step index to saved filenames or capture only the prefill prompt.

### No multimodal Qwen3-VL

This is scoped to text-only `Qwen/Qwen3-8B`. Qwen-VL models may use multimodal RoPE/position conventions.

### No full dense attention by default

The analyzer computes targeted rows and span/window statistics. It can save full rows for critical tokens, but it does not save `[T, T]` for all query positions.

### Speed

The analysis is exact but simple. It uses blockwise PyTorch matmul, not Triton. For repeated large sweeps, the next optimization would be to process multiple heads together or write a custom kernel/reducer.

---

## 12. References checked

1. Hugging Face Qwen3 model documentation: https://huggingface.co/docs/transformers/en/model_doc/qwen3
2. Hugging Face Qwen3-8B model card/config: https://huggingface.co/Qwen/Qwen3-8B
3. Hugging Face Qwen3 implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
4. Hugging Face AttentionInterface documentation: https://huggingface.co/docs/transformers/en/attention_interface
5. BertViz GitHub repository: https://github.com/jessevig/bertviz
6. TransformerLens GitHub repository: https://github.com/TransformerLensOrg/TransformerLens
7. FlashAttention paper: https://arxiv.org/abs/2205.14135
8. FlashAttention GitHub repository: https://github.com/Dao-AILab/flash-attention
9. PyTorch `scaled_dot_product_attention` documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
10. StreamingLLM / Attention Sinks paper: https://arxiv.org/abs/2309.17453
