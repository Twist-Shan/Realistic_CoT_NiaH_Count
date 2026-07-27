#!/usr/bin/env python3
"""Add audited, passage-redacted prompt formats to a Realistic NiaH report.

The frozen requests are authoritative: every rendered example comes directly
from a requests.jsonl record, not from a model-name heuristic.  The passage is
replaced with a fixed placeholder in all new artifacts; hashes still refer to
the full frozen rendered prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS = [
    "Qwen3-8B",
    "Qwen3-1.7B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "OLMo-Hybrid-7B",
    "Llama3.1-8B",
    "Llama3.2-3B",
]
CAPABLE_MODELS = {
    "Qwen3-8B",
    "Qwen3-1.7B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
}
PROMPT_MODES = ("direct", "enumeration", "native_thinking")
QUERY_ORDERS = ("query_first", "query_last")
FAMILIES = {
    "Qwen3-8B": "qwen3",
    "Qwen3-1.7B": "qwen3",
    "Qwen3-32B": "qwen3",
    "Gemma4-E4B": "gemma4",
    "Gemma4-12B": "gemma4",
    "OLMo-Hybrid-7B": "olmo",
    "Llama3.1-8B": "llama3",
    "Llama3.2-3B": "llama3",
}

DIRECT_TASK = """The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

How many city-score audit records are in the passage?
In the final answer, output exactly one line in this form:
Total: <integer>"""

ENUMERATION_TASK = """The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

Find every city-score audit record in the passage. In passage order, output
one record per line as:
<k>. <city>: <score>
where k starts at 1 and increases by 1.
Then output one final line:
Total: <integer>
Do not include any other text."""

MODE_LABELS = {
    "direct": "直接计数（thinking 关）",
    "enumeration": "逐条枚举（thinking 关）",
    "native_thinking": "直接计数（原生 thinking 开）",
}
ORDER_LABELS = {
    "query_first": "任务在前、passage 在后",
    "query_last": "passage 在前、任务在后",
}

WRAPPER_SUMMARIES = {
    "Qwen3-8B": (
        "<|im_start|>user … <|im_end|> → <|im_start|>assistant",
        "无 tokenizer 注入的 system message。",
        "thinking 关时 assistant 起始处预置空 <think></think>；native thinking 时不预置闭合块。",
    ),
    "Qwen3-1.7B": (
        "<|im_start|>user … <|im_end|> → <|im_start|>assistant",
        "无 tokenizer 注入的 system message。",
        "thinking 关时 assistant 起始处预置空 <think></think>；native thinking 时不预置闭合块。",
    ),
    "Qwen3-32B": (
        "<|im_start|>user … <|im_end|> → <|im_start|>assistant",
        "无 tokenizer 注入的 system message。",
        "thinking 关时 assistant 起始处预置空 <think></think>；native thinking 时不预置闭合块。",
    ),
    "Gemma4-E4B": (
        "<bos><|turn>user … <turn|> → <|turn>model",
        "native thinking 时自动增加 system turn：<|think|>；关闭时无 system turn。",
        "关闭时停在 model generation marker；native thinking 由 system <|think|> turn 开启。",
    ),
    "Gemma4-12B": (
        "<bos><|turn>user … <turn|> → <|turn>model",
        "native thinking 时自动增加 system turn：<|think|>；关闭时无 system turn。",
        "关闭时 model marker 后仍出现空 thought-channel 后缀；native thinking 改用 system <|think|> turn。",
    ),
    "OLMo-Hybrid-7B": (
        "ChatML system → user → assistant",
        "自动注入 function-calling assistant system 文本（声明当前无 functions）。",
        "本实验未注册 native thinking；direct 与 enumeration 均为 thinking 关。",
    ),
    "Llama3.1-8B": (
        "<|begin_of_text|> + system/user/assistant header",
        "自动注入 knowledge date 与 Today Date: 26 Jul 2024。",
        "本实验未注册 native thinking；日期是冻结 tokenizer 模板文本，不是用户任务内容。",
    ),
    "Llama3.2-3B": (
        "<|begin_of_text|> + system/user/assistant header",
        "自动注入 knowledge date 与 Today Date: 24 Jul 2026。",
        "本实验未注册 native thinking；日期是冻结 tokenizer 模板文本，不是用户任务内容。",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_passage(value: str) -> str:
    redacted, count = re.subn(
        r"<passage>\r?\n.*?\r?\n</passage>",
        "<passage>\n[PASSAGE OMITTED]\n</passage>",
        value,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Expected exactly one <passage> block")
    return redacted


def expected_keys(model: str) -> set[tuple[str, str]]:
    modes = PROMPT_MODES if model in CAPABLE_MODELS else PROMPT_MODES[:2]
    return {(mode, order) for mode in modes for order in QUERY_ORDERS}


def enable_thinking_argument(model: str, mode: str) -> str:
    if model in CAPABLE_MODELS:
        return "true" if mode == "native_thinking" else "false"
    return "not passed"


def scan_request_sources(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total_rows = 0
    source_checks: list[dict[str, Any]] = []
    observed_models = [item["model"] for item in manifest["request_sources"]]
    if observed_models != MODELS:
        raise ValueError(
            f"Unexpected request-source model order: {observed_models}"
        )

    for source in manifest["request_sources"]:
        model = source["model"]
        path = Path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(path)

        digest = hashlib.sha256()
        counts: Counter[tuple[str, str]] = Counter()
        samples: dict[tuple[str, str], dict[str, Any]] = {}
        model_ids: set[str] = set()
        roles_seen: set[tuple[str, ...]] = set()
        decoding_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
        rows = 0

        with path.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                obj = json.loads(raw_line)
                rows += 1
                total_rows += 1
                if obj["model_label"] != model:
                    raise ValueError(
                        f"{path}: model_label {obj['model_label']} != {model}"
                    )
                key = (obj["prompt_mode"], obj["query_order"])
                counts[key] += 1
                samples.setdefault(key, obj)
                model_ids.add(str(obj["model_id"]))
                roles = tuple(item["role"] for item in obj["messages"])
                roles_seen.add(roles)
                decoding_signature = {
                    name: value
                    for name, value in obj["decoding"].items()
                    if name != "seed"
                }
                decoding_seen[key].add(
                    json.dumps(decoding_signature, sort_keys=True)
                )

        observed_sha = digest.hexdigest()
        if observed_sha != source["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")
        if rows != int(source["rows"]):
            raise ValueError(
                f"Row mismatch for {path}: {rows} != {source['rows']}"
            )
        if set(counts) != expected_keys(model):
            raise ValueError(
                f"{model}: observed combinations {sorted(counts)}"
            )
        if set(counts.values()) != {150}:
            raise ValueError(f"{model}: each combination must have n=150")
        if len(model_ids) != 1:
            raise ValueError(f"{model}: unstable model_id {model_ids}")
        if roles_seen != {("user",)}:
            raise ValueError(f"{model}: message roles {roles_seen}")
        if any(len(values) != 1 for values in decoding_seen.values()):
            raise ValueError(f"{model}: decoding changed within a combination")

        model_id = next(iter(model_ids))
        modes = [
            mode
            for mode in PROMPT_MODES
            if any(key[0] == mode for key in counts)
        ]
        wrapper, injected, thinking = WRAPPER_SUMMARIES[model]
        summaries.append(
            {
                "model_label": model,
                "model_id": model_id,
                "family": FAMILIES[model],
                "registered_prompt_modes": ", ".join(modes),
                "native_thinking_supported": model in CAPABLE_MODELS,
                "message_roles": "user",
                "query_orders": ", ".join(QUERY_ORDERS),
                "tokenizer_wrapper_summary": wrapper,
                "tokenizer_injected_system_text": injected,
                "thinking_template_behavior": thinking,
                "source_requests_file": str(path),
                "source_requests_sha256": observed_sha,
            }
        )

        for mode in PROMPT_MODES:
            for order in QUERY_ORDERS:
                key = (mode, order)
                if key not in samples:
                    continue
                obj = samples[key]
                if len(obj["messages"]) != 1:
                    raise ValueError(f"{model} {key}: expected one message")
                message = str(obj["messages"][0]["content"])
                rendered = str(obj["rendered_prompt"])
                expected_task = (
                    ENUMERATION_TASK if mode == "enumeration" else DIRECT_TASK
                )
                if expected_task not in message or expected_task not in rendered:
                    raise ValueError(f"{model} {key}: task block mismatch")
                decoding = obj["decoding"]
                examples.append(
                    {
                        "model_label": model,
                        "model_id": model_id,
                        "family": FAMILIES[model],
                        "prompt_mode": mode,
                        "prompt_mode_label": MODE_LABELS[mode],
                        "thinking_enabled": mode == "native_thinking",
                        "query_order": order,
                        "query_order_label": ORDER_LABELS[order],
                        "request_count": counts[key],
                        "message_roles": "user",
                        "tokenize": False,
                        "add_generation_prompt": True,
                        "enable_thinking_argument": enable_thinking_argument(
                            model, mode
                        ),
                        "max_tokens": decoding["max_tokens"],
                        "temperature": decoding["temperature"],
                        "top_p": decoding["top_p"],
                        "top_k": decoding["top_k"],
                        "min_p": decoding["min_p"],
                        "sample_request_id": obj["request_id"],
                        "sample_full_rendered_prompt_sha256": sha256_text(
                            rendered
                        ),
                        "user_message_redacted": redact_passage(message),
                        "rendered_prompt_redacted": redact_passage(rendered),
                    }
                )

        source_checks.append(
            {
                "model": model,
                "path": str(path),
                "rows": rows,
                "sha256": observed_sha,
                "status": "pass",
            }
        )

    if total_rows != 6300:
        raise ValueError(f"Expected 6300 requests, got {total_rows}")
    if len(examples) != 42:
        raise ValueError(f"Expected 42 combinations, got {len(examples)}")
    validate_observed_templates(examples)
    return summaries, examples, {
        "status": "pass",
        "requests": total_rows,
        "models": len(summaries),
        "combinations": len(examples),
        "sources": source_checks,
    }


def example_map(
    examples: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (
            str(item["model_label"]),
            str(item["prompt_mode"]),
            str(item["query_order"]),
        ): item
        for item in examples
    }


def validate_observed_templates(examples: list[dict[str, Any]]) -> None:
    lookup = example_map(examples)
    for model in MODELS[:3]:
        for mode in ("direct", "enumeration"):
            for order in QUERY_ORDERS:
                rendered = lookup[(model, mode, order)][
                    "rendered_prompt_redacted"
                ]
                if "<think>\n\n</think>" not in rendered:
                    raise ValueError(f"{model} {mode} missing empty think block")
        for order in QUERY_ORDERS:
            rendered = lookup[(model, "native_thinking", order)][
                "rendered_prompt_redacted"
            ]
            if "<think>\n\n</think>" in rendered:
                raise ValueError(f"{model} native prompt pre-closes thinking")

    for model in ("Gemma4-E4B", "Gemma4-12B"):
        for order in QUERY_ORDERS:
            rendered = lookup[(model, "native_thinking", order)][
                "rendered_prompt_redacted"
            ]
            if "<|turn>system\n<|think|>\n<turn|>" not in rendered:
                raise ValueError(f"{model} native system think turn missing")

    for mode in ("direct", "enumeration"):
        for order in QUERY_ORDERS:
            e4b = lookup[("Gemma4-E4B", mode, order)][
                "rendered_prompt_redacted"
            ]
            b12 = lookup[("Gemma4-12B", mode, order)][
                "rendered_prompt_redacted"
            ]
            if "<|channel>thought" in e4b:
                raise ValueError("Gemma4-E4B unexpectedly has thought suffix")
            if "<|channel>thought" not in b12:
                raise ValueError("Gemma4-12B missing observed thought suffix")

    olmo_system = (
        "You are a helpful function-calling AI assistant. You do not currently "
        "have access to any functions. <functions></functions>"
    )
    for mode in ("direct", "enumeration"):
        for order in QUERY_ORDERS:
            if olmo_system not in lookup[
                ("OLMo-Hybrid-7B", mode, order)
            ]["rendered_prompt_redacted"]:
                raise ValueError("OLMo tokenizer-injected system text missing")

    dates = {
        "Llama3.1-8B": "Today Date: 26 Jul 2024",
        "Llama3.2-3B": "Today Date: 24 Jul 2026",
    }
    for model, date_text in dates.items():
        for mode in ("direct", "enumeration"):
            for order in QUERY_ORDERS:
                if date_text not in lookup[(model, mode, order)][
                    "rendered_prompt_redacted"
                ]:
                    raise ValueError(f"{model}: frozen date text missing")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_html(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table class="data-table prompt-matrix">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def rendered_examples_html(
    model: str,
    examples: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    model_examples = [
        item for item in examples if item["model_label"] == model
    ]
    for item in model_examples:
        decoding = (
            f"n={item['request_count']} · max_tokens={item['max_tokens']} · "
            f"temperature={item['temperature']} · "
            f"top_p={item['top_p']} · top_k={item['top_k']}"
        )
        summary = (
            f"{item['prompt_mode']} × {item['query_order']} · {decoding}"
        )
        message_code = html.escape(str(item["user_message_redacted"]))
        rendered_code = html.escape(str(item["rendered_prompt_redacted"]))
        blocks.append(
            f"""
      <details class="prompt-example-combination">
        <summary>{html.escape(summary)}</summary>
        <p class="prompt-meta"><strong>message 层</strong>：程序构造的唯一 user message；passage 已脱敏。</p>
        <pre class="prompt-code"><code>{message_code}</code></pre>
        <p class="prompt-meta"><strong>rendered 层</strong>：实际冻结的 tokenizer chat-template 输出；passage 已脱敏。</p>
        <pre class="prompt-code"><code>{rendered_code}</code></pre>
        <p class="prompt-meta">样例 request_id：<code>{html.escape(str(item['sample_request_id']))}</code><br>
        完整 rendered prompt SHA256：<code>{item['sample_full_rendered_prompt_sha256']}</code></p>
      </details>"""
        )
    return "\n".join(blocks)


def build_prompt_section(
    summaries: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    summary_rows = [
        [
            item["model_label"],
            item["model_id"],
            item["registered_prompt_modes"],
            item["tokenizer_wrapper_summary"],
            item["tokenizer_injected_system_text"],
            item["thinking_template_behavior"],
        ]
        for item in summaries
    ]
    summary_table = table_html(
        [
            "模型",
            "Hugging Face model_id",
            "注册模式",
            "rendered wrapper",
            "tokenizer 注入内容",
            "thinking 行为",
        ],
        summary_rows,
    )
    model_blocks: list[str] = []
    by_model = {item["model_label"]: item for item in summaries}
    for model in MODELS:
        item = by_model[model]
        model_blocks.append(
            f"""
  <details class="prompt-model">
    <summary>{html.escape(model)}｜{html.escape(str(item['model_id']))}</summary>
    <p><strong>wrapper：</strong>{html.escape(str(item['tokenizer_wrapper_summary']))}<br>
    <strong>tokenizer 注入：</strong>{html.escape(str(item['tokenizer_injected_system_text']))}<br>
    <strong>thinking：</strong>{html.escape(str(item['thinking_template_behavior']))}</p>
    {rendered_examples_html(model, examples)}
  </details>"""
        )

    direct = html.escape(DIRECT_TASK)
    enumeration = html.escape(ENUMERATION_TASK)
    query_first = html.escape(
        "[TASK BLOCK]\n\n<passage>\n[PASSAGE]\n</passage>"
    )
    query_last = html.escape(
        "<passage>\n[PASSAGE]\n</passage>\n\n[TASK BLOCK]"
    )
    return f"""
<section id="prompt-formats">
  <h2>八个模型的实际 prompt 格式</h2>
  <p>本节以冻结的 6,300 条 <code>requests.jsonl</code> 为准，而不是根据模型名称猜测模板。每条请求在 <strong>message 层</strong>都只有一个 <code>user</code> message；随后调用 tokenizer chat template，形成真正送入模型的 <strong>rendered 层</strong>。后者可能自动注入 system 文本、角色 token、日期、thinking 标记或 generation marker。</p>
  <div class="callout"><strong>脱敏说明。</strong> 下列示例仅把 <code>&lt;passage&gt;…&lt;/passage&gt;</code> 内的长 haystack 替换为 <code>[PASSAGE OMITTED]</code>；任务文本、角色顺序、chat-template token、换行和 generation 后缀均按冻结请求保留。完整 prompt 未改写，其 SHA256 也逐项记录。42 个模型 × 模式 × query-order 组合各有 n=150。</div>

  <h3>所有模型共用的任务文字</h3>
  <div class="two-col prompt-grid">
    <div>
      <h4>direct 与 native_thinking</h4>
      <pre class="prompt-code"><code>{direct}</code></pre>
    </div>
    <div>
      <h4>enumeration</h4>
      <pre class="prompt-code"><code>{enumeration}</code></pre>
    </div>
  </div>
  <p class="table-note"><code>native_thinking</code> 与 <code>direct</code> 使用完全相同的用户任务文字；差异来自 chat-template 的 thinking 开关以及解码参数。</p>

  <h3>Query order</h3>
  <div class="two-col prompt-grid">
    <div><h4>query_first</h4><pre class="prompt-code"><code>{query_first}</code></pre></div>
    <div><h4>query_last</h4><pre class="prompt-code"><code>{query_last}</code></pre></div>
  </div>

  <h3>逐模型 chat-template 差异</h3>
  {summary_table}
  <p class="table-note">所有请求均使用 <code>tokenize=False</code> 与 <code>add_generation_prompt=True</code>。只有 Qwen3/Gemma4 家族传入 <code>enable_thinking</code>：direct/enumeration 为 false，native_thinking 为 true；Llama/OLMo 不传该参数。</p>

  <h3>逐模型、逐模式、逐顺序的冻结示例</h3>
  <p>展开任一模型即可查看它实际注册的全部 rendered 格式。完整 42 行宽表另存为 <code>tables/model_prompt_format_examples.csv</code>，结构化 JSON 位于 <code>prompt_formats/model_prompt_formats.json</code>。</p>
  {''.join(model_blocks)}

  <div class="callout warn"><strong>解释边界。</strong> Llama3.1 的 <code>Today Date: 26 Jul 2024</code> 与 Llama3.2 的 <code>Today Date: 24 Jul 2026</code> 是冻结 tokenizer 自动插入的 system 文本；OLMo 的 function-calling system 声明也是模板自动注入。它们不是实验作者写入的 counting 指令，也没有在报告中“修正”。</div>
</section>
"""


def update_report_html(report_path: Path, section: str) -> None:
    text = report_path.read_text(encoding="utf-8")
    if 'id="prompt-formats"' in text:
        raise RuntimeError("Prompt-format section already present")
    nav_anchor = '    <a href="#setup">实验设定</a>'
    if nav_anchor not in text:
        raise RuntimeError("Navigation insertion anchor missing")
    text = text.replace(
        nav_anchor,
        nav_anchor + '\n    <a href="#prompt-formats">Prompt 格式</a>',
        1,
    )
    section_anchor = '<section id="overall">'
    if section_anchor not in text:
        raise RuntimeError("Section insertion anchor missing")
    text = text.replace(section_anchor, section + "\n" + section_anchor, 1)
    css_anchor = "@media (max-width: 760px) {"
    if css_anchor not in text:
        raise RuntimeError("CSS insertion anchor missing")
    prompt_css = """
.prompt-code {
  margin: 10px 0 18px;
  padding: 14px 16px;
  border: 1px solid var(--line);
  background: var(--wash);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  overflow-x: auto;
  max-height: 38rem;
}
.prompt-code code { font-size: .78rem; line-height: 1.55; }
.prompt-model {
  border: 1px solid var(--line);
  border-left: 4px solid var(--accent-2);
  margin: 14px 0;
  padding: 0 16px 14px;
}
.prompt-example-combination {
  border-top: 1px solid var(--line);
  margin-top: 10px;
  padding: 0 10px 4px;
}
.prompt-example-combination summary { font-size: .92rem; }
.prompt-meta { color: var(--muted); font-size: .84rem; max-width: 110ch; }
.prompt-matrix { min-width: 1200px; }
.prompt-matrix td { text-align: left; min-width: 130px; }
.prompt-grid > div { min-width: 0; }
"""
    text = text.replace(css_anchor, prompt_css + css_anchor, 1)
    report_path.write_text(text, encoding="utf-8")


def update_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    addition = """

## Audited prompt formats

This report version adds every frozen model x prompt-mode x query-order format.
The HTML shows both the single-user-message layer and the tokenizer-rendered
layer. Passage bodies are replaced by `[PASSAGE OMITTED]`; full rendered-prompt
SHA256 values are retained. The complete 42-row matrix is available in
`tables/model_prompt_format_examples.csv` and
`prompt_formats/model_prompt_formats.json`.

Reproduce this addendum with:

```powershell
python scripts/build_prompt_format_addendum.py `
  --base-report <verified-base-report-directory> `
  --output-dir <new-empty-output-directory> `
  --repo-root <Realistic_CoT_NiaH_Count-repository>
```
"""
    path.write_text(text.rstrip() + addition, encoding="utf-8")


def update_manifest(
    path: Path,
    output_dir: Path,
    script_source: Path,
    code_sources: list[dict[str, Any]],
    source_audit: dict[str, Any],
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_root"] = str(output_dir)
    manifest["modified_at_utc"] = utc_now()
    scripts = manifest.setdefault("reproduction_scripts", [])
    scripts = [
        item
        for item in scripts
        if item.get("destination") != script_source.name
    ]
    scripts.append(
        {
            "source": str(script_source.resolve()),
            "destination": script_source.name,
            "sha256": sha256(script_source),
        }
    )
    manifest["reproduction_scripts"] = scripts
    manifest["prompt_formats_v1"] = {
        "created_at_utc": utc_now(),
        "authority": "frozen requests.jsonl rendered_prompt and messages fields",
        "requests": source_audit["requests"],
        "models": source_audit["models"],
        "registered_combinations": source_audit["combinations"],
        "requests_per_combination": 150,
        "message_layer": "exactly one user message",
        "rendering": {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": (
                "passed only for qwen3/gemma4; false for direct/enumeration "
                "and true for native_thinking"
            ),
        },
        "redaction": (
            "Only the <passage> body is replaced with [PASSAGE OMITTED] in "
            "new report artifacts; source requests are unmodified."
        ),
        "source_audit_status": source_audit["status"],
        "source_requests": source_audit["sources"],
        "code_sources": code_sources,
        "tables": [
            "model_prompt_format_summary.csv",
            "model_prompt_format_examples.csv",
        ],
        "structured_artifact": "prompt_formats/model_prompt_formats.json",
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def regenerate_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS.tsv"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{sha256(path)}\t{path.relative_to(root).as_posix()}"
        for path in files
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def main() -> None:
    started_at = utc_now()
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    base_report = args.base_report.resolve()
    output_dir = args.output_dir.resolve()
    repo_root = args.repo_root.resolve()
    if not base_report.is_dir():
        raise FileNotFoundError(base_report)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for relative in (
        "report.html",
        "README.md",
        "analysis_manifest.json",
        "SHA256SUMS.tsv",
    ):
        if not (base_report / relative).is_file():
            raise FileNotFoundError(base_report / relative)

    prompts_code = repo_root / "src" / "realistic_niah" / "prompts.py"
    runner_code = repo_root / "src" / "realistic_niah" / "runner.py"
    for path in (prompts_code, runner_code):
        if not path.is_file():
            raise FileNotFoundError(path)
    code_sources = [
        {"path": str(path), "sha256": sha256(path)}
        for path in (prompts_code, runner_code)
    ]

    manifest = json.loads(
        (base_report / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    summaries, examples, source_audit = scan_request_sources(manifest)

    shutil.copytree(base_report, output_dir)
    tables_dir = output_dir / "tables"
    scripts_dir = output_dir / "scripts"
    prompt_dir = output_dir / "prompt_formats"
    logs_dir = output_dir / "logs"
    prompt_dir.mkdir()

    write_csv(tables_dir / "model_prompt_format_summary.csv", summaries)
    write_csv(tables_dir / "model_prompt_format_examples.csv", examples)
    structured = {
        "schema_version": "realistic-niah-prompt-formats-v1",
        "created_at_utc": utc_now(),
        "authority": "frozen request records",
        "passage_redaction": {
            "enabled": True,
            "placeholder": "[PASSAGE OMITTED]",
            "only_passage_body_changed": True,
        },
        "rendering_call": {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": (
                "qwen3/gemma4 only: mode == native_thinking"
            ),
        },
        "task_blocks": {
            "direct_and_native_thinking": DIRECT_TASK,
            "enumeration": ENUMERATION_TASK,
        },
        "query_orders": {
            "query_first": (
                "[TASK BLOCK]\\n\\n<passage>\\n[PASSAGE]\\n</passage>"
            ),
            "query_last": (
                "<passage>\\n[PASSAGE]\\n</passage>\\n\\n[TASK BLOCK]"
            ),
        },
        "source_audit": source_audit,
        "code_sources": code_sources,
        "model_summaries": summaries,
        "examples": examples,
    }
    (prompt_dir / "model_prompt_formats.json").write_text(
        json.dumps(structured, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)
    validator_source = Path(__file__).with_name("validate_prompt_report.py")
    if validator_source.is_file():
        shutil.copy2(
            validator_source, scripts_dir / validator_source.name
        )

    section = build_prompt_section(summaries, examples)
    update_report_html(output_dir / "report.html", section)
    update_readme(output_dir / "README.md")
    update_manifest(
        output_dir / "analysis_manifest.json",
        output_dir,
        Path(__file__),
        code_sources,
        source_audit,
    )

    build_log = {
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "base_report": str(base_report),
        "output_dir": str(output_dir),
        "repo_root": str(repo_root),
        "requests_scanned": source_audit["requests"],
        "models": source_audit["models"],
        "registered_combinations": source_audit["combinations"],
        "passage_redaction": "pass",
        "source_integrity": source_audit["status"],
        "new_files": [
            "tables/model_prompt_format_summary.csv",
            "tables/model_prompt_format_examples.csv",
            "prompt_formats/model_prompt_formats.json",
            "scripts/build_prompt_format_addendum.py",
        ],
    }
    (logs_dir / "prompt_format_build_log.json").write_text(
        json.dumps(build_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_entries = regenerate_checksums(output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "requests": source_audit["requests"],
                "models": source_audit["models"],
                "registered_combinations": source_audit["combinations"],
                "checksum_entries": checksum_entries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
