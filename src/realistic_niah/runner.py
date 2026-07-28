from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .parsing import evaluate_generation
from .prompts import (
    build_messages,
    reasoning_expected,
    render_generation_prompt,
    resolve_model_spec,
)
from .spec import (
    ENUMERATION_PROMPT_MODES,
    QUERY_LAYOUT,
    THINKING_PROMPT_MODES,
    ModelSpec,
)
from .stimuli import load_stimuli, select_stimuli


@dataclass(frozen=True)
class DecodingConfig:
    max_tokens: int
    temperature: float
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0


@dataclass(frozen=True)
class EngineConfig:
    tensor_parallel_size: int = 1
    max_model_len: int = 32_768
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    enable_prefix_caching: bool = True
    request_batch_size: int = 32


def decoding_config(model_spec: ModelSpec, prompt_mode: str) -> DecodingConfig:
    if model_spec.reasoning_policy == "always_on":
        if prompt_mode not in model_spec.prompt_modes:
            raise ValueError(
                f"Unsupported decoding combination: "
                f"{model_spec.label}/{prompt_mode}"
            )
        if model_spec.family == "glm_z1":
            return DecodingConfig(
                max_tokens=4096,
                temperature=0.6,
                top_p=0.95,
                top_k=40,
            )
        if model_spec.family == "deepseek_r1_qwen3":
            return DecodingConfig(
                max_tokens=4096,
                temperature=0.6,
                top_p=0.95,
            )
        if model_spec.family == "olmo3":
            return DecodingConfig(
                max_tokens=4096,
                temperature=0.6,
                top_p=0.95,
            )
        raise ValueError(
            f"No always-on reasoning decoding is registered for "
            f"{model_spec.label}"
        )
    if prompt_mode == "direct":
        return DecodingConfig(max_tokens=64, temperature=0.0)
    if prompt_mode in ENUMERATION_PROMPT_MODES:
        return DecodingConfig(max_tokens=1536, temperature=0.0)
    if prompt_mode not in THINKING_PROMPT_MODES or not model_spec.native_thinking:
        raise ValueError(
            f"Unsupported decoding combination: {model_spec.label}/{prompt_mode}"
        )
    if model_spec.family == "gemma4":
        return DecodingConfig(
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
    return DecodingConfig(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )


def request_id(
    *,
    model_spec: ModelSpec,
    prompt_mode: str,
    query_layout: str,
    stimulus_id: str,
) -> str:
    return "/".join(
        (model_spec.label, prompt_mode, query_layout, stimulus_id)
    )


def build_requests(
    stimuli: Iterable[dict[str, Any]],
    *,
    model_spec: ModelSpec,
    prompt_modes: Iterable[str] | None = None,
    query_layout: str = QUERY_LAYOUT,
) -> list[dict[str, Any]]:
    modes = tuple(prompt_modes or model_spec.prompt_modes)
    supported = set(model_spec.prompt_modes)
    unsupported = sorted(set(modes) - supported)
    if unsupported:
        raise ValueError(
            f"{model_spec.label} does not support prompt modes: {unsupported}"
        )
    if query_layout != QUERY_LAYOUT:
        raise ValueError(f"Unsupported query layout: {query_layout}")

    requests: list[dict[str, Any]] = []
    for stimulus in stimuli:
        for prompt_mode in modes:
            messages = build_messages(
                stimulus["passage"],
                prompt_mode=prompt_mode,
                query_layout=query_layout,
            )
            requests.append(
                {
                    "request_id": request_id(
                        model_spec=model_spec,
                        prompt_mode=prompt_mode,
                        query_layout=query_layout,
                        stimulus_id=stimulus["stimulus_id"],
                    ),
                    "model_label": model_spec.label,
                    "model_id": model_spec.model_id,
                    "prompt_mode": prompt_mode,
                    "query_layout": query_layout,
                    "stimulus_id": stimulus["stimulus_id"],
                    "seed": int(stimulus["seed"]),
                    "messages": messages,
                    "stimulus": stimulus,
                }
            )
    ids = [item["request_id"] for item in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request IDs detected")
    return requests


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = str(row["request_id"])
        if identifier in completed:
            raise ValueError(f"Duplicate completed request: {identifier}")
        completed[identifier] = row
    return completed


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _ordered_id_digest(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evaluation_summary(
    completed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not completed:
        return {
            "parse_failures": 0,
            "truncations": 0,
            "overthinking_flags": 0,
            "exact_count_accuracy": None,
            "registered_accuracy": None,
        }
    return {
        "parse_failures": sum(
            row["evaluation"]["parse_status"] == "parse_fail"
            for row in completed.values()
        ),
        "truncations": sum(
            bool(row["evaluation"]["truncated"]) for row in completed.values()
        ),
        "overthinking_flags": sum(
            bool(row["evaluation"].get("overthinking_flag"))
            for row in completed.values()
        ),
        "exact_count_accuracy": sum(
            bool(row["evaluation"]["exact_count"])
            for row in completed.values()
        )
        / len(completed),
        "registered_accuracy": sum(
            bool(row["evaluation"].get("registered_success"))
            for row in completed.values()
        )
        / len(completed),
    }


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "gpus": [line.strip() for line in result.stdout.splitlines() if line.strip()],
    }


def resolve_model_revision(model_id: str, revision: str | None) -> str:
    from huggingface_hub import model_info

    info = model_info(model_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Unable to resolve immutable revision for {model_id}")
    return str(info.sha)


def _load_generated_text_decoder(
    model_spec: ModelSpec,
    *,
    revision: str,
    cache_dir: str | Path | None,
) -> Any | None:
    """Load a tokenizers.json decoder for models with broken auto decoding.

    The DeepSeek R1 Qwen3 tokenizer currently preserves ByteLevel display
    markers (notably U+0120/U+010A) when loaded through AutoTokenizer under
    the registered Transformers version. Its tokenizer.json decoder restores
    the intended spaces and newlines. Generated token IDs remain the
    authoritative output and the original vLLM text is retained separately.
    """

    if model_spec.family != "deepseek_r1_qwen3":
        return None
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    tokenizer_file = hf_hub_download(
        repo_id=model_spec.model_id,
        filename="tokenizer.json",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


def _decode_generated_text(
    *,
    model_spec: ModelSpec,
    engine_text: str,
    token_ids: Iterable[int],
    token_json_decoder: Any | None,
) -> tuple[str, str]:
    if model_spec.family != "deepseek_r1_qwen3":
        return engine_text, "vllm_output_text"
    if token_json_decoder is None:
        raise RuntimeError(
            "DeepSeek R1 Qwen3 requires its tokenizer.json output decoder"
        )
    decoded = token_json_decoder.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not decoded:
        raise RuntimeError("DeepSeek tokenizer.json decoded an empty response")
    return str(decoded), "tokenizer_json_from_output_token_ids"


def _sampling_params_kwargs(
    config: DecodingConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "n": 1,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "seed": seed,
        # Preserve Qwen <think>...</think> and Gemma 4 structured channel
        # delimiters so reasoning and final responses can be audited.
        "skip_special_tokens": False,
    }


def _batched(
    values: list[dict[str, Any]],
    size: int,
) -> Iterable[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("request_batch_size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_vllm_experiment(
    *,
    stimuli_path: str | Path,
    output_dir: str | Path,
    model: str,
    revision: str | None = None,
    passage_lengths: Iterable[int] | None = None,
    needle_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    prompt_modes: Iterable[str] | None = None,
    query_layout: str = QUERY_LAYOUT,
    engine_config: EngineConfig | None = None,
    cache_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    require_clean_git: bool = False,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_spec = resolve_model_spec(model)
    selected = select_stimuli(
        load_stimuli(stimuli_path),
        passage_lengths=passage_lengths,
        needle_counts=needle_counts,
        seeds=seeds,
    )
    if not selected:
        raise ValueError("Stimulus selection is empty")
    requests = build_requests(
        selected,
        model_spec=model_spec,
        prompt_modes=prompt_modes,
        query_layout=query_layout,
    )
    output = Path(output_dir)
    results_path = output / "requests.jsonl"
    manifest_path = output / "run_manifest.json"
    completed = _load_completed(results_path)
    request_ids = [str(request["request_id"]) for request in requests]
    request_id_set = set(request_ids)
    unexpected_completed = sorted(set(completed) - request_id_set)
    if unexpected_completed:
        raise RuntimeError(
            "Output directory contains results outside this request grid: "
            f"{unexpected_completed[:3]}"
        )
    pending = [
        request for request in requests if request["request_id"] not in completed
    ]

    provenance = _git_provenance(Path(repo_root).resolve())
    if require_clean_git and provenance["dirty"]:
        raise RuntimeError("Formal run requires a clean Git worktree")
    engine = engine_config or EngineConfig()
    if engine.request_batch_size <= 0:
        raise ValueError("request_batch_size must be positive")
    stimuli_file = Path(stimuli_path).resolve()
    stimuli_sha256 = hashlib.sha256(stimuli_file.read_bytes()).hexdigest()
    request_ids_sha256 = _ordered_id_digest(request_ids)
    existing_manifest = _load_json(manifest_path)
    if existing_manifest is None:
        immutable_revision = resolve_model_revision(model_spec.model_id, revision)
        created_at_utc = datetime.now(timezone.utc).isoformat()
        elapsed_before_seconds = 0.0
    else:
        expected_existing = {
            "schema_version": existing_manifest.get("schema_version"),
            "model_id": existing_manifest.get("model", {}).get("model_id"),
            "query_layout": existing_manifest.get("query_layout"),
            "stimuli_sha256": existing_manifest.get("stimuli_sha256"),
            "request_ids_sha256": existing_manifest.get("request_ids_sha256"),
            "engine": existing_manifest.get("engine"),
            "git_commit": existing_manifest.get("git", {}).get("commit"),
        }
        current = {
            "schema_version": "realistic_niah_run_manifest_v2",
            "model_id": model_spec.model_id,
            "query_layout": query_layout,
            "stimuli_sha256": stimuli_sha256,
            "request_ids_sha256": request_ids_sha256,
            "engine": asdict(engine),
            "git_commit": provenance["commit"],
        }
        if expected_existing != current:
            raise RuntimeError(
                "Refusing to mix incompatible results in one output directory: "
                f"existing={expected_existing}, current={current}"
            )
        immutable_revision = str(existing_manifest["model_revision"])
        if revision is not None:
            requested_revision = resolve_model_revision(
                model_spec.model_id,
                revision,
            )
            if requested_revision != immutable_revision:
                raise RuntimeError(
                    "Requested model revision does not match the existing run "
                    f"({requested_revision} != {immutable_revision})"
                )
        created_at_utc = str(existing_manifest["created_at_utc"])
        elapsed_before_seconds = float(
            existing_manifest.get("elapsed_generation_seconds", 0.0)
        )

    for identifier, row in completed.items():
        if row.get("model_id") != model_spec.model_id:
            raise RuntimeError(f"Model mismatch in completed row {identifier}")
        if row.get("model_revision") != immutable_revision:
            raise RuntimeError(
                f"Model revision mismatch in completed row {identifier}"
            )

    manifest = {
        "schema_version": "realistic_niah_run_manifest_v2",
        "protocol_version": "realistic_niah_v2",
        "created_at_utc": created_at_utc,
        "last_updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": asdict(model_spec),
        "model_revision": immutable_revision,
        "query_layout": query_layout,
        "prompt_modes": list(dict.fromkeys(
            str(request["prompt_mode"]) for request in requests
        )),
        "engine": asdict(engine),
        "stimuli_path": str(stimuli_file),
        "stimuli_sha256": stimuli_sha256,
        "selected_stimulus_ids_sha256": _ordered_id_digest(
            str(row["stimulus_id"]) for row in selected
        ),
        "request_ids_sha256": request_ids_sha256,
        "expected_requests": len(requests),
        "completed_requests": len(completed),
        "elapsed_generation_seconds": elapsed_before_seconds,
        "git": provenance,
        "hardware": _hardware_snapshot(),
        "packages": _package_versions(
            ("torch", "transformers", "vllm", "huggingface-hub")
        ),
    }
    if not pending:
        manifest |= {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **_evaluation_summary(completed),
        }
        _atomic_json(manifest_path, manifest)
        return manifest
    _atomic_json(manifest_path, manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.model_id,
        revision=immutable_revision,
        trust_remote_code=engine.trust_remote_code,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    generated_text_decoder = _load_generated_text_decoder(
        model_spec,
        revision=immutable_revision,
        cache_dir=cache_dir,
    )
    for request in pending:
        rendered = render_generation_prompt(
            tokenizer,
            request["messages"],
            model_spec=model_spec,
            prompt_mode=request["prompt_mode"],
        )
        request["rendered_prompt"] = rendered
        request["input_ids"] = tokenizer.encode(
            rendered,
            add_special_tokens=False,
        )
        request["model_passage_tokens"] = len(
            tokenizer.encode(
                request["stimulus"]["passage"],
                add_special_tokens=False,
            )
        )
        decode = decoding_config(model_spec, request["prompt_mode"])
        total_budget = len(request["input_ids"]) + decode.max_tokens
        if total_budget > engine.max_model_len:
            raise RuntimeError(
                f"{request['request_id']} needs {total_budget} tokens "
                f"but max_model_len={engine.max_model_len}"
            )

    llm_kwargs: dict[str, Any] = {
        "model": model_spec.model_id,
        "revision": immutable_revision,
        "tokenizer_revision": immutable_revision,
        "dtype": engine.dtype,
        "tensor_parallel_size": engine.tensor_parallel_size,
        "max_model_len": engine.max_model_len,
        "gpu_memory_utilization": engine.gpu_memory_utilization,
        "trust_remote_code": engine.trust_remote_code,
        "enable_prefix_caching": engine.enable_prefix_caching,
    }
    if engine.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = engine.max_num_seqs
    if cache_dir is not None:
        llm_kwargs["download_dir"] = str(cache_dir)
    llm = LLM(**llm_kwargs)

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for request in pending:
        groups[(request["prompt_mode"], request["seed"])].append(request)

    started = time.perf_counter()
    for (prompt_mode, generation_seed), group in sorted(groups.items()):
        decode = decoding_config(model_spec, prompt_mode)
        sampling = SamplingParams(
            **_sampling_params_kwargs(decode, seed=generation_seed)
        )
        for batch_index, batch in enumerate(
            _batched(group, engine.request_batch_size),
            start=1,
        ):
            batch_started = time.perf_counter()
            outputs = llm.generate(
                [item["rendered_prompt"] for item in batch],
                sampling,
                use_tqdm=True,
            )
            batch_elapsed = time.perf_counter() - batch_started
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM output count does not match request count")
            for request, generated in zip(batch, outputs):
                candidate = generated.outputs[0]
                stimulus = request["stimulus"]
                decoded_output_text, output_decode_strategy = (
                    _decode_generated_text(
                        model_spec=model_spec,
                        engine_text=candidate.text,
                        token_ids=candidate.token_ids,
                        token_json_decoder=generated_text_decoder,
                    )
                )
                expects_reasoning = reasoning_expected(
                    model_spec,
                    prompt_mode,
                )
                evaluation = evaluate_generation(
                    decoded_output_text,
                    prompt_mode=prompt_mode,
                    reasoning_expected=expects_reasoning,
                    gold_pairs=stimulus["gold_pairs"],
                    finish_reason=candidate.finish_reason,
                    output_tokens=len(candidate.token_ids),
                    max_output_tokens=decode.max_tokens,
                )
                completed[request["request_id"]] = {
                    "schema_version": "realistic_niah_request_v2",
                    "protocol_version": "realistic_niah_v2",
                    "request_id": request["request_id"],
                    "model_label": model_spec.label,
                    "model_id": model_spec.model_id,
                    "model_revision": immutable_revision,
                    "stimulus_id": stimulus["stimulus_id"],
                    "seed": stimulus["seed"],
                    "target_passage_tokens": stimulus["target_passage_tokens"],
                    "canonical_passage_tokens": stimulus[
                        "canonical_passage_tokens"
                    ],
                    "canonical_tokenizer": stimulus["canonical_tokenizer"],
                    "canonical_tokenizer_revision": stimulus.get(
                        "canonical_tokenizer_revision"
                    ),
                    "clean_filler_tokens": stimulus["clean_filler_tokens"],
                    "num_needles": stimulus["num_needles"],
                    "nominal_density_per_1k": stimulus["nominal_density_per_1k"],
                    "passage_sha256": stimulus["passage_sha256"],
                    "gold_count": stimulus["gold_count"],
                    "gold_pairs": stimulus["gold_pairs"],
                    "needles": stimulus["needles"],
                    "realized_insertions": stimulus["realized_insertions"],
                    "length_search": stimulus["length_search"],
                    "prompt_mode": prompt_mode,
                    "query_layout": request["query_layout"],
                    "reasoning_policy": model_spec.reasoning_policy,
                    "reasoning_expected": expects_reasoning,
                    "messages": request["messages"],
                    "rendered_prompt": request["rendered_prompt"],
                    "input_ids": request["input_ids"],
                    "model_passage_tokens": request["model_passage_tokens"],
                    "model_input_tokens": len(request["input_ids"]),
                    "model_density_per_1k": stimulus["num_needles"]
                    / (request["model_passage_tokens"] / 1000),
                    "decoding": asdict(decode) | {"seed": generation_seed},
                    "raw_output_text": decoded_output_text,
                    "vllm_output_text": (
                        candidate.text
                        if candidate.text != decoded_output_text
                        else None
                    ),
                    "output_decode_strategy": output_decode_strategy,
                    "output_token_ids": list(candidate.token_ids),
                    "output_tokens": len(candidate.token_ids),
                    "finish_reason": candidate.finish_reason,
                    "stop_reason": getattr(candidate, "stop_reason", None),
                    "batch_wall_time_seconds": batch_elapsed,
                    "batch_size": len(batch),
                    "batch_index_within_seed_mode": batch_index,
                    "seed_mode_group_size": len(group),
                    "evaluation": evaluation,
                }
            _atomic_jsonl(
                results_path,
                (completed[key] for key in sorted(completed)),
            )
            manifest |= {
                "last_updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_requests": len(completed),
                "elapsed_generation_seconds": elapsed_before_seconds
                + (time.perf_counter() - started),
            }
            _atomic_json(manifest_path, manifest)

    elapsed = time.perf_counter() - started
    manifest |= {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_requests": len(completed),
        "elapsed_generation_seconds": elapsed_before_seconds + elapsed,
        **_evaluation_summary(completed),
    }
    _atomic_json(manifest_path, manifest)
    return manifest
