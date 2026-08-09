from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

from realistic_niah.runner import EngineConfig, load_vllm_runtime
from realistic_niah.spec import QUERY_LAYOUT
from realistic_niah_v3_1.runner import run_v31_experiment
from realistic_niah_v3_1.sharding import _task_id
from realistic_niah_v3_1.spec import MODEL_REVISIONS, resolve_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a nonempty subset of one V3.1 model's prompt modes."
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt-modes", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--request-batch-size", type=int, required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Formal A100 execution requires a clean frozen worktree")

    spec = resolve_model_spec(args.model)
    if args.revision != MODEL_REVISIONS[spec.label]:
        raise ValueError(f"revision mismatch for {spec.label}")
    modes = tuple(mode for mode in args.prompt_modes.split(",") if mode)
    if not modes or len(modes) != len(set(modes)):
        raise ValueError("prompt modes must be nonempty and unique")
    unknown = set(modes) - set(spec.prompt_modes)
    if unknown:
        raise ValueError(f"unregistered prompt modes for {spec.label}: {sorted(unknown)}")

    engine = EngineConfig(
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        request_batch_size=args.request_batch_size,
    )
    runtime = load_vllm_runtime(
        model_spec=spec,
        revision=args.revision,
        engine_config=engine,
        cache_dir=args.cache_dir,
    )
    run_root = Path(args.run_root).resolve()
    manifests: dict[str, dict[str, object]] = {}
    for mode in modes:
        task_id = _task_id(spec.label, mode)
        manifests[mode] = run_v31_experiment(
            stimuli_path=args.stimuli,
            output_dir=run_root / "shards" / task_id / "main",
            model=spec.label,
            revision=args.revision,
            prompt_modes=(mode,),
            query_layout=QUERY_LAYOUT,
            engine_config=engine,
            cache_dir=args.cache_dir,
            repo_root=repo,
            require_clean_git=True,
            loaded_runtime=runtime,
        )
        gc.collect()
    print(
        json.dumps(
            {
                "model_label": spec.label,
                "model_revision": args.revision,
                "physical_model_loads": 1,
                "logical_shards": len(manifests),
                "completed_requests": sum(
                    int(manifest["completed_requests"])
                    for manifest in manifests.values()
                ),
                "prompt_modes": list(manifests),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
