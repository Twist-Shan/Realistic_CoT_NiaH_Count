from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.pipeline import (
    run_model_stage,
    run_representation_analysis,
    write_runtime_provenance,
)
from realistic_niah_v4.spec import ANSWER_FORMATS, DESIGN_VARIANTS, MODEL_SPECS


MODEL_STAGES = (
    "preflight",
    "representation-capture",
    "attention",
    "ablation",
    "patching",
)


def _csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Qwen3-8B or Gemma4-E4B Realistic NIAH V4 "
            "non-thinking mechanistic stage."
        )
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            *MODEL_STAGES,
            "representation-analyze",
            "all",
        ),
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--config",
        default="configs/realistic_niah_v4.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--answer-format",
        default="numeric",
        choices=ANSWER_FORMATS,
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--variants")
    parser.add_argument("--seeds")
    parser.add_argument("--counts")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    variants = _csv_strings(args.variants)
    if variants is not None:
        unknown = sorted(set(variants) - set(DESIGN_VARIANTS))
        if unknown:
            parser.error(f"unknown design variants: {unknown}")
    provenance = write_runtime_provenance(
        output_dir=Path(args.output_dir) / args.model / args.answer_format,
        config_path=args.config,
        stimuli_path=args.stimuli,
        model_label=args.model,
        answer_format=args.answer_format,
        repo_root=args.repo_root,
    )
    common = {
        "stimuli_path": args.stimuli,
        "config_path": args.config,
        "output_dir": args.output_dir,
        "model_label": args.model,
        "answer_format": args.answer_format,
        "cache_dir": args.cache_dir,
        "device_map": args.device_map,
        "variants": variants,
        "seeds": _csv_ints(args.seeds),
        "counts": _csv_ints(args.counts),
        "overwrite": args.overwrite,
        "forward_smoke": args.forward_smoke,
    }
    outputs: dict[str, object] = {"provenance": str(provenance)}
    if args.stage == "representation-analyze":
        outputs[args.stage] = run_representation_analysis(
            config_path=args.config,
            output_dir=args.output_dir,
            model_label=args.model,
            answer_format=args.answer_format,
        )
    elif args.stage == "all":
        # Separate model loads keep every expensive stage restartable and make
        # failure provenance unambiguous. For cluster jobs, invoke each stage
        # separately instead of `all`.
        for stage in MODEL_STAGES[1:]:
            outputs[stage] = run_model_stage(stage=stage, **common)
            if stage == "representation-capture":
                outputs["representation-analyze"] = run_representation_analysis(
                    config_path=args.config,
                    output_dir=args.output_dir,
                    model_label=args.model,
                    answer_format=args.answer_format,
                )
    else:
        outputs[args.stage] = run_model_stage(stage=args.stage, **common)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
