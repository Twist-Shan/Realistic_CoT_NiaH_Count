from __future__ import annotations

import argparse
import json

from realistic_niah.stimuli import FreezeSpec, freeze_grid


def _csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the registered Realistic NIAH master dataset."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canonical-tokenizer-revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--passage-lengths")
    parser.add_argument("--needle-counts")
    parser.add_argument("--seeds")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    defaults = FreezeSpec()
    paths = freeze_grid(
        output_dir=args.output_dir,
        spec=FreezeSpec(
            passage_lengths=_csv_ints(args.passage_lengths)
            or defaults.passage_lengths,
            needle_counts=_csv_ints(args.needle_counts)
            or defaults.needle_counts,
            seeds=_csv_ints(args.seeds) or defaults.seeds,
            canonical_tokenizer_revision=args.canonical_tokenizer_revision,
            tokenizer_cache_dir=args.cache_dir,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
