#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.cross_mode_geometry import compare_position_geometry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare position-wise classification and covariance-aware cluster "
            "quality for V4.4 non-thinking and V5 native thinking"
        )
    )
    parser.add_argument("--non-thinking-capture-index", type=Path, required=True)
    parser.add_argument("--native-thinking-capture-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design-variant", default="v4.4")
    parser.add_argument("--non-thinking-pooling", default="span_end")
    parser.add_argument("--native-site-kind", default="item_end")
    parser.add_argument(
        "--native-cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    paths = compare_position_geometry(
        args.non_thinking_capture_index,
        args.native_thinking_capture_index,
        args.output,
        design_variant=args.design_variant,
        non_thinking_pooling=args.non_thinking_pooling,
        native_site_kind=args.native_site_kind,
        native_cohort=args.native_cohort,
        pca_dim=args.pca_dim,
        layers=args.layers,
        random_state=args.random_state,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
