#!/usr/bin/env python3
"""Export unique shorthand stacks for font coverage probing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SYNTH_DIR = REPO_ROOT / "synthetic_benchmark"
if str(SYNTH_DIR) not in sys.path:
    sys.path.insert(0, str(SYNTH_DIR))

from coverage_common import DEFAULT_SHORTHAND_STACKS_CSV
from shorthand_aug import DEFAULT_SHORTHANDS_CSV, load_shorthand_pairs, write_shorthand_stacks_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export shorthand stacks for coverage probes.")
    parser.add_argument("--shorthands", type=Path, default=DEFAULT_SHORTHANDS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_SHORTHAND_STACKS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.shorthands.is_file():
        raise SystemExit(
            f"Missing shorthand lexicon: {args.shorthands}\n"
            "Build it with: python synthetic_benchmark/build_shorthand_lexicon.py"
        )
    pairs = load_shorthand_pairs(args.shorthands)
    count = write_shorthand_stacks_csv(pairs, args.output)
    print(f"Wrote {args.output} ({count} unique shorthand stack(s) from {len(pairs)} pair(s))")


if __name__ == "__main__":
    main()
