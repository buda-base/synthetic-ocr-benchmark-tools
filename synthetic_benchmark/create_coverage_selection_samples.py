#!/usr/bin/env python3
"""Write reviewable text samples from the coverage-driven chunk selector."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from build_render_plan import load_chunks, make_per_font_plan_rows, normalized_script
from synthetic_common import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_CHUNKS_PARQUET,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    DEFAULT_SUPPORT_PARQUET,
    RARE_STACK_THRESHOLD,
    has_tibetan_letter,
    load_font_catalog,
    load_stack_rarity_scores,
    load_supported_stacks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PARQUET)
    parser.add_argument("--support-parquet", type=Path, default=DEFAULT_SUPPORT_PARQUET)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=DEFAULT_SCRIPTS_CSV)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--fonts-per-script", type=int, default=2)
    parser.add_argument("--samples-per-font", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supported = load_supported_stacks(args.support_parquet)
    rarity_by_stack = load_stack_rarity_scores()
    fonts = load_font_catalog(args.benchmark_csv, args.scripts_csv, args.fonts_csv)
    selected_fonts = []
    for script in ("uchen", "ume"):
        candidates = [
            font
            for font in fonts
            if normalized_script(font.script_type) == script
            and font.basename in supported
        ]
        candidates.sort(
            key=lambda font: (
                sum(
                    has_tibetan_letter(stack)
                    and rarity_by_stack.get(stack, 1.0) >= RARE_STACK_THRESHOLD
                    for stack in supported[font.basename]
                ),
                len(supported[font.basename]),
                font.basename,
            ),
            reverse=True,
        )
        selected_fonts.extend(candidates[: args.fonts_per_script])

    rows = make_per_font_plan_rows(
        fonts=selected_fonts,
        chunks=load_chunks(args.chunks),
        supported=supported,
        images_per_font=args.samples_per_font,
        seed=args.seed,
        max_chunk_reuse_ratio=0.10,
    )
    by_font: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_font[str(row["font_name"])].append(row)

    lines = [
        "Coverage-driven source text selection samples",
        "=" * 46,
        "",
        "Policy: 50% normal / 50% difficult; 90% repetition-penalized / "
        "10% repetition-rewarded.",
        "Difficult candidates use bounded greedy ranking: 70% new rare-stack "
        "coverage, 20% repetition objective, 10% page difficulty.",
        "",
    ]
    for font_name in sorted(by_font):
        font_rows = sorted(by_font[font_name], key=lambda row: int(row["image_id"]))
        script = str(font_rows[0]["script"])
        total_new = sum(
            int(row["source_text_new_rare_stack_count"]) for row in font_rows
        )
        lines.extend(
            [
                f"FONT: {font_name} ({script})",
                f"New rare stacks covered across samples: {total_new}",
                "-" * 72,
            ]
        )
        for index, row in enumerate(font_rows, start=1):
            rare_stacks = sorted(
                stack
                for stack in str(row.get("stacks") or "").split()
                if has_tibetan_letter(stack)
                and rarity_by_stack.get(stack, 1.0) >= RARE_STACK_THRESHOLD
            )
            lines.extend(
                [
                    (
                        f"[{index:02d}] difficulty={row['source_text_difficulty_tier']} "
                        f"rarity={float(row['stack_difficulty_score']):.3f} "
                        f"repetition={float(row['source_text_repetition_score']):.3f} "
                        f"repetition_policy={row['source_text_repetition_policy']} "
                        f"rare={row['source_text_rare_stack_count']} "
                        f"new_rare={row['source_text_new_rare_stack_count']} "
                        f"coverage_gain={float(row['source_text_rarity_coverage_gain']):.3f}"
                    ),
                    f"Rare stacks: {' '.join(rare_stacks) if rare_stacks else '(none)'}",
                    str(row["text"]).strip(),
                    "",
                ]
            )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(
        f"Wrote {len(rows)} samples across {len(by_font)} fonts to {args.output}"
    )


if __name__ == "__main__":
    main()
