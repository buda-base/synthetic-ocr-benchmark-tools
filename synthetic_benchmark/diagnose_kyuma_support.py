#!/usr/bin/env python3
"""Diagnose Kyuma font/chunk compatibility and likely support false negatives."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from build_render_plan import (
    assign_font_splits,
    filter_chunks_by_text_policy,
    font_name,
    font_supports_chunk,
    load_chunks,
    normalized_script,
)
from synthetic_common import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_CHUNKS_PARQUET,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    DEFAULT_SUPPORT_PARQUET,
    FontCatalogRow,
    load_font_catalog,
    load_supported_stacks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PARQUET)
    parser.add_argument("--support-parquet", type=Path, default=DEFAULT_SUPPORT_PARQUET)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=DEFAULT_SCRIPTS_CSV)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-text-repetition-score", type=float, default=0.50)
    parser.add_argument("--min-zstd-compression-ratio", type=float, default=0.18)
    parser.add_argument(
        "--sample-rejected-chunks",
        type=int,
        default=40,
        help="Per Kyuma font, sample this many filtered chunks rejected only by support.",
    )
    return parser.parse_args()


def eligible_fonts(
    fonts: list[FontCatalogRow],
    supported: dict[str, set[str]],
) -> list[FontCatalogRow]:
    return [
        font
        for font in fonts
        if font.basename in supported
        and font.script_category
        and normalized_script(font.script_type)
    ]


def missing_stacks_for_chunk(
    font: FontCatalogRow,
    chunk: dict[str, object],
    supported: dict[str, set[str]],
) -> list[str]:
    font_supported = supported.get(font.basename, set())
    stacks = sorted(str(stack) for stack in (chunk.get("_stack_set") or set()))
    return [stack for stack in stacks if stack not in font_supported]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fonts = load_font_catalog(args.benchmark_csv, args.scripts_csv, args.fonts_csv)
    supported = load_supported_stacks(args.support_parquet)
    eligible = eligible_fonts(fonts, supported)
    kyuma = [font for font in eligible if font.script_category.startswith("Kyuma")]
    rng = random.Random(args.seed)
    split_by = assign_font_splits(eligible, supported, rng)

    print(f"Eligible fonts: {len(eligible)}; Kyuma faces: {len(kyuma)}")
    for font in sorted(kyuma, key=lambda item: (split_by.get(font_name(item), ""), item.basename)):
        print(
            f"  {split_by.get(font_name(font), '?'):5} "
            f"{font.basename:55} support={len(supported.get(font.basename, set()))}"
        )

    print("Loading chunks...", flush=True)
    all_chunks = load_chunks(args.chunks)
    filtered = filter_chunks_by_text_policy(
        all_chunks,
        max_repetition_score=args.max_text_repetition_score,
        min_zstd_compression_ratio=args.min_zstd_compression_ratio,
    )
    print(f"Filtered chunks: {len(filtered)}/{len(all_chunks)}", flush=True)

    compat_rows: list[dict[str, object]] = []
    rejected_stack_counter: Counter[str] = Counter()
    rejected_examples: list[dict[str, object]] = []

    for font in kyuma:
        compatible = 0
        rejected = 0
        sampled = 0
        for chunk in filtered:
            if font_supports_chunk(font, chunk, supported):
                compatible += 1
                continue
            rejected += 1
            missing = missing_stacks_for_chunk(font, chunk, supported)
            if missing:
                rejected_stack_counter.update(missing)
            if sampled < args.sample_rejected_chunks and missing:
                rejected_examples.append(
                    {
                        "basename": font.basename,
                        "split": split_by.get(font_name(font), ""),
                        "chunk_id": chunk["chunk_id"],
                        "missing_stack_count": len(missing),
                        "missing_stacks": " ".join(missing[:24]),
                        "char_count": chunk.get("char_count"),
                        "unique_stack_count": chunk.get("unique_stack_count"),
                        "text_repetition_score": chunk.get("text_repetition_score"),
                        "zstd_compression_ratio": chunk.get("zstd_compression_ratio"),
                        "text_preview": str(chunk.get("text") or "")[:180].replace("\n", " "),
                    }
                )
                sampled += 1
        compat_rows.append(
            {
                "basename": font.basename,
                "font_name": font_name(font),
                "split": split_by.get(font_name(font), ""),
                "script_type": font.script_type,
                "supported_stack_count": len(supported.get(font.basename, set())),
                "compatible_filtered_chunks": compatible,
                "rejected_filtered_chunks": rejected,
                "compatible_fraction": round(compatible / max(1, len(filtered)), 8),
            }
        )
        print(
            f"compat {font.basename}: {compatible}/{len(filtered)} "
            f"({compatible / max(1, len(filtered)):.4%})",
            flush=True,
        )

    plan_stats: dict[str, object] = {}
    if args.plan and args.plan.exists():
        plan_rows = pq.read_table(
            args.plan,
            columns=[
                "script_8",
                "basename",
                "source_text_category_topup",
                "suggested_split",
            ],
        ).to_pylist()
        kyuma_plan = [row for row in plan_rows if str(row["script_8"]).startswith("Kyuma")]
        plan_stats = {
            "kyuma_plan_rows": len(kyuma_plan),
            "kyuma_unique_rows": sum(
                int(row.get("source_text_category_topup") or 0) == 0 for row in kyuma_plan
            ),
            "kyuma_topup_rows": sum(
                int(row.get("source_text_category_topup") or 0) for row in kyuma_plan
            ),
            "kyuma_unique_fonts": dict(
                Counter(
                    str(row["basename"])
                    for row in kyuma_plan
                    if int(row.get("source_text_category_topup") or 0) == 0
                )
            ),
            "kyuma_split_counts": dict(
                Counter(str(row["suggested_split"]) for row in kyuma_plan)
            ),
        }

    # Global stack scarcity: which unsupported stacks most often block Kyuma fonts.
    top_blocking = rejected_stack_counter.most_common(200)
    blocking_rows = [
        {
            "stack": stack,
            "times_blocking_kyuma_font_chunk_pair": count,
            "supported_by_kyuma_font_count": sum(
                stack in supported.get(font.basename, set()) for font in kyuma
            ),
            "supported_by_any_eligible_font_count": sum(
                stack in supported.get(font.basename, set()) for font in eligible
            ),
        }
        for stack, count in top_blocking
    ]

    compat_csv = args.output_dir / "kyuma_filtered_chunk_compatibility.csv"
    blocking_csv = args.output_dir / "kyuma_blocking_stacks.csv"
    examples_csv = args.output_dir / "kyuma_rejected_chunk_examples.csv"
    summary_json = args.output_dir / "kyuma_support_diagnosis.json"

    for path, rows in (
        (compat_csv, compat_rows),
        (blocking_csv, blocking_rows),
        (examples_csv, rejected_examples),
    ):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "filtered_chunks": len(filtered),
        "all_chunks": len(all_chunks),
        "kyuma_faces": len(kyuma),
        "kyuma_split_assignment": dict(
            Counter(split_by.get(font_name(font), "") for font in kyuma)
        ),
        "median_compatible_filtered_chunks": sorted(
            int(row["compatible_filtered_chunks"]) for row in compat_rows
        )[len(compat_rows) // 2],
        "fonts_with_under_100_compatible_chunks": sum(
            int(row["compatible_filtered_chunks"]) < 100 for row in compat_rows
        ),
        "fonts_with_under_1000_compatible_chunks": sum(
            int(row["compatible_filtered_chunks"]) < 1000 for row in compat_rows
        ),
        "top_blocking_stacks": blocking_rows[:30],
        "plan_stats": plan_stats,
        "files": {
            "compatibility": compat_csv.name,
            "blocking_stacks": blocking_csv.name,
            "rejected_examples": examples_csv.name,
        },
        "interpretation": (
            "If many Kyuma fonts have thousands of compatible filtered chunks but the "
            "plan still used almost only top-ups, the bottleneck is chunk ownership / "
            "planner order, not support. If compatible counts are tiny, inspect "
            "kyuma_blocking_stacks.csv for likely support false negatives and re-probe "
            "those stacks with a looser or font-specific support check."
        ),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
