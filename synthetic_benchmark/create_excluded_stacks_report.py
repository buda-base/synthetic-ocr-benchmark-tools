#!/usr/bin/env python3
"""Frequency-weighted report of corpus stacks absent from a render plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from build_render_plan import (
    text_repetition_score,
    zstd_compression_ratio,
)
from selection_coverage_report import _bucket, _letter_stacks
from synthetic_common import (
    DEFAULT_BOCORPUS_PARQUET,
    DEFAULT_CHUNKS_PARQUET,
    DEFAULT_STACKS_CSV,
    DEFAULT_SUPPORT_PARQUET,
    RARITY_IGNORED_MARKS,
    has_tibetan_letter,
    load_font_catalog,
    load_stack_rarity_scores,
    load_supported_stacks,
    parse_int,
    rarity_equivalent_stack,
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_SCRIPTS_CSV,
    DEFAULT_FONTS_CSV,
)
from build_render_plan import normalized_script


def is_coverage_counted_stack(stack: str) -> bool:
    """Exclude annotation-mark stacks from coverage accounting entirely."""
    return has_tibetan_letter(stack) and not any(
        mark in stack for mark in RARITY_IGNORED_MARKS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PARQUET)
    parser.add_argument("--bocorpus", type=Path, default=DEFAULT_BOCORPUS_PARQUET)
    parser.add_argument("--support-parquet", type=Path, default=DEFAULT_SUPPORT_PARQUET)
    parser.add_argument("--stacks-csv", type=Path, default=DEFAULT_STACKS_CSV)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=DEFAULT_SCRIPTS_CSV)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--max-text-repetition-score", type=float, default=0.50)
    parser.add_argument("--min-zstd-compression-ratio", type=float, default=0.18)
    return parser.parse_args()


def load_collection_by_row(path: Path) -> list[str]:
    table = pq.read_table(path, columns=["collection"])
    return [str(value or "unknown") for value in table.column(0).to_pylist()]


def load_occurrence_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stack = (row.get("stack") or "").strip()
            if not stack or not is_coverage_counted_stack(stack):
                continue
            counts[stack] = parse_int(
                row.get("nb_occurences") or row.get("nb_occurrences"),
                0,
            )
    return counts


def chunk_filter_reason(
    repetition: float,
    zstd_ratio: float,
    *,
    max_repetition: float,
    min_zstd: float,
) -> str | None:
    rep_fail = repetition >= max_repetition
    zstd_fail = zstd_ratio < min_zstd
    if not rep_fail and not zstd_fail:
        return None
    if rep_fail and zstd_fail:
        return "repetition_and_density"
    if rep_fail:
        return "repetition_only"
    return "density_only"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    collections = load_collection_by_row(args.bocorpus)
    occurrence_counts = load_occurrence_counts(args.stacks_csv)
    rarity_scores = load_stack_rarity_scores(str(args.stacks_csv))
    supported = load_supported_stacks(args.support_parquet)
    fonts = load_font_catalog(args.benchmark_csv, args.scripts_csv, args.fonts_csv)
    eligible_fonts = [
        font
        for font in fonts
        if font.basename in supported
        and font.script_category
        and normalized_script(font.script_type)
    ]
    supported_by_any = set()
    for font in eligible_fonts:
        supported_by_any.update(supported.get(font.basename, set()))

    plan_rows = pq.read_table(args.plan, columns=["text"]).to_pylist()
    selected_stacks: set[str] = set()
    for row in tqdm(plan_rows, desc="Selected stacks", unit="page"):
        selected_stacks.update(
            stack
            for stack in _letter_stacks(str(row.get("text") or ""))
            if is_coverage_counted_stack(stack)
        )

    # Per-stack evidence across all chunks.
    corpus_stacks: set[str] = set()
    eligible_stacks: set[str] = set()
    chunk_token_counts: Counter[str] = Counter()
    eligible_token_counts: Counter[str] = Counter()
    excluded_token_counts: Counter[str] = Counter()
    filter_reason_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    collection_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    excluded_collection_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    chunks_by_filter_reason: Counter[str] = Counter()

    chunk_table = pq.read_table(
        args.chunks,
        columns=["bocorpus_row", "text", "stacks"],
    )
    bocorpus_rows = chunk_table.column("bocorpus_row").to_pylist()
    texts = chunk_table.column("text").to_pylist()
    stacks_cols = chunk_table.column("stacks").to_pylist()

    for index in tqdm(range(len(texts)), desc="Score chunks", unit="chunk"):
        text = str(texts[index] or "")
        stacks = [
            stack
            for stack in str(stacks_cols[index] or "").split()
            if is_coverage_counted_stack(stack)
        ]
        if not stacks:
            stacks = [
                stack
                for stack in _letter_stacks(text)
                if is_coverage_counted_stack(stack)
            ]
        if not stacks:
            continue

        row_index = int(bocorpus_rows[index] or 0)
        collection = (
            collections[row_index]
            if 0 <= row_index < len(collections)
            else "unknown"
        )
        repetition = text_repetition_score(text)
        zstd_ratio = zstd_compression_ratio(text)
        reason = chunk_filter_reason(
            repetition,
            zstd_ratio,
            max_repetition=args.max_text_repetition_score,
            min_zstd=args.min_zstd_compression_ratio,
        )
        unique = set(stacks)
        corpus_stacks.update(unique)
        chunk_token_counts.update(stacks)
        for stack in stacks:
            collection_token_counts[stack][collection] += 1

        if reason is None:
            eligible_stacks.update(unique)
            eligible_token_counts.update(stacks)
            chunks_by_filter_reason["eligible"] += 1
        else:
            excluded_token_counts.update(stacks)
            chunks_by_filter_reason[reason] += 1
            for stack in stacks:
                filter_reason_token_counts[stack][reason] += 1
                excluded_collection_token_counts[stack][collection] += 1

    excluded_stacks = corpus_stacks - selected_stacks
    detail_rows: list[dict[str, object]] = []
    for stack in excluded_stacks:
        supported_count = sum(
            stack in supported.get(font.basename, set()) for font in eligible_fonts
        )
        in_eligible = stack in eligible_stacks
        if supported_count == 0:
            reason_group = "font_support"
            detail_reason = "unsupported_by_all_eligible_fonts"
        elif not in_eligible:
            reason_group = "chunk_exclusion"
            filter_counts = filter_reason_token_counts[stack]
            if not filter_counts:
                detail_reason = "chunk_exclusion_unknown"
            elif len(filter_counts) == 1:
                detail_reason = next(iter(filter_counts))
            else:
                # Present only outside the eligible pool; pick dominant filter.
                detail_reason = filter_counts.most_common(1)[0][0]
                if len(filter_counts) > 1:
                    detail_reason = "mixed_" + "+".join(
                        name for name, _count in filter_counts.most_common()
                    )
        else:
            reason_group = "selection"
            detail_reason = "text_policy_eligible_but_not_selected"

        collection_counts = collection_token_counts[stack]
        excluded_collections = excluded_collection_token_counts[stack]
        top_collections = collection_counts.most_common()
        top_excluded_collections = excluded_collections.most_common()
        occurrence = int(occurrence_counts.get(stack, 0))
        rarity = float(rarity_scores.get(stack, 1.0))
        detail_rows.append(
            {
                "stack": stack,
                "rarity_equivalent_stack": rarity_equivalent_stack(stack),
                "bocorpus_occurrences": occurrence,
                "rarity_score": round(rarity, 6),
                "rarity_bucket": _bucket(stack, rarity_scores),
                "reason_group": reason_group,
                "detail_reason": detail_reason,
                "supported_by_eligible_font_count": supported_count,
                "appears_in_text_policy_eligible_chunks": int(in_eligible),
                "chunk_token_occurrences_all": int(chunk_token_counts[stack]),
                "chunk_token_occurrences_eligible": int(eligible_token_counts[stack]),
                "chunk_token_occurrences_filtered_out": int(excluded_token_counts[stack]),
                "filter_repetition_only_tokens": int(
                    filter_reason_token_counts[stack]["repetition_only"]
                ),
                "filter_density_only_tokens": int(
                    filter_reason_token_counts[stack]["density_only"]
                ),
                "filter_repetition_and_density_tokens": int(
                    filter_reason_token_counts[stack]["repetition_and_density"]
                ),
                "collections": " | ".join(
                    f"{name}:{count}" for name, count in top_collections
                ),
                "top_collection": top_collections[0][0] if top_collections else "",
                "top_collection_chunk_tokens": (
                    top_collections[0][1] if top_collections else 0
                ),
                "excluded_chunk_collections": " | ".join(
                    f"{name}:{count}" for name, count in top_excluded_collections
                ),
                "top_excluded_chunk_collection": (
                    top_excluded_collections[0][0] if top_excluded_collections else ""
                ),
            }
        )

    detail_rows.sort(
        key=lambda row: (
            -int(row["bocorpus_occurrences"]),
            str(row["reason_group"]),
            str(row["detail_reason"]),
            str(row["stack"]),
        )
    )

    def weighted(rows: list[dict[str, object]]) -> int:
        return sum(int(row["bocorpus_occurrences"]) for row in rows)

    by_group = defaultdict(list)
    by_detail = defaultdict(list)
    for row in detail_rows:
        by_group[str(row["reason_group"])].append(row)
        by_detail[str(row["detail_reason"])].append(row)

    group_summary = []
    for group, rows in sorted(by_group.items()):
        group_summary.append(
            {
                "reason_group": group,
                "distinct_stacks": len(rows),
                "bocorpus_occurrence_mass": weighted(rows),
                "share_of_excluded_distinct": round(len(rows) / max(1, len(detail_rows)), 6),
                "share_of_excluded_occurrence_mass": round(
                    weighted(rows) / max(1, weighted(detail_rows)), 6
                ),
            }
        )

    detail_summary = []
    for detail, rows in sorted(by_detail.items(), key=lambda item: -weighted(item[1])):
        detail_summary.append(
            {
                "detail_reason": detail,
                "reason_group": rows[0]["reason_group"],
                "distinct_stacks": len(rows),
                "bocorpus_occurrence_mass": weighted(rows),
                "share_of_excluded_occurrence_mass": round(
                    weighted(rows) / max(1, weighted(detail_rows)), 6
                ),
                "example_stacks": " | ".join(
                    f"{row['stack']}({row['bocorpus_occurrences']})"
                    for row in rows[:8]
                ),
            }
        )

    # Collection x reason for chunk-exclusion stacks only.
    collection_reason_rows = []
    collection_reason_mass: Counter[tuple[str, str]] = Counter()
    collection_reason_stacks: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in by_group.get("chunk_exclusion", []):
        stack = str(row["stack"])
        detail = str(row["detail_reason"])
        occ = int(row["bocorpus_occurrences"])
        # Attribute frequency mass to collections proportional to excluded chunk tokens.
        excluded_counts = excluded_collection_token_counts[stack]
        total_tokens = sum(excluded_counts.values()) or 1
        for collection, token_count in excluded_counts.items():
            share = token_count / total_tokens
            collection_reason_mass[(collection, detail)] += int(round(occ * share))
            collection_reason_stacks[(collection, detail)].add(stack)
    for (collection, detail), mass in sorted(
        collection_reason_mass.items(), key=lambda item: (-item[1], item[0])
    ):
        collection_reason_rows.append(
            {
                "collection": collection,
                "detail_reason": detail,
                "distinct_stacks": len(collection_reason_stacks[(collection, detail)]),
                "attributed_bocorpus_occurrence_mass": mass,
            }
        )

    top_excluded_by_frequency = detail_rows[:100]

    detail_csv = args.output_dir / "excluded_stacks_detail.csv"
    group_csv = args.output_dir / "excluded_stacks_by_reason_group.csv"
    detail_reason_csv = args.output_dir / "excluded_stacks_by_detail_reason.csv"
    collection_csv = args.output_dir / "excluded_stacks_chunk_exclusion_by_collection.csv"
    top_csv = args.output_dir / "excluded_stacks_top_by_frequency.csv"
    write_csv(detail_csv, detail_rows)
    write_csv(group_csv, group_summary)
    write_csv(detail_reason_csv, detail_summary)
    if collection_reason_rows:
        write_csv(collection_csv, collection_reason_rows)
    write_csv(top_csv, top_excluded_by_frequency)

    corpus_occurrence_total = sum(
        occurrence_counts.get(stack, 0) for stack in corpus_stacks
    )
    selected_occurrence_mass = sum(
        occurrence_counts.get(stack, 0) for stack in selected_stacks & corpus_stacks
    )
    summary = {
        "planned_pages": len(plan_rows),
        "corpus_distinct_letter_stacks": len(corpus_stacks),
        "selected_distinct_letter_stacks": len(selected_stacks & corpus_stacks),
        "excluded_distinct_letter_stacks": len(detail_rows),
        "corpus_occurrence_mass": corpus_occurrence_total,
        "selected_occurrence_mass": selected_occurrence_mass,
        "excluded_occurrence_mass": weighted(detail_rows),
        "excluded_occurrence_mass_fraction": round(
            weighted(detail_rows) / max(1, corpus_occurrence_total), 6
        ),
        "chunks_by_filter_reason": dict(chunks_by_filter_reason),
        "text_policy": {
            "max_text_repetition_score": args.max_text_repetition_score,
            "min_zstd_compression_ratio": args.min_zstd_compression_ratio,
        },
        "reason_groups": group_summary,
        "detail_reasons": detail_summary,
        "files": {
            "detail": detail_csv.name,
            "by_reason_group": group_csv.name,
            "by_detail_reason": detail_reason_csv.name,
            "chunk_exclusion_by_collection": collection_csv.name,
            "top_by_frequency": top_csv.name,
        },
        "definitions": {
            "font_support": (
                "Stack never supported by any eligible benchmark font under the "
                "current stack_support.parquet policy."
            ),
            "chunk_exclusion": (
                "Stack appears in BoCorpus chunks, but only in chunks removed by the "
                "text policy (repetition and/or Zstd density)."
            ),
            "selection": (
                "Stack appears in text-policy-eligible chunks and is font-supported, "
                "but was not chosen by coverage/quota selection."
            ),
            "occurrence_mass": (
                "Sum of BoCorpus stack occurrence counts for the listed distinct "
                "stacks; used to weight exclusions by corpus frequency."
            ),
            "ignored_marks": (
                "Stacks containing U+0F35 or U+0F37 are excluded from corpus, "
                "selected, and excluded accounting entirely (not folded, not counted)."
            ),
        },
    }
    summary_path = args.output_dir / "excluded_stacks_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = args.output_dir / "README.md"
    readme.write_text(
        "# Excluded stack report\n\n"
        f"- Corpus distinct stacks: {len(corpus_stacks):,}\n"
        f"- Selected: {len(selected_stacks & corpus_stacks):,}\n"
        f"- Excluded: {len(detail_rows):,}\n"
        f"- Excluded occurrence mass: {weighted(detail_rows):,} "
        f"({weighted(detail_rows) / max(1, corpus_occurrence_total):.2%} of corpus mass)\n\n"
        "## Reason groups\n\n"
        + "\n".join(
            f"- **{row['reason_group']}**: {row['distinct_stacks']:,} stacks, "
            f"occurrence mass {row['bocorpus_occurrence_mass']:,} "
            f"({row['share_of_excluded_occurrence_mass']:.1%} of excluded mass)"
            for row in group_summary
        )
        + "\n\n"
        "## Detail reasons\n\n"
        + "\n".join(
            f"- `{row['detail_reason']}` ({row['reason_group']}): "
            f"{row['distinct_stacks']:,} stacks, mass {row['bocorpus_occurrence_mass']:,}"
            for row in detail_summary
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
