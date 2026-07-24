"""Report per-font stack coverage and representation in a render plan."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from synthetic_common import (
    FontCatalogRow,
    has_tibetan_letter,
    load_stack_rarity_scores,
    tokenize_tibetan_stacks,
)

RARITY_BUCKETS = (
    ("common", 0.0, 0.5),
    ("uncommon", 0.5, 0.8),
    ("rare", 0.8, 0.95),
    ("very_rare", 0.95, 1.000001),
)


def _font_key(font: FontCatalogRow) -> tuple[str, str]:
    return str(font.font_abs_path), str(font.ttc_face_index or "")


def _row_font_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["font_abs_path"]), str(row.get("ttc_face_index") or "")


def _letter_stacks(text: str) -> list[str]:
    return [
        stack
        for stack in tokenize_tibetan_stacks(text)
        if has_tibetan_letter(stack)
    ]


def _bucket(stack: str, rarity_by_stack: dict[str, float]) -> str:
    rarity = float(rarity_by_stack.get(stack, 1.0))
    for name, lower, upper in RARITY_BUCKETS:
        if lower <= rarity < upper:
            return name
    return "very_rare"


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = round((len(ordered) - 1) * fraction)
    return float(ordered[position])


def write_selection_coverage_report(
    *,
    rows: list[dict[str, object]],
    chunks: list[dict[str, object]],
    fonts: list[FontCatalogRow],
    supported: dict[str, set[str]],
    output_plan: Path,
) -> tuple[Path, Path, Path]:
    """Write per-font totals, rarity buckets, and report definitions."""
    rarity_by_stack = load_stack_rarity_scores()
    corpus_stacks: set[str] = set()
    for chunk in chunks:
        stacks = chunk.get("_stack_set")
        if stacks is None:
            stacks = _letter_stacks(str(chunk.get("text") or ""))
        corpus_stacks.update(stack for stack in stacks if has_tibetan_letter(stack))

    selected_by_font: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        selected_by_font[_row_font_key(row)].update(
            _letter_stacks(str(row.get("text") or ""))
        )

    coverage_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    globally_selected: set[str] = set()
    for font in sorted(fonts, key=_font_key):
        selected_counts = selected_by_font[_font_key(font)]
        selected_stacks = set(selected_counts)
        globally_selected.update(selected_stacks)
        renderable_stacks = corpus_stacks & supported.get(font.basename, set())
        selected_renderable = selected_stacks & renderable_stacks
        occurrence_counts = list(selected_counts.values())
        coverage_rows.append(
            {
                "basename": font.basename,
                "font_name": font.ps_name or font.basename,
                "font_file": font.font_file,
                "font_abs_path": str(font.font_abs_path),
                "ttc_face_index": font.ttc_face_index or "",
                "script_type": font.script_type,
                "script_category": font.script_category,
                "selected_pages": sum(
                    _row_font_key(row) == _font_key(font) for row in rows
                ),
                "corpus_distinct_stacks": len(corpus_stacks),
                "renderable_corpus_distinct_stacks": len(renderable_stacks),
                "selected_distinct_stacks": len(selected_stacks),
                "selected_renderable_distinct_stacks": len(selected_renderable),
                "renderable_coverage_fraction": round(
                    len(selected_renderable) / len(renderable_stacks), 8
                )
                if renderable_stacks
                else 0.0,
                "selected_stack_token_occurrences": sum(occurrence_counts),
                "selected_once_distinct_stacks": sum(
                    count == 1 for count in occurrence_counts
                ),
                "selected_median_occurrences_per_stack": _percentile(
                    occurrence_counts, 0.5
                ),
                "selected_p90_occurrences_per_stack": _percentile(
                    occurrence_counts, 0.9
                ),
            }
        )
        for bucket_name, _lower, _upper in RARITY_BUCKETS:
            corpus_bucket = {
                stack
                for stack in corpus_stacks
                if _bucket(stack, rarity_by_stack) == bucket_name
            }
            renderable_bucket = renderable_stacks & corpus_bucket
            selected_bucket = selected_stacks & corpus_bucket
            selected_bucket_occurrences = sum(
                selected_counts[stack] for stack in selected_bucket
            )
            bucket_rows.append(
                {
                    "basename": font.basename,
                    "font_name": font.ps_name or font.basename,
                    "font_abs_path": str(font.font_abs_path),
                    "ttc_face_index": font.ttc_face_index or "",
                    "script_type": font.script_type,
                    "rarity_bucket": bucket_name,
                    "corpus_distinct_stacks": len(corpus_bucket),
                    "renderable_distinct_stacks": len(renderable_bucket),
                    "selected_distinct_stacks": len(selected_bucket),
                    "selected_token_occurrences": selected_bucket_occurrences,
                    "renderable_coverage_fraction": round(
                        len(selected_bucket) / len(renderable_bucket), 8
                    )
                    if renderable_bucket
                    else 0.0,
                    "mean_occurrences_per_selected_stack": round(
                        selected_bucket_occurrences / len(selected_bucket), 6
                    )
                    if selected_bucket
                    else 0.0,
                }
            )

    coverage_path = output_plan.with_suffix(".stack_coverage_by_font.csv")
    buckets_path = output_plan.with_suffix(".stack_rarity_by_font.csv")
    summary_path = output_plan.with_suffix(".stack_coverage_summary.json")
    for path, report_rows in (
        (coverage_path, coverage_rows),
        (buckets_path, bucket_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=list(report_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(report_rows)

    summary = {
        "selectable_corpus_distinct_letter_stacks": len(corpus_stacks),
        "selection_distinct_letter_stacks_across_all_fonts": len(globally_selected),
        "font_faces": len(fonts),
        "selected_pages": len(rows),
        "rarity_score": (
            "percentile-clipped self-information in [0, 1]; absent corpus stacks "
            "are assigned 1"
        ),
        "rarity_equivalence": (
            "U+0F35 and U+0F37 annotation marks are removed before corpus "
            "frequency and rare-stack coverage are calculated"
        ),
        "rarity_buckets": [
            {"name": name, "minimum_inclusive": lower, "maximum_exclusive": upper}
            for name, lower, upper in RARITY_BUCKETS
        ],
        "coverage_denominator": (
            "distinct Tibetan letter stacks present in selectable chunks and "
            "supported by the font"
        ),
        "files": {
            "per_font": coverage_path.name,
            "per_font_rarity": buckets_path.name,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return coverage_path, buckets_path, summary_path
