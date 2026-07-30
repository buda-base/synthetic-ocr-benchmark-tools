"""Report per-font stack coverage and representation in a render plan."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from synthetic_common import (
    DEFAULT_STACKS_CSV,
    FontCatalogRow,
    has_tibetan_letter,
    load_stack_rarity_scores,
    parse_int,
    rarity_equivalent_stack,
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


def _stack_occurrence_counts(
    path: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    raw_counts: dict[str, int] = {}
    equivalent_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            stack = (row.get("stack") or "").strip()
            if not stack or not has_tibetan_letter(stack):
                continue
            count = parse_int(
                row.get("nb_occurences") or row.get("nb_occurrences"),
                0,
            )
            raw_counts[stack] = count
            equivalent_counts[rarity_equivalent_stack(stack)] += count
    return raw_counts, dict(equivalent_counts)


def write_selection_coverage_report(
    *,
    rows: list[dict[str, object]],
    chunks: list[dict[str, object]],
    fonts: list[FontCatalogRow],
    supported: dict[str, set[str]],
    output_plan: Path,
    stacks_csv: Path = DEFAULT_STACKS_CSV,
    selection_candidate_chunks: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Write per-font totals, rarity buckets, and report definitions."""
    rarity_by_stack = load_stack_rarity_scores()
    corpus_stacks: set[str] = set()
    for chunk in chunks:
        stacks = chunk.get("_stack_set")
        if stacks is None:
            stacks = _letter_stacks(str(chunk.get("text") or ""))
        corpus_stacks.update(stack for stack in stacks if has_tibetan_letter(stack))
    candidate_stacks: set[str] = set()
    for chunk in selection_candidate_chunks or chunks:
        stacks = chunk.get("_stack_set")
        if stacks is None:
            stacks = _letter_stacks(str(chunk.get("text") or ""))
        candidate_stacks.update(stack for stack in stacks if has_tibetan_letter(stack))

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
    excluded_path = output_plan.with_suffix(".excluded_corpus_stacks.csv")
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

    supported_by_any_font: set[str] = set()
    for font in fonts:
        supported_by_any_font.update(supported.get(font.basename, set()))
    selected_from_corpus = globally_selected & corpus_stacks
    excluded_stacks = corpus_stacks - selected_from_corpus
    raw_counts, equivalent_counts = _stack_occurrence_counts(stacks_csv)
    excluded_rows = []
    for stack in sorted(
        excluded_stacks,
        key=lambda item: (
            item not in supported_by_any_font,
            -float(rarity_by_stack.get(item, 1.0)),
            raw_counts.get(item, 0),
            item,
        ),
    ):
        equivalent = rarity_equivalent_stack(stack)
        excluded_rows.append(
            {
                "stack": stack,
                "rarity_equivalent_stack": equivalent,
                "bocorpus_occurrences": raw_counts.get(stack, 0),
                "rarity_equivalent_bocorpus_occurrences": equivalent_counts.get(
                    equivalent, 0
                ),
                "rarity_score": round(float(rarity_by_stack.get(stack, 1.0)), 6),
                "rarity_bucket": _bucket(stack, rarity_by_stack),
                "supported_by_eligible_font_count": sum(
                    stack in supported.get(font.basename, set()) for font in fonts
                ),
                "exclusion_reason": (
                    "unsupported_by_all_eligible_fonts"
                    if stack not in supported_by_any_font
                    else (
                        "excluded_by_text_policy"
                        if stack not in candidate_stacks
                        else "text_policy_eligible_but_not_selected"
                    )
                ),
            }
        )
    with excluded_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(excluded_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(excluded_rows)

    excluded_by_reason = Counter(
        str(row["exclusion_reason"]) for row in excluded_rows
    )
    excluded_by_rarity = Counter(str(row["rarity_bucket"]) for row in excluded_rows)
    summary = {
        "selectable_corpus_distinct_letter_stacks": len(corpus_stacks),
        "selection_distinct_letter_stacks_across_all_fonts": len(globally_selected),
        "selected_corpus_distinct_letter_stacks": len(selected_from_corpus),
        "excluded_corpus_distinct_letter_stacks": len(excluded_stacks),
        "excluded_renderable_distinct_letter_stacks": excluded_by_reason[
            "excluded_by_text_policy"
        ]
        + excluded_by_reason[
            "text_policy_eligible_but_not_selected"
        ],
        "excluded_unrenderable_distinct_letter_stacks": excluded_by_reason[
            "unsupported_by_all_eligible_fonts"
        ],
        "excluded_by_text_policy_distinct_letter_stacks": excluded_by_reason[
            "excluded_by_text_policy"
        ],
        "text_policy_eligible_but_not_selected_distinct_letter_stacks": (
            excluded_by_reason["text_policy_eligible_but_not_selected"]
        ),
        "excluded_distinct_stacks_by_rarity_bucket": dict(
            sorted(excluded_by_rarity.items())
        ),
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
            "excluded_corpus_stacks": excluded_path.name,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return coverage_path, buckets_path, excluded_path, summary_path
