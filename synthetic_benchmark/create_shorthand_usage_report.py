#!/usr/bin/env python3
"""Report exact shorthand replacements selected for a synthetic render plan."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from shorthand_aug import (
    BOUNDARY_CHARS,
    DEFAULT_DENYLIST_CSV,
    DEFAULT_SHORTHANDS_CSV,
    ShorthandPair,
    _select_non_overlapping,
    load_denylist,
    load_shorthand_pairs,
    mode_for_script,
    pair_is_allowed,
    syllable_count,
)
from synthetic_common import (
    DEFAULT_STACKS_CSV,
    DEFAULT_SUPPORT_PARQUET,
    has_tibetan_letter,
    load_stack_rarity_scores,
    load_supported_stacks,
    parse_int,
    rarity_equivalent_stack,
)


EXTREMELY_RARE_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shorthands", type=Path, default=DEFAULT_SHORTHANDS_CSV)
    parser.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST_CSV)
    parser.add_argument("--support-parquet", type=Path, default=DEFAULT_SUPPORT_PARQUET)
    parser.add_argument("--stacks-csv", type=Path, default=DEFAULT_STACKS_CSV)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-per-100-syllables", type=float, default=4.0)
    return parser.parse_args()


def build_long_form_trie(pairs: list[ShorthandPair]) -> dict[str, object]:
    root: dict[str, object] = {}
    for long_form in dict.fromkeys(pair.long_form for pair in pairs):
        node = root
        for char in long_form:
            node = node.setdefault(char, {})  # type: ignore[assignment]
        node.setdefault("", []).append(long_form)
    return root


def long_form_occurrences(
    text: str,
    trie: dict[str, object],
) -> dict[str, list[int]]:
    occurrences: dict[str, list[int]] = defaultdict(list)
    for start in range(len(text)):
        node = trie
        cursor = start
        while cursor < len(text) and text[cursor] in node:
            node = node[text[cursor]]  # type: ignore[assignment]
            cursor += 1
            for long_form in node.get("", []):  # type: ignore[union-attr]
                left_ok = start == 0 or text[start - 1] in BOUNDARY_CHARS
                right_ok = cursor >= len(text) or text[cursor] in BOUNDARY_CHARS
                if left_ok and right_ok:
                    occurrences[str(long_form)].append(start)
    return occurrences


def allowed_by_font(
    pairs: list[ShorthandPair],
    *,
    supported_by_font: dict[str, set[str]],
    denylist,
) -> dict[str, dict[str, list[ShorthandPair]]]:
    result: dict[str, dict[str, list[ShorthandPair]]] = {}
    for basename, supported in supported_by_font.items():
        by_long: dict[str, list[ShorthandPair]] = {}
        for pair in pairs:
            if pair_is_allowed(
                pair,
                supported_stacks=supported,
                denylist=denylist,
                basename=basename,
            ):
                by_long.setdefault(pair.long_form, []).append(pair)
        result[basename] = by_long
    return result


def selected_pairs_for_row(
    row: dict[str, object],
    *,
    by_long: dict[str, list[ShorthandPair]],
    trie: dict[str, object],
    seed: int,
    max_per_100_syllables: float,
) -> list[ShorthandPair]:
    image_id = int(row["image_id"])
    mode = str(row.get("shorthand_mode") or "").strip().lower()
    if mode not in {"sparse", "dense", "none"}:
        mode = mode_for_script(str(row.get("script") or ""), image_id)
    if mode == "none" or not by_long:
        return []

    text = str(row.get("text") or "")
    rng = random.Random(seed + image_id)
    occurrences = long_form_occurrences(text, trie)
    candidates: list[tuple[int, int, ShorthandPair]] = []
    for long_form, options in by_long.items():
        for start in occurrences.get(long_form, []):
            pair = rng.choice(options)
            candidates.append((start, start + len(long_form), pair))
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    if not candidates:
        return []

    if mode == "dense":
        selected = _select_non_overlapping(
            candidates,
            max_replacements=None,
            rng=None,
            dense=True,
        )
    else:
        maximum = max(
            0,
            int(syllable_count(text) * max_per_100_syllables / 100.0),
        )
        if maximum == 0:
            return []
        target = rng.randint(0, maximum)
        if target == 0:
            return []
        selected = _select_non_overlapping(
            candidates,
            max_replacements=target,
            rng=rng,
            dense=False,
        )
    return [pair for _start, _end, pair in selected]


def load_equivalent_stack_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stack = (row.get("stack") or "").strip()
            if not stack or not has_tibetan_letter(stack):
                continue
            count = parse_int(
                row.get("nb_occurences") or row.get("nb_occurrences"),
                0,
            )
            if count > 0:
                counts[rarity_equivalent_stack(stack)] += count
    return dict(counts)


def shorthand_metadata(
    pairs: list[ShorthandPair],
    *,
    counts: dict[str, int],
    rarity_scores: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[ShorthandPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.shorthand].append(pair)

    summaries: list[dict[str, object]] = []
    stack_rows: list[dict[str, object]] = []
    for shorthand, shorthand_pairs in grouped.items():
        stacks = list(
            dict.fromkeys(
                rarity_equivalent_stack(stack)
                for pair in shorthand_pairs
                for stack in pair.shorthand_stacks
                if has_tibetan_letter(stack)
            )
        )
        stack_data = [
            (
                stack,
                counts.get(stack, 0),
                float(rarity_scores.get(stack, 1.0)),
            )
            for stack in stacks
        ]
        rarest_stack, rarest_count, maximum_rarity = (
            max(stack_data, key=lambda item: (item[2], -item[1], item[0]))
            if stack_data
            else ("", 0, 1.0)
        )
        extremely_rare = [
            (stack, count, rarity)
            for stack, count, rarity in stack_data
            if rarity >= EXTREMELY_RARE_THRESHOLD
        ]
        summaries.append(
            {
                "shorthand": shorthand,
                "long_forms": " | ".join(
                    dict.fromkeys(pair.long_form for pair in shorthand_pairs)
                ),
                "sources": " | ".join(
                    dict.fromkeys(pair.source for pair in shorthand_pairs)
                ),
                "distinct_stacks": len(stacks),
                "stacks": " | ".join(stacks),
                "rarest_stack": rarest_stack,
                "rarest_stack_bocorpus_occurrences": rarest_count,
                "maximum_stack_rarity_score": round(maximum_rarity, 6),
                "extremely_rare_stacks": " | ".join(
                    stack for stack, _count, _rarity in extremely_rare
                ),
                "extremely_rare_stack_counts": " | ".join(
                    f"{stack}:{count}" for stack, count, _rarity in extremely_rare
                ),
                "complexity": "complex" if extremely_rare else "simple",
            }
        )
        for stack, count, rarity in stack_data:
            stack_rows.append(
                {
                    "shorthand": shorthand,
                    "stack": stack,
                    "bocorpus_occurrences": count,
                    "rarity_score": round(rarity, 6),
                    "extremely_rare": int(
                        rarity >= EXTREMELY_RARE_THRESHOLD
                    ),
                }
            )
    return summaries, stack_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty report: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs = load_shorthand_pairs(args.shorthands)
    denylist = load_denylist(args.denylist)
    supported = load_supported_stacks(args.support_parquet)
    by_font = allowed_by_font(
        pairs,
        supported_by_font=supported,
        denylist=denylist,
    )
    trie = build_long_form_trie(pairs)

    usage: Counter[str] = Counter()
    pages_with_shorthands = 0
    rows = pq.read_table(args.render_plan).to_pylist()
    for row in rows:
        selected = selected_pairs_for_row(
            row,
            by_long=by_font.get(str(row["basename"]), {}),
            trie=trie,
            seed=args.seed,
            max_per_100_syllables=args.max_per_100_syllables,
        )
        if selected:
            pages_with_shorthands += 1
        usage.update(pair.shorthand for pair in selected)

    counts = load_equivalent_stack_counts(args.stacks_csv)
    rarity_scores = load_stack_rarity_scores(str(args.stacks_csv))
    summaries, stack_rows = shorthand_metadata(
        pairs,
        counts=counts,
        rarity_scores=rarity_scores,
    )
    for summary in summaries:
        summary["benchmark_occurrences"] = usage[str(summary["shorthand"])]
    summaries.sort(
        key=lambda row: (
            -int(row["benchmark_occurrences"]),
            str(row["shorthand"]),
        )
    )
    stack_rows.sort(key=lambda row: (str(row["shorthand"]), -float(row["rarity_score"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "shorthand_occurrences.csv"
    stacks_csv = args.output_dir / "shorthand_stack_rarity.csv"
    write_csv(summary_csv, summaries)
    write_csv(stacks_csv, stack_rows)

    used = [row for row in summaries if int(row["benchmark_occurrences"]) > 0]
    complex_defined = sum(row["complexity"] == "complex" for row in summaries)
    simple_defined = len(summaries) - complex_defined
    complex_used = sum(row["complexity"] == "complex" for row in used)
    simple_used = len(used) - complex_used
    complex_occurrences = sum(
        int(row["benchmark_occurrences"])
        for row in summaries
        if row["complexity"] == "complex"
    )
    simple_occurrences = sum(
        int(row["benchmark_occurrences"])
        for row in summaries
        if row["complexity"] == "simple"
    )
    report = {
        "planned_pages": len(rows),
        "pages_with_at_least_one_shorthand": pages_with_shorthands,
        "total_shorthand_occurrences": sum(usage.values()),
        "distinct_defined_shorthands": len(summaries),
        "distinct_used_shorthands": len(used),
        "distinct_complex_shorthands_defined": complex_defined,
        "distinct_simple_shorthands_defined": simple_defined,
        "distinct_complex_shorthands_used": complex_used,
        "distinct_simple_shorthands_used": simple_used,
        "complex_shorthand_occurrences": complex_occurrences,
        "simple_shorthand_occurrences": simple_occurrences,
        "complex_shorthand_occurrence_fraction": round(
            complex_occurrences / max(1, sum(usage.values())),
            6,
        ),
        "complex_definition": (
            f"At least one rarity-equivalent Tibetan letter stack has normalized "
            f"BoCorpus rarity score >= {EXTREMELY_RARE_THRESHOLD}."
        ),
        "rarity_equivalence": (
            "U+0F35 and U+0F37 annotation marks are removed and their BoCorpus "
            "counts merged with the unmarked stack."
        ),
        "files": {
            "per_shorthand": summary_csv.name,
            "per_stack": stacks_csv.name,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# Synthetic benchmark shorthand usage\n\n"
        f"- Planned source rows: {len(rows):,}\n"
        f"- Rows with at least one shorthand: {pages_with_shorthands:,}\n"
        f"- Total shorthand replacements: {sum(usage.values()):,}\n"
        f"- Distinct shorthands used: {len(used):,} / {len(summaries):,} defined\n"
        f"- Complex shorthand replacements: {complex_occurrences:,} "
        f"({complex_occurrences / max(1, sum(usage.values())):.1%})\n"
        f"- Simple shorthand replacements: {simple_occurrences:,}\n"
        f"- Distinct complex shorthands used: {complex_used:,} / "
        f"{complex_defined:,} defined\n"
        f"- Distinct simple shorthands used: {simple_used:,} / "
        f"{simple_defined:,} defined\n\n"
        "A **complex** shorthand contains at least one rarity-equivalent Tibetan "
        f"letter stack with normalized BoCorpus rarity score >= "
        f"{EXTREMELY_RARE_THRESHOLD}. The per-shorthand CSV records the raw "
        "BoCorpus occurrence count of the rarest stack; the per-stack CSV gives "
        "the complete preliminary stack-level breakdown.\n\n"
        "Selections are reconstructed from the production render plan with the "
        "same font support, denylist, shorthand mode, seed, and density cap used "
        "during rendering.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
