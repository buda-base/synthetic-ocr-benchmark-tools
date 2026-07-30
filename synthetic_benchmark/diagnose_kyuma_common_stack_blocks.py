#!/usr/bin/env python3
"""Find high-frequency corpus stacks that Kyuma fonts fail, suggesting false negatives."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from build_render_plan import normalized_script
from synthetic_common import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    DEFAULT_STACKS_CSV,
    DEFAULT_SUPPORT_PARQUET,
    has_tibetan_letter,
    load_font_catalog,
    parse_int,
)


def main() -> None:
    fonts = load_font_catalog(
        DEFAULT_BENCHMARK_CSV, DEFAULT_SCRIPTS_CSV, DEFAULT_FONTS_CSV
    )
    kyuma_fonts = [
        font
        for font in fonts
        if font.script_category.startswith("Kyuma")
        and normalized_script(font.script_type)
    ]
    kyuma = {font.basename for font in kyuma_fonts}

    occurrences: dict[str, int] = {}
    with DEFAULT_STACKS_CSV.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stack = (row.get("stack") or "").strip()
            if not stack or not has_tibetan_letter(stack):
                continue
            occurrences[stack] = parse_int(
                row.get("nb_occurences") or row.get("nb_occurrences"), 0
            )

    # Focus on broad Kyuma fonts first (support > 5000 under current strict policy).
    status: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    columns = [
        "test_kind",
        "basename",
        "stack",
        "ok",
        "support_class",
        "reason",
        "notdef_count",
    ]
    pf = pq.ParquetFile(DEFAULT_SUPPORT_PARQUET)
    for batch in pf.iter_batches(batch_size=50_000, columns=columns):
        for row in batch.to_pylist():
            if row["test_kind"] != "stack" or row["basename"] not in kyuma:
                continue
            status[row["basename"]][row["stack"]] = {
                "ok": bool(row["ok"]),
                "support_class": row.get("support_class"),
                "reason": row.get("reason"),
                "notdef_count": int(row.get("notdef_count") or 0),
            }

    broad = [
        font.basename
        for font in kyuma_fonts
        if sum(1 for item in status[font.basename].values() if item["ok"]) >= 5000
    ]
    print(f"Broad Kyuma fonts (>=5000 ok stacks): {len(broad)}")
    for basename in broad:
        print(" ", basename)

    # Stacks that fail on at least half of broad fonts, ranked by corpus frequency.
    candidates: list[dict[str, object]] = []
    all_stacks = set().union(*(status[basename] for basename in broad))
    for stack in all_stacks:
        fail_fonts = []
        ok_fonts = []
        reasons: list[str] = []
        for basename in broad:
            item = status[basename].get(stack)
            if item is None:
                continue
            if item["ok"]:
                ok_fonts.append(basename)
            else:
                fail_fonts.append(basename)
                reasons.append(str(item.get("reason") or item.get("support_class")))
        if len(fail_fonts) >= max(1, len(broad) // 2):
            candidates.append(
                {
                    "stack": stack,
                    "bocorpus_occurrences": occurrences.get(stack, 0),
                    "broad_fail_count": len(fail_fonts),
                    "broad_ok_count": len(ok_fonts),
                    "example_reason": reasons[0] if reasons else "",
                    "fail_fonts": "|".join(fail_fonts[:8]),
                }
            )

    candidates.sort(
        key=lambda row: (
            -int(row["bocorpus_occurrences"]),
            -int(row["broad_fail_count"]),
            str(row["stack"]),
        )
    )
    print("\nMost frequent stacks failing on >= half of broad Kyuma fonts:")
    for row in candidates[:40]:
        print(
            f"{row['bocorpus_occurrences']:10d} fails={row['broad_fail_count']}/"
            f"{row['broad_fail_count'] + row['broad_ok_count']} "
            f"{row['stack']!r} reason={row['example_reason']}"
        )

    out = Path("reports/v3_density018/kyuma_common_stack_blocks.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Also write remotely if cwd is project root on server.
    remote_out = Path("/mnt/synthbench/v3_analysis/kyuma_diag/kyuma_common_stack_blocks.csv")
    for path in (out, remote_out):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(candidates[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(candidates)
            print(f"Wrote {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Skip {path}: {exc}")


if __name__ == "__main__":
    main()
