#!/usr/bin/env python3
"""Summarize why Kyuma stack probes fail in stack_support.parquet."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from build_render_plan import normalized_script
from synthetic_common import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    DEFAULT_SUPPORT_PARQUET,
    load_font_catalog,
)


def main() -> None:
    fonts = load_font_catalog(
        DEFAULT_BENCHMARK_CSV, DEFAULT_SCRIPTS_CSV, DEFAULT_FONTS_CSV
    )
    kyuma = {
        font.basename
        for font in fonts
        if font.script_category.startswith("Kyuma")
        and normalized_script(font.script_type)
    }
    columns = [
        "test_kind",
        "basename",
        "ok",
        "support_class",
        "reason",
        "placement_warning_count",
        "notdef_count",
    ]
    reasons: Counter[str] = Counter()
    by_font_class: dict[str, Counter[str]] = defaultdict(Counter)
    pf = pq.ParquetFile(DEFAULT_SUPPORT_PARQUET)
    for batch in pf.iter_batches(batch_size=50_000, columns=columns):
        for row in batch.to_pylist():
            if row["test_kind"] != "stack" or row["basename"] not in kyuma:
                continue
            if row["ok"]:
                continue
            support_class = str(row.get("support_class") or "fail")
            reason = str(row.get("reason") or "")
            first_reason = reason.split("|")[0] if reason else ""
            key = support_class if not first_reason else f"{support_class}|{first_reason}"
            reasons[key] += 1
            by_font_class[str(row["basename"])][support_class] += 1

    print("Top fail classes/reasons across Kyuma:")
    for key, count in reasons.most_common(30):
        print(f"{count:8d} {key}")
    print("\nPer-font support_class for failures:")
    for basename in sorted(kyuma):
        print(basename, by_font_class[basename].most_common(5))


if __name__ == "__main__":
    main()
