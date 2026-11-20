#!/usr/bin/env python3
"""Re-render retention region crops that violate the 20–1,500 char contract."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from paper_backgrounds import load_paper_backgrounds, prepare_paper_background_cache
from render_retention_en_zh import (
    make_contact_sheet,
    parquet_row,
    render_one,
    write_outputs,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()

    manifest_path = args.out_dir / "render_manifest.parquet"
    plan_path = args.out_dir / "render_plan.parquet"
    results = pq.read_table(manifest_path).to_pylist()
    plan_table = pq.read_table(plan_path)
    plan = plan_table.to_pylist()
    bad_keys = {
        (row["language_group"], int(row["image_id"]))
        for row in results
        if row["layout_class"] == "region"
        and not (
            20 <= int(row["character_count"]) <= 1500
            and 1 <= int(row["line_count"]) <= 30
        )
    }
    if not bad_keys:
        print("No invalid regions found")
        return

    # A very short or formula-heavy Wikipedia extract can contain fewer than
    # 20 useful characters even after the renderer extends it with the title.
    # Replace only that source slice with another already-approved Batch source
    # of the same language/variant and region kind. This preserves source and
    # font holdout exclusivity while avoiding duplicated filler text.
    source_fields = (
        "source_text",
        "source_title",
        "source_document_id",
        "etext_source",
    )
    plan_by_key = {
        (row["language_group"], int(row["image_id"])): row for row in plan
    }
    for key in sorted(bad_keys):
        row = plan_by_key[key]
        donors = [
            candidate
            for candidate in plan
            if candidate["language_group"] == row["language_group"]
            and candidate["language"] == row["language"]
            and candidate["region_kind"] == row["region_kind"]
            and candidate["source_document_id"] != row["source_document_id"]
            and len("".join(candidate["source_text"].split())) >= 100
        ]
        if not donors:
            raise RuntimeError(f"No suitable replacement source for {key}")
        donor = min(
            donors,
            key=lambda candidate: (
                abs(int(candidate["image_id"]) - int(row["image_id"])),
                int(candidate["image_id"]),
            ),
        )
        for field in source_fields:
            row[field] = donor[field]
    pq.write_table(
        pa.Table.from_pylist(plan, schema=plan_table.schema),
        plan_path,
        compression="zstd",
    )

    backgrounds = load_paper_backgrounds()
    prepare_paper_background_cache(
        plan,
        backgrounds,
        cache_dir=args.cache_dir / "paper_backgrounds",
        workers=min(16, args.jobs * 2),
    )
    repair_rows = [
        row
        for row in plan
        if (row["language_group"], int(row["image_id"])) in bad_keys
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        repaired = list(
            tqdm(
                executor.map(
                    render_one,
                    ((row, str(args.out_dir)) for row in repair_rows),
                    chunksize=4,
                ),
                total=len(repair_rows),
                desc="Repair retention regions",
                unit="page",
            )
        )
    repaired_by_key = {
        (row["language_group"], int(row["image_id"])): row for row in repaired
    }
    merged = [
        repaired_by_key.get((row["language_group"], int(row["image_id"])), row)
        for row in results
    ]
    remaining = [
        row
        for row in merged
        if row["layout_class"] == "region"
        and not (
            20 <= int(row["character_count"]) <= 1500
            and 1 <= int(row["line_count"]) <= 30
        )
    ]
    if remaining:
        raise RuntimeError(f"{len(remaining)} invalid regions remain after repair")

    augmentation_manifest = json.loads(
        (args.out_dir / "document_augmentation_manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (args.out_dir / "reports" / "report.json").read_text(encoding="utf-8")
    )
    write_outputs(args.out_dir, merged, augmentation_manifest)
    write_report(
        args.out_dir,
        merged,
        {},
        report["near_duplicate_check"],
    )
    make_contact_sheet(args.out_dir, merged, "en")
    make_contact_sheet(args.out_dir, merged, "zh")
    pq.write_table(
        pa.Table.from_pylist([parquet_row(row) for row in merged]),
        manifest_path,
        compression="zstd",
    )
    print(f"Repaired {len(repaired)} regions; all region bounds now pass")


if __name__ == "__main__":
    main()
