#!/usr/bin/env python3
"""Build an equal-Uchen/Ume review plan covering every eligible font face."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_render_plan import (
    make_per_font_plan_rows,
    normalized_script,
    write_summary,
)
from synthetic_common import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_CHUNKS_PARQUET,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    DEFAULT_SUPPORT_PARQUET,
    load_font_catalog,
    load_supported_stacks,
)
from build_render_plan import load_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-images", type=int, default=1000)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PARQUET)
    parser.add_argument("--support-parquet", type=Path, default=DEFAULT_SUPPORT_PARQUET)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=DEFAULT_SCRIPTS_CSV)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def face_key(row: dict[str, object]) -> tuple[str, str]:
    return (
        str(row["font_abs_path"]),
        str(row.get("ttc_face_index") or ""),
    )


def select_round_robin(
    rows: list[dict[str, object]],
    *,
    script: str,
    target: int,
) -> list[dict[str, object]]:
    by_face: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if normalized_script(str(row.get("script_type") or "")) == script:
            by_face[face_key(row)].append(row)
    if target < len(by_face):
        raise ValueError(
            f"{target} {script} slots cannot cover {len(by_face)} font faces"
        )
    for face_rows in by_face.values():
        face_rows.sort(key=lambda row: int(row["image_id"]))

    selected: list[dict[str, object]] = []
    depth = 0
    ordered_faces = sorted(by_face)
    while len(selected) < target:
        for key in ordered_faces:
            face_rows = by_face[key]
            if not face_rows:
                continue
            # A few highly restricted fonts have fewer compatible source chunks
            # than their review quota. Reuse those chunks rather than dropping
            # the face; rendering still assigns independent font/document
            # augmentations to each copied row.
            selected.append(dict(face_rows[depth % len(face_rows)]))
            if len(selected) == target:
                break
        depth += 1
    if len(selected) != target:
        raise ValueError(f"Only selected {len(selected)}/{target} {script} rows")
    return selected


def main() -> None:
    args = parse_args()
    if args.target_images % 2:
        raise SystemExit("--target-images must be even for a 50/50 script split")

    fonts = load_font_catalog(
        args.benchmark_csv,
        args.scripts_csv,
        args.fonts_csv,
    )
    supported = load_supported_stacks(args.support_parquet)
    eligible = [
        font
        for font in fonts
        if font.basename in supported
        and normalized_script(font.script_type) in {"uchen", "ume"}
    ]
    face_counts = Counter(normalized_script(font.script_type) for font in eligible)
    target_per_script = args.target_images // 2
    images_per_font = max(
        math.ceil(target_per_script / face_counts[script])
        for script in ("uchen", "ume")
    )
    rows = make_per_font_plan_rows(
        fonts=eligible,
        chunks=load_chunks(args.chunks),
        supported=supported,
        images_per_font=images_per_font,
        seed=args.seed,
        max_chunk_reuse_ratio=0.10,
    )

    selected: list[dict[str, object]] = []
    next_start = {"uchen": 1, "ume": 1001}
    for script in ("uchen", "ume"):
        script_rows = select_round_robin(
            rows,
            script=script,
            target=target_per_script,
        )
        for offset, row in enumerate(script_rows):
            row["image_id"] = next_start[script] + offset
        selected.extend(script_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), args.output, compression="zstd")
    write_summary(selected, args.output.with_suffix(".summary.csv"))

    selected_counts = Counter(str(row["script"]) for row in selected)
    selected_faces = Counter(
        (str(row["script"]), face_key(row))
        for row in selected
    )
    ranges = {
        script: (
            min(
                count
                for (row_script, _key), count in selected_faces.items()
                if row_script == script
            ),
            max(
                count
                for (row_script, _key), count in selected_faces.items()
                if row_script == script
            ),
        )
        for script in ("uchen", "ume")
    }
    print(
        f"Wrote {len(selected)} rows to {args.output}; scripts={dict(selected_counts)}; "
        f"eligible_faces={dict(face_counts)}; pages_per_face={ranges}"
    )


if __name__ == "__main__":
    main()
