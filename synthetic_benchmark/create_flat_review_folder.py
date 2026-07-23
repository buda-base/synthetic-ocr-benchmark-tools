#!/usr/bin/env python3
"""Expose rendered benchmark images in one flat folder using hard links."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "font"


def main() -> None:
    args = parse_args()
    output = args.output or args.render_dir / "flat"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    catalog_dir = args.render_dir / "checkpoints" / "catalog_batches"
    for path in sorted(catalog_dir.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["output_sequence"]))

    manifest_rows = []
    for row in rows:
        source = args.render_dir / row["image_file_name"]
        face = str(row.get("ttc_face_index") or "0")
        destination_name = (
            f"{int(row['output_sequence']):04d}__"
            f"{safe_name(row.get('font_file') or row.get('font_name') or 'font')}"
            f"__f{face}__{source.name}"
        )
        destination = output / destination_name
        if destination.exists():
            destination.unlink()
        os.link(source, destination)
        manifest = dict(row)
        manifest["flat_file_name"] = destination_name
        manifest_rows.append(manifest)

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(manifest_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} hard-linked images to {output}")


if __name__ == "__main__":
    main()
