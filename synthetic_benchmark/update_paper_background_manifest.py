#!/usr/bin/env python3
"""Rebuild the committed paper manifest from manually retained review files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from paper_backgrounds import DEFAULT_BACKGROUND_MANIFEST, classify_luminance


DEFAULT_REVIEW_DIR = Path(__file__).resolve().parent / "out" / "blank_paper_review"
LUMINANCE_STATS_VERSION = "luminance-v1-1024px"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_BACKGROUND_MANIFEST)
    return parser.parse_args()


def luminance_stats(path: Path) -> dict[str, object]:
    with Image.open(path) as source:
        mode = source.mode
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        values = np.asarray(image, dtype=np.uint8)
    mean = float(np.mean(values))
    p10 = float(np.percentile(values, 10))
    return {
        "pil_mode": mode,
        "mean_luminance": round(mean, 4),
        "median_luminance": round(float(np.median(values)), 4),
        "p10_luminance": round(p10, 4),
        "p90_luminance": round(float(np.percentile(values, 90)), 4),
        "luminance_std": round(float(np.std(values)), 4),
        "dark_pixel_fraction": round(float(np.mean(values < 128)), 6),
        "luminance_tier": classify_luminance(mean, p10),
        "luminance_stats_version": LUMINANCE_STATS_VERSION,
    }


def main() -> None:
    args = parse_args()
    source_csv = args.review_dir / "blank_paper_samples.csv"
    originals_dir = args.review_dir / "originals"
    with source_csv.open(encoding="utf-8", newline="") as stream:
        source_rows = {
            str(row["local_path"]).split("/", 1)[-1]: row
            for row in csv.DictReader(stream)
        }

    selected: list[dict[str, object]] = []
    paths = sorted(item for item in originals_dir.iterdir() if item.is_file())
    for path in tqdm(paths, desc="Measure paper luminance", unit="image"):
        row = source_rows.get(path.name)
        if row is None:
            raise ValueError(f"Selected background is absent from review CSV: {path.name}")
        source_uri = row["source_s3_uri"]
        selected.append(
            {
                "background_id": hashlib.sha256(source_uri.encode("utf-8")).hexdigest()[:16],
                "w_id": row["w_id"],
                "i_id": row["i_id"],
                "i_version": row["version"],
                "filename": row["filename"],
                "source_s3_uri": source_uri,
                "width": row["width"],
                "height": row["height"],
                "display_width": row["display_width"],
                "display_height": row["display_height"],
                "aspect_ratio": row["aspect_ratio"],
                "exif_orientation": row["exif_orientation"],
                **luminance_stats(path),
                "size_bytes": path.stat().st_size,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(selected[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selected)
    print(f"Wrote {args.output} ({len(selected)} selected background(s))")


if __name__ == "__main__":
    main()
