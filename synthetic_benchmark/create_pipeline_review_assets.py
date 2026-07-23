#!/usr/bin/env python3
"""Create browser-friendly previews from a mixed JPEG/TIFF review render."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_RENDER_DIR = Path(__file__).resolve().parent / "out" / "pipeline_review_50"
DEFAULT_BLOG_ASSET_DIR = (
    Path(__file__).resolve().parents[1] / "blog" / "assets" / "07-paper-backgrounds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", type=Path, nargs="?", default=DEFAULT_RENDER_DIR)
    parser.add_argument("--blog-asset-dir", type=Path, default=DEFAULT_BLOG_ASSET_DIR)
    return parser.parse_args()


def load_rows(render_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    checkpoint_dir = render_dir / "checkpoints" / "catalog_batches"
    for path in sorted(checkpoint_dir.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["output_sequence"]))
    if not rows:
        raise ValueError(f"No checkpoint rows found under {checkpoint_dir}")
    return rows


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def image_path(render_dir: Path, row: dict[str, str]) -> Path:
    return render_dir / row["image_file_name"]


def font_label(row: dict[str, str]) -> str:
    return Path(row.get("font_file") or row.get("font_name") or "font").stem


def rgb_preview(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def row_label(index: int, row: dict[str, str]) -> tuple[str, str]:
    paper = row["document_augmentation_paper_source"]
    mode = row["document_augmentation_output_mode"]
    width = row["output_width_px"]
    quality = row["output_jpeg_quality"] if row["image_file_name"].endswith(".jpg") else "G4"
    line_one = (
        f"{index:02d}  {font_label(row)}  |  {paper}/{mode}  |  "
        f"{width}px  |  Q{quality}"
    )
    effects = [
        value
        for value in (
            row.get("document_augmentation_local"),
            row.get("document_augmentation_spatial"),
            "TPS" if row.get("document_augmentation_tps_strength") else "",
            (
                f"rot {float(row['document_augmentation_rotation_deg']):+.2f}"
                if row.get("document_augmentation_rotation_deg")
                else ""
            ),
            (
                f"blur {row['document_augmentation_blur']}"
                if row.get("document_augmentation_blur")
                else ""
            ),
        )
        if value
    ]
    return line_one, " + ".join(effects) or "no ink/geometric effect"


def make_contact_sheet(
    render_dir: Path,
    indexed_rows: list[tuple[int, dict[str, str]]],
    destination: Path,
) -> None:
    columns = 5
    tile_width = 520
    image_height = 130
    label_height = 58
    rows_count = (len(indexed_rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows_count * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(14)
    detail_font = load_font(12)
    for position, (index, row) in enumerate(indexed_rows):
        column = position % columns
        row_number = position // columns
        left = column * tile_width
        top = row_number * (image_height + label_height)
        preview = ImageOps.fit(
            rgb_preview(image_path(render_dir, row)),
            (tile_width - 12, image_height - 12),
            method=Image.Resampling.LANCZOS,
        )
        sheet.paste(preview, (left + 6, top + 6))
        line_one, line_two = row_label(index, row)
        draw.text((left + 7, top + image_height + 3), line_one, fill="black", font=title_font)
        draw.text(
            (left + 7, top + image_height + 26),
            line_two,
            fill=(60, 60, 60),
            font=detail_font,
        )
        draw.rectangle(
            (left, top, left + tile_width - 1, top + image_height + label_height - 1),
            outline=(190, 190, 190),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=90, subsampling=0)


def select_representatives(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    selectors = {
        "real_rgb": lambda row: row["document_augmentation_paper_source"] == "real"
        and row["document_augmentation_output_mode"] == "RGB",
        "real_grayscale": lambda row: row["document_augmentation_paper_source"] == "real"
        and row["document_augmentation_output_mode"] == "L",
        "real_bilevel_preview": lambda row: row["document_augmentation_paper_source"] == "real"
        and row["document_augmentation_output_mode"] == "1",
        "synthetic_paper": lambda row: row["document_augmentation_paper_source"] == "synthetic",
        "clean": lambda row: row["document_augmentation_paper_source"] == "clean",
        "jpeg_quality_65": lambda row: row["image_file_name"].endswith(".jpg")
        and row["output_jpeg_quality"] == "65",
    }
    selected: dict[str, dict[str, str]] = {}
    for name, selector in selectors.items():
        selected[name] = next(row for row in rows if selector(row))
    return selected


def write_review_manifest(
    path: Path,
    rows: list[dict[str, str]],
    preview_paths: list[Path],
) -> None:
    fields = [
        "review_index",
        "preview_file",
        "image_file_name",
        "font_file",
        "font_name",
        "document_augmentation_paper_source",
        "document_augmentation_background_id",
        "document_augmentation_output_mode",
        "output_width_px",
        "output_jpeg_quality",
        "document_augmentation_local",
        "document_augmentation_spatial",
        "document_augmentation_tps_strength",
        "document_augmentation_rotation_deg",
        "document_augmentation_blur",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (row, preview_path) in enumerate(zip(rows, preview_paths), start=1):
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "review_index": index,
                    "preview_file": preview_path.name,
                }
            )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.render_dir)
    review_dir = args.render_dir / "review"
    preview_dir = review_dir / "images"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_paths: list[Path] = []
    for index, row in enumerate(rows, start=1):
        destination = preview_dir / f"{index:02d}_{font_label(row)}.jpg"
        preview = rgb_preview(image_path(args.render_dir, row))
        preview.thumbnail((1600, 500), Image.Resampling.LANCZOS)
        preview.save(destination, "JPEG", quality=90, subsampling=0)
        preview_paths.append(destination)

    args.blog_asset_dir.mkdir(parents=True, exist_ok=True)
    indexed_rows = list(enumerate(rows, start=1))
    for sheet_number, start in enumerate(range(0, len(rows), 25), start=1):
        destination = args.blog_asset_dir / f"contact_sheet_{sheet_number:02d}.jpg"
        make_contact_sheet(
            args.render_dir,
            indexed_rows[start : start + 25],
            destination,
        )
        (review_dir / destination.name).write_bytes(destination.read_bytes())

    for name, row in select_representatives(rows).items():
        preview = rgb_preview(image_path(args.render_dir, row))
        preview.thumbnail((1800, 600), Image.Resampling.LANCZOS)
        preview.save(
            args.blog_asset_dir / f"{name}.jpg",
            "JPEG",
            quality=92,
            subsampling=0,
        )

    write_review_manifest(review_dir / "manifest.csv", rows, preview_paths)
    print(f"Wrote {len(preview_paths)} previews under {review_dir}")
    print(f"Wrote blog assets under {args.blog_asset_dir}")


if __name__ == "__main__":
    main()
