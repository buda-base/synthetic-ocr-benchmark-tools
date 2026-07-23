#!/usr/bin/env python3
"""Build compact, self-contained image assets for blog posts 4 and 5."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "out"
BLOG_ASSETS = REPO_ROOT / "blog" / "assets"


def label_font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def save_resized(source: Path, destination: Path, *, width: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        height = round(width * image.height / image.width)
        image.resize((width, height), Image.Resampling.LANCZOS).save(destination, quality=92)


def montage(
    cells: list[tuple[str, Path]],
    destination: Path,
    *,
    columns: int,
    cell_width: int = 1000,
    image_height: int = 300,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    label_height = 64
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = label_font(21)
    for index, (label, path) in enumerate(cells):
        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + label_height)
        wrapped = "\n".join(textwrap.wrap(label, width=max(28, cell_width // 13))[:2])
        draw.multiline_text((x + 12, y + 8), wrapped, fill="black", font=font, spacing=2)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width - 24, image_height - 8), Image.Resampling.LANCZOS)
            paste_x = x + (cell_width - image.width) // 2
            paste_y = y + label_height + (image_height - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
    sheet.save(destination, quality=92)


def build_part4_assets() -> None:
    source = OUT_DIR / "font_augmentation_demo"
    destination = BLOG_ASSETS / "04-font-augmentation"
    save_resized(
        source / "font_augmentation_demo.jpg",
        destination / "overview.jpg",
        width=1800,
    )
    montage(
        [
            ("Uchen · original", source / "png" / "Aathup__combo_1.png"),
            ("Uchen · narrow, slanted, thin vertical", source / "png" / "Aathup__combo_2.png"),
            ("Uchen · wide, slanted, thin horizontal", source / "png" / "Aathup__combo_5.png"),
            ("Ume · original", source / "png" / "GangJie-Drutsa__combo_1.png"),
            ("Ume · narrow, slanted, thick horizontal", source / "png" / "GangJie-Drutsa__combo_3.png"),
            ("Ume · wide, slanted, thick both axes", source / "png" / "GangJie-Drutsa__combo_6.png"),
        ],
        destination / "examples.jpg",
        columns=3,
        cell_width=850,
        image_height=330,
    )
    save_resized(
        OUT_DIR / "font_augmentation_review" / "round4" / "contact_sheet.jpg",
        destination / "qc_boundaries.jpg",
        width=1800,
    )


def build_part5_type_assets() -> None:
    review = OUT_DIR / "document_augmentation_review"
    source = REPO_ROOT / "blog" / "assets" / "02-rendering" / "pecha_uchen_example.jpg"
    samples = review / "pecha_uchen_example"
    destination = BLOG_ASSETS / "05-image-augmentation"
    montage(
        [
            ("Original", source),
            ("Bleed-through · medium", samples / "bleedthrough__medium.jpg"),
            ("Ink bleed · medium", samples / "inkbleed__medium.jpg"),
            ("Letterpress · medium", samples / "letterpress__medium.jpg"),
            ("Dirty drum · mild", samples / "dirtydrum__mild.jpg"),
            ("Dithering · mild", samples / "dithering__mild.jpg"),
        ],
        destination / "local_ink.jpg",
        columns=3,
        cell_width=800,
        image_height=230,
    )
    montage(
        [
            ("Original", source),
            ("Color paper · medium", samples / "colorpaper__medium.jpg"),
            ("Noise texture · medium", samples / "noisetexturize__medium.jpg"),
            ("Subtle noise · medium", samples / "subtlenoise__medium.jpg"),
        ],
        destination / "paper_and_noise.jpg",
        columns=2,
        cell_width=1050,
        image_height=280,
    )
    montage(
        [
            ("Original", source),
            ("InkShifter · medium", samples / "inkshifter__medium.jpg"),
            ("Folding · medium", samples / "folding__medium.jpg"),
            ("Gaussian blur · medium", samples / "gaussian_blur__medium.jpg"),
        ],
        destination / "spatial_and_blur.jpg",
        columns=2,
        cell_width=1050,
        image_height=280,
    )


def load_combined_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted((root / "checkpoints" / "catalog_batches").glob("*.csv")):
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return sorted(rows, key=lambda row: int(row["output_sequence"]))


def combined_label(row: dict[str, str]) -> str:
    parts = [row.get("font_name") or "font"]
    for spec in (row.get("font_augmentation_specs") or "").split("|"):
        replacements = (
            ("vertical_stroke_", "v-stroke "),
            ("horizontal_stroke_", "h-stroke "),
            ("width_", "width "),
            ("slant_", "slant "),
        )
        for source, replacement in replacements:
            if spec.startswith(source):
                spec = spec.replace(source, replacement, 1)
                break
        if spec:
            parts.append(spec)
    local = row.get("document_augmentation_local") or ""
    if local:
        parts.append(f"{local} {row.get('document_augmentation_local_strength')}")
    spatial = row.get("document_augmentation_spatial") or ""
    if spatial:
        parts.append(f"{spatial} {row.get('document_augmentation_spatial_strength')}")
    blur = row.get("document_augmentation_blur") or ""
    if blur:
        parts.append(f"blur {blur}")
    shorthand_count = int(row.get("shorthand_count") or 0)
    if shorthand_count:
        parts.append(f"{shorthand_count} shorthands")
    return " · ".join(parts)


def build_combined_asset() -> None:
    root = OUT_DIR / "blog_part5_combined"
    rows = load_combined_rows(root)
    from document_augmentation import apply_document_augmentations

    forced_finishing = (
        ("inkshifter", "mild", "medium"),
        ("folding", "mild", "mild"),
        ("inkshifter", "medium", ""),
        ("folding", "medium", "medium"),
        ("", "", "medium"),
        ("inkshifter", "mild", "mild"),
    )
    finished_dir = OUT_DIR / "blog_part5_combined_finishing"
    finished_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for row, (spatial, spatial_strength, blur) in zip(rows, forced_finishing):
        apply_row = dict(row)
        # The source JPEG already contains its assigned local appearance effect.
        apply_row["document_augmentation_local"] = ""
        apply_row["document_augmentation_local_strength"] = ""
        apply_row["document_augmentation_spatial"] = spatial
        apply_row["document_augmentation_spatial_strength"] = spatial_strength
        apply_row["document_augmentation_blur"] = blur
        output = finished_dir / f"{int(row['output_sequence']):04d}.jpg"
        apply_document_augmentations(
            root / row["image_file_name"],
            output,
            apply_row,
            jpeg_quality=88,
        )
        display_row = dict(row)
        display_row["document_augmentation_spatial"] = spatial
        display_row["document_augmentation_spatial_strength"] = spatial_strength
        display_row["document_augmentation_blur"] = blur
        cells.append((combined_label(display_row), output))
    montage(
        cells,
        BLOG_ASSETS / "05-image-augmentation" / "combined_pipeline.jpg",
        columns=2,
        cell_width=1200,
        image_height=330,
    )


def main() -> None:
    build_part4_assets()
    build_part5_type_assets()
    build_combined_asset()
    print(f"Wrote blog assets under {BLOG_ASSETS}")


if __name__ == "__main__":
    main()
