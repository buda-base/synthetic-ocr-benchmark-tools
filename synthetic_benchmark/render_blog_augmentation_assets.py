#!/usr/bin/env python3
"""Build compact, self-contained image assets for blog posts 4 and 5."""

from __future__ import annotations

import csv
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
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


def save_center_square(image: Image.Image, destination: Path, *, fraction: float = 0.88) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    side = round(min(image.width, image.height) * fraction)
    left = (image.width - side) // 2
    top = max(0, (image.height - side) // 2 - round(image.height * 0.03))
    image.crop((left, top, left + side, top + side)).save(destination, quality=94)


def crop_center_square(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        save_center_square(image, destination)


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
    source = REPO_ROOT / "blog" / "assets" / "02-rendering" / "pecha_uchen_example.jpg"
    destination = BLOG_ASSETS / "05-image-augmentation"
    from render_document_augmentation_audit import _factories, apply_effect

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {source}")
    save_center_square(
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        destination / "original.jpg",
    )
    selected = {
        "bleedthrough": "medium",
        "inkbleed": "medium",
        "letterpress": "medium",
        "dirtydrum": "mild",
        "colorpaper": "medium",
        "noisetexturize": "medium",
        "subtlenoise": "medium",
        "inkshifter": "medium",
        "folding": "medium",
        "gaussian_blur": "medium",
    }
    factories = _factories()
    for index, (name, strength) in enumerate(selected.items()):
        sample_seed = 5000 + index
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        output = apply_effect(image, factories[name][strength]())
        save_center_square(
            Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)),
            destination / f"{name}.jpg",
        )


def load_combined_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted((root / "checkpoints" / "catalog_batches").glob("*.csv")):
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return sorted(rows, key=lambda row: int(row["output_sequence"]))


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
    for row, (spatial, spatial_strength, blur) in zip(rows, forced_finishing):
        if int(row["output_sequence"]) not in {2, 5, 7}:
            continue
        apply_row = dict(row)
        # The source JPEG already contains its assigned local appearance effect.
        apply_row["document_augmentation_local"] = ""
        apply_row["document_augmentation_local_strength"] = ""
        apply_row["document_augmentation_rotation_strength"] = ""
        apply_row["document_augmentation_rotation_deg"] = ""
        apply_row["document_augmentation_tps_strength"] = ""
        apply_row["document_augmentation_tps_y_norm"] = ""
        apply_row["document_augmentation_tps_offsets_height"] = ""
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
        crop_center_square(
            output,
            BLOG_ASSETS
            / "05-image-augmentation"
            / f"combined_{int(row['output_sequence']):02d}.jpg",
        )


def build_part6_assets() -> None:
    from document_augmentation import apply_document_augmentations

    source = REPO_ROOT / "blog" / "assets" / "02-rendering" / "pecha_uchen_example.jpg"
    full_dir = OUT_DIR / "blog_part6_geometry_full"
    destination = BLOG_ASSETS / "06-geometric-augmentation"
    cases = (
        {
            "name": "rotation_weak",
            "rotation_deg": 0.55,
            "rotation_strength": "typical",
        },
        {
            "name": "rotation_strong",
            "rotation_deg": -1.80,
            "rotation_strength": "high",
        },
        {
            "name": "tps_rotation_weak",
            "rotation_deg": 0.40,
            "rotation_strength": "typical",
            "tps_strength": "typical",
            "tps_y_norm": 0.50,
            "tps_offsets": "-0.02000000|0.01000000|0.02500000|0.00500000|-0.02000000",
        },
        {
            "name": "tps_rotation_strong",
            "rotation_deg": -1.20,
            "rotation_strength": "high",
            "tps_strength": "high",
            "tps_y_norm": 0.50,
            "tps_offsets": "-0.04500000|0.02000000|0.04000000|0.01500000|-0.03000000",
        },
    )
    for index, case in enumerate(cases):
        row = {
            "document_augmentation_seed": 6000 + index,
            "document_augmentation_local": "",
            "document_augmentation_local_strength": "",
            "document_augmentation_spatial": "",
            "document_augmentation_spatial_strength": "",
            "document_augmentation_blur": "",
            "document_augmentation_rotation_deg": case["rotation_deg"],
            "document_augmentation_rotation_strength": case["rotation_strength"],
            "document_augmentation_tps_strength": case.get("tps_strength", ""),
            "document_augmentation_tps_y_norm": case.get("tps_y_norm", ""),
            "document_augmentation_tps_offsets_height": case.get("tps_offsets", ""),
        }
        full_image = full_dir / f"{case['name']}.jpg"
        apply_document_augmentations(source, full_image, row, jpeg_quality=94)
        crop_center_square(full_image, destination / f"{case['name']}.jpg")


def main() -> None:
    build_part4_assets()
    build_part5_type_assets()
    build_combined_asset()
    build_part6_assets()
    print(f"Wrote blog assets under {BLOG_ASSETS}")


if __name__ == "__main__":
    main()
