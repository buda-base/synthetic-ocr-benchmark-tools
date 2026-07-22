#!/usr/bin/env python3
"""Render a presentation image of combined Tibetan font augmentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from font_augmentation import AugmentationSpec, create_augmented_font
from font_augmentation_raster_qc import compare_rasters
from render_font_augmentation_audit import (
    coverage_font_row,
    load_probes,
    probe_stacks,
    render_sample,
    validate_shaping,
)
from synthetic_common import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    load_font_catalog,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "font_augmentation_demo"
DEFAULT_PROBES = SCRIPT_DIR / "data" / "font_augmentation" / "probes.txt"
DEFAULT_FONTS = ("Aathup", "GangJie-Drutsa")

COMBINATIONS = (
    ("Original", ()),
    (
        "Narrow · strong slant · thin vertical",
        (
            AugmentationSpec("width", 0.80),
            AugmentationSpec("slant", -14.0),
            AugmentationSpec("vertical_stroke", -0.015),
        ),
    ),
    (
        "Narrow · medium slant · thick horizontal",
        (
            AugmentationSpec("width", 0.80),
            AugmentationSpec("slant", -10.0),
            AugmentationSpec("horizontal_stroke", 0.030),
        ),
    ),
    (
        "Moderately narrow · strong slant · thick vertical",
        (
            AugmentationSpec("width", 0.90),
            AugmentationSpec("slant", -14.0),
            AugmentationSpec("vertical_stroke", 0.030),
        ),
    ),
    (
        "Wide · medium slant · thin horizontal",
        (
            AugmentationSpec("width", 1.20),
            AugmentationSpec("slant", -10.0),
            AugmentationSpec("horizontal_stroke", -0.010),
        ),
    ),
    (
        "Wide · strong slant · thick both axes",
        (
            AugmentationSpec("width", 1.20),
            AugmentationSpec("slant", -14.0),
            AugmentationSpec("vertical_stroke", 0.030),
            AugmentationSpec("horizontal_stroke", 0.030),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a two-font augmentation demo image.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--font", action="append", default=[])
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=DEFAULT_SCRIPTS_CSV)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--fontforge", default="fontforge")
    parser.add_argument("--hb-view", default="hb-view")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def select_fonts(args: argparse.Namespace):
    requested = tuple(args.font) or DEFAULT_FONTS
    by_name = {
        font.basename: font
        for font in load_font_catalog(args.benchmark_csv, args.scripts_csv, args.fonts_csv)
        if font.font_abs_path.is_file()
    }
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown demo font(s): {', '.join(missing)}")
    return [by_name[name] for name in requested]


def apply_combination(font, specs, cache_dir: Path, fontforge_bin: str):
    current_path = font.font_abs_path
    face_index = int(font.ttc_face_index or 0)
    provenance = []
    # Directional PathOps work from the source contours first. FontForge affine
    # rewrites follow afterward; eroding an already rewritten font can lose
    # counter topology in some Tibetan glyphs.
    priority = {
        "vertical_stroke": 0,
        "horizontal_stroke": 0,
        "width": 1,
        "slant": 2,
    }
    for spec in sorted(specs, key=lambda item: priority.get(item.operation, 1)):
        current_path, step = create_augmented_font(
            current_path,
            face_index=face_index,
            spec=spec,
            cache_dir=cache_dir,
            fontforge_bin=fontforge_bin,
        )
        face_index = 0
        provenance.append(step)
    return current_path, provenance


def default_label_font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_demo_sheet(rows: list[dict[str, object]], out_dir: Path) -> Path:
    columns = 3
    cell_w, cell_h = 900, 500
    image_h = 420
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, sheet_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = default_label_font(22)
    detail_font = default_label_font(16)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        draw.text((x + 16, y + 10), str(row["combination"]), fill="black", font=title_font)
        detail = f"{row['font_basename']} · {row['script_type']} · {row['parameters']}"
        draw.text((x + 16, y + 39), detail, fill="black", font=detail_font)
        image_path = out_dir / str(row["image"])
        if image_path.is_file():
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_w - 32, image_h))
                sheet.paste(image, (x + 16, y + 70))
    path = out_dir / "font_augmentation_demo.jpg"
    sheet.save(path, quality=94)
    return path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "png"
    cache_dir = args.out_dir / "font_cache"
    image_dir.mkdir(parents=True, exist_ok=True)

    probes = load_probes(args.probes)
    # The fourth content line is the longer natural-text line; omit it for the
    # compact colleague-facing comparison requested for this demo.
    demo_probes = probes[:3] + probes[4:]
    render_text = "\n".join(demo_probes)
    stacks = probe_stacks(demo_probes)
    rows = []

    for font in select_fonts(args):
        source_row = coverage_font_row(font)
        baseline_png: Path | None = None
        for combo_index, (label, specs) in enumerate(COMBINATIONS):
            variant_path, provenance = apply_combination(
                font,
                specs,
                cache_dir,
                args.fontforge,
            )
            variant_row = coverage_font_row(font, variant_path, 0 if specs else None)
            validation = validate_shaping(source_row, variant_row, stacks)
            sample_id = f"{font.basename}__combo_{combo_index + 1}"
            png_path = image_dir / f"{sample_id}.png"
            render_ok, render_error = render_sample(
                variant_row,
                render_text,
                png_path,
                hb_view_bin=args.hb_view,
                overwrite=args.overwrite_images,
            )
            if baseline_png is None:
                baseline_png = png_path
            raster_qc = (
                compare_rasters(baseline_png, png_path)
                if render_ok and baseline_png.is_file()
                else {"automatic_pass": False, "warnings": ["render_failed"]}
            )
            shape_pass = bool(validation["automatic_pass"])
            validation["shape_automatic_pass"] = shape_pass
            validation["automatic_pass"] = shape_pass and bool(raster_qc["automatic_pass"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "font_basename": font.basename,
                    "script_type": font.script_type,
                    "script_category": font.script_category,
                    "combination": label,
                    "parameters": " · ".join(spec.label for spec in specs) or "unmodified",
                    "variant_font": str(variant_path),
                    "image": str(png_path.relative_to(args.out_dir)),
                    "render_ok": render_ok,
                    "render_error": render_error,
                    "provenance": provenance,
                    "raster_qc": raster_qc,
                    **validation,
                }
            )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sheet_path = make_demo_sheet(rows, args.out_dir)
    failures = sum(1 for row in rows if not row["automatic_pass"])
    print(f"Wrote {sheet_path}")
    print(f"Wrote {manifest_path}")
    print(f"Demo cells: {len(rows)}; automated failures: {failures}")


if __name__ == "__main__":
    main()
