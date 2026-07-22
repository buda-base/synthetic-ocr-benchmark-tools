#!/usr/bin/env python3
"""Generate paired baseline/font-augmentation sheets for manual Tibetan review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
COVERAGE_DIR = REPO_ROOT / "coverage_report"
if str(COVERAGE_DIR) not in sys.path:
    sys.path.insert(0, str(COVERAGE_DIR))

from coverage_common import FontRow, HarfbuzzShaper, hb_view_render, slugify  # noqa: E402
from font_augmentation import AugmentationSpec, create_augmented_font  # noqa: E402
from font_augmentation_raster_qc import compare_rasters  # noqa: E402
from synthetic_common import (  # noqa: E402
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    load_font_catalog,
    tokenize_tibetan_stacks,
)


DEFAULT_PROBES = SCRIPT_DIR / "data" / "font_augmentation" / "probes.txt"
DEFAULT_REVIEW_ROOT = SCRIPT_DIR / "out" / "font_augmentation_review"
PRESET_SPECS = {
    "round1": (
        AugmentationSpec("width", 0.92),
        AugmentationSpec("width", 1.08),
        AugmentationSpec("slant", -6.0),
        AugmentationSpec("slant", 6.0),
        AugmentationSpec("weight", -0.020),
        AugmentationSpec("weight", 0.020),
    ),
    # Round 1 showed that ±8% width was visually too subtle, while negative
    # slant looked natural and positive slant was unusual for Tibetan.
    "round2": (
        AugmentationSpec("width", 0.80),
        AugmentationSpec("width", 0.88),
        AugmentationSpec("width", 1.15),
        AugmentationSpec("width", 1.25),
        AugmentationSpec("slant", -10.0),
        AugmentationSpec("slant", -14.0),
    ),
    "round3": (
        AugmentationSpec("vertical_stroke", -0.015),
        AugmentationSpec("vertical_stroke", 0.015),
        AugmentationSpec("horizontal_stroke", -0.015),
        AugmentationSpec("horizontal_stroke", 0.015),
    ),
    "round4": (
        AugmentationSpec("vertical_stroke", -0.030),
        AugmentationSpec("vertical_stroke", 0.030),
        AugmentationSpec("horizontal_stroke", -0.030),
        AugmentationSpec("horizontal_stroke", 0.030),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paired baseline and conservative font-augmentation samples."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: out/font_augmentation_review/PRESET).",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_SPECS),
        default="round1",
        help="Reviewed parameter grid to render (default: round1).",
    )
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=DEFAULT_SCRIPTS_CSV)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--font", action="append", default=[], help="Exact basename; repeatable")
    parser.add_argument("--max-fonts", type=int, default=8)
    parser.add_argument("--fontforge", default="fontforge")
    parser.add_argument("--hb-view", default="hb-view")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def load_probes(path: Path) -> list[str]:
    probes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not probes:
        raise ValueError(f"No probes found in {path}")
    return probes


def select_fonts(args: argparse.Namespace):
    fonts = [
        font
        for font in load_font_catalog(args.benchmark_csv, args.scripts_csv, args.fonts_csv)
        if font.font_abs_path.is_file() and font.skt_ok == "1"
    ]
    unique = {}
    for font in fonts:
        unique.setdefault(font.basename, font)
    fonts = list(unique.values())
    if args.font:
        requested = set(args.font)
        selected = [font for font in fonts if font.basename in requested]
        missing = sorted(requested - {font.basename for font in selected})
        if missing:
            raise ValueError(f"Unknown or unavailable font basename(s): {', '.join(missing)}")
        return sorted(selected, key=lambda font: font.basename)

    groups = defaultdict(list)
    for font in fonts:
        groups[(font.script_type, font.script_category)].append(font)
    for group in groups.values():
        group.sort(key=lambda font: font.basename)

    selected = []
    ordered_keys = sorted(groups)
    depth = 0
    while len(selected) < args.max_fonts:
        added = False
        for key in ordered_keys:
            group = groups[key]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) >= args.max_fonts:
                    break
        if not added:
            break
        depth += 1
    return selected


def coverage_font_row(font, path: Path | None = None, face_index: int | None = None) -> FontRow:
    actual_path = path or font.font_abs_path
    actual_face = int(font.ttc_face_index or 0) if face_index is None else face_index
    return FontRow(
        basename=font.basename,
        font_path=actual_path,
        font_path_csv=str(actual_path),
        ttc_face_index=actual_face,
        ttc_face_index_csv=str(actual_face) if actual_face else "",
        ps_name=font.ps_name,
        other_names="",
        font_size_pt=max(32.0, font.font_size_pt),
        dpi=font.dpi,
        skt_ok=1,
    )


def probe_stacks(probes: list[str]) -> list[str]:
    seen = set()
    stacks = []
    for probe in probes:
        for stack in tokenize_tibetan_stacks(probe):
            if stack not in seen:
                seen.add(stack)
                stacks.append(stack)
    return stacks


def validate_shaping(source: FontRow, variant: FontRow, stacks: list[str]) -> dict[str, object]:
    source_shaper = HarfbuzzShaper(source)
    variant_shaper = HarfbuzzShaper(variant)
    source_supported = 0
    regressions = []
    for stack in stacks:
        before = source_shaper.shape(stack)
        if not before["ok"]:
            continue
        source_supported += 1
        after = variant_shaper.shape(stack)
        if not after["ok"]:
            regressions.append({"stack": stack, "reason": after["reason"]})
    return {
        "source_supported_stacks": source_supported,
        "regression_count": len(regressions),
        "regressions": regressions,
        "automatic_pass": not regressions and source_supported > 0,
    }


def render_sample(
    font_row: FontRow,
    text: str,
    path: Path,
    *,
    hb_view_bin: str,
    overwrite: bool,
) -> tuple[bool, str]:
    if path.is_file() and not overwrite:
        return True, ""
    return hb_view_render(font_row, text, path, hb_view_bin=hb_view_bin, margin=60)


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_review_csv(rows: list[dict[str, object]], path: Path) -> None:
    prior = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            prior = {row["sample_id"]: row for row in csv.DictReader(stream)}
    fieldnames = [
        "sample_id",
        "font_basename",
        "script_type",
        "script_category",
        "operation",
        "value",
        "automatic_pass",
        "automatic_failure",
        "decision",
        "reason_tags",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row["operation"] == "baseline":
                continue
            old = prior.get(str(row["sample_id"]), {})
            failures = "; ".join(
                str(item.get("reason") or "")
                for item in row.get("regressions", [])
                if item.get("reason")
            )
            raster_failures = "; ".join(
                str(item) for item in row.get("raster_qc", {}).get("warnings", [])
            )
            row["automatic_failure"] = (
                row.get("generation_error") or failures or raster_failures
            )
            writer.writerow(
                {
                    key: (
                        old.get(key, "")
                        if key in {"decision", "reason_tags", "notes"}
                        else row.get(key, "")
                    )
                    for key in fieldnames
                }
            )


def make_contact_sheet(rows: list[dict[str, object]], out_dir: Path, columns: int) -> Path:
    cell_w, cell_h = 500, 280
    image_h = 215
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, sheet_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        image_path = out_dir / str(row["image"])
        if image_path.is_file():
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_w - 20, image_h - 10))
                sheet.paste(image, (x + 10, y + 5))
        passed = bool(row["automatic_pass"])
        if not passed:
            draw.rectangle((x + 2, y + 2, x + cell_w - 3, y + cell_h - 3), outline="red", width=3)
        label = (
            f"{row['font_basename']} | {row['script_type']} | {row['script_category']}\n"
            f"{row['operation']} {row['value']} | auto={'PASS' if passed else 'FAIL'} | "
            f"{row['sample_id']}"
        )
        draw.multiline_text((x + 10, y + image_h + 5), label, fill="black", spacing=2)
    path = out_dir / "contact_sheet.jpg"
    sheet.save(path, quality=92)
    return path


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = DEFAULT_REVIEW_ROOT / args.preset
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "png"
    cache_dir = args.out_dir / "font_cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    probes = load_probes(args.probes)
    # hb-view preserves explicit newlines, keeping long probe sets legible when
    # they are reduced into contact-sheet cells.
    render_text = "\n".join(probes)
    stacks = probe_stacks(probes)
    fonts = select_fonts(args)
    specs = PRESET_SPECS[args.preset]
    if not fonts:
        raise SystemExit("No eligible fonts selected")

    rows = []
    for font in fonts:
        source_row = coverage_font_row(font)
        baseline_id = f"{slugify(font.basename)}__baseline"
        baseline_png = image_dir / f"{baseline_id}.png"
        render_ok, render_error = render_sample(
            source_row,
            render_text,
            baseline_png,
            hb_view_bin=args.hb_view,
            overwrite=args.overwrite_images,
        )
        baseline_shape = validate_shaping(source_row, source_row, stacks)
        baseline_raster = (
            compare_rasters(baseline_png, baseline_png)
            if render_ok
            else {"automatic_pass": False, "warnings": ["render_failed"]}
        )
        baseline_shape_pass = bool(baseline_shape["automatic_pass"])
        baseline_shape["shape_automatic_pass"] = baseline_shape_pass
        baseline_shape["automatic_pass"] = baseline_shape_pass and bool(
            baseline_raster["automatic_pass"]
        )
        rows.append(
            {
                "sample_id": baseline_id,
                "font_basename": font.basename,
                "script_type": font.script_type,
                "script_category": font.script_category,
                "operation": "baseline",
                "value": "",
                "variant_font": str(font.font_abs_path),
                "image": str(baseline_png.relative_to(args.out_dir)),
                "render_ok": render_ok,
                "render_error": render_error,
                "raster_qc": baseline_raster,
                **baseline_shape,
            }
        )

        for spec in specs:
            sample_id = f"{slugify(font.basename)}__{spec.label}"
            try:
                variant_path, provenance = create_augmented_font(
                    font.font_abs_path,
                    face_index=int(font.ttc_face_index or 0),
                    spec=spec,
                    cache_dir=cache_dir,
                    fontforge_bin=args.fontforge,
                )
                variant_row = coverage_font_row(font, variant_path, 0)
                shape_result = validate_shaping(source_row, variant_row, stacks)
                png_path = image_dir / f"{sample_id}.png"
                render_ok, render_error = render_sample(
                    variant_row,
                    render_text,
                    png_path,
                    hb_view_bin=args.hb_view,
                    overwrite=args.overwrite_images,
                )
                raster_result = (
                    compare_rasters(baseline_png, png_path)
                    if render_ok
                    else {"automatic_pass": False, "warnings": ["render_failed"]}
                )
                shape_pass = bool(shape_result["automatic_pass"])
                shape_result["shape_automatic_pass"] = shape_pass
                shape_result["automatic_pass"] = shape_pass and bool(
                    raster_result["automatic_pass"]
                )
                error = ""
            except Exception as exc:
                variant_path = Path("")
                provenance = {}
                shape_result = {
                    "source_supported_stacks": 0,
                    "regression_count": 1,
                    "regressions": [],
                    "automatic_pass": False,
                }
                png_path = image_dir / f"{sample_id}.png"
                render_ok, render_error = False, ""
                raster_result = {"automatic_pass": False, "warnings": ["generation_failed"]}
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "font_basename": font.basename,
                    "script_type": font.script_type,
                    "script_category": font.script_category,
                    "operation": spec.operation,
                    "value": spec.value,
                    "variant_font": str(variant_path),
                    "image": str(png_path.relative_to(args.out_dir)),
                    "render_ok": render_ok,
                    "render_error": render_error,
                    "generation_error": error,
                    "provenance": provenance,
                    "raster_qc": raster_result,
                    **shape_result,
                }
            )

    manifest_path = args.out_dir / "manifest.json"
    review_path = args.out_dir / "review.csv"
    write_manifest(rows, manifest_path)
    write_review_csv(rows, review_path)
    sheet_path = make_contact_sheet(rows, args.out_dir, columns=len(specs) + 1)
    failures = sum(1 for row in rows if not row["automatic_pass"])
    print(f"Wrote {sheet_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {review_path}")
    print(f"Samples: {len(rows)} across {len(fonts)} fonts; automated failures: {failures}")


if __name__ == "__main__":
    main()
