#!/usr/bin/env python3
"""Render mild/medium/strong document-augmentation sheets for manual review."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from augraphy import (
    BleedThrough,
    ColorPaper,
    DirtyDrum,
    Dithering,
    Folding,
    Hollow,
    InkBleed,
    InkShifter,
    Letterpress,
    LinesDegradation,
    LowLightNoise,
    NoiseTexturize,
    SubtleNoise,
)
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "document_augmentation_review"
DEFAULT_INPUTS = (
    REPO_ROOT / "blog" / "assets" / "02-rendering" / "pecha_uchen_example.jpg",
    REPO_ROOT / "blog" / "assets" / "02-rendering" / "ume_none_crop.jpg",
)
STRENGTHS = ("mild", "medium", "strong")


def _factories() -> dict[str, dict[str, Callable[[], object]]]:
    """Reviewed starting grids; strong samples are boundary tests, not defaults."""
    return {
        "bleedthrough": {
            "mild": lambda: BleedThrough(
                intensity_range=(0.05, 0.10), ksize=(9, 9), sigmaX=0, alpha=0.08, offsets=(8, 16), p=1
            ),
            "medium": lambda: BleedThrough(
                intensity_range=(0.10, 0.20), ksize=(13, 13), sigmaX=1, alpha=0.16, offsets=(12, 24), p=1
            ),
            "strong": lambda: BleedThrough(
                intensity_range=(0.20, 0.35), ksize=(17, 17), sigmaX=1, alpha=0.28, offsets=(16, 32), p=1
            ),
        },
        "colorpaper": {
            "mild": lambda: ColorPaper(hue_range=(30, 38), saturation_range=(4, 9), p=1),
            "medium": lambda: ColorPaper(hue_range=(28, 42), saturation_range=(10, 20), p=1),
            "strong": lambda: ColorPaper(hue_range=(24, 45), saturation_range=(22, 38), p=1),
        },
        "dirtydrum": {
            "mild": lambda: DirtyDrum(
                line_width_range=(1, 2), line_concentration=0.02, direction=-1,
                noise_intensity=0.12, noise_value=(0, 20), ksize=(3, 3), sigmaX=0, p=1
            ),
            "medium": lambda: DirtyDrum(
                line_width_range=(1, 3), line_concentration=0.06, direction=-1,
                noise_intensity=0.28, noise_value=(0, 30), ksize=(3, 3), sigmaX=0, p=1
            ),
            "strong": lambda: DirtyDrum(
                line_width_range=(2, 5), line_concentration=0.12, direction=-1,
                noise_intensity=0.48, noise_value=(0, 40), ksize=(5, 5), sigmaX=0, p=1
            ),
        },
        "dithering": {
            "mild": lambda: Dithering(dither="ordered", order=(5, 5), numba_jit=0, p=1),
            "medium": lambda: Dithering(dither="ordered", order=(3, 4), numba_jit=0, p=1),
            "strong": lambda: Dithering(
                dither="floyd-steinberg", order=(2, 3), numba_jit=0, p=1
            ),
        },
        "inkbleed": {
            "mild": lambda: InkBleed(
                intensity_range=(0.25, 0.35), kernel_size=(3, 3), severity=(0.08, 0.14), p=1
            ),
            "medium": lambda: InkBleed(
                intensity_range=(0.35, 0.50), kernel_size=(3, 3), severity=(0.16, 0.26), p=1
            ),
            "strong": lambda: InkBleed(
                intensity_range=(0.45, 0.65), kernel_size=(5, 5), severity=(0.30, 0.45), p=1
            ),
        },
        "hollow_rare": {
            "mild": lambda: Hollow(
                hollow_max_width_range=(60, 90), hollow_max_height_range=(60, 90),
                hollow_max_area_range=(500, 900), hollow_dilation_kernel_size_range=(1, 1), p=1
            ),
            "medium": lambda: Hollow(
                hollow_max_width_range=(100, 150), hollow_max_height_range=(100, 150),
                hollow_max_area_range=(1200, 2500), hollow_dilation_kernel_size_range=(1, 2), p=1
            ),
            "strong": lambda: Hollow(
                hollow_max_width_range=(150, 220), hollow_max_height_range=(150, 220),
                hollow_max_area_range=(3000, 6000), hollow_dilation_kernel_size_range=(2, 3), p=1
            ),
        },
        "letterpress": {
            "mild": lambda: Letterpress(
                n_samples=(40, 100), n_clusters=(40, 100), std_range=(300, 900),
                value_range=(225, 250), value_threshold_range=(96, 128), blur=1, p=1
            ),
            "medium": lambda: Letterpress(
                n_samples=(120, 280), n_clusters=(100, 240), std_range=(700, 1800),
                value_range=(190, 240), value_threshold_range=(96, 144), blur=1, p=1
            ),
            "strong": lambda: Letterpress(
                n_samples=(300, 650), n_clusters=(250, 550), std_range=(1500, 3500),
                value_range=(150, 225), value_threshold_range=(96, 160), blur=1, p=1
            ),
        },
        "linesdegradation": {
            "mild": lambda: LinesDegradation(
                line_gradient_range=(180, 245), line_split_probability=(0.05, 0.12),
                line_replacement_probability=(0.08, 0.15), line_replacement_thickness=(1, 1), p=1
            ),
            "medium": lambda: LinesDegradation(
                line_gradient_range=(120, 240), line_split_probability=(0.15, 0.28),
                line_replacement_probability=(0.20, 0.35), line_replacement_thickness=(1, 2), p=1
            ),
            "strong": lambda: LinesDegradation(
                line_gradient_range=(64, 220), line_split_probability=(0.30, 0.50),
                line_replacement_probability=(0.40, 0.58), line_replacement_thickness=(1, 3), p=1
            ),
        },
        "lowlightnoise": {
            "mild": lambda: LowLightNoise(
                num_photons_range=(140, 180), alpha_range=(0.90, 0.98), beta_range=(2, 8),
                gamma_range=(1.0, 1.1), bias_range=(2, 8), p=1
            ),
            "medium": lambda: LowLightNoise(
                num_photons_range=(80, 130), alpha_range=(0.78, 0.90), beta_range=(8, 18),
                gamma_range=(1.1, 1.35), bias_range=(8, 20), p=1
            ),
            "strong": lambda: LowLightNoise(
                num_photons_range=(35, 70), alpha_range=(0.62, 0.78), beta_range=(18, 32),
                gamma_range=(1.35, 1.70), bias_range=(20, 38), p=1
            ),
        },
        "noisetexturize": {
            "mild": lambda: NoiseTexturize(
                sigma_range=(1, 2), turbulence_range=(4, 6),
                texture_width_range=(300, 600), texture_height_range=(100, 250), p=1
            ),
            "medium": lambda: NoiseTexturize(
                sigma_range=(3, 5), turbulence_range=(3, 5),
                texture_width_range=(180, 450), texture_height_range=(80, 220), p=1
            ),
            "strong": lambda: NoiseTexturize(
                sigma_range=(6, 10), turbulence_range=(4, 7),
                texture_width_range=(100, 300), texture_height_range=(60, 180), p=1
            ),
        },
        "subtlenoise": {
            "mild": lambda: SubtleNoise(subtle_range=3, p=1),
            "medium": lambda: SubtleNoise(subtle_range=6, p=1),
            "strong": lambda: SubtleNoise(subtle_range=10, p=1),
        },
        "inkshifter": {
            "mild": lambda: InkShifter(
                text_shift_scale_range=(4, 7), text_shift_factor_range=(1, 1),
                text_fade_range=(0, 1), blur_kernel_size=(3, 3), blur_sigma=0,
                noise_type="random", p=1
            ),
            "medium": lambda: InkShifter(
                text_shift_scale_range=(8, 13), text_shift_factor_range=(1, 2),
                text_fade_range=(0, 1), blur_kernel_size=(3, 3), blur_sigma=0,
                noise_type="random", p=1
            ),
            "strong": lambda: InkShifter(
                text_shift_scale_range=(14, 22), text_shift_factor_range=(2, 4),
                text_fade_range=(0, 2), blur_kernel_size=(5, 5), blur_sigma=0,
                noise_type="random", p=1
            ),
        },
        "folding": {
            "mild": lambda: Folding(
                fold_count=1, fold_noise=0.01, fold_angle_range=(0, 0),
                gradient_width=(0.025, 0.045), gradient_height=(0.004, 0.008),
                backdrop_color=(255, 255, 255), p=1
            ),
            "medium": lambda: Folding(
                fold_count=1, fold_noise=0.04, fold_angle_range=(-1, 1),
                gradient_width=(0.05, 0.09), gradient_height=(0.008, 0.015),
                backdrop_color=(255, 255, 255), p=1
            ),
            "strong": lambda: Folding(
                fold_count=2, fold_noise=0.10, fold_angle_range=(-3, 3),
                gradient_width=(0.09, 0.16), gradient_height=(0.015, 0.030),
                backdrop_color=(255, 255, 255), p=1
            ),
        },
        "gaussian_blur": {
            "mild": lambda: ("gaussian_blur", 3, 0.45),
            "medium": lambda: ("gaussian_blur", 5, 0.90),
            "strong": lambda: ("gaussian_blur", 7, 1.60),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--review-width",
        type=int,
        default=1400,
        help="Downscale wide inputs before review generation; 0 keeps full resolution.",
    )
    return parser.parse_args()


def default_label_font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def apply_effect(image: np.ndarray, effect: object) -> np.ndarray:
    if isinstance(effect, tuple) and effect[0] == "gaussian_blur":
        _name, kernel, sigma = effect
        return cv2.GaussianBlur(image, (int(kernel), int(kernel)), float(sigma))
    output = effect(image.copy(), force=True)
    if not isinstance(output, np.ndarray):
        raise TypeError(f"Augraphy returned {type(output).__name__}, expected ndarray")
    return output


def effect_description(effect: object) -> str:
    if isinstance(effect, tuple):
        return f"kernel={effect[1]}, sigma={effect[2]}"
    fields = {
        key: value
        for key, value in vars(effect).items()
        if key != "p" and not key.startswith("_") and isinstance(value, (str, int, float, tuple, list))
    }
    return ", ".join(f"{key}={value}" for key, value in fields.items())


def make_contact_sheet(
    source: Path,
    source_image: np.ndarray,
    rows: list[dict[str, object]],
    out_dir: Path,
    *,
    suffix: str = "contact_sheet",
) -> Path:
    cell_w = 760
    image_h = max(160, round(cell_w * source_image.shape[0] / source_image.shape[1]))
    label_h = 68
    cell_h = image_h + label_h
    sheet = Image.new("RGB", (cell_w * 4, cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = default_label_font(20)
    detail_font = default_label_font(13)
    for row_index, row in enumerate(rows):
        for column, strength in enumerate(("original", *STRENGTHS)):
            x = column * cell_w
            y = row_index * cell_h
            path = source if strength == "original" else out_dir / str(row[f"{strength}_image"])
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_w, image_h))
                sheet.paste(image, (x, y + label_h))
            heading = str(row["augmentation"]) if strength == "original" else strength
            detail = "source" if strength == "original" else str(row[f"{strength}_description"])
            draw.text((x + 8, y + 7), heading, fill="black", font=title_font)
            draw.text((x + 8, y + 34), detail[:105], fill="#444444", font=detail_font)
    path = out_dir / f"{source.stem}__{suffix}.jpg"
    sheet.save(path, quality=92)
    return path


def main() -> None:
    args = parse_args()
    inputs = tuple(args.input) or DEFAULT_INPUTS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    factories = _factories()
    manifest: list[dict[str, object]] = []
    for source_index, source in enumerate(inputs):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read input image: {source}")
        if args.review_width > 0 and image.shape[1] > args.review_width:
            scale = args.review_width / image.shape[1]
            image = cv2.resize(
                image,
                (args.review_width, round(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        source_dir = args.out_dir / source.stem
        source_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for effect_index, (name, strengths) in enumerate(factories.items()):
            print(f"Rendering {source.stem}: {name}", flush=True)
            row: dict[str, object] = {"augmentation": name}
            for strength_index, strength in enumerate(STRENGTHS):
                sample_seed = args.seed + source_index * 10_000 + effect_index * 100 + strength_index
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                effect = strengths[strength]()
                output = apply_effect(image, effect)
                image_name = f"{name}__{strength}.jpg"
                image_path = source_dir / image_name
                cv2.imwrite(str(image_path), output, [cv2.IMWRITE_JPEG_QUALITY, 94])
                row[f"{strength}_image"] = str(image_path.relative_to(args.out_dir))
                row[f"{strength}_description"] = effect_description(effect)
                row[f"{strength}_seed"] = sample_seed
            rows.append(row)
        groups = {
            "local_ink": {
                "bleedthrough",
                "dirtydrum",
                "dithering",
                "inkbleed",
                "hollow_rare",
                "letterpress",
                "linesdegradation",
            },
            "paper_and_noise": {
                "colorpaper",
                "lowlightnoise",
                "noisetexturize",
                "subtlenoise",
            },
            "spatial_and_blur": {"inkshifter", "folding", "gaussian_blur"},
        }
        sheets = {
            group: make_contact_sheet(
                source,
                image,
                [row for row in rows if row["augmentation"] in names],
                args.out_dir,
                suffix=group,
            )
            for group, names in groups.items()
        }
        manifest.append(
            {
                "source": str(source),
                "contact_sheets": {
                    group: str(sheet.relative_to(args.out_dir)) for group, sheet in sheets.items()
                },
                "samples": rows,
            }
        )
        for sheet in sheets.values():
            print(f"Wrote {sheet}")
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
