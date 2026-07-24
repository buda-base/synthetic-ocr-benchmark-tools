#!/usr/bin/env python3
"""Create full-resolution and cropped A/B raster-naturalization examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from document_augmentation import (
    _apply_raster_naturalization,
    _naturalization_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def labelled_crop(image: np.ndarray, label: str) -> np.ndarray:
    height, width = image.shape[:2]
    crop = image[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    border = 52
    panel = cv2.copyMakeBorder(
        crop,
        border,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    cv2.putText(
        panel,
        label,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        0,
        2,
        cv2.LINE_AA,
    )
    return panel


def labelled_zoom(image: np.ndarray, label: str) -> np.ndarray:
    height, width = image.shape[:2]
    crop = image[height // 4 : 3 * height // 4, 3 * width // 8 : 5 * width // 8]
    zoom = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    panel = cv2.copyMakeBorder(
        zoom,
        72,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    cv2.putText(
        panel,
        label,
        (20, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        0,
        2,
        cv2.LINE_AA,
    )
    return panel


def labelled_detail(image: np.ndarray, label: str) -> np.ndarray:
    height, width = image.shape[:2]
    crop = image[
        height // 2 - 100 : height // 2 + 100,
        width // 2 - 120 : width // 2 + 120,
    ]
    zoom = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    panel = cv2.copyMakeBorder(
        zoom,
        64,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    cv2.putText(
        panel,
        f"{label} - 4x pixel detail",
        (16, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        0,
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    source = cv2.imread(str(args.source), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise ValueError(f"Could not read {args.source}")

    profile_a = _apply_raster_naturalization(
        source,
        _naturalization_parameters("A", 101),
    )
    profile_b = _apply_raster_naturalization(
        source,
        _naturalization_parameters("B", 202),
    )
    stronger_options = []
    for label, seed, displacement, dropout, blur, noise in (
        ("Option 1 - moderate", 303, 2.5, 0.030, 0.55, 1.2),
        ("Option 2 - strong", 404, 4.0, 0.050, 0.65, 1.6),
        ("Option 3 - very strong", 505, 6.0, 0.080, 0.75, 2.0),
    ):
        parameters = _naturalization_parameters("B", seed)
        parameters.update(
            {
                "raster_naturalization_displacement_px": displacement,
                "raster_naturalization_white_dropout_rate": dropout,
                "raster_naturalization_blur_sigma": blur,
                "raster_naturalization_luminance_noise_sigma": noise,
            }
        )
        stronger_options.append(
            (label, _apply_raster_naturalization(source, parameters))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "00_original.png": source,
        "01_naturalization_A.png": profile_a,
        "02_naturalization_B.png": profile_b,
        "03_comparison_crop.png": np.vstack(
            [
                labelled_crop(source, "Original"),
                labelled_crop(profile_a, "Naturalization A (all images)"),
                labelled_crop(profile_b, "Naturalization B (15% of images)"),
            ]
        ),
        "04_comparison_2x_zoom.png": np.vstack(
            [
                labelled_zoom(source, "Original - 2x nearest-neighbor zoom"),
                labelled_zoom(profile_a, "Naturalization A - 2x zoom"),
                labelled_zoom(profile_b, "Naturalization B - 2x zoom"),
            ]
        ),
        "05_white_dropout_4x_detail.png": np.vstack(
            [
                labelled_detail(source, "Original"),
                labelled_detail(profile_a, "Naturalization A"),
                labelled_detail(profile_b, "Naturalization B"),
            ]
        ),
        "06_stronger_options_4x_detail.png": np.vstack(
            [
                labelled_detail(source, "Original"),
                labelled_detail(profile_a, "Current A - extreme lower end"),
                *[
                    labelled_detail(image, label)
                    for label, image in stronger_options
                ],
            ]
        ),
    }
    for index, (label, image) in enumerate(stronger_options, start=1):
        outputs[f"option_{index}_{label.split(' - ')[1].replace(' ', '_')}.png"] = image
    for filename, image in outputs.items():
        if not cv2.imwrite(str(args.output_dir / filename), image):
            raise RuntimeError(f"Could not write {args.output_dir / filename}")
    print(f"Wrote {len(outputs)} review images to {args.output_dir}")


if __name__ == "__main__":
    main()
