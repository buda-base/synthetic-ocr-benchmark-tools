#!/usr/bin/env python3
"""Deterministic output-size and JPEG-quality assignment."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict


def _stable_seed(seed: int, *values: object) -> int:
    payload = ":".join((str(seed), *(str(value) for value in values)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def assign_image_output_parameters(
    rows: list[dict[str, object]],
    *,
    min_width_px: int = 1800,
    max_width_px: int = 3500,
    jpeg_quality: int = 85,
    low_jpeg_quality: int = 65,
    low_jpeg_quality_rate: float = 0.10,
    seed: int = 13,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if min_width_px <= 0 or max_width_px < min_width_px:
        raise ValueError("Image width range must be positive and ordered")
    for name, value in (
        ("jpeg_quality", jpeg_quality),
        ("low_jpeg_quality", low_jpeg_quality),
    ):
        if not 1 <= value <= 100:
            raise ValueError(f"{name} must be in [1, 100], got {value}")
    if not 0.0 <= low_jpeg_quality_rate <= 1.0:
        raise ValueError("low_jpeg_quality_rate must be in [0, 1]")

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["basename"]), str(row.get("ttc_face_index") or ""))].append(row)

    prepared: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for (basename, face_index), font_rows in sorted(groups.items()):
        font_rows = sorted(font_rows, key=lambda item: int(item["image_id"]))
        font_seed = _stable_seed(seed, basename, face_index)
        jpeg_indices = [
            index
            for index, row in enumerate(font_rows)
            if str(row.get("document_augmentation_output_extension") or ".jpg") == ".jpg"
        ]
        low_count = min(
            len(jpeg_indices),
            max(0, round(len(jpeg_indices) * low_jpeg_quality_rate)),
        )
        random.Random(_stable_seed(font_seed, "low_jpeg_quality")).shuffle(jpeg_indices)
        low_quality_indices = set(jpeg_indices[:low_count])
        widths: Counter[int] = Counter()
        quality_counts: Counter[int] = Counter()
        for index, row in enumerate(font_rows):
            item = dict(row)
            width_rng = random.Random(
                _stable_seed(font_seed, row["image_id"], "output_width")
            )
            width = width_rng.randint(min_width_px, max_width_px)
            quality = low_jpeg_quality if index in low_quality_indices else jpeg_quality
            item["output_width_px"] = width
            item["output_jpeg_quality"] = quality
            widths[width] += 1
            if str(item.get("document_augmentation_output_extension") or ".jpg") == ".jpg":
                quality_counts[quality] += 1
            prepared.append(item)
        summaries.append(
            {
                "basename": basename,
                "source_face_index": face_index,
                "images": len(font_rows),
                "jpeg_images": sum(quality_counts.values()),
                "low_quality_jpeg_images": quality_counts[low_jpeg_quality],
                "minimum_assigned_width_px": min(widths) if widths else None,
                "maximum_assigned_width_px": max(widths) if widths else None,
            }
        )

    prepared.sort(key=lambda item: int(item["image_id"]))
    return prepared, {
        "seed": seed,
        "width_px": {"minimum": min_width_px, "maximum": max_width_px},
        "jpeg_quality": {
            "default": jpeg_quality,
            "low": low_jpeg_quality,
            "low_rate": low_jpeg_quality_rate,
        },
        "fonts": summaries,
    }
