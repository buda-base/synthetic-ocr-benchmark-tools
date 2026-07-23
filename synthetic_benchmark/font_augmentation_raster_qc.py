#!/usr/bin/env python3
"""Raster-level checks for destructive font augmentation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


RASTER_QC_VERSION = 2
COMPONENT_SIZE_BINS = (4, 8, 16, 32, 64, 128, 256)


def _component_sizes(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.asarray([], dtype=np.int64)
    return np.bincount(labels.ravel())[1:]


def _component_count(mask: np.ndarray, *, enclosed_only: bool) -> int:
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel())
    eligible = sizes >= 4
    eligible[0] = False
    if enclosed_only:
        border_labels = np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        eligible[border_labels] = False
    return int(eligible.sum())


def raster_metrics(path: Path, *, threshold: int = 128) -> dict[str, object]:
    pixels = np.asarray(Image.open(path).convert("L"))
    ink = pixels < threshold
    ys, xs = np.where(ink)
    if not len(xs):
        return {
            "ink_pixels": 0,
            "ink_density": 0.0,
            "ink_components": 0,
            "enclosed_holes": 0,
            "content_width": 0,
            "content_height": 0,
            "component_size_p10": 0.0,
            "small_component_counts": {
                str(limit): 0 for limit in COMPONENT_SIZE_BINS
            },
        }
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    component_sizes = _component_sizes(ink)
    component_size_p10 = float(np.quantile(component_sizes, 0.10))
    return {
        "ink_pixels": int(ink.sum()),
        "ink_density": float(ink.mean()),
        "ink_components": _component_count(ink, enclosed_only=False),
        "enclosed_holes": _component_count(~ink, enclosed_only=True),
        "content_width": int(ink.shape[1]),
        "content_height": int(ink.shape[0]),
        "component_size_p10": round(component_size_p10, 3),
        "small_component_counts": {
            str(limit): int((component_sizes <= limit).sum())
            for limit in COMPONENT_SIZE_BINS
        },
    }


def compare_rasters(
    baseline_path: Path,
    variant_path: Path,
    *,
    min_density_ratio: float = 0.60,
    min_counter_retention: float = 0.80,
    max_component_ratio: float = 1.45,
) -> dict[str, object]:
    baseline = raster_metrics(baseline_path)
    variant = raster_metrics(variant_path)

    def ratio(key: str, default: float = 1.0) -> float:
        denominator = float(baseline[key])
        return float(variant[key]) / denominator if denominator else default

    density_ratio = ratio("ink_density")
    ink_ratio = ratio("ink_pixels")
    component_ratio = ratio("ink_components")
    counter_retention = ratio("enclosed_holes")
    baseline_component_p10 = float(baseline.get("component_size_p10") or 0.0)
    eligible_limits = [
        limit
        for limit in COMPONENT_SIZE_BINS
        if limit <= max(4.0, baseline_component_p10 * 0.8)
    ]
    tiny_component_threshold = max(eligible_limits, default=4)
    baseline_tiny_components = int(
        baseline["small_component_counts"][str(tiny_component_threshold)]
    )
    variant_tiny_components = int(
        variant["small_component_counts"][str(tiny_component_threshold)]
    )
    tiny_component_excess = variant_tiny_components - baseline_tiny_components
    warnings = []
    if density_ratio < min_density_ratio:
        warnings.append("severe_ink_loss")
    if int(baseline["enclosed_holes"]) >= 5 and counter_retention < min_counter_retention:
        warnings.append("counter_loss_or_black_fill")
    if int(baseline["ink_components"]) >= 10 and component_ratio > max_component_ratio:
        warnings.append("stroke_fragmentation")
    if (
        tiny_component_excess >= 4
        and variant_tiny_components
        > max(baseline_tiny_components + 3, baseline_tiny_components * 1.5)
    ):
        warnings.append("tiny_component_excess")
    return {
        "qc_version": RASTER_QC_VERSION,
        "automatic_pass": not warnings,
        "warnings": warnings,
        "ink_ratio": round(ink_ratio, 4),
        "ink_density_ratio": round(density_ratio, 4),
        "component_ratio": round(component_ratio, 4),
        "counter_retention": round(counter_retention, 4),
        "tiny_component_threshold": tiny_component_threshold,
        "baseline_tiny_components": baseline_tiny_components,
        "variant_tiny_components": variant_tiny_components,
        "tiny_component_excess": tiny_component_excess,
        "baseline": baseline,
        "variant": variant,
    }
