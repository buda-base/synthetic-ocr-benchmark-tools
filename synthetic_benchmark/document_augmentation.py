#!/usr/bin/env python3
"""Deterministic production policy for document-level image augmentation."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections import Counter, defaultdict
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
    NoiseTexturize,
    SubtleNoise,
)


LOCAL_EFFECT_WEIGHTS = (
    ("subtlenoise", 25.00),
    ("noisetexturize", 19.00),
    ("inkbleed", 17.00),
    ("letterpress", 14.00),
    ("bleedthrough", 11.00),
    ("dirtydrum", 7.00),
    ("dithering", 3.75),
    ("colorpaper", 3.00),
    ("hollow_rare", 0.25),
)
STRENGTH_WEIGHTS = {
    "subtlenoise": (("mild", 0.60), ("medium", 0.35), ("strong", 0.05)),
    "noisetexturize": (("mild", 0.60), ("medium", 0.35), ("strong", 0.05)),
    "inkbleed": (("mild", 0.55), ("medium", 0.40), ("strong", 0.05)),
    "letterpress": (("mild", 0.55), ("medium", 0.40), ("strong", 0.05)),
    "bleedthrough": (("mild", 0.60), ("medium", 0.35), ("strong", 0.05)),
    "dirtydrum": (("mild", 0.75), ("medium", 0.25)),
    "dithering": (("mild", 0.70), ("medium", 0.30)),
    "colorpaper": (("mild", 0.70), ("medium", 0.30)),
    "hollow_rare": (("mild", 1.0),),
}
DOCUMENT_AUGMENTATION_LOCK = threading.Lock()
TPS_X_NORMALIZED = np.asarray(
    [0.0833757316, 0.2767441956, 0.5082266718, 0.7219677239, 0.9150948801],
    dtype=np.float64,
)
TPS_SHAPE_COVARIANCE = np.asarray(
    [
        [0.000427648, 0.000063214, -0.000186558, -0.000201829, -0.000100111],
        [0.000063214, 0.000144754, 0.000081550, -0.000061410, -0.000226010],
        [-0.000186558, 0.000081550, 0.000223369, 0.000074149, -0.000190991],
        [-0.000201829, -0.000061410, 0.000074149, 0.000127497, 0.000062748],
        [-0.000100111, -0.000226010, -0.000190991, 0.000062748, 0.000455582],
    ],
    dtype=np.float64,
)
TPS_REGULARIZATION = 0.5


def _stable_seed(seed: int, *values: object) -> int:
    payload = ":".join((str(seed), *(str(value) for value in values)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _weighted_choice(rng: random.Random, values) -> str:
    draw = rng.random() * sum(weight for _name, weight in values)
    cumulative = 0.0
    for name, weight in values:
        cumulative += weight
        if draw <= cumulative:
            return name
    return values[-1][0]


def _balanced_selection(
    rows: list[dict[str, object]],
    rate: float,
    *,
    seed: int,
    label: str,
) -> set[int]:
    count = min(len(rows), max(0, round(len(rows) * rate)))
    indices = list(range(len(rows)))
    random.Random(_stable_seed(seed, label)).shuffle(indices)
    return set(indices[:count])


def _balanced_subselection(
    rows: list[dict[str, object]],
    candidates: set[int],
    rate_of_all: float,
    *,
    seed: int,
    label: str,
) -> set[int]:
    count = min(len(candidates), max(0, round(len(rows) * rate_of_all)))
    indices = sorted(candidates)
    random.Random(_stable_seed(seed, label)).shuffle(indices)
    return set(indices[:count])


def _sample_rotation_angle(rng: random.Random, strength: str) -> float:
    if strength == "high":
        magnitude = rng.triangular(0.90, 2.40, 1.15)
        return magnitude if rng.random() < 0.5 else -magnitude
    while True:
        angle = rng.gauss(0.0, 0.55)
        if -0.90 <= angle <= 0.90:
            return angle


def _sample_tps_parameters(seed: int, strength: str) -> tuple[float, list[float]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    offsets = np_rng.multivariate_normal(
        np.zeros(len(TPS_X_NORMALIZED)),
        TPS_SHAPE_COVARIANCE,
        check_valid="raise",
    )
    offsets -= offsets.mean()
    maximum = float(np.max(np.abs(offsets)))
    if maximum < 1e-9:
        offsets = np.asarray([-1.0, 0.3, 0.8, 0.2, -0.3], dtype=np.float64)
        maximum = 1.0
    if strength == "high":
        target_maximum = rng.triangular(0.0304, 0.0526, 0.0408)
    else:
        target_maximum = rng.triangular(0.0117, 0.0304, 0.0217)
    offsets *= target_maximum / maximum
    y_normalized = 0.10 + 0.80 * rng.betavariate(2.3, 2.3)
    return y_normalized, offsets.tolist()


def assign_document_augmentations(
    rows: list[dict[str, object]],
    *,
    local_rate: float = 0.90,
    inkshifter_rate: float = 0.08,
    folding_rate: float = 0.03,
    blur_rate: float = 0.10,
    rotation_rate: float = 0.70,
    rotation_high_rate: float = 0.10,
    tps_rate: float = 0.30,
    tps_high_rate: float = 0.10,
    seed: int = 13,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Assign exact per-font counts and deterministic effect choices."""
    for name, value in (
        ("local_rate", local_rate),
        ("inkshifter_rate", inkshifter_rate),
        ("folding_rate", folding_rate),
        ("blur_rate", blur_rate),
        ("rotation_rate", rotation_rate),
        ("rotation_high_rate", rotation_high_rate),
        ("tps_rate", tps_rate),
        ("tps_high_rate", tps_high_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if rotation_high_rate > rotation_rate:
        raise ValueError("rotation_high_rate cannot exceed rotation_rate")
    if tps_high_rate > tps_rate:
        raise ValueError("tps_high_rate cannot exceed tps_rate")
    if tps_rate > rotation_rate:
        raise ValueError("tps_rate cannot exceed rotation_rate because TPS is assigned to rotated pages")

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["basename"]), str(row.get("ttc_face_index") or ""))].append(row)

    prepared: list[dict[str, object]] = []
    font_summaries = []
    for (basename, face_index), font_rows in sorted(groups.items()):
        font_rows = sorted(font_rows, key=lambda item: int(item["image_id"]))
        font_seed = _stable_seed(seed, basename, face_index)
        local_indices = _balanced_selection(font_rows, local_rate, seed=font_seed, label="local")
        ink_indices = _balanced_selection(font_rows, inkshifter_rate, seed=font_seed, label="inkshifter")
        folding_candidates = [i for i in range(len(font_rows)) if i not in ink_indices]
        folding_selected = _balanced_selection(
            [font_rows[i] for i in folding_candidates],
            min(1.0, folding_rate * len(font_rows) / max(1, len(folding_candidates))),
            seed=font_seed,
            label="folding",
        )
        folding_indices = {folding_candidates[i] for i in folding_selected}
        blur_indices = _balanced_selection(font_rows, blur_rate, seed=font_seed, label="blur")
        rotation_indices = _balanced_selection(
            font_rows, rotation_rate, seed=font_seed, label="rotation"
        )
        rotation_high_indices = _balanced_subselection(
            font_rows,
            rotation_indices,
            rotation_high_rate,
            seed=font_seed,
            label="rotation_high",
        )
        tps_indices = _balanced_subselection(
            font_rows,
            rotation_indices,
            tps_rate,
            seed=font_seed,
            label="tps",
        )
        tps_high_indices = _balanced_subselection(
            font_rows,
            tps_indices,
            tps_high_rate,
            seed=font_seed,
            label="tps_high",
        )
        local_counts: Counter[str] = Counter()

        for index, row in enumerate(font_rows):
            item = dict(row)
            item["document_augmentation_enabled"] = 1
            item["document_augmentation_seed"] = _stable_seed(seed, basename, face_index, row["image_id"])
            if index in local_indices:
                rng = random.Random(_stable_seed(font_seed, row["image_id"], "local"))
                effect = _weighted_choice(rng, LOCAL_EFFECT_WEIGHTS)
                strength = _weighted_choice(rng, STRENGTH_WEIGHTS[effect])
                item["document_augmentation_local"] = effect
                item["document_augmentation_local_strength"] = strength
                local_counts[f"{effect}:{strength}"] += 1
            else:
                item["document_augmentation_local"] = ""
                item["document_augmentation_local_strength"] = ""

            if index in ink_indices:
                item["document_augmentation_spatial"] = "inkshifter"
                item["document_augmentation_spatial_strength"] = (
                    "mild"
                    if random.Random(_stable_seed(font_seed, row["image_id"], "ink")).random() < 0.75
                    else "medium"
                )
            elif index in folding_indices:
                item["document_augmentation_spatial"] = "folding"
                item["document_augmentation_spatial_strength"] = (
                    "mild"
                    if random.Random(_stable_seed(font_seed, row["image_id"], "fold")).random() < 0.80
                    else "medium"
                )
            else:
                item["document_augmentation_spatial"] = ""
                item["document_augmentation_spatial_strength"] = ""

            if index in blur_indices:
                item["document_augmentation_blur"] = (
                    "mild"
                    if random.Random(_stable_seed(font_seed, row["image_id"], "blur")).random() < 0.65
                    else "medium"
                )
            else:
                item["document_augmentation_blur"] = ""

            if index in rotation_indices:
                rotation_strength = "high" if index in rotation_high_indices else "typical"
                rotation_rng = random.Random(
                    _stable_seed(font_seed, row["image_id"], "rotation_parameters")
                )
                item["document_augmentation_rotation_strength"] = rotation_strength
                item["document_augmentation_rotation_deg"] = round(
                    _sample_rotation_angle(rotation_rng, rotation_strength), 6
                )
            else:
                item["document_augmentation_rotation_strength"] = ""
                item["document_augmentation_rotation_deg"] = ""

            if index in tps_indices:
                tps_strength = "high" if index in tps_high_indices else "typical"
                tps_y, tps_offsets = _sample_tps_parameters(
                    _stable_seed(font_seed, row["image_id"], "tps_parameters"),
                    tps_strength,
                )
                item["document_augmentation_tps_strength"] = tps_strength
                item["document_augmentation_tps_y_norm"] = round(tps_y, 6)
                item["document_augmentation_tps_offsets_height"] = "|".join(
                    f"{offset:.8f}" for offset in tps_offsets
                )
            else:
                item["document_augmentation_tps_strength"] = ""
                item["document_augmentation_tps_y_norm"] = ""
                item["document_augmentation_tps_offsets_height"] = ""
            prepared.append(item)

        font_summaries.append(
            {
                "basename": basename,
                "source_face_index": face_index,
                "images": len(font_rows),
                "local_images": len(local_indices),
                "local_rate": len(local_indices) / len(font_rows),
                "inkshifter_images": len(ink_indices),
                "folding_images": len(folding_indices),
                "blur_images": len(blur_indices),
                "rotation_images": len(rotation_indices),
                "rotation_high_images": len(rotation_high_indices),
                "tps_images": len(tps_indices),
                "tps_high_images": len(tps_high_indices),
                "local_distribution": dict(sorted(local_counts.items())),
            }
        )
    prepared.sort(key=lambda item: int(item["image_id"]))
    return prepared, {
        "seed": seed,
        "requested_rates": {
            "local": local_rate,
            "inkshifter": inkshifter_rate,
            "folding": folding_rate,
            "blur": blur_rate,
            "rotation": rotation_rate,
            "rotation_high": rotation_high_rate,
            "tps": tps_rate,
            "tps_high": tps_high_rate,
        },
        "local_effect_weights": dict(LOCAL_EFFECT_WEIGHTS),
        "geometric_policy": {
            "rate_interpretation": "high rates are shares of all images, not of augmented images",
            "tps_is_subset_of_rotation": True,
            "rotation_typical_range_deg": [-0.90, 0.90],
            "rotation_high_magnitude_range_deg": [0.90, 2.40],
            "tps_typical_max_displacement_height": [0.0117, 0.0304],
            "tps_high_max_displacement_height": [0.0304, 0.0526],
            "tps_x_normalized": TPS_X_NORMALIZED.tolist(),
            "tps_regularization": TPS_REGULARIZATION,
            "tps_corner_anchors": "four fixed identity image corners",
        },
        "fonts": font_summaries,
    }


def _effect(name: str, strength: str):
    factories = {
        ("subtlenoise", "mild"): lambda: SubtleNoise(subtle_range=3, p=1),
        ("subtlenoise", "medium"): lambda: SubtleNoise(subtle_range=6, p=1),
        ("subtlenoise", "strong"): lambda: SubtleNoise(subtle_range=10, p=1),
        ("noisetexturize", "mild"): lambda: NoiseTexturize(
            sigma_range=(1, 2), turbulence_range=(4, 6),
            texture_width_range=(300, 600), texture_height_range=(100, 250), p=1
        ),
        ("noisetexturize", "medium"): lambda: NoiseTexturize(
            sigma_range=(3, 5), turbulence_range=(3, 5),
            texture_width_range=(180, 450), texture_height_range=(80, 220), p=1
        ),
        ("noisetexturize", "strong"): lambda: NoiseTexturize(
            sigma_range=(6, 8), turbulence_range=(2, 4),
            texture_width_range=(120, 300), texture_height_range=(70, 180), p=1
        ),
        ("inkbleed", "mild"): lambda: InkBleed(
            intensity_range=(0.25, 0.35), kernel_size=(3, 3), severity=(0.08, 0.14), p=1
        ),
        ("inkbleed", "medium"): lambda: InkBleed(
            intensity_range=(0.35, 0.50), kernel_size=(3, 3), severity=(0.16, 0.26), p=1
        ),
        ("inkbleed", "strong"): lambda: InkBleed(
            intensity_range=(0.45, 0.60), kernel_size=(5, 5), severity=(0.28, 0.38), p=1
        ),
        ("letterpress", "mild"): lambda: Letterpress(
            n_samples=(40, 100), n_clusters=(40, 100), std_range=(300, 900),
            value_range=(225, 250), value_threshold_range=(96, 128), blur=1, p=1
        ),
        ("letterpress", "medium"): lambda: Letterpress(
            n_samples=(120, 280), n_clusters=(100, 240), std_range=(700, 1800),
            value_range=(190, 240), value_threshold_range=(96, 144), blur=1, p=1
        ),
        ("letterpress", "strong"): lambda: Letterpress(
            n_samples=(250, 500), n_clusters=(220, 450), std_range=(1200, 2800),
            value_range=(165, 225), value_threshold_range=(96, 152), blur=1, p=1
        ),
        ("bleedthrough", "mild"): lambda: BleedThrough(
            intensity_range=(0.05, 0.10), ksize=(9, 9), sigmaX=0, alpha=0.08, offsets=(8, 16), p=1
        ),
        ("bleedthrough", "medium"): lambda: BleedThrough(
            intensity_range=(0.10, 0.20), ksize=(13, 13), sigmaX=1, alpha=0.16, offsets=(12, 24), p=1
        ),
        ("bleedthrough", "strong"): lambda: BleedThrough(
            intensity_range=(0.18, 0.30), ksize=(17, 17), sigmaX=1, alpha=0.24, offsets=(16, 32), p=1
        ),
        ("dirtydrum", "mild"): lambda: DirtyDrum(
            line_width_range=(1, 2), line_concentration=0.02, direction=-1,
            noise_intensity=0.12, noise_value=(0, 20), ksize=(3, 3), sigmaX=0, p=1
        ),
        ("dirtydrum", "medium"): lambda: DirtyDrum(
            line_width_range=(1, 3), line_concentration=0.045, direction=-1,
            noise_intensity=0.22, noise_value=(0, 28), ksize=(3, 3), sigmaX=0, p=1
        ),
        ("dithering", "mild"): lambda: Dithering(
            dither="ordered", order=(5, 5), numba_jit=0, p=1
        ),
        ("dithering", "medium"): lambda: Dithering(
            dither="ordered", order=(3, 4), numba_jit=0, p=1
        ),
        ("colorpaper", "mild"): lambda: ColorPaper(
            hue_range=(30, 38), saturation_range=(4, 9), p=1
        ),
        ("colorpaper", "medium"): lambda: ColorPaper(
            hue_range=(28, 42), saturation_range=(10, 20), p=1
        ),
        ("hollow_rare", "mild"): lambda: Hollow(
            hollow_max_width_range=(35, 60), hollow_max_height_range=(35, 60),
            hollow_max_area_range=(200, 450), hollow_dilation_kernel_size_range=(1, 1), p=1
        ),
        ("inkshifter", "mild"): lambda: InkShifter(
            text_shift_scale_range=(4, 7), text_shift_factor_range=(1, 1),
            text_fade_range=(0, 1), blur_kernel_size=(3, 3), blur_sigma=0,
            noise_type="random", p=1
        ),
        ("inkshifter", "medium"): lambda: InkShifter(
            text_shift_scale_range=(8, 13), text_shift_factor_range=(1, 2),
            text_fade_range=(0, 1), blur_kernel_size=(3, 3), blur_sigma=0,
            noise_type="random", p=1
        ),
        ("folding", "mild"): lambda: Folding(
            fold_count=1, fold_noise=0.01, fold_angle_range=(0, 0),
            gradient_width=(0.025, 0.045), gradient_height=(0.004, 0.008),
            backdrop_color=(255, 255, 255), p=1
        ),
        ("folding", "medium"): lambda: Folding(
            fold_count=1, fold_noise=0.04, fold_angle_range=(-1, 1),
            gradient_width=(0.05, 0.09), gradient_height=(0.008, 0.015),
            backdrop_color=(255, 255, 255), p=1
        ),
    }
    try:
        return factories[(name, strength)]()
    except KeyError as exc:
        raise ValueError(f"Unsupported document augmentation: {name}/{strength}") from exc


def _apply_vertical_tps(
    image: np.ndarray,
    *,
    y_normalized: float,
    offsets_height: list[float],
) -> np.ndarray:
    if len(offsets_height) != len(TPS_X_NORMALIZED):
        raise ValueError(
            f"Expected {len(TPS_X_NORMALIZED)} TPS offsets, got {len(offsets_height)}"
        )
    height, width = image.shape[:2]
    ideal = np.column_stack(
        (
            TPS_X_NORMALIZED * (width - 1),
            np.full(len(TPS_X_NORMALIZED), y_normalized * (height - 1)),
        )
    )
    observed = ideal.copy()
    observed[:, 1] += np.asarray(offsets_height, dtype=np.float64) * height
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype=np.float64,
    )
    ideal = np.concatenate((ideal, corners)).astype(np.float32)[None, :, :]
    observed = np.concatenate((observed, corners)).astype(np.float32)[None, :, :]
    matches = [cv2.DMatch(index, index, 0) for index in range(ideal.shape[1])]
    transformer = cv2.createThinPlateSplineShapeTransformer(TPS_REGULARIZATION)
    transformer.estimateTransformation(observed, ideal, matches)
    return transformer.warpImage(
        image,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def apply_document_augmentations(
    source: Path,
    destination: Path,
    row: dict[str, object],
    *,
    jpeg_quality: int,
) -> None:
    """Apply a row's deterministic policy and save a color JPEG."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read rendered page: {source}")
    sample_seed = int(row["document_augmentation_seed"])
    # Augraphy uses module-global Python and NumPy RNGs. Serialize this short
    # post-processing stage so --jobs N remains reproducible across threads.
    with DOCUMENT_AUGMENTATION_LOCK:
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32 - 1))

        for name_key, strength_key in (
            ("document_augmentation_local", "document_augmentation_local_strength"),
            ("document_augmentation_spatial", "document_augmentation_spatial_strength"),
        ):
            name = str(row.get(name_key) or "")
            strength = str(row.get(strength_key) or "")
            if name:
                image = _effect(name, strength)(image, force=True)

        tps_strength = str(row.get("document_augmentation_tps_strength") or "")
        if tps_strength:
            offsets = [
                float(value)
                for value in str(row["document_augmentation_tps_offsets_height"]).split("|")
            ]
            image = _apply_vertical_tps(
                image,
                y_normalized=float(row["document_augmentation_tps_y_norm"]),
                offsets_height=offsets,
            )

        rotation = row.get("document_augmentation_rotation_deg")
        if rotation not in (None, ""):
            image = _apply_rotation(image, float(rotation))

        blur = str(row.get("document_augmentation_blur") or "")
        if blur:
            kernel, sigma = (3, 0.45) if blur == "mild" else (5, 0.90)
            image = cv2.GaussianBlur(image, (kernel, kernel), sigma)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    ):
        raise RuntimeError(f"Could not write augmented JPEG: {destination}")


def write_document_augmentation_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
