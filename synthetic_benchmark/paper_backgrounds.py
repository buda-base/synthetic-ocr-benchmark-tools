#!/usr/bin/env python3
"""Paper-background manifest loading, caching, and image preparation."""

from __future__ import annotations

import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from PIL import Image, ImageOps
from tqdm import tqdm


DEFAULT_BACKGROUND_MANIFEST = (
    Path(__file__).resolve().parent / "data" / "paper_backgrounds.csv"
)
THREAD_LOCAL = threading.local()


def classify_luminance(mean_luminance: float, p10_luminance: float) -> str:
    if mean_luminance >= 200 and p10_luminance >= 160:
        return "light"
    if mean_luminance < 150 or p10_luminance < 40:
        return "dark"
    return "medium"


@dataclass(frozen=True)
class PaperBackground:
    background_id: str
    w_id: str
    i_id: str
    i_version: str
    filename: str
    source_s3_uri: str
    width: int
    height: int
    display_width: int
    display_height: int
    aspect_ratio: float
    exif_orientation: int | None
    pil_mode: str
    size_bytes: int
    mean_luminance: float = 255.0
    median_luminance: float = 255.0
    p10_luminance: float = 255.0
    p90_luminance: float = 255.0
    luminance_std: float = 0.0
    dark_pixel_fraction: float = 0.0
    luminance_tier: str = "light"
    luminance_stats_version: str = ""


def load_paper_backgrounds(path: Path = DEFAULT_BACKGROUND_MANIFEST) -> list[PaperBackground]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    backgrounds = [
        PaperBackground(
            background_id=row["background_id"],
            w_id=row["w_id"],
            i_id=row["i_id"],
            i_version=row["i_version"],
            filename=row["filename"],
            source_s3_uri=row["source_s3_uri"],
            width=int(row["width"]),
            height=int(row["height"]),
            display_width=int(row["display_width"]),
            display_height=int(row["display_height"]),
            aspect_ratio=float(row["aspect_ratio"]),
            exif_orientation=(
                int(row["exif_orientation"]) if row["exif_orientation"] else None
            ),
            pil_mode=row["pil_mode"],
            size_bytes=int(row["size_bytes"]),
            mean_luminance=float(row.get("mean_luminance") or 255),
            median_luminance=float(row.get("median_luminance") or 255),
            p10_luminance=float(row.get("p10_luminance") or 255),
            p90_luminance=float(row.get("p90_luminance") or 255),
            luminance_std=float(row.get("luminance_std") or 0),
            dark_pixel_fraction=float(row.get("dark_pixel_fraction") or 0),
            luminance_tier=str(row.get("luminance_tier") or "light"),
            luminance_stats_version=str(row.get("luminance_stats_version") or ""),
        )
        for row in rows
    ]
    if not backgrounds:
        raise ValueError(f"No paper backgrounds found in {path}")
    unsupported = sorted({item.pil_mode for item in backgrounds} - {"1", "L", "RGB"})
    if unsupported:
        raise ValueError(f"Unsupported paper background modes in {path}: {unsupported}")
    return backgrounds


def _s3_client():
    client = getattr(THREAD_LOCAL, "s3_client", None)
    if client is None:
        client = boto3.client("s3")
        THREAD_LOCAL.s3_client = client
    return client


def _local_review_name(background: PaperBackground) -> str:
    return f"{background.w_id}__{background.i_id}__{background.filename}"


def _download_background(background: PaperBackground, destination: Path) -> None:
    parsed = urlparse(background.source_s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid paper background URI: {background.source_s3_uri}")
    temporary = destination.with_name(destination.name + ".part")
    payload = _s3_client().get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )["Body"].read()
    temporary.write_bytes(payload)
    temporary.replace(destination)


def prepare_paper_background_cache(
    rows: list[dict[str, object]],
    backgrounds: list[PaperBackground],
    *,
    cache_dir: Path,
    local_review_dir: Path | None = None,
    workers: int = 16,
) -> dict[str, Path]:
    """Resolve every assigned background to a local path and annotate runtime rows."""
    by_id = {background.background_id: background for background in backgrounds}
    requested_ids = sorted(
        {
            str(row.get(field) or "")
            for row in rows
            for field in (
                "document_augmentation_background_id",
                "_document_augmentation_fallback_background_id",
            )
            if row.get(field)
        }
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    pending: list[tuple[PaperBackground, Path]] = []
    for background_id in requested_ids:
        background = by_id.get(background_id)
        if background is None:
            raise ValueError(f"Assigned paper background is absent from manifest: {background_id}")
        local_path = (
            local_review_dir / _local_review_name(background)
            if local_review_dir is not None
            else None
        )
        if local_path is not None and local_path.exists():
            resolved[background_id] = local_path
            continue
        suffix = Path(background.filename).suffix.lower() or ".img"
        cache_path = cache_dir / f"{background.background_id}{suffix}"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[background_id] = cache_path
        else:
            pending.append((background, cache_path))

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda item: _download_background(item[0], item[1]),
                        pending,
                    ),
                    total=len(pending),
                    desc="Cache paper backgrounds",
                    unit="image",
                )
            )
        resolved.update(
            {background.background_id: path for background, path in pending}
        )

    for row in rows:
        background_id = str(row.get("document_augmentation_background_id") or "")
        if background_id:
            row["_document_augmentation_background_path"] = str(resolved[background_id])
        fallback_id = str(row.get("_document_augmentation_fallback_background_id") or "")
        if fallback_id:
            row["_document_augmentation_fallback_background_path"] = str(
                resolved[fallback_id]
            )
    return resolved


def crop_resize_background(
    path: Path,
    *,
    target_size: tuple[int, int],
    mode: str,
) -> Image.Image:
    """EXIF-orient and stretch a background to the exact output dimensions."""
    target_width, target_height = target_size
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if mode == "1":
            image = image.convert("1")
        else:
            image = image.convert(mode)
        resampling = Image.Resampling.NEAREST if mode == "1" else Image.Resampling.LANCZOS
        return image.resize((target_width, target_height), resampling)
