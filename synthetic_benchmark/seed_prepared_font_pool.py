#!/usr/bin/env python3
"""Migrate a completed font manifest into the aggregate prepared-pool cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

from font_augmentation_runtime import (
    DEFAULT_PROBES,
    PREPARED_POOL_CACHE_VERSION,
    _font_group_key,
    _prepared_pool_key,
)
from font_variant_cache import DEFAULT_FONT_CACHE_URI, S3FontVariantCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_plan", type=Path)
    parser.add_argument("font_manifest", type=Path)
    parser.add_argument("--local-cache-dir", type=Path, required=True)
    parser.add_argument("--s3-cache-uri", default=DEFAULT_FONT_CACHE_URI)
    parser.add_argument("--variants-per-font", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    return parser.parse_args()


def parse_spec_label(label: str) -> dict[str, object]:
    if label.startswith("width_"):
        return {"operation": "width", "value": float(label.removeprefix("width_"))}
    if label.startswith("slant_") and label.endswith("deg"):
        return {
            "operation": "slant",
            "value": float(label.removeprefix("slant_").removesuffix("deg")),
        }
    for operation in ("vertical_stroke", "horizontal_stroke", "weight"):
        prefix = operation + "_"
        if label.startswith(prefix) and label.endswith("em"):
            return {
                "operation": operation,
                "value": float(label.removeprefix(prefix).removesuffix("em")),
            }
    raise ValueError(f"Unknown augmentation label: {label}")


def main() -> None:
    args = parse_args()
    rows = pq.read_table(
        args.render_plan,
        columns=["font_abs_path", "ttc_face_index", "basename"],
    ).to_pylist()
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["font_abs_path"]),
            str(row.get("ttc_face_index") or ""),
            str(row["basename"]),
        )
        groups.setdefault(key, []).append(row)
    pool_key, policy_key = _prepared_pool_key(
        groups,
        variants_per_font=args.variants_per_font,
        seed=args.seed,
        probes_path=args.probes,
    )

    manifest = json.loads(args.font_manifest.read_text(encoding="utf-8"))
    manifest_fonts = {
        _font_group_key(
            str(font["source_font"]),
            str(font.get("source_face_index") or ""),
            str(font["basename"]),
        ): font
        for font in manifest["fonts"]
    }
    expected = {
        _font_group_key(path, face, basename)
        for path, face, basename in groups
    }
    if set(manifest_fonts) != expected:
        missing = sorted(expected - set(manifest_fonts))
        extra = sorted(set(manifest_fonts) - expected)
        raise ValueError(f"Font manifest mismatch; missing={missing}, extra={extra}")

    pool_groups: dict[str, dict[str, object]] = {}
    for group_key, font in manifest_fonts.items():
        accepted_records = []
        for variant in font["accepted"]:
            specs = [parse_spec_label(label) for label in variant["specs"]]
            record: dict[str, object] = {
                "variant_id": variant["variant_id"],
                "specs": specs,
                "raster_qc": variant["raster_qc"],
            }
            if specs:
                s3_uri = next(
                    (
                        str(step.get("source"))
                        for step in reversed(variant["provenance"])
                        if str(step.get("source") or "").startswith("s3://")
                    ),
                    "",
                )
                if not s3_uri:
                    raise ValueError(
                        f"Variant {variant['variant_id']} has no S3 provenance"
                    )
                record["s3_uri"] = s3_uri
                record["font_filename"] = Path(urlparse(s3_uri).path).name
            else:
                record["source_path"] = str(font["source_font"])
            accepted_records.append(record)
        pool_groups[group_key] = {
            "accepted": accepted_records,
            "manifest_font": font,
        }

    cache = S3FontVariantCache(args.s3_cache_uri, args.local_cache_dir)
    uri = cache.store_prepared_pool(
        pool_key,
        {
            "cache_version": PREPARED_POOL_CACHE_VERSION,
            "policy_key": policy_key,
            "pool_key": pool_key,
            "groups": pool_groups,
        },
    )
    print(f"Prepared pool {pool_key}: {len(pool_groups)} font groups")
    print(uri)


if __name__ == "__main__":
    main()
