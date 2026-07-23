#!/usr/bin/env python3
"""Deterministic page-layout variation that preserves Tibetan shaping."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict


def _stable_seed(seed: int, *values: object) -> int:
    payload = ":".join((str(seed), *(str(value) for value in values)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def assign_page_layout_parameters(
    rows: list[dict[str, object]],
    *,
    min_line_spacing_factor: float = 1.18,
    max_line_spacing_factor: float = 1.32,
    seed: int = 13,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if min_line_spacing_factor <= 0 or max_line_spacing_factor < min_line_spacing_factor:
        raise ValueError("Line-spacing range must be positive and ordered")
    midpoint = (min_line_spacing_factor + max_line_spacing_factor) / 2
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["basename"]), str(row.get("ttc_face_index") or ""))
        ].append(row)
    sparse_image_ids: set[int] = set()
    for (basename, face_index), font_rows in groups.items():
        sparse_count = int(len(font_rows) * 0.10 + 0.5)
        ordered = sorted(
            font_rows,
            key=lambda row: _stable_seed(
                seed,
                basename,
                face_index,
                row["image_id"],
                "sparse_page",
            ),
        )
        sparse_image_ids.update(int(row["image_id"]) for row in ordered[:sparse_count])

    prepared: list[dict[str, object]] = []
    values: list[float] = []
    sparse_count = 0
    for row in rows:
        item = dict(row)
        rng = random.Random(
            _stable_seed(
                seed,
                row["basename"],
                row.get("ttc_face_index") or "",
                row["image_id"],
                "line_spacing",
            )
        )
        base_factor = rng.triangular(
            min_line_spacing_factor,
            max_line_spacing_factor,
            midpoint,
        )
        maximum_stack_codepoints = max(
            (len(stack) for stack in str(row.get("stacks") or "").split()),
            default=0,
        )
        if maximum_stack_codepoints >= 7:
            stack_spacing_extra = 0.90
        elif maximum_stack_codepoints == 6:
            stack_spacing_extra = 0.55
        elif maximum_stack_codepoints == 5:
            stack_spacing_extra = 0.30
        else:
            stack_spacing_extra = 0.0
        factor = round(
            base_factor + stack_spacing_extra,
            6,
        )
        item["layout_line_spacing_factor"] = factor
        item["layout_base_line_spacing_factor"] = round(base_factor, 6)
        item["layout_max_stack_codepoints"] = maximum_stack_codepoints
        item["layout_stack_spacing_extra"] = stack_spacing_extra
        if int(row["image_id"]) in sparse_image_ids:
            font_scale_multiplier = round(rng.uniform(1.20, 1.35), 6)
            density_tier = "sparse"
            sparse_count += 1
        else:
            font_scale_multiplier = 1.0
            density_tier = "condensed"
        item["layout_font_scale_multiplier"] = font_scale_multiplier
        item["layout_density_tier"] = density_tier
        values.append(factor)
        prepared.append(item)
    prepared.sort(key=lambda item: int(item["image_id"]))
    return prepared, {
        "seed": seed,
        "line_spacing_factor": {
            "minimum": min_line_spacing_factor,
            "maximum": max_line_spacing_factor,
            "distribution": "triangular",
            "mode": midpoint,
            "minimum_assigned": min(values) if values else None,
            "maximum_assigned": max(values) if values else None,
        },
        "collision_policy": {
            "method": "stack-height-aware leading plus LuaTeX glyph-offset safety",
            "stack_codepoint_extra": {"5": 0.30, "6": 0.55, "7_or_more": 0.90},
            "scope": "ordinary pages retain the base range; tall-stack pages get extra leading",
        },
        "font_size_policy": {
            "condensed_rate": 0.90,
            "condensed_multiplier": 1.0,
            "sparse_rate": 0.10,
            "sparse_multiplier_range": [1.20, 1.35],
            "sparse_images": sparse_count,
        },
    }
