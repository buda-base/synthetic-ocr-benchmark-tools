#!/usr/bin/env python3
"""Build a split-aware, difficulty-balanced synthetic benchmark render plan."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from shorthand_aug import mode_for_script
from synthetic_common import (
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_CHUNKS_PARQUET,
    DEFAULT_FONTS_CSV,
    DEFAULT_RENDER_PLAN,
    DEFAULT_SCRIPTS_CSV,
    FontCatalogRow,
    load_font_catalog,
    load_supported_stacks,
    stack_difficulty_score,
    tokenize_tibetan_stacks,
)

GROUP_SIZE = 1000
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLITS = tuple(SPLIT_RATIOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced synthetic benchmark render plan.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PARQUET)
    parser.add_argument("--support-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_RENDER_PLAN)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--scripts-csv", type=Path, default=DEFAULT_SCRIPTS_CSV)
    parser.add_argument("--fonts-csv", type=Path, default=DEFAULT_FONTS_CSV)
    parser.add_argument("--target-images", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-chunk-reuse-ratio", type=float, default=0.10)
    parser.add_argument(
        "--oversample-ume-dense",
        action="store_true",
        help=(
            "For ume even image_id rows, concatenate one extra coverage-compatible chunk "
            "(~2x source text) so dense shorthand contraction still fills a pecha page."
        ),
    )
    parser.add_argument(
        "--images-per-font",
        type=int,
        default=None,
        help=(
            "If set, ignore --target-images quotas and allocate exactly this many images "
            "per eligible font (useful for small with/without shorthand smoke runs)."
        ),
    )
    parser.add_argument(
        "--pair-shorthand-modes",
        action="store_true",
        help=(
            "After planning, assign each font consecutive even/odd image_ids and pin "
            "shorthand_mode: even=with (uchen sparse / ume dense), odd=none. "
            "Intended for --images-per-font 2 smoke plans."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class FontPlanInfo:
    font: FontCatalogRow
    split: str


def normalized_script(value: str) -> str:
    text = value.strip().lower()
    if "ume" in text or "transitional" in text:
        return "ume"
    if "uchen" in text:
        return "uchen"
    return "uchen"


def font_name(font: FontCatalogRow) -> str:
    return font.ps_name or font.basename or font.font_file


def load_chunks(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(path)
    chunks = table.to_pylist()
    for chunk in tqdm(chunks, desc="Load chunk difficulty", unit="chunk"):
        stack_text = chunk.get("stacks") or ""
        chunk["_stack_set"] = frozenset(str(stack_text).split())
        if chunk.get("stack_difficulty_score") is None:
            chunk["stack_difficulty_score"] = stack_difficulty_score(str(chunk.get("text") or ""))
        else:
            chunk["stack_difficulty_score"] = float(chunk["stack_difficulty_score"])
    return chunks


def category_quotas(categories: list[str], target: int) -> dict[str, int]:
    base, rem = divmod(target, len(categories))
    return {category: base + (1 if i < rem else 0) for i, category in enumerate(categories)}


def font_supports_chunk(font: FontCatalogRow, chunk: dict[str, object], supported: dict[str, set[str]]) -> bool:
    stacks = chunk["_stack_set"]
    if not stacks:
        return False
    font_supported = supported.get(font.basename)
    if not font_supported:
        return False
    return stacks.issubset(font_supported)


def chunk_difficulty_tier(chunk: dict[str, object]) -> str:
    return "difficult" if float(chunk["stack_difficulty_score"]) > 0 else "normal"


def build_difficulty_pools(
    chunks: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    normal_chunks = [
        chunk for chunk in chunks if chunk_difficulty_tier(chunk) == "normal"
    ]
    return {
        "normal": sorted(
            normal_chunks,
            key=lambda row: (
                int(row.get("unique_stack_count") or 0),
                int(row.get("stack_count") or 0),
                str(row["chunk_id"]),
            ),
        ),
        "relative_difficult": sorted(
            normal_chunks,
            key=lambda row: (
                -max(
                    (len(stack) for stack in str(row.get("stacks") or "").split()),
                    default=0,
                ),
                -int(row.get("unique_stack_count") or 0),
                -int(row.get("stack_count") or 0),
                str(row["chunk_id"]),
            ),
        ),
        "difficult": sorted(
            (chunk for chunk in chunks if chunk_difficulty_tier(chunk) == "difficult"),
            key=lambda row: (-float(row["stack_difficulty_score"]), str(row["chunk_id"])),
        ),
    }


def split_target_counts(total: int) -> dict[str, int]:
    train = int(total * SPLIT_RATIOS["train"])
    val = int(total * SPLIT_RATIOS["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def apportion_counts(total: int, keys: list[str], weights: dict[str, int]) -> dict[str, int]:
    """Allocate `total` integer slots across keys using largest remainders."""
    active = [key for key in keys if weights.get(key, 0) > 0]
    if not active or total <= 0:
        return {key: 0 for key in keys}
    weight_sum = sum(weights[key] for key in active)
    raw = {key: total * weights[key] / weight_sum for key in active}
    counts = {key: int(raw[key]) for key in active}
    remaining = total - sum(counts.values())
    for key in sorted(active, key=lambda item: (raw[item] - counts[item], weights[item]), reverse=True)[:remaining]:
        counts[key] += 1
    return {key: counts.get(key, 0) for key in keys}


def assign_font_splits(
    fonts: list[FontCatalogRow],
    supported: dict[str, set[str]],
    rng: random.Random,
) -> dict[str, str]:
    """Assign each font_name to one split, balancing estimated capacity by script_8."""
    by_script_8: dict[str, list[FontCatalogRow]] = defaultdict(list)
    for font in fonts:
        by_script_8[font.script_category].append(font)

    split_by_font_name: dict[str, str] = {}
    for _script_8, script_fonts in sorted(by_script_8.items()):
        unique_by_name: dict[str, FontCatalogRow] = {}
        capacity_by_name: dict[str, int] = defaultdict(int)
        for font in script_fonts:
            name = font_name(font)
            unique_by_name.setdefault(name, font)
            capacity_by_name[name] = max(capacity_by_name[name], len(supported.get(font.basename, set())))
        font_names = sorted(unique_by_name, key=lambda name: (capacity_by_name[name], rng.random()), reverse=True)
        total_capacity = sum(max(1, capacity_by_name[name]) for name in font_names)
        capacity_targets = {split: total_capacity * ratio for split, ratio in SPLIT_RATIOS.items()}
        assigned_capacity = Counter()
        assigned_counts = Counter()

        remaining_names = list(font_names)
        # Give every split at least one font when a script_8 category has enough
        # fonts. Without this, small/capricious categories can lose an entire
        # validation or test slice despite having some usable fonts.
        for split, name in zip(SPLITS, remaining_names[: len(SPLITS)]):
            split_by_font_name[name] = split
            assigned_counts[split] += 1
            assigned_capacity[split] += max(1, capacity_by_name[name])
        remaining_names = remaining_names[len(SPLITS) :]

        for name in remaining_names:
            split = min(
                SPLITS,
                key=lambda item: (
                    assigned_capacity[item] / max(1.0, capacity_targets[item]),
                    assigned_counts[item],
                ),
            )
            split_by_font_name[name] = split
            assigned_counts[split] += 1
            assigned_capacity[split] += max(1, capacity_by_name[name])
    return split_by_font_name


def next_chunk_for_font(
    font_info: FontPlanInfo,
    chunks: list[dict[str, object]],
    supported: dict[str, set[str]],
    used_font_chunks: set[tuple[str, str]],
    chunk_split: dict[str, str],
    chunk_use_counts: Counter[str],
    reused_chunk_ids: set[str],
    max_reused_chunks: int,
    cursors: dict[tuple[str, str, str, bool], int],
    *,
    tier: str,
    allow_reuse: bool,
) -> dict[str, object] | None:
    font = font_info.font
    cursor_key = (font.basename, font_info.split, tier, allow_reuse)
    idx = cursors.get(cursor_key, 0)
    while idx < len(chunks):
        chunk = chunks[idx]
        idx += 1
        chunk_id = str(chunk["chunk_id"])
        font_chunk_key = (font.basename, chunk_id)
        if font_chunk_key in used_font_chunks:
            continue
        if not font_supports_chunk(font, chunk, supported):
            continue

        owner_split = chunk_split.get(chunk_id)
        if owner_split is not None and owner_split != font_info.split:
            continue
        if allow_reuse:
            if owner_split != font_info.split:
                continue
            if chunk_use_counts[chunk_id] == 1 and len(reused_chunk_ids) >= max_reused_chunks:
                continue
        elif owner_split is not None:
            continue

        cursors[cursor_key] = idx
        return chunk
    cursors[cursor_key] = idx
    return None


def _plan_row_dict(
    *,
    image_id: int,
    chunk: dict[str, object],
    font: FontCatalogRow,
    script: str,
    split: str,
    requested_text_difficulty_tier: str,
    selected_pool: str,
) -> dict[str, object]:
    rarity_tier = chunk_difficulty_tier(chunk)
    if requested_text_difficulty_tier == "difficult":
        difficulty_basis = (
            "rare_stacks"
            if rarity_tier == "difficult"
            else "relative_compatible_complexity"
        )
    else:
        difficulty_basis = (
            "no_rare_stacks"
            if rarity_tier == "normal"
            else "rare_stack_fallback"
        )
    return {
        "image_id": image_id,
        "chunk_id": chunk["chunk_id"],
        "bocorpus_row": chunk["bocorpus_row"],
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "text": chunk["text"],
        "char_count": chunk["char_count"],
        "stack_count": chunk["stack_count"],
        "unique_stack_count": chunk["unique_stack_count"],
        "stacks": chunk.get("stacks", ""),
        "stack_difficulty_score": float(chunk["stack_difficulty_score"]),
        "source_text_difficulty_tier": requested_text_difficulty_tier,
        "source_text_rarity_tier": rarity_tier,
        "source_text_difficulty_basis": difficulty_basis,
        "source_text_selection_pool": selected_pool,
        "requested_text_difficulty_tier": requested_text_difficulty_tier,
        "basename": font.basename,
        "font_name": font_name(font),
        "font_file": font.font_file,
        "font_path": font.font_path,
        "font_abs_path": str(font.font_abs_path),
        "ps_name": font.ps_name,
        "ttc_face_index": font.ttc_face_index,
        "font_size_pt": font.font_size_pt,
        "dpi": font.dpi,
        "skt_ok": font.skt_ok,
        "script_id": font.script_id,
        "script_category": font.script_category,
        "script_8": font.script_category,
        "script_type": font.script_type,
        "script": script,
        "script_name": font.script_name,
        "etext_source": (
            f"bocorpus:{chunk['bocorpus_row']}:{chunk['char_start']}:{chunk['char_end']}"
        ),
        "suggested_split": split,
        "ume_dense_oversampled": 0,
        "shorthand_mode": "",
    }


def assign_paired_shorthand_modes(rows: list[dict[str, object]]) -> None:
    """Pin even=with / odd=without shorthand_mode and reassign image_ids per font."""
    by_font: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_font[str(row["basename"])].append(row)

    next_id = 2  # start on even so the first "with" page is dense/sparse-eligible
    for basename in sorted(by_font):
        font_rows = by_font[basename]
        for index, row in enumerate(font_rows):
            script = str(row.get("script") or "")
            if index % 2 == 0:
                if next_id % 2 != 0:
                    next_id += 1
                row["image_id"] = next_id
                row["shorthand_mode"] = mode_for_script(script, next_id)
            else:
                if next_id % 2 == 0:
                    next_id += 1
                row["image_id"] = next_id
                row["shorthand_mode"] = "none"
            next_id += 1


def make_per_font_plan_rows(
    *,
    fonts: list[FontCatalogRow],
    chunks: list[dict[str, object]],
    supported: dict[str, set[str]],
    images_per_font: int,
    seed: int,
    max_chunk_reuse_ratio: float,
    oversample_ume_dense: bool = False,
    pair_shorthand_modes: bool = False,
) -> list[dict[str, object]]:
    """Allocate exactly ``images_per_font`` coverage-compatible chunks per font."""
    if images_per_font <= 0:
        raise ValueError("--images-per-font must be positive")
    rng = random.Random(seed)
    eligible_fonts = [
        font
        for font in fonts
        if font.basename in supported and font.script_category and normalized_script(font.script_type)
    ]
    split_by_font_name = assign_font_splits(eligible_fonts, supported, rng)
    difficulty_pools = build_difficulty_pools(chunks)

    rows: list[dict[str, object]] = []
    used_font_chunks: set[tuple[str, str]] = set()
    chunk_split: dict[str, str] = {}
    chunk_use_counts: Counter[str] = Counter()
    reused_chunk_ids: set[str] = set()
    cursors: dict[tuple[str, str, str, bool], int] = {}
    target_images = len(eligible_fonts) * images_per_font
    max_reused_chunks = int(target_images * max(0.0, max_chunk_reuse_ratio))
    image_id = 1

    ordered_fonts = list(eligible_fonts)
    rng.shuffle(ordered_fonts)
    short_fonts = 0
    for font in tqdm(ordered_fonts, desc="Plan per font", unit="font"):
        script = normalized_script(font.script_type)
        split = split_by_font_name.get(font_name(font))
        if not split:
            continue
        font_info = FontPlanInfo(font=font, split=split)
        made = 0
        for slot in range(images_per_font):
            chunk = None
            requested_tier = "normal" if slot % 2 == 0 else "difficult"
            candidate_tiers = (
                ("difficult", "relative_difficult")
                if requested_tier == "difficult"
                else ("normal", "difficult")
            )
            for tier in candidate_tiers:
                for allow_reuse in (False, True):
                    chunk = next_chunk_for_font(
                        font_info,
                        difficulty_pools[tier],
                        supported,
                        used_font_chunks,
                        chunk_split,
                        chunk_use_counts,
                        reused_chunk_ids,
                        max_reused_chunks,
                        cursors,
                        tier=tier,
                        allow_reuse=allow_reuse,
                    )
                    if chunk is not None:
                        break
                if chunk is not None:
                    break
            if chunk is None:
                break
            chunk_id = str(chunk["chunk_id"])
            used_font_chunks.add((font.basename, chunk_id))
            chunk_split.setdefault(chunk_id, split)
            chunk_use_counts[chunk_id] += 1
            if chunk_use_counts[chunk_id] > 1:
                reused_chunk_ids.add(chunk_id)
            rows.append(
                _plan_row_dict(
                    image_id=image_id,
                    chunk=chunk,
                    font=font,
                    script=script,
                    split=split,
                    requested_text_difficulty_tier=requested_tier,
                    selected_pool=tier,
                )
            )
            image_id += 1
            made += 1
        if made < images_per_font:
            short_fonts += 1
            print(
                f"WARNING: {font.basename} only got {made}/{images_per_font} image(s); "
                "no more compatible chunks"
            )
    if short_fonts:
        print(f"WARNING: {short_fonts} font(s) under-allocated")

    if pair_shorthand_modes:
        assign_paired_shorthand_modes(rows)
    else:
        rows.sort(
            key=lambda row: (
                str(row["script"]),
                str(row["script_8"]),
                str(row["suggested_split"]),
                -float(row["stack_difficulty_score"]),
                str(row["font_name"]),
            )
        )
        next_image_id = 1
        previous_script = ""
        for row in rows:
            script = str(row["script"])
            if previous_script and script != previous_script:
                page_offset = (next_image_id - 1) % GROUP_SIZE
                if page_offset:
                    next_image_id += GROUP_SIZE - page_offset
            row["image_id"] = next_image_id
            next_image_id += 1
            previous_script = script

    if oversample_ume_dense:
        oversample_ume_dense_rows(
            rows,
            chunks=[*difficulty_pools["difficult"], *difficulty_pools["normal"]],
            supported=supported,
            fonts={font.basename: font for font in eligible_fonts},
            used_font_chunks=used_font_chunks,
        )
    return rows


def make_plan_rows(
    *,
    fonts: list[FontCatalogRow],
    chunks: list[dict[str, object]],
    supported: dict[str, set[str]],
    target_images: int,
    seed: int,
    max_chunk_reuse_ratio: float,
    oversample_ume_dense: bool = False,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    eligible_fonts = [
        font
        for font in fonts
        if font.basename in supported and font.script_category and normalized_script(font.script_type)
    ]
    split_by_font_name = assign_font_splits(eligible_fonts, supported, rng)

    difficulty_pools = build_difficulty_pools(chunks)
    categories = sorted({font.script_category for font in eligible_fonts})
    quotas = category_quotas(categories, target_images)

    fonts_by_category_script_split: dict[tuple[str, str, str], list[FontPlanInfo]] = defaultdict(list)
    split_font_counts = Counter()
    script_font_counts = Counter()
    for font in fonts:
        name = font_name(font)
        split = split_by_font_name.get(name)
        if split and font.basename in supported and font.script_category:
            info = FontPlanInfo(font=font, split=split)
            key = (font.script_category, normalized_script(font.script_type), split)
            fonts_by_category_script_split[key].append(info)
            split_font_counts[split] += 1
            script_font_counts[(font.script_category, normalized_script(font.script_type), split)] += 1

    rows: list[dict[str, object]] = []
    used_font_chunks: set[tuple[str, str]] = set()
    chunk_split: dict[str, str] = {}
    chunk_use_counts: Counter[str] = Counter()
    reused_chunk_ids: set[str] = set()
    cursors: dict[tuple[str, str, str, bool], int] = {}
    difficulty_counts_by_font: dict[str, Counter[str]] = defaultdict(Counter)
    max_reused_chunks = int(target_images * max(0.0, max_chunk_reuse_ratio))
    image_id = 1

    for category in categories:
        for split, category_split_quota in split_target_counts(quotas[category]).items():
            script_weights = {
                script: len(fonts_by_category_script_split.get((category, script, split), []))
                for script in ("uchen", "ume")
            }
            script_quotas = apportion_counts(category_split_quota, ["uchen", "ume"], script_weights)
            for script, split_quota in script_quotas.items():
                if split_quota <= 0:
                    continue
                font_infos = fonts_by_category_script_split.get((category, script, split), [])
                if not font_infos:
                    continue
                rng.shuffle(font_infos)
                remaining = split_quota
                progress = tqdm(total=split_quota, desc=f"Plan {category}/{script}/{split}", unit="image")
                while remaining > 0:
                    made_progress = False
                    for font_info in font_infos:
                        if remaining <= 0:
                            break
                        font_counts = difficulty_counts_by_font[font_info.font.basename]
                        requested_tier = (
                            "normal"
                            if font_counts["normal"] <= font_counts["difficult"]
                            else "difficult"
                        )
                        chunk = None
                        for tier in (
                            ("difficult", "relative_difficult")
                            if requested_tier == "difficult"
                            else ("normal", "difficult")
                        ):
                            for allow_reuse in (False, True):
                                chunk = next_chunk_for_font(
                                    font_info,
                                    difficulty_pools[tier],
                                    supported,
                                    used_font_chunks,
                                    chunk_split,
                                    chunk_use_counts,
                                    reused_chunk_ids,
                                    max_reused_chunks,
                                    cursors,
                                    tier=tier,
                                    allow_reuse=allow_reuse,
                                )
                                if chunk is not None:
                                    break
                            if chunk is not None:
                                break
                        if chunk is not None:
                            font = font_info.font
                            chunk_id = str(chunk["chunk_id"])
                            used_font_chunks.add((font.basename, chunk_id))
                            chunk_split.setdefault(chunk_id, split)
                            chunk_use_counts[chunk_id] += 1
                            if chunk_use_counts[chunk_id] > 1:
                                reused_chunk_ids.add(chunk_id)
                            difficulty_counts_by_font[font.basename][requested_tier] += 1
                            rows.append(
                                _plan_row_dict(
                                    image_id=image_id,
                                    chunk=chunk,
                                    font=font,
                                    script=script,
                                    split=split,
                                    requested_text_difficulty_tier=requested_tier,
                                    selected_pool=tier,
                                )
                            )
                            image_id += 1
                            remaining -= 1
                            progress.update(1)
                            made_progress = True
                    if not made_progress:
                        print(
                            f"WARNING: {category}/{script}/{split} short by {remaining} image(s); "
                            "no more compatible font/chunk pairs"
                        )
                        break
                progress.close()

    # Keep benchmark volumes script-pure: assign final IDs after grouping by script.
    rows.sort(
        key=lambda row: (
            str(row["script"]),
            str(row["script_8"]),
            str(row["suggested_split"]),
            -float(row["stack_difficulty_score"]),
            str(row["font_name"]),
        )
    )
    next_image_id = 1
    previous_script = ""
    for row in rows:
        script = str(row["script"])
        if previous_script and script != previous_script:
            page_offset = (next_image_id - 1) % GROUP_SIZE
            if page_offset:
                next_image_id += GROUP_SIZE - page_offset
        row["image_id"] = next_image_id
        next_image_id += 1
        previous_script = script

    if oversample_ume_dense:
        oversample_ume_dense_rows(
            rows,
            chunks=[*difficulty_pools["difficult"], *difficulty_pools["normal"]],
            supported=supported,
            fonts={font.basename: font for font in eligible_fonts},
            used_font_chunks=used_font_chunks,
        )
    return rows


def oversample_ume_dense_rows(
    rows: list[dict[str, object]],
    *,
    chunks: list[dict[str, object]],
    supported: dict[str, set[str]],
    fonts: dict[str, FontCatalogRow],
    used_font_chunks: set[tuple[str, str]],
) -> None:
    """Append ~1 extra compatible chunk to ume even-image rows for dense shorthand fill."""
    # Prefer harder unused chunks first, then allow already-used ones as filler.
    ordered = sorted(chunks, key=lambda row: float(row["stack_difficulty_score"]), reverse=True)
    oversampled = 0
    for row in rows:
        if str(row.get("script") or "") != "ume":
            continue
        mode = str(row.get("shorthand_mode") or "").strip().lower()
        try:
            image_id = int(row["image_id"])
        except (TypeError, ValueError):
            continue
        # Prefer explicit plan mode; otherwise even image_ids are dense pages.
        if mode:
            if mode != "dense":
                continue
        elif image_id % 2 != 0:
            continue
        basename = str(row["basename"])
        font = fonts.get(basename)
        if font is None:
            continue
        existing_ids = set(str(row.get("chunk_id") or "").split("|"))
        extra = None
        for allow_used in (False, True):
            for chunk in ordered:
                chunk_id = str(chunk["chunk_id"])
                if chunk_id in existing_ids:
                    continue
                key = (basename, chunk_id)
                if not allow_used and key in used_font_chunks:
                    continue
                if not font_supports_chunk(font, chunk, supported):
                    continue
                extra = chunk
                used_font_chunks.add(key)
                break
            if extra is not None:
                break
        if extra is None:
            continue
        combined_text = f"{row['text']} {extra['text']}".strip()
        stacks = tokenize_tibetan_stacks(combined_text)
        unique_stacks = sorted(set(stacks))
        row["text"] = combined_text
        row["char_count"] = len(combined_text)
        row["stack_count"] = len(stacks)
        row["unique_stack_count"] = len(unique_stacks)
        row["stacks"] = " ".join(unique_stacks)
        row["stack_difficulty_score"] = stack_difficulty_score(combined_text)
        row["chunk_id"] = f"{row['chunk_id']}|{extra['chunk_id']}"
        row["etext_source"] = (
            f"{row['etext_source']}|bocorpus:{extra['bocorpus_row']}:"
            f"{extra['char_start']}:{extra['char_end']}"
        )
        row["ume_dense_oversampled"] = 1
        oversampled += 1
    print(f"Oversampled {oversampled} ume dense-shorthand plan row(s) with extra chunk text")


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "script_category": Counter(row["script_category"] for row in rows),
        "script": Counter(row["script"] for row in rows),
        "suggested_split": Counter(row["suggested_split"] for row in rows),
        "split_script_8": Counter(f"{row['suggested_split']}|{row['script_8']}" for row in rows),
        "script_id": Counter(row["script_id"] for row in rows),
        "basename": Counter(row["basename"] for row in rows),
        "font_name": Counter(row["font_name"] for row in rows),
        "source_text_difficulty_tier": Counter(
            row["source_text_difficulty_tier"] for row in rows
        ),
        "source_text_rarity_tier": Counter(
            row["source_text_rarity_tier"] for row in rows
        ),
        "source_text_difficulty_basis": Counter(
            row["source_text_difficulty_basis"] for row in rows
        ),
        "font_text_difficulty_tier": Counter(
            f"{row['basename']}|{row['source_text_difficulty_tier']}" for row in rows
        ),
    }
    reused_chunks = sum(1 for _chunk_id, count in Counter(row["chunk_id"] for row in rows).items() if count > 1)
    unique_chunks = len({row["chunk_id"] for row in rows})
    reused_ratio = reused_chunks / unique_chunks if unique_chunks else 0.0
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "value", "count"])
        writer.writerow(["summary", "total_rows", len(rows)])
        writer.writerow(["summary", "unique_chunks", unique_chunks])
        writer.writerow(["summary", "reused_chunks", reused_chunks])
        writer.writerow(["summary", "reused_chunk_percent", f"{reused_ratio:.4%}"])
        for group, counter in counts.items():
            for value, count in sorted(counter.items()):
                writer.writerow([group, value, count])


def main() -> None:
    args = parse_args()
    if not args.chunks.exists():
        raise SystemExit(f"Missing chunks parquet: {args.chunks}")
    if not args.support_parquet.exists():
        raise SystemExit(f"Missing support parquet: {args.support_parquet}")

    fonts = load_font_catalog(args.benchmark_csv, args.scripts_csv, args.fonts_csv)
    supported = load_supported_stacks(args.support_parquet)
    chunks = load_chunks(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found in {args.chunks}")

    if args.images_per_font is not None:
        rows = make_per_font_plan_rows(
            fonts=fonts,
            chunks=chunks,
            supported=supported,
            images_per_font=args.images_per_font,
            seed=args.seed,
            max_chunk_reuse_ratio=args.max_chunk_reuse_ratio,
            oversample_ume_dense=args.oversample_ume_dense,
            pair_shorthand_modes=args.pair_shorthand_modes,
        )
    else:
        if args.pair_shorthand_modes:
            raise SystemExit("--pair-shorthand-modes requires --images-per-font")
        rows = make_plan_rows(
            fonts=fonts,
            chunks=chunks,
            supported=supported,
            target_images=args.target_images,
            seed=args.seed,
            max_chunk_reuse_ratio=args.max_chunk_reuse_ratio,
            oversample_ume_dense=args.oversample_ume_dense,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output, compression="zstd")
    summary = args.summary_csv or args.output.with_suffix(".summary.csv")
    write_summary(rows, summary)
    print(f"Wrote {args.output} ({len(rows)} image plan row(s))")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()

