#!/usr/bin/env python3
"""Short (≤10 syllables) and long (~1000 syllables) text length buckets."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from shorthand_aug import syllable_count
from synthetic_common import SHAD, TSHEG, stack_difficulty_score, tokenize_tibetan_stacks

SHORT_MIN_SYLLABLES = 1
SHORT_MAX_SYLLABLES = 10
LONG_MIN_SYLLABLES = 1000
LONG_TARGET_SYLLABLES = 1100
DEFAULT_SHORT_FRACTION = 0.10
DEFAULT_LONG_FRACTION = 0.10
# Long pages shrink font to fit ~1000 syllables; keep final raster ≥1.5× denser
# on both axes so glyphs stay OCR-readable after that shrink.
LONG_OUTPUT_RESOLUTION_SCALE = 1.5
# Tighter pecha margins than the default 20×16 mm so dense text uses more of the page.
LONG_MARGIN_X_MM = 8.0
LONG_MARGIN_Y_MM = 6.0


def _stable_seed(seed: int, *values: object) -> int:
    payload = ":".join((str(seed), *(str(value) for value in values)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _tibetan_letter_count(text: str) -> int:
    return sum(1 for char in text if "\u0f40" <= char <= "\u0f6c")


def truncate_to_short_prefix(
    text: str,
    *,
    max_syllables: int = SHORT_MAX_SYLLABLES,
    min_syllables: int = SHORT_MIN_SYLLABLES,
) -> str | None:
    """Cut a shad-ended prefix with ``min_syllables``…``max_syllables`` syllables."""
    text = (text or "").strip()
    if not text:
        return None
    if min_syllables < 1:
        raise ValueError("min_syllables must be ≥ 1")
    if min_syllables > max_syllables:
        raise ValueError("min_syllables must be ≤ max_syllables")

    def acceptable(candidate: str) -> bool:
        count = syllable_count(candidate)
        return (
            min_syllables <= count <= max_syllables
            and _tibetan_letter_count(candidate) >= 1
        )

    # Prefer the longest shad-terminated prefix with ≤ max_syllables.
    best: str | None = None
    for index, ch in enumerate(text):
        if ch != SHAD:
            continue
        candidate = text[: index + 1].strip()
        count = syllable_count(candidate)
        if acceptable(candidate):
            best = candidate
        if count > max_syllables:
            break
    if best is not None:
        return best

    # No shad in the first window: take ≤ max syllables and force a trailing shad.
    parts = [part for part in text.split(TSHEG) if part]
    kept: list[str] = []
    count = 0
    for part in parts:
        trial = TSHEG.join(kept + [part])
        trial_count = syllable_count(trial if trial.endswith(SHAD) else trial + SHAD)
        if trial_count > max_syllables and kept:
            break
        kept.append(part)
        count = trial_count
        if count >= max_syllables:
            break
    if not kept:
        return None
    forced = TSHEG.join(kept).rstrip()
    if not forced.endswith(SHAD):
        forced = forced.rstrip(TSHEG) + SHAD
    if not acceptable(forced):
        return None
    return forced


def merge_texts_to_long(
    seed_text: str,
    extra_texts: list[str],
    *,
    min_syllables: int = LONG_MIN_SYLLABLES,
    target_syllables: int = LONG_TARGET_SYLLABLES,
) -> str | None:
    """Grow ``seed_text`` by appending extras until at least ``min_syllables``."""
    combined = (seed_text or "").strip()
    if not combined:
        return None
    if syllable_count(combined) >= min_syllables:
        return combined
    for extra in extra_texts:
        piece = (extra or "").strip()
        if not piece:
            continue
        combined = f"{combined} {piece}".strip()
        if syllable_count(combined) >= target_syllables:
            return combined
        if syllable_count(combined) >= min_syllables:
            return combined
    if syllable_count(combined) >= min_syllables:
        return combined
    return None


def _refresh_row_text_fields(row: dict[str, object], text: str, *, chunk_id_suffix: str) -> None:
    stacks = tokenize_tibetan_stacks(text)
    unique = sorted(set(stacks))
    row["text"] = text
    row["char_count"] = len(text)
    row["stack_count"] = len(stacks)
    row["unique_stack_count"] = len(unique)
    row["stacks"] = " ".join(unique)
    row["stack_difficulty_score"] = stack_difficulty_score(text)
    row["chunk_id"] = f"{row['chunk_id']}|{chunk_id_suffix}"
    row["syllable_count"] = syllable_count(text)


def apply_length_bucket_rewrites(
    rows: list[dict[str, object]],
    *,
    chunks: list[dict[str, object]],
    supported: dict[str, set[str]],
    short_fraction: float = DEFAULT_SHORT_FRACTION,
    long_fraction: float = DEFAULT_LONG_FRACTION,
    seed: int = 13,
    short_max_syllables: int = SHORT_MAX_SYLLABLES,
    short_min_syllables: int = SHORT_MIN_SYLLABLES,
    long_min_syllables: int = LONG_MIN_SYLLABLES,
) -> list[dict[str, object]]:
    """Rewrite a fraction of plan rows into short/long length buckets in place."""
    if not rows:
        return rows
    if short_fraction < 0 or long_fraction < 0 or short_fraction + long_fraction > 1:
        raise ValueError("Length-bucket fractions must be non-negative and sum to ≤ 1")
    if short_min_syllables < 1:
        raise ValueError("short_min_syllables must be ≥ 1")

    short_n = int(round(len(rows) * short_fraction))
    long_n = int(round(len(rows) * long_fraction))
    for row in rows:
        row.setdefault("text_length_bucket", "normal")
        row.setdefault("syllable_count", syllable_count(str(row.get("text") or "")))

    # Spread picks across fonts so buckets stay category-balanced.
    # Prefer non-repetitive slots so length rewrites do not erase the
    # deliberate mid/high repetition quota.
    by_font: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_font[str(row["basename"])].append(row)
    font_names = sorted(by_font)
    rng = random.Random(seed)
    protected_policies = {"repetitive", "highly_repetitive", "rewarded"}

    def pick_rows(count: int, exclude: set[int]) -> list[dict[str, object]]:
        picked: list[dict[str, object]] = []
        cursors = {name: 0 for name in font_names}
        # Deterministic shuffle within each font; low-repetition rows first.
        ordered = {
            name: sorted(
                by_font[name],
                key=lambda row: (
                    str(row.get("source_text_repetition_policy") or "")
                    in protected_policies,
                    _stable_seed(seed, name, row["image_id"], "pick"),
                ),
            )
            for name in font_names
        }
        while len(picked) < count:
            progressed = False
            for name in font_names:
                if len(picked) >= count:
                    break
                bucket = ordered[name]
                while cursors[name] < len(bucket):
                    row = bucket[cursors[name]]
                    cursors[name] += 1
                    image_id = int(row["image_id"])
                    if image_id in exclude:
                        continue
                    picked.append(row)
                    exclude.add(image_id)
                    progressed = True
                    break
            if not progressed:
                break
        rng.shuffle(picked)
        return picked

    used_ids: set[int] = set()
    short_rows = pick_rows(short_n, used_ids)
    long_rows = pick_rows(long_n, used_ids)

    short_ok = 0
    for row in short_rows:
        truncated = truncate_to_short_prefix(
            str(row.get("text") or ""),
            max_syllables=short_max_syllables,
            min_syllables=short_min_syllables,
        )
        if truncated is None:
            continue
        _refresh_row_text_fields(row, truncated, chunk_id_suffix="short")
        row["text_length_bucket"] = "short"
        short_ok += 1

    # Compatible extras cached per font (built lazily).
    extras_by_font: dict[str, list[str]] = {}

    def font_extra_texts(basename: str) -> list[str]:
        cached = extras_by_font.get(basename)
        if cached is not None:
            return cached
        font_stacks = supported.get(basename) or set()
        texts: list[str] = []
        if font_stacks:
            for chunk in chunks:
                stack_set = chunk.get("_stack_set")
                if stack_set is None:
                    stack_set = frozenset(str(chunk.get("stacks") or "").split())
                    chunk["_stack_set"] = stack_set
                if stack_set and stack_set.issubset(font_stacks):
                    text = str(chunk.get("text") or "").strip()
                    if text:
                        texts.append(text)
        extras_by_font[basename] = texts
        return texts

    long_ok = 0
    for row in long_rows:
        basename = str(row["basename"])
        seed_text = str(row.get("text") or "")
        if syllable_count(seed_text) >= long_min_syllables:
            row["text_length_bucket"] = "long"
            row["syllable_count"] = syllable_count(seed_text)
            long_ok += 1
            continue
        extras = list(font_extra_texts(basename))
        extras.sort(
            key=lambda text: _stable_seed(seed, basename, row["image_id"], text[:64])
        )
        merged = merge_texts_to_long(
            seed_text,
            extras,
            min_syllables=long_min_syllables,
        )
        if merged is None:
            continue
        _refresh_row_text_fields(row, merged, chunk_id_suffix="long")
        row["text_length_bucket"] = "long"
        long_ok += 1

    print(
        f"Length buckets: short={short_ok}/{short_n}, long={long_ok}/{long_n}, "
        f"normal={len(rows) - short_ok - long_ok}"
    )
    if short_ok < int(short_n * 0.9) or long_ok < int(long_n * 0.9):
        print(
            "WARNING: length-bucket rewrite under-filled "
            f"(short {short_ok}/{short_n}, long {long_ok}/{long_n})"
        )
    return rows
