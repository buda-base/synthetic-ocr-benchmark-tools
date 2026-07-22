#!/usr/bin/env python3
"""Load and apply Tibetan shorthand replacements for the synthetic benchmark."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from synthetic_common import TSHEG, SHAD, tokenize_tibetan_stacks

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SHORTHANDS_CSV = SCRIPT_DIR / "data" / "shorthands" / "shorthands.csv"
DEFAULT_DENYLIST_CSV = SCRIPT_DIR / "data" / "shorthands" / "denylist.csv"

BOUNDARY_CHARS = set(TSHEG + SHAD + " \t\n\r")


@dataclass(frozen=True)
class ShorthandPair:
    long_form: str
    shorthand: str
    source: str
    shorthand_stacks: tuple[str, ...]


@dataclass(frozen=True)
class DenylistEntry:
    shorthand: str
    basename: str = ""
    stack: str = ""
    reason: str = ""


def syllable_count(text: str) -> int:
    parts = [part for part in text.split(TSHEG) if any("\u0f40" <= ch <= "\u0fbc" for ch in part)]
    return max(1, len(parts)) if text.strip() else 0


def stack_missing_base_letter(stack: str) -> bool:
    """Reject mark-only stacks (e.g. U+0F39 + vowels) that need a dotted-circle base."""
    if not stack:
        return False
    has_base = any("\u0f40" <= ch <= "\u0f6c" for ch in stack)
    if has_base:
        return False
    has_marks = any(
        ("\u0f71" <= ch <= "\u0f84")
        or ("\u0f90" <= ch <= "\u0fbc")
        or ch == "\u0f39"
        for ch in stack
    )
    return has_marks


def load_shorthand_pairs(path: Path = DEFAULT_SHORTHANDS_CSV) -> list[ShorthandPair]:
    pairs: list[ShorthandPair] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            long_form = (row.get("long_form") or "").strip()
            shorthand = (row.get("shorthand") or "").strip()
            source = (row.get("source") or "").strip()
            if not long_form or not shorthand:
                continue
            stacks = tuple(tokenize_tibetan_stacks(shorthand))
            pairs.append(
                ShorthandPair(
                    long_form=long_form,
                    shorthand=shorthand,
                    source=source,
                    shorthand_stacks=stacks,
                )
            )
    pairs.sort(key=lambda pair: (-len(pair.long_form), pair.long_form, pair.shorthand))
    return pairs


def load_denylist(path: Path | None = DEFAULT_DENYLIST_CSV) -> list[DenylistEntry]:
    if path is None or not path.is_file():
        return []
    entries: list[DenylistEntry] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        for row in reader:
            entries.append(
                DenylistEntry(
                    shorthand=(row.get("shorthand") or "").strip(),
                    basename=(row.get("basename") or "").strip(),
                    stack=(row.get("stack") or "").strip(),
                    reason=(row.get("reason") or "").strip(),
                )
            )
    return entries


def extract_shorthand_stacks(pairs: list[ShorthandPair]) -> list[str]:
    seen: set[str] = set()
    stacks: list[str] = []
    for pair in pairs:
        for stack in pair.shorthand_stacks:
            if stack in seen:
                continue
            seen.add(stack)
            stacks.append(stack)
    return stacks


def pair_is_allowed(
    pair: ShorthandPair,
    *,
    supported_stacks: set[str] | None,
    denylist: list[DenylistEntry],
    basename: str = "",
) -> bool:
    # U+0F39 + vowels (and other mark-only stacks) are common in shorthand
    # lists but OpenType inserts a dotted circle; never inject them.
    if any(stack_missing_base_letter(stack) for stack in pair.shorthand_stacks):
        return False
    if stack_missing_base_letter(pair.shorthand):
        return False
    if supported_stacks is not None:
        if not pair.shorthand_stacks:
            return False
        if not set(pair.shorthand_stacks).issubset(supported_stacks):
            return False
    for entry in denylist:
        if entry.basename and basename and entry.basename != basename:
            continue
        if entry.shorthand and entry.shorthand == pair.shorthand:
            return False
        if entry.stack and entry.stack in pair.shorthand_stacks:
            return False
    return True


def find_matches(
    text: str,
    pairs: list[ShorthandPair],
    *,
    supported_stacks: set[str] | None,
    denylist: list[DenylistEntry],
    basename: str = "",
    rng: random.Random | None = None,
) -> list[tuple[int, int, ShorthandPair]]:
    """Return non-overlapping candidate matches as (start, end, pair)."""
    by_long: dict[str, list[ShorthandPair]] = {}
    for pair in pairs:
        if not pair_is_allowed(
            pair,
            supported_stacks=supported_stacks,
            denylist=denylist,
            basename=basename,
        ):
            continue
        by_long.setdefault(pair.long_form, []).append(pair)

    candidates: list[tuple[int, int, ShorthandPair]] = []
    for long_form, options in by_long.items():
        start = 0
        while True:
            idx = text.find(long_form, start)
            if idx < 0:
                break
            end = idx + len(long_form)
            left_ok = idx == 0 or text[idx - 1] in BOUNDARY_CHARS
            right_ok = end >= len(text) or text[end] in BOUNDARY_CHARS
            if left_ok and right_ok:
                chosen = options[0] if rng is None else rng.choice(options)
                candidates.append((idx, end, chosen))
            start = idx + 1
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    return candidates


def _select_non_overlapping(
    candidates: list[tuple[int, int, ShorthandPair]],
    *,
    max_replacements: int | None,
    rng: random.Random | None,
    dense: bool,
) -> list[tuple[int, int, ShorthandPair]]:
    if not candidates:
        return []
    ordered = list(candidates)
    if not dense and rng is not None:
        rng.shuffle(ordered)
        ordered.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    selected: list[tuple[int, int, ShorthandPair]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, pair in ordered:
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        selected.append((start, end, pair))
        occupied.append((start, end))
        if max_replacements is not None and len(selected) >= max_replacements:
            break
    selected.sort(key=lambda item: item[0])
    return selected


def apply_shorthands(
    text: str,
    pairs: list[ShorthandPair],
    *,
    mode: str,
    supported_stacks: set[str] | None = None,
    denylist: list[DenylistEntry] | None = None,
    basename: str = "",
    rng: random.Random | None = None,
    max_per_100_syllables: float = 4.0,
) -> tuple[str, int]:
    """Replace long forms with shorthands.

    mode:
      - "sparse": random non-overlapping replacements capped by density
      - "dense": greedily replace as many supported matches as possible
      - "none": return text unchanged
    """
    if mode == "none" or not text or not pairs:
        return text, 0
    if mode not in {"sparse", "dense"}:
        raise ValueError(f"Unsupported shorthand mode: {mode}")

    deny = denylist or []
    local_rng = rng or random.Random(0)
    candidates = find_matches(
        text,
        pairs,
        supported_stacks=supported_stacks,
        denylist=deny,
        basename=basename,
        rng=local_rng,
    )
    if not candidates:
        return text, 0

    if mode == "dense":
        selected = _select_non_overlapping(candidates, max_replacements=None, rng=None, dense=True)
    else:
        syllables = syllable_count(text)
        max_replacements = max(0, int(syllables * max_per_100_syllables / 100.0))
        if max_replacements == 0:
            return text, 0
        # Often fewer than the cap: sample a count in [0, max].
        target = local_rng.randint(0, max_replacements)
        if target == 0:
            return text, 0
        selected = _select_non_overlapping(
            candidates,
            max_replacements=target,
            rng=local_rng,
            dense=False,
        )

    if not selected:
        return text, 0

    pieces: list[str] = []
    cursor = 0
    for start, end, pair in selected:
        pieces.append(text[cursor:start])
        pieces.append(pair.shorthand)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), len(selected)


def ume_dense_for_image(image_id: int) -> bool:
    """Dense shorthands on even image IDs; odd IDs stay clean."""
    return int(image_id) % 2 == 0


def mode_for_script(script: str, image_id: int) -> str:
    script_l = (script or "").strip().lower()
    if script_l == "uchen":
        return "sparse"
    if script_l == "ume":
        return "dense" if ume_dense_for_image(image_id) else "none"
    return "none"


def write_shorthand_stacks_csv(
    pairs: list[ShorthandPair],
    output: Path,
) -> int:
    stacks = extract_shorthand_stacks(pairs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stack", "nb_occurences", "hunspell_bo", "probe_source"],
        )
        writer.writeheader()
        for stack in stacks:
            writer.writerow(
                {
                    "stack": stack,
                    "nb_occurences": 0,
                    "hunspell_bo": 0,
                    "probe_source": "shorthand",
                }
            )
    return len(stacks)
