#!/usr/bin/env python3
"""Contiguous multi-chunk text fill for the v6 layout recipe.

A layout page needs enough text to fill its sampled line budget (1-60 lines, at
a width that varies with aspect). The source chunks (~1300 chars each) are
contiguous slices of ~1000 bocorpus rows. This module builds a once-per-process
index and, given a page's primary chunk, extends it with the following
contiguous chunks (natural running text) until the frame is covered, threading a
cursor across the blocks of a composite so each block gets a distinct slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

# Over-provide factor: the LaTeX frame caps visible lines, so a little extra is
# free. Bounded by MAX_FILL_CHARS to avoid TeX main-memory exhaustion on the
# big-slice + many-breakpoint path.
OVERFILL = 1.5
MAX_FILL_CHARS = 12000
MIN_FILL_CHARS = 60


@dataclass
class ChunkIndex:
    ordered: list[dict]                 # chunks sorted by (bocorpus_row, char_start)
    by_id: dict[str, int]               # chunk_id -> global index

    @classmethod
    def load(cls, path: Path) -> "ChunkIndex":
        tb = pq.read_table(
            path,
            columns=["chunk_id", "bocorpus_row", "char_start", "char_end", "text"],
        )
        rows = tb.to_pylist()
        rows.sort(key=lambda r: (r["bocorpus_row"], int(r["char_start"])))
        by_id = {str(r["chunk_id"]): i for i, r in enumerate(rows)}
        return cls(ordered=rows, by_id=by_id)


@dataclass
class FilledBlock:
    text: str
    chunk_ids: list[str] = field(default_factory=list)
    bocorpus_rows: list[str] = field(default_factory=list)
    char_ranges: list[str] = field(default_factory=list)


def estimate_needed_chars(box_w_px: float, font_px: float, n_lines: int) -> int:
    """Rough chars needed to fill ``n_lines`` at a column of ``box_w_px``.

    Tibetan average advance is ~0.5x the glyph px; over-provide and clamp so the
    frame (not the text) decides the realized line count.
    """
    chars_per_line = max(1.0, box_w_px / max(1.0, 0.5 * font_px))
    need = int(chars_per_line * max(1, n_lines) * OVERFILL)
    return max(MIN_FILL_CHARS, min(MAX_FILL_CHARS, need))


def fill_from(index: ChunkIndex, start_global_idx: int, needed_chars: int) -> tuple[FilledBlock, int]:
    """Concatenate chunks forward from ``start_global_idx`` until ``needed_chars``.

    Returns the filled block and the next global index (cursor) to continue from
    for the following block of a composite.
    """
    n = len(index.ordered)
    parts: list[str] = []
    chunk_ids: list[str] = []
    bo_rows: list[str] = []
    ranges: list[str] = []
    total = 0
    idx = start_global_idx % n
    guard = 0
    while total < needed_chars and guard < n:
        c = index.ordered[idx]
        t = str(c["text"])
        parts.append(t)
        chunk_ids.append(str(c["chunk_id"]))
        bo_rows.append(str(c["bocorpus_row"]))
        ranges.append(f"{c['char_start']}:{c['char_end']}")
        total += len(t) + 1
        idx = (idx + 1) % n
        guard += 1
    block = FilledBlock(
        text="\n".join(parts).strip(),
        chunk_ids=chunk_ids,
        bocorpus_rows=bo_rows,
        char_ranges=ranges,
    )
    return block, idx


def fill_page(
    index: ChunkIndex,
    primary_chunk_id: str,
    block_needs: list[int],
) -> list[FilledBlock]:
    """Fill every block of a page, threading a contiguous cursor across blocks.

    ``block_needs`` is the per-block needed-char count (see
    :func:`estimate_needed_chars`). Block 0 starts at the page's primary chunk so
    the plan's coverage/rarity accounting still points at real selected text.
    """
    start = index.by_id.get(str(primary_chunk_id), 0)
    blocks: list[FilledBlock] = []
    cursor = start
    for need in block_needs:
        block, cursor = fill_from(index, cursor, need)
        blocks.append(block)
    return blocks
