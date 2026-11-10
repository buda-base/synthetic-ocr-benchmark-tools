#!/usr/bin/env python3
"""Deterministic joint page-layout sampling for the v6 "layout" recipe.

The v5/earlier recipes rendered every page in a fixed pecha frame (aspect ~4,
generous margins, text flowed to whatever line count fit). PaddleOCR-VL learns
the "next line" pointer from that fixed frame and drops lines whenever the real
page is outside it. This module samples the page *layout* jointly and
continuously so the row position of a line must be read from content, not from a
learned page frame.

It is pure-stdlib and deterministic: every page's layout is a function of
``(seed, page_id)`` only, so a plan is reproducible and a render can recompute
the exact same geometry without carrying large sidecars.

Sampled per page (see :class:`PageLayout`):

* composite kind: single (75%), vertical stack of 2-3 blocks (20%), or a
  photograph-style two-folio composite on black (5%).
* per-side margins as a fraction of the shorter image side: 10% of pages are
  zero on all four sides, 30% have at least one zero side, the rest log-uniform
  in 0.5-15%.
* image aspect ratio w/h, log-uniform over 0.06-12 (singles sample it directly;
  composites realise it from their stacked blocks).
* line count per block, uniform 1-60.
* glyph size 12-28 px-equivalent and an inter-line gap log-uniform in
  0.05-0.6 of the glyph height (tight <=0.15 lands on >=20% of blocks).
* ragged pages (30%): 20-60% of lines replaced by 1-3 syllable short lines.
* blank-line pages (10%): empty rendered lines whose label is skipped.

Units: margins/gaps are fractions; glyph size is px; line counts are integers.
Pixel geometry is realised lazily (see :func:`realize_geometry`) from a chosen
raster density so the sampler stays rendering-agnostic.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Recipe constants (the knobs the training recipe pins down).
# ---------------------------------------------------------------------------

MARGIN_MAX = 0.15
MARGIN_MIN = 0.005
ASPECT_MIN = 0.06
ASPECT_MAX = 12.0
GAP_MIN = 0.05
GAP_MAX = 0.60
# Glyph size in output px. Raised from the earlier 12-28 "px equivalent": QC
# review found those too small / too many characters per page. ~42 px was
# approved as "much better", so the range is centred there.
FONT_PX_MIN = 30.0
FONT_PX_MAX = 56.0
LINE_MIN = 1
LINE_MAX = 60

RATE_ALL_ZERO_MARGIN = 0.10
RATE_ANY_ZERO_MARGIN = 0.30  # cumulative: includes the all-zero pages
# Reserved fraction rendered in the *classic pecha* format (wide landscape
# folio, generous margins on all sides, a handful of lines, no ragged / blank /
# composite). Guards against pecha regression during fine-tuning. The remaining
# (1 - RATE_PECHA) is split among the new-layout kinds at their relative rates.
RATE_PECHA = 0.15
RATE_PHOTO = 0.05
RATE_STACK = 0.20
RATE_RAGGED = 0.30
RATE_BLANK = 0.10

# Classic-pecha geometry (mimics the previous fixed-frame recipe).
PECHA_ASPECT_MIN = 2.8
PECHA_ASPECT_MAX = 6.0
PECHA_MARGIN_MIN = 0.03
PECHA_MARGIN_MAX = 0.10
PECHA_LINES_MIN = 5
PECHA_LINES_MAX = 12
PECHA_GAP_MIN = 0.12
PECHA_GAP_MAX = 0.35

RAGGED_MIN_FRACTION = 0.20
RAGGED_MAX_FRACTION = 0.60
RAGGED_SHORT_MIN_SYLL = 1
RAGGED_SHORT_MAX_SYLL = 3

STACK_MIN_BLOCKS = 2
STACK_MAX_BLOCKS = 3
# Separator between stacked blocks: a dark "folio bar" or a blank paper gap.
RATE_SEPARATOR_BAR = 0.55
BAR_FRACTION_MIN = 0.003  # of image height
BAR_FRACTION_MAX = 0.03
BAR_ROUNDED_RATE = 0.5
GAP_PITCHES_MIN = 1.0  # blank paper gap, in line pitches of the upper block
GAP_PITCHES_MAX = 4.0

# Composite image width (px) is sampled directly; height emerges from the
# stacked blocks. Kept modest so most stacks end up tall (small aspect).
STACK_WIDTH_PX_MIN = 600.0
STACK_WIDTH_PX_MAX = 2400.0
PHOTO_FOLIO_WIDTH_PX_MIN = 700.0
PHOTO_FOLIO_WIDTH_PX_MAX = 1800.0

# Real paper backgrounds only read well on roughly book/wide pages; elsewhere we
# fall back to synthetic/clean paper. Composites manage their own background.
REAL_BG_ASPECT_MIN = 0.30
REAL_BG_ASPECT_MAX = 3.5

# Hard cap on the longer image side after realisation. Extreme aspect x line
# count combinations are brought into budget by TRIMMING CONTENT (fewer lines),
# never by shrinking the glyphs -- so the approved font size is always preserved.
MAX_IMAGE_SIDE_PX = 4000


# ---------------------------------------------------------------------------
# Deterministic RNG helpers.
# ---------------------------------------------------------------------------


def _stable_seed(seed: int, *values: object) -> int:
    payload = ":".join((str(seed), *(str(value) for value in values)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _rng(seed: int, *values: object) -> random.Random:
    return random.Random(_stable_seed(seed, *values))


def _loguniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


# ---------------------------------------------------------------------------
# Data model.
# ---------------------------------------------------------------------------


@dataclass
class Margins:
    left: float
    right: float
    top: float
    bottom: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.left, self.right, self.top, self.bottom)

    @property
    def any_zero(self) -> bool:
        return min(self.as_tuple()) <= 0.0

    @property
    def all_zero(self) -> bool:
        return max(self.as_tuple()) <= 0.0


@dataclass
class BlockSpec:
    """One independently-rendered text block (a whole single page, or one slab
    of a vertical composite / one folio of a photo composite)."""

    font_px: float
    gap_frac: float
    n_lines: int
    margins: Margins
    ragged: bool = False
    ragged_fraction: float = 0.0
    blank_lines: bool = False

    @property
    def line_pitch_px(self) -> float:
        # Baseline-to-baseline distance. Tight gaps (<=0.15) are left tight on
        # purpose; collision avoidance is deliberately NOT applied here.
        return self.font_px * (1.0 + self.gap_frac)


@dataclass
class Separator:
    kind: str  # "bar" | "gap"
    bar_fraction: float = 0.0  # of image height, for kind == "bar"
    bar_rounded: bool = False
    bar_dark_brown: bool = False
    gap_pitches: float = 0.0  # in line pitches of the upper block, for "gap"


@dataclass
class PageLayout:
    page_id: int
    kind: str  # "single" | "stack" | "photo"
    aspect: float  # realised w/h (singles: sampled; composites: emergent)
    blocks: list[BlockSpec] = field(default_factory=list)
    separators: list[Separator] = field(default_factory=list)
    # Composite-only: sampled target image width in px before caps.
    target_width_px: Optional[float] = None
    # Whether a real paper background is permitted for this page.
    real_background_allowed: bool = False


# ---------------------------------------------------------------------------
# Field samplers.
# ---------------------------------------------------------------------------


def sample_margins(rng: random.Random) -> Margins:
    u = rng.random()
    sides = [_loguniform(rng, MARGIN_MIN, MARGIN_MAX) for _ in range(4)]
    if u < RATE_ALL_ZERO_MARGIN:
        return Margins(0.0, 0.0, 0.0, 0.0)
    if u < RATE_ANY_ZERO_MARGIN:
        # Zero at least one side, but not all four.
        k = rng.randint(1, 3)
        for idx in rng.sample(range(4), k):
            sides[idx] = 0.0
    return Margins(*sides)


def sample_aspect(rng: random.Random) -> float:
    return _loguniform(rng, ASPECT_MIN, ASPECT_MAX)


def sample_gap(rng: random.Random) -> float:
    return _loguniform(rng, GAP_MIN, GAP_MAX)


def sample_font_px(rng: random.Random) -> float:
    return rng.uniform(FONT_PX_MIN, FONT_PX_MAX)


def sample_line_count(rng: random.Random) -> int:
    return rng.randint(LINE_MIN, LINE_MAX)


def sample_block(
    rng: random.Random,
    *,
    ragged: bool,
    blank_lines: bool,
) -> BlockSpec:
    ragged_fraction = (
        rng.uniform(RAGGED_MIN_FRACTION, RAGGED_MAX_FRACTION) if ragged else 0.0
    )
    return BlockSpec(
        font_px=sample_font_px(rng),
        gap_frac=sample_gap(rng),
        n_lines=sample_line_count(rng),
        margins=sample_margins(rng),
        ragged=ragged,
        ragged_fraction=ragged_fraction,
        blank_lines=blank_lines,
    )


def sample_pecha_block(rng: random.Random) -> BlockSpec:
    """A clean classic-pecha block: generous margins on all sides, moderate
    pitch, a handful of lines, never ragged / blank."""
    sides = [_loguniform(rng, PECHA_MARGIN_MIN, PECHA_MARGIN_MAX) for _ in range(4)]
    return BlockSpec(
        font_px=sample_font_px(rng),
        gap_frac=_loguniform(rng, PECHA_GAP_MIN, PECHA_GAP_MAX),
        n_lines=rng.randint(PECHA_LINES_MIN, PECHA_LINES_MAX),
        margins=Margins(*sides),
        ragged=False,
        ragged_fraction=0.0,
        blank_lines=False,
    )


def sample_separator(rng: random.Random) -> Separator:
    if rng.random() < RATE_SEPARATOR_BAR:
        return Separator(
            kind="bar",
            bar_fraction=rng.uniform(BAR_FRACTION_MIN, BAR_FRACTION_MAX),
            bar_rounded=rng.random() < BAR_ROUNDED_RATE,
            bar_dark_brown=rng.random() < 0.5,
        )
    return Separator(
        kind="gap",
        gap_pitches=rng.uniform(GAP_PITCHES_MIN, GAP_PITCHES_MAX),
    )


# ---------------------------------------------------------------------------
# Page sampler.
# ---------------------------------------------------------------------------


def sample_page_layout(seed: int, page_id: int) -> PageLayout:
    kind_rng = _rng(seed, page_id, "kind")
    u = kind_rng.random()
    rest = 1.0 - RATE_PECHA
    if u < RATE_PECHA:
        kind = "pecha"
    elif u < RATE_PECHA + rest * RATE_PHOTO:
        kind = "photo"
    elif u < RATE_PECHA + rest * (RATE_PHOTO + RATE_STACK):
        kind = "stack"
    else:
        kind = "single"

    ragged = _rng(seed, page_id, "ragged").random() < RATE_RAGGED
    blank = _rng(seed, page_id, "blank").random() < RATE_BLANK

    block_rng = _rng(seed, page_id, "blocks")

    if kind == "pecha":
        aspect = _loguniform(_rng(seed, page_id, "aspect"), PECHA_ASPECT_MIN, PECHA_ASPECT_MAX)
        block = sample_pecha_block(block_rng)
        layout = PageLayout(page_id=page_id, kind="pecha", aspect=aspect, blocks=[block])
        layout.real_background_allowed = True
        return layout

    if kind == "single":
        aspect = sample_aspect(_rng(seed, page_id, "aspect"))
        block = sample_block(block_rng, ragged=ragged, blank_lines=blank)
        layout = PageLayout(
            page_id=page_id,
            kind=kind,
            aspect=aspect,
            blocks=[block],
        )
        layout.real_background_allowed = (
            REAL_BG_ASPECT_MIN <= aspect <= REAL_BG_ASPECT_MAX
        )
        return layout

    if kind == "stack":
        n_blocks = block_rng.randint(STACK_MIN_BLOCKS, STACK_MAX_BLOCKS)
    else:  # photo: exactly two folios
        n_blocks = 2

    blocks = [
        sample_block(_rng(seed, page_id, "block", i), ragged=ragged, blank_lines=blank)
        for i in range(n_blocks)
    ]
    separators = [
        sample_separator(_rng(seed, page_id, "sep", i)) for i in range(n_blocks - 1)
    ]
    width_rng = _rng(seed, page_id, "width")
    if kind == "photo":
        target_width = _loguniform(
            width_rng, PHOTO_FOLIO_WIDTH_PX_MIN, PHOTO_FOLIO_WIDTH_PX_MAX
        )
    else:
        target_width = _loguniform(width_rng, STACK_WIDTH_PX_MIN, STACK_WIDTH_PX_MAX)

    layout = PageLayout(
        page_id=page_id,
        kind=kind,
        aspect=0.0,  # realised below
        blocks=blocks,
        separators=separators,
        target_width_px=target_width,
        real_background_allowed=True,  # composites manage their own paper
    )
    layout.aspect = realize_composite_aspect(layout)
    return layout


# ---------------------------------------------------------------------------
# Geometry realisation (lazy, rendering-agnostic).
# ---------------------------------------------------------------------------


def _single_dimensions_px(layout: PageLayout) -> tuple[float, float]:
    """Return (width, height) px for a single-block page at its sampled aspect.

    Solves the coupled system where margins are fractions of the *shorter* image
    side and the content height is n_lines * pitch.
    """
    block = layout.blocks[0]
    content_h = max(1.0, block.n_lines * block.line_pitch_px)
    a = layout.aspect
    t = block.margins.top + block.margins.bottom
    lr = block.margins.left + block.margins.right
    if a >= 1.0:
        # Wide page: shorter side is height.
        height = content_h / max(1e-3, (1.0 - t))
        width = a * height
    else:
        # Tall page: shorter side is width. content_h = H - (t)*W, H = W/a.
        # => content_h = W*(1/a - t) => W = content_h / (1/a - t)
        width = content_h / max(1e-3, (1.0 / a - t))
        height = width / a
    # lr only affects the text box width (how much text per line), not the
    # outer image dimensions here, which are pinned by aspect + content height.
    _ = lr
    return width, height


def _block_slot_height_px(block: BlockSpec, shorter_side_px: float) -> float:
    content_h = max(1.0, block.n_lines * block.line_pitch_px)
    return (
        block.margins.top * shorter_side_px
        + content_h
        + block.margins.bottom * shorter_side_px
    )


def realize_composite_aspect(layout: PageLayout) -> float:
    width, height = _composite_dimensions_px(layout)
    return width / max(1.0, height)


def _composite_dimensions_px(layout: PageLayout) -> tuple[float, float]:
    width = float(layout.target_width_px or STACK_WIDTH_PX_MIN)
    shorter = width  # composites are overwhelmingly tall; shorter side ~ width
    height = 0.0
    for i, block in enumerate(layout.blocks):
        height += _block_slot_height_px(block, shorter)
        if i < len(layout.separators):
            sep = layout.separators[i]
            if sep.kind == "gap":
                height += sep.gap_pitches * block.line_pitch_px
            # bar_fraction is a fraction of final height; approximate its
            # additive contribution by solving H = base / (1 - sum(bar_frac)).
    bar_sum = sum(
        sep.bar_fraction for sep in layout.separators if sep.kind == "bar"
    )
    if bar_sum < 0.9:
        height = height / (1.0 - bar_sum)
    return width, height


def is_single_kind(kind: str) -> bool:
    """Single-block page kinds (one text column): new 'single' and 'pecha'."""
    return kind in ("single", "pecha")


def realize_dimensions_px(layout: PageLayout) -> tuple[float, float]:
    """Realised (width, height) in px, before the max-side cap."""
    if is_single_kind(layout.kind):
        return _single_dimensions_px(layout)
    return _composite_dimensions_px(layout)


def capped_dimensions_px(layout: PageLayout) -> tuple[float, float, float]:
    """Return (width, height, content_scale) after clamping the longer side.

    ``content_scale`` <= 1 is applied to the *line count* (content), NOT the
    glyph size: image dimensions scale with the number of rendered lines, so
    trimming lines shrinks the page while the font px stays exactly as sampled.
    """
    width, height = realize_dimensions_px(layout)
    longer = max(width, height)
    scale = 1.0
    if longer > MAX_IMAGE_SIDE_PX:
        scale = MAX_IMAGE_SIDE_PX / longer
    return width * scale, height * scale, scale


def effective_line_counts(layout: PageLayout) -> list[int]:
    """Per-block line count after the size cap trims content (font preserved)."""
    _, _, scale = capped_dimensions_px(layout)
    return [max(1, int(round(block.n_lines * scale))) for block in layout.blocks]


# ---------------------------------------------------------------------------
# Marginal-distribution report (sanity before rendering).
# ---------------------------------------------------------------------------


def _quantiles(values: list[float], qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for q in qs:
        idx = min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))
        out[f"p{int(q * 100)}"] = round(ordered[idx], 4)
    return out


def summarize_layouts(layouts: list[PageLayout]) -> dict:
    n = len(layouts)
    if not n:
        return {}

    kinds = {"pecha": 0, "single": 0, "stack": 0, "photo": 0}
    any_zero = all_zero = 0
    side_zero = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    nonzero_margins: list[float] = []
    aspects: list[float] = []
    aspect_lt_05 = aspect_book = aspect_gt2 = 0
    line_counts: list[int] = []
    lines_gt15 = 0
    font_px: list[float] = []
    gaps: list[float] = []
    gap_tight = 0
    ragged_pages = blank_pages = 0
    sep_bar = sep_gap = 0
    capped = 0
    realized_line_counts: list[float] = []

    for layout in layouts:
        kinds[layout.kind] += 1
        aspects.append(layout.aspect)
        if layout.aspect < 0.5:
            aspect_lt_05 += 1
        elif layout.aspect <= 2.0:
            aspect_book += 1
        else:
            aspect_gt2 += 1

        page_any_zero = page_all_zero = False
        page_ragged = page_blank = False
        for block in layout.blocks:
            m = block.margins
            if m.any_zero:
                page_any_zero = True
            for name in ("left", "right", "top", "bottom"):
                if getattr(m, name) <= 0.0:
                    side_zero[name] += 1
            nonzero_margins.extend(v for v in m.as_tuple() if v > 0.0)
            line_counts.append(block.n_lines)
            if block.n_lines > 15:
                lines_gt15 += 1
            font_px.append(block.font_px)
            gaps.append(block.gap_frac)
            if block.gap_frac <= 0.15:
                gap_tight += 1
            if block.ragged:
                page_ragged = True
            if block.blank_lines:
                page_blank = True
        if layout.blocks and all(b.margins.all_zero for b in layout.blocks):
            page_all_zero = True
        if page_any_zero:
            any_zero += 1
        if page_all_zero:
            all_zero += 1
        if page_ragged:
            ragged_pages += 1
        if page_blank:
            blank_pages += 1
        for sep in layout.separators:
            if sep.kind == "bar":
                sep_bar += 1
            else:
                sep_gap += 1

        _, _, scale = capped_dimensions_px(layout)
        if scale < 1.0:
            capped += 1
        for eff in effective_line_counts(layout):
            realized_line_counts.append(float(eff))

    n_blocks = len(line_counts)

    def pct(count: int, denom: int = n) -> float:
        return round(100.0 * count / denom, 2) if denom else 0.0

    return {
        "pages": n,
        "blocks": n_blocks,
        "composite_mix_pct": {
            "pecha": pct(kinds["pecha"]),
            "single": pct(kinds["single"]),
            "stack": pct(kinds["stack"]),
            "photo": pct(kinds["photo"]),
        },
        "margins": {
            "pages_any_side_zero_pct": pct(any_zero),
            "pages_all_sides_zero_pct": pct(all_zero),
            "per_side_zero_pct_of_blocks": {
                k: pct(v, n_blocks) for k, v in side_zero.items()
            },
            "nonzero_margin_fraction_quantiles": _quantiles(nonzero_margins),
        },
        "aspect": {
            "lt_0.5_pct": pct(aspect_lt_05),
            "0.5_to_2_pct": pct(aspect_book),
            "gt_2_pct": pct(aspect_gt2),
            "quantiles": _quantiles(aspects),
        },
        "line_count": {
            "gt_15_pct_of_blocks": pct(lines_gt15, n_blocks),
            "quantiles": _quantiles([float(v) for v in line_counts]),
        },
        "font_px": {"quantiles": _quantiles(font_px), "note": "preserved; cap trims content not glyphs"},
        "realized_line_count_after_cap": {"quantiles": _quantiles(realized_line_counts)},
        "inter_line_gap": {
            "tight_le_0.15_pct_of_blocks": pct(gap_tight, n_blocks),
            "quantiles": _quantiles(gaps),
        },
        "ragged_pages_pct": pct(ragged_pages),
        "blank_line_pages_pct": pct(blank_pages),
        "separators": {
            "bar_pct": pct(sep_bar, max(1, sep_bar + sep_gap)),
            "gap_pct": pct(sep_gap, max(1, sep_bar + sep_gap)),
            "total": sep_bar + sep_gap,
        },
        "pages_line_trimmed_by_cap_pct": pct(capped),
    }


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    layouts = [sample_page_layout(args.seed, pid) for pid in range(1, args.pages + 1)]
    print(json.dumps(summarize_layouts(layouts), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
