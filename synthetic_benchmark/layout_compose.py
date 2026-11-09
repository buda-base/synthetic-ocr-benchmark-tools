#!/usr/bin/env python3
"""Image-level composition for the v6 layout recipe.

Operates on grayscale ``uint8`` rasters where 255 is paper and 0 is ink (the
convention the document-augmentation stage multiplies a paper background into).
All functions are pure array ops so they are testable without LuaLaTeX.

Responsibilities:

* :func:`crop_to_ink` / :func:`pad_to_margins` turn a rendered text block into a
  page with exact per-side margins expressed as a fraction of the shorter image
  side. A zero margin leaves the ink touching that edge.
* :func:`stack_blocks` vertically stacks independently rendered blocks for a
  composite, separated by a dark "folio" bar (optionally with rounded ends) or a
  blank paper gap.
* :func:`photo_composite` places two folios on a black background with a slight
  perspective warp and a soft drop shadow, approximating a phone photo of two
  pecha leaves.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAPER = 255
INK = 0
INK_THRESHOLD = 200  # pixels below this are treated as ink when locating content


# ---------------------------------------------------------------------------
# Ink bounding box and margin padding.
# ---------------------------------------------------------------------------


def ink_bbox(gray: np.ndarray, threshold: int = INK_THRESHOLD):
    """Return (top, left, bottom, right) of the ink, or None if the page is blank."""
    mask = gray < threshold
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def crop_to_ink(gray: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    box = ink_bbox(gray, threshold)
    if box is None:
        return gray
    top, left, bottom, right = box
    return gray[top:bottom, left:right]


def _solve_shorter_side(iw: int, ih: int, l: float, r: float, t: float, b: float) -> float:
    """Fixed-point solve for the final shorter side, since margins are a fraction
    of it. Converges in a few iterations because total margin fraction < 1."""
    s = float(min(iw, ih))
    for _ in range(8):
        w = iw + (l + r) * s
        h = ih + (t + b) * s
        s = min(w, h)
    return s


@dataclass
class PaddedMeta:
    width: int
    height: int
    aspect: float
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int


def pad_to_margins(
    gray: np.ndarray,
    margins,  # (left, right, top, bottom) as fractions of the shorter side
    *,
    paper: int = PAPER,
    crop_first: bool = True,
) -> tuple[np.ndarray, PaddedMeta]:
    """Pad a block so each side's margin equals ``frac * shorter_final_side``."""
    if crop_first:
        gray = crop_to_ink(gray)
    ih, iw = gray.shape[:2]
    l, r, t, b = (float(m) for m in margins)
    s = _solve_shorter_side(iw, ih, l, r, t, b)
    pad_l, pad_r = round(l * s), round(r * s)
    pad_t, pad_b = round(t * s), round(b * s)
    out = cv2.copyMakeBorder(
        gray, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=paper
    )
    h, w = out.shape[:2]
    meta = PaddedMeta(
        width=w,
        height=h,
        aspect=w / max(1, h),
        pad_left=pad_l,
        pad_right=pad_r,
        pad_top=pad_t,
        pad_bottom=pad_b,
    )
    return out, meta


# ---------------------------------------------------------------------------
# Vertical composite stacking (bars / gaps).
# ---------------------------------------------------------------------------


def _pad_to_width(gray: np.ndarray, width: int, *, paper: int = PAPER) -> np.ndarray:
    h, w = gray.shape[:2]
    if w >= width:
        return gray
    extra = width - w
    left = extra // 2
    right = extra - left
    return cv2.copyMakeBorder(gray, 0, 0, left, right, cv2.BORDER_CONSTANT, value=paper)


def make_bar(
    width: int,
    bar_px: int,
    *,
    rounded: bool = False,
    value: int = 30,
) -> np.ndarray:
    """A dark horizontal bar (folio edge). Rounded ends emulate a leaf edge."""
    bar_px = max(1, int(bar_px))
    bar = np.full((bar_px, width), PAPER, dtype=np.uint8)
    if not rounded:
        bar[:, :] = value
        return bar
    radius = min(bar_px, width // 2)
    # Rounded rectangle via a filled poly with circular caps.
    cv2.rectangle(bar, (radius, 0), (width - radius, bar_px - 1), value, thickness=-1)
    cv2.circle(bar, (radius, bar_px // 2), max(1, bar_px // 2), value, thickness=-1)
    cv2.circle(
        bar, (width - radius, bar_px // 2), max(1, bar_px // 2), value, thickness=-1
    )
    return bar


@dataclass
class StackMeta:
    width: int
    height: int
    aspect: float


def stack_blocks(
    blocks: list[np.ndarray],
    separators: list,  # list of layout_recipe.Separator-like objects
    pitches_px: list[float],
    *,
    paper: int = PAPER,
) -> tuple[np.ndarray, StackMeta]:
    """Stack blocks vertically, inserting a bar or paper gap between them."""
    assert len(blocks) >= 2
    assert len(separators) == len(blocks) - 1
    width = max(block.shape[1] for block in blocks)
    width = int(width)
    sum_heights = sum(block.shape[0] for block in blocks)
    bar_fraction_sum = sum(
        getattr(sep, "bar_fraction", 0.0) for sep in separators if getattr(sep, "kind", "") == "bar"
    )
    bar_fraction_sum = min(0.6, bar_fraction_sum)

    parts: list[np.ndarray] = []
    for i, block in enumerate(blocks):
        parts.append(_pad_to_width(block, width, paper=paper))
        if i < len(separators):
            sep = separators[i]
            if getattr(sep, "kind", "gap") == "bar":
                # Bar height is a fraction of the final image height; approximate
                # the final height from the content and bar fractions.
                base = sum_heights / max(1e-6, (1.0 - bar_fraction_sum))
                bar_px = max(2, round(sep.bar_fraction * base))
                value = 45 if getattr(sep, "bar_dark_brown", False) else 20
                parts.append(
                    make_bar(
                        width,
                        bar_px,
                        rounded=getattr(sep, "bar_rounded", False),
                        value=value,
                    )
                )
            else:
                gap_px = max(1, round(sep.gap_pitches * pitches_px[i]))
                parts.append(np.full((gap_px, width), paper, dtype=np.uint8))
    out = np.vstack(parts)
    h, w = out.shape[:2]
    return out, StackMeta(width=w, height=h, aspect=w / max(1, h))


# ---------------------------------------------------------------------------
# Photograph-style two-folio composite.
# ---------------------------------------------------------------------------


def _warp_folio(folio: np.ndarray, rng: np.random.Generator, canvas_wh: tuple[int, int]):
    """Perspective-warp a folio into a canvas, return (warped_rgb, alpha)."""
    h, w = folio.shape[:2]
    cw, ch = canvas_wh
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter_x = 0.06 * w
    jitter_y = 0.06 * h
    dst = np.float32(
        [
            [rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
            [w - rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
            [w - rng.uniform(0, jitter_x), h - rng.uniform(0, jitter_y)],
            [rng.uniform(0, jitter_x), h - rng.uniform(0, jitter_y)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    folio_rgb = cv2.cvtColor(folio, cv2.COLOR_GRAY2BGR)
    warped = cv2.warpPerspective(
        folio_rgb, matrix, (w, h), borderValue=(0, 0, 0)
    )
    alpha = cv2.warpPerspective(
        np.full((h, w), 255, np.uint8), matrix, (w, h), borderValue=0
    )
    return warped, alpha


def photo_composite(
    folios: list[np.ndarray],
    *,
    seed: int = 0,
    gap_frac: float = 0.06,
    margin_frac: float = 0.10,
) -> tuple[np.ndarray, "StackMeta"]:
    """Two folios stacked on a black background with perspective + drop shadow.

    Returns a grayscale image (0 = black background / ink, 255 = paper)."""
    rng = np.random.default_rng(seed)
    width = max(folio.shape[1] for folio in folios)
    folios = [_pad_to_width(folio, width, paper=PAPER) for folio in folios]
    gap = round(gap_frac * width)
    margin = round(margin_frac * width)
    total_h = sum(f.shape[0] for f in folios) + gap + 2 * margin
    total_w = width + 2 * margin
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    y = margin
    for folio in folios:
        fh = folio.shape[0]
        warped, alpha = _warp_folio(folio, rng, (width, fh))
        # Soft drop shadow.
        shadow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(3, width // 80))
        sy0, sx0 = y + max(4, fh // 60), margin + max(4, width // 80)
        sy1, sx1 = sy0 + fh, sx0 + width
        sy1c, sx1c = min(sy1, total_h), min(sx1, total_w)
        sh = shadow[: sy1c - sy0, : sx1c - sx0].astype(np.float32) / 255.0
        region = canvas[sy0:sy1c, sx0:sx1c].astype(np.float32)
        canvas[sy0:sy1c, sx0:sx1c] = (region * (1.0 - 0.6 * sh[..., None])).astype(np.uint8)
        # Paste warped folio.
        a = (alpha > 127)
        roi = canvas[y : y + fh, margin : margin + width]
        roi[a] = warped[a]
        y += fh + gap

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    return gray, StackMeta(width=w, height=h, aspect=w / max(1, h))
