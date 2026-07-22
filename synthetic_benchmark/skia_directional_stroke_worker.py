#!/usr/bin/env python3
"""Apply anisotropic outline dilation/erosion to a TrueType font with PathOps."""

from __future__ import annotations

import sys
from pathlib import Path

import pathops
from fontTools.ttLib import TTFont
from fontTools.ttLib.removeOverlaps import skPathFromGlyph, ttfGlyphFromSkPath


def directional_stroke(
    path: pathops.Path,
    amount: float,
    *,
    vertical: bool,
    anisotropy: float = 10.0,
) -> pathops.Path:
    """Change thickness mostly across one stem orientation.

    Scaling the orthogonal axis before a circular stroke and reversing that
    scale afterwards produces an approximately elliptical expansion/erosion.
    PathOps keeps nested contour winding intact, which is essential for Tibetan
    counters and stacked glyph components.
    """
    if not path or amount == 0:
        return pathops.Path(path)
    sx, sy = (1.0, anisotropy) if vertical else (anisotropy, 1.0)
    working = path.transform(sx, 0.0, 0.0, sy, 0.0, 0.0)
    boundary = pathops.Path(working)
    boundary.stroke(
        abs(amount),
        pathops.LineCap.BUTT_CAP,
        pathops.LineJoin.BEVEL_JOIN,
        4.0,
    )
    result = pathops.op(
        working,
        boundary,
        pathops.PathOp.UNION if amount > 0 else pathops.PathOp.DIFFERENCE,
        fix_winding=True,
        keep_starting_points=False,
        clockwise=True,
    )
    result = result.transform(1.0 / sx, 0.0, 0.0, 1.0 / sy, 0.0, 0.0)
    return pathops.simplify(
        result,
        fix_winding=True,
        keep_starting_points=False,
        clockwise=True,
    )


def transform_font(
    source: Path,
    face_index: int,
    output: Path,
    operation: str,
    em_fraction: float,
) -> None:
    font = TTFont(source, fontNumber=face_index, recalcBBoxes=True, recalcTimestamp=False)
    if "glyf" not in font:
        font.close()
        raise ValueError("Directional stroke currently requires TrueType glyf outlines")
    source_glyph_set = font.getGlyphSet()
    glyph_order = font.getGlyphOrder()
    amount = float(font["head"].unitsPerEm) * em_fraction
    vertical = operation == "vertical_stroke"
    new_glyphs = {}
    skipped = 0
    for name in glyph_order:
        source_path = skPathFromGlyph(name, source_glyph_set)
        try:
            result = directional_stroke(source_path, amount, vertical=vertical)
        except pathops.PathOpsError:
            # Preserve a pathological glyph unchanged rather than invalidating
            # an otherwise useful variant. The count is retained in provenance.
            result = pathops.simplify(
                source_path,
                fix_winding=True,
                keep_starting_points=False,
                clockwise=True,
            )
            skipped += 1
        new_glyphs[name] = ttfGlyphFromSkPath(result)

    glyf = font["glyf"]
    glyf.glyphs = new_glyphs
    glyf.glyphOrder = glyph_order
    for name in glyph_order:
        glyph = new_glyphs[name]
        glyph.recalcBounds(glyf)
        advance, _lsb = font["hmtx"].metrics[name]
        font["hmtx"].metrics[name] = (advance, getattr(glyph, "xMin", 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    font.save(output)
    font.close()
    print(f"directional_stroke_skipped_glyphs={skipped}")


def main(argv: list[str]) -> None:
    if len(argv) != 6:
        raise SystemExit(
            "Usage: skia_directional_stroke_worker.py SOURCE FACE_INDEX OUTPUT OPERATION VALUE"
        )
    source, face_index, output, operation, value = argv[1:]
    if operation not in {"vertical_stroke", "horizontal_stroke"}:
        raise ValueError(f"Unsupported directional operation: {operation}")
    transform_font(Path(source), int(face_index), Path(output), operation, float(value))


if __name__ == "__main__":
    main(sys.argv)
