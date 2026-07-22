#!/usr/bin/env fontforge
"""FontForge worker invoked by font_augmentation.py; not run with the project venv."""

from __future__ import annotations

import math
import re
import sys

import fontforge
import psMat


def open_face(path: str, face_index: int):
    names = tuple(fontforge.fontsInFile(path) or ())
    if names:
        if not 0 <= face_index < len(names):
            raise ValueError(f"Face index {face_index} outside 0..{len(names) - 1} for {path}")
        return fontforge.open(f"{path}({names[face_index]})")
    if face_index not in (0,):
        raise ValueError(f"Non-collection font cannot use face index {face_index}: {path}")
    return fontforge.open(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")[:52] or "TibetanFont"


def set_variant_names(font, variant_id: str) -> None:
    suffix = f"Aug-{variant_id[:10]}"
    family = safe_name(font.familyname or font.fontname or "TibetanFont")
    font.familyname = f"{family} {suffix}"
    font.fullname = f"{family} {suffix}"
    font.fontname = safe_name(f"{family}-{suffix}")


def transform_width(font, scale: float) -> None:
    original_widths = {glyph.encoding: glyph.width for glyph in font.glyphs()}
    font.selection.all()
    font.transform(psMat.scale(scale, 1.0))
    for glyph in font.glyphs():
        if glyph.encoding in original_widths:
            glyph.width = int(round(original_widths[glyph.encoding] * scale))


def transform_slant(font, degrees: float) -> None:
    font.selection.all()
    font.transform(psMat.skew(math.radians(degrees)))


def transform_weight(font, em_fraction: float) -> None:
    amount = float(font.em) * em_fraction
    font.selection.all()
    font.changeWeight(
        amount,
        "LCG",
        0,
        0,
        "squish",
        True,
    )


def transform_directional_stroke(font, em_fraction: float, *, vertical: bool) -> None:
    """Expand/contract contours mainly across one stem orientation."""
    amount = float(font.em) * em_fraction
    cross_amount = amount * 0.10
    angle = 0.0 if vertical else math.pi / 2
    for glyph in font.glyphs():
        layer = glyph.foreground
        layer = layer.stroke(
            "elliptical",
            amount,
            cross_amount,
            angle,
            "butt",
            "round",
        )
        glyph.foreground = layer


def main(argv: list[str]) -> None:
    if len(argv) != 7:
        raise SystemExit(
            "Usage: fontforge_augment_worker.py SOURCE FACE_INDEX OUTPUT OPERATION VALUE VARIANT_ID"
        )
    source, face_index_raw, output, operation, value_raw, variant_id = argv[1:]
    font = open_face(source, int(face_index_raw))
    try:
        set_variant_names(font, variant_id)
        value = float(value_raw)
        if operation == "width":
            transform_width(font, value)
        elif operation == "slant":
            transform_slant(font, value)
        elif operation == "weight":
            transform_weight(font, value)
        elif operation == "vertical_stroke":
            transform_directional_stroke(font, value, vertical=True)
        elif operation == "horizontal_stroke":
            transform_directional_stroke(font, value, vertical=False)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        font.generate(output)
    finally:
        font.close()


if __name__ == "__main__":
    main(sys.argv)
