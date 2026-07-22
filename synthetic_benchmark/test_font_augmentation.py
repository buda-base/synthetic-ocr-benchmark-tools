from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from font_augmentation import AugmentationSpec, variant_cache_key
from font_augmentation_raster_qc import compare_rasters


def make_minimal_font(path: Path) -> None:
    builder = FontBuilder(1000, isTTF=True)
    glyph_order = [".notdef", "space"]
    builder.setupGlyphOrder(glyph_order)
    builder.setupCharacterMap({0x20: "space"})
    glyphs = {}
    for name in glyph_order:
        pen = TTGlyphPen(None)
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (500, 0) for name in glyph_order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable(
        {
            "familyName": "Augmentation Test",
            "styleName": "Regular",
            "uniqueFontIdentifier": "Augmentation Test Regular",
            "fullName": "Augmentation Test Regular",
            "psName": "AugmentationTest-Regular",
        }
    )
    builder.setupOS2()
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)


class AugmentationSpecTests(unittest.TestCase):
    def test_valid_specs(self) -> None:
        for spec in (
            AugmentationSpec("width", 0.92),
            AugmentationSpec("slant", -6),
            AugmentationSpec("weight", 0.02),
            AugmentationSpec("vertical_stroke", -0.015),
            AugmentationSpec("horizontal_stroke", 0.015),
        ):
            spec.validate()

    def test_rejects_unsafe_values(self) -> None:
        with self.assertRaises(ValueError):
            AugmentationSpec("width", 0.5).validate()
        with self.assertRaises(ValueError):
            AugmentationSpec("slant", 30).validate()
        with self.assertRaises(ValueError):
            AugmentationSpec("unknown", 1).validate()

    def test_cache_key_is_deterministic_and_parameterized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.ttf"
            make_minimal_font(source)
            first = variant_cache_key(source, 0, AugmentationSpec("width", 0.92))
            second = variant_cache_key(source, 0, AugmentationSpec("width", 0.92))
            other = variant_cache_key(source, 0, AugmentationSpec("width", 1.08))
            self.assertEqual(first, second)
            self.assertNotEqual(first, other)


class RasterQCTests(unittest.TestCase):
    def test_detects_filled_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.png"
            variant_path = Path(directory) / "variant.png"
            baseline = Image.new("L", (160, 80), "white")
            draw = ImageDraw.Draw(baseline)
            for x in range(10, 140, 25):
                draw.rectangle((x, 15, x + 18, 60), fill="black")
                draw.rectangle((x + 5, 25, x + 13, 50), fill="white")
            baseline.save(baseline_path)
            variant = baseline.copy()
            draw = ImageDraw.Draw(variant)
            for x in range(10, 140, 25):
                draw.rectangle((x + 5, 25, x + 13, 50), fill="black")
            variant.save(variant_path)
            result = compare_rasters(baseline_path, variant_path)
            self.assertIn("counter_loss_or_black_fill", result["warnings"])


if __name__ == "__main__":
    unittest.main()
