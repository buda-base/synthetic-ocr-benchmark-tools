from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import random
import sys

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from font_augmentation import AugmentationSpec, variant_cache_key
from font_augmentation_raster_qc import compare_rasters
from font_augmentation_runtime import (
    RuntimeVariant,
    assign_variants_to_rows,
    sample_augmentation_specs,
)
from render_batches import assign_batches, fontspec_options


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


class RuntimeAugmentationTests(unittest.TestCase):
    def test_sampled_policy_stays_within_reviewed_ranges(self) -> None:
        rng = random.Random(13)
        for _ in range(500):
            specs = sample_augmentation_specs(rng)
            values = {spec.operation: spec.value for spec in specs}
            self.assertGreaterEqual(values["width"], 0.80)
            self.assertLessEqual(values["width"], 1.20)
            self.assertGreaterEqual(values["slant"], -14.0)
            self.assertLessEqual(values["slant"], 0.0)
            operations = [spec.operation for spec in specs]
            self.assertLess(operations.index("width"), operations.index("slant"))
            for operation in ("vertical_stroke", "horizontal_stroke"):
                if operation in values:
                    self.assertGreaterEqual(values[operation], -0.015)
                    self.assertLessEqual(values[operation], 0.030)
                    self.assertLess(operations.index(operation), operations.index("width"))

    def test_variant_assignment_is_stable_by_image_id(self) -> None:
        variants = [
            RuntimeVariant(
                variant_id=f"v{index}",
                path=Path(f"/tmp/v{index}.ttf"),
                specs=(AugmentationSpec("width", 0.9 + index * 0.1),),
                raster_qc={},
                provenance=(),
            )
            for index in range(2)
        ]
        rows = [
            {"image_id": image_id, "font_abs_path": "/source.ttf", "ttc_face_index": ""}
            for image_id in (4, 2, 3)
        ]
        assigned = assign_variants_to_rows(rows, variants)
        by_id = {row["image_id"]: row["font_augmentation_id"] for row in assigned}
        self.assertEqual(by_id, {2: "v1", 3: "v0", 4: "v1"})

    def test_lualatex_batches_and_fontspec_use_runtime_variant(self) -> None:
        rows = []
        for image_id, variant_id in ((1, "abc"), (2, "def"), (3, "abc")):
            rows.append(
                {
                    "image_id": image_id,
                    "basename": "Example",
                    "ttc_face_index": "2",
                    "font_augmentation_id": variant_id,
                    "render_font_abs_path": f"/tmp/{variant_id}.ttf",
                    "render_ttc_face_index": 0,
                }
            )
        batches = assign_batches(rows, 100)
        self.assertEqual(len(batches), 2)
        options = fontspec_options(rows[0])
        self.assertIn("UprightFont={abc.ttf}", options)
        self.assertIn("FontIndex=0", options)


if __name__ == "__main__":
    unittest.main()
