"""Tests for style-aware geometry relaxation policy."""

import unittest

from coverage_common import (
    FontRow,
    apply_geometry_relaxation,
    detect_placement_warnings,
    relaxed_geometry_rules,
)


class RelaxedGeometryRulesTests(unittest.TestCase):
    def test_short_descender_relaxes_mid_stem_only(self):
        self.assertEqual(
            relaxed_geometry_rules("Short (1:1.2-1.5)", "Zabma (Printscripts)"),
            frozenset({"mid_stem"}),
        )

    def test_legless_relaxes_mid_stem(self):
        self.assertIn("mid_stem", relaxed_geometry_rules("Legless (1:1)", "Nama (Early Printscripts)"))

    def test_medium_descender_keeps_mid_stem(self):
        self.assertNotIn("mid_stem", relaxed_geometry_rules("Medium (1:2)", "Zabma (Printscripts)"))

    def test_tall_and_extra_keep_mid_stem(self):
        self.assertEqual(relaxed_geometry_rules("Tall (1:3)", "Parma (Printed Scripts)"), frozenset())
        self.assertEqual(relaxed_geometry_rules("Extra (1:5-1:5+)", "Parma (Printed Scripts)"), frozenset())

    def test_kyuma_relaxes_top_mark_misalignment(self):
        rules = relaxed_geometry_rules("Legless (1:1)", "Kyuma (Cursives)")
        self.assertEqual(rules, frozenset({"mid_stem", "top_mark_horizontal_misalignment"}))

    def test_tsugma_relaxes_top_mark_misalignment_but_keeps_mid_stem_when_descended(self):
        rules = relaxed_geometry_rules("Medium (1:2)", "Tsugma (Stylish Script)")
        self.assertEqual(rules, frozenset({"top_mark_horizontal_misalignment"}))

    def test_other_categories_keep_top_mark_rule(self):
        self.assertNotIn(
            "top_mark_horizontal_misalignment",
            relaxed_geometry_rules("Short (1:1.2-1.5)", "Druma (Oval Scripts)"),
        )

    def test_empty_metadata_relaxes_nothing(self):
        self.assertEqual(relaxed_geometry_rules("", ""), frozenset())


class ApplyRelaxationTests(unittest.TestCase):
    def _font(self, basename):
        return FontRow(
            basename=basename,
            font_path=None,
            font_path_csv="",
            ttc_face_index=0,
            ttc_face_index_csv="",
            ps_name="",
            other_names="",
            font_size_pt=24.0,
            dpi=300,
            skt_ok=1,
        )

    def test_apply_uses_map_and_defaults_empty(self):
        fonts = [self._font("A"), self._font("B")]
        enriched = apply_geometry_relaxation(
            fonts, {"A": frozenset({"mid_stem"})}
        )
        by_name = {f.basename: f.relaxed_rules for f in enriched}
        self.assertEqual(by_name["A"], frozenset({"mid_stem"}))
        self.assertEqual(by_name["B"], frozenset())


class RelaxedWarningSkipTests(unittest.TestCase):
    # Base descends to -600 while the bottom vowel sits mid-stem (ybot -200),
    # i.e. the vowel floats above the base foot -> mid_stem.
    BOXES = [
        {"x0": 0, "ytop": 1000, "x1": 800, "ybot": -600},
        {"x0": 150, "ytop": 300, "x1": 650, "ybot": -200},
    ]

    def test_mid_stem_fires_without_relaxation(self):
        warnings = detect_placement_warnings("ཀུ", self.BOXES)
        self.assertIn("mid_stem", warnings)

    def test_mid_stem_suppressed_when_relaxed(self):
        warnings = detect_placement_warnings(
            "ཀུ", self.BOXES, frozenset({"mid_stem"})
        )
        self.assertNotIn("mid_stem", warnings)


if __name__ == "__main__":
    unittest.main()
