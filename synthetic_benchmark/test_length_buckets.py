#!/usr/bin/env python3
from __future__ import annotations

import unittest

from length_buckets import (
    LONG_MARGIN_X_MM,
    LONG_MARGIN_Y_MM,
    LONG_OUTPUT_RESOLUTION_SCALE,
    truncate_to_short_prefix,
    merge_texts_to_long,
)
from shorthand_aug import syllable_count


class LengthBucketTests(unittest.TestCase):
    def test_short_and_long(self):
        text = (
            chr(0x0F40)
            + chr(0x0FB2)
            + chr(0x0F0B)
            + chr(0x0F64)
            + chr(0x0F72)
            + chr(0x0F66)
            + chr(0x0F0D)
            + " "
            + chr(0x0F51)
            + chr(0x0F0B)
            + chr(0x0F63)
            + chr(0x0F9F)
            + chr(0x0F0D)
        )
        short = truncate_to_short_prefix(text, max_syllables=10, min_syllables=1)
        self.assertIsNotNone(short)
        self.assertTrue(short.endswith(chr(0x0F0D)))
        self.assertGreaterEqual(syllable_count(short), 1)
        self.assertLessEqual(syllable_count(short), 10)
        unit = (
            chr(0x0F56)
            + chr(0x0F7C)
            + chr(0x0F51)
            + chr(0x0F0B)
            + chr(0x0F61)
            + chr(0x0F72)
            + chr(0x0F42)
            + chr(0x0F0D)
        )
        merged = merge_texts_to_long(unit, [unit * 20] * 80, min_syllables=1000)
        self.assertIsNotNone(merged)
        self.assertGreaterEqual(syllable_count(merged), 1000)

    def test_short_rejects_mark_only_prefix(self):
        # Yig-mgo + shad only: no Tibetan letters / syllables.
        self.assertIsNone(truncate_to_short_prefix("༄༅།", min_syllables=1))

    def test_long_resolution_scale_default(self):
        self.assertGreaterEqual(LONG_OUTPUT_RESOLUTION_SCALE, 1.5)

    def test_long_margins_are_tighter_than_defaults(self):
        self.assertLess(LONG_MARGIN_X_MM, 20.0)
        self.assertLess(LONG_MARGIN_Y_MM, 16.0)


if __name__ == "__main__":
    unittest.main()
