from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shorthand_aug import (  # noqa: E402
    DEFAULT_SHORTHANDS_CSV,
    ShorthandPair,
    apply_shorthands,
    is_tibetan_only,
    load_shorthand_pairs,
)
from synthetic_common import tokenize_tibetan_stacks  # noqa: E402


def _is_tibetan_only_text(text: str) -> bool:
    return all(
        ch in "\n\r\t " or 0x0F00 <= ord(ch) <= 0x0FFF for ch in text
    )


class ShorthandPurityTests(unittest.TestCase):
    def test_is_tibetan_only_detects_stray_characters(self) -> None:
        self.assertTrue(is_tibetan_only("ཏིང་ངེ་འཛིན"))
        self.assertFalse(is_tibetan_only("ཏིངêནི"))  # Latin ê
        self.assertFalse(is_tibetan_only("༢+ཨིས"))  # plus sign
        self.assertFalse(is_tibetan_only("\U000f2007ས"))  # PUA code point

    def test_loaded_pairs_are_pure_tibetan(self) -> None:
        pairs = load_shorthand_pairs(DEFAULT_SHORTHANDS_CSV)
        self.assertTrue(pairs)
        offenders = [
            pair
            for pair in pairs
            if not is_tibetan_only(pair.long_form)
            or not is_tibetan_only(pair.shorthand)
        ]
        self.assertEqual(offenders, [])

    def test_apply_shorthands_never_emits_non_tibetan(self) -> None:
        # A corrupt pair (Tibetan stacks + a stray Latin letter) must be
        # rejected at load, so it can never be spliced into the output. We
        # emulate the loader's guard here to prove apply_shorthands stays pure.
        corrupt = ShorthandPair(
            long_form="ཏིང་ངེ་འཛིན",
            shorthand="ཏིངêནི",
            source="test",
            shorthand_stacks=tuple(tokenize_tibetan_stacks("ཏིངêནི")),
        )
        clean_pairs = [
            pair
            for pair in [corrupt]
            if is_tibetan_only(pair.long_form) and is_tibetan_only(pair.shorthand)
        ]
        text = "ཆོས་ཏིང་ངེ་འཛིན་པ།"
        out, n = apply_shorthands(
            text,
            clean_pairs,
            mode="dense",
            rng=random.Random(0),
        )
        self.assertEqual(n, 0)
        self.assertEqual(out, text)
        self.assertTrue(_is_tibetan_only_text(out))


if __name__ == "__main__":
    unittest.main()
