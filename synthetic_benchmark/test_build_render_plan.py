from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_render_plan import (  # noqa: E402
    build_difficulty_pools,
    chunk_difficulty_tier,
    make_per_font_plan_rows,
    text_repetition_score,
)
from synthetic_common import FontCatalogRow  # noqa: E402


def sample_chunks() -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for index in range(20):
        difficult = index >= 10
        chunks.append(
            {
                "chunk_id": f"chunk-{index:02d}",
                "bocorpus_row": index,
                "char_start": 0,
                "char_end": 10,
                "text": f"text-{index}",
                "char_count": 10,
                "stack_count": 2,
                "unique_stack_count": 1,
                "stacks": f"stack-{index}",
                "_stack_set": frozenset({f"stack-{index}"}),
                "stack_difficulty_score": 0.5 if difficult else 0.0,
                "text_repetition_score": (index % 10) / 10,
            }
        )
    return chunks


def sample_font() -> FontCatalogRow:
    return FontCatalogRow(
        basename="TestFont",
        font_file="TestFont.ttf",
        font_path="fonts/TestFont.ttf",
        font_abs_path=Path("/tmp/TestFont.ttf"),
        ps_name="TestFont",
        ttc_face_index="",
        font_size_pt=24.0,
        dpi=300,
        skt_ok="y",
        script_id="1",
        script_category="uchen",
        script_type="uchen",
        script_name="Test",
    )


class RenderPlanDifficultyTests(unittest.TestCase):
    def test_difficulty_pools_use_rare_stack_score_boundary(self) -> None:
        chunks = sample_chunks()
        pools = build_difficulty_pools(chunks)
        self.assertEqual(len(pools["normal"]), 10)
        self.assertEqual(len(pools["difficult"]), 10)
        self.assertEqual(chunk_difficulty_tier(pools["normal"][0]), "normal")
        self.assertEqual(chunk_difficulty_tier(pools["difficult"][0]), "difficult")

    def test_per_font_plan_is_half_normal_half_difficult(self) -> None:
        chunks = sample_chunks()
        font = sample_font()
        supported = {font.basename: {f"stack-{index}" for index in range(20)}}
        first = make_per_font_plan_rows(
            fonts=[font],
            chunks=chunks,
            supported=supported,
            images_per_font=10,
            seed=13,
            max_chunk_reuse_ratio=0.10,
        )
        second = make_per_font_plan_rows(
            fonts=[font],
            chunks=chunks,
            supported=supported,
            images_per_font=10,
            seed=13,
            max_chunk_reuse_ratio=0.10,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["source_text_difficulty_tier"] for row in first),
            Counter({"normal": 5, "difficult": 5}),
        )
        self.assertTrue(
            all(
                row["source_text_difficulty_tier"]
                == row["requested_text_difficulty_tier"]
                for row in first
            )
        )
        self.assertEqual(
            Counter(row["source_text_repetition_policy"] for row in first),
            Counter({"penalized": 9, "rewarded": 1}),
        )
        rewarded_score = next(
            row["source_text_repetition_score"]
            for row in first
            if row["source_text_repetition_policy"] == "rewarded"
        )
        penalized_scores = [
            row["source_text_repetition_score"]
            for row in first
            if row["source_text_repetition_policy"] == "penalized"
        ]
        self.assertGreater(rewarded_score, max(penalized_scores))

    def test_repetition_score_distinguishes_repeated_sequences(self) -> None:
        diverse = "བོད་ཡིག་སྐད་རིགས་གཞུང་ལུགས་མཛོད།"
        repeated = "བོད་ཡིག་བོད་ཡིག་བོད་ཡིག་བོད་ཡིག།"
        self.assertGreater(
            text_repetition_score(repeated),
            text_repetition_score(diverse),
        )

    def test_fonts_without_rare_compatible_stacks_use_relative_difficulty(self) -> None:
        chunks = sample_chunks()
        font = sample_font()
        supported = {font.basename: {f"stack-{index}" for index in range(10)}}
        rows = make_per_font_plan_rows(
            fonts=[font],
            chunks=chunks,
            supported=supported,
            images_per_font=10,
            seed=13,
            max_chunk_reuse_ratio=0.10,
        )
        self.assertEqual(
            Counter(row["source_text_difficulty_tier"] for row in rows),
            Counter({"normal": 5, "difficult": 5}),
        )
        difficult_rows = [
            row for row in rows if row["source_text_difficulty_tier"] == "difficult"
        ]
        self.assertTrue(
            all(row["source_text_rarity_tier"] == "normal" for row in difficult_rows)
        )
        self.assertTrue(
            all(
                row["source_text_difficulty_basis"]
                == "relative_compatible_complexity"
                for row in difficult_rows
            )
        )


if __name__ == "__main__":
    unittest.main()
