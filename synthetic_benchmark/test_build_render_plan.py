from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_render_plan import (  # noqa: E402
    build_difficulty_pools,
    chunk_difficulty_tier,
    filter_chunks_by_text_policy,
    make_per_font_plan_rows,
    parse_category_quotas,
    text_repetition_score,
    top_up_category_quotas,
    zstd_compression_ratio,
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
                "_rare_stack_set": (
                    frozenset({f"stack-{index}"}) if difficult else frozenset()
                ),
                "_rare_stack_weight": (
                    {f"stack-{index}": 0.9} if difficult else {}
                ),
                "stack_difficulty_score": 0.5 if difficult else 0.0,
                "text_repetition_score": (index % 10) / 10,
                "zstd_compression_ratio": 0.15 + index / 100,
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
    def test_category_top_up_reaches_equal_quotas_and_marks_copies(self) -> None:
        rows: list[dict[str, object]] = []
        splits = {
            "Category A": ["train", "val", "test"],
            "Category B": ["train"] * 8 + ["val", "test"],
        }
        image_id = 1
        for category, category_splits in splits.items():
            for index, split in enumerate(category_splits):
                rows.append(
                    {
                        "image_id": image_id,
                        "script_8": category,
                        "script": "ume",
                        "suggested_split": split,
                        "font_abs_path": f"/fonts/{category}-{index % 2}.ttf",
                        "ttc_face_index": "",
                        "stack_difficulty_score": 0.0,
                        "font_name": f"{category}-{index % 2}",
                    }
                )
                image_id += 1

        prepared = top_up_category_quotas(rows, target_images=20)

        self.assertEqual(len(prepared), 20)
        self.assertEqual(
            Counter(row["script_8"] for row in prepared),
            Counter({"Category A": 10, "Category B": 10}),
        )
        self.assertEqual(
            sum(int(row["source_text_category_topup"]) for row in prepared),
            7,
        )
        self.assertEqual(
            [int(row["image_id"]) for row in prepared],
            list(range(1, 21)),
        )

    def test_category_top_up_does_not_invent_missing_split(self) -> None:
        rows = [
            {
                "image_id": index + 1,
                "script_8": "Category A" if index < 2 else "Category B",
                "script": "ume",
                "suggested_split": (
                    ("train", "test")[index]
                    if index < 2
                    else "train"
                ),
                "font_abs_path": f"/fonts/{index}.ttf",
                "ttc_face_index": "",
                "stack_difficulty_score": 0.0,
                "font_name": f"Font {index}",
            }
            for index in range(3)
        ]

        prepared = top_up_category_quotas(rows, target_images=10)

        category_a = [row for row in prepared if row["script_8"] == "Category A"]
        self.assertEqual(len(category_a), 5)
        self.assertNotIn("val", {row["suggested_split"] for row in category_a})

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

    def test_rewarded_repetition_can_be_disabled(self) -> None:
        chunks = sample_chunks()
        font = sample_font()
        rows = make_per_font_plan_rows(
            fonts=[font],
            chunks=chunks,
            supported={font.basename: {f"stack-{index}" for index in range(20)}},
            images_per_font=10,
            seed=13,
            max_chunk_reuse_ratio=0.10,
            reward_repetition=False,
        )
        self.assertEqual(
            Counter(row["source_text_repetition_policy"] for row in rows),
            Counter({"penalized": 10}),
        )

    def test_text_policy_applies_hard_repetition_and_density_guards(self) -> None:
        chunks = sample_chunks()
        selected = filter_chunks_by_text_policy(
            chunks,
            max_repetition_score=0.5,
            min_zstd_compression_ratio=0.18,
        )
        self.assertTrue(selected)
        self.assertTrue(
            all(float(chunk["text_repetition_score"]) < 0.5 for chunk in selected)
        )
        self.assertTrue(
            all(float(chunk["zstd_compression_ratio"]) >= 0.18 for chunk in selected)
        )

    def test_zstd_compression_ratio_is_deterministic_and_detects_redundancy(self) -> None:
        repetitive = "བོད་ཡིག་" * 200
        varied = "".join(chr(0x0F40 + index % 40) for index in range(1600))
        self.assertEqual(
            zstd_compression_ratio(repetitive),
            zstd_compression_ratio(repetitive),
        )
        self.assertLess(
            zstd_compression_ratio(repetitive),
            zstd_compression_ratio(varied),
        )

    def test_explicit_category_quotas_must_be_complete_and_exact(self) -> None:
        quotas = parse_category_quotas(
            ["Category A=6", "Category B=4"],
            categories=["Category A", "Category B"],
            target_images=10,
        )
        self.assertEqual(quotas, {"Category A": 6, "Category B": 4})
        with self.assertRaises(ValueError):
            parse_category_quotas(
                ["Category A=10"],
                categories=["Category A", "Category B"],
                target_images=10,
            )

    def test_repetition_score_distinguishes_repeated_sequences(self) -> None:
        diverse = "བོད་ཡིག་སྐད་རིགས་གཞུང་ལུགས་མཛོད།"
        repeated = "བོད་ཡིག་བོད་ཡིག་བོད་ཡིག་བོད་ཡིག།"
        self.assertGreater(
            text_repetition_score(repeated),
            text_repetition_score(diverse),
        )

    def test_difficult_selection_expands_rare_stack_coverage(self) -> None:
        chunks = sample_chunks()
        difficult = [chunk for chunk in chunks if chunk["stack_difficulty_score"]]
        difficult[0]["_rare_stack_set"] = frozenset({"rare-a"})
        difficult[0]["_rare_stack_weight"] = {"rare-a": 0.9}
        difficult[1]["_rare_stack_set"] = frozenset({"rare-a"})
        difficult[1]["_rare_stack_weight"] = {"rare-a": 0.9}
        difficult[2]["_rare_stack_set"] = frozenset({"rare-b"})
        difficult[2]["_rare_stack_weight"] = {"rare-b": 0.95}
        font = sample_font()
        supported = {font.basename: {f"stack-{index}" for index in range(20)}}
        rows = make_per_font_plan_rows(
            fonts=[font],
            chunks=chunks,
            supported=supported,
            images_per_font=4,
            seed=13,
            max_chunk_reuse_ratio=0.10,
        )
        difficult_rows = [
            row for row in rows if row["source_text_difficulty_tier"] == "difficult"
        ]
        self.assertEqual(len(difficult_rows), 2)
        self.assertTrue(
            all(row["source_text_new_rare_stack_count"] >= 1 for row in difficult_rows)
        )
        self.assertEqual(
            sum(row["source_text_new_rare_stack_count"] for row in difficult_rows),
            2,
        )
        self.assertTrue(
            all(
                row["source_text_coverage_policy"]
                == "bounded_greedy_rare_stack_coverage"
                for row in difficult_rows
            )
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
