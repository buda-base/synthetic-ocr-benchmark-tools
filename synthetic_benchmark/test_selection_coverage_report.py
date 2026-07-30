from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from selection_coverage_report import write_selection_coverage_report
from synthetic_common import FontCatalogRow, load_stack_rarity_scores


class SelectionCoverageReportTests(unittest.TestCase):
    def test_annotation_marks_do_not_create_independent_rarity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stacks_csv = Path(temporary) / "stacks.csv"
            stacks_csv.write_text(
                "stack,nb_occurrences\n"
                "ཀ,1000\n"
                "ཀ༵,1\n"
                "ཀ༷,2\n"
                "ཁ,100\n"
                "ག,10\n"
                "ང,1\n",
                encoding="utf-8",
            )
            rarity = load_stack_rarity_scores(str(stacks_csv))
        self.assertEqual(rarity["ཀ"], rarity["ཀ༵"])
        self.assertEqual(rarity["ཀ"], rarity["ཀ༷"])

    def test_report_compares_corpus_renderable_and_selected_stacks(self) -> None:
        font = FontCatalogRow(
            basename="Example",
            font_file="Example.ttf",
            font_path="fonts/Example.ttf",
            font_abs_path=Path("/fonts/Example.ttf"),
            ps_name="Example",
            ttc_face_index="",
            font_size_pt=20.0,
            dpi=300,
            skt_ok="1",
            script_id="1",
            script_category="Zabma",
            script_type="Uchen",
            script_name="Example",
        )
        chunks = [
            {"text": "ཀ་ ཁ་", "_stack_set": {"ཀ", "ཁ"}},
            {"text": "ག་", "_stack_set": {"ག"}},
        ]
        rows = [
            {
                "font_abs_path": "/fonts/Example.ttf",
                "ttc_face_index": "",
                "text": "ཀ་ ཀ་ ཁ་",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "plan.parquet"
            coverage_path, buckets_path, excluded_path, summary_path = (
                write_selection_coverage_report(
                    rows=rows,
                    chunks=chunks,
                    fonts=[font],
                    supported={"Example": {"ཀ", "ཁ"}},
                    output_plan=plan,
                )
            )
            with coverage_path.open(encoding="utf-8", newline="") as source:
                coverage = next(csv.DictReader(source))
            self.assertEqual(coverage["corpus_distinct_stacks"], "3")
            self.assertEqual(coverage["renderable_corpus_distinct_stacks"], "2")
            self.assertEqual(coverage["selected_distinct_stacks"], "2")
            self.assertEqual(coverage["selected_stack_token_occurrences"], "3")
            self.assertEqual(coverage["renderable_coverage_fraction"], "1.0")

            with buckets_path.open(encoding="utf-8", newline="") as source:
                buckets = list(csv.DictReader(source))
            self.assertEqual(
                sum(int(row["selected_distinct_stacks"]) for row in buckets),
                2,
            )
            self.assertEqual(
                sum(int(row["selected_token_occurrences"]) for row in buckets),
                3,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["selectable_corpus_distinct_letter_stacks"], 3)
            self.assertEqual(summary["excluded_corpus_distinct_letter_stacks"], 1)
            self.assertEqual(
                summary["excluded_unrenderable_distinct_letter_stacks"], 1
            )
            self.assertEqual(summary["font_faces"], 1)
            with excluded_path.open(encoding="utf-8", newline="") as source:
                excluded = list(csv.DictReader(source))
            self.assertEqual(
                [(row["stack"], row["exclusion_reason"]) for row in excluded],
                [("ག", "unsupported_by_all_eligible_fonts")],
            )


if __name__ == "__main__":
    unittest.main()
