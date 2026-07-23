from __future__ import annotations

import tempfile
import sys
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_output_policy import assign_image_output_parameters
from page_layout_policy import assign_page_layout_parameters
from paper_backgrounds import crop_resize_background


def sample_rows(images_per_font: int = 100) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    image_id = 1
    for font_index in range(2):
        for index in range(images_per_font):
            rows.append(
                {
                    "image_id": image_id,
                    "basename": f"Font{font_index}",
                    "ttc_face_index": "",
                    "document_augmentation_output_extension": (
                        ".tif" if index < 10 else ".jpg"
                    ),
                }
            )
            image_id += 1
    return rows


class ImageOutputPolicyTests(unittest.TestCase):
    def test_widths_and_low_jpeg_quality_are_deterministic(self) -> None:
        first, manifest = assign_image_output_parameters(sample_rows(), seed=23)
        second, _manifest = assign_image_output_parameters(sample_rows(), seed=23)
        self.assertEqual(first, second)
        self.assertEqual(manifest["width_px"], {"minimum": 1800, "maximum": 3500})
        for font_index in range(2):
            rows = [row for row in first if row["basename"] == f"Font{font_index}"]
            self.assertTrue(
                all(1800 <= int(row["output_width_px"]) <= 3500 for row in rows)
            )
            jpeg_qualities = Counter(
                int(row["output_jpeg_quality"])
                for row in rows
                if row["document_augmentation_output_extension"] == ".jpg"
            )
            self.assertEqual(jpeg_qualities, Counter({85: 81, 65: 9}))

    def test_fixed_width_override(self) -> None:
        prepared, _manifest = assign_image_output_parameters(
            sample_rows(images_per_font=10),
            min_width_px=2400,
            max_width_px=2400,
        )
        self.assertEqual({row["output_width_px"] for row in prepared}, {2400})

    def test_background_is_stretched_instead_of_cropped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "paper.png"
            source = Image.new("RGB", (200, 100), (255, 0, 0))
            for y in range(50, 100):
                for x in range(200):
                    source.putpixel((x, y), (0, 0, 255))
            source.save(source_path)
            output = crop_resize_background(
                source_path,
                target_size=(400, 100),
                mode="RGB",
            )
            self.assertEqual(output.size, (400, 100))
            self.assertEqual(output.getpixel((200, 10)), (255, 0, 0))
            self.assertEqual(output.getpixel((200, 90)), (0, 0, 255))

    def test_line_spacing_varies_deterministically_within_safe_range(self) -> None:
        rows = sample_rows(images_per_font=30)
        first, manifest = assign_page_layout_parameters(rows, seed=29)
        second, _manifest = assign_page_layout_parameters(rows, seed=29)
        first_values = [row["layout_line_spacing_factor"] for row in first]
        self.assertEqual(first, second)
        self.assertTrue(all(1.18 <= value <= 1.32 for value in first_values))
        self.assertGreater(len(set(first_values)), 50)
        self.assertEqual(
            Counter(row["layout_density_tier"] for row in first),
            Counter({"condensed": 54, "sparse": 6}),
        )
        self.assertTrue(
            all(
                1.20 <= row["layout_font_scale_multiplier"] <= 1.35
                for row in first
                if row["layout_density_tier"] == "sparse"
            )
        )
        self.assertEqual(
            manifest["collision_policy"]["method"],
            "stack-height-aware leading plus LuaTeX glyph-offset safety",
        )

    def test_tall_stacks_receive_extra_leading_only_on_their_pages(self) -> None:
        rows = sample_rows(images_per_font=1)
        rows[0]["stacks"] = "ཀ abcdefg"
        rows[1]["stacks"] = "ཀ ཁ"
        prepared, _manifest = assign_page_layout_parameters(rows, seed=29)
        tall = prepared[0]
        ordinary = prepared[1]
        self.assertEqual(tall["layout_max_stack_codepoints"], 7)
        self.assertEqual(tall["layout_stack_spacing_extra"], 0.9)
        self.assertGreater(tall["layout_line_spacing_factor"], 2.0)
        self.assertEqual(ordinary["layout_stack_spacing_extra"], 0.0)


if __name__ == "__main__":
    unittest.main()
