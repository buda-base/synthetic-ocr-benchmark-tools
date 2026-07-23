from __future__ import annotations

import tempfile
import sys
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_output_policy import assign_image_output_parameters
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


if __name__ == "__main__":
    unittest.main()
