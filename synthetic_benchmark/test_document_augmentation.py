from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_augmentation import _apply_vertical_tps, assign_document_augmentations


def sample_rows(fonts: int = 3, images_per_font: int = 20) -> list[dict[str, object]]:
    rows = []
    image_id = 1
    for font_index in range(fonts):
        for _ in range(images_per_font):
            rows.append(
                {
                    "image_id": image_id,
                    "basename": f"Font{font_index}",
                    "ttc_face_index": "",
                }
            )
            image_id += 1
    return rows


class DocumentAugmentationPolicyTests(unittest.TestCase):
    def test_local_rate_is_balanced_per_font(self) -> None:
        prepared, manifest = assign_document_augmentations(sample_rows(), local_rate=0.90, seed=13)
        counts: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        for row in prepared:
            basename = str(row["basename"])
            totals[basename] += 1
            if row["document_augmentation_local"]:
                counts[basename] += 1
        self.assertEqual(totals, Counter({"Font0": 20, "Font1": 20, "Font2": 20}))
        self.assertEqual(counts, Counter({"Font0": 18, "Font1": 18, "Font2": 18}))
        self.assertTrue(all(font["local_rate"] == 0.9 for font in manifest["fonts"]))

    def test_policy_is_deterministic_and_excludes_removed_effects(self) -> None:
        first, _manifest = assign_document_augmentations(sample_rows(images_per_font=100), seed=42)
        second, _manifest = assign_document_augmentations(sample_rows(images_per_font=100), seed=42)
        fields = (
            "document_augmentation_local",
            "document_augmentation_local_strength",
            "document_augmentation_spatial",
            "document_augmentation_spatial_strength",
            "document_augmentation_blur",
            "document_augmentation_rotation_strength",
            "document_augmentation_rotation_deg",
            "document_augmentation_tps_strength",
            "document_augmentation_tps_y_norm",
            "document_augmentation_tps_offsets_height",
            "document_augmentation_seed",
        )
        self.assertEqual(
            [[row[field] for field in fields] for row in first],
            [[row[field] for field in fields] for row in second],
        )
        local_effects = {str(row["document_augmentation_local"]) for row in first}
        self.assertNotIn("lowlightnoise", local_effects)
        self.assertNotIn("linesdegradation", local_effects)
        self.assertTrue(
            all(row["document_augmentation_blur"] in {"", "mild", "medium"} for row in first)
        )

    def test_geometric_rates_are_balanced_per_font(self) -> None:
        prepared, manifest = assign_document_augmentations(
            sample_rows(images_per_font=100),
            rotation_rate=0.70,
            rotation_high_rate=0.10,
            tps_rate=0.30,
            tps_high_rate=0.10,
            seed=31,
        )
        for font_index in range(3):
            font_rows = [row for row in prepared if row["basename"] == f"Font{font_index}"]
            rotated = [row for row in font_rows if row["document_augmentation_rotation_strength"]]
            high_rotation = [
                row
                for row in rotated
                if row["document_augmentation_rotation_strength"] == "high"
            ]
            tps = [row for row in font_rows if row["document_augmentation_tps_strength"]]
            high_tps = [row for row in tps if row["document_augmentation_tps_strength"] == "high"]
            self.assertEqual(len(rotated), 70)
            self.assertEqual(len(high_rotation), 10)
            self.assertEqual(len(tps), 30)
            self.assertEqual(len(high_tps), 10)
            self.assertTrue(all(row in rotated for row in tps))
            self.assertTrue(
                all(
                    0.90 <= abs(float(row["document_augmentation_rotation_deg"])) <= 2.40
                    for row in high_rotation
                )
            )
            for row in tps:
                offsets = [
                    float(value)
                    for value in str(row["document_augmentation_tps_offsets_height"]).split("|")
                ]
                self.assertEqual(len(offsets), 5)
                maximum = max(abs(value) for value in offsets)
                if row["document_augmentation_tps_strength"] == "high":
                    self.assertGreaterEqual(maximum, 0.0304 - 1e-7)
                    self.assertLessEqual(maximum, 0.0526 + 1e-7)
                else:
                    self.assertGreaterEqual(maximum, 0.0117 - 1e-7)
                    self.assertLessEqual(maximum, 0.0304 + 1e-7)
        self.assertEqual(manifest["requested_rates"]["rotation"], 0.70)
        self.assertEqual(manifest["requested_rates"]["rotation_high"], 0.10)
        self.assertEqual(manifest["requested_rates"]["tps"], 0.30)
        self.assertEqual(manifest["requested_rates"]["tps_high"], 0.10)

    def test_vertical_tps_preserves_dimensions_and_changes_pixels(self) -> None:
        image = np.full((120, 300, 3), 255, dtype=np.uint8)
        image[58:63, 20:280] = 0
        warped = _apply_vertical_tps(
            image,
            y_normalized=0.5,
            offsets_height=[-0.02, 0.01, 0.025, 0.005, -0.02],
        )
        self.assertEqual(warped.shape, image.shape)
        self.assertFalse(np.array_equal(warped, image))

    def test_high_geometric_rates_cannot_exceed_total_rates(self) -> None:
        with self.assertRaises(ValueError):
            assign_document_augmentations(
                sample_rows(),
                rotation_rate=0.10,
                rotation_high_rate=0.20,
            )
        with self.assertRaises(ValueError):
            assign_document_augmentations(
                sample_rows(),
                tps_rate=0.10,
                tps_high_rate=0.20,
            )


if __name__ == "__main__":
    unittest.main()
