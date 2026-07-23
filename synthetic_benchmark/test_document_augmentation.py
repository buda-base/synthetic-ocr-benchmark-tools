from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_augmentation import assign_document_augmentations


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


if __name__ == "__main__":
    unittest.main()
