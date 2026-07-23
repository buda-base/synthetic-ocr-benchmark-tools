from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_augmentation import (
    _apply_vertical_tps,
    apply_document_augmentations,
    assign_document_augmentations,
    enforce_background_readability_for_line_count,
)
from paper_backgrounds import PaperBackground


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


def sample_backgrounds() -> list[PaperBackground]:
    return [
        PaperBackground(
            background_id=f"background-{mode}",
            w_id="W1",
            i_id="I1",
            i_version="v1",
            filename=f"paper-{mode}.tif",
            source_s3_uri=f"s3://example/paper-{mode}.tif",
            width=400,
            height=100,
            display_width=400,
            display_height=100,
            aspect_ratio=4.0,
            exif_orientation=None,
            pil_mode=mode,
            size_bytes=100,
        )
        for mode in ("1", "L", "RGB")
    ]


class DocumentAugmentationPolicyTests(unittest.TestCase):
    def test_local_rate_is_balanced_per_font(self) -> None:
        prepared, manifest = assign_document_augmentations(
            sample_rows(),
            paper_backgrounds=sample_backgrounds(),
            local_rate=0.90,
            seed=13,
        )
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
        first, _manifest = assign_document_augmentations(
            sample_rows(images_per_font=100),
            paper_backgrounds=sample_backgrounds(),
            seed=42,
        )
        second, _manifest = assign_document_augmentations(
            sample_rows(images_per_font=100),
            paper_backgrounds=sample_backgrounds(),
            seed=42,
        )
        fields = (
            "document_augmentation_paper_source",
            "document_augmentation_background_id",
            "document_augmentation_paper_effect",
            "document_augmentation_output_mode",
            "document_augmentation_output_extension",
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
            paper_backgrounds=sample_backgrounds(),
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

    def test_paper_sources_are_exclusive_and_balanced_per_font(self) -> None:
        prepared, manifest = assign_document_augmentations(
            sample_rows(images_per_font=100),
            paper_backgrounds=sample_backgrounds(),
            background_rate=0.60,
            synthetic_paper_rate=0.30,
            seed=17,
        )
        for font_index in range(3):
            font_rows = [row for row in prepared if row["basename"] == f"Font{font_index}"]
            counts = Counter(
                str(row["document_augmentation_paper_source"]) for row in font_rows
            )
            self.assertEqual(counts, Counter({"real": 60, "synthetic": 30, "clean": 10}))
            for row in font_rows:
                source = row["document_augmentation_paper_source"]
                if source == "real":
                    self.assertTrue(row["document_augmentation_background_id"])
                    self.assertFalse(row["document_augmentation_paper_effect"])
                elif source == "synthetic":
                    self.assertFalse(row["document_augmentation_background_id"])
                    self.assertTrue(row["document_augmentation_paper_effect"])
                else:
                    self.assertFalse(row["document_augmentation_background_id"])
                    self.assertFalse(row["document_augmentation_paper_effect"])
        self.assertEqual(manifest["requested_rates"]["clean_paper"], 0.10)

    def test_fragile_text_only_uses_light_backgrounds(self) -> None:
        backgrounds = [
            PaperBackground(
                **{
                    **background.__dict__,
                    "background_id": f"{background.background_id}-{tier}",
                    "luminance_tier": tier,
                    "mean_luminance": 230.0 if tier == "light" else 110.0,
                }
            )
            for background in sample_backgrounds()
            for tier in ("light", "dark")
        ]
        rows = sample_rows(fonts=1, images_per_font=20)
        for row in rows:
            row["font_size_pt"] = 24
            row["output_width_px"] = 1800
        prepared, _manifest = assign_document_augmentations(
            rows,
            paper_backgrounds=backgrounds,
            background_rate=1.0,
            synthetic_paper_rate=0.0,
            local_rate=0.0,
        )
        self.assertEqual(
            {row["document_augmentation_background_luminance_tier"] for row in prepared},
            {"light"},
        )

    def test_dense_page_replaces_dark_background_with_light_fallback(self) -> None:
        row = {
            "document_augmentation_paper_source": "real",
            "document_augmentation_background_luminance_tier": "dark",
            "document_augmentation_background_readability_fallback": 0,
            "_document_augmentation_fallback_background_id": "light-id",
            "_document_augmentation_fallback_background_uri": "s3://example/light.jpg",
            "_document_augmentation_fallback_background_mode": "RGB",
            "_document_augmentation_fallback_background_luminance_tier": "light",
            "_document_augmentation_fallback_background_mean_luminance": 230.0,
            "_document_augmentation_fallback_background_path": "/tmp/light.jpg",
        }
        self.assertTrue(enforce_background_readability_for_line_count(row, 8))
        self.assertEqual(row["document_augmentation_background_id"], "light-id")
        self.assertEqual(row["document_augmentation_background_luminance_tier"], "light")
        self.assertEqual(row["document_augmentation_background_readability_fallback"], 1)

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
                paper_backgrounds=sample_backgrounds(),
                rotation_rate=0.10,
                rotation_high_rate=0.20,
            )
        with self.assertRaises(ValueError):
            assign_document_augmentations(
                sample_rows(),
                paper_backgrounds=sample_backgrounds(),
                tps_rate=0.10,
                tps_high_rate=0.20,
            )

    def test_real_background_mode_and_encoding_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source_image = np.full((100, 400), 255, dtype=np.uint8)
            source_image[40:60, 100:300] = 0
            Image.fromarray(source_image).save(source)

            cases = (
                ("L", 210, ".jpg", "JPEG"),
                ("RGB", (210, 220, 230), ".jpg", "JPEG"),
                ("1", 255, ".tif", "TIFF"),
            )
            for mode, color, extension, image_format in cases:
                background = root / f"background-{mode}.tif"
                Image.new(mode, (800, 200), color=color).save(background)
                destination = root / f"output-{mode}{extension}"
                row = {
                    "document_augmentation_seed": 13,
                    "document_augmentation_local": "",
                    "document_augmentation_local_strength": "",
                    "document_augmentation_spatial": "",
                    "document_augmentation_spatial_strength": "",
                    "document_augmentation_paper_source": "real",
                    "document_augmentation_background_mode": mode,
                    "_document_augmentation_background_path": str(background),
                    "document_augmentation_output_mode": mode,
                    "document_augmentation_tps_strength": "",
                    "document_augmentation_rotation_deg": "",
                    "document_augmentation_blur": "",
                }
                apply_document_augmentations(
                    source,
                    destination,
                    row,
                    jpeg_quality=95,
                )
                with Image.open(destination) as output:
                    self.assertEqual(output.mode, mode)
                    self.assertEqual(output.format, image_format)
                    if mode == "1":
                        self.assertEqual(output.tag_v2.get(259), 4)
                    self.assertEqual(output.getpixel((200, 50)), 0 if mode != "RGB" else (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
