from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from font_variant_cache import S3FontVariantCache


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.get_calls = 0

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kwargs) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        self.get_calls += 1
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            ) from exc
        return {"Body": io.BytesIO(body)}


class FontVariantRejectionCacheTests(unittest.TestCase):
    def test_rejected_variant_round_trips_and_is_policy_scoped(self) -> None:
        client = FakeS3Client()
        specs = [{"operation": "width", "value": 1.1}]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "font_variant_cache._s3_client",
            return_value=client,
        ):
            cache = S3FontVariantCache(
                "s3://example/font-cache/",
                Path(temporary),
            )
            uri = cache.store_rejection(
                basename="Example",
                source_hash="a" * 64,
                face_index=0,
                variant_id="variant-1",
                specs=specs,
                policy_key="policy-1",
                raster_qc_version=2,
                reason="ValueError: raster QC failed",
            )
            rejection = cache.fetch_rejection(
                basename="Example",
                source_hash="a" * 64,
                face_index=0,
                variant_id="variant-1",
                specs=specs,
                policy_key="policy-1",
                raster_qc_version=2,
            )
            missing = cache.fetch_rejection(
                basename="Example",
                source_hash="a" * 64,
                face_index=0,
                variant_id="variant-1",
                specs=specs,
                policy_key="new-policy",
                raster_qc_version=2,
            )

        self.assertIsNotNone(rejection)
        self.assertEqual(rejection["reason"], "ValueError: raster QC failed")
        self.assertEqual(rejection["s3_uri"], uri)
        self.assertIsNone(missing)

    def test_policy_index_serves_rejections_with_one_s3_read(self) -> None:
        client = FakeS3Client()
        source_hash = "b" * 64
        specs = [{"operation": "slant", "value": -8.0}]
        rejection = {
            "source_sha256": source_hash,
            "face_index": 0,
            "variant_id": "variant-2",
            "specs": specs,
            "policy_key": "policy-2",
            "raster_qc_version": 2,
            "decision": "reject_variant",
            "reason": "ValueError: unsafe",
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "font_variant_cache._s3_client",
            return_value=client,
        ):
            writer = S3FontVariantCache(
                "s3://example/font-cache/",
                Path(temporary) / "writer",
            )
            writer.store_policy_index(
                source_hash=source_hash,
                face_index=0,
                policy_key="policy-2",
                raster_qc_version=2,
                accepted=[],
                rejected=[rejection],
            )
            reader = S3FontVariantCache(
                "s3://example/font-cache/",
                Path(temporary) / "reader",
            )
            first = reader.fetch_rejection(
                basename="Example",
                source_hash=source_hash,
                face_index=0,
                variant_id="variant-2",
                specs=specs,
                policy_key="policy-2",
                raster_qc_version=2,
            )
            second = reader.fetch_rejection(
                basename="Example",
                source_hash=source_hash,
                face_index=0,
                variant_id="variant-2",
                specs=specs,
                policy_key="policy-2",
                raster_qc_version=2,
            )

        self.assertEqual(first["reason"], "ValueError: unsafe")
        self.assertEqual(second["reason"], "ValueError: unsafe")
        self.assertEqual(client.get_calls, 1)


if __name__ == "__main__":
    unittest.main()
