#!/usr/bin/env python3
"""Immutable local/S3 cache for validated synthetic font variants."""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


DEFAULT_FONT_CACHE_URI = "s3://bec.bdrc.io/synthetic/font_cache/"
THREAD_LOCAL = threading.local()


def unique_variant_name(
    basename: str,
    source_hash: str,
    face_index: int,
    variant_id: str,
) -> str:
    safe_basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._") or "font"
    return (
        f"{safe_basename}--{source_hash[:16]}--f{face_index:03d}--"
        f"{variant_id}.ttf"
    )


def _s3_client():
    client = getattr(THREAD_LOCAL, "s3_client", None)
    if client is None:
        client = boto3.client("s3")
        THREAD_LOCAL.s3_client = client
    return client


@dataclass(frozen=True)
class CachedVariant:
    path: Path
    raster_qc: dict[str, object]
    provenance: tuple[dict[str, object], ...]


class S3FontVariantCache:
    def __init__(self, uri: str, local_dir: Path):
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid font cache S3 URI: {uri}")
        self.uri = uri.rstrip("/") + "/"
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.local_dir = local_dir
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._policy_indexes: dict[
            tuple[str, int, str, int], dict[str, object] | None
        ] = {}
        self._index_lock = threading.Lock()

    def _key(self, source_hash: str, face_index: int, filename: str) -> str:
        relative = f"{source_hash}/f{face_index:03d}/{filename}"
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def fetch_prepared_pool(self, pool_key: str) -> dict[str, object] | None:
        """Fetch one complete prepared-pool manifest, preferring local disk."""
        local_path = self.local_dir / "_prepared_pools" / f"{pool_key}.json"
        if local_path.exists():
            return json.loads(local_path.read_text(encoding="utf-8"))
        key = (
            f"{self.prefix}/_prepared_pools/{pool_key}.json"
            if self.prefix
            else f"_prepared_pools/{pool_key}.json"
        )
        try:
            payload = _s3_client().get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return None
            raise
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(payload)
        return json.loads(payload)

    def store_prepared_pool(
        self,
        pool_key: str,
        manifest: dict[str, object],
    ) -> str:
        """Persist a complete pool manifest locally and in one S3 object."""
        payload = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        local_path = self.local_dir / "_prepared_pools" / f"{pool_key}.json"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_suffix(".json.part")
        temporary.write_bytes(payload)
        temporary.replace(local_path)
        key = (
            f"{self.prefix}/_prepared_pools/{pool_key}.json"
            if self.prefix
            else f"_prepared_pools/{pool_key}.json"
        )
        _s3_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"

    def materialize_prepared_variant(
        self,
        *,
        font_filename: str,
        s3_uri: str,
    ) -> Path:
        """Return a persistent local font, downloading only on a cold worker."""
        destination = self.local_dir / font_filename
        if destination.exists() and destination.stat().st_size:
            return destination
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError(f"Invalid prepared variant URI: {s3_uri}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        _s3_client().download_file(self.bucket, parsed.path.lstrip("/"), str(temporary))
        temporary.replace(destination)
        return destination

    def load_policy_index(
        self,
        *,
        source_hash: str,
        face_index: int,
        policy_key: str,
        raster_qc_version: int,
    ) -> dict[str, object] | None:
        """Load one aggregate cache index instead of probing every variant."""
        cache_key = (source_hash, face_index, policy_key, raster_qc_version)
        with self._index_lock:
            if cache_key in self._policy_indexes:
                return self._policy_indexes[cache_key]
        key = self._key(
            source_hash,
            face_index,
            f"policy-index--{policy_key}--qc{raster_qc_version}.json",
        )
        try:
            index = json.loads(
                _s3_client()
                .get_object(Bucket=self.bucket, Key=key)["Body"]
                .read()
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                index = None
            else:
                raise
        if index is not None:
            expected = {
                "source_sha256": source_hash,
                "face_index": face_index,
                "policy_key": policy_key,
                "raster_qc_version": raster_qc_version,
            }
            if any(index.get(field) != value for field, value in expected.items()):
                raise ValueError(f"Font policy index mismatch: s3://{self.bucket}/{key}")
            index["s3_uri"] = f"s3://{self.bucket}/{key}"
        with self._index_lock:
            self._policy_indexes[cache_key] = index
        return index

    def store_policy_index(
        self,
        *,
        source_hash: str,
        face_index: int,
        policy_key: str,
        raster_qc_version: int,
        accepted: list[dict[str, object]],
        rejected: list[dict[str, object]],
    ) -> str:
        """Publish an aggregate immutable index for a completed font policy."""
        key = self._key(
            source_hash,
            face_index,
            f"policy-index--{policy_key}--qc{raster_qc_version}.json",
        )
        index = {
            "cache_version": 1,
            "source_sha256": source_hash,
            "face_index": face_index,
            "policy_key": policy_key,
            "raster_qc_version": raster_qc_version,
            "accepted": {
                str(record["variant_id"]): record for record in accepted
            },
            "rejected": {
                str(record["variant_id"]): record for record in rejected
            },
        }
        _s3_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=(
                json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8"),
            ContentType="application/json",
        )
        cache_key = (source_hash, face_index, policy_key, raster_qc_version)
        index["s3_uri"] = f"s3://{self.bucket}/{key}"
        with self._index_lock:
            self._policy_indexes[cache_key] = index
        return f"s3://{self.bucket}/{key}"

    def fetch_unaugmentable(
        self,
        *,
        source_hash: str,
        face_index: int,
        policy_key: str,
    ) -> dict[str, object] | None:
        key = self._key(
            source_hash,
            face_index,
            f"unaugmentable--{policy_key}.json",
        )
        try:
            decision = json.loads(
                _s3_client()
                .get_object(Bucket=self.bucket, Key=key)["Body"]
                .read()
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return None
            raise
        expected = {
            "source_sha256": source_hash,
            "face_index": face_index,
            "policy_key": policy_key,
            "decision": "use_original_font",
        }
        if any(decision.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Font cache decision mismatch: s3://{self.bucket}/{key}")
        decision["s3_uri"] = f"s3://{self.bucket}/{key}"
        return decision

    def store_unaugmentable(
        self,
        *,
        source_hash: str,
        face_index: int,
        policy_key: str,
        policy: dict[str, object],
        reason: str,
        rejected_attempts: list[dict[str, object]],
    ) -> str:
        key = self._key(
            source_hash,
            face_index,
            f"unaugmentable--{policy_key}.json",
        )
        decision = {
            "cache_version": 1,
            "source_sha256": source_hash,
            "face_index": face_index,
            "policy_key": policy_key,
            "policy": policy,
            "decision": "use_original_font",
            "reason": reason,
            "rejected_attempts": rejected_attempts,
        }
        _s3_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=(
                json.dumps(decision, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            ).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"

    def fetch_rejection(
        self,
        *,
        basename: str,
        source_hash: str,
        face_index: int,
        variant_id: str,
        specs: list[dict[str, object]],
        policy_key: str,
        raster_qc_version: int,
    ) -> dict[str, object] | None:
        index = self.load_policy_index(
            source_hash=source_hash,
            face_index=face_index,
            policy_key=policy_key,
            raster_qc_version=raster_qc_version,
        )
        if index is not None:
            rejection = dict(
                (index.get("rejected") or {}).get(variant_id) or {}
            )
            if not rejection:
                return None
            if rejection.get("specs") != specs:
                raise ValueError(
                    f"Font rejection policy index mismatch for {variant_id}"
                )
            rejection["s3_uri"] = str(index["s3_uri"])
            return rejection
        filename = unique_variant_name(
            basename,
            source_hash,
            face_index,
            variant_id,
        )
        key = self._key(
            source_hash,
            face_index,
            f"{filename}.rejected--{policy_key}.json",
        )
        try:
            rejection = json.loads(
                _s3_client()
                .get_object(Bucket=self.bucket, Key=key)["Body"]
                .read()
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return None
            raise
        expected = {
            "source_sha256": source_hash,
            "face_index": face_index,
            "variant_id": variant_id,
            "specs": specs,
            "policy_key": policy_key,
            "raster_qc_version": raster_qc_version,
            "decision": "reject_variant",
        }
        if any(rejection.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Font rejection cache mismatch: s3://{self.bucket}/{key}")
        rejection["s3_uri"] = f"s3://{self.bucket}/{key}"
        return rejection

    def store_rejection(
        self,
        *,
        basename: str,
        source_hash: str,
        face_index: int,
        variant_id: str,
        specs: list[dict[str, object]],
        policy_key: str,
        raster_qc_version: int,
        reason: str,
    ) -> str:
        filename = unique_variant_name(
            basename,
            source_hash,
            face_index,
            variant_id,
        )
        key = self._key(
            source_hash,
            face_index,
            f"{filename}.rejected--{policy_key}.json",
        )
        rejection = {
            "cache_version": 1,
            "source_sha256": source_hash,
            "face_index": face_index,
            "variant_id": variant_id,
            "specs": specs,
            "policy_key": policy_key,
            "raster_qc_version": raster_qc_version,
            "decision": "reject_variant",
            "reason": reason,
        }
        _s3_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=(
                json.dumps(rejection, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            ).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"

    def fetch(
        self,
        *,
        basename: str,
        source_hash: str,
        face_index: int,
        variant_id: str,
        specs: list[dict[str, object]],
        policy_key: str,
        raster_qc_version: int,
    ) -> CachedVariant | None:
        filename = unique_variant_name(
            basename,
            source_hash,
            face_index,
            variant_id,
        )
        metadata_key = self._key(source_hash, face_index, filename + ".json")
        font_key = self._key(source_hash, face_index, filename)
        index = self.load_policy_index(
            source_hash=source_hash,
            face_index=face_index,
            policy_key=policy_key,
            raster_qc_version=raster_qc_version,
        )
        if index is not None:
            metadata = dict((index.get("accepted") or {}).get(variant_id) or {})
            if not metadata:
                return None
        else:
            try:
                metadata = json.loads(
                    _s3_client()
                    .get_object(Bucket=self.bucket, Key=metadata_key)["Body"]
                    .read()
                )
            except _s3_client().exceptions.NoSuchKey:
                return None
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    return None
                raise
        expected = {
            "source_sha256": source_hash,
            "face_index": face_index,
            "variant_id": variant_id,
            "specs": specs,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Font cache metadata mismatch: s3://{self.bucket}/{metadata_key}"
            )
        if (
            metadata.get("raster_qc", {}).get("qc_version")
            != raster_qc_version
        ):
            return None
        destination = self.local_dir / filename
        if not destination.exists() or destination.stat().st_size == 0:
            temporary = destination.with_suffix(destination.suffix + ".part")
            _s3_client().download_file(self.bucket, font_key, str(temporary))
            temporary.replace(destination)
        return CachedVariant(
            path=destination,
            raster_qc=dict(metadata.get("raster_qc") or {}),
            provenance=(
                {
                    "operation": "s3_cache",
                    "source": f"s3://{self.bucket}/{font_key}",
                    "destination": str(destination),
                },
            ),
        )

    def store(
        self,
        *,
        basename: str,
        source_hash: str,
        face_index: int,
        variant_id: str,
        specs: list[dict[str, object]],
        font_path: Path,
        raster_qc: dict[str, object],
        provenance: tuple[dict[str, object], ...],
    ) -> str:
        filename = unique_variant_name(
            basename,
            source_hash,
            face_index,
            variant_id,
        )
        font_key = self._key(source_hash, face_index, filename)
        metadata_key = self._key(source_hash, face_index, filename + ".json")
        metadata = {
            "cache_version": 1,
            "source_sha256": source_hash,
            "face_index": face_index,
            "variant_id": variant_id,
            "specs": specs,
            "raster_qc": raster_qc,
            "provenance": list(provenance),
            "font_filename": filename,
        }
        _s3_client().upload_file(
            str(font_path),
            self.bucket,
            font_key,
            ExtraArgs={"ContentType": "font/ttf"},
        )
        _s3_client().put_object(
            Bucket=self.bucket,
            Key=metadata_key,
            Body=(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            ).encode("utf-8"),
            ContentType="application/json",
        )
        destination = self.local_dir / filename
        if not destination.exists() or destination.stat().st_size == 0:
            temporary = destination.with_suffix(destination.suffix + ".part")
            shutil.copy2(font_path, temporary)
            temporary.replace(destination)
        return f"s3://{self.bucket}/{font_key}"
