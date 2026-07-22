#!/usr/bin/env python3
"""Create deterministic, cached font variants for Tibetan rendering experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from fontTools.ttLib import TTFont


SCRIPT_DIR = Path(__file__).resolve().parent
FONTFORGE_WORKER = SCRIPT_DIR / "fontforge_augment_worker.py"
SKIA_WORKER = SCRIPT_DIR / "skia_directional_stroke_worker.py"
FONTFORGE_BACKEND_VERSION = "fontforge-20230101-v1"
SKIA_BACKEND_VERSION = "pathops-directional-stroke-v7"


def backend_version(spec: AugmentationSpec) -> str:
    if spec.operation in {"vertical_stroke", "horizontal_stroke"}:
        return SKIA_BACKEND_VERSION
    return FONTFORGE_BACKEND_VERSION


@dataclass(frozen=True)
class AugmentationSpec:
    operation: str
    value: float

    def validate(self) -> None:
        limits = {
            "width": (0.75, 1.25),
            "slant": (-15.0, 15.0),
            "weight": (-0.06, 0.06),
            "vertical_stroke": (-0.04, 0.04),
            "horizontal_stroke": (-0.04, 0.04),
        }
        if self.operation not in limits:
            raise ValueError(f"Unsupported font augmentation operation: {self.operation}")
        low, high = limits[self.operation]
        if not low <= self.value <= high:
            raise ValueError(
                f"{self.operation} value {self.value} is outside the safety limit [{low}, {high}]"
            )
        if self.operation == "width" and self.value == 0:
            raise ValueError("Width scale cannot be zero")

    @property
    def label(self) -> str:
        if self.operation == "width":
            return f"width_{self.value:.3f}"
        if self.operation == "slant":
            return f"slant_{self.value:+.1f}deg"
        if self.operation in {"vertical_stroke", "horizontal_stroke"}:
            return f"{self.operation}_{self.value:+.3f}em"
        return f"weight_{self.value:+.3f}em"


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_cache_key(source: Path, face_index: int, spec: AugmentationSpec) -> str:
    spec.validate()
    payload = {
        "backend": backend_version(spec),
        "source_sha256": source_sha256(source),
        "face_index": int(face_index),
        **asdict(spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def validate_font_file(path: Path) -> dict[str, object]:
    """Open a generated font and report the tables needed for Tibetan shaping."""
    with TTFont(path, lazy=True) as font:
        glyph_count = len(font.getGlyphOrder())
        tables = set(font.keys())
        result = {
            "glyph_count": glyph_count,
            "has_cmap": "cmap" in tables,
            "has_gsub": "GSUB" in tables,
            "has_gpos": "GPOS" in tables,
        }
    if not result["glyph_count"] or not result["has_cmap"]:
        raise ValueError(f"Generated font is missing glyphs or cmap: {path}")
    return result


def create_augmented_font(
    source: Path,
    *,
    face_index: int,
    spec: AugmentationSpec,
    cache_dir: Path,
    fontforge_bin: str = "fontforge",
    timeout_seconds: float = 30.0,
) -> tuple[Path, dict[str, object]]:
    """Return a cached augmented TTF and its provenance manifest."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    spec.validate()
    key = variant_cache_key(source, face_index, spec)
    output = cache_dir / f"{source.stem}_{key}.ttf"
    manifest_path = output.with_suffix(".json")
    if output.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cache_hit"] = True
        validate_font_file(output)
        return output, manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    directional = spec.operation in {"vertical_stroke", "horizontal_stroke"}
    if directional:
        command = [
            sys.executable,
            str(SKIA_WORKER),
            str(source),
            str(int(face_index)),
            str(output),
            spec.operation,
            repr(float(spec.value)),
        ]
    else:
        command = [
            fontforge_bin,
            "-lang=py",
            "-script",
            str(FONTFORGE_WORKER),
            str(source),
            str(int(face_index)),
            str(output),
            spec.operation,
            repr(float(spec.value)),
            key,
        ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(timeout_seconds, 90.0) if directional else timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Font augmentation timed out after "
            f"{(max(timeout_seconds, 90.0) if directional else timeout_seconds):g}s for "
            f"{source.name} {spec.label}"
        ) from exc
    if proc.returncode != 0 or not output.is_file():
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Font augmentation failed ({proc.returncode}) for {source.name} "
            f"{spec.label}:\n{(proc.stdout + proc.stderr)[-2000:]}"
        )

    font_info = validate_font_file(output)
    manifest = {
        "variant_id": key,
        "backend": backend_version(spec),
        "source_path": str(source),
        "source_sha256": source_sha256(source),
        "source_face_index": int(face_index),
        "operation": spec.operation,
        "value": spec.value,
        "label": spec.label,
        "output_path": str(output),
        "cache_hit": False,
        "font_info": font_info,
        "backend_output": (proc.stdout + proc.stderr)[-2000:].strip(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, manifest
